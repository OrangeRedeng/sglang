"""Actual output-tree and packed-cache regressions from the A3 final round."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[5]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_graph_weak_refs_preserve_dynamic_integers_nested_outputs_and_none():
    from sglang.srt.utils import common

    native = ModuleType("torch_npu._C")
    native._weak_ref_tensor = lambda x: x + 10
    with (
        patch.object(common, "is_cuda", return_value=False),
        patch.object(common, "is_hip", return_value=False),
        patch.object(common, "is_musa", return_value=False),
        patch.object(common, "is_xpu", return_value=False),
        patch.object(common, "is_npu", return_value=True),
        patch.dict(sys.modules, {"torch_npu._C": native}),
    ):
        module = load(
            "python/sglang/srt/model_executor/runner_backend_utils/weak_ref_tensor.py",
            "weak_mixed",
        )
    for rows in (1, 2, 40):
        output = (rows, [torch.ones(rows), None], {"enabled": True, "scale": 0.5})
        result = module.weak_ref_tensors(output)
        assert result[0] == rows
        assert result[1][1] is None and result[2] == output[2]
        torch.testing.assert_close(result[1][0], torch.full((rows,), 11.0))


def test_packed_cache_read_uses_prepared_ring_without_dtype_or_storage_allocations():
    path = ROOT / "python/sglang/srt/hardware_backend/npu/memory_pool_npu.py"
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "NPUMLATokenToKVPool"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_prepare_dcp_packed_read_buffers", "_gather_packed_sfa_rows")
    ]
    namespace = dict(torch=torch, envs=envs)
    for method in methods:
        exec("from __future__ import annotations\n" + ast.unparse(method), namespace)
    rows, latent, rope_width = 8, 512, 64
    values = (
        torch.arange(rows * latent)
        .reshape(rows, latent)
        .remainder(255)
        .sub(127)
        .to(torch.int8)
    )
    rope = torch.arange(rows * rope_width).reshape(rows, rope_width).to(torch.bfloat16)
    scale = torch.full((rows, 4), 0.03)
    raw = torch.cat(
        (values.view(torch.uint8), rope.view(torch.uint8), scale.view(torch.uint8)), -1
    ).view(torch.int8)
    pool = SimpleNamespace(
        sfa_packed_kv_storage=True,
        dcp_size=16,
        k_buffer=torch.stack((raw, raw)),
        kv_cache_dim=656,
        store_dtype=torch.int8,
        dtype=torch.bfloat16,
        device="cpu",
        kv_lora_rank=latent,
        qk_rope_head_dim=rope_width,
        start_layer=0,
        layer_transfer_counter=None,
    )
    with (
        envs.SGLANG_NPU_DSA_CP_TOKEN_SHARDED.override(True),
        patch(
            "sglang.srt.runtime_context.get_schedule",
            return_value=SimpleNamespace(prefill_max_requests=2),
        ),
    ):
        namespace["_prepare_dcp_packed_read_buffers"](pool)
    pointers = {}
    for layer, ids in ((0, [4, 2, 7]), (1, [6, 1]), (0, [0, 3, 5]), (1, [])):
        loc = torch.tensor(ids, dtype=torch.int32)
        expected = (
            (values[ids].float().reshape(-1, 4, 128) * scale[ids].unsqueeze(-1))
            .reshape(len(ids), 512)
            .bfloat16()
        )
        with (
            patch.object(torch, "empty", side_effect=AssertionError("hot allocation")),
            patch.object(
                torch.Tensor, "to", side_effect=AssertionError("hot conversion")
            ),
        ):
            nope, actual_rope = namespace["_gather_packed_sfa_rows"](
                pool, layer, loc, None
            )
        torch.testing.assert_close(nope[:, 0], expected)
        torch.testing.assert_close(actual_rope[:, 0], rope[ids])
        assert nope.is_contiguous() and actual_rope.is_contiguous()
        if ids:
            assert pointers.setdefault(layer, nope.data_ptr()) == nope.data_ptr()
    assert pointers[0] != pointers[1]


def test_stream_bound_and_final_plan_preserve_untested_coverage_and_budget():
    from sglang.srt.hardware_backend.npu.moe.tp_fusion import (
        shared_stream_rows_eligible,
    )

    with (
        envs.SGLANG_NPU_TP_MOE_SHARED_STREAM_MAX_TOKENS.override(16384),
        envs.SGLANG_NPU_TP_MOE_EAGER_MULTI_STREAM_MIN_TOKENS.override(1),
    ):
        assert shared_stream_rows_eligible(10832)
        assert not shared_stream_rows_eligible(32768)
        assert not shared_stream_rows_eligible(0)
    night = load("benchmark/prefill_1m/run_a3_nightly.py", "final_night")
    plan = night.final_plan()
    assert len(plan) * 40 <= 8 * 60
    assert plan[1]["variant"] == "graph-piecewise-dcp16"
    assert plan[3]["requires"] == plan[2]["name"]
    assert {"best-int8-batch2", "best-int8-packed-batch2"} <= {
        c["variant"] for c in plan
    }
    assert not {
        "tp-routed-scale",
        "tp-prequant",
        "tp-shared",
        "tp-finalize",
        "megamoe-gate-int8",
    } & {c["variant"] for c in plan}
    runner = load("benchmark/prefill_1m/run_bde_overnight.py", "final_runner")
    cfg = dict(length=1000000, repeat_rate=0.99, requests=16, output_len=1)
    resolved = dict(page_size=128, dcp_size=16, max_running_requests=2)
    assert runner.required_kv_capacity(cfg, resolved) == 1017856


def test_ragged_logit_row_selection_uses_host_lengths_without_tensor_readback():
    from sglang.srt.layers.attention.dsa import dsa_cp

    x = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    plan = SimpleNamespace(local_start=0, local_end=7)
    group = SimpleNamespace(
        all_gather_into_tensor=lambda output, value: output.copy_(value)
    )
    parallel = SimpleNamespace(attn_tp_size=1, attn_tp_group=group)
    batch = SimpleNamespace(extend_seq_lens_cpu=[3, 4])
    with (
        patch.object(dsa_cp, "get_parallel", return_value=parallel),
        patch.object(dsa_cp, "get_dsa_cp_plan", return_value=plan),
        patch.object(
            torch,
            "tensor",
            side_effect=AssertionError("device-dependent metadata tensor"),
        ),
        patch.object(
            torch.Tensor, "tolist", side_effect=AssertionError("metadata readback")
        ),
    ):
        actual = dsa_cp.dsa_cp_gather_last_token_rows(x, batch)
    torch.testing.assert_close(actual, x[[2, 6]])
