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
    assert {
        "gate-int8-norm",
        "moe-fused-combined",
        "best-int8-fused",
        "best-int8-metadata",
    } <= {c["variant"] for c in plan}
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


def test_prefix_output_capacity_does_not_grow_with_cold_prefix_chunks():
    from sglang.srt.layers.dcp import layout

    with patch.dict(layout._dcp_extend_gather_buffers, {}, clear=True):
        ref = torch.empty(0, 1, 3)
        first = layout.dcp_extend_gather_buffer("prefix", ref, 8, capacity_rows=1024)
        first.fill_(7)
        with patch.object(
            torch, "empty", side_effect=AssertionError("late prefix growth")
        ):
            last = layout.dcp_extend_gather_buffer(
                "prefix", ref, 1000, capacity_rows=1024
            )
        assert first.data_ptr() == last.data_ptr()
        assert last.shape == (1000, 1, 3)
        torch.testing.assert_close(last[:8], first)


def test_moe_collectives_break_compilation_and_preserve_quantized_values():
    from sglang.srt.hardware_backend.npu.modules import dsa_token_residual as residual
    from sglang.srt.layers.attention.dsa import dsa_cp

    calls = []

    def gather(out, value):
        assert not torch.compiler.is_compiling()
        calls.append("gather")
        out.copy_(torch.cat((value, value)))

    def scatter(out, value):
        assert not torch.compiler.is_compiling()
        calls.append("scatter")
        out.copy_(value[: out.shape[0]] * 2)

    group = SimpleNamespace(
        world_size=2, all_gather_into_tensor=gather, reduce_scatter_tensor=scatter
    )
    parallel = SimpleNamespace(attn_tp_group=group, attn_tp_size=2)
    native = ModuleType("torch_npu")
    native.npu_dynamic_quant = lambda value, dst_type: (
        value.to(dst_type),
        torch.full((value.shape[0],), 0.5),
    )

    def forward(value):
        gathered, operand = residual.gather_moe_input(value + 1)
        assert operand is None
        return dsa_cp.dsa_cp_reduce_scatter_tokens(gathered * 3)

    value = torch.arange(8).float().reshape(2, 4)
    with (
        patch.dict(sys.modules, {"torch_npu": native}),
        patch.object(residual, "get_parallel", return_value=parallel),
        patch.object(dsa_cp, "get_parallel", return_value=parallel),
    ):
        result = torch.compile(forward, backend="eager")(value)
    torch.testing.assert_close(result, (value + 1) * 3)
    assert calls == ["gather", "gather", "scatter"]


def test_native_graph_errors_and_batch_statistics_survive_worker_exit():
    runner = load("benchmark/prefill_1m/run_bde_overnight.py", "stream_failure_runner")
    assert "unjoined" in runner.server_failure(
        "capture model contains a stream that was not joined to the original stream"
    )
    assert "HcclAllGather" in runner.server_failure(
        "capture failed: RuntimeError: HcclAllGather"
    )
    metrics = runner.batch_metrics(
        "Prefill batch, #new-seq: 1, #cached-token: 0\n"
        "Prefill batch, #new-seq: 2, #cached-token: 990000\n"
        "Decode batch, #running-req: 2\n"
    )
    assert metrics["mean_prefill_batch"] == 1.5
    assert metrics["mean_cached_prefill_batch"] == 2
    assert metrics["mean_decode_batch"] == 2
    assert runner.batch_metrics("")["mean_decode_batch"] is None


def test_dcp_capture_finishes_eager_work_without_synchronizing_replay():
    from contextlib import contextmanager
    from sglang.srt import runtime_context

    path = ROOT / "python/sglang/srt/compilation/npu_piecewise_backend.py"
    cls = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__"
    )
    calls = []
    entry = SimpleNamespace(
        runnable=lambda *_: torch.ones(2), cudagraph=None, num_finished_warmup=1
    )
    instance = SimpleNamespace(
        first_run_finished=True,
        sym_shape_indices=[0],
        concrete_size_entries={16: entry},
        compile_config=SimpleNamespace(get_enable_debug_mode=lambda: False),
        graph_pool=None,
        is_first_graph=True,
        is_last_graph=False,
    )

    @contextmanager
    def graph(*args, **kwargs):
        assert calls[-1] == "synchronize"
        calls.append("capture")
        yield

    npu = SimpleNamespace(
        synchronize=lambda: calls.append("synchronize"),
        NPUGraph=lambda: SimpleNamespace(replay=lambda: calls.append("replay")),
        graph=graph,
    )
    namespace = dict(
        torch=torch,
        ExitStack=__import__("contextlib").ExitStack,
        is_in_torch_compile_warmup=lambda: False,
        get_pcg_capture_stream=lambda: object(),
        weak_ref_tensors=lambda x: x,
        compilation_counter=SimpleNamespace(num_cudagraph_captured=0),
    )
    exec(ast.unparse(method), namespace)
    with (
        patch.object(torch, "npu", npu, create=True),
        patch.object(
            runtime_context,
            "get_parallel",
            return_value=SimpleNamespace(dcp_enabled=True),
        ),
    ):
        first = namespace["__call__"](instance, 16)
        replay = namespace["__call__"](instance, 16)
    torch.testing.assert_close(first, replay)
    assert calls == ["synchronize", "capture", "replay"]


def test_batch2_launches_have_identical_capacity_and_serial_gather_settings():
    harness = load(
        "benchmark/prefill_1m/test_glm53_features.py", "batch2_capacity_harness"
    )
    first = harness.scenario_for("best-int8-batch2", "historical-bde-capacity")
    packed = harness.scenario_for("best-int8-packed-batch2", "historical-bde-capacity")
    assert first["launch_overrides"] == packed["launch_overrides"]
    assert first["launch_overrides"]["kv_rows"] * 16 >= 1017856
    for variant in (first, packed):
        assert variant["env"]["SGLANG_NPU_ENABLE_DCP_GATHER_PREFETCH"] == "False"
        assert variant["env"]["SGLANG_NPU_ENABLE_DCP_SFA_GATHER_OVERLAP"] == "False"
    assert packed["env"]["SGLANG_NPU_ENABLE_DCP_PACKED_READ_BATCH2"] == "True"


def test_piecewise_moe_inputs_keep_ring_addresses_after_eager_gathers():
    from sglang.srt.hardware_backend.npu import dcp_piecewise
    from sglang.srt.hardware_backend.npu.modules import dsa_token_residual as residual
    from sglang.srt.hardware_backend.npu.moe.contracts import INT8, QuantizedOperand

    group = SimpleNamespace(
        world_size=2,
        all_gather_into_tensor=lambda out, value: out.copy_(torch.cat((value, value))),
    )
    native = ModuleType("torch_npu")
    buffers = {}
    owners = [
        SimpleNamespace(
            _npu_dcp_piecewise_buffers=buffers, _npu_dcp_piecewise_slot=i % 2
        )
        for i in range(3)
    ]
    pointers = {}
    with (
        patch.dict(sys.modules, {"torch_npu": native}),
        patch.object(
            residual, "get_parallel", return_value=SimpleNamespace(attn_tp_group=group)
        ),
        patch.object(dcp_piecewise, "active", return_value=True),
        patch.object(QuantizedOperand, "record_stream"),
    ):
        for i, owner in enumerate(owners):
            data = torch.full((2, 4), i + 1, dtype=torch.int8)
            scale = torch.full((2,), 0.5)
            operand = QuantizedOperand(data, scale, INT8, (2, 4), torch.bfloat16)
            output, gathered = residual.gather_moe_input(operand, operand, owner=owner)
            expected = torch.full((4, 4), (i + 1) * 0.5, dtype=torch.bfloat16)
            torch.testing.assert_close(output, expected)
            torch.testing.assert_close(gathered.data, torch.cat((data, data)))
            actual = (
                output.data_ptr(),
                gathered.data.data_ptr(),
                gathered.scale.data_ptr(),
            )
            assert pointers.setdefault(i % 2, actual) == actual
    assert pointers[0] != pointers[1]
