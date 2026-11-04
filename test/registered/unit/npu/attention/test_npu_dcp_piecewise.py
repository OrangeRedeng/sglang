"""DCP planning stays eager while piecewise feed-forward work remains compiled."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.arg_groups import cuda_graph_hook
from sglang.srt.hardware_backend.npu import dcp_piecewise
from sglang.srt.hardware_backend.npu.modules import dsa_token_residual as residual
from sglang.srt.layers.attention.dsa.dsa_cp_layout import plan_dsa_cp_shard
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def config(**changes):
    values = dict(
        device="npu",
        tp_size=16,
        dcp_size=16,
        pp_size=1,
        attn_cp_size=1,
        enable_dp_attention=False,
        enable_lora=False,
        speculative_algorithm=None,
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(backend=Backend.TC_PIECEWISE),
            decode=SimpleNamespace(backend=Backend.DISABLED),
        ),
    )
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "changes,allowed",
    [
        ({}, True),
        ({"dcp_size": 8}, False),
        ({"pp_size": 2}, False),
        ({"attn_cp_size": 2}, False),
        ({"enable_dp_attention": True}, False),
        ({"enable_lora": True}, False),
        ({"ep_size": 16}, False),
    ],
)
def test_dcp16_admission_preserves_other_parallel_guards(changes, allowed):
    with (
        patch.object(cuda_graph_hook, "resolving_view", side_effect=lambda x: x),
        patch.object(cuda_graph_hook, "resolved_view", side_effect=lambda x: x),
        patch.object(
            cuda_graph_hook, "attention_backends_of", return_value=("ascend", "ascend")
        ),
    ):
        if allowed:
            cuda_graph_hook.validate_cuda_graph_config(config(**changes))
        else:
            with pytest.raises(ValueError):
                cuda_graph_hook.validate_cuda_graph_config(config(**changes))


def test_planning_refreshes_prefix_and_request_boundaries():
    calls = []

    def prepare(*args):
        calls.append(args)
        return SimpleNamespace(prefix=list(args[2]))

    runner = SimpleNamespace(
        model=SimpleNamespace(prepare_context_parallel_metadata_for_dcp=prepare),
        req_to_token_pool=SimpleNamespace(req_to_token=torch.arange(64).reshape(2, 32)),
        token_to_kv_pool=SimpleNamespace(get_kv_buffer_shape=lambda: [(32, 1, 4)]),
        kv_cache_dtype=torch.float32,
        device="cpu",
    )
    with patch.object(
        dcp_piecewise, "get_parallel", return_value=SimpleNamespace(attn_dcp_size=16)
    ):
        for prefixes, lengths in (([0], [16]), ([12, 24], [3, 5])):
            batch = SimpleNamespace(
                seq_lens=torch.tensor([p + n for p, n in zip(prefixes, lengths)]),
                extend_prefix_lens=torch.tensor(prefixes),
                extend_prefix_lens_cpu=prefixes,
                extend_seq_lens=torch.tensor(lengths),
                req_pool_indices=torch.arange(len(lengths)),
                seq_lens_sum=sum(p + n for p, n in zip(prefixes, lengths)),
            )
            dcp_piecewise.prepare_batch(runner, batch)
            assert batch.attn_dcp_metadata.prefix == prefixes
    assert calls[0][2] != calls[1][2]
    assert calls[1][5] is runner.req_to_token_pool.req_to_token


def test_local_ffn_shapes_are_included_in_capture_configuration():
    from sglang.srt.model_executor.runner_backend import (
        tc_piecewise_cuda_graph_backend as backend,
    )

    prefill = SimpleNamespace(bs=[16, 128, 4096, 32768], tc_compiler="eager")
    execution = SimpleNamespace(
        graph=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(prefill=prefill),
            enable_torch_compile_debug_mode=False,
        )
    )
    with (
        patch.object(backend, "get_exec", return_value=execution),
        patch.object(
            backend,
            "get_parallel",
            return_value=SimpleNamespace(dcp_enabled=True, attn_tp_size=16),
        ),
        patch.object(
            backend,
            "get_moe_a2a_backend",
            return_value=SimpleNamespace(
                is_deepep=lambda: False, is_mooncake=lambda: False
            ),
        ),
    ):
        config = backend.TcPiecewiseCudaGraphBackend.build_compilation_config(
            SimpleNamespace(device="npu")
        )
    assert {1, 8, 256, 2048, 16, 128, 4096, 32768} <= set(config.get_capture_sizes())


def test_staging_keeps_replay_addresses_and_uses_two_layer_slots():
    buffers = {}
    owners = [
        SimpleNamespace(
            _npu_dcp_piecewise_buffers=buffers, _npu_dcp_piecewise_slot=i % 2
        )
        for i in range(3)
    ]
    first = dcp_piecewise.stage_tensor(owners[0], "ffn_input", torch.ones(4, 2))
    other = dcp_piecewise.stage_tensor(owners[1], "ffn_input", torch.ones(4, 2) * 2)
    replay = dcp_piecewise.stage_tensor(owners[2], "ffn_input", torch.ones(4, 2) * 3)
    assert first.data_ptr() == replay.data_ptr()
    assert other.data_ptr() != replay.data_ptr()
    torch.testing.assert_close(other, torch.ones(4, 2) * 2)
    torch.testing.assert_close(first, torch.ones(4, 2) * 3)
    assert len(buffers) == 2


def test_whole_graph_capture_guard_is_only_lifted_at_eager_boundary():
    batch = SimpleNamespace(
        attn_dcp_metadata=object(),
        forward_mode=SimpleNamespace(is_extend=lambda: True),
        can_run_tbo=False,
    )
    plan = object()
    with (
        residual.envs.SGLANG_NPU_DSA_CP_TOKEN_SHARDED.override(True),
        patch.object(residual, "get_parallel", return_value=SimpleNamespace(pp_size=1)),
        patch.object(residual, "get_dsa_cp_plan", return_value=plan),
        patch(
            "sglang.srt.model_executor.runner.get_is_capture_mode", return_value=True
        ),
    ):
        with pytest.raises(ValueError, match="eager-only"):
            residual.full_head_plan(batch)
        with dcp_piecewise.eager_attention_scope():
            assert residual.full_head_plan(batch) is plan
        with pytest.raises(ValueError, match="eager-only"):
            residual.full_head_plan(batch)


def test_compiled_ffn_uses_live_prefix_metadata_and_local_rows():
    class Norm(torch.nn.Module):
        def forward(self, x, other=None):
            return x if other is None else (x + other, x + other)

    class Attention:
        indexer = None

        def __call__(self, **kwargs):
            return kwargs["hidden_states"] * 0.1 + kwargs["forward_batch"].prefix, None

    class Mlp(torch.nn.Module):
        reduce_results = True
        _enable_a2a_moe = True

        def forward(self, x, batch, unused):
            return F.linear(x, torch.eye(4) * 2)

    layer = SimpleNamespace(
        self_attn=Attention(),
        input_layernorm=Norm(),
        post_attention_layernorm=Norm(),
        is_layer_sparse=True,
        mlp=Mlp(),
    )
    layers = [layer, SimpleNamespace(**vars(layer)), SimpleNamespace(**vars(layer))]
    buffers = {}
    for index, item in enumerate(layers):
        item._npu_dcp_piecewise_buffers = buffers
        item._npu_dcp_piecewise_slot = index % 2
    graphs = []

    def backend(gm, inputs):
        graphs.append(gm)
        return gm.forward

    @torch.compiler.disable
    def final_norm(hidden, batch):
        return residual_batch.final_norm(hidden, batch, Norm())

    def step(x, batch):
        dcp_piecewise.reset_forward(batch, x.shape[0])
        residual_batch.start(batch)
        hidden = x
        for item in layers:
            hidden, _ = residual.decoder_forward(
                item, torch.arange(x.shape[0]), hidden, batch, None
            )
        return final_norm(hidden, batch)

    with (
        patch.object(dcp_piecewise, "is_in_tc_piecewise_cuda_graph", return_value=True),
        patch.object(
            dcp_piecewise,
            "get_parallel",
            return_value=SimpleNamespace(dcp_enabled=True),
        ),
        patch.object(
            residual, "full_head_plan", side_effect=lambda batch, *args: batch.plan
        ),
    ):
        compiled = torch.compile(step, backend=backend, fullgraph=False)
        for lengths, padded, rank, prefix in (
            ([16], 16, 0, 0),
            ([11, 21], 32, 0, 100),
            ([1], 16, 15, 200),
        ):
            plan = plan_dsa_cp_shard(
                lengths,
                [prefix + n for n in lengths],
                16,
                rank,
                padded_num_tokens=padded,
            )
            batch = SimpleNamespace(plan=plan, prefix=prefix)
            x = torch.arange(padded * 4, dtype=torch.float32).reshape(padded, 4) / 31
            actual = compiled(x, batch)
            local = residual.dsa_cp_slice(x, plan)
            expected = local
            for _ in layers:
                expected = 3.3 * expected + 3 * prefix
            torch.testing.assert_close(actual, expected)
    assert graphs
    assert any("linear" in gm.code for gm in graphs), (
        "feed-forward compute fell back to eager"
    )
