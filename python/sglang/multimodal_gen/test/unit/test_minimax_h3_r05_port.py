# SPDX-License-Identifier: Apache-2.0
"""Host tests for the R05 #53105 selective port (E1-E5).

Every test runs on CPU. Device-only behavior (NPU operator rounding, HCCL
exchange bytes, Laser kernel bytes) stays device-pending and is only probed
through its gating predicates here.
"""

import importlib
import json
import sys
import types
import weakref
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackendEnum,
)
from sglang.multimodal_gen.runtime.layers.usp import (
    ulysses_head_group_pipelined_attention,
)
from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as h3_dit
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    _apply_indexed_gate_runs_inplace_,
    _apply_scale_shift_runs_,
    _minimax_h3_collective_pipelined_attention,
    _modulate_gate,
    _modulate_scale_shift,
)

_REPO_ROOT = Path(__file__).resolve().parents[5]


# ---------------------------------------------------------------------------
# E1: contiguous AdaLN runs
# ---------------------------------------------------------------------------


def test_contiguous_index_runs_shapes():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
        _contiguous_index_runs,
    )

    indices = torch.tensor([1, 1, 0, 0, 0, 3, 3], dtype=torch.long)
    assert _contiguous_index_runs(indices) == ((0, 2, 1), (2, 5, 0), (5, 7, 3))
    # A single value is one run covering the whole row set.
    assert _contiguous_index_runs(torch.tensor([5, 5, 5])) == ((0, 3, 5),)
    assert _contiguous_index_runs(torch.tensor([7])) == ((0, 1, 7),)
    # Empty input and run-count overflow fall back to index_select.
    assert _contiguous_index_runs(torch.empty(0, dtype=torch.long)) is None
    assert _contiguous_index_runs(torch.arange(9)) is None
    assert _contiguous_index_runs(torch.arange(9), max_runs=16) == (
        (0, 1, 0),
        (1, 2, 1),
        (2, 3, 2),
        (3, 4, 3),
        (4, 5, 4),
        (5, 6, 5),
        (6, 7, 6),
        (7, 8, 7),
        (8, 9, 8),
    )


def test_scale_shift_runs_match_index_select_bitwise():
    """Slice broadcast over runs equals the per-row gather, element for element."""
    torch.manual_seed(0)
    table_rows, hidden = 6, 96
    scale = torch.randn(table_rows, hidden).to(torch.bfloat16)
    shift = torch.randn(table_rows, hidden).to(torch.bfloat16)
    indices = torch.tensor(
        [0] * 8 + [2] * 12 + [5] * 11 + [1] * 1, dtype=torch.long
    )
    runs = ((0, 8, 0), (8, 20, 2), (20, 31, 5), (31, 32, 1))
    rows = indices.numel()

    base = torch.randn(rows, hidden).to(torch.bfloat16)
    scale_p1 = 1.0 + scale
    reference = base.clone().mul_(scale_p1.index_select(0, indices)).add_(
        shift.index_select(0, indices)
    )
    work = base.clone()
    fast = _apply_scale_shift_runs_(work, scale_p1, shift, runs)
    assert torch.equal(fast, reference)
    # In-place contract: the returned tensor is the mutated input.
    assert fast is work


def test_gate_runs_match_index_select_bitwise():
    torch.manual_seed(1)
    table_rows, hidden, rows = 4, 64, 24
    gate = torch.randn(table_rows, hidden).to(torch.bfloat16)
    indices = torch.tensor([0] * 6 + [1] * 6 + [2] * 6 + [3] * 6, dtype=torch.long)
    runs = ((0, 6, 0), (6, 12, 1), (12, 18, 2), (12 + 6, 24, 3))
    x = torch.randn(rows, hidden).to(torch.bfloat16)
    other = torch.randn(rows, hidden).to(torch.bfloat16)

    reference = x + gate.index_select(0, indices) * other
    fast_other = other.clone()
    _apply_indexed_gate_runs_inplace_(fast_other, gate, runs)
    assert torch.equal(x + fast_other, reference)


def test_npu_modulation_branches_stay_off_cpu():
    """The NPU dispatch conditions require device 'npu'; CPU keeps the
    elementwise fallback, whose values must equal the index_select reference."""
    torch.manual_seed(2)
    rows, hidden, table_rows = 8, 32, 3
    x = torch.randn(rows, hidden).to(torch.bfloat16)
    shift = torch.randn(table_rows, hidden).to(torch.bfloat16)
    scale = torch.randn(table_rows, hidden).to(torch.bfloat16)
    indices = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1], dtype=torch.long)

    result = _modulate_scale_shift(
        x.clone(), shift, scale, indices, dtype=torch.bfloat16
    )
    expected = (
        x * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)
    ).to(torch.bfloat16)
    assert torch.equal(result, expected)

    gate = torch.randn(table_rows, hidden).to(torch.bfloat16)
    other = torch.randn(rows, hidden).to(torch.bfloat16)
    gated = _modulate_gate(x.clone(), gate, other.clone(), indices, dtype=torch.bfloat16)
    assert torch.equal(
        gated, (x + gate.index_select(0, indices) * other).to(torch.bfloat16)
    )


def test_denoise_branch_emits_and_validates_cached_runs():
    """forward_kwargs emits block_combined_runs only while the cached weakref
    still resolves to the exact tensor carried by the step plan."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
        MiniMaxH3DenoiseBranch,
    )

    seq_len = 8
    packed = {
        "seq_len": seq_len,
        "text_pos": torch.tensor([0, 1], dtype=torch.long),
        "img_pos": torch.tensor([2, 3, 4, 5], dtype=torch.long),
        "audio_pos": torch.tensor([6, 7], dtype=torch.long),
        "update_mask": torch.tensor([False, True, True, True]),
        "audio_update_mask": torch.tensor([True, True]),
        "cu_seqlens": torch.tensor([0, seq_len, seq_len], dtype=torch.int32),
        "img_position_ids": torch.zeros(4, 3, dtype=torch.float64),
    }
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(2, 8),
        token_tags=torch.zeros(seq_len, dtype=torch.long),
        device=torch.device("cpu"),
    )
    assert torch.equal(branch.block_token_tags, branch.block_token_tags_host)
    steps = branch.prepare_timestep_plan(
        video_timesteps=[0.5],
        audio_timesteps=[0.5],
        imgvid_cond_noise_aug=0.999,
        audio_ref_cond_noise_aug=1.0,
    )
    assert len(steps) == 1
    block_combined = steps[0].block_combined_indices
    runs = ((0, 5, 1), (5, 8, 3))
    video_rows = torch.zeros(4, 96)
    audio_rows = torch.zeros(2, 32)

    kwargs = branch.forward_kwargs(
        video_rows=video_rows, audio_rows=audio_rows, step_timesteps=steps[0]
    )
    assert "block_combined_runs" not in kwargs

    branch._block_combined_runs_by_id[id(block_combined)] = (
        weakref.ref(block_combined),
        runs,
    )
    kwargs = branch.forward_kwargs(
        video_rows=video_rows, audio_rows=audio_rows, step_timesteps=steps[0]
    )
    assert kwargs["block_combined_runs"] == runs

    # A dead weakref must never serve a recycled tensor id.
    stale_target = torch.zeros(2)
    stale_ref = weakref.ref(stale_target)
    del stale_target
    branch._block_combined_runs_by_id[id(block_combined)] = (stale_ref, runs)
    kwargs = branch.forward_kwargs(
        video_rows=video_rows, audio_rows=audio_rows, step_timesteps=steps[0]
    )
    assert "block_combined_runs" not in kwargs

    # A new denoise pass rebuilds the plan from scratch.
    branch.prepare_timestep_plan(
        video_timesteps=[0.4],
        audio_timesteps=[0.4],
        imgvid_cond_noise_aug=0.999,
        audio_ref_cond_noise_aug=1.0,
    )
    assert branch._block_combined_runs_by_id == {}


def test_padder_drops_block_combined_runs():
    from sglang.multimodal_gen.runtime.breakable_cuda_graph.model_padders.minimax_h3 import (
        pad_minimax_h3_prompt_kwargs,
    )

    call_kwargs = {
        "prompt_embeds": torch.zeros(2, 16),
        "text_pos_info": {"position_ids": torch.tensor([0, 1])},
        "img_pos_info": {"position_ids": torch.tensor([2, 3, 4, 5])},
        "audio_pos_info": {"position_ids": torch.tensor([6, 7])},
        "x": torch.zeros(1, 128, 96),
        "block_combined_runs": ((0, 4, 1),),
        "local_embedding_layout": {"img_row_ids": torch.zeros(4)},
        "packed_seq_params": {"cu_seqlens_q": torch.tensor([0, 8, 128])},
        "refiner_packed_seq_params": {"cu_seqlens_q": torch.tensor([0, 2, 8])},
    }
    padded = pad_minimax_h3_prompt_kwargs(call_kwargs, object(), buckets=(8, 16))
    assert padded is not call_kwargs
    assert "block_combined_runs" not in padded
    assert "local_embedding_layout" not in padded
    assert torch.equal(padded["x"], call_kwargs["x"])


# ---------------------------------------------------------------------------
# E2: packed QKV
# ---------------------------------------------------------------------------


def test_packed_qkv_cat_pack_matches_strided_copy():
    """The cat/view/permute pack produces the exact destination-major bytes of
    the three strided copies it replaces."""
    torch.manual_seed(3)
    s_local, world_size, h_local, head_size = 5, 2, 4, 16
    h_global = world_size * h_local
    qkv = [
        torch.randn(s_local, h_global, head_size).to(torch.bfloat16)
        for _ in range(3)
    ]

    reference = torch.empty(world_size, s_local, h_local, 3 * head_size).to(
        torch.bfloat16
    )
    for index, tensor in enumerate(qkv):
        head_shards = tensor.view(s_local, world_size, h_local, head_size).permute(
            1, 0, 2, 3
        )
        reference[..., index * head_size : (index + 1) * head_size].copy_(head_shards)

    packed = (
        torch.cat(qkv, dim=-1)
        .view(s_local, world_size, h_local, 3 * head_size)
        .permute(1, 0, 2, 3)
        .contiguous()
    )
    assert torch.equal(packed, reference)


def test_pipelined_attention_rejects_prepacked_off_npu():
    with pytest.raises(ValueError, match="only supported on NPU"):
        ulysses_head_group_pipelined_attention(
            None,
            None,
            None,
            attend=lambda *args: args,
            groups=2,
            prepacked_qkv=torch.zeros(2, 2, 4, 4, 12).to(torch.bfloat16),
        )


# ---------------------------------------------------------------------------
# E3: direct Laser varlen postprocessing
# ---------------------------------------------------------------------------


def _load_laser_impl():
    """Import laser_attn, stubbing the NPU operator package when absent."""
    module_name = (
        "sglang.multimodal_gen.runtime.layers.attention.backends.laser_attn"
    )
    if sys.modules.get("attentions") is None:
        stub = types.ModuleType("attentions")
        stub.la = None
        sys.modules["attentions"] = stub
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        pytest.skip("laser attention module is unavailable in this environment")
    return module


def _fake_laser_output(query_la: torch.Tensor, key_la: torch.Tensor, value_la):
    """Deterministic fake of the attention operator's BNSD output."""
    summary = (
        query_la.sum(dim=-2, keepdim=True) * key_la.sum(dim=-2, keepdim=True)
    ).to(value_la.dtype)
    return summary + value_la


def _fake_laser_call(q, k, v, *args, **kwargs):
    return None, _fake_laser_output(q, k, v)


def test_laser_varlen_fast_path_matches_dense_segment_path():
    laser = _load_laser_impl()
    impl = laser.LaserAttentionImpl(
        num_heads=8, head_size=128, causal=False, softmax_scale=0.125
    )
    torch.manual_seed(4)
    heads, head_dim, real, padded = 8, 128, 2048, 2560
    query = torch.randn(padded, heads, head_dim).to(torch.bfloat16)
    key = torch.randn(padded, heads, head_dim).to(torch.bfloat16)
    value = torch.randn(padded, heads, head_dim).to(torch.bfloat16)
    cu = torch.tensor([0, real, padded], dtype=torch.int32)
    cu_host = (0, real, padded)

    with patch.object(impl, "_laser_attention", _fake_laser_call), patch.object(
        laser.LaserAttentionImpl, "_forward_dense"
    ) as dense_mock:
        output = impl.forward_varlen(
            query,
            key,
            value,
            cu_seqlens=cu,
            max_seqlen=real,
            cu_seqlens_host=cu_host,
        )
        # The fast path must not dispatch the dense path at or above min_seqlen.
        dense_mock.assert_not_called()

    # Reference: exactly what the replaced segment branch produced.
    with patch.object(impl, "_laser_attention", _fake_laser_call):
        segment = impl._forward_dense(
            query[:real].unsqueeze(0),
            key[:real].unsqueeze(0),
            value[:real].unsqueeze(0),
            None,
            preserve_bf16_range=True,
        )
    expected = torch.empty_like(query)
    expected[:real].copy_(segment[0])
    expected[real:].zero_()

    assert torch.equal(output, expected)
    assert torch.equal(output[real:], torch.zeros_like(output[real:]))


def test_laser_postprocess_crop_cast_order_is_bitwise_neutral():
    """Cropping the padded operator output before the BF16 cast is the same
    bytes as cast-then-crop; the R01 crop-order toggle stays valid."""
    _load_laser_impl()
    torch.manual_seed(5)
    la_output = torch.randn(1, 4, 300, 128).to(torch.float16)
    qseqlen, head_dim, dtype = 256, 128, torch.bfloat16

    crop_first = la_output[:, :, :qseqlen, :head_dim].to(dtype).transpose(1, 2)
    cast_first = la_output.to(dtype)[:, :, :qseqlen, :head_dim].transpose(1, 2)
    assert torch.equal(crop_first.contiguous(), cast_first.contiguous())


# ---------------------------------------------------------------------------
# E4: NPU fused residual + norm guards
# ---------------------------------------------------------------------------


def _meta_block():
    """A MiniMaxH3DiTBlock on the meta device with NPU-style RMSNorm layers."""
    from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm as sgl_rms

    arch = MiniMaxH3DiTArchConfig()
    with torch.device("meta"):
        block = h3_dit.MiniMaxH3DiTBlock(
            arch, None, prefix="blocks.0", layer_index=0
        )
        # NPU builds use the sgl RMSNorm, which exposes variance_epsilon and
        # no eps attribute; replicate that contract explicitly.
        block.norm1 = sgl_rms(arch.hidden_size, eps=arch.norm_eps)
        block.norm2 = sgl_rms(arch.hidden_size, eps=arch.norm_eps)
    return block


def test_fuses_gate_norm2_disabled_on_npu_without_eps(single_process_model_parallel):
    """The NPU exclusion must short-circuit before reading norm2.eps (the
    #53105 crash guard: NPU RMSNorm carries variance_epsilon, not eps)."""
    block = _meta_block()
    assert not hasattr(block.norm2, "eps")
    assert hasattr(block.norm2, "variance_epsilon")

    rows = 6
    hidden = MiniMaxH3DiTArchConfig().hidden_size
    bf16 = torch.bfloat16
    x = torch.zeros(rows, hidden, dtype=bf16)
    gate = torch.zeros(rows, hidden, dtype=bf16)
    other = torch.zeros(rows, hidden, dtype=bf16)
    shift = torch.zeros(rows, hidden, dtype=bf16)
    scale = torch.zeros(rows, hidden, dtype=bf16)
    indices = torch.arange(rows)
    with patch.object(h3_dit.current_platform, "is_npu", return_value=True):
        # Must return False, not raise AttributeError on norm2.eps.
        assert block._fuses_gate_norm2(x, gate, other, shift, scale, indices) is False


def test_uses_npu_fused_gate_norm2_off_cpu(single_process_model_parallel):
    block = _meta_block()
    rows = 6
    hidden = MiniMaxH3DiTArchConfig().hidden_size
    bf16 = torch.bfloat16
    args = [torch.zeros(rows, hidden, dtype=bf16) for _ in range(5)]
    # Mount the fusion as enabled on the block (the gate's per-module flag),
    # so only the CPU-resident tensors can veto the NPU fused path.
    block._sgl_minimax_h3_norm_modulate_enabled = True
    with patch.object(h3_dit.current_platform, "is_npu", return_value=True):
        # CPU-resident tensors never take the NPU fused path.
        assert block._uses_npu_fused_gate_norm2(*args) is False


# ---------------------------------------------------------------------------
# E5: NPU Laser collective pipelining gates
# ---------------------------------------------------------------------------


def _pipeline_attention(attention, q):
    return _minimax_h3_collective_pipelined_attention(
        attention,
        q,
        torch.zeros_like(q),
        torch.zeros_like(q),
        cu_seqlens=torch.tensor([0, q.shape[0] * 8], dtype=torch.int32),
        cu_seqlens_host=(0, q.shape[0] * 8),
        max_seqlen=q.shape[0] * 8,
    )


def test_npu_laser_pipeline_gates():
    marker = torch.zeros(1)
    stub = Mock(return_value=marker)
    attention = Mock()
    attention._attention_backend_enum = AttentionBackendEnum.LASER_ATTN
    attention._attention_impl = Mock()

    # Global rows 2048 * 8 = 16384 >= 8192: two groups accepted. Default env
    # flags (flashinfer off, pipeline groups -1 -> 2) must serve the call.
    q = torch.zeros(2048, 16, 128).to(torch.bfloat16)
    with (
        patch.object(h3_dit.current_platform, "is_npu", return_value=True),
        patch.object(h3_dit, "get_ulysses_ctx", return_value=(8, 0)),
        patch.object(h3_dit, "is_in_breakable_cuda_graph", return_value=False),
        patch.object(h3_dit, "ulysses_head_group_pipelined_attention", stub),
    ):
        assert _pipeline_attention(attention, q) is marker
        assert stub.call_args.args[4] == 2

    # Below the 8192-token threshold the sequential exchange stays.
    stub.reset_mock()
    q_small = torch.zeros(512, 16, 128).to(torch.bfloat16)
    with (
        patch.object(h3_dit.current_platform, "is_npu", return_value=True),
        patch.object(h3_dit, "get_ulysses_ctx", return_value=(8, 0)),
        patch.object(h3_dit, "is_in_breakable_cuda_graph", return_value=False),
        patch.object(h3_dit, "ulysses_head_group_pipelined_attention", stub),
    ):
        assert _pipeline_attention(attention, q_small) is None
        stub.assert_not_called()

    # A captured breakable graph never pipelines.
    with (
        patch.object(h3_dit.current_platform, "is_npu", return_value=True),
        patch.object(h3_dit, "get_ulysses_ctx", return_value=(8, 0)),
        patch.object(h3_dit, "is_in_breakable_cuda_graph", return_value=True),
        patch.object(h3_dit, "ulysses_head_group_pipelined_attention", stub),
    ):
        assert _pipeline_attention(attention, q) is None

    # Off NPU, Laser keeps the sequential exchange; the AITER path is intact.
    attention_aiter = Mock()
    attention_aiter._attention_backend_enum = AttentionBackendEnum.AITER
    attention_aiter._attention_impl = Mock()
    with (
        patch.object(h3_dit.current_platform, "is_npu", return_value=False),
        patch.object(h3_dit, "get_ulysses_ctx", return_value=(8, 0)),
        patch.object(h3_dit, "is_in_breakable_cuda_graph", return_value=False),
        patch.object(h3_dit, "ulysses_head_group_pipelined_attention", stub),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
    ):
        assert _pipeline_attention(attention_aiter, q) is marker
        assert stub.call_args.args[4] == 2


def test_r05_comm_probe_reports_without_device():
    """The E6/E7 probe must run host-only and emit explicit verdicts."""
    import subprocess

    probe = _REPO_ROOT / "test" / "manual" / "minimax_h3_realtime" / "research" / (
        "r05_comm_probe.py"
    )
    result = subprocess.run(
        [sys.executable, str(probe), "--json"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert set(report["subtasks"]) >= {
        "E4_residual_norm_fusion",
        "E5_ulysses_head_group_pipeline",
        "E6_row_shard_tp_probe",
        "E7_quantized_a2a_probe",
    }
    # Without a device, E6/E7 must report an explicit verdict, never silently
    # claim readiness.
    assert report["subtasks"]["E6_row_shard_tp_probe"]["verdict"] in {
        "blocked",
        "available",
    }
    assert report["subtasks"]["E7_quantized_a2a_probe"]["verdict"] in {
        "blocked",
        "available",
    }
