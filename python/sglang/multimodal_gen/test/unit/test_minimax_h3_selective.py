# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3DiTBlock
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_selective import (
    selective_ffn_rows,
)


@pytest.mark.parametrize("fraction", [0.5, 0.75])
def test_packed_plans_preserve_modalities_boundaries_and_sp_coordinates(fraction):
    # Reference span, generated span, then audio/padding. Only target interiors skip.
    spans = [
        dict(start=2, latent_shape=[3, 1, 2]),
        dict(start=8, latent_shape=[8, 1, 2]),
    ]
    kwargs = dict(
        seq_len=28,
        video_spans=spans,
        target_rows=torch.arange(8, 24),
        active_fraction=fraction,
        device="cpu",
    )
    global_plans = selective_ffn_rows(row_start=0, row_stop=28, **kwargs)
    local_plans = selective_ffn_rows(row_start=14, row_stop=28, **kwargs)
    protected = set(range(10)) | set(range(22, 28))
    for global_rows, local_rows in zip(global_plans, local_plans):
        assert protected <= set(global_rows.tolist())
        assert torch.equal(global_rows[global_rows >= 14] - 14, local_rows)
        assert global_rows.unique().numel() == global_rows.numel()
    for row in range(10, 22):
        assert sum(row in ids for ids in global_plans) == int(4 * fraction)


class _Attention(nn.Module):
    qkv_proj = None

    def forward(self, x, **kwargs):
        return torch.zeros_like(x)


class _MLP(nn.Module):
    fc1 = None

    def forward(self, x):
        self.rows = x.shape[0]
        return x * 2


@pytest.mark.parametrize("layer", [0, 2, 9])
@torch.inference_mode()
def test_ffn_compacts_before_gemm_and_preserves_inactive_residuals(layer):
    block = MiniMaxH3DiTBlock.__new__(MiniMaxH3DiTBlock)
    nn.Module.__init__(block)
    block.layer_index, block.num_layers = layer, 10
    block.attn, block.mlp = _Attention(), _MLP()
    block.norm1 = block.norm2 = nn.Identity()
    block.preserve_input_for_cache_dit = False
    block._fuses_gate_norm2 = lambda *args: False
    x = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    # Different per-row modulation detects accidental compact index rebasing.
    indices = torch.tensor([0, 1, 0, 1, 0, 1])
    zero, one = torch.zeros(2, 4), torch.ones(2, 4)
    params = (zero, zero, zero, one, zero, one)
    kwargs = dict(
        adaln_input=torch.empty(0),
        combined_indices=indices,
        rope_cache=None,
        cu_seqlens=torch.tensor([0, 6]),
        max_seqlen=6,
        adaln_params=params,
    )
    expected = block(x.clone(), **kwargs)
    active = torch.tensor([0, 3, 5])
    result = block(x.clone(), selective_ffn_rows=(active,) * 4, **kwargs)
    if layer in (0, 9):
        assert torch.equal(result, expected)
        assert block.mlp.rows == 6
    else:
        assert block.mlp.rows == 3
        assert torch.equal(result[active], expected[active])
        assert torch.equal(result[[1, 2, 4]], x[[1, 2, 4]])


def test_reject_missing_video_geometry():
    with pytest.raises(ValueError, match="video spans"):
        selective_ffn_rows(
            seq_len=4,
            row_start=0,
            row_stop=4,
            video_spans=[],
            target_rows=torch.arange(4),
            active_fraction=0.5,
            device="cpu",
        )
