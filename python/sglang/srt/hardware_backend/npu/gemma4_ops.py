"""Traceable boundary for the opt-in Gemma 4 NPU Q/K/V norm fusion."""

import torch
from sglang.kernels.ops.layernorm.gemma4_fused_ops import (
    _gemma_qkv_rmsnorm_npu_tiled,
    gemma_qkv_rmsnorm,
)
from sglang.srt.utils.custom_op import register_custom_op


@register_custom_op(op_name="npu_gemma4_qkv_rmsnorm", mutates_args=["q", "k", "v"])
def npu_gemma4_qkv_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor | None,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float = 1e-6,
    *,
    rows_per_program: int = 0,
) -> None:
    if rows_per_program:
        _gemma_qkv_rmsnorm_npu_tiled(
            q,
            k,
            v,
            q_weight,
            k_weight,
            num_q_heads,
            num_kv_heads,
            head_dim,
            eps,
            rows_per_program=rows_per_program,
        )
        return
    gemma_qkv_rmsnorm(
        q,
        k,
        v,
        q_weight,
        k_weight,
        num_q_heads,
        num_kv_heads,
        head_dim,
        eps,
    )
