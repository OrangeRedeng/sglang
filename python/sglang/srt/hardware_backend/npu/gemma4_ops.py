"""Traceable boundary for the opt-in Gemma 4 NPU Q/K/V norm fusion."""

from typing import Optional

import torch

from sglang.kernels.ops.layernorm.gemma4_fused_ops import gemma_qkv_rmsnorm
from sglang.srt.utils.custom_op import register_custom_op


@register_custom_op(op_name="npu_gemma4_qkv_rmsnorm", mutates_args=["q", "k", "v"])
def npu_gemma4_qkv_rmsnorm(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    q_weight: torch.Tensor,
    k_weight: Optional[torch.Tensor],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float = 1e-6,
) -> None:
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
