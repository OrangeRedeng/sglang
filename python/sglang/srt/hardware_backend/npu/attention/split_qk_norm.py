"""Compiler boundary for the external split/QK RMSNorm Triton launcher."""

from types import SimpleNamespace
from typing import Optional, Tuple

import torch

from sglang.srt.utils.custom_op import register_custom_op


def _fake(x, q_weight, k_weight, q_bias, k_bias, q_rank, kv_rank, rope_dim, eps):
    return (
        x.new_empty((x.shape[0], q_rank)),
        x.new_empty((x.shape[0], 1, kv_rank)),
        x.new_empty((x.shape[0], 1, rope_dim)),
    )


@register_custom_op(op_name="npu_split_qk_norm", fake_impl=_fake)
def _split_qk_norm(
    x: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_bias: Optional[torch.Tensor],
    k_bias: Optional[torch.Tensor],
    q_rank: int,
    kv_rank: int,
    rope_dim: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from sgl_kernel_npu.norm.fused_split_qk_norm import fused_split_qk_norm

    # Device discovery and Triton launch must execute outside Dynamo tracing.
    return fused_split_qk_norm(
        x,
        SimpleNamespace(weight=q_weight, bias=q_bias),
        SimpleNamespace(weight=k_weight, bias=k_bias),
        q_rank,
        kv_rank,
        rope_dim,
        eps=eps,
    )


def fused_split_qk_norm(x, q_norm, k_norm, q_rank, kv_rank, rope_dim, eps=1e-6):
    return _split_qk_norm(
        x,
        q_norm.weight,
        k_norm.weight,
        getattr(q_norm, "bias", None),
        getattr(k_norm, "bias", None),
        q_rank,
        kv_rank,
        rope_dim,
        eps,
    )
