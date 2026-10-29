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


def _rope_kv_fake(q, k, *args, **kwargs):
    return (
        torch.empty(q.shape, dtype=q.dtype, device=q.device),
        torch.empty(k.shape, dtype=k.dtype, device=k.device),
    )


@register_custom_op(
    op_name="npu_gemma4_rope_kv",
    mutates_args=["key_cache", "value_cache"],
    fake_impl=_rope_kv_fake,
)
def npu_gemma4_rope_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    locations: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    slot_mapping: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from sglang.kernels.ops.attention.gemma4_rope_kv_npu import gemma4_rope_kv_npu

    return gemma4_rope_kv_npu(
        q,
        k,
        v,
        cache,
        positions,
        locations,
        key_cache,
        value_cache,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        slot_mapping,
    )


def try_npu_gemma4_rope_kv(rotary, positions, q, k, v, layer, batch):
    from sglang.kernels.ops.attention.gemma4_rope_kv_npu import _check_layout
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
    from sglang.srt.layers.rotary_embedding.rope_variant import Gemma4RotaryEmbedding
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
    from sglang.srt.runtime_context import get_parallel

    if (
        not batch.forward_mode.is_extend()
        or batch.forward_mode.is_target_verify()
        or get_parallel().attn_cp_size != 1
        or get_parallel().dcp_enabled
        or type(rotary) not in (RotaryEmbedding, Gemma4RotaryEmbedding)
        or not rotary.is_neox_style
        or hasattr(rotary, "sin_cos_cache")
        or rotary._force_native
        or layer.k_scale is not None
        or layer.v_scale is not None
    ):
        return None
    pool = get_token_to_kv_pool()
    mapping = None
    physical_pool = pool
    if isinstance(pool, SWAKVPool):
        _, swa = pool.layers_mapping[layer.layer_id]
        physical_pool = pool.swa_kv_pool if swa else pool.full_kv_pool
        if swa:
            mapping = pool.full_to_swa_index_mapping
            if mapping is None:
                return None
    if type(physical_pool) is not NPUMHATokenToKVPool or not physical_pool.use_fia:
        return None
    kc = pool.get_key_buffer(layer.layer_id)
    vc = pool.get_value_buffer(layer.layer_id)
    v = v.flatten(1)
    # Reject unsupported layouts before the opaque op; kernel failures remain visible.
    try:
        _check_layout(
            q,
            k,
            v,
            rotary.cos_sin_cache,
            positions,
            batch.out_cache_loc,
            kc,
            vc,
            layer.tp_q_head_num,
            layer.tp_k_head_num,
            layer.qk_head_dim,
            rotary.rotary_dim,
        )
    except ValueError:
        return None
    return npu_gemma4_rope_kv(
        q,
        k,
        v,
        rotary.cos_sin_cache,
        positions,
        batch.out_cache_loc,
        kc,
        vc,
        layer.tp_q_head_num,
        layer.tp_k_head_num,
        layer.qk_head_dim,
        rotary.rotary_dim,
        mapping,
    )
