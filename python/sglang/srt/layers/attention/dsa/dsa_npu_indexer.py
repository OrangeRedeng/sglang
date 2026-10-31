from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

import torch
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_cp import (
    dsa_cp_token_sharded,
    get_dsa_cp_plan,
)
from sglang.srt.layers.attention.dsa.dsa_token_shard import get_dsa_token_shard_plan
from sglang.srt.layers.attention.dsa.dsa_token_shard_layout import (
    cumulative,
    plan_dsa_token_shard,
)
from sglang.srt.layers.cp.utils import cp_gather_full_sequence_states
from sglang.srt.layers.dp_attention import attn_tp_all_gather_into_tensor
from sglang.srt.layers.layer_boundary import get_attn_tp_context
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
    has_forward_context,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_npu, print_info_once

if is_npu():
    import torch_npu
    from sglang.srt.hardware_backend.npu.utils import get_indexer_weight_stream

_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()


@lru_cache(maxsize=1)
def _create_hadamard_128_cpu() -> torch.Tensor:
    matrix = [[1.0]]
    while len(matrix) < 128:
        matrix = [row + row for row in matrix] + [
            row + [-value for value in row] for row in matrix
        ]
    return torch.tensor(matrix, dtype=torch.bfloat16)


def create_npu_hadamard_128(head_dim: int, device) -> torch.Tensor:
    assert head_dim == 128
    # Match vllm-ascend SFA: BF16 matrix, normalized once on the pool's device.
    return (_create_hadamard_128_cpu().to(device=device) / (128**0.5)).contiguous()


def _quantize_npu_indexer_activation(x, hadamard, dst_type, scale_type):
    assert x.dtype == torch.bfloat16 and x.shape[-1] == 128
    if x.numel() == 0:
        return (
            torch.empty_like(x, dtype=dst_type),
            torch.empty(x.shape[:-1], dtype=scale_type, device=x.device),
        )
    rotated = x @ hadamard
    quantized, scale = torch_npu.npu_dynamic_quant(
        rotated.reshape(-1, 128), dst_type=dst_type
    )
    return quantized.reshape(x.shape), scale.to(scale_type).reshape(x.shape[:-1])


def plan_indexer_query_shard(
    prefix_lens: List[int], extend_lens: List[int], tp_size: int, tp_rank: int
):
    """One attention-TP rank's share of an extend batch's indexer queries.

    The token shard's partition, derived rather than recomputed: the token
    shard skips this top-k's all-gather on the strength of both picking the
    same rows, so they must not be two implementations.

    Returns ``(start, rows, num_real, cum_query_lens, key_lens)``.
    """
    plan = plan_dsa_token_shard(
        extend_lens,
        [p + e for p, e in zip(prefix_lens, extend_lens)],
        tp_size,
        tp_rank,
    )
    return (
        plan.local_start,
        plan.rows,
        max(0, plan.local_end - plan.local_start),
        cumulative(plan.query_lens),
        plan.key_lens,
    )


@dataclass
class _IndexerQueryShard:
    """This rank's rows of an extend batch, and how to reassemble the top-k."""

    start: int
    rows: int
    num_real: int
    total: int
    tp_size: int
    actual_seq_lengths_q: torch.Tensor
    actual_seq_lengths_kv: torch.Tensor

    def take(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_real == self.rows:
            return x[self.start : self.start + self.rows]
        # Padding rows lie past actual_seq_lengths_q[-1]; gather() drops them.
        out = x.new_zeros((self.rows, *x.shape[1:]))
        out[: self.num_real] = x[self.start : self.start + self.num_real]
        return out

    def gather(self, topk_indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """All-gather the per-rank top-k back to ``num_tokens`` rows."""
        out = topk_indices.new_empty((self.rows * self.tp_size, topk_indices.shape[-1]))
        attn_tp_all_gather_into_tensor(out, topk_indices.contiguous())
        return out[:num_tokens]

    def resolve(
        self, topk_indices: torch.Tensor, num_tokens: int, forward_batch
    ) -> torch.Tensor:
        """Full-width top-k, or this rank's own rows when the token shard will
        slice the gathered tensor straight back to them anyway.

        Sets ``npu_indexer_topk_is_local`` so attention knows which it got.
        """
        plan = (
            get_dsa_cp_plan(forward_batch)
            if getattr(forward_batch, "npu_dsa_cp_attention_local", False)
            or dsa_cp_token_sharded(forward_batch)
            else get_dsa_token_shard_plan(forward_batch)
        )
        local = plan is not None and (
            dsa_cp_token_sharded(forward_batch)
            or envs.SGLANG_NPU_DSA_CP_LOCAL_TOPK.get()
            or not getattr(forward_batch, "npu_dsa_cp_attention_local", False)
        )
        if local and (plan.rows != self.rows or plan.local_start != self.start):
            # Same arithmetic, but the two sites read their inputs separately;
            # if those ever drift, the gather stays correct where this would not.
            if envs.SGLANG_NPU_DEBUG_DCP.get():
                print_info_once(
                    "DSA token-shard and indexer sharding disagree on the row range "
                    f"(plan {plan.local_start}+{plan.rows} against shard "
                    f"{self.start}+{self.rows}); keeping the top-k all-gather"
                )
            local = False
        forward_batch.npu_indexer_topk_is_local = local
        if not local:
            return self.gather(topk_indices, num_tokens)
        # Every rank logs, so a grep counts ranks: the output is identical either
        # way, which makes this line the only evidence it ran.
        if envs.SGLANG_NPU_DEBUG_DCP.get():
            print_info_once(
                "DSA token-shard: skipping the indexer top-k all-gather; each rank "
                "keeps the rows it scored"
            )
        if self.num_real < self.rows:
            topk_indices[self.num_real :] = 0
        from sglang.srt.hardware_backend.npu.feature_evidence import record

        record("dsa-local-topk")
        return topk_indices


def prepare_graph_indexer_query_shard(forward_batch, cache):
    parallel = get_parallel()
    total = forward_batch.input_ids.numel()
    rows = (total + parallel.attn_tp_size - 1) // parallel.attn_tp_size
    start = parallel.attn_tp_rank * rows
    key = (total, forward_batch.batch_size)
    if key not in cache:
        device = forward_batch.seq_lens.device
        cache[key] = _IndexerQueryShard(
            start,
            rows,
            max(0, min(rows, total - start)),
            total,
            parallel.attn_tp_size,
            torch.empty(forward_batch.batch_size, dtype=torch.int32, device=device),
            torch.empty(forward_batch.batch_size, dtype=torch.int32, device=device),
        )
    shard = cache[key]
    # Refresh captured addresses before replay; request boundaries can change at fixed token count.
    shard.actual_seq_lengths_q.copy_(
        (forward_batch.extend_seq_lens.cumsum(0) - start).clamp(min=0, max=rows)
    )
    shard.actual_seq_lengths_kv.copy_(forward_batch.seq_lens)
    return shard


def _build_indexer_query_shard(
    forward_batch: ForwardBatch,
) -> Optional[_IndexerQueryShard]:
    parallel = get_parallel()
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    extend_lens = forward_batch.extend_seq_lens_cpu
    if (
        parallel.attn_tp_size == 1
        or forward_batch.forward_mode not in (ForwardMode.EXTEND, ForwardMode.MIXED)
        or prefix_lens is None
        or extend_lens is None
        or sum(extend_lens) == 0
    ):
        return None
    start, rows, num_real, cum_query_lens, key_lens = plan_indexer_query_shard(
        prefix_lens, extend_lens, parallel.attn_tp_size, parallel.attn_tp_rank
    )
    # Debug output includes every rank; /server_info cannot show this path.
    if envs.SGLANG_NPU_DEBUG_DCP.get():
        print_info_once(
            "DSA indexer query sharding is active: prefill indexer queries split "
            f"across attn_tp_size={parallel.attn_tp_size}"
        )
    device = forward_batch.seq_lens.device
    return _IndexerQueryShard(
        start=start,
        rows=rows,
        num_real=num_real,
        total=sum(extend_lens),
        tp_size=parallel.attn_tp_size,
        actual_seq_lengths_q=torch.tensor(
            cum_query_lens, dtype=torch.int32, device=device
        ),
        actual_seq_lengths_kv=torch.tensor(key_lens, dtype=torch.int32, device=device),
    )


def _get_indexer_query_shard(
    forward_batch: ForwardBatch, num_tokens: int
) -> Optional[_IndexerQueryShard]:
    """The query shard for a prefill indexer call, or None to score every row.

    Every rank already holds the full index-K, so each scores 1/attn_tp_size of
    the queries and all-gathers the top-k: 15.97x less indexer work at 1M and
    tp16. Planned once per forward from inputs identical across the group, so
    all ranks take the collective or none do.
    """
    if getattr(
        forward_batch, "npu_dsa_cp_attention_local", False
    ) or dsa_cp_token_sharded(forward_batch):
        plan = get_dsa_cp_plan(forward_batch)
        if plan is None or num_tokens != plan.rows:
            raise ValueError("indexer received a different local token grid")
        device = forward_batch.positions.device
        return _IndexerQueryShard(
            start=plan.local_start,
            rows=plan.rows,
            num_real=plan.num_local_tokens,
            total=plan.num_tokens,
            tp_size=plan.num_tokens_pad // plan.rows,
            actual_seq_lengths_q=torch.tensor(
                cumulative(plan.query_lens), dtype=torch.int32, device=device
            ),
            actual_seq_lengths_kv=torch.tensor(
                plan.key_lens, dtype=torch.int32, device=device
            ),
        )
    if get_attn_tp_context().input_scattered:
        # A scattered input is already sliced.
        if envs.SGLANG_NPU_DEBUG_DCP.get():
            print_info_once(
                "DSA indexer query sharding is off: the attention input "
                "is scattered across the TP group"
            )
        return None
    backend = get_attn_backend() if has_forward_context() else None
    if getattr(backend, "graph_mode", False) and hasattr(
        backend, "npu_graph_indexer_shard"
    ):
        return backend.npu_graph_indexer_shard
    if not hasattr(forward_batch, "npu_indexer_query_shard"):
        forward_batch.npu_indexer_query_shard = _build_indexer_query_shard(
            forward_batch
        )
    shard = forward_batch.npu_indexer_query_shard
    if shard is None:
        return None
    # SGLang pads the width to a multiple of attn_tp_size; comparing against
    # total alone would disable sharding for 15 token counts in 16.
    padded = shard.rows * shard.tp_size
    if not shard.total <= num_tokens <= padded:
        if envs.SGLANG_NPU_DEBUG_DCP.get():
            print_info_once(
                "DSA indexer query sharding is off for this forward: "
                f"num_tokens={num_tokens} is outside [{shard.total}, {padded}]"
            )
        return None
    return shard


class DSANPUIndexerMixin:
    def forward_npu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        input_on_attn_tp_slices: bool = False,
        dynamic_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        # DSA prefill CP metadata is Ascend-local and is first consumed by this
        # indexer (before the attention backend's forward_extend), so prepare it
        # here rather than in the shared eager runner.
        attn_backend = get_attn_backend()
        if hasattr(attn_backend, "prepare_dsa_cp_metadata"):
            attn_backend.prepare_dsa_cp_metadata(forward_batch)
        if get_attn_backend().forward_metadata.seq_lens_cpu_int is None:
            actual_seq_lengths_kv = get_attn_backend().forward_metadata.seq_lens
        else:
            actual_seq_lengths_kv = get_attn_backend().forward_metadata.seq_lens_cpu_int
        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_target_verify()
        )

        bs = q_lora.shape[0]
        token_local = getattr(forward_batch, "npu_dsa_cp_attention_local", False)

        # A rank that scores only its own rows projects only those: wq_b is
        # 25.2 MFLOP per token. k stays full width -- every rank writes the whole
        # index-K, which is what lets any of them score a subset. Neox only: the
        # other branch rotates q and k in one call. Prefill CP is excluded: its
        # indexer below knows nothing of this shard.
        uses_prefill_cp = (
            self.dsa_enable_prefill_cp and forward_batch.attn_cp_metadata is not None
        )
        shard = (
            _get_indexer_query_shard(forward_batch, bs)
            if is_prefill
            and (token_local or envs.SGLANG_NPU_ENABLE_DSA_INDEXER_QUERY_SHARDING.get())
            and not uses_prefill_cp
            else None
        )
        q_sliced_early = shard is not None and self.rotary_emb.is_neox_style
        if q_sliced_early and not token_local:
            if dynamic_scale is not None:
                dynamic_scale = shard.take(dynamic_scale)
            q_lora = shard.take(q_lora)
            bs_q = shard.rows
        else:
            bs_q = bs

        if self.rotary_emb.is_neox_style:
            if not hasattr(forward_batch, "npu_indexer_sin_cos_cache"):
                cos_sin = self.rotary_emb.cos_sin_cache[positions]
                cos, sin = cos_sin.chunk(2, dim=-1)
                cos = cos.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
                sin = sin.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
                forward_batch.npu_indexer_sin_cos_cache = (sin, cos)
            else:
                sin, cos = forward_batch.npu_indexer_sin_cos_cache

            # q rotates on this rank's rows; k still needs every one of them.
            sin_q, cos_q = (
                (shard.take(sin), shard.take(cos))
                if q_sliced_early and not token_local
                else (sin, cos)
            )

            if self.alt_stream is not None:
                self.alt_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(self.alt_stream):
                    q_lora = (
                        (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
                    )
                    q = self.wq_b(q_lora)[
                        0
                    ]  # [rows, 1536] @ [1536, 64 * 128] = [rows, 64 * 128]
                    q = q.view(bs_q, self.n_heads, self.head_dim)  # [rows, 64, 128]
                    q_pe, q_nope = torch.split(
                        q,
                        [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                        dim=-1,
                    )  # [rows, 64, 64 + 64]
                    q_pe = q_pe.view(bs_q, self.n_heads, 1, self.rope_head_dim)
                    q_pe = torch_npu.npu_rotary_mul(q_pe, cos_q, sin_q).view(
                        bs_q, self.n_heads, self.rope_head_dim
                    )  # [rows, n, d]
                    q = torch.cat([q_pe, q_nope], dim=-1)
                    q.record_stream(self.alt_stream)
                    q_rope_event = self.alt_stream.record_event()
            else:
                q_lora = (
                    (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
                )
                q = self.wq_b(q_lora)[
                    0
                ]  # [rows, 1536] @ [1536, 64 * 128] = [rows, 64 * 128]
                q = q.view(bs_q, self.n_heads, self.head_dim)  # [rows, 64, 128]
                q_pe, q_nope = torch.split(
                    q,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )  # [rows, 64, 64 + 64]
                q_pe = q_pe.view(bs_q, self.n_heads, 1, self.rope_head_dim)
                q_pe = torch_npu.npu_rotary_mul(q_pe, cos_q, sin_q).view(
                    bs_q, self.n_heads, self.rope_head_dim
                )  # [rows, n, d]
                q = torch.cat([q_pe, q_nope], dim=-1)

            if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
                indexer_weight_stream = get_indexer_weight_stream()
                indexer_weight_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(indexer_weight_stream):
                    x = x.view(-1, self.hidden_size)
                    weights = self.weights_proj(x.float())[0].to(torch.bfloat16)
                    weights.record_stream(indexer_weight_stream)
                    weights_event = indexer_weight_stream.record_event()
            else:
                x = x.view(-1, self.hidden_size)
                weights = self.weights_proj(x.float())[0].to(torch.bfloat16)

            k_proj = self.wk(x)[0]  # [b, s, 7168] @ [7168, 128] = [b, s, 128]
            k = self.k_norm(k_proj)
            if _use_ag_after_qlora and input_on_attn_tp_slices:
                k = scattered_to_tp_attn_full(k, forward_batch)
            k_pe, k_nope = torch.split(
                k,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )  # [bs, 64 + 64]

            k_pe = k_pe.view(-1, 1, 1, self.rope_head_dim)
            k_pe = torch.ops.npu.npu_rotary_mul(k_pe, cos, sin).view(
                bs, 1, self.rope_head_dim
            )  # [bs, 1, d]
            k = torch.cat([k_pe, k_nope.unsqueeze(1)], dim=-1)  # [bs, 1, 128]

        else:
            if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
                indexer_weight_stream = get_indexer_weight_stream()
                indexer_weight_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(indexer_weight_stream):
                    x = x.view(-1, self.hidden_size)
                    weights = self.weights_proj(x.float())[0].to(torch.bfloat16)
                    weights.record_stream(indexer_weight_stream)
                    weights_event = indexer_weight_stream.record_event()
            else:
                x = x.view(-1, self.hidden_size)
                weights = self.weights_proj(x.float())[0].to(torch.bfloat16)

            q_lora = (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
            q = self.wq_b(q_lora)[0]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
            q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
            q_pe, q_nope = torch.split(
                q,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )  # [bs, 64, 64 + 64]

            k_proj = self.wk(x)[0]  # [b, s, 7168] @ [7168, 128] = [b, s, 128]
            k = self.k_norm(k_proj)
            k_pe, k_nope = torch.split(
                k,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )  # [bs, 64 + 64]

            k_pe = k_pe.unsqueeze(1)

            if layer_id == get_token_to_kv_pool().start_layer:
                self.rotary_emb.sin_cos_cache = (
                    self.rotary_emb.cos_sin_cache.index_select(0, positions)
                )

            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
            k_pe = k_pe.squeeze(1)
            q = torch.cat([q_pe, q_nope], dim=-1)
            k = torch.cat([k_pe, k_nope], dim=-1)

        if (
            is_prefill
            and self.dsa_enable_prefill_cp
            and forward_batch.attn_cp_metadata is not None
        ):
            k = cp_gather_full_sequence_states(
                k.contiguous().view(-1, self.head_dim),
                forward_batch,
                torch.npu.current_stream(),
            )

        indexer_cache_loc = forward_batch.out_cache_loc
        parallel = get_parallel()
        if (
            parallel.dcp_enabled
            and parallel.attn_dcp_size > 1
            and not get_attn_backend().is_draft_worker
        ):
            indexer_cache_loc = (
                get_attn_backend().forward_metadata.dcp_origin_out_cache_loc
            )
            assert indexer_cache_loc is not None, (
                "NPU DSA+DCP requires allocator-global origin_out_cache_loc metadata"
            )
            expected_rows = (
                get_dsa_cp_plan(forward_batch).num_tokens_pad
                if token_local
                else positions.shape[0]
            )
            assert indexer_cache_loc.shape[0] == expected_rows, (
                "NPU DSA+DCP origin_out_cache_loc metadata has an incompatible "
                f"length: {indexer_cache_loc.shape[0]} != {expected_rows}"
            )
        pool = get_token_to_kv_pool()
        use_quant_indexer = pool.index_k_scale_buffer is not None
        if use_quant_indexer:
            k, k_scale = _quantize_npu_indexer_activation(
                k,
                pool.indexer_hadamard_128,
                pool.index_k_store_dtype,
                pool.index_k_scale_dtype,
            )
        if token_local:
            # Every rank scores local queries against the complete replicated index-K.
            group = parallel.attn_tp_group
            send = k.reshape(-1, k.shape[-1]).contiguous()
            full = send.new_empty((send.shape[0] * group.world_size, send.shape[-1]))
            group.all_gather_into_tensor(full, send)
            k = full
            if use_quant_indexer:
                send_scale = k_scale.reshape(-1).contiguous()
                full_scale = send_scale.new_empty(send_scale.numel() * group.world_size)
                group.all_gather_into_tensor(full_scale, send_scale)
                k_scale = full_scale
        if use_quant_indexer:
            pool.set_index_k_scale_buffer(layer_id, indexer_cache_loc, k_scale)
        pool.set_index_k_buffer(layer_id, indexer_cache_loc, k)
        if is_prefill:
            if (
                self.dsa_enable_prefill_cp
                and forward_batch.attn_cp_metadata is not None
            ):
                # V2: local metadata already prepared by prepare_dsa_cp_metadata.
                actual_seq_lengths_q = (
                    get_attn_backend().forward_metadata.actual_seq_lengths_q
                )
                actual_seq_lengths_kv = (
                    get_attn_backend().forward_metadata.actual_seq_lengths_kv
                )
            else:
                actual_seq_lengths_kv = forward_batch.seq_lens
                actual_seq_lengths_q = forward_batch.extend_seq_lens.cumsum(dim=0)
        else:
            if get_attn_backend().forward_metadata.actual_seq_lengths_q is None:
                if (
                    forward_batch.forward_mode.is_draft_extend_v2()
                    or forward_batch.forward_mode.is_target_verify()
                ):
                    num_draft_tokens = get_attn_backend().speculative_num_draft_tokens
                    actual_seq_lengths_q = torch.arange(
                        num_draft_tokens,
                        num_draft_tokens + bs,
                        num_draft_tokens,
                        dtype=torch.int32,
                        device=k.device,
                    )
                else:
                    actual_seq_lengths_q = torch.tensor(
                        [1 + i * 1 for i in range(bs)],
                        dtype=torch.int32,
                        device=k.device,
                    )
            else:
                actual_seq_lengths_q = (
                    get_attn_backend().forward_metadata.actual_seq_lengths_q
                )

        past_key_states = get_token_to_kv_pool().get_index_k_buffer(layer_id)

        if self.rotary_emb.is_neox_style and self.alt_stream is not None:
            torch.npu.current_stream().wait_event(q_rope_event)
        if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
            torch.npu.current_stream().wait_event(weights_event)
        if _use_ag_after_qlora and input_on_attn_tp_slices:
            weights = scattered_to_tp_attn_full(weights, forward_batch)
        block_table = get_attn_backend().forward_metadata.block_tables
        if (
            is_prefill
            and self.dsa_enable_prefill_cp
            and forward_batch.attn_cp_metadata is not None
        ):
            if use_quant_indexer:
                raise RuntimeError(
                    "DSANPUIndexerMixin.do_npu_cp_balance_indexer is not implemented for quantized indexer cache"
                )
            topk_indices = self.do_npu_cp_balance_indexer(
                q.view(-1, self.n_heads, self.head_dim),
                past_key_states,
                weights,
                actual_seq_lengths_q,
                actual_seq_lengths_kv,
                block_table,
            )
            return topk_indices
        else:
            block_table = (
                block_table[: actual_seq_lengths_q.size()[0]]
                if is_prefill
                else block_table
            )
            query = q.view(-1, self.n_heads, self.head_dim)
            # The full width for the gather, NOT query.shape[0] (already cut).
            num_query_tokens = (
                get_dsa_cp_plan(forward_batch).num_tokens_pad if token_local else bs
            )
            if shard is not None:
                if not q_sliced_early and not token_local:
                    query = shard.take(query)
                # A late take: under _use_ag_after_qlora weights is gathered
                # to full width above, so an earlier slice would cut twice.
                if not token_local:
                    weights = shard.take(weights)
                actual_seq_lengths_q = shard.actual_seq_lengths_q
                actual_seq_lengths_kv = shard.actual_seq_lengths_kv

            if use_quant_indexer:
                query, query_scale = _quantize_npu_indexer_activation(
                    query,
                    pool.indexer_hadamard_128,
                    pool.index_k_store_dtype,
                    pool.index_k_scale_dtype,
                )

                # int8 keys requires fp16 weights
                op_weights = (
                    weights.to(torch.float16)
                    if pool.index_k_store_dtype == torch.int8
                    else weights
                )

                if pool.index_k_store_dtype == torch.int8:
                    # torch_npu.npu_quant_lightning_indexer is not working
                    # for int8 cache in case of custom_ops imported
                    quant_lightning_indexer = torch.ops.npu.npu_quant_lightning_indexer
                else:
                    quant_lightning_indexer = torch_npu.npu_quant_lightning_indexer

                topk_indices = quant_lightning_indexer(
                    query=query,
                    key=past_key_states,
                    weights=op_weights,
                    query_dequant_scale=query_scale,
                    key_dequant_scale=pool.get_index_k_scale_buffer(layer_id),
                    actual_seq_lengths_query=actual_seq_lengths_q.to(torch.int32),
                    actual_seq_lengths_key=actual_seq_lengths_kv.to(
                        device=k.device, dtype=torch.int32
                    ),
                    block_table=block_table,
                    layout_query="TND",
                    layout_key="PA_BSND",
                    sparse_count=self.index_topk,
                    sparse_mode=3,
                    query_quant_mode=0,
                    key_quant_mode=0,
                ).squeeze(1)
            else:
                topk_indices = torch_npu.npu_lightning_indexer(
                    query=query,
                    key=past_key_states,
                    weights=weights,
                    actual_seq_lengths_query=actual_seq_lengths_q.to(torch.int32),
                    actual_seq_lengths_key=actual_seq_lengths_kv.to(k.device).to(
                        torch.int32
                    ),
                    block_table=block_table,
                    layout_query="TND",
                    layout_key="PA_BSND",
                    sparse_count=self.index_topk,
                    sparse_mode=3,
                )[0].squeeze(1)
            from sglang.srt.hardware_backend.npu.feature_evidence import record

            if use_quant_indexer:
                record(
                    "indexer-int8"
                    if pool.index_k_store_dtype == torch.int8
                    else "indexer-mxfp8"
                )
            if shard is not None:
                record("indexer-query-shard")
                topk_indices = shard.resolve(
                    topk_indices, num_query_tokens, forward_batch
                )
            else:
                forward_batch.npu_indexer_topk_is_local = False
            # Keep DSA top-k as [T, K]; NPU attention expands it when needed.
            return topk_indices

    def do_npu_cp_balance_indexer(
        self,
        q,
        past_key_states,
        indexer_weights,
        actual_seq_lengths_q,
        actual_seq_lengths_kv,
        block_table,
    ):
        # V2: Q is local (sharded); single indexer call against full KV.
        if indexer_weights is not None:
            indexer_weights = indexer_weights.contiguous().view(
                -1, indexer_weights.shape[-1]
            )
        topk_indices = torch_npu.npu_lightning_indexer(
            query=q,
            key=past_key_states,
            weights=indexer_weights,
            actual_seq_lengths_query=actual_seq_lengths_q.to(
                device=q.device, dtype=torch.int32
            ),
            actual_seq_lengths_key=actual_seq_lengths_kv.to(
                device=q.device, dtype=torch.int32
            ),
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
        )
        return topk_indices[0].squeeze(1)


def scattered_to_tp_attn_full(
    hidden_states: torch.Tensor,
    forward_batch,
) -> torch.Tensor:
    hidden_states, local_hidden_states = (
        torch.empty(
            (forward_batch.input_ids.shape[0], hidden_states.shape[1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ),
        hidden_states,
    )
    attn_tp_all_gather_into_tensor(hidden_states, local_hidden_states.contiguous())
    return hidden_states
