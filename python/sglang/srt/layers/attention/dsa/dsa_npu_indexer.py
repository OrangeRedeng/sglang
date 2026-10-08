from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.autotune import (
    effective_prefill_query_tokens,
    threshold_allows,
)
from sglang.srt.layers.cp.utils import cp_gather_full_sequence_states
from sglang.srt.layers.dp_attention import attn_tp_all_gather_into_tensor
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
from sglang.srt.runtime_context import get_parallel, get_stream
from sglang.srt.utils import is_npu, print_info_once

if is_npu():
    # Registers the CANN ops-transformer kernels under
    # torch.ops.cann_ops_transformer; without this import the
    # torch.ops namespace is empty and op calls raise AttributeError.
    import cann_ops_transformer  # noqa: F401
    import torch_npu

    from sglang.srt.hardware_backend.npu.utils import get_indexer_weight_stream

_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()
_shard_indexer_queries = envs.SGLANG_NPU_ENABLE_DSA_INDEXER_QUERY_SHARDING.get()


def _indexer_stream_mode():
    mode = envs.SGLANG_NPU_DSA_INDEXER_STREAM_MODE.get()
    if mode not in ("legacy", "inline", "resource"):
        raise ValueError(f"Unknown SGLANG_NPU_DSA_INDEXER_STREAM_MODE: {mode!r}")
    return mode


def _indexer_resource_reason(indexer, forward_batch, input_on_attn_tp_slices):
    mode = forward_batch.forward_mode
    if not mode.is_extend() or mode.is_draft_extend_v2() or mode.is_target_verify():
        return "requires prefill extend"
    if get_is_capture_mode():
        return "capture_mode=1"
    if is_in_breakable_cuda_graph():
        return "breakable_graph=1"
    if is_in_tc_piecewise_cuda_graph():
        return "piecewise_graph=1"
    if not indexer.rotary_emb.is_neox_style:
        if forward_batch.attn_cp_metadata is not None or get_parallel().dcp_enabled:
            return "non-NeoX CP/DCP paired RoPE is unsupported"
        if _use_ag_after_qlora and input_on_attn_tp_slices:
            return "non-NeoX sliced qlora gather is unsupported"
    return None


def _use_indexer_resource_stream(indexer, forward_batch, input_on_attn_tp_slices):
    return (
        _indexer_resource_reason(indexer, forward_batch, input_on_attn_tp_slices)
        is None
    )


def _record_indexer_tensors(stream, *tensors):
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensor.record_stream(stream)


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


def _quantize_npu_indexer_activation(x, hadamard, dst_type, *, tensor_name="activation"):
    # Hadamard-rotate x and MX-quantize its 128-dim vectors.
    # Returns (quantized, scale): quantized has x's shape in dst_type (fp8)
    # and scale holds one E8M0 byte per 32-element block, shaped
    # x.shape[:-1] + (d/64, 2) == x.shape[:-1] + (2, 2) — the descale layout
    # quant_lightning_indexer (v2) expects for quant_mode 3 (MXFP8).
    mode = envs.SGLANG_NPU_DSA_INDEXER_HADAMARD_MODE.get()
    if mode != "matmul":
        raise ValueError(
            "SGLANG_NPU_DSA_INDEXER_HADAMARD_MODE supports only matmul; "
            "FHT requires a separately accuracy-validated existing NPU primitive"
        )
    assert x.dtype == torch.bfloat16 and x.shape[-1] == 128
    if x.numel() == 0:
        return (
            torch.empty_like(x, dtype=dst_type),
            torch.zeros(x.shape[:-1] + (2, 2), dtype=torch.uint8, device=x.device).view(
                torch.float8_e8m0fnu
            ),
        )
    trace = envs.SGLANG_NPU_RESOURCE_SCHED_DIAGNOSTICS.get()
    with (
        torch.profiler.record_function(f"npu_dsa_indexer.{tensor_name}.hadamard.matmul")
        if trace
        else nullcontext()
    ):
        rotated = x @ hadamard
    with (
        torch.profiler.record_function(f"npu_dsa_indexer.{tensor_name}.mx_quant")
        if trace
        else nullcontext()
    ):
        quantized, scale = torch.ops.npu.npu_dynamic_mx_quant(
            rotated.reshape(-1, 128), dst_type=dst_type, axis=-1
        )
    # npu_dynamic_mx_quant may return the block scales as [N, 4] or [N, 2, 2];
    # normalize to the kernel's (d/64, 2) == (2, 2) layout.
    scale = scale.reshape(x.shape[:-1] + (4,)).view(x.shape[:-1] + (2, 2))
    if scale.dtype != torch.float8_e8m0fnu:
        scale = scale.view(torch.float8_e8m0fnu)

    return quantized.reshape(x.shape), scale


def _check_quant_lightning_indexer_constraints(pool) -> None:
    # quant_lightning_indexer (v2) PA_BBND layout: block_size == pool
    # page_size must lie in [16, 1024] and be a multiple of 16.
    page_size = pool.page_size
    assert 16 <= page_size <= 1024 and page_size % 16 == 0, (
        "quant_lightning_indexer (v2) PA_BBND layout requires the index-k "
        f"pool page_size in [16, 1024] and a multiple of 16, got {page_size}. "
        "Relaunch with page_size=64 (engine kwarg / --page-size 64)."
    )


def plan_indexer_query_shard(
    prefix_lens: List[int], extend_lens: List[int], tp_size: int, tp_rank: int
):
    """One attention-TP rank's share of an extend batch's indexer queries.

    Follows vLLM-Ascend DSA-CP (``_prepare_parallel_metadata`` in ``sfa_cp.py``):
    the batch's flat token rows are padded to a multiple of ``tp_size`` and rank
    r owns rows ``[r * rows, (r + 1) * rows)``. A request's local query count is
    its overlap with that range, and its key length ends at its last local
    token -- prefix plus local end -- which is what ``sparse_mode=3``'s
    right-down causal crop needs. A request with no token here gets key length 0.

    Returns ``(start, rows, num_real, cum_query_lens, key_lens)``; the first
    ``num_real`` of the rank's ``rows`` rows are real tokens.
    """
    total = sum(extend_lens)
    rows = -(-total // tp_size)
    start = tp_rank * rows
    end = start + rows
    num_real = max(0, min(end, total) - start)
    cum_query_lens, key_lens = [], []
    num_local = 0
    req_start = 0
    for prefix_len, extend_len in zip(prefix_lens, extend_lens):
        req_end = req_start + extend_len
        local_end = min(req_end, end)
        req_local = max(0, local_end - max(req_start, start))
        num_local += req_local
        cum_query_lens.append(num_local)
        key_lens.append(prefix_len + local_end - req_start if req_local else 0)
        req_start = req_end
    return start, rows, num_real, cum_query_lens, key_lens


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
    # ``(cu_seqlens_q, seqused_k, metadata)`` for the quantized indexer, or
    # None until this forward's first indexer layer plans it. The backend's
    # per-batch pre-plan cannot be used on a sharded call -- see the quant
    # branch of ``forward_npu`` -- and these lengths are the same for all 78
    # layers of one forward, so the plan is cached here rather than redone.
    quant_indexer_plan: Optional[tuple] = None

    def take(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_real == self.rows:
            return x[self.start : self.start + self.rows]
        # Padding rows lie past actual_seq_lengths_q[-1]. The operator leaves
        # uninitialized values in their top-k, which gather() slices off.
        out = x.new_zeros((self.rows, *x.shape[1:]))
        out[: self.num_real] = x[self.start : self.start + self.num_real]
        return out

    def gather(self, topk_indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """All-gather the per-rank top-k back to ``num_tokens`` rows.

        ``num_tokens`` is the width of the query tensor this call was sliced
        from: ``total`` padded up to a multiple of ``tp_size`` (see the gate in
        ``_get_indexer_query_shard``). ``rows * tp_size`` is that same padded
        width, so the gather buffer is already the right length and the slice
        only trims when the caller passed an unpadded tensor.
        """
        out = topk_indices.new_empty((self.rows * self.tp_size, topk_indices.shape[-1]))
        attn_tp_all_gather_into_tensor(out, topk_indices.contiguous())
        return out[:num_tokens]


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
    if parallel.attn_tp_rank == 0:
        # The switch is an env var and /server_info cannot show it, so this line
        # is the evidence that a run was sharded at all.
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

    Every attention-TP rank holds the same full batch and has already written
    the full index-K cache, so each rank can score only its own
    ``1/attn_tp_size`` of the queries and all-gather the top-k. The kernel's
    work per call drops by the TP size -- measured 15.97x at 1M context and
    TP 16 on A3 -- and the indexer is the part of long-context prefill that
    grows with n^2. Top-k sets match the unsharded call row for row; order
    within a row may differ, which sparse attention does not see.

    Planned once per forward. Every input to the decision is identical across
    the attention-TP group, so all ranks take the collective or none do.
    """
    if not hasattr(forward_batch, "npu_indexer_query_shard"):
        forward_batch.npu_indexer_query_shard = _build_indexer_query_shard(
            forward_batch
        )
    shard = forward_batch.npu_indexer_query_shard
    if shard is None:
        return None
    # SGLang pads the query width to a multiple of attn_tp_size, which is what
    # ``rows * tp_size`` is, so admit anything inside [total, padded]. Comparing
    # against total alone disables sharding for 15 token counts in 16.
    padded = shard.rows * shard.tp_size
    if not shard.total <= num_tokens <= padded:
        print_info_once(
            "DSA indexer query sharding is off for this forward: "
            f"num_tokens={num_tokens} is outside [{shard.total}, {padded}]"
        )
        return None
    return shard


@dataclass
class _PendingIndexerTopk:
    indices: torch.Tensor
    ready: object
    shard: Optional[_IndexerQueryShard]
    num_tokens: int

    def wait_and_gather(self) -> torch.Tensor:
        stream = torch.npu.current_stream()
        stream.wait_event(self.ready)
        self.indices.record_stream(stream)
        # HCCL retains the caller's collective order and stream.
        if self.shard is not None:
            return self.shard.gather(self.indices, self.num_tokens)
        return self.indices


class DSANPUIndexerMixin:
    def _neox_sin_cos(self, positions, forward_batch):
        if not hasattr(forward_batch, "npu_indexer_sin_cos_cache"):
            cos, sin = self.rotary_emb.cos_sin_cache[positions].chunk(2, dim=-1)
            cos = cos.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
            sin = sin.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
            forward_batch.npu_indexer_sin_cos_cache = (sin, cos)
        return forward_batch.npu_indexer_sin_cos_cache

    def _resource_projections_neox(
        self,
        x,
        q_lora,
        positions,
        forward_batch,
        input_on_attn_tp_slices,
        dynamic_scale,
    ):
        main = torch.npu.current_stream()
        vector = get_stream("npu_dsa_indexer_vector")
        bs = q_lora.shape[0]
        x = x.view(-1, self.hidden_size)
        sin, cos = self._neox_sin_cos(positions, forward_batch)
        operand = (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
        q_raw = self.wq_b(operand)[0]
        q_ready = main.record_event()
        with torch.npu.stream(vector):
            vector.wait_event(q_ready)
            _record_indexer_tensors(vector, q_raw, sin, cos)
            q_pe, q_nope = torch.split(
                q_raw.view(bs, self.n_heads, self.head_dim),
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )
            q_pe = torch_npu.npu_rotary_mul(
                q_pe.view(bs, self.n_heads, 1, self.rope_head_dim),
                cos,
                sin,
            ).view(bs, self.n_heads, self.rope_head_dim)
            q = torch.cat([q_pe, q_nope], dim=-1)

        weights_raw = self.weights_proj(x.float())[0]
        weights_ready = main.record_event()
        with torch.npu.stream(vector):
            vector.wait_event(weights_ready)
            _record_indexer_tensors(vector, weights_raw)
            weights = weights_raw.to(torch.bfloat16)

        k_raw = self.wk(x)[0]
        k_ready = main.record_event()
        gather_k = _use_ag_after_qlora and input_on_attn_tp_slices
        with torch.npu.stream(vector):
            vector.wait_event(k_ready)
            _record_indexer_tensors(vector, k_raw)
            for tensor in (*self.k_norm.parameters(), *self.k_norm.buffers()):
                tensor.record_stream(vector)
            k = self.k_norm(k_raw)
            if not gather_k:
                k = self._neox_k_rope(k, cos, sin, bs)
            ready = vector.record_event()
        main.wait_event(ready)
        _record_indexer_tensors(main, q, k, weights)
        if gather_k:
            # Retain k-gather before CP and weights-gather, on the caller stream.
            k = scattered_to_tp_attn_full(k, forward_batch)
            k = self._neox_k_rope(k, cos, sin, bs)
        return q, k, weights

    def _neox_k_rope(self, k, cos, sin, bs):
        k_pe, k_nope = torch.split(
            k,
            [self.rope_head_dim, self.head_dim - self.rope_head_dim],
            dim=-1,
        )
        k_pe = torch.ops.npu.npu_rotary_mul(
            k_pe.view(-1, 1, 1, self.rope_head_dim),
            cos,
            sin,
        ).view(bs, 1, self.rope_head_dim)
        return torch.cat([k_pe, k_nope.unsqueeze(1)], dim=-1)

    def _resource_projections(self, x, q_lora, positions, layer_id, dynamic_scale):
        """One projection queue; cast/split/norm/paired RoPE on one side queue.

        The established non-NeoX RoPE consumes q and k together. Keep that op
        intact rather than replacing it with a numerically different rotation.
        """
        main = torch.npu.current_stream()
        vector = get_stream("npu_dsa_indexer_vector")
        bs = q_lora.shape[0]
        x = x.view(-1, self.hidden_size)
        if layer_id == get_token_to_kv_pool().start_layer:
            self.rotary_emb.sin_cos_cache = self.rotary_emb.cos_sin_cache.index_select(
                0, positions
            )
        weights_raw = self.weights_proj(x.float())[0]
        weights_ready = main.record_event()
        with torch.npu.stream(vector):
            vector.wait_event(weights_ready)
            _record_indexer_tensors(vector, weights_raw)
            weights = weights_raw.to(torch.bfloat16)

        operand = (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
        q_raw = self.wq_b(operand)[0]
        q_ready = main.record_event()
        with torch.npu.stream(vector):
            vector.wait_event(q_ready)
            _record_indexer_tensors(vector, q_raw)
            q_pe, q_nope = torch.split(
                q_raw.view(bs, self.n_heads, self.head_dim),
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )

        k_raw = self.wk(x)[0]
        k_ready = main.record_event()
        with torch.npu.stream(vector):
            vector.wait_event(k_ready)
            _record_indexer_tensors(vector, k_raw, positions)
            for tensor in self.k_norm.parameters():
                tensor.record_stream(vector)
            for tensor in self.k_norm.buffers():
                tensor.record_stream(vector)
            for tensor in self.rotary_emb.buffers():
                tensor.record_stream(vector)
            _record_indexer_tensors(
                vector, getattr(self.rotary_emb, "sin_cos_cache", None)
            )
            k = self.k_norm(k_raw)
            k_pe, k_nope = torch.split(
                k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
            q = torch.cat([q_pe, q_nope], dim=-1)
            k = torch.cat([k_pe.squeeze(1), k_nope], dim=-1)
            _record_indexer_tensors(vector, q, k, weights)
            ready = vector.record_event()
        # Cache writes, Hadamard Cube work and all collectives stay on the caller.
        main.wait_event(ready)
        _record_indexer_tensors(main, q, k, weights)
        return q, k, weights

    def can_forward_npu_eager(
        self, forward_batch: ForwardBatch, input_on_attn_tp_slices: bool
    ) -> bool:
        return (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_target_verify()
            and forward_batch.attn_cp_metadata is None
            and not get_parallel().dcp_enabled
            and not (_use_ag_after_qlora and input_on_attn_tp_slices)
        )

    def forward_npu_eager(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        input_on_attn_tp_slices: bool = False,
    ) -> _PendingIndexerTopk:
        stream = get_stream("npu_dsa_indexer")
        input_ready = torch.npu.current_stream().record_event()
        metadata = get_attn_backend().forward_metadata
        with torch.npu.stream(stream):
            stream.wait_event(input_ready)
            tensors = [
                x,
                q_lora,
                positions,
                forward_batch.seq_lens,
                forward_batch.extend_seq_lens,
                forward_batch.out_cache_loc,
                metadata.seq_lens,
                metadata.seq_lens_cpu_int,
                metadata.block_tables,
                getattr(metadata, "quant_indexer_cu_seqlens_q", None),
                getattr(metadata, "quant_indexer_seqused_k", None),
                getattr(metadata, "quant_indexer_metadata", None),
                getattr(self.rotary_emb, "sin_cos_cache", None),
            ]
            tensors.extend(getattr(forward_batch, "npu_indexer_sin_cos_cache", ()))
            shard = getattr(forward_batch, "npu_indexer_query_shard", None)
            if shard is not None:
                tensors.extend(
                    (shard.actual_seq_lengths_q, shard.actual_seq_lengths_kv)
                )
                if shard.quant_indexer_plan is not None:
                    tensors.extend(shard.quant_indexer_plan)
            for tensor in tensors:
                if isinstance(tensor, torch.Tensor) and tensor.device.type == "npu":
                    tensor.record_stream(stream)
            indices, shard, num_tokens = self.forward_npu(
                x,
                q_lora,
                positions,
                forward_batch,
                layer_id,
                input_on_attn_tp_slices,
                defer_query_gather=True,
            )
            indices.record_stream(stream)
            ready = stream.record_event()
        return _PendingIndexerTopk(indices, ready, shard, num_tokens)

    def _plan_quant_lightning_indexer(
        self,
        cum_query_lens: torch.Tensor,
        key_lens: torch.Tensor,
        device,
    ):
        """``(cu_seqlens_q, seqused_k, metadata)`` for one quant indexer call.

        ``cum_query_lens`` is per-request *cumulative* query rows, as the v1
        operator's ``actual_seq_lengths_query`` was; the v2 operator instead
        wants it prefixed with a 0, and its last value is how many rows of
        ``query`` the kernel will read. ``key_lens`` is per-request and not
        cumulative. Both must describe the tensors this call actually passes.
        """
        cum_q = cum_query_lens.to(device=device, dtype=torch.int32)
        cu_seqlens_q = torch.cat([cum_q.new_zeros(1), cum_q])
        seqused_k = key_lens.to(device=device, dtype=torch.int32)
        metadata = torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata(
            self.n_heads,
            1,
            self.head_dim,
            self.index_topk,
            3,  # QUANT_MODE_MXFP8 (5 == QUANT_MODE_MXFP4)
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            batch_size=int(cum_q.numel()),
            max_seqlen_q=-1,
            max_seqlen_k=-1,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=1,
        )
        return cu_seqlens_q, seqused_k, metadata

    def forward_npu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        input_on_attn_tp_slices: bool = False,
        dynamic_scale: torch.Tensor = None,
        *,
        defer_query_gather: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[_IndexerQueryShard], int]]:
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
        stream_mode = _indexer_stream_mode()
        weight_multistream = (
            stream_mode == "legacy" and envs.SGLANG_NPU_USE_MULTI_STREAM.get()
        )
        resource_stream = stream_mode == "resource" and _use_indexer_resource_stream(
            self, forward_batch, input_on_attn_tp_slices
        )
        if stream_mode == "resource":
            reason = _indexer_resource_reason(
                self, forward_batch, input_on_attn_tp_slices
            )
            print_info_once(
                "DSA indexer resource stream is ACTIVE (serialized projections, "
                "Vector tails; CP/DCP collectives on caller)"
                if resource_stream
                else f"DSA indexer resource stream REQUESTED but is OFF: {reason}"
            )
        elif envs.SGLANG_NPU_RESOURCE_SCHED_DIAGNOSTICS.get():
            print_info_once(f"DSA indexer stream mode is {stream_mode}")

        if resource_stream and self.rotary_emb.is_neox_style:
            q, k, weights = self._resource_projections_neox(
                x,
                q_lora,
                positions,
                forward_batch,
                input_on_attn_tp_slices,
                dynamic_scale,
            )
        elif resource_stream:
            q, k, weights = self._resource_projections(
                x, q_lora, positions, layer_id, dynamic_scale
            )
        elif self.rotary_emb.is_neox_style:
            sin, cos = self._neox_sin_cos(positions, forward_batch)

            if self.alt_stream is not None:
                self.alt_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(self.alt_stream):
                    if defer_query_gather:
                        q_lora.record_stream(self.alt_stream)
                        cos.record_stream(self.alt_stream)
                        sin.record_stream(self.alt_stream)
                        if dynamic_scale is not None:
                            dynamic_scale.record_stream(self.alt_stream)
                    q_lora = (
                        (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
                    )
                    q = self.wq_b(q_lora)[
                        0
                    ]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
                    q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
                    q_pe, q_nope = torch.split(
                        q,
                        [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                        dim=-1,
                    )  # [bs, 64, 64 + 64]
                    q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
                    q_pe = torch_npu.npu_rotary_mul(q_pe, cos, sin).view(
                        bs, self.n_heads, self.rope_head_dim
                    )  # [bs, n, d]
                    q = torch.cat([q_pe, q_nope], dim=-1)
                    q.record_stream(self.alt_stream)
                    q_rope_event = self.alt_stream.record_event()
            else:
                q_lora = (
                    (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
                )
                q = self.wq_b(q_lora)[
                    0
                ]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
                q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
                q_pe, q_nope = torch.split(
                    q,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )  # [bs, 64, 64 + 64]
                q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
                q_pe = torch_npu.npu_rotary_mul(q_pe, cos, sin).view(
                    bs, self.n_heads, self.rope_head_dim
                )  # [bs, n, d]
                q = torch.cat([q_pe, q_nope], dim=-1)

            if weight_multistream:
                indexer_weight_stream = get_indexer_weight_stream()
                indexer_weight_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(indexer_weight_stream):
                    if defer_query_gather:
                        x.record_stream(indexer_weight_stream)
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
            if weight_multistream:
                indexer_weight_stream = get_indexer_weight_stream()
                indexer_weight_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(indexer_weight_stream):
                    if defer_query_gather:
                        x.record_stream(indexer_weight_stream)
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
            assert indexer_cache_loc.shape[0] == positions.shape[0], (
                "NPU DSA+DCP origin_out_cache_loc metadata has an incompatible "
                f"length: {indexer_cache_loc.shape[0]} != {positions.shape[0]}"
            )
        pool = get_token_to_kv_pool()
        use_quant_indexer = pool.index_k_scale_buffer is not None
        if use_quant_indexer:
            _check_quant_lightning_indexer_constraints(pool)
            k, k_scale = _quantize_npu_indexer_activation(
                k, pool.indexer_hadamard_128, pool.dtype, tensor_name="k"
            )
            pool.set_index_k_scale_buffer(layer_id, indexer_cache_loc, k_scale)
        pool.set_index_k_buffer(layer_id, indexer_cache_loc, k)
        if is_prefill:
            if (
                self.dsa_enable_prefill_cp
                and forward_batch.attn_cp_metadata is not None
            ):
                get_attn_backend().forward_metadata.actual_seq_lengths_q = (
                    forward_batch.attn_cp_metadata.actual_seq_q_prev_tensor,
                    forward_batch.attn_cp_metadata.actual_seq_q_next_tensor,
                )
                if sum(forward_batch.extend_prefix_lens_cpu) > 0:
                    total_kv_len_prev_tensor = (
                        forward_batch.attn_cp_metadata.kv_len_prev_tensor
                        + forward_batch.extend_prefix_lens.squeeze()
                    )
                    total_kv_len_next_tensor = (
                        forward_batch.attn_cp_metadata.kv_len_next_tensor
                        + forward_batch.extend_prefix_lens.squeeze()
                    )
                    get_attn_backend().forward_metadata.actual_seq_lengths_kv = (
                        total_kv_len_prev_tensor,
                        total_kv_len_next_tensor,
                    )
                else:
                    get_attn_backend().forward_metadata.actual_seq_lengths_kv = (
                        forward_batch.attn_cp_metadata.kv_len_prev_tensor,
                        forward_batch.attn_cp_metadata.kv_len_next_tensor,
                    )
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

        if (
            self.rotary_emb.is_neox_style
            and self.alt_stream is not None
            and not resource_stream
        ):
            torch.npu.current_stream().wait_event(q_rope_event)
            if defer_query_gather:
                q.record_stream(torch.npu.current_stream())
        if weight_multistream:
            torch.npu.current_stream().wait_event(weights_event)
            if defer_query_gather:
                weights.record_stream(torch.npu.current_stream())
        if _use_ag_after_qlora and input_on_attn_tp_slices:
            weights = scattered_to_tp_attn_full(weights, forward_batch)
        block_table = get_attn_backend().forward_metadata.block_tables
        if (
            is_prefill
            and self.dsa_enable_prefill_cp
            and forward_batch.attn_cp_metadata is not None
        ):
            block_table = block_table[: actual_seq_lengths_q[0].numel()]
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
            num_query_tokens = effective_prefill_query_tokens(forward_batch, query)
            shard = (
                _get_indexer_query_shard(forward_batch, num_query_tokens)
                if is_prefill
                and _shard_indexer_queries
                and threshold_allows(
                    "SGLANG_NPU_DSA_INDEXER_QUERY_SHARDING_MIN_TOKENS", num_query_tokens
                )
                else None
            )
            if shard is not None:
                query = shard.take(query)
                weights = shard.take(weights)
                actual_seq_lengths_q = shard.actual_seq_lengths_q
                actual_seq_lengths_kv = shard.actual_seq_lengths_kv

            if use_quant_indexer:
                query, query_scale = _quantize_npu_indexer_activation(
                    query,
                    pool.indexer_hadamard_128,
                    pool.dtype,
                    tensor_name="q",
                )
                # quant_lightning_indexer (v2) contract, quant_mode 3 (MXFP8):
                #  - layout_q TND: q (q_t, q_n, d) fp8, cu_seqlens_q (b+1,) int32
                #    required (first value 0, last value q_t)
                #  - layout_k PA_BBND: k (block_num, block_size, k_n, d) fp8 with
                #    block_table (b, max_blocks) and seqused_k (b,) both required
                #  - descales are E8M0: q (q_t, q_n, d/64, 2),
                #    k (block_num, block_size, k_n, d/64, 2)
                #  - w is float32 (q_t, q_n)
                # The metadata op (task list / load balancing) is pre-planned
                # once per batch by AscendAttnBackend.init_forward_metadata;
                # fall back to computing it inline when the backend did not
                # pre-plan it (cuda-graph capture, other backends, CP, spec
                # draft model), and ALWAYS re-plan when this call was sharded.
                #
                # The pre-plan describes the whole batch. A sharded call was
                # handed 1/attn_tp_size of the query rows, so the batch-wide
                # cu_seqlens_q ends at the batch's token total and the operator
                # would walk attn_tp_size times past the end of `query`,
                # `weights` and `query_scale`. Nothing on the host catches it:
                # the lengths are device tensors, so the tiling cannot check
                # them against the query shape, and the kernel reports the
                # out-of-bounds read as an AICore trap from inside
                # quant_lightning_indexer. It is a write too: v2's infershape
                # sizes sparse_indices from queryShape.GetDim(0), not from
                # cu_seqlens_q.
                #
                # That same rule is what lets shard.gather() work here: the
                # output has one row per row of `query`, so a sharded call
                # returns shard.rows rows and the all-gather widths line up.
                # Rows past cu_seqlens_q[-1] are untouched, and gather() slices
                # them off.
                _fm = get_attn_backend().forward_metadata
                if shard is None:
                    cu_seqlens_q = getattr(_fm, "quant_indexer_cu_seqlens_q", None)
                    seqused_k = getattr(_fm, "quant_indexer_seqused_k", None)
                    metadata = getattr(_fm, "quant_indexer_metadata", None)
                    if metadata is None:
                        cu_seqlens_q, seqused_k, metadata = (
                            self._plan_quant_lightning_indexer(
                                actual_seq_lengths_q, actual_seq_lengths_kv, k.device
                            )
                        )
                else:
                    # Planned on this forward's first indexer layer and reused
                    # by the rest, which is what the backend's pre-plan buys.
                    if shard.quant_indexer_plan is None:
                        shard.quant_indexer_plan = self._plan_quant_lightning_indexer(
                            shard.actual_seq_lengths_q,
                            shard.actual_seq_lengths_kv,
                            k.device,
                        )
                    cu_seqlens_q, seqused_k, metadata = shard.quant_indexer_plan
                block_table = block_table.to(torch.int32)
                topk_indices, _ = (
                    torch.ops.cann_ops_transformer.quant_lightning_indexer(
                        query,
                        past_key_states,
                        weights.to(torch.float32),
                        query_scale,
                        pool.get_index_k_scale_buffer(layer_id),
                        self.index_topk,
                        3,  # QUANT_MODE_MXFP8   5, # QUANT_MODE_MXFP4,
                        cu_seqlens_q=cu_seqlens_q,
                        seqused_k=seqused_k,
                        block_table=block_table,
                        metadata=metadata,
                        max_seqlen_q=-1,
                        layout_q="TND",
                        layout_k="PA_BBND",
                        mask_mode=3,
                        cmp_ratio=1,
                    )
                )
                topk_indices = topk_indices.squeeze(1)
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
            if defer_query_gather:
                return topk_indices, shard, num_query_tokens
            if shard is not None:
                topk_indices = shard.gather(topk_indices, num_query_tokens)
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
        q_prev, q_next = torch.split(q, (q.size(0) + 1) // 2, dim=0)
        weights_prev, weights_next = None, None
        if indexer_weights is not None:
            weights_prev, weights_next = torch.split(
                indexer_weights, (indexer_weights.size(0) + 1) // 2, dim=0
            )
            weights_prev = weights_prev.contiguous().view(-1, weights_prev.shape[-1])
            weights_next = weights_next.contiguous().view(-1, weights_next.shape[-1])

        actual_seq_lengths_q_prev, actual_seq_lengths_q_next = actual_seq_lengths_q
        actual_seq_lengths_kv_prev, actual_seq_lengths_kv_next = actual_seq_lengths_kv

        topk_indices_prev = torch_npu.npu_lightning_indexer(
            query=q_prev,
            key=past_key_states,
            weights=weights_prev,
            actual_seq_lengths_query=actual_seq_lengths_q_prev.to(
                device=q.device, dtype=torch.int32
            ),
            actual_seq_lengths_key=actual_seq_lengths_kv_prev.to(
                device=q.device, dtype=torch.int32
            ),
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
        )
        topk_indices_next = torch_npu.npu_lightning_indexer(
            query=q_next,
            key=past_key_states,
            weights=weights_next,
            actual_seq_lengths_query=actual_seq_lengths_q_next.to(
                device=q.device, dtype=torch.int32
            ),
            actual_seq_lengths_key=actual_seq_lengths_kv_next.to(
                device=q.device, dtype=torch.int32
            ),
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
        )
        return torch.cat([topk_indices_prev[0], topk_indices_next[0]], dim=0).squeeze(1)


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
