# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Pure index math for decode context parallel (DCP): per-rank lengths,
the owner-rule local-index filter, and the NPU extend-gather plan."""

import logging
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import print_info_once

logger = logging.getLogger(__name__)


def dcp_owner_count(length: int, dcp_size: int, dcp_rank: int, interleave: int) -> int:
    """Scalar ``get_dcp_lens``: how many of ``[0, length)`` this rank owns."""
    cycle = dcp_size * interleave
    full, rem = divmod(length, cycle)
    return full * interleave + min(max(rem - dcp_rank * interleave, 0), interleave)


def dcp_local_row(loc: torch.Tensor, dcp_size: int, interleave: int) -> torch.Tensor:
    """The row ``loc`` occupies in its owner's pool."""
    if interleave == 1:
        return loc // dcp_size
    return (loc // (interleave * dcp_size)) * interleave + loc % interleave


def get_dcp_lens(
    lens: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    start: torch.Tensor | None = None,
    interleave_size: int = 1,
) -> torch.Tensor:
    """Return the KV length owned by one DCP rank.

    Consecutive runs of ``interleave_size`` positions rotate across ranks.  The
    default value preserves the token-interleaved layout used by the generic
    CUDA/ROCm DCP paths; NPU DSA passes its physical KV page size.
    """
    if dcp_size == 1:
        return lens
    if interleave_size < 1:
        raise ValueError(f"interleave_size must be positive, got {interleave_size}")

    cycle_size = dcp_size * interleave_size

    def _count_before(end: torch.Tensor) -> torch.Tensor:
        full_cycles = end // cycle_size
        remainder = end % cycle_size
        rank_remainder = torch.clamp(
            remainder - dcp_rank * interleave_size,
            min=0,
            max=interleave_size,
        )
        return full_cycles * interleave_size + rank_remainder

    if start is None:
        return _count_before(lens)
    return _count_before(start + lens) - _count_before(start)


def localize_dcp_indices(
    indices: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int = 1,
) -> torch.Tensor:
    """Map global DCP indices to one rank, using ``-1`` for non-local rows."""
    if dcp_size == 1:
        return indices
    if interleave_size < 1:
        raise ValueError(f"interleave_size must be positive, got {interleave_size}")

    interleave_block = torch.div(indices, interleave_size, rounding_mode="floor")
    is_local = (indices >= 0) & (interleave_block % dcp_size == dcp_rank)
    local_block = torch.div(interleave_block, dcp_size, rounding_mode="floor")
    local_indices = local_block * interleave_size + indices % interleave_size
    return torch.where(is_local, local_indices, torch.full_like(indices, -1))


def maybe_dcp_kernel_indices(
    indices: torch.Tensor, dcp_size: int, dcp_rank: int
) -> torch.Tensor:
    """Widened logical slots -> this rank's physical rows.

    Owner rule: slot % dcp_size == dcp_rank, row = slot // dcp_size. The run
    starts page-aligned, so a strided view selects the owned slots without a mask.
    """
    if dcp_size == 1:
        return indices
    return indices[dcp_rank::dcp_size] // dcp_size


def filter_dcp_local_kv_indices(kv_indices: torch.Tensor):
    """Keep this rank's share of a read-index tensor, still WIDENED.

    Selection only; the caller collapses via translate_dcp_read_ids.
    """
    parallel = get_parallel()
    if parallel.dcp_enabled:
        kv_indices = kv_indices[kv_indices % parallel.dcp_size == parallel.dcp_rank]
    return kv_indices


def filter_dcp_local_chunk_kv_indices(
    kv_indices: torch.Tensor,
    chunk_starts_cpu: torch.Tensor,
    chunk_seq_lens_cpu: torch.Tensor,
) -> torch.Tensor:
    parallel = get_parallel()
    if not parallel.dcp_enabled:
        return kv_indices

    dcp_size = parallel.dcp_size
    parts = []
    offset = 0
    for start, length in zip(chunk_starts_cpu.tolist(), chunk_seq_lens_cpu.tolist()):
        first = (parallel.dcp_rank - start) % dcp_size
        parts.append(kv_indices[offset + first : offset + length : dcp_size])
        offset += length
    return torch.cat(parts)


def remap_dcp_sparse_indices(
    topk_indices: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int = 1,
) -> torch.Tensor:
    """Map global sparse token indices to one rank's compact DCP KV layout.

    Keep indices owned by this rank, convert them to local KV positions, and
    stably move them before ``-1`` padding while preserving their score order.
    """
    if dcp_size == 1:
        return topk_indices
    if interleave_size < 1:
        raise ValueError(f"interleave_size must be positive, got {interleave_size}")

    # Float32 is faster than integer division/remainder on Ascend for this hot
    # path.  Keep the math equivalent to localize_dcp_indices above.
    topk_indices_fp32 = topk_indices.to(torch.float32)
    interleave_blocks = torch.floor(topk_indices_fp32 / interleave_size)
    local_owner_mask = (topk_indices_fp32 >= 0) & (
        torch.remainder(interleave_blocks, dcp_size) == dcp_rank
    )
    local_offsets = torch.remainder(topk_indices_fp32, interleave_size)
    local_indices = (
        torch.floor(interleave_blocks / dcp_size) * interleave_size + local_offsets
    )
    remapped_indices = torch.where(
        local_owner_mask,
        local_indices,
        torch.full_like(topk_indices_fp32, -1.0),
    ).to(topk_indices.dtype)

    # Move valid entries before padding without changing their top-k order.
    topk_count = topk_indices.shape[-1]
    original_order = torch.arange(
        topk_count, dtype=torch.float32, device=topk_indices.device
    ).expand_as(topk_indices_fp32)
    pack_keys = original_order + (~local_owner_mask).to(torch.float32) * topk_count
    pack_order = torch.argsort(pack_keys, dim=-1).to(torch.int64)
    return torch.gather(remapped_indices, dim=-1, index=pack_order)


def get_dcp_chain_spec_lens(
    total_kv_lens: torch.Tensor,
    tokens_per_req: int,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int = 1,
) -> torch.Tensor:
    """Return request-major local KV frontiers for a speculative chain."""
    if tokens_per_req < 1:
        raise ValueError(f"tokens_per_req must be >= 1, got {tokens_per_req}")
    total_kv_lens = total_kv_lens.int()
    steps = torch.arange(
        1, tokens_per_req + 1, dtype=total_kv_lens.dtype, device=total_kv_lens.device
    )
    global_query_lens = total_kv_lens.unsqueeze(1) - tokens_per_req + steps
    global_query_lens = torch.where(
        total_kv_lens.unsqueeze(1) >= tokens_per_req,
        global_query_lens,
        torch.zeros_like(global_query_lens),
    )
    return get_dcp_lens(
        global_query_lens.reshape(-1),
        dcp_size,
        dcp_rank,
        interleave_size=interleave_size,
    ).int()


def update_local_kv_lens_for_dcp(kv_len_arr):
    """In-place per-rank KV length: the start=0 case of get_dcp_lens.

    floor((len - rank - 1) / N) + 1  ==  len // N + (rank < len % N)  for len >= 0
    (bit-identical; see test/registered/cp/test_dcp_layout_unit.py). Kept as an
    in-place mutation because callers (plan_dcp_decode_metadata, the FlashInfer-MLA
    cuda-graph replay path) rely on it.
    """
    parallel = get_parallel()
    if not parallel.dcp_enabled:
        return
    kv_len_arr.copy_(get_dcp_lens(kv_len_arr, parallel.dcp_size, parallel.dcp_rank))


def dcp_owned_prefix_positions(
    prefix_lens: Sequence[int], dcp_size: int, dcp_rank: int, interleave: int
) -> torch.Tensor:
    """Offsets into the concatenated prefixes of the positions this rank owns.

    Per request, in position order: the order ``plan_dcp_extend_gather`` expects
    its send in. Built on the host from the lengths, so no ``nonzero()`` sync,
    and correct for a prefix that ends mid-cycle (a chunked request's partial
    page), where a fixed stride over the concatenation would drift.
    """
    cycle = dcp_size * interleave
    run = torch.arange(interleave, dtype=torch.int64)
    parts = []
    offset = 0
    for prefix_len in prefix_lens:
        prefix_len = int(prefix_len)
        if dcp_rank * interleave >= prefix_len:
            offset += prefix_len
            continue
        starts = torch.arange(
            dcp_rank * interleave, prefix_len, cycle, dtype=torch.int64
        )
        positions = (starts[:, None] + run).reshape(-1)
        parts.append(positions[positions < prefix_len] + offset)
        offset += prefix_len
    if not parts:
        return torch.empty(0, dtype=torch.int64)
    return torch.cat(parts)


def dcp_local_prefix_rows(
    prefix_slots: torch.Tensor,
    prefix_lens: Sequence[int],
    dcp_size: int,
    dcp_rank: int,
    interleave: int,
) -> torch.Tensor:
    """This rank's pool rows for its share of the concatenated prefix slots.

    Selected by position, mapped by slot: the two agree only because allocator
    pages are widened to ``dcp_size * interleave`` slots, and the row must be
    the one ``localize_dcp_indices`` wrote. Ordered for ``plan_dcp_extend_gather``.
    """
    owned = dcp_owned_prefix_positions(prefix_lens, dcp_size, dcp_rank, interleave)
    return dcp_local_row(
        prefix_slots[owned.to(prefix_slots.device)], dcp_size, interleave
    )


# Above this many new tokens a DSA extend gathers the KV whatever the bytes say.
# The query gather's fp32 ``[out, lse]`` exchange grows with the new tokens and
# is held several times over: a 16k-token batch of three 1M-prefix requests ran
# out of memory there on A3, while 4,096-token chunks run on main. Memory-driven,
# so a deployment can move it.
DCP_EXTEND_QUERY_GATHER_MAX_TOKENS = (
    envs.SGLANG_NPU_DCP_EXTEND_QUERY_GATHER_MAX_TOKENS.get()
)


def dcp_extend_gathers_kv(
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    kv_elem_size: int,
) -> bool:
    """Should this DSA extend gather the prefix KV rather than the query?

    Gathering the query costs, per query token, every head's query plus the
    fp32 ``[out, lse]`` merge; gathering the KV costs one latent row per prefix
    token, whatever the head count. Both scale by ``(dcp - 1) / dcp``, so it
    cancels. For GLM-5.2 (64 heads, 512 + 64 bf16) the crossover is a tail of
    prefix / 178. Ties go to the KV gather, so an extend with no prefix -- which
    gathers nothing -- takes it. A batch counts each request's prefix, shared or
    not, because the KV gather sends each one.

    Bytes are not the only cost: past ``DCP_EXTEND_QUERY_GATHER_MAX_TOKENS``
    the query gather's memory decides, and the KV gather is taken.
    """
    if sum(extend_lens) > DCP_EXTEND_QUERY_GATHER_MAX_TOKENS:
        return True
    row = (kv_lora_rank + qk_rope_head_dim) * kv_elem_size
    per_query_token = num_heads * (row + (kv_lora_rank + 1) * 4)
    return sum(prefix_lens) * row <= sum(extend_lens) * per_query_token


# Reusable device buffers for the extend gather, keyed by purpose, dtype, device
# and row shape. See dcp_extend_gather_buffer.
_dcp_extend_gather_buffers: Dict[Tuple, torch.Tensor] = {}


def dcp_extend_gather_buffer(name: str, ref: torch.Tensor, rows: int) -> torch.Tensor:
    """A reusable ``rows``-row buffer shaped and typed like ``ref``. Grow-only.

    Allocated per layer, the peak moved with whatever the allocator held when a
    layer asked for another ~1.1 GiB. Two callers sharing ``name`` share the
    buffer and must not hold slices across each other's calls.
    """
    key = (name, ref.dtype, ref.device, tuple(ref.shape[1:]))
    buf = _dcp_extend_gather_buffers.get(key)
    if buf is None or buf.shape[0] < rows:
        # Drop every reference first, or the peak is briefly both buffers.
        _dcp_extend_gather_buffers.pop(key, None)
        buf = None
        buf = torch.empty((rows, *ref.shape[1:]), dtype=ref.dtype, device=ref.device)
        _dcp_extend_gather_buffers[key] = buf
        logger.info(
            "DCP extend gather buffer %r reserved: %d rows, %.3f GiB",
            name,
            rows,
            buf.numel() * buf.element_size() / (1 << 30),
        )
    return buf[:rows]


class DcpExtendGatherPiece(NamedTuple):
    """One collective of the extend gather. The all-gather lays the sends out
    rank-major, this chunk's own KV follows, and ``index`` maps output rows
    ``[out_start, out_end)`` onto that scratch."""

    send_start: int
    send_end: int
    extend_start: int
    extend_end: int
    out_start: int
    out_end: int
    index: torch.Tensor
    scratch_rows: int


class DcpExtendGatherPlan(NamedTuple):
    """How one rank gathers a batch's prefix KV at extend. Sends are padded
    to ``padded_lens`` so every rank's is ``send_rows`` long; the output is each
    request's prefix then its extend, in order, tiled by ``pieces``."""

    local_lens: List[int]
    padded_lens: List[int]
    send_rows: int
    scratch_rows: int
    pieces: List[DcpExtendGatherPiece]


def plan_dcp_extend_gather(
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    dcp_size: int,
    dcp_rank: int,
    max_piece_gather_rows: int,
    interleave_size: int = 1,
) -> DcpExtendGatherPlan:
    """Plan the extend-time gather of the prefix KV into position order.

    Position p lives on rank ``(p // B) % dcp_size`` at local row
    ``(p // (B * dcp_size)) * B + p % B``, ``B = interleave_size``. Local shards
    are concatenated per request, as ``dcp_local_prefix_kv_indices`` lists them.

    Pieces gather at most ``max_piece_gather_rows`` rows each, and cut on whole
    ``B``-blocks: that keeps local rows ``[row, row + take)`` covering global
    positions ``[row * dcp_size, (row + take) * dcp_size)`` on every rank. The
    pieces are identical on every rank; only ``local_lens`` is not.
    """
    prefix_lens = [int(p) for p in prefix_lens]
    extend_lens = [int(e) for e in extend_lens]
    b = interleave_size
    local_lens = [dcp_owner_count(p, dcp_size, dcp_rank, b) for p in prefix_lens]
    # Rank 0 holds the most; every rank pads to it.
    padded_lens = [dcp_owner_count(p, dcp_size, 0, b) for p in prefix_lens]
    # Rounded down to a whole block, so a cut never splits one rank's run.
    piece_rows = max(b, (max_piece_gather_rows // dcp_size) // b * b)

    pieces = []
    parts = []
    send_start = extend_start = out_start = 0
    send_end = extend_end = out_end = 0
    send_offset = 0
    for prefix_len, extend_len, padded_len in zip(
        prefix_lens, extend_lens, padded_lens
    ):
        row = 0
        while row < padded_len:
            room = piece_rows - (send_end - send_start)
            # A cut inside a request lands on a whole block of it: the previous
            # request's rows need not fill whole blocks of the piece.
            take = padded_len - row if padded_len - row <= room else room // b * b
            if take == 0:
                pieces.append(
                    _dcp_extend_gather_piece(
                        parts,
                        dcp_size,
                        send_start,
                        send_end,
                        extend_start,
                        extend_end,
                        out_start,
                        out_end,
                        b,
                    )
                )
                parts = []
                send_start, extend_start, out_start = send_end, extend_end, out_end
                continue
            positions = torch.arange(
                row * dcp_size,
                min((row + take) * dcp_size, prefix_len),
                dtype=torch.int64,
            )
            parts.append(("prefix", positions, send_offset))
            row += take
            send_end += take
            out_end += positions.numel()
        if extend_len:
            parts.append(("extend", extend_end, extend_len))
            extend_end += extend_len
            out_end += extend_len
        send_offset += padded_len
    if parts:
        pieces.append(
            _dcp_extend_gather_piece(
                parts,
                dcp_size,
                send_start,
                send_end,
                extend_start,
                extend_end,
                out_start,
                out_end,
                b,
            )
        )
    return DcpExtendGatherPlan(
        local_lens=local_lens,
        padded_lens=padded_lens,
        send_rows=sum(padded_lens),
        scratch_rows=max((p.scratch_rows for p in pieces), default=0),
        pieces=pieces,
    )


def _dcp_extend_gather_piece(
    parts,
    dcp_size: int,
    send_start: int,
    send_end: int,
    extend_start: int,
    extend_end: int,
    out_start: int,
    out_end: int,
    interleave_size: int = 1,
) -> DcpExtendGatherPiece:
    """Close one piece: index its output rows into its rank-major scratch."""
    send_len = send_end - send_start
    gathered_rows = send_len * dcp_size
    b = interleave_size
    index = []
    for kind, first, second in parts:
        if kind == "prefix":
            positions, request_send_offset = first, second
            owner = (positions // b) % dcp_size
            local_row = dcp_local_row(positions, dcp_size, b)
            index.append(
                owner * send_len + (request_send_offset - send_start) + local_row
            )
        else:
            start = gathered_rows + first - extend_start
            index.append(torch.arange(start, start + second, dtype=torch.int64))
    return DcpExtendGatherPiece(
        send_start=send_start,
        send_end=send_end,
        extend_start=extend_start,
        extend_end=extend_end,
        out_start=out_start,
        out_end=out_end,
        index=torch.cat(index),
        scratch_rows=gathered_rows + extend_end - extend_start,
    )


def dcp_crop_free_extend(forward_batch, index_topk: Optional[int]) -> bool:
    """May this extend forward drop the operator's causal crop (sparse_mode 0)?

    Only if every prefix reaches ``index_topk``: below it the top-k takes every
    key, so the crop is what keeps the result causal. Cached per forward, and
    the FIRST caller's ``index_topk`` wins -- the cache does not key on it.
    """
    cached = getattr(forward_batch, "npu_dcp_crop_free", None)
    if cached is not None:
        return cached
    prefix_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    ok = bool(
        index_topk is not None
        and prefix_lens
        and all(p + 1 >= index_topk for p in prefix_lens)
    )
    if not ok and index_topk is not None and prefix_lens:
        print_info_once(
            f"DCP extend keeps the operator's causal crop: a request here has a "
            f"prefix under index_topk={index_topk} (shortest {min(prefix_lens)})"
        )
    forward_batch.npu_dcp_crop_free = ok
    return ok


def dcp_local_kv_block_table(loc_rows: torch.Tensor, page_size: int) -> torch.Tensor:
    """The rank-local KV page table, from the same ``req_to_token`` slice.

    Under DCP the backend needs two page tables over one allocation: the
    replicated indexer pages the full virtual span, the sharded latent KV pages
    this rank's rows. The backend's existing expression already produces the
    indexer's; this produces the other.

    The allocator pages virtual space at ``page_size * dcp_size``, and a rank
    owns the positions with ``v % dcp_size == rank``, which map under
    ``// dcp_size`` onto the physical page of the same index. So a rank-local
    page is exactly an allocator page, and reading every
    ``page_size * dcp_size``-th location recovers its physical id. At
    ``dcp_size == 1`` this is the existing expression unchanged.
    """
    stride = page_size * get_parallel().attn_dcp_size
    return loc_rows[:, ::stride] // stride


def remap_dcp_local_topk_indices(
    topk_indices: torch.Tensor, invalid: int = -1
) -> torch.Tensor:
    """Global sparse top-k positions -> rank-local KV coordinates, shape preserved.

    The sparse indexer selects top-k over the *replicated* index-K buffer, so it
    returns positions in the full sequence's coordinate system. Sparse attention
    reads the *sharded* latent KV and needs rank-local ones. This is the
    translation between the two.

    ``filter_dcp_local_kv_indices`` cannot be reused despite the identical
    owner rule: it selects, and a top-k row is ``[..., K]`` with K fixed, so
    selecting would make it ragged. Non-owned entries are marked ``invalid``
    and the survivors stably compacted to the front, preserving shape and
    top-k order, as vLLM-Ascend does (``sfa_cp.py:1023-1045``).

    The ``>= 0`` guard is load-bearing: the indexer pads short rows with -1,
    and ``-1 % dcp_size`` is ``dcp_size - 1``, so on the highest rank every pad
    would otherwise look owned.
    """
    parallel = get_parallel()
    dcp_size = parallel.attn_dcp_size
    if dcp_size == 1:
        return topk_indices

    # Checked before any work, because it is a precondition on the compaction
    # below rather than a property of the result -- see the sort-key note.
    k = topk_indices.shape[-1]
    assert 2 * k <= 1 << 24, (
        f"top-k width {k} exceeds the exactly-representable float32 sort-key "
        "range; keys would collide and the compaction would stop being a "
        "permutation. Sort integer keys here instead, and pay AiCpu on Ascend."
    )

    owned = (topk_indices >= 0) & (topk_indices % dcp_size == parallel.attn_dcp_rank)
    local = torch.where(
        owned,
        topk_indices // dcp_size,
        torch.full_like(topk_indices, invalid),
    )

    # Offsetting a non-owned key by K keeps every key distinct, so the sort is a
    # unique permutation. float32 because Ascend argsorts int32/int64 on AiCpu;
    # exact below 2**24, which is asserted above.
    order = torch.arange(k, device=topk_indices.device, dtype=torch.float32)
    keys = order + (~owned).to(torch.float32) * k
    return torch.gather(local, -1, torch.argsort(keys, dim=-1))


def plan_dcp_owner_write(
    loc: torch.Tensor, dcp_size: int, dcp_rank: int, interleave_size: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rows of ``loc`` this rank owns, and the physical rows they land on.

    The owner rule is CUDA's, from the Triton kernel at
    ``kernels/ops/kvcache/mla_buffer.py:42``::

        is_valid = loc % DCP_WORLD_SIZE == DCP_RANK
        loc      = loc // DCP_WORLD_SIZE

    Returns ``(owned_idx, dest)``: positions into ``loc`` that this rank owns,
    and their destinations in its own pool. Together across the group the
    ``owned_idx`` partition every row of ``loc`` exactly once, so a write
    filtered through this covers the same rows as an unfiltered one.

    ``torch.nonzero`` makes the output shape data-dependent, which costs a
    stream synchronisation and cannot run inside a captured stream -- so the
    NPU decode path aims non-owned rows at a padding row instead. Extend is not
    captured and shares one write location across layers, so there the filter
    is computed once per forward.
    """
    local = localize_dcp_indices(loc, dcp_size, dcp_rank, interleave_size)
    owned_idx = torch.nonzero(local >= 0).squeeze(1)
    return owned_idx, local.index_select(0, owned_idx)


class DcpPackedReadPlan(NamedTuple):
    """Where each KV position of a ONE-REQUEST extend lives in the packed buffer.

    ``plan_dcp_extend_gather`` permutes the all-gather's rank-major output back
    into position order so the sparse operator can read it. That permutation is
    a context-sized ``index_select`` -- 784 launches x 0.275 ms = 216 ms a
    forward -- plus the position-ordered output buffer it writes into, ~1.04 GiB
    a rank at a 972k context. Neither is needed: the operator reads the KV
    through ``sparse_indices``, so it does not care what order the rows are in,
    only that the indices point at the right ones. Remapping a ``[tokens, k]``
    index tensor is arithmetic on a few million integers; permuting the KV is a
    gigabyte of traffic. Upstream reached the same conclusion for CUDA in
    ``#39215``'s ``create_packed_dsa_dcp_kv_indices``.

    The packed buffer is exactly what the collective produces, in one piece::

        [ rank 0's send rows | rank 1's | ... | rank C-1's ][ this chunk's KV ]
          <--- send_rows ---> each                            <-- extend_len -->

    Under the owner rule ``pos % C == rank`` a request's position ``p`` sits in
    rank ``p % C``'s send at local row ``p // C``, so::

        p <  prefix_len   ->  (p % C) * send_rows + p // C
        p >= prefix_len   ->  gathered_rows + (p - prefix_len)

    ``send_rows`` is the padded local length, so ranks below ``prefix_len % C``
    hold one row more than the rest and the spare rows are never addressed.

    **ONE REQUEST ONLY, and the reason is the same one DSA-CP has.** With two
    requests the all-gather still lays out rank-major over the whole send, so a
    request's rows land in C separate blocks and the operator's cumulative
    ``actual_seq_lengths_kv`` -- which needs each request contiguous -- cannot
    describe them. Multi-request extends keep the permuting path. That covers
    87% of the tokens in the AISBench run as measured on 2026-09-18.
    """

    prefix_len: int
    extend_len: int
    dcp_size: int
    send_rows: int
    gathered_rows: int
    rows: int


def plan_dcp_packed_read(
    prefix_len: int, extend_len: int, dcp_size: int
) -> DcpPackedReadPlan:
    """Sizes of the packed rank-major buffer for a one-request extend."""
    send_rows = -(-int(prefix_len) // dcp_size)
    gathered_rows = send_rows * dcp_size
    return DcpPackedReadPlan(
        prefix_len=int(prefix_len),
        extend_len=int(extend_len),
        dcp_size=dcp_size,
        send_rows=send_rows,
        gathered_rows=gathered_rows,
        rows=gathered_rows + int(extend_len),
    )


def packed_row_of(pos: int, plan: DcpPackedReadPlan) -> int:
    """Reference implementation of the remap, one position at a time.

    Exists to be the thing ``remap_topk_to_packed`` is checked against on CPU:
    the tensor version has to fuse the branch into ``where`` and do floor
    division on a signed tensor, and those are the two places this kind of
    arithmetic goes wrong.
    """
    if pos < 0:
        return pos
    if pos < plan.prefix_len:
        return (pos % plan.dcp_size) * plan.send_rows + pos // plan.dcp_size
    return plan.gathered_rows + (pos - plan.prefix_len)


def remap_topk_to_packed(
    topk_indices: torch.Tensor, plan: DcpPackedReadPlan
) -> torch.Tensor:
    """Top-k positions within a request -> rows of the packed buffer.

    Shape and dtype are preserved, and so are the negatives: ``_pad_topk_indices``
    fills unused slots with -1 and the operator reads that tail as "no more
    entries", so mapping them into a real row would silently add keys. Every
    branch here is elementwise, which is what keeps this ~ a millisecond against
    the 216 ms it replaces.
    """
    c = plan.dcp_size
    # floor division, not trunc: torch's // on a signed tensor rounds toward
    # zero for negatives, and the -1 sentinels must come through untouched.
    local_row = torch.div(topk_indices, c, rounding_mode="floor")
    prefix_row = (topk_indices % c) * plan.send_rows + local_row
    extend_row = plan.gathered_rows + (topk_indices - plan.prefix_len)
    packed = torch.where(topk_indices < plan.prefix_len, prefix_row, extend_row)
    return torch.where(topk_indices < 0, topk_indices, packed)


_enable_dcp_packed_read = envs.SGLANG_NPU_ENABLE_DCP_PACKED_READ.get()


class _PackedMissing:
    """Distinguishes "not resolved yet" from "resolved, and it is None"."""


_PACKED_MISSING = _PackedMissing()


def dcp_packed_read_enabled() -> bool:
    """Whether the packed rank-major read is on for this process."""
    return _enable_dcp_packed_read


def dcp_packed_causal_crop_is_dead(prefix_len: int, index_topk: int) -> bool:
    """Whether the operator's causal crop can be dropped for this chunk.

    THE PACKED READ NEEDS ``sparse_mode = 0``: the crop aligns query i to key
    ``K - Q + i``, which means nothing once the keys are in rank-major order.
    The first version of this claimed the top-k is causal by construction, so
    the crop was redundant. That is false, and a stage-A run measured it --
    scattered logprob differences up to 2.09 against a 0.354 noise floor,
    starting at position 2 and confined to a region about ``index_topk`` wide.

    The top-k is causal only where there is something to select FROM. Upstream
    says it plainly at dsa_indexer.py:402 -- "topk_transform selects every valid
    page slot when kv_len <= index_topk". At or below that length the top-k is
    not a selection, it is everything, and the crop is the only thing making the
    result causal. The NPU indexer itself leans on this: it calls
    npu_lightning_indexer with sparse_mode=3 (dsa_npu_indexer.py:527), so the
    indices it hands back are not pre-masked.

    Above it, every query in the chunk has at least ``index_topk`` keys strictly
    before it, the scores of later keys are -inf, and a top-k of that can only
    name causal keys. Then, and only then, the crop is dead weight and dropping
    it changes nothing.

    The bound is on the FIRST query in the chunk, which is the one with the
    least context: it sees ``prefix_len`` keys before it plus itself. Every
    later query in the chunk sees more.

    This is also exactly where the packed read pays: it exists for a 16-32k tail
    on a ~958k cached prefix, where ``prefix_len`` clears 2048 by five hundred
    times. At ``prefix_len = 0`` it saves nothing anyway -- there is no gathered
    prefix to permute, and the remap is the identity.
    """
    return prefix_len + 1 >= index_topk


def dcp_packed_read_plan(
    forward_batch, index_topk: Optional[int]
) -> Optional[DcpPackedReadPlan]:
    """This forward's packed-buffer plan, or None to keep the permuting path.

    Resolved once per forward and cached on the batch. It lives here, beside the
    arithmetic, rather than in the NPU model module, because BOTH the model (to
    fill the buffer and remap the top-k) and the attention backend (to set the
    operator's KV length and drop its causal crop) have to agree on it, and a
    backend importing a model module inverts the layering -- the same reason
    ``dsa_cp.py`` sits where it does.

    Every refusal is logged once. A silent refusal here would look exactly like
    a feature that is on and doing nothing, which is a mistake this port has
    already paid four weeks for.
    """
    if not _enable_dcp_packed_read:
        return None
    cached = getattr(forward_batch, "npu_dcp_packed_plan", _PACKED_MISSING)
    if cached is not _PACKED_MISSING:
        return cached

    plan = None
    extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
    prefix_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    if index_topk is None:
        # No top-k to measure, so the causal bound below cannot be checked.
        # Refuse, and CACHE the refusal: a forward has to take one path the
        # whole way down. Resolving later would leave earlier layers' top-k
        # unremapped while later layers read a packed buffer.
        print_info_once(
            "DCP packed read is off: this forward reached the extend gather "
            "with no top-k, so the causal bound cannot be checked"
        )
    elif not extend_lens or prefix_lens is None:
        print_info_once("DCP packed read is off: no CPU length metadata on this extend")
    elif len(extend_lens) != 1:
        # ONE REQUEST, for the layout reason in DcpPackedReadPlan: the
        # all-gather is rank-major over the WHOLE send, so with two requests a
        # request's rows land in dcp_size separate blocks and the operator's
        # cumulative actual_seq_lengths_kv cannot describe them. The permuting
        # path handles those; it is 13% of the tokens in the served benchmark.
        print_info_once(
            f"DCP packed read is off for multi-request extends "
            f"({len(extend_lens)} requests here); the all-gather is rank-major "
            "over the whole send, so no request is contiguous in it"
        )
    elif not dcp_crop_free_extend(forward_batch, index_topk):
        # Measured, stage A, 2026-09-19: dropping the crop below this bound
        # moved prefill logprobs by up to 2.09 against a 0.354 noise floor.
        # dcp_crop_free_extend has already logged which request fell short.
        print_info_once(
            "DCP packed read is off: this chunk needs the operator's causal "
            "crop, and the packed layout cannot use one"
        )
    else:
        plan = plan_dcp_packed_read(
            prefix_lens[0], extend_lens[0], get_parallel().dcp_size
        )
        print_info_once(
            "DCP packed read is ON: the sparse operator reads the all-gather's "
            "own rank-major output and the top-k is remapped instead"
        )
    forward_batch.npu_dcp_packed_plan = plan
    return plan


def dcp_packed_kv_lens(forward_batch, plan: DcpPackedReadPlan, device) -> torch.Tensor:
    """The operator's ``actual_seq_lengths_kv`` for a packed read, built once.

    One request, so one entry, and it is the whole buffer: with the causal crop
    off there is nothing to shorten, and the padding rows the ranks sent to keep
    their sends equal are never named by any remapped index.

    Cached on the batch rather than built where it is used, because that use is
    once per layer and ``torch.tensor(list, device=npu)`` is a blocking
    host-to-device copy -- 78 of them a forward, each draining the queue. The
    same mistake was already made and fixed once on the DSA-CP path.
    """
    cached = getattr(forward_batch, "npu_dcp_packed_kv_lens", None)
    if cached is None:
        cached = torch.tensor([plan.rows], dtype=torch.int32, device=device)
        forward_batch.npu_dcp_packed_kv_lens = cached
    return cached
