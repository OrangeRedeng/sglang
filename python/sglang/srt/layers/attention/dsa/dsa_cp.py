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

"""Runtime gate and tensor plumbing for DSA-CP token sharding.

The arithmetic lives in ``dsa_cp_layout`` on purpose: everything here needs the
runtime topology and real tensors, while the layout is plain integers and can
be replayed against a brute-force reference with no torch at all.

**Only the query path is sharded here, not the whole attention block.** The
slice is taken after ``fused_qkv_a_proj_with_mqa`` and the QK norm, and only
for ``q_lora``. The KV cache write, the DCP context gather and the indexer all
keep running at full width, so the older indexer-only query sharding
(``SGLANG_NPU_ENABLE_DSA_INDEXER_QUERY_SHARDING``) composes with this rather
than conflicting: it shards the indexer and gathers the top-k back, and this
takes its own slice of that result.

What is given up is the saving on the K-side projections, which run full width
today anyway. What is kept is the reason for the exercise: sparse attention is
bound by a per-query top-k KV read, which falls by ``attn_tp_size`` when the
queries do.
"""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.autotune import threshold_allows
from sglang.srt.layers.attention.dsa.dsa_cp_layout import (
    DsaCpPlan,
    cumulative,
    plan_dsa_cp_shard,
)
from sglang.srt.layers.dcp.layout import dcp_crop_free_extend
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_npu, print_info_once

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# Read once, at import: the flag decides whether the full-head RadixAttention
# is built, which happens when the model is constructed.
_enable_dsa_cp = envs.SGLANG_NPU_ENABLE_DSA_CP.get()

if _enable_dsa_cp and envs.SGLANG_NPU_USE_MLAPO.get():
    # The fused MLA preprocess writes the KV cache at a slot mapping DSA-CP has
    # already sliced. DSA-CP defaults on, so it yields unless both were set.
    if envs.SGLANG_NPU_ENABLE_DSA_CP.is_set():
        raise ValueError(
            "SGLANG_NPU_ENABLE_DSA_CP does not compose with "
            "SGLANG_NPU_USE_MLAPO. The fused MLA preprocess writes the KV cache "
            "itself, at a slot mapping DSA-CP has already sliced."
        )
    _enable_dsa_cp = False
    print_info_once(
        "DSA-CP is off because SGLANG_NPU_USE_MLAPO is on: the fused MLA "
        "preprocess writes the KV cache at a slot mapping DSA-CP would have "
        "sliced. Unset SGLANG_NPU_USE_MLAPO to get the sharded attention back"
    )


_enable_dsa_cp_multi_request = envs.SGLANG_NPU_ENABLE_DSA_CP_MULTI_REQUEST.get()

# D1v3: the full token-sharded rebuild (see SGLANG_NPU_DSA_CP_TOKEN_SHARDED).
# Read once, at import, like the other flags here.
_enable_dsa_cp_token_sharded = envs.SGLANG_NPU_DSA_CP_TOKEN_SHARDED.get()


def dsa_cp_token_sharded_enabled() -> bool:
    """The process-level flag. The per-forward state is
    ``forward_batch.npu_dsa_cp_token_sharded`` (see
    ``dsa_cp_token_sharded_active``)."""
    return _enable_dsa_cp_token_sharded


def dsa_cp_token_sharded(forward_batch) -> bool:
    """Whether THIS forward runs the token-sharded rebuild.

    The value is stashed on the batch by the attention prepare (the first
    place in a forward that has both the resolved plan and the layer's
    weights), before the prepare_mlp / MoE / logits consumers read it.
    """
    return bool(getattr(forward_batch, "npu_dsa_cp_token_sharded", False))


def dsa_cp_set_token_sharded(forward_batch, active: bool) -> None:
    """Stash the per-forward decision; a mismatch across layers is a wiring
    bug (the condition depends on nothing that varies by layer once the plan
    exists), so it is an error rather than a second guess."""
    prev = getattr(forward_batch, "npu_dsa_cp_token_sharded", _MISSING)
    if prev is not _MISSING and bool(prev) != bool(active):
        raise RuntimeError(
            "DSA-CP token sharding disagrees within one forward "
            f"(earlier {prev}, now {active}); the per-layer condition must "
            "depend only on the batch and the process, not the layer"
        )
    forward_batch.npu_dsa_cp_token_sharded = active


# D2v3: this rank's dequantized o_proj RowParallel shard, per DSA layer id.
# Filled once by the model runner's token-sharded preparation and consumed by
# the NPU gather side stream, which all-gathers the shards into the full
# matrix on demand (see SGLANG_NPU_STREAM_O_PROJ). A registry rather than a
# per-module lookup: the side stream prefetches layer L+1 from inside layer
# L's forward, where it only holds layer L's module.
_o_proj_stream_shards: Dict[int, torch.Tensor] = {}


def register_o_proj_stream_shard(layer_id: int, shard: torch.Tensor) -> None:
    _o_proj_stream_shards[layer_id] = shard


def get_o_proj_stream_shard(layer_id: int) -> Optional[torch.Tensor]:
    return _o_proj_stream_shards.get(layer_id)


def dsa_cp_reduce_scatter_tokens(x: torch.Tensor) -> torch.Tensor:
    """``[rows * tp, ...]`` -> ``[rows, ...]`` over the attention-TP group.

    The inverse ownership of ``dsa_cp_slice``: each rank holds the complete
    (summed) tensor and keeps this rank's row block. The MoE output under the
    token-sharded rebuild takes this instead of the all-reduce: the reduce is
    the same sum the all-reduce computes, and the scatter drops the other
    ranks' rows, which nobody consumes (the residual stays token-sharded).
    """
    parallel = get_parallel()
    tp = parallel.attn_tp_size
    if x.shape[0] % tp:
        raise RuntimeError(
            f"DSA-CP token sharding: {x.shape[0]} rows do not divide by the "
            f"attention-TP size {tp}; the batch padding is wider than the plan"
        )
    rows = x.shape[0] // tp
    out = x.new_empty((rows,) + x.shape[1:])
    parallel.attn_tp_group.reduce_scatter_tensor(out, x)
    return out


def dsa_cp_gather_last_token_rows(x: torch.Tensor, forward_batch) -> torch.Tensor:
    """``[rows, hidden]`` (this rank's slice after the final norm) ->
    ``[num_requests, hidden]`` with each request's last-token row.

    The end of the token-sharded forward: the logits path indexes
    ``hidden_states[cumsum(extend_seq_lens) - 1]`` (logits_processor.py),
    which needs every request's last row on every rank. Only the owners of
    those rows have real data, so each rank contributes its own rows and
    zeros elsewhere, the group all-gathers the small ``[bs, hidden]``
    buffers, and the sum keeps exactly one non-zero row per request.
    """
    parallel = get_parallel()
    tp = parallel.attn_tp_size
    plan = get_dsa_cp_plan(forward_batch)
    if plan is None:
        raise RuntimeError(
            "DSA-CP token sharding: the logits gather needs the forward's "
            "plan but it is gone"
        )
    extend_lens = forward_batch.extend_seq_lens_cpu
    # Request i's last new token sits at the EXCLUSIVE END minus one; cumsum
    # alone is the first token of the NEXT request (and past the end for the
    # last one), so using it as the index shifts every request by one and
    # zeroes the final request's logits row.
    last_rows = list(
        (torch.cumsum(torch.tensor(extend_lens), dim=0) - 1).tolist()
    )
    bs = len(last_rows)
    mine = x.new_zeros((bs,) + x.shape[1:])
    for i, row in enumerate(last_rows):
        if plan.local_start <= row < plan.local_end:
            mine[i] = x[row - plan.local_start]
    full = mine.new_empty((tp * bs,) + mine.shape[1:])
    parallel.attn_tp_group.all_gather_into_tensor(full, mine)
    return full.view(tp, bs, -1).sum(dim=0)


def dsa_cp_multi_request_enabled() -> bool:
    """Whether DSA-CP also shards batches carrying more than one request.

    The restriction was never about the query arithmetic -- ``plan_dsa_cp_shard``
    already handles a slice that straddles request boundaries. It was about
    ``actual_seq_lengths_kv``: DSA-CP shortened each request's entry so the
    operator's causal crop would align, and because those lengths are
    cumulative, shortening one moves where the next request starts.

    So do not shorten them. Pass the full per-request lengths, the same ones
    the unsharded path passes, and set ``sparse_mode`` to 0 so there is no crop
    to align. That is safe only above a bound: below ``index_topk`` the top-k
    selects every key it is offered, so the crop is what makes the result
    causal. ``dcp_crop_free_extend`` decides that once per forward and the
    attention backend reads the same answer.

    When on, this applies to single-request extends too, deliberately -- one
    convention rather than two, and it can be tested on an ordinary tail.
    """
    return _enable_dsa_cp_multi_request


class _Missing:
    """Distinguishes "no plan cached yet" from "cached, and it is None"."""


_MISSING = _Missing()


def dsa_cp_enabled() -> bool:
    """Whether DSA-CP is on for this process. Read at model build AND forward.

    Both sides must agree, so both ask here rather than reading the env var
    twice.

    ``is_npu()`` guards it because this is asked from ``deepseek_v2.py``, which
    every backend shares, and a true answer builds an extra full-head
    ``RadixAttention`` at the same ``layer_id``. The forward path that would use
    it exists only under ``hardware_backend/npu``, so on any other device that
    module is dead weight registered with the attention backend. Harmless while
    the flag defaulted off; not something to discover by flipping the default.
    """
    return _enable_dsa_cp and is_npu() and get_parallel().attn_tp_size > 1


def get_dsa_cp_plan(
    forward_batch: "ForwardBatch",
    index_topk: Optional[int] = None,
) -> Optional[DsaCpPlan]:
    """This forward's token slice for this rank, or None if DSA-CP is off here.

    Built once per forward and cached on the batch: the split is by token
    position and does not vary by layer.

    **Extend only, deliberately, for the first version.** The measured 34% is
    entirely in the tail prefill, and decode is already fast from graph capture,
    so leaving decode on the unsharded path keeps the capture untouched and
    removes most of the risk. vLLM-Ascend does slice at decode too
    (``sfa_cp.py``), and composes with DCP by MRO; that is a later step, not a
    prerequisite.

    Every refusal is logged once. The indexer-sharding bug this supersedes cost
    four weeks precisely because its refusal was silent.
    """
    if not _enable_dsa_cp:
        return None

    cached = getattr(forward_batch, "npu_dsa_cp_plan", _MISSING)
    if cached is not _MISSING:
        return cached

    plan = _build_dsa_cp_plan(forward_batch, index_topk)
    forward_batch.npu_dsa_cp_plan = plan
    return plan


def _build_dsa_cp_plan(forward_batch, index_topk=None) -> Optional[DsaCpPlan]:
    parallel = get_parallel()
    if parallel.attn_tp_size <= 1:
        print_info_once("DSA-CP is off: attention TP size is 1, nothing to shard")
        return None

    mode = forward_batch.forward_mode
    if not mode.is_extend():
        # Decode and the speculative modes keep the unsharded path. Not a
        # defect: see the docstring.
        return None
    if mode.is_draft_extend_v2() or mode.is_target_verify():
        print_info_once(
            f"DSA-CP is off for {mode}: the draft and verify paths carry their "
            "own per-step sequence-length tables, which this does not build yet"
        )
        return None

    extend_lens = forward_batch.extend_seq_lens_cpu
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    if not extend_lens or prefix_lens is None:
        print_info_once("DSA-CP is off: this extend carries no CPU length metadata")
        return None

    if not threshold_allows("SGLANG_NPU_DSA_CP_MIN_TOKENS", sum(extend_lens)):
        return None

    plan = plan_dsa_cp_shard(
        extend_lens,
        [p + e for p, e in zip(prefix_lens, extend_lens)],
        parallel.attn_tp_size,
        parallel.attn_tp_rank,
    )
    multi_request = sum(1 for n in extend_lens if n > 0) > 1
    # The lift applies only where the causal crop is not load-bearing.
    lift_applies = _enable_dsa_cp_multi_request and dcp_crop_free_extend(
        forward_batch, index_topk
    )
    if not lift_applies and multi_request:
        # The restriction is the KV layout, not the query arithmetic: the
        # buffer's cumulative KV lengths double as its request boundaries.
        print_info_once(
            "DSA-CP is off for multi-request extends "
            f"({sum(1 for n in extend_lens if n > 0)} requests here); the "
            "cumulative KV lengths it would need describe the buffer's own "
            "request boundaries. SGLANG_NPU_ENABLE_DSA_CP_MULTI_REQUEST lifts this"
        )
        return None

    if plan.num_tokens < parallel.attn_tp_size:
        # Fewer tokens than ranks: most ranks would hold nothing but padding and
        # the all-gathers would move more than the attention saves.
        print_info_once(
            f"DSA-CP is off for batches under {parallel.attn_tp_size} tokens "
            f"(this one has {plan.num_tokens}); the slice would be mostly padding"
        )
        return None

    # Log when it engages, not only when it refuses: an unset flag leaves this
    # function before every refusal, so no refusal line proves nothing.
    print_info_once(
        f"DSA-CP is ON: the attention query is sharded across "
        f"{parallel.attn_tp_size} ranks at extend"
    )
    return plan


def dsa_cp_cumulative_lens(
    forward_batch: "ForwardBatch", plan: DsaCpPlan, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The operator's two length vectors as device tensors, built ONCE per forward.

    Returns ``(cumulative query lengths, cumulative key lengths)``. Both carry
    one entry per request, and neither varies by layer -- the split is by token
    position. Building them at the point of use meant
    ``torch.tensor(list, device=npu)`` twice per layer, which at 78 layers is
    **156 host-to-device copies per forward**. Each one drains the queue before
    the next kernel is enqueued, so the cost is a synchronisation rather than a
    copy, and it lands squarely inside the win this feature exists to produce.

    Cached on the batch beside the plan, and built as a pair because a forward
    that needs one needs the other.

    ``int32`` on the query's device, which is what both call sites ask for, so
    their ``.to()`` is a no-op rather than another copy.
    """
    cached = getattr(forward_batch, "npu_dsa_cp_cu_lens", None)
    if cached is None:
        cached = (
            torch.tensor(cumulative(plan.query_lens), dtype=torch.int32, device=device),
            torch.tensor(cumulative(plan.key_lens), dtype=torch.int32, device=device),
        )
        forward_batch.npu_dsa_cp_cu_lens = cached
    return cached


def dsa_cp_slice(x: torch.Tensor, plan: DsaCpPlan) -> torch.Tensor:
    """This rank's ``plan.rows`` rows of a full-width, token-major tensor.

    Always returns exactly ``rows`` rows, padding the tail when the slice runs
    past the real tokens. Every rank must produce the same row count or the
    all-gather that rebuilds full width has nothing regular to work with, and
    the padded rows are cheap: they are the last rank's only, at most
    ``tp_size - 1`` of them, and their attention output is discarded.
    """
    sliced = x[plan.local_start : plan.local_end]
    missing = plan.rows - sliced.shape[0]
    if missing <= 0:
        return sliced
    pad = sliced.new_zeros((missing, *x.shape[1:]))
    return torch.cat([sliced, pad], dim=0)


def dsa_cp_redistribute_heads(x: torch.Tensor, plan: DsaCpPlan) -> torch.Tensor:
    """``[num_tokens, h, d]`` -> ``[rows, h * tp_size, d]``, by all-to-all.

    Turns "my heads for every token" into "every head for my tokens". Nothing
    is duplicated and nothing is dropped: the group holds the same
    (token, head) pairs before and after, they just move to a different owner.

    The head order that comes out is the global one: ``all_to_all_single``
    fills output chunk ``s`` from rank ``s``, and a ColumnParallelLinear gives
    rank ``s`` the contiguous head block ``[s*h, (s+1)*h)``.

    ``num_tokens`` is rounded up to ``num_tokens_pad`` with zero rows so every
    rank sends the same width, which is what makes one all-to-all enough.
    SGLang usually pads the batch to exactly that width already.
    """
    parallel = get_parallel()
    tp = parallel.attn_tp_size
    h, d = x.shape[1], x.shape[2]
    assert x.shape[0] <= plan.num_tokens_pad, (
        f"DSA-CP was handed {x.shape[0]} rows but planned for at most "
        f"{plan.num_tokens_pad} ({plan.num_tokens} tokens aligned to {tp}). "
        "The batch is padded by a wider rule than attn_tp_size; the plan must "
        "be built from the padded width, not from extend_seq_lens_cpu."
    )
    missing = plan.num_tokens_pad - x.shape[0]
    if missing > 0:
        x = torch.cat([x, x.new_zeros((missing, h, d))], dim=0)
    # reshape, not view: q_nope_out arrives non-contiguous from
    # npu_transpose_batchmatmul and view() refuses that.
    send = x.reshape(tp, plan.rows, h, d).contiguous()
    recv = torch.empty_like(send)
    parallel.attn_tp_group.all_to_all_single(recv, send)
    return recv.permute(1, 0, 2, 3).reshape(plan.rows, tp * h, d)


def dsa_cp_redistribute_heads_async(
    x: torch.Tensor, plan: DsaCpPlan
) -> Tuple[torch.Tensor, Optional[object], torch.Tensor]:
    """Async sibling of ``dsa_cp_redistribute_heads``.

    Returns ``(recv, work, send)``. Same padding, same send layout -- only the
    collective is enqueued without a host block, so the LightningIndexer (or
    whatever the caller queues next) runs while the bytes move. ``work`` is the
    handle to ``wait()`` on before the result is read; it is None when the
    group has one rank (nothing was exchanged) and the caller skips the wait.

    ``recv`` is the RAW all-to-all output, ``[tp, rows, h, d]``, and it must be
    handed to ``dsa_cp_redistribute_heads_recv`` only AFTER the wait. The
    synchronous function's final ``permute+reshape`` is a COPY (the permuted
    layout is not viewable), and a copy enqueued here would run on the
    compute stream while the collective is still writing ``recv`` on the
    communication stream -- so the materialisation deliberately waits for the
    collective to be waited on.

    ``send`` is the padded, laid-out send buffer: it is returned because the
    async collective reads ``send`` on the communication stream at execution
    time, and the caller must keep it alive (with the work handle) until after
    the wait -- the caching allocator would otherwise be free to hand the
    block to the next allocation the moment this function returns.
    """
    parallel = get_parallel()
    tp = parallel.attn_tp_size
    h, d = x.shape[1], x.shape[2]
    assert x.shape[0] <= plan.num_tokens_pad, (
        f"DSA-CP was handed {x.shape[0]} rows but planned for at most "
        f"{plan.num_tokens_pad} ({plan.num_tokens} tokens aligned to {tp}). "
        "The batch is padded by a wider rule than attn_tp_size; the plan must "
        "be built from the padded width, not from extend_seq_lens_cpu."
    )
    missing = plan.num_tokens_pad - x.shape[0]
    if missing > 0:
        x = torch.cat([x, x.new_zeros((missing, h, d))], dim=0)
    send = x.reshape(tp, plan.rows, h, d).contiguous()
    recv = torch.empty_like(send)
    work = parallel.attn_tp_group.all_to_all_single_async(recv, send)
    return recv, work, send


def dsa_cp_redistribute_heads_recv(
    recv: torch.Tensor, plan: DsaCpPlan
) -> torch.Tensor:
    """``[tp, rows, h, d]`` -> ``[rows, h * tp, d]``, after the wait.

    The post-wait half of ``dsa_cp_redistribute_heads_async``: the same
    permute-plus-reshape the synchronous function applies to its completed
    receive buffer. Doing it here, once the communication stream has finished
    writing ``recv``, is what keeps the copy out of the collective's window.
    """
    parallel = get_parallel()
    tp = parallel.attn_tp_size
    h, d = recv.shape[2], recv.shape[3]
    return recv.permute(1, 0, 2, 3).reshape(plan.rows, tp * h, d)


def dsa_cp_restore_tokens(
    x: torch.Tensor, plan: DsaCpPlan, num_rows: int
) -> torch.Tensor:
    """``[rows, h * tp_size, d]`` -> ``[num_rows, h, d]``. Inverse of the above.

    Everything downstream of attention -- ``w_vc``, o_proj, the layer
    communicator -- expects this rank's own heads for the whole batch, so the
    sharding is undone here rather than propagated.

    ``num_rows`` must be the width handed to ``dsa_cp_redistribute_heads``: a
    width that arrives padded must leave padded, or the next layer's KV write
    indexes a narrower tensor than its slot map describes. It is required
    rather than defaulted because defaulting it is what broke this.

    Rows past ``plan.num_tokens`` are zeroed: the operator was told to process
    only ``query_lens`` queries, so its output there is untouched device
    memory, which can be NaN.
    """
    parallel = get_parallel()
    tp = parallel.attn_tp_size
    h, d = x.shape[1] // tp, x.shape[2]
    assert x.shape[1] == h * tp, (
        f"DSA-CP restore expects a full head set, got {x.shape[1]} for tp_size {tp}"
    )
    assert plan.num_tokens <= num_rows <= plan.num_tokens_pad, (
        f"DSA-CP restore asked for {num_rows} rows, outside "
        f"[{plan.num_tokens}, {plan.num_tokens_pad}]"
    )
    send = x.reshape(plan.rows, tp, h, d).permute(1, 0, 2, 3).contiguous()
    recv = torch.empty_like(send)
    parallel.attn_tp_group.all_to_all_single(recv, send)
    out = recv.reshape(plan.num_tokens_pad, h, d)[:num_rows]
    if num_rows > plan.num_tokens:
        out[plan.num_tokens :].zero_()
    return out
