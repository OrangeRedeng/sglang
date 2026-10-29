"""CPU unit test for the SFA gather / LightningIndexer overlap (branch D).

The overlap is a re-sequencing, not a new layout: the DCP extend gather's
collectives are issued at the top of prepare, before the QKV (they move only
the pool prefix, which earlier forwards wrote, so they depend on nothing the
QKV or the indexer computes), and its elementwise tail (own-KV copy +
index_select) moves to core, after a wait on the Work handles, right before
the sparse operator reads the buffer. This chunk's own KV -- the one part the
collective cannot carry -- is attached to the pending once the QKV produces
it. Issuing the gather before the QKV, not just before the indexer (the vLLM
spot), is what gives the 57 reuse layers, which run no indexer, a window to
hide it in. The permuting path therefore stops recycling one widest-piece
scratch across pieces (every collective is in flight at once) and keeps one
gather scratch per piece.

The value this test guards is the one that re-sequencing can break silently:
that the async split hands the operator the EXACT rows the sync path did. A
piece's scratch recycled into its neighbour's while a collective is still in
flight, a send slice taken for the wrong piece, or an elementwise tail applied
before its gather, all produce a position-ordered buffer that looks plausible
and stops following the prompt. So the test emulates BOTH orderings against a
labelled world -- sync (gather_i, tail_i, gather_{i+1}, ...) and async (every
gather, then every tail) -- with per-piece scratch on the async side, and
requires the two outputs, and the position-ordered reference, to be bitwise
equal. The wiring itself is pinned by source checks, in the style of the A4
step: the start must sit before the QKV (and hence the indexer) in prepare,
the own KV must be attached after the QKV, and the wait must sit before the
sparse operator in core.

Usage:
    python -m pytest test_dcp_sfa_gather_overlap.py -v
    python test_dcp_sfa_gather_overlap.py
"""

import os
import unittest

import torch

from sglang.srt.layers.dcp.layout import plan_dcp_extend_gather
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

MLA_NPU = (
    "python/sglang/srt/hardware_backend/npu/modules/"
    "deepseek_v2_attention_mla_npu.py"
)
PARALLEL_STATE = "python/sglang/srt/distributed/parallel_state.py"
ENVIRON = "python/sglang/srt/environ.py"

DCP_SIZES = [2, 4, 16]
DIM = 3
SERVED_PIECE_ROWS = 1 << 18

# (prefix_lens, extend_lens, piece_rows). The first rows are what a
# prefix-cache hit and a chunked prefill produce -- page-aligned prefixes --
# and the rest are ragged. The piece cuts span one-piece, few-piece and the
# served 4-piece shape at dcp16.
CASES = [
    ([0], [7], 4096),
    ([16], [5], 1),
    ([2048 * 3], [1558], 4096),
    ([16384, 32768], [16384, 1], 4096),
    ([5], [3], 16),
    ([1], [1], 1),
    ([33, 0, 48, 7], [1, 9, 16, 2], 5),
    ([48, 5], [0, 3], 64),
    ([0, 0], [4, 4], 4096),
    ([976384], [16384], SERVED_PIECE_ROWS),
]


def _combinations():
    for prefix_lens, extend_lens, piece_rows in CASES:
        for dcp_size in DCP_SIZES:
            # Thousands of one-row pieces test nothing a few hundred do not.
            if sum(prefix_lens) <= 256 * piece_rows or piece_rows == 1:
                yield prefix_lens, extend_lens, dcp_size, piece_rows


def _world(prefix_lens, extend_lens, dcp_size, piece_rows, seed=0):
    """The labelled world: rank r holds position r, r + dcp_size, ..."""
    g = torch.Generator().manual_seed(seed)
    prefixes = [torch.randn(p, 1, DIM, generator=g) for p in prefix_lens]
    extends = [torch.randn(e, 1, DIM, generator=g) for e in extend_lens]
    # One plan per rank at the test's piece cut: the sends are built from
    # them, and the emulation runs the rank-0 plan (every rank plans the same
    # collectives -- pinned by test_dcp_extend_gather_plan.py).
    plans = [
        plan_dcp_extend_gather(prefix_lens, extend_lens, dcp_size, rank, piece_rows)
        for rank in range(dcp_size)
    ]
    sends = []
    for rank, plan in enumerate(plans):
        local = [kv[rank::dcp_size] for kv in prefixes]
        send = torch.full((plan.send_rows, 1, DIM), float("nan"))
        dst = 0
        for shard, local_len, padded_len in zip(
            local, plan.local_lens, plan.padded_lens
        ):
            assert shard.shape[0] == local_len, (shard.shape[0], local_len)
            send[dst : dst + local_len] = shard
            dst += padded_len
        sends.append(send)
    own = torch.cat(extends)
    expected = torch.cat([torch.cat([p, e]) for p, e in zip(prefixes, extends)])
    return sends, own, expected


def _sync_out(sends, own, expected, plan, dcp_size):
    """The base ordering: each piece's gather and elementwise back to back,
    recycling one shared scratch (the shape the sync device code takes)."""
    out = torch.full_like(expected, float("nan"))
    for piece in plan.pieces:
        gathered = [send[piece.send_start : piece.send_end] for send in sends]
        scratch = torch.cat(gathered + [own[piece.extend_start : piece.extend_end]])
        out[piece.out_start : piece.out_end] = scratch.index_select(0, piece.index)
    return out


def _async_out(sends, own, expected, plan, dcp_size):
    """The branch D ordering: every collective first (per-piece scratch, all
    in flight), the elementwise tails only after the wait."""
    out = torch.full_like(expected, float("nan"))
    # prepare: issue every piece's all-gather into its own scratch, all of
    # them before any elementwise (the indexer runs in between on device).
    scratches = []
    for piece in plan.pieces:
        gathered_rows = (piece.send_end - piece.send_start) * dcp_size
        rows = gathered_rows + piece.extend_end - piece.extend_start
        scratch = torch.full((rows, 1, DIM), float("nan"))
        if gathered_rows:
            scratch[:gathered_rows] = torch.cat(
                [send[piece.send_start : piece.send_end] for send in sends]
            )
        scratches.append((piece, scratch))
    # core: wait (no-op here), then the per-piece tail.
    for piece, scratch in scratches:
        gathered_rows = (piece.send_end - piece.send_start) * dcp_size
        scratch[gathered_rows:] = own[piece.extend_start : piece.extend_end]
        out[piece.out_start : piece.out_end] = scratch.index_select(0, piece.index)
    return out


class TestDcpSfaGatherOverlapReordering(CustomTestCase):
    def test_async_split_is_bitwise_identical_to_sync(self):
        for prefix_lens, extend_lens, dcp_size, piece_rows in _combinations():
            with self.subTest(
                prefix_lens=prefix_lens,
                extend_lens=extend_lens,
                dcp_size=dcp_size,
                piece_rows=piece_rows,
            ):
                sends, own, expected = _world(
                    prefix_lens, extend_lens, dcp_size, piece_rows
                )
                plan = plan_dcp_extend_gather(
                    prefix_lens, extend_lens, dcp_size, 0, piece_rows
                )
                out_sync = _sync_out(sends, own, expected, plan, dcp_size)
                out_async = _async_out(sends, own, expected, plan, dcp_size)
                self.assertTrue(
                    torch.equal(out_sync, out_async),
                    "async re-sequencing moved bytes relative to the sync path",
                )
                self.assertTrue(
                    torch.equal(out_async, expected),
                    "async re-sequencing does not reproduce the position order",
                )

    def test_async_path_multiplies_pieces_not_wire(self):
        # The overlap must not change what moves: same collectives, same sends,
        # same order -- only the scratch each one lands in differs.
        for prefix_lens, extend_lens, dcp_size, piece_rows in _combinations():
            with self.subTest(
                prefix_lens=prefix_lens,
                extend_lens=extend_lens,
                dcp_size=dcp_size,
                piece_rows=piece_rows,
            ):
                plan = plan_dcp_extend_gather(
                    prefix_lens, extend_lens, dcp_size, 0, piece_rows
                )
                total_scratch = 0
                for piece in plan.pieces:
                    total_scratch += (
                        (piece.send_end - piece.send_start) * dcp_size
                        + piece.extend_end
                        - piece.extend_start
                    )
                # Summed over the pieces, the async scratch holds exactly the
                # gathered rows (the rank-major all-gather output, which is
                # send_rows per rank, padded up from the prefix) plus this
                # chunk's own KV -- the flag's documented memory cost. For
                # page-aligned prefixes (the served case) the padding is zero
                # and this equals the output's rows.
                self.assertEqual(
                    total_scratch,
                    plan.send_rows * dcp_size + sum(extend_lens),
                )


if __name__ == "__main__":
    unittest.main()
