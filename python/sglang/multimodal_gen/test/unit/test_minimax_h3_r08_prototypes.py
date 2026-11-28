# SPDX-License-Identifier: Apache-2.0
"""Host tests for the R08 attention/cache pieces (exact math + instrumentation)."""

import importlib.util
import sys
import types
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[5]
_RESEARCH = _REPO_ROOT / "test" / "manual" / "minimax_h3_realtime" / "research"


def _load_prototypes():
    """Import the research prototypes module by path (flat-layout research dir)."""
    spec = importlib.util.spec_from_file_location(
        "r08_prototypes", _RESEARCH / "prototypes.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# B: compensated sparsity exact-math prototypes
# ---------------------------------------------------------------------------


def _full_attention(q, k, v, scale):
    """Reference dense attention in float64: the exact target."""
    logits = torch.einsum("qhd,khd->hqk", q.double(), k.double()) * scale
    weights = torch.softmax(logits, dim=-1)
    return torch.einsum("hqk,khd->qhd", weights, v.double())


def test_merge_softmax_branches_equals_full_softmax():
    """Exact local + disjoint representatives merged by LSE must reproduce the
    full softmax exactly (no double counting), in FP32."""
    proto = _load_prototypes()
    torch.manual_seed(0)
    heads, q_len, k_len, dim = 2, 6, 10, 8
    scale = 0.4
    q = torch.randn(q_len, heads, dim)
    k = torch.randn(k_len, heads, dim)
    v = torch.randn(k_len, heads, dim)
    split = 4  # exact branch sees the first `split` keys, reps the rest

    logits = torch.einsum("qhd,khd->hqk", q.double(), k.double()) * scale
    exact_lse = torch.logsumexp(logits[:, :, :split], dim=-1)
    rep_lse = torch.logsumexp(logits[:, :, split:], dim=-1)
    # Branch outputs follow the merge contract: [heads, queries, dim].
    exact = torch.einsum(
        "hqk,khd->hqd", torch.softmax(logits[:, :, :split], dim=-1), v[:split].double()
    )
    rep = torch.einsum(
        "hqk,khd->hqd", torch.softmax(logits[:, :, split:], dim=-1), v[split:].double()
    )

    merged = proto.merge_softmax_branches(exact, exact_lse, rep, rep_lse)
    reference = _full_attention(q, k, v, scale)
    # merged is [h, q, d]; the reference einsum output is [q, h, d].
    assert torch.allclose(merged.permute(1, 0, 2).double(), reference, rtol=1e-4, atol=1e-5)
    # Output is FP32, not FP64.
    assert merged.dtype == torch.float32


def test_merge_softmax_branches_handles_empty_representatives():
    """A branch with -inf LSE (no keys) contributes zero weight, not NaN."""
    proto = _load_prototypes()
    exact = torch.ones(2, 2, 4)
    exact_lse = torch.log(torch.tensor([[4.0], [4.0]]))
    rep = torch.zeros(2, 2, 4)
    rep_lse = torch.full((2, 1), float("-inf"))
    merged = proto.merge_softmax_branches(exact, exact_lse, rep, rep_lse)
    assert torch.isfinite(merged).all()
    assert torch.allclose(merged.double(), exact.double(), rtol=1e-6)


def test_bounded_kv_plan_bounds_and_mandatory():
    proto = _load_prototypes()
    plan = proto.bounded_kv_plan(
        5,
        blocks=16,
        local_radius=1,
        mandatory=(0, 15),
        candidates=[(0.9, 7), (0.4, 3), (0.99, 9)],
        budget=6,
    )
    # mandatory + local window first, then candidates by importance, bounded
    assert plan[0] == 0 and plan[-1] == 15
    assert set(plan) == {0, 4, 5, 6, 9, 15}
    assert len(plan) <= 6

    # Mandatory sets that overflow the budget are a loud error, never silent.
    import pytest

    with pytest.raises(ValueError, match="Mandatory dependencies exceed budget"):
        proto.bounded_kv_plan(
            0, blocks=8, local_radius=0, mandatory=(0, 1, 2, 3), candidates=(), budget=2
        )
    with pytest.raises(ValueError, match="bounded router budget"):
        proto.bounded_kv_plan(
            0,
            blocks=8,
            local_radius=0,
            mandatory=(),
            candidates=[(float(i), i) for i in range(9)],
            budget=2,
        )


def test_sparse_demands_deduplicates_per_owner():
    proto = _load_prototypes()
    plans = [(0, 5, 9), (5, 7), (9,)]
    owners = {block: block % 3 for block in range(10)}
    demands = proto.sparse_demands(plans, owners, rank=0)
    # rank 0 owns blocks {0, 3, 6, 9}: block 9 is demanded but owned locally;
    # block 5 (owner 2) appears in two plans and must deduplicate to one.
    assert demands[1] == (7,)
    assert demands[2] == (5,)


def test_compact_reference_rows_preserves_positions():
    proto = _load_prototypes()
    hidden = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    positions = torch.arange(10) * 3
    reference_mask = torch.tensor([True, True, False, False, True, False, True, False, False, False])
    keep_reference = torch.tensor([True, False, False, False, True, False, False, False, False, False])
    compact, compact_pos, inverse = proto.compact_reference_rows(
        hidden, positions, reference_mask, keep_reference
    )
    assert compact.shape[0] == compact_pos.shape[0] == inverse.shape[0]
    # dropped row 1 (reference, not kept) and row 6 (reference, not kept)
    assert inverse.tolist() == [0, 2, 3, 4, 5, 7, 8, 9]
    assert torch.equal(compact_pos, positions.index_select(0, inverse))
    assert torch.equal(compact, hidden.index_select(0, inverse))


# ---------------------------------------------------------------------------
# A: dispatch recorder
# ---------------------------------------------------------------------------


def test_dispatch_recorder_gate_and_ring(monkeypatch):
    # The sparse backend imports mindiesd at module scope; stub it on hosts
    # without MindIE-SD (only the recorder plumbing is under test here).
    if "mindiesd" not in sys.modules:
        stub = types.ModuleType("mindiesd")
        stub.sparse_attention = None
        sys.modules["mindiesd"] = stub
    from sglang.multimodal_gen.runtime.layers.attention.backends.ascend_sparse_attn import (
        _DISPATCH_LOG_CAP,
        _record_dispatch,
        clear_dispatch_log,
        dispatch_log_snapshot,
    )

    monkeypatch.delenv("SGLANG_H3_ATTN_DISPATCH_LOG", raising=False)
    clear_dispatch_log()
    _record_dispatch(kind="dense", step=0)
    assert dispatch_log_snapshot() == ()  # gated off by default

    monkeypatch.setenv("SGLANG_H3_ATTN_DISPATCH_LOG", "1")
    for step in range(_DISPATCH_LOG_CAP + 64):
        _record_dispatch(kind="sparse", step=step)
    snapshot = dispatch_log_snapshot()
    assert len(snapshot) <= _DISPATCH_LOG_CAP
    assert snapshot[-1]["step"] == _DISPATCH_LOG_CAP + 63  # most recent kept
    clear_dispatch_log()
    assert dispatch_log_snapshot() == ()


# ---------------------------------------------------------------------------
# C/D: instrumentation
# ---------------------------------------------------------------------------


def test_selective_ffn_rows_counts_and_validation(monkeypatch):
    import sglang.multimodal_gen.runtime.models.dits.minimax_h3_selective as selective

    monkeypatch.setenv("SGLANG_H3_SELECTIVE_FFN_LOG", "1")
    seq_len = 64
    spans = [{"start": 0, "latent_shape": (4, 4, 4)}]
    target_rows = torch.arange(16, 48)
    plans = selective.selective_ffn_rows(
        seq_len=seq_len,
        row_start=0,
        row_stop=seq_len,
        video_spans=spans,
        target_rows=target_rows,
        active_fraction=0.5,
        device=torch.device("cpu"),
    )
    assert len(plans) == 4
    # frames=4 -> interior frames 1 and 2 carry the phase groups (rows 16-47);
    # rows 0-15 and 48-63 stay ungrouped and are kept in every phase.
    # keep=2 phases see group1(16 rows) and/or group2(16 rows):
    #   phase0 keeps g1; phase1 keeps g1+g2; phase2 keeps g2; phase3 keeps none
    expected_total = [48, 64, 48, 32]
    for plan, expected in zip(plans, expected_total):
        assert int(plan.numel()) == expected
        assert plan.dtype == torch.long
    # The four phase plans tile the interior target rows without overlap.
    interior_kept = set()
    for plan in plans:
        interior = [int(r) for r in plan if 16 <= int(r) < 48]
        interior_kept |= set(interior)
    assert len(interior_kept) == 32

    # Only the approved diagnostic fractions are accepted.
    import pytest

    with pytest.raises(ValueError, match="0.5 or 0.75"):
        selective.selective_ffn_rows(
            seq_len=seq_len,
            row_start=0,
            row_stop=seq_len,
            video_spans=spans,
            target_rows=target_rows,
            active_fraction=0.8,
            device=torch.device("cpu"),
        )


def test_dispatch_probe_reports_without_device():
    import subprocess

    probe = _RESEARCH / "r08_dispatch_probe.py"
    result = subprocess.run(
        [sys.executable, str(probe), "--json"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    report = __import__("json").loads(result.stdout)
    # All three failing-op layers are reported even on host.
    assert set(report["layers"]) >= {
        "native_fused_attention",
        "ascend_fa_backend",
        "mindiesd_sparse",
    }
    assert report["minimal_calls"][0]["status"] == "skipped"
    assert report["packages"]["torch"] is None or isinstance(report["packages"]["torch"], str)
