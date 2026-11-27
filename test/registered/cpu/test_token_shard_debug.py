"""E120 host regressions: seam diagnostics for the token-shard rebuild.

The r8-r10 shard-arm crashes (async 507035 / HcclCreateOpResCtx) surface at a
victim sync point, so their tracebacks never name the faulting kernel. These
tests pin the local half of the diagnostic story: the TSDBG plumbing is inert
unless enabled, describes tensors truthfully, and the seam it instruments
zero-pads correctly for the exact crash geometry (26 tokens -> pad 32 @ tp16).
"""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))

from sglang.srt.layers.attention.dsa import dsa_token_shard as dts
from sglang.srt.layers.attention.dsa.dsa_token_shard_layout import (
    plan_dsa_token_shard,
)


def _reset():
    dts._TS_SEEN.clear()
    dts._TS_DEBUG_CACHE.clear()


def test_ts_debug_inert_without_env(monkeypatch):
    monkeypatch.delenv("SGLANG_NPU_TOKEN_SHARD_DEBUG", raising=False)
    _reset()
    dts.ts_debug("seam.off", x=torch.zeros(2, 2))
    assert dts._TS_SEEN == {}  # returned before counting: a true no-op


def test_ts_debug_enabled_counts_and_throttles(monkeypatch):
    monkeypatch.setenv("SGLANG_NPU_TOKEN_SHARD_DEBUG", "true")
    _reset()
    for _ in range(dts._TS_DEBUG_HITS + 2):
        dts.ts_debug("seam.on", x=torch.zeros(2, 2))
    assert dts._TS_SEEN["seam.on"] == dts._TS_DEBUG_HITS


def test_ts_desc_reports_finiteness_only_for_floats():
    int_desc = dts._ts_desc(torch.zeros(3, dtype=torch.int32))
    assert "finite" not in int_desc and "int32" in int_desc
    nan = torch.ones(2, dtype=torch.float32)
    nan[1] = float("nan")
    assert "finite=False" in dts._ts_desc(nan)
    assert "finite=True" in dts._ts_desc(torch.ones(2))
    assert dts._ts_desc(7).startswith("7")


def _plan(rank: int):
    # r10 crash geometry: 26 real tokens, padded to 32 at attn_tp 16 -> 2
    # rows per rank; extend lens chosen non-divisible on purpose.
    return plan_dsa_token_shard([16, 10], [100, 111], 16, rank)


def test_slice_pads_zero_row_rank(monkeypatch):
    monkeypatch.setenv("SGLANG_NPU_TOKEN_SHARD_DEBUG", "true")
    _reset()
    x = torch.arange(26 * 4, dtype=torch.float32).reshape(26, 4)
    plan = _plan(rank=13)  # local rows entirely past the real tokens
    out = dts.dsa_token_shard_slice(x, plan)
    assert out.shape == (plan.rows, 4)
    assert torch.equal(out, torch.zeros_like(out))
    assert dts._TS_SEEN.get("slice") == 1  # seam fired


def test_slice_keeps_real_rows(monkeypatch):
    monkeypatch.delenv("SGLANG_NPU_TOKEN_SHARD_DEBUG", raising=False)
    _reset()
    x = torch.arange(26 * 4, dtype=torch.float32).reshape(26, 4)
    plan = _plan(rank=0)
    out = dts.dsa_token_shard_slice(x, plan)
    assert torch.equal(out, x[: plan.rows])


def test_crash_geometry_plan_fields():
    plan = _plan(rank=0)
    assert plan.num_tokens == 26
    assert plan.num_tokens_pad == 32
    assert plan.rows == 2


def test_boundary_gather_reconciles_token_shard_widths(monkeypatch):
    """E126 (r15): the E120 blocking crash -- attn_tp_gather_input under the
    rebuild handed the plain gather a rank-local width while the output
    buffer followed a wider rule ("output tensor size must be equal to
    world_size times input tensor size"). The reconciled gather must pad
    this rank's rows to plan.rows, gather rows*tp, and trim to
    num_tokens_pad; a wider input is a loud failure, never a truncation."""
    from sglang.srt.layers.layer_boundary import ops as boundary_ops
    from sglang.srt.hardware_backend.npu import besttp12

    calls = []

    def fake_all_gather(output, tensor):
        calls.append((output.shape[0], tensor.shape[0]))
        for rank in range(4):
            output[rank * tensor.shape[0] : (rank + 1) * tensor.shape[0]] = tensor + rank

    monkeypatch.setattr(
        boundary_ops, "get_parallel", lambda: SimpleNamespace(attn_tp_size=4)
    )
    monkeypatch.setattr(boundary_ops, "attn_tp_all_gather_into_tensor", fake_all_gather)
    evidence = []
    monkeypatch.setattr(besttp12, "evidence", lambda name, **kw: evidence.append(name))
    plan = plan_dsa_token_shard([3, 2, 1], [10, 11, 12], 4, 1)  # 6 tokens, rows 2, pad 8
    batch = SimpleNamespace(
        npu_dsa_cp_token_sharded=True, npu_dsa_token_shard_plan=plan
    )
    local = torch.ones(2, 3)
    out = boundary_ops.attn_tp_gather_input(local, batch)
    assert out.shape == (plan.num_tokens_pad, 3)
    # HCCL all_gather semantics: output = world_size x input, both uniform.
    assert calls == [(plan.rows * 4, plan.rows)]
    assert torch.equal(out[:2], torch.full((2, 3), 1.0))  # this rank's rows first
    assert evidence == ["ep-shard-gather-reconciled"]
    # A wider input is a loud divergence, not a silent truncation.
    with pytest.raises(RuntimeError, match="diverged"):
        boundary_ops.attn_tp_gather_input(torch.ones(5, 3), batch)
    # The flag off keeps the plain path (no reconciliation, no evidence).
    plain = SimpleNamespace(npu_dsa_cp_token_sharded=False)
    monkeypatch.setattr(
        boundary_ops,
        "attn_tp_gather_with",
        lambda tensor, gather: tensor,
    )
    assert boundary_ops.attn_tp_gather_input(local, plain) is local
    assert len(evidence) == 1
