# R08 — attention recovery, compensated sparsity, cache/freeze instrumentation: code-phase report

**Branch:** `feat/minimax-h3-r08-attention-cache-recovery` (from R01 `68e373c50c47492e5860e2d18a9599133ea9c754`)
**Date:** 28 November 2026 · code phase · host tests only, no device runs
**Scope:** P0 dispatch-isolation prerequisites plus the exact-math and instrumentation pieces of P1. No new kernels, no trained predictors, no guard removals, no approximate feature stacking. Everything device-dependent stays explicit and blocked.

## 0. Baseline audit (what R01 already owns)

- `ascend_sparse_attn.py` already chooses the dense fallback (Laser on A2/A3 BF16 non-causal MHA, else AscendFA), validates fail-closed (Laser rejects non-BF16 precision; quantized dense rejects causal/cross/Ring), and handles skip-first/last, `keep_sink`/`keep_recent`, `sparsity==0`, empty spans, packed trailing padding, per-document spans. The ERR00100 ABI question is *not* answered in code anywhere — that is exactly the A0 gap.
- `_adaln_memo_params` (TP1 eager AdaLN memo) exists with guard set and version-keyed invalidation, but **no hit/miss counters** — any reuse claim was unmeasurable (handoff C0).
- `selective_ffn_rows` enforces fractions {0.5, 0.75} with native-boundary protection, but logs nothing — the actual compacted row counts behind a `.5` diagnostic were invisible (handoff D/F0).
- `prototypes.py` ships `bounded_kv_plan` / `merge_softmax_branches` / `sparse_demands` / `compact_reference_rows` — untested exact math (handoff B: "FP32 logaddexp LSE normalization, no double count").

## 1. Commit map

| Commit | Files |
|---|---|
| `Add R08 dispatch isolation probe and per-step backend recording` | `research/r08_dispatch_probe.py` (new), `.../backends/ascend_sparse_attn.py` |
| `Add AdaLN memo and selective-FFN instrumentation` | `.../models/dits/minimax_h3.py`, `.../models/dits/minimax_h3_selective.py` |
| `Add R08 exact-math and instrumentation host tests` | `test/unit/test_minimax_h3_r08_prototypes.py` (new), `research/R08_REPORT.md` |

## 1.5 Test results (host, 28 Nov)

- `test_minimax_h3_r08_prototypes.py`: **8 passed** (0 failed, 0 skipped).
  CPU torch 2.15.0+cpu with a host-only triton import shim
  (`/tmp/h3_host_shim`, never committed); `mindiesd` stubbed at import for
  the recorder test. Host execution is **not** hardware validation.
- Regression: `test_minimax_h3_dit_contract.py` + `test_ascend_sparse_attention.py`
  combined → **157 passed, 4 failed, 4 skipped — the 4 are the known
  pre-existing graph-replay cases, identical on pristine R01 in this
  environment**. Zero regressions attributable to R08.
- Lint: `ruff check` clean on all five touched files.

## 2. Per-subtask status

### A. Backend recovery (P0) — probe + per-step recording implemented; ERR00100 root cause device-pending
- `r08_dispatch_probe.py` isolates the three dispatch layers the H08 failure could involve — (1) `torch_npu.npu_fused_infer_attention_score`, (2) the repo's AscendFA dense executor, (3) `mindiesd.sparse_attention` — each with signature introspection, installed-package provenance (torch/torch-npu/mindiesd/sgl-kernel-npu/sglang versions), and `--device-calls` minimal-call execution that captures the **complete traceback** of any operator error. The first layer that errors is the failing op; no blind reinstalls. Host runs report degraded verdicts without crashing.
- Opt-in per-step dispatch recording (`SGLANG_H3_ATTN_DISPATCH_LOG=1`): a capped ring buffer records the actual backend decision per attention call (dense backend + precision, or sparse type/precision/block/sparsity, with the step), snapshot via `dispatch_log_snapshot()`. A run can now *prove* which path served each step/layer instead of inferring from profiler tags. Off by default; zero overhead when unset.
- Device-pending: the actual ERR00100 traceback capture, dense-BF16 reference, then conservative RF policies per the handoff gates. The existing fail-closed behavior is unchanged; no backend selection logic was altered.

### B. Compensated-sparsity exact math (P1) — prototype correctness proven on host
New exact-math tests over the existing prototypes (no prototype code changed):
- `merge_softmax_branches` equals the full softmax exactly when the exact branch and disjoint representatives partition the KV (FP64 reference, `rtol=1e-4`), output is FP32, and an empty representative branch (`-inf` LSE) contributes zero weight without NaN — the "no double count / LSE normalization" contract.
- `bounded_kv_plan`: mandatory + local window + importance-ranked candidates under an absolute budget; mandatory overflow and oversized candidate lists are loud errors (no silent quadratic routing).
- `sparse_demands` per-owner deduplication; `compact_reference_rows` position/inverse preservation.
- Not claimed: real operator LSE returns, logit-bias support, or gathered-K/V contracts — those remain operator probes (the handoff's "if the operator cannot supply needed stats/bias, stop that design").

### C. AdaLN/cache instrumentation (P1) — counters implemented; behavior unchanged
- Opt-in `SGLANG_H3_ADALN_MEMO_STATS=1` attaches a counter dict to the memo: hits, misses, stores, evictions, live entries — updated in the existing read path without changing any cache decision (the pop-to-consume/store semantics are untouched). This is the C0 control: the 4-step warmup question ("actual scheduler/model/block counts... hits logged") now has a number instead of a narrative.
- The memo's own guards (TP>1 exclusion, graph/compile/Cache-DiT/offload exclusion, capture-query fallback, version-keyed invalidation) were audited and left as-is; no duplicate of R01's slot cleanup.

### D. Static FFN diagnostic instrumentation (P1) — row counts made observable
- Opt-in `SGLANG_H3_SELECTIVE_FFN_LOG=1` logs the actual per-phase compacted row counts and the realized compaction ratio when `selective_ffn_rows` builds a plan (counts are phase-dependent, e.g. 48/64/48/32 for a 4-frame span at 0.5 — verified exactly in tests). `.5` remains diagnostic-only; no promotion path, no dynamic selection implemented (F1 stays conditional on the static probe showing measurable row reduction with acceptable quality).

### E. Low-priority ideas — documented, not implemented
Adaptive temporal density, interpolation auxiliaries, progressive resolution: feasibility notes only (handoff allows "record deferred rather than implement"). Trained substitutions (RAVEN/Physis/VDN/pi-H3/#36318 SLA) excluded as before.

## 3. Tests and environment

The dispatch probe emits a stable H3 experiment identity (batch 1, four tokens,
four heads, head dimension 128, BF16, BNSD/BSND layouts) and separate capability
receipts for native dense FA, Laser dense fallback, MindIE-SD import, quantized
dense FA, RFv2 and RFv3. Host runs mark device-dependent receipts
`device_pending`. Import or signature presence is not execution evidence.

`test_minimax_h3_r08_prototypes.py` — 8 CPU tests: LSE-merge exactness vs full softmax (incl. empty-branch case), bounded-plan bounds/overflow errors, demand dedup, reference-row compaction, dispatch-recorder gate/ring/overwrite, selective-FFN exact per-phase counts + fraction validation, probe host-runnability. Environment: CPU torch 2.15.0+cpu, host-only triton shim; `mindiesd` stubbed at import for the recorder test (the real op is probed on device).

Regression: combined `test_minimax_h3_dit_contract.py` + `test_ascend_sparse_attention.py` → **157 passed, 4 failed, 4 skipped — the 4 failures are the known pre-existing graph-replay cases, identical on pristine R01**. Zero regressions. Lint clean on all five touched files.

## 4. Pending device gates and honest gaps

- A0: run the probe with `--device-calls` on the operator host; the ERR00100 traceback and package set are the deliverable that unblocks (or honestly blocks) RF.
- A1/A2/A3: conservative RF BF16, precision ablations after R02, protected boundary policies — all conditional on A0 and existing configs; not started.
- B: operator probes for external block indices / LSE returns / logit bias — the gathered-dense fallback stays unwritten until those answers exist.
- C1/C2: memo counters on device (TP1) — measure before any claim; #50256 folding audit not started.
- F0/F1: `.5` diagnostic runs use the new counters; dynamic freezing (FIS-DiT/PreSTP/FreezePlan) not started — gated on static results.
- No approximate combination anywhere; quantization/seam quality assessed before any sparse/cache stacking.

## 5. Runbook snippet

```bash
# host tests
PYTHONPATH=python python -m pytest python/sglang/multimodal_gen/test/unit/test_minimax_h3_r08_prototypes.py -q
# dispatch probe (operator host; full traceback capture)
python test/manual/minimax_h3_realtime/research/r08_dispatch_probe.py --device-calls
# instrumentation flags (all opt-in, zero overhead when unset)
SGLANG_H3_ATTN_DISPATCH_LOG=1   # per-call backend ring buffer
SGLANG_H3_ADALN_MEMO_STATS=1    # memo hit/miss/eviction counters
SGLANG_H3_SELECTIVE_FFN_LOG=1   # compacted row counts per plan
```
