# R09 — campaign controller extension and supervision: code-phase report

**Branch:** `feat/minimax-h3-r09-campaign-supervision` (from R01 `68e373c50c47492e5860e2d18a9599133ea9c754`)
**Date:** 28 November 2026 · code phase · host tests only, no device runs, no campaign executed
**Scope:** upgrade the existing H09/R01 controller in place (no second controller), per the handoff's P0 items: durable ledger with real accounting, dedup/resume by case fingerprint, bounded retry, raw-sample aggregation, quality receipts bound to artifacts, supervisor bounded fallback, and R05–R08 adapters registered only as implemented. No campaign was run; the device campaign remains operator-initiated.

## 0. Baseline audit

The existing `experiment_controller.py` already provided: typed manifest validation (v2, exact field sets), adapters keyed to real launcher variants, capability-proof gates, per-device/port reservation locks, preflight (host/SHA/disk/deps/package versions/CANN/checkpoint identity), worst-case reservation at launch, stop-file cancellation, `--no-model` deterministic mode, and a loopback-only supervisor with strict decision validation. The R09 gaps it did not cover: ledger durability, budget reconciliation (worst-case charge was never reconciled down), case-fingerprint resume (identical rerun required after a manifest change), retry bound, multi-sample aggregation (median==p90 from a single sample), quality receipts, H02 gating, and endpoint-outage behavior (hard stop). This branch closes exactly those.

## 1. Commit map

| Commit | Files |
|---|---|
| `Harden the campaign ledger with reconciliation and fingerprints` | `experiment_controller.py` |
| `Bound retries, gate evidence, and add raw-sample aggregation` | `experiment_controller.py` |
| `Register R05–R08 adapters and the supervisor fallback` | `experiment_controller.py`, `campaign.example.json`, `SUPERVISOR.md` |
| `Add the R09 controller host tests and report` | `test_minimax_h3_r09_controller.py` (new), `R09_REPORT.md` |

## 1.5 Test results (host, 28 Nov)

- `test_minimax_h3_r09_controller.py` (14 new) + `test_experiment_controller.py`
  (8 pre-existing, one synthetic case updated for the new required `samples`
  field; assertions unchanged): **22 passed** (0 failed, 0 skipped). CPU
  torch 2.15.0+cpu with a host-only triton import shim (`/tmp/h3_host_shim`,
  never committed). Host execution is **not** hardware validation.
- Lint: `ruff check` clean on all three touched controller files.
- Deterministic mode (`--no-model`) exercised via config=None paths in the
  pre-existing suite; the campaign runner itself stays device-pending.

## 2. What changed, per handoff requirement

The campaign manifest now pins both an absolute source worktree and the
branch-specific launcher. Preflight verifies the worktree SHA and cleanliness,
and launch uses that exact launcher and source root. Every completed run must
also contain a `comparison.json` receipt whose execution SHA matches the
approved source. Conservative A5 R02 cases are registered as `R02.Q5_FFN` and
`R02.Q5_ALLOWLIST`, mapping to the R02 launcher's Q6/Q7 implementation behind
separate capability gates.

### Durable ledger, unique run IDs, case fingerprints (resume only identical evidence)
- `journal()` now sets `PRAGMA journal_mode=WAL` + `synchronous=FULL`: a killed controller cannot lose a reservation or terminal record (which would otherwise double-spend the budget).
- Every launch records a fresh UUID `run_id`; the terminal record carries the same id, so paired reservation/reconciliation is unambiguous even across retries.
- `fingerprint_case(manifest, case)` digests source SHA, checkpoint hash, the full contract (seed/quality/software/family), adapter, repeat-of, sample count and run timeout. `latest_cases(events, manifest)` demotes completed evidence whose fingerprint no longer matches to `stale`: it can never satisfy a dependency or gate, and the case re-runs under the new identity. Resume of an unchanged manifest keeps prior evidence (unchanged behavior).

### Reservation → reconciliation accounting
- Launch reserves the worst case (devices × run timeout) as before. New `outstanding_charges(events)` computes the budget as: reconciled actuals (only when cleanup was **confirmed**: completed/cancelled/failed/timed_out terminal records carry `actual_device_s` = ⌈devices × measured wall⌉) + still-outstanding reservations. Unknown cleanup (`cleanup_unverified`) or an unknown clock keeps the full worst-case charge. `remaining = max_device_s − outstanding_charges(...)`; the launch-time `charge > remaining` guard still applies, so spent + outstanding can never exceed the approved limit.

### Bounded retries; OOM/correctness block the track
- `classify_failure()` maps failures to `oom` / `correctness` / `cleanup` / `transient` from the recorded error text and terminal status. OOM and correctness (geometry/parity/denoise-evidence) failures block the track immediately — no retry. `transient` failures retry exactly once (`attempt` is recorded per run; the second failure pauses the campaign with `track exhausted`). A retry re-checks the wall deadline and the remaining budget first. `cleanup` failures pause for operator confirmation (the existing unreconciled-run guard on resume keeps this safe).

### Raw-sample aggregation; no p90 significance from 2
- `aggregate_samples()` computes median and nearest-rank p90 over raw client samples and reports `sample_count` and `significance` (`screening` below `PROMOTION_MIN_SAMPLES=10`, else `promotable`). `results()` now consumes `samples` (manifest case field, 1–12, default sets `H3_REPEATS`), validates every completed row (geometry/FPS/frames/AV-duration/finiteness unchanged), validates measured counters (`logical_forwards`/`scheduler_updates`/`block_calls`) **when the artifacts carry them** (absent stays absent — an expected work count is never treated as measured), and keeps diagnostics scope-separated (no client-seconds claim at all).

### Quality receipts bound to artifacts; H02 cannot unlock gates
- `review_receipt()` builds a human receipt (case id, artifacts path, exact output hashes, prompt, policy, criteria, typed `decision`) and binds it with a canonical digest. `apply_quality_receipt()` verifies the digest (stale/tampered receipts rejected), rejects any blocklisted identity (`EVIDENCE_BLOCKLIST = ("H02",)` — the invalid historical comparison), requires the cited artifacts and output hashes to match the ledger record, then records `quality_accepted`/`quality_rejected` carrying the evidence binding forward. A `quality_rejected` case (e.g. the observed LLLL seam) never satisfies a dependency — `readiness` counts only `measured`/`diagnostic` — and never auto-rejects other prompts. Output hashes remain evidence, not perceptual quality: only a human receipt changes quality, thresholds stay immutable in the approved manifest.
- `select()` rejects supervisor decisions citing blocklisted evidence ids before any other check.

### Supervisor bounded fallback; deterministic `--no-model` intact
- On an invalid/unreachable endpoint the campaign no longer stops: it records the rejection with `fallback=true` + consecutive-failure count, runs the deterministic queue head, and pauses only after `SUPERVISOR_FALLBACK_LIMIT=3` consecutive failures. Success resets the counter. `--no-model` (config=None) is untouched — it never consults the endpoint. `select()` still enforces the typed action schema; the model still cannot touch manifests, thresholds, source, or commands.

### R05–R08 adapters — registered only as implemented
New REGISTRY entries use **real env flags verified in each branch's code**, each behind a capability gate so the case stays blocked until the operator supplies evidence from the branch that implements it (on a checkout without the flag the env var is inert):
- `R05.NPUADALN.ON` → `SGLANG_H3_NPU_ADALN_MEMO=1` (gate `r05_npu_adaln`)
- `R05.SP8PAD.L16` → launcher `L16` (validation requires 16 devices; gate `r05_sp8_pad`)
- `R06.GRAPH.W0R` → `SGLANG_H3_VAE_NPU_GRAPH=1` (gate `r06_vae_graph`)
- `R08.SPARSE.H08` → `H3_ATTENTION_BACKEND=ascend_sparse_attn` with `H3_ATTENTION_CONFIG` bound to the hash-pinned capability file (gate `r08_sparse`)
- `R08.DISPATCHLOG` / `R08.MEMOSTATS` / `R08.FFNLOG` → diagnostics (`SGLANG_H3_ATTN_DISPATCH_LOG` / `SGLANG_H3_ADALN_MEMO_STATS` / `SGLANG_H3_SELECTIVE_FFN_LOG` at fraction 0.5)

Manifest validation was extended consistently (exact-field `samples`, L16 device count); `campaign.example.json` remains a valid v2 template — explicitly **not** pre-approved. R07's protocol layer has no device path on its branch, so no adapter is registered for it (honest gap, recorded below).

### Preflight addition
- `ports_available()` bind-tests both reserved ports before the campaign starts (the reservation locks alone did not prove bindability).

## 3. Tests and environment

`test_minimax_h3_r09_controller.py` — 14 host tests: nearest-rank aggregation (5-sample median/p90, single sample, 10-sample promotability), failure taxonomy, fingerprint coverage over all identity inputs, stale-fingerprint demotion + dependency blocking, reservation reconciliation (confirmed/unknown/outstanding), receipt bind/apply/reapply, tampered-stale rejection, wrong-hash rejection, unknown-case rejection, H02 blocklist (receipts and decisions), port bindability, adapter/capability registration, L16 device validation, `launch_environment` real-flag wiring (including the sparse config binding), and loopback-only supervisor config. Plus the pre-existing `test_experiment_controller.py` (8 tests, one synthetic case updated for the new required `samples` field; assertions unchanged).

Environment: CPU torch 2.15.0+cpu, host-only triton import shim (never committed). `runner.digest` is the canonical JSON digest used for fingerprints/receipts. Host execution is not hardware validation; the controller's device paths (`check_devices`, launcher, ports under load) remain device-pending exactly as in H09.

## 4. Per-case status (handoff deliverable)

- R01 cases (`R01.CLEAN4/PARITY/CROP/QUALITY/HIST`): implemented + host-tested (registered, launcher-backed, previously exercised H09-side). Device-validated: pending operator campaign.
- R05–R08 adapters: implemented (env flags verified in branch code) + host-tested registration/gating; device-validated: pending; quality-accepted: none (correctly — no device evidence exists).
- R07: no adapter — its branch ships a protocol layer without a device transport; registering a case would be an impossible-case manifest entry, which the handoff forbids. Recorded as blocked-by-transport instead.
- Small-model supervision: proposed-action loop host-tested via strict schema + fallback; no real endpoint exercised (loopback config validated only). `SUPERVISOR.md` updated with evidence semantics v2.

## 5. Honest gaps / pending device gates

- Ledger reconciliation uses launcher wall-clock as the allocation measure; per-device utilization accounting (busy vs reserved) is not yet derivable from artifacts — the reservation bound is the conservative truth.
- `results()` validates counters only when artifacts carry them; if the launcher's `denoise_matches.json` schema gains canonical counter names, validation should tighten (currently tolerant by design, not silent).
- ABBA paired ordering and held-out prompts are campaign-manifest concerns (ordering via `after`/`repeat_of`); the controller enforces budgets and uniqueness but does not schedule pair orders automatically.
- No real supervisor endpoint was exercised; the outage path is tested with raised `OSError`s (the type `urllib` raises on connect failure/timeout).
- The device campaign itself (order list in the handoff) remains operator-initiated; nothing here requests or performs runs.
