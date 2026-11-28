# R07 — GLM-inspired distributed AV windows: code-phase report

**Branch:** `feat/minimax-h3-r07-dist-av-workers` (from R01 `68e373c50c47492e5860e2d18a9599133ea9c754`)
**Date:** 28 November 2026 · code phase · host tests only, no device runs
**Scope:** the P0 feasibility code and the minimal G2 worker/coordinator protocol per the assignment. G5 (latent-tail continuation) and G6 (long-video/streaming) were **not** started, per the handoff ordering (G0–G2 first). No new disagg framework; no throughput claim as latency.

## 0. Baseline audit

- R01's `windowing.py` runs the two-window policy as a *sequential* in-process loop (`WindowPredictions.predict`): per window, build `forward_kwargs` from the branch, run one forward, then FP32 `_blend`. The coordinator module generalizes exactly this shape: per step, per window, one batch-1 forward over an immutable snapshot; coordinator-side FP32 fusion; one scheduler update. Nothing in the existing loop is duplicated — `distributed_windows.py` is the protocol layer callers plug the existing forwards into.
- The device budgets in the handoff (16 vs 2×8 A3, 8 vs 2×4 A5) are represented as topology *plans*, never as claims: subgroup collective layouts and per-group HBM fits are operator-campaign facts.

## 1. Commit map

| Commit | Files |
|---|---|
| `Add step-major distributed window coordinator and workers` | `.../minimax_h3/distributed_windows.py` (new) |
| `Add the R07 topology/transport probe and protocol host tests` | `research/r07_topology_probe.py` (new), `test/unit/test_minimax_h3_r07_distributed_windows.py` (new), `research/R07_REPORT.md` |

## 1.5 Test results (host, 28 Nov)

- `test_minimax_h3_r07_distributed_windows.py`: **12 passed** (0 failed,
  0 skipped). CPU torch 2.15.0+cpu with a host-only triton import shim
  (`/tmp/h3_host_shim`, never committed). Host execution is **not** hardware
  validation.
- Regression: `test_minimax_h3_dit_contract.py` **39 passed, 4 failed, 4
  skipped — identical on the pristine R01 base in this environment** (the
  PDD/AdaLN graph-replay failures need device support absent here). Zero
  regressions attributable to R07.
- Lint: `ruff check` clean on all three touched files.

## 2. Protocol (exact)

1. **Ownership.** `DistributedWindowCoordinator` owns the request id, the step counter and the scheduler callback. The caller owns the global `(video, audio)` state; the coordinator never mutates it — steps consume `WindowDispatch` snapshots the caller built as clones.
2. **Dispatch.** One `WindowDispatch` per window per step: `(request_id, step, WindowContract(index, video_shape, audio_shape), video_snapshot, audio_snapshot)`. Snapshot clones make in-flight windows immune to coordinator-side state mutation (worker-mutation isolation is tested).
3. **Execution.** Workers implement `submit(dispatch) -> Future` + `cancel()`. `LocalWindowWorker` runs the window's forward inline (the G2 numerical-reference path: predictions equal direct calls, so fused runs diff exactly against the sequential loop). `RemoteWindowWorkerStub` executes through a caller-supplied executor and is explicitly a **stub** — it does not claim a working bulk-latent transport.
4. **Validation.** Every prediction is checked against its window contract (shape) and the expected window set. Duplicate window indices in a dispatch list are rejected up front; a repeated step is rejected (`already completed`); a worker failure fails the step; a missing window fails the step. Every failure path leaves the scheduler un-stepped and cancels in-flight futures.
5. **Fusion + scheduler.** On a complete set, `fuse(predictions ordered by window index)` runs coordinator-side (the two-window FP32 ramp blend of `windowing.py` is the reference callable), then `scheduler_step()` runs **exactly once** — enforced and tested (a repeated step cannot double-step; failures never step).
6. **Timeout/cancellation.** Per-step budget via injectable clock; expiry raises `WindowStepTimeout`, counts the timeout, cancels pending futures and aborts only that step's in-flight work (worker stays usable — tested with the remote stub).

Counters (`CoordinatorCounters`) record dispatched/completed/duplicates/shape-rejections/worker-failures/cancellations/timeouts/scheduler-updates so a device run can prove which path actually served.

## 3. Topology and transport probes (G0/G1)

`r07_topology_probe.py` (host-runnable, stdlib core with explicit degraded verdicts):

- **Transports**: `zmq` (control-plane; present in the repo deps), `mooncake_transfer_engine` (bulk tensors; **unavailable here, unverified on Ascend** — this is the recorded blocker for any real remote worker), `torch`/`torch.npu` availability and device count.
- **Topology plans** for the fixed budgets via `plan_topologies(devices=…)`: serial `[16]` vs concurrent `[8, 8]` (A3 16), `[8]` vs `[4, 4]` (A3 8 / A5 8), `[4]` vs `[2, 2]` (A5 4). The concurrent entry carries `verified: False` and the critical-path model `max(T_A, T_B) + transfer + fusion + barrier` against the serial baseline `T(A) + T(B)` — no free ÷2 latency is implied anywhere.
- **G1 runnable plan**: measurement steps for serial vs two disjoint concurrently running groups (subgroup-collective proof, per-group HBM snapshot, fabric contention counters), plus G0 `npu-smi` / `torch.npu.memory_summary` commands.

Explicit handoff constraints honored: no new workers before the probe shows benefit; no process-group choreography beyond what the operator proves; no LLLL forcing (the coordinator is policy-agnostic — `fuse` receives whatever windows the policy selects); GLM's 17× throughput figure appears nowhere as a latency prediction.

## 4. Global-step placement (G3) — planning only

The three handoff options are represented in the topology plans and this report, not in code: (i) one group runs the full global step while the other idles (always feasible, no code needed); (ii) a fixed compatible larger collective layout with subgroups — blocked on the G0/G1 subgroup proof; (iii) compact global-context windows (R03 policy) — out of scope here since R07 branches from R01 and R03 is a separate branch. If quality demands GLLL/GLLG, the full global cost is accounted in the serial baseline; nothing in the code prefers LLLL.

## 5. Tests and environment

`test_minimax_h3_r07_distributed_windows.py` — 12 CPU tests: two-window completion with exactly-once scheduler; local-worker equivalence to direct forwards; snapshot immutability against a mutating worker; missing-prediction failure without scheduler update; wrong-shape rejection; repeated-step rejection; empty-dispatch rejection; step timeout (hung worker, injectable clock) without scheduler update; worker-exception failure; remote-stub-over-thread-pool protocol equivalence plus cancel-then-reuse; multi-step counter totals; topology plan shapes for 16/8/4/2 budgets.

Environment: CPU torch 2.15.0+cpu, host-only triton import shim (never committed). Regression: `test_minimax_h3_dit_contract.py` **39 passed, 4 failed, 4 skipped — identical on the pristine R01 base in this environment**; zero regressions attributable to R07. Lint clean on all three touched files.

## 6. Pending device gates and honest gaps

- G0: per-worker HBM (weights, KV/activation/graph pools, HCCL, encoder, VAE) on real A3/A5 — probe scaffolding only; the nightly ~62.75 GB/16-A3 figure means smaller replicas are **not** automatically feasible.
- G1: subgroup collectives + fabric contention + the critical-path measurement. Until then the concurrent topology stays `verified: False` and the code has no device worker — by design.
- G2 on device: wire `LocalWindowWorker`'s callable to the real per-window forward from `windowing.py` and compare same policy/seed predictions and final latents against the sequential loop (numerical tolerance only if group reduction changes rounding).
- G4: stage overlap/shared prep — not started (design notes only: cache only proven-invariant conditioning; never treat shared text/ref tokens as invariant KV).
- Real transport: Mooncake-on-Ascend capability measurement is the blocker; until it exists, `RemoteWindowWorkerStub` is the only remote shape and no cross-rank tensor movement is claimed.
- G5/G6: not started (handoff ordering).

## 7. Runbook snippet

```bash
# host tests
PYTHONPATH=python python -m pytest python/sglang/multimodal_gen/test/unit/test_minimax_h3_r07_distributed_windows.py -q
# capability/topology probe (operator host)
PYTHONPATH=python python test/manual/minimax_h3_realtime/research/r07_topology_probe.py
```
