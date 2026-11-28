# R06 — NPU Graph for LightVAE, AV overlap and output pipeline: report

**Branch:** `feat/minimax-h3-r06-vae-graph-pipeline` (from R01 `68e373c50c47492e5860e2d18a9599133ea9c754`)
**Date:** 28 November 2026 · code phase · host tests only, no device runs
**Scope:** real opt-in graph capture/replay for the tiled video-VAE decode, the NPU audio side-stream overlap, and the capability probes. Per assignment R06 branches from R01 (not R05); the R05 DiT/comm ports are intentionally absent here. No new kernels; #52052-style output assembly and the DiT graph stay deferred pending device evidence.

## 0. Baseline audit

- The NPU VAE already caps `decoder_tile_batch <= 2` and `window_batch == 1` with the fast-path gate registered (`MiniMaxH3VideoVAE.__init__`); exact quality keeps one tile. The R06 code does **not** change these caps (handoff item 1: "don't bypass exact-mode restrictions"; batch4/window2 was **not** added — memory/temporal evidence is pending).
- `_overlaps_audio_decode` was CUDA-gated; H04's streamed-encoder callback guard remains CUDA-only (untouched).
- The tile decode runs `forward_many(local, forward=forward_fn, capacity=...)`: one stateless `forward(x)` per stacked tile group; gather/blend/halo/temporal state live outside it — the safe capture unit.

## 1. Commit map

| Commit | Files |
|---|---|
| `Add opt-in NPU graph capture/replay for the H3 tile decoder` | `runtime/models/vaes/minimax_h3_vae_npu_graph.py` (new), `runtime/models/vaes/minimax_h3.py`, `runtime/models/vaes/minimax_h3_lightvae.py` |
| `Allow the NPU audio side stream under the resident-audio gate` | `.../minimax_h3/stages/decoding.py` |
| `Add the R06 graph probe and host tests` | `research/r06_vae_graph_probe.py` (new), `test/unit/test_minimax_h3_r06_vae_graph.py` (new), `research/R06_REPORT.md` (this file) |

## 1.5 Test results (host, 28 Nov)

Executed on this host with CPU torch 2.15.0+cpu and a `triton` import shim
(`/tmp/h3_host_shim`, never committed — the chain only needs Triton importable;
no kernel executes on host). Host execution is **not** hardware validation;
see `../../../../../R05_R06_DEVICE_VALIDATION.md` for the operator procedure.

- `test_minimax_h3_r06_vae_graph.py`: **12 passed** (0 failed, 0 skipped).
- Regression: `test_minimax_h3_dit_contract.py` **39 passed, 4 failed, 4
  skipped — identical on the pristine R01 base in this environment** (the
  PDD/AdaLN graph-replay failures need device support absent here). Zero
  regressions attributable to R06.
- Lint: `ruff check` clean on all six touched files.

## 2. Per-subtask status

### V2/V3 LightVAE tile-decoder NPUGraph — implemented, host-tested contract, device-pending
New `minimax_h3_vae_npu_graph.py`:
- `TileDecodeGraphCache` wraps the per-group decoder forward: on first sight of a `(shape, dtype, device, decoder-revision)` bucket it clones the input into a static buffer, warms up outside the pool, captures with `torch.npu.graph(graph, pool=shared_handle)`, verifies the captured output contract against warmup, and stores the entry. Replays copy new bytes into the static input, replay, and return a **private clone** of the static output (ownership: no caller sees a buffer the next replay overwrites).
- One shared graph pool per cache (`graph_pool_handle`), entries bounded by `DEFAULT_MAX_ENTRIES=8`; overflow runs eager. `revision()` is `(graph_epoch, decode_dtype_cache_tag)`; the epoch bumps and the cache clears when `prepare_decoder_autocast_weights` recasts weights and when `load_minimax_h3_lightvae_decoder` swaps the decoder.
- Fail-closed gates before any capture attempt: opt-in env `SGLANG_H3_VAE_NPU_GRAPH=1`, `npu_graph_api_available()` (torch.npu `graph`/`NPUGraph`/`graph_pool_handle`/`current_stream`), NPU device tensor, inference mode, contiguous input. Any capture/replay exception logs once and runs the unwrapped forward; counters (`captures/replays/eager_calls/capture_failures/invalidations`) make the runbook honest about which path actually served.
- Explicitly **not** claimed: capture time, replay byte parity with eager, and graph-pool memory are device-pending (cases V2/V3). API import is never treated as working capture.
- The cache wraps only `forward_fn is self.decode` on NPU tiles of an eval-mode VAE; encode, blend and windowing stay untouched, so temporal/halo state cannot be captured accidentally.

### V4 audio/video decode overlap — implemented (NPU gate + device-agnostic stream), device-pending
- `_overlaps_audio_decode` now accepts NPU under the same conditions as CUDA: resident `audio_vae` residency mode and decode world > 1.
- `_start_owner_audio`/`_join_owner_audio` resolve the device module via `torch.get_device_module(audio.device.type)` (TorchNPU mirrors the CUDA stream API): side `Stream(priority=-1)`, `wait_stream`, `record_stream`, device switch and the worker-thread stream context all route through it. No NPU-specific branching beyond device type.
- The H04 streamed-encoder callback stays CUDA-only (handoff requirement). Audio VAE graph capture is explicitly **not** added (handoff: separate only after the video path is proven).
- Device-pending: actual overlap/resource contention (`max(audio, video)` vs sequential), AV sync and stream/event safety on Ascend; standalone eager control first (case V4), graph interaction only after.

### V5 output pipeline — deferred
No code. The bounded double-buffered gather/blend/D2H/encode assembly (#52052 idea) is scoped after V4 evidence; the existing EncodeWhileDecoding CUDA path is unchanged.

### V7 DiT graph — deferred (conditional per handoff)
No capture attempted: R06 branches from R01, so R04 trace/R05 collective geometry is absent by construction. Revisit only on an integration branch after R04/R05.

### Stage accounting (handoff item 1) — probe/runbook only
`research/r06_vae_graph_probe.py` reports torch.npu graph/stream API availability with explicit blocked/available verdicts; per-stage VAE timing remains the operator campaign's job (recover `vae/` probes from the night archive). No E2E deltas are claimed.

## 3. Tests and environment

`test_minimax_h3_r06_vae_graph.py` — 12 CPU tests: disabled-by-default wrap identity; unavailable-API eager fallback; capture-once/replay-per-bucket with per-replay input bytes reflected and private output copies (mutation isolation); bucket keys tracking shape and decoder revision; max-entries bound with overflow eager; capture-failure fallback counted in both `capture_failures` and `eager_calls`; explicit invalidation; non-contiguous and grad-mode eager fallback; VAE/loader wiring and invalidation sites; NPU overlap gate (resident + world > 1); probe host-runnability. The torch.npu capture path itself is exercised through the documented seams (device gate, API probe, capture hook) with an in-place-replay fake — the real capture/replay bytes stay device-pending.

Regression: `test_minimax_h3_dit_contract.py` **39 passed, 4 failed, 4 skipped — identical on pristine R01 base in this environment** (graph-replay tests need device support absent here). Zero regressions attributable to R06. Lint: `ruff check` clean on all six touched files.

Environment: CPU torch 2.15.0+cpu with a host-only `triton` import shim (`/tmp/h3_host_shim`, never committed); see the device-validation handoff for the operator-side procedure.

## 4. Feature-off behavior

With `SGLANG_H3_VAE_NPU_GRAPH` unset the cache is never created as enabled, `wrap` returns the original callable, and the decode path is byte-for-byte R01. With the flag on, every gate miss or exception still runs the unwrapped forward. The audio overlap stays off on NPU unless residency mode is resident and world > 1 — same predicate as CUDA.

## 5. Operator prerequisites and pending device gates

1. Probe first on the A3 host: `python3 test/manual/minimax_h3_realtime/research/r06_vae_graph_probe.py`.
2. Cases (handoff §Cases): V0 isolated eager same-latent decode control → V2 graph/eager parity (decoded frames bit-compare + pre-encoding outputs) → V3 warmed replay timing + peak graph pool (report capture/startup separately from warmed replay) → V4 audio side stream (resident audio; eager control first) → V6 combined only after each passes. Tile batch4/window2 only if V0-V3 memory fits and temporal semantics hold; exact mode unchanged.
3. Record per-stage timings and peak graph-pool bytes; never infer VAE speedup from full-E2E batch deltas (H04 lesson).
4. If capture is unsupported on the installed TorchNPU, the probe's blocked verdict is the report; the code stays eager — that is a supported outcome, not a failure.

## 6. Runbook snippet

```bash
# host tests
PYTHONPATH=python python -m pytest python/sglang/multimodal_gen/test/unit/test_minimax_h3_r06_vae_graph.py -q
# capability probe (operator host)
python3 test/manual/minimax_h3_realtime/research/r06_vae_graph_probe.py
# opt-in graph capture on device
SGLANG_H3_VAE_NPU_GRAPH=1 <existing A3 launch command>
# counters land in the worker log: captures/replays/eager_calls/capture_failures/invalidations
```
