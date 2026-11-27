# Continuous MiniMax-H3 research campaigns

This directory implements the research harness and host-testable planning
primitives for the supplied handoff. It does **not** implement or validate all
H3 execution paths. The native runtime currently integrates static video FFN
compaction and NPU AdaLN memoization. Further research adapters plug into the
same campaign without changing the runner or reporting format.

## Layout and track coverage

| Track | Handoff IDs | Code available now | Remaining model/device work |
|---|---|---|---|
| Distributed Chunk Diffusion | A01–A03 | Native-aligned window plans, dependency-aware fixed-budget scheduling, ordered results, bounded concurrency, retries/cancellation | Native AV continuation, anchor generation and conditioning, global positions/noise, quality at seams |
| BADE | C01/C03 | Runtime static FFN compaction; shared experiment contract for further policies | Dynamic masks, smaller QKV/attention calls, quantized-shape acceptance and device profitability |
| RainFusion | B03 | Existing backend; precision/sparsity/schedule sweep driver | Verify SDK formats and dispatch, quality-matched tuning |
| BMKA | B04–B06 | Bounded candidate selector, mandatory-budget enforcement, FP32 LSE branch merge | External indices/LSE operator probes, representative construction, integration and quality |
| Reference Token Compaction | C02 | Reference-only gather preserving original positions and inverse IDs | Packed metadata rebuild, prune policy, per-layer cache isolation/error validation |
| Adaptive Temporal Resolution | E01/E02 | Candidate native-aligned RGB/latent geometry with physical timestamps | Checkpoint/AV/RoPE compatibility, actual latent reduction, supported interpolation |
| Sparse Context Parallelism | D01–D04 | Deduplicated remote KV demand unions | Public HCCL transport, scales/layout, overlap and rank balance |
| Asynchronous AV Decoding | A04 | Bounded producer/decode/output pipeline with backpressure | Native VAE temporal state/halo, device events and streams, AV synchronization |

`catalog.py` preserves all 22 experiment IDs, priorities and capability gates.
`prototypes.py` contains pure orchestration/PyTorch primitives. They are reusable
building blocks, not a new model backend. `runner.py` controls campaigns and
`probe_environment.py` records the installed environment/public signatures.
`test_research.py` checks scheduling, geometry, routing bounds, reference positions,
softmax composition, campaign resume, and comparison-contract rejection.

## Executable native H3 chunk paths

See [CHUNKS.md](CHUNKS.md) for `h3_chunks.py`: sequential RGB endpoint-conditioned
continuation and independent anchor-conditioned H3 requests on disjoint servers.
This executes native H3 HTTP jobs and muxes one AV output; latent-tail reuse and
global RoPE continuation remain unimplemented. Executed NFE is not exposed by the
HTTP API and remains unverified.

See the [H01 night agent handoff](H01_NIGHTLY.md) to recover saved A3 timings
and run one missing sample against the pinned deployment.

## Start a campaign

Run from the repository root, using the intended deployment Python environment:

```bash
python test/manual/minimax_h3_realtime/research/runner.py init /tmp/h3-suite.json
python test/manual/minimax_h3_realtime/research/runner.py plan /tmp/h3-suite.json
python test/manual/minimax_h3_realtime/research/probe_environment.py --output /tmp/h3-environment.json
```

Fill `contract` with the **actual** SoC, eight-device budget, immutable model,
FlashGen and decoder revisions, software versions, native FPS/canvas and measured
NFE. The initial file intentionally leaves unknown values null. Use separate
suites for native VAE and LightVAE; 28.33 s belongs to the LightVAE system.
Use separate suites for T2VA and reference modes. A model/mode/decoder change
must never appear as a same-contract speedup.

Each case has an experiment ID and distinct case ID, workload (prompt, seed,
references), duration, optional feature parameters, and an adapter `command`.
Clone cases to sweep prompt/seed/duration/parameters; give each a unique ID.
Native 5/7.5/10/15-second requests belong in baseline scaling; 30/60/120-second
requests require the validated long-video path. The experiment catalog includes
them without claiming monolithic H3 supports them.

```bash
python test/manual/minimax_h3_realtime/research/runner.py run /tmp/h3-suite.json \
  --output /tmp/h3-campaign --rounds 3 --track baseline rainfusion bade
python test/manual/minimax_h3_realtime/research/runner.py run /tmp/h3-suite.json \
  --output /tmp/h3-campaign --rounds 5 --track baseline rainfusion bade --resume
python test/manual/minimax_h3_realtime/research/runner.py report /tmp/h3-campaign
```

`--rounds N` means total numbered rounds 0…N−1, including on resume. Completed
measurements are skipped; failed/blocked cases are retried. Order reverses on
alternate rounds. For a final controlled comparison, select baseline/candidate
pairs and use at least ten repetitions. No background service or automatic
schedule is installed. Run more rounds explicitly or invoke the command from
your existing job scheduler.

Each run saves:

```text
manifest.json          exact suite, source SHA, dirty/untracked hashes, host packages
results.jsonl          append-only round/case status and result records
report.md              timing/RTF table, comparable baseline speedups, blockers
samples.csv            one row per measured sample, excluding warmups
round-NNN/case-TIME/    context, server/setup/client logs, sample JSON, media, evaluation
```

The source/config/capability-evidence fingerprint must remain unchanged on resume.
A feature implementation, new capability evidence, changed decoder, or changed
configuration starts a **new campaign output directory**. Reports are local only.
Keep large result directories outside the source tree. No best-result promotion
occurs automatically; all decisions start `inconclusive`.

## Model adapter contract

Commands are argv arrays executed without a shell. An adapter can call a local
pipeline, a server, or a track-specific scheduler. The runner passes:

- `H3_RESEARCH_CONTEXT`: JSON containing full contract, case, `contract_hash` and
  `workload_hash`. Read feature parameters and prompt/seed/references here.
- `H3_RESEARCH_RESULT`: unique per-sample JSON destination.
- `H3_RESEARCH_SAMPLE`: sample number, including warmups.

The adapter must wait for **completed generation and final AV output** and write:

```json
{
  "status": "completed",
  "actual_nfe": 4,
  "contract_hash": "copy from context after verifying deployed configuration",
  "workload_hash": "copy from context after verifying request inputs",
  "output_path": "/absolute/path/to/unique-output.mp4",
  "critical_path_breakdown_ms": {},
  "active_rows_and_padding": {},
  "selected_attention_blocks": {},
  "peak_hbm_per_rank": [],
  "hccl_bytes": null,
  "streaming_metrics": {},
  "known_limitations": []
}
```

Only the first five fields are mandatory. NFE must come from executed forwards,
not the requested step flag. Preserve raw metadata/profiler evidence. The runner
requires ffprobe and checks actual audio/video presence, native-aligned duration,
canvas/FPS, and audio duration agreement. Default duration tolerance is one
17-frame native interval; default AV duration tolerance is 0.1 s. Tighten or
explicitly justify changes per case. It records the output hash and full sample
JSON. It cannot detect repeated frames, correct lip-sync, or prompt adherence
from container metadata alone.

Timing encloses the complete adapter process. Keep model loading/server startup
outside it. Put anchors, boundary refinement, transfer, decode/interpolation and
mux inside it when they belong to generation. Returning only latents or an async
job ID is invalid. Use a fresh output per sample to preserve quality artifacts.

Optional `server_command` starts a foreground server before warmups and requires
`health_url` on an unused local port. The runner waits for readiness and stops
only its own process group. Omit it to connect to a managed server. Optional
`setup_command`/`teardown_command` prepare/clean the environment outside timing;
they are synchronous and their process groups are cleaned on exit. They must
not hide anchor generation or other per-request work. A case-level `timeout_s`
overrides the suite timeout (1800 s default) for server and adapter operations.

The earlier `benchmark_minimax_h3_realtime.py` remains the quick native
RainFusion/FFN sweep. This campaign runner is the broader interface for all
research tracks and verified media/quality artifacts. It does not pretend
unimplemented research algorithms are runnable variants of that quick sweep.

## Capability evidence and quality

A case with no adapter or missing capability proof is recorded `blocked` with a
reason. `plan` works before filling unknown contract fields. It never starts
servers or changes model execution. `run` validates the complete contract first.

Capabilities are explicit, never inferred from a function name/signature. Add:

```json
"capabilities": {
  "external_indices": {
    "status": "available",
    "device": "exact value from contract.hardware",
    "evidence_path": "/absolute/path/to/probe-result.json"
  }
}
```

The referenced JSON must contain `result: "pass"`, the same `device`, and
`software_hash` equal to SHA-256 of the sorted JSON encoding of
`contract.software` (use `runner.digest`). Record exact signatures, shapes,
versions, correctness/latency samples, logs, and whether a fallback occurred in
that evidence. The runner checks identity and fingerprints the evidence file;
the engineer remains responsible for the probe's correctness and scope.
`probe_environment.py` produces **unverified** inventory, not passing evidence.
Case `required_capabilities` adds stricter gates, e.g. unequal-Q/K attention for
query compaction or reference isolation for exact KV caching.

Optional `evaluation_command` runs after generation, outside latency timing.
It receives the context and `H3_RESEARCH_EVALUATION` output path and writes
`{"metrics": {"metric_name": 0.0}}` using real evaluation. Configure
`quality_thresholds` on the suite, e.g. `{"metric_name": {"min": 0.9}}`, using
thresholds agreed **before held-out testing**. Missing/nonfinite required metrics
fail the metric gate. With no thresholds quality stays pending. Passing metrics
is reported `metrics_pass`, not human quality acceptance or automatic promotion.
The command must compare against the matching saved baseline artifacts; include
those artifact paths and revision IDs in the case/evaluation output.

Keep calibration and held-out prompts disjoint. Review at least twenty held-out
prompts including difficult motion, faces, fine text, audio and dialogue. Oracle
anchor experiments use `classification: "oracle"`; existing interpolation/model
replacement experiments use `auxiliary`. Neither is reported as a primary
speedup. Preserve negative results. Compare combined winners explicitly.

For streaming, adapters should record first **playable AV** time, chunk commit
intervals, delivered FPS, queue high-water mark, stalls, and timestamp drift.
For sparse/context experiments, record actual selected links, GEMM M, demand
union sizes, HCCL bytes and exposed communication. Prototypes alone establish
none of these device results.

## Local verification

```bash
python -m pytest -q test/manual/minimax_h3_realtime/research/test_research.py
```

The campaign integration tests deliberately mock ffprobe/device observations.
They verify orchestration and rejection rules, not encoded media or real NPU
results. Run the real probes and campaign adapters on A5 before making claims.
