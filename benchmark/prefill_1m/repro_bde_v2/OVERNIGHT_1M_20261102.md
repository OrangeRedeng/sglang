# A3 overnight queue: 8–10 hour window

## Verified short baseline

Source: `.codex-artifacts/res/retest-quick-20261102-043525`, commit
`3eb2d81368a8d88252ba15f9074e33c6c5505b79`.
Serving qualification and one AISBench repetition passed; Conda startup worked.

| Metric | Measured result |
|---|---:|
| Requests / requested input / output | 2 / 1024 / 16 tokens |
| Actual input including chat framing | 1037 tokens average |
| TTFT mean / P90 | 0.742266 / 0.993871 s |
| TPOT mean / P90 | 0.305108 / 0.321908 s |
| Mean request E2E / whole measured run | 5.318882 / 5.319730 s |
| Requests/s / AISBench prefill tokens/s | 0.376 / 1397.0732 |

Runtime: GLM5.2 W4A8, Ascend910_9362 x16, CANN 9.1.0, torch 2.10.0+cpu,
torch_npu 2.10.0.post6, sgl-kernel-npu 2026.9.0, eager TP16/DCP1, chunk 4096,
local KV cap 8192. Requested shared prefix was 0.5, but AISBench warned that
cache metrics were unavailable: this is not verified 50% cache reuse.
This short repetition establishes neither 1M acceptance nor speedup.

## Time calculation

Evidence: the collected October 31 serving report totals **16,494.969 s =
4 h 35 min** for its 39 cases. Typical nonbaseline qualification took
**378 s**; baseline qualification took **691–698 s**. The earlier handoff's
1M **cold-prefix time was about 182.6 s**. Its 24.589 s mean belongs to a
cached 16-request phase and cannot estimate cold prefill.

Planning uses **240 s per 1M cold prefill** (31% above the historical value),
**15 s per single cached request**, 380 s per candidate qualification and
300 s per timed-server startup. These are planning allowances, not current-head
1M measurements. No concurrency speedup or early test failure is assumed.

| Work | Planned seconds | Planned time |
|---|---:|---:|
| Twelve previously timed baseline/MoE/indexer qualifications | 5014 | 1 h 24 min |
| Reserve: MegaMoE (900), two MoE combinations (450 each), optional FuseEP 1/2 (900 each) | 3600 | 1 h |
| Seven 1M variants, 2645 s each | 18515 | 5 h 9 min |
| Two long-reference preparations | 640 | 11 min |
| Host/NPU preflight and unit groups | 180 | 3 min |
| **Total** | **27949** | **7 h 46 min** |

Per long variant: 380 qualification + 300 startup + 960 cold work
(one 240 s warmup + three 240 s measured requests) + 1005 cached work
(one 240 s warmup + three repetitions of 240 s prefix priming + 15 s measurement).

This leaves about **2 h 14 min** inside the 10-hour ceiling. Missing W8A8
checkpoints or failed parity may shorten the run, but the plan does not rely
on that. Old timings are imperfect predictors of the new head/long geometry.

A **9.5-hour global execution deadline** leaves 30 minutes for cleanup and
archiving. A qualification needs 15 minutes remaining to start; a complete long
case needs 50 minutes. Otherwise it is recorded as `budget_blocked`. The host
also bounds its worker wait by the remaining global budget and closes owned
workers on expiry. This is a time limit, not a guarantee against an external
Docker/device hang.

## Selected coverage

The focused qualification phase contains **17 scenarios**: baseline; routed
scale, fused GMM2/finalize, prequantization, shared fusion and shared activation;
multistream; automatic stream thresholds, automatic HCCL and joint autotuning;
INT8 indexer and query sharding; FuseEP 1/2 and MegaMoE; and two MoE combinations
(stream + automatic thresholds/HCCL, and routed scale + prequant + shared fusion
+ shared activation). Independent NPU INT8-indexer and MoE device tests also run.

Exactly **seven** matched 1M configurations are queued:

1. DCP baseline, first.
2. Packed read.
3. Gather prefetch.
4. SFA gather overlap.
5. Query-sharded indexer.
6. Multistream MoE under the same DCP geometry.
7. Packed read + prefetch + overlap together.

The triple combination requires each of its three independent components to
qualify, then must pass its own unchanged parity tolerance and all activation
markers. The MoE combinations have the same component gates. Failed, inactive,
smoke-only, absent and dependency-blocked components never become timing passes.
FuseEP 1/2 require a compatible explicit `W8A8_MODEL_PATH`; W4A8 remains rejected.

Removed from this time-limited queue: the full 51-case serving rerun, the
50-configuration long sweep, all 32K timing points, concurrency-2 long points,
multiple requests per long repetition, repeated warmups, BF16-reference online
quantization variants, graph/chat sweeps, SFA INT8 smoke-only timing, legacy
residual/full-width-Q/streamed-o_proj/BDE timing, and additional I/O pair/stack
permutations. Their definitions remain available for a later focused run.
No MXFP or multi-request packed-read optimization is added.

## Measurement contract

All long variants use full weights, **TP16/DCP16**, eager execution, context
limit 1,048,576, chunk 16384, prefill-request cap 1, running-request cap 64 and
local KV cap 81920. Actual global capacity plus 4096-token headroom is checked;
insufficient memory is reported without shrinking the workload.

Each cache regime has **one untimed warmup**, then **three matched repetitions**,
one request at concurrency 1, **exactly 1,000,000 input IDs**, one output token.
Every repetition flushes; cached repetitions additionally prime a fresh 99%
prefix. Returned prompt and cached-token counts are verified. Seeds match
across variants. Cold and cached results remain separate. Native `/generate`
measurements avoid AISBench chat-template token overhead.

Output includes TTFT mean/P90/P95, E2E mean/P90, run wall time, requests/s,
logical prompt and actual uncached-prefill throughput, all samples and output
text. TPOT is null with one output token. Scheduler batch means are null because
this client does not expose them; raw server logs are retained. Memory-state
snapshots and qualification model-load peaks are retained, not mislabeled as
continuously measured runtime peaks. Resolved arguments, checkpoint/source
identity and raw artifacts remain reviewable.

## Deploy and start on the A3 host

Transfer the prepared `glm-a3-overnight-1m.tar.gz` to host `/tmp`. It overlays
benchmark files on the existing `3eb2d81368` checkout, without dependency changes.

```bash
REPO=/home/swx1199799/megamoe_test/sglang
docker exec sglang-a3 git -C "$REPO" log -1 --oneline
docker cp /tmp/glm-a3-overnight-1m.tar.gz sglang-a3:/tmp/glm-a3-overnight-1m.tar.gz
docker exec -w "$REPO" sglang-a3 tar -xzf /tmp/glm-a3-overnight-1m.tar.gz
docker cp "sglang-a3:$REPO/benchmark/prefill_1m/start_a3_retest.sh" /tmp/start_a3_retest.sh
screen -dmS a3-overnight bash /tmp/start_a3_retest.sh overnight
```

Export `W8A8_MODEL_PATH` first only if a real compatible checkpoint exists.
The existing lock prevents a duplicate driver; unrelated servers are not killed.
`overnight-core` keeps the MoE/indexer qualifications but drops the two long
query-sharding and multistream variants, saving roughly **1 h 28 min**.

```bash
RESULTS=$(ls -dt /home/tbaydasov/aisbench/overnight-results/retest-overnight-*/ | head -n 1)
tail -f "$RESULTS/driver.log"
```

The host summary updates after each stage. While the long stage runs:

```bash
docker exec sglang-a3 bash -lc 'RUN=$(ls -dt /tmp/sglang-bde-night-*/ | head -n 1); tail -n 40 "$RUN/long/feature.log"'
```

Each repetition writes a `long-*-samples.json` file. Final `results.json`,
`SUMMARY.md`, detailed reports and the `.tar.gz` retain partial results and
budget-blocked cases. A nonzero exit does not invalidate independent passing
measurements; it also does not turn the full queue into a pass.

Local CPU checks validate orchestration and workload contracts only. This
prepared queue has not been executed on A3.
