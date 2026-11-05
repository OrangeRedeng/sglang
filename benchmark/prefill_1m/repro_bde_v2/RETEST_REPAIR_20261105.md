# A3 retest repairs and controlled rerun

Evidence: `retest-nightly-20261105-071607`, source commit `5ba09eae3e`.
Ascend 910_9362 x16, full GLM5.2 ModelSlim W4A8, TP16/DCP16 (MegaMoE EP16),
CANN 9.1.0, torch 2.10.0+cpu, torch_npu 2.10.0.post6,
sgl-kernel-npu 2026.9.0, DeepEP 00db6cd6. Numerical quality was unchecked.
This remains a GLM5.2 checkpoint experiment; GLM5.3 checkpoint acceptance is separate.

## Findings from the completed run

| Case | Cold prefix TTFT (s) | Cached TTFT mean / P90 (s) | E2E run (s) | Prefill tokens/s |
|---|---:|---:|---:|---:|
| Baseline first |181.503|23.775 / 40.402|44.874|42060.70|
| MegaMoE 500K |60.984|11.107 / 18.864|20.924|45017.92|
| Bounded shared stream |180.836|23.689 / 40.277|44.646|42215.03|
| INT8 norm + fusion |211.403|26.721 / 43.036|47.357|37424.71|
| Combination + metadata reuse |207.777|28.418 / 44.835|49.130|35189.09|
| Baseline last |181.673|23.840 / 40.523|44.930|41947.22|

All completed runs show sampled prefill batch mean 1 and cached-prefill batch
mean 1. There are no decode-batch samples; TPOT is undefined at output length 1.
The 500K result is not comparable with 1M results. Baseline mean TTFT drift is
0.27%. The combined cases regress on cold preparation, cached TTFT, E2E and
throughput. These artifacts do not isolate a fusion kernel or metadata operation
as the cause; the new round includes the missing matched component controls.

Piecewise startup completed several shapes but failed with `HcclAllGather`,
`GetCaptureEventFromTask`, and a stream not joined to the capture stream.
The sparse-MoE quantized-input all-gathers and output reduce-scatter were still
reachable inside compiled FFN code. They now execute through explicit eager
boundaries. Gathered MoE inputs/operands are copied into stable two-slot buffers
before compiled compute, with that storage included in the graph reservation.
The `graph-piecewise-moe-boundary` marker is required by the graph case.
DCP compute capture synchronizes outstanding eager work before
opening each new native graph; this adds startup work, not replay synchronization.
Summary attribution retains the graph/HCCL failure instead of only exit -9.
Native changing-shape replay still needs A3 evidence.

MegaMoE 1M failed during the AISBench invocation with native `ConcatD` allocation
failure, reported at an asynchronous metadata operation. This does not prove
that the metadata operation itself exhausted memory. Growing packed output
buffers leave previous allocations cached by PyTorch while CANN/HCCL requests
memory outside that allocator. The serial path now reserves final output capacity
once, reuses one output slot, and budgets both retained output layouts plus
bounded generic scratch. EP additionally reserves 1 GiB of native headroom.
That headroom is a conservative starting allowance, not a measured native peak
or proof that the OOM is resolved. Keep native failure logs and memory evidence.

Both batch-2 launches obtained only 479232 virtual KV slots, versus a required
1017856, after reserving 10.997 GiB of runtime gathers. Both now disable gather
prefetch and SFA overlap, use the existing sequential generic fallback with
bounded scratch, and request the same 63616 local rows with mem_fraction 0.85.
The packed variant additionally opts into packed batch-2. Reservation accounts
for both layouts remaining resident across forwards and the two cache-decode
slots. Capacity checks remain strict; a configuration that cannot fit is still
capacity-blocked, with actual and required slots in its error.

## Controlled eight-hour round

`--round final --hours 8 --case-minutes 40 --repeats 1` now schedules:

1. Baseline 1M.
2. Piecewise-prefill DCP16 1M, decode graphs disabled.
3. MegaMoE no-lookahead 500K, same 1M context allocation.
4. MegaMoE no-lookahead 1M, only after case 3 completes with activation evidence.
5. INT8 norm/gate/shared conversion alone.
6. Fused MoE alone.
7. Norm + fused MoE.
8. Combination + cached-attention metadata reuse.
9. Combination + ordinary batch-2.
10. Matched packed batch-2.
11. Baseline 1M.

Eleven 40-minute ceilings total 7h20m, inside the 8-hour deadline. Use fresh
servers, one measured AISBench invocation per server, separate cold-prefix
preparation, 16 requests/concurrency 16, prefix99, one output token, context
1048576 and chunk32768. Batch-2 comparisons change only the packed-read opt-in;
compare their timings against each other before attributing differences to packing.

Artifacts retain resolved server arguments, launch environment, activation,
cold-prefix metrics, TTFT/P90, E2E and throughput. `batch-metrics-1.json` adds
prefill/decode batch means and sample counts over this AISBench invocation,
excluding startup/warmup. `mean_cached_prefill_batch` includes sampled prefill
entries with positive cached-token counts; the overall mean includes cold-prefix
preparation. These are sampled scheduler statistics, not complete execution traces.

Run on the A3 host (the user operates hardware tests):

```bash
(
set -euo pipefail
REPO=/home/swx1199799/megamoe_test/sglang
RUN_ID=$(date +%Y%m%d-%H%M%S)
LOG="/tmp/a3-repair-$RUN_ID.log"
docker exec -w "$REPO" sglang-a3 git -c \
  'url.https://ghfast.top/https://github.com/.insteadOf=https://github.com/' \
  fetch https://github.com/OrangeRedeng/sglang.git glm52-a3-optimizations
docker exec -w "$REPO" sglang-a3 git show \
  FETCH_HEAD:benchmark/prefill_1m/run_a3_nightly.py > /tmp/run_a3_nightly.py
python3 /tmp/run_a3_nightly.py --stop-only
docker exec -w "$REPO" sglang-a3 git merge --ff-only FETCH_HEAD
screen -dmS "a3-repair-$RUN_ID" bash -c \
  'exec python3 /tmp/run_a3_nightly.py --round final --hours 8 --case-minutes 40 --repeats 1 > "$1" 2>&1' \
  _ "$LOG"
printf 'Progress: %s\n' "$LOG"
)
```

Acceptance requires A3 changing-shape graph replay, MegaMoE 1M completion,
actual two-request/packed activation, sufficient KV slots and stable memory.
Repeat any promising comparison before claiming a speedup. Host validation
covers boundary placement, arithmetic/storage, capacity construction and reporting;
it cannot establish native replay, HCCL stability, quality or performance.

Local validation: 106 tests and 140 subtests passed across the graph, DCP buffer,
packed batch-2, runner/qualification and residual suites. The existing
socket-dependent `worker_disconnect` test was excluded in this sandbox.
Python syntax, fatal lint, whitespace and launch-shell syntax passed; the
11-case dry run fits the configured eight-hour deadline. No A3 execution was
performed locally. Native memory/replay acceptance and performance attribution
remain pending the user-operated rerun.
