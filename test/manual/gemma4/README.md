# Gemma 4 two-NPU throughput experiments

For the fixed random C24 workload and recovery of the earlier ~9.3req/s
two-NPU result, use [recover_throughput.md](recover_throughput.md).
It provides explicit settings, matched MTP controls and repeated-run comparisons.

Target: recommendation **21.75 completed requests/s**, RAG **8 completed requests/s**,
for all four cases each, with total concurrency below 30. Preserve checkpoint
precision, datasets, output lengths and scenario latency SLOs. No draft model.

Batch-level FIA length-list reuse and hoisted BSND views are independently
opt-in; both default off for controlled regression comparisons. The existing opt-in serial QKV fusion remains available. Experimental
tiled QKV and RoPE/KV fusion were removed after target-NPU launch crashes.

## Current experiment order

Use the [three-launch Docker plan](docker_ab.md):

The helper currently defaults to one NPU2, TP1, without DP for debugging.
Set `GEMMA_DEVICES=2,3` for two TP1 replicas. The full scenario matrix below
retains its two-NPU target and configuration.

| Order | Profile | Comparison |
| --- | --- | --- |
| 1 | `metadata` | Reference, serial QKV, existing graphs and delayer |
| 2 | `mtp-baseline` | Serial QKV, graphs/overlap off, delayer0.5/50ms/4/cap6, memory0.82 |
| 3 | `mtp` | Same as2 plus the matching assistant |

Run three unprofiled repetitions per launch with the same random C24/RPS87/1600
workload. Compare MTP against its matched eager control. Validate fixed-prompt
greedy outputs/logprobs and long-prefix SWA/shared-KV behavior first. Report
per-replica batches alongside TTFT, TPOT, E2E and completed throughput.

The supplied serial validation log passes108 cases. This does not establish
serving performance or full scenario acceptance. Do not rerun removed tiled
or RoPE/KV candidates. `run_plan.py` covers the original no-assistant matrix;
MTP remains a separate comparison, requiring its own full scenario acceptance.

For opt-in NPU target-verify and frozen-draft graphs, see the
[MTP graph trial](mtp.md#npu-graph-trial). Eager profiles remain available.

## Isolate serial QKV normalization

`SGLANG_ENABLE_NPU_GEMMA4_QKV_NORM` is independent of both FIA switches. Set it
explicitly before your direct launch or `serve_ab.sh metadata`:

| Value | QKV normalization |
| --- | --- |
| `0` | Native separate Q/K/V normalization |
| `1` | Existing serial fused normalization |

The Python runtime defaults to0. The serving helper preserves its historical
reference default of1, but now honors an explicit environment override. The
`native` profile always selects0. Keep both FIA flags0 when isolating QKV.
Do not change graph, scheduling, memory or workload settings between these runs.

```bash
SGLANG_ENABLE_NPU_GEMMA4_QKV_NORM=0 \
SGLANG_ENABLE_NPU_FIA_METADATA_REUSE=0 \
SGLANG_ENABLE_NPU_FIA_KV_VIEW_REUSE=0 \
bash test/manual/gemma4/serve_ab.sh metadata

# Stop the preceding server first.
SGLANG_ENABLE_NPU_GEMMA4_QKV_NORM=1 \
SGLANG_ENABLE_NPU_FIA_METADATA_REUSE=0 \
SGLANG_ENABLE_NPU_FIA_KV_VIEW_REUSE=0 \
bash test/manual/gemma4/serve_ab.sh metadata
```

## Isolate FIA preparation changes

The following expert switches are disabled by default and read at backend
initialization. Restart serving between tests. Keep the same source, runtime,
launch command, workloads and warmup; run three unprofiled repetitions each.

| Test | Serving command |
| --- | --- |
| Original preparation paths | `SGLANG_ENABLE_NPU_FIA_METADATA_REUSE=0 SGLANG_ENABLE_NPU_FIA_KV_VIEW_REUSE=0 bash test/manual/gemma4/serve_ab.sh metadata` |
| Metadata reuse only | `SGLANG_ENABLE_NPU_FIA_METADATA_REUSE=1 SGLANG_ENABLE_NPU_FIA_KV_VIEW_REUSE=0 bash test/manual/gemma4/serve_ab.sh metadata` |
| KV view reuse only | `SGLANG_ENABLE_NPU_FIA_METADATA_REUSE=0 SGLANG_ENABLE_NPU_FIA_KV_VIEW_REUSE=1 bash test/manual/gemma4/serve_ab.sh metadata` |
| Both changes | `SGLANG_ENABLE_NPU_FIA_METADATA_REUSE=1 SGLANG_ENABLE_NPU_FIA_KV_VIEW_REUSE=1 bash test/manual/gemma4/serve_ab.sh metadata` |

The metadata switch controls CPU query/KV length-list caching, reuse of replay
length lists, and removal of the duplicate sequence-length reduction. The view
switch hoists paged BSND KV views out of the per-request loop. With both off,
these preparation paths follow the pre-optimization implementation; this does
not assert that an unidentified earlier benchmark commit has been reproduced.
Serial QKV, long-prefix masks, MTP SWA correctness and graph settings are retained.
The `metadata` profile name remains for compatibility; it does not enable these
new switches. The matrix records their inherited environment values.

## Prerequisites

Run on the existing NPU host/environment. Stop the previous server on NPUs 2/3
first: the matrix runner starts its own server and refuses an occupied port.
It never checks out branches, changes datasets or installs packages. Keep the
server checkout clean so each result has a reproducible commit.

The external `bench_updated.py` and its existing `datasets/recommendation` and
`datasets/rag` directories are required. Each case needs 256 distinct measured
prompts and 32 separate warmup prompts. Requests, completions, fixed-length
outputs and the original input/chat rendering are preserved. The runner hashes
all inputs and the benchmark script and rejects changes during a run.

Set paths once (adjust the benchmark/dataset directory if different):

```bash
GEMMA_ROOT=/home/nnai/modified_sglang/sglang
GEMMA_BASE=/home/nnai/modified_sglang/sglang-gemma4-baseline
GEMMA_BENCH=/home/nnai/gemma-31B/bench_updated.py
GEMMA_DATA=/home/nnai/gemma-31B/datasets
GEMMA_MODEL=/home/test1/models/gemma4-26B-A4B-w8a8
GEMMA_RUNS=/home/nnai/gemma4-experiments
```

## Round 0: real two-NPU baseline

Create the pinned baseline checkout once:

```bash
git -C "$GEMMA_ROOT" worktree add --detach "$GEMMA_BASE" c8d7020179
```

Run the tool from the updated checkout, launching the old baseline:

```bash
python3 "$GEMMA_ROOT/test/manual/gemma4/throughput_matrix.py" \
  --phase baseline --server-root "$GEMMA_BASE" \
  --bench-script "$GEMMA_BENCH" --datasets "$GEMMA_DATA" \
  --model-path "$GEMMA_MODEL" --output "$GEMMA_RUNS/baseline"
```

This runs C24/C28/C29, all eight cases once per concurrency, with exact decode
buckets 12/14/15, TP1/DP2, queue1, 8192-token budgets, memory fraction0.92 and the
0.5/50ms/4-pass/cap6 delayer. Both DP worker identities and effective backend
settings are checked. Native server args, versions, NPU topology, checkpoint
metadata hashes, commands and complete raw benchmark results are saved.

One-run results are exploratory. `base/results.json` contains independent strict
SLO checks; the external script's 98% rate allowance is not used. Sampled batch
means include the external script's warmup traffic and are separate from complete-benchmark latency and throughput.

## Round 1: metadata candidate

Use a new output directory and change only `--server-root`:

```bash
python3 "$GEMMA_ROOT/test/manual/gemma4/throughput_matrix.py" \
  --phase baseline --server-root "$GEMMA_ROOT" \
  --bench-script "$GEMMA_BENCH" --datasets "$GEMMA_DATA" \
  --model-path "$GEMMA_MODEL" --output "$GEMMA_RUNS/metadata"
```

Compare matching concurrency/case rows and quality files. The runner records
prefill/decode report counts and means separately for DP0/DP1. Lower kernel or
host timing alone is not a serving speedup.

## Round 2: native versus serial QKV

Repeat the candidate matrix with `--qkv native` and a new output directory.
Compare with `--qkv serial` at identical settings. This changes only the existing
QKV fusion flag; weights and checkpoint precision remain unchanged.

Optional serial-only correctness/timing rerun, with serving stopped:

```bash
ASCEND_RT_VISIBLE_DEVICES=2 TASK_QUEUE_ENABLE=1 PYTHONPATH=python \
python3 test/manual/kernels/benchmark_gemma4_qkv_norm_npu.py \
  --candidate serial --output "$GEMMA_RUNS/qkv-serial-validation.json"
```

The validator covers FP16/BF16, dimensions128/256/512, strided QKV, shared KV,
fullgraph tracing and graph replay in queue modes0/1. Use a fresh output path.
It cannot select a tiled kernel: the implementation and selectors were removed.
Standard rotary embedding and KV writes remain unchanged.

## Bounded profiles

Use a **separate** run, never acceptance results:

```bash
python3 "$GEMMA_ROOT/test/manual/gemma4/throughput_matrix.py" \
  --phase profile --concurrency 28 --server-root "$GEMMA_ROOT" \
  --bench-script "$GEMMA_BENCH" --datasets "$GEMMA_DATA" \
  --model-path "$GEMMA_MODEL" --output "$GEMMA_RUNS/profile-timing"
```

The collector waits for new loaded decode reports from both DP workers, requests
three steps per stage, disables stacks and shapes, retains separate traces, and
checks actual EXTEND/DECODE annotations for both physical devices2/3. Requested
step count is not substituted for observed coverage. A second diagnostic run
with `--record-shapes` must use a different output directory. Keep all rank
folders and `profile/coverage.json`; do not infer recoverable throughput from
aggregate free-time percentages. Reports include per-device kernel CSV row
counts, per-trace device/synchronization/graph/profiler interval unions, and
unassigned gaps between device events. Nested CPU durations are never summed
as wall time, and unassigned gaps are not claimed as recoverable dispatch time.

For an already-running TP1/DP2 server with a saved log, use
`collect_profile.py --server-log /path/server.log --output /tmp/new-profile`
and generate traffic in another terminal. The output directory is server-side;
these commands assume the client and server share the same local filesystem.

## Later rounds and acceptance

At the best measured C24/C28/C29, use `--phase scheduling --concurrency 28` to
compare disabled delaying and the enabled ratio0.25/0.5, 25/50ms, cap5/6 grid.
Each launch is recorded; enabled neighbors change one setting. Compare `off`
against `base`, and each grid row against its one-knob parent, not arbitrary
neighbors. Keep tokens, graph backends and checkpoint precision fixed.

For a candidate passing correctness and exploratory serving checks, use:

```bash
python3 "$GEMMA_ROOT/test/manual/gemma4/throughput_matrix.py" \
  --phase acceptance --concurrency 28 --server-root "$GEMMA_ROOT" \
  --bench-script "$GEMMA_BENCH" --datasets "$GEMMA_DATA" \
  --model-path "$GEMMA_MODEL" \
  --quality-baseline "$GEMMA_RUNS/baseline/base/quality.json" \
  --output "$GEMMA_RUNS/acceptance"
```

Replace28 with the measured winning concurrency; do not exceed29. All eight
cases must pass in **each of three unprofiled repetitions**, with zero failures,
exact RPS targets and the original scenario latency limits. Quality capture
uses fixed greedy completions and token logprobs on four prompts from every
case. Changed greedy outputs, missing logprobs, changed model/runtime identity,
or logprob differences beyond the existing BF16 tolerance reject acceptance.

At C29, the 512-token recommendation target implies mean E2E<=1.333s and
1200-token RAG implies <=3.625s. If validated precision-preserving candidates
plateau below these targets, report the measured ceiling and required speedup;
do not relax concurrency, output lengths, precision or SLOs.

## GLM branch compatibility review

Reference inspected: `glm52-a3-optimizations` at `0b3f0e2baf`.

| Path | Applicability under the approved constraints |
| --- | --- |
| Reusable CPU metadata / buffer preparation | Applied narrowly to FIA host lengths and per-layer BSND cache views. |
| Already-quantized INT8 operands | Expert grouped matmuls are already INT8 in the profile; preserve their existing contract. Further transport changes require identical operand/scale evidence. |
| Add/norm/quant boundary | A quantized-only readout cannot replace BF16 intermediates required by Gemma's router/dense consumers. No eligible boundary is selected without proving all consumer contracts. |
| SwiGLU quantization fusion | Does not preserve Gemma GeGLU; not applicable. |
| TP/FuseEP and DSA/DCP-specific paths | Do not apply to two independent TP1 Gemma replicas; no blanket branch merge. |

The existing `NPUW8A8Int8DynamicLinearMethod.apply` already calls
`unpack_operand(x, INT8)` for a prequantized operand. GLM's `8f489ed1c9` repair
therefore requires no additional port for this checkpoint. No floating router
intermediate is dropped, and GeGLU is unchanged.

## Run all remaining rounds on the NPU host

After setting the paths above and creating the pinned baseline worktree, run:

```bash
python3 "$GEMMA_ROOT/test/manual/gemma4/run_plan.py" \
  --server-root "$GEMMA_ROOT" --baseline-root "$GEMMA_BASE" \
  --bench-script "$GEMMA_BENCH" --datasets "$GEMMA_DATA" \
  --model-path "$GEMMA_MODEL" --output "$GEMMA_RUNS/full-plan"
```

Use a clean source checkout and a new output directory. The runner executes:

1. Pinned TP1/DP2 baseline, then metadata candidate at C24/C28/C29.
2. Native versus serial QKV and serial-only operator validation.
3. The pinned GLM compatibility audit, retaining existing precision/INT8/GeGLU contracts.
4. Disabled delaying and the one-knob scheduling grid, with kernel settings fixed.
5. Separate timing and shape diagnostic profiles, followed by all eight cases in three unprofiled repetitions.

Exploratory configurations are ranked by the geometric mean of completed RPS
normalized by their fixed scenario targets. This ranking does not accept a
configuration or relax any SLO. Each schedule records its one-knob parent and
only a measured parent-relative improvement is eligible for selection.
`plan-result.json` reports strict final acceptance, per-case completed RPS and
remaining required speedup. `experiments.jsonl` records commands and decisions;
each matrix retains quality, runtime/graph settings, memory snapshots, full
server logs, raw metrics and per-replica batch statistics. Failed kernel
candidates remain rejected while independent rounds continue. No Git, dataset,
precision, output-length or concurrency changes are made by this runner.

A nonzero final exit means the targets or profiling gates were not met. Report
these results as observed configurations, not a proven hardware ceiling; no
candidate default is changed automatically. Hardware execution and acceptance
remain pending until this command has actually completed on the NPU server.

For separately managed Docker containers and the fixed vLLM random C24 benchmark, see [docker_ab.md](docker_ab.md). It includes isolated kernel, scheduling and matched MTP/non-MTP commands.

For the matched eager/piecewise MTP prefill trial, see [mtp.md](mtp.md#piecewise-prefill-trial).
