# Gemma 4 two-NPU throughput experiments

Target: recommendation **21.75 completed requests/s**, RAG **8 completed requests/s**,
for all four cases each, with total concurrency below 30. Preserve checkpoint
precision, datasets, output lengths and scenario latency SLOs. No draft model.

This change implements host length-list reuse and hoists repeated BSND cache
views. Tiled QKV and RoPE/KV-write candidates are **opt-in**, with baseline defaults
unchanged. The matrix runner requires complete, matching NPU correctness and
graph-replay reports before it launches a candidate. Source hashes, CANN,
torch/torch_npu, sgl-kernel-npu, Triton and NPU model must match. CPU diagnostics
cannot authorize serving selection. No candidate is accepted from an operator
timing or a single serving run.

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

## Round 2: native versus serial, then tiled QKV

First repeat the same matrix with `--qkv native` and a new output directory.
This disables only the existing QKV fusion flag; it does not change weights or
quantization. Compare against `--qkv serial` at identical settings.

With serving stopped, validate the serial and tiles4/8/16 on one NPU:

```bash
cd "$GEMMA_ROOT"
ASCEND_RT_VISIBLE_DEVICES=2 PYTHONPATH=python \
python3 test/manual/kernels/benchmark_gemma4_qkv_norm_npu.py \
  --candidate all --output "$GEMMA_RUNS/qkv-validation.json"
```

The validator checks FP16/BF16, head dimensions128/256/512, representative token
counts, strided shared-buffer Q/K/V, unweighted V, Q-only shared-KV layers,
untouched sentinel storage, fullgraph tracing and graph replay in queue modes0/1.
It reports native and candidate device/wall medians, and saves partial evidence
if a case fails. Compilation/correctness must pass before timing each case.
After validation, use `--qkv tile4`, `tile8` or `tile16` together with
`--qkv-validation "$GEMMA_RUNS/qkv-validation.json"` in a candidate matrix.
For one concurrency use `--phase candidate --concurrency 28`.
The direct expert toggle is `SGLANG_NPU_GEMMA4_QKV_NORM_ROWS=4|8|16`, with the
existing fusion flag enabled; zero retains the serial implementation. Do not
set the toggle outside controlled validation before correctness has passed.

## Round 3: RoPE and KV-write fusion

With serving stopped:

```bash
ASCEND_RT_VISIBLE_DEVICES=2 PYTHONPATH="$GEMMA_ROOT/python" \
python3 "$GEMMA_ROOT/test/manual/kernels/benchmark_gemma4_rope_kv_npu.py" \
  --output "$GEMMA_RUNS/rope-validation.json"
```

The candidate combines NeoX rotation and two KV scatters into one vector kernel.
It preserves the Gemma angle cache, including the identity columns for
proportional rotation, and writes the already-normalized V unchanged. Live SWA
mapping is read by the kernel so changing slot mappings are observed on replay.
Shared-KV layers retain the Q-only baseline. Unsupported cache layouts, scaled
KV, DCP and nonstandard rotary subclasses retain the unfused path.

Validation covers FP16/BF16, full/SWA slots, strided QKV, long-prefix positions,
zero-token inputs, padding, untouched cache rows, changing positions/mappings,
fullgraph tracing and NPUGraph replay in queue modes0/1. Run a serving candidate
with `--rope-kv on --rope-validation "$GEMMA_RUNS/rope-validation.json"` and a new
output directory. The expert toggle `SGLANG_ENABLE_NPU_GEMMA4_ROPE_KV` defaults
false and affects ordinary prefill only. Full/decode attention semantics and
prefix masks are unchanged.

For CPU indexing diagnostics only:

```bash
SGLANG_USE_CPU_ENGINE=1 TRITON_INTERPRET=1 PYTHONPATH=python \
python3 test/manual/kernels/benchmark_gemma4_rope_kv_npu.py \
  --cpu --tokens 0 1 15 --samples 1 --output /tmp/gemma-rope-cpu.json
```

This emits a `cpu_diagnostic` report rejected by the serving evidence gate.

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
2. Native versus serial QKV, full operator validation, and serving trials of tiles4/8/16.
3. RoPE/KV correctness/replay validation, then a serving trial if validation passes.
4. The pinned GLM compatibility audit, retaining existing precision/INT8/GeGLU contracts.
5. Disabled delaying and the one-knob scheduling grid, with kernel settings fixed.
6. Separate timing and shape diagnostic profiles, followed by all eight cases in three unprofiled repetitions.

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
