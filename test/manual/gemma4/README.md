# Gemma 4 two-NPU throughput experiments

Target: recommendation **21.75 completed requests/s**, RAG **8 completed requests/s**,
for all four cases each, with total concurrency below 30. Preserve checkpoint
precision, datasets, output lengths and scenario latency SLOs. No draft model.

This change implements host length-list reuse and hoists repeated BSND cache
views. The tiled QKV kernels are **validation-only**. Serving still uses the
existing serial kernel when `SGLANG_ENABLE_NPU_GEMMA4_QKV_NORM=1`; no tile is
selected from CPU tests or operator timings.

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
A tile is not wired into serving until this evidence is reviewed, then the
selected opt-in candidate must pass an identical serving matrix.

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
aggregate free-time percentages.

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

RoPE/KV fusion and any further GLM port remain dependent rounds: validate the
existing ND/SWA cache and rotary contracts on NPU before adding or selecting
another kernel. CPU results do not close those gates.
