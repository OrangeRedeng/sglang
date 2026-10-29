# Four launch tests: SGLang and a separate vLLM container

Use `--network host` for both containers and benchmark URL
`http://127.0.0.1:30014`. On a shared bridge network, set
`GEMMA_BENCH_URL=http://<sglang-container-name>:30014` in the benchmark container.
Mount the tokenizer and `bench_ab.sh` in that container.

## Prepare in the SGLang container

`GEMMA_ROOT` is the SGLang repository directory, not the weights directory.
With serving stopped, use a fresh evidence directory and validate QKV once:

```bash
export GEMMA_ROOT="$PWD"
export GEMMA_EVIDENCE_DIR=/home/nnai/gemma4-ab-evidence-qkv
mkdir -p "$GEMMA_EVIDENCE_DIR"
ASCEND_RT_VISIBLE_DEVICES=2 TASK_QUEUE_ENABLE=1 PYTHONPATH="$GEMMA_ROOT/python" \
python3 test/manual/kernels/benchmark_gemma4_qkv_norm_npu.py \
  --candidate all --output "$GEMMA_EVIDENCE_DIR/qkv-validation.json"
```

Evidence must match sources, runtime, CANN and NPU. If CANN capture is empty,
set `ASCEND_HOME_PATH` to the active toolkit. A failed tile cannot authorize
serving; another tile with complete passing coverage can still be tested.
The validator resets Dynamo's compile cache between cases outside timed regions.

The experimental RoPE/KV fusion and its validator were removed after a native
Triton-Ascend launch crash. Standard rotary embedding and KV writes remain.

## Four separate server launches

Benchmark each server from the vLLM container, then stop it before the next test.
Rows8 is an initial trial; use tile4 or tile16 instead if validation favors it.
Use that same passing row count for both MTP control and candidate.

| Test | Profile | Comparison |
| --- | --- | --- |
| 1 | `metadata` | Current reference, serial QKV |
| 2 | `tile8` | Tiled QKV with the same graph and scheduling settings |
| 3 | `combined-mtp-baseline` | Tiled QKV, eager execution, overlap disabled |
| 4 | `combined-mtp` | Same as test 3 plus the matching assistant |

```bash
# Test 1
bash test/manual/gemma4/serve_ab.sh metadata

# Test 2, after stopping test 1
bash test/manual/gemma4/serve_ab.sh tile8

# Test 3, after stopping test 2
GEMMA_QKV_ROWS=8 bash test/manual/gemma4/serve_ab.sh combined-mtp-baseline

# Test 4, after stopping test 3
GEMMA_QKV_ROWS=8 \
GEMMA_ASSISTANT_MODEL=/home/test1/models/gemma-4-26B-A4B-it-assistant \
bash test/manual/gemma4/serve_ab.sh combined-mtp
```

Tests 1/2 retain the supplied prefill/decode buckets, TP1/DP2 on NPUs2/3,
queue1, FIA, context8192, max-running64, token budgets8192, memory0.92,
and delayer0.5 /50ms /4passes /cap6. Tests 3/4 both use memory0.82,
graphs disabled and overlap disabled; compare these two to isolate MTP.
The assistant must match the target checkpoint. MTP uses NEXTN, steps3,
top-k1 and draft width4. Fixed-prompt greedy and logprob checks remain necessary
before accepting serving results. No target-NPU speedup has been established.

Append `--print` to inspect any launch without starting the server. To retain
logs, append `2>&1 | tee test1.server.log`, changing the name for each test.

## Benchmark in the vLLM container

Run only the command matching the active server:

```bash
bash test/manual/gemma4/bench_ab.sh metadata 1
bash test/manual/gemma4/bench_ab.sh tile8 1
bash test/manual/gemma4/bench_ab.sh combined-mtp-baseline 1
bash test/manual/gemma4/bench_ab.sh combined-mtp 1
```

The helper preserves random prefix250, input1100, output100, C24, RPS87,
1600 prompts, temperature0, ignore-EOS and percentiles50/90/99 for
TTFT/TPOT/ITL/E2E. Change the final argument to 2/3 for repetitions.
It saves results under `./results/<profile>_c24_run<N>.json` and captures
before/after server information. MTP acceptance-length statistics, when exposed,
are cumulative server statistics. Append `--print` to inspect the benchmark.

This short-output test does not establish all eight recommendation/RAG targets.
The complete acceptance matrix remains in `run_plan.py`.
