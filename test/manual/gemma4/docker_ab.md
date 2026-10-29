# Five launch tests: SGLang server and separate vLLM benchmark container

Use `--network host` for both containers so the benchmark URL remains
`http://127.0.0.1:30014`. On a shared Docker bridge network instead, set
`GEMMA_BENCH_URL=http://<sglang-container-name>:30014` in the benchmark container.
The server binds `0.0.0.0`. Mount the tokenizer path in the vLLM container and
mount/copy `bench_ab.sh` there; no SGLang Python installation is needed there.

## Prepare in the SGLang container

```bash
git switch gemma_fix_and_perf_impr
git pull --ff-only https://github.com/OrangeRedeng/sglang.git gemma_fix_and_perf_impr
export GEMMA_ROOT="$PWD"
export GEMMA_EVIDENCE_DIR=/home/nnai/gemma4-ab-evidence
mkdir -p "$GEMMA_EVIDENCE_DIR"
```

With serving stopped, validate the new kernels once (no server launches):

```bash
ASCEND_RT_VISIBLE_DEVICES=2 TASK_QUEUE_ENABLE=1 PYTHONPATH="$GEMMA_ROOT/python" \
python3 test/manual/kernels/benchmark_gemma4_qkv_norm_npu.py \
  --candidate all --output "$GEMMA_EVIDENCE_DIR/qkv-validation.json"

ASCEND_RT_VISIBLE_DEVICES=2 TASK_QUEUE_ENABLE=1 PYTHONPATH="$GEMMA_ROOT/python" \
python3 test/manual/kernels/benchmark_gemma4_rope_kv_npu.py \
  --output "$GEMMA_EVIDENCE_DIR/rope-validation.json"
```

Reports must match current sources, runtime, CANN and NPU. Use a new evidence
directory for a rerun. If CANN version capture is empty, point `ASCEND_HOME_PATH`
to the active toolkit installation containing version/installation metadata.
One failed tile does not exclude another tile with complete passing coverage.

## Tests 1–3: focused comparisons

Run each command separately, benchmark from the vLLM container, then stop the
server with Ctrl-C. Do not paste all launch commands into one running shell.

| Test | Profile | Purpose |
| --- | --- | --- |
| 1 | `metadata` | Current code with your existing serving flags; fresh reference |
| 2 | `tile8` | Change only QKV normalization to rows8 |
| 3 | `rope-kv` | Change only RoPE/KV fusion; serial QKV retained |
| 4 | `combined` | QKV rows8 + RoPE/KV with existing graphs |
| 5 | `combined-mtp` | Both kernel flags + matching Gemma assistant, eager execution |

```bash
# Test 1
bash test/manual/gemma4/serve_ab.sh metadata

# Test 2
bash test/manual/gemma4/serve_ab.sh tile8

# Test 3
bash test/manual/gemma4/serve_ab.sh rope-kv
```

Rows8 is an initial serving trial, not a demonstrated winner. Operator validation
still evaluates tiles4/8/16. If that evidence suggests another passing tile,
replace `tile8` with `tile4`/`tile16` and use that same row count below.
Keep test1 as a fresh control because the code changed. Skip extra native-QKV,
eager-prefill and delayer sweeps already explored in earlier runs.

All three retain your exact prefill and decode buckets, TP1/DP2 on NPUs2/3,
queue1, FIA, context8192, max-running64, token budgets8192, memory0.92 and
prefill delayer0.5 /50ms /4passes /cap6. Candidate flags are reset each launch.

## Tests 4–5: combined configurations

```bash
# Test 4: use the tile chosen for test2.
GEMMA_QKV_ROWS=8 bash test/manual/gemma4/serve_ab.sh combined

# Test 5: run after stopping test4.
GEMMA_QKV_ROWS=8 \
GEMMA_ASSISTANT_MODEL=/home/test1/models/gemma-4-26B-A4B-it-assistant \
bash test/manual/gemma4/serve_ab.sh combined-mtp
```

The assistant must be the matching `google/gemma-4-26B-A4B-it-assistant`
checkpoint. Test5 uses NEXTN, steps3, top-k1, verify width4, memory0.82, graphs
disabled and overlap disabled. This is a complete-configuration comparison,
not an isolated MTP speedup: graph/overlap modes and memory also differ from
test4. The current Ascend frozen draft has no NPU graph runner. The RoPE/KV
candidate applies to ordinary prefill; shared-KV assistant layers keep their
read-only path. For a later matched MTP control, the optional
`combined-mtp-baseline` profile uses test5's settings without the assistant.

Every command can be inspected without starting serving by appending `--print`.
To retain server logs, append `2>&1 | tee test1.server.log` (choose a new name
for each test). These CPU-checked candidates still require NPU correctness and
serving validation; passing a random speed benchmark does not prove output
correctness. Compare fixed-prompt greedy outputs before using a candidate.

## Benchmark commands in the vLLM container

The helper runs your exact random benchmark: prefix250, input1100, output100,
C24, RPS87,1600 prompts, temperature0, ignore-EOS and the requested percentiles.
Only result filenames and before/after `/server_info` snapshots are added.
Set the URL/tokenizer overrides only if container paths/network differ.

Run the corresponding command while that server is active:

```bash
bash test/manual/gemma4/bench_ab.sh metadata 1
bash test/manual/gemma4/bench_ab.sh tile8 1
bash test/manual/gemma4/bench_ab.sh rope-kv 1
bash test/manual/gemma4/bench_ab.sh combined 1
bash test/manual/gemma4/bench_ab.sh combined-mtp 1
```

These five benchmark commands are separate trials, not concurrent traffic.
Change the last number to2/3 for repetitions without additional server launches;
compare medians and tails rather than declaring a winner from one run.
Use `--print` to inspect the full vLLM command. Results are saved under
`./results/<profile>_c24_run<N>.json`. For MTP, snapshots retain per-replica
`avg_spec_accept_length` when exposed; these are cumulative server statistics,
not an invented per-run acceptance rate.

This short-output C24 workload does not establish all eight recommendation/RAG
case targets. The full acceptance matrix remains available in `run_plan.py`.
