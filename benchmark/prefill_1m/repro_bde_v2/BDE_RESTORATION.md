# BDE execution-path restoration

This candidate restores the archived A3 INT recipe to the current runtime.
Host checks establish execution wiring and CPU collective/reference parity;
they do not establish NPU correctness, HCCL behavior, memory fit or speedup.

The measured baseline log used real ModelSlim weights, TP16/DCP16, eager
execution, `max_running_requests=1` and a 16384-token prefill chunk. Its script
defaults differ from that resolved configuration. Keep these settings fixed
for the first comparison; the benchmark still sends sixteen requests.

## Qualify correctness first

From the repository root, run the existing TP16 feature validation in
`NEXT_STEPS_TP16.md`, especially `serve-dcp-full-width-q`,
`serve-dcp-legacy-token-shard`, `serve-dcp-stream-o-proj`,
`serve-dcp-decode-q-preissue` and `serve-bde-v2`. Require numerical parity
and feature evidence, including input-logprob and output-only requests.
The full-head path requires PP1 and eager prefill; multi-request legacy
sharding remains opt-in. No MXFP additions are included.

## Reproduce the cached-prefix workload

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
CHECKPOINT=/home/weights/GLM5.2-w4a8-int

SGLANG_DEBUG_STAGE_LATENCY=0 SGLANG_NPU_DEBUG_DCP=0 \
SGLANG_NPU_FEATURE_EVIDENCE=0 \
CAPTURE=0 CHUNK=16384 MEMFRAC=0.88 MAXREQ=1 PORT=30000 \
EXTRA_ARGS='--quantization modelslim --watchdog-timeout 7200 --max-total-tokens 81920' \
python3 benchmark/prefill_1m/launch_bde_v2.py 16 --tp 16 \
  --model-path "$CHECKPOINT" --real-weights --context-length 1052672 \
  2>&1 | tee serve-bde-restored.log
```

Check `/server_info` for the resolved settings and the startup log for
`max_total_num_tokens=1310720`. A smaller profiled pool overrides the requested
cap; stop if the actual capacity cannot hold the request. This count is
logical capacity: this branch interprets the cap as 81920 local rows at DCP16.
The context includes 1048576 input tokens plus 4096 output tokens, provided
the checkpoint supports that context.

In the existing AISBench directory, use the workload from the supplied logs:

```bash
python3 aisbench_test.py --input_len 1000000 --output_len 1 \
  --request_rate 0 --dataset_type prefix_cache --prefix_test --seed 4 \
  --repeat_rate 0.99 --data_num 16 --concurrency 16
```

Save cold-prefix and cached-request results separately. Repeat at least three
times and retain server configuration, checkpoint identity, environment
versions, cache-hit counts, batch sizes and complete benchmark summaries.
Record mean/P90 TTFT, E2E, throughput and peak memory; TPOT is undefined with
one output token. After the matched MAXREQ1 comparison, change MAXREQ to 64
as a separate round, then test CHUNK32768 separately. Keep enough KV capacity
for the shared prefix plus all admitted suffixes. Do not assume batching wins.

Use `SGLANG_NPU_FEATURE_EVIDENCE=1` in a separate diagnostic run to confirm
`dcp-packed-read`, `dcp-gather-prefetch`, `dsa-residual-shard`,
`dsa-local-topk`, `dsa-o-proj-stream`, `indexer-int8` and
`dcp-decode-q-preissue`. Markers are emitted only when their paths execute.

## Source selection

The archive supplies the packed-read/prefetch and full-head residual design;
the existing reviewed indexer/token-shard safety checks are retained. Page
interleaving and current layer-boundary residual state remain in use.
Main was inspected at `51d5502aff999cf3a80e0c2d211d56523310561c`.
Its #45949 targets decode graph replay, #43469 targets MLAProlog, and #46179
targets the prefill-CP path; none repairs this eager DCP call graph. These
remain separate experiments rather than additions to this restoration.
