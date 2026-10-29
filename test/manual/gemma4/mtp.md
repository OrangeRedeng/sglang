# Gemma 4 assistant on Ascend

SGLang already registers `Gemma4AssistantForCausalLM` in `gemma4_mtp.py`.
`NEXTN` with this architecture resolves to `FROZEN_KV_MTP`, not the ordinary
EAGLE draft worker. The assistant borrows the target embedding and reads its
committed KV cache without writing KV; its recurrent state uses the target's
hidden width. Use the assistant checkpoint matching the target model.

The Ascend backend must use the target allocator's SWA mapping even though the
assistant has a dummy local pool and is not classified as a hybrid-SWA target.
The accompanying regression checks translated SWA page tables separately from
full-attention page tables. CPU checks do not establish NPU correctness.

## Initial correctness run

Download `google/gemma-4-26B-A4B-it-assistant` into the assistant directory below
using your existing Hugging Face access. Its config must declare
`Gemma4AssistantForCausalLM`; do not pass the target W8A8 directory as the draft.
Keep the assistant's original BF16 weights and the target's existing precision.

Stop the previous server first. Start with top-k 1 and all graphs disabled:

```bash
ASCEND_RT_VISIBLE_DEVICES=2,3 \
TASK_QUEUE_ENABLE=1 \
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
ASCEND_USE_FIA=1 \
python3 -m sglang.launch_server \
  --model-path /home/test1/models/gemma4-26B-A4B-w8a8 \
  --served-model-name gemma4-w8a8-test \
  --speculative-algorithm NEXTN \
  --speculative-draft-model-path /home/test1/models/gemma-4-26B-A4B-it-assistant \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --host 0.0.0.0 --port 30014 \
  --device npu --attention-backend ascend \
  --tp-size 1 --dp-size 2 \
  --dtype bfloat16 --context-length 8192 \
  --max-running-requests 29 \
  --max-prefill-tokens 8192 --chunked-prefill-size 8192 \
  --mem-fraction-static 0.82 \
  --disable-overlap-schedule --disable-cuda-graph \
  --trust-remote-code
```

The startup log should report promotion to `FROZEN_KV_MTP`. Check both replicas
and compare fixed-prompt greedy outputs against the same target running without
the five speculative flags. Include cached prefixes longer than the 1024-token
sliding window, multiple requests with different lengths, and multiple decoding
rounds. Check `avg_spec_accept_length` in `/server_info` after generation.

Ascend top-k > 1 is not implemented in the frozen draft backend selector.
The command above deliberately starts with eager execution. The NPU graph
implementation below supports top-k1 with FIA; overlap remains disabled. After correctness passes,
compare unprofiled completed RPS and TTFT/TPOT/E2E against the non-speculative
baseline with identical datasets, output lengths and client concurrency below
30. MTP is a separate experiment from the original no-assistant throughput plan;
no speedup is established here.

Reference: [Google Gemma MTP documentation](https://ai.google.dev/gemma/docs/mtp/mtp).

## NPU graph trial

After eager MTP correctness passes, stop the server and compare these profiles
on the same source/runtime. Both use memory0.82, disable overlap and delaying,
and keep prefill eager. They retain the supplied full-decode buckets. The MTP
profile captures target verification and the assistant's recurrent draft loop.
The non-MTP graph control captures ordinary decode. This does not enable or
validate prefill graphs or overlap with speculative execution.

| Trial | Server | Benchmark in separate vLLM container |
| --- | --- | --- |
| Graph control without assistant | `bash test/manual/gemma4/serve_ab.sh mtp-graph-baseline` | `bash test/manual/gemma4/bench_ab.sh mtp-graph-baseline 1` |
| MTP with target/draft graphs | `bash test/manual/gemma4/serve_ab.sh mtp-graph` | `bash test/manual/gemma4/bench_ab.sh mtp-graph 1` |
| Eager MTP control | `bash test/manual/gemma4/serve_ab.sh mtp` | `bash test/manual/gemma4/bench_ab.sh mtp 1` |

Append `--print` to inspect serving commands. Keep QKV and both FIA preparation
switches identical between runs. Capture complete logs, including graph capture
and fallback warnings. A captured bucket without FIA host-update records falls
back to eager. Buckets outside the capture range and missing CPU lengths also
keep the assistant eager. Target verification rebinds prefix+draft lengths;
every assistant step rebinds the unchanged committed prefix. Wide-head verify
masks are computed from replayed device lengths rather than capture-time values.
These paths have CPU orchestration coverage, not target-NPU acceptance.

With the fixed recommendation/RAG datasets available, capture greedy completions
and logprobs from eager MTP, then repeat against graph MTP:

```bash
# While the eager mtp server is running:
python3 test/manual/gemma4/quality_check.py capture \
  --datasets "$GEMMA_DATA" --output /tmp/mtp-eager-quality.json

# After restarting with mtp-graph:
python3 test/manual/gemma4/quality_check.py capture \
  --datasets "$GEMMA_DATA" --output /tmp/mtp-graph-quality.json
python3 test/manual/gemma4/quality_check.py compare \
  --baseline /tmp/mtp-eager-quality.json --candidate /tmp/mtp-graph-quality.json
```

Use fresh filenames. Also exercise concurrent mixed lengths, cached prefixes
beyond the SWA window, changing request-slot assignments, and partial graph
buckets. Verify both replicas and inspect a bounded trace for actual target
verify and draft-loop graph replay; startup capture alone does not establish
that serving uses graphs. Only then run three unprofiled benchmark repetitions.
If capture fails, retain the full traceback and return to `mtp`; do not assume
that a CPU pass proves NPU graph numerics or a throughput improvement.
