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
ASCEND_RT_VISIBLE_DEVICES=2 \
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
  --tp-size 1 \
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
on the same source/runtime. The helpers default to one NPU2, TP1, without DP; set `GEMMA_DEVICES=3`
to select NPU3, or `GEMMA_DEVICES=2,3` to restore the two-replica test.
Both use memory0.82, disable overlap, enable delaying0.5/50ms/4/cap6,
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
buckets. Verify every active replica and inspect a bounded trace for actual target
verify and draft-loop graph replay; startup capture alone does not establish
that serving uses graphs. Only then run three unprofiled benchmark repetitions.
If capture fails, retain the full traceback and return to `mtp`; do not assume
that a CPU pass proves NPU graph numerics or a throughput improvement.

## Isolate a device stall

A wait copying accepted sequence lengths to CPU after MTP returns means the
queued device work has not completed. It does not identify which kernel or
which graph caused the stall. Do not remove this necessary scheduler copy.
Use these NPU-only switches to test each graph phase independently:

| Trial | `SGLANG_DISABLE_NPU_MTP_DRAFT_GRAPH` | `SGLANG_DISABLE_NPU_MTP_VERIFY_GRAPH` |
| --- | --- | --- |
| Target verification graph only | `1` | `0` |
| Assistant draft graph only | `0` | `1` |
| Eager control | `1` | `1` |
| Both graphs | `0` | `0` |

Both disable switches default0; other speculative families are unchanged.
`SGLANG_DEBUG_NPU_MTP_GRAPH_SYNC=1` explicitly synchronizes after prefill,
draft, verify and next-draft seed preparation, logging begin/complete markers.
This is diagnostic instrumentation; do not use its timing for performance.

First run target-only, then stop/restart for draft-only with the two disable
values exchanged. Keep NPU, model, memory, QKV and FIA settings fixed:

```bash
GEMMA_DEVICES=2 \
SGLANG_DISABLE_NPU_MTP_DRAFT_GRAPH=1 \
SGLANG_DISABLE_NPU_MTP_VERIFY_GRAPH=0 \
SGLANG_DEBUG_NPU_MTP_GRAPH_SYNC=1 \
bash test/manual/gemma4/serve_ab.sh mtp-graph 2>&1 | tee /tmp/mtp-target-only.log
```

From another terminal, send one short request rather than benchmark traffic:

```bash
curl --max-time 60 -sS http://127.0.0.1:30014/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4-w8a8-test","messages":[{"role":"user","content":"Count from one to ten."}],"temperature":0,"max_tokens":16,"stream":false}'
```

Report the last `synchronize begin` without a matching `synchronize complete`,
the complete startup log, and a Python thread dump while frozen. A completed
HTTP header or successful graph capture does not establish decode completion.
Turn diagnostic sync off before unprofiled benchmarks after the stall is fixed.

## Piecewise prefill trial

`mtp-prefill-graph` enables experimental NPU `tc_piecewise` target prefill.
Compute pieces are graphed; attention runs eagerly between pieces. The target
captures all hidden rows needed to seed the frozen assistant. The assistant
has no prefill forward to capture.

Restart the server between these matched trials:

```bash
# Control: eager prefill, eager draft, graphed target verification.
GEMMA_DEVICES=2 SGLANG_DISABLE_NPU_MTP_DRAFT_GRAPH=1 \
SGLANG_DISABLE_NPU_MTP_VERIFY_GRAPH=0 SGLANG_DEBUG_NPU_MTP_GRAPH_SYNC=0 \
bash test/manual/gemma4/serve_ab.sh mtp-graph

# Candidate: change only target prefill to piecewise graphs.
GEMMA_DEVICES=2 SGLANG_DISABLE_NPU_MTP_DRAFT_GRAPH=1 \
SGLANG_DISABLE_NPU_MTP_VERIFY_GRAPH=0 SGLANG_DEBUG_NPU_MTP_GRAPH_SYNC=0 \
bash test/manual/gemma4/serve_ab.sh mtp-prefill-graph
```

Both profiles retain memory fraction0.82, overlap off, delayer0.5/50ms/4/cap6, identical decode
buckets and assistant settings. Keep the same benchmark command and warmup for
both. No public return-hidden-states flag is required. Draft graphs default off
in the new profile because the combined draft/verify graph stall remains unresolved.

CPU regression checks cover the configuration gate, target capture, assistant
skip and padded hidden-state handoff. NPU capture/replay, numerical parity and
performance remain unvalidated. Before timing, use the quality commands above
with fresh output files for these two profiles. Exercise mixed request lengths,
cached prefixes beyond the SWA window, partial buckets and chunked prefills.
Successful startup capture alone is insufficient; verify actual replay, complete
requests and compare greedy outputs/logprobs before three unprofiled repetitions.

## Prefill batching with synchronous MTP

TP1 replicas can use the prefill delayer without overlap scheduling. All MTP
profiles and matched controls enable ratio0.5, maximum50ms/four passes, and cap6.
Queue negotiation stays on CPU without a device collective. TP above one still
requires overlap scheduling. The request cap is a maximum, not a minimum;
small prefills can still occur when a timeout expires or no decode is running.
Rebenchmark against the previous run: fewer prefills may increase TTFT, and a
throughput improvement is not guaranteed.

## Draft metadata reuse

`SGLANG_ENABLE_NPU_FIA_METADATA_REUSE=1` also reuses query/KV length lists in
MTP attention across layers. Lists refresh for each batch; graph replay retains
its existing host-length rebinding. Draft setup uses the scheduler's CPU length
mirror when available, retaining device reduction as a fallback. Verification
must preserve the committed CPU lengths, including int32 inputs.

These changes do not enable draft graphs or change draft sampling probabilities.
NPU replay, output parity and end-to-end performance still require validation.

## Frozen-KV graph update ordering

Frozen-KV draft and target verification now finish host-side FIA updates before
submitting graph replay. This follows the TorchNPU update-then-replay lifecycle
and propagates update errors before launching a graph that waits for its external
events. Other speculative families retain their existing scheduling. The previous
replay fence is retained to protect host length bindings.

This is a host-tested ordering fix, not confirmation that the observed NPU stall
is resolved. Keep the draft graph disabled by default until a target-NPU smoke
request completes with both graph switches set to0 and steps greater than1.
Then check greedy output/logprob parity and changing batch buckets before timing.
