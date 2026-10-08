# GLM-5.2 A5 resource scheduling

Target: Ascend 950/A5, devices 4–7, CANN 9.2.beta1, torch_npu 2.10.0.post6,
TP4/EP1 native TP MoE, eager execution, W4A8C8 checkpoint with FP8 KV cache.
These versions describe the supplied target; this patch was validated on CPU.
Record the actual CANN, torch, torch_npu, sgl-kernel-npu, SGLang commit and HCCL
network configuration alongside each A5 run.

## Changes and exclusions

- The previous qproj/KV-norm experiment excluded NeoX. NeoX already overlaps
  q_b_proj with KV norm. `SGLANG_NPU_DSA_NEOX_QPROJ_KVNORM_SERIAL=1` now serializes
  those operations on the caller stream as a negative control. Default `0`
  preserves the asynchronous path. Decode, capture and missing-stream cases
  retain their existing behavior and report a fallback reason.
- Indexer `resource` previously excluded NeoX and inherited eager-indexer's
  CP/DCP exclusions. NeoX resource mode now queues wq_b, weights_proj and wk on
  the caller; q RoPE, weights cast and k norm use one side stream. Events join
  the tails before the established CP/DCP, cache and query-sharding path.
  Sliced-qLoRA k gathering stays on the caller before CP and weights gathering.
  Legacy/inline defaults and whole-eager-indexer eligibility are unchanged.
- Non-NeoX resource mode retains its paired-RoPE implementation; CP/DCP and
  sliced-qLoRA remain unsupported there. All resource modes fall back during
  capture/breakable/piecewise graphs and outside ordinary prefill.
- Shared MoE reports every eligibility field and failing guard. Its partial
  pipeline and `runner_inplace` guard remain. The native TP dispatcher routes
  into a separate operator output, and the Ascend runner returns GMM outputs;
  Python inspection alone does not prove the installed operators cannot alias
  or overwrite their inputs. Relaxing this guard requires target lifetime and
  aliasing evidence. Do not enable a broader shared pipeline to bypass it.

## Screening on A5

From this branch's checkout, create the command files:

```bash
mkdir -p /tmp/a5-resource-commands
cat > /tmp/a5-resource-commands/server.txt <<'SERVER'
--attention-backend ascend --device npu --tp-size 4 --disable-cuda-graph
--trust-remote-code --mem-fraction-static 0.9
--chunked-prefill-size 16384 --max-prefill-tokens 16384
--moe-a2a-backend none --max-running-requests 16
--page-size 128 --disable-overlap-schedule --kv-cache-dtype fp8_e4m3
--enable-request-time-stats-logging
SERVER
cat > /tmp/a5-resource-commands/benchmark.sh <<'BENCHMARK'
python -m sglang.benchmark.serving \
  --port "$AUTOTUNE_PORT" --dataset-name generated-shared-prefix \
  --gsp-num-groups 1 --gsp-prompts-per-group 17 \
  --gsp-system-prompt-len 200000 --gsp-question-len 10828 --gsp-output-len 1
BENCHMARK
cat > /tmp/a5-resource-commands/profile.sh <<'PROFILE'
python -m sglang.benchmark.serving \
  --port "$AUTOTUNE_PORT" --dataset-name generated-shared-prefix \
  --gsp-num-groups 1 --gsp-prompts-per-group 17 \
  --gsp-system-prompt-len 200000 --gsp-question-len 10828 --gsp-output-len 1 \
  --profile --profile-output-dir "$PROFILE_DIR"
PROFILE
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export SGLANG_NPU_AUTO_HCCL_BUFFSIZE=1
export TASK_QUEUE_ENABLE=2
unset HCCL_BUFFSIZE
PYTHONPATH=python python benchmark/kernels/run_a5_resource_overlap_experiments.py \
  --model-path /home/weights/GLM-5.2-W4A8C8-A5-0731/ \
  --server-args-file /tmp/a5-resource-commands/server.txt \
  --benchmark-command-file /tmp/a5-resource-commands/benchmark.sh \
  --profile-command-file /tmp/a5-resource-commands/profile.sh \
  --phase screen --output-dir /tmp/a5-resource-screen
```

The runner pins the visible devices, TASK_QUEUE_ENABLE=2, automatic HCCL sizing,
and enabled CP/multi-request CP/query sharding in each subprocess. It checks
17 successful requests, 3,731,608 input tokens and 17 generated tokens.
Screening TTFT is a sanity signal; it never produces a promotion decision.
For numerical validation, also pass `--correctness-command-file` with the target
capture/replay script comparing router logits, top-k IDs/weights and final
outputs to the established path. A workload signature alone is not accuracy.

The analyzer automatically locates kernel_details.csv, falling back to
op_summary*.csv. Each device export is analyzed separately; no durations are summed across ranks.
Use `--profile-task-glob` to select one logical rank with
`--profile-task-glob`, for example `**/*_0_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv`
when that matches the installed export layout. Check the actual filenames first.
Explicit rank_id profiler metadata identifies comparable rows when available;
otherwise multiple exports are labeled unassigned and not automatically compared.
A single Chrome trace
is also supported with its exact glob and `--profile-device-pid` for the Ascend
Hardware process. Keep the same rank and capture geometry across cases.

Every successful capture writes per-case `resource_overlap.json` and
`resource_overlap.txt`; the root `resource_overlap_report.md` contains Cube/Vector,
Cube/Cube, Vector/Vector, compute/COMM, unknown/mixed coverage, compute busy,
COMM total/busy, TopK count/mean and SparseAttention count. Resource labels come
from task/core metadata. Unknown/mixed tasks do not contribute to Cube/Vector
pair totals; compute/COMM retains all compute. Missing logical operator rows are
marked unavailable. Full device-kernel name/shape/count/mean rows remain in JSON;
use launch correlation for anonymous GEMMs rather than assigning by stream ID.
Raw totals are marked comparable only for matching nonzero TopK/SparseAttention
counts; also verify capture-window alignment before interpreting them.

Only opt into `MOE_SHARED_RESOURCE` after inspecting its detailed blockers. If
it falls back, screening records it as ineligible and skips its profile. Request
`IDX_RESOURCE_QNOPE` after `IDX_RESOURCE` in the same screening invocation; it
is skipped unless the resource profile engages successfully. For example:
`--cases IDX_RESOURCE IDX_RESOURCE_QNOPE`. The default matrix has no eager-indexer
case, HCCL enumeration or combinations.

## Confirmation after profile review

Shortlist only mechanisms demonstrated by the profile: serial-ablation changes
in useful Cube/Vector overlap; reduced indexer Cube/Cube without slower
projection GEMMs; or eligible shared GateUp overlap without slower routed GMMs.
An aggregate overlap value alone does not prove a critical-path gain.

```bash
PYTHONPATH=python python benchmark/kernels/run_a5_resource_overlap_experiments.py \
  --model-path /home/weights/GLM-5.2-W4A8C8-A5-0731/ \
  --server-args-file /tmp/a5-resource-commands/server.txt \
  --benchmark-command-file /tmp/a5-resource-commands/benchmark.sh \
  --correctness-command-file /path/to/check-routes-and-output.sh \
  --phase confirm --cases IDX_RESOURCE --runs 3 \
  --screen-results-file /tmp/a5-resource-screen/profiles.json \
  --output-dir /tmp/a5-resource-confirm
```

Confirmation uses BASE/candidate/BASE pairs and median per-pair gain against
the adjacent baseline mean. It requires successful reviewed baseline/candidate
profiles with matching settings/checkpoint, at least three pairs, passing
correctness, and a median paired gain of at least 0.5% for promotion eligibility.
Profiler TTFT is rejected. Target event lifetimes, numerical equivalence,
HCCL correctness and performance remain unverified on this CPU-only host.
