# GLM-5.2 A5 prefill scheduling experiments

These flags only reschedule existing operators. They do not change kernels,
chunk size, shared-expert math, routing, quantization, or finalization.
All gains and numerical equivalence still require validation on A5.

## Flags and dependencies

| Flag | Default | Schedule |
| --- | --- | --- |
| `SGLANG_NPU_DSA_OVERLAP_QNOPE_ROPE` | `0` | TransposeBatchMatMul on process stream `npu_dsa_qnope`, RoPE on the caller; join before returning the attention inputs. |
| `SGLANG_NPU_DSA_EAGER_INDEXER` | `0` | Start process stream `npu_dsa_indexer` after Q-LoRA normalization; join after MLA preparation, then gather sharded top-k on the caller. |
| `SGLANG_NPU_TP_MOE_SHARED_STREAM_START` | `pre_gate` | Start existing `npu_tp_moe_shared` stream before gate, after gate (`post_gate`), or after TopK (`post_topk`). |

The attention experiments run only in ordinary eager extend/prefill. Decode,
draft/verification, capture, breakable graphs, and piecewise graphs retain the
original attention schedule. Eager indexer also falls back for CP metadata,
DCP, and attention-TP input slices requiring AllGather after Q-LoRA. Its
replicated projections, normalization, cache writes, and lightning indexer run
on the side stream; query sharding and its final AllGather remain enabled, with
AllGather on the main stream. Existing indexer Q/RoPE and weight-projection
streams retain their event dependencies and `SGLANG_NPU_USE_MULTI_STREAM` policy.
DSA attention token redistribution remains on the caller after both joins.

Shared startup modes require the existing eager shared-stream eligibility,
including `SGLANG_NPU_USE_MULTI_STREAM=1` and
`SGLANG_NPU_TP_MOE_EAGER_MULTI_STREAM=1`. They preserve existing capture/SP,
collective, dtype, and finalization restrictions. Unsupported configurations
use the original shared-expert path. Invalid startup modes fail at model setup.
Streams are reused across layers. All joins use stream events; tensors crossing
streams are recorded with the allocator.

## Independent A/B runs

Use the supplied GLM-5.2 W4A8C8 launch command with TP4/EP1, CANN 9.2.beta1,
torch_npu 2.10.0.post6, a 200K prefill request, and both prefill limits fixed at
16384. Keep query sharding and the existing multistream policy enabled.
Keep these settings fixed in every case:

```bash
export SGLANG_NPU_TP_MOE_NATIVE_NORM_MXFP8=1
export SGLANG_NPU_TP_MOE_MXFP8_GATE=1
export SGLANG_NPU_TP_MOE_FUSE_ROUTED_SCALE=1
export SGLANG_NPU_TP_MOE_FUSE_SHARED_EXPERT=1
export SGLANG_NPU_TP_MOE_FUSE_GMM2_FINALIZE=0
export SGLANG_NPU_TP_MOE_PREQUANT_INPUT=1
export SGLANG_NPU_TP_MOE_REUSE_MXFP8=1
export SGLANG_NPU_TP_MOE_NORM_MXFP8=0
export SGLANG_NPU_TP_MOE_SHARED_GMM1_MODE=grouped_fused
export SGLANG_NPU_TP_MOE_EAGER_MULTI_STREAM=1
export SGLANG_NPU_USE_MULTI_STREAM=1
export SGLANG_NPU_ENABLE_DSA_INDEXER_QUERY_SHARDING=1
```

Restart the server per case; the shared startup mode is read during model setup.
Set all three experiment settings explicitly each time:

| Case | `DSA_OVERLAP_QNOPE_ROPE` | `DSA_EAGER_INDEXER` | `TP_MOE_SHARED_STREAM_START` |
| --- | --- | --- | --- |
| A | `0` | `0` | `pre_gate` |
| B | `1` | `0` | `pre_gate` |
| C | `0` | `1` | `pre_gate` |
| D | `0` | `0` | `post_gate` |
| E | `1` | `1` | `post_gate` |
| D2 (optional) | `0` | `0` | `post_topk` |

All abbreviated flags in the table have the `SGLANG_NPU_` prefix. For example,
case B adds these variables to the existing server launch:

```bash
SGLANG_NPU_DSA_OVERLAP_QNOPE_ROPE=1 \
SGLANG_NPU_DSA_EAGER_INDEXER=0 \
SGLANG_NPU_TP_MOE_SHARED_STREAM_START=pre_gate \
python -m sglang.launch_server ...
```

Compare fixed inputs/seeds and generated tokens against A before timing.
For query-sharded top-k, compare per-row index sets as well as model outputs;
index order may vary. Include repeated chunks/layers to exercise cache reuse
and tensor lifetimes. Check disabled flags, skipped-indexer layers, and capture
fallback separately. The optional Q/K-projection experiment is deferred until
the whole-indexer experiment passes target-hardware validation.

## Profiler analysis

Use the existing workload's wall and forward timings. Report wall time and
steady-state mean/median forward per case, with identical warmup exclusion and
chunk positions. Also report compute busy, comm busy, idle, sparse-attention,
indexer, and MoE buckets; averages for QuantLightningIndexer,
TransposeBatchMatMul, RoPE, router MXFP8 QuantMatmul, MoeGatingTopK, InitRouting,
and shared-expert kernels. Distinguish router QuantMatmul by its input shape or
operator attribution; other QuantMatmul calls are not router measurements.

`analyze_npu_multistream.py` supplies the additional **compute vs compute**
overlap analysis. It uses interval unions, avoiding double-counting concurrent
kernels. Analyze one device/rank at a time, with a steady-state window selected
from the same chunk positions in each trace. It accepts an exported
`kernel_details.csv` (microsecond start/duration, stream ID, name; optional input
shape and task type) or a Chrome trace JSON/JSON.gz. Chrome traces require the
**Ascend Hardware** process PID, and stream IDs are its `tid` values; CPU launch
threads must not be included. For CSV, stream IDs come from `Stream ID`.

Identify streams in the profiler: the q-nope stream contains
TransposeBatchMatMul, the indexer stream contains QuantLightningIndexer, and the
shared stream contains the existing shared GMMs. Include the pre-existing
indexer weight and Q streams separately when examining case C/E. Stream IDs and
PIDs can change on restart, so identify them again for each case.

```bash
python benchmark/kernels/analyze_npu_multistream.py B/kernel_details.csv \
  --main-stream MAIN_ID \
  --side-stream qnope=QNOPE_ID --side-stream shared=SHARED_ID \
  --start-us START --end-us END \
  --baseline A/kernel_details.csv --baseline-main-stream BASE_MAIN_ID \
  --baseline-start-us BASE_START --baseline-end-us BASE_END > B/overlap.json
```

For JSON, add `--device-pid PID` and `--baseline-device-pid BASE_PID`.
For C/E, add `--side-stream indexer=INDEXER_ID`. The result gives per-stream
compute busy time, overlap with main compute, percentage concurrently executing
with main compute (`hidden_pct_proxy`), and per-kernel mean/median durations.
It also compares main kernels executing during overlap with A by name and
input shape, marking comparisons as `name_only` when shapes are absent.

The helper recognizes communication by HCCL/HCOM/collective names, ignores
notify/event-wait tasks, and reports memory-copy/set tasks as other busy time.
Inspect the exported task names/types to confirm this classification. Busy
intervals can overlap and do not sum to wall time. Boundary-crossing tasks are
clipped for busy-time calculations and excluded from kernel-duration summaries.
The window duration is a trace span, not request wall time or TTFT. Kernel
summaries are not whole-module bucket attribution, and shape-free comparisons
are not matched performance evidence.

`hidden_pct_proxy` measures concurrency, not causal time saved. Reject any
experiment that increases steady forward time even if overlap improves. Keep
the compute-vs-communication overlap metric separate from this report.
