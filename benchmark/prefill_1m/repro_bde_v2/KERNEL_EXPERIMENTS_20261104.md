# A3 INT kernel experiments

Use `run_a3_nightly.py --round kernels --hours 12 --repeats 3`.
Fifteen attempts have a 45-minute ceiling each (11h15m total attempt budget),
within the 12-hour wall budget. Actual duration depends on startup and prefill.
The runner exports full logs to the host every 600 seconds and on completion,
prints progress every 15 seconds, and removes empty output files at completion.
Numerical qualification remains skipped: measurements are diagnostic-unqualified.

| Case | Execution evidence required |
|---|---|
| Baseline, repeated before MegaMoE | Historical BDE capacity recipe, HCCL 256, DCP16 residual sharding |
| Separate shared stream | `tp-shared-stream`, after DCP quantized input gathering; no shared finalize fusion |
| Routed scale | `tp-routed-scale` |
| Input quantization reuse | `tp-prequant` |
| Shared output fusion | `tp-shared` |
| Shared activation/quantization | `tp-shared-activation` (`npu_dequant_swiglu_quant`) |
| GMM2/finalize | `tp-finalize`; remains inactive if the installed kernel/weight dtype is ineligible |
| TP fused combination | All component markers, not merely the environment switches |
| INT8 router + AddRMSNormQuant | `gate-int8-executed`, `add-rmsnorm-quant-int8`, `weight-layout-audit` |
| INT8 router + fused norm + separate stream | All gate/norm/DCP/stream markers |
| Separate stream + input reuse + shared activation | All three component markers |
| Prefill graph | Piecewise capture + replay + DCP16 markers; decode graphs disabled |
| MegaMoE, then MegaMoE + INT8 gate | Native MegaMoE and EP16 token-local execution; converted gate case also requires shared-stream and gate markers |

All long cases keep 1,000,000 input tokens, 16 requests/concurrency, 99% shared
prefix, one output token, chunk 32768 and three matched repetitions. INT8 indexer,
SFA INT8 and DCP gather optimizations are inherited from the BDE control; their
existing activation checks remain in place. The control is the historical 24s
workload recipe with the previously introduced capacity/HCCL settings, not a
promise of 24s latency. Cold preparation and prefix-cached timing stay separate.

The INT8 router/norm experiment runs at full 1M context with TP16/DCP16.
It converts only the gate and shared expert roles online, preserving the
checkpoint's routed W4A8 weights. The native fused norm requires matching INT8
activation contracts for the router, W13 and shared input. It must actually
execute with a residual add to emit its marker. DCP gathers its typed INT8 rows
and FP32 per-token scales, preserves source dtype and residual ownership, and
passes the gathered operand directly to gate/W13/shared consumers without
requantizing. Missing native ABI support or a mismatched consumer leaves the
experiment incomplete. `gate-int8-norm-probe` remains available separately at
4096 input tokens / TP16 / DCP1 / 16384-token context for short bring-up.

The final `megamoe-gate-int8` experiment applies router-only online conversion
to the same 1M EP16 workload and requires both the actual INT8 gate and separate
shared-stream markers. MegaMoE keeps its BF16 native input: its current contract
does not accept the typed norm operand. TP prequantization/finalization/shared
slot fusions remain TP/EP1 operations; applying them to MegaMoE would require
changes to DeepEP's ownership/API, which are outside this repair. Existing
checkpoint-selected shared projection kernels still run on that path.
Native stream selection now uses the input device API, records shared-input and
output lifetimes, and joins before routed/shared addition, avoiding CUDA stream
assumptions on the NPU path.

## Layout audit

`weight-layout-audit` in each server log groups loaded matrix parameters/buffers
by role, dtype and physical format, with counts, example names and shapes.
Format 29 is FRACTAL_NZ; format 2 is ND. It runs at model loading, outside capture,
and reads metadata only. Linear and routed expert loaders already use the
alignment-aware `npu_format_cast` helper. The online INT8 gate now does too,
before transposing [experts, hidden] into the GEMM view. Disable-format flags and
unaligned matrices retain the helper's explicit fallback.

NZ applies to supported GEMM weight storage. Activations, residuals, indexer
metadata, routing payloads and quantization scales retain their native operator
contracts. Packed W4 and MegaMoE-specific prepared layouts must not be replaced
by a blanket format cast. The audit records the actual installed-stack result;
CPU tests do not establish A3 kernel layout acceptance or performance. MXFP
fusion paths remain outside this INT-only experiment scope.

## Shared stream fix

Token-sharded DCP previously returned through `forward_normal`, bypassing the
existing eager dual-stream branch. It now gathers the quantized token input
first, forks the separate shared expert onto its existing alternate stream,
runs router/routed work on the main stream, and joins before shared addition.
Input/prequantized/output storage is recorded on each consuming stream. Capture
and compiled execution keep their existing path; this is an eager experiment.

## Local validation

120 focused CPU tests and 108 subtests passed. The existing socket-dependent
`worker_disconnect` test was deselected because this sandbox forbids sockets.
Fatal lint, whitespace checks and the 15-case launch-plan check passed. Regression
coverage executes production forward methods to check gather/fork/join ordering,
shared/routed addition once, typed gather without requantization, native device
stream APIs, residual preservation and gate NZ preparation before transpose.
A3 execution, native ABI/layout acceptance, numerical quality, memory and latency
remain pending; these host checks are not device acceptance evidence.
