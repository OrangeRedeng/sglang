# A5 TP MoE fusion experiments

All switches default to off. These are unvalidated experiments for GLM-5.2,
H=6144, 256 routed experts, routed TopK=8, I=2048, TP4/TP8, EP1 and the native
TP dispatcher. The existing `deepep` and `allreduce-deepep` paths are preserved.
The requested compatibility target is torch_npu 2.10.0.post6 / CANN 9.2.beta1;
the installed operator schemas and numerical behavior still require A5 probes.
Routing weights remain FP32 through dispatch and finalization, independently
of the hidden-state quantization or fusion switches.

| Switch (prefix `SGLANG_NPU_TP_MOE_`) | Implemented boundary | Constraints |
| --- | --- | --- |
| `FUSE_ROUTED_SCALE` | Routed scale in TopK | W4A8 MXFP routed experts |
| `FUSE_SHARED_EXPERT` | Finalize `skip1` shared merge | Implies routed scaling; separate TP-sharded shared branch |
| `FUSE_GMM2_FINALIZE` | Native GMM2/finalize, optionally with shared input | Mixed MXFP8/MXFP4 ABI required; token-major scatter mapping |
| `PREQUANT_INPUT` | Quantize normalized input before routing, route payload/scales without requantization | BF16 input, MXFP8 dispatcher |
| `REUSE_MXFP8` | Share the exact input payload/scales with compatible shared linears | Use with `PREQUANT_INPUT`; shared W4A8 MXFP or MXFP8 |
| `NORM_MXFP8` | Custom residual/RMSNorm/bias with BF16 and MXFP8 outputs | Requires `PREQUANT_INPUT`; contiguous H=6144 inputs |
| `SHARED_GMM1` | E=1 shared W13/SwiGLU/MX quant | Bias-free, unclamped MX-compatible shared projections |
| `SHARED_GMM1_MODE` | `baseline`, `grouped_fused`, `split_group_quant`, `split3` | Explicit mode overrides `SHARED_GMM1`; unset preserves the legacy switch |
| `EAGER_MULTI_STREAM` | Shared compute on one process-wide NPU stream; event join before finalize/add | Requires `SGLANG_NPU_USE_MULTI_STREAM=1`, native W4A8 MXFP TP, EP1, separate unreduced shared branch; excludes graph capture, SP collectives and decode-attention reductions |
| `FUSED_SHARED_EXPERT` | Expert 256 always selected with weight 1; E=257/K=9 grouped execution | All shared projections and routed descriptors must be W4A8_MXFP; no EPLB, Waterfill or SBO/TBO |

GMM2/finalize requests FP32 output for compatibility with deployed wrappers
that reject BF16, then casts to BF16 before TP reduction. This adds a cast and
changes the rounding boundary relative to standalone BF16 GMM2/finalize;
the captured-tail comparison and serving accuracy checks remain required.

`FUSE_SHARED_EXPERT` merges a separately computed shared output;
`FUSED_SHARED_EXPERT` instead uses the existing shared-to-grouped weight loader.
The latter appends its route in one kernel and can lose shared/routed overlap.
It rejects a checkpoint whose shared branch uses a different quantization.
No weights are converted to a lower precision to enable that experiment.

The dual-output normalization keeps BF16 for the router and quantizes that
rounded BF16 value using 32-value MX blocks. Payload/scales are attached only
to that normalized tensor, with no persistent cache across forwards.
Consumers record their streams to retain reused payloads through asynchronous work.

## Captured-input comparisons

Run on A5 with the runtime dependencies installed:

```bash
python benchmark/kernels/bench_npu_tp_moe_input.py --inputs ffn_input.pt
python benchmark/kernels/bench_npu_tp_moe_finalize.py --inputs tail.pt
python benchmark/kernels/bench_npu_tp_moe_finalize.py --inputs tail.pt --gmm2-inputs gmm2.pt
```

Each script describes its capture dictionary. Use real 16K-prefill routing,
at least 20 warmups and 100 iterations; reports include median, p95 and mean.
The input comparison verifies BF16/residual outputs and exact routed MX
payload/scale bytes against the existing quantized implementation.

For the optional GMM2 comparison, capture `gmm1_output` and `gmm1_scale` after
the existing fused GMM1, and `expert_tokens` as counts. Supply W2 as logical
checkpoint-layout uint8 packed weights `[256,6144,I_local/2]` and E8M0 byte
scales `[256,6144,I_local/32]`, named `w2_weight` and `w2_scale`.
Do not serialize a transformed NZ storage buffer as if it were ND: the script
recreates the current production NZ/transposed representation from logical
checkpoint weights using the production preparation helper. The tail capture
must use the baseline `row_idx_type=0` mapping and match these GMM1 inputs.

The standalone tail replay has the shared output already ready. It cannot
measure the shared-stream overlap loss. Repeat the unchanged serving workload
for each switch and compare routes, logits/generation, complete layer wall
time, stream waits, TP collectives and TTFT. Do not promote a switch from a
kernel-duration sum or static validation.

## Selective fusion and eager overlap

The supplied handoff recommends retaining routed scaling, shared `skip1` merge,
prequantized routing and MXFP8 reuse. Keep `NORM_MXFP8=0`,
`FUSE_GMM2_FINALIZE=0` and `FUSED_SHARED_EXPERT=0`. Its operator tables have
different layer counts, so they do not establish an end-to-end speedup.

GLM-5.2's `GlmMoeDsaForCausalLM` uses `DeepseekV2MoE`, rather than
`Glm4MoeSparseMoeBlock`. The eager experiment extends that implementation's
normal path and retains its final TP reduction. It records normalized/prequant
input readiness on the calling stream, queues compute-only shared work on the
auxiliary stream, then waits on shared output immediately before the consumer.
For `FUSE_SHARED_EXPERT=1`, the existing callback joins after routed GMM2 and
before `Finalize(skip1)`. Otherwise it joins before the existing shared add.
Empty inputs, skipped shared work and unsupported configurations retain their
existing path. Only the two cross-stream dependencies create events.

Starting from the handoff's recommended launch, select an experiment with:

```bash
SGLANG_NPU_TP_MOE_SHARED_GMM1_MODE=grouped_fused
SGLANG_NPU_TP_MOE_EAGER_MULTI_STREAM=0
```

Compare `baseline`, `grouped_fused`, `split_group_quant` and `split3` first.
Then compare the best measured mode with `EAGER_MULTI_STREAM=0` and `1` while
retaining `SGLANG_NPU_USE_MULTI_STREAM=1`. No new mode is promoted by default.
The grouped mode reuses one immutable E=1 count tensor for the most recent
token count/device; changing the count replaces it without mutating a tensor
that an earlier asynchronous invocation may still consume.

### Shared and finalizer replay

```bash
python benchmark/kernels/bench_npu_tp_shared_expert.py --inputs shared.pt
python benchmark/kernels/bench_npu_tp_shared_expert.py --inputs shared.pt --routed-inputs routed.pt
python benchmark/kernels/bench_npu_tp_finalize_shared.py --inputs finalize.pt
```

The scripts' module docstrings specify their capture dictionaries. Capture
normalized BF16 `x` and the exact `quantized_x`/`x_scale` handed to dispatch;
retain TP-local checkpoint gate/up and down weights before format conversion.
Shared weights are uint8 packed `[2*I_local,H/2]` and `[H,I_local/2]`, with
uint8 scales `[2*I_local,H/32]` and `[H,I_local/32]`; `I_local=2048/TP`.
Save routed W13/W2 weights and scales in the same logical layout with a leading
256-expert dimension. Do not serialize NZ buffers as logical ND weights.
Replay reconstructs the production NZ/transposed representations using the
existing weight preparation methods.

Shared replay defaults to real captured prefixes of
`1,16,64,256,512,1024,4096,8192,16384` tokens. Use `--tokens` to request only
counts present in a smaller capture; inputs are never padded or synthesized.
Routed and shared captures must contain exactly the same normalized input.
The overlap replay includes dispatch, existing fused routed GMM1, standalone
GMM2 and shared finalization, with captured FP32 TopK weights already scaled
once. It excludes router execution and TP collectives, so its makespan is a
compute-boundary measurement rather than full serving layer time.

Each replay compares BF16 output with its unfused reference before timing.
Shared replay also verifies input payload/scale bytes against native quantization.
JSONL results report total/stage p50, p95, mean, host enqueue p50 and NPU allocator
requests per iteration (null if the runtime lacks the cumulative counter).
Allocator requests are measured separately and include runtime workspaces;
they are not a count of Python tensor objects. The grouped gate/up, activation
and quantization are inseparable and reported as one stage. Baseline's gate/up
includes its input quantization; all activation/quant stages include down-input
quantization, so down-stage timing uses an already quantized operand.
Tolerances are configurable; passing them does not establish logits/generation
accuracy. Routing IDs and weights are consumed unchanged by these experiments.

The finalizer script reports `finalize_no_skip`, `add`, `finalize_plus_add`
and `finalize_skip1` separately on identical captured inputs. Its capture
requires `routing_weights_scaled=True`, BF16 shared/GMM2 outputs and FP32
weights. Choose a merge from the measured complete boundary.

### Serving validation still required

Run original (experimental switches unset), current selective fusion, best
shared mode, and best shared mode plus eager overlap with the identical launch,
request seed, warmup/cache state and profiler window. The serving benchmark
accepts `--profile --profile-start-step N --profile-steps M` with
`--profile-output-dir DIR --profile-prefix NAME`. Set
`SGLANG_TORCH_PROFILER_DIR` on the server and keep N/M fixed for every run.
Verify matching forward/token shapes, not only matching profiler step numbers.

For each rank report window wall, union device busy, idle, compute-only,
communication-only and their overlap, plus per-forward wall and TTFT.
Classify collectives by caller; event waits are not compute busy. Report operator
counts and MoE/norm/quant/GEMM/collective buckets normalized by sparse-layer
executions, including shared GMM1, finalizer, quantization, separate add/mul.
Compare exact TopK IDs/counts, FP32 weights/scaling, BF16 layer output, logits
and deterministic generation against the original. Hardware runs, production
captures and these profile/accuracy results are pending on this CPU-only host.

## Remaining P2/P3 research

The following requested boundaries are **not implemented**:

- Router FP32 GEMM plus grouped TopK in a single custom kernel. Exact route
  preservation, including FP32 correction bias and selection ties, needs an
  A5 compiler and captured router inputs.
- GMM2/finalize/shared plus TP AllReduce or tiled MC2 overlap. No deployed
  mixed-MXFP grouped/finalize collective ABI has been established.
- Finalize plus the next residual/RMSNorm, or shared W2 plus routed finalize.
  These require a real single custom kernel across the different layouts;
  simply moving residual addition would duplicate the existing fusion.
- Attention transpose/cat elimination. The supplied aggregate timings do not
  identify standalone operators, producer/consumer layouts or call sites.
  Obtain shapes, call stacks, streams and event dependencies before editing.
- General cumsum/cast elimination. The routed GMM1 still needs cumulative
  counts while GMM2 takes counts; removing that conversion without a supported
  count-input GMM1 ABI changes its contract.

The one-kernel shared-route append and finalize metadata packing cover the
small-kernel work needed by the implemented paths. The remaining boundaries
need an accessible A5 runtime and the actual timeline/captured inputs.

## Reference provenance

Compatibility research used Ascend op-plugin commit
`d83570a35dfe0d8e9869c3ecfca6647cfccdd9c8`, not an inspection of the requested
installed extension:

- [Operator schemas](https://github.com/Ascend/op-plugin/blob/d83570a35dfe0d8e9869c3ecfca6647cfccdd9c8/op_plugin/config/op_plugin_functions.yaml)
- [GMM/finalize wrapper](https://github.com/Ascend/op-plugin/blob/d83570a35dfe0d8e9869c3ecfca6647cfccdd9c8/op_plugin/ops/opapi/GroupedMatmulFinalizeRoutingKernelNpuOpApi.cpp)
- [Routing contract](https://github.com/Ascend/op-plugin/blob/d83570a35dfe0d8e9869c3ecfca6647cfccdd9c8/docs/en/custom_APIs/torch_npu/torch_npu-npu_moe_init_routing_v2.md)
- [vLLM shared MXFP pipeline](https://github.com/vllm-project/vllm-ascend/blob/9f0f60de677972fba9de69996b7699c98a521f81/vllm_ascend/ops/fused_moe/shared_experts.py)
- [SGLang MX SwiGLU contract](https://github.com/sgl-project/sgl-kernel-npu/blob/653e519cf9556b0cc8d9e7966aa4551eb65143ae/tests/python/sgl_kernel_npu/test_swiglu_group_quant.py)

`split_group_quant` requires sgl-kernel-npu's registered
`torch.ops.npu.swiglu_group_quant`, the ABI already used by routed MXFP8
activation in this checkout. It checks the installed schema explicitly.
The similarly named torch_npu `npu_swiglu_group_quant` at the pinned op-plugin
revision has a different quant-mode/output contract and is not interchangeable.

ABI checks fail explicitly rather than silently replacing a missing fused
operator with separate kernels. Schema presence alone does not establish
mixed-MXFP support in CANN or support for the original weight layout.
