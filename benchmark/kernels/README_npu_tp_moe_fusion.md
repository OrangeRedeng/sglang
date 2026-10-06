# A5 TP MoE fusion experiments

All switches default to off. These are unvalidated experiments for GLM-5.2,
H=6144, 256 routed experts, routed TopK=8, I=2048, TP4/TP8, EP1 and the native
TP dispatcher. The existing `deepep` and `allreduce-deepep` paths are preserved.
The requested compatibility target is torch_npu 2.10.0.post6 / CANN 9.2.beta1;
the installed operator schemas and numerical behavior still require A5 probes.

| Switch (prefix `SGLANG_NPU_TP_MOE_`) | Implemented boundary | Constraints |
| --- | --- | --- |
| `FUSE_ROUTED_SCALE` | Routed scale in TopK | W4A8 MXFP routed experts |
| `FUSE_SHARED_EXPERT` | Finalize `skip1` shared merge | Implies routed scaling; separate TP-sharded shared branch |
| `FUSE_GMM2_FINALIZE` | Native GMM2/finalize, optionally with shared input | Mixed MXFP8/MXFP4 ABI required; token-major scatter mapping |
| `PREQUANT_INPUT` | Quantize normalized input before routing, route payload/scales without requantization | BF16 input, MXFP8 dispatcher |
| `REUSE_MXFP8` | Share the exact input payload/scales with compatible shared linears | Use with `PREQUANT_INPUT`; shared W4A8 MXFP or MXFP8 |
| `NORM_MXFP8` | Custom residual/RMSNorm/bias with BF16 and MXFP8 outputs | Requires `PREQUANT_INPUT`; contiguous H=6144 inputs |
| `SHARED_GMM1` | E=1 shared W13/SwiGLU/MX quant | Bias-free, unclamped MX-compatible shared projections |
| `FUSED_SHARED_EXPERT` | Expert 256 always selected with weight 1; E=257/K=9 grouped execution | All shared projections and routed descriptors must be W4A8_MXFP; no EPLB, Waterfill or SBO/TBO |

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

ABI checks fail explicitly rather than silently replacing a missing fused
operator with separate kernels. Schema presence alone does not establish
mixed-MXFP support in CANN or support for the original weight layout.
