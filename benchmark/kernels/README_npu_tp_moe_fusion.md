# A5 TP MoE retained configuration

Target: Ascend 950, CANN 9.2.beta1, torch_npu 2.10.0.post6,
GLM-5.2 W4A8 MXFP, TP4/EP1, native TP MoE, eager execution.
The supplied campaign established the retained configuration. Cleanup still
requires target numerical validation and a final regression check.

Source `benchmark/kernels/a5_final_env.sh` in a fresh shell. Feature switches
remain opt-in. Shared compute uses the established pre-gate stream with event
dependencies; HCCL stays on the caller. Capture/graph, SP and incompatible
linears retain their fallback paths. Shared GMM1 supports `baseline` and
`grouped_fused`; an explicit mode overrides the legacy boolean switch.

Native norm/gate require GLM-5.2 W4A8 MXFP TP4/EP1 with prequantization/reuse.
The gate accepts only FP32 logits, transposed weights, default TopK layout
and scale algorithm 0. Unsupported gate selections fail explicitly.
FP32 correction bias and routing weights are preserved.

Removed: QNoPE/RoPE overlap, eager/inline/resource indexer scheduling,
Q-projection/KV-norm ablations, shared resource pipeline, post-gate/post-TopK
starts, alternate Triton norm, GMM2/finalize fusion, appended shared-expert
slots, split shared-GMM1 variants, gate-layout/dtype/scale tuning, Hadamard
experiment controls, their runners, and dead policy tests.
Removed scheduling flags are no longer read. Native norm is retained.
CP/DCP/query sharding, automatic HCCL sizing and existing fallbacks remain.

Useful capture/replay tools remain:

- `bench_npu_tp_moe_norm_gate.py`: native norm/router/routing replay and optional
  captured final-output comparisons. Use `--probe-only` to inspect target ABI.
- `bench_npu_tp_moe_input.py`: prequantized native routing replay.
- `bench_npu_tp_shared_expert.py`: baseline/grouped-fused shared-expert replay.
- `analyze_npu_multistream.py`: profiler interval/resource analysis.

For numerical validation compare reference and cleaned-branch captures with
identical inputs, checkpoint/tokenization, routing, KV dtype and TP geometry.
Report router-logit max/mean error, TopK ID mismatch count/rate, TopK-weight
error, and final-logit/output error. Unexpected routing-ID mismatches block
release. Matching generated tokens alone does not establish equivalence.
Operator replay alone does not validate the complete serving schedule.

See [A5 final validation](README_a5_final.md) for launch and regression commands.
