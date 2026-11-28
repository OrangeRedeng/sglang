# MiniMax-H3 real-time experiments

Target: 15 seconds of 1344×768 T2VA in ≤15 seconds client E2E on the same
8× Ascend A5 budget. The reported 28.33-second baseline already uses FlashGen,
MXFP8 linears, first-step dense MXFP4 attention, subsequent RainFusion mix
attention at sparsity 0.8, and LightVAE. No new A5 measurements are included.

## Selective FFN execution

Append `--minimax-h3-ffn-active-fraction 0.75` or `0.5` to the existing server
command. The default `1.0` retains full execution. Explicitly disable compile
and breakable graphs and use requests without Cache-DiT, Spectrum, or rollout.
Use the native MiniMax-H3 video pipeline; other pipeline configurations reject
this setting.

The planner runs once per packed branch on CPU and transfers four static row
lists to each sequence-parallel rank. It rotates active temporal groups across
middle layers. Text, audio, references, padding, first/last video latent frames,
and the first/last 10% of DiT layers remain fully computed. Eligible video rows
are active in two or three of each four layer phases. The reported fraction
applies to eligible rows; protected rows increase the actual GEMM row count.
All four denoising evaluations remain. Inactive FFN rows retain the post-attention
residual. Original modulation indices and attention positions remain intact.

This is approximate generation. CPU tests establish row selection, sharding,
modulation, and residual semantics; they do not establish quantized NPU operator
support, visual/audio quality, graph compatibility, or a speedup. Measure actual
MLP GEMM M and gather/scatter cost before promoting it.

## Warm-server sweep

Create a JSON configuration containing `server_command` and `client_command`
(nonempty arrays of argv strings), `health_url`, and optional `environment`,
`warmups` (default 3), `repeats` (default 5), `startup_timeout_s` (1800), and
`request_timeout_s` (600). Commands are executed directly without a shell.
Set the checkpoint manifest and receipt in `environment` and point
`server_command` to the [R01 launcher](research/R01_REPORT.md). Pass the
verified server ID from `H3_RESEARCH_MODEL_ID` to the fixed-seed client.

The client **must wait for generation completion and final audio-video download**,
exit nonzero on failed requests, and preserve each output for quality review.
Submitting an asynchronous request alone measures submission latency. Avoid
restarting the server or loading models in the client command. Include resolution,
15-second duration, seed, exact FlashGen schedule, and decoder in the saved config.
Commands and explicitly supplied environment variables are saved locally; omit
credentials from them.

Remove `--attention-backend`, `--attention-backend-config`, and
`--minimax-h3-ffn-active-fraction` from the base server command; the driver owns
these options. Keep all other precision/topology/decoder flags. Optional
`attention_config` supplies the exact baseline backend configuration. Defaults
are RFv3 mix, sparsity 0.8, one dense initial step, zero dense final steps, and
MXFP4 dense attention. Confirm these match installed MindIE-SD behavior.

```bash
python python/sglang/multimodal_gen/tools/benchmark_minimax_h3_realtime.py \
  experiment.json --output /tmp/h3-plan --dry-run
python python/sglang/multimodal_gen/tools/benchmark_minimax_h3_realtime.py \
  experiment.json --output /tmp/h3-baseline --only baseline
python python/sglang/multimodal_gen/tools/benchmark_minimax_h3_realtime.py \
  experiment.json --output /tmp/h3-ffn --only ffn-0.75 ffn-0.5
```

The output directory must be new. Each variant starts its own server, waits for
readiness, warms it, records repeated client wall times, then stops only that
server's process group. A pre-existing healthy endpoint is rejected. Results
include raw samples, median, nearest-rank p90, exact commands, and server/client
logs. Failed runs are recorded and stop the sweep. `--dry-run` launches nothing.
Sparsity 0.75/0.80/0.85 and zero/one dense first step are available; FFN variants
hold baseline attention fixed. Use separate configs for native-aligned 5/10/15
second requests, tile batch 1/2, and legal fixed-device topology comparisons.
For final candidates use ≥10 samples and alternate baseline/candidate launches.

## Acceptance and remaining capability gates

Calibrate on eight prompts, evaluate twenty held-out prompts, and retain audio
and video artifacts. Review motion, faces/text, prompt adherence, speech/music,
and AV synchronization against the current baseline. Same-seed pixel similarity
alone is insufficient. Collect a separate profiler run for GEMM shapes, attention
routing/precision, exposed HCCL, decode, and output time; exclude profiling from
latency runs. Record peak memory and actual output duration/frame count.

Combine only individually validated winners, then measure the combination.
The driver does not automatically choose a winner or claim quality acceptance.

The following paths remain gated, not implemented as speculative defaults:

- Compact-query attention and weighted-KV compensation require validated unequal
  Q/K, external routing and/or compatible LSE contracts from installed operators.
- Dynamic activity selection requires the static version to show net device gain.
- Continuation requires proving unchanged FlashGen T2VA conditioning compatibility;
  the published Ref2VA/no-LoRA implementation is insufficient evidence.
- Chunk scheduling and asynchronous incremental AV decode follow coherent sequential
  continuation and decoder-halo validation, within the same eight-device budget.
- HCCL overlap and quantized exchange require a profile and supported operand/scale
  transport contracts. Existing GPU-specific kernels are outside this work.

Do not count LightVAE replacement or CUDA-only AdaLN claims as additional measured
savings. The NPU memo change uses the backend capture query and retains ordinary
projection when the API is unavailable. First verify hits on warmed eager TP1 runs.

## Full research campaign

For all eight tracks, capability gates, repeated/resumable rounds, media checks,
quality evaluation hooks, and structured reports, see [research/README.md](research/README.md).
The quick sweep above remains available for native attention/FFN ablations.
