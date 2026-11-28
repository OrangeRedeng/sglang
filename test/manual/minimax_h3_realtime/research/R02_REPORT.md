# R02 selective Ascend quantization

R02 starts from R01 commit 68e373c50c47492e5860e2d18a9599133ea9c754. All cases use the R01 verified offline-merged checkpoint and decoder. The component override applies to the DiT transformer. The default B0 launch retains its original precision.

## Source and implementation

| Source | R02 change |
|---|---|
| [#44738](https://github.com/sgl-project/sglang/pull/44738) at aa8c89c5fe7d9fd92367696c7b7996ae0791597d | Port online W8A8 config, exclusions and the shared dynamic INT8 backend. Convert each assembled TP-local weight once and release its floating-point source. Bias is finalized after loading. Use property-backed metadata for rank-local reads. Parameter families with custom or replicated-GQA layouts retain the full-source loader when they lack a safe rank-local transform. |
| [#34855](https://github.com/sgl-project/sglang/pull/34855) and [#53105](https://github.com/sgl-project/sglang/pull/53105) | Retain diffusion’s process-wide format policy. W8A8 reuses the explicit format cast, records ND or FRACTAL_NZ, and probes native matmul at each distinct local layout. |
| [#50117](https://github.com/sgl-project/sglang/pull/50117) | The MiniMax-H3 MXFP8 dispatch fix is already in R01. Q5 uses the existing native Ascend 950 kernel, protected online exclusions and TP-local paired 32-element scale groups. |
| [#43367](https://github.com/sgl-project/sglang/pull/43367) and [#46029](https://github.com/sgl-project/sglang/pull/46029) | Their quantized attention paths are independent of transformer weight quantization. Q5 uses dense FA. |
| [vLLM Ascend #21987](https://github.com/vllm-project/vllm-ascend/pull/21987) | Related online MXFP8 work in the LLM runtime. R02 uses SGLang's existing native operator path. |

The online Ascend converters require complete local linear weights. FSDP DTensor conversion is blocked because the backend transposes the local matrix and installs scales with their own layout. Unquantized FSDP keeps its existing path. R02 uses eager TP/SP with resident DiT, graphs and Cache-DiT disabled. Graph replay, cache reuse, TP1 and worker-group residency require separate device measurements.

Each rank writes a precision JSON file with module paths, method, operators, weight layout and dtype, scale shape and dtype, logical weight and scale bytes, load wall time and allocator snapshots. Rank-local bytes include TP replication and exclude workspaces, format padding and unnamed backend allocations. Allocator peak counters are cumulative since their last reset. Use request profiling and device-level measurements for activation scratch, workspace and peak HBM.

## Cases

| Case | Selected transformer linears |
|---|---|
| Q0 | Original floating-point precision with the resolved inventory |
| Q1 | W8A8 with token refiner, final, conditioning and AdaLN projections protected |
| Q2 | W8A8 FC1 and FC2 in the middle 80% of the actual numbered block stack |
| Q3 | W8A8 explicit whole projections from the same middle blocks, selected from sensitivity evidence |
| Q4 | W8A8 whole QKV and output projections in the middle blocks |
| Q5_FFN / Q5_ALLOWLIST | Ascend 950 MXFP8 on middle-block FFN or approved whole projections, using the matching Q0 inventory |
| Q5 | Ascend 950 MXFP8 with the same protected floor as Q1 and dense FA |

Q2–Q4 use a Q0 rank-0 map from the same commit, checkpoint, topology and attention backend. The launcher passes the complement through the existing transformer ignored-layer option. Each rank checks its resolved map. BF16 and FP32 islands retain their original precision. Joint AV projections share weights across video and audio. Q6 depends on accepted R03 seam results.

## Run

Fill the existing controller inputs in [the A3 example](campaign.r02-a3.example.json) or [the A5 example](campaign.r02-a5.example.json) with the published R02 SHA, allocation, software fingerprint and R01 manifest receipt. Both examples start with Q0. Q3 also requires an approved sensitivity receipt with the selected projection paths in its allowlist.

For a manual trial inside the allocated A3 container, set the approved device IDs and the R01 manifest and receipt. Use the same prompt, seed and topology for each paired sample.

~~~bash
research=test/manual/minimax_h3_realtime/research
export H3_CHECKPOINT_MANIFEST=/absolute/path/to/flashgen4.json
export H3_CHECKPOINT_RECEIPT=/absolute/path/to/flashgen4.receipt.json
export H3_NUM_GPUS=8 H3_PIN_CPU_MEMORY=false H3_SEED=1101 H3_REPEATS=2

H3_QUANT_CASE=Q0 bash "$research/run_a3_windowing.sh" --inside "$PWD" B0 /absolute/output/r02-q0
H3_QUANT_CASE=Q1 bash "$research/run_a3_windowing.sh" --inside "$PWD" B0 /absolute/output/r02-q1
H3_QUANT_CASE=Q2 H3_QUANT_REFERENCE=/absolute/output/q0-run/precision-rank-0.json \
  bash "$research/run_a3_windowing.sh" --inside "$PWD" B0 /absolute/output/r02-q2
~~~

For an A5 pair, use H3_ATTENTION_BACKEND=fa for both Q0 and Q5.

### Sensitivity and quality

1. Write a JSON array containing actual whole projection names from Q0, such as ["blocks.10.mlp.fc1","blocks.10.mlp.fc2"].
2. Run Q0 with H3_PROFILE=1 and H3_QUANT_CAPTURE pointing to the JSON array. The early, middle and final measured denoise steps each save up to eight rank-local rows per present AV modality, using packed latent row IDs and the BF16 projection output before TP reduction.
3. Run the candidate with H3_PROFILE=1 and H3_QUANT_REPLAY pointing to one of the Q0 replay/step-N directories. The candidate replays the same inputs through its loaded local weight and reports normalized L2, cosine and maximum absolute error for those slices. Run the pair for each calibration prompt and keep the captured tensors.
4. Combine paired layer errors with weight and scale bytes. Review whole AV trajectories for speech, lip sync, seam, identity and prompt adherence. Choose the smallest accepted map that meets the measured residency target, then qualify it with paired warmed timing and held-out prompts.

### Historical context

The earlier #44738 runs used eight A3 devices, TP2/SP4, Laser, 5 s, 50 requested denoising steps and Cache-DiT. Reported E2E was 55.19 s BF16, 49.91 s protected W8A8 and 51.01 s full W8A8. Reported peak HBM was 49,686 MB, 45,110 MB and 34,440 MB respectively. The runs came from separate days and preceded R02.

## Device acceptance

A3 W8A8 and A5 MXFP8 layout, numerical quality, loading peak, steady-state latency, graph replay and cache compatibility are pending a device allocation and verified checkpoint files. Preserve the operator probes and loaded precision-map files with each run.

## Host validation

The 38 CPU research tests passed, including protected path selection,
whole-projection allowlists, baseline identity and campaign deployment gates.
An isolated loader smoke executed the loader function with mocked conversion
and verified one-time conversion, early source release and BF16 retention.
Both campaign plans, Python compilation, repo-selected Ruff rules and Bash
syntax checks passed.

### Phase 2.1

Screen Q2 on A3 or Q5_FFN on A5 immediately after Q0. The broad Q1/Q5 screens are independent. On every rank the audit resolves protected projection families, checks the declared native parameter precision (including intentionally FP32 output heads), and requires all four supported block projections. The controller compares projection paths, precision, rank-local weight/scale shapes, operator lists, placement and source across the allocation. Replay files bind source, checkpoint, rank, modality and step. Capture is limited to three steps, 32 selected layers and eight rows per modality. Existing TP-local once-only conversion guards and the diffusion format policy remain in force. Actual device matmul, scale layouts, memory and quality require operator execution.
