# R03 seam-aware native AV windowing

R03 branches from clean R01 at 68e373c50c47492e5860e2d18a9599133ea9c754. The archived A3 media and operator seam timestamp were unavailable locally. The reported LLLL seam remains a quality-negative observation. Its relationship to window boundaries, decoder tiling and intentional cuts is unresolved. The related [vLLM-Omni sliding-window PR](https://github.com/vllm-project/vllm-omni/pull/6779) addresses longer generation. R03 keeps the existing SGLang native shared-state window path.

| Policy | Change |
|---|---|
| fixed | Existing centered linear blend and explicit G/L schedule. |
| shifted | Prepack center and symmetric offsets within 5% of latent length. Cycle center, right, left, center by scheduler step. Preserve latent overlap and map its edges to audio using native positions. |
| adaptive | Score the same buckets using normalized video and audio temporal change from the completed scheduler state, audio overlap and window imbalance. The replica leader broadcasts the decision once per eligible local step. Use center initially and on nonfinite scores. |
| context | Append a few global visual anchor frames from the same input state, keeping original position IDs. Discard their predictions before fusion. This uses joint packed attention and pays for the extra tokens. |
| cosine | Change only the positive complementary FP32 fusion ramp. |

All policies preserve the configured G/L budget. Both windows read the same step input. The existing loop applies one scheduler update per modality. B0 and W0R remain admission controls. Three windows and graph replay remain gated on two-window device quality. The R02 quantization branch is separate.

## Operator screen

Use the R01 verified merged checkpoint and decoder, the same A3 allocation, and the same prompt and seed. Fill the source SHA and operator fields in [the R03 campaign example](campaign.r03.example.json). Start with B0/W0R and the archived schedule review. If artifacts are incomparable, rerun the matched GLLL, LLGL, GLLG, GGLL, GLGL and LLLL controls with the existing launcher.

~~~bash
research=test/manual/minimax_h3_realtime/research
export H3_WINDOW_POLICY=GLLL
H3_SEAM_POLICY=fixed bash "$research/run_a3_windowing.sh" --inside "$PWD" W2 /absolute/output/r03-fixed
H3_SEAM_POLICY=shifted bash "$research/run_a3_windowing.sh" --inside "$PWD" W2 /absolute/output/r03-shifted
H3_SEAM_POLICY=adaptive bash "$research/run_a3_windowing.sh" --inside "$PWD" W2 /absolute/output/r03-adaptive
~~~

Repeat separately with H3_PROFILE=1. Join the H3 seam records to the sample request ID. They report video and channel-major audio ranges, native position bounds, seconds, ramp endpoints, prediction disagreement, packed row counts and context work. Inspect decoded AV around the logged overlap and away from it. Review motion, identity, background, camera continuity, flicker, speech and lip sync on paired clips and held-out prompts. Use the R01 decoder comparison for suspected VAE seams. Promote a policy only after human review and matched warmed latency. Run context and cosine as separate follow-up ablations.

## Validation

Four focused CPU tests exercised the actual packing, geometry, scoring and prediction fusion code with stubbed DiT execution. They check global positions, padding, odd-length video and channel-major audio coverage, anchor exclusion, fixed forward count and adaptive fallback. The R03 controller plan and source checks passed. Ascend dispatch, W0R device parity on this revision, media review and latency remain pending.

### Phase 2.1

The replica leader computes the bounded candidate score vector in FP32. Every replica receives it on its existing device group before selecting its next plan. One host read per rank supplies both the plan decision and diagnostic scores. A nonfinite score selects the centered fallback. Profiled records report adaptive_decision_host_ms through the required host read and the extra context token count across both forwards. Each plan references the same refined text tensor. RoPE and packed metadata depend on absolute window positions and retain distinct storage. Three-window execution remains pending. LLLL still requires seam repair and human AV review.
