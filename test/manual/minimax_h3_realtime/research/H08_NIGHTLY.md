# H08 night agent handoff

Establish the quality reference before screening attention or quantization.
Use one approved eight-A3 allocation for the commands below. Use the same
prompt, seed, canvas, duration and decoder within each comparison. Save the
source SHA, model/adapter/decoder hashes, CANN and TorchNPU versions, device
IDs, media, dispatch trace and per-request timings.

## Dense reference and adapters

Set `H08_BASE_MODEL` to the undistilled checkpoint, `H08_FLASHGEN_MODEL` to
the independently verified merged four-step checkpoint, `H3_PROMPT` to the
saved prompt and `H08_OUTPUT_PREFIX` to a shared artifact prefix. The reference
uses native VAE and dense Ascend FA BF16. Use that same decoder and backend
when measuring adapter quality against the reference.

```bash
: "${ASCEND_RT_VISIBLE_DEVICES:?Set the eight approved A3 device IDs}"
: "${H08_BASE_MODEL:?Set the pinned undistilled checkpoint}"
: "${H08_FLASHGEN_MODEL:?Set the pinned merged FlashGen checkpoint}"
: "${H08_OUTPUT_PREFIX:?Set the shared output prefix}"
: "${H3_PROMPT:?Set the saved prompt}"
unset H3_STEPS H3_LORA H3_MERGED_FLASHGEN H3_WINDOW_POLICY H3_FFN_ACTIVE_FRACTION
unset SGLANG_H3_DENOISE_RUNGS
export ASCEND_RT_VISIBLE_DEVICES H3_PROMPT
export H3_NUM_GPUS=8 H3_PIN_CPU_MEMORY=false H3_NATIVE_VAE=1
export H3_QUALITY=exact H3_SEED=1101 H3_FLOW_SHIFT=12 H3_REPEATS=1 H3_PROFILE=0
export H3_ATTENTION_BACKEND=fa H3_ATTENTION_CONFIG='{"precision":"bf16"}'
h08_launcher=test/manual/minimax_h3_realtime/research/run_a3_windowing.sh
H3_MODEL="$H08_BASE_MODEL" H3_NATIVE_REFERENCE=1 \
  bash "$h08_launcher" --inside "$PWD" B0 "$H08_OUTPUT_PREFIX-native50"
H3_MODEL="$H08_FLASHGEN_MODEL" H3_MERGED_FLASHGEN=1 \
  bash "$h08_launcher" --inside "$PWD" B0 "$H08_OUTPUT_PREFIX-flashgen4"
```

Turbo is a separate adapter case. If its exact revision, compatible base and
sampling manifest are available, set `H3_MODEL`, `H3_LORA`, `H3_LORA_ALPHA`,
`H3_STEPS=8`, `H3_FLOW_SHIFT` and `SGLANG_H3_DENOISE_RUNGS` using that
manifest, then run B0. The earlier A3 run used video/audio shifts 6/3 and
1000,875,750,625,500,375,250,125. The public adapter listing alone does not
establish those settings. The launcher uses audio shift 3 throughout.
[Turbo release](https://huggingface.co/lightx2v/Minimax-h3-Turbo)

## RainFusion screen

**Install compatible MindIE-SD first** as described in
[H02_NIGHTLY.md](H02_NIGHTLY.md). Use the merged FlashGen checkpoint and the
same decoder on both sides. Start with sparsity zero and RFv2 BF16, then
change only sparsity to 0.75. The candidate keeps the first and last steps
dense. Use profiles to verify the actual routes before considering 0.80/0.85
or changing the number of dense steps.

```bash
export H3_ATTENTION_BACKEND=ascend_sparse_attn
for h08_sparsity in 0 0.75; do
    H3_MODEL="$H08_FLASHGEN_MODEL" H3_MERGED_FLASHGEN=1 \
    H3_ATTENTION_CONFIG="{\"sparse_type\":\"rf_v2\",\"precision\":\"bf16\",\"sparsity\":$h08_sparsity,\"skip_first_steps\":1,\"skip_last_steps\":1,\"dense_kwargs\":{\"precision\":\"bf16\"}}" \
      bash "$h08_launcher" --inside "$PWD" B0 "$H08_OUTPUT_PREFIX-rfv2-$h08_sparsity"
done
```

Repeat the pair separately with `H3_PROFILE=1`. Check
`ascend_sparse_step_*` ranges and subordinate device events. A dense Laser
range with configured BF16 can use an internal FP16 kernel with range
compensation. The labels show the call path and requested precision. Confirm
kernel format, routing and actual selected blocks from vendor/device evidence
where available. Record missing counters explicitly.

Predeclare the AV quality rubric and acceptance threshold. Check geometry,
motion, details, audio continuity, lip sync and prompt adherence. Preserve
failures and stop on unsupported dispatch or quality regression. Keep native
50-step, four-step, eight-step and historical sparse results as separate
run contracts. A5, mixed precision and integration require separate evidence.
