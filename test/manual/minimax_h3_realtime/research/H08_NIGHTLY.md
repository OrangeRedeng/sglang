# H08 night agent handoff

Establish the quality reference before screening attention or quantization.
Use one approved eight-A3 allocation for the commands below. Use the same
prompt, seed, canvas, duration and decoder within each comparison. Save the
source SHA, model/adapter/decoder hashes, CANN and TorchNPU versions, device
IDs, media, dispatch trace and per-request timings.

## Dense reference and adapters

Set `H08_BASE_MANIFEST` to the undistilled checkpoint, `H08_FLASHGEN_MANIFEST` to
the independently verified merged four-step checkpoint, `H3_PROMPT` to the
saved prompt and `H08_OUTPUT_PREFIX` to a shared artifact prefix. The reference
uses native VAE and dense Ascend FA BF16. Use that same decoder and backend
when measuring adapter quality against the reference.

```bash
: "${ASCEND_RT_VISIBLE_DEVICES:?Set the eight approved A3 device IDs}"
: "${H08_BASE_MANIFEST:?Set the pinned undistilled checkpoint}"
: "${H08_FLASHGEN_MANIFEST:?Set the pinned merged FlashGen checkpoint}"
: "${H08_OUTPUT_PREFIX:?Set the shared output prefix}"
: "${H3_PROMPT:?Set the saved prompt}"
unset H3_STEPS H3_LORA H3_MERGED_FLASHGEN H3_WINDOW_POLICY H3_FFN_ACTIVE_FRACTION
unset SGLANG_H3_DENOISE_RUNGS
export ASCEND_RT_VISIBLE_DEVICES H3_PROMPT
: "${H08_BASE_RECEIPT:?Set the native checkpoint receipt}"
: "${H08_FLASHGEN_RECEIPT:?Set the FlashGen checkpoint receipt}"
export H3_NUM_GPUS=8 H3_PIN_CPU_MEMORY=false
export H3_QUALITY=exact H3_SEED=1101 H3_REPEATS=1 H3_PROFILE=0
export H3_ATTENTION_BACKEND=fa H3_ATTENTION_CONFIG='{"precision":"bf16"}'
h08_launcher=test/manual/minimax_h3_realtime/research/run_a3_windowing.sh
H3_CHECKPOINT_MANIFEST="$H08_BASE_MANIFEST" H3_CHECKPOINT_RECEIPT="$H08_BASE_RECEIPT" \
  bash "$h08_launcher" --inside "$PWD" B0 "$H08_OUTPUT_PREFIX-native50"
H3_CHECKPOINT_MANIFEST="$H08_FLASHGEN_MANIFEST" H3_CHECKPOINT_RECEIPT="$H08_FLASHGEN_RECEIPT" \
  bash "$h08_launcher" --inside "$PWD" B0 "$H08_OUTPUT_PREFIX-flashgen4"
```

Use a separately verified Turbo8 merge and TURBO manifest for the eight-step
reference. See [R01 checkpoint preparation](R01_REPORT.md).

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
    H3_CHECKPOINT_MANIFEST="$H08_FLASHGEN_MANIFEST" H3_CHECKPOINT_RECEIPT="$H08_FLASHGEN_RECEIPT" \
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
