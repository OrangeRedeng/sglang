# H06 night agent handoff

Start after H05 B0/W0R parity and the W1 serial-window screen. Use the same
approved A3 reservation for both topology measurements. Compare the actual W1
windows on one 16-device group and on one eight-device subset. Both runs remain
serial. Their time and memory determine whether a two-worker prototype is worth
building.

Reuse the merged-checkpoint exports from [H05_NIGHTLY.md](H05_NIGHTLY.md).
Set `H06_DEVICES_16` to all approved devices, `H06_DEVICES_8` to an approved
subset, and `H06_OUTPUT_PREFIX` to a shared artifact prefix. Pin the same
request, overlap, schedule and checkpoint/decoder hashes.

```bash
: "${H06_DEVICES_16:?Set the 16 approved A3 device IDs}"
: "${H06_DEVICES_8:?Set the eight approved A3 device IDs}"
: "${H06_OUTPUT_PREFIX:?Set the shared artifact prefix}"
unset H3_WINDOW_POLICY SGLANG_H3_WINDOW_OVERLAP
export SGLANG_PROFILE_RECORD_SHAPES=1
h06_launcher=test/manual/minimax_h3_realtime/research/run_a3_windowing.sh

# Warmed latency pair. Run sequentially.
ASCEND_RT_VISIBLE_DEVICES="$H06_DEVICES_16" H3_NUM_GPUS=16 H3_LASER16=1 H3_PROFILE=0 \
  bash "$h06_launcher" --inside "$PWD" W1 "$H06_OUTPUT_PREFIX-16-latency"
ASCEND_RT_VISIBLE_DEVICES="$H06_DEVICES_8" H3_NUM_GPUS=8 H3_PROFILE=0 \
  bash "$h06_launcher" --inside "$PWD" W1 "$H06_OUTPUT_PREFIX-8-latency"

# Separate trace pair.
ASCEND_RT_VISIBLE_DEVICES="$H06_DEVICES_16" H3_NUM_GPUS=16 H3_LASER16=1 H3_PROFILE=1 \
  bash "$h06_launcher" --inside "$PWD" W1 "$H06_OUTPUT_PREFIX-16-trace"
ASCEND_RT_VISIBLE_DEVICES="$H06_DEVICES_8" H3_NUM_GPUS=8 H3_PROFILE=1 \
  bash "$h06_launcher" --inside "$PWD" W1 "$H06_OUTPUT_PREFIX-8-trace"
```

Expect four scheduler updates and eight logical DiT calls per successful
request. On NPU traces, align the zero-based step/window
`h3_window_s*_w*_prepare`, `h3_window_s*_w*_dit` and `h3_window_s*_fusion`
ranges across ranks. Use device events for kernel and collective duration.
Preserve both media outputs, `samples.jsonl`, `denoise_matches.json`,
`launch.txt`, `profiles/`, per-rank peak HBM and device IDs. Record OOM as an
eight-card feasibility failure and stop that case.

Do not divide serial client E2E by two. Before any two-worker experiment,
account for both resident replicas, state/prediction transfers, worker skew,
fusion/barrier cost and final decode. Treat A5 8-vs-4 as a separate deployment
contract. No continuation or streaming experiment is part of this screen.
