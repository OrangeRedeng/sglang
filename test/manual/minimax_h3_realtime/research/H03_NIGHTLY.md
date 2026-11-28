# H03 night agent handoff

Capture the reported step-3 slowdown on one approved A3 allocation. Use the
published H03 revision on the server and record its exact SHA. Hold checkpoint,
LightVAE, devices, Laser/FA, TP/Ulysses/Ring, cache-off policy and the four-step
schedule fixed. Use the saved monolithic DEV-C4 request. Leave graph and compile
disabled for this eager diagnostic.

1. Recover H01 data and reproduce the matched unprofiled control first.
2. Verify the checkpoint and decoder using [R01](R01_REPORT.md). Set the
   matched prompt, seed and topology. Run B0 and then run the profile as a
   separate launch on the same allocation.

```bash
research=test/manual/minimax_h3_realtime/research
H3_PROFILE=1 SGLANG_PROFILE_RECORD_SHAPES=1 \
  bash "$research/run_a3_windowing.sh" --inside "$PWD" B0 "$H03_OUT"
```

Confirm all expected ranks produced traces and four successful step records.
Join by request ID and rank. The `denoising_step_2` trace span corresponds to
step 3 in `H3 NPU step memory` log records. Entry and exit memory figures are
allocator snapshots in MiB. Peak fields are cumulative since the allocator's
last reset. Use allocation events for transient peaks where available.

Compare steps 2–4 across ranks. Identify the actual Ulysses all-to-all and TP
all-reduce device events. Check rank arrival skew, host enqueue gaps, device
dispatch and waits, layout copies, and changes in allocator state. Name any
missing counters. Keep profiled timings separate from the warmed control.
If the slowdown reproduces, preserve the trace interval and matching tensor
shapes. If it does not, record that result and the measured stack. An eight-step
diagnostic requires its own verified adapter and separately matched control.
