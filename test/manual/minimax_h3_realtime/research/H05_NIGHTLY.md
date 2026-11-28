# H05 night agent handoff

Use the published H05 server and launcher on an approved A3 allocation.
Recover H01 evidence first. Pin the merged FlashGen and LightVAE hashes, A3
device IDs, TorchNPU/CANN, request and revision. B0 and all W cases must use
the same backend and topology. Use a separate artifact prefix for each run.

Inside the A3 container, use the R01 checkout for the merged-only launcher:

```bash
: "${H3_CHECKPOINT_MANIFEST:?Set the verified FlashGen4 manifest}"
: "${H3_CHECKPOINT_RECEIPT:?Set the verification receipt}"
: "${H3_PROMPT:?Set the saved matched baseline prompt}"
: "${ASCEND_RT_VISIBLE_DEVICES:?Set the 16 approved A3 device IDs}"
: "${H05_OUTPUT_PREFIX:?Set an artifact prefix on the shared filesystem}"
export H3_CHECKPOINT_MANIFEST H3_CHECKPOINT_RECEIPT H3_PROMPT ASCEND_RT_VISIBLE_DEVICES
export H3_LASER16=1 H3_NUM_GPUS=16 H3_PIN_CPU_MEMORY=false
export H3_SEED=1101 H3_REPEATS=1
unset H3_LORA H3_WINDOW_POLICY
h3_case=W0R
bash test/manual/minimax_h3_realtime/research/run_a3_windowing.sh \
  --inside "$PWD" "$h3_case" "$H05_OUTPUT_PREFIX"
```

Start with a same-revision B0 control or reuse its verified artifacts. W0
retains the original native-path control. W0R repacks the whole latent window.
Check full-window predictions, scheduler updates and final AV against B0,
including positions and audio order. The focused host test covers the same
comparison with a deterministic model stub. Host parity does not establish
A3 parity.

| Case | Schedule at four steps | Updates | Logical DiT calls |
|---|---|---:|---:|
| B0 / W0 / W0R | Full sequence | 4 | 4 |
| W1 | LLLL | 4 | 8 |
| W2 | GLLL | 4 | 7 |
| W3 | GGLL | 4 | 6 |
| W4 | GLLL, 15% overlap by default | 4 | 7 |

After W0R parity, screen W1/W2 first. To compare placement at equal work,
set `H3_WINDOW_POLICY=LLGL` with W2, or `GLLG`/`GLGL` with W3. Keep overlap,
sigmas and the rest of the contract fixed for each pair. Stop on parity
failure or a deterministic unsupported path. Leave adaptive policies and
three-window work pending these results.

The launcher saves request-correlated denoise logs and samples. Its summary
requires a matching successful server log for each client request. The DiT
counters count logical calls. Use separate profile runs for per-step device
time, collective work and peak HBM. Report perceptual AV quality separately.
