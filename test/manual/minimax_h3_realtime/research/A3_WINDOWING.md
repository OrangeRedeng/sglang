# MiniMax H3 windowing on Ascend A3

The local research branch uses Allor-maker's `feat/minimax-h3-adaln-memo-npu`
at `7ef3f01daa` and retains the OrangeRedeng research tools. The runtime experiment
runs deterministic T2VA with dense attention. It starts from one set of global
AV latents, slices overlapping windows from that state, preserves the native
position IDs and channel order, and applies one scheduler update after fusing
the velocities. The final AV decode runs once on the full latents. The
overlap fusion and early global steps are inspired by [FlowLong](https://arxiv.org/abs/2605.20910)
and [Diff-VF](https://arxiv.org/abs/2608.05976).

Run from the checkout on the A3 Docker host. The existing `sglang-a3-diffusion`
container needs the checkout mounted. Set `H3_CONTAINER_REPO` if its path inside
Docker differs. The optional second argument is an output prefix inside the
container and defaults to `/home/swx1199799/h3-a3`. The launcher appends
`-EXPERIMENT-HHMMSS-YYYYMMDD` using Moscow time and prints the full path.
Server logs stream to the console and are saved to `server.log`.

```bash
runner=test/manual/minimax_h3_realtime/research/run_a3_windowing.sh
bash "$runner" B0
bash "$runner" B1
bash "$runner" B2
bash "$runner" B3
bash "$runner" L16
bash "$runner" W0
```

| Case | Target seconds | NPUs | Mode | Forwards with FlashGen |
|---|---:|---:|---|---:|
| B0 | 15 | 8 | Monolithic | 4 |
| B1 | 7.5 | 8 | Monolithic | 4 |
| B2 | 5 | 8 | Monolithic | 4 |
| B3 | 7.5 | 4 | Monolithic | 4 |
| L16 | 15 | 16 | Monolithic, Laser Attention, TP2 × Ulysses8 | 4 |
| W0 | 15 | 8 | One full window, native execution | 4 |
| W1 | 15 | 8 | Two sequential windows at each step | 8 |
| W2 | 15 | 8 | One global step, then local | 7 |
| W3 | 15 | 8 | Two global steps, then local | 6 |
| W4 | 15 | 8 | Best global prefix with 15% overlap | Depends on prefix |

`L16` defaults to devices 0–15 and uses port 30088 by default. The Laser
experiment pads each TP rank's 28 QKV heads to 32 for Ulysses8 and removes
the extra heads before output projection. Compare against `H3_NUM_GPUS=16`
`B0` (TP2 × Ulysses4 × Ring2, FA) with the same adapter, schedule, prompt,
seed, and decoder.

W0 uses the original full-sequence packing and scheduler path. The host tests
exercise the two-window path with a position-dependent predictor and check
both windows receive the same pre-update state and original position IDs.
Before W1–W4, compare the matched B0 and W0 AV outputs on A3. For deterministic
media streams, compare decoded signatures from inside the container.

```bash
docker exec sglang-a3-diffusion bash -ec '
  for out in "$@"; do
    ffmpeg -v error -i "$out/sample-1.mp4" -map 0:v -map 0:a -f framemd5 "$out/decoded.md5"
  done
  diff -u "$1/decoded.md5" "$2/decoded.md5"
' bash "<printed B0 directory>" "<printed W0 directory>"
bash "$runner" W1
bash "$runner" W2
bash "$runner" W3
H3_GLOBAL_STEPS=1 bash "$runner" W4
```

Defaults are `/home/weights/MiniMax-H3/`,
`/home/weights/LightVAE/lightvae-lite-h3-decoder.safetensors`, and the FlashGen
safetensors under `/home/weights/Beidouqixing/minimax-h3-4step-lora-flashgen/`.
FlashGen uses the published unshifted grid `[1, .7, .4, .15, 0]` and native 12/3
modality shifts. To run a verified eight-step release, supply its matching
`H3_MODEL`, `H3_LORA`, `H3_LORA_ALPHA`, `H3_STEPS=8`, and
`SGLANG_H3_DENOISE_RUNGS` as decreasing integers in thousandths, excluding
terminal zero. The runtime rejects a grid conflicting with checkpoint release
metadata. [FlashGen model card](https://huggingface.co/Beidouqixing/minimax-h3-4step-lora-flashgen)

The launcher saves the command, seed, source, device inventory, server logs,
AV clips, raw timings, median and p90. `denoise.txt` records scheduler updates,
logical forwards, token rows and the realized sigma grids. Use
`H3_REPEATS=10` for a final latency comparison and change `H3_PROMPT` for camera
movement, interacting objects and human speech. Use `H3_PROFILE=1` in a separate
run to request stage profiling. Treat profile samples separately from latency
runs. Compare against B0 using the same decoder and precision.

Sequential windowing is the current candidate. Distributed execution depends on
the 8/4-NPU measurements and matched AV quality. No A3 timing, media parity or
perceptual comparison has been collected in the development workspace.
