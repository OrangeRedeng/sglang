# H04 night agent handoff

On an approved A3 allocation, compare the H04 H3 output callback guard against
parent `7993559bf7d049aa162207323ae07ea1622b7373`. Keep the saved monolithic
four-step request, weights, LightVAE decoder, topology and quality mode fixed.
Record both exact source SHAs and the decoder hash.

## Eager decode and output

1. Preserve the startup log. Confirm it reports the LightVAE decoder load,
   resolved VAE precision and NPU tile cap. Start with the recorded tile batch
   of 2, then compare batch 1 in a separate run if budget permits. Treat exact
   quality as its own control. The current NPU path caps tile batch at 2 and
   temporal window batch at 1.
2. Capture a warmed unprofiled request, then a separate full-stage profile
   using the request and profiler commands in [H03_NIGHTLY.md](H03_NIGHTLY.md).
   Compare `h3_vae_audio_owner`, `h3_vae_video_decode`, `h3_vae_audio_share` and
   `h3_vae_video_finish`, and inspect the subsequent worker encode and mux tail.
   Use device events and the client E2E clock alongside the host spans.
3. Check that both runs emit a downloadable MP4 with 362 frames at 24 FPS,
   1344×768 and a complete audio stream. Record audio sample rate, AV durations,
   timestamps, warmed latency and peak memory. Compare decoded frame and audio
   samples on matched inputs before promoting the change.

```bash
command -v ffmpeg
command -v ffprobe
: "${H04_MEDIA:?Set the result MP4 path}"
ffprobe -v error -count_frames -show_streams -show_format -of json "$H04_MEDIA"
```

## Graph and overlap capability

In the same SGLang environment, record TorchNPU and its graph and stream API
availability. This check only establishes API presence.

```bash
python3 - <<'PY'
from importlib.metadata import version

import torch
import torch_npu

print("torch", torch.__version__, "torch-npu", version("torch-npu"))
for name in ("NPUGraph", "graph", "Stream", "Event", "current_stream"):
    print(name, callable(getattr(torch.npu, name, None)))
PY
```

For the graph follow-up, capture only a resident decoder tile forward. Keep
gather, timeline blending, audio and output encoding eager. Require parity
across two distinct tile inputs and boundary shapes, safe output lifetime,
and separate capture/replay HBM before enabling it in service. Test NPU
audio/video overlap independently once stream and tensor lifetime semantics
are verified. Report unavailable operator evidence as blocked.
