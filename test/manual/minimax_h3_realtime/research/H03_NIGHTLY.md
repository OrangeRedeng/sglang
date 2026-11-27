# H03 night agent handoff

Capture the reported step-3 slowdown on one approved A3 allocation. Use the
published H03 revision on the server and record its exact SHA. Hold checkpoint,
LightVAE, devices, Laser/FA, TP/Ulysses/Ring, cache-off policy and the four-step
schedule fixed. Use the saved monolithic DEV-C4 request. Leave graph and compile
disabled for this eager diagnostic.

1. Recover the H01 data first. Record an unprofiled warmed control on H03 if
   the allocation permits. Measure client E2E and server stages separately.
2. Before starting the matched H03 server for profiling, export
   `SGLANG_TORCH_PROFILER_DIR` to a fresh shared directory and set
   `SGLANG_PROFILE_RECORD_SHAPES=1`. Save the server log, launch command,
   device inventory and Torch/TorchNPU/CANN versions. Use a new HTTP, broker
   and rendezvous port set when any existing service occupies the old ports.
3. In the same container, set `H03_REQUEST` to the saved single-request JSON,
   `H03_ENDPOINT` to the H03 server URL and `H03_OUT` to a new output path.
   From the H03 checkout, submit one profiled request after server warmup.

```bash
export H03_REQUEST H03_ENDPOINT H03_OUT
PYTHONPATH=test/manual/minimax_h3_realtime/research python3 - <<'PY'
import json
import os
from pathlib import Path
from threading import Event

from h3_chunks import Client

request = json.loads(Path(os.environ["H03_REQUEST"]).read_text())
if request["task"] != "t2va" or request["num_inference_steps"] != 4:
    raise ValueError("Use the pinned four-step monolithic T2VA request")
request.update(profile=True, profile_all_stages=True, num_profiled_timesteps=-1)
out = Path(os.environ["H03_OUT"]).resolve()
out.mkdir(parents=True)
(out / "request.json").write_text(json.dumps(request, indent=2) + "\n")
response = Client(os.environ["H03_ENDPOINT"], 1800).generate(
    request, out / "profile.mp4", Event()
)
(out / "response.json").write_text(json.dumps(response, indent=2) + "\n")
PY
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
