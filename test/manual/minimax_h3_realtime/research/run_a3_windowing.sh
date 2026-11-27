#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} != --inside ]]; then
    root=$(cd "$(dirname "$0")/../../../.." && pwd)
    container=${H3_CONTAINER:-sglang-a3-diffusion}
    checkout=${H3_CONTAINER_REPO:-$root}
    script=test/manual/minimax_h3_realtime/research/run_a3_windowing.sh
    command -v docker >/dev/null || { echo "Run this on the A3 Docker host." >&2; exit 1; }
    env_args=()
    for key in H3_MODEL H3_LORA H3_LIGHTVAE H3_STEPS H3_LORA_ALPHA H3_PORT H3_MASTER_PORT H3_B3_DEVICES \
        H3_GLOBAL_STEPS H3_PROMPT H3_SEED H3_REPEATS H3_PROFILE H3_PIN_CPU_MEMORY H3_STARTUP_TIMEOUT H3_REQUEST_TIMEOUT \
        SGLANG_H3_DENOISE_RUNGS SGLANG_H3_WINDOW_OVERLAP ASCEND_RT_VISIBLE_DEVICES; do
        [[ ! -v "$key" ]] || env_args+=(-e "$key")
    done
    exec docker exec "${env_args[@]}" "$container" bash "$checkout/$script" --inside "$checkout" "$@"
fi
shift
cd "$1"
shift

case_id=${1:-B0}
output=${2:-/tmp/h3-a3-$case_id-$(date +%s)}
devices=8
sp=4
visible=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
duration=15
unset SGLANG_H3_WINDOW_MODE SGLANG_H3_WINDOW_GLOBAL_STEPS
case "$case_id" in
    B0) ;;
    B1) duration=7.5 ;;
    B2) duration=5 ;;
    B3) duration=7.5; devices=4; sp=2; visible=${H3_B3_DEVICES:-0,1,2,3} ;;
    W0) export SGLANG_H3_WINDOW_MODE=single ;;
    W1) export SGLANG_H3_WINDOW_MODE=local ;;
    W2) export SGLANG_H3_WINDOW_MODE=global-local SGLANG_H3_WINDOW_GLOBAL_STEPS=1 ;;
    W3) export SGLANG_H3_WINDOW_MODE=global-local SGLANG_H3_WINDOW_GLOBAL_STEPS=2 ;;
    W4)
        export SGLANG_H3_WINDOW_MODE=global-local SGLANG_H3_WINDOW_OVERLAP=0.15
        export SGLANG_H3_WINDOW_GLOBAL_STEPS=${H3_GLOBAL_STEPS:-1}
        ;;
    *) echo "Expected B0–B3 or W0–W4" >&2; exit 2 ;;
esac

model=${H3_MODEL:-/home/weights/MiniMax-H3/}
flashgen=/home/weights/Beidouqixing/minimax-h3-4step-lora-flashgen/minimax_h3_4step_lora_flashgen_v1.0_768p_bf16.safetensors
lora=${H3_LORA:-$flashgen}
decoder=${H3_LIGHTVAE:-/home/weights/LightVAE/lightvae-lite-h3-decoder.safetensors}
steps=${H3_STEPS:-4}
port=${H3_PORT:-30088}
if [[ "$lora" == "$flashgen" && "$steps" != 4 ]]; then
    echo "FlashGen requires four steps. Supply the verified eight-step adapter and schedule." >&2
    exit 2
fi
if [[ "$lora" == "$flashgen" ]]; then
    export SGLANG_H3_DENOISE_RUNGS=1000,700,400,150
else
    : "${SGLANG_H3_DENOISE_RUNGS:?Supply the verified adapter schedule in thousandths}"
fi
[[ -d "$model" && -r "$lora" && -r "$decoder" ]] || {
    echo "Missing model, LoRA or LightVAE inside the container." >&2; exit 1;
}
[[ "$steps" == 4 || "$steps" == 8 ]] || exit 2
IFS=, read -r -a rungs <<< "$SGLANG_H3_DENOISE_RUNGS"
[[ ${#rungs[@]} == "$steps" ]] || { echo "Step count and adapter schedule differ" >&2; exit 2; }

export ASCEND_RT_VISIBLE_DEVICES="$visible" PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_DIFFUSION_DISABLE_STAGE_NAN_CHECK=1 SGLANG_CACHE_DIT_ENABLED=false
export SGLANG_CACHE_DIT_TAYLORSEER=false HCCL_BUFFSIZE=256 SGLANG_H3_LOG_DENOISE=1
command -v ffprobe >/dev/null
if curl -sS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    echo "Port $port is occupied." >&2; exit 1
fi
master_port=$(python3 - "${H3_MASTER_PORT:-0}" "$port" <<'PY'
import socket
import sys

requested, http_port = map(int, sys.argv[1:])
if requested == http_port:
    sys.exit("The rendezvous and HTTP ports must differ.")
with socket.socket() as listener:
    try:
        listener.bind(("127.0.0.1", requested))
    except OSError as error:
        sys.exit(f"Cannot use H3_MASTER_PORT={requested}: {error}")
    print(listener.getsockname()[1])
PY
)
mkdir -p "$(dirname "$output")"
mkdir "$output"
output=$(realpath "$output")
if command -v npu-smi >/dev/null; then npu-smi info > "$output/npu-smi.txt"; fi
args=(sglang serve --model-path "$model" --model-type diffusion --model-variant fl2va
    --lora-path "$lora" --lora-scale 1.0 --lora-alpha "${H3_LORA_ALPHA:-64}"
    --dit-cpu-offload false --num-gpus "$devices" --tp-size 2 --sp-degree "$sp"
    --ulysses-degree "$sp" --port "$port" --master-port "$master_port"
    --component-residency text_encoder=layerwise-offload
    --pin-cpu-memory "${H3_PIN_CPU_MEMORY:-true}"
    --attention-backend laser_attn --warmup-steps "$steps" --warmup-mode server
    --enable-breakable-cuda-graph false --component-attention-backends video-vae=fa
    --vae-config.decoder-tile-batch 2 --minimax-h3-lightvae-decoder-path "$decoder"
    --output-path "$output/server-media")
printf '%q ' "${args[@]}" > "$output/launch.txt"
printf '\nvisible_devices=%s\nrungs=%s\n' "$visible" "$SGLANG_H3_DENOISE_RUNGS" >> "$output/launch.txt"
git rev-parse HEAD > "$output/source.txt" 2>/dev/null || true
git diff --binary HEAD > "$output/source.diff" 2>/dev/null || true
setsid "${args[@]}" > "$output/server.log" 2>&1 &
server=$!
cleanup() {
    kill -TERM -- "-$server" 2>/dev/null || true
    for _ in {1..20}; do
        kill -0 -- "-$server" 2>/dev/null || break
        sleep 1
    done
    kill -KILL -- "-$server" 2>/dev/null || true
    wait "$server" 2>/dev/null || true
}
trap cleanup EXIT
deadline=$((SECONDS + ${H3_STARTUP_TIMEOUT:-1800}))
until curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    kill -0 "$server" 2>/dev/null && (( SECONDS < deadline )) || {
        tail -60 "$output/server.log" >&2; exit 1;
    }
    sleep 2
done

python3 - "$output" "$port" "$duration" "$steps" "$case_id" <<'PY'
import json
import math
import os
from pathlib import Path
import statistics
import sys
from threading import Event
import time

sys.path.insert(0, "test/manual/minimax_h3_realtime/research")
from h3_chunks import Client
from runner import probe_media

out, port, duration, steps, case = sys.argv[1:]
out = Path(out)
duration = float(duration)
frames = round(duration * 24)
frames += (5 - frames) % 17
payload = dict(model="MiniMaxAI/MiniMax-H3", prompt=os.getenv("H3_PROMPT", "Currious raccoon"),
               task="t2va", conditions=[], target=dict(short_edge=768, aspect_ratio="16:9",
               duration_seconds=duration), num_outputs_per_prompt=1, num_inference_steps=int(steps),
               flow_shift=12.0, audio_flow_shift=3.0, seed=int(os.getenv("H3_SEED", "1101")),
               quality="lossless")
client = Client(f"http://127.0.0.1:{port}", int(os.getenv("H3_REQUEST_TIMEOUT", "1800")))
(out / "request.json").write_text(json.dumps(payload, indent=2))
times = []
profiling = os.getenv("H3_PROFILE") == "1"
repeats = 1 if profiling else int(os.getenv("H3_REPEATS", "3"))
if repeats < 1:
    raise ValueError("H3_REPEATS must be positive")
for index in range(repeats + 1):
    media_file = out / f"sample-{index}.mp4"
    start = time.perf_counter()
    request = dict(payload)
    if profiling and index:
        request.update(profile=True, profile_all_stages=True, num_profiled_timesteps=int(steps))
    response = client.generate(request, media_file, Event())
    elapsed = time.perf_counter() - start
    media = probe_media(media_file, require_stream_durations=True)
    if (media["width"], media["height"], media["fps"], media["frames"]) != (1344, 768, 24, frames):
        raise ValueError(f"Unexpected output geometry: {media}")
    if abs(media["duration_s"] - frames / 24) > 1 / 24 or abs(media["audio_duration_s"] - frames / 24) > 0.1:
        raise ValueError(f"Unexpected stream durations: {media}")
    record = dict(case=case, warmup=index == 0, profiled=profiling and index > 0, e2e_s=elapsed, media=media, response=response)
    with (out / "samples.jsonl").open("a") as stream:
        print(json.dumps(record), file=stream)
    if index:
        times.append(elapsed)
if not profiling:
    summary = dict(case=case, median_e2e_s=statistics.median(times), samples_s=times,
                   p90_e2e_s=sorted(times)[math.ceil(0.9 * len(times)) - 1])
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
PY
grep 'H3 denoise mode=' "$output/server.log" > "$output/denoise.txt"
python3 - "$output/denoise.txt" "$steps" <<'PYLOG'
import ast
import math
import os
from pathlib import Path
import re
import sys

path, steps = sys.argv[1], int(sys.argv[2])
mode = os.getenv("SGLANG_H3_WINDOW_MODE", "native")
global_steps = int(os.getenv("SGLANG_H3_WINDOW_GLOBAL_STEPS", "0"))
local_steps = steps - global_steps if mode in ("local", "global-local") else 0
grid = [int(rung) / 1000 for rung in os.environ["SGLANG_H3_DENOISE_RUNGS"].split(",")] + [0]
pattern = re.compile(r"mode=(\S+) scheduler_updates=(\d+) dit_evaluations=(\d+).* sigma_video=(\[.*?\]) sigma_audio=(\[.*?\])")
for line in Path(path).read_text().splitlines():
    match = pattern.search(line)
    if not match or match.group(1, 2, 3) != (mode, str(steps), str(steps + local_steps)):
        raise RuntimeError(f"Unexpected denoising execution: {line}")
    for field, shift in ((4, 12), (5, 3)):
        actual = ast.literal_eval(match.group(field))
        expected = [shift * sigma / (1 + (shift - 1) * sigma) for sigma in grid]
        if len(actual) != len(expected) or not all(
            math.isclose(a, b, abs_tol=1e-6) for a, b in zip(actual, expected)
        ):
            raise RuntimeError(f"Unexpected sigma schedule: {line}")
PYLOG
