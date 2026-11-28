#!/usr/bin/env bash
set -euo pipefail

for key in ${!H3_LORA@} ${!SGLANG_LORA@}; do
    echo "Runtime LoRA is prohibited in MiniMax-H3 research ($key)." >&2
    exit 2
done

if [[ ${1:-} != --inside ]]; then
    root=$(cd "$(dirname "$0")/../../../.." && pwd)
    container=${H3_CONTAINER:-sglang-a3-diffusion}
    checkout=${H3_CONTAINER_REPO:-$root}
    script=test/manual/minimax_h3_realtime/research/run_a3_windowing.sh
    command -v docker >/dev/null || { echo "Run this on the A3 Docker host." >&2; exit 1; }
    env_args=()
    for key in H3_CHECKPOINT_MANIFEST H3_CHECKPOINT_RECEIPT H3_EXPECTED_CHECKPOINT_SHA H3_MODEL H3_LIGHTVAE H3_STEPS H3_FLOW_SHIFT H3_PORT H3_MASTER_PORT H3_B3_DEVICES \
        H3_MERGED_FLASHGEN H3_LASER16 H3_WINDOW_POLICY SGLANG_PROFILE_RECORD_SHAPES \
        H3_NATIVE_REFERENCE H3_NATIVE_VAE H3_QUALITY H3_ATTENTION_BACKEND H3_ATTENTION_CONFIG \
        H3_FFN_ACTIVE_FRACTION SGLANG_H3_NPU_ADALN_MEMO \
        H3_QUANT_CASE H3_QUANT_REFERENCE H3_QUANT_ALLOWLIST H3_QUANT_CAPTURE H3_QUANT_REPLAY H3_SERVE_ONLY H3_BIND_HOST H3_LASER_CROP_BEFORE_CAST H3_WINDOW_PARITY H3_GLOBAL_STEPS H3_PROMPT H3_SEED H3_REPEATS H3_PROFILE H3_PIN_CPU_MEMORY H3_NUM_GPUS H3_STARTUP_TIMEOUT H3_REQUEST_TIMEOUT \
        SGLANG_H3_DENOISE_RUNGS SGLANG_H3_WINDOW_OVERLAP ASCEND_RT_VISIBLE_DEVICES; do
        [[ ! -v "$key" ]] || env_args+=(-e "$key")
    done
    exec docker exec "${env_args[@]}" "$container" bash "$checkout/$script" --inside "$checkout" "$@"
fi
shift
cd "$1"
shift

case_id=${1:-B0}
if (( $# > 2 )) || [[ "$case_id" == -* || "${2:-}" == -* ]]; then
    echo "Supply only a research case and optional artifact prefix." >&2
    exit 2
fi
research=test/manual/minimax_h3_realtime/research
: "${H3_CHECKPOINT_MANIFEST:?Set the checkpoint manifest path}"
: "${H3_CHECKPOINT_RECEIPT:?Set the checkpoint verification receipt path}"
launch=$(python3 "$research/checkpoint_contract.py" launch "$H3_CHECKPOINT_MANIFEST" "$H3_CHECKPOINT_RECEIPT")
mapfile -t contract <<< "$launch"
model=${contract[0]} decoder=${contract[1]} family=${contract[2]} steps=${contract[3]}
rungs=${contract[4]} video_shift=${contract[5]} audio_shift=${contract[6]} identity=${contract[7]}
export SGLANG_H3_RESEARCH_FAMILY="$family"
export SGLANG_H3_RESEARCH_STEPS="$steps" SGLANG_H3_RESEARCH_VIDEO_SHIFT="$video_shift"
export SGLANG_H3_RESEARCH_AUDIO_SHIFT="$audio_shift"
if [[ -n "$rungs" ]]; then
    export SGLANG_H3_DENOISE_RUNGS="$rungs"
else
    unset SGLANG_H3_DENOISE_RUNGS
fi
devices=${H3_NUM_GPUS:-8}
sp=4
ring=1
attention=laser_attn
case "$devices" in
    8) ;;
    16) sp=8; ring=2; attention=fa ;;
    *) echo "H3_NUM_GPUS must be 8 or 16." >&2; exit 2 ;;
esac
duration=15
default_prompt="Currious raccoon"
unset SGLANG_H3_WINDOW_MODE SGLANG_H3_WINDOW_GLOBAL_STEPS SGLANG_H3_WINDOW_POLICY SGLANG_H3_LASER_SP8_PAD_HEADS
case "$case_id" in
    B0) ;;
    B1) duration=7.5 ;;
    B2) duration=5 ;;
    HIST-OFF|HIST-ON)
        duration=5
        default_prompt="At night, while their owner sleeps in a bedroom, three cats march in loudly playing tiny brass instruments, then abruptly file out."
        ;;
    B3)
        duration=7.5; devices=4; sp=2; ring=1; attention=laser_attn
        ;;
    L16)
        devices=16; sp=8; ring=1; attention=laser_attn
        export SGLANG_H3_LASER_SP8_PAD_HEADS=1
        ;;
    W0) export SGLANG_H3_WINDOW_MODE=single ;;
    W0R) export SGLANG_H3_WINDOW_MODE=repack ;;
    W1) export SGLANG_H3_WINDOW_MODE=local ;;
    W2) export SGLANG_H3_WINDOW_MODE=global-local SGLANG_H3_WINDOW_GLOBAL_STEPS=1 ;;
    W3) export SGLANG_H3_WINDOW_MODE=global-local SGLANG_H3_WINDOW_GLOBAL_STEPS=2 ;;
    W4)
        export SGLANG_H3_WINDOW_MODE=global-local SGLANG_H3_WINDOW_OVERLAP=0.15
        export SGLANG_H3_WINDOW_GLOBAL_STEPS=${H3_GLOBAL_STEPS:-1}
        ;;
    *) echo "Expected B0–B3, L16, W0–W4, W0R, HIST-OFF or HIST-ON" >&2; exit 2 ;;
esac
if [[ ${H3_LASER16:-0} == 1 && "$devices" == 16 ]]; then
    ring=1; attention=laser_attn
    export SGLANG_H3_LASER_SP8_PAD_HEADS=1
fi
visible=${ASCEND_RT_VISIBLE_DEVICES:-$(seq -s, 0 "$((devices - 1))")}
if [[ "$case_id" == B3 ]]; then visible=${H3_B3_DEVICES:-0,1,2,3}; fi
if [[ -v H3_ATTENTION_BACKEND ]]; then
    if [[ "$case_id" != B0 || "$devices" != 8 ||
        ( "$H3_ATTENTION_BACKEND" != fa && "$H3_ATTENTION_BACKEND" != ascend_sparse_attn ) ]]; then
        echo "The H08 attention override requires eight-device B0 and fa or ascend_sparse_attn." >&2
        exit 2
    fi
    attention=$H3_ATTENTION_BACKEND
fi
if [[ -v H3_ATTENTION_CONFIG && ! -v H3_ATTENTION_BACKEND ]]; then
    echo "H3_ATTENTION_CONFIG requires H3_ATTENTION_BACKEND." >&2
    exit 2
fi
if [[ "$attention" == ascend_sparse_attn && ! -v H3_ATTENTION_CONFIG ]]; then
    echo "Supply the sparse policy explicitly, including the dense step counts." >&2
    exit 2
fi
if [[ ${SGLANG_H3_LASER_SP8_PAD_HEADS:-0} == 1 ]]; then
    IFS=, read -r -a laser_devices <<< "$visible"
    if [[ ${#laser_devices[@]} != 16 ]]; then
        echo "Laser SP8 requires 16 visible Ascend devices." >&2
        exit 2
    fi
fi
output="${2:-/home/swx1199799/h3-a3}-$case_id-$(TZ=Europe/Moscow date +%H%M%S-%Y%m%d)"

port=${H3_PORT:-30088}
if [[ "$family" == native50 ]]; then
    if [[ ! "$case_id" =~ ^(B0|HIST-OFF|HIST-ON)$ || "$attention" == ascend_sparse_attn || -n "$decoder" ]]; then
        echo "Native50 requires dense B0 or HIST and the native VAE." >&2
        exit 2
    fi
elif [[ "$case_id" == HIST-* ]]; then
    echo "Historical cases require the clean Native50 checkpoint." >&2
    exit 2
fi
if [[ "$case_id" == HIST-* && ( "$devices" != 8 || "$attention" != laser_attn ) ]]; then
    echo "Historical controls require eight devices and Laser." >&2
    exit 2
fi
if [[ -v H3_LASER_CROP_BEFORE_CAST ]]; then
    if [[ "$family" != flashgen4 || "$case_id" != B0 || "$attention" != laser_attn ||
        ! "$H3_LASER_CROP_BEFORE_CAST" =~ ^[01]$ ]]; then
        echo "Crop/cast ablation requires FlashGen4 B0 with Laser and a 0/1 order." >&2
        exit 2
    fi
fi
if [[ ${H3_WINDOW_PARITY:-0} == 1 && ( "$family" != flashgen4 || ! "$case_id" =~ ^(B0|W0R)$ ) ]]; then
    echo "Tensor parity requires FlashGen4 B0 or W0R." >&2
    exit 2
fi
export SGLANG_H3_LASER_CROP_BEFORE_CAST="${H3_LASER_CROP_BEFORE_CAST:-1}"
export SGLANG_H3_WINDOW_PARITY="${H3_WINDOW_PARITY:-0}"
case ${H3_QUALITY:-lossless} in
    exact|lossless) ;;
    *) echo "H3_QUALITY must be exact or lossless." >&2; exit 2 ;;
esac
if [[ -v H3_WINDOW_POLICY ]]; then
    if [[ ${SGLANG_H3_WINDOW_MODE:-} != global-local || ! "$H3_WINDOW_POLICY" =~ ^[GL]+$ ||
        ${#H3_WINDOW_POLICY} != "$steps" || "$H3_WINDOW_POLICY" != *L* ]]; then
        echo "H3_WINDOW_POLICY requires W2–W4 and one G/L per step with at least one L." >&2
        exit 2
    fi
    export SGLANG_H3_WINDOW_POLICY="$H3_WINDOW_POLICY"
fi
if [[ -v H3_FFN_ACTIVE_FRACTION && -v SGLANG_H3_WINDOW_MODE ]]; then
    echo "Run selective FFN on the native baseline." >&2
    exit 2
fi

export ASCEND_RT_VISIBLE_DEVICES="$visible" PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SGLANG_DIFFUSION_DISABLE_STAGE_NAN_CHECK=1 SGLANG_CACHE_DIT_ENABLED=false
export SGLANG_CACHE_DIT_TAYLORSEER=false HCCL_BUFFSIZE=256 SGLANG_H3_LOG_DENOISE=1
if [[ "$case_id" == HIST-ON ]]; then
    export SGLANG_CACHE_DIT_ENABLED=true SGLANG_CACHE_DIT_TAYLORSEER=true
    export SGLANG_CACHE_DIT_FN=2 SGLANG_CACHE_DIT_BN=1 SGLANG_CACHE_DIT_WARMUP=4
    export SGLANG_CACHE_DIT_RDT=0.4 SGLANG_CACHE_DIT_MC=4 SGLANG_CACHE_DIT_TS_ORDER=2
    export SGLANG_H3_COUNT_BLOCKS=1
else
    export SGLANG_H3_COUNT_BLOCKS=0
    if [[ "$case_id" == HIST-OFF ]]; then export SGLANG_H3_COUNT_BLOCKS=1; fi
fi
command -v ffprobe >/dev/null
master_port=$(python3 - "${H3_MASTER_PORT:-0}" "$port" <<'PY'
from contextlib import ExitStack
import socket
import sys

requested, http_port = map(int, sys.argv[1:])
broker_port = http_port + 1
if not 1 <= http_port < 65535 or not 0 <= requested <= 65535:
    sys.exit("HTTP, broker and rendezvous ports must be in range.")
if requested in (http_port, broker_port):
    sys.exit("The rendezvous, broker and HTTP ports must differ.")
with ExitStack() as stack:
    for role, port in (("HTTP", http_port), ("broker", broker_port), ("rendezvous", requested)):
        listener = stack.enter_context(socket.socket())
        try:
            listener.bind(("0.0.0.0", port))
        except OSError as error:
            sys.exit(f"Cannot bind {role} port {port}: {error}")
    print(listener.getsockname()[1])
PY
)
mkdir -p "$(dirname "$output")"
mkdir "$output"
output=$(realpath "$output")
if [[ ${H3_PROFILE:-0} == 1 ]]; then
    export SGLANG_DIFFUSION_TORCH_PROFILER_DIR="$output/profiles"
fi
if command -v npu-smi >/dev/null; then npu-smi info > "$output/npu-smi.txt"; fi
served_model="h3-research-$identity"
args=(sglang serve --model-path "$model" --served-model-name "$served_model"
    --host "${H3_BIND_HOST:-127.0.0.1}" --model-type diffusion --model-variant fl2va
    --dit-cpu-offload false --num-gpus "$devices" --tp-size 2 --sp-degree "$sp"
    --ulysses-degree "$((sp / ring))" --ring-degree "$ring" --port "$port" --master-port "$master_port"
    --component-residency text_encoder=layerwise-offload
    --pin-cpu-memory "${H3_PIN_CPU_MEMORY:-true}"
    --attention-backend "$attention" --warmup-steps "$steps" --warmup-mode server
    --enable-breakable-cuda-graph false --enable-torch-compile false --component-attention-backends video-vae=fa
    --output-path "$output/server-media")
if [[ "$family" == native50 ]]; then
    args+=(--vae-config.decoder-tile-batch 1)
fi
if [[ -n "$decoder" ]]; then
    args+=(--minimax-h3-lightvae-decoder-path "$decoder" --vae-config.decoder-tile-batch 2)
fi
if [[ -v H3_ATTENTION_CONFIG ]]; then
    args+=(--attention-backend-config "$H3_ATTENTION_CONFIG")
fi
if [[ -v H3_FFN_ACTIVE_FRACTION ]]; then
    args+=(--minimax-h3-ffn-active-fraction "$H3_FFN_ACTIVE_FRACTION")
fi
export H3_LAUNCH_PLACEMENT="$visible:TP2:SP$sp:Ulysses$((sp / ring)):Ring$ring:$attention:pin=${H3_PIN_CPU_MEMORY:-true}:pad=${SGLANG_H3_LASER_SP8_PAD_HEADS:-0}"
unset SGLANG_H3_QUANT_AUDIT SGLANG_H3_QUANT_CAPTURE SGLANG_H3_QUANT_REPLAY
if [[ -v H3_QUANT_CASE ]]; then
    if [[ "$case_id" != B0 || "$family" == native50 || "$attention" == ascend_sparse_attn ||
        -v H3_ATTENTION_CONFIG || -v H3_FFN_ACTIVE_FRACTION || -v H3_LASER_CROP_BEFORE_CAST ||
        ${H3_WINDOW_PARITY:-0} == 1 ]]; then
        echo "R02 cases require merged, monolithic B0 and the unmodified dense attention/FFN path." >&2
        exit 2
    fi
    if [[ "$H3_QUANT_CASE" == Q5 && ( "$devices" != 8 || "$attention" != fa ) ]]; then
        echo "Q5 requires eight Ascend 950 devices with dense FA." >&2
        exit 2
    fi
    git diff --quiet HEAD || { echo "R02 cases require a clean tracked source checkout." >&2; exit 2; }
    source_sha=$(git rev-parse HEAD)
    resolved=$(python3 "$research/quantization_cases.py" "$H3_QUANT_CASE" "$identity" \
        "$H3_LAUNCH_PLACEMENT" "$source_sha" "$output/quantization-selection.json")
    mapfile -t quantization <<< "$resolved"
    export SGLANG_H3_QUANT_AUDIT="$output" SGLANG_H3_QUANT_CHECKPOINT="$identity"
    export SGLANG_H3_QUANT_METHOD="${quantization[0]}" SGLANG_H3_QUANT_PLACEMENT="$H3_LAUNCH_PLACEMENT"
    export SGLANG_H3_QUANT_SOURCE="$source_sha"
    if [[ -v H3_QUANT_CAPTURE ]]; then
        if [[ "$H3_QUANT_CASE" != Q0 || ${H3_PROFILE:-0} != 1 || -v H3_QUANT_REPLAY ]]; then
            echo "Layer inputs can be captured only during a profiled BF16 Q0." >&2
            exit 2
        fi
        export SGLANG_H3_QUANT_CAPTURE="$H3_QUANT_CAPTURE"
    fi
    if [[ -v H3_QUANT_REPLAY ]]; then
        if [[ "$H3_QUANT_CASE" == Q0 || ${H3_PROFILE:-0} != 1 ]]; then
            echo "Layer replay requires a separate profiled quantized trial." >&2
            exit 2
        fi
        export SGLANG_H3_QUANT_REPLAY="$H3_QUANT_REPLAY"
    fi
    if [[ "${quantization[0]}" != bf16 ]]; then
        args+=(--component-quantizations.transformer "${quantization[0]}"
            --component-quantization-ignored-layers.transformer "${quantization[@]:1}")
    fi
fi
printf '%q ' "${args[@]}" > "$output/launch.txt"
printf '\nvisible_devices=%s\nrungs=%s\nquality=%s\n' \
    "$visible" "${SGLANG_H3_DENOISE_RUNGS:-native}" "${H3_QUALITY:-lossless}" >> "$output/launch.txt"
printf 'laser_sp8_pad_heads=%s\n' "${SGLANG_H3_LASER_SP8_PAD_HEADS:-0}" >> "$output/launch.txt"
printf 'npu_adaln_memo=%s\n' "${SGLANG_H3_NPU_ADALN_MEMO:-default}" >> "$output/launch.txt"
printf 'window_mode=%s\nwindow_policy=%s\nwindow_global_steps=%s\nwindow_overlap=%s\nfamily=%s\n' \
    "${SGLANG_H3_WINDOW_MODE:-native}" "${SGLANG_H3_WINDOW_POLICY:-}" \
    "${SGLANG_H3_WINDOW_GLOBAL_STEPS:-0}" "${SGLANG_H3_WINDOW_OVERLAP:-0.08}" "$family" >> "$output/launch.txt"
printf 'checkpoint_manifest_sha256=%s\nvideo_shift=%s\naudio_shift=%s\nlaser_crop_before_cast=%s\nwindow_parity=%s\ncache_dit=%s\n' \
    "$identity" "$video_shift" "$audio_shift" "$SGLANG_H3_LASER_CROP_BEFORE_CAST" \
    "$SGLANG_H3_WINDOW_PARITY" "$SGLANG_CACHE_DIT_ENABLED" >> "$output/launch.txt"
cp "$H3_CHECKPOINT_MANIFEST" "$output/checkpoint.json"
git rev-parse HEAD > "$output/source.txt" 2>/dev/null || true
git diff --binary HEAD > "$output/source.diff" 2>/dev/null || true
echo "Artifacts: $output"
echo "Starting $case_id with $attention, TP2, SP$sp (Ulysses $((sp / ring)), Ring $ring)"
echo "Devices $visible (HTTP $port, rendezvous $master_port)"
if [[ ${H3_SERVE_ONLY:-0} == 1 ]]; then
    exec "${args[@]}"
fi
setsid env PYTHONUNBUFFERED=1 "${args[@]}" > "$output/server.log" 2>&1 &
server=$!
tail --pid="$server" -s 0.2 -n +1 -f "$output/server.log" &
log_follower=$!
cleanup() {
    kill -TERM -- "-$server" 2>/dev/null || true
    for _ in {1..20}; do
        kill -0 -- "-$server" 2>/dev/null || break
        sleep 1
    done
    kill -KILL -- "-$server" 2>/dev/null || true
    wait "$server" 2>/dev/null || true
    wait "$log_follower" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
deadline=$((SECONDS + ${H3_STARTUP_TIMEOUT:-1800}))
until curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    kill -0 "$server" 2>/dev/null && (( SECONDS < deadline )) || {
        echo "Server exited or startup timed out. See $output/server.log" >&2
        exit 1
    }
    sleep 2
done

export H3_PROMPT="${H3_PROMPT:-$default_prompt}"
echo "Server ready. Running $case_id requests."
python3 -u - "$output" "$port" "$duration" "$steps" "$case_id" "$served_model" "$video_shift" "$audio_shift" <<'PY'
import json
import os
from pathlib import Path
import sys
from threading import Event
import time

sys.path.insert(0, "test/manual/minimax_h3_realtime/research")
from checkpoint_contract import assert_endpoint, digest, validate_request
from h3_chunks import Client
from runner import probe_media

out, port, duration, steps, case, served_model, video_shift, audio_shift = sys.argv[1:]
out = Path(out)
duration = float(duration)
frames = round(duration * 24)
frames += (5 - frames) % 17
payload = dict(model=served_model, prompt=os.environ["H3_PROMPT"],
               task="t2va", conditions=[], target=dict(short_edge=768, aspect_ratio="16:9",
               duration_seconds=duration), num_outputs_per_prompt=1, num_inference_steps=int(steps),
               flow_shift=float(video_shift), audio_flow_shift=float(audio_shift),
               seed=int(os.getenv("H3_SEED", "1101")),
               quality=os.getenv("H3_QUALITY", "lossless"))
if case.startswith("HIST-"):
    # An explicit quality tier overrides the legacy generic Cache-DiT policy.
    payload.pop("quality")
client = Client(f"http://127.0.0.1:{port}", int(os.getenv("H3_REQUEST_TIMEOUT", "1800")))
validate_request(payload, os.environ["SGLANG_H3_RESEARCH_FAMILY"], served_model)
assert_endpoint(client, served_model)
(out / "request.json").write_text(json.dumps(payload, indent=2))
git_state = dict(
    sha=(out / "source.txt").read_text(),
    diff=(out / "source.diff").read_text(),
)
comparison = dict(
    request=payload, source_sha=git_state["sha"], source_diff_sha256=digest(git_state["diff"]),
    placement=os.environ["H3_LAUNCH_PLACEMENT"],
    adaln_memo=os.getenv("SGLANG_H3_NPU_ADALN_MEMO", "default"),
    attention_config=os.getenv("H3_ATTENTION_CONFIG"),
    crop_before_cast=os.environ["SGLANG_H3_LASER_CROP_BEFORE_CAST"],
    cache_dit=os.environ["SGLANG_CACHE_DIT_ENABLED"],
    profile=os.getenv("H3_PROFILE", "0"),
    quantization=(
        json.loads((out / "quantization-selection.json").read_text())
        if (out / "quantization-selection.json").exists()
        else None
    ),
    ffn_active_fraction=os.getenv("H3_FFN_ACTIVE_FRACTION", "1"),
)
(out / "comparison.json").write_text(json.dumps(comparison, indent=2))
profiling = os.getenv("H3_PROFILE") == "1"
repeats = 1 if profiling else int(os.getenv("H3_REPEATS", "3"))
if repeats < 1:
    raise ValueError("H3_REPEATS must be positive")
for index in range(repeats + 1):
    label = "warmup" if index == 0 else f"sample {index}/{repeats}"
    print(f"{case} {label}: submitting request", flush=True)
    media_file = out / f"sample-{index}.mp4"
    start = time.perf_counter()
    request = dict(payload)
    if profiling and index:
        request["perf_dump_path"] = str(out / f"sample-{index}.perf.json")
        request.update(profile=True, profile_all_stages=True, num_profiled_timesteps=int(steps))
    response = client.generate(request, media_file, Event())
    elapsed = time.perf_counter() - start
    http_timings = client.last_timing
    media = probe_media(media_file, require_stream_durations=True)
    if (media["width"], media["height"], media["fps"], media["frames"]) != (1344, 768, 24, frames):
        raise ValueError(f"Unexpected output geometry: {media}")
    if abs(media["duration_s"] - frames / 24) > 1 / 24 or abs(media["audio_duration_s"] - frames / 24) > 0.1:
        raise ValueError(f"Unexpected stream durations: {media}")
    if profiling and index:
        perf = json.loads((out / f"sample-{index}.perf.json").read_text())
        if perf["request_id"] != response["id"] or not perf["steps"]:
            raise ValueError("Missing request-scoped server stage timings")
    record = dict(case=case, warmup=index == 0, profiled=profiling and index > 0, e2e_s=elapsed, media=media, response=response, http_timings=http_timings)
    with (out / "samples.jsonl").open("a") as stream:
        print(json.dumps(record), file=stream)
    print(f"{case} {label}: {elapsed:.2f}s, {frames} frames", flush=True)
PY
if command -v npu-smi >/dev/null; then npu-smi info > "$output/npu-smi-after.txt"; fi
python3 - "$output" "$steps" "$video_shift" "$audio_shift" <<'PYLOG'
import ast
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

out, steps = Path(sys.argv[1]), int(sys.argv[2])
samples = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
mode = os.getenv("SGLANG_H3_WINDOW_MODE", "native")
global_steps = int(os.getenv("SGLANG_H3_WINDOW_GLOBAL_STEPS", "0"))
policy = "-"
if mode in ("local", "repack"):
    policy = "L" * steps
elif mode == "global-local":
    policy = os.getenv("SGLANG_H3_WINDOW_POLICY", "G" * global_steps + "L" * (steps - global_steps))
evaluations = steps + (policy.count("L") if mode in ("local", "global-local") else 0)
explicit_rungs = os.getenv("SGLANG_H3_DENOISE_RUNGS")
if explicit_rungs is None:
    grid = [1 - step / steps for step in range(steps + 1)]
else:
    grid = [int(rung) / 1000 for rung in explicit_rungs.split(",")] + [0]
pattern = re.compile(
    r"mode=(\S+) scheduler_updates=(\d+) dit_evaluations=(\d+) "
    r"request=(\S+) phase=request policy=(\S+).* sigma_video=(\[.*?\]) sigma_audio=(\[.*?\])"
)
expected_ids = {sample["response"]["id"] for sample in samples if not sample["warmup"]}
records = {}
for line in (out / "server.log").read_text().splitlines():
    match = pattern.search(line)
    if not match or match[4] not in expected_ids:
        continue
    if match[4] in records or match.group(1, 2, 3, 5) != (
        mode, str(steps), str(evaluations), policy
    ):
        raise RuntimeError(f"Unexpected denoising execution: {line}")
    for field, shift in ((6, float(sys.argv[3])), (7, float(sys.argv[4]))):
        actual = ast.literal_eval(match.group(field))
        expected = [shift * sigma / (1 + (shift - 1) * sigma) for sigma in grid]
        if len(actual) != len(expected) or not all(
            math.isclose(a, b, abs_tol=1e-6) for a, b in zip(actual, expected)
        ):
            raise RuntimeError(f"Unexpected sigma schedule: {line}")
    records[match[4]] = line
if os.getenv("SGLANG_H3_COUNT_BLOCKS") == "1":
    counters = {}
    count_pattern = re.compile(
        r"H3 block executions request=(\S+) cfg_forwards=(\d+) cache_mounted=(True|False) blocks_per_forward=(\[.*?\])"
    )
    for line in (out / "server.log").read_text().splitlines():
        match = count_pattern.search(line)
        if not match or match[1] not in expected_ids:
            continue
        counts = ast.literal_eval(match[4])
        expected_mount = os.environ["SGLANG_CACHE_DIT_ENABLED"] == "true"
        if match[1] in counters or int(match[2]) != evaluations or len(counts) != evaluations or min(counts) <= 0 or (match[3] == "True") != expected_mount:
            raise RuntimeError(f"Missing or invalid physical execution counts: {line}")
        if not expected_mount and len(set(counts)) != 1:
            raise RuntimeError(f"Unexpected block skipping in the cache-off control: {line}")
        counters[match[1]] = dict(blocks_per_forward=counts, cfg_forwards=int(match[2]),
                                  cache_mounted=expected_mount)
    if counters.keys() != expected_ids:
        raise RuntimeError("Missing measured-request physical execution counts")
    (out / "block_counts.json").write_text(json.dumps(counters, indent=2))
if missing := expected_ids - records.keys():
    raise RuntimeError(f"Missing denoising records for request IDs: {sorted(missing)}")
(out / "denoise.txt").write_text("\n".join(records.values()) + "\n")
(out / "denoise_matches.json").write_text(json.dumps(records, indent=2))
times = [sample["e2e_s"] for sample in samples if not sample["warmup"] and not sample["profiled"]]
if times:
    summary = dict(
        case=samples[0]["case"],
        median_e2e_s=statistics.median(times),
        samples_s=times,
        p90_e2e_s=sorted(times)[math.ceil(0.9 * len(times)) - 1],
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
PYLOG
