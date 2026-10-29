#!/usr/bin/env bash
# One foreground server per profile; the benchmark runs in another container.
set -euo pipefail

profile=${1:-metadata}
print_only=${2:-}
if [[ -n "$print_only" && "$print_only" != --print ]]; then
    echo 'Usage: bash test/manual/gemma4/serve_ab.sh PROFILE [--print]' >&2
    exit 2
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=${GEMMA_ROOT:-$(cd -- "$script_dir/../../.." && pwd)}
server_root=$root
model=${GEMMA_MODEL:-/home/test1/models/gemma4-26B-A4B-w8a8}
python=${GEMMA_PYTHON:-python3}
norm=${SGLANG_ENABLE_NPU_GEMMA4_QKV_NORM:-1}
ratio=0.5
delay=50
cap=6
delayer=1
prefill_backend=tc_piecewise
memory=0.92
extra=()
case "$profile" in
    baseline) server_root=${GEMMA_BASELINE_ROOT:-$root/../sglang-gemma4-baseline} ;;
    metadata-only) server_root=${GEMMA_METADATA_ROOT:-$root/../sglang-gemma4-metadata} ;;
    metadata) ;;
    native) norm=0 ;;
    eager-prefill) prefill_backend=disabled ;;
    delayer-off) delayer=0 ;;
    ratio025) ratio=0.25 ;;
    delay25) delay=25 ;;
    cap5) cap=5 ;;
    r025-ms50-cap6) ratio=0.25 ;;
    r025-ms25-cap6) ratio=0.25; delay=25 ;;
    r05-ms25-cap6) delay=25 ;;
    r05-ms25-cap5) delay=25; cap=5 ;;
    r05-ms50-cap5) cap=5 ;;
    r025-ms50-cap5) ratio=0.25; cap=5 ;;
    r025-ms25-cap5) ratio=0.25; delay=25; cap=5 ;;
    mtp-baseline|mtp|mtp-graph-baseline|mtp-graph)
        delayer=0
        memory=0.82
        prefill_backend=disabled
        extra+=(--disable-overlap-schedule)
        if [[ "$profile" == mtp || "$profile" == mtp-graph ]]; then
            assistant=${GEMMA_ASSISTANT_MODEL:-/home/test1/models/gemma-4-26B-A4B-it-assistant}
            extra+=(--speculative-algorithm NEXTN --speculative-draft-model-path "$assistant"
                --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
        fi
        ;;
    *) echo "Unknown profile: $profile" >&2; exit 2 ;;
esac

export ASCEND_RT_VISIBLE_DEVICES=2,3
export TASK_QUEUE_ENABLE=1
export ASCEND_USE_FIA=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_ENABLE_NPU_ADAPTIVE_TASK_QUEUE=0
export SGLANG_ENABLE_NPU_GEMMA4_QKV_NORM=$norm
export PYTHONPATH="$server_root/python${PYTHONPATH:+:$PYTHONPATH}"

decode='{"backend":"full","bs":[1,2,4,8,12,15,16,24,28,32,48,64]}'
prefill='{"backend":"tc_piecewise","bs":[128,256,1280,1536,2048,2560,3072,4096,5632,5760,6400,7680,8192]}'
if [[ "$prefill_backend" == disabled ]]; then
    prefill='{"backend":"disabled"}'
fi
if [[ "$profile" == mtp || "$profile" == mtp-baseline ]]; then
    decode='{"backend":"disabled"}'
fi
cmd=("$python" -m sglang.launch_server
    --model-path "$model" --served-model-name gemma4-w8a8-test
    --host 0.0.0.0 --port 30014 --device npu --attention-backend ascend
    --tp-size 1 --dp-size 2 --context-length 8192 --max-running-requests 64
    --max-prefill-tokens 8192 --chunked-prefill-size 8192 --mem-fraction-static "$memory"
    --cuda-graph-config "{\"prefill\":$prefill,\"decode\":$decode}"
    --prefill-max-requests "$cap" --prefill-delayer-queue-min-ratio "$ratio"
    --prefill-delayer-max-delay-ms "$delay" --prefill-delayer-max-delay-passes 4
    --trust-remote-code)
if [[ "$delayer" == 1 ]]; then
    cmd+=(--enable-prefill-delayer)
fi
cmd+=("${extra[@]}")

printf 'Profile=%s source=%q\n' "$profile" "$server_root"
printf 'TASK_QUEUE_ENABLE=1 QKV_NORM=%s\n' "$norm"
printf 'FIA_METADATA_REUSE=%s FIA_KV_VIEW_REUSE=%s\n' \
    "${SGLANG_ENABLE_NPU_FIA_METADATA_REUSE:-0}" "${SGLANG_ENABLE_NPU_FIA_KV_VIEW_REUSE:-0}"
printf '%q ' "${cmd[@]}"
printf '\n'
if [[ "$print_only" == --print ]]; then
    exit 0
fi
[[ -f "$server_root/python/sglang/launch_server.py" ]] || { echo 'Missing server checkout' >&2; exit 2; }
[[ -d "$model" ]] || { echo 'Missing model directory' >&2; exit 2; }
if [[ "$profile" == baseline ]]; then
    baseline_sha=$(git -C "$server_root" rev-parse HEAD)
    [[ "$baseline_sha" == c8d7020179* ]] || { echo 'Baseline must be pinned to c8d7020179' >&2; exit 2; }
fi
if [[ "$profile" == metadata-only ]]; then
    metadata_sha=$(git -C "$server_root" rev-parse HEAD)
    [[ "$metadata_sha" == 420e720def* ]] || { echo 'Metadata-only checkout must be pinned to 420e720def' >&2; exit 2; }
fi
if [[ "$profile" == mtp || "$profile" == mtp-graph ]]; then
    [[ -f "$assistant/config.json" ]] || { echo 'Missing matching assistant checkpoint' >&2; exit 2; }
fi
printf 'Commit: '
git -C "$server_root" rev-parse HEAD
cd -- "$server_root"
exec "${cmd[@]}"
