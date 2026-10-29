#!/usr/bin/env bash
# Run inside the vLLM container against the separately managed SGLang server.
set -euo pipefail

profile=${1:?Usage: bench_ab.sh PROFILE RUN [--print]}
run=${2:?Usage: bench_ab.sh PROFILE RUN [--print]}
case "$profile" in
    baseline|metadata-only|metadata|native|tile4|tile8|tile16|rope-kv|combined|combined-mtp|combined-mtp-baseline|eager-prefill|delayer-off|ratio025|delay25|cap5|mtp-baseline|mtp|r025-ms50-cap6|r025-ms25-cap6|r05-ms25-cap6|r05-ms25-cap5|r05-ms50-cap5|r025-ms50-cap5|r025-ms25-cap5) ;;
    *) echo 'Unknown benchmark profile' >&2; exit 2 ;;
esac
[[ "$run" =~ ^[1-9][0-9]*$ ]] || { echo 'RUN must be a positive integer' >&2; exit 2; }
[[ -z "${3:-}" || "${3:-}" == --print ]] || { echo 'Expected --print' >&2; exit 2; }
url=${GEMMA_BENCH_URL:-http://127.0.0.1:30014}
tokenizer=${GEMMA_TOKENIZER:-/home/test1/models/gemma4-26B-A4B-w8a8}
results=${GEMMA_RESULT_DIR:-./results}
stem=${profile}_c24_run${run}
cmd=(vllm bench serve --backend openai-chat --base-url "$url"
    --endpoint /v1/chat/completions --model gemma4-w8a8-test --tokenizer "$tokenizer"
    --dataset-name random --random-prefix-len 250 --random-input-len 1100
    --random-output-len 100 --random-range-ratio 0 --max-concurrency 24
    --request-rate 87 --num-prompts 1600 --temperature 0 --ignore-eos
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99
    --save-result --result-dir "$results" --result-filename "$stem.json")
printf '%q ' "${cmd[@]}"
printf '\n'
[[ "${3:-}" != --print ]] || exit 0
mkdir -p -- "$results"
[[ ! -e "$results/$stem.json" ]] || { echo 'Result exists; choose a new RUN number' >&2; exit 2; }

snapshot() {
    python3 - "$url" "$results/$stem.$1.server-info.json" "$profile" "$1" <<'PY'
import json
import sys
import urllib.request
from pathlib import Path

with urllib.request.urlopen(sys.argv[1].rstrip("/") + "/server_info", timeout=30) as response:
    info = json.load(response)
Path(sys.argv[2]).write_text(json.dumps(info, indent=2))
args = info.get("server_args", {})
algorithm = args.get("speculative_algorithm", info.get("speculative_algorithm"))
if sys.argv[4] == "before":
    if sys.argv[3] in ("mtp", "combined-mtp") and algorithm not in ("FROZEN_KV_MTP", "NEXTN"):
        raise SystemExit(f"Expected Gemma MTP server, got algorithm={algorithm!r}")
    if sys.argv[3] in ("mtp-baseline", "combined-mtp-baseline") and algorithm not in (None, "", "NONE"):
        raise SystemExit(f"Expected non-speculative baseline, got algorithm={algorithm!r}")
for rank, state in enumerate(info.get("internal_states", [])):
    if "avg_spec_accept_length" in state:
        print(f"replica={rank} avg_spec_accept_length={state['avg_spec_accept_length']}")
PY
}
snapshot before
status=0
"${cmd[@]}" || status=$?
snapshot after
exit "$status"
