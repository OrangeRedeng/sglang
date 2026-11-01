#!/usr/bin/env bash
# Run on the A3 host. Each invocation owns one foreground experiment and log.
set -euo pipefail
mode=${1:-quick}
if (($#)); then shift; fi
container=${SGLANG_CONTAINER:-sglang-a3}
repo=${SGLANG_REPO:-/home/swx1199799/megamoe_test/sglang}
root=${A3_RESULTS_ROOT:-/home/tbaydasov/aisbench/overnight-results}
stamp=$(date +%Y%m%d-%H%M%S)
out="$root/retest-$mode-$stamp"
mkdir -p "$out"
exec > >(tee -a "$out/driver.log") 2>&1
scratch=$(mktemp -d /tmp/sglang-a3-retest.XXXXXX)
trap 'rm -rf -- "$scratch"' EXIT

docker exec -w "$repo" "$container" git log -1 --format='%H %s' | tee "$out/commit.txt"
docker exec -w "$repo" "$container" git diff --stat > "$out/source-diff.txt"
docker exec "$container" npu-smi info > "$out/npu-smi-before.txt"
args=(--feature-kv-rows 8192 --chunk 16384)
if [[ -n ${W8A8_MODEL_PATH:-} ]]; then
  args+=(--w8a8-checkpoint "$W8A8_MODEL_PATH")
fi
case "$mode" in
  quick)
    args+=(--feature-stages serve --feature-cases serve-baseline --variants baseline --repeats 1
           --requests 2 --moe-length 1024 --moe-output-len 16 --moe-kv-rows 8192)
    ;;
  moe)
    args+=(--feature-cases 'serve-baseline,serve-multistream,serve-auto-stream-thresholds,serve-auto-hccl,serve-autotune-hccl'
           --variants baseline,multistream,auto-stream-thresholds,auto-hccl,autotune-hccl --repeats 3)
    ;;
  dcp)
    args+=(--feature-cases 'serve-dcp-packed-read,serve-dcp-gather-prefetch,serve-dcp-sfa-overlap,serve-dcp-decode-q-preissue'
           --variants dcp-packed-read,dcp-gather-prefetch,dcp-sfa-overlap,dcp-decode-q-preissue --repeats 3)
    ;;
  failures)
    args+=(--features-only --feature-cases 'serve-baseline,serve-w8a8-load,serve-w4a4-mixed-load,serve-online-*,serve-tp-*,serve-indexer-int8,serve-dcp-full-width-q,serve-dcp-legacy-token-shard,serve-dcp-stream-o-proj,serve-bde-v2,serve-graph-*,serve-fuseep-*,serve-megamoe')
    ;;
  all)
    args+=(--repeats 3)
    ;;
  *) printf 'Usage: %s {quick|moe|dcp|failures|all} [runner arguments]\n' "$0"; exit 2 ;;
esac

docker cp "$container:$repo/benchmark/prefill_1m/start_bde_overnight.sh" "$scratch/start.sh"
export SGLANG_CONTAINER="$container" SGLANG_REPO="$repo"
bash "$scratch/start.sh" --check-paths
printf 'Artifacts: %s\n' "$out"
# User overrides come last; qualification gates and the runner lock still apply.
set +e
bash "$scratch/start.sh" "${args[@]}" --output "$out/results" "$@"
code=$?
set -e
docker exec "$container" npu-smi info > "$out/npu-smi-after.txt" || true
cp "$out/driver.log" "$out/driver-snapshot.log"
tar -czf "$out.tar.gz" --exclude="$(basename "$out")/driver.log" \
  -C "$(dirname "$out")" "$(basename "$out")"
printf 'Runner exit: %s\nResults: %s/results/SUMMARY.md\nArchive: %s.tar.gz\n' "$code" "$out" "$out"
exit "$code"
