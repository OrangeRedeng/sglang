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
docker exec -w "$repo" "$container" git diff --binary HEAD > "$out/source-diff.txt"
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
  megamoe)
    args+=(--features-only --feature-stages serve --feature-cases serve-baseline,serve-megamoe
           --max-hours 1 --feature-timeout 3600)
    ;;
  metadata-compare)
    args+=(--performance-only --reference-recipe historical-bde
           --variants baseline,dsa-metadata,dsa-metadata-batch2
           --length 1000000 --requests 16 --maxreq 1 --chunk 32768
           --kv-rows 67584 --context 1048576 --memfrac 0.85 --repeats 1
           --megamoe-dispatch-tokens 2048 --startup-timeout 1200 --max-hours 3)
    ;;
  megamoe-compare)
    args+=(--performance-only --reference-recipe historical-bde --variants baseline,megamoe
           --length 1000000 --requests 16 --maxreq 1 --chunk 32768
           --kv-rows 67584 --context 1048576 --memfrac 0.85 --repeats 1
           --megamoe-dispatch-tokens 2048 --startup-timeout 1200
           --max-hours 2)
    ;;
  megamoe-diagnostic)
    args+=(--feature-stages serve --reference-recipe historical-bde --allow-parity-failure
           --feature-cases serve-baseline,serve-megamoe --variants baseline,megamoe
           --length 1000000 --requests 16 --maxreq 1 --chunk 32768
           --kv-rows 67584 --context 1048576 --memfrac 0.85 --repeats 1
           --megamoe-dispatch-tokens 2048 --startup-timeout 1200
           --max-hours 2 --feature-timeout 7200)
    ;;
  historical-repro)
    args+=(--feature-stages serve --reference-recipe historical-bde --feature-cases serve-historical-bde --variants historical-bde
           --length 1000000 --requests 16 --maxreq 1 --chunk 32768
           --kv-rows 67584 --context 1048576 --memfrac 0.85 --repeats 1
           --max-hours 2 --feature-timeout 7200)
    ;;
  baseline-repro)
    args+=(--features-only --feature-stages long --long-variants dcp-baseline
           --length 1000000 --long-lengths 1000000 --long-concurrency 16 --long-requests 16
           --long-cache-modes prefix99 --long-warmup once
           --long-kv-rows 81920 --context 1048576 --chunk 16384 --repeats 1
           --max-hours 2 --feature-timeout 7200)
    ;;
  megamoe-500k)
    args+=(--features-only --feature-stages long --diagnostic-megamoe
           --long-variants moe-baseline,megamoe --long-dcp-size 16
           --length 500000 --long-lengths 500000 --long-concurrency 16 --long-requests 16
           --long-cache-modes prefix99 --long-warmup once --megamoe-dispatch-tokens 1024
           --long-kv-rows 39936 --context 524288 --chunk 16384 --repeats 1
           --max-hours 2 --feature-timeout 7200)
    ;;
  megamoe-measure)
    args+=(--features-only --feature-stages long --diagnostic-megamoe
           --long-variants moe-baseline,megamoe --long-dcp-size 16
           --length 1000000 --long-lengths 1000000 --long-concurrency 16 --long-requests 16
           --long-cache-modes prefix99 --long-warmup once
           --long-kv-rows 74752 --context 1048576 --chunk 16384 --repeats 1
           --megamoe-dispatch-tokens 1024
           --max-hours 2 --feature-timeout 7200)
    ;;
  fix-check)
    args+=(--features-only --feature-stages serve,long
           --feature-cases serve-baseline,serve-megamoe,serve-dcp-packed-read,serve-dcp-gather-prefetch,serve-dcp-sfa-overlap
           --long-variants dcp-baseline --long-lengths 1000000 --long-concurrency 1
           --long-requests 1 --long-cache-modes prefix99 --long-warmup once
           --long-kv-rows 81920 --repeats 3 --max-hours 2 --feature-timeout 7200)
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
  overnight-repair)
    args+=(--feature-stages host,npu,serve --reference-recipe historical-bde --repeats 1
           --feature-cases 'serve-baseline,serve-tp-*,serve-indexer-int8,serve-indexer-fp16,serve-megamoe,serve-graph-*,serve-auto-stream-thresholds,serve-auto-hccl,serve-autotune-hccl,serve-moe-stream-auto,serve-moe-fused-combined'
           --variants 'baseline,megamoe,tp-routed-scale,tp-finalize,tp-prequant,tp-shared,tp-shared-activation,indexer-fp16,auto-stream-thresholds,auto-hccl,autotune-hccl,moe-stream-auto,moe-fused-combined'
           --length 1000000 --requests 16 --maxreq 1 --chunk 32768
           --kv-rows 67584 --context 1048576 --memfrac 0.85
           --megamoe-dispatch-tokens 2048 --startup-timeout 1200
           --max-hours 9.5 --feature-timeout 34200)
    ;;
  overnight|overnight-core)
    args+=(--features-only --feature-stages host,npu,serve,long --repeats 3
           --feature-cases 'serve-baseline,serve-tp-*,serve-multistream,serve-auto-stream-thresholds,serve-auto-hccl,serve-autotune-hccl,serve-indexer-int8,serve-indexer-query-shard,serve-fuseep-*,serve-megamoe,serve-moe-stream-auto,serve-moe-fused-combined'
           --long-lengths 1000000 --long-concurrency 1 --long-requests 1
           --long-warmup once --long-kv-rows 81920 --max-hours 9.5
           --feature-timeout 34200
           --long-variants 'dcp-baseline,dcp-packed-read,dcp-gather-prefetch,dcp-sfa-overlap,dcp-indexer-query-shard,dcp-multistream,dcp-io-combined')
    if [[ $mode == overnight-core ]]; then
      args+=(--long-variants 'dcp-baseline,dcp-packed-read,dcp-gather-prefetch,dcp-sfa-overlap,dcp-io-combined')
    fi
    ;;
  all)
    args+=(--repeats 3)
    ;;
  *) printf 'Usage: %s {quick|megamoe|metadata-compare|megamoe-compare|megamoe-diagnostic|historical-repro|baseline-repro|megamoe-500k|megamoe-measure|fix-check|moe|dcp|failures|overnight-repair|overnight|overnight-core|all} [runner arguments]\n' "$0"; exit 2 ;;
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
python3 - "$out" <<'PYLOGS'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
empty = [path for path in sorted(root.rglob("*")) if path.is_file() and path.stat().st_size == 0]
(root / "removed-empty-files.json").write_text(json.dumps([str(path.relative_to(root)) for path in empty], indent=2) + "\n")
for path in empty:
    path.unlink()
PYLOGS
tar -czf "$out.tar.gz" --exclude="$(basename "$out")/driver.log" \
  -C "$(dirname "$out")" "$(basename "$out")"
printf 'Runner exit: %s\nResults: %s/results/SUMMARY.md\nArchive: %s.tar.gz\n' "$code" "$out" "$out"
exit "$code"
