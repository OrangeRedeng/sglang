#!/usr/bin/env bash
# Run on the host; take the runner from the same checkout as the container server.
set -euo pipefail

container=${SGLANG_CONTAINER:-sglang-a3}
repo=${SGLANG_REPO:-/home/swx1199799/megamoe_test/sglang}
task_dir=$(mktemp -d /tmp/sglang-overnight.XXXXXX)
runner="$task_dir/run_bde_overnight.py"
trap 'rm -f -- "$runner"; rmdir -- "$task_dir"' EXIT

docker exec -w "$repo" "$container" git log -1 --format='%h %s'
docker cp "$container:$repo/benchmark/prefill_1m/run_bde_overnight.py" "$runner"
python3 -u "$runner" --container "$container" --repo "$repo" "$@"
