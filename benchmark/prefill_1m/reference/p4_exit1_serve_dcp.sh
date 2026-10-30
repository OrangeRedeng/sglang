#!/usr/bin/env bash
# Portable archived BDE-v2 runtime recipe. Add --real-weights for quality tests.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "${script_dir}/../launch_bde_v2.py" "$@"
