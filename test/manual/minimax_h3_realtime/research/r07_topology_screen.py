# SPDX-License-Identifier: Apache-2.0
"""Launch one fixed-budget R07 short-window screen with fail-closed gates.

The command is supplied by the operator and must emit its normal run receipt.
This wrapper only owns device-group selection and records wall-clock results.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def run(case: str, groups: list[list[str]], command: list[str], output: Path) -> dict:
    if not groups or any(not group for group in groups):
        raise ValueError("R07 requires non-empty device groups")
    if len({device for group in groups for device in group}) != sum(
        len(group) for group in groups
    ):
        raise ValueError("R07 device groups must be disjoint")
    if not command:
        raise ValueError("R07 requires an actual short-window command")
    if not os.getenv("ASCEND_RT_VISIBLE_DEVICES"):
        raise RuntimeError("ASCEND_RT_VISIBLE_DEVICES is required for a device screen")
    expected = {device for group in groups for device in group}
    visible = set(os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(","))
    if expected != visible:
        raise RuntimeError("visible devices do not exactly match the requested groups")
    started = time.monotonic()
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    receipt = {
        "case": case,
        "groups": groups,
        "command": command,
        "returncode": completed.returncode,
        "elapsed_s": time.monotonic() - started,
        "stdout": completed.stdout[-20000:],
        "stderr": completed.stderr[-20000:],
        "status": "completed" if completed.returncode == 0 else "failed",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    if completed.returncode:
        raise RuntimeError(f"R07 screen failed with exit code {completed.returncode}")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("--groups", required=True, help="comma-separated groups, e.g. 0,1;2,3")
    parser.add_argument("--output", required=True, type=Path)
    args, command = parser.parse_known_args()
    groups = [group.split(",") for group in args.groups.split(";")]
    if command and command[0] == "--":
        command = command[1:]
    run(args.case, groups, command, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
