# SPDX-License-Identifier: Apache-2.0
"""R07 topology and transport capability probe (G0/G1 prerequisites).

Host-runnable; reports with explicit verdicts and never infers device
behavior from imports. On an allocated NPU host it also emits the G0 memory
snapshot commands and the G1 measurement plan (serial one-group vs two
disjoint concurrently running groups) for the fixed A3/A5 budgets.

Usage (from a worktree root; needs the repo runtime deps on the interpreter):
    PYTHONPATH=python /path/to/venv/python test/manual/minimax_h3_realtime/research/r07_topology_probe.py
Without them the transport/device sections still report; only the
plan_topologies section degrades to an explicit unavailable note.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys


def _module_report(module_name: str, names: tuple[str, ...]) -> dict:
    report: dict = {"module": module_name, "operators": {}}
    try:
        loaded = importlib.import_module(module_name)
        report["status"] = "imported"
    except Exception as error:  # noqa: BLE001 - probe must not crash
        report["status"] = f"unavailable: {type(error).__name__}: {error}"
        for name in names:
            report["operators"][name] = "unknown (module unavailable)"
        return report
    for name in names:
        report["operators"][name] = (
            "available" if hasattr(loaded, name) else "missing"
        )
    return report


def collect() -> dict:
    result: dict = {
        "probe": "r07_topology_probe",
        "transports": {},
        "device": {},
        "topologies": {},
        "screen_cases": [
            {"id": "R07.A3.16.serial", "soc": "A3", "devices": 16, "groups": [[16]], "status": "runnable_when_allocation_is_supplied"},
            {"id": "R07.A3.8.serial", "soc": "A3", "devices": 8, "groups": [[8]], "status": "runnable_when_allocation_is_supplied"},
            {"id": "R07.A3.2x8.concurrent", "soc": "A3", "devices": 16, "groups": [[8], [8]], "status": "blocked_until_subgroup_proof"},
            {"id": "R07.A5.8.serial", "soc": "A5", "devices": 8, "groups": [[8]], "status": "runnable_when_allocation_is_supplied"},
            {"id": "R07.A5.2x4.concurrent", "soc": "A5", "devices": 8, "groups": [[4], [4]], "status": "blocked_until_subgroup_proof"},
        ],
        "g1_plan": [],
        "note": (
            "Transport import availability is necessary, not sufficient: no "
            "bulk-latent transport for H3 AV state is verified on Ascend. "
            "GLM #31320 carries prior tokens, not tensors; #40957's Mooncake "
            "path needs an Ascend capability measurement before any worker "
            "uses it."
        ),
    }

    result["transports"]["zmq_control"] = _module_report("zmq", ())
    result["transports"]["mooncake_tensor"] = _module_report(
        "mooncake_transfer_engine", ("TransferEngine",)
    )
    result["transports"]["torch"] = _module_report(
        "torch", ("distributed", "get_device_module")
    )

    npu = _module_report("torch.npu", ("device_count", "memory_summary"))
    device_count = None
    if npu["status"] == "imported" and npu["operators"]["device_count"] == "available":
        try:
            device_count = int(__import__("torch").npu.device_count())
        except Exception:  # noqa: BLE001 - probing must not crash
            device_count = None
    result["device"] = {
        "torch_npu": npu,
        "device_count": device_count,
        "npu_smi": (
            {"status": "available"}
            if shutil.which("npu-smi")
            else {"status": "unavailable"}
        ),
    }

    for label, budget in (("a3_16", 16), ("a3_8", 8), ("a5_8", 8), ("a5_4", 4)):
        try:
            from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.distributed_windows import (
                plan_topologies,
            )

            result["topologies"][label] = plan_topologies(devices=budget)
        except Exception as error:  # noqa: BLE001 - probe must not crash
            result["topologies"][label] = {
                "status": (
                    f"unavailable (needs the sglang package on this "
                    f"interpreter: {type(error).__name__}: {error})"
                )
            }

    # G1 runnable plan: same window policy/seed, two orders of execution.
    result["g1_plan"] = [
        {
            "case": "G1-serial",
            "steps": [
                "run window A forward to completion on the full group",
                "then run window B forward on the full group",
                "record T(A), T(B) and HBM per group member",
            ],
            "metric": "T(A) + T(B)",
            "launcher": "r07_topology_screen.py R07.<SOC>.<N>.serial --groups <ids> --output receipt.json -- <short-window-command>",
        },
        {
            "case": "G1-concurrent",
            "steps": [
                "verify subgroup collectives for the two disjoint groups",
                "snapshot HBM of both groups before launch",
                "run A and B concurrently; record wall clock, per-group HBM, "
                "fabric counters",
            ],
            "metric": "max(T_A, T_B) + transfer + fusion + barrier",
            "blocked_on": "subgroup collective layout proof + per-group HBM fit",
            "launcher": "r07_topology_screen.py R07.<SOC>.2x<N>.concurrent --groups <idsA>;<idsB> --output receipt.json -- <short-window-command>",
        },
    ]
    return result


def _memory_snapshot_commands() -> list[str]:
    return [
        "npu-smi info",
        "python -c 'import torch, torch_npu; print(torch.npu.memory_summary())'",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()
    result = collect()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    print("R07 topology and transport capability probe")
    for name, report in sorted(result["transports"].items()):
        print(f"  transport {name}: {report['status']}")
    print(f"  device: count={result['device']['device_count']}")
    for label, plans in sorted(result["topologies"].items()):
        if "status" in plans:
            print(f"  {label}: {plans['status']}")
            continue
        for plan in plans:
            print(
                f"  {label}: {plan['kind']} groups={plan['groups']} "
                f"verified={plan['verified']}"
            )
    print("  G0 memory snapshot commands:")
    for command in _memory_snapshot_commands():
        print(f"    {command}")
    print(f"\n{result['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
