# SPDX-License-Identifier: Apache-2.0
"""R06 VAE graph capability probe (device prerequisites, no correctness claims).

Reports whether the installed TorchNPU exposes the graph capture/replay and
side-stream APIs the R06 paths need. Host-runnable: without a device it
reports ``unavailable`` instead of failing. A successful import is necessary,
not sufficient — capture, replay byte parity and memory behavior are only
proven by the operator's device campaign (V2/V3/V4 cases).
"""

from __future__ import annotations

import argparse
import importlib
import json
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
    torch_npu = _module_report(
        "torch",
        (),  # torch imports without torch_npu; the npu module is what matters
    )
    del torch_npu
    result: dict = {
        "probe": "r06_vae_graph_probe",
        "subtasks": {},
        "note": (
            "API availability gates capture attempts at runtime; it never "
            "proves replay byte parity, capture time, or graph-pool memory. "
            "Run cases V0-V4 from the R06 report on allocated devices."
        ),
    }

    torch_npu = _module_report(
        "torch.npu",
        (
            "graph",
            "NPUGraph",
            "graph_pool_handle",
            "current_stream",
            "Stream",
        ),
    )
    api_ok = torch_npu["status"] == "imported" and "available" in set(
        torch_npu["operators"].values()
    )
    result["subtasks"]["V2_V3_tile_decoder_npugraph"] = dict(
        operators=torch_npu,
        verdict="available" if api_ok else "blocked",
        detail=(
            "TileDecodeGraphCache captures one stateless per-group decode "
            "forward into a static input/output pair with a shared bounded "
            "pool (SGLANG_H3_VAE_NPU_GRAPH=1). Runtime gates re-check every "
            "capture and fall back to eager."
        ),
    )

    streams = _module_report("torch.npu", ("Stream", "current_stream", "stream"))
    stream_ok = streams["status"] == "imported" and "available" in set(
        streams["operators"].values()
    )
    result["subtasks"]["V4_audio_side_stream"] = dict(
        operators=streams,
        verdict="available" if stream_ok else "blocked",
        detail=(
            "_overlaps_audio_decode now accepts NPU with a resident audio "
            "VAE and decode world > 1; the side stream, device switch and "
            "record_stream/wait_stream run through torch.get_device_module."
        ),
    )

    result["subtasks"]["V5_output_pipeline_overlap"] = dict(
        operators={"status": "design-pending"},
        verdict="deferred",
        detail=(
            "Bounded double-buffered gather/blend/D2H/encode assembly is "
            "scoped after V4 device evidence; no code path was added yet."
        ),
    )
    result["subtasks"]["V7_dit_graph"] = dict(
        operators={"status": "conditional"},
        verdict="deferred",
        detail=(
            "DiT graph feasibility stays conditional on R04/R05 geometry and "
            "collective support; no capture was attempted in R06."
        ),
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()
    result = collect()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    print("R06 VAE graph capability probe")
    for name, subtask in sorted(result["subtasks"].items()):
        print(f"\n[{name}] verdict: {subtask['verdict']}")
        for key, value in subtask.items():
            if key != "verdict":
                print(f"  {key}: {json.dumps(value, sort_keys=True)}")
    print(f"\n{result['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
