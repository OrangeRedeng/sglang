# SPDX-License-Identifier: Apache-2.0
"""R05 communication-path capability probe (E5/E6/E7 prerequisites).

Reports which Ascend operators the R05 overlap and communication tracks need,
with an explicit available/blocked verdict per subtask. Host-runnable: without
a device the probe reports ``unavailable`` instead of failing, and never
infers device correctness from a successful import.

Usage: python3 r05_comm_probe.py [--json]
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys


def _operator_report(module_name: str, names: tuple[str, ...]) -> dict:
    """Availability of named operators on one module; never raises."""
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
        "probe": "r05_comm_probe",
        "subtasks": {},
        "note": (
            "Import availability is necessary, not sufficient: every "
            "subtask still needs the operator-driven device campaign for "
            "numeric parity, bandwidth and overlap evidence."
        ),
    }

    # E4: fused residual + norm, and the exact-default guard.
    e4 = _operator_report("torch_npu", ("npu_add_rms_norm", "npu_rms_norm"))
    result["subtasks"]["E4_residual_norm_fusion"] = dict(
        operators=e4,
        verdict="available"
        if "available" in e4["operators"].values()
        else "blocked",
        detail=(
            "npu_add_rms_norm gates the fused gate+norm2 and cross-block "
            "norm1 paths; absent operators fall back to the exact standalone "
            "BF16 chain at runtime (hasattr guards in minimax_h3.py)."
        ),
    )

    # E5: Ulysses head-group pipelining needs the Laser operator and HCCL
    # all_to_all on the Ulysses group.
    laser = _operator_report("attentions", ("la",))
    result["subtasks"]["E5_ulysses_head_group_pipeline"] = dict(
        operators={"laser_attention": laser, "collective": "torch.distributed"},
        verdict="available"
        if laser["status"] == "imported" and "available" in laser["operators"].values()
        else "blocked",
        detail=(
            "Gated at runtime by: LASER_ATTN backend, global rows >= 8192 "
            "(_NPU_LASER_PIPELINE_MIN_SEQ_LEN), heads divisible by "
            "world_size*groups, and not under a breakable CUDA graph. "
            "Bit-exactness of a head-group Laser call against the full-head "
            "call is device-pending evidence, not an import fact."
        ),
    )

    # E6 probe: row-sharded TP (matmul fused with ReduceScatter/AllGather).
    e6 = _operator_report(
        "torch_npu",
        (
            "npu_matmul_reduce_scatter",
            "npu_matmul_all_gather",
            "npu_mm_reduce_scatter",
            "npu_mm_all_gather",
        ),
    )
    present = [name for name, state in e6["operators"].items() if state == "available"]
    result["subtasks"]["E6_row_shard_tp_probe"] = dict(
        operators=e6,
        verdict="available" if present else "blocked",
        available_operators=present,
        detail=(
            "No runtime path uses these yet: R05 owns a probe only, and a TP "
            "rewrite stays blocked until the operator confirms which fused "
            "GEMM/collective interfaces exist on the installed CANN, plus "
            "rounding and scale-layout contracts."
        ),
    )

    # E7 probe: quantized pre-A2A QKV exchange.
    e7 = _operator_report(
        "torch_npu",
        (
            "npu_quant_matmul",
            "npu_dynamic_quant",
        ),
    )
    result["subtasks"]["E7_quantized_a2a_probe"] = dict(
        operators=e7,
        verdict="available"
        if "available" in e7["operators"].values()
        else "blocked",
        detail=(
            "Probe only. Quantizing QKV before the Ulysses exchange must "
            "carry packed values and scales to an attention operator that "
            "accepts them; re-quantization or forced low attention precision "
            "is rejected by the R02/R08 quality contracts."
        ),
    )

    # Deferred in this port: the #53105 NPU Triton fused kernels.
    triton = _operator_report("triton", ("__version__",))
    result["subtasks"]["deferred_npu_triton_kernels"] = dict(
        operators=triton,
        verdict="deferred",
        detail=(
            "laser_attn_triton / minimax_h3_npu_triton from #53105 were "
            "deliberately not ported (no new kernels in R05). They remain an "
            "optional follow-up and need triton_ascend plus operator review."
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
    print("R05 communication-path capability probe")
    for name, subtask in sorted(result["subtasks"].items()):
        print(f"\n[{name}] verdict: {subtask['verdict']}")
        for key, value in subtask.items():
            if key != "verdict":
                print(f"  {key}: {json.dumps(value, sort_keys=True)}")
    print(f"\n{result['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
