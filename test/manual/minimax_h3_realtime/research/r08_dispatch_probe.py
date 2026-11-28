# SPDX-License-Identifier: Apache-2.0
"""R08 attention dispatch probe (A0: ERR00100 isolation prerequisites).

Separately probes the three dispatch layers the H08 failure could involve,
with complete traceback capture and installed-package provenance, so the
failing operator is identified before any installation change:

1. ``torch_npu.npu_fused_infer_attention_score`` (native FA call).
2. The repo's Ascend FA backend import/init path (dense fallback).
3. ``mindiesd.sparse_attention`` (the MindIE-SD sparse op).

Host-runnable: without a device every verdict is ``unavailable``/``blocked``
instead of crashing. Import success is necessary, not sufficient - the
minimal-call probes only run where a device exists, and any operator error
is reported with its full traceback for the operator campaign.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import sys
import traceback


def _package_versions(names: tuple[str, ...]) -> dict:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _signature_report(module_name: str, names: tuple[str, ...]) -> dict:
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
        entry: dict = {}
        function = getattr(loaded, name, None)
        entry["present"] = function is not None
        try:
            import inspect

            entry["signature"] = (
                str(inspect.signature(function)) if function else None
            )
        except (TypeError, ValueError):
            entry["signature"] = "not introspectable"
        report["operators"][name] = entry
    return report


def _minimal_call_report(label: str, callable_fn, kwargs: dict) -> dict:
    """Run one minimal operator call; capture the complete traceback.

    An error here is a *finding* (recorded with traceback), never a crash of
    the probe. Run on the device host only; skipped when tensors cannot be
    created.
    """
    try:
        callable_fn(**kwargs)
    except Exception as error:  # noqa: BLE001 - full capture is the point
        return {
            "label": label,
            "status": "error",
            "exception": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    return {"label": label, "status": "ok"}


def collect(run_device_calls: bool) -> dict:
    result: dict = {
        "probe": "r08_dispatch_probe",
        "packages": _package_versions(
            ("torch", "torch-npu", "mindiesd", "sgl-kernel-npu", "sglang")
        ),
        "layers": {},
        "minimal_calls": [],
        "h08_isolation_plan": [
            "1. probe native torch_npu fused attention alone (layer 1)",
            "2. probe the AscendFA dense path alone (layer 2, no MindIE import)",
            "3. probe mindiesd sparse_attention alone (layer 3)",
            "The first layer that errors with ERR00100 is the failing op; "
            "record its traceback and package set before any reinstall.",
            "No blind upgrades: compare against the known-good matched stack "
            "only with operator authorization.",
        ],
        "note": (
            "Import availability never proves dispatch. The A0 gates stay: "
            "dense BF16 reference first, conservative sparse policies, and "
            "fail-closed fallback logging per step."
        ),
    }

    # Layer 1: the native operator (what ERR00100 names).
    result["layers"]["native_fused_attention"] = _signature_report(
        "torch_npu", ("npu_fused_infer_attention_score",)
    )
    # Layer 2: the repo's dense fallback executor.
    result["layers"]["ascend_fa_backend"] = _signature_report(
        "sglang.multimodal_gen.runtime.layers.attention.backends.ascend_fa",
        ("AscendFABackend",),
    )
    # Layer 3: the MindIE-SD sparse op.
    result["layers"]["mindiesd_sparse"] = _signature_report(
        "mindiesd", ("sparse_attention",)
    )
    laser = _signature_report(
        "sglang.multimodal_gen.runtime.layers.attention.backends.laser_attn",
        ("LaserAttentionBackend",),
    )
    result["layers"]["laser_dense_fallback"] = laser

    if not run_device_calls:
        result["minimal_calls"] = [
            {"label": "skipped (host without device)", "status": "skipped"}
        ]
        return result

    # Device-only minimal calls; every failure is captured with traceback.
    try:
        import torch

        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise RuntimeError("no NPU device visible")
        device = "npu:0"
        query = torch.randn(1, 4, 128, 128, device=device, dtype=torch.bfloat16)
        key = torch.randn(1, 4, 128, 128, device=device, dtype=torch.bfloat16)
        value = torch.randn(1, 4, 128, 128, device=device, dtype=torch.bfloat16)
        score = torch.npu.npu_fused_infer_attention_score
        result["minimal_calls"].append(
            _minimal_call_report(
                "layer1:npu_fused_infer_attention_score",
                score,
                dict(
                    query=query.transpose(1, 2),
                    key=key.transpose(1, 2),
                    value=value.transpose(1, 2),
                    num_heads=4,
                    input_layout="BNSD",
                    scale=1.0 / 8,
                    inner_precise=0,
                ),
            )
        )
        from mindiesd import sparse_attention

        result["minimal_calls"].append(
            _minimal_call_report(
                "layer3:mindiesd.sparse_attention(bf16)",
                sparse_attention,
                dict(
                    query=query,
                    key=key,
                    value=value,
                    sparse_type="rf_v2",
                    precision="bf16",
                    block_size=128,
                    inner_precise=0,
                    sparsity=0.5,
                    input_layout="BSND",
                    head_num=4,
                    scale=1.0 / 8,
                    is_causal=False,
                ),
            )
        )
    except Exception as error:  # noqa: BLE001
        result["minimal_calls"].append(
            {
                "label": "device setup",
                "status": "error",
                "exception": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    parser.add_argument(
        "--device-calls",
        action="store_true",
        help="run the minimal operator calls (device host only)",
    )
    args = parser.parse_args()
    result = collect(run_device_calls=args.device_calls)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    print("R08 attention dispatch probe")
    print(f"  packages: {json.dumps(result['packages'], sort_keys=True)}")
    for name, layer in sorted(result["layers"].items()):
        print(f"  layer {name}: {layer['status']}")
    for call in result["minimal_calls"]:
        print(f"  call {call['label']}: {call['status']}")
        if call.get("traceback"):
            print(call["traceback"])
    for step in result["h08_isolation_plan"]:
        print(f"  plan: {step}")
    print(f"\n{result['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
