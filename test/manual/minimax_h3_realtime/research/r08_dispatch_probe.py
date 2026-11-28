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
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
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

            entry["signature"] = str(inspect.signature(function)) if function else None
        except (TypeError, ValueError):
            entry["signature"] = "not introspectable"
        report["operators"][name] = entry
    return report


CASES = (
    "native_dense",
    "laser_dense",
    "mindiesd_import",
    "quant_dense",
    "rfv2",
    "rfv3",
)


def collect(
    run_device_calls: bool,
    *,
    tokens=2048,
    heads=28,
    head_dim=128,
    config=None,
    case=None,
) -> dict:
    result: dict = {
        "probe": "r08_dispatch_probe",
        "experiment_identity": {
            "model": "MiniMax-H3-FL2VA",
            "shape": {
                "batch": 1,
                "tokens": tokens,
                "heads": heads,
                "head_dim": head_dim,
            },
            "dtype": "bfloat16",
            "layouts": {"native_dense": "BNSD", "mindiesd_sparse": "BSND"},
        },
        "packages": _package_versions(
            ("torch", "torch-npu", "mindiesd", "sgl-kernel-npu", "sglang")
        ),
        "layers": {},
        "minimal_calls": [],
        "capability_receipts": {},
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
    result["layers"]["mindiesd_quant"] = _signature_report(
        "mindiesd", ("quant_attention",)
    )
    laser = _signature_report(
        "sglang.multimodal_gen.runtime.layers.attention.backends.laser_attn",
        ("LaserAttentionBackend",),
    )
    result["layers"]["laser_dense_fallback"] = laser

    root = Path(__file__).resolve().parents[4]
    policy = (
        config
        if config is not None
        else {"video_spans": [{"start": 0, "latent_shape": [1, 1, tokens]}]}
    )
    identity = result["experiment_identity"]
    identity.update(
        source_sha=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        config_sha256=hashlib.sha256(
            json.dumps(policy, sort_keys=True).encode()
        ).hexdigest(),
        geometry="operator input"
        if config is not None
        else "synthetic single video span",
    )
    selected = (case,) if case else CASES
    receipts = result["capability_receipts"]
    for name in selected:
        receipts[name] = {"status": "device_pending", "experiment_identity": identity}
    if not run_device_calls:
        result["minimal_calls"] = [
            {"label": "skipped (host without device)", "status": "skipped"}
        ]
        return result

    def probe(name):
        import torch
        import torch_npu

        if name == "mindiesd_import":
            module = importlib.import_module("mindiesd")
            extensions = [
                str(path.relative_to(Path(module.__file__).parent))
                for path in Path(module.__file__).parent.rglob("*.so")
            ]
            return {
                "package": module.__file__,
                "extension_files": extensions,
                "registered_ops": sorted(
                    op
                    for op in torch._C._dispatch_get_all_op_names()
                    if "mindie" in op.lower()
                ),
            }
        if not torch.npu.is_available():
            raise RuntimeError("no NPU device visible")
        shape = (1, tokens, heads, head_dim)
        q, k, v = (
            torch.randn(shape, device="npu", dtype=torch.bfloat16) for _ in range(3)
        )
        scale = head_dim**-0.5
        kwargs = dict(
            num_heads=heads, head_size=head_dim, causal=False, softmax_scale=scale
        )
        if name == "native_dense":
            q, k, v = (tensor.transpose(1, 2).contiguous() for tensor in (q, k, v))
            output = torch_npu.npu_fused_infer_attention_score(
                q, k, v, num_heads=heads, input_layout="BNSD", scale=scale
            )[0]
            path = "torch_npu.npu_fused_infer_attention_score"
        elif name == "laser_dense":
            from sglang.multimodal_gen.runtime.layers.attention.backends.laser_attn import (
                LaserAttentionImpl,
            )

            impl = LaserAttentionImpl(**kwargs)
            output = impl.forward(q, k, v, None, preserve_bf16_range=True)
            path = (
                "LaserAttentionImpl.SDPA"
                if tokens < impl.min_seqlen
                else "LaserAttentionImpl._laser_attention"
            )
        elif name == "quant_dense":
            from sglang.multimodal_gen.runtime.layers.attention.backends.ascend_fa import (
                AscendFAImpl,
                AscendFAMetadata,
            )

            metadata = AscendFAMetadata(
                **policy.get("dense_kwargs", {"precision": "fp8"})
            )
            if metadata.precision in (None, "bf16"):
                raise ValueError("quant_dense requires an explicit quantized precision")
            output = AscendFAImpl(**kwargs).forward(q, k, v, metadata)
            path = f"AscendFAImpl.{metadata.precision}"
        else:
            from sglang.multimodal_gen.runtime.layers.attention.backends.ascend_sparse_attn import (
                AscendSparseAttentionImpl,
                AscendSparseAttentionMetadataBuilder,
            )

            options = dict(
                policy,
                sparse_type="rf_v2" if name == "rfv2" else "rf_v3",
                skip_first_steps=0,
                skip_last_steps=0,
            )
            metadata = AscendSparseAttentionMetadataBuilder().build(
                current_timestep=0, **options
            )
            output = AscendSparseAttentionImpl(**kwargs).forward(q, k, v, metadata)
            path = f"mindiesd.sparse_attention.{metadata.sparse_type}"
        torch.npu.synchronize()
        if output.shape != q.shape or not torch.isfinite(output).all().item():
            raise ValueError(f"Invalid attention output from {path}")
        return {
            "path": path,
            "input_shape": list(q.shape),
            "strides": list(q.stride()),
            "policy": policy,
        }

    for name in selected:
        # A single-case invocation runs in its own process for device fault isolation.
        try:
            details = probe(name)
            receipts[name].update(
                status="imported" if name == "mindiesd_import" else "ok", **details
            )
        except Exception as error:
            receipts[name].update(
                status="error",
                exception=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            )
        result["minimal_calls"].append(dict(label=name, **receipts[name]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    parser.add_argument(
        "--device-calls",
        action="store_true",
        help="run the minimal operator calls (device host only)",
    )
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=28)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON sparse attention options including captured video_spans or latent grid",
    )
    args = parser.parse_args()
    if args.device_calls and not args.case:
        parser.error("select one --case per device invocation for fault isolation")
    if min(args.tokens, args.heads, args.head_dim) < 1:
        parser.error("shape dimensions must be positive")
    result = collect(
        args.device_calls,
        tokens=args.tokens,
        heads=args.heads,
        head_dim=args.head_dim,
        config=json.loads(args.config.read_text()) if args.config else None,
        case=args.case,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return int(
            any(r["status"] == "error" for r in result["capability_receipts"].values())
        )
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
    return int(
        any(r["status"] == "error" for r in result["capability_receipts"].values())
    )


if __name__ == "__main__":
    sys.exit(main())
