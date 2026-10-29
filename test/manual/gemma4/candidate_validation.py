"""Bind kernel correctness evidence to its exact sources and NPU runtime."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

SOURCES = (
    "python/sglang/kernels/ops/layernorm/gemma4_fused_ops.py",
    "python/sglang/srt/hardware_backend/npu/gemma4_ops.py",
    "python/sglang/srt/models/gemma4_causal.py",
    "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py",
    "python/sglang/srt/hardware_backend/npu/memory_pool_npu.py",
    "python/sglang/srt/mem_cache/swa_memory_pool.py",
    "python/sglang/srt/layers/rotary_embedding/base.py",
    "python/sglang/srt/layers/rotary_embedding/rope_variant.py",
    "test/manual/kernels/validate_gemma4_qkv_norm_npu.py",
    "test/manual/kernels/benchmark_gemma4_qkv_norm_npu.py",
    "test/manual/gemma4/candidate_validation.py",
)
QKV_TOKENS = (1, 12, 14, 15, 24, 128, 1235, 4096, 8192)


def compile_case(fn):
    """Each diagnostic case owns its compile cache, outside measured timing."""
    import torch

    torch.compiler.reset()
    return torch.compile(fn, backend="eager", fullgraph=True)


def source_hashes(root):
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES
    }


def triton_version():
    for package in ("triton-ascend", "triton"):
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return None


def cann_versions():
    roots = [
        Path("/usr/local/Ascend/cann"),
        Path("/usr/local/Ascend/ascend-toolkit/latest"),
    ]
    for name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        if os.environ.get(name):
            roots.append(Path(os.environ[name]))
    candidates = [
        root / filename
        for root in roots
        for filename in (
            "version.cfg",
            "version.info",
            "ascend_toolkit_install.info",
            "x86_64-linux/ascend_toolkit_install.info",
            "aarch64-linux/ascend_toolkit_install.info",
        )
    ]
    return {str(path): path.read_text() for path in candidates if path.is_file()}


def new_report(kind, torch, torch_npu):
    return {
        "kind": kind,
        "complete": False,
        "serving_selected": False,
        "source_sha256": source_hashes(Path(__file__).resolve().parents[3]),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "sgl_kernel_npu": importlib.metadata.version("sgl_kernel_npu"),
        "triton": triton_version(),
        "cann_version_files": cann_versions(),
        "device_name": torch.npu.get_device_name(),
        "cases": [],
    }


def validate_report(
    path, kind, candidate, root, versions, device_name, *, cann_versions=None
):
    if kind != "qkv":
        raise ValueError("Only QKV candidates are supported")
    report = json.loads(Path(path).read_text())
    if cann_versions is not None and not cann_versions:
        raise ValueError(
            "CANN version capture is empty; set ASCEND_HOME_PATH to the active toolkit"
        )
    if (
        report.get("kind") != kind
        or report.get("complete") is not True
        or report.get("source_sha256") != source_hashes(root)
        or any(report.get(name) != versions[name] for name in ("torch", "torch_npu"))
        or report.get("device_name") != device_name
        or any(
            report.get(name) != versions[name]
            for name in ("sgl_kernel_npu", "triton")
            if versions.get(name) is not None
        )
        or (
            cann_versions is not None
            and report.get("cann_version_files") != cann_versions
        )
    ):
        raise ValueError(
            "Kernel validation is incomplete, stale, or from a different runtime/NPU"
        )
    if (
        kind == "qkv"
        and report.get("candidate_results", {}).get(candidate, {}).get("complete")
        is False
    ):
        raise ValueError("Requested QKV candidate failed validation")
    required = set()
    for dtype in ("torch.float16", "torch.bfloat16"):
        for dim in (128, 256, 512):
            for tokens in QKV_TOKENS:
                for variant in (False, True):
                    required.add((dtype, dim, tokens, variant, None))
    observed = set()
    for row in report.get("cases", []):
        if row.get("candidate") != candidate:
            continue
        if (
            row.get("eager_correct") is not True
            or row.get("fullgraph_trace_correct") is not True
            or row.get("graph_replay_modes") != [0, 1]
        ):
            raise ValueError("Kernel candidate failed correctness or graph replay")
        observed.add(
            (
                row["dtype"],
                row["head_dim"],
                row["tokens"],
                row["shared_kv"],
                None,
            )
        )
    if not required.issubset(observed):
        raise ValueError(
            f"Missing {len(required - observed)} required {candidate} correctness cases"
        )
    return report
