# SPDX-License-Identifier: Apache-2.0
"""Capture environment and public signatures; never infer device correctness."""

import argparse
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def collect():
    result = dict(
        python=sys.version,
        host=platform.platform(),
        packages={},
        operators={},
        capabilities_status="unverified",
        environment={
            key: value
            for key, value in os.environ.items()
            if key.startswith(("ASCEND_", "CANN_", "HCCL_", "SGLANG_"))
            and not any(
                word in key.upper() for word in ("TOKEN", "KEY", "SECRET", "PASSWORD")
            )
        },
    )
    for name in ("torch", "torch-npu", "mindiesd", "sgl-kernel-npu", "sglang"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    for executable, arguments in (("npu-smi", ["info"]), ("ffprobe", ["-version"])):
        if shutil.which(executable):
            completed = subprocess.run(
                [executable, *arguments], text=True, capture_output=True, timeout=30
            )
            result[executable] = dict(
                returncode=completed.returncode,
                output=(completed.stdout + completed.stderr)[:20000],
            )
        else:
            result[executable] = dict(status="unavailable")
    for module, names in (
        ("mindiesd", ("quant_attention", "sparse_attention")),
        ("torch_npu", ("npu_fused_infer_attention_score",)),
    ):
        try:
            loaded = importlib.import_module(module)
            for name in names:
                function = getattr(loaded, name, None)
                try:
                    signature = str(inspect.signature(function)) if function else None
                except (TypeError, ValueError):
                    signature = "not introspectable"
                result["operators"][f"{module}.{name}"] = dict(
                    present=function is not None, signature=signature
                )
        except Exception as error:
            result["operators"][module] = dict(error=str(error))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.output.open("x") as output:
        output.write(json.dumps(collect(), indent=2) + "\n")
