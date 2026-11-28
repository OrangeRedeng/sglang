# SPDX-License-Identifier: Apache-2.0
"""Resolve R02 cases to existing component quantization CLI flags."""

import argparse
import json
import math
import os
import re
from pathlib import Path

PROTECTED = ("token_refiner", "final_layer", "condition_proj", "adaln_proj")
METHODS = {
    "Q0": "bf16",
    "Q1": "w8a8_int",
    "Q2": "w8a8_int",
    "Q3": "w8a8_int",
    "Q4": "w8a8_int",
    "Q5": "mxfp8",
    "Q5_FFN": "mxfp8",
    "Q5_ALLOWLIST": "mxfp8",
}


def selection(case, checkpoint, placement, source, reference=None, allowlist=None):
    method = METHODS[case]
    if case == "Q0":
        return dict(method=method, selected=[], exclusions=[])
    if case in ("Q1", "Q5"):
        return dict(method=method, selected=None, exclusions=list(PROTECTED))
    if reference is None:
        raise ValueError(f"{case} requires the matching Q0 rank-0 precision map")
    if (
        reference["method"] != "bf16"
        or reference["rank"] != 0
        or reference["checkpoint_manifest_sha256"] != checkpoint
        or reference["placement"] != placement
        or reference["source_sha"] != source
    ):
        raise ValueError("Q0 source, checkpoint or placement differs from this launch")
    layers = reference["layers"]
    if any(
        row["precision"] not in ("float32", "float16", "bfloat16")
        for row in layers.values()
    ):
        raise ValueError("Q0 precision map must contain only floating-point linears")
    if any(
        not re.fullmatch(r"[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)*", layer) for layer in layers
    ):
        raise ValueError("Invalid Q0 module path")
    block_linears = {
        prefix: (int(match[1]), match[2])
        for prefix, row in layers.items()
        if row["precision"] == "bfloat16"
        and (
            match := re.fullmatch(
                r"blocks\.(\d+)\.(mlp\.fc[12]|attn\.(?:qkv_proj|out_proj))", prefix
            )
        )
    }
    blocks = sorted({block for block, _ in block_linears.values()})
    if blocks != list(range(len(blocks))) or len(blocks) < 3:
        raise ValueError("Q0 map must enumerate the complete numbered DiT block stack")
    projections = {"mlp.fc1", "mlp.fc2", "attn.qkv_proj", "attn.out_proj"}
    if any(
        {projection for block, projection in block_linears.values() if block == i}
        != projections
        for i in blocks
    ):
        raise ValueError("Incomplete Q0 block projection inventory")
    edge = math.ceil(len(blocks) / 10)
    middle = {
        prefix: projection
        for prefix, (block, projection) in block_linears.items()
        if edge <= block < len(blocks) - edge
    }
    if case in ("Q3", "Q5_ALLOWLIST"):
        if (
            not isinstance(allowlist, list)
            or not allowlist
            or not all(isinstance(path, str) and path in middle for path in allowlist)
            or len(set(allowlist)) != len(allowlist)
        ):
            raise ValueError(
                "Q3 allowlist must contain distinct whole middle-block projections from Q0"
            )
        included = set(allowlist)
    else:
        group = "mlp." if case in ("Q2", "Q5_FFN") else "attn."
        included = {
            name for name, projection in middle.items() if projection.startswith(group)
        }
    if not included:
        raise ValueError(f"No supported linears found for {case}")
    return dict(
        method=method,
        selected=sorted(included),
        exclusions=sorted((layers.keys() - included) | set(PROTECTED)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=METHODS)
    parser.add_argument("checkpoint")
    parser.add_argument("placement")
    parser.add_argument("source")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        reference = os.getenv("H3_QUANT_REFERENCE")
        allowlist = os.getenv("H3_QUANT_ALLOWLIST")
        resolved = selection(
            args.case,
            args.checkpoint,
            args.placement,
            args.source,
            json.loads(Path(reference).read_text()) if reference else None,
            json.loads(Path(allowlist).read_text()) if allowlist else None,
        )
        resolved["case"] = args.case
        args.output.write_text(json.dumps(resolved, indent=2) + "\n")
        print(resolved["method"])
        print("\n".join(resolved["exclusions"]))
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"R02 quantization: {error}\n")


if __name__ == "__main__":
    main()
