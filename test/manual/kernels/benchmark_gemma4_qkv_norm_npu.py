"""Validate opt-in serial and tiled QKV norm through the serving op boundary."""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import torch
import torch_npu
from sglang.srt.hardware_backend.npu.gemma4_ops import npu_gemma4_qkv_rmsnorm
from validate_gemma4_qkv_norm_npu import validate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gemma4.candidate_validation import new_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--candidate",
        choices=["all", "serial", "tile4", "tile8", "tile16"],
        default="all",
    )
    p.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 12, 14, 15, 24, 128, 1235, 4096, 8192],
    )
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--output", type=Path, default=Path("gemma4-qkv-validation.json"))
    args = p.parse_args()
    if args.samples < 1 or any(n < 1 for n in args.tokens):
        p.error("Sample count and token counts must be positive")
    if args.output.exists():
        p.error("Use a new output file")
    names = (
        ["serial", "tile4", "tile8", "tile16"]
        if args.candidate == "all"
        else [args.candidate]
    )
    report = new_report("qkv", torch, torch_npu)
    report["candidate_results"] = {}
    torch.manual_seed(0)
    try:
        for name in names:
            print(f"BEGIN candidate={name}", flush=True)
            try:
                kwargs = (
                    {}
                    if name == "serial"
                    else {
                        "norm": partial(
                            npu_gemma4_qkv_rmsnorm, rows_per_program=int(name[4:])
                        )
                    }
                )
                for dtype in (torch.float16, torch.bfloat16):
                    for head_dim in (128, 256, 512):
                        for tokens in args.tokens:
                            for shared in (False, True):
                                report["current_case"] = {
                                    "candidate": name,
                                    "tokens": tokens,
                                    "head_dim": head_dim,
                                    "dtype": str(dtype),
                                    "shared_kv": shared,
                                }
                                timings = validate(
                                    tokens,
                                    head_dim,
                                    dtype,
                                    shared,
                                    samples=args.samples,
                                    **kwargs,
                                )
                                report["cases"].append(
                                    {
                                        "candidate": name,
                                        "dtype": str(dtype),
                                        "head_dim": head_dim,
                                        "tokens": tokens,
                                        "shared_kv": shared,
                                        "eager_correct": True,
                                        "fullgraph_trace_correct": True,
                                        "graph_replay_modes": [0, 1],
                                        "timing": timings,
                                    }
                                )
                report["candidate_results"][name] = {"complete": True}
            except (RuntimeError, AssertionError, ValueError) as error:
                report["candidate_results"][name] = {
                    "complete": False,
                    "error": str(error),
                    "case": report.get("current_case"),
                }
                print(f"Rejected {name}: {error}", flush=True)
        report["complete"] = True
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    print(
        f"Validation attempts completed: {args.output}; serving selection still requires identical-workload measurements"
    )
    return (
        0 if any(row["complete"] for row in report["candidate_results"].values()) else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
