"""Validate serial and tiled QKV norm candidates; no serving kernel is selected."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch_npu
from sglang.kernels.ops.layernorm.gemma4_fused_ops import _gemma_qkv_rmsnorm_npu_tiled
from sglang.srt.utils.custom_op import register_custom_op
from validate_gemma4_qkv_norm_npu import validate


@register_custom_op(op_name="gemma4_qkv_norm_validation", mutates_args=["q", "k", "v"])
def candidate(
    q: torch.Tensor,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor | None,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    rows: int,
) -> None:
    _gemma_qkv_rmsnorm_npu_tiled(
        q,
        k,
        v,
        q_weight,
        k_weight,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rows_per_program=rows,
    )


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
    report = {
        "complete": False,
        "serving_selected": False,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "cases": [],
    }
    torch.manual_seed(0)
    try:
        for name in names:
            kwargs = (
                {}
                if name == "serial"
                else {"norm": partial(candidate, rows=int(name[4:]))}
            )
            for dtype in (torch.float16, torch.bfloat16):
                for head_dim in (128, 256, 512):
                    for tokens in args.tokens:
                        for shared in (False, True):
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
        report["complete"] = True
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    print(
        f"Validated candidates: {args.output}; serving selection still requires identical-workload measurements"
    )


if __name__ == "__main__":
    main()
