"""Replay the shared merge on a production capture, without BF16 route weights.

Capture dictionary: expert_output (BF16 GMM2), topk_weights (FP32, with routed
scale already applied exactly once), topk_ids, expanded_row_idx, shared_output
(BF16), tp_size (4/8), routing_weights_scaled=True. H=6144, E=256, K=8.
"""

import argparse
import json

import torch
import torch_npu
from npu_tp_bench_utils import timing

from sglang.srt.hardware_backend.npu.moe.finalize_routing import NPUFinalizeRouting


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.02)
    args = parser.parse_args()
    if args.warmup < 20 or args.iterations < 100:
        parser.error("Use at least 20 warmups and 100 iterations")
    torch.npu.set_device(args.device)
    data = torch.load(args.inputs, map_location="cpu", weights_only=True)
    if data["tp_size"] not in (4, 8) or data.get("routing_weights_scaled") is not True:
        parser.error("Use TP4/8 inputs with FP32 routing weights scaled exactly once")
    expert, weights, ids, rows, shared = (
        data[key].to(args.device)
        for key in (
            "expert_output",
            "topk_weights",
            "topk_ids",
            "expanded_row_idx",
            "shared_output",
        )
    )
    tokens = shared.shape[0]
    if (
        shared.shape != (tokens, 6144)
        or tokens == 0
        or expert.shape != (tokens * 8, 6144)
        or weights.shape != (tokens, 8)
        or ids.shape != weights.shape
        or expert.dtype != torch.bfloat16
        or shared.dtype != torch.bfloat16
        or weights.dtype != torch.float32
        or ids.dtype not in (torch.int32, torch.int64)
        or rows.dtype not in (torch.int32, torch.int64)
        or ids.min().item() < 0
        or ids.max().item() >= 256
        or rows.numel() != tokens * 8
    ):
        parser.error(
            "Expected production BF16 H=6144, FP32 K=8 routing and row mapping"
        )
    ids = ids.to(torch.int32)
    finalize = NPUFinalizeRouting(drop_pad_mode=2)

    def routed(skip=None):
        return finalize._finalize_routing(expert, weights, rows, ids, skip1=skip)

    routed_output = routed()
    reference = routed_output + shared
    fused = routed(shared)
    torch.testing.assert_close(fused, reference, atol=args.atol, rtol=args.rtol)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "max_abs_error": (fused.float() - reference.float()).abs().max().item(),
                "shared_stride": shared.stride(),
                "shared_contiguous": shared.is_contiguous(),
            }
        )
    )
    cases = {
        "finalize_no_skip": routed,
        "add": lambda: routed_output + shared,
        "finalize_plus_add": lambda: routed() + shared,
        "finalize_skip1": lambda: routed(shared),
    }
    for name, run in cases.items():
        print(
            json.dumps(
                {
                    "case": name,
                    "tokens": tokens,
                    "tp_size": data["tp_size"],
                    **timing(run, args.warmup, args.iterations),
                }
            )
        )


if __name__ == "__main__":
    main()
