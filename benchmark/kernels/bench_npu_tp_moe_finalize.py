"""Replay a GLM TP MoE tail captured from an unchanged serving workload.

Save a torch dictionary containing expert_output (GMM2 BF16 output),
topk_weights (unscaled FP32 TopK output before dispatch), topk_ids,
expanded_row_idx, shared_output, routed_scaling_factor, and tp_size (4 or 8).
Use real prefill routing, H=6144, E=256, TopK=8; no synthetic distribution.

python benchmark/kernels/bench_npu_tp_moe_finalize.py --inputs tail.pt

Serving A/B switches (both default off):
  A: leave both switches unset
  B: SGLANG_NPU_TP_MOE_FUSE_ROUTED_SCALE=1
  C: SGLANG_NPU_TP_MOE_FUSE_SHARED_EXPERT=1 (also enables scaled TopK)
This replay measures the tail with shared output ready. Measure stream waits,
layer wall time, logits/generation, and TTFT separately in the serving A/B.
"""

import argparse
import json
import math
import statistics

import torch
import torch_npu  # noqa: F401

from sglang.srt.hardware_backend.npu.moe.finalize_routing import NPUFinalizeRouting


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument(
        "--gmm2-inputs",
        help="Matching captured GMM1 outputs and checkpoint-layout W2 weights",
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.02)
    args = parser.parse_args()
    if args.warmup < 20 or args.iterations < 100:
        parser.error("Use at least 20 warmups and 100 timed iterations")

    torch.npu.set_device(args.device)
    data = torch.load(args.inputs, map_location="cpu", weights_only=True)
    if data["tp_size"] not in (4, 8):
        parser.error("Capture TP4 or TP8 production inputs")
    tensors = {
        name: data[name].to(args.device)
        for name in (
            "expert_output",
            "topk_weights",
            "topk_ids",
            "expanded_row_idx",
            "shared_output",
        )
    }
    expert = tensors["expert_output"]
    weights = tensors["topk_weights"]
    ids = tensors["topk_ids"].to(torch.int32)
    shared = tensors["shared_output"]
    if (
        expert.ndim != 2
        or shared.ndim != 2
        or shared.shape[0] == 0
        or expert.shape[1] != 6144
        or shared.shape[1] != 6144
        or weights.shape != (shared.shape[0], 8)
        or ids.shape != weights.shape
        or expert.shape[0] != shared.shape[0] * 8
        or weights.dtype != torch.float32
        or expert.dtype != torch.bfloat16
        or shared.dtype != expert.dtype
        or ids.min().item() < 0
        or ids.max().item() >= 256
    ):
        parser.error("Expected BF16 H=6144, FP32 TopK=8 weights, routed IDs in [0,256)")

    factor = float(data["routed_scaling_factor"])
    unscaled_weights = weights
    # Production folds this multiplication into the TopK operator.
    scaled_weights = weights * factor
    finalize = NPUFinalizeRouting(drop_pad_mode=2)

    def routed(route_weights, skip1=None):
        return finalize._finalize_routing(
            expert, route_weights, tensors["expanded_row_idx"], ids, skip1=skip1
        )

    cases = {
        "baseline_bf16_weights": lambda: (
            routed(weights.to(expert.dtype)).mul_(factor).add_(shared)
        ),
        "baseline": lambda: routed(unscaled_weights).mul_(factor).add_(shared),
        "scaled_topk": lambda: routed(scaled_weights).add_(shared),
        "scaled_topk_shared": lambda: routed(scaled_weights, skip1=shared),
    }
    if args.gmm2_inputs:
        from sglang.srt.hardware_backend.npu.moe.tp_fusion import fused_gmm2_finalize
        from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
            prepare_w4a8_mxfp_weight,
            w4a8_mxfp_gmm,
        )

        captured = torch.load(args.gmm2_inputs, map_location="cpu", weights_only=True)
        mid, mid_scale, w2, w2_scale, counts = (
            captured[key].to(args.device)
            for key in (
                "gmm1_output",
                "gmm1_scale",
                "w2_weight",
                "w2_scale",
                "expert_tokens",
            )
        )
        local_i = 2048 // data["tp_size"]
        if (
            mid.shape != (expert.shape[0], local_i)
            or mid.dtype != torch.float8_e4m3fn
            or w2.shape != (256, 6144, local_i // 2)
            or w2.dtype != torch.uint8
            or w2_scale.shape != (256, 6144, local_i // 32)
            or counts.shape != (256,)
            or counts.sum().item() != expert.shape[0]
        ):
            parser.error(
                "Expected captured TP-local MXFP8 GMM1 outputs and packed checkpoint-layout W2"
            )
        w2, w2_scale = prepare_w4a8_mxfp_weight(w2, w2_scale)
        # The stock finalizer capture uses row_idx_type=0; the fused op needs its inverse.
        sorted_rows = tensors["expanded_row_idx"].argsort().to(torch.int32)
        counts = counts.to(torch.int64)

        def gmm2():
            return w4a8_mxfp_gmm(
                input=mid,
                input_scale=mid_scale,
                weight=w2,
                weight_scale=w2_scale,
                group_list_type=1,
                group_list=counts,
                output_dtype=torch.bfloat16,
            )

        torch.testing.assert_close(gmm2(), expert, atol=args.atol, rtol=args.rtol)

        def unfused_tail():
            return finalize._finalize_routing(
                gmm2(), scaled_weights, tensors["expanded_row_idx"], ids, skip1=shared
            )

        cases["gmm2_finalize_shared_baseline"] = unfused_tail
        cases["gmm2_finalize_shared_fused"] = lambda: fused_gmm2_finalize(
            mid, mid_scale, w2, w2_scale, counts, scaled_weights, sorted_rows, shared
        )
    reference = cases["baseline"]()
    for name, run in cases.items():
        output = run()
        torch.testing.assert_close(output, reference, atol=args.atol, rtol=args.rtol)
        error = (output.float() - reference.float()).abs()
        print(json.dumps({"case": name, "max_abs_error": error.max().item()}))

    for name, run in cases.items():
        for _ in range(args.warmup):
            run()
        torch.npu.synchronize()
        samples = []
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        for _ in range(args.iterations):
            start.record()
            run()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        print(
            json.dumps(
                {
                    "case": name,
                    "tp_size": data["tp_size"],
                    "tokens": shared.shape[0],
                    "median_ms": statistics.median(samples),
                    "p95_ms": sorted(samples)[math.ceil(len(samples) * 0.95) - 1],
                    "mean_ms": statistics.mean(samples),
                }
            )
        )


if __name__ == "__main__":
    main()
