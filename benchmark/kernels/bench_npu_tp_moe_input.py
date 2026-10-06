"""Compare A5 TP normalization/quantization/routing on captured prefill inputs.

Capture x, residual (or None), weight, bias (or None), eps, topk_ids, tp_size
from the FFN boundary of a real GLM-5.2 prefill. Inputs must be BF16 [T,6144],
TP4/8, with routed IDs [T,8]. Do not substitute balanced synthetic routing.
"""

import argparse
import json
import math
import statistics

import torch
import torch_npu  # noqa: F401

from sglang.srt.hardware_backend.npu.moe.init_routing import (
    MXFP8_QUANT_MODE,
    NPUMoEInitRouting_v2,
)
from sglang.srt.hardware_backend.npu.moe.tp_fusion import require_npu_op
from sglang.srt.hardware_backend.npu.triton_kernel.tp_moe_fusion import (
    add_rmsnorm_mxfp8,
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.01)
    args = parser.parse_args()
    if args.warmup < 20 or args.iterations < 100:
        parser.error("Use at least 20 warmups and 100 iterations")
    torch.npu.set_device(args.device)
    data = torch.load(args.inputs, map_location="cpu", weights_only=True)
    if data["tp_size"] not in (4, 8):
        parser.error("Expected a TP4/TP8 capture")
    x, residual, weight, bias, ids = (
        data[name].to(args.device) if data[name] is not None else None
        for name in ("x", "residual", "weight", "bias", "topk_ids")
    )
    if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != 6144 or not x.shape[0]:
        parser.error("Expected nonempty BF16 [T,6144] input")
    if ids.shape != (x.shape[0], 8) or ids.min().item() < 0 or ids.max().item() >= 256:
        parser.error("Expected captured TopK8 routed IDs in [0,256)")
    ids = ids.to(torch.int32)
    eps = float(data["eps"])
    routing = NPUMoEInitRouting_v2(quant_mode=MXFP8_QUANT_MODE)
    print(json.dumps({"torch": torch.__version__, "torch_npu": torch_npu.__version__}))
    for name in (
        "npu_dynamic_mx_quant",
        "npu_moe_init_routing_v2",
    ):
        op = require_npu_op(name)
        print(json.dumps({"operator": name, "schema": str(op.default._schema)}))

    def normalize():
        if residual is None:
            out = torch_npu.npu_rms_norm(x, weight, eps)[0]
            return (out if bias is None else (out + bias).to(x.dtype)), None
        if bias is None:
            out, _, residual_out = torch_npu.npu_add_rms_norm(residual, x, weight, eps)
        else:
            from sgl_kernel_npu.norm.add_rmsnorm_bias import add_rmsnorm_bias

            out, residual_out = add_rmsnorm_bias(x, residual, weight, bias, eps)
        return out.to(x.dtype), residual_out

    def run_case(mode):
        if mode == "dual_norm_routing":
            out, residual_out = add_rmsnorm_mxfp8(x, residual, weight, bias, eps)
            operand = out._npu_mxfp8_operand
        else:
            out, residual_out = normalize()
            operand = None
            if mode == "quant_before_routing":
                operand = torch.ops.npu.npu_dynamic_mx_quant(
                    out, dst_type=torch.float8_e4m3fn
                )
        routed = routing._init_routing(
            out if operand is None else operand[0],
            ids,
            256,
            8,
            input_scale=None if operand is None else operand[1],
        )
        return out, residual_out, routed

    reference = run_case("baseline")
    for mode in ("baseline", "quant_before_routing", "dual_norm_routing"):
        out, residual_out, routed = run_case(mode)
        torch.testing.assert_close(out, reference[0], atol=args.atol, rtol=args.rtol)
        if residual is not None:
            torch.testing.assert_close(residual_out, reference[1], atol=0, rtol=0)
        for index, (actual, expected) in enumerate(zip(routed, reference[2])):
            if index in (0, 3):
                # Exact payload/scales expose quantization-boundary discrepancies.
                torch.testing.assert_close(
                    actual.view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0
                )
            else:
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for _ in range(args.warmup):
            run_case(mode)
        torch.npu.synchronize()
        samples = []
        start, end = (
            torch.npu.Event(enable_timing=True),
            torch.npu.Event(enable_timing=True),
        )
        for _ in range(args.iterations):
            start.record()
            run_case(mode)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        print(
            json.dumps(
                {
                    "case": mode,
                    "tokens": x.shape[0],
                    "tp_size": data["tp_size"],
                    "median_ms": statistics.median(samples),
                    "p95_ms": sorted(samples)[math.ceil(len(samples) * 0.95) - 1],
                    "mean_ms": statistics.mean(samples),
                }
            )
        )


if __name__ == "__main__":
    main()
