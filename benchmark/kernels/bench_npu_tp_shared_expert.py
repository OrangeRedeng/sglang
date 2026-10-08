"""Replay real A5 shared W4A8-MXFP weights and normalized activations.

--inputs dictionary: tp_size (4/8), x (BF16 [T,6144]), quantized_x (E4M3),
x_scale (E8M0 bytes), gate_up_weight, gate_up_scale, down_weight, down_scale.
Weights/scales must be TP-local logical checkpoint tensors: packed uint8
[out,in/2] and uint8 [out,in/32], before NZ conversion. Gate is the first half.

Optional --routed-inputs dictionary: x (identical normalized BF16 input),
tp_size, topk_ids [T,8], topk_weights (FP32, already scaled once),
routing_weights_scaled=True, w13_weight/scale, w2_weight/scale (E=256,
logical checkpoint layout). Replays dispatch + fused GMM1 + GMM2 + finalize
with serial or concurrent shared compute. Excludes router and TP reduction;
serving profiles are still required to measure full layer wall time and TTFT.
"""

import argparse
import json
from types import SimpleNamespace

import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu
from npu_tp_bench_utils import timing

from sglang.srt.hardware_backend.npu.moe.tp_fusion import (
    mxfp8_input,
    shared_gateup_quant,
    shared_gmm1,
)
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUMXFP4W4A8OfflineLinearMethod,
)


class CapturedLinear(torch.nn.Module):
    def __init__(self, weight, scale, device):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.to(device), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(scale.to(device), requires_grad=False)
        self.bias = None
        self.quant_method = NPUMXFP4W4A8OfflineLinearMethod()
        self.quant_method.process_weights_after_loading(self)

    def forward(self, operand):
        return self.quant_method.apply(self, operand), None


def routed_replay(data, tp_size, device):
    from sglang.srt.hardware_backend.npu.moe.finalize_routing import NPUFinalizeRouting
    from sglang.srt.hardware_backend.npu.moe.init_routing import (
        MXFP8_QUANT_MODE,
        NPUMoEInitRouting_v2,
    )
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        NPUW4A8MXFP4MoEMethod,
        prepare_w4a8_mxfp_weight,
    )

    local_i = 2048 // tp_size
    info = SimpleNamespace()
    for prefix, out_dim, in_dim in (("w13", 2 * local_i, 6144), ("w2", 6144, local_i)):
        weight, scale = data[f"{prefix}_weight"], data[f"{prefix}_scale"]
        if (
            weight.shape != (256, out_dim, in_dim // 2)
            or scale.shape != (256, out_dim, in_dim // 32)
            or weight.dtype != torch.uint8
            or scale.dtype != torch.uint8
        ):
            raise ValueError(f"Expected logical checkpoint-layout TP-local {prefix}")
        weight, scale = prepare_w4a8_mxfp_weight(weight.to(device), scale.to(device))
        setattr(info, f"{prefix}_weight", weight)
        setattr(info, f"{prefix}_weight_scale", scale)
    if data["topk_ids"].dtype not in (torch.int32, torch.int64):
        raise ValueError("Captured TopK IDs must be integers")
    ids = data["topk_ids"].to(device=device, dtype=torch.int32)
    weights = data["topk_weights"].to(device)
    if (
        data["tp_size"] != tp_size
        or data.get("routing_weights_scaled") is not True
        or weights.dtype != torch.float32
        or ids.shape != weights.shape
        or weights.shape != (data["x"].shape[0], 8)
        or ids.min().item() < 0
        or ids.max().item() >= 256
    ):
        raise ValueError("Expected captured K=8 IDs and scaled FP32 route weights")
    init = NPUMoEInitRouting_v2(quant_mode=MXFP8_QUANT_MODE)
    finalize = NPUFinalizeRouting(drop_pad_mode=2)
    method = NPUW4A8MXFP4MoEMethod()

    def run(operand, tokens, shared, ready=None):
        qx, scale = operand
        routed, rows, counts, routed_scale = init._init_routing(
            qx, ids[:tokens], 256, 8, input_scale=scale
        )
        mid, mid_scale = method.apply_fused_gmm1_swiglu(
            info, routed, counts, routed_scale, group_list_type=1
        )
        output = method.apply(
            info, mid, counts, mid_scale, torch.bfloat16, "w2", group_list_type=1
        )
        if ready is not None:
            torch.npu.current_stream().wait_event(ready)
            shared.record_stream(torch.npu.current_stream())
        return finalize._finalize_routing(
            output, weights[:tokens], rows, ids[:tokens], skip1=shared
        )

    return run


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--routed-inputs")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--tokens",
        nargs="+",
        type=int,
        default=[1, 16, 64, 256, 512, 1024, 4096, 8192, 16384],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["baseline", "grouped_fused"],
        default=["baseline", "grouped_fused"],
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.02)
    args = parser.parse_args()
    if args.warmup < 20 or args.iterations < 100:
        parser.error("Use at least 20 warmups and 100 iterations")
    torch.npu.set_device(args.device)
    data = torch.load(args.inputs, map_location="cpu", weights_only=True)
    tp_size = data["tp_size"]
    if tp_size not in (4, 8):
        parser.error("Expected TP4 or TP8 capture")
    local_i = 2048 // tp_size
    x = data["x"].to(args.device)
    qx, scale = data["quantized_x"].to(args.device), data["x_scale"].to(args.device)
    if (
        x.ndim != 2
        or x.shape[1] != 6144
        or x.dtype != torch.bfloat16
        or qx.shape != x.shape
        or qx.dtype != torch.float8_e4m3fn
        or scale.numel() != x.shape[0] * 6144 // 32
        or scale.dtype not in (torch.uint8, getattr(torch, "float8_e8m0fnu", None))
        or any(t <= 0 or t > x.shape[0] for t in args.tokens)
    ):
        parser.error(
            "Capture real BF16/E4M3 [T,6144] input for all requested token counts"
        )
    for prefix, out_dim, in_dim in (
        ("gate_up", 2 * local_i, 6144),
        ("down", 6144, local_i),
    ):
        weight, weight_scale = data[f"{prefix}_weight"], data[f"{prefix}_scale"]
        if (
            weight.shape != (out_dim, in_dim // 2)
            or weight_scale.shape != (out_dim, in_dim // 32)
            or weight.dtype != torch.uint8
            or weight_scale.dtype != torch.uint8
        ):
            parser.error(f"Expected TP-local packed checkpoint weights for {prefix}")
    # A no-flags dense reference must receive the same quantized normalized value.
    expected_qx, expected_scale = mxfp8_input(x)
    torch.testing.assert_close(
        qx.view(torch.uint8), expected_qx.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(
        scale.view(torch.uint8).reshape(-1),
        expected_scale.view(torch.uint8).reshape(-1),
        atol=0,
        rtol=0,
    )
    scale = scale.reshape(x.shape[0], 6144 // 64, 2)
    mlp = SimpleNamespace(
        gate_up_proj=CapturedLinear(
            data["gate_up_weight"], data["gate_up_scale"], args.device
        ),
        down_proj=CapturedLinear(data["down_weight"], data["down_scale"], args.device),
        act_fn=torch.ops.npu.npu_swiglu,
    )
    routed = None
    if args.routed_inputs:
        routed_data = torch.load(
            args.routed_inputs, map_location="cpu", weights_only=True
        )
        torch.testing.assert_close(data["x"], routed_data["x"], atol=0, rtol=0)
        routed = routed_replay(routed_data, tp_size, args.device)
    shared_stream = torch.npu.Stream() if routed is not None else None
    print(json.dumps({"torch": torch.__version__, "torch_npu": torch_npu.__version__}))

    for tokens in args.tokens:
        hidden = x[:tokens]
        operand = (qx[:tokens], scale[:tokens])

        def baseline():
            return mlp.down_proj(mlp.act_fn(mlp.gate_up_proj(hidden)[0]))[0]

        reference = baseline()
        for mode in args.modes:

            def run():
                if mode == "baseline":
                    return baseline()
                return shared_gmm1(mlp, hidden, operand, mode=mode)

            output = run()
            torch.testing.assert_close(
                output, reference, atol=args.atol, rtol=args.rtol
            )
            print(
                json.dumps(
                    {
                        "case": mode,
                        "boundary": "shared_total",
                        "tokens": tokens,
                        "tp_size": tp_size,
                        "max_abs_error": (output.float() - reference.float())
                        .abs()
                        .max()
                        .item(),
                        **timing(run, args.warmup, args.iterations),
                    }
                )
            )
            if mode == "grouped_fused":

                def gateup():
                    return shared_gateup_quant(mlp, hidden, operand)

                mid = gateup()
                stages = {
                    "gateup_activation_quant_fused": gateup,
                    "down": lambda: mlp.down_proj(mid)[0],
                }
            else:

                def gateup():
                    return mlp.gate_up_proj(hidden)[0]

                gate_out = gateup()

                def activation():
                    return mxfp8_input(mlp.act_fn(gate_out))

                mid = activation()
                stages = {
                    "gateup": gateup,
                    "activation_quant": activation,
                    "down": lambda: mlp.down_proj(mid)[0],
                }
            for stage, stage_run in stages.items():
                print(
                    json.dumps(
                        {
                            "case": mode,
                            "boundary": stage,
                            "tokens": tokens,
                            **timing(stage_run, args.warmup, args.iterations),
                        }
                    )
                )

            if routed is not None:

                def serial():
                    return routed(operand, tokens, run())

                def overlap():
                    main_stream = torch.npu.current_stream()
                    input_ready = main_stream.record_event()
                    with torch.npu.stream(shared_stream):
                        shared_stream.wait_event(input_ready)
                        hidden.record_stream(shared_stream)
                        shared = run()
                        ready = shared_stream.record_event()
                    return routed(operand, tokens, shared, ready)

                layer_reference = routed(operand, tokens, reference)
                for schedule, layer_run in (("serial", serial), ("overlap", overlap)):
                    torch.testing.assert_close(
                        layer_run(), layer_reference, atol=args.atol, rtol=args.rtol
                    )
                    print(
                        json.dumps(
                            {
                                "case": mode,
                                "boundary": schedule,
                                "tokens": tokens,
                                "tp_size": tp_size,
                                **timing(layer_run, args.warmup, args.iterations),
                            }
                        )
                    )


if __name__ == "__main__":
    main()
