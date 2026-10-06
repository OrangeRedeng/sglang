"""Replay captured GLM-5.2 TP4 prefill norm and gate boundaries on Ascend A5.

Capture dictionary: x, residual, gamma, eps, loaded gate_weight, correction_bias,
tp_size=4 and topk (the production TopKConfig fields saved by input capture).
Optional deterministic flag records the production FP32 reference override.
Route mismatches fail acceptance but all timing/error diagnostics are printed.
Optional --outputs compares separately captured moe_output, model_logits and
generated_ids from identical deterministic A/B serving requests.
"""

import argparse
import json

import torch
import torch_npu
from npu_tp_bench_utils import npu_router_reference, timing

from sglang.srt.hardware_backend.npu.moe.norm_gate import (
    MXFP8GateMethod,
    mxfp8_gate_op,
    native_add_rmsnorm_mxfp8,
    native_norm_op,
)


def error(actual, reference):
    diff = (actual.float() - reference.float()).abs()
    return {"max_abs": diff.max().item(), "mean_abs": diff.mean().item()}


def route(logits, bias, cfg):
    return torch.ops.npu.npu_moe_gating_top_k(
        logits.float(),
        k=cfg["top_k"],
        bias=bias,
        k_group=cfg["topk_group"] if cfg["use_grouped_topk"] else 1,
        group_count=cfg["num_expert_group"] if cfg["use_grouped_topk"] else 1,
        group_select_mode=1 if cfg["use_grouped_topk"] else 0,
        renorm=cfg["renormalize"],
        norm_type=0 if cfg["scoring_func"] == "softmax" else 1,
        routed_scaling_factor=(
            cfg["routed_scaling_factor"]
            if cfg["apply_routed_scaling_factor_on_output"]
            else 1
        ),
        eps=1e-20,
    )[:2]


def routing_report(reference, actual, bias, cfg):
    if not (
        torch.isfinite(reference).all().item() and torch.isfinite(actual).all().item()
    ):
        raise RuntimeError("Router logits contain NaN or Inf")
    reference, actual = reference.float(), actual.float()
    w0, ids0 = route(reference, bias, cfg)
    w1, ids1 = route(actual, bias, cfg)
    changed = ids0 != ids1
    changed_rows = changed.any(dim=1)
    set_match = (ids0.sort(dim=1).values == ids1.sort(dim=1).values).all(dim=1)
    sorted_logits = reference.topk(cfg["top_k"] + 1, dim=1).values
    margin = sorted_logits[:, -2] - sorted_logits[:, -1]
    scores = (
        reference.softmax(dim=1)
        if cfg["scoring_func"] == "softmax"
        else reference.sigmoid()
    )
    if bias is not None:
        scores = scores + bias
    sorted_scores = scores.topk(cfg["top_k"] + 1, dim=1).values
    score_margin = sorted_scores[:, -2] - sorted_scores[:, -1]
    dense0, dense1 = torch.zeros_like(reference), torch.zeros_like(actual)
    dense0.scatter_(1, ids0.long(), w0.float())
    dense1.scatter_(1, ids1.long(), w1.float())
    report = {
        "topk_ids_exact_match": (~changed_rows).float().mean().item(),
        "topk_set_match": set_match.float().mean().item(),
        "top1_match": (reference.argmax(dim=1) == actual.argmax(dim=1))
        .float()
        .mean()
        .item(),
        "first_route_match": (ids0[:, 0] == ids1[:, 0]).float().mean().item(),
        "changed_routes": changed.sum().item(),
        "changed_route_memberships": (
            ~(ids1.unsqueeze(2) == ids0.unsqueeze(1)).any(dim=2)
        )
        .sum()
        .item(),
        "total_routes": ids0.numel(),
        "changed_route_fraction": changed.float().mean().item(),
        "changed_tokens": changed_rows.sum().item(),
        "router_logit_error": error(actual, reference),
        "routing_weight_slot_error": error(w1, w0),
        "routing_weight_by_expert_error": error(dense1, dense0),
        "reference_kth_kplus1_logit_margin_on_changed_tokens": (
            {
                "min": margin[changed_rows].min().item(),
                "mean": margin[changed_rows].mean().item(),
            }
            if changed_rows.any().item()
            else None
        ),
        "score_plus_bias_margin_on_changed_tokens": (
            {
                "min": score_margin[changed_rows].min().item(),
                "mean": score_margin[changed_rows].mean().item(),
            }
            if changed_rows.any().item()
            else None
        ),
        "accepted_exact_routes": not changed.any().item(),
    }
    return report


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--allow-route-mismatch", action="store_true")
    parser.add_argument("--outputs", nargs=2, metavar=("A_PT", "B_PT"))
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.01)
    args = parser.parse_args()
    if args.warmup < 20 or args.iterations < 100:
        parser.error("Use at least 20 warmups and 100 iterations")
    torch.npu.set_device(args.device)
    print(json.dumps({"torch": torch.__version__, "torch_npu": torch_npu.__version__}))
    method = MXFP8GateMethod()
    for op in (native_norm_op(), method.quantize, mxfp8_gate_op()):
        print(json.dumps({"schema": str(op.default._schema)}))
    if args.probe_only:
        return
    if not args.inputs:
        parser.error("--inputs is required for replay")
    data = torch.load(args.inputs, map_location="cpu", weights_only=True)
    x, gamma, gate_weight = (
        data[key].to(args.device) for key in ("x", "gamma", "gate_weight")
    )
    if data["tp_size"] != 4 or x.ndim != 2 or x.shape[1] != 6144 or not x.shape[0]:
        parser.error("Expected nonempty captured TP4 [T,6144] input")
    if x.dtype != torch.bfloat16 or gate_weight.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        parser.error("Expected BF16 input and loaded FP16/BF16/FP32 gate weight")
    residual = data["residual"]
    residual = torch.zeros_like(x) if residual is None else residual.to(args.device)
    bias = data["correction_bias"]
    bias = None if bias is None else bias.to(args.device)
    cfg = data["topk"]
    if cfg["top_k"] != 8 or cfg["scoring_func"] not in ("softmax", "sigmoid"):
        parser.error("Expected the production GLM TopK8 configuration")
    gate = torch.nn.Module()
    gate.register_parameter(
        "weight", torch.nn.Parameter(gate_weight, requires_grad=False)
    )
    method.process_weights_after_loading(gate)
    deterministic = bool(data.get("deterministic", False))

    def baseline_gate(normalized):
        return npu_router_reference(
            normalized, gate.weight, deterministic=deterministic
        )

    def baseline_norm():
        normalized, _, residual_out = torch_npu.npu_add_rms_norm(
            residual, x, gamma, float(data["eps"])
        )
        return normalized, residual_out

    def native_norm():
        return native_add_rmsnorm_mxfp8(x, residual, gamma, float(data["eps"]))

    def baseline():
        normalized, residual_out = baseline_norm()
        logits = baseline_gate(normalized)
        operand = torch.ops.npu.npu_dynamic_mx_quant(
            normalized, dst_type=torch.float8_e4m3fn
        )
        return logits, residual_out, operand

    def native(output_dtype=torch.float32):
        q, residual_out, scale, _ = native_norm()
        return (
            method.apply(gate, (q, scale), output_dtype=output_dtype),
            residual_out,
            (q, scale),
        )

    logits0, residual0, operand0 = baseline()
    logits1, residual1, operand1 = native()
    torch.testing.assert_close(residual1, residual0, atol=0, rtol=0)
    norm_report = {}
    norm_report["input_payload_byte_match"] = (
        (operand0[0].view(torch.uint8) == operand1[0].view(torch.uint8))
        .float()
        .mean()
        .item()
    )
    norm_report["input_scale_byte_match"] = (
        (
            operand0[1].view(torch.uint8).reshape(-1)
            == operand1[1].view(torch.uint8).reshape(-1)
        )
        .float()
        .mean()
        .item()
    )
    print(
        json.dumps(
            {
                "capture": args.inputs,
                "tokens": x.shape[0],
                "gate_weight_dtype": str(gate.weight.dtype),
                "baseline_router_dtype": str(logits0.dtype),
                "deterministic_reference": deterministic,
                "norm_correctness": norm_report,
            }
        )
    )
    reports = {}
    for mode, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        logits = (
            logits1
            if mode == "fp32"
            else method.apply(gate, operand1, output_dtype=dtype)
        )
        reports[mode] = routing_report(logits0, logits, bias, cfg)
        print(
            json.dumps(
                {
                    "mode": mode,
                    "router_dtype": str(logits.dtype),
                    "correctness": reports[mode],
                }
            )
        )
    normalized, _ = baseline_norm()
    stages = {
        "baseline_total": baseline,
        "native_fp32_total": native,
        "native_bf16_total": lambda: native(torch.bfloat16),
        "baseline_norm": baseline_norm,
        "native_norm_quant": native_norm,
        "baseline_gate": lambda: baseline_gate(normalized),
        "native_fp32_gate": lambda: method.apply(
            gate, operand1, output_dtype=torch.float32
        ),
        "native_bf16_gate": lambda: method.apply(
            gate, operand1, output_dtype=torch.bfloat16
        ),
        "baseline_quant": lambda: torch.ops.npu.npu_dynamic_mx_quant(
            normalized, dst_type=torch.float8_e4m3fn
        ),
    }
    for stage, run in stages.items():
        print(json.dumps({"stage": stage, **timing(run, args.warmup, args.iterations)}))
    print(json.dumps({"stage": "native_standalone_quant", "p50_ms": 0, "p95_ms": 0}))
    if args.outputs:
        a, b = (
            torch.load(path, map_location="cpu", weights_only=True)
            for path in args.outputs
        )
        for key in ("moe_output", "model_logits"):
            print(json.dumps({key: error(b[key], a[key])}))
            torch.testing.assert_close(b[key], a[key], atol=args.atol, rtol=args.rtol)
        torch.testing.assert_close(
            b["generated_ids"], a["generated_ids"], atol=0, rtol=0
        )
        print(json.dumps({"deterministic_generation_exact": True}))
    failed_modes = [
        mode for mode, report in reports.items() if not report["accepted_exact_routes"]
    ]
    if failed_modes and not args.allow_route_mismatch:
        raise SystemExit(
            f"Routing acceptance failed for {failed_modes}: TopK IDs must match exactly"
        )


if __name__ == "__main__":
    main()
