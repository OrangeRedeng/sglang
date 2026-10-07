"""Load-time layout comparison using real gate weights and fixed router dtype."""

import logging
import statistics
import time

import torch

from sglang.srt.hardware_backend.npu.autotune import (
    LAYOUTS,
    NpuTuningDecision,
    active_context,
    active_profile,
    choose_measured,
    enabled,
    log_decision,
    profile_key,
    representative_prefill_rows,
    save_profile,
)

logger = logging.getLogger(__name__)
_winners = {}
_logged_choices = set()


def env_nz_disabled():
    from sglang.srt.environ import envs

    return envs.SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT.get()


def tune_gate(method, layer, q_weight, scale_weight):
    context = active_context()
    requested = {
        key: getattr(method, attr) == "auto"
        for key, attr in (
            ("gate_weight_layout", "weight_layout"),
            ("gate_topk_layout", "topk_layout"),
        )
    }
    defaults = {"gate_weight_layout": "transposed", "gate_topk_layout": "default"}
    baseline = {
        key: defaults[key] if requested[key] else getattr(method, attr)
        for key, attr in (
            ("gate_weight_layout", "weight_layout"),
            ("gate_topk_layout", "topk_layout"),
        )
    }
    profile = active_profile()
    selected = dict(baseline)
    reason = "no serving context; retain baseline"
    if context is not None:
        from sglang.srt.runtime_context import get_parallel

        group = get_parallel().tp_group
        key = (profile_key(context), str(method.output_dtype), tuple(baseline.items()))
        payload = [None]
        if group.rank_in_group == 0:
            selected = {
                k: profile.get(k, baseline[k]) if requested[k] else baseline[k]
                for k in baseline
            }
            if method.format_cast is None or env_nz_disabled():
                if selected["gate_weight_layout"] == "nz":
                    selected["gate_weight_layout"] = baseline["gate_weight_layout"]
                    profile.pop("gate_weight_layout", None)
            if method.format_cast is None and selected["gate_topk_layout"] == "nd":
                selected["gate_topk_layout"] = baseline["gate_topk_layout"]
                profile.pop("gate_topk_layout", None)
            missing = [k for k in selected if requested[k] and k not in profile]
            reason = "cached-profile"
            if missing:
                if key in _winners:
                    selected, reason = _winners[key]
                else:
                    try:
                        selected, evidence = _measure(
                            method,
                            layer,
                            q_weight,
                            scale_weight,
                            context,
                            selected,
                            missing,
                        )
                        reason = evidence["status"]
                        if not enabled("SGLANG_NPU_AUTOTUNE_DRY_RUN"):
                            additions = {
                                k: selected[k]
                                for k in missing
                                if evidence["accepted"].get(k)
                            }
                            if additions:
                                save_profile(
                                    context, {**active_profile(), **additions}, evidence
                                )
                    except Exception as exc:
                        logger.warning(
                            "NPU gate autotune failed; retaining baseline: %s", exc
                        )
                        selected = dict(baseline)
                        reason = f"fallback: {type(exc).__name__}"
                    _winners[key] = (selected, reason)
            payload[0] = (selected, reason)
        # Use rank zero's cache even when cache directories differ between nodes.
        if group.world_size > 1:
            torch.distributed.broadcast_object_list(
                payload, src=group.ranks[0], group=group.cpu_group
            )
        selected, reason = payload[0]
    for key, attr in (
        ("gate_weight_layout", "weight_layout"),
        ("gate_topk_layout", "topk_layout"),
    ):
        if requested[key]:
            choice_key = (key, selected[key], str(method.output_dtype))
            if choice_key not in _logged_choices:
                _logged_choices.add(choice_key)
                log_decision(
                    NpuTuningDecision(
                        key,
                        selected[key],
                        "cached-profile" if key in profile else "microbenchmark",
                        f"{reason} dtype={method.output_dtype} shape={tuple(q_weight.shape)}",
                    )
                )
            setattr(
                method,
                attr,
                baseline[key]
                if enabled("SGLANG_NPU_AUTOTUNE_DRY_RUN")
                else selected[key],
            )


def _measure(method, layer, q_weight, scale_weight, context, selected, missing):
    from sglang.srt.environ import envs
    from sglang.srt.hardware_backend.npu.moe.norm_gate import mx_scale_layout
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        _require_e8m0_dtype,
    )

    cfg = method.topk_config
    primary = representative_prefill_rows(context)
    if cfg.scoring_func not in ("sigmoid", "softmax"):
        raise ValueError("unsupported serving shape or router scoring function")
    points = [(primary, 0.9), (min(1024, primary), 0.1)]
    operands = []
    for rows, weight in points:
        generator = torch.Generator(device="cpu").manual_seed(0)
        x = torch.randn(
            (rows, q_weight.shape[1]), generator=generator, dtype=torch.bfloat16
        ).to(q_weight.device)
        q, scale = method.quantize(
            x,
            dst_type=torch.float8_e4m3fn,
            block_size=32,
            scale_alg=0,
            round_mode="rint",
        )
        operands.append((q, mx_scale_layout(scale, rows, q.shape[1]), weight))
    bias = (
        cfg.correction_bias.to(torch.float32)
        if cfg.correction_bias is not None
        else None
    )

    def run(layout, topk, operand):
        q, scale, _ = operand
        w, ws = weights[layout]
        logits = method.matmul(
            q,
            w,
            ws,
            pertoken_scale=scale,
            scale_dtype=_require_e8m0_dtype(),
            pertoken_scale_dtype=_require_e8m0_dtype(),
            output_dtype=method.output_dtype,
            group_sizes=[1, 1, 32],
        )
        topk_logits = logits.float()
        if topk == "contiguous":
            topk_logits = topk_logits.contiguous()
        elif topk == "nd":
            topk_logits = method.format_cast(topk_logits, 2)
        elif topk == "clone":
            topk_logits = topk_logits.clone(memory_format=torch.contiguous_format)
        if not cfg.use_grouped_topk and bias is None:
            routed = torch.ops.npu.npu_moe_gating_top_k_softmax(
                logits, k=cfg.top_k - cfg.num_fused_shared_experts
            )[:2]
        else:
            routed = torch.ops.npu.npu_moe_gating_top_k(
                topk_logits,
                k=cfg.top_k - cfg.num_fused_shared_experts,
                bias=bias,
                k_group=cfg.topk_group if cfg.use_grouped_topk else 1,
                group_count=cfg.num_expert_group if cfg.use_grouped_topk else 1,
                group_select_mode=int(cfg.use_grouped_topk),
                renorm=cfg.renormalize,
                norm_type=0 if cfg.scoring_func == "softmax" else 1,
                routed_scaling_factor=cfg.routed_scaling_factor
                if cfg.apply_routed_scaling_factor_on_output
                else 1,
                eps=1e-20,
            )[:2]
        return logits, *routed

    weights = {}
    choices = list(LAYOUTS["gate_weight_layout"])
    if envs.SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT.get() or method.format_cast is None:
        choices.remove("nz")
    for layout in dict.fromkeys(
        choices + [selected["gate_weight_layout"], "transposed"]
    ):
        w, ws = (
            q_weight.T,
            mx_scale_layout(
                scale_weight, q_weight.shape[0], q_weight.shape[1]
            ).transpose(0, 1),
        )
        if layout != "transposed":
            w, ws = w.contiguous(), ws.contiguous()
        if layout == "nz":
            try:
                w = method.format_cast(w, 29)
            except RuntimeError:
                continue
        weights[layout] = w, ws
    reference = [run("transposed", "default", op) for op in operands]
    evidence = {
        "status": "inconclusive",
        "accepted": {},
        "scores": {},
        "rejected": {},
        "correctness": {},
        "final_layer_output_error": None,
        "generation_match": None,
        "rows": [p[0] for p in points],
        "warmup": 5,
        "iterations": 20,
    }
    for key in missing:
        candidates = (
            choices
            if key == "gate_weight_layout"
            else tuple(
                c for c in LAYOUTS[key] if c != "nd" or method.format_cast is not None
            )
        )
        scores = {}
        for candidate in candidates:
            layouts = dict(selected, **{key: candidate})
            try:
                samples = []
                reports = []
                evidence["correctness"][f"{key}:{candidate}"] = reports
                for i, op in enumerate(operands):
                    result = run(
                        layouts["gate_weight_layout"], layouts["gate_topk_layout"], op
                    )
                    logits, route_weights, route_ids = result
                    ref_logits, ref_weights, ref_ids = reference[i]
                    reports.append(
                        {
                            "rows": op[0].shape[0],
                            "topk_ordered_match": bool(torch.equal(route_ids, ref_ids)),
                            "topk_unordered_match": bool(
                                torch.equal(
                                    route_ids.sort(dim=-1).values,
                                    ref_ids.sort(dim=-1).values,
                                )
                            ),
                            "top1_match": bool(
                                torch.equal(route_ids[:, 0], ref_ids[:, 0])
                            ),
                            "changed_routes": int((route_ids != ref_ids).sum().item()),
                            "total_routes": route_ids.numel(),
                            "routing_weight_max_error": float(
                                (route_weights.float() - ref_weights.float())
                                .abs()
                                .max()
                                .item()
                            ),
                        }
                    )
                    if not bool(
                        torch.isfinite(logits).all().item()
                        and torch.isfinite(ref_logits).all().item()
                    ):
                        raise ValueError("router logits are not finite")
                    # Exact logits, ordered routes and weights protect the router boundary.
                    if not all(torch.equal(a, b) for a, b in zip(result, reference[i])):
                        raise ValueError(
                            "logits/routes/weights differ from fixed-dtype baseline"
                        )
                    for _ in range(5):
                        run(
                            layouts["gate_weight_layout"],
                            layouts["gate_topk_layout"],
                            op,
                        )
                    times = []
                    for _ in range(20):
                        torch.npu.synchronize()
                        start = time.perf_counter()
                        run(
                            layouts["gate_weight_layout"],
                            layouts["gate_topk_layout"],
                            op,
                        )
                        torch.npu.synchronize()
                        times.append(time.perf_counter() - start)
                    samples.append(times)
                scores[candidate] = [
                    sum(samples[p][i] * operands[p][2] for p in range(len(operands)))
                    for i in range(20)
                ]
            except (RuntimeError, ValueError, KeyError) as exc:
                evidence["rejected"][f"{key}:{candidate}"] = str(exc)
        winner, status = choose_measured(scores, selected[key])
        selected[key] = winner
        evidence["scores"][key] = {k: statistics.median(v) for k, v in scores.items()}
        evidence["accepted"][key] = status == "measured"
        if status == "measured":
            evidence["status"] = status
    return selected, evidence
