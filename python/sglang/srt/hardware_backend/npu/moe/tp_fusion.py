"""Opt-in A5 TP fusion contracts for MXFP8 activations and MXFP4 weights."""

from typing import Optional

import torch


def require_npu_op(name: str, arguments=()):
    op = getattr(torch.ops.npu, name, None)
    if op is None:
        raise RuntimeError(f"A5 TP fusion requires torch_npu operator {name}")
    schema = op.default._schema
    available = {arg.name for arg in schema.arguments}
    missing = set(arguments) - available
    if missing:
        raise RuntimeError(f"{name} ABI lacks {sorted(missing)}; found {schema}")
    return op


def mxfp8_input(hidden_states: torch.Tensor):
    operand = getattr(hidden_states, "_npu_mxfp8_operand", None)
    if operand is not None:
        return operand
    return torch.ops.npu.npu_dynamic_mx_quant(
        hidden_states, dst_type=torch.float8_e4m3fn
    )


def record_mxfp8_operand(operand):
    # Quantization can run on a different stream from either consumer.
    stream = torch.get_device_module().current_stream()
    for tensor in operand:
        tensor.record_stream(stream)


def supports_mxfp8_linear(layer) -> bool:
    from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
        NPUMXFP4W4A8OfflineLinearMethod,
        NPUMXFP8LinearMethod,
    )

    return isinstance(
        getattr(getattr(layer, "scheme", None), "kernel", layer.quant_method),
        (NPUMXFP4W4A8OfflineLinearMethod, NPUMXFP8LinearMethod),
    )


def prequantize_tp_input(experts, hidden_states):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.token_dispatcher.ascend_tp import AscendTPDispatcher
    from sglang.srt.layers.moe.utils import DispatcherOutputDtype

    dispatcher = experts.dispatcher
    if (
        not envs.SGLANG_NPU_TP_MOE_PREQUANT_INPUT.get()
        or hidden_states.shape[0] == 0
        or hidden_states.dtype != torch.bfloat16
        or not isinstance(dispatcher, AscendTPDispatcher)
        or dispatcher.local_ep
        or experts.moe_ep_size != 1
        or dispatcher.ascend_dispatcher_output_dtype != DispatcherOutputDtype.MXFP8
    ):
        return None
    return mxfp8_input(hidden_states)


def configure_tp_mxfp8_norm(norm, experts):
    from sglang.srt.environ import envs
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        NPUW4A8MXFP4MoEMethod,
    )
    from sglang.srt.layers.moe.token_dispatcher.ascend_tp import AscendTPDispatcher

    if not (
        envs.SGLANG_NPU_TP_MOE_PREQUANT_INPUT.get()
        and isinstance(experts.dispatcher, AscendTPDispatcher)
        and not experts.dispatcher.local_ep
        and experts.moe_ep_size == 1
        and isinstance(getattr(experts, "w2_kernel", None), NPUW4A8MXFP4MoEMethod)
    ):
        raise ValueError("Dual MXFP8 norm requires W4A8 MXFP TP prequantized routing")
    norm._npu_tp_mxfp8 = True


def tp_fused_shared_expert_reason(config, quant_config):
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.layers.moe.utils import is_sbo_enabled, is_tbo_enabled
    from sglang.srt.runtime_context import get_exec, get_parallel

    if (
        not get_moe_a2a_backend().is_none()
        or get_parallel().moe_ep_size != 1
        or config.n_shared_experts != 1
        or config.n_routed_experts != 256
        or quant_config is None
        or quant_config.get_name() != "modelslim"
        or is_sbo_enabled()
        or is_tbo_enabled()
        or getattr(config, "swiglu_limit", None) is not None
        or getattr(config, "num_experts_per_tok", None) != 8
        or getattr(config, "hidden_size", None) != 6144
        or get_exec().moe.enable_eplb
        or get_exec().moe.enable_waterfill
    ):
        raise ValueError("Fused shared expert requires ModelSlim TP with 256+1 experts")
    description = quant_config.quant_description
    shared = [
        key
        for key in description
        if isinstance(key, str)
        and ".shared_experts." in key
        and key.endswith(".weight")
    ]
    if not shared or any(
        description[key] != "W4A8_MXFP"
        or description.get(key.replace(".shared_experts.", ".experts.0."))
        != "W4A8_MXFP"
        for key in shared
    ):
        raise ValueError(
            "Fused shared expert requires identical W4A8_MXFP routed/shared weights"
        )
    layers = {key.rsplit(".shared_experts.", 1)[0] for key in shared}
    if any(
        value != "W4A8_MXFP"
        for key, value in description.items()
        if isinstance(key, str)
        and key.endswith(".weight")
        and any(key.startswith(f"{layer}.experts.") for layer in layers)
    ):
        raise ValueError("Fused shared expert requires uniform routed quantization")
    if any(
        f"{layer}.shared_experts.{projection}.weight" not in shared
        for layer in layers
        for projection in ("gate_proj", "up_proj", "down_proj")
    ):
        raise ValueError(
            "Fused shared expert requires descriptors for all shared projections"
        )
    return None


def shared_gmm1_weight_views(weight, weight_scale):
    # CANN checks full batch strides even for the singleton expert dimension.
    return (
        weight.transpose(0, 1).unsqueeze(0).transpose(1, 2),
        weight_scale.transpose(0, 1).unsqueeze(0).transpose(1, 2),
    )


def shared_gmm1_mode():
    from sglang.srt.environ import envs

    mode = envs.SGLANG_NPU_TP_MOE_SHARED_GMM1_MODE.get()
    if not mode:
        return (
            "grouped_fused" if envs.SGLANG_NPU_TP_MOE_SHARED_GMM1.get() else "baseline"
        )
    if mode not in ("baseline", "grouped_fused", "split_group_quant", "split3"):
        raise ValueError(f"Unknown SGLANG_NPU_TP_MOE_SHARED_GMM1_MODE: {mode!r}")
    return mode


def shared_gateup_quant(mlp, x, pre_quant_input=None, *, mode="grouped_fused"):
    from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
        NPUMXFP8LinearMethod,
    )
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        _get_float4_e2m1fn_x2_dtype,
        _require_e8m0_dtype,
    )

    gate = mlp.gate_up_proj
    if not supports_mxfp8_linear(gate) or not supports_mxfp8_linear(mlp.down_proj):
        raise ValueError("Shared GMM1 fusion requires MXFP8-compatible shared linears")
    if getattr(mlp, "swiglu_limit", None) is not None:
        raise ValueError("Shared GMM1 fusion does not support a clamped activation")
    if getattr(gate, "bias", None) is not None:
        raise ValueError("Shared GMM1 fusion requires a bias-free gate/up projection")
    qx, scale = mxfp8_input(x) if pre_quant_input is None else pre_quant_input
    record_mxfp8_operand((qx, scale))
    if mode in ("split_group_quant", "split3"):
        gate_up = gate((qx, scale))[0]
        return shared_activation_quant(mlp, gate_up, mode=mode)
    if mode != "grouped_fused":
        raise ValueError(f"Unsupported shared gate/up mode: {mode!r}")
    kernel = getattr(getattr(gate, "scheme", None), "kernel", gate.quant_method)
    weight_dtype = (
        None
        if isinstance(kernel, NPUMXFP8LinearMethod)
        else _get_float4_e2m1fn_x2_dtype()
    )
    weight_scale = gate.weight_scale_inv if weight_dtype is None else gate.weight_scale
    grouped_weight, grouped_scale = shared_gmm1_weight_views(gate.weight, weight_scale)
    group_key = (qx.shape[0], qx.device)
    cached = getattr(mlp, "_npu_shared_group_list", None)
    if cached is None or cached[0] != group_key:
        group_list = torch.full((1,), qx.shape[0], dtype=torch.int64, device=qx.device)
        mlp._npu_shared_group_list = (group_key, group_list)
    else:
        group_list = cached[1]
    group_list.record_stream(torch.get_device_module().current_stream())
    op = require_npu_op("npu_grouped_matmul_swiglu_quant_v2")
    quantized, output_scale = op(
        x=qx,
        weight=[grouped_weight],
        weight_scale=[grouped_scale],
        x_scale=scale,
        group_list=group_list,
        dequant_mode=2,
        quant_mode=2,
        dequant_dtype=torch.float32,
        quant_dtype=torch.float8_e4m3fn,
        weight_dtype=weight_dtype,
        weight_scale_dtype=_require_e8m0_dtype(),
        x_scale_dtype=_require_e8m0_dtype(),
    )
    return quantized, output_scale


def shared_activation_quant(mlp, gate_up, *, mode):
    if mode == "split_group_quant":
        # sgl-kernel-npu's MX ABI differs from torch_npu.npu_swiglu_group_quant.
        op = require_npu_op(
            "swiglu_group_quant",
            ("group_index", "dst_type", "quant_mode", "group_list_type", "clamp_value"),
        )
        quantized, scale, _ = op(
            x=gate_up,
            group_index=None,
            dst_type=torch.float8_e4m3fn,
            quant_mode=2,
            group_list_type=0,
            clamp_value=0.0,
        )
        return quantized, scale
    if mode == "split3":
        return mxfp8_input(mlp.act_fn(gate_up))
    raise ValueError(f"Unsupported shared activation mode: {mode!r}")


def shared_gmm1(mlp, x, pre_quant_input=None, *, mode="grouped_fused"):
    operand = shared_gateup_quant(mlp, x, pre_quant_input, mode=mode)
    return mlp.down_proj(operand)[0]


def fused_gmm2_finalize(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    expert_tokens: torch.Tensor,
    topk_weights: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    shared_output: Optional[torch.Tensor] = None,
):
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        _get_float4_e2m1fn_x2_dtype,
        _require_e8m0_dtype,
    )
    from sglang.srt.hardware_backend.npu.triton_kernel.tp_moe_fusion import (
        pack_finalize_routing,
    )

    op = require_npu_op(
        "npu_grouped_matmul_finalize_routing",
        ("w_dtype", "scale_dtype", "pertoken_scale_dtype", "shared_input", "dtype"),
    )
    row_index, logit = pack_finalize_routing(expanded_row_idx, topk_weights)
    output = op(
        x,
        weight,
        expert_tokens,
        scale=weight_scale,
        pertoken_scale=x_scale,
        shared_input=shared_output,
        shared_input_weight=1.0,
        logit=logit,
        row_index=row_index,
        output_bs=topk_weights.shape[0],
        group_list_type=1,
        # Deployed torch_npu 2.10 wrappers require FP32 output.
        dtype=torch.float32,
        w_dtype=_get_float4_e2m1fn_x2_dtype(),
        scale_dtype=_require_e8m0_dtype(),
        pertoken_scale_dtype=_require_e8m0_dtype(),
    )
    return output.to(torch.bfloat16)
