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


def shared_gmm1(mlp, x, pre_quant_input=None):
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
    kernel = getattr(getattr(gate, "scheme", None), "kernel", gate.quant_method)
    weight_dtype = (
        None
        if isinstance(kernel, NPUMXFP8LinearMethod)
        else _get_float4_e2m1fn_x2_dtype()
    )
    weight_scale = gate.weight_scale_inv if weight_dtype is None else gate.weight_scale
    group_list = torch.full((1,), qx.shape[0], dtype=torch.int64, device=qx.device)
    op = require_npu_op("npu_grouped_matmul_swiglu_quant_v2")
    quantized, output_scale = op(
        x=qx,
        weight=[gate.weight.unsqueeze(0)],
        weight_scale=[weight_scale.unsqueeze(0)],
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
    return mlp.down_proj((quantized, output_scale))[0]


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
    return op(
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
        dtype=torch.bfloat16,
        w_dtype=_get_float4_e2m1fn_x2_dtype(),
        scale_dtype=_require_e8m0_dtype(),
        pertoken_scale_dtype=_require_e8m0_dtype(),
    )
