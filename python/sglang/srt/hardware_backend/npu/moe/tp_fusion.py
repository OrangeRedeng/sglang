"""Opt-in A5 TP fusion contracts for MXFP8 activations and MXFP4 weights."""

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Optional

import torch

logger = logging.getLogger(__name__)
_shared_pipeline = ContextVar("npu_tp_shared_pipeline", default=None)


def shared_pipeline_mode():
    from sglang.srt.environ import envs

    mode = envs.SGLANG_NPU_TP_MOE_SHARED_PIPELINE.get()
    if mode not in ("legacy", "resource"):
        raise ValueError(f"Unknown SGLANG_NPU_TP_MOE_SHARED_PIPELINE: {mode!r}")
    return mode


def current_shared_pipeline():
    return _shared_pipeline.get()


@contextmanager
def use_shared_pipeline(pipeline):
    token = _shared_pipeline.set(pipeline)
    try:
        yield
    finally:
        _shared_pipeline.reset(token)


def shared_resource_blockers(state):
    required = {
        "has_shared_stream": True,
        "is_extend_in_batch": True,
        "is_nextn": False,
        "is_glm_moe_dsa": True,
        "shared_gmm1_mode": "grouped_fused",
        "swiglu_limit": None,
        "runner_inplace": False,
        "capture_mode": False,
        "breakable_graph": False,
        "piecewise_graph": False,
        "sp_active": False,
        "down_proj_decode_attn_tp": False,
        "skip_shared_experts": False,
        "token_threshold_met": True,
    }
    return tuple(key for key, expected in required.items() if state[key] != expected)


@lru_cache(maxsize=None)
def log_shared_pipeline(active, state_items):
    state = dict(state_items)
    fields = " ".join(f"{key}={value}" for key, value in state_items)
    blockers = ",".join(shared_resource_blockers(state)) or "none"
    logger.info(
        "%s: blockers=%s %s",
        "TP shared resource pipeline is ACTIVE"
        if active
        else (
            "TP shared resource pipeline REQUESTED but OFF"
            if state["requested"]
            else "TP shared resource pipeline is OFF"
        ),
        blockers,
        fields,
    )


class TPSharedResourcePipeline:
    """Per-forward events, with no persistent dispatcher hooks or tensor state.

    The grouped_fused contract has no Vector-only activation boundary. Do not
    split either fused GMM1 just to claim more overlap: enqueue Down only after
    routed GMM2, and retain the existing shared-input finalization math.
    """

    def __init__(self, mlp, stream, *, fuse_shared):
        self.mlp = mlp
        self.stream = stream
        self.fuse_shared = fuse_shared
        self.operand = None
        self.output = None
        self.shared_gateup_ready = None
        self.shared_quant_ready = None
        self.shared_down_ready = None

    def start_gateup(self, hidden_states, pre_quant_input):
        main = torch.npu.current_stream()
        ready = main.record_event()
        with torch.npu.stream(self.stream):
            self.stream.wait_event(ready)
            hidden_states.record_stream(self.stream)
            for tensor in self.mlp.parameters():
                tensor.record_stream(self.stream)
            for tensor in self.mlp.buffers():
                tensor.record_stream(self.stream)
            self.operand = shared_gateup_quant(
                self.mlp, hidden_states, pre_quant_input, mode="grouped_fused"
            )
            record_mxfp8_operand(self.operand)
            self.shared_gateup_ready = self.stream.record_event()
            # Fused output already contains activation and MX quantization.
            self.shared_quant_ready = self.shared_gateup_ready

    def before_routed_gmm1(self):
        torch.npu.current_stream().wait_event(self.shared_quant_ready)

    def after_routed_gmm2(self):
        ready = torch.npu.current_stream().record_event()
        with torch.npu.stream(self.stream):
            self.stream.wait_event(ready)
            record_mxfp8_operand(self.operand)
            self.output = shared_down(self.mlp, self.operand)
            self.output.record_stream(self.stream)
            self.shared_down_ready = self.stream.record_event()

    def wait_output(self):
        if self.shared_down_ready is None:
            raise RuntimeError("TP resource pipeline did not reach routed GMM2")
        main = torch.npu.current_stream()
        main.wait_event(self.shared_down_ready)
        self.output.record_stream(main)
        return self.output


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
    scale = getattr(hidden_states, "_npu_mxfp8_scale", None)
    if scale is not None:
        return hidden_states, scale
    operand = getattr(hidden_states, "_npu_mxfp8_operand", None)
    if operand is not None:
        return operand
    if hidden_states.dtype == torch.float8_e4m3fn:
        raise RuntimeError("MXFP8 input is missing its E8M0 scales")
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
    if getattr(hidden_states, "_npu_mxfp8_scale", None) is not None:
        return mxfp8_input(hidden_states)
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


def shared_gateup(mlp, x, pre_quant_input=None):
    operand = mxfp8_input(x) if pre_quant_input is None else pre_quant_input
    record_mxfp8_operand(operand)
    return mlp.gate_up_proj(operand)[0]


def shared_down(mlp, operand):
    record_mxfp8_operand(operand)
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
