"""Checkpoint-selected router precision and fused boundary selection."""

import logging

import torch

from sglang.srt.hardware_backend.npu.moe.contracts import ActivationContract

logger = logging.getLogger(__name__)


def gate_input_contract(gate) -> "ActivationContract | None":
    """The activation contract the configured gate consumes, if quantized."""
    if hasattr(gate, "quant_method"):
        from sglang.srt.hardware_backend.npu.moe.contracts import (
            linear_input_contract,
        )

        return linear_input_contract(gate)
    return None


def offline_gate_config(quant_config, prefix):
    """Resolve this router's descriptor, including checkpoint prefix aliases."""
    if quant_config is None or quant_config.get_name() != "modelslim":
        return None
    scheme = quant_config.resolve_projection_scheme(prefix)
    if scheme in (None, "", "FLOAT"):
        return None
    if scheme not in quant_config.linear_scheme_types():
        raise ValueError(f"Offline router {prefix} uses unsupported ModelSlim {scheme}")
    return quant_config


def configure_offline_gate(gate, layer_prefix: str = "", quant_config=None) -> bool:
    """Prepare an offline-quantized router from its checkpoint descriptor."""
    config = offline_gate_config(quant_config, layer_prefix)
    if config is not None:
        # Custom FP32 routers use the same checkpoint parameters and loaders
        # as ReplicatedLinear, without changing their correction-bias slots.
        from sglang.srt.layers.quantization.modelslim.modelslim import (
            ModelSlimLinearMethod,
        )
        from sglang.srt.model_loader.weight_utils import default_weight_loader

        output_size, input_size = gate.weight.shape
        gate.scheme = config.get_linear_scheme(gate, layer_prefix)
        gate.quant_method = ModelSlimLinearMethod(config)
        gate.quant_method.create_weights(
            gate,
            input_size,
            [output_size],
            input_size,
            output_size,
            torch.bfloat16,
            weight_loader=default_weight_loader,
        )
        gate._npu_gate_method = gate.quant_method
        return False
    return False


def select_moe_boundary_quant_format(moe_block) -> str:
    """Select the fused norm+quant format for a MoE input boundary, or ''.

    Fusion requires the COMPLETE consumer set (router gate, W13, shared
    experts) to accept one typed activation contract: a floating-point gate
    without online quantization disables the fused boundary (the unfused
    reference norm + independent quantization stays). Packed contracts
    (MXFP4/INT4) keep the separate norm and quantization path.
    """
    from sglang.srt.hardware_backend.npu.layernorm_quant import (
        can_fuse_norm_quant,
    )
    from sglang.srt.hardware_backend.npu.moe.contracts import (
        moe_input_contract,
    )
    from sglang.srt.layers.dp_attention import is_dp_attention_enabled
    from sglang.srt.layers.moe.token_dispatcher.ascend_tp import AscendTPDispatcher
    from sglang.srt.runtime_context import get_parallel

    if (
        is_dp_attention_enabled()
        or get_parallel().attn_tp_size != get_parallel().tp_size
    ):
        return ""
    if getattr(moe_block, "is_hash", False) or getattr(
        moe_block, "_shared_expert_tp1", False
    ):
        return ""
    if getattr(moe_block, "shared_expert_gate", None) is not None:
        return ""
    parallel = get_parallel()
    if (
        parallel.attn_cp_size != 1
        or parallel.moe_dp_size != 1
        or parallel.enable_prefill_cp
    ):
        return ""
    experts = moe_block.experts
    if not isinstance(experts.dispatcher, AscendTPDispatcher):
        return ""
    if experts.moe_ep_size != 1:
        return ""
    consumers = [gate_input_contract(moe_block.gate)]
    w13 = moe_input_contract(getattr(experts, "w13_kernel", None))
    consumers.append(w13)
    shared = getattr(moe_block, "shared_experts", None)
    if shared is not None and not getattr(moe_block, "num_fused_shared_experts", 0):
        from sglang.srt.hardware_backend.npu.moe.contracts import (
            linear_input_contract,
        )

        consumers.append(linear_input_contract(shared.gate_up_proj))
    contract, reason = can_fuse_norm_quant(consumers)
    if contract is None:
        logger.info_once("NPU fused MoE norm+quant declined: %s", reason)
        return ""
    if contract.kind in ("int4", "mxfp4"):
        # Packed passthrough is unavailable; keep norm and quantization separate.
        logger.info(
            "NPU fused MoE norm+quant declined: %s passthrough unavailable",
            contract.kind,
        )
        return ""
    logger.info_once("NPU fused MoE norm+quant selected: %s", contract.kind)
    return f"npu_{contract.kind}"


def gate_forward(gate, hidden_states):
    """Run an offline-quantized router, or the original floating-point gate."""
    method = getattr(gate, "_npu_gate_method", None)
    if method is not None:
        return method.apply(gate, hidden_states).float()
    return gate(hidden_states)


def select_linear_boundary_quant_format(projections):
    from sglang.srt.hardware_backend.npu.layernorm_quant import can_fuse_norm_quant
    from sglang.srt.hardware_backend.npu.moe.contracts import linear_input_contract
    from sglang.srt.layers.dp_attention import is_dp_attention_enabled
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if (
        is_dp_attention_enabled()
        or parallel.attn_cp_size != 1
        or parallel.enable_prefill_cp
        or parallel.attn_tp_size != parallel.tp_size
    ):
        return ""
    contract, _ = can_fuse_norm_quant(
        [linear_input_contract(proj) for proj in projections]
    )
    return f"npu_{contract.kind}" if contract is not None else ""
