from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

import torch

from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    NPUW4A4Int4DynamicMoEMethod,
)
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimMoEScheme
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)

__all__ = [
    "ModelSlimW4A4Int4MoE",
]


class ModelSlimW4A4Int4MoE(ModelSlimMoEScheme):

    def __init__(
        self,
        quant_config: Dict[str, Any],
        prefix: str = None,
    ):
        self.quant_config = quant_config
        self.kernel = NPUW4A4Int4DynamicMoEMethod()

    def create_moe_weight(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_type: str,  # "w13" or "w2"
        **extra_weight_attrs,
    ) -> None:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        self.num_experts = num_experts
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        if weight_type == "w13":
            prefix = "w13"
            a_dim = 2 * intermediate_size_per_partition
            b_dim = hidden_size                           
        elif weight_type == "w2":
            prefix = "w2"
            a_dim = hidden_size
            b_dim = intermediate_size_per_partition
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}. Use 'w13' or 'w2'.")

        # Create and register weight
        weight_name = f"{prefix}_weight"
        weight = torch.nn.Parameter(
            torch.empty(num_experts, a_dim, b_dim, dtype=torch.int8),
            requires_grad=False,
        )
        layer.register_parameter(weight_name, weight)
        set_weight_attrs(weight, extra_weight_attrs)

        # Create and register scale
        scale_name = f"{prefix}_weight_scale"
        scale = torch.nn.Parameter(
            torch.empty(num_experts, a_dim, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter(scale_name, scale)
        set_weight_attrs(scale, extra_weight_attrs)

        # Create and register offset
        offset_name = f"{prefix}_weight_offset"
        offset = torch.nn.Parameter(
            torch.empty(num_experts, a_dim, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter(offset_name, offset)
        set_weight_attrs(offset, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def process_quant_params_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_quant_params_after_loading(layer)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        moe_runner_config.quantization = "ModelSlimW4A4Int4MoE"
        self.kernel.create_moe_runner(layer, moe_runner_config)

    def apply_weights(
        self,
        layer,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        return self.kernel.apply(layer, dispatch_output)

    def apply_without_routing_weights(
        self,
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    ):
        # FIXME W4A4 MoE does not work with DeepEP
        raise NotImplementedError(
            f"DeepEP currently does not support quantization in int4, please disable --moe-a2a-backend deepep"
        )
