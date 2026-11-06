"""Native TP MoE capability checks and operand lifetime validation."""

import ctypes
from functools import lru_cache

import torch

from sglang.srt.hardware_backend.npu.moe.contracts import (
    QuantizedOperand,
    moe_input_contract,
)


@lru_cache(maxsize=None)
def supports_npu_op(name, arguments=()):
    op = getattr(torch.ops.npu, name, None)
    if op is None or not hasattr(op, "default"):
        return False
    available = {arg.name for arg in op.default._schema.arguments}
    return set(arguments).issubset(available)


@lru_cache(maxsize=None)
def supports_cann_symbol(symbol):
    for name in (None, "libopapi.so", "libopapi_transformer.so"):
        try:
            library = ctypes.CDLL(name)
        except OSError:
            continue
        if hasattr(library, symbol) and hasattr(library, symbol + "GetWorkspaceSize"):
            return True
    return False


def supports_fused_mxfp_gmm(kernel, weight=None, scale=None):
    from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

    if weight is None or scale is None or weight.device.type != "npu":
        return False
    import torch_npu

    # op-plugin selects the ABI from the prepared weight format, not its dtype.
    nz = torch_npu.get_npu_format(weight) in (29, 50)
    if not fused_mxfp_layout(kernel, weight, scale):
        return False
    symbol = (
        "aclnnGroupedMatmulSwigluQuantWeightNzV2"
        if nz
        else "aclnnGroupedMatmulSwigluQuantV2"
    )
    return (
        is_npu_arch35()
        and supports_npu_op("npu_grouped_matmul_swiglu_quant_v2")
        and supports_cann_symbol(symbol)
    )


def fused_mxfp_layout(kernel, weight, scale):
    from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
        NPUMXFP4W4A8OfflineLinearMethod,
    )
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        NPUW4A8MXFP4MoEMethod,
    )

    if (
        weight.ndim != 3
        or scale.ndim != 4
        or scale.shape[0] != weight.shape[0]
        or scale.shape[-1] != 2
        or scale.device != weight.device
        or scale.dtype
        not in (torch.uint8, getattr(torch, "float8_e8m0fnu", torch.uint8))
    ):
        return False
    packed = isinstance(
        kernel, (NPUW4A8MXFP4MoEMethod, NPUMXFP4W4A8OfflineLinearMethod)
    )
    experts, stored_k, n = weight.shape
    k = stored_k * (2 if packed else 1)
    return (
        k > 0
        and k % 64 == 0
        and n > 0
        and n % 64 == 0
        and (not packed or n >= 128)
        and weight.dtype
        in ((torch.uint8, torch.float8_e4m3fn) if packed else (torch.float8_e4m3fn,))
        and tuple(scale.shape) == (experts, k // 64, n, 2)
        and weight.stride() == (stored_k * n, 1, stored_k)
        and scale.stride() == (k // 32 * n, 2, k // 32, 1)
    )


def record_quantized_operand(operand):
    if isinstance(operand, QuantizedOperand):
        operand.validate(operand.contract)
        operand.record_stream()
        return
    stream = torch.get_device_module(operand[0].device).current_stream()
    for tensor in operand:
        tensor.record_stream(stream)


def record_tp_shared_load(layer, param, shard_id, expert_id):
    from sglang.srt.layers.moe.token_dispatcher.ascend_tp import AscendTPDispatcher

    if (
        isinstance(layer.dispatcher, AscendTPDispatcher)
        and layer.moe_ep_size == 1
        and layer.num_fused_shared_experts == 1
        and expert_id == layer.num_local_experts - 1
        and layer.quant_config is not None
        and layer.quant_config.get_name() == "modelslim"
    ):
        loaded = getattr(param, "_npu_shared_loaded_shards", set())
        loaded.add(shard_id)
        param._npu_shared_loaded_shards = loaded


def validate_tp_expert_weights(layer, *, prepared=True):
    from sglang.srt.hardware_backend.npu.moe.contracts import projection_contract
    from sglang.srt.layers.moe.token_dispatcher.ascend_tp import AscendTPDispatcher

    if (
        not isinstance(getattr(layer, "dispatcher", None), AscendTPDispatcher)
        or layer.moe_ep_size != 1
    ):
        return
    if prepared:
        layer._npu_projection_contracts = tuple(
            projection_contract(layer, prefix) for prefix in ("w13", "w2")
        )
    if not layer.num_fused_shared_experts:
        return
    if not prepared:
        for name, (shape, dtype) in getattr(
            layer, "_npu_checkpoint_geometry", {}
        ).items():
            tensor = getattr(layer, name, None)
            if tensor is None or tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError(f"NPU shared checkpoint geometry changed for {name}")
    for prefix in ("w13", "w2"):
        weight = getattr(layer, f"{prefix}_weight", None)
        if weight is None or weight.ndim != 3:
            raise ValueError(f"NPU appended {prefix} requires a grouped weight tensor")
        for name, tensor in layer.named_parameters(recurse=False):
            if not name.startswith(prefix + "_"):
                continue
            if tensor.ndim == 0 or tensor.shape[0] != layer.num_local_experts:
                raise ValueError(
                    f"NPU appended expert metadata {name} has invalid expert count"
                )
            if tensor.device != weight.device:
                raise ValueError(
                    f"NPU appended expert metadata {name} is on the wrong device"
                )
            if not prepared:
                needed = {"w1", "w3"} if prefix == "w13" else {"w2"}
                loaded = getattr(tensor, "_npu_shared_loaded_shards", set())
                if not needed.issubset(loaded) and not (
                    prefix == "w13" and "w13" in loaded
                ):
                    raise ValueError(
                        f"NPU shared expert {name} is missing checkpoint shards {needed - loaded}"
                    )
        if prepared:
            kernel = getattr(layer, f"{prefix}_kernel")
            contract = moe_input_contract(kernel)
            if hasattr(layer, "hidden_size") and hasattr(
                layer, "intermediate_size_per_partition"
            ):
                from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
                    NPUMXFP8MoEMethod,
                    NPUUnquantMoEMethod,
                    NPUW4A4MXFP4MoEMethod,
                    NPUW4A8MXFP4MoEMethod,
                    NPUW8A8Int8MoEMethod,
                )

                h, i = layer.hidden_size, layer.intermediate_size_per_partition
                k, n = (h, 2 * i) if prefix == "w13" else (i, h)
                if isinstance(kernel, NPUUnquantMoEMethod):
                    shape = (layer.num_local_experts, n, k)
                elif isinstance(kernel, (NPUW4A4MXFP4MoEMethod, NPUW4A8MXFP4MoEMethod)):
                    shape = (layer.num_local_experts, k // 2, n)
                elif isinstance(kernel, (NPUMXFP8MoEMethod, NPUW8A8Int8MoEMethod)):
                    shape = (layer.num_local_experts, k, n)
                else:
                    shape = (layer.num_local_experts, k, n // 8)
                if tuple(weight.shape) != shape:
                    raise ValueError(
                        f"NPU appended {prefix} weight shape {tuple(weight.shape)} != {shape}"
                    )
                scale = getattr(layer, f"{prefix}_weight_scale", None)
                if contract is not None and contract.block_size:
                    if scale is None or tuple(scale.shape) != (
                        layer.num_local_experts,
                        k // 64,
                        n,
                        2,
                    ):
                        raise ValueError(
                            f"NPU appended {prefix} has incompatible block scale geometry"
                        )
            if contract is not None:
                scale = getattr(layer, f"{prefix}_weight_scale", None)
                if scale is None or scale.numel() == 0:
                    raise ValueError(
                        f"NPU appended {prefix} is missing quantization scales"
                    )
                if contract.block_size and scale.dtype not in (
                    torch.uint8,
                    getattr(torch, "float8_e8m0fnu", torch.uint8),
                ):
                    raise ValueError(f"NPU appended {prefix} requires E8M0 byte scales")
                if contract.block_size and (scale.ndim != 4 or scale.shape[-1] != 2):
                    raise ValueError(
                        f"NPU appended {prefix} requires paired block scales"
                    )
            if contract is not None and contract.kind in ("int4", "int8"):
                from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
                    NPUW8A8Int8MoEMethod,
                )

                expected = (
                    torch.int8
                    if isinstance(kernel, NPUW8A8Int8MoEMethod)
                    else torch.int32
                )
                if weight.dtype != expected:
                    raise ValueError(
                        f"NPU appended {prefix} requires prepared {expected} weights"
                    )
    if not prepared:
        # Consume this load generation before preparation replaces parameters.
        for name, tensor in layer.named_parameters(recurse=False):
            if name.startswith(("w13_", "w2_")):
                tensor._npu_shared_loaded_shards = set()
