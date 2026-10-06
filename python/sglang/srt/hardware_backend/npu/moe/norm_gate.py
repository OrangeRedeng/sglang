"""Opt-in native MXFP8 input and router for GLM-5.2 TP MoE."""

import logging
from functools import lru_cache

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.moe.tp_fusion import (
    mxfp8_input,
    record_mxfp8_operand,
    require_npu_op,
    supports_mxfp8_linear,
)
from sglang.srt.hardware_backend.npu.quantization.moe_methods import _require_e8m0_dtype
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.quantization.base_config import QuantizeMethodBase

logger = logging.getLogger(__name__)


def _gate_option(name, choices):
    value = getattr(envs, name).get()
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}; got {value!r}")
    return value


def _tensor_layout(tensor):
    import torch_npu

    return (
        f"dtype={tensor.dtype} shape={tuple(tensor.shape)} stride={tensor.stride()} "
        f"contiguous={tensor.is_contiguous()} format={torch_npu.get_npu_format(tensor)}"
    )


@lru_cache(maxsize=None)
def _log_gate_config(output_dtype, weight_layout, topk_layout, scale_alg):
    logger.info(
        "MXFP8 gate: output_dtype=%s weight_layout=%s topk_layout=%s "
        "weight_scale_alg=%s; grouped TopK uses FP32 logits/bias",
        output_dtype,
        weight_layout,
        topk_layout,
        scale_alg,
    )


@lru_cache(maxsize=None)
def _log_gate_output(output_dtype):
    logger.info("MXFP8 gate actual QuantMatmul output_dtype=%s", output_dtype)


@lru_cache(maxsize=None)
def native_norm_op():
    op = require_npu_op(
        "npu_add_rms_norm_dynamic_mx_quant",
        ("beta", "epsilon", "scale_alg", "round_mode", "dst_type"),
    )
    if len(op.default._schema.returns) != 4:
        raise RuntimeError(f"Unsupported native MX norm ABI: {op.default._schema}")
    return op


@lru_cache(maxsize=None)
def mxfp8_gate_op():
    _require_e8m0_dtype()
    return require_npu_op(
        "npu_quant_matmul",
        (
            "pertoken_scale",
            "output_dtype",
            "scale_dtype",
            "pertoken_scale_dtype",
            "group_sizes",
        ),
    )


def mx_scale_layout(scale, rows, hidden_size):
    if scale.dtype not in (torch.uint8, getattr(torch, "float8_e8m0fnu", None)):
        raise ValueError(f"Expected E8M0 scale bytes, got {scale.dtype}")
    # The existing W4A8 TP consumers require adjacent pairs of 32-value scales.
    return scale.view(rows, hidden_size // 64, 2)


def native_add_rmsnorm_mxfp8(x, residual, gamma, eps):
    op = native_norm_op()
    if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != 6144:
        raise ValueError("Native MoE norm requires BF16 [T,6144]")
    if residual.shape != x.shape or residual.dtype != x.dtype:
        raise ValueError("Native MoE norm requires a matching BF16 residual")
    q, residual_out, scale, rstd = op(
        x,
        residual,
        gamma,
        beta=None,
        epsilon=eps,
        scale_alg=0,
        round_mode="rint",
        dst_type=torch.float8_e4m3fn,
    )
    if (
        q.dtype != torch.float8_e4m3fn
        or q.shape != x.shape
        or residual_out.dtype != torch.bfloat16
        or residual_out.shape != x.shape
        or rstd.dtype != torch.float32
    ):
        raise RuntimeError("Unsupported native MX norm output order/dtypes/shapes")
    return q, residual_out, mx_scale_layout(scale, x.shape[0], x.shape[1]), rstd


class NativeMXFP8MoENorm(RMSNorm):
    def __init__(self, hidden_size, eps):
        native_norm_op()
        super().__init__(hidden_size, eps=eps)

    def forward_npu(
        self, x, residual=None, post_residual_addition=None, quant_linear=None
    ):
        if post_residual_addition is not None or quant_linear is not None:
            raise ValueError("Native MoE norm does not support extra fused branches")
        residual_input = torch.zeros_like(x) if residual is None else residual
        q, residual_out, scale, _ = native_add_rmsnorm_mxfp8(
            x, residual_input, self.weight.data, self.variance_epsilon
        )
        # Keep the scale on the FP8 tensor without a tensor -> tuple -> tensor cycle.
        q._npu_mxfp8_scale = scale
        return q if residual is None else (q, residual_out)

    # This experiment must not select a backend that emits normalized BF16.
    forward = forward_npu


def _mxfp8_gate_logits_dtype():
    mode = envs.SGLANG_NPU_TP_MOE_MXFP8_GATE_LOGITS_DTYPE.get()
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp32":
        return torch.float32
    raise ValueError(
        f"SGLANG_NPU_TP_MOE_MXFP8_GATE_LOGITS_DTYPE must be bf16 or fp32; got {mode!r}"
    )


class MXFP8GateMethod(QuantizeMethodBase):
    def __init__(self):
        self.output_dtype = _mxfp8_gate_logits_dtype()
        self.weight_layout = _gate_option(
            "SGLANG_NPU_TP_MOE_MXFP8_GATE_WEIGHT_LAYOUT",
            ("transposed", "contiguous", "nz"),
        )
        self.topk_layout = _gate_option(
            "SGLANG_NPU_TP_MOE_MXFP8_GATE_TOPK_LAYOUT",
            ("default", "contiguous", "clone", "nd"),
        )
        self.scale_alg = _gate_option("SGLANG_NPU_TP_MOE_MXFP8_GATE_SCALE_ALG", (0, 1))
        self.diagnostics = envs.SGLANG_NPU_TP_MOE_MXFP8_GATE_DIAGNOSTICS.get()
        self._topk_layout_logged = set()
        self.matmul = mxfp8_gate_op()
        self.quantize = require_npu_op(
            "npu_dynamic_mx_quant",
            ("dst_type", "block_size", "scale_alg", "round_mode"),
        )
        self.format_cast = (
            require_npu_op("npu_format_cast")
            if self.weight_layout == "nz" or self.topk_layout == "nd"
            else None
        )
        if (
            self.weight_layout == "nz"
            and envs.SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT.get()
        ):
            raise ValueError("MXFP8 gate NZ conflicts with DISABLE_ACL_FORMAT_WEIGHT=1")
        _log_gate_config(
            self.output_dtype, self.weight_layout, self.topk_layout, self.scale_alg
        )

    @torch.no_grad()
    def process_weights_after_loading(self, layer):
        if layer.weight.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ) or layer.weight.shape != (256, 6144):
            raise ValueError(
                "GLM-5.2 MXFP8 gate requires FP16/BF16/FP32 [256,6144] weights; "
                f"got {layer.weight.dtype} {tuple(layer.weight.shape)}"
            )
        # Retain the loaded gate dtype/values for the production reference.
        q, scale = self.quantize(
            layer.weight.data,
            dst_type=torch.float8_e4m3fn,
            block_size=32,
            scale_alg=self.scale_alg,
            round_mode="rint",
        )
        if q.dtype != torch.float8_e4m3fn or q.shape != layer.weight.shape:
            raise RuntimeError("Unsupported native gate MXFP8 quantization output")
        weight = q.transpose(0, 1)
        weight_scale = mx_scale_layout(scale, 256, 6144).transpose(0, 1)
        if self.weight_layout != "transposed":
            # Match the scale's reduction-axis layout to the persistent weight.
            weight = weight.contiguous()
            weight_scale = weight_scale.contiguous()
        if self.weight_layout == "nz":
            weight = self.format_cast(weight, 29)
        layer.register_buffer("mxfp8_weight", weight, persistent=False)
        layer.register_buffer("mxfp8_weight_scale", weight_scale, persistent=False)
        if self.diagnostics:
            logger.info("MXFP8 gate weight: %s", _tensor_layout(weight))
            logger.info("MXFP8 gate weight scale: %s", _tensor_layout(weight_scale))

    def prepare_topk_logits(self, logits):
        # Preserve the FP32 correction-bias routing boundary before changing layout.
        topk_logits = logits.to(torch.float32)
        if self.topk_layout == "contiguous":
            topk_logits = topk_logits.contiguous()
        elif self.topk_layout == "clone":
            topk_logits = topk_logits.clone(memory_format=torch.contiguous_format)
        elif self.topk_layout == "nd":
            topk_logits = self.format_cast(topk_logits, 2)
        if self.diagnostics:
            key = (tuple(logits.shape), logits.dtype, logits.stride())
            if key not in self._topk_layout_logged:
                logger.info("MXFP8 gate output: %s", _tensor_layout(logits))
                logger.info("MXFP8 grouped TopK input: %s", _tensor_layout(topk_logits))
                self._topk_layout_logged.add(key)
        return topk_logits

    def apply(self, layer, hidden_states, *, output_dtype=None):
        if output_dtype is None:
            output_dtype = self.output_dtype
        if output_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("MXFP8 gate supports BF16 or FP32 logits")
        if not hasattr(layer, "mxfp8_weight"):
            raise RuntimeError("MXFP8 gate weights were not prepared after loading")
        operand = (
            hidden_states
            if isinstance(hidden_states, tuple)
            else mxfp8_input(hidden_states)
        )
        q, scale = operand
        if q.dtype != torch.float8_e4m3fn or q.ndim != 2 or q.shape[1] != 6144:
            raise ValueError("MXFP8 gate requires E4M3 [T,6144]")
        record_mxfp8_operand(operand)
        logits = self.matmul(
            q,
            layer.mxfp8_weight,
            layer.mxfp8_weight_scale,
            pertoken_scale=mx_scale_layout(scale, q.shape[0], q.shape[1]),
            scale_dtype=_require_e8m0_dtype(),
            pertoken_scale_dtype=_require_e8m0_dtype(),
            output_dtype=output_dtype,
            group_sizes=[1, 1, 32],
        )
        if logits.dtype != output_dtype:
            raise RuntimeError(f"MXFP8 gate did not return {output_dtype} logits")
        _log_gate_output(logits.dtype)
        if self.topk_layout != "default" or self.diagnostics:
            logits._npu_mxfp8_gate_method = self
        return logits


def configure_native_norm_gate(moe):
    from sglang.srt.configs.model_config import is_glm_moe_dsa
    from sglang.srt.environ import envs
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        NPUW4A8MXFP4MoEMethod,
    )
    from sglang.srt.layers.moe.token_dispatcher.ascend_tp import AscendTPDispatcher
    from sglang.srt.layers.moe.utils import is_sbo_enabled, is_tbo_enabled
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if not (
        is_glm_moe_dsa(moe.config)
        and moe.config.hidden_size == 6144
        and moe.config.n_routed_experts == 256
        and moe.config.num_experts_per_tok == 8
        and moe.tp_size == 4
        and moe.moe_ep_size == 1
        and parallel.moe_dp_size == 1
        and parallel.attn_dp_size == 1
        and parallel.attn_cp_size == 1
        and not parallel.enable_prefill_cp
        and not (is_sbo_enabled() or is_tbo_enabled())
        and not moe._enable_a2a_moe
        and not moe._shared_expert_tp1
        and not moe._fuse_shared_experts_inside_sbo
        and moe.num_fused_shared_experts == 0
        and isinstance(moe.experts.dispatcher, AscendTPDispatcher)
        and not moe.experts.dispatcher.local_ep
        and isinstance(getattr(moe.experts, "w2_kernel", None), NPUW4A8MXFP4MoEMethod)
        and envs.SGLANG_NPU_TP_MOE_PREQUANT_INPUT.get()
        and envs.SGLANG_NPU_TP_MOE_REUSE_MXFP8.get()
        and hasattr(moe, "shared_experts")
        and supports_mxfp8_linear(moe.shared_experts.gate_up_proj)
        and supports_mxfp8_linear(moe.shared_experts.down_proj)
    ):
        raise ValueError(
            "Native norm/gate requires GLM-5.2 W4A8 MXFP TP4/EP1 prequant/reuse"
        )
    if (
        envs.SGLANG_NPU_TP_MOE_NORM_MXFP8.get()
        or envs.SGLANG_NPU_TP_MOE_FUSE_GMM2_FINALIZE.get()
        or envs.SGLANG_NPU_TP_MOE_FUSED_SHARED_EXPERT.get()
    ):
        raise ValueError(
            "Keep NORM_MXFP8, FUSE_GMM2_FINALIZE and FUSED_SHARED_EXPERT off"
        )
    if envs.SGLANG_NPU_TP_MOE_NATIVE_NORM_MXFP8.get():
        if not envs.SGLANG_NPU_TP_MOE_MXFP8_GATE.get():
            raise ValueError("Native MXFP8 norm requires MXFP8_GATE=1")
        native_norm_op()
    moe.gate.quant_method = MXFP8GateMethod()
    moe.gate._npu_mxfp8_gate = True


def install_input_capture(norm, moe):
    """Capture one real prefill per sparse layer/rank, outside timed serving runs."""
    from pathlib import Path

    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_exec, get_parallel

    if (
        envs.SGLANG_NPU_TP_MOE_NATIVE_NORM_MXFP8.get()
        or envs.SGLANG_NPU_TP_MOE_MXFP8_GATE.get()
    ):
        raise ValueError("Capture norm/gate inputs with both experiments disabled")
    directory = Path(envs.SGLANG_NPU_TP_MOE_NORM_GATE_CAPTURE_DIR.get())

    def capture(module, args, kwargs):
        x = args[0] if args else kwargs["x"]
        if x.shape[0] < envs.SGLANG_NPU_TP_MOE_NORM_GATE_CAPTURE_MIN_TOKENS.get():
            return
        residual = args[1] if len(args) > 1 else kwargs.get("residual")
        extra = args[2] if len(args) > 2 else kwargs.get("post_residual_addition")
        if extra is not None:
            raise ValueError("Capture requires the plain sparse-MoE norm boundary")
        cfg = moe.topk.topk_config
        data = {
            "x": x.detach().cpu(),
            "residual": None if residual is None else residual.detach().cpu(),
            "gamma": module.weight.detach().cpu(),
            "eps": module.variance_epsilon,
            "gate_weight": moe.gate.weight.detach().cpu(),
            "correction_bias": None
            if cfg.correction_bias is None
            else cfg.correction_bias.detach().cpu(),
            "tp_size": moe.tp_size,
            "layer_id": moe.layer_id,
            "deterministic": get_exec().deterministic.enable_deterministic_inference,
            "topk": {
                key: getattr(cfg, key)
                for key in (
                    "top_k",
                    "renormalize",
                    "use_grouped_topk",
                    "num_expert_group",
                    "topk_group",
                    "scoring_func",
                    "routed_scaling_factor",
                    "apply_routed_scaling_factor_on_output",
                )
            },
        }
        directory.mkdir(parents=True, exist_ok=True)
        path = (
            directory
            / f"rank{get_parallel().tp_rank}-layer{moe.layer_id}-tokens{x.shape[0]}.pt"
        )
        torch.save(data, path)
        handle.remove()

    handle = norm.register_forward_pre_hook(capture, with_kwargs=True)
