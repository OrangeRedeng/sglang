"""CPU-safe tests for Ascend fused norm+quant and offline router loading.

No torch_npu install or NPU device is required: operator probes are mocked
and only pure selection logic runs.
"""

import unittest
from unittest.mock import patch

import torch

import sglang.srt.layers.quantization  # noqa: F401  (canonical import entry)
from sglang.srt.hardware_backend.npu.layernorm_quant import (
    NormQuantizedReadout,
    can_fuse_norm_quant,
)
from sglang.srt.hardware_backend.npu.moe.contracts import INT8, MXFP8
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNormQuantSelection(unittest.TestCase):
    def test_no_consumers(self):
        contract, reason = can_fuse_norm_quant([])
        self.assertIsNone(contract)
        self.assertIn("no quantized consumer", reason)

    def test_float_consumer_declines(self):
        contract, reason = can_fuse_norm_quant([MXFP8, None])
        self.assertIsNone(contract)
        self.assertIn("floating-point", reason)

    def test_disagreeing_consumers_decline(self):
        contract, reason = can_fuse_norm_quant([MXFP8, INT8])
        self.assertIsNone(contract)
        self.assertIn("disagree", reason)

    def test_probe_gate(self):
        # All consumers agree on MXFP8 but the device probe fails closed on
        # CPU (no NPU, no fused ABI): the selection must decline, not guess.
        contract, reason = can_fuse_norm_quant([MXFP8, MXFP8])
        self.assertIsNone(contract)
        self.assertIn("cannot emit", reason)

    def test_readout_dataclass_is_frozen(self):
        operand = torch.zeros(2, 64, dtype=torch.float8_e4m3fn)
        scale = torch.zeros(2, 1, 2, dtype=torch.uint8)
        from sglang.srt.hardware_backend.npu.moe.contracts import QuantizedOperand

        q = QuantizedOperand(operand, scale, MXFP8, (2, 64))
        residual = torch.zeros(2, 64)
        readout = NormQuantizedReadout(q, residual)
        self.assertIs(readout.normalized_float, None)
        self.assertIs(readout.operand, q)
        self.assertIs(readout.updated_residual, residual)


class TestNpuNormQuantKernelDispatch(unittest.TestCase):
    def test_npu_format_selects_npu_kernel(self):
        from sglang.srt.layers.layer_boundary.residual.add_norm import (
            _norm_quant_kernel,
            _update_and_read_residual_npu,
        )

        for fmt in ("npu_mxfp8", "npu_mxfp4", "npu_int8", "npu_int4"):
            kernel = _norm_quant_kernel(fmt)
            self.assertIs(kernel.func, _update_and_read_residual_npu)
            self.assertEqual(kernel.keywords, {"quant_format": fmt})

    def test_non_npu_format_unchanged(self):
        from sglang.srt.layers.layer_boundary.residual.add_norm import (
            _norm_quant_kernel,
            _update_and_read_residual_plain,
        )

        self.assertIs(_norm_quant_kernel(""), _update_and_read_residual_plain)


class TestBoundaryOperandMoeForward(unittest.TestCase):
    """The GLM MoE block routes a boundary QuantizedOperand to the fused path
    without touching the tensor paths (CPU, dispatcher mocked)."""

    def _make_block(self):
        from sglang.srt.models import glm4_moe

        block = glm4_moe.Glm4MoeSparseMoeBlock.__new__(glm4_moe.Glm4MoeSparseMoeBlock)
        torch.nn.Module.__init__(block)
        block._enable_a2a_moe = False
        block._fuse_npu_tp_shared = False
        block._fuse_npu_tp_routed_scale = False
        block.routed_scaling_factor = 2.5
        block.gate = object()
        block.topk = None
        block.experts = None
        return block

    def test_operand_routes_to_quantized_path(self):
        block = self._make_block()
        from sglang.srt.hardware_backend.npu.moe.contracts import QuantizedOperand
        from sglang.srt.models import glm4_moe

        operand = QuantizedOperand(
            torch.zeros(2, 64, dtype=torch.float8_e4m3fn),
            torch.zeros(2, 1, 2, dtype=torch.uint8),
            MXFP8,
            (2, 64),
        )
        with (
            patch.object(glm4_moe, "_is_npu", True),
            patch.object(
                block, "forward_normal_quantized", return_value="out"
            ) as fused,
        ):
            result = block.forward(operand)
        fused.assert_called_once_with(operand)
        self.assertEqual(result, "out")


class TestFusedNormABI(unittest.TestCase):
    def test_integer_five_outputs_and_int4_dtype(self):
        from sglang.srt.hardware_backend.npu import layernorm_quant as nq
        from sglang.srt.hardware_backend.npu.moe.contracts import INT4

        x = torch.zeros(2, 64, dtype=torch.bfloat16)
        residual = torch.ones_like(x)
        q = torch.zeros(2, 8, dtype=torch.int32)
        scale = torch.ones(2, dtype=torch.float32)
        with patch.object(nq, "native_int_norm_op") as factory:
            factory.return_value.return_value = (q, None, residual, scale, None)
            readout = nq.native_add_rmsnorm_quant(x, residual, x[0], 1e-6, INT4)
        self.assertIs(readout.operand.data, q)
        self.assertIs(readout.updated_residual, residual)
        self.assertEqual(
            factory.return_value.call_args.kwargs["y_dtype"], torch.quint4x2
        )
        self.assertEqual(
            factory.return_value.call_args.kwargs["output_mask"], [True, False]
        )

    def test_residual_free_mx_uses_three_output_operator(self):
        from sglang.srt.hardware_backend.npu import layernorm_quant as nq

        x = torch.zeros(2, 64, dtype=torch.bfloat16)
        q = torch.zeros_like(x, dtype=torch.float8_e4m3fn)
        scale = torch.zeros(2, 1, 2, dtype=torch.uint8)
        with patch.object(nq, "native_norm_no_residual_op") as factory:
            factory.return_value.return_value = (q, scale, None)
            readout = nq.native_add_rmsnorm_quant(x, None, x[0], 1e-6, MXFP8)
        self.assertIs(readout.operand.data, q)
        self.assertIsNone(readout.updated_residual)

    def test_boundary_keeps_type_effective_gamma_and_add_order(self):
        from types import SimpleNamespace

        from sglang.srt.hardware_backend.npu import layernorm_quant as nq
        from sglang.srt.hardware_backend.npu.moe.contracts import QuantizedOperand
        from sglang.srt.layers.layer_boundary.residual.add_norm import (
            _update_and_read_residual_npu,
        )

        x = torch.zeros(2, 64, dtype=torch.bfloat16)
        residual = torch.ones_like(x)
        post = torch.full_like(x, 2)
        norm = SimpleNamespace(
            weight=x[0], gemma_weight=residual[0], variance_epsilon=1e-6
        )
        operand = QuantizedOperand(
            torch.zeros_like(x, dtype=torch.int8), torch.ones(2), INT8, (2, 64)
        )
        with patch.object(
            nq,
            "native_add_rmsnorm_quant",
            return_value=nq.NormQuantizedReadout(operand, post),
        ) as fused:
            output, updated = _update_and_read_residual_npu(
                "npu_int8", norm, x, residual, post
            )
        self.assertIs(output, operand)
        self.assertIs(updated, post)
        torch.testing.assert_close(fused.call_args.args[2], norm.gemma_weight)
        torch.testing.assert_close(fused.call_args.args[1], residual + post)


class TestOfflineRouterLoading(unittest.TestCase):
    def test_custom_gate_uses_checkpoint_scheme_and_preserves_bias(self):
        from types import SimpleNamespace
        from unittest.mock import Mock

        from sglang.srt.hardware_backend.npu.moe.norm_gate import (
            configure_offline_gate,
            gate_input_contract,
        )
        from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
            NPUW8A8Int8DynamicLinearMethod,
        )
        from sglang.srt.layers.quantization.modelslim.modelslim import (
            ModelSlimLinearMethod,
        )

        gate = torch.nn.Module()
        gate.weight = torch.nn.Parameter(torch.empty(4, 64))
        gate.e_score_correction_bias = torch.nn.Parameter(
            torch.arange(4, dtype=torch.float32)
        )
        correction = gate.e_score_correction_bias
        config = Mock()
        config.get_name.return_value = "modelslim"
        config.resolve_projection_scheme.return_value = "W8A8_DYNAMIC"
        config.linear_scheme_types.return_value = {"W8A8_DYNAMIC": object}
        config.get_linear_scheme.return_value = SimpleNamespace(
            kernel=NPUW8A8Int8DynamicLinearMethod()
        )
        with patch.object(ModelSlimLinearMethod, "create_weights") as create:
            self.assertFalse(
                configure_offline_gate(gate, "model.layers.0.mlp.gate", config)
            )
        create.assert_called_once()
        self.assertIs(gate.e_score_correction_bias, correction)
        self.assertEqual(gate_input_contract(gate), INT8)

    def test_descriptor_resolution_is_per_layer(self):
        from unittest.mock import Mock

        from sglang.srt.hardware_backend.npu.moe.norm_gate import offline_gate_config

        config = Mock()
        config.get_name.return_value = "modelslim"
        config.resolve_projection_scheme.side_effect = lambda prefix: (
            "FLOAT" if ".0." in prefix else "W8A8_MXFP8"
        )
        config.linear_scheme_types.return_value = {"W8A8_MXFP8": object}
        self.assertIsNone(offline_gate_config(config, "model.layers.0.mlp.gate"))
        self.assertIs(offline_gate_config(config, "model.layers.1.mlp.gate"), config)


if __name__ == "__main__":
    unittest.main()
