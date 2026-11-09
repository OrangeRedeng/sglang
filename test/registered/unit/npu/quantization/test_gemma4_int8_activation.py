"""Gemma requests GELU even when its expert weights use INT8 quantization."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.moe.activation import NPUGeluAndMul, NPUSwigluQuant
from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUW4A8Int8MoEMethod,
    NPUW8A8Int8MoEMethod,
)
from sglang.srt.layers.moe.moe_runner import ascend as mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestGemma4Int8Activation(CustomTestCase):
    def make_runner(self, kernel_type, activation):
        # No weights or device are needed for activation selection.
        kernel = kernel_type.__new__(kernel_type)
        config = SimpleNamespace(
            layer=SimpleNamespace(w2_kernel=kernel),
            activation=activation,
            swiglu_limit=None,
            gemm1_alpha=None,
            gemm1_clamp_limit=None,
        )
        with patch.object(
            mod, "get_moe_a2a_backend",
            return_value=SimpleNamespace(is_deepep=lambda: False),
        ):
            return mod.AscendRunnerCore(config)

    def test_gemma_gelu_is_preserved_for_int8_experts(self):
        x = torch.tensor([[-2.0, 0.7, 1.5, -0.8], [0.3, -1.2, 2.0, 0.4]])
        expected = F.gelu(x[:, :2], approximate="tanh") * x[:, 2:]
        for kernel in (NPUW4A8Int8MoEMethod, NPUW8A8Int8MoEMethod):
            with self.subTest(kernel=kernel.__name__):
                runner = self.make_runner(kernel, "gelu")
                self.assertIsInstance(runner.activation, NPUGeluAndMul)
                output, scale = runner.activation._apply_activation(x)
                torch.testing.assert_close(output, expected)
                self.assertIsNone(scale)

    def test_silu_keeps_existing_fused_int8_activation(self):
        for kernel in (NPUW4A8Int8MoEMethod, NPUW8A8Int8MoEMethod):
            with self.subTest(kernel=kernel.__name__):
                runner = self.make_runner(kernel, "silu")
                self.assertIsInstance(runner.activation, NPUSwigluQuant)


if __name__ == "__main__":
    unittest.main()
