"""Device checks for the A3 INT8 indexer activation/FP16-scale contract."""

from __future__ import annotations

import unittest

import torch

from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=30, suite="stage-a-unit-test-npu")
HAVE_NPU = is_npu() and torch.npu.is_available()


@unittest.skipUnless(HAVE_NPU, "Ascend NPU device required")
class TestInt8IndexerActivation(CustomTestCase):
    def setUp(self):
        from sglang.srt.layers.attention.dsa.dsa_npu_indexer import (
            create_npu_hadamard_128,
        )

        self.hadamard = create_npu_hadamard_128(128, "npu")

    def test_shapes_dtypes_and_empty_input(self):
        from sglang.srt.layers.attention.dsa.dsa_npu_indexer import (
            _quantize_npu_indexer_activation,
        )

        for shape in ((7, 128), (3, 4, 128), (0, 128)):
            with self.subTest(shape=shape):
                x = torch.randn(shape, dtype=torch.bfloat16, device="npu")
                q, scale = _quantize_npu_indexer_activation(
                    x, self.hadamard, torch.int8, torch.float16
                )
                self.assertEqual(q.shape, x.shape)
                self.assertEqual(q.dtype, torch.int8)
                self.assertEqual(scale.shape, x.shape[:-1])
                self.assertEqual(scale.dtype, torch.float16)

    def test_dequantized_values_track_rotated_input(self):
        from sglang.srt.layers.attention.dsa.dsa_npu_indexer import (
            _quantize_npu_indexer_activation,
        )

        x = torch.randn((64, 128), dtype=torch.bfloat16, device="npu")
        q, scale = _quantize_npu_indexer_activation(
            x, self.hadamard, torch.int8, torch.float16
        )
        reference = (x @ self.hadamard).float().cpu()
        restored = q.float().cpu() * scale.float().cpu().unsqueeze(-1)
        bound = reference.abs().amax(-1, keepdim=True) / 127 * 1.1
        self.assertTrue(torch.isfinite(restored).all().item())
        self.assertTrue(((restored - reference).abs() <= bound + 1e-5).all().item())


if __name__ == "__main__":
    unittest.main()
