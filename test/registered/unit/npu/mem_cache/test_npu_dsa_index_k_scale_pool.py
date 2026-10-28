"""A3 INT8 index-K pages and FP16 per-token scales round-trip without aliasing."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=10, suite="stage-a-unit-test-npu")
HAVE_NPU = is_npu() and torch.npu.is_available()
PAGE_SIZE, SIZE, LAYERS, WIDTH = 64, 256, 2, 128


def _pool(hicache=False):
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool

    with (
        envs.SGLANG_NPU_INDEXER_INT8.override(True),
        get_parallel().override(attn_dcp_size=1, attn_dcp_rank=0, dcp_enabled=False),
        patch(
            "sglang.srt.hardware_backend.npu.memory_pool_npu.get_memory",
            return_value=SimpleNamespace(enable_hierarchical_cache=hicache),
        ),
    ):
        return NPUMLATokenToKVPool(
            size=SIZE,
            page_size=PAGE_SIZE,
            dtype=torch.bfloat16,
            kv_lora_rank=128,
            qk_rope_head_dim=64,
            layer_num=LAYERS,
            device="npu",
            enable_memory_saver=False,
            index_head_dim=WIDTH,
        )


@unittest.skipUnless(HAVE_NPU, "Ascend NPU device required")
class TestNPUIndexKScaleBuffer(CustomTestCase):
    def test_int8_keys_and_fp16_scales_cross_page_boundaries(self):
        pool = _pool()
        loc = torch.tensor(
            [PAGE_SIZE - 1, PAGE_SIZE, SIZE - 1], dtype=torch.int64, device="npu"
        )
        keys = (
            torch.arange(3 * WIDTH, dtype=torch.int64, device="npu")
            .remainder(127)
            .to(torch.int8)
            .view(3, WIDTH)
        )
        scales = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float16, device="npu")
        pool.set_index_k_buffer(0, loc, keys)
        pool.set_index_k_scale_buffer(0, loc, scales)
        actual_keys = pool.get_index_k_buffer(0).view(-1, WIDTH).index_select(0, loc)
        actual_scales = (
            pool.get_index_k_scale_buffer(0).reshape(-1).index_select(0, loc)
        )
        self.assertEqual(actual_keys.dtype, torch.int8)
        self.assertEqual(actual_scales.dtype, torch.float16)
        self.assertTrue(torch.equal(actual_keys, keys))
        self.assertTrue(torch.equal(actual_scales, scales))
        self.assertTrue(bool((pool.get_index_k_buffer(1) == 0).all()))
        self.assertTrue(bool((pool.get_index_k_scale_buffer(1) == 0).all()))

    def test_hicache_keeps_the_supported_float_storage(self):
        pool = _pool(hicache=True)
        self.assertFalse(pool.indexer_kv_store_int8)
        self.assertEqual(pool.index_k_store_dtype, torch.bfloat16)
        self.assertIsNone(pool.index_k_scale_buffer)


if __name__ == "__main__":
    unittest.main()
