"""A3 INT8 index-K pages and FP16 per-token scales round-trip without aliasing."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_cpu_ci, register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=10, suite="stage-a-unit-test-npu")
register_cpu_ci(est_time=1, suite="base-a-test-cpu")
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


class TestIndexWriteDtypeContract(CustomTestCase):
    def test_integer_codes_survive_the_write_and_float_inputs_still_cast(self):
        source = (
            Path(__file__).resolve().parents[5]
            / "python/sglang/srt/hardware_backend/npu/memory_pool_npu.py"
        )
        tree = ast.parse(source.read_text())
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "set_index_k_buffer"
        )

        def scatter(target, indices, values):
            target.index_copy_(0, indices.reshape(-1).long(), values)

        namespace = dict(
            torch=torch,
            torch_npu=SimpleNamespace(npu_scatter_nd_update_=scatter),
            maybe_detect_oob=lambda *a: None,
        )
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        loc = torch.tensor([1, 3, 6])
        for quantized in (True, False):
            dtype = torch.int8 if quantized else torch.bfloat16
            pool = SimpleNamespace(
                dtype=torch.bfloat16,
                indexer_kv_store_int8=quantized,
                index_k_store_dtype=dtype,
                index_head_dim=4,
                index_size=8,
                index_page_size=1,
                index_page_padding=1,
                indexer_layer_id_to_slot={0: 0},
                index_k_buffer=torch.zeros((1, 8, 1, 4), dtype=dtype),
                _get_indexer_slot=lambda layer: 0,
            )
            values = torch.tensor(
                [[-128, -1, 0, 127], [-3, 3, 32, -32], [1, 2, 3, 4]],
                dtype=torch.int8 if quantized else torch.float32,
            )
            with self.subTest(quantized=quantized):
                namespace[method.name](pool, 0, loc, values)
                actual = pool.index_k_buffer[0].index_select(0, loc).reshape(3, 4)
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(actual, values.to(dtype), rtol=0, atol=0)
                self.assertTrue(bool((pool.index_k_buffer[0, 0] == 0).all()))


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
