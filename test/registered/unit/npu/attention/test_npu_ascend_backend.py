"""
Unit tests for sglang.srt.hardware_backend.npu.attention.ascend_backend.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase
from torch.utils._python_dispatch import TorchDispatchMode

register_npu_ci(est_time=5, suite="base-a-test-npu")

# Mock NPU-only modules before importing the source module.
for _ in (
    "torch_npu",
    "torch_npu.contrib",
    "sgl_kernel_npu",
    "sgl_kernel_npu.attention",
    "sgl_kernel_npu.attention.sinks_attention",
    "sglang.srt.speculative",
    "sglang.srt.speculative.decoupled_spec_io",
    "sglang.srt.speculative.spec_info",
    "sglang.srt.speculative.eagle_info",
):
    sys.modules.setdefault(_, MagicMock())

from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
    AscendAttnBackend,
    AscendAttnMaskBuilder,
    _expand_dsa_sparse_indices,
    _reshape_kv_for_fia_nz,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class TestFrozenKvSwaMetadata(CustomTestCase):
    def test_assistant_reads_target_swa_pages_instead_of_full_pages(self):
        """A dummy assistant pool must not hide the target's SWA page mapping."""
        target_pool = MagicMock(spec=SWAKVPool)
        target_pool.swa_layer_nums = 3
        target_pool.full_to_swa_index_mapping = torch.arange(16) + 16
        for frozen in (False, True):
            with self.subTest(frozen=frozen):
                backend = object.__new__(AscendAttnBackend)
                backend.reuse_fia_metadata = False
                backend.reuse_fia_kv_views = False
                backend.token_to_kv_pool = object()
                backend.page_size = 4
                backend.use_mla = False
                backend.device = "cpu"
                backend.req_to_token_pool = SimpleNamespace(
                    req_to_token=torch.arange(16).reshape(2, 8)
                )
                runner = SimpleNamespace(
                    is_draft_worker=True,
                    is_hybrid_swa=False,
                    sliding_window_size=3,
                    spec_algorithm=SimpleNamespace(
                        is_frozen_kv_mtp=MagicMock(return_value=frozen)
                    ),
                    token_to_kv_pool_allocator=SimpleNamespace(
                        get_kvcache=lambda: target_pool
                    ),
                )
                backend._init_swa_metadata(runner)
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.DECODE,
                    seq_lens=torch.tensor([7, 8]),
                    seq_lens_cpu=torch.tensor([7, 8]),
                    spec_info=None,
                    req_pool_indices=torch.arange(2),
                    extend_seq_lens=None,
                    extend_seq_lens_cpu=None,
                    out_cache_loc=None,
                )
                with patch(
                    "sglang.srt.hardware_backend.npu.attention.ascend_backend.get_parallel",
                    return_value=SimpleNamespace(dcp_enabled=False),
                ):
                    backend.init_forward_metadata(batch)
                self.assertEqual(backend.is_hybrid_swa, frozen)
                self.assertEqual(
                    backend._is_swa_layer(SimpleNamespace(sliding_window_size=3)),
                    frozen,
                )
                self.assertEqual(
                    backend.forward_metadata.block_tables.tolist(), [[0, 1], [2, 3]]
                )
                if frozen:
                    self.assertEqual(
                        backend.forward_metadata.block_tables_swa.tolist(),
                        [[4, 5], [6, 7]],
                    )
                else:
                    self.assertIsNone(backend.forward_metadata.block_tables_swa)


class TestFiaLengthMetadata(unittest.TestCase):
    def test_refreshes_plain_lengths_once_per_batch(self):
        for reuse in (False, True):
            backend = object.__new__(AscendAttnBackend)
            backend.reuse_fia_metadata = reuse
            backend.reuse_fia_kv_views = False
            backend.page_size = 128
            backend.use_mla = False
            backend.use_sliding_window_kv_pool = False
            backend.req_to_token_pool = SimpleNamespace(
                req_to_token=torch.arange(3 * 512).reshape(3, 512)
            )
            for hybrid in (False, True):
                backend.is_hybrid_swa = hybrid
                backend.full_to_swa_index_mapping = torch.arange(3 * 512)
                for lengths, extends in (
                    ([3, 4, 5], [2, 0, 1]),
                    ([9, 10, 11], [0, 3, 2]),
                ):
                    with self.subTest(hybrid=hybrid, lengths=lengths):
                        batch = SimpleNamespace(
                            forward_mode=ForwardMode.EXTEND,
                            seq_lens=torch.tensor(lengths),
                            seq_lens_cpu=torch.tensor(lengths),
                            spec_info=None,
                            req_pool_indices=torch.arange(3),
                            extend_seq_lens=torch.tensor(extends),
                            extend_seq_lens_cpu=extends,
                        )
                        with patch(
                            "sglang.srt.hardware_backend.npu.attention.ascend_backend.get_parallel",
                            return_value=SimpleNamespace(dcp_enabled=False),
                        ):
                            backend.init_forward_metadata(batch)
                        self.assertEqual(
                            backend.forward_metadata.seq_lens_cpu_list,
                            lengths if reuse else None,
                        )
                        expected = [sum(extends[: i + 1]) for i in range(len(extends))]
                        self.assertEqual(
                            list(backend.forward_metadata.seq_lens_list_cumsum),
                            expected,
                        )
                        self.assertEqual(
                            all(
                                type(v) is int
                                for v in backend.forward_metadata.seq_lens_list_cumsum
                            ),
                            reuse,
                        )

    def test_tnd_adapters_do_not_extract_scalars_per_layer(self):
        for reuse in (False, True):

            class CountScalars(TorchDispatchMode):
                count = 0

                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    if func == torch.ops.aten._local_scalar_dense.default:
                        self.count += 1
                    return func(*args, **(kwargs or {}))

            backend = object.__new__(AscendAttnBackend)
            backend.reuse_fia_metadata = reuse
            backend.reuse_fia_kv_views = False
            backend.use_mla = False
            backend.use_fia = True
            backend.is_dllm_model = False
            backend.attn_cp_size = 1
            backend.page_size = 128
            backend.fia_mask = torch.ones(8, 8, dtype=torch.bool).triu_(1)
            cache = torch.zeros(2, 128, 256)
            backend.token_to_kv_pool = SimpleNamespace(
                get_key_buffer=lambda _, cache=cache: cache,
                get_value_buffer=lambda _, cache=cache: cache,
            )
            batch = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                global_num_token_non_padded_cpu=3,
                attn_cp_metadata=None,
            )
            layer = SimpleNamespace(
                layer_id=0,
                sliding_window_size=2,
                qk_head_dim=256,
                v_head_dim=256,
                tp_q_head_num=1,
                tp_k_head_num=1,
                tp_v_head_num=1,
                scaling=256**-0.5,
            )
            q = torch.randn(3, 256)
            backend.forward_metadata = SimpleNamespace(
                seq_lens_list_cumsum=[2, 2, 3],
                seq_lens_cpu_list=[3, 4, 5],
                seq_lens_cpu_int=torch.tensor([3, 4, 5]),
                block_tables=torch.zeros(3, 1, dtype=torch.int32),
                block_tables_swa=torch.zeros(3, 1, dtype=torch.int32),
            )

            def fia(**kwargs):
                lengths = kwargs.get(
                    "actual_seq_kvlen", kwargs.get("actual_seq_lengths_kv")
                )
                self.assertEqual(list(map(int, lengths)), [3, 4, 5])
                queries = kwargs.get(
                    "actual_seq_qlen", kwargs.get("actual_seq_lengths")
                )
                self.assertEqual(list(map(int, queries)), [2, 2, 3])
                return kwargs["query"].clone(), None

            for hybrid in (False, True):
                backend.is_hybrid_swa = hybrid
                name = (
                    "npu_fused_infer_attention_score_v2"
                    if hybrid
                    else "npu_fused_infer_attention_score"
                )
                with patch(
                    "sglang.srt.hardware_backend.npu.attention.ascend_backend.torch_npu."
                    + name,
                    side_effect=fia,
                ):
                    with CountScalars() as mode:
                        for _ in range(2):
                            result = backend.forward_extend(
                                q, None, None, layer, batch, False
                            )
                    if reuse:
                        self.assertEqual(mode.count, 0)
                    else:
                        self.assertGreater(mode.count, 0)
                    torch.testing.assert_close(result, q)


class TestExpandDsaSparseIndices(unittest.TestCase):
    def test_2d_input_adds_unsqueeze(self):
        """A [T, K] tensor becomes [T, 1, K]."""
        topk = torch.tensor([[1, 2, 3], [4, 5, 6]])
        result = _expand_dsa_sparse_indices(topk)
        self.assertEqual(result.shape, (2, 1, 3))
        self.assertTrue(torch.equal(result.squeeze(1), topk))

    def test_3d_input_passthrough(self):
        topk = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = _expand_dsa_sparse_indices(topk)
        self.assertEqual(result.shape, (2, 2, 2))
        self.assertTrue(torch.equal(result, topk))

    def test_2d_shape_correctness(self):
        topk = torch.zeros(5, 8)
        result = _expand_dsa_sparse_indices(topk)
        self.assertEqual(result.dim(), 3)
        self.assertEqual(result.shape[0], 5)
        self.assertEqual(result.shape[1], 1)
        self.assertEqual(result.shape[2], 8)

    def test_2d_single_row(self):
        topk = torch.tensor([[1, 2, 3, 4]])
        result = _expand_dsa_sparse_indices(topk)
        self.assertEqual(result.shape, (1, 1, 4))


class TestReshapeKvForFiaNz(unittest.TestCase):
    def test_output_shape(self):
        """Output shape is (-1, 1, num_heads*head_dim//16, page_size, 16)."""
        num_heads = 2
        head_dim = 64
        page_size = 16
        total = 1 * 1 * (num_heads * head_dim // 16) * page_size * 16
        tensor = torch.arange(total, dtype=torch.float32)
        result = _reshape_kv_for_fia_nz(tensor, num_heads, head_dim, page_size)
        self.assertEqual(
            result.shape, (1, 1, num_heads * head_dim // 16, page_size, 16)
        )

    def test_element_preservation(self):
        num_heads = 2
        head_dim = 64
        page_size = 16
        total = 2 * 1 * (num_heads * head_dim // 16) * page_size * 16
        tensor = torch.arange(total, dtype=torch.float32)
        result = _reshape_kv_for_fia_nz(tensor, num_heads, head_dim, page_size)
        self.assertEqual(result.numel(), tensor.numel())
        self.assertTrue(torch.equal(result.flatten(), tensor))

    def test_different_parameters(self):
        num_heads = 4
        head_dim = 128
        page_size = 32
        total = 3 * 1 * (num_heads * head_dim // 16) * page_size * 16
        tensor = torch.randn(total)
        result = _reshape_kv_for_fia_nz(tensor, num_heads, head_dim, page_size)
        self.assertEqual(
            result.shape, (3, 1, num_heads * head_dim // 16, page_size, 16)
        )

    def test_view_relationship(self):
        num_heads = 2
        head_dim = 64
        page_size = 16
        total = 1 * 1 * (num_heads * head_dim // 16) * page_size * 16
        tensor = torch.arange(total, dtype=torch.float32)
        result = _reshape_kv_for_fia_nz(tensor, num_heads, head_dim, page_size)
        self.assertEqual(result.data_ptr(), tensor.data_ptr())


class TestGenerateMaskFlag(unittest.TestCase):
    def test_shape(self):
        mask = AscendAttnMaskBuilder.generate_mask_flag(8)
        self.assertEqual(mask.shape, (8, 8))

    def test_dtype_bool(self):
        mask = AscendAttnMaskBuilder.generate_mask_flag(4)
        self.assertEqual(mask.dtype, torch.bool)

    def test_upper_triangular_pattern(self):
        """generate_mask_flag returns ~tril, i.e. True above the diagonal."""
        mask = AscendAttnMaskBuilder.generate_mask_flag(4)
        self.assertTrue(mask[0, 1].item())
        self.assertTrue(mask[0, 3].item())
        self.assertTrue(mask[2, 3].item())
        self.assertFalse(mask[0, 0].item())
        self.assertFalse(mask[1, 0].item())
        self.assertFalse(mask[3, 3].item())

    def test_diagonal_is_false(self):
        mask = AscendAttnMaskBuilder.generate_mask_flag(5)
        for i in range(5):
            self.assertFalse(mask[i, i].item())

    def test_1x1(self):
        mask = AscendAttnMaskBuilder.generate_mask_flag(1)
        self.assertEqual(mask.shape, (1, 1))
        self.assertFalse(mask[0, 0].item())

    def test_symmetric_upper(self):
        n = 6
        mask = AscendAttnMaskBuilder.generate_mask_flag(n)
        for i in range(n):
            for j in range(i + 1, n):
                self.assertTrue(mask[i, j].item())
                self.assertFalse(mask[j, i].item())


class TestPagedPrefillMask(unittest.TestCase):
    def setUp(self):
        self.builder = object.__new__(AscendAttnMaskBuilder)
        self.builder.device = "cpu"
        self.builder.fia_mask = AscendAttnMaskBuilder.generate_mask_flag(2048)
        self.builder.paged_fia_mask = None

    def test_short_sequences_keep_compressed_mask(self):
        mask, sparse_mode = self.builder.get_paged_prefill_mask(1200, 2000, 2048)
        self.assertEqual(sparse_mode, 3)
        self.assertEqual(mask.data_ptr(), self.builder.fia_mask.data_ptr())
        self.assertIsNone(self.builder.paged_fia_mask)

    def test_long_paged_masks_preserve_absolute_causality(self):
        for q_len, kv_len, kv_capacity in (
            (1200, 2929, 23 * 128),
            (3500, 3500, 28 * 128),
            (1200, 1200, 32 * 128),
        ):
            with self.subTest(q_len=q_len, kv_len=kv_len, capacity=kv_capacity):
                mask, sparse_mode = self.builder.get_paged_prefill_mask(
                    q_len, kv_len, kv_capacity
                )
                self.assertEqual(sparse_mode, 0)
                self.assertGreaterEqual(mask.shape[-1], kv_capacity)
                self.assertEqual(mask.shape[:2], (1, q_len))
                self.assertTrue(mask.is_contiguous())
                positions = torch.arange(kv_len - q_len, kv_len)
                keys = torch.arange(mask.shape[-1])
                torch.testing.assert_close(mask[0], keys[None, :] > positions[:, None])

    def test_reuses_full_mask_storage_for_prefix_changes(self):
        first, _ = self.builder.get_paged_prefill_mask(1200, 2929, 2944)
        second, _ = self.builder.get_paged_prefill_mask(1200, 2800, 2816)
        self.assertEqual(
            first.untyped_storage().data_ptr(), second.untyped_storage().data_ptr()
        )

    def test_bsnd_forward_covers_long_paged_kv(self):
        backend = object.__new__(AscendAttnBackend)
        backend.reuse_fia_metadata = False
        backend.reuse_fia_kv_views = False
        backend.use_mla = False
        backend.use_fia = True
        backend.is_dllm_model = False
        backend.is_hybrid_swa = False
        backend.attn_cp_size = 1
        backend.page_size = 128
        backend.ascend_attn_mask_builder = self.builder
        backend.fia_mask = self.builder.fia_mask
        backend.forward_metadata = SimpleNamespace(
            block_tables=torch.arange(23, dtype=torch.int32).reshape(1, -1),
            seq_lens_cpu_list=[2929],
        )
        cache = torch.zeros(23, 128, 512)
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda _: cache, get_value_buffer=lambda _: cache
        )
        layer = SimpleNamespace(
            layer_id=0,
            qk_head_dim=512,
            v_head_dim=512,
            tp_q_head_num=1,
            tp_k_head_num=1,
            tp_v_head_num=1,
            scaling=512**-0.5,
        )
        batch = SimpleNamespace(
            forward_mode=MagicMock(),
            seq_lens_cpu=torch.tensor([2929]),
            extend_seq_lens_cpu=[1200],
        )
        batch.forward_mode.is_target_verify.return_value = False
        batch.forward_mode.is_draft_extend_v2.return_value = False
        batch.forward_mode.is_context_parallel_extend.return_value = False

        def fia(**kwargs):
            mask = kwargs["atten_mask"]
            capacity = kwargs["block_table"].shape[1] * kwargs["block_size"]
            self.assertGreaterEqual(mask.shape[-1], capacity)
            self.assertEqual(kwargs["sparse_mode"], 0)
            self.assertFalse(mask[0, 0, 1729].item())
            self.assertTrue(mask[0, 0, 1730].item())
            return kwargs["query"].clone(), None

        q = torch.randn(1200, 512)
        for reuse_metadata in (False, True):
            for reuse_views in (False, True):
                backend.reuse_fia_metadata = reuse_metadata
                backend.reuse_fia_kv_views = reuse_views
                with (
                    self.subTest(metadata=reuse_metadata, views=reuse_views),
                    patch(
                        "sglang.srt.hardware_backend.npu.attention.ascend_backend."
                        "torch_npu.npu_fused_infer_attention_score_v2",
                        side_effect=fia,
                    ),
                ):
                    output = backend.forward_extend(q, None, None, layer, batch, False)
                torch.testing.assert_close(output, q)


class TestGenerateAttnMask(unittest.TestCase):
    def test_shape(self):
        mask = AscendAttnMaskBuilder.generate_attn_mask(8, "norm", torch.float16)
        self.assertEqual(mask.shape, (8, 8))

    def test_dtype(self):
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "norm", torch.bfloat16)
        self.assertEqual(mask.dtype, torch.bfloat16)

    def test_default_dtype_float16(self):
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "norm")
        self.assertEqual(mask.dtype, torch.float16)

    def test_mix_mode_float16(self):
        """mix + float16 -> upper triangle is -inf, lower is 0."""
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "mix", torch.float16)
        self.assertEqual(mask.dtype, torch.float16)
        self.assertTrue(torch.isinf(mask[0, 1]))
        self.assertTrue(mask[0, 1] < 0)
        self.assertEqual(mask[0, 0].item(), 0.0)
        self.assertEqual(mask[1, 0].item(), 0.0)

    def test_mix_mode_bfloat16(self):
        """mix + bfloat16 -> upper triangle is -inf, lower is 0."""
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "mix", torch.bfloat16)
        self.assertEqual(mask.dtype, torch.bfloat16)
        self.assertTrue(torch.isinf(mask[0, 1]))
        self.assertTrue(mask[0, 1] < 0)
        self.assertEqual(mask[1, 1].item(), 0.0)

    def test_norm_mode_float16(self):
        """norm + float16 -> upper triangle is -inf (overflow), lower is 0."""
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "norm", torch.float16)
        self.assertEqual(mask.dtype, torch.float16)
        self.assertTrue(torch.isinf(mask[0, 1]))
        self.assertTrue(mask[0, 1] < 0)
        self.assertEqual(mask[0, 0].item(), 0.0)

    def test_norm_mode_bfloat16(self):
        """norm + bfloat16 -> upper triangle is 1, lower is 0."""
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "norm", torch.bfloat16)
        self.assertEqual(mask.dtype, torch.bfloat16)
        self.assertEqual(mask[0, 1].item(), 1.0)
        self.assertEqual(mask[0, 0].item(), 0.0)

    def test_norm_mode_float32(self):
        """norm + float32 -> upper triangle is 1, lower is 0."""
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "norm", torch.float32)
        self.assertEqual(mask.dtype, torch.float32)
        self.assertEqual(mask[0, 1].item(), 1.0)
        self.assertEqual(mask[0, 0].item(), 0.0)

    def test_mix_mode_float32(self):
        """mix + float32 -> upper triangle is 1, lower is 0."""
        mask = AscendAttnMaskBuilder.generate_attn_mask(4, "mix", torch.float32)
        self.assertEqual(mask.dtype, torch.float32)
        self.assertEqual(mask[0, 1].item(), 1.0)
        self.assertEqual(mask[0, 0].item(), 0.0)

    def test_lower_triangle_all_zero(self):
        """The lower triangle (including diagonal) is always zero."""
        n = 6
        mask = AscendAttnMaskBuilder.generate_attn_mask(n, "norm", torch.float32)
        for i in range(n):
            for j in range(i + 1):
                self.assertEqual(mask[i, j].item(), 0.0)

    def test_upper_triangle_all_masked(self):
        n = 6
        mask = AscendAttnMaskBuilder.generate_attn_mask(n, "norm", torch.float32)
        for i in range(n):
            for j in range(i + 1, n):
                self.assertEqual(mask[i, j].item(), 1.0)

    def test_diagonal_is_zero(self):
        """Diagonal is always zero regardless of mode/dtype."""
        for mode in ("mix", "norm"):
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                mask = AscendAttnMaskBuilder.generate_attn_mask(4, mode, dtype)
                for i in range(4):
                    self.assertEqual(mask[i, i].item(), 0.0)


class TestGetAttentionMaskId(unittest.TestCase):
    def test_flat_tensor(self):
        """Produces a flat tensor of arange ranges concatenated."""
        seq_lens = torch.tensor([10, 20])
        extend_lens = torch.tensor([3, 5])
        result = AscendAttnMaskBuilder.get_attention_mask_id(seq_lens, extend_lens)
        expected = torch.tensor([7, 8, 9, 15, 16, 17, 18, 19])
        self.assertEqual(result.dim(), 1)
        self.assertTrue(torch.equal(result, expected))

    def test_single_sequence(self):
        seq_lens = torch.tensor([5])
        extend_lens = torch.tensor([2])
        result = AscendAttnMaskBuilder.get_attention_mask_id(seq_lens, extend_lens)
        expected = torch.tensor([3, 4])
        self.assertTrue(torch.equal(result, expected))

    def test_multiple_sequences(self):
        seq_lens = torch.tensor([3, 6, 10])
        extend_lens = torch.tensor([1, 2, 4])
        result = AscendAttnMaskBuilder.get_attention_mask_id(seq_lens, extend_lens)
        expected = torch.tensor([2, 4, 5, 6, 7, 8, 9])
        self.assertTrue(torch.equal(result, expected))

    def test_total_length(self):
        """Result length equals sum of extend_lens per row."""
        seq_lens = torch.tensor([10, 20])
        extend_lens = torch.tensor([3, 5])
        result = AscendAttnMaskBuilder.get_attention_mask_id(seq_lens, extend_lens)
        expected_len = 3 + 5
        self.assertEqual(result.numel(), expected_len)

    def test_values_correct(self):
        seq_lens = torch.tensor([4, 8])
        extend_lens = torch.tensor([2, 3])
        result = AscendAttnMaskBuilder.get_attention_mask_id(seq_lens, extend_lens)
        self.assertEqual(result[0].item(), 2)
        self.assertEqual(result[1].item(), 3)
        self.assertEqual(result[2].item(), 5)
        self.assertEqual(result[4].item(), 7)


class TestUpdateAttnCache(unittest.TestCase):
    def setUp(self):
        self.builder = object.__new__(AscendAttnMaskBuilder)
        self.builder.device = "cpu"

    def test_seqlen_greater_than_cached(self):
        """When seqlen > cached, the mask is regenerated and cache updated."""
        old_mask = torch.zeros(4, 4)
        result_mask, result_len = self.builder.update_attn_cache(
            seqlen=8,
            mask_cache=old_mask,
            seq_len_cached=4,
            dtype=torch.float16,
            mode="norm",
        )
        self.assertEqual(result_len, 8)
        self.assertEqual(result_mask.shape, (8, 8))
        self.assertEqual(result_mask.dtype, torch.float16)

    def test_seqlen_less_equal_cached(self):
        """When seqlen <= cached, the existing mask is kept."""
        old_mask = torch.ones(8, 8, dtype=torch.float16)
        result_mask, result_len = self.builder.update_attn_cache(
            seqlen=4,
            mask_cache=old_mask,
            seq_len_cached=8,
            dtype=torch.float16,
            mode="norm",
        )
        self.assertEqual(result_len, 8)
        self.assertIs(result_mask, old_mask)

    def test_dtype_change(self):
        """When dtype differs, the mask is converted but not regenerated."""
        old_mask = torch.ones(8, 8, dtype=torch.float16)
        result_mask, result_len = self.builder.update_attn_cache(
            seqlen=4,
            mask_cache=old_mask,
            seq_len_cached=8,
            dtype=torch.float32,
            mode="norm",
        )
        self.assertEqual(result_len, 8)
        self.assertEqual(result_mask.dtype, torch.float32)
        self.assertIsNot(result_mask, old_mask)

    def test_no_change(self):
        """When seqlen <= cached and dtype matches, nothing changes."""
        old_mask = torch.ones(8, 8, dtype=torch.float16)
        result_mask, result_len = self.builder.update_attn_cache(
            seqlen=8,
            mask_cache=old_mask,
            seq_len_cached=8,
            dtype=torch.float16,
            mode="norm",
        )
        self.assertEqual(result_len, 8)
        self.assertIs(result_mask, old_mask)

    def test_seqlen_greater_and_dtype_change(self):
        """Both regeneration and dtype conversion happen."""
        old_mask = torch.zeros(4, 4, dtype=torch.float16)
        result_mask, result_len = self.builder.update_attn_cache(
            seqlen=16,
            mask_cache=old_mask,
            seq_len_cached=4,
            dtype=torch.float32,
            mode="norm",
        )
        self.assertEqual(result_len, 16)
        self.assertEqual(result_mask.shape, (16, 16))
        self.assertEqual(result_mask.dtype, torch.float32)

    def test_seqlen_equal_cached_no_regen(self):
        """seqlen == cached should not trigger regeneration."""
        old_mask = torch.ones(8, 8, dtype=torch.float32)
        result_mask, result_len = self.builder.update_attn_cache(
            seqlen=8,
            mask_cache=old_mask,
            seq_len_cached=8,
            dtype=torch.float32,
            mode="mix",
        )
        self.assertEqual(result_len, 8)
        self.assertIs(result_mask, old_mask)


class TestGetSplitfuseAttnMask(unittest.TestCase):
    def setUp(self):
        self.builder = object.__new__(AscendAttnMaskBuilder)
        self.builder.device = "cpu"

    def test_output_shape(self):
        mask = self.builder.get_splitfuse_attn_mask(8)
        self.assertEqual(mask.shape, (8, 8))

    def test_dtype_int8(self):
        mask = self.builder.get_splitfuse_attn_mask(4)
        self.assertEqual(mask.dtype, torch.int8)

    def test_upper_triangular(self):
        """Upper triangle (excluding diagonal) is 1, rest is 0."""
        mask = self.builder.get_splitfuse_attn_mask(4)
        self.assertEqual(mask[0, 1].item(), 1)
        self.assertEqual(mask[0, 3].item(), 1)
        self.assertEqual(mask[0, 0].item(), 0)
        self.assertEqual(mask[1, 0].item(), 0)
        self.assertEqual(mask[1, 1].item(), 0)

    def test_lower_triangle_zero(self):
        n = 5
        mask = self.builder.get_splitfuse_attn_mask(n)
        for i in range(n):
            for j in range(i + 1):
                self.assertEqual(mask[i, j].item(), 0)

    def test_upper_triangle_one(self):
        n = 5
        mask = self.builder.get_splitfuse_attn_mask(n)
        for i in range(n):
            for j in range(i + 1, n):
                self.assertEqual(mask[i, j].item(), 1)


class TestGetSwaMask(unittest.TestCase):
    def setUp(self):
        self.builder = object.__new__(AscendAttnMaskBuilder)
        self.builder.device = "cpu"

    def test_output_shape(self):
        """Output shape is (batch, 1, s2)."""
        seq_lens = torch.tensor([5, 10])
        mask = self.builder.get_swa_mask(seq_lens, s2=15, left_context=512)
        self.assertEqual(mask.shape, (2, 1, 15))

    def test_1d_input_unsqueezed(self):
        """1-D input of shape (B,) is handled; output still (B, 1, s2)."""
        seq_lens = torch.tensor([3, 6, 9])
        mask = self.builder.get_swa_mask(seq_lens, s2=12, left_context=512)
        self.assertEqual(mask.shape, (3, 1, 12))

    def test_2d_input(self):
        """2-D input of shape (B, 1) is also accepted."""
        seq_lens = torch.tensor([[5], [10]])
        mask = self.builder.get_swa_mask(seq_lens, s2=15, left_context=512)
        self.assertEqual(mask.shape, (2, 1, 15))

    def test_mask_values_large_left_context(self):
        """With left_context >= max seq_len, only indices >= seq_lens are True."""
        seq_lens = torch.tensor([3, 5])
        mask = self.builder.get_swa_mask(seq_lens, s2=8, left_context=512)
        row0 = mask[0, 0]
        self.assertFalse(row0[0].item())
        self.assertFalse(row0[2].item())
        self.assertTrue(row0[3].item())
        self.assertTrue(row0[7].item())
        row1 = mask[1, 0]
        self.assertFalse(row1[4].item())
        self.assertTrue(row1[5].item())
        self.assertTrue(row1[7].item())

    def test_mask_values_small_left_context(self):
        """With a small left_context, earlier positions are also masked."""
        seq_lens = torch.tensor([10, 20])
        mask = self.builder.get_swa_mask(seq_lens, s2=30, left_context=5)
        row0 = mask[0, 0]
        self.assertTrue(row0[0].item())
        self.assertTrue(row0[4].item())
        self.assertFalse(row0[5].item())
        self.assertFalse(row0[9].item())
        self.assertTrue(row0[10].item())
        self.assertTrue(row0[29].item())
        row1 = mask[1, 0]
        self.assertTrue(row1[0].item())
        self.assertTrue(row1[14].item())
        self.assertFalse(row1[15].item())
        self.assertFalse(row1[19].item())
        self.assertTrue(row1[20].item())

    def test_clamp_to_zero(self):
        """When seq_len < left_context, start is clamped to 0."""
        seq_lens = torch.tensor([2])
        mask = self.builder.get_swa_mask(seq_lens, s2=5, left_context=10)
        row = mask[0, 0]
        self.assertFalse(row[0].item())
        self.assertFalse(row[1].item())
        self.assertTrue(row[2].item())
        self.assertTrue(row[4].item())

    def test_default_left_context(self):
        """Default left_context is 512."""
        seq_lens = torch.tensor([10])
        mask = self.builder.get_swa_mask(seq_lens, s2=15)
        self.assertEqual(mask.shape, (1, 1, 15))
        row = mask[0, 0]
        self.assertFalse(row[9].item())
        self.assertTrue(row[10].item())

    def test_dtype_bool(self):
        seq_lens = torch.tensor([5, 10])
        mask = self.builder.get_swa_mask(seq_lens, s2=15, left_context=512)
        self.assertEqual(mask.dtype, torch.bool)


class TestGenerateAlibiBias(unittest.TestCase):
    def setUp(self):
        self.backend = object.__new__(AscendAttnBackend)
        self.backend.reuse_fia_metadata = False
        self.backend.reuse_fia_kv_views = False

    def test_shape(self):
        """Output shape is (num_heads, 1, seq_len)."""
        slopes = torch.tensor([0.1, 0.2, 0.3, 0.4])
        result = self.backend._generate_alibi_bias(
            seq_len=8,
            slopes=slopes,
            num_heads=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(result.shape, (4, 1, 8))

    def test_values(self):
        """Each element is slopes[h] * position."""
        slopes = torch.tensor([1.0, 2.0, 3.0, 4.0])
        seq_len = 5
        result = self.backend._generate_alibi_bias(
            seq_len=seq_len,
            slopes=slopes,
            num_heads=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for h in range(4):
            for p in range(seq_len):
                expected = slopes[h].item() * p
                self.assertAlmostEqual(result[h, 0, p].item(), expected, places=5)

    def test_dtype(self):
        slopes = torch.tensor([0.5, 1.0])
        result = self.backend._generate_alibi_bias(
            seq_len=4,
            slopes=slopes,
            num_heads=2,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        self.assertEqual(result.dtype, torch.bfloat16)

    def test_single_head(self):
        slopes = torch.tensor([1.5])
        result = self.backend._generate_alibi_bias(
            seq_len=3,
            slopes=slopes,
            num_heads=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(result.shape, (1, 1, 3))
        self.assertAlmostEqual(result[0, 0, 0].item(), 0.0)
        self.assertAlmostEqual(result[0, 0, 1].item(), 1.5)
        self.assertAlmostEqual(result[0, 0, 2].item(), 3.0)

    def test_zero_position_is_zero(self):
        """Position 0 always yields 0 regardless of slopes."""
        slopes = torch.tensor([1.0, 2.0, 3.0])
        result = self.backend._generate_alibi_bias(
            seq_len=4,
            slopes=slopes,
            num_heads=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for h in range(3):
            self.assertAlmostEqual(result[h, 0, 0].item(), 0.0)

    def test_default_dtype_bfloat16(self):
        slopes = torch.tensor([0.5, 1.0])
        result = self.backend._generate_alibi_bias(
            seq_len=4,
            slopes=slopes,
            num_heads=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(result.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
