"""Page-table parity and eligibility for opt-in eager DSA metadata."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.attention import dsa_metadata
from sglang.srt.layers.dcp import layout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDsaMetadata(CustomTestCase):
    @staticmethod
    def inputs(lengths):
        table = (torch.arange(3 * 8192).reshape(3, 8192) * 3 + 2048).long()
        reads = []

        class Map:
            def __getitem__(self, index):
                reads.append(index)
                return table[index]

        backend = SimpleNamespace(
            use_dsa=True,
            is_hybrid_swa=False,
            graph_mode=False,
            page_size=128,
            req_to_token_pool=SimpleNamespace(req_to_token=Map()),
            forward_metadata=SimpleNamespace(),
        )
        mode = SimpleNamespace(
            is_extend=lambda: True,
            is_target_verify=lambda: False,
            is_draft_extend_v2=lambda: False,
        )
        batch = SimpleNamespace(
            seq_lens_cpu=torch.tensor(lengths),
            extend_prefix_lens_cpu=[max(0, n - 7) for n in lengths],
            extend_seq_lens_cpu=[min(n, 7) for n in lengths],
            req_pool_indices=torch.tensor([2, 0][: len(lengths)]),
            forward_mode=mode,
        )
        return backend, batch, table, reads

    def test_virtual_and_local_pages_match_legacy_for_ragged_batches(self):
        for size in (1, 2, 16):
            for lengths in ([0], [1], [4097], [4097, 2183], [0, 4097]):
                with self.subTest(size=size, lengths=lengths):
                    backend, batch, table, reads = self.inputs(lengths)
                    parallel = SimpleNamespace(dcp_enabled=size > 1, attn_dcp_size=size)
                    with (
                        envs.SGLANG_NPU_DSA_METADATA_REUSE.override(True),
                        patch.object(
                            dsa_metadata, "get_parallel", return_value=parallel
                        ),
                        patch.object(dsa_metadata, "record"),
                    ):
                        self.assertTrue(
                            dsa_metadata.prepare_dsa_page_tables(backend, batch)
                        )
                    old_rows = table[batch.req_pool_indices, : max(lengths)]
                    torch.testing.assert_close(
                        backend.forward_metadata.block_tables,
                        old_rows[:, ::128] // 128,
                        rtol=0,
                        atol=0,
                    )
                    if size > 1:
                        torch.testing.assert_close(
                            backend.forward_metadata.dcp_local_block_tables,
                            old_rows[:, :: 128 * size] // (128 * size),
                            rtol=0,
                            atol=0,
                        )
                    self.assertEqual(len(reads), 1)
                    self.assertEqual(reads[0][1].step, 128)

    def test_default_and_unsupported_paths_do_not_read_token_map(self):
        backend, batch, _, reads = self.inputs([4097, 2183])
        with envs.SGLANG_NPU_DSA_METADATA_REUSE.override(False):
            self.assertFalse(dsa_metadata.prepare_dsa_page_tables(backend, batch))
        for key, value in (
            ("use_dsa", False),
            ("is_hybrid_swa", True),
            ("graph_mode", True),
            ("is_draft_worker", True),
        ):
            backend, batch, _, reads = self.inputs([4097, 2183])
            setattr(backend, key, value)
            with envs.SGLANG_NPU_DSA_METADATA_REUSE.override(True):
                self.assertFalse(dsa_metadata.prepare_dsa_page_tables(backend, batch))
            self.assertEqual(reads, [])
        backend, batch, _, reads = self.inputs([4097, 2183])
        batch.seq_lens_cpu[0] += 1
        with envs.SGLANG_NPU_DSA_METADATA_REUSE.override(True):
            self.assertFalse(dsa_metadata.prepare_dsa_page_tables(backend, batch))
        self.assertEqual(reads, [])

    def test_two_requests_keep_request_contiguous_gather(self):
        batch = SimpleNamespace(
            extend_prefix_lens_cpu=[4096, 6144], extend_seq_lens_cpu=[512, 768]
        )
        with patch.object(layout, "_enable_dcp_packed_read", True):
            self.assertIsNone(layout.dcp_packed_read_plan(batch, 2048))
        self.assertIsNone(batch.npu_dcp_packed_plan)


if __name__ == "__main__":
    unittest.main()
