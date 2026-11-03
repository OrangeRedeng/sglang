"""Request boundaries and distributed packing for opt-in two-request DCP."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.attention.dcp_gather import start_prefix_gather
from sglang.srt.layers.attention.dsa.dsa_cp_layout import plan_dsa_cp_shard
from sglang.srt.layers.dcp import layout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from test_dcp_prefix_gather import CollectiveWorld

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDcpPackedBatch2(CustomTestCase):
    def gather(self, prefixes, extends, size, page, dim=1):
        prefix = torch.arange(sum(prefixes) * dim).float().reshape(-1, 1, dim) / 100
        own = 20 + torch.arange(sum(extends) * dim).float().reshape(-1, 1, dim) / 100
        # The production query/KV all-gather can carry trailing TP padding.
        own = torch.cat((own, torch.full((size, 1, dim), -99.0)))
        world, pending, plans = CollectiveWorld(size), [], []
        for rank in range(size):
            plan = layout.plan_dcp_packed_batch_read(
                prefixes, extends, size, rank, page
            )
            positions = layout.dcp_owned_prefix_positions(prefixes, size, rank, page)
            send = prefix[positions]
            with patch.object(
                torch, "index_select", side_effect=AssertionError("full KV permutation")
            ):
                pending.append(
                    start_prefix_gather(
                        world.rank(rank),
                        (send, send + 1),
                        (prefix, prefix),
                        plan,
                        size,
                        True,
                    )
                )
            plans.append(plan)
        with patch.object(
            torch, "index_select", side_effect=AssertionError("full KV permutation")
        ):
            outputs = [item.finish((own, own + 1)) for item in pending]
        expected_collectives = 2 * sum(
            request.gathered_rows > 0 for request in plans[0].requests
        )
        self.assertEqual(len(world.calls), expected_collectives)
        return prefix, own, plans, outputs

    def test_request_segments_match_logical_kv_without_permutation(self):
        for size in (2, 4, 16):
            for page in (1, 8, 128):
                for prefixes, extends in (
                    ([3, 11], [3, 7]),
                    ([129, 385], [0, 7]),
                    ([1, 2], [1, 1]),
                    ([0, 0], [2, 3]),
                ):
                    with self.subTest(
                        size=size, page=page, prefixes=prefixes, extends=extends
                    ):
                        prefix, own, plans, outputs = self.gather(
                            prefixes, extends, size, page
                        )
                        for plan, (nope, rope) in zip(plans, outputs):
                            pstart = estart = 0
                            for request, offset in zip(plan.requests, plan.row_offsets):
                                logical = torch.cat(
                                    (
                                        prefix[pstart : pstart + request.prefix_len],
                                        own[estart : estart + request.extend_len],
                                    )
                                )
                                indices = torch.arange(
                                    request.prefix_len + request.extend_len
                                )
                                mapped = (
                                    layout.remap_topk_to_packed(indices, request).long()
                                    + offset
                                )
                                torch.testing.assert_close(
                                    nope[mapped], logical, rtol=0, atol=0
                                )
                                torch.testing.assert_close(
                                    rope[mapped], logical + 1, rtol=0, atol=0
                                )
                                pstart += request.prefix_len
                                estart += request.extend_len

    @staticmethod
    def backend_method():
        root = Path(__file__).resolve().parents[5]
        path = (
            root / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
        )
        tree = ast.parse(path.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "AscendAttnBackend"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "_forward_sparse_dcp_gathered"
        )
        return compile(
            "from __future__ import annotations\n" + ast.unparse(method),
            str(path),
            "exec",
        )

    def test_native_backend_uses_request_relative_ids_and_packed_cumulative_lengths(
        self,
    ):
        prefixes, extends, size = [259, 513], [3, 7], 4
        prefix, own, plans, outputs = self.gather(prefixes, extends, size, 128, dim=3)
        padded = 16
        queries = torch.arange(padded * 3).float().reshape(padded, 1, 3) / 50
        ids = torch.full((padded, 4), -1, dtype=torch.int32)
        full_reference = torch.zeros_like(queries)
        row = pstart = estart = 0
        for plen, elen in zip(prefixes, extends):
            logical = torch.cat(
                (prefix[pstart : pstart + plen], own[estart : estart + elen])
            )
            for index in range(elen):
                selected = torch.tensor([0, plen - 1, plen + index])
                ids[row, :3] = selected.int()
                values = logical[selected, 0]
                weights = (queries[row, 0] @ values.T).softmax(0)
                full_reference[row, 0] = weights @ values
                row += 1
            pstart += plen
            estart += elen
        code = self.backend_method()
        for rank, (packed, gathered) in enumerate(zip(plans, outputs)):
            query_plan = plan_dsa_cp_shard(
                extends,
                [p + e for p, e in zip(prefixes, extends)],
                size,
                rank,
                padded_num_tokens=padded,
            )
            local_ids = ids[query_plan.local_start : query_plan.local_end_with_pad]
            mapped = layout.remap_topk_to_packed(
                local_ids, packed, query_plan.query_lens
            )
            if query_plan.num_local_tokens == 0:
                self.assertTrue(bool((mapped == -1).all()))
                continue

            def kernel(**kwargs):
                result = torch.zeros_like(kwargs["query"])
                qends = kwargs["actual_seq_lengths_query"].tolist()
                kends = kwargs["actual_seq_lengths_kv"].tolist()
                self.assertEqual(kends, list(packed.key_lens))
                qstart = kstart = 0
                for qend, kend in zip(qends, kends):
                    for row in range(qstart, qend):
                        indices = kwargs["sparse_indices"][row, 0]
                        indices = indices[indices >= 0].long()
                        self.assertTrue(bool((indices < kend - kstart).all()))
                        values = kwargs["key"][kstart + indices, 0]
                        weights = (kwargs["query"][row, 0] @ values.T).softmax(0)
                        result[row, 0] = weights @ values
                    qstart, kstart = qend, kend
                return result, None, None

            namespace = dict(
                torch=torch,
                torch_npu=SimpleNamespace(npu_sparse_flash_attention=kernel),
                dcp_packed_kv_lens=layout.dcp_packed_kv_lens,
                _expand_dsa_sparse_indices=lambda x: x.unsqueeze(1),
            )
            exec(code, namespace)
            backend = SimpleNamespace(
                kv_cache_dtype=torch.bfloat16,
                _pad_topk_indices=lambda value, rows: value,
            )
            batch = SimpleNamespace(
                npu_dcp_prefix_layout=packed,
                attn_dcp_metadata=SimpleNamespace(
                    dcp_kv_indptr=torch.tensor([0, 262, 782])
                ),
            )
            query = queries[query_plan.local_start : query_plan.local_end_with_pad]
            qlens = torch.tensor(query_plan.query_lens).cumsum(0).int()
            result = namespace["_forward_sparse_dcp_gathered"](
                backend,
                query,
                torch.zeros_like(query),
                gathered,
                mapped,
                SimpleNamespace(scaling=1.0),
                qlens,
                batch,
                query_plan,
            )
            torch.testing.assert_close(
                result,
                full_reference[query_plan.local_start : query_plan.local_end_with_pad],
                rtol=1e-6,
                atol=1e-6,
            )

    def test_metadata_refreshes_when_partition_changes_but_total_rows_match(self):
        a = layout.plan_dcp_packed_batch_read([32, 64], [2, 2], 2, 0, 8)
        b = layout.plan_dcp_packed_batch_read([64, 32], [2, 2], 2, 0, 8)
        self.assertEqual(a.rows, b.rows)
        batch = SimpleNamespace()
        with patch.object(
            torch, "tensor", side_effect=AssertionError("blocking host copy")
        ):
            first = layout.dcp_packed_kv_lens(batch, a, "cpu")
            second = layout.dcp_packed_kv_lens(batch, b, "cpu")
        self.assertIsNot(first, second)
        self.assertEqual(first.tolist(), list(a.key_lens))
        self.assertEqual(second.tolist(), list(b.key_lens))

    def test_selection_keeps_default_and_causal_fallback(self):
        def batch(prefixes):
            return SimpleNamespace(
                extend_prefix_lens_cpu=prefixes, extend_seq_lens_cpu=[3, 7]
            )

        parallel = SimpleNamespace(dcp_size=16, dcp_rank=15)
        with (
            patch.object(layout, "_enable_dcp_packed_read", True),
            patch.object(layout, "get_parallel", return_value=parallel),
            patch.object(
                layout, "get_attn_backend", return_value=SimpleNamespace(page_size=128)
            ),
        ):
            with envs.SGLANG_NPU_ENABLE_DCP_PACKED_READ_BATCH2.override(False):
                self.assertIsNone(
                    layout.dcp_packed_read_plan(batch([4096, 6144]), 2048)
                )
            with envs.SGLANG_NPU_ENABLE_DCP_PACKED_READ_BATCH2.override(True):
                selected = layout.dcp_packed_read_plan(batch([4096, 6144]), 2048)
                self.assertIsInstance(selected, layout.DcpPackedBatchReadPlan)
                self.assertIsNone(layout.dcp_packed_read_plan(batch([4, 6144]), 2048))
                self.assertIsNone(
                    layout.dcp_packed_read_plan(batch([4096, 6144]), None)
                )
                with self.assertRaisesRegex(ValueError, "query lengths"):
                    layout.remap_topk_to_packed(torch.zeros((1, 2)), selected)


if __name__ == "__main__":
    unittest.main()
