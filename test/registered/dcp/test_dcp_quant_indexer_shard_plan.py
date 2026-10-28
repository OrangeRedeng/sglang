"""The indexer length vectors must describe the query rows actually passed.

The previous metadata API planned the full batch and could read past a sharded
query tensor. Check the same bound on the current per-call length-vector ABI.
"""

import unittest

import torch

from sglang.srt.layers.attention.dsa.dsa_npu_indexer import (
    _IndexerQueryShard,
    plan_indexer_query_shard,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestQuantIndexerShardLengths(CustomTestCase):
    def test_lengths_fit_the_tensor_on_every_rank(self):
        for prefixes, extends in (
            ([500], [10828]),
            ([0, 50, 900], [5, 0, 9]),
            ([100], [3]),
        ):
            for tp_size in (2, 4, 8, 16):
                parts = []
                query = torch.arange(sum(extends)).view(-1, 1)
                for rank in range(tp_size):
                    with self.subTest(extends=extends, tp=tp_size, rank=rank):
                        start, rows, real, cumulative, keys = plan_indexer_query_shard(
                            prefixes, extends, tp_size, rank
                        )
                        shard = _IndexerQueryShard(
                            start,
                            rows,
                            real,
                            sum(extends),
                            tp_size,
                            torch.tensor(cumulative, dtype=torch.int32),
                            torch.tensor(keys, dtype=torch.int32),
                        )
                        local = shard.take(query)
                        self.assertEqual(local.shape[0], rows)
                        self.assertEqual(int(shard.actual_seq_lengths_q[-1]), real)
                        self.assertLessEqual(real, local.shape[0])
                        self.assertTrue(
                            bool(
                                (
                                    shard.actual_seq_lengths_q[1:]
                                    >= shard.actual_seq_lengths_q[:-1]
                                ).all()
                            )
                        )
                        self.assertEqual(
                            shard.actual_seq_lengths_q.numel(), len(extends)
                        )
                        self.assertEqual(
                            shard.actual_seq_lengths_kv.numel(), len(extends)
                        )
                        self.assertEqual(shard.actual_seq_lengths_kv.dtype, torch.int32)
                        parts.append(local[:real])
                self.assertTrue(torch.equal(torch.cat(parts), query))

    def test_batch_wide_lengths_would_overrun_a_shard(self):
        _, rows, real, cumulative, _ = plan_indexer_query_shard([500], [10828], 16, 0)
        self.assertGreater(10828, rows)
        self.assertLessEqual(cumulative[-1], rows)
        self.assertEqual(cumulative[-1], real)

    def test_key_lengths_are_absolute_per_request(self):
        _, _, _, cumulative, keys = plan_indexer_query_shard([100, 205], [5, 9], 2, 0)
        self.assertEqual(cumulative, [5, 7])
        self.assertEqual(keys, [105, 207])


if __name__ == "__main__":
    unittest.main()
