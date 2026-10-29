"""Guard asynchronous prefix ownership, wait ordering and packed remapping."""

import unittest

import torch
from sglang.srt.hardware_backend.npu.attention.dcp_gather import (
    start_prefix_gather,
    validate_legacy_dcp_flags,
)
from sglang.srt.layers.dcp.layout import (
    dcp_owned_prefix_positions,
    plan_dcp_extend_gather,
    plan_dcp_packed_read,
    remap_topk_to_packed,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class CollectiveWorld:
    """Complete deferred collectives from the actual sends of every rank."""

    def __init__(self, size):
        self.size, self.calls = size, {}

    def rank(self, rank):
        world = self

        class Group:
            ordinal = 0

            def all_gather_into_tensor_async(self, output, send):
                key = self.ordinal
                self.ordinal += 1
                slots = world.calls.setdefault(key, {})
                slots[rank] = (output, send)

                class Work:
                    def wait(self):
                        assert len(slots) == world.size
                        output.copy_(
                            torch.cat([slots[i][1] for i in range(world.size)])
                        )

                return Work()

        return Group()


class TestDcpPrefixGather(CustomTestCase):
    def test_deferred_gather_matches_position_order(self):
        for size in (2, 4):
            for page in (1, 8, 128):
                for prefixes, extends in (
                    ([0], [0]),
                    ([1], [7]),
                    ([259], [33]),
                    ([3, 267], [5, 7]),
                ):
                    with self.subTest(size=size, page=page, prefixes=prefixes):
                        prefix = torch.arange(
                            sum(prefixes), dtype=torch.float32
                        ).reshape(-1, 1, 1)
                        own = (
                            10000 + torch.arange(sum(extends), dtype=torch.float32)
                        ).reshape(-1, 1, 1)
                        expected = []
                        a = b = 0
                        for plen, elen in zip(prefixes, extends):
                            expected.extend((prefix[a : a + plen], own[b : b + elen]))
                            a += plen
                            b += elen
                        expected = torch.cat(expected)
                        world, pending = CollectiveWorld(size), []
                        for rank in range(size):
                            indices = dcp_owned_prefix_positions(
                                prefixes, size, rank, page
                            )
                            send = prefix[indices]
                            plan = plan_dcp_extend_gather(
                                prefixes, extends, size, rank, 128, page
                            )
                            pending.append(
                                start_prefix_gather(
                                    world.rank(rank),
                                    (send, send + 1),
                                    (prefix, prefix),
                                    plan,
                                    size,
                                )
                            )
                        for item in pending:
                            a, b = item.finish((own, own + 1))
                            torch.testing.assert_close(a, expected, rtol=0, atol=0)
                            torch.testing.assert_close(b, expected + 1, rtol=0, atol=0)

    def test_packed_rows_and_attention_match_page_owned_reference(self):
        for size in (2, 4):
            for page in (1, 8, 128):
                for length in (0, 1, 127, 259, 1025):
                    with self.subTest(size=size, page=page, length=length):
                        prefix = torch.arange(length, dtype=torch.float32).reshape(
                            -1, 1, 1
                        )
                        own = (10000 + torch.arange(7, dtype=torch.float32)).reshape(
                            -1, 1, 1
                        )
                        plan = plan_dcp_packed_read(length, 7, size, page)
                        world, pending = CollectiveWorld(size), []
                        for rank in range(size):
                            send = prefix[
                                dcp_owned_prefix_positions([length], size, rank, page)
                            ]
                            pending.append(
                                start_prefix_gather(
                                    world.rank(rank),
                                    (send, send),
                                    (prefix, prefix),
                                    plan,
                                    size,
                                    True,
                                )
                            )
                        logical = torch.cat((prefix, own))
                        ids = torch.arange(length + 7, dtype=torch.int32)
                        mapped = remap_topk_to_packed(ids, plan).long()
                        for item in pending:
                            packed, _ = item.finish((own, own))
                            torch.testing.assert_close(
                                packed[mapped], logical, rtol=0, atol=0
                            )
                            query = torch.tensor([0.0001])
                            a = (
                                torch.softmax(logical.flatten() * query, 0)
                                @ logical.flatten()
                            )
                            b = (
                                torch.softmax(packed[mapped].flatten() * query, 0)
                                @ packed[mapped].flatten()
                            )
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                        self.assertEqual(
                            remap_topk_to_packed(torch.tensor([-1]), plan).item(), -1
                        )

    def test_disconnected_flags_fail_before_allocating(self):
        for key in (
            "SGLANG_NPU_DSA_CP_TOKEN_SHARDED",
            "SGLANG_NPU_STREAM_O_PROJ",
            "SGLANG_NPU_DSA_CP_Q_FULL_WIDTH",
            "SGLANG_NPU_DSA_CP_Q_A2A_PREISSUE",
            "SGLANG_NPU_DSA_CP_LOCAL_TOPK",
            "SGLANG_NPU_DCP_DECODE_Q_PREISSUE",
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_legacy_dcp_flags({key: "True"})
        validate_legacy_dcp_flags({"SGLANG_NPU_DSA_CP_TOKEN_SHARDED": "False"})


if __name__ == "__main__":
    unittest.main()
