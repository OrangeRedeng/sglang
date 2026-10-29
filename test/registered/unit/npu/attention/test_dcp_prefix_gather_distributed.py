"""Real HCCL prefix gather and two-stream lifecycle; launch with torchrun."""

import os
import unittest
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
from sglang.srt.hardware_backend.npu.attention.dcp_gather import DcpGatherPipeline
from sglang.srt.layers.dcp.layout import (
    dcp_owned_prefix_positions,
    plan_dcp_extend_gather,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=30, suite="nightly-8-npu-a3", nightly=True)


class TestDistributedPrefix(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if int(os.environ.get("WORLD_SIZE", "1")) < 2:
            raise unittest.SkipTest("requires torchrun with at least two NPU ranks")
        import torch_npu  # noqa: F401

        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("hccl", timeout=timedelta(seconds=120))

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_forward_reset_and_layer_lookahead(self):
        class Group:
            def all_gather_into_tensor_async(self, output, send):
                return dist.all_gather_into_tensor(output, send, async_op=True)

        size, rank = dist.get_world_size(), dist.get_rank()
        parallel = SimpleNamespace(dcp_group=Group(), dcp_size=size)
        pipeline = DcpGatherPipeline()
        # Reuse the pipeline across forwards with different sizes and layouts.
        for prefixes, extends in (([259], [7]), ([0, 517], [5, 33]), ([1], [1])):
            pipeline.reset()
            prefix = torch.arange(
                sum(prefixes), device="npu", dtype=torch.float32
            ).reshape(-1, 1, 1)
            own = (
                10000 + torch.arange(sum(extends), device="npu", dtype=torch.float32)
            ).reshape(-1, 1, 1)
            indices = dcp_owned_prefix_positions(prefixes, size, rank, 128).to("npu")
            plan = plan_dcp_extend_gather(prefixes, extends, size, rank, 128, 128)
            plan = plan._replace(
                pieces=[
                    piece._replace(index=piece.index.to("npu")) for piece in plan.pieces
                ]
            )
            pool = SimpleNamespace(
                get_mla_kv_buffer=lambda layer, ids: (
                    prefix[ids] + layer.layer_id,
                    prefix[ids] + layer.layer_id,
                )
            )
            pipeline.issue(0, pool, indices, (prefix, prefix), plan, parallel, False)
            for layer in range(3):
                actual, _ = pipeline.take(layer, (own + layer, own + layer))
                if layer < 2:
                    pipeline.issue(
                        layer + 1,
                        pool,
                        indices,
                        (prefix, prefix),
                        plan,
                        parallel,
                        False,
                    )
                expected, a, b = [], 0, 0
                for plen, elen in zip(prefixes, extends):
                    expected.extend(
                        (prefix[a : a + plen] + layer, own[b : b + elen] + layer)
                    )
                    a += plen
                    b += elen
                torch.testing.assert_close(actual, torch.cat(expected), rtol=0, atol=0)
            torch.npu.synchronize()


if __name__ == "__main__":
    unittest.main()
