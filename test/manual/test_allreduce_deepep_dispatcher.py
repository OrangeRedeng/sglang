"""Run on Ascend to check local EP routing against a replicated expert reference."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.ascend_tp import (
    AscendTPCombineInput,
    AscendTPDispatcher,
)
from sglang.test.test_utils import CustomTestCase


class TestAllreduceDeepEPDispatcher(CustomTestCase):
    def test_local_outputs_sum_to_replicated_expert_output(self):
        x = torch.arange(4 * 128, device="npu", dtype=torch.float32).reshape(4, 128)
        x = (x / 512).to(torch.bfloat16)
        ids = torch.tensor([[0, 3], [1, 2], [0, 1], [2, 3]], device="npu")
        weights = torch.tensor([[0.25, 0.75]] * 4, device="npu", dtype=x.dtype)
        outputs = []
        # Rank 2 has no selected experts, including for the entire batch.
        for rank in range(3):
            with patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_tp.get_parallel",
                return_value=SimpleNamespace(moe_tp_size=1, moe_ep_rank=rank),
            ):
                dispatcher = AscendTPDispatcher(
                    MoeRunnerConfig(
                        num_experts=6,
                        num_local_experts=2,
                        top_k=2,
                        num_fused_shared_experts=0,
                    ),
                    local_ep=True,
                )
            dispatched = dispatcher.dispatch(x, (weights, ids, None))
            expert_output = torch.full_like(dispatched.hidden_states, float("nan"))
            offset = 0
            for local, count in enumerate(dispatched.expert_tokens.cpu().tolist()):
                # Distinct expert functions expose incorrect expert ordering.
                expert_output[offset : offset + count] = dispatched.hidden_states[
                    offset : offset + count
                ] * (rank * 2 + local + 1)
                offset += count
            outputs.append(dispatcher.combine(AscendTPCombineInput(expert_output)))
        expected = x * (weights * (ids + 1)).sum(dim=1, keepdim=True)
        torch.testing.assert_close(sum(outputs), expected, rtol=0.02, atol=0.02)
        self.assertEqual(torch.count_nonzero(outputs[2]).item(), 0)


if __name__ == "__main__":
    unittest.main()
