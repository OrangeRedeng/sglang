"""Frozen MTP must accept the scheduler's shared forward-call arguments."""

import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.speculative import frozen_kv_mtp_worker_v2 as worker_mod
from sglang.srt.speculative.frozen_kv_mtp_info import FrozenKVMTPDraftInput
from sglang.srt.speculative.frozen_kv_mtp_utils import set_frozen_kv_positions
from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import (
    FrozenKVMTPDraftWorker,
    FrozenKVMTPWorkerV2,
)


class TestFrozenKVMTPForward(CustomTestCase):
    def contexts(self):
        stack = ExitStack()
        for name in (
            "draft_tp_context",
            "speculative_moe_backend_context",
            "speculative_moe_a2a_backend_context",
            "spec_stage_span",
        ):
            stack.enter_context(
                patch(
                    "sglang.srt.speculative.frozen_kv_mtp_worker_v2." + name,
                    side_effect=lambda *args, **kwargs: nullcontext(),
                )
            )
        return stack

    def test_recurrent_draft_reuses_seed_positions_without_advancing_prefix(self):
        for lengths in ([7, 9], [1, 1025]):
            with self.subTest(lengths=lengths):
                spec = object.__new__(FrozenKVMTPDraftInput)
                spec.bonus_tokens = torch.tensor([2, 3])
                spec.hidden_states = torch.ones(2, 4)
                batch = SimpleNamespace(
                    spec_info=spec,
                    seq_lens=torch.tensor(lengths),
                    positions=torch.zeros(2, dtype=torch.int64),
                    needs_forward_metadata_init=lambda: False,
                )
                observations = []

                def forward(fb, observations=observations):
                    observations.append(fb.positions.clone())
                    return SimpleNamespace(
                        logits_output=SimpleNamespace(
                            next_token_logits=torch.tensor(
                                [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]
                            ),
                            hidden_states=fb.spec_info.hidden_states + 1,
                        )
                    )

                def choose(i, probs, indices, hidden, scores, topk):
                    return indices.flatten(), hidden, probs, (probs, indices, indices)

                worker = SimpleNamespace(
                    topk=1,
                    speculative_num_steps=3,
                    speculative_num_draft_tokens=4,
                    draft_attn_backend=object(),
                    draft_model_runner=SimpleNamespace(forward=forward),
                    _set_positions=Mock(
                        side_effect=lambda fb: set_frozen_kv_positions(fb, 1)
                    ),
                    _target_kv_pool_view=lambda fb: nullcontext(),
                )
                with (
                    patch.object(worker_mod, "select_top_k_tokens", side_effect=choose),
                    patch.object(worker_mod, "fast_topk", side_effect=torch.topk),
                    patch.object(
                        worker_mod, "organize_draft_results", return_value="tree"
                    ),
                ):
                    result = FrozenKVMTPDraftWorker.draft_forward(worker, batch)
                self.assertEqual(result, "tree")
                self.assertEqual(len(observations), 3)
                for positions in observations:
                    torch.testing.assert_close(positions, torch.tensor(lengths) - 1)
                torch.testing.assert_close(batch.seq_lens, torch.tensor(lengths))
                worker._set_positions.assert_called_once_with(batch)

    def test_draft_uses_cpu_lengths_and_retains_device_fallback(self):
        for use_cpu in (True, False):
            with self.subTest(use_cpu=use_cpu):
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.DECODE,
                    spec_info=object.__new__(FrozenKVMTPDraftInput),
                    seq_lens_cpu=torch.tensor([7, 10]) if use_cpu else None,
                    seq_lens=object() if use_cpu else torch.tensor([7, 10]),
                )
                worker = SimpleNamespace(
                    topk=1,
                    _position_for_batch=Mock(),
                    draft_model_runner=object(),
                )
                with (
                    patch(
                        "sglang.srt.speculative.frozen_kv_mtp_worker_v2.ForwardBatch.init_new",
                        side_effect=RuntimeError("stop before device forward"),
                    ) as init,
                    self.assertRaisesRegex(RuntimeError, "stop before device"),
                ):
                    FrozenKVMTPDraftWorker.draft(worker, batch)
                init.assert_called_once()
                self.assertEqual(batch.seq_lens_sum, 17)

    def test_prefill_accepts_scheduler_proxy_argument(self):
        for proxy in (None, object()):
            with self.subTest(proxy=proxy):
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND,
                    is_extend_in_batch=False,
                    seq_lens=object(),
                )
                output = SimpleNamespace(
                    logits_output=SimpleNamespace(
                        hidden_states=object(), mm_input_embeds=None
                    ),
                    next_token_ids=object(),
                )
                worker = SimpleNamespace(
                    target_worker=Mock(), draft_worker=Mock(draft_owns_attention=False)
                )
                worker.target_worker.forward_batch_generation.return_value = output
                publish = Mock()
                with self.contexts():
                    result = FrozenKVMTPWorkerV2.forward_batch_generation(
                        worker, batch, on_publish=publish, pp_proxy_tensors=proxy
                    )
                self.assertIs(result, output)
                worker.target_worker.forward_batch_generation.assert_called_once_with(
                    batch,
                    pp_proxy_tensors=proxy,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                )
                publish.assert_called_once_with(batch.seq_lens)
                worker.draft_worker._draft_extend_for_prefill.assert_called_once_with(
                    batch,
                    output.logits_output.hidden_states,
                    output.next_token_ids,
                    None,
                )
                self.assertIs(
                    result.next_draft_input,
                    worker.draft_worker._draft_extend_for_prefill.return_value,
                )

    def test_decode_forwards_proxy_and_grammar_to_verify(self):
        for proxy in (None, object()):
            with self.subTest(proxy=proxy):
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.DECODE,
                    is_extend_in_batch=False,
                    seq_lens=SimpleNamespace(shape=(2,)),
                    spec_info=object(),
                )
                output = SimpleNamespace(new_seq_lens=object())
                verify_input = Mock()
                verify_input.is_verify_input.return_value = True
                worker = SimpleNamespace(
                    activate_step_by_batch=Mock(),
                    draft_worker=Mock(draft_owns_attention=False),
                    verify=Mock(return_value=output),
                )
                worker.draft_worker.draft.return_value = verify_input
                grammar = object()
                publish = Mock()
                with self.contexts():
                    result = FrozenKVMTPWorkerV2.forward_batch_generation(
                        worker,
                        batch,
                        on_publish=publish,
                        grammar_barrier=grammar,
                        pp_proxy_tensors=proxy,
                    )
                self.assertIs(result, output)
                self.assertIs(batch.spec_info, verify_input)
                worker.verify.assert_called_once_with(
                    batch, pp_proxy_tensors=proxy, grammar_barrier=grammar
                )
                publish.assert_called_once_with(output.new_seq_lens)
                worker.draft_worker._draft_extend_for_decode.assert_called_once_with(
                    batch, output
                )


if __name__ == "__main__":
    unittest.main()
