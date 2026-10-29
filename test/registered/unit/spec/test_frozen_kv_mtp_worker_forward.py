"""Frozen MTP must accept the scheduler's shared forward-call arguments."""

import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import FrozenKVMTPWorkerV2


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
