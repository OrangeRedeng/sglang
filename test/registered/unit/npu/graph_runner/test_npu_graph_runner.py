"""CPU coverage for NPU decode-graph replay orchestration."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.hardware_backend.npu.graph_runner import npu_graph_runner as mod
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_replay_updates_sequence_lengths_and_slices_logits_to_raw_batch():
    runner = mod.NPUGraphRunner.__new__(mod.NPUGraphRunner)
    runner.require_mlp_tp_gather = False
    runner.capture_bs = [2]
    runner.captured_req_width = 1
    runner.deepep_adapter = SimpleNamespace(replay=mock.Mock())
    runner.buffers = SimpleNamespace(
        input_ids=torch.empty(2, dtype=torch.int64),
        positions=torch.empty(2, dtype=torch.int64),
        seq_lens=torch.empty(2, dtype=torch.int32),
        seq_lens_cpu=torch.empty(2, dtype=torch.int32),
        req_pool_indices=torch.empty(2, dtype=torch.int32),
    )
    runner.model_runner = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_dflash=lambda: True),
        is_draft_worker=False,
        model_config=SimpleNamespace(hf_config=object()),
    )
    runner.seq_len_fill_value = 0
    runner.capture_forward_mode = object()
    runner.is_encoder_decoder = False
    runner._pad_to_bucket = lambda raw_bs, buckets: raw_bs
    runner._make_graph_key = lambda bs: ("graph", bs)
    attention = SimpleNamespace(init_forward_metadata_out_graph=mock.Mock())
    runner._replay_attn_backend = lambda: attention
    runner.backend = SimpleNamespace(
        replay_with_input_update=mock.Mock(
            return_value=LogitsProcessorOutput(
                next_token_logits=torch.arange(3, dtype=torch.float32).view(3, 1),
                hidden_states=torch.arange(3, dtype=torch.float32).view(3, 1),
            )
        )
    )
    runner.is_dllm = False
    runner.if_use_v2 = False
    runner.use_fias_v2_bsnd = False
    runner.capture_forward_mode = SimpleNamespace(is_target_verify=lambda: False)
    runner._init_arch_map()

    forward_batch = SimpleNamespace(
        needs_forward_metadata_init=lambda: False,
        batch_size=2,
        input_ids=torch.tensor([4, 5]),
        positions=torch.tensor([8, 9]),
        seq_lens=torch.tensor([10, 11]),
        seq_lens_cpu=torch.tensor([10, 11], dtype=torch.int32),
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        out_cache_loc=None,
        input_embeds=None,
        mrope_positions=None,
        forward_mode=SimpleNamespace(is_target_verify=lambda: False),
    )

    with (
        mock.patch.object(mod, "enable_num_token_non_padded", return_value=False),
        mock.patch.object(
            mod.envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM, "get", return_value=False
        ),
        mock.patch.object(mod, "build_replay_fb_view", return_value="replay-view"),
        mock.patch.object(mod, "is_deepseek_dsa", return_value=False),
        mock.patch.object(mod, "is_deepseek_v4", return_value=False),
    ):
        output = runner.execute(forward_batch)

    runner.deepep_adapter.replay.assert_called_once_with()
    attention.init_forward_metadata_out_graph.assert_called_once_with("replay-view")
    runner.backend.replay_with_input_update.assert_called_once_with(
        ("graph", 2),
        seq_lens=[10, 11],
        attr_name=runner._get_update_attr_name(),
        attr_type=runner._get_update_attr_type(),
    )
    torch.testing.assert_close(output.next_token_logits, torch.tensor([[0.0], [1.0]]))
    torch.testing.assert_close(output.hidden_states, torch.tensor([[0.0], [1.0]]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_frozen_verify_rebinds_from_cpu_prefix_without_device_to_host_sync():
    runner = mod.NPUGraphRunner.__new__(mod.NPUGraphRunner)
    runner.use_frozen_kv_mtp = True
    runner.use_fias_v2_bsnd = False
    runner.if_use_v2 = False
    runner.is_dllm = False
    runner.capture_forward_mode = SimpleNamespace(is_target_verify=lambda: True)
    runner.captured_req_width = 4
    runner.raw_bs = 2
    runner.bs = 4
    runner.raw_num_token = 8
    runner.require_mlp_tp_gather = False
    runner.load_batch = mock.Mock()
    runner.deepep_adapter = SimpleNamespace(replay=mock.Mock())
    runner.model_runner = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_dflash=lambda: False),
        model_config=SimpleNamespace(hf_config=object()),
    )
    runner._make_graph_key = lambda bs: ("graph", bs)
    runner._init_arch_map()
    runner.backend = SimpleNamespace(
        replay_with_input_update=mock.Mock(
            return_value=LogitsProcessorOutput(
                next_token_logits=torch.arange(16).view(16, 1),
                hidden_states=torch.arange(16).view(16, 1),
            )
        )
    )
    device_copy = mock.Mock(
        side_effect=AssertionError("unexpected D2H synchronization")
    )
    cpu_prefix = torch.tensor([100, 2900], dtype=torch.int32)
    batch = SimpleNamespace(
        needs_forward_metadata_init=lambda: True,
        forward_mode=SimpleNamespace(is_target_verify=lambda: True),
        seq_lens=SimpleNamespace(cpu=device_copy),
        seq_lens_cpu=cpu_prefix,
    )
    with (
        mock.patch.object(mod, "is_deepseek_dsa", return_value=False),
        mock.patch.object(mod, "is_deepseek_v4", return_value=False),
    ):
        output = runner.execute(batch)
    device_copy.assert_not_called()
    runner.backend.replay_with_input_update.assert_called_once_with(
        ("graph", 4),
        seq_lens=[104, 2904, 0, 0],
        attr_name="actual_seq_kvlen",
        attr_type=[],
    )
    assert cpu_prefix.tolist() == [100, 2900]
    assert output.next_token_logits.shape[0] == 8
    assert output.hidden_states.shape[0] == 8


def test_frozen_verify_without_cpu_prefix_uses_eager_fallback():
    runner = mod.NPUGraphRunner.__new__(mod.NPUGraphRunner)
    runner.use_frozen_kv_mtp = True
    assert not runner.can_run_graph(SimpleNamespace(seq_lens_cpu=None))
