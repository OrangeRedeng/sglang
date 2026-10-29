"""NPU piecewise prefill gates and frozen-assistant hidden-state handoff."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from sglang.srt.arg_groups import cuda_graph_hook
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.cuda_graph_config import Backend, CudaGraphConfig
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.model_executor.model_runner_components import cuda_graph_setup
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.frozen_kv_mtp_utils import select_last_extend_hidden
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def config(**changes):
    values = {
        "cuda_graph_config": CudaGraphConfig.from_dict(
            {
                "prefill": {"backend": "tc_piecewise", "bs": [4]},
                "decode": {"backend": "full"},
            }
        ),
        "device": "npu",
        "pp_size": 1,
        "dcp_size": 1,
        "attn_cp_size": 1,
        "enable_dp_attention": False,
        "enable_lora": False,
        "speculative_algorithm": "FROZEN_KV_MTP",
    }
    return SimpleNamespace(**(values | changes))


def validate(cfg):
    with (
        patch.object(cuda_graph_hook, "resolving_view", return_value=cfg),
        patch.object(cuda_graph_hook, "resolved_view", return_value=cfg),
        patch.object(
            cuda_graph_hook, "attention_backends_of", return_value=("ascend", "ascend")
        ),
    ):
        cuda_graph_hook.validate_cuda_graph_config(cfg)


def test_frozen_mtp_piecewise_prefill_is_allowed():
    validate(config())


@pytest.mark.parametrize("algorithm", ["NEXTN", "EAGLE"])
@pytest.mark.parametrize("assistant", [True, False])
def test_graph_gate_resolves_assistant_alias_before_speculative_hook(
    algorithm, assistant
):
    architecture = (
        "Gemma4AssistantForCausalLM" if assistant else "OtherDraftForCausalLM"
    )
    with patch(
        "sglang.srt.utils.hf_transformers_utils.get_config",
        return_value=SimpleNamespace(architectures=[architecture]),
    ):
        cfg = config(
            speculative_algorithm=algorithm,
            speculative_draft_model_path="/models/assistant",
        )
        if assistant:
            validate(cfg)
        else:
            with pytest.raises(ValueError, match="other than FROZEN_KV_MTP"):
                validate(cfg)


@pytest.mark.parametrize("algorithm", ["EAGLE", "EAGLE3", "DFLASH", "NGRAM", "NEXTN"])
def test_other_speculative_piecewise_paths_remain_rejected(algorithm):
    with pytest.raises(ValueError, match="other than FROZEN_KV_MTP"):
        validate(config(speculative_algorithm=algorithm))


@pytest.mark.parametrize(
    "changes",
    [
        {"pp_size": 2},
        {"dcp_size": 2},
        {"attn_cp_size": 2},
        {"enable_dp_attention": True},
        {"enable_lora": True},
    ],
)
def test_topology_and_lora_guards_remain(changes):
    with pytest.raises(ValueError, match="does not support"):
        validate(config(**changes))


def test_frozen_target_reaches_capture_without_server_hidden_state_flag():
    override = get_context().override_server_args(
        enable_return_hidden_states=False,
        enable_lora=False,
        enable_prefill_cp=False,
        pp_size=1,
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(backend=Backend.TC_PIECEWISE, bs=[4])
        ),
    )
    override.install()
    try:
        runner = SimpleNamespace(
            device="npu",
            gpu_id=0,
            is_draft_worker=False,
            lora_manager=None,
            spec_algorithm=SpeculativeAlgorithm.FROZEN_KV_MTP,
            model=SimpleNamespace(),
            server_args=SimpleNamespace(),
            model_config=SimpleNamespace(context_len=8192, num_hidden_layers=1),
            layer_info=SimpleNamespace(start_layer=0, end_layer=1),
            req_to_token_pool=SimpleNamespace(size=2),
            get_cuda_graph_layers=lambda _: ([object()], [], [], [], [None]),
        )
        captured = object()
        with (
            patch.object(
                cuda_graph_setup,
                "check_cuda_graph_backend",
                side_effect=lambda phase, backend: backend == Backend.TC_PIECEWISE,
            ),
            patch.object(
                cuda_graph_setup,
                "resolve_language_model",
                return_value=SimpleNamespace(layers=[object()]),
            ),
            patch.object(
                cuda_graph_setup, "get_available_gpu_memory", side_effect=[4, 3]
            ),
            patch.object(
                cuda_graph_setup, "PrefillCudaGraphRunner", return_value=captured
            ) as factory,
        ):
            result = cuda_graph_setup.capture_prefill_graph(
                model_runner=runner, eager_runner=object()
            )
        assert result.runner is captured
        factory.assert_called_once_with(runner)
    finally:
        override.restore()


def test_frozen_assistant_does_not_capture_a_nonexistent_prefill_forward():
    runner = SimpleNamespace(
        device="npu",
        is_draft_worker=True,
        spec_algorithm=SpeculativeAlgorithm.FROZEN_KV_MTP,
    )
    with patch.object(
        cuda_graph_setup,
        "check_cuda_graph_backend",
        side_effect=lambda phase, backend: backend == Backend.TC_PIECEWISE,
    ):
        result = cuda_graph_setup.capture_prefill_graph(
            model_runner=runner,
            eager_runner=object(),
            force_for_draft_worker=True,
        )
    assert result.runner is None


def test_piecewise_output_preserves_all_target_hidden_rows_for_assistant_seed():
    runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
    runner.raw_num_tokens = 5
    runner.raw_bs = 2
    runner._is_full_backend = False
    runner.model_runner = SimpleNamespace(
        spec_algorithm=SpeculativeAlgorithm.FROZEN_KV_MTP
    )
    runner.capture_hidden_mode = CaptureHiddenMode.FULL
    runner._validate_capture_hidden_mode(
        SimpleNamespace(capture_hidden_mode=CaptureHiddenMode.FULL)
    )
    for offset in (0, 100):
        hidden = torch.arange(8 * 4).reshape(8, 4) + offset
        output = runner._trim_logits_output(
            LogitsProcessorOutput(
                next_token_logits=torch.zeros(2, 3),
                hidden_states=hidden,
                mm_input_embeds=hidden.clone(),
            )
        )
        assert output.hidden_states.shape == (5, 4)
        assert output.mm_input_embeds.shape == (5, 4)
        batch = SimpleNamespace(batch_size=lambda: 2, extend_lens=[2, 3])
        seed = select_last_extend_hidden(batch, output.hidden_states)
        torch.testing.assert_close(seed, hidden[[1, 4]])
