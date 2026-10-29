"""CPU orchestration checks for frozen-prefix NPU graph replay."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from sglang.srt.hardware_backend.npu.graph_runner.frozen_kv_mtp_npu_graph_runner import (
    FrozenKVMTPNpuGraphRunner,
)
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner import DecodeCudaGraphRunner, ShapeKey
from sglang.srt.speculative.frozen_kv_mtp_cuda_graph_runner import (
    FrozenKVMTPCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def draft_runner():
    runner = object.__new__(FrozenKVMTPNpuGraphRunner)
    runner.speculative_num_steps = 3
    runner.capture_bs = [1, 2, 4]
    runner.raw_bs = 2
    runner.bs = 4
    runner.backend = Mock()
    return runner


def test_draft_rebinds_unchanged_prefix_for_seed_and_each_recurrent_step():
    runner = draft_runner()
    for lengths in ([7, 1030], [2929, 19]):
        batch = SimpleNamespace(seq_lens_cpu=torch.tensor(lengths + [0, 0]))
        runner._replay_graph(ShapeKey(size=4), batch)
        runner.backend.replay_with_input_update.assert_called_with(
            ShapeKey(size=4),
            seq_lens=None,
            cpu_update_input=[{"actual_seq_kvlen": lengths + [0, 0]}] * 3,
        )


def test_recordless_draft_bucket_falls_back_to_eager():
    runner = draft_runner()
    missing, supported = ShapeKey(size=2), ShapeKey(size=4)
    runner.backend._graphs = {
        missing: Mock(update=Mock(side_effect=RuntimeError("0 operators"))),
        supported: Mock(),
    }
    runner._probe_graph_updates()
    assert runner._updatable_graphs == {supported}
    with patch.object(FrozenKVMTPCudaGraphRunner, "can_run_graph", return_value=True):
        assert not runner.can_run_graph(
            SimpleNamespace(batch_size=2, seq_lens_cpu=torch.tensor([8, 9]))
        )
        assert runner.can_run_graph(
            SimpleNamespace(batch_size=3, seq_lens_cpu=torch.tensor([8, 9, 10]))
        )
        assert not runner.can_run_graph(SimpleNamespace(seq_lens_cpu=None))


def test_unexpected_update_failure_is_not_hidden():
    runner = draft_runner()
    runner.backend._graphs = {
        ShapeKey(size=2): Mock(update=Mock(side_effect=RuntimeError("bad shape")))
    }
    with pytest.raises(RuntimeError, match="bad shape"):
        runner._probe_graph_updates()


def test_verify_selects_v2_host_attribute_for_frozen_mtp():
    runner = object.__new__(NPUGraphRunner)
    runner.is_dllm = False
    runner.if_use_v2 = False
    runner.use_fias_v2_bsnd = False
    runner.use_frozen_kv_mtp = True
    runner.capture_forward_mode = ForwardMode.TARGET_VERIFY
    runner._init_arch_map()
    assert runner._get_update_attr_name() == "actual_seq_kvlen"
    assert runner._get_update_attr_type() == []
    runner.capture_forward_mode = ForwardMode.DECODE
    assert runner._get_update_attr_name() == "actual_seq_lengths_kv"


def test_recordless_verify_capture_prevents_replay():
    runner = object.__new__(NPUGraphRunner)
    runner._frozen_graph_updates_available = True
    runner.model_runner = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_frozen_kv_mtp=lambda: True)
    )
    runner.captured_req_width = 4
    runner._make_graph_key = lambda size: ShapeKey(size=size)
    graph = Mock(update=Mock(side_effect=RuntimeError("0 operators")))
    runner.backend = SimpleNamespace(_graphs={ShapeKey(size=2): graph})
    with patch.object(DecodeCudaGraphRunner, "capture_one_shape"):
        runner.capture_one_shape(2, Mock())
    assert not runner.can_run_graph(Mock())
    graph.update.assert_called_once_with(
        cpu_update_input=[{"actual_seq_kvlen": [4, 4]}]
    )


def test_worker_selects_npu_draft_graph_runner():
    from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import FrozenKVMTPDraftWorker

    worker = SimpleNamespace(
        target_worker=SimpleNamespace(device="npu"),
        device="npu",
        gpu_id=0,
        topk=1,
        speculative_num_steps=3,
        cuda_graph_runner=None,
        _specialized_graph_memory_usage={},
        _specialized_graph_time_usage={},
    )
    config = SimpleNamespace(
        graph=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(backend="full"))
        )
    )
    with (
        patch(
            "sglang.srt.speculative.frozen_kv_mtp_worker_v2.cuda_graph_fully_disabled",
            return_value=False,
        ),
        patch("sglang.srt.runtime_context.get_exec", return_value=config),
        patch(
            "sglang.srt.speculative.frozen_kv_mtp_worker_v2.get_available_gpu_memory",
            side_effect=[10, 9],
        ),
        patch(
            "sglang.srt.hardware_backend.npu.graph_runner."
            "frozen_kv_mtp_npu_graph_runner.FrozenKVMTPNpuGraphRunner"
        ) as factory,
    ):
        FrozenKVMTPDraftWorker._capture_cuda_graphs(worker)
    factory.assert_called_once_with(worker)
    assert worker.cuda_graph_runner is factory.return_value
    assert worker._specialized_graph_memory_usage["draft_decode"] == 1
