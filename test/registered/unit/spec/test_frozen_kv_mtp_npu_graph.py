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
            cpu_update_input=[{"actual_seq_kvlen": lengths + [0, 0]}],
        )


@pytest.mark.parametrize("operator_count", [4, 12, 24])
def test_inspection_is_side_effect_free_and_replay_updates_every_operator(
    operator_count,
):
    runner = draft_runner()
    key = ShapeKey(size=4)
    records = [None] * operator_count

    def update(*, cpu_update_input):
        # torch_npu accepts a broadcast entry or exactly one per captured op.
        if len(cpu_update_input) not in (1, operator_count):
            raise RuntimeError("Update count must match recorded operators")
        entries = (
            cpu_update_input * operator_count
            if len(cpu_update_input) == 1
            else cpu_update_input
        )
        records[:] = [list(entry["actual_seq_kvlen"]) for entry in entries]

    graph = SimpleNamespace(
        graph_dispatch_mode=SimpleNamespace(graph_dispatch_records=records),
        update=Mock(side_effect=update),
    )

    def replay(shape_key, *, seq_lens, cpu_update_input):
        assert shape_key == key
        assert seq_lens is None
        graph.update(cpu_update_input=cpu_update_input)
        return "replayed"

    runner.backend = SimpleNamespace(
        _graphs={key: graph}, replay_with_input_update=replay
    )
    runner._check_graph_update_records()
    assert key in runner._updatable_graphs
    assert records == [None] * operator_count
    graph.update.assert_not_called()
    for lengths in ([7, 1030], [2929, 19]):
        batch = SimpleNamespace(seq_lens_cpu=torch.tensor(lengths + [0, 0]))
        assert runner._replay_graph(key, batch) == "replayed"
        assert records == [lengths + [0, 0]] * operator_count


def test_recordless_draft_bucket_falls_back_to_eager():
    runner = draft_runner()
    missing, supported = ShapeKey(size=2), ShapeKey(size=4)
    runner.backend._graphs = {
        missing: SimpleNamespace(
            graph_dispatch_mode=SimpleNamespace(graph_dispatch_records=[]),
            update=Mock(),
        ),
        supported: SimpleNamespace(
            graph_dispatch_mode=SimpleNamespace(graph_dispatch_records=[object()]),
            update=Mock(),
        ),
    }
    runner._check_graph_update_records()
    assert runner._updatable_graphs == {supported}
    with patch.object(FrozenKVMTPCudaGraphRunner, "can_run_graph", return_value=True):
        assert not runner.can_run_graph(
            SimpleNamespace(batch_size=2, seq_lens_cpu=torch.tensor([8, 9]))
        )
        assert runner.can_run_graph(
            SimpleNamespace(batch_size=3, seq_lens_cpu=torch.tensor([8, 9, 10]))
        )
        assert not runner.can_run_graph(SimpleNamespace(seq_lens_cpu=None))


def test_empty_dispatch_records_are_rejected_even_when_broadcast_update_is_a_noop():
    runner = draft_runner()
    graph = SimpleNamespace(
        graph_dispatch_mode=SimpleNamespace(graph_dispatch_records=[]), update=Mock()
    )
    runner.backend._graphs = {ShapeKey(size=2): graph}
    runner._check_graph_update_records()
    assert not runner._updatable_graphs
    graph.update.assert_not_called()


def test_unexpected_replay_update_failure_is_not_hidden():
    runner = draft_runner()
    runner.backend.replay_with_input_update.side_effect = RuntimeError("bad shape")
    with pytest.raises(RuntimeError, match="bad shape"):
        runner._replay_graph(
            ShapeKey(size=4), SimpleNamespace(seq_lens_cpu=torch.tensor([8, 9, 0, 0]))
        )


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
    graph = SimpleNamespace(
        graph_dispatch_mode=SimpleNamespace(graph_dispatch_records=[]), update=Mock()
    )
    runner.backend = SimpleNamespace(_graphs={ShapeKey(size=2): graph})
    with patch.object(DecodeCudaGraphRunner, "capture_one_shape"):
        runner.capture_one_shape(2, Mock())
    assert not runner.can_run_graph(Mock())
    graph.update.assert_not_called()


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


@pytest.mark.parametrize("has_records", [False, True])
def test_verify_capture_only_inspects_records_without_signaling_graph_events(
    has_records,
):
    runner = object.__new__(NPUGraphRunner)
    runner._frozen_graph_updates_available = True
    runner.model_runner = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_frozen_kv_mtp=lambda: True)
    )
    runner._make_graph_key = lambda size: ShapeKey(size=size)
    graph = SimpleNamespace(
        graph_dispatch_mode=SimpleNamespace(
            graph_dispatch_records=[object()] if has_records else []
        ),
        update=Mock(),
    )
    runner.backend = SimpleNamespace(_graphs={ShapeKey(size=2): graph})
    with patch.object(DecodeCudaGraphRunner, "capture_one_shape"):
        runner.capture_one_shape(2, Mock())
    assert runner._frozen_graph_updates_available is has_records
    graph.update.assert_not_called()


def test_npu_draft_graph_can_be_disabled_without_disabling_verify_configuration():
    from sglang.srt.environ import envs
    from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import FrozenKVMTPDraftWorker

    worker = SimpleNamespace(
        target_worker=SimpleNamespace(device="npu"),
        speculative_num_steps=3,
        cuda_graph_runner=None,
    )
    with (
        envs.SGLANG_DISABLE_NPU_MTP_DRAFT_GRAPH.override(True),
        patch(
            "sglang.srt.speculative.frozen_kv_mtp_worker_v2.cuda_graph_fully_disabled",
            return_value=False,
        ),
        patch(
            "sglang.srt.hardware_backend.npu.graph_runner."
            "frozen_kv_mtp_npu_graph_runner.FrozenKVMTPNpuGraphRunner"
        ) as factory,
    ):
        FrozenKVMTPDraftWorker._capture_cuda_graphs(worker)
    assert worker.cuda_graph_runner is None
    factory.assert_not_called()


def test_npu_verify_graph_can_be_disabled_before_capture():
    from sglang.srt.environ import envs
    from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
        capture_decode_graph,
    )

    runner = SimpleNamespace(
        device="npu",
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(
            is_speculative=lambda: True,
            is_frozen_kv_mtp=lambda: True,
        ),
    )
    with envs.SGLANG_DISABLE_NPU_MTP_VERIFY_GRAPH.override(True):
        result = capture_decode_graph(model_runner=runner)
    assert result.runner is None
    assert result.memory_phase == "target_verify"
    assert result.memory_usage_gb == 0
