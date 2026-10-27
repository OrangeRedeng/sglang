"""CPU coverage of queue transition ordering and unsupported runtime guards."""

import sys
import threading
from types import SimpleNamespace

import pytest

from sglang.srt.hardware_backend.npu import task_queue
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class FakeNPU:
    def __init__(self):
        self.mode = 1
        self.capturing = False
        self.calls = []

    def get_task_queue_enable(self):
        return self.mode

    def is_current_stream_capturing(self):
        return self.capturing

    def synchronize(self):
        self.calls.append("sync")

    def set_task_queue_enable(self, mode):
        self.calls.append(mode)
        self.mode = mode


def test_switches_only_on_phase_transitions():
    npu = FakeNPU()
    policy = task_queue.NPUTaskQueuePolicy(npu)
    for mode in (
        ForwardMode.EXTEND,
        ForwardMode.EXTEND,
        ForwardMode.MIXED,
        ForwardMode.DECODE,
        ForwardMode.DECODE,
        ForwardMode.IDLE,
        ForwardMode.EXTEND,
    ):
        policy.before_forward(mode)
    assert npu.calls == ["sync", 0, "sync", 1, "sync", 0]


def test_capture_and_replay_share_phase_modes():
    npu = FakeNPU()
    policy = task_queue.NPUTaskQueuePolicy(npu)
    policy.set_mode(0)
    policy.set_mode(1)
    npu.calls.clear()
    policy.before_forward(ForwardMode.DECODE)
    assert npu.calls == []
    policy.before_forward(ForwardMode.EXTEND)
    assert npu.calls == ["sync", 0]


def test_rejects_mode_change_inside_capture():
    npu = FakeNPU()
    policy = task_queue.NPUTaskQueuePolicy(npu)
    npu.capturing = True
    with pytest.raises(RuntimeError, match="inside graph capture"):
        policy.set_mode(0)
    assert npu.calls == []


def test_rejects_submission_from_another_thread():
    npu = FakeNPU()
    policy = task_queue.NPUTaskQueuePolicy(npu)
    errors = []

    def submit():
        try:
            policy.before_forward(ForwardMode.EXTEND)
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=submit)
    thread.start()
    thread.join()
    assert errors == ["Adaptive NPU task queues require one submission thread"]
    assert npu.calls == []


def test_failed_switch_is_not_cached(monkeypatch):
    npu = FakeNPU()
    policy = task_queue.NPUTaskQueuePolicy(npu)
    with monkeypatch.context() as patch:
        patch.setattr(npu, "set_task_queue_enable", lambda mode: None)
        with pytest.raises(RuntimeError, match="did not apply"):
            policy.set_mode(0)
    policy.set_mode(0)
    assert npu.calls == ["sync", "sync", 0]


def test_missing_runtime_api_fails_early():
    with pytest.raises(ValueError, match="runtime queue setters"):
        task_queue.NPUTaskQueuePolicy(SimpleNamespace())


@pytest.fixture
def runtime(monkeypatch):
    execution = SimpleNamespace(
        graph=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="disabled"),
                decode=SimpleNamespace(backend="full"),
            )
        ),
        offload=SimpleNamespace(cpu_offload_gb=0, offload_group_size=-1),
        overlap=SimpleNamespace(enable_two_batch_overlap=False),
    )
    spec = SimpleNamespace(speculative_algorithm=None)
    parallel = SimpleNamespace(pp_size=1)
    memory = SimpleNamespace(enable_hierarchical_cache=False)
    disagg = SimpleNamespace(disaggregation_mode="null")
    for name, value in (
        ("get_exec", execution),
        ("get_spec", spec),
        ("get_parallel", parallel),
        ("get_memory", memory),
        ("get_disagg", disagg),
    ):
        monkeypatch.setattr(task_queue, name, lambda value=value: value)
    monkeypatch.delenv("ASCEND_LAUNCH_BLOCKING", raising=False)
    return execution, spec, parallel, memory, disagg


def test_opt_in_and_non_npu_gate_without_loading_torch_npu():
    with task_queue.envs.SGLANG_ENABLE_NPU_ADAPTIVE_TASK_QUEUE.override(False):
        assert (
            task_queue.NPUTaskQueuePolicy.create(device="npu", is_draft_worker=False)
            is None
        )
    with task_queue.envs.SGLANG_ENABLE_NPU_ADAPTIVE_TASK_QUEUE.override(True):
        assert (
            task_queue.NPUTaskQueuePolicy.create(device="cuda", is_draft_worker=False)
            is None
        )


@pytest.mark.parametrize(
    "configuration",
    [
        "draft",
        "speculative",
        "pp",
        "tbo",
        "hicache",
        "offload",
        "group_offload",
        "pd",
        "blocking",
    ],
)
def test_unsupported_configurations_fail_early(runtime, monkeypatch, configuration):
    execution, spec, parallel, memory, disagg = runtime
    if configuration == "speculative":
        spec.speculative_algorithm = "EAGLE3"
    elif configuration == "pp":
        parallel.pp_size = 2
    elif configuration == "tbo":
        execution.overlap.enable_two_batch_overlap = True
    elif configuration == "hicache":
        memory.enable_hierarchical_cache = True
    elif configuration == "offload":
        execution.offload.cpu_offload_gb = 1
    elif configuration == "group_offload":
        execution.offload.offload_group_size = 2
    elif configuration == "pd":
        disagg.disaggregation_mode = "prefill"
    elif configuration == "blocking":
        monkeypatch.setenv("ASCEND_LAUNCH_BLOCKING", "1")
    with (
        task_queue.envs.SGLANG_ENABLE_NPU_ADAPTIVE_TASK_QUEUE.override(True),
        pytest.raises(ValueError),
    ):
        task_queue.NPUTaskQueuePolicy.create(
            device="npu", is_draft_worker=configuration == "draft"
        )


def test_supported_runtime_uses_public_api(runtime, monkeypatch):
    npu = FakeNPU()
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu=npu))
    with task_queue.envs.SGLANG_ENABLE_NPU_ADAPTIVE_TASK_QUEUE.override(True):
        policy = task_queue.NPUTaskQueuePolicy.create(
            device="npu", is_draft_worker=False
        )
    policy.before_forward(ForwardMode.EXTEND)
    assert npu.calls == ["sync", 0]


def test_eager_prefill_two_decode_zero(runtime, monkeypatch):
    npu = FakeNPU()
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu=npu))
    with (
        task_queue.envs.SGLANG_ENABLE_NPU_ADAPTIVE_TASK_QUEUE.override(True),
        task_queue.envs.SGLANG_NPU_PREFILL_TASK_QUEUE_MODE.override(2),
        task_queue.envs.SGLANG_NPU_DECODE_TASK_QUEUE_MODE.override(0),
    ):
        policy = task_queue.NPUTaskQueuePolicy.create(
            device="npu", is_draft_worker=False
        )
    for mode in (ForwardMode.EXTEND, ForwardMode.EXTEND, ForwardMode.DECODE):
        policy.before_forward(mode)
    assert npu.calls == ["sync", 2, "sync", 0]


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_mode_two_requires_graphs_disabled(runtime, phase):
    execution = runtime[0]
    getattr(execution.graph.cuda_graph_config, phase).backend = "full"
    setting = getattr(task_queue.envs, f"SGLANG_NPU_{phase.upper()}_TASK_QUEUE_MODE")
    with (
        task_queue.envs.SGLANG_ENABLE_NPU_ADAPTIVE_TASK_QUEUE.override(True),
        setting.override(2),
        pytest.raises(ValueError, match=f"requires {phase} graphs disabled"),
    ):
        task_queue.NPUTaskQueuePolicy.create(device="npu", is_draft_worker=False)


@pytest.mark.parametrize("modes", [(3, 0), (0, -1)])
def test_rejects_invalid_modes(modes):
    with pytest.raises(ValueError, match="must be 0, 1, or 2"):
        task_queue.NPUTaskQueuePolicy(
            FakeNPU(), prefill_mode=modes[0], decode_mode=modes[1]
        )
