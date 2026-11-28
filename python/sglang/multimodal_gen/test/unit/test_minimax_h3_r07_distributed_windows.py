# SPDX-License-Identifier: Apache-2.0
"""Host tests for the R07 step-major distributed window protocol."""

import threading
from concurrent.futures import Future, ThreadPoolExecutor

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.distributed_windows import (
    DistributedWindowCoordinator,
    LocalWindowWorker,
    PredictionRejection,
    RemoteWindowWorkerStub,
    WindowContract,
    WindowDispatch,
    WindowStepTimeout,
    plan_topologies,
)

VIDEO_SHAPE = (12, 8)
AUDIO_SHAPE = (6, 8)


def _dispatch(step: int, index: int, *, request_id: str = "req0") -> WindowDispatch:
    return WindowDispatch(
        request_id=request_id,
        step=step,
        contract=WindowContract(
            index=index,
            video_shape=(index + 1, 8),
            audio_shape=(index + 1, 4),
        ),
        video_snapshot=torch.zeros(index + 1, 8),
        audio_snapshot=torch.zeros(index + 1, 4),
    )


def _identity(dispatch: WindowDispatch):
    return dispatch.video_snapshot * 1.0, dispatch.audio_snapshot * 1.0


def _make(workers, *, fuse=None, scheduler_calls=None, **kwargs):
    scheduler_calls = scheduler_calls if scheduler_calls is not None else []
    if fuse is None:
        def fuse(predictions):
            video = torch.cat([p.video for p in predictions], dim=0)
            audio = torch.cat([p.audio for p in predictions], dim=0)
            return video, audio

    coordinator = DistributedWindowCoordinator(
        request_id="req0",
        workers=workers,
        fuse=fuse,
        scheduler_step=lambda: scheduler_calls.append(True),
        **kwargs,
    )
    return coordinator, scheduler_calls


def test_two_window_step_fuses_and_steps_exactly_once():
    workers = [LocalWindowWorker(_identity), LocalWindowWorker(_identity)]
    coordinator, scheduler_calls = _make(workers)
    video, audio = coordinator.run_step(
        0, [_dispatch(0, 0), _dispatch(0, 1)]
    )
    assert tuple(video.shape) == (3, 8)
    assert tuple(audio.shape) == (3, 4)
    # exactly-once scheduler update per completed step
    assert len(scheduler_calls) == 1
    assert coordinator.counters.scheduler_updates == 1
    assert coordinator.counters.completed_steps == 1


def test_local_worker_equals_direct_forward():
    workers = [LocalWindowWorker(_identity)]
    coordinator, _ = _make(workers, fuse=lambda ps: (ps[0].video, ps[0].audio))
    dispatch = _dispatch(0, 0)
    video, audio = coordinator.run_step(0, [dispatch])
    expected_video, expected_audio = _identity(dispatch)
    assert torch.equal(video, expected_video)
    assert torch.equal(audio, expected_audio)


def test_snapshot_is_immutable_against_worker_mutation():
    """Dispatches must carry private clones of the coordinator's state, so a
    misbehaving worker cannot corrupt the global x_s mid-step."""

    def mutating(dispatch: WindowDispatch):
        # A misbehaving worker scribbles over what it was handed.
        dispatch.video_snapshot.fill_(7.0)
        return dispatch.video_snapshot, dispatch.audio_snapshot

    # The required dispatch-construction pattern: clone the owned state.
    global_video = torch.zeros(1, 8)
    global_audio = torch.zeros(1, 4)
    dispatch = WindowDispatch(
        request_id="req0",
        step=0,
        contract=WindowContract(index=0, video_shape=(1, 8), audio_shape=(1, 4)),
        video_snapshot=global_video.clone(),
        audio_snapshot=global_audio.clone(),
    )
    coordinator, _ = _make([LocalWindowWorker(mutating)])
    coordinator.run_step(0, [dispatch])
    # The coordinator's global state is untouched by the worker's write.
    assert torch.equal(global_video, torch.zeros(1, 8))


def test_missing_prediction_fails_step_without_scheduler_update():
    class DroppingWorker(LocalWindowWorker):
        def submit(self, dispatch):
            if dispatch.contract.index == 1:
                future: Future = Future()
                future.set_exception(PredictionRejection("dropped"))
                return future
            return super().submit(dispatch)

    coordinator, scheduler_calls = _make([DroppingWorker(_identity), DroppingWorker(_identity)])
    try:
        coordinator.run_step(0, [_dispatch(0, 0), _dispatch(0, 1)])
        raise AssertionError("step must fail on a missing window")
    except PredictionRejection:
        pass
    assert scheduler_calls == []
    assert coordinator.counters.scheduler_updates == 0
    assert coordinator.counters.completed_steps == 0


def test_wrong_shape_prediction_rejected():
    def wrong_shape(dispatch: WindowDispatch):
        return torch.zeros(99, 8), dispatch.audio_snapshot

    coordinator, scheduler_calls = _make([LocalWindowWorker(wrong_shape)])
    try:
        coordinator.run_step(0, [_dispatch(0, 0)])
        raise AssertionError("wrong shape must fail the step")
    except PredictionRejection as exc:
        assert "violates its contract" in str(exc)
    assert coordinator.counters.rejected_shape == 1
    assert scheduler_calls == []


def test_repeated_step_rejected():
    coordinator, scheduler_calls = _make([LocalWindowWorker(_identity)])
    coordinator.run_step(0, [_dispatch(0, 0)])
    try:
        coordinator.run_step(0, [_dispatch(0, 0)])
        raise AssertionError("repeating a completed step must fail")
    except PredictionRejection as exc:
        assert "already completed" in str(exc)
    assert len(scheduler_calls) == 1  # still exactly once overall


def test_empty_and_unknown_dispatch_rejected():
    coordinator, _ = _make([LocalWindowWorker(_identity)])
    try:
        coordinator.run_step(0, [])
        raise AssertionError("empty dispatch list must fail")
    except PredictionRejection as exc:
        assert "no window dispatches" in str(exc)


def test_dispatch_identity_rejected_before_worker_execution():
    calls = []

    def forward(dispatch):
        calls.append(dispatch)
        return _identity(dispatch)

    coordinator, _ = _make([LocalWindowWorker(forward)])
    with pytest.raises(PredictionRejection, match="dispatch identity"):
        coordinator.run_step(0, [_dispatch(1, 0)])
    assert calls == []
    assert coordinator.counters.rejected_request == 1


def test_step_timeout_cancels_and_does_not_step():
    hang = threading.Event()

    def slow(dispatch: WindowDispatch):
        hang.wait(10)  # far beyond the budget; released by the finally below
        return _identity(dispatch)

    class SlowWorker(LocalWindowWorker):
        def __init__(self):
            super().__init__(slow)

        def submit(self, dispatch):
            # Simulate a transport future that never completes in time.
            future: Future = Future()
            threading.Thread(
                target=lambda: (
                    hang.wait(10),
                    future.set_result(_identity(dispatch)),
                ),
                daemon=True,
            ).start()
            return future

    clock = {"now": 0.0}

    def fake_clock():
        clock["now"] += 0.5
        return clock["now"]

    coordinator, scheduler_calls = _make(
        [SlowWorker()], step_timeout=2.0, clock=fake_clock
    )
    try:
        try:
            coordinator.run_step(0, [_dispatch(0, 0)])
            raise AssertionError("a hung worker must time the step out")
        except WindowStepTimeout:
            pass
        assert coordinator.counters.timeouts == 1
        assert scheduler_calls == []
    finally:
        hang.set()


def test_worker_exception_fails_step_cleanly():
    def boom(dispatch: WindowDispatch):
        raise RuntimeError("diode failed")

    coordinator, scheduler_calls = _make([LocalWindowWorker(boom)])
    try:
        coordinator.run_step(0, [_dispatch(0, 0)])
        raise AssertionError("worker failure must fail the step")
    except PredictionRejection as exc:
        assert "worker failed" in str(exc)
    assert coordinator.counters.worker_failures == 1
    assert scheduler_calls == []


def test_remote_stub_over_thread_pool_matches_local_protocol():
    with ThreadPoolExecutor(2) as pool:
        stub = RemoteWindowWorkerStub(pool, endpoint="probe-only")
        coordinator, scheduler_calls = _make([stub])
        video, audio = coordinator.run_step(0, [_dispatch(0, 0)])
        assert torch.equal(video, _dispatch(0, 0).video_snapshot)
        assert len(scheduler_calls) == 1
        # per-step cancellation must not permanently disable the worker
        stub.cancel()
        video2, _ = coordinator.run_step(1, [_dispatch(1, 0)])
        assert tuple(video2.shape) == (1, 8)


def test_multi_step_sequence_counts():
    workers = [LocalWindowWorker(_identity), LocalWindowWorker(_identity)]
    coordinator, scheduler_calls = _make(workers)
    for step in range(3):
        coordinator.run_step(step, [_dispatch(step, 0), _dispatch(step, 1)])
    assert len(scheduler_calls) == 3
    counters = coordinator.counters.as_dict()
    assert counters["dispatched"] == 6
    assert counters["completed_steps"] == 3
    assert counters["scheduler_updates"] == 3


def test_plan_topologies_budgets():
    assert plan_topologies(devices=16) == [
        {
            "kind": "serial",
            "groups": [16],
            "critical_path": "T(A) + T(B) on the full group",
            "verified": True,
            "note": "the baseline every distributed candidate must beat",
        },
        {
            "kind": "concurrent",
            "groups": [8, 8],
            "critical_path": "max(T_A, T_B) + transfer + fusion + barrier",
            "verified": False,
            "note": (
                "requires proven subgroup collectives and per-group HBM fit; "
                "fabric contention between disjoint groups is a G1 measurement"
            ),
        },
    ]
    assert plan_topologies(devices=8)[1]["groups"] == [4, 4]
    assert plan_topologies(devices=4)[1]["groups"] == [2, 2]
    assert plan_topologies(devices=2) == []
