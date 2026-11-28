# SPDX-License-Identifier: Apache-2.0
"""Step-major distributed AV-window workers and coordinator (R07, code phase).

Protocol contract (see R07_REPORT.md for the full text):

- The coordinator owns one global state ``(video, audio)``, the positions,
  conditioning, scheduler state and the request id. Workers never update the
  global state and never decode independently.
- Per step the coordinator selects the window policy, takes an *immutable
  snapshot* of every dispatched window's rows, and dispatches one batch-1
  local forward per window.
- A worker returns a ``WindowPrediction`` carrying the request id, step and
  window index. The coordinator rejects stale (old step), duplicate (same
  window twice in one step), cross-request and wrong-shape predictions with
  explicit counters; a step completes only with every expected window.
- Fusion runs on the coordinator and the scheduler steps exactly once per
  completed step. The next step cannot begin before the previous update is
  complete and valid.

This module implements the protocol and its local (in-process) worker; a
remote worker is a ``submit``/``cancel`` implementation over a probed
transport (see ``research/r07_topology_probe.py``). Everything here is
CPU-testable with synchronous fakes; no transport or device is claimed.
"""

from __future__ import annotations

import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Callable, Protocol

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

__all__ = [
    "WindowContract",
    "WindowDispatch",
    "WindowPrediction",
    "WindowWorker",
    "LocalWindowWorker",
    "RemoteWindowWorkerStub",
    "DistributedWindowCoordinator",
    "PredictionRejection",
    "WindowStepTimeout",
    "CoordinatorCounters",
    "plan_topologies",
]


class PredictionRejection(RuntimeError):
    """A step cannot complete: predictions missing/rejected or a stale step."""


class WindowStepTimeout(RuntimeError):
    """A dispatched step did not produce every prediction within the budget."""


@dataclass(frozen=True)
class WindowContract:
    """The exact output contract one window must satisfy for one step."""

    index: int
    video_shape: tuple[int, ...]
    audio_shape: tuple[int, ...]


@dataclass(frozen=True)
class WindowDispatch:
    """One window's per-step work order with an immutable input snapshot.

    ``video_snapshot``/``audio_snapshot`` are private copies: the coordinator
    may keep mutating its global state while a worker still reads these.
    """

    request_id: str
    step: int
    contract: WindowContract
    video_snapshot: torch.Tensor
    audio_snapshot: torch.Tensor


@dataclass(frozen=True)
class WindowPrediction:
    request_id: str
    step: int
    window: int
    video: torch.Tensor
    audio: torch.Tensor


class WindowWorker(Protocol):
    """One batch-1 window worker.

    ``submit`` returns immediately; wait semantics live on the returned
    future so the coordinator can bound a step without assuming a transport.
    ``cancel`` aborts every in-flight dispatch of this worker.
    """

    def submit(self, dispatch: WindowDispatch) -> Future: ...

    def cancel(self) -> None: ...


class LocalWindowWorker:
    """In-process worker: runs one window's forward inline on submit.

    This is the G2 numerical-reference path: the prediction equals calling
    the same forward on the same snapshot directly, so a fused distributed
    run can be diffed against the sequential loop exactly.
    """

    def __init__(
        self,
        forward: Callable[[WindowDispatch], tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        self._forward = forward

    def submit(self, dispatch: WindowDispatch) -> Future:
        future: Future = Future()
        try:
            future.set_result(self._forward(dispatch))
        except BaseException as exc:  # propagate worker failures to the step
            future.set_exception(exc)
        return future

    def cancel(self) -> None:  # inline work cannot outlive submit
        return None


class RemoteWindowWorkerStub:
    """Stub for a remote window worker over an operator-provided transport.

    R07 does not ship a working bulk-latent transport: GLM's #31320 carries
    prior tokens, not AV latent tensors, and #40957's Mooncake path is not
    verified on Ascend. This stub therefore executes through a caller-supplied
    executor (testable with a thread pool; on a real deployment the executor
    would be the probed transport client) and exists so cancellation,
    timeouts and result handling are exercised against the same protocol as
    the local worker.
    """

    def __init__(self, executor, endpoint: str = "unprobed") -> None:
        self._executor = executor
        self.endpoint = endpoint
        self._inflight: set[Future] = set()

    def submit(self, dispatch: WindowDispatch) -> Future:
        future = self._executor.submit(
            lambda d=dispatch: (
                # A real transport sends (contract, snapshot) and returns
                # payload rows; both directions are transport capability,
                # not protocol.
                d.video_snapshot * 1.0,
                d.audio_snapshot * 1.0,
            )
        )
        self._inflight.add(future)
        return future

    def cancel(self) -> None:
        # Abort only in-flight dispatches; the worker stays usable for the
        # next step (cancellation is per step, never a permanent disable).
        for future in self._inflight:
            future.cancel()
        self._inflight.clear()


@dataclass
class CoordinatorCounters:
    dispatched: int = 0
    completed_steps: int = 0
    rejected_duplicate: int = 0
    rejected_shape: int = 0
    rejected_request: int = 0
    worker_failures: int = 0
    canceled_steps: int = 0
    timeouts: int = 0
    scheduler_updates: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "dispatched": self.dispatched,
            "completed_steps": self.completed_steps,
            "rejected_duplicate": self.rejected_duplicate,
            "rejected_shape": self.rejected_shape,
            "rejected_request": self.rejected_request,
            "worker_failures": self.worker_failures,
            "canceled_steps": self.canceled_steps,
            "timeouts": self.timeouts,
            "scheduler_updates": self.scheduler_updates,
        }


class DistributedWindowCoordinator:
    """Owns the step-major protocol; one instance per request.

    ``fuse`` receives one step's predictions ordered by window index and
    returns the fused ``(video, audio)`` update; the two-window FP32 ramp
    blend of ``windowing.py`` is the reference implementation callers pass.
    ``scheduler_step`` is invoked exactly once per completed step, after
    fusion, and never on a failed step.
    """

    def __init__(
        self,
        *,
        request_id: str,
        workers: list[WindowWorker],
        fuse: Callable[[list[WindowPrediction]], tuple[torch.Tensor, torch.Tensor]],
        scheduler_step: Callable[[], None],
        step_timeout: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not workers:
            raise ValueError("a coordinator needs at least one window worker")
        self.request_id = request_id
        self.workers = list(workers)
        self.fuse = fuse
        self.scheduler_step = scheduler_step
        self.step_timeout = step_timeout
        self.clock = clock
        self.counters = CoordinatorCounters()
        self._completed_steps: set[int] = set()
        self.last_rejections: list[str] = []

    # -- dispatch / collect -------------------------------------------------

    def run_step(
        self, step: int, dispatches: list[WindowDispatch]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one global step to completion; returns the fused update.

        Raises :class:`PredictionRejection` (missing/rejected predictions, a
        worker failure or a repeated step) or :class:`WindowStepTimeout`; in
        every failure case the scheduler has *not* stepped and in-flight work
        is canceled.
        """
        if step in self._completed_steps:
            raise PredictionRejection(f"step {step} already completed")
        if not dispatches:
            raise PredictionRejection(f"step {step} has no window dispatches")
        invalid = [
            dispatch
            for dispatch in dispatches
            if dispatch.request_id != self.request_id or dispatch.step != step
        ]
        if invalid:
            self.counters.rejected_request += len(invalid)
            raise PredictionRejection(
                f"step {step} dispatch identity does not match request {self.request_id!r}"
            )
        indexes = [dispatch.contract.index for dispatch in dispatches]
        if len(set(indexes)) != len(indexes):
            raise PredictionRejection(
                f"step {step} dispatches repeat window indices {sorted(indexes)}"
            )
        self.last_rejections = []
        started = self.clock()
        self.counters.dispatched += len(dispatches)
        futures: dict[int, Future] = {}
        contracts = {d.contract.index: d.contract for d in dispatches}
        try:
            for dispatch in dispatches:
                worker = self.workers[dispatch.contract.index % len(self.workers)]
                futures[dispatch.contract.index] = worker.submit(dispatch)
            predictions: dict[int, WindowPrediction] = {}
            for index, future in futures.items():
                budget = (
                    None
                    if self.step_timeout is None
                    else max(self.step_timeout - (self.clock() - started), 0.0)
                )
                try:
                    video, audio = future.result(budget)
                except TimeoutError:
                    self.counters.timeouts += 1
                    raise WindowStepTimeout(
                        f"step {step} timed out waiting for window {index}"
                    ) from None
                except PredictionRejection:
                    raise
                except Exception as exc:  # noqa: BLE001 - worker failure fails the step
                    self.counters.worker_failures += 1
                    raise PredictionRejection(
                        f"step {step} window {index} worker failed: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                self._absorb(step, index, contracts, predictions, video, audio)
            missing = sorted(set(contracts) - set(predictions))
            if missing:
                raise PredictionRejection(
                    f"step {step} missing predictions for windows {missing}"
                )
            ordered = [predictions[index] for index in sorted(contracts)]
            fused_video, fused_audio = self.fuse(ordered)
            # Exactly-once: only a fully valid, fused step advances the sampler.
            self.scheduler_step()
            self.counters.scheduler_updates += 1
            self._completed_steps.add(step)
            self.counters.completed_steps += 1
            return fused_video, fused_audio
        finally:
            # Futures are one-shot; drained completions are harmless, pending
            # ones must not outlive a failed step.
            for future in futures.values():
                future.cancel()
            for worker in self.workers:
                worker.cancel()

    def _absorb(
        self,
        step: int,
        index: int,
        contracts: dict[int, WindowContract],
        predictions: dict[int, WindowPrediction],
        video: torch.Tensor,
        audio: torch.Tensor,
    ) -> None:
        if index not in contracts:
            raise PredictionRejection(f"step {step} unexpected window index {index}")
        if index in predictions:
            self.counters.rejected_duplicate += 1
            raise PredictionRejection(f"step {step} duplicate prediction for window {index}")
        for tensor, shape in (
            (video, contracts[index].video_shape),
            (audio, contracts[index].audio_shape),
        ):
            if tuple(tensor.shape) != shape:
                self.counters.rejected_shape += 1
                self.last_rejections.append(f"window {index} shape mismatch")
                raise PredictionRejection(
                    f"step {step} window {index} prediction shape "
                    f"{tuple(tensor.shape)} violates its contract {shape}"
                )
        predictions[index] = WindowPrediction(
            request_id=self.request_id, step=step, window=index, video=video, audio=audio
        )


def plan_topologies(*, devices: int, min_group: int = 2) -> list[dict]:
    """Feasible R07 topologies for one fixed device budget (G0/G3 planning).

    Returns both the serial baseline and the two-disjoint-group candidate
    with its critical-path model. ``verified`` is always False here: subgroup
    collective layouts and per-group memory fit are operator-campaign facts
    (G0/G1), never planning outputs. No entry invents a topology switch that
    divides serial latency by two for free.
    """
    if devices < min_group * 2:
        return []
    topologies = [
        {
            "kind": "serial",
            "groups": [devices],
            "critical_path": "T(A) + T(B) on the full group",
            "verified": True,
            "note": "the baseline every distributed candidate must beat",
        },
        {
            "kind": "concurrent",
            "groups": [devices // 2, devices - devices // 2],
            "critical_path": "max(T_A, T_B) + transfer + fusion + barrier",
            "verified": False,
            "note": (
                "requires proven subgroup collectives and per-group HBM fit; "
                "fabric contention between disjoint groups is a G1 measurement"
            ),
        },
    ]
    return topologies
