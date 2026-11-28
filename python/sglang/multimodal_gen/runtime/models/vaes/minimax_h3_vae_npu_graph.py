# SPDX-License-Identifier: Apache-2.0
"""Opt-in NPU graph capture/replay for the MiniMax-H3 tile decoder.

The tiled decode calls one pure ``forward(x)`` per stacked tile group
(``forward_many``); gathering, blending, halos and temporal state live
outside it, so a group decode is a stateless function of its input bytes --
the safe capture unit. This module wraps that call: warmup, capture into a
static input/output pair, then replay per identical (shape, dtype, device,
decoder revision) bucket, with an explicit eager fallback everywhere.

Everything is opt-in and fails closed: without ``SGLANG_H3_VAE_NPU_GRAPH=1``,
without the installed TorchNPU graph API, or on any capture/replay error the
unwrapped forward runs. Host testing covers the cache contract only; capture,
replay bytes and memory behavior are device-pending.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Capture is opt-in; set to 1 to enable. Keep the flag name in sync with the
# report/runbook.
VAE_NPU_GRAPH_ENV = "SGLANG_H3_VAE_NPU_GRAPH"
# Bounded capture table: one entry per distinct (shape, dtype, device,
# revision) bucket. A request with many tile-group shapes captures at most
# this many graphs; further shapes run eager.
DEFAULT_MAX_ENTRIES = 8


def _is_fatal_device_error(error: BaseException) -> bool:
    """Do not continue after TorchNPU has poisoned the device context."""
    text = f"{type(error).__name__}: {error}".lower()
    return any(
        marker in text
        for marker in (
            "aicore error",
            "device error",
            "device is unavailable",
            "device has been reset",
            "rt error",
            "acl error",
            "out of memory",
        )
    )


def _recoverable_graph_error(error: RuntimeError) -> bool:
    if isinstance(error, torch.OutOfMemoryError) or _is_fatal_device_error(error):
        return False
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "not supported",
            "unsupported",
            "not implemented",
            "no capture on this host",
        )
    )


@dataclass
class _GraphEntry:
    graph: Any
    static_input: torch.Tensor
    static_output: torch.Tensor


@dataclass
class TileGraphStats:
    """Counters surfaced for the runbook; cheap enough to keep always."""

    captures: int = 0
    replays: int = 0
    eager_calls: int = 0
    replay_failures: int = 0
    capture_failures: int = 0
    invalidations: int = 0
    entries: dict[tuple, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "captures": self.captures,
            "replays": self.replays,
            "eager_calls": self.eager_calls,
            "capture_failures": self.capture_failures,
            "replay_failures": self.replay_failures,
            "invalidations": self.invalidations,
            "entries": len(self.entries),
        }

    def receipt(self, *, request_id: str, case: str) -> dict:
        """Return a hashable per-request evidence record for campaign ledgers."""
        if not request_id or not case:
            raise ValueError("request_id and case are required for graph evidence")
        return {
            "request_id": request_id,
            "case": case,
            "counters": self.as_dict(),
        }


def npu_graph_api_available() -> bool:
    """Whether the installed TorchNPU exposes the capture APIs used here.

    Import presence is necessary, not sufficient: capture itself is only
    proven on the operator's device campaign (never inferred from this check).
    """
    if not hasattr(torch, "npu"):
        return False
    return all(
        hasattr(torch.npu, name) for name in ("graph", "NPUGraph", "current_stream")
    )


class TileDecodeGraphCache:
    """Capture/replay cache for one decoder's per-group forward.

    Each entry owns a private graph pool because bucket replay order varies.
    """

    def __init__(
        self,
        *,
        revision: Callable[[], Any],
        max_entries: int = DEFAULT_MAX_ENTRIES,
        enabled: bool | None = None,
    ) -> None:
        self._revision = revision
        self._max_entries = max(1, int(max_entries))
        if enabled is None:
            enabled = os.getenv(VAE_NPU_GRAPH_ENV, "0") == "1"
        self._enabled = enabled
        self._entries: dict[tuple, _GraphEntry] = {}
        self._unsupported: set[tuple] = set()
        self.stats = TileGraphStats()
        self._epoch = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def clear(self, *, reason: str) -> None:
        """Drop every captured graph (weight/dtype/state epoch changed)."""
        if self._entries:
            self.stats.invalidations += 1
            logger.info(
                "MiniMax H3 VAE graph cache cleared (%s): %d entries dropped",
                reason,
                len(self._entries),
            )
        self._entries.clear()
        self._unsupported.clear()
        self.stats.entries.clear()

    def wrap(self, forward_fn: Callable[[torch.Tensor], torch.Tensor]):
        """Return ``forward_fn`` with graph replay for repeated buckets."""
        if not self._enabled:
            return forward_fn

        def wrapped(x: torch.Tensor) -> torch.Tensor:
            return self._run(forward_fn, x)

        return wrapped

    def _bucket_key(self, x: torch.Tensor) -> tuple:
        return (
            tuple(x.shape),
            str(x.dtype),
            x.device.type,
            None if x.device.index is None else int(x.device.index),
            repr(self._revision()),
        )

    def _device_matches(self, x: torch.Tensor) -> bool:
        """The capture device contract; host tests override this seam."""
        return x.device.type == "npu"

    def _run(
        self, forward_fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
    ) -> torch.Tensor:
        if (
            not self._enabled
            or not self._device_matches(x)
            or not npu_graph_api_available()
            or torch.is_grad_enabled()
            or not x.is_contiguous()
        ):
            self.stats.eager_calls += 1
            return forward_fn(x)

        key = self._bucket_key(x)
        if self._epoch != key[-1]:
            self.clear(reason="decoder revision changed")
            self._epoch = key[-1]
        if x.device.type == "npu":
            key += (torch.npu.current_stream(x.device).npu_stream,)
        if key in self._unsupported:
            self.stats.eager_calls += 1
            return forward_fn(x)
        entry = self._entries.get(key)
        if entry is None:
            if len(self._entries) >= self._max_entries:
                self.stats.eager_calls += 1
                return forward_fn(x)
            try:
                entry = self._capture(forward_fn, x)
            except RuntimeError as exc:
                self.stats.capture_failures += 1
                if not _recoverable_graph_error(exc):
                    raise
                self._unsupported.add(key)
                self.stats.eager_calls += 1
                logger.warning("H3 tile graph capture disabled for %s: %s", key, exc)
                return forward_fn(x)
            self._entries[key] = entry
            self.stats.entries[key] = 1
            self.stats.captures += 1
        else:
            entry.static_input.copy_(x)
        try:
            # Capture records work. Explicit replay also initializes output on the first call.
            entry.graph.replay()
        except RuntimeError as exc:
            self.stats.replay_failures += 1
            self.stats.invalidations += 1
            del self._entries[key], self.stats.entries[key]
            if not _recoverable_graph_error(exc):
                raise
            self._unsupported.add(key)
            self.stats.eager_calls += 1
            logger.warning("H3 tile graph replay disabled for %s: %s", key, exc)
            return forward_fn(x)
        self.stats.replays += 1
        return entry.static_output.clone()

    def _capture(
        self, forward_fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
    ) -> _GraphEntry:
        static_input = x.clone()
        # Warmup on the side of the capture pool: primed caches, workspace and
        # allocator state must exist before capture, and warmup work must not
        # land in the graph's private pool.
        warm = forward_fn(static_input)
        torch.npu.current_stream().synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            static_output = forward_fn(static_input)
        if static_output.shape != warm.shape or static_output.dtype != warm.dtype:
            raise RuntimeError(
                "MiniMax H3 VAE graph capture changed the output contract: "
                f"{tuple(warm.shape)}/{warm.dtype} vs "
                f"{tuple(static_output.shape)}/{static_output.dtype}"
            )
        return _GraphEntry(graph, static_input, static_output)
