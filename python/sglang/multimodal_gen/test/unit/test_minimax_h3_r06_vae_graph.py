# SPDX-License-Identifier: Apache-2.0
"""Host tests for the R06 VAE graph cache and audio-overlap gating.

CPU-only: the real torch.npu capture/replay stays device-pending. The tests
exercise the cache contract through the documented seams (device gate, API
probe, capture hook) and the decoding stage's platform gating.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import torch

import sglang.multimodal_gen.runtime.models.vaes.minimax_h3_vae_npu_graph as npu_graph_module
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_vae_npu_graph import (
    TileDecodeGraphCache,
    npu_graph_api_available,
)

_REPO_ROOT = Path(__file__).resolve().parents[5]


def _allow_all_devices(cache: TileDecodeGraphCache) -> None:
    """Point the device gate seam at CPU tensors for contract tests."""
    cache._device_matches = lambda x: True  # noqa: SLF001 - documented seam


def _fake_graph_api(monkeypatch) -> None:
    monkeypatch.setattr(npu_graph_module, "npu_graph_api_available", lambda: True)


class _FakeGraph:
    """Stands in for torch.npu.NPUGraph: replay recomputes from the static
    input with the captured function, like a replayed capture would."""

    def __init__(self, fn, static_input):
        self._fn = fn
        self._static_input = static_input
        self.replays = 0
        self.static_output = None

    def replay(self):
        self.replays += 1
        # Real capture writes the static output buffer in place; the fake
        # mirrors that contract.
        self.static_output.copy_(self._fn(self._static_input))


def _fake_capture(fn, x):
    static_input = x.clone()
    graph = _FakeGraph(fn, static_input)
    graph.static_output = fn(static_input)
    return npu_graph_module._GraphEntry(graph, static_input, graph.static_output)


def _make_cache(monkeypatch, **kwargs) -> TileDecodeGraphCache:
    _fake_graph_api(monkeypatch)
    cache = TileDecodeGraphCache(revision=lambda: "rev0", enabled=True, **kwargs)
    cache._capture = _fake_capture  # noqa: SLF001 - documented seam
    return cache


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv(npu_graph_module.VAE_NPU_GRAPH_ENV, raising=False)
    cache = TileDecodeGraphCache(revision=lambda: 0)
    assert cache.enabled is False
    calls = []

    def fn(x):
        calls.append(x)
        return x + 1

    x = torch.zeros(4, 8)
    # A disabled cache wraps to the original callable; no stats accumulate.
    wrapped = cache.wrap(fn)
    assert wrapped is fn
    assert torch.equal(wrapped(x), x + 1)
    assert not cache.stats.captures and not cache.stats.replays


def test_unavailable_api_stays_eager(monkeypatch):
    monkeypatch.setenv(npu_graph_module.VAE_NPU_GRAPH_ENV, "1")
    assert npu_graph_api_available() is False  # host torch has no npu module
    cache = TileDecodeGraphCache(revision=lambda: 0, enabled=True)
    _allow_all_devices(cache)
    calls = []

    def fn(x):
        calls.append(x)
        return x * 2

    x = torch.ones(2, 3)
    out = cache.wrap(fn)(x)
    assert torch.equal(out, x * 2)
    assert cache.stats.eager_calls == 1 and not cache.stats.captures


def test_capture_once_replay_per_bucket(monkeypatch):
    cache = _make_cache(monkeypatch)
    _allow_all_devices(cache)
    forward_calls = []

    def fn(x):
        forward_calls.append(x.shape)
        return x * 3.0

    wrapped = cache.wrap(fn)
    with torch.no_grad():
        first = wrapped(torch.full((4, 8), 2.0))
        second = wrapped(torch.full((4, 8), 5.0))
        third = wrapped(torch.full((4, 8), 7.0))
    # one capture, two replays, and outputs reflect each replay's input bytes
    assert cache.stats.captures == 1 and cache.stats.replays == 2
    assert torch.equal(first, torch.full((4, 8), 6.0))
    assert torch.equal(second, torch.full((4, 8), 15.0))
    assert torch.equal(third, torch.full((4, 8), 21.0))
    # replay outputs are private copies: mutating one must not corrupt others
    second.mul_(0.0)
    assert torch.equal(third, torch.full((4, 8), 21.0))


def test_bucket_key_tracks_shape_and_revision(monkeypatch):
    cache = _make_cache(monkeypatch)
    revision = {"epoch": 0}
    cache._revision = lambda: revision["epoch"]  # noqa: SLF001 - seam
    _allow_all_devices(cache)
    wrapped = cache.wrap(lambda x: x + 1)
    with torch.no_grad():
        wrapped(torch.zeros(4, 8))
        wrapped(torch.zeros(2, 8))  # distinct shape -> distinct bucket
        assert len(cache._entries) == 2
        revision["epoch"] = 1
        wrapped(torch.zeros(4, 8))  # new revision -> new bucket, old stays
    assert len(cache._entries) == 3


def test_max_entries_bound_and_overflow_eager(monkeypatch):
    cache = _make_cache(monkeypatch, max_entries=2)
    _allow_all_devices(cache)
    wrapped = cache.wrap(lambda x: x + 1)
    with torch.no_grad():
        wrapped(torch.zeros(1, 4))
        wrapped(torch.zeros(2, 4))
        wrapped(torch.zeros(3, 4))  # overflow -> eager, table stays bounded
    assert len(cache._entries) == 2
    assert cache.stats.eager_calls == 1


def test_capture_failure_falls_back_and_stays_eager(monkeypatch):
    _fake_graph_api(monkeypatch)
    cache = TileDecodeGraphCache(revision=lambda: 0, enabled=True)
    _allow_all_devices(cache)

    def broken_capture(fn, x):
        raise RuntimeError("no capture on this host")

    cache._capture = broken_capture  # noqa: SLF001 - seam
    wrapped = cache.wrap(lambda x: x * 2)
    x = torch.ones(2, 2)
    with torch.no_grad():
        assert torch.equal(wrapped(x), x * 2)
        assert torch.equal(wrapped(x), x * 2)
    assert cache.stats.capture_failures == 2
    assert cache.stats.eager_calls == 2 and not cache.stats.captures


def test_clear_invalidation(monkeypatch):
    cache = _make_cache(monkeypatch)
    _allow_all_devices(cache)
    wrapped = cache.wrap(lambda x: x + 1)
    with torch.no_grad():
        wrapped(torch.zeros(4, 4))
    assert len(cache._entries) == 1
    cache.clear(reason="weights recast")
    assert not cache._entries and cache.stats.invalidations == 1


def test_non_contiguous_input_stays_eager(monkeypatch):
    cache = _make_cache(monkeypatch)
    _allow_all_devices(cache)
    eager_calls = []

    def fn(x):
        eager_calls.append(1)
        return x + 1

    base = torch.zeros(4, 8).t()  # non-contiguous view
    assert not base.is_contiguous()
    assert torch.equal(cache.wrap(fn)(base), base + 1)
    assert len(eager_calls) == 1 and not cache.stats.captures


def test_grad_mode_stays_eager(monkeypatch):
    cache = _make_cache(monkeypatch)
    _allow_all_devices(cache)
    with torch.enable_grad():
        x = torch.zeros(4, 4, requires_grad=True)
        assert torch.equal(cache.wrap(lambda t: t + 1)(x), x + 1)
    assert cache.stats.eager_calls == 1


def test_vae_wires_cache_and_invalidation_paths():
    """The video VAE owns a cache only where the NPU graph API exists, and the
    LightVAE loader invalidates it when the decoder changes."""
    source = (
        _REPO_ROOT
        / "python"
        / "sglang"
        / "multimodal_gen"
        / "runtime"
        / "models"
        / "vaes"
        / "minimax_h3.py"
    ).read_text()
    assert "TileDecodeGraphCache(" in source
    assert "npu_graph_api_available" in source
    assert '_tile_graph_cache.clear(reason="decoder weights recast")' in source
    loader = (
        _REPO_ROOT
        / "python"
        / "sglang"
        / "multimodal_gen"
        / "runtime"
        / "models"
        / "vaes"
        / "minimax_h3_lightvae.py"
    ).read_text()
    assert 'graph_cache.clear(reason="LightVAE decoder loaded")' in loader


def test_audio_overlap_gate_accepts_npu():
    """The NPU side stream is gated exactly like CUDA: resident audio VAE and
    a decode world > 1."""
    from sglang.multimodal_gen.runtime.platforms import current_platform
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.decoding import (
        MiniMaxH3DecodingStage,
    )

    args = type(
        "Args",
        (),
        {"residency_mode": lambda self, name: "resident"},
    )()
    with (
        patch.object(current_platform, "is_cuda", return_value=False),
        patch.object(current_platform, "is_npu", return_value=True),
        patch.object(current_platform, "is_rocm", return_value=False),
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages."
            "model_specific_stages.minimax_h3.stages.decoding."
            "model_parallel_is_initialized",
            return_value=True,
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages."
            "model_specific_stages.minimax_h3.stages.decoding."
            "get_decode_parallel_world_size",
            return_value=4,
        ),
    ):
        assert MiniMaxH3DecodingStage._overlaps_audio_decode(args) is True


def test_r06_probe_reports_without_device():
    import json
    import subprocess

    probe = _REPO_ROOT / "test" / "manual" / "minimax_h3_realtime" / "research" / (
        "r06_vae_graph_probe.py"
    )
    result = subprocess.run(
        [sys.executable, str(probe), "--json"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert set(report["subtasks"]) >= {
        "V2_V3_tile_decoder_npugraph",
        "V4_audio_side_stream",
    }
    # Without torch_npu, the graph path must report blocked, never available.
    assert report["subtasks"]["V2_V3_tile_decoder_npugraph"]["verdict"] in {
        "blocked",
        "available",
    }
