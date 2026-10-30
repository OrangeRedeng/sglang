"""CPU regression checks for explicit Frozen-KV FIA task ownership and replay."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from sglang.srt.hardware_backend.npu.graph_runner import (
    frozen_kv_mtp_npu_graph_backend as mod,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture
def runtime(monkeypatch):
    order = []

    class Event:
        def __init__(self):
            self.complete = False

        def wait(self, stream):
            order.append("wait")

        def reset(self, stream):
            order.append("reset")

        def record(self, stream):
            order.append("signal" if stream != "compute" else "completion")

        def query(self):
            return self.complete

        def synchronize(self):
            order.append("host-wait")
            self.complete = True

    def stream():
        return SimpleNamespace(wait_stream=lambda compute: order.append("stream-wait"))

    npu = SimpleNamespace(
        current_device=lambda: 0,
        current_stream=lambda: "compute",
        Stream=lambda **kwargs: stream(),
        stream=lambda stream: nullcontext(),
        Event=Event,
        ExternalEvent=Event,
        graph_task_group_begin=lambda stream: order.append("capture-begin"),
        graph_task_group_end=lambda stream: object(),
        graph_task_update_begin=lambda stream, handle: order.append("update-begin"),
        graph_task_update_end=lambda stream: order.append("update-end"),
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)

    def out(**kwargs):
        order.append("operation")
        kwargs["out"][0].copy_(kwargs["query"][..., : kwargs["out"][0].shape[-1]])

    op = Mock()
    op.out = Mock(side_effect=out)
    plugin = SimpleNamespace(
        npu_fused_infer_attention_score_v2=op,
        _npu_fused_infer_attention_score_v2_get_max_workspace=Mock(
            return_value=torch.empty(16)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", plugin)
    monkeypatch.setattr(
        mod, "_borrow", lambda value: tuple(value) if isinstance(value, list) else value
    )
    return npu, plugin, order


def backend(runtime):
    npu, _, _ = runtime
    with (
        get_parallel().override(tp_group=SimpleNamespace(barrier=Mock())),
        patch.object(TorchMemorySaverAdapter, "create", return_value=None),
    ):
        return mod.FrozenKVMTPNpuGraphBackend(SimpleNamespace(device_module=npu))


def capture(b, runtime, count=3, layout="BSND"):
    _, _, order = runtime
    key = ShapeKey(size=2)
    graph = SimpleNamespace(replay=Mock(side_effect=lambda: order.append("replay")))
    query = torch.randn(2, 1, 1, 8) if layout == "BSND" else torch.randn(2, 1, 8)
    cache = torch.empty(2, 128, 4)
    with b._graph_capture_context(graph):
        for _ in range(count):
            output, _ = mod.frozen_kv_fia_v2(
                query,
                cache,
                cache,
                output_dim=4,
                input_layout=layout,
                actual_seq_qlen=[1, 1] if layout == "BSND" else [1, 2],
                actual_seq_kvlen=[7, 9],
                num_key_value_heads=1,
                block_table=torch.zeros(2, 1, dtype=torch.int32),
                block_size=128,
            )
            torch.testing.assert_close(output, query[..., :4])
    b._graphs[key], b._outputs[key] = graph, output
    return key, graph


@pytest.mark.parametrize("layout", ["BSND", "TND"])
def test_repeated_tasks_have_distinct_events_and_stable_workspace(runtime, layout):
    b = backend(runtime)
    _, graph = capture(b, runtime, layout=layout)
    tasks = graph._frozen_fia_tasks
    assert len(tasks) == 3 and len({id(t.event) for t in tasks}) == 3
    assert len({id(t.handle) for t in tasks}) == 3
    assert (
        runtime[1]._npu_fused_infer_attention_score_v2_get_max_workspace.call_count == 1
    )
    assert all(
        task.kwargs["workspace"] is tasks[0].kwargs["workspace"] for task in tasks
    )
    assert b._update_executor is None
    with pytest.raises(TypeError):
        tasks[0].kwargs["actual_seq_kvlen"] = [1, 2]


def test_replay_orders_events_without_host_wait_and_owns_snapshots(runtime):
    b = backend(runtime)
    key, graph = capture(b, runtime)
    runtime[2].clear()
    lengths = [17, 0]
    b.replay_with_input_update(
        key, None, cpu_update_input=[{"actual_seq_kvlen": lengths}]
    )
    assert runtime[2] == ["stream-wait", "replay"] + [
        "update-begin",
        "operation",
        "update-end",
        "signal",
    ] * 3 + ["completion"]
    lengths[0] = 999
    previous = b._inflight[0]
    assert all(params["actual_seq_kvlen"] == [17, 0] for params in previous[1])
    assert graph._frozen_fia_tasks[0].kwargs["actual_seq_kvlen"] == (7, 9)
    b.replay_with_input_update(key, [27, 31], attr_name="actual_seq_kvlen")
    assert len(b._inflight) == 2
    previous[0].complete = True
    b.replay_with_input_update(key, [2, 3], attr_name="actual_seq_kvlen")
    assert len(b._inflight) == 2 and b._inflight[0] is not previous
    assert "host-wait" not in runtime[2]
    b.cleanup()
    assert runtime[2].count("host-wait") == 2
    assert not b._graphs and not b._inflight


@pytest.mark.parametrize(
    "update",
    [
        [],
        [{"actual_seq_kvlen": [1]}],
        [{"actual_seq_kvlen": [1, -1]}],
        [{"actual_seq_kvlen": [1, True]}],
        [{"actual_seq_kvlen": [129, 1]}],
        [{"context_lens": [1, 2]}],
        [{"actual_seq_kvlen": [1, 2]}] * 2,
    ],
)
def test_invalid_metadata_is_rejected_before_replay(runtime, update):
    b = backend(runtime)
    key, graph = capture(b, runtime)
    with pytest.raises(ValueError):
        b.replay_with_input_update(key, None, cpu_update_input=update)
    graph.replay.assert_not_called()
    assert not b._inflight


def test_failed_update_is_fatal_and_retains_resources(runtime):
    b = backend(runtime)
    key, graph = capture(b, runtime)
    runtime[1].npu_fused_infer_attention_score_v2.out.side_effect = RuntimeError(
        "FIA failed"
    )
    with pytest.raises(RuntimeError, match="FIA failed"):
        b.replay_with_input_update(key, [7, 9], attr_name="actual_seq_kvlen")
    assert len(b._inflight) == 1
    with pytest.raises(RuntimeError, match="restart"):
        b.replay_with_input_update(key, [7, 9], attr_name="actual_seq_kvlen")
    assert graph.replay.call_count == 1
    with pytest.raises(RuntimeError, match="restart"):
        b.cleanup()
    assert b._graphs


def test_runner_streams_and_graph_records_are_independent(runtime):
    draft, target = backend(runtime), backend(runtime)
    assert draft._update_stream is not target._update_stream
    _, dg = capture(draft, runtime)
    _, tg = capture(target, runtime)
    assert dg._frozen_fia_tasks[0].event is not tg._frozen_fia_tasks[0].event


def test_eager_call_keeps_original_operator(runtime):
    query = torch.zeros(2, 1, 8)
    mod.frozen_kv_fia_v2(query, query, query, output_dim=8, input_layout="TND")
    runtime[1].npu_fused_infer_attention_score_v2.assert_called_once()
    runtime[1].npu_fused_infer_attention_score_v2.out.assert_not_called()


def test_numpy_verify_lengths_capture_reuse_workspace_and_own_host_arrays(runtime):
    b = backend(runtime)
    graph = SimpleNamespace()
    query = torch.zeros(4, 1, 8)
    qlen = np.arange(2, 6, 2, dtype=np.int64)
    kvlen = np.array([7, 9], dtype=np.int32)
    with b._graph_capture_context(graph):
        for q, kv in ((qlen, kvlen), (qlen.tolist(), kvlen.tolist())):
            mod.frozen_kv_fia_v2(
                query,
                query,
                query,
                output_dim=8,
                input_layout="TND",
                actual_seq_qlen=q,
                actual_seq_kvlen=kv,
            )
    plugin = runtime[1]
    assert plugin._npu_fused_infer_attention_score_v2_get_max_workspace.call_count == 1
    for call in plugin.npu_fused_infer_attention_score_v2.out.call_args_list:
        assert call.kwargs["actual_seq_qlen"] == [2, 4]
        assert call.kwargs["actual_seq_kvlen"] == [7, 9]
    qlen[:] = 99
    kvlen[:] = 99
    assert len(graph._frozen_fia_tasks) == 2
    for task in graph._frozen_fia_tasks:
        assert task.kwargs["actual_seq_qlen"] == (2, 4)
        assert task.kwargs["actual_seq_kvlen"] == (7, 9)
        assert task.bind({"actual_seq_kvlen": [17, 19]})["actual_seq_qlen"] == [2, 4]


def test_capture_disables_automatic_dispatch_and_excludes_warmup(runtime, monkeypatch):
    b = backend(runtime)
    npu, plugin, _ = runtime
    npu.synchronize = Mock()
    graph = SimpleNamespace()
    npu.NPUGraph = Mock(return_value=graph)
    npu.graph = Mock(side_effect=lambda *args, **kwargs: nullcontext())
    key = ShapeKey(size=2)
    query = torch.zeros(2, 1, 1, 8)
    plugin.npu_fused_infer_attention_score_v2.return_value = (query, None)

    def forward():
        return mod.frozen_kv_fia_v2(
            query,
            query,
            query,
            output_dim=8,
            input_layout="BSND",
            actual_seq_qlen=[1, 1],
            actual_seq_kvlen=[7, 9],
        )

    b.capture_one(key, forward)
    assert npu.graph.call_args.kwargs["auto_dispatch_capture"] is False
    assert plugin.npu_fused_infer_attention_score_v2.call_count == 2
    assert plugin.npu_fused_infer_attention_score_v2.out.call_count == 1
    assert len(graph._frozen_fia_tasks) == 1
    assert mod._capture.get() is None
    with pytest.raises(ValueError, match="no FIA"):
        b.capture_one(key, lambda: None)
    assert mod._capture.get() is None


def test_only_frozen_kv_selects_explicit_backend():
    from sglang.srt.hardware_backend.npu.graph_runner import (
        npu_cudagraph_backend as legacy,
    )
    from sglang.srt.model_executor.runner_backend import utils

    for frozen in (False, True):
        runner = SimpleNamespace(
            model_runner=SimpleNamespace(
                device="npu",
                spec_algorithm=SimpleNamespace(
                    is_frozen_kv_mtp=lambda frozen=frozen: frozen
                ),
            )
        )
        config = SimpleNamespace(
            graph=SimpleNamespace(cuda_graph_config=None),
            features=SimpleNamespace(enable_memory_saver=False),
        )
        with (
            patch.object(utils, "get_exec", return_value=config),
            patch.object(
                mod, "FrozenKVMTPNpuGraphBackend", return_value="explicit"
            ) as explicit,
            patch.object(legacy, "NPUCudaGraphBackend", return_value="legacy") as old,
        ):
            assert utils.resolve_decode_backend(runner) == (
                "explicit" if frozen else "legacy"
            )
            assert explicit.called is frozen
            assert old.called is not frozen


def test_unsupported_attention_path_rejected_during_explicit_capture(runtime):
    b = backend(runtime)
    graph = SimpleNamespace()
    with (
        pytest.raises(ValueError, match="unsupported attention"),
        b._graph_capture_context(graph),
    ):
        mod.reject_unhandled_fia_capture()
    assert mod._capture.get() is None
