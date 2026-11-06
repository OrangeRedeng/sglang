"""CPU checks for NPU piecewise capture lifecycle and backend selection."""

import importlib
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.compilation.compile_phase import (
    enable_torch_compile_warmup,
    set_pcg_capture_stream,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture(scope="module")
def npu_backend_module():
    # CPU has no accelerator weak-reference primitive; lifecycle checks use owned tensors.
    weak_refs = SimpleNamespace(weak_ref_tensors=lambda value: value)
    name = "sglang.srt.model_executor.runner_backend_utils.weak_ref_tensor"
    with patch.dict(sys.modules, {name: weak_refs}):
        yield importlib.import_module("sglang.srt.compilation.npu_piecewise_backend")


def make_backend(module, sym_shape_indices=None):
    return module.NPUPiecewiseBackend(
        graph=None,
        compile_config=CompilationConfig([4]),
        inductor_config={},
        graph_pool="pool",
        piecewise_compile_index=0,
        total_piecewise_compiles=2,
        sym_shape_indices=[0] if sym_shape_indices is None else sym_shape_indices,
        compiled_graph_for_general_shape=lambda size, x: x + 1,
        sglang_backend=None,
    )


def test_static_piece_does_not_index_missing_symbolic_shape(npu_backend_module):
    backend = make_backend(npu_backend_module, sym_shape_indices=[])
    for _ in range(3):
        torch.testing.assert_close(backend(4, torch.zeros(4)), torch.ones(4))


def test_compile_warmup_does_not_capture_or_consume_capture_warmup(npu_backend_module):
    backend = make_backend(npu_backend_module)
    with enable_torch_compile_warmup():
        for _ in range(4):
            torch.testing.assert_close(backend(4, torch.zeros(4)), torch.ones(4))
    entry = backend.concrete_size_entries[4]
    assert entry.cudagraph is None
    assert entry.num_finished_warmup == 0


def test_runtime_recompilation_without_capture_stream_runs_eagerly(npu_backend_module):
    backend = make_backend(npu_backend_module)
    for _ in range(4):
        torch.testing.assert_close(backend(4, torch.zeros(4)), torch.ones(4))
    assert backend.concrete_size_entries[4].cudagraph is None


def test_capture_uses_runner_stream_and_replay_reuses_graph(npu_backend_module):
    backend = make_backend(npu_backend_module)
    graph = Mock()
    npu = SimpleNamespace(NPUGraph=Mock(return_value=graph), graph=Mock())
    npu.graph.return_value = nullcontext()
    stream = object()
    inputs = torch.zeros(4)
    module = "sglang.srt.compilation.npu_piecewise_backend"
    with (
        patch.object(torch, "npu", npu, create=True),
        patch(f"{module}.weak_ref_tensors", side_effect=lambda value: value),
        set_pcg_capture_stream(stream),
    ):
        for _ in range(3):
            backend(4, inputs)
    npu.graph.assert_called_once_with(graph, pool="pool", stream=stream)
    torch.testing.assert_close(backend(4, inputs), torch.ones(4))
    graph.replay.assert_called_once_with()


@pytest.mark.parametrize("device", ["npu", "cuda"])
def test_prefill_resolver_selects_piecewise_only_on_npu(device, npu_backend_module):
    from sglang.srt.model_executor.runner_backend import utils

    config = SimpleNamespace(prefill=SimpleNamespace(backend="tc_piecewise"))
    execution = SimpleNamespace(graph=SimpleNamespace(cuda_graph_config=config))
    runner = SimpleNamespace(model_runner=SimpleNamespace(device=device))
    module = "sglang.srt.model_executor.runner_backend.tc_piecewise_cuda_graph_backend"
    with (
        patch.object(utils, "get_exec", return_value=execution),
        patch(f"{module}.TcPiecewiseCudaGraphBackend") as backend,
    ):
        if device == "npu":
            assert utils.resolve_prefill_backend(runner) is backend.return_value
            backend.assert_called_once_with(runner)
        else:
            with pytest.raises(ValueError, match="only supported on NPU"):
                utils.resolve_prefill_backend(runner)
            backend.assert_not_called()


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({}, None),
        ({"device": "cuda"}, "only supported on NPU"),
        ({"pp_size": 2}, "does not support"),
        ({"dcp_size": 2}, "does not support"),
        ({"attn_cp_size": 2}, "does not support"),
        ({"enable_dp_attention": True}, "does not support"),
        ({"enable_lora": True}, "does not support"),
        ({"speculative_algorithm": "EAGLE3"}, "does not support"),
    ],
)
def test_piecewise_configuration_rejects_unsupported_topologies(overrides, error):
    from sglang.srt.arg_groups import cuda_graph_hook as hook
    from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig

    config = SimpleNamespace(
        cuda_graph_config=CudaGraphConfig.from_dict(
            {"prefill": {"backend": "tc_piecewise"}}
        ),
        device="npu",
        pp_size=1,
        dcp_size=1,
        attn_cp_size=1,
        enable_dp_attention=False,
        enable_lora=False,
        speculative_algorithm=None,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    with (
        patch.object(hook, "resolving_view", return_value=config),
        patch.object(hook, "resolved_view", return_value=config),
        patch.object(hook, "attention_backends_of", return_value=("ascend", "ascend")),
    ):
        if error:
            with pytest.raises(ValueError, match=error):
                hook.validate_cuda_graph_config(object())
        else:
            hook.validate_cuda_graph_config(object())
