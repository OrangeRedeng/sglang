"""CPU tracing checks for the external NPU routing normalization boundary."""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.fixture(scope="module")
def topk_module():
    def external_l1_norm(value):
        output = torch.empty(value.shape, device=value.device, dtype=torch.float32)
        weights = value.float()
        output.copy_(weights / weights.sum(dim=-1, keepdim=True))
        return output

    name = "sgl_kernel_npu.norm.l1_norm"
    with patch.dict(sys.modules, {name: SimpleNamespace(l1_norm=external_l1_norm)}):
        module = importlib.import_module("sglang.srt.hardware_backend.npu.moe.topk")
        # Exercise the real custom-op boundary with a CPU stand-in for the NPU kernel.
        library = torch.library.Library("sglang", "IMPL", "CPU")
        try:
            library.impl("npu_topk_l1_norm", external_l1_norm)
            yield module
        finally:
            library._destroy()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fullgraph_normalization_hides_transfer_wrapper(topk_module, dtype):
    empty = torch.empty

    @torch.compiler.disable
    def remap_devices(args):
        return args

    def wrapped_empty(*args, **kwargs):
        return empty(*remap_devices(args), **kwargs)

    graphs = []

    def backend(graph, example_inputs):
        nodes = [
            node
            for node in graph.graph.nodes
            if node.target
            in (
                torch.ops.sglang.npu_topk_l1_norm,
                torch.ops.sglang.npu_topk_l1_norm.default,
            )
        ]
        assert len(nodes) == 1
        assert nodes[0].meta["example_value"].dtype == torch.float32
        graphs.append(graph)
        return graph.forward

    try:
        with patch.object(torch, "empty", wrapped_empty):
            compiled = torch.compile(
                topk_module.l1_norm, backend=backend, fullgraph=True, dynamic=True
            )
            for rows in (2, 5):
                weights = torch.arange(1, rows * 4 + 1).reshape(rows, 4).to(dtype)
                expected = weights.float() / weights.float().sum(dim=-1, keepdim=True)
                output = compiled(weights)
                torch.testing.assert_close(output, expected)
                assert output.dtype == torch.float32
                assert output.shape == weights.shape
                assert output.is_contiguous()
        assert graphs
    finally:
        torch._dynamo.reset()
