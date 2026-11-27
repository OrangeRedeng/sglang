"""Stable collective outputs and eager-only shared overlap for the TP12 matrix."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

from sglang.srt.hardware_backend.npu.besttp12 import (
    Int8Operand,
    enabled,
    evidence,
    gather_workspace,
    release_operand,
    release_workspace,
    _INPUT,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
    is_in_breakable_cuda_graph,
)
from sglang.srt.runtime_context import get_parallel

_SHARED_STREAMS = {}

_PENDING = ContextVar("tp12_pending", default=None)


def clear_input():
    release_operand(_PENDING.get())
    _PENDING.set(None)


def publish_input(q, scale, dtype):
    if enabled("INPUT_REUSE"):
        _PENDING.set(Int8Operand(q, scale, dtype))
    else:
        # Workspace-backed gather without operand publication: nothing will
        # hold the lease, release it now (same-stream ordering keeps the
        # buffer valid for the immediately following dequant read).
        release_workspace(q)


def take_input():
    value = _PENDING.get()
    _PENDING.set(None)
    return value


def current_input():
    return _INPUT.get()


def _gather_outputs(q, scale):
    size = get_parallel().attn_tp_group.world_size
    return q.new_empty((q.shape[0] * size, *q.shape[1:])), scale.new_empty(
        (scale.numel() * size,)
    )


@eager_on_graph(output_factory=_gather_outputs, pass_output=True)
def gather_input(q, scale, out=None):
    workspace = None
    if out is None:
        workspace = gather_workspace(q, scale)
        output = workspace if workspace is not None else _gather_outputs(q, scale)
    else:
        output = out
    group = get_parallel().attn_tp_group
    group.all_gather_into_tensor(output[0], q)
    group.all_gather_into_tensor(output[1], scale)
    if workspace is not None:
        evidence("gather-input-workspace", rows=int(q.shape[0]))
    return output


@contextmanager
def shared_overlap(layer, hidden, skip, allocator):
    from sglang.srt.model_executor.runner import get_is_capture_mode

    eligible = (
        enabled("SHARED_STREAM")
        and not skip
        and 0 < hidden.shape[0] <= 16384
        and layer.shared_experts is not None
        and not layer.num_fused_shared_experts
        and not layer._shared_expert_tp1
        and not layer._fuse_shared_experts_inside_sbo
        and layer.experts.runner_backend.is_ascend()
        and not layer.experts.reduce_results
        and get_parallel().moe_ep_size == 1
        and not get_is_capture_mode()
        and not is_in_breakable_cuda_graph()
        and not torch.compiler.is_compiling()
    )
    if not eligible:
        yield None
        return
    device = torch.get_device_module(hidden.device)
    stream = _SHARED_STREAMS.get(str(hidden.device))
    if stream is None:
        stream = device.Stream()
        _SHARED_STREAMS[str(hidden.device)] = stream
    current = device.current_stream()
    stream.wait_stream(current)
    operand = _INPUT.get()
    output = None
    try:
        with device.stream(stream):
            hidden.record_stream(stream)
            if operand is not None:
                operand.record_stream(stream)
            output = layer._forward_shared_experts(hidden, allocator)
        evidence("shared-stream", rows=int(hidden.shape[0]))
        yield output
    finally:
        current.wait_stream(stream)
        if output is not None:
            output.record_stream(current)


@eager_on_graph
def collective_into(kind, output, value):
    group = get_parallel().attn_tp_group
    getattr(group, kind)(output, value)
