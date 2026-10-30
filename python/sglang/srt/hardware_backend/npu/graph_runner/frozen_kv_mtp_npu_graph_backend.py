# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team
"""Explicit FIA task lifecycle for Frozen-KV draft and verification graphs.

Follows vLLM-Ascend's UpdatableGraph task/event protocol:
https://github.com/vllm-project/vllm-ascend/blob/0f515fe1e246a967dc41c9aa3d4e8ecb24cbbae2/vllm_ascend/compilation/updatable_graph.py
"""

import logging
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import torch
from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
    NPUCudaGraphBackend,
)

_capture = ContextVar("frozen_kv_fia_capture", default=None)
logger = logging.getLogger(__name__)


def reject_unhandled_fia_capture():
    if _capture.get() is not None:
        raise ValueError("Frozen-KV graph contains an unsupported attention path")


def _borrow(value):
    if isinstance(value, torch.Tensor):
        from torch_npu._C import _weak_ref_tensor

        return _weak_ref_tensor(value)
    if isinstance(value, (tuple, list)):
        return tuple(_borrow(item) for item in value)
    return value


@dataclass(frozen=True)
class _FIATask:
    operation: object
    kwargs: object
    handle: object
    event: object

    def bind(self, update):
        if set(update) != {"actual_seq_kvlen"}:
            raise ValueError("Frozen-KV graphs only update actual_seq_kvlen")
        lengths = update["actual_seq_kvlen"]
        if (
            not isinstance(lengths, (list, tuple))
            or len(lengths) != len(self.kwargs["actual_seq_kvlen"])
            or any(type(n) is not int or n < 0 for n in lengths)
        ):
            raise ValueError("Invalid Frozen-KV graph KV lengths")
        table = self.kwargs.get("block_table")
        if table is not None:
            capacity = table.shape[1] * self.kwargs["block_size"]
            if any(length > capacity for length in lengths):
                raise ValueError("Frozen-KV graph KV length exceeds its block table")
        # Each replay owns its host arrays until the compute completion event.
        kwargs = dict(self.kwargs)
        kwargs["actual_seq_kvlen"] = list(lengths)
        kwargs["actual_seq_qlen"] = list(kwargs["actual_seq_qlen"])
        kwargs["out"] = list(kwargs["out"])
        return kwargs


def frozen_kv_fia_v2(query, key, value, *, output_dim, **kwargs):
    import torch_npu

    state = _capture.get()
    if state is None:
        return torch_npu.npu_fused_infer_attention_score_v2(query, key, value, **kwargs)
    tasks, resources = state
    if kwargs.get("input_layout") not in ("TND", "BSND"):
        raise ValueError("Frozen-KV FIA graphs require TND or BSND")
    for name in ("actual_seq_qlen", "actual_seq_kvlen"):
        if isinstance(kwargs.get(name), np.ndarray):
            kwargs[name] = kwargs[name].tolist()
    params = dict(query=query, key=key, value=value, **kwargs)
    # Workspace is shared only between identical operator signatures in one graph.
    signature = tuple(
        (
            name,
            (tuple(v.shape), v.dtype, tuple(v.stride()))
            if isinstance(v, torch.Tensor)
            else tuple(v)
            if isinstance(v, list)
            else v,
        )
        for name, v in params.items()
    )
    if signature not in resources:
        resources[signature] = (
            torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(**params)
        )
    output = query.new_empty((*query.shape[:-1], output_dim))
    lse = query.new_empty(1)
    params.update(workspace=resources[signature], out=[output, lse])
    stream = torch.npu.current_stream()
    event = torch.npu.ExternalEvent()
    event.wait(stream)
    event.reset(stream)
    torch.npu.graph_task_group_begin(stream)
    operation = torch_npu.npu_fused_infer_attention_score_v2.out
    operation(**params)
    handle = torch.npu.graph_task_group_end(stream)
    captured = {k: v if k == "workspace" else _borrow(v) for k, v in params.items()}
    tasks.append(_FIATask(operation, MappingProxyType(captured), handle, event))
    return output, lse


class FrozenKVMTPNpuGraphBackend(NPUCudaGraphBackend):
    _auto_dispatch_capture = False

    def __init__(self, runner, **kwargs):
        super().__init__(runner, **kwargs)
        self._update_stream = self._device_module.Stream(device=self._device_id)
        self._inflight = deque()
        self._failed = False
        self._logged_replay = False
        self._role = (
            "draft"
            if getattr(getattr(runner, "model_runner", None), "is_draft_worker", False)
            else "verify"
        )

    @contextmanager
    def _graph_capture_context(self, graph):
        tasks, resources = [], {}
        token = _capture.set((tasks, resources))
        try:
            yield
        finally:
            _capture.reset(token)
            graph._frozen_fia_tasks = tuple(tasks)
            graph._frozen_fia_resources = resources

    def capture_one(self, shape_key, *args, **kwargs):
        super().capture_one(shape_key, *args, **kwargs)
        if not self._graphs[shape_key]._frozen_fia_tasks:
            raise ValueError("Frozen-KV graph captured no FIA update tasks")
        logger.info(
            "Frozen-KV NPU %s bucket %s captured %d FIA tasks.",
            self._role,
            shape_key.size,
            len(self._graphs[shape_key]._frozen_fia_tasks),
        )

    def replay_with_input_update(
        self, shape_key, seq_lens, attr_name=None, attr_type=None, cpu_update_input=None
    ):
        if self._failed:
            raise RuntimeError("Frozen-KV graph update failed; restart this worker")
        graph = self._graphs[shape_key]
        tasks = graph._frozen_fia_tasks
        updates = (
            cpu_update_input
            if cpu_update_input is not None
            else [{attr_name: seq_lens}]
        )
        if len(updates) == 1:
            updates = updates * len(tasks)
        if not tasks or len(updates) != len(tasks):
            raise ValueError("Frozen-KV graph update count does not match capture")
        resolved = tuple(task.bind(update) for task, update in zip(tasks, updates))
        while self._inflight and self._inflight[0][0].query():
            self._inflight.popleft()
        stream = self._device_module.current_stream()
        self._update_stream.wait_stream(stream)
        done = self._device_module.Event()
        # Keep the snapshots even if submission fails: a partial graph cannot be retried.
        self._inflight.append((done, resolved))
        try:
            graph.replay()
            with self._device_module.stream(self._update_stream):
                for task, params in zip(tasks, resolved):
                    self._device_module.graph_task_update_begin(
                        self._update_stream, task.handle
                    )
                    task.operation(**params)
                    self._device_module.graph_task_update_end(self._update_stream)
                    task.event.record(self._update_stream)
            done.record(stream)
        except Exception:
            self._failed = True
            raise
        if not self._logged_replay:
            logger.info(
                "Frozen-KV NPU %s explicit FIA graph replay active (bucket %s).",
                self._role,
                shape_key.size,
            )
            self._logged_replay = True
        return self._outputs[shape_key]

    def replay(self, shape_key, static_forward_batch, **kwargs):
        raise RuntimeError("Frozen-KV graph replay requires fresh host KV lengths")

    def cleanup(self):
        if self._failed:
            # Unsatisfied external events can make synchronization hang forever.
            raise RuntimeError(
                "Cannot release a partially updated Frozen-KV graph; restart this worker"
            )
        for done, _ in self._inflight:
            done.synchronize()
        self._inflight.clear()
        super().cleanup()
