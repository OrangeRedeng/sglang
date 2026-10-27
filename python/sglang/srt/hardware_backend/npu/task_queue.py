"""Process-wide task-queue selection at serialized model-forward boundaries."""

import logging
import os
import threading

from sglang.srt.environ import envs
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_memory,
    get_parallel,
    get_spec,
)

logger = logging.getLogger(__name__)


class NPUTaskQueuePolicy:
    def __init__(self, npu):
        required = (
            "set_task_queue_enable",
            "get_task_queue_enable",
            "synchronize",
            "is_current_stream_capturing",
        )
        if any(not callable(getattr(npu, name, None)) for name in required):
            raise ValueError(
                "Adaptive NPU task queues require torch_npu runtime queue setters "
                "(available in torch_npu 2.10.0.post7)."
            )
        self._npu = npu
        self._owner_thread = threading.get_ident()
        self._mode = npu.get_task_queue_enable()
        logger.info(
            "Adaptive NPU task queue enabled: prefill=0, decode=1; "
            "phase transitions synchronize the device."
        )

    @classmethod
    def create(cls, *, device, is_draft_worker):
        if device != "npu" or not envs.SGLANG_ENABLE_NPU_ADAPTIVE_TASK_QUEUE.get():
            return None
        offload = get_exec().offload
        if os.environ.get("ASCEND_LAUNCH_BLOCKING") == "1":
            raise ValueError(
                "Adaptive NPU task queues require ASCEND_LAUNCH_BLOCKING!=1"
            )
        if (
            is_draft_worker
            or get_spec().speculative_algorithm is not None
            or get_parallel().pp_size > 1
            or get_exec().overlap.enable_two_batch_overlap
            or get_memory().enable_hierarchical_cache
            or offload.cpu_offload_gb > 0
            or offload.offload_group_size > 0
            or get_disagg().disaggregation_mode != "null"
        ):
            raise ValueError(
                "Adaptive NPU task queues require non-speculative serving without "
                "PP, two-batch overlap, HiCache, CPU offload, or PD disaggregation."
            )
        import torch_npu

        return cls(torch_npu.npu)

    def before_forward(self, forward_mode):
        if forward_mode.is_decode():
            self.set_mode(1)
        elif forward_mode.is_extend_without_speculative():
            self.set_mode(0)

    def set_mode(self, mode):
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Adaptive NPU task queues require one submission thread")
        if mode == self._mode:
            return
        if self._npu.is_current_stream_capturing():
            raise RuntimeError("Cannot switch NPU task queues inside graph capture")
        # Finish submitted device work before changing the process-wide dispatch path.
        self._npu.synchronize()
        self._npu.set_task_queue_enable(mode)
        if self._npu.get_task_queue_enable() != mode:
            raise RuntimeError("torch_npu did not apply the requested task queue mode")
        logger.debug("NPU task queue mode: %d -> %d", self._mode, mode)
        self._mode = mode
