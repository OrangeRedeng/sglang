"""Frozen-KV draft graph replay with Ascend FIA host-length rebinding."""

import logging

from sglang.srt.speculative.frozen_kv_mtp_cuda_graph_runner import (
    FrozenKVMTPCudaGraphRunner,
)

logger = logging.getLogger(__name__)


class FrozenKVMTPNpuGraphRunner(FrozenKVMTPCudaGraphRunner):
    """Reuse static draft buffers with the existing NPU graph backend."""

    def __init__(self, worker):
        from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
            AscendAttnBackend,
        )

        if worker.topk != 1 or not isinstance(
            worker.draft_attn_backend, AscendAttnBackend
        ):
            raise ValueError(
                "Frozen-KV NPU graphs require top-k 1 and Ascend attention"
            )
        if not worker.draft_attn_backend.use_fia:
            raise ValueError("Frozen-KV NPU graphs require ASCEND_USE_FIA=1")
        super().__init__(worker)
        self._probe_graph_updates()

    def _probe_graph_updates(self):
        self._updatable_graphs = set()
        for key, graph in self.backend._graphs.items():
            try:
                graph.update(cpu_update_input=self._length_updates([0] * key.size))
            except RuntimeError as error:
                if "0 operators" not in str(error):
                    raise
                logger.warning(
                    "Frozen-KV draft bucket %s has no FIA update records; using eager.",
                    key.size,
                )
            else:
                self._updatable_graphs.add(key)

    def _length_updates(self, lengths):
        # Seed and recurrent forwards read the same committed target prefix.
        return [
            {"actual_seq_kvlen": list(lengths)}
            for _ in range(self.speculative_num_steps)
        ]

    def can_run_graph(self, forward_batch):
        if forward_batch.seq_lens_cpu is None or not super().can_run_graph(
            forward_batch
        ):
            return False
        bs = self._pad_to_bucket(forward_batch.batch_size, self.capture_bs)
        return self._make_graph_key(bs) in self._updatable_graphs

    def _replay_graph(self, shape_key, forward_batch):
        lengths = forward_batch.seq_lens_cpu[: self.raw_bs].tolist()
        lengths += [0] * (self.bs - self.raw_bs)
        return self.backend.replay_with_input_update(
            shape_key, seq_lens=None, cpu_update_input=self._length_updates(lengths)
        )
