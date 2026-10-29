"""Page-owned DCP prefix gathering, with bounded double-buffered prefetch."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch


def validate_legacy_dcp_flags(environment):
    unsupported = (
        "SGLANG_NPU_ENABLE_DSA_CP",
        "SGLANG_NPU_DSA_CP_TOKEN_SHARDED",
        "SGLANG_NPU_STREAM_O_PROJ",
        "SGLANG_NPU_DSA_CP_Q_FULL_WIDTH",
        "SGLANG_NPU_DSA_CP_Q_A2A_PREISSUE",
        "SGLANG_NPU_DSA_CP_LOCAL_TOPK",
        "SGLANG_NPU_DCP_DECODE_Q_PREISSUE",
    )
    enabled = [
        key
        for key in unsupported
        if str(environment.get(key, "")).lower() in ("1", "true", "yes")
    ]
    if enabled:
        raise ValueError(
            "Legacy DSA-CP features have no compatible consumer in the current attention path: "
            + ", ".join(enabled)
            + ". Use SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD for query sharding; "
            "the legacy residual/o_proj memory optimization remains unvalidated."
        )


@dataclass
class PendingGather:
    outputs: tuple
    scratches: list
    sends: tuple
    works: list
    plan: object
    packed: bool
    dcp_size: int

    def finish(self, own):
        for work in self.works:
            if work is not None:
                work.wait()
        if self.packed:
            for out, value in zip(self.outputs, own):
                out[self.plan.gathered_rows :] = value[: self.plan.extend_len]
        else:
            for piece, pair in zip(self.plan.pieces, self.scratches):
                gathered = (piece.send_end - piece.send_start) * self.dcp_size
                for out, scratch, value in zip(self.outputs, pair, own):
                    scratch[gathered:] = value[piece.extend_start : piece.extend_end]
                    torch.index_select(
                        scratch,
                        0,
                        piece.index,
                        out=out[piece.out_start : piece.out_end],
                    )
        return self.outputs


def start_prefix_gather(group, sends, refs, plan, dcp_size, packed=False):
    """Retain sends and disjoint per-piece scratch until all work completes."""
    works, scratches = [], []
    if packed:
        outputs = tuple(ref.new_empty((plan.rows, *ref.shape[1:])) for ref in refs)
        padded = []
        for send in sends:
            value = send.new_zeros((plan.send_rows, *send.shape[1:]))
            value[: send.shape[0]] = send
            padded.append(value)
        sends = tuple(padded)
        if plan.gathered_rows:
            works = [
                group.all_gather_into_tensor_async(out[: plan.gathered_rows], send)
                for out, send in zip(outputs, sends)
            ]
    else:
        rows = plan.pieces[-1].out_end if plan.pieces else 0
        outputs = tuple(ref.new_empty((rows, *ref.shape[1:])) for ref in refs)
        padded = []
        for send in sends:
            value = send.new_zeros((plan.send_rows, *send.shape[1:]))
            src = dst = 0
            for local, width in zip(plan.local_lens, plan.padded_lens):
                value[dst : dst + local] = send[src : src + local]
                src += local
                dst += width
            padded.append(value)
        sends = tuple(padded)
        for piece in plan.pieces:
            gathered = (piece.send_end - piece.send_start) * dcp_size
            rows = gathered + piece.extend_end - piece.extend_start
            pair = tuple(ref.new_empty((rows, *ref.shape[1:])) for ref in refs)
            scratches.append(pair)
            if gathered:
                works.extend(
                    group.all_gather_into_tensor_async(
                        buf[:gathered], send[piece.send_start : piece.send_end]
                    )
                    for buf, send in zip(pair, sends)
                )
    return PendingGather(outputs, scratches, sends, works, plan, packed, dcp_size)


class DcpGatherPipeline:
    def __init__(self):
        self.stream = torch.npu.Stream()
        self.slots = {}

    def reset(self):
        # A reused ForwardBatch is not a forward identity. Prepare calls reset
        # on the first local layer of every forward, after prior consumers.
        self.stream.wait_stream(torch.npu.current_stream())
        self.slots.clear()

    def issue(self, layer_id, pool, indices, refs, plan, parallel, packed):
        if layer_id in self.slots:
            return
        self.stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(self.stream):
            sends = pool.get_mla_kv_buffer(SimpleNamespace(layer_id=layer_id), indices)
            pending = start_prefix_gather(
                parallel.dcp_group, sends, refs, plan, parallel.dcp_size, packed
            )
            # Work completion must be queued on the producing stream before
            # its event; the consumer's wait then orders every output read.
            for work in pending.works:
                if work is not None:
                    work.wait()
            pending.works.clear()
            event = self.stream.record_event()
            for tensor in (*sends, *refs):
                tensor.record_stream(self.stream)
        self.slots[layer_id] = (pending, event)

    def take(self, layer_id, own):
        pending, event = self.slots.pop(layer_id)
        current = torch.npu.current_stream()
        current.wait_event(event)
        for tensor in (
            *pending.outputs,
            *pending.sends,
            *(t for pair in pending.scratches for t in pair),
        ):
            tensor.record_stream(current)
        return pending.finish(own)
