"""Page-owned DCP prefix gathering, with bounded double-buffered prefetch."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch


def validate_legacy_dcp_flags(environment, *, tp_size=None, dcp_size=None):
    def enabled(key):
        return str(environment.get(key, "")).lower() in ("1", "true", "yes")

    cp = enabled("SGLANG_NPU_ENABLE_DSA_CP")
    residual = enabled("SGLANG_NPU_DSA_CP_TOKEN_SHARDED")
    full_q = enabled("SGLANG_NPU_DSA_CP_Q_FULL_WIDTH")
    stream = enabled("SGLANG_NPU_STREAM_O_PROJ")
    if enabled("SGLANG_NPU_DSA_CP_Q_A2A_PREISSUE"):
        raise ValueError(
            "SGLANG_NPU_DSA_CP_Q_A2A_PREISSUE is incompatible with the restored full-head path; leave it off"
        )
    for flag in (
        "SGLANG_NPU_DSA_CP_TOKEN_SHARDED",
        "SGLANG_NPU_DSA_CP_Q_FULL_WIDTH",
        "SGLANG_NPU_DSA_CP_LOCAL_TOPK",
    ):
        if enabled(flag) and not cp:
            raise ValueError(flag + " requires SGLANG_NPU_ENABLE_DSA_CP=True")
    if cp and not (residual or full_q):
        raise ValueError(
            "SGLANG_NPU_ENABLE_DSA_CP requires full-width Q or the token-residual rebuild"
        )
    if stream and not residual:
        raise ValueError(
            "SGLANG_NPU_STREAM_O_PROJ requires SGLANG_NPU_DSA_CP_TOKEN_SHARDED=True"
        )
    if stream and (tp_size is None or dcp_size != tp_size):
        raise ValueError(
            "SGLANG_NPU_STREAM_O_PROJ requires equal TP/DCP sizes and rank order"
        )
    if (cp or enabled("SGLANG_NPU_DCP_DECODE_Q_PREISSUE")) and (
        dcp_size is None or dcp_size < 2
    ):
        raise ValueError(
            "SGLANG_NPU_ENABLE_DSA_CP / SGLANG_NPU_DCP_DECODE_Q_PREISSUE require DCP >= 2"
        )


_o_proj_gather_buffers = {}


def _o_proj_key(group, shard):
    return (
        tuple(getattr(group, "ranks", ())),
        getattr(group, "rank_in_group", id(group)),
        group.world_size,
        shard.dtype,
        shard.device,
        tuple(shard.shape),
    )


def prepare_o_proj_gather(group, shard):
    key = _o_proj_key(group, shard)
    if key not in _o_proj_gather_buffers:
        rows, cols = shard.shape[1], shard.shape[0]
        _o_proj_gather_buffers[key] = [
            (
                shard.new_empty((rows * group.world_size, cols)),
                shard.new_empty((rows, cols)),
            )
            for _ in range(2)
        ]


def start_o_proj_gather(group, shard, *, buffer_slot=None):
    # Rank r contributes columns r*in_r:(r+1)*in_r of [out, in].
    if buffer_slot is None:
        send = shard.t().contiguous()
        gathered = send.new_empty((send.shape[0] * group.world_size, send.shape[1]))
    else:
        buffers = _o_proj_gather_buffers.get(_o_proj_key(group, shard))
        if buffers is None:
            raise RuntimeError(
                "Streamed o_proj buffers must be prepared before serving"
            )
        if buffer_slot not in (0, 1):
            raise ValueError("Streamed o_proj uses exactly two buffer slots")
        gathered, send = buffers[buffer_slot]
        send.copy_(shard.t())
    work = group.all_gather_into_tensor_async(gathered, send)
    return gathered, send, work


@dataclass
class PendingQueryHeads:
    gathered: torch.Tensor
    send: torch.Tensor
    work: object
    rope_dim: int

    def finish(self):
        if self.work is not None:
            self.work.wait()
        rope, nope = self.gathered.split(
            [self.rope_dim, self.gathered.shape[-1] - self.rope_dim], dim=-1
        )
        return nope.transpose(0, 1), rope.transpose(0, 1)


def start_query_head_gather(group, nope, rope):
    send = torch.cat((rope.transpose(0, 1), nope.transpose(0, 1)), dim=-1).contiguous()
    gathered = send.new_empty((send.shape[0] * group.world_size, *send.shape[1:]))
    return PendingQueryHeads(
        gathered,
        send,
        group.all_gather_into_tensor_async(gathered, send),
        rope.shape[-1],
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
        if self.packed and hasattr(self.plan, "requests"):
            source = 0
            for request, offset in zip(self.plan.requests, self.plan.row_offsets):
                for out, value in zip(self.outputs, own):
                    out[offset + request.gathered_rows : offset + request.rows] = value[
                        source : source + request.extend_len
                    ]
                source += request.extend_len
        elif self.packed:
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


def start_prefix_gather(
    group,
    sends,
    refs,
    plan,
    dcp_size,
    packed=False,
    *,
    buffer_slot=None,
    output_capacity=0,
):
    """Retain sends and disjoint per-piece scratch until all work completes."""

    def allocate(name, ref, rows):
        if buffer_slot is None:
            return ref.new_empty((rows, *ref.shape[1:]))
        from sglang.srt.layers.dcp.layout import dcp_extend_gather_buffer

        return dcp_extend_gather_buffer(
            f"pipeline-{buffer_slot}-{name}",
            ref,
            rows,
            capacity_rows=output_capacity if name.startswith("out-") else 0,
        )

    works, scratches = [], []
    if packed and hasattr(plan, "requests"):
        if any(sum(plan.local_lens) != send.shape[0] for send in sends):
            raise ValueError("Packed batch prefix ownership does not match the sends")
        outputs = tuple(
            allocate(f"out-{i}", ref, plan.rows) for i, ref in enumerate(refs)
        )
        retained = []
        source = 0
        for request_id, (request, local, offset) in enumerate(
            zip(plan.requests, plan.local_lens, plan.row_offsets)
        ):
            for value_id, (out, send) in enumerate(zip(outputs, sends)):
                padded = allocate(
                    f"send-{request_id}-{value_id}", send, request.send_rows
                )
                padded.zero_()
                padded[:local] = send[source : source + local]
                retained.append(padded)
                if request.gathered_rows:
                    works.append(
                        group.all_gather_into_tensor_async(
                            out[offset : offset + request.gathered_rows], padded
                        )
                    )
            source += local
        sends = tuple(retained)
    elif packed:
        outputs = tuple(
            allocate(f"out-{i}", ref, plan.rows) for i, ref in enumerate(refs)
        )
        padded = []
        for value_id, send in enumerate(sends):
            value = allocate(f"send-{value_id}", send, plan.send_rows)
            value.zero_()
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
        outputs = tuple(allocate(f"out-{i}", ref, rows) for i, ref in enumerate(refs))
        padded = []
        for value_id, send in enumerate(sends):
            value = allocate(f"send-{value_id}", send, plan.send_rows)
            value.zero_()
            src = dst = 0
            for local, width in zip(plan.local_lens, plan.padded_lens):
                value[dst : dst + local] = send[src : src + local]
                src += local
                dst += width
            padded.append(value)
        sends = tuple(padded)
        for piece_id, piece in enumerate(plan.pieces):
            gathered = (piece.send_end - piece.send_start) * dcp_size
            rows = gathered + piece.extend_end - piece.extend_start
            pair = tuple(
                allocate(f"scratch-{piece_id}-{i}", ref, rows)
                for i, ref in enumerate(refs)
            )
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
        self.o_proj_slots = {}

    def reset(self):
        # A reused ForwardBatch is not a forward identity. Prepare calls reset
        # on the first local layer of every forward, after prior consumers.
        self.stream.wait_stream(torch.npu.current_stream())
        self.slots.clear()
        self.o_proj_slots.clear()

    def issue(
        self, layer_id, pool, indices, refs, plan, parallel, packed, stream_o_proj=False
    ):
        if layer_id in self.slots:
            return
        self.stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(self.stream):
            sends = pool.get_mla_kv_buffer(SimpleNamespace(layer_id=layer_id), indices)
            pending = start_prefix_gather(
                parallel.dcp_group,
                sends,
                refs,
                plan,
                parallel.dcp_size,
                packed,
                buffer_slot=layer_id % 2,
            )
            # Work completion must be queued on the producing stream before
            # its event; the consumer's wait then orders every output read.
            for work in pending.works:
                if work is not None:
                    work.wait()
            pending.works.clear()
            event = self.stream.record_event()
            if stream_o_proj:
                from sglang.srt.layers.attention.dsa.dsa_cp import (
                    get_o_proj_stream_shard,
                )

                shard = get_o_proj_stream_shard(layer_id)
                if shard is None:
                    raise RuntimeError(
                        f"o_proj shard for layer {layer_id} was not prepared"
                    )
                if list(parallel.dcp_group.ranks) != list(parallel.attn_tp_group.ranks):
                    raise ValueError(
                        "Streamed o_proj requires identical DCP and attention-TP rank order"
                    )
                gathered, send, work = start_o_proj_gather(
                    parallel.dcp_group, shard, buffer_slot=layer_id % 2
                )
                if work is not None:
                    work.wait()
                self.o_proj_slots[layer_id] = (
                    gathered,
                    send,
                    self.stream.record_event(),
                )
                shard.record_stream(self.stream)
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

    def take_o_proj(self, layer_id):
        gathered, send, event = self.o_proj_slots.pop(layer_id)
        current = torch.npu.current_stream()
        current.wait_event(event)
        gathered.record_stream(current)
        from sglang.srt.hardware_backend.npu.feature_evidence import record

        record("dsa-o-proj-stream")
        return gathered.t()
