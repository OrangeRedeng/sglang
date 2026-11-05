"""Eager DCP boundaries between NPU compute graph pieces."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.runtime_context import get_parallel

_attention_scope = ContextVar("npu_dcp_piecewise_attention", default=False)


def active():
    return is_in_tc_piecewise_cuda_graph() and get_parallel().dcp_enabled


def in_eager_attention():
    return _attention_scope.get()


@contextmanager
def eager_attention_scope():
    token = _attention_scope.set(True)
    try:
        yield
    finally:
        _attention_scope.reset(token)


def stage_tensor(owner, role, value):
    if value is None:
        return None
    from sglang.srt.hardware_backend.npu.moe.contracts import QuantizedOperand

    if isinstance(value, QuantizedOperand):
        return QuantizedOperand(
            stage_tensor(owner, role + "_data", value.data),
            stage_tensor(owner, role + "_scale", value.scale),
            value.contract,
            value.source_shape,
            value.source_dtype,
        )
    buffers = getattr(owner, "_npu_dcp_piecewise_buffers", None)
    if buffers is None:
        buffers = {}
        owner._npu_dcp_piecewise_buffers = buffers
    key = (
        getattr(owner, "_npu_dcp_piecewise_slot", 0),
        role,
        tuple(value.shape),
        value.dtype,
        value.device,
    )
    if key not in buffers:
        buffers[key] = torch.empty_like(value)
        if value.ndim:
            torch._dynamo.maybe_mark_dynamic(buffers[key], 0)
    buffers[key].copy_(value)
    return buffers[key]


@torch.compiler.disable
def reset_forward(batch, num_tokens):
    from sglang.srt.hardware_backend.npu.modules.dsa_token_residual import (
        reset_forward as reset,
    )

    reset(batch, num_tokens)


@torch.compiler.disable
def attention(
    module, positions, hidden, batch, allocator, input_on_slices, scaling, previous_topk
):
    # Metadata, cache writes, and HCCL stay outside the captured compute pieces.
    nested = in_eager_attention()
    with eager_attention_scope():
        state = module.forward_prepare(
            positions=positions,
            hidden_states=hidden,
            forward_batch=batch,
            zero_allocator=allocator,
            input_on_attn_tp_slices=input_on_slices,
            llama_4_scaling=scaling,
            prev_topk_indices=previous_topk,
        )
        result = module.forward_core(state)
        if nested:
            return result
        if isinstance(result, tuple):
            return stage_tensor(module, "attention", result[0]), result[1]
        return stage_tensor(module, "attention", result)


@torch.compiler.disable
def prepare_decoder(layer, positions, hidden, batch, allocator, previous_topk, plan):
    from sglang.srt.hardware_backend.npu.modules import dsa_token_residual as residual

    with eager_attention_scope():
        if plan is None:
            plan = residual.full_head_plan(
                batch,
                layer.self_attn.indexer.index_topk
                if layer.self_attn.indexer is not None
                else None,
            )
        if plan is None:
            if getattr(batch, "npu_dsa_cp_token_sharded", False):
                raise RuntimeError(
                    "DSA residual sharding cannot change within a forward"
                )
            return None
        normalized, topk = residual.prepare_decoder_attention(
            layer, positions, hidden, batch, allocator, previous_topk, plan
        )
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

        stream = residual_batch.stream_of(batch)
        stream.write(stage_tensor(layer, "residual", stream.residual))
        return stage_tensor(layer, "ffn_input", normalized), topk, plan


@torch.compiler.disable
def gather_dense(layer, hidden, plan):
    group = get_parallel().attn_tp_group
    result = hidden.new_empty((plan.num_tokens_pad, hidden.shape[-1]))
    group.all_gather_into_tensor(result, hidden.contiguous())
    return stage_tensor(layer, "dense_input", result)


@torch.compiler.disable
def finish_decoder(layer, hidden, batch, plan):
    from sglang.srt.hardware_backend.npu.modules.dsa_token_residual import (
        finish_decoder_ffn,
    )

    return finish_decoder_ffn(layer, hidden, batch, plan, stage_output=True)


def prepare_batch(runner, batch):
    model = getattr(runner.model, "model", runner.model)
    buffers = getattr(model, "_npu_dcp_piecewise_buffers", None)
    if buffers is None:
        buffers = {}
        model._npu_dcp_piecewise_buffers = buffers
        # Adjacent layers alternate storage; eager boundaries consume the prior slot first.
        for index, layer in enumerate(getattr(model, "layers", ())):
            for owner in (layer, layer.self_attn, layer.mlp):
                owner._npu_dcp_piecewise_buffers = buffers
                owner._npu_dcp_piecewise_slot = index % 2
    from sglang.srt.model_executor.forward_batch_deepseek_mha_mixin import (
        create_chunked_prefix_cache_kv_indices,
    )

    prepare = getattr(runner.model, "prepare_context_parallel_metadata_for_dcp", None)
    if prepare is None or batch.extend_prefix_lens is None:
        raise ValueError("NPU DCP piecewise requires model DCP prefill metadata")
    batch.attn_dcp_metadata = prepare(
        batch.seq_lens,
        batch.extend_prefix_lens,
        batch.extend_prefix_lens_cpu,
        batch.extend_seq_lens,
        batch.req_pool_indices,
        runner.req_to_token_pool.req_to_token,
        batch.seq_lens_sum,
        runner.token_to_kv_pool.get_kv_buffer_shape()[0],
        runner.kv_cache_dtype,
        runner.device,
        create_chunked_prefix_cache_kv_indices,
    )
    from sglang.srt.hardware_backend.npu.feature_evidence import record

    record("graph-piecewise-dcp", dcp_size=get_parallel().attn_dcp_size)
