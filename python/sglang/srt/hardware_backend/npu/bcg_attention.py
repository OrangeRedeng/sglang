"""Explicit eager DSA islands between captured NPU FFN segments."""
from contextvars import ContextVar
import torch
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import eager_on_graph

ACTIVE = ContextVar("npu_bcg_attention_island", default=False)


def describe(value):
    if torch.is_tensor(value):
        return ("tensor", tuple(value.shape), value.dtype, value.device)
    if value is None:
        return ("none",)
    if isinstance(value, tuple):
        return ("tuple", tuple(describe(x) for x in value))
    raise TypeError("Unsupported DSA graph boundary output")


def allocate(description):
    if description[0] == "tensor":
        return torch.empty(description[1], dtype=description[2], device=description[3])
    if description[0] == "none":
        return None
    return tuple(allocate(x) for x in description[1])


def output_factory(owner, *args, **kwargs):
    if not hasattr(owner, "_npu_bcg_attention_layout"):
        raise RuntimeError("DSA BCG boundary requires eager warmup")
    return allocate(owner._npu_bcg_attention_layout)


@eager_on_graph(output_factory=output_factory)
def attention(owner, positions, hidden_states, forward_batch, zero_allocator,
              input_on_attn_tp_slices=False, llama_4_scaling=None, prev_topk_indices=None):
    if torch.npu.is_current_stream_capturing():
        raise RuntimeError("Dynamic DSA island must execute outside native capture")
    # These caches depend on request lengths; graph input storage is refreshed in place.
    for key in ("npu_dsa_token_shard_plan", "npu_dcp_crop_free", "npu_dcp_packed_plan", "npu_dcp_packed_kv_lens"):
        if key in vars(forward_batch):
            delattr(forward_batch, key)
    token = ACTIVE.set(True)
    try:
        result = owner._forward_npu_bcg_body(positions, hidden_states, forward_batch,
            zero_allocator, input_on_attn_tp_slices, llama_4_scaling, prev_topk_indices)
        layout = describe(result)
        # Capture factory freezes shape per graph; replay copy validates every result.
        owner._npu_bcg_attention_layout = layout
        return result
    finally:
        ACTIVE.reset(token)
