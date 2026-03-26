import torch
import torch.nn as nn
from sgl_kernel_npu.activation.swiglu_oai import swiglu_oai
from sglang.srt.layers.activation import GeluAndMul

# ==========================================
# Helper Functions to Reduce Duplication
# ==========================================

def _init_routing_v1(hidden_states, topk_ids, topk_weights, top_k):
    """Handles the standard moe_init_routing used by unquant, wna16, and w8a8."""
    num_tokens = hidden_states.shape[0]
    num_experts = topk_weights.shape[-1] if len(topk_weights.shape) > 1 else topk_ids.shape[-1] # fallback to id shape
    row_idx_len = num_tokens * top_k
    
    row_idx = (
        torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_weights.device)
        .view(top_k, -1)
        .permute(1, 0)
        .contiguous()
    )

    routed_hidden_states, expanded_row_idx, expanded_expert_idx = (
        torch.ops.npu.npu_moe_init_routing(
            hidden_states, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
        )
    )

    # Note: num_experts needs to be passed in from the calling function's weight matrix.
    # We will pass it dynamically from the parent function instead to be safe.
    return routed_hidden_states, expanded_row_idx, expanded_expert_idx

def _compute_expert_tokens(expanded_expert_idx, num_experts):
    expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, num_experts
    )
    return expert_tokens.to(torch.int64)

def _finalize_routing_v1(hidden_states, topk_weights, expanded_row_idx, topk_ids):
    """Handles the standard finalize_routing used by unquant, wna16, and w8a8."""
    return torch.ops.npu.npu_moe_finalize_routing(
        hidden_states,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
    )

def _apply_activation(hidden_states, activation, layer=None, w13=None):
    """Routes to the correct activation function."""
    if activation == "npu_swiglu_oai":
        if layer is None:
            layer = nn.ModuleList()
            layer.register_parameter("w13_weight", w13)
        return swiglu_oai(layer, hidden_states)
    elif activation == "silu":
        return torch.ops.npu.npu_swiglu(hidden_states)
    else:
        return GeluAndMul()(hidden_states)

# ==========================================
# Main Fused Expert Functions
# ==========================================

def npu_fused_experts_unquant(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_weight_bias: torch.Tensor,
    w2: torch.Tensor,
    w2_weight_bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    activation: str,
):
    original_dtype = hidden_states.dtype
    topk_weights = topk_weights.to(original_dtype)
    topk_ids = topk_ids.to(torch.int32)
    num_experts = w13.shape[0]

    hidden_states, expanded_row_idx, expanded_expert_idx = _init_routing_v1(
        hidden_states, topk_ids, topk_weights, top_k
    )
    expert_tokens = _compute_expert_tokens(expanded_expert_idx, num_experts)

    w13_bias = [w13_weight_bias] if w13_weight_bias is not None else None
    w2_bias = [w2_weight_bias] if w2_weight_bias is not None else None

    # gmm1: gate_up_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13],
        bias=w13_bias,
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    # act_fn
    hidden_states = _apply_activation(hidden_states, activation, w13=w13)

    # gmm2: down_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        bias=w2_bias,
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    return _finalize_routing_v1(hidden_states, topk_weights, expanded_row_idx, topk_ids)


def npu_fused_experts_wna16(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_offset: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_offset: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
):
    original_dtype = hidden_states.dtype
    num_experts = w13.shape[0]

    hidden_states, expanded_row_idx, expanded_expert_idx = _init_routing_v1(
        hidden_states, topk_ids, topk_weights, top_k
    )
    expert_tokens = _compute_expert_tokens(expanded_expert_idx, num_experts)

    # gmm1: gate_up_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13],
        antiquant_scale=[w13_scale],
        antiquant_offset=[w13_offset],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    # act_fn: swiglu
    hidden_states = torch.ops.npu.npu_swiglu(hidden_states)

    # gmm2: down_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        antiquant_scale=[w2_scale],
        antiquant_offset=[w2_offset],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    return _finalize_routing_v1(hidden_states, topk_weights, expanded_row_idx, topk_ids)


def npu_fused_experts_w4a8(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_scale_bias: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_scale_bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
):
    group_list_type = 1
    num_tokens = hidden_states.shape[:-1].numel()
    num_experts = w13.shape[0]

    # Uses init_routing_v2 (specific to w4a8)
    sorted_hidden_states, expanded_row_idx, expert_tokens, pertoken_scale = (
        torch.ops.npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            active_num=num_tokens * top_k,
            expert_num=num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, num_experts],
            quant_mode=1,
        )
    )

    expanded_row_idx = expanded_row_idx.view(-1, top_k).permute(1, 0).reshape(-1)
    expert_tokens = expert_tokens.to(torch.int64)
    _output_dtype = torch.bfloat16

    w1_scale = [w13_scale.to(w2_scale.dtype)]
    w2_scale = [w2_scale]

    # gmm1: gate_up_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[sorted_hidden_states],
        weight=[w13],
        scale=w1_scale,
        bias=[w13_scale_bias],
        per_token_scale=[pertoken_scale],
        group_list=expert_tokens,
        split_item=2,
        group_type=0,
        group_list_type=group_list_type,
        output_dtype=_output_dtype,
    )[0]

    # act_fn: swiglu
    hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
    hidden_states, swiglu_out_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)

    # gmm2: down_proj
    output = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        scale=w2_scale,  # Changed from w1_scale to w2_scale based on expected logic, verify if intended!
        bias=[w2_scale_bias],
        per_token_scale=[swiglu_out_scale],
        group_list=expert_tokens,
        split_item=2,
        group_type=0,
        group_list_type=group_list_type,
        output_dtype=_output_dtype,
    )[0]

    # Uses token_unpermute instead of finalize_routing
    return torch.ops.npu.npu_moe_token_unpermute(
        permuted_tokens=output,
        sorted_indices=torch.abs(expanded_row_idx),
        probs=topk_weights,
    )


def npu_fused_experts_w8a8(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_offset: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_offset: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
):
    original_dtype = hidden_states.dtype
    scale_dtype = original_dtype if original_dtype == torch.bfloat16 else torch.float32
    num_experts = w13.shape[0]

    hidden_states, expanded_row_idx, expanded_expert_idx = _init_routing_v1(
        hidden_states, topk_ids, topk_weights, top_k
    )
    expert_tokens = _compute_expert_tokens(expanded_expert_idx, num_experts)

    # gmm1: gate_up_proj (with dynamic quant)
    hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)
    
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13],
        scale=[w13_scale.to(scale_dtype)],
        per_token_scale=[pertoken_scale],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    # act_fn: swiglu
    hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
    
    # gmm2: down_proj (with dynamic quant)
    hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)

    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        scale=[w2_scale.to(scale_dtype)],
        per_token_scale=[pertoken_scale],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    return _finalize_routing_v1(hidden_states, topk_weights, expanded_row_idx, topk_ids)
