"""DSA-CP full-head attention and token-local residual execution (BDE D1v3)."""

import torch
import torch.nn.functional as F
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.feature_evidence import record
from sglang.srt.layers.attention.dsa.dsa_cp import (
    _MISSING,
    dsa_cp_reduce_scatter_tokens,
    dsa_cp_slice,
    get_dsa_cp_plan,
)
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.runtime_context import get_forward, get_parallel


def reset_forward(batch, padded_rows):
    batch.npu_dsa_cp_plan = _MISSING
    batch.npu_dsa_cp_input_rows = padded_rows
    batch.npu_dsa_cp_token_sharded = False
    batch.npu_dsa_cp_attention_local = False
    batch.npu_dsa_cp_logits_pruned = False
    batch.npu_indexer_topk_is_local = False
    for key in (
        "npu_dsa_cp_cu_lens",
        "npu_indexer_query_shard",
        "npu_indexer_sin_cos_cache",
        "npu_dsa_cp_indexer_key_lens",
    ):
        if hasattr(batch, key):
            delattr(batch, key)


def full_head_plan(batch, index_topk=None):
    if not (
        envs.SGLANG_NPU_DSA_CP_Q_FULL_WIDTH.get()
        or envs.SGLANG_NPU_DSA_CP_TOKEN_SHARDED.get()
    ):
        return None
    if batch.attn_dcp_metadata is None or not batch.forward_mode.is_extend():
        return None
    if getattr(batch, "can_run_tbo", False):
        raise ValueError(
            "BDE residual sharding requires sequential layer execution without TBO"
        )
    from sglang.srt.model_executor.runner import get_is_capture_mode

    if get_is_capture_mode():
        raise ValueError(
            "DSA-CP full-head extend is eager-only; disable prefill graph capture"
        )
    return get_dsa_cp_plan(batch, index_topk)


def _read_norm(hidden, batch, norm, plan=None):
    stream = residual_batch.stream_of(batch)
    output, residual = stream.export(hidden)
    if plan is not None:
        output = dsa_cp_slice(output, plan) if output.shape[0] != plan.rows else output
        if residual is not None and residual.shape[0] != plan.rows:
            residual = dsa_cp_slice(residual, plan)
    if residual is None:
        residual = output
        normalized = norm(output)
    else:
        normalized, residual = norm(output, residual)
    stream.write(residual)
    return normalized


def decoder_forward(
    layer, positions, hidden, batch, allocator, previous_topk=None, *, plan=None
):
    plan = (
        plan
        if plan is not None
        else full_head_plan(
            batch,
            layer.self_attn.indexer.index_topk
            if layer.self_attn.indexer is not None
            else None,
        )
    )
    if plan is None:
        if getattr(batch, "npu_dsa_cp_token_sharded", False):
            raise RuntimeError("DSA residual sharding cannot change within a forward")
        return None
    batch.npu_dsa_cp_token_sharded = True
    normalized = _read_norm(hidden, batch, layer.input_layernorm, plan)
    output = layer.self_attn(
        positions=positions,
        hidden_states=normalized,
        forward_batch=batch,
        zero_allocator=allocator,
        input_on_attn_tp_slices=False,
        prev_topk_indices=previous_topk,
    )
    if isinstance(output, tuple):
        output, topk = output
    else:
        topk = None
    stream = residual_batch.stream_of(batch)
    output = stream.record(output, PLAIN_ADD)
    normalized = _read_norm(output, batch, layer.post_attention_layernorm)
    # Sparse MoE gathers its quantized local input in its own forward. Dense
    # layers still need every token before their tensor-parallel projection.
    if not layer.is_layer_sparse:
        group = get_parallel().attn_tp_group
        full = normalized.new_empty((plan.num_tokens_pad, normalized.shape[-1]))
        group.all_gather_into_tensor(full, normalized.contiguous())
        normalized = full
    with get_forward().scoped(npu_dsa_cp_token_sharded=layer.is_layer_sparse):
        output = layer.mlp(normalized, batch, None)
    if not layer.mlp.reduce_results:
        output = dsa_cp_reduce_scatter_tokens(output)
    if output.shape[0] != plan.rows:
        raise RuntimeError("DSA-CP FFN must return this rank's local residual rows")
    record("dsa-residual-shard")
    return stream.record(output, PLAIN_ADD), topk


def gather_logit_rows(hidden, batch):
    from sglang.srt.layers.attention.dsa.dsa_cp import dsa_cp_gather_last_token_rows

    wants_input = bool(
        batch.return_logprob
        and batch.extend_logprob_start_lens_cpu is not None
        and any(
            e > start
            for e, start in zip(
                batch.extend_seq_lens_cpu, batch.extend_logprob_start_lens_cpu
            )
        )
    )
    batch.npu_dsa_cp_logits_pruned = not wants_input
    if not wants_input:
        record("dsa-last-token-gather")
        return dsa_cp_gather_last_token_rows(hidden, batch)
    # Input logprobs need all positions: gather final normalized rows only,
    # keeping attention and every preceding residual update token-local.
    group = get_parallel().attn_tp_group
    full = hidden.new_empty((hidden.shape[0] * group.world_size, hidden.shape[1]))
    group.all_gather_into_tensor(full, hidden.contiguous())
    record("dsa-input-logprob-gather")
    return full


def prepare_attention(m, positions, hidden, batch, allocator, previous_topk, plan):
    from sglang.srt.hardware_backend.npu.modules import (
        deepseek_v2_attention_mla_npu as mla,
    )

    batch.npu_dsa_cp_attention_local = True
    if hidden.shape[0] != plan.rows:
        hidden = dsa_cp_slice(hidden, plan)
    local_positions = (
        dsa_cp_slice(positions, plan) if positions.shape[0] != plan.rows else positions
    )
    if mla._dsa_dcp_extend_gathers_kv(batch):
        mla._prepare_dcp_prefix(m, batch, hidden)
    qkv = m.fused_qkv_a_proj_with_mqa(hidden)[0]
    q, latent = qkv.split([m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim], dim=-1)
    q_lora = m.q_a_layernorm(q)
    q = F.linear(q_lora, m.q_b_proj_qrep_weight).view(plan.rows, -1, m.qk_head_dim)
    q_nope, q_pe = q.split([m.qk_nope_head_dim, m.qk_rope_head_dim], dim=-1)
    key_nope, key_pe = latent.unsqueeze(1).split(
        [m.kv_lora_rank, m.qk_rope_head_dim], dim=-1
    )
    key_nope = m.kv_a_layernorm(key_nope)
    q_nope = torch.bmm(q_nope.transpose(0, 1), m.w_kc_qrep).transpose(0, 1)
    if mla.is_mla_preprocess_enabled() and not m.rotary_emb.is_neox_style:
        q_pe, key_pe = mla._apply_interleaved_rope_with_half_output(
            m.rotary_emb, local_positions, q_pe, key_pe
        )
    else:
        m.rotary_emb.sin_cos_cache = m.rotary_emb.cos_sin_cache.index_select(
            0, local_positions
        )
        q_pe, key_pe = m.rotary_emb(local_positions, q_pe, key_pe)
    group = get_parallel().attn_tp_group
    keys, works, sends = [], [], []
    for value in (key_nope, key_pe):
        send = value.contiguous()
        full = send.new_empty((plan.num_tokens_pad, *send.shape[1:]))
        works.append(group.all_gather_into_tensor_async(full, send))
        sends.append(send)
        keys.append(full)
    batch.npu_dsa_cp_chunk_kv = (works, sends)
    if not m.skip_topk or (m.is_nextn and previous_topk is None):
        topk = m.indexer(
            hidden, q_lora, local_positions, batch, m.layer_id, False, None
        )
    else:
        topk = previous_topk
    if topk is None:
        raise RuntimeError("DSA-CP attention requires top-k indices")
    if not getattr(batch, "npu_indexer_topk_is_local", False):
        topk = dsa_cp_slice(topk, plan)
    # All subsequent skip-indexer layers consume the very same local rows.
    batch.npu_indexer_topk_is_local = True
    record("dsa-full-width-q")
    return q_pe, keys[1], q_nope, keys[0], topk, batch, allocator, positions, False


def sparse_attention_or_empty(m, q_nope, q_pe, key_nope, key_pe, indices, batch, plan):
    if plan.num_local_tokens == 0:
        # Every rank must publish its owned KV even if it owns no queries.
        # Zero cumulative Q lengths are not accepted by every SFA release.
        get_token_to_kv_pool().set_kv_buffer(
            m.attn_mqa_for_dsa_cp, batch.out_cache_loc, key_nope, key_pe
        )
        record("dsa-empty-query-rank")
        return q_nope.new_zeros((plan.rows, q_nope.shape[1] * m.kv_lora_rank))
    return m.attn_mqa_for_dsa_cp(
        q_nope.contiguous(),
        key_nope,
        key_nope,
        batch,
        q_rope=q_pe.contiguous(),
        k_rope=key_pe,
        topk_indices=indices,
        save_kv_cache=True,
    )


def core_attention(m, q_pe, key_pe, q_nope, key_nope, topk, batch, gate=None):
    from sglang.srt.hardware_backend.npu.modules import (
        deepseek_v2_attention_mla_npu as mla,
    )
    from sglang.srt.layers.attention.dsa.dsa_cp import dsa_cp_restore_tokens

    plan = get_dsa_cp_plan(batch)
    works, sends = batch.npu_dsa_cp_chunk_kv
    for work in works:
        if work is not None:
            work.wait()
    batch.npu_dsa_cp_chunk_kv = None
    batch.npu_dcp_extend_kv = mla._take_dcp_prefix(m, batch, key_nope, key_pe)
    indices = topk
    packed = batch.npu_dcp_prefix_layout
    if packed is not None:
        indices = mla.dcp_layout.remap_topk_to_packed(indices, packed)
    output = sparse_attention_or_empty(
        m, q_nope, q_pe, key_nope, key_pe, indices, batch, plan
    )
    batch.npu_dcp_extend_kv = None
    if batch.npu_dsa_cp_token_sharded:
        heads = m.w_vc_qrep.shape[0]
        output = (
            torch.bmm(
                output.reshape(plan.rows, heads, m.kv_lora_rank).transpose(0, 1),
                m.w_vc_qrep,
            )
            .transpose(0, 1)
            .reshape(plan.rows, -1)
        )
        if gate is not None:
            raise ValueError(
                "DSA residual sharding does not support gated-attention subclasses"
            )
        if envs.SGLANG_NPU_STREAM_O_PROJ.get():
            if envs.SGLANG_NPU_ENABLE_DCP_GATHER_PREFETCH.get():
                weight = mla._dcp_pipeline.take_o_proj(m.layer_id)
            else:
                from sglang.srt.hardware_backend.npu.attention.dcp_gather import (
                    start_o_proj_gather,
                )
                from sglang.srt.layers.attention.dsa.dsa_cp import (
                    get_o_proj_stream_shard,
                )

                gathered, send, work = start_o_proj_gather(
                    get_parallel().attn_tp_group, get_o_proj_stream_shard(m.layer_id)
                )
                if work is not None:
                    work.wait()
                weight = gathered.t()
                record("dsa-o-proj-stream")
        else:
            weight = m.o_proj_qrep_weight
        output = F.linear(output, weight)
    else:
        output = dsa_cp_restore_tokens(
            output.reshape(plan.rows, -1, m.kv_lora_rank), plan, plan.num_tokens_pad
        )
        output = (
            torch.bmm(
                output.reshape(-1, m.num_local_heads, m.kv_lora_rank).transpose(0, 1),
                m.w_vc,
            )
            .transpose(0, 1)
            .reshape(plan.num_tokens_pad, -1)
        )
        if gate is not None:
            output = m._apply_gated(output, gate)
        output = m.o_proj(output)[0]
    return output, topk if m.next_skip_topk else None
