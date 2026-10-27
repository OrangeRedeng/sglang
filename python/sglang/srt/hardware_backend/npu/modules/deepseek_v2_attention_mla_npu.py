import logging
from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple

import torch
import torch_npu
from sgl_kernel_npu.norm.fused_split_qk_norm import fused_split_qk_norm

import sglang.srt.layers.dcp.comm as dcp_comm
import sglang.srt.layers.dcp.layout as dcp_layout
import sglang.srt.model_executor.forward_context as forward_context
import sglang.srt.runtime_context as runtime_context
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.attention.mla_preprocess import (
    NPUFusedMLAPreprocess,
    is_fia_nz,
    is_mla_preprocess_enabled,
)
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.srt.layers.attention.dsa.dsa_cp import (
    dsa_cp_redistribute_heads,
    dsa_cp_redistribute_heads_async,
    dsa_cp_redistribute_heads_recv,
    dsa_cp_restore_tokens,
    dsa_cp_set_token_sharded,
    dsa_cp_slice,
    dsa_cp_token_sharded,
    dsa_cp_token_sharded_enabled,
    get_dsa_cp_plan,
    get_o_proj_stream_shard,
)
from sglang.srt.layers.attention.dsa.dsa_npu_indexer import scattered_to_tp_attn_full
from sglang.srt.layers.attention.dsa.utils import (
    dsa_use_prefill_cp,
)
from sglang.srt.layers.dcp import (
    all_gather_q_for_mla_decode,
    cp_lse_ag_out_rs_mla,
    dcp_a2a_lse_reduce,
)
from sglang.srt.layers.dcp.layout import (
    DcpPackedReadPlan,
    dcp_extend_gather_buffer,
    dcp_packed_read_plan,
    plan_dcp_extend_gather,
    remap_topk_to_packed,
)
from sglang.srt.layers.layer_boundary import get_attn_tp_context
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    is_mla_dcp_lse_base_on_e,
)
from sglang.srt.runtime_context import get_disagg, get_parallel
from sglang.srt.state_capturer.indexer_topk import maybe_capture_indexer_topk
from sglang.srt.utils.common import print_info_once

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
    from sglang.srt.utils import BumpAllocator

logger = logging.getLogger(__name__)

_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()
_is_npu_arch35 = is_npu_arch35()
_debug_dcp_extend_memory = envs.SGLANG_DEBUG_NPU_DCP_EXTEND_MEMORY.get()


def _use_dsa_dcp_partial_attention(forward_batch: "ForwardBatch") -> bool:
    return (
        runtime_context.get_parallel().dcp_enabled
        and not forward_context.get_attn_backend().is_draft_worker
        and not dsa_use_prefill_cp(forward_batch)
        and not forward_batch.forward_mode.is_idle()
    )


def _is_dcp_mla_decode_phase(forward_batch: "ForwardBatch") -> bool:
    # The bde NPU path keeps its own decode-phase predicate: upstream moved
    # is_dcp_mla_decode_phase onto DeepseekMLAForwardMixin with extended
    # semantics (it also admits the DSA extend phase), which would re-route
    # the D1v3/D2v3 seams below. Keep the narrow decode/target-verify form
    # the NPU prepare/core pairs were built against.
    if not get_parallel().dcp_enabled:
        return False
    mode = forward_batch.forward_mode
    return mode.is_decode() or mode.is_target_verify()


# region MHA
def forward_mha_prepare_npu(
    m: "DeepseekV2AttentionMLA",
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    input_on_attn_tp_slices: bool,
):
    if m.q_lora_rank is not None:
        q, latent_cache = (
            get_attn_tp_context()
            .fetch_qkv_latent()
            .split(
                [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim],
                dim=-1,
            )
        )

        # DSA Indexer: cache quantized keys, auto-skip topk for sequences <= dsa_index_topk

        if m.use_dsa:
            q_lora = m.q_a_layernorm(q)
            q = m.q_b_proj(q_lora)[0].view(-1, m.num_local_heads, m.qk_head_dim)
            _ = m.indexer(
                x=hidden_states,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=m.layer_id,
                return_indices=False,
            )

        else:
            q = m.q_a_layernorm(q)
            if _use_ag_after_qlora and input_on_attn_tp_slices:
                q = scattered_to_tp_attn_full(q, forward_batch)
                latent_cache = scattered_to_tp_attn_full(latent_cache, forward_batch)
            q = m.q_b_proj(q)[0].view(-1, m.num_local_heads, m.qk_head_dim)

    else:
        q = m.q_proj(hidden_states)[0].view(-1, m.num_local_heads, m.qk_head_dim)
        latent_cache = m.kv_a_proj_with_mqa(hidden_states)[0]

    _, q_pe = q.split([m.qk_nope_head_dim, m.qk_rope_head_dim], dim=-1)
    kv_a, _ = latent_cache.split([m.kv_lora_rank, m.qk_rope_head_dim], dim=-1)
    latent_cache = latent_cache.unsqueeze(1)

    if m.use_deepseek_yarn_rope:
        B, S = q.shape[0], 1
        cos, sin = m.rotary_emb.get_cos_sin_cache(
            positions, hidden_states.dtype, offsets=None
        )
        q_pe = torch_npu.npu_interleave_rope(
            q_pe.reshape(B, -1, S, m.qk_rope_head_dim),
            cos,
            sin,
        )
        q_pe = q_pe.reshape(B, -1, m.qk_rope_head_dim)

        ckv_cache, k_rope_cache = get_token_to_kv_pool().get_kv_buffer(m.layer_id)
        _, _, k_pe, kv_a = torch_npu.npu_kv_rmsnorm_rope_cache(
            latent_cache.view(-1, 1, 1, m.kv_lora_rank + m.qk_rope_head_dim),  # bnsd
            m.kv_a_layernorm.weight,
            cos,
            sin,
            forward_batch.out_cache_loc.to(torch.int64),
            k_rope_cache,
            ckv_cache,
            k_rope_scale=None,
            c_kv_scale=None,
            k_rope_offset=None,
            c_kv_offset=None,
            epsilon=m.kv_a_layernorm.variance_epsilon,
            cache_mode="PA_NZ" if is_fia_nz() else "PA_BNSD",
            is_output_kv=True,
        )  # adapter NZ

        k_pe = k_pe.reshape(B, -1, m.qk_rope_head_dim)
    else:
        kv_a = m.kv_a_layernorm(kv_a)
        k_pe = latent_cache[:, :, m.kv_lora_rank :]
        if m.rotary_emb is not None:
            q_pe, k_pe = m.rotary_emb(positions, q_pe, k_pe)
        # this is for model kimi-vl-a3B-instruct
        get_token_to_kv_pool().set_kv_buffer(
            m, forward_batch.out_cache_loc, kv_a.unsqueeze(1), k_pe
        )

    q[..., m.qk_nope_head_dim :] = q_pe

    kv = m.kv_b_proj(kv_a)[0]
    kv = kv.view(-1, m.num_local_heads, m.qk_nope_head_dim + m.v_head_dim)
    k_nope = kv[..., : m.qk_nope_head_dim]
    v = kv[..., m.qk_nope_head_dim :]

    k = m._concat_and_cast_mha_k(k_nope, k_pe, forward_batch)
    return q, k, v, forward_batch


def forward_mha_core_npu(
    m: "DeepseekV2AttentionMLA",
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    forward_batch: "ForwardBatch",
    # Gated attention (Ling-V3 / BailingMoeV3): the subclass appends its gate
    # to inner_state, so every *_core dispatched from forward_core takes it as
    # a trailing arg. None everywhere else.
    gate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_output = m.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
    attn_output = attn_output.reshape(-1, m.num_local_heads * m.v_head_dim)
    if gate is not None:
        attn_output = m._apply_gated(attn_output, gate)
    output, _ = m.o_proj(attn_output)
    return output


# endregion


# region MLA
def forward_mla_prepare_npu(
    m: "DeepseekV2AttentionMLA",
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    input_on_attn_tp_slices: bool,
):
    if is_mla_preprocess_enabled():
        if not hasattr(m, "mla_preprocess"):
            m.mla_preprocess = NPUFusedMLAPreprocess(
                m.fused_qkv_a_proj_with_mqa,
                m.q_a_layernorm,
                m.kv_a_layernorm,
                m.q_b_proj,
                m.w_kc,
                m.rotary_emb,
                m.layer_id,
                m.num_local_heads,
                m.qk_nope_head_dim,
                m.qk_rope_head_dim,
                m.quant_config,
            )
        (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            forward_batch,
            zero_allocator,
            positions,
        ) = m.mla_preprocess.forward(
            positions, hidden_states, forward_batch, zero_allocator
        )
        topk_indices = None
    else:
        q_lora = None
        if m.q_lora_rank is not None:
            qkv_latent = get_attn_tp_context().fetch_qkv_latent()
            if _use_ag_after_qlora and input_on_attn_tp_slices:
                q, latent_cache = qkv_latent.split(
                    [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim],
                    dim=-1,
                )
                k_nope = latent_cache[..., : m.kv_lora_rank]

                q = m.q_a_layernorm(q)
                q = scattered_to_tp_attn_full(q, forward_batch)
                latent_cache = scattered_to_tp_attn_full(latent_cache, forward_batch)

                k_nope = m.kv_a_layernorm(k_nope).unsqueeze(1)
                k_pe = latent_cache[..., m.kv_lora_rank :].unsqueeze(1)
            else:
                if (
                    qkv_latent.shape[0] < 65536
                    and not dsa_use_prefill_cp(forward_batch)
                    and not getattr(m, "_disable_npu_fused_split_qk_norm", False)
                ):
                    q, k_nope, k_pe = fused_split_qk_norm(
                        qkv_latent,
                        m.q_a_layernorm,
                        m.kv_a_layernorm,
                        m.q_lora_rank,
                        m.kv_lora_rank,
                        m.qk_rope_head_dim,
                        eps=m.q_a_layernorm.variance_epsilon,
                    )
                else:
                    # The fused split+RMSNorm kernel is not numerically equivalent
                    # on Ascend. Keep the unfused path for models that opt out.
                    q, latent_cache = qkv_latent.split(
                        [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim],
                        dim=-1,
                    )
                    k_nope = latent_cache[..., : m.kv_lora_rank]

                    q = m.q_a_layernorm(q)

                    k_nope = m.kv_a_layernorm(k_nope).unsqueeze(1)
                    k_pe = latent_cache[..., m.kv_lora_rank :].unsqueeze(1)

            # q_lora needed by indexer
            if m.use_dsa:
                q_lora = q

            q = m.q_b_proj(q)[0].view(-1, m.num_local_heads, m.qk_head_dim)
        else:
            q = m.q_proj(hidden_states)[0].view(-1, m.num_local_heads, m.qk_head_dim)
            latent_cache = m.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope = latent_cache[..., : m.kv_lora_rank]
            k_nope = m.kv_a_layernorm(k_nope).unsqueeze(1)
            k_pe = latent_cache[..., m.kv_lora_rank :].unsqueeze(1)

        q_nope, q_pe = q.split([m.qk_nope_head_dim, m.qk_rope_head_dim], dim=-1)

        q_nope_out = torch.bmm(q_nope.transpose(0, 1), m.w_kc)

        q_nope_out = q_nope_out.transpose(0, 1)

        if m.rotary_emb is not None:
            q_pe, k_pe = m.rotary_emb(positions, q_pe, k_pe)

        if dsa_use_prefill_cp(forward_batch):
            # support allgather+rerrange
            k_nope, k_pe = m.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )
        topk_indices = None
        if q_lora is not None:
            topk_indices = m.indexer(
                x=hidden_states,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=m.layer_id,
            )

    return (
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
    )


def forward_mla_core_npu(
    m: "DeepseekV2AttentionMLA",
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    q_nope_out: torch.Tensor,
    k_nope: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    positions: torch.Tensor,
    topk_indices: torch.Tensor,
    # Gated attention (Ling-V3 / BailingMoeV3): the subclass appends its gate
    # to inner_state, so every *_core dispatched from forward_core takes it as
    # a trailing arg. None everywhere else.
    gate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_output = m.attn_mqa(
        q_nope_out,
        k_nope,
        k_nope,
        forward_batch,
        q_rope=q_pe,
        k_rope=k_pe,
        **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
    )

    attn_output = attn_output.view(-1, m.num_local_heads, m.kv_lora_rank)

    attn_output = attn_output.contiguous()
    if (
        attn_output.shape[0] >= 65536
        or attn_output.shape[-1] * attn_output.shape[-2] >= 65536
        or m.w_vc.shape[-1] >= 65536
    ):
        # npu_transpose_batchmatmul does not support dimensions >= 65536.
        attn_bmm_output = torch.empty(
            (attn_output.shape[0], m.num_local_heads, m.v_head_dim),
            dtype=attn_output.dtype,
            device=attn_output.device,
        )
        torch.ops.npu.batch_matmul_transpose(attn_output, m.w_vc, attn_bmm_output)
    else:
        # Use the numerically validated torch_npu implementation when supported.
        attn_bmm_output = torch_npu.npu_transpose_batchmatmul(
            attn_output,
            m.w_vc,
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        )

    attn_bmm_output = attn_bmm_output.reshape(-1, m.num_local_heads * m.v_head_dim)
    if gate is not None:
        attn_bmm_output = m._apply_gated(attn_bmm_output, gate)
    output, _ = m.o_proj(attn_bmm_output)

    return output


# endregion


# region DSA
def _apply_interleaved_rope_with_half_output(rotary_emb, positions, q_pe, k_pe):
    """Apply RoPE to interleaved Q/K and return half-layout outputs."""
    rotary_emb.get_cos_sin_with_position(positions)
    cos = rotary_emb.position_cos.to(device=q_pe.device, dtype=q_pe.dtype).view(
        -1, 1, 1, q_pe.shape[-1]
    )
    sin = rotary_emb.position_sin.to(device=q_pe.device, dtype=q_pe.dtype).view(
        -1, 1, 1, q_pe.shape[-1]
    )
    q_pe = torch_npu.npu_interleave_rope(q_pe.unsqueeze(2), cos, sin).squeeze(2)
    k_pe = torch_npu.npu_interleave_rope(k_pe.unsqueeze(2), cos, sin).squeeze(2)
    return q_pe, k_pe


def forward_dsa_prepare_npu(
    m: "DeepseekV2AttentionMLA",
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    input_on_attn_tp_slices: bool,
    prev_topk_indices: torch.Tensor = None,
):
    dynamic_scale = None
    # Resolve DSA-CP once per forward, here rather than in the core, because
    # this is the half of the pair that receives layer_scatter_modes and the
    # gate needs it: the slice assumes this rank was handed the whole batch.
    # The result is cached on the batch and the core reads it back.
    # index_topk decides whether the multi-request lift may drop the operator's
    # causal crop, and this is the FIRST place in a forward that can supply it.
    # m.indexer exists only on the layers that compute a top-k
    # (deepseek_v2.py:1828) -- layer 0 is one of them, and it is layer 0 that
    # resolves the cached plan -- but None is answered "keep the crop" rather
    # than trusted, so a model that ordered its layers differently degrades
    # instead of breaking.
    dsa_cp_plan = get_dsa_cp_plan(
        forward_batch,
        layer_scatter_modes,
        m.indexer.index_topk if m.indexer is not None else None,
    )
    mla_preprocess_used = (
        is_mla_preprocess_enabled()
        and not forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
    )
    # Whether this layer runs the indexer. The gather placement below and the
    # indexer call both branch on this, so the expression lives in one place.
    indexer_runs = not m.skip_topk or (m.is_nextn and prev_topk_indices is None)
    # Version B of the DSA-CP Q all-to-all removal: the gathered full-head
    # weights let this layer compute every head for this rank's tokens, so the
    # QKV and the w_kc matmul below take the full-width branch and the core
    # skips the head redistribution.
    q_full_width = (
        _enable_dsa_cp_q_full_width
        and dsa_cp_plan is not None
        and not mla_preprocess_used
        and getattr(m, "q_b_proj_qrep_weight", None) is not None
    )
    # D1v3: the token-sharded rebuild. It needs the version B full-head Q
    # weights (which it reuses) plus the full w_vc / o_proj prepared by the
    # model runner. The residual input is sliced to this rank's rows here
    # (layer 0: right after the embedding allReduce; layers 1+: it already
    # arrived sliced), the per-token projections run on the slice, and the
    # KV latent and the index-K are gathered back to full width for the pool
    # writes (seams S1/S2). The output side (restore a2a, O-proj allReduce,
    # MoE allReduce) is replaced further down.
    if dsa_cp_token_sharded_enabled() and dsa_cp_plan is not None:
        from sglang.srt.model_executor.runner import get_is_capture_mode

        if get_is_capture_mode():
            raise RuntimeError(
                "SGLANG_NPU_DSA_CP_TOKEN_SHARDED is on and a graph capture is "
                "in progress: the rebuild's collectives depend on the batch's "
                "token grid and cannot be captured. Disable graph capture for "
                "extend (the box runs with it off) or turn the flag off. "
                "Decode capture does not reach this check."
            )
    # A forward that asks for INPUT logprobs needs the hidden state at every
    # new-token position, but the rebuild's logits tail only gathers the
    # last-token rows (that is what makes it cheap). So an input-logprob
    # forward must stay unsharded, while an OUTPUT-logprob-only forward is
    # fine: its next_token_logits come from exactly the gathered rows (the
    # scheduler turns them into the output top-logprobs). Mirror
    # LogitsMetadata.from_forward_batch so the two never drift; the flag is
    # per-batch, so every rank declines at the same layer.
    wants_input_logprobs = (
        forward_batch.forward_mode.is_extend()
        and forward_batch.return_logprob
        and not forward_batch.forward_mode.is_target_verify()
        and forward_batch.extend_seq_lens_cpu is not None
        and forward_batch.extend_logprob_start_lens_cpu is not None
        and any(
            extend_len - start_len > 0
            for extend_len, start_len in zip(
                forward_batch.extend_seq_lens_cpu,
                forward_batch.extend_logprob_start_lens_cpu,
            )
        )
    )
    # D2v3: the o_proj full matrix comes from the stream when
    # SGLANG_NPU_STREAM_O_PROJ is on (this rank's dequantized shard, registered
    # by layer id by the model runner) and from the static replica otherwise.
    o_proj_present = (
        get_o_proj_stream_shard(m.layer_id) is not None
        if _stream_o_proj
        else getattr(m, "o_proj_qrep_weight", None) is not None
    )
    token_sharded = (
        dsa_cp_token_sharded_enabled()
        and dsa_cp_plan is not None
        and not mla_preprocess_used
        and not wants_input_logprobs
        and getattr(m, "q_b_proj_qrep_weight", None) is not None
        and getattr(m, "w_kc_qrep", None) is not None
        and getattr(m, "w_vc_qrep", None) is not None
        and o_proj_present
    )
    if dsa_cp_token_sharded_enabled() and dsa_cp_plan is not None:
        # The startup gate raised on an incomplete preparation, so reaching
        # here with the flag on, the plan built, and a weight missing is a
        # wiring bug, not a configuration. Check the weights directly rather
        # than `token_sharded`: a forward can legitimately decline the rebuild
        # for input logprobs or MLA preprocess while every weight is present.
        if (
            getattr(m, "q_b_proj_qrep_weight", None) is None
            or getattr(m, "w_kc_qrep", None) is None
            or getattr(m, "w_vc_qrep", None) is None
            or not o_proj_present
        ):
            raise RuntimeError(
                "SGLANG_NPU_DSA_CP_TOKEN_SHARDED is on and the DSA-CP plan is "
                "built, but this layer's full-width weights are incomplete "
                "(q_b/w_kc/w_vc/o_proj"
                + (" shard" if _stream_o_proj else " qrep")
                + "); the model-runner preparation must have run before any "
                "forward"
            )
    dsa_cp_set_token_sharded(forward_batch, token_sharded)

    if token_sharded:
        print_info_once(
            "D1v3 token-sharded rebuild ENGAGED for this extend: the residual "
            "is sharded across the "
            f"{get_parallel().attn_tp_size} attention-TP ranks and every "
            "per-token projection runs on this rank's rows."
        )
        # The residual stream is token-sharded from here on: this rank owns
        # rows [local_start, local_end) of the batch, and every per-token
        # projection in the layer runs on that slice.
        #
        # The attention input is FULL width only on layer 0 (the embedding).
        # From layer 1 on the residual already carries this rank's rows
        # (the MoE reduce-scatter leaves it sharded and prepare_mlp slices
        # it, below), so slicing it a second time would read past a tensor
        # that only has rows left: every rank above 0 would attend over
        # zeros. Guard on the width, not the layer id, exactly as
        # prepare_mlp guards its residual slice.
        if hidden_states.shape[0] != dsa_cp_plan.rows:
            hidden_states = dsa_cp_slice(hidden_states, dsa_cp_plan)
        # positions is the per-forward full token-position vector for every
        # layer (rope and the indexer's cos_sin lookup index by it), so it
        # still needs the row slice on every layer, not just the first.
        if positions.shape[0] != dsa_cp_plan.rows:
            positions = dsa_cp_slice(positions, dsa_cp_plan)

    def _run_q_b(q_in: torch.Tensor) -> torch.Tensor:
        # The three ways to reach the head set. Token-sharded (D1v3) runs the
        # full weight on the rows prepare already sliced: rows x 64 heads,
        # the same FLOPs as the head-sharded GEMM on all rows (rows = N/8 at
        # 8x the columns). Slicing q_in again here would be a NO-OP on rank 0
        # and an EMPTY slice on every other rank -- q_in is [rows], not [N].
        # Version B (D1v2_3b) runs the full weight on every row and slices
        # after the split below. The baseline runs this rank's head shard.
        if token_sharded:
            return torch.nn.functional.linear(q_in, m.q_b_proj_qrep_weight).view(
                -1,
                m.num_local_heads * get_parallel().attn_tp_size,
                m.qk_head_dim,
            )
        if q_full_width:
            return torch.nn.functional.linear(q_in, m.q_b_proj_qrep_weight).view(
                -1,
                m.num_local_heads * get_parallel().attn_tp_size,
                m.qk_head_dim,
            )
        return m.q_b_proj(q_in)[0].view(-1, m.num_local_heads, m.qk_head_dim)

    if _enable_dcp_sfa_gather_overlap and _dcp_extend_active(forward_batch):
        if not indexer_runs:
            # Reuse layer: no indexer to hide behind, so the QKV is the window
            # (the vLLM timing, one step earlier than the vLLM spot). The
            # all-gather moves only this layer's pool prefix rows, and those
            # were written by earlier forwards, so the collective depends on
            # nothing this layer computes -- not even the QKV. Issue it here,
            # before the QKV, and the QKV projection runs underneath it. This
            # chunk's own KV -- the one part the gather cannot carry, because
            # it is the QKV's output -- is attached once the QKV has produced
            # it, below.
            forward_batch.npu_dcp_sfa_gather_pending = _dcp_start_extend_gather_npu(
                m,
                forward_batch,
                hidden_states,
                _dcp_sfa_gather_topk_width(m, prev_topk_indices),
            )
        # Indexer layer: issued right before the indexer, below, where this
        # chunk's KV already exists and the gather's window is the indexer
        # itself (measured 206 ms at extend) rather than the QKV.
    if (
        _dcp_extend_active(forward_batch)
        and _enable_dcp_gather_prefetch
    ):
        # D2 first-layer prime: with the prefetch on, every layer's gather is
        # owned by the prefetcher's take() in the core -- EXCEPT the first
        # layer, which take() would issue after this layer's indexer (the 154
        # ms stall). Prime it here, before the QKV, so the first context-KV
        # gather starts as soon as inference does and finishes before the
        # indexer instead of with it. Only the first DSA layer of the forward
        # actually primes; prime() no-ops for the rest. When the packed plan
        # refuses (multi-request, sub-index_topk) there is nothing to prime and
        # the layer falls back to the permuting path, exactly as before.
        packed_plan = dcp_packed_read_plan(
            forward_batch, _dcp_sfa_gather_topk_width(m, prev_topk_indices)
        )
        if packed_plan is not None:
            _dcp_gather_prefetcher_get().prime(
                m,
                forward_batch.attn_dcp_metadata,
                packed_plan,
                hidden_states,
                m.layer_id,
            )
    if mla_preprocess_used:
        (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            q_lora,
            forward_batch,
            zero_allocator,
            positions,
            dynamic_scale,
        ) = npu_mla_preprocess(
            m,
            hidden_states,
            positions,
            forward_batch,
            zero_allocator,
        )
    else:
        fused_qkv_a_proj_out = m.fused_qkv_a_proj_with_mqa(hidden_states)[0]
        if m.rotary_emb.is_neox_style:
            q, latent_cache = fused_qkv_a_proj_out.split(
                [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim], dim=-1
            )
            # overlap qk norm
            q = m.q_a_layernorm(q)
            if _use_ag_after_qlora and input_on_attn_tp_slices:
                q = scattered_to_tp_attn_full(q, forward_batch)
                latent_cache = scattered_to_tp_attn_full(latent_cache, forward_batch)
            q_lora = q.clone()  # required for topk_indices

            q_event = None
            if m.alt_stream is not None:
                m.alt_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(m.alt_stream):
                    q = _run_q_b(q_lora)
                    # record q to ensure memory space will not be released
                    q.record_stream(m.alt_stream)
                    q_event = m.alt_stream.record_event()
            else:
                q = _run_q_b(q_lora)

            k_nope, k_pe = latent_cache.unsqueeze(1).split(
                [m.kv_lora_rank, m.qk_rope_head_dim], dim=-1
            )
            k_nope = m.kv_a_layernorm(k_nope)
            # main stream waits for the completion of the event on the alt stream to ensure data dependency is complete
            if q_event is not None:
                torch.npu.current_stream().wait_event(q_event)
        else:
            if (
                fused_qkv_a_proj_out.shape[0] < 65535
                and not dsa_use_prefill_cp(forward_batch)
                and not getattr(m, "_disable_npu_fused_split_qk_norm", False)
            ):
                q_lora, k_nope, k_pe = fused_split_qk_norm(
                    fused_qkv_a_proj_out,
                    m.q_a_layernorm,
                    m.kv_a_layernorm,
                    m.q_lora_rank,
                    m.kv_lora_rank,
                    m.qk_rope_head_dim,
                    eps=m.q_a_layernorm.variance_epsilon,
                )
            else:
                # Keep the numerically validated unfused path for models that
                # explicitly opt out of the fused split and RMSNorm kernel.
                q, latent_cache = fused_qkv_a_proj_out.split(
                    [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim], dim=-1
                )
                # overlap qk norm
                q = m.q_a_layernorm(q)

                q_lora = q.clone()  # required for topk_indices
                k_nope, k_pe = latent_cache.unsqueeze(1).split(
                    [m.kv_lora_rank, m.qk_rope_head_dim], dim=-1
                )
                k_nope = m.kv_a_layernorm(k_nope)
            q = _run_q_b(q_lora)

        q_nope, q_pe = q.split([m.qk_nope_head_dim, m.qk_rope_head_dim], dim=-1)

        if q_full_width and not token_sharded:
            # Slice to this rank's rows BEFORE the w_kc batch-matmul: the tmm
            # then costs rows x 64 heads, the same FLOPs as the sharded path's
            # total x 8, and the write it would otherwise make drops from
            # 737 MB to 92 MB at a 16k chunk. dsa_cp_slice pads the tail to
            # plan.rows, the same width the head redistribution would produce.
            # (Token sharded already sliced q_lora before q_b_proj, so its q
            # is at rows x all heads here.)
            q_nope = dsa_cp_slice(q_nope, dsa_cp_plan)

        q_nope_out = torch_npu.npu_transpose_batchmatmul(
            q_nope,
            m.w_kc_qrep if (q_full_width or token_sharded) else m.w_kc,
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        )

        if is_mla_preprocess_enabled() and not m.rotary_emb.is_neox_style:
            # Match the half-layout RoPE outputs used by MLA preprocessing.
            q_pe, k_pe = _apply_interleaved_rope_with_half_output(
                m.rotary_emb, positions, q_pe, k_pe
            )
        else:
            if m.layer_id == get_token_to_kv_pool().start_layer:
                m.rotary_emb.sin_cos_cache = m.rotary_emb.cos_sin_cache.index_select(
                    0, positions
                )
            q_pe, k_pe = m.rotary_emb(positions, q_pe, k_pe)

        if q_full_width and not token_sharded:
            # RoPE ran on the full head set (it is per-head, so the extra
            # heads cost the rope nothing); take this rank's rows now, the
            # same slice q_nope got before the w_kc matmul.
            q_pe = dsa_cp_slice(q_pe, dsa_cp_plan)

        if token_sharded:
            # The chunk-KV allGather (seam S1). The KV latent was computed on
            # this rank's [rows] tokens, but the pool write (the sparse
            # backend scatters k at the FULL out_cache_loc) and the SFA read
            # address every token, so the chunk KV has to go back to full
            # width here. Issued async BEFORE the indexer so the collective
            # overlaps the top-k run (the vLLM-Ascend pattern); the core
            # waits on the handles before anything reads the gathered rows.
            # From that point on the pool write, the DCP gather's local chunk
            # copy, and the SFA read all see the full [N] chunk KV exactly as
            # without the rebuild. Chunk-sized, not context-sized: ~19 MB per
            # layer at a 16.4k chunk.
            group = get_parallel().attn_tp_group
            k_nope_send = k_nope.contiguous()
            k_pe_send = k_pe.contiguous()
            k_nope = k_nope_send.new_empty(
                (dsa_cp_plan.num_tokens_pad, *k_nope_send.shape[1:])
            )
            k_pe = k_pe_send.new_empty(
                (dsa_cp_plan.num_tokens_pad, *k_pe_send.shape[1:])
            )
            forward_batch.npu_dsa_cp_chunk_kv_works = (
                group.all_gather_into_tensor_async(k_nope, k_nope_send),
                group.all_gather_into_tensor_async(k_pe, k_pe_send),
                k_nope_send,
                k_pe_send,
            )

        if dsa_use_prefill_cp(forward_batch):
            # support allgather+rerrange
            k_nope, k_pe = m.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )

    if _enable_dcp_decode_q_preissue and _is_dcp_mla_decode_phase(forward_batch):
        # The decode Q all-gather depends only on this QKV's output, so issue
        # it now, in prepare: the layer's indexer (and the rest of the layer)
        # runs under it instead of after it. The core's decode branch waits on
        # the handle where the synchronous gather used to run.
        #
        # Skipped while a graph is being captured: the async path's
        # work.wait() is a host synchronisation, which capture forbids, so
        # the captured graph records the synchronous gather exactly as this
        # path did before the flag existed. Eager decode steps take the
        # async path.
        from sglang.srt.model_executor.runner import get_is_capture_mode

        if not get_is_capture_mode():
            dcp_group = get_parallel().dcp_group
            combined = torch.cat(
                [q_pe.transpose(0, 1), q_nope_out.transpose(0, 1)], dim=-1
            )
            gathered = combined.new_empty(
                (dcp_group.world_size * combined.shape[0], *combined.shape[1:])
            )
            forward_batch.npu_dcp_decode_q_pre = _DcpDecodeQPre(
                work=dcp_group.all_gather_into_tensor_async(gathered, combined),
                gathered=gathered,
                send=combined,
                d_pe=q_pe.size(-1),
                d_nope=q_nope_out.size(-1),
            )
            print_info_once(
                "DCP decode Q pre-issue is ON: the decode Q all-gather is "
                "issued in prepare, under the layer's indexer, instead of "
                "after it (eager decode only; captured graphs keep the "
                "synchronous gather)"
            )

    if (
        _enable_dsa_cp_q_a2a_preissue
        and dsa_cp_plan is not None
        and not mla_preprocess_used
        and not token_sharded
    ):
        # Version A: both head redistributions depend only on this QKV's
        # (suppressed under token sharding: its q is already rows x all
        # heads, there is nothing to redistribute, and a stray all-to-all
        # whose work handle nobody waits on would only move dead bytes)
        # output, so issue them now, under the indexer, instead of in the
        # core between the indexer and the sparse operator (measured ~3.6 ms
        # per indexer layer exposed there at extend). The core waits on the
        # handles right before the operator reads the tensors.
        q_nope_pre, q_nope_work, q_nope_send = dsa_cp_redistribute_heads_async(
            q_nope_out, dsa_cp_plan
        )
        q_pe_pre, q_pe_work, q_pe_send = dsa_cp_redistribute_heads_async(
            q_pe, dsa_cp_plan
        )
        # The stash carries the raw recv buffers, the work handles, and the
        # send buffers: the communication stream reads the sends (and writes
        # the recvs) at execution time, so all of them must survive until the
        # core's wait, not just this function's frame. The recv layout is
        # materialised in the core, after the wait.
        forward_batch.npu_dsa_cp_q_pre = (
            q_nope_pre,
            q_pe_pre,
            q_nope_work,
            q_pe_work,
            q_nope_send,
            q_pe_send,
        )
        print_info_once(
            "DSA-CP Q pre-issue is ON: the Q all-to-alls run under the "
            "indexer instead of between the indexer and the sparse operator"
        )

    if _enable_dcp_sfa_gather_overlap and _dcp_extend_active(forward_batch):
        # The gather was issued before the QKV (reuse layer) or right before
        # the indexer (indexer layer, below); attach this chunk's own KV now
        # that the QKV has produced it -- the one part a pre-QKV issue cannot
        # carry. The core waits on the handle right before the sparse
        # operator reads the buffer.
        pending = getattr(forward_batch, "npu_dcp_sfa_gather_pending", None)
        if pending is not None and pending.k_nope is None:
            if pending.out_nope.dtype != k_nope.dtype:
                raise AssertionError(
                    "DCP SFA gather overlap: the gather buffer was typed from "
                    f"hidden_states ({pending.out_nope.dtype}) but this layer's "
                    f"KV is {k_nope.dtype}; the buffer and the KV must share a "
                    "dtype or the sparse operator reads the wrong type"
                )
            forward_batch.npu_dcp_sfa_gather_pending = pending._replace(
                k_nope=k_nope, k_pe=k_pe
            )

    if (
        _enable_dcp_sfa_gather_overlap
        and _dcp_extend_active(forward_batch)
        and indexer_runs
        and getattr(forward_batch, "npu_dcp_sfa_gather_pending", None) is None
    ):
        # Indexer layer, the vLLM placement: issue right before the indexer.
        # This chunk's KV already exists, so it goes in with the collective
        # instead of a second step, and the gather's window is the indexer
        # itself rather than the QKV.
        forward_batch.npu_dcp_sfa_gather_pending = _dcp_start_extend_gather_npu(
            m,
            forward_batch,
            hidden_states,
            _dcp_sfa_gather_topk_width(m, prev_topk_indices),
            k_nope=k_nope,
            k_pe=k_pe,
        )

    if indexer_runs:
        topk_indices = m.indexer(
            hidden_states,
            q_lora,
            positions,
            forward_batch,
            m.layer_id,
            input_on_attn_tp_slices,
            dynamic_scale,
        )
        # index_topk from the tensor rather than from m.indexer: this is the
        # width the operator will actually be handed, and the gate is a claim
        # about that width.
        packed_plan = (
            dcp_packed_read_plan(forward_batch, topk_indices.shape[-1])
            if topk_indices is not None
            else None
        )
        if packed_plan is not None:
            # Remapped HERE, where the indexer produces it, and not at the
            # operator. Only 21 of 78 layers run the indexer; the other 57 reuse
            # what it returned, so this runs 21 times a forward instead of 78.
            # A remapped top-k stays valid as it propagates because the packed
            # layout depends on the batch's prefix and extend lengths, not on
            # the layer -- every layer gathers into the same shape.
            topk_indices = remap_topk_to_packed(topk_indices, packed_plan)
        elif _use_dsa_dcp_partial_attention(forward_batch):
            # DSA layers that skip the indexer reuse ``prev_topk_indices``. Remap
            # only when a fresh global top-k is produced so shared-index layers do
            # not repeat the same DCP partitioning work. Mutually exclusive with
            # the packed shared-read plan above: each defines its own operator
            # contract for the top-k layout.
            parallel = runtime_context.get_parallel()
            topk_indices = dcp_layout.remap_dcp_sparse_indices(
                topk_indices,
                parallel.attn_dcp_size,
                parallel.attn_dcp_rank,
                interleave_size=forward_context.get_attn_backend().page_size,
            )
    else:
        topk_indices = prev_topk_indices

    if _use_dsa_dcp_partial_attention(forward_batch):
        q_nope_out, q_pe = dcp_comm.all_gather_q_for_mla_decode(q_nope_out, q_pe)

    topk_indices = maybe_capture_indexer_topk(m.layer_id, topk_indices)

    return (
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        topk_indices,
        forward_batch,
        zero_allocator,
        positions,
        mla_preprocess_used,
    )


# Gathered rows per extend-gather collective. A bf16 latent row is 1024 bytes,
# so the default caps a piece's scratch at 256 MiB of latent KV plus 32 MiB of
# rope key. At a ~976k prefix and dcp16 that is four pieces -- eight collectives
# per layer, since the two keys move separately, against one for the whole
# context before. The bytes moved are the same either way, and each piece is
# still a 16.8 MiB send per rank, well into the bandwidth-bound regime; the
# trade is launch overhead against the scratch held beside the output. Tunable
# because that trade is arithmetic here and measurable only on the box.
_dcp_extend_gather_piece_rows = envs.SGLANG_NPU_DCP_EXTEND_GATHER_PIECE_ROWS.get()
if _dcp_extend_gather_piece_rows <= 0:
    _dcp_extend_gather_piece_rows = 1 << 62

_last_dcp_extend_rows: Optional[Tuple[int, int]] = None


def _log_dcp_extend_memory(prefix_rows: int, extend_rows: int) -> None:
    """Log the previous DCP extend's peak device memory on this rank, then reset.

    Called on the first layer of each DCP extend forward, so the peak spans one
    whole forward -- plus whatever ran between the two extends, such as decode
    steps. It is the number a chunk size and ``--mem-fraction-static`` have to
    fit under; npu-smi shows only what the allocator has reserved, which at a
    stall is the whole die.
    """
    global _last_dcp_extend_rows
    if _last_dcp_extend_rows is not None:
        gib = 1 << 30
        logger.info(
            "DCP extend memory: prefix=%d extend=%d peak_allocated=%.2f GiB "
            "allocated=%.2f GiB reserved=%.2f GiB",
            *_last_dcp_extend_rows,
            torch.npu.max_memory_allocated() / gib,
            torch.npu.memory_allocated() / gib,
            torch.npu.memory_reserved() / gib,
        )
    torch.npu.reset_peak_memory_stats()
    _last_dcp_extend_rows = (prefix_rows, extend_rows)


def _pad_dcp_extend_send(shards: torch.Tensor, plan) -> torch.Tensor:
    """Lay this rank's per-request shards out at their padded send offsets."""
    send = shards.new_empty((plan.send_rows, *shards.shape[1:]))
    src = dst = 0
    for local_len, padded_len in zip(plan.local_lens, plan.padded_lens):
        send[dst : dst + local_len] = shards[src : src + local_len]
        src += local_len
        dst += padded_len
    return send


_enable_dcp_gather_prefetch = envs.SGLANG_NPU_ENABLE_DCP_GATHER_PREFETCH.get()

# D2v3: stream the o_proj full matrix on the DCP gather side stream instead of
# holding a static per-rank replica. The side stream assembles it right after
# the same layer's context-KV gather, one layer ahead, double buffered. OFF
# restores the D1v3 static o_proj_qrep_weight, byte for byte. It only takes
# effect under the D1v3 token-sharded rebuild, which is what makes the full
# matrix necessary in the first place.
_stream_o_proj = (
    envs.SGLANG_NPU_STREAM_O_PROJ.get()
    and envs.SGLANG_NPU_DSA_CP_TOKEN_SHARDED.get()
)

if _enable_dcp_gather_prefetch and not envs.SGLANG_NPU_ENABLE_DCP_PACKED_READ.get():
    # The prefetch fills the buffer the operator reads directly. On the
    # permuting path that buffer is produced by an index_select from a scratch,
    # which cannot be prefetched a layer ahead without also double-buffering the
    # scratch and splitting the plan's index. Refuse rather than be silently
    # inert -- the whole point of the flag is that it does something.
    raise ValueError(
        "SGLANG_NPU_ENABLE_DCP_GATHER_PREFETCH requires "
        "SGLANG_NPU_ENABLE_DCP_PACKED_READ: there is nothing to prefetch into "
        "until the operator reads the gathered buffer directly."
    )


class _DcpGatherPrefetcher:
    """The existing side-stream prefetch: gather layer l+1's prefix on a side
    stream while layer l computes.

    The all-gather is ~0.52 s of a chunk at a ~990k prefix -- measured twice
    over, once from collective volume (94.2 GB/rank at ~197 GB/s) and once as
    the marginal cost of doubling the chunk count in the AISBench warm-up
    (61 -> 121 batches for +31.13 s). It depends on nothing the current layer
    computes: the prefix KV was written by earlier forwards and sits in the
    pool. So it can run underneath the layer before it, and on a 78-layer model
    only the first layer ever has to wait for one.

    Two slots, indexed ``layer_id % 2``, exactly as upstream's page-interleave
    pool does it (``_slots[layer_id % 2]``). One is being read by this layer's
    attention while the other is being filled for the next.

    **Every dcp-group gather goes on the side stream, including the first.** Not
    an optimisation -- it keeps one HCCL communicator bound to one stream for
    the life of the process. Alternating a communicator between streams is the
    part of this design with no precedent on NPU here, so this does not do it.
    The compute stream only ever waits on events.

    **Only this chunk's own KV is written from the compute stream**, because it
    is the one part that is not available early: ``k_nope``/``k_pe`` for layer
    l+1 are computed during layer l+1. They are a contiguous tail slice, so that
    write is a copy rather than a gather, and it is ordered after the slot's
    ready event.

    Memory: two slots instead of one, ~2.08 GiB a rank at a 972k context against
    the packed read's ~1.04 GiB, so ~0.74 GiB more than the permuting path held.
    The AISBench
    run peaked at 49.90 GiB allocated against a ceiling near 51.7, so this may
    need ``--mem-fraction-static`` a notch below 0.70. That is the cost of the
    overlap and it is worth knowing before the run rather than after.
    """

    def __init__(self) -> None:
        self.stream = torch.npu.Stream()
        # Events come from stream.record_event(), the idiom this module already
        # uses for the q_b_proj side stream, rather than a reused Event object.
        self.ready: list = [None, None]
        # Which layer each slot currently holds the PREFIX for, per forward.
        self.filled: list = [None, None]
        # D2v3: the full o_proj matrix for a layer, assembled on THIS side
        # stream immediately after that layer's context-KV gather (one layer
        # ahead, so it hides under the layer's own SFA and absorb). Double
        # buffered by layer parity, like the KV slots: the gather for
        # layer L+1 must not overwrite the slot layer L-1's o_proj GEMM --
        # the previous reader -- is still consuming, and the wait_stream at
        # the top of _gather_prefix is what orders that.
        self.o_proj_slots: list = [None, None]
        self.o_proj_ready: list = [None, None]
        # Which layer each o_proj slot currently holds (None: never filled).
        # The consumer checks this, not just the event's presence: a parity
        # slot can carry a live event from layer L-2 while layer L's gather
        # was never issued (a forward the prefetch declined) -- waiting on
        # the stale event would read L-2's matrix for L, silently wrong.
        self.o_proj_filled: list = [None, None]
        self._o_proj_ranks_checked = False
        self.prev_layer: Optional[int] = None
        self._max_layer: Optional[int] = None
        # The layer whose prefix prime() issued in prepare, before any take().
        # take() must not let _note_layer reset that slot: the prime ran before
        # the first take, so the structural "layer id stopped advancing" test
        # cannot tell the primed first layer apart from a fresh forward.
        self.primed: Optional[int] = None

    def _note_layer(self, layer_id: int) -> None:
        """Drop both slots when a new forward starts.

        Detected structurally -- layers run 0, 1, 2, ... and then the next
        forward starts over, so a layer id that does not advance means a new
        batch -- rather than by the identity of the ForwardBatch. Identity is
        what a reused object breaks, and this port has already shipped one bug
        of exactly that shape (the KV write plan, fixed in 960d3a031d). A stale
        slot here would be worse than a crash: it holds a real prefix, just the
        previous batch's.

        The one exception is a primed forward: prime() runs in prepare, before
        the first take(), and fills the first layer's slot. The structural test
        above cannot see that the first take() belongs to the primed forward
        (the layer id has not advanced past the prime's), so the marker is
        consumed here and the primed slot is kept.
        """
        if self.primed is not None:
            if layer_id == self.primed:
                # The take of the very layer prime() filled: keep its slot.
                self.primed = None
                self.prev_layer = layer_id
                return
            # A take that is not the primed layer: the prime is stale (its
            # forward never consumed it) or this is a newer forward. Drop the
            # marker and fall through to the normal reset below.
            self.primed = None
        if self.prev_layer is None or layer_id <= self.prev_layer:
            self.filled = [None, None]
        self.prev_layer = layer_id

    def slots(self, plan, k_nope, k_pe, slot: int):
        return (
            dcp_extend_gather_buffer(f"packed_latent_{slot}", k_nope, plan.rows),
            dcp_extend_gather_buffer(f"packed_rope_{slot}", k_pe, plan.rows),
        )

    def _gather_prefix(self, m, md, plan, ref_nope, ref_rope, layer_id: int) -> None:
        """Issue one layer's prefix all-gather on the side stream.

        ``ref_nope``/``ref_rope`` only type the slot buffers (dtype, device,
        row shape); they are the chunk's own KV on a take() and a (1, 1, dim)
        slice of hidden_states on a prime(), before the QKV has produced them.
        """
        slot = layer_id % 2
        out_nope, out_rope = self.slots(plan, ref_nope, ref_rope, slot)
        # The slot buffers are allocated by the caching allocator on the COMPUTE
        # stream and written here on the side stream. record_stream is what stops
        # the allocator handing that memory to something else while this stream
        # is still writing it. The buffers are cached and long-lived, so this
        # matters on exactly one event -- a context longer than any before it,
        # which grows the reservation and frees the old block -- and that is the
        # first forward of a longer prompt, not a rare one.
        out_nope.record_stream(self.stream)
        out_rope.record_stream(self.stream)
        # Wait for everything already issued on compute, so the slot's previous
        # reader -- layer_id - 2's attention -- is done before it is overwritten.
        self.stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(self.stream):
            if plan.gathered_rows:
                parallel = get_parallel()
                send_nope, send_rope = get_token_to_kv_pool().get_mla_kv_buffer(
                    m.attn_mqa, md.dcp_local_prefix_kv_indices, layer_id=layer_id
                )
                send_nope = _pad_dcp_packed_send(send_nope, plan, parallel.dcp_rank)
                send_rope = _pad_dcp_packed_send(send_rope, plan, parallel.dcp_rank)
                parallel.dcp_group.all_gather_into_tensor(
                    out_nope[: plan.gathered_rows], send_nope
                )
                parallel.dcp_group.all_gather_into_tensor(
                    out_rope[: plan.gathered_rows], send_rope
                )
            # The KV-only ready event is recorded here, before the o_proj
            # gather, so the SFA read does not wait for the weight assembly.
            self.ready[slot] = self.stream.record_event()
            # D2v3: the full o_proj matrix for this layer, on the same side
            # stream and immediately after the KV gather -- the ~2 ms
            # collective then hides under this layer's SFA and absorb instead
            # of 14.1 GiB of it standing on every device.
            if _stream_o_proj:
                o_shard = get_o_proj_stream_shard(layer_id)
                if o_shard is not None:
                    self._issue_o_proj_gather(layer_id, o_shard)
        self.filled[slot] = layer_id

    def _issue_o_proj_gather(self, layer_id: int, o_shard: torch.Tensor) -> None:
        """All-gather the full o_proj matrix for ``layer_id`` on the side
        stream (see SGLANG_NPU_STREAM_O_PROJ).

        The shards are RowParallel over the attention-TP group, but the
        collect runs on the DCP group -- the same communicator as the KV
        gather this rides on, one communicator per stream. That is correct
        only while both groups are the same ranks in the same order (true at
        TP==DCP, the supported configuration), which is checked once, loudly.
        """
        parallel = get_parallel()
        if not self._o_proj_ranks_checked:
            if list(parallel.dcp_group.ranks) != list(
                parallel.attn_tp_group.ranks
            ):
                raise RuntimeError(
                    "SGLANG_NPU_STREAM_O_PROJ assembles the attention-TP "
                    "o_proj shards on the DCP group, which requires both "
                    "groups to hold the same ranks in the same order; got "
                    f"dcp={list(parallel.dcp_group.ranks)} vs "
                    f"attn_tp={list(parallel.attn_tp_group.ranks)}"
                )
            self._o_proj_ranks_checked = True
            print_info_once(
                "D2v3 o_proj streaming ON: the full o_proj matrix is "
                "all-gathered on the DCP gather side stream, right after the "
                "same layer's context-KV gather, instead of standing "
                "replicated on every device"
            )
        slot = layer_id % 2
        full_shape = (
            o_shard.shape[0],
            o_shard.shape[1] * parallel.attn_tp_size,
        )
        slot_buf = self.o_proj_slots[slot]
        if (
            slot_buf is None
            or slot_buf.shape != full_shape
            or slot_buf.dtype != o_shard.dtype
            or slot_buf.device != o_shard.device
        ):
            slot_buf = o_shard.new_empty(full_shape)
            self.o_proj_slots[slot] = slot_buf
        # Same allocator bookkeeping as the KV slots above: written on the
        # side stream, read on the compute stream, cached for good.
        slot_buf.record_stream(self.stream)
        parallel.dcp_group.all_gather_into_tensor(slot_buf, o_shard)
        # A separate event: the o_proj GEMM waits on this one, and it is
        # ordered after the KV ready event by the side stream itself.
        self.o_proj_ready[slot] = self.stream.record_event()
        self.o_proj_filled[slot] = layer_id

    def take_o_proj(self, layer_id: int) -> torch.Tensor:
        """This layer's full o_proj matrix: wait for its side-stream gather
        and take the double-buffered slot. The wait is the last moment before
        the GEMM reads the buffer -- the same placement as the KV ready wait
        in take()."""
        slot = layer_id % 2
        if self.o_proj_ready[slot] is None:
            raise RuntimeError(
                f"o_proj streaming: layer {layer_id}'s full matrix was never "
                "gathered; the side stream must issue it one layer ahead "
                "(prime/take), the same as the KV gather"
            )
        torch.npu.current_stream().wait_event(self.o_proj_ready[slot])
        return self.o_proj_slots[slot]

    def prime(self, m, md, plan, hidden_states, layer_id: int) -> None:
        """Issue the first layer's prefix gather at the start of the forward.

        take() issues a layer's gather from the core, AFTER that layer's
        indexer has run; with no previous layer to prefetch from, layer 0's
        gather then starts exactly where the 154 ms stall was measured -- in
        the indexer's window, idle. prime() runs in prepare, BEFORE the QKV,
        so the first gather moves bytes under the QKV and the indexer instead
        of starting with them: it is done (or nearly so) by the time the core
        needs it.

        Only the first DSA layer of a forward primes; the rest no-op here and
        are prefetched by the previous layer's take() as usual. The slot it
        fills is the same one take() would fill, typed from hidden_states
        (the QKV output does not exist yet) rather than from the chunk's own
        KV, so the two share one grow-only buffer.
        """
        if self.prev_layer is not None and layer_id > self.prev_layer:
            # A layer earlier in THIS forward already ran, so this is not the
            # first layer: nothing to prime, take() will handle it.
            return
        # First DSA layer of a (possibly new) forward: drop the previous
        # forward's slots, then fill this layer's.
        self.filled = [None, None]
        self.prev_layer = layer_id
        ref_nope = hidden_states.new_empty((1, 1, m.kv_lora_rank))
        ref_rope = hidden_states.new_empty((1, 1, m.qk_rope_head_dim))
        self._gather_prefix(m, md, plan, ref_nope, ref_rope, layer_id)
        self.primed = layer_id
        print_info_once(
            "DCP gather first-layer prime is ON: layer "
            f"{layer_id}'s prefix all-gather is issued in prepare, before the "
            "QKV, so it runs under the QKV and the indexer instead of starting "
            "with them"
        )

    def _max_servable_layer(self) -> int:
        """Highest layer id the KV pool can serve, from the list it indexes.

        NOT from ``pool.end_layer``. That field is inconsistent: the pool's own
        constructor defaults it to ``layer_num - 1`` (inclusive) while
        model_runner passes ``layer_info.end_layer``, which is EXCLUSIVE -- 78
        on a 78-layer model. Stage B asked to prefetch layer 78 and got
        ``IndexError: index 78 is out of bounds for dimension 0 with size 78``.

        ``get_key_buffer`` reads ``k_buffer[layer_id - start_layer]``, so that
        buffer's length is the only statement of the bound that cannot disagree
        with the read it guards. ``len`` covers both shapes it takes here, a
        list of per-layer tensors and one stacked tensor whose first dimension
        is the layer.
        """
        if self._max_layer is None:
            pool = get_token_to_kv_pool()
            k = getattr(pool, "k_buffer", None)
            start = getattr(pool, "start_layer", 0)
            # -1 disables prefetching entirely, which costs the overlap and
            # nothing else: every layer then gathers its own prefix inline.
            self._max_layer = -1 if k is None else start + len(k) - 1
        return self._max_layer

    def take(self, m, md, plan, k_nope, k_pe, layer_id: int):
        """This layer's KV buffer, with next layer's gather already in flight."""
        self._note_layer(layer_id)
        slot = layer_id % 2
        if self.filled[slot] != layer_id:
            # Layer 0 of a forward, or a layer whose prefetch was skipped. Issue
            # it now and pay for it, exactly as the unprefetched path would.
            self._gather_prefix(m, md, plan, k_nope, k_pe, layer_id)
        else:
            # Logged on a HIT, not on construction. A prefetcher that is built
            # and then misses on every layer does no overlapping at all, and
            # would log identically if this sat in __init__ -- which is the
            # silently-inert failure the flag exists to rule out. Reaching here
            # means a previous layer's side-stream gather filled this slot.
            print_info_once(
                "DCP gather prefetch is ON: layer l+1's prefix all-gather runs "
                "on a side stream underneath layer l's compute"
            )
        out_nope, out_rope = self.slots(plan, k_nope, k_pe, slot)
        torch.npu.current_stream().wait_event(self.ready[slot])
        # The one part that could not be prefetched: this chunk's own KV.
        _write_packed_extend_tail(out_nope, out_rope, k_nope, k_pe, plan)
        if layer_id < self._max_servable_layer():
            self._gather_prefix(m, md, plan, k_nope, k_pe, layer_id + 1)
        return out_nope, out_rope


_dcp_gather_prefetcher: Optional[_DcpGatherPrefetcher] = None


def _dcp_gather_prefetcher_get() -> _DcpGatherPrefetcher:
    """The process-wide prefetcher, created on first use.

    Two call sites need it -- prepare (to prime the first layer's gather
    before the QKV) and the core's packed path (to take a layer's buffer) --
    and either can run first, so the creation lives here rather than in one
    of them.
    """
    global _dcp_gather_prefetcher
    if _dcp_gather_prefetcher is None:
        _dcp_gather_prefetcher = _DcpGatherPrefetcher()
    return _dcp_gather_prefetcher


_eager_o_proj_slots: list = [None, None]

# One line per process for the fallback below. print_info_once dedupes on the
# message TEXT, and the message names the layer -- without this guard every
# layer of a rejected forward would log its own "once" line (78 per rank).
_dcp_o_proj_fallback_logged = False


def _dcp_o_proj_weight(m: "DeepseekV2AttentionMLA") -> torch.Tensor:
    """This layer's full o_proj matrix for the token-sharded o_proj GEMM
    (D2v3).

    Prefetch path: the matrix was all-gathered one layer ahead, right after
    this layer's KV gather -- wait on its ready event and take the
    double-buffered slot. The filled-layer check, not just the event's
    presence: a parity slot can carry a live event from layer L-2 while
    layer L's gather was never issued (a forward the prefetch declined),
    and waiting on the stale event would read L-2's matrix for L, silently
    wrong. Fallback (the prefetch off, DCP disabled, or a forward whose
    packed plan declined the prefetch): gather synchronously on the compute
    stream over the attention-TP group -- the shards' own group, so no
    rank-order assumption across groups -- into the parity slot. Correct,
    and exposed on the critical path, which is why the prefetch path
    exists; the slot is safe across layers because the collect is enqueued
    after the previous consumer's GEMM in stream order."""
    global _dcp_o_proj_fallback_logged
    parallel = get_parallel()
    if _enable_dcp_gather_prefetch and parallel.dcp_enabled:
        prefetcher = _dcp_gather_prefetcher_get()
        if prefetcher.o_proj_filled[m.layer_id % 2] == m.layer_id:
            return prefetcher.take_o_proj(m.layer_id)
        if not _dcp_o_proj_fallback_logged:
            _dcp_o_proj_fallback_logged = True
            print_info_once(
                "D2v3 o_proj: the side stream did not gather layer "
                f"{m.layer_id}'s matrix this forward (DCP disabled, or the "
                "packed plan declined the prefetch -- see the 'DCP packed "
                "read is off ...' line for the reason); assembling it "
                "synchronously on the compute stream instead -- correct, "
                "but exposed on the critical path. Further occurrences of "
                "the fallback are not logged."
            )
    shard = get_o_proj_stream_shard(m.layer_id)
    if shard is None:
        raise RuntimeError(
            f"o_proj streaming: layer {m.layer_id}'s shard was never "
            "registered; the model-runner token-sharded preparation must "
            "have run before any forward"
        )
    slot = m.layer_id % 2
    full_shape = (
        shard.shape[0],
        shard.shape[1] * parallel.attn_tp_size,
    )
    buf = _eager_o_proj_slots[slot]
    if (
        buf is None
        or buf.shape != full_shape
        or buf.dtype != shard.dtype
        or buf.device != shard.device
    ):
        buf = shard.new_empty(full_shape)
        _eager_o_proj_slots[slot] = buf
    parallel.attn_tp_group.all_gather_into_tensor(buf, shard)
    return buf


_enable_dcp_sfa_gather_overlap = envs.SGLANG_NPU_ENABLE_DCP_SFA_GATHER_OVERLAP.get()
_enable_dsa_cp_local_topk = envs.SGLANG_NPU_DSA_CP_LOCAL_TOPK.get()
_enable_dsa_cp_q_a2a_preissue = envs.SGLANG_NPU_DSA_CP_Q_A2A_PREISSUE.get()
_enable_dsa_cp_q_full_width = envs.SGLANG_NPU_DSA_CP_Q_FULL_WIDTH.get()
_enable_dcp_decode_q_preissue = envs.SGLANG_NPU_DCP_DECODE_Q_PREISSUE.get()

if _enable_dsa_cp_q_a2a_preissue and _enable_dsa_cp_q_full_width:
    # Both are versions of the same removal (the per-layer Q all-to-all).
    # Running either alone is the A/B this pair exists for; running both is
    # a configuration error, not a fallback order.
    raise ValueError(
        "SGLANG_NPU_DSA_CP_Q_A2A_PREISSUE and SGLANG_NPU_DSA_CP_Q_FULL_WIDTH "
        "are mutually exclusive: both replace the DSA-CP Q all-to-all, and the "
        "point of having both is to profile one against the other. Enable "
        "exactly one (or neither)."
    )


class _DcpDecodeQPre(NamedTuple):
    """The DCP decode Q all-gather, in flight: issued in prepare (right after
    the QKV, so the layer's indexer runs under it), finished in the core's
    decode branch where the synchronous gather used to run.

    ``gathered`` is the all-gather's output, ``[H * dcp_size, B, d_pe +
    d_nope]`` -- the same rank-major layout ``all_gather_q_for_mla_decode``
    builds, so the split below it is the same split. ``send`` is the
    all-gather's input: the communication stream reads it at execution time,
    so it must survive until the core's wait, not just prepare's frame.
    """

    work: Optional[object]  # None when dcp_size is 1 (a copy, nothing to wait on)
    gathered: torch.Tensor
    send: torch.Tensor
    d_pe: int
    d_nope: int


def _dcp_extend_active(forward_batch: "ForwardBatch") -> bool:
    """Whether this forward's DCP extend gather runs, per layer.

    prepare and core must agree on exactly this, so the expression lives in one
    place: prepare decides whether to issue the async gather, core whether to
    wait for it. It is the former inline block in forward_dsa_core_npu, moved.
    """
    return (
        get_parallel().dcp_enabled
        and forward_batch.forward_mode.is_extend()
        and not _is_dcp_mla_decode_phase(forward_batch)
        and forward_batch.attn_dcp_metadata is not None
    )


def _dcp_sfa_gather_topk_width(
    m: "DeepseekV2AttentionMLA", prev_topk_indices: Optional[torch.Tensor]
) -> Optional[int]:
    """This layer's eventual top-k width, known before the indexer runs.

    The gather planner takes the width the sparse operator will be handed
    (``topk_indices.shape[-1]`` in the core), which is the indexer's
    ``index_topk`` on the layers that run it and the previous layer's on the
    57 that reuse it. The path decision (permuting vs packed) has to be made
    in prepare, before the indexer has produced the tensor, so derive the same
    width from the same condition the indexer call is gated on.
    """
    if not m.skip_topk or (m.is_nextn and prev_topk_indices is None):
        return m.indexer.index_topk if m.indexer is not None else None
    return prev_topk_indices.shape[-1] if prev_topk_indices is not None else None


class _DcpSfaGatherPending(NamedTuple):
    """The DCP extend gather, in flight: issued in prepare, finished in core.

    ``works`` are the async all-gather handles (None entries are world-1 no-ops
    and stay skipped, as the collectives they stand in for are). The scratch
    lists are permuting-path only: one gather scratch per piece, because with
    every collective in flight before the elementwise pass, piece i's scratch
    can no longer be recycled into piece i+1 while piece i is still waiting.

    ``k_nope``/``k_pe`` are this chunk's own KV, the one part the collective
    cannot carry: None when the gather was issued before the QKV (a reuse
    layer) and attached by prepare once the QKV has produced them, or the real
    tensors straight away when the gather is issued after the QKV (an indexer
    layer).
    """

    kind: str  # "permuting" or "packed"
    works: list
    plan: object  # DcpExtendGatherPlan or DcpPackedReadPlan
    out_nope: torch.Tensor
    out_rope: torch.Tensor
    k_nope: Optional[torch.Tensor] = None
    k_pe: Optional[torch.Tensor] = None
    scratch_nope: Optional[list] = None
    scratch_rope: Optional[list] = None


def _dcp_start_extend_gather_permuting_npu(
    m: "DeepseekV2AttentionMLA",
    forward_batch: "ForwardBatch",
    hidden_states: torch.Tensor,
    k_nope: Optional[torch.Tensor] = None,
    k_pe: Optional[torch.Tensor] = None,
) -> _DcpSfaGatherPending:
    """Issue the permuting-path gather now (before the QKV on a reuse layer,
    before the indexer on an indexer layer); its elementwise tail runs later.

    Everything the sync path did before and between its collectives happens
    here: the once-per-forward plan (and its write-plan and buffer-release
    duties), the send tensors, the reserved output and scratch. What moves to
    ``_dcp_finish_extend_gather_npu`` is the per-piece copy-plus-index_select,
    which is what waits on the gathered bytes. The buffers are typed from
    this chunk's KV when it already exists and from ``hidden_states``
    otherwise -- that is the point: the gather is in flight under the QKV or
    the indexer, not after it.
    """
    parallel = get_parallel()
    md = forward_batch.attn_dcp_metadata
    plan = getattr(forward_batch, "npu_dcp_extend_gather", None)
    if plan is None:
        # Sized by the shared planner for the whole context, for CUDA's
        # kernels. Nothing on this path reads it, and held it is a
        # context-sized tensor alive through every layer.
        md.dcp_kv_buffer = None
        plan = plan_dcp_extend_gather(
            forward_batch.extend_prefix_lens_cpu,
            forward_batch.extend_seq_lens_cpu,
            parallel.dcp_size,
            parallel.dcp_rank,
            _dcp_extend_gather_piece_rows,
        )
        plan = plan._replace(
            pieces=[
                piece._replace(index=piece.index.to(hidden_states.device))
                for piece in plan.pieces
            ]
        )
        forward_batch.npu_dcp_extend_gather = plan
        # Same branch, same once-per-forward guarantee: tell the pool it may
        # drop the rows this rank does not own from this forward's KV write,
        # instead of writing all of them and aiming 15/16 at a padding row.
        # getattr because a wrapper pool (SWA, hybrid) may forward
        # get_mla_kv_buffer without knowing about this; declining is correct
        # there, and silent because it is the pre-existing behaviour.
        plan_write = getattr(get_token_to_kv_pool(), "plan_dcp_extend_write", None)
        if plan_write is not None:
            plan_write(forward_batch.out_cache_loc)
        if _debug_dcp_extend_memory:
            _log_dcp_extend_memory(
                sum(forward_batch.extend_prefix_lens_cpu),
                sum(forward_batch.extend_seq_lens_cpu),
            )

    # Typed from this chunk's KV when it already exists (the indexer-layer
    # placement) and from hidden_states otherwise (the pre-QKV placement,
    # where the QKV has not run yet). Both share the model dtype, and prepare
    # asserts the agreement once k_nope exists, before the core reads the
    # buffers.
    ref_nope = (k_nope if k_nope is not None else hidden_states).new_empty(
        (1, 1, m.kv_lora_rank)
    )
    ref_rope = (k_pe if k_pe is not None else hidden_states).new_empty(
        (1, 1, m.qk_rope_head_dim)
    )
    total_rows = plan.pieces[-1].out_end if plan.pieces else 0
    out_nope = dcp_extend_gather_buffer("latent", ref_nope, total_rows)
    out_rope = dcp_extend_gather_buffer("rope", ref_rope, total_rows)

    # One gather scratch per piece (see _DcpSfaGatherPending): the sync path
    # recycles one widest-piece scratch because its elementwise consumes each
    # piece before the next collective; here every collective is already in
    # flight, so the pieces must not share a scratch. Summed over the pieces
    # the rows equal the output's, which is the flag's documented memory cost.
    scratch_nope: list = []
    scratch_rope: list = []
    for i, piece in enumerate(plan.pieces):
        rows = (
            (piece.send_end - piece.send_start) * parallel.dcp_size
            + piece.extend_end
            - piece.extend_start
        )
        scratch_nope.append(
            dcp_extend_gather_buffer(f"latent_scratch_{i}", ref_nope, rows)
        )
        scratch_rope.append(
            dcp_extend_gather_buffer(f"rope_scratch_{i}", ref_rope, rows)
        )

    works: list = []
    if plan.send_rows:
        send_nope, send_rope = get_token_to_kv_pool().get_mla_kv_buffer(
            m.attn_mqa,
            md.dcp_local_prefix_kv_indices,
        )
        if plan.local_lens != plan.padded_lens:
            # Served prefixes are dcp_size-aligned (the widened allocator page)
            # and need no padding; this is the general case.
            send_nope = _pad_dcp_extend_send(send_nope, plan)
            send_rope = _pad_dcp_extend_send(send_rope, plan)
        # Every rank plans the same pieces, so every rank issues -- or, for a
        # piece with no prefix rows, skips -- the same collectives in the same
        # order, exactly as the sync loop does.
        for i, piece in enumerate(plan.pieces):
            gathered = (piece.send_end - piece.send_start) * parallel.dcp_size
            if gathered:
                works.append(
                    parallel.dcp_group.all_gather_into_tensor_async(
                        scratch_nope[i][:gathered],
                        send_nope[piece.send_start : piece.send_end],
                    )
                )
                works.append(
                    parallel.dcp_group.all_gather_into_tensor_async(
                        scratch_rope[i][:gathered],
                        send_rope[piece.send_start : piece.send_end],
                    )
                )
    return _DcpSfaGatherPending(
        "permuting", works, plan, out_nope, out_rope, k_nope, k_pe,
        scratch_nope, scratch_rope,
    )


def _dcp_start_extend_gather_packed_npu(
    m: "DeepseekV2AttentionMLA",
    forward_batch: "ForwardBatch",
    hidden_states: torch.Tensor,
    plan: "DcpPackedReadPlan",
    k_nope: Optional[torch.Tensor] = None,
    k_pe: Optional[torch.Tensor] = None,
) -> _DcpSfaGatherPending:
    """Issue the packed-path gather now (before the QKV on a reuse layer,
    before the indexer on an indexer layer); the extend tail is written later.

    The packed path needs no scratch at all: the collective lands directly in
    the buffer the operator reads, so the only thing that moves to the finish
    is the local copy of this chunk's own KV after the gathered prefix. The
    buffer is typed from this chunk's KV when it already exists and from
    ``hidden_states`` otherwise (the QKV output does not exist yet).
    """
    parallel = get_parallel()
    md = forward_batch.attn_dcp_metadata
    if not getattr(forward_batch, "npu_dcp_packed_ready", False):
        # Once per forward, on the first layer, and it carries the same
        # once-per-forward duty the permuting path does in its plan-building
        # branch.
        forward_batch.npu_dcp_packed_ready = True
        md.dcp_kv_buffer = None
        plan_write = getattr(get_token_to_kv_pool(), "plan_dcp_extend_write", None)
        if plan_write is not None:
            plan_write(forward_batch.out_cache_loc)
        if _debug_dcp_extend_memory:
            _log_dcp_extend_memory(plan.prefix_len, plan.extend_len)

    ref_nope = (k_nope if k_nope is not None else hidden_states).new_empty(
        (1, 1, m.kv_lora_rank)
    )
    ref_rope = (k_pe if k_pe is not None else hidden_states).new_empty(
        (1, 1, m.qk_rope_head_dim)
    )
    out_nope = dcp_extend_gather_buffer("packed_latent", ref_nope, plan.rows)
    out_rope = dcp_extend_gather_buffer("packed_rope", ref_rope, plan.rows)

    works: list = []
    if plan.gathered_rows:
        send_nope, send_rope = get_token_to_kv_pool().get_mla_kv_buffer(
            m.attn_mqa, md.dcp_local_prefix_kv_indices
        )
        send_nope = _pad_dcp_packed_send(send_nope, plan, parallel.dcp_rank)
        send_rope = _pad_dcp_packed_send(send_rope, plan, parallel.dcp_rank)
        works.append(
            parallel.dcp_group.all_gather_into_tensor_async(
                out_nope[: plan.gathered_rows], send_nope
            )
        )
        works.append(
            parallel.dcp_group.all_gather_into_tensor_async(
                out_rope[: plan.gathered_rows], send_rope
            )
        )
    return _DcpSfaGatherPending(
        "packed", works, plan, out_nope, out_rope, k_nope, k_pe
    )


def _dcp_start_extend_gather_npu(
    m: "DeepseekV2AttentionMLA",
    forward_batch: "ForwardBatch",
    hidden_states: torch.Tensor,
    index_topk: Optional[int],
    k_nope: Optional[torch.Tensor] = None,
    k_pe: Optional[torch.Tensor] = None,
) -> Optional[_DcpSfaGatherPending]:
    """Start this layer's SFA KV gather, in flight under the QKV (reuse
    layer) or under the indexer (indexer layer, the vLLM placement).

    Returns None when the layer should take the sync path instead: the
    existing side-stream prefetch (SGLANG_NPU_ENABLE_DCP_GATHER_PREFETCH)
    owns the packed path when it is on (it starts the same gather a layer
    earlier, and a second gather of the same layer into the same buffer
    would be wrong).

    The reserved buffers are typed from ``k_nope``/``k_pe`` when this chunk's
    KV already exists (the indexer-layer placement) and from
    ``hidden_states`` otherwise (the pre-QKV placement, where the QKV output
    that the sync path would type them with has not been computed yet). The
    pending therefore carries k_nope/k_pe = None until the QKV has run and
    prepare attaches the real tensors, or the real tensors straight away.
    """
    print_info_once(
        "DCP SFA gather overlap ACTIVE (async in-layer): dcp_size="
        f"{get_parallel().dcp_size}"
    )
    packed_plan = dcp_packed_read_plan(forward_batch, index_topk)
    if packed_plan is not None:
        if _enable_dcp_gather_prefetch:
            print_info_once(
                "DCP SFA gather overlap defers to the side-stream prefetch "
                "on the packed path (SGLANG_NPU_ENABLE_DCP_GATHER_PREFETCH), "
                "which starts the same gather a layer earlier"
            )
            return None
        return _dcp_start_extend_gather_packed_npu(
            m, forward_batch, hidden_states, packed_plan, k_nope, k_pe
        )
    return _dcp_start_extend_gather_permuting_npu(
        m, forward_batch, hidden_states, k_nope, k_pe
    )


def _dcp_finish_extend_gather_npu(
    pending: _DcpSfaGatherPending,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Wait for the in-flight gather and run the elementwise tail (the vLLM
    wait point, right before the sparse operator reads the buffer)."""
    for work in pending.works:
        if work is not None:
            work.wait()
    if pending.k_nope is None or pending.k_pe is None:
        raise AssertionError(
            "DCP SFA gather overlap: the gather was finished before prepare "
            "attached this chunk's KV; the QKV output must be attached in "
            "prepare, after the QKV, before the core waits"
        )
    if pending.kind == "packed":
        _write_packed_extend_tail(
            pending.out_nope,
            pending.out_rope,
            pending.k_nope,
            pending.k_pe,
            pending.plan,
        )
        return pending.out_nope, pending.out_rope

    parallel = get_parallel()
    plan = pending.plan
    for i, piece in enumerate(plan.pieces):
        gathered = (piece.send_end - piece.send_start) * parallel.dcp_size
        for out, scratch, own in (
            (pending.out_nope, pending.scratch_nope[i], pending.k_nope),
            (pending.out_rope, pending.scratch_rope[i], pending.k_pe),
        ):
            scratch = scratch[
                : gathered + piece.extend_end - piece.extend_start
            ]
            # scratch[:gathered] holds the all-gather's rank-major output,
            # written by the collective while the indexer ran.
            scratch[gathered:] = own[piece.extend_start : piece.extend_end]
            torch.index_select(
                scratch, 0, piece.index, out=out[piece.out_start : piece.out_end]
            )
    return pending.out_nope, pending.out_rope


def _dcp_gather_extend_kv_packed_npu(
    m: "DeepseekV2AttentionMLA",
    forward_batch: "ForwardBatch",
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    plan: "DcpPackedReadPlan",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Packed read: leave the gathered prefix in the order the collective
    produced it.

    The permuting path below writes the all-gather into a scratch and then
    ``index_select``s it into a position-ordered output. This does neither: the
    collective writes straight into the buffer the operator will read, and this
    chunk's own KV is appended after it. What pays for that is
    ``remap_topk_to_packed``, arithmetic on the ``[tokens, k]`` index tensor,
    which is a few million integers against a gigabyte of KV.

    Two things go away, not one. The ``index_select`` -- 784 launches x 0.275 ms
    = **216 ms a forward** -- and the separate output buffer, so the standing
    reservation drops from ~1.34 GiB (output + piece scratch) to ~1.04 GiB even
    though this gathers the whole prefix in one collective rather than four.

    **No pieces here, deliberately.** Pieces exist to cap the scratch held
    beside the output, and there is no scratch now -- the buffer IS the output.
    Piecing would also break the remap: with several pieces the buffer is
    piece-major then rank-major, so a position's row would depend on which piece
    its local row fell in. One collective, one layout, one formula.
    """
    parallel = get_parallel()
    md = forward_batch.attn_dcp_metadata
    if not getattr(forward_batch, "npu_dcp_packed_ready", False):
        # Once per forward, on the first layer, and it carries the same
        # once-per-forward duty the permuting path does in its plan-building
        # branch. Forgetting the write plan here would not fail: the pool's
        # identity check would simply miss and every rank would fall back to
        # writing all 16/16 of the KV at row 0 -- correct, and 487 ms slower.
        forward_batch.npu_dcp_packed_ready = True
        md.dcp_kv_buffer = None
        plan_write = getattr(get_token_to_kv_pool(), "plan_dcp_extend_write", None)
        if plan_write is not None:
            plan_write(forward_batch.out_cache_loc)
        if _debug_dcp_extend_memory:
            _log_dcp_extend_memory(plan.prefix_len, plan.extend_len)

    if _enable_dcp_gather_prefetch:
        return _dcp_gather_prefetcher_get().take(m, md, plan, k_nope, k_pe, m.layer_id)

    out_nope = dcp_extend_gather_buffer("packed_latent", k_nope, plan.rows)
    out_rope = dcp_extend_gather_buffer("packed_rope", k_pe, plan.rows)

    if plan.gathered_rows:
        send_nope, send_rope = get_token_to_kv_pool().get_mla_kv_buffer(
            m.attn_mqa, md.dcp_local_prefix_kv_indices
        )
        send_nope = _pad_dcp_packed_send(send_nope, plan, parallel.dcp_rank)
        send_rope = _pad_dcp_packed_send(send_rope, plan, parallel.dcp_rank)
        parallel.dcp_group.all_gather_into_tensor(
            out_nope[: plan.gathered_rows], send_nope
        )
        parallel.dcp_group.all_gather_into_tensor(
            out_rope[: plan.gathered_rows], send_rope
        )
    _write_packed_extend_tail(out_nope, out_rope, k_nope, k_pe, plan)
    return out_nope, out_rope


def _write_packed_extend_tail(
    out_nope: torch.Tensor,
    out_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    plan: "DcpPackedReadPlan",
) -> None:
    """This chunk's own KV, after the gathered prefix. Not gathered: identical
    on every rank.

    ``k_nope`` can be WIDER than the request's token count. Every batch is
    padded to a multiple of ``attn_tp_size`` before the model runs
    (``ceil_align``, forward_batch_info.py:1454), so the server's own 6-token
    warm-up arrives as 16 rows. The plan is built from ``extend_seq_lens_cpu``,
    which is the true count, so the two disagree by exactly the padding.

    Drop it, which is what the permuting path already does -- ``scratch[
    gathered:] = own[piece.extend_start : piece.extend_end]``. Safe because the
    operator is given true query lengths as well (``actual_seq_qlen`` comes from
    ``extend_seq_lens``), so it never reads a padded query row, and a real
    query's top-k only names positions below ``prefix_len + extend_len``.
    """
    n = plan.extend_len
    assert k_nope.shape[0] >= n, (
        f"DCP packed read: this chunk's KV has {k_nope.shape[0]} rows but the "
        f"plan expects at least {n}. The plan comes from extend_seq_lens_cpu; "
        "if the batch is now NARROWER than that, the two no longer describe "
        "the same forward."
    )
    out_nope[plan.gathered_rows :] = k_nope[:n]
    out_rope[plan.gathered_rows :] = k_pe[:n]


def _pad_dcp_packed_send(
    shards: torch.Tensor, plan: "DcpPackedReadPlan", dcp_rank: int
) -> torch.Tensor:
    """This rank's send, padded to ``send_rows`` so every rank sends alike.

    A served prefix is a whole number of ``page_size * dcp_size`` allocator
    pages, so it divides by ``dcp_size`` and no padding is needed at all -- the
    shards go to the collective untouched, which is why this checks before it
    copies rather than always allocating. Only ranks below ``prefix_len %
    dcp_size`` hold the extra row when a prefix is ragged.
    """
    local_len = plan.prefix_len // plan.dcp_size + int(
        dcp_rank < plan.prefix_len % plan.dcp_size
    )
    # Same disagreement as the extend tail, on the other side of the buffer: the
    # plan's arithmetic and the pool's actual index must describe the same rows,
    # and a mismatch would otherwise surface as an HCCL size error inside the
    # collective, where it says nothing about which of the two is wrong.
    assert shards.shape[0] == local_len, (
        f"DCP packed read: rank {dcp_rank} holds {shards.shape[0]} prefix rows "
        f"but the plan computes {local_len} from prefix_len={plan.prefix_len} "
        f"over {plan.dcp_size} ranks"
    )
    if local_len == plan.send_rows:
        return shards.contiguous()
    send = shards.new_zeros((plan.send_rows, *shards.shape[1:]))
    send[:local_len] = shards[:local_len]
    return send


def _dcp_gather_extend_kv_npu(
    m: "DeepseekV2AttentionMLA",
    forward_batch: "ForwardBatch",
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    index_topk: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialise each request's FULL prefix+extend KV, for one layer.

    Under DCP a rank holds only 1/c of the context, so at extend it cannot
    attend over the pool: the sparse operator would need every position and this
    rank has a sixteenth of them. SGLang's answer for extend is to gather the KV
    rather than shard the attention and merge -- prefill has many query tokens
    and few heads (no query all-gather happens here, so the head count is
    num_local_heads, not the decode path's num_local_heads * dcp_size), which
    makes moving the KV the cheap direction and moving the query the expensive
    one. Decode is the mirror image and gathers the query instead.

    The layout is the reason this is written here rather than calling
    ``all_gather_kv_cache_for_mla_extend``. That helper fills ``dcp_kv_buffer``
    the way CUDA's kernels read it -- every request's prefix in one region, then
    every request's extend tokens in another -- so a single request's KV is two
    disjoint runs. Measured on the box (p6_prefill_nonpaged_sfa_probe.py),
    ``npu_sparse_flash_attention`` under a non-paged layout wants
    ``sparse_indices`` **relative to each request's KV start** and
    ``actual_seq_lengths_kv`` **cumulative**, which together require one
    contiguous run per request.

    **The context-sized footprint is one output plus one piece, reserved once.**
    The prefix is gathered in pieces of at most
    ``SGLANG_NPU_DCP_EXTEND_GATHER_PIECE_ROWS`` rows, and each piece -- the gathered rows, rank-major as
    ``all_gather_into_tensor`` writes them, then this chunk's own KV for the
    requests that end in it -- is written straight into its place in the output
    with one ``index_select`` (``plan_dcp_extend_gather``) before the next is
    gathered. The output is returned as two contiguous tensors, the latent and
    the rope key. Both, and the piece scratch, come from
    ``dcp_extend_gather_buffer`` and are reused by every layer and every
    forward, so the high-water mark is about 1.3x the context at a ~976k prefix
    and is reached once, on the first extend, instead of being re-reached on
    each of 78 layers.

    Both halves were learned at ``--mem-fraction-static 0.76`` with decode
    capture, where the dies have little headroom and a shortfall shows up as a
    stall rather than an OOM: every collective waits on whichever rank is
    freeing memory. The shared ``all_gather_kv_cache_for_dcp`` made three
    transient copies of the context per layer on top of the planner's
    ``dcp_kv_buffer``, and a ~976k prefix-cache hit's first chunk stalled for
    minutes. A first rewrite (4c7684f874) cut the transient to one copy but kept
    that scratch, and ``dcp_kv_buffer``, alive for the whole forward -- and the
    stall came *earlier*, in the cold warm-up from a ~590k prefix. So the
    binding peak is not the gather itself but whatever else runs while
    context-sized tensors are held, which is why the planner's buffer is still
    released on the first layer: nothing on this path reads it, so holding it is
    pure cost.

    **A permanent reservation is not the thing that stalled.** What stalled was
    a context-sized tensor being *allocated* while held: each forward asked the
    allocator for it again, so the request landed on a differently-fragmented
    pool each time and the ranks freed and refilled out of step, which is what
    makes a collective wait on the slowest rank for minutes. The buffers here
    are allocated once and then only sliced, so after the first extend no rank
    asks the allocator for anything context-sized again. The footprint is the
    same size; it just stops moving. ``SGLANG_DEBUG_NPU_DCP_EXTEND_MEMORY``
    prints the peak per forward, which is how to check that claim rather than
    trust it.

    The two keys are gathered and written separately, not as one 576-wide row,
    because the operator takes them as separate tensors: slices of a wide buffer
    are not contiguous, and non-contiguous inputs can cost a full-context copy
    inside the operator.

    Only the prefix is gathered. This chunk's own KV is computed identically on
    every rank and arrives as ``k_nope``/``k_pe``, so it is copied in locally.
    """
    parallel = get_parallel()
    md = forward_batch.attn_dcp_metadata
    packed_plan = dcp_packed_read_plan(forward_batch, index_topk)
    if packed_plan is not None:
        return _dcp_gather_extend_kv_packed_npu(
            m, forward_batch, k_nope, k_pe, packed_plan
        )
    plan = getattr(forward_batch, "npu_dcp_extend_gather", None)
    if plan is None:
        # Sized by the shared planner for the whole context, for CUDA's
        # kernels. Nothing on this path reads it, and held it is a
        # context-sized tensor alive through every layer.
        md.dcp_kv_buffer = None
        plan = plan_dcp_extend_gather(
            forward_batch.extend_prefix_lens_cpu,
            forward_batch.extend_seq_lens_cpu,
            parallel.dcp_size,
            parallel.dcp_rank,
            _dcp_extend_gather_piece_rows,
        )
        plan = plan._replace(
            pieces=[
                piece._replace(index=piece.index.to(k_nope.device))
                for piece in plan.pieces
            ]
        )
        forward_batch.npu_dcp_extend_gather = plan
        # Same branch, same once-per-forward guarantee: tell the pool it may
        # drop the rows this rank does not own from this forward's KV write,
        # instead of writing all of them and aiming 15/16 at a padding row.
        # getattr because a wrapper pool (SWA, hybrid) may forward
        # get_mla_kv_buffer without knowing about this; declining is correct
        # there, and silent because it is the pre-existing behaviour.
        plan_write = getattr(get_token_to_kv_pool(), "plan_dcp_extend_write", None)
        if plan_write is not None:
            plan_write(forward_batch.out_cache_loc)
        if _debug_dcp_extend_memory:
            _log_dcp_extend_memory(
                sum(forward_batch.extend_prefix_lens_cpu),
                sum(forward_batch.extend_seq_lens_cpu),
            )

    total_rows = plan.pieces[-1].out_end if plan.pieces else 0
    out_nope = dcp_extend_gather_buffer("latent", k_nope, total_rows)
    out_rope = dcp_extend_gather_buffer("rope", k_pe, total_rows)

    # One scratch per key, sized for the widest piece and sliced per piece, so
    # the reservation happens once rather than once per piece per layer. The
    # widest is not always the first: the last piece carries this chunk's own
    # KV on top of its share of the prefix.
    scratch_rows = max(
        (
            (piece.send_end - piece.send_start) * parallel.dcp_size
            + piece.extend_end
            - piece.extend_start
            for piece in plan.pieces
        ),
        default=0,
    )
    scratch_nope = dcp_extend_gather_buffer("latent_scratch", k_nope, scratch_rows)
    scratch_rope = dcp_extend_gather_buffer("rope_scratch", k_pe, scratch_rows)

    send_nope = send_rope = None
    if plan.send_rows:
        send_nope, send_rope = get_token_to_kv_pool().get_mla_kv_buffer(
            m.attn_mqa,
            md.dcp_local_prefix_kv_indices,
        )
        if plan.local_lens != plan.padded_lens:
            # Served prefixes are dcp_size-aligned (the widened allocator page)
            # and need no padding; this is the general case.
            send_nope = _pad_dcp_extend_send(send_nope, plan)
            send_rope = _pad_dcp_extend_send(send_rope, plan)

    # Every rank plans the same pieces, so every rank runs -- or, for a piece
    # with no prefix rows, skips -- the same collectives in the same order.
    for piece in plan.pieces:
        gathered = (piece.send_end - piece.send_start) * parallel.dcp_size
        rows = gathered + piece.extend_end - piece.extend_start
        for out, buf, send, own in (
            (out_nope, scratch_nope, send_nope, k_nope),
            (out_rope, scratch_rope, send_rope, k_pe),
        ):
            scratch = buf[:rows]
            if gathered:
                parallel.dcp_group.all_gather_into_tensor(
                    scratch[:gathered], send[piece.send_start : piece.send_end]
                )
            scratch[gathered:] = own[piece.extend_start : piece.extend_end]
            torch.index_select(
                scratch, 0, piece.index, out=out[piece.out_start : piece.out_end]
            )
    return out_nope, out_rope


def forward_dsa_core_npu(
    m: "DeepseekV2AttentionMLA",
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    q_nope_out: torch.Tensor,
    k_nope: torch.Tensor,
    topk_indices: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    positions: torch.Tensor,
    mla_preprocess_used: bool,
    # Gated attention (Ling-V3 / BailingMoeV3): the subclass appends its gate
    # to inner_state, so every *_core dispatched from forward_core takes it as
    # a trailing arg. None everywhere else.
    gate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # GLM-5.2 reaches this function, not forward_absorb_core: forward_core
    # dispatches AttnForwardMethod.DSA_NPU here (deepseek_v2.py:2228). So
    # forward_mla.py's DCP block never runs for this model on NPU, and decode
    # context parallelism has to be composed here as well. This mirrors
    # forward_mla.py:640-809 rather than sharing it -- the two prepare/core
    # pairs have different shapes -- so the two must be kept in step by hand.
    if dsa_cp_token_sharded(forward_batch):
        # Wait on the chunk-KV allGather issued in prepare, the last moment
        # before anything reads the gathered rows: the DCP gather's local
        # chunk copy, the DCP SFA-gather finish, and the pool write inside
        # the sparse operator all run from here on. The stash also holds the
        # send buffers, which must survive until the communication engine
        # has read them; dropping it here releases both.
        works = getattr(forward_batch, "npu_dsa_cp_chunk_kv_works", None)
        if works is not None:
            for work in works[:2]:
                if work is not None:
                    work.wait()
            print_info_once(
                "D1v3: the chunk-KV allGather (seam S1) issued in prepare is "
                "done; the gathered KV rows are now readable on every rank."
            )
            forward_batch.npu_dsa_cp_chunk_kv_works = None

    dcp_extend = _dcp_extend_active(forward_batch)
    if dcp_extend:
        # Extend under DCP: gather the context so this rank can see all of it,
        # and hand the result to the backend on the batch. Without this the
        # backend attends over its own shard with a full-span page table -- in
        # bounds and wrong, which is why generation came out fluent and
        # unrelated to the prompt rather than crashing.
        # From the TENSOR, never from m.indexer: the indexer is built only on
        # the layers that compute a top-k (deepseek_v2.py:1828), so it is None
        # on the 57 of 78 that reuse the previous layer's. topk_indices is a
        # real tensor on every layer -- fresh or inherited -- and its width is
        # the same index_topk either way.
        pending = getattr(forward_batch, "npu_dcp_sfa_gather_pending", None)
        forward_batch.npu_dcp_sfa_gather_pending = None
        if pending is not None:
            # The gather was issued at the top of prepare, before the QKV, and
            # has been moving bytes underneath the QKV and (where the layer
            # runs one) the indexer: wait for it NOW, the last moment before
            # the sparse operator reads the buffer (the vLLM wait point).
            forward_batch.npu_dcp_extend_kv = _dcp_finish_extend_gather_npu(
                pending
            )
        else:
            forward_batch.npu_dcp_extend_kv = _dcp_gather_extend_kv_npu(
                m,
                forward_batch,
                k_nope,
                k_pe,
                topk_indices.shape[-1] if topk_indices is not None else None,
            )

    if is_dcp_mla_decode_phase(forward_batch):
        # Every rank attends with the FULL head set against its own KV shard and
        # keeps only its own share after the merge, so the query is gathered
        # across the group first and the attention runs on
        # attn_mqa_for_dcp_decode, which is built at num_local_heads * dcp_size
        # (deepseek_v2.py:1922-1925). --dcp-replicate-q-proj would compute those
        # heads locally and skip this collective; it prepares 0 layers on this
        # checkpoint, so the gather is not optional here.
        q_pre = getattr(forward_batch, "npu_dcp_decode_q_pre", None)
        if q_pre is not None:
            # Issued in prepare, under the layer's indexer: wait for it now,
            # where the synchronous gather used to run, and split the gathered
            # buffer the same way the synchronous path does.
            forward_batch.npu_dcp_decode_q_pre = None
            if q_pre.work is not None:
                q_pre.work.wait()
            gathered_pe, gathered_nope = q_pre.gathered.split(
                [q_pre.d_pe, q_pre.d_nope], dim=-1
            )
            q_pe = gathered_pe.transpose(0, 1)
            q_nope_out = gathered_nope.transpose(0, 1)
        else:
            q_nope_out, q_pe = all_gather_q_for_mla_decode(
                q_nope_out=q_nope_out, q_pe=q_pe
            )
        # save_kv_cache stays True here while the non-DCP branch below took
        # upstream's `not mla_preprocess_used`: the DCP write goes through
        # _resolve_dcp_write, and whether MLA preprocess already wrote these
        # rows under DCP is untested. A duplicate write is idempotent (same
        # data, same translated destination), so True is safe-but-wasteful;
        # `not mla_preprocess_used` here would need a box run to confirm.
        attn_output, lse = m.attn_mqa_for_dcp_decode(
            q_nope_out.contiguous(),
            k_nope.contiguous(),
            k_nope.contiguous(),
            forward_batch,
            save_kv_cache=True,
            q_rope=q_pe.contiguous(),
            k_rope=k_pe.contiguous(),
            topk_indices=topk_indices,
        )
        # The partials are per-head over this rank's tokens; the merge reduces
        # the head axis back to num_local_heads, which is why the shared view
        # below is correct for both branches.
        attn_output = attn_output.view(
            -1, m.num_local_heads * get_parallel().attn_dcp_size, m.kv_lora_rank
        )
        comm_backend = get_parallel().dcp_comm_backend
        # Not cosmetic: feeding a base-e LSE to the base-2 combine is a monotone
        # reweighting, so it stays finite and plausible and only acceptance
        # degrades. "ascend" is a natural-log backend.
        base_on_e = is_mla_dcp_lse_base_on_e(m.current_attention_backend)
        if comm_backend in ("a2a", "fi_a2a"):
            attn_output = dcp_a2a_lse_reduce(
                attn_output.contiguous(),
                lse.contiguous(),
                get_parallel().dcp_group,
                is_lse_base_on_e=base_on_e,
                comm_backend=comm_backend,
            )
        else:
            attn_output = cp_lse_ag_out_rs_mla(
                attn_output,
                lse,
                get_parallel().dcp_group,
                is_lse_base_on_e=base_on_e,
            )
            attn_output = attn_output.transpose(0, 1)
    else:
        attn_mqa = m.attn_mqa
        dsa_cp_plan = get_dsa_cp_plan(forward_batch)
        if dsa_cp_plan is not None and (
            topk_indices is None or m.attn_mqa_for_dsa_cp is None
        ):
            # Both are built from the same condition as the plan, so a mismatch
            # is a wiring bug rather than a configuration. Say which, instead of
            # failing later on a shape.
            raise RuntimeError(
                "DSA-CP planned this forward but the layer is not set up for "
                f"it: attn_mqa_for_dsa_cp={m.attn_mqa_for_dsa_cp is not None}, "
                f"topk_indices={topk_indices is not None}"
            )
        if dsa_cp_plan is not None:
            # DSA-CP. Swap "this rank's heads for every token" for "every head
            # for this rank's tokens". The group holds the same set of
            # (token, head) pairs either way and each is still computed exactly
            # once, so this is the same attention rather than an approximation
            # of it -- what changes is that the operator now reads the top-k KV
            # for 1/attn_tp_size as many queries, and that read is what it is
            # bound by (measured 803 GB/s, HBM speed, at 32.7 GB per layer).
            #
            # k_nope and k_pe stay full width and are NOT touched: the KV cache
            # write and the DCP context gather below both address every token.
            # The width the batch actually carries, which is NOT always
            # plan.num_tokens: SGLang pads the token count up to a multiple of
            # attn_tp_size before the model runs. Everything sized by that
            # padded width -- out_cache_loc above all -- keeps it, so attention
            # has to give it back, not the unpadded count.
            # The same gate as prepare (which additionally needs the plan,
            # which this block is inside of): mla_preprocess runs the fused
            # q path, where the full-width sites do not exist, so prepare
            # would have computed the sharded q. Unreachable today -- the plan
            # is extend-only and preprocess is non-extend-only -- but the two
            # gates must not be allowed to drift apart.
            # Token sharded implies the full-head weights too (its q is rows x
            # all heads from prepare), so it takes the no-redistribute branch
            # even when the version B flag itself is off.
            token_sharded = dsa_cp_token_sharded(forward_batch)
            q_full_width = (
                (_enable_dsa_cp_q_full_width or token_sharded)
                and not mla_preprocess_used
                and getattr(m, "q_b_proj_qrep_weight", None) is not None
            )
            if q_full_width:
                # Version B: prepare computed every head for this rank's rows
                # with the gathered full-head weights, so there is nothing to
                # redistribute. The width the restore hands back is the batch's
                # padded width, not this rank's row count.
                dsa_cp_rows = dsa_cp_plan.num_tokens_pad
            else:
                dsa_cp_rows = q_nope_out.shape[0]
                q_pre = getattr(forward_batch, "npu_dsa_cp_q_pre", None)
                if q_pre is not None:
                    # Version A: both all-to-alls ran under the indexer; wait
                    # on them now, the last moment before the operator reads
                    # the tensors, and only then materialise the recv buffers
                    # (their permute+reshape is a copy, which must not run
                    # while the collectives are still writing). Dropping the
                    # stash here also releases the send buffers the
                    # communication stream is done with.
                    forward_batch.npu_dsa_cp_q_pre = None
                    for work in q_pre[2:4]:
                        if work is not None:
                            work.wait()
                    q_nope_out = dsa_cp_redistribute_heads_recv(
                        q_pre[0], dsa_cp_plan
                    )
                    q_pe = dsa_cp_redistribute_heads_recv(q_pre[1], dsa_cp_plan)
                else:
                    q_nope_out = dsa_cp_redistribute_heads(q_nope_out, dsa_cp_plan)
                    q_pe = dsa_cp_redistribute_heads(q_pe, dsa_cp_plan)
            if (
                token_sharded
                or (_enable_dsa_cp_local_topk
                    and getattr(forward_batch, "npu_dsa_topk_local", False))
            ):
                # The indexer kept its top-k in this rank's query rows (the
                # plan's own split), so there is nothing to slice; the local
                # rows already are this rank's. See the indexer for why the
                # two splits agree. Token sharding takes the same branch
                # unconditionally: under it the top-k is local by
                # construction, independent of the standalone local-top-k
                # switch.
                attn_topk_indices = topk_indices
            else:
                # The indexer ran at full width, so its top-k is full width
                # too; take this rank's rows of it. Padded rows get index 0,
                # which is a valid position whose output is discarded on the
                # way back.
                #
                # A SEPARATE NAME, and this is load-bearing rather than style.
                # This function RETURNS topk_indices (see the end), and the
                # caller feeds it to the next layer as prev_topk_indices: only
                # 21 of 78 layers run the indexer, the other 57 reuse what came
                # before. Rebinding the name here handed the next layer this
                # rank's slice instead of the full-width top-k -- and the layer
                # after that would slice the slice, which for every rank above
                # 0 starts past the end and so reads as all-zero: attend to
                # position 0, for 57 layers.
                attn_topk_indices = dsa_cp_slice(topk_indices, dsa_cp_plan)
            attn_mqa = m.attn_mqa_for_dsa_cp
        else:
            attn_topk_indices = topk_indices
        attn_output = attn_mqa(
            q_nope_out.contiguous(),
            k_nope.contiguous(),
            k_nope.contiguous(),
            forward_batch,
            save_kv_cache=not mla_preprocess_used,
            q_rope=q_pe.contiguous(),
            k_rope=k_pe.contiguous(),
            topk_indices=attn_topk_indices,
        )
        if dsa_cp_plan is not None and not token_sharded:
            # Undo the swap before anything else sees it. w_vc, o_proj and the
            # layer communicator all expect this rank's own heads for the whole
            # batch, which is what they get without DSA-CP, so the sharding
            # ends here instead of propagating through the rest of the layer.
            # Token sharded has no swap to undo: the full w_vc and the full
            # o_proj below consume rows x all heads and emit the COMPLETE
            # [rows, hidden] result, so the sharding propagates through the
            # rest of the layer instead (the MoE gathers it back to full width).
            attn_output = dsa_cp_restore_tokens(
                attn_output.reshape(dsa_cp_plan.rows, -1, m.kv_lora_rank),
                dsa_cp_plan,
                dsa_cp_rows,
            )
    if dcp_extend:
        # Dropped here, before the MoE, not at the end of the forward. The
        # tensors themselves are now reserved buffers and survive either way
        # (_dcp_extend_gather_buffer); what this drops is the batch's reference,
        # so a stale gather can never be read by a later forward that took a
        # different path.
        forward_batch.npu_dcp_extend_kv = None
    # Token sharded keeps rows x ALL heads here (no restore a2a ran), so the
    # absorb and the o_proj take the full replicated weights and emit the
    # complete [rows, hidden] result; everything else keeps this rank's head
    # shard.
    if dsa_cp_token_sharded(forward_batch):
        attn_output = attn_output.view(-1, m.num_heads, m.kv_lora_rank)
        w_vc = m.w_vc_qrep
    else:
        attn_output = attn_output.view(-1, m.num_local_heads, m.kv_lora_rank)
        w_vc = m.w_vc

    if _is_npu_arch35 or (
        forward_batch.forward_mode.is_extend()
        and not forward_batch.forward_mode.is_draft_extend_v2()
        and not forward_batch.forward_mode.is_target_verify()
    ):
        attn_bmm_output = torch_npu.npu_transpose_batchmatmul(
            attn_output,
            w_vc,
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        )
    else:
        attn_bmm_output = torch.empty(
            (attn_output.shape[0], attn_output.shape[1], m.v_head_dim),
            dtype=attn_output.dtype,
            device=attn_output.device,
        )
        attn_output = attn_output.contiguous()
        torch.ops.npu.batch_matmul_transpose(attn_output, w_vc, attn_bmm_output)

    attn_bmm_output = attn_bmm_output.reshape(-1, attn_output.shape[1] * m.v_head_dim)

    if gate is not None:
        attn_bmm_output = m._apply_gated(attn_bmm_output, gate)
    if dsa_cp_token_sharded(forward_batch):
        # Full W_O: the complete result for this rank's rows, no allReduce.
        if _stream_o_proj:
            # D2v3: the full matrix was assembled on the gather side stream
            # one layer ahead; take_o_proj waits on its ready event here, the
            # last moment before this GEMM reads the slot.
            print_info_once(
                "D2v3: o_proj runs the STREAMED full W_O (all-gathered on the "
                "DCP side stream after this layer's KV gather), so the "
                "attention output is complete and the o_proj allReduce is "
                "skipped, with no standing per-device replica."
            )
            output = torch.nn.functional.linear(
                attn_bmm_output, _dcp_o_proj_weight(m)
            )
        else:
            print_info_once(
                "D1v3: o_proj runs the full replicated W_O (o_proj_qrep_weight) "
                "on this rank's rows, so the attention output is complete and the "
                "o_proj allReduce is skipped."
            )
            output = torch.nn.functional.linear(attn_bmm_output, m.o_proj_qrep_weight)
    else:
        output, _ = m.o_proj(attn_bmm_output)
    if not m.next_skip_topk:
        return output, None
    else:
        return output, topk_indices


def npu_mla_preprocess(
    m: "DeepseekV2AttentionMLA",
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
):
    dynamic_scale = None
    if not hasattr(m, "mla_preprocess"):
        m.mla_preprocess = NPUFusedMLAPreprocess(
            m.fused_qkv_a_proj_with_mqa,
            m.q_a_layernorm,
            m.kv_a_layernorm,
            m.q_b_proj,
            m.w_kc,
            m.rotary_emb,
            m.layer_id,
            m.num_local_heads,
            m.qk_nope_head_dim,
            m.qk_rope_head_dim,
            m.v_head_dim,
            m.quant_config,
        )
        if (
            get_disagg().disaggregation_mode == "decode"
            and m.mla_preprocess.uses_mlaprolog()
            and m.w_kc is not None
        ):
            m.w_kc.untyped_storage().resize_(0)
    # mlaprolog does not require additional calculation of q_lora
    if m.mla_preprocess.uses_mlaprolog():
        (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            q_lora,
            forward_batch,
            positions,
            dynamic_scale,
        ) = m.mla_preprocess.forward(
            positions, hidden_states, forward_batch, zero_allocator
        )
    else:
        if m.alt_stream is not None:
            mla_event = torch.npu.Event()
            mla_event.record()
            with torch.npu.stream(m.alt_stream):
                # alt stream waits for the completion of the event on the main stream to ensure data dependency is complete
                torch.npu.current_stream().wait_event(mla_event)
                (
                    q_pe,
                    k_pe,
                    q_nope_out,
                    k_nope,
                    forward_batch,
                    zero_allocator,
                    positions,
                ) = m.mla_preprocess.forward(
                    positions, hidden_states, forward_batch, zero_allocator
                )

            fused_qkv_a_proj_out = m.fused_qkv_a_proj_with_mqa(hidden_states)[0]
            q, _ = fused_qkv_a_proj_out.split(
                [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim], dim=-1
            )
            q_lora = m.q_a_layernorm(q)
            torch.npu.current_stream().wait_event(m.alt_stream)
        else:
            (
                q_pe,
                k_pe,
                q_nope_out,
                k_nope,
                forward_batch,
                zero_allocator,
                positions,
            ) = m.mla_preprocess.forward(
                positions, hidden_states, forward_batch, zero_allocator
            )
            fused_qkv_a_proj_out = m.fused_qkv_a_proj_with_mqa(hidden_states)[0]
            q, _ = fused_qkv_a_proj_out.split(
                [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim], dim=-1
            )
            q_lora = m.q_a_layernorm(q)

    return (
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        q_lora,
        forward_batch,
        zero_allocator,
        positions,
        dynamic_scale,
    )


# endregion
