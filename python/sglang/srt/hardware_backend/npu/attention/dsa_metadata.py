"""Opt-in page-table preparation for eager DSA prefill."""

import torch
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.feature_evidence import record
from sglang.srt.runtime_context import get_parallel


def prepare_dsa_page_tables(backend, batch):
    if not envs.SGLANG_NPU_DSA_METADATA_REUSE.get():
        return False
    if (
        not backend.use_dsa
        or backend.is_hybrid_swa
        or getattr(backend, "is_draft_worker", False)
        or getattr(backend, "graph_mode", False)
        or not batch.forward_mode.is_extend()
        or batch.forward_mode.is_target_verify()
        or batch.forward_mode.is_draft_extend_v2()
    ):
        return False
    lengths = batch.seq_lens_cpu
    if not isinstance(lengths, torch.Tensor) or lengths.device.type != "cpu":
        return False
    lengths = lengths.tolist()
    prefix, extend = batch.extend_prefix_lens_cpu, batch.extend_seq_lens_cpu
    if (
        prefix is None
        or extend is None
        or len(lengths) != len(prefix)
        or len(lengths) != len(extend)
        or any(n != p + e for n, p, e in zip(lengths, prefix, extend))
    ):
        return False
    max_len = max(lengths, default=0)
    if max_len < 0:
        return False
    page_size = backend.page_size
    # Read page starts directly instead of gathering the entire token map twice.
    page_locs = backend.req_to_token_pool.req_to_token[
        batch.req_pool_indices, :max_len:page_size
    ]
    backend.forward_metadata.block_tables = page_locs // page_size
    parallel = get_parallel()
    if parallel.dcp_enabled:
        size = parallel.attn_dcp_size
        # Indexer pages span virtual tokens; latent KV pages are rank-local.
        backend.forward_metadata.dcp_local_block_tables = page_locs[:, ::size] // (
            page_size * size
        )
    record("dsa-metadata-reuse", max_length=max_len, batch_size=len(lengths))
    return True
