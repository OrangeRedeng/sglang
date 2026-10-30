# Code audit: bde_v2 snapshot vs `a3_910c_int_glm53` (tip `0b3f0e2` + pending restoration)

Baseline: `.codex-artifacts/sglang_bde_v2` (`combined` @ `5d60526`) with the
`p4_exit1_serve_dcp.sh` recipe. Branch: this worktree at `0b3f0e2` plus the
uncommitted restoration delta (verified together; committed together).

Verdict: **every environment knob of the bde recipe is accounted for.** The
A-branch/V3 tuning knobs are the only removals, and both were superseded by the
restored packed-read implementation rather than dropped. The recipe runs as-is
at TP16/DCP16 — `validate_legacy_dcp_flags` accepts it, and every hard
precondition of the D1v3 rebuild passes (see the gate table at the bottom).

## Recipe knob mapping

| bde env (recipe value) | Class | Branch evidence (`file:symbol`) | Phase B enable |
| --- | --- | --- | --- |
| `SGLANG_NPU_ENABLE_DCP_SHARED_GATHER=True` | REPLACED | Superseded by the packed prefix read: `environ.py:1240-1252` (packed = "two collectives per layer, no index_select"), `layers/dcp/layout.py:736` | omit — no consumer exists |
| `SGLANG_NPU_DCP_GATHER_ASYNC=True` | REPLACED | Async gather is the implementation default: `dcp_gather.py:start_o_proj_gather`, `DcpGatherPipeline` side-stream waits | omit |
| `SGLANG_NPU_DCP_SHARED_PACKED_MAX_ROWS=0` | REPLACED | Plan policy owned by `dcp_packed_read_plan`; every refusal is logged (cannot turn off silently) | omit |
| `SGLANG_NPU_DCP_SHARED_READ_PAGE_SIZE=128` | REPLACED | Page-owned mapping in `dcp_gather.py`; new tuning knob `SGLANG_NPU_DCP_EXTEND_GATHER_PIECE_ROWS` (`environ.py:1398`) | omit |
| `SGLANG_NPU_DCP_SHARED_READ_SKIP_CONTIGUOUS=False` | REPLACED | Same — contiguous handling moved into the plan | omit |
| `SGLANG_NPU_INDEXER_INT8=True` | RETAINED (default False) | `environ.py:1411`; `dsa_indexer_kpool.py:66`, `memory_pool_npu.py:739`, `utils.py:211` (custom_ops force-disabled) | launcher sets `True` (fixed) |
| `SGLANG_NPU_SFA_KV_8BIT=1` | RETAINED (default False; needs DCP≥2) | `environ.py:1426`; `memory_pool_npu.py:717,759,878`; `kv_cache_configurator.py:3485`; `mla_preprocess.py:406`; `ascend_backend.py:1857` | launcher `setdefault 1` |
| `SGLANG_NPU_ENABLE_DCP_SFA_GATHER_OVERLAP=1` | RETAINED (default False, "pending box") | `environ.py:1305` with measured A/B data in comments | launcher sets `True` |
| `SGLANG_NPU_ENABLE_DSA_CP=True` | RETAINED (default False) | `environ.py:1219`; `dcp_gather.py:validate_legacy_dcp_flags` | launcher sets `True` (fixed) |
| `SGLANG_NPU_DSA_CP_TOKEN_SHARDED=1` | RESTORED, hard-gated (default False) | `environ.py:1364`; `model_runner.py:_prepare_dsa_cp_token_sharded` (refuses instead of degrading); `modules/dsa_token_residual.py` (new) | launcher `setdefault 1` |
| `SGLANG_NPU_DSA_CP_Q_FULL_WIDTH=1` | RETAINED (default True) | `environ.py:1387`; `model_runner.py:_prepare_dsa_cp_full_q_weights` — W4A8 supported via identity-GEMM dequant | launcher `setdefault 1` |
| `SGLANG_NPU_DSA_CP_LOCAL_TOPK=1` | RETAINED (default True) | `environ.py:1337` | launcher `setdefault 1` |
| `SGLANG_NPU_DCP_DECODE_Q_PREISSUE=1` | RETAINED (default True; needs DCP≥2) | `environ.py:1396`; `dcp_gather.py:43` | launcher `setdefault 1` |
| `SGLANG_NPU_ENABLE_DCP_PACKED_READ=True` | RETAINED (default True) | `environ.py:1248`; `layers/dcp/layout.py:736` | launcher sets `True` |
| `SGLANG_NPU_ENABLE_DCP_GATHER_PREFETCH=True` | RETAINED (default True, D2 placement fix) | `environ.py:1273` | launcher sets `True` |
| `SGLANG_NPU_STREAM_O_PROJ=1` | RETAINED (default True; needs TOKEN_SHARDED + TP==DCP) | `environ.py:1327`; `dcp_gather.py:31-38`; `dsa_token_residual.py:263`; `model_runner.py` stream path | launcher `setdefault 1` |
| `SGLANG_ENABLE_CHAT_ENCODE_POOL=True` | RETAINED (default True) | `environ.py:341`; `tokenizer_manager.py:681` | launcher `True` (fixed) |
| `SGLANG_CHAT_ENCODE_MEMO=True` | RETAINED (default True) | `environ.py:352`; `managers/chat_encode_memo.py`; `tokenizer_manager.py:695` | launcher `True` (fixed) |
| `SGLANG_CHAT_ENCODE_RAW=True` | RETAINED (default False) | `environ.py:358` | launcher `True` (fixed) |
| `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP=True` | RETAINED (default False) | `environ.py:716` | launcher `True` (fixed) |
| `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` | RETAINED (default False) | `environ.py:1820` | launcher sets when `SAFE_ENV!=1` |
| `SGLANG_NPU_USE_MULTI_STREAM=1` | RETAINED (default False) | `environ.py:1171`; `dsa_npu_indexer.py:326,355,479`; `deepseek_v2.py:1060,3271` | launcher sets when `SAFE_ENV!=1` |
| `DEEPEP_HCCL_BUFFSIZE=2500` | passthrough (DeepEP) | consumed by DeepEP, not sglang | launcher sets when `SAFE_ENV!=1` |
| `HCCL_BUFFSIZE / HCCL_OP_EXPANSION_MODE / HCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME / STREAMS_PER_DEVICE / PYTORCH_NPU_ALLOC_CONF` | passthrough (HCCL / torch_npu) | not sglang consumers | launcher sets all |
| `SGLANG_SET_CPU_AFFINITY=1` | RETAINED (default False) | `environ.py:2024` | launcher `1` (fixed) |
| `[SGL-TS]` stage probes | RETAINED — all five sites | `serving_base.py:90` (RECV-HTTP), `tokenizer_manager.py:978` (RECV-TM), `scheduler.py:4721,5036` (BATCH-START/END), `model_runner.py:2172,2266` (MODEL-FWD-START/END), `serving_chat.py:2984` (SENDBACK); test `unit/managers/test_scheduler_stage_latency.py` | automatic (always on) |

## Branch additions beyond the bde recipe (off by default = bde-parity)

- MoE INT stack: TP fusion + GMM2/shared finalize, INT8 reuse, fused norm+INT
  quant (`hardware_backend/npu/moe/*`, `layernorm_quant.py`).
- Prefill CUDA graph + piecewise backend (`compilation/npu_piecewise_backend.py`);
  the launcher pins `--cuda-graph-backend-prefill disabled` and keeps decode
  graphs off unless `CAPTURE=1`, matching the bde recipe.
- Online INT quantization `--online-quantization w8a8_int|w4a4_int` (load-time).
- Dynamo traceability fixes (residual handoff, FFN exit, routing L1-norm custom
  op, aten-based attention-output allocation in `radix_attention.py`).
- Autotune knobs `SGLANG_NPU_AUTO_STREAM_THRESHOLDS`,
  `SGLANG_NPU_AUTO_DCP_EXTEND_GATHER_PIECE_ROWS`, per-path `*_MIN_TOKENS`
  (`environ.py:1151-1161`) — all default off.

## D1v3 gate table for the TP16/DCP16 Phase B config

| `_prepare_dsa_cp_token_sharded` precondition | Phase B value | Pass |
| --- | --- | --- |
| NPU process | NPU host | yes |
| `attn_tp_size > 1` | 16 | yes |
| MoE a2a backend is none | `--moe-a2a-backend none` (recipe + launcher) | yes |
| No speculative decoding | none | yes |
| No DP attention | off | yes |
| No prefill CP | off (flag removed upstream; recipe omits it) | yes |
| Not `SGLANG_NPU_USE_MLAPO` | unset | yes |
| Not `SGLANG_SHARED_EXPERT_TP1` | unset | yes |
| Model has DSA layers, all MLA with q_b/kv_b | GLM-5.2-W4A8 (glm4_moe DSA) | yes |
| `validate_legacy_dcp_flags` combo | DSA_CP=1 + TOKEN_SHARDED=1 + Q_FULL_WIDTH=1 + STREAM_O_PROJ=1, TP==DCP==16≥2 | yes |

Caveat (by design): with TOKEN_SHARDED on, EXTEND graph capture raises — the
recipe runs `CAPTURE=0`, so this never triggers in the A/B.

## Fix applied during this audit

`launch_bde_v2.py`: the three restored DCP opt-ins (packed read, gather
prefetch, SFA-gather overlap) moved from the fixed dict to defaults
(`setdefault`), preserving the bde recipe values (True/True/True) while letting
an attribution run override any of them from the environment. No other drift:
the launcher already omits the superseded A/V3 knobs and reproduces the rest of
the recipe.
