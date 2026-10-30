# BDE runtime restoration (2026-10-30)

The eager A3 INT path now connects the archived packed-prefix/prefetch design,
full-head attention, token-local residual/MoE execution, replicated index-K
writes, streamed output projection and decode query preissue to their actual
runtime callers. This supersedes the earlier statement that legacy D1v3 was
disabled. Normal inference paths and page-interleaved KV ownership are retained.
The pending SGL-TS probes are now default-off via `SGLANG_DEBUG_STAGE_LATENCY`.

Focused host validation: 99 tests and 122 subtests passed; the pre-existing
`test_plain_extend_plans_int32_inputs` mock-planning failure is excluded.
The strengthened native decoder regression fails against the pre-fix entry
point. Native indexer row ownership covers both rotary layouts and replicated
INT8 keys/scales; packed attention verifies padded KV extent and causal mode.
These are CPU/operator-boundary checks, not an NPU performance result.

Mainline selection and the controlled TP16 launch/benchmark procedure are in
[`BDE_RESTORATION.md`](benchmark/prefill_1m/repro_bde_v2/BDE_RESTORATION.md).
Target-NPU numerical parity, activation, HCCL, peak memory and the reported
20-second baseline comparison remain pending. No fastest configuration is
claimed from host tests.

---

# Current integration update (2026-10-28)

Integrated code from the following live PR heads without changing their upstream PRs:

| PR | Head | Scope |
|---|---|---|
| #42977 | 3941dc3719 | DSA indexer query sharding |
| #42978 | ccbd6b1692 | Token sharding, padded-width/CP refusal, paged key frontier |
| #44320 | a8866e572d | DCP prefix-KV gather and buffer reuse |
| #44321 | 7c0c8398ad | Token shard composed with DCP KV gather |
| #43141 | 3dd268b4f4 | DCP group nesting and KV-dtype validation |
| #42224 | b71a0c85bd | INT8 indexer/FP16 scales, HiCache exclusion |

The combined #44321 delta supplies its parent PRs; newer #42978 safety fixes
are applied on top. These are semantic integrations onto the existing branch,
not wholesale replacements with upstream main. Graph and INT components remain.

Repaired the function-local `get_flags` shadow, added the GLM DSA W4A4 allowlist,
restored the packed-read helper definitions needed during native model import,
and updated the tests and feature matrix for supported INT contracts and the
current runtime-context/shared-expert APIs. Legacy D1v3 is disabled in favor of
the reviewed token-shard path. The indexer INT8 path follows the PR's HiCache
exclusion. Physical KV write IDs are not translated a second time.

Validation for this integration: all seven host-runner checks passed; 36
host encoding/scheduling tests passed; 21 isolated CPU planner tests passed
against real PyTorch with runtime/import boundaries stubbed. Compile checks,
F821/F823 and diff checks passed. Broader local suite collection is blocked
by missing TVM FFI/datasets/torchvision dependencies. A3 operator, native-model
startup, NPUGraph replay, quality and performance must be rerun on the target.

The original A3 report is user-provided evidence. Host/static verification
does not establish A3 graph replay, quantized model quality or performance.

---

# Branch report: `a3_910c_int_glm53`

**HEAD:** `24b0afbe9b` · worktree `.codex-artifacts/worktrees/a3-int-glm53` · clean
**Base lineage:** bde_v2 `combined` (5d60526) → gemma4 graph-dispatch → moe-opts/megamoe/cleanup → 29048-SRT → origin/main (b74fd392, Oct 28) → gemma_fix_and_perf_impr (task-queue excluded) → INT-only registry → merge-repair fixes
**Target:** GLM-5.3 (`glm_moe_dsa` / `Glm5NextForConditionalGeneration`) on Ascend A3 910C; INT-only online quantization

---

## Supported features

### Quantization (INT-only online stack)
| Feature | Flag | Notes |
|---|---|---|
| Load-time W8A8 (dense + MoE experts) | `--online-quantization w8a8_int` | From #29048 SRT: q/k/v/qkv/o, gate_up, down; completion-tracked shard-aware loaders |
| Mixed load-time quant | `--online-quantization w4a4_int` | dense W8A8 + MoE w13 W4A4 / w2 W8A8; MoE-only (allowlist: Qwen3MoE, **Glm5Next**) |
| Scoped post-load conversion | `--online-quantization TARGET:SCHEME` | INT schemes only: `w8a8_dynamic`/`w8a8_int8`/`int8` (INT8), `w4a4_dynamic` (INT4); targets gate/qkv/dense/shared_expert/all. MX aliases removed, MX schemes fail closed |
| Independent W13/W2 precision | fused-MoE INT4/INT8 schemes | canonical schemes, per-tensor precision choice |
| Router-gate online conversion | gate contract registry | INT8 dynamic (`W8A8_DYNAMIC`); norm fused with quant via `norm_gate` |

### MoE optimizations (A3, INT paths)
| Feature | Where | GLM-5.3 |
|---|---|---|
| TP dispatch/finalize fusion (routed-scale fold, shared skip1) | `tp_fusion.py` + `configure_ascend_tp_finalize_fusion` | yes — instance-based on DeepseekV2MoE lineage |
| Shared-expert TP fusion + count refinement | `npu_tp_shared_expert_count` | yes |
| TP-shared load tracking | `record_tp_shared_load` per local expert | yes |
| Fused post-attn norm + FFN boundary INT quant | `NORM_QUANT_READOUT` + `ffn_quant_format` | **wired this round** (`Glm5NextDecoderLayer`) |
| Top-k routing L1-norm as traceable custom op | `npu_topk_l1_norm` | yes (DSA path) |
| Fused Gemma4 QKV norm | `SGLANG_ENABLE_NPU_GEMMA4_QKV_NORM` | n/a (Gemma-only; GLM-5.3 MLA has no q/k RMSNorm) |

### Attention / context parallelism (DSA)
| Feature | Gate | GLM-5.3 |
|---|---|---|
| int8 LightningIndexer index-K pool | `SGLANG_NPU_INDEXER_INT8` | yes (DSA indexer) |
| Packed int8 SFA latent KV (656 B/row) | `SGLANG_NPU_SFA_KV_8BIT` | yes, requires DCP > 1 |
| DCP shared-KV gather (+async, packed read) | `SGLANG_NPU_ENABLE_DCP_SHARED_GATHER`, `_GATHER_ASYNC`, `DCP_SHARED_*` | yes |
| Indexer query sharding (measured 15.97× kernel-side at 1M/TP16) | `SGLANG_NPU_DSA_INDEXER_QUERY_SHARDING_MIN_TOKENS` | yes |
| DSA-CP token-sharded extend rebuild (D1v3) | `get_forward().npu_dsa_cp_token_sharded` | yes (prepare/core split) |
| Prefill body-capture CUDA graph | capability-gated: FIA + non-DCP + no offload | yes, automatically |
| Piecewise prefill graph backend (fork) | `cuda_graph_config` | yes |

### Host / serving path
| Feature | Gate |
|---|---|
| Off-loop chat render+encode pool (E1) | `SGLANG_ENABLE_CHAT_ENCODE_POOL` |
| Render+encode family memo (E2a) | `SGLANG_CHAT_ENCODE_MEMO` |
| Raw Rust encode (E2d) | `SGLANG_CHAT_ENCODE_RAW` |
| `[SGL-TS]` stage-latency probes (RECV-HTTP→RECV-TM→BATCH→FWD→SENDBACK) | always-on prints (strip before quoting clean numbers) |
| Prefill multistream scheduling / autotune framework / DCP budget reuse | megamoe keep-list, opt-in via env/args |

### Deliberately excluded
- task-queue / phase-adaptive dispatch policy (`task_queue.py`, env vars, 6 wiring sites) — per request
- MXFP in *our additions*: scoped MX schemes, MX gate contracts — removed; upstream main's own W4A8-MXFP checkpoint serving and mxfp8 kernel paths remain (not ours to delete)

---

## Bugs found and fixed (`24b0afbe9b`)

All were undefined names (`ruff F821`) — silent crashes at runtime, found by static sweep of the merge hot zones:

1. **`dsa_npu_indexer.py`** — `_get_indexer_query_shard(..., layer_scatter_modes)`: upstream #41438 removed the per-layer scatter-mode argument (attention input is `TP_ATTN_FULL` by declaration); call now matches the 2-arg helper. Would `NameError` on every DSA prefill.
2. **`forward_mla.py`** — the narrow `is_dcp_mla_decode_phase` predicate (decode/target-verify only) that the NPU DCP extend-gather seam imports had been dropped in favor of the same-named mixin method that *also* admits DSA-extend; restored as a free function with the narrow semantics the seam requires (the extend path handles extend itself).
3. **`deepseek_v2.py`** — `forward_normal(..., num_token_non_padded=num_token_non_padded)` referenced a bare local that no longer exists after upstream moved the threading to `forward_batch.moe_num_token_non_padded()`; call now derives from `forward_batch` (None-safe).
4. **`deepseek_v2.py`** — dead `dispatch_info` block referencing `ExpertLocationDispatchInfo`, an import upstream removed, and consumed by nothing; deleted.
5. **`chat_encode_memo.py`** — missing `List` import for the `commit_full` annotation.
6. **`linear.py`** — the `norm_gate.online_linear_method` quant-method rebind now sits under the module-level `_is_npu` flag (was an undefined bare `_is_npu` that would `NameError` on *every* platform at first linear-layer construction).

## Verification
- `ruff --select F821` over `python/sglang/srt`: **clean**
- `python -m compileall` over `srt`, `kernels`, `benchmark/prefill_1m`: **green**
- conflict markers: **0**; ancestry: all 8 source heads (`b74fd392`, fork branches, 29048, megamoe/cleanup/gemma4) reachable
- bench `--dry-run` at 1,048,576 tokens: green
- Remaining ruff F722/F405/F841 noise: verified **pre-existing on the clean baseline** (upstream `A[...]` annotation style, unused locals in untouched files) — not introduced by this branch

## Known deferred items (need A3 hardware)
1. D1v3 `prepare_mlp` hook → `layer_boundary` declared-boundary port (optimization degraded to safe default)
2. `dcp_packed_read_plan` vs `remap_dcp_sparse_indices` precedence on device
3. Indexer-K paged sizing (`index_size` × `index_page_padding`) at DCP > 1
4. Prefill CUDA graph excludes DCP runs by design — capture win applies to non-DCP configs
5. `[SGL-TS]` probes always-on — gate or strip before clean throughput numbers
