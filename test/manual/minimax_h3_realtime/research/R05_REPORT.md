# R05 — Selective #53105 DiT port and communication overlap: report

**Branch:** `feat/minimax-h3-r05-53105-port-overlap` (from R01 `68e373c50c47492e5860e2d18a9599133ea9c754`)
**Date:** 28 November 2026 · code phase · host tests only, no device runs
**Scope:** atomic port of the applicable, exact/numerical parts of [sgl-project/sglang#53105](https://github.com/sgl-project/sglang/pull/53105) (head `b14183c4e3d8f05dd1f5b4d496b3566b946ec781`) onto the R01 H3 baseline, plus gated overlap enablement and E6/E7 probes. The NPU Triton kernel files of #53105 were deliberately **not** ported (R05 rule: no new kernels).

## 0. Baseline audit (what R01 already had)

The R01 base already contains machinery the PR was written against or duplicates:

- `_usp_input_all_to_all_packed_qkv` (single destination-major QKV collective) with `pack_qkv_destination_major` CUDA kernels and cached staging buffers; `uncached_a2a_staging`; IPC A2A transports.
- `ulysses_head_group_pipelined_attention` (collective variant, aiter-only) and the copy-engine pipeline (`ipc_a2a_multi.ulysses_pipelined_attention`) — CUDA/ROCm only.
- `_fuses_gate_norm2` already returns `False` on NPU (the `norm2.eps` crash guard is **already present**; this port adds the regression test instead of the fix).
- `SGLANG_H3_LASER_CROP_BEFORE_CAST` crop-order toggle and `preserve_bf16_range` FP16 range compensation in `laser_attn.py`.
- `SGLANG_H3_LASER_SP8_PAD_HEADS` (TP2/Ulysses8 28→32 head padding).

Therefore the R05 delta is exactly: E1 runs, the cat-pack fallback + NPU pipelining, the direct varlen postprocess, the `npu_add_rms_norm` fused paths, and the NPU Laser pipeline gate.

## 1. Commit map (atomic)

| Commit | Subtask | Files |
|---|---|---|
| `Port #53105 packed QKV exchange and NPU head-group pipeline` | E2/E5 | `runtime/layers/usp.py` |
| `Port #53105 direct Laser varlen postprocessing` | E3 | `runtime/layers/attention/backends/laser_attn.py` |
| `Port #53105 contiguous-run AdaLN modulation (block_combined_runs)` | E1 | `denoise_loop.py`, `runtime/models/dits/minimax_h3.py`, `breakable_cuda_graph/model_padders/minimax_h3.py` |
| `Port #53105 NPU fused residual+norm with cross-block norm1 fusion` | E4 | `runtime/models/dits/minimax_h3.py` |
| `Add R05 E5 gating and E6/E7 probes with host tests` | E5–E7 | `minimax_h3.py`, `research/r05_comm_probe.py`, `test/unit/test_minimax_h3_r05_port.py` |

## 1.5 Test results (host, 28 Nov)

Executed on this host with CPU torch 2.15.0+cpu and a `triton` import shim
(`/tmp/h3_host_shim`, never committed — the chain only needs Triton importable;
no kernel executes on host). Host execution is **not** hardware validation.

- `test_minimax_h3_r05_port.py`: **14 passed** (0 failed, 0 skipped).
- Regression: `test_minimax_h3_dit_contract.py` **38 passed, 5 failed, 4 skipped
  — identical before and after this port** (the 5 PDD/AdaLN graph-replay
  failures pre-exist on pristine R01 `68e373c50c` in this temporary
  environment; they need graph/device support absent here). Zero regressions
  attributable to R05.
- Lint: `ruff check` clean on all seven touched files (pristine-base check of
  the same files also clean).

## 2. Per-subtask status

### E0 same-source control — device-pending
No code change (by design). Runbook entry: one warmed B0 run on the R05 branch with every R05 path disabled (`MINIMAX_H3_NORM_MODULATE` off, no `SGLANG_H3_LASER_SP8_PAD_HEADS`, pipeline groups off) must reproduce the R01 B0 output hash and timing distribution before any ablation. The R01 verified-checkpoint contract applies unchanged.

### E1 contiguous-run AdaLN (`block_combined_runs`) — implemented, host-tested, device-pending
- `denoise_loop.py`: `_contiguous_index_runs(indices_cpu, max_runs=8)` compresses the combined AdaLN index row set into `(start, stop, value)` runs on the **host**, from new host mirrors (`img_cond_seq_idx_host`, `audio_*_seq_idx_host`, `block_token_tags_host`) — no `.item()`, no device-to-host sync. Runs are built per step pattern in `_expand_step_timesteps` (NPU-only), cached per device tensor behind a `weakref` and emitted by `forward_kwargs` as `block_combined_runs` only while the weakref still resolves to the exact tensor; `prepare_timestep_plan` clears the cache per pass.
- `minimax_h3.py`: `_apply_scale_shift_runs_` / `_apply_indexed_gate_runs_inplace_` broadcast compact table rows per contiguous run (one `mul_`/`add_` pair per run instead of per-row gathers; `1.0 + scale` computed on the tiny `[M*3, H]` table). Dispatched only on NPU, BF16 tables, inference mode, and outside breakable CUDA graphs; anything else keeps `index_select`. Elementwise BF16 rounding is unchanged — CPU parity tests prove run-vs-gather `torch.equal`.
- Runs are invalidated when selective-FFN compaction reindexes rows; fragmented layouts (>8 runs) and direct `forward()` callers without the matching tensor fall back. The BCG padder drops `block_combined_runs`; the forward kwarg contract registers it.
- Device-pending: launch-count/byte savings on A3; cache-dit + runs interaction is prevented (BCG only), not yet measured.

### E2 QKV packing — implemented, host-tested, device-pending
- The non-kernel fallback of `_usp_input_all_to_all_packed_qkv` now builds the packed source with one `cat/view/permute/contiguous` instead of `empty` + three strided `copy_` writes; `torch.equal` against the strided-copy reference (CPU test). The `pack_qkv_destination_major` CUDA path is untouched.
- New `_usp_exchange_prepacked_qkv(packed)` for callers that already hold the `[W, S, H, 3D]` layout.
- `ulysses_head_group_pipelined_attention` accepts `prepacked_qkv=` (NPU-only, `ValueError` elsewhere) and pipelines plain `all_to_all_single` exchanges per head group on NPU: all input exchanges queued first, group g's output exchange queued while group g+1 attends, single final permute. No extra per-step blocking wait is inserted on the attended path; the CUDA/ROCm `pack_qkv_destination_major` path is unchanged.
- Device-pending: HCCL allocation counts and end-to-end effect vs the strided-copy variant (the handoff required measuring both — done bitwise on host; timing needs A3/A5).

### E3 direct Laser varlen postprocessing — implemented, host-tested, device-pending
- In the single-real-segment varlen branch (`[0, used, total]`), one call attends the real rows and only the valid region is written/scaled/zeroed, instead of dispatching `_forward_dense` and materializing its padded cast/output.
- Bitwise-parity note: the value rescale is applied to the operator's FP16 output **before** the crop and BF16 cast, exactly matching `_forward_dense`'s order; a CPU test proves `forward_varlen` output `torch.equal` to the replaced segment path (deterministic fake operator) and that the dense path is not called at/above `min_seqlen`. Below `min_seqlen` the dense/SDPA path still serves the call (guard `padding_start >= self.min_seqlen`).
- `_la_postprocess_output` was **not** reordered: on this base it already crops before cast under the R01 `SGLANG_H3_LASER_CROP_BEFORE_CAST` toggle, and crop-then-cast ≡ cast-then-crop bitwise for the elementwise cast (regression test added). Handoff requirement "preserve the crop switch" satisfied by keeping the toggle rather than the PR's unconditional reorder.
- The `npu_triton_la_preprocess/postprocess` sub-path of #53105 is deferred (new-kernel rule).

### E4 residual/norm fusion — implemented (torch_npu path only), host-tested predicates, device-pending
- New `_uses_npu_fused_gate_norm2`: NPU + inference + no Cache-DiT retention + non-quantized fc1 + `MINIMAX_H3_NORM_MODULATE` enabled (numerical, quality-gated; the exact default keeps the standalone BF16 chain) + BF16 tensors + `variance_epsilon` availability.
- Fused path: gate applied in place (runs-aware), then `torch_npu.npu_add_rms_norm(residual, gated, norm2.weight, eps)`; MLP; optionally the same fusion for the MLP gated residual + **next block's norm1**, returning `(residual, next_normed)` which the model loop feeds back as `pre_norm1_hidden` so the next block only applies its AdaLN modulation.
- Cross-block gating at the model level: NPU, inference, no Spectrum, no layerwise offload, >1 plain blocks, no Cache-DiT retention, **no selective-FFN request** (base-specific addition: compaction breaks a full-row norm1 hand-off), and quantized next-projections excluded per block.
- Base deltas vs the PR: the `norm2.eps` NPU crash guard already existed here (kept, now regression-tested); the Triton fused variants (`minimax_h3_npu_triton.py`) are **deferred**, so the tuple's `is_modulated=True` arm of the PR contract is not implemented — the port returns the `npu_add_rms_norm` (norm-only) contract.
- Device-pending: `npu_add_rms_norm` rounding vs the standalone chain on CANN 9.2 (the quality gate is the acceptance mechanism), per-block and end-to-end same-seed parity.

### E5 Ulysses head-group overlap — implemented (gate + collective path), host-tested gates, device-pending
- `_NPU_LASER_PIPELINE_MIN_SEQ_LEN = 8192` (candidate threshold from the PR, to be confirmed per topology).
- `_minimax_h3_collective_pipelined_attention` accepts NPU `LASER_ATTN` when global rows ≥ threshold, heads divisible by `world_size * groups`, and not under a breakable graph; the CUDA capture-check is skipped on NPU (per the PR) so the gate does not depend on CUDA APIs. Auto picks two groups (existing `-1 → 2` default).
- The PR's additional `MINIMAX_H3_NORM_MODULATE.is_enabled(attention)` condition was **not** ported: on this base the collective path runs the plain Laser backend (no Triton fused attention), so that flag has no functional link here. Deviation recorded intentionally.
- Device validation remains the blocker: a head-group Laser call must be shown equal to those heads inside the full call on Ascend (aiter's equivalence was validated on MI355X, not transferred), plus measured exposed overlap vs sequential on 4×A3/8×A5. If unsupported, the gate returns `None` and everything stays eager — fail-closed.

### E6 row-sharded TP probe — probe only, blocked (by design)
`test/manual/minimax_h3_realtime/research/r05_comm_probe.py` reports `torch_npu` fused GEMM/collective operator availability (`npu_matmul_reduce_scatter`, `npu_matmul_all_gather`, `npu_mm_*`). No runtime path consumes them; a TP rewrite stays blocked until the operator confirms installed-CANN interfaces, rounding and scale-layout contracts. Host run reports `blocked` without crashing.

### E7 quantized pre-A2A QKV probe — probe only, blocked (by design)
Same probe covers `npu_quant_matmul` / `npu_dynamic_quant` availability. Payload-ratio arithmetic (BF16/MXFP8 ≈ 1.94, MXFP4 ≈ 3.76) remains a bandwidth bound, not a time-spread claim; re-quantization or forced low attention precision is rejected per the R02/R08 contracts.

### E8 combination — deferred
Only after E1–E4 individually pass device quality gates. The pieces compose through independent gates (`block_combined_runs` kwarg, `MINIMAX_H3_NORM_MODULATE`, pipeline-groups env), but no combined config is registered yet.

### Deferred: NPU Triton kernels
`laser_attn_triton.py` and `minimax_h3_npu_triton.py` from #53105 were not ported (R05 rule A: no new Triton kernels). They remain an optional follow-up needing `triton_ascend` and operator review; every fast path that would have used them falls back to the plain paths ported here.

## 3. Tests and environment

- `python/sglang/multimodal_gen/test/unit/test_minimax_h3_r05_port.py` — 13 CPU tests: run construction exactness (incl. PR's `(0,2,1),(2,5,0),(5,7,3)` and overflow→None), run-vs-gather `torch.equal` parity for scale/shift and gate, NPU-branch-off-CPU fallback parity, denoise-branch weakref cache emission/staleness/clearing, BCG padder drop, cat-pack vs strided-copy `torch.equal`, prepacked non-NPU `ValueError`, Laser fast-path vs segment-path `torch.equal` + dense-not-called + tail zeros, crop/cast order neutrality, `_fuses_gate_norm2` NPU canary (no `AttributeError` on missing `eps`), `_uses_npu_fused_gate_norm2` off-CPU, E5 threshold/BCG/non-NPU gates, probe host-runnability.
- Environment: CPU-only host (no NPU), torch 2.15.0+cpu, pytest from the workspace venv; `PYTHONPATH=python`. Host execution is **not** hardware validation.
- Shell syntax of the probe: `python3 r05_comm_probe.py --json | python3 -m json.tool` (covered by the probe test).

## 4. Feature-off behavior and compatibility

- Every added path is gated: runs require the kwarg + NPU + no-grad + non-BCG; the cat pack replaces only the non-kernel fallback; the Laser fast path only inside the single-segment branch above `min_seqlen`; E4 requires the quality flag and its exclusion list; E5 requires the backend + threshold + divisibility + non-BCG. With everything off, the code path is byte-for-byte the R01 behavior; the modified fallbacks are bitwise-equal to the ones they replace (tested).
- CUDA/ROCm behavior: unchanged (NPU-specific branches return early; `pack_qkv_destination_major`, indexed BF16 kernels, quality fusion untouched).
- R02 quant: `block_combined_runs` is orthogonal to precision maps; E4 excludes mxfp8 fc1/next-projections; E1's mxfp8 branches thread `runs` only into the non-quantized modulate.

## 5. Operator prerequisites and pending device gates

1. Runbook: E0 control → E1 (`MINIMAX_H3_NORM_MODULATE` lossless/high on A3, launch counters) → E2 (allocator counters, HCCL bytes) → E3 (varlen media parity + timing) → E4 (same-seed parity, then quality review at pinned pair) → E5 (group-vs-sequential exposed overlap at ≥8192-row shapes; 4×A3 first) → E8 only after individual acceptance.
2. Quality gates unchanged: human AV review, ≥10 warmed samples for promotion, three difficult prompts, raw distributions retained.
3. E6/E7: probe output on an A3/A5 operator host decides whether either track has any interface to build on; both stay documentation-level until then.
4. Not done here (honest gaps): no device runs, no A3/A5 launch adapters verified, no Cache-DiT+runs runtime check (BCG only), no head-group Laser equivalence evidence on Ascend, R04 trace integration deliberately out of scope (R05 branches from R01 per assignment).

## 6. Runbook snippet

```bash
# host tests
PYTHONPATH=python python -m pytest python/sglang/multimodal_gen/test/unit/test_minimax_h3_r05_port.py -q
# capability probe (operator host)
python3 test/manual/minimax_h3_realtime/research/r05_comm_probe.py
# E1/E4 ablation flags (existing quality gate, no new flags added)
#   exact default:        quality gate off  -> standalone BF16 chain, index_select
#   numerical opt-in:     MINIMAX_H3_NORM_MODULATE enabled -> runs + npu_add_rms_norm
```
