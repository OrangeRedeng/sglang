# Focused A3 final round

Superseded for the next launch by [RETEST_REPAIR_20261105.md](RETEST_REPAIR_20261105.md).

Source evidence: `retest-nightly-20261105-004431`, commit d295528925.
A3 x16, GLM5.2 ModelSlim W4A8, CANN9.1.0, torch2.10.0+cpu,
torch_npu2.10.0.post6, sgl-kernel-npu2026.9.0, DeepEP00db6cd6,
cann_ops_transformer1.0.0. Numerical qualification remains unchecked.

## What the completed round establishes

| Case | Cached TTFT mean / P90 seconds | Cold-prefix TTFT seconds |
|---|---|---|
| Baseline first | 23.977 / 40.722 | 181.452 |
| Baseline last | 23.878 / 40.623 | See raw metrics |
| Separate shared stream | 24.801 / 41.487 | 595.541 |
| Fused MoE combination | 23.669 / 40.331 | See raw metrics |
| INT8 gate + fused norm | 23.629 / 40.176 | 215.170 |
| INT8 norm + shared stream | 34.377 / 54.004 | 681.284 |
| Shared stream + prequant/activation | 40.141 / 65.522 | 679.539 |

Baseline drift is about0.42%. The candidate gains are only about1–1.5% from
one fresh-server sample, not demonstrated production speedups. Shared stream's
cold-prefill penalty is much larger than its cached-only regression. The code
launches the shared GEMMs before routing for every32768-token cold chunk;
resource contention/launch scheduling is a hypothesis, not a profiler finding.
The bounded stream case changes only its size eligibility: cold chunks remain
serial; cached suffixes around10832 padded tokens can overlap. It is a diagnostic
ablation rather than an assumed winner. The previous stream combinations are
excluded from best combinations.

## Fixes and limits

- Piecewise graph output lists contained integer40. The old weak-reference
  helper called native `_weak_ref_tensor` on every tuple/list element. It now
  traverses the output pytree and weak-references tensor leaves only, preserving
  symbolic integers, None, dictionaries and flags. Capture/replay still require
  A3 validation; the earlier replay marker did not imply complete qualification.
- MegaMoE watchdog stacks stopped in packed-cache decoding's allocating FP32
  conversion on the DCP gather stream after11 cold-prefill chunks. This locates
  a device synchronization point, not a proven failing DeepEP kernel. DCP packed
  readers now prepare two bounded scratch/output slots at pool initialization,
  use index_select(out=), copy conversion and in-place FP32 scaling, and keep
  the original BF16 rounding and rope layout. Memory reservation includes new
  raw/FP32/index scratch before KV sizing. No dependency upgrade or precision
  fallback is included. If the native watchdog persists, the installed native
  stack remains an unresolved prerequisite; no pass is manufactured.
- Token-sharded logits derive last-row indices from existing host lengths with
  integer prefix sums. The old implicit-device tensor construction plus tolist
  could cause an avoidable D2H transfer under Ascend transfer/default-device
  routing. The new `dsa-logits-host-metadata` marker identifies this path; it is
  not a total runtime D2H count. Existing per-forward DCP cumulative tensors
  already avoid repeated per-layer H2D copies.
- Page-table reuse is tested separately in cached-prefix attention; it reads
  page starts once per forward instead of rebuilding full token maps per layer.
  This reuses metadata, not attention outputs. All measured requests still have
  independent suffixes and one newly generated token.
- KV preflight now rounds the shared prefix and each live suffix to virtual
  DCP pages. Batch2 requires1017856 virtual token slots including4096 headroom,
  so those matched cases request63616 local KV rows, instead of63488. Context
  remains1048576 and chunk32768. Both batch2 variants use identical capacity.
- Watchdog and graph-output type errors are retained in failure attribution,
  rather than showing only the resulting worker exit code.

## Eight-hour plan

Run `run_a3_nightly.py --round final --hours 8 --case-minutes 40 --repeats 1`.
Every case uses a fresh server and one AISBench invocation with separate cold
prefix preparation and cached results. There are16 requests at concurrency16,
one output token,99% shared prefix, TP16/DCP16. MegaMoE uses EP16. No graphs
other than piecewise prefill are requested; decode graphs stay disabled.

1. Baseline control,1M.
2. Repaired DCP16 piecewise graph,1M.
3. Repaired/no-lookahead MegaMoE,500K input with the same1M context allocation.
4. MegaMoE1M, only if case3 actually completes with activation evidence.
5. Size-bounded shared stream,1M.
6. INT8 norm + routed/shared/activation fusions,1M; norm replaces separate prequant.
7. Case6 + cached-attention page-table reuse,1M.
8. Case7 + batch2, normal packed-read selection,1M.
9. Matched batch2 + explicit two-request packed read,1M.
10. Baseline control,1M.

Ten40-minute ceilings total6h40m; startup, warmup, measurement and export are
inside each case budget, with the overall8-hour deadline still enforced. The
500K result is a failure-isolation gate, not comparable with1M TTFT. If it fails,
case4 is explicitly prerequisite-blocked. Full logs remain outside Docker,
exported initially then every600s; progress is printed every15s.

## Feature coverage audit and exclusions

- Already measured individually: routed scale, prequant, shared-output fusion,
  shared activation and their combination; retest only in the new norm/metadata
  combinations and baseline controls.
- INT8 indexer, query sharding, full-width Q, residual sharding, streamed o_proj,
  packed/gather/overlap/decode-preissue components: inherited BDE control paths;
  existing activation checks remain. No duplicate standalone cases.
- Poorly covered: piecewise replay, cached-attention metadata, batch2 and packed
  batch2, stream size eligibility, new best combinations: selected above.
- MegaMoE and MegaMoE+gate both watchdog on the same cache-read path. Test the
  repaired native path first; exclude another gate-conversion case until it works.
- GMM2 finalizer and FuseEP1/2 require W8A8 expert storage: blocked for the given
  W4A8 checkpoint. They need an explicitly supplied compatible checkpoint.
- BF16-based online full-model/QKV/dense conversion needs the relevant full-weight
  checkpoint and separate accuracy acceptance. Do not spend the eight-hour INT
  performance budget on those previously blocked loading cases.
- Full graphs are excluded by the piecewise-only preference. MXFP is excluded by
  A3 INT-only scope. Legacy40s/34s recipes, host chat pools/memo, automatic HCCL
  tuning and previously tested FP16-indexer ablations are not part of this round.

Local validation:83 focused CPU tests and178 subtests passed, including mixed
scalar/tensor graph outputs, exact packed INT8/rope/scale reconstruction, empty
reads, stable two-slot storage with hot allocations/conversions forbidden,
stream row bounds, batch2 capacity and final-plan coverage/budget. One existing
socket-dependent test was excluded by sandbox restrictions. Launch construction,
fatal lint and whitespace checks passed. Native A3 replay, watchdog resolution,
quality and performance remain pending.
A further ragged-logit regression passed with tensor construction and tolist
forbidden, confirming correct request-last-row selection from host lengths.
