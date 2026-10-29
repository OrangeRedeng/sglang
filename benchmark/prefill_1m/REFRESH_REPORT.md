# Main refresh and A3 performance preservation — 2026-10-29

**Verdict: host regressions pass; required legacy features remain coverage-blocked. No target-device or performance acceptance is claimed.**

## Provenance and merge state

- Worktree: `/home/orangeredeng/code/sglang/.codex-artifacts/worktrees/a3-int-glm53`; branch `a3_910c_int_glm53`.
- Starting HEAD: `5df5c1384594688f5e29046ef845b9fdb7d9c8f6`.
- Fetched main and pending merge parent: `1ab008f5b090e6595479ba57934e9211ab083565`.
- Previous main merge: `5814d13716`, through `b74fd392e1`; this refresh incorporates the next 146 main commits.
- Executed `git fetch origin main` and `git merge --no-commit --no-ff origin/main`. Seven conflicts were resolved. No commit, rebase, push or force-push was performed.
- Main adds Ascend-Ops transport/routing (#46282) and block-FP8 MoE (#46212). Those interfaces, standard hooks and block-FP8 clamp precision are retained. The existing native dispatcher still owns branch-specific INT reuse/shared/GMM2 finalization. Both formats retain checker/runner registrations; upstream GGUF final-gather state is honored by the native path too.
- The source workspace remains at `b9bb06c92509734c9d8fc6d032f7f752df9e092b` with unfinished merge `de26bb16a972d93f354218a29dbe177bce5655a4`. All 4464 initially recorded tracked-file hashes, HEAD and MERGE_HEAD were checked unchanged.
- The BDE archive's HEAD `5d6052683c` is already an ancestor. All eight Alexander Zuev commits remain in history. Its 70 commits absent from current upstream main are preserved through that ancestry. Archive dirty Python changes are timestamp instrumentation; they were not imported as optimizations.

| Source branch | Checked tip | History |
|---|---|---|
| `base_bde_combined` | `5d6052683c7598ec2477db4793ef2347153f330d` | Ancestor of starting integration HEAD |
| `glm52-a3-megamoe` | `b9bb06c92509734c9d8fc6d032f7f752df9e092b` | Ancestor of starting integration HEAD |
| `fork_glm52_a3_megamoe` | `b9bb06c92509734c9d8fc6d032f7f752df9e092b` | Ancestor of starting integration HEAD |
| `a5_cleanup_main` | `051fc667de7a000e02b4e7152f79d7617c23ff0c` | Ancestor of starting integration HEAD |

History preservation is separate from runtime preservation. The matrix below records the latter; missing connections are not treated as successful replacements.

## Preservation matrix

| Source feature | Implementation status / consumer | Test and acceptance status |
|---|---|---|
| Zuev `e4c73f3575`: INT8 LightningIndexer | Retained; INT8 storage avoids MXFP metadata planning | `npu-int8-indexer`, `serve-indexer-int8`; device rerun pending |
| Zuev `22ad5f3923`: packed INT8 SFA KV | Retained; pool row format and BF16 dequantized DCP read | `npu-sfa-int8-roundtrip`, `serve-sfa-int8`; memory/device rerun pending |
| Zuev `2bf9c8996f`, `f0c7665e04`, `4f3abb1a07`: chat pool/memo/raw | Retained in chat serving/tokenizer paths | `unit-host-encode`, separate chat serving cases; full-model activation/performance pending |
| Tamir C1 + Zuev D2 `81df8073e7`: packed prefix read | Restored in DSA prepare/core and gathered sparse backend; page-owned rank-major mapping, preserved sentinels, short-prefix causal fallback | `test_dcp_prefix_gather.py` checks exact row access and attention parity; `serve-dcp-packed-read` requires activation plus logprob parity |
| Tamir C3: next-layer gather prefetch | Restored by `DcpGatherPipeline`; side-stream event waits, retained sends, per-piece scratch, bounded lookahead and actual pool layer bound | Deferred-collective CPU parity passes; `npu-dcp-prefix-hccl` and `serve-dcp-gather-prefetch` prepared, NPU pending |
| Zuev D1 overlap part of `b58485497b` | Restored before QKV; wait/own-KV assembly before sparse attention | Multi-request/ragged/empty-owner/multi-piece CPU parity passes; `serve-dcp-sfa-overlap` and HCCL test prepared |
| Tamir owner-only writes, physical page rows, query-indexer sharding | Retained in current pool/layout/indexer implementations | Existing layout/owner/gather/indexer unit selections pass; full distributed/model acceptance pending |
| Tamir DSA query token redistribution and multi-request lift | Newer reviewed query-sharding path retained; does not shorten cumulative lengths | Shard-plan/indexer unit selections pass; `serve-dsa-token-shard[-dcp]` requires target rerun; not equivalent to residual sharding |
| Zuev D1v3 residual token-sharded rebuild | **Disconnected/incompatible** with current attention/boundary path; preserved historical preparation is not execution | Explicit flag rejected before loading; `serve-dcp-legacy-token-shard` reports `coverage_blocked` |
| Zuev D2v3 `5d6052683c`: streamed vs resident full o_proj | **Disconnected**, depends on residual token sharding; not replaced by query sharding | Explicit flag rejected before loading; `serve-dcp-stream-o-proj` reports `coverage_blocked`; no memory-parity claim |
| Legacy full-width Q, Q-A2A/decode preissue, local top-k | **Disconnected**; old flags have no compatible current consumer | Explicit enables rejected; `serve-dcp-legacy-q-variants` reports `coverage_blocked` |
| A5/native INT routed scale, GMM2 finalize, shared finalization, INT8 reuse, shared activation | Retained across the dispatcher refresh | Individual `tp-routed-scale`, `tp-finalize`, `tp-shared`, `tp-prequant`, `tp-shared-activation` scenarios; CPU contracts pass, device parity pending |
| Multistream, auto thresholds and HCCL sizing | Retained | Separate scenarios plus host resource/autotune/HCCL suites; actual overlap/throughput pending |
| FuseEP 1/2 and MegaMoE 3 | Retained with ModelSlim preparation and dependency preflight | Independent EP layouts/scenarios; actual operator compatibility, compensation parity and serving pending |
| Streaming online W8A8/W4A4 and target overrides | Retained; fixed MLA logical weight view and nullable shared-expert configuration | Completion/reload/lifetime host regressions pass; full-model NPU peak allocation/reservation and remaining floating bytes not measured |
| Prefill full/piecewise graphs | Retained independently; DCP remains excluded | Host graph tests pass; capture/replay and changing-length device validation pending |
| A5-only MXFP and Gemma-specific experiments | Intentionally excluded from this INT-only A3 acceptance matrix | No acceptance claim; shared upstream code is not removed solely for this scope |

Restored packed reads and prefetch are opt-in while target validation is pending. The archived launch reference now selects reviewed query sharding and leaves these opt-ins off. It no longer silently requests unsupported residual/o_proj features. Legacy full-width-Q/local-top-k/streamed-o_proj defaults are off, and explicit enables fail early.

## Bug repairs

1. Evidence parsing recognizes the standalone event marker; the ordinary environment log `get env SGLANG_NPU_FEATURE_EVIDENCE = True` no longer causes JSON decoding failure. Malformed genuine events still fail validation.
2. Dense online methods expose a logical `[out, in]` INT8-dequantized projection for MLA splitting. The converted matmul layout `[in, out]` is never guessed from shape; temporary floating data is limited to the local projection.
3. Shared-expert packed-format checks accept `quant_config=None`.
4. GMM2 device reference uses production BF16 scales for BF16 per-token A8W8 output, with the same represented scales in both comparisons.
5. INT8 routing reuse is checked against the same prequantized payload and its source scales; separate quantizers are not required to produce bitwise-identical integers. Empty inputs bypass unsupported zero-row launches.
6. Shared-activation contract tests explicitly exercise disabled/enabled flags. DCP defaults select the Ascend attention backend without inventing a CUDA DSA backend for NPU.
7. Failed references are cached so dependent scenarios report the first blocker instead of reloading repeatedly. Response/JUnit report sections are summarized; sample-level benchmark results remain available.
8. Small as well as long stages honor `--benchmark-variants`; explicit `--cases` selection remains supported. Selecting a DCP variant aligns the small baseline to the same DCP size. Unsupported scenarios prevent an all-validated verdict and return a distinct coverage blocker.

Historical failure excerpts reproduced on the pre-fix implementation:

```text
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
AttributeError: 'NoneType' object has no attribute 'get_name'
RuntimeError: unflatten: Provided sizes [-1, 448] don't multiply up to the size of dim 0 (512)
```

## Validation performed

Host environment: Python 3.14, existing repository venv torch `2.14.1+cpu`; no dependency installation or upgrade. The system Python lacks torch. The venv has no torch_npu, `npu-smi` is unavailable, and neither default checkpoint directory exists here.

| Check | Result |
|---|---|
| Full runner UNIT_SUITES selection, actual pytest | **265 passed, 2 skipped**, 2007 subtests, 87.29 s; skips remain incomplete evidence |
| Additional upstream Ascend-Ops / native TP routing / graph compatibility selection | **27 passed**, 35 subtests |
| Final native/Ascend-Ops compatibility after the last merge fix | **35 passed**, 18 subtests |
| Final loading/null-config/native-final-gather regression selection | **13 passed**; four NPU device tests skipped on this host |
| Archived numerical overlap/prefetch properties | **11 passed**, 60 subtests; obsolete source-string wiring assertions replaced by executable gather tests |
| Host syntax/shell/dry-run/workloads/suite-command/fusion-import/boundary cases | All seven passed |
| Red/green parser, null-config and MLA-layout reproduction | Old failures reproduced; corrected parser/layout and pytest regressions pass |
| Syntax across all changed Python files | 765 parsed successfully |
| Ruff undefined-name/unbound-local checks | Passed for NPU runtime, native/Ascend-Ops dispatch, runner and benchmark code |
| Small / DCP case plans | Dry-run passed; explicit small variants select only requested cases |
| Source workspace preservation | All recorded tracked contents and both source refs unchanged |
| Distributed HCCL / NPU correctness / graphs / full model / benchmarks | **Dependency/environment blocked; not executed** |

The main merge touched a CRLF file whose three new comment lines failed whitespace checking; only those three line endings were normalized. No model behavior was changed there.

Exact full unit invocation:

```bash
PYTHONPATH=python /home/orangeredeng/code/sglang/.venv/bin/python -m pytest -q -p no:cacheprovider test/registered/unit/npu/quantization/test_glm_a3_regressions.py test/registered/unit/managers/test_scheduler_stage_latency.py test/registered/unit/npu/quantization/test_fused_norm_quant_contracts.py test/registered/unit/layers/moe/test_ascend_tp_optimizations.py test/registered/unit/npu/quantization/test_tp_moe_routing_weights.py test/registered/unit/npu/quantization/test_tp_moe_shared_weight_views.py test/registered/unit/layers/moe/test_npu_topk_l1_norm.py test/registered/dcp/test_dcp_block_tables.py test/registered/dcp/test_dcp_owner_write.py test/registered/dcp/test_dcp_topk_remap.py test/registered/dcp/test_dsa_cp_shard_plan.py test/registered/unit/npu/attention/test_dcp_prefix_gather.py test/registered/dcp/test_dcp_sfa_gather_overlap.py test/registered/dcp/test_dcp_extend_gather_prefetch.py test/registered/dcp/test_dcp_extend_gather_plan.py test/registered/dcp/test_dcp_sparse_attention_parity.py test/registered/attention/unittests/dsa/test_dsa_npu_indexer_query_shard.py test/registered/attention/unittests/dsa/test_dsa_token_shard_plan.py test/registered/dcp/test_dcp_quant_indexer_shard_plan.py test/registered/unit/npu/mem_cache/test_npu_dsa_index_k_scale_pool.py test/registered/unit/mem_cache/test_dcp_index_buf_budget.py test/registered/unit/model_executor/test_cuda_graph_config_backends.py test/registered/unit/model_executor/test_npu_piecewise_backend.py test/registered/unit/model_executor/test_prefill_cuda_graph_runner.py test/registered/chat_encode/test_chat_encode_pool.py test/registered/unit/npu/test_npu_autotune.py test/registered/unit/npu/test_npu_hccl_config.py test/registered/unit/npu/test_npu_resource_scheduling.py --disable-warnings --maxfail=3
```

Additional checks used:

```bash
python3 benchmark/prefill_1m/test_glm53_features.py --stage host --cases 'host-syntax,host-shell,host-1m-dry-run,host-workloads,host-suite-commands,host-fusion-imports,host-boundary' --output /tmp/a3-refresh-host.md
python3 benchmark/prefill_1m/test_glm53_features.py --self-check-suite-commands
python3 benchmark/prefill_1m/test_glm53_features.py --self-check-workloads
PYTHONPATH=python /home/orangeredeng/code/sglang/.venv/bin/python /tmp/a3-refresh-red-check.py
git diff --check HEAD
```

## Target execution and remaining acceptance

Run from the integration worktree using the target machine's existing environment. Every invocation writes one new Markdown file; output paths must not already exist.

```bash
python3 benchmark/prefill_1m/test_glm53_features.py --stage all \
  --model-path /home/weights/GLM5.2-w4a8-int/ \
  --bf16-model-path /home/weights/GLM-5.2 --tp 16 --dcp 8 --output refreshed-features.md

# Distributed prefix test separately; the feature runner also includes it.
PYTHONPATH=python python3 -m torch.distributed.run --standalone --nproc_per_node=16 \
  test/registered/unit/npu/attention/test_dcp_prefix_gather_distributed.py

# Run measurements only after each variant passes activation and parity.
python3 benchmark/prefill_1m/test_glm53_features.py --stage small \
  --benchmark-variants baseline,dcp-packed-read,dcp-gather-prefetch,dcp-sfa-overlap,megamoe \
  --tp 16 --dcp 8 --output refreshed-small.md
python3 benchmark/prefill_1m/test_glm53_features.py --stage long \
  --benchmark-variants baseline,dcp-packed-read,dcp-gather-prefetch,megamoe \
  --tp 16 --dcp 8 --output refreshed-long.md
```

Small measurements use `python -m sglang.benchmark.serving`; the long driver uses SGLang's streaming API for first-token samples and observed cache reuse. Neither uses AISBench. FuseEP/MegaMoE use EP layouts required by their transport; compare their performance only with a reference using that same EP geometry, not a TP-only baseline. Workload lengths/concurrency/repeats remain explicit; capacity limits must produce blockers rather than smaller requests.

Required remaining work: port the legacy residual/boundary/o_proj execution before claiming those features preserved; establish real HCCL ordering and numerical parity for restored gathers; run full-model streaming-load memory measurements; validate all individual INT/MegaMoE/graph branches on A3, then collect the performance tables. Peak NPU allocated/reserved memory, TTFT, ITL, end-to-end latency and throughput are **not measured by this host refresh**.

The pending merge is reviewable and uncommitted. Do not publish an all-validated verdict while coverage/device blockers remain.
