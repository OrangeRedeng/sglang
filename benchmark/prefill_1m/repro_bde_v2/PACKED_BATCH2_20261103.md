# Experimental two-request packed DCP

Enable `SGLANG_NPU_ENABLE_DCP_PACKED_READ_BATCH2=1` together with the existing packed-read flag and `--max-running-requests 2 --prefill-max-requests 2`. The new flag defaults off. Single-request packed reads and the existing general fallback remain unchanged when it is off.

## What changes

The previous batch-two fallback gathered a combined rank-major prefix and permuted it into logical request order. The new plan gathers each request directly into its own contiguous segment:

```
[request 0: rank-ordered padded prefix | request 0 new KV]
[request 1: rank-ordered padded prefix | request 1 new KV]
```

This removes the full-context index_select and its per-piece scratch outputs. Two keys and two nonempty request prefixes require four all-gathers per layer. Each rank uses the same collective order even when it owns no prefix or query rows. Query ranks can span either request or contain only padding.

Top-k indices remain relative to their request's segment. Remapping uses the actual per-request local query lengths; trailing query padding stays -1. The backend receives cumulative ends of the packed segments, including unused prefix padding. Because the sparse operator's causal crop is disabled, every request must satisfy the existing prefix/top-k eligibility bound. Short-prefix, unknown-top-k and unsupported batch-size cases retain the general path. Cropping eligibility is reset for each forward.

Packed length buffers refresh when either segment boundary changes, even if the total row count is unchanged. Preparation remains ahead of gather issue and uses device fills rather than blocking host tensor copies.

Both requests still gather their own prefix. Shared-prefix identity checking and one-copy prefix reuse are separate work. Memory and TTFT improvement must be measured on A3; this change does not promise a recovery to 24.59 seconds.

## Compare on A3

After updating `glm52-a3-optimizations`, copy the host launcher from the container and run:

```bash
bash /tmp/start_a3_retest.sh packed-batch2-compare
```

Four sequential cases: historical baseline, metadata-only batch one, metadata with the old batch-two fallback, and metadata with the new packed batch two. All retain nominal 1M input, 16 client requests, one output token, 99% shared prefix, one repetition, global chunk 32768, memory fraction .85 and local KV rows 67584. The batch-two cases have running/prefill caps two. This isolates packing from the metadata feature and batching policy. The profile has a three-hour budget.

For MegaMoE:

```bash
bash /tmp/start_a3_retest.sh packed-batch2-compare   --variants baseline,megamoe,dsa-metadata-megamoe-batch2,dsa-metadata-packed-megamoe-batch2
```

The timed run must record both `dcp-prefill-batch-2` and `dcp-packed-read-batch2` for the new case. A cap of two or activation of ordinary packed batch one is insufficient. Full launch settings, activation, capacities, raw AISBench logs, results and server logs remain saved. Numerical checks stay disabled in this performance-only profile as requested, so results are labelled diagnostic-unqualified.

For numerical qualification, run `test_glm53_features.py --stage serve --reference-recipe historical-bde --cases serve-dsa-metadata-packed-batch2` with the full model path and TP16/DCP16. Batch-two scenarios now send two distinct ragged suffixes after priming the same prefix; both responses are retained individually for comparison with the baseline reference. The reference receives the same probe but can process its requests serially. Activation is checked separately. Existing tolerances are unchanged.

## Validation boundary

CPU regressions check actual distributed sends, independent request segments, rank/page ownership, empty/padded query ranks, negative indices, native backend argument construction and numerical equality against a logical sparse-attention reference. They explicitly forbid index_select in the packed gather and test same-total/different-partition metadata refresh. These tests simulate collectives and the NPU operator; they do not establish CANN/HCCL correctness or full-model A3 parity. Target validation is pending.
