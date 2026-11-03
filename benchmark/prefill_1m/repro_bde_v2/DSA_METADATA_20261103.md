# Opt-in DSA metadata preparation and prefill batch two

`SGLANG_NPU_DSA_METADATA_REUSE=1` enables the separate metadata feature. Default is off. It adapts the Gemma branch's metadata-reuse/page-gather principle to eager GLM DSA, preserving the virtual indexer and rank-local DCP page-number conventions.

The fast path validates CPU lengths against prefix + extend lengths, then reads only page starts from the request/token map once. It avoids duplicate full-token-map gathers and the device reduction used to obtain maximum length. No device page table is reused across requests: each forward rebuilds it from current request slots, lengths and allocation. Non-DSA, hybrid SWA, graph, draft/verify, absent CPU mirrors and inconsistent lengths use the original path. Default launch flags are unchanged.

## Optional prefill batch two

Set both `--max-running-requests 2` and `--prefill-max-requests 2`. These are caps, not a minimum batch size. Long cached-prefix DSA already supports multiple requests when every request satisfies the existing crop-free/top-k eligibility bound. The packed rank-major KV layout is still single-request only; a two-request prefill uses the existing request-contiguous permutation/gather path. This preserves per-request KV boundaries and does not implement a new multi-request packed layout. The extra gather/permutation cost can offset batching gains.

The global chunk remains 32768. MegaMoE's per-rank dispatch capacity stays 2048 at TP16; adding a second request does not double the global chunk. The runner checks capacity using the actual resolved running cap and records resolved caps in each case's workload/capacity artifacts. A 1M independent-prompt batch two may need more KV capacity; the provided workload uses a 99% shared prefix.

## A3 experiment

After updating `glm52-a3-optimizations`, copy `benchmark/prefill_1m/start_a3_retest.sh` from the container to the host as before. Run:

```bash
bash /tmp/start_a3_retest.sh metadata-compare
```

This runs three sequential 1M/C16, one-repetition cases: unchanged historical baseline, baseline + metadata, baseline + metadata with prefill/running caps two. The context limit, INT8 KV/indexer, memory fraction .85 and KV rows 67584 remain matched. Budget is three hours.

For MegaMoE combinations:

```bash
bash /tmp/start_a3_retest.sh metadata-compare   --variants baseline,megamoe,dsa-metadata-megamoe,dsa-metadata-megamoe-batch2
```

Performance-only mode keeps numerical checks disabled per request and labels results `diagnostic-unqualified`. It requires `dsa-metadata-reuse` evidence for the new feature, `dcp-prefill-batch-2` for an actual two-request prefill, and MegaMoE/local-EP evidence for MegaMoE variants. A cap of two without a two-request batch is incomplete coverage, not an optimization pass. Original `megamoe-compare` still runs only baseline and MegaMoE.

Inspect `results/SUMMARY.md`, per-case raw AISBench logs, `activation.json`, `resolved-capacity.json`, `workload.json` and full container server logs. Host regression tests cover page-table equivalence for ragged/empty rows at DCP1/2/16, default-off and fallback behavior, two-request packed refusal, and launch/activation wiring. No A3 performance or numerical correctness claim is made until target execution.
