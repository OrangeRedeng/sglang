# Ordered A3 nightly comparisons

Current default is the repaired piecewise-only round described in
[NIGHTLY_REPAIR_20261104.md](NIGHTLY_REPAIR_20261104.md). The profile below records
the original round at commit9c1b27f5d9; its results are retained as historical evidence.

Budget: 12 hours, 30 minutes per case, 15 minutes server startup, 10 minutes
AISBench, scheduler watchdog 180 seconds. One repetition, nominal 1M tokens,
16 simultaneous client requests, 99% shared prefix, one output token.
Numerical checks are skipped as requested. Every completed measurement remains
`diagnostic-unqualified`; optimization activation is still required.

Order: legacy 40s recipe, historical 24s recipe, full and piecewise prefill graphs,
TP routed scaling, prequantization, shared fusion, shared activation, MoE combined,
metadata reuse, metadata batch two, packed metadata batch two, FP16 indexer ablation,
auto stream thresholds, auto HCCL, stream/HCCL combined, repeated 24s recipe,
legacy MegaMoE 34s recipe, historical recipe plus MegaMoE, then MegaMoE plus metadata.
INT8 indexer and multistream already run in the historical control; the FP16
ablation isolates indexer quantization. Known inactive tp-finalize and incompatible
W4A8 FuseEP modes 1/2 are omitted from timed launches.

The legacy controls restore chunk 16384, memory fraction .88, local KV rows 74752,
max-running-requests 64, prefill-max-requests 1, separate shared experts and eager
DCP16. Legacy MegaMoE adds EP16, mode3, capacity1024 and DeepEP buffer768.
Their historical 39.50s/34.77s results used native exact-token measurement; the
new comparisons all use AISBench. This is recipe reproduction, not a promise of
identical historical timing. Contemporary source, inherited environment, and the
measurement client can differ. The new MegaMoE comparisons use the historical
24s recipe with chunk32768, capacity2048, INT8 KV/indexer, residual token sharding
and overlap features. A result below24s is a target, not a validated capability.

The graph cases explicitly set prefill graph backends and bounded capture buckets;
capture and replay evidence are both required. Batch-two variants require an
actual two-request prefill event, and the packed variant requires packed batch-two
evidence. Unsupported paths and missing activation remain failed/incomplete.

`run_a3_nightly.py` runs on the host, never inside Docker. It snapshots source
files, hashes, dirty diff, commit, installed versions/paths and runtime environment.
Each case saves its exact host command, resolved server launch, workload, full
AISBench log, raw results, activation and cold-prefix metrics. Host `progress.json`
and `SUMMARY.md` update every15 seconds. The console bar counts completed cases,
not estimated token completion. Every10 minutes and at case completion it copies
full container logs to the host, captures npu-smi/processes, and retains a dated
32KiB server-log tail for the intermediate state. Native CANN dumps are exported
by the worker on cleanup. No automatic compression delays the following case.
Zero-byte files are removed after completion and listed in a manifest.

The final three MegaMoE attempts have reserved budget; earlier optional cases are
skipped when necessary. After a timeout only processes belonging to that runner
are stopped. A surviving server stops the nightly sequence to prevent overlapping
allocations. A native device hang may require manual recovery and cannot be
reliably repaired by a host timeout. `--stop-previous` sends SIGINT to a verified
run_bde_overnight.py lock owner before starting; it refuses unrelated lock owners.

Host checks: focused runner/qualification suites, resolved launch plans for every
case, fatal lint, syntax and diff checks. A3 execution remains user-run. Exit2
means all measurements completed with unchecked quality; exit1 means at least one
case failed, timed out or was incomplete.
