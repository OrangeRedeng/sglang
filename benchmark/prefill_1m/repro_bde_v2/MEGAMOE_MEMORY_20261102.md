# MegaMoE memory retest and baseline reproduction

The f8b4f71991 run reached the MegaMoE operator but failed allocating 18.02 GiB:
44.12 GiB active, 11.14 GiB free, 46.69 GiB reserved on a 61.27 GiB NPU. The
allocator traceback is inside `cann_ops_transformer.npu_mega_moe`, after allocating
81,920 local KV slots (13.44 GB) per rank. The later server exit -9 is secondary.
This is different from the earlier HCCL buffer-capacity failure.

## Changes

The pilot uses dispatch capacity 1024 instead of 4096, with chunk 1024 on both
baseline and MegaMoE. Lowering dispatch capacity is intended to reduce the
operator's capacity-dependent workspace; the actual workspace reduction requires
A3 verification. HCCL remains 768 MB. Weight packing, compensation, quantization
and parity tolerances are unchanged. This is a memory mitigation experiment,
not a verified numerical or workspace fix.

Capacity preflight now rounds every outstanding suffix, including its output,
to the resolved DCP allocation page. The previous nominal-99% formula undercounted
these pages and was unsafe when trimming the KV cache. Reports retain the
calculated working set and actual exposed capacity.

All profiles use 16 requests/concurrency 16, one repetition, prefix99 only,
one output token, eager TP16/DCP16 and full GLM5.2 W4A8 weights. Prefix preparation
is excluded from TTFT; queueing is included. TPOT is undefined.

| Launcher mode | Input tokens | Local KV slots | Aggregate KV slots | Chunk / MegaMoE dispatch | Variants |
|---|---:|---:|---:|---|---|
| baseline-repro | 1,000,000 | 81,920 | 1,310,720 | 16,384 / unused | dcp-baseline |
| megamoe-500k | 500,000 | 39,936 | 638,976 | 1,024 / 1,024 | moe-baseline, megamoe |
| megamoe-measure | 1,000,000 | 74,752 | 1,196,032 | 1,024 / 1,024 | moe-baseline, megamoe |

At DCP16 the effective page is 2048 tokens. For 1M, the reusable prefix is 989184;
16 suffixes each need 12288 slots. Required capacity including 4096 headroom is
1189888, so a KV cap of exactly 1M would be too small. At 500k, reusable prefix is
493568 and each suffix needs 8192 slots, giving a requirement of 628736. Neither
profile shortens the requested prompt to fit memory.

Relative to the failed run, the 1M KV allocation is about 8.75% smaller (roughly
1.18 GiB saved per rank), while the 500k KV allocation is about 51.25% smaller
(roughly 6.89 GiB saved). These estimates scale the observed 13.44 GB allocation;
new logs must confirm actual memory and operator workspace.

`baseline-repro` restores chunk 16384 and the ordinary shared-expert policy,
removing the chunk-4096/separate-shared changes from the 79.54 s run. It targets
the earlier 24.589 s workload, but cannot promise that latency: current source,
client and hardware conditions differ from the historical measurement. Its result
must not be used as a matched speedup denominator for chunk-1024 MegaMoE. The
`moe-baseline` inside each MegaMoE profile is the matched comparison.

## Execution order

Do not disturb the baseline experiment already launched by the user. Run the
500k pilot after the current runner releases its lock. Run the 1M pilot only
after checking that 500k completes and its memory logs show useful headroom.
Do not run profiles concurrently or delete the runner lock.

After installing these updated source files, run on the A3 host:

```bash
REPO=/home/swx1199799/megamoe_test/sglang
docker cp "sglang-a3:$REPO/benchmark/prefill_1m/start_a3_retest.sh" /tmp/start_a3_retest.sh
screen -dmS "a3-megamoe-500k-$(date +%Y%m%d-%H%M%S)" bash /tmp/start_a3_retest.sh megamoe-500k
```

Use `baseline-repro` for a separate historical-baseline reproduction, or
`megamoe-measure` for the trimmed 1M pilot. Each profile has a two-hour budget;
smaller chunks can increase warmup/priming time substantially. The 500k run is
intended to be the first memory test, not proof that the 1M path will fit.

MegaMoE remains failed/unqualified if the existing parity test fails. Diagnostic
TTFT may then be recorded as `unqualified-parity`; it does not establish an
accepted optimization. Preserve SUMMARY.md, full raw server logs, runtime
identities, resolved arguments, sample token counts, peak memory evidence and
parity.json before deciding the next step.
