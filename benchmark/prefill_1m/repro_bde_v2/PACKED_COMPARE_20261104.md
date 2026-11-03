# A3 comparison: 20261104-032840

Input: `retest-packed-batch2-compare-20261104-032840`, deployed commit
`b7b88767e454e255fd26390b4c85e9079bbc254b`.

Despite the directory name, the resolved variants were `baseline,megamoe`.
No packed batch-two or metadata-reuse variant was measured.

The historical baseline reproduced: TTFT mean 24.2932548 s, P90 41.1296346 s,
E2E run 45.5938535 s. Workload: 16 requests/concurrency 16, one output token,
mean input length 1000012.125, 99% shared prefix. Numerical quality was unchecked.

MegaMoE made progress until about 400K prefix tokens, then the native MegaMoe
kernel timed out. The server recorded `retCode=0x25 [aicore timeout]` and MC2
dumps under `extra-info/data-dump/`. No OOM was reported. The later indexer
host-copy exception and HCCL watchdog errors observed the outstanding native
failure; they do not establish that the copy or HCCL caused it. AISBench had
no successful requests, so no MegaMoE TTFT can be reported.

## Repairs

- Remove synchronous host-to-device construction of indexer shard length tensors.
  Cache full-head shard metadata across layers, refreshing when request boundaries,
  padded grid, or device change. Preserve per-request indexer KV lengths.
- Stop AISBench when its serving worker exits, rather than waiting for client retries.
- Attribute native timeout/OOM evidence to server execution while retaining the
  original client phase and exception, plus a full server-log snapshot.
- Collect newly written CANN/MC2 files from `extra-info/data-dump/` into each
  variant's `native-dumps/` before exporting container artifacts.

The metadata synchronization defect is repaired. The native MegaMoe timeout
is not yet proven resolved: repeat on A3 and inspect the captured native dumps
if it recurs. No kernel workaround, dependency upgrade, context reduction,
chunk reduction, or baseline flag change was introduced.

## Retest

On the A3 host, pull `glm52-a3-optimizations`, copy `start_a3_retest.sh` from
the container, and run its `megamoe-compare` mode. This uses the same 1M/C16/R16
workload, chunk 32768, one repetition, and two serving launches. Use
`packed-batch2-compare` separately to actually exercise the batch-two variants.
Performance-only results remain diagnostic and numerically unqualified.

Host validation: 41 focused tests and 118 subtests passed; the pre-existing
worker-disconnect test was deselected. Fatal lint and `git diff --check` passed.
Target A3 execution remains pending.
