# DCP16 piecewise and bounded runtime buffers

Implementation is prepared locally on top of 146b0af917. Target-A3 acceptance is
pending. This note updates the earlier nightly repair plan; it does not change
or relabel the recorded results from 9c1b27f5d9.

## Piecewise DCP16

The argument guard now permits equal TP/DCP sizes on NPU. The implementation is
restricted to DeepSeek/GLM DSA, EP1, PP1, no separate attention CP, no DP attention,
LoRA or speculation. Full graph capture remains unsupported for the token-sharded
residual path; its guard is lifted only inside explicit eager attention boundaries.

DCP prefill metadata is rebuilt before each capture/warmup/replay forward, outside
Dynamo and NPUGraph recording. Capture and replay retain allocator-global write
IDs beside the rank-local write slots. Prefix lengths and request layouts are
read from the current batch, rather than captured from the dummy forward.

DCP attention, KV writes, routing metadata and layout collectives execute eagerly
between pieces. Feed-forward compute remains compilable. Dynamo graph breaks are
enabled only for this NPU DCP route. Compiler capture sizes include both the
scheduler token bucket and its TP-local row count; otherwise a 32768-token bucket
would never capture the sparse FFN's 2048-row input.

Eager outputs feed captured kernels through persistent staging buffers. Two layer
slots share these buffers across the transformer, avoiding one full set per layer
while keeping captured addresses stable. CPU tests compile three sequential FFNs
and vary lengths, prefixes, ragged requests and an empty rank; they check numerical
outputs and verify that linear compute actually reaches the compiler. These tests
cannot establish NPU replay or HCCL correctness.

The repair round now requests `graph-piecewise-dcp16` at full 1M context. It requires
DCP, capture and replay evidence. Decode graphs remain disabled. The old DCP1 short
probe remains explicitly available but is no longer the nightly graph case.

## Memory and MegaMoE stall point

The nightly watchdog stacks stopped at `start_o_proj_gather` allocating a new
output on the prefetch stream. The streamed projection now reserves two output
buffers and two transposed send buffers when its weights are registered, before
KV sizing. Forward execution copies the logical shard into a prepared send and
issues the collective without allocating. Direct and prefetch paths use layer
parity consistently. Existing stream/event ordering protects a slot until its
prior consumer has finished. Column-shard ordering is tested for FP32 and BF16.

Prefix-gather padded sends now reuse the same bounded slots as gather outputs.
Padding is zeroed on every use so a shorter prefix cannot retain the prior
request's data. This applies to prefetch, overlap-only and direct packed paths.

The KV profiler previously took the maximum of ordinary runtime slack and a
chunk-sized gather estimate. It now adds future context-sized gathers to runtime
headroom, accounts for both prefetch slots, generic reorder scratch, multiple
prefill requests, local sends and piecewise staging buckets. Resident projection
buffers are already charged by the free-memory sample and are not charged twice.
An impossible reservation reports an explicit capacity error, rather than
suggesting a memory fraction above one.

This removes demonstrated allocation churn and corrects missing memory accounting.
It does not prove that CANN's MegaMoe operator or its overlap with HCCL is sound.
The lookahead/no-lookahead pair remains in the repair round. If the new buffers
still leave a native stall, native MC2/CANN evidence is required before changing
DeepEP or the operator contract. No activation/bias arguments are discarded and no
dependency upgrade or unsupported operator fallback is included.

## Other nightly issues and validation

The already prepared scheduler budget, client timeout, activation-reporting,
HCCL controls, repetition statistics and control-drift checks remain in force.
The batch-two token budget is 32768. Full 1M requests must pass the explicit
working-set check; insufficient memory remains blocked rather than truncating
context or silently shrinking the chunk. Numerical quality stays unchecked in
the performance-only profile as requested.

Every new run additionally copies the changed runtime source files and their
SHA256 identities outside the container, alongside full logs, intermediate
snapshots, launch commands and installed versions. Copies remain every ten
minutes and after each case. No archive blocks the following experiment.

Remaining device evidence: repeated TP16/DCP16 piecewise replay across changing
lengths/batches; complete batch-two activation and memory measurements; MegaMoE
with the prepared buffers; and three matched repetitions of the 1M workload.
No target-device success or latency improvement is inferred from CPU tests.

Host validation: 117 tests and 229 subtests passed, with one pre-existing
worker-disconnect test deselected. All 15 launch recipes resolve to TP16/DCP16
and 1M context with decode graphs disabled. Fatal lint and diff checks pass.
