# A3 nightly analysis and repair round, November 4

The current implementation and remaining A3 acceptance requirements are described
in [DCP_RUNTIME_REPAIR_20261104.md](DCP_RUNTIME_REPAIR_20261104.md). The observations
and first repair plan below are retained as the original evidence.

Evidence: `retest-nightly-20261104-061011`, commit 9c1b27f5d9, full GLM5.2
ModelSlim W4A8, Ascend910_9362x16, CANN 9.1, torch 2.10.0+cpu,
torch_npu 2.10.0.post6, sgl-kernel-npu 2026.9.0, DeepEP 00db6cd6,
cann_ops_transformer 1.0. Numerical checks were disabled; no result establishes quality.

## Findings

The historical HCCL 1000 control measured49.296s TTFT. TP routed scaling48.462s,
shared fusion47.200s, shared activation46.901s, metadata48.454s and automatic
stream thresholds46.287s do not establish a large improvement. Auto HCCL selected
256 MiB and measured23.955s; the stream/HCCL combination measured23.799s. Auto mode
also removed the explicit DeepEP buffer environment setting, so the next round
isolates HCCL 1000 versus256 with other control settings held fixed.

The full graph failed on missing DCP prefill metadata (`None.copy_`). The next
round uses PIECEWISE ONLY; the full-graph issue remains unqualified and no full
capture is scheduled. Piecewise explicitly rejects DCP16, so its next test is a
TP16/DCP1 short 8192-token probe with decode graphs disabled. The 1M DCP16 serving
measurements stay eager. Do not remove the CP guard or label this probe as 1M
piecewise coverage. All graph acceptance still requires capture and replay evidence.

The legacy40s control, input prequantization, combined TP fusion, FP16 indexer and
legacy MegaMoE exceeded the 600-second CLIENT budget while the server was making
progress. The final 2-second poll leaked into the exception text, misleadingly
suggesting a 2-second timeout. This was not evidence of an immediate client failure.
The client timeout now reports its total budget accurately. Repair cases get 1200s
per client repetition and 45minutes per case; the legacy MegaMoE client gets 1800s.

Both ordinary metadata batch-two and the repeated historical control hit native
RUNTIME allocator OOMs (Slice/Concat kernel initialization). HBM on the fullest
card was near 64 GiB; no inference that streamed o_proj caused these failures is
supported. Smaller explicit KV capacity is a headroom experiment, not a proven
memory fix: the new capacity recipe pins 63488 local rows, mem_fraction .80 and
HCCL 256. This is still 1M context; runtime working-set checks must pass before any
benchmark. Insufficient capacity stays blocked; there is no automatic truncation.

Packed batch-two completed requests but never executed a two-request prefill.
The resolved prefill token budget was 16384, while two cached tails need about
21,656 tokens. Batch-two launch overrides now set max_prefill_tokens=32768, with
runtime validation of the resolved value. A matched batch-one metadata control
uses the same budget. Activation remains mandatory; raw timing JSON survives an
activation failure and is labelled incomplete, never a pass.

Both current MegaMoE cases stopped after about 100K prefix tokens and triggered
the180 s watchdog. Native stacks show rtDeviceSynchronizeWithTimeout at allocation
in start_o_proj_gather, inside one-layer DCP prefetch lookahead. This does not
prove a DeepEP operator bug. The final diagnostic pair holds the capacity/HCCL
recipe fixed and compares MegaMoE with/without DCP gather/SFA lookahead. Streamed
o_proj remains enabled to avoid replicated-weight preparation. No speculative
kernel or communicator rewrite is included.

## Stabilized protocol

Default: `run_a3_nightly.py --round repair --hours 12 --repeats 3`. Fifteen ordered
cases; the legacy 34s recipe gets one diagnostic attempt, other successful variants
get three repetitions. Each repetition retains separate cold-prefix and cached
results, launch arguments, actual workload, activation, raw client logs/CSV and
server logs. Cold preparation stays within each client's identical protocol.
Historical 24s/34s labels are targets, not promised measurements. The legacy native
exact-token result and current AISBench recipe are not identical benchmarks.

The summary reports mean TTFT, mean of per-run P90 values, completed repetitions
and CV of per-run TTFT means. It does not report the last repetition as the aggregate
or pool P90 values as a request-level percentile. Partial repetitions remain
incomplete. Capacity controls bracket the optimization cases; drift or within-control CV over 5%, or fewer than three completed repetitions,
keeps cross-case comparisons provisional. Numerical quality stays unchecked as requested.

The global budget remains 12 hours. Final MegaMoE attempts have reserved time;
optional cases are explicitly skipped when budget is insufficient. Progress
updates every 15 s, host log copies every 10 minutes and on case completion. Full logs,
intermediate snapshots and source/runtime identities remain outside Docker. No
archive blocks progression; empty files are removed only at the end.

No target-device validation or native MegaMoE timeout resolution is claimed by
host tests. Deferred prerequisites: DCP16 piecewise support, full-graph metadata
repair (excluded by request), and target acceptance of the memory/lookahead trials.

Host validation: 52 tests and 42 subtests passed; one pre-existing worker-disconnect
test was deselected. All 15 repair launch recipes and CLI configurations resolved;
no full graph launch was present. Fatal lint and diff checks passed.
