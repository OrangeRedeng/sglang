# Recover two-NPU Gemma4 throughput

Use `recover_throughput.py` for the fixed random C24 workload. It keeps the
older helpers intact. The runner defaults to TP1/DP2 on devices2,3, non-MTP,
piecewise prefill/full decode, serial QKV1, overlap on, memory0.92,
delayer0.5/50ms/4/cap6, 8192-token budgets and FIA reuse off.

The handoff documents 9.29req/s with this general configuration; the original
9.33req/s/~174s summary and its exact memory/environment were not recovered.
Treat this as a reconstructed control. 174s is benchmark duration, whereas
the historical mean request E2E was about2.57s. Do not label a reproduction
successful until measured on the target system.

## Launch and measure

Mount this checkout and one shared artifact directory in both containers.
The benchmark container needs these Python helpers, vLLM and the tokenizer;
the serving container needs the actual NPU runtime and checkpoint. Source
the normal CANN environment first. Paths below must resolve identically in
both containers. No other clients may use the server during an experiment:
the benchmark runner flushes its prefix cache.

In the SGLang container, from the Gemma checkout:

```bash
python3 test/manual/gemma4/recover_throughput.py serve \
  --output /artifacts/gemma-recovery/control-server1
```

The process stays in the foreground; logs go to `server.log`. Wait for the
server-ready message in that log. Ctrl-C stops the process group it owns.
Use `--print` to inspect the command and environment without launching or
creating files. All inherited `SGLANG_*` flags are removed; the runner sets
the experiment flags explicitly and records retained CANN/HCCL/host settings.
It sets `PYTHONPATH` to this checkout's `python` directory.

In the vLLM container:

```bash
python3 test/manual/gemma4/recover_throughput.py bench \
  --launch /artifacts/gemma-recovery/control-server1/launch.json \
  --server-log /artifacts/gemma-recovery/control-server1/server.log \
  --output /artifacts/gemma-recovery/control1
```

Each run checks resolved server arguments, captures deterministic greedy
outputs/logprobs, performs32 warmup requests, flushes the prefix cache, then
measures1600 requests. Both use seed48, prefix250/input1100/output100,
C24/RPS87, temperature0 and ignore EOS. The measured requests start with a
cold prefix cache and warm compiled kernels. This cache policy is explicit;
the historical run's cache state is unknown.

The built-in quality workload submits waves of1,7,12 concurrent completions,
mixing short, medium and roughly5200-token repeated-prefix prompts. It
compares greedy tokens/text exactly and logprobs with the existing helper's
0.02 absolute/relative tolerance. This is a serving smoke check. Verify the
checkpoint's actual sliding-window length and supplement with the existing
`quality_check.py` datasets if its SWA boundary or shared-KV cases are not
covered. Concurrency waves do not prove a particular graph bucket executed.

Raw vLLM results, detailed per-request lengths/stream intervals, client source
implementing peak metrics, server snapshots and benchmark-window logs are
saved. `metric-audit.json` recomputes completed RPS and aggregate output TPS.
Inspect `client-source/` and the detailed results to resolve peak concurrency
above24 or peak output TPS below the aggregate rate; neither peak determines
the winner. A missing installed client source remains unresolved evidence.
The server window includes any vLLM built-in probe; its start/end and byte
offset are saved. `batches.json` reports per-replica prefill sequences/tokens
and sampled decode batch histograms. Decode samples are not per-forward or
time-weighted statistics. Keep the complete server log for failures and graph
diagnostics.

## One change per experiment

First collect control1/control2/control3 with new output directories. If the
control remains materially below the historical result, recover the original
summary, command, hardware/software and prefix-cache state before tuning.

Restart the server for every candidate. Pass all settings retained from the
current winner explicitly; then change only the next tested setting:

| Order | Candidate option | Comparison `--change` |
| --- | --- | --- |
| 1 | `--delayer 0` | `delayer` |
| 2 | `--delay 25` versus50 | `delay` |
| 3 | `--ratio 0.25` versus0.5 | `ratio` |
| 4 | `--cap 5` versus6 | `cap` |
| 5 | `--metadata 1` versus0 | `metadata` |
| 6 | `--kv-views 1` versus0 | `kv_views` |
| 7 | `--add-decode-buckets <observed missing sizes>` | `add_decode_buckets` |

For example, add `--delay 25` to `serve`, use a fresh server directory and
benchmark into `delay25-1`. Supply `--quality-baseline
/artifacts/gemma-recovery/control1/quality.json` to fail before timing if the
quality smoke check regresses. Screen once, then confirm a finalist using
three interleaved pairs: control1/candidate1, control2/candidate2,
control3/candidate3, restarting servers and using new directories as needed.

Do not guess extra graph buckets from C24 alone. Use per-replica decode
histograms to identify frequent missing sizes, then corroborate selected
buckets with a separate shape/graph trace using `collect_profile.py`. The
runner retains all existing buckets. Scheduler batch logs alone do not expose
the actual selected graph bucket; report inferred padding as inferred.

Compare a finalist:

```bash
python3 test/manual/gemma4/recover_throughput.py compare --change delay \
  --baseline /artifacts/gemma-recovery/control1 /artifacts/gemma-recovery/control2 /artifacts/gemma-recovery/control3 \
  --candidate /artifacts/gemma-recovery/delay25-1 /artifacts/gemma-recovery/delay25-2 /artifacts/gemma-recovery/delay25-3 \
  --output /artifacts/gemma-recovery/delay25-comparison.json
```

The report requires matched workload, client sources, checkpoint metadata,
runtime, devices and all untested launch settings. It rejects partial/failed
runs and missing100-token per-request evidence. It reports median and range
of completed/duration RPS, output TPS, duration, mean/P99 TTFT/TPOT/E2E,
quality comparisons and batch statistics. `keep` requires three runs each,
at least3% median gain, all six latency metrics within5%, matching quality,
batch observations for both replicas and non-overlapping throughput ranges.
Overlapping ranges or fewer repetitions are inconclusive. This report is
a performance/quality-smoke gate; inspect full logs for stalls/OOMs and obtain
the required SWA/graph correctness evidence before declaring target acceptance.

After each hardware round, record the actual commands, one hypothesis and
keep/reject/inconclusive decision under the existing Gemma goal using
`workflow-exp record`; never insert dry-run or synthetic metrics as NPU results.
The report's `exceeds_9_33` field is separate from the regression checks.

## Optional matched MTP comparison

After selecting the non-MTP winner, repeat its settings at `--memory 0.82`.
Compare that fresh control against the same command plus `--mtp 1`. This uses
one draft step/two verification tokens, eager draft, target verification graph
and identical piecewise prefill/QKV/delayer settings. MTP requires explicit
`--memory 0.82` to avoid inheriting the control's0.92 allocation.

Validate MTP with `--overlap 0` first, using an equally non-overlapped
non-MTP control. Only then test MTP overlap0/1 independently; use
`--change overlap` for that comparison. A stable MTP/non-MTP pair uses
`--change mtp`. No combined draft/verification graph trial is enabled.
If MTP loses, retain non-MTP. Profile the stable winner separately with
`collect_profile.py` while driving the same workload; exclude profiled runs
from the comparison. Kernel work follows measured bottlenecks.
