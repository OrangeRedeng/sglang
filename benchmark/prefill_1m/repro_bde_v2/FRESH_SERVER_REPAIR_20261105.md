# Fresh-server repair after the 192648 kernel round

The 192648 run used 8c54335e59. Baseline TTFT repetitions were 23.913,
33.560 and 34.095 seconds. Other TP cases showed the same first/later pattern;
these means cannot establish fusion speedups. Cached prefix remained 989184
and new suffix about 10828 tokens. Arrival/queue distributions differed.
The exact source of that within-server drift remains unproven.

## Changes

- The nightly driver accepts one measured AISBench invocation per server only.
  The client's cold prefix preparation stays separate. Each variant and each
  bracketed control starts a fresh server; chunk32768, 1M context, 16 requests,
  concurrency16 and prefix99 are retained. One sample is exploratory evidence.
- Stop the outer suite before cleaning up its worker; otherwise the suite can
  launch the next case. A singleton suite lock prevents parallel orchestrators.
  SIGTERM runs the active suite's cleanup and stops further case launches.
  `--stop-only` cleans up without starting a new experiment.
- Retry the initial artifact export every15s until a remote path/copy succeeds,
  then every600s. Previously an export before path creation delayed logs10min.
- Aggregate every numeric metric consistently; previously TTFT/E2E means were
  combined with first-repetition throughput and E2EL fields.
- Save resolved server arguments in the workload record alongside requested
  settings: the capacity recipe resolves to KV63488 and mem_fraction0.80,
  although outer requested fields say KV67584 and0.85. Context remains1048576.
- Keep DCP prefix position construction explicitly on CPU. Its documented host
  `arange`/empty factories previously inherited the process default device.
  Under Ascend transfer/default-device routing this ran index construction and
  concatenation on NPU. Cases08/09 failed with native runtime OOM (ConcatD),
  including case09's prefix metadata construction stack. Explicit CPU allocation
  removes those unintended device temporaries/kernel launches. Target rerun is
  required to establish whether this fully resolves the memory failures.
- Block the GMM2 finalizer before server launch for W4A8 expert checkpoints.
  It requires W8A8 expert storage plus its installed WeightNz operator; changing
  dispatch to call it on packed W4 storage would be incorrect. This prerequisite
  remains blocked, not a pass. No dependency upgrades are included.

Case10 previously hit its45min total case budget during repetition3, with only
711s remaining for that invocation. Removing shared-server repetitions removes
that specific accumulation of cold preparation/client time within one case.
The45min ceiling and real client timeout remain enforced.

## Scope and evidence

Graph/MegaMoE cases11–15 had no results in the supplied previous run, so there is
no new graph/MegaMoE failure traceback to repair. Their existing cases remain in
the new round. Numerical qualification remains unchecked as requested. Host
regressions do not prove A3 native correctness, memory, HCCL or performance.

Launch with `run_a3_nightly.py --round kernels --hours 12 --repeats 1`.
The old command using `--repeats 3` is intentionally rejected.

Validation: 42 focused host tests and116 subtests passed; the existing
socket-dependent worker-disconnect test was excluded by sandbox restrictions.
The five fresh-server regressions also passed after the final cleanup-wait
change. Fatal lint, whitespace and15-case single-measurement plan checks passed.
No A3 measurements were executed locally.
