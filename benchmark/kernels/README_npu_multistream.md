# NPU multistream analysis

The final A5 runtime retains the TP MoE pre-gate shared stream. Rejected DSA
streams and late/shared-resource scheduling have been removed.
Use [the final configuration](README_a5_final.md) for serving validation.

Analyze one rank's exported tasks, using stream IDs from its actual capture:

```bash
python benchmark/kernels/analyze_npu_multistream.py tasks.csv \
  --main-stream MAIN_ID --side-stream shared=SHARED_ID \
  --baseline baseline_tasks.csv --baseline-main-stream BASE_MAIN_ID
```

Optional `--device-pid`, `--start-us`, and `--end-us` restrict the window.
The analyzer separates computation, communication, waits and copies, and
keeps unknown/mixed core resources visible. Increased overlap or reduced
summed kernel time does not establish a critical-path improvement. Compare
wall time, stream waits, steady forward time and fresh-server TTFT.
