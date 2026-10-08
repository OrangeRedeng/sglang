# A5 final validation

Destination: `OrangeRedeng/sglang`, `A5_megamoe_experiments`.
Reference: `f02419e395ae7deb8dd51f5caa0b9a00325f6a32` with AUTO_TQ2 and
all rejected scheduling modes disabled. The branch is updated by a cleanup
commit; its existing history is preserved without a force push.

The retained changes come from `eac3159b5d`, `c8e199f2d2`, `703deadcc9`,
`095c271136`, `3655444700`, `c5ab0c4cb9`, `dca09b2b3a`, and `731252c7d0`.
Experimental portions of these commits and `d0e6b04c9b` were removed;
runtime scheduling additions from `c0a9d75a89` and `f02419e395` were removed.
Shared-stream tensor lifetime handling and the general profiler analyzer remain.
The rest of the branch history, including CP/DCP, is retained.

Target checks are performed by the user. Numerical correctness and final
regression are pending; CPU/static checks cannot establish A5 correctness or
performance. Verdict remains BLOCKED for production until both gates pass.

Development-host checks: focused policy tests passed (26 passed, 2 skipped),
modified Python files compiled, Ruff passed, shell syntax passed and
`git diff --check` passed. Tests for deleted resource scheduling were removed.
Static comparison confirmed unchanged attention helpers (including DCP gather)
and unchanged indexer query-sharding planning.

| Target validation | Result |
| --- | --- |
| Router-logit max/mean error | Pending |
| TopK ID mismatch count/rate | Pending |
| TopK-weight error | Pending |
| Final-logit/output error | Pending |
| Reference before / final / reference after / anchor / gain | Pending |

Use a fresh shell, outside profiling/capture runs:

```bash
source benchmark/kernels/a5_final_env.sh

python -m sglang.launch_server \
  --model-path /home/weights/GLM-5.2-W4A8C8-A5-0731/ \
  --attention-backend ascend --device npu --tp-size 4 \
  --disable-cuda-graph --trust-remote-code --mem-fraction-static 0.9 \
  --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
  --moe-a2a-backend none --max-running-requests 16 \
  --host 127.0.0.1 --port 30088 --page-size 128 \
  --disable-overlap-schedule --kv-cache-dtype fp8_e4m3 \
  --enable-request-time-stats-logging
```

Wait for the server to become ready, then run:

```bash
python -m sglang.benchmark.serving \
  --port 30088 --dataset-name generated-shared-prefix \
  --gsp-num-groups 1 --gsp-prompts-per-group 17 \
  --gsp-system-prompt-len 200000 --gsp-question-len 10828 \
  --gsp-output-len 1
```

Each run must report 17 successful requests, 3,731,608 input tokens and
17 output tokens. Record device generation/count, CANN, PyTorch, torch_npu,
sgl-kernel-npu, commit, checkpoint, environment and actual backend/CP/DCP path.
Use the same environment script for the reference in a fresh shell, where its
retired controls default to disabled/legacy. Keep CP/DCP/query sharding enabled.

First validate deterministic captured inputs against the reference:
router logits (max/mean error), TopK IDs (count/rate), TopK weights (error),
final logits/output (error). Use the accepted target operator tolerances;
explain every unexpected routing-ID mismatch before proceeding.

Then run one fresh-server sequence: REFERENCE, FINAL, REFERENCE. Record the
same TTFT statistic and workload signature for all three. Compute:

```python
anchor = (reference_before + reference_after) / 2
gain_percent = 100 * (1 - final / anchor)
```

Reject crashes/OOM, unexpected fallbacks, numerical mismatches or systematic
regression beyond observed noise. The campaign observed about 0.6–1.7% baseline
drift; investigate repeated regression above 1–1.5%. Do not reopen parameter
grids or rejected scheduling experiments.
