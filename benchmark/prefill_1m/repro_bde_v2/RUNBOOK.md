# RUNBOOK: bde_v2 baseline vs `a3_910c_int_glm53` on the A3 (aisbench, TP16/DCP16)

Same harness (your aisbench), same day, same server config — the only variable
is the code tree. Run A then B per length before switching lengths.

Fixed server config (both phases): `TP=16 DCP=16 CAPTURE=0 CHUNK=32768
MEMFRAC=0.70 MAXREQ=64 PORT=30000`, decode graphs disabled, dummy weights
(the archive recipe). aisbench params are inside `bench_ais.sh`: `--output_len 1
--request_rate 0 --dataset_type prefix_cache --prefix_test --seed 4
--repeat_rate 0.99 --data_num 16 --concurrency 16`.

## 0. Bundle to the A3 host

```bash
# on the dev box
rsync -a /home/orangeredeng/code/sglang/.codex-artifacts/sglang_bde_v2/ a3:~/repro/bde_v2/
rsync -a /home/orangeredeng/code/sglang/.codex-artifacts/worktrees/a3-int-glm53/ a3:~/repro/branch/
```

On the host you need: the GLM-5.2-W4A8 checkpoint
(default `/home/ckpt/GLM-5.2-Provide-0610-W4A8-crop-tiny`, or export
`GLM_5_2_W4A8_MODEL_PATH_OVERRIDE`), CANN + torch_npu + `sgl_kernel_npu`
importable, and your aisbench checkout at
`/home/krodyushkin/aisbench_auto_tools_prefix` (or edit `AISBENCH_DIR` below).

## 1. Pre-flight (once per session)

```bash
unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy ALL_PROXY all_proxy
npu-smi info                                  # 16 NPUs idle
python -c "import sgl_kernel_npu"             # must import (else: conda libstdc++ LD_PRELOAD, see p4 script header)
ls ~/ckpt/GLM-5.2-Provide-0610-W4A8-crop-tiny # model present
```

## 2. Phase A — bde_v2 baseline (unmodified archived scripts)

```bash
RUNDIR=~/repro/run_A_$(date +%m%d_%H%M); mkdir -p $RUNDIR && cd $RUNDIR
AISBENCH_DIR=/home/krodyushkin/aisbench_auto_tools_prefix     # edit if different
cp ~/repro/bde_v2/{p4_exit1_serve_dcp.sh,bench_ais.sh} .

# terminal 1 — server (writes ./profile_dir_name_30000.sh here on startup)
DCP=16 TP=16 CAPTURE=0 CHUNK=32768 MEMFRAC=0.70 MAXREQ=64 PORT=30000 \
  EXTRA_ARGS="--watchdog-timeout 7200" \
  bash p4_exit1_serve_dcp.sh 16   2>&1 | tee server_A.log

# terminal 2 — wait for "The server is fired up", then verify and bench:
curl -s 127.0.0.1:30000/server_info | python3 -c \
  "import json,sys; i=json.load(sys.stdin); print('tp',i['tp_size'],'dcp',i['dcp_size'])"
# expect: tp 16 dcp 16 — do not proceed on anything else

bash bench_ais.sh 131072                      # 128k sanity point
bash bench_ais.sh 1048576                     # 1M point
```

Artifacts: `server_A.log`, both `*_aisbench.log` + aisbench output dirs (paths
are echoed into the logs), `profile_dir_name_30000.sh`. Then `^C` the server
and confirm `npu-smi info` is idle again.

## 3. Phase B — `a3_910c_int_glm53` (same recipe via the portable launcher)

```bash
RUNDIR=~/repro/run_B_$(date +%m%d_%H%M); mkdir -p $RUNDIR && cd $RUNDIR
cp ~/repro/bde_v2/bench_ais.sh .
cp ~/repro/branch/benchmark/prefill_1m/launch_bde_v2.py .
# the launcher does not create this; bench_ais.sh sources it from the CWD:
echo 'export SGLANG_TORCH_PROFILER_DIR='$RUNDIR > profile_dir_name_30000.sh

# terminal 1 — server (same env semantics as Phase A)
DCP=16 TP=16 CAPTURE=0 CHUNK=32768 MEMFRAC=0.70 MAXREQ=64 PORT=30000 \
  EXTRA_ARGS="--watchdog-timeout 7200" \
  python3 launch_bde_v2.py 16 --tp 16 --dry-run   # inspect first, then drop --dry-run
```

Then rerun without `--dry-run` (it execs the server in the foreground). The
command it builds is the archived recipe: dummy weights, ascend backend,
`--moe-a2a-backend none`, graphs off, plus the full recipe env stack
(see `CODE_AUDIT.md` for the knob mapping). Terminal 2 is identical to
Phase A (`/server_info` check, then `bash bench_ais.sh 131072` and
`bash bench_ais.sh 1048576`).

Optional attribution variant (one extra server cycle): the restored DCP
opt-ins are now overridable —

```bash
SGLANG_NPU_ENABLE_DCP_GATHER_PREFETCH=0 python3 launch_bde_v2.py 16 --tp 16 ...
SGLANG_NPU_ENABLE_DCP_SFA_GATHER_OVERLAP=0 python3 launch_bde_v2.py 16 --tp 16 ...
```

## 4. Order and comparability

- Interleave by length: A(128k) → B(128k) → A(1M) → B(1M). Never compare
  across days or across harnesses.
- `/server_info` must report tp 16 / dcp 16 before every bench; a leftover
  server on the port invalidates the run.
- If 1M × concurrency 16 OOMs at MEMFRAC 0.70: drop concurrency (edit a copy
  of `bench_ais.sh`, e.g. `--data_num 4 --concurrency 4`) and use the same
  copy for BOTH phases — note it in the results.
- If `/home/krodyushkin/sglang-code` is unwritable (Phase A output root): sed
  the single `SGLANG_TORCH_PROFILER_DIR=...` line in `p4_exit1_serve_dcp.sh`
  to a writable path. That is the only permitted edit to the archived script.

## 5. What to bring back

Per run: `server_*.log`, the aisbench log + output dir, and the first lines of
the server log (`===GIT LOG===` block records the exact tree for Phase A;
Phase B provenance is the launcher + this repo's HEAD). The `[SGL-TS]` lines in
the server log give the stage breakdown (RECV-HTTP → RECV-TM → BATCH →
MODEL-FWD → SENDBACK) for attribution beyond the aisbench totals.
