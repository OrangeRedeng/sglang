# GLM A3 feature-fix validation

Based on the GLM5.2 serving report at `ce785fb100`: 10 pass, 7 smoke-only,
1 inactive, 21 fail. These changes have CPU regression coverage; A3 acceptance
and overnight AISBench completion remain pending. A real GLM5.3 checkpoint is
a separate acceptance run.

## Changes and remaining prerequisites

| Family | Change | Required target evidence |
|---|---|---|
| Seven online INT cases | Restore logical MLA weight layout; guard nullable shared-expert config | Full-depth load, activation, teacher-forced parity |
| Full-width Q | Preserve supported dynamic INT8 projection weights/scales and activation quantization | TP16/DCP16 numerical parity, including ragged and empty ranks |
| Legacy token shard / streamed o_proj | Build replicas before KV sizing; reserve gather workspace; identify reference OOM separately | Peak memory, serving parity, stream activation |
| Full prefill graph | Refresh persistent indexer lengths outside capture | Repeated replay with changing request lengths and batch sizes |
| Piecewise graph | Pre-plan indexer shard metadata before tracing; retain full logs | Recover complete `NPUDriver.get_current_device` traceback if it persists, then repair its actual call site |
| Routed scale / shared / prequant / shared activation | Retain separate qualification and unchanged tolerance | A3 layer-output comparison; remaining numerical failures are unresolved |
| INT8 indexer / BDE combination | Retain strict parity gate | Compare scores and top-k against the unquantized indexer before qualifying the combination |
| TP finalize | Report dtype, bias and activation ineligibility | Compatible INT8 W2/operator ABI; W4A8 W2 is ineligible |
| FuseEP 1/2 | Require explicit W8A8 checkpoint and inspect its expert schemes | Compatible ModelSlim W8A8 checkpoint, EP16 activation and numerical parity |
| MegaMoE | Check DeepEP strategy keywords against the installed operator; record package paths, versions and signatures | Matching installed dependency pair; no keyword dropping or automatic dependency upgrade |

Reports retain full logs and responses in the `artifacts` directory recorded
in the report header. `failure_phase` distinguishes preflight, reference,
candidate, activation and parity. A capacity blockage is never a pass.
Smoke-only results return partial status and cannot qualify optimized timing.

## A3 qualification

Run inside the existing `sglang-a3` container after any earlier driver/server
has ended. Preserve its lock file and unrelated processes. Use the edited source
checkout; do not combine these uncommitted changes with an older installed tree.

```bash
cd /home/swx1199799/megamoe_test/sglang
export PYTHONPATH=python
REPORT_ROOT=/tmp/glm-a3-fix-validation-$(date +%Y%m%d-%H%M%S)
mkdir -p "$REPORT_ROOT"

python3 -m pytest -q -ra \
  test/registered/unit/npu/quantization/test_glm_a3_device_features.py \
  test/registered/unit/npu/attention/test_npu_quant_lightning_indexer.py \
  > "$REPORT_ROOT/device-tests.log" 2>&1

python3 -m torch.distributed.run --standalone --nproc_per_node=16 \
  test/registered/unit/npu/attention/test_dcp_prefix_gather_distributed.py \
  > "$REPORT_ROOT/distributed-tests.log" 2>&1

COMMON=(--stage serve --model-path /home/weights/GLM5.2-w4a8-int
  --checkpoint-format modelslim --bf16-model-path /home/weights/GLM-5.2
  --tp 16 --dcp 16 --context-length 8192 --chunk 4096
  --mem-fraction 0.88 --kv-rows 8192
  --startup-timeout 1800 --request-timeout 7200 --case-timeout 14400)

# Set this only to a real, compatible W8A8 ModelSlim checkpoint.
if [[ -n ${W8A8_MODEL_PATH:-} ]]; then
  COMMON+=(--w8a8-model-path "$W8A8_MODEL_PATH")
fi

python3 benchmark/prefill_1m/test_glm53_features.py "${COMMON[@]}" \
  --cases 'serve-baseline,serve-w8a8-load,serve-w4a4-mixed-load,serve-int8-dense-gate,serve-online-*' \
  --output "$REPORT_ROOT/online.md"

python3 benchmark/prefill_1m/test_glm53_features.py "${COMMON[@]}" \
  --cases 'serve-baseline,serve-tp-*,serve-fuseep-*,serve-megamoe' \
  --output "$REPORT_ROOT/moe.md"

python3 benchmark/prefill_1m/test_glm53_features.py "${COMMON[@]}" \
  --cases 'serve-indexer-*,serve-dcp-*,serve-bde-v2' \
  --output "$REPORT_ROOT/dsa.md"

python3 benchmark/prefill_1m/test_glm53_features.py "${COMMON[@]}" \
  --cases 'serve-graph-*' --output "$REPORT_ROOT/graphs.md"

# Run after reviewing the family reports; failures must remain visible.
python3 benchmark/prefill_1m/test_glm53_features.py "${COMMON[@]}" \
  --cases '*' --output "$REPORT_ROOT/complete.md"
```

Exit 0 means all selected checks passed, 1 means a failure, and 2 means partial
or blocked coverage. Numerical acceptance retains `atol=0.01, rtol=0.01`.
The harness repeats requests, checks batch-2 generation, and exercises cached,
ragged DCP requests; graph activation requires capture and replay markers.
Full A3 graph acceptance also requires checking replay across distinct buckets
and batches, rather than relying on startup capture.

## Timing

Use `OVERNIGHT.md` after serving qualification. The host runner now accepts
`--w8a8-checkpoint` and `--feature-kv-rows`; it carries the W8A8 checkpoint into
both qualification and timed FuseEP launches. Retain three repetitions with
matched weights, software, topology, graph mode, cache state and workload.
Record cold and cached phases separately, TTFT/P90, E2E, throughput, batch means
and TPOT where defined. Keep unresolved parity cases excluded from timing.
