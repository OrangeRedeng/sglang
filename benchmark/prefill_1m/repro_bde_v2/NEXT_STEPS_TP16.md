# TP16/DCP16 full-weight validation

The half-depth dummy-weight smoke passed on A3. Full weights plus KV cache
require sixteen NPUs. Full-model loading, numerical parity, and performance
remain to be checked. Run the stages below in order from the repository root,
with no other server holding the sixteen selected NPUs.

## 1. Full-model INT serving and short-request correctness

Use the full ModelSlim W4A8/W8A8 checkpoint, not a cropped model. These commands
use real weights and preserve its layer count. Both graph backends stay disabled.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
CHECKPOINT=/path/to/full/GLM-5.3-ModelSlim-INT
RESULTS="$PWD/.codex-artifacts/tp16-validation-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS"
git rev-parse HEAD > "$RESULTS/commit.txt"
npu-smi info > "$RESULTS/npu-before.txt"

COMMON=(--model-path "$CHECKPOINT" --checkpoint-format modelslim
        --tp 16 --dcp 16 --mem-fraction 0.70 --startup-timeout 1800)

python3 benchmark/prefill_1m/test_glm53_features.py --stage serve \
  --cases serve-dcp-kv-gather "${COMMON[@]}" \
  --context-length 8192 --chunk 4096 --output "$RESULTS/01-full-int.md"
```

The runner starts and stops the servers itself. Continue only when the report
shows successful real-weight loading, generation, and parity checks. A blocked,
dependency-blocked, failed, or coverage-blocked case is not a pass. If this full
checkpoint cannot fit on sixteen NPUs, stop and report the memory failure; a
cropped or dummy model cannot qualify the full-model configuration.

For the registered TP16/DCP16 launch/generation smoke alone, after exporting the
variables above:

```bash
GLM_5_2_W4A8_MODEL_PATH_OVERRIDE="$CHECKPOINT" \
  python3 test/registered/e2e/attention/test_npu_bde_v2_launch_tp16_dcp16.py
```

## 2. Qualify the DCP features and combined recipe

```bash
python3 benchmark/prefill_1m/test_glm53_features.py --stage serve \
  --cases 'serve-indexer-int8,serve-sfa-int8,serve-dcp-full-width-q,serve-dcp-legacy-token-shard,serve-dcp-stream-o-proj,serve-dcp-decode-q-preissue,serve-bde-v2' \
  "${COMMON[@]}" --context-length 8192 --chunk 4096 \
  --output "$RESULTS/02-dcp-features.md"
```

Require numerical parity and the expected activation evidence for each feature.
The streamed output projection and combined BDE recipe require TP == DCP; keep
both at sixteen. The first INT indexer-only case uses TP16 without DCP; the DCP
cases use TP16/DCP16. All cases still use only the sixteen visible NPUs.

## 3. Measure short serving, then 32K/128K prefixes

Compare the harness DCP reference with the qualified BDE-v2 recipe. The reference
retains the branch's default sharding policy; it is not an all-features-off run.

```bash
python3 benchmark/prefill_1m/test_glm53_features.py --stage small \
  --benchmark-variants dcp-kv-gather,bde-v2 "${COMMON[@]}" \
  --context-length 8192 --chunk 4096 --small-lengths 512,4096 \
  --small-concurrency 1,2 --small-requests 8 --benchmark-repeats 3 \
  --output "$RESULTS/03-short-serving.md"

python3 benchmark/prefill_1m/test_glm53_features.py --stage long \
  --benchmark-variants dcp-kv-gather,bde-v2 "${COMMON[@]}" \
  --context-length 131104 --chunk 32768 --long-lengths 32768,131072 \
  --long-concurrency 1,2 --long-requests 4 --benchmark-repeats 3 \
  --request-timeout 7200 --case-timeout 43200 \
  --output "$RESULTS/04-long-prefixes.md"
```

The long matrix includes cold and 99% shared-prefix requests. Check the reported
cached-token counts. Each point must fit both the checkpoint's supported context
and the server's available KV capacity; a skipped capacity point is not a pass.
Record TTFT, TPOT, E2E, throughput, peak memory, and mean prefill/decode batch sizes
where available. Long tests generate one token, so TPOT is undefined there.
Compare variants only within the same stage and identical launch/workload flags.

## 4. Gate the 1M point and the external AISBench run

Attempt 1M only after the preceding stages pass and the full checkpoint supports
1,048,576 input tokens plus output space. Confirm KV capacity first. Start at
concurrency one; keep the accepted lower-concurrency configuration for both
variants if higher concurrency cannot fit.

```bash
python3 benchmark/prefill_1m/test_glm53_features.py --stage long \
  --benchmark-variants dcp-kv-gather,bde-v2 "${COMMON[@]}" \
  --context-length 1048608 --chunk 32768 --long-lengths 1048576 \
  --long-concurrency 1 --long-requests 1 --benchmark-repeats 3 \
  --request-timeout 7200 --case-timeout 43200 \
  --output "$RESULTS/05-one-million.md"
```

For the user's external AISBench, start the qualified full-weight recipe in one
terminal; keep this launcher in the repository because it resolves paths from
its location:

```bash
CAPTURE=0 CHUNK=32768 MEMFRAC=0.70 MAXREQ=4 PORT=30000 \
  EXTRA_ARGS='--quantization modelslim --watchdog-timeout 7200' \
  python3 benchmark/prefill_1m/launch_bde_v2.py 16 --tp 16 \
  --model-path "$CHECKPOINT" --real-weights --context-length 131104 \
  2>&1 | tee "$RESULTS/aisbench-server-128k.log"
```

In a second terminal, save `/server_info` and verify TP16/DCP16, real weights,
disabled prefill/decode graphs, and the intended checkpoint. Run the existing
AISBench at 128K with concurrency one first; record its exact command, seed,
request count, output length, and repeat rate. Match those settings for the
archived-code comparison. Use a distinct output directory for each run. Raise
the server context for 1M only after the capacity gate above passes.

Return the Markdown reports, AISBench command/output, server logs, resolved
server configuration, and device/environment versions. Keep BF16 online
conversion, MXFP, and graph capture outside this validation sequence.
