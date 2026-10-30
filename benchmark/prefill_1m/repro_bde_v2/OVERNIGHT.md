# Overnight TP16 MoE, MegaMoE and DCP runner

Run `benchmark/prefill_1m/run_bde_overnight.py` on the **A3 host**, outside
Docker. It uses the running `sglang-a3` container for SGLang and the host's
AISBench environment. It does not attach to or modify your existing screens.
Stop the current server in `MoE_test` and any existing AISBench run first.
Port 30000 and all sixteen selected NPUs must be available for this run.

Defaults match the supplied installation:

| Setting | Default |
|---|---|
| Container | `sglang-a3`, running with host networking |
| Checkout inside container | `/home/swx1199799/megamoe_test/sglang` |
| Full ModelSlim checkpoint | `/home/weights/GLM5.2-w4a8-int` |
| Host AISBench directory | `/home/tbaydasov/aisbench/aisbench_auto_tools_prefix` |
| Host environment script | `/home/tbaydasovset_env.sh` |
| DCP server | TP16/DCP16, eager, CHUNK32768, MAXREQ64, MEMFRAC0.88 |
| KV cap | 81920 local rows, or 1310720 logical tokens |
| Context | 1048576; benchmark input is 1000000 plus chat framing |
| Workload | seed 4, prefix repeat 0.99, 16 requests/concurrency, one output token |
| AISBench repetitions | 3 per qualified variant |
| MoE server/workload | TP16, DCP1, context8192, chunk4096; input4096/output128; prefix repeat0.5 |
| MoE KV cap | 65536 tokens; MAXREQ64, sixteen concurrent benchmark requests |
| MegaMoE | `ascend_fuseep`, mode3, EP16; modes1/2 tested separately |

The environment-script spelling above is taken literally from your message.
If its actual path is `/home/tbaydasov/set_env.sh`, pass that with `--ais-env`.
Use `--repo`, `--checkpoint`, `--container`, `--ais-dir` and `--output` to
override paths. The checkpoint and repo paths are paths **inside Docker**;
AISBench, its environment script and the output directory are **host paths**.

## Install the prepared changes

The runner and the final-chunk padding fix must both be present in the
container checkout. If using the supplied patch, download it to the A3 host
as `/tmp/bde-overnight.patch`, then run on that host:

```bash
REPO=/home/swx1199799/megamoe_test/sglang
docker cp /tmp/bde-overnight.patch sglang-a3:/tmp/bde-overnight.patch
docker exec -w "$REPO" sglang-a3 git apply --check /tmp/bde-overnight.patch
docker exec -w "$REPO" sglang-a3 git apply /tmp/bde-overnight.patch
docker cp "sglang-a3:$REPO/benchmark/prefill_1m/run_bde_overnight.py" /tmp/run_bde_overnight.py
```

The patch is based on `8319071dde`. Do not apply it again if these changes
are already installed; `git apply --check` refuses conflicting application.
The padding fix aligns input IDs, positions, embeddings and KV write IDs
before eager-buffer loading and metadata planning. Real request lengths stay
unchanged. This covers the 6972-token final chunk in the supplied crash and
the 10828-token cached tails, while six-token warmup remains unsharded.

## Launch on the host

```bash
python3 /tmp/run_bde_overnight.py --moe-only --dry-run
screen -dmS bde-night bash -lc \
  'python3 -u /tmp/run_bde_overnight.py --moe-only > /tmp/bde-night.log 2>&1'
tail -f /tmp/bde-night.log
```

The command above focuses on **MoE**. Omit `--moe-only` to include the 1M DCP
matrix too. No model loading or Docker calls happen during `--dry-run`. For a smaller
AISBench matrix while retaining feature qualification, add
`--variants dcp-kv-gather,bde-v2`. To run qualification without AISBench, use
`--features-only`. This matrix can take many hours; each server start may
load the full checkpoint again.

## What runs and what qualifies

1. Existing host tests, NPU feature checks, and the serving feature matrix.
   The matrix runs numerical/reference comparisons and activation probes.
   Cases needing a BF16 source checkpoint, unavailable operators or unsupported
   configurations remain explicitly blocked; they do not count as passes.
2. Qualified DCP variants: reference KV gather, packed read, prefix prefetch,
   gather/indexer overlap, full-width Q/local top-k, token-local residuals,
   streamed output projection, decode Q preissue and the combined BDE recipe.
   The MoE group additionally benchmarks the ordinary TP reference, routed-scale
   fusion, GMM2 finalize fusion, input prequantization/reuse, TP shared-expert
   fusion, shared-expert GMM1 reuse, multistream, automatic stream thresholds,
   automatic HCCL sizing, and FuseEP modes1/2/3 (MegaMoE). The shorter MoE
   workload has 128 output tokens so decode and TPOT can also be measured.
3. Three AISBench repetitions per qualified variant. The cache is flushed
   before **each** invocation; AISBench then performs its own prefix prewarm.
   Identical seeded requests cannot accidentally become full-cache hits from
   the preceding repetition. Cold-prefix and measured cached phases stay in
   the raw AISBench log; aggregate TTFT uses the 16-request measured phase.

Timed servers disable stage prints, DCP debug prints and feature-evidence
logging. The script checks resolved topology, real weights and KV capacity,
and only benchmarks cases with verified serving quality and required markers.
Failures are saved and independent cases continue. Disconnecting the host
worker stops its own server/process groups; existing servers are never killed.
Run one AISBench driver at a time because its configuration/dataset files are shared.

MegaMoE requires the matching `deep_ep`/operator API and compatible checkpoint
weights. Missing capabilities are reported, not silently replaced by the TP
backend. The BDE residual path requires EP1, so it is not combined with the
EP16 FuseEP modes. MoE and DCP are different workload groups; compare MoE
variants to the MoE reference, not to the reported 20-second 1M DCP baseline.
Override MoE input/output lengths and prefix reuse with `--moe-length`,
`--moe-output-len` and `--moe-repeat-rate`; input plus output must fit context8192.

## Results

Default host output: `/home/tbaydasov/aisbench/overnight-results/<UTC timestamp>/`.

- `SUMMARY.md`: per-stage/case status and per-repeat TTFT, E2E duration and throughput.
- `results.json`: metrics and error reasons; `aggregate.json` stores median TTFT per variant.
- `container-artifacts/`: serving reports, resolved launches/configuration and server logs.
- Per-variant `aisbench-*.log` and CSV snapshots when AISBench provides its CSV.
- `settings.json`, `commit.txt` and the exact runner used for this run.

Exit code 0 means all selected stages/cases passed; 1 means a failure; 2 means
partial/blocked/unqualified coverage. DCP's one output token makes TPOT undefined;
the MoE group retains AISBench's measured TPOT in `results.json`.
Compare only matched launch/workload settings. These are candidate measurements,
not a guarantee that the branch beats the earlier 20-second baseline.
