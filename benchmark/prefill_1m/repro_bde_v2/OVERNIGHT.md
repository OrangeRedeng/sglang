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
| Host environment script | First existing: `/home/tbaydasov/set_env.sh`, `/home/tbaydasovset_env.sh` |
| DCP server | TP16/DCP16, eager, CHUNK32768, MAXREQ64, one request per prefill batch, MEMFRAC0.88 |
| KV cap | 81920 local rows, or 1310720 logical tokens |
| Context | 1048576; benchmark input is 1000000 plus chat framing |
| Workload | seed 4, prefix repeat 0.99, 16 requests/concurrency, one output token |
| AISBench repetitions | 3 per qualified variant |
| MoE server/workload | TP16, DCP1, context8192, chunk4096; input4096/output128; prefix repeat0.5 |
| MoE KV cap | 65536 tokens; MAXREQ64, sixteen concurrent benchmark requests |
| MegaMoE | `ascend_fuseep`, mode3, EP16; modes1/2 tested separately |

An explicit `--ais-env` overrides detection. Missing paths are reported separately,
with the exact host filenames. Check them in the foreground before starting screen:

```bash
bash /tmp/start_bde_overnight.sh --check-paths
```

If needed, pass `--ais-dir /actual/aisbench_auto_tools_prefix --ais-env /actual/set_env.sh`
to both the path check and the overnight command. The check validates file presence;
the actual AISBench invocation sources the environment script.
Use `--repo`, `--checkpoint`, `--container`, `--ais-dir` and `--output` to
override paths. The checkpoint and repo paths are paths **inside Docker**;
AISBench, its environment script and the output directory are **host paths**.

## Update the branch

The runner and padding fix are committed to `glm52-a3-optimizations`.
The old `bde-overnight.patch` was only for delivery before publication;
do not apply it to the updated branch. Already-existing-file errors mean
that patch cannot be applied to the current checkout. Normal `git apply`
without `--reject` makes no partial changes when it fails.

Run on the host, with the container checkout on the integration branch:

```bash
REPO=/home/swx1199799/megamoe_test/sglang
docker exec -w "$REPO" sglang-a3 git pull --ff-only \
  https://github.com/OrangeRedeng/sglang.git glm52-a3-optimizations
docker cp "sglang-a3:$REPO/benchmark/prefill_1m/start_bde_overnight.sh" /tmp/start_bde_overnight.sh
```

The padding fix aligns input IDs, positions, embeddings and KV write IDs
before eager-buffer loading and metadata planning. Real request lengths stay
unchanged. This covers the 6972-token final chunk in the supplied crash and
the 10828-token cached tails, while six-token warmup remains unsharded.

## Launch on the host

```bash
bash /tmp/start_bde_overnight.sh --moe-only --dry-run
screen -dmS bde-night bash -lc \
  'bash /tmp/start_bde_overnight.sh --moe-only > /tmp/bde-night.log 2>&1'
tail -f /tmp/bde-night.log
```

The wrapper copies the current runner from the container each time and prints
its checkout commit, avoiding an outdated host copy. It uses Docker even for
`--dry-run`; the runner itself does not start servers or benchmark during dry run.
Set `SGLANG_CONTAINER`/`SGLANG_REPO` to override the wrapper's container/repo.

The command above focuses on **MoE**. Omit `--moe-only` to include the 1M DCP
matrix too. For a smaller
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
Packed DCP reads support a single request per prefill batch. The DCP group
therefore pins `--prefill-max-requests 1` while keeping `MAXREQ64` for running
requests/decode. Batching two 1M contexts selects the permutation-based gather
instead, duplicating the shared prefix and increasing KV movement. MoE tests
retain their normal batching. For manual launcher experiments, override the
prefill cap with `PREFILL_MAXREQ` or `EXTRA_ARGS='--prefill-max-requests ...'`.

For the supplied 1M prefix workload, the 67584-row TP16 pool holds 1081344
logical tokens. The shared prefix plus sixteen distinct tails needs roughly
1150000 tokens before page/decode overhead; the runner's 81920-row cap provides
1310720. To retest manually with the baseline's 16384-token chunk size:

```bash
CAPTURE=0 CHUNK=16384 MEMFRAC=0.88 MAXREQ=64 PREFILL_MAXREQ=1 PORT=30000 \
EXTRA_ARGS='--quantization modelslim --watchdog-timeout 7200 --max-total-tokens 81920' \
python3 benchmark/prefill_1m/launch_bde_v2.py 16 --tp 16 \
  --model-path /home/weights/GLM5.2-w4a8-int --real-weights --context-length 1048576
```

Check resolved `prefill_max_requests=1` and `#new-seq: 1` in prefill batches.
Repeat the original AISBench input1000000/output1/concurrency16/prefix0.99
workload after a cache flush. Compare its measured sixteen-request phase;
the initial one-request 990k-token cold-prefix phase is separate.
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
