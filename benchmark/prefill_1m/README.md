# GLM-5.3 1M-token prefill benchmark (Ascend A3 910C)

Prefill-dominated benchmark for the `a3_910c_int_glm53` integration branch:
GLM-5.3 (`glm_moe_dsa`) at ~1M-token prompts, TTFT and prefill tokens/s, with
the branch's INT quantization, prefill CUDA graph and DCP features exposed as
switches.

## Files

- `bench_glm53_prefill_1m.py` — client. Synthetic seeded token-id prompts
  (no dataset or external aisbench checkout needed), `output_len=1`,
  concurrent batch sweep, warmup + repeats, JSON output, `--dry-run` mode.
- `serve_glm53_1m.sh` — server launcher, adapted from the bde_v2 print-build
  `p4_exit1_serve_dcp.sh` recipe (proxy handling, int8 LightningIndexer, DCP
  shared-gather envs, chat-encode opts).
- `reference/` — the original zip scripts (`bench_ais.sh`,
  `p4_exit1_serve_dcp.sh`), kept for provenance. `bench_ais.sh` drives an
  external `aisbench_test.py` checkout that is not part of this repo; the
  Python bench replaces it.

## Run

```bash
# 1. Start the server (bf16 checkpoint, eager prefill baseline):
MODEL_PATH=/path/to/GLM-5.3 bash benchmark/prefill_1m/serve_glm53_1m.sh

# 2. In another shell:
python3 benchmark/prefill_1m/bench_glm53_prefill_1m.py \
    --context-length 1048576 --batch-sizes 1,2,4,8 --repeats 3 \
    --output results_1m_bs_bf16.json
```

Sweeps this stack was built for:

```bash
# Prefill CUDA graph on (decode graphs + NPU prefill body capture):
CAPTURE=1 MODEL_PATH=... bash benchmark/prefill_1m/serve_glm53_1m.sh

# Online INT quantization from a bf16 checkpoint (no pre-quantized weights):
ONLINE_QUANT=w8a8_int MODEL_PATH=... bash benchmark/prefill_1m/serve_glm53_1m.sh
ONLINE_QUANT=w4a4_int MODEL_PATH=... bash benchmark/prefill_1m/serve_glm53_1m.sh

# DCP (context parallel) sweep; DCP must divide TP. The packed int8 SFA KV
# (SGLANG_NPU_SFA_KV_8BIT) only engages with DCP > 1:
TP=16 DCP=8 SGLANG_NPU_SFA_KV_8BIT=True MODEL_PATH=... \
    bash benchmark/prefill_1m/serve_glm53_1m.sh
```

The client defaults to `--cache-mode cold`: every warmup and measured batch
uses fresh seeded prompts. `--cache-mode cached` primes all prompts at least
once and reuses them for measurements. Run against a dedicated server with
enough cache capacity; check each sample's `cached_tokens` to confirm hits.
Use a different `--seed` when repeating cold runs against the same server.

TTFT is measured per request from submission to its first output token.
`prompt_tokens_per_s` counts all supplied tokens; `prefill_tokens_per_s`
subtracts server-reported cached tokens and is null when those counts are
unavailable. Each sample retains request timings and cache counts.

KV budget note: a 1M-token prompt at one concurrency already needs a very
large pool; `serve_glm53_1m.sh` defaults to `--mem-fraction-static 0.85` and
`--context-length 116864`. For the 1M run raise `CTX` (the bench prompt is
1,048,576 tokens plus one output token) and be ready to drop `--batch-sizes`
to `1,2` if pool construction OOMs. Chunked prefill stays on
(`--chunked-prefill-size 32768` default, the cookbook's long-input starting
point).

## Automatic feature checks

Run from the repository root with the same Python environment used for serving:

```bash
# Host checks: syntax, benchmark regressions, decoder forwarding, imports, pytest commands, 1M dry-run.
python3 benchmark/prefill_1m/test_glm53_features.py

# Inspect the full matrix without loading weights or initializing devices.
python3 benchmark/prefill_1m/test_glm53_features.py --stage all --list
python3 benchmark/prefill_1m/test_glm53_features.py --stage all --dry-run

# Full A3 run: existing unit suites, NPU operator tests, sequential server variants.
python3 benchmark/prefill_1m/test_glm53_features.py --stage all \
    --model-path /path/to/GLM-5.3-BF16 --tp 16 --dcp 8

# Optional 1M-token cold-prefill benchmark on the baseline server.
python3 benchmark/prefill_1m/test_glm53_features.py --stage serve \
    --cases serve-baseline --model-path /path/to/GLM-5.3-BF16 \
    --tp 16 --dcp 8 --context-length 1048608 --long-context 1048576

# Rerun a subset; case globs are comma-separated.
python3 benchmark/prefill_1m/test_glm53_features.py --stage unit \
    --cases 'unit-int-*,unit-prefill-*'
```

Use a local BF16 GLM-5.3 checkpoint for the matrix: the online conversion
cases need floating-point source weights. Install SGLang from this checkout
and its matching CANN, PyTorch, torch_npu, sgl-kernel-npu, Triton-Ascend and
pytest dependencies beforehand. Each serving case reloads the model, so the
full matrix is expensive. Use an idle A3 host; TP must fit the visible NPU
count and DCP must divide TP. Serving smoke uses 8K context by default;
the optional 1M benchmark needs a much larger KV budget.

Replace `/path/to/GLM-5.3-BF16` with an existing checkpoint directory on the
A3 machine. Alternatively, export `MODEL_PATH` to that directory and omit
`--model-path`. A missing or example path leaves serving cases blocked;
unit/operator suites can still run independently.

For the offline INT checkpoint at `/home/weights/GLM5.2-w4a8-int/`:

```bash
python3 benchmark/prefill_1m/test_glm53_features.py --stage all \
    --model-path /home/weights/GLM5.2-w4a8-int/ --checkpoint-format modelslim \
    --tp 16 --dcp 8 --output report.md
```

This passes `--quantization modelslim` to each server. The three online
conversion cases remain blocked because they require BF16 source weights;
the other suites and serving variants still run. Exit code 2 retains that
incomplete coverage. The checkpoint's config selects its architecture;
the GLM5.2 directory name alone does not establish GLM-5.3 coverage.

To cover online conversion in the same run, add the BF16 checkpoint:

```bash
python3 benchmark/prefill_1m/test_glm53_features.py --stage all \
    --model-path /home/weights/GLM5.2-w4a8-int/ --checkpoint-format modelslim \
    --bf16-model-path /actual/path/to/GLM-5.3-BF16 \
    --tp 16 --dcp 8 --output report.md
```

Only the three online-conversion cases use `--bf16-model-path`; the other
serving cases use the primary offline INT checkpoint. Each case records its
checkpoint path and format. Use matching model revisions if comparing them.

| Coverage | Automated checks |
|---|---|
| Benchmark fixes | Concurrent HTTP requests, fresh cold prefixes, cached priming, first-token timing, launcher arguments |
| GLM boundary fix | Actual decoder forward with host dependency stubs; selected INT format reaches its consumer |
| INT quantization | Contract/loader suites; W8A8, mixed W4A4/W8A8, scoped INT8 dense/router/shared serving |
| TP MoE | Routing/shared-weight/top-k suites; finalize fusion and shared-expert serving variants |
| DCP/indexer/SFA | Layout, ownership, gather/overlap, sparse parity, query shard and index-pool suites; real NPU pool/INT8 round trips; reviewed KV-gather/token-shard and packed serving |
| Graphs | Configuration, full-prefill and piecewise suites; non-DCP full/piecewise serving variants |
| Host path | Pool/memo/raw encoding suites and chat-completion serving variants |
| Scheduling | Autotune, HCCL sizing and resource-scheduling suites; multistream/autotune serving variants |

Each run produces one Markdown file under
`.codex-artifacts/glm53-feature-tests/<UTC timestamp>.md`. It contains the
summary, commands, environment/version evidence, complete logs, pytest JUnit
results, server information and responses inline. `--output report.md` selects
a fresh report path. Raw logs, JSON and XML are temporary and removed after
they are included in the report; no separate result files need collecting.
Failures do not stop the remaining independent cases. Failed device preflight
blocks device cases; skipped pytest cases produce `partial`, never `pass`.
Exit codes: `0` = selected checks completed, `1` = failure, `2` = blocked/partial.
A dry run records `planned` and executes no checks.

Serving success is recorded as `smoke_pass`, with activation, model quality
and graph replay explicitly **unverified**. Capability gates can choose eager
or unfused fallbacks. Inspect the saved logs/operator evidence before claiming
an optimization ran; these checks do not establish speedups or full model
accuracy. Existing component tests run unchanged, including any stale MXFP
expectations that the INT-only policy now rejects. Gemma-only paths are outside
this GLM matrix, and the deferred D1v3 boundary optimization remains unvalidated.

## Stage-latency breakdown

The branch carries the bde_v2 print-build `[SGL-TS]` probes:
`RECV-HTTP` (serving_base), `RECV-TM` (tokenizer_manager), `BATCH-START/END`
(scheduler), `MODEL-FWD-START/END` (model_runner), `SENDBACK`
(serving_chat). Grep the server log for `SGL-TS` to segment TTFT into
tokenize / schedule / forward / send-back. These prints are always-on in the
current import commit; gate or strip them before quoting clean throughput
numbers.

## Divergence from the cookbook

`docs/cookbook/autoregressive/GLM/GLM-5.3.mdx` documents GPU deployments
(H200/B200/GB300/AMD, FP8 weights, DSA backends auto-selected). This bench
targets A3 910C NPU: `--device npu --attention-backend ascend` replace the
cookbook's backend selection, TP8/16 replaces its TP layouts, and the INT
online-quantization modes (`w8a8_int`/`w4a4_int`) are this branch's Ascend
load-time conversion, not the cookbook's FP8 checkpoints.

## Deferred follow-ups (need A3 hardware validation)

1. **D1v3 `prepare_mlp` port**: the token-sharded all-reduce-skip hook in
   `layers/communicator.py` became unreachable after main's `layer_boundary`
   refactor; D1v3 degrades to the safe default (full all-reduce). Port the
   hook onto the declared-boundary API to restore the optimization, then
   re-measure DCP runs.
2. **DCP packed-read precedence**: in
   `deepseek_v2_attention_mla_npu.py`, ours' `dcp_packed_read_plan` and
   main's `remap_dcp_sparse_indices` were made mutually exclusive
   (if/elif). When `SGLANG_NPU_DCP_SHARED_PACKED_*` is on, the sparse remap
   is skipped — confirm on device that the packed path matches the ascend
   backend operator contract.
3. **Indexer-K pool sizing**: `memory_pool_npu.py` merged the bde
   `index_buf_size` capacity override with main's paged
   `index_page_size`/`index_page_padding` layout by construction; validate
   pool construction at DCP > 1.
4. **Prefill graph vs DCP**: `can_run_prefill_cuda_graph` excludes DCP runs,
   so the CAPTURE=1 win applies to the non-DCP 1M configuration only.

## Integrated PR validation

The matrix now follows #42977/#42978 indexer/token sharding, #44320/#44321
DCP KV gather composition, #43141 nesting validation and #42224 INT8 indexer.
Legacy D1v3 and asynchronous gather experiments are not selected by this matrix.
Scoped routed-MoE/INT4 conversion is unsupported; use `w8a8_int`/`w4a4_int`
load-time modes. The dense/router/shared INT8 case uses explicit targets and
never requests unsupported `moe:W8A8_DYNAMIC`.

Packed INT8 KV uses a BF16 page read for DCP partial attention; its temporary
read allocation and performance require A3 validation. HiCache disables INT8
indexer storage as in the current #42224 head. Tests of removed experimental
stream APIs have been removed; runtime state is explicit in the pool/graph tests.

## A3 validation, individual MoE tests and TTFT benchmarks

The primary checkpoint is ModelSlim INT. BF16 weights are used only as the
source of online-quantization cases. Integer presets create meta parameters,
stage incomplete local weights on CPU, and serialize completed-group NPU
conversion. A preset can be followed by target overrides, for example
`--online-quantization w8a8_int,gate:w8a8_dynamic`. The override wins for that
target; routed experts remain streamed and compressed. Standalone scoped
requests leave unselected weights floating. CPU staging consumes host memory
for outstanding loading groups; this is separate from NPU peak memory.

```bash
# Host + device regressions, independent feature activation and numerical parity.
python3 benchmark/prefill_1m/test_glm53_features.py --stage all \
  --model-path /home/weights/GLM5.2-w4a8-int/ --checkpoint-format modelslim \
  --bf16-model-path /home/weights/GLM-5.2 --tp 16 --dcp 8 --output validation.md

# Small SGLang serving benchmarks: baseline and selected qualified variants.
python3 benchmark/prefill_1m/test_glm53_features.py --stage small \
  --cases 'small-baseline,small-tp-*,small-megamoe' --output small.md

# Cold and 99% shared-prefix TTFT, 32K/128K/256K/512K/1M, concurrency 1/16.
python3 benchmark/prefill_1m/test_glm53_features.py --stage long \
  --benchmark-variants baseline,megamoe --tp 16 --dcp 8 --output long.md

# Inspect commands before allocating devices or loading weights.
python3 benchmark/prefill_1m/test_glm53_features.py --stage long --dry-run \
  --benchmark-variants baseline,megamoe --output long-plan.md
```

Use `--list` and `--cases` to select individual cases. Separate cases cover
routed scaling, GMM2 finalization, shared finalization, INT8 input reuse,
shared activation quantization, multistream scheduling, automatic thresholds,
HCCL sizing, FuseEP modes 1/2 and MegaMoE mode 3. Online gate, QKV, dense and
shared-expert overrides each have their own case. MXFP experiments are excluded.

Every serving variant runs against an otherwise matching reference and checks
teacher-forced input-token logprobs (`rtol=0.01`, `atol=0.01`). Dedicated NPU
layer tests compare the same quantized operands numerically. `PASS` requires
activation evidence where specified and numerical parity; generation-only
cases retain `SMOKE_PASS`. Missing activation is `INACTIVE`; missing APIs are
`DEPENDENCY_BLOCKED`. Skipped device tests remain visible as incomplete evidence.
These states do not count as a fully validated suite.

Small benchmarks use `python -m sglang.benchmark.serving`, exact random token
IDs, 512/4096 inputs, 32 outputs, 32 requests, concurrency 1/8 and three repeats.
Long benchmarks use 16 requests per repeat and one output token, with independent
cold and 99% prefix-reuse workloads. Both restart the variant with diagnostic
evidence disabled for timing. Warmup is excluded; cold repeats flush the cache,
and prefix runs flush then prime their shared prefix. Observed cached-token
counts are checked. The report includes individual TTFT samples, median/p95,
repeat variability, commands and bounded diagnostic excerpts.

Override workload sizes with `--small-lengths`, `--small-concurrency`,
`--small-requests`, `--long-lengths`, `--long-concurrency`, `--long-requests`,
`--benchmark-repeats` and `--request-timeout`. Length/concurrency arguments are
comma-separated integers. Longer-than-supported contexts or insufficient KV
capacity produce explicit blocked points; requests are never shortened. The
server uses at most the checkpoint's declared context limit by default. A 1M
point needs a checkpoint/configuration that actually supports that context.

MegaMoE mode 3 requires an A3-compatible ModelSlim W4A8/W8A8 checkpoint and
matching CANN/operator/DeepEP packages exposing `FuseMode.MEGA_MOE` and the
current activation/bias API. Capability probes run before model loading.
This runner does not install dependencies or reinterpret W4A4 as W4A8.
The primary checkpoint is authoritative for its format; use a compatible W8A8
checkpoint in a separate invocation to cover that mode-3 combination.

Reports are a single fresh `.md` file per invocation. Source paths, commit,
tracked-diff fingerprint and loading peak-memory evidence are embedded.
Host checks do not establish target-NPU accuracy, graph replay or performance.
