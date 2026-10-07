# NPU automatic configuration and measured tuning

All automatic behavior is opt-in. Existing feature flags retain their defaults.
No operator kernels are added, and router output dtype and quantization scale
algorithm remain explicit accuracy choices.

Selection order is explicit CLI/environment, matching cached profile,
deterministic calculation, then the existing default. Each automatic decision
logs the value, source, inputs and fallback. `SGLANG_NPU_AUTOTUNE_DRY_RUN=1`
logs proposed settings without applying them or writing measured winners.
Internal context propagation and an explicitly requested context export still occur.

## Startup configuration

| Environment variable | Default | Behavior |
| --- | --- | --- |
| `SGLANG_NPU_AUTO_HCCL_BUFFSIZE` | `0` | Resolve before scheduler launch and HCCL initialization. |
| `HCCL_BUFFSIZE` | runtime default | Explicit values always win, including in auto mode. |
| `SGLANG_NPU_HCCL_HEADROOM` | `1.25` | Collective-size multiplier, must be at least one. |
| `SGLANG_NPU_HCCL_QUANTUM_MB` | `32` | Round up in MiB. |
| `SGLANG_NPU_HCCL_MIN_MB` / `SGLANG_NPU_HCCL_MAX_MB` | `64` / `1024` | Clamp the estimate; a clamp can be smaller than the collective. HCCL may split the transfer. |
| `SGLANG_NPU_TUNING_PROFILE` | `0` | Consume matching buffer, expansion, task queue, layout and threshold entries. Does not enable experimental features. |
| `SGLANG_NPU_TUNING_CACHE_DIR` | `~/.cache/sglang/npu_tuning` | Profile directory. |
| `SGLANG_NPU_AUTOTUNE_DRY_RUN` | `0` | Print proposed decisions without applying them. |
| `SGLANG_NPU_TUNING_CONTEXT_FILE` | unset | Export the startup context as JSON for the restart tuner. |
| `SGLANG_NPU_TUNING_DEVICE` / `SGLANG_NPU_TUNING_ARCH` | discovery / unknown | Override hardware identity, e.g. `Ascend 950` / `arch35`. Discovery uses a separate `npu-smi` process and recognizes 950. |
| `SGLANG_NPU_TUNING_CANN_VERSION` | toolkit `version.cfg` / unknown | Override runtime identity, e.g. `9.2.beta1`. |
| `SGLANG_NPU_AUTO_STREAM_THRESHOLDS` | `0` | Consume calibrated thresholds; preserve zero if unavailable. |
| `SGLANG_NPU_AUTO_DCP_EXTEND_GATHER_PIECE_ROWS` | `0` | Size DCP gather scratch from actual packed width or separate BF16 row widths. |
| `SGLANG_NPU_DCP_SCRATCH_BUDGET_MB` | `256` | Scratch cap in MiB, additionally bounded by half the smallest rank's free device memory. |
| `SGLANG_NPU_DCP_EXTEND_GATHER_PIECE_ROWS` | `262144` | Existing explicit integer wins over auto, including zero (whole-prefix mode). |
| `SGLANG_NPU_MEMORY_DIAGNOSTICS` | `0` | Log allocator/KV/weight observations after KV pool allocation. Never alters `mem_fraction_static`. |
| `SGLANG_NPU_TP_MOE_MXFP8_GATE_WEIGHT_LAYOUT` | `transposed` | Existing modes plus `auto`; benchmarks transposed/contiguous/NZ. |
| `SGLANG_NPU_TP_MOE_MXFP8_GATE_TOPK_LAYOUT` | `default` | Existing modes plus `auto`; benchmarks default/contiguous/ND. `clone` remains explicit diagnostic mode. |

HCCL uses `T * hidden_size * activation_element_size` with
`T = min(chunked_prefill_size, max_prefill_tokens)` (unchunked uses max prefill).
BF16/FP16 use two bytes; FP32 uses four. For 16,384 rows and H=6,144 BF16,
the collective is 192 MiB; 1.25 headroom and 32 MiB rounding yield 256 MiB.
These are arithmetic results, not measured performance conclusions.

DCP calculates **local** aligned piece rows, then passes **gathered** rows to
the existing planner. The budget includes DCP size, one/two prefetch scratch
slots, the actual packed FP8 KV record (including scales), or both separate
KV tensors, and this chunk's own KV appended to gather scratch. Alignment is
the attention page/ownership quantum. Insufficient space for one aligned cycle
raises an error rather than exceeding the budget. A prefix that fits becomes
one piece and can use the existing opt-in prefetch. Context output buffers and
local send buffers are additional memory; the scratch budget covers scratch
only. Rank free-memory minima are reduced once per forward before planning so
all ranks choose identical collective shapes.

The following thresholds default to zero and preserve existing behavior.
Positive values engage the corresponding feature only at or above that many
current token rows. Existing boolean feature flags remain required.

| Environment variable | Profile entry |
| --- | --- |
| `SGLANG_NPU_DSA_OVERLAP_QNOPE_ROPE_MIN_TOKENS` | `qnope_rope_overlap_min_tokens` |
| `SGLANG_NPU_DSA_EAGER_INDEXER_MIN_TOKENS` | `eager_indexer_min_tokens` |
| `SGLANG_NPU_TP_MOE_EAGER_MULTI_STREAM_MIN_TOKENS` | `moe_eager_multi_stream_min_tokens` |
| `SGLANG_NPU_DSA_INDEXER_QUERY_SHARDING_MIN_TOKENS` | `indexer_sharding_min_tokens` |
| `SGLANG_NPU_DSA_CP_MIN_TOKENS` | `dsa_cp_min_tokens` |

Indexer sharding and DSA-CP use independent thresholds. The indexer metadata
planner and executor consult the same sharding threshold. Stream thresholds
retain existing graph/capture, topology and correctness exclusions. Thresholds
are consumed from calibrated profiles; this tool does not fabricate crossover
values or calibrate them automatically.

## Gate layout microtuning

Request either/both layout variables as `auto` with the native gate feature
explicitly enabled. Rank zero benchmarks the first loaded gate's **real**
quantized weights/scales using deterministic synthetic BF16 activations, and
broadcasts its selection to TP peers through their CPU group. Other sparse
layers reuse the selected shape/configuration. The primary serving shape
`min(chunked_prefill_size, max_prefill_tokens)` has weight 0.9 and a second
`min(1024, primary)` point has weight 0.1.

Candidates time QuantMatmul, FP32 routing-boundary conversion, layout transform
and the actual TopK operation together, with 5 warmups, 20 measured synchronized
iterations and medians. Dtype is fixed throughout. Candidates must exactly match
baseline logits, ordered routed IDs and route weights at both synthetic points.
The existing default is retained if improvement is under 1% or smaller than the
measured spread. Unsupported operators/layouts fall back with a log.

This startup check does **not** establish real-workload routing accuracy,
final-layer accuracy or generation equivalence. Before production rollout,
compare real captured inputs: ordered and unordered TopK matches, top-1,
changed/total routes, routing weight error, final BF16 output error and generation
match. `bench_npu_tp_moe_norm_gate.py` already provides real input replay and
optional captured output comparisons. BF16 router dtype is never included in
layout search; select and validate it separately.

## Profile schema and invalidation

Files are named `<sha256>.json`. Schema version 1 contains `schema_version`,
`context`, `selected` and `evidence`. `context` serializes `NpuTuningContext`;
its canonical sorted JSON determines the hash. It includes device/arch, CANN,
torch_npu, SGLang commit/version, architecture and dimensions, expert count,
TopK, model quantization/router and experiment signatures, topology, serving shape, dtype/KV
dtype, page size, gate output dtype and scale algorithm. Absolute model paths
are excluded. Unsupported schema, invalid entries and mismatched contexts
are ignored with a warning. Incomplete hardware/runtime identity disables
cache consumption. Set the documented identity overrides if discovery cannot
read the installed toolkit's version file.

Delete the matching file or use a fresh `SGLANG_NPU_TUNING_CACHE_DIR` to invalidate.
Commit/version changes also invalidate automatically. Local uncommitted source
changes are not represented by the commit; use a fresh cache after such edits.
The listed NPU feature flags enter the experiment signature; other launch changes
likewise warrant a fresh cache.
Atomic writes avoid partial profiles. Use one tuner writer per profile directory;
concurrent independent jobs writing the same key are not merged transactionally.

Profile fields include `hccl_buffsize_mb`, `hccl_op_expansion_mode`,
`task_queue_enable`, `gate_weight_layout`, `gate_topk_layout`, and the threshold
entries above. Restart results also record `page_size`, `chunked_prefill_size`,
`max_running_requests` when requested, and reusable `restart_evidence`.
Page/chunk/concurrency winners must be supplied through explicit CLI on replay;
core startup does not rewrite those resolved CLI fields. The published profile
is keyed to the winning page/chunk/concurrency context. This prevents a profile
measured at a different page/chunk shape from being applied accidentally.

## Restart tuning

The server-arguments file contains shell-style CLI arguments (not shell commands).
The benchmark/correctness/analyzer files are executable Bash scripts passed to
`bash`; they may use `AUTOTUNE_PORT`, `AUTOTUNE_RESULT_DIR`,
`AUTOTUNE_RUN_INDEX`, `AUTOTUNE_BENCHMARK_LOG` and `PROFILE_DIR`.
Each benchmark must print SGLang's `Mean TTFT (ms):` and/or `Median TTFT (ms):`.
A correctness script must exit nonzero for mismatched routing/output/generation.
Use `--expected-requests` to reject partial completion; differing completion counts
are excluded from ranking. Publishing a cache requires that script. Without it, results are exploratory.

An optional analyzer writes `analyzer.json` into `AUTOTUNE_RESULT_DIR` after each
run with `forward_mean_ms`, `forward_median_ms`, `profiler_buckets` and/or memory
observations. Use `PROFILE_DIR` for profiler captures. A TTFT winner whose
reported steady forward mean/median regresses is rejected. Unavailable analyzer
metrics remain absent; they are not inferred from wall time.

```bash
python benchmark/kernels/autotune_npu_prefill.py \
  --model-path "$MODEL_PATH" \
  --server-args-file /tmp/server_args.txt \
  --benchmark-command-file /tmp/bench.sh \
  --correctness-command-file /tmp/check_accuracy.sh \
  --context-file /tmp/npu_context.json \
  --metric median_ttft --expected-requests 17 \
  --output /tmp/a5_tuning.json \
  --publish-profile
```

Export the context from a matching startup with
`SGLANG_NPU_AUTO_HCCL_BUFFSIZE=1 SGLANG_NPU_AUTOTUNE_DRY_RUN=1
SGLANG_NPU_TUNING_CONTEXT_FILE=/tmp/npu_context.json`.
Use `--dry-run` on the restart tool to write its search plan without starting a
server. `--help` works without torch/NPU dependencies.

Default search is staged: expansion unset/AIV, buffers around the deterministic
estimate (0.5/0.75/1/1.5/2 times, aligned/clamped), task queue unset/2, page 64/128.
Each stage includes the current baseline. An unset `HCCL_BUFFSIZE` baseline
remains absent from the child environment; no runtime default is guessed. For
16K rows / H6144 BF16 the buffer stage compares unset, 128, 192, 256, 384 and
512 MiB. With `HCCL_BUFFSIZE=1000`, the stage is skipped unless `--tune-explicit`
is supplied, in which case 1000 is the baseline. The tool carries the conservative
winner into the next stage. No Cartesian product is formed. Explicit env/CLI settings
are skipped unless `--tune-explicit` explicitly authorizes varying them.

Use `--stages expansion buffer task_queue page chunk --chunk-max 32768` to
include half/current/double chunk sizes within the declared supported bound.
Concurrency is included only with `--stages ... concurrency` and
`--concurrency-candidates ...`. The tool never infers serving policy.

Servers run in their own process groups, wait for `/health`, clear prefix cache
between trials, and receive TERM followed by KILL if necessary during cleanup.
An occupied port is rejected. Startup/benchmark timeouts are configurable.
Logs, per-case settings, TTFT, optional analyzer output, wall time and OOM/failure
are retained beside the output JSON. Coarse comparisons use at least two trials;
close differences under 1% receive at least three. Variance larger than the gain
preserves the baseline. Valid cached identical cases can be reused. Match the
model, all topology/runtime settings and experimental flags when reusing evidence.

## GLM-5.2 TP4 long prefill example

Use the target CANN 9.2.beta1 / torch_npu 2.10.0.post6 environment. Keep the
existing validated launch flags in `/tmp/server_args.txt`, including
`--device npu --tp-size 4 --ep-size 1 --chunked-prefill-size 16384` and the
existing native norm/gate/quantization settings in the environment.
`/tmp/bench.sh`:

```bash
python -m sglang.benchmark.serving \
  --port "$AUTOTUNE_PORT" \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 1 --gsp-prompts-per-group 17 \
  --gsp-system-prompt-len 200000 --gsp-question-len 10828 \
  --gsp-output-len 1 --profile --profile-output-dir "$PROFILE_DIR"
```

The objective is median TTFT. Do not select a winner from aggregate wall time
when steady forward time regresses. Supply a correctness script comparing the
same requests with an accepted reference and a profiler analyzer if available.
The framework does not have a built-in reference for arbitrary models.

## Additional integration and target validation

Native MXFP8 norm/gate now admits the existing explicitly enabled fused shared
expert path while retaining its topology/quantization restrictions. The gate
still emits 256 routed logits. Existing fused TopK appends shared ID 256 with
weight exactly 1 to routed K8, producing E257/K9. Existing weight remapping loads
the shared expert into slot 256; the separate MLP is absent and no separate
shared stream is constructed. This compatibility change is experimental until
real shared-weight loading, routed IDs, unit shared weight, final BF16 output,
and absence of missing/duplicated contribution are checked on A5. Compare
native norm/gate plus separate overlap against native norm/gate plus fused E257/K9.
No automatic shared-strategy selection is enabled.

`python benchmark/kernels/probe_npu_hccl_options.py` inspects installed options
fields and tries the existing branch's `hccl_config={"hccl_buffer_size":256}`
assignment without creating a communicator. The branch already applies this
schema to default/MoE/DCP groups. Assignment acceptance alone cannot prove the
runtime honors it or establish send/receive allocation counts. Global HCCL sizing
remains the fallback. TP4 reserve accounting needs actual communicator/allocation
measurements; the diagnostic prints unknown counts/headroom instead of multiplying
by every Python process group.

MLAProlog EXTEND remains excluded: the native prolog writes paged KV directly and
returns cache tensors plus normalized low-rank query and dynamic scale; EXTEND
uses freshly produced per-token KV and an explicit gathered context, including
DCP ownership and packed FP8 records. Reusing it requires a separate validated
cache-write/gather adaptation and DSA-CP/query-sharding checks. Removing the
existing mode exclusion alone is insufficient. Shared strategy auto and shared
GMM1 auto likewise remain deferred until clean correctness/performance A/B data
exists; existing strategy/GMM1 defaults are preserved.

Production rollout requires target measurements. Deterministic HCCL sizing and
explicit calibrated thresholds are conservative opt-in configuration tools;
layout microbenchmarks, fused-shared compatibility and restart-selected AIV/task
queue/page/chunk changes need matched correctness and performance acceptance.
No NPU performance or numerical claim is established by CPU tests.

CPU verification:

```bash
python test/registered/unit/npu/test_npu_autotune.py
```

Restart CLI reference:

| Option | Meaning/default |
| --- | --- |
| `--model-path` | Required checkpoint; never part of the cache key. |
| `--server-args-file` | Required CLI argument file. |
| `--benchmark-command-file` | Required Bash benchmark file. |
| `--correctness-command-file` | Acceptance script, required for profile publication. |
| `--analyzer-command-file` | Optional profiler/forward analyzer script. |
| `--context-file` | Required matching startup context JSON. |
| `--metric` | `median_ttft` (default) or `mean_ttft`. |
| `--output` | Required JSON result path; adjacent directory retains artifacts. |
| `--port` / `--host` | `30088` / `127.0.0.1`. |
| `--python` | Current interpreter; must have the target SGLang/runtime installed. |
| `--stages` | Defaults to `expansion buffer task_queue page`. |
| `--chunk-max` | Declared supported upper chunk bound for optional chunk stage. |
| `--concurrency-candidates` | Explicit serving-policy candidates for optional concurrency stage. |
| `--runs` | At least 2; default 2, close cases receive at least 3. |
| `--expected-requests` | Optional required completion count (17 for the example). |
| `--startup-timeout` / `--benchmark-timeout` | 900 / 3600 seconds. |
| `--tune-explicit` | Authorize varying user-set values in requested stages. |
| `--publish-profile` | Publish accepted selections under the winning context key. |
| `--dry-run` | Write a search plan without starting servers. |
