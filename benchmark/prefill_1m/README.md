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
