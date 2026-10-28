#!/usr/bin/env bash
# Serve GLM-5.3 on Ascend A3 910C for the 1M-token prefill benchmark
# (benchmark/prefill_1m/bench_glm53_prefill_1m.py).
#
# Adapted from the bde_v2 print-build p4_exit1_serve_dcp.sh (GLM-5.2-W4A8
# DCP serving recipe): same proxy handling, env families and DCP knobs,
# retargeted to GLM-5.3 (glm_moe_dsa) with the INT quantization options
# from this branch.
#
# Usage:
#   MODEL_PATH=/path/to/GLM-5.3 bash benchmark/prefill_1m/serve_glm53_1m.sh
#   TP=16 DCP=8 bash benchmark/prefill_1m/serve_glm53_1m.sh   # DCP sweep
#   ONLINE_QUANT=w8a8_int bash benchmark/prefill_1m/serve_glm53_1m.sh
#
# Flags:
#   TP            tensor-parallel size (default 8; 16 = full A3 host)
#   DCP           --dcp-size, must divide TP (default 1 = off; try 8/16
#                 to exercise the DCP gather path with the 1M prompt)
#   CAPTURE       1 = capture decode graphs AND the NPU prefill body-capture
#                 CUDA graph; 0 = eager prefill (default, the controlled
#                 baseline; compare against CAPTURE=1 to measure the
#                 prefill-graph win at 1M)
#   ONLINE_QUANT  empty (bf16 checkpoint, no conversion) | w8a8_int |
#                 w4a4_int (mixed dense-W8A8 / MoE w13-W4A4 / w2-W8A8;
#                 GLM-5.3 is on the MoE allowlist)
#   CTX           --context-length (default 116864; raise >= 1048576 + output
#                 for the 1M bench, which needs a very large KV budget)
#   CHUNK         --chunked-prefill-size (default 32768, the cookbook's
#                 long-input starting point)
#   MEMFRAC       --mem-fraction-static (default 0.85; 1M-token KV needs
#                 most of the pool)

set -uo pipefail

unset HTTP_PROXY http_proxy HTTPS_PROXY https_PROXY ALL_PROXY all_proxy 2>/dev/null
unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy ALL_PROXY all_proxy

TP=${TP:-8}
DCP=${DCP:-1}
CAPTURE=${CAPTURE:-0}
ONLINE_QUANT=${ONLINE_QUANT:-}
MODEL=${MODEL_PATH:-/home/ckpt/GLM-5.3}
PORT=${PORT:-30000}
CTX=${CTX:-116864}
CHUNK=${CHUNK:-32768}
MEMFRAC=${MEMFRAC:-0.85}

# --- bde print-build env families (kept as-is; defaults preserved) ---------
# int8 LightningIndexer index-K pool (upstream PR #42224 port).
export SGLANG_NPU_INDEXER_INT8=${SGLANG_NPU_INDEXER_INT8:-True}
# Packed int8 SFA latent KV (656 B/row) — requires DCP > 1.
export SGLANG_NPU_SFA_KV_8BIT=${SGLANG_NPU_SFA_KV_8BIT:-False}
# DCP shared-gather path.
export SGLANG_NPU_ENABLE_DCP_SHARED_GATHER=${SGLANG_NPU_ENABLE_DCP_SHARED_GATHER:-True}
export SGLANG_NPU_DCP_GATHER_ASYNC=${SGLANG_NPU_DCP_GATHER_ASYNC:-True}
export SGLANG_NPU_DCP_SHARED_PACKED_MAX_ROWS=${SGLANG_NPU_DCP_SHARED_PACKED_MAX_ROWS:-0}
export SGLANG_NPU_DCP_SHARED_READ_PAGE_SIZE=${SGLANG_NPU_DCP_SHARED_READ_PAGE_SIZE:-128}
export SGLANG_NPU_DCP_SHARED_READ_SKIP_CONTIGUOUS=${SGLANG_NPU_DCP_SHARED_READ_SKIP_CONTIGUOUS:-False}
# Host chat-encode optimizations (model-agnostic, env-gated).
export SGLANG_ENABLE_CHAT_ENCODE_POOL=${SGLANG_ENABLE_CHAT_ENCODE_POOL:-True}
export SGLANG_CHAT_ENCODE_MEMO=${SGLANG_CHAT_ENCODE_MEMO:-True}

# --- stage-latency probes ---------------------------------------------------
# The [SGL-TS] print probes are enabled by default in this branch's import
# commit. For a clean throughput number disable them; leave them on to get
# the RECV-HTTP -> RECV-TM -> BATCH -> MODEL-FWD -> SENDBACK breakdown.

QUANT_ARGS=()
if [ -n "$ONLINE_QUANT" ]; then
  QUANT_ARGS=(--online-quantization "$ONLINE_QUANT")
fi

GRAPH_ARGS=(--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled)
if [ "$CAPTURE" = "1" ]; then
  GRAPH_ARGS=()
fi

DCP_ARGS=()
if [ "$DCP" -gt 1 ]; then
  DCP_ARGS=(--dcp-size "$DCP")
  if [ "$SGLANG_NPU_SFA_KV_8BIT" = "True" ]; then
    echo "note: SGLANG_NPU_SFA_KV_8BIT=True requires DCP > 1 (satisfied: DCP=$DCP)"
  fi
fi

exec python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --device npu \
  --attention-backend ascend \
  --tp-size "$TP" \
  "${DCP_ARGS[@]}" \
  --context-length "$CTX" \
  --chunked-prefill-size "$CHUNK" \
  --mem-fraction-static "$MEMFRAC" \
  --port "$PORT" \
  "${GRAPH_ARGS[@]}" \
  "${QUANT_ARGS[@]}"
