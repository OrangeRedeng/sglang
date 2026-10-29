#!/usr/bin/env bash
# P4 exit criterion 1 -- part 1 of 2: serve GLM-5.2-w4a8 at a chosen --dcp-size.
#
#   cd /home/tbaydasov/glm5.2_testing
#   bash p4_exit1_serve_dcp.sh 1          # leave running
#   #   in a second shell:  bash p4_exit1_gsm8k.sh 1
#   # then ^C the server and:
#   bash p4_exit1_serve_dcp.sh 16
#   #   in a second shell:  bash p4_exit1_gsm8k.sh 16    (compares automatically)
#
# WHY THE dcp1 BASELINE IS RE-MEASURED RATHER THAN REUSED
# ------------------------------------------------------
# P2 recorded GSM8K 0.940 / Invalid 0.000 at dcp_size = 1, and that number is a
# good sanity anchor -- but it is not the right thing to compare against, for
# two independent reasons.
#
#   1. It was run with graph capture ON and --chunked-prefill-size 16384. This
#      run has capture OFF (see below) and chunking disabled. Comparing across
#      those is comparing two measurements, which is exactly the mistake the P2
#      eval script warns about in its own header.
#   2. It kept only the aggregate. P4's real instrument is the PAIRED
#      per-question comparison -- same 1319 questions, which ones flipped and in
#      which direction -- and that needs --raw-result-file from both configs.
#      An aggregate difference at n=200 has a standard error of about 1.7 points
#      per run, so two runs differ by ~5 points at 2 sigma with nothing wrong.
#      Paired, the same evidence is roughly an order of magnitude sharper,
#      because every question is its own control.
#
# So: two servers, same script, same flags, differing in ONE argument. The cost
# is a second 387 GiB load. That is the price of a controlled comparison and it
# is worth paying once.
#
# WHY GRAPH CAPTURE IS OFF FOR BOTH
# ---------------------------------
# Every DCP run in this project so far has been decode-graph-off, on
# purpose, to keep one variable at a time. Capture at dcp16 is untested and is
# its own P4 step: the DCP decode path does an all-gather and an all-to-all
# INSIDE attention, and whether those capture on this backend is an open
# question. If capture fails, it must not take the accuracy measurement down
# with it. Run this step capture-off, get the parity number, then re-run with
# CAPTURE=1 as a separate step and compare dcp16-off against dcp16-on.
#
# WHY CHUNKED PREFILL IS DISABLED FOR BOTH
# ----------------------------------------
# --chunked-prefill-size -1 keeps each request in a single forward. This is NOT
# cosmetic: at --parallel 64 with ~1.2k-token 5-shot prompts, a 16384 budget
# means the scheduler chunks whichever request straddles the boundary, so the
# accuracy run would silently exercise chunked prefill -- the one piece of P6
# that is still open (the context gather and order restore). That belongs in its
# own run with its own verdict, not as an uncontrolled variable inside the
# parity number. Batches stay bounded by --max-prefill-tokens regardless, so
# this does not mean 64 prompts in one forward.
#
# To test chunking deliberately afterwards:  CHUNK=2048 bash p4_exit1_serve_dcp.sh 16
#
# IF IT WILL NOT LAUNCH AT dcp16, TRY SAFE_ENV=1 BEFORE SUSPECTING THE PORT
# ------------------------------------------------------------------------
# The env below is P2's baseline recipe, which is the right control. But the
# only environment ever proven to work at dcp16 is p3b_logprob_parity.sh's,
# which omits the two stream flags and DEEPEP_HCCL_BUFFSIZE. SAFE_ENV=1 falls
# back to that set. If SAFE_ENV=1 launches and the default does not, the finding
# is about multi-stream or DeepEP buffer sizing under DCP, not about attention
# -- and both configs must then be re-run with SAFE_ENV=1 to stay paired.

set -uo pipefail

# Kill any corporate proxy for this shell. The internal 127.0.0.1 warmup and the
# bench/profile HTTP calls are local; with a proxy in the env (e.g. 70.251.86.90:
# 8080) requests.get("http://127.0.0.1:30000/model_info") goes off-network, times
# out, and the warmup retries / scheduler watchdog take the run down. no_proxy
# alone only helps if the client matches the host, so unset is the real fix.
unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy ALL_PROXY all_proxy

# Positional first, then $DCP, then 16. Every other knob here is an env var, so
# `DCP=1 bash p4_exit1_serve_dcp.sh` is the natural thing to type -- and with a
# bare ${1:-16} it silently launched dcp16 anyway. Positional still wins, so
# existing `... .sh 1` invocations are unchanged.
DCP=${DCP:-8}
DESC=${DESC:-"NA"}
# Device count. 16 is the A3's full complement; TP=8 is for when only half the
# box is free. attn_tp_size follows TP, so the capture-bucket rule below is
# stated in terms of it and the two move together. --dcp-size must divide it.
TP=${TP:-8}
MODEL=${GLM_5_2_W4A8_MODEL_PATH_OVERRIDE:-/home/ckpt/GLM-5.2-Provide-0610-W4A8-crop-tiny}
#MODEL=${GLM_5_2_W4A8_MODEL_PATH_OVERRIDE:-/home/ckpt/GLM-5.2-Provide-0610-W4A8-crop-tiny}
PORT=${PORT:-30000}
CTX=${CTX:-32768}
CHUNK=${CHUNK:-32768}
CAPTURE=${CAPTURE:-0}
MAXREQ=${MAXREQ:-64}
PAGE=${PAGE:-}
MEMFRAC=${MEMFRAC:-0.70}


#. /home/krodyushkin/cann_kr2/cann/set_env.sh
#. /home/krodyushkin/cann_kr2/nnal/atb/set_env.sh
#. /home/krodyushkin/cann_kr2/nnal/asdsip/set_env.sh



#export SGLANG_NPU_ENABLE_DSA_INDEXER_QUERY_SHARDING=True
#export SGLANG_NPU_ENABLE_DSA_CP=True
#xport SGLANG_DEBUG_NPU_DCP_SHARED=1
#A 
export SGLANG_NPU_ENABLE_DCP_SHARED_GATHER=True
export SGLANG_NPU_DCP_GATHER_ASYNC=True

#V3
export SGLANG_NPU_DCP_SHARED_PACKED_MAX_ROWS=0
export SGLANG_NPU_DCP_SHARED_READ_PAGE_SIZE=128
export SGLANG_NPU_DCP_SHARED_READ_SKIP_CONTIGUOUS=False

#B_fix (upstream PR #42224, ported): int8 LightningIndexer index-K pool,
#fp16 op weights, custom_ops skipped while it is on.
export SGLANG_NPU_INDEXER_INT8=True
#C
#nothing

#F  (SFA int8 main KV-cache, A2/A3): store the DSA latent KV as the 656 B
#    packed INT8 row (npu_dynamic_quant int8) instead of 1152 B bf16 -- ~x1.76
#    pool capacity at fixed GiB. The model still computes in bf16; serving reads
#    dequant on read (the DCP gather paths). Decoupled from --kv-cache-dtype
#    (stays bf16) and from SGLANG_NPU_INDEXER_INT8 (the index-K keeps its own
#    dtype). Requires --dcp-size >= 2 (the pool refuses the flag at init
#    without DCP). Default ON (this is the config under test); set
#    SGLANG_NPU_SFA_KV_8BIT=0 for the paired bf16 baseline.
export SGLANG_NPU_SFA_KV_8BIT=${SGLANG_NPU_SFA_KV_8BIT:-1}

# Use the reviewed query-sharding path. Legacy residual sharding and
# streamed o_proj are rejected until they have a compatible attention consumer.
export SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD=${SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD:-1}
export SGLANG_NPU_ENABLE_DCP_SFA_GATHER_OVERLAP=${SGLANG_NPU_ENABLE_DCP_SFA_GATHER_OVERLAP:-0}
export SGLANG_NPU_ENABLE_DCP_PACKED_READ=${SGLANG_NPU_ENABLE_DCP_PACKED_READ:-0}
export SGLANG_NPU_ENABLE_DCP_GATHER_PREFETCH=${SGLANG_NPU_ENABLE_DCP_GATHER_PREFETCH:-0}

#E1 (branch E, clean): off-loop render+encode pool in the TokenizerManager.
export SGLANG_ENABLE_CHAT_ENCODE_POOL=True
#E2a (branch E, clean): render+encode memo (full-prompt LRU + family prefix
#assembly). Must stay on for E2d to be beneficial.
export SGLANG_CHAT_ENCODE_MEMO=True
#E2d (branch E, clean): raw Rust encode (tokenizer._tokenizer.encode)
#bypassing the HF wrapper. Requires the E2a memo.
export SGLANG_CHAT_ENCODE_RAW=True
#Recommended with branch E: the E1 pool + E2a memo tokenize faster than the
#base path, so the burst of ready requests hits the scheduler faster than the
#consecutive-prefill overlap window drains; disabling overlap removes the ~1s
#first-token hold on min TTFT.
export SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP=True
#E2e (direct-compiled render) is NOT part of this combined branch (clean E
#only has E1/E2a/E2d), so this flag does not exist in the tree -- left off.
#export SGLANG_CHAT_ENCODE_FAST_RENDER=True


DATE_NAME=$(date +"%F-%H%M%S")
#export 
MN=`basename ${MODEL}`
SGLANG_TORCH_PROFILER_DIR=/home/krodyushkin/sglang-code/_tamir_dcp_${DATE_NAME}_docker_server_${MN}_TP${TP}DCP${DCP}_CPS${CHUNK}_P${PORT}_${DESC}
mkdir ${SGLANG_TORCH_PROFILER_DIR}
echo "SGLANG_TORCH_PROFILER_DIR=${SGLANG_TORCH_PROFILER_DIR}" > profile_dir_name_${PORT}.sh

LOG=${SGLANG_TORCH_PROFILER_DIR}/_server.log
echo "bash $0 $@" 2>&1 | tee -a ${LOG}
cp $0 ${SGLANG_TORCH_PROFILER_DIR}
cd /home/z00600541/sglang/branch_D/sglang/
echo ===GIT LOG=== /home/krodyushkin/sglang/
git log -n 5 2>&1 | tee -a ${LOG}
echo ===GIT DIFF=== /home/krodyushkin/sglang/
git diff 2>&1 | tee -a ${LOG}
cd -
npu-smi info 2>&1 | tee -a ${LOG}
env 2>&1 | tee -a ${LOG}
pip list -v | tee -a ${LOG}

#LOG=${LOG:-p4_gsm8k_dcp${DCP}_server.log}

if [ ! -d "$MODEL" ]; then
  echo "model not found: $MODEL"  2>&1 | tee -a "$LOG"
  echo "set GLM_5_2_W4A8_MODEL_PATH_OVERRIDE to the checkout"  2>&1 | tee -a "$LOG"
  exit 2
fi

# Per-shell, and it recurs in every fresh shell. The sgl_kernel_npu wheel is
# built against a newer libstdc++ than the system one found first, and
# mem_cache/pool_host/mha.py imports it at MODULE scope, so launch_server dies
# on import long before it loads a weight.
if [ -n "${CONDA_PREFIX:-}" ] && [ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]; then
  export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}$CONDA_PREFIX/lib/libstdc++.so.6"
fi
if ! python -c "import sgl_kernel_npu" 2>/dev/null; then
  echo "sgl_kernel_npu still will not import -- launch_server would die on import."  2>&1 | tee -a "$LOG"
  echo "Check: strings \$CONDA_PREFIX/lib/libstdc++.so.6 | grep GLIBCXX_3.4.29"  2>&1 | tee -a "$LOG"
  echo "If absent: conda install -c conda-forge libstdcxx-ng"  2>&1 | tee -a "$LOG"
  exit 2
fi

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_SET_CPU_AFFINITY=1
# Per-communicator HCCL scratch, in MB, and it does NOT scale with
# --mem-fraction-static. That makes it a candidate for the width wall: captured
# dcp16 is correct at batch 16/32/48 and dead at 64, freeing 10 GiB of HBM
# changed nothing, and only dcp16 pushes a num_local_heads*dcp_size query and a
# same-width merge through HCCL on each of 78 layers. Per layer at batch 64 that
# is ~4.7 MB gathered plus ~4.2 MB merged, against ~3.5 + 3.1 at batch 48.
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1000}
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
# AG_RS=1 holds the DCP merge on the all-gather/reduce-scatter path instead of
# the fused all-to-all. It is the cut that separates the merge collective from
# everything else the DCP decode branch does, which is what a capture-only
# defect needs. No flag: ag_rs is the field's own default, so only the env var
# can express "I meant it" -- see _dcp_comm_backend_default. Three collectives
# per layer against a2a's one, so never leave it on for a timed run.
if [ "${AG_RS:-0}" = "1" ]; then
  export SGLANG_DCP_KEEP_AG_RS=1
fi
if [ "${SAFE_ENV:-0}" != "1" ]; then
  export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
  export SGLANG_NPU_USE_MULTI_STREAM=1
  export DEEPEP_HCCL_BUFFSIZE=${DEEPEP_HCCL_BUFFSIZE:-2500}
elif [ "$HCCL_BUFFSIZE" -gt 1000 ]; then
  # Without DEEPEP_HCCL_BUFFSIZE, DeepEP inherits HCCL_BUFFSIZE -- and HCCL
  # reserves TWICE it. HCCL_BUFFSIZE=4000 under SAFE_ENV asked
  # aclnnMoeLowLatencyDispatchV2 for 8389656576 bytes (exactly 2 x 4000 MiB) and
  # died with Memory_Allocation_Failure EL0004. Raise one or run SAFE_ENV, not both.
  echo "SAFE_ENV=1 with HCCL_BUFFSIZE=$HCCL_BUFFSIZE: DeepEP inherits it and"  2>&1 | tee -a "$LOG"
  echo "reserves 2x that per die, which OOMs. Leave HCCL_BUFFSIZE at 1000 here,"  2>&1 | tee -a "$LOG"
  echo "or set DEEPEP_HCCL_BUFFSIZE explicitly to cap the dispatch separately."  2>&1 | tee -a "$LOG"
  exit 2
fi

# Capture has to cover the batch you intend to measure or the measurement runs
# eager anyway: a batch above the decode capture ceiling falls back to the eager
# path silently, which would look like "capture bought nothing". Default it to
# the batch width rather than to the P2 recipe's 16.
GRAPH_BS=${GRAPH_BS:-$MAXREQ}
# Graph flags are per phase since the merge with main. --cuda-graph-max-bs no
# longer exists, and because sglang's parser accepts abbreviations it is now an
# ambiguous prefix of --cuda-graph-max-bs-{decode,prefill}: argparse refuses the
# launch. --disable-cuda-graph survives only as a deprecated alias. Decode is the
# only phase to set -- NPU and dcp_size > 1 each force prefill capture off
# (cuda_graph_hook.py:185-251) -- and /server_info reports the result under
# cuda_graph_config.decode.backend, which the eval scripts read back.
GRAPH_ARG=(--cuda-graph-backend-decode disabled)
if [ "$CAPTURE" = "1" ]; then
  # Capture batch sizes must be multiples of attn_tp_size -- $TP here.
  # get_cuda_graph_batch_size_alignment (utils/common.py:3914-3922) multiplies by
  # attn_tp_size whenever a gathered buffer is required, and --moe-a2a-backend
  # deepep requires one. A non-multiple empties the candidate list and the server
  # dies at startup with `AssertionError: capture_bs=[]`, naming neither the flag
  # nor the rule. So the earlier advice to drop GRAPH_BS to 16 on an OOM was
  # right by luck; 8 would not have been.
  if [ $((GRAPH_BS % TP)) -ne 0 ] || [ "$GRAPH_BS" -lt "$TP" ]; then
    echo "GRAPH_BS=$GRAPH_BS must be a positive multiple of $TP at tp$TP."  2>&1 | tee -a "$LOG"
    exit 2
  fi
  GRAPH_ARG=(--cuda-graph-max-bs-decode "$GRAPH_BS")
  # Capture an EXPLICIT bucket list instead of everything up to GRAPH_BS. Every
  # run so far captured [16, 32, 48, 64] together, which leaves two readings of
  # the width wall tangled: bs=64 is broken, or capturing the narrower shapes
  # breaks it. They share the graph_metadata buffers through [:bs, :] views, so
  # the second is not far-fetched. GRAPH_BS_LIST="64" captures only 64.
  # Same multiple-of-$TP rule applies to every entry.
  if [ -n "${GRAPH_BS_LIST:-}" ]; then
    for _bs in $GRAPH_BS_LIST; do
      if [ $((_bs % TP)) -ne 0 ] || [ "$_bs" -lt "$TP" ]; then
        echo "GRAPH_BS_LIST entry $_bs must be a positive multiple of $TP at tp$TP."  2>&1 | tee -a "$LOG"
        exit 2
      fi
    done
    GRAPH_ARG+=(--cuda-graph-bs-decode $GRAPH_BS_LIST)
  fi
fi

PAGE_ARG=()
if [ -n "$PAGE" ]; then
  PAGE_ARG=(--page-size "$PAGE")
  # validation_hook.py:125 asserts chunked_prefill_size % page_size == 0, but
  # only when it is > 0. CHUNK is -1 here, so the two do not interact -- worth
  # knowing before anyone re-enables chunking alongside a non-default page.
fi

# No prefill context parallelism here, deliberately. NPU prefill CP was removed
# upstream (85d39401c8, #38293): --enable-prefill-cp raises "Prefill CP on
# HIP/NPU/MUSA is deprecated" at argument resolution (parallel_hook.py:640), and
# the --enable-nsa-prefill-context-parallel flag this script used to pass is gone.
if [ -n "${CP:-}" ] && [ "$CP" != "1" ]; then
  echo "CP=$CP: prefill context parallelism is deprecated on NPU in this sglang"  2>&1 | tee -a "$LOG"
  echo "tree and the server would refuse to start. Unset CP."  2>&1 | tee -a "$LOG"
  exit 2
fi

echo "model   : $MODEL"  2>&1 | tee -a "$LOG"
echo "dcp     : $DCP   (tp $TP)"  2>&1 | tee -a "$LOG"
echo "ctx     : $CTX   chunked-prefill $CHUNK   graph capture $([ "$CAPTURE" = 1 ] && echo "ON (max-bs $GRAPH_BS)" || echo OFF)"  2>&1 | tee -a "$LOG"
echo "batch   : max-running-requests $MAXREQ   (pass --parallel $MAXREQ to the eval)"  2>&1 | tee -a "$LOG"
echo "page    : ${PAGE:-128 (default)}   -> radix prefix matching in $(( ${PAGE:-128} * DCP ))-token units"  2>&1 | tee -a "$LOG"
echo "memfrac : $MEMFRAC   (lower leaves more room for the captured graph pool)"  2>&1 | tee -a "$LOG"
echo "safeenv : ${SAFE_ENV:-0}   merge $([ "${AG_RS:-0}" = 1 ] && echo "ag_rs (diagnostic)" || echo a2a)"  2>&1 | tee -a "$LOG"
echo "log     : $LOG"  2>&1 | tee -a "$LOG"
echo  2>&1 | tee -a "$LOG"
echo "the eval script checks /server_info and refuses to score a mismatched dcp_size,"  2>&1 | tee -a "$LOG"
echo "so a leftover server on this port cannot be scored as this one."  2>&1 | tee -a "$LOG"
# Extra launch flags, word-split, e.g. EXTRA_ARGS="--watchdog-timeout 7200" for a
# profiled run: the scheduler watchdog kills the server when one forward batch
# runs longer than 300 s, and a profiled 1M prefill chunk is the closest thing to it.
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"  2>&1 | tee -a "$LOG"
echo "extra   : ${EXTRA_ARGS:-none}"  2>&1 | tee -a "$LOG"
echo


echo "bash $0 $@" 2>&1 | tee -a ${LOG}
cp $0 ${SGLANG_TORCH_PROFILER_DIR}


cd /home/krodyushkin/sglang/
echo ===GIT LOG=== /home/krodyushkin/sglang/
git log -n 5 2>&1 | tee -a ${LOG}
echo ===GIT DIFF=== /home/krodyushkin/sglang/
git diff 2>&1 | tee -a ${LOG}
cd -
npu-smi info 2>&1 | tee -a ${LOG}
env 2>&1 | tee -a ${LOG}
pip list -v | tee -a ${LOG}


echo GRAPH_ARG "${GRAPH_ARG[@]}"  2>&1 | tee -a "$LOG"

PARAMS="--load-format dummy"
#  --context-length "$CTX" \

PARAMS="${PARAMS} --model-path ${MODEL} --attention-backend ascend --device npu"
PARAMS="${PARAMS} --tp-size ${TP} --dcp-size ${DCP} "
PARAMS="${PARAMS} ${GRAPH_ARG[@]} ${PAGE_ARG[@]} ${EXTRA[@]}"
PARAMS="${PARAMS} --enable-metrics --trust-remote-code "
PARAMS="${PARAMS} --mem-fraction-static ${MEMFRAC} --chunked-prefill-size ${CHUNK}"
#PARAMS="${PARAMS} --moe-a2a-backend deepep --deepep-mode auto --max-running-requests ${MAXREQ}"
PARAMS="${PARAMS} --moe-a2a-backend none --max-running-requests ${MAXREQ}"
PARAMS="${PARAMS} --host 127.0.0.1 --port ${PORT}"


export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="${no_proxy}"

echo "python -m sglang.launch_server ${PARAMS}" 2>&1 | tee -a "$LOG"
python -m sglang.launch_server ${PARAMS} 2>&1 | tee -a "$LOG"
