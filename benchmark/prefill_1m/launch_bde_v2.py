#!/usr/bin/env python3
"""Portable launch of the archived p4_exit1_serve_dcp.sh runtime recipe."""

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def launch_config(args, inherited):
    env = dict(inherited)
    tp = args.tp or int(env.get("TP", "8"))
    dcp = args.dcp or int(env.get("DCP", "8"))
    model = args.model_path or env.get(
        "GLM_5_2_W4A8_MODEL_PATH_OVERRIDE",
        "/home/ckpt/GLM-5.2-Provide-0610-W4A8-crop-tiny",
    )
    fixed = {
        "SGLANG_NPU_INDEXER_INT8": "True",
        "SGLANG_NPU_ENABLE_DSA_CP": "True",
        "SGLANG_ENABLE_CHAT_ENCODE_POOL": "True",
        "SGLANG_CHAT_ENCODE_MEMO": "True",
        "SGLANG_CHAT_ENCODE_RAW": "True",
        "SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP": "True",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_SET_CPU_AFFINITY": "1",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "HCCL_SOCKET_IFNAME": "lo",
        "GLOO_SOCKET_IFNAME": "lo",
    }
    defaults = {
        # Restored DCP opt-ins: the archived recipe runs all three on. They
        # stay overridable (setdefault) so an attribution run can turn one
        # off from the environment without editing this file.
        "SGLANG_NPU_ENABLE_DCP_PACKED_READ": "True",
        "SGLANG_NPU_ENABLE_DCP_GATHER_PREFETCH": "True",
        "SGLANG_NPU_ENABLE_DCP_SFA_GATHER_OVERLAP": "True",
        "SGLANG_NPU_SFA_KV_8BIT": "1",
        "SGLANG_NPU_DSA_CP_TOKEN_SHARDED": "1",
        "SGLANG_NPU_DSA_CP_Q_FULL_WIDTH": "1",
        "SGLANG_NPU_DSA_CP_LOCAL_TOPK": "1",
        "SGLANG_NPU_DCP_DECODE_Q_PREISSUE": "1",
        "SGLANG_NPU_STREAM_O_PROJ": "1",
        "HCCL_BUFFSIZE": "1000",
    }
    env.update(fixed)
    for name, value in defaults.items():
        env.setdefault(name, value)
    if env.get("SAFE_ENV", "0") != "1":
        env.update(
            SGLANG_ENABLE_OVERLAP_PLAN_STREAM="1", SGLANG_NPU_USE_MULTI_STREAM="1"
        )
        env.setdefault("DEEPEP_HCCL_BUFFSIZE", "2500")
    if env.get("AG_RS", "0") == "1":
        env["SGLANG_DCP_KEEP_AG_RS"] = "1"
    for name in (
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        env.pop(name, None)
    env["NO_PROXY"] = env["no_proxy"] = "localhost,127.0.0.1,::1"
    env["PYTHONPATH"] = str(ROOT / "python") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--load-format",
        "auto" if args.real_weights else "dummy",
        "--model-path",
        str(model),
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--tp-size",
        str(tp),
        "--dcp-size",
        str(dcp),
        "--cuda-graph-backend-prefill",
        "disabled",
    ]
    if env.get("CAPTURE", "0") == "1":
        buckets = env.get("GRAPH_BS_LIST", "").split()
        maximum = int(env.get("GRAPH_BS", "64"))
        if any(int(n) < tp or int(n) % tp for n in [maximum, *buckets]):
            raise ValueError("decode graph buckets must be positive multiples of TP")
        command += ["--cuda-graph-max-bs-decode", str(maximum)]
        if buckets:
            command += ["--cuda-graph-bs-decode", *buckets]
    else:
        command += ["--cuda-graph-backend-decode", "disabled"]
    if env.get("PAGE"):
        command += ["--page-size", env["PAGE"]]
    # CTX was printed but not passed by the archive. Only an explicit option
    # adds it, so the default recipe retains checkpoint-derived context limits.
    if args.context_length is not None:
        command += ["--context-length", str(args.context_length)]
    if env.get("AG_RS", "0") == "1":
        command += ["--dcp-comm-backend", "ag_rs"]
    extra = shlex.split(env.get("EXTRA_ARGS", ""))
    command += extra
    # Packed DCP reads require one request; larger batches duplicate and reorder
    # each long prefix. Keep decode capacity independent of prefill batching.
    if not any(arg.split("=", 1)[0] == "--prefill-max-requests" for arg in extra):
        prefill_max = env.get("PREFILL_MAXREQ", "1" if dcp > 1 else None)
        if prefill_max is not None:
            command += ["--prefill-max-requests", prefill_max]
    command += [
        "--enable-metrics",
        "--trust-remote-code",
        "--mem-fraction-static",
        env.get("MEMFRAC", "0.70"),
        "--chunked-prefill-size",
        env.get("CHUNK", "32768"),
        "--moe-a2a-backend",
        "none",
        "--max-running-requests",
        env.get("MAXREQ", "64"),
        "--host",
        "127.0.0.1",
        "--port",
        env.get("PORT", "30000"),
    ]
    selected = {
        key: env[key]
        for key in sorted(
            set(fixed)
            | set(defaults)
            | {
                "DEEPEP_HCCL_BUFFSIZE",
                "SGLANG_ENABLE_OVERLAP_PLAN_STREAM",
                "SGLANG_NPU_USE_MULTI_STREAM",
                "SGLANG_DCP_KEEP_AG_RS",
            }
        )
        if key in env
    }
    return command, env, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dcp", type=int, nargs="?")
    parser.add_argument("--tp", type=int)
    parser.add_argument("--model-path")
    parser.add_argument("--context-length", type=int)
    parser.add_argument(
        "--real-weights",
        action="store_true",
        help="Load real checkpoint weights; archive default is dummy (not a quality test).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command, env, selected = launch_config(args, os.environ)
    tp = int(command[command.index("--tp-size") + 1])
    dcp = int(command[command.index("--dcp-size") + 1])
    if tp < 1 or dcp < 2 or tp % dcp:
        parser.error("TP must be positive and divisible by DCP >= 2")
    if env["SGLANG_NPU_STREAM_O_PROJ"].lower() in ("1", "true") and tp != dcp:
        parser.error(
            "streamed o_proj requires TP == DCP (the archived defaults are both 8)"
        )
    print(
        json.dumps(
            {
                "command": command,
                "environment": selected,
                "weights": "real" if args.real_weights else "dummy",
                "context": args.context_length
                or "checkpoint default; archive CTX was not passed",
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return 0
    if not Path(command[command.index("--model-path") + 1]).is_dir():
        parser.error(
            "checkpoint directory does not exist; set --model-path or GLM_5_2_W4A8_MODEL_PATH_OVERRIDE"
        )
    os.chdir(ROOT)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    raise SystemExit(main())
