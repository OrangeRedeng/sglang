#!/usr/bin/env python3
"""Run the GLM-5.3 A3 integration checks, with separate host and device evidence."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import importlib.metadata
import importlib.util
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

# These suites include their own capability/skip checks. Keep skipped tests visible.
UNIT_SUITES = {
    "a3-regressions": ["unit/npu/quantization/test_glm_a3_regressions.py"],
    "scheduler-probes": ["unit/managers/test_scheduler_stage_latency.py"],
    "int-contracts": [
        "unit/npu/quantization/test_fused_norm_quant_contracts.py",
        "unit/layers/moe/test_ascend_tp_optimizations.py",
    ],
    "tp-routing-shared": [
        "unit/npu/quantization/test_tp_moe_routing_weights.py",
        "unit/npu/quantization/test_tp_moe_shared_weight_views.py",
        "unit/layers/moe/test_npu_topk_l1_norm.py",
    ],
    "dcp-layout-owner": [
        "dcp/test_dcp_block_tables.py",
        "dcp/test_dcp_owner_write.py",
        "dcp/test_dcp_topk_remap.py",
        "dcp/test_dsa_cp_shard_plan.py",
    ],
    "dcp-gather": [
        "dcp/test_dcp_extend_gather_plan.py",
        "dcp/test_dcp_sparse_attention_parity.py",
    ],
    "indexer-sharding": [
        "attention/unittests/dsa/test_dsa_npu_indexer_query_shard.py",
        "attention/unittests/dsa/test_dsa_token_shard_plan.py",
        "dcp/test_dcp_quant_indexer_shard_plan.py",
        "unit/npu/mem_cache/test_npu_dsa_index_k_scale_pool.py",
        "unit/mem_cache/test_dcp_index_buf_budget.py",
    ],
    "prefill-graphs": [
        "unit/model_executor/test_cuda_graph_config_backends.py",
        "unit/model_executor/test_npu_piecewise_backend.py",
        "unit/model_executor/test_prefill_cuda_graph_runner.py",
    ],
    "host-encode": ["chat_encode/test_chat_encode_pool.py"],
    "autotune-hccl-streams": [
        "unit/npu/test_npu_autotune.py",
        "unit/npu/test_npu_hccl_config.py",
        "unit/npu/test_npu_resource_scheduling.py",
    ],
}
NPU_SUITES = {
    "kv-pool-layout": [
        "test_npu_mla_dcp_pool_bytes.py",
        "test_npu_mla_dcp_sharded_write.py",
    ],
    "sfa-int8-roundtrip": [
        "test_npu_mla_sfa_kv_int8_row.py",
        "test_npu_mla_sfa_kv_int8_roundtrip.py",
    ],
}
# One opt-in family per launch; this makes failures attributable.
SCENARIOS = {
    "baseline": {},
    "w8a8-load": {"quant": "w8a8_int"},
    "w4a4-mixed-load": {"quant": "w4a4_int"},
    "int8-dense-gate": {
        "quant": "gate:w8a8_dynamic,qkv:w8a8_dynamic,dense:w8a8_dynamic,shared_expert:w8a8_dynamic"
    },
    "tp-finalize": {
        "env": {
            "SGLANG_NPU_TP_MOE_FUSE_ROUTED_SCALE": "True",
            "SGLANG_NPU_TP_MOE_FUSE_GMM2_FINALIZE": "True",
        }
    },
    "tp-shared": {
        "env": {
            "SGLANG_NPU_TP_MOE_FUSE_ROUTED_SCALE": "True",
            "SGLANG_NPU_TP_MOE_FUSE_SHARED_EXPERT": "True",
            "SGLANG_NPU_TP_MOE_SHARED_GMM1": "True",
        }
    },
    "multistream": {
        "env": {
            "SGLANG_NPU_USE_MULTI_STREAM": "True",
            "SGLANG_NPU_TP_MOE_EAGER_MULTI_STREAM": "True",
            "SGLANG_NPU_TP_MOE_EAGER_MULTI_STREAM_MIN_TOKENS": "1",
        }
    },
    "autotune-hccl": {
        "env": {
            "SGLANG_NPU_TUNING_PROFILE": "True",
            "SGLANG_NPU_AUTO_STREAM_THRESHOLDS": "True",
            "SGLANG_NPU_AUTO_HCCL_BUFFSIZE": "True",
        }
    },
    "indexer-int8": {"env": {"SGLANG_NPU_INDEXER_INT8": "True"}},
    "indexer-query-shard": {
        "env": {"SGLANG_NPU_ENABLE_DSA_INDEXER_QUERY_SHARDING": "True"}
    },
    "sfa-int8": {"dcp": True, "env": {"SGLANG_NPU_SFA_KV_8BIT": "True"}},
    "dsa-token-shard": {"env": {"SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD": "True"}},
    "dsa-token-shard-dcp": {
        "dcp": True,
        "env": {
            "SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD": "True",
            "SGLANG_NPU_ENABLE_DSA_INDEXER_QUERY_SHARDING": "True",
        },
    },
    "graph-full": {"graph": "full", "env": {"ASCEND_USE_FIA": "1"}},
    "graph-piecewise": {"graph": "tc_piecewise"},
    "chat-pool": {"env": {"SGLANG_ENABLE_CHAT_ENCODE_POOL": "True"}},
    "chat-memo": {
        "env": {
            "SGLANG_ENABLE_CHAT_ENCODE_POOL": "True",
            "SGLANG_CHAT_ENCODE_MEMO": "True",
        }
    },
    "chat-raw": {
        "env": {
            "SGLANG_ENABLE_CHAT_ENCODE_POOL": "True",
            "SGLANG_CHAT_ENCODE_RAW": "True",
        }
    },
    "dcp-kv-gather": {"dcp": True},
}


# Every opt-in family is compared with an otherwise identical reference.
SCENARIOS["int8-dense-gate"]["quant"] = (
    "w8a8_int,gate:w8a8_dynamic,qkv:w8a8_dynamic,dense:w8a8_dynamic,shared_expert:w8a8_dynamic"
)
SCENARIOS["tp-finalize"] = {
    "env": {"SGLANG_NPU_TP_MOE_FUSE_GMM2_FINALIZE": "True"},
    "expected": ["tp-finalize"],
    "probe": "tp-finalize",
}
SCENARIOS["tp-shared"] = {
    "env": {
        "SGLANG_NPU_TP_MOE_FUSE_ROUTED_SCALE": "True",
        "SGLANG_NPU_TP_MOE_FUSE_SHARED_EXPERT": "True",
    },
    "separate_shared": True,
    "expected": ["tp-shared"],
}
SCENARIOS.update(
    {
        "tp-routed-scale": {
            "env": {"SGLANG_NPU_TP_MOE_FUSE_ROUTED_SCALE": "True"},
            "expected": ["tp-routed-scale"],
        },
        "tp-prequant": {
            "env": {"SGLANG_NPU_TP_MOE_PREQUANT_INPUT": "True"},
            "separate_shared": True,
            "expected": ["tp-prequant"],
        },
        "tp-shared-activation": {
            "env": {"SGLANG_NPU_TP_MOE_SHARED_GMM1": "True"},
            "separate_shared": True,
            "expected": ["tp-shared-activation"],
        },
        "auto-stream-thresholds": {
            "env": {"SGLANG_NPU_AUTO_STREAM_THRESHOLDS": "True"},
            "expected": ["auto-stream-thresholds"],
        },
        "auto-hccl": {
            "env": {"SGLANG_NPU_AUTO_HCCL_BUFFSIZE": "True"},
            "expected": ["auto-hccl"],
        },
    }
)
SCENARIOS["multistream"].update(separate_shared=True, expected=["multistream"])
SCENARIOS["autotune-hccl"]["expected"] = ["auto-hccl", "auto-stream-thresholds"]
for target in ("gate", "qkv", "dense", "shared_expert"):
    SCENARIOS["online-" + target] = {
        "quant": "w8a8_int," + target + ":w8a8_dynamic",
        "expected": [
            "online-gate" if target == "gate" else "online-dense",
            "online-moe-w13",
            "online-moe-w2",
        ],
    }
for name in ("w8a8-load", "w4a4-mixed-load", "int8-dense-gate"):
    SCENARIOS[name]["expected"] = ["online-dense", "online-moe-w13", "online-moe-w2"]
for mode in (1, 2, 3):
    name = "megamoe" if mode == 3 else f"fuseep-{mode}"
    SCENARIOS[name] = {
        "fuseep": mode,
        "probe": name,
        "expected": [name],
        "separate_shared": True,
        "env": {"SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "4096"},
    }
SCENARIOS["indexer-int8"].update(expected=["indexer-int8"], probe="indexer-int8")
SCENARIOS["indexer-query-shard"]["expected"] = ["indexer-query-shard"]
for name in ("graph-full", "graph-piecewise"):
    SCENARIOS[name]["expected"] = [name + "-capture", name + "-replay"]
SERVING_STAGES = ("serve", "small", "long")


def stop(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run(command, log, timeout, env):
    with log.open("w") as output:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout)
        finally:
            stop(process)


def request(url, payload=None, timeout=60):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    # Local validation should not go through deployment proxy settings.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
        req, timeout=timeout
    ) as response:
        body = response.read()
        return json.loads(body) if body else None


def client_self_check():
    """Regression checks for serial batching, cache reuse, and buffered TTFT."""
    spec = importlib.util.spec_from_file_location(
        "prefill_bench", Path(__file__).with_name("bench_glm53_prefill_1m.py")
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    state = {"seen": set(), "requests": [], "active": 0, "peak": 0}
    lock = threading.Lock()
    rendezvous = threading.Barrier(3, timeout=10)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            key = tuple(data["input_ids"])
            with lock:
                cached = len(key) if key in state["seen"] else 0
                state["seen"].add(key)
                state["requests"].append(key)
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            try:
                rendezvous.wait()  # Serial dispatch cannot reach this barrier.
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.end_headers()
                for count in (0, 1):
                    event = {
                        "meta_info": {
                            "completion_tokens": count,
                            "cached_tokens": cached,
                        }
                    }
                    self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                    self.wfile.flush()
                    time.sleep(0.05 if count == 0 else 0.15)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            finally:
                with lock:
                    state["active"] -= 1

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        cold = bench.bench_batch(base, 32, 3, 2, 1, 7, 1, 15, "cold")
        assert state["peak"] == 3 and len(set(state["requests"])) == 9
        assert all(
            r["cached_tokens"] == 0 for s in cold["samples"] for r in s["requests"]
        )
        assert all(
            0.04 <= r["ttft_s"] < r["e2e_s"] - 0.1
            for s in cold["samples"]
            for r in s["requests"]
        )
        cached = bench.bench_batch(base, 32, 3, 2, 0, 9, 1, 15, "cached")
        assert all(
            r["cached_tokens"] == 32 for s in cached["samples"] for r in s["requests"]
        )
        assert cached["prefill_tokens_per_s"] == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    with tempfile.TemporaryDirectory() as directory:
        stub = Path(directory) / "python3"
        stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
        stub.chmod(0o755)
        env = dict(
            os.environ, PATH=directory + os.pathsep + os.environ["PATH"], CAPTURE="0"
        )
        argv = subprocess.check_output(
            ["bash", str(Path(__file__).with_name("serve_glm53_1m.sh"))],
            env=env,
            text=True,
        ).splitlines()
        for phase in ("decode", "prefill"):
            assert argv[argv.index(f"--cuda-graph-backend-{phase}") + 1] == "disabled"
        assert "--cuda-graph-backend" not in argv and "--disable-cuda-graph" not in argv
    print(
        "PASS: concurrent requests, cold/cache isolation, streamed TTFT, launcher args"
    )


def boundary_self_check():
    """Exercise actual decoder forwarding with host-only dependency stubs."""

    tree = ast.parse((ROOT / "python/sglang/srt/models/glm5_next.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Glm5NextDecoderLayer"
    )
    fwd = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )
    fwd.returns = None
    fwd.decorator_list = []
    for arg in fwd.args.args + fwd.args.kwonlyargs:
        arg.annotation = None

    class Tensor:
        shape = (2, 4)

    class Operand:
        shape = (2, 4)

    class MLP:
        def __call__(self, value, *_):
            assert isinstance(value, Operand)
            return Tensor()

    class MoE:
        pass

    class Boundary:
        input_on_attn_tp_slices = False

        def prepare(self, value, batch, **kw):
            assert kw == {"quant_format": "npu_int8"}, kw
            return Operand()

        def finish(self, value, batch):
            return value

    attn = SimpleNamespace(
        input_on_attn_tp_slices=False,
        prepare=lambda x, *a, **k: x,
        finish=lambda x, *a: x,
    )
    namespace = dict(
        torch=SimpleNamespace(
            Tensor=Tensor, compiler=SimpleNamespace(is_compiling=lambda: False)
        ),
        residual_access=SimpleNamespace(buffer=lambda x: x),
        get_attn_tp_context=lambda: SimpleNamespace(clear_attn_inputs=lambda: None),
        Glm5NextMLP=MLP,
        Glm5NextMoE=MoE,
        nullcontext=nullcontext,
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[fwd], type_ignores=[])),
            "<decoder forward>",
            "exec",
        ),
        namespace,
    )
    layer = SimpleNamespace(
        attn_boundary=attn,
        ffn_boundary=Boundary(),
        ffn_quant_format="npu_int8",
        self_attn=lambda **k: Tensor(),
        mlp=MLP(),
    )
    result = namespace["forward"](layer, None, Tensor(), SimpleNamespace(), None)
    assert isinstance(result[0], Tensor)
    print(
        "PASS: actual decoder forward passes npu_int8 to boundary and quantized operand to MLP (host stubs only)"
    )


def case_checkpoint(args, scenario):
    if scenario.get("quant") and args.checkpoint_format == "modelslim":
        return args.bf16_model_path, "bf16"
    return args.model_path, args.checkpoint_format


def server_command(args, scenario):
    model_path, checkpoint_format = case_checkpoint(args, scenario)
    context_length = args.context_length
    config_path = Path(model_path) / "config.json" if model_path else None
    if (
        getattr(args, "stage", "serve") == "long"
        and config_path
        and config_path.is_file()
    ):
        config = json.loads(config_path.read_text())
        config = config.get("text_config", config)
        declared = config.get("max_position_embeddings")
        if isinstance(declared, int) and declared > 0:
            context_length = min(context_length, declared)
    command = [
        PYTHON,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_path),
        "--device",
        "npu",
        "--attention-backend",
        "ascend",
        "--tp-size",
        str(args.tp),
        "--port",
        str(args.port),
        "--context-length",
        str(context_length),
        "--chunked-prefill-size",
        str(args.chunk),
        "--mem-fraction-static",
        str(args.mem_fraction),
    ]
    if checkpoint_format == "modelslim":
        command += ["--quantization", "modelslim"]
    if scenario.get("graph"):
        command += [
            "--cuda-graph-backend-decode",
            "full",
            "--cuda-graph-backend-prefill",
            scenario["graph"],
        ]
    else:
        command += [
            "--cuda-graph-backend-decode",
            "disabled",
            "--cuda-graph-backend-prefill",
            "disabled",
        ]
    if scenario.get("quant"):
        command += ["--online-quantization", scenario["quant"]]
    if scenario.get("separate_shared"):
        command += ["--disable-shared-experts-fusion"]
    if scenario.get("fuseep"):
        command += [
            "--moe-a2a-backend",
            "ascend_fuseep",
            "--fuseep-mode",
            str(scenario["fuseep"]),
            "--ep-size",
            str(args.tp),
            "--deepep-mode",
            "low_latency",
        ]
    if scenario.get("dcp"):
        command += ["--dcp-size", str(args.dcp)]
    return command


def serve(args, case, directory, env):
    scenario = case["scenario"]
    base = f"http://127.0.0.1:{args.port}"
    # Never send test traffic to, or stop, a pre-existing server.
    with socket.socket() as connection:
        connection.settimeout(1)
        if connection.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError(f"port {args.port} is already occupied")
    with (directory / "server.log").open("w") as output:
        process = subprocess.Popen(
            case["command"],
            cwd=ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"server exited {process.returncode}; see server.log"
                    )
                try:
                    request(base + "/health", timeout=2)
                    info = request(base + "/get_model_info", timeout=2)
                    break
                except (urllib.error.URLError, TimeoutError, OSError, ValueError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("server startup timed out")
                    time.sleep(1)
            (directory / "model_info.json").write_text(json.dumps(info, indent=2))
            models = request(base + "/v1/models")
            model = models["data"][0]["id"]
            responses = []
            for _ in range(2):
                for text in (
                    "What is 2 + 3? Answer with one number.",
                    "Name the capital of France.",
                ):
                    result = request(
                        base + "/generate",
                        {
                            "text": text,
                            "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                            "return_logprob": True,
                            "logprob_start_len": 0,
                        },
                        args.request_timeout,
                    )
                    if (
                        not result.get("text")
                        or result.get("meta_info", {}).get("completion_tokens", 0) < 1
                    ):
                        raise RuntimeError(f"empty/invalid generation: {result}")
                    responses.append(result)
                chat = request(
                    base + "/v1/chat/completions",
                    {
                        "model": model,
                        "messages": [{"role": "user", "content": "Reply with hello."}],
                        "temperature": 0,
                        "max_tokens": 32,
                    },
                    args.request_timeout,
                )
                if not chat.get("choices", [{}])[0].get("message", {}).get("content"):
                    raise RuntimeError(f"empty chat response: {chat}")
                responses.append(chat)
            from concurrent.futures import ThreadPoolExecutor

            prompts = ["Explain addition briefly. " * size for size in (8, 64, 256)]
            for text in prompts:
                responses.append(
                    request(
                        base + "/generate",
                        {
                            "text": text,
                            "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                            "return_logprob": True,
                            "logprob_start_len": 0,
                        },
                        args.request_timeout,
                    )
                )
            with ThreadPoolExecutor(max_workers=2) as pool:
                concurrent = list(
                    pool.map(
                        lambda text: request(
                            base + "/generate",
                            {
                                "text": text,
                                "sampling_params": {
                                    "temperature": 0,
                                    "max_new_tokens": 8,
                                },
                            },
                            args.request_timeout,
                        ),
                        prompts[:2],
                    )
                )
            if any(not item.get("text") for item in concurrent):
                raise RuntimeError("concurrent generation failed")
            (directory / "responses.json").write_text(json.dumps(responses, indent=2))
            if args.long_context and scenario == SCENARIOS["baseline"]:
                command = [
                    PYTHON,
                    str(Path(__file__).with_name("bench_glm53_prefill_1m.py")),
                    "--port",
                    str(args.port),
                    "--context-length",
                    str(args.long_context),
                    "--batch-sizes",
                    args.batch_sizes,
                    "--cache-mode",
                    "cold",
                    "--repeats",
                    "2",
                    "--output",
                    str(directory / "prefill.json"),
                ]
                if run(command, directory / "prefill.log", args.case_timeout, env):
                    raise RuntimeError("long-context benchmark failed; see prefill.log")
            from feature_checks import evidence

            events = evidence((directory / "server.log").read_text(errors="replace"))
            seen = {item["feature"] for item in events}
            missing = set(scenario.get("expected", [])) - seen
            outcome = dict(
                status="inactive" if missing else "smoke_pass",
                activation="verified"
                if scenario.get("expected") and not missing
                else "unverified",
                missing=sorted(missing),
                evidence=events,
                quality="unverified",
                graph_replay="verified"
                if scenario.get("graph") and not missing
                else "not_requested",
            )
            if case["stage"] == "small":
                from feature_benchmarks import small_commands

                # Timed runs use fresh processes with evidence disabled, configured by the caller.
                points = []
                for command, output in small_commands(args, directory):
                    code = run(
                        command, output.with_suffix(".log"), args.case_timeout, env
                    )
                    if code:
                        raise RuntimeError(f"small benchmark exited {code}")
                    rows = [
                        json.loads(line)
                        for line in output.read_text().splitlines()
                        if line.strip()
                    ]
                    if not rows or any(
                        row.get("completed", 0) < args.small_requests for row in rows
                    ):
                        raise RuntimeError(
                            "small benchmark did not complete every request"
                        )
                    points.append(dict(command=command, results=rows))
                outcome["benchmarks"] = points
            elif case["stage"] == "long":
                from feature_benchmarks import long_matrix

                points = long_matrix(
                    args, base, request, case["checkpoint_path"], directory
                )
                outcome["benchmarks"] = points
                if any(point["status"] == "capacity_blocked" for point in points):
                    outcome["status"] = "capacity_blocked"
            return outcome
        finally:
            stop(process)


_REFERENCE_RESPONSES = {}


def validate_serving(args, case, directory, case_env, base_env):
    from feature_checks import compare_responses

    scenario = case["scenario"]
    if scenario.get("graph") and scenario.get("dcp"):
        return dict(status="blocked", reason="prefill graphs do not support DCP")
    if scenario.get("probe"):
        probe = [
            PYTHON,
            str(Path(__file__).resolve()),
            "--probe-feature",
            scenario["probe"],
        ]
        code = run(probe, directory / "capability.log", 60, case_env)
        if code:
            return dict(
                status="dependency_blocked",
                reason="required operator/API unavailable; see capability.log",
            )
    reference = {
        key: value
        for key, value in scenario.items()
        if key in ("quant", "dcp", "separate_shared")
    }
    reference_key = json.dumps(reference, sort_keys=True)
    if reference_key not in _REFERENCE_RESPONSES:
        reference_case = dict(
            case,
            stage="serve",
            scenario=reference,
            command=server_command(args, reference),
        )
        reference_directory = directory / "reference"
        reference_directory.mkdir()
        serve(args, reference_case, reference_directory, base_env)
        _REFERENCE_RESPONSES[reference_key] = json.loads(
            (reference_directory / "responses.json").read_text()
        )
        # Embed the reference once, in the same Markdown report.
        for artifact in reference_directory.iterdir():
            artifact.rename(directory / ("reference-" + artifact.name))
        reference_directory.rmdir()
    diagnostic_case = dict(case, stage="serve")
    result = serve(args, diagnostic_case, directory, case_env)
    responses = json.loads((directory / "responses.json").read_text())
    result["parity"] = compare_responses(_REFERENCE_RESPONSES[reference_key], responses)
    result["quality"] = "verified"
    if result["status"] != "inactive" and (
        scenario.get("expected") or not scenario or case["id"].endswith("-baseline")
    ):
        result["status"] = "pass"
    if case["stage"] in ("small", "long"):
        if result["status"] != "pass":
            return dict(result, benchmark="blocked: activation and parity required")
        measured = directory / "measured"
        measured.mkdir()
        measured_env = dict(case_env, SGLANG_NPU_FEATURE_EVIDENCE="False")
        from feature_benchmarks import CapacityBlocked

        try:
            timed = serve(args, case, measured, measured_env)
        except CapacityBlocked as exc:
            timed = dict(status="capacity_blocked", reason=str(exc))
            result["reason"] = str(exc)
        result["benchmarks"] = timed.get("benchmarks", [])
        if timed["status"] == "capacity_blocked":
            result["status"] = "capacity_blocked"
        for artifact in measured.iterdir():
            artifact.rename(directory / ("measured-" + artifact.name))
        measured.rmdir()
    return result


def markdown_block(text, language="text"):
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    fence = "`" * max(3, 1 + max((len(m) for m in re.findall(r"`+", text)), default=0))
    return f"{fence}{language}\n{text.rstrip()}\n{fence}\n"


def append_result(path, result, directory):
    with path.open("a") as output:
        output.write(f"\n## {result['id']}: {result['status'].upper()}\n\n")
        output.write(markdown_block(json.dumps(result, indent=2), "json"))
        for artifact in sorted(directory.rglob("*")):
            if artifact.is_file():
                output.write(
                    f"\n<details><summary>{artifact.relative_to(directory)}</summary>\n\n"
                )
                from feature_checks import bounded_log

                text = artifact.read_text(errors="replace")
                output.write(
                    markdown_block(
                        bounded_log(text) if artifact.suffix == ".log" else text
                    )
                )
                output.write("\n</details>\n")


def fusion_import_self_check():
    """Catch stale imports before they prevent pytest collection/server startup."""
    for module in (
        "sglang.srt.hardware_backend.npu.moe.tp_fusion",
        "sglang.srt.layers.dcp.layout",
        "sglang.srt.layers.attention.dsa.dsa_token_shard",
        "sglang.srt.layers.attention.dsa.dsa_token_shard_layout",
    ):
        source = ROOT / "python" / (module.replace(".", "/") + ".py")
        exported = set()
        for node in ast.parse(source.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                exported.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                exported.update(
                    alias.asname or alias.name.split(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.Assign):
                exported.update(t.id for t in node.targets if isinstance(t, ast.Name))
        for file in (ROOT / "python/sglang/srt").rglob("*.py"):
            text = file.read_text()
            if module not in text:
                continue
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.ImportFrom) and node.module == module:
                    missing = {alias.name for alias in node.names} - exported
                    assert not missing, (
                        f"{file.relative_to(ROOT)}:{node.lineno}: missing {sorted(missing)}"
                    )
    print("PASS: TP fusion and DSA/DCP imports resolve against current exports")


def plan(args):
    cases = [
        {
            "id": "host-syntax",
            "stage": "host",
            "command": [
                PYTHON,
                "-m",
                "compileall",
                "-q",
                "python/sglang/srt",
                "python/sglang/kernels",
                "benchmark/prefill_1m",
            ],
        },
        {
            "id": "host-shell",
            "stage": "host",
            "command": ["bash", "-n", "benchmark/prefill_1m/serve_glm53_1m.sh"],
        },
        {
            "id": "host-client",
            "stage": "host",
            "command": [PYTHON, str(Path(__file__).resolve()), "--self-check-client"],
        },
        {
            "id": "host-1m-dry-run",
            "stage": "host",
            "command": [
                PYTHON,
                "benchmark/prefill_1m/bench_glm53_prefill_1m.py",
                "--dry-run",
            ],
        },
    ]
    cases.append(
        {
            "id": "host-workloads",
            "stage": "host",
            "command": [
                PYTHON,
                str(Path(__file__).resolve()),
                "--self-check-workloads",
            ],
        }
    )
    cases.append(
        {
            "id": "host-suite-commands",
            "stage": "host",
            "command": [
                PYTHON,
                str(Path(__file__).resolve()),
                "--self-check-suite-commands",
            ],
        }
    )
    cases.append(
        {
            "id": "host-fusion-imports",
            "stage": "host",
            "command": [PYTHON, str(Path(__file__).resolve()), "--self-check-imports"],
        }
    )
    cases.append(
        {
            "id": "host-boundary",
            "stage": "host",
            "command": [PYTHON, str(Path(__file__).resolve()), "--self-check-boundary"],
        }
    )
    for stage, suites in (("unit", UNIT_SUITES), ("npu", NPU_SUITES)):
        for name, paths in suites.items():
            prefix = (
                "test/registered/"
                if stage == "unit"
                else "test/registered/npu/basic_function/memory_and_scheduling/"
            )
            cases.append(
                {
                    "id": f"{stage}-{name}",
                    "stage": stage,
                    "command": [
                        PYTHON,
                        "-m",
                        "pytest",
                        "-q",
                        "-ra",
                        "-p",
                        "no:cacheprovider",
                        *[prefix + p for p in paths],
                    ],
                    "junit": True,
                }
            )
    cases.append(
        {
            "id": "npu-int8-indexer",
            "stage": "npu",
            "command": [
                PYTHON,
                "-m",
                "pytest",
                "-q",
                "-ra",
                "-p",
                "no:cacheprovider",
                "test/registered/unit/npu/attention/test_npu_quant_lightning_indexer.py",
            ],
            "junit": True,
        }
    )
    cases.append(
        {
            "id": "npu-moe-parity",
            "stage": "npu",
            "junit": True,
            "command": [
                PYTHON,
                "-m",
                "pytest",
                "-q",
                "-ra",
                "-p",
                "no:cacheprovider",
                "test/registered/unit/npu/quantization/test_glm_a3_device_features.py",
            ],
        }
    )
    for name, scenario in SCENARIOS.items():
        scenario = dict(scenario)
        stage = args.stage if args.stage in ("small", "long") else "serve"
        if stage == "long":
            if name not in getattr(args, "benchmark_variants", "baseline").split(","):
                continue
            scenario["dcp"] = args.dcp > 1
        model_path, checkpoint_format = case_checkpoint(args, scenario)
        cases.append(
            {
                "id": stage + "-" + name,
                "stage": stage,
                "scenario": scenario,
                "command": server_command(args, scenario),
                "requires_bf16": bool(scenario.get("quant")),
                "checkpoint_path": str(model_path) if model_path else None,
                "checkpoint_format": checkpoint_format,
            }
        )
    return [
        case
        for case in cases
        if (args.stage == "all" or case["stage"] == args.stage)
        and any(
            fnmatch.fnmatchcase(case["id"], pattern)
            for pattern in args.cases.split(",")
        )
    ]


def suite_command_self_check():
    """Verify generated pytest commands actually collect their selected file."""
    args = SimpleNamespace(
        stage="all",
        cases="*",
        model_path=None,
        checkpoint_format="bf16",
        bf16_model_path=None,
        tp=8,
        dcp=2,
        port=31000,
        context_length=8192,
        chunk=4096,
        mem_fraction=0.85,
    )
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        sentinel = directory / "collected"
        probe = directory / "test_selection_probe.py"
        probe.write_text(f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n")
        for case in plan(args):
            if not case.get("junit"):
                continue
            selections = [
                value
                for value in case["command"]
                if value.startswith("test/registered/")
            ]
            assert selections and all(
                (ROOT / value).is_file() for value in selections
            ), case["id"]
            command = [value for value in case["command"] if value not in selections]
            command += ["--collect-only", "-c", os.devnull, str(probe)]
            sentinel.unlink(missing_ok=True)
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"),
            )
            # Exit 5 is expected: the probe module is collected but defines no tests.
            if completed.returncode != 5 or not sentinel.exists():
                raise AssertionError(
                    f"{case['id']}: invalid pytest command\n{completed.stdout}\n{completed.stderr}"
                )
    print("PASS: all unit/NPU suite commands collect their intended selection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("host", "unit", "npu", "serve", "small", "long", "all"),
        default="host",
    )
    parser.add_argument(
        "--cases", default="*", help="Comma-separated case globs; see --list."
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--self-check-client", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--self-check-boundary", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--self-check-imports", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--probe-npu", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--self-check-suite-commands", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=os.environ.get("MODEL_PATH", "/home/weights/GLM5.2-w4a8-int/"),
        help="Local checkpoint directory (or set MODEL_PATH).",
    )
    parser.add_argument(
        "--checkpoint-format",
        choices=("bf16", "modelslim"),
        default="modelslim",
        help="Use modelslim for an offline W4A8/W8A8 INT checkpoint; online conversion cases require bf16.",
    )
    parser.add_argument(
        "--bf16-model-path",
        type=Path,
        default=Path("/home/weights/GLM-5.2"),
        help="Optional BF16 source for online-conversion cases when the primary checkpoint is ModelSlim.",
    )
    parser.add_argument("--tp", type=int, default=16)
    parser.add_argument("--dcp", type=int, default=8)
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--mem-fraction", type=float, default=0.85)
    parser.add_argument(
        "--long-context",
        type=int,
        default=0,
        help="Also benchmark this many input tokens on the baseline server.",
    )
    parser.add_argument("--batch-sizes", default="1,2")
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--case-timeout", type=float, default=7200)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".codex-artifacts/glm53-feature-tests")
        / time.strftime("%Y%m%d-%H%M%S.md", time.gmtime()),
        help="Single Markdown report path (default: timestamp.md).",
    )
    parser.add_argument(
        "--benchmark-variants",
        default="baseline",
        help="Long-context variants, comma separated",
    )
    parser.add_argument(
        "--small-lengths",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[512, 4096],
    )
    parser.add_argument(
        "--small-concurrency",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[1, 8],
    )
    parser.add_argument("--small-requests", type=int, default=32)
    parser.add_argument(
        "--long-lengths",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[32768, 131072, 262144, 524288, 1048576],
    )
    parser.add_argument(
        "--long-concurrency",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[1, 16],
    )
    parser.add_argument("--long-requests", type=int, default=16)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument(
        "--self-check-workloads", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--probe-feature", choices=tuple(SCENARIOS), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.self_check_workloads:
        from feature_checks import self_check_workloads

        self_check_workloads()
        return 0
    if args.probe_feature:
        from feature_checks import probe_feature

        probe_feature(args.probe_feature)
        return 0
    if args.stage == "long" and args.context_length == 8192:
        args.context_length = max(args.long_lengths) + 32

    if args.self_check_client:
        client_self_check()
        return 0
    if args.self_check_boundary:
        boundary_self_check()
        return 0
    if args.self_check_imports:
        fusion_import_self_check()
        return 0
    if args.self_check_suite_commands:
        suite_command_self_check()
        return 0
    if args.probe_npu:
        import torch
        import torch_npu  # noqa: F401

        required = args.tp
        count = torch.npu.device_count()
        if not torch.npu.is_available() or count < required:
            raise RuntimeError(f"need {required} visible NPUs, found {count}")
        info = {
            "count": count,
            "devices": [torch.npu.get_device_name(i) for i in range(count)],
        }
        for filename in (
            "/usr/local/Ascend/ascend-toolkit/latest/version.cfg",
            "/usr/local/Ascend/cann/latest/version.cfg",
        ):
            if Path(filename).is_file():
                info[filename] = Path(filename).read_text()
        print(json.dumps(info, indent=2))
        try:
            subprocess.run(["npu-smi", "info"], timeout=30, check=True)
        except FileNotFoundError:
            print("npu-smi unavailable")
        return 0
    if (
        any(
            value < 1
            for values in (
                args.small_lengths,
                args.small_concurrency,
                args.long_lengths,
                args.long_concurrency,
            )
            for value in values
        )
        or min(args.benchmark_repeats, args.small_requests, args.long_requests) < 1
    ):
        parser.error(
            "benchmark lengths, concurrency, request counts and repeats must be positive"
        )
    if args.tp < 1 or args.dcp < 1 or args.tp % args.dcp:
        parser.error("TP must be positive and divisible by DCP")
    if args.long_context and args.context_length <= args.long_context:
        parser.error("context-length must exceed long-context to leave output space")
    if args.stage == "small" and max(args.small_lengths) + 32 > args.context_length:
        parser.error(
            "context-length must accommodate the largest small input plus 32 output tokens"
        )
    cases = plan(args)
    if not cases:
        parser.error("no cases selected")
    if any(case["stage"] in ("npu", *SERVING_STAGES) for case in cases):
        required = (
            args.tp if any(case["stage"] in SERVING_STAGES for case in cases) else 1
        )
        cases.insert(
            0,
            {
                "id": "device-environment",
                "stage": "preflight",
                "command": [
                    PYTHON,
                    str(Path(__file__).resolve()),
                    "--probe-npu",
                    "--tp",
                    str(required),
                ],
            },
        )
    if args.list:
        for case in cases:
            print(f"{case['id']:32} {case['stage']}")
        return 0
    args.output = args.output.resolve()
    if args.output.suffix.lower() != ".md":
        parser.error("--output must name a .md file")
    if args.output.exists():
        parser.error("--output already exists; choose a fresh .md file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="glm53-feature-tests-") as workspace:
        env = dict(
            os.environ,
            PYTHONPATH=str(ROOT / "python")
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
        )
        # Isolate opt-in families from inherited shell settings.
        for name in list(env):
            if name.startswith(
                (
                    "SGLANG_NPU_TP_MOE_",
                    "SGLANG_NPU_DSA_",
                    "SGLANG_NPU_ENABLE_DSA_",
                    "SGLANG_NPU_ENABLE_DCP_",
                )
            ):
                env.pop(name)
        for scenario in SCENARIOS.values():
            for name in scenario.get("env", {}):
                env.pop(name, None)
        # These are boolean opt-ins; remove typed thresholds/modes instead of writing False into them.
        for name in ("ASCEND_USE_FIA",):
            env[name] = "0"
        env["SGLANG_NPU_FEATURE_EVIDENCE"] = "True"
        env.pop("HCCL_BUFFSIZE", None)
        env["SGLANG_NPU_TUNING_CACHE_DIR"] = str(Path(workspace) / "tuning-cache")
        env["SGLANG_NPU_TP_MOE_REUSE_MXFP8"] = "False"
        env["SGLANG_NPU_TP_MOE_NORM_MXFP8"] = "False"
        versions = {}
        for package in (
            "torch",
            "torch-npu",
            "sgl-kernel-npu",
            "sglang",
            "pytest",
            "pytest-asyncio",
            "requests",
        ):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        report = {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "versions": versions,
            "working_tree_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=ROOT, text=True
                )
            ),
            "source_paths": {"root": str(ROOT), "python": str(ROOT / "python")},
            "tracked_diff_sha256": __import__("hashlib")
            .sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT))
            .hexdigest(),
            "environment": {
                key: env.get(key)
                for key in (
                    "ASCEND_RT_VISIBLE_DEVICES",
                    "ASCEND_HOME_PATH",
                    "CANN_HOME",
                    "HCCL_BUFFSIZE",
                    "HCCL_SOCKET_IFNAME",
                    "HCCL_IF_IP",
                )
            },
            "arguments": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "scope": "GLM DSA A3 INT integration; Gemma-only and MXFP additions excluded",
            "results": [],
            "limitations": [
                "Unit/mock passes are host evidence only.",
                "Serving smoke does not establish optimization activation, numerical quality, graph replay or performance.",
                "Legacy experimental D1v3 is disabled; reviewed PR token sharding is tested instead.",
            ],
        }
        args.output.write_text(
            "# GLM-5.3 A3 feature validation\n\n"
            + markdown_block(
                json.dumps(
                    {k: v for k, v in report.items() if k != "results"}, indent=2
                ),
                "json",
            )
        )
        hardware_ready = True
        for case in cases:
            directory = Path(workspace) / case["id"]
            directory.mkdir()
            result = dict(case, env=case.get("scenario", {}).get("env", {}))
            started = time.monotonic()
            case_env = dict(env, **result["env"])
            if case.get("junit"):
                result["command"] = case["command"] + [
                    "--junitxml=" + str(directory / "junit.xml")
                ]
            try:
                if case.get("requires_bf16") and not case.get("checkpoint_path"):
                    result.update(
                        status="blocked",
                        reason="online conversion requires BF16 source weights; provide --bf16-model-path alongside the ModelSlim checkpoint",
                    )
                elif args.dry_run:
                    result["status"] = "planned"
                elif case["stage"] in ("npu", *SERVING_STAGES) and not hardware_ready:
                    result.update(status="blocked", reason="device preflight failed")
                elif case["stage"] in SERVING_STAGES and (
                    not case.get("checkpoint_path")
                    or not Path(case["checkpoint_path"]).is_dir()
                ):
                    result.update(
                        status="blocked",
                        reason=f"checkpoint directory does not exist: {case.get('checkpoint_path')}; set --model-path (or --bf16-model-path for online conversion) to the real local checkpoint",
                    )
                elif case.get("scenario", {}).get("dcp") and args.dcp == 1:
                    result.update(
                        status="blocked", reason="DCP scenario needs --dcp > 1"
                    )
                elif case["stage"] in SERVING_STAGES:
                    result.update(
                        validate_serving(args, case, directory, case_env, env)
                    )
                else:
                    code = run(
                        result["command"],
                        directory / "test.log",
                        args.case_timeout,
                        case_env,
                    )
                    result.update(
                        status="pass" if code == 0 else "fail", returncode=code
                    )
                    if case.get("junit") and (directory / "junit.xml").exists():
                        suites = (
                            ET.parse(directory / "junit.xml")
                            .getroot()
                            .iter("testsuite")
                        )
                        counts = {
                            key: 0 for key in ("tests", "failures", "errors", "skipped")
                        }
                        for suite in suites:
                            for key in counts:
                                counts[key] += int(suite.get(key, 0))
                        result["counts"] = counts
                        if code == 0 and (counts["skipped"] or not counts["tests"]):
                            result["status"] = "partial"
            except Exception as error:
                result.update(status="fail", error=str(error))
                (directory / "exception.log").write_text(traceback.format_exc())
            result["elapsed_s"] = round(time.monotonic() - started, 3)
            if case["stage"] == "preflight" and result["status"] == "fail":
                hardware_ready = False
            report["results"].append(result)
            append_result(args.output, result, directory)
            print(
                f"{result['status'].upper():12} {case['id']} ({result['elapsed_s']}s)",
                flush=True,
            )
        with args.output.open("a") as output:
            output.write("\n## Summary\n\n| Case | Result | Seconds |\n|---|---|---|\n")
            for result in report["results"]:
                output.write(
                    f"| {result['id']} | {result['status']} | {result['elapsed_s']} |\n"
                )
        print(f"Results: {args.output}")
        statuses = {item["status"] for item in report["results"]}
        return (
            1
            if "fail" in statuses
            else 2
            if statuses
            & {
                "blocked",
                "partial",
                "inactive",
                "dependency_blocked",
                "capacity_blocked",
            }
            else 0
        )


if __name__ == "__main__":
    raise SystemExit(main())
