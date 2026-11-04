#!/usr/bin/env python3
"""Host AISBench orchestration with container-owned SGLang feature checks."""

import argparse
import ast
import contextlib
import datetime
import fcntl
import importlib.util
import json
import os
import re
import selectors
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

AISBENCH_SHELL = """\
set -e
if ! declare -F conda >/dev/null; then
    conda_command=${CONDA_EXE:-conda}
    if ! command -v "$conda_command" >/dev/null; then
        conda_command=conda
    fi
    if command -v "$conda_command" >/dev/null; then
        conda_hook=$("$conda_command" shell.bash hook)
        eval "$conda_hook"
    fi
fi
source "$1"
cd "$2"
unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1,::1
exec python3 aisbench_test.py "${@:3}"
"""

RELATIVE = "benchmark/prefill_1m/run_bde_overnight.py"
AIS_ENV_CANDIDATES = (
    "/home/tbaydasov/set_env.sh",
    "/home/tbaydasovset_env.sh",
)
DCP_VARIANTS = (
    "dcp-kv-gather",
    "dcp-packed-read",
    "dcp-gather-prefetch",
    "dcp-sfa-overlap",
    "dcp-full-width-q",
    "dcp-legacy-token-shard",
    "dcp-stream-o-proj",
    "dcp-decode-q-preissue",
    "bde-v2",
    "historical-bde",
    "legacy-control-40s",
    "legacy-megamoe-34s",
    "graph-full",
    "graph-piecewise",
    "graph-piecewise-probe",
    "graph-piecewise-dcp16",
    "metadata-budget-control",
    "megamoe-no-lookahead",
    "dsa-metadata",
    "dsa-metadata-batch2",
    "dsa-metadata-megamoe",
    "dsa-metadata-megamoe-batch2",
    "dsa-metadata-packed-batch2",
    "dsa-metadata-packed-megamoe-batch2",
)
MOE_VARIANTS = (
    "baseline",
    "tp-routed-scale",
    "tp-finalize",
    "tp-prequant",
    "tp-shared-stream",
    "tp-shared-stream-bounded",
    "best-int8-fused",
    "best-int8-metadata",
    "best-int8-packed-batch2",
    "best-int8-batch2",
    "tp-shared-stream-combined",
    "gate-int8-norm-probe",
    "gate-int8-norm",
    "gate-int8-norm-stream",
    "megamoe-gate-int8",
    "tp-shared",
    "tp-shared-activation",
    "multistream",
    "auto-stream-thresholds",
    "auto-hccl",
    "autotune-hccl",
    "fuseep-1",
    "fuseep-2",
    "megamoe",
    "indexer-fp16",
    "moe-stream-auto",
    "moe-fused-combined",
)
VARIANTS = DCP_VARIANTS + MOE_VARIANTS

BASE_FLAGS = (
    "SGLANG_NPU_ENABLE_DSA_CP",
    "SGLANG_NPU_DSA_CP_TOKEN_SHARDED",
    "SGLANG_NPU_DSA_CP_Q_FULL_WIDTH",
    "SGLANG_NPU_DSA_CP_LOCAL_TOPK",
    "SGLANG_NPU_STREAM_O_PROJ",
    "SGLANG_NPU_DCP_DECODE_Q_PREISSUE",
    "SGLANG_NPU_ENABLE_DCP_PACKED_READ",
    "SGLANG_NPU_ENABLE_DCP_GATHER_PREFETCH",
    "SGLANG_NPU_ENABLE_DCP_SFA_GATHER_OVERLAP",
    "SGLANG_NPU_SFA_KV_8BIT",
    "SGLANG_NPU_INDEXER_INT8",
    "SGLANG_ENABLE_CHAT_ENCODE_POOL",
    "SGLANG_CHAT_ENCODE_MEMO",
    "SGLANG_CHAT_ENCODE_RAW",
)


def resolve_ais_env(explicit):
    if explicit is not None:
        return explicit
    return next(
        (path for path in AIS_ENV_CANDIDATES if Path(path).is_file()),
        AIS_ENV_CANDIDATES[0],
    )


def check_ais_paths(directory, env_file):
    missing = []
    for option, path in (
        ("--ais-dir", Path(directory) / "aisbench_test.py"),
        ("--ais-env", Path(env_file)),
    ):
        if not path.is_file():
            missing.append(f"{option}: missing host file {path}")
    if missing:
        raise ValueError("; ".join(missing))


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def http(path, *, post=False, timeout=10, expect_json=True):
    request = urllib.request.Request(
        "http://127.0.0.1:30000" + path,
        data=b"{}" if post else None,
    )
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
        request, timeout=timeout
    ) as response:
        body = response.read()
        if not expect_json:
            return body.decode("utf-8", errors="replace")
        return json.loads(body) if body else None


def require_free_port():
    with socket.socket() as connection:
        connection.settimeout(1)
        if connection.connect_ex(("127.0.0.1", 30000)) == 0:
            raise RuntimeError(
                "Port 30000 is occupied; stop your existing server first"
            )


def stop_owned(process, *, include_exited=False):
    if process.poll() is not None and not include_exited:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)
    except ProcessLookupError:
        pass
    # A server parent can exit before its workers; this group belongs to this run.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def report_cases(text):
    results = {}
    for block in re.findall(r"```json\s*\n(.*?)```", text, re.DOTALL):
        try:
            item = json.loads(block)
        except ValueError:
            continue
        if isinstance(item, dict) and "id" in item and "status" in item:
            results[item["id"]] = item
    return results


def benchmark_admitted(gate, *, baseline=False, allow_parity_failure=False):
    if gate.get("missing") or (not baseline and gate.get("activation") != "verified"):
        return False
    if gate.get("status") == "pass" and gate.get("quality") == "verified":
        return True
    return bool(
        allow_parity_failure
        and gate.get("status") == "fail"
        and gate.get("failure_phase") == "parity"
        and gate.get("activation") == "verified"
    )


def ais_metrics(text, requests):
    metrics = None
    for line in text.splitlines():
        if "Failed Requests" in line:
            columns = [part.strip() for part in line.split("│") if part.strip()]
            if len(columns) >= 3 and columns[-1].isdigit() and int(columns[-1]):
                raise RuntimeError("AISBench reported failed requests")
        if "'TTFT avg'" in line and "{" in line:
            try:
                candidate = ast.literal_eval(line[line.index("{") :])
            except (ValueError, SyntaxError):
                continue
            if candidate.get("total_req") == requests:
                metrics = candidate
    if metrics is None or metrics.get("TTFT avg", 0) <= 0:
        raise RuntimeError(
            "Missing successful AISBench summary for the measured requests"
        )
    if metrics.get("output_len") == 1:
        metrics["TPOT avg"] = None
        metrics["TPOT SLO_P90"] = None
    return metrics


def wait_benchmark(process, worker_process, limit):
    deadline = time.monotonic() + limit
    while True:
        if worker_process.poll() is not None:
            raise RuntimeError("Server worker exited during AISBench; see server.log")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"AISBench exceeded {limit:.0f}s total budget; see client and server logs"
            )
        try:
            return process.wait(timeout=min(2, remaining))
        except subprocess.TimeoutExpired:
            pass


def required_kv_capacity(config, resolved):
    page = resolved["page_size"] * resolved["dcp_size"]

    def rounded(tokens):
        return (tokens + page - 1) // page * page

    cached = max(0, int(config["length"] * config["repeat_rate"]) - 1) // page * page
    active = min(resolved["max_running_requests"], config["requests"])
    full = rounded(config["length"] + config["output_len"])
    suffix = rounded(config["length"] + config["output_len"] - cached)
    return max(full, cached + active * suffix) + 4096


def server_failure(log):
    for pattern, reason in (
        (r"aicore timeout", "native AICore timeout"),
        (r"out of memory|NPU out of memory", "server out of memory"),
        (r"Scheduler watchdog timeout", "scheduler watchdog timeout"),
        (r"TypeError: _weak_ref_tensor", "piecewise graph output type error"),
    ):
        match = re.search(pattern, log, re.IGNORECASE)
        if match:
            return reason
    return None


def collect_native_dumps(repo, folder, started):
    source = repo / "extra-info" / "data-dump"
    if not source.is_dir():
        return
    for path in source.rglob("*"):
        if path.is_file() and path.stat().st_mtime >= started:
            target = folder / "native-dumps" / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def variant_launch(harness, repo, config):
    scenario = harness.scenario_for(
        config["variant"], config.get("reference_recipe", "ordinary")
    )
    config = dict(config)
    for source, target in (
        ("chunk", "chunk"),
        ("mem_fraction", "memfrac"),
        ("kv_rows", "kv_rows"),
        ("max_running_requests", "maxreq"),
        ("context_length", "context"),
    ):
        if source in scenario.get("launch_overrides", {}):
            config[target] = scenario["launch_overrides"][source]
    if scenario.get("fuseep") == 3:
        scenario["env"]["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = str(
            config.get("megamoe_dispatch_tokens", 4096)
        )
    checkpoint = config["checkpoint"]
    if scenario.get("fuseep") in (1, 2):
        checkpoint = config.get("w8a8_checkpoint")
        if not checkpoint:
            raise ValueError("FuseEP modes 1/2 require --w8a8-checkpoint")
    launcher = load_module(
        repo / "benchmark/prefill_1m/launch_bde_v2.py", "night_launch"
    )
    inherited = dict(
        os.environ,
        CAPTURE="0",
        CHUNK=str(config["chunk"]),
        MEMFRAC=str(config["memfrac"]),
        MAXREQ=str(config["maxreq"]),
        PORT="30000",
        EXTRA_ARGS=f"--quantization modelslim --watchdog-timeout {config.get('watchdog_timeout', 7200)} --max-total-tokens {config['kv_rows']}",
    )
    if scenario.get("dcp"):
        inherited["PREFILL_MAXREQ"] = str(
            scenario.get("launch_overrides", {}).get("prefill_max_requests", 1)
        )
    else:
        inherited.pop("PREFILL_MAXREQ", None)
    args = argparse.Namespace(
        tp=16,
        dcp=16 if scenario.get("dcp") else 1,
        model_path=checkpoint,
        real_weights=True,
        context_length=config["context"],
    )
    command, env, _selected = launcher.launch_config(args, inherited)
    # Explicitly neutralize default-on BDE flags before selecting one family.
    from sglang.srt.environ import EnvBool, envs

    for entry in harness.SCENARIOS.values():
        for name in entry.get("env", {}):
            if isinstance(getattr(type(envs), name, None), EnvBool):
                env[name] = "False"
            else:
                env.pop(name, None)
    for name in BASE_FLAGS:
        env[name] = "False"
    env.update(scenario.get("env", {}))
    env.update(
        SGLANG_NPU_FEATURE_EVIDENCE="True"
        if config.get("performance_only")
        else "False",
        SGLANG_DEBUG_STAGE_LATENCY="False",
        SGLANG_NPU_DEBUG_DCP="False",
    )
    if "max_prefill_tokens" in scenario.get("launch_overrides", {}):
        command += [
            "--max-prefill-tokens",
            str(scenario["launch_overrides"]["max_prefill_tokens"]),
        ]
    if scenario.get("graph"):
        command[command.index("--cuda-graph-backend-prefill") + 1] = scenario["graph"]
        command[command.index("--cuda-graph-backend-decode") + 1] = scenario.get(
            "decode_graph", "full"
        )
        command += ["--cuda-graph-max-bs-decode", "32", "--cuda-graph-bs-prefill"]
        command += [
            str(n)
            for n in (16, 128, 1024, 4096, 16384, 32768)
            if n <= min(config["chunk"], config["context"])
        ]
    if scenario.get("quant"):
        command += ["--online-quantization", scenario["quant"]]
    if scenario.get("separate_shared"):
        command.append("--disable-shared-experts-fusion")
    if scenario.get("fuseep"):
        command[command.index("--moe-a2a-backend") + 1] = "ascend_fuseep"
        command += [
            "--fuseep-mode",
            str(scenario["fuseep"]),
            "--ep-size",
            "16",
            "--deepep-mode",
            "low_latency",
        ]
    if "auto-hccl" in scenario.get("expected", []):
        env.pop("HCCL_BUFFSIZE", None)
        env.pop("DEEPEP_HCCL_BUFFSIZE", None)
    return command, env


def validate_variant_topology(harness, config, resolved):
    scenario = harness.scenario_for(
        config["variant"], config.get("reference_recipe", "ordinary")
    )
    overrides = scenario.get("launch_overrides", {})
    for key, expected in (
        ("tp_size", 16),
        ("dcp_size", 16 if scenario.get("dcp") else 1),
        (
            "max_running_requests",
            overrides.get("max_running_requests", config["maxreq"]),
        ),
        ("quantization", "modelslim"),
        ("cuda_graph_backend_prefill", scenario.get("graph", "disabled")),
        (
            "cuda_graph_backend_decode",
            scenario.get("decode_graph", "full")
            if scenario.get("graph")
            else "disabled",
        ),
    ):
        if resolved.get(key) != expected:
            raise RuntimeError(
                f"Unexpected {key}: {resolved.get(key)}; expected {expected}"
            )
    prefill_budget = overrides.get("max_prefill_tokens")
    if (
        prefill_budget is not None
        and resolved.get("max_prefill_tokens") != prefill_budget
    ):
        raise RuntimeError(
            f"Expected max_prefill_tokens={prefill_budget}; got {resolved.get('max_prefill_tokens')}"
        )
    prefill_cap = overrides.get("prefill_max_requests", 1)
    if scenario.get("dcp") and resolved.get("prefill_max_requests") != prefill_cap:
        raise RuntimeError(f"DCP timing requires prefill_max_requests={prefill_cap}")
    fuseep = scenario.get("fuseep")
    if fuseep and (
        resolved.get("ep_size", resolved.get("moe_ep_size")) != 16
        or resolved.get("fuseep_mode") != fuseep
        or resolved.get("moe_a2a_backend") != "ascend_fuseep"
    ):
        raise RuntimeError("Resolved FuseEP topology/mode differs from launch")


def timed_activation(log, variant, scenario=None):
    features = set()
    for line in log.splitlines():
        if "NPU_FEATURE_EVIDENCE " not in line:
            continue
        try:
            features.add(
                json.loads(line.split("NPU_FEATURE_EVIDENCE ", 1)[1])["feature"]
            )
        except (ValueError, KeyError, TypeError):
            continue
    required = set(scenario.get("expected", [])) if scenario is not None else set()
    if not variant.startswith("legacy-") and variant not in (
        "graph-piecewise-probe",
        "gate-int8-norm-probe",
    ):
        required.add("dsa-residual-shard")
    if "megamoe" in variant:
        required.add("megamoe")
        if not variant.startswith("legacy-"):
            required.add("dsa-ep-moe-local")
    if variant.startswith("dsa-metadata"):
        required.add("dsa-metadata-reuse")
    if variant.endswith("-batch2"):
        required.add("dcp-prefill-batch-2")
        if "packed" in variant:
            required.add("dcp-packed-read-batch2")
    missing = sorted(required - features)
    if missing:
        raise RuntimeError("Timed workload missing activation: " + ", ".join(missing))
    return sorted(features)


def worker():
    """Runs inside Docker. EOF or stop on stdin cleans up only owned children."""
    config = json.loads(sys.stdin.readline())
    repo, folder = Path(config["repo"]), Path(config["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    os.chdir(repo)
    owned = []
    started = time.time()

    def spawn(*args, **kwargs):
        process = subprocess.Popen(*args, **kwargs)
        owned.append(process)
        return process

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    def watch_parent():
        sys.stdin.readline()
        os.kill(os.getpid(), signal.SIGTERM)

    def emit(value):
        print(json.dumps(value), flush=True)

    signal.signal(signal.SIGTERM, interrupted)
    threading.Thread(target=watch_parent, daemon=True).start()
    try:
        require_free_port()
        harness = load_module(
            repo / "benchmark/prefill_1m/test_glm53_features.py", "night_features"
        )
        if config["action"] == "features":

            class TrackedSubprocess:
                Popen = staticmethod(spawn)

                def __getattr__(self, key):
                    return getattr(subprocess, key)

            harness.subprocess = TrackedSubprocess()
            sys.argv = [str(harness.__file__), *config["arguments"]]
            with (
                (folder / "feature.log").open("w") as log,
                contextlib.redirect_stdout(log),
                contextlib.redirect_stderr(log),
            ):
                code = harness.main()
            emit({"status": "done", "returncode": code})
            return code
        if config["variant"] == "tp-finalize":
            checks = load_module(
                repo / "benchmark/prefill_1m/feature_checks.py", "finalize_check"
            )
            schemes = checks.expert_quantization(config["checkpoint"])
            if not schemes or not schemes <= {"W8A8", "W8A8_DYNAMIC"}:
                raise ValueError(
                    f"GMM2 finalizer requires W8A8 expert weights; found {sorted(schemes)}"
                )
        command, env = variant_launch(harness, repo, config)
        save(
            folder / "launch.json",
            {
                "command": command,
                "env": {
                    k: v
                    for k, v in env.items()
                    if k
                    in set(BASE_FLAGS)
                    | {
                        name
                        for entry in harness.SCENARIOS.values()
                        for name in entry.get("env", {})
                    }
                    | {
                        "SGLANG_NPU_FEATURE_EVIDENCE",
                        "SGLANG_DEBUG_STAGE_LATENCY",
                        "SGLANG_NPU_DEBUG_DCP",
                    }
                },
            },
        )
        with (folder / "server.log").open("w") as log:
            process = spawn(
                command,
                cwd=repo,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            (folder / "server.pid").write_text(str(process.pid))
            deadline = time.monotonic() + config["startup_timeout"]
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    reason = server_failure(
                        (folder / "server.log").read_text(errors="replace")
                    )
                    raise RuntimeError(
                        reason or f"Server exited {process.returncode}; see server.log"
                    )
                try:
                    http("/health")
                    info = http("/server_info")
                    break
                except (OSError, ValueError):
                    time.sleep(2)
            else:
                raise TimeoutError("Server startup timed out")
            save(folder / "server_info.json", info)
            resolved = info.get("server_args", info)
            validate_variant_topology(harness, config, resolved)
            overrides = resolved.get("json_model_override_args") or {}
            if isinstance(overrides, str):
                overrides = json.loads(overrides)
            if (
                resolved.get("load_format") == "dummy"
                or overrides.get("num_hidden_layers") is not None
            ):
                raise RuntimeError("Full real weights are required")
            contents = (folder / "server.log").read_text(errors="replace")
            capacities = re.findall(r"max_total_num_tokens=(\d+)", contents)
            required = required_kv_capacity(config, resolved)
            if not capacities or min(map(int, capacities)) < required:
                raise RuntimeError(
                    f"KV capacity cannot cover shared-prefix working set {required}"
                )
            emit(
                {
                    "status": "ready",
                    "pid": process.pid,
                    "capacity": min(map(int, capacities)),
                    "resolved_server_args": resolved,
                    "max_running_requests": resolved["max_running_requests"],
                    "prefill_max_requests": resolved.get("prefill_max_requests"),
                    "scenario": harness.scenario_for(
                        config["variant"], config.get("reference_recipe", "ordinary")
                    ),
                }
            )
            while process.poll() is None:
                time.sleep(1)
            raise RuntimeError(f"Server exited during benchmark: {process.returncode}")
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # noqa: BLE001 - retain failures and continue independent cases
        emit({"status": "error", "error": str(exc)})
        return 1
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for process in reversed(owned):
            stop_owned(process, include_exited=config["action"] == "serve")
        try:
            collect_native_dumps(repo, folder, started)
        except OSError as exc:
            (folder / "native-dumps-error.log").write_text(str(exc))


class ContainerJob:
    def __init__(self, args, folder, config):
        self.process = None
        self.log = (folder / "worker.log").open("w")
        command = [
            "docker",
            "exec",
            "-i",
            args.container,
            args.container_python,
            f"{args.repo}/{RELATIVE}",
            "--worker",
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            start_new_session=True,
        )
        self.process.stdin.write(json.dumps(config) + "\n")
        self.process.stdin.flush()

    def receive(self, timeout):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                raise TimeoutError("Container worker timed out")
            line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(
                f"Container worker exited: {self.process.poll()}; see worker.log"
            )
        result = json.loads(line)
        self.log.write(json.dumps(dict(worker_response=result)) + "\n")
        self.log.flush()
        if result["status"] == "error":
            raise RuntimeError(result["error"])
        return result

    def close(self):
        if self.process is not None:
            try:
                self.process.stdin.write('{"stop":true}\n')
                self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            with contextlib.suppress(BrokenPipeError, OSError):
                self.process.stdin.close()
            try:
                self.process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                stop_owned(self.process)
        self.log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--container", default="sglang-a3")
    parser.add_argument("--container-python", default="python3")
    parser.add_argument("--repo", default="/home/swx1199799/megamoe_test/sglang")
    parser.add_argument("--checkpoint", default="/home/weights/GLM5.2-w4a8-int")
    parser.add_argument(
        "--w8a8-checkpoint",
        help="Explicit ModelSlim W8A8 checkpoint for FuseEP modes 1/2.",
    )
    parser.add_argument(
        "--feature-kv-rows",
        type=int,
        help="Local KV cap shared by each qualification reference/candidate pair.",
    )
    parser.add_argument(
        "--ais-dir", default="/home/tbaydasov/aisbench/aisbench_auto_tools_prefix"
    )
    parser.add_argument(
        "--ais-env", help="Host environment script; auto-detect known paths"
    )
    parser.add_argument(
        "--check-paths", action="store_true", help="Check host AISBench paths and exit"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument(
        "--reference-recipe",
        choices=(
            "ordinary",
            "historical-bde",
            "historical-bde-hccl256",
            "historical-bde-capacity",
        ),
        default="ordinary",
    )
    parser.add_argument("--feature-cases", default="*")
    parser.add_argument(
        "--long-variants",
        default="dcp-baseline,dcp-packed-read,dcp-gather-prefetch,dcp-sfa-overlap,dcp-indexer-query-shard,dcp-multistream,dcp-packed-prefetch,dcp-packed-overlap,dcp-prefetch-overlap,dcp-io-combined,dcp-io-indexer,dcp-io-multistream",
    )
    parser.add_argument("--long-lengths", default="32768,1000000")
    parser.add_argument("--long-concurrency", default="1,2")
    parser.add_argument("--long-requests", type=int, default=2)
    parser.add_argument("--long-dcp-size", type=int, choices=(1, 16), default=16)
    parser.add_argument("--diagnostic-megamoe", action="store_true")
    parser.add_argument(
        "--performance-only",
        action="store_true",
        help="Skip separate qualification launches; label timings diagnostic with numerical quality unchecked",
    )
    parser.add_argument(
        "--allow-parity-failure",
        action="store_true",
        help="Time activated variants despite numerical parity failure; retain failed quality in reports",
    )
    parser.add_argument("--megamoe-dispatch-tokens", type=int, default=4096)
    parser.add_argument("--long-cache-modes", default="cold,prefix99")
    parser.add_argument("--long-warmup", choices=("once", "each"), default="each")
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--long-kv-rows", type=int, default=131072)
    parser.add_argument(
        "--feature-stages",
        default="host,npu,serve",
        help="Comma-separated host,npu,serve,long stages; AISBench requires serve.",
    )
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument(
        "--moe-only",
        action="store_true",
        help="Qualify and benchmark the MoE group only",
    )
    parser.add_argument("--moe-length", type=int, default=4096)
    parser.add_argument("--moe-output-len", type=int, default=128)
    parser.add_argument("--moe-repeat-rate", type=float, default=0.5)
    parser.add_argument("--moe-kv-rows", type=int, default=65536)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--length", type=int, default=1000000)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--maxreq", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=32768)
    parser.add_argument("--kv-rows", type=int, default=81920)
    parser.add_argument("--context", type=int, default=1048576)
    parser.add_argument("--memfrac", type=float, default=0.88)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--benchmark-timeout", type=int, default=14400)
    parser.add_argument("--watchdog-timeout", type=int, default=7200)
    parser.add_argument("--feature-timeout", type=int, default=43200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.megamoe_dispatch_tokens <= 4096:
        parser.error("--megamoe-dispatch-tokens must be in [1,4096]")
    if args.feature_kv_rows is not None and args.feature_kv_rows <= 0:
        parser.error("--feature-kv-rows must be positive")
    try:
        lengths = [int(x) for x in args.long_lengths.split(",")]
        concurrencies = [int(x) for x in args.long_concurrency.split(",")]
    except ValueError:
        parser.error("Long lengths and concurrency must be comma-separated integers")
    if min(*lengths, *concurrencies, args.long_requests, args.long_kv_rows) <= 0:
        parser.error("Long workload settings must be positive")
    if "long" in args.feature_stages.split(",") and max(lengths) + 1 > args.context:
        parser.error("Long prompt must leave space for output in context")
    if args.worker:
        return worker()
    if args.max_hours is not None and not 0 < args.max_hours <= 24:
        parser.error("--max-hours must be in (0,24]")
    args.deadline_unix = time.time() + args.max_hours * 3600 if args.max_hours else None
    args.ais_env = resolve_ais_env(args.ais_env)
    if args.check_paths or not (args.features_only or args.dry_run):
        try:
            check_ais_paths(args.ais_dir, args.ais_env)
        except ValueError as error:
            parser.error(str(error))
    if args.check_paths:
        print(f"AISBench host paths OK: {args.ais_dir}; environment: {args.ais_env}")
        return 0
    if args.performance_only and args.features_only:
        parser.error("--performance-only cannot be combined with --features-only")
    stages = args.feature_stages.split(",")
    if (
        not stages
        or set(stages) - {"host", "npu", "serve", "long"}
        or len(set(stages)) != len(stages)
    ):
        parser.error("--feature-stages must select unique host,npu,serve,long stages")
    if not args.features_only and "serve" not in stages:
        parser.error("Benchmark timing requires serve qualification")
    if args.performance_only:
        selected = args.variants.split(",")
        supported = set(VARIANTS) - {"fuseep-1", "fuseep-2"}
        if (
            args.reference_recipe
            not in (
                "historical-bde",
                "historical-bde-hccl256",
                "historical-bde-capacity",
            )
            or not selected
            or len(set(selected)) != len(selected)
            or set(selected) - supported
            or args.moe_only
        ):
            parser.error(
                "--performance-only requires historical-bde and unique supported variants"
            )
        stages = []
    variants = list(MOE_VARIANTS) if args.moe_only else args.variants.split(",")
    if args.moe_only and args.feature_cases == "*":
        args.feature_cases = ",".join("serve-" + name for name in MOE_VARIANTS)
    if (
        args.moe_length + args.moe_output_len + 32 > 8192
        or not 0 < args.moe_repeat_rate < 1
    ):
        parser.error(
            "MoE workload must fit context 8192 and have repeat rate between zero and one"
        )
    if (
        set(variants) - set(VARIANTS)
        or min(
            args.repeats,
            args.length,
            args.requests,
            args.maxreq,
            args.chunk,
            args.kv_rows,
        )
        <= 0
    ):
        parser.error("Unknown variant or nonpositive workload setting")
    if args.context <= args.length + 32 or args.chunk % 128 or not 0 < args.memfrac < 1:
        parser.error(
            "Context must leave output space; chunk must align to 128; memfrac must be between zero and one"
        )
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = args.output or Path(args.ais_dir).parent / "overnight-results" / stamp
    remote = f"/tmp/sglang-bde-night-{stamp}-{os.getpid()}"
    common = {
        "repo": args.repo,
        "reference_recipe": args.reference_recipe,
        "performance_only": args.performance_only,
        "megamoe_dispatch_tokens": args.megamoe_dispatch_tokens,
        "checkpoint": args.checkpoint,
        "w8a8_checkpoint": args.w8a8_checkpoint,
        "chunk": args.chunk,
        "memfrac": args.memfrac,
        "maxreq": args.maxreq,
        "kv_rows": args.kv_rows,
        "context": args.context,
        "length": args.length,
        "requests": args.requests,
        "output_len": 1,
        "repeat_rate": 0.99,
        "startup_timeout": args.startup_timeout,
        "watchdog_timeout": args.watchdog_timeout,
    }
    feature_commands = {}
    for stage in stages:
        feature_commands[stage] = [
            "--stage",
            stage,
            "--cases",
            args.feature_cases if stage == "serve" else "*",
            "--model-path",
            args.checkpoint,
            "--checkpoint-format",
            "modelslim",
            "--reference-recipe",
            args.reference_recipe,
            "--megamoe-dispatch-tokens",
            str(args.megamoe_dispatch_tokens),
            "--tp",
            "16",
            "--dcp",
            "16",
            "--context-length",
            "8192",
            "--chunk",
            "4096",
            "--mem-fraction",
            str(args.memfrac),
            "--startup-timeout",
            str(args.startup_timeout),
            "--request-timeout",
            "7200",
            "--case-timeout",
            "14400",
            "--output",
            f"{remote}/{stage}/report.md",
        ]
        if args.deadline_unix:
            feature_commands[stage] += ["--deadline-unix", str(args.deadline_unix)]
        if args.w8a8_checkpoint:
            feature_commands[stage] += ["--w8a8-model-path", args.w8a8_checkpoint]
        if stage == "long":
            feature_commands[stage][feature_commands[stage].index("--dcp") + 1] = str(
                args.long_dcp_size
            )
            if args.diagnostic_megamoe:
                feature_commands[stage] += ["--diagnostic-megamoe"]
            feature_commands[stage][
                feature_commands[stage].index("--context-length") + 1
            ] = str(args.context)
            feature_commands[stage][feature_commands[stage].index("--chunk") + 1] = str(
                args.chunk
            )
            feature_commands[stage] += [
                "--benchmark-variants",
                args.long_variants,
                "--long-lengths",
                args.long_lengths,
                "--long-concurrency",
                args.long_concurrency,
                "--long-requests",
                str(args.long_requests),
                "--long-cache-modes",
                args.long_cache_modes,
                "--long-warmup",
                args.long_warmup,
                "--benchmark-repeats",
                str(args.repeats),
                "--kv-rows",
                str(args.long_kv_rows),
            ]
        elif args.feature_kv_rows is not None:
            feature_commands[stage] += ["--kv-rows", str(args.feature_kv_rows)]
    ais_args = [
        "--input_len",
        str(args.length),
        "--output_len",
        "1",
        "--request_rate",
        "0",
        "--dataset_type",
        "prefix_cache",
        "--prefix_test",
        "--seed",
        "4",
        "--repeat_rate",
        "0.99",
        "--data_num",
        str(args.requests),
        "--concurrency",
        str(args.requests),
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "container": args.container,
                    "feature_commands": feature_commands,
                    "server_settings": common,
                    "variants": variants,
                    "repeats": args.repeats,
                    "ais_args": ais_args,
                    "ais_dir": args.ais_dir,
                    "ais_env": args.ais_env,
                    "moe_workload": {
                        "input_len": args.moe_length,
                        "output_len": args.moe_output_len,
                        "repeat_rate": args.moe_repeat_rate,
                        "context": 8192,
                        "chunk": 4096,
                        "kv_rows": args.moe_kv_rows,
                    },
                    "host_output": str(output),
                    "container_output": remote,
                },
                indent=2,
            )
        )
        return 0

    def remaining_timeout(limit):
        if not args.deadline_unix:
            return limit
        remaining = args.deadline_unix - time.time()
        if remaining <= 0:
            raise TimeoutError("Overnight time budget exhausted")
        return min(limit, remaining)

    require_free_port()
    lock = open("/tmp/sglang-bde-overnight.lock", "w")  # noqa: SIM115 - held for the driver lifetime
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / "runner.py")
    save(output / "settings.json", vars(args) | {"output": str(output)})
    (output / "remote-path.txt").write_text(remote + "\n")
    inspect = json.loads(
        subprocess.check_output(["docker", "inspect", args.container], text=True)
    )[0]
    if (
        not inspect["State"]["Running"]
        or inspect["HostConfig"]["NetworkMode"] != "host"
    ):
        raise RuntimeError(
            "This runner requires the existing running container with host networking"
        )
    docker = ["docker", "exec", args.container]
    subprocess.run([*docker, "test", "-f", f"{args.repo}/{RELATIVE}"], check=True)
    metadata = subprocess.check_output(
        [*docker, "git", "-C", args.repo, "rev-parse", "HEAD"], text=True
    )
    (output / "commit.txt").write_text(metadata)
    records, qualification = [], {}

    def persist():
        save(output / "results.json", records)
        rows = [
            "# Overnight BDE results",
            "",
            "| Case | Status | TTFT mean / P90 (s) | E2E run (s) | Requests/s | Prefill tokens/s |",
            "|---|---|---|---|---|---|",
        ]
        for item in records:
            metric = item.get("metrics", {})
            ttft = metric.get("TTFT avg")
            p90 = metric.get("TTFT P90")
            rows.append(
                f"| {item['case']} | {item['status']} | {ttft / 1000 if ttft is not None else ''} / {p90 / 1000 if p90 is not None else ''} | {metric.get('E2E_time', '')} | {metric.get('qps', '')} | {metric.get('prefill_token_throughput', '')} |"
            )
        long_cases = [item for item in records if "benchmarks" in item]
        if long_cases:
            rows.extend(
                [
                    "",
                    "## Exact-token long-context measurements",
                    "",
                    "| Case | Tokens | Concurrency | Cache | Status | TTFT mean / P90 (s) | E2E mean (s) | Run wall mean (s) | Requests/s | Uncached prefill tokens/s |",
                    "|---|---|---|---|---|---|---|---|---|---|",
                ]
            )
            for item in long_cases:
                for point in item["benchmarks"]:
                    rows.append(
                        f"| {item['case']} | {point.get('length', '')} | {point.get('concurrency', '')} | {point.get('cache', '')} | {point['status']} | "
                        f"{point.get('mean_ttft_s', '')} / {point.get('p90_ttft_s', '')} | {point.get('mean_e2e_s', '')} | "
                        f"{point.get('mean_run_wall_s', '')} | {point.get('requests_per_s', '')} | {point.get('uncached_prefill_tokens_per_s', '')} |"
                    )
        rows.extend(
            [
                "",
                "Raw reports and server logs: `container-artifacts/`. Blocked/inactive cases are not passes.",
                "Historical-baseline AISBench comparisons use one output token (TPOT undefined); ordinary MoE workloads use multiple. Compare only matched workloads.",
                "diagnostic-parity-failed means numerical quality failed; diagnostic-unqualified means it was not checked. Neither is a qualified pass.",
            ]
        )
        (output / "SUMMARY.md").write_text("\n".join(rows) + "\n")

    try:
        for stage, arguments in feature_commands.items():
            folder = output / stage
            folder.mkdir()
            print(f"Feature stage {stage}; artifacts {remote}/{stage}", flush=True)
            job = ContainerJob(
                args,
                folder,
                dict(
                    common,
                    action="features",
                    folder=f"{remote}/{stage}",
                    arguments=arguments,
                ),
            )
            try:
                timeout = args.feature_timeout
                if args.deadline_unix:
                    timeout = min(timeout, max(0, args.deadline_unix - time.time()))
                result = job.receive(timeout)
                code = result["returncode"]
                record = {
                    "case": stage,
                    "status": "pass"
                    if code == 0
                    else "partial"
                    if code == 2
                    else "fail",
                }
                if stage in ("serve", "long"):
                    report = subprocess.check_output(
                        [*docker, "cat", f"{remote}/{stage}/report.md"], text=True
                    )
                    if stage == "serve":
                        qualification = report_cases(report)
                        for case in qualification.values():
                            if case.get("stage") == "serve":
                                records.append(
                                    dict(
                                        case=case["id"],
                                        status=case["status"],
                                        activation=case.get("activation"),
                                        quality=case.get("quality"),
                                        failure_phase=case.get("failure_phase"),
                                        reason=case.get("reason", case.get("error")),
                                    )
                                )
                    else:
                        for case in report_cases(report).values():
                            if case.get("stage") == "long":
                                records.append(
                                    {
                                        "case": case["id"],
                                        "status": case["status"],
                                        "benchmarks": case.get("benchmarks", []),
                                        "reason": case.get("reason", case.get("error")),
                                    }
                                )
                    (folder / "report.md").write_text(report)
            except Exception as exc:  # noqa: BLE001 - retain failures and continue independent cases
                record = {
                    "case": stage,
                    "status": "budget_blocked"
                    if args.deadline_unix and time.time() >= args.deadline_unix
                    else "fail",
                    "error": str(exc),
                }
            finally:
                job.close()
            records.append(record)
            persist()
        if not args.features_only:
            for variant in variants:
                if args.deadline_unix and time.time() + 900 > args.deadline_unix:
                    records.append(
                        dict(
                            case=variant,
                            status="budget_blocked",
                            reason="Less than 15 minutes remain for a complete benchmark",
                        )
                    )
                    persist()
                    continue
                gate = qualification.get("serve-" + variant, {})
                if not args.performance_only and not benchmark_admitted(
                    gate,
                    baseline=variant == "baseline",
                    allow_parity_failure=args.allow_parity_failure,
                ):
                    records.append(
                        {
                            "case": variant,
                            "status": "skipped-unqualified",
                            "reason": gate.get("status", "missing qualification"),
                        }
                    )
                    persist()
                    continue
                diagnostic = gate.get("failure_phase") == "parity"
                measurement_status = (
                    "diagnostic-unqualified"
                    if args.performance_only
                    else "diagnostic-parity-failed"
                    if diagnostic
                    else "pass"
                )
                measurement_quality = (
                    "not_checked"
                    if args.performance_only
                    else "failed"
                    if diagnostic
                    else gate.get("quality")
                )
                folder = output / variant
                folder.mkdir()
                print(f"AISBench {variant}, {args.repeats} repeats", flush=True)
                settings = dict(common)
                variant_ais_args = list(ais_args)
                if variant in MOE_VARIANTS and args.reference_recipe == "ordinary":
                    settings.update(
                        length=args.moe_length,
                        output_len=args.moe_output_len,
                        repeat_rate=args.moe_repeat_rate,
                        context=8192,
                        chunk=4096,
                        kv_rows=args.moe_kv_rows,
                    )
                    for flag, value in (
                        ("--input_len", args.moe_length),
                        ("--output_len", args.moe_output_len),
                        ("--repeat_rate", args.moe_repeat_rate),
                    ):
                        variant_ais_args[variant_ais_args.index(flag) + 1] = str(value)
                save(
                    folder / "workload.json", dict(settings, ais_args=variant_ais_args)
                )
                job = ContainerJob(
                    args,
                    folder,
                    dict(
                        settings,
                        action="serve",
                        variant=variant,
                        folder=f"{remote}/{variant}",
                    ),
                )
                phase = "server-startup"
                try:
                    ready = job.receive(remaining_timeout(args.startup_timeout + 30))
                    save(folder / "resolved-capacity.json", ready)
                    settings.update(
                        maxreq=ready.get("max_running_requests", settings["maxreq"]),
                        prefill_max_requests=ready.get("prefill_max_requests"),
                    )
                    save(
                        folder / "workload.json",
                        dict(
                            settings,
                            ais_args=variant_ais_args,
                            resolved_server_args=ready.get("resolved_server_args"),
                        ),
                    )
                    for repeat in range(1, args.repeats + 1):
                        # Each AISBench invocation performs its own prefix prewarm after this flush.
                        phase = f"cache-flush/{repeat}"
                        flushed = http(
                            "/flush_cache?timeout=60",
                            post=True,
                            timeout=remaining_timeout(90),
                            expect_json=False,
                        )
                        (folder / f"cache-flush-{repeat}.txt").write_text(flushed)
                        phase = f"aisbench/{repeat}"
                        logpath = folder / f"aisbench-{repeat}.log"
                        command = [
                            "bash",
                            "--noprofile",
                            "--norc",
                            "-c",
                            AISBENCH_SHELL,
                            "aisbench",
                            args.ais_env,
                            args.ais_dir,
                            *variant_ais_args,
                        ]
                        with logpath.open("w") as log:
                            process = subprocess.Popen(
                                command,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                start_new_session=True,
                            )
                            try:
                                code = wait_benchmark(
                                    process,
                                    job.process,
                                    remaining_timeout(args.benchmark_timeout),
                                )
                            finally:
                                stop_owned(process, include_exited=True)
                        if code:
                            raise RuntimeError(f"AISBench exited {code}; see {logpath}")
                        client_log = logpath.read_text(errors="replace")
                        try:
                            cold = ais_metrics(client_log, 1)
                        except RuntimeError as exc:
                            cold = {"status": "incomplete", "error": str(exc)}
                        save(folder / f"cold-prefix-metrics-{repeat}.json", cold)
                        metrics = ais_metrics(client_log, args.requests)
                        save(folder / f"metrics-{repeat}.json", metrics)
                        csv = Path(args.ais_dir) / "aisbench_result.csv"
                        if csv.is_file():
                            shutil.copyfile(
                                csv, folder / f"aisbench-results-{repeat}.csv"
                            )
                        if args.performance_only:
                            log = subprocess.check_output(
                                [*docker, "cat", f"{remote}/{variant}/server.log"],
                                text=True,
                            )
                            features = timed_activation(
                                log, variant, ready.get("scenario")
                            )
                            save(folder / "activation.json", features)
                        records.append(
                            {
                                "case": f"{variant}/{repeat}",
                                "status": measurement_status,
                                "quality": measurement_quality,
                                "qualification_failure": gate.get("error")
                                if diagnostic
                                else None,
                                "metrics": metrics,
                            }
                        )
                        persist()
                    values = [
                        r["metrics"]["TTFT avg"]
                        for r in records
                        if r["case"].startswith(variant + "/")
                        and r["status"]
                        in (
                            "pass",
                            "diagnostic-parity-failed",
                            "diagnostic-unqualified",
                        )
                    ]
                    save(
                        folder / "aggregate.json",
                        {
                            "median_ttft_ms": statistics.median(values),
                            "runs": len(values),
                            "status": measurement_status,
                            "quality": measurement_quality,
                        },
                    )
                except Exception as exc:  # noqa: BLE001 - retain failures and continue independent cases
                    (folder / "exception.log").write_text(traceback.format_exc())
                    failure_phase, error = phase, str(exc)
                    snapshot = subprocess.run(
                        [*docker, "cat", f"{remote}/{variant}/server.log"],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if snapshot.returncode == 0:
                        (folder / "server-failure.log").write_text(snapshot.stdout)
                        reason = server_failure(snapshot.stdout)
                        if reason:
                            failure_phase, error = (
                                "server-execution",
                                f"{reason}; client: {exc}",
                            )
                    records.append(
                        {
                            "case": variant,
                            "status": "fail",
                            "phase": failure_phase,
                            "client_phase": phase,
                            "error": error,
                            "log": str(folder / "exception.log"),
                        }
                    )
                finally:
                    job.close()
                persist()
    finally:
        persist()
        copied = subprocess.run(
            [
                "docker",
                "cp",
                f"{args.container}:{remote}",
                str(output / "container-artifacts"),
            ],
            check=False,
        )
        if copied.returncode:
            print(
                f"Artifact copy failed; recover {args.container}:{remote}",
                file=sys.stderr,
            )
    lock.close()
    print(f"Results: {output / 'SUMMARY.md'}", flush=True)
    return (
        1
        if any(r["status"] == "fail" for r in records)
        else 2
        if any(r["status"] != "pass" for r in records)
        else 0
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted; owned server cleanup requested.", file=sys.stderr)
        sys.exit(130)
