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
import urllib.request
from pathlib import Path

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
)
MOE_VARIANTS = (
    "baseline",
    "tp-routed-scale",
    "tp-finalize",
    "tp-prequant",
    "tp-shared",
    "tp-shared-activation",
    "multistream",
    "auto-stream-thresholds",
    "auto-hccl",
    "autotune-hccl",
    "fuseep-1",
    "fuseep-2",
    "megamoe",
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


def http(path, *, post=False, timeout=10):
    request = urllib.request.Request(
        "http://127.0.0.1:30000" + path,
        data=b"{}" if post else None,
    )
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
        request, timeout=timeout
    ) as response:
        body = response.read()
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


def variant_launch(harness, repo, config):
    scenario = harness.SCENARIOS[config["variant"]]
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
        EXTRA_ARGS=f"--quantization modelslim --watchdog-timeout 7200 --max-total-tokens {config['kv_rows']}",
    )
    if scenario.get("dcp"):
        inherited["PREFILL_MAXREQ"] = "1"
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
        SGLANG_NPU_FEATURE_EVIDENCE="False",
        SGLANG_DEBUG_STAGE_LATENCY="False",
        SGLANG_NPU_DEBUG_DCP="False",
    )
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


def worker():
    """Runs inside Docker. EOF or stop on stdin cleans up only owned children."""
    config = json.loads(sys.stdin.readline())
    repo, folder = Path(config["repo"]), Path(config["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    os.chdir(repo)
    owned = []

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
                    raise RuntimeError(
                        f"Server exited {process.returncode}; see server.log"
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
            for key, expected in (
                ("tp_size", 16),
                (
                    "dcp_size",
                    16 if harness.SCENARIOS[config["variant"]].get("dcp") else 1,
                ),
                ("max_running_requests", config["maxreq"]),
                ("quantization", "modelslim"),
                ("cuda_graph_backend_prefill", "disabled"),
                ("cuda_graph_backend_decode", "disabled"),
            ):
                if resolved.get(key) != expected:
                    raise RuntimeError(f"Unexpected {key}: {resolved.get(key)}")
            if (
                harness.SCENARIOS[config["variant"]].get("dcp")
                and resolved.get("prefill_max_requests") != 1
            ):
                raise RuntimeError("DCP timing requires prefill_max_requests=1")
            fuseep = harness.SCENARIOS[config["variant"]].get("fuseep")
            if fuseep and (
                resolved.get("moe_ep_size") != 16
                or resolved.get("fuseep_mode") != fuseep
                or resolved.get("moe_a2a_backend") != "ascend_fuseep"
            ):
                raise RuntimeError(
                    "Resolved FuseEP topology/mode differs from qualification"
                )
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
            required = (
                int(
                    config["length"]
                    * (
                        config["repeat_rate"]
                        + (1 - config["repeat_rate"])
                        * min(config["maxreq"], config["requests"])
                    )
                    + config["output_len"] * min(config["maxreq"], config["requests"])
                )
                + 4096
            )
            if not capacities or min(map(int, capacities)) < required:
                raise RuntimeError(
                    f"KV capacity cannot cover shared-prefix working set {required}"
                )
            emit(
                {
                    "status": "ready",
                    "pid": process.pid,
                    "capacity": min(map(int, capacities)),
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
        "--feature-kv-rows", type=int,
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
    parser.add_argument("--feature-cases", default="*")
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
    parser.add_argument("--feature-timeout", type=int, default=43200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.feature_kv_rows is not None and args.feature_kv_rows <= 0:
        parser.error("--feature-kv-rows must be positive")
    if args.worker:
        return worker()
    args.ais_env = resolve_ais_env(args.ais_env)
    if args.check_paths or not (args.features_only or args.dry_run):
        try:
            check_ais_paths(args.ais_dir, args.ais_env)
        except ValueError as error:
            parser.error(str(error))
    if args.check_paths:
        print(f"AISBench host paths OK: {args.ais_dir}; environment: {args.ais_env}")
        return 0
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
    }
    feature_commands = {}
    for stage in ("host", "npu", "serve"):
        feature_commands[stage] = [
            "--stage",
            stage,
            "--cases",
            args.feature_cases if stage == "serve" else "*",
            "--model-path",
            args.checkpoint,
            "--checkpoint-format",
            "modelslim",
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
        if args.w8a8_checkpoint:
            feature_commands[stage] += ["--w8a8-model-path", args.w8a8_checkpoint]
        if args.feature_kv_rows is not None:
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
    require_free_port()
    lock = open("/tmp/sglang-bde-overnight.lock", "w")  # noqa: SIM115 - held for the driver lifetime
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / "runner.py")
    save(output / "settings.json", vars(args) | {"output": str(output)})
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
        rows.extend(
            [
                "",
                "Raw reports and server logs: `container-artifacts/`. Blocked/inactive cases are not passes.",
                "DCP cases use one output token (TPOT undefined); MoE cases use multi-token outputs. Compare only within a workload group.",
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
                result = job.receive(args.feature_timeout)
                code = result["returncode"]
                record = {
                    "case": stage,
                    "status": "pass"
                    if code == 0
                    else "partial"
                    if code == 2
                    else "fail",
                }
                if stage == "serve":
                    report = subprocess.check_output(
                        [*docker, "cat", f"{remote}/serve/report.md"], text=True
                    )
                    qualification = report_cases(report)
                    (folder / "report.md").write_text(report)
            except Exception as exc:  # noqa: BLE001 - retain failures and continue independent cases
                record = {"case": stage, "status": "fail", "error": str(exc)}
            finally:
                job.close()
            records.append(record)
            persist()
        if not args.features_only:
            for variant in variants:
                gate = qualification.get("serve-" + variant, {})
                if (
                    gate.get("status") != "pass"
                    or gate.get("quality") != "verified"
                    or gate.get("missing")
                    or (variant != "baseline" and gate.get("activation") != "verified")
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
                folder = output / variant
                folder.mkdir()
                print(f"AISBench {variant}, {args.repeats} repeats", flush=True)
                settings = dict(common)
                variant_ais_args = list(ais_args)
                if variant in MOE_VARIANTS:
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
                try:
                    job.receive(args.startup_timeout + 30)
                    for repeat in range(1, args.repeats + 1):
                        # Each AISBench invocation performs its own prefix prewarm after this flush.
                        http("/flush_cache?timeout=60", post=True, timeout=90)
                        logpath = folder / f"aisbench-{repeat}.log"
                        command = [
                            "bash",
                            "--noprofile",
                            "--norc",
                            "-c",
                            'set -e; source "$1"; cd "$2"; unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy ALL_PROXY all_proxy; export NO_PROXY=localhost,127.0.0.1,::1; exec python3 aisbench_test.py "${@:3}"',
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
                                code = process.wait(timeout=args.benchmark_timeout)
                            finally:
                                stop_owned(process, include_exited=True)
                        if code:
                            raise RuntimeError(f"AISBench exited {code}; see {logpath}")
                        metrics = ais_metrics(
                            logpath.read_text(errors="replace"), args.requests
                        )
                        csv = Path(args.ais_dir) / "aisbench_result.csv"
                        if csv.is_file():
                            shutil.copyfile(
                                csv, folder / f"aisbench-results-{repeat}.csv"
                            )
                        records.append(
                            {
                                "case": f"{variant}/{repeat}",
                                "status": "pass",
                                "metrics": metrics,
                            }
                        )
                        persist()
                    values = [
                        r["metrics"]["TTFT avg"]
                        for r in records
                        if r["case"].startswith(variant + "/") and r["status"] == "pass"
                    ]
                    save(
                        folder / "aggregate.json",
                        {
                            "median_ttft_ms": statistics.median(values),
                            "runs": len(values),
                        },
                    )
                except Exception as exc:  # noqa: BLE001 - retain failures and continue independent cases
                    records.append(
                        {"case": variant, "status": "fail", "error": str(exc)}
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
