"""Staged, restart-based NPU tuning with retained logs and conservative selection."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

# Permit --help and policy tests on hosts without the SGLang runtime dependencies.
_policy_path = (
    Path(__file__).resolve().parents[2]
    / "python/sglang/srt/hardware_backend/npu/autotune.py"
)
_spec = importlib.util.spec_from_file_location("npu_tuning_policy", _policy_path)
policy = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = policy
_spec.loader.exec_module(policy)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--server-args-file", type=Path, required=True)
    p.add_argument("--benchmark-command-file", type=Path, required=True)
    p.add_argument(
        "--correctness-command-file",
        type=Path,
        help="Must pass for every candidate before publishing a profile",
    )
    p.add_argument(
        "--analyzer-command-file",
        type=Path,
        help="Write analyzer.json in AUTOTUNE_RESULT_DIR; may include forward_mean_ms, forward_median_ms and profiler_buckets",
    )
    p.add_argument(
        "--context-file",
        type=Path,
        required=True,
        help="JSON NpuTuningContext from a matching startup",
    )
    p.add_argument(
        "--metric", choices=("median_ttft", "mean_ttft"), default="median_ttft"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=30088)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--python", default=sys.executable)
    p.add_argument(
        "--stages",
        nargs="+",
        choices=("expansion", "buffer", "task_queue", "page", "chunk", "concurrency"),
        default=["expansion", "buffer", "task_queue", "page"],
    )
    p.add_argument("--concurrency-candidates", nargs="+", type=int)
    p.add_argument(
        "--chunk-max",
        type=int,
        help="Required to include chunk tuning; declare a supported memory limit",
    )
    p.add_argument("--runs", type=int, default=2)
    p.add_argument(
        "--expected-requests",
        type=int,
        help="Reject partially completed benchmark trials",
    )
    p.add_argument("--startup-timeout", type=float, default=900)
    p.add_argument("--benchmark-timeout", type=float, default=3600)
    p.add_argument(
        "--tune-explicit",
        action="store_true",
        help="Authorize varying explicitly supplied values in requested stages",
    )
    p.add_argument("--publish-profile", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def option_value(argv, name):
    for i, arg in enumerate(argv):
        if arg == name:
            return argv[i + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None


def set_option(argv, name, value):
    result = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg == name:
            skip = True
        elif not arg.startswith(name + "="):
            result.append(arg)
    if value is not None:
        result.extend([name, str(value)])
    return result


def candidates(stage, context, args):
    estimate = policy.hccl_buffer_mb(
        policy.representative_prefill_rows(context),
        context.hidden_size,
        4 if "float32" in context.dtype else 2,
    )
    if stage == "expansion":
        return "HCCL_OP_EXPANSION_MODE", [None, "AIV"]
    if stage == "buffer":
        return "HCCL_BUFFSIZE", sorted(
            {
                max(64, min(1024, math.ceil(estimate * f / 32) * 32))
                for f in (0.5, 0.75, 1, 1.5, 2)
            }
        )
    if stage == "task_queue":
        return "TASK_QUEUE_ENABLE", [None, "2"]
    if stage == "page":
        return "--page-size", [64, 128]
    if stage == "chunk":
        if not args.chunk_max:
            raise ValueError("--chunk-max is required for chunk tuning")
        base = context.chunked_prefill_size
        return "--chunked-prefill-size", sorted(
            {int(base * f) for f in (0.5, 1, 2) if 0 < base * f <= args.chunk_max}
        )
    if not args.concurrency_candidates:
        raise ValueError("concurrency tuning requires --concurrency-candidates")
    return "--max-running-requests", args.concurrency_candidates


def stop(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    except ProcessLookupError:
        pass


def wait_ready(process, url, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with {process.returncode}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(1)
    raise TimeoutError("server readiness timeout")


def command(file, directory, env, timeout, log_name):
    with (directory / log_name).open("w") as log:
        proc = subprocess.Popen(
            ["bash", str(file.resolve())],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            status = proc.wait(timeout=timeout)
            if status:
                raise RuntimeError(f"{log_name} exited with {status}")
        finally:
            stop(proc)
    return (directory / log_name).read_text()


def parse_metrics(output):
    result = {}
    for metric, label in (("mean_ttft", "Mean TTFT"), ("median_ttft", "Median TTFT")):
        match = re.search(re.escape(label) + r" \(ms\):\s*([\d.eE+-]+)", output)
        if match:
            result[metric] = float(match[1])
    if not result or any(not math.isfinite(v) or v <= 0 for v in result.values()):
        raise ValueError("benchmark emitted no valid positive TTFT metrics")
    # A failed request cannot make a candidate look faster.
    completed = re.search(r"Successful requests:\s*(\d+)", output)
    if completed:
        result["completed"] = int(completed[1])
        if result["completed"] == 0:
            raise ValueError("benchmark completed zero requests")
    return result


def run_case(args, argv, overrides, directory, repetitions):
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # Tuning must not recursively apply an old profile or run load-time timers.
    for name in (
        "SGLANG_NPU_TUNING_PROFILE",
        "SGLANG_NPU_AUTO_HCCL_BUFFSIZE",
        "SGLANG_NPU_AUTO_STREAM_THRESHOLDS",
    ):
        env[name] = "0"
    for name, value in overrides.items():
        if name.startswith("--"):
            argv = set_option(argv, name, value)
        elif value is None:
            env.pop(name, None)
        else:
            env[name] = str(value)
    argv = set_option(argv, "--model-path", args.model_path)
    argv = set_option(argv, "--port", args.port)
    argv = set_option(argv, "--host", args.host)
    env.update(
        AUTOTUNE_PORT=str(args.port),
        AUTOTUNE_RESULT_DIR=str(directory.resolve()),
        PROFILE_DIR=str((directory / "profile").resolve()),
    )
    record = {
        "settings": overrides.copy(),
        "runs": [],
        "status": "failed",
        "correctness": False,
    }
    proc = None
    start = time.monotonic()
    try:
        try:
            connection = socket.create_connection((args.host, args.port), timeout=2)
        except ConnectionRefusedError:
            pass
        else:
            connection.close()
            raise RuntimeError("server port is already in use")
        with (directory / "server.log").open("w") as log:
            proc = subprocess.Popen(
                [args.python, "-m", "sglang.launch_server", *argv],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            wait_ready(proc, f"http://{args.host}:{args.port}", args.startup_timeout)
            if args.correctness_command_file:
                command(
                    args.correctness_command_file,
                    directory,
                    env,
                    args.benchmark_timeout,
                    "correctness.log",
                )
                record["correctness"] = True
            for n in range(repetitions):
                env["PROFILE_DIR"] = str((directory / "profile" / str(n)).resolve())
                env["AUTOTUNE_RUN_INDEX"] = str(n)
                env["AUTOTUNE_BENCHMARK_LOG"] = str(
                    (directory / f"benchmark-{n}.log").resolve()
                )
                # Clear request/prefix state before each repetition.
                request = urllib.request.Request(
                    f"http://{args.host}:{args.port}/flush_cache",
                    data=b"",
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 200:
                        raise RuntimeError("prefix cache flush failed")
                trial_start = time.monotonic()
                output = command(
                    args.benchmark_command_file,
                    directory,
                    env,
                    args.benchmark_timeout,
                    f"benchmark-{n}.log",
                )
                metrics = parse_metrics(output)
                if (
                    args.expected_requests is not None
                    and metrics.get("completed") != args.expected_requests
                ):
                    raise ValueError(
                        "benchmark did not complete the expected request count"
                    )
                metrics["wall_seconds"] = time.monotonic() - trial_start
                if args.analyzer_command_file:
                    analyzer_path = directory / "analyzer.json"
                    if analyzer_path.exists():
                        analyzer_path.unlink()
                    command(
                        args.analyzer_command_file,
                        directory,
                        env,
                        args.benchmark_timeout,
                        f"analyzer-{n}.log",
                    )
                    metrics.update(json.loads(analyzer_path.read_text()))
                record["runs"].append(metrics)
            record["status"] = "ok"
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        record["error"] = str(exc)
        if (directory / "server.log").exists() and "out of memory" in (
            directory / "server.log"
        ).read_text(errors="replace").lower():
            record["status"] = "oom"
    finally:
        stop(proc)
        record["total_wall_seconds"] = time.monotonic() - start
    (directory / "result.json").write_text(json.dumps(record, indent=2))
    return record


def rank_stage(records, metric, baseline):
    valid = {
        i: [r[metric] for r in record["runs"]]
        for i, record in enumerate(records)
        if record["status"] == "ok" and all(metric in r for r in record["runs"])
    }
    if baseline not in valid:
        return baseline, "baseline failed; no automatic selection"
    baseline_completed = records[baseline]["runs"][0].get("completed")
    if baseline_completed is not None:
        valid = {
            i: samples
            for i, samples in valid.items()
            if all(r.get("completed") == baseline_completed for r in records[i]["runs"])
        }
        if baseline not in valid:
            return baseline, "baseline request counts varied"
    winner, status = policy.choose_measured(valid, baseline)
    if winner != baseline:
        for key in ("forward_mean_ms", "forward_median_ms"):
            base = [r[key] for r in records[baseline]["runs"] if key in r]
            new = [r[key] for r in records[winner]["runs"] if key in r]
            if base and new and statistics.median(new) > statistics.median(base):
                return baseline, "forward time regressed"
    return winner, status


def main():
    p = parser()
    args = p.parse_args()
    if args.runs < 2:
        p.error("at least two runs are required")
    if args.publish_profile and not args.correctness_command_file:
        p.error("--publish-profile requires --correctness-command-file")
    context = policy.NpuTuningContext(**json.loads(args.context_file.read_text()))
    argv = shlex.split(args.server_args_file.read_text(), comments=True)
    for option, expected in (
        ("--tp-size", context.tp_size),
        ("--page-size", context.page_size),
        ("--chunked-prefill-size", context.chunked_prefill_size),
    ):
        supplied = option_value(argv, option)
        if supplied is not None and int(supplied) != expected:
            p.error(f"context disagrees with {option}")
    # Make workload-defining settings reproducible even when absent from the file.
    argv = set_option(argv, "--page-size", context.page_size)
    argv = set_option(argv, "--chunked-prefill-size", context.chunked_prefill_size)
    original_argv = shlex.split(args.server_args_file.read_text(), comments=True)
    best = {}
    result = {
        "schema_version": 1,
        "context": context.key_data(),
        "stages": [],
        "selected": {},
    }
    root = args.output.with_suffix("")
    cached = policy.load_profile(context)
    for stage in args.stages:
        name, values = candidates(stage, context, args)
        explicit = (
            option_value(original_argv, name)
            if name.startswith("--")
            else os.environ.get(name)
        )
        if explicit is not None and not args.tune_explicit:
            result["stages"].append(
                {
                    "stage": stage,
                    "status": "explicit value preserved",
                    "value": explicit,
                }
            )
            continue
        baseline = best.get(
            name,
            option_value(argv, name) if name.startswith("--") else os.environ.get(name),
        )
        values = [baseline, *[v for v in values if str(v) != str(baseline)]]
        plan = {"stage": stage, "parameter": name, "candidates": values}
        if args.dry_run:
            result["stages"].append(plan)
            continue
        records = []
        for i, value in enumerate(values):
            settings = {**best, name: value}
            # A validated identical case may be reused; no losing candidate rerun.
            evidence = cached.get("restart_evidence", {}).get(
                json.dumps(settings, sort_keys=True)
            )
            if (
                evidence
                and evidence.get("status") == "ok"
                and evidence.get("correctness")
                and len(evidence.get("runs", [])) >= args.runs
            ):
                records.append(evidence)
            else:
                records.append(
                    run_case(args, argv, settings, root / stage / str(i), args.runs)
                )
        valid = [r for r in records if r["status"] == "ok"]
        if len(valid) > 1:
            medians = sorted(
                statistics.median(r[args.metric] for r in item["runs"])
                for item in valid
            )
            if medians[1] / medians[0] < 1.01:
                records = [
                    run_case(
                        args,
                        argv,
                        r["settings"],
                        root / stage / f"confirm-{i}",
                        max(3, args.runs),
                    )
                    if r["status"] == "ok" and len(r["runs"]) < 3
                    else r
                    for i, r in enumerate(records)
                ]
        winner, status = rank_stage(records, args.metric, 0)
        best[name] = records[winner]["settings"][name]
        result["stages"].append(
            {**plan, "status": status, "winner": winner, "results": records}
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    mapping = {
        "HCCL_BUFFSIZE": "hccl_buffsize_mb",
        "HCCL_OP_EXPANSION_MODE": "hccl_op_expansion_mode",
        "TASK_QUEUE_ENABLE": "task_queue_enable",
        "--page-size": "page_size",
        "--chunked-prefill-size": "chunked_prefill_size",
        "--max-running-requests": "max_running_requests",
    }
    selected = {
        mapping[k]: int(v)
        if k in ("HCCL_BUFFSIZE", "TASK_QUEUE_ENABLE") and v is not None
        else v
        for k, v in best.items()
    }
    result["selected"] = selected
    if args.publish_profile and not args.dry_run:
        if "unknown" in (
            context.device_name,
            context.device_arch,
            context.cann_version,
            context.torch_npu_version,
            context.sglang_version,
        ):
            raise RuntimeError(
                "refusing profile publication: hardware/runtime identity is incomplete"
            )
        if not any("results" in stage for stage in result["stages"]):
            raise RuntimeError("refusing profile publication: no measured stages")
        ok = all(
            stage.get("results", [{}])[stage.get("winner", 0)].get("correctness", False)
            for stage in result["stages"]
            if "results" in stage
        )
        if not ok:
            raise RuntimeError(
                "refusing profile publication: winning candidate lacks correctness acceptance"
            )
        final_context = replace(
            context,
            page_size=selected.get("page_size", context.page_size),
            chunked_prefill_size=selected.get(
                "chunked_prefill_size", context.chunked_prefill_size
            ),
            max_running_requests=selected.get(
                "max_running_requests", context.max_running_requests
            ),
        )
        evidence = {
            json.dumps(r["settings"], sort_keys=True): r
            for stage in result["stages"]
            for r in stage.get("results", [])
        }
        profile_selected = {
            **policy.load_profile(final_context),
            **selected,
            "restart_evidence": evidence,
        }
        result["profile_path"] = str(
            policy.save_profile(
                final_context,
                profile_selected,
                {"metric": args.metric, "results": str(args.output.resolve())},
            )
        )
        result["profile_context"] = final_context.key_data()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"selected": selected, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
