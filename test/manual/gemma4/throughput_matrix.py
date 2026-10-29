"""Run controlled Gemma 4 TP1/DP2 benchmarks and enforce scaled scenario SLOs.

This launches and terminates only its own server process group. It never edits
checkouts, datasets, or the external bench_updated.py script.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

TARGETS = {
    "recommendation": (
        21.75,
        {
            "mean_ttft_ms": 400,
            "p99_ttft_ms": 1000,
            "mean_tpot_ms": 20,
            "p99_tpot_ms": 30,
            "p99_e2el_ms": 4000,
        },
    ),
    "rag": (
        8.0,
        {
            "mean_ttft_ms": 500,
            "p99_ttft_ms": 1500,
            "mean_tpot_ms": 20,
            "p99_tpot_ms": 30,
            "p99_e2el_ms": 9000,
        },
    ),
}
CASES = ("avg_avg", "high_high", "avg_high", "high_avg")
# Neighboring entries in the enabled sweep change one setting.
SCHEDULES = {
    "base": (0.5, 50, 6),
    "off": None,
    "r025-ms50-cap6": (0.25, 50, 6),
    "r025-ms25-cap6": (0.25, 25, 6),
    "r05-ms25-cap6": (0.5, 25, 6),
    "r05-ms25-cap5": (0.5, 25, 5),
    "r05-ms50-cap5": (0.5, 50, 5),
    "r025-ms50-cap5": (0.25, 50, 5),
    "r025-ms25-cap5": (0.25, 25, 5),
}


def request(url, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
        if url.endswith("/health"):
            return {"healthy": True}
        return json.loads(raw) if raw else {}


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def evaluate_case(raw, scenario, count, output_len=None):
    rate, limits = TARGETS[scenario]
    checks = {
        name: finite_number(raw.get(name)) and 0 <= raw[name] <= limit
        for name, limit in limits.items()
    }
    checks["completed"] = raw.get("completed") == count
    checks["failed"] = raw.get("failed") == 0
    checks["throughput"] = (
        finite_number(raw.get("request_throughput"))
        and raw["request_throughput"] >= rate
    )
    if output_len is not None:
        checks["output_tokens"] = raw.get("total_output_tokens") == count * output_len
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "target_rps": rate,
        "completed_rps": raw.get("request_throughput"),
        "metrics_ms": {name: raw.get(name) for name in limits},
    }


def validate_datasets(directory, requests, warmups):
    hashes = {}
    for scenario in TARGETS:
        for case in CASES:
            prompts = []
            for suffix, count in [("", requests), (".warmup", warmups)]:
                if not count:
                    prompts.append(set())
                    continue
                path = directory / scenario / f"{case}{suffix}.jsonl"
                content = path.read_bytes()
                rows = [
                    json.loads(line)["prompt"] for line in content.decode().splitlines()
                ]
                if len(rows) != count or len(set(rows)) != count:
                    raise ValueError(f"{path}: expected {count} unique prompts")
                prompts.append(set(rows))
                hashes[str(path)] = hashlib.sha256(content).hexdigest()
            if prompts[0] & prompts[1]:
                raise ValueError(f"Warmup/test overlap: {scenario}/{case}")
    return hashes


def launch_command(args, schedule):
    graph = {
        "prefill": {
            "backend": "tc_piecewise",
            "bs": [
                128,
                256,
                1280,
                1536,
                2048,
                2560,
                3072,
                3840,
                4096,
                5120,
                5632,
                5760,
                6400,
                7680,
                8192,
            ],
        },
        "decode": {
            "backend": "full",
            "bs": [1, 2, 4, 8, 10, 11, 12, 14, 15, 16, 24, 28, 32, 48, 64],
        },
    }
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.model_path),
        "--served-model-name",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--device",
        "npu",
        "--attention-backend",
        "ascend",
        "--tp-size",
        "1",
        "--dp-size",
        "2",
        "--context-length",
        "8192",
        "--max-running-requests",
        "64",
        "--max-prefill-tokens",
        "8192",
        "--chunked-prefill-size",
        "8192",
        "--mem-fraction-static",
        "0.92",
        "--cuda-graph-config",
        json.dumps(graph),
        "--trust-remote-code",
    ]
    ratio, delay, cap = schedule if schedule is not None else SCHEDULES["base"]
    cmd += [
        "--prefill-delayer-queue-min-ratio",
        str(ratio),
        "--prefill-delayer-max-delay-ms",
        str(delay),
        "--prefill-delayer-max-delay-passes",
        "4",
        "--prefill-max-requests",
        str(cap),
    ]
    if schedule is not None:
        cmd += ["--enable-prefill-delayer"]
    return cmd


def batch_statistics(text):
    stats = {}
    for rank in (0, 1):
        lines = [line for line in text.splitlines() if f"DP{rank}]" in line]
        stats[str(rank)] = {}
        for phase, pattern in [
            ("prefill", r"Prefill batch.*#new-seq:\s*(\d+)"),
            ("decode", r"Decode batch.*#running-req:\s*(\d+)"),
        ]:
            values = [int(m[1]) for line in lines if (m := re.search(pattern, line))]
            stats[str(rank)][phase] = {
                "samples": len(values),
                "mean": sum(values) / len(values) if values else None,
            }
    return stats


def command_output(cmd, cwd=None):
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": 127, "output": str(error)}
    return {"returncode": result.returncode, "output": result.stdout}


def runtime_identity(args):
    versions = {}
    for package in ("torch", "torch_npu", "sgl_kernel_npu", "triton_ascend", "sglang"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    from candidate_validation import cann_versions, triton_version

    versions["triton"] = triton_version()
    return {
        "cann_version_files": cann_versions(),
        "benchmark_sha256": hashlib.sha256(args.bench_script.read_bytes()).hexdigest(),
        "commit": command_output(["git", "rev-parse", "HEAD"], args.server_root),
        "dirty": command_output(
            ["git", "status", "--porcelain", "-uno"], args.server_root
        ),
        "versions": versions,
        "npu_smi": command_output(["npu-smi", "info"]),
        "topology": command_output(["npu-smi", "info", "-t", "topo"]),
        "checkpoint_metadata": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.model_path.iterdir()
            if p.name in ("config.json", "quant_model_description.json")
        },
    }


def stop_owned_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_server(args, schedule_name, out, concurrencies, repeats, env):
    server_log = out / "server.log"
    command = launch_command(args, SCHEDULES[schedule_name])
    (out / "launch.json").write_text(
        json.dumps(
            {
                "command": command,
                "environment": {
                    k: env.get(k)
                    for k in (
                        "ASCEND_RT_VISIBLE_DEVICES",
                        "TASK_QUEUE_ENABLE",
                        "ASCEND_USE_FIA",
                        "PYTORCH_NPU_ALLOC_CONF",
                        "SGLANG_ENABLE_NPU_GEMMA4_QKV_NORM",
                        "SGLANG_ENABLE_NPU_FIA_METADATA_REUSE",
                        "SGLANG_ENABLE_NPU_FIA_KV_VIEW_REUSE",
                        "SGLANG_ENABLE_NPU_ADAPTIVE_TASK_QUEUE",
                    )
                },
            },
            indent=2,
        )
    )
    url = f"http://127.0.0.1:{args.port}"
    try:
        request(url + "/health", timeout=2)
    except (OSError, ValueError):
        pass
    else:
        raise RuntimeError(f"Port {args.port} already has a server; choose a free port")
    results = []
    profile_thread = None
    profile_errors = []
    with server_log.open("w") as log:
        server = subprocess.Popen(
            command,
            cwd=args.server_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited; inspect {server_log}")
                try:
                    request(url + "/health", timeout=2)
                    break
                except (OSError, ValueError):
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"Server readiness timeout: {server_log}")
                    time.sleep(2)
            info = request(url + "/server_info")
            (out / "server_info.json").write_text(json.dumps(info, indent=2))
            (out / "npu-after-start.json").write_text(
                json.dumps(command_output(["npu-smi", "info"]), indent=2)
            )
            effective = info.get("server_args", {})
            for name, expected in [
                ("tp_size", 1),
                ("dp_size", 2),
                ("context_length", 8192),
                ("max_prefill_tokens", 8192),
                ("chunked_prefill_size", 8192),
            ]:
                if effective.get(name) != expected:
                    raise RuntimeError(
                        f"Unexpected effective {name}: {effective.get(name)!r}"
                    )
            graph = effective.get("cuda_graph_config", {})
            if (
                graph.get("prefill", {}).get("backend") != "tc_piecewise"
                or graph.get("decode", {}).get("backend") != "full"
            ):
                raise RuntimeError("Unexpected graph backend path")
            startup = server_log.read_text(errors="replace")
            if not all(f"DP{rank}]" in startup for rank in (0, 1)):
                raise RuntimeError("Startup log does not identify both DP0 and DP1")
            for concurrency in concurrencies:
                for repeat in range(repeats):
                    for scenario, (rate, _) in TARGETS.items():
                        work = (
                            out / f"c{concurrency}" / f"repeat{repeat + 1}" / scenario
                        )
                        work.mkdir(parents=True)
                        before = server_log.stat().st_size
                        cmd = [
                            sys.executable,
                            str(args.bench_script),
                            "--scenario",
                            scenario,
                            "--url",
                            url,
                            "--model",
                            args.model,
                            "--tokenizer",
                            str(args.model_path),
                            "--datasets",
                            str(args.datasets),
                            "--concurrency",
                            str(concurrency),
                            "--requests",
                            str(args.requests),
                            "--warmups",
                            str(args.warmups),
                            "--rps",
                            str(rate),
                        ]
                        (work / "command.json").write_text(json.dumps(cmd, indent=2))
                        if args.phase == "profile" and profile_thread is None:
                            from collect_profile import collect

                            def capture(concurrency=concurrency, log_offset=before):
                                try:
                                    collect(
                                        url,
                                        server_log,
                                        out / "profile",
                                        min_running=concurrency // 2,
                                        record_shapes=args.record_shapes,
                                        log_offset=log_offset,
                                    )
                                except Exception as error:  # noqa: BLE001 -- propagate worker failures to the owner
                                    profile_errors.append(str(error))

                            profile_thread = threading.Thread(
                                target=capture, daemon=True
                            )
                            profile_thread.start()
                        with (work / "benchmark.log").open("w") as bench_log:
                            completed = subprocess.run(
                                cmd,
                                cwd=work,
                                env=env,
                                stdout=bench_log,
                                stderr=subprocess.STDOUT,
                                timeout=args.benchmark_timeout,
                                check=False,
                            )
                        with server_log.open("rb") as f:
                            f.seek(before)
                            window = f.read().decode(errors="replace")
                        (work / "server-window.log").write_text(window)
                        summary_files = list(work.glob("results_*/summary.json"))
                        if completed.returncode or len(summary_files) != 1:
                            raise RuntimeError(f"Benchmark failed: {work}")
                        summary = json.loads(summary_files[0].read_text())
                        if (
                            summary.get("status") in ("ERROR", "INCOMPLETE")
                            or len(summary.get("cases", [])) != 4
                        ):
                            raise RuntimeError(
                                f"Incomplete benchmark: {summary_files[0]}"
                            )
                        if {case["case"] for case in summary["cases"]} != set(CASES):
                            raise RuntimeError("Benchmark case coverage changed")
                        for case in summary["cases"]:
                            raw = json.loads(
                                (
                                    summary_files[0].parent / case["raw_result"]
                                ).read_text()
                            )
                            high = case["case"] in ("avg_high", "high_high")
                            output_len = (
                                (512 if high else 100)
                                if scenario == "recommendation"
                                else (1200 if high else 250)
                            )
                            result = evaluate_case(
                                raw, scenario, args.requests, output_len=output_len
                            )
                            result.update(
                                scenario=scenario,
                                case=case["case"],
                                concurrency=concurrency,
                                repeat=repeat + 1,
                                schedule=schedule_name,
                                qkv=args.qkv,
                                measurement_kind="profiled"
                                if args.phase == "profile"
                                else "unprofiled",
                                raw_result=str(
                                    summary_files[0].parent / case["raw_result"]
                                ),
                            )
                            results.append(result)
                        record = {
                            "scenario": scenario,
                            "concurrency": concurrency,
                            "repeat": repeat + 1,
                            "sampled_batch_statistics": batch_statistics(window),
                            "includes_warmups": True,
                        }
                        with (out / "batches.jsonl").open("a") as f:
                            f.write(json.dumps(record) + "\n")
                        (out / "results.json").write_text(json.dumps(results, indent=2))
                        if profile_errors:
                            raise RuntimeError(
                                f"Profile collection failed: {profile_errors}"
                            )
                        print(
                            f"{schedule_name} c{concurrency} repeat{repeat + 1} {scenario}: "
                            f"{sum(r['pass'] for r in results[-4:])}/4 cases pass",
                            flush=True,
                        )
            from quality_check import capture, compare

            quality = capture(url, args.model, args.datasets, out / "quality.json")
            if args.quality_baseline is not None:
                check = compare(json.loads(args.quality_baseline.read_text()), quality)
                (out / "quality-comparison.json").write_text(
                    json.dumps(check, indent=2)
                )
                if not check["pass"]:
                    raise RuntimeError(
                        f"Quality regression: {out / 'quality-comparison.json'}"
                    )
            if profile_thread is not None:
                profile_thread.join(timeout=65)
                if profile_thread.is_alive() or profile_errors:
                    raise RuntimeError(
                        f"Profile collection incomplete: {profile_errors}"
                    )
        finally:
            stop_owned_process(server)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        choices=["baseline", "candidate", "acceptance", "scheduling", "profile"],
        default="baseline",
    )
    p.add_argument("--server-root", type=Path, required=True)
    p.add_argument("--bench-script", type=Path, required=True)
    p.add_argument("--datasets", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--model", default="gemma4-w8a8-test")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--concurrency", type=int, choices=[24, 28, 29], default=28)
    p.add_argument("--schedule", choices=list(SCHEDULES), default="base")
    p.add_argument(
        "--qkv",
        choices=["native", "serial"],
        default="serial",
    )
    p.add_argument("--port", type=int, default=30014)
    p.add_argument("--requests", type=int, default=256)
    p.add_argument("--warmups", type=int, default=32)
    p.add_argument("--quality-baseline", type=Path)
    p.add_argument(
        "--record-shapes",
        action="store_true",
        help="Diagnostic profile only; timing profiles omit shapes",
    )
    p.add_argument("--startup-timeout", type=int, default=900)
    p.add_argument("--benchmark-timeout", type=int, default=7200)
    args = p.parse_args()
    for name in ("server_root", "bench_script", "datasets", "model_path", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if (
        args.requests < 1
        or args.warmups < 0
        or (args.startup_timeout <= 0 or args.benchmark_timeout <= 0)
    ):
        p.error("Invalid request count, warmups or timeout")
    if (
        not (args.server_root / "python/sglang/launch_server.py").is_file()
        or not args.bench_script.is_file()
    ):
        p.error("Missing server checkout or bench_updated.py")
    if not args.model_path.is_dir():
        p.error("Missing model directory")
    if args.phase == "acceptance" and (
        args.quality_baseline is None or not args.quality_baseline.is_file()
    ):
        p.error(
            "Acceptance requires --quality-baseline from the unchanged checkpoint run"
        )
    hashes = validate_datasets(args.datasets, args.requests, args.warmups)
    identity = runtime_identity(args)
    if any(
        identity["versions"][name] is None
        for name in ("torch", "torch_npu", "sgl_kernel_npu")
    ):
        p.error("Run this on the NPU host with its existing environment")
    if identity["npu_smi"]["returncode"] or identity["dirty"]["output"].strip():
        p.error("NPU tooling failed or server checkout has tracked modifications")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "environment.json").write_text(json.dumps(identity, indent=2))
    (args.output / "datasets.json").write_text(json.dumps(hashes, indent=2))
    env = os.environ.copy()
    env.update(
        ASCEND_RT_VISIBLE_DEVICES="2,3",
        TASK_QUEUE_ENABLE="1",
        ASCEND_USE_FIA="1",
        PYTORCH_NPU_ALLOC_CONF="expandable_segments:True",
        SGLANG_ENABLE_NPU_ADAPTIVE_TASK_QUEUE="0",
        SGLANG_ENABLE_NPU_GEMMA4_QKV_NORM="0" if args.qkv == "native" else "1",
    )
    env["PYTHONPATH"] = (
        str(args.server_root / "python") + os.pathsep + env.get("PYTHONPATH", "")
    )
    if env.get("ASCEND_LAUNCH_BLOCKING") == "1":
        p.error("Unset ASCEND_LAUNCH_BLOCKING before performance measurements")
    schedules = list(SCHEDULES) if args.phase == "scheduling" else [args.schedule]
    concurrencies = [24, 28, 29] if args.phase == "baseline" else [args.concurrency]
    repeats = 3 if args.phase == "acceptance" else 1
    all_results = []
    for schedule in schedules:
        out = args.output / schedule
        out.mkdir()
        all_results += run_server(args, schedule, out, concurrencies, repeats, env)
    if (
        validate_datasets(args.datasets, args.requests, args.warmups) != hashes
        or hashlib.sha256(args.bench_script.read_bytes()).hexdigest()
        != identity["benchmark_sha256"]
    ):
        raise RuntimeError("Dataset or benchmark script changed during the experiment")
    if (
        command_output(["git", "rev-parse", "HEAD"], args.server_root)
        != identity["commit"]
        or command_output(["git", "status", "--porcelain", "-uno"], args.server_root)[
            "output"
        ].strip()
    ):
        raise RuntimeError("Server checkout changed during the experiment")
    passed = (
        args.phase == "acceptance"
        and len(all_results) == 24
        and all(r["pass"] for r in all_results)
    )
    (args.output / "acceptance.json").write_text(
        json.dumps(
            {
                "acceptance_pass": passed,
                "phase": args.phase,
                "results": all_results,
                "note": "Batch means are sampled server reports; profile and one-run results are not acceptance.",
            },
            indent=2,
        )
    )
    print(f"Results: {args.output}; acceptance_pass={passed}")
    return 0 if args.phase != "acceptance" or passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
