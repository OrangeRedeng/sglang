"""Controlled C24 Gemma4 experiments across separate serving/benchmark containers."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from quality_check import IDENTITY, compare, rpc
from throughput_matrix import runtime_identity, stop_owned_process

MODEL = "/home/test1/models/gemma4-26B-A4B-w8a8"
NAME = "gemma4-w8a8-test"
LATENCIES = [f"{p}_{m}_ms" for m in ("ttft", "tpot", "e2el") for p in ("mean", "p99")]
WORKLOAD = {
    "seed": 48,
    "requests": 1600,
    "concurrency": 24,
    "rate": 87,
    "warmups": 32,
    "prefix": 250,
    "input": 1100,
    "output": 100,
    "temperature": 0,
    "ignore_eos": True,
}
ENV_PREFIXES = (
    "SGLANG_",
    "ASCEND_",
    "HCCL_",
    "CANN_",
    "PYTORCH_",
    "TORCH_",
    "OMP_",
    "MKL_",
)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log_stats(path):
    ranks = defaultdict(lambda: defaultdict(list))
    with path.open(errors="replace") as stream:
        for line in stream:
            rank = re.search(r"\bDP(\d+)\b", line)
            rank = rank[1] if rank else "unknown"
            phase = "prefill" if "Prefill batch" in line else "decode"
            if "Prefill batch" not in line and "Decode batch" not in line:
                continue
            for key, pattern in (
                ("prefill_sequences", r"#new-seq:\s*(\d+)"),
                ("prefill_tokens", r"#new-token:\s*(\d+)"),
                ("decode_sequences", r"#running-req:\s*(\d+)"),
            ):
                if key.startswith(phase) and (match := re.search(pattern, line)):
                    ranks[rank][key].append(int(match[1]))
    return {
        rank: {
            key: {
                "samples": len(values),
                "mean": statistics.mean(values),
                "histogram": dict(sorted(Counter(values).items())),
            }
            for key, values in counters.items()
        }
        for rank, counters in ranks.items()
    }


def serve(args):
    root = Path(__file__).resolve().parents[3]
    settings = {
        key: getattr(args, key)
        for key in (
            "devices",
            "memory",
            "delayer",
            "delay",
            "ratio",
            "cap",
            "qkv",
            "metadata",
            "kv_views",
            "overlap",
            "mtp",
            "add_decode_buckets",
            "model_path",
        )
    }
    buckets = sorted(
        set([1, 2, 4, 8, 12, 15, 16, 24, 28, 32, 48, 64] + args.add_decode_buckets)
    )
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
                4096,
                5632,
                5760,
                6400,
                7680,
                8192,
            ],
        },
        "decode": {"backend": "full", "bs": buckets},
    }
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--served-model-name",
        NAME,
        "--host",
        "0.0.0.0",
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
        str(args.memory),
        "--random-seed",
        "48",
        "--cuda-graph-config",
        json.dumps(graph),
        "--prefill-max-requests",
        str(args.cap),
        "--prefill-delayer-queue-min-ratio",
        str(args.ratio),
        "--prefill-delayer-max-delay-ms",
        str(args.delay),
        "--prefill-delayer-max-delay-passes",
        "4",
        "--trust-remote-code",
    ]
    if args.delayer:
        command.append("--enable-prefill-delayer")
    if not args.overlap:
        command.append("--disable-overlap-schedule")
    if args.mtp:
        command += [
            "--speculative-algorithm",
            "NEXTN",
            "--speculative-draft-model-path",
            args.assistant,
            "--speculative-num-steps",
            "1",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "2",
        ]
    # Remove inherited SGLang experiments; retain and record CANN/network setup.
    env = {k: v for k, v in os.environ.items() if not k.startswith("SGLANG_")}
    env.update(
        {
            "ASCEND_RT_VISIBLE_DEVICES": args.devices,
            "TASK_QUEUE_ENABLE": "1",
            "ASCEND_USE_FIA": "1",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
            "SGLANG_ENABLE_NPU_GEMMA4_QKV_NORM": str(args.qkv),
            "SGLANG_ENABLE_NPU_FIA_METADATA_REUSE": str(args.metadata),
            "SGLANG_ENABLE_NPU_FIA_KV_VIEW_REUSE": str(args.kv_views),
            "SGLANG_DISABLE_NPU_MTP_DRAFT_GRAPH": "1",
            "SGLANG_DISABLE_NPU_MTP_VERIFY_GRAPH": "0",
            "SGLANG_DEBUG_NPU_MTP_GRAPH_SYNC": "0",
            "PYTHONPATH": str(root / "python"),
        }
    )
    recorded_env = {
        k: v
        for k, v in env.items()
        if k.startswith(ENV_PREFIXES)
        or k in ("TASK_QUEUE_ENABLE", "PYTHONPATH", "PATH", "LD_LIBRARY_PATH")
    }
    manifest = {"settings": settings, "command": command, "environment": recorded_env}
    print(shlex.join(command), flush=True)
    if args.print_only:
        print(json.dumps(manifest, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    manifest["runtime"] = runtime_identity(
        SimpleNamespace(
            server_root=root,
            bench_script=Path(__file__),
            model_path=Path(args.model_path),
        )
    )
    hardware = subprocess.check_output(
        [
            sys.executable,
            "-c",
            (
                "import json, torch, torch_npu; "
                "print(json.dumps([torch.npu.get_device_name(i) for i in range(torch.npu.device_count())]))"
            ),
        ],
        env=env,
        text=True,
        timeout=60,
    )
    manifest["runtime"]["device_names"] = json.loads(hardware)
    if len(manifest["runtime"]["device_names"]) != 2:
        raise ValueError("Expected exactly two visible NPUs")
    manifest["started_at"] = time.time()
    write(args.output / "launch.json", manifest)
    print(f"Server log: {args.output / 'server.log'}", flush=True)
    with (args.output / "server.log").open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            status = process.wait()
        finally:
            stop_owned_process(process)
    if status:
        raise RuntimeError(f"Server exited with status {status}")


def quality(url):
    prompts = [
        "Explain why rivers flow downhill.",
        "The river flows past the village. " * 160 + "Summarize this sentence.",
        "The river flows past the village. " * 650 + "Summarize this sentence.",
    ]
    rows = []

    def one(item):
        case, prompt = item
        response = rpc(
            url + "/v1/completions",
            {
                "model": NAME,
                "prompt": prompt,
                "temperature": 0,
                "seed": 48,
                "max_tokens": 64,
                "ignore_eos": True,
                "logprobs": 1,
            },
        )
        choice = response["choices"][0]
        logs = choice.get("logprobs") or {}
        tokens, values = logs.get("tokens"), logs.get("token_logprobs")
        if (
            response.get("usage", {}).get("completion_tokens") != 64
            or not tokens
            or not values
            or len(tokens) != len(values)
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
        ):
            raise ValueError("Invalid quality response or output length")
        return {
            "scenario": "random",
            "case": case,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "text": choice["text"],
            "tokens": tokens,
            "logprobs": values,
        }

    # Exercise mixed lengths and changing live batches outside measured timing.
    for size in (1, 7, 12):
        with ThreadPoolExecutor(max_workers=size) as pool:
            rows.extend(
                pool.map(one, [(f"{size}-{i}", prompts[i % 3]) for i in range(size)])
            )
    info = rpc(url + "/server_info")
    return {
        "model": NAME,
        "server_args": {k: info["server_args"].get(k) for k in IDENTITY},
        "cases": rows,
    }


def vllm_command(args, count, filename):
    return [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--base-url",
        args.url,
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        NAME,
        "--tokenizer",
        args.tokenizer,
        "--dataset-name",
        "random",
        "--random-prefix-len",
        "250",
        "--random-input-len",
        "1100",
        "--random-output-len",
        "100",
        "--random-range-ratio",
        "0",
        "--max-concurrency",
        "24",
        "--request-rate",
        "87",
        "--num-prompts",
        str(count),
        "--temperature",
        "0",
        "--ignore-eos",
        "--seed",
        "48",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,99",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(args.output),
        "--result-filename",
        filename,
    ]


def benchmark(args):
    args.url = args.url.rstrip("/")
    command = vllm_command(args, 1600, "result.json")
    print(shlex.join(command), flush=True)
    if args.print_only:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    launch = load(args.launch)
    info = rpc(args.url + "/server_info")
    write(args.output / "server-before.json", info)
    actual = info["server_args"]
    # Resolved flags must agree with the supplied launch manifest.
    expected = {
        "tp_size": 1,
        "dp_size": 2,
        "device": "npu",
        "attention_backend": "ascend",
        "model_path": launch["settings"]["model_path"],
        "served_model_name": NAME,
        "mem_fraction_static": launch["settings"]["memory"],
        "disable_overlap_schedule": not bool(launch["settings"]["overlap"]),
        "enable_prefill_delayer": bool(launch["settings"]["delayer"]),
        "prefill_max_requests": launch["settings"]["cap"],
        "prefill_delayer_queue_min_ratio": launch["settings"]["ratio"],
        "prefill_delayer_max_delay_ms": launch["settings"]["delay"],
    }
    mismatches = {
        k: (v, actual.get(k)) for k, v in expected.items() if actual.get(k) != v
    }
    is_mtp = actual.get("speculative_algorithm") not in (None, "", "NONE")
    if mismatches or is_mtp != bool(launch["settings"]["mtp"]):
        raise ValueError(
            f"Server differs from launch manifest: {mismatches}, MTP={is_mtp}"
        )
    write(args.output / "launch.json", launch)
    for key, value in {
        "context_length": 8192,
        "max_running_requests": 64,
        "max_prefill_tokens": 8192,
        "chunked_prefill_size": 8192,
        "random_seed": 48,
        "prefill_delayer_max_delay_passes": 4,
    }.items():
        if actual.get(key) != value:
            raise ValueError(f"Unexpected resolved {key}: {actual.get(key)}")
    if is_mtp and any(
        actual.get(k) != v
        for k, v in {
            "speculative_num_steps": 1,
            "speculative_eagle_topk": 1,
            "speculative_num_draft_tokens": 2,
        }.items()
    ):
        raise ValueError("Expected one draft step and two verification tokens")
    write(args.output / "quality.json", quality(args.url))
    if args.quality_baseline:
        check = compare(load(args.quality_baseline), load(args.output / "quality.json"))
        write(args.output / "quality-comparison.json", check)
        if not check["pass"]:
            raise ValueError(f"Quality regression: {check['errors']}")
    help_text = subprocess.check_output(["vllm", "bench", "serve", "--help"], text=True)
    (args.output / "vllm-help.txt").write_text(help_text)
    for flag in ("--seed", "--save-detailed"):
        if flag not in help_text:
            raise ValueError(f"Installed vLLM lacks required flag {flag}")
    implementation = {}
    dist = importlib.metadata.distribution("vllm")
    for relative in dist.files or []:
        if str(relative).startswith("vllm/benchmarks/") and str(relative).endswith(
            ".py"
        ):
            source = Path(dist.locate_file(relative))
            text = source.read_text()
            if any(s in text for s in ("peak_concurrent", "peak_output")):
                target = args.output / "client-source" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text)
                implementation[str(relative)] = digest(source)
    write(
        args.output / "client.json",
        {
            "version": dist.version,
            "source": implementation,
            "command": command,
            "workload": WORKLOAD,
            "tokenizer": args.tokenizer,
            "tokenizer_metadata": {
                name: digest(Path(args.tokenizer) / name)
                for name in (
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "chat_template.jinja",
                )
                if (Path(args.tokenizer) / name).is_file()
            },
        },
    )
    for count, filename in ((32, "warmup.json"), (1600, "result.json")):
        # Flush after warmup too, so compilation is warm and prefix caches are cold.
        with urllib.request.urlopen(args.url + "/flush_cache", timeout=60) as response:
            response.read()
        offset = args.server_log.stat().st_size
        started = time.time()
        try:
            with (args.output / (filename + ".log")).open("w") as log:
                subprocess.run(
                    vllm_command(args, count, filename),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                    check=True,
                )
            raw = load(args.output / filename)
            if (
                raw.get("completed") != count
                or raw.get("total_output_tokens") != count * 100
            ):
                raise ValueError(f"{filename}: incomplete warmup/measurement")
        finally:
            if count == 1600:
                with (
                    args.server_log.open("rb") as source,
                    (args.output / "server-window.log").open("wb") as dest,
                ):
                    source.seek(offset)
                    while chunk := source.read(1024 * 1024):
                        dest.write(chunk)
                write(
                    args.output / "window.json",
                    {"start": started, "end": time.time(), "offset": offset},
                )
    write(args.output / "server-after.json", rpc(args.url + "/server_info"))
    write(args.output / "batches.json", log_stats(args.output / "server-window.log"))
    raw = load(args.output / "result.json")
    write(
        args.output / "metric-audit.json",
        {
            "completed_rps": raw["completed"] / raw["duration"],
            "aggregate_output_tps": raw["total_output_tokens"] / raw["duration"],
            "reported_peaks": {k: v for k, v in raw.items() if "peak" in k},
            "note": "Inspect client-source and detailed streaming intervals before interpreting peak concurrency or token rates; chunks may contain multiple tokens.",
        },
    )
    print(
        f"Saved {args.output}; decode batch observations are sampled, not every forward."
    )


def summarize(paths):
    records, identity = [], None
    for path in paths:
        raw, client, launch = (
            load(path / "result.json"),
            load(path / "client.json"),
            load(path / "launch.json"),
        )
        before, after = (
            load(path / "server-before.json"),
            load(path / "server-after.json"),
        )
        if before["server_args"] != after["server_args"]:
            raise ValueError(f"{path}: server arguments changed during run")
        runtime = {
            k: launch["runtime"][k]
            for k in (
                "versions",
                "cann_version_files",
                "commit",
                "dirty",
                "checkpoint_metadata",
                "benchmark_sha256",
                "device_names",
            )
        }
        fixed = {
            "workload": client["workload"],
            "tokenizer": client["tokenizer"],
            "tokenizer_metadata": client["tokenizer_metadata"],
            "total_input_tokens": raw["total_input_tokens"],
            "client": client["version"],
            "client_source": client["source"],
            "runtime": runtime,
            "environment": launch["environment"],
            "settings": launch["settings"],
            "server_args": before["server_args"],
        }
        if identity is None:
            identity = fixed
        elif fixed != identity:
            raise ValueError(
                f"{path}: repetitions have different configuration or environment"
            )
        duration, completed = raw.get("duration"), raw.get("completed")
        if (
            completed != 1600
            or raw.get("failed", 0) != 0
            or raw.get("total_output_tokens") != 160000
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise ValueError(f"{path}: incomplete, failed or invalid workload")
        lengths = raw.get("output_lens")
        if not lengths or len(lengths) != 1600 or any(n != 100 for n in lengths):
            raise ValueError(f"{path}: missing per-request 100-token output evidence")
        for name in LATENCIES:
            if (
                not isinstance(raw.get(name), (int, float))
                or not math.isfinite(raw[name])
                or raw[name] < 0
            ):
                raise ValueError(f"{path}: missing or invalid {name}")
        records.append(
            dict(
                rps=completed / duration,
                duration=duration,
                output_tps=raw["total_output_tokens"] / duration,
                **{k: raw[k] for k in LATENCIES},
            )
        )
    return {
        "identity": identity,
        "runs": len(records),
        "median": {k: statistics.median(r[k] for r in records) for k in records[0]},
        "rps_range": [min(r["rps"] for r in records), max(r["rps"] for r in records)],
    }


def report(args):
    paths = [p.resolve() for p in args.baseline + args.candidate]
    if len(paths) != len(set(paths)):
        raise ValueError(
            "Use distinct benchmark directories for independent repetitions"
        )
    baseline, candidate = summarize(args.baseline), summarize(args.candidate)
    old, new = baseline["identity"], candidate["identity"]
    # Model, runtime and workload must match; candidate settings are reported explicitly.
    differences = {}
    for key in (
        "workload",
        "tokenizer",
        "tokenizer_metadata",
        "total_input_tokens",
        "client",
        "client_source",
        "runtime",
    ):
        if old[key] != new[key]:
            raise ValueError(f"Comparison confounded by {key}")
    for key in ("settings", "environment", "server_args"):
        differences[key] = {
            k: [old[key].get(k), new[key].get(k)]
            for k in old[key].keys() | new[key].keys()
            if old[key].get(k) != new[key].get(k)
        }
    changed_settings = set(differences["settings"])
    if changed_settings != {args.change}:
        raise ValueError(
            f"Expected only {args.change} to change; got {sorted(changed_settings)}"
        )
    allowed_env, allowed_args = {
        "delayer": ([], ["enable_prefill_delayer"]),
        "delay": ([], ["prefill_delayer_max_delay_ms"]),
        "ratio": ([], ["prefill_delayer_queue_min_ratio"]),
        "cap": ([], ["prefill_max_requests"]),
        "metadata": (["SGLANG_ENABLE_NPU_FIA_METADATA_REUSE"], []),
        "kv_views": (["SGLANG_ENABLE_NPU_FIA_KV_VIEW_REUSE"], []),
        "add_decode_buckets": (
            [],
            ["cuda_graph_config", "cuda_graph_bs", "cuda_graph_max_bs"],
        ),
        "overlap": ([], ["disable_overlap_schedule"]),
        "mtp": (
            [],
            [
                "speculative_algorithm",
                "speculative_draft_model_path",
                "speculative_num_steps",
                "speculative_eagle_topk",
                "speculative_num_draft_tokens",
            ],
        ),
    }[args.change]
    if set(differences["environment"]) - set(allowed_env):
        raise ValueError(
            f"Unexpected environment changes: {differences['environment']}"
        )
    if set(differences["server_args"]) - set(allowed_args):
        raise ValueError(
            f"Unexpected resolved argument changes: {differences['server_args']}"
        )
    checks = [
        compare(load(a / "quality.json"), load(b / "quality.json"))
        for a in args.baseline
        for b in args.candidate
    ]
    a, b = baseline["median"], candidate["median"]
    gain = b["rps"] / a["rps"] - 1
    tails = {k: b[k] <= a[k] * 1.05 for k in LATENCIES}
    confirmed = baseline["runs"] >= 3 and candidate["runs"] >= 3
    overlap = max(candidate["rps_range"][0], baseline["rps_range"][0]) <= min(
        candidate["rps_range"][1], baseline["rps_range"][1]
    )
    windows = sorted(
        (load(p / "window.json")["start"], role)
        for role, paths in (("baseline", args.baseline), ("candidate", args.candidate))
        for p in paths
    )
    interleaved = all(
        windows[i][1] != windows[i - 1][1] for i in range(1, len(windows))
    )
    batches = {
        str(path): load(path / "batches.json")
        for path in args.baseline + args.candidate
    }
    batch_evidence = all(
        all(
            rank in stats
            and stats[rank].get("prefill_sequences", {}).get("samples", 0)
            and stats[rank].get("decode_sequences", {}).get("samples", 0)
            for rank in ("0", "1")
        )
        for stats in batches.values()
    )
    accepted = (
        confirmed
        and not overlap
        and batch_evidence
        and interleaved
        and gain >= 0.03
        and all(tails.values())
        and all(c["pass"] for c in checks)
    )
    write(
        args.output,
        {
            "baseline": baseline,
            "candidate": candidate,
            "differences": differences,
            "throughput_gain": gain,
            "latency_checks": tails,
            "quality": checks,
            "batches": batches,
            "batch_evidence_complete": batch_evidence,
            "interleaved": interleaved,
            "accepted": accepted,
            "exceeds_9_33": b["rps"] > 9.33,
            "decision": "keep"
            if accepted
            else "inconclusive"
            if not confirmed or overlap or not batch_evidence or not interleaved
            else "reject",
        },
    )
    print(
        f"Median RPS {a['rps']:.3f} -> {b['rps']:.3f} ({gain:+.1%}); accepted={accepted}"
    )
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    server = sub.add_parser("serve")
    server.add_argument("--output", type=Path, required=True)
    server.add_argument("--devices", default="2,3")
    server.add_argument("--model-path", default=MODEL)
    server.add_argument(
        "--assistant", default="/home/test1/models/gemma-4-26B-A4B-it-assistant"
    )
    server.add_argument("--port", type=int, default=30014)
    server.add_argument("--memory", type=float, default=0.92)
    server.add_argument("--delay", type=int, default=50)
    server.add_argument("--ratio", type=float, default=0.5)
    server.add_argument("--cap", type=int, default=6)
    for name, default in (
        ("delayer", 1),
        ("qkv", 1),
        ("metadata", 0),
        ("kv-views", 0),
        ("overlap", 1),
        ("mtp", 0),
    ):
        server.add_argument("--" + name, type=int, choices=(0, 1), default=default)
    server.add_argument("--add-decode-buckets", type=int, nargs="*", default=[])
    server.add_argument("--print", dest="print_only", action="store_true")
    bench = sub.add_parser("bench")
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--launch", type=Path, required=True)
    bench.add_argument("--server-log", type=Path, required=True)
    bench.add_argument("--quality-baseline", type=Path)
    bench.add_argument("--url", default="http://127.0.0.1:30014")
    bench.add_argument("--tokenizer", default=MODEL)
    bench.add_argument("--print", dest="print_only", action="store_true")
    comparison = sub.add_parser("compare")
    comparison.add_argument("--baseline", type=Path, nargs="+", required=True)
    comparison.add_argument("--candidate", type=Path, nargs="+", required=True)
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument(
        "--change",
        choices=(
            "delayer",
            "delay",
            "ratio",
            "cap",
            "metadata",
            "kv_views",
            "add_decode_buckets",
            "overlap",
            "mtp",
        ),
        required=True,
    )
    args = parser.parse_args()
    if args.mode == "serve":
        devices = args.devices.split(",")
        if (
            len(devices) != 2
            or len(set(devices)) != 2
            or not all(d.isdigit() for d in devices)
        ):
            parser.error("Use two distinct NPU indices")
        if not 0 < args.memory < 1 or args.delay < 0 or args.ratio < 0 or args.cap < 1:
            parser.error("Invalid memory or prefill admission settings")
        if any(not 1 <= n <= 64 for n in args.add_decode_buckets):
            parser.error("Decode buckets must be in [1,64]")
        if args.mtp and args.memory != 0.82:
            parser.error("Use --memory 0.82 for MTP and its matched non-MTP control")
        serve(args)
    elif args.mode == "bench":
        benchmark(args)
    else:
        if args.output.exists():
            parser.error("Use a new comparison output file")
        return 0 if report(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
