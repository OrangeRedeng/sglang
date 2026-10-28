"""Reproducible serving workloads; all result files live in the report workspace."""

import json
import math
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bench_glm53_prefill_1m import run_one


class CapacityBlocked(RuntimeError):
    pass


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = math.floor(position)
    hi = math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def limits(info):
    found = {"max_total_num_tokens": [], "max_req_input_len": []}

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in found and isinstance(item, int) and item > 0:
                    found[key].append(item)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(info)
    return {key: min(values) if values else None for key, values in found.items()}


def small_commands(args, directory):
    for length in args.small_lengths:
        for concurrency in args.small_concurrency:
            for repeat in range(args.benchmark_repeats):
                output = directory / f"small-{length}-{concurrency}-{repeat}.jsonl"
                yield (
                    [
                        sys.executable,
                        "-m",
                        "sglang.benchmark.serving",
                        "--backend",
                        "sglang",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(args.port),
                        "--dataset-name",
                        "random-ids",
                        "--tokenize-prompt",
                        "--random-input-len",
                        str(length),
                        "--random-output-len",
                        "32",
                        "--random-range-ratio",
                        "1",
                        "--num-prompts",
                        str(args.small_requests),
                        "--max-concurrency",
                        str(concurrency),
                        "--request-rate",
                        "inf",
                        "--warmup-requests",
                        "1",
                        "--flush-cache",
                        "--seed",
                        str(4 + repeat),
                        "--output-file",
                        str(output),
                    ],
                    output,
                )


def long_matrix(args, base, request, checkpoint, directory):
    config = json.loads((Path(checkpoint) / "config.json").read_text())
    config = config.get("text_config", config)
    vocab = int(config["vocab_size"])
    if vocab <= 256:
        raise ValueError("invalid model vocabulary")
    server_info = request(base + "/get_server_info")
    capacity = limits(server_info)
    if capacity["max_total_num_tokens"] is None:
        raise CapacityBlocked("server did not expose KV token capacity")
    results = []
    for length in args.long_lengths:
        if length + 1 > args.context_length or (
            capacity["max_req_input_len"] and length > capacity["max_req_input_len"]
        ):
            results.append(
                dict(length=length, status="capacity_blocked", reason="context limit")
            )
            continue
        for concurrency in args.long_concurrency:
            for cache in ("cold", "prefix99"):
                # Prime + suffixes share physical prefix KV pages in prefix99 mode.
                required = (
                    concurrency * (length + 1)
                    if cache == "cold"
                    else length + concurrency * (length - int(length * 0.99) + 1)
                )
                if required > capacity["max_total_num_tokens"]:
                    results.append(
                        dict(
                            length=length,
                            concurrency=concurrency,
                            cache=cache,
                            status="capacity_blocked",
                            required_tokens=required,
                            capacity=capacity,
                        )
                    )
                    continue
                samples = []
                for repeat in range(args.benchmark_repeats):
                    rng = random.Random(4 + repeat)

                    def fresh(count):
                        return [rng.randrange(256, vocab) for _ in range(count)]

                    # One warmup, followed by a cache reset, for every measured repeat.
                    run_one(base, fresh(length), 1, args.request_timeout)
                    request(base + "/flush_cache", {}, args.request_timeout)
                    prefix = fresh(int(length * 0.99)) if cache == "prefix99" else []
                    if prefix:
                        run_one(base, prefix, 1, args.request_timeout)
                    prompts = [
                        prefix + fresh(length - len(prefix))
                        for _ in range(args.long_requests)
                    ]
                    start = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        measured = list(
                            pool.map(
                                lambda ids: run_one(base, ids, 1, args.request_timeout),
                                prompts,
                            )
                        )
                    wall = time.perf_counter() - start
                    for item in measured:
                        cached = item["cached_tokens"]
                        if cached is None:
                            raise RuntimeError(
                                "cache reuse cannot be verified: cached_tokens missing"
                            )
                        if cache == "cold" and cached > max(64, length * 0.001):
                            raise RuntimeError(f"cold run reused {cached} tokens")
                        if cache == "prefix99" and cached < max(0, len(prefix) - 128):
                            raise RuntimeError(
                                f"prefix run cached only {cached}/{len(prefix)} tokens"
                            )
                    samples.append(dict(repeat=repeat, wall_s=wall, requests=measured))
                ttft = [r["ttft_s"] for sample in samples for r in sample["requests"]]
                results.append(
                    dict(
                        length=length,
                        concurrency=concurrency,
                        cache=cache,
                        status="pass",
                        median_ttft_s=statistics.median(ttft),
                        p95_ttft_s=percentile(ttft, 0.95),
                        mean_ttft_s=statistics.mean(ttft),
                        stdev_ttft_s=statistics.stdev(ttft) if len(ttft) > 1 else 0,
                        samples=samples,
                    )
                )
    (directory / "long-results.json").write_text(json.dumps(results, indent=2))
    return results
