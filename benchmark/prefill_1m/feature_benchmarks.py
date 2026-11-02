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
from feature_checks import cache_page_size, long_working_set, minimum_cached_prefix


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
    (directory / "server_info.json").write_text(json.dumps(server_info, indent=2))
    page = cache_page_size(server_info)
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
            for cache in getattr(args, "long_cache_modes", ("cold", "prefix99")):
                working_set = long_working_set(
                    length, min(concurrency, args.long_requests), cache, server_info
                )
                required = working_set["required_tokens"]
                if required > capacity["max_total_num_tokens"]:
                    results.append(
                        dict(
                            length=length,
                            concurrency=concurrency,
                            cache=cache,
                            status="capacity_blocked",
                            required_tokens=required,
                            working_set=working_set,
                            capacity=capacity,
                        )
                    )
                    continue
                samples = []
                for repeat in range(args.benchmark_repeats):
                    rng = random.Random(4 + repeat)

                    def fresh(count):
                        return [rng.randrange(256, vocab) for _ in range(count)]

                    # The bounded profile warms each cache regime once; flush every repeat.
                    if repeat == 0 or getattr(args, "long_warmup", "each") == "each":
                        run_one(base, fresh(length), 1, args.request_timeout)
                    request(
                        base + "/flush_cache",
                        {},
                        args.request_timeout,
                        expect_json=False,
                    )
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
                    sample = dict(
                        repeat=repeat,
                        wall_s=wall,
                        requests=measured,
                        validation_status="pending",
                        cache_page_size=page,
                    )
                    samples.append(sample)
                    sample_path = (
                        directory / f"long-{length}-{concurrency}-{cache}-samples.json"
                    )
                    # Preserve client timing before a validation error can abort the point.
                    sample_path.write_text(json.dumps(samples, indent=2))
                    try:
                        for item in measured:
                            if item.get("prompt_tokens") != length:
                                raise RuntimeError(
                                    f"Expected {length} prompt tokens, got {item.get('prompt_tokens')}"
                                )
                            cached = item["cached_tokens"]
                            if cached is None:
                                raise RuntimeError(
                                    "cache reuse cannot be verified: cached_tokens missing"
                                )
                            if cache == "cold" and cached > max(64, length * 0.001):
                                raise RuntimeError(f"cold run reused {cached} tokens")
                            if cache == "prefix99" and cached < minimum_cached_prefix(
                                len(prefix), server_info
                            ):
                                raise RuntimeError(
                                    f"prefix run cached only {cached}/{len(prefix)} tokens"
                                )
                        sample["validation_status"] = "verified"
                    except Exception as exc:
                        sample.update(validation_status="fail", error=str(exc))
                        raise
                    finally:
                        sample_path.write_text(json.dumps(samples, indent=2))
                ttft = [r["ttft_s"] for sample in samples for r in sample["requests"]]
                e2e = [r["e2e_s"] for sample in samples for r in sample["requests"]]
                results.append(
                    dict(
                        length=length,
                        concurrency=concurrency,
                        cache=cache,
                        working_set=working_set,
                        capacity=capacity,
                        status="pass",
                        median_ttft_s=statistics.median(ttft),
                        p90_ttft_s=percentile(ttft, 0.90),
                        p95_ttft_s=percentile(ttft, 0.95),
                        mean_e2e_s=statistics.mean(e2e),
                        p90_e2e_s=percentile(e2e, 0.90),
                        mean_run_wall_s=statistics.mean(s["wall_s"] for s in samples),
                        requests_per_s=statistics.mean(
                            len(s["requests"]) / s["wall_s"] for s in samples
                        ),
                        prompt_tokens_per_s=statistics.mean(
                            length * len(s["requests"]) / s["wall_s"] for s in samples
                        ),
                        uncached_prefill_tokens_per_s=statistics.mean(
                            sum(length - r["cached_tokens"] for r in s["requests"])
                            / s["wall_s"]
                            for s in samples
                        ),
                        tpot_s=None,
                        mean_prefill_batch_size=None,
                        mean_decode_batch_size=None,
                        batch_means_status="not exposed by streaming client; inspect server log",
                        mean_ttft_s=statistics.mean(ttft),
                        stdev_ttft_s=statistics.stdev(ttft) if len(ttft) > 1 else 0,
                        samples=samples,
                    )
                )
                (directory / "long-results.json").write_text(
                    json.dumps(results, indent=2)
                )
    (directory / "server_info_after.json").write_text(
        json.dumps(request(base + "/get_server_info"), indent=2)
    )
    (directory / "long-results.json").write_text(json.dumps(results, indent=2))
    return results
