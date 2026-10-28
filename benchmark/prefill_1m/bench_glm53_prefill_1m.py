"""GLM-5.3 1M-token prefill benchmark for Ascend A3 910C.

Measures prefill-dominated latency at very long context: each request sends a
~1M-token prompt with output_len=1 so the run time is prefill time. Metrics are
TTFT (time to first token) and prefill tokens/s, reported per batch size with
warmup and repeats.

Modeled on the bde_v2 print-build `bench_ais.sh` workflow (prefix-cache style
input, request_rate=0, concurrency sweep) but self-contained: prompts are
synthetic seeded token-id sequences, so no external aisbench checkout or
dataset is needed.

Usage:
    # Against a running server:
    python3 benchmark/prefill_1m/bench_glm53_prefill_1m.py \
        --host 127.0.0.1 --port 30000 --context-length 1048576 \
        --batch-sizes 1,2,4,8 --repeats 3

    # Without a server: build the requests and print the server command:
    python3 benchmark/prefill_1m/bench_glm53_prefill_1m.py --dry-run

Launch the matching server with benchmark/prefill_1m/serve_glm53_1m.sh.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

# GLM-5.3 vocab is ~151k; sample ids well inside it so no token is special.
VOCAB_ID_LOW = 20000
VOCAB_ID_HIGH = 120000


def build_prompt_ids(context_length: int, seed: int) -> list[int]:
    """Deterministic pseudo-random token ids of exactly context_length tokens."""
    rng = random.Random(seed)
    return [rng.randint(VOCAB_ID_LOW, VOCAB_ID_HIGH) for _ in range(context_length)]


def run_one(
    base_url: str,
    prompt_ids: list[int],
    max_new_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    """Send one request, stream, and time to the first output token."""
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
        },
        "stream": True,
    }
    start = time.perf_counter()
    first_token_ts: float | None = None
    cached_tokens = None
    with requests.post(
        f"{base_url}/generate", json=payload, stream=True, timeout=timeout
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(chunk_size=1, decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            chunk = line[len("data:") :].strip()
            if chunk == "[DONE]":
                break
            data = json.loads(chunk)
            if "error" in data:
                raise RuntimeError(data["error"])
            meta = data.get("meta_info", {})
            cached_tokens = meta.get("cached_tokens", cached_tokens)
            if first_token_ts is None and meta.get("completion_tokens", 0) > 0:
                first_token_ts = time.perf_counter()
    end = time.perf_counter()
    if first_token_ts is None:
        raise RuntimeError("no streamed chunk received")
    return {
        "ttft_s": first_token_ts - start,
        "e2e_s": end - start,
        "cached_tokens": cached_tokens,
    }


def bench_batch(
    base_url: str,
    context_length: int,
    batch_size: int,
    repeats: int,
    warmup: int,
    seed: int,
    max_new_tokens: int,
    timeout: float,
    cache_mode: str = "cold",
) -> dict[str, Any]:
    """One batch-size point: warmup runs, then repeats, then aggregate."""
    rng = random.Random(seed)

    def new_prompts():
        return [
            build_prompt_ids(context_length, rng.getrandbits(128))
            for _ in range(batch_size)
        ]

    prompts = new_prompts() if cache_mode == "cached" else None
    samples = []
    warmup_runs = max(warmup, 1) if cache_mode == "cached" else warmup
    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        for r in range(-warmup_runs, repeats):
            batch_prompts = prompts if prompts is not None else new_prompts()
            t0 = time.perf_counter()
            futures = [
                pool.submit(run_one, base_url, ids, max_new_tokens, timeout)
                for ids in batch_prompts
            ]
            requests_result = [future.result() for future in futures]
            wall = time.perf_counter() - t0
            if r < 0:
                continue
            ttft = sum(item["ttft_s"] for item in requests_result) / batch_size
            samples.append(
                {"wall_s": wall, "ttft_s_mean": ttft, "requests": requests_result}
            )
            print(
                f"  bs={batch_size} repeat={r + 1}/{repeats} "
                f"wall={wall:.1f}s ttft_mean={ttft:.1f}s"
            )
    best = min(samples, key=lambda sample: sample["wall_s"])
    cached = [item["cached_tokens"] for item in best["requests"]]
    uncached_tokens = (
        sum(context_length - count for count in cached)
        if all(count is not None for count in cached)
        else None
    )
    return {
        "batch_size": batch_size,
        "context_length": context_length,
        "repeats": repeats,
        "cache_mode": cache_mode,
        "wall_s_best": best["wall_s"],
        "wall_s_all": [sample["wall_s"] for sample in samples],
        "ttft_s_best_avg_per_req": best["ttft_s_mean"],
        "prompt_tokens_per_s": context_length * batch_size / best["wall_s"],
        "prefill_tokens_per_s": (
            uncached_tokens / best["wall_s"] if uncached_tokens is not None else None
        ),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument(
        "--context-length",
        type=int,
        default=1_048_576,
        help="Prompt length in tokens (default 1M).",
    )
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--cache-mode",
        choices=("cold", "cached"),
        default="cold",
        help="Fresh prefixes per batch, or reuse prompts after priming the cache.",
    )
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--output", default="", help="Write JSON results here.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the requests and print the server command; no server needed.",
    )
    args = parser.parse_args()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    if not batch_sizes or min(batch_sizes) < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("batch sizes/repeats must be positive and warmup nonnegative")
    if args.context_length < 1 or args.max_new_tokens < 1:
        parser.error("context length and max new tokens must be positive")
    print(
        f"plan: context_length={args.context_length} "
        f"batch_sizes={batch_sizes} repeats={args.repeats} warmup={args.warmup}"
    )
    print(
        "server: bash benchmark/prefill_1m/serve_glm53_1m.sh "
        "(env: MODEL_PATH, TP, DCP, CAPTURE, ONLINE_QUANT)"
    )
    if args.dry_run:
        sample = build_prompt_ids(args.context_length, args.seed)
        print(
            f"dry-run: first prompt built with {len(sample)} tokens "
            f"(ids {sample[:4]}...{sample[-4:]})"
        )
        return

    base_url = f"http://{args.host}:{args.port}"
    results = []
    for bs in batch_sizes:
        print(f"batch_size={bs}")
        results.append(
            bench_batch(
                base_url,
                args.context_length,
                bs,
                args.repeats,
                args.warmup,
                args.seed + bs * 10_000,
                args.max_new_tokens,
                args.timeout,
                args.cache_mode,
            )
        )
    report = {"config": vars(args), "results": results}
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w") as f:
            f.write(rendered)
        print(f"results written to {args.output}")


if __name__ == "__main__":
    main()
