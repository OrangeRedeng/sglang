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
) -> dict[str, float]:
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
    with requests.post(
        f"{base_url}/generate", json=payload, stream=True, timeout=timeout
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            chunk = line[len("data:") :].strip()
            if chunk == "[DONE]":
                break
            if first_token_ts is None:
                first_token_ts = time.perf_counter()
    end = time.perf_counter()
    if first_token_ts is None:
        raise RuntimeError("no streamed chunk received")
    return {
        "ttft_s": first_token_ts - start,
        "e2e_s": end - start,
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
) -> dict[str, Any]:
    """One batch-size point: warmup runs, then repeats, then aggregate."""
    # One prompt per request in the batch; distinct seeds avoid identical
    # payloads so the scheduler does not fold them into one prefix.
    prompts = [build_prompt_ids(context_length, seed + i) for i in range(batch_size)]
    for _ in range(warmup):
        for ids in prompts[:1]:
            run_one(base_url, ids, max_new_tokens, timeout)
    samples = []
    for r in range(repeats):
        t0 = time.perf_counter()
        for ids in prompts:
            run_one(base_url, ids, max_new_tokens, timeout)
        wall = time.perf_counter() - t0
        samples.append(wall)
        print(
            f"  bs={batch_size} repeat={r + 1}/{repeats} "
            f"wall={wall:.1f}s ttft_per_req≈{wall / batch_size:.1f}s"
        )
    best = min(samples)
    return {
        "batch_size": batch_size,
        "context_length": context_length,
        "repeats": repeats,
        "wall_s_best": best,
        "wall_s_all": samples,
        "ttft_s_best_avg_per_req": best / batch_size,
        "prefill_tokens_per_s": context_length * batch_size / best,
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
