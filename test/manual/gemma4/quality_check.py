"""Capture and compare deterministic completions and token logprobs on fixed prompts."""

import argparse
import hashlib
import json
import math
import urllib.request
from pathlib import Path

CASES = ("avg_avg", "high_high", "avg_high", "high_avg")
IDENTITY = (
    "model_path",
    "dtype",
    "quantization",
    "kv_cache_dtype",
    "device",
    "attention_backend",
    "context_length",
    "tp_size",
    "dp_size",
)


def rpc(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.load(response)


def capture(url, model, datasets, output, per_case=4):
    info = rpc(url + "/server_info")
    rows = []
    for scenario in ("recommendation", "rag"):
        for case in CASES:
            path = datasets / scenario / f"{case}.jsonl"
            prompts = [
                json.loads(line)["prompt"] for line in path.read_text().splitlines()
            ]
            if len(prompts) < per_case:
                raise ValueError(f"Not enough quality prompts: {path}")
            for prompt in prompts[:per_case]:
                response = rpc(
                    url + "/v1/completions",
                    {
                        "model": model,
                        "prompt": prompt,
                        "temperature": 0,
                        "seed": 48,
                        "max_tokens": 64,
                        "ignore_eos": True,
                        "logprobs": 1,
                    },
                )
                choice = response["choices"][0]
                logprobs = choice.get("logprobs") or {}
                tokens = logprobs.get("tokens")
                values = logprobs.get("token_logprobs")
                if (
                    not tokens
                    or not values
                    or len(tokens) != len(values)
                    or any(
                        isinstance(v, bool)
                        or not isinstance(v, (int, float))
                        or not math.isfinite(v)
                        for v in values
                    )
                ):
                    raise ValueError("Missing or invalid generated-token logprobs")
                rows.append(
                    {
                        "scenario": scenario,
                        "case": case,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "text": choice["text"],
                        "tokens": tokens,
                        "logprobs": values,
                    }
                )
    result = {
        "server_args": {
            name: info.get("server_args", {}).get(name) for name in IDENTITY
        },
        "model": model,
        "per_case": per_case,
        "cases": rows,
    }
    output.write_text(json.dumps(result, indent=2))
    return result


def compare(baseline, candidate):
    # Same BF16 tolerance as the existing QKV correctness validator; never relax automatically.
    errors = []
    if baseline.get("server_args") != candidate.get("server_args") or baseline.get(
        "model"
    ) != candidate.get("model"):
        errors.append("Model/runtime identity changed")
    old, new = baseline.get("cases", []), candidate.get("cases", [])
    if not old or len(old) != len(new):
        errors.append("Quality prompt coverage differs or is empty")
    if old and baseline.get("per_case"):
        coverage = {
            (scenario, case) for scenario in ("recommendation", "rag") for case in CASES
        }
        if {(r.get("scenario"), r.get("case")) for r in old} != coverage or len(
            old
        ) != 8 * baseline["per_case"]:
            errors.append("Baseline does not cover all eight cases")
    for index, (a, b) in enumerate(zip(old, new)):
        if any(
            a.get(key) != b.get(key)
            for key in ("scenario", "case", "prompt_sha256", "text", "tokens")
        ):
            errors.append(f"Greedy output or prompt changed: {index}")
        av, bv = a.get("logprobs", []), b.get("logprobs", [])
        if not av or len(av) != len(bv) or len(av) != len(a.get("tokens", [])):
            errors.append(f"Missing logprob coverage: {index}")
        elif not all(
            isinstance(x, (int, float))
            and not isinstance(x, bool)
            and math.isfinite(x)
            and isinstance(y, (int, float))
            and not isinstance(y, bool)
            and math.isfinite(y)
            and math.isclose(x, y, rel_tol=0.02, abs_tol=0.02)
            for x, y in zip(av, bv)
        ):
            errors.append(f"Logprobs changed beyond tolerance: {index}")
    return {"pass": not errors, "atol": 0.02, "rtol": 0.02, "errors": errors}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--url", default="http://127.0.0.1:30014")
    cap.add_argument("--model", default="gemma4-w8a8-test")
    cap.add_argument("--datasets", type=Path, required=True)
    cap.add_argument("--output", type=Path, required=True)
    cmp = sub.add_parser("compare")
    cmp.add_argument("--baseline", type=Path, required=True)
    cmp.add_argument("--candidate", type=Path, required=True)
    args = p.parse_args()
    if args.mode == "capture":
        if args.output.exists():
            p.error("Use a new output file")
        capture(args.url.rstrip("/"), args.model, args.datasets, args.output)
        return 0
    result = compare(
        json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text())
    )
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
