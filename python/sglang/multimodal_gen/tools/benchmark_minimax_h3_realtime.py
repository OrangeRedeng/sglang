# SPDX-License-Identifier: Apache-2.0
"""Run warm-server H3 ablations from explicit server/client command arrays."""

import argparse
import json
import os
import re
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request


def variants(baseline=None):
    baseline = baseline or {
        "sparse_type": "rf_v3",
        "precision": "mix",
        "sparsity": 0.8,
        "skip_first_steps": 1,
        "skip_last_steps": 0,
        "dense_kwargs": {"precision": "mxfp4"},
    }
    yield "baseline", baseline, 1.0
    for density in (0.75, 0.8, 0.85):
        for first in (0, 1):
            if density == 0.8 and first == 1:
                continue
            yield (
                f"sparse-{density}-first-{first}",
                dict(baseline, sparsity=density, skip_first_steps=first),
                1.0,
            )
    for fraction in (0.75, 0.5):
        yield f"ffn-{fraction}", baseline, fraction


def healthy(url):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def stop(process):
    # Only signal the process group created for this experiment.
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    research = (
        Path(__file__).resolve().parents[4] / "test/manual/minimax_h3_realtime/research"
    )
    if not research.is_dir():
        parser.error(
            "Run from the source checkout with the MiniMax-H3 research launcher"
        )
    sys.path.insert(0, str(research))
    from checkpoint_contract import (
        assert_endpoint,
        digest,
        load_verified,
        model_id,
        reject_runtime_lora,
        require_launcher,
    )
    from h3_chunks import Client

    config = json.loads(args.config.read_text())
    for key in ("server_command", "client_command"):
        command = config[key]
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(part, str) for part in command)
        ):
            parser.error(f"{key} must be a nonempty array of strings")
    environment = dict(os.environ, **config.get("environment", {}))
    for key in ("server_command", "client_command"):
        reject_runtime_lora(config[key], environment)
    require_launcher(config["server_command"])
    checkpoint = load_verified(
        environment["H3_CHECKPOINT_MANIFEST"], environment["H3_CHECKPOINT_RECEIPT"]
    )
    if checkpoint["family"] not in ("flashgen4", "turbo8"):
        parser.error("The sparse sweep requires a verified merged checkpoint")
    checkpoint_sha = digest(checkpoint)
    served_model = model_id(checkpoint_sha)
    environment.update(
        H3_SERVE_ONLY="1",
        H3_RESEARCH_MODEL_ID=served_model,
        H3_EXPECTED_CHECKPOINT_SHA=checkpoint_sha,
    )
    warmups, repeats = config.get("warmups", 3), config.get("repeats", 5)
    if (
        type(warmups) is not int
        or warmups < 1
        or type(repeats) is not int
        or repeats < 1
    ):
        parser.error("warmups and repeats must be positive integers")
    cases = list(variants(config.get("attention_config")))
    if args.only:
        if set(args.only) - {case[0] for case in cases}:
            parser.error("Unknown variant in --only")
        cases = [case for case in cases if case[0] in args.only]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name, attention, fraction in cases:
        command = config["server_command"]
        variant_env = dict(
            environment,
            H3_ATTENTION_BACKEND="ascend_sparse_attn",
            H3_ATTENTION_CONFIG=json.dumps(attention),
            H3_FFN_ACTIVE_FRACTION=str(fraction),
        )
        record = {
            "variant": name,
            "server_command": command,
            "client_command": config["client_command"],
            "checkpoint_manifest_sha256": checkpoint_sha,
            "attention": attention,
            "ffn_active_fraction": fraction,
            "status": "planned",
        }
        print(json.dumps(record), flush=True)
        if args.dry_run:
            with (args.output / "results.jsonl").open("a") as output:
                output.write(json.dumps(record) + "\n")
            continue
        url = config["health_url"]
        if healthy(url):
            raise RuntimeError(
                "Health URL already serves a process; choose an unused port"
            )
        try:
            with (args.output / f"{name}-server.log").open("w") as log:
                process = subprocess.Popen(
                    command,
                    env=variant_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    deadline = time.monotonic() + config.get("startup_timeout_s", 1800)
                    while not healthy(url):
                        if process.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError(f"{name}: server failed readiness")
                        time.sleep(1)
                    assert_endpoint(
                        Client(url.removesuffix("/health"), 30), served_model
                    )
                    samples = []
                    evidence = {}
                    server_log = args.output / f"{name}-server.log"
                    for index in range(warmups + repeats):
                        log_offset = server_log.stat().st_size
                        with (args.output / f"{name}-client-{index}.log").open(
                            "w"
                        ) as client_log:
                            started = time.perf_counter()
                            subprocess.run(
                                config["client_command"],
                                env=environment,
                                stdout=client_log,
                                stderr=subprocess.STDOUT,
                                timeout=config.get("request_timeout_s", 600),
                                check=True,
                            )
                            elapsed = time.perf_counter() - started
                        with server_log.open("rb") as trace:
                            trace.seek(log_offset)
                            matches = re.findall(
                                rb"H3 denoise mode=\S+ scheduler_updates=(\d+) dit_evaluations=(\d+) request=(\S+) phase=request policy=\S+ profiled=0 diagnostic=0\b",
                                trace.read(),
                            )
                        if len(matches) != 1:
                            raise ValueError(
                                "Expected one unprofiled request with measured execution counters"
                            )
                        updates, forwards, request = matches[0]
                        request = request.decode()
                        if (
                            request in evidence
                            or int(updates) != checkpoint["schedule"]["steps"]
                            or int(forwards) < int(updates)
                        ):
                            raise ValueError(
                                "Invalid request identity or measured execution counts"
                            )
                        evidence[request] = {
                            "scheduler_updates": int(updates),
                            "dit_evaluations": int(forwards),
                            "warmup": index < warmups,
                        }
                        if index >= warmups:
                            samples.append(elapsed)
                    ordered = sorted(samples)
                    record.update(
                        status="completed",
                        execution=evidence,
                        latency_samples_s=samples,
                        median_s=statistics.median(samples),
                        p90_s=ordered[max(0, (9 * len(ordered) + 9) // 10 - 1)],
                    )
                finally:
                    stop(process)
        except Exception as error:
            record.update(status="failed", error=str(error))
            raise
        finally:
            with (args.output / "results.jsonl").open("a") as output:
                output.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
