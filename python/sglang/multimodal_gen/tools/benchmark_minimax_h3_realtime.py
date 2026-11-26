# SPDX-License-Identifier: Apache-2.0
"""Run warm-server H3 ablations from explicit server/client command arrays."""

import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
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
    config = json.loads(args.config.read_text())
    for key in ("server_command", "client_command"):
        command = config[key]
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(part, str) for part in command)
        ):
            parser.error(f"{key} must be a nonempty array of strings")
    managed = (
        "--attention-backend",
        "--attention-backend-config",
        "--minimax-h3-ffn-active-fraction",
    )
    if any(part.split("=")[0] in managed for part in config["server_command"]):
        parser.error("Remove managed attention/FFN flags from server_command")
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
    environment = dict(os.environ, **config.get("environment", {}))
    for name, attention, fraction in cases:
        command = config["server_command"] + [
            "--attention-backend",
            "ascend_sparse_attn",
            "--attention-backend-config",
            json.dumps(attention),
            "--minimax-h3-ffn-active-fraction",
            str(fraction),
        ]
        record = {
            "variant": name,
            "server_command": command,
            "client_command": config["client_command"],
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
                    env=environment,
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
                    samples = []
                    for index in range(warmups + repeats):
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
                        if index >= warmups:
                            samples.append(elapsed)
                    ordered = sorted(samples)
                    record.update(
                        status="completed",
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
