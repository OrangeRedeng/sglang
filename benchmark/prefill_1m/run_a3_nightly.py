#!/usr/bin/env python3
"""Bounded host-side A3 comparisons with live logs outside Docker."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def plan():
    # Historical timings label recipes, never expected outcomes.
    names = [
        ("01-control-40s-recipe", "legacy-control-40s", 1024),
        ("02-control-24s-recipe", "baseline", 2048),
        ("03-prefill-graph-full", "graph-full", 2048),
        ("04-prefill-graph-piecewise", "graph-piecewise", 2048),
        ("05-tp-routed-scale", "tp-routed-scale", 2048),
        ("06-tp-prequant", "tp-prequant", 2048),
        ("07-tp-shared", "tp-shared", 2048),
        ("08-tp-shared-activation", "tp-shared-activation", 2048),
        ("09-tp-combined", "moe-fused-combined", 2048),
        ("10-metadata-reuse", "dsa-metadata", 2048),
        ("11-metadata-batch2", "dsa-metadata-batch2", 2048),
        ("12-packed-metadata-batch2", "dsa-metadata-packed-batch2", 2048),
        ("13-indexer-fp16-ablation", "indexer-fp16", 2048),
        ("14-auto-stream-thresholds", "auto-stream-thresholds", 2048),
        ("15-auto-hccl", "auto-hccl", 2048),
        ("16-stream-auto-combined", "moe-stream-auto", 2048),
        ("17-control-24s-repeat", "baseline", 2048),
        ("18-megamoe-34s-recipe", "legacy-megamoe-34s", 1024),
        ("19-megamoe-24s-target", "megamoe", 2048),
        ("20-megamoe-metadata-combined", "dsa-metadata-megamoe", 2048),
    ]
    return [dict(name=n, variant=v, dispatch=d) for n, v, d in names]


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def run(command, *, timeout=20, **kwargs):
    return subprocess.run(command, timeout=timeout, check=False, **kwargs)


def stop_previous():
    holders = run(
        ["fuser", "/tmp/sglang-bde-overnight.lock"], capture_output=True, text=True
    )
    for word in holders.stdout.split():
        if not word.isdigit():
            continue
        pid = int(word)
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes()
            if b"run_bde_overnight.py" not in command:
                raise RuntimeError(f"Refusing to stop unrelated lock owner {pid}")
            os.kill(pid, signal.SIGINT)
            for _ in range(45):
                if not Path(f"/proc/{pid}").exists():
                    break
                time.sleep(1)
            else:
                os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def export_live(args, results, folder, stamp):
    remote_file = results / "remote-path.txt"
    if not remote_file.is_file():
        return
    remote = remote_file.read_text().strip()
    if not remote.startswith("/tmp/sglang-bde-night-"):
        raise RuntimeError("Invalid container artifact directory")
    live = folder / "live-container"
    live.mkdir(exist_ok=True)
    with (folder / "export.log").open("a") as log:
        copied = run(
            ["docker", "cp", f"{args.container}:{remote}/.", str(live)],
            stdout=log,
            stderr=log,
            timeout=30,
        )
        log.write(f"{stamp} artifact export exit={copied.returncode}\n")
    checkpoints = folder / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    for name, command in (
        ("npu-smi", ["docker", "exec", args.container, "npu-smi", "info"]),
        (
            "processes",
            [
                "docker",
                "exec",
                args.container,
                "ps",
                "-eo",
                "pid,ppid,stat,etime,comm,args",
            ],
        ),
    ):
        with (checkpoints / f"{stamp}-{name}.txt").open("w") as log:
            run(command, stdout=log, stderr=log)
    for server in live.glob("*/server.log"):
        with server.open("rb") as log:
            log.seek(max(0, server.stat().st_size - 32768))
            tail = log.read()
        (checkpoints / f"{stamp}-{server.parent.name}-server-tail.log").write_bytes(
            tail
        )


def progress(completed, total, name, elapsed, status="running"):
    cells = int(24 * completed / total)
    print(
        f"[{'#' * cells}{'.' * (24 - cells)}] {completed}/{total} "
        f"{name} | {elapsed / 60:.1f} min | {status}",
        flush=True,
    )


def summarize(root, records):
    save(root / "progress.json", records)
    lines = [
        "# A3 nightly progress",
        "",
        "Performance only; numerical quality unchecked. Recipe names are historical labels.",
        "",
        "| Case | Status | TTFT mean / P90 (s) | Wall (min) |",
        "|---|---|---|---|",
    ]
    for record in records:
        metrics = record.get("metrics", {})
        mean, p90 = metrics.get("TTFT avg"), metrics.get("TTFT P90")
        ttft = (
            f"{mean / 1000:.3f} / {p90 / 1000:.3f}"
            if mean is not None and p90 is not None
            else ""
        )
        lines.append(
            f"| {record['name']} | {record['status']} | {ttft} | {record.get('wall_s', 0) / 60:.1f} |"
        )
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="sglang-a3")
    parser.add_argument("--repo", default="/home/swx1199799/megamoe_test/sglang")
    parser.add_argument(
        "--root", type=Path, default=Path("/home/tbaydasov/aisbench/overnight-results")
    )
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--case-minutes", type=float, default=30)
    parser.add_argument("--stop-previous", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 < args.hours <= 24 or args.case_minutes < 20:
        parser.error("hours must be in (0,24]; case-minutes must be at least 20")
    cases = plan()
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    cases=cases,
                    hours=args.hours,
                    case_minutes=args.case_minutes,
                    requests=16,
                    concurrency=16,
                    tokens=1000000,
                    repeats=1,
                ),
                indent=2,
            )
        )
        return 0
    if args.stop_previous:
        stop_previous()
    holders = run(
        ["fuser", "/tmp/sglang-bde-overnight.lock"], capture_output=True, text=True
    )
    if holders.stdout.strip():
        raise RuntimeError(
            "Runner still owns the lock; previous cleanup has not completed"
        )
    root = args.root / (
        "retest-nightly-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    root.mkdir(parents=True)
    print(f"Host logs: {root}", flush=True)
    docker = ["docker", "exec", "-w", args.repo, args.container]
    source = root / "source"
    source.mkdir()
    for name in (
        "run_bde_overnight.py",
        "run_a3_nightly.py",
        "feature_combinations.py",
        "test_glm53_features.py",
        "launch_bde_v2.py",
    ):
        copied = run(
            [
                "docker",
                "cp",
                f"{args.container}:{args.repo}/benchmark/prefill_1m/{name}",
                str(source / name),
            ]
        )
        if copied.returncode:
            raise RuntimeError(f"Cannot snapshot {name}")
    for name, command in (
        ("commit.txt", [*docker, "git", "log", "-1", "--format=%H %s"]),
        ("source-diff.patch", [*docker, "git", "diff", "--binary", "HEAD"]),
    ):
        with (source / name).open("w") as log:
            run(command, stdout=log, stderr=log)
    identity = """import importlib.metadata as m, importlib.util as u, json, os, sys
packages = ('torch','torch-npu','deep-ep','cann-ops-transformer','sgl-kernel-npu','sglang')
versions = {}
for package in packages:
    try: versions[package] = m.version(package)
    except m.PackageNotFoundError: versions[package] = None
paths = {}
for module in ('torch','torch_npu','deep_ep','cann_ops_transformer','sgl_kernel_npu','sglang'):
    try:
        spec = u.find_spec(module)
        paths[module] = spec.origin if spec else None
    except (ImportError, ValueError): paths[module] = None
print(json.dumps(dict(python=sys.version, executable=sys.executable, versions=versions,
 paths=paths, environment={k:os.environ.get(k) for k in
 ('ASCEND_HOME_PATH','CANN_HOME','ASCEND_RT_VISIBLE_DEVICES','HCCL_SOCKET_IFNAME','HCCL_IF_IP')}), indent=2))
"""
    with (source / "runtime-identity.json").open("w") as log:
        run([*docker, "python3", "-c", identity], stdout=log, stderr=log, timeout=60)
    save(
        source / "sha256.json",
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.iterdir()
            if p.is_file()
        },
    )
    save(
        root / "plan.json",
        dict(
            cases=cases,
            arguments=vars(args) | {"root": str(args.root)},
            numerical_quality="unchecked",
            target_ttft_is_guaranteed=False,
        ),
    )
    records = []
    deadline = time.monotonic() + args.hours * 3600
    for index, case in enumerate(cases):
        # Reserve bounded attempts for the three MegaMoE cases at the end.
        reserve = min(5400, args.case_minutes * 180) if index < len(cases) - 3 else 0
        remaining = deadline - time.monotonic()
        if remaining < 1200 + reserve:
            records.append(dict(case, status="skipped-budget"))
            summarize(root, records)
            continue
        folder = root / case["name"]
        folder.mkdir()
        cap = min(args.case_minutes * 60, remaining - reserve)
        command = [
            sys.executable,
            "-u",
            str(source / "run_bde_overnight.py"),
            "--container",
            args.container,
            "--repo",
            args.repo,
            "--performance-only",
            "--reference-recipe",
            "historical-bde",
            "--variants",
            case["variant"],
            "--length",
            "1000000",
            "--requests",
            "16",
            "--maxreq",
            "1",
            "--chunk",
            "32768",
            "--kv-rows",
            "67584",
            "--context",
            "1048576",
            "--memfrac",
            "0.85",
            "--repeats",
            "1",
            "--startup-timeout",
            "900",
            "--benchmark-timeout",
            "600",
            "--watchdog-timeout",
            "180",
            "--max-hours",
            str(cap / 3600),
            "--megamoe-dispatch-tokens",
            str(case["dispatch"]),
            "--output",
            str(folder / "results"),
        ]
        save(folder / "command.json", command)
        started = time.monotonic()
        record = dict(case, status="running")
        records.append(record)
        summarize(root, records)
        with (folder / "driver.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            next_export = 0
            try:
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    record["wall_s"] = round(elapsed, 1)
                    summarize(root, records)
                    progress(index, len(cases), case["name"], elapsed)
                    if elapsed >= next_export:
                        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                        try:
                            export_live(args, folder / "results", folder, stamp)
                        except (OSError, subprocess.TimeoutExpired) as exc:
                            with (folder / "export.log").open("a") as error:
                                error.write(f"{stamp}: {exc}\n")
                        next_export = elapsed + 600
                    if elapsed > cap:
                        record["status"] = "timeout"
                        process.send_signal(signal.SIGINT)
                        break
                    time.sleep(15)
            except KeyboardInterrupt:
                process.send_signal(signal.SIGINT)
                raise
            finally:
                try:
                    process.wait(timeout=150)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                try:
                    export_live(
                        args,
                        folder / "results",
                        folder,
                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    (folder / "final-export-error.txt").write_text(str(exc))
        result = folder / "results/results.json"
        details = json.loads(result.read_text()) if result.is_file() else []
        record.update(
            exit_code=process.returncode, wall_s=round(time.monotonic() - started, 1)
        )
        measured = next((item for item in details if item.get("metrics")), None)
        if measured:
            record.update(status=measured["status"], metrics=measured["metrics"])
        elif record["status"] == "running":
            record["status"] = "failed-or-blocked"
        record["details"] = details
        summarize(root, records)
        progress(
            index + 1, len(cases), case["name"], record["wall_s"], record["status"]
        )
        # Never stack another experiment on a surviving server from this case.
        live = folder / "live-container"
        for pid_file in live.glob("*/server.pid"):
            pid = pid_file.read_text().strip()
            if pid.isdigit():
                exists = run(
                    ["docker", "exec", args.container, "test", "-d", f"/proc/{pid}"]
                )
                if exists.returncode == 0:
                    print(
                        f"STOP: owned server {pid} survived cleanup; logs preserved at {root}",
                        flush=True,
                    )
                    return 1
    empty = [
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and not p.stat().st_size
    ]
    save(root / "removed-empty-files.json", empty)
    for path in empty:
        (root / path).unlink()
    print(f"Finished: {root / 'SUMMARY.md'}", flush=True)
    return 1 if any(r["status"] != "diagnostic-unqualified" for r in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
