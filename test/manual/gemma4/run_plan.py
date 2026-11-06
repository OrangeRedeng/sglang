"""Run all Gemma TP1/DP2 experiment rounds on the NPU host, without changing Git."""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from candidate_validation import source_hashes
from throughput_matrix import CASES, SCHEDULES, TARGETS, command_output, finite_number


def exploratory_score(rows):
    """Rank complete, correct-length unprofiled runs; this is not acceptance."""
    expected = {(scenario, case) for scenario in TARGETS for case in CASES}
    if len(rows) != 8 or {(r["scenario"], r["case"]) for r in rows} != expected:
        return None
    ratios = []
    for row in rows:
        metrics = row.get("metrics_ms", {})
        if set(metrics) != set(TARGETS[row["scenario"]][1]) or any(
            not finite_number(value) or value < 0 for value in metrics.values()
        ):
            return None
        if row.get("measurement_kind") != "unprofiled":
            return None
        if any(
            row["checks"].get(key) is not True
            for key in ("completed", "failed", "output_tokens")
        ):
            return None
        rate = row["completed_rps"]
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(rate)
            or rate <= 0
        ):
            return None
        ratios.append(rate / TARGETS[row["scenario"]][0])
    return math.exp(sum(math.log(value) for value in ratios) / 8)


def feasibility(rows):
    result = []
    for scenario in TARGETS:
        for case in CASES:
            matches = [
                r for r in rows if r["scenario"] == scenario and r["case"] == case
            ]
            rates = [
                r["completed_rps"]
                for r in matches
                if finite_number(r.get("completed_rps")) and r["completed_rps"] > 0
            ]
            target = TARGETS[scenario][0]
            result.append(
                {
                    "scenario": scenario,
                    "case": case,
                    "target_rps": target,
                    "repetitions": len(matches),
                    "valid_rate_repetitions": len(rates),
                    "all_runs_pass": len(matches) == 3
                    and all(r["pass"] for r in matches),
                    "minimum_completed_rps": min(rates) if rates else None,
                    "mean_completed_rps": sum(rates) / len(rates) if rates else None,
                    "required_speedup_from_minimum": max(1.0, target / min(rates))
                    if rates and min(rates) > 0
                    else None,
                }
            )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server-root", type=Path, required=True)
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--bench-script", type=Path, required=True)
    p.add_argument("--datasets", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="gemma4-w8a8-test")
    p.add_argument("--port", type=int, default=30014)
    p.add_argument("--requests", type=int, default=256)
    p.add_argument("--warmups", type=int, default=32)
    args = p.parse_args()
    for key in (
        "server_root",
        "baseline_root",
        "bench_script",
        "datasets",
        "model_path",
        "output",
    ):
        setattr(args, key, getattr(args, key).resolve())
    baseline_sha = command_output(["git", "rev-parse", "HEAD"], args.baseline_root)
    if baseline_sha["returncode"] or not baseline_sha["output"].startswith(
        "c8d7020179"
    ):
        p.error("Baseline checkout must be pinned to c8d7020179")
    args.output.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    fingerprint = source_hashes(args.server_root)
    (args.output / "plan.json").write_text(
        json.dumps(
            {
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "source_sha256": fingerprint,
                "glm_reference": "0b3f0e2baf16c8dcb21882de0b8de623fd18635f",
                "note": "Exploratory choices are not accepted defaults. Only the final 24 unprofiled SLO checks can accept a candidate.",
            },
            indent=2,
        )
    )
    env = os.environ.copy()
    env.update(
        ASCEND_RT_VISIBLE_DEVICES="2",
        TASK_QUEUE_ENABLE="1",
        PYTHONPATH=str(args.server_root / "python")
        + os.pathsep
        + env.get("PYTHONPATH", ""),
    )
    ledger = args.output / "experiments.jsonl"

    def record(**entry):
        with ledger.open("a") as f:
            f.write(json.dumps({"time": time.time(), **entry}) + "\n")

    def run(name, command):
        if source_hashes(args.server_root) != fingerprint:
            raise RuntimeError("Candidate sources changed during the plan")
        record(round=name, status="started", command=command)
        with (args.output / f"{name}.log").open("w") as log:
            status = subprocess.run(
                command,
                cwd=args.server_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        record(round=name, status="finished", returncode=status)
        print(f"{name}: returncode={status}", flush=True)
        return status == 0

    reference_environment = None
    observations = []

    def matrix(
        name,
        root,
        phase,
        *,
        qkv="serial",
        concurrency=28,
        schedule="base",
        quality=None,
    ):
        nonlocal reference_environment
        destination = args.output / name
        cmd = [
            sys.executable,
            str(scripts / "throughput_matrix.py"),
            "--phase",
            phase,
            "--server-root",
            str(root),
            "--bench-script",
            str(args.bench_script),
            "--datasets",
            str(args.datasets),
            "--model-path",
            str(args.model_path),
            "--model",
            args.model,
            "--port",
            str(args.port),
            "--requests",
            str(args.requests),
            "--warmups",
            str(args.warmups),
            "--output",
            str(destination),
            "--qkv",
            qkv,
            "--concurrency",
            str(concurrency),
            "--schedule",
            schedule,
        ]
        if quality is not None:
            cmd += ["--quality-baseline", str(quality)]
        ok = run(name, cmd)
        environment_path = destination / "environment.json"
        if environment_path.exists():
            environment = json.loads(environment_path.read_text())
            fixed_environment = {
                key: environment[key]
                for key in (
                    "versions",
                    "cann_version_files",
                    "benchmark_sha256",
                    "checkpoint_metadata",
                )
            }
            if reference_environment is None:
                reference_environment = fixed_environment
            elif reference_environment != fixed_environment:
                raise RuntimeError(
                    "Runtime, checkpoint or benchmark identity drifted between rounds"
                )
        if not ok and phase != "acceptance":
            return None
        report = destination / "acceptance.json"
        result = json.loads(report.read_text()) if report.exists() else None
        if result and phase in ("baseline", "candidate"):
            observations.extend(result["results"])
        return result

    baseline = matrix("00-baseline", args.baseline_root, "baseline")
    if baseline is None:
        raise RuntimeError("Two-NPU baseline failed; inspect 00-baseline.log")
    quality = args.output / "00-baseline/base/quality.json"
    metadata = matrix("01-metadata", args.server_root, "baseline", quality=quality)
    if metadata is None:
        raise RuntimeError("Metadata candidate failed quality or serving checks")
    grouped = [
        (c, [r for r in metadata["results"] if r["concurrency"] == c])
        for c in (24, 28, 29)
    ]
    valid = [
        (exploratory_score(rows), c)
        for c, rows in grouped
        if exploratory_score(rows) is not None
    ]
    if not valid:
        raise RuntimeError("No complete metadata concurrency result")
    _, concurrency = max(valid)
    best = {"qkv": "serial", "schedule": "base"}
    best_rows = [r for r in metadata["results"] if r["concurrency"] == concurrency]
    score = exploratory_score(best_rows)

    def consider(name, report, config):
        nonlocal best, best_rows, score
        candidate_score = exploratory_score(report["results"]) if report else None
        improved = candidate_score is not None and candidate_score > score
        record(
            round=name,
            decision="exploratory-improvement" if improved else "reject",
            normalized_geomean_rps=candidate_score,
            parent_score=score,
            config=config,
        )
        if improved:
            best, best_rows, score = config.copy(), report["results"], candidate_score

    native = matrix(
        "02-native-qkv",
        args.server_root,
        "candidate",
        qkv="native",
        concurrency=concurrency,
        quality=quality,
    )
    consider("02-native-qkv", native, {**best, "qkv": "native"})
    # Kernel checks run with the serving processes stopped by matrix's finally.
    kernels = scripts.parent / "kernels"
    run(
        "02-qkv-correctness",
        [
            sys.executable,
            str(kernels / "benchmark_gemma4_qkv_norm_npu.py"),
            "--candidate",
            "serial",
            "--output",
            str(args.output / "qkv-validation.json"),
        ],
    )
    record(
        round="04-glm-audit",
        decision="no-additional-port",
        reference="0b3f0e2baf",
        reason="Existing INT8 consumer already unpacks QuantizedOperand; SwiGLU, DSA and TP communication paths do not match Gemma GeGLU / independent TP1 replicas",
    )
    retuned = matrix(
        "05-concurrency",
        args.server_root,
        "baseline",
        qkv=best["qkv"],
        quality=quality,
    )
    if retuned is None:
        raise RuntimeError("Post-kernel concurrency comparison failed")
    choices = [
        (exploratory_score([r for r in retuned["results"] if r["concurrency"] == c]), c)
        for c in (24, 28, 29)
    ]
    choices = [(value, c) for value, c in choices if value is not None]
    if not choices:
        raise RuntimeError("No valid post-kernel concurrency result")
    score, concurrency = max(choices)
    best_rows = [r for r in retuned["results"] if r["concurrency"] == concurrency]
    record(
        round="05-concurrency", concurrency=concurrency, normalized_geomean_rps=score
    )
    # Keep kernel settings fixed for every scheduling comparison.
    fixed = best.copy()
    parents = {
        "off": "base",
        "r025-ms50-cap6": "base",
        "r025-ms25-cap6": "r025-ms50-cap6",
        "r05-ms25-cap6": "r025-ms25-cap6",
        "r05-ms25-cap5": "r05-ms25-cap6",
        "r05-ms50-cap5": "r05-ms25-cap5",
        "r025-ms50-cap5": "r05-ms50-cap5",
        "r025-ms25-cap5": "r025-ms50-cap5",
    }
    schedule_scores = {"base": score}
    schedule_reports = {}
    for schedule in SCHEDULES:
        if schedule == "base":
            continue
        report = matrix(
            f"05-{schedule}",
            args.server_root,
            "candidate",
            qkv=fixed["qkv"],
            concurrency=concurrency,
            schedule=schedule,
            quality=quality,
        )
        schedule_reports[schedule] = report
    for schedule, report in schedule_reports.items():
        local = exploratory_score(report["results"]) if report else None
        schedule_scores[schedule] = local
        parent = schedule_scores[parents[schedule]]
        record(
            round=f"05-{schedule}",
            parent=parents[schedule],
            parent_score=parent,
            normalized_geomean_rps=local,
            one_knob_improved=local is not None
            and parent is not None
            and local > parent,
        )
        if local is not None and parent is not None and local > parent:
            consider(f"05-{schedule}", report, {**fixed, "schedule": schedule})
    # Separate timing and diagnostic traces from all acceptance traffic.
    timing = matrix(
        "06-profile-timing",
        args.server_root,
        "profile",
        qkv=best["qkv"],
        concurrency=concurrency,
        schedule=best["schedule"],
        quality=quality,
    )
    diagnostic = args.output / "06-profile-shapes"
    # Reuse the same command recorded for the timing stage, only adding shapes and output.
    entries = [json.loads(line) for line in ledger.read_text().splitlines()]
    profile_cmd = next(
        e["command"]
        for e in reversed(entries)
        if e.get("round") == "06-profile-timing" and "command" in e
    )
    profile_cmd[profile_cmd.index("--output") + 1] = str(diagnostic)
    shape_ok = run("06-profile-shapes", profile_cmd + ["--record-shapes"])
    accepted = matrix(
        "07-acceptance",
        args.server_root,
        "acceptance",
        qkv=best["qkv"],
        concurrency=concurrency,
        schedule=best["schedule"],
        quality=quality,
    )
    rows = accepted["results"] if accepted else []
    baseline_rows = [r for r in baseline["results"] if r["concurrency"] == concurrency]
    baseline_score = exploratory_score(baseline_rows)
    passed = bool(accepted and accepted["acceptance_pass"] and timing and shape_ok)
    case_results = feasibility(rows)
    for result in case_results:
        observed = [
            row["completed_rps"]
            for row in observations
            if row["scenario"] == result["scenario"]
            and row["case"] == result["case"]
            and finite_number(row.get("completed_rps"))
        ]
        result["best_exploratory_rps"] = max(observed) if observed else None
    summary = {
        "acceptance_pass": passed,
        "concurrency": concurrency,
        "exploratory_best": best,
        "normalized_geomean_rps": score,
        "baseline_normalized_geomean_rps": baseline_score,
        "exploratory_speedup": score / baseline_score if baseline_score else None,
        "profiles_complete": bool(timing and shape_ok),
        "cases": case_results,
        "note": "Targets are unchanged. A failed run is not an accepted configuration or a proven hardware ceiling.",
    }
    (args.output / "plan-result.json").write_text(json.dumps(summary, indent=2))
    record(
        round="07-acceptance",
        decision="keep" if passed else "inconclusive",
        summary=summary,
    )
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
