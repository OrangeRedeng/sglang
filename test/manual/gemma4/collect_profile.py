"""Collect bounded, stage-separated profiles from both loaded TP1/DP2 workers."""

import argparse
import csv
import json
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path


def rpc(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        raw = response.read()
        result = json.loads(raw) if raw else {}
    if isinstance(result, dict) and result.get("success") is False:
        raise RuntimeError(result)
    return result


def summarize(output):
    reports = []
    for path in sorted(output.glob("**/ASCEND_PROFILER_OUTPUT/trace_view.json")):
        data = json.loads(path.read_text())
        events = data.get("traceEvents", []) if isinstance(data, dict) else data
        steps = [
            e["name"]
            for e in events
            if e.get("cat") == "cpu_op" and e.get("name", "").startswith("step[")
        ]
        stages = Counter(
            "prefill" if "EXTEND" in name else "decode"
            for name in steps
            if "EXTEND" in name or "DECODE" in name
        )
        details = path.parent / "kernel_details.csv"
        devices = []
        if details.is_file():
            with details.open() as f:
                devices = sorted({row["Device_id"] for row in csv.DictReader(f)})
        top = []
        stat = path.parent / "op_statistic.csv"
        if stat.is_file():
            with stat.open() as f:
                rows = list(csv.DictReader(f))
            top = sorted(
                rows, key=lambda row: float(row["Total Time(us)"]), reverse=True
            )[:10]
        reports.append(
            {
                "trace": str(path),
                "devices": devices,
                "observed_steps": dict(stages),
                "step_labels": steps,
                "top_device_ops": top,
            }
        )
    return reports


def complete(reports):
    coverage = {device: set() for device in ("2", "3")}
    for report in reports:
        for device in report["devices"]:
            if device in coverage:
                coverage[device].update(
                    stage for stage, count in report["observed_steps"].items() if count
                )
    return all(stages == {"prefill", "decode"} for stages in coverage.values())


def collect(
    url,
    server_log,
    output,
    *,
    steps=3,
    min_running=12,
    timeout=600,
    record_shapes=False,
    log_offset=None,
):
    info = rpc(url + "/server_info")
    effective = info.get("server_args", {})
    if effective.get("tp_size") != 1 or effective.get("dp_size") != 2:
        raise ValueError("Profiling requires the intended TP1/DP2 server")
    if output.exists():
        raise ValueError(f"Use a new profile directory: {output}")
    output.mkdir(parents=True)
    (output / "server_info.json").write_text(json.dumps(info, indent=2))
    offset = server_log.stat().st_size if log_offset is None else log_offset
    deadline = time.monotonic() + timeout
    # Observe new traffic from both replicas; stale log lines cannot arm capture.
    loaded = set()
    while len(loaded) != 2:
        if time.monotonic() > deadline:
            raise TimeoutError("Both DP workers did not reach the requested batch size")
        with server_log.open("rb") as f:
            f.seek(offset)
            text = f.read().decode(errors="replace")
        for rank, batch in re.findall(
            r"DP([01])\].*Decode batch.*#running-req:\s*(\d+)", text
        ):
            if int(batch) >= min_running:
                loaded.add(rank)
        time.sleep(0.2)
    body = {
        "activities": ["CPU", "GPU"],
        "num_steps": steps,
        "profile_by_stage": True,
        "output_dir": str(output),
        "with_stack": False,
        "record_shapes": record_shapes,
        "merge_profiles": False,
        "profile_prefix": "gemma4-two-npu",
    }
    (output / "request.json").write_text(json.dumps(body, indent=2))
    rpc(url + "/start_profile", body)
    while True:
        try:
            reports = summarize(output)
        except (json.JSONDecodeError, OSError):
            reports = []  # Profiler export can still be writing a trace.
        if complete(reports):
            (output / "coverage.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "reports": reports,
                        "note": "Counts are observed step annotations, not the requested num_steps. Timing is instrumented.",
                    },
                    indent=2,
                )
            )
            return reports
        if time.monotonic() > deadline:
            try:
                rpc(url + "/stop_profile", {})
            except (OSError, RuntimeError):
                pass
            (output / "coverage.json").write_text(
                json.dumps({"complete": False, "reports": reports}, indent=2)
            )
            raise TimeoutError(
                "Missing prefill/decode traces for device2 or device3; inspect coverage.json"
            )
        time.sleep(2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:30014")
    p.add_argument("--server-log", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--min-running", type=int, default=12)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--record-shapes", action="store_true")
    args = p.parse_args()
    if args.steps < 1 or args.min_running < 1 or args.timeout <= 0:
        p.error("Steps, minimum batch size and timeout must be positive")
    collect(
        args.url.rstrip("/"),
        args.server_log,
        args.output.resolve(),
        steps=args.steps,
        min_running=args.min_running,
        timeout=args.timeout,
        record_shapes=args.record_shapes,
    )
    print(f"Verified both devices and both stages: {args.output}")


if __name__ == "__main__":
    main()
