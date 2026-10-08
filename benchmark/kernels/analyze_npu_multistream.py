"""Measure compute-stream overlap from one device's exported profiler tasks."""

import argparse
import bisect
import csv
import gzip
import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

COMM = re.compile(
    r"hccl|hcom|all.?reduce|all.?gather|reduce.?scatter|all.?to.?all", re.I
)
WAIT = re.compile(
    r"notify|event.?wait|event.?record|stream.?wait|record.?event|wait.?event", re.I
)
COPY = re.compile(r"memcpy|memset|mem.?copy|\bdma\b", re.I)


@dataclass
class Task:
    name: str
    stream: str
    start: float
    end: float
    shape: str
    kind: str
    resource: str = "unknown"


def core_resource(metadata):
    # AI_CORE alone is insufficient: it does not establish Cube vs Vector.
    # Mixed kernels also cannot be split into Cube/Vector intervals without
    # finer-grained device evidence. Never infer resources from stream/name.
    text = str(metadata).upper()
    if "MIX" in text:
        return "mixed"
    cube = bool(re.search(r"\bCUBE\b|AI_CUBE|\bAIC\b", text))
    vector = bool(re.search(r"\bVECTOR\b|AI_VECTOR|\bAIV\b", text))
    if cube and vector:
        return "mixed"
    return "cube" if cube else "vector" if vector else "unknown"


def resource_metadata(row):
    return " ".join(
        str(row[key])
        for key in (
            "Task Type", "Task Category", "task_type", "AI Core Type",
            "Core Type", "core_type", "aicore_type", "kind",
        )
        if key in row
    )


def field(row, *names, default=None):
    for name in names:
        if name in row:
            return row[name]
    if default is not None:
        return default
    raise ValueError(f"Missing column {names}; available: {list(row)}")


def read_tasks(path, device_pid):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as source:
        if ".csv" in path.suffixes:
            rows = csv.DictReader(source)
            values = [
                (
                    field(row, "Name", "OP Name", "Op Name", "name"),
                    field(row, "Stream ID", "Stream Id", "stream"),
                    float(field(row, "Task Start Time(us)", "Start Time(us)", "ts")),
                    float(field(row, "Task Duration(us)", "Duration(us)", "dur")),
                    field(row, "Input Shapes", "Input Dims", "shape", default=""),
                    resource_metadata(row),
                )
                for row in rows
            ]
        else:
            if device_pid is None:
                raise ValueError("Chrome traces require --device-pid (Ascend Hardware)")
            trace = json.load(source)
            events = trace["traceEvents"] if isinstance(trace, dict) else trace
            values = []
            for event in events:
                if event.get("ph") != "X" or str(event.get("pid")) != device_pid:
                    continue
                args = event.get("args", {})
                values.append(
                    (
                        event["name"],
                        str(event["tid"]),
                        float(event["ts"]),
                        float(event.get("dur", 0)),
                        json.dumps(
                            args.get("Input Shapes", args.get("Input Dims", ""))
                        ),
                        resource_metadata(args) + " " + str(event.get("cat", "")),
                    )
                )
    tasks = []
    for name, stream, start, duration, shape, category in values:
        if duration <= 0 or WAIT.search(name):
            continue
        kind = (
            "comm" if COMM.search(name + category)
            else "other" if COPY.search(name + category)
            else "compute"
        )
        tasks.append(Task(
            name, str(stream), start, start + duration, shape, kind,
            core_resource(category) if kind == "compute" else kind,
        ))
    if not tasks:
        raise ValueError("No device tasks found; check the file and device PID")
    return tasks


def merged(tasks):
    result = []
    for start, end in sorted((task.start, task.end) for task in tasks):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def length(intervals):
    return sum(end - start for start, end in intervals)


def intersection(a, b):
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def kernel_stats(tasks):
    groups = defaultdict(list)
    for task in tasks:
        groups[(task.name, task.shape)].append(task.end - task.start)
    return [
        dict(
            name=name,
            shape=shape,
            calls=len(times),
            mean_us=statistics.mean(times),
            median_us=statistics.median(times),
        )
        for (name, shape), times in sorted(groups.items())
    ]


def resource_overlaps(tasks):
    events = defaultdict(lambda: defaultdict(int))
    for task in tasks:
        if task.kind not in ("compute", "comm"):
            continue
        resources = ["compute", task.resource] if task.kind == "compute" else ["comm"]
        for resource in resources:
            events[task.start][resource] += 1
            events[task.end][resource] -= 1
    counts = defaultdict(int)
    overlaps = {
        "cube_vector_ms": 0.0, "cube_cube_ms": 0.0,
        "vector_vector_ms": 0.0, "compute_comm_ms": 0.0,
    }
    previous = None
    for point in sorted(events):
        if previous is not None:
            duration = (point - previous) / 1000
            for key, active in (
                ("cube_vector_ms", counts["cube"] > 0 and counts["vector"] > 0),
                ("cube_cube_ms", counts["cube"] >= 2),
                ("vector_vector_ms", counts["vector"] >= 2),
                ("compute_comm_ms", counts["compute"] > 0 and counts["comm"] > 0),
            ):
                if active:
                    overlaps[key] += duration
        for resource, delta in events[point].items():
            counts[resource] += delta
        previous = point
    compute = [t for t in tasks if t.kind == "compute"]
    return {
        **overlaps,
        "unknown_compute_calls": sum(t.resource == "unknown" for t in compute),
        "mixed_compute_calls": sum(t.resource == "mixed" for t in compute),
        "unknown_compute_busy_ms": length(merged([
            t for t in compute if t.resource == "unknown"
        ])) / 1000,
        "mixed_compute_busy_ms": length(merged([
            t for t in compute if t.resource == "mixed"
        ])) / 1000,
        "classification": "profiler task/core metadata only; unknown/mixed excluded from core-pair overlap",
    }


def analyze(tasks, main_stream, sides, start, end, baseline):
    start = min(task.start for task in tasks) if start is None else start
    end = max(task.end for task in tasks) if end is None else end
    if end <= start:
        raise ValueError("The analysis window must have positive duration")
    window = [task for task in tasks if task.start < end and task.end > start]
    clipped = [
        Task(t.name, t.stream, max(t.start, start), min(t.end, end), t.shape, t.kind, t.resource)
        for t in window
    ]
    compute = [t for t in clipped if t.kind == "compute"]
    main = merged([t for t in compute if t.stream == main_stream])
    if not main:
        raise ValueError("No main-stream compute in the selected window")
    # Boundary tasks contribute busy time but cannot provide a full kernel duration.
    complete = [t for t in window if t.start >= start and t.end <= end]
    result = {
        "window_start_us": start,
        "window_end_us": end,
        "window_ms": (end - start) / 1000,
        "compute_busy_ms": length(merged(compute)) / 1000,
        "comm_busy_ms": length(merged([t for t in clipped if t.kind == "comm"]))
        / 1000,
        "comm_total_ms": sum(t.end - t.start for t in clipped if t.kind == "comm") / 1000,
        "resource_overlap": resource_overlaps(clipped),
        "other_busy_ms": length(merged([t for t in clipped if t.kind == "other"]))
        / 1000,
        "idle_ms": ((end - start) - length(merged(clipped))) / 1000,
        "side_streams": {},
        "kernels": kernel_stats([t for t in complete if t.kind == "compute"]),
    }
    for name, stream in sides.items():
        intervals = merged([t for t in compute if t.stream == stream])
        busy = length(intervals)
        overlap = intersection(intervals, main)
        result["side_streams"][name] = {
            "stream": stream,
            "compute_busy_ms": busy / 1000,
            "overlap_with_main_compute_ms": overlap / 1000,
            "hidden_pct_proxy": 100 * overlap / busy if busy else None,
            "kernels": kernel_stats(
                [t for t in complete if t.kind == "compute" and t.stream == stream]
            ),
        }
    side_intervals = merged([t for t in compute if t.stream in sides.values()])
    side_starts = [start for start, _ in side_intervals]
    overlapping_main = []
    for task in complete:
        if task.stream != main_stream or task.kind != "compute":
            continue
        index = bisect.bisect_left(side_starts, task.end) - 1
        if index >= 0 and side_intervals[index][1] > task.start:
            overlapping_main.append(task)
    baseline_stats = {
        (item["name"], item["shape"]): item for item in kernel_stats(baseline)
    }
    comparisons = []
    for item in kernel_stats(overlapping_main):
        reference = baseline_stats.get((item["name"], item["shape"]))
        if reference is None:
            continue
        comparisons.append({
            **item,
            "baseline_calls": reference["calls"],
            "baseline_median_us": reference["median_us"],
            "median_change_pct": 100 * (item["median_us"] / reference["median_us"] - 1),
            "match": (
                "name_and_shape"
                if item["shape"] not in ("", '""', "-", "N/A", "null")
                else "name_only"
            ),
        })
    result["main_kernels_during_overlap_vs_baseline"] = comparisons
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", type=Path)
    parser.add_argument("--device-pid", help="Chrome trace Ascend Hardware PID")
    parser.add_argument("--main-stream", required=True)
    parser.add_argument("--side-stream", action="append", default=[], metavar="NAME=ID")
    parser.add_argument("--start-us", type=float)
    parser.add_argument("--end-us", type=float)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-device-pid")
    parser.add_argument("--baseline-main-stream")
    parser.add_argument("--baseline-start-us", type=float)
    parser.add_argument("--baseline-end-us", type=float)
    args = parser.parse_args()
    try:
        sides = dict(value.split("=", 1) for value in args.side_stream)
        if args.main_stream in sides.values():
            raise ValueError("A side stream must differ from the main stream")
        tasks = read_tasks(args.tasks, args.device_pid)
        baseline = []
        if args.baseline:
            baseline = [
                task
                for task in read_tasks(args.baseline, args.baseline_device_pid)
                if task.kind == "compute"
                and task.stream == (args.baseline_main_stream or args.main_stream)
                and (
                    args.baseline_start_us is None
                    or task.start >= args.baseline_start_us
                )
                and (args.baseline_end_us is None or task.end <= args.baseline_end_us)
            ]
            if not baseline:
                raise ValueError(
                    "No baseline main-stream compute in the selected window"
                )
        result = analyze(
            tasks, args.main_stream, sides, args.start_us, args.end_us, baseline
        )
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
