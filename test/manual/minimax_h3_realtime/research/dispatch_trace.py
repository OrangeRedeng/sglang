# SPDX-License-Identifier: Apache-2.0
"""Aggregate clock-aligned trace events using explicit producer correlation."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

KINDS = {"host", "enqueue", "dispatch", "compute", "wait", "hccl"}


def interval_union(intervals):
    total = 0
    end = -math.inf
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def analyze(records):
    """Input times use one aligned microsecond clock per rank."""
    owners = {}
    seen = set()
    unmatched = defaultdict(int)
    steps = defaultdict(list)
    for item in records:
        identity = item["rank"], item["event_id"]
        if identity in seen:
            raise ValueError(f"Duplicate trace event: {identity}")
        seen.add(identity)
        start, end = item["start_us"], item["end_us"]
        if (
            item["kind"] not in KINDS
            or not all(map(math.isfinite, (start, end)))
            or end < start
        ):
            raise ValueError(f"Invalid trace event: {identity}")
        if item["kind"] == "host" and "request" in item and "step" in item:
            owners[identity] = item["request"], item["step"], item["rank"]

    events = {(item["rank"], item["event_id"]): item for item in records}

    def resolve(identity):
        visited = set()
        while identity not in owners and identity in events:
            if identity in visited:
                raise ValueError(f"Cyclic trace link: {identity}")
            visited.add(identity)
            parent = events[identity].get("producer")
            if parent is None:
                break
            identity = identity[0], parent
        return identity if identity in owners else None

    for identity, item in events.items():
        producer = resolve(identity)
        if producer is None:
            unmatched[item["kind"]] += 1
            continue
        steps[owners[producer]].append(item)

    output = []
    for (request, step, rank), step_events in sorted(steps.items()):
        resources = defaultdict(list)
        device = []
        queue = []
        invalid_queue_links = 0
        for event in step_events:
            kind = event["kind"]
            span = event["start_us"], event["end_us"]
            resources[kind, event["resource"]].append(span)
            if kind in {"compute", "wait", "hccl"}:
                device.append(span)
            if kind == "dispatch" and event.get("producer") is not None:
                producer = events.get((rank, event["producer"]))
                if producer is not None and producer["kind"] == "enqueue":
                    queued_at = producer["end_us"]
                    if queued_at <= span[0]:
                        queue.append((queued_at, span[0]))
                    else:
                        invalid_queue_links += 1
        endpoints = defaultdict(int)
        for start, end in queue:
            endpoints[start] += 1
            endpoints[end] -= 1
        depth = high = 0
        for _, change in sorted(endpoints.items()):
            depth += change
            high = max(high, depth)

        collective = {}
        for event in step_events:
            if event["kind"] == "hccl" and all(
                key in event
                for key in ("group", "sequence", "callsite", "payload_bytes")
            ):
                key = (event["group"], event["sequence"])
                evidence = {
                    name: event[name]
                    for name in ("group", "sequence", "callsite", "payload_bytes")
                }
                if key in collective and collective[key] != evidence:
                    raise ValueError(f"Conflicting collective evidence: {key}")
                collective[key] = evidence
        last_host = max(
            event["end_us"] for event in step_events if event["kind"] == "host"
        )
        last_device = max((end for _, end in device), default=None)
        busy = interval_union(device)
        output.append(
            dict(
                request=request,
                step=step,
                rank=rank,
                resource_unions_us=[
                    dict(kind=kind, resource=resource, active_us=interval_union(spans))
                    for (kind, resource), spans in sorted(resources.items())
                ],
                device_busy_us=busy,
                device_uncovered_us=(
                    last_device - min(start for start, _ in device) - busy
                    if device
                    else None
                ),
                host_return_us=last_host,
                device_completion_us=last_device,
                observed_queue_depth=high,
                invalid_queue_links=invalid_queue_links,
                collective_calls=list(collective.values()),
            )
        )
    return dict(steps=output, unmatched=dict(unmatched))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trace", type=Path, help="Clock-aligned normalized trace JSON array"
    )
    args = parser.parse_args()
    print(json.dumps(analyze(json.loads(args.trace.read_text())), indent=2))


if __name__ == "__main__":
    main()
