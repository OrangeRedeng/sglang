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
    request_queues = defaultdict(list)
    collective_sequences = defaultdict(lambda: defaultdict(dict))
    group_members = {}
    incomplete_hccl = set()
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
        request_queues[request, rank].extend(queue)

        collective = {}
        for event in step_events:
            if event["kind"] != "hccl":
                continue
            if not all(
                key in event
                for key in ("group", "sequence", "callsite", "payload_bytes")
            ):
                incomplete_hccl.add((request, rank))
                continue
            key = (event["group"], event["sequence"])
            evidence = {
                name: event[name]
                for name in ("group", "sequence", "callsite", "payload_bytes")
            }
            if key in collective and collective[key] != evidence:
                raise ValueError(f"Conflicting collective evidence: {key}")
            collective[key] = evidence
            group_key = request, event["group"]
            members = event.get("group_ranks")
            if members is not None:
                if (
                    not isinstance(members, list)
                    or rank not in members
                    or len(set(members)) != len(members)
                ):
                    raise ValueError(f"Invalid collective membership: {group_key}")
                if group_members.setdefault(group_key, tuple(members)) != tuple(
                    members
                ):
                    raise ValueError(f"Inconsistent collective membership: {group_key}")
            per_rank = collective_sequences[group_key][rank]
            if per_rank.setdefault(key, evidence) != evidence:
                raise ValueError(f"Inconsistent collective sequence: {key}")
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
                device_busy_us=busy,  # Retained for normalized trace consumers.
                occupied_device_us=busy,
                useful_compute_us=interval_union(
                    (e["start_us"], e["end_us"])
                    for e in step_events
                    if e["kind"] == "compute"
                ),
                hccl_us=interval_union(
                    (e["start_us"], e["end_us"])
                    for e in step_events
                    if e["kind"] == "hccl"
                ),
                wait_us=interval_union(
                    (e["start_us"], e["end_us"])
                    for e in step_events
                    if e["kind"] == "wait"
                ),
                device_uncovered_us=(
                    last_device - min(start for start, _ in device) - busy
                    if device
                    else None
                ),
                host_return_us=last_host,
                device_completion_us=last_device,
                invalid_queue_links=invalid_queue_links,
                collective_calls=list(collective.values()),
            )
        )
    peaks = {}
    for key, spans in request_queues.items():
        endpoints = defaultdict(int)
        for start, end in spans:
            endpoints[start] += 1
            endpoints[end] -= 1
        peak = depth = 0
        for timestamp in sorted(endpoints):
            depth += endpoints[timestamp]
            peak = max(peak, depth)
        peaks[key] = peak
    for row in output:
        row["observed_queue_depth"] = peaks[row["request"], row["rank"]]
    groups = []
    for key, ranks in collective_sequences.items():
        members = group_members.get(key)
        complete = (
            members is not None
            and set(members) == ranks.keys()
            and all((key[0], rank) not in incomplete_hccl for rank in ranks)
            and not unmatched.get("hccl")
        )
        comparable = complete and all(
            seq == next(iter(ranks.values())) for seq in ranks.values()
        )
        groups.append(
            dict(
                request=key[0],
                group=key[1],
                comparable=comparable,
                calls_per_rank=len(next(iter(ranks.values()))) if comparable else None,
            )
        )
    matched = len(events) - sum(unmatched.values())
    return dict(
        steps=output,
        unmatched=dict(unmatched),
        total_events=len(events),
        matched_events=matched,
        correlation_coverage=matched / len(events) if events else None,
        collective_groups=groups,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trace", type=Path, help="Clock-aligned normalized trace JSON array"
    )
    args = parser.parse_args()
    print(json.dumps(analyze(json.loads(args.trace.read_text())), indent=2))


if __name__ == "__main__":
    main()
