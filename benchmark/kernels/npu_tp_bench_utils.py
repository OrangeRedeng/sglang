"""Timing helpers for captured A5 TP boundaries."""

import math
import statistics
import time

import torch


def timing(run, warmup, iterations):
    for _ in range(warmup):
        run()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    device_ms, enqueue_ms = [], []
    for _ in range(iterations):
        start.record()
        before = time.perf_counter()
        output = run()
        enqueue_ms.append((time.perf_counter() - before) * 1000)
        end.record()
        end.synchronize()
        device_ms.append(start.elapsed_time(end))
        del output

    # Count allocator requests separately so querying stats does not perturb enqueue.
    key = "allocation.all.allocated"
    before = torch.npu.memory_stats().get(key)
    for _ in range(iterations):
        output = run()
        del output
    torch.npu.synchronize()
    after = torch.npu.memory_stats().get(key)
    return {
        "p50_ms": statistics.median(device_ms),
        "p95_ms": sorted(device_ms)[math.ceil(iterations * 0.95) - 1],
        "mean_ms": statistics.mean(device_ms),
        "host_enqueue_p50_ms": statistics.median(enqueue_ms),
        "allocator_requests_per_iteration": (
            None if before is None or after is None else (after - before) / iterations
        ),
    }
