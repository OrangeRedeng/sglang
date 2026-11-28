# R04 dispatch and request workspace

Base is R01 at 68e373c50c47492e5860e2d18a9599133ea9c754, separate from R02 and R03. The supplied H03 summary shows late-step host enqueue growth. The original trace files, actual old launch and device access were unavailable. This leaves TaskQueue, allocator, host scheduling, HCCL dependencies and device effects open. [Prepared QKV workspace work in #49827](https://github.com/sgl-project/sglang/pull/49827) targets NVIDIA Sage exchange. R04 reuses only the existing eager AV fusion destination on this branch.

## Run

Use R01's verified merged checkpoint and decoder. Keep devices, prompt, seed, topology and source fixed. D0 runs B0 and W1 unprofiled. D1 repeats B0 with H3_PROFILE=1, then runs the same profiled case with H3_DISPATCH_SYNC=1. It collects per-rank step allocator counters, device snapshots, and before/after process-tree PSS, faults, affinity, scheduler and cgroup data. The synchronized diagnostic has a different execution schedule and stays outside latency comparisons.

D2 repeats W1 unprofiled with H3_WINDOW_WORKSPACE=1. The workspace has two FP32 fusion buffers, retains only the active shape, and belongs to one eager request and one current stream. The scheduler consumes or overwrites velocity on that stream before the next fusion. The workspace does not own or alias frozen inputs and leaves collective, QKV and graph storage alone. The request owns the reference, including on exceptions and cancellation. Device allocator lifetime and exact AV parity need a matched run. Compare decoded video and PCM hashes, counter evidence, high water, p50/p90 and raw slow episodes across at least 10 repeats. The controller registers R04.D0.B0, R04.D0.W1, R04.D1.TRACE, R04.D1.SYNC and R04.D2.

## Trace input

Export a clock-aligned JSON array with rank, event_id, kind, resource, start_us and end_us. Each host producer carries request and logical step. Link enqueue, runtime dispatch, compute, waits and HCCL events through producer event IDs derived from the profiler correlation identifiers. Retain original sequence IDs, stream/task IDs and the mapping alongside the normalized input. Clock alignment and SDK-specific extraction are operator prerequisites. Run dispatch_trace.py on the normalized array. The analyzer reports unbound events separately, unions overlapping intervals per resource and across device activity, separates host return from device completion, and reports linked queue depth. HCCL counts require verified group, sequence, callsite and payload_bytes. Validate the conversion on a small event-timed target trace before applying it to the archived night trace.

D3 awaits the installed torch_npu feature and documentation audit. D4 awaits independent evidence for each candidate. No allocator or HCCL root cause is established.

Three focused CPU tests check correlation across step boundaries, interval unions, bitwise fusion, channel-major audio, shape replacement and request isolation. Ascend parity, memory and performance remain pending.

### Phase 2.1 trace input and interpretation

Provide a normalized JSON array on one aligned microsecond clock per rank. Every event requires integer rank, rank-unique event_id, kind (host/enqueue/dispatch/compute/hccl/wait), resource, start_us and end_us. A root host event requires request and step. Other events require a rank-local producer event_id. A queue interval pairs enqueue end with direct dispatch start. Each step reports its request/rank peak over all observed pairs. Missing events reduce correlation coverage. Device occupation is the union of compute, wait and HCCL. These three unions may overlap. Host return and last observed device completion are reported independently.

HCCL records require group, sequence, callsite and payload_bytes. Include ordered group_ranks for membership-validated comparisons. Missing or inconsistent participants or sequences suppress comparable counts. No raw hardware trace was supplied. The existing fusion workspace stays request-local. Gather, padding and scheduler buffers may remain live across asynchronous consumers. Further aliasing requires device lifetime evidence. Step-3 causality remains open.
