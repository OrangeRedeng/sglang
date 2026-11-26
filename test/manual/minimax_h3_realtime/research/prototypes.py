# SPDX-License-Identifier: Apache-2.0
"""Host-testable research primitives, not validated H3 execution policies."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import math
from queue import Queue
from threading import Event, Thread


@dataclass(frozen=True)
class Window:
    index: int
    start: int
    stop: int
    commit_start: int
    sampled_frames: int
    dependencies: tuple[int, ...]


def windows(total_frames, window_frames, overlap_frames, *, sequential=True):
    """Native 17n+5 sample windows and disjoint committed RGB intervals."""
    if total_frames < 1 or window_frames < 5 or (window_frames - 5) % 17:
        raise ValueError("Window size must follow native 17n+5 geometry")
    if not 0 <= overlap_frames < window_frames:
        raise ValueError("Overlap must be smaller than the window")
    result = []
    start, committed = 0, 0
    while committed < total_frames:
        stop = min(start + window_frames, total_frames)
        sampled = max(5, stop - start)
        sampled += (5 - sampled) % 17
        index = len(result)
        result.append(
            Window(
                index,
                start,
                stop,
                committed,
                sampled,
                (index - 1,) if sequential and index else (),
            )
        )
        committed = stop
        start = stop - overlap_frames
    return result


def execute_chunks(tasks, groups, adapter, *, budget=8, retries=0):
    """Run one video's dependency DAG on disjoint device groups, return in order.

    adapter(task, device_ids, dependency_outputs, cancel_event) must enforce native
    conditioning, AV timing, its own timeout, and device buffer ownership.
    """
    devices = [device for group in groups for device in group]
    if (
        not groups
        or any(not group for group in groups)
        or len(devices) > budget
        or len(set(devices)) != len(devices)
    ):
        raise ValueError("Device groups must be nonempty, disjoint, within budget")
    ids = [task.index for task in tasks]
    if ids != list(range(len(tasks))) or retries < 0:
        raise ValueError(
            "Tasks must have ordered consecutive IDs and nonnegative retries"
        )
    for task in tasks:
        if any(dep >= task.index or dep < 0 for dep in task.dependencies):
            raise ValueError("Dependencies must refer to earlier tasks")
    pending = list(tasks)
    results, attempts, running = {}, {}, {}
    available = list(groups)
    cancelled = Event()
    with ThreadPoolExecutor(max_workers=len(groups)) as executor:
        try:
            while pending or running:
                for task in list(pending):
                    if not available:
                        break
                    if all(dep in results for dep in task.dependencies):
                        group = available.pop(0)
                        pending.remove(task)
                        future = executor.submit(
                            adapter,
                            task,
                            tuple(group),
                            {dep: results[dep] for dep in task.dependencies},
                            cancelled,
                        )
                        running[future] = (task, group)
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in finished:
                    task, group = running.pop(future)
                    available.append(group)
                    try:
                        results[task.index] = future.result()
                    except Exception:
                        attempts[task.index] = attempts.get(task.index, 0) + 1
                        if attempts[task.index] > retries:
                            raise
                        pending.insert(0, task)
        except BaseException:
            cancelled.set()
            for future in running:
                future.cancel()
            raise
    return [results[index] for index in ids]


def stream_decode(tasks, generate, decode, emit, *, capacity=2):
    """Overlap one producer with ordered decoding and synchronous output backpressure.

    Callbacks must own stream/event synchronization and native VAE halo state.
    This helper does not assert that GPU work overlaps or AV seams are coherent.
    """
    if capacity < 1:
        raise ValueError("Positive capacity required")
    queue = Queue(maxsize=capacity)
    cancelled, done = Event(), Event()
    errors = []

    def produce():
        from queue import Full

        try:
            for task in tasks:
                if cancelled.is_set():
                    break
                latent = generate(task)
                while not cancelled.is_set():
                    try:
                        queue.put((task, latent), timeout=0.05)
                        break
                    except Full:
                        pass
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    thread = Thread(target=produce, daemon=True)
    thread.start()
    try:
        from queue import Empty

        while not done.is_set() or not queue.empty():
            try:
                task, latent = queue.get(timeout=0.05)
            except Empty:
                continue
            emit(task, decode(task, latent))
        if errors:
            raise errors[0]
    finally:
        cancelled.set()
        thread.join()


def bounded_kv_plan(
    query_block, *, blocks, local_radius, mandatory, candidates, budget
):
    """Bound selector work by accepting only an already bounded candidate list.

    candidates: (importance, block_id) pairs supplied without a full B×B router.
    Mandatory sets must fit the same absolute budget; no silent overflow.
    """
    if not 0 <= query_block < blocks or budget < 1 or local_radius < 0:
        raise ValueError("Invalid attention geometry or budget")
    required = set(mandatory) | set(
        range(
            max(0, query_block - local_radius),
            min(blocks, query_block + local_radius + 1),
        )
    )
    if len(candidates) > budget * 4:
        raise ValueError("Candidate list exceeds bounded router budget")
    if any(not 0 <= block < blocks for block in required) or len(required) > budget:
        raise ValueError("Mandatory dependencies exceed budget or valid range")
    for score, block in sorted(candidates, key=lambda item: (-item[0], item[1])):
        if not math.isfinite(score) or not 0 <= block < blocks:
            raise ValueError("Invalid routing candidate")
        if len(required) < budget:
            required.add(block)
    return tuple(sorted(required))


def sparse_demands(plans, owners, rank):
    """Remote KV demand union, deduplicated per source; no HCCL transport assumed."""
    result = {}
    for plan in plans:
        for block in plan:
            owner = owners[block]
            if owner != rank:
                result.setdefault(owner, set()).add(block)
    return {owner: tuple(sorted(blocks)) for owner, blocks in sorted(result.items())}


def temporal_grid(duration_s, fps):
    """Candidate native-aligned geometry; model compatibility is a separate probe."""
    if (
        not math.isfinite(duration_s)
        or not math.isfinite(fps)
        or duration_s <= 0
        or fps <= 0
    ):
        raise ValueError("Positive finite duration and FPS required")
    frames = max(5, math.ceil(duration_s * fps))
    frames += (5 - frames) % 17
    return dict(
        frames=frames,
        latent_frames=((frames - 5) // 17) * 5 + 2,
        fps=fps,
        actual_duration_s=frames / fps,
        timestamps=[index / fps for index in range(frames)],
    )


def compact_reference_rows(hidden, positions, reference_mask, keep_reference):
    """Prune only reference rows; preserve native positions and return inverse IDs."""
    import torch

    if reference_mask.dtype != torch.bool or reference_mask.shape != (hidden.shape[0],):
        raise ValueError("reference_mask must describe every input row")
    if (
        keep_reference.dtype != torch.bool
        or keep_reference.shape != reference_mask.shape
    ):
        raise ValueError("keep_reference must be a full boolean row mask")
    ids = torch.where(~reference_mask | keep_reference)[0]
    return hidden.index_select(0, ids), positions.index_select(0, ids), ids


def merge_softmax_branches(exact, exact_lse, representative, representative_lse):
    """Merge normalized disjoint branches with natural-log LSE, in FP32."""
    import torch

    left, right = exact_lse.float(), representative_lse.float()
    total = torch.logaddexp(left, right)
    valid = torch.isfinite(total)
    safe_total = torch.where(valid, total, torch.zeros_like(total))
    left_weight = torch.where(valid, (left - safe_total).exp(), 0)
    right_weight = torch.where(valid, (right - safe_total).exp(), 0)
    exact = torch.where(torch.isfinite(left).unsqueeze(-1), exact.float(), 0)
    representative = torch.where(
        torch.isfinite(right).unsqueeze(-1), representative.float(), 0
    )
    return exact * left_weight.unsqueeze(-1) + representative * right_weight.unsqueeze(
        -1
    )
