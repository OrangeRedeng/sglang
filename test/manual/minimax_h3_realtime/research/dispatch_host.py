# SPDX-License-Identifier: Apache-2.0
"""Bounded Linux host snapshots for a diagnostic server process tree."""

import json
import sys
from pathlib import Path


def read(path):
    try:
        return path.read_text()
    except OSError:
        return None


def snapshot(pid):
    pending = [pid]
    processes = {}
    while pending and len(processes) < 128:
        pid = pending.pop()
        if pid in processes:
            continue
        proc = Path("/proc") / str(pid)
        processes[pid] = {
            name: read(proc / name)
            for name in ("status", "smaps_rollup", "stat", "schedstat", "cgroup")
        }
        for children in (proc / "task").glob("*/children"):
            pending.extend(map(int, (read(children) or "").split()))
    cgroups = {}
    for process in processes.values():
        for line in (process["cgroup"] or "").splitlines():
            if line.startswith("0::"):
                relative = Path(line[3:].lstrip("/"))
                if ".." in relative.parts:
                    continue
                root = Path("/sys/fs/cgroup") / relative
                cgroups[str(relative)] = {
                    key: read(root / key)
                    for key in (
                        "memory.current",
                        "memory.max",
                        "memory.events",
                        "memory.swap.current",
                        "memory.numa_stat",
                        "cpu.stat",
                        "cpu.pressure",
                        "memory.pressure",
                    )
                }
    return dict(
        processes=processes,
        cgroups=cgroups,
        truncated=bool(pending),
        loadavg=read(Path("/proc/loadavg")),
        vmstat=read(Path("/proc/vmstat")),
    )


if __name__ == "__main__":
    print(json.dumps(snapshot(int(sys.argv[1])), indent=2))
