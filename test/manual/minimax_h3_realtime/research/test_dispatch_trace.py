# SPDX-License-Identifier: Apache-2.0
import ast
from pathlib import Path

import torch

from dispatch_trace import analyze


def test_cross_step_device_events_use_producers_and_unions():
    def event(i, kind, start, end, **extra):
        return dict(
            rank=0,
            event_id=i,
            kind=kind,
            start_us=start,
            end_us=end,
            resource="stream",
            **extra,
        )

    report = analyze(
        [
            event(1, "host", 0, 5, step=0, request="r"),
            event(2, "enqueue", 1, 2, producer=1),
            event(3, "dispatch", 6, 7, producer=2),
            event(4, "compute", 8, 16, producer=3),
            event(5, "wait", 10, 18, producer=3),
            event(6, "host", 6, 8, step=1, request="r"),
            event(7, "hccl", 18, 20, producer=100),
        ]
    )
    first, second = report["steps"]
    assert (first["device_busy_us"], first["device_uncovered_us"]) == (10, 0)
    assert (first["host_return_us"], first["device_completion_us"]) == (5, 18)
    assert first["observed_queue_depth"] == 1
    assert second["device_completion_us"] is None
    assert report["unmatched"] == {"hccl": 1}


def test_workspace_reuse_after_scheduler_mutation_and_resize():
    source = next(
        (Path(__file__).resolve().parents[4] / "python").rglob(
            "minimax_h3/windowing.py"
        )
    )
    nodes = [
        node
        for node in ast.parse(source.read_text()).body
        if getattr(node, "name", None) in {"WindowWorkspace", "_blend"}
    ]
    ns = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), ns)
    workspace = ns["WindowWorkspace"]()
    blend = ns["_blend"]
    v, a = torch.zeros(5, 3), torch.zeros(10, 3)
    x, y = workspace.outputs(v, a)
    left, right = torch.randn(3, 3).bfloat16(), torch.randn(3, 3).bfloat16()
    ramp = torch.tensor([0.5])
    expected = blend(left, right, ramp)
    for _ in range(2):
        same, audio = workspace.outputs(v, a)
        assert same is x and audio is y
        blend(left, right, ramp, out=same)
        torch.testing.assert_close(same, expected, rtol=0, atol=0)
        same.zero_()
    assert workspace.allocations == 2
    assert workspace.outputs(v[:4], a)[0].shape == (4, 3)
    assert workspace.allocations == 4
    assert ns["WindowWorkspace"]().outputs(v, a)[0] is not x


def test_channel_major_audio_workspace_matches_existing_fusion():
    source = next(
        (Path(__file__).resolve().parents[4] / "python").rglob(
            "minimax_h3/windowing.py"
        )
    )
    function = next(
        node
        for node in ast.parse(source.read_text()).body
        if getattr(node, "name", None) == "_blend"
    )
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), ns)
    a, b = torch.randn(3, 2, 4), torch.randn(4, 2, 4)
    output = torch.empty(2, 5, 4)
    ramp = torch.tensor([0.25, 0.75])
    ns["_blend"](a, b, ramp, out=output.transpose(0, 1))
    torch.testing.assert_close(
        output, ns["_blend"](a, b, ramp).transpose(0, 1), rtol=0, atol=0
    )
