# SPDX-License-Identifier: Apache-2.0
import json
import sys
import time
from threading import Lock

import pytest
import torch

import prototypes as proto
import runner
from test_checkpoint_contract import make_checkpoint


@pytest.mark.parametrize("overlap", [0, 22, 56])
def test_window_commits_cover_video_once(overlap):
    tasks = proto.windows(362, 124, overlap)
    assert tasks[0].start == tasks[0].commit_start == 0
    assert tasks[-1].stop == 362
    assert all((task.sampled_frames - 5) % 17 == 0 for task in tasks)
    assert sum(task.stop - task.commit_start for task in tasks) == 362
    for previous, current in zip(tasks, tasks[1:]):
        assert current.commit_start == previous.stop
        assert current.dependencies == (previous.index,)


def test_chunk_scheduler_respects_groups_dependencies_retry_and_order():
    tasks = proto.windows(362, 124, 22, sequential=False)
    lock = Lock()
    active, attempts = set(), {}

    def adapter(task, devices, dependencies, cancelled):
        with lock:
            assert not active.intersection(devices)
            active.update(devices)
            attempts[task.index] = attempts.get(task.index, 0) + 1
        try:
            time.sleep(0.005)
            if task.index == 1 and attempts[1] == 1:
                raise RuntimeError("transient")
            return task.index
        finally:
            with lock:
                active.difference_update(devices)

    assert proto.execute_chunks(tasks, [[0, 1], [2, 3]], adapter, retries=1) == list(
        range(len(tasks))
    )
    assert attempts[1] == 2
    with pytest.raises(ValueError, match="disjoint"):
        proto.execute_chunks(tasks, [[0], [0]], adapter)


def test_stream_backpressure_order_and_failure_propagation():
    committed = []
    proto.stream_decode(
        range(8),
        lambda i: i * 2,
        lambda i, x: x + 1,
        lambda i, x: committed.append((i, x)),
        capacity=1,
    )
    assert committed == [(i, 2 * i + 1) for i in range(8)]

    def fail(_):
        raise RuntimeError("producer failed")

    with pytest.raises(RuntimeError, match="producer failed"):
        proto.stream_decode([0], fail, lambda i, x: x, lambda i, x: None)


def test_kv_budget_bounds_router_and_remote_union():
    plan = proto.bounded_kv_plan(
        2,
        blocks=100000,
        local_radius=1,
        mandatory=[9],
        candidates=[(0.9, 7), (0.8, 8)],
        budget=5,
    )
    assert plan == (1, 2, 3, 7, 9)
    with pytest.raises(ValueError, match="Mandatory"):
        proto.bounded_kv_plan(
            2, blocks=20, local_radius=2, mandatory=[9], candidates=[], budget=3
        )
    with pytest.raises(ValueError, match="router"):
        proto.bounded_kv_plan(
            0,
            blocks=20,
            local_radius=0,
            mandatory=[],
            candidates=[(1, 1)] * 9,
            budget=2,
        )
    assert proto.sparse_demands([[0, 1, 1], [1, 2]], {0: 0, 1: 1, 2: 1}, 0) == {
        1: (1, 2)
    }


def test_softmax_partition_merge_and_empty_branches():
    torch.manual_seed(4)
    logits, values = torch.randn(3, 7), torch.randn(7, 4)
    left, right = logits[:, :3], logits[:, 3:]
    expected = logits.softmax(-1) @ values
    actual = proto.merge_softmax_branches(
        left.softmax(-1) @ values[:3],
        left.logsumexp(-1),
        right.softmax(-1) @ values[3:],
        right.logsumexp(-1),
    )
    torch.testing.assert_close(actual, expected)
    empty = torch.full((3,), -torch.inf)
    zero = torch.zeros(3, 4)
    assert torch.equal(proto.merge_softmax_branches(zero, empty, zero, empty), zero)


def test_reference_pruning_preserves_positions_and_nonreferences():
    hidden, positions = (
        torch.arange(20).reshape(5, 4),
        torch.tensor([8, 12, 19, 20, 25]),
    )
    out, pos, ids = proto.compact_reference_rows(
        hidden,
        positions,
        torch.tensor([True, True, False, False, True]),
        torch.tensor([True, False, False, False, False]),
    )
    assert ids.tolist() == [0, 2, 3]
    assert pos.tolist() == [8, 19, 20]
    assert torch.equal(out, hidden[ids])
    assert proto.temporal_grid(15, 24)["frames"] == 362


def make_suite(tmp_path):
    adapter = tmp_path / "adapter.py"
    adapter.write_text("""import json, os
from pathlib import Path
context=json.loads(Path(os.environ['H3_RESEARCH_CONTEXT']).read_text())
result=Path(os.environ['H3_RESEARCH_RESULT'])
media=result.with_suffix('.mp4'); media.write_bytes(b'host-fixture')
result.write_text(json.dumps(dict(status='completed',actual_nfe=4,
contract_hash=context['contract_hash'],workload_hash=context['workload_hash'],output_path=str(media),
checkpoint_manifest_sha256=context['contract']['checkpoint_manifest_sha256'])))
""")
    manifest, receipt, identity = make_checkpoint(tmp_path)
    suite = runner.template()
    suite["contract"].update(
        hardware="mock-A5",
        checkpoint_manifest=str(manifest),
        checkpoint_receipt=str(receipt),
        checkpoint_manifest_sha256=identity,
        software={"test": True},
    )
    suite.update(warmups=1, repeats=2)
    suite["cases"] = [
        dict(
            id="baseline",
            experiment="B00",
            duration_s=15,
            workload=dict(prompt="host-fixture", seed=1),
            command=[sys.executable, str(adapter)],
        )
    ]
    return suite


def test_campaign_rounds_resume_and_config_drift(tmp_path, monkeypatch):
    suite = make_suite(tmp_path)
    path = tmp_path / "suite.json"
    output = tmp_path / "results"
    path.write_text(json.dumps(suite))
    monkeypatch.setattr(
        runner, "source_state", lambda *args: {"commit": "host-fixture"}
    )
    monkeypatch.setattr(
        runner,
        "probe_media",
        lambda path: dict(
            width=1344,
            height=768,
            fps=24,
            frames=362,
            duration_s=362 / 24,
            audio_duration_s=362 / 24,
        ),
    )
    args = ["runner.py", "run", str(path), "--output", str(output), "--rounds", "2"]
    monkeypatch.setattr(sys, "argv", args)
    runner.main()
    records = [
        json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2 and all(r["status"] == "measured" for r in records)
    assert all(len(r["result"]["samples"]) == 2 for r in records)
    assert all(r["result"]["quality"] == "pending" for r in records)
    monkeypatch.setattr(sys, "argv", args + ["--resume"])
    runner.main()
    assert len((output / "results.jsonl").read_text().splitlines()) == 2
    suite["contract"]["checkpoint_manifest_sha256"] = "f" * 64
    path.write_text(json.dumps(suite))
    with pytest.raises(ValueError, match="changed"):
        runner.main()


def test_capability_gate_and_workload_mismatch(tmp_path):
    suite = make_suite(tmp_path)
    case = dict(suite["cases"][0], experiment="B05")
    assert len(runner.evidence_errors(suite, case)) == 2
    result = tmp_path / "sample.json"
    result.write_text(
        json.dumps(
            dict(
                status="completed",
                actual_nfe=4,
                contract_hash=runner.digest(suite["contract"]),
                workload_hash="wrong",
                checkpoint_manifest_sha256=suite["contract"][
                    "checkpoint_manifest_sha256"
                ],
            )
        )
    )
    with pytest.raises(ValueError, match="workload"):
        runner.verify_sample(result, suite["contract"], case)


def test_capability_proof_is_bound_to_device_and_software(tmp_path):
    suite = make_suite(tmp_path)
    case = dict(suite["cases"][0], experiment="B04")
    proof = tmp_path / "proof.json"
    proof.write_text(
        json.dumps(
            dict(
                result="pass",
                device="mock-A5",
                software_hash=runner.digest(suite["contract"]["software"]),
            )
        )
    )
    suite["capabilities"]["external_indices"] = dict(
        status="available", device="mock-A5", evidence_path=str(proof)
    )
    assert not runner.evidence_errors(suite, case)
    suite["contract"]["software"]["test"] = False
    assert "stale" in runner.evidence_errors(suite, case)[0]


def test_media_probe_rejects_silent_output(monkeypatch):
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda *args, **kwargs: json.dumps(
            dict(streams=[dict(codec_type="video")], format=dict(duration="15"))
        ).encode(),
    )
    with pytest.raises(ValueError, match="both video and audio"):
        runner.probe_media("unused.mp4")


def test_empty_softmax_branch_ignores_undefined_output():
    exact = torch.full((2, 3), float("nan"))
    representative = torch.ones(2, 3)
    merged = proto.merge_softmax_branches(
        exact, torch.full((2,), -torch.inf), representative, torch.zeros(2)
    )
    assert torch.equal(merged, representative)


def test_exact_repack_evidence_rejects_measured_tensor_difference(
    tmp_path, monkeypatch
):
    runs = [tmp_path / name for name in ("B0", "W0R")]
    for run in runs:
        run.mkdir()
        runner.save(run / "checkpoint.json", dict(family="flashgen4"))
        runner.save(run / "comparison.json", dict(placement="0:TP1", request={}))
        sample = dict(case=run.name, warmup=False, response=dict(id=run.name))
        (run / "samples.jsonl").write_text(json.dumps(sample) + "\n")
        runner.save(
            run / "denoise_matches.json",
            {run.name: "scheduler_updates=4 dit_evaluations=4"},
        )
        lines = [
            f'H3 window parity request={run.name} rank=0 point={point} step={step} tensors={{"value":"same"}}'
            for point, steps in (
                ("initial", [-1]),
                ("input", range(4)),
                ("prediction", range(4)),
                ("update", range(4)),
            )
            for step in steps
        ]
        (run / "server.log").write_text("\n".join(lines))
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda *_, **__: "SHA256=same"
    )
    assert runner.verify_parity(*runs)["tensors"] == 13
    candidate_log = runs[1] / "server.log"
    candidate_log.write_text(
        candidate_log.read_text().replace('"same"', '"different"', 1)
    )
    with pytest.raises(ValueError, match="Tensor records"):
        runner.verify_parity(*runs)
