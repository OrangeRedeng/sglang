# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

import h3_chunks as chunks
from checkpoint_contract import model_id


def config(tmp_path, mode="continuation"):
    deployment = dict(family="flashgen4", manifest_sha256="a" * 64)
    result = dict(
        mode=mode,
        duration_s=15,
        deployment=deployment,
        endpoints=[
            dict(
                url="http://127.0.0.1:30010",
                devices=list(range(8)),
                deployment=deployment,
            )
        ],
        request=dict(
            model=model_id(deployment["manifest_sha256"]),
            prompt="Continuous motion",
            num_inference_steps=4,
            flow_shift=12,
            audio_flow_shift=3,
            seed=1101,
            target=dict(short_edge=768, aspect_ratio="16:9"),
        ),
    )
    if mode == "anchor_parallel":
        result.update(anchor_source="user", anchors=[])
        for index in range(4):
            image = tmp_path / f"{index}.png"
            image.write_bytes(b"fixture")
            result["anchors"].append(str(image))
    return result


@pytest.fixture
def model_endpoints(monkeypatch):
    def endpoint_json(self, method, path):
        assert method == "GET"
        if path == "/v1/models":
            return {"data": [{"id": model_id("a" * 64)}]}
        assert path == "/v1/list_loras"
        return {"active": {}, "loaded_adapters": []}

    monkeypatch.setattr(chunks.Client, "json", endpoint_json)


def test_real_request_plan_and_native_minimum(tmp_path):
    cfg = config(tmp_path)
    cfg["request"].update(seconds=5, fps=24, num_frames=124)
    tasks = chunks.plan(cfg)
    assert len(tasks) == 3
    assert tasks[-1].stop == 362
    assert sum(task.stop - task.commit_start for task in tasks) == 362
    assert all(107 <= task.sampled_frames <= 362 for task in tasks)
    for task in tasks:
        request = chunks.payload(
            cfg, task, [chunks.keyframe(tmp_path / "input.png", 0)]
        )
        assert {"seconds", "fps", "num_frames"}.isdisjoint(request)
        assert request["task"] == "fl2va"
        assert request["seed"] == 1101 + task.index
        assert 4 <= request["target"]["duration_seconds"] <= 15


def test_parallel_windows_share_one_discarded_anchor_frame(tmp_path):
    cfg = config(tmp_path, "anchor_parallel")
    tasks = chunks.plan(cfg)
    assert all(not task.dependencies for task in tasks)
    for previous, current in zip(tasks, tasks[1:]):
        assert current.start == previous.stop - 1
        assert current.commit_start - current.start == 1
    cfg["endpoints"].append(dict(cfg["endpoints"][0], url="http://127.0.0.1:30011"))
    with pytest.raises(ValueError, match="disjoint"):
        chunks.plan(cfg)


@pytest.mark.parametrize("devices", [list(range(8, 16)), list(range(16))])
def test_continuation_runs_native_jobs_and_reuses_generated_boundary(
    tmp_path, monkeypatch, devices, model_endpoints
):
    cfg = config(tmp_path)
    cfg["endpoints"][0]["devices"] = devices
    seen = []
    extractions = []

    def generate(self, request, destination, cancelled):
        seen.append(request)
        destination.write_bytes(b"fake-model-output")
        return dict(status="completed", id=str(len(seen)))

    def frame(video, index, image, timeout):
        extractions.append((Path(video).name, index))
        image.write_bytes(b"frame")

    monkeypatch.setattr(chunks.Client, "generate", generate)
    monkeypatch.setattr(chunks, "extract_frame", frame)
    monkeypatch.setattr(chunks.shutil, "which", lambda _: "available")
    monkeypatch.setattr(
        chunks,
        "trim_chunk",
        lambda source, task, destination, timeout: destination.write_bytes(b"trim"),
    )

    def concat(parts, directory, timeout):
        assert len(parts) == 3
        result = directory / "output.mp4"
        result.write_bytes(b"muxed")
        return result

    monkeypatch.setattr(chunks, "concatenate", concat)
    monkeypatch.setattr(
        chunks,
        "probe_media",
        lambda path: dict(
            fps=24,
            frames=362 if path.name == "output.mp4" else 124,
            width=1344,
            height=768,
            duration_s=362 / 24,
            audio_duration_s=362 / 24,
        ),
    )
    result = chunks.generate(cfg, tmp_path / "result")
    assert seen[0]["task"] == "t2va"
    assert [request["task"] for request in seen[1:]] == ["fl2va", "fl2va"]
    assert extractions == [("raw-0000.mp4", 123), ("raw-0001.mp4", 123)]
    assert result["executed_nfe"] == "unverified"
    assert result["media"]["frames"] == 362
    assert [chunk["devices"] for chunk in result["chunks"]] == [devices] * 3
    assert (tmp_path / "result/result.json").is_file()


def test_trim_uses_matching_video_audio_intervals(tmp_path, monkeypatch):
    task = chunks.plan(config(tmp_path))[1]
    commands = []
    monkeypatch.setattr(
        chunks, "run_media", lambda argv, timeout: commands.append(argv)
    )
    chunks.trim_chunk(Path("raw.mp4"), task, Path("trim.mkv"), 10)
    graph = commands[0][commands[0].index("-filter_complex") + 1]
    assert "start_frame=1:end_frame=124" in graph
    assert "atrim=start=0.041666666667:end=5.166666666667" in graph


def test_parallel_jobs_use_two_endpoint_groups_and_shared_anchors(
    tmp_path, monkeypatch, model_endpoints
):
    import time
    from threading import Lock

    cfg = config(tmp_path, "anchor_parallel")
    cfg["endpoints"][0]["devices"] = [0, 1, 2, 3]
    cfg["endpoints"].append(
        dict(cfg["endpoints"][0], url="http://127.0.0.1:30011", devices=[4, 5, 6, 7])
    )
    seen = []
    state = dict(active=0, peak=0)
    lock = Lock()

    def generate(self, request, destination, cancelled):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            seen.append((self.url, request))
        time.sleep(0.03)
        destination.write_bytes(b"fake")
        with lock:
            state["active"] -= 1
        return dict(status="completed", id=destination.stem)

    monkeypatch.setattr(chunks.Client, "generate", generate)
    monkeypatch.setattr(chunks.shutil, "which", lambda _: "available")
    monkeypatch.setattr(
        chunks,
        "trim_chunk",
        lambda source, task, destination, timeout: destination.write_bytes(b"trim"),
    )

    def concat(parts, directory, timeout):
        result = directory / "output.mp4"
        result.write_bytes(b"output")
        return result

    monkeypatch.setattr(chunks, "concatenate", concat)
    monkeypatch.setattr(
        chunks,
        "probe_media",
        lambda path: dict(
            fps=24,
            frames=362 if path.name == "output.mp4" else 124,
            width=1344,
            height=768,
            duration_s=362 / 24,
            audio_duration_s=362 / 24,
        ),
    )
    result = chunks.generate(cfg, tmp_path / "parallel")
    assert state["peak"] == 2
    assert len({url for url, request in seen}) == 2
    ordered = sorted(
        (request for url, request in seen), key=lambda request: request["seed"]
    )
    assert ordered[0]["conditions"][-1]["uri"] == ordered[1]["conditions"][0]["uri"]
    assert [record["window"]["index"] for record in result["chunks"]] == [0, 1, 2]


def test_http_client_submits_polls_and_downloads(tmp_path, monkeypatch):
    import io
    import json
    from threading import Event

    calls = []
    responses = [
        json.dumps(dict(id="chunk", status="queued")).encode(),
        json.dumps(dict(id="chunk", status="completed")).encode(),
        b"av-output",
    ]

    def open_url(request, timeout):
        calls.append(
            request if isinstance(request, str) else (request.method, request.full_url)
        )
        return io.BytesIO(responses.pop(0))

    monkeypatch.setattr(chunks.urllib.request, "urlopen", open_url)
    output = tmp_path / "raw.mp4"
    result = chunks.Client("http://localhost:30010", 2).generate({}, output, Event())
    assert result["status"] == "completed"
    assert output.read_bytes() == b"av-output"
    assert calls == [
        ("POST", "http://localhost:30010/v1/videos"),
        ("GET", "http://localhost:30010/v1/videos/chunk"),
        "http://localhost:30010/v1/videos/chunk/content",
    ]


def test_http_client_cancels_outstanding_job(tmp_path, monkeypatch):
    from threading import Event

    client = chunks.Client("http://localhost", 2)
    calls = []

    def request(method, path, body=None):
        calls.append(method)
        return dict(id="chunk", status="queued")

    monkeypatch.setattr(client, "json", request)
    cancelled = Event()
    cancelled.set()
    with pytest.raises(TimeoutError):
        client.generate({}, tmp_path / "unused.mp4", cancelled)
    assert calls == ["POST", "DELETE"]
