# SPDX-License-Identifier: Apache-2.0
"""Generate one H3 video through native FL2VA HTTP chunk requests."""

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

from prototypes import execute_chunks, windows
from runner import probe_media, save


def plan(config):
    canvas = config.get("expected_canvas", [1344, 768])
    if len(canvas) != 2 or any(
        type(size) is not int or size <= 0 or size % 32 for size in canvas
    ):
        raise ValueError(
            "expected_canvas must contain two positive 32-aligned dimensions"
        )
    mode = config["mode"]
    if mode not in ("continuation", "anchor_parallel"):
        raise ValueError("mode must be continuation or anchor_parallel")
    if not math.isfinite(config["duration_s"]) or config["duration_s"] <= 0:
        raise ValueError("duration_s must be positive and finite")
    total = math.ceil(config["duration_s"] * 24)
    total = max(5, total + (5 - total) % 17)
    window = config.get("window_frames", 124)
    overlap = config.get("overlap_frames", 1)
    if not 107 <= window <= 362:
        raise ValueError("Native H3 requests require 4–15 seconds per window")
    if mode == "continuation" and overlap < 1:
        raise ValueError("Continuation needs at least one overlapping boundary frame")
    if mode == "anchor_parallel" and overlap != 1:
        raise ValueError("Anchor-parallel requires one shared boundary frame")
    tasks = windows(total, window, overlap, sequential=mode == "continuation")
    tasks = [
        replace(task, sampled_frames=max(107, task.sampled_frames)) for task in tasks
    ]
    if config.get("first_frame") and not Path(config["first_frame"]).is_file():
        raise ValueError("first_frame must exist on the shared filesystem")
    deployment = config.get("deployment", {})
    if any(
        not deployment.get(key)
        for key in ("model_revision", "adapter_revision", "decoder_revision")
    ):
        raise ValueError("Declare exact model, adapter, and decoder revisions")
    endpoints = config["endpoints"]
    if any(endpoint.get("deployment") != deployment for endpoint in endpoints):
        raise ValueError("All endpoints must declare the identical deployment")
    groups = [endpoint["devices"] for endpoint in endpoints]
    devices = [device for group in groups for device in group]
    urls = [endpoint["url"].rstrip("/") for endpoint in endpoints]
    if (
        not groups
        or any(not group for group in groups)
        or len(devices) > 8
        or len(set(devices)) != len(devices)
        or len(set(urls)) != len(urls)
    ):
        raise ValueError(
            "Endpoints need unique URLs and disjoint nonempty groups within eight NPUs"
        )
    if any(type(device) is not int or device < 0 for device in devices):
        raise ValueError("Device IDs must be nonnegative integers")
    if mode == "anchor_parallel":
        anchors = config["anchors"]
        if len(anchors) != len(tasks) + 1:
            raise ValueError("Provide one shared boundary anchor per chunk boundary")
        if any(not Path(path).is_file() for path in anchors):
            raise ValueError("Every anchor must be an existing shared-filesystem image")
        if config.get("anchor_source") not in ("user", "oracle", "timed_draft"):
            raise ValueError("Declare anchor_source as user, oracle, or timed_draft")
        if config["anchor_source"] == "timed_draft":
            raise ValueError(
                "Generate draft anchors inside the timed system; precomputed drafts are oracle inputs"
            )
    request = config["request"]
    if request.get("conditions") or request.get("num_outputs_per_prompt", 1) != 1:
        raise ValueError(
            "The orchestrator owns conditions and requires one output per request"
        )
    if request.get("num_inference_steps") != 4 or type(request.get("seed")) is not int:
        raise ValueError("Supply the existing four-step request with an integer seed")
    if any(key not in request for key in ("model", "prompt", "target")):
        raise ValueError("request requires model, prompt, and target")
    prompts = config.get("continuation_prompts")
    if prompts is not None and (
        len(prompts) != len(tasks) or any(not p for p in prompts)
    ):
        raise ValueError(
            "continuation_prompts must contain one nonempty prompt per chunk"
        )
    return tasks


def run_media(argv, timeout):
    subprocess.run(
        argv,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def extract_frame(video, index, image, timeout):
    run_media(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select=eq(n\\,{index})",
            "-frames:v",
            "1",
            "-y",
            str(image),
        ],
        timeout,
    )
    if not image.is_file():
        raise RuntimeError("Boundary frame extraction produced no image")


def payload(config, task, conditions):
    result = dict(config["request"])
    for field in ("seconds", "fps", "num_frames"):
        result.pop(field, None)
    target = dict(result["target"])
    # Ask for the native frame bin without exceeding the public 15-second cap.
    duration = min(15, max(4, (task.sampled_frames - 1) / 24))
    target.update(duration_seconds=duration)
    result.update(
        target=target,
        task="fl2va" if conditions else "t2va",
        conditions=conditions,
        num_outputs_per_prompt=1,
        seed=result["seed"] + task.index,
    )
    if config.get("continuation_prompts"):
        result["prompt"] = config["continuation_prompts"][task.index]
    return result


def keyframe(path, index):
    return dict(
        type="image",
        role="keyframe",
        frame_index=index,
        uri=Path(path).resolve().as_uri(),
    )


class Client:
    def __init__(self, url, timeout):
        self.url, self.timeout = url.rstrip("/"), timeout

    def json(self, method, path, body=None):
        request = urllib.request.Request(
            self.url + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request, timeout=min(30, self.timeout)
            ) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            with error:
                detail = error.read(4096).decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"{method} {path} returned HTTP {error.code}: {detail or error.reason}"
            ) from error

    def generate(self, request, destination, cancelled):
        job = self.json("POST", "/v1/videos", request)
        path = "/v1/videos/" + urllib.parse.quote(job["id"], safe="")
        deadline = time.monotonic() + self.timeout
        try:
            while job["status"] != "completed":
                if job["status"] in ("failed", "cancelled"):
                    raise RuntimeError(
                        f"H3 generation failed: {job.get('error', job['status'])}"
                    )
                if cancelled.wait(0.1) or time.monotonic() > deadline:
                    raise TimeoutError("Chunk generation cancelled or timed out")
                job = self.json("GET", path)
            with urllib.request.urlopen(
                self.url + path + "/content", timeout=self.timeout
            ) as response:
                with destination.open("wb") as output:
                    shutil.copyfileobj(response, output)
            return job
        except BaseException:
            try:
                self.json("DELETE", path)
            except Exception:
                pass
            raise


def trim_chunk(source, task, destination, timeout):
    first = task.commit_start - task.start
    count = task.stop - task.commit_start
    stop = first + count
    # Keep RGB frame indices and audio timestamps on the same committed interval.
    graph = (
        f"[0:v]trim=start_frame={first}:end_frame={stop},setpts=PTS-STARTPTS[v];"
        f"[0:a]atrim=start={first / 24:.12f}:end={stop / 24:.12f},asetpts=PTS-STARTPTS[a]"
    )
    run_media(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(source),
            "-filter_complex",
            graph,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "ffv1",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(destination),
        ],
        timeout,
    )


def concatenate(parts, directory, timeout):
    # Generated fixed filenames contain no quoting-sensitive characters.
    listing = directory / "concat.txt"
    listing.write_text("".join(f"file '{part.name}'\n" for part in parts))
    destination = directory / "output.mp4"
    run_media(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "1",
            "-i",
            str(listing),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "32000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            "-y",
            str(destination),
        ],
        timeout,
    )
    return destination


def generate(config, directory):
    tasks = plan(config)
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe are required")
    directory.mkdir(parents=True, exist_ok=False)
    directory = directory.resolve()
    save(directory / "config.json", config)
    save(directory / "windows.json", [asdict(task) for task in tasks])
    timeout = config.get("timeout_s", 600)
    by_group = {
        tuple(endpoint["devices"]): endpoint for endpoint in config["endpoints"]
    }
    started = time.perf_counter()

    def adapter(task, group, dependencies, cancelled):
        conditions = []
        if config["mode"] == "continuation" and task.index:
            previous = dependencies[task.index - 1]
            previous_task = tasks[task.index - 1]
            image = directory / f"boundary-{task.index:04d}.png"
            extract_frame(
                previous["raw"], task.start - previous_task.start, image, timeout
            )
            conditions = [keyframe(image, 0)]
        elif config["mode"] == "continuation" and config.get("first_frame"):
            conditions = [keyframe(config["first_frame"], 0)]
        elif config["mode"] == "anchor_parallel":
            conditions = [keyframe(config["anchors"][task.index], 0)]
            # A padded final request is cropped; its generated final frame is not
            # the requested timeline endpoint, so do not impose a misplaced anchor.
            if task.sampled_frames == task.stop - task.start:
                conditions.append(keyframe(config["anchors"][task.index + 1], -1))
        request = payload(config, task, conditions)
        save(directory / f"request-{task.index:04d}.json", request)
        raw = directory / f"raw-{task.index:04d}.mp4"
        begin = time.perf_counter()
        response = Client(by_group[group]["url"], timeout).generate(
            request, raw, cancelled
        )
        observed = probe_media(raw)
        if [observed["width"], observed["height"]] != config.get(
            "expected_canvas", [1344, 768]
        ):
            raise ValueError("Conditioning changed the requested output canvas")
        if observed["fps"] != 24 or observed["frames"] != task.sampled_frames:
            raise ValueError("Server output violates requested native frame geometry")
        committed = directory / f"committed-{task.index:04d}.mkv"
        trim_chunk(raw, task, committed, timeout)
        record = dict(
            window=asdict(task),
            devices=list(group),
            response=response,
            media=observed,
            elapsed_s=time.perf_counter() - begin,
            raw=str(raw),
            committed=str(committed),
            seed=request["seed"],
        )
        save(directory / f"chunk-{task.index:04d}.json", record)
        return record

    records = execute_chunks(
        tasks, [endpoint["devices"] for endpoint in config["endpoints"]], adapter
    )
    output = concatenate(
        [Path(record["committed"]) for record in records], directory, timeout
    )
    observed = probe_media(output)
    if observed["frames"] != tasks[-1].stop or observed["fps"] != 24:
        raise ValueError("Muxed output lost or duplicated committed frames")
    if abs(observed["duration_s"] - observed["audio_duration_s"]) > 0.1:
        raise ValueError("Muxed AV duration mismatch")
    canvases = {
        (record["media"]["width"], record["media"]["height"]) for record in records
    }
    if len(canvases) != 1 or (observed["width"], observed["height"]) not in canvases:
        raise ValueError("Chunk canvases differ")
    elapsed = time.perf_counter() - started
    result = dict(
        status="completed",
        mode=config["mode"],
        output_path=str(output),
        client_e2e_s=elapsed,
        rtf=elapsed / observed["duration_s"],
        media=observed,
        requested_steps_per_chunk=4,
        executed_nfe="unverified",
        chunks=records,
        classification="oracle"
        if config.get("anchor_source") == "oracle"
        else "approximate",
        limitations=[
            "RGB endpoint conditioning, no latent-tail or global RoPE reuse",
            "Per-window seed offsets, not global-coordinate noise",
            "Generated audio is trimmed/concatenated; continuity and lip-sync require review",
            "Device group assignments are declarations for externally managed servers",
        ],
    )
    save(directory / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.plan_only:
        print(json.dumps([asdict(task) for task in plan(config)], indent=2))
    else:
        existed = args.output.exists()
        try:
            print(json.dumps(generate(config, args.output), indent=2))
        except Exception as error:
            if not existed and args.output.is_dir():
                save(
                    args.output / "failure.json",
                    dict(status="failed", error=str(error)),
                )
            raise


if __name__ == "__main__":
    main()
