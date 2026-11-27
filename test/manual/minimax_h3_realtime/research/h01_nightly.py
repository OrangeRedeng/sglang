"""Recover H01 evidence or run one missing A3 case on an existing server."""

import argparse
import json
from pathlib import Path
from threading import Event
import time
from urllib.request import urlopen

from h3_chunks import Client, generate, plan
from runner import probe_media, save


BASELINE = "h3-B0-L16-merged-154033-20261127.6CbUOM"
SIX_CHUNKS = "h3-C4-L16-6chunks-2runs-154841-20261127.1fppVP"
THREE_CHUNKS = "h3-chunks-C4-L16-merged-152744-20261127.4YqScR"
EVIDENCE = {
    "b0": (f"{BASELINE}/summary.json", f"{BASELINE}/samples.jsonl"),
    "six-1": (f"{SIX_CHUNKS}/sample-1/result.json",),
    "six-2": (f"{SIX_CHUNKS}/sample-2/result.json",),
    "three": (f"{THREE_CHUNKS}/chunks/result.json",),
}


def run_baseline(config, output):
    request = dict(config["request"])
    for field in ("seconds", "fps", "num_frames"):
        request.pop(field, None)
    request.update(
        task="t2va",
        conditions=[],
        num_outputs_per_prompt=1,
        target=dict(request["target"], duration_seconds=15),
    )
    save(output / "request.json", request)
    client = Client(config["endpoints"][0]["url"], config.get("timeout_s", 600))
    for label in ("warmup", "measured"):
        media_path = output / f"{label}.mp4"
        start = time.perf_counter()
        response = client.generate(request, media_path, Event())
        elapsed = time.perf_counter() - start
        media = probe_media(media_path)
        geometry = (media["width"], media["height"], media["fps"], media["frames"])
        if geometry != (1344, 768, 24, 362) or (
            abs(media["audio_duration_s"] - media["duration_s"]) > 0.1
        ):
            raise ValueError(f"Unexpected H3 output geometry: {media}")
        save(
            output / f"{label}.json",
            dict(
                client_e2e_s=elapsed,
                executed_nfe="unverified",
                media=media,
                output_path=str(media_path),
                response=response,
            ),
        )
        print(f"{label}: {elapsed:.2f} s client E2E", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--run", choices=("b0", "six-1"))
    parser.add_argument("--output", type=Path, help="New directory for one missing case")
    args = parser.parse_args()
    if not args.artifacts.is_dir():
        parser.error("The original A3 artifact root must be accessible")

    evidence = {
        case: [args.artifacts / name for name in names]
        for case, names in EVIDENCE.items()
    }
    for case, paths in evidence.items():
        for path in paths:
            status = "found" if path.is_file() else "missing"
            print(f"{case} {status}: {path}", flush=True)
    if args.run is None:
        return
    if any(path.exists() for path in evidence[args.run]):
        parser.error("Review existing evidence before considering any repeat")
    if args.output is None or args.output.exists():
        parser.error("--run requires a new --output directory")

    config_path = args.artifacts / THREE_CHUNKS / "chunks/config.json"
    config = json.loads(config_path.read_text())
    endpoints = config["endpoints"]
    if (
        config["mode"] != "continuation"
        or len(endpoints) != 1
        or len(set(endpoints[0]["devices"])) != 16
        or config["deployment"]["adapter_revision"] != "merged-into-model"
        or config.get("expected_canvas", [1344, 768]) != [1344, 768]
        or config.get("first_frame")
    ):
        parser.error("Expected the saved merged FlashGen 16-NPU continuation config")
    config.pop("continuation_prompts", None)
    config.update(duration_s=15, window_frames=107, overlap_frames=56)
    if len(plan(config)) != 6:
        parser.error("The saved request must produce six native windows")

    with urlopen(endpoints[0]["url"].rstrip("/") + "/health", timeout=10) as health:
        if health.status != 200:
            raise RuntimeError(f"H3 server is not ready: HTTP {health.status}")

    output = args.output.resolve()
    output.mkdir(parents=True)
    save(output / "config.json", config)
    try:
        if args.run == "b0":
            run_baseline(config, output)
        else:
            generate(config, output / "chunks")
    except Exception as error:
        save(output / "failure.json", dict(status="failed", error=str(error)))
        raise


if __name__ == "__main__":
    main()
