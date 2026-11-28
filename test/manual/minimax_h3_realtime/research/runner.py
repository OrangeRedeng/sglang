# SPDX-License-Identifier: Apache-2.0
"""Resumable fixed-contract research campaigns. Commands are argv arrays."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import statistics
import subprocess
import sys
import time

from catalog import CATALOG
from checkpoint_contract import SCHEDULES
from checkpoint_contract import digest as checkpoint_digest
from checkpoint_contract import (
    is_sha,
    load_verified,
    model_id,
    reject_runtime_lora,
    require_launcher,
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def command(argv, *, env, log, timeout):
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(x, str) for x in argv)
    ):
        raise ValueError("Commands must be nonempty arrays of strings")
    reject_runtime_lora(argv, env)
    with log.open("w") as output:
        process = subprocess.Popen(
            argv,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise RuntimeError(f"Command exited {code}; see {log}")
        finally:
            # Children may survive an exited launcher. The session belongs to this run.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def source_state(root, output):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])

    tracked = git("diff", "HEAD", "--binary")
    untracked = git("ls-files", "--others", "--exclude-standard").decode().splitlines()
    files = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in untracked
        if (root / name).is_file()
        and not (root / name).resolve().is_relative_to(output.resolve())
    }
    return dict(
        commit=git("rev-parse", "HEAD").decode().strip(),
        dirty_sha256=hashlib.sha256(tracked).hexdigest(),
        untracked=files,
    )


def evidence_errors(suite, case):
    errors = []
    if not case.get("command"):
        errors.append("No model experiment adapter command configured")
    for capability in set(
        CATALOG[case["experiment"]]["requires"] + case.get("required_capabilities", [])
    ):
        evidence = suite.get("capabilities", {}).get(capability, {})
        path = Path(evidence.get("evidence_path", ""))
        if (
            evidence.get("status") != "available"
            or evidence.get("device") != suite["contract"]["hardware"]
        ):
            errors.append(f"{capability}: no matching device validation")
        elif not path.is_file():
            errors.append(f"{capability}: evidence file missing")
        else:
            try:
                proof = json.loads(path.read_text())
                if (
                    proof.get("result") != "pass"
                    or proof.get("device") != suite["contract"]["hardware"]
                    or proof.get("software_hash")
                    != digest(suite["contract"]["software"])
                ):
                    errors.append(f"{capability}: stale or failed probe evidence")
            except (ValueError, OSError):
                errors.append(f"{capability}: unreadable probe evidence")
    return errors


def validate(suite):
    if suite.get("version") != 1:
        raise ValueError("Unsupported suite version")
    contract = suite["contract"]
    for key in (
        "hardware",
        "npu_count",
        "checkpoint_manifest",
        "checkpoint_receipt",
        "checkpoint_manifest_sha256",
        "mode",
        "width",
        "height",
        "fps",
        "actual_nfe",
        "software",
    ):
        if contract.get(key) is None:
            raise ValueError(f"Fill contract.{key} before running")
    if contract["actual_nfe"] not in (4, 8) or contract["npu_count"] != 8:
        raise ValueError(
            "Primary campaign requires four or eight scheduler updates and eight NPUs"
        )
    if not is_sha(contract["checkpoint_manifest_sha256"]):
        raise ValueError("Pin the verified checkpoint manifest hash")
    reject_runtime_lora(environment=dict(os.environ, **suite.get("environment", {})))
    ids = [case["id"] for case in suite["cases"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Case IDs must be unique")
    for case in suite["cases"]:
        if case.get("command") and not case.get("workload", {}).get("prompt"):
            raise ValueError("Executable cases require workload.prompt and seed")
        if (
            case.get("command")
            and type(case.get("workload", {}).get("seed")) is not int
        ):
            raise ValueError("Executable cases require integer workload.seed")
        if case["experiment"] not in CATALOG:
            raise ValueError(f"Unknown experiment: {case['experiment']}")
        if not case["id"].replace("-", "").replace("_", "").isalnum():
            raise ValueError("Case IDs must contain letters, digits, '-' or '_'")
        if case.get("classification", "primary") not in (
            "primary",
            "auxiliary",
            "oracle",
        ):
            raise ValueError("Invalid case classification")
        if (
            not isinstance(case.get("duration_s", 15), (int, float))
            or case.get("duration_s", 15) <= 0
        ):
            raise ValueError("Positive duration_s required")
        for count in ("warmups", "repeats"):
            value = case.get(count, suite.get(count, 3 if count == "warmups" else 5))
            if type(value) is not int or value < 1:
                raise ValueError(f"{count} must be a positive integer")


def probe_media(path, *, require_stream_durations=False):
    data = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        timeout=120,
    )
    probe = json.loads(data)
    videos = [s for s in probe["streams"] if s["codec_type"] == "video"]
    audios = [s for s in probe["streams"] if s["codec_type"] == "audio"]
    if not videos or not audios:
        raise ValueError("Output must contain both video and audio")
    video, audio = videos[0], audios[0]
    if require_stream_durations and (
        "duration" not in video or "duration" not in audio
    ):
        raise ValueError("Output must report independent video and audio durations")
    numerator, denominator = map(int, video["avg_frame_rate"].split("/"))
    return dict(
        width=video["width"],
        height=video["height"],
        fps=numerator / denominator,
        frames=int(video["nb_read_frames"]),
        duration_s=float(video.get("duration", probe["format"]["duration"])),
        audio_duration_s=float(audio.get("duration", probe["format"]["duration"])),
    )


def verify_sample(path, contract, case):
    sample = json.loads(path.read_text())
    if (
        sample.get("status") != "completed"
        or type(sample.get("scheduler_updates")) is not int
        or sample["scheduler_updates"] != contract["actual_nfe"]
    ):
        raise ValueError(
            "Adapter must report completed generation and measured scheduler updates"
        )
    if sample.get("profiled") is not False or sample.get("diagnostic") is not False:
        raise ValueError(
            "Timing requires explicitly unprofiled, non-diagnostic samples"
        )
    if type(sample.get("dit_evaluations")) is not int or sample["dit_evaluations"] < 1:
        raise ValueError(
            "DiT evaluations must count logical forwards, without TP/SP replication"
        )
    if (
        sample.get("checkpoint_manifest_sha256")
        != contract["checkpoint_manifest_sha256"]
    ):
        raise ValueError("Adapter checkpoint identity does not match campaign")
    if sample.get("contract_hash") != digest(contract):
        raise ValueError("Adapter contract hash does not match campaign")
    if sample.get("workload_hash") != digest(case.get("workload")):
        raise ValueError(
            "Adapter workload hash does not match requested prompt/seed/references"
        )
    media = Path(sample["output_path"])
    if not media.is_absolute():
        media = path.parent / media
    observed = probe_media(media)
    for key in ("width", "height"):
        if observed[key] != contract[key]:
            raise ValueError(f"Output {key} differs from contract")
    if abs(observed["fps"] - contract["fps"]) > 0.01:
        raise ValueError("Output FPS differs from contract")
    if "frame_count" in case and observed["frames"] != case["frame_count"]:
        raise ValueError("Output frame count differs from the case")
    target = case.get("duration_s", 15)
    tolerance = case.get("duration_tolerance_s", 17 / contract["fps"])
    if not target <= observed["duration_s"] <= target + tolerance:
        raise ValueError("Useful duration differs from native-aligned target")
    if abs(observed["duration_s"] - observed["audio_duration_s"]) > case.get(
        "av_duration_tolerance_s", 0.1
    ):
        raise ValueError("Audio/video durations differ beyond tolerance")
    sample.update(
        media=observed, output_sha256=hashlib.sha256(media.read_bytes()).hexdigest()
    )
    return sample


def run_case(suite, case, directory):
    directory.mkdir(parents=True)
    contract = suite["contract"]
    context = dict(
        contract=contract,
        contract_hash=digest(contract),
        case=case,
        workload_hash=digest(case.get("workload")),
    )
    save(directory / "context.json", context)
    env = dict(
        os.environ,
        **suite.get("environment", {}),
        H3_RESEARCH_CONTEXT=str((directory / "context.json").resolve()),
    )
    checkpoint = load_verified(
        contract["checkpoint_manifest"], contract["checkpoint_receipt"]
    )
    if (
        checkpoint_digest(checkpoint) != contract["checkpoint_manifest_sha256"]
        or SCHEDULES[checkpoint["family"]][0] != contract["actual_nfe"]
    ):
        raise ValueError("Checkpoint identity or schedule differs from the campaign")
    env.update(
        H3_CHECKPOINT_MANIFEST=contract["checkpoint_manifest"],
        H3_CHECKPOINT_RECEIPT=contract["checkpoint_receipt"],
        H3_EXPECTED_CHECKPOINT_SHA=contract["checkpoint_manifest_sha256"],
        H3_RESEARCH_MODEL_ID=model_id(contract["checkpoint_manifest_sha256"]),
    )
    samples = []
    warmups = case.get("warmups", suite.get("warmups", 3))
    repeats = case.get("repeats", suite.get("repeats", 5))
    timeout = case.get("timeout_s", suite.get("timeout_s", 1800))
    server = None
    server_log = None
    # Only environment preparation/server startup belongs outside timed commands.
    # Anchor preparation, transfers, decode, and final mux belong inside command.
    try:
        if case.get("setup_command"):
            command(
                case["setup_command"],
                env=env,
                log=directory / "setup.log",
                timeout=timeout,
            )
        if case.get("server_command"):
            import urllib.error
            import urllib.request

            def healthy():
                try:
                    with urllib.request.urlopen(
                        case["health_url"], timeout=2
                    ) as response:
                        return response.status == 200
                except (urllib.error.URLError, TimeoutError):
                    return False

            if healthy():
                raise ValueError("Health endpoint already occupied")
            from checkpoint_contract import assert_endpoint
            from h3_chunks import Client

            argv = case["server_command"]
            reject_runtime_lora(argv, env)
            require_launcher(argv)
            server_log = (directory / "server.log").open("w")
            server = subprocess.Popen(
                case["server_command"],
                env=dict(env, H3_SERVE_ONLY="1"),
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + timeout
            while not healthy():
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Server did not become ready")
                time.sleep(1)
            assert_endpoint(
                Client(case["health_url"].removesuffix("/health"), timeout),
                model_id(contract["checkpoint_manifest_sha256"]),
            )
        for index in range(warmups + repeats):
            result = (directory / f"sample-{index}.json").resolve()
            run_env = dict(
                env, H3_RESEARCH_RESULT=str(result), H3_RESEARCH_SAMPLE=str(index)
            )
            started = time.perf_counter()
            command(
                case["command"],
                env=run_env,
                log=directory / f"sample-{index}.log",
                timeout=timeout,
            )
            elapsed = time.perf_counter() - started
            sample = verify_sample(result, contract, case)
            sample.update(client_e2e_s=elapsed, warmup=index < warmups)
            save(result, sample)
            if index >= warmups:
                samples.append(sample)
    finally:
        if server is not None:
            try:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        if server_log is not None:
            server_log.close()
        if case.get("teardown_command"):
            command(
                case["teardown_command"],
                env=env,
                log=directory / "teardown.log",
                timeout=timeout,
            )
    quality = "pending"
    quality_metrics = {}
    if case.get("evaluation_command"):
        evaluation_path = (directory / "evaluation.json").resolve()
        command(
            case["evaluation_command"],
            env=dict(env, H3_RESEARCH_EVALUATION=str(evaluation_path)),
            log=directory / "evaluation.log",
            timeout=timeout,
        )
        evaluation = json.loads(evaluation_path.read_text())
        quality_metrics = evaluation["metrics"]
        thresholds = suite.get("quality_thresholds", {})
        passed = bool(thresholds)
        for metric, limits in thresholds.items():
            value = quality_metrics.get(metric)
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                passed = False
            elif ("min" in limits and value < limits["min"]) or (
                "max" in limits and value > limits["max"]
            ):
                passed = False
        if thresholds:
            quality = "metrics_pass" if passed else "metrics_fail"
    times = sorted(s["client_e2e_s"] for s in samples)
    return dict(
        status="measured",
        samples=samples,
        median_s=statistics.median(times),
        p90_s=times[math.ceil(0.9 * len(times)) - 1],
        median_rtf=statistics.median(
            s["client_e2e_s"] / s["media"]["duration_s"] for s in samples
        ),
        quality=quality,
        quality_metrics=quality_metrics,
        decision="reject" if quality == "metrics_fail" else "inconclusive",
    )


def append(path, record):
    with path.open("a") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


def report(output):
    records = [
        json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()
    ]
    rows = [
        "# H3 research campaign",
        "",
        "Host/mock checks do not establish NPU performance or quality.",
        "",
        "| Round | Case | Track | Class | Status | Median s | RTF | Quality |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    baselines = {
        (record["round"], record["contract_hash"], record["workload_hash"]): record[
            "result"
        ]["median_s"]
        for record in records
        if record["experiment"] == "B00"
        and record["status"] == "measured"
        and record["classification"] == "primary"
    }
    for record in records:
        result = record.get("result", {})
        rows.append(
            f"| {record['round']} | {record['case_id']} | {record['track']} | "
            f"{record['classification']} | {record['status']} | {result.get('median_s', '')} | "
            f"{result.get('median_rtf', '')} | {result.get('quality', 'pending')} |"
        )
    rows += ["", "## Comparisons", ""]
    for record in records:
        reference = baselines.get(
            (record["round"], record["contract_hash"], record["workload_hash"])
        )
        if (
            reference
            and record["status"] == "measured"
            and record["classification"] == "primary"
            and record["experiment"] != "B00"
        ):
            speedup = reference / record["result"]["median_s"]
            rows.append(
                f"- Round {record['round']} {record['case_id']}: {speedup:.3f}x; quality remains {record['result']['quality']}."
            )
    rows += ["", "## Blockers", ""]
    for record in records:
        if record["status"] in ("blocked", "failed"):
            rows.append(
                f"- {record['case_id']}: {record.get('reasons', record.get('error'))}"
            )
    (output / "report.md").write_text("\n".join(rows) + "\n")
    with (output / "samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "round",
                "case",
                "track",
                "classification",
                "sample",
                "e2e_s",
                "duration_s",
                "rtf",
                "quality",
            ]
        )
        for record in records:
            for index, sample in enumerate(record.get("result", {}).get("samples", [])):
                writer.writerow(
                    [
                        record["round"],
                        record["case_id"],
                        record["track"],
                        record["classification"],
                        index,
                        sample["client_e2e_s"],
                        sample["media"]["duration_s"],
                        sample["client_e2e_s"] / sample["media"]["duration_s"],
                        record["result"]["quality"],
                    ]
                )


def template():
    contract = dict(
        hardware="Ascend A5 (fill exact SoC)",
        npu_count=8,
        checkpoint_manifest=None,
        checkpoint_receipt=None,
        checkpoint_manifest_sha256=None,
        mode="t2va",
        width=1344,
        height=768,
        fps=24,
        actual_nfe=4,
        software=None,
    )
    return dict(
        version=1,
        contract=contract,
        warmups=3,
        repeats=5,
        capabilities={},
        cases=[
            dict(
                id=key.lower(),
                experiment=key,
                duration_s=15,
                workload=dict(prompt=None, seed=1101, references=[]),
                command=None,
                classification="oracle"
                if key == "A02"
                else "auxiliary"
                if key == "E02"
                else "primary",
            )
            for key in CATALOG
        ],
    )


def verify_parity(root, candidate, *, decoder_only=False):
    """Compare a matched B0/W0R pair using measured requests and decoded streams."""
    manifest = json.loads((root / "checkpoint.json").read_text())
    if manifest["family"] != "flashgen4":
        raise ValueError("B0/W0R parity requires the merged FlashGen4 checkpoint")
    identity = json.loads((root / "comparison.json").read_text())
    candidate_identity = json.loads((candidate / "comparison.json").read_text())
    candidate_manifest = json.loads((candidate / "checkpoint.json").read_text())
    checkpoint = checkpoint_digest(manifest)
    if decoder_only:
        decoders = manifest.pop("decoder"), candidate_manifest.pop("decoder")
        if {entry["kind"] for entry in decoders} != {"native", "lightvae"}:
            raise ValueError("Decoder comparison requires native VAE and LightVAE")
        identity["request"].pop("model")
        candidate_identity["request"].pop("model")
    if candidate_identity != identity or candidate_manifest != manifest:
        raise ValueError("Parity launch contracts differ")
    ranks = len(identity["placement"].split(":", 1)[0].split(","))

    def evidence(run, case):
        samples = [
            row
            for line in (run / "samples.jsonl").read_text().splitlines()
            if not (row := json.loads(line))["warmup"] and row["case"] == case
        ]
        if len(samples) != 1:
            raise ValueError("Use one measured request per parity launch")
        request_id = samples[0]["response"]["id"]
        denoise = json.loads((run / "denoise_matches.json").read_text())[request_id]
        if "scheduler_updates=4 dit_evaluations=4" not in denoise:
            raise ValueError(
                "B0 and W0R must both execute four updates and four forwards"
            )
        pattern = re.compile(
            rf"H3 window parity request={re.escape(request_id)} rank=(\d+) "
            r"point=(\w+) step=(-?\d+) tensors=(\{.*\})"
        )
        observed = {}
        for line in (run / "server.log").read_text().splitlines():
            if match := pattern.search(line):
                key = int(match[1]), match[2], int(match[3])
                if key in observed:
                    raise ValueError(f"Duplicate tensor record: {key}")
                observed[key] = json.loads(match[4])
        expected = {
            (rank, point, step)
            for rank in range(ranks)
            for point, steps in {
                "initial": (-1,),
                "input": range(4),
                "prediction": range(4),
                "update": range(4),
            }.items()
            for step in steps
        }
        if set(observed) != expected:
            raise ValueError(f"Missing or unexpected tensor records in {run}")
        media = []
        for stream, codec in (("v", "rawvideo"), ("a", "pcm_s32le")):
            media.append(
                subprocess.check_output(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-i",
                        str(run / "sample-1.mp4"),
                        "-map",
                        f"0:{stream}:0",
                        "-c",
                        codec,
                        "-threads",
                        "1",
                        "-f",
                        "hash",
                        "-hash",
                        "sha256",
                        "-",
                    ],
                    text=True,
                    timeout=180,
                ).strip()
            )
        return observed, media

    b0_tensors, b0_media = evidence(root, "B0")
    repack_tensors, repack_media = evidence(candidate, "B0" if decoder_only else "W0R")
    differences = sorted(
        key for key in b0_tensors if b0_tensors[key] != repack_tensors[key]
    )
    if differences or (not decoder_only and b0_media != repack_media):
        raise ValueError(
            f"B0/W0R parity failed. Tensor records: {differences}. Media: {b0_media} vs {repack_media}"
        )
    return dict(
        checkpoint=checkpoint,
        tensors=len(b0_tensors),
        decoded_video=b0_media[0],
        decoded_pcm=b0_media[1],
        candidate_decoded_video=repack_media[0],
        candidate_decoded_pcm=repack_media[1],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "plan", "run", "report", "parity"))
    parser.add_argument("suite", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--track", nargs="+")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--decoder-only", action="store_true")
    args = parser.parse_args()
    if args.action == "parity":
        if args.candidate is None:
            parser.error(
                "parity requires --candidate pointing at the W0R artifact directory"
            )
        print(
            json.dumps(
                verify_parity(
                    args.suite, args.candidate, decoder_only=args.decoder_only
                ),
                indent=2,
            )
        )
        return
    if args.action == "init":
        with args.suite.open("x") as stream:
            stream.write(json.dumps(template(), indent=2) + "\n")
        return
    if args.action == "report":
        report(args.suite)
        return
    suite = json.loads(args.suite.read_text())
    selected = [
        c
        for c in suite["cases"]
        if not args.track or CATALOG[c["experiment"]]["track"] in args.track
    ]
    if args.track and set(args.track) - {item["track"] for item in CATALOG.values()}:
        parser.error("Unknown track")
    if args.action == "plan":
        for case in selected:
            print(
                json.dumps(
                    dict(
                        case=case["id"],
                        **CATALOG[case["experiment"]],
                        blockers=evidence_errors(suite, case),
                    )
                )
            )
        return
    validate(suite)
    if args.rounds < 1 or args.output is None:
        parser.error("run requires --output and positive --rounds")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )
    source = source_state(root, args.output)
    evidence_hashes = {
        key: hashlib.sha256(Path(value["evidence_path"]).read_bytes()).hexdigest()
        for key, value in suite.get("capabilities", {}).items()
        if Path(value.get("evidence_path", "")).is_file()
    }
    packages = {}
    for name in ("torch", "torch-npu", "mindiesd", "sgl-kernel-npu"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    runtime = dict(
        packages=packages,
        python=sys.version,
        host=platform.platform(),
        environment={
            key: value
            for key, value in os.environ.items()
            if key.startswith(("ASCEND_", "CANN_", "HCCL_", "SGLANG_", "PYTORCH_"))
            and not any(
                word in key.upper() for word in ("TOKEN", "SECRET", "PASSWORD", "KEY")
            )
        },
    )
    fingerprint = digest(
        dict(suite=suite, source=source, evidence=evidence_hashes, runtime=runtime)
    )
    manifest = args.output / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Source/config changed; use a new output directory")
    else:
        save(
            manifest,
            dict(
                fingerprint=fingerprint,
                suite=suite,
                source=source,
                host=platform.platform(),
                python=sys.version,
                runtime=runtime,
            ),
        )
    ledger = args.output / "results.jsonl"
    old = (
        [json.loads(line) for line in ledger.read_text().splitlines()]
        if ledger.exists()
        else []
    )
    done = {(r["round"], r["case_id"]) for r in old if r["status"] == "measured"}
    for cycle in range(args.rounds):
        # Reverse alternating rounds to reduce systematic ordering bias.
        for case in selected if cycle % 2 == 0 else reversed(selected):
            if (cycle, case["id"]) in done:
                continue
            record = dict(
                round=cycle,
                case_id=case["id"],
                experiment=case["experiment"],
                track=CATALOG[case["experiment"]]["track"],
                classification=case.get("classification", "primary"),
                contract_hash=digest(suite["contract"]),
                workload_hash=digest(
                    dict(
                        workload=case.get("workload"),
                        duration_s=case.get("duration_s", 15),
                    )
                ),
                fingerprint=fingerprint,
            )
            blockers = evidence_errors(suite, case)
            if blockers:
                record.update(status="blocked", reasons=blockers)
            else:
                directory = (
                    args.output
                    / f"round-{cycle:03d}"
                    / f"{case['id']}-{time.time_ns()}"
                )
                try:
                    record["result"] = run_case(suite, case, directory)
                    record["status"] = "measured"
                except Exception as error:
                    record.update(
                        status="failed", error=str(error), artifacts=str(directory)
                    )
            append(ledger, record)
            report(args.output)
            print(
                json.dumps({k: v for k, v in record.items() if k != "result"}),
                flush=True,
            )


if __name__ == "__main__":
    main()
