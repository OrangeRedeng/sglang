# SPDX-License-Identifier: Apache-2.0
"""Bounded H3 experiment controller for the existing A3 launcher."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request
from contextlib import ExitStack, closing, contextmanager
from importlib import metadata
from pathlib import Path

from checkpoint_contract import digest as checkpoint_digest
from checkpoint_contract import is_sha, load_verified, reject_runtime_lora
from runner import digest, save

ROOT = Path(__file__).resolve().parents[4]
LAUNCHER = Path(__file__).with_name("run_a3_windowing.sh")
PROMPT = Path(__file__).with_name("SUPERVISOR.md")
REGISTRY = {
    **{
        f"R03.{name.upper()}": ("W2", False, None)
        for name in ("fixed", "shifted", "adaptive", "context", "cosine")
    },
    "H05.B0": ("B0", False, None),
    "H05.W0R": ("W0R", False, None),
    "H05.W1": ("W1", False, "h05_repack_parity"),
    "H05.W2": ("W2", False, "h05_repack_parity"),
    "H03.STEP3_TRACE": ("B0", True, None),
    "H08.NATIVE50": ("B0", False, None),
    "R01.CLEAN4.B0": ("B0", False, None),
    "R01.CLEAN4.W0R": ("W0R", False, None),
    "R01.PARITY.B0": ("B0", False, None),
    "R01.PARITY.W0R": ("W0R", False, None),
    "R01.CROP.BEFORE": ("B0", False, None),
    "R01.CROP.AFTER": ("B0", False, None),
    "R01.QUALITY.NATIVE": ("B0", False, None),
    "R01.QUALITY.FLASHGEN": ("B0", False, None),
    "R01.QUALITY.TURBO": ("B0", False, None),
    "R01.HIST.OFF": ("HIST-OFF", False, None),
    "R01.HIST.ON": ("HIST-ON", False, None),
}
CHECKPOINT_FAMILIES = {
    "A3-C4": "flashgen4",
    "A3-T8": "turbo8",
    "A3-NATIVE50": "native50",
    "A5-TARGET": "flashgen4",
}
CASE_FAMILIES = {
    "H08.NATIVE50": "native50",
    "R01.QUALITY.NATIVE": "native50",
    "R01.QUALITY.TURBO": "turbo8",
    "R01.HIST.OFF": "native50",
    "R01.HIST.ON": "native50",
}
ACTIONS = {
    "run_case",
    "repeat_case",
    "request_diagnostics",
    "mark_blocked",
    "request_review",
    "stop_campaign",
}
CLEANUP_SECONDS = 30


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path, deadline=None):
    with path.open("rb") as stream:
        fingerprint = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 * 2**20), b""):
            if deadline is not None:
                require(
                    time.monotonic() < deadline,
                    "Fingerprinting exceeded the time budget",
                )
            fingerprint.update(chunk)
        return fingerprint.hexdigest()


def fields(record, names):
    require(
        isinstance(record, dict) and record.keys() == set(names.split()),
        f"Expected fields: {names}",
    )


def named(value):
    return isinstance(value, str) and re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value
    )


def positive(value):
    return type(value) is int and value > 0


def absolute_path(value):
    return isinstance(value, str) and Path(value).is_absolute()


def validate(manifest):
    fields(
        manifest, "version id approved source_sha contract limits capabilities cases"
    )
    require(
        type(manifest["version"]) is int and manifest["version"] == 2,
        "Unsupported manifest version",
    )
    require(
        named(manifest["id"]) and type(manifest["approved"]) is bool,
        "Invalid campaign identity or approval",
    )
    require(
        re.fullmatch(r"[0-9a-f]{40}", manifest["source_sha"] or "") is not None,
        "Pin a full Git SHA",
    )

    contract = manifest["contract"]
    fields(
        contract,
        "family host devices soc_version port checkpoint_manifest checkpoint_receipt "
        "checkpoint_manifest_sha256 software prompt seed quality",
    )
    sizes = {
        "A3-C4": (8, 16),
        "A3-T8": (8, 16),
        "A3-NATIVE50": (8, 16),
        "A5-TARGET": (8,),
    }
    family = contract["family"]
    require(family in sizes, "Unknown deployment family")
    devices = contract["devices"]
    require(
        isinstance(devices, list)
        and len(devices) in sizes[family]
        and all(type(device) is int and device >= 0 for device in devices)
        and len(set(devices)) == len(devices),
        "Invalid or overlapping device IDs",
    )
    require(
        positive(contract["port"]) and contract["port"] < 65535,
        "Reserve distinct HTTP and broker ports",
    )
    require(positive(contract["soc_version"]), "Pin the device runtime SoC version")
    require(
        named(contract["host"])
        and absolute_path(contract["checkpoint_manifest"])
        and absolute_path(contract["checkpoint_receipt"])
        and is_sha(contract["checkpoint_manifest_sha256"]),
        "Pin the host, verified checkpoint manifest and receipt",
    )
    software = contract["software"]
    fields(software, "torch torch_npu mindiesd cann_version_file cann_sha256")
    for name in ("torch", "torch_npu", "mindiesd"):
        require(
            isinstance(software[name], str) and 0 < len(software[name]) <= 80,
            "Record the approved package versions; use 'absent' for optional MindIE-SD",
        )
    require(
        absolute_path(software["cann_version_file"])
        and re.fullmatch(r"[0-9a-f]{64}", software["cann_sha256"] or ""),
        "Pin the CANN version file",
    )
    require(
        isinstance(contract["prompt"], str) and 0 < len(contract["prompt"]) <= 8192,
        "Supply a bounded prompt",
    )
    require(
        type(contract["seed"]) is int and contract["quality"] in ("exact", "lossless"),
        "Select a seed and an explicit quality tier",
    )
    limits = manifest["limits"]
    fields(
        limits,
        "max_runs max_wall_s max_device_s run_timeout_s min_free_mib min_disk_mib",
    )
    require(
        all(positive(value) for value in limits.values()), "Invalid campaign limits"
    )
    require(
        limits["run_timeout_s"] > CLEANUP_SECONDS, "Timeout leaves no cleanup allowance"
    )
    require(isinstance(manifest["capabilities"], dict), "Invalid capability evidence")
    for name, proof in manifest["capabilities"].items():
        fields(proof, "path sha256")
        require(
            named(name)
            and absolute_path(proof["path"])
            and re.fullmatch(r"[0-9a-f]{64}", proof["sha256"] or ""),
            "Invalid capability receipt",
        )

    cases = manifest["cases"]
    require(isinstance(cases, list) and 0 < len(cases) <= 32, "Expected 1–32 cases")
    seen = {}
    for case in cases:
        fields(case, "id adapter after repeat_of")
        case_id, adapter = case["id"], case["adapter"]
        require(
            named(case_id) and case_id not in seen and adapter in REGISTRY,
            "Invalid case identity or adapter",
        )
        require(
            CHECKPOINT_FAMILIES[family] == CASE_FAMILIES.get(adapter, "flashgen4"),
            "Adapter and schedule must use the same deployment family",
        )
        if adapter.startswith("R01.HIST."):
            require(len(devices) == 8, "Historical controls require eight devices")
        after, repeat = case["after"], case["repeat_of"]
        require(
            isinstance(after, list) and all(item in seen for item in after),
            "Dependencies must precede the case",
        )
        similar = [key for key, value in seen.items() if value == adapter]
        if similar:
            require(
                repeat in similar and repeat in after,
                "Duplicate adapter requires an explicit repeat dependency",
            )
        else:
            require(
                repeat is None, "Repeat must name an earlier case of the same adapter"
            )
        seen[case_id] = adapter


def readiness(manifest, latest):
    result = {}
    for case in manifest["cases"]:
        case_id = case["id"]
        _, _, capability = REGISTRY[case["adapter"]]
        if case_id in latest:
            result[case_id] = latest[case_id]["status"]
        elif manifest["contract"]["family"] == "A5-TARGET":
            result[case_id] = "blocked: no A5 launcher"
        elif any(
            latest.get(dep, {}).get("status") not in ("measured", "diagnostic")
            for dep in case["after"]
        ):
            result[case_id] = "blocked: dependency"
        elif capability and not capability_passed(manifest, capability):
            result[case_id] = f"blocked: {capability}"
        else:
            result[case_id] = "ready"
    return result


def capability_passed(manifest, capability):
    proof = manifest["capabilities"].get(capability)
    if proof is None:
        return False
    try:
        payload = Path(proof["path"]).read_bytes()
        record = json.loads(payload)
        return (
            hashlib.sha256(payload).hexdigest() == proof["sha256"]
            and record.get("result") == "pass"
            and record.get("contract_hash") == digest(manifest["contract"])
        )
    except (OSError, ValueError, AttributeError):
        return False


def snapshot(manifest, latest, remaining_s, remaining_runs, remaining_wall_s=None):
    ready = readiness(manifest, latest)
    cases = [
        dict(
            case_id=case["id"],
            variant_id="default",
            adapter=case["adapter"],
            action=action_for(case),
            dependencies=case["after"],
        )
        for case in manifest["cases"]
        if ready[case["id"]] == "ready"
    ]
    data = dict(
        eligible_cases=cases,
        remaining_device_s=remaining_s,
        remaining_runs=remaining_runs,
        remaining_wall_s=remaining_wall_s,
        evidence=[
            dict(
                case_id=key,
                status=value["status"],
                median_client_s=value.get("median_client_s"),
                quality=value.get("quality"),
            )
            for key, value in latest.items()
        ],
    )
    return dict(snapshot_id=digest(data), **data)


def action_for(case):
    if case["repeat_of"] is not None:
        return "repeat_case"
    return "request_diagnostics" if REGISTRY[case["adapter"]][1] else "run_case"


def select(decision, current):
    fields(
        decision,
        "schema_version snapshot_id action case_id variant_id evidence_ids reason",
    )
    require(
        type(decision["schema_version"]) is int
        and decision["schema_version"] == 1
        and decision["snapshot_id"] == current["snapshot_id"],
        "Stale supervisor decision",
    )
    action, case_id, variant = (
        decision["action"],
        decision["case_id"],
        decision["variant_id"],
    )
    require(
        action in ACTIONS
        and isinstance(decision["reason"], str)
        and 0 < len(decision["reason"]) <= 400,
        "Invalid decision action or rationale",
    )
    known = {item["case_id"] for item in current["evidence"]}
    evidence = decision["evidence_ids"]
    require(
        isinstance(evidence, list) and all(item in known for item in evidence),
        "Unknown evidence ID",
    )
    eligible = {item["case_id"]: item for item in current["eligible_cases"]}
    if action == "stop_campaign":
        require(case_id is None and variant is None, "Stop has no case")
    elif action == "request_review":
        require(
            (case_id is None or case_id in eligible or case_id in known)
            and variant is None,
            "Unknown review case",
        )
    elif action == "mark_blocked":
        require(
            case_id in eligible and variant is None, "Block requires an eligible case"
        )
    else:
        require(
            case_id in eligible
            and variant == "default"
            and action == eligible[case_id]["action"],
            "Ineligible action or variant",
        )
    return action, case_id


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def supervisor(config, current):
    fields(config, "endpoint model")
    require(
        isinstance(config["endpoint"], str)
        and re.fullmatch(
            r"http://(?:127\.0\.0\.1|\[::1\]):[0-9]{1,5}/v1/chat/completions",
            config["endpoint"],
        ),
        "Use a loopback OpenAI-compatible endpoint",
    )
    require(
        isinstance(config["model"], str) and 0 < len(config["model"]) <= 128,
        "Set a supervisor model",
    )
    payload = dict(
        model=config["model"],
        temperature=0,
        max_tokens=500,
        messages=[
            dict(role="system", content=PROMPT.read_text()),
            dict(role="user", content=json.dumps(current)),
        ],
    )
    request = urllib.request.Request(
        config["endpoint"],
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=8) as response:
        data = json.loads(response.read(65537))
    content = data["choices"][0]["message"]["content"]
    require(
        isinstance(content, str) and len(content) <= 4096,
        "Supervisor response too long",
    )
    return json.loads(content)


def journal(directory):
    database = sqlite3.connect(directory / "ledger.sqlite3", timeout=5)
    database.execute("CREATE TABLE IF NOT EXISTS events (body TEXT NOT NULL)")
    database.commit()
    return database


def record(database, kind, **kwargs):
    value = dict(kind=kind, at=time.time(), **kwargs)
    with database:
        database.execute("INSERT INTO events VALUES (?)", (json.dumps(value),))


def history(database):
    return [
        json.loads(row[0])
        for row in database.execute("SELECT body FROM events ORDER BY rowid")
    ]


def latest_cases(events):
    return {item["case_id"]: item for item in events if item["kind"] == "case"}


@contextmanager
def exclusive(path):
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Reservation busy: {path.name}") from error
        yield lock.fileno()


def reserve(stack, paths):
    return [stack.enter_context(exclusive(path)) for path in paths]


def preflight(manifest, directory, deadline):
    contract = manifest["contract"]
    require(
        socket.gethostname() == contract["host"], "Host does not match the allocation"
    )
    source = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"])
    require(
        source == manifest["source_sha"] and not dirty,
        "Use the approved clean checkout",
    )
    require(
        shutil.disk_usage(directory).free >= manifest["limits"]["min_disk_mib"] * 2**20,
        "Insufficient free disk space",
    )
    for tool in ("bash", "sglang", "ffprobe", "ffmpeg", "npu-smi"):
        require(shutil.which(tool) is not None, f"Missing dependency: {tool}")
    software = contract["software"]
    for name in ("torch", "torch_npu", "mindiesd"):
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = "absent"
        require(version == software[name], f"Runtime fingerprint mismatch: {name}")
    require(
        sha256_file(Path(software["cann_version_file"]), deadline)
        == software["cann_sha256"],
        "CANN fingerprint mismatch",
    )
    checkpoint = load_verified(
        contract["checkpoint_manifest"], contract["checkpoint_receipt"]
    )
    require(
        checkpoint_digest(checkpoint) == contract["checkpoint_manifest_sha256"]
        and checkpoint["family"] == CHECKPOINT_FAMILIES[contract["family"]],
        "Checkpoint identity or schedule differs from the approved campaign",
    )
    require(
        checkpoint["family"] != "native50" or checkpoint["decoder"]["kind"] == "native",
        "Native50 requires the native decoder",
    )


def launch_environment(manifest, case):
    contract = manifest["contract"]
    reject_runtime_lora()
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("H3_", "SGLANG_"))
        and key not in ("PYTHONPATH", "BASH_ENV", "ENV")
    }
    env.update(
        ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, contract["devices"])),
        H3_CHECKPOINT_MANIFEST=contract["checkpoint_manifest"],
        H3_CHECKPOINT_RECEIPT=contract["checkpoint_receipt"],
        H3_EXPECTED_CHECKPOINT_SHA=contract["checkpoint_manifest_sha256"],
        H3_PROMPT=contract["prompt"],
        H3_NUM_GPUS=str(len(contract["devices"])),
        H3_LASER16="1",
        H3_PORT=str(contract["port"]),
        H3_PIN_CPU_MEMORY="false",
        H3_SEED=str(contract["seed"]),
        H3_QUALITY=contract["quality"],
        H3_PROFILE=str(int(REGISTRY[case["adapter"]][1])),
        H3_REPEATS="1",
        SGLANG_PROFILE_RECORD_SHAPES="1",
        SGLANG_H3_NPU_ADALN_MEMO="0",
    )
    adapter = case["adapter"]
    if adapter.startswith("R03."):
        env["H3_SEAM_POLICY"] = adapter.split(".")[1].lower()
    if adapter in ("R01.PARITY.B0", "R01.PARITY.W0R"):
        env["H3_WINDOW_PARITY"] = "1"
    if adapter.startswith("R01.CROP."):
        env["H3_LASER_CROP_BEFORE_CAST"] = str(int(adapter.endswith(".BEFORE")))
    return env


def check_devices(env, manifest):
    expected = len(manifest["contract"]["devices"])
    probe = (
        "import json, torch, torch_npu\n"
        "devices = []\n"
        "for index in range(torch.npu.device_count()):\n"
        "    torch.npu.set_device(index)\n"
        "    devices.append({'soc': torch.npu.get_soc_version(),"
        " 'free_bytes': torch.npu.mem_get_info(index)[0]})\n"
        "print(json.dumps(devices))\n"
    )
    raw = subprocess.check_output(
        [sys.executable, "-c", probe], env=env, timeout=30, text=True
    )
    visible = json.loads(raw.splitlines()[-1])
    require(
        len(visible) == expected
        and all(
            device["soc"] == manifest["contract"]["soc_version"]
            and device["free_bytes"] >= manifest["limits"]["min_free_mib"] * 2**20
            for device in visible
        ),
        "Visible device count, SoC or free HBM differs from the approved allocation",
    )
    return visible


def run_launcher(argv, env, log, timeout, stop, lock_fds):
    if stop.exists():
        return "cancelled"
    if timeout <= CLEANUP_SECONDS:
        return "timed_out"
    deadline = time.monotonic() + timeout - CLEANUP_SECONDS
    with log.open("w") as output:
        process = subprocess.Popen(
            argv,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=tuple(lock_fds),
        )
        status = None
        heartbeat_at = 0
        try:
            while process.poll() is None:
                if stop.exists():
                    status = "cancelled"
                    break
                if time.monotonic() >= deadline:
                    status = "timed_out"
                    break
                if time.monotonic() >= heartbeat_at:
                    save(
                        log.parent / "heartbeat.json",
                        dict(pid=process.pid, at=time.time()),
                    )
                    heartbeat_at = time.monotonic() + 5
                time.sleep(0.25)
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=CLEANUP_SECONDS)
                except ProcessLookupError:
                    process.wait()
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    status = "cleanup_unverified"
    return status or ("completed" if process.returncode == 0 else "failed")


def results(directory, adapter, deadline):
    runs = list(directory.glob("run-*"))
    require(
        len(runs) == 1 and runs[0].is_dir(), "Expected one launcher artifact directory"
    )
    run = runs[0]
    records = [
        json.loads(line) for line in (run / "samples.jsonl").read_text().splitlines()
    ]
    matches = json.loads((run / "denoise_matches.json").read_text())
    require(
        len(records) == 2
        and records[0]["warmup"] is True
        and records[1]["warmup"] is False
        and all(row["response"]["status"] == "completed" for row in records),
        "Expected one warmup and one completed sample",
    )
    request_ids = {records[1]["response"]["id"]}
    require(
        request_ids == matches.keys(),
        "Missing or duplicate denoise evidence",
    )
    profile = REGISTRY[adapter][1]
    frames = 124 if adapter.startswith("R01.HIST.") else 362
    hashes = []
    for index, row in enumerate(records):
        media = row["media"]
        require(
            (media["width"], media["height"], media["fps"], media["frames"])
            == (1344, 768, 24, frames),
            "Unexpected output geometry",
        )
        require(
            all(
                math.isfinite(media[name]) and abs(media[name] - frames / 24) <= 0.1
                for name in ("duration_s", "audio_duration_s")
            ),
            "Invalid AV durations",
        )
        hashes.append(sha256_file(run / f"sample-{index}.mp4", deadline))
    elapsed = records[1]["e2e_s"]
    require(
        records[1]["profiled"] == profile and math.isfinite(elapsed) and elapsed > 0,
        "Invalid measured clock",
    )
    if profile:
        require(
            any(path.is_file() for path in (run / "profiles").rglob("*")),
            "Missing device profiler artifacts",
        )
    return dict(
        artifacts=str(run),
        output_sha256=hashes,
        quality="review_required",
        median_client_s=None if profile else elapsed,
        p90_client_s=None if profile else elapsed,
        request_ids=sorted(request_ids),
    )


def campaign(manifest, directory, config=None):
    validate(manifest)
    require(manifest["approved"], "Campaign approval is required")
    require(
        not directory.is_relative_to(ROOT),
        "Put campaign artifacts outside the source tree",
    )
    directory.mkdir(parents=True, exist_ok=True)
    locked = directory / "manifest.json"
    stop = directory / "stop.requested"
    with ExitStack() as stack:
        campaign_fd = reserve(stack, [directory / "controller.lock"])
        if locked.exists():
            require(
                json.loads(locked.read_text()) == manifest, "Resume manifest mismatch"
            )
        else:
            save(locked, manifest)
        database = stack.enter_context(closing(journal(directory)))
        events = history(database)
        require(
            not any(
                item["status"] in ("running", "cleanup_unverified")
                for item in latest_cases(events).values()
            ),
            "Unreconciled run; have the operator confirm the allocation is clean",
        )
        if stop.exists():
            return
        if not any(
            value == "ready"
            for value in readiness(manifest, latest_cases(events)).values()
        ):
            return
        limits = manifest["limits"]
        if not events:
            record(database, "campaign_started")
            events = history(database)
        started_at = events[0]["at"]
        contract = manifest["contract"]
        lock_dir = Path("/tmp/sglang-h3-reservations")
        lock_dir.mkdir(exist_ok=True)
        paths = [
            lock_dir / f"npu-{device}.lock" for device in sorted(contract["devices"])
        ]
        paths += [
            lock_dir / f"port-{port}.lock"
            for port in (contract["port"], contract["port"] + 1)
        ]
        lock_fds = campaign_fd + reserve(stack, paths)
        deadline = time.monotonic() + started_at + limits["max_wall_s"] - time.time()
        preflight(manifest, directory, deadline)
        while not stop.exists():
            events = history(database)
            latest = latest_cases(events)
            charged = [item for item in events if item.get("charged_device_s")]
            remaining = limits["max_device_s"] - sum(
                item["charged_device_s"] for item in charged
            )
            remaining_runs = limits["max_runs"] - len(charged)
            charge = len(contract["devices"]) * limits["run_timeout_s"]
            remaining_wall = max(
                0, math.floor(started_at + limits["max_wall_s"] - time.time())
            )
            current = snapshot(
                manifest, latest, remaining, remaining_runs, remaining_wall
            )
            if (
                not current["eligible_cases"]
                or remaining_runs <= 0
                or charge > remaining
                or time.time() + limits["run_timeout_s"]
                > started_at + limits["max_wall_s"]
            ):
                record(
                    database, "pause", reason="No ready case fits the remaining budget"
                )
                return

            chosen = current["eligible_cases"][0]
            action, case_id = chosen["action"], chosen["case_id"]
            if config is not None:
                try:
                    proposal = supervisor(config, current)
                    action, case_id = select(proposal, current)
                    record(
                        database,
                        "decision",
                        accepted=True,
                        model=config["model"],
                        prompt_sha256=sha256_file(PROMPT),
                        proposal=proposal,
                    )
                except (
                    ValueError,
                    OSError,
                    TypeError,
                    KeyError,
                    IndexError,
                    AttributeError,
                ) as error:
                    record(
                        database,
                        "decision",
                        accepted=False,
                        snapshot_id=current["snapshot_id"],
                        reason=type(error).__name__,
                    )
                    return
            if action in ("request_review", "stop_campaign"):
                record(database, "pause", reason=action, case_id=case_id)
                return
            if action == "mark_blocked":
                record(database, "case", case_id=case_id, status="blocked")
                continue
            if (
                stop.exists()
                or time.time() + limits["run_timeout_s"]
                > started_at + limits["max_wall_s"]
            ):
                record(
                    database, "pause", reason="Stop requested or wall budget exhausted"
                )
                return
            case = next(item for item in manifest["cases"] if item["id"] == case_id)
            output = directory / "cases" / case_id
            output.mkdir(parents=True)
            env = launch_environment(manifest, case)
            argv = [
                "bash",
                str(LAUNCHER),
                "--inside",
                str(ROOT),
                REGISTRY[case["adapter"]][0],
                str(output / "run"),
            ]
            record(
                database,
                "case",
                case_id=case_id,
                status="running",
                charged_device_s=charge,
                artifacts=str(output),
                argv=argv,
                action=action,
            )
            run_started = time.monotonic()
            status = "blocked"
            evidence = dict(artifacts=str(output))
            try:
                visible = check_devices(env, manifest)
                save(output / "device_preflight.json", visible)
                timeout = limits["run_timeout_s"] - (time.monotonic() - run_started)
                status = run_launcher(
                    argv, env, output / "launcher.log", timeout, stop, lock_fds
                )
                if status == "completed":
                    evidence = results(
                        output, case["adapter"], run_started + limits["run_timeout_s"]
                    )
                    status = (
                        "diagnostic" if REGISTRY[case["adapter"]][1] else "measured"
                    )
            except (
                ValueError,
                OSError,
                KeyError,
                TypeError,
                subprocess.SubprocessError,
            ) as error:
                evidence["error"] = str(error)
                if status == "completed":
                    status = "failed"
            record(database, "case", case_id=case_id, status=status, **evidence)
            if status not in ("measured", "diagnostic"):
                return


def dashboard(directory, manifest, events):
    latest = latest_cases(events)
    used = sum(item.get("charged_device_s", 0) for item in events)
    ready = readiness(manifest, latest)
    data = dict(
        campaign=manifest["id"], charged_device_s=used, states=ready, cases=latest
    )
    save(directory / "status.json", data)
    lines = [
        f"# {manifest['id']}",
        "",
        f"Reserved device-seconds recorded: {used}",
        "",
        "| Case | Status | Quality | Client seconds | Evidence |",
        "|---|---|---|---:|---|",
    ]
    for case, state in ready.items():
        row = latest.get(case, {})
        duration = row.get("median_client_s")
        artifact = row.get("artifacts")
        evidence = f"[Artifacts]({Path(artifact).as_uri()})" if artifact else "—"
        lines.append(
            f"| {case} | {state} | {row.get('quality', 'pending')} | "
            f"{'—' if duration is None else round(duration, 3)} | {evidence} |"
        )
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    for operation in ("plan", "run"):
        command = sub.add_parser(operation)
        command.add_argument("--manifest", required=True, type=Path)
        if operation == "run":
            command.add_argument("--dry-run", action="store_true")
    for operation in ("resume", "status", "stop", "report"):
        sub.add_parser(operation).add_argument(
            "--campaign-dir", required=True, type=Path
        )
    for operation in ("run", "resume"):
        mode = sub.choices[operation].add_mutually_exclusive_group()
        mode.add_argument("--no-model", action="store_true")
        mode.add_argument("--supervisor-config", type=Path)
    sub.choices["run"].add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.operation in ("plan", "run"):
            manifest = json.loads(args.manifest.read_text())
            validate(manifest)
            if args.operation == "plan" or args.dry_run:
                print(json.dumps(readiness(manifest, {}), indent=2))
                return
            require(args.output is not None, "Run requires --output")
            directory = args.output.resolve()
            require(
                not (directory / "manifest.json").exists(),
                "Use resume for an existing campaign",
            )
        else:
            directory = args.campaign_dir.resolve()
            manifest = json.loads((directory / "manifest.json").read_text())
            validate(manifest)
        if args.operation == "stop":
            (directory / "stop.requested").touch()
            return
        if args.operation in ("run", "resume"):
            config = (
                json.loads(args.supervisor_config.read_text())
                if args.supervisor_config
                else None
            )
            campaign(manifest, directory, config)
        with closing(journal(directory)) as database:
            print(
                json.dumps(dashboard(directory, manifest, history(database)), indent=2)
            )
    except (
        ValueError,
        TypeError,
        OSError,
        KeyError,
        sqlite3.Error,
        subprocess.SubprocessError,
    ) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
