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
import uuid
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
    # R05-R08 adapters, registered only for features implemented on their
    # branches. Each carries a capability gate so the case stays blocked
    # until the operator supplies evidence from the branch that implements
    # it; on a checkout without the flag the env var is an inert no-op.
    "R05.NPUADALN.ON": ("B0", False, "r05_npu_adaln"),
    "R05.SP8PAD.L16": ("L16", False, "r05_sp8_pad"),
    "R06.GRAPH.W0R": ("W0R", False, "r06_vae_graph"),
    "R08.SPARSE.H08": ("B0", False, "r08_sparse"),
    "R08.DISPATCHLOG": ("B0", True, None),
    "R08.MEMOSTATS": ("B0", True, None),
    "R08.FFNLOG": ("B0", True, None),
    "R02.Q5_FFN": ("B0", False, "r02_q5_ffn"),
    "R02.Q5_ALLOWLIST": ("B0", False, "r02_q5_allowlist"),
}
# Evidence identifiers that must never unlock a gate (invalid historical
# comparisons; see R08/H02 audit in RESEARCH_STATE).
EVIDENCE_BLOCKLIST = ("H02",)
# Consecutive supervisor-endpoint failures tolerated before the campaign
# pauses instead of silently degrading to the deterministic queue forever.
SUPERVISOR_FALLBACK_LIMIT = 3
# Quality promotion requires at least this many matched repetitions.
PROMOTION_MIN_SAMPLES = 10
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
for _adapter in ("R02.Q5_FFN", "R02.Q5_ALLOWLIST"):
    CASE_FAMILIES[_adapter] = "flashgen4"
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
        manifest,
        "version id approved source_sha source_root launcher contract limits capabilities cases",
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
    require(
        absolute_path(manifest["source_root"])
        and absolute_path(manifest["launcher"])
        ,
        "Pin the source worktree and its branch-specific launcher",
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
        fields(case, "id adapter after repeat_of samples")
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
        if adapter == "R05.SP8PAD.L16":
            require(len(devices) == 16, "L16 head padding requires sixteen devices")
        if adapter in ("R02.Q5_FFN", "R02.Q5_ALLOWLIST"):
            require(family == "A5-TARGET", "Conservative A5 R02 cases require the A5 target family")
        samples = case["samples"]
        require(
            type(samples) is int and 1 <= samples <= 12,
            "Case samples must be 1-12 raw repetitions",
        )
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


def aggregate_samples(samples):
    """Median and nearest-rank p90 over raw client samples.

    Significance is ``screening`` below the promotion threshold: a p90 from
    two or three samples is a screen, never promotion evidence.
    """
    require(bool(samples), "No samples to aggregate")
    ordered = sorted(float(value) for value in samples)
    count = len(ordered)

    def nearest_rank(percent):
        return ordered[math.ceil(percent / 100 * count) - 1]

    return dict(
        sample_count=count,
        median_client_s=nearest_rank(50),
        p90_client_s=nearest_rank(90),
        significance="screening" if count < PROMOTION_MIN_SAMPLES else "promotable",
    )


def validate_denoise_match(match, adapter):
    """Require measured scheduler and logical-forward counters from the launcher."""
    require(isinstance(match, str), "Denoise evidence must be a server log line")
    expected_updates = 50 if adapter in ("H08.NATIVE50", "R01.HIST.OFF", "R01.HIST.ON") else 8 if adapter == "R01.QUALITY.TURBO" else 4
    pattern = re.search(
        r"scheduler_updates=(\d+) dit_evaluations=(\d+)", match
    )
    require(pattern is not None, "Missing scheduler and logical DiT counters")
    updates, evaluations = map(int, pattern.groups())
    require(
        updates == expected_updates and evaluations > 0,
        "Invalid scheduler or logical DiT counters",
    )
    return updates, evaluations


def classify_failure(error_text, status):
    """Bounded failure taxonomy for retry policy.

    OOM and correctness/geometry failures block the track immediately; only
    unclassified (transient) failures may retry, once.
    """
    if status == "cleanup_unverified":
        return "cleanup"
    text = (error_text or "").lower()
    if "out of memory" in text or "oom" in text or "hbm" in text:
        return "oom"
    if "geometry" in text or "parity" in text or "expected" in text or "denoise" in text:
        return "correctness"
    return "transient"


def review_receipt(
    case_id, artifacts, output_hashes, prompt, policy, criteria, decision, at=None
):
    """A human quality receipt bound to the artifact hashes it judges."""
    require(named(case_id), "Invalid receipt case id")
    require(
        policy in ("exact", "lossless", "approximate"),
        "Receipt must name the quality policy",
    )
    require(
        isinstance(criteria, str) and 0 < len(criteria) <= 2000,
        "Receipt must state the acceptance criteria",
    )
    require(
        all(re.fullmatch(r"[0-9a-f]{64}", item or "") for item in output_hashes),
        "Receipt must cite concrete output hashes",
    )
    require(
        decision in ("quality_accepted", "quality_rejected"),
        "Receipt decision must be quality_accepted or quality_rejected",
    )
    body = dict(
        kind="quality_review",
        case_id=case_id,
        at=time.time() if at is None else at,
        artifacts=str(artifacts),
        output_sha256=list(output_hashes),
        prompt=prompt,
        policy=policy,
        criteria=criteria,
        decision=decision,
        receipt_sha256="",
    )
    body["receipt_sha256"] = digest(
        {key: value for key, value in body.items() if key != "receipt_sha256"}
    )
    return body


def apply_quality_receipt(database, receipt):
    """Validate a human receipt against the ledger and record its verdict.

    The receipt must hash-verify (stale/tampered receipts are rejected) and
    must cite the output hashes the case actually recorded. ``quality_rejected``
    never satisfies a dependency (readiness counts only measured/diagnostic),
    and no receipt can cite blocklisted historical evidence (H02).
    """
    fields(
        receipt,
        "kind case_id at artifacts output_sha256 prompt policy criteria decision "
        "receipt_sha256",
    )
    require(receipt["kind"] == "quality_review", "Not a quality review receipt")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    require(
        digest(body) == receipt["receipt_sha256"],
        "Receipt hash mismatch: stale or tampered",
    )
    case_id = receipt["case_id"]
    require(
        not any(case_id.startswith(prefix) for prefix in EVIDENCE_BLOCKLIST),
        "Blocklisted historical comparison cannot unlock gates",
    )
    latest = latest_cases(history(database))
    require(case_id in latest, "Receipt names an unknown case")
    recorded = latest[case_id]
    require(
        recorded.get("artifacts") == receipt["artifacts"],
        "Receipt artifacts differ from the recorded evidence",
    )
    require(
        list(recorded.get("output_sha256", [])) == list(receipt["output_sha256"]),
        "Receipt output hashes differ from the recorded evidence",
    )
    # The verdict is the human's word, typed: accepted/rejected must come in
    # as an explicit decision field to stay auditable.
    verdict = receipt["decision"]
    require(
        verdict in ("quality_accepted", "quality_rejected"),
        "Receipt decision must be quality_accepted or quality_rejected",
    )
    record(
        database,
        "case",
        case_id=case_id,
        status=recorded["status"],
        # Carry the evidence binding forward so a later receipt still
        # validates against the latest record for this case.
        artifacts=recorded.get("artifacts"),
        output_sha256=list(recorded.get("output_sha256", [])),
        run_id=recorded.get("run_id"),
        fingerprint=recorded.get("fingerprint"),
        quality=verdict,
        review_receipt=body | {"receipt_sha256": receipt["receipt_sha256"]},
    )
    return verdict


def ports_available(ports):
    """Preflight: every reserved port must be bindable now."""
    for port in ports:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))
        except OSError as error:
            raise ValueError(f"Port {port} is not available: {error}") from error
        finally:
            probe.close()


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
        isinstance(evidence, list),
        "Evidence IDs must be a list",
    )
    require(
        all(
            not any(item.startswith(prefix) for prefix in EVIDENCE_BLOCKLIST)
            for item in evidence
        ),
        "Blocklisted historical comparison cannot unlock gates",
    )
    require(
        all(item in known for item in evidence),
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
    # Durable transactions: a killed controller must not lose a reservation
    # or a terminal case record, or the budget could be double-spent.
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("PRAGMA synchronous=FULL")
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


def fingerprint_case(manifest, case):
    """Canonical identity of one case execution.

    Covers source, weights, the full contract (seed/quality/software/geometry
    family), the workload adapter, and the bounded time budget. Resume only
    counts evidence with an identical fingerprint; anything else is stale.
    """
    return digest(
        dict(
            source_sha=manifest["source_sha"],
            checkpoint=manifest["contract"]["checkpoint_manifest_sha256"],
            contract=manifest["contract"],
            adapter=case["adapter"],
            repeat_of=case["repeat_of"],
            samples=case["samples"],
            run_timeout_s=manifest["limits"]["run_timeout_s"],
        )
    )


def latest_cases(events, manifest=None):
    """Latest record per case id, oldest-wins reversed.

    With a manifest, completed evidence whose case fingerprint differs from
    the current manifest is demoted to ``stale``: it can never satisfy a
    dependency or a gate, and the case must re-run under the new identity.
    """
    latest = {item["case_id"]: item for item in events if item["kind"] == "case"}
    if manifest is None:
        return latest
    by_id = {case["id"]: case for case in manifest["cases"]}
    for case_id, record_item in latest.items():
        case = by_id.get(case_id)
        expected = fingerprint_case(manifest, case) if case else None
        if (
            record_item["status"] in ("measured", "diagnostic")
            and record_item.get("fingerprint") != expected
        ):
            latest[case_id] = dict(record_item, status="stale")
    return latest


def outstanding_charges(events):
    """Total device-seconds charged against the campaign budget.

    Every launch reserves the worst case (devices × run timeout). A terminal
    record reconciles the reservation down to the actual measured allocation
    only when cleanup was confirmed (``actual_device_s`` present); unknown
    cleanup or clock keeps the full worst-case charge.
    """
    total = 0
    reservations = {}
    for item in events:
        if item.get("kind") != "case":
            continue
        run_id = item.get("run_id")
        charge = item.get("charged_device_s")
        if item["status"] == "running" and charge is not None:
            reservations[run_id] = charge
        elif run_id in reservations:
            reserved = reservations.pop(run_id)
            actual = item.get("actual_device_s")
            total += reserved if actual is None else min(reserved, actual)
    return total + sum(reservations.values())


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
    source_root = Path(manifest["source_root"])
    require(source_root.is_dir() and Path(manifest["launcher"]).is_file(),
            "Pinned source worktree or launcher is missing")
    source = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain"]
    )
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
        H3_SOURCE_SHA=manifest["source_sha"],
        H3_SOURCE_ROOT=manifest["source_root"],
        H3_LAUNCHER=manifest["launcher"],
        H3_PROMPT=contract["prompt"],
        H3_NUM_GPUS=str(len(contract["devices"])),
        H3_LASER16="1",
        H3_PORT=str(contract["port"]),
        H3_PIN_CPU_MEMORY="false",
        H3_SEED=str(contract["seed"]),
        H3_QUALITY=contract["quality"],
        H3_PROFILE=str(int(REGISTRY[case["adapter"]][1])),
        H3_REPEATS=str(case["samples"]),
        SGLANG_PROFILE_RECORD_SHAPES="1",
        SGLANG_H3_NPU_ADALN_MEMO="0",
    )
    adapter = case["adapter"]
    if adapter in ("R01.PARITY.B0", "R01.PARITY.W0R"):
        env["H3_WINDOW_PARITY"] = "1"
    if adapter.startswith("R01.CROP."):
        env["H3_LASER_CROP_BEFORE_CAST"] = str(int(adapter.endswith(".BEFORE")))
    if adapter == "R05.NPUADALN.ON":
        env["SGLANG_H3_NPU_ADALN_MEMO"] = "1"
    if adapter == "R06.GRAPH.W0R":
        env["SGLANG_H3_VAE_NPU_GRAPH"] = "1"
    if adapter == "R08.DISPATCHLOG":
        env["SGLANG_H3_ATTN_DISPATCH_LOG"] = "1"
    if adapter == "R08.MEMOSTATS":
        env["SGLANG_H3_ADALN_MEMO_STATS"] = "1"
        env["SGLANG_H3_NPU_ADALN_MEMO"] = "1"
    if adapter == "R08.FFNLOG":
        env["SGLANG_H3_SELECTIVE_FFN_LOG"] = "1"
        env["SGLANG_H3_SELECTIVE_FFN_FRACTION"] = "0.5"
    if adapter == "R08.SPARSE.H08":
        env["H3_ATTENTION_BACKEND"] = "ascend_sparse_attn"
        # The pinned capability file doubles as the sparse policy config the
        # launcher requires; validation has already hash-pinned it.
        env["H3_ATTENTION_CONFIG"] = manifest["capabilities"]["r08_sparse"]["path"]
    if adapter in ("R02.Q5_FFN", "R02.Q5_ALLOWLIST"):
        env["H3_QUANT_CASE"] = "Q6" if adapter.endswith("FFN") else "Q7"
        env["H3_ATTENTION_BACKEND"] = "fa"
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


def results(directory, adapter, deadline, samples=1, source_sha=None):
    runs = list(directory.glob("run-*"))
    require(
        len(runs) == 1 and runs[0].is_dir(), "Expected one launcher artifact directory"
    )
    run = runs[0]
    comparison_path = run / "comparison.json"
    require(comparison_path.is_file(), "Missing execution identity receipt")
    comparison = json.loads(comparison_path.read_text())
    if source_sha is not None:
        require(
            comparison.get("source_sha", "").strip() == source_sha,
            "Execution source SHA differs from the approved branch",
        )
    records = [
        json.loads(line) for line in (run / "samples.jsonl").read_text().splitlines()
    ]
    matches = json.loads((run / "denoise_matches.json").read_text())
    require(
        len(records) == samples + 1
        and records[0]["warmup"] is True
        and all(row["warmup"] is False for row in records[1:])
        and all(row["response"]["status"] == "completed" for row in records),
        "Expected one warmup and the requested completed samples",
    )
    request_ids = {row["response"]["id"] for row in records[1:]}
    require(
        len(request_ids) == samples and request_ids == matches.keys(),
        "Missing or duplicate denoise evidence",
    )
    for match in matches.values():
        validate_denoise_match(match, adapter)
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
    client_samples = [row["e2e_s"] for row in records[1:]]
    require(
        all(row["profiled"] == profile for row in records[1:])
        and all(math.isfinite(value) and value > 0 for value in client_samples),
        "Invalid measured clock",
    )
    if profile:
        require(
            any(path.is_file() for path in (run / "profiles").rglob("*")),
            "Missing device profiler artifacts",
        )
    evidence = dict(
        artifacts=str(run),
        output_sha256=hashes,
        quality="review_required",
        request_ids=sorted(request_ids),
        **aggregate_samples(client_samples),
    )
    if profile:
        # Diagnostics are scope-separated: no client-seconds claim at all.
        evidence["median_client_s"] = None
        evidence["p90_client_s"] = None
    return evidence


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
            for value in readiness(manifest, latest_cases(events, manifest)).values()
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
        ports_available((contract["port"], contract["port"] + 1))
        preflight(manifest, directory, deadline)
        supervisor_failures = 0
        while not stop.exists():
            events = history(database)
            latest = latest_cases(events, manifest)
            remaining = limits["max_device_s"] - outstanding_charges(events)
            runs_used = sum(
                1 for item in events if item.get("kind") == "case" and item["status"] == "running"
            )
            remaining_runs = limits["max_runs"] - runs_used
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
                    supervisor_failures = 0
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
                    supervisor_failures += 1
                    record(
                        database,
                        "decision",
                        accepted=False,
                        snapshot_id=current["snapshot_id"],
                        reason=type(error).__name__,
                        fallback=True,
                        consecutive_failures=supervisor_failures,
                    )
                    if supervisor_failures >= SUPERVISOR_FALLBACK_LIMIT:
                        record(
                            database,
                            "pause",
                            reason="supervisor endpoint unavailable",
                        )
                        return
                    # Bounded fallback: run the deterministic queue head.
                    action, case_id = chosen["action"], chosen["case_id"]
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
                manifest["launcher"],
                "--inside",
                manifest["source_root"],
                REGISTRY[case["adapter"]][0],
                str(output / "run"),
            ]
            attempt = 0
            while True:
                run_id = uuid.uuid4().hex
                record(
                    database,
                    "case",
                    case_id=case_id,
                    status="running",
                    run_id=run_id,
                    fingerprint=fingerprint_case(manifest, case),
                    attempt=attempt,
                    charged_device_s=charge,
                    artifacts=str(output),
                    argv=argv,
                    action=action,
                )
                run_started = time.monotonic()
                status = "blocked"
                evidence = dict(artifacts=str(output))
                cleanup_confirmed = False
                try:
                    visible = check_devices(env, manifest)
                    save(output / "device_preflight.json", visible)
                    timeout = limits["run_timeout_s"] - (time.monotonic() - run_started)
                    status = run_launcher(
                        argv, env, output / "launcher.log", timeout, stop, lock_fds
                    )
                    if status == "completed":
                        evidence = results(
                            output,
                            case["adapter"],
                            run_started + limits["run_timeout_s"],
                            case["samples"],
                            manifest["source_sha"],
                        )
                        status = (
                            "diagnostic" if REGISTRY[case["adapter"]][1] else "measured"
                        )
                    cleanup_confirmed = status in (
                        "completed",
                        "cancelled",
                        "failed",
                        "timed_out",
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
                    cleanup_confirmed = status == "failed"
                wall_s = time.monotonic() - run_started
                terminal = dict(evidence)
                terminal["run_id"] = run_id
                terminal["fingerprint"] = fingerprint_case(manifest, case)
                terminal["attempt"] = attempt
                if cleanup_confirmed:
                    # Confirmed termination: reconcile the worst-case
                    # reservation down to the measured allocation.
                    terminal["actual_device_s"] = math.ceil(
                        len(contract["devices"]) * wall_s
                    )
                record(database, "case", case_id=case_id, status=status, **terminal)
                if status in ("measured", "diagnostic"):
                    break
                failure = classify_failure(terminal.get("error"), status)
                if failure == "cleanup":
                    record(
                        database,
                        "pause",
                        reason="cleanup unverified; operator must confirm the allocation",
                        case_id=case_id,
                    )
                    return
                if failure in ("oom", "correctness"):
                    record(
                        database,
                        "case",
                        case_id=case_id,
                        status="blocked",
                        run_id=run_id,
                        fingerprint=fingerprint_case(manifest, case),
                        blocked_reason=failure,
                    )
                    return
                if (
                    stop.exists()
                    or time.time() + limits["run_timeout_s"]
                    > started_at + limits["max_wall_s"]
                ):
                    record(
                        database,
                        "pause",
                        reason="Stop requested or wall budget exhausted",
                    )
                    return
                if attempt >= 1:
                    record(
                        database,
                        "pause",
                        reason=f"{case_id} failed twice; track exhausted",
                        case_id=case_id,
                    )
                    return
                remaining_after = limits["max_device_s"] - outstanding_charges(
                    history(database)
                )
                if charge > remaining_after:
                    record(
                        database,
                        "pause",
                        reason="retry does not fit the remaining budget",
                        case_id=case_id,
                    )
                    return
                attempt += 1
                record(database, "retry", case_id=case_id, attempt=attempt)


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
