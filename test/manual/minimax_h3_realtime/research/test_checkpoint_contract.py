# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
from pathlib import Path

import pytest

from checkpoint_contract import (
    SCHEDULES,
    assert_endpoint,
    digest,
    load_verified,
    model_id,
    sha256_file,
    validate_request,
)

RESEARCH = Path(__file__).resolve().parent


def make_checkpoint(root, family="flashgen4"):
    model = root / family
    weights = model / "FL2VA/transformer/model.safetensors"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"test-only synthetic checkpoint")
    index = model / "FL2VA/model_index.json"
    index.write_text("{}")
    files = {
        path.relative_to(model).as_posix(): sha256_file(path)
        for path in (weights, index)
    }
    steps, rungs, video, audio = SCHEDULES[family]
    native = family == "native50"
    manifest = dict(
        version=1,
        family=family,
        model=str(model),
        files=files,
        checkpoint_sha256=digest(files),
        clean_base_sha256=digest(files) if native else "a" * 64,
        decoder=dict(kind="native"),
        variant="fl2va",
        dtype="bf16",
        merge=None
        if native
        else dict(
            adapter="adapter",
            adapter_sha256="b" * 64,
            recipe="merge once",
            tool_revision="tool-sha",
            alpha=64,
            scale=1,
        ),
        schedule=dict(
            steps=steps, rungs=list(rungs), video_shift=video, audio_shift=audio
        ),
    )
    spec = root / f"{family}.json"
    spec.write_text(json.dumps(manifest))
    receipt = root / f"{family}.receipt.json"
    load_verified(spec, receipt, verify=True)
    return spec, receipt, digest(manifest)


def test_receipt_rejects_changed_weights_and_schedule(tmp_path):
    spec, receipt, _ = make_checkpoint(tmp_path)
    manifest = load_verified(spec, receipt)
    path = Path(manifest["model"]) / "FL2VA/transformer/model.safetensors"
    path.write_bytes(b"another test checkpoint")
    with pytest.raises(ValueError, match="changed since verification"):
        load_verified(spec, receipt)
    manifest["schedule"]["steps"] = 8
    spec.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schedule conflicts"):
        load_verified(spec, receipt)


@pytest.mark.parametrize("family", ["flashgen4", "turbo8", "native50"])
def test_launcher_argv_and_runtime_lora_rejection(tmp_path, monkeypatch, family):
    spec, receipt, identity = make_checkpoint(tmp_path, family)
    sglang = tmp_path / "sglang"
    sglang.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    sglang.chmod(0o700)
    ffprobe = tmp_path / "ffprobe"
    ffprobe.symlink_to(sglang)
    import os
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = str(probe.getsockname()[1])
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ.get('PATH', '')}")
    monkeypatch.setenv("H3_CHECKPOINT_MANIFEST", str(spec))
    monkeypatch.setenv("H3_CHECKPOINT_RECEIPT", str(receipt))
    monkeypatch.setenv("H3_SERVE_ONLY", "1")
    monkeypatch.setenv("H3_PORT", port)
    root = RESEARCH.parents[3]
    command = [
        "bash",
        str(RESEARCH / "run_a3_windowing.sh"),
        "--inside",
        str(root),
        "B0",
        str(tmp_path / "case"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert model_id(identity) in result.stdout.splitlines()
    assert "--lora" not in result.stdout
    args = result.stdout.splitlines()
    assert args[args.index("--warmup-steps") + 1] == str(SCHEDULES[family][0])

    monkeypatch.setenv("H3_LORA_SCALE", "0")
    rejected = subprocess.run(command, capture_output=True, text=True)
    assert rejected.returncode == 2
    assert "Runtime LoRA is prohibited" in rejected.stderr


def test_endpoint_and_request_fail_closed():
    identity = model_id("e" * 64)

    class Client:
        def json(self, method, path):
            if path == "/v1/models":
                return {"data": [{"id": identity}]}
            return {"active": {"transformer": ["old"]}, "loaded_adapters": []}

    with pytest.raises(ValueError, match="adapter state"):
        assert_endpoint(Client(), identity)
    request = dict(
        model=identity, num_inference_steps=4, flow_shift=12, audio_flow_shift=3
    )
    validate_request(request, "flashgen4", identity)
    with pytest.raises(ValueError, match="schedule conflicts"):
        validate_request(request, "turbo8", identity)
    with pytest.raises(ValueError, match="Runtime LoRA"):
        validate_request(dict(request, lora_scale=0), "flashgen4", identity)
