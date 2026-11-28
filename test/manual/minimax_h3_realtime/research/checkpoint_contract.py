# SPDX-License-Identifier: Apache-2.0
"""Verified local checkpoint identity for MiniMax-H3 research launches."""

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

SCHEDULES = {
    "flashgen4": (4, (1000, 700, 400, 150), 12, 3),
    "turbo8": (8, tuple(range(1000, 0, -125)), 6, 3),
    "native50": (50, (), 12, 3),
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def absolute(value):
    return isinstance(value, str) and Path(value).is_absolute()


def validate(manifest):
    family = manifest["family"]
    require(
        manifest["version"] == 1 and family in SCHEDULES, "Unknown checkpoint contract"
    )
    steps, rungs, video, audio = SCHEDULES[family]
    require(
        manifest["schedule"]
        == dict(steps=steps, rungs=list(rungs), video_shift=video, audio_shift=audio),
        "Checkpoint schedule conflicts with its family",
    )
    require(
        absolute(manifest["model"])
        and manifest["variant"] == "fl2va"
        and manifest["dtype"] == "bf16",
        "Pin a local FL2VA checkpoint and BF16 baseline dtype",
    )
    files = manifest["files"]
    require(
        isinstance(files, dict)
        and bool(files)
        and all(
            name == Path(name).as_posix()
            and not Path(name).is_absolute()
            and not {"..", "."}.intersection(Path(name).parts)
            and name != "."
            and is_sha(sha)
            for name, sha in files.items()
        )
        and "FL2VA/model_index.json" in files
        and any(
            name.startswith("FL2VA/transformer/") and name.endswith(".safetensors")
            for name in files
        )
        and digest(files) == manifest["checkpoint_sha256"],
        "Invalid checkpoint content inventory",
    )
    base = manifest["clean_base_sha256"]
    require(is_sha(base), "Pin the clean base content identity")
    merge = manifest["merge"]
    if family == "native50":
        require(
            merge is None and base == digest(files),
            "Native reference must be clean and unadapted",
        )
    else:
        require(
            isinstance(merge, dict)
            and base != digest(files)
            and all(
                isinstance(merge.get(key), str) and merge[key].strip()
                for key in ("adapter", "recipe", "tool_revision")
            )
            and is_sha(merge.get("adapter_sha256"))
            and type(merge.get("alpha")) is int
            and merge["alpha"] > 0
            and type(merge.get("scale")) in (int, float)
            and math.isfinite(merge["scale"])
            and merge["scale"] > 0,
            "Pin a single offline merge from the clean base, including adapter hash, alpha and scale",
        )
    decoder = manifest["decoder"]
    require(
        isinstance(decoder, dict)
        and (
            decoder == {"kind": "native"}
            or (
                decoder.get("kind") == "lightvae"
                and absolute(decoder.get("path"))
                and is_sha(decoder.get("sha256"))
            )
        ),
        "Pin the native decoder or a local LightVAE file and content hash",
    )


def snapshot(manifest):
    model = Path(manifest["model"])
    require(
        model.is_dir() and not model.is_symlink(), "Checkpoint directory is missing"
    )
    files = {}
    for path in model.rglob("*"):
        require(not path.is_symlink(), f"Checkpoint must be self-contained: {path}")
        if path.is_file():
            files[path.relative_to(model).as_posix()] = path
    require(
        files.keys() == manifest["files"].keys(),
        "Checkpoint file set differs from inventory",
    )
    for name in ("model_index.json", "FL2VA/model_index.json"):
        if name in files:
            index = json.loads(files[name].read_text())
            release = index.get("_minimax_h3") or {}
            require(
                not release.get("lora_adapter"),
                "Checkpoint declares a runtime release adapter",
            )
            rungs = manifest["schedule"]["rungs"]
            embedded = release.get("dmd_denoising_steps")
            require(
                embedded is None or bool(rungs) and embedded == rungs,
                "Release metadata conflicts with checkpoint schedule",
            )
    decoder = manifest["decoder"]
    if decoder["kind"] == "lightvae":
        files["/lightvae"] = Path(decoder["path"])
    states = {}
    for name, path in files.items():
        require(path.is_file(), f"Missing checkpoint file: {path}")
        stat = path.stat()
        states[name] = [
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        ]
    return files, states


def load_verified(manifest_path, receipt_path, *, verify=False, expected_family=None):
    manifest = json.loads(Path(manifest_path).read_text())
    validate(manifest)
    if expected_family is not None:
        require(manifest["family"] == expected_family, "Incorrect reference family")
    if manifest["family"] == "turbo8":
        # Bind the merge input to the verified native inventory, independently of the merged output.
        for family in ("native50", "flashgen4"):
            reference = manifest["references"][family]
            require(
                absolute(reference["manifest"]) and absolute(reference["receipt"]),
                "Pin absolute reference paths",
            )
            verified = load_verified(
                reference["manifest"], reference["receipt"], expected_family=family
            )
            require(
                verified["clean_base_sha256"] == manifest["clean_base_sha256"],
                "Turbo8 and FlashGen4 must derive from the verified Native50 base",
            )
            require(
                verified["checkpoint_sha256"] != manifest["checkpoint_sha256"],
                "Turbo8 must have a distinct checkpoint inventory",
            )
            if family == "flashgen4":
                require(
                    verified["merge"]["adapter_sha256"]
                    != manifest["merge"]["adapter_sha256"],
                    "Turbo8 and FlashGen4 must use distinct adapters",
                )
    files, states = snapshot(manifest)
    receipt = dict(manifest_sha256=digest(manifest), files=states)
    if verify:
        for name, path in files.items():
            expected = (
                manifest["decoder"]["sha256"]
                if name == "/lightvae"
                else manifest["files"][name]
            )
            require(sha256_file(path) == expected, f"Content hash mismatch: {path}")
        require(
            snapshot(manifest)[1] == states, "Checkpoint changed during verification"
        )
        with Path(receipt_path).open("x") as output:
            json.dump(receipt, output, sort_keys=True)
    else:
        require(
            json.loads(Path(receipt_path).read_text()) == receipt,
            "Checkpoint or decoder changed since verification",
        )
    return manifest


def reject_runtime_lora(argv=(), environment=None):
    environment = os.environ if environment is None else environment
    require(
        not any(key.startswith(("H3_LORA", "SGLANG_LORA")) for key in environment)
        and not any(
            arg.replace("_", "-").startswith(("--lora", "--runtime-lora"))
            for arg in argv
        ),
        "Runtime LoRA is prohibited in MiniMax-H3 research",
    )


def require_launcher(argv):
    launcher = Path(__file__).with_name("run_a3_windowing.sh").resolve()
    offset = int(bool(argv) and argv[0] == "bash")
    require(
        len(argv) > offset and Path(argv[offset]).resolve() == launcher,
        "Use run_a3_windowing.sh for managed research servers",
    )


def model_id(manifest_hash):
    require(is_sha(manifest_hash), "Pin a verified checkpoint manifest SHA-256")
    return f"h3-research-{manifest_hash}"


def validate_request(request, family, served_model):
    steps, _, video, audio = SCHEDULES[family]
    require(
        not any("lora" in key.lower() for key in request),
        "Runtime LoRA is prohibited in research requests",
    )
    require(
        (
            request.get("model"),
            request.get("num_inference_steps"),
            request.get("flow_shift"),
            request.get("audio_flow_shift"),
        )
        == (served_model, steps, video, audio),
        "Request identity or schedule conflicts with verified checkpoint",
    )


def assert_endpoint(client, served_model):
    models = client.json("GET", "/v1/models")
    status = client.json("GET", "/v1/list_loras")
    require(
        [card["id"] for card in models["data"]] == [served_model],
        "Unexpected endpoint checkpoint",
    )
    require(
        status.get("active") == {} and status.get("loaded_adapters") == [],
        "Unexpected or unverified endpoint adapter state",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "verify", "launch"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("receipt", type=Path, nargs="?")
    args = parser.parse_args()
    try:
        if args.action == "inventory":
            model = args.manifest
            require(
                model.is_dir() and model.is_absolute(),
                "Supply an absolute checkpoint directory",
            )
            paths = sorted(model.rglob("*"))
            require(
                not any(path.is_symlink() for path in paths),
                "Materialize checkpoint symlinks first",
            )
            files = {
                path.relative_to(model).as_posix(): sha256_file(path)
                for path in paths
                if path.is_file()
            }
            print(
                json.dumps(dict(files=files, checkpoint_sha256=digest(files)), indent=2)
            )
            return
        if args.receipt is None:
            parser.error("verify and launch require a verification receipt path")
        manifest = load_verified(
            args.manifest, args.receipt, verify=args.action == "verify"
        )
        identity = digest(manifest)
        if args.action == "verify":
            print(identity)
            return
        reject_runtime_lora()
        family = manifest["family"]
        steps, rungs, video, audio = SCHEDULES[family]
        rung_string = ",".join(map(str, rungs))
        decoder = manifest["decoder"].get("path", "")
        for key, expected in dict(
            H3_EXPECTED_CHECKPOINT_SHA=identity,
            H3_MODEL=manifest["model"],
            H3_LIGHTVAE=decoder,
            H3_STEPS=str(steps),
            H3_FLOW_SHIFT=str(video),
            H3_NATIVE_VAE=str(int(not decoder)),
            H3_NATIVE_REFERENCE=str(int(family == "native50")),
            H3_MERGED_FLASHGEN=str(int(family == "flashgen4")),
            SGLANG_H3_DENOISE_RUNGS=rung_string,
        ).items():
            require(
                os.environ.get(key, expected) == expected,
                f"{key} conflicts with checkpoint manifest",
            )
        print(
            "\n".join(
                map(
                    str,
                    (
                        manifest["model"],
                        decoder,
                        family,
                        steps,
                        rung_string,
                        video,
                        audio,
                        identity,
                    ),
                )
            )
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(2, f"Checkpoint contract: {error}\n")


if __name__ == "__main__":
    main()
