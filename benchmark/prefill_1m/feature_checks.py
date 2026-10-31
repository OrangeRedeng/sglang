"""Feature activation, numerical comparisons and bounded report excerpts."""

import hashlib
import json
import math
import re
from pathlib import Path


def checkpoint_identity(model_path):
    root = Path(model_path)
    hashes = {}
    for name in (
        "config.json",
        "quant_model_description.json",
        "model.safetensors.index.json",
    ):
        path = root / name
        if path.is_file():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    config = root / "config.json"
    return dict(
        path=str(root.resolve()),
        metadata_sha256=hashes,
        config=json.loads(config.read_text()) if config.is_file() else None,
    )


def expert_quantization(model_path):
    path = Path(model_path) / "quant_model_description.json"
    if not path.is_file():
        return set()
    description = json.loads(path.read_text())
    return {
        value
        for name, value in description.items()
        if ".experts." in name
        and name.endswith(".weight")
        and name.rsplit(".", 2)[-2] in {"gate_proj", "up_proj", "down_proj"}
        and isinstance(value, str)
    }


def serving_failure(error, directory, phase):
    directory = Path(directory)
    log = directory / "server.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    capacity = bool(re.search(r"(?:OutOfMemoryError|out of memory)", text, re.I))
    return dict(
        status="capacity_blocked"
        if capacity
        else "blocked"
        if phase == "reference"
        else "fail",
        failure_phase=phase,
        failure_kind="capacity" if capacity else "runtime",
        error=str(error),
        artifacts=str(directory.resolve()),
    )


def benchmark_qualified(result, *, baseline=False):
    return (
        result.get("status") == "pass"
        and result.get("quality") == "verified"
        and not result.get("missing")
        and (baseline or result.get("activation") == "verified")
    )


def evidence(log):
    events = []
    for line in log.splitlines():
        match = re.search(r"(?:^|\s)NPU_FEATURE_EVIDENCE (.*)$", line)
        if match is None:
            continue
        event = json.loads(match.group(1))
        if not isinstance(event, dict) or not isinstance(event.get("feature"), str):
            raise ValueError("malformed NPU feature evidence event")
        events.append(event)
    return events


def compare_responses(reference, actual, atol=0.01, rtol=0.01):
    if len(reference) != len(actual):
        raise AssertionError("parity response count differs")
    errors = []
    for before, after in zip(reference, actual):
        if "meta_info" not in before:
            continue
        a = before["meta_info"].get("input_token_logprobs")
        b = after["meta_info"].get("input_token_logprobs")
        if not a or not b or len(a) != len(b):
            raise AssertionError("teacher-forced input logprobs missing or mismatched")
        compared = 0
        for left, right in zip(a, b):
            if left[0] is None and right[0] is None:
                continue
            if left[1] != right[1] or left[0] is None or right[0] is None:
                raise AssertionError("teacher-forced token identities differ")
            x, y = float(left[0]), float(right[0])
            if not math.isfinite(x) or not math.isfinite(y):
                raise AssertionError("non-finite logprob")
            error = abs(x - y)
            if error > atol + rtol * abs(x):
                raise AssertionError(f"input logprob parity failed: {x} versus {y}")
            errors.append(error)
            compared += 1
        if not compared:
            raise AssertionError("no comparable teacher-forced logprobs")
    if not errors:
        raise AssertionError("no numerical parity evidence")
    return dict(compared=len(errors), max_abs_error=max(errors), atol=atol, rtol=rtol)


def bounded_log(text, max_lines=180):
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    chosen = set(range(15)) | set(range(len(lines) - 45, len(lines)))
    signatures = set()
    for i, line in enumerate(lines):
        if re.search(
            r"(?:Error:|Exception:|NPU_FEATURE_EVIDENCE |Scheduler hit|from user code:)",
            line,
        ):
            signature = re.sub(r"\d+", "#", line)[-220:]
            if signature in signatures:
                continue
            signatures.add(signature)
            chosen.update(range(max(0, i - 5), min(len(lines), i + 4)))
            if len(chosen) >= max_lines:
                break
    output = []
    previous = -2
    for i in sorted(chosen)[:max_lines]:
        if i > previous + 1:
            output.append("... omitted ...")
        output.append(lines[i])
        previous = i
    return "\n".join(output)


def probe_feature(name):
    import torch
    import torch_npu  # noqa: F401

    if name == "megamoe":
        from sglang.srt.hardware_backend.npu.moe.fuseep import _check_mega_moe_support

        _check_mega_moe_support()
    elif name.startswith("fuseep-"):
        from deep_ep import Buffer

        if not callable(getattr(Buffer, "fused_deep_moe", None)):
            raise RuntimeError("installed DeepEP lacks fused_deep_moe")
    elif name == "tp-finalize":
        from sglang.srt.hardware_backend.npu.moe.tp_fusion import supports_npu_op

        required = ("scale", "pertoken_scale", "row_index", "output_bs")
        if not supports_npu_op(
            "npu_grouped_matmul_finalize_routing", required
        ) or not supports_npu_op("npu_moe_init_routing_v2", ("row_idx_type",)):
            raise RuntimeError("installed operators lack the GMM2 finalize/routing ABI")
    elif name == "indexer-int8":
        if not hasattr(torch.ops.npu, "npu_quant_lightning_indexer"):
            raise RuntimeError("installed torch_npu lacks the INT8 LightningIndexer")


def self_check_workloads():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    from unittest.mock import patch

    from feature_benchmarks import long_matrix, percentile, small_commands

    # Regression: environment logging used to abort every serving reference.
    assert evidence("[TP0] get env SGLANG_NPU_FEATURE_EVIDENCE = True") == []
    assert evidence('[TP0] NPU_FEATURE_EVIDENCE {"feature": "model-loaded"}') == [
        {"feature": "model-loaded"}
    ]
    for broken in ("NPU_FEATURE_EVIDENCE {", 'NPU_FEATURE_EVIDENCE {"pid": 1}'):
        try:
            evidence(broken)
        except (ValueError, json.JSONDecodeError):
            pass
        else:
            raise AssertionError("malformed genuine evidence was accepted")

    args = SimpleNamespace(
        port=31000,
        small_lengths=[512, 4096],
        small_concurrency=[1, 8],
        small_requests=32,
        benchmark_repeats=2,
        long_lengths=[1000],
        long_concurrency=[1, 2],
        long_requests=4,
        context_length=1032,
        request_timeout=30,
    )
    with TemporaryDirectory() as temp:
        directory = Path(temp)
        (directory / "config.json").write_text(json.dumps({"vocab_size": 1024}))
        state = {"prefix": [], "flushed": 0, "measured": []}

        def request(url, *args):
            if url.endswith("/get_server_info"):
                return {
                    "internal_states": [
                        {"max_total_num_tokens": 10000, "max_req_input_len": 1032}
                    ]
                }
            state["prefix"] = []
            state["flushed"] += 1

        def send(base, ids, *args):
            cached = (
                len(state["prefix"])
                if ids[: len(state["prefix"])] == state["prefix"]
                else 0
            )
            if len(ids) == 990:
                state["prefix"] = list(ids)
            if len(ids) == 1000:
                state["measured"].append(list(ids))
            return {"ttft_s": 0.1, "e2e_s": 0.2, "cached_tokens": cached}

        with patch("feature_benchmarks.run_one", side_effect=send):
            points = long_matrix(args, "local", request, directory, directory)
        assert len(points) == 4 and all(point["status"] == "pass" for point in points)
        assert state["flushed"] == 8
        assert all(
            len(ids) == 1000 and min(ids) >= 256 and max(ids) < 1024
            for ids in state["measured"]
        )
        commands = list(small_commands(args, directory))
        assert len(commands) == 8 and all(
            "--flush-cache" in command and "--tokenize-prompt" in command
            for command, _ in commands
        )
        assert percentile([1, 2, 3], 0.5) == 2
        reference = [{"meta_info": {"input_token_logprobs": [[None, 1], [0.2, 2]]}}]
        assert compare_responses(reference, reference)["compared"] == 1
        try:
            compare_responses(
                reference,
                [{"meta_info": {"input_token_logprobs": [[None, 1], [1.0, 2]]}}],
            )
        except AssertionError:
            pass
        else:
            raise AssertionError("bad numerical parity accepted")
    import importlib.util

    runner_path = Path(__file__).with_name("test_glm53_features.py")
    spec = importlib.util.spec_from_file_location(
        "workload_validation_runner", runner_path
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    with TemporaryDirectory() as temp:
        directory = Path(temp)
        case = dict(
            id="long-baseline",
            stage="long",
            scenario={"dcp": True},
            command=[],
            checkpoint_path=str(directory),
            checkpoint_format="bf16",
        )
        launch_args = SimpleNamespace(
            model_path=directory,
            checkpoint_format="bf16",
            tp=16,
            dcp=16,
            context_length=8192,
            chunk=4096,
            mem_fraction=0.85,
            port=31000,
        )

        def serve(args, case, path, env):
            (path / "responses.json").write_text(json.dumps(reference))
            return dict(status="smoke_pass", benchmarks=[{"status": "pass"}])

        with patch.object(runner, "serve", side_effect=serve):
            result = runner.validate_serving(launch_args, case, directory, {}, {})
        assert result["status"] == "pass" and result["benchmarks"]
        from feature_benchmarks import CapacityBlocked

        def no_capacity(args, case, path, env):
            if case["stage"] == "long":
                raise CapacityBlocked("KV capacity unavailable")
            return serve(args, case, path, env)

        with patch.object(runner, "serve", side_effect=no_capacity):
            result = runner.validate_serving(launch_args, case, directory, {}, {})
        assert result["status"] == "capacity_blocked"
    print(
        "PASS: long cache regimes, flush/prime lifecycle, exact token lengths, benchmark commands and parity failure"
    )
