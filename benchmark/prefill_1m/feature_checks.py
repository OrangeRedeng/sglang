"""Feature activation, numerical comparisons and bounded report excerpts."""

import json
import math
import re


def evidence(log):
    events = []
    for line in log.splitlines():
        if "NPU_FEATURE_EVIDENCE " in line:
            events.append(json.loads(line.split("NPU_FEATURE_EVIDENCE ", 1)[1]))
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
    print(
        "PASS: long cache regimes, flush/prime lifecycle, exact token lengths, benchmark commands and parity failure"
    )
