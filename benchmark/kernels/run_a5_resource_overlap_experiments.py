"""Fresh-server A5 resource scheduling experiments; profiles never rank TTFT.

Reuse autotune_npu_prefill's server lifecycle and command-file interface. The
benchmark command must emit the standard SGLang summary for the fixed workload.
Command files receive AUTOTUNE_PORT, AUTOTUNE_RESULT_DIR and PROFILE_DIR.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shlex
import statistics
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "a5_restart_runner", Path(__file__).with_name("autotune_npu_prefill.py")
)
restart = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(restart)

BASE = {
    "HCCL_BUFFSIZE": None,
    "TASK_QUEUE_ENABLE": None,
    "HCCL_OP_EXPANSION_MODE": None,
    "SGLANG_NPU_TP_MOE_FUSE_ROUTED_SCALE": "1",
    "SGLANG_NPU_TP_MOE_FUSE_SHARED_EXPERT": "1",
    "SGLANG_NPU_TP_MOE_FUSE_GMM2_FINALIZE": "0",
    "SGLANG_NPU_TP_MOE_PREQUANT_INPUT": "1",
    "SGLANG_NPU_TP_MOE_REUSE_MXFP8": "1",
    "SGLANG_NPU_TP_MOE_SHARED_GMM1_MODE": "grouped_fused",
    "SGLANG_NPU_TP_MOE_FUSED_SHARED_EXPERT": "0",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "SGLANG_NPU_TP_MOE_EAGER_MULTI_STREAM": "0",
    "SGLANG_NPU_TP_MOE_SHARED_STREAM_START": "pre_gate",
    "SGLANG_NPU_TP_MOE_SHARED_PIPELINE": "legacy",
    "SGLANG_NPU_DSA_OVERLAP_QPROJ_KVNORM": "0",
    "SGLANG_NPU_DSA_OVERLAP_QPROJ_KVNORM_MIN_TOKENS": "0",
    "SGLANG_NPU_DSA_OVERLAP_QNOPE_ROPE": "0",
    "SGLANG_NPU_DSA_OVERLAP_QNOPE_ROPE_MIN_TOKENS": "0",
    "SGLANG_NPU_DSA_EAGER_INDEXER": "0",
    "SGLANG_NPU_DSA_EAGER_INDEXER_MIN_TOKENS": "0",
    "SGLANG_NPU_TP_MOE_EAGER_MULTI_STREAM_MIN_TOKENS": "0",
    "SGLANG_NPU_DSA_INDEXER_STREAM_MODE": "legacy",
    "SGLANG_NPU_DSA_INDEXER_HADAMARD_MODE": "matmul",
    "SGLANG_NPU_TP_MOE_MXFP8_GATE_TOPK_LAYOUT": "default",
    "SGLANG_NPU_ENABLE_DCP_EXTEND_GATHER_PREFETCH": "0",
    "SGLANG_NPU_RESOURCE_SCHED_DIAGNOSTICS": "1",
}
BEST = {**BASE, "HCCL_BUFFSIZE": "256", "TASK_QUEUE_ENABLE": "2"}
SINGLES = {
    "qproj": {"SGLANG_NPU_DSA_OVERLAP_QPROJ_KVNORM": "1"},
    "qnope": {"SGLANG_NPU_DSA_OVERLAP_QNOPE_ROPE": "1"},
    "qproj_qnope": {
        "SGLANG_NPU_DSA_OVERLAP_QPROJ_KVNORM": "1",
        "SGLANG_NPU_DSA_OVERLAP_QNOPE_ROPE": "1",
    },
    "indexer_inline": {"SGLANG_NPU_DSA_INDEXER_STREAM_MODE": "inline"},
    "indexer_resource": {"SGLANG_NPU_DSA_INDEXER_STREAM_MODE": "resource"},
    "eager_indexer_resource": {
        "SGLANG_NPU_DSA_INDEXER_STREAM_MODE": "resource",
        "SGLANG_NPU_DSA_EAGER_INDEXER": "1",
    },
    "shared_post_gate": {
        "SGLANG_NPU_TP_MOE_EAGER_MULTI_STREAM": "1",
        "SGLANG_NPU_TP_MOE_SHARED_STREAM_START": "post_gate",
    },
    "shared_resource": {"SGLANG_NPU_TP_MOE_SHARED_PIPELINE": "resource"},
    "dcp_prefetch": {"SGLANG_NPU_ENABLE_DCP_EXTEND_GATHER_PREFETCH": "1"},
    "gate_materialized_nd": {
        "SGLANG_NPU_TP_MOE_MXFP8_GATE": "1",
        "SGLANG_NPU_TP_MOE_MXFP8_GATE_TOPK_LAYOUT": "materialized_nd",
    },
}
SIGNATURE = {"completed": 17, "input_tokens": 3731608, "output_tokens": 17}


def workload_signature(output):
    result = {}
    for key, label in (
        ("completed", "Successful requests"),
        ("input_tokens", "Total input tokens"),
        ("output_tokens", "Total generated tokens"),
    ):
        matches = re.findall(re.escape(label) + r"\s*:\s*([\d,]+)", output, re.I)
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one workload field: {label}")
        result[key] = int(matches[0].replace(",", ""))
    if result != SIGNATURE:
        raise ValueError(f"Workload signature mismatch: {result} != {SIGNATURE}")
    return result


def bimodal(samples):
    if len(samples) < 3:
        return False
    ordered = sorted(samples)
    # Conservatively flag separated regimes rather than hide them in a median.
    return any(right / left > 1.025 for left, right in zip(ordered, ordered[1:]))


def summarize(pairs):
    valid = [p for p in pairs if all(r["status"] == "ok" for r in p)]
    if len(valid) != len(pairs):
        return {"status": "FAILED_OR_INELIGIBLE", "valid_pairs": len(valid)}
    values = [p[1]["runs"][0]["median_ttft"] for p in valid]
    anchors = [
        (p[0]["runs"][0]["median_ttft"] + p[2]["runs"][0]["median_ttft"]) / 2
        for p in valid
    ]
    if not values:
        return {"status": "UNMEASURED"}
    gain = 100 * (1 - statistics.median(values) / statistics.median(anchors))
    wins = sum(value < anchor for value, anchor in zip(values, anchors))
    correct = all(r["correctness"] for p in valid for r in p)
    unstable = bimodal(values) or bimodal(anchors)
    return {
        "status": "BIMODAL" if unstable else "MEASURED",
        "candidate_median_ttft_ms": statistics.median(values),
        "interpolated_anchor_median_ttft_ms": statistics.median(anchors),
        "paired_gain_pct": gain,
        "paired_wins": wins,
        "candidate_runs": len(values),
        "correctness_passed": correct,
        "nonregression": (
            not unstable and gain >= 0 and wins * 3 >= len(values) * 2 and correct
        ),
        "promotion_eligible": (
            not unstable and len(values) >= 3 and gain >= 0.5
            and wins * 3 >= len(values) * 2 and correct
        ),
    }


def engagement(settings, server_log):
    for flag, message in (
        ("SGLANG_NPU_DSA_OVERLAP_QPROJ_KVNORM", "DSA qproj/KV norm overlap is ACTIVE"),
        ("SGLANG_NPU_DSA_EAGER_INDEXER", "DSA eager indexer is ACTIVE"),
        ("SGLANG_NPU_ENABLE_DCP_EXTEND_GATHER_PREFETCH", "DCP extend gather prefetch is ACTIVE"),
    ):
        if settings.get(flag) == "1" and message not in server_log:
            return f"Requested path did not engage: {flag}"
    if settings.get("SGLANG_NPU_DSA_INDEXER_STREAM_MODE") == "resource":
        if "DSA indexer resource stream is ACTIVE" not in server_log:
            return "Requested indexer resource path did not engage"
    if settings.get("SGLANG_NPU_TP_MOE_SHARED_PIPELINE") == "resource":
        if "TP shared resource pipeline is ACTIVE" not in server_log:
            return "Requested shared resource path did not engage"
    return None


def run(args, argv, name, settings, directory, *, profile=False):
    case_args = argparse.Namespace(**vars(args))
    case_args.analyzer_command_file = args.analyzer_command_file if profile else None
    if profile:
        case_args.benchmark_command_file = args.profile_command_file
    record = restart.run_case(case_args, argv, settings, directory, 1)
    record.update(name=name, profiled=profile)
    try:
        if record["status"] == "ok":
            record["signature"] = workload_signature(
                (directory / "benchmark-0.log").read_text()
            )
            reason = engagement(settings, (directory / "server.log").read_text())
            if reason:
                record.update(status="ineligible", error=reason)
    except (OSError, ValueError) as exc:
        record.update(status="failed", error=str(exc))
    (directory / "result.json").write_text(json.dumps(record, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--server-args-file", type=Path, required=True)
    parser.add_argument("--benchmark-command-file", type=Path, required=True)
    parser.add_argument("--correctness-command-file", type=Path)
    parser.add_argument("--profile-command-file", type=Path)
    parser.add_argument("--analyzer-command-file", type=Path)
    parser.add_argument("--profile-mode", choices=("none", "best", "all"), default="none")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", choices=("A", "B", "C", "sweep"), default=["A", "B", "C"])
    parser.add_argument("--cases", nargs="+", choices=tuple(SINGLES), default=list(SINGLES))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30088)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--benchmark-timeout", type=float, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.expected_requests = 17
    if args.runs < 3:
        parser.error("at least three fresh-server candidate runs are required")
    if args.profile_mode != "none" and not args.profile_command_file:
        parser.error("profiling requires --profile-command-file (separate from unprofiled benchmark)")
    if "C" in args.stages and "B" not in args.stages:
        parser.error("Stage C requires measured Stage B in the same invocation")
    argv = shlex.split(args.server_args_file.read_text(), comments=True)
    # Preserve topology/geometry; refuse mismatches instead of silently tuning.
    for option, expected in (
        ("--tp-size", "4"), ("--ep-size", "1"),
        ("--chunked-prefill-size", "16384"), ("--max-prefill-tokens", "16384"),
        ("--page-size", "128"),
    ):
        value = restart.option_value(argv, option)
        if value is not None and value != expected:
            parser.error(f"Target workload requires {option}={expected}; got {value}")
        argv = restart.set_option(argv, option, expected)
    kv_dtype = restart.option_value(argv, "--kv-cache-dtype")
    if kv_dtype is None or not kv_dtype.startswith("fp8"):
        parser.error("server args must explicitly select the checkpoint's FP8 KV cache dtype")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summaries, settings_by_name = {}, {"A_BASE": BASE, "BEST_BASE": BEST}

    def experiment(name, settings, baseline):
        settings_by_name[name] = settings
        if args.dry_run:
            summaries[name] = {"status": "DRY_RUN", "settings": settings, "anchor": baseline}
            return
        pairs = []
        for repetition in range(args.runs):
            trial = args.output_dir / name / str(repetition)
            pairs.append(tuple(
                run(args, argv, label, config, trial / suffix)
                for suffix, label, config in (
                    ("before", "anchor", baseline),
                    ("candidate", name, settings),
                    ("after", "anchor", baseline),
                )
            ))
        summaries[name] = summarize(pairs)
        (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2))

    if "A" in args.stages:
        experiment("BEST_BASE", BEST, BASE)
    if "B" in args.stages:
        for name in args.cases:
            # Layout-only gate hypothesis must compare against an enabled gate
            # control, not conflate a gate implementation change with a layout.
            control = BEST
            if name == "gate_materialized_nd":
                control = {**BEST, "SGLANG_NPU_TP_MOE_MXFP8_GATE": "1"}
            experiment(name, {**control, **SINGLES[name]}, control)
    if "sweep" in args.stages:
        for buffer in (128, 192, 256, 320, 384, 512, None):
            experiment(f"hccl_{buffer}_taskq2", {**BEST, "HCCL_BUFFSIZE": buffer}, BEST)
    if "C" in args.stages:
        ingredients = ("qproj", "qnope", "indexer_resource", "shared_resource")
        accepted = [name for name in ingredients if summaries.get(name, {}).get("nonregression")]
        if accepted:
            combined = dict(BEST)
            for name in accepted:
                combined.update(SINGLES[name])
            experiment("combined_resource", combined, BEST)
        else:
            summaries["combined_resource"] = {"status": "SKIPPED", "reason": "No measured nonregression singles"}
    if not args.dry_run and args.profile_mode != "none":
        selected = ["A_BASE", "BEST_BASE"]
        measured = [name for name, data in summaries.items() if data.get("status") == "MEASURED"]
        if args.profile_mode == "all":
            selected += measured
        else:
            for group in (("qproj", "qnope", "qproj_qnope"), ("indexer_resource", "eager_indexer_resource", "indexer_inline"), ("shared_resource", "shared_post_gate"), ("combined_resource",)):
                candidates = [name for name in group if name in measured]
                if candidates:
                    selected.append(max(candidates, key=lambda name: summaries[name]["paired_gain_pct"]))
        profiles = {}
        for name in dict.fromkeys(selected):
            profiles[name] = run(args, argv, name, settings_by_name[name], args.output_dir / "profiles" / name, profile=True)
        (args.output_dir / "profiles.json").write_text(json.dumps(profiles, indent=2))
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
