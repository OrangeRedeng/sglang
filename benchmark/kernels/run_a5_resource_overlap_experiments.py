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
    "TASK_QUEUE_ENABLE": "2",
    "SGLANG_NPU_AUTO_HCCL_BUFFSIZE": "1",
    "ASCEND_RT_VISIBLE_DEVICES": "4,5,6,7",
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
    "SGLANG_NPU_DSA_NEOX_QPROJ_KVNORM_SERIAL": "0",
    "SGLANG_NPU_DSA_OVERLAP_QPROJ_KVNORM_MIN_TOKENS": "0",
    "SGLANG_NPU_DSA_OVERLAP_QNOPE_ROPE": "0",
    "SGLANG_NPU_DSA_OVERLAP_QNOPE_ROPE_MIN_TOKENS": "0",
    "SGLANG_NPU_DSA_EAGER_INDEXER": "0",
    "SGLANG_NPU_ENABLE_DSA_CP": "1",
    "SGLANG_NPU_ENABLE_DSA_CP_MULTI_REQUEST": "1",
    "SGLANG_NPU_ENABLE_DSA_INDEXER_QUERY_SHARDING": "1",
    "SGLANG_NPU_DSA_EAGER_INDEXER_MIN_TOKENS": "0",
    "SGLANG_NPU_TP_MOE_EAGER_MULTI_STREAM_MIN_TOKENS": "0",
    "SGLANG_NPU_DSA_INDEXER_STREAM_MODE": "legacy",
    "SGLANG_NPU_DSA_INDEXER_HADAMARD_MODE": "matmul",
    "SGLANG_NPU_TP_MOE_MXFP8_GATE_TOPK_LAYOUT": "default",
    "SGLANG_NPU_ENABLE_DCP_EXTEND_GATHER_PREFETCH": "0",
    "SGLANG_NPU_RESOURCE_SCHED_DIAGNOSTICS": "1",
}
SINGLES = {
    "NEOX_QPROJ_SERIAL": {"SGLANG_NPU_DSA_NEOX_QPROJ_KVNORM_SERIAL": "1"},
    "OLD_QNOPE": {"SGLANG_NPU_DSA_OVERLAP_QNOPE_ROPE": "1"},
    "IDX_RESOURCE": {"SGLANG_NPU_DSA_INDEXER_STREAM_MODE": "resource"},
    "MOE_SHARED_RESOURCE": {"SGLANG_NPU_TP_MOE_SHARED_PIPELINE": "resource"},
    "IDX_RESOURCE_QNOPE": {
        "SGLANG_NPU_DSA_INDEXER_STREAM_MODE": "resource",
        "SGLANG_NPU_DSA_OVERLAP_QNOPE_ROPE": "1",
    },
}
DEFAULT_CASES = ["NEOX_QPROJ_SERIAL", "OLD_QNOPE", "IDX_RESOURCE"]
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


def summarize(pairs, *, mechanism_confirmed=False):
    if any(r.get("profiled", False) for p in pairs for r in p):
        return {"status": "PROFILE_ONLY", "promotion_eligible": False}
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
    gain = 100 * (
        1 - statistics.median(value / anchor for value, anchor in zip(values, anchors))
    )
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
            mechanism_confirmed
            and not unstable
            and len(values) >= 3
            and gain >= 0.5
            and wins * 3 >= len(values) * 2
            and correct
        ),
    }


def engagement(settings, server_log):
    for flag, message in (
        (
            "SGLANG_NPU_DSA_NEOX_QPROJ_KVNORM_SERIAL",
            "DSA NeoX qproj/KV-norm serial ablation is ACTIVE",
        ),
        ("SGLANG_NPU_DSA_EAGER_INDEXER", "DSA eager indexer is ACTIVE"),
        (
            "SGLANG_NPU_ENABLE_DCP_EXTEND_GATHER_PREFETCH",
            "DCP extend gather prefetch is ACTIVE",
        ),
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


# Logical projections require launch correlation; anonymous MatMul/GMM rows
# remain visible with shapes and are never assigned to a projection by guess.
TARGET_OPS = {
    "Attention q_b_proj": r"q_b_proj",
    "Attention KV RMSNorm": r"kv.*(?:norm|rms)",
    "Attention q_nope BMM": r"q_nope|transpose.*batch.*matmul",
    "q RoPE": r"q.*rope",
    "k RoPE": r"k.*rope",
    "Indexer wq_b": r"wq_b",
    "Indexer weights_proj": r"weights_proj",
    "Indexer wk": r"(?:^|[./])wk(?:$|[./])",
    "Indexer k_norm": r"k_norm",
    "Hadamard q": r"(?:q.*hadamard|hadamard.*q)",
    "Hadamard k": r"(?:k.*hadamard|hadamard.*k)",
    "MX quant q": r"(?:q.*mx_quant|mx_quant.*q)",
    "MX quant k": r"(?:k.*mx_quant|mx_quant.*k)",
    "QuantLightningIndexer": r"quant.*lightning.*indexer",
    "Gate QuantMatmul": r"gate.*quant.*matmul",
    "TopK": r"topk",
    "InitRouting": r"init.*routing",
    "routed GMM1": r"routed.*gmm1",
    "routed GMM2": r"routed.*gmm2",
    "shared GateUp": r"shared.*gateup",
    "shared activation/quant": r"shared.*(?:activation|quant)",
    "shared Down": r"shared.*down",
    "FinalizeRouting": r"finalize.*routing",
    "Unattributed projection GEMMs": r"matmul|gemm|gmm",
    "Unattributed RMSNorm": r"rms.*norm",
    "Unattributed RoPE": r"rotary|rope",
    "Unattributed MX quant": r"dynamic.*mx.*quant",
}


def op_rows(analysis):
    rows = {}
    for label, pattern in TARGET_OPS.items():
        matches = [
            k for k in analysis["kernels"] if re.search(pattern, k["name"], re.I)
        ]
        count = sum(k["calls"] for k in matches)
        rows[label] = {
            "calls": count,
            "mean_us": sum(k["mean_us"] * k["calls"] for k in matches) / count
            if count
            else None,
            "kernels": matches,
        }
    return rows


def analyze_capture(args, directory):
    profile_dir = directory / "profile"
    pattern = args.profile_task_glob
    matches = (
        sorted(profile_dir.glob(pattern))
        if pattern
        else sorted(profile_dir.rglob("kernel_details.csv"))
    )
    if not pattern and not matches:
        matches = sorted(profile_dir.rglob("op_summary*.csv"))
    if not matches:
        raise ValueError(
            f"No device task export found for {pattern or 'automatic CSV discovery'}"
        )
    spec = importlib.util.spec_from_file_location(
        "a5_overlap_analyzer", Path(__file__).with_name("analyze_npu_multistream.py")
    )
    analyzer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = analyzer
    spec.loader.exec_module(analyzer)
    results = []
    for path in matches:
        tasks = analyzer.read_tasks(path, args.profile_device_pid)
        compute = [task for task in tasks if task.kind == "compute"]
        if not compute:
            raise ValueError(f"Profile contains no compute tasks: {path}")
        # Stream selection affects stream diagnostics, never core classification.
        main_stream = max(
            {t.stream for t in compute},
            key=lambda stream: sum(
                t.end - t.start for t in compute if t.stream == stream
            ),
        )
        result = analyzer.analyze(tasks, main_stream, {}, None, None, [])
        result["task_file"] = str(path)
        ranks = set()

        def find_rank(metadata):
            if isinstance(metadata, dict):
                rank = metadata.get("rank_id")
                if isinstance(rank, int) and rank >= 0:
                    ranks.add(rank)
                for value in metadata.values():
                    find_rank(value)
            elif isinstance(metadata, list):
                for value in metadata:
                    find_rank(value)

        for info in path.parent.parent.glob("profiler_info*.json"):
            find_rank(json.loads(info.read_text()))
        result["device_key"] = (
            f"rank_{next(iter(ranks))}"
            if len(ranks) == 1
            else "selected_device"
            if len(matches) == 1
            else None
        )
        result["target_ops"] = op_rows(result)
        result["topk_calls"] = result["target_ops"]["TopK"]["calls"]
        result["topk_mean_us"] = result["target_ops"]["TopK"]["mean_us"]
        sparse = [
            k
            for k in result["kernels"]
            if re.search(r"sparse.*(?:attn|attention)", k["name"], re.I)
        ]
        result["sparse_attn_calls"] = sum(k["calls"] for k in sparse)
        results.append(result)
    result = results[0] if len(results) == 1 else {"device_profiles": results}
    (directory / "resource_overlap.json").write_text(json.dumps(result, indent=2))
    (directory / "resource_overlap.txt").write_text(json.dumps(result, indent=2))
    return result


def write_report(directory, profiles):
    fields = [
        "compute_busy_ms",
        "comm_total_ms",
        "comm_busy_ms",
        "topk_calls",
        "sparse_attn_calls",
        "topk_mean_us",
        "cube_vector_ms",
        "cube_cube_ms",
        "vector_vector_ms",
        "compute_comm_ms",
        "unknown_compute_calls",
        "mixed_compute_calls",
        "unknown_compute_busy_ms",
        "mixed_compute_busy_ms",
    ]
    lines = [
        "# Resource overlap screening",
        "",
        "Profiler TTFT is excluded from ranking. Compare raw totals only with matching call counts.",
        "Unknown/mixed kernels are excluded from Cube/Vector pairs; compute/COMM includes all compute.",
        "Missing logical rows need launch correlation; anonymous GEMMs are listed by name/shape in JSON.",
        "",
        "| Case | " + " | ".join(fields) + " | Comparable calls |",
        "|---|" + "---|" * (len(fields) + 1),
    ]

    def devices(record):
        result = record.get("resource_analysis", {})
        return result.get("device_profiles", [result]) if result else []

    base = {
        r["device_key"]: r
        for r in devices(profiles.get("AUTO_TQ2", {}))
        if r["device_key"]
    }
    for name, record in profiles.items():
        if not devices(record):
            lines += [f"| {name}: {record.get('error', record['status'])} |"]
        for index, result in enumerate(devices(record)):
            flat = {**result, **result["resource_overlap"]}
            reference = base.get(result["device_key"], {})
            comparable = all(
                result[k] == reference.get(k) and result[k] > 0
                for k in ("topk_calls", "sparse_attn_calls")
            )
            label = f"{name}/{result['device_key'] or f'unassigned_export_{index}'}"
            lines += [
                f"| {label} | "
                + " | ".join(str(flat[k]) for k in fields)
                + f" | {comparable} |"
            ]
    for name, record in profiles.items():
        for index, result in enumerate(devices(record)):
            label = f"{name}/{result['device_key'] or f'unassigned_export_{index}'}"
            lines += [
                "",
                f"## {label} kernel rows",
                "",
                f"Task file: `{result['task_file']}`",
                "",
                "| Operation | Calls | Mean us |",
                "|---|---:|---:|",
            ]
            for label, row in result["target_ops"].items():
                lines += [
                    f"| {label} | {row['calls']} | {row['mean_us'] if row['mean_us'] is not None else 'unavailable'} |"
                ]
    (directory / "resource_overlap_report.md").write_text("\n".join(lines) + "\n")


def run(args, argv, name, settings, directory, *, profile=False):
    case_args = argparse.Namespace(**vars(args))
    case_args.analyzer_command_file = args.analyzer_command_file if profile else None
    if profile:
        case_args.benchmark_command_file = args.profile_command_file
    record = restart.run_case(case_args, argv, settings, directory, 1)
    record.update(
        name=name, profiled=profile, model_path=args.model_path, server_args=argv
    )
    try:
        if record["status"] == "ok":
            record["signature"] = workload_signature(
                (directory / "benchmark-0.log").read_text()
            )
            reason = engagement(settings, (directory / "server.log").read_text())
            if reason:
                record.update(status="ineligible", error=reason)
            if profile:
                record["resource_analysis"] = analyze_capture(args, directory)
    except (OSError, ValueError, KeyError) as exc:
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
    parser.add_argument(
        "--profile-task-glob",
        help="One device CSV/trace relative to the case profile directory",
    )
    parser.add_argument(
        "--profile-device-pid", help="Ascend Hardware PID for Chrome traces"
    )
    parser.add_argument("--phase", choices=("screen", "confirm"), default="screen")
    parser.add_argument(
        "--screen-results-file",
        type=Path,
        help="Reviewed profiles.json for confirmation shortlist",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cases", nargs="+", choices=tuple(SINGLES), default=DEFAULT_CASES
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Interleaved confirmation pairs only"
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30088)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--benchmark-timeout", type=float, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.expected_requests = 17
    if args.phase == "screen" and not args.profile_command_file and not args.dry_run:
        parser.error("screening requires --profile-command-file")
    if args.phase == "confirm" and (
        args.runs < 3
        or not args.screen_results_file
        or not args.correctness_command_file
    ):
        parser.error(
            "confirmation requires >=3 pairs, --screen-results-file and --correctness-command-file"
        )
    argv = shlex.split(args.server_args_file.read_text(), comments=True)
    for option, expected in (
        ("--tp-size", "4"),
        ("--ep-size", "1"),
        ("--chunked-prefill-size", "16384"),
        ("--max-prefill-tokens", "16384"),
        ("--page-size", "128"),
    ):
        value = restart.option_value(argv, option)
        if value is not None and value != expected:
            parser.error(f"Target workload requires {option}={expected}; got {value}")
        argv = restart.set_option(argv, option, expected)
    kv_dtype = restart.option_value(argv, "--kv-cache-dtype")
    if kv_dtype is None or not kv_dtype.startswith("fp8"):
        parser.error(
            "server args must explicitly select the checkpoint's FP8 KV cache dtype"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    settings = {
        "AUTO_TQ2": BASE,
        **{name: {**BASE, **SINGLES[name]} for name in args.cases},
    }
    summaries, profiles = {}, {}
    if args.dry_run:
        summaries = {
            name: {"status": "DRY_RUN", "settings": config}
            for name, config in settings.items()
        }
    elif args.phase == "screen":
        for name, config in settings.items():
            if (
                name == "IDX_RESOURCE_QNOPE"
                and profiles.get("IDX_RESOURCE", {}).get("status") != "ok"
            ):
                summaries[name] = {
                    "status": "SKIPPED",
                    "reason": "Screen IDX_RESOURCE first",
                }
                continue
            sanity = run(args, argv, name, config, args.output_dir / "sanity" / name)
            if sanity["status"] == "ok":
                profiles[name] = run(
                    args,
                    argv,
                    name,
                    config,
                    args.output_dir / "profiles" / name,
                    profile=True,
                )
            summaries[name] = {
                "status": "SCREENED"
                if profiles.get(name, {}).get("status") == "ok"
                else profiles.get(name, sanity)["status"],
                "sanity": sanity,
                "promotion_eligible": False,
            }
            (args.output_dir / "profiles.json").write_text(
                json.dumps(profiles, indent=2)
            )
            write_report(args.output_dir, profiles)
    else:
        if len(args.cases) > 3:
            parser.error("confirm only the best 2-3 mechanisms from screening")
        evidence = json.loads(args.screen_results_file.read_text())
        for name in args.cases:
            profile = evidence.get(name, {})
            base = evidence.get("AUTO_TQ2", {})
            if any(
                r.get("status") != "ok" or not r.get("resource_analysis")
                for r in (base, profile)
            ):
                parser.error(
                    f"{name} requires successful baseline/candidate resource profiles"
                )
            for label, screened in (("AUTO_TQ2", base), (name, profile)):
                if (
                    screened.get("settings") != settings[label]
                    or screened.get("model_path") != args.model_path
                    or screened.get("server_args") != argv
                ):
                    parser.error(
                        f"{label} screen settings/checkpoint do not match confirmation"
                    )
            pairs = []
            for repetition in range(args.runs):
                trial = args.output_dir / name / str(repetition)
                pairs.append(
                    tuple(
                        run(args, argv, label, config, trial / suffix)
                        for suffix, label, config in (
                            ("before", "AUTO_TQ2", BASE),
                            ("candidate", name, settings[name]),
                            ("after", "AUTO_TQ2", BASE),
                        )
                    )
                )
            # Selecting --cases with reviewed profiles is an explicit shortlist,
            # not an automatic claim that aggregate overlap proves a mechanism.
            summaries[name] = summarize(pairs, mechanism_confirmed=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
