"""CPU coverage for strict scaled acceptance and experiment evidence handling."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

root = Path(__file__).resolve().parents[3] / "manual/gemma4"


def load(name):
    spec = importlib.util.spec_from_file_location(name, root / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matrix = load("throughput_matrix")
profiles = load("collect_profile")
quality = load("quality_check")
validation = load("candidate_validation")
with patch.dict(
    sys.modules, {"candidate_validation": validation, "throughput_matrix": matrix}
):
    plan = load("run_plan")


class TestCandidateEvidence(CustomTestCase):
    def test_stale_runtime_and_partial_cases_cannot_select_a_kernel(self):
        """A correctness report must match the code, runtime and complete case matrix."""
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name in validation.SOURCES:
                path = directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("original")
            versions = {"torch": "2.10", "torch_npu": "2.10.post7"}
            report = {
                "kind": "qkv",
                "complete": True,
                "source_sha256": validation.source_hashes(directory),
                **versions,
                "device_name": "test-npu",
                "cases": [],
            }
            for dtype in ("torch.float16", "torch.bfloat16"):
                for dim in (128, 256, 512):
                    for tokens in validation.QKV_TOKENS:
                        for shared in (False, True):
                            report["cases"].append(
                                {
                                    "candidate": "tile8",
                                    "dtype": dtype,
                                    "head_dim": dim,
                                    "tokens": tokens,
                                    "shared_kv": shared,
                                    "eager_correct": True,
                                    "fullgraph_trace_correct": True,
                                    "graph_replay_modes": [0, 1],
                                }
                            )
            path = directory / "report.json"

            def check(data):
                path.write_text(json.dumps(data))
                return validation.validate_report(
                    path, "qkv", "tile8", directory, versions, "test-npu"
                )

            check(report)
            for changed in (
                {**report, "complete": False},
                {**report, "kind": "cpu_diagnostic"},
                {**report, "torch_npu": "older"},
                {**report, "device_name": "other"},
                {**report, "cases": report["cases"][:-1]},
            ):
                with self.assertRaises(ValueError):
                    check(changed)
            (directory / validation.SOURCES[0]).write_text("modified")
            with self.assertRaises(ValueError):
                check(report)

    def test_exploratory_ranking_cannot_hide_missing_or_profiled_cases(self):
        rows = [
            {
                "scenario": scenario,
                "case": case,
                "completed_rps": rate,
                "metrics_ms": matrix.TARGETS[scenario][1],
                "checks": {"completed": True, "failed": True, "output_tokens": True},
                "measurement_kind": "unprofiled",
            }
            for scenario, (rate, _) in matrix.TARGETS.items()
            for case in matrix.CASES
        ]
        self.assertAlmostEqual(plan.exploratory_score(rows), 1.0)
        self.assertIsNone(plan.exploratory_score(rows[:-1]))
        self.assertIsNone(plan.exploratory_score(rows[:-1] + [rows[0]]))
        rows[0]["measurement_kind"] = "profiled"
        self.assertIsNone(plan.exploratory_score(rows))

    def test_overlapping_cpu_intervals_are_not_summed_as_wall_time(self):
        events = [
            {"name": "aclSynchronize", "ts": 0, "dur": 10, "ph": "X"},
            {"name": "aclSynchronizeStream", "ts": 2, "dur": 5, "ph": "X"},
            {"name": "op1", "cat": "kernel", "ts": 0, "dur": 4, "ph": "X"},
            {"name": "op2", "cat": "kernel", "ts": 8, "dur": 2, "ph": "X"},
        ]
        timing = profiles.trace_timing(events)
        self.assertEqual(timing["union_us"]["synchronization"], 10)
        self.assertEqual(timing["union_us"]["device"], 6)
        self.assertEqual(timing["gaps_between_device_events_us"], 4)


class TestAcceptance(unittest.TestCase):
    def result(self, scenario):
        rate, limits = matrix.TARGETS[scenario]
        return {**limits, "request_throughput": rate, "completed": 256, "failed": 0}

    def test_exact_targets_and_every_latency_gate(self):
        for scenario in matrix.TARGETS:
            raw = self.result(scenario)
            self.assertTrue(matrix.evaluate_case(raw, scenario, 256)["pass"])
            for name, value in raw.items():
                with self.subTest(scenario=scenario, field=name):
                    bad = dict(raw)
                    bad[name] = (
                        value - 0.001 if name == "request_throughput" else value + 1
                    )
                    self.assertFalse(matrix.evaluate_case(bad, scenario, 256)["pass"])

    def test_rejects_old_98_percent_allowance_and_missing_metrics(self):
        for scenario in matrix.TARGETS:
            raw = self.result(scenario)
            raw["request_throughput"] *= 0.99
            self.assertFalse(matrix.evaluate_case(raw, scenario, 256)["pass"])
            raw = self.result(scenario)
            for name in raw:
                incomplete = dict(raw)
                del incomplete[name]
                self.assertFalse(
                    matrix.evaluate_case(incomplete, scenario, 256)["pass"]
                )

    def test_fixed_output_lengths_are_required(self):
        raw = self.result("recommendation")
        self.assertFalse(
            matrix.evaluate_case(raw, "recommendation", 256, output_len=100)["pass"]
        )
        raw["total_output_tokens"] = 25600
        self.assertTrue(
            matrix.evaluate_case(raw, "recommendation", 256, output_len=100)["pass"]
        )
        raw["total_output_tokens"] -= 1
        self.assertFalse(
            matrix.evaluate_case(raw, "recommendation", 256, output_len=100)["pass"]
        )

    def test_rejects_nonfinite_metrics(self):
        for value in (float("nan"), float("inf"), True):
            raw = self.result("rag")
            raw["mean_ttft_ms"] = value
            self.assertFalse(matrix.evaluate_case(raw, "rag", 256)["pass"])

    def test_batch_means_keep_replicas_separate(self):
        text = (
            "DP0] Prefill batch, #new-seq: 2, npu graph: True\n"
            "DP0] Prefill batch, #new-seq: 6, npu graph: True\n"
            "DP1] Prefill batch, #new-seq: 1, npu graph: True\n"
            "DP1] Decode batch, #running-req: 14, npu graph: True"
        )
        stats = matrix.batch_statistics(text)
        self.assertEqual(stats["0"]["prefill"], {"samples": 2, "mean": 4})
        self.assertEqual(stats["1"]["prefill"]["mean"], 1)
        self.assertEqual(stats["1"]["decode"]["mean"], 14)
        self.assertIsNone(stats["0"]["decode"]["mean"])

    def test_distinct_warmup_and_measured_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for scenario in matrix.TARGETS:
                (directory / scenario).mkdir()
                for case in matrix.CASES:
                    (directory / scenario / f"{case}.jsonl").write_text(
                        json.dumps({"prompt": "measured"}) + "\n"
                    )
                    (directory / scenario / f"{case}.warmup.jsonl").write_text(
                        json.dumps({"prompt": "warmup"}) + "\n"
                    )
            before = matrix.validate_datasets(directory, 1, 1)
            changed = directory / "rag/high_high.jsonl"
            changed.write_text(json.dumps({"prompt": "different"}) + "\n")
            self.assertNotEqual(before, matrix.validate_datasets(directory, 1, 1))
            changed.write_text(json.dumps({"prompt": "warmup"}) + "\n")
            with self.assertRaisesRegex(ValueError, "overlap"):
                matrix.validate_datasets(directory, 1, 1)

    def test_profile_needs_both_devices_and_both_stages(self):
        reports = [{"devices": ["2"], "observed_steps": {"prefill": 3, "decode": 3}}]
        self.assertFalse(profiles.complete(reports))
        reports.append({"devices": ["3"], "observed_steps": {"decode": 3}})
        self.assertFalse(profiles.complete(reports))
        reports.append({"devices": ["3"], "observed_steps": {"prefill": 1}})
        self.assertTrue(profiles.complete(reports))


class TestQualityGate(unittest.TestCase):
    def snapshot(self):
        return {
            "model": "gemma4",
            "server_args": {"tp_size": 1, "dp_size": 2},
            "per_case": 1,
            "cases": [
                {
                    "scenario": scenario,
                    "case": case,
                    "prompt_sha256": scenario + case,
                    "text": "answer",
                    "tokens": ["answer"],
                    "logprobs": [-0.4],
                }
                for scenario in ("recommendation", "rag")
                for case in matrix.CASES
            ],
        }

    def test_requires_identical_prompts_and_greedy_outputs(self):
        baseline = self.snapshot()
        self.assertTrue(quality.compare(baseline, self.snapshot())["pass"])
        for field in ("text", "tokens", "prompt_sha256"):
            changed = self.snapshot()
            changed["cases"][0][field] = (
                ["different"] if field == "tokens" else "different"
            )
            self.assertFalse(quality.compare(baseline, changed)["pass"])

    def test_rejects_missing_nonfinite_or_different_logprobs(self):
        baseline = self.snapshot()
        for values in ([], [None], [float("nan")], [-0.7]):
            changed = self.snapshot()
            changed["cases"][0]["logprobs"] = values
            self.assertFalse(quality.compare(baseline, changed)["pass"])

    def test_rejects_missing_cases_and_changed_runtime(self):
        baseline = self.snapshot()
        changed = self.snapshot()
        changed["cases"].pop()
        self.assertFalse(quality.compare(baseline, changed)["pass"])
        changed = self.snapshot()
        changed["server_args"]["dp_size"] = 1
        self.assertFalse(quality.compare(baseline, changed)["pass"])


if __name__ == "__main__":
    unittest.main()
