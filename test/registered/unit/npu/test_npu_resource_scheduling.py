"""CPU scheduling/experiment policy checks; no NPU execution is simulated."""

import ast
import importlib.util
import os
import sys
import unittest
from enum import IntEnum, auto
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ModuleNotFoundError:
    pass
else:
    register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
NPU = ROOT / "python/sglang/srt/hardware_backend/npu"
INDEXER = ROOT / "python/sglang/srt/layers/attention/dsa/dsa_npu_indexer.py"
ATTENTION = NPU / "modules/deepseek_v2_attention_mla_npu.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def definitions(path, names, **namespace):
    # Execute the actual decisions without importing the device runtime.
    nodes = [
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert len(nodes) == len(names)
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
    return namespace


envs = load_module("resource_test_environ", ROOT / "python/sglang/srt/environ.py").envs
policy = load_module("resource_test_autotune", NPU / "autotune.py")
runner = load_module(
    "resource_test_runner",
    ROOT / "benchmark/kernels/run_a5_resource_overlap_experiments.py",
)
modes = definitions(
    ROOT / "python/sglang/srt/model_executor/forward_batch_info.py",
    {"ForwardMode"},
    IntEnum=IntEnum,
    auto=auto,
)["ForwardMode"]


class TestResourceScheduling(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.capture = SimpleNamespace(full=False, breakable=False, piecewise=False)
        self.namespace = dict(
            envs=envs,
            threshold_allows=policy.threshold_allows,
            get_is_capture_mode=lambda: self.capture.full,
            is_in_breakable_cuda_graph=lambda: self.capture.breakable,
            is_in_tc_piecewise_cuda_graph=lambda: self.capture.piecewise,
        )

    def test_defaults_preserve_baseline(self):
        for name, expected in (
            ("SGLANG_NPU_DSA_OVERLAP_QPROJ_KVNORM", False),
            ("SGLANG_NPU_DSA_OVERLAP_QPROJ_KVNORM_MIN_TOKENS", 0),
            ("SGLANG_NPU_DSA_INDEXER_STREAM_MODE", "legacy"),
            ("SGLANG_NPU_DSA_INDEXER_HADAMARD_MODE", "matmul"),
            ("SGLANG_NPU_TP_MOE_SHARED_PIPELINE", "legacy"),
            ("SGLANG_NPU_RESOURCE_SCHED_DIAGNOSTICS", False),
        ):
            self.assertEqual(getattr(envs, name).get(), expected)

    def test_runner_never_combines_candidate_without_correctness(self):
        def record(value, correct):
            return {
                "status": "ok",
                "runs": [{"median_ttft": value}],
                "correctness": correct,
            }

        for correct in (False, True):
            pairs = [
                (record(100, correct), record(98, correct), record(100, correct))
                for _ in range(3)
            ]
            result = runner.summarize(pairs)
            self.assertEqual(result["nonregression"], correct)
            self.assertEqual(result["promotion_eligible"], correct)

    def test_runner_rejects_failed_pairs_and_bimodal_results(self):
        def record(value):
            return {
                "status": "ok",
                "runs": [{"median_ttft": value}],
                "correctness": True,
            }

        pairs = [(record(100), record(value), record(100)) for value in (95, 95, 99)]
        result = runner.summarize(pairs)
        self.assertEqual(result["status"], "BIMODAL")
        self.assertFalse(result["nonregression"])
        pairs[0][1]["status"] = "ineligible"
        self.assertEqual(runner.summarize(pairs)["status"], "FAILED_OR_INELIGIBLE")


if __name__ == "__main__":
    unittest.main()
