"""CPU scheduling/experiment policy checks; no NPU execution is simulated."""

import ast
import importlib.util
import os
import sys
import unittest
from contextlib import contextmanager
from contextvars import ContextVar
from enum import IntEnum, auto
from pathlib import Path
from types import MethodType, SimpleNamespace
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

    def test_qproj_threshold_and_disabled_path(self):
        ns = definitions(
            ATTENTION,
            {"_use_dsa_eager_streams", "_use_dsa_qproj_kvnorm_overlap"},
            **self.namespace,
        )
        batch = SimpleNamespace(forward_mode=modes.EXTEND)
        decision = ns["_use_dsa_qproj_kvnorm_overlap"]
        self.assertFalse(decision(batch, 4096))
        os.environ.update(
            SGLANG_NPU_DSA_OVERLAP_QPROJ_KVNORM="1",
            SGLANG_NPU_DSA_OVERLAP_QPROJ_KVNORM_MIN_TOKENS="4096",
        )
        self.assertFalse(decision(batch, 4095))
        self.assertTrue(decision(batch, 4096))
        os.environ["SGLANG_NPU_DSA_OVERLAP_QPROJ_KVNORM_MIN_TOKENS"] = "-1"
        with self.assertRaises(ValueError):
            decision(batch, 4096)

    def test_qproj_capture_and_forward_mode_fallback(self):
        ns = definitions(
            ATTENTION,
            {"_use_dsa_eager_streams", "_use_dsa_qproj_kvnorm_overlap"},
            **self.namespace,
        )
        os.environ["SGLANG_NPU_DSA_OVERLAP_QPROJ_KVNORM"] = "1"
        decision = ns["_use_dsa_qproj_kvnorm_overlap"]
        for mode, eligible in (
            (modes.EXTEND, True),
            (modes.MIXED, True),
            (modes.DECODE, False),
            (modes.IDLE, False),
            (modes.TARGET_VERIFY, False),
            (modes.DRAFT_EXTEND_V2, False),
        ):
            self.assertEqual(
                decision(SimpleNamespace(forward_mode=mode), 4096), eligible
            )
        batch = SimpleNamespace(forward_mode=modes.EXTEND)
        for kind in vars(self.capture):
            setattr(self.capture, kind, True)
            self.assertFalse(decision(batch, 4096))
            setattr(self.capture, kind, False)

    def test_stream_mode_validation(self):
        ns = definitions(INDEXER, {"_indexer_stream_mode"}, envs=envs)
        for mode in ("legacy", "inline", "resource"):
            with envs.SGLANG_NPU_DSA_INDEXER_STREAM_MODE.override(mode):
                self.assertEqual(ns["_indexer_stream_mode"](), mode)
        with (
            envs.SGLANG_NPU_DSA_INDEXER_STREAM_MODE.override("typo"),
            self.assertRaises(ValueError),
        ):
            ns["_indexer_stream_mode"]()

    def test_resource_indexer_collective_and_capture_fallbacks(self):
        parallel = SimpleNamespace(dcp_enabled=False)
        ns = definitions(
            INDEXER,
            {"_use_indexer_resource_stream", "can_forward_npu_eager"},
            **self.namespace,
            get_parallel=lambda: parallel,
            _use_ag_after_qlora=True,
        )
        indexer = SimpleNamespace(rotary_emb=SimpleNamespace(is_neox_style=False))
        indexer.can_forward_npu_eager = MethodType(ns["can_forward_npu_eager"], indexer)
        batch = SimpleNamespace(forward_mode=modes.EXTEND, attn_cp_metadata=None)
        decision = ns["_use_indexer_resource_stream"]
        self.assertTrue(decision(indexer, batch, False))
        self.assertFalse(decision(indexer, batch, True))
        batch.attn_cp_metadata = object()
        self.assertFalse(decision(indexer, batch, False))
        batch.attn_cp_metadata = None
        parallel.dcp_enabled = True
        self.assertFalse(decision(indexer, batch, False))
        parallel.dcp_enabled = False
        indexer.rotary_emb.is_neox_style = True
        self.assertFalse(decision(indexer, batch, False))
        indexer.rotary_emb.is_neox_style = False
        for kind in vars(self.capture):
            setattr(self.capture, kind, True)
            self.assertFalse(decision(indexer, batch, False))
            setattr(self.capture, kind, False)

    def test_shared_context_restored_after_failure_and_nested_forward(self):
        ns = definitions(
            NPU / "moe/tp_fusion.py",
            {"current_shared_pipeline", "use_shared_pipeline"},
            contextmanager=contextmanager,
            _shared_pipeline=ContextVar("test_pipeline", default=None),
        )
        outer, inner = object(), object()
        with ns["use_shared_pipeline"](outer):
            with self.assertRaises(RuntimeError):
                with ns["use_shared_pipeline"](inner):
                    self.assertIs(ns["current_shared_pipeline"](), inner)
                    raise RuntimeError("injected routed GEMM failure")
            self.assertIs(ns["current_shared_pipeline"](), outer)
        self.assertIsNone(ns["current_shared_pipeline"]())

    def test_shared_pipeline_mode_validation(self):
        # The local import is the stdlib-only environ module used above.
        with patch.dict(
            sys.modules, {"sglang.srt.environ": sys.modules["resource_test_environ"]}
        ):
            ns = definitions(NPU / "moe/tp_fusion.py", {"shared_pipeline_mode"})
            for mode in ("legacy", "resource"):
                with envs.SGLANG_NPU_TP_MOE_SHARED_PIPELINE.override(mode):
                    self.assertEqual(ns["shared_pipeline_mode"](), mode)
            with (
                envs.SGLANG_NPU_TP_MOE_SHARED_PIPELINE.override("typo"),
                self.assertRaises(ValueError),
            ):
                ns["shared_pipeline_mode"]()

    def test_materialized_nd_validation(self):
        path = NPU / "moe/norm_gate.py"
        choices = next(
            ast.literal_eval(node.args[1])
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_gate_option"
            and ast.literal_eval(node.args[0])
            == "SGLANG_NPU_TP_MOE_MXFP8_GATE_TOPK_LAYOUT"
        )
        ns = definitions(path, {"_gate_option"}, envs=envs)
        with envs.SGLANG_NPU_TP_MOE_MXFP8_GATE_TOPK_LAYOUT.override("materialized_nd"):
            self.assertEqual(
                ns["_gate_option"]("SGLANG_NPU_TP_MOE_MXFP8_GATE_TOPK_LAYOUT", choices),
                "materialized_nd",
            )

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
