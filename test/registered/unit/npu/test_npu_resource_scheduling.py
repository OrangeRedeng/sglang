"""CPU scheduling/experiment policy checks; no NPU execution is simulated."""

import ast
import csv
import json
import tempfile
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
            ("SGLANG_NPU_DSA_NEOX_QPROJ_KVNORM_SERIAL", False),
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

    def test_neox_serial_path_eligibility(self):
        ns = definitions(
            ATTENTION,
            {"_use_dsa_eager_streams", "_neox_qproj_serial_reason"},
            **self.namespace,
        )
        m = SimpleNamespace(
            rotary_emb=SimpleNamespace(is_neox_style=True), alt_stream=object()
        )
        batch = SimpleNamespace(forward_mode=modes.EXTEND)
        decision = ns["_neox_qproj_serial_reason"]
        self.assertIsNone(decision(m, batch))
        m.rotary_emb.is_neox_style = False
        self.assertIn("not NeoX", decision(m, batch))
        m.rotary_emb.is_neox_style = True
        m.alt_stream = None
        self.assertIn("no asynchronous", decision(m, batch))
        m.alt_stream = object()
        batch.forward_mode = modes.DECODE
        self.assertIsNotNone(decision(m, batch))
        batch.forward_mode = modes.EXTEND
        for kind in vars(self.capture):
            setattr(self.capture, kind, True)
            self.assertIsNotNone(decision(m, batch))
            setattr(self.capture, kind, False)

    def test_resource_indexer_target_cp_dcp_and_sliced_gather(self):
        parallel = SimpleNamespace(dcp_enabled=False)
        ns = definitions(
            INDEXER,
            {"_use_indexer_resource_stream", "_indexer_resource_reason"},
            **self.namespace,
            get_parallel=lambda: parallel,
            _use_ag_after_qlora=True,
        )
        indexer = SimpleNamespace(rotary_emb=SimpleNamespace(is_neox_style=True))
        batch = SimpleNamespace(forward_mode=modes.EXTEND, attn_cp_metadata=None)
        decision = ns["_use_indexer_resource_stream"]
        # No eager-indexer method is provided: scheduling must be independent.
        for cp in (None, object()):
            for dcp in (False, True):
                for sliced in (False, True):
                    batch.attn_cp_metadata = cp
                    parallel.dcp_enabled = dcp
                    self.assertTrue(decision(indexer, batch, sliced))
        for kind in vars(self.capture):
            setattr(self.capture, kind, True)
            self.assertFalse(decision(indexer, batch, False))
            self.assertIn("=1", ns["_indexer_resource_reason"](indexer, batch, False))
            setattr(self.capture, kind, False)
        for mode in (modes.DECODE, modes.TARGET_VERIFY, modes.DRAFT_EXTEND_V2):
            batch.forward_mode = mode
            self.assertFalse(decision(indexer, batch, False))

    def test_non_neox_resource_fallbacks_preserved(self):
        parallel = SimpleNamespace(dcp_enabled=False)
        ns = definitions(
            INDEXER,
            {"_use_indexer_resource_stream", "_indexer_resource_reason"},
            **self.namespace,
            get_parallel=lambda: parallel,
            _use_ag_after_qlora=True,
        )
        indexer = SimpleNamespace(rotary_emb=SimpleNamespace(is_neox_style=False))
        batch = SimpleNamespace(forward_mode=modes.EXTEND, attn_cp_metadata=None)
        decision = ns["_use_indexer_resource_stream"]
        self.assertTrue(decision(indexer, batch, False))
        self.assertFalse(decision(indexer, batch, True))
        batch.attn_cp_metadata = object()
        self.assertFalse(decision(indexer, batch, False))
        batch.attn_cp_metadata = None
        parallel.dcp_enabled = True
        self.assertFalse(decision(indexer, batch, False))

    def test_resource_projections_queue_and_caller_gather(self):
        trace, current = [], ["main"]

        class Tensor:
            shape = (4,)

            def __init__(self, name):
                self.name = name

            def view(self, *args):
                return self

            def float(self):
                return self

            def to(self, dtype):
                trace.append(("cast", current[0]))
                return Tensor("weights")

            def record_stream(self, stream):
                trace.append(("record:" + self.name, stream.name))

        class Stream:
            def __init__(self, name):
                self.name = name

            def record_event(self):
                event = len(trace)
                trace.append(("event", self.name))
                return event

            def wait_event(self, event):
                trace.append(("wait", self.name))

        main, vector = Stream("main"), Stream("vector")

        @contextmanager
        def stream_context(stream):
            previous = current[0]
            current[0] = stream.name
            try:
                yield
            finally:
                current[0] = previous

        def projection(name):
            def apply(operand):
                trace.append((name, current[0]))
                return (Tensor(name),)

            return apply

        def norm(value):
            trace.append(("k_norm", current[0]))
            return Tensor("k")

        norm.parameters = lambda: [Tensor("norm_weight")]
        norm.buffers = lambda: []

        def gather(k, batch):
            trace.append(("gather", current[0]))
            return k

        ns = definitions(
            INDEXER,
            {"_resource_projections_neox", "_record_indexer_tensors"},
            torch=SimpleNamespace(
                Tensor=Tensor,
                bfloat16=object(),
                npu=SimpleNamespace(current_stream=lambda: main, stream=stream_context),
                split=lambda *args, **kwargs: (Tensor("q_pe"), Tensor("q_nope")),
                cat=lambda *args, **kwargs: Tensor("q"),
            ),
            torch_npu=SimpleNamespace(npu_rotary_mul=lambda *args: Tensor("q_pe")),
            get_stream=lambda name: vector,
            _use_ag_after_qlora=True,
            scattered_to_tp_attn_full=gather,
        )
        indexer = SimpleNamespace(
            hidden_size=128,
            n_heads=2,
            head_dim=128,
            rope_head_dim=64,
            wq_b=projection("wq_b"),
            weights_proj=projection("weights_proj"),
            wk=projection("wk"),
            k_norm=norm,
            _neox_sin_cos=lambda *args: (Tensor("sin"), Tensor("cos")),
            _neox_k_rope=lambda k, *args: k,
        )
        method = MethodType(ns["_resource_projections_neox"], indexer)
        q, k, weights = method(
            Tensor("x"), Tensor("q_lora"), Tensor("positions"), object(), True, None
        )
        self.assertEqual(
            [item for item in trace if item[0] in ("wq_b", "weights_proj", "wk")],
            [("wq_b", "main"), ("weights_proj", "main"), ("wk", "main")],
        )
        self.assertIn(("k_norm", "vector"), trace)
        self.assertEqual(trace[-1], ("gather", "main"))
        for name in ("wq_b", "weights_proj", "wk", "sin", "cos", "norm_weight"):
            self.assertIn(("record:" + name, "vector"), trace)
        for tensor in (q, k, weights):
            self.assertIn(("record:" + tensor.name, "main"), trace)

    def test_shared_reason_builder_exposes_each_blocker(self):
        ns = definitions(NPU / "moe/tp_fusion.py", {"shared_resource_blockers"})
        state = dict(
            has_shared_stream=True,
            is_extend_in_batch=True,
            is_nextn=False,
            is_glm_moe_dsa=True,
            shared_gmm1_mode="grouped_fused",
            swiglu_limit=None,
            runner_inplace=False,
            capture_mode=False,
            breakable_graph=False,
            piecewise_graph=False,
            sp_active=False,
            down_proj_decode_attn_tp=False,
            skip_shared_experts=False,
            token_threshold_met=True,
            fuse_shared=True,
        )
        blockers = ns["shared_resource_blockers"]
        self.assertEqual(blockers(state), ())
        for key, value in state.items():
            if key == "fuse_shared":
                continue
            wrong = not value if isinstance(value, bool) else "unsupported"
            self.assertEqual(blockers({**state, key: wrong}), (key,))
        self.assertEqual(blockers({**state, "fuse_shared": False}), ())

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
            result = runner.summarize(pairs, mechanism_confirmed=True)
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

    def test_runner_requires_mechanism_and_rejects_profiler_ttft(self):
        def record(profiled=False):
            return {
                "status": "ok",
                "runs": [{"median_ttft": 100}],
                "correctness": True,
                "profiled": profiled,
            }

        pairs = [
            (record(), {**record(), "runs": [{"median_ttft": 98}]}, record())
            for _ in range(3)
        ]
        self.assertFalse(runner.summarize(pairs)["promotion_eligible"])
        pairs[0][1]["profiled"] = True
        self.assertEqual(
            runner.summarize(pairs, mechanism_confirmed=True)["status"], "PROFILE_ONLY"
        )

    def test_runner_minimal_matrix_and_auto_hccl(self):
        self.assertEqual(
            runner.DEFAULT_CASES, ["NEOX_QPROJ_SERIAL", "OLD_QNOPE", "IDX_RESOURCE"]
        )
        self.assertEqual(runner.BASE["SGLANG_NPU_AUTO_HCCL_BUFFSIZE"], "1")
        self.assertIsNone(runner.BASE["HCCL_BUFFSIZE"])
        self.assertEqual(runner.BASE["TASK_QUEUE_ENABLE"], "2")
        self.assertEqual(runner.BASE["ASCEND_RT_VISIBLE_DEVICES"], "4,5,6,7")

    def test_runner_analyzes_capture_and_keeps_unknown_mixed_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            profile = directory / "profile" / "0"
            profile.mkdir(parents=True)
            path = profile / "kernel_details.csv"
            with path.open("w") as output:
                writer = csv.writer(output)
                writer.writerow(
                    [
                        "Name",
                        "Stream ID",
                        "Task Start Time(us)",
                        "Task Duration(us)",
                        "Task Type",
                    ]
                )
                writer.writerows(
                    [
                        ["TopK", 1, 0, 1000, "AI_CUBE"],
                        ["QuantLightningIndexer", 2, 100, 200, "AI_CUBE"],
                        ["SparseAttention", 3, 200, 400, "AI_VECTOR"],
                        ["mixed", 4, 50, 700, "MIX_AIC"],
                        ["unknown", 5, 0, 800, "AI_CORE"],
                        ["HcomAllGather", 6, 400, 500, "HCCL"],
                    ]
                )
            args = SimpleNamespace(
                profile_task_glob="**/kernel_details.csv", profile_device_pid=None
            )
            result = runner.analyze_capture(args, directory)
            overlap = result["resource_overlap"]
            self.assertAlmostEqual(overlap["cube_vector_ms"], 0.4)
            self.assertAlmostEqual(overlap["cube_cube_ms"], 0.2)
            self.assertEqual(overlap["vector_vector_ms"], 0)
            self.assertAlmostEqual(overlap["compute_comm_ms"], 0.5)
            self.assertEqual(overlap["unknown_compute_calls"], 1)
            self.assertEqual(overlap["mixed_compute_calls"], 1)
            self.assertAlmostEqual(overlap["unknown_compute_busy_ms"], 0.8)
            self.assertAlmostEqual(overlap["mixed_compute_busy_ms"], 0.7)
            self.assertEqual(result["topk_calls"], 1)
            self.assertEqual(result["sparse_attn_calls"], 1)
            self.assertEqual(result["target_ops"]["Indexer wq_b"]["mean_us"], None)
            runner.write_report(
                directory, {"AUTO_TQ2": {"status": "ok", "resource_analysis": result}}
            )
            self.assertIn(
                "cube_vector_ms", (directory / "resource_overlap_report.md").read_text()
            )
            self.assertEqual(
                json.loads((directory / "resource_overlap.json").read_text()), result
            )
            (profile / "device_1").mkdir()
            (profile / "device_1" / "kernel_details.csv").write_text(path.read_text())
            multi = runner.analyze_capture(args, directory)
            self.assertEqual(len(multi["device_profiles"]), 2)
            for device in multi["device_profiles"]:
                self.assertIsNone(device["device_key"])
                self.assertEqual(device["topk_calls"], 1)
            runner.write_report(
                directory, {"AUTO_TQ2": {"status": "ok", "resource_analysis": multi}}
            )
            self.assertIn(
                "unassigned_export_1",
                (directory / "resource_overlap_report.md").read_text(),
            )


if __name__ == "__main__":
    unittest.main()
