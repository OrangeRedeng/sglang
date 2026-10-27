"""CPU-only policy coverage; importing the runtime is unnecessary."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ModuleNotFoundError:
    pass
else:
    register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "npu_autotune_policy_test",
    ROOT / "python/sglang/srt/hardware_backend/npu/autotune.py",
)
auto = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = auto
SPEC.loader.exec_module(auto)
SPEC_TUNER = importlib.util.spec_from_file_location(
    "npu_restart_tuner_test", ROOT / "benchmark/kernels/autotune_npu_prefill.py"
)
tuner = importlib.util.module_from_spec(SPEC_TUNER)
SPEC_TUNER.loader.exec_module(tuner)


def context():
    return auto.NpuTuningContext(
        "Ascend 950",
        "arch35",
        131072,
        "GlmMoeDsaForCausalLM",
        6144,
        78,
        256,
        8,
        4,
        1,
        4,
        1,
        1,
        16384,
        32768,
        32,
        "torch.bfloat16",
        "fp8_e4m3",
        "modelslim",
        "9.2.beta1",
        "2.10.0.post6",
        "test-commit",
        128,
    )


class TestNpuAutotune(unittest.TestCase):
    def test_hccl_current_shape(self):
        self.assertEqual(auto.hccl_buffer_mb(16384, 6144, 2), 256)

    def test_representative_prefill_rows(self):
        for chunk in (16384, -1, 0):
            c = replace(context(), chunked_prefill_size=chunk, max_prefill_tokens=16384)
            self.assertEqual(auto.representative_prefill_rows(c), 16384)
            self.assertEqual(
                tuner.candidates("buffer", c, None)[1], [128, 192, 256, 384, 512]
            )
        self.assertEqual(
            auto.representative_prefill_rows(
                replace(context(), chunked_prefill_size=8192)
            ),
            8192,
        )
        for chunk, maximum in ((16384, 0), (0, 0), (-1, -1)):
            with (
                self.subTest(chunk=chunk, maximum=maximum),
                self.assertRaises(ValueError),
            ):
                auto.representative_prefill_rows(
                    replace(
                        context(),
                        chunked_prefill_size=chunk,
                        max_prefill_tokens=maximum,
                    )
                )

    def test_restart_hccl_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "context.json").write_text(json.dumps(context().key_data()))
            (root / "server.txt").write_text("--tp-size 4")
            argv = [
                "autotune",
                "--model-path",
                "/model",
                "--context-file",
                str(root / "context.json"),
                "--server-args-file",
                str(root / "server.txt"),
                "--benchmark-command-file",
                str(root / "benchmark.txt"),
                "--output",
                str(root / "plan.json"),
                "--stages",
                "buffer",
                "--dry-run",
            ]
            for explicit, tune_explicit in (
                (None, False),
                ("1000", False),
                ("1000", True),
            ):
                env = {} if explicit is None else {"HCCL_BUFFSIZE": explicit}
                with (
                    patch.dict(os.environ, env, clear=True),
                    patch.object(
                        sys,
                        "argv",
                        argv + (["--tune-explicit"] if tune_explicit else []),
                    ),
                    patch.object(tuner.policy, "load_profile", return_value={}),
                    patch("builtins.print"),
                ):
                    tuner.main()
                stage = json.loads((root / "plan.json").read_text())["stages"][0]
                if explicit and not tune_explicit:
                    self.assertEqual(stage["status"], "explicit value preserved")
                    self.assertEqual(stage["value"], "1000")
                else:
                    self.assertEqual(
                        stage["candidates"], [explicit, 128, 192, 256, 384, 512]
                    )

    def test_restart_child_unsets_hccl(self):
        args = SimpleNamespace(
            model_path="/model", port=30088, host="127.0.0.1", python=sys.executable
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"HCCL_BUFFSIZE": "1000"}),
            patch.object(
                tuner.socket, "create_connection", side_effect=ConnectionRefusedError
            ),
            patch.object(
                tuner.subprocess,
                "Popen",
                side_effect=RuntimeError("stop after env capture"),
            ) as popen,
        ):
            tuner.run_case(args, [], {"HCCL_BUFFSIZE": None}, Path(directory), 2)
            self.assertNotIn("HCCL_BUFFSIZE", popen.call_args.kwargs["env"])

    def test_hccl_rounding_and_clamping(self):
        self.assertEqual(auto.hccl_buffer_mb(1, 1, 2), 64)
        self.assertEqual(auto.hccl_buffer_mb(1 << 20, 6144, 2), 1024)
        self.assertEqual(auto.hccl_buffer_mb(16384, 6144, 2, headroom=1), 192)
        self.assertEqual(auto.hccl_buffer_mb(16385, 6144, 2, headroom=1), 224)

    def test_hccl_rejects_bad_parameters(self):
        for kw in (
            {"headroom": 0.5},
            {"headroom": float("nan")},
            {"quantum_mb": 0},
            {"max_mb": 32},
        ):
            with self.assertRaises(ValueError):
                auto.hccl_buffer_mb(16384, 6144, 2, **kw)

    def test_precedence(self):
        self.assertEqual(
            auto.resolve("x", explicit=0, cached=2, automatic=3, default=4).value, 0
        )
        self.assertEqual(
            auto.resolve("x", cached=2, automatic=3, default=4).source, "cached-profile"
        )
        self.assertEqual(auto.resolve("x", automatic=3, default=4).value, 3)
        self.assertEqual(auto.resolve("x", default=4).source, "default")

    def test_cached_unset_is_a_decision(self):
        decision = auto.resolve("TASK_QUEUE_ENABLE", cached=None, default="2")
        self.assertIsNone(decision.value)
        self.assertEqual(decision.source, "cached-profile")

    def test_key_stability_and_material_changes(self):
        c = context()
        self.assertEqual(
            auto.profile_key(c),
            auto.profile_key(
                auto.NpuTuningContext(
                    **json.loads(json.dumps(c.key_data(), sort_keys=True))
                )
            ),
        )
        for kw in (
            {"tp_size": 8},
            {"cann_version": "new"},
            {"page_size": 64},
            {"gate_output_dtype": "bf16"},
            {"gate_scale_alg": 1},
            {"chunked_prefill_size": 8192},
        ):
            self.assertNotEqual(auto.profile_key(c), auto.profile_key(replace(c, **kw)))
        self.assertNotIn("model_path", c.key_data())

    def test_profile_roundtrip_and_invalid_cache(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"SGLANG_NPU_TUNING_CACHE_DIR": directory}),
        ):
            c = context()
            self.assertEqual(auto.load_profile(c), {})
            p = auto.save_profile(
                c, {"hccl_buffsize_mb": 256, "dsa_cp_min_tokens": 2048}
            )
            self.assertEqual(auto.load_profile(c)["hccl_buffsize_mb"], 256)
            for data in (
                "{",
                "null",
                json.dumps({"schema_version": 99}),
                json.dumps(
                    {
                        "schema_version": 1,
                        "context": c.key_data(),
                        "selected": {"dsa_cp_min_tokens": -1},
                    }
                ),
                json.dumps({"schema_version": 1, "context": {}, "selected": {}}),
            ):
                p.write_text(data)
                with self.assertLogs(auto.logger, level="WARNING"):
                    self.assertEqual(auto.load_profile(c), {})

    def test_dcp_budget_bound_for_packed_and_separate(self):
        for row_bytes in (656, (512 + 64) * 2):
            for dcp in (1, 2, 4, 8):
                for slots in (1, 2):
                    budget = 256 * auto.MIB
                    rows = auto.dcp_piece_rows(
                        budget, row_bytes, dcp, slots, 128, 1 << 20
                    )
                    self.assertGreaterEqual(rows, 128)
                    self.assertEqual(rows % 128, 0)
                    self.assertLessEqual(rows * row_bytes * dcp * slots, budget)

    def test_dcp_whole_prefix(self):
        self.assertEqual(
            auto.dcp_piece_rows(256 * auto.MIB, 656, 4, 2, 128, 1000), 1024
        )

    def test_dcp_insufficient_budget(self):
        with self.assertRaisesRegex(ValueError, "ownership cycle"):
            auto.dcp_piece_rows(100, 656, 4, 2, 128, 1000)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch required")
    def test_dcp_grow_only_buffers(self):
        import torch

        runtime = ModuleType("sglang.srt.runtime_context")
        runtime.get_parallel = Mock()
        utils = ModuleType("sglang.srt.utils")
        utils.print_info_once = Mock()
        spec = importlib.util.spec_from_file_location(
            "dcp_layout_policy_test", ROOT / "python/sglang/srt/layers/dcp/layout.py"
        )
        layout = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules, {runtime.__name__: runtime, utils.__name__: utils}
        ):
            spec.loader.exec_module(layout)
        ref = torch.empty((1, 8))
        with patch.object(torch, "empty", wraps=torch.empty) as allocate:
            large = layout.dcp_extend_gather_buffer("scratch", ref, 256)
            small = layout.dcp_extend_gather_buffer("scratch", ref, 128)
            self.assertEqual(small.shape, (128, 8))
            self.assertEqual(small.data_ptr(), large.data_ptr())
            self.assertEqual(allocate.call_count, 1)
            grown = layout.dcp_extend_gather_buffer("scratch", ref, 512)
            self.assertEqual(grown.shape, (512, 8))
            self.assertNotEqual(grown.data_ptr(), large.data_ptr())
            self.assertEqual(allocate.call_count, 2)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch required")
    def test_dcp_cached_common_memory_budget(self):
        import torch

        auto._cached_dcp_scratch_budget.cache_clear()
        self.addCleanup(auto._cached_dcp_scratch_budget.cache_clear)
        device = torch.device("cpu")
        group = object()
        npu = SimpleNamespace(
            mem_get_info=Mock(return_value=(512 * auto.MIB, 1024 * auto.MIB))
        )
        with (
            patch.dict(
                os.environ,
                {"SGLANG_NPU_AUTO_DCP_EXTEND_GATHER_PIECE_ROWS": "1"},
                clear=True,
            ),
            patch.object(torch, "npu", npu, create=True),
            patch.object(
                torch.distributed,
                "all_reduce",
                side_effect=lambda tensor, **kw: tensor.fill_(256 * auto.MIB),
            ) as reduce,
            self.assertLogs(auto.logger, level="INFO") as logs,
        ):
            sizes = []
            for prefix in (1000, 1 << 20):
                budget = auto.dcp_scratch_budget(device=device, dcp_size=4, group=group)
                self.assertEqual(budget, 128 * auto.MIB)
                sizes.append(
                    auto.auto_dcp_gather_rows(
                        prefix_lens=[prefix],
                        dcp_size=4,
                        alignment=128,
                        bytes_per_row=656,
                        budget_bytes=budget,
                        prefetch=True,
                    )
                )
            self.assertEqual(npu.mem_get_info.call_count, 1)
            self.assertEqual(reduce.call_count, 1)
            self.assertEqual(reduce.call_args.kwargs["group"], group)
            self.assertEqual(
                reduce.call_args.kwargs["op"], torch.distributed.ReduceOp.MIN
            )
            self.assertLess(sizes[0], sizes[1])
            self.assertEqual(
                sum("NPU DCP scratch budget:" in line for line in logs.output), 1
            )
            with patch.dict(os.environ, {"SGLANG_NPU_DCP_SCRATCH_BUDGET_MB": "64"}):
                self.assertEqual(
                    auto.dcp_scratch_budget(device=device, dcp_size=4, group=group),
                    64 * auto.MIB,
                )
            auto.dcp_scratch_budget(device=device, dcp_size=2, group=group)
            auto.dcp_scratch_budget(device=device, dcp_size=4, group=object())
            self.assertEqual(reduce.call_count, 4)

    def test_indexer_threshold_includes_padding(self):
        batch = SimpleNamespace(
            input_ids=SimpleNamespace(shape=(1024,)), extend_seq_lens_cpu=[1023]
        )
        query = SimpleNamespace(shape=(1024, 64, 128))
        self.assertEqual(
            auto.effective_prefill_query_tokens(batch),
            auto.effective_prefill_query_tokens(batch, query),
        )
        self.assertEqual(
            auto.effective_prefill_query_tokens(batch, SimpleNamespace(shape=(1023,))),
            1023,
        )
        with patch.dict(
            os.environ, {"SGLANG_NPU_DSA_INDEXER_QUERY_SHARDING_MIN_TOKENS": "1024"}
        ):
            self.assertTrue(
                auto.threshold_allows(
                    "SGLANG_NPU_DSA_INDEXER_QUERY_SHARDING_MIN_TOKENS",
                    auto.effective_prefill_query_tokens(batch),
                )
            )

    def test_dcp_explicit_and_dry_run(self):
        kwargs = dict(
            prefix_lens=[1000],
            dcp_size=4,
            alignment=128,
            bytes_per_row=656,
            budget_bytes=256 * auto.MIB,
            prefetch=True,
        )
        with patch.dict(
            os.environ,
            {"SGLANG_NPU_AUTO_DCP_EXTEND_GATHER_PIECE_ROWS": "1"},
            clear=True,
        ):
            self.assertEqual(auto.auto_dcp_gather_rows(**kwargs), 1024)
            with patch.dict(os.environ, {"SGLANG_NPU_AUTOTUNE_DRY_RUN": "1"}):
                self.assertIsNone(auto.auto_dcp_gather_rows(**kwargs))
            with patch.dict(
                os.environ, {"SGLANG_NPU_DCP_EXTEND_GATHER_PIECE_ROWS": "0"}
            ):
                self.assertIsNone(auto.auto_dcp_gather_rows(**kwargs))

    def test_dcp_budget_includes_appended_extend(self):
        kwargs = dict(
            prefix_lens=[1 << 20],
            dcp_size=4,
            alignment=128,
            bytes_per_row=656,
            budget_bytes=256 * auto.MIB,
            prefetch=True,
            extend_rows=16384,
        )
        with patch.dict(
            os.environ,
            {"SGLANG_NPU_AUTO_DCP_EXTEND_GATHER_PIECE_ROWS": "1"},
            clear=True,
        ):
            rows = auto.auto_dcp_gather_rows(**kwargs)
            self.assertLessEqual((rows + 16384) * 656 * 2, 256 * auto.MIB)
        with patch.dict(
            os.environ,
            {
                "SGLANG_NPU_AUTO_DCP_EXTEND_GATHER_PIECE_ROWS": "1",
                "SGLANG_NPU_DCP_SCRATCH_BUDGET_MB": "1",
            },
            clear=True,
        ):
            with self.assertRaises(ValueError):
                auto.auto_dcp_gather_rows(**{**kwargs, "budget_bytes": auto.MIB})

    def test_threshold_independence(self):
        with patch.dict(
            os.environ,
            {
                "SGLANG_NPU_DSA_CP_MIN_TOKENS": "2048",
                "SGLANG_NPU_DSA_INDEXER_QUERY_SHARDING_MIN_TOKENS": "1024",
            },
            clear=True,
        ):
            self.assertFalse(
                auto.threshold_allows("SGLANG_NPU_DSA_CP_MIN_TOKENS", 1500)
            )
            self.assertTrue(
                auto.threshold_allows(
                    "SGLANG_NPU_DSA_INDEXER_QUERY_SHARDING_MIN_TOKENS", 1500
                )
            )
            self.assertTrue(auto.threshold_allows("unset", 0))

    def test_timing_selection(self):
        self.assertEqual(
            auto.choose_measured({"old": [100] * 20, "new": [90] * 20}, "old"),
            ("new", "measured"),
        )
        self.assertEqual(
            auto.choose_measured({"old": [100] * 20, "new": [99.5] * 20}, "old"),
            ("old", "inconclusive"),
        )
        self.assertEqual(
            auto.choose_measured({"old": [90, 110] * 10, "new": [95] * 20}, "old"),
            ("old", "inconclusive"),
        )

    def test_initialization_explicit_cache_and_dry_run(self):
        c = context()
        from types import SimpleNamespace

        for dry in (False, True):
            with (
                tempfile.TemporaryDirectory() as directory,
                patch.dict(
                    os.environ,
                    {
                        "SGLANG_NPU_TUNING_CACHE_DIR": directory,
                        "SGLANG_NPU_AUTO_HCCL_BUFFSIZE": "1",
                        "SGLANG_NPU_AUTO_STREAM_THRESHOLDS": "1",
                        "SGLANG_NPU_AUTOTUNE_DRY_RUN": str(int(dry)),
                    },
                    clear=True,
                ),
                patch.object(auto, "context_from_server_args", return_value=c),
            ):
                auto.save_profile(
                    c, {"hccl_buffsize_mb": 384, "dsa_cp_min_tokens": 2048}
                )
                auto.initialize(SimpleNamespace())
                self.assertEqual(
                    os.environ.get("HCCL_BUFFSIZE"), None if dry else "384"
                )
                self.assertEqual(
                    os.environ.get("SGLANG_NPU_DSA_CP_MIN_TOKENS"),
                    None if dry else "2048",
                )
                with patch.dict(
                    os.environ,
                    {"HCCL_BUFFSIZE": "128", "SGLANG_NPU_DSA_CP_MIN_TOKENS": "0"},
                ):
                    auto.initialize(SimpleNamespace())
                    self.assertEqual(os.environ["HCCL_BUFFSIZE"], "128")
                    self.assertEqual(os.environ["SGLANG_NPU_DSA_CP_MIN_TOKENS"], "0")

    def test_current_shape_dry_run_does_not_set_hccl(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {
                    "SGLANG_NPU_TUNING_CACHE_DIR": directory,
                    "SGLANG_NPU_AUTO_HCCL_BUFFSIZE": "1",
                    "SGLANG_NPU_AUTOTUNE_DRY_RUN": "1",
                },
                clear=True,
            ),
            patch.object(auto, "context_from_server_args", return_value=context()),
            self.assertLogs(auto.logger, level="INFO") as logs,
        ):
            auto.initialize(SimpleNamespace())
            self.assertNotIn("HCCL_BUFFSIZE", os.environ)
            output = "\n".join(logs.output)
            self.assertIn("HCCL_BUFFSIZE=256", output)
            self.assertIn("source=deterministic-auto", output)
            self.assertIn("collective_bytes=201326592", output)

    def test_disabled_initialization(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(
                auto,
                "context_from_server_args",
                side_effect=AssertionError("must not inspect model"),
            ),
        ):
            auto.initialize(None)

    def test_metric_parser(self):
        self.assertEqual(
            tuner.parse_metrics(
                "Mean TTFT (ms): 100.0\nMedian TTFT (ms): 95.5\nSuccessful requests: 17"
            ),
            {"mean_ttft": 100.0, "median_ttft": 95.5, "completed": 17},
        )
        with self.assertRaises(ValueError):
            tuner.parse_metrics("Median TTFT (ms): 0")

    def test_stage_rejects_forward_regression_and_failed_baseline(self):
        records = [
            {"status": "ok", "runs": [{"median_ttft": t, "forward_mean_ms": f}] * 3}
            for t, f in ((100, 10), (90, 11))
        ]
        self.assertEqual(
            tuner.rank_stage(records, "median_ttft", 0), (0, "forward time regressed")
        )
        records[0]["status"] = "oom"
        self.assertEqual(tuner.rank_stage(records, "median_ttft", 0)[0], 0)

    def test_restart_stage_candidates(self):
        from types import SimpleNamespace

        args = SimpleNamespace(chunk_max=32768, concurrency_candidates=None)
        self.assertEqual(
            tuner.candidates("buffer", context(), args)[1], [128, 192, 256, 384, 512]
        )
        self.assertEqual(
            tuner.candidates("chunk", context(), args)[1], [8192, 16384, 32768]
        )
        with self.assertRaises(ValueError):
            tuner.candidates("concurrency", context(), args)

    def test_process_group_cleanup(self):
        import subprocess

        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        tuner.stop(process)
        self.assertIsNotNone(process.poll())

    def test_option_replacement(self):
        self.assertEqual(
            tuner.set_option(["--page-size=64", "--tp-size", "4"], "--page-size", 128),
            ["--tp-size", "4", "--page-size", "128"],
        )


if __name__ == "__main__":
    unittest.main()
