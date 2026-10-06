"""CPU-only policy coverage; importing the runtime is unnecessary."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

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

    def test_dcp_explicit_and_dry_run(self):
        kwargs = dict(
            prefix_lens=[1000],
            dcp_size=4,
            alignment=128,
            bytes_per_row=656,
            free_bytes=1 << 30,
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
            free_bytes=1 << 30,
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
                auto.auto_dcp_gather_rows(**kwargs)

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
