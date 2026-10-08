"""CPU coverage for automatic HCCL sizing and explicit override precedence."""

import importlib.util
import os
import unittest
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
    "npu_hccl_config_test",
    ROOT / "python/sglang/srt/hardware_backend/npu/hccl_config.py",
)
auto = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(auto)


class TestNpuHcclConfig(unittest.TestCase):
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

    def test_startup_sizing_and_overrides(self):
        resolver = ModuleType("sglang.srt.arg_groups.model_override_base")
        args = SimpleNamespace(max_prefill_tokens=16384, chunked_prefill_size=16384)
        for enabled, explicit, chunk, dtype, expected in (
            ("0", None, 16384, "torch.bfloat16", None),
            ("1", "128", 16384, "torch.bfloat16", "128"),
            ("1", None, 16384, "torch.bfloat16", "256"),
            ("1", None, 8192, "torch.bfloat16", "128"),
            ("true", None, -1, "torch.bfloat16", "256"),
            ("1", None, 16384, "torch.float32", "480"),
        ):
            with self.subTest(
                enabled=enabled, explicit=explicit, chunk=chunk, dtype=dtype
            ):
                env = {"SGLANG_NPU_AUTO_HCCL_BUFFSIZE": enabled}
                if explicit is not None:
                    env["HCCL_BUFFSIZE"] = explicit
                args.chunked_prefill_size = chunk
                resolver.model_config_of = Mock(
                    return_value=SimpleNamespace(
                        dtype=dtype, hf_text_config=SimpleNamespace(hidden_size=6144)
                    )
                )
                with (
                    patch.dict(os.environ, env, clear=True),
                    patch.dict("sys.modules", {resolver.__name__: resolver}),
                ):
                    auto.initialize_hccl_buffer(args)
                    self.assertEqual(os.environ.get("HCCL_BUFFSIZE"), expected)
                    if enabled == "0" or explicit is not None:
                        resolver.model_config_of.assert_not_called()
