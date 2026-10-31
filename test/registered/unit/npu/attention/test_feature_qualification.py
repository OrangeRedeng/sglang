"""Regressions for GLM qualification attribution and graph metadata reuse."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[5]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "benchmark/prefill_1m" / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestFeatureQualification(CustomTestCase):
    def test_megamoe_rejects_installed_operator_keyword_mismatch(self):
        import sglang.srt.layers.quantization  # noqa: F401
        from sglang.srt.hardware_backend.npu.moe.fuseep import _check_mega_moe_keywords

        def old_operator(x):
            pass

        def compatible_operator(x, activation_params):
            pass

        source = (
            "def run(x, params):\n    return mega_moe(x=x, activation_params=params)\n"
        )
        with self.assertRaisesRegex(RuntimeError, "activation_params"):
            _check_mega_moe_keywords(old_operator, source)
        self.assertEqual(
            _check_mega_moe_keywords(compatible_operator, source),
            ["activation_params", "x"],
        )

    def test_benchmark_requires_activation_except_baseline(self):
        checks = load("checks", "feature_checks.py")
        result = dict(status="pass", quality="verified", missing=[])
        self.assertFalse(checks.benchmark_qualified(result))
        self.assertTrue(checks.benchmark_qualified(result, baseline=True))
        result.update(activation="verified")
        self.assertTrue(checks.benchmark_qualified(result))
        result.update(status="smoke_pass")
        self.assertFalse(checks.benchmark_qualified(result))

    def test_reference_oom_is_preserved_and_not_candidate_failure(self):
        harness = load("harness", "test_glm53_features.py")
        checks = load("checks", "feature_checks.py")
        args = SimpleNamespace(
            model_path="weights",
            checkpoint_format="modelslim",
            tp=16,
            dcp=16,
            context_length=8192,
            chunk=4096,
            mem_fraction=0.85,
            port=31000,
        )
        case = dict(
            id="serve-baseline",
            stage="serve",
            scenario={},
            checkpoint_path="weights",
            checkpoint_format="modelslim",
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)

            def fail_reference(args, case, destination, env):
                (destination / "server.log").write_text(
                    "torch.OutOfMemoryError: NPU out of memory"
                )
                raise RuntimeError("server exited -9")

            with (
                patch.dict(sys.modules, feature_checks=checks),
                patch.object(harness, "serve", side_effect=fail_reference),
            ):
                first = harness.validate_serving(args, case, directory, {}, {})
                second = harness.validate_serving(args, case, directory, {}, {})
            self.assertEqual(first["status"], "capacity_blocked")
            self.assertEqual(first["failure_phase"], "reference")
            self.assertEqual(first, second)
            self.assertTrue((Path(first["artifacts"]) / "server.log").is_file())
            self.assertTrue((Path(first["artifacts"]) / "exception.log").is_file())

    def test_fuseep_uses_explicit_w8a8_checkpoint_and_shared_kv_cap(self):
        harness = load("harness", "test_glm53_features.py")
        checks = load("checks", "feature_checks.py")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            description = path / "quant_model_description.json"
            description.write_text(
                json.dumps({"model.layers.0.mlp.experts.0.gate_proj.weight": "W4A8"})
            )
            self.assertEqual(checks.expert_quantization(path), {"W4A8"})
            args = SimpleNamespace(
                model_path="w4a8",
                checkpoint_format="modelslim",
                w8a8_model_path=path,
                context_length=8192,
                chunk=4096,
                mem_fraction=0.85,
                tp=16,
                port=31000,
                kv_rows=8192,
            )
            command = harness.server_command(args, dict(fuseep=1))
            self.assertEqual(command[command.index("--model-path") + 1], str(path))
            self.assertEqual(command[command.index("--max-total-tokens") + 1], "8192")
            case = dict(
                id="serve-fuseep-1", scenario=dict(fuseep=1), checkpoint_path=str(path)
            )
            with patch.dict(sys.modules, feature_checks=checks):
                result = harness.validate_serving(args, case, path, {}, {})
            self.assertEqual(result["status"], "dependency_blocked")

    def test_candidate_crash_and_numerical_failure_are_distinct(self):
        checks = load("checks", "feature_checks.py")
        for phase in ("candidate", "parity"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                harness = load("harness", "test_glm53_features.py")
                args = SimpleNamespace(
                    model_path="weights",
                    checkpoint_format="modelslim",
                    tp=16,
                    dcp=16,
                    context_length=8192,
                    chunk=4096,
                    mem_fraction=0.85,
                    port=31000,
                )
                case = dict(
                    id="serve-baseline",
                    stage="serve",
                    scenario={},
                    checkpoint_path="weights",
                    checkpoint_format="modelslim",
                )

                def serve(args, case, directory, env):
                    reference = directory.name == "reference"
                    if not reference and phase == "candidate":
                        (directory / "server.log").write_text(
                            "RuntimeError: kernel failed"
                        )
                        raise RuntimeError("server exited -9")
                    logprob = -3.0 if reference else -3.3
                    (directory / "responses.json").write_text(
                        json.dumps(
                            [{"meta_info": {"input_token_logprobs": [[logprob, 1]]}}]
                        )
                    )
                    return dict(
                        status="smoke_pass",
                        quality="unverified",
                        activation="unverified",
                        missing=[],
                    )

                with (
                    patch.dict(sys.modules, feature_checks=checks),
                    patch.object(harness, "serve", side_effect=serve),
                ):
                    result = harness.validate_serving(
                        args, case, Path(temporary), {}, {}
                    )
                self.assertEqual(result["status"], "fail")
                self.assertEqual(result["failure_phase"], phase)
                self.assertNotEqual(result.get("quality"), "verified")

    def test_online_logical_weight_has_mla_geometry_and_source_dtype(self):
        import sglang.srt.layers.quantization  # noqa: F401 - Initialize registrations before native methods.
        from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
            NPUOnlineW8A8Int8LinearMethod,
        )

        layer = SimpleNamespace(
            weight=torch.arange(512 * 448, dtype=torch.int32)
            .remainder(17)
            .to(torch.int8)
            .reshape(512, 448),
            weight_scale=torch.linspace(0.01, 0.02, 448),
            _npu_online_dense_loader=SimpleNamespace(params_dtype=torch.bfloat16),
        )
        restored = NPUOnlineW8A8Int8LinearMethod().logical_weight(layer)
        self.assertEqual(restored.shape, (448, 512))
        self.assertEqual(restored.dtype, torch.bfloat16)
        expected = (layer.weight.float().t() * layer.weight_scale[:, None]).bfloat16()
        torch.testing.assert_close(restored, expected, atol=0, rtol=0)

    def test_graph_lengths_reuse_addresses_for_ragged_and_empty_requests(self):
        from sglang.srt.layers.attention.dsa import dsa_npu_indexer as indexer

        batch = SimpleNamespace(
            input_ids=torch.zeros(8),
            batch_size=3,
            extend_seq_lens=torch.tensor([2, 6, 0]),
            seq_lens=torch.tensor([12, 26, 0]),
        )
        cache = {}
        with patch.object(
            indexer,
            "get_parallel",
            return_value=SimpleNamespace(attn_tp_size=2, attn_tp_rank=1),
        ):
            first = indexer.prepare_graph_indexer_query_shard(batch, cache)
            pointer = first.actual_seq_lengths_q.data_ptr()
            self.assertEqual(first.actual_seq_lengths_q.tolist(), [0, 4, 4])
            batch.extend_seq_lens.copy_(torch.tensor([6, 0, 2]))
            batch.seq_lens.copy_(torch.tensor([16, 0, 32]))
            second = indexer.prepare_graph_indexer_query_shard(batch, cache)
        self.assertIs(first, second)
        self.assertEqual(second.actual_seq_lengths_q.data_ptr(), pointer)
        self.assertEqual(second.actual_seq_lengths_q.tolist(), [2, 2, 4])
        self.assertEqual(second.actual_seq_lengths_kv.tolist(), [16, 0, 32])
        with (
            patch.object(
                indexer,
                "get_attn_tp_context",
                return_value=SimpleNamespace(input_scattered=True),
            ),
            patch.object(indexer, "has_forward_context", return_value=True),
            patch.object(
                indexer,
                "get_attn_backend",
                return_value=SimpleNamespace(
                    graph_mode=True, npu_graph_indexer_shard=second
                ),
            ),
        ):
            self.assertIsNone(
                indexer._get_indexer_query_shard(
                    SimpleNamespace(npu_dsa_cp_token_sharded=False), 8
                )
            )

    def test_full_width_q_retains_dynamic_activation_quantization(self):
        import sglang.srt.layers.quantization  # noqa: F401
        from sglang.srt.hardware_backend.npu.modules import (
            dsa_token_residual as residual,
        )
        from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
            NPUW8A8Int8DynamicLinearMethod,
        )

        kernel = NPUW8A8Int8DynamicLinearMethod()
        weights = [
            torch.tensor([[11, 23], [31, 17]], dtype=torch.int8),
            torch.tensor([[7, 19], [13, 29]], dtype=torch.int8),
        ]
        scales = [torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4])]
        layers = [
            SimpleNamespace(weight=w, weight_scale=s, quant_method=kernel)
            for w, s in zip(weights, scales)
        ]
        group = SimpleNamespace(
            all_gather=lambda value, dim: torch.cat(
                weights if value.ndim == 2 else scales, dim=dim
            )
        )
        attention = SimpleNamespace(q_b_proj=layers[0])

        def quantize(x):
            scale = x.float().abs().amax(-1) / 127
            return (x.float() / scale[:, None]).round().to(torch.int8), scale

        def matmul(x, weight, scale, *, pertoken_scale, bias, output_dtype):
            return (
                (x.float() @ weight.float()) * pertoken_scale[:, None] * scale[None, :]
            ).to(output_dtype)

        x = torch.tensor([[0.13, 0.91], [0.73, -0.19]], dtype=torch.bfloat16)
        with (
            patch.object(torch.ops.npu, "npu_dynamic_quant", quantize, create=True),
            patch.object(torch.ops.npu, "npu_quant_matmul", matmul, create=True),
        ):
            self.assertTrue(residual.prepare_quantized_q_projection(attention, group))
            actual = residual.full_width_q_projection(attention, x)
            expected = torch.cat([kernel.apply(layer, x) for layer in layers], dim=1)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(attention.q_b_proj_qrep_weight.dtype, torch.int8)


if __name__ == "__main__":
    unittest.main()
