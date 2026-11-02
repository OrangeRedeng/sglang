"""Regressions for GLM qualification attribution and graph metadata reuse."""

import contextlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
    def test_dcp_cache_geometry_and_chunk_aware_probe(self):
        checks = load("cache_geometry", "feature_checks.py")
        info = dict(
            page_size=128,
            dcp_size=16,
            chunked_prefill_size=16384,
            context_length=1048576,
        )
        self.assertEqual(checks.cache_page_size(info), 2048)
        self.assertEqual(checks.minimum_cached_prefix(990000, info), 989184)
        self.assertEqual(checks.dcp_probe_lengths(info), (18432, 1024, 16384))
        prime, cached = list(checks.dcp_probe_payloads(info))
        from sglang.srt.managers.schedule_batch import Req

        request_state = SimpleNamespace(
            dllm_config=None,
            return_logprob=True,
            logprob_start_len=0,
            is_dllm=lambda: False,
        )
        self.assertEqual(
            Req._compute_max_prefix_len(request_state, len(cached["input_ids"])), 0
        )
        request_state.logprob_start_len = cached["logprob_start_len"]
        self.assertEqual(
            Req._compute_max_prefix_len(request_state, len(cached["input_ids"])),
            len(prime["input_ids"]),
        )
        self.assertEqual(prime["logprob_start_len"], 0)
        self.assertEqual(cached["logprob_start_len"], len(prime["input_ids"]))
        self.assertGreater(
            cached["logprob_start_len"],
            checks.minimum_cached_prefix(len(prime["input_ids"]), info),
        )
        self.assertEqual(len(cached["input_ids"]) - len(prime["input_ids"]), 1024)
        short = dict(info, chunked_prefill_size=4096, context_length=8192)
        self.assertEqual(checks.dcp_probe_lengths(short), (6144, 1024, 4096))
        with self.assertRaises(ValueError):
            checks.cache_page_size({"page_size": 128})
        with self.assertRaises(ValueError):
            checks.dcp_probe_lengths(dict(short, context_length=4096))

    def test_failed_cached_validation_keeps_raw_timing_and_completed_cold_point(self):
        bench_path = str(ROOT / "benchmark/prefill_1m")
        with patch.object(sys, "path", [bench_path, *sys.path]):
            bench = load("sample_retention", "feature_benchmarks.py")
        args = SimpleNamespace(
            long_lengths=[10000],
            long_concurrency=[1],
            long_requests=1,
            benchmark_repeats=2,
            context_length=10032,
            request_timeout=30,
            long_warmup="once",
        )
        state = {"primed": False}

        def request(url, *args, **kwargs):
            if url.endswith("/get_server_info"):
                return {
                    "page_size": 128,
                    "dcp_size": 16,
                    "max_total_num_tokens": 1000000,
                }
            state["primed"] = False

        def send(base, ids, *args):
            cached = 6144 if state["primed"] and len(ids) == 10000 else 0
            if len(ids) == 9900:
                state["primed"] = True
            return dict(
                ttft_s=42.0, e2e_s=43.0, cached_tokens=cached, prompt_tokens=len(ids)
            )

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "config.json").write_text(json.dumps({"vocab_size": 2048}))
            with (
                patch.object(bench, "run_one", side_effect=send),
                self.assertRaisesRegex(RuntimeError, "cached only"),
            ):
                bench.long_matrix(args, "local", request, directory, directory)
            samples = json.loads(
                (directory / "long-10000-1-prefix99-samples.json").read_text()
            )
            self.assertEqual(samples[0]["requests"][0]["ttft_s"], 42.0)
            self.assertEqual(samples[0]["validation_status"], "fail")
            cold = json.loads((directory / "long-results.json").read_text())
            self.assertEqual(len(cold), 1)
            self.assertEqual(cold[0]["cache"], "cold")
            self.assertEqual(cold[0]["status"], "pass")

    def test_timed_failure_returns_completed_points_to_summary(self):
        harness = load("partial_results", "test_glm53_features.py")
        checks = load("partial_checks", "feature_checks.py")
        args = SimpleNamespace(
            model_path="weights",
            checkpoint_format="modelslim",
            tp=16,
            dcp=16,
            context_length=1048576,
            chunk=16384,
            mem_fraction=0.88,
            port=31000,
        )
        case = dict(
            id="long-dcp-baseline",
            stage="long",
            scenario={"dcp": True},
            checkpoint_path="weights",
            checkpoint_format="modelslim",
        )
        response = [{"meta_info": {"input_token_logprobs": [[None, 1], [0.2, 2]]}}]
        point = {"length": 1000000, "cache": "cold", "status": "pass"}

        def serve(args, case, destination, env):
            (destination / "responses.json").write_text(json.dumps(response))
            if case["stage"] == "long":
                (destination / "long-results.json").write_text(json.dumps([point]))
                raise RuntimeError("fixture cached phase failed")
            return dict(status="smoke_pass", activation="unverified", missing=[])

        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.object(harness, "serve", side_effect=serve),
                patch.object(
                    sys, "path", [str(ROOT / "benchmark/prefill_1m"), *sys.path]
                ),
                patch.dict(sys.modules, feature_checks=checks),
            ):
                result = harness.validate_serving(args, case, Path(temporary), {}, {})
            self.assertEqual(result["status"], "fail")
            self.assertEqual(result["failure_phase"], "benchmark")
            self.assertEqual(result["benchmarks"], [point])
            self.assertTrue((Path(temporary) / "measured-exception.log").exists())

    def test_megamoe_comm_buffer_override_reaches_moe_but_not_dcp_group(self):
        from sglang.srt.distributed import parallel_state

        matrix = load("megamoe_buffer", "test_glm53_features.py")
        megamoe = matrix.SCENARIOS["megamoe"]["env"]
        self.assertEqual(megamoe["DEEPEP_HCCL_BUFFSIZE"], "768")
        self.assertNotIn("DEEPEP_HCCL_BUFFSIZE", matrix.SCENARIOS["fuseep-1"]["env"])
        native = SimpleNamespace(
            _C=SimpleNamespace(
                _distributed_c10d=SimpleNamespace(
                    ProcessGroupHCCL=SimpleNamespace(Options=lambda: SimpleNamespace())
                )
            )
        )
        with (
            patch.object(parallel_state, "_is_npu", True),
            patch.dict(sys.modules, {"torch_npu": native}),
            patch.dict("os.environ", dict(megamoe, HCCL_BUFFSIZE="200")),
        ):
            self.assertEqual(
                parallel_state.get_torch_distributed_pg_options("moe_ep").hccl_config[
                    "hccl_buffer_size"
                ],
                768,
            )
            self.assertEqual(
                parallel_state.get_torch_distributed_pg_options("dcp").hccl_config[
                    "hccl_buffer_size"
                ],
                200,
            )

    def test_long_context_flush_accepts_text_and_metadata_stays_json(self):
        matrix = load("long_http", "test_glm53_features.py")
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value.read.return_value = (
            b"Cache flushed."
        )
        with patch.object(matrix.urllib.request, "build_opener", return_value=opener):
            self.assertEqual(
                matrix.request("http://local/flush_cache", {}, expect_json=False),
                "Cache flushed.",
            )
            with self.assertRaises(json.JSONDecodeError):
                matrix.request("http://local/get_server_info")

    def test_combinations_require_matching_component_qualification(self):
        matrix = load("overnight_combinations", "test_glm53_features.py")
        combos = matrix._combinations
        combined = matrix.SCENARIOS["dcp-io-combined"]
        self.assertEqual(
            set(combined["expected"]),
            {"dcp-packed-read", "dcp-gather-prefetch", "dcp-sfa-overlap"},
        )
        self.assertTrue(all(value == "True" for value in combined["env"].values()))
        case = dict(
            id="long-dcp-io-combined",
            stage="long",
            checkpoint_path="/weights",
            scenario=combined,
        )
        qualified = {}
        self.assertEqual(
            combos.component_blockers(case, qualified), list(combined["components"])
        )
        for component in combined["components"]:
            qualified[combos.qualification_key(case, component)] = True
        self.assertEqual(combos.component_blockers(case, qualified), [])
        for changes in (
            dict(stage="serve"),
            dict(checkpoint_path="/different"),
            dict(scenario=dict(combined, dcp=False)),
        ):
            self.assertTrue(combos.component_blockers(dict(case, **changes), qualified))
        qualified[combos.qualification_key(case, combined["components"][0])] = False
        self.assertTrue(combos.component_blockers(case, qualified))

    def test_int8_finalize_requires_real_nz_storage_and_cann_entrypoint(self):
        from sglang.srt.hardware_backend.npu.moe import tp_fusion

        native = SimpleNamespace(get_npu_format=lambda weight: weight.storage_format)
        weight = SimpleNamespace(
            dtype=torch.int8,
            ndim=3,
            device=SimpleNamespace(type="npu"),
            storage_format=2,
        )
        with patch.dict(sys.modules, {"torch_npu": native}):
            for available in (
                set(),
                {"aclnnGroupedMatmulFinalizeRoutingWeightNz"},
                {"aclnnGroupedMatmulFinalizeRoutingWeightNzV2"},
            ):
                with patch.object(
                    tp_fusion,
                    "supports_cann_symbol",
                    side_effect=lambda name: name in available,
                ):
                    for storage_format in (2, 29, 50):
                        weight.storage_format = storage_format
                        with self.subTest(
                            symbols=available, storage_format=storage_format
                        ):
                            self.assertEqual(
                                tp_fusion.supports_int8_gmm2_finalize(weight),
                                bool(available) and storage_format in (29, 50),
                            )
            for dtype, device in ((torch.int32, "npu"), (torch.int8, "cpu")):
                weight.dtype, weight.device = dtype, SimpleNamespace(type=device)
                self.assertFalse(tp_fusion.supports_int8_gmm2_finalize(weight))

    def test_shared_slot_cancels_post_moe_scale(self):
        from sglang.srt.configs.shared_experts import SharedExpertsConfig
        from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKConfig

        stub = types.ModuleType("sgl_kernel_npu.norm.l1_norm")
        stub.l1_norm = lambda x: x / x.sum(-1, keepdim=True)
        with patch.dict(sys.modules, {"sgl_kernel_npu.norm.l1_norm": stub}):
            from sglang.srt.hardware_backend.npu.moe import topk

        def kernel(logits, k):
            weights, ids = logits.softmax(-1).topk(k)
            return weights, ids, None

        def finalize(**kwargs):
            return StandardTopKOutput(
                kwargs["topk_weights"], kwargs["topk_ids"], kwargs["router_logits"]
            )

        x, logits = torch.ones(2, 4), torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        expert_outputs = torch.tensor([2.0, 4.0, 8.0, 16.0])
        with (
            patch.object(
                torch.ops.npu, "npu_moe_gating_top_k_softmax", kernel, create=True
            ),
            patch.object(topk, "_finalize_topk_npu", side_effect=finalize),
            patch.object(topk, "l1_norm", side_effect=stub.l1_norm),
        ):
            for fused in (False, True):
                for multiplier in (None, 0.5):
                    with self.subTest(fused=fused, multiplier=multiplier):
                        config = TopKConfig(
                            top_k=3,
                            shared_experts=SharedExpertsConfig(
                                num_fused_shared_experts=1
                            ),
                            routed_scaling_factor=2.5,
                            apply_routed_scaling_factor_on_output=fused,
                            fused_shared_experts_scaling_factor=multiplier,
                        )
                        out = topk.fused_topk_npu(x, logits, config)
                        actual = (
                            expert_outputs[out.topk_ids.long()] * out.topk_weights
                        ).sum(-1)
                        if not fused:
                            actual *= 2.5
                        weights, ids = logits.softmax(-1).topk(2)
                        weights /= weights.sum(-1, keepdim=True)
                        expected = (weights * expert_outputs[ids]).sum(-1) * 2.5
                        expected += 16.0 * (1.0 if multiplier is None else multiplier)
                        torch.testing.assert_close(actual, expected)

    def test_split_qk_norm_launcher_stays_outside_fullgraph_compile(self):
        from sglang.srt.hardware_backend.npu.attention.split_qk_norm import (
            _split_qk_norm,
            fused_split_qk_norm,
        )

        stub = types.ModuleType("sgl_kernel_npu.norm.fused_split_qk_norm")
        calls = []

        def external(x, q_norm, k_norm, q_rank, kv_rank, rope_dim, eps):
            self.assertFalse(torch.compiler.is_compiling())
            calls.append((q_norm.bias, k_norm.bias, eps))
            q, k, pe = x.split([q_rank, kv_rank, rope_dim], -1)

            def norm(v, module):
                v = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps)
                return v * module.weight + module.bias

            return (
                norm(q, q_norm).clone(),
                norm(k, k_norm).unsqueeze(1).clone(),
                pe.unsqueeze(1).clone(),
            )

        stub.fused_split_qk_norm = external
        q_norm = SimpleNamespace(weight=torch.ones(4), bias=torch.ones(4))
        k_norm = SimpleNamespace(weight=torch.ones(3), bias=torch.ones(3) * 2)

        def run(x):
            return fused_split_qk_norm(x, q_norm, k_norm, 4, 3, 2, eps=1e-5)

        with (
            patch.dict(sys.modules, {"sgl_kernel_npu.norm.fused_split_qk_norm": stub}),
            contextlib.ExitStack() as stack,
        ):
            library = torch.library.Library("sglang", "IMPL", "CPU")
            stack.callback(library._destroy)
            library.impl("npu_split_qk_norm", _split_qk_norm.__wrapped__)
            compiled = torch.compile(run, backend="eager", fullgraph=True)
            for rows in (1, 2, 7):
                x = torch.randn(rows, 9)
                for actual, expected in zip(compiled(x), run(x)):
                    torch.testing.assert_close(actual, expected)
        self.assertTrue(calls)
        self.assertTrue(all(eps == 1e-5 for _, _, eps in calls))

    def test_token_shard_graph_lengths_keep_addresses_across_replay(self):
        from sglang.srt.layers.attention.dsa import dsa_token_shard as shard
        from sglang.srt.layers.attention.dsa.dsa_token_shard_layout import (
            plan_dsa_token_shard,
        )

        cache = {}
        batch = SimpleNamespace(
            input_ids=torch.zeros(8), batch_size=3, seq_lens=torch.zeros(3)
        )
        pointers = None
        for q, k in (([2, 6, 0], [12, 26, 0]), ([6, 0, 2], [16, 0, 32])):
            plan = plan_dsa_token_shard(q, k, 2, 1)
            with patch.object(shard, "_build_dsa_token_shard_plan", return_value=plan):
                shard.prepare_graph_token_shard_lens(batch, cache)
            buffers = cache[(8, 3)]
            for name in (
                "npu_dsa_token_shard_cu_query_lens",
                "npu_dsa_token_shard_cu_key_lens",
                "npu_dsa_token_shard_key_lens",
            ):
                delattr(batch, name)
            with (
                patch.object(shard, "has_forward_context", return_value=True),
                patch.object(
                    shard,
                    "get_attn_backend",
                    return_value=SimpleNamespace(
                        graph_mode=True, npu_graph_token_shard_lens=buffers
                    ),
                ),
                patch.object(
                    shard.torch,
                    "tensor",
                    side_effect=AssertionError("copy during capture"),
                ),
            ):
                actual = (
                    shard.dsa_token_shard_cumulative_query_lens(batch, plan, "cpu"),
                    shard.dsa_token_shard_cumulative_key_lens(batch, plan, "cpu"),
                    shard.dsa_token_shard_key_lens(batch, plan, "cpu"),
                )
            current = tuple(v.data_ptr() for v in actual)
            if pointers is not None:
                self.assertEqual(pointers, current)
            pointers = current
            self.assertEqual(actual[0].tolist(), shard.cumulative(plan.query_lens))
            self.assertEqual(actual[1].tolist(), shard.cumulative(plan.key_lens))
            self.assertEqual(actual[2].tolist(), plan.key_lens)

    def test_megamoe_legacy_swiglu_preserves_biases_and_rejects_other_activations(self):
        import sglang.srt.layers.quantization  # noqa: F401
        from sglang.srt.hardware_backend.npu.moe.fuseep import (
            _adapt_mega_moe_operator,
            _check_mega_moe_keywords,
        )

        def legacy_operator(x, *, activation="swiglu", l1_bias=None, l2_bias=None):
            return x + l1_bias + l2_bias

        compatible = _adapt_mega_moe_operator(legacy_operator)
        source = "def run(x, a, b):\n    return mega_moe(x=x, activation='swiglu', activation_params=None, l1_bias=a, l2_bias=b)\n"
        self.assertIn("activation_params", _check_mega_moe_keywords(compatible, source))
        x, first, second = torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([4.0])
        actual = compatible(
            x,
            activation="swiglu",
            activation_params=None,
            l1_bias=first,
            l2_bias=second,
        )
        torch.testing.assert_close(actual, torch.tensor([7.0]))
        for activation, parameters in (
            ("situglu", None),
            ("swiglu", {"beta": 4}),
            ("swiglu", {}),
        ):
            with (
                self.subTest(activation=activation, parameters=parameters),
                self.assertRaisesRegex(ValueError, "Legacy MegaMoE"),
            ):
                compatible(
                    x,
                    activation=activation,
                    activation_params=parameters,
                    l1_bias=first,
                    l2_bias=second,
                )
        with self.assertRaises(TypeError):
            compatible(x, activation="swiglu", unknown=None)

    def test_megamoe_modern_api_is_unchanged(self):
        import sglang.srt.layers.quantization  # noqa: F401
        from sglang.srt.hardware_backend.npu.moe.fuseep import _adapt_mega_moe_operator

        def modern_operator(x, *, activation, activation_params):
            return x * activation_params["beta"]

        self.assertIs(_adapt_mega_moe_operator(modern_operator), modern_operator)
        result = _adapt_mega_moe_operator(modern_operator)(
            torch.tensor([2.0]), activation="situglu", activation_params={"beta": 4}
        )
        torch.testing.assert_close(result, torch.tensor([8.0]))

    def test_megamoe_forward_installs_adapter_only_on_selected_strategy(self):
        import sglang.srt.layers.quantization  # noqa: F401
        from sglang.srt.hardware_backend.npu.moe import fuseep

        def operator(x, *, activation="swiglu", l1_bias=None, l2_bias=None):
            return x + l1_bias + l2_bias, None

        class Strategy:
            @staticmethod
            def _load_mega_moe_ops():
                return None, operator

        class Buffer:
            def __init__(self):
                self._fused_strategies = {1: Strategy(), 3: Strategy()}

            def fused_deep_moe(self, **kwargs):
                _, selected = self._fused_strategies[3]._load_mega_moe_ops()
                return selected(
                    kwargs["x"],
                    activation=kwargs["activation"],
                    activation_params=None,
                    l1_bias=kwargs["l1_bias"],
                    l2_bias=kwargs["l2_bias"],
                )

        buffer = Buffer()
        module = types.ModuleType("deep_ep.buffer")
        module.FuseMode = SimpleNamespace(MEGA_MOE=3)
        layer = SimpleNamespace(
            _mega_moe_weights_processed=True,
            num_experts=2,
            w13_weight=None,
            w13_weight_scale=None,
            w2_weight=None,
            w2_weight_scale=None,
            w13_scale_bias=torch.tensor([[2.0]], dtype=torch.bfloat16),
            w2_scale_bias=torch.tensor([[4.0]], dtype=torch.bfloat16),
        )
        topk = SimpleNamespace(
            topk_ids=torch.tensor([[0]]), topk_weights=torch.tensor([[1.0]])
        )
        ops = (None, fuseep._adapt_mega_moe_operator(operator))
        with (
            patch.dict(sys.modules, {"deep_ep.buffer": module}),
            patch.object(fuseep, "_check_mega_moe_support", return_value=ops),
            patch.object(fuseep, "_get_fuseep_buffer", return_value=buffer),
            patch.object(fuseep, "_mega_moe_capacity", return_value=4),
            patch.object(
                fuseep,
                "get_exec",
                return_value=SimpleNamespace(moe=SimpleNamespace(fuseep_mode=3)),
            ),
        ):
            for _ in range(2):
                actual = fuseep.forward_fuseep(
                    layer, torch.tensor([[1.0]], dtype=torch.bfloat16), topk
                )
                torch.testing.assert_close(
                    actual, torch.tensor([[7.0]], dtype=torch.bfloat16)
                )
        self.assertIs(buffer._fused_strategies[1]._load_mega_moe_ops()[1], operator)
        self.assertIs(Strategy._load_mega_moe_ops()[1], operator)

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
