"""Guard the A3 startup failures and completion-tracked weight lifetime."""

import ast
import importlib.util
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.hardware_backend.npu.online_quant_config import (
    resolve_online_quantization,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[5]


def definitions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *nodes], type_ignores=[])
            ),
            path,
            "exec",
        ),
        namespace,
    )
    return namespace


class TestOnlineConfiguration(CustomTestCase):
    def test_preset_and_overrides_keep_experts_compressed(self):
        config = resolve_online_quantization("w4a4_int,gate:int8,qkv:int8")
        self.assertEqual(config.integer_mode(), "w4a4_int")
        self.assertEqual(config.scheme("gate"), "W8A8_DYNAMIC")
        self.assertEqual(config.scheme("qkv"), "W8A8_DYNAMIC")
        self.assertEqual(config.scheme("dense"), "W8A8_DYNAMIC")
        self.assertIsNone(resolve_online_quantization("gate:int8").integer_mode())

    def test_ambiguous_and_unsupported_requests_fail(self):
        for request in (
            "w8a8_int,w4a4_int",
            "gate:int8,w8a8_int",
            "gate:int8,gate:int8",
            "gate:mxfp8",
        ):
            with self.subTest(request=request), self.assertRaises(ValueError):
                resolve_online_quantization(request)


class TestStreamingDenseLifecycle(CustomTestCase):
    def loader(self):
        from sglang.srt.hardware_backend.npu.quantization.online_quantization import (
            NPUOnlineDenseWeightLoader,
            get_npu_online_integer_quant_spec,
        )
        from sglang.srt.layers.parameter import ModelWeightParameter

        layer = torch.nn.Module()
        converted = []

        def load(param, tensor, loaded_shard_id=None):
            rows = (
                slice(None)
                if loaded_shard_id is None
                else slice(loaded_shard_id * 2, (loaded_shard_id + 1) * 2)
            )
            param.data[rows].copy_(tensor)

        def quantize():
            converted.append(layer.weight.detach().clone())
            layer.weight = torch.nn.Parameter(
                layer.weight.to(torch.int8), requires_grad=False
            )

        loader = NPUOnlineDenseWeightLoader(
            layer,
            torch.bfloat16,
            load,
            get_npu_online_integer_quant_spec("w8a8_int"),
            quantize,
        )
        weight = ModelWeightParameter(
            data=torch.empty((4, 4), dtype=torch.bfloat16, device="meta"),
            input_dim=1,
            output_dim=0,
            weight_loader=loader.weight_loader,
        )
        layer.weight = weight
        loader.register_source(weight)
        return layer, loader, weight, converted

    def test_reordered_shards_convert_release_and_reload(self):
        layer, loader, initial, converted = self.loader()
        self.assertEqual(layer.weight.device.type, "meta")
        with self.assertRaisesRegex(RuntimeError, "completely loaded"):
            loader.finish_post_load()
        loader.weight_loader(initial, torch.full((2, 4), 2, dtype=torch.bfloat16), 1)
        self.assertEqual(layer.weight.device.type, "cpu")
        self.assertFalse(converted)
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            loader.weight_loader(initial, torch.ones((2, 4), dtype=torch.bfloat16), 1)
        loader.weight_loader(initial, torch.ones((2, 4), dtype=torch.bfloat16), 0)
        self.assertEqual(layer.weight.dtype, torch.int8)
        torch.testing.assert_close(
            converted[0],
            torch.tensor([[1] * 4] * 2 + [[2] * 4] * 2, dtype=torch.bfloat16),
        )
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            loader.weight_loader(initial, torch.ones((4, 4), dtype=torch.bfloat16))
        loader.finish_post_load()
        loader.weight_loader(layer.weight, torch.full((4, 4), 3, dtype=torch.bfloat16))
        self.assertEqual(len(converted), 2)
        self.assertEqual(layer.weight.dtype, torch.int8)
        self.assertFalse(any(p.is_floating_point() for p in layer.parameters()))

    def test_partial_groups_stay_on_host_and_overflow_fails(self):
        layers = [self.loader() for _ in range(4)]
        for layer, loader, weight, _ in layers:
            loader.weight_loader(weight, torch.ones((2, 4), dtype=torch.bfloat16), 0)
            self.assertEqual(layer.weight.device.type, "cpu")
        layer, loader, weight, _ = layers[0]
        with self.assertRaisesRegex(RuntimeError, "overflow"):
            loader.weight_loader(weight, torch.ones((4, 4), dtype=torch.bfloat16))


class TestOnlineOperandReuse(CustomTestCase):
    def test_fused_norm_operand_is_not_quantized_again(self):
        from sglang.srt.hardware_backend.npu.moe.contracts import (
            INT8,
            QuantizedOperand,
            unpack_operand,
        )

        tree = ast.parse(
            (
                ROOT
                / "python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py"
            ).read_text()
        )
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "_NPUOnlineIntegerLinearMethod"
        )
        method = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "apply"
        )
        ns = dict(
            torch=torch,
            QuantizedOperand=QuantizedOperand,
            unpack_operand=unpack_operand,
            INT8=INT8,
            Optional=__import__("typing").Optional,
        )
        exec(
            compile(
                ast.Module(body=[method], type_ignores=[]), "<online-apply>", "exec"
            ),
            ns,
        )
        q = torch.tensor([[1, 2], [3, 4]], dtype=torch.int8)
        scale = torch.tensor([0.5, 0.25])
        operand = QuantizedOperand(q, scale, INT8, (2, 2))
        layer = SimpleNamespace(
            weight=torch.eye(2, dtype=torch.int8), weight_scale=torch.ones(2)
        )

        def matmul(x, w, ws, **kwargs):
            return ((x.float() @ w.float()) * kwargs["pertoken_scale"][:, None]).to(
                kwargs["output_dtype"]
            )

        with (
            patch.object(QuantizedOperand, "record_stream"),
            patch.object(
                torch.ops.npu, "npu_quant_matmul", side_effect=matmul, create=True
            ),
            patch.object(
                torch.ops.npu,
                "npu_dynamic_quant",
                side_effect=AssertionError("quantized twice"),
                create=True,
            ),
        ):
            result = ns["apply"](SimpleNamespace(), layer, operand)
        torch.testing.assert_close(result, (q.float() * scale[:, None]).bfloat16())


class TestStreamingMoELifecycle(CustomTestCase):
    def test_independent_projections_nonlocal_experts_and_duplicates(self):
        from sglang.srt.hardware_backend.npu.quantization.online_quantization import (
            NPUOnlineMoEWeightLoader,
            get_npu_online_moe_integer_quant_spec,
        )

        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(
            torch.empty((1, 4, 4), dtype=torch.bfloat16, device="meta"),
            requires_grad=False,
        )
        layer.w2_weight = torch.nn.Parameter(
            torch.empty((1, 4, 4), dtype=torch.bfloat16, device="meta"),
            requires_grad=False,
        )
        layer._weight_loader_expert_ids = lambda param, expert: (
            [0] if expert == 0 else []
        )

        def load(param, value, name, shard, expert):
            if "w13" in name:
                param.data[0, slice(0, 2) if shard == "gate" else slice(2, 4)].copy_(
                    value
                )
            else:
                param.data[0].copy_(value)

        specs = {
            p: get_npu_online_moe_integer_quant_spec(p, "w4a4_int")
            for p in ("w13", "w2")
        }
        loader = NPUOnlineMoEWeightLoader(layer, torch.bfloat16, load, specs)
        loader.register_sources(layer.w13_weight, layer.w2_weight)
        converted = []

        def convert(prefix):
            loader._validate_complete(prefix)
            converted.append(prefix)
            name = prefix + "_weight"
            setattr(
                layer,
                name,
                torch.nn.Parameter(
                    getattr(layer, name).to(torch.int8), requires_grad=False
                ),
            )

        layer.w13_kernel = SimpleNamespace(
            process_weights_after_loading=lambda layer, prefix: convert(prefix)
        )
        layer.w2_kernel = SimpleNamespace(
            process_weights_after_loading=lambda layer, prefix: convert(prefix)
        )
        loader.weight_loader(
            layer.w13_weight, torch.ones((2, 4)), "w13_weight", "gate", 1
        )
        self.assertEqual(layer.w13_weight.device.type, "meta")
        loader.weight_loader(
            layer.w2_weight, torch.ones((4, 4)), "w2_weight", "down", 0
        )
        self.assertEqual(converted, ["w2"])
        loader.weight_loader(
            layer.w13_weight, torch.ones((2, 4)), "w13_weight", "up", 0
        )
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            loader.weight_loader(
                layer.w13_weight, torch.ones((2, 4)), "w13_weight", "up", 0
            )
        loader.weight_loader(
            layer.w13_weight, torch.ones((2, 4)), "w13_weight", "gate", 0
        )
        self.assertEqual(converted, ["w2", "w13"])
        self.assertEqual(specs["w13"].mode, "w4a4_int")
        self.assertEqual(specs["w2"].mode, "w8a8_int")


class TestReportedStartupFailures(CustomTestCase):
    def test_radix_eager_attention_is_a_class_method(self):
        # A nested definition parsed successfully but failed on first graph execution.
        tree = ast.parse(
            (ROOT / "python/sglang/srt/layers/radix_attention.py").read_text()
        )
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "RadixAttention"
        )
        self.assertIn(
            "_eager_attention",
            {n.name for n in cls.body if isinstance(n, ast.FunctionDef)},
        )

    def test_compile_wrapper_avoids_python_frame_inspection(self):
        namespace = definitions(
            "python/sglang/srt/distributed/parallel_state.py",
            {"_warn_if_called_from_outside"},
            dict(
                functools=__import__("functools"),
                torch=torch,
                sys=__import__("sys"),
                warnings=__import__("warnings"),
                _ALREADY_WARNED=set(),
                _EXEMPT_CALLERS=("sglang",),
            ),
        )
        fn = namespace["_warn_if_called_from_outside"]("get_tp_group", "tp_group")(
            lambda x: x + 1
        )
        with (
            patch.object(torch.compiler, "is_compiling", return_value=True),
            patch("sys._getframe", side_effect=AssertionError("frame inspection")),
        ):
            self.assertEqual(fn(4), 5)

    def test_int8_metadata_never_calls_mxfp_operator(self):
        tree = ast.parse(
            (
                ROOT
                / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
            ).read_text()
        )
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "AscendAttnBackend"
        )
        init = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        assignment = next(
            n
            for n in ast.walk(init)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Attribute) and t.attr == "quant_indexer_enabled"
                for t in n.targets
            )
        )
        for dtype, expected in ((torch.int8, False), (torch.float8_e4m3fn, True)):
            pool = SimpleNamespace(
                index_k_scale_buffer=object(), index_k_store_dtype=dtype
            )
            ns = dict(
                self=SimpleNamespace(use_mla=True),
                model_runner=SimpleNamespace(token_to_kv_pool=pool),
                torch=torch,
            )
            exec(
                compile(
                    ast.Module(body=[assignment], type_ignores=[]),
                    "<metadata-selection>",
                    "exec",
                ),
                ns,
            )
            self.assertEqual(ns["self"].quant_indexer_enabled, expected)

    def test_actual_tuning_context_reads_current_flags(self):
        import hashlib
        import json
        import os
        import subprocess
        from dataclasses import asdict, dataclass

        from sglang.srt.environ import envs

        namespace = definitions(
            "python/sglang/srt/hardware_backend/npu/autotune.py",
            {"NpuTuningContext", "context_from_server_args"},
            dict(
                dataclass=dataclass,
                asdict=asdict,
                hashlib=hashlib,
                json=json,
                os=os,
                Path=Path,
                subprocess=subprocess,
                _version=lambda package: "test",
                __file__=str(
                    ROOT / "python/sglang/srt/hardware_backend/npu/autotune.py"
                ),
            ),
        )
        hf = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            hidden_size=6144,
            num_hidden_layers=78,
        )
        model = SimpleNamespace(
            hf_text_config=hf,
            hf_config=hf,
            dtype=torch.bfloat16,
            quantization="modelslim",
        )
        args = SimpleNamespace(
            tp_size=16,
            ep_size=1,
            attn_cp_size=1,
            attn_dp_size=1,
            dcp_size=8,
            chunked_prefill_size=4096,
            max_prefill_tokens=4096,
            max_running_requests=16,
            kv_cache_dtype="auto",
            page_size=64,
        )
        with (
            patch(
                "sglang.srt.arg_groups.model_override_base.model_config_of",
                return_value=model,
            ),
            patch.dict(
                os.environ,
                {
                    "SGLANG_NPU_TUNING_DEVICE": "Ascend910_9391",
                    "SGLANG_NPU_TUNING_ARCH": "arch22",
                    "SGLANG_NPU_TUNING_CANN_VERSION": "9.1",
                },
            ),
            patch.object(subprocess, "check_output", return_value="test-commit"),
        ):
            result = namespace["context_from_server_args"](args)
        self.assertEqual(result.tp_size, 16)
        self.assertEqual(result.hidden_size, 6144)
        self.assertEqual(len(result.experiment_signature), 64)


class TestOnlineMlaWeightContract(CustomTestCase):
    def test_decomposition_reads_output_first_not_matmul_storage(self):
        from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
            NPUOnlineW8A8Int8LinearMethod,
        )

        method = NPUOnlineW8A8Int8LinearMethod()
        source = (
            torch.arange(896 * 512, dtype=torch.int64)
            .reshape(896, 512)
            .remainder(127)
            .to(torch.int8)
        )
        scale = torch.linspace(0.01, 0.1, 896)
        layer = SimpleNamespace(
            weight=source.t().contiguous(),
            weight_scale=scale,
            _npu_online_dense_loader=SimpleNamespace(params_dtype=torch.bfloat16),
        )
        logical = method.logical_weight(layer)
        expected = source.to(torch.bfloat16) * scale.to(torch.bfloat16)[:, None]
        torch.testing.assert_close(logical, expected, rtol=0, atol=0)
        key, value = logical.unflatten(0, (-1, 448)).split([192, 256], dim=1)
        self.assertEqual(key.shape, (2, 192, 512))
        self.assertEqual(value.shape, (2, 256, 512))

    def test_unquantized_shared_expert_accepts_null_config(self):
        tree = ast.parse((ROOT / "python/sglang/srt/models/deepseek_v2.py").read_text())
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DeepseekV2MoE"
        )
        assignment = next(
            node
            for node in ast.walk(cls)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "is_packed_weight"
                for target in node.targets
            )
        )
        for config, expected in (
            (None, False),
            (SimpleNamespace(get_name=lambda: "awq"), True),
        ):
            mlp = SimpleNamespace(
                gate_up_proj=SimpleNamespace(
                    quant_method=SimpleNamespace(quant_config=config)
                )
            )
            namespace = dict(self=SimpleNamespace(shared_experts=mlp))
            exec(
                compile(
                    ast.Module(body=[assignment], type_ignores=[]),
                    "<shared-expert>",
                    "exec",
                ),
                namespace,
            )
            self.assertEqual(namespace["is_packed_weight"], expected)


class TestDispatcherRefresh(CustomTestCase):
    def test_native_combine_honors_upstream_final_gather_state(self):
        from sglang.srt.hardware_backend.npu.moe.standard_dispatch import (
            set_finalize_all_gather,
        )
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.token_dispatcher.ascend_tp import (
            AscendTPCombineInput,
            AscendTPDispatcher,
        )

        layer = torch.nn.Module()
        set_finalize_all_gather(layer)
        config = MoeRunnerConfig(num_experts=2, top_k=1, layer=layer)
        dispatcher = AscendTPDispatcher(config)
        dispatcher._dispatch_output = SimpleNamespace(
            topk_weights=torch.ones(2, 1),
            topk_ids=torch.zeros(2, 1, dtype=torch.int32),
            expanded_row_idx=torch.arange(2),
        )
        dispatcher.finalize = SimpleNamespace(
            _finalize_routing=lambda value, **kwargs: value + 1
        )
        with (
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_tp.get_parallel",
                return_value=SimpleNamespace(tp_size=2),
            ),
            patch(
                "sglang.srt.distributed.communication_op.tensor_model_parallel_all_gather",
                side_effect=lambda value, dim: torch.cat((value, value), dim),
            ),
        ):
            result = dispatcher.combine(AscendTPCombineInput(torch.ones(2, 4)))
        torch.testing.assert_close(result, torch.full((2, 8), 2.0), rtol=0, atol=0)
        self.assertIsNone(dispatcher._dispatch_output)


if __name__ == "__main__":
    unittest.main()
