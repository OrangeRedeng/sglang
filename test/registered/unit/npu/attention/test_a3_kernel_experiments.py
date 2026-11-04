"""Regression coverage for DCP shared-stream ordering and experiment launch wiring."""

import ast
import contextlib
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[5]


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "benchmark/prefill_1m" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestKernelExperiments(unittest.TestCase):
    def test_norm_launch_keeps_short_cp1_and_online_gate(self):
        runner, harness = load("run_bde_overnight"), load("test_glm53_features")
        config = dict(
            variant="gate-int8-norm-probe",
            reference_recipe="historical-bde-capacity",
            checkpoint="/weights",
            chunk=32768,
            context=1048576,
            memfrac=0.85,
            maxreq=1,
            kv_rows=67584,
        )
        command, env = runner.variant_launch(harness, ROOT, config)
        self.assertEqual(command[command.index("--context-length") + 1], "16384")
        self.assertEqual(command[command.index("--dcp-size") + 1], "1")
        self.assertIn(
            "gate:w8a8_dynamic", command[command.index("--online-quantization") + 1]
        )
        scenario = harness.scenario_for(config["variant"], config["reference_recipe"])
        with self.assertRaisesRegex(RuntimeError, "add-rmsnorm-quant-int8") as caught:
            runner.timed_activation("", config["variant"], scenario)
        self.assertNotIn("dsa-residual-shard", str(caught.exception))

    def test_dcp_stream_gathers_before_fork_and_joins_before_shared_add(self):
        self._run_dcp_stream(False)

    def test_dcp_boundary_operand_is_gathered_without_requantizing(self):
        self._run_dcp_stream(True)

    def _run_dcp_stream(self, boundary):
        path = ROOT / "python/sglang/srt/models/deepseek_v2.py"
        cls = next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "DeepseekV2MoE"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "forward_normal"
        )
        events = []

        class Stream:
            def __init__(self, name):
                self.name = name

            def wait_stream(self, other):
                events.append((self.name, "wait", other.name))

        main, shared = Stream("main"), Stream("shared")

        class Group:
            world_size = 2

            def all_gather_into_tensor(self, output, value):
                events.append("gather")
                output.copy_(value.repeat((2,) + (1,) * (value.ndim - 1)))

        class Experts:
            moe_runner_config = SimpleNamespace(inplace=False)
            quant_method = None

            def __call__(self, value, topk, **kwargs):
                events.append("routed")
                return value * 2

        def shared_forward(value, *args, **kwargs):
            events.append("shared")
            self.assertEqual(value.shape[0], 4)
            return value * 3

        def merge(experts, routed, shared, scale):
            events.append("merge")
            return routed + shared

        block = SimpleNamespace(
            shared_experts=None,
            experts=Experts(),
            _fuse_npu_tp_shared=False,
            _npu_tp_shared_overlap=True,
            alt_stream=shared,
            _fuse_shared_experts_inside_sbo=False,
            _shared_expert_tp1=False,
            _fuse_npu_tp_routed_scale=False,
            routed_scaling_factor=2.5,
            reduce_results=False,
            _maybe_quant_moe_input_once=lambda x: None,
            _forward_shared_experts=shared_forward,
            gate=lambda value, allocator: value,
            topk=lambda *args, **kwargs: None,
        )
        namespace = dict(
            torch=torch,
            envs=envs,
            _is_hip=False,
            _is_cuda=False,
            _is_musa=False,
            _is_xpu=False,
            _use_aiter=False,
            get_forward=lambda: SimpleNamespace(npu_dsa_cp_token_sharded=True),
            get_parallel=lambda: SimpleNamespace(attn_tp_group=Group()),
            get_is_capture_mode=lambda: False,
            maybe_fuse_routed_scale_and_shared_add=merge,
        )
        exec("from __future__ import annotations\n" + ast.unparse(method), namespace)
        fake_npu = SimpleNamespace(
            npu_dynamic_quant=lambda x, **kw: (x.to(torch.int8), torch.ones(x.shape[0]))
        )
        device = SimpleNamespace(
            current_stream=lambda: main, stream=lambda stream: contextlib.nullcontext()
        )
        from sglang.srt.hardware_backend.npu.moe.contracts import INT8, QuantizedOperand

        value = torch.ones(2, 64, dtype=torch.bfloat16)
        operand = (
            QuantizedOperand(value.to(torch.int8), torch.ones(2), INT8, (2, 64))
            if boundary
            else None
        )
        if boundary:

            def reject_requant(*args, **kwargs):
                raise AssertionError(
                    "Already quantized boundary must not be quantized again"
                )

            fake_npu.npu_dynamic_quant = reject_requant
        with (
            patch(
                "sglang.srt.hardware_backend.npu.moe.norm_gate.gate_forward",
                return_value=torch.ones(4, 32),
            ),
            patch.dict(sys.modules, torch_npu=fake_npu),
            patch.object(torch, "get_device_module", return_value=device),
            patch.object(torch.Tensor, "record_stream"),
            envs.SGLANG_NPU_TP_MOE_EAGER_MULTI_STREAM_MIN_TOKENS.override(1),
        ):
            result = namespace["forward_normal"](
                block, operand if boundary else value, boundary_operand=operand
            )
        torch.testing.assert_close(result, torch.full((4, 64), 8, dtype=torch.bfloat16))
        self.assertEqual(
            events,
            [
                "gather",
                "gather",
                ("shared", "wait", "main"),
                "shared",
                "routed",
                ("main", "wait", "shared"),
                "merge",
            ],
        )

    def test_megamoe_shared_stream_uses_device_api_and_waits_before_merge(self):
        path = ROOT / "python/sglang/srt/models/deepseek_v2.py"
        cls = next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "DeepseekV2MoE"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "forward_deepep"
        )
        events = []
        stream = SimpleNamespace(
            wait_stream=lambda other: events.append("fork"), record_event=lambda: "done"
        )
        main = SimpleNamespace(wait_event=lambda event: events.append("join"))
        device = SimpleNamespace(
            current_stream=lambda: main, stream=lambda value: contextlib.nullcontext()
        )

        class Experts:
            _use_ascend_fuseep = True

            def __call__(self, hidden_states, **kwargs):
                events.append("routed")
                return hidden_states * 2

        def shared(value):
            events.append("shared")
            return value * 3

        def merge(routed, shared, **kwargs):
            events.append("merge")
            return routed * kwargs["routed_scaling_factor"] + shared

        block = SimpleNamespace(
            _fuse_shared_experts_inside_sbo=False,
            is_nextn=False,
            num_fused_shared_experts=0,
            alt_stream=stream,
            experts=Experts(),
            gate=lambda x, **kwargs: x,
            topk=lambda *args, **kwargs: None,
            _forward_shared_experts=shared,
            routed_scaling_factor=2.5,
            routed_scaling_factor_fused_in_topk=False,
        )
        namespace = dict(
            torch=torch,
            _is_npu=True,
            envs=envs,
            is_in_breakable_cuda_graph=lambda: False,
            _scale_routed_and_add_shared=merge,
            SboFlags=SimpleNamespace(),
        )
        exec(
            "from __future__ import annotations" + chr(10) + ast.unparse(method),
            namespace,
        )
        with (
            patch.object(torch, "get_device_module", return_value=device),
            patch.object(torch.Tensor, "record_stream"),
            patch.object(
                torch.cuda, "current_stream", side_effect=AssertionError("CUDA API")
            ),
            patch.object(torch.cuda, "stream", side_effect=AssertionError("CUDA API")),
        ):
            result = namespace["forward_deepep"](block, torch.ones(2, 64), object())
        torch.testing.assert_close(result, torch.full((2, 64), 8.0))
        self.assertEqual(events, ["fork", "shared", "routed", "join", "merge"])

    def test_dcp_fused_norm_preserves_residual_and_quantized_rows(self):
        from sglang.srt.hardware_backend.npu.modules import dsa_token_residual as dcp
        from sglang.srt.hardware_backend.npu import layernorm_quant as nq
        from sglang.srt.hardware_backend.npu.moe.contracts import INT8, QuantizedOperand

        hidden = torch.ones(3, 64, dtype=torch.bfloat16)
        residual = hidden * 2
        total = hidden + residual
        operand = QuantizedOperand(
            torch.ones(3, 64, dtype=torch.int8), torch.ones(3), INT8, (3, 64)
        )
        writes = []
        stream = SimpleNamespace(export=lambda x: (x, residual), write=writes.append)
        norm = SimpleNamespace(weight=torch.ones(64), variance_epsilon=1e-6)
        with (
            patch.object(dcp.residual_batch, "stream_of", return_value=stream),
            patch.object(
                nq,
                "native_add_rmsnorm_quant",
                return_value=nq.NormQuantizedReadout(operand, total),
            ) as fused,
        ):
            result = dcp._read_norm(hidden, object(), norm, quant_format="npu_int8")
        self.assertIs(result, operand)
        self.assertIs(writes[0], total)
        fused.assert_called_once()
        self.assertIs(fused.call_args.args[0], hidden)
        self.assertIs(fused.call_args.args[1], residual)

    def test_router_casts_before_transpose_and_probe_is_not_activation(self):
        import sglang.srt.layers.quantization  # noqa: F401
        from sglang.srt.hardware_backend.npu.moe.norm_gate import QuantizedMoEGateMethod

        layer = torch.nn.Module()
        layer.register_parameter("weight", torch.nn.Parameter(torch.ones(32, 64)))
        method = QuantizedMoEGateMethod("W8A8_DYNAMIC")
        quant = lambda x: (x.to(torch.int8), torch.ones(x.shape[0]))
        matmul = lambda x, weight, scale, **kw: torch.ones(
            x.shape[0], weight.shape[1], dtype=torch.bfloat16
        )
        with (
            patch(
                "sglang.srt.hardware_backend.npu.moe.norm_gate.supports_npu_op",
                return_value=True,
            ),
            patch("torch.ops.npu.npu_dynamic_quant", side_effect=quant, create=True),
            patch("torch.ops.npu.npu_quant_matmul", side_effect=matmul, create=True),
            patch(
                "sglang.srt.hardware_backend.npu.utils.npu_format_cast",
                side_effect=lambda x: x.clone(),
            ) as cast,
            patch(
                "sglang.srt.hardware_backend.npu.feature_evidence.record"
            ) as evidence,
        ):
            method.process_weights_after_loading(layer)
            evidence.assert_not_called()
            self.assertEqual(cast.call_args.args[0].shape, (32, 64))
            self.assertEqual(layer.gate_quant_weight.stride(), (1, 64))
            method.apply(layer, torch.ones(2, 64, dtype=torch.bfloat16))
            evidence.assert_called_once_with(
                "gate-int8-executed", output_dtype="torch.bfloat16"
            )
