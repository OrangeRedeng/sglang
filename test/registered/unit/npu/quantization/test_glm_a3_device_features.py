"""Device parity for the INT contracts used by the GLM feature matrix."""

import unittest
from types import SimpleNamespace

import torch
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=90, suite="stage-a-unit-test-npu")
HAVE_NPU = is_npu() and torch.npu.is_available()


@unittest.skipUnless(HAVE_NPU, "Ascend NPU required")
class TestIntegerFeatureParity(CustomTestCase):
    def setUp(self):
        torch.manual_seed(4)

    def test_streaming_full_size_projection_and_gate(self):
        from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
            NPUOnlineW8A8Int8LinearMethod,
        )
        from sglang.srt.model_loader.weight_utils import default_weight_loader

        for outputs in (256, 2048):
            with self.subTest(outputs=outputs), torch.device("npu"):
                layer = torch.nn.Module()
                method = NPUOnlineW8A8Int8LinearMethod()
                method.create_weights(
                    layer,
                    6144,
                    [outputs],
                    6144,
                    outputs,
                    torch.bfloat16,
                    weight_loader=default_weight_loader,
                )
                self.assertEqual(layer.weight.device.type, "meta")
                source = (
                    torch.randn((outputs, 6144), dtype=torch.bfloat16, device="cpu")
                    * 0.01
                )
                layer.weight.weight_loader(layer.weight, source)
                method.process_weights_after_loading(layer)
                self.assertEqual(layer.weight.dtype, torch.int8)
                self.assertFalse(
                    any(
                        p.is_floating_point() and p.numel() == source.numel()
                        for p in layer.parameters()
                    )
                )
                x = torch.randn((7, 6144), dtype=torch.bfloat16, device="npu")
                actual = method.apply(layer, x)
                q, scale = torch.ops.npu.npu_dynamic_quant(x)
                # Reference uses the exact stored quantized values, not the pre-quantization weight.
                import torch_npu
                from sglang.srt.hardware_backend.npu.utils import npu_format_cast

                stored = torch_npu.npu_format_cast(layer.weight, 2).cpu().float()
                reference = (
                    (q.cpu().float() @ stored)
                    * scale.cpu().float()[:, None]
                    * layer.weight_scale.cpu().float().reshape(1, -1)
                )
                torch.testing.assert_close(
                    actual.float().cpu(), reference, rtol=0.01, atol=0.01
                )
        from sglang.srt.hardware_backend.npu.moe.norm_gate import StreamingGateMethod

        with torch.device("npu"):
            gate = torch.nn.Module()
            gate.weight = torch.nn.Parameter(
                torch.empty((256, 6144), dtype=torch.bfloat16)
            )
            method = StreamingGateMethod("W8A8_DYNAMIC")
            method.install(gate)
            gate.weight.weight_loader(
                gate.weight,
                torch.randn((256, 6144), dtype=torch.bfloat16, device="cpu") * 0.01,
            )
            method.process_weights_after_loading(gate)
            x = torch.randn((7, 6144), dtype=torch.bfloat16)
            actual = method.apply(gate, x)
            q, scale = torch.ops.npu.npu_dynamic_quant(x)
            reference = (
                (q.cpu().float() @ gate.gate_quant_weight.cpu().float())
                * scale.cpu().float()[:, None]
                * gate.gate_quant_scale.cpu().float()[None, :]
            )
            torch.testing.assert_close(
                actual.float().cpu(), reference, rtol=0.01, atol=0.01
            )

    def test_routing_reuses_identical_int8_payload(self):
        from sglang.srt.hardware_backend.npu.moe.init_routing import (
            NPUMoEInitRouting_v2,
        )
        from sglang.srt.hardware_backend.npu.moe.tp_fusion import supports_npu_op

        if not supports_npu_op("npu_moe_init_routing_v2", ("scale",)):
            self.skipTest("installed init-routing has no scale passthrough")
        op = NPUMoEInitRouting_v2(quant_mode=1)
        for rows in (0, 1, 7, 33):
            with self.subTest(rows=rows):
                x = torch.randn((rows, 128), dtype=torch.bfloat16, device="npu")
                ids = torch.tensor(
                    [[0, 1] if i % 3 else [2, 0] for i in range(rows)],
                    dtype=torch.int32,
                    device="npu",
                ).reshape(rows, 2)
                if not rows:
                    continue
                baseline = op._init_routing(x, ids, 4, 2)
                q, scale = torch.ops.npu.npu_dynamic_quant(x)
                actual = op._init_routing(q, ids, 4, 2, input_scale=scale)
                for a, b in zip(actual, baseline):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_routed_scale_and_shared_finalize(self):
        from sglang.srt.hardware_backend.npu.moe.finalize_routing import (
            NPUFinalizeRouting,
        )
        from sglang.srt.hardware_backend.npu.moe.init_routing import (
            NPUMoEInitRouting_v2,
        )

        for rows in (1, 7, 33):
            with self.subTest(rows=rows):
                x = torch.randn((rows, 128), dtype=torch.bfloat16, device="npu")
                ids = torch.tensor(
                    [[i % 3, (i + 1) % 3] for i in range(rows)],
                    dtype=torch.int32,
                    device="npu",
                )
                routed, mapping, counts, _ = NPUMoEInitRouting_v2(-1)._init_routing(
                    x, ids, 4, 2
                )
                weights = torch.rand((rows, 2), device="npu", dtype=torch.float32)
                weights /= weights.sum(-1, keepdim=True)
                shared = torch.randn_like(x)
                op = NPUFinalizeRouting(2)
                plain = op._finalize_routing(routed, weights, mapping, ids)
                scaled = op._finalize_routing(routed, weights * 2.5, mapping, ids)
                combined = op._finalize_routing(
                    routed, weights * 2.5, mapping, ids, skip1=shared
                )
                torch.testing.assert_close(scaled, plain * 2.5, rtol=0.01, atol=0.01)
                torch.testing.assert_close(
                    combined, scaled + shared, rtol=0.01, atol=0.01
                )

    def test_fused_gmm2_matches_grouped_matmul_and_finalize(self):
        from sglang.srt.hardware_backend.npu.moe.finalize_routing import (
            NPUFinalizeRouting,
        )
        from sglang.srt.hardware_backend.npu.moe.init_routing import (
            NPUMoEInitRouting_v2,
        )
        from sglang.srt.hardware_backend.npu.moe.tp_fusion import (
            fused_gmm2_finalize,
            supports_npu_op,
        )

        if not supports_npu_op(
            "npu_grouped_matmul_finalize_routing", ("row_index", "output_bs")
        ) or not supports_npu_op("npu_moe_init_routing_v2", ("row_idx_type",)):
            self.skipTest("fused GMM2 ABI unavailable")
        rows = 7
        hidden = 6144
        intermediate = 2048
        x = torch.randn((rows, intermediate), device="npu", dtype=torch.bfloat16)
        ids = torch.tensor(
            [[0, 1], [0, 2], [0, 1], [2, 1], [0, 1], [0, 1], [0, 2]],
            device="npu",
            dtype=torch.int32,
        )
        init = NPUMoEInitRouting_v2(1)
        q, mapping, counts, scale = init._init_routing(x, ids, 4, 2)
        weight = (
            torch.randn((4, hidden, intermediate), device="npu", dtype=torch.bfloat16)
            * 0.01
        )
        w, ws = torch.ops.npu.npu_dynamic_quant(weight)
        import torch_npu

        w = torch_npu.npu_format_cast(w.transpose(1, 2).contiguous(), 29)
        grouped = torch.ops.npu.npu_grouped_matmul(
            x=[q],
            weight=[w],
            scale=[ws],
            per_token_scale=[scale],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=counts,
            output_dtype=torch.bfloat16,
        )[0]
        weights = torch.rand((rows, 2), device="npu", dtype=torch.float32)
        weights /= weights.sum(-1, keepdim=True)
        reference = NPUFinalizeRouting(2)._finalize_routing(
            grouped, weights, mapping, ids
        )
        init.row_idx_type = 1
        q, mapping, counts, scale = init._init_routing(x, ids, 4, 2)
        actual = fused_gmm2_finalize(q, scale, w, ws, counts, weights, mapping)
        torch.testing.assert_close(actual, reference, rtol=0.01, atol=0.01)


if __name__ == "__main__":
    unittest.main()
