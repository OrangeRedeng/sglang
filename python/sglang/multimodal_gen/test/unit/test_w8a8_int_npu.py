# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

from sglang.multimodal_gen.runtime.layers.linear import (
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.w8a8_int_npu import (
    NPUOnlineW8A8DiffusionConfig,
    NPUOnlineW8A8DiffusionLinearMethod,
)


class TestOnlineW8A8IgnoredLayers(unittest.TestCase):
    def setUp(self):
        platform = patch(
            "sglang.multimodal_gen.runtime.layers.quantization.w8a8_int_npu."
            "current_platform.is_npu",
            return_value=True,
        )
        platform.start()
        self.addCleanup(platform.stop)

    def test_ignored_fused_layer_names(self):
        for name in ("qkv_proj", "gate_up_proj"):
            prefix = f"blocks.0.attn.{name}"
            for ignored in ([prefix], [name], ["blocks.0.attn"]):
                with self.subTest(name=name, ignored=ignored):
                    config = NPUOnlineW8A8DiffusionConfig(ignored_layers=ignored)
                    layer = ReplicatedLinear(4, 8, quant_config=config, prefix=prefix)
                    self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)

    def test_ignored_shards_preserve_consistency_checks(self):
        config = NPUOnlineW8A8DiffusionConfig(
            ignored_layers=["q_proj", "k_proj", "v_proj"]
        )
        layer = ReplicatedLinear(
            4, 8, quant_config=config, prefix="blocks.0.attn.qkv_proj"
        )
        self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)

        config = NPUOnlineW8A8DiffusionConfig(ignored_layers=["q_proj"])
        with self.assertRaisesRegex(ValueError, "some but not all shards"):
            ReplicatedLinear(4, 8, quant_config=config, prefix="blocks.0.attn.qkv_proj")

        layer = ReplicatedLinear(
            4, 8, quant_config=config, prefix="blocks.0.attn.out_proj"
        )
        self.assertIsInstance(layer.quant_method, NPUOnlineW8A8DiffusionLinearMethod)


if __name__ == "__main__":
    unittest.main()


def test_exact_block_boundary():
    import torch

    from sglang.multimodal_gen.runtime.layers.quantization.w8a8_int_npu import (
        current_platform,
    )

    with patch.object(current_platform, "is_npu", return_value=True):
        config = NPUOnlineW8A8DiffusionConfig(ignored_layers=["blocks.0"])
        protected = ReplicatedLinear(
            4, 8, quant_config=config, prefix="blocks.0.attn.out_proj"
        )
        quantized = ReplicatedLinear(
            4, 8, quant_config=config, prefix="blocks.01.attn.out_proj"
        )
    assert isinstance(protected.quant_method, UnquantizedLinearMethod)
    assert isinstance(quantized.quant_method, NPUOnlineW8A8DiffusionLinearMethod)
    assert protected.weight.dtype == torch.get_default_dtype()


def test_loader_converts_each_local_weight_once_and_releases_sources(monkeypatch):
    import weakref

    import torch

    from sglang.multimodal_gen.runtime.loader.fsdp_load import (
        load_model_from_full_model_state_dict,
    )
    from sglang.multimodal_gen.runtime.layers.quantization.w8a8_int_npu import (
        current_platform,
    )

    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.utils import quantization_utils

    monkeypatch.setattr(current_platform, "is_npu", lambda: True)
    format_policy = SimpleNamespace(allow_internal_format=False)
    monkeypatch.setattr(quantization_utils, "is_npu", lambda: True)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(config=format_policy, empty_cache=lambda: None),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.npu, "npu_dynamic_quant", lambda *_: None, raising=False
    )
    monkeypatch.setattr(
        torch.ops.npu, "npu_quant_matmul", lambda *_: None, raising=False
    )
    config = NPUOnlineW8A8DiffusionConfig(ignored_layers=["protected"])
    model = torch.nn.Module()
    source = {}
    weak_sources = {}
    converted = []
    for name in ("a", "b", "protected"):
        module = ReplicatedLinear(
            4,
            8,
            bias=True,
            params_dtype=torch.bfloat16,
            quant_config=config,
            prefix=name,
        )
        model.add_module(name, module)
        source[f"{name}.weight"] = torch.ones(8, 4, dtype=torch.bfloat16)
        source[f"{name}.bias"] = torch.zeros(8, dtype=torch.bfloat16)
        weak_sources[name] = weakref.ref(source[f"{name}.weight"])
        if name == "protected":
            continue

        def convert(layer, name=name):
            if name == "b":
                assert weak_sources["a"]() is None
            converted.append(name)
            layer.weight = torch.nn.Parameter(
                layer.weight.to(torch.int8).T.contiguous(), requires_grad=False
            )
            layer.weight_scale = torch.nn.Parameter(torch.ones(8), requires_grad=False)

        monkeypatch.setattr(
            module.quant_method.kernel, "process_weights_after_loading", convert
        )

    load_model_from_full_model_state_dict(
        model,
        iter(()),
        torch.device("cpu"),
        torch.bfloat16,
        strict=True,
        preconverted_state_dict=(source, {}),
    )
    quantization_utils.process_model_weights_after_loading(model)
    assert not format_policy.allow_internal_format
    for module in (model.a, model.b):
        assert module.weight.shape == (4, 8) and module.weight.dtype == torch.int8
        assert module.bias.dtype == torch.float32
    assert converted == ["a", "b"]
    assert all(weak_sources[name]() is None for name in converted)
    assert model.protected.weight.dtype == torch.bfloat16
    assert model.protected.bias.dtype == torch.bfloat16


def test_ascend_online_mxfp8_keeps_protected_projections(monkeypatch):
    import torch

    from sglang.multimodal_gen.runtime.layers.quantization.mxfp8 import (
        MXFP8Config,
        NPUOnlineMXFP8DiffusionLinearMethod,
        current_platform,
    )

    monkeypatch.setattr(current_platform, "is_npu", lambda: True)
    config = MXFP8Config(ignored_layers=["blocks.0"])
    protected = ReplicatedLinear(
        64,
        32,
        quant_config=config,
        params_dtype=torch.bfloat16,
        prefix="blocks.0.attn.qkv_proj",
    )
    selected = ReplicatedLinear(
        64,
        32,
        quant_config=config,
        params_dtype=torch.bfloat16,
        prefix="blocks.1.attn.qkv_proj",
    )
    assert isinstance(protected.quant_method, UnquantizedLinearMethod)
    assert protected.weight.dtype == torch.bfloat16
    assert isinstance(selected.quant_method, NPUOnlineMXFP8DiffusionLinearMethod)
