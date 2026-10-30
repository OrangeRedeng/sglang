"""Q-only projection preserves the assistant's packed checkpoint layout."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from sglang.srt.environ import envs
from sglang.srt.layers.linear import QKVParallelLinear
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.gemma4_causal import Gemma4Attention
from sglang.srt.models.gemma4_mtp import Gemma4AssistantForCausalLM
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_q_rows_loaded_by_existing_weight_loader_match_full_projection(dtype, bias):
    with get_parallel().override(tp_size=1, tp_rank=0, tp_group=None):
        linear = QKVParallelLinear(16, 4, 2, 1, bias=bias, params_dtype=dtype)
    with torch.no_grad():
        linear.weight.zero_()
        if bias:
            linear.bias.zero_()
        for shard, rows in (("q", 8), ("k", 4), ("v", 4)):
            linear.weight.weight_loader(
                linear.weight, torch.randn(rows, 16, dtype=dtype), shard
            )
            if bias:
                linear.bias.weight_loader(
                    linear.bias, torch.randn(rows, dtype=dtype), shard
                )
    attention = SimpleNamespace(qkv_proj=linear, q_size=8, kv_size=4)
    keys_before = tuple(linear.state_dict())
    for bs in (1, 3, 12):
        hidden = torch.randn(bs, 16, dtype=dtype)
        reference = Gemma4Attention._project_qkv(attention, hidden)
        attention._npu_frozen_q_only = True
        with patch.object(
            linear,
            "forward",
            side_effect=AssertionError("Unused K/V projection executed"),
        ):
            result, k, v = Gemma4Attention._project_qkv(attention, hidden)
        torch.testing.assert_close(result, reference[0])
        assert k is None and v is None
        attention._npu_frozen_q_only = False
    assert tuple(linear.state_dict()) == keys_before


@pytest.mark.parametrize(
    "enabled,npu,tp,quant,method,skip_bias,expected",
    [
        (True, True, 1, None, UnquantizedLinearMethod(), False, True),
        (False, True, 1, None, UnquantizedLinearMethod(), False, False),
        (True, False, 1, None, UnquantizedLinearMethod(), False, False),
        (True, True, 2, None, UnquantizedLinearMethod(), False, False),
        (True, True, 1, object(), UnquantizedLinearMethod(), False, False),
        (True, True, 1, None, object(), False, False),
        (True, True, 1, None, UnquantizedLinearMethod(), True, False),
    ],
)
def test_only_unquantized_tp1_npu_assistant_enables_q_only(
    enabled, npu, tp, quant, method, skip_bias, expected
):
    attn = SimpleNamespace(
        attn=SimpleNamespace(),
        qkv_proj=SimpleNamespace(quant_method=method, skip_bias_add=skip_bias),
    )
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)]),
        quant_config=quant,
    )
    context = SimpleNamespace(get_physical_layer_id=Mock(return_value=7))
    with (
        envs.SGLANG_ENABLE_NPU_GEMMA4_MTP_Q_ONLY.override(enabled),
        patch("sglang.srt.models.gemma4_mtp.is_npu", return_value=npu),
        patch(
            "sglang.srt.models.gemma4_mtp.get_parallel",
            return_value=SimpleNamespace(tp_size=tp),
        ),
    ):
        Gemma4AssistantForCausalLM.bind_frozen_kv_context(model, context)
    assert attn._npu_frozen_q_only is expected
    assert attn.is_kv_shared_layer and attn.kv_shared_layer_index == 7
    assert attn.attn.layer_id == 7
