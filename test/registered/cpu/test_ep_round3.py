"""Host regressions for round-3 EP arms: sliced dispatch and shared fusion.

CPU-only; a mocked DeepEP buffer stands in for the native fused call. These
tests cover slicing boundaries, output ordering, payload encoding (scale-bit
uint64 + routed-scaling pre-division), incompatibility reasons, and the
injection adapter's evidence/contract. Device parity (G3) and speed remain
A3 gates.
"""

import json
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.hardware_backend.npu import besttp12
from sglang.srt.hardware_backend.npu.moe import fuseep


@pytest.fixture(autouse=True)
def _mock_exec(monkeypatch):
    """forward_fuseep reads get_exec().moe.fuseep_mode; publish a stub."""
    monkeypatch.setattr(
        fuseep,
        "get_exec",
        lambda: NS(moe=NS(fuseep_mode=3)),
    )
    monkeypatch.setattr(fuseep, "_native_shared_supported", lambda: True)


class _FakeBuffer:
    """Records fused_deep_moe calls; returns identifiable row slices."""

    def __init__(self):
        self.calls = []

    def fused_deep_moe(self, x=None, topk_idx=None, topk_weights=None, **kwargs):
        self.calls.append(dict(rows=x.shape[0], start=None, kwargs=kwargs, topk_idx=topk_idx, topk_weights=topk_weights))
        out = x.clone() * 2.0
        return out, None


def _layer(rows, hidden=8):
    topk = NS(
        topk_ids=torch.zeros(rows, 1, dtype=torch.int64),
        topk_weights=torch.ones(rows, 1, dtype=torch.float32),
    )
    layer = NS(
        w13_weight=torch.zeros(1),
        w13_weight_scale=torch.zeros(1),
        w2_weight=torch.zeros(1),
        w2_weight_scale=torch.zeros(1),
        w13_scale_bias=None,
        w2_scale_bias=None,
        num_experts=16,
        moe_ep_size=16,
        moe_tp_size=1,
        _mega_moe_weights_processed=True,
    )
    return layer, topk


def _rows_of(buffer):
    return [c["rows"] for c in buffer.calls]


def test_shared_w4_requant_pack_scale_and_compensation():
    weight = torch.tensor(
        [[0, 7, -7, 127, -128, 2, -2, 0], [0, -7, 7, -127, 127, -2, 2, 0]],
        dtype=torch.int8,
    )
    scales = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    packed, scale4, bias = fuseep._requantize_shared_w4(weight, scales)
    raw = packed.view(torch.uint8).to(torch.int16)
    low, high = raw.bitwise_and(15), raw.bitwise_right_shift(4)
    unpacked = torch.stack((low, high), dim=-1).reshape_as(weight)
    unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked)
    expected_scale = weight.float().abs().amax(dim=0).clamp_min(1) * scales / 7
    torch.testing.assert_close(scale4, expected_scale)
    torch.testing.assert_close(bias, 8 * unpacked.float().sum(dim=0) * scale4)
    assert unpacked.min() >= -7 and unpacked.max() <= 7
    assert torch.all((unpacked.float() * scale4 - weight.float() * scales).abs() <= scale4 / 2 + 1e-5)
    with pytest.raises(ValueError):
        fuseep._requantize_shared_w4(weight[:, :7], scales[:7])
    with pytest.raises(ValueError):
        fuseep._requantize_shared_w4(weight, torch.full((8,), float("nan")))


def test_shared_w4_routing_remaps_routed_slots_and_appends_local_expert():
    ids = torch.tensor([[0, 3, 4, 7], [1, -1, 5, 6]], dtype=torch.int32)
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.0, 0.3, 0.5]])
    mapped, mapped_weights = fuseep._route_shared_w4(ids, weights, 4, 2, 1, 0.5)
    assert mapped.tolist() == [[0, 3, 5, 8, 9], [1, -1, 6, 7, 9]]
    torch.testing.assert_close(mapped_weights[:, :-1], weights)
    torch.testing.assert_close(mapped_weights[:, -1], torch.full((2,), 0.5))
    with pytest.raises(ValueError):
        fuseep._route_shared_w4(ids, weights, 4, 2, 2, 0.5)
    with pytest.raises(ValueError):
        fuseep._route_shared_w4(torch.zeros((1, 16)), torch.zeros((1, 16)), 4, 2, 0, 1.0)
    _, padded_weights = fuseep._route_shared_w4(
        torch.tensor([[-1, -1]]), torch.zeros((1, 2)), 4, 2, 0, 0.5
    )
    assert padded_weights[0, -1] == 0


def test_prepare_shared_w4_appends_one_routed_slot_atomically(monkeypatch):
    monkeypatch.setattr(fuseep, "_check_shared_geometry", lambda *args: None)
    monkeypatch.setattr(fuseep, "npu_format_cast", lambda weight: weight)
    def projection(inputs, channels):
        return NS(weight=torch.randint(-7, 8, (inputs, channels), dtype=torch.int8), weight_scale=torch.ones(channels))
    experts = NS(hidden_size=8, intermediate_size_per_partition=8, num_local_experts=2, num_experts=4,
                 moe_ep_size=2, moe_tp_size=1, _mega_moe_weights_processed=True)
    for prefix in ("w13", "w2"):
        for suffix in ("weight", "weight_scale", "scale_bias"):
            setattr(experts, f"{prefix}_{suffix}", ["first", "second"])
    module = NS(experts=experts, shared_experts=NS(gate_up_proj=projection(8, 16), down_proj=projection(8, 8)),
                num_fused_shared_experts=0, routed_scaling_factor_fused_in_topk=False, routed_scaling_factor=2.0)
    fuseep.prepare_fuseep_shared_w4(module)
    assert experts._fuseep_shared_w4 == {"local_experts": 2, "scale": 0.5}
    assert experts.w13_weight[-1].shape == (8, 2)
    assert experts.w2_weight[-1].shape == (8, 1)
    assert experts.w13_weight_scale[-1].shape == (16,)
    assert experts.w2_scale_bias[-1].shape == (8,)
    assert all(len(getattr(experts, f"{prefix}_{suffix}")) == 3 for prefix in ("w13", "w2") for suffix in ("weight", "weight_scale", "scale_bias"))


def test_prepare_shared_w4_does_not_mutate_experts_on_failure(monkeypatch):
    monkeypatch.setattr(fuseep, "_check_shared_geometry", lambda *args: None)
    monkeypatch.setattr(fuseep, "npu_format_cast", lambda weight: weight)
    experts = NS(hidden_size=8, intermediate_size_per_partition=8, num_local_experts=1, num_experts=2,
                 moe_ep_size=2, moe_tp_size=1, _mega_moe_weights_processed=True)
    for prefix in ("w13", "w2"):
        for suffix in ("weight", "weight_scale", "scale_bias"):
            setattr(experts, f"{prefix}_{suffix}", ["original"])
    gate = NS(weight=torch.ones((8, 16), dtype=torch.int8), weight_scale=torch.ones(16))
    down = NS(weight=torch.ones((8, 8), dtype=torch.int8), weight_scale=torch.full((8,), float("nan")))
    module = NS(experts=experts, shared_experts=NS(gate_up_proj=gate, down_proj=down),
                num_fused_shared_experts=0, routed_scaling_factor_fused_in_topk=True)
    with pytest.raises(ValueError):
        fuseep.prepare_fuseep_shared_w4(module)
    assert experts.w13_weight == ["original"]
    assert experts.w2_weight == ["original"]
    assert not hasattr(experts, "_fuseep_shared_w4")


def test_shared_w4_forward_uses_expanded_experts(monkeypatch):
    monkeypatch.setenv("SGLANG_NPU_EP_SHARED_W4_ROUTED", "true")
    monkeypatch.setattr(fuseep, "_mega_moe_capacity", lambda: 2048)
    monkeypatch.setattr(fuseep, "_log_mega_moe_geometry", lambda *args: None)
    buffer = _FakeBuffer()
    monkeypatch.setattr(fuseep, "_get_fuseep_buffer", lambda layer: buffer)
    monkeypatch.setattr(fuseep, "_prepare_mega_moe_buffer", lambda value: value)
    layer, topk = _layer(2)
    layer.num_experts = 4
    layer.moe_ep_size = 2
    layer.moe_ep_rank = 1
    layer._fuseep_shared_w4 = {"local_experts": 2, "scale": 0.5}
    topk.topk_ids = torch.tensor([[0, 3], [1, 2]])
    topk.topk_weights = torch.ones((2, 2))
    assert fuseep.fuseep_shared_fusion_active_for(layer)
    fuseep.forward_fuseep(layer, torch.zeros((2, 8), dtype=torch.bfloat16), topk)
    call = buffer.calls[0]["kwargs"]
    assert call["num_experts"] == 6
    assert buffer.calls[0]["topk_idx"].tolist() == [[0, 4, 5], [1, 3, 5]]
    assert buffer.calls[0]["topk_weights"][:, -1].tolist() == [0.5, 0.5]


def test_shared_fusion_selects_only_own_layer_payload(monkeypatch):
    """r8 per-layer binding: forward_fuseep binds ITS layer's payload (or
    clears the slot); a layer without a payload never inherits another
    layer's shared weights — the cross-layer bug r7 rejected."""
    monkeypatch.setenv("SGLANG_NPU_EP_SHARED_FUSION", "true")
    layer_a, topk = _layer(1)
    layer_b, _ = _layer(1)
    payload = {"l1": {"weights": 1}, "l2": {"weights": 2}}
    layer_a._fuseep_shared_payload = payload
    # layer_b deliberately has no payload.

    assert fuseep.select_layer_payload(layer_a) is True
    assert fuseep._ACTIVE_PAYLOAD["l1"]["weights"] == 1
    assert fuseep.fuseep_shared_fusion_active_for(layer_a) is True

    assert fuseep.select_layer_payload(layer_b) is False
    assert fuseep._ACTIVE_PAYLOAD["l1"] == {}  # cleared, not inherited
    assert fuseep.fuseep_shared_fusion_active_for(layer_b) is False


def test_shared_fusion_disabled_flag_inert(monkeypatch):
    monkeypatch.setenv("SGLANG_NPU_EP_SHARED_FUSION", "false")
    layer, _ = _layer(1)
    layer._fuseep_shared_payload = {"l1": {}, "l2": {}}
    assert fuseep.select_layer_payload(layer) is False
    assert fuseep._ACTIVE_PAYLOAD["l1"] == {}
    assert fuseep.fuseep_shared_fusion_active_for(layer) is False


def test_unverified_native_shared_fusion_keeps_separate_experts(monkeypatch):
    monkeypatch.setenv("SGLANG_NPU_EP_SHARED_FUSION", "true")
    monkeypatch.setattr(fuseep, "_native_shared_supported", lambda: False)
    layer, _ = _layer(1)
    layer._fuseep_shared_payload = {"l1": {}, "l2": {}}
    assert fuseep.fuseep_shared_fusion_active_for(layer) is False
    assert fuseep.select_layer_payload(layer) is False
    assert "does not pass shared inputs" in fuseep.prepare_fuseep_shared_payload(NS())


def test_sliced_dispatch_splits_and_concatenates(monkeypatch):
    monkeypatch.setenv("SGLANG_NPU_EP_SLICED_DISPATCH", "true")
    monkeypatch.setattr(fuseep, "_mega_moe_capacity", lambda: 2048)
    monkeypatch.setattr(fuseep, "_log_mega_moe_geometry", lambda *a: None)
    monkeypatch.setattr(fuseep, "_get_fuseep_buffer", lambda layer: _FakeBuffer())
    monkeypatch.setattr(fuseep, "_prepare_mega_moe_buffer", lambda b: b)
    buffer = _FakeBuffer()
    monkeypatch.setattr(fuseep, "_get_fuseep_buffer", lambda layer: buffer)

    rows = 4099  # odd tail: 2048 + 2048 + 3
    layer, topk = _layer(rows)
    x = torch.arange(rows * 8, dtype=torch.bfloat16).reshape(rows, 8)
    out = fuseep.forward_fuseep(layer, x, topk)
    assert _rows_of(buffer) == [2048, 2048, 3]
    assert out.shape == (rows, 8)
    # Row order preserved: out == x*2 elementwise in original order.
    assert torch.equal(out.float(), (x.float() * 2.0))


def test_unsliced_within_capacity_single_call(monkeypatch):
    monkeypatch.setenv("SGLANG_NPU_EP_SLICED_DISPATCH", "true")
    monkeypatch.setattr(fuseep, "_mega_moe_capacity", lambda: 2048)
    monkeypatch.setattr(fuseep, "_log_mega_moe_geometry", lambda *a: None)
    buffer = _FakeBuffer()
    monkeypatch.setattr(fuseep, "_get_fuseep_buffer", lambda layer: buffer)
    monkeypatch.setattr(fuseep, "_prepare_mega_moe_buffer", lambda b: b)
    layer, topk = _layer(2048)
    fuseep.forward_fuseep(layer, torch.zeros(2048, 8, dtype=torch.bfloat16), topk)
    assert _rows_of(buffer) == [2048]


def test_overflow_without_flag_raises(monkeypatch):
    monkeypatch.setenv("SGLANG_NPU_EP_SLICED_DISPATCH", "false")
    monkeypatch.setattr(fuseep, "_mega_moe_capacity", lambda: 2048)
    monkeypatch.setattr(fuseep, "_log_mega_moe_geometry", lambda *a: None)
    monkeypatch.setattr(fuseep, "_get_fuseep_buffer", lambda layer: _FakeBuffer())
    monkeypatch.setattr(fuseep, "_prepare_mega_moe_buffer", lambda b: b)
    layer, topk = _layer(2049)
    with pytest.raises(ValueError, match="SLICED_DISPATCH"):
        fuseep.forward_fuseep(layer, torch.zeros(2049, 8, dtype=torch.bfloat16), topk)


class TestSharedPayload:
    def _proj(self, channels, inputs, scale_value=0.01):
        return NS(
            weight=torch.full((channels, inputs), 3, dtype=torch.int8),
            weight_scale=torch.full((channels, 1), scale_value, dtype=torch.float32),
        )

    def _moe(self, scaling=2.5, fused_in_topk=False):
        return NS(
            shared_experts=NS(
                gate_up_proj=self._proj(16, 8),
                down_proj=self._proj(8, 8),
            ),
            experts=NS(
                intermediate_size_per_partition=8,
                hidden_size=8,
                _fuseep_shared_payload=None,
            ),
            num_fused_shared_experts=0,
            routed_scaling_factor=scaling,
            routed_scaling_factor_fused_in_topk=fused_in_topk,
            _fuseep_shared_attempted=False,
        )

    def test_payload_encodes_scale_bits_and_output_scale(self, monkeypatch):
        monkeypatch.setenv("SGLANG_NPU_EP_SHARED_FUSION", "true")
        monkeypatch.setattr(fuseep, "_check_shared_geometry", lambda h, i: None)
        besttp12._COUNTS.clear()
        moe = self._moe(scaling=2.5)
        reason = fuseep.prepare_fuseep_shared_payload(moe)
        assert reason == ""
        payload = moe.experts._fuseep_shared_payload
        # Scale-bit encoding: float bits reinterpreted as uint64.
        l1_sf = payload["l1"]["weights_sf"]
        assert l1_sf.dtype == torch.uint64
        assert l1_sf.numel() == 16
        # Each uint64 holds the raw float32 bits (low half) of a scale.
        got = l1_sf.view(torch.int32)[0::2].view(torch.float32)
        # output_scale = 1/2.5 pre-divides the scales.
        torch.testing.assert_close(
            got, torch.full((16,), 0.01 / 2.5, dtype=torch.float32), rtol=1e-6, atol=0.0
        )
        assert besttp12._COUNTS["ep-shared-fusion-payload"] == 1

    def test_loaded_layout_inputs_by_channels_accepted(self, monkeypatch):
        monkeypatch.setenv("SGLANG_NPU_EP_SHARED_FUSION", "true")
        monkeypatch.setattr(fuseep, "_check_shared_geometry", lambda h, i: None)
        moe = self._moe()
        # Loaded NPU W8A8 linears are [inputs, channels]: gate_up (8, 16).
        moe.shared_experts.gate_up_proj.weight = torch.full(
            (8, 16), 3, dtype=torch.int8
        )
        assert fuseep.prepare_fuseep_shared_payload(moe) == ""
        assert tuple(moe.experts._fuseep_shared_payload["l1"]["weights"].shape) == (8, 16)

    def test_half_dtype_scales_cast_and_bad_scales_named(self, monkeypatch):
        """r11 (evidence 20261121-221805-151653): every device prep rejected
        with the bare 'finite positive' message -- fp16 scales are legal and
        must cast; genuine failures must name the failing term."""
        monkeypatch.setenv("SGLANG_NPU_EP_SHARED_FUSION", "true")
        monkeypatch.setattr(fuseep, "_check_shared_geometry", lambda h, i: None)
        moe = self._moe()
        moe.shared_experts.gate_up_proj.weight = torch.full(
            (8, 16), 3, dtype=torch.int8
        )
        moe.shared_experts.gate_up_proj.weight_scale = torch.full(
            (16,), 0.5, dtype=torch.float16
        )
        moe.shared_experts.down_proj.weight_scale = torch.full(
            (8,), 0.25, dtype=torch.float16
        )
        assert fuseep.prepare_fuseep_shared_payload(moe) == ""
        bad = torch.full((16,), 0.5, dtype=torch.float32)
        bad[3] = float("nan")
        moe2 = self._moe()  # fresh module: the first prep sets _attempted
        moe2.shared_experts.gate_up_proj.weight = torch.full(
            (8, 16), 3, dtype=torch.int8
        )
        moe2.shared_experts.gate_up_proj.weight_scale = bad
        reason = fuseep.prepare_fuseep_shared_payload(moe2)
        assert "scales unusable" in reason and "finite" in reason

    def test_no_epilogue_scale_means_untouched_scales(self, monkeypatch):
        monkeypatch.setenv("SGLANG_NPU_EP_SHARED_FUSION", "true")
        monkeypatch.setattr(fuseep, "_check_shared_geometry", lambda h, i: None)
        moe = self._moe(scaling=2.5, fused_in_topk=True)
        assert fuseep.prepare_fuseep_shared_payload(moe) == ""
        got = moe.experts._fuseep_shared_payload["l1"]["weights_sf"].view(
            torch.int32
        )[0::2].view(torch.float32)
        torch.testing.assert_close(
            got, torch.full((16,), 0.01, dtype=torch.float32), rtol=1e-6, atol=0.0
        )

    def test_incompatible_geometry_recorded_not_raised(self, monkeypatch):
        monkeypatch.setenv("SGLANG_NPU_EP_SHARED_FUSION", "true")
        moe = self._moe()
        # Tiny test geometry fails the REAL MegaMoE limits (hidden/intermediate
        # must be >=1024 and 512-aligned); recorded as reason, never raised.
        reason = fuseep.prepare_fuseep_shared_payload(moe)
        assert "shared geometry outside MegaMoE limits" in reason
        assert fuseep.shared_payload_reason() == reason

    def test_offset_projection_rejected(self, monkeypatch):
        monkeypatch.setenv("SGLANG_NPU_EP_SHARED_FUSION", "true")
        monkeypatch.setattr(fuseep, "_check_shared_geometry", lambda h, i: None)
        moe = self._moe()
        moe.shared_experts.gate_up_proj.weight_offset = torch.ones(4)
        reason = fuseep.prepare_fuseep_shared_payload(moe)
        assert "weight_offset" in reason

    def test_disabled_env_reports_reason(self, monkeypatch):
        monkeypatch.setenv("SGLANG_NPU_EP_SHARED_FUSION", "false")
        reason = fuseep.prepare_fuseep_shared_payload(self._moe())
        assert "disabled" in reason

    def test_attempted_flag_prevents_double_prep(self, monkeypatch):
        monkeypatch.setenv("SGLANG_NPU_EP_SHARED_FUSION", "true")
        moe = self._moe()
        moe._fuseep_shared_attempted = True
        assert fuseep.prepare_fuseep_shared_payload(moe) == fuseep.shared_payload_reason()


def test_injection_adapter_requires_native_params(monkeypatch):
    monkeypatch.setenv("SGLANG_NPU_EP_SHARED_FUSION", "true")

    def native_without_shared(x, fuse_mode, activation):  # no shared kwargs
        return x

    with pytest.raises(RuntimeError, match="none of the shared payload"):
        fuseep._adapt_shared_fusion_operator(native_without_shared)

    def native_with_shared(x, fuse_mode, activation, shared_l1_weights=None, **kw):
        return x

    wrapped = fuseep._adapt_shared_fusion_operator(native_with_shared)
    # Without a prepared payload the wrapper passes nothing through.
    assert wrapped(1, fuse_mode=3, activation="swiglu") == 1


def test_set_active_payload_feeds_injection(monkeypatch):
    monkeypatch.setenv("SGLANG_NPU_EP_SHARED_FUSION", "true")
    besttp12._COUNTS.clear()
    payload = {
        "l1": {"weights": "w1", "weights_sf": "s1", "bias": None},
        "l2": {"weights": "w2", "weights_sf": "s2", "bias": None},
    }
    fuseep._set_active_payload(payload)
    monkeypatch.setattr(fuseep, "_SHARED_PREPARED", True)

    captured = {}

    def native(
        fuse_mode,
        activation,
        shared_l1_weights=None,
        shared_l1_weights_sf=None,
        shared_l1_bias=None,
        shared_l2_weights=None,
        shared_l2_weights_sf=None,
        shared_l2_bias=None,
    ):
        captured.update(
            l1w=shared_l1_weights,
            l1sf=shared_l1_weights_sf,
            l1b=shared_l1_bias,
            l2w=shared_l2_weights,
            l2sf=shared_l2_weights_sf,
            l2b=shared_l2_bias,
        )
        return "out"

    wrapped = fuseep._adapt_shared_fusion_operator(native)
    result = wrapped(fuse_mode=3, activation="swiglu")
    assert result == "out"
    assert captured["l1w"] == ["w1"]
    assert captured["l2sf"] == ["s2"]
    assert captured["l1b"] is None
    assert besttp12._COUNTS["ep-shared-fusion-injected"] == 1


def test_d1v3_kv_rows_pad_and_trim_to_loc_width():
    from sglang.srt.hardware_backend.npu.memory_pool_npu import align_d1v3_kv_rows

    k26 = torch.arange(26 * 4).reshape(26, 4)
    v26 = k26 + 100
    k32, v32 = align_d1v3_kv_rows(k26, v26, 32)
    assert k32.shape == (32, 4) and v32.shape == (32, 4)
    assert torch.equal(k32[:26], k26)
    assert torch.count_nonzero(k32[26:]) == 0

    k32, v32 = align_d1v3_kv_rows(k32, v32, 26)
    assert k32.shape == (26, 4) and v32.shape == (26, 4)
    assert torch.equal(k32, k26)


def test_native_alias_shared_weight1_is_injected_as_list(monkeypatch):
    """r13 A3 runtime contract (r12 log): shared_weight1 is
    Optional[List[Tensor]], not a bare Tensor. The Python probe names
    (shared_l1_*) and runtime names (shared_weight1/2) must coexist."""
    monkeypatch.setenv("SGLANG_NPU_EP_SHARED_FUSION", "true")
    payload = {
        "l1": {"weights": "w1", "weights_sf": "s1", "bias": None},
        "l2": {"weights": "w2", "weights_sf": "s2", "bias": None},
    }
    fuseep._set_active_payload(payload)
    monkeypatch.setattr(fuseep, "_SHARED_PREPARED", True)
    captured = {}

    def native(
        fuse_mode,
        activation,
        shared_weight1=None,
        shared_weight1_scale=None,
        shared_weight1_bias=None,
        shared_weight2=None,
        shared_weight2_scale=None,
        shared_weight2_bias=None,
    ):
        captured.update(
            w1=shared_weight1,
            s1=shared_weight1_scale,
            b1=shared_weight1_bias,
            w2=shared_weight2,
            s2=shared_weight2_scale,
            b2=shared_weight2_bias,
        )
        return "out"

    wrapped = fuseep._adapt_shared_fusion_operator(native)
    assert wrapped(fuse_mode=3, activation="swiglu") == "out"
    assert captured["w1"] == ["w1"]
    assert captured["s1"] == ["s1"]
    assert captured["b1"] is None
    assert captured["w2"] == ["w2"]
    assert captured["s2"] == ["s2"]
    assert captured["b2"] is None
