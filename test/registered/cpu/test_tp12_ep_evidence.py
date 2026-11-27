"""Host regressions for the 2026-11-19 TP/EP evidence and staging work.

Covers (CPU only, no NPU required):
- besttp12 activation counting, snapshot, JSON dump, eligibility inventory
- MATRIX case 85 persistent gather workspace: lease lifecycle and rejection
- E92 FuseEP routing-metadata staging: no-op passthrough, staged copy, counters
- E93 cached DeepEP buffer geometry validation: match passes, mismatch raises
- E94 shared-fusion ABI probe: unavailable stack reports ineligible, never raises

These are host contract tests; they do not establish device activation, speed,
or numerical parity (FULL_PIPELINE_VALIDATION gates G2+).
"""

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.hardware_backend.npu import besttp12, besttp12_graph
from sglang.srt.hardware_backend.npu.moe import fuseep


def _setflag(monkeypatch, name, value):
    monkeypatch.setenv("SGLANG_NPU_TP12_" + name, "1" if value else "0")


class TestEvidenceCounting:
    def test_counts_aggregate_and_first_logged(self, monkeypatch, caplog):
        besttp12._COUNTS.clear()
        besttp12._FIRST.clear()
        with caplog.at_level("INFO", logger="sglang.srt.hardware_backend.npu.besttp12"):
            for _ in range(5):
                besttp12.evidence("unit-path", rows=4)
        assert besttp12._COUNTS["unit-path"] == 5
        assert besttp12._FIRST["unit-path"] == json.dumps({"rows": 4}, sort_keys=True)
        assert any("TP12 ACTIVE unit-path" in r.message for r in caplog.records)
        # Repeated identical details log once; no spam per call.
        assert len([r for r in caplog.records if "TP12 ACTIVE" in r.message]) == 1

    def test_snapshot_reports_flags(self, monkeypatch):
        _setflag(monkeypatch, "INPUT_REUSE", True)
        _setflag(monkeypatch, "SHARED_STREAM", False)
        snapshot = besttp12.evidence_snapshot()
        assert snapshot["flags"]["INPUT_REUSE"] is True
        assert snapshot["flags"]["SHARED_STREAM"] is False

    def test_dump_writes_rank_file_and_recovers_from_log(self, monkeypatch, tmp_path, caplog):
        besttp12._COUNTS.clear()
        besttp12._FIRST.clear()
        besttp12.evidence("dump-path")
        monkeypatch.setenv("SGLANG_TP12_EVIDENCE_DIR", str(tmp_path))
        with caplog.at_level("INFO", logger="sglang.srt.hardware_backend.npu.besttp12"):
            besttp12.dump_evidence()
        files = list(tmp_path.glob("tp12-evidence-*.json"))
        assert len(files) == 1
        payload = json.loads(files[0].read_text())
        assert payload["counts"]["dump-path"] == 1
        assert any("TP12 EVIDENCE_FINAL" in r.message for r in caplog.records)

    def test_inventory_classifies_shared_experts(self):
        class DeepseekV2MLP:  # name must match the inventory's layer filter
            pass

        gate_up = SimpleNamespace(
            quant_method=besttp12.OnlineInt8Method(),
        )
        down = SimpleNamespace(quant_method=object())
        mlp = DeepseekV2MLP()
        mlp.gate_up_proj = gate_up
        mlp.down_proj = down
        mlp.swiglu_limit = None
        model = SimpleNamespace(named_modules=lambda: [("m.shared_experts", mlp)])
        besttp12._ELIGIBILITY.clear()
        inventory = besttp12.shared_eligibility_inventory(model)
        entry = inventory["m.shared_experts"]
        # OnlineInt8Method is INT8-accepted via accepts_int8's isinstance chain.
        assert entry["gate_up_int8"] is True
        # down_proj's plain object method is not INT8-capable.
        assert entry["down_int8"] is False
        assert entry["input_reuse_eligible"] is True
        assert entry["fusion_eligible"] is False


class TestGatherWorkspace:
    def test_disabled_returns_none(self, monkeypatch):
        _setflag(monkeypatch, "GATHER_WORKSPACE", False)
        q = torch.zeros(4, 8, dtype=torch.int8)
        scale = torch.ones(4, dtype=torch.float32)
        assert besttp12.gather_workspace(q, scale) is None

    def test_lease_release_cycle_and_concurrent_rejection(self, monkeypatch):
        _setflag(monkeypatch, "GATHER_WORKSPACE", True)
        besttp12._WORKSPACES.clear()
        besttp12._COUNTS.clear()
        monkeypatch.setattr(
            besttp12, "_workspace_world", lambda: 2, raising=False
        )
        q = torch.zeros(4, 8, dtype=torch.int8)
        scale = torch.ones(4, dtype=torch.float32)
        buffers = besttp12.gather_workspace(q, scale)
        assert buffers is not None
        q_full, scale_full = buffers
        assert q_full.shape == (8, 8)
        assert scale_full.shape == (8,)
        # A second lease of the same key without release is rejected.
        with pytest.raises(RuntimeError, match="already leased"):
            besttp12.gather_workspace(q, scale)
        # Scope exit releases the lease held by the operand.
        operand = SimpleNamespace(data=q_full)
        besttp12.release_operand(operand)
        assert besttp12.gather_workspace(q, scale)[0] is q_full
        besttp12.release_operand(SimpleNamespace(data=q_full))
        # Double release is a contract violation.
        with pytest.raises(RuntimeError, match="already released"):
            besttp12.release_operand(SimpleNamespace(data=q_full))
        assert besttp12._COUNTS["gather-workspace-lease"] == 2
        assert besttp12._COUNTS["gather-workspace-release"] == 2

    def test_publish_without_reuse_releases_workspace(self, monkeypatch):
        _setflag(monkeypatch, "GATHER_WORKSPACE", True)
        _setflag(monkeypatch, "INPUT_REUSE", False)
        besttp12._WORKSPACES.clear()
        besttp12._COUNTS.clear()
        monkeypatch.setattr(besttp12, "_workspace_world", lambda: 2, raising=False)
        q = torch.zeros(4, 8, dtype=torch.int8)
        scale = torch.ones(4, dtype=torch.float32)
        q_full, _ = besttp12.gather_workspace(q, scale)
        besttp12_graph.publish_input(q_full, scale, torch.bfloat16)
        # INPUT_REUSE off: publish must not hold the lease.
        assert not besttp12._WORKSPACES[
            next(iter(besttp12._WORKSPACES))
        ]["leased"]
        # A new gather of the same shape is admissible again.
        assert besttp12.gather_workspace(q, scale) is not None

    def test_publish_with_reuse_holds_until_scope_exit(self, monkeypatch):
        _setflag(monkeypatch, "GATHER_WORKSPACE", True)
        _setflag(monkeypatch, "INPUT_REUSE", True)
        besttp12._WORKSPACES.clear()
        monkeypatch.setattr(besttp12, "_workspace_world", lambda: 2, raising=False)
        q = torch.zeros(4, 8, dtype=torch.int8)
        scale = torch.ones(4, dtype=torch.float32)
        q_full, scale_full = besttp12.gather_workspace(q, scale)
        besttp12_graph.publish_input(q_full, scale_full, torch.bfloat16)
        operand = besttp12_graph.take_input()
        assert operand is not None
        key = next(iter(besttp12._WORKSPACES))
        assert besttp12._WORKSPACES[key]["leased"]
        # Abandoned pending input (no scope) must also release the lease.
        besttp12_graph.publish_input(q_full, scale_full, torch.bfloat16)
        besttp12_graph.clear_input()
        assert not besttp12._WORKSPACES[key]["leased"]


class TestRoutingMetaStaging:
    def test_same_dtype_is_identity_noop(self, monkeypatch):
        monkeypatch.setenv("SGLANG_NPU_EP_META_REUSE", "true")
        ids = torch.tensor([[1, 2]], dtype=torch.int32)
        staged = fuseep._stage_routing_meta(ids, torch.int32)
        assert staged is ids
        assert fuseep.fuseep_meta_evidence()["noop"] >= 1

    def test_dtype_cast_reuses_persistent_buffer(self, monkeypatch):
        monkeypatch.setenv("SGLANG_NPU_EP_META_REUSE", "true")
        fuseep._META_STAGING.clear()
        weights = torch.tensor([[0.5, 0.5]], dtype=torch.bfloat16)
        first = fuseep._stage_routing_meta(weights, torch.float32)
        assert first.dtype == torch.float32
        assert torch.allclose(first, weights.float())
        second = fuseep._stage_routing_meta(weights + 0, torch.float32)
        assert second is first  # same persistent storage, rewritten
        evidence = fuseep.fuseep_meta_evidence()
        assert evidence["staged"] >= 2
        assert evidence["allocations_avoided"] >= 1
        assert evidence["enabled"] is True

    def test_shape_change_rekeys(self, monkeypatch):
        monkeypatch.setenv("SGLANG_NPU_EP_META_REUSE", "true")
        fuseep._META_STAGING.clear()
        small = torch.zeros((2, 2), dtype=torch.bfloat16)
        big = torch.zeros((4, 2), dtype=torch.bfloat16)
        a = fuseep._stage_routing_meta(small, torch.float32)
        b = fuseep._stage_routing_meta(big, torch.float32)
        assert a is not b
        assert b.shape == (4, 2)


class _FakeState:
    def __init__(self, hidden_size, num_experts, capacity):
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_max_dispatch_tokens_per_rank = capacity


class TestBufferGeometryValidation:
    def test_matching_geometry_passes(self, monkeypatch):
        monkeypatch.setenv("SGLANG_NPU_EP_BUFFER_VALIDATE", "true")
        state = _FakeState(5120, 160, 2048)
        fuseep._validate_fuseep_buffer(state, None, 5120, 160, 2048)

    def test_mismatch_raises_with_details(self, monkeypatch):
        monkeypatch.setenv("SGLANG_NPU_EP_BUFFER_VALIDATE", "true")
        state = _FakeState(4096, 128, 1024)
        with pytest.raises(RuntimeError, match="hidden_size cached=4096"):
            fuseep._validate_fuseep_buffer(state, None, 5120, 160, 2048)
        with pytest.raises(RuntimeError, match="num_experts cached=128"):
            fuseep._validate_fuseep_buffer(
                _FakeState(5120, 128, 1024), None, 5120, 160, 2048
            )
        with pytest.raises(
            RuntimeError, match="num_max_dispatch_tokens_per_rank cached=1024"
        ):
            fuseep._validate_fuseep_buffer(
                _FakeState(5120, 160, 1024), None, 5120, 160, 2048
            )

    def test_guard_can_be_disabled_for_diagnosis(self, monkeypatch):
        monkeypatch.setenv("SGLANG_NPU_EP_BUFFER_VALIDATE", "false")
        state = _FakeState(4096, 128, 1024)
        fuseep._validate_fuseep_buffer(state, None, 5120, 160, 2048)


class TestSharedFusionAbiProbe:
    def test_bf16_signature_does_not_claim_wire_precision(self, monkeypatch):
        class Buffer:
            def fused_deep_moe(self, x, topk_idx, topk_weights):
                pass

        def mega_moe(x):
            pass

        monkeypatch.setitem(sys.modules, "deep_ep", SimpleNamespace(Buffer=Buffer))
        monkeypatch.setitem(
            sys.modules,
            "cann_ops_transformer",
            SimpleNamespace(ops=SimpleNamespace(mega_moe=mega_moe)),
        )
        monkeypatch.setitem(
            sys.modules, "cann_ops_transformer.ops", SimpleNamespace(mega_moe=mega_moe)
        )
        result = fuseep.probe_shared_fusion_abi()
        assert result["external_prequantized_input_params"] == []
        assert result["internal_dispatch_quantization"] == "not established by signature"
        assert "dispatch_int8_eligible" not in result
        assert all("payload stays BF16" not in reason for reason in result["reasons"])

    def test_unavailable_stack_reports_ineligible(self, caplog):
        caplog.clear()
        result = fuseep.probe_shared_fusion_abi()
        assert result["eligible"] is False
        assert any("probe failed" in r for r in result["reasons"])
        assert any(
            "A3 MEGAMOE SHARED_FUSION_ABI" in r.message for r in caplog.records
        )

    def test_probe_never_raises(self):
        # Whatever is (not) installed, the probe returns a JSON-serializable
        # verdict instead of crashing startup.
        result = fuseep.probe_shared_fusion_abi()
        json.dumps(result, sort_keys=True)
