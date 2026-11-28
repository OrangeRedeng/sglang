# SPDX-License-Identifier: Apache-2.0

import pytest

from quantization_cases import PROTECTED, selection


@pytest.fixture
def baseline():
    names = {
        f"blocks.{i}.{layer}"
        for i in range(10)
        for layer in ("mlp.fc1", "mlp.fc2", "attn.qkv_proj", "attn.out_proj")
    } | set(PROTECTED)
    return dict(
        method="bf16",
        rank=0,
        checkpoint_manifest_sha256="checkpoint",
        placement="placement",
        source_sha="source",
        layers={name: {"precision": "bfloat16"} for name in names},
    )


def test_middle_ffn_quantizes_whole_supported_linears(baseline):
    result = selection("Q2", "checkpoint", "placement", "source", baseline)
    assert result["selected"] == sorted(
        f"blocks.{index}.mlp.fc{part}" for index in range(1, 9) for part in (1, 2)
    )
    assert set(result["exclusions"]) & set(result["selected"]) == set()
    assert (
        set(result["exclusions"]) | set(result["selected"]) == baseline["layers"].keys()
    )
    attention = selection("Q4", "checkpoint", "placement", "source", baseline)
    assert len(attention["selected"]) == 16
    assert "blocks.1.attn.qkv_proj" in attention["selected"]


def test_allowlist_and_baseline_identity_fail_closed(baseline):
    for invalid in ("blocks.0.mlp.fc1", "blocks.1.attn.q_proj", "adaln_proj"):
        with pytest.raises(ValueError, match="whole middle-block projections"):
            selection("Q3", "checkpoint", "placement", "source", baseline, [invalid])
    with pytest.raises(ValueError, match="source, checkpoint or placement"):
        selection("Q2", "checkpoint", "other placement", "source", baseline)
    assert selection(
        "Q3", "checkpoint", "placement", "source", baseline, ["blocks.2.mlp.fc2"]
    )["selected"] == ["blocks.2.mlp.fc2"]


def test_registered_reference_and_a5_gate(tmp_path):
    import json
    from pathlib import Path

    from experiment_controller import launch_environment, validate

    root = Path(__file__).parent
    manifest = json.loads((root / "campaign.r02-a3.example.json").read_text())
    validate(manifest)
    candidate = next(case for case in manifest["cases"] if case["id"] == "q2")
    env = launch_environment(manifest, candidate, {"q0": {"artifacts": str(tmp_path)}})
    assert env["H3_QUANT_REFERENCE"] == str(tmp_path / "precision-rank-0.json")
    candidate["after"] = []
    with pytest.raises(ValueError, match="Q0 dependency"):
        validate(manifest)

    manifest = json.loads((root / "campaign.r02-a5.example.json").read_text())
    validate(manifest)
    for case in manifest["cases"]:
        assert (
            launch_environment(manifest, case, {"q0": {"artifacts": str(tmp_path)}})[
                "H3_ATTENTION_BACKEND"
            ]
            == "fa"
        )
    manifest["contract"]["soc_version"] = 250
    with pytest.raises(ValueError, match="Ascend 950"):
        validate(manifest)


def test_conservative_mxfp8_requires_complete_bf16_reference(baseline):
    args = ("checkpoint", "placement", "source", baseline)
    assert selection("Q5_FFN", *args)["selected"] == selection("Q2", *args)["selected"]
    assert (
        selection("Q5_ALLOWLIST", *args, ["blocks.2.attn.qkv_proj"])["method"]
        == "mxfp8"
    )
    del baseline["layers"]["blocks.2.attn.qkv_proj"]
    with pytest.raises(ValueError, match="Incomplete"):
        selection("Q5_FFN", *args)
