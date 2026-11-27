# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from pathlib import Path

import pytest
from experiment_controller import (
    history,
    journal,
    latest_cases,
    readiness,
    record,
    select,
    snapshot,
    validate,
)


@pytest.fixture
def manifest():
    return json.loads(Path(__file__).with_name("campaign.example.json").read_text())


def decision(current, **overrides):
    result = dict(
        schema_version=1,
        snapshot_id=current["snapshot_id"],
        action="run_case",
        case_id="b0",
        variant_id="default",
        evidence_ids=[],
        reason="Establish the baseline first.",
    )
    return result | overrides


def test_dependency_and_quality_receipt_gates(manifest):
    validate(manifest)
    assert readiness(manifest, {}) == {
        "b0": "ready",
        "w0r": "blocked: dependency",
        "w1": "blocked: dependency",
    }
    latest = {"b0": {"status": "measured"}, "w0r": {"status": "measured"}}
    assert readiness(manifest, latest)["w1"] == "blocked: h05_repack_parity"


@pytest.mark.parametrize(
    "override",
    [
        {"action": "run_case", "case_id": "w0r"},
        {"variant_id": "../../run"},
        {"snapshot_id": "stale"},
        {"evidence_ids": ["invented"]},
        {"command": ["bash", "-c", "exit"]},
    ],
)
def test_disallowed_model_decisions(manifest, override):
    current = snapshot(manifest, {}, 28800, 2)
    assert select(decision(current), current) == ("run_case", "b0")
    with pytest.raises(ValueError):
        select(decision(current, **override), current)


def test_no_unapproved_repeat_or_cross_family_adapter(manifest):
    invalid = deepcopy(manifest)
    invalid["cases"].append(
        dict(id="repeat", adapter="H05.B0", after=["b0"], repeat_of=None)
    )
    with pytest.raises(ValueError, match="explicit repeat"):
        validate(invalid)
    invalid["cases"].pop()
    invalid["cases"][0]["adapter"] = "H08.NATIVE50"
    with pytest.raises(ValueError, match="same deployment family"):
        validate(invalid)


def test_failed_run_survives_restart_without_retry(manifest, tmp_path):
    database = journal(tmp_path)
    record(database, "case", case_id="b0", status="running", charged_device_s=1000)
    record(database, "case", case_id="b0", status="failed")
    database.close()
    database = journal(tmp_path)
    events = history(database)
    database.close()
    assert sum(row.get("charged_device_s", 0) for row in events) == 1000
    assert readiness(manifest, latest_cases(events))["b0"] == "failed"
    assert readiness(manifest, latest_cases(events))["w0r"] == "blocked: dependency"
