# SPDX-License-Identifier: Apache-2.0
"""R09 host tests: ledger durability, reconciliation, bounded retries, receipts,
supervisor fallback, and the new adapters — against the existing controller."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from experiment_controller import (
    EVIDENCE_BLOCKLIST,
    PROMOTION_MIN_SAMPLES,
    REGISTRY,
    aggregate_samples,
    apply_quality_receipt,
    classify_failure,
    fingerprint_case,
    latest_cases,
    outstanding_charges,
    ports_available,
    readiness,
    review_receipt,
    select,
    snapshot,
    validate,
    validate_denoise_match,
)


@pytest.fixture
def manifest():
    return json.loads(Path(__file__).with_name("campaign.example.json").read_text())


@pytest.fixture
def database(tmp_path):
    from experiment_controller import journal

    database = journal(tmp_path)
    yield database
    database.close()


# --- aggregation -----------------------------------------------------------


def test_aggregate_samples_nearest_rank_and_significance():
    samples = [10.0, 4.0, 8.0, 6.0, 2.0]  # sorted: 2,4,6,8,10
    result = aggregate_samples(samples)
    assert result["sample_count"] == 5
    assert result["median_client_s"] == 6.0
    # nearest-rank p90: ceil(0.9 * 5) = 5th smallest
    assert result["p90_client_s"] == 10.0
    assert result["significance"] == "screening"
    assert aggregate_samples([3.0])["p90_client_s"] == 3.0
    big = list(range(1, PROMOTION_MIN_SAMPLES + 1))
    assert aggregate_samples(big)["significance"] == "promotable"
    assert aggregate_samples(big)["p90_client_s"] == 9.0  # ceil(0.9*10)=9th
    with pytest.raises(ValueError):
        aggregate_samples([])


def test_denoise_evidence_requires_measured_execution_counters():
    line = "mode=native scheduler_updates=4 dit_evaluations=4 request=req phase=request"
    assert validate_denoise_match(line, "R01.CLEAN4.B0") == (4, 4)
    with pytest.raises(ValueError, match="Missing scheduler"):
        validate_denoise_match("completed request=req", "R01.CLEAN4.B0")
    with pytest.raises(ValueError, match="Invalid scheduler"):
        validate_denoise_match(
            "mode=native scheduler_updates=3 dit_evaluations=4 request=req",
            "R01.CLEAN4.B0",
        )
    assert validate_denoise_match(
        "mode=native scheduler_updates=50 dit_evaluations=50 request=req",
        "H08.NATIVE50",
    ) == (50, 50)


# --- failure taxonomy / bounded retry --------------------------------------


def test_classify_failure_blocks_oom_and_correctness():
    assert classify_failure("NPU out of memory for tensor", "failed") == "oom"
    assert classify_failure("hbm exhausted", "failed") == "oom"
    assert classify_failure("Unexpected output geometry", "failed") == "correctness"
    assert classify_failure("Missing or duplicate denoise evidence", "failed") == (
        "correctness"
    )
    assert classify_failure(None, "cleanup_unverified") == "cleanup"
    assert classify_failure("exit code 1", "failed") == "transient"


# --- ledger: fingerprint / stale evidence / no overwrite --------------------


def test_fingerprint_covers_identity_inputs(manifest):
    case = manifest["cases"][0]
    base = fingerprint_case(manifest, case)
    assert base == fingerprint_case(manifest, case)
    for mutation in (
        lambda m: m.update(source_sha="1" * 40),
        lambda m: m["cases"][0].update(adapter="R01.CLEAN4.W0R"),
        lambda m: m["contract"].update(seed=m["contract"]["seed"] + 1),
        lambda m: m["limits"].update(run_timeout_s=m["limits"]["run_timeout_s"] + 1),
        lambda m: m["cases"][0].update(samples=2),
    ):
        changed = deepcopy(manifest)
        mutation(changed)
        assert fingerprint_case(changed, changed["cases"][0]) != base


def test_stale_fingerprint_demotes_evidence(manifest):
    case = manifest["cases"][0]
    events = [
        dict(
            kind="case",
            case_id=case["id"],
            status="measured",
            fingerprint=fingerprint_case(manifest, case),
        ),
        dict(
            kind="case",
            case_id=manifest["cases"][1]["id"],
            status="measured",
            fingerprint="different",
        ),
    ]
    latest = latest_cases(events, manifest)
    assert latest[case["id"]]["status"] == "measured"
    assert latest[manifest["cases"][1]["id"]]["status"] == "stale"
    # stale evidence never satisfies a dependency
    ready = readiness(manifest, latest)
    assert ready[manifest["cases"][2]["id"]] == "blocked: dependency"


# --- budget reconciliation --------------------------------------------------


def test_outstanding_charges_reconciles_only_confirmed_cleanup():
    events = [
        dict(kind="case", case_id="a", status="running", run_id="r1", charged_device_s=1000),
        dict(kind="case", case_id="a", status="measured", run_id="r1", actual_device_s=300),
        dict(kind="case", case_id="b", status="running", run_id="r2", charged_device_s=1000),
        dict(kind="case", case_id="b", status="failed", run_id="r2", actual_device_s=200),
        dict(kind="case", case_id="c", status="running", run_id="r3", charged_device_s=1000),
        dict(kind="case", case_id="c", status="cleanup_unverified", run_id="r3"),
    ]
    # r1 reconciles to 300; r2 to 200; r3 unknown cleanup keeps worst case.
    assert outstanding_charges(events) == 300 + 200 + 1000
    # a still-running reservation stays outstanding at worst case
    assert outstanding_charges(events[:1]) == 1000
    assert outstanding_charges([]) == 0


# --- quality receipts -------------------------------------------------------


def _completed_case(database, case_id="b0"):
    from experiment_controller import record

    record(
        database,
        "case",
        case_id=case_id,
        status="measured",
        artifacts="/tmp/campaign/cases/b0",
        output_sha256=["a" * 64, "b" * 64],
    )


def test_review_receipt_binds_and_applies(database):
    _completed_case(database)
    receipt = review_receipt(
        "b0",
        "/tmp/campaign/cases/b0",
        ["a" * 64, "b" * 64],
        "prompt",
        "exact",
        "matches native reference within tolerance",
        "quality_accepted",
        at=1000.0,
    )
    assert apply_quality_receipt(database, receipt) == "quality_accepted"
    latest = latest_cases(_events(database))
    assert latest["b0"]["quality"] == "quality_accepted"

    rejected = review_receipt(
        "b0",
        "/tmp/campaign/cases/b0",
        ["a" * 64, "b" * 64],
        "prompt",
        "approximate",
        "visible seam in overlap region",
        "quality_rejected",
        at=1001.0,
    )
    assert apply_quality_receipt(database, rejected) == "quality_rejected"


def _events(database):
    from experiment_controller import history

    return history(database)


def test_stale_or_tampered_receipt_rejected(database):
    _completed_case(database)
    receipt = review_receipt(
        "b0",
        "/tmp/campaign/cases/b0",
        ["a" * 64, "b" * 64],
        "prompt",
        "exact",
        "criteria",
        "quality_accepted",
        at=1000.0,
    )
    tampered = dict(receipt, criteria="different criteria")
    with pytest.raises(ValueError, match="stale or tampered"):
        apply_quality_receipt(database, tampered)
    wrong_hashes = review_receipt(
        "b0",
        "/tmp/campaign/cases/b0",
        ["c" * 64],
        "prompt",
        "exact",
        "criteria",
        "quality_accepted",
        at=1000.0,
    )
    with pytest.raises(ValueError, match="output hashes differ"):
        apply_quality_receipt(database, wrong_hashes)
    with pytest.raises(ValueError, match="unknown case"):
        apply_quality_receipt(
            database,
            review_receipt(
                "ghost",
                "/tmp/x",
                ["a" * 64],
                "prompt",
                "exact",
                "criteria",
                "quality_accepted",
            ),
        )


def test_blocklisted_h02_receipt_rejected(database):
    _completed_case(database, case_id="H02.CROP.BEFORE")
    receipt = review_receipt(
        "H02.CROP.BEFORE",
        "/tmp/campaign/cases/H02.CROP.BEFORE",
        ["a" * 64],
        "prompt",
        "exact",
        "criteria",
        "quality_accepted",
    )
    with pytest.raises(ValueError, match="Blocklisted"):
        apply_quality_receipt(database, receipt)
    assert "H02" in EVIDENCE_BLOCKLIST


def test_decisions_cannot_cite_blocklisted_evidence(manifest):
    current = snapshot(manifest, {"b0": {"status": "measured"}}, 100, 5)
    with pytest.raises(ValueError, match="Blocklisted"):
        select(
            dict(
                schema_version=1,
                snapshot_id=current["snapshot_id"],
                action="run_case",
                case_id="b0",
                variant_id="default",
                evidence_ids=["H02"],
                reason="cite invalid history",
            ),
            current,
        )


# --- ports preflight ---------------------------------------------------------


def test_ports_available_rejects_bound_port():
    import socket

    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    bound = blocker.getsockname()[1]
    ports_available((bound + 1, bound + 2))  # free ports pass
    with pytest.raises(ValueError, match="not available"):
        ports_available((bound,))
    blocker.close()


# --- new adapters ------------------------------------------------------------


def test_r_series_adapters_registered_with_gates():
    for adapter, capability in (
        ("R05.NPUADALN.ON", "r05_npu_adaln"),
        ("R05.SP8PAD.L16", "r05_sp8_pad"),
        ("R06.GRAPH.W0R", "r06_vae_graph"),
        ("R08.SPARSE.H08", "r08_sparse"),
    ):
        assert REGISTRY[adapter][2] == capability
    for diagnostic in ("R08.DISPATCHLOG", "R08.MEMOSTATS", "R08.FFNLOG"):
        assert REGISTRY[diagnostic][1] is True  # profiled diagnostics
    assert REGISTRY["R02.Q5_FFN"][2] == "r02_q5_ffn"
    assert REGISTRY["R02.Q5_ALLOWLIST"][2] == "r02_q5_allowlist"
    # L16 requires sixteen devices
    assert REGISTRY["R05.SP8PAD.L16"][0] == "L16"


def test_l16_requires_sixteen_devices(manifest):
    invalid = deepcopy(manifest)
    invalid["cases"].append(
        dict(
            id="sp8pad",
            adapter="R05.SP8PAD.L16",
            after=["w1"],
            repeat_of=None,
            samples=1,
        )
    )
    with pytest.raises(ValueError, match="sixteen"):
        validate(invalid)


def test_a5_q6_q7_require_a5_family(manifest):
    invalid = deepcopy(manifest)
    invalid["cases"].append(
        dict(id="q5ffn", adapter="R02.Q5_FFN", after=["b0"], repeat_of=None, samples=1)
    )
    with pytest.raises(ValueError, match="A5 target"):
        validate(invalid)
    invalid["contract"]["family"] = "A5-TARGET"
    invalid["contract"]["devices"] = list(range(8))
    invalid["cases"].insert(
        1, dict(id="q0", adapter="R02.Q0", after=["b0"], repeat_of=None, samples=1)
    )
    invalid["cases"][-1]["after"] = ["q0"]
    validate(invalid)


def test_launch_environment_sets_real_flags(manifest):
    from experiment_controller import launch_environment

    case = dict(
        id="graph",
        adapter="R06.GRAPH.W0R",
        after=["b0"],
        repeat_of=None,
        samples=2,
    )
    with patch_dict_os_environ():
        env = launch_environment(manifest, case)
    assert env["H3_REPEATS"] == "2"
    assert env["SGLANG_H3_VAE_NPU_GRAPH"] == "1"
    sparse = dict(case, id="sparse", adapter="R08.SPARSE.H08")
    manifest_sparse = deepcopy(manifest)
    manifest_sparse["capabilities"]["r08_sparse"] = {
        "path": "/tmp/sparse-policy.json",
        "sha256": "0" * 64,
    }
    with patch_dict_os_environ():
        env = launch_environment(manifest_sparse, sparse)
    assert env["H3_ATTENTION_BACKEND"] == "ascend_sparse_attn"
    assert env["H3_ATTENTION_CONFIG"] == "/tmp/sparse-policy.json"


class patch_dict_os_environ:
    """Isolate os.environ mutation inside launch_environment."""

    def __enter__(self):
        import os

        self._environ = os.environ
        os.environ = {}  # plain mapping: launch_environment only calls .items()
        return self

    def __exit__(self, *args):
        import os

        os.environ = self._environ
        return False


# --- supervisor resilience ---------------------------------------------------


def test_supervisor_configures_loopback_and_rejects_otherwise():
    from experiment_controller import supervisor

    with pytest.raises(ValueError, match="loopback"):
        supervisor(
            {"endpoint": "http://example.com/v1/chat/completions", "model": "m"},
            {},
        )


# --- phase 2.2: launcher-integration fixes ----------------------------------


def test_r02_quant_backend_never_empty_and_matched(manifest):
    from experiment_controller import launch_environment

    manifest = deepcopy(manifest)
    manifest["cases"] = [
        {"id": "q0", "adapter": "R02.Q0", "after": [], "repeat_of": None, "samples": 1},
        {"id": "q2", "adapter": "R02.Q2", "after": ["q0"], "repeat_of": None, "samples": 1},
    ]
    for case in manifest["cases"]:
        env = launch_environment(manifest, case)
        # Every quant case pins the same dense backend: matched Q0/candidate
        # comparison, and never a set-but-empty override.
        assert env["H3_ATTENTION_BACKEND"] == "fa"
        assert env["H3_ATTENTION_BACKEND"] != ""


def test_r02_candidates_reference_q0_precision_map(manifest, tmp_path):
    from experiment_controller import launch_environment

    manifest = deepcopy(manifest)
    manifest["cases"] = [
        {"id": "q0", "adapter": "R02.Q0", "after": [], "repeat_of": None, "samples": 1},
        {"id": "q2", "adapter": "R02.Q2", "after": ["q0"], "repeat_of": None, "samples": 1},
        {
            "id": "q3",
            "adapter": "R02.Q3",
            "after": ["q0"],
            "repeat_of": None,
            "samples": 1,
        },
    ]
    manifest["capabilities"]["r02_q3"] = {"path": "/tmp/allowlist.json", "sha256": "0" * 64}
    (tmp_path / "q0").mkdir()
    q0_map = tmp_path / "q0" / "precision-rank-0.json"
    q0_map.write_text("{}")

    env = launch_environment(manifest, manifest["cases"][1], tmp_path)
    assert env["H3_QUANT_REFERENCE"] == str(tmp_path / "q0" / "precision-rank-0.json")
    # The allowlist comes from the verified capability receipt (Q3 only).
    assert "H3_QUANT_ALLOWLIST" not in env
    env = launch_environment(manifest, manifest["cases"][2], tmp_path)
    assert env["H3_QUANT_ALLOWLIST"] == "/tmp/allowlist.json"
    assert env["H3_QUANT_REFERENCE"] == str(tmp_path / "q0" / "precision-rank-0.json")

    # A missing reference fails the launch instead of silently quantizing
    # against an unmatched baseline.
    with pytest.raises(ValueError, match="R02.Q0 first"):
        launch_environment(manifest, manifest["cases"][1], tmp_path / "empty")


def test_r03_seam_and_window_policies_are_distinct(manifest):
    from experiment_controller import launch_environment, validate

    seam_case = dict(id="adaptive", adapter="R03.ADAPTIVE", after=[], repeat_of=None, samples=1)
    env = launch_environment(manifest, seam_case)
    # Seam-policy names go to H3_SEAM_POLICY, never to the G/L schedule.
    assert env["H3_SEAM_POLICY"] == "adaptive"
    assert "H3_WINDOW_POLICY" not in env
    assert "SGLANG_H3_WINDOW_POLICY" not in env

    window_case = dict(id="glll", adapter="R03.WINDOW.GLLL", after=[], repeat_of=None, samples=1)
    env = launch_environment(manifest, window_case)
    assert env["H3_WINDOW_POLICY"] == "GLLL"
    assert env["H3_GLOBAL_STEPS"] == "4"
    assert "H3_SEAM_POLICY" not in env

    # Unregistered schedules are rejected before any launch (the registry is
    # the source of truth for runnable cases).
    invalid = deepcopy(manifest)
    invalid["cases"].append(
        dict(id="gggg", adapter="R03.WINDOW.GGGG", after=[], repeat_of=None, samples=1)
    )
    with pytest.raises(ValueError, match="Invalid case identity or adapter"):
        validate(invalid)


def test_r04_workspace_flag_and_sync_requires_profile(manifest):
    from experiment_controller import launch_environment

    d2 = dict(id="d2", adapter="R04.D2", after=[], repeat_of=None, samples=1)
    env = launch_environment(manifest, d2)
    # The launcher consumes H3_WINDOW_WORKSPACE; the old
    # H3_REQUEST_LOCAL_WORKSPACE name is gone.
    assert env["H3_WINDOW_WORKSPACE"] == "1"
    assert "H3_REQUEST_LOCAL_WORKSPACE" not in env

    sync = dict(id="sync", adapter="R04.D1.SYNC", after=[], repeat_of=None, samples=1)
    env = launch_environment(manifest, sync)
    assert env["H3_DISPATCH_SYNC"] == "1"
    # Diagnostic synchronization requires profiling; the registry marks the
    # adapter profiled, so H3_PROFILE=1 must accompany the sync flag.
    assert env["H3_PROFILE"] == "1"
    assert "H3_DISPATCH_TRACE" not in env
