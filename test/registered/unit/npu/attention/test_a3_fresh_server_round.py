"""Regressions for host metadata and fresh-server experiment orchestration."""

import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.dcp.layout import dcp_owned_prefix_positions
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[5]


def night():
    spec = importlib.util.spec_from_file_location(
        "fresh_night", ROOT / "benchmark/prefill_1m/run_a3_nightly.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prefix_indices_stay_on_host_with_non_cpu_default_device():
    with torch.device("meta"):
        owned = dcp_owned_prefix_positions([5, 3], 2, 1, 2)
        empty = dcp_owned_prefix_positions([1], 2, 1, 2)
    assert owned.device.type == empty.device.type == "cpu"
    assert owned.tolist() == [2, 3, 7]
    assert empty.numel() == 0


def test_missing_remote_path_retries_export_instead_of_claiming_success():
    module = night()
    with tempfile.TemporaryDirectory() as tmp, patch.object(module, "run") as run:
        assert (
            module.export_live(SimpleNamespace(), Path(tmp), Path(tmp), "now") is False
        )
        run.assert_not_called()


def test_default_is_one_measurement_and_reusing_server_for_three_is_rejected():
    module = night()
    with (
        patch("sys.argv", ["night", "--round", "kernels", "--dry-run"]),
        patch("builtins.print") as output,
    ):
        assert module.main() == 0
    data = json.loads(output.call_args.args[0])
    assert data["repeats"] == 1
    assert all(c.get("repeats", 1) == 1 for c in data["cases"])
    with patch("sys.argv", ["night", "--repeats", "3"]):
        try:
            module.main()
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("Shared-server repetitions must be rejected")


def test_metrics_do_not_mix_first_run_throughput_with_mean_ttft():
    result = night().aggregate_measurements(
        [
            dict(
                status="diagnostic-unqualified",
                metrics={"TTFT avg": 10, "E2EL avg": 12, "input_token_throughput": 100},
            ),
            dict(
                status="diagnostic-unqualified",
                metrics={"TTFT avg": 20, "E2EL avg": 22, "input_token_throughput": 200},
            ),
        ],
        2,
    )
    assert result["metrics"]["E2EL avg"] == 17
    assert result["metrics"]["input_token_throughput"] == 150


def test_stop_previous_stops_outer_suite_before_worker():
    module = night()
    events = []
    real_path = Path
    with tempfile.TemporaryDirectory() as tmp:
        proc = Path(tmp)
        for pid, command in (
            ("987654", b"python3\0/tmp/run_a3_nightly.py\0"),
            ("987655", b"python3\0unrelated.py\0run_a3_nightly.py\0"),
        ):
            folder = proc / pid
            folder.mkdir()
            (folder / "cmdline").write_bytes(command)

        def path(value):
            return proc if value == "/proc" else real_path(value)

        with (
            patch.object(module, "Path", side_effect=path),
            patch.object(
                module.os,
                "kill",
                side_effect=lambda pid, sig: events.append((pid, sig)),
            ),
            patch.object(
                module,
                "stop_previous_worker",
                side_effect=lambda: events.append("worker"),
            ),
        ):
            module.stop_previous()
    assert events == [(987654, module.signal.SIGTERM), "worker"]
