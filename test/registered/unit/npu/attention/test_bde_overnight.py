"""Unattended host/container orchestration without devices or a Docker daemon."""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestBdeOvernight(CustomTestCase):
    @staticmethod
    def runner():
        path = (
            Path(__file__).resolve().parents[5]
            / "benchmark/prefill_1m/run_bde_overnight.py"
        )
        spec = importlib.util.spec_from_file_location("night_runner", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_native_failure_attribution_and_run_scoped_dumps(self):
        runner = self.runner()
        self.assertEqual(
            runner.server_failure("retCode=0x25 [aicore timeout]"),
            "native AICore timeout",
        )
        self.assertIsNone(runner.server_failure("Missing successful AISBench summary"))
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            source = repo / "extra-info/data-dump/0"
            source.mkdir(parents=True)
            (source / "old").write_text("old run")
            os.utime(source / "old", (1, 1))
            (source / "mc2_exception_info_MegaMoe").write_text("native evidence")
            target = repo / "artifacts"
            runner.collect_native_dumps(repo, target, 2)
            self.assertFalse((target / "native-dumps/0/old").exists())
            self.assertEqual(
                (target / "native-dumps/0/mc2_exception_info_MegaMoe").read_text(),
                "native evidence",
            )

    def test_client_polling_reports_total_timeout_budget(self):
        module = self.runner()
        client, worker = MagicMock(), MagicMock()
        worker.poll.return_value = None
        client.wait.side_effect = subprocess.TimeoutExpired("aisbench", 2)
        with patch.object(module.time, "monotonic", side_effect=[0, 0, 2, 4, 6]):
            with self.assertRaisesRegex(TimeoutError, "6s total budget"):
                module.wait_benchmark(client, worker, 6)
        self.assertEqual(client.wait.call_count, 3)

    def test_repair_round_has_legal_piecewise_and_fixed_batch_budget(self):
        module = self.runner()
        root = Path(__file__).resolve().parents[5]
        harness = module.load_module(
            root / "benchmark/prefill_1m/test_glm53_features.py", "repair_launch"
        )
        config = dict(
            variant="dsa-metadata-packed-batch2",
            reference_recipe="historical-bde-capacity",
            checkpoint="/tmp/weights",
            chunk=32768,
            context=1048576,
            memfrac=0.85,
            maxreq=1,
            kv_rows=67584,
        )
        command, env = module.variant_launch(harness, root, config)
        self.assertEqual(command[command.index("--max-prefill-tokens") + 1], "32768")
        self.assertEqual(command[command.index("--max-total-tokens") + 1], "63488")
        self.assertEqual(env["HCCL_BUFFSIZE"], "256")
        config["variant"] = "graph-piecewise-probe"
        command, env = module.variant_launch(harness, root, config)
        self.assertEqual(command[command.index("--dcp-size") + 1], "1")
        self.assertEqual(
            command[command.index("--cuda-graph-backend-prefill") + 1], "tc_piecewise"
        )
        self.assertEqual(
            command[command.index("--cuda-graph-backend-decode") + 1], "disabled"
        )
        self.assertEqual(env["SGLANG_NPU_DSA_CP_TOKEN_SHARDED"], "False")

    def test_short_piecewise_probe_ignores_unselected_long_workloads(self):
        module = self.runner()
        output = io.StringIO()
        with (
            patch.object(
                sys,
                "argv",
                [
                    "night",
                    "--dry-run",
                    "--performance-only",
                    "--reference-recipe",
                    "historical-bde-capacity",
                    "--variants",
                    "graph-piecewise-probe",
                    "--length",
                    "8192",
                    "--context",
                    "16384",
                ],
            ),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(module.main(), 0)
        self.assertEqual(
            json.loads(output.getvalue())["variants"], ["graph-piecewise-probe"]
        )

    def test_nightly_averages_repetitions_and_marks_partial_results(self):
        module = self.runner()
        root = Path(__file__).resolve().parents[5]
        night = module.load_module(
            root / "benchmark/prefill_1m/run_a3_nightly.py", "repair_summary"
        )
        rows = [
            dict(
                status="diagnostic-unqualified",
                metrics={
                    "TTFT avg": value,
                    "TTFT P90": value * 2,
                    "E2E_time": 4,
                    "qps": 1,
                    "prefill_token_throughput": 20,
                },
            )
            for value in (10, 12, 14)
        ]
        result = night.aggregate_measurements(rows, 3)
        self.assertEqual(result["metrics"]["TTFT avg"], 12)
        self.assertEqual(result["metrics"]["TTFT P90"], 24)
        self.assertAlmostEqual(result["ttft_cv_pct"], 13.60827635, places=5)
        self.assertEqual(
            night.aggregate_measurements(rows[:2], 3)["status"],
            "incomplete-repetitions",
        )

    def test_global_budget_and_single_warmup_reach_container_command(self):
        module = self.runner()
        output = io.StringIO()
        with (
            patch.object(
                sys,
                "argv",
                [
                    "night",
                    "--dry-run",
                    "--features-only",
                    "--feature-stages",
                    "long",
                    "--max-hours",
                    "9.5",
                    "--long-cache-modes",
                    "prefix99",
                    "--long-warmup",
                    "once",
                    "--long-dcp-size",
                    "1",
                    "--diagnostic-megamoe",
                    "--megamoe-dispatch-tokens",
                    "1024",
                ],
            ),
            patch.object(module.time, "time", return_value=1000),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(module.main(), 0)
        command = json.loads(output.getvalue())["feature_commands"]["long"]
        self.assertEqual(float(command[command.index("--deadline-unix") + 1]), 35200)
        self.assertEqual(command[command.index("--long-warmup") + 1], "once")
        self.assertEqual(command[command.index("--long-cache-modes") + 1], "prefix99")
        self.assertEqual(command[command.index("--dcp") + 1], "1")
        self.assertIn("--diagnostic-megamoe", command)
        self.assertEqual(
            command[command.index("--megamoe-dispatch-tokens") + 1], "1024"
        )

    def test_overnight_long_plan_uses_explicit_capacity_and_matched_workloads(self):
        module = self.runner()
        output = io.StringIO()
        with (
            patch.object(
                sys,
                "argv",
                [
                    "night",
                    "--dry-run",
                    "--features-only",
                    "--feature-stages",
                    "long",
                    "--feature-kv-rows",
                    "8192",
                    "--chunk",
                    "16384",
                ],
            ),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(module.main(), 0)
        plan = json.loads(output.getvalue())
        command = plan["feature_commands"]["long"]
        for flag, value in (
            ("--context-length", "1048576"),
            ("--kv-rows", "131072"),
            ("--long-lengths", "32768,1000000"),
            ("--long-concurrency", "1,2"),
            ("--benchmark-repeats", "3"),
            ("--chunk", "16384"),
        ):
            self.assertEqual(command[command.index(flag) + 1], value)
        variants = command[command.index("--benchmark-variants") + 1].split(",")
        self.assertIn("dcp-baseline", variants)
        self.assertIn("dcp-io-multistream", variants)

    def test_aisbench_initializes_conda_before_environment_activation(self):
        module = self.runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary, ais = root / "bin", root / "ais"
            binary.mkdir()
            ais.mkdir()
            conda = binary / "conda"
            conda.write_text(
                "#!/bin/bash\n"
                "if [[ $1 == shell.bash && $2 == hook ]]; then\n"
                "  echo 'conda() { export A3_FIXTURE_ACTIVE=1; };'\n"
                "else\n  echo 'CondaError: initialize before activation' >&2; exit 1\nfi\n"
            )
            conda.chmod(0o755)
            environment = root / "env.sh"
            environment.write_text("conda activate fixture\n")
            (ais / "aisbench_test.py").write_text(
                "import os, sys, json\n"
                "assert os.environ['A3_FIXTURE_ACTIVE'] == '1'\n"
                "assert 'HTTP_PROXY' not in os.environ\n"
                "print(json.dumps(sys.argv[1:]))\n"
            )
            for executable_on_path in (True, False):
                with self.subTest(executable_on_path=executable_on_path):
                    env = dict(os.environ, HTTP_PROXY="http://invalid:9")
                    env.pop("CONDA_EXE", None)
                    env.pop("BASH_FUNC_conda%%", None)
                    if executable_on_path:
                        env["PATH"] = str(binary) + os.pathsep + env["PATH"]
                    else:
                        env["CONDA_EXE"] = str(conda)
                    command = [
                        "bash",
                        "--noprofile",
                        "--norc",
                        "-c",
                        module.AISBENCH_SHELL,
                        "aisbench",
                        str(environment),
                        str(ais),
                        "--input_len",
                        "1024",
                    ]
                    result = subprocess.run(
                        command, env=env, text=True, capture_output=True, timeout=10
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(json.loads(result.stdout), ["--input_len", "1024"])
            environment.write_text("conda activate fixture\nfalse\n")
            result = subprocess.run(
                command, env=env, text=True, capture_output=True, timeout=10
            )
            self.assertNotEqual(result.returncode, 0)

    def test_baseline_only_stage_selection_preserves_qualification(self):
        module = self.runner()
        for stages, expected in (
            ("serve", ["serve"]),
            ("host,npu,serve", ["host", "npu", "serve"]),
        ):
            output = io.StringIO()
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "night",
                        "--dry-run",
                        "--feature-stages",
                        stages,
                        "--feature-cases",
                        "serve-baseline",
                        "--variants",
                        "baseline",
                    ],
                ),
                contextlib.redirect_stdout(output),
            ):
                self.assertEqual(module.main(), 0)
            plan = json.loads(output.getvalue())
            self.assertEqual(list(plan["feature_commands"]), expected)
            self.assertEqual(plan["variants"], ["baseline"])
            command = plan["feature_commands"]["serve"]
            self.assertEqual(command[command.index("--cases") + 1], "serve-baseline")
        for stages in ("host,npu", "serve,serve", "serve,invalid", ""):
            with (
                self.subTest(stages=stages),
                patch.object(
                    sys, "argv", ["night", "--dry-run", "--feature-stages", stages]
                ),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                module.main()
            self.assertEqual(error.exception.code, 2)

    def test_http_flush_accepts_text_but_server_info_requires_json(self):
        module = self.runner()
        opener = MagicMock()
        response = opener.open.return_value.__enter__.return_value
        with patch.object(module.urllib.request, "build_opener", return_value=opener):
            response.read.return_value = b"Cache flushed.\n"
            self.assertEqual(
                module.http("/flush_cache", post=True, expect_json=False),
                "Cache flushed.\n",
            )
            with self.assertRaises(json.JSONDecodeError):
                module.http("/server_info")
            response.read.return_value = b'{"max_total_num_tokens": 123}'
            self.assertEqual(module.http("/server_info")["max_total_num_tokens"], 123)
            opener.open.side_effect = module.urllib.error.HTTPError(
                "http://127.0.0.1:30000/flush_cache", 400, "Busy", {}, None
            )
            with self.assertRaises(module.urllib.error.HTTPError):
                module.http("/flush_cache", post=True, expect_json=False)

    def test_host_path_detection_and_explicit_missing_path(self):
        module = self.runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            envfile = root / "set_env.sh"
            missing = root / "missing.sh"
            envfile.touch()
            with patch.object(
                module, "AIS_ENV_CANDIDATES", (str(missing), str(envfile))
            ):
                self.assertEqual(module.resolve_ais_env(None), str(envfile))
                self.assertEqual(module.resolve_ais_env(str(missing)), str(missing))
            with self.assertRaises(ValueError) as caught:
                module.check_ais_paths(root, missing)
            self.assertIn(str(root / "aisbench_test.py"), str(caught.exception))
            self.assertIn(str(missing), str(caught.exception))
            (root / "aisbench_test.py").touch()
            module.check_ais_paths(root, envfile)
            with patch.object(
                sys,
                "argv",
                [
                    "night",
                    "--check-paths",
                    "--ais-dir",
                    str(root),
                    "--ais-env",
                    str(envfile),
                ],
            ):
                self.assertEqual(module.main(), 0)

    def test_failed_server_does_not_stop_remaining_cases_or_lose_results(self):
        module = self.runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary, ais = root / "bin", root / "ais"
            binary.mkdir()
            ais.mkdir()
            envfile = root / "env.sh"
            envfile.write_text("# AISBench environment fixture\n")
            docker = binary / "docker"
            docker.write_text(
                f"#!{sys.executable}\n"
                + textwrap.dedent("""\
                import json, os, sys
                from pathlib import Path
                arguments = sys.argv[1:]
                if arguments[0] == "inspect":
                    print(json.dumps([{"State":{"Running":True}, "HostConfig":{"NetworkMode":"host"}}]))
                elif arguments[0] == "cp":
                    Path(arguments[-1]).mkdir()
                    (Path(arguments[-1]) / "copied.txt").touch()
                elif "-i" in arguments:
                    config = json.loads(sys.stdin.readline())
                    with open(os.environ["NIGHT_TEST_EVENTS"], "a") as log:
                        log.write(json.dumps(config) + "\\n")
                    if config.get("variant") == "dcp-packed-read":
                        print(json.dumps({"status":"error", "error":"fixture startup failure"}), flush=True)
                        sys.exit(1)
                    if config["action"] == "features":
                        result = {"status":"done", "returncode":0}
                    else:
                        result = {"status":"ready", "capacity":1310720}
                    print(json.dumps(result), flush=True)
                    sys.stdin.readline()
                elif "rev-parse" in arguments:
                    print("a" * 40)
                elif "cat" in arguments:
                    for name in ("dcp-kv-gather", "dcp-packed-read", "bde-v2"):
                        print("```json\\n" + json.dumps({"id":"serve-"+name, "status":"pass", "quality":"verified", "activation":"verified", "missing":[]}) + "\\n```")
            """)
            )
            docker.chmod(0o755)
            (ais / "aisbench_test.py").write_text(
                textwrap.dedent("""\
                from pathlib import Path
                counter = Path("runs.txt")
                count = int(counter.read_text()) + 1 if counter.exists() else 1
                counter.write_text(str(count))
                print({"TTFT avg":1000+count, "total_req":16, "output_len":1, "E2E_time":10, "qps":1.6})
            """)
            )
            output, events = root / "results", root / "events.jsonl"
            argv = [
                "night",
                "--output",
                str(output),
                "--ais-dir",
                str(ais),
                "--ais-env",
                str(envfile),
                "--variants",
                "dcp-kv-gather,dcp-packed-read,bde-v2",
                "--repeats",
                "2",
            ]
            flushes = []
            with (
                patch.object(sys, "argv", argv),
                patch.dict(
                    os.environ,
                    {
                        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
                        "NIGHT_TEST_EVENTS": str(events),
                    },
                ),
                patch.object(module, "require_free_port"),
                patch.object(
                    module,
                    "http",
                    side_effect=lambda *args, **kwargs: (
                        flushes.append(args[0]) or "Cache flushed.\n"
                    ),
                ),
            ):
                code = module.main()
            self.assertEqual(code, 1)
            rows = json.loads((output / "results.json").read_text())
            self.assertEqual(
                sum(row["status"] == "pass" and "metrics" in row for row in rows), 4
            )
            self.assertEqual(
                [row["case"] for row in rows if row["status"] == "fail"],
                ["dcp-packed-read"],
            )
            self.assertEqual(flushes, ["/flush_cache?timeout=60"] * 4)
            self.assertEqual((ais / "runs.txt").read_text(), "4")
            self.assertEqual(
                json.loads((output / "bde-v2/aggregate.json").read_text())[
                    "median_ttft_ms"
                ],
                1003.5,
            )
            self.assertTrue((output / "container-artifacts/copied.txt").is_file())
            self.assertEqual(len(events.read_text().splitlines()), 6)
            self.assertIn("bde-v2/2", (output / "SUMMARY.md").read_text())

    def test_historical_topology_validation_uses_composed_recipe_and_ep_size(self):
        module = self.runner()
        root = Path(__file__).resolve().parents[5]
        harness = module.load_module(
            root / "benchmark/prefill_1m/test_glm53_features.py", "composed_topology"
        )
        config = dict(variant="baseline", reference_recipe="historical-bde", maxreq=1)
        resolved = dict(
            tp_size=16,
            dcp_size=16,
            max_running_requests=1,
            prefill_max_requests=1,
            quantization="modelslim",
            cuda_graph_backend_prefill="disabled",
            cuda_graph_backend_decode="disabled",
        )
        module.validate_variant_topology(harness, config, resolved)
        with self.assertRaisesRegex(RuntimeError, "dcp_size"):
            module.validate_variant_topology(
                harness, config, dict(resolved, dcp_size=1)
            )
        config["variant"] = "megamoe"
        resolved.update(ep_size=16, fuseep_mode=3, moe_a2a_backend="ascend_fuseep")
        module.validate_variant_topology(harness, config, resolved)
        with self.assertRaisesRegex(RuntimeError, "FuseEP"):
            module.validate_variant_topology(harness, config, dict(resolved, ep_size=1))

    def test_nightly_legacy_control_and_graph_launch_remain_distinct(self):
        module = self.runner()
        root = Path(__file__).resolve().parents[5]
        harness = module.load_module(
            root / "benchmark/prefill_1m/test_glm53_features.py", "nightly_launch"
        )
        config = dict(
            variant="legacy-control-40s",
            reference_recipe="historical-bde",
            checkpoint="/tmp/weights",
            chunk=32768,
            context=1048576,
            memfrac=0.85,
            maxreq=1,
            kv_rows=67584,
        )
        command, env = module.variant_launch(harness, root, config)
        self.assertEqual(command[command.index("--chunked-prefill-size") + 1], "16384")
        self.assertEqual(command[command.index("--max-running-requests") + 1], "64")
        self.assertEqual(env["SGLANG_NPU_DSA_CP_TOKEN_SHARDED"], "False")
        config["variant"] = "graph-piecewise"
        command, env = module.variant_launch(harness, root, config)
        self.assertEqual(
            command[command.index("--cuda-graph-backend-prefill") + 1], "tc_piecewise"
        )
        self.assertEqual(command[command.index("--chunked-prefill-size") + 1], "32768")
        with self.assertRaisesRegex(RuntimeError, "graph-piecewise-replay"):
            module.timed_activation(
                'NPU_FEATURE_EVIDENCE {"feature":"dsa-residual-shard"}',
                config["variant"],
                harness.scenario_for(config["variant"], "historical-bde"),
            )

    def test_performance_only_has_no_qualification_launches(self):
        module = self.runner()
        output = io.StringIO()
        with (
            patch.object(
                sys,
                "argv",
                [
                    "night",
                    "--dry-run",
                    "--performance-only",
                    "--reference-recipe",
                    "historical-bde",
                    "--variants",
                    "baseline,megamoe",
                    "--repeats",
                    "1",
                ],
            ),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(module.main(), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["feature_commands"], {})
        self.assertEqual(plan["variants"], ["baseline", "megamoe"])
        self.assertTrue(plan["server_settings"]["performance_only"])
        self.assertEqual(plan["repeats"], 1)

    def test_metadata_batch2_launch_and_topology_preserve_baseline_settings(self):
        module = self.runner()
        root = Path(__file__).resolve().parents[5]
        harness = module.load_module(
            root / "benchmark/prefill_1m/test_glm53_features.py", "metadata_launch"
        )
        config = dict(
            variant="dsa-metadata-batch2",
            reference_recipe="historical-bde",
            checkpoint="/tmp/weights",
            chunk=32768,
            context=1048576,
            memfrac=0.85,
            maxreq=1,
            kv_rows=67584,
            performance_only=True,
        )
        command, env = module.variant_launch(harness, root, config)
        for flag, value in (
            ("--max-running-requests", "2"),
            ("--prefill-max-requests", "2"),
            ("--chunked-prefill-size", "32768"),
        ):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertEqual(env["SGLANG_NPU_DSA_METADATA_REUSE"], "True")
        self.assertEqual(env["SGLANG_NPU_ENABLE_DCP_PACKED_READ"], "True")
        module.validate_variant_topology(
            harness,
            config,
            dict(
                tp_size=16,
                dcp_size=16,
                max_running_requests=2,
                prefill_max_requests=2,
                max_prefill_tokens=32768,
                quantization="modelslim",
                cuda_graph_backend_prefill="disabled",
                cuda_graph_backend_decode="disabled",
            ),
        )
        features = ("dsa-residual-shard", "dsa-metadata-reuse", "dcp-prefill-batch-2")
        log = "\n".join(
            "NPU_FEATURE_EVIDENCE " + json.dumps(dict(feature=name))
            for name in features
        )
        module.timed_activation(log, "dsa-metadata-batch2")
        with self.assertRaisesRegex(RuntimeError, "dcp-prefill-batch-2"):
            module.timed_activation(log.rsplit("\n", 1)[0], "dsa-metadata-batch2")

    def test_packed_batch2_requires_its_own_activation(self):
        module = self.runner()
        root = Path(__file__).resolve().parents[5]
        harness = module.load_module(
            root / "benchmark/prefill_1m/test_glm53_features.py", "packed_batch2_launch"
        )
        config = dict(
            variant="dsa-metadata-packed-batch2",
            reference_recipe="historical-bde",
            checkpoint="/tmp/weights",
            chunk=32768,
            context=1048576,
            memfrac=0.85,
            maxreq=1,
            kv_rows=67584,
            performance_only=True,
        )
        command, env = module.variant_launch(harness, root, config)
        self.assertEqual(command[command.index("--prefill-max-requests") + 1], "2")
        self.assertEqual(env["SGLANG_NPU_ENABLE_DCP_PACKED_READ_BATCH2"], "True")
        features = (
            "dsa-residual-shard",
            "dsa-metadata-reuse",
            "dcp-prefill-batch-2",
            "dcp-packed-read-batch2",
        )
        log = "\n".join(
            "NPU_FEATURE_EVIDENCE " + json.dumps(dict(feature=name))
            for name in features
        )
        module.timed_activation(log, config["variant"])
        with self.assertRaisesRegex(RuntimeError, "dcp-packed-read-batch2"):
            module.timed_activation(log.rsplit("\n", 1)[0], config["variant"])

    def test_timed_megamoe_requires_kernel_and_local_ep_activation(self):
        module = self.runner()
        log = "\n".join(
            "NPU_FEATURE_EVIDENCE " + json.dumps(dict(feature=name))
            for name in ("dsa-residual-shard", "megamoe", "dsa-ep-moe-local")
        )
        self.assertIn("megamoe", module.timed_activation(log, "megamoe"))
        with self.assertRaisesRegex(RuntimeError, "dsa-ep-moe-local"):
            module.timed_activation(log.rsplit("\n", 1)[0], "megamoe")

    def test_diagnostic_gate_only_bypasses_activated_parity_failures(self):
        module = self.runner()
        gate = dict(
            status="fail",
            failure_phase="parity",
            activation="verified",
            quality="unverified",
            missing=[],
        )
        self.assertFalse(module.benchmark_admitted(gate))
        self.assertTrue(module.benchmark_admitted(gate, allow_parity_failure=True))
        self.assertTrue(
            module.benchmark_admitted(gate, baseline=True, allow_parity_failure=True)
        )
        for changes in (
            dict(failure_phase="candidate"),
            dict(failure_phase="reference"),
            dict(activation="unverified"),
            dict(missing=["megamoe"]),
            dict(status="inactive"),
            dict(status="dependency_blocked"),
        ):
            with self.subTest(changes=changes):
                self.assertFalse(
                    module.benchmark_admitted(
                        dict(gate, **changes), allow_parity_failure=True
                    )
                )
        passed = dict(status="pass", quality="verified", activation="unverified")
        self.assertTrue(module.benchmark_admitted(passed, baseline=True))
        self.assertFalse(module.benchmark_admitted(passed))

    def test_failed_requests_and_missing_summary_cannot_pass(self):
        module = self.runner()
        for text in ("no summary", "│ Failed Requests │ total │ 2 │"):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                module.ais_metrics(text, 16)

    def test_moe_launch_matches_qualified_fuseep_topology(self):
        module = self.runner()
        repo = Path(__file__).resolve().parents[5]
        harness = module.load_module(
            repo / "benchmark/prefill_1m/test_glm53_features.py", "night_matrix"
        )
        config = {
            "checkpoint": "/tmp/weights",
            "chunk": 4096,
            "context": 8192,
            "memfrac": 0.88,
            "maxreq": 64,
            "kv_rows": 65536,
        }
        for variant in (
            "baseline",
            "tp-shared",
            "multistream",
            "megamoe",
            "auto-hccl",
            "bde-v2",
        ):
            with self.subTest(variant=variant):
                command, env = module.variant_launch(
                    harness, repo, dict(config, variant=variant)
                )
                self.assertEqual(
                    command[command.index("--dcp-size") + 1],
                    "16" if variant == "bde-v2" else "1",
                )
                if variant == "bde-v2":
                    self.assertEqual(
                        command[command.index("--prefill-max-requests") + 1], "1"
                    )
                else:
                    self.assertNotIn("--prefill-max-requests", command)
                self.assertEqual(
                    env["SGLANG_NPU_DSA_CP_TOKEN_SHARDED"],
                    "True" if variant == "bde-v2" else "False",
                )
                if variant == "megamoe":
                    self.assertEqual(command[command.index("--fuseep-mode") + 1], "3")
                    self.assertEqual(command[command.index("--ep-size") + 1], "16")
                    self.assertEqual(
                        command[command.index("--moe-a2a-backend") + 1], "ascend_fuseep"
                    )
                if variant in ("tp-shared", "multistream", "megamoe"):
                    self.assertIn("--disable-shared-experts-fusion", command)
                if variant == "baseline":
                    self.assertEqual(env["SGLANG_NPU_USE_MULTI_STREAM"], "False")
                if variant == "auto-hccl":
                    self.assertNotIn("HCCL_BUFFSIZE", env)

    def test_historical_megamoe_timing_keeps_dispatch_capacity(self):
        module = self.runner()
        repo = Path(__file__).resolve().parents[5]
        harness = module.load_module(
            repo / "benchmark/prefill_1m/test_glm53_features.py", "historical_timing"
        )
        config = dict(
            variant="megamoe",
            reference_recipe="historical-bde",
            checkpoint="/tmp/weights",
            chunk=16384,
            context=1048576,
            memfrac=0.85,
            maxreq=64,
            kv_rows=67584,
            megamoe_dispatch_tokens=2048,
        )
        command, env = module.variant_launch(harness, repo, config)
        for flag, value in [
            ("--chunked-prefill-size", "32768"),
            ("--max-running-requests", "1"),
            ("--dcp-size", "16"),
            ("--ep-size", "16"),
        ]:
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertEqual(env["SGLANG_NPU_DSA_CP_TOKEN_SHARDED"], "True")
        self.assertEqual(env["SGLANG_NPU_INDEXER_INT8"], "True")
        self.assertEqual(env["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"], "2048")

    def test_worker_disconnect_stops_owned_feature_process(self):
        module = self.runner()
        try:
            module.require_free_port()
        except RuntimeError:
            self.skipTest("Port 30000 is occupied")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = root / "benchmark/prefill_1m/test_glm53_features.py"
            harness.parent.mkdir(parents=True)
            pidfile = root / "child.pid"
            harness.write_text(
                textwrap.dedent(f"""\
                import subprocess, sys, time
                from pathlib import Path
                def main():
                    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
                    Path({str(pidfile)!r}).write_text(str(child.pid))
                    time.sleep(120)
            """)
            )
            process = subprocess.Popen(
                [sys.executable, module.__file__, "--worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                process.stdin.write(
                    json.dumps(
                        {
                            "repo": str(root),
                            "folder": str(root / "logs"),
                            "action": "features",
                            "arguments": [],
                        }
                    )
                    + "\n"
                )
                process.stdin.flush()
                deadline = time.monotonic() + 10
                while (
                    not pidfile.exists()
                    and process.poll() is None
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.05)
                self.assertTrue(pidfile.exists())
                child_pid = int(pidfile.read_text())
                process.stdin.close()
                self.assertEqual(process.wait(timeout=15), 0)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=15)


if __name__ == "__main__":
    unittest.main()
