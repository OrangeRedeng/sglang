"""Unattended host/container orchestration without devices or a Docker daemon."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import patch

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
                        print("```json\\n" + json.dumps({"id":"serve-"+name, "status":"pass", "quality":"verified", "missing":[]}) + "\\n```")
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
                    side_effect=lambda *args, **kwargs: flushes.append(args[0]),
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
