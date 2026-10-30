"""TP16/DCP16 serving smoke with full ModelSlim INT weights."""

import importlib.util
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_npu_ci(est_time=1800, suite="base-b-test-16-npu-a3")

MODEL_PATH = os.environ.get(
    "GLM_5_2_W4A8_MODEL_PATH_OVERRIDE",
    os.environ.get("MODEL_PATH", "/home/weights/GLM5.2-w4a8-int/"),
)


@unittest.skipUnless(
    os.path.isfile(os.path.join(MODEL_PATH, "config.json")),
    f"model config not found at {MODEL_PATH}",
)
class TestNPUBDEV2LaunchTP16DCP16(CustomTestCase):
    """Launch the bde_v2 recipe on 16 NPUs and verify serve + one generation."""

    @classmethod
    def setUpClass(cls):
        path = (
            Path(__file__).resolve().parents[4]
            / "benchmark/prefill_1m/launch_bde_v2.py"
        )
        spec = importlib.util.spec_from_file_location("bde_launch", path)
        launch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launch)

        args = SimpleNamespace(
            tp=16,
            dcp=16,
            model_path=MODEL_PATH,
            real_weights=True,
            context_length=4096,
        )
        inherited = dict(os.environ)
        inherited.update(
            CAPTURE="0",
            EXTRA_ARGS="",
            MAXREQ="4",
            CHUNK="4096",
            PORT=DEFAULT_URL_FOR_TEST.rsplit(":", 1)[1],
        )
        command, env, _ = launch.launch_config(args, inherited)
        cls.model = MODEL_PATH
        cls.other_args = command[3:]
        for flag in ("--model-path", "--host", "--port"):
            at = cls.other_args.index(flag)
            del cls.other_args[at : at + 2]
        cls.other_args += ["--quantization", "modelslim"]
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=max(1800, DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH),
            other_args=cls.other_args,
            env=env,
            device="npu",
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)

    def _get_server_info(self):
        with urlopen(f"{DEFAULT_URL_FOR_TEST}/server_info", timeout=30) as resp:
            return json.load(resp)

    def test_server_info_reports_tp16_dcp16(self):
        info = self._get_server_info()
        self.assertEqual(info["tp_size"], 16)
        self.assertEqual(info["dcp_size"], 16)
        # Capture would change the recipe measured by the perf A/B.
        self.assertEqual(info["cuda_graph_config"]["decode"]["backend"], "disabled")
        self.assertEqual(info["cuda_graph_config"]["prefill"]["backend"], "disabled")
        self.assertNotEqual(info["load_format"], "dummy")
        self.assertEqual(info["quantization"], "modelslim")
        self.assertNotIn(
            "num_hidden_layers", json.loads(info["json_model_override_args"])
        )

    def test_generation_round_trip(self):
        body = json.dumps(
            {
                "input_ids": [1] * 32,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 8,
                    "ignore_eos": True,
                },
            }
        ).encode()
        request = Request(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=600) as response:
            output = json.load(response)
        self.assertEqual(output["meta_info"]["completion_tokens"], 8)


if __name__ == "__main__":
    unittest.main()
