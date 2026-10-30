"""TP8/DCP8 serving smoke with half-depth dummy weights; no quality claim."""

import importlib.util
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen

from sglang.test.ascend.test_ascend_utils import GLM_5_2_BF16_MODEL_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_npu_ci(est_time=900, suite="base-b-test-8-npu-a3")

MODEL_PATH = os.environ.get("GLM_5_2_W4A8_MODEL_PATH_OVERRIDE", GLM_5_2_BF16_MODEL_PATH)


@unittest.skipUnless(
    os.path.isfile(os.path.join(MODEL_PATH, "config.json")),
    f"model config not found at {MODEL_PATH}",
)
class TestNPUBDEV2LaunchTP8DCP8(CustomTestCase):
    """Launch the bde_v2 recipe on 8 NPUs and verify serve + one generation."""

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
            tp=8,
            dcp=8,
            model_path=MODEL_PATH,
            real_weights=False,
            context_length=4096,
        )
        inherited = dict(os.environ)
        inherited.update(
            CAPTURE="0", EXTRA_ARGS="", PORT=DEFAULT_URL_FOR_TEST.rsplit(":", 1)[1]
        )
        command, env, _ = launch.launch_config(args, inherited)
        with open(Path(MODEL_PATH) / "config.json") as f:
            config = json.load(f)
        # Retain the model geometry while reducing launch memory and time.
        layers = max(1, config["num_hidden_layers"] // 2)
        cls.model = MODEL_PATH
        cls.other_args = command[3:]
        for flag in ("--model-path", "--host", "--port"):
            at = cls.other_args.index(flag)
            del cls.other_args[at : at + 2]
        cls.other_args += [
            "--json-model-override-args",
            json.dumps({"num_hidden_layers": layers}),
        ]
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=max(900, DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH),
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

    def test_server_info_reports_tp8_dcp8(self):
        info = self._get_server_info()
        self.assertEqual(info["tp_size"], 8)
        self.assertEqual(info["dcp_size"], 8)
        # Capture would change the recipe measured by the perf A/B.
        self.assertEqual(info["cuda_graph_config"]["decode"]["backend"], "disabled")
        self.assertEqual(info["cuda_graph_config"]["prefill"]["backend"], "disabled")
        self.assertEqual(info["load_format"], "dummy")

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
