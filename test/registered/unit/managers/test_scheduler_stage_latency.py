"""Stage-latency probes must format without removed scheduler/runner attributes."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSchedulerStageLatency(CustomTestCase):
    def _assert_probes(self, relative_path, marker, **state):
        source = Path(__file__).resolve().parents[4] / relative_path
        tree = ast.parse(source.read_text())
        probes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.JoinedStr)
            and any(
                isinstance(part, ast.Constant)
                and isinstance(part.value, str)
                and marker in part.value
                for part in node.values
            )
        ]
        self.assertEqual(len(probes), 2)
        namespace = dict(
            self=SimpleNamespace(),
            get_parallel=lambda: SimpleNamespace(tp_rank=3),
            _ts_t0=1.0,
            _ts_t1=2.0,
            **state,
        )
        for probe in probes:
            rendered = eval(
                compile(ast.Expression(probe), str(source), "eval"), namespace
            )
            self.assertIn("rank=3", rendered)

    def test_batch_probes_read_the_published_rank(self):
        self._assert_probes(
            "python/sglang/srt/managers/scheduler.py",
            "[SGL-TS] BATCH-",
            batch=SimpleNamespace(
                reqs=[1, 2], extend_num_tokens=32, forward_mode="EXTEND"
            ),
        )

    def test_model_runner_probes_read_the_published_rank(self):
        self._assert_probes(
            "python/sglang/srt/model_executor/model_runner.py",
            "[SGL-TS] MODEL-FWD-",
            forward_batch=SimpleNamespace(forward_mode="EXTEND"),
            _ts_nseq=2,
            _ts_ntok=32,
        )


if __name__ == "__main__":
    unittest.main()
