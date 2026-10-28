"""Stage-latency probes must format without removed Scheduler attributes."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSchedulerStageLatency(CustomTestCase):
    def test_batch_probes_read_the_published_rank(self):
        source = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/managers/scheduler.py"
        )
        tree = ast.parse(source.read_text())
        probes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.JoinedStr)
            and any(
                isinstance(part, ast.Constant)
                and isinstance(part.value, str)
                and "[SGL-TS] BATCH-" in part.value
                for part in node.values
            )
        ]
        self.assertEqual(len(probes), 2)
        namespace = dict(
            self=SimpleNamespace(),
            get_parallel=lambda: SimpleNamespace(tp_rank=3),
            batch=SimpleNamespace(
                reqs=[1, 2], extend_num_tokens=32, forward_mode="EXTEND"
            ),
            _ts_t0=1.0,
            _ts_t1=2.0,
        )
        for probe in probes:
            rendered = eval(
                compile(ast.Expression(probe), str(source), "eval"), namespace
            )
            self.assertIn("rank=3", rendered)


if __name__ == "__main__":
    unittest.main()
