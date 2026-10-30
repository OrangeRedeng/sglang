"""CPU unit tests for the NPU CUDA-graph backend's Python orchestration."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from sglang.srt.hardware_backend.npu.graph_runner import npu_cudagraph_backend as mod
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_backend(*, frozen_kv=False):
    device_module = SimpleNamespace(
        current_device=mock.Mock(return_value=3),
        synchronize=mock.Mock(),
        set_device=mock.Mock(),
        Event=mock.Mock(),
    )
    runner = SimpleNamespace(
        device_module=device_module,
        enable_torch_compile=False,
        model_runner=SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_frozen_kv_mtp=lambda: frozen_kv)
        ),
    )
    with (
        get_parallel().override(tp_group=SimpleNamespace(barrier=mock.Mock())),
        mock.patch.object(mod.TorchMemorySaverAdapter, "create", return_value=None),
    ):
        return mod.NPUCudaGraphBackend(runner), runner


class TestNPUCudaGraphBackend(unittest.TestCase):
    def test_frozen_updates_finish_before_replay_and_preserve_previous_fence(self):
        for frozen_kv in (False, True):
            with self.subTest(frozen_kv=frozen_kv):
                backend, runner = _make_backend(frozen_kv=frozen_kv)
                key = ShapeKey(size=2)
                graph = mock.Mock()
                backend._graphs[key] = graph
                backend._outputs[key] = "output"
                order = []
                graph.update.side_effect = lambda order=order, **kwargs: order.append(
                    "update"
                )
                graph.replay.side_effect = lambda order=order: order.append("replay")
                previous = mock.Mock()
                previous.synchronize.side_effect = lambda order=order: order.append(
                    "fence"
                )
                backend._rebind_fence = previous

                def defer_update(fn, **kwargs):
                    return SimpleNamespace(result=lambda: fn(**kwargs))

                with mock.patch.object(
                    backend._update_executor, "submit", side_effect=defer_update
                ):
                    output = backend.replay_with_input_update(
                        key,
                        seq_lens=None,
                        cpu_update_input=[{"actual_seq_kvlen": [7, 9]}],
                    )
                expected = ["update", "replay"] if frozen_kv else ["replay", "update"]
                self.assertEqual(order, ["fence"] + expected)
                self.assertEqual(output, "output")
                runner.device_module.Event.return_value.record.assert_called_once()
                backend.cleanup()

    def test_failed_frozen_update_does_not_launch_a_graph_waiting_for_events(self):
        backend, runner = _make_backend(frozen_kv=True)
        key = ShapeKey(size=2)
        graph = mock.Mock()
        graph.update.side_effect = RuntimeError("FIA update failed")
        backend._graphs[key] = graph
        backend._outputs[key] = "output"
        try:
            with self.assertRaisesRegex(RuntimeError, "FIA update failed"):
                backend.replay_with_input_update(
                    key, seq_lens=None, cpu_update_input=[{"actual_seq_kvlen": [7, 9]}]
                )
            graph.replay.assert_not_called()
            runner.device_module.Event.assert_not_called()
        finally:
            backend.cleanup()

    def test_replay_update_converts_legacy_sequence_lengths_to_int32_tensor(self):
        backend, runner = _make_backend()
        graph = mock.Mock()
        shape_key = ShapeKey(size=2)
        backend._graphs[shape_key] = graph
        backend._outputs[shape_key] = "output"

        output = backend.replay_with_input_update(
            shape_key,
            seq_lens=[7, 9],
            attr_name="seq_lens",
            attr_type=torch.empty(0),
        )

        runner.device_module.set_device.assert_called_once_with(3)
        graph.replay.assert_called_once_with()
        update_input = graph.update.call_args.kwargs["cpu_update_input"]
        self.assertEqual(len(update_input), 1)
        self.assertEqual(set(update_input[0]), {"seq_lens"})
        self.assertEqual(update_input[0]["seq_lens"].dtype, torch.int32)
        self.assertTrue(torch.equal(update_input[0]["seq_lens"], torch.tensor([7, 9])))
        self.assertEqual(output, "output")


if __name__ == "__main__":
    unittest.main()
