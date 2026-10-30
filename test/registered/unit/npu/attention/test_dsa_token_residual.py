"""Residual-state and collective parity for the BDE D1v3/D2v3 port."""

import importlib.util
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from sglang.srt.hardware_backend.npu.attention.dcp_gather import (
    start_o_proj_gather,
    start_query_head_gather,
    validate_legacy_dcp_flags,
)
from sglang.srt.hardware_backend.npu.modules import dsa_token_residual as residual
from sglang.srt.layers.attention.dsa import dsa_cp
from sglang.srt.layers.attention.dsa.dsa_cp_layout import plan_dsa_cp_shard
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class World:
    """Blocking transport boundary backed by the sends of both test ranks."""

    def __init__(self, size):
        self.size, self.calls = size, {}
        self.condition = threading.Condition()

    def group(self, rank):
        world = self

        class Group:
            world_size = world.size
            ranks = list(range(world.size))
            ordinal = 0

            def collective(self, output, value, reduce=False):
                key = self.ordinal
                self.ordinal += 1
                with world.condition:
                    slots = world.calls.setdefault(key, {})
                    slots[rank] = value.clone()
                    world.condition.notify_all()
                    if not world.condition.wait_for(
                        lambda: len(slots) == world.size, timeout=10
                    ):
                        raise RuntimeError("rank collective order diverged")
                    values = [slots[i] for i in range(world.size)]
                    result = sum(values) if reduce else torch.cat(values)
                    if reduce:
                        result = result.chunk(world.size)[rank]
                    output.copy_(result)

            def all_gather_into_tensor(self, output, value):
                self.collective(output, value)

            def reduce_scatter_tensor(self, output, value):
                self.collective(output, value, True)

            def all_gather_into_tensor_async(self, output, value):
                self.collective(output, value)
                return None

        return Group()


class Norm:
    def __call__(self, value, residual=None):
        written = (
            value
            if residual is None
            else (value.float() + residual.float()).to(value.dtype)
        )
        normalized = (
            written.float()
            * torch.rsqrt(written.float().square().mean(-1, keepdim=True) + 1e-6)
        ).to(value.dtype)
        return normalized if residual is None else (normalized, written)


class TestDsaTokenResidual(CustomTestCase):
    def test_decoder_stream_and_last_rows_match_full_width(self):
        for dtype in (torch.float32, torch.bfloat16):
            for lengths, padded in (([5], 6), ([3, 4], 8), ([5], 16)):
                with self.subTest(dtype=dtype, lengths=lengths, padded=padded):
                    world, local = World(2), threading.local()
                    hidden = (
                        torch.arange(padded * 4).reshape(padded, 4).float() / 31
                    ).to(dtype)
                    wa = (torch.arange(16).reshape(4, 4).float() / 37).to(dtype)
                    wf = torch.eye(4, dtype=dtype) * 0.25
                    reference = hidden.clone()
                    norm = Norm()
                    for _ in range(2):
                        reference = (
                            reference.float() + F.linear(norm(reference), wa).float()
                        ).to(dtype)
                        reference = (
                            reference.float() + F.linear(norm(reference), wf).float()
                        ).to(dtype)
                    reference = norm(reference)
                    last = torch.tensor(lengths).cumsum(0) - 1

                    def worker(rank):
                        group = world.group(rank)
                        local.parallel = SimpleNamespace(
                            attn_tp_group=group, attn_tp_size=2, attn_tp_rank=rank
                        )
                        plan = plan_dsa_cp_shard(
                            lengths, lengths, 2, rank, padded_num_tokens=padded
                        )
                        batch = SimpleNamespace(
                            npu_dsa_cp_plan=plan,
                            extend_seq_lens_cpu=lengths,
                            return_logprob=False,
                            extend_logprob_start_lens_cpu=None,
                        )
                        residual_batch.start(batch)

                        class Attention:
                            indexer = None

                            def __call__(self, **kwargs):
                                return F.linear(kwargs["hidden_states"], wa), None

                        class Mlp:
                            reduce_results = False

                            def __call__(self, value, *unused):
                                return F.linear(value, wf / 2)

                        layer = SimpleNamespace(
                            self_attn=Attention(),
                            input_layernorm=norm,
                            post_attention_layernorm=norm,
                            is_layer_sparse=False,
                            mlp=Mlp(),
                        )
                        output = hidden.clone()
                        for _ in range(2):
                            output, _ = residual.decoder_forward(
                                layer,
                                torch.arange(padded),
                                output,
                                batch,
                                None,
                                plan=plan,
                            )
                        output = residual_batch.final_norm(output, batch, norm)
                        pruned = residual.gather_logit_rows(output, batch)
                        torch.testing.assert_close(
                            pruned, reference[last], rtol=0.01, atol=0.01
                        )
                        # Input-logprob gathering keeps the same normalized values.
                        batch.return_logprob = True
                        batch.extend_logprob_start_lens_cpu = [0] * len(lengths)
                        full = residual.gather_logit_rows(output, batch)
                        torch.testing.assert_close(
                            full[: sum(lengths)],
                            reference[: sum(lengths)],
                            rtol=0.01,
                            atol=0.01,
                        )
                        self.assertFalse(batch.npu_dsa_cp_logits_pruned)

                    lookup = lambda: local.parallel
                    with (
                        patch.object(residual, "get_parallel", side_effect=lookup),
                        patch.object(dsa_cp, "get_parallel", side_effect=lookup),
                        patch.object(dsa_cp, "_enable_dsa_cp", True),
                        ThreadPoolExecutor(2) as pool,
                    ):
                        list(pool.map(worker, range(2)))

    def test_streamed_column_shards_match_resident_matrix(self):
        world = World(2)
        weights = [
            torch.arange(12, dtype=torch.float32).reshape(4, 3) + rank * 100
            for rank in range(2)
        ]
        expected = torch.cat(weights, dim=1)

        def worker(rank):
            gathered, send, work = start_o_proj_gather(world.group(rank), weights[rank])
            if work is not None:
                work.wait()
            torch.testing.assert_close(gathered.t(), expected, rtol=0, atol=0)
            x = torch.arange(18, dtype=torch.float32).reshape(3, 6)
            torch.testing.assert_close(
                F.linear(x, gathered.t()), F.linear(x, expected), rtol=0, atol=0
            )

        with ThreadPoolExecutor(2) as pool:
            list(pool.map(worker, range(2)))

    def test_preissued_decode_heads_match_synchronous_head_order(self):
        world = World(2)
        nope = [
            torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) + rank * 100
            for rank in range(2)
        ]
        rope = [
            torch.arange(8, dtype=torch.float32).reshape(2, 2, 2) + rank * 1000
            for rank in range(2)
        ]

        def worker(rank):
            actual_nope, actual_rope = start_query_head_gather(
                world.group(rank), nope[rank], rope[rank]
            ).finish()
            torch.testing.assert_close(
                actual_nope, torch.cat(nope, dim=1), rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual_rope, torch.cat(rope, dim=1), rtol=0, atol=0
            )

        with ThreadPoolExecutor(2) as pool:
            list(pool.map(worker, range(2)))

    def test_indexer_keeps_the_actual_padded_local_grid(self):
        from sglang.srt.layers.attention.dsa import dsa_npu_indexer as indexer

        plan = plan_dsa_cp_shard([5], [6149], 2, 1, padded_num_tokens=16)
        batch = SimpleNamespace(
            npu_dsa_cp_plan=plan,
            npu_dsa_cp_attention_local=True,
            npu_dsa_cp_token_sharded=True,
            positions=torch.arange(16),
        )
        with (
            patch.object(dsa_cp, "_enable_dsa_cp", True),
            patch.object(
                indexer, "get_parallel", return_value=SimpleNamespace(attn_tp_size=2)
            ),
        ):
            shard = indexer._get_indexer_query_shard(batch, 8)
            self.assertEqual((shard.start, shard.rows, shard.num_real), (8, 8, 0))
            topk = torch.arange(16).reshape(8, 2)
            torch.testing.assert_close(
                shard.resolve(topk, 16, batch), topk, rtol=0, atol=0
            )
            self.assertTrue(batch.npu_indexer_topk_is_local)
            with self.assertRaisesRegex(ValueError, "different local token grid"):
                indexer._get_indexer_query_shard(batch, 4)

    def test_indexer_keys_use_per_request_not_cumulative_lengths(self):
        from sglang.srt.layers.attention.dsa import dsa_npu_indexer as indexer

        plan = plan_dsa_cp_shard([3, 5], [6003, 6105], 2, 0)
        batch = SimpleNamespace(
            npu_dsa_cp_plan=plan,
            npu_dsa_cp_attention_local=True,
            npu_dsa_cp_token_sharded=True,
            positions=torch.arange(8),
        )
        with (
            patch.object(dsa_cp, "_enable_dsa_cp", True),
            patch.object(
                indexer, "get_parallel", return_value=SimpleNamespace(attn_tp_size=2)
            ),
        ):
            shard = indexer._get_indexer_query_shard(batch, 4)
        torch.testing.assert_close(
            shard.actual_seq_lengths_q, torch.tensor([3, 4], dtype=torch.int32)
        )
        torch.testing.assert_close(
            shard.actual_seq_lengths_kv, torch.tensor([6003, 6101], dtype=torch.int32)
        )

    def test_collective_control_flag_is_compile_visible_and_scoped(self):
        from sglang.srt.runtime_context import get_forward

        fn = torch.compile(
            lambda x: x + int(get_forward().npu_dsa_cp_token_sharded),
            backend="eager",
            fullgraph=True,
        )
        x = torch.ones(2)
        with get_forward().scoped(npu_dsa_cp_token_sharded=True):
            torch.testing.assert_close(fn(x), x + 1)
        torch.testing.assert_close(fn(x), x)

    def test_empty_query_rank_still_publishes_full_kv(self):
        writes = []
        plan = plan_dsa_cp_shard([5], [6149], 2, 1, padded_num_tokens=16)
        key = torch.arange(48).reshape(16, 1, 3).float()
        rope = torch.ones(16, 1, 2)

        class Pool:
            def set_kv_buffer(self, layer, loc, k, r):
                writes.append((k.clone(), r.clone()))

        def forbidden(*args, **kwargs):
            raise AssertionError("zero-query rank must not launch sparse attention")

        m = SimpleNamespace(attn_mqa_for_dsa_cp=forbidden, kv_lora_rank=3)
        batch = SimpleNamespace(out_cache_loc=torch.arange(16))
        with patch.object(residual, "get_token_to_kv_pool", return_value=Pool()):
            result = residual.sparse_attention_or_empty(
                m,
                torch.zeros(8, 2, 3),
                torch.zeros(8, 2, 2),
                key,
                rope,
                torch.full((8, 2), -1),
                batch,
                plan,
            )
        torch.testing.assert_close(result, torch.zeros(8, 6), rtol=0, atol=0)
        torch.testing.assert_close(writes[0][0], key, rtol=0, atol=0)
        torch.testing.assert_close(writes[0][1], rope, rtol=0, atol=0)

    def test_archive_launch_defaults_and_real_weights_are_explicit(self):
        path = (
            Path(__file__).resolve().parents[5]
            / "benchmark/prefill_1m/launch_bde_v2.py"
        )
        spec = importlib.util.spec_from_file_location("bde_launch", path)
        launch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launch)
        args = SimpleNamespace(
            tp=None, dcp=None, model_path=None, real_weights=False, context_length=None
        )
        command, env, selected = launch.launch_config(args, {})
        for flag, value in (
            ("--tp-size", "8"),
            ("--dcp-size", "8"),
            ("--load-format", "dummy"),
            ("--chunked-prefill-size", "32768"),
            ("--max-running-requests", "64"),
            ("--mem-fraction-static", "0.70"),
        ):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn("--context-length", command)
        validate_legacy_dcp_flags(env, tp_size=8, dcp_size=8)
        with self.assertRaisesRegex(ValueError, "equal TP/DCP"):
            validate_legacy_dcp_flags(env, tp_size=16, dcp_size=8)
        args.real_weights, args.context_length = True, 65536
        command, _, _ = launch.launch_config(args, {})
        self.assertEqual(command[command.index("--load-format") + 1], "auto")
        self.assertEqual(command[command.index("--context-length") + 1], "65536")


if __name__ == "__main__":
    unittest.main()
