"""Residual-state and collective parity for the BDE D1v3/D2v3 port."""

import ast
import importlib.util
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.attention.dcp_gather import (
    start_o_proj_gather,
    start_query_head_gather,
    validate_legacy_dcp_flags,
)
from sglang.srt.hardware_backend.npu.modules import dsa_token_residual as residual
from sglang.srt.layers.attention.dsa import dsa_cp
from sglang.srt.layers.attention.dsa.dsa_cp_layout import plan_dsa_cp_shard
from sglang.srt.layers.layer_boundary.residual import access as residual_access
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
    def test_tp16_short_unpadded_warmup_uses_unsharded_path(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        with (
            patch.object(dsa_cp, "_enable_dsa_cp", True),
            patch.object(
                dsa_cp,
                "get_parallel",
                return_value=SimpleNamespace(attn_tp_size=16, attn_tp_rank=0),
            ),
            envs.SGLANG_NPU_DSA_CP_MIN_TOKENS.override(0),
        ):
            for tokens in (1, 6, 15, 16):
                with self.subTest(tokens=tokens):
                    batch = SimpleNamespace(
                        forward_mode=ForwardMode.EXTEND,
                        extend_seq_lens_cpu=[tokens],
                        extend_prefix_lens_cpu=[0],
                        npu_dsa_cp_input_rows=tokens,
                    )
                    plan = dsa_cp.get_dsa_cp_plan(batch, 2048)
                    if tokens < 16:
                        self.assertIsNone(plan)
                    else:
                        self.assertEqual(plan.rows, 1)

    @staticmethod
    def _native_decoder_forward():
        # Execute the real entry point without importing accelerator-only model classes.
        path = (
            Path(__file__).resolve().parents[5]
            / "python/sglang/srt/models/deepseek_v2.py"
        )
        tree = ast.parse(path.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "DeepseekV2DecoderLayer"
        )
        forward = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "forward"
        )
        namespace = {
            "_is_npu": True,
            "envs": envs,
            "residual_access": residual_access,
            "get_attn_tp_context": lambda: SimpleNamespace(
                clear_attn_inputs=lambda: None
            ),
        }
        exec(  # noqa: S102 - executes the production entry point
            compile(
                "from __future__ import annotations\n" + ast.unparse(forward),
                str(path),
                "exec",
            ),
            namespace,
        )
        return namespace["forward"]

    def test_decoder_stream_and_last_rows_match_full_width(self):
        native_forward = self._native_decoder_forward()
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

                    def worker(rank, local=local):
                        group = world.group(rank)
                        local.parallel = SimpleNamespace(
                            attn_tp_group=group, attn_tp_size=2, attn_tp_rank=rank
                        )
                        plan = plan_dsa_cp_shard(
                            lengths, lengths, 2, rank, padded_num_tokens=padded
                        )
                        local.plan = plan
                        batch = SimpleNamespace(
                            npu_dsa_cp_plan=plan,
                            extend_seq_lens_cpu=lengths,
                            return_logprob=False,
                            extend_logprob_start_lens_cpu=None,
                        )
                        residual_batch.start(batch)

                        class Attention:
                            indexer = None
                            use_dsa = True

                            def __call__(self, **kwargs):
                                return F.linear(kwargs["hidden_states"], wa), None

                        class Mlp:
                            down_proj = SimpleNamespace(reduce_results=False)

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
                            output, _ = native_forward(
                                layer,
                                torch.arange(padded),
                                output,
                                batch,
                                None,
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
                        patch.object(
                            residual,
                            "full_head_plan",
                            side_effect=lambda *args, local=local: local.plan,
                        ),
                        envs.SGLANG_NPU_DSA_CP_TOKEN_SHARDED.override(True),
                        patch.object(dsa_cp, "get_parallel", side_effect=lookup),
                        patch.object(dsa_cp, "_enable_dsa_cp", True),
                        ThreadPoolExecutor(2) as pool,
                    ):
                        list(pool.map(worker, range(2)))

    def test_native_indexer_keeps_local_queries_and_replicates_int8_keys(self):
        """The full entry point must neither slice local rows twice nor write partial index-K."""
        for neox in (False, True):
            with self.subTest(neox=neox):
                self._check_native_indexer(neox)

    def _check_native_indexer(self, neox):
        from sglang.srt.layers.attention.dsa import dsa_npu_indexer as indexer
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        world, local = World(2), threading.local()
        hidden = torch.arange(64).reshape(16, 4).float() / 2
        expected_key = hidden.to(torch.int8)

        class Rope:
            is_neox_style = neox
            cos_sin_cache = torch.tensor([[1.0, 0.0]]).expand(16, 2)

            def __call__(self, positions, q, k):
                return q, k

        def quantize(value, *unused):
            return value.to(torch.int8), torch.ones(value.shape[:-1])

        def lightning(**kwargs):
            q, k = kwargs["query"][:, 0].float(), kwargs["key"].float()
            scores = q @ k.T
            end = int(kwargs["actual_seq_lengths_key"][-1])
            frontier = torch.arange(q.shape[0]) + end - q.shape[0]
            scores.masked_fill_(
                torch.arange(k.shape[0])[None] > frontier[:, None],
                -float("inf"),
            )
            return scores.topk(2, dim=-1).indices.unsqueeze(1)

        def worker(rank):
            group = world.group(rank)
            plan = plan_dsa_cp_shard([16], [16], 2, rank)
            batch = SimpleNamespace(
                npu_dsa_cp_plan=plan,
                npu_dsa_cp_attention_local=True,
                npu_dsa_cp_token_sharded=True,
                positions=torch.arange(16),
                out_cache_loc=torch.arange(16),
                forward_mode=ForwardMode.EXTEND,
                attn_cp_metadata=None,
                seq_lens=torch.tensor([16]),
                extend_seq_lens=torch.tensor([16]),
            )
            cache = SimpleNamespace(
                start_layer=0,
                index_k_scale_buffer=True,
                indexer_hadamard_128=None,
                index_k_store_dtype=torch.int8,
                index_k_scale_dtype=torch.float32,
            )
            cache.set_index_k_buffer = lambda layer, loc, value: setattr(
                cache, "keys", value
            )
            cache.set_index_k_scale_buffer = lambda layer, loc, value: setattr(
                cache, "scales", value
            )
            cache.get_index_k_buffer = lambda layer: cache.keys
            cache.get_index_k_scale_buffer = lambda layer: cache.scales
            local.pool, local.parallel = (
                cache,
                SimpleNamespace(
                    dcp_enabled=True,
                    attn_dcp_size=2,
                    attn_tp_size=2,
                    attn_tp_rank=rank,
                    attn_tp_group=group,
                ),
            )
            local.backend = SimpleNamespace(
                is_draft_worker=False,
                forward_metadata=SimpleNamespace(
                    seq_lens_cpu_int=torch.tensor([16]),
                    dcp_origin_out_cache_loc=torch.arange(16),
                    block_tables=torch.tensor([[0]]),
                ),
            )
            projection = lambda value: (value, None)
            model = SimpleNamespace(
                dsa_enable_prefill_cp=False,
                rotary_emb=Rope(),
                alt_stream=None,
                wq_b=projection,
                wk=projection,
                k_norm=lambda value: value,
                weights_proj=lambda value: (
                    torch.ones((value.shape[0], 1)),
                    None,
                ),
                hidden_size=4,
                n_heads=1,
                head_dim=4,
                rope_head_dim=2,
                index_topk=2,
            )
            own = hidden[plan.local_start : plan.local_start + plan.rows]
            actual = indexer.DSANPUIndexerMixin.forward_npu(
                model,
                own,
                own,
                batch.positions[plan.local_start : plan.local_end],
                batch,
                0,
            )
            q = own.reshape(8, 1, 4)
            if neox:
                q = torch.cat((q[..., :2], q[..., 2:]), -1)
            expected = lightning(
                query=q.to(torch.int8),
                key=expected_key,
                actual_seq_lengths_key=torch.tensor([plan.key_lens[-1]]),
            ).squeeze(1)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(cache.keys, expected_key)
            self.assertEqual(cache.scales.numel(), 16)
            self.assertTrue(batch.npu_indexer_topk_is_local)

        with (
            envs.SGLANG_NPU_USE_MULTI_STREAM.override(False),
            patch.object(dsa_cp, "_enable_dsa_cp", True),
            patch.object(indexer, "get_parallel", side_effect=lambda: local.parallel),
            patch.object(
                indexer, "get_attn_backend", side_effect=lambda: local.backend
            ),
            patch.object(
                indexer, "get_token_to_kv_pool", side_effect=lambda: local.pool
            ),
            patch.object(indexer, "_use_ag_after_qlora", False),
            patch.object(
                indexer,
                "_quantize_npu_indexer_activation",
                side_effect=quantize,
            ),
            patch.object(
                indexer,
                "torch_npu",
                SimpleNamespace(npu_rotary_mul=lambda value, *args: value),
                create=True,
            ),
            patch.object(
                torch.ops.npu,
                "npu_rotary_mul",
                lambda value, *args: value,
                create=True,
            ),
            patch.object(
                torch.ops.npu,
                "npu_quant_lightning_indexer",
                lightning,
                create=True,
            ),
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
                shard.resolve(topk, 16, batch), torch.zeros_like(topk), rtol=0, atol=0
            )
            self.assertTrue(batch.npu_indexer_topk_is_local)
            with self.assertRaisesRegex(ValueError, "different local token grid"):
                indexer._get_indexer_query_shard(batch, 4)

    @staticmethod
    def _tp8_indexer_case(legacy, total, rank):
        from sglang.srt.layers.attention.dsa import dsa_npu_indexer as indexer
        from sglang.srt.layers.attention.dsa.dsa_token_shard_layout import (
            plan_dsa_token_shard,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        planner = plan_dsa_cp_shard if legacy else plan_dsa_token_shard
        plan = planner([total], [4096 + total], 8, rank)
        batch = SimpleNamespace(
            npu_dsa_cp_plan=plan if legacy else None,
            npu_dsa_token_shard_plan=None if legacy else plan,
            npu_dsa_cp_token_sharded=legacy,
            positions=torch.arange(plan.num_tokens_pad),
            seq_lens=torch.tensor([4096 + total]),
            extend_seq_lens_cpu=[total],
            extend_prefix_lens_cpu=[4096],
            forward_mode=ForwardMode.EXTEND,
        )
        with patch.object(
            indexer,
            "get_parallel",
            return_value=SimpleNamespace(attn_tp_size=8, attn_tp_rank=rank),
        ):
            shard = indexer._get_indexer_query_shard(
                batch, plan.rows if legacy else plan.num_tokens_pad
            )
        return indexer, shard, batch, plan

    def test_tp8_indexer_gathers_when_row_plans_disagree_or_are_absent(self):
        from sglang.srt.environ import envs

        with (
            patch.object(dsa_cp, "_enable_dsa_cp", True),
            envs.SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD.override(True),
        ):
            for legacy in (False, True):
                for mismatch in ("absent", "start", "rows"):
                    with self.subTest(legacy=legacy, mismatch=mismatch):
                        indexer, shard, batch, plan = self._tp8_indexer_case(
                            legacy, 17, 2
                        )
                        plan = (
                            None
                            if mismatch == "absent"
                            else plan._replace(
                                **(
                                    {"local_start": plan.local_start + plan.rows}
                                    if mismatch == "start"
                                    else {"rows": plan.rows + 1}
                                )
                            )
                        )
                        if legacy:
                            batch.npu_dsa_cp_plan = plan
                        else:
                            batch.npu_dsa_token_shard_plan = plan
                        full = torch.arange(shard.rows * 8 * 2).reshape(-1, 2)
                        local = full[shard.start : shard.start + shard.rows].clone()
                        with patch.object(
                            indexer,
                            "attn_tp_all_gather_into_tensor",
                            side_effect=lambda out, value: out.copy_(full),
                        ):
                            result = shard.resolve(local, 17, batch)
                        torch.testing.assert_close(result, full[:17])
                        self.assertFalse(batch.npu_indexer_topk_is_local)

    def test_tp8_local_topk_clears_partial_and_empty_rank_padding(self):
        from sglang.srt.environ import envs

        with (
            patch.object(dsa_cp, "_enable_dsa_cp", True),
            envs.SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD.override(True),
        ):
            for legacy in (False, True):
                for total in (9, 17):
                    for rank in range(8):
                        with self.subTest(legacy=legacy, total=total, rank=rank):
                            indexer, shard, batch, _ = self._tp8_indexer_case(
                                legacy, total, rank
                            )
                            topk = torch.arange(shard.rows * 2).reshape(-1, 2) + 1
                            expected = topk.clone()
                            expected[shard.num_real :] = 0
                            with patch.object(
                                indexer,
                                "attn_tp_all_gather_into_tensor",
                                side_effect=AssertionError(
                                    "matching rows must stay local"
                                ),
                            ):
                                result = shard.resolve(topk, total, batch)
                            torch.testing.assert_close(result, expected)
                            self.assertTrue(batch.npu_indexer_topk_is_local)

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
            validate_legacy_dcp_flags(env, tp_size=8, dcp_size=4)
        args.real_weights, args.context_length = True, 65536
        command, _, _ = launch.launch_config(args, {})
        self.assertEqual(command[command.index("--load-format") + 1], "auto")
        self.assertEqual(command[command.index("--context-length") + 1], "65536")

    @staticmethod
    def _load_launch_module():
        path = (
            Path(__file__).resolve().parents[5]
            / "benchmark/prefill_1m/launch_bde_v2.py"
        )
        spec = importlib.util.spec_from_file_location("bde_launch", path)
        launch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launch)
        return launch

    @staticmethod
    def _launch_args():
        return SimpleNamespace(
            tp=None, dcp=None, model_path=None, real_weights=False, context_length=None
        )

    def test_archive_recipe_env_stack_at_tp16_dcp16(self):
        # Pin the archived recipe so launcher drift cannot invalidate A/B runs.
        launch = self._load_launch_module()
        command, env, selected = launch.launch_config(
            self._launch_args(), {"TP": "16", "DCP": "16"}
        )
        self.assertEqual(command[command.index("--tp-size") + 1], "16")
        self.assertEqual(command[command.index("--dcp-size") + 1], "16")
        expected = {
            "SGLANG_NPU_INDEXER_INT8": "True",
            "SGLANG_NPU_ENABLE_DSA_CP": "True",
            "SGLANG_NPU_SFA_KV_8BIT": "1",
            "SGLANG_NPU_DSA_CP_TOKEN_SHARDED": "1",
            "SGLANG_NPU_DSA_CP_Q_FULL_WIDTH": "1",
            "SGLANG_NPU_DSA_CP_LOCAL_TOPK": "1",
            "SGLANG_NPU_DCP_DECODE_Q_PREISSUE": "1",
            "SGLANG_NPU_STREAM_O_PROJ": "1",
            "SGLANG_NPU_ENABLE_DCP_PACKED_READ": "True",
            "SGLANG_NPU_ENABLE_DCP_GATHER_PREFETCH": "True",
            "SGLANG_NPU_ENABLE_DCP_SFA_GATHER_OVERLAP": "True",
            "SGLANG_ENABLE_CHAT_ENCODE_POOL": "True",
            "SGLANG_CHAT_ENCODE_MEMO": "True",
            "SGLANG_CHAT_ENCODE_RAW": "True",
            "SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP": "True",
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "SGLANG_NPU_USE_MULTI_STREAM": "1",
            "DEEPEP_HCCL_BUFFSIZE": "2500",
            "HCCL_BUFFSIZE": "1000",
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "HCCL_SOCKET_IFNAME": "lo",
            "GLOO_SOCKET_IFNAME": "lo",
            "STREAMS_PER_DEVICE": "32",
            "SGLANG_SET_CPU_AFFINITY": "1",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        }
        self.assertEqual(selected, expected)
        # The exact Phase A recipe combo must pass the restored-tree gate.
        validate_legacy_dcp_flags(selected, tp_size=16, dcp_size=16)

    def test_recipe_env_overrides_win(self):
        # Attribution runs must be able to disable one optimization at a time.
        launch = self._load_launch_module()
        inherited = {
            "TP": "8",
            "DCP": "8",
            "SGLANG_NPU_ENABLE_DCP_GATHER_PREFETCH": "0",
            "SGLANG_NPU_DSA_CP_TOKEN_SHARDED": "0",
            "SGLANG_NPU_STREAM_O_PROJ": "0",
        }
        command, env, selected = launch.launch_config(self._launch_args(), inherited)
        self.assertEqual(selected["SGLANG_NPU_ENABLE_DCP_GATHER_PREFETCH"], "0")
        self.assertEqual(selected["SGLANG_NPU_DSA_CP_TOKEN_SHARDED"], "0")
        self.assertEqual(selected["SGLANG_NPU_STREAM_O_PROJ"], "0")
        validate_legacy_dcp_flags(selected, tp_size=8, dcp_size=8)
        validate_legacy_dcp_flags(selected, tp_size=8, dcp_size=4)

    def test_decode_graph_capture_paths(self):
        # Misaligned buckets previously emptied capture_bs and killed startup.
        launch = self._load_launch_module()
        command, _, _ = launch.launch_config(self._launch_args(), {})
        self.assertEqual(
            command[command.index("--cuda-graph-backend-prefill") + 1], "disabled"
        )
        self.assertEqual(
            command[command.index("--cuda-graph-backend-decode") + 1], "disabled"
        )
        self.assertNotIn("--cuda-graph-max-bs-decode", command)
        command, _, _ = launch.launch_config(
            self._launch_args(), {"CAPTURE": "1", "GRAPH_BS": "16"}
        )
        self.assertEqual(command[command.index("--cuda-graph-max-bs-decode") + 1], "16")
        self.assertNotIn("--cuda-graph-backend-decode", command)
        command, _, _ = launch.launch_config(
            self._launch_args(),
            {"CAPTURE": "1", "GRAPH_BS": "16", "GRAPH_BS_LIST": "8 16", "TP": "8"},
        )
        bucket_at = command.index("--cuda-graph-bs-decode")
        self.assertEqual(command[bucket_at + 1 : bucket_at + 3], ["8", "16"])
        with self.assertRaisesRegex(ValueError, "multiples of TP"):
            launch.launch_config(
                self._launch_args(), {"CAPTURE": "1", "GRAPH_BS": "12", "TP": "8"}
            )


if __name__ == "__main__":
    unittest.main()
