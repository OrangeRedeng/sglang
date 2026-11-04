"""Bounded DCP runtime storage avoids allocation at asynchronous collectives."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.hardware_backend.npu.attention import dcp_gather as gather
from sglang.srt.hardware_backend.npu.modules import dsa_token_residual as residual
from sglang.srt.layers.dcp import layout
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class Group:
    world_size = 2
    ranks = [0, 1]
    rank_in_group = 0

    def all_gather_into_tensor_async(self, output, send):
        return SimpleNamespace(wait=lambda: output.copy_(torch.cat((send, send + 10))))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_preallocated_projection_slots_keep_column_order_without_allocating(dtype):
    group = Group()
    shard = torch.arange(12, dtype=dtype).reshape(4, 3)
    with patch.dict(gather._o_proj_gather_buffers, {}, clear=True):
        gather.prepare_o_proj_gather(group, shard)
        pointers = {}
        for step, slot in enumerate((0, 1, 0, 1)):
            value = shard + step
            with patch.object(
                torch.Tensor, "new_empty", side_effect=AssertionError("hot allocation")
            ):
                output, send, work = gather.start_o_proj_gather(
                    group, value, buffer_slot=slot
                )
                work.wait()
            torch.testing.assert_close(
                output.t(), torch.cat((value, value + 10), dim=1)
            )
            actual = (output.data_ptr(), send.data_ptr())
            assert pointers.setdefault(slot, actual) == actual
        assert pointers[0] != pointers[1]


def test_missing_projection_reservation_is_explicit():
    with patch.dict(gather._o_proj_gather_buffers, {}, clear=True):
        with pytest.raises(RuntimeError, match="prepared before serving"):
            gather.start_o_proj_gather(Group(), torch.ones(4, 3), buffer_slot=0)


def test_prefix_send_buffers_reuse_storage_and_clear_old_padding():
    group = Group()
    refs = (torch.empty(0, 1, 1), torch.empty(0, 1, 1))
    pointers = None
    with patch.dict(layout._dcp_extend_gather_buffers, {}, clear=True):
        for prefix, local in ((5, 4), (4, 4), (1, 1)):
            plan = layout.plan_dcp_packed_read(prefix, 2, 2, 4)
            send = torch.arange(local, dtype=torch.float32).reshape(-1, 1, 1) + 3
            if pointers is None:
                pending = gather.start_prefix_gather(
                    group, (send, send), refs, plan, 2, True, buffer_slot=0
                )
            else:
                with patch.object(
                    torch, "empty", side_effect=AssertionError("hot allocation")
                ):
                    pending = gather.start_prefix_gather(
                        group, (send, send), refs, plan, 2, True, buffer_slot=0
                    )
            for work in pending.works:
                work.wait()
            assert torch.count_nonzero(pending.sends[0][local:]) == 0
            actual = tuple(x.data_ptr() for x in (*pending.outputs, *pending.sends))
            if pointers is not None:
                assert actual == pointers
            pointers = actual


def test_context_and_ragged_fallback_are_reserved_alongside_chunk_storage():
    layer = SimpleNamespace(
        w_vc_qrep=torch.empty(1, 1, 1, dtype=torch.bfloat16),
        o_proj=SimpleNamespace(output_size=6144),
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    model = SimpleNamespace(modules=lambda: [layer])
    with (
        patch.object(
            residual, "get_parallel", return_value=SimpleNamespace(attn_tp_size=16)
        ),
        residual.envs.SGLANG_NPU_ENABLE_DCP_GATHER_PREFETCH.override(True),
        residual.envs.SGLANG_NPU_ENABLE_DCP_PACKED_READ.override(True),
    ):
        reserve = residual.runtime_gather_reserve_bytes(
            model, 32768, context_tokens=1000000, prefill_requests=2
        )
        assert reserve >= 4 * 2 * 1000000 * (512 + 64) * 2
        captured = residual.runtime_gather_reserve_bytes(
            model,
            32768,
            context_tokens=1000000,
            prefill_requests=2,
            capture_tokens=(1024, 4096, 16384, 32768),
        )
        assert captured > reserve


def test_kv_profile_adds_gather_reserve_to_runtime_headroom():
    from sglang.srt.mem_cache import kv_cache_configurator as kcc

    cfg = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
    cfg.device, cfg.gpu_id, cfg.model, cfg.mambaish_config = "npu", 0, object(), None
    cfg.model_config = SimpleNamespace(context_len=1000000, is_multimodal=False)
    schedule = SimpleNamespace(
        mem_fraction_static=0.8,
        chunked_prefill_size=32768,
        max_prefill_tokens=32768,
        prefill_max_requests=2,
        max_running_requests=2,
        max_mamba_cache_size=None,
    )
    prefill = SimpleNamespace(
        backend=kcc.Backend.DISABLED, bs=None, full_prefill_max_req=None
    )
    with (
        patch.object(kcc, "get_available_gpu_memory", return_value=28),
        patch.object(
            kcc,
            "get_parallel",
            return_value=SimpleNamespace(
                launch_world_size=1, world_group=SimpleNamespace(cpu_group=None)
            ),
        ),
        patch.object(kcc, "get_schedule", return_value=schedule),
        patch.object(
            kcc,
            "get_exec",
            return_value=SimpleNamespace(
                graph=SimpleNamespace(
                    cuda_graph_config=SimpleNamespace(prefill=prefill)
                )
            ),
        ),
        patch.object(
            kcc, "get_disagg", return_value=SimpleNamespace(disaggregation_mode=None)
        ),
        patch.object(
            kcc, "get_mm", return_value=SimpleNamespace(mm_feature_transport="cpu")
        ),
        patch.object(kcc, "mm_runtime_reservation_gb", return_value=0),
        patch.object(
            residual, "runtime_gather_reserve_bytes", return_value=5 * (1 << 30)
        ),
        residual.envs.SGLANG_NPU_DSA_CP_TOKEN_SHARDED.override(True),
    ):
        assert abs(cfg._profile_available_bytes(60) - 11 * (1 << 30)) <= 1
        with patch.object(kcc, "get_available_gpu_memory", return_value=2):
            with pytest.raises(
                ValueError, match="Runtime reserves leave no KV capacity"
            ):
                cfg._profile_available_bytes(60)
