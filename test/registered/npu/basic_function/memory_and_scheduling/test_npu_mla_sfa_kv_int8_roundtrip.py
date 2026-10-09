"""Packed 8-bit (INT8) SFA KV: the pack is the inverse of the dequant, and the
DCP owner rule still partitions the packed pool.

[Test Category] Memory
[Test Target] NPUMLATokenToKVPool packed 8-bit SFA path (SGLANG_NPU_SFA_KV_8BIT)

On A2/A3 (910B) the KvQuantSparseFlashAttention kernel accepts INT8-only for
key/value, so the DSA latent KV is stored as a 656 B packed row -- latent INT8
| rope BF16 | per-128-tile FP32 scales -- while the model computes in bf16.
Writes quantise the latent at set_kv_buffer time with the vendor's
npu_dynamic_quant(dst_type=int8); serving reads dequantise on read
(pool._gather_packed_sfa_rows). This is the A2/A3 int8 variant of the fp8 row
branch_F carries (the 950PR-only kernel); the op call and the 656 B layout are
identical, only the 512-byte nope half is int8 instead of fp8.

This pins the halves the A2/A3 run must pass:

  1. the round trip. A written latent+rope goes through the REAL
     npu_dynamic_quant(int8) pack and the fp32-scale dequant and comes back:
     the rope exactly (a straight bf16 byte reinterpretation) and the latent
     within the per-128-tile int8 rounding (|err| <= tile_max/127). Neither the
     NPU quant op nor the int8->fp32 cast can be stood in for on CPU.
  2. the owner rule. The DCP sharded write of the packed pool partitions the
     sequence the same way the bf16 pool does: each rank writes only the rows it
     owns, at the physical row its owner expects, and reading them back
     dequantises to what was written.

The flag is refused at init without DCP, because the temporary read covers only
the DCP gather paths; the pool raises rather than serve a packed row it cannot
read back, so the flag's DCP requirement is asserted here too.

One NPU, no weights, no server.
"""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=90, suite="full-1-npu-a3")

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
INDEX_HEAD_DIM = 128
PAGE_SIZE = 128
LAYER_NUM = 4
SIZE = 2048
DCP_SIZE = 4
DEVICE = "npu:0"
TILES = KV_LORA_RANK // 128  # 4

# 512 latent(int8) + 16 fp32 scales + 128 rope bytes = 656, one byte per element.
ROW_BYTES = KV_LORA_RANK + KV_LORA_RANK // 128 * 4 + QK_ROPE_HEAD_DIM * 2


class _FakeLayer:
    """set_kv_buffer / get_mla_kv_buffer read only layer_id off the layer."""

    def __init__(self, layer_id: int):
        self.layer_id = layer_id


def _packed_kwargs(**overrides):
    kwargs = dict(
        size=SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        layer_num=LAYER_NUM,
        device=DEVICE,
        enable_memory_saver=False,
        index_head_dim=INDEX_HEAD_DIM,
        start_layer=0,
        end_layer=LAYER_NUM,
        kv_cache_dim=ROW_BYTES,
    )
    kwargs.update(overrides)
    return kwargs


def _build_packed(dcp_size, **overrides):
    """A packed 8-bit SFA pool: flag on, bf16 compute, DCP > 1, 656 B row."""
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool

    with envs.SGLANG_NPU_SFA_KV_8BIT.override(True), get_parallel().override(
        attn_dcp_size=dcp_size
    ):
        return NPUMLATokenToKVPool(**_packed_kwargs(**overrides))


def _per_tile_int8_bound(ref: torch.Tensor) -> torch.Tensor:
    """|dequant - x| <= tile_max/127 for a per-128-tile dynamic int8 quant
    (rounding is <= tile_max/254; /127 leaves 2x headroom plus bf16 of the ref)."""
    r = ref.float().reshape(-1, TILES, 128)
    tile_max = r.abs().amax(dim=-1)  # [..., TILES]
    return tile_max.repeat_interleave(128, dim=-1).reshape(ref.shape) / 127.0


class TestNpuMlaSfaKvInt8RoundTrip(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)

    def tearDown(self):
        torch.npu.empty_cache()

    def test_the_row_is_656_bytes_of_int8(self):
        """The geometry the flag's capacity claim (x1.76 vs 1152 B bf16) rests on."""
        pool = _build_packed(DCP_SIZE)
        self.assertTrue(pool.sfa_kv_store_8bit)
        self.assertTrue(pool.sfa_packed_kv_storage)
        # The model still computes in bf16; only the storage is int8.
        self.assertEqual(pool.dtype, torch.bfloat16)
        self.assertEqual(pool.store_dtype, torch.int8)
        self.assertEqual(pool.kv_cache_dim, ROW_BYTES)
        # rope folds into the latent row, so the separate v dimension is gone.
        self.assertEqual(pool.kr_cache_dim, 0)
        self.assertEqual(ROW_BYTES, 656)
        # The buffer really is one-byte storage of 656-wide rows.
        rows = pool.k_buffer[0].view(-1, ROW_BYTES)
        self.assertEqual(rows.shape[-1], ROW_BYTES)

    def test_pack_is_the_inverse_of_dequant(self):
        """The NPU round trip: real npu_dynamic_quant(int8) in, fp32-scale dequant out."""
        pool = _build_packed(DCP_SIZE)
        n = 32
        g = torch.Generator().manual_seed(0)
        latent = torch.randn(n, 1, KV_LORA_RANK, generator=g).to(torch.bfloat16).to(DEVICE)
        rope = torch.randn(n, 1, QK_ROPE_HEAD_DIM, generator=g).to(torch.bfloat16).to(DEVICE)
        loc = torch.arange(n, dtype=torch.int64, device=DEVICE)

        # DCP 1 during the round trip: no sharding, so each loc maps to itself.
        with get_parallel().override(attn_dcp_size=1):
            pool.set_kv_buffer(_FakeLayer(0), loc, latent, rope)
            key_nope, key_rope = pool.get_mla_kv_buffer(_FakeLayer(0), loc)

        # rope is a straight bf16 byte reinterpretation: exact, not approximate.
        self.assertTrue(torch.equal(key_rope, rope), "the rope did not round-trip exactly")
        # int8 per-tile: absolute error bounded by the tile's dynamic range.
        err = (key_nope.float() - latent.float()).abs()
        bound = _per_tile_int8_bound(latent)
        self.assertLessEqual(
            (err - bound).max().item(),
            1e-2,
            f"latent int8 dequant exceeded per-tile bound by {(err - bound).max().item():.4f}",
        )

    def test_the_packed_write_round_trips_to_the_owning_rank_only(self):
        """The DCP owner rule on a packed pool: each rank writes only its rows, at
        the physical row its owner expects, and reading them back dequantises to
        what was written. An owner filter without its matching divide would write
        dcp_size times too many rows and this would find the misplaced data."""
        first = PAGE_SIZE * DCP_SIZE
        n = 64
        loc = torch.arange(first, first + n, dtype=torch.int64, device=DEVICE)
        g = torch.Generator().manual_seed(1)
        latent = torch.randn(n, 1, KV_LORA_RANK, generator=g).to(torch.bfloat16).to(DEVICE)
        rope = torch.randn(n, 1, QK_ROPE_HEAD_DIM, generator=g).to(torch.bfloat16).to(DEVICE)

        for rank in range(DCP_SIZE):
            pool = _build_packed(DCP_SIZE)
            with get_parallel().override(attn_dcp_size=DCP_SIZE, attn_dcp_rank=rank):
                pool.set_kv_buffer(_FakeLayer(0), loc, latent, rope)
                for i in range(n):
                    virtual = first + i
                    if virtual % DCP_SIZE != rank:
                        continue
                    physical = virtual // DCP_SIZE
                    read = torch.tensor([physical], dtype=torch.int64, device=DEVICE)
                    k_nope, k_rope = pool.get_mla_kv_buffer(_FakeLayer(0), read)
                    with self.subTest(rank=rank, virtual=virtual):
                        err = (k_nope.float() - latent[i].float()).abs()
                        bound = _per_tile_int8_bound(latent[i])
                        self.assertLessEqual((err - bound).max().item(), 1e-2)
                        self.assertTrue(torch.equal(k_rope, rope[i]))
            del pool
            torch.npu.empty_cache()

    def test_a_written_row_is_not_read_through_the_bf16_door(self):
        """get_key_buffer must not view a packed row as bf16: doing so would halve
        656 to 328 and corrupt the layout the quant op and the dequant both
        address. It hands back the raw one-byte storage instead -- the int8 store
        dtype at 656 elements, not bf16 at 328."""
        pool = _build_packed(DCP_SIZE)
        key = pool.get_key_buffer(0)
        self.assertEqual(key.dtype, torch.int8)
        self.assertEqual(key.view(-1, ROW_BYTES).shape[-1], ROW_BYTES)

    def test_dcp_size_one_refuses_the_flag(self):
        """The temporary read covers the DCP gather paths only, so a packed pool
        without DCP has no working serving read; the pool refuses at init rather
        than serve rows it cannot decode."""
        from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool

        with envs.SGLANG_NPU_SFA_KV_8BIT.override(True), get_parallel().override(
            attn_dcp_size=1
        ):
            with self.assertRaises(ValueError):
                NPUMLATokenToKVPool(**_packed_kwargs())


if __name__ == "__main__":
    unittest.main()
