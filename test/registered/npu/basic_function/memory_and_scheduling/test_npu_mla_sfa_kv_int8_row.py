"""CPU roundtrip for the int8 SFA packed row (656 B) used by
SGLANG_NPU_SFA_KV_8BIT on A2/A3 (KvQuantSparseFlashAttention int8 path).

The row is [latent INT8 (kv_lora_rank) | rope BF16 (qk_rope_head_dim) |
per-128-tile FP32 scales (kv_lora_rank // 128)] = 656 B for a 512/64 MLA model.
npu_dynamic_quant is NPU-only, so the per-128-tile int8 quant is simulated here;
the LAYOUT and the dequant math (the exact view/scale ops in
memory_pool_npu._gather_packed_sfa_rows) are CPU-exact and are what this checks.
"""
import torch

KV_LORA_RANK = 512
ROPE = 64
TILES = KV_LORA_RANK // 128  # 4


def _fake_npu_dynamic_quant(latent_bf16: torch.Tensor):
    """Simulate torch_npu.npu_dynamic_quant(x.reshape(-1,128), dst_type=int8)."""
    x = latent_bf16.reshape(-1, 128).float()
    scale = x.abs().amax(dim=-1, keepdim=True) / 127.0  # per-128-tile fp32 scale
    code = (x / scale).round().clamp(-127, 127).to(torch.int8)
    n = latent_bf16.shape[0]
    return code.reshape(latent_bf16.shape), scale.reshape(n, TILES)


def _pack(latent_bf16, rope_bf16):
    code, scale = _fake_npu_dynamic_quant(latent_bf16)  # int8 [N,512], f32 [N,4]
    row = torch.cat(
        [
            code.view(torch.uint8),                       # [N,512]
            rope_bf16.contiguous().view(torch.uint8),     # [N,128]
            scale.float().contiguous().view(torch.uint8), # [N,16]
        ],
        dim=-1,
    )
    return row.view(torch.int8), code, scale  # stored as int8 [N,656]


def _dequant(rows_int8, n):
    # exact replica of memory_pool_npu._gather_packed_sfa_rows
    rope_bytes = ROPE * 2
    scale = (
        rows_int8[:, KV_LORA_RANK + rope_bytes:]
        .view(torch.uint8)
        .view(torch.float32)
        .reshape(n, TILES, 1)
    )
    nope = (
        rows_int8[:, :KV_LORA_RANK]
        .to(torch.float32)
        .reshape(n, TILES, 128)
        * scale
    ).reshape(n, KV_LORA_RANK)
    rope = (
        rows_int8[:, KV_LORA_RANK:KV_LORA_RANK + rope_bytes]
        .view(torch.uint8)
        .view(torch.bfloat16)
    )
    return nope, rope


def test_int8_sfa_row_roundtrip():
    torch.manual_seed(0)
    for n in (1, 3, 17):
        latent = torch.randn(n, KV_LORA_RANK, dtype=torch.bfloat16) * 8.0
        rope = torch.randn(n, ROPE, dtype=torch.bfloat16) * 0.5
        row, code, scale = _pack(latent, rope)

        assert row.shape == (n, 656), f"row width {row.shape} != 656"
        assert row.dtype == torch.int8
        assert row.nbytes == n * 656

        nope_d, rope_d = _dequant(row, n)
        assert rope_d.dtype == torch.bfloat16 and rope_d.shape == (n, ROPE)
        assert torch.equal(rope_d, rope), "rope must roundtrip exactly"

        # int8 per-tile error bound: |err| <= scale/2 per element
        err = (nope_d - latent.float()).abs()
        bound = scale.unsqueeze(-1).expand(n, TILES, 128).reshape(n, KV_LORA_RANK) / 2.0
        assert (err <= bound + 1e-6).all(), "dequant exceeded int8 per-tile error bound"


if __name__ == "__main__":
    test_int8_sfa_row_roundtrip()
    print("int8 SFA row roundtrip OK")
