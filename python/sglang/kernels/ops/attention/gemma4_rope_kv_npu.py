# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Experimental NeoX RoPE and ND KV write for Gemma 4 on Ascend."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rotate_rows(
    X,
    CACHE,
    POS,
    token,
    head,
    valid_token,
    SX: tl.constexpr,
    SC: tl.constexpr,
    SP: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    cols = tl.arange(0, BLOCK)
    valid = valid_token[:, None] & (cols[None, :] < D)
    base = token[:, None] * SX + head[:, None] * D
    mate = tl.where(cols < R // 2, cols + R // 2, cols - R // 2)
    mate = tl.where(cols < R, mate, cols)
    x = tl.load(X + base + cols[None, :], valid, 0).to(tl.float32)
    y = tl.load(X + base + mate[None, :], valid, 0).to(tl.float32)
    pos = tl.load(POS + token * SP, valid_token, 0)
    angle = cols % (R // 2)
    cos = tl.load(CACHE + pos[:, None] * SC + angle[None, :], valid, 1).to(tl.float32)
    sin = tl.load(CACHE + pos[:, None] * SC + angle[None, :] + R // 2, valid, 0).to(
        tl.float32
    )
    rotated = x * cos + tl.where(cols[None, :] < R // 2, -y, y) * sin
    return tl.where(cols[None, :] < R, rotated, x)


@triton.jit
def _rope_kv_kernel(
    Q,
    K,
    V,
    CACHE,
    POS,
    LOC,
    QO,
    KO,
    KC,
    VC,
    MAP,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    SV: tl.constexpr,
    SC: tl.constexpr,
    SP: tl.constexpr,
    SL: tl.constexpr,
    M,
    CAPACITY: tl.constexpr,
    QH: tl.constexpr,
    KH: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    MAPPED: tl.constexpr,
    MAP_SIZE: tl.constexpr,
):
    q_tiles = tl.cdiv(M * QH, ROWS)
    k_tiles = tl.cdiv(M * KH, ROWS)
    cols = tl.arange(0, BLOCK)
    for tile in range(tl.program_id(0), q_tiles + k_tiles, tl.num_programs(0)):
        # Keep Q-only programs out of the cache-scatter address space entirely.
        if tile < q_tiles:
            rows = tile * ROWS + tl.arange(0, ROWS)
            token, head = rows // QH, rows % QH
            valid = (token[:, None] < M) & (cols[None, :] < D)
            out = _rotate_rows(
                Q, CACHE, POS, token, head, token < M, SQ, SC, SP, D, R, BLOCK
            )
            tl.store(
                QO + (token[:, None] * QH + head[:, None]) * D + cols[None, :],
                out,
                valid,
            )
        else:
            rows = (tile - q_tiles) * ROWS + tl.arange(0, ROWS)
            token, head = rows // KH, rows % KH
            valid = (token[:, None] < M) & (cols[None, :] < D)
            out = _rotate_rows(
                K, CACHE, POS, token, head, token < M, SK, SC, SP, D, R, BLOCK
            )
            tl.store(
                KO + (token[:, None] * KH + head[:, None]) * D + cols[None, :],
                out,
                valid,
            )
            slot = tl.load(LOC + token * SL, token < M, 0)
            live_slot = slot > 0
            if MAPPED:
                slot = tl.load(
                    MAP + slot, (token < M) & (slot >= 0) & (slot < MAP_SIZE), 0
                )
            write = (
                valid
                & live_slot[:, None]
                & (slot[:, None] > 0)
                & (slot[:, None] < CAPACITY)
            )
            dest = (slot[:, None].to(tl.int64) * KH + head[:, None]) * D + cols[None, :]
            value = tl.load(
                V + token[:, None] * SV + head[:, None] * D + cols[None, :], valid, 0
            )
            tl.store(KC + dest, out, write)
            tl.store(VC + dest, value, write)


def gemma4_rope_kv_npu(
    q,
    k,
    v,
    cache,
    positions,
    locations,
    key_cache,
    value_cache,
    num_q_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    slot_mapping=None,
):
    """Return rotated Q/K and write normalized V plus rotated K into ND slots.

    Inputs are immutable, including aliased QKV views. Locations must be unique
    positive physical slots; zero and negative padding slots are skipped. Gemma's
    proportional RoPE is represented by identity columns in its angle cache.
    """
    if q.device.type not in ("npu", "privateuseone"):
        raise ValueError("Gemma4 RoPE/KV candidate requires an NPU")
    _check_layout(
        q,
        k,
        v,
        cache,
        positions,
        locations,
        key_cache,
        value_cache,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
    )
    if slot_mapping is not None and (
        slot_mapping.ndim != 1
        or slot_mapping.stride(0) != 1
        or slot_mapping.dtype not in (torch.int32, torch.int64)
        or slot_mapping.device != q.device
    ):
        raise ValueError("SWA mapping must be a contiguous integer vector")
    qo = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    ko = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    if q.shape[0]:
        from sgl_kernel_npu.utils.triton_utils import get_device_properties

        _, cores = get_device_properties()
        _launch(
            q,
            k,
            v,
            cache,
            positions,
            locations,
            qo,
            ko,
            key_cache,
            value_cache,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            cores,
            slot_mapping,
        )
    return qo, ko


def _check_layout(q, k, v, cache, positions, locations, kc, vc, qh, kh, d, r):
    if d not in (128, 256, 512) or r < 2 or r > d or r % 2:
        raise ValueError("Unsupported Gemma4 RoPE head or rotary dimension")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Gemma4 RoPE/KV requires FP16 or BF16")
    for tensor, width in ((q, qh * d), (k, kh * d), (v, kh * d)):
        if (
            tensor.shape != (q.shape[0], width)
            or tensor.stride(-1) != 1
            or tensor.dtype != q.dtype
            or tensor.device != q.device
        ):
            raise ValueError("Unsupported strided QKV layout")
    for tensor in (kc, vc):
        if (
            tensor.ndim != 4
            or tensor.shape[1:] != (1, kh, d)
            or not tensor.is_contiguous()
            or tensor.dtype != q.dtype
            or tensor.device != q.device
        ):
            raise ValueError("KV buffers must be matching ND [slot, 1, head, dim]")
    if kc.shape != vc.shape:
        raise ValueError("K/V cache capacity differs")
    if (
        cache.ndim != 2
        or cache.shape[1] != r
        or cache.stride(-1) != 1
        or cache.dtype not in (q.dtype, torch.float32)
        or cache.device != q.device
    ):
        raise ValueError("Unsupported rotary cache")
    for tensor in (positions, locations):
        if (
            tensor.shape != (q.shape[0],)
            or tensor.device != q.device
            or tensor.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "Positions and physical slots must have one integer per token"
            )


def _launch(
    q,
    k,
    v,
    cache,
    positions,
    locations,
    qo,
    ko,
    kc,
    vc,
    qh,
    kh,
    d,
    r,
    cores,
    slot_mapping=None,
):
    rows = 4
    programs = min(
        cores, triton.cdiv(q.shape[0] * qh, rows) + triton.cdiv(q.shape[0] * kh, rows)
    )
    _rope_kv_kernel[(programs,)](
        q,
        k,
        v,
        cache,
        positions,
        locations,
        qo,
        ko,
        kc,
        vc,
        locations if slot_mapping is None else slot_mapping,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        cache.stride(0),
        positions.stride(0),
        locations.stride(0),
        q.shape[0],
        kc.shape[0],
        qh,
        kh,
        d,
        r,
        rows,
        triton.next_power_of_2(d),
        slot_mapping is not None,
        0 if slot_mapping is None else slot_mapping.numel(),
        enable_fp_fusion=False,
    )
