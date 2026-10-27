import torch
import triton
import triton.language as tl


@triton.jit
def _pack_finalize_routing(
    mapping,
    weights,
    rows,
    logits,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    destination = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = destination < N
    source = tl.load(mapping + destination, mask, other=0)
    probability = tl.load(weights + source, mask, other=0).to(tl.float32)
    tl.store(rows + destination, source // K, mask)
    tl.store(logits + destination, probability, mask)


def pack_finalize_routing(expanded_row_idx, topk_weights):
    # row_idx_type=1 maps sorted routes back to token-major source rows.
    n = topk_weights.numel()
    rows = torch.empty(n, dtype=torch.int64, device=topk_weights.device)
    logits = torch.empty(n, dtype=torch.float32, device=topk_weights.device)
    if n:
        _pack_finalize_routing[(triton.cdiv(n, 256),)](
            expanded_row_idx, topk_weights, rows, logits, n, topk_weights.shape[1], 256
        )
    return rows, logits


@triton.jit
def _append_shared_topk(
    weights,
    ids,
    out_weights,
    out_ids,
    T: tl.constexpr,
    K: tl.constexpr,
    EXPERT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < T * (K + 1)
    token = offsets // (K + 1)
    route = offsets % (K + 1)
    routed_mask = mask & (route < K)
    probability = tl.load(weights + token * K + route, routed_mask, other=1)
    expert = tl.load(ids + token * K + route, routed_mask, other=EXPERT)
    tl.store(out_weights + offsets, probability, mask)
    tl.store(out_ids + offsets, expert, mask)


def append_shared_topk(weights, ids, expert):
    t, k = weights.shape
    out_weights = torch.empty((t, k + 1), dtype=weights.dtype, device=weights.device)
    out_ids = torch.empty((t, k + 1), dtype=ids.dtype, device=ids.device)
    if t:
        _append_shared_topk[(triton.cdiv(t * (k + 1), 256),)](
            weights, ids, out_weights, out_ids, t, k, expert, 256
        )
    return out_weights, out_ids


@triton.jit
def _add_rmsnorm_mxfp8(
    x,
    residual,
    weight,
    bias,
    normalized,
    residual_out,
    quantized,
    scales,
    H: tl.constexpr,
    EPS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    values = tl.load(x + row * H + cols, cols < H, other=0).to(tl.float32)
    if HAS_RESIDUAL:
        values += tl.load(residual + row * H + cols, cols < H, other=0).to(tl.float32)
        tl.store(residual_out + row * H + cols, values, cols < H)
    variance = tl.sum(values * values, 0) / H
    gamma = tl.load(weight + cols, cols < H, other=0).to(tl.float32)
    values = values * tl.rsqrt(variance + EPS) * gamma
    if HAS_BIAS:
        if not HAS_RESIDUAL:
            # Match ModelSlim's standalone RMSNorm rounding before its bias add.
            values = values.to(tl.bfloat16).to(tl.float32)
        values += tl.load(bias + cols, cols < H, other=0).to(tl.float32)
    rounded = values.to(tl.bfloat16)
    tl.store(normalized + row * H + cols, rounded, cols < H)
    blocks = tl.reshape(
        tl.where(cols < H, rounded.to(tl.float32), 0), (BLOCK // 32, 32)
    )
    amax = tl.max(tl.abs(blocks), 1)
    bits = amax.to(tl.int32, bitcast=True)
    exponent = tl.maximum(((bits >> 23) & 255) - 8, 0)
    exponent = tl.minimum(exponent, 254)
    exponent = tl.where(amax == 0, 0, exponent)
    nonfinite = (tl.sum((blocks != blocks).to(tl.int32), 1) > 0) | (
        amax == float("inf")
    )
    exponent = tl.where(nonfinite, 255, exponent)
    inverse = tl.exp2((127 - exponent).to(tl.float32))
    payload = tl.minimum(tl.maximum(blocks * inverse[:, None], -448.0), 448.0)
    payload = tl.where(nonfinite[:, None], float("nan"), payload)
    payload = tl.reshape(payload, (BLOCK,)).to(
        quantized.dtype.element_ty, fp_downcast_rounding="rtne"
    )
    tl.store(quantized + row * H + cols, payload, cols < H)
    scale_cols = tl.arange(0, BLOCK // 32)
    tl.store(scales + row * (H // 32) + scale_cols, exponent, scale_cols < H // 32)


def add_rmsnorm_mxfp8(x, residual, weight, bias, eps):
    if x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape[1] != 6144:
        raise ValueError("Dual MXFP8 norm requires BF16 [tokens, 6144] inputs")
    if not x.is_contiguous() or (residual is not None and not residual.is_contiguous()):
        raise ValueError("Dual MXFP8 norm requires contiguous inputs")
    if residual is not None and (
        residual.shape != x.shape
        or residual.dtype != x.dtype
        or residual.device != x.device
    ):
        raise ValueError("Dual MXFP8 norm requires a matching BF16 residual")
    if any(
        tensor is not None
        and (
            tensor.shape != (x.shape[1],)
            or not tensor.is_contiguous()
            or tensor.device != x.device
        )
        for tensor in (weight, bias)
    ):
        raise ValueError("Dual MXFP8 norm requires contiguous hidden-size weight/bias")
    normalized = torch.empty_like(x)
    residual_out = x if residual is None else torch.empty_like(x)
    quantized = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (x.shape[0], x.shape[1] // 64, 2), dtype=torch.uint8, device=x.device
    )
    if x.shape[0]:
        _add_rmsnorm_mxfp8[(x.shape[0],)](
            x,
            residual,
            weight,
            bias,
            normalized,
            residual_out,
            quantized,
            scales,
            x.shape[1],
            eps,
            residual is not None,
            bias is not None,
            triton.next_power_of_2(x.shape[1]),
        )
    normalized._npu_mxfp8_operand = (quantized, scales.view(torch.float8_e8m0fnu))
    return normalized, residual_out
