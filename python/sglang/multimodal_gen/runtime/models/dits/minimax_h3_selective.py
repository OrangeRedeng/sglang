# SPDX-License-Identifier: Apache-2.0
"""Static video-only FFN row plans; all coordinates remain in the native layout."""

import os

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def selective_ffn_rows(
    *, seq_len, row_start, row_stop, video_spans, target_rows, active_fraction, device
):
    if active_fraction not in (0.5, 0.75):
        raise ValueError("Selective FFN active fraction must be 0.5 or 0.75")
    if not video_spans:
        raise ValueError("Selective FFN requires explicit video spans")
    if not 0 <= row_start <= row_stop <= seq_len:
        raise ValueError("Invalid selective FFN rank-local row bounds")
    if target_rows.ndim != 1 or target_rows.dtype not in (torch.int32, torch.int64):
        raise ValueError("Selective FFN target rows must be integer indices")
    indices = target_rows.to(device="cpu", dtype=torch.long)
    if indices.numel() and (indices.min() < 0 or indices.max() >= seq_len):
        raise ValueError("Selective FFN target row outside packed sequence")
    target = torch.zeros(seq_len, dtype=torch.bool)
    target[indices] = True
    groups = torch.full((seq_len,), -1, dtype=torch.long)
    previous_stop = 0
    for span in video_spans:
        start = span["start"]
        frames, height, width = span["latent_shape"]
        stop = start + frames * height * width
        if min(frames, height, width) <= 0 or start < previous_stop or stop > seq_len:
            raise ValueError("Invalid ordered video spans for selective FFN")
        previous_stop = stop
        # Native boundary frames and every non-target row always receive updates.
        for frame in range(1, frames - 1):
            lo = start + frame * height * width
            hi = lo + height * width
            groups[lo:hi] = torch.where(target[lo:hi], frame % 4, -1)
    groups = groups[row_start:row_stop]
    rows = torch.arange(row_stop - row_start)
    keep = int(active_fraction * 4)
    plans = tuple(
        rows[(groups < 0) | ((groups - phase) % 4 < keep)].to(device)
        for phase in range(4)
    )
    if os.getenv("SGLANG_H3_SELECTIVE_FFN_LOG") == "1":
        # F0 diagnostic: the actual compacted GEMM row counts behind any
        # speed claim (per-phase counts differ; protected layers excluded
        # upstream by the eligible-layer band).
        counts = [int(plan.numel()) for plan in plans]
        total_rows = row_stop - row_start
        logger.info(
            "Selective FFN rows: fraction=%s seq=%d local_rows=%d "
            "per_phase=%s compacted=%.3f",
            active_fraction,
            seq_len,
            total_rows,
            counts,
            sum(counts) / (4 * max(total_rows, 1)),
        )
    return plans
