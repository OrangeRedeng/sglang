# SPDX-License-Identifier: Apache-2.0
"""Static video-only FFN row plans; all coordinates remain in the native layout."""

import torch


def selective_ffn_rows(
    *, seq_len, row_start, row_stop, video_spans, target_rows, active_fraction, device
):
    if active_fraction not in (0.5, 0.75):
        raise ValueError("Selective FFN active fraction must be 0.5 or 0.75")
    if not video_spans:
        raise ValueError("Selective FFN requires explicit video spans")
    target = torch.zeros(seq_len, dtype=torch.bool)
    target[target_rows.to(device="cpu", dtype=torch.long)] = True
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
    return tuple(
        rows[(groups < 0) | ((groups - phase) % 4 < keep)].to(device)
        for phase in range(4)
    )
