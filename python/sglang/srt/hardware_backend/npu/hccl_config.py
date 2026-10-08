"""Opt-in HCCL buffer sizing before scheduler initialization."""

import logging
import math
import os

MIB = 1 << 20
logger = logging.getLogger(__name__)


def hccl_buffer_mb(
    token_rows,
    hidden_size,
    element_size,
    headroom=1.25,
    quantum_mb=32,
    min_mb=64,
    max_mb=1024,
):
    if min(token_rows, hidden_size, element_size, quantum_mb, min_mb) <= 0:
        raise ValueError("HCCL dimensions and bounds must be positive")
    if not math.isfinite(headroom) or headroom < 1 or max_mb < min_mb:
        raise ValueError("invalid HCCL headroom/bounds")
    raw = token_rows * hidden_size * element_size * headroom / MIB
    return min(max_mb, max(min_mb, math.ceil(raw / quantum_mb) * quantum_mb))


def initialize_hccl_buffer(server_args):
    if (
        os.environ.get("SGLANG_NPU_AUTO_HCCL_BUFFSIZE", "0").lower()
        not in ("1", "true")
        or "HCCL_BUFFSIZE" in os.environ
    ):
        return
    from sglang.srt.arg_groups.model_override_base import model_config_of

    model = model_config_of(server_args)
    rows = server_args.max_prefill_tokens
    if server_args.chunked_prefill_size > 0:
        rows = min(rows, server_args.chunked_prefill_size)
    element_size = 4 if str(model.dtype) in ("float32", "torch.float32") else 2
    hidden_size = model.hf_text_config.hidden_size
    buffer_mb = hccl_buffer_mb(
        rows,
        hidden_size,
        element_size,
        headroom=float(os.environ.get("SGLANG_NPU_HCCL_HEADROOM", "1.25")),
        quantum_mb=int(os.environ.get("SGLANG_NPU_HCCL_QUANTUM_MB", "32")),
        min_mb=int(os.environ.get("SGLANG_NPU_HCCL_MIN_MB", "64")),
        max_mb=int(os.environ.get("SGLANG_NPU_HCCL_MAX_MB", "1024")),
    )
    os.environ.setdefault("HCCL_BUFFSIZE", str(buffer_mb))
    logger.info(
        "NPU HCCL_BUFFSIZE=%d MiB: tokens=%d hidden=%d element_bytes=%d",
        buffer_mb,
        rows,
        hidden_size,
        element_size,
    )
