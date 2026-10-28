"""Opt-in evidence emitted only after a feature executes."""

import json
import logging
import os

import torch
from sglang.srt.environ import envs

_seen = set()
logger = logging.getLogger(__name__)


def record(feature, **details):
    if torch.compiler.is_compiling() or not envs.SGLANG_NPU_FEATURE_EVIDENCE.get():
        return
    if feature in _seen:
        return
    _seen.add(feature)
    logger.warning(
        "NPU_FEATURE_EVIDENCE %s",
        json.dumps(dict(feature=feature, pid=os.getpid(), **details)),
    )


def loading_memory(layer, feature):
    if not envs.SGLANG_NPU_FEATURE_EVIDENCE.get():
        return
    record(
        feature,
        floating_bytes=sum(
            p.numel() * p.element_size()
            for p in layer.parameters()
            if p.is_floating_point() and p.device.type != "meta"
        ),
        peak_allocated=torch.npu.max_memory_allocated()
        if hasattr(torch, "npu")
        else None,
        peak_reserved=torch.npu.max_memory_reserved()
        if hasattr(torch, "npu")
        else None,
    )
