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
    if feature == "model-loaded":
        record_weight_layouts(layer)
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


def record_weight_layouts(model):
    """Report physical matrix layouts without reading device tensor contents."""
    import torch_npu

    groups = {}
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    for name, tensor in tensors:
        matrix = "weight" in name or name.rsplit(".", 1)[-1].startswith(
            ("w_kc", "w_vc")
        )
        if tensor.device.type != "npu" or tensor.ndim < 2 or not matrix:
            continue
        role = ".".join(name.split(".")[-2:])
        fmt = int(torch_npu.get_npu_format(tensor))
        key = (role, str(tensor.dtype), fmt)
        entry = groups.setdefault(
            key,
            dict(
                role=role,
                dtype=str(tensor.dtype),
                format=fmt,
                count=0,
                example=name,
                shape=list(tensor.shape),
            ),
        )
        entry["count"] += 1
    record("weight-layout-audit", matrices=list(groups.values()))
