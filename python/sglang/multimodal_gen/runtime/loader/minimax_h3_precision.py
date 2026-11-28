# SPDX-License-Identifier: Apache-2.0
"""Record the effective rank-local precision of an H3 research transformer."""

import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from sglang.multimodal_gen.runtime.distributed import get_world_rank
from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.mxfp8 import (
    NPUOnlineMXFP8DiffusionLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.w8a8_int_npu import (
    NPUOnlineW8A8DiffusionLinearMethod,
)
from sglang.multimodal_gen.runtime.utils.perf_logger import capture_memory_snapshot
from sglang.srt.hardware_backend.npu.utils import NPUACLFormat


@torch.no_grad()
def record_precision_map(model, directory, load_started, before):
    import torch_npu

    requested = os.environ["SGLANG_H3_QUANT_METHOD"]
    selected_path = Path(directory) / "quantization-selection.json"
    selected = json.loads(selected_path.read_text()).get("selected")
    if selected is not None:
        selected = set(selected)
    formats = {
        int(NPUACLFormat.ACL_FORMAT_ND): "ND",
        int(NPUACLFormat.ACL_FORMAT_FRACTAL_NZ): "FRACTAL_NZ",
    }
    layers = {}
    probes = {}
    after_load = capture_memory_snapshot().to_dict()
    load_wall_s = time.perf_counter() - load_started
    replay = {}
    replay_root = os.getenv("SGLANG_H3_QUANT_REPLAY")
    rank = get_world_rank()
    if replay_root:
        from safetensors import safe_open

        with safe_open(
            Path(replay_root) / f"rank-{rank}.safetensors", framework="pt", device="cpu"
        ) as handle:
            metadata = handle.metadata() or {}
            for key in ("CHECKPOINT", "PLACEMENT", "SOURCE"):
                if metadata.get(key) != os.environ[f"SGLANG_H3_QUANT_{key}"]:
                    raise ValueError(
                        f"BF16 replay {key} differs from the active R02 contract"
                    )
            if (
                metadata.get("rank") != str(rank)
                or not metadata.get("step", "").isdigit()
            ):
                raise ValueError("BF16 replay is missing rank or step identity")
            replay = {key: handle.get_tensor(key) for key in handle.keys()}
    protected = set()
    for path, layer in model.named_modules():
        if not isinstance(layer, LinearBase):
            continue
        prefix = layer.prefix
        method = layer.quant_method
        weight = layer.weight
        precision = str(weight.dtype).removeprefix("torch.")
        scale = None
        operators = []
        if isinstance(method, NPUOnlineW8A8DiffusionLinearMethod):
            precision, scale = "w8a8_int", layer.weight_scale
            operators = ["npu_dynamic_quant", "npu_quant_matmul"]
        elif isinstance(method, NPUOnlineMXFP8DiffusionLinearMethod):
            precision, scale = "mxfp8", layer.weight_scale_inv
            operators = ["npu_dynamic_mx_quant", "npu_quant_matmul"]
        elif not isinstance(method, UnquantizedLinearMethod):
            raise RuntimeError(
                f"Unexpected quantization method at {path}: {type(method).__name__}"
            )
        if scale is None and weight.dtype not in (
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ):
            raise RuntimeError(
                f"Expected an unquantized floating-point weight at {path}"
            )
        if scale is not None and precision != requested:
            raise RuntimeError(f"Unexpected precision {precision} at {path}")
        protected_parts = set(prefix.split(".")) & {
            "token_refiner",
            "final_layer",
            "condition_proj",
            "adaln_proj",
        }
        if protected_parts:
            protected.update(protected_parts)
            # Some output projections are explicitly FP32 in the native H3 architecture.
            if (
                scale is not None
                or weight.dtype != layer.params_dtype
                or weight.dtype not in (torch.bfloat16, torch.float32)
            ):
                raise RuntimeError(f"Protected projection precision changed at {path}")
        if precision == "w8a8_int" and (
            weight.dtype != torch.int8 or tuple(scale.shape) != (weight.shape[1],)
        ):
            raise RuntimeError(f"Invalid TP-local W8A8 scale layout at {path}")

        layout = None
        statistics = {}
        if weight.device.type == "npu":
            format_id = torch_npu.get_npu_format(weight)
            layout = formats.get(format_id, str(format_id))
        if scale is not None:
            if layout not in formats.values():
                raise RuntimeError(f"Unexpected weight format {layout} at {path}")
            # Row-parallel bias is added after the TP reduction.
            bias = (
                layer.bias
                if not (layer.skip_bias_add or isinstance(layer, RowParallelLinear))
                else None
            )
            key = (
                precision,
                tuple(weight.shape),
                tuple(weight.stride()),
                layout,
                str(layer.params_dtype),
                bias is not None,
            )
            if key not in probes:
                x = torch.full(
                    (16, weight.shape[0]),
                    0.125,
                    device=weight.device,
                    dtype=layer.params_dtype,
                )
                out = method.apply(layer, x, bias)
                if (
                    out.shape != (16, weight.shape[1])
                    or out.dtype != x.dtype
                    or not torch.isfinite(out).all().item()
                ):
                    raise RuntimeError(
                        f"Ascend quantized matmul smoke failed at {path}"
                    )
                probes[key] = prefix
                del x, out

            for group in ("video", "audio", "text", "joint"):
                x = replay.pop(f"{prefix}.{group}.input", None)
                ref = replay.pop(f"{prefix}.{group}.output", None)
                if x is None and ref is None:
                    continue
                if (
                    x is None
                    or ref is None
                    or x.ndim != 2
                    or not 0 < x.shape[0] <= 256
                    or x.shape[1] != weight.shape[0]
                    or x.dtype != layer.params_dtype
                    or ref.shape != (x.shape[0], weight.shape[1])
                    or ref.dtype != x.dtype
                ):
                    raise ValueError(
                        f"Invalid rank-local replay tensors at {prefix}.{group}"
                    )
                pred = (
                    method.apply(layer, x.to(weight.device), bias)
                    .float()
                    .cpu()
                    .flatten()
                )
                ref = ref.float().flatten()
                if not torch.isfinite(pred).all() or not torch.isfinite(ref).all():
                    raise ValueError(f"Nonfinite replay tensors at {prefix}.{group}")
                statistics[group] = {
                    "step": int(metadata["step"]),
                    "rank": rank,
                    "modality": group,
                    "rows": x.shape[0],
                    "normalized_l2": (
                        (pred - ref).norm() / ref.norm().clamp_min(1e-12)
                    ).item(),
                    "cosine": torch.nn.functional.cosine_similarity(
                        pred, ref, dim=0
                    ).item(),
                    "max_abs": (pred - ref).abs().max().item(),
                }

        if prefix in layers:
            raise RuntimeError(f"Duplicate precision prefix {prefix} on rank {rank}")
        layers[prefix] = {
            "module": path,
            "precision": precision,
            "method": type(method).__name__,
            "operators": operators,
            "weight_shape": list(weight.shape),
            "weight_dtype": str(weight.dtype),
            "parameter_input_dtype": str(layer.params_dtype),
            "weight_format": layout,
            "weight_bytes": weight.numel() * weight.element_size(),
            "scale_dtype": str(scale.dtype) if scale is not None else None,
            "scale_shape": list(scale.shape) if scale is not None else None,
            "scale_bytes": scale.numel() * scale.element_size()
            if scale is not None
            else 0,
            "replay": statistics,
        }

    actual = {prefix for prefix, row in layers.items() if row["operators"]}
    if (
        not layers
        or (requested != "bf16") != bool(actual)
        or (selected is not None and actual != selected)
    ):
        raise RuntimeError(
            f"Resolved precision differs from the requested layer selection: {sorted(actual ^ (selected or actual))}"
        )
    required = {"condition_proj", "token_refiner", "final_layer"}
    if any("adaln_proj" in name.split(".") for name, _ in model.named_modules()):
        required.add("adaln_proj")
    missing = required - protected
    for index in range(len(model.blocks)):
        for projection in ("mlp.fc1", "mlp.fc2", "attn.qkv_proj", "attn.out_proj"):
            prefix = f"blocks.{index}.{projection}"
            if prefix not in layers:
                missing.add(prefix)
    if missing or not len(model.blocks):
        raise RuntimeError(
            f"Incomplete protected or block precision inventory: {sorted(missing)}"
        )
    if replay:
        raise ValueError(
            f"Replay tensors reference unselected or unknown linears: {sorted(replay)}"
        )
    artifact = {
        "rank": rank,
        "source_sha": os.environ["SGLANG_H3_QUANT_SOURCE"],
        "checkpoint_manifest_sha256": os.environ["SGLANG_H3_QUANT_CHECKPOINT"],
        "placement": os.environ["SGLANG_H3_QUANT_PLACEMENT"],
        "method": requested,
        "load_wall_s": load_wall_s,
        "allocator_before_load": before,
        "allocator_after_load": after_load,
        "allocator_after_probe": capture_memory_snapshot().to_dict(),
        "weight_bytes": sum(row["weight_bytes"] for row in layers.values()),
        "scale_bytes": sum(row["scale_bytes"] for row in layers.values()),
        "probed_layers": list(probes.values()),
        "layers": layers,
    }
    target = Path(directory) / f"precision-rank-{rank}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(artifact, sort_keys=True, indent=2) + "\n")
    temporary.replace(target)


@contextmanager
def capture_precision_replay(model, call_kwargs, destination, *, request_id, step):
    """Capture actual TP-local projection input/output before collectives."""
    from safetensors.torch import save_file

    root = Path(os.environ["SGLANG_H3_QUANT_AUDIT"])
    target = root / "replay" / f"step-{step}" / f"rank-{get_world_rank()}.safetensors"
    if target.exists():
        yield
        return
    paths = json.loads(Path(destination).read_text())
    if (
        not isinstance(paths, list)
        or not 1 <= len(paths) <= 32
        or not all(
            isinstance(path, str)
            and re.fullmatch(
                r"blocks\.\d+\.(mlp\.fc[12]|attn\.(?:qkv_proj|out_proj))", path
            )
            for path in paths
        )
        or len(set(paths)) != len(paths)
    ):
        raise ValueError("Capture requires 1–32 distinct whole H3 block projections")
    wanted = set(paths)
    candidates = {
        layer.prefix: layer
        for layer in model.modules()
        if isinstance(layer, LinearBase) and layer.prefix in wanted
    }
    if candidates.keys() != wanted:
        raise ValueError(
            f"Unknown projection capture paths: {sorted(wanted - candidates.keys())}"
        )

    layout = call_kwargs["local_embedding_layout"]
    indices = {}
    for group, key in (("video", "img_row_ids"), ("audio", "audio_row_ids")):
        rows = layout[key]
        if rows.numel():
            indices[group] = rows[:: max(1, (rows.numel() + 7) // 8)][:8]
    expected_rows = call_kwargs["block_token_tags"].numel()
    tensors = {}
    called = set()
    methods = []

    def capture(prefix, apply):
        def wrapped(layer, x, bias=None):
            out = apply(layer, x, bias)
            if prefix not in called:
                if (
                    x.ndim != 2
                    or out.ndim != 2
                    or x.size(0) != expected_rows
                    or out.size(0) != expected_rows
                ):
                    raise ValueError(
                        f"Cannot align TP-local capture with packed AV rows at {prefix}"
                    )
                for group, rows in indices.items():
                    tensors[f"{prefix}.{group}.input"] = (
                        x.index_select(0, rows).detach().cpu().contiguous()
                    )
                    tensors[f"{prefix}.{group}.output"] = (
                        out.index_select(0, rows).detach().cpu().contiguous()
                    )
                called.add(prefix)
            return out

        return wrapped

    try:
        for prefix, layer in candidates.items():
            method = layer.quant_method
            if not isinstance(method, UnquantizedLinearMethod):
                raise ValueError(
                    f"Capture requires an unquantized BF16 reference at {prefix}"
                )
            was_patched = "apply" in vars(method)
            methods.append((method, was_patched, getattr(method, "apply")))
            method.apply = capture(prefix, method.apply)
        yield
    finally:
        for method, was_patched, original in methods:
            if was_patched:
                method.apply = original
            else:
                del method.apply
    if called != wanted:
        raise RuntimeError(
            f"BF16 capture missed projections: {sorted(wanted - called)}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        key: os.environ[f"SGLANG_H3_QUANT_{key}"]
        for key in ("CHECKPOINT", "PLACEMENT", "SOURCE")
    }
    temporary = target.with_suffix(".tmp")
    save_file(
        tensors,
        str(temporary),
        metadata={
            **metadata,
            "request_id": request_id,
            "step": str(step),
            "rank": str(get_world_rank()),
            "modalities": ",".join(indices),
        },
    )
    temporary.replace(target)
