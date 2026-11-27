"""A2/A3 integer TP experiments; all candidate switches default off."""

import atexit
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
import logging
import os

import torch

logger = logging.getLogger(__name__)
_INPUT = ContextVar("tp12_input", default=None)
_SEEN = set()
_COUNTS = {}
_FIRST = {}
_SUMMARY_EVERY = 1000
_DUMP_REGISTERED = False


def enabled(name):
    return os.environ.get("SGLANG_NPU_TP12_" + name, "False").lower() in ("1", "true")


def evidence(name, **details):
    """Count one activation of a candidate path.

    A flag being set is not activation: only paths that actually executed an
    optimized branch may call this. Counts are aggregated (not deduplicated)
    and dumped to JSON at process exit; the first occurrence of each name is
    also logged for cheap server.log parsing.
    """
    key = (name, json.dumps(details, sort_keys=True))
    count = _COUNTS.get(name, 0) + 1
    _COUNTS[name] = count
    if name not in _FIRST:
        _FIRST[name] = key[1]
        logger.info("TP12 ACTIVE %s %s", name, key[1])
    elif count % _SUMMARY_EVERY == 0:
        logger.info(
            "TP12 EVIDENCE_SUMMARY %s",
            json.dumps({"name": name, "count": count}, sort_keys=True),
        )


def evidence_snapshot():
    """Point-in-time aggregate for the runner's per-phase collection."""
    return {
        "counts": dict(sorted(_COUNTS.items())),
        "first": dict(sorted(_FIRST.items())),
        "flags": {
            name: enabled(name)
            for name in (
                "INPUT_REUSE",
                "SHARED_STREAM",
                "FUSION",
                "GRAPH_EVIDENCE",
                "BCG_ATTENTION_ISLAND",
                "GATHER_WORKSPACE",
            )
        },
        "inventory": dict(sorted(_ELIGIBILITY.items())),
    }


def dump_evidence():
    """Write evidence JSON into SGLANG_TP12_EVIDENCE_DIR if configured.

    One file per process (rank-stamped by the launcher's PID); a summary is
    always logged so the runner can recover counts from server.log even when
    a worker exits without running atexit handlers.
    """
    summary = evidence_snapshot()
    logger.info("TP12 EVIDENCE_FINAL %s", json.dumps(summary, sort_keys=True))
    directory = os.environ.get("SGLANG_TP12_EVIDENCE_DIR", "")
    if not directory:
        return
    try:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"tp12-evidence-{os.getpid()}.json")
        with open(path, "w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    except OSError:
        logger.exception("TP12 evidence dump failed")


def _ensure_dump_registered():
    global _DUMP_REGISTERED
    if not _DUMP_REGISTERED and os.environ.get("SGLANG_TP12_EVIDENCE_DIR"):
        atexit.register(dump_evidence)
        _DUMP_REGISTERED = True


_ELIGIBILITY = {}


def shared_eligibility_inventory(model):
    """Startup per-layer shared-expert fusion eligibility evidence.

    Records, per shared-expert MLP, the actual quant method class names and
    whether the INT8 reuse/fusion contracts are satisfiable. A FUSION flag
    without eligible layers here is not an activated optimization.
    """
    for name, module in model.named_modules():
        if ".shared_experts" not in name or module.__class__.__name__ not in (
            "DeepseekV2MLP",
            "Glm4MoeMLP",
        ):
            continue
        gate_up = getattr(module, "gate_up_proj", None)
        down = getattr(module, "down_proj", None)
        entry = {
            "gate_up_method": type(kernel(gate_up)).__name__ if gate_up else None,
            "down_method": type(kernel(down)).__name__ if down else None,
            "gate_up_int8": accepts_int8(gate_up) if gate_up else False,
            "down_int8": accepts_int8(down) if down else False,
            "swiglu_limit": getattr(module, "swiglu_limit", None),
            "input_reuse_eligible": accepts_int8(gate_up) if gate_up else False,
            "fusion_eligible": (
                accepts_int8(down)
                and getattr(module, "swiglu_limit", None) is None
                if down
                else False
            ),
        }
        _ELIGIBILITY[name] = entry
    logger.info(
        "TP12 SHARED ELIGIBILITY %s",
        json.dumps(
            {
                "layers": len(_ELIGIBILITY),
                "input_reuse_eligible": sum(
                    1 for v in _ELIGIBILITY.values() if v["input_reuse_eligible"]
                ),
                "fusion_eligible": sum(
                    1 for v in _ELIGIBILITY.values() if v["fusion_eligible"]
                ),
            },
            sort_keys=True,
        ),
    )
    return _ELIGIBILITY


@dataclass(frozen=True)
class Int8Operand:
    data: torch.Tensor
    scale: torch.Tensor
    source_dtype: torch.dtype

    def validate(self, shape):
        if (
            self.data.dtype != torch.int8
            or tuple(self.data.shape) != tuple(shape)
            or self.scale.dtype != torch.float32
            or self.scale.numel() != shape[0]
            or self.data.device != self.scale.device
            or self.source_dtype != torch.bfloat16
        ):
            raise ValueError("Invalid TP12 per-token INT8 operand")
        return self

    def pair(self):
        return self.data, self.scale

    def record_stream(self, stream):
        self.data.record_stream(stream)
        self.scale.record_stream(stream)


@contextmanager
def input_scope(operand):
    token = _INPUT.set(operand)
    try:
        yield
    finally:
        _INPUT.reset(token)
        release_operand(operand)


# ---------------------------------------------------------------------------
# Persistent eager gather workspace (MATRIX case 85).
# Leased per (device, shape): one live lease per key, released when the
# operand scope exits; same-stream ordering makes reuse safe, record_stream
# covers any secondary-stream readers.
# ---------------------------------------------------------------------------
_WORKSPACES = {}


def gather_workspace(q, scale):
    """Return persistent (q_full, scale_full) buffers, or None when disabled.

    Rejects concurrent leases of the same key: overlapping use without an
    intervening release raises instead of silently aliasing buffers.
    """
    if not enabled("GATHER_WORKSPACE"):
        return None
    key = (str(q.device), q.dtype, tuple(q.shape), scale.dtype, scale.numel())
    entry = _WORKSPACES.get(key)
    if entry is None:
        entry = {
            "q": torch.empty((q.shape[0] * _workspace_world(), *q.shape[1:]), dtype=q.dtype, device=q.device),
            "scale": torch.empty(scale.numel() * _workspace_world(), dtype=scale.dtype, device=scale.device),
            "leased": False,
            "bytes": 0,
        }
        entry["bytes"] = entry["q"].numel() * entry["q"].element_size() + entry[
            "scale"
        ].numel() * entry["scale"].element_size()
        _WORKSPACES[key] = entry
        logger.info(
            "TP12 GATHER WORKSPACE alloc rows=%d bytes=%d",
            q.shape[0],
            entry["bytes"],
        )
    if entry["leased"]:
        raise RuntimeError(
            "TP12 gather workspace key already leased; concurrent use rejected"
        )
    entry["leased"] = True
    evidence("gather-workspace-lease", rows=int(q.shape[0]))
    return entry["q"], entry["scale"]


def _workspace_world():
    from sglang.srt.runtime_context import get_parallel

    return get_parallel().attn_tp_group.world_size


def release_workspace(tensor):
    """Release the lease held on a workspace gather-output tensor.

    Returns True when a lease was found and released; False when the tensor is
    not workspace-backed. Double release is a contract violation and raises.
    """
    for entry in _WORKSPACES.values():
        if tensor is entry["q"]:
            if not entry["leased"]:
                raise RuntimeError("TP12 gather workspace lease already released")
            entry["leased"] = False
            evidence("gather-workspace-release")
            return True
    return False


def release_operand(operand):
    """Release workspace leases held by a dropped operand (scope exit)."""
    if operand is None:
        return
    release_workspace(getattr(operand, "data", None))


def kernel(layer):
    return getattr(getattr(layer, "scheme", None), "kernel", None) or getattr(
        layer, "quant_method", None
    )


def accepts_int8(layer):
    from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
        NPUW8A8Int8DynamicLinearMethod,
    )

    method = kernel(layer)
    # ModelSlim stores its executable method on the scheme.
    return isinstance(method, NPUW8A8Int8DynamicLinearMethod) or isinstance(
        method, OnlineInt8Method
    )


def input_for_shared(mlp, hidden):
    operand = _INPUT.get()
    if (
        enabled("INPUT_REUSE")
        and operand is not None
        and accepts_int8(mlp.gate_up_proj)
    ):
        operand.validate(hidden.shape)
        evidence("shared-input-reuse")
        return operand.pair()
    return None


def shared_activation(mlp, gate_up):
    if (
        enabled("FUSION")
        and accepts_int8(mlp.down_proj)
        and getattr(mlp, "swiglu_limit", None) is None
        and not torch.compiler.is_compiling()
    ):
        from sglang.srt.model_executor.runner import get_is_capture_mode
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            is_in_breakable_cuda_graph,
        )

        if not get_is_capture_mode() and not is_in_breakable_cuda_graph():
            q, scale = torch.ops.npu.npu_dequant_swiglu_quant(
                gate_up, quant_mode=1, activate_left=True
            )
            evidence("shared-activation-quant")
            out, _ = mlp.down_proj((q, scale))
            return out
    return None


def parse_online(spec):
    if not spec:
        return set()
    result = set()
    for pair in spec.split(","):
        target, sep, scheme = pair.strip().partition(":")
        if (
            sep != ":"
            or target not in ("gate", "shared_expert", "qkv", "dense")
            or scheme != "w8a8_dynamic"
        ):
            raise ValueError(
                "A2/A3 online quantization requires gate/shared_expert/qkv/dense:w8a8_dynamic"
            )
        if target in result:
            raise ValueError("Duplicate online target: " + target)
        result.add(target)
    return result


def projection_role(name):
    parts = name.split(".")
    if "experts" in parts:
        return None
    if "shared_experts" in parts and parts[-1] in ("gate_up_proj", "down_proj"):
        return "shared_expert"
    if "mlp" in parts and parts[-1] == "gate":
        return "gate"
    if "self_attn" in parts and parts[-1] in (
        "qkv_proj",
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "fused_qkv_a_proj_with_mqa",
    ):
        return "qkv"
    if (
        "mlp" in parts
        and "shared_experts" not in parts
        and parts[-1] in ("gate_up_proj", "down_proj")
    ):
        return "dense"
    return None


class OnlineInt8Method:
    def apply(self, layer, x, bias=None):
        if isinstance(x, tuple):
            q, scale = x
            dtype = torch.bfloat16
        else:
            dtype = x.dtype
            operand = _INPUT.get()
            if (
                enabled("INPUT_REUSE")
                and operand is not None
                and getattr(layer, "_tp12_role", None) == "gate"
            ):
                operand.validate(x.shape)
                q, scale = operand.pair()
                evidence("gate-input-reuse")
            else:
                q, scale = torch.ops.npu.npu_dynamic_quant(x)
        evidence("online-" + layer._tp12_role)
        return torch.ops.npu.npu_quant_matmul(
            q,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=scale.flatten(),
            bias=bias,
            output_dtype=dtype,
        )


def convert_online(model, spec):
    targets = parse_online(spec)
    if not targets:
        return {}
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.hardware_backend.npu.utils import npu_format_cast

    counts = {key: 0 for key in targets}
    skipped = {}
    for name, layer in model.named_modules():
        role = projection_role(name)
        if role not in targets:
            continue
        weight = getattr(layer, "weight", None)
        floating_gate = (
            role == "gate"
            and type(layer).__name__ == "MoEGate"
            and not hasattr(layer, "quant_method")
        )
        if not floating_gate and not isinstance(
            getattr(layer, "quant_method", None), UnquantizedLinearMethod
        ):
            skipped[name] = "checkpoint-selected method preserved"
            continue
        if (
            weight is None
            or weight.ndim != 2
            or weight.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        ):
            raise ValueError("Unsupported floating projection storage: " + name)
        q, scale = torch.ops.npu.npu_dynamic_quant(weight.to(torch.bfloat16))
        layer.weight = torch.nn.Parameter(
            npu_format_cast(q).transpose(0, 1), requires_grad=False
        )
        layer.weight_scale = torch.nn.Parameter(
            scale.float().flatten(), requires_grad=False
        )
        layer.quant_method = OnlineInt8Method()
        layer._tp12_role = role
        counts[role] += 1
    logger.info(
        "TP12 ONLINE INVENTORY %s",
        json.dumps(dict(converted=counts, preserved=skipped), sort_keys=True),
    )
    # Targets that are wholly offline-quantized remain unchanged and are reported.
    if not any(counts.values()):
        raise RuntimeError("Online arm has no floating projections to convert")
    return counts


def supports(name):
    try:
        op = getattr(torch.ops.npu, name).default
        return str(op._schema)
    except (AttributeError, RuntimeError):
        return None


def probe():
    """Run outside capture, before model loading and expensive startup."""
    _ensure_dump_registered()
    import torch_npu

    device = torch.device("npu")
    name = torch_npu.npu.get_device_name(0)
    import acl

    arch, code = acl.rt.get_device_info(0, 601)
    if code != 0 or arch != 2201 or "910" not in name:
        raise RuntimeError(
            f"TP12 requires an identified 910B/910C (DAV2201); got {name}, arch={arch}, status={code}"
        )
    names = (
        "npu_dynamic_quant",
        "npu_moe_init_routing_v2",
        "npu_quant_matmul",
        "npu_dequant_swiglu_quant",
        "npu_add_rms_norm_dynamic_quant",
        "npu_moe_finalize_routing",
    )
    result = dict(device=name, arch=arch, operators={n: supports(n) for n in names})
    x = torch.linspace(-2, 2, 4 * 128, device=device, dtype=torch.bfloat16).reshape(
        4, 128
    )
    q, scale = torch.ops.npu.npu_dynamic_quant(x)
    Int8Operand(q, scale, x.dtype).validate(x.shape)
    wq, ws = torch.ops.npu.npu_dynamic_quant(x)
    actual = torch.ops.npu.npu_quant_matmul(
        q, wq.transpose(0, 1), ws, pertoken_scale=scale, output_dtype=torch.bfloat16
    )
    reference = (q.float() * scale[:, None]) @ (wq.float() * ws[:, None]).transpose(
        0, 1
    )
    torch.testing.assert_close(actual.float(), reference, atol=0.2, rtol=0.02)
    result["int8_matmul_parity"] = True
    if enabled("INPUT_REUSE"):
        from sglang.srt.hardware_backend.npu.moe.routing import npu_moe_init_routing

        ids = torch.zeros((4, 1), dtype=torch.int32, device=device)
        a = npu_moe_init_routing(x, ids, 1, quant_mode=1, active_num=4)
        b = npu_moe_init_routing(q, ids, 1, quant_mode=1, scale=scale, active_num=4)
        torch.testing.assert_close(a.hidden_states, b.hidden_states, atol=0, rtol=0)
        torch.testing.assert_close(a.scale, b.scale, atol=0, rtol=0)
        result["routing_reuse_parity"] = True
    if enabled("FUSION"):
        q, scale = torch.ops.npu.npu_dequant_swiglu_quant(
            x, quant_mode=1, activate_left=True
        )
        a, b = x.chunk(2, dim=-1)
        ref = (torch.nn.functional.silu(a.float()) * b.float()).to(x.dtype)
        torch.testing.assert_close(
            q.float() * scale[:, None], ref.float(), atol=0.04, rtol=0.04
        )
        result["activation_quant_parity"] = True
    # This source's layer-boundary API requires a BF16 normalized readout.
    # Native fused norm returns only quantized data and the residual.
    result["fused_norm"] = (
        "ineligible: BF16 boundary readout is still required; no dequantized substitute"
    )
    torch_npu.npu.synchronize()
    return result
