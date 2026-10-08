"""Opt-in NPU tuning policy; importing this module never initializes a device."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import math
import os
import re
import statistics
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)
MIB = 1 << 20
SCHEMA_VERSION = 1
_UNSET = object()
THRESHOLDS = {
    "moe_eager_multi_stream_min_tokens": "SGLANG_NPU_TP_MOE_EAGER_MULTI_STREAM_MIN_TOKENS",
    "indexer_sharding_min_tokens": "SGLANG_NPU_DSA_INDEXER_QUERY_SHARDING_MIN_TOKENS",
    "dsa_cp_min_tokens": "SGLANG_NPU_DSA_CP_MIN_TOKENS",
}
LAYOUTS = {
    "gate_weight_layout": ("transposed",),
    "gate_topk_layout": ("default",),
}


def enabled(name):
    return os.environ.get(name, "0").lower() in ("1", "true")


@dataclass(frozen=True)
class NpuTuningContext:
    device_name: str
    device_arch: str
    device_memory_mb: int | None
    model_type: str
    hidden_size: int
    num_layers: int
    num_routed_experts: int | None
    num_experts_per_tok: int | None
    tp_size: int
    ep_size: int
    attn_tp_size: int
    attn_cp_size: int
    dcp_size: int
    chunked_prefill_size: int
    max_prefill_tokens: int
    max_running_requests: int | None
    dtype: str
    kv_cache_dtype: str
    quantization: str | None
    cann_version: str = "unknown"
    torch_npu_version: str = "unknown"
    sglang_version: str = "unknown"
    page_size: int = 1
    gate_output_dtype: str = "fp32"
    gate_scale_alg: int = 0
    model_signature: str = ""
    experiment_signature: str = ""

    def key_data(self):
        # Capacity/policy fields also affect representative shapes and budgets.
        return asdict(self)


def representative_prefill_rows(context: NpuTuningContext) -> int:
    if context.max_prefill_tokens <= 0:
        raise ValueError("max_prefill_tokens must be positive for NPU autotuning")
    if context.chunked_prefill_size > 0:
        return min(context.chunked_prefill_size, context.max_prefill_tokens)
    return context.max_prefill_tokens


def effective_prefill_query_tokens(forward_batch, query=None) -> int:
    # Before query projection, input_ids carries the same serving padding.
    tokens = forward_batch.input_ids if query is None else query
    return tokens.shape[0]


@dataclass(frozen=True)
class NpuTuningDecision:
    name: str
    value: object
    source: str
    reason: str


def resolve(
    name, *, explicit=_UNSET, cached=_UNSET, automatic=_UNSET, default=None, reason=""
):
    for source, value in (
        ("explicit", explicit),
        ("cached-profile", cached),
        ("deterministic-auto", automatic),
        ("default", default),
    ):
        if value is not _UNSET:
            return NpuTuningDecision(name, value, source, reason)
    return NpuTuningDecision(name, None, "default", reason)


@lru_cache(maxsize=128)
def log_decision(decision):
    logger.info(
        "NPU autotune: %s=%s source=%s dry_run=%s %s",
        decision.name,
        decision.value,
        decision.source,
        enabled("SGLANG_NPU_AUTOTUNE_DRY_RUN"),
        decision.reason,
    )


def profile_key(context):
    return hashlib.sha256(
        json.dumps(context.key_data(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def cache_dir():
    return Path(
        os.environ.get("SGLANG_NPU_TUNING_CACHE_DIR", "~/.cache/sglang/npu_tuning")
    ).expanduser()


def validate_selected(selected):
    if not isinstance(selected, dict):
        raise ValueError("selected must be an object")
    for key, value in selected.items():
        if key in THRESHOLDS or key == "hccl_buffsize_mb":
            if type(value) is not int or value < (
                1 if key == "hccl_buffsize_mb" else 0
            ):
                raise ValueError(f"invalid {key}")
        elif key == "hccl_op_expansion_mode" and value not in (None, "AIV"):
            raise ValueError(f"invalid {key}")
        elif key == "task_queue_enable" and value not in (None, 2):
            raise ValueError(f"invalid {key}")
        elif key in ("page_size", "chunked_prefill_size", "max_running_requests") and (
            type(value) is not int or value <= 0
        ):
            raise ValueError(f"invalid {key}")
        elif key in LAYOUTS and value not in LAYOUTS[key]:
            raise ValueError(f"invalid {key}")
    return selected


def load_profile(context):
    path = cache_dir() / f"{profile_key(context)}.json"
    try:
        data = json.loads(path.read_text())
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")
        if data.get("context") != context.key_data():
            raise ValueError("context mismatch")
        return validate_selected(data["selected"])
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        logger.warning("NPU autotune: ignoring profile %s: %s", path, exc)
        return {}


def save_profile(context, selected, evidence=None):
    validate_selected(selected)
    directory = cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{profile_key(context)}.json"
    # Atomic replacement prevents scheduler ranks from observing partial JSON.
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as f:
        json.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "context": context.key_data(),
                "selected": selected,
                "evidence": evidence or {},
            },
            f,
            indent=2,
        )
        temporary = f.name
    os.replace(temporary, path)
    return path


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


def dcp_piece_rows(
    budget_bytes, bytes_per_row, dcp_size, slots, alignment, prefix_local_rows
):
    if min(budget_bytes, bytes_per_row, dcp_size, slots, alignment) <= 0:
        raise ValueError("DCP budget and dimensions must be positive")
    rows = budget_bytes // (bytes_per_row * dcp_size * slots)
    rows = rows // alignment * alignment
    if rows < alignment:
        raise ValueError("DCP scratch budget cannot hold one aligned ownership cycle")
    whole = max(alignment, math.ceil(prefix_local_rows / alignment) * alignment)
    return min(rows, whole)


def choose_measured(scores, baseline, min_improvement=0.01):
    """Keep the baseline when timing spread cannot distinguish the candidates."""
    if baseline not in scores or not scores[baseline]:
        raise ValueError("a measured baseline is required")
    medians = {k: statistics.median(v) for k, v in scores.items() if v}
    best = min(medians, key=medians.get)
    gain = medians[baseline] - medians[best]
    spread = max(statistics.pstdev(scores[baseline]), statistics.pstdev(scores[best]))
    if gain <= max(spread, medians[baseline] * min_improvement):
        return baseline, "inconclusive"
    return best, "measured"


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def context_from_server_args(server_args):
    from sglang.srt.arg_groups.model_override_base import model_config_of
    from sglang.srt.environ import envs

    model = model_config_of(server_args)
    hf = model.hf_text_config
    device = os.environ.get("SGLANG_NPU_TUNING_DEVICE", "unknown")
    arch = os.environ.get("SGLANG_NPU_TUNING_ARCH", "unknown")
    # npu-smi runs in a separate process; never initialize the parent's NPU.
    if device == "unknown":
        try:
            info = subprocess.check_output(["npu-smi", "info"], text=True, timeout=3)
            match = re.search(r"\|\s*\d+\s+(?:Ascend\s*)?(950[A-Za-z0-9_-]*)\s", info)
            if match:
                device, arch = f"Ascend {match[1]}", "arch35"
        except (OSError, subprocess.SubprocessError):
            pass
    cann = os.environ.get("SGLANG_NPU_TUNING_CANN_VERSION", "unknown")
    if cann == "unknown":
        for base in (
            os.environ.get("ASCEND_HOME_PATH", ""),
            "/usr/local/Ascend/ascend-toolkit/latest",
        ):
            try:
                contents = (Path(base) / "version.cfg").read_text()
                match = re.search(r"(?m)^Version=(.+)$", contents)
                if match:
                    cann = match[1].strip()
                    break
            except OSError:
                pass
    try:
        revision = subprocess.check_output(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parents[5]),
                "rev-parse",
                "HEAD",
            ],
            text=True,
            timeout=2,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        revision = _version("sglang")
    tp = server_args.tp_size
    cp = server_args.attn_cp_size
    dp = server_args.attn_dp_size
    return NpuTuningContext(
        device,
        arch,
        None,
        (getattr(hf, "architectures", None) or [hf.model_type])[0],
        hf.hidden_size,
        hf.num_hidden_layers,
        getattr(hf, "n_routed_experts", getattr(hf, "num_local_experts", None)),
        getattr(hf, "num_experts_per_tok", None),
        tp,
        server_args.ep_size,
        tp // dp // cp,
        cp,
        server_args.dcp_size,
        server_args.chunked_prefill_size,
        server_args.max_prefill_tokens,
        server_args.max_running_requests,
        str(model.dtype),
        server_args.kv_cache_dtype,
        model.quantization,
        cann,
        _version("torch-npu"),
        revision,
        server_args.page_size,
        os.environ.get("SGLANG_NPU_TP_MOE_MXFP8_GATE_LOGITS_DTYPE", "fp32"),
        int(os.environ.get("SGLANG_NPU_TP_MOE_MXFP8_GATE_SCALE_ALG", "0")),
        hashlib.sha256(
            json.dumps(
                {
                    "quantization": getattr(
                        model.hf_config, "quantization_config", None
                    ),
                    "router": {
                        key: getattr(hf, key, None)
                        for key in (
                            "scoring_func",
                            "n_group",
                            "topk_group",
                            "routed_scaling_factor",
                            "kv_lora_rank",
                            "qk_rope_head_dim",
                        )
                    },
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest(),
        hashlib.sha256(
            json.dumps(
                {
                    name: getattr(envs, name).get()
                    for name in (
                        "SGLANG_NPU_USE_MULTI_STREAM",
                        "SGLANG_NPU_TP_MOE_NATIVE_NORM_MXFP8",
                        "SGLANG_NPU_TP_MOE_MXFP8_GATE",
                        "SGLANG_NPU_TP_MOE_PREQUANT_INPUT",
                        "SGLANG_NPU_TP_MOE_REUSE_MXFP8",
                        "SGLANG_NPU_TP_MOE_SHARED_GMM1_MODE",
                        "SGLANG_NPU_TP_MOE_SHARED_GMM1",
                        "SGLANG_NPU_ENABLE_DSA_CP",
                        "SGLANG_NPU_ENABLE_DSA_INDEXER_QUERY_SHARDING",
                        "SGLANG_NPU_ENABLE_DCP_EXTEND_GATHER_PREFETCH",
                    )
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    )


def active_context():
    payload = os.environ.get("SGLANG_NPU_TUNING_CONTEXT")
    return NpuTuningContext(**json.loads(payload)) if payload else None


def active_profile():
    context = active_context()
    if context is None:
        return {}
    if "unknown" in (
        context.device_name,
        context.device_arch,
        context.cann_version,
        context.torch_npu_version,
        context.sglang_version,
    ):
        return {}
    return load_profile(context)


def initialize(server_args):
    """Resolve inherited settings before scheduler/HCCL initialization."""
    requested = any(
        enabled(name)
        for name in (
            "SGLANG_NPU_AUTO_HCCL_BUFFSIZE",
            "SGLANG_NPU_AUTO_STREAM_THRESHOLDS",
            "SGLANG_NPU_AUTO_DCP_EXTEND_GATHER_PIECE_ROWS",
            "SGLANG_NPU_TUNING_PROFILE",
            "SGLANG_NPU_MEMORY_DIAGNOSTICS",
        )
    )
    if not requested:
        return
    context = context_from_server_args(server_args)
    os.environ["SGLANG_NPU_TUNING_CONTEXT"] = json.dumps(asdict(context))
    profile = active_profile()
    context_file = os.environ.get("SGLANG_NPU_TUNING_CONTEXT_FILE")
    if context_file:
        Path(context_file).write_text(json.dumps(asdict(context), indent=2))
    if enabled("SGLANG_NPU_TUNING_PROFILE"):
        for key, name in (
            ("hccl_op_expansion_mode", "HCCL_OP_EXPANSION_MODE"),
            ("task_queue_enable", "TASK_QUEUE_ENABLE"),
        ):
            if key in profile:
                decision = resolve(
                    name,
                    explicit=os.environ.get(name, _UNSET),
                    cached=profile[key],
                    reason=f"tp={context.tp_size} cached runtime benchmark",
                )
                log_decision(decision)
                if (
                    not enabled("SGLANG_NPU_AUTOTUNE_DRY_RUN")
                    and decision.value is not None
                ):
                    os.environ.setdefault(name, str(decision.value))
    dry = enabled("SGLANG_NPU_AUTOTUNE_DRY_RUN")
    if enabled("SGLANG_NPU_AUTO_HCCL_BUFFSIZE") or enabled("SGLANG_NPU_TUNING_PROFILE"):
        rows = representative_prefill_rows(context)
        element = 4 if context.dtype in ("float32", "torch.float32") else 2
        headroom = float(os.environ.get("SGLANG_NPU_HCCL_HEADROOM", "1.25"))
        quantum = int(os.environ.get("SGLANG_NPU_HCCL_QUANTUM_MB", "32"))
        minimum = int(os.environ.get("SGLANG_NPU_HCCL_MIN_MB", "64"))
        maximum = int(os.environ.get("SGLANG_NPU_HCCL_MAX_MB", "1024"))
        auto = (
            hccl_buffer_mb(
                rows, context.hidden_size, element, headroom, quantum, minimum, maximum
            )
            if enabled("SGLANG_NPU_AUTO_HCCL_BUFFSIZE")
            else _UNSET
        )
        decision = resolve(
            "HCCL_BUFFSIZE",
            explicit=os.environ.get("HCCL_BUFFSIZE", _UNSET),
            cached=profile.get("hccl_buffsize_mb", _UNSET),
            automatic=auto,
            reason=f"collective_bytes={rows * context.hidden_size * element} headroom={headroom} quantum_mb={quantum} tp={context.tp_size} dtype={context.dtype}; HCCL default if unset",
        )
        log_decision(decision)
        if not dry and decision.value is not None:
            os.environ.setdefault(decision.name, str(decision.value))
    if enabled("SGLANG_NPU_AUTO_STREAM_THRESHOLDS") or enabled(
        "SGLANG_NPU_TUNING_PROFILE"
    ):
        for key, name in THRESHOLDS.items():
            decision = resolve(
                name,
                explicit=os.environ.get(name, _UNSET),
                cached=profile.get(key, _UNSET),
                default=0,
                reason=f"chunk={context.chunked_prefill_size} tp={context.tp_size}; calibrated profile"
                if key in profile
                else "no calibrated threshold; preserve existing behavior",
            )
            log_decision(decision)
            if not dry:
                os.environ.setdefault(name, str(decision.value))


def threshold_allows(name, rows):
    value = int(os.environ.get(name, "0"))
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return rows >= value


def dcp_scratch_budget(*, device, dcp_size, group):
    configured = int(os.environ.get("SGLANG_NPU_DCP_SCRATCH_BUDGET_MB", "256")) * MIB
    return _cached_dcp_scratch_budget(device, dcp_size, configured, group)


def _common_dcp_free_bytes(device, group):
    import torch

    free, _ = torch.npu.mem_get_info(device)
    free_min = torch.tensor(free, dtype=torch.int64, device=device)
    torch.distributed.all_reduce(
        free_min, op=torch.distributed.ReduceOp.MIN, group=group
    )
    return int(free_min.item())


@lru_cache(maxsize=None)
def _cached_dcp_scratch_budget(device, dcp_size, configured, group):
    # Cache the rank-agreed limit, never the prefix-dependent piece size.
    free = _common_dcp_free_bytes(device, group)
    budget = min(configured, free // 2)
    logger.info(
        "NPU DCP scratch budget: device=%s dcp=%s configured_bytes=%s "
        "common_free_bytes=%s budget_bytes=%s",
        device,
        dcp_size,
        configured,
        free,
        budget,
    )
    return budget


def auto_dcp_gather_rows(
    *,
    prefix_lens,
    dcp_size,
    alignment,
    bytes_per_row,
    budget_bytes,
    prefetch,
    extend_rows=0,
):
    name = "SGLANG_NPU_DCP_EXTEND_GATHER_PIECE_ROWS"
    if (
        not enabled("SGLANG_NPU_AUTO_DCP_EXTEND_GATHER_PIECE_ROWS")
        or name in os.environ
    ):
        return None
    budget = budget_bytes
    cycle = dcp_size * alignment
    local_rows = sum(math.ceil(p / cycle) * alignment for p in prefix_lens)
    slots = 2 if prefetch else 1
    own_bytes = extend_rows * bytes_per_row * slots
    rows = dcp_piece_rows(
        budget - own_bytes, bytes_per_row, dcp_size, slots, alignment, local_rows
    )
    footprint = rows * dcp_size * slots * bytes_per_row + own_bytes
    decision = NpuTuningDecision(
        name,
        rows * dcp_size,
        "deterministic-auto",
        f"local_rows={rows} row_bytes={bytes_per_row} dcp={dcp_size} slots={slots} alignment={alignment} extend_rows={extend_rows} scratch_bytes={footprint} budget_bytes={budget}",
    )
    log_decision(decision)
    return None if enabled("SGLANG_NPU_AUTOTUNE_DRY_RUN") else decision.value


def log_memory_diagnostics(model, kv_pool):
    """Report observed allocator state without modifying the memory planner."""
    if not enabled("SGLANG_NPU_MEMORY_DIAGNOSTICS"):
        return
    import torch

    free, total = torch.npu.mem_get_info()
    kv_bytes = (
        kv_pool.get_kv_size_bytes() if hasattr(kv_pool, "get_kv_size_bytes") else None
    )
    if isinstance(kv_bytes, tuple):
        kv_bytes = sum(kv_bytes)
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    allocated = torch.npu.memory_allocated()
    reserved = torch.npu.memory_reserved()
    context = active_context()
    activation = None
    if context is not None:
        rows = representative_prefill_rows(context)
        activation = (
            rows * context.hidden_size * (4 if "float32" in context.dtype else 2)
        )
    logger.info(
        "NPU memory diagnostic: total_bytes=%s weight_tensor_bytes=%s "
        "allocated_bytes=%s reserved_bytes=%s peak_allocated_bytes=%s "
        "kv_pool_bytes=%s activation_tensor_bytes=%s hccl_per_buffer_mb=%s hccl_buffer_count=unknown "
        "dcp_scratch_budget_mb=%s multistream_scratch_bytes=unknown "
        "observed_free_bytes=%s predicted_headroom=unknown; communicator reserves unvalidated",
        total,
        weights,
        allocated,
        reserved,
        torch.npu.max_memory_allocated(),
        kv_bytes,
        activation,
        os.environ.get("HCCL_BUFFSIZE", "runtime-default"),
        os.environ.get("SGLANG_NPU_DCP_SCRATCH_BUDGET_MB", "256"),
        free,
    )
