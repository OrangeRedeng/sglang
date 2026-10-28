"""Resolve streaming integer presets and per-projection overrides."""

from dataclasses import dataclass
from functools import lru_cache

PRESETS = ("w8a8_int", "w4a4_int")
TARGETS = ("gate", "moe", "shared_expert", "qkv", "dense")
SCHEMES = {
    "w8a8_dynamic": "W8A8_DYNAMIC",
    "w8a8_int8": "W8A8_DYNAMIC",
    "int8": "W8A8_DYNAMIC",
    "w4a4_dynamic": "W4A4_DYNAMIC",
}


@dataclass(frozen=True)
class OnlineQuantization:
    preset: str | None
    overrides: tuple[tuple[str, str], ...]

    def scheme(self, target):
        explicit = dict(self.overrides).get(target)
        if explicit is not None:
            return explicit
        if self.preset is None or target == "gate":
            return None
        return (
            "W4A4_DYNAMIC"
            if self.preset == "w4a4_int" and target == "moe"
            else "W8A8_DYNAMIC"
        )

    def integer_mode(self, target="moe"):
        scheme = self.scheme(target)
        return {"W8A8_DYNAMIC": "w8a8_int", "W4A4_DYNAMIC": "w4a4_int"}.get(scheme)


@lru_cache(maxsize=128)
def resolve_online_quantization(spec: str | None) -> OnlineQuantization:
    preset = None
    overrides = {}
    for entry in (spec or "").lower().split(","):
        entry = entry.strip()
        if not entry:
            continue
        if entry in PRESETS:
            if preset is not None or overrides:
                raise ValueError(
                    "A single online integer preset must precede overrides"
                )
            preset = entry
            continue
        target, sep, scheme = entry.partition(":")
        target, scheme = target.strip(), scheme.strip()
        if not sep or target not in (*TARGETS, "all") or scheme not in SCHEMES:
            raise ValueError(f"Invalid online quantization TARGET:SCHEME: {entry!r}")
        if target in overrides:
            raise ValueError(f"Duplicate online quantization target: {target}")
        overrides[target] = SCHEMES[scheme]
    if "all" in overrides:
        common = overrides.pop("all")
        overrides = {**dict.fromkeys(TARGETS, common), **overrides}
    return OnlineQuantization(preset, tuple(sorted(overrides.items())))


def projection_target(prefix):
    parts = prefix.split(".")
    if "shared_experts" in parts or "shared_expert" in parts:
        return "shared_expert"
    if parts[-1] in (
        "qkv_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
    ):
        return "qkv"
    if (
        "mlp" in parts
        and "experts" not in parts
        and parts[-1] in ("gate_up_proj", "gate_proj", "up_proj", "down_proj")
    ):
        return "dense"
    return None
