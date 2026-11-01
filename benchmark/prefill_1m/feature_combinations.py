"""Explicit combinations and component qualification for the overnight matrix."""

import copy


def add_combinations(scenarios):
    scenarios["dcp-baseline"] = {"dcp": True}
    for source in ("indexer-query-shard", "multistream"):
        scenarios["dcp-" + source] = dict(copy.deepcopy(scenarios[source]), dcp=True)
    groups = {
        "dcp-packed-prefetch": ("dcp-packed-read", "dcp-gather-prefetch"),
        "dcp-packed-overlap": ("dcp-packed-read", "dcp-sfa-overlap"),
        "dcp-prefetch-overlap": ("dcp-gather-prefetch", "dcp-sfa-overlap"),
        "dcp-io-combined": (
            "dcp-packed-read",
            "dcp-gather-prefetch",
            "dcp-sfa-overlap",
        ),
        "dcp-io-indexer": ("dcp-io-combined", "dcp-indexer-query-shard"),
        "dcp-io-multistream": ("dcp-io-indexer", "dcp-multistream"),
        "moe-stream-auto": ("multistream", "auto-stream-thresholds", "auto-hccl"),
        "moe-fused-combined": (
            "tp-routed-scale",
            "tp-prequant",
            "tp-shared",
            "tp-shared-activation",
        ),
        "dcp-bde-indexer-int8": ("bde-v2", "indexer-int8"),
    }
    for name, components in groups.items():
        merged = {"components": components, "env": {}, "expected": []}
        for component in components:
            source = scenarios[component]
            for key, value in source.get("env", {}).items():
                if key in merged["env"] and merged["env"][key] != value:
                    raise ValueError(f"{name}: conflicting {key}")
                merged["env"][key] = value
            merged["expected"] = list(
                dict.fromkeys(merged["expected"] + source.get("expected", []))
            )
            for key in ("dcp", "separate_shared", "equal_tp_dcp"):
                if source.get(key):
                    merged[key] = True
            if source.get("probe"):
                merged["probe"] = source["probe"]
        scenarios[name] = merged


def qualification_key(case, name=None):
    return (
        case["stage"],
        str(case["checkpoint_path"]),
        bool(case["scenario"].get("dcp")),
        name or case["id"].removeprefix(case["stage"] + "-"),
    )


def component_blockers(case, qualified):
    return [
        name
        for name in case["scenario"].get("components", ())
        if not qualified.get(qualification_key(case, name), False)
    ]
