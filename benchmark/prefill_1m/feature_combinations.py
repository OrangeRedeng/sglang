"""Explicit combinations and component qualification for the overnight matrix."""

import copy


def add_combinations(scenarios):
    scenarios["dcp-baseline"] = {"dcp": True}
    historical = copy.deepcopy(scenarios["bde-v2"])
    historical["env"].update(
        DEEPEP_HCCL_BUFFSIZE="2500",
        HCCL_BUFFSIZE="1000",
        HCCL_SOCKET_IFNAME="lo",
        GLOO_SOCKET_IFNAME="lo",
        HCCL_OP_EXPANSION_MODE="AIV",
        PYTORCH_NPU_ALLOC_CONF="expandable_segments:True",
        SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP="True",
        SGLANG_ENABLE_OVERLAP_PLAN_STREAM="1",
        SGLANG_NPU_USE_MULTI_STREAM="1",
        SGLANG_SET_CPU_AFFINITY="1",
        STREAMS_PER_DEVICE="32",
    )
    historical["launch_overrides"] = dict(
        chunk=32768,
        mem_fraction=0.85,
        kv_rows=67584,
        max_running_requests=1,
        context_length=1048576,
    )
    scenarios["historical-bde"] = historical
    legacy = dict(
        dcp=True,
        separate_shared=True,
        launch_overrides=dict(
            chunk=16384,
            mem_fraction=0.88,
            kv_rows=74752,
            max_running_requests=64,
            prefill_max_requests=1,
            context_length=1048576,
        ),
    )
    scenarios["legacy-control-40s"] = copy.deepcopy(legacy)
    scenarios["legacy-megamoe-34s"] = dict(
        copy.deepcopy(scenarios["megamoe"]), **legacy
    )
    for name, moe, batch2 in (
        ("dsa-metadata", False, False),
        ("dsa-metadata-batch2", False, True),
        ("dsa-metadata-megamoe", True, False),
        ("dsa-metadata-megamoe-batch2", True, True),
    ):
        feature = copy.deepcopy(scenarios["megamoe"]) if moe else {"dcp": True}
        feature.setdefault("env", {})["SGLANG_NPU_DSA_METADATA_REUSE"] = "True"
        feature.setdefault("expected", []).append("dsa-metadata-reuse")
        feature["dcp"] = True
        if batch2:
            feature["probe_batch2"] = True
            feature["launch_overrides"] = dict(
                max_running_requests=2, prefill_max_requests=2
            )
            feature["expected"].append("dcp-prefill-batch-2")
        scenarios[name] = feature
    for name, source in (
        ("dsa-metadata-packed-batch2", "dsa-metadata-batch2"),
        ("dsa-metadata-packed-megamoe-batch2", "dsa-metadata-megamoe-batch2"),
    ):
        scenarios[name] = copy.deepcopy(scenarios[source])
        scenarios[name]["env"]["SGLANG_NPU_ENABLE_DCP_PACKED_READ_BATCH2"] = "True"
        scenarios[name]["expected"].append("dcp-packed-read-batch2")
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


def scenario_for(scenarios, name, reference_recipe="ordinary"):
    variant = copy.deepcopy(scenarios[name])
    if (
        reference_recipe != "historical-bde"
        or name == "historical-bde"
        or name.startswith("legacy-")
    ):
        return variant
    baseline = copy.deepcopy(scenarios["historical-bde"])
    if name in ("baseline", "dcp-baseline"):
        return baseline
    if name == "indexer-int8":
        baseline["unsupported"] = (
            "INT8 indexer is already enabled in the 24s baseline; use indexer-fp16 ablation"
        )
        return baseline
    baseline.update(
        {
            k: v
            for k, v in variant.items()
            if k not in ("env", "expected", "launch_overrides")
        }
    )
    baseline["launch_overrides"].update(variant.get("launch_overrides", {}))
    overlay = dict(variant.get("env", {}))
    if "auto-hccl" not in variant.get("expected", []):
        for key in ("HCCL_BUFFSIZE", "DEEPEP_HCCL_BUFFSIZE"):
            overlay.pop(key, None)
    baseline["env"].update(overlay)
    baseline["expected"] = list(
        dict.fromkeys(baseline["expected"] + variant.get("expected", []))
    )
    if name == "indexer-fp16":
        baseline["expected"].remove("indexer-int8")
    if variant.get("fuseep") == 3:
        baseline["expected"].append("dsa-ep-moe-local")
    return baseline
