# SPDX-License-Identifier: Apache-2.0
"""Experiment IDs from the research handoff; availability is never inferred."""

# id, track, priority, hypothesis, required device capability evidence
EXPERIMENTS = [
    (
        "B00",
        "baseline",
        "must",
        "Reproduce 28.33 s and effective four-step precision",
        [],
    ),
    ("B01", "baseline", "must", "Measure stage and duration scaling", []),
    ("B02", "baseline", "must", "Establish installed operator contracts", []),
    (
        "B03",
        "rainfusion",
        "P0",
        "Tune native sparse precision and schedule",
        ["rainfusion"],
    ),
    (
        "A01",
        "distributed_chunks",
        "P0",
        "Sequential AV latent-tail continuation",
        ["continuation"],
    ),
    (
        "A02",
        "distributed_chunks",
        "P0",
        "Oracle-anchor parallel quality feasibility",
        ["anchor_conditioning"],
    ),
    (
        "A03",
        "distributed_chunks",
        "P1",
        "Real-anchor fixed-budget chunk scheduler",
        ["continuation", "anchor_conditioning"],
    ),
    ("C01", "bade", "P0", "Static FFN and query compaction", ["compact_linears"]),
    (
        "C02",
        "reference_compaction",
        "P0",
        "Reference pruning with original positions",
        ["reference_layout"],
    ),
    (
        "C03",
        "bade",
        "P1",
        "Dynamic activity against matched static budgets",
        ["compact_linears"],
    ),
    ("B04", "bmka", "P1", "Bound exact KV and router candidates", ["external_indices"]),
    (
        "B05",
        "bmka",
        "P1",
        "Disjoint exact/representative softmax merge",
        ["external_indices", "lse"],
    ),
    (
        "B06",
        "bmka",
        "P1",
        "Hierarchical budgets and routing reuse",
        ["external_indices"],
    ),
    (
        "E01",
        "temporal_resolution",
        "P1",
        "Reduce latent frames at fixed physical duration",
        ["temporal_geometry"],
    ),
    (
        "E02",
        "temporal_resolution",
        "P2",
        "Existing interpolation/decoder auxiliary system",
        ["interpolation"],
    ),
    (
        "A04",
        "async_av_decode",
        "P1",
        "Bounded incremental AV decode and output overlap",
        ["incremental_av_decode"],
    ),
    (
        "D01",
        "sparse_context_parallel",
        "P1",
        "Overlap exposed HCCL with attention",
        ["hccl_overlap"],
    ),
    (
        "D02",
        "sparse_context_parallel",
        "P2",
        "Exchange prequantized operands and scales",
        ["prequantized_exchange"],
    ),
    (
        "D03",
        "sparse_context_parallel",
        "P2",
        "Route remote KV demand unions",
        ["external_indices", "sparse_exchange"],
    ),
    (
        "D04",
        "sparse_context_parallel",
        "P2",
        "Row-sharded TP and AdaLN folding",
        ["row_sharded_tp"],
    ),
    ("Z01", "combined", "final", "Measure combinations of validated winners", []),
    (
        "Z02",
        "combined",
        "final",
        "Measure 30/60/120 s and streaming stability",
        ["continuation"],
    ),
]

CATALOG = {
    item[0]: dict(
        id=item[0],
        track=item[1],
        priority=item[2],
        hypothesis=item[3],
        requires=item[4],
    )
    for item in EXPERIMENTS
}
