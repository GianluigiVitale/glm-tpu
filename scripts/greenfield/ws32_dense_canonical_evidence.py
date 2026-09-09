"""Replay one saved candidate against authenticated DB605 narrow originals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from scripts.greenfield import ws32_dense_canonical as canonical
from scripts.greenfield.prefill_window_evidence import same_json


def expected_stages() -> list[str]:
    stages = ["identity"] + [
        stage
        for _ in canonical.PROGRAMS
        for stage in ("lower_compile_started", "compiled", "raw_written", "inspected")
    ]
    stages += ["dense/admission"]
    for phase, _ in canonical.CALLS[:4]:
        stages += [phase + "/" + s for s in ("memory", "execute", "memory_after")]
    stages += ["canonical/preflight", "canonical/initial", "canonical/inputs"]
    stages += ["canonical/first128/" + s for s in ("memory", "execute", "memory_after")]
    return stages + ["canonical/reproduction"]


def replay_outputs(
    root: Path, record: Mapping, slots: Mapping[int, int], originals: Mapping
) -> tuple[dict, dict]:
    from scripts.greenfield.ws32_dense_frontier_evidence import read_model_capsules

    report = record["dense_canonical"]
    same_json(
        {
            k: report.get(k)
            for k in ("complete", "numerical_promotion", "performance_claim")
        },
        dict(complete=True, numerical_promotion=False, performance_claim=False),
        "canonical output scope",
    )
    _, fingerprints, arrays = read_model_capsules(
        root, {canonical.CAPSULE: report["original"]}, slots, {}, canonical_dense=True
    )
    comparison = canonical.compare(
        arrays[canonical.CAPSULE], originals, tuple(slots.values())
    )
    same_json(report["comparison"], comparison, "canonical retained narrow comparison")
    same_json(
        json.loads((root / "comparison.json").read_bytes()),
        comparison,
        "canonical saved comparison",
    )
    return comparison, fingerprints
