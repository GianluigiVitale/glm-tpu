"""Bounded MoE diagnostic evidence; never an arithmetic acceptance override."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import ml_dtypes
import numpy as np

BOUNDARIES = (
    "routed_partial",
    "routed_reduced",
    "weighted_routes",
    "local_sum_operand",
    "routed",
    "shared_partial",
    "shared_reduced",
    "shared",
)
F32 = {"routed_partial", "shared_partial", "local_sum_operand"}


def compare_arrays(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("boundary shape/dtype differs")
    kind = np.uint32 if a.dtype == np.float32 else np.uint16
    aa, bb = a.astype(np.float64), b.astype(np.float64)
    if not np.isfinite(aa).all() or not np.isfinite(bb).all():
        raise ValueError("non-finite boundary")
    return dict(
        shape=list(a.shape),
        dtype=str(a.dtype),
        bit_mismatches=int(np.count_nonzero(a.view(kind) != b.view(kind))),
        max_abs=float(np.max(np.abs(aa - bb))),
        mean_abs=float(np.mean(np.abs(aa - bb))),
        candidate_sha256=sha256(a.tobytes()).hexdigest(),
        reference_sha256=sha256(b.tobytes()).hexdigest(),
    )


def save_boundaries(candidate: dict, references: list[dict], path: Path) -> dict:
    if set(candidate) != set(BOUNDARIES) or len(references) != 17:
        raise ValueError("boundary capture coverage differs")
    tensors, comparisons = {}, {}
    for name in BOUNDARIES:
        reference_shards = [
            {
                int(s.device.id): np.asarray(s.data)[0, 0]
                for s in r[name].addressable_shards
            }
            for r in references
        ]
        for shard in candidate[name].addressable_shards:
            device = int(shard.device.id)
            a = np.asarray(shard.data)[0, 0]
            b = np.concatenate([r[device] for r in reference_shards], axis=0)
            key = f"{name}_{device}"
            comparisons[key] = compare_arrays(a, b)
            for label, value in (("candidate", a), ("reference", b)):
                tensors[f"{label}_{key}"] = (
                    value if name in F32 else value.view(np.uint16)
                )
    np.savez_compressed(path, **tensors)
    return dict(
        comparisons=comparisons,
        npz_sha256=sha256(path.read_bytes()).hexdigest(),
        scope="instrumented graph only; original-lowering equivalence requires separate review",
    )


def validate_boundaries(path: Path, record: dict) -> None:
    if sha256(path.read_bytes()).hexdigest() != record["boundaries"]["npz_sha256"]:
        raise ValueError("boundary NPZ digest differs")
    slots = record["local_device_slots"]
    expected = {
        f"{label}_{name}_{s['device_id']}"
        for label in ("candidate", "reference")
        for name in BOUNDARIES
        for s in slots
    }
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != expected:
            raise ValueError("boundary tensor coverage differs")
        for name in BOUNDARIES:
            for s in slots:
                key = f"{name}_{s['device_id']}"
                a, b = (
                    arrays[f"{label}_{key}"] for label in ("candidate", "reference")
                )
                if name not in F32:
                    if a.dtype != np.uint16 or b.dtype != np.uint16:
                        raise ValueError("boundary BF16 storage differs")
                    a, b = a.view(ml_dtypes.bfloat16), b.view(ml_dtypes.bfloat16)
                elif a.dtype != np.float32 or b.dtype != np.float32:
                    raise ValueError("boundary FP32 storage differs")
                if compare_arrays(a, b) != record["boundaries"]["comparisons"][key]:
                    raise ValueError(
                        "boundary comparison differs from original tensors"
                    )


def lowering_manifest(hlo: str) -> dict:
    """Retain relevant raw instructions/computations for a narrow human comparison.

    Names and source coordinates change under capture; this is NOT a mechanical
    proof that equal strategy labels mean equal numerical reduction schedules.
    """
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

    module = parse_hlo_module(hlo)
    selected = [
        op
        for op in module.instructions
        if op.is_collective
        or (
            op.op_name
            and ("reduce_sum" in op.op_name or "expert_reduce" in op.op_name)
            and any(s.dtype in ("bf16", "f32") for s in op.result_shapes)
        )
    ]
    computations = {op.computation for op in selected}
    for op in selected:
        computations.update(re.findall(r"(?:calls|to_apply)=%?([\w.]+)", op.raw_line))
    blocks = {}
    for name in computations:
        lines = [
            op.raw_line
            for op in module.instructions
            if op.computation.lstrip("%") == name.lstrip("%")
        ]
        if lines:
            raw = "\n".join(lines)
            blocks[name] = {"sha256": sha256(raw.encode()).hexdigest(), "text": raw}
    collectives = []
    for op in module.collectives:
        collectives.append(
            dict(
                opcode=op.opcode,
                role=op.op_name,
                results=[s.to_dict() for s in op.result_shapes],
                operands=list(op.operand_names),
                replica_groups=op.replica_groups,
                strategy=re.findall(r'"strategy":"([^"]+)"', op.raw_line),
                emitter=re.findall(r'"emitter":"([^"]+)"', op.raw_line),
                raw=op.raw_line,
            )
        )
    return dict(
        hlo_sha256=sha256(hlo.encode()).hexdigest(),
        collectives=collectives,
        blocks=blocks,
        original_lowering_equivalent="REQUIRES_REVIEW",
    )


def original_comparison(bucket: Any, destination: Path, record: dict) -> dict:
    """Compare instrumented final witnesses against the exact archived refusal."""
    root = Path(__file__).resolve().parents[2]
    evidence = json.loads(
        (
            root / "docs/artifacts/prefill-real-moe-arithmetic-refusal-20260907.json"
        ).read_text()
    )
    rank = record["launch_rank"]
    receipt = next(r for r in evidence["verified_runner_objects"] if r["rank"] == rank)
    prefix = f"results/{evidence['tag']}/workers/rank{rank}/"
    g = int(receipt["generation"])
    data = bucket.blob(prefix + "runner.json", generation=g).download_as_bytes(
        if_generation_match=g
    )
    if sha256(data).hexdigest() != receipt["sha256"]:
        raise ValueError("original failure runner digest differs")
    old = json.loads(data)
    if (
        old["cases"]["normal"]["input_sha256"]
        != record["cases"]["normal"]["input_sha256"]
    ):
        raise ValueError("diagnostic inputs differ from original failure")
    comparisons = []
    by_device = {s["device_id"]: s for s in old["cases"]["normal"]["shards"]}
    for s in record["cases"]["normal"]["shards"]:
        original = by_device[s["device_id"]]
        comparisons.append(
            dict(
                device_id=s["device_id"],
                candidate_matches_original=s["output_sha256"]
                == original["output_sha256"],
                reference_matches_original=s["reference_sha256"]
                == original["reference_sha256"],
            )
        )
    manifests = {}
    for kind, key in (("candidate", "hlo"), ("reference", "reference_hlo_sha256")):
        name = f"{kind}.optimized_hlo.txt"
        blob = bucket.get_blob(prefix + name)
        original_bytes = blob.download_as_bytes(if_generation_match=blob.generation)
        expected = old[key]["sha256"] if key == "hlo" else old[key]
        if sha256(original_bytes).hexdigest() != expected:
            raise ValueError("original failure graph digest differs")
        manifests[kind] = {
            "original": lowering_manifest(original_bytes.decode()),
            "instrumented": lowering_manifest((destination / name).read_text()),
        }
    result = dict(
        original_tag=evidence["tag"],
        original_runner=receipt,
        final_witnesses=comparisons,
        final_witnesses_preserved=all(
            s["candidate_matches_original"] and s["reference_matches_original"]
            for s in comparisons
        ),
        lowering=manifests,
        causal_claim=False,
    )
    (destination / "original_comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return {k: v for k, v in result.items() if k != "lowering"}
