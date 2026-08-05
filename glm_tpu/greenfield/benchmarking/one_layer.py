"""Correctness and optimized-HLO gates for the protected real MoE layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..sharding.hlo_contract import parse_hlo_module


@dataclass(frozen=True, slots=True)
class TensorTolerance:
    max_abs: float
    p99_abs: float
    mean_abs: float

    def __post_init__(self) -> None:
        if min(self.max_abs, self.p99_abs, self.mean_abs) < 0:
            raise ValueError("tensor tolerances must be non-negative")

    def to_dict(self) -> dict[str, float]:
        return {
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "p99_abs": self.p99_abs,
        }


REAL_LAYER_OUTPUT_TOLERANCE = TensorTolerance(
    max_abs=0.125,
    p99_abs=0.0625,
    mean_abs=0.02,
)
ROUTE_WEIGHT_TOLERANCE = TensorTolerance(
    max_abs=5e-4,
    p99_abs=5e-4,
    mean_abs=2e-4,
)


def compare_bounded_tensor(
    observed: Any,
    reference: Any,
    tolerance: TensorTolerance,
) -> dict[str, Any]:
    """Return a complete bounded-error record and fail only at the caller."""

    observed_array = np.asarray(observed, dtype=np.float32)
    reference_array = np.asarray(reference, dtype=np.float32)
    if observed_array.shape != reference_array.shape:
        raise ValueError(
            "bounded tensor shape mismatch: "
            f"observed={observed_array.shape} reference={reference_array.shape}"
        )
    if not np.isfinite(observed_array).all() or not np.isfinite(
        reference_array
    ).all():
        raise ValueError("bounded tensor comparison contains non-finite values")
    error = np.abs(observed_array - reference_array)
    maximum = float(np.max(error, initial=0.0))
    mean = float(np.mean(error)) if error.size else 0.0
    p99 = float(np.quantile(error, 0.99)) if error.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(error)))) if error.size else 0.0
    maximum_flat_index = int(np.argmax(error)) if error.size else 0
    maximum_index = (
        list(np.unravel_index(maximum_flat_index, error.shape))
        if error.size
        else []
    )
    passed = (
        maximum <= tolerance.max_abs
        and p99 <= tolerance.p99_abs
        and mean <= tolerance.mean_abs
    )
    return {
        "error": {
            "max_abs": maximum,
            "max_abs_index": maximum_index,
            "mean_abs": mean,
            "p99_abs": p99,
            "rms": rms,
        },
        "observed_range": [
            float(np.min(observed_array, initial=0.0)),
            float(np.max(observed_array, initial=0.0)),
        ],
        "passed": passed,
        "reference_range": [
            float(np.min(reference_array, initial=0.0)),
            float(np.max(reference_array, initial=0.0)),
        ],
        "shape": list(error.shape),
        "tolerance": tolerance.to_dict(),
    }


def validate_real_layer_hlo(
    optimized_hlo: str,
    *,
    hidden_size: int = 6144,
    stage_size: int = 4,
) -> dict[str, Any]:
    """Require one BF16 stacked combine over the exact local stage ranks."""

    module = parse_hlo_module(optimized_hlo)
    violations = []
    collectives = module.collectives
    if len(collectives) != 1:
        violations.append(
            f"expected exactly one collective, found {len(collectives)}"
        )
    if collectives:
        collective = collectives[0]
        if collective.opcode != "all-reduce":
            violations.append(
                f"expected all-reduce combine, found {collective.opcode}"
            )
        expected_groups = (tuple(range(stage_size)),)
        if collective.replica_groups != expected_groups:
            violations.append(
                "expected exact local replica group "
                f"{expected_groups}, found {collective.replica_groups}"
            )
        shapes = collective.operand_shapes + collective.result_shapes
        expected_payload = (2, 1, hidden_size)
        if not any(
            shape.dtype.lower() in {"bf16", "bfloat16"}
            and shape.dimensions == expected_payload
            for shape in shapes
        ):
            violations.append(
                "combine payload is not exact "
                f"bf16{expected_payload}: {[shape.to_dict() for shape in shapes]}"
            )
    forbidden = []
    for instruction in module.instructions:
        for shape in instruction.result_shapes:
            if shape.dimensions == (32, hidden_size):
                forbidden.append(instruction.name)
    if forbidden:
        violations.append(
            f"found forbidden dead-row/full-pod hidden shapes: {forbidden}"
        )
    if module.num_partitions not in (None, stage_size):
        violations.append(
            f"expected {stage_size} partitions, found {module.num_partitions}"
        )
    return {
        "collective_count": len(collectives),
        "collectives": [item.to_dict() for item in collectives],
        "module_name": module.name,
        "num_partitions": module.num_partitions,
        "num_replicas": module.num_replicas,
        "passed": not violations,
        "violations": violations,
    }
