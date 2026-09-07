"""Fixed bounded-internal contract for the distinct FP32 route-sum candidate.

Used by both worker and original-evidence replay. Does not accept a tolerance
argument and cannot change the historical exact-admission protocol.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from glm_tpu.greenfield.benchmarking import (
    REAL_LAYER_OUTPUT_TOLERANCE,
    compare_bounded_tensor,
)


def compare_outputs(
    actual: np.ndarray, reference: np.ndarray, legacy: np.ndarray
) -> dict[str, Any]:
    if (
        actual.shape != (17, 1536)
        or reference.shape != actual.shape
        or legacy.shape != (1, 1536)
    ):
        raise ValueError("bounded real-MoE output geometry differs")
    aggregate = compare_bounded_tensor(actual, reference, REAL_LAYER_OUTPUT_TOLERANCE)
    rows = [
        compare_bounded_tensor(
            actual[i : i + 1], reference[i : i + 1], REAL_LAYER_OUTPUT_TOLERANCE
        )
        for i in range(17)
    ]
    direct = compare_bounded_tensor(actual[:1], legacy, REAL_LAYER_OUTPUT_TOLERANCE)
    return dict(
        passed=aggregate["passed"]
        and direct["passed"]
        and all(r["passed"] for r in rows),
        candidate_vs_m1=aggregate,
        per_row=rows,
        candidate_row0_vs_legacy=direct,
        classification="BOUNDED_INTERNAL_NOT_BIT_EXACT",
    )


def load_legacy_outputs() -> dict[str, np.ndarray]:
    """Authenticate the retained small oracle, without reading model payloads."""
    import json
    from scripts.greenfield.probe_ws32_prefill_moe import (
        PACK,
        ORACLE,
        ORACLE_SHA,
        CASES,
    )
    from scripts.greenfield.run_real_one_layer_ws32 import _load_oracle, _bfloat16_numpy

    _, oracle = _load_oracle(
        ORACLE,
        expected_manifest_sha256=ORACLE_SHA,
        pack_manifest=json.loads((PACK / "manifest.json").read_text()),
    )
    return {case: _bfloat16_numpy(oracle[f"{case}_output"]) for case in CASES}
