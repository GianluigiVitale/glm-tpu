"""Fixed host-side contract for real-weight/synthetic-state layer admission.

Replayed from each owner's original arrays by the controller. This is NOT a
legacy oracle or full-model gate. See PREFILL_FULL_LAYER_ADMISSION.md before use.
"""

from __future__ import annotations

from typing import Any, Mapping

import ml_dtypes
import numpy as np

from glm_tpu.greenfield.benchmarking.one_layer import (
    REAL_LAYER_OUTPUT_TOLERANCE,
    ROUTE_WEIGHT_TOLERANCE,
    compare_bounded_tensor,
)

PROTOCOL = "ws32-prefill-complete-layer-raw-reference-v1"
ROWS = 17
CAPACITY = 1024
PAGE_TABLE = (1, 0)
CASES = {"empty": (0, 17), "boundary": (505, 17), "tail": (505, 11)}
FIELDS = (
    "output",
    "residual",
    "kv",
    "index",
    "repair",
    "positions",
    "counts",
    "scores",
    "routes",
    "route_weights",
    "health",
    "normalized",
)


def _shape_contract(values: Mapping[str, np.ndarray]) -> None:
    expected = {
        "output": ((ROWS, 1536), ml_dtypes.bfloat16),
        "residual": ((ROWS, 1536), ml_dtypes.bfloat16),
        "kv": ((2, 64, 640), ml_dtypes.bfloat16),
        "index": ((2, 64, 128), ml_dtypes.bfloat16),
        "repair": ((2, 64, 128), ml_dtypes.bfloat16),
        "positions": ((ROWS, 2048), np.int32),
        "counts": ((ROWS,), np.int32),
        "scores": ((ROWS, 2048), np.float32),
        "routes": ((ROWS, 8), np.int32),
        "route_weights": ((ROWS, 8), np.float32),
        "health": ((ROWS,), np.bool_),
        "normalized": ((ROWS, 1536), ml_dtypes.bfloat16),
    }
    if set(values) != set(expected):
        raise ValueError("complete-layer observation fields differ")
    for name, (shape, dtype) in expected.items():
        if values[name].shape != shape or values[name].dtype != dtype:
            raise ValueError(f"complete-layer observation shape/dtype differs: {name}")


def _bounded_rows(
    actual: np.ndarray, reference: np.ndarray, *, routes: bool = False
) -> dict[str, Any]:
    tolerance = ROUTE_WEIGHT_TOLERANCE if routes else REAL_LAYER_OUTPUT_TOLERANCE
    aggregate = compare_bounded_tensor(actual, reference, tolerance)
    rows = [
        compare_bounded_tensor(actual[i : i + 1], reference[i : i + 1], tolerance)
        for i in range(actual.shape[0])
    ]
    return {
        "passed": aggregate["passed"] and all(row["passed"] for row in rows),
        "aggregate": aggregate,
        "rows": rows,
    }


def _selection_contract(
    values: Mapping[str, np.ndarray], *, offset: int, count: int
) -> None:
    for row in range(ROWS):
        length = offset + row + 1 if row < count else 0
        if values["counts"][row] != length:
            raise ValueError("complete-layer causal count differs")
        positions = values["positions"][row, :length]
        scores = values["scores"][row, :length]
        if (
            not np.array_equal(np.sort(positions), np.arange(length))
            or not np.isfinite(scores).all()
        ):
            raise ValueError("complete-layer causal coverage/scores differ")
        if not np.array_equal(np.lexsort((positions, -scores)), np.arange(length)):
            raise ValueError("complete-layer own-score order/ties differ")
        if not np.all(values["positions"][row, length:] == -1) or not np.all(
            np.isneginf(values["scores"][row, length:])
        ):
            raise ValueError("complete-layer selection padding differs")


def written_addresses(
    *, slot: int, offset: int, count: int
) -> tuple[tuple[int, int, int], ...]:
    """Independent host arithmetic: (query row, physical page, owner-local row)."""
    if (
        type(slot) is not int
        or not 0 <= slot < 32
        or not 0 <= offset < CAPACITY
        or not 0 <= count <= ROWS
        or offset + count > CAPACITY
    ):
        raise ValueError("complete-layer address input differs")
    result = []
    for row in range(count):
        position = offset + row
        logical_page, within_page = divmod(position, 512)
        owner, local_row = divmod(within_page, 64)
        if owner == slot // 4:
            result.append((row, PAGE_TABLE[logical_page], local_row))
    return tuple(result)


def compare_layer_case(
    actual: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    initial: Mapping[str, np.ndarray],
    *,
    slot: int,
    layer: int,
    case: str,
    actual_m64_keys: np.ndarray | None,
    reference_m64_keys: np.ndarray | None,
) -> dict[str, Any]:
    """Re-derive every bounded admission decision from observed tensor bytes.

    Reference observations are padded to17 with the canonical empty metadata.
    Initial state includes kv/index/repair and supplied positions/counts/scores.
    M64 key rows come from each path's actual captured normalized inputs. They
    are separate executable evidence, not an assertion of full-model fidelity.
    """
    if layer not in (0, 3) or case not in CASES:
        raise ValueError("unregistered complete-layer case")
    offset, count = CASES[case]
    _shape_contract(actual)
    _shape_contract(reference)
    for values in (actual, reference):
        if not values["health"].all():
            raise ValueError("complete-layer owner health failed")
        for name in (
            "output",
            "residual",
            "kv",
            "index",
            "repair",
            "normalized",
            "route_weights",
        ):
            if not np.isfinite(values[name]).all():
                raise ValueError(f"complete-layer nonfinite {name}")
        for name in ("output", "residual", "route_weights"):
            if np.any(values[name][count:] != 0):
                raise ValueError("complete-layer padded output differs")
        _selection_contract(values, offset=offset, count=count)
    if not np.array_equal(actual["routes"][:count], reference["routes"][:count]):
        raise ValueError("complete-layer route IDs differ; routed comparison forbidden")
    if layer == 0:
        if (
            np.any(actual["routes"] != -1)
            or np.any(reference["routes"] != -1)
            or np.any(actual["route_weights"] != 0)
            or np.any(reference["route_weights"] != 0)
        ):
            raise ValueError("dense layer exposed routes")
        for keys in (actual_m64_keys, reference_m64_keys):
            if (
                keys is None
                or keys.shape != (ROWS, 128)
                or keys.dtype != ml_dtypes.bfloat16
                or not np.isfinite(keys).all()
            ):
                raise ValueError("own-normalization M64 evidence missing")
    else:
        for values in (actual, reference):
            routes = values["routes"][:count]
            if (
                np.any(routes < 0)
                or np.any(routes >= 256)
                or any(len(set(row)) != 8 for row in routes)
            ):
                raise ValueError("MoE route IDs invalid")
            weights = values["route_weights"][:count]
            if np.any(weights < 0) or not np.allclose(
                weights.sum(axis=1), 1, rtol=0, atol=1e-6
            ):
                raise ValueError("MoE route weights invalid")
            for name in ("positions", "counts", "scores"):
                if not np.array_equal(values[name], initial[name]):
                    raise ValueError("IndexShare selection changed")
    comparisons = {
        name: _bounded_rows(actual[name][:count], reference[name][:count])
        for name in ("output", "residual")
    }
    comparisons["route_weights"] = _bounded_rows(
        actual["route_weights"][:count], reference["route_weights"][:count], routes=True
    )
    addresses = written_addresses(slot=slot, offset=offset, count=count)
    for name in ("kv", "index", "repair"):
        if (
            name not in initial
            or initial[name].shape != actual[name].shape
            or initial[name].dtype != ml_dtypes.bfloat16
            or not np.isfinite(initial[name]).all()
        ):
            raise ValueError("initial cache evidence differs")
        mask = np.zeros((2, 64), dtype=bool)
        if name == "kv" or layer == 0:
            for _, page, row in addresses:
                mask[page, row] = True
        for values in (actual, reference):
            if not np.array_equal(
                values[name][~mask].view(np.uint16),
                initial[name][~mask].view(np.uint16),
            ):
                raise ValueError(f"untouched {name} cache bytes changed")
            if name == "kv" and np.any(values[name][mask, 576:] != 0):
                raise ValueError("written KV structural padding changed")
        if layer == 0 and name == "repair":
            for values, keys in (
                (actual, actual_m64_keys),
                (reference, reference_m64_keys),
            ):
                for query_row, page, row in addresses:
                    if not np.array_equal(
                        values[name][page, row].view(np.uint16),
                        keys[query_row].view(np.uint16),
                    ):
                        raise ValueError(
                            "repair differs from own observed-input M64 reconstruction"
                        )
        comparisons[name + "_written_rows"] = _bounded_rows(
            actual[name][mask], reference[name][mask]
        )
    return {
        "passed": all(value["passed"] for value in comparisons.values()),
        "comparisons": comparisons,
        "written_rows_on_owner": len(addresses),
        "classification": "REAL_WEIGHTS_SYNTHETIC_STATE_RAW_LAYOUT_COMPONENT_ONLY",
        "dsa_scope": "OWN_SCORE_ORDER_AND_ALL_CAUSAL_COVERAGE_BELOW_TOPK_NOT_CUTOFF_PROOF",
        "cache_tolerance_scope": "EXPLICIT_NEW_EXPERIMENT_BOUND_NOT_HISTORICAL_CACHE_GUARANTEE",
        "reference_dsa": {"key_norm": "multiply_rsqrt", "score_precision": "highest"},
        "candidate_dsa": {"key_norm": "divide_sqrt", "score_precision": "default"},
        "performance_claim": False,
    }
