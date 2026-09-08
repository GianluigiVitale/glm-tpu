"""Fixed real-layer6 B128 versus fourB32 admission, not a model benchmark.

No workflow launch or TPU initialization. Historical B17 constants/comparators
remain unchanged. Original inputs and outputs must be retained on all32 owners.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import ml_dtypes
import numpy as np

from glm_tpu.greenfield.benchmarking.one_layer import (
    REAL_LAYER_OUTPUT_TOLERANCE,
    ROUTE_WEIGHT_TOLERANCE,
    compare_bounded_tensor,
)
from scripts.greenfield.prefill_layer_numerical import FIELDS
from scripts.greenfield.prefill_layer_evidence import (
    INPUT_FIELDS,
    decode_arrays,
    equal_bytes,
    input_hashes,
    owner_inputs,
)

PROTOCOL = "ws32-prefill-layer6-window128-control4x32-v1"
KERNEL = "ws32_prefill_layer_window_numerical"
REFERENCE_SCOPE = "B128_WINDOW_VS_FOUR_COMPLETED_B32_LAYERS_SYNTHETIC_HISTORY"
LAYER = 6
ROWS = 128
CONTROL_ROWS = 32
CAPACITY = 4096
KEY_TILE = 512  # One local512-key owner at capacity4096, not a full-context tile.
PAGE_TABLE = (7, 2, 5, 0, 6, 1, 4, 3)
CASES = {"boundary": (505, 128), "competitive": (2553, 128), "tail": (2553, 33)}
PAYLOAD_BYTES_PER_CHIP = 326_079_840
TENSORS_PER_CHIP = 35
BF16 = ml_dtypes.bfloat16


def host_case(case: str, rope: np.ndarray) -> dict[str, np.ndarray]:
    """Fixed synthetic history with real weights; no authenticated-resume claim."""
    if case not in CASES or rope.shape != (CAPACITY, 64) or rope.dtype != BF16:
        raise ValueError("unregistered window case/rotary table")
    offset, count = CASES[case]
    rng = np.random.RandomState(20260908)
    result = {
        "update": (rng.standard_normal((ROWS, 6144)) * 0.01).astype(BF16),
        "residual": (rng.standard_normal((ROWS, 6144)) * 0.01).astype(BF16),
        "kv": np.zeros((8, 8, 64, 640), BF16),
        "index": np.zeros((8, 8, 64, 128), BF16),
        "repair": np.zeros((8, 8, 64, 128), BF16),
        "positions": np.full((ROWS, 2048), -1, np.int32),
        "counts": np.zeros(ROWS, np.int32),
        "scores": np.full((ROWS, 2048), -np.inf, np.float32),
        "offset": np.asarray(offset, np.int32),
        "count": np.asarray(count, np.int32),
        "table": np.asarray([PAGE_TABLE], np.int32),
        "health": np.ones((8, 4, ROWS), np.bool_),
        "rope": rope[offset : offset + ROWS].copy(),
    }
    for position in range(offset):
        page, within = divmod(position, 512)
        owner, row = divmod(within, 64)
        address = owner, PAGE_TABLE[page], row
        result["kv"][address][:576] = (rng.standard_normal(576) * 0.05).astype(BF16)
        for name in ("index", "repair"):
            result[name][address] = (rng.standard_normal(128) * 0.05).astype(BF16)
    for name in ("update", "residual", "rope"):
        result[name][count:] = np.nan
    return result


def control_inputs(
    values: tuple[Any, ...], tile: int, previous: tuple[Any, ...] | None = None
) -> tuple[Any, ...]:
    """Prepare outside timing; carry ALL three proposed cache outputs on device."""
    import jax.numpy as jnp

    if len(values) != 20 or type(tile) is not int or not 0 <= tile < 4:
        raise ValueError("window control requires20 inputs and tile0..3")
    if values[0].shape[0] != ROWS:
        raise ValueError("control source is not the complete128-row window")
    start, end = tile * CONTROL_ROWS, (tile + 1) * CONTROL_ROWS
    result = list(values)
    for i in (0, 1, 5, 6, 7, 19):
        result[i] = values[i][start:end]
    result[8] = jnp.minimum(values[8] + jnp.int32(start), CAPACITY - 1)
    result[9] = jnp.clip(values[9] - start, 0, CONTROL_ROWS)
    result[18] = values[18][:, :, start:end]
    if previous is not None:
        if len(previous) != len(FIELDS):
            raise ValueError("control prior output inventory differs")
        result[2:5] = previous[2:5]
    elif tile != 0:
        raise ValueError("later control tile requires its actual prior output")
    return tuple(result)


def stack_control(rows: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Preserve last cache, concatenate four real32-row observations in order."""
    if len(rows) != 4 or any(set(r) != set(FIELDS) for r in rows):
        raise ValueError("window control needs four complete observed blocks")
    result = {}
    for name in FIELDS:
        if name in ("kv", "index", "repair"):
            result[name] = rows[-1][name].copy()
        else:
            first = rows[0][name]
            if first.shape[0] != CONTROL_ROWS or any(
                r[name].shape != first.shape or r[name].dtype != first.dtype
                for r in rows
            ):
                raise ValueError("control row geometry drifted")
            result[name] = np.concatenate([r[name] for r in rows])
    return result


def build_programs(
    mesh: Any, specs: tuple[Any, ...], **options: Any
) -> tuple[Any, Any]:
    """Reuse the exact layer20-in/12-out schema; never use the scalar reference."""
    from scripts.greenfield.prefill_layer_programs import build_layer_programs

    common = dict(full_indexer=True, sparse_mlp=True, key_tile=KEY_TILE, **options)
    wide, _ = build_layer_programs(mesh, specs, candidate_window=True, **common)
    small, _ = build_layer_programs(mesh, specs, **common)
    return wide, small


def written_addresses(
    slot: int, offset: int, count: int
) -> tuple[tuple[int, int, int], ...]:
    if (
        type(slot) is not int
        or not 0 <= slot < 32
        or (offset, count) not in CASES.values()
    ):
        raise ValueError("unregistered window owner/span")
    addresses = []
    for q in range(count):
        page, within = divmod(offset + q, 512)
        owner, row = divmod(within, 64)
        if owner == slot // 4:
            addresses.append((q, PAGE_TABLE[page], row))
    return tuple(addresses)


def _shape_contract(values: Mapping[str, np.ndarray]) -> None:
    expected = {
        "output": ((ROWS, 1536), BF16),
        "residual": ((ROWS, 1536), BF16),
        "kv": ((8, 64, 640), BF16),
        "index": ((8, 64, 128), BF16),
        "repair": ((8, 64, 128), BF16),
        "normalized": ((ROWS, 1536), BF16),
        "positions": ((ROWS, 2048), np.int32),
        "counts": ((ROWS,), np.int32),
        "scores": ((ROWS, 2048), np.float32),
        "routes": ((ROWS, 8), np.int32),
        "route_weights": ((ROWS, 8), np.float32),
        "health": ((ROWS,), np.bool_),
    }
    if set(values) != set(expected):
        raise ValueError("window observation fields differ")
    for name, (shape, dtype) in expected.items():
        if values[name].shape != shape or values[name].dtype != dtype:
            raise ValueError(f"window observation geometry/dtype differs: {name}")


def _selection_contract(
    values: Mapping[str, np.ndarray], offset: int, count: int
) -> None:
    for row in range(ROWS):
        length = offset + row + 1 if row < count else 0
        n = min(length, 2048)
        if values["counts"][row] != n:
            raise ValueError("window causal selection count differs")
        positions, scores = values["positions"][row, :n], values["scores"][row, :n]
        if (
            len(np.unique(positions)) != n
            or np.any(positions < 0)
            or np.any(positions >= length)
            or not np.isfinite(scores).all()
            or not np.array_equal(np.lexsort((positions, -scores)), np.arange(n))
        ):
            raise ValueError("window own selected-score order/ties/causality differs")
        if (
            np.any(values["positions"][row, n:] != -1)
            or not np.isneginf(values["scores"][row, n:]).all()
        ):
            raise ValueError("window selection padding differs")


def compare_case(
    actual: Mapping[str, np.ndarray],
    control: Mapping[str, np.ndarray],
    initial: Mapping[str, np.ndarray],
    *,
    slot: int,
    case: str,
) -> dict[str, Any]:
    """Fixed per-row bounds, exact route IDs, selections and untouched cache bytes.

    Competitive set agreement is against the independently composed B32 control,
    not an independent full canonical score row or a full-model §21 proof.
    """
    if case not in CASES:
        raise ValueError("unregistered window comparison case")
    offset, count = CASES[case]
    for values in (actual, control):
        _shape_contract(values)
        _selection_contract(values, offset, count)
        if not values["health"].all():
            raise ValueError("window owner health failed")
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
                raise ValueError(f"window nonfinite {name}")
        for name in ("output", "residual", "route_weights"):
            if np.any(values[name][count:] != 0):
                raise ValueError("window padded output changed")
        routes, weights = values["routes"][:count], values["route_weights"][:count]
        if (
            np.any(routes < 0)
            or np.any(routes >= 256)
            or any(len(set(row)) != 8 for row in routes)
            or np.any(weights < 0)
            or not np.allclose(weights.sum(1), 1, rtol=0, atol=1e-6)
        ):
            raise ValueError("window routing invalid")
    for name in ("positions", "counts"):
        if not equal_bytes(actual[name], control[name]):
            raise ValueError("window/control ordered DSA selection differs")
    if not equal_bytes(actual["routes"][:count], control["routes"][:count]):
        raise ValueError("window/control ordered routes differ")

    def bounded(a: np.ndarray, b: np.ndarray, route: bool = False) -> dict[str, Any]:
        tolerance = ROUTE_WEIGHT_TOLERANCE if route else REAL_LAYER_OUTPUT_TOLERANCE
        aggregate = compare_bounded_tensor(a, b, tolerance)
        per_row = [
            compare_bounded_tensor(a[i : i + 1], b[i : i + 1], tolerance)
            for i in range(len(a))
        ]
        return dict(
            passed=aggregate["passed"] and all(r["passed"] for r in per_row),
            aggregate=aggregate,
            per_row=per_row,
        )

    comparisons = {
        name: bounded(
            actual[name][:count], control[name][:count], name == "route_weights"
        )
        for name in ("output", "residual", "route_weights")
    }
    addresses = written_addresses(slot, offset, count)
    mask = np.zeros((8, 64), np.bool_)
    for _, p, r in addresses:
        mask[p, r] = True
    for name in ("kv", "index", "repair"):
        if (
            initial[name].shape != actual[name].shape
            or initial[name].dtype != BF16
            or not np.isfinite(initial[name]).all()
        ):
            raise ValueError("window initial cache evidence differs")
        for values in (actual, control):
            if not equal_bytes(values[name][~mask], initial[name][~mask]):
                raise ValueError(f"window untouched {name} changed")
            if name == "kv" and np.any(values[name][mask, 576:] != 0):
                raise ValueError("window KV structural padding changed")
        comparisons[name] = bounded(actual[name][mask], control[name][mask])
    return dict(
        passed=all(c["passed"] for c in comparisons.values()),
        comparisons=comparisons,
        written_rows_on_owner=len(addresses),
        protocol=PROTOCOL,
        dsa_scope="OWN_SELECTED_SCORE_ORDER_AND_COMPETITIVE_ORDERED_CONTROL_AGREEMENT_NOT_FULL_CANONICAL_ROW",
        state_scope="REAL_LAYER6_WEIGHTS_SYNTHETIC_HISTORY_NOT_AUTHENTICATED_RESUME",
        performance_claim=False,
    )


def replay_case(
    path: Path, *, case: str, slots_by_device: Mapping[int, int]
) -> dict[str, Any]:
    """Recompute decisions from original arrays, not worker verdicts.

    Local controller/worker identity is supplied by the authenticated fleet
    envelope. It must cover four distinct physical owners before calling here.
    """
    from glm_tpu.greenfield.kernels.reference.rotary import build_rotary_table_host

    if (
        len(slots_by_device) != 4
        or len(set(slots_by_device.values())) != 4
        or any(
            type(d) is not int or type(s) is not int or not 0 <= s < 32
            for d, s in slots_by_device.items()
        )
    ):
        raise ValueError("window replay needs four distinct authenticated owners")
    canonical = host_case(
        case, build_rotary_table_host(CAPACITY, rotary_dim=64, theta=8e6)
    )
    with np.load(path, allow_pickle=False) as arrays:
        inputs = decode_arrays(arrays, "input", INPUT_FIELDS)
        if any(not equal_bytes(inputs[n], canonical[n]) for n in INPUT_FIELDS):
            raise ValueError("window original inputs differ from fixed protocol")
        expected = {f"input__{n}" for n in INPUT_FIELDS}
        owners = {}
        for device, slot in slots_by_device.items():
            values = {}
            for kind in ("actual", "control"):
                prefix = f"{kind}_{device}"
                expected.update(f"{prefix}__{n}" for n in FIELDS)
                values[kind] = decode_arrays(arrays, prefix, FIELDS)
            owners[str(device)] = compare_case(
                values["actual"],
                values["control"],
                owner_inputs(inputs, slot),
                slot=slot,
                case=case,
            )
        if set(arrays.files) != expected:
            raise ValueError("window original array inventory differs")
    return dict(
        passed=all(c["passed"] for c in owners.values()),
        owners=owners,
        input_sha256=input_hashes(inputs),
        protocol=PROTOCOL,
    )
