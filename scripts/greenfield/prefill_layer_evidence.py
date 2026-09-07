"""Original-array protocol for the bounded full-layer TPU discriminator.

No JAX import/initialization. Synthetic inputs are reproducible, while actual
device observations remain mandatory. Neither this protocol nor its scalar
reference certifies the promoted decoder or a performance improvement.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import ml_dtypes
import numpy as np

from scripts.greenfield.prefill_layer_numerical import (
    CASES,
    FIELDS,
    PAGE_TABLE,
    ROWS,
    compare_layer_case,
    written_addresses,
)

BF16 = ml_dtypes.bfloat16
INPUT_FIELDS = (
    "update",
    "residual",
    "kv",
    "index",
    "repair",
    "positions",
    "counts",
    "scores",
    "offset",
    "count",
    "table",
    "health",
    "rope",
)
BF16_FIELDS = {
    "update",
    "output",
    "residual",
    "kv",
    "index",
    "repair",
    "normalized",
    "rope",
    "m64",
}
METAMORPHIC = {
    "boundary": (
        "future",
        "repair_history",
        "incoming_health",
        "bad_offset",
        "bad_page",
    ),
    "tail": ("clean_padding",),
    "empty": (),
}


def host_case(case: str, rope: np.ndarray) -> dict[str, np.ndarray]:
    """Fixed synthetic split residuals and populated prefix, global host layout.

    RNG/seed and addresses are independent of JAX cache writers. Slots are
    expert*4 + feature; feature owners replicate caches. Prefix values are nonzero
    and distinct from future zeros, including the separate repaired history.
    """
    if case not in CASES or rope.shape != (1024, 64) or rope.dtype != BF16:
        raise ValueError("unregistered case/rotary table")
    offset, count = CASES[case]
    rng = np.random.RandomState(20260907)
    result = {
        "update": (rng.standard_normal((ROWS, 6144)) * 0.01).astype(BF16),
        "residual": (rng.standard_normal((ROWS, 6144)) * 0.01).astype(BF16),
        "kv": np.zeros((8, 2, 64, 640), BF16),
        "index": np.zeros((8, 2, 64, 128), BF16),
        "repair": np.zeros((8, 2, 64, 128), BF16),
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
        address = (owner, PAGE_TABLE[page], row)
        result["kv"][address][:576] = (rng.standard_normal(576) * 0.05).astype(BF16)
        for name in ("index", "repair"):
            result[name][address] = (rng.standard_normal(128) * 0.05).astype(BF16)
    for row in range(count):
        length = offset + row + 1
        result["positions"][row, :length] = np.arange(length, dtype=np.int32)
        result["counts"][row] = length
        result["scores"][row, :length] = 0  # Canonical equal-score IndexShare fixture.
    for name in ("update", "residual"):
        result[name][count:] = np.nan
    return result


def mutate_case(inputs: Mapping[str, np.ndarray], kind: str) -> dict[str, np.ndarray]:
    """One intervention at a time, with original inputs retained separately."""
    result = {name: value.copy() for name, value in inputs.items()}
    if kind == "future":
        result["update"][8] = result["update"][8].astype(np.float32) + 0.5
    elif kind == "repair_history":
        for position in range(int(inputs["offset"])):
            page, within = divmod(position, 512)
            owner, row = divmod(within, 64)
            result["repair"][owner, PAGE_TABLE[page], row] = 0.75
    elif kind == "incoming_health":
        result["health"][:, :, 0] = False
    elif kind == "bad_offset":
        result["offset"] = np.asarray(-1, np.int32)
    elif kind == "bad_page":
        result["table"] = np.asarray([[1, 1]], np.int32)
    elif kind == "clean_padding":
        for name in ("update", "residual"):
            result[name][int(inputs["count"]) :] = 0
    else:
        raise ValueError("unregistered layer intervention")
    return result


def owner_inputs(inputs: Mapping[str, np.ndarray], slot: int) -> dict[str, np.ndarray]:
    if type(slot) is not int or not 0 <= slot < 32:
        raise ValueError("invalid physical owner")
    result = dict(inputs)
    for name in ("update", "residual"):
        result[name] = inputs[name][:, (slot % 4) * 1536 : (slot % 4 + 1) * 1536]
    for name in ("kv", "index", "repair"):
        result[name] = inputs[name][slot // 4]
    result["health"] = inputs["health"][slot // 4, slot % 4]
    return result


def local_observations(values: tuple[Any, ...]) -> dict[int, dict[str, np.ndarray]]:
    """Read only addressable shards; never fetch a distributed global tensor."""
    if len(values) != len(FIELDS):
        raise ValueError("layer returned a different field inventory")
    result: dict[int, dict[str, np.ndarray]] = {}
    devices = None
    for name, value in zip(FIELDS, values, strict=True):
        current = set()
        for shard in value.addressable_shards:
            device = int(shard.device.id)
            if device in current:
                raise ValueError("duplicate local device observation")
            current.add(device)
            array = np.asarray(shard.data).copy()
            if name in ("kv", "index", "repair"):
                array = array[0]
            elif name == "health":
                array = array[0, 0]
            result.setdefault(device, {})[name] = array
        if devices is not None and current != devices:
            raise ValueError("output leaves have different physical owners")
        devices = current
    return result


def stack_reference(rows: list[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Keep the last carried cache; concatenate row outputs and canonical padding."""
    if not 1 <= len(rows) <= ROWS or any(set(row) != set(FIELDS) for row in rows):
        raise ValueError("incomplete scalar reference observations")
    result = {}
    for name in FIELDS:
        if name in ("kv", "index", "repair"):
            result[name] = rows[-1][name].copy()
            continue
        fill = (
            True
            if name == "health"
            else (
                -np.inf
                if name == "scores"
                else -1 if name in ("positions", "routes") else 0
            )
        )
        first = rows[0][name]
        if any(
            row[name].shape != first.shape or row[name].dtype != first.dtype
            for row in rows
        ):
            raise ValueError("scalar output geometry changed")
        if first.shape[0] != 1:
            raise ValueError("reference is not one live row")
        result[name] = np.full((ROWS, *first.shape[1:]), fill, first.dtype)
        result[name][: len(rows)] = np.concatenate([row[name] for row in rows])
    return result


def equal_bytes(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def check_intervention(
    baseline: Mapping[str, np.ndarray],
    observed: Mapping[str, np.ndarray],
    inputs: Mapping[str, np.ndarray],
    *,
    slot: int,
    kind: str,
) -> dict[str, Any]:
    """Exact metamorphic checks, separate from cross-implementation tolerance."""
    if set(baseline) != set(FIELDS) or set(observed) != set(FIELDS):
        raise ValueError("intervention fields differ")
    if any(
        baseline[n].shape != observed[n].shape or baseline[n].dtype != observed[n].dtype
        for n in FIELDS
    ):
        raise ValueError("intervention shape/dtype differs")
    own = owner_inputs(inputs, slot)
    checks = {}
    if kind in ("bad_offset", "bad_page"):
        checks["health_refused"] = not observed["health"].any()
        for name in ("kv", "index", "repair"):
            checks[name + "_no_write"] = equal_bytes(observed[name], own[name])
    elif kind == "incoming_health":
        expected = baseline["health"].copy()
        expected[0] = False
        checks["health_propagated"] = equal_bytes(observed["health"], expected)
        for name in FIELDS:
            if name != "health":
                checks[name] = equal_bytes(baseline[name], observed[name])
    elif kind in ("future", "repair_history", "clean_padding"):
        for name in FIELDS:
            if name in ("kv", "index", "repair"):
                if kind == "future":
                    addresses = written_addresses(
                        slot=slot, offset=int(inputs["offset"]), count=8
                    )
                    checks[name] = all(
                        equal_bytes(baseline[name][p, r], observed[name][p, r])
                        for _, p, r in addresses
                    )
                elif kind == "repair_history" and name == "repair":
                    expected = own[name].copy()
                    for _, p, r in written_addresses(
                        slot=slot,
                        offset=int(inputs["offset"]),
                        count=int(inputs["count"]),
                    ):
                        expected[p, r] = baseline[name][p, r]
                    checks[name] = equal_bytes(expected, observed[name])
                else:
                    checks[name] = equal_bytes(baseline[name], observed[name])
            else:
                stop = 8 if kind == "future" else ROWS
                checks[name] = equal_bytes(baseline[name][:stop], observed[name][:stop])
        checks["healthy"] = bool(observed["health"].all())
    else:
        raise ValueError("unregistered layer intervention")
    return {"passed": all(checks.values()), "checks": checks}


def encode_arrays(
    prefix: str, values: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}__{name}": value.view(np.uint16) if value.dtype == BF16 else value
        for name, value in values.items()
    }


def decode_arrays(
    arrays: Mapping[str, np.ndarray], prefix: str, fields: tuple[str, ...]
) -> dict[str, np.ndarray]:
    result = {}
    for name in fields:
        value = arrays[f"{prefix}__{name}"]
        if name in BF16_FIELDS:
            if value.dtype != np.uint16:
                raise ValueError("original BF16 storage dtype differs")
            value = value.view(BF16)
        result[name] = value
    return result


def input_hashes(inputs: Mapping[str, np.ndarray]) -> dict[str, str]:
    return {name: sha256(value.tobytes()).hexdigest() for name, value in inputs.items()}


def replay_case(
    path: Path, *, layer: int, case: str, slots_by_device: Mapping[int, int]
) -> dict[str, Any]:
    """Worker and controller use the same fixed comparator on ORIGINAL NPZs."""
    with np.load(path, allow_pickle=False) as arrays:
        inputs = decode_arrays(arrays, "input", INPUT_FIELDS)
        from glm_tpu.greenfield.kernels.reference.rotary import build_rotary_table_host

        canonical = host_case(
            case, build_rotary_table_host(1024, rotary_dim=64, theta=8e6)
        )
        if any(not equal_bytes(inputs[n], canonical[n]) for n in INPUT_FIELDS):
            raise ValueError("original synthetic inputs differ from fixed protocol")
        expected = {f"input__{n}" for n in INPUT_FIELDS}
        results = {}
        for device, slot in slots_by_device.items():
            values = {}
            for kind in ("actual", "reference", *METAMORPHIC[case]):
                prefix = f"{kind}_{device}"
                expected.update(f"{prefix}__{n}" for n in FIELDS)
                values[kind] = decode_arrays(arrays, prefix, FIELDS)
            keys = {}
            for kind in ("actual", "reference"):
                if layer == 0:
                    prefix = f"{kind}_{device}"
                    expected.add(f"{prefix}__m64")
                    keys[kind] = decode_arrays(arrays, prefix, ("m64",))["m64"]
                else:
                    keys[kind] = None
            comparison = compare_layer_case(
                values["actual"],
                values["reference"],
                owner_inputs(inputs, slot),
                slot=slot,
                layer=layer,
                case=case,
                actual_m64_keys=keys["actual"],
                reference_m64_keys=keys["reference"],
            )
            interventions = {
                kind: check_intervention(
                    values["actual"],
                    values[kind],
                    mutate_case(inputs, kind),
                    slot=slot,
                    kind=kind,
                )
                for kind in METAMORPHIC[case]
            }
            results[str(device)] = {
                "comparison": comparison,
                "interventions": interventions,
                "passed": comparison["passed"]
                and all(v["passed"] for v in interventions.values()),
            }
        if set(arrays.files) != expected:
            raise ValueError("original layer array inventory differs")
    return {
        "passed": bool(results) and all(r["passed"] for r in results.values()),
        "owners": results,
        "input_sha256": input_hashes(inputs),
    }
