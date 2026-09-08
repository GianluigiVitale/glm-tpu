"""Original-array contract for B128 versus B32 suffixes on shared prefixes.

This checks the suffix realization, NOT independent full-layer DSA exactness.
No timing promotion or launcher is provided. Historical comparisons are intact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.greenfield import prefill_window_protocol as window
from scripts.greenfield.prefill_layer_evidence import (
    INPUT_FIELDS,
    decode_arrays,
    equal_bytes,
    input_hashes,
    owner_inputs,
)
from scripts.greenfield.prefill_layer_numerical import FIELDS

PROTOCOL = "ws32-prefill-layer6-completed-prefix-suffix128-control4x32-v1"
KERNEL = "ws32_prefill_completed_window_numerical"
REFERENCE_SCOPE = "SHARED_COMPLETED_PREFIX_SUFFIX_ONLY_NOT_INDEPENDENT_FULL_LAYER_DSA"
PREFIX_FIELDS = (
    "mlp_input",
    "residual",
    "kv",
    "index",
    "repair",
    "positions",
    "counts",
    "scores",
    "health",
    "normalized",
)
SUFFIX_FIELDS = ("output", "routes", "route_weights", "health")
BF16_FIELDS = {"mlp_input", "residual", "kv", "index", "repair", "normalized", "output"}
CASES = tuple(window.CASES)


def observe(
    values: tuple[Any, ...], fields: Sequence[str]
) -> dict[int, dict[str, np.ndarray]]:
    """Only addressable shards, preserving actual completed executable outputs."""
    if tuple(fields) not in (PREFIX_FIELDS, SUFFIX_FIELDS) or len(values) != len(
        fields
    ):
        raise ValueError("completed output inventory differs")
    result: dict[int, dict[str, np.ndarray]] = {}
    owners = None
    for name, value in zip(fields, values, strict=True):
        current = set()
        for shard in value.addressable_shards:
            device = int(shard.device.id)
            if device in current:
                raise ValueError("duplicate completed output owner")
            current.add(device)
            array = np.asarray(shard.data).copy()
            if name in ("kv", "index", "repair"):
                array = array[0]
            elif name == "health":
                array = array[0, 0]
            result.setdefault(device, {})[name] = array
        if owners is not None and current != owners:
            raise ValueError("completed output leaves have different owners")
        owners = current
    return result


def encode(prefix: str, values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        f"{prefix}__{name}": (
            value.view(np.uint16) if value.dtype == window.BF16 else value
        )
        for name, value in values.items()
    }


def _decode(arrays: Any, prefix: str, fields: Sequence[str]) -> dict[str, np.ndarray]:
    result = {}
    for name in fields:
        value = arrays[f"{prefix}__{name}"]
        if name in BF16_FIELDS:
            if value.dtype != np.uint16:
                raise ValueError("completed BF16 storage dtype differs")
            value = value.view(window.BF16)
        result[name] = value
    return result


def assemble(
    prefixes: Sequence[Mapping[str, np.ndarray]], suffix: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Independent host reconstruction of the published12-field proposal."""
    if (
        len(prefixes) != 4
        or any(set(p) != set(PREFIX_FIELDS) for p in prefixes)
        or set(suffix) != set(SUFFIX_FIELDS)
    ):
        raise ValueError("completed component inventory differs")
    result = dict(suffix)
    for name in ("kv", "index", "repair"):
        result[name] = prefixes[-1][name]
    for name in ("residual", "positions", "counts", "scores", "normalized"):
        result[name] = np.concatenate([p[name] for p in prefixes])
    return result


def _prefix_contract(values: Mapping[str, np.ndarray]) -> None:
    expected = {
        "mlp_input": ((32, 1536), window.BF16),
        "residual": ((32, 1536), window.BF16),
        "kv": ((8, 64, 640), window.BF16),
        "index": ((8, 64, 128), window.BF16),
        "repair": ((8, 64, 128), window.BF16),
        "positions": ((32, 2048), np.int32),
        "counts": ((32,), np.int32),
        "scores": ((32, 2048), np.float32),
        "health": ((32,), np.bool_),
        "normalized": ((32, 1536), window.BF16),
    }
    if set(values) != set(expected) or any(
        values[n].shape != shape or values[n].dtype != dtype
        for n, (shape, dtype) in expected.items()
    ):
        raise ValueError("completed prefix geometry/dtype differs")
    if not values["health"].all() or any(
        not np.isfinite(values[n]).all() for n in BF16_FIELDS if n in values
    ):
        raise ValueError("completed prefix health/nonfinite operand")


def replay_case(
    path: Path, *, case: str, slots_by_device: Mapping[int, int]
) -> dict[str, Any]:
    """Rebuild candidate/control from their actual component bytes before comparing."""
    from glm_tpu.greenfield.kernels.reference.rotary import build_rotary_table_host

    if (
        case not in CASES
        or len(slots_by_device) != 4
        or len(set(slots_by_device.values())) != 4
        or any(
            type(d) is not int or type(s) is not int or not 0 <= s < 32
            for d, s in slots_by_device.items()
        )
    ):
        raise ValueError("completed replay requires case and four authenticated owners")
    canonical = window.host_case(
        case, build_rotary_table_host(window.CAPACITY, rotary_dim=64, theta=8e6)
    )
    with np.load(path, allow_pickle=False) as arrays:
        initial = decode_arrays(arrays, "input", INPUT_FIELDS)
        if any(not equal_bytes(initial[n], canonical[n]) for n in INPUT_FIELDS):
            raise ValueError("completed original fixture differs")
        expected = {f"input__{n}" for n in INPUT_FIELDS}
        owners = {}
        for device, slot in slots_by_device.items():

            def read(kind, fields):
                key = f"{kind}_{device}"
                expected.update(f"{key}__{n}" for n in fields)
                return _decode(arrays, key, fields)

            prefixes = [read(f"prefix{t}", PREFIX_FIELDS) for t in range(4)]
            for p in prefixes:
                _prefix_contract(p)
            wide = read("wide", SUFFIX_FIELDS)
            narrow = [read(f"narrow{t}", SUFFIX_FIELDS) for t in range(4)]
            for p in narrow:
                if any(
                    p[n].shape != (32, *wide[n].shape[1:])
                    or p[n].dtype != wide[n].dtype
                    for n in SUFFIX_FIELDS
                ):
                    raise ValueError("completed narrow suffix geometry differs")
            reference_suffix = {
                n: np.concatenate([p[n] for p in narrow]) for n in SUFFIX_FIELDS
            }
            actual = assemble(prefixes, wide)
            control = assemble(prefixes, reference_suffix)
            # Device assembly is an output too: do not silently replace it with host reconstruction.
            for kind, rebuilt in (("actual", actual), ("control", control)):
                saved = read(kind, FIELDS)
                if any(not equal_bytes(saved[n], rebuilt[n]) for n in FIELDS):
                    raise ValueError(
                        "completed device assembly differs from original components"
                    )
            comparison = window.compare_case(
                actual, control, owner_inputs(initial, slot), slot=slot, case=case
            )
            comparison.update(
                protocol=PROTOCOL,
                dsa_scope="SHARED_PREFIX_BY_CONSTRUCTION_NOT_INDEPENDENT_CANONICAL_SELECTION",
            )
            owners[str(device)] = comparison
        if set(arrays.files) != expected:
            raise ValueError("completed original array inventory differs")
    return dict(
        passed=all(r["passed"] for r in owners.values()),
        owners=owners,
        protocol=PROTOCOL,
        reference_scope=REFERENCE_SCOPE,
        input_sha256=input_hashes(initial),
        independent_full_layer_admission=False,
        performance_claim=False,
    )
