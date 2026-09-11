"""Read history observations from authenticated addressable owners only.

Adapt the dense-frontier owner's index validation to the history boundary
schema. A host may own only some hidden-feature slices, with local replicas of
those slices. No operation here gathers a distributed array onto the host.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from scripts.greenfield.ws32_history_frontier import LAYERS
from scripts.greenfield.ws32_prefill_frontier_state import _index_key

FIELDS = ("update", "residual", "normalized_input", "route_ids", "route_weights")
FEATURE_FIELDS = FIELDS[:3]
DTYPES = ("bfloat16", "bfloat16", "bfloat16", "int32", "float32", "bool")


class ReplicaMismatch(ValueError):
    """A terminal replica disagreement carrying the bounded conflicting pair.

    Geometry has already passed before these payloads are read. The worker can
    therefore retain ``originals`` with ``metadata`` before refusing the call,
    without presenting either disagreeing replica as an accepted observation.
    """

    def __init__(
        self, context: str, first: tuple[Any, np.ndarray],
        conflicting: tuple[Any, np.ndarray], *, global_shape: tuple[int, ...],
        local_slots: Mapping[int, int],
    ) -> None:
        super().__init__(f"history capture inconsistent local replica bytes: {context}")
        self.originals = {}
        replicas = {}
        for name, (shard, value) in zip(("replica0", "replica1"), (first, conflicting), strict=True):
            self.originals[name] = np.array(value, copy=True, order="C")
            replicas[name] = dict(
                device_id=shard.device.id, slot=local_slots[shard.device.id],
                index=[list(part) for part in _index_key(shard.index, global_shape)],
            )
        self.metadata = dict(
            context=context, global_shape=list(global_shape),
            shape=list(first[1].shape), dtype=str(first[1].dtype), replicas=replicas,
        )


def _require_owners(
    local_slots: Mapping[int, int], process_index: int, platform: str,
) -> None:
    if (platform not in ("cpu", "tpu") or type(process_index) is not int
            or process_index < 0 or not 1 <= len(local_slots) <= 32
            or (platform == "tpu" and len(local_slots) != 4)
            or len(set(local_slots.values())) != len(local_slots)
            or any(type(device) is not int or device < 0 or type(slot) is not int
                   or not 0 <= slot < 32 for device, slot in local_slots.items())):
        raise ValueError("history capture owner map or runtime identity differs")


def _owner_shards(
    array: Any, *, shape: tuple[int, ...], dtype: str,
    index_for_slot: Callable[[int], tuple[slice, ...]],
    local_slots: Mapping[int, int], process_index: int, platform: str,
) -> dict[int, Any]:
    if tuple(array.shape) != shape or str(array.dtype) != dtype:
        raise ValueError("history capture global shape/dtype differs")
    owners = {}
    for shard in array.addressable_shards:
        device = shard.device
        if (type(device.id) is not int or device.id not in local_slots
                or device.id in owners or device.platform != platform
                or type(device.process_index) is not int
                or device.process_index != process_index):
            raise ValueError("history capture actual physical owner differs")
        expected = _index_key(index_for_slot(local_slots[device.id]), shape)
        if _index_key(shard.index, shape) != expected:
            raise ValueError("history capture actual shard index differs")
        local_shape = tuple(len(range(*part)) for part in expected)
        if tuple(shard.data.shape) != local_shape or str(shard.data.dtype) != dtype:
            raise ValueError("history capture local shard shape/dtype differs")
        owners[device.id] = shard
    if set(owners) != set(local_slots):
        raise ValueError("history capture missing addressable owner")
    return owners


def _read(shard: Any) -> np.ndarray:
    value = np.asarray(shard.data)
    if value.shape != tuple(shard.data.shape) or str(value.dtype) != str(shard.data.dtype):
        raise ValueError("history capture actual host shape/dtype differs")
    # Preserve scalar rank as well as the exact bytes of BF16, NaNs and -0.
    return np.array(value, copy=True, order="C")


def _same_bytes(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _replica(
    owners: Mapping[int, Any], *, shape: tuple[int, ...],
    local_slots: Mapping[int, int], context: str,
) -> np.ndarray:
    first = None
    first_shard = None
    for device in sorted(owners):
        value = _read(owners[device])
        if first is not None and not _same_bytes(first, value):
            raise ReplicaMismatch(context, (first_shard, first), (owners[device], value),
                                  global_shape=shape, local_slots=local_slots)
        if first is None:
            first = value
            first_shard = owners[device]
    if first is None:
        raise ValueError("history capture has no local replica")
    return first


def replicated_host(
    array: Any, *, local_slots: Mapping[int, int], process_index: int, platform: str,
    context: str = "replicated",
) -> np.ndarray:
    """Return one P() replica after validating every actual local owner and byte.

    The caller binds the scalar/DSA field's expected global shape and dtype.
    This reader requires every shard to cover the complete declared array; a
    feature-sharded array cannot be accidentally read as a full replica.
    A byte disagreement raises ``ReplicaMismatch`` with the conflicting pair;
    ``context`` identifies the scalar/DSA field in its preservation metadata.
    """
    _require_owners(local_slots, process_index, platform)
    shape, dtype = tuple(array.shape), str(array.dtype)
    if (any(type(size) is not int or size < 0 for size in shape)
            or dtype not in DTYPES):
        raise ValueError("history replicated array geometry differs")
    owners = _owner_shards(array, shape=shape, dtype=dtype,
        index_for_slot=lambda _: (slice(None),) * len(shape),
        local_slots=local_slots, process_index=process_index, platform=platform)
    return _replica(owners, shape=shape, local_slots=local_slots, context=context)


def capture_boundaries(
    result: Any, count: int, *, local_slots: Mapping[int, int],
    process_index: int, platform: str,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any], list[str]]:
    """Capture live rows and return numerical failures for preservation first.

    Feature fields concatenate UNIQUE locally owned slices in global-column
    order; local replicas must agree byte-for-byte before being deduplicated.
    Routes are full replicas. ``health`` is [live rows, local owner count], in
    the increasing-slot order in ``layout['health_slots']``. Layout is unchanged
    across physical B128/B114 and one-row observer calls. Invalid geometry
    raises; conflicting replicas raise ``ReplicaMismatch`` carrying their pair.
    Nonfinite live streams and false live health return their original arrays
    with errors, for the caller to save and refuse.
    """
    _require_owners(local_slots, process_index, platform)
    if (type(count) is not int or count <= 0
            or len(result.boundaries) != len(LAYERS)):
        raise ValueError("history boundary capture geometry differs")
    shape = tuple(result.boundaries[0].update.shape)
    if (len(shape) != 2 or any(type(size) is not int for size in shape)
            or shape[0] not in (1, 114, 128) or not count <= shape[0]
            or shape[1] <= 0 or shape[1] % 4):
        raise ValueError("history boundary row/feature geometry differs")
    physical_rows, hidden = shape
    width = hidden // 4
    features = sorted({slot % 4 for slot in local_slots.values()})
    columns = [column for feature in features
               for column in range(feature * width, (feature + 1) * width)]
    ordered_owners = sorted(local_slots, key=local_slots.__getitem__)
    layout = dict(
        schema_version=1, process_index=process_index, platform=platform,
        local_slots={str(device): slot for device, slot in sorted(local_slots.items())},
        global_hidden_size=hidden, feature_columns=columns,
        health_slots=[local_slots[device] for device in ordered_owners],
        owners={str(local_slots[device]): dict(
            device_id=device, slot=local_slots[device],
            feature_columns=list(range((local_slots[device] % 4) * width,
                                       (local_slots[device] % 4 + 1) * width)),
            capture_columns=list(range(features.index(local_slots[device] % 4) * width,
                                       (features.index(local_slots[device] % 4) + 1) * width)),
        ) for device in ordered_owners},
    )
    shapes = (shape, shape, shape, (physical_rows, 8), (physical_rows, 8),
              (8, 4, physical_rows))
    validated = {}
    # Authenticate every boundary leaf before reading any payload.
    for layer, boundary in zip(LAYERS, result.boundaries, strict=True):
        if len(boundary) != len(DTYPES):
            raise ValueError("history boundary field inventory differs")
        validated[layer] = {}
        for field, expected_shape, dtype in zip((*FIELDS, "health"), shapes, DTYPES, strict=True):
            def index(slot: int, field=field, expected_shape=expected_shape) -> tuple[slice, ...]:
                if field in FEATURE_FIELDS:
                    return (slice(None), slice((slot % 4) * width, (slot % 4 + 1) * width))
                if field == "health":
                    return (slice(slot // 4, slot // 4 + 1),
                            slice(slot % 4, slot % 4 + 1), slice(None))
                return (slice(None),) * len(expected_shape)

            validated[layer][field] = _owner_shards(
                getattr(boundary, field), shape=expected_shape, dtype=dtype,
                index_for_slot=index, local_slots=local_slots,
                process_index=process_index, platform=platform)
    rows, errors = {}, []
    for layer in LAYERS:
        saved = {}
        for field in FIELDS:
            owners = validated[layer][field]
            if field in FEATURE_FIELDS:
                values, first_owners = {}, {}
                for device in ordered_owners:
                    feature = local_slots[device] % 4
                    value = _read(owners[device])
                    if feature in values and not _same_bytes(values[feature], value):
                        raise ReplicaMismatch(f"layer{layer}/{field}",
                            (first_owners[feature], values[feature]), (owners[device], value),
                            global_shape=shape, local_slots=local_slots)
                    values.setdefault(feature, value)
                    first_owners.setdefault(feature, owners[device])
                saved[field] = np.concatenate([values[feature][:count] for feature in features], axis=1)
            else:
                saved[field] = _replica(owners, shape=shapes[FIELDS.index(field)],
                    local_slots=local_slots, context=f"layer{layer}/{field}")[:count].copy()
            if field != "route_ids" and not np.isfinite(saved[field]).all():
                errors.append(f"layer{layer}/{field}/nonfinite")
        health = []
        for device in ordered_owners:
            value = _read(validated[layer]["health"][device])[0, 0, :count]
            health.append(value)
            if not value.all():
                errors.append(f"layer{layer}/slot{local_slots[device]}/health")
        saved["health"] = np.stack(health, axis=1)
        rows[layer] = saved
    return rows, layout, errors
