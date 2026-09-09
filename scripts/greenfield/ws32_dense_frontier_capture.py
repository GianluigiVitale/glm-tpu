"""Owner-local dense diagnostic output capture, not a global array readback."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from scripts.greenfield.prefill_layer_evidence import encode_arrays
from scripts.greenfield.prefill_layer_numerical import FIELDS
from scripts.greenfield.ws32_prefill_frontier_state import _index_key


def capture(
    result: tuple, *, local_slots: Mapping[int, int], process_index: int,
    count: int, keep_caches: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Preserve all live row outputs and full endpoint cache bits before refusal.

    Exact global/shard geometry and physical owner indices are checked before
    addressable-only reads. Health/numerical failures are RETURNED for archival.
    Non-endpoint caches stay device-resident and are not downloaded again.
    """
    if (count not in (32, 128) or type(count) is not int or type(keep_caches) is not bool
            or len(result) != 2 or len(local_slots) != 4
            or len(set(local_slots.values())) != 4
            or any(type(d) is not int or type(s) is not int or not 0 <= s < 32
                   for d, s in local_slots.items())):
        raise ValueError("dense capture requires two layers and four physical owners")
    shapes = ((128, 6144), (128, 6144), (8, 16, 64, 640), (8, 16, 64, 128),
              (8, 16, 64, 128), (128, 2048), (128,), (128, 2048), (128, 8),
              (128, 8), (8, 4, 128), (128, 6144))
    dtypes = ("bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16",
              "int32", "int32", "float32", "int32", "float32", "bool", "bfloat16")
    originals, errors = {}, []
    for layer_id, layer in enumerate(result):
        if len(layer) != len(FIELDS):
            raise ValueError("dense output inventory differs")
        for name, array, shape, dtype in zip(FIELDS, layer, shapes, dtypes, strict=True):
            if tuple(array.shape) != shape or str(array.dtype) != dtype:
                raise ValueError(f"dense output geometry differs: {layer_id}/{name}")
            observed = set()
            for shard in array.addressable_shards:
                device = shard.device
                if (device.id not in local_slots or device.id in observed
                        or device.platform != "tpu" or device.process_index != process_index):
                    raise ValueError("dense output physical owner differs")
                observed.add(device.id)
                slot = local_slots[device.id]
                index = [slice(None)] * len(shape)
                if name in ("kv", "index", "repair", "health"):
                    index[0] = slice(slot // 4, slot // 4 + 1)
                if name == "health":
                    index[1] = slice(slot % 4, slot % 4 + 1)
                if name in ("output", "residual", "normalized"):
                    index[1] = slice((slot % 4) * 1536, (slot % 4 + 1) * 1536)
                expected = _index_key(tuple(index), shape)
                if _index_key(shard.index, shape) != expected:
                    raise ValueError("dense output shard index differs")
                local_shape = tuple(len(range(*v)) for v in expected)
                if tuple(shard.data.shape) != local_shape or str(shard.data.dtype) != dtype:
                    raise ValueError("dense output local shape differs")
            if observed != set(local_slots):
                raise ValueError("dense output missing addressable owner")
        # Adapt the existing12-field addressable reader: skip intermediate cache
        # downloads rather than discard them after a needless device readback.
        observed = {d: {} for d in local_slots}
        for name, array in zip(FIELDS, layer, strict=True):
            if name in ("kv", "index", "repair") and not keep_caches:
                continue
            for shard in array.addressable_shards:
                value = np.asarray(shard.data).copy()
                if name in ("kv", "index", "repair"):
                    value = value[0]
                elif name == "health":
                    value = value[0, 0]
                observed[shard.device.id][name] = value
        for device, fields in observed.items():
            slot = local_slots[device]
            saved = {}
            for name, value in fields.items():
                if name in ("kv", "index", "repair"):
                    if keep_caches:
                        saved[name] = value
                    continue
                saved[name] = value if name == "health" else value[:count]
            if not fields["health"].all():
                errors.append(f"layer{layer_id}/slot{slot}/health")
            for name in ("output", "residual", "normalized", "route_weights"):
                if not np.isfinite(fields[name][:count]).all():
                    errors.append(f"layer{layer_id}/slot{slot}/{name}/nonfinite")
            if keep_caches:
                for name in ("kv", "index", "repair"):
                    if not np.isfinite(fields[name]).all():
                        errors.append(f"layer{layer_id}/slot{slot}/{name}/nonfinite")
            originals.update(encode_arrays(f"slot{slot}_layer{layer_id}", saved))
    return originals, dict(count=count, layers=[0, 1], local_slots=dict(local_slots),
                           keep_caches=keep_caches, valid=not errors, errors=errors,
                           numerical_promotion=False, performance_claim=False)


def cache_bits(arrays: Mapping[str, np.ndarray], slot: int) -> dict[str, np.ndarray]:
    """Stack the two original owner layers; never synthesize missing cache pages."""
    return {family: np.stack([arrays[f"slot{slot}_layer{layer}__{family}"]
                             for layer in (0, 1)]) for family in ("kv", "index", "repair")}
