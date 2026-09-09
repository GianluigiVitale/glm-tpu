"""Owner-local originals for the frozen first-window diagnostic, not a launcher.

The caller supplies the authenticated physical mesh and process. No global
device array is converted to NumPy; read one addressable cache shard at a time.
Malformed numerical state is returned with its originals for publication BEFORE
the existing worker's matched refusal vote. Production shape binding is strict.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping

import numpy as np

from scripts.greenfield.ws32_prefill_frontier import BF16, capture_cache_evidence

PROMPT_LENGTH = 8155
CAPACITY = 8192
INDEX_LAYERS = (0, 1, 2, *range(6, 78, 4))
METADATA = {
    "selected_positions": ((1, 2048), "int32"),
    "selected_valid_counts": ((1,), "int32"),
    "selected_scores": ((1, 2048), "float32"),
    "position": ((1,), "int32"),
    "block_tables": ((1, 16), "int32"),
    "context_lengths": ((1,), "int32"),
    "contract_valid": ((1,), "bool"),
    "prompt_length": ((), "int32"),
    "finished": ((), "bool"),
    "next_token": ((1,), "int32"),
}


def require_config(config: Any) -> None:
    if (
        config.context_capacity != CAPACITY or config.logical_page_size != 512
        or config.local_rows_per_page != 64
        or config.kv_cache_shape != (78, 16, 512, 640)
        or config.index_cache_shape != (21, 16, 512, 128)
        or config.full_index_slots != INDEX_LAYERS
        or config.geometry.hidden_size != 6144 or config.geometry.dsa_top_k != 2048
        or config.exact_dsa or config.strategy_nd_dense
        or not config.host_main_rope_table
    ):
        raise ValueError("first-window requires frozen production config")


def _digest(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _index_key(index: tuple[Any, ...], shape: tuple[int, ...]) -> tuple[Any, ...]:
    if len(index) != len(shape) or any(not isinstance(s, slice) for s in index):
        raise ValueError("first-window shard index must retain every dimension")
    return tuple(s.indices(size) for s, size in zip(index, shape))


def owner_shards(
    array: Any, *, shape: tuple[int, ...], dtype: str, cache: bool,
    local_slots: Mapping[int, int], process_index: int,
) -> dict[int, Any]:
    """Check actual shard indices against authenticated expert*4+feature slots."""
    if (
        len(local_slots) != 4 or len(set(local_slots.values())) != 4
        or any(type(d) is not int or type(s) is not int or not 0 <= s < 32
               for d, s in local_slots.items())
        or tuple(array.shape) != shape or str(array.dtype) != dtype
    ):
        raise ValueError("first-window array geometry or owner map differs")
    output = {}
    for shard in array.addressable_shards:
        device = shard.device
        if (device.id not in local_slots or device.id in output
                or device.process_index != process_index or device.platform != "tpu"):
            raise ValueError("first-window actual device owner differs")
        expected = [slice(None)] * len(shape)
        if cache:
            expert = local_slots[device.id] // 4
            expected[2] = slice(expert * 64, (expert + 1) * 64)
        if _index_key(shard.index, shape) != _index_key(tuple(expected), shape):
            raise ValueError("first-window actual stripe ownership differs")
        local_shape = list(shape)
        if cache:
            local_shape[2] = 64
        if tuple(shard.data.shape) != tuple(local_shape) or str(shard.data.dtype) != dtype:
            raise ValueError("first-window local shard geometry differs")
        output[device.id] = shard
    if set(output) != set(local_slots):
        raise ValueError("first-window missing addressable owners")
    return output


def metadata_errors(values: Mapping[str, np.ndarray], *, frontier: int) -> list[str]:
    """Exact scheduling/empty-state contract; selection bytes remain originals."""
    if type(frontier) is not int or frontier not in (0, 32, 64, 96, 128):
        raise ValueError("first-window unregistered frontier")
    if set(values) != set(METADATA):
        raise ValueError("first-window metadata fields differ")
    for name, (shape, dtype) in METADATA.items():
        if values[name].shape != shape or str(values[name].dtype) != dtype:
            raise ValueError(f"first-window metadata shape/dtype differs: {name}")
    expected = dict(
        position=frontier, context_lengths=frontier+1, prompt_length=PROMPT_LENGTH,
        finished=False, contract_valid=True, next_token=-1,
        selected_valid_counts=frontier,
    )
    errors = [name for name, value in expected.items() if not (values[name] == value).all()]
    if not np.array_equal(values["block_tables"], np.arange(16, dtype=np.int32)[None]):
        errors.append("block_tables")
    positions, scores = values["selected_positions"][0], values["selected_scores"][0]
    if (not np.array_equal(np.sort(positions[:frontier]), np.arange(frontier))
            or not (positions[frontier:] == -1).all()
            or not np.isfinite(scores[:frontier]).all()
            or not np.isneginf(scores[frontier:]).all()):
        errors.append("selected_metadata")
    return errors


def capture_state(
    state: Any, *, next_token: Any | None, config: Any,
    local_slots: Mapping[int, int], process_index: int, frontier: int,
    caches: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return bounded originals and validation errors, without numerical promotion.

    Initial states retain hashes of verified zero caches, not redundant zero row
    payloads. Intermediate32/64/96 captures are metadata-only. A failed initial
    cache retains its first128 rows and bounded offending samples. Outer caller
    must write both arrays/report before raising on report['valid']==False.
    """
    require_config(config)
    if type(caches) is not bool or (caches and frontier not in (0, 128)):
        raise ValueError("cache capture only supports initial/final first-window")
    common = dict(local_slots=local_slots, process_index=process_index)
    fields = {name: getattr(state.decoder, name) for name in METADATA
              if name not in ("prompt_length", "finished", "next_token")}
    fields.update(prompt_length=state.prompt_length, finished=state.finished)
    if next_token is not None:
        fields["next_token"] = next_token
    elif frontier != 0:
        raise ValueError("completed first-window call must expose actual next_token")
    originals, owners, errors = {}, {}, []
    by_field = {
        name: owner_shards(value, shape=METADATA[name][0], dtype=METADATA[name][1],
                           cache=False, **common)
        for name, value in fields.items()
    }
    first_meta = None
    for device, slot in sorted(local_slots.items()):
        values = {name: np.asarray(shards[device].data).copy()
                  for name, shards in by_field.items()}
        if frontier == 0:
            values["next_token"] = np.asarray([-1], np.int32)
        problems = metadata_errors(values, frontier=frontier)
        hashes = {name: _digest(value) for name, value in values.items()}
        if first_meta is not None and hashes != first_meta:
            problems.append("local_metadata_replica_bytes")
        first_meta = hashes if first_meta is None else first_meta
        originals.update({f"slot{slot}_{name}": value for name, value in values.items()})
        owners[str(slot)] = dict(device_id=device, slot=slot, metadata_sha256=hashes, caches={})
        errors.extend(f"slot{slot}:{name}" for name in problems)
    if caches:
        for name, array, layers, shape in (
            ("kv", state.decoder.kv_cache_local, tuple(range(78)), config.kv_cache_shape),
            ("index", state.decoder.index_cache_local, INDEX_LAYERS, config.index_cache_shape),
            ("repair", state.repaired_index_local, INDEX_LAYERS, config.index_cache_shape),
        ):
            shards = owner_shards(array, shape=shape, dtype=str(BF16), cache=True, **common)
            replica_hashes = {}
            for device, slot in sorted(local_slots.items()):
                # One transient local leaf. Never np.asarray(distributed array).
                host = np.asarray(shards[device].data)
                rows, evidence = capture_cache_evidence(
                    host, slot=slot, block_table=np.arange(16, dtype=np.int32)[None],
                    layer_ids=layers, initial=frontier == 0,
                )
                del host
                owners[str(slot)]["caches"][name] = evidence
                if frontier != 0 or not evidence["valid"]:
                    originals[f"slot{slot}_{name}_rows"] = rows
                digest = evidence["cache"]["whole_cache_sha256"]
                if slot//4 in replica_hashes and replica_hashes[slot//4] != digest:
                    errors.append(f"slot{slot}:{name}:local_feature_replica_bytes")
                replica_hashes[slot//4] = digest
                if not evidence["valid"]:
                    errors.append(f"slot{slot}:{name}:cache_bytes")
    return originals, dict(
        schema_version=1, frontier=frontier, prompt_length=PROMPT_LENGTH,
        context_capacity=CAPACITY, owners=owners, errors=errors, valid=not errors,
        cache_scope="initial_zero" if frontier == 0 else "first128" if caches else "not_captured",
        numerical_promotion=False, performance_claim=False,
    )
