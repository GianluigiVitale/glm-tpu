"""Bounded layer admission from retained final-owner files, not a model loader.

The complete manifest, SUCCESS and placement schema are authenticated. Only
selected layer payloads (and an explicitly opted-in embedding) are checked.
Unselected bytes and whole-file
digests are deliberately NOT verified, so these types cannot certify full state.
No checkpoint copy, full-file hash or load-then-filter is performed.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import re
import stat
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

from ..errors import CheckpointValidationError
from ..partitioning.source_inventory import SourceInventory
from ..types import ModelGeometry
from .ws32_runtime_checkpoint import (
    Ws32RuntimeFilePlan,
    Ws32RuntimeMetadata,
    Ws32RuntimeTensorPlan,
    _memory_stats,
    _read_ws32_runtime_metadata,
    _validate_finite_chunk,
)

_LAYER_NAME = re.compile(r"model\.layers\.(\d+)\.")


@dataclass(frozen=True, slots=True)
class Ws32LayerSubsetMetadata:
    """Metadata admission only; selected payloads are checked by the loader."""

    metadata: Ws32RuntimeMetadata
    layer_ids: tuple[int, ...]
    tensor_indices: tuple[int, ...]
    local_slots: tuple[int, ...]
    payload_bytes_per_chip: int
    include_embedding: bool = False


@dataclass(frozen=True, slots=True)
class LoadedWs32LayerSubset:
    arrays: Mapping[str, Any]
    layer_ids: tuple[int, ...]
    local_device_slots: tuple[Mapping[str, Any], ...]
    device_memory_before: tuple[Mapping[str, int] | None, ...]
    device_memory_after: tuple[Mapping[str, int] | None, ...]
    payload_bytes_per_chip: int
    include_embedding: bool = False

    @property
    def integrity_scope(self) -> str:
        if self.include_embedding:
            return "selected_layers_and_embedding_only_not_complete_checkpoint"
        return "selected_layer_tensors_only_not_complete_checkpoint"


def ws32_expected_layer_names(geometry: ModelGeometry, layer: int) -> frozenset[str]:
    """Raw-layout schema adapted from ws32_decoder_weight_names (no overlays)."""

    prefix = f"model.layers.{layer}"
    names = {
        f"{prefix}.{role}"
        for role in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_a_layernorm.weight",
            "self_attn.kv_a_layernorm.weight",
        )
    }
    projections = [
        f"self_attn.{role}"
        for role in (
            "q_a_proj",
            "kv_a_proj_with_mqa",
            "q_b_proj",
            "kv_b_proj",
            "o_proj",
        )
    ]
    if geometry.indexer_types[layer] == "full":
        projections.extend(("self_attn.indexer.wq_b", "self_attn.indexer.wk"))
        names.update(
            f"{prefix}.self_attn.indexer.{role}"
            for role in ("k_norm.weight", "k_norm.bias", "weights_proj.weight")
        )
    if geometry.mlp_layer_types[layer] == "dense":
        projections.extend(f"mlp.{role}_proj" for role in ("gate", "up", "down"))
    else:
        projections.extend(
            f"mlp.{kind}.{role}_proj"
            for kind in ("experts", "shared_experts")
            for role in ("gate", "up", "down")
        )
        names.update(
            f"{prefix}.mlp.gate.{role}"
            for role in ("weight", "e_score_correction_bias")
        )
    names.update(
        f"{prefix}.{projection}.{role}"
        for projection in projections
        for role in ("weight_bits", "scale_inv")
    )
    return frozenset(names)


def _checked_ids(values: Sequence[int], *, limit: int, label: str) -> tuple[int, ...]:
    result = tuple(values)
    if not result or any(
        type(value) is not int or not 0 <= value < limit for value in result
    ):
        raise ValueError(f"invalid WS32 subset {label}")
    if len(set(result)) != len(result):
        raise ValueError(f"duplicate WS32 subset {label}")
    return tuple(sorted(result))


def _check_owner_header(stream: BinaryIO, plan: Ws32RuntimeFilePlan) -> None:
    observed = os.fstat(stream.fileno())
    if not stat.S_ISREG(observed.st_mode) or observed.st_size != plan.file_bytes:
        raise CheckpointValidationError(
            f"WS32 subset file missing/truncated: {plan.filename}"
        )
    stream.seek(0)
    if stream.read(len(plan.header)) != plan.header:
        raise CheckpointValidationError(f"WS32 subset header drifted: {plan.filename}")


def _check_local_files(subset: Ws32LayerSubsetMetadata) -> None:
    owned = set(subset.local_slots)
    for plan in subset.metadata.plans:
        path = subset.metadata.root / plan.filename
        if plan.device_slot not in owned:
            if path.exists() or path.is_symlink():
                raise CheckpointValidationError(
                    f"WS32 subset foreign slot: {plan.filename}"
                )
            continue
        if not path.is_file() or path.is_symlink():
            raise CheckpointValidationError(
                f"WS32 subset file missing/truncated: {plan.filename}"
            )
        with path.open("rb") as stream:
            _check_owner_header(stream, plan)


def read_ws32_layer_subset_metadata(
    root: Path,
    *,
    layer_ids: Sequence[int],
    local_slots: Sequence[int],
    max_payload_bytes_per_chip: int,
    expected_manifest_sha256: str,
    expected_success_sha256: str,
    expected_mesh_hash: str,
    expected_topology_hash: str,
    inventory: SourceInventory,
    geometry: ModelGeometry,
    include_embedding: bool = False,
) -> Ws32LayerSubsetMetadata:
    """Authenticate metadata/headers and reject incomplete or over-budget layers.

    The caller supplies an explicit byte budget and physical local slots. The
    device loader additionally binds those slots to the actual addressable mesh.
    Payload finiteness and SHA verification occur during the selected-only read.
    include_embedding opts in the complete original embedding tensor; its bytes
    count against the same budget and its full selected-leaf checksum is checked.
    """

    if type(include_embedding) is not bool:
        raise ValueError("WS32 subset embedding selection must be explicit bool")
    layers = _checked_ids(layer_ids, limit=geometry.num_layers, label="layers")
    slots = _checked_ids(local_slots, limit=32, label="slots")
    if len(slots) not in (4, 32):
        raise ValueError("WS32 subset requires four local slots (32 for CPU tests)")
    if type(max_payload_bytes_per_chip) is not int or max_payload_bytes_per_chip <= 0:
        raise ValueError("WS32 subset requires a positive explicit payload budget")
    metadata = _read_ws32_runtime_metadata(
        root,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_success_sha256=expected_success_sha256,
        expected_mesh_hash=expected_mesh_hash,
        expected_topology_hash=expected_topology_hash,
        inventory=inventory,
        geometry=geometry,
    )
    schema = metadata.plans[0].tensors
    indices = tuple(
        index
        for index, tensor in enumerate(schema)
        if ((match := _LAYER_NAME.match(tensor.name)) is not None
            and int(match.group(1)) in layers)
        or (include_embedding and tensor.name == "model.embed_tokens.weight")
    )
    expected = frozenset().union(
        *(ws32_expected_layer_names(geometry, layer) for layer in layers)
    )
    if include_embedding:
        expected |= {"model.embed_tokens.weight"}
    if frozenset(schema[index].name for index in indices) != expected:
        raise CheckpointValidationError(
            "WS32 subset selected layer schema is incomplete or unexpected"
        )
    byte_count = sum(schema[index].byte_count for index in indices)
    if byte_count > max_payload_bytes_per_chip:
        raise CheckpointValidationError(
            "WS32 subset selected payload exceeds per-chip budget"
        )
    subset = Ws32LayerSubsetMetadata(
        metadata, layers, indices, slots, byte_count, include_embedding
    )
    _check_local_files(subset)
    return subset


def iter_ws32_layer_subset_host_tensors(
    subset: Ws32LayerSubsetMetadata,
) -> Iterator[tuple[int, Ws32RuntimeTensorPlan, tuple[tuple[int, Any, str], ...]]]:
    """Yield one checked leaf across local owners; never read unselected payload.

    Each host value is a view of the same immutable bytes whose digest and
    finiteness were checked. No model-sized host tree is accumulated. Consumers
    must discard yielded values before requesting another leaf to bound memory.
    """

    import ml_dtypes
    import numpy as np

    dtypes = {"BF16": ml_dtypes.bfloat16, "F32": np.dtype("<f4"), "U8": np.uint8}
    metadata = subset.metadata
    _check_local_files(subset)
    with ExitStack() as stack:
        streams = {}
        plans = {plan.device_slot: plan for plan in metadata.plans}
        for slot in subset.local_slots:
            plan = plans[slot]
            stream = stack.enter_context((metadata.root / plan.filename).open("rb"))
            _check_owner_header(stream, plan)
            streams[slot] = stream
        for index in subset.tensor_indices:
            tensor = metadata.plans[0].tensors[index]
            owners = []
            for slot in subset.local_slots:
                stream = streams[slot]
                stream.seek(len(plans[slot].header) + tensor.data_offset_start)
                raw = stream.read(tensor.byte_count)
                digest = sha256(raw).hexdigest()
                if (
                    len(raw) != tensor.byte_count
                    or digest != metadata.records_by_slot[slot]["tensor_sha256"][index]
                ):
                    raise CheckpointValidationError(
                        f"WS32 subset tensor checksum drifted: {tensor.name} slot {slot}"
                    )
                _validate_finite_chunk(raw, tensor.dtype)
                host = np.frombuffer(raw, dtype=dtypes[tensor.dtype]).reshape(
                    tensor.local_shape
                )
                owners.append((slot, host, digest))
            yield index, tensor, tuple(owners)
            del owners, raw, host


def load_ws32_layer_subset(
    subset: Ws32LayerSubsetMetadata,
    *,
    mesh: object,
    physical_mesh: object,
) -> LoadedWs32LayerSubset:
    """Direct final-owner placement with selected-only integrity evidence."""

    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P

    rows = np.asarray(mesh.devices, dtype=object).tolist()
    if (
        tuple(tuple(int(device.id) for device in row) for row in rows)
        != physical_mesh.device_ids
        or subset.metadata.manifest["mesh_hash"] != physical_mesh.mesh_hash
    ):
        raise CheckpointValidationError(
            "WS32 subset physical mesh identity/order drifted"
        )
    addressable = tuple(
        device
        for row in rows
        for device in row
        if int(device.process_index) == jax.process_index()
    )
    expected_count = (
        32 if jax.default_backend() == "cpu" and jax.process_count() == 1 else 4
    )
    if len(addressable) != expected_count or set(addressable) != set(
        jax.local_devices()
    ):
        raise CheckpointValidationError(
            "WS32 subset addressable device geometry drifted"
        )
    slot_by_id = {
        device_id: slot
        for slot, device_id in enumerate(physical_mesh.flattened_device_ids)
    }
    device_by_slot = {slot_by_id[int(device.id)]: device for device in addressable}
    if set(device_by_slot) != set(subset.local_slots):
        raise CheckpointValidationError(
            "WS32 subset local slots differ from addressable owners"
        )
    before = tuple(_memory_stats(device) for device in addressable)
    arrays: dict[str, Any] = {}
    observed_by_slot: dict[int, dict[str, str]] = {
        slot: {} for slot in subset.local_slots
    }
    pending: list[Any] = []
    try:
        for _, tensor, owners in iter_ws32_layer_subset_host_tensors(subset):
            sharding = NamedSharding(mesh, P(*tensor.partition_spec))
            devices = tuple(
                sharding.addressable_devices_indices_map(tensor.global_shape)
            )
            if (
                set(devices) != set(addressable)
                or tuple(sharding.shard_shape(tensor.global_shape))
                != tensor.local_shape
            ):
                raise CheckpointValidationError(
                    f"WS32 subset shard ownership/shape drifted: {tensor.name}"
                )
            hosts = {slot: (host, digest) for slot, host, digest in owners}
            local_arrays = []
            for device in devices:
                slot = slot_by_id[int(device.id)]
                host, digest = hosts[slot]
                local = jax.device_put(host, device)
                pending.append(local)
                local.block_until_ready()
                if local.devices() != {device}:
                    local.delete()
                    raise CheckpointValidationError(
                        "WS32 subset tensor missed final owner"
                    )
                local_arrays.append(local)
                observed_by_slot[slot][tensor.name] = digest
            arrays[tensor.name] = jax.make_array_from_single_device_arrays(
                tensor.global_shape, sharding, local_arrays
            )
            pending.clear()
            del hosts, owners, host, local_arrays, local
        jax.block_until_ready(tuple(arrays.values()))
    except BaseException:
        for array in pending:
            array.delete()
        for array in arrays.values():
            array.delete()
        raise
    records = tuple(
        {
            "device_id": int(device.id),
            "device_slot": slot_by_id[int(device.id)],
            "expected_full_file_sha256_not_verified": subset.metadata.records_by_slot[
                slot_by_id[int(device.id)]
            ]["sha256"],
            "observed_selected_tensor_sha256": observed_by_slot[
                slot_by_id[int(device.id)]
            ],
            "selected_payload_bytes": subset.payload_bytes_per_chip,
        }
        for device in addressable
    )
    return LoadedWs32LayerSubset(
        arrays,
        subset.layer_ids,
        records,
        before,
        tuple(_memory_stats(device) for device in addressable),
        subset.payload_bytes_per_chip,
        subset.include_embedding,
    )
