"""Loading a verified packed checkpoint onto the device mesh."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping

from glm_tpu.exceptions import CheckpointValidationError
from glm_tpu.model_loader.sharded_state.verify import VerifiedWs32RuntimeCheckpoint


@dataclass(frozen=True, slots=True)
class LoadedWs32RuntimeCheckpoint:
    arrays: Mapping[str, Any]
    local_device_slots: tuple[Mapping[str, Any], ...]
    device_memory_before: tuple[Mapping[str, int] | None, ...]
    device_memory_after: tuple[Mapping[str, int] | None, ...]


def _memory_stats(device: object) -> Mapping[str, int] | None:
    value = device.memory_stats()
    if value is None:
        return None
    return {
        str(name): int(number)
        for name, number in value.items()
        if isinstance(number, int) and not isinstance(number, bool)
    }


def _host_tensor(tensor: Any, *, dtype: str, name: str) -> tuple[Any, str]:
    import ml_dtypes
    import numpy as np
    import torch

    contiguous = tensor.contiguous()
    raw = contiguous.view(torch.uint8).numpy()
    if dtype == "U8" and contiguous.dtype == torch.uint8:
        host = np.asarray(contiguous.numpy())
        storage_dtype = "U8"
    elif dtype == "F32" and contiguous.dtype == torch.float32:
        if not bool(torch.isfinite(contiguous).all()):
            raise CheckpointValidationError(f"WS32 loader found non-finite {name!r}")
        host = np.asarray(contiguous.numpy())
        storage_dtype = "F32"
    elif dtype == "BF16" and contiguous.dtype == torch.bfloat16:
        if not bool(torch.isfinite(contiguous).all()):
            raise CheckpointValidationError(f"WS32 loader found non-finite {name!r}")
        host = contiguous.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
        storage_dtype = "BF16"
    else:
        raise CheckpointValidationError(
            f"WS32 loader dtype drifted for {name!r}: {contiguous.dtype}"
        )
    if dtype == "U8" and bool(np.any((host == 0x7F) | (host == 0xFF))):
        raise CheckpointValidationError(
            f"WS32 loader found non-finite FP8 bits in {name!r}"
        )
    return host, sha256(memoryview(raw).cast("B")).hexdigest()


def load_ws32_runtime_checkpoint(
    checkpoint: VerifiedWs32RuntimeCheckpoint,
    *,
    mesh: object,
    physical_mesh: object,
) -> LoadedWs32RuntimeCheckpoint:
    """Direct-load only this host's final-owner files onto its four chips."""

    if not isinstance(checkpoint, VerifiedWs32RuntimeCheckpoint):
        raise CheckpointValidationError("WS32 full loader requires full-runtime verification")

    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from safetensors import safe_open

    if checkpoint.manifest.get("mesh_hash") != physical_mesh.mesh_hash:
        raise CheckpointValidationError("WS32 loader physical mesh identity drifted")
    mesh_ids = tuple(
        tuple(int(device.id) for device in row)
        for row in np.asarray(mesh.devices, dtype=object).tolist()
    )
    if mesh_ids != physical_mesh.device_ids:
        raise CheckpointValidationError(
            "WS32 loader JAX mesh order differs from physical slots"
        )
    addressable = tuple(
        device
        for row in np.asarray(mesh.devices, dtype=object).tolist()
        for device in row
        if int(device.process_index) == jax.process_index()
    )
    expected_addressable = (
        32
        if jax.default_backend() == "cpu" and jax.process_count() == 1
        else 4
    )
    if len(addressable) != expected_addressable or set(addressable) != set(
        jax.local_devices()
    ):
        raise CheckpointValidationError(
            "WS32 loader addressable-device geometry drifted"
        )
    slot_by_device_id = {
        device_id: slot
        for slot, device_id in enumerate(physical_mesh.flattened_device_ids)
    }
    plan_by_slot = {plan.device_slot: plan for plan in checkpoint.plans}
    before = tuple(_memory_stats(device) for device in addressable)
    arrays: dict[str, Any] = {}
    with ExitStack() as stack:
        handles = {}
        for device in addressable:
            slot = slot_by_device_id[int(device.id)]
            plan = plan_by_slot[slot]
            handles[int(device.id)] = stack.enter_context(
                safe_open(
                    checkpoint.root / plan.filename,
                    framework="pt",
                    device="cpu",
                )
            )
        for tensor_index, tensor_plan in enumerate(checkpoint.plans[0].tensors):
            sharding = NamedSharding(
                mesh, P(*tensor_plan.partition_spec)
            )
            shard_devices = tuple(
                sharding.addressable_devices_indices_map(
                    tensor_plan.global_shape
                )
            )
            if set(shard_devices) != set(addressable):
                raise CheckpointValidationError(
                    f"WS32 {tensor_plan.name!r} addressable owners drifted"
                )
            expected_local_shape = sharding.shard_shape(
                tensor_plan.global_shape
            )
            local_arrays = []
            for device in shard_devices:
                slot = slot_by_device_id[int(device.id)]
                record = checkpoint.records_by_slot[slot]
                tensor = handles[int(device.id)].get_tensor(tensor_plan.name)
                host, digest = _host_tensor(
                    tensor,
                    dtype=tensor_plan.dtype,
                    name=tensor_plan.name,
                )
                if digest != record["tensor_sha256"][tensor_index]:
                    raise CheckpointValidationError(
                        f"WS32 tensor checksum drifted while loading "
                        f"{tensor_plan.name!r} from slot {slot}"
                    )
                if tuple(host.shape) != expected_local_shape:
                    raise CheckpointValidationError(
                        f"WS32 local shape drifted for {tensor_plan.name!r}"
                    )
                local = jax.device_put(host, device)
                local.block_until_ready()
                if tuple(local.devices()) != (device,):
                    local.delete()
                    raise CheckpointValidationError(
                        f"WS32 {tensor_plan.name!r} missed its final owner"
                    )
                local_arrays.append(local)
            arrays[tensor_plan.name] = jax.make_array_from_single_device_arrays(
                tensor_plan.global_shape,
                sharding,
                tuple(local_arrays),
            )
    jax.block_until_ready(tuple(arrays.values()))
    after = tuple(_memory_stats(device) for device in addressable)
    local_records = tuple(
        {
            "device_id": int(device.id),
            "device_slot": slot_by_device_id[int(device.id)],
            "expert_coordinate": slot_by_device_id[int(device.id)] // 4,
            "feature_coordinate": slot_by_device_id[int(device.id)] % 4,
            "file_sha256": checkpoint.records_by_slot[
                slot_by_device_id[int(device.id)]
            ]["sha256"],
        }
        for device in addressable
    )
    return LoadedWs32RuntimeCheckpoint(
        arrays=arrays,
        local_device_slots=local_records,
        device_memory_before=before,
        device_memory_after=after,
    )
