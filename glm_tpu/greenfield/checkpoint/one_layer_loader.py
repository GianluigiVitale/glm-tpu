"""Fail-closed direct loader for the bounded PP8 real-layer artifact.

The loader verifies the append-only artifact before importing JAX, resolves
the four runtime devices from the protected physical-topology capture, and
maps packed device slots in physical group order. FP8 block scales are folded
on the host in small chunks and each complete local shard is transferred
directly to its owning device. No global expert or shared-expert tensor is
ever assembled in host memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import gc
import json
from pathlib import Path
import resource
import socket
from typing import Any, Mapping, Sequence

from .one_layer import inspect_one_layer_artifact


@dataclass(frozen=True, slots=True)
class OneLayerLoadExpectation:
    """Immutable identities required before any packed payload is read."""

    manifest_sha256: str
    source_revision: str
    topology_hash: str
    plan_group_hash: str
    plan_id: str = "PP8_LP4"
    model_id: str = "zai-org/GLM-5.2-FP8"
    layer: int = 3

    def __post_init__(self) -> None:
        for name in (
            "manifest_sha256",
            "topology_hash",
            "plan_group_hash",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not self.source_revision.strip():
            raise ValueError("source_revision must be non-empty")
        if self.plan_id != "PP8_LP4":
            raise ValueError("bounded one-layer loader supports only PP8_LP4")
        if self.model_id != "zai-org/GLM-5.2-FP8" or self.layer != 3:
            raise ValueError("bounded loader supports only GLM-5.2-FP8 layer 3")


@dataclass(frozen=True, slots=True)
class StageDeviceResolution:
    """Runtime devices ordered exactly like the captured PP8 stage group."""

    devices: tuple[Any, ...]
    coordinates: tuple[tuple[int, ...], ...]
    captured_device_ids: tuple[int, ...]
    captured_process_index: int
    stage_id: int


@dataclass(frozen=True, slots=True)
class LoadedOneLayer:
    """Device-resident dequantized weights and an auditable load record."""

    mesh: Any
    expert_gate: Any
    expert_up: Any
    expert_down: Any
    shared_gate: Any
    shared_up: Any
    shared_down: Any
    router_weight: Any
    correction_bias: Any
    manifest: Mapping[str, Any]
    load_record: Mapping[str, Any]

    @property
    def kernel_weights(self) -> tuple[Any, ...]:
        return (
            self.router_weight,
            self.correction_bias,
            self.expert_gate,
            self.expert_up,
            self.expert_down,
            self.shared_gate,
            self.shared_up,
            self.shared_down,
        )


def _runtime_attribute(device: object, name: str) -> Any:
    try:
        value = getattr(device, name)
    except AttributeError as error:
        raise ValueError(
            f"runtime device lacks required {name!r}: {device!r}"
        ) from error
    return value() if callable(value) else value


def verify_one_layer_load_contract(
    artifact_dir: Path,
    expectation: OneLayerLoadExpectation,
) -> dict[str, Any]:
    """Hash every file and require exact artifact identities."""

    manifest = inspect_one_layer_artifact(Path(artifact_dir))
    expected = {
        "layer": expectation.layer,
        "manifest_sha256": expectation.manifest_sha256,
        "model_id": expectation.model_id,
        "plan_group_hash": expectation.plan_group_hash,
        "plan_id": expectation.plan_id,
        "source_revision": expectation.source_revision,
        "topology_hash": expectation.topology_hash,
    }
    mismatches = {
        name: {"expected": value, "observed": manifest.get(name)}
        for name, value in expected.items()
        if manifest.get(name) != value
    }
    geometry = manifest.get("geometry", {})
    if geometry.get("stage_size") != 4:
        mismatches["geometry.stage_size"] = {
            "expected": 4,
            "observed": geometry.get("stage_size"),
        }
    if mismatches:
        raise ValueError(f"one-layer load contract mismatch: {mismatches}")
    return manifest


def resolve_pp8_stage_devices(
    runtime_devices: Sequence[object],
    topology_capture: Path | Mapping[str, Any],
    expectation: OneLayerLoadExpectation,
) -> StageDeviceResolution:
    """Match one local four-chip runtime to an authenticated captured group.

    Device ids and runtime list order are deliberately ignored for matching:
    standalone JAX initialization may renumber ids, while physical coordinates
    remain authoritative. Returned order follows the PP8 group contract.
    """

    from ..topology import build_pp8_lp4_groups, group_manifest_hash
    from ..types import PhysicalTopology, PlanName

    if isinstance(topology_capture, Mapping):
        capture = dict(topology_capture)
    else:
        capture = json.loads(Path(topology_capture).read_text())
    contract = capture.get("contract", capture)
    topology_value = contract.get("topology")
    if not isinstance(topology_value, Mapping):
        raise ValueError("topology capture has no physical topology contract")
    topology = PhysicalTopology.from_dict(topology_value)
    if topology.topology_hash != expectation.topology_hash:
        raise ValueError(
            "captured topology hash mismatch: "
            f"expected={expectation.topology_hash} "
            f"observed={topology.topology_hash}"
        )
    groups = build_pp8_lp4_groups(topology)
    observed_group_hash = group_manifest_hash(PlanName.PP8_LP4, groups)
    if observed_group_hash != expectation.plan_group_hash:
        raise ValueError(
            "captured PP8 group hash mismatch: "
            f"expected={expectation.plan_group_hash} "
            f"observed={observed_group_hash}"
        )
    devices = tuple(runtime_devices)
    if len(devices) != 4:
        raise ValueError(
            f"PP8 real-layer run requires four local devices, got {len(devices)}"
        )
    runtime_by_coordinate: dict[tuple[int, ...], object] = {}
    for device in devices:
        coordinate = tuple(
            int(value) for value in _runtime_attribute(device, "coords")
        )
        if coordinate in runtime_by_coordinate:
            raise ValueError(f"duplicate runtime TPU coordinate {coordinate}")
        if str(_runtime_attribute(device, "platform")) != "tpu":
            raise ValueError("real-layer runtime device is not a TPU")
        if str(_runtime_attribute(device, "device_kind")) != "TPU v4":
            raise ValueError("real-layer runtime device is not TPU v4")
        if int(_runtime_attribute(device, "core_on_chip")) != 0:
            raise ValueError("unexpected TPU core_on_chip for v4 physical chip")
        runtime_by_coordinate[coordinate] = device
    runtime_coordinates = frozenset(runtime_by_coordinate)
    captured_hostname = capture.get("hostname")
    captured_process = capture.get("jax_process_index")
    fleet_runtime_order = capture.get(
        "fleet_local_device_ids_in_runtime_order"
    )
    if (
        isinstance(captured_hostname, str)
        and isinstance(captured_process, int)
        and isinstance(fleet_runtime_order, list)
    ):
        if socket.gethostname() != captured_hostname:
            raise ValueError(
                "topology host record does not belong to this runtime: "
                f"capture={captured_hostname} runtime={socket.gethostname()}"
            )
        try:
            captured_runtime_ids = tuple(
                int(value) for value in fleet_runtime_order[captured_process]
            )
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError(
                "topology capture has invalid host runtime device order"
            ) from error
        if len(captured_runtime_ids) != len(devices):
            raise ValueError(
                "captured/runtime local device counts differ: "
                f"capture={captured_runtime_ids} runtime={len(devices)}"
            )
        host_groups = [
            group
            for group in groups
            if group.process_index == captured_process
        ]
        if len(host_groups) != 1:
            raise ValueError(
                "captured PP8 process does not own exactly one stage: "
                f"process={captured_process} groups={len(host_groups)}"
            )
        group = host_groups[0]
        runtime_by_captured_id = dict(
            zip(captured_runtime_ids, devices, strict=True)
        )
        if set(runtime_by_captured_id) != set(group.device_ids):
            raise ValueError(
                "captured runtime ids disagree with PP8 group: "
                f"runtime={captured_runtime_ids} group={group.device_ids}"
            )
        topology_by_id = {
            device.device_id: device for device in topology.devices
        }
        captured_runtime_coordinates = tuple(
            topology_by_id[device_id].coordinates
            for device_id in captured_runtime_ids
        )

        def normalize(
            coordinates: Sequence[tuple[int, ...]],
        ) -> tuple[tuple[int, ...], ...]:
            minima = tuple(
                min(item[axis] for item in coordinates)
                for axis in range(len(coordinates[0]))
            )
            return tuple(
                tuple(
                    value - minima[axis]
                    for axis, value in enumerate(item)
                )
                for item in coordinates
            )

        runtime_in_runtime_order = tuple(
            tuple(
                int(value)
                for value in _runtime_attribute(device, "coords")
            )
            for device in devices
        )
        if runtime_in_runtime_order not in (
            captured_runtime_coordinates,
            normalize(captured_runtime_coordinates),
        ):
            raise ValueError(
                "runtime coordinates are neither physical nor the exact "
                "local-subcube normalization: "
                f"runtime={runtime_in_runtime_order} "
                f"physical={captured_runtime_coordinates}"
            )
        ordered_coordinates = tuple(tuple(item) for item in group.coordinates)
        return StageDeviceResolution(
            devices=tuple(
                runtime_by_captured_id[device_id]
                for device_id in group.device_ids
            ),
            coordinates=ordered_coordinates,
            captured_device_ids=tuple(group.device_ids),
            captured_process_index=int(group.process_index),
            stage_id=int(group.stage_id),
        )

    matching = [
        group
        for group in groups
        if frozenset(group.coordinates) == runtime_coordinates
    ]
    if len(matching) != 1:
        raise ValueError(
            "runtime devices do not match exactly one protected PP8 stage: "
            f"coordinates={sorted(runtime_coordinates)} matches={len(matching)}"
        )
    group = matching[0]
    ordered_coordinates = tuple(tuple(item) for item in group.coordinates)
    return StageDeviceResolution(
        devices=tuple(
            runtime_by_coordinate[item] for item in ordered_coordinates
        ),
        coordinates=ordered_coordinates,
        captured_device_ids=tuple(group.device_ids),
        captured_process_index=int(group.process_index),
        stage_id=int(group.stage_id),
    )


def _rss_peak_bytes() -> int:
    # Linux reports KiB; protected TPU workflows run on Linux TPU VMs.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _memory_stats(device: object) -> dict[str, int] | None:
    value = device.memory_stats()
    if value is None:
        return None
    return {
        str(name): int(number)
        for name, number in value.items()
        if isinstance(number, int) and not isinstance(number, bool)
    }


def _torch_bfloat16_numpy(tensor: Any) -> Any:
    import ml_dtypes
    import torch

    if tensor.dtype != torch.bfloat16 or tensor.device.type != "cpu":
        raise ValueError("direct transfer requires a CPU bfloat16 tensor")
    contiguous = tensor.contiguous()
    return contiguous.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)


def _torch_float8_bits_numpy(tensor: Any) -> Any:
    import torch

    if tensor.dtype != torch.float8_e4m3fn or tensor.device.type != "cpu":
        raise ValueError("direct FP8 transfer requires CPU E4M3FN tensor")
    return tensor.contiguous().view(torch.uint8).numpy()


def _validate_finite_float8_bits(tensor: Any, chunk_size: int) -> None:
    """Reject the two E4M3FN NaN encodings without a full-size temporary."""

    import torch

    bits = tensor.view(torch.uint8)
    leading = 1 if bits.ndim == 2 else bits.shape[0]
    for start in range(0, leading, chunk_size):
        part = bits if bits.ndim == 2 else bits[start : start + chunk_size]
        if bool(torch.any((part == 0x7F) | (part == 0xFF))):
            raise ValueError("packed FP8 weight contains non-finite values")


def _fp8_e4m3fn_lookup() -> tuple[float, ...]:
    values = []
    for bits in range(256):
        sign = -1.0 if bits & 0x80 else 1.0
        exponent = (bits >> 3) & 0x0F
        mantissa = bits & 0x07
        if exponent == 0:
            value = (mantissa / 8.0) * (2.0**-6)
        elif exponent == 15 and mantissa == 7:
            value = float("nan")
        else:
            value = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
        values.append(sign * value)
    return tuple(values)


@lru_cache(maxsize=4)
def _device_dequantizer(block_shape: tuple[int, int]) -> Any:
    import jax
    import jax.numpy as jnp

    lookup_values = _fp8_e4m3fn_lookup()

    @jax.jit
    def dequantize(weight_bits: Any, scale: Any) -> Any:
        # Keep the lookup a compile-time literal instead of a closed-over JAX
        # array committed to whichever physical device initialized the cache.
        lookup = jnp.asarray(lookup_values, dtype=jnp.float32)
        out_blocks = jnp.arange(weight_bits.shape[-2]) // block_shape[0]
        in_blocks = jnp.arange(weight_bits.shape[-1]) // block_shape[1]
        expanded_scale = scale[..., out_blocks[:, None], in_blocks[None, :]]
        weight = lookup[weight_bits.astype(jnp.int32)]
        return (weight * expanded_scale).astype(jnp.bfloat16)

    return dequantize


def _device_put_dequantized(
    jax: Any,
    weight: Any,
    scale: Any,
    device: object,
    *,
    block_shape: tuple[int, int],
    expert_chunk_size: int,
) -> Any:
    import torch

    if scale.dtype != torch.float32:
        raise ValueError("packed FP8 scale must be float32")
    if not bool(torch.isfinite(scale).all()):
        raise ValueError("packed FP8 scale contains non-finite values")
    _validate_finite_float8_bits(weight, expert_chunk_size)
    weight_array = jax.device_put(_torch_float8_bits_numpy(weight), device)
    scale_array = jax.device_put(scale.numpy(), device)
    result = _device_dequantizer(block_shape)(weight_array, scale_array)
    result.block_until_ready()
    weight_array.delete()
    scale_array.delete()
    return result


def dequantize_packed_fp8(
    weight: Any,
    scale: Any,
    *,
    block_shape: tuple[int, int],
    expert_chunk_size: int = 2,
) -> Any:
    """Fold rank-2/rank-3 block scales into BF16 with bounded temporaries."""

    import torch

    if weight.ndim not in (2, 3) or scale.ndim != weight.ndim:
        raise ValueError(
            "packed FP8 weight/scale must be matching rank two or three"
        )
    if tuple(weight.shape[:-2]) != tuple(scale.shape[:-2]):
        raise ValueError("packed FP8 leading dimensions do not match")
    expected_tail = tuple(
        (dimension + block - 1) // block
        for dimension, block in zip(
            weight.shape[-2:], block_shape, strict=True
        )
    )
    if tuple(scale.shape[-2:]) != expected_tail:
        raise ValueError(
            f"packed FP8 scale tail must be {expected_tail}, "
            f"got {tuple(scale.shape[-2:])}"
        )
    if expert_chunk_size <= 0:
        raise ValueError("expert_chunk_size must be positive")
    if not bool(torch.isfinite(scale).all()):
        raise ValueError("packed FP8 scale contains non-finite values")
    output = torch.empty(
        tuple(weight.shape), dtype=torch.bfloat16, device="cpu"
    )

    def convert(weight_part: Any, scale_part: Any, destination: Any) -> None:
        weight_float = weight_part.float()
        if not bool(torch.isfinite(weight_float).all()):
            raise ValueError("packed FP8 weight contains non-finite values")
        expanded = scale_part.float().repeat_interleave(
            block_shape[0], dim=-2
        )
        expanded = expanded.repeat_interleave(block_shape[1], dim=-1)
        expanded = expanded[
            ..., : weight_part.shape[-2], : weight_part.shape[-1]
        ]
        destination.copy_(
            (weight_float * expanded).to(torch.bfloat16)
        )

    if weight.ndim == 2:
        convert(weight, scale, output)
    else:
        for start in range(0, weight.shape[0], expert_chunk_size):
            end = min(start + expert_chunk_size, weight.shape[0])
            convert(weight[start:end], scale[start:end], output[start:end])
    return output


def _device_put_torch_bfloat16(
    jax: Any, tensor: Any, device: object
) -> Any:
    array = jax.device_put(_torch_bfloat16_numpy(tensor), device)
    array.block_until_ready()
    return array


def _assemble_global(
    jax: Any,
    shape: tuple[int, ...],
    sharding: Any,
    shards: Sequence[Any],
) -> Any:
    if len(shards) != len(sharding.addressable_devices):
        raise ValueError(
            "local shard count does not match addressable stage devices"
        )
    by_device = {}
    for shard in shards:
        shard_devices = tuple(shard.devices())
        if len(shard_devices) != 1:
            raise ValueError("direct loader shard is not single-device")
        device = shard_devices[0]
        if device in by_device:
            raise ValueError("direct loader produced duplicate device shards")
        by_device[device] = shard
    ordered_devices = tuple(sharding.addressable_devices)
    if set(by_device) != set(ordered_devices):
        raise ValueError("direct loader shards do not cover addressable devices")
    return jax.make_array_from_single_device_arrays(
        shape, sharding, [by_device[device] for device in ordered_devices]
    )


def load_pp8_one_layer(
    artifact_dir: Path,
    expectation: OneLayerLoadExpectation,
    resolution: StageDeviceResolution,
    *,
    expert_chunk_size: int = 2,
) -> LoadedOneLayer:
    """Verify, dequantize, and directly place one PP8 layer on four devices."""

    manifest = verify_one_layer_load_contract(artifact_dir, expectation)
    if len(resolution.devices) != 4:
        raise ValueError("resolved PP8 stage must contain four devices")

    import jax
    import numpy as np
    import torch
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from safetensors import safe_open

    geometry = manifest["geometry"]
    hidden = int(geometry["hidden_size"])
    intermediate = int(geometry["intermediate_size"])
    experts = int(geometry["num_experts"])
    stage_size = int(geometry["stage_size"])
    local_experts = experts // stage_size
    local_intermediate = intermediate // stage_size
    block_shape = tuple(
        int(value) for value in geometry["fp8_block_shape"]
    )
    mesh = Mesh(np.asarray(resolution.devices), ("expert",))
    expert_sharding = NamedSharding(mesh, P("expert"))
    replicated_sharding = NamedSharding(mesh, P())
    shared_down_sharding = NamedSharding(mesh, P(None, "expert"))
    device_before = [
        _memory_stats(device) for device in resolution.devices
    ]
    host_rss_before = _rss_peak_bytes()
    local: dict[str, list[Any]] = {
        name: []
        for name in (
            "expert_gate",
            "expert_up",
            "expert_down",
            "shared_gate",
            "shared_up",
            "shared_down",
            "router_weight",
            "correction_bias",
        )
    }
    replicated_reference: dict[str, Any] = {}
    files = sorted(
        manifest["files"], key=lambda record: record["device_slot"]
    )
    for slot, (file_record, device) in enumerate(
        zip(files, resolution.devices, strict=True)
    ):
        if file_record["device_slot"] != slot:
            raise ValueError("packed device slots are not contiguous from zero")
        path = Path(artifact_dir) / file_record["filename"]
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in (
                "expert_gate",
                "expert_up",
                "expert_down",
                "shared_gate",
                "shared_up",
                "shared_down",
            ):
                weight = handle.get_tensor(name)
                scale = handle.get_tensor(f"{name}_scale")
                dequantized = _device_put_dequantized(
                    jax,
                    weight,
                    scale,
                    device,
                    block_shape=block_shape,
                    expert_chunk_size=expert_chunk_size,
                )
                local[name].append(dequantized)
                del weight, scale, dequantized
                gc.collect()
            for name in ("router_weight", "correction_bias"):
                tensor = handle.get_tensor(name).contiguous()
                if name == "router_weight":
                    if tensor.dtype != torch.bfloat16:
                        raise ValueError("router weight must be bfloat16")
                    placed = _device_put_torch_bfloat16(
                        jax, tensor, device
                    )
                else:
                    if tensor.dtype != torch.float32:
                        raise ValueError("correction bias must be float32")
                    placed = jax.device_put(tensor.numpy(), device)
                    placed.block_until_ready()
                reference = replicated_reference.setdefault(
                    name, tensor.clone()
                )
                if not torch.equal(reference, tensor):
                    raise ValueError(
                        f"replicated packed tensor {name!r} differs across slots"
                    )
                local[name].append(placed)
                del tensor
        gc.collect()
    replicated_reference.clear()
    gc.collect()

    expert_gate = _assemble_global(
        jax,
        (experts, intermediate, hidden),
        expert_sharding,
        local["expert_gate"],
    )
    expert_up = _assemble_global(
        jax,
        (experts, intermediate, hidden),
        expert_sharding,
        local["expert_up"],
    )
    expert_down = _assemble_global(
        jax,
        (experts, hidden, intermediate),
        expert_sharding,
        local["expert_down"],
    )
    shared_gate = _assemble_global(
        jax,
        (intermediate, hidden),
        expert_sharding,
        local["shared_gate"],
    )
    shared_up = _assemble_global(
        jax,
        (intermediate, hidden),
        expert_sharding,
        local["shared_up"],
    )
    shared_down = _assemble_global(
        jax,
        (hidden, intermediate),
        shared_down_sharding,
        local["shared_down"],
    )
    router_weight = _assemble_global(
        jax,
        (experts, hidden),
        replicated_sharding,
        local["router_weight"],
    )
    correction_bias = _assemble_global(
        jax,
        (experts,),
        replicated_sharding,
        local["correction_bias"],
    )
    del local
    gc.collect()
    device_after = [_memory_stats(device) for device in resolution.devices]
    load_record = {
        "captured_device_ids_in_slot_order": list(
            resolution.captured_device_ids
        ),
        "coordinates_in_slot_order": [
            list(item) for item in resolution.coordinates
        ],
        "device_dequantizations": 4 * 6,
        "device_memory_after": device_after,
        "device_memory_before": device_before,
        "device_slot_count": 4,
        "expert_chunk_size": expert_chunk_size,
        "host_global_concatenations": 0,
        "host_fp8_dequantizations": 0,
        "host_peak_rss_after_bytes": _rss_peak_bytes(),
        "host_peak_rss_before_bytes": host_rss_before,
        "local_experts_per_device": local_experts,
        "local_shared_intermediate_per_device": local_intermediate,
        "packed_manifest_sha256": manifest["manifest_sha256"],
        "packed_single_device_transfers": 4 * 14,
        "stage_id": resolution.stage_id,
    }
    return LoadedOneLayer(
        mesh=mesh,
        expert_gate=expert_gate,
        expert_up=expert_up,
        expert_down=expert_down,
        shared_gate=shared_gate,
        shared_up=shared_up,
        shared_down=shared_down,
        router_weight=router_weight,
        correction_bias=correction_bias,
        manifest=manifest,
        load_record=load_record,
    )
