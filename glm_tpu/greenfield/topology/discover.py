"""Convert runtime JAX metadata into a fail-closed physical inventory."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from ..errors import TopologyValidationError
from ..types import PhysicalDevice, PhysicalTopology


def _attribute(device: object, name: str) -> Any:
    try:
        value = getattr(device, name)
    except AttributeError as exc:
        raise TopologyValidationError(
            f"runtime device {device!r} does not expose required attribute {name!r}"
        ) from exc
    return value() if callable(value) else value


def discover_physical_topology(
    devices: Sequence[object],
    *,
    slice_name: str,
    observed_local_order: Mapping[int, int] | None = None,
) -> PhysicalTopology:
    """Capture only observed metadata; never infer placement from list order.

    TPU discovery requires ``coords`` and ``core_on_chip``.  CPU devices and
    incomplete mocks are rejected because they cannot prove physical groups.
    ``local_hardware_id`` is used when present. TPU v4 currently reports it as
    ``None``; in that case callers must supply ``observed_local_order`` built
    from a fleet gather of each process's actual ``jax.local_devices()`` list.
    It is never reconstructed from global ids.
    """

    if not devices:
        raise TopologyValidationError("runtime returned no devices")

    captured = []
    for device in devices:
        device_id = int(_attribute(device, "id"))
        runtime_local_id = _attribute(device, "local_hardware_id")
        observed_local_id = (
            None
            if observed_local_order is None
            else observed_local_order.get(device_id)
        )
        if runtime_local_id is None and observed_local_id is None:
            raise TopologyValidationError(
                f"runtime device {device_id} has no local_hardware_id and no "
                "observed local-device ordering"
            )
        if runtime_local_id is not None:
            local_device_id = int(runtime_local_id)
            if (
                observed_local_id is not None
                and local_device_id != observed_local_id
            ):
                raise TopologyValidationError(
                    f"runtime and observed local ids disagree for device {device_id}: "
                    f"{local_device_id} != {observed_local_id}"
                )
        else:
            local_device_id = int(observed_local_id)
        coordinates = tuple(_attribute(device, "coords"))
        captured.append(
            PhysicalDevice(
                device_id=device_id,
                process_index=int(_attribute(device, "process_index")),
                local_device_id=local_device_id,
                coordinates=coordinates,
                core_on_chip=int(_attribute(device, "core_on_chip")),
                platform=str(_attribute(device, "platform")),
                device_kind=str(_attribute(device, "device_kind")),
            )
        )

    dimensions = len(captured[0].coordinates)
    if any(len(device.coordinates) != dimensions for device in captured):
        raise TopologyValidationError(
            "runtime devices report inconsistent coordinate dimensionality"
        )
    minima = tuple(
        min(device.coordinates[axis] for device in captured)
        for axis in range(dimensions)
    )
    if any(minimum != 0 for minimum in minima):
        raise TopologyValidationError(
            f"runtime coordinates must be zero-based, got minima {minima}"
        )
    topology_shape = tuple(
        max(device.coordinates[axis] for device in captured) + 1
        for axis in range(dimensions)
    )
    return PhysicalTopology(
        slice_name=slice_name,
        topology_shape=topology_shape,
        devices=tuple(captured),
    )


def validate_target_v4_64(topology: PhysicalTopology) -> None:
    """Require the exact existing eight-host / 32-chip TPU-v4 target."""

    if len(topology.devices) != 32:
        raise TopologyValidationError(
            f"target requires 32 JAX-visible chips, found {len(topology.devices)}"
        )
    if sorted(topology.topology_shape) != [2, 4, 4]:
        raise TopologyValidationError(
            "target requires a permutation of physical topology 2x4x4, got "
            f"{topology.topology_shape}"
        )
    if len(topology.process_indices) != 8:
        raise TopologyValidationError(
            f"target requires eight processes/hosts, found {len(topology.process_indices)}"
        )
    if topology.process_indices != tuple(range(8)):
        raise TopologyValidationError(
            f"target process indices must be 0..7, got {topology.process_indices}"
        )
    counts = Counter(device.process_index for device in topology.devices)
    if set(counts.values()) != {4}:
        raise TopologyValidationError(
            f"target requires four chips per host, got {dict(sorted(counts.items()))}"
        )
    if {device.platform.lower() for device in topology.devices} != {"tpu"}:
        raise TopologyValidationError("every target device must report platform='tpu'")
    kinds = {device.device_kind.lower() for device in topology.devices}
    if len(kinds) != 1 or not all("tpu v4" in kind for kind in kinds):
        raise TopologyValidationError(
            f"every target device must report TPU v4, got {sorted(kinds)}"
        )
