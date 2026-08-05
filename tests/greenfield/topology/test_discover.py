from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from glm_tpu.greenfield.errors import TopologyValidationError
from glm_tpu.greenfield.topology.discover import (
    discover_physical_topology,
    validate_target_v4_64,
)


@dataclass(frozen=True)
class RuntimeDevice:
    id: int
    process_index: int
    local_hardware_id: int
    coords: tuple[int, int, int]
    core_on_chip: int = 0
    platform: str = "tpu"
    device_kind: str = "TPU v4"


def runtime_devices() -> tuple[RuntimeDevice, ...]:
    devices = []
    device_id = 0
    for y in range(4):
        for z_pair in range(2):
            process = y * 2 + z_pair
            local_id = 0
            for z in range(z_pair * 2, z_pair * 2 + 2):
                for x in range(2):
                    devices.append(
                        RuntimeDevice(
                            id=device_id,
                            process_index=process,
                            local_hardware_id=local_id,
                            coords=(x, y, z),
                        )
                    )
                    device_id += 1
                    local_id += 1
    return tuple(reversed(devices))


def test_discovery_uses_metadata_not_input_order() -> None:
    topology = discover_physical_topology(
        runtime_devices(), slice_name="db-v4-64-od"
    )
    validate_target_v4_64(topology)
    assert topology.topology_shape == (2, 4, 4)
    assert len(topology.devices) == 32
    assert topology.process_indices == tuple(range(8))
    assert tuple(device.device_id for device in topology.devices) == tuple(range(32))


def test_discovery_refuses_missing_physical_coordinates() -> None:
    @dataclass
    class CpuDevice:
        id: int = 0
        process_index: int = 0
        local_hardware_id: int = 0
        platform: str = "cpu"
        device_kind: str = "cpu"

    with pytest.raises(TopologyValidationError, match="coords"):
        discover_physical_topology([CpuDevice()], slice_name="not-a-tpu")


def test_target_refuses_wrong_device_kind() -> None:
    devices = [replace(device, device_kind="TPU v5p") for device in runtime_devices()]
    topology = discover_physical_topology(devices, slice_name="db-v4-64-od")
    with pytest.raises(TopologyValidationError, match="TPU v4"):
        validate_target_v4_64(topology)


def test_target_refuses_nonuniform_host_capacity() -> None:
    devices = list(runtime_devices())
    source = next(device for device in devices if device.process_index == 7)
    devices[devices.index(source)] = replace(
        source, process_index=6, local_hardware_id=4
    )
    with pytest.raises(TopologyValidationError):
        topology = discover_physical_topology(devices, slice_name="db-v4-64-od")
        validate_target_v4_64(topology)
