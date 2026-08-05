from __future__ import annotations

from dataclasses import replace

import pytest

from glm_tpu.greenfield.errors import TopologyValidationError
from glm_tpu.greenfield.topology.discover import discover_physical_topology
from glm_tpu.greenfield.topology.groups import (
    build_pp16_lp2_groups,
    build_pp8_lp4_groups,
    collective_groups_for_size,
    group_manifest_hash,
    stage_transfer_lanes,
    stage_transfer_pairs,
    validate_local_groups,
)
from glm_tpu.greenfield.types import PlanName

from .test_discover import runtime_devices


def topology():
    return discover_physical_topology(
        runtime_devices(), slice_name="db-v4-64-od"
    )


def assert_stage_ring_is_all_lane_adjacent(topo, groups) -> None:
    devices = {device.device_id: device for device in topo.devices}
    shape = topo.topology_shape
    for left_group, right_group in zip(groups, groups[1:] + groups[:1]):
        left = [devices[device_id] for device_id in left_group.device_ids]
        right = [devices[device_id] for device_id in right_group.device_ids]
        adjacency = {
            item.device_id: {
                candidate.device_id
                for candidate in right
                if sum(
                    min(abs(a - b), size - abs(a - b)) != 0
                    for a, b, size in zip(
                        item.coordinates, candidate.coordinates, shape
                    )
                )
                == 1
                and sum(
                    min(abs(a - b), size - abs(a - b))
                    for a, b, size in zip(
                        item.coordinates, candidate.coordinates, shape
                    )
                )
                == 1
            }
            for item in left
        }
        assert all(adjacency.values())
        assert len(set().union(*adjacency.values())) == len(right)


def test_pp8_groups_are_host_aligned_and_physically_ordered() -> None:
    groups = build_pp8_lp4_groups(topology())
    assert len(groups) == 8
    assert {len(group.device_ids) for group in groups} == {4}
    assert {group.process_index for group in groups} == set(range(8))
    assert_stage_ring_is_all_lane_adjacent(topology(), groups)
    assert len(group_manifest_hash(PlanName.PP8_LP4, groups)) == 64


def test_stage_order_uses_host_coordinates_not_process_index() -> None:
    devices = [
        replace(device, process_index=7 - device.process_index)
        for device in runtime_devices()
    ]
    topo = discover_physical_topology(devices, slice_name="db-v4-64-od")
    original = build_pp8_lp4_groups(topology())
    groups = build_pp8_lp4_groups(topo)
    assert tuple(group.coordinates for group in groups) == tuple(
        group.coordinates for group in original
    )
    assert tuple(group.process_index for group in groups) == tuple(
        7 - group.process_index for group in original
    )


def test_pp16_prefers_adjacent_length_two_axis() -> None:
    groups = build_pp16_lp2_groups(topology())
    assert len(groups) == 16
    assert {len(group.device_ids) for group in groups} == {2}
    assert_stage_ring_is_all_lane_adjacent(topology(), groups)
    for group in groups:
        left, right = group.coordinates
        differences = [axis for axis, (a, b) in enumerate(zip(left, right)) if a != b]
        assert differences == [0]


def test_pp16_pairing_is_independent_of_global_device_ids() -> None:
    original = build_pp16_lp2_groups(topology())
    renumbered = [
        replace(device, id=31 - device.id) for device in runtime_devices()
    ]
    renumbered_topology = discover_physical_topology(
        renumbered, slice_name="db-v4-64-od"
    )
    changed = build_pp16_lp2_groups(renumbered_topology)
    assert tuple(group.coordinates for group in changed) == tuple(
        group.coordinates for group in original
    )


def test_group_hash_is_stable() -> None:
    groups_a = build_pp16_lp2_groups(topology())
    groups_b = build_pp16_lp2_groups(topology())
    assert group_manifest_hash(PlanName.PP16_LP2, groups_a) == group_manifest_hash(
        PlanName.PP16_LP2, groups_b
    )


def test_collective_benchmark_groups_partition_physical_rings() -> None:
    topo = topology()
    by_id = {device.device_id: device for device in topo.devices}
    for size in (2, 4, 8, 32):
        groups = collective_groups_for_size(topo, size)
        assert {len(group) for group in groups} == {size}
        assert sorted(device for group in groups for device in group) == list(
            range(32)
        )
        for group in groups:
            for left, right in zip(group, group[1:] + group[:1]):
                left_coordinates = by_id[left].coordinates
                right_coordinates = by_id[right].coordinates
                distances = [
                    min(abs(a - b), dimension - abs(a - b))
                    for a, b, dimension in zip(
                        left_coordinates,
                        right_coordinates,
                        topo.topology_shape,
                    )
                ]
                assert sum(distances) == 1


@pytest.mark.parametrize(
    ("builder", "stage_count", "lane_count"),
    ((build_pp8_lp4_groups, 8, 4), (build_pp16_lp2_groups, 16, 2)),
)
def test_stage_transfer_lanes_are_closed_physical_rings(
    builder, stage_count: int, lane_count: int
) -> None:
    topo = topology()
    lanes = stage_transfer_lanes(topo, builder(topo))
    pairs = stage_transfer_pairs(topo, builder(topo))
    by_id = {device.device_id: device for device in topo.devices}
    assert len(lanes) == lane_count
    assert {len(lane) for lane in lanes} == {stage_count}
    assert sorted(device for lane in lanes for device in lane) == list(range(32))
    assert len(pairs) == 32
    assert {source for source, _ in pairs} == set(range(32))
    assert {target for _, target in pairs} == set(range(32))
    for source, target in pairs:
        left = by_id[source].coordinates
        right = by_id[target].coordinates
        distances = [
            min(abs(a - b), size - abs(a - b))
            for a, b, size in zip(left, right, topo.topology_shape)
        ]
        assert sum(distances) == 1


def test_validator_refuses_cross_host_group() -> None:
    topo = topology()
    groups = list(build_pp8_lp4_groups(topo))
    groups[0] = replace(
        groups[0],
        device_ids=(*groups[0].device_ids[:3], groups[1].device_ids[0]),
        coordinates=(*groups[0].coordinates[:3], groups[1].coordinates[0]),
    )
    with pytest.raises(TopologyValidationError, match="not confined"):
        validate_local_groups(topo, PlanName.PP8_LP4, tuple(groups))


def test_pp16_refuses_host_without_adjacent_perfect_matching() -> None:
    topo = topology()
    # Keep the inventory rectangular but assign one host a disconnected set.
    devices = list(topo.devices)
    process_zero = [device for device in devices if device.process_index == 0]
    distant = next(device for device in devices if device.coordinates == (0, 2, 2))
    distant_process = distant.process_index
    devices[devices.index(process_zero[-1])] = replace(
        process_zero[-1], process_index=distant_process
    )
    devices[devices.index(distant)] = replace(distant, process_index=0)
    for process in (0, distant_process):
        members = sorted(
            (device for device in devices if device.process_index == process),
            key=lambda device: device.device_id,
        )
        for local_id, device in enumerate(members):
            devices[devices.index(device)] = replace(
                device, local_device_id=local_id
            )
    broken = replace(topo, devices=tuple(devices))
    with pytest.raises(TopologyValidationError, match="perfect matching"):
        build_pp16_lp2_groups(broken)
