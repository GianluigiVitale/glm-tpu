"""Deterministic PP8/PP16 grouping from observed TPU coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from itertools import combinations
from typing import Any, Iterable

from ..errors import TopologyValidationError
from ..types import PhysicalDevice, PhysicalTopology, PlanName


@dataclass(frozen=True, slots=True)
class LocalReplicaGroup:
    """One repeated-layer collective group, explicit in physical ids."""

    stage_id: int
    process_index: int
    device_ids: tuple[int, ...]
    coordinates: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "device_ids", tuple(self.device_ids))
        object.__setattr__(
            self,
            "coordinates",
            tuple(tuple(coordinate) for coordinate in self.coordinates),
        )
        if self.stage_id < 0 or self.process_index < 0:
            raise TopologyValidationError("stage and process ids must be non-negative")
        if not self.device_ids or len(set(self.device_ids)) != len(self.device_ids):
            raise TopologyValidationError(
                "local group device ids must be non-empty and unique"
            )
        if len(self.device_ids) != len(self.coordinates):
            raise TopologyValidationError(
                "local group coordinates must align one-to-one with device ids"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinates": [list(coordinate) for coordinate in self.coordinates],
            "device_ids": list(self.device_ids),
            "process_index": self.process_index,
            "stage_id": self.stage_id,
        }


def _device_key(device: PhysicalDevice) -> tuple[tuple[int, ...], int, int]:
    return (device.coordinates, device.core_on_chip, device.device_id)


def _devices_by_process(
    topology: PhysicalTopology,
) -> tuple[tuple[int, tuple[PhysicalDevice, ...]], ...]:
    by_process: dict[int, list[PhysicalDevice]] = {}
    for device in topology.devices:
        by_process.setdefault(device.process_index, []).append(device)
    groups = [
        (process, tuple(sorted(devices, key=_device_key)))
        for process, devices in by_process.items()
    ]
    # Stage order follows physical host anchors, never process or JAX list order.
    return tuple(sorted(groups, key=lambda entry: _device_key(entry[1][0])))


def _differing_axis(
    left: PhysicalDevice,
    right: PhysicalDevice,
    topology: PhysicalTopology,
) -> int | None:
    differing = []
    for axis, (a, b, size) in enumerate(
        zip(left.coordinates, right.coordinates, topology.topology_shape)
    ):
        distance = abs(a - b)
        if distance == 0:
            continue
        if distance not in (1, size - 1):
            return None
        differing.append(axis)
    return differing[0] if len(differing) == 1 else None


def _connected(
    devices: tuple[PhysicalDevice, ...], topology: PhysicalTopology
) -> bool:
    pending = {device.device_id: device for device in devices}
    reached = {devices[0].device_id}
    changed = True
    while changed:
        changed = False
        for device_id, device in tuple(pending.items()):
            if device_id in reached:
                continue
            if any(
                _differing_axis(device, pending[other], topology) is not None
                for other in reached
            ):
                reached.add(device_id)
                changed = True
    return len(reached) == len(devices)


def _groups_have_adjacent_perfect_matching(
    left: tuple[PhysicalDevice, ...],
    right: tuple[PhysicalDevice, ...],
    topology: PhysicalTopology,
) -> bool:
    """Whether every transfer lane can use one distinct physical neighbor."""

    if len(left) != len(right):
        return False

    def search(index: int, available: frozenset[int]) -> bool:
        if index == len(left):
            return True
        candidates = sorted(
            (
                right_index
                for right_index in available
                if _differing_axis(left[index], right[right_index], topology)
                is not None
            ),
            key=lambda right_index: _device_key(right[right_index]),
        )
        return any(
            search(index + 1, available - {right_index})
            for right_index in candidates
        )

    return search(0, frozenset(range(len(right))))


def _topology_ring(
    entries: tuple[tuple[int, tuple[PhysicalDevice, ...]], ...],
    topology: PhysicalTopology,
    *,
    label: str,
) -> tuple[tuple[int, tuple[PhysicalDevice, ...]], ...]:
    """Find a deterministic Hamiltonian ring over physical transfer groups."""

    ordered = tuple(
        sorted(entries, key=lambda entry: min(_device_key(d) for d in entry[1]))
    )
    if len(ordered) < 2:
        return ordered
    neighbors = {
        index: tuple(
            candidate
            for candidate in range(len(ordered))
            if candidate != index
            and _groups_have_adjacent_perfect_matching(
                ordered[index][1], ordered[candidate][1], topology
            )
        )
        for index in range(len(ordered))
    }

    def search(path: tuple[int, ...], remaining: frozenset[int]) -> tuple[int, ...] | None:
        if not remaining:
            return path if path[0] in neighbors[path[-1]] else None
        candidates = sorted(
            remaining.intersection(neighbors[path[-1]]),
            key=lambda index: min(_device_key(d) for d in ordered[index][1]),
        )
        for candidate in candidates:
            result = search(path + (candidate,), remaining - {candidate})
            if result is not None:
                return result
        return None

    indices = search((0,), frozenset(range(1, len(ordered))))
    if indices is None:
        raise TopologyValidationError(
            f"{label} groups have no all-lane topology-adjacent stage ring"
        )
    return tuple(ordered[index] for index in indices)


def _make_group(
    stage_id: int, process: int, devices: Iterable[PhysicalDevice]
) -> LocalReplicaGroup:
    ordered = tuple(sorted(devices, key=_device_key))
    return LocalReplicaGroup(
        stage_id=stage_id,
        process_index=process,
        device_ids=tuple(device.device_id for device in ordered),
        coordinates=tuple(device.coordinates for device in ordered),
    )


def build_pp8_lp4_groups(
    topology: PhysicalTopology,
) -> tuple[LocalReplicaGroup, ...]:
    """Build eight host-aligned four-chip groups in physical host order."""

    process_groups = _devices_by_process(topology)
    if len(process_groups) != 8 or any(len(devices) != 4 for _, devices in process_groups):
        raise TopologyValidationError(
            "PP8_LP4 requires exactly eight hosts with four devices each"
        )
    stage_ring = _topology_ring(process_groups, topology, label="PP8_LP4")
    groups = tuple(
        _make_group(stage, process, devices)
        for stage, (process, devices) in enumerate(stage_ring)
    )
    validate_local_groups(topology, PlanName.PP8_LP4, groups)
    return groups


def _perfect_matchings(
    devices: tuple[PhysicalDevice, ...],
) -> tuple[tuple[tuple[PhysicalDevice, PhysicalDevice], ...], ...]:
    if len(devices) != 4:
        raise TopologyValidationError("PP16 pairing requires four devices per host")
    first = devices[0]
    result = []
    for partner in devices[1:]:
        remaining = tuple(device for device in devices if device not in (first, partner))
        result.append(((first, partner), (remaining[0], remaining[1])))
    return tuple(result)


def _pairing_score(
    matching: tuple[tuple[PhysicalDevice, PhysicalDevice], ...],
    topology: PhysicalTopology,
) -> tuple[
    int,
    tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
] | None:
    axes = []
    coordinate_pairs = []
    for left, right in matching:
        axis = _differing_axis(left, right, topology)
        if axis is None:
            return None
        axes.append(axis)
        coordinate_pairs.append(tuple(sorted((left.coordinates, right.coordinates))))
    # Prefer the physical dimension of length two, then stable physical order.
    penalty = sum(topology.topology_shape[axis] != 2 for axis in axes)
    return penalty, tuple(sorted(coordinate_pairs))


def build_pp16_lp2_groups(
    topology: PhysicalTopology,
) -> tuple[LocalReplicaGroup, ...]:
    """Build adjacent two-chip stages, preferring the length-two dimension."""

    process_groups = _devices_by_process(topology)
    if len(process_groups) != 8 or any(len(devices) != 4 for _, devices in process_groups):
        raise TopologyValidationError(
            "PP16_LP2 requires exactly eight hosts with four devices each"
        )
    pairs = []
    for process, devices in process_groups:
        candidates = []
        for matching in _perfect_matchings(devices):
            score = _pairing_score(matching, topology)
            if score is not None:
                candidates.append((score, matching))
        if not candidates:
            raise TopologyValidationError(
                f"process {process} has no topology-adjacent two-chip perfect matching"
            )
        _, selected = min(candidates, key=lambda candidate: candidate[0])
        for pair in selected:
            pairs.append((process, tuple(pair)))
    stage_ring = _topology_ring(tuple(pairs), topology, label="PP16_LP2")
    groups = tuple(
        _make_group(stage, process, devices)
        for stage, (process, devices) in enumerate(stage_ring)
    )
    validate_local_groups(topology, PlanName.PP16_LP2, groups)
    return groups


def validate_local_groups(
    topology: PhysicalTopology,
    plan: PlanName,
    groups: tuple[LocalReplicaGroup, ...],
) -> None:
    """Reject hidden cross-host, disconnected, incomplete, or reused groups."""

    expected = {
        PlanName.PP8_LP4: (8, 4, 1),
        PlanName.PP16_LP2: (16, 2, 2),
    }
    if plan not in expected:
        raise TopologyValidationError(f"no local-group contract exists for {plan}")
    expected_groups, expected_size, groups_per_process = expected[plan]
    if len(groups) != expected_groups:
        raise TopologyValidationError(
            f"{plan.value} requires {expected_groups} groups, got {len(groups)}"
        )
    if tuple(group.stage_id for group in groups) != tuple(range(expected_groups)):
        raise TopologyValidationError("group stage ids must be contiguous from zero")
    devices_by_id = {device.device_id: device for device in topology.devices}
    seen = []
    process_counts: dict[int, int] = {}
    for group in groups:
        if len(group.device_ids) != expected_size:
            raise TopologyValidationError(
                f"stage {group.stage_id} must contain {expected_size} devices"
            )
        if set(group.device_ids) - devices_by_id.keys():
            raise TopologyValidationError(
                f"stage {group.stage_id} references an unknown physical device"
            )
        devices = tuple(devices_by_id[device_id] for device_id in group.device_ids)
        if {device.process_index for device in devices} != {group.process_index}:
            raise TopologyValidationError(
                f"stage {group.stage_id} is not confined to process {group.process_index}"
            )
        expected_coordinates = tuple(device.coordinates for device in devices)
        if group.coordinates != expected_coordinates:
            raise TopologyValidationError(
                f"stage {group.stage_id} coordinate record does not match device ids"
            )
        if not _connected(devices, topology):
            raise TopologyValidationError(
                f"stage {group.stage_id} is not physically connected"
            )
        if plan is PlanName.PP16_LP2 and _differing_axis(
            devices[0], devices[1], topology
        ) is None:
            raise TopologyValidationError(
                f"stage {group.stage_id} PP16 pair is not physically adjacent"
            )
        seen.extend(group.device_ids)
        process_counts[group.process_index] = process_counts.get(group.process_index, 0) + 1
    if len(seen) != len(set(seen)):
        raise TopologyValidationError("a physical device appears in multiple local groups")
    if set(seen) != set(devices_by_id):
        raise TopologyValidationError(
            "local groups must cover every physical device exactly once"
        )
    if set(process_counts) != set(topology.process_indices) or set(
        process_counts.values()
    ) != {groups_per_process}:
        raise TopologyValidationError(
            f"{plan.value} requires {groups_per_process} group(s) on every process"
        )
    group_devices = [
        tuple(devices_by_id[device_id] for device_id in group.device_ids)
        for group in groups
    ]
    for stage, (left, right) in enumerate(
        zip(group_devices, group_devices[1:] + group_devices[:1])
    ):
        if not _groups_have_adjacent_perfect_matching(left, right, topology):
            raise TopologyValidationError(
                f"stage boundary {stage}->{(stage + 1) % len(groups)} lacks an "
                "all-lane topology-adjacent transfer matching"
            )


def groups_to_dict(
    plan: PlanName, groups: tuple[LocalReplicaGroup, ...]
) -> dict[str, Any]:
    return {
        "groups": [group.to_dict() for group in groups],
        "plan": plan.value,
    }


def group_manifest_hash(
    plan: PlanName, groups: tuple[LocalReplicaGroup, ...]
) -> str:
    encoded = json.dumps(
        groups_to_dict(plan, groups),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()
