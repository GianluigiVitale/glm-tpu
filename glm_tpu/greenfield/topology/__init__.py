"""Physical TPU discovery and topology-aligned execution groups."""

from .discover import discover_physical_topology, validate_target_v4_64
from .groups import (
    LocalReplicaGroup,
    build_pp16_lp2_groups,
    build_pp8_lp4_groups,
    collective_groups_for_size,
    group_manifest_hash,
    groups_to_dict,
    physical_device_ring,
    validate_local_groups,
)

__all__ = [
    "LocalReplicaGroup",
    "build_pp16_lp2_groups",
    "build_pp8_lp4_groups",
    "collective_groups_for_size",
    "discover_physical_topology",
    "group_manifest_hash",
    "groups_to_dict",
    "physical_device_ring",
    "validate_local_groups",
    "validate_target_v4_64",
]
