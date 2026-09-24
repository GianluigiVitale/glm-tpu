from __future__ import annotations

import json
from pathlib import Path

import pytest

from glm_tpu.distributed.mesh import Ws32MeshContract, build_ws32_physical_mesh
from glm_tpu.config.model import ModelGeometry, PhysicalDevice, PhysicalTopology
# The pinned GLM-5.3 config; its geometry equals the archived GLM-5.2 file's (test_glm53_model).
from tools.equivalence.fixture import config_json


REPO = Path(__file__).resolve().parents[2]


def _topology() -> PhysicalTopology:
    devices = []
    for device_id, coordinates in enumerate(
        (x, y, z) for x in range(2) for y in range(4) for z in range(4)
    ):
        devices.append(
            PhysicalDevice(
                device_id=device_id,
                process_index=device_id // 4,
                local_device_id=device_id % 4,
                coordinates=coordinates,
                core_on_chip=0,
                platform="tpu",
                device_kind="TPU v4",
            )
        )
    return PhysicalTopology(
        slice_name="example-slice",
        topology_shape=(2, 4, 4),
        devices=tuple(devices),
    )


def test_ws32_real_geometry_layout_is_one_row_and_reciprocal() -> None:
    exact_geometry = ModelGeometry.from_hf_config(
        config_json()
    )
    contract = Ws32MeshContract()
    layout = contract.layout_summary(exact_geometry)
    assert layout["mesh_shape"] == [8, 4]
    assert layout["residual"] == {
        "global_shape": [1, 6144],
        "local_shape": [1, 1536],
        "partition_spec": [None, "feature"],
        "replicated_axis": "expert",
    }
    assert layout["dense_gate_up"]["local_shape"] == [1536, 1536]
    assert layout["dense_down"]["local_shape"] == [1536, 1536]
    assert layout["routed_gate_up"]["local_shape"] == [32, 2048, 1536]
    assert layout["routed_down"]["local_shape"] == [32, 1536, 2048]
    assert layout["attention"]["selected_cache"] == {
        "context_partition_axis": "expert",
        "replicated_axis": "feature",
        "exchange_axis": "expert",
        "physical_group_size": 8,
    }
    assert layout["attention"]["o_projection"]["reduction_axis"] == "expert"
    assert layout["dsa"]["score_head_reduction_axis"] == "expert"
    assert layout["forbidden"] == {
        "batch_32_decode_rows": True,
        "full_pod_hidden_reconstruction": True,
        "repeated_collective_group_size_32": True,
    }


def test_ws32_rejects_non_32_mesh() -> None:
    with pytest.raises(Exception, match="exactly 32"):
        Ws32MeshContract(expert_axis_size=4, feature_axis_size=4)


def test_ws32_physical_mesh_is_explicit_8x4_topology_mapping() -> None:
    mesh = build_ws32_physical_mesh(_topology())
    assert mesh.device_ids == (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (8, 9, 10, 11),
        (12, 13, 14, 15),
        (16, 17, 18, 19),
        (20, 21, 22, 23),
        (24, 25, 26, 27),
        (28, 29, 30, 31),
    )
    assert mesh.feature_groups == mesh.device_ids
    assert mesh.expert_groups == (
        (0, 4, 8, 12, 16, 20, 24, 28),
        (1, 5, 9, 13, 17, 21, 25, 29),
        (2, 6, 10, 14, 18, 22, 26, 30),
        (3, 7, 11, 15, 19, 23, 27, 31),
    )
    assert len(mesh.mesh_hash) == 64
