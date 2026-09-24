from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from glm_tpu.exceptions import GeometryValidationError, TopologyValidationError
from glm_tpu.config.model import ModelGeometry
from glm_tpu.distributed.topology import PhysicalDevice, PhysicalTopology
# The pinned GLM-5.3 config; its geometry equals the archived GLM-5.2 file's (test_glm53_model).
from tools.equivalence.fixture import config_json


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def geometry() -> ModelGeometry:
    config = config_json()
    return ModelGeometry.from_hf_config(config)


def topology(*, reverse: bool = False) -> PhysicalTopology:
    devices = []
    device_id = 0
    for x in range(2):
        for y in range(4):
            for z in range(4):
                process = device_id // 4
                devices.append(
                    PhysicalDevice(
                        device_id=device_id,
                        process_index=process,
                        local_device_id=device_id % 4,
                        coordinates=(x, y, z),
                        core_on_chip=0,
                        platform="tpu",
                        device_kind="TPU v4",
                    )
                )
                device_id += 1
    if reverse:
        devices.reverse()
    return PhysicalTopology(
        slice_name="example-slice",
        topology_shape=(2, 4, 4),
        devices=tuple(devices),
    )


def test_checked_in_glm_geometry_is_exact(geometry: ModelGeometry) -> None:
    assert geometry.model_id == "zai-org/GLM-5.2-FP8"
    assert geometry.num_layers == 78
    assert geometry.first_dense_layers == 3
    assert geometry.hidden_size == 6144
    assert geometry.num_routed_experts == 256
    assert geometry.routed_top_k == 8
    assert geometry.dsa_top_k == 2048
    assert geometry.dsa_indexer_heads == 32
    assert geometry.index_share_group_size == 4
    assert geometry.qk_nope_head_dim == 192
    assert geometry.qk_rope_head_dim == 64
    assert geometry.v_head_dim == 256
    assert geometry.num_nextn_predict_layers == 1
    assert geometry.fp8_block_shape == (128, 128)
    assert geometry.weight_storage_dtype == "fp8:e4m3"
    assert len(geometry.mlp_layer_types) == len(geometry.indexer_types) == 78


def test_geometry_refuses_inconsistent_layer_schedule(
    geometry: ModelGeometry,
) -> None:
    with pytest.raises(GeometryValidationError, match="first_dense_layers"):
        replace(
            geometry,
            mlp_layer_types=("sparse",) + geometry.mlp_layer_types[1:],
        )


def test_geometry_refuses_boolean_integer(geometry: ModelGeometry) -> None:
    with pytest.raises(GeometryValidationError, match="hidden_size"):
        replace(geometry, hidden_size=True)


def test_topology_is_canonical_and_content_addressed() -> None:
    ordered = topology()
    reversed_input = topology(reverse=True)
    assert ordered == reversed_input
    assert ordered.topology_hash == reversed_input.topology_hash
    assert len(ordered.topology_hash) == 64
    assert ordered.process_indices == tuple(range(8))


def test_topology_refuses_duplicate_coordinates() -> None:
    good = topology()
    duplicate = replace(good.devices[1], coordinates=good.devices[0].coordinates)
    with pytest.raises(TopologyValidationError, match="coordinates"):
        replace(good, devices=(good.devices[0], duplicate, *good.devices[2:]))
