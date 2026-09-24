from __future__ import annotations

import json
from pathlib import Path

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from glm_tpu.distributed.mesh import MeshContract, build_physical_mesh
from glm_tpu.config.model import ModelGeometry
from glm_tpu.distributed.topology import PhysicalDevice, PhysicalTopology

# The pinned GLM-5.3 config; its geometry equals the archived GLM-5.2 file's (tests/config/test_model.py).
from tools.equivalence.fixture import config_json
from glm_tpu.layers.norm import final_norm, rms_norm
from tests.reference.norm import fused_add_rms_norm


REPO = Path(__file__).resolve().parents[2]


def _topology() -> PhysicalTopology:
    devices = []
    for device_id, coordinates in enumerate((x, y, z) for x in range(2) for y in range(4) for z in range(4)):
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
    exact_geometry = ModelGeometry.from_hf_config(config_json())
    contract = MeshContract()
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
        MeshContract(expert_axis_size=4, feature_axis_size=4)


def test_ws32_physical_mesh_is_explicit_8x4_topology_mapping() -> None:
    mesh = build_physical_mesh(_topology())
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


def test_rms_norm_matches_glm_fp32_then_activation_rounding() -> None:
    hidden = jnp.asarray([[1.0, -2.0, 3.0, -4.0]], dtype=jnp.bfloat16)
    weight = jnp.asarray([1.0, 0.5, -0.25, 2.0], dtype=jnp.bfloat16)
    got = rms_norm(hidden, weight, epsilon=1e-5)
    value = hidden.astype(jnp.float32)
    normalized = value * jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + 1e-5)
    expected = normalized.astype(jnp.bfloat16) * weight
    assert got.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(final_norm(hidden, weight, epsilon=1e-5)), np.asarray(got))


def test_rms_norm_refuses_shape_dtype_and_epsilon_drift() -> None:
    with pytest.raises(ValueError, match="weight"):
        rms_norm(jnp.ones((1, 4)), jnp.ones((3,)), epsilon=1e-5)
    with pytest.raises(ValueError, match="epsilon"):
        rms_norm(jnp.ones((1, 4)), jnp.ones((4,)), epsilon=0)
    with pytest.raises(ValueError, match="inexact"):
        rms_norm(jnp.ones((1, 4), jnp.int32), jnp.ones((4,)), epsilon=1e-5)


def test_fused_add_rms_norm_uses_unrounded_fp32_sum() -> None:
    rng = np.random.default_rng(0)
    hidden = jnp.asarray(rng.normal(size=(1, 64)) * 3.0, dtype=jnp.bfloat16)
    residual = jnp.asarray(rng.normal(size=(1, 64)) * 3.0, dtype=jnp.bfloat16)
    weight = jnp.asarray(rng.normal(loc=1.0, scale=0.1, size=(64,)), dtype=jnp.bfloat16)

    normalized, carried = fused_add_rms_norm(hidden, residual, weight, epsilon=1e-5)
    summed = hidden.astype(jnp.float32) + residual.astype(jnp.float32)
    expected_carried = summed.astype(jnp.bfloat16)
    expected_normalized = (
        (summed * jax.lax.rsqrt(jnp.mean(summed * summed, axis=-1, keepdims=True) + 1e-5)).astype(jnp.bfloat16) * weight
    ).astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(carried), np.asarray(expected_carried))
    np.testing.assert_array_equal(np.asarray(normalized), np.asarray(expected_normalized))

    rounded_first = rms_norm(expected_carried, weight, epsilon=1e-5)
    mismatch_count = int(
        jnp.count_nonzero(
            jax.lax.bitcast_convert_type(normalized, jnp.uint16)
            != jax.lax.bitcast_convert_type(rounded_first, jnp.uint16)
        )
    )
    assert mismatch_count == 18


def test_fused_add_rms_norm_refuses_state_contract_drift() -> None:
    hidden = jnp.ones((1, 4), dtype=jnp.bfloat16)
    residual = jnp.ones((1, 4), dtype=jnp.bfloat16)
    weight = jnp.ones((4,), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="shapes"):
        fused_add_rms_norm(hidden, residual[:, :3], weight, epsilon=1e-5)
    with pytest.raises(ValueError, match="dtypes"):
        fused_add_rms_norm(hidden, residual.astype(jnp.float32), weight, epsilon=1e-5)
    with pytest.raises(ValueError, match="weight"):
        fused_add_rms_norm(hidden, residual, weight[:3], epsilon=1e-5)
    with pytest.raises(ValueError, match="epsilon"):
        fused_add_rms_norm(hidden, residual, weight, epsilon=0)
