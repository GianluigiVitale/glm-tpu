from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from glm_tpu.greenfield.errors import PartitioningValidationError
from glm_tpu.greenfield.partitioning import (
    MemoryPolicy,
    layer_indexer_cache_bytes,
    layer_kv_cache_bytes,
    stage_runtime_memory,
)
from glm_tpu.greenfield.types import ModelGeometry


REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def geometry() -> ModelGeometry:
    config = json.loads((REPO / "configs/glm-5.2-fp8-config.json").read_text())
    return ModelGeometry.from_hf_config(config)


def test_256k_mla_cache_is_context_sharded_only_inside_local_stage(
    geometry: ModelGeometry,
) -> None:
    policy = MemoryPolicy()
    assert layer_kv_cache_bytes(
        geometry, local_parallel_size=4, policy=policy
    ) == 65_536 * 640 * 2
    assert layer_kv_cache_bytes(
        geometry, local_parallel_size=2, policy=policy
    ) == 131_072 * 640 * 2


def test_only_full_indexer_layers_own_key_cache(geometry: ModelGeometry) -> None:
    policy = MemoryPolicy()
    assert geometry.indexer_types[3] == "shared"
    assert geometry.indexer_types[6] == "full"
    assert layer_indexer_cache_bytes(
        geometry, 3, local_parallel_size=4, policy=policy
    ) == 0
    assert layer_indexer_cache_bytes(
        geometry, 6, local_parallel_size=4, policy=policy
    ) == 65_536 * 128 * 2


def test_stage_memory_has_explicit_cache_dsa_transport_and_measured_floor(
    geometry: ModelGeometry,
) -> None:
    policy = MemoryPolicy()
    result = stage_runtime_memory(
        geometry,
        layer_start=6,
        layer_end_exclusive=10,
        local_parallel_size=4,
        policy=policy,
    )
    assert result.kv_cache_bytes == 4 * 65_536 * 640 * 2
    assert result.indexer_cache_bytes == 65_536 * 128 * 2
    assert result.selected_index_bytes == 2048 * 4 * 2
    assert result.transport_buffer_bytes == 6144 * 2 + 2048 * 4
    assert result.temporary_floor_bytes == 779_642_880
    assert result.reserved_overlay_bytes == 0
    assert result.accounted_bytes == sum(
        (
            result.kv_cache_bytes,
            result.dsa_state_bytes,
            result.temporary_floor_bytes,
            result.reserved_overlay_bytes,
        )
    )


def test_memory_policy_hash_covers_unknown_overlay_state() -> None:
    policy = MemoryPolicy()
    measured = replace(
        policy,
        reserved_overlay_bytes=123,
        full_decoder_overlay_measured=True,
        provenance="full-decoder-measurement",
    )
    assert policy.policy_sha256 != measured.policy_sha256
    assert not policy.full_decoder_overlay_measured


def test_memory_model_refuses_rope_truncation_or_excess_context(
    geometry: ModelGeometry,
) -> None:
    with pytest.raises(PartitioningValidationError, match="truncate"):
        layer_kv_cache_bytes(
            geometry,
            local_parallel_size=4,
            policy=MemoryPolicy(kv_rope_cache_width=32),
        )
    with pytest.raises(PartitioningValidationError, match="maximum"):
        layer_kv_cache_bytes(
            geometry,
            local_parallel_size=4,
            policy=MemoryPolicy(target_context_length=2_000_000),
        )
