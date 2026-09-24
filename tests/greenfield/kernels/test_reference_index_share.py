from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.kernels.reference.index_share import (
    IndexShareSchedule,
    IndexShareState,
    pack_index_share_transfer,
    produce_index_share_state,
    resolve_index_share_layer,
    restore_index_share_transfer,
    validate_index_share_state_host,
)
from glm_tpu.optimized.reference.dsa import SelectedPositions
from glm_tpu.optimized.geometry import ModelGeometry


REPO = Path(__file__).resolve().parents[3]


def real_schedule() -> IndexShareSchedule:
    config = json.loads((REPO / "configs/glm-5.2-fp8-config.json").read_text())
    return IndexShareSchedule.from_geometry(ModelGeometry.from_hf_config(config))


def selected_fixture() -> SelectedPositions:
    return SelectedPositions(
        jnp.asarray([[7, 2, 5, -1]], dtype=jnp.int32),
        jnp.asarray([3], dtype=jnp.int32),
    )


def test_real_glm_index_share_schedule_and_pp8_crossings_are_exact() -> None:
    schedule = real_schedule()
    assert schedule.group_size == 4
    assert schedule.source_for(0) == 0
    assert schedule.source_for(1) == 1
    assert schedule.source_for(2) == 2
    assert schedule.source_for(3) == 2
    assert schedule.source_for(5) == 2
    assert schedule.source_for(6) == 6
    assert schedule.source_for(12) == 10
    assert schedule.source_for(31) == 30
    assert schedule.source_for(40) == 38
    assert schedule.source_for(59) == 58
    assert schedule.source_for(69) == 66
    assert schedule.crossings((0, 12, 22, 31, 40, 50, 59, 69)) == (
        12,
        31,
        40,
        59,
        69,
    )


def test_index_share_schedule_refuses_missing_or_overlong_producer_groups() -> None:
    with pytest.raises(ValueError, match="preceding full"):
        IndexShareSchedule(("shared",), 4)
    with pytest.raises(ValueError, match="exceeds"):
        IndexShareSchedule(("full", "shared", "shared"), 2)
    with pytest.raises(ValueError, match="full/shared"):
        IndexShareSchedule(("full", "skip"), 4)
    with pytest.raises(ValueError, match="sorted"):
        real_schedule().crossings((0, 12, 12))


def test_full_layer_produces_and_shared_layers_reuse_elementwise_state() -> None:
    schedule = IndexShareSchedule(("full", "shared", "shared", "full"), 3)
    fresh = selected_fixture()
    first_selection, state = resolve_index_share_layer(
        schedule,
        0,
        fresh_selection=fresh,
        shared_state=None,
    )
    reused, same_state = resolve_index_share_layer(
        schedule,
        1,
        fresh_selection=None,
        shared_state=state,
    )
    np.testing.assert_array_equal(
        np.asarray(first_selection.positions), [[7, 2, 5, -1]]
    )
    np.testing.assert_array_equal(np.asarray(reused.positions), [[7, 2, 5, -1]])
    np.testing.assert_array_equal(np.asarray(reused.valid_counts), [3])
    assert same_state is state

    replacement = SelectedPositions(
        jnp.asarray([[1, 0, -1, -1]], dtype=jnp.int32),
        jnp.asarray([2], dtype=jnp.int32),
    )
    _, replaced_state = resolve_index_share_layer(
        schedule,
        3,
        fresh_selection=replacement,
        shared_state=state,
    )
    assert replaced_state.source_layer == 3


def test_shared_layer_refuses_missing_fresh_or_wrong_producer_state() -> None:
    schedule = IndexShareSchedule(("full", "shared"), 2)
    with pytest.raises(ValueError, match="requires prior"):
        resolve_index_share_layer(
            schedule, 1, fresh_selection=None, shared_state=None
        )
    with pytest.raises(ValueError, match="must not compute"):
        resolve_index_share_layer(
            schedule,
            1,
            fresh_selection=selected_fixture(),
            shared_state=produce_index_share_state(selected_fixture(), source_layer=0),
        )
    with pytest.raises(ValueError, match="expected producer"):
        resolve_index_share_layer(
            schedule,
            1,
            fresh_selection=None,
            shared_state=IndexShareState(3, selected_fixture().positions),
        )


def test_cross_stage_payload_is_only_score_ordered_int32_positions() -> None:
    positions = jnp.concatenate(
        (
            jnp.arange(1536, dtype=jnp.int32)[None, ::-1],
            jnp.full((1, 512), -1, dtype=jnp.int32),
        ),
        axis=1,
    )
    selected = SelectedPositions(positions, jnp.asarray([1536], dtype=jnp.int32))
    state = produce_index_share_state(selected, source_layer=10)
    payload = pack_index_share_transfer(state)
    assert payload.shape == (1, 2048)
    assert payload.dtype == jnp.int32
    assert payload.size * payload.dtype.itemsize == 8192
    restored = restore_index_share_transfer(payload, source_layer=10)
    np.testing.assert_array_equal(np.asarray(restored.positions), np.asarray(positions))
    np.testing.assert_array_equal(np.asarray(restored.selection.valid_counts), [1536])
    validate_index_share_state_host(restored, expected_source_layer=10)


def test_host_validation_refuses_malformed_tail_duplicate_or_event_identity() -> None:
    with pytest.raises(ValueError, match="-1 tail"):
        validate_index_share_state_host(
            IndexShareState(
                2, jnp.asarray([[1, -1, 3, -1]], dtype=jnp.int32)
            )
        )
    with pytest.raises(ValueError, match="duplicate"):
        validate_index_share_state_host(
            IndexShareState(
                2, jnp.asarray([[1, 1, -1, -1]], dtype=jnp.int32)
            )
        )
    with pytest.raises(ValueError, match="producer identity"):
        validate_index_share_state_host(
            produce_index_share_state(selected_fixture(), source_layer=2),
            expected_source_layer=6,
        )


def test_index_share_shape_and_dtype_contracts_fail_loudly() -> None:
    with pytest.raises(ValueError, match="dtype int32"):
        IndexShareState(0, jnp.ones((1, 4), dtype=jnp.float32))
    with pytest.raises(ValueError, match="positive_width"):
        IndexShareState(0, jnp.ones((1, 0), dtype=jnp.int32))
    with pytest.raises(ValueError, match="fresh selection"):
        resolve_index_share_layer(
            IndexShareSchedule(("full",), 1),
            0,
            fresh_selection=None,
            shared_state=None,
        )
    state = IndexShareState(
        0, jnp.full((1, 4), -1, dtype=jnp.int32)
    )
    with pytest.raises(ValueError, match=r"int32\[1,2048\]"):
        pack_index_share_transfer(state)
    with pytest.raises(ValueError, match=r"int32\[1,2048\]"):
        restore_index_share_transfer(state.positions, source_layer=0)
