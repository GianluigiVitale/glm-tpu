from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from glm_tpu.greenfield.validation.layer0_dsa_association import (
    inspect_greenfield_layer0_dsa_internal_observation,
    inspect_greenfield_layer0_dsa_selected_observation,
    pack_stage_local_index_keys,
    stitch_stage_local_scores,
)


def _digest(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _file_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_internal_observation(root: Path) -> tuple[str, str]:
    arrays = {
        "normalized_hidden_bfloat16_bits": np.arange(
            21 * 6144, dtype=np.uint16
        ).reshape(21, 6144),
        "q_a_state_bfloat16_bits": np.arange(
            21 * 2048, dtype=np.uint16
        ).reshape(21, 2048),
        "query": np.arange(21 * 32 * 128, dtype=np.float32).reshape(
            21, 32, 128
        ),
        "head_weights": np.arange(21 * 32, dtype=np.float32).reshape(21, 32),
        "current_key": np.arange(21 * 128, dtype=np.float32).reshape(21, 128),
        "producer_layer_ids": np.asarray(
            [0, 1, 2, *range(6, 75, 4)], dtype=np.int32
        ),
        "decode_position": np.asarray([8155], dtype=np.int32),
    }
    tensor_path = root / "position_8155_internals.npz"
    np.savez(tensor_path, **arrays)
    names = {
        "normalized_hidden": (
            "normalized_hidden_bfloat16_bits",
            "bfloat16",
        ),
        "q_a_state": ("q_a_state_bfloat16_bits", "bfloat16"),
        "query": ("query", "float32"),
        "head_weights": ("head_weights", "float32"),
        "current_key": ("current_key", "float32"),
        "producer_layer_ids": ("producer_layer_ids", "int32"),
    }
    fields = {
        logical: {
            "dtype": dtype,
            "sha256": _digest(arrays[storage]),
            "shape": list(arrays[storage].shape),
        }
        for logical, (storage, dtype) in names.items()
    }
    layer0 = {}
    for logical, (storage, _) in names.items():
        if logical == "producer_layer_ids":
            continue
        value = arrays[storage][0]
        digest = _digest(value)
        layer0[logical] = {
            "actual_sha256": digest,
            "elementwise_exact": True,
            "expected_sha256": digest,
            "max_abs": 0.0,
            "mismatch_count": 0,
            "shape": list(value.shape),
        }
    contract = {
        "decode_position": 8155,
        "event_count": 21,
        "field_records": fields,
        "lane_mismatches": [],
        "layer0_query_exact": True,
        "layer0_reference": layer0,
        "padded_slot_mismatches": [],
        "producer_layer_ids": arrays["producer_layer_ids"].tolist(),
    }
    contract_path = root / "contract.json"
    contract_path.write_text(json.dumps(contract, sort_keys=True) + "\n")
    return _file_digest(contract_path), _file_digest(tensor_path)


def test_internal_observation_round_trips_and_refuses_drift(
    tmp_path: Path,
) -> None:
    contract_sha, tensor_sha = _write_internal_observation(tmp_path)
    contract, arrays = inspect_greenfield_layer0_dsa_internal_observation(
        tmp_path,
        expected_contract_sha256=contract_sha,
        expected_tensor_sha256=tensor_sha,
    )
    assert contract["layer0_query_exact"] is True
    assert arrays["producer_layer_ids"][0] == 0
    with pytest.raises(ValueError, match="tensor identity"):
        inspect_greenfield_layer0_dsa_internal_observation(
            tmp_path,
            expected_contract_sha256=contract_sha,
            expected_tensor_sha256="0" * 64,
        )


def test_selected_observation_round_trips_and_refuses_lane_drift(
    tmp_path: Path,
) -> None:
    positions = np.arange(2048, dtype=np.int32)
    scores = np.linspace(3.0, 1.0, 2048, dtype=np.float32)
    row = np.full((4098,), -1, dtype=np.int32)
    row[:2048] = positions
    row[2048:4096] = scores.view(np.int32)
    row[4096:] = (2048, 0)
    observation = np.full((32, 5, 4098), -1, dtype=np.int32)
    observation[:4, 0] = row
    path = tmp_path / "step_00_position_8155.npz"
    np.savez(
        path,
        decode_position=np.asarray([8155], dtype=np.int32),
        observation=observation,
        token_observation=np.full((32, 32), -1, dtype=np.int32),
    )
    actual_positions, actual_scores = (
        inspect_greenfield_layer0_dsa_selected_observation(
            path, expected_sha256=_file_digest(path)
        )
    )
    np.testing.assert_array_equal(actual_positions, positions)
    np.testing.assert_array_equal(actual_scores, scores)

    observation[1, 0, 0] = 99
    np.savez(
        path,
        decode_position=np.asarray([8155], dtype=np.int32),
        observation=observation,
        token_observation=np.full((32, 32), -1, dtype=np.int32),
    )
    with pytest.raises(ValueError, match="lanes disagree"):
        inspect_greenfield_layer0_dsa_selected_observation(
            path, expected_sha256=_file_digest(path)
        )


def test_lp4_cache_pack_and_score_stitch_are_exact_inverses() -> None:
    prompt = np.arange(9 * 3, dtype=np.uint16).reshape(9, 3)
    current = np.asarray([1.0, -2.0, 0.25], dtype=np.float32)
    global_bits, lanes, positions = pack_stage_local_index_keys(
        prompt,
        current,
        logical_page_size=8,
        local_parallel_size=2,
    )
    assert global_bits.shape == (10, 3)
    assert lanes.shape == (2, 8, 3)
    expected_current = current.astype(ml_dtypes.bfloat16).view(np.uint16)
    np.testing.assert_array_equal(global_bits[-1], expected_current)
    lane_scores = positions.astype(np.float32)
    lane_scores[positions < 0] = 0.0
    logical = stitch_stage_local_scores(lane_scores, positions, context=10)
    np.testing.assert_array_equal(logical, np.arange(10, dtype=np.float32))

    duplicate = positions.copy()
    duplicate[1, 0] = duplicate[0, 0]
    with pytest.raises(ValueError, match="incomplete"):
        stitch_stage_local_scores(lane_scores, duplicate, context=10)
