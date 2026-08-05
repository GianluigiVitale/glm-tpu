from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from glm_tpu.greenfield.kernels.reference import StageLocalKvLayout


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "greenfield"
    / "run_gate_c_equivalence.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "run_gate_c_equivalence", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_c_runner_owner_packers_roundtrip_partial_last_page() -> None:
    module = _load_script()
    layout = StageLocalKvLayout(
        logical_page_size=8,
        local_parallel_size=4,
        packed_cache_width=6,
    )
    history = np.arange(13 * 3, dtype=np.float32).reshape(13, 3)
    positions = np.arange(13, dtype=np.int32)
    packed_history, packed_positions = module.pack_history_by_owner(
        history, positions, layout=layout
    )
    assert packed_history.shape == (4, 4, 3)
    assert packed_positions.shape == (4, 4)
    assert int(np.count_nonzero(packed_positions == -1)) == 3
    np.testing.assert_array_equal(
        module.unpack_history_by_position(
            packed_history,
            packed_positions,
            context_length=13,
        ),
        history,
    )

    cache = np.arange(13 * 6, dtype=np.float32).reshape(13, 6)
    packed_cache = module.pack_cache_by_owner(cache, layout=layout)
    assert packed_cache.shape == (4, 2, 2, 6)
    np.testing.assert_array_equal(
        module.unpack_cache_by_position(
            packed_cache,
            context_length=13,
            layout=layout,
        ),
        cache,
    )


def test_gate_c_runner_packers_reject_ambiguous_rows() -> None:
    module = _load_script()
    layout = StageLocalKvLayout(
        logical_page_size=8,
        local_parallel_size=4,
        packed_cache_width=6,
    )
    with pytest.raises(ValueError, match="canonical and contiguous"):
        module.pack_history_by_owner(
            np.zeros((3, 2), dtype=np.float32),
            np.asarray([0, 2, 1], dtype=np.int32),
            layout=layout,
        )
    packed = np.zeros((4, 2, 1), dtype=np.float32)
    duplicate_positions = np.full((4, 2), -1, dtype=np.int32)
    duplicate_positions[0, 0] = 0
    duplicate_positions[1, 0] = 0
    with pytest.raises(ValueError, match="duplicate"):
        module.unpack_history_by_position(
            packed, duplicate_positions, context_length=1
        )


def test_gate_c_runner_canonical_selection_and_observed_gather() -> None:
    module = _load_script()
    selected = module._canonical_topk_from_scores(
        np.asarray([[1.0, 1.0, 2.0, 1.0]], dtype=np.float32),
        top_k=3,
    )
    np.testing.assert_array_equal(
        selected, np.asarray([[2, 0, 1]], dtype=np.int32)
    )
    assert module._selection_drift_record(
        selected, np.asarray([[2, 1, 3]], dtype=np.int32)
    ) == {
        "elementwise_match": False,
        "elementwise_mismatch_count": 2,
        "observed_only_count": 1,
        "overlap_count": 2,
        "reference_only_count": 1,
    }

    values = np.zeros((2, 1, 4, 2), dtype=np.float32)
    positions = np.full((2, 1, 4), -1, dtype=np.int32)
    counts = np.asarray([[2], [1]], dtype=np.int32)
    positions[0, 0, :2] = [4, 0]
    positions[1, 0, 0] = 3
    values[0, 0, :2] = [[4, 40], [0, 10]]
    values[1, 0, 0] = [3, 30]
    observed_values, observed_positions = module._reassemble_selected_cache(
        values, positions, counts
    )
    np.testing.assert_array_equal(observed_positions, [[0, 3, 4]])
    np.testing.assert_array_equal(
        observed_values, [[[0, 10], [3, 30], [4, 40]]]
    )
