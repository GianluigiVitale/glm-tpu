"""Spec §21.2 items 3-4: the offline first-divergent-event adjudicator."""
from __future__ import annotations

import importlib.util
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/greenfield/adjudicate_ws32_first_divergent_event.py"
SPEC = importlib.util.spec_from_file_location("ws32_event_adjudicator", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _fixture(tmp_path: Path, *, engine_bias: float, swap: bool) -> tuple[Path, Path, Path]:
    """One synthetic event: 64 reference scores, top-32 selected, optional boundary swap."""
    rng = np.random.RandomState(7)
    reference = np.sort(rng.uniform(0.0, 1.0, size=64))[::-1].astype(np.float64).copy()
    top_k = 32
    oracle_positions = np.arange(top_k, dtype=np.int64)
    oracle_scores = reference[:top_k] + rng.normal(0.0, 1e-4, size=top_k)
    engine_positions = oracle_positions.copy()
    if swap:
        # Exchange the last selected position for the first rejected one; both
        # sit inside the reference cutoff band.
        reference[top_k - 1] = reference[top_k] + 1e-6
        engine_positions[top_k - 1] = top_k
    engine_scores = reference[engine_positions] + engine_bias + rng.normal(0.0, 1e-4, size=top_k)

    reference_row = tmp_path / "reference.npy"
    np.save(reference_row, reference)
    archive = tmp_path / "runner.rank0.npz"
    np.savez(
        archive,
        dsa_selected_positions=engine_positions.reshape(1, 1, 1, top_k).astype(np.int32),
        dsa_selected_scores=engine_scores.reshape(1, 1, 1, top_k).astype(np.float32),
        dsa_selected_valid_counts=np.full((1, 1, 1), top_k, dtype=np.int32),
        dsa_producer_layer_ids=np.asarray([1], dtype=np.int32),
    )
    oracle_dir = tmp_path / "oracle"
    oracle_dir.mkdir()
    from safetensors.numpy import save_file

    save_file(
        {
            "selected_positions": oracle_positions.reshape(1, 1, top_k).astype(np.int32),
            "selected_scores": oracle_scores.reshape(1, 1, top_k).astype(np.float32),
            "valid_counts": np.full((1, 1), top_k, dtype=np.int32),
        },
        str(oracle_dir / "dsa_events.safetensors"),
    )
    return archive, oracle_dir, reference_row


def test_boundary_swap_within_the_oracles_own_error_is_adjudicated(tmp_path: Path) -> None:
    archive, oracle_dir, reference_row = _fixture(tmp_path, engine_bias=0.0, swap=True)
    analysis = MODULE.adjudicate(
        archive=archive, oracle_dir=oracle_dir, reference_row=reference_row, step=0, event=0
    )
    assert analysis["verdict"] == "PASS"
    assert analysis["expected_only"] == [31] and analysis["observed_only"] == [32]
    assert analysis["producer_layer_id"] == 1
    assert analysis["checks"]["equal_sized_disjoint_swap"]["pass"] is True
    assert analysis["checks"]["reference_band"]["pass"] is True
    assert json.dumps(analysis)


def test_an_engine_bias_beyond_the_caps_is_refused(tmp_path: Path) -> None:
    archive, oracle_dir, reference_row = _fixture(tmp_path, engine_bias=0.05, swap=True)
    analysis = MODULE.adjudicate(
        archive=archive, oracle_dir=oracle_dir, reference_row=reference_row, step=0, event=0
    )
    assert analysis["verdict"] == "FAIL"
    assert analysis["checks"]["cap_max_abs"]["pass"] is False
    assert analysis["checks"]["bias"]["pass"] is False


def test_an_exact_event_is_not_adjudicable(tmp_path: Path) -> None:
    archive, oracle_dir, reference_row = _fixture(tmp_path, engine_bias=0.0, swap=False)
    with pytest.raises(SystemExit, match="no divergent DSA event"):
        MODULE._first_divergent_event(archive, oracle_dir, 1, 1)
    analysis = MODULE.adjudicate(
        archive=archive, oracle_dir=oracle_dir, reference_row=reference_row, step=0, event=0
    )
    # An exact event has no swap, so the equal-sized-swap requirement refuses it
    # as an adjudication basis: §21.2 adjudicates a divergence, not an identity.
    assert analysis["checks"]["equal_sized_disjoint_swap"]["pass"] is False
    assert analysis["verdict"] == "FAIL"


def test_the_emitted_record_matches_the_loader_schema(tmp_path: Path) -> None:
    from glm_tpu.greenfield.validation.ws32_short_context import (
        load_ws32_adjudicated_divergence,
    )

    record = {
        "artifact_kind": "gate_d_ws32_8k_adjudicated_divergence",
        "schema_version": 1,
        "spec_section": "21.2 items 3-4 (scope: first divergent event)",
        "date_utc": "2026-09-05",
        "context": "8k",
        "decode_position": 8155,
        "step": 0,
        "event_index": 1,
        "producer_layer_id": 1,
        "expected_only": [31],
        "observed_only": [32],
        "later_event_alarm": 1024,
        "basis": [
            {
                "path": "docs/artifacts/gate-d-event1-fp64-reference-row-20260905.npy",
                "sha256": sha256(
                    (ROOT / "docs/artifacts/gate-d-event1-fp64-reference-row-20260905.npy").read_bytes()
                ).hexdigest(),
            }
        ],
        "oracle": {"dsa_manifest_sha256": "a" * 64, "token_manifest_sha256": "c" * 64},
        "engine_source_run": "greenfield_ws32_short_decoder_8k_numerical_20260905T000000000000000Z",
        "semantics": "test",
        "gate_d_closed": False,
        "performance_claim": False,
    }
    path = tmp_path / "record.json"
    MODULE._write_once(path, record)
    loaded = load_ws32_adjudicated_divergence(
        path, expected_sha256=sha256(path.read_bytes()).hexdigest(), repository_root=ROOT
    )
    assert loaded.step == 0 and loaded.event_index == 1
    assert loaded.expected_only == (31,) and loaded.observed_only == (32,)
    with pytest.raises(SystemExit, match="append-only"):
        MODULE._write_once(path, record)
