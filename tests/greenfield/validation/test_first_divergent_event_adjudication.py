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
        MODULE._first_divergent_event(archive, oracle_dir)
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


ARCHIVE_0827 = Path(
    "/home/gianl/glm-run/greenfield_ws32_short_decoder_8k_numerical_20260827T011711674195301Z/runner.rank0.npz"
)
ORACLE_8K = Path(
    "/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/8k/"
    "greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z/oracle"
)
REFERENCE_ROW = ROOT / "docs/artifacts/gate-d-event1-fp64-reference-row-20260905.npy"


@pytest.mark.skipif(
    not (ARCHIVE_0827.is_file() and (ORACLE_8K / "dsa_events.safetensors").is_file()),
    reason="the 2026-08-27 observer archive or the sealed 8K DSA oracle is unavailable",
)
def test_reproduces_the_sealed_event1_adjudication_exactly() -> None:
    """The tool must reproduce the numbers of the adjudication that closed Gate D
    (docs/artifacts/gate-d-event1-math-reference-adjudication-20260905.json)."""
    sealed = json.loads(
        (ROOT / "docs/artifacts/gate-d-event1-math-reference-adjudication-20260905.json").read_text()
    )["event1_vs_math_reference"]
    analysis = MODULE.adjudicate(
        archive=ARCHIVE_0827,
        oracle_dir=ORACLE_8K,
        reference_row=REFERENCE_ROW,
        step=0,
        event=1,
        decode_position=8155,
        expected_producer_layer_id=1,
    )
    assert analysis["epsilon_oracle_vs_reference"] == sealed["item3"]["eps_event"]
    assert analysis["engine_delta"]["max_abs"] == sealed["item3"]["engine_max_abs"]
    assert analysis["oracle_delta"]["std"] == sealed["item3"]["std_o"]
    assert analysis["engine_delta"]["std"] == sealed["item3"]["std_e"]
    assert analysis["reference_band_size"] == sealed["item3"]["ref_band_size"] == 451
    assert analysis["cutoff_reference_score"] == sealed["item3"]["reference_cutoff"]
    assert analysis["shared_positions"] == sealed["item4"]["n"] == 2041
    assert analysis["checks"]["bias"]["bound"] == sealed["item4"]["bound"]
    assert sorted(analysis["expected_only"] + analysis["observed_only"]) == sealed["item3"]["sym_diff"]
    assert analysis["expected_only"] == [680, 1052, 2024, 2436, 6322, 7473, 7850]
    assert analysis["verdict"] == "PASS"
    assert all(item["pass"] for item in analysis["checks"].values())
    # The event is auto-detectable from the same inputs.
    assert MODULE._first_divergent_event(ARCHIVE_0827, ORACLE_8K) == (0, 1)


@pytest.mark.skipif(
    not (ARCHIVE_0827.is_file() and (ORACLE_8K / "dsa_events.safetensors").is_file()),
    reason="the 2026-08-27 observer archive or the sealed 8K DSA oracle is unavailable",
)
def test_reference_row_must_be_bound_to_the_adjudicated_event() -> None:
    """A reference row built for another decode position or producer layer must
    refuse rather than silently shift the cutoff."""
    with pytest.raises(SystemExit, match="does not match decode position"):
        MODULE.adjudicate(
            archive=ARCHIVE_0827, oracle_dir=ORACLE_8K, reference_row=REFERENCE_ROW,
            step=0, event=1, decode_position=8154,
        )
    with pytest.raises(SystemExit, match="reference row was built for layer"):
        MODULE.adjudicate(
            archive=ARCHIVE_0827, oracle_dir=ORACLE_8K, reference_row=REFERENCE_ROW,
            step=0, event=1, decode_position=8155, expected_producer_layer_id=2,
        )


def test_the_band_capacity_rule_refuses_more_swaps_than_the_band_holds(tmp_path: Path) -> None:
    """§21.2 item 3: the reference ambiguity band cannot explain more swaps than
    it contains. A tight band with a wide swap set must refuse."""
    import numpy as np
    from safetensors.numpy import save_file

    top_k = 8
    # Reference scores far apart, so the band around the cutoff holds few positions.
    reference = np.asarray([100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 29.999, 20.0], dtype=np.float64)
    reference_row = tmp_path / "reference.npy"
    np.save(reference_row, reference)
    oracle_positions = np.arange(top_k, dtype=np.int64)
    engine_positions = oracle_positions.copy()
    engine_positions[top_k - 1] = 8
    archive = tmp_path / "runner.rank0.npz"
    np.savez(
        archive,
        dsa_selected_positions=engine_positions.reshape(1, 1, 1, top_k).astype(np.int32),
        dsa_selected_scores=reference[engine_positions].reshape(1, 1, 1, top_k).astype(np.float32),
        dsa_selected_valid_counts=np.full((1, 1, 1), top_k, dtype=np.int32),
        dsa_producer_layer_ids=np.asarray([1], dtype=np.int32),
    )
    oracle_dir = tmp_path / "oracle"
    oracle_dir.mkdir()
    save_file(
        {
            "selected_positions": oracle_positions.reshape(1, 1, top_k).astype(np.int32),
            "selected_scores": reference[oracle_positions].reshape(1, 1, top_k).astype(np.float32),
            "valid_counts": np.full((1, 1), top_k, dtype=np.int32),
        },
        str(oracle_dir / "dsa_events.safetensors"),
    )
    analysis = MODULE.adjudicate(
        archive=archive, oracle_dir=oracle_dir, reference_row=reference_row, step=0, event=0
    )
    # The oracle scores are the reference exactly, so eps is 0 and the band holds
    # only the cutoff position itself: it cannot explain two swapped positions.
    capacity = analysis["checks"]["reference_band_capacity"]
    assert capacity["band_size"] == 1 and capacity["swapped"] == 2
    assert capacity["pass"] is False
    assert analysis["verdict"] == "FAIL"
    assert analysis["checks"]["equal_sized_disjoint_swap"]["pass"] is True
    assert "reference_band_size" in analysis and analysis["std_ddof"] == 0


def _multi_event_fixture(tmp_path: Path, *, divergent_at: tuple[int, int]) -> tuple[Path, Path]:
    """A 3x4 observation window whose only divergence is at ``divergent_at``."""
    steps, events, top_k = 3, 4, 8
    positions = np.tile(np.arange(top_k, dtype=np.int32), (steps, events, 1))
    scores = np.tile(
        np.linspace(1.0, 0.1, top_k, dtype=np.float32), (steps, events, 1)
    )
    engine_positions = positions.copy()
    engine_positions[divergent_at[0], divergent_at[1], top_k - 1] = top_k
    archive = tmp_path / "runner.rank0.npz"
    np.savez(
        archive,
        dsa_selected_positions=engine_positions[:, :, None, :],
        dsa_selected_scores=scores[:, :, None, :],
        dsa_selected_valid_counts=np.full((steps, events, 1), top_k, dtype=np.int32),
        dsa_producer_layer_ids=np.ones(events, dtype=np.int32),
    )
    oracle_dir = tmp_path / "oracle"
    oracle_dir.mkdir()
    from safetensors.numpy import save_file

    save_file(
        {
            "selected_positions": positions,
            "selected_scores": scores,
            "valid_counts": np.full((steps, events), top_k, dtype=np.int32),
        },
        str(oracle_dir / "dsa_events.safetensors"),
    )
    return archive, oracle_dir


def test_the_scan_window_comes_from_the_archives_not_the_operator(tmp_path: Path) -> None:
    """P1-3: a truncated window could pre-register the wrong event."""
    archive, oracle_dir = _multi_event_fixture(tmp_path, divergent_at=(2, 3))
    assert MODULE._scan_window(archive, oracle_dir) == (3, 4)
    assert MODULE._first_divergent_event(archive, oracle_dir) == (2, 3)


def test_a_window_disagreement_between_engine_and_oracle_is_refused(tmp_path: Path) -> None:
    archive, oracle_dir = _multi_event_fixture(tmp_path, divergent_at=(0, 1))
    from safetensors.numpy import load_file, save_file

    data = load_file(str(oracle_dir / "dsa_events.safetensors"))
    save_file(
        {name: value[:, :3] for name, value in data.items()},
        str(oracle_dir / "dsa_events.safetensors"),
    )
    with pytest.raises(SystemExit, match="does not match the oracle"):
        MODULE._scan_window(archive, oracle_dir)


def test_naming_a_later_benign_event_is_refused(tmp_path: Path) -> None:
    """P2-4: the record claims every earlier event is exact, so it is checked."""
    archive, oracle_dir = _multi_event_fixture(tmp_path, divergent_at=(0, 1))
    assert MODULE._resolve_event(archive, oracle_dir, None, None) == (0, 1)
    assert MODULE._resolve_event(archive, oracle_dir, 0, 1) == (0, 1)
    with pytest.raises(SystemExit, match="is not the first divergent event"):
        MODULE._resolve_event(archive, oracle_dir, 2, 3)
    with pytest.raises(SystemExit, match="must be given together"):
        MODULE._resolve_event(archive, oracle_dir, 0, None)


def test_the_reference_row_must_be_a_committed_pre_registered_artifact(tmp_path: Path) -> None:
    """P1-1: an unbound row lets a fitted reference produce a clean PASS."""
    committed = ROOT / "docs/artifacts/gate-d-event1-fp64-reference-row-20260905.npy"
    digest = sha256(committed.read_bytes()).hexdigest()
    resolved, relative, seen = MODULE._committed_reference_row(committed, digest)
    assert resolved == committed.resolve()
    assert relative == "docs/artifacts/gate-d-event1-fp64-reference-row-20260905.npy"
    assert seen == digest

    with pytest.raises(SystemExit, match="does not match the pre-registered"):
        MODULE._committed_reference_row(committed, "0" * 64)

    outside = tmp_path / "reference.npy"
    np.save(outside, np.zeros(4, dtype=np.float64))
    with pytest.raises(SystemExit, match="must be a committed"):
        MODULE._committed_reference_row(outside, sha256(outside.read_bytes()).hexdigest())

    untracked = ROOT / "docs/artifacts/gate-d-untracked-probe-row.npy"
    assert not untracked.exists(), "leftover probe artifact from an earlier run"
    np.save(untracked, np.zeros(4, dtype=np.float64))
    try:
        with pytest.raises(SystemExit, match="not tracked by git"):
            MODULE._committed_reference_row(
                untracked, sha256(untracked.read_bytes()).hexdigest()
            )
    finally:
        untracked.unlink()


def test_the_observer_archive_must_belong_to_the_declared_run(tmp_path: Path) -> None:
    """P2-3: otherwise --engine-source-run is free text the seal guard trusts."""
    tag = "greenfield_ws32_short_decoder_8k_numerical_20260905T000000000000000Z"
    run_dir = tmp_path / tag
    run_dir.mkdir()
    archive = run_dir / "runner.rank0.npz"
    archive.write_bytes(b"")
    assert MODULE._archive_belongs_to_run(archive, tag) is True
    assert MODULE._archive_belongs_to_run(archive, "another_run") is False

    loose = tmp_path / "elsewhere"
    loose.mkdir()
    moved = loose / "runner.rank0.npz"
    moved.write_bytes(b"")
    assert MODULE._archive_belongs_to_run(moved, tag) is False
    (loose / "summary.json").write_text(json.dumps({"run_tag": tag}), encoding="utf-8")
    assert MODULE._archive_belongs_to_run(moved, tag) is True


def test_undisclosed_earlier_attempts_on_the_same_event_are_refused(tmp_path: Path) -> None:
    """P2-5: otherwise an operator can re-run until a PASS appears."""
    analysis = {
        "artifact_kind": MODULE.ANALYSIS_ARTIFACT_KIND,
        "engine_source_run": "run_a",
        "event_index": 1,
        "step": 0,
        "verdict": "FAIL",
    }
    earlier = tmp_path / "attempt-one.json"
    earlier.write_text(json.dumps(analysis), encoding="utf-8")
    output = tmp_path / "attempt-two.json"

    with pytest.raises(SystemExit, match="not declared in --basis"):
        MODULE._require_prior_attempts_disclosed(output, analysis, [])

    disclosed = [{"path": str(earlier), "sha256": "0" * 64}]
    MODULE._require_prior_attempts_disclosed(output, analysis, disclosed)

    other_event = dict(analysis, event_index=2)
    MODULE._require_prior_attempts_disclosed(output, other_event, [])


def test_the_bias_bound_moves_with_kappa(tmp_path: Path) -> None:
    """P1-2: §21.2 item 4 requires the same pre-registered kappa as item 3."""
    archive, oracle_dir, reference_row = _fixture(tmp_path, engine_bias=0.0, swap=True)
    at_two = MODULE.adjudicate(
        archive=archive, oracle_dir=oracle_dir, reference_row=reference_row,
        step=0, event=0, kappa=2.0,
    )
    at_one = MODULE.adjudicate(
        archive=archive, oracle_dir=oracle_dir, reference_row=reference_row,
        step=0, event=0, kappa=1.0,
    )
    oracle_mean_abs = abs(at_two["oracle_delta"]["mean"])
    assert oracle_mean_abs > 0.0
    assert at_two["checks"]["bias"]["bound"] > at_one["checks"]["bias"]["bound"]
    assert at_one["bias_factor"] == 1.0 and at_two["bias_factor"] == 2.0


def test_repeated_selected_positions_are_refused(tmp_path: Path) -> None:
    archive, oracle_dir, reference_row = _fixture(tmp_path, engine_bias=0.0, swap=True)
    with np.load(archive, allow_pickle=False) as handle:
        arrays = {name: handle[name] for name in handle.files}
    arrays["dsa_selected_positions"][0, 0, 0, 1] = arrays["dsa_selected_positions"][0, 0, 0, 0]
    broken = tmp_path / "repeated.npz"
    np.savez(broken, **arrays)
    with pytest.raises(SystemExit, match="positions repeat"):
        MODULE.adjudicate(
            archive=broken, oracle_dir=oracle_dir, reference_row=reference_row, step=0, event=0
        )
