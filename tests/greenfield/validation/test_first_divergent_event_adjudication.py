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

from glm_tpu.greenfield.validation.ws32_short_context import (  # noqa: E402
    ANALYSIS_ARTIFACT_KIND,
    REFERENCE_ROWS,
)

REGISTERED = REFERENCE_ROWS[("8k", 8155, 1, "rms_norm_eps_1e-5")]

ARCHIVE_0827 = Path(
    "/home/gianl/glm-run/greenfield_ws32_short_decoder_8k_numerical_20260827T011711674195301Z/runner.rank0.npz"
)
ORACLE_8K = Path(
    "/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/8k/"
    "greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z/oracle"
)
REFERENCE_ROW = ROOT / "docs/artifacts/gate-d-event1-fp64-reference-row-20260905.npy"




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
    import shutil

    from glm_tpu.greenfield.validation.ws32_short_context import (
        load_ws32_adjudicated_divergence,
    )

    artifacts = tmp_path / "docs" / "artifacts"
    artifacts.mkdir(parents=True)
    row_relative = REGISTERED["path"]
    shutil.copyfile(ROOT / row_relative, tmp_path / row_relative)
    shutil.copyfile(ROOT / REGISTERED["validation_path"], tmp_path / REGISTERED["validation_path"])
    row_entry = {"path": row_relative, "sha256": REGISTERED["sha256"]}
    validation_entry = {
        "path": REGISTERED["validation_path"],
        "sha256": REGISTERED["validation_sha256"],
    }
    reference_row = {
        "convention": "rms_norm_eps_1e-5",
        "implementation_tree_sha1": REGISTERED["implementation_tree_sha1"],
        "path": REGISTERED["path"],
        "sha256": REGISTERED["sha256"],
    }
    analysis = {
        "artifact_kind": ANALYSIS_ARTIFACT_KIND,
        "checks": {
            name: {"pass": True}
            for name in (
                "bias",
                "cap_max_abs",
                "cap_std",
                "equal_sized_disjoint_swap",
                "reference_band",
                "reference_band_capacity",
            )
        },
        "context": "8k",
        "decode_position": 8155,
        "engine_source_run": (
            "greenfield_ws32_short_decoder_8k_numerical_20260905T000000000000000Z"
        ),
        "event_index": 1,
        "expected_only": [31],
        "observed_only": [32],
        "producer_layer_id": 1,
        "reference_row": reference_row,
        "step": 0,
        "verdict": "PASS",
    }
    analysis_relative = "docs/artifacts/gate-d-event1-analysis.json"
    (tmp_path / analysis_relative).write_text(json.dumps(analysis), encoding="utf-8")
    analysis_entry = {
        "path": analysis_relative,
        "sha256": sha256((tmp_path / analysis_relative).read_bytes()).hexdigest(),
    }
    record = {
        "analysis": analysis_entry,
        "artifact_kind": "gate_d_ws32_8k_adjudicated_divergence",
        "schema_version": 1,
        "spec_section": "21.2 items 3-4 (scope: first divergent event)",
        "date_utc": "2026-09-06",
        "context": "8k",
        "decode_position": 8155,
        "step": 0,
        "event_index": 1,
        "producer_layer_id": 1,
        "expected_only": [31],
        "observed_only": [32],
        "later_event_alarm": 1024,
        "basis": [row_entry, validation_entry, analysis_entry],
        "prior_attempts": [],
        "reference_row": reference_row,
        "oracle": {"dsa_manifest_sha256": "a" * 64, "token_manifest_sha256": "c" * 64},
        "engine_source_run": analysis["engine_source_run"],
        "semantics": "test",
        "gate_d_closed": False,
        "performance_claim": False,
    }
    path = tmp_path / "record.json"
    MODULE._write_once(path, record)
    loaded = load_ws32_adjudicated_divergence(
        path, expected_sha256=sha256(path.read_bytes()).hexdigest(), repository_root=tmp_path
    )
    assert loaded.step == 0 and loaded.event_index == 1
    assert loaded.expected_only == (31,) and loaded.observed_only == (32,)
    with pytest.raises(SystemExit, match="append-only"):
        MODULE._write_once(path, record)


def test_a_record_that_names_no_adjudication_analysis_is_refused(tmp_path: Path) -> None:
    """P1-1: the cross-check is vacuous if no analysis need be named."""
    import hashlib

    from glm_tpu.greenfield.validation.ws32_short_context import (
        load_ws32_adjudicated_divergence,
    )

    committed = ROOT / "docs/artifacts/gate-d-ws32-8k-adjudicated-divergence-20260905.json"
    fabricated = json.loads(committed.read_text(encoding="utf-8"))
    fabricated["date_utc"] = "2026-09-06"
    fabricated["expected_only"] = list(range(1, 11))
    fabricated["observed_only"] = list(range(11, 21))
    fabricated["reference_row"] = {
        "convention": "rms_norm_eps_1e-5",
        "implementation_tree_sha1": REGISTERED["implementation_tree_sha1"],
        "path": REGISTERED["path"],
        "sha256": REGISTERED["sha256"],
    }
    fabricated["prior_attempts"] = []
    path = tmp_path / "fabricated.json"
    payload = json.dumps(fabricated).encode()
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="does not declare analysis"):
        load_ws32_adjudicated_divergence(
            path, expected_sha256=hashlib.sha256(payload).hexdigest(), repository_root=ROOT
        )


def test_a_reference_row_outside_the_reviewed_registry_is_refused_by_the_loader(
    tmp_path: Path,
) -> None:
    """P1-2: the registry is enforced where the sealer reads, not where the tool writes."""
    import hashlib

    from glm_tpu.greenfield.validation.ws32_short_context import (
        load_ws32_adjudicated_divergence,
    )

    artifacts = tmp_path / "docs" / "artifacts"
    artifacts.mkdir(parents=True)
    fitted = artifacts / "gate-d-fitted-reference-row.npy"
    np.save(fitted, np.zeros(8, dtype=np.float64))
    row_entry = {
        "path": "docs/artifacts/gate-d-fitted-reference-row.npy",
        "sha256": sha256(fitted.read_bytes()).hexdigest(),
    }
    reference_row = dict(
        row_entry,
        convention="rms_norm_eps_1e-5",
        implementation_tree_sha1=REGISTERED["implementation_tree_sha1"],
    )
    analysis_relative = "docs/artifacts/gate-d-fitted-analysis.json"
    analysis = {
        "artifact_kind": ANALYSIS_ARTIFACT_KIND,
        "checks": {
            name: {"pass": True}
            for name in (
                "bias",
                "cap_max_abs",
                "cap_std",
                "equal_sized_disjoint_swap",
                "reference_band",
                "reference_band_capacity",
            )
        },
        "context": "8k",
        "decode_position": 8155,
        "engine_source_run": (
            "greenfield_ws32_short_decoder_8k_numerical_20260905T000000000000000Z"
        ),
        "event_index": 1,
        "expected_only": [31],
        "observed_only": [32],
        "producer_layer_id": 1,
        "reference_row": reference_row,
        "step": 0,
        "verdict": "PASS",
    }
    (tmp_path / analysis_relative).write_text(json.dumps(analysis), encoding="utf-8")
    analysis_entry = {
        "path": analysis_relative,
        "sha256": sha256((tmp_path / analysis_relative).read_bytes()).hexdigest(),
    }
    import shutil

    shutil.copyfile(ROOT / REGISTERED["validation_path"], tmp_path / REGISTERED["validation_path"])
    validation_entry = {
        "path": REGISTERED["validation_path"],
        "sha256": REGISTERED["validation_sha256"],
    }
    record = {
        "analysis": analysis_entry,
        "artifact_kind": "gate_d_ws32_8k_adjudicated_divergence",
        "schema_version": 1,
        "spec_section": "21.2 items 3-4 (scope: first divergent event)",
        "date_utc": "2026-09-06",
        "context": "8k",
        "decode_position": 8155,
        "step": 0,
        "event_index": 1,
        "producer_layer_id": 1,
        "expected_only": [31],
        "observed_only": [32],
        "later_event_alarm": 1024,
        "basis": [row_entry, validation_entry, analysis_entry],
        "prior_attempts": [],
        "reference_row": reference_row,
        "oracle": {"dsa_manifest_sha256": "a" * 64, "token_manifest_sha256": "c" * 64},
        "engine_source_run": analysis["engine_source_run"],
        "semantics": "test",
        "gate_d_closed": False,
        "performance_claim": False,
    }
    path = tmp_path / "record.json"
    payload = json.dumps(record).encode()
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="not the reviewed row for this event"):
        load_ws32_adjudicated_divergence(
            path, expected_sha256=hashlib.sha256(payload).hexdigest(), repository_root=tmp_path
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


def test_a_shorter_oracle_window_scans_the_common_prefix(tmp_path: Path) -> None:
    """A shorter window is adjudicable over what both archives hold."""
    archive, oracle_dir = _multi_event_fixture(tmp_path, divergent_at=(2, 3))
    from safetensors.numpy import load_file, save_file

    data = load_file(str(oracle_dir / "dsa_events.safetensors"))
    save_file(
        {name: value[:, :3] for name, value in data.items()},
        str(oracle_dir / "dsa_events.safetensors"),
    )
    assert MODULE._scan_window(archive, oracle_dir) == (3, 3)
    # Event (2, 3) is outside the common prefix, so nothing divergent is in scope.
    with pytest.raises(SystemExit, match="no divergent DSA event"):
        MODULE._first_divergent_event(archive, oracle_dir)


def test_naming_a_later_benign_event_is_refused(tmp_path: Path) -> None:
    """P2-4: the record claims every earlier event is exact, so it is checked."""
    archive, oracle_dir = _multi_event_fixture(tmp_path, divergent_at=(0, 1))
    assert MODULE._resolve_event(archive, oracle_dir, None, None) == (0, 1)
    assert MODULE._resolve_event(archive, oracle_dir, 0, 1) == (0, 1)
    with pytest.raises(SystemExit, match="is not the first divergent event"):
        MODULE._resolve_event(archive, oracle_dir, 2, 3)
    with pytest.raises(SystemExit, match="must be given together"):
        MODULE._resolve_event(archive, oracle_dir, 0, None)


def test_the_reference_row_must_be_a_reviewed_pre_registered_artifact(tmp_path: Path) -> None:
    """P1-1: an unbound row lets a fitted reference produce a clean PASS."""
    committed = ROOT / "docs/artifacts/gate-d-event1-fp64-reference-row-20260905.npy"
    digest = sha256(committed.read_bytes()).hexdigest()
    event = dict(context="8k", decode_position=8155, producer_layer_id=1,
                 convention="rms_norm_eps_1e-5")
    resolved, relative, seen, entry = MODULE._committed_reference_row(
        committed, digest, **event
    )
    assert resolved == committed.resolve()
    assert relative == "docs/artifacts/gate-d-event1-fp64-reference-row-20260905.npy"
    assert seen == digest == entry["sha256"]
    assert entry["validation_path"].endswith("math-reference-adjudication-20260905.json")

    with pytest.raises(SystemExit, match="does not match the pre-registered"):
        MODULE._committed_reference_row(committed, "0" * 64, **event)

    outside = tmp_path / "reference.npy"
    np.save(outside, np.zeros(4, dtype=np.float64))
    with pytest.raises(SystemExit, match="must be a committed"):
        MODULE._committed_reference_row(
            outside, sha256(outside.read_bytes()).hexdigest(), **event
        )

    # No reviewed row exists for another event or the other norm-eps convention.
    with pytest.raises(SystemExit, match="no reviewed FP64 reference row"):
        MODULE._committed_reference_row(
            committed, digest, **dict(event, decode_position=8154)
        )
    with pytest.raises(SystemExit, match="no reviewed FP64 reference row"):
        MODULE._committed_reference_row(
            committed, digest, **dict(event, convention="rms_norm_eps_1e-6")
        )


def test_a_tracked_artifact_overwritten_in_place_is_refused(monkeypatch, tmp_path: Path) -> None:
    """The demonstrated attack: a tracked PATH is not committed CONTENT.

    Run against a throwaway repository so a killed test can never leave the
    reviewed artifact tree modified.
    """
    import subprocess as sp

    repository = tmp_path / "repo"
    (repository / "docs" / "artifacts").mkdir(parents=True)
    relative = "docs/artifacts/gate-d-row.npy"
    target = repository / relative
    np.save(target, np.arange(8, dtype=np.float64))
    for command in (
        ["init", "-q"],
        ["-c", "user.email=t@t", "-c", "user.name=t", "add", "--", relative],
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "row"],
    ):
        sp.run(["git", "-C", str(repository), *command], check=True, capture_output=True)
    monkeypatch.setattr(MODULE, "REPO_ROOT", repository)

    MODULE._require_committed_content(relative)

    np.save(target, np.zeros(8, dtype=np.float64))
    with pytest.raises(SystemExit, match="differs from the committed blob"):
        MODULE._require_committed_content(relative)

    with pytest.raises(SystemExit, match="not committed at HEAD"):
        MODULE._require_committed_content("docs/artifacts/gate-d-absent.npy")


def test_the_reference_implementation_hash_is_recorded() -> None:
    tree = MODULE._reference_implementation_tree()
    assert len(tree) == 40 and all(c in "0123456789abcdef" for c in tree)


def test_the_observer_archive_must_match_what_the_run_published(monkeypatch) -> None:
    """P2-a: a directory name and a summary.json are both operator-writable.

    The binding is the archive the protected run uploaded under its own tag, so
    copying the run being sealed into a differently named directory no longer
    walks past the sealer's anti-circularity guard.
    """
    import tempfile

    tag = "greenfield_ws32_short_decoder_8k_numerical_20260905T000000000000000Z"
    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory) / "runner.rank0.npz"
        archive.write_bytes(b"observer arrays")
        local = MODULE._local_hashes(archive)
        assert set(local) == {"crc32c", "md5"}

        monkeypatch.setattr(MODULE, "_remote_archive_hashes", lambda run_tag: dict(local))
        witness = MODULE._require_archive_belongs_to_run(archive, tag)
        assert local["md5"] in witness and local["crc32c"] in witness

        # A composite upload carries only CRC32C; that is still a binding.
        monkeypatch.setattr(
            MODULE, "_remote_archive_hashes", lambda run_tag: {"crc32c": local["crc32c"]}
        )
        assert MODULE._require_archive_belongs_to_run(archive, tag) == (
            f"crc32c={local['crc32c']}"
        )

        monkeypatch.setattr(
            MODULE, "_remote_archive_hashes", lambda run_tag: {"md5": "0" * 32}
        )
        with pytest.raises(SystemExit, match="does not match the archive run"):
            MODULE._require_archive_belongs_to_run(archive, tag)

        # gcloud missing, unauthenticated, timed out or the object absent.
        monkeypatch.setattr(MODULE, "_remote_archive_hashes", lambda run_tag: {})
        with pytest.raises(SystemExit, match="published no readable observer archive"):
            MODULE._require_archive_belongs_to_run(archive, tag)

        monkeypatch.setattr(
            MODULE, "_remote_archive_hashes", lambda run_tag: {"sha512": "f" * 128}
        )
        with pytest.raises(SystemExit, match="no hash this tool can compare"):
            MODULE._require_archive_belongs_to_run(archive, tag)


def test_adjudication_outputs_must_live_in_the_reviewed_artifact_directory(tmp_path: Path) -> None:
    """P1-3-new: the disclosure scan globs one directory, so outputs go there."""
    good = ROOT / "docs/artifacts/gate-d-probe-analysis.json"
    assert MODULE._committed_output_path(good, ".json") == good.resolve()
    for bad in (
        tmp_path / "analysis.json",
        ROOT / "docs/artifacts/analysis.json",
        ROOT / "docs/other/gate-d-analysis.json",
    ):
        with pytest.raises(SystemExit, match="must be docs/artifacts/gate-d-"):
            MODULE._committed_output_path(bad, ".json")


def test_earlier_attempts_on_the_same_event_are_collected_for_disclosure(
    monkeypatch, tmp_path: Path
) -> None:
    """P2-5 and P1-2-new: attempts are disclosed, never relied on."""
    monkeypatch.setattr(MODULE, "COMMITTED_ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(MODULE, "REPO_ROOT", tmp_path.parent)
    analysis = {
        "artifact_kind": ANALYSIS_ARTIFACT_KIND,
        "checks": {},
        "engine_source_run": "run_a",
        "event_index": 1,
        "step": 0,
        "verdict": "FAIL",
    }
    earlier = tmp_path / "gate-d-attempt-one.json"
    earlier.write_text(json.dumps(analysis), encoding="utf-8")
    output = tmp_path / "gate-d-attempt-two.json"

    disclosed = MODULE._collect_prior_attempts(output, analysis)
    assert [item["path"] for item in disclosed] == [
        f"{tmp_path.name}/gate-d-attempt-one.json"
    ]
    assert disclosed[0]["verdict"] == "FAIL"
    assert disclosed[0]["sha256"] == sha256(earlier.read_bytes()).hexdigest()

    edited = tmp_path / "gate-d-attempt-three.json"
    edited.write_text(
        json.dumps(dict(analysis, artifact_kind="something_else")), encoding="utf-8"
    )
    assert len(MODULE._collect_prior_attempts(output, analysis)) == 2

    assert MODULE._collect_prior_attempts(output, dict(analysis, event_index=2)) == []


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
    assert at_one["kappa"] == 1.0 and at_two["kappa"] == 2.0
    assert "bias_factor" not in at_two


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


def test_the_published_hash_parser_rejects_a_null_md5() -> None:
    """P2-1: a composite upload prints `md5_hash: null`, it does not omit it."""
    captured = (
        "---\n"
        "crc32c_hash: 7fb43721\n"
        "digest_format: hex\n"
        "md5_hash: null\n"
        "url: gs://driftbench-dsv4-uc/results/tag/host_records/runner.rank0.npz\n"
    )

    class _Completed:
        returncode = 0
        stdout = captured

    import subprocess as sp

    original = sp.run
    try:
        sp.run = lambda *a, **k: _Completed()
        MODULE.subprocess.run = sp.run
        hashes = MODULE._remote_archive_hashes("tag")
    finally:
        sp.run = original
        MODULE.subprocess.run = original
    assert hashes == {"crc32c": "7fb43721"}, "null is not a digest"


def test_the_registry_path_and_digest_are_both_enforced_by_the_producer(
    monkeypatch, tmp_path: Path
) -> None:
    """P2-5: neither registry check had a test that could fail."""
    import subprocess as sp

    repository = tmp_path / "repo"
    (repository / "docs" / "artifacts").mkdir(parents=True)
    rival = "docs/artifacts/gate-d-event1-fp64-reference-row-v2-20260906.npy"
    registered = "docs/artifacts/gate-d-event1-fp64-reference-row-20260905.npy"
    import shutil

    shutil.copyfile(ROOT / registered, repository / registered)
    shutil.copyfile(ROOT / registered, repository / rival)
    validation = REGISTERED["validation_path"]
    shutil.copyfile(ROOT / validation, repository / validation)
    for command in (
        ["init", "-q"],
        ["-c", "user.email=t@t", "-c", "user.name=t", "add", "-A"],
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "rows"],
    ):
        sp.run(["git", "-C", str(repository), *command], check=True, capture_output=True)
    monkeypatch.setattr(MODULE, "REPO_ROOT", repository)
    monkeypatch.setattr(MODULE, "COMMITTED_ARTIFACT_DIR", repository / "docs" / "artifacts")
    monkeypatch.setattr(MODULE, "_reference_implementation_tree",
                        lambda: REGISTERED["implementation_tree_sha1"])
    event = dict(context="8k", decode_position=8155, producer_layer_id=1,
                 convention="rms_norm_eps_1e-5")

    # A second, byte-identical copy under a plausible name is still not the
    # reviewed row: the registry names one path.
    with pytest.raises(SystemExit, match="the reviewed reference row for this event is"):
        MODULE._committed_reference_row(repository / rival, REGISTERED["sha256"], **event)

    # And the registered path must still carry the registered content.
    monkeypatch.setitem(
        REFERENCE_ROWS, ("8k", 8155, 1, "rms_norm_eps_1e-5"),
        dict(REGISTERED, sha256="0" * 64),
    )
    with pytest.raises(SystemExit, match="is not the reviewed row"):
        MODULE._committed_reference_row(repository / registered, "0" * 64, **event)


def test_the_reference_implementation_must_match_the_registered_one(
    monkeypatch, tmp_path: Path
) -> None:
    committed = ROOT / REGISTERED["path"]
    monkeypatch.setattr(MODULE, "_reference_implementation_tree", lambda: "0" * 40)
    with pytest.raises(SystemExit, match="reference implementation at HEAD"):
        MODULE._committed_reference_row(
            committed, REGISTERED["sha256"], context="8k", decode_position=8155,
            producer_layer_id=1, convention="rms_norm_eps_1e-5",
        )


def test_the_scan_window_uses_both_archives_step_counts(tmp_path: Path) -> None:
    """P2-5: dropping the oracle's step count survived mutation."""
    archive, oracle_dir = _multi_event_fixture(tmp_path, divergent_at=(2, 3))
    from safetensors.numpy import load_file, save_file

    data = load_file(str(oracle_dir / "dsa_events.safetensors"))
    save_file(
        {name: value[:1] for name, value in data.items()},
        str(oracle_dir / "dsa_events.safetensors"),
    )
    assert MODULE._scan_window(archive, oracle_dir) == (1, 4)


def _main_ast() -> "object":
    import ast

    return ast.parse(SCRIPT.read_text(encoding="utf-8"))


def _function(name: str):
    import ast

    for node in ast.walk(_main_ast()):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is absent")


def _calls(node, name: str) -> list:
    import ast

    return [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and (
            (isinstance(item.func, ast.Name) and item.func.id == name)
            or (isinstance(item.func, ast.Attribute) and item.func.attr == name)
        )
    ]


def test_main_still_performs_every_guard_it_is_credited_with() -> None:
    """P2-2: these guards had no test; deleting their CALLS left the suite green."""
    import ast

    main = _function("main")
    for name in (
        "_committed_reference_row",
        "_require_archive_belongs_to_run",
        "_resolve_event",
        "_scan_window",
        "_collect_prior_attempts",
        "_committed_output_path",
        "_write_once",
    ):
        assert _calls(main, name), f"main() no longer calls {name}"

    source = SCRIPT.read_text(encoding="utf-8")
    # The verdict gate must precede the record write, and must return non-zero.
    verdict_gate = source.index('if analysis["verdict"] != "PASS":')
    record_write = source.index("_write_once(args.record_output, record)")
    assert verdict_gate < record_write, "a refused adjudication must not write a record"
    assert "return 2" in source[verdict_gate:record_write]

    # The convention and alarm guards replaced argparse-enforced ones.
    assert "--reference-convention must be one of" in source
    assert "fixes the later-event alarm at" in source

    # The row and its validation record are named in the basis whether or not
    # the operator passed them.
    assert 'reference_entry["validation_path"]' in source
    constants = {
        node.value
        for node in ast.walk(main)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "prior_attempts" in constants and "analysis" in constants


def test_the_analysis_check_names_are_the_computations_own() -> None:
    """P3-6: the loader's required set must not drift from what is emitted."""
    from glm_tpu.greenfield.validation.ws32_first_divergent_event import (
        ADJUDICATION_CHECKS,
    )

    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        archive, oracle_dir, reference_row = _fixture(
            Path(directory), engine_bias=0.0, swap=True
        )
        analysis = MODULE.adjudicate(
            archive=archive, oracle_dir=oracle_dir, reference_row=reference_row,
            step=0, event=0,
        )
    assert set(analysis["checks"]) == set(ADJUDICATION_CHECKS)


def test_the_help_text_names_the_contract_values_it_validates_against() -> None:
    """P3-8: the lazy import must not hide the legal values from --help."""
    from glm_tpu.greenfield.validation.ws32_short_context import (
        LATER_EVENT_ALARM,
        REFERENCE_CONVENTIONS,
    )

    source = SCRIPT.read_text(encoding="utf-8")
    for convention in REFERENCE_CONVENTIONS:
        assert convention in source, f"--help no longer names {convention}"
    assert f"(default: {LATER_EVENT_ALARM})" in source


def _bare_event(*, reference, oracle_positions, oracle_scores, engine_positions, engine_scores):
    from glm_tpu.greenfield.validation.ws32_first_divergent_event import (
        adjudicate_first_divergent_event,
    )

    return adjudicate_first_divergent_event(
        oracle_positions=np.asarray(oracle_positions),
        oracle_scores=np.asarray(oracle_scores),
        engine_positions=np.asarray(engine_positions),
        engine_scores=np.asarray(engine_scores),
        reference=np.asarray(reference),
        producer_layer_id=1,
        step=0,
        event=0,
    )


def test_cap_std_refuses_an_engine_whose_spread_exceeds_the_oracles() -> None:
    """§21.2 item 3's std cap had no fixture that could fail it."""
    width = 8
    reference = np.linspace(1.0, 0.0, width + 1)
    oracle_positions = np.arange(width)
    engine_positions = oracle_positions.copy()
    engine_positions[-1] = width
    # The oracle sits a constant epsilon above the reference, so its spread is
    # zero; the engine alternates within the same max-abs bound.
    epsilon = 0.01
    oracle_scores = reference[oracle_positions] + epsilon
    engine_scores = reference[engine_positions] + epsilon * np.where(
        np.arange(width) % 2 == 0, 1.0, -1.0
    )
    result = _bare_event(
        reference=reference,
        oracle_positions=oracle_positions,
        oracle_scores=oracle_scores,
        engine_positions=engine_positions,
        engine_scores=engine_scores,
    )
    assert result["checks"]["cap_max_abs"]["pass"] is True
    assert result["checks"]["cap_std"]["pass"] is False
    assert result["verdict"] == "FAIL"


def test_reference_band_refuses_a_swap_far_from_the_cutoff() -> None:
    """The clause carrying the whole boundary-explained claim had no fixture."""
    width = 8
    reference = np.concatenate([np.linspace(1.0, 0.5, width), np.asarray([-5.0])])
    oracle_positions = np.arange(width)
    engine_positions = oracle_positions.copy()
    engine_positions[-1] = width  # the far-below-cutoff position
    epsilon = 0.2
    oracle_scores = reference[oracle_positions] + epsilon
    engine_scores = reference[engine_positions] + epsilon
    result = _bare_event(
        reference=reference,
        oracle_positions=oracle_positions,
        oracle_scores=oracle_scores,
        engine_positions=engine_positions,
        engine_scores=engine_scores,
    )
    assert result["checks"]["reference_band"]["pass"] is False
    assert result["verdict"] == "FAIL"


def test_event_arrays_take_only_the_valid_prefix() -> None:
    """A short event's padding tail would make every cap pass vacuously."""
    from glm_tpu.greenfield.validation.ws32_first_divergent_event import event_arrays

    width = 6
    positions = np.full((1, 1, 1, width), -1, dtype=np.int32)
    positions[0, 0, 0, :3] = [4, 5, 6]
    scores = np.full((1, 1, 1, width), -np.inf, dtype=np.float32)
    scores[0, 0, 0, :3] = [0.3, 0.2, 0.1]
    counts = np.full((1, 1, 1), 3, dtype=np.int32)
    got_positions, got_scores = event_arrays(
        selected_positions=positions,
        selected_scores=scores,
        selected_valid_counts=counts,
        step=0,
        event=0,
    )
    assert got_positions.tolist() == [4, 5, 6]
    assert got_scores.tolist() == [
        pytest.approx(0.3, abs=1e-7),
        pytest.approx(0.2, abs=1e-7),
        pytest.approx(0.1, abs=1e-7),
    ]
    assert np.isfinite(got_scores).all()
