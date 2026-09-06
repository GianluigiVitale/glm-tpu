from __future__ import annotations

from pathlib import Path

import numpy as np

from glm_tpu.greenfield.validation.ws32_short_context import (
    compare_ws32_dsa_step,
    compare_ws32_raw_tokens,
    load_ws32_short_context_oracle,
    validate_ws32_cache_probe,
)


TOKEN_DIR = Path(
    "/home/gianl/gcs-models/oracles/greenfield/glm52/short_context/2k/"
    "greenfield_short_context_oracle_20260806T202544155912103Z/oracle"
)
DSA_DIR = Path(
    "/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/2k/"
    "greenfield_short_context_dsa_oracle_recovery_20260806T231905802593249Z/oracle"
)
TOKEN_SHA = "f580c14954bcbd0d973b6fe8158520992a18a1375ed88cff9cceb8e01c7efe19"
DSA_SHA = "71224832652ce61024786d39d43dcbfdc6cde76bf2eff0350531b272f38f4f57"
TOKEN_SUCCESS_SHA = "07700db5a732f04663f0298625bbdbb68a1c73e63652a3aef27f398993e86eec"
DSA_SUCCESS_SHA = "c091d0b56f712eb2f106ee248f69f599586ade0d5c202cbb5b46b8aa115411b2"


def test_ws32_short_context_replays_sealed_oracle_and_refuses_drift() -> None:
    oracle = load_ws32_short_context_oracle(
        TOKEN_DIR,
        DSA_DIR,
        expected_token_manifest_sha256=TOKEN_SHA,
        expected_dsa_manifest_sha256=DSA_SHA,
        expected_token_success_sha256=TOKEN_SUCCESS_SHA,
        expected_dsa_success_sha256=DSA_SUCCESS_SHA,
    )
    assert oracle.prompt_token_ids.shape == (2034,)
    assert oracle.selected_positions.shape == (14, 21, 2048)
    exact = compare_ws32_dsa_step(
        producer_layer_ids=oracle.producer_layer_ids,
        selected_positions=oracle.selected_positions[0, :, None, :],
        selected_valid_counts=oracle.valid_counts[0, :, None],
        selected_scores=oracle.selected_scores[0, :, None, :],
        oracle=oracle,
        step=0,
    )
    assert exact["passed"], exact
    tokens = compare_ws32_raw_tokens(
        oracle.generated_token_ids[:15].tolist(), oracle
    )
    assert tokens["exact_prefix_match"]

    mutated_positions = oracle.selected_positions[0, :, None, :].copy()
    mutated_positions[0, 0, 0] = mutated_positions[0, 0, 1]
    refused = compare_ws32_dsa_step(
        producer_layer_ids=oracle.producer_layer_ids,
        selected_positions=mutated_positions,
        selected_valid_counts=oracle.valid_counts[0, :, None],
        selected_scores=oracle.selected_scores[0, :, None, :],
        oracle=oracle,
        step=0,
    )
    assert not refused["passed"]

    wrong = oracle.generated_token_ids[:3].copy()
    wrong[1] += 1
    assert not compare_ws32_raw_tokens(wrong, oracle)["exact_prefix_match"]

    with np.testing.assert_raises_regex(ValueError, "SUCCESS identity"):
        load_ws32_short_context_oracle(
            TOKEN_DIR,
            DSA_DIR,
            expected_token_manifest_sha256=TOKEN_SHA,
            expected_dsa_manifest_sha256=DSA_SHA,
            expected_token_success_sha256="0" * 64,
            expected_dsa_success_sha256=DSA_SUCCESS_SHA,
        )


def test_ws32_cache_probe_requires_every_layer_and_index_slot() -> None:
    kv = np.ones((3, 8), dtype=np.float32)
    index = np.ones((2, 4), dtype=np.float32)
    exact = validate_ws32_cache_probe(
        position=np.asarray([7], dtype=np.int32),
        kv_rows=kv,
        index_rows=index,
        contract_valid=np.asarray([True]),
        expected_position=7,
        num_layers=3,
        full_indexer_count=2,
        packed_cache_width=8,
        index_width=4,
    )
    assert exact["passed"]
    kv[1] = 0
    refused = validate_ws32_cache_probe(
        position=np.asarray([7], dtype=np.int32),
        kv_rows=kv,
        index_rows=index,
        contract_valid=np.asarray([True]),
        expected_position=7,
        num_layers=3,
        full_indexer_count=2,
        packed_cache_width=8,
        index_width=4,
    )
    assert not refused["passed"]


TOKEN_DIR_8K = Path(
    "/home/gianl/gcs-models/oracles/greenfield/glm52/short_context/8k/"
    "greenfield_short_context_oracle_8k_20260807T172307269147351Z/oracle"
)
DSA_DIR_8K = Path(
    "/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/8k/"
    "greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z/oracle"
)
TOKEN_SHA_8K = "e4fbcbdbf0fc8b1969e2f82ee457ab1563db4a8b37d2dea2bc4d1e828a13acf2"
DSA_SHA_8K = "f8154c5f79b909efd9ebc14c8e004925482844d05ef28fcf0a4d29bb4a7b26da"
TOKEN_SUCCESS_SHA_8K = "38c0aeb6c4833a0256d4e50152b645e85d24a4f00ca7b2b1731db2d892c5b3cc"
DSA_SUCCESS_SHA_8K = "0b798974ae8a9f95c32d3aa2eff532213624f1e2ae7f1de809a161e18dbdf1b9"


def _oracle():
    # 8K: 2,048 of 8,156 causal positions are selected, so set divergences exist.
    return load_ws32_short_context_oracle(
        TOKEN_DIR_8K,
        DSA_DIR_8K,
        expected_token_manifest_sha256=TOKEN_SHA_8K,
        expected_dsa_manifest_sha256=DSA_SHA_8K,
        expected_token_success_sha256=TOKEN_SUCCESS_SHA_8K,
        expected_dsa_success_sha256=DSA_SUCCESS_SHA_8K,
    )


def _swap_one(oracle, positions, event, count=1):
    """Replace `count` selected positions of `event` with unselected valid ones.

    Returns (expected_only, observed_only) sorted lists of the induced divergence.
    """
    decode_position = int(oracle.decode_positions[0])
    valid = int(oracle.valid_counts[0, event])
    live = set(positions[event, 0, :valid].tolist())
    replacements = [p for p in range(decode_position + 1) if p not in live][:count]
    removed = sorted(positions[event, 0, :valid].tolist())[-count:]
    for slot, (old, new) in enumerate(zip(removed, replacements)):
        index = int(np.flatnonzero(positions[event, 0, :valid] == old)[0])
        positions[event, 0, index] = new
    return sorted(removed), sorted(replacements)


def _adjudication(expected_only, observed_only, *, step=0, event=1, alarm=1024):
    from glm_tpu.greenfield.validation.ws32_short_context import (
        Ws32AdjudicatedDivergence,
    )

    return Ws32AdjudicatedDivergence(
        record_sha256="f" * 64,
        step=step,
        event_index=event,
        expected_only=tuple(expected_only),
        observed_only=tuple(observed_only),
        context="8k",
        oracle_dsa_manifest_sha256=DSA_SHA_8K,
        decode_position=8155,
        producer_layer_id=1,
        later_event_alarm=alarm,
    )


def test_ws32_first_divergent_event_adjudication_is_exact_and_fail_closed() -> None:
    oracle = _oracle()
    base = oracle.selected_positions[0, :, None, :].copy()
    scores = oracle.selected_scores[0, :, None, :]
    counts = oracle.valid_counts[0, :, None]
    kwargs = dict(
        producer_layer_ids=oracle.producer_layer_ids,
        selected_valid_counts=counts,
        selected_scores=scores,
        oracle=oracle,
        step=0,
    )
    # Exact mode unchanged: no adjudication -> exact data passes with mode "exact".
    exact = compare_ws32_dsa_step(selected_positions=base.copy(), **kwargs)
    assert exact["passed"] and exact["adjudication"]["mode"] == "exact"

    diverged = base.copy()
    expected_only, observed_only = _swap_one(oracle, diverged, event=1)
    record = _adjudication(expected_only, observed_only)

    # The pre-registered divergence at (0, 1) passes and is reported as matching.
    result = compare_ws32_dsa_step(selected_positions=diverged.copy(), adjudication=record, **kwargs)
    assert result["passed"], result["adjudication"]
    assert result["adjudication"]["mode"] == "first_divergent_event"
    assert result["adjudication"]["adjudicated_event_matches_record"] is True
    assert result["adjudication"]["event_status"][0] == "exact_required"
    assert result["adjudication"]["event_status"][1] == "adjudicated"
    assert result["adjudication"]["event_status"][2] == "recorded"
    assert result["selected_set_mismatches"][0]["event_index"] == 1
    assert result["exact_selected_set_and_tail"] is False

    # Exact data under an adjudication record fails: the record predicts a divergence.
    predicted_but_exact = compare_ws32_dsa_step(selected_positions=base.copy(), adjudication=record, **kwargs)
    assert predicted_but_exact["passed"] is False
    assert predicted_but_exact["adjudication"]["adjudicated_event_matches_record"] is False

    # A different divergence at the adjudicated event fails.
    other = base.copy()
    _swap_one(oracle, other, event=1, count=2)
    assert compare_ws32_dsa_step(selected_positions=other, adjudication=record, **kwargs)["passed"] is False

    # Any divergence at an earlier (exact-required) event fails.
    early = diverged.copy()
    _swap_one(oracle, early, event=0)
    early_result = compare_ws32_dsa_step(selected_positions=early, adjudication=record, **kwargs)
    assert early_result["passed"] is False
    assert early_result["adjudication"]["unexplained_set_mismatch_events"] == [0]

    # Later-event divergence is recorded, not refused; the alarm flags large ones.
    later = diverged.copy()
    _swap_one(oracle, later, event=5, count=3)
    later_result = compare_ws32_dsa_step(selected_positions=later.copy(), adjudication=record, **kwargs)
    assert later_result["passed"] is True
    assert later_result["adjudication"]["recorded_divergence_sizes"][:1] == [
        {"event_index": 2, "symmetric_difference": 0}
    ]
    assert {"event_index": 5, "symmetric_difference": 6} in later_result["adjudication"]["recorded_divergence_sizes"]
    assert later_result["adjudication"]["alarm_events"] == []
    alarmed = compare_ws32_dsa_step(
        selected_positions=later.copy(),
        adjudication=_adjudication(expected_only, observed_only, alarm=5),
        **kwargs,
    )
    assert alarmed["passed"] is True and alarmed["adjudication"]["alarm_events"] == [5]

    # Structural failures stay hard in adjudication mode: a duplicate position.
    duplicate = diverged.copy()
    duplicate[7, 0, 0] = duplicate[7, 0, 1]
    assert compare_ws32_dsa_step(selected_positions=duplicate, adjudication=record, **kwargs)["passed"] is False

    # Steps other than the adjudicated step require exact sets.
    step1 = oracle.selected_positions[1, :, None, :].copy()
    _swap_one(oracle, step1, event=3)
    step1_result = compare_ws32_dsa_step(
        producer_layer_ids=oracle.producer_layer_ids,
        selected_positions=step1,
        selected_valid_counts=oracle.valid_counts[1, :, None],
        selected_scores=oracle.selected_scores[1, :, None, :],
        oracle=oracle,
        step=1,
        adjudication=record,
    )
    assert step1_result["passed"] is True  # later step: recorded only
    assert step1_result["adjudication"]["event_status"] == ["recorded"] * 21


def test_ws32_adjudicated_divergence_record_loader_binds_sha_and_schema(tmp_path: Path) -> None:
    import hashlib
    import json

    from glm_tpu.greenfield.validation.ws32_short_context import (
        load_ws32_adjudicated_divergence,
    )

    repo = Path(__file__).resolve().parents[3]
    committed = repo / "docs/artifacts/gate-d-ws32-8k-adjudicated-divergence-20260905.json"
    raw = committed.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    loaded = load_ws32_adjudicated_divergence(committed, expected_sha256=digest, repository_root=repo)
    assert loaded.step == 0 and loaded.event_index == 1 and loaded.context == "8k"
    assert loaded.expected_only == (680, 1052, 2024, 2436, 6322, 7473, 7850)
    assert loaded.observed_only == (754, 1904, 2029, 3651, 4899, 5536, 6951)
    assert loaded.later_event_alarm == 1024
    assert loaded.oracle_dsa_manifest_sha256 == "f8154c5f79b909efd9ebc14c8e004925482844d05ef28fcf0a4d29bb4a7b26da"
    assert loaded.status(0, 0) == "exact_required"
    assert loaded.status(0, 1) == "adjudicated"
    assert loaded.status(0, 2) == "recorded" and loaded.status(3, 0) == "recorded"

    with np.testing.assert_raises_regex(ValueError, "identity drifted"):
        load_ws32_adjudicated_divergence(committed, expected_sha256="0" * 64)

    from glm_tpu.greenfield.validation.ws32_short_context import REFERENCE_ROWS

    registered = REFERENCE_ROWS[("8k", 8155, 1, "rms_norm_eps_1e-5")]

    def _bound(payload: dict) -> dict:
        """Any record other than the grandfathered one must bind row and analysis."""
        row = payload["basis"][2]
        return dict(
            payload,
            analysis=dict(payload["basis"][1]),
            prior_attempts=[],
            reference_row={
                "convention": "rms_norm_eps_1e-5",
                "implementation_tree_sha1": registered["implementation_tree_sha1"],
                "path": row["path"],
                "sha256": row["sha256"],
            },
        )

    unbound = json.dumps(json.loads(raw)).encode()
    path = tmp_path / "unbound.json"
    path.write_bytes(unbound)
    with np.testing.assert_raises_regex(ValueError, "does not declare analysis"):
        load_ws32_adjudicated_divergence(
            path, expected_sha256=hashlib.sha256(unbound).hexdigest()
        )
    detached = _bound(json.loads(raw))
    detached["reference_row"]["sha256"] = "0" * 64
    payload = json.dumps(detached).encode()
    path = tmp_path / "detached.json"
    path.write_bytes(payload)
    with np.testing.assert_raises_regex(ValueError, "reference row is not in the basis"):
        load_ws32_adjudicated_divergence(
            path, expected_sha256=hashlib.sha256(payload).hexdigest()
        )

    record = json.loads(raw)
    for mutate in (
        lambda r: r.__setitem__("gate_d_closed", True),
        lambda r: r.__setitem__("expected_only", [680, 680]),
        lambda r: r.__setitem__("observed_only", [754]),
        lambda r: r.__setitem__("later_event_alarm", 0),
        lambda r: r.pop("semantics"),
        lambda r: r.__setitem__("context", "4k"),
    ):
        hostile = _bound(json.loads(raw))
        mutate(hostile)
        path = tmp_path / "hostile.json"
        payload = json.dumps(hostile).encode()
        path.write_bytes(payload)
        with np.testing.assert_raises_regex(ValueError, "adjudicated-divergence"):
            load_ws32_adjudicated_divergence(path, expected_sha256=hashlib.sha256(payload).hexdigest())
    tampered_basis = _bound(json.loads(raw))
    tampered_basis["basis"][0]["sha256"] = "0" * 64
    path = tmp_path / "basis.json"
    payload = json.dumps(tampered_basis).encode()
    path.write_bytes(payload)
    with np.testing.assert_raises_regex(ValueError, "basis drifted"):
        load_ws32_adjudicated_divergence(path, expected_sha256=hashlib.sha256(payload).hexdigest(), repository_root=repo)


def test_ws32_adjudication_binding_refuses_foreign_oracle_and_geometry() -> None:
    from glm_tpu.greenfield.validation.ws32_short_context import (
        bind_ws32_adjudication,
    )

    oracle = _oracle()
    record = _adjudication([680], [754])
    bind_ws32_adjudication(record, oracle, observer_steps=14, context_label="8k")
    from dataclasses import replace

    for hostile, message in (
        (replace(record, oracle_dsa_manifest_sha256="0" * 64), "different DSA oracle"),
        (replace(record, step=14), "outside the observed steps"),
        (replace(record, event_index=21), "outside the oracle events"),
        (replace(record, decode_position=8154), "decode position differs"),
        (replace(record, producer_layer_id=2), "producer layer differs"),
    ):
        with np.testing.assert_raises_regex(ValueError, message):
            bind_ws32_adjudication(hostile, oracle, observer_steps=14, context_label="8k")
    with np.testing.assert_raises_regex(ValueError, "context differs"):
        bind_ws32_adjudication(record, oracle, observer_steps=14, context_label="2k")


def _adjudication_record_fixture(root: Path) -> tuple[dict, dict]:
    """A minimal repository holding a record plus its PASS analysis and row."""
    import hashlib
    import json
    import shutil

    artifacts = root / "docs" / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    row = artifacts / "gate-d-row.npy"
    row.write_bytes(b"reference row bytes")
    row_entry = {
        "path": "docs/artifacts/gate-d-row.npy",
        "sha256": hashlib.sha256(row.read_bytes()).hexdigest(),
    }
    from glm_tpu.greenfield.validation.ws32_short_context import REFERENCE_ROWS

    registered = REFERENCE_ROWS[("8k", 8155, 1, "rms_norm_eps_1e-5")]
    row.unlink()
    (root / registered["path"]).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        Path(__file__).resolve().parents[3] / registered["path"], root / registered["path"]
    )
    row_entry = {"path": registered["path"], "sha256": registered["sha256"]}
    shutil.copyfile(
        Path(__file__).resolve().parents[3] / registered["validation_path"],
        root / registered["validation_path"],
    )
    validation_entry = {
        "path": registered["validation_path"],
        "sha256": registered["validation_sha256"],
    }
    reference_row = dict(
        row_entry,
        convention="rms_norm_eps_1e-5",
        implementation_tree_sha1=registered["implementation_tree_sha1"],
    )
    source_run = "greenfield_ws32_short_decoder_8k_numerical_20260906T000000000000000Z"
    analysis = {
        "artifact_kind": "gate_d_ws32_first_divergent_event_adjudication",
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
        "engine_source_run": source_run,
        "event_index": 1,
        "producer_layer_id": 1,
        "expected_only": [31],
        "observed_only": [32],
        "reference_row": reference_row,
        "step": 0,
        "verdict": "PASS",
    }
    passed = artifacts / "gate-d-analysis.json"
    passed.write_text(json.dumps(analysis), encoding="utf-8")
    refused = artifacts / "gate-d-attempt.json"
    refused.write_text(json.dumps(dict(analysis, verdict="FAIL")), encoding="utf-8")
    analysis_entry = {
        "path": "docs/artifacts/gate-d-analysis.json",
        "sha256": hashlib.sha256(passed.read_bytes()).hexdigest(),
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
        "basis": [dict(row_entry), dict(validation_entry), dict(analysis_entry)],
        "prior_attempts": [
            {
                "path": "docs/artifacts/gate-d-attempt.json",
                "sha256": hashlib.sha256(refused.read_bytes()).hexdigest(),
                "verdict": "FAIL",
            }
        ],
        "oracle": {"dsa_manifest_sha256": "a" * 64, "token_manifest_sha256": "c" * 64},
        "engine_source_run": source_run,
        "reference_row": reference_row,
        "semantics": "test",
        "gate_d_closed": False,
        "performance_claim": False,
    }
    return record, reference_row


def test_a_disclosed_failed_attempt_does_not_make_a_record_unsealable(tmp_path: Path) -> None:
    """Disclosure must not be punished: prior attempts are not the basis."""
    import hashlib
    import json

    from glm_tpu.greenfield.validation.ws32_short_context import (
        load_ws32_adjudicated_divergence,
    )

    record, _ = _adjudication_record_fixture(tmp_path)
    path = tmp_path / "record.json"
    payload = json.dumps(record).encode()
    path.write_bytes(payload)
    loaded = load_ws32_adjudicated_divergence(
        path, expected_sha256=hashlib.sha256(payload).hexdigest(), repository_root=tmp_path
    )
    assert loaded.step == 0 and loaded.event_index == 1

    # The same refused analysis in the BASIS is a refusal, because the basis is
    # the ground the record stands on.
    hostile = json.loads(payload)
    attempt = dict(hostile["prior_attempts"][0])
    attempt.pop("verdict")
    hostile["basis"].append(attempt)
    payload = json.dumps(hostile).encode()
    path = tmp_path / "hostile.json"
    path.write_bytes(payload)
    with np.testing.assert_raises_regex(ValueError, "names a refused adjudication"):
        load_ws32_adjudicated_divergence(
            path, expected_sha256=hashlib.sha256(payload).hexdigest(), repository_root=tmp_path
        )


def test_the_declared_reference_row_must_be_the_one_the_analysis_used(tmp_path: Path) -> None:
    """A declared row the PASS analysis did not use is decorative."""
    import hashlib
    import json

    from glm_tpu.greenfield.validation.ws32_short_context import (
        load_ws32_adjudicated_divergence,
    )

    record, reference_row = _adjudication_record_fixture(tmp_path)
    analysis_path = tmp_path / "docs/artifacts/gate-d-analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["reference_row"] = dict(reference_row, convention="rms_norm_eps_1e-6")
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    digest = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
    record["basis"][2]["sha256"] = digest
    record["analysis"]["sha256"] = digest

    payload = json.dumps(record).encode()
    path = tmp_path / "record.json"
    path.write_bytes(payload)
    with np.testing.assert_raises_regex(ValueError, "different reference row"):
        load_ws32_adjudicated_divergence(
            path, expected_sha256=hashlib.sha256(payload).hexdigest(), repository_root=tmp_path
        )


def test_a_prior_attempt_whose_bytes_drifted_is_refused(tmp_path: Path) -> None:
    """P2-5: prior-attempt SHA verification survived mutation."""
    import hashlib
    import json

    from glm_tpu.greenfield.validation.ws32_short_context import (
        load_ws32_adjudicated_divergence,
    )

    record, _ = _adjudication_record_fixture(tmp_path)
    record["prior_attempts"][0]["sha256"] = "0" * 64
    payload = json.dumps(record).encode()
    path = tmp_path / "record.json"
    path.write_bytes(payload)
    with np.testing.assert_raises_regex(ValueError, "prior attempt drifted"):
        load_ws32_adjudicated_divergence(
            path, expected_sha256=hashlib.sha256(payload).hexdigest(), repository_root=tmp_path
        )


def test_an_unpinned_later_event_alarm_is_refused(tmp_path: Path) -> None:
    """P2-3: §21.2 fixes the threshold; raising it skips the alarm entirely."""
    import hashlib
    import json

    from glm_tpu.greenfield.validation.ws32_short_context import (
        load_ws32_adjudicated_divergence,
    )

    record, _ = _adjudication_record_fixture(tmp_path)
    record["later_event_alarm"] = 4096
    payload = json.dumps(record).encode()
    path = tmp_path / "record.json"
    path.write_bytes(payload)
    with np.testing.assert_raises_regex(ValueError, "schema drifted"):
        load_ws32_adjudicated_divergence(
            path, expected_sha256=hashlib.sha256(payload).hexdigest(), repository_root=tmp_path
        )


def test_artifact_paths_may_not_escape_the_reviewed_directory() -> None:
    """P3: `startswith` admitted `..` components."""
    from glm_tpu.greenfield.validation.ws32_short_context import _committed_artifact_path

    assert _committed_artifact_path("docs/artifacts/gate-d-row.npy", ".npy") is True
    assert _committed_artifact_path("docs/artifacts/gate-d-a.json", ".json", ".npy") is True
    for hostile in (
        "docs/artifacts/gate-d-x/../../../etc/passwd.npy",
        "docs/artifacts/gate-d-x/nested.npy",
        "docs/artifacts/other.npy",
        "/docs/artifacts/gate-d-row.npy",
        "docs/artifacts/gate-d-row.txt",
        17,
    ):
        assert _committed_artifact_path(hostile, ".json", ".npy") is False, hostile


def _hostile(tmp_path: Path, mutate) -> tuple[Path, str]:
    import hashlib
    import json

    record, _ = _adjudication_record_fixture(tmp_path)
    mutate(record)
    payload = json.dumps(record).encode()
    path = tmp_path / "hostile-record.json"
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def test_every_analysis_binding_is_enforced(tmp_path: Path) -> None:
    """P2-1: each of these deletions previously left the suite green."""
    import json

    from glm_tpu.greenfield.validation.ws32_short_context import (
        load_ws32_adjudicated_divergence,
    )

    cases = (
        # A null value would skip every check guarded on the field.
        (lambda r: r.__setitem__("analysis", None), "does not declare analysis"),
        (lambda r: r.__setitem__("reference_row", None), "does not declare reference_row"),
        (lambda r: r.__setitem__("prior_attempts", None), "does not declare prior_attempts"),
        # Schema, membership, kind, event and payload bindings.
        (lambda r: r["analysis"].__setitem__("sha256", "zz"), "analysis binding drifted"),
        (lambda r: r["analysis"].__setitem__("path", "docs/artifacts/other.json"),
         "analysis binding drifted"),
        (lambda r: r["basis"].pop(2), "analysis is not in the basis"),
        (lambda r: r.__setitem__("expected_only", [999]), "not the analysis's own sets"),
    )
    for mutate, message in cases:
        path, digest = _hostile(tmp_path, mutate)
        with np.testing.assert_raises_regex(ValueError, message):
            load_ws32_adjudicated_divergence(
                path, expected_sha256=digest, repository_root=tmp_path
            )
        path.unlink()

    # The analysis file itself must be a §21.2 adjudication of this event that
    # passed, carrying the record's own sets.
    analysis_path = tmp_path / "docs/artifacts/gate-d-analysis.json"
    for mutate_analysis, message in (
        (lambda a: a.__setitem__("artifact_kind", "something_else"),
         "is not a §21.2 adjudication"),
        (lambda a: a.__setitem__("event_index", 7), "adjudicates a different event"),
        (lambda a: a.__setitem__("step", 3), "adjudicates a different event"),
        (lambda a: a.__setitem__("engine_source_run", "other"), "adjudicates a different event"),
        (lambda a: a.__setitem__("expected_only", [999]), "not the analysis's own sets"),
        (lambda a: a.__setitem__("observed_only", [999]), "not the analysis's own sets"),
        (lambda a: a.pop("checks"), "does not carry the §21.2 checks"),
        (lambda a: a["checks"]["bias"].__setitem__("pass", False),
         "does not carry the §21.2 checks"),
        (lambda a: a["checks"].pop("reference_band_capacity"),
         "does not carry the §21.2 checks"),
        (lambda a: a["checks"].__setitem__("a_future_check", {"pass": False}),
         "does not carry the §21.2 checks"),
    ):
        record, _ = _adjudication_record_fixture(tmp_path)
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        mutate_analysis(analysis)
        analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
        import hashlib

        record["analysis"]["sha256"] = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
        record["basis"][2] = record["analysis"]
        payload = json.dumps(record).encode()
        path = tmp_path / "hostile-analysis.json"
        path.write_bytes(payload)
        with np.testing.assert_raises_regex(ValueError, message):
            load_ws32_adjudicated_divergence(
                path, expected_sha256=hashlib.sha256(payload).hexdigest(),
                repository_root=tmp_path,
            )
        path.unlink()


def test_a_reference_row_for_an_unregistered_event_is_refused(tmp_path: Path) -> None:
    """P2-1 M5: the registry must have an entry for THIS event."""
    from glm_tpu.greenfield.validation.ws32_short_context import (
        load_ws32_adjudicated_divergence,
    )

    path, digest = _hostile(tmp_path, lambda r: r.__setitem__("decode_position", 4096))
    with np.testing.assert_raises_regex(ValueError, "no reviewed FP64 reference row"):
        load_ws32_adjudicated_divergence(
            path, expected_sha256=digest, repository_root=tmp_path
        )


def test_a_basis_artifact_reached_through_a_symlink_is_refused(tmp_path: Path) -> None:
    """A string predicate does not stop a symlink out of the reviewed tree."""
    import hashlib
    import json
    import os

    from glm_tpu.greenfield.validation.ws32_short_context import (
        load_ws32_adjudicated_divergence,
    )

    record, _ = _adjudication_record_fixture(tmp_path)
    outside = tmp_path.parent / "outside-row.npy"
    outside.write_bytes((tmp_path / record["reference_row"]["path"]).read_bytes())
    link = tmp_path / "docs/artifacts/gate-d-linked-row.npy"
    os.symlink(outside, link)
    record["basis"].append(
        {
            "path": "docs/artifacts/gate-d-linked-row.npy",
            "sha256": hashlib.sha256(link.read_bytes()).hexdigest(),
        }
    )
    payload = json.dumps(record).encode()
    path = tmp_path / "symlink-record.json"
    path.write_bytes(payload)
    with np.testing.assert_raises_regex(ValueError, "leaves the reviewed tree"):
        load_ws32_adjudicated_divergence(
            path, expected_sha256=hashlib.sha256(payload).hexdigest(),
            repository_root=tmp_path,
        )


def test_an_adjudication_with_extra_passing_checks_is_accepted(tmp_path: Path) -> None:
    """A future §21.2 check must not make a thorough adjudication unloadable."""
    import hashlib
    import json

    from glm_tpu.greenfield.validation.ws32_short_context import (
        load_ws32_adjudicated_divergence,
    )

    record, _ = _adjudication_record_fixture(tmp_path)
    analysis_path = tmp_path / "docs/artifacts/gate-d-analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["checks"]["a_future_check"] = {"pass": True}
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    digest = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
    record["analysis"]["sha256"] = digest
    record["basis"][2]["sha256"] = digest
    payload = json.dumps(record).encode()
    path = tmp_path / "record.json"
    path.write_bytes(payload)
    loaded = load_ws32_adjudicated_divergence(
        path, expected_sha256=hashlib.sha256(payload).hexdigest(), repository_root=tmp_path
    )
    assert loaded.event_index == 1


def test_the_basis_reference_row_resolution_is_unambiguous_or_absent() -> None:
    """The grandfathered row comes from the basis; ambiguity must be refused."""
    from glm_tpu.greenfield.validation.ws32_short_context import (
        _sole_basis_reference_row,
    )

    one = [
        {"path": "docs/artifacts/gate-d-a.json", "sha256": "0" * 64},
        {"path": "docs/artifacts/gate-d-row.npy", "sha256": "1" * 64},
    ]
    assert _sole_basis_reference_row(one) == "docs/artifacts/gate-d-row.npy"
    two = one + [{"path": "docs/artifacts/gate-d-other.npy", "sha256": "2" * 64}]
    assert _sole_basis_reference_row(two) is None
    assert _sole_basis_reference_row(one[:1]) is None
    assert _sole_basis_reference_row([]) is None


def test_the_basis_must_name_the_record_that_validated_the_reference_row(tmp_path) -> None:
    """P3-7 / §21.2 item 3: the row is usable only because a review checked it."""
    from glm_tpu.greenfield.validation.ws32_short_context import (
        load_ws32_adjudicated_divergence,
    )

    path, digest = _hostile(
        tmp_path, lambda r: r["basis"].__delitem__(1)  # the validation record
    )
    with np.testing.assert_raises_regex(ValueError, "record that validated"):
        load_ws32_adjudicated_divergence(
            path, expected_sha256=digest, repository_root=tmp_path
        )


def test_the_analysis_must_agree_on_the_event_identity_keys(tmp_path) -> None:
    """P2-3: these clauses used to default to the record's own values."""
    import hashlib
    import json

    from glm_tpu.greenfield.validation.ws32_short_context import (
        load_ws32_adjudicated_divergence,
    )

    analysis_path = tmp_path / "docs/artifacts/gate-d-analysis.json"
    for key, value in (
        ("context", "2k"),
        ("decode_position", 2033),
        ("producer_layer_id", 62),
    ):
        record, _ = _adjudication_record_fixture(tmp_path)
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        analysis[key] = value
        analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
        digest = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
        record["analysis"]["sha256"] = digest
        record["basis"][2]["sha256"] = digest
        payload = json.dumps(record).encode()
        path = tmp_path / f"hostile-{key}.json"
        path.write_bytes(payload)
        with np.testing.assert_raises_regex(ValueError, "adjudicates a different event"):
            load_ws32_adjudicated_divergence(
                path,
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                repository_root=tmp_path,
            )


def test_a_record_declaring_a_row_carries_its_validation_record_path(tmp_path) -> None:
    """The sealer pins that path to the run's commit; the loader must fill it."""
    import hashlib
    import json

    from glm_tpu.greenfield.validation.ws32_short_context import (
        REFERENCE_ROWS,
        load_ws32_adjudicated_divergence,
    )

    record, _ = _adjudication_record_fixture(tmp_path)
    payload = json.dumps(record).encode()
    path = tmp_path / "record.json"
    path.write_bytes(payload)
    loaded = load_ws32_adjudicated_divergence(
        path, expected_sha256=hashlib.sha256(payload).hexdigest(), repository_root=tmp_path
    )
    registered = REFERENCE_ROWS[("8k", 8155, 1, "rms_norm_eps_1e-5")]
    assert loaded.reference_validation_path == registered["validation_path"]
    assert loaded.reference_row_path == registered["path"]


def test_the_grandfathered_record_declares_no_validation_path() -> None:
    """It predates the registry, so there is nothing to pin; the row still is."""
    import hashlib
    from pathlib import Path as _Path

    from glm_tpu.greenfield.validation.ws32_short_context import (
        load_ws32_adjudicated_divergence,
    )

    root = _Path(__file__).resolve().parents[3]
    committed = root / "docs/artifacts/gate-d-ws32-8k-adjudicated-divergence-20260905.json"
    loaded = load_ws32_adjudicated_divergence(
        committed,
        expected_sha256=hashlib.sha256(committed.read_bytes()).hexdigest(),
        repository_root=root,
    )
    assert loaded.reference_validation_path is None
    assert loaded.reference_row_path == (
        "docs/artifacts/gate-d-event1-fp64-reference-row-20260905.npy"
    )
