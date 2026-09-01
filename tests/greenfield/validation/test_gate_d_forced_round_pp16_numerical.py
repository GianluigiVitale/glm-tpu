from __future__ import annotations

import importlib.util
import json
import zipfile
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
SOURCE = ROOT / "scripts/greenfield/run_gate_d_forced_round_pp16_numerical.py"
CAUSAL_REPORT = ROOT / (
    "docs/artifacts/gate-d-forced-round-pp16-hlo-causal-adjudication.json"
)
SOURCE_CERTIFICATE = ROOT / (
    "docs/artifacts/gate-d-forced-round-pp16-numerical-source-v2.json"
)
ACQUIRED_OPTIMIZED_HLO = Path(
    "/home/gianl/gate-d-runs/"
    "gate_d_forced_round_pp16_hlo_20260901T141137500138602Z/"
    "hlo/forced_round_pp16_stage0.optimized_hlo.txt"
)
FAILED_V1_RUNTIME_OPTIMIZED_HLO = Path(
    "/home/gianl/gate-d-runs/"
    "gate_d_forced_round_pp16_numerical_20260901T161156923004972Z/"
    "hlo/forced_round_pp16_stage0.optimized_hlo.txt"
)
SPEC = importlib.util.spec_from_file_location(
    "gate_d_forced_round_pp16_numerical", SOURCE
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _hlo_report() -> dict[str, object]:
    return json.loads(CAUSAL_REPORT.read_text())


def _admission() -> dict[str, object]:
    arrays = {"x": {"shape": [1], "storage_dtype": "|u1", "array_sha256": "a" * 64}}
    return {
        "candidate_results": [
            {
                "id": "compensated_auxiliary_dependency",
                "capsule": {"producer": {"input_arrays": arrays}},
            }
        ],
        "precompile_numerical_policy": {
            "accepted_tpu_equality": {
                "event1_outputs": {
                    "positions_sha256": MODULE.EXPECTED_ACCEPTED_POSITIONS_SHA256,
                    "scores_sha256": MODULE.EXPECTED_ACCEPTED_SCORES_SHA256,
                    "valid_count": MODULE.EXPECTED_VALID_COUNT,
                },
                "required_future_checks": [
                    "identical_protected_input_and_event_identity",
                    "exact_event1_selected_positions",
                    "exact_event1_selected_scores",
                    "exact_event1_valid_count",
                    "descending_score_then_lowest_global_position_tie_order",
                ],
                "status": "deferred.mandatory.protected_tpu_numerical_replay",
            }
        },
    }


def _capsule() -> dict[str, object]:
    return {
        "artifact": {
            "path": "candidate-state.npz",
            "sha256": MODULE.CAPSULE_STATE_SHA256,
        },
        "candidate_id": "compensated_auxiliary_dependency",
        "coherence_id": "greenfield_gate_d_compensated_capsule_20260831T124838Z",
        "producer_input_artifact": {
            "path": "candidate-inputs.npz",
            "sha256": MODULE.CAPSULE_INPUT_SHA256,
        },
        "schema_version": 2,
    }


def _authority(admission: dict[str, object]) -> dict[str, object]:
    candidate = admission["candidate_results"][0]  # type: ignore[index]
    arrays = candidate["capsule"]["producer"]["input_arrays"]  # type: ignore[index]
    return {
        "authority_kind": "gate.d.capsule.execution.v1",
        "candidate_id": "compensated_auxiliary_dependency",
        "input_arrays": arrays,
        "schema_version": 1,
    }


def _host_materialization_authority() -> dict[str, object]:
    return {
        "artifact_kind": "gate_d_pp16_numerical_host_materialization_equivalence",
        "backend": {
            "device_count": 2,
            "jax_platform": "cpu",
            "xla_flags": "--xla_force_host_platform_device_count=2",
        },
        "capsule_input_sha256": MODULE.CAPSULE_INPUT_SHA256,
        "classification": (
            "HOST_DERIVED_INPUTS_BYTE_EXACT;TPU_NUMERICAL_UNPROVEN;GATE_D_OPEN"
        ),
        "comparison": {
            "host_query_matches_sealed_jax_reference": True,
            "host_wk_bf16_then_fp32_matches_sealed_jax_reference": True,
            "query_weight_fp32_sha256": MODULE.EXPECTED_QUERY_WEIGHT_FP32_SHA256,
            "wk_weight_fp32_sha256": MODULE.EXPECTED_WK_WEIGHT_FP32_SHA256,
        },
        "gate_d_closed": False,
        "libtpu_lock_holder_after_test": False,
        "performance_claim": False,
        "recorded_at_utc": MODULE.EXPECTED_HOST_MATERIALIZATION_RECORDED_AT_UTC,
        "schema_version": 1,
        "tpu_backend_initialized": False,
        "tpu_execution_performed": False,
    }


def _watchpoint_capsule(
    host: dict[str, np.ndarray],
    inputs: dict[str, np.ndarray],
    rms_input_fp32: np.ndarray,
) -> dict[str, object]:
    state = {
        "cache_history": host["index_cache_owners"].view(np.uint16),
        "current_key": host["current_key_owners"][0, 0],
        "event1_positions": host["selected_positions_owners"][0],
        "event1_scores": host["selected_scores_owners"][0],
        "event1_valid_count": host["valid_counts_owners"][0],
        "head_weights": host["head_weights_owners"],
        "normalized": host["normalized_hidden_owners"].view(np.uint16),
        "query": host["query_owners"],
        "rms_hidden_update": inputs["rms_hidden_update_bf16_bits"],
        "rms_input": rms_input_fp32,
        "rms_residual": inputs["rms_residual_bf16_bits"],
    }
    definitions = [
        ("layer1.cache_history", [("value", "cache_history", [])]),
        ("layer1.current_key", [("value", "current_key", [])]),
        (
            "layer1.head_weights",
            [("owner0", "head_weights", [0, 0]), ("owner1", "head_weights", [1, 0])],
        ),
        (
            "layer1.normalized",
            [("owner0", "normalized", [0, 0]), ("owner1", "normalized", [1, 0])],
        ),
        (
            "layer1.query",
            [("owner0", "query", [0, 0]), ("owner1", "query", [1, 0])],
        ),
        ("layer1.rms_input_fp32", [("value", "rms_input", [])]),
        (
            "layer1.rms_operands_bf16",
            [
                ("hidden_update", "rms_hidden_update", []),
                ("residual", "rms_residual", []),
            ],
        ),
        (
            "layer1.scorer_event1",
            [
                ("positions", "event1_positions", []),
                ("scores", "event1_scores", []),
                ("valid_count", "event1_valid_count", []),
            ],
        ),
    ]
    watchpoints = []
    for watchpoint_id, records in definitions:
        arrays = []
        for role, key, prefix in records:
            value = state[key][tuple(prefix)] if prefix else state[key]
            arrays.append(
                {
                    "array_key": key,
                    "array_sha256": MODULE._array_sha256(value),
                    "index_prefix": prefix,
                    "role": role,
                }
            )
        watchpoints.append({"id": watchpoint_id, "arrays": arrays})
    return {"watchpoints": watchpoints}


def _host_outputs() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    positions = np.arange(2048, dtype=np.int32).reshape(1, 2048)
    scores = np.arange(2048, 0, -1, dtype=np.float32).reshape(1, 2048)

    def owners(value: np.ndarray) -> np.ndarray:
        return np.stack([value, value])

    host = {
        "normalized_hidden_owners": owners(np.zeros((1, 6144), dtype=np.uint16)),
        "query_owners": owners(np.zeros((1, 32, 128), dtype=np.float32)),
        "head_weights_owners": owners(np.zeros((1, 32), dtype=np.float32)),
        "current_key_owners": owners(np.zeros((1, 128), dtype=np.float32)),
        "index_cache_owners": np.zeros((2, 16, 256, 128), dtype=np.uint16),
        "selected_positions_owners": owners(positions),
        "valid_counts_owners": owners(np.asarray([2048], dtype=np.int32)),
        "selected_scores_owners": owners(scores),
        "contract_valid_owners": np.asarray([1, 1], dtype=np.uint8),
    }
    inputs = {
        "rms_hidden_update_bf16_bits": np.zeros((6144,), dtype=np.uint16),
        "rms_residual_bf16_bits": np.zeros((6144,), dtype=np.uint16),
    }
    return host, inputs, np.zeros((6144,), dtype=np.float32)


def test_hlo_adjudication_accepts_exact_boundary() -> None:
    assert MODULE.validate_hlo_adjudication(_hlo_report()) == {
        "optimized_hlo_sha256": MODULE.EXPECTED_OPTIMIZED_HLO_SHA256,
        "stablehlo_sha256": MODULE.EXPECTED_STABLEHLO_SHA256,
    }


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("schema_version",), True),
        (("schema_version",), 1.0),
        (("authorization", "numerical_execution"), True),
        (("optimized_hlo", "collective_count"), True),
        (("optimized_hlo", "collective_groups"), [[0, 1, 2]]),
        (("optimized_hlo", "sha256"), "0" * 64),
        (("stablehlo", "sha256"), "0" * 64),
    ],
)
def test_hlo_adjudication_rejects_hostile_rebinding(
    path: tuple[object, ...], replacement: object
) -> None:
    report = deepcopy(_hlo_report())
    cursor: object = report
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = replacement  # type: ignore[index]
    with pytest.raises(RuntimeError, match="HLO adjudication"):
        MODULE.validate_hlo_adjudication(report)


def test_numerical_policy_accepts_only_exact_independent_event_authority() -> None:
    admission = _admission()
    assert MODULE.validate_numerical_policy(admission)["valid_count"] == 2048
    for replacement in (True, 2048.0, 2047):
        attack = deepcopy(admission)
        attack["precompile_numerical_policy"]["accepted_tpu_equality"][  # type: ignore[index]
            "event1_outputs"
        ]["valid_count"] = replacement
        with pytest.raises(RuntimeError, match="numerical policy"):
            MODULE.validate_numerical_policy(attack)


def test_host_materialization_authority_accepts_only_exact_cpu_proof() -> None:
    authority = _host_materialization_authority()
    validated = MODULE.validate_host_materialization_authority(authority)
    assert validated["authority_sha256"] == MODULE.HOST_MATERIALIZATION_AUTHORITY_SHA256
    attacks = [
        (("schema_version",), True),
        (("schema_version",), 1.0),
        (("backend", "device_count"), True),
        (("backend", "jax_platform"), "tpu"),
        (("capsule_input_sha256",), "0" * 64),
        (("classification",), "HOST_DERIVED_INPUTS_BYTE_EXACT"),
        (("comparison", "host_query_matches_sealed_jax_reference"), 1),
        (("comparison", "query_weight_fp32_sha256"), "0" * 64),
        (("comparison", "wk_weight_fp32_sha256"), "0" * 64),
        (("gate_d_closed",), True),
        (("libtpu_lock_holder_after_test",), True),
        (("performance_claim",), True),
        (("tpu_backend_initialized",), True),
        (("tpu_execution_performed",), True),
    ]
    for path, replacement in attacks:
        attack = deepcopy(authority)
        cursor: object = attack
        for key in path[:-1]:
            cursor = cursor[key]  # type: ignore[index]
        cursor[path[-1]] = replacement  # type: ignore[index]
        with pytest.raises(RuntimeError, match="host-materialization authority"):
            MODULE.validate_host_materialization_authority(attack)

    extra = deepcopy(authority)
    extra["unbound"] = False
    with pytest.raises(RuntimeError, match="host-materialization authority"):
        MODULE.validate_host_materialization_authority(extra)


def test_capsule_binding_rejects_schema_aliases_and_cross_candidate_arrays() -> None:
    admission = _admission()
    capsule = _capsule()
    authority = _authority(admission)
    assert (
        MODULE._validate_capsule_bindings(capsule, authority, admission)["input_sha256"]
        == MODULE.CAPSULE_INPUT_SHA256
    )
    for value in (True, 2.0):
        attack = deepcopy(capsule)
        attack["schema_version"] = value
        with pytest.raises(RuntimeError, match="capsule binding"):
            MODULE._validate_capsule_bindings(attack, authority, admission)
    cross = deepcopy(authority)
    cross["input_arrays"] = {"different": {}}
    with pytest.raises(RuntimeError, match="capsule binding"):
        MODULE._validate_capsule_bindings(capsule, cross, admission)


def test_deterministic_npz_round_trip_and_duplicate_member_rejection() -> None:
    values = {
        "a": np.arange(8, dtype=np.int32),
        "b": np.asarray([1, 2], dtype=np.uint8),
    }
    raw = MODULE._deterministic_npz(values, np)
    assert raw == MODULE._deterministic_npz(values, np)
    records = {
        name: {
            "array_sha256": MODULE._array_sha256(value),
            "shape": list(value.shape),
            "storage_dtype": value.dtype.str,
        }
        for name, value in values.items()
    }
    loaded = MODULE._load_npz_arrays(raw, records, np)
    assert all(np.array_equal(loaded[name], value) for name, value in values.items())

    duplicate = BytesIO()
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr("a.npy", b"one")
        archive.writestr("a.npy", b"two")
        archive.writestr("b.npy", b"three")
    with pytest.raises(RuntimeError, match="member catalogue"):
        MODULE._load_npz_arrays(duplicate.getvalue(), records, np)

    aggregate_bomb = BytesIO()
    with zipfile.ZipFile(
        aggregate_bomb, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr("a.npy", bytes(17 * 1024 * 1024))
        archive.writestr("b.npy", bytes(17 * 1024 * 1024))
    with pytest.raises(RuntimeError, match="member catalogue"):
        MODULE._load_npz_arrays(aggregate_bomb.getvalue(), records, np)


def test_host_fp8_materialization_has_exact_query_and_bf16_wk_phases() -> None:
    bits = np.asarray([[[0x38, 0x3C, 0xB8], [0x30, 0x40, 0xB0]]], dtype=np.uint8)
    scales = np.asarray([[[2.0]]], dtype=np.float32)
    inputs = {
        "wq_b_weight_bits": bits,
        "wq_b_scale_inv": scales,
        "wk_weight_bits": bits,
        "wk_scale_inv": scales,
    }
    query, wk = MODULE.materialize_derived_weights(inputs, np, ml_dtypes)
    expected = MODULE._fp8_lookup(np)[bits.astype(np.int32)] * np.float32(2.0)
    assert query.dtype == np.float32
    assert query.tobytes() == np.ascontiguousarray(expected, dtype=np.float32).tobytes()
    expected_wk = np.ascontiguousarray(expected, dtype=ml_dtypes.bfloat16).astype(
        np.float32
    )
    assert wk.tobytes() == np.ascontiguousarray(expected_wk).tobytes()


REAL_CAPSULE_INPUTS = Path(
    "/home/gianl/gate-d-runs/"
    "greenfield_gate_d_compensated_capsule_20260831T124838Z/candidate-inputs.npz"
)
REAL_CAPSULE_AUTHORITY = Path(
    "/home/gianl/gate-d-runs/"
    "greenfield_gate_d_compensated_capsule_20260831T124838Z-execution-authority.json"
)


@pytest.mark.skipif(
    not REAL_CAPSULE_INPUTS.is_file() or not REAL_CAPSULE_AUTHORITY.is_file(),
    reason="exact local Gate-D capsule is unavailable",
)
def test_real_capsule_derived_weights_match_bound_authority_and_reject_bit_flip() -> (
    None
):
    authority = json.loads(REAL_CAPSULE_AUTHORITY.read_text(encoding="ascii"))
    raw = MODULE._snapshot_regular(REAL_CAPSULE_INPUTS)
    assert sha256(raw).hexdigest() == MODULE.CAPSULE_INPUT_SHA256
    inputs = MODULE._load_npz_arrays(raw, authority["input_arrays"], np)
    query, wk = MODULE.materialize_derived_weights(inputs, np, ml_dtypes)
    observed = MODULE.validate_derived_weights(query, wk, np)
    assert observed == {
        "query_weight": {
            "array_sha256": MODULE.EXPECTED_QUERY_WEIGHT_FP32_SHA256,
            "semantic_dtype": "float32",
            "shape": list(MODULE.EXPECTED_QUERY_WEIGHT_SHAPE),
        },
        "wk_weight": {
            "array_sha256": MODULE.EXPECTED_WK_WEIGHT_FP32_SHA256,
            "semantic_dtype": "float32",
            "shape": list(MODULE.EXPECTED_WK_WEIGHT_SHAPE),
        },
    }
    attacked = query.copy()
    attacked.view(np.uint8).reshape(-1)[0] ^= np.uint8(1)
    with pytest.raises(RuntimeError, match="host-derived graph input identity"):
        MODULE.validate_derived_weights(attacked, wk, np)


def test_tie_order_requires_descending_score_lowest_position_and_unique_range() -> None:
    positions = np.arange(2048, dtype=np.int32)
    scores = np.arange(2048, 0, -1, dtype=np.float32)
    assert MODULE._tie_order_is_exact(positions, scores, 2048, np)
    tied = scores.copy()
    tied[:2] = tied[0]
    swapped = positions.copy()
    swapped[:2] = [1, 0]
    assert not MODULE._tie_order_is_exact(swapped, tied, 2048, np)
    duplicate = positions.copy()
    duplicate[-1] = duplicate[-2]
    assert not MODULE._tie_order_is_exact(duplicate, scores, 2048, np)
    outside = positions.copy()
    outside[-1] = 8156
    assert not MODULE._tie_order_is_exact(outside, scores, 2048, np)


def test_classification_requires_causal_watchpoints_and_protected_tpu_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host, inputs, rms_input_fp32 = _host_outputs()
    capsule = _watchpoint_capsule(host, inputs, rms_input_fp32)
    rejected = MODULE.classify_outputs(host, capsule, inputs, rms_input_fp32, np)
    assert rejected["accepted_tpu_event_match"] is False
    assert rejected["causal_candidate_watchpoints_exact"] is True
    assert all(rejected["candidate_watchpoint_matches"].values())

    observed_positions = MODULE._array_sha256(host["selected_positions_owners"][0])
    observed_scores = MODULE._array_sha256(host["selected_scores_owners"][0])
    monkeypatch.setattr(
        MODULE, "EXPECTED_ACCEPTED_POSITIONS_SHA256", observed_positions
    )
    monkeypatch.setattr(MODULE, "EXPECTED_ACCEPTED_SCORES_SHA256", observed_scores)
    accepted = MODULE.classify_outputs(host, capsule, inputs, rms_input_fp32, np)
    assert accepted["accepted_tpu_event_match"] is True

    host["normalized_hidden_owners"][:, 0, 0] ^= np.uint16(1)
    causal_rejected = MODULE.classify_outputs(
        host, capsule, inputs, rms_input_fp32, np
    )
    assert all(causal_rejected["owner_agreement"].values())
    assert causal_rejected["observed_event1"] == accepted["observed_event1"]
    assert causal_rejected["causal_candidate_watchpoints_exact"] is False
    assert causal_rejected["accepted_tpu_event_match"] is False
    host["normalized_hidden_owners"][:, 0, 0] ^= np.uint16(1)

    host["selected_positions_owners"][1, 0, 0] += 1
    assert (
        MODULE.classify_outputs(host, capsule, inputs, rms_input_fp32, np)[
            "accepted_tpu_event_match"
        ]
        is False
    )


def test_source_orders_hlo_authentication_before_exactly_one_invocation_and_transfer() -> (
    None
):
    source = SOURCE.read_text()
    assert source.count("result = compiled(*arguments)") == 1
    assert source.count("transferred = jax.device_get(result)") == 1
    assert source.index("observed_hlo = validate_runtime_hlo_identity(") < source.index(
        "result = compiled(*arguments)"
    )
    assert source.index("invocation_count += 1") < source.index(
        "result = compiled(*arguments)"
    )
    assert source.count("materialize_derived_weights(inputs, np, ml_dtypes)") == 1
    assert source.index("derived_weights = validate_derived_weights") < source.index(
        "lowered = replay.lower(*abstract_arguments)"
    )
    assert '"performance_claim": False' in source
    assert '"gate_d_closed": False' in source


def test_git_reads_ignore_user_system_replace_and_network_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_check_output(arguments: list[str], *, env: dict[str, str]) -> bytes:
        observed["arguments"] = arguments
        observed["env"] = env
        return b"payload"

    monkeypatch.setattr(MODULE.subprocess, "check_output", fake_check_output)
    assert MODULE._git_bytes("show", "HEAD:path") == b"payload"
    assert observed["arguments"] == [
        "/usr/bin/git",
        "-C",
        str(MODULE.WORKTREE),
        "show",
        "HEAD:path",
    ]
    environment = observed["env"]
    assert isinstance(environment, dict)
    assert environment == {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PROTOCOL_FROM_USER": "0",
        "GIT_SSH_COMMAND": "/bin/false",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }


def test_source_and_compile_helper_have_expected_hash_and_no_unreviewed_graph_change() -> (
    None
):
    helper = ROOT / MODULE.COMPILE_HELPER_REPOSITORY_PATH
    assert sha256(helper.read_bytes()).hexdigest() == MODULE.COMPILE_HELPER_SHA256
    assert MODULE.HLO_SOURCE_CODE_HASH == "b8bdeb14c12a3742f3cc27ef91a4a2e406d412a5"
    assert MODULE.EXPECTED_STABLEHLO_SHA256.startswith("45eae705")
    assert MODULE.EXPECTED_OPTIMIZED_HLO_SHA256.startswith("a0b87e2b")


def test_source_certificate_binds_driver_predecessors_and_execution_boundary() -> None:
    certificate = json.loads(SOURCE_CERTIFICATE.read_text())
    assert certificate["artifact_kind"] == ("gate_d_forced_round_pp16_numerical_source")
    assert certificate["authorization"] == {
        "cloud_mutation": False,
        "full_decoder": False,
        "persistence_only": True,
        "tpu_execution": False,
    }
    assert certificate["driver"] == {
        "byte_count": SOURCE.stat().st_size,
        "installed_path": str(MODULE.INSTALLED_DRIVER_PATH),
        "repository_path": MODULE.DRIVER_REPOSITORY_PATH,
        "sha256": sha256(SOURCE.read_bytes()).hexdigest(),
    }
    assert certificate["hlo_contract"]["derived_numerical_optimized"] == {
        "byte_count": MODULE.EXPECTED_NUMERICAL_OPTIMIZED_HLO_BYTES,
        "sha256": MODULE.EXPECTED_NUMERICAL_OPTIMIZED_HLO_SHA256,
    }
    assert certificate["predecessors"]["causal_hlo_adjudication_sha256"] == (
        MODULE.HLO_ADJUDICATION_SHA256
    )
    assert certificate["predecessors"]["capsule_sha256"] == MODULE.CAPSULE_SHA256
    assert certificate["execution_contract"]["input_logical_rows"] == 1
    assert certificate["execution_contract"]["physical_device_ids"] == [0, 1]
    assert certificate["numerical_acceptance"] == {
        "independent_event1_authority_required": True,
        "owner_agreement_required": True,
        "protected_causal_watchpoints_required": list(
            MODULE.REQUIRED_CAUSAL_WATCHPOINT_LABELS
        ),
        "scorer_cpu_watchpoints_diagnostic_only": True,
        "tie_order_required": True,
    }
    assert certificate["verification"]["focused_tests_passed"] == 24
    assert certificate["verification"]["adjacent_tests_passed"] == 0
    assert certificate["verification"]["historical_v1_adjacent_tests_reused"] == 221
    assert certificate["verification"]["sealed_history_tests_deselected"] == 0
    assert certificate["verification"]["ruff_version"] == "0.16.5"


def test_source_location_derivation_replays_real_accepted_hlo_bytes_exactly() -> None:
    accepted = ACQUIRED_OPTIMIZED_HLO.read_bytes()
    derived, binding = MODULE.derive_numerical_optimized_hlo(
        accepted, SOURCE.read_bytes()
    )
    assert binding["replacement_count"] == 3
    assert binding["source_sha256"] == MODULE.EXPECTED_OPTIMIZED_HLO_SHA256
    assert binding["derived_sha256"] == sha256(derived).hexdigest()
    assert binding["derived_byte_count"] == len(derived)
    restored = derived
    for replacement in reversed(binding["replacements"]):
        assert accepted.count(replacement["old"].encode("ascii")) == 1
        assert accepted.count(replacement["new"].encode("ascii")) == 0
        assert restored.count(replacement["new"].encode("ascii")) == 1
        restored = restored.replace(
            replacement["new"].encode("ascii"),
            replacement["old"].encode("ascii"),
            1,
        )
    assert restored == accepted


def test_source_location_derivation_matches_failed_runtime_after_v2_path_rebind() -> (
    None
):
    accepted = ACQUIRED_OPTIMIZED_HLO.read_bytes()
    failed_v1 = FAILED_V1_RUNTIME_OPTIMIZED_HLO.read_bytes()
    derived_v2, binding = MODULE.derive_numerical_optimized_hlo(
        accepted, SOURCE.read_bytes()
    )
    old_path = (
        b"/usr/local/libexec/glm-tpu/gate-d-forced-round-pp16-numerical-v1/"
        b"run_gate_d_forced_round_pp16_numerical.py"
    )
    new_path = str(MODULE.INSTALLED_DRIVER_PATH).encode("ascii")
    assert failed_v1.count(old_path) == 1
    assert failed_v1.count(new_path) == 0
    assert derived_v2 == failed_v1.replace(old_path, new_path, 1)
    module_call = next(
        item
        for item in binding["replacements"]
        if item["surface"] == "FileLocations module call"
    )
    assert module_call["new"] == "line=1273 end_line=1273"


def test_runtime_hlo_identity_distinguishes_acquired_and_derived_optimized_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stablehlo = b"exact stablehlo"
    acquired_optimized = b"accepted optimized preimage"
    derived_optimized = b"metadata-derived optimized runtime"
    monkeypatch.setattr(
        MODULE, "EXPECTED_STABLEHLO_SHA256", sha256(stablehlo).hexdigest()
    )
    monkeypatch.setattr(
        MODULE,
        "EXPECTED_OPTIMIZED_HLO_SHA256",
        sha256(acquired_optimized).hexdigest(),
    )
    monkeypatch.setattr(
        MODULE,
        "EXPECTED_NUMERICAL_OPTIMIZED_HLO_SHA256",
        sha256(derived_optimized).hexdigest(),
    )
    authority = {
        "stablehlo_sha256": sha256(stablehlo).hexdigest(),
        "optimized_hlo_sha256": sha256(acquired_optimized).hexdigest(),
    }
    assert MODULE.validate_runtime_hlo_identity(
        stablehlo,
        derived_optimized,
        stablehlo,
        derived_optimized,
        authority,
    ) == {
        "stablehlo_sha256": sha256(stablehlo).hexdigest(),
        "optimized_hlo_sha256": sha256(derived_optimized).hexdigest(),
    }

    with pytest.raises(RuntimeError, match="executable HLO identity"):
        MODULE.validate_runtime_hlo_identity(
            stablehlo,
            acquired_optimized,
            stablehlo,
            derived_optimized,
            authority,
        )
    with pytest.raises(RuntimeError, match="acquired HLO authority"):
        MODULE.validate_runtime_hlo_identity(
            stablehlo,
            derived_optimized,
            stablehlo,
            derived_optimized,
            {
                "stablehlo_sha256": authority["stablehlo_sha256"],
                "optimized_hlo_sha256": authority["stablehlo_sha256"],
            },
        )


def test_source_location_derivation_rejects_preimage_and_callsite_mutation() -> None:
    accepted = ACQUIRED_OPTIMIZED_HLO.read_bytes()
    corrupted = bytearray(accepted)
    corrupted[-1] ^= 1
    with pytest.raises(RuntimeError, match="preimage drifted"):
        MODULE.derive_numerical_optimized_hlo(bytes(corrupted), SOURCE.read_bytes())
    hostile_source = SOURCE.read_text().replace(
        "lowered = replay.lower(*abstract_arguments)",
        "lowered = replay.compile(*abstract_arguments)",
        1,
    )
    with pytest.raises(RuntimeError, match="callsite catalogue drifted"):
        MODULE.derive_numerical_optimized_hlo(accepted, hostile_source.encode("utf-8"))
