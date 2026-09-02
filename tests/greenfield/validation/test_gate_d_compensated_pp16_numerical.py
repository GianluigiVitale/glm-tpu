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
SOURCE = ROOT / "scripts/greenfield/run_gate_d_compensated_pp16_numerical.py"
SOURCE_LOCATION_BRIDGE = ROOT / (
    "docs/artifacts/gate-d-compensated-pp16-hlo-source-location-bridge.json"
)
ACQUIRED_OPTIMIZED_HLO = Path(
    "/home/gianl/gate-d-runs/"
    "gate_d_compensated_pp16_hlo_20260901T070612366187759Z/"
    "hlo/compensated_pp16_stage0.optimized_hlo.txt"
)
SPEC = importlib.util.spec_from_file_location("gate_d_pp16_numerical", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _hlo_report() -> dict[str, object]:
    collectives = [
        {
            "channel_id": channel,
            "dimension": 0,
            "opcode": "all-gather",
            "operand": {"dtype": "f32", "shape": [1, 1, 2048]},
            "replica_groups": [[0, 1]],
            "result": {"dtype": "f32", "shape": [2, 1, 2048]},
            "use_global_device_ids": True,
        }
        for channel in (2, 3, 4)
    ]
    return {
        "artifact_kind": "gate_d_compensated_pp16_hlo_adjudication",
        "claim_scope": "compile-only",
        "classification": (
            "HLO_LOCALITY_POLICY_ACCEPTED;TPU_NUMERICAL_UNPROVEN;"
            "NO_PERFORMANCE_CLAIM;GATE_D_OPEN"
        ),
        "code_hash": MODULE.HLO_SOURCE_CODE_HASH,
        "gate_d_closed": False,
        "hlo": {
            "optimized": {
                "collective_count": 3,
                "collectives": collectives,
                "logical_row_axis": 1,
                "num_partitions": 2,
                "primary_noninterference": True,
                "rooted_compensated_witness": True,
                "sha256": MODULE.EXPECTED_ACQUIRED_OPTIMIZED_HLO_SHA256,
            },
            "stablehlo": {
                "logical_rows": 1,
                "mesh_size": 2,
                "sha256": MODULE.EXPECTED_STABLEHLO_SHA256,
            },
        },
        "numerical_claim": False,
        "optimized_hlo_locality_policy_accepted": True,
        "performance_claim": False,
        "provenance": {},
        "remote_archive": {},
        "run_tag": "gate_d_compensated_pp16_hlo_20260901T070612366187759Z",
        "schema_version": 1,
        "tpu_numerical_execution_performed": False,
        "tpu_numerical_proven": False,
        "tpu_successor_authorized": False,
        "validator": {},
    }


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
    host: dict[str, np.ndarray], inputs: dict[str, np.ndarray]
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
        "rms_input": host["rms_input_fp32_owners"][0, 0],
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


def _host_outputs() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    positions = np.arange(2048, dtype=np.int32).reshape(1, 2048)
    scores = np.arange(2048, 0, -1, dtype=np.float32).reshape(1, 2048)

    def owners(value: np.ndarray) -> np.ndarray:
        return np.stack([value, value])

    host = {
        "rms_input_fp32_owners": owners(np.zeros((1, 6144), dtype=np.float32)),
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
    return host, inputs


def test_hlo_adjudication_accepts_exact_boundary() -> None:
    assert MODULE.validate_hlo_adjudication(_hlo_report()) == {
        "optimized_hlo_sha256": MODULE.EXPECTED_NUMERICAL_OPTIMIZED_HLO_SHA256,
        "stablehlo_sha256": MODULE.EXPECTED_STABLEHLO_SHA256,
    }


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("schema_version",), True),
        (("schema_version",), 1.0),
        (("tpu_successor_authorized",), True),
        (("hlo", "optimized", "collective_count"), True),
        (("hlo", "optimized", "collectives", 0, "replica_groups"), [[0, 1, 2]]),
        (("hlo", "optimized", "sha256"), "0" * 64),
        (("hlo", "stablehlo", "sha256"), "0" * 64),
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
    with pytest.raises(RuntimeError, match="HLO adjudication|locality catalogue"):
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


def test_classification_keeps_cpu_watchpoints_diagnostic_and_tpu_event_decisive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host, inputs = _host_outputs()
    capsule = _watchpoint_capsule(host, inputs)
    rejected = MODULE.classify_outputs(host, capsule, inputs, np)
    assert rejected["accepted_tpu_event_match"] is False
    assert all(rejected["cpu_candidate_watchpoint_matches_diagnostic_only"].values())

    observed_positions = MODULE._array_sha256(host["selected_positions_owners"][0])
    observed_scores = MODULE._array_sha256(host["selected_scores_owners"][0])
    monkeypatch.setattr(
        MODULE, "EXPECTED_ACCEPTED_POSITIONS_SHA256", observed_positions
    )
    monkeypatch.setattr(MODULE, "EXPECTED_ACCEPTED_SCORES_SHA256", observed_scores)
    accepted = MODULE.classify_outputs(host, capsule, inputs, np)
    assert accepted["accepted_tpu_event_match"] is True

    host["selected_positions_owners"][1, 0, 0] += 1
    assert (
        MODULE.classify_outputs(host, capsule, inputs, np)["accepted_tpu_event_match"]
        is False
    )


def test_source_orders_hlo_authentication_before_exactly_one_invocation_and_transfer() -> (
    None
):
    source = SOURCE.read_text()
    assert source.count("result = compiled(*arguments)") == 1
    assert source.count("transferred = jax.device_get(result)") == 1
    assert source.index('optimized_hlo.encode("utf-8") != derived_optimized_hlo_raw') < source.index(
        "result = compiled(*arguments)"
    )
    assert source.index("invocation_count += 1") < source.index(
        "result = compiled(*arguments)"
    )
    assert source.count("materialize_derived_weights(inputs, np, ml_dtypes)") == 1
    assert source.index("derived_weights = validate_derived_weights") < source.index(
        "lowered = replay.lower(*arguments)"
    )
    assert '"performance_claim": False' in source
    assert '"gate_d_closed": False' in source


def test_source_and_compile_helper_have_expected_hash_and_no_unreviewed_graph_change() -> (
    None
):
    helper = ROOT / MODULE.COMPILE_HELPER_REPOSITORY_PATH
    assert sha256(helper.read_bytes()).hexdigest() == MODULE.COMPILE_HELPER_SHA256
    assert MODULE.HLO_SOURCE_CODE_HASH == "94518b7d4ce788157afa98b7bc1f8144613852d5"
    assert MODULE.EXPECTED_STABLEHLO_SHA256.startswith("55d7940c")
    assert MODULE.EXPECTED_ACQUIRED_OPTIMIZED_HLO_SHA256.startswith("b6362349")
    assert MODULE.EXPECTED_NUMERICAL_OPTIMIZED_HLO_SHA256.startswith("fc6384d8")


def test_source_location_hlo_bridge_is_exact_and_bound_to_current_call_sites() -> None:
    bridge = json.loads(SOURCE_LOCATION_BRIDGE.read_bytes())
    assert bridge["source_hlo"]["sha256"] == (
        MODULE.EXPECTED_ACQUIRED_OPTIMIZED_HLO_SHA256
    )
    assert bridge["derived_numerical_hlo"]["sha256"] == (
        MODULE.EXPECTED_NUMERICAL_OPTIMIZED_HLO_SHA256
    )
    assert bridge["derivation"]["replacement_count"] == 3
    assert [item["occurrence_count"] for item in bridge["derivation"]["replacements"]] == [
        1,
        1,
        1,
    ]
    source_lines = SOURCE.read_text(encoding="ascii").splitlines()
    assert source_lines[1022].strip() == "lowered = replay.lower(*arguments)"
    assert source_lines[1196].strip() == "raise SystemExit(main())"


def test_source_location_bridge_replays_real_accepted_hlo_bytes_exactly() -> None:
    bridge_raw = SOURCE_LOCATION_BRIDGE.read_bytes()
    assert sha256(bridge_raw).hexdigest() == MODULE.HLO_SOURCE_LOCATION_BRIDGE_SHA256
    bridge = json.loads(bridge_raw)
    accepted = ACQUIRED_OPTIMIZED_HLO.read_bytes()
    derived, binding = MODULE.validate_hlo_source_location_bridge(bridge, accepted)
    assert len(derived) == MODULE.EXPECTED_NUMERICAL_OPTIMIZED_HLO_BYTES
    assert sha256(derived).hexdigest() == (
        MODULE.EXPECTED_NUMERICAL_OPTIMIZED_HLO_SHA256
    )
    assert binding["artifact_sha256"] == MODULE.HLO_SOURCE_LOCATION_BRIDGE_SHA256
    for replacement in bridge["derivation"]["replacements"]:
        assert accepted.count(replacement["old"].encode("ascii")) == 1
        assert accepted.count(replacement["new"].encode("ascii")) == 0


def test_source_location_bridge_rejects_schema_and_preimage_mutation() -> None:
    bridge = json.loads(SOURCE_LOCATION_BRIDGE.read_bytes())
    accepted = ACQUIRED_OPTIMIZED_HLO.read_bytes()
    attacked = deepcopy(bridge)
    attacked["derivation"]["replacement_count"] = 2
    with pytest.raises(RuntimeError, match="schema drifted"):
        MODULE.validate_hlo_source_location_bridge(attacked, accepted)
    corrupted = bytearray(accepted)
    corrupted[-1] ^= 1
    with pytest.raises(RuntimeError, match="preimage drifted"):
        MODULE.validate_hlo_source_location_bridge(bridge, bytes(corrupted))
