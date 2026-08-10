from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ml_dtypes import bfloat16

from scripts.greenfield.compile_short_decoder import (
    _canonicalize_dsa_internal_observation,
    _canonicalize_layer_residual_observation,
    _encode_bfloat16_bits,
    _load_short_context_dsa_oracle,
    _materialize_global_array,
    _observer_hlo_isolation_contract,
    _protected_short_context_label,
    _raw_token_sequence_contract,
    _validate_dsa_observation_step,
    _validate_completed_step_selected_states,
    _validate_token_observation_step,
)
from scripts.greenfield import compile_short_decoder as compile_module


REPO = Path(__file__).resolve().parents[3]
PROTECTED_RUNNER = REPO / "scripts/greenfield/run_short_decoder_compile_pp8.sh"
PROTECTED_8K_RUNNER = (
    REPO / "scripts/greenfield/run_short_decoder_compile_pp8_8k.sh"
)


def test_protected_runner_pins_fp32_feature_boundary_kernel() -> None:
    source = PROTECTED_RUNNER.read_text()
    assert (
        'if feature_reconstruct_down_fp32:\n'
        '        expected_kernel_counts["greenfield_fp32_to_bf16_r8_h6144"] = 75'
        in source
    )


def test_protected_runner_exposes_fail_closed_device_roundtrip() -> None:
    compiler = (REPO / "scripts/greenfield/compile_short_decoder.py").read_text()
    runner = PROTECTED_RUNNER.read_text()

    assert '"--verify-device-roundtrip"' in compiler
    assert "verify_device_roundtrip=args.verify_device_roundtrip" in compiler
    assert (
        "readonly VERIFY_DEVICE_ROUNDTRIP="
        "${GLM_GREENFIELD_RUNTIME_DEVICE_ROUNDTRIP:-0}" in runner
    )
    assert '--verify-device-roundtrip "$verify_device_roundtrip"' in runner
    assert '["device_roundtrip_verified"]\n    != verify_device_roundtrip' in runner
    assert '["device_roundtrip_bytes"]\n    != expected_roundtrip_bytes' in runner


def test_protected_runner_classifies_prefill_loops_fail_closed() -> None:
    runner = PROTECTED_RUNNER.read_text()

    assert '["outer_loop_count"] != 1' in runner
    assert '"expected_fused_qkv_internal_loop_count"' in runner
    assert '"fused_qkv_internal_loop_count"' in runner
    assert '"unclassified_loops"' in runner
    assert '["loop_count"] != 1' not in runner


def test_selected_linear_runtime_defaults_to_fused_qkv_gate_b_artifact() -> None:
    runner = PROTECTED_RUNNER.read_text()

    assert (
        "greenfield_runtime_feature_qkv_pack_pp8_20260808T141032190315066Z"
        in runner
    )
    assert (
        "123394906a153238e464fc096b626c77996b7cf22b95077b9377b8dcafbe699a"
        in runner
    )
    assert (
        "523afb1dc1ff2b954a9795c4deabdc4fd599c244c0bf1a9e3b8f971700548cb4"
        in runner
    )
    assert (
        "protected prefill repair requires the Gate-B-approved fused qkv-a runtime"
        in runner
    )


def test_prefill_index_repair_is_default_off_and_prerequisites_pinned() -> None:
    compiler = (REPO / "scripts/greenfield/compile_short_decoder.py").read_text()
    runner = PROTECTED_RUNNER.read_text()

    assert '"--prefill-index-repair"' in compiler
    assert "observe_prefill_index_inputs=True" in compiler
    assert '"prefill_index_repair": args.prefill_index_repair' in compiler
    assert (
        "readonly PREFILL_INDEX_REPAIR="
        "${GLM_GREENFIELD_PREFILL_INDEX_REPAIR:-0}" in runner
    )
    assert '--prefill-index-repair "$prefill_index_repair"' in runner
    assert "prefill index repair requires the protected 8K" in runner
    assert (
        "prefill index repair requires the accepted split residual state"
        in compiler
    )
    assert (
        "prefill index repair must remain isolated from residual observation"
        in compiler
    )
    assert (
        "readonly DSA_INTERNAL_BASELINE_NPZ="
        "${GLM_GREENFIELD_DSA_INTERNAL_BASELINE_NPZ:-" in runner
    )
    assert (
        "readonly DSA_INTERNAL_BASELINE_SHA="
        "${GLM_GREENFIELD_DSA_INTERNAL_BASELINE_SHA:-" in runner
    )
    assert (
        "readonly DSA_INTERNAL_LAYER0_REFERENCE_NPZ="
        "${GLM_GREENFIELD_DSA_INTERNAL_LAYER0_REFERENCE_NPZ:-" in runner
    )
    assert (
        "readonly DSA_INTERNAL_LAYER0_REFERENCE_SHA="
        "${GLM_GREENFIELD_DSA_INTERNAL_LAYER0_REFERENCE_SHA:-" in runner
    )
    assert "args.prefill_index_repair and args.observe_dsa_internals" not in compiler
    assert (
        "LAYER_RESIDUAL_OBSERVER == 0 && $DSA_INTERNAL_OBSERVER == 0"
        not in runner
    )
    assert '"schema_version": 12' in compiler
    assert 'record["schema_version"] for record in records} != {12}' in runner
    assert "results_db_run_id\": 518" in runner
    assert (
        "a8d370166257622875feafd4d1da3f8d666204a8609baaffef2573b659f6bfee"
        in runner
    )
    assert "_prefill_keyfix" in runner
    assert 'repair["expected_call_count"] != 84' in runner
    assert 'repair["physical_sqrt_count"] != 168' in runner
    assert 'repair["repair_collectives"]' in runner
    assert 'repair["repair_weight_round_count"] != 0' in runner
    assert "results_db_run_id\": 519" in runner
    assert "results_db_run_id\": 520" in runner
    assert "greenfield_layer0_prompt_key_norm_m64_20260809T200559393031635Z" in runner
    assert "prefill_index_weight_split_prerequisite" in runner
    assert '"--dsa-query-exact-association"' in compiler
    assert (
        "readonly DSA_QUERY_EXACT_ASSOCIATION="
        "${GLM_GREENFIELD_DSA_QUERY_EXACT_ASSOCIATION:-0}" in runner
    )
    assert '--dsa-query-exact-association "$dsa_query_exact_association"' in runner
    assert "dsa_observer.input_specs[state_spec_offset + 4]" in compiler
    assert "dsa_observer.input_specs[5]" not in compiler
    assert "results_db_run_id\": 525" in runner
    assert "physical_owner_tuple4_barrier_m1_n1024" in runner
    assert "results_db_run_id\": 526" in runner
    assert "physical_production_fused_q_a_tuple4_exact_m1_n1024" in runner
    assert (
        "greenfield_layer0_physical_lp4_dsa_query_production_exact_"
        "20260810T080508327295662Z" in runner
    )
    assert "DB526 direct remote SUCCESS hash drifted" in runner
    assert "dsa_query_exact_association_production_prerequisite" in runner
    assert "_queryexact" in runner
    assert "DB520 direct remote SUCCESS hash drifted" in runner
    assert "external_stage_local_raw_fp8_to_bf16" in runner
    assert "external_stage_local_bf16_to_fp32" in runner
    assert "prefill_wk_materialization_hlo_contract" in runner


@pytest.mark.parametrize(
    ("capacity", "label"),
    ((2048, "2k"), (8192, "8k")),
)
def test_protected_short_context_label(capacity: int, label: str) -> None:
    assert _protected_short_context_label(capacity) == label


def test_protected_short_context_label_rejects_unsealed_capacity() -> None:
    with pytest.raises(ValueError, match="must be 2048 or 8192"):
        _protected_short_context_label(4096)


def test_protected_8k_runner_pins_paired_oracles() -> None:
    source = PROTECTED_RUNNER.read_text()
    wrapper = PROTECTED_8K_RUNNER.read_text()

    assert "GLM_GREENFIELD_SHORT_DECODER_PROFILE:-2k" in source
    assert "PROMPT_TOKEN_COUNT=8155" in source
    assert "CONTEXT_CAPACITY=8192" in source
    assert (
        "SHORT_CONTEXT_ORACLE_MANIFEST_SHA="
        "e4fbcbdbf0fc8b1969e2f82ee457ab1563db4a8b37d2dea2bc4d1e828a13acf2"
        in source
    )
    assert (
        "SHORT_CONTEXT_DSA_ORACLE_MANIFEST_SHA="
        "f8154c5f79b909efd9ebc14c8e004925482844d05ef28fcf0a4d29bb4a7b26da"
        in source
    )
    assert "export GLM_GREENFIELD_SHORT_DECODER_PROFILE=8k" in wrapper


class _Jax:
    def __init__(self, materialized: np.ndarray) -> None:
        self.materialized = materialized
        self.device_get_calls = 0

    def device_get(self, value: object) -> np.ndarray:
        self.device_get_calls += 1
        return self.materialized


class _Multihost:
    def __init__(self, gathered: np.ndarray) -> None:
        self.gathered = gathered
        self.process_allgather_calls = 0
        self.tiled_values: list[bool] = []

    def process_allgather(self, value: object, *, tiled: bool) -> np.ndarray:
        self.process_allgather_calls += 1
        self.tiled_values.append(tiled)
        return self.gathered


def test_materialize_global_array_uses_device_get_when_fully_addressable() -> None:
    expected = np.arange(12, dtype=np.int32).reshape(4, 3)
    value = SimpleNamespace(is_fully_addressable=True, shape=expected.shape)
    jax = _Jax(expected)
    multihost = _Multihost(np.empty((0,), dtype=np.int32))

    actual = _materialize_global_array(jax, multihost, value)

    np.testing.assert_array_equal(actual, expected)
    assert jax.device_get_calls == 1
    assert multihost.process_allgather_calls == 0
    assert multihost.tiled_values == []


def test_materialize_global_array_gathers_non_addressable_global_array() -> None:
    expected = np.arange(24, dtype=np.int32).reshape(8, 3)
    value = SimpleNamespace(is_fully_addressable=False, shape=expected.shape)
    jax = _Jax(np.empty((0,), dtype=np.int32))
    multihost = _Multihost(expected)

    actual = _materialize_global_array(jax, multihost, value)

    np.testing.assert_array_equal(actual, expected)
    assert jax.device_get_calls == 0
    assert multihost.process_allgather_calls == 1
    assert multihost.tiled_values == [True]


def test_materialize_global_array_fails_closed_on_gather_shape_drift() -> None:
    value = SimpleNamespace(is_fully_addressable=False, shape=(8, 3))
    jax = _Jax(np.empty((0,), dtype=np.int32))
    multihost = _Multihost(np.empty((1, 8, 3), dtype=np.int32))

    with pytest.raises(RuntimeError, match="global array gather changed shape"):
        _materialize_global_array(jax, multihost, value)


def _layer_residual_observation_fixture() -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[tuple[int, ...], ...],
    SimpleNamespace,
]:
    groups = ((0, 1), (2, 3))
    schedule = SimpleNamespace(
        stages=(
            SimpleNamespace(
                assignment=SimpleNamespace(stage_id=0),
                layers=(
                    SimpleNamespace(layer_id=0),
                    SimpleNamespace(layer_id=1),
                ),
            ),
            SimpleNamespace(
                assignment=SimpleNamespace(stage_id=1),
                layers=(
                    SimpleNamespace(layer_id=2),
                    SimpleNamespace(layer_id=3),
                ),
            ),
        )
    )
    canonical = np.asarray(
        np.arange(1, 16, dtype=np.float32).reshape(5, 3),
        dtype=bfloat16,
    )
    observation = np.zeros((4, 5, 3), dtype=bfloat16)
    writers = ((0,), (0,), (0, 1), (1,), (1,))
    for boundary, writer_stages in enumerate(writers):
        for stage_id in writer_stages:
            observation[list(groups[stage_id]), boundary] = canonical[boundary]
    return observation, canonical, groups, schedule


def test_layer_residual_observer_canonicalizes_exact_stage_writers() -> None:
    observation, expected, groups, schedule = (
        _layer_residual_observation_fixture()
    )

    canonical, contract = _canonicalize_layer_residual_observation(
        observation,
        groups=groups,
        schedule=schedule,
        hidden_size=3,
    )

    np.testing.assert_array_equal(canonical, expected)
    assert contract["passed"]
    assert contract["boundary_count"] == 5
    assert contract["dtype"] == "bfloat16"
    assert contract["storage_byte_order"] == "little"
    assert contract["storage_dtype"] == "<u2"
    assert contract["storage_field"] == "residual_bfloat16_bits"
    assert contract["lane_mismatch_boundaries"] == []
    assert contract["writer_mismatch_boundaries"] == []
    assert contract["nonwriter_nonzero_boundaries"] == []
    assert [record["writer_stages"] for record in contract["boundary_records"]] == [
        [0],
        [0],
        [0, 1],
        [1],
        [1],
    ]


def test_layer_residual_bfloat16_storage_is_portable_uint16_bits() -> None:
    values = np.asarray([[0.0, 1.0, -2.5]], dtype=bfloat16)

    bits = _encode_bfloat16_bits(values)

    assert bits.dtype.str == "<u2"
    assert bits.tolist() == [[0x0000, 0x3F80, 0xC020]]
    assert bits.tobytes() == b"\x00\x00\x80\x3f\x20\xc0"
    with pytest.raises(ValueError, match="requires bfloat16"):
        _encode_bfloat16_bits(values.astype(np.float32))


@pytest.mark.parametrize(
    ("mutation", "contract_key", "expected_boundary"),
    (
        ("lane", "lane_mismatch_boundaries", 1),
        ("writer", "writer_mismatch_boundaries", 2),
        ("nonwriter", "nonwriter_nonzero_boundaries", 1),
    ),
)
def test_layer_residual_observer_fails_closed_on_replication_drift(
    mutation: str,
    contract_key: str,
    expected_boundary: int,
) -> None:
    observation, _, groups, schedule = _layer_residual_observation_fixture()
    if mutation == "lane":
        observation[1, 1, 0] += bfloat16(1)
    elif mutation == "writer":
        observation[list(groups[1]), 2, 0] += bfloat16(1)
    else:
        observation[2, 1, 0] = bfloat16(1)

    _, contract = _canonicalize_layer_residual_observation(
        observation,
        groups=groups,
        schedule=schedule,
        hidden_size=3,
    )

    assert not contract["passed"]
    assert contract[contract_key] == [expected_boundary]


def test_layer_residual_observer_rejects_shape_and_dtype_drift() -> None:
    observation, _, groups, schedule = _layer_residual_observation_fixture()
    with pytest.raises(RuntimeError, match="tensor contract drifted"):
        _canonicalize_layer_residual_observation(
            observation[:, :-1],
            groups=groups,
            schedule=schedule,
            hidden_size=3,
        )
    with pytest.raises(RuntimeError, match="tensor contract drifted"):
        _canonicalize_layer_residual_observation(
            observation.astype(np.float32),
            groups=groups,
            schedule=schedule,
            hidden_size=3,
        )


def _dsa_internal_observation_fixture() -> tuple[
    SimpleNamespace,
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
]:
    groups = ((0, 1), (2, 3))
    producers = ((0, 1), (6,))
    normalized_hidden = np.zeros((4, 2, 3), dtype=bfloat16)
    q_a_state = np.zeros((4, 2, 2), dtype=bfloat16)
    query = np.zeros((4, 2, 2, 2), dtype=np.float32)
    head_weights = np.zeros((4, 2, 2), dtype=np.float32)
    current_key = np.zeros((4, 2, 2), dtype=np.float32)
    producer_layer_ids = np.full((4, 2), -1, dtype=np.int32)
    fields = (
        normalized_hidden,
        q_a_state,
        query,
        head_weights,
        current_key,
    )
    for stage, (group, stage_producers) in enumerate(
        zip(groups, producers, strict=True)
    ):
        for slot, producer in enumerate(stage_producers):
            producer_layer_ids[list(group), slot] = producer
            for field_index, value in enumerate(fields, start=1):
                value[list(group), slot] = field_index * 10 + producer
    return (
        SimpleNamespace(
            normalized_hidden=normalized_hidden,
            q_a_state=q_a_state,
            query=query,
            head_weights=head_weights,
            current_key=current_key,
            producer_layer_ids=producer_layer_ids,
        ),
        groups,
        producers,
    )


def test_dsa_internal_observer_canonicalizes_stage_slots() -> None:
    observation, groups, producers = _dsa_internal_observation_fixture()

    canonical, contract = _canonicalize_dsa_internal_observation(
        observation,
        groups=groups,
        stage_producer_layer_ids=producers,
        hidden_size=3,
        q_lora_rank=2,
        num_heads=2,
        head_dim=2,
    )

    assert contract["passed"]
    assert contract["event_count"] == 3
    assert contract["producer_layer_ids"] == [0, 1, 6]
    assert contract["lane_mismatches"] == []
    assert contract["padded_slot_mismatches"] == []
    np.testing.assert_array_equal(
        canonical["producer_layer_ids"], np.asarray([0, 1, 6], np.int32)
    )
    assert canonical["normalized_hidden"].shape == (3, 3)
    assert canonical["normalized_hidden"].dtype.name == "bfloat16"
    assert canonical["q_a_state"].shape == (3, 2)
    assert canonical["query"].shape == (3, 2, 2)
    assert canonical["head_weights"].shape == (3, 2)
    assert canonical["current_key"].shape == (3, 2)


@pytest.mark.parametrize("mutation", ("lane", "padded"))
def test_dsa_internal_observer_fails_closed_on_slot_drift(
    mutation: str,
) -> None:
    observation, groups, producers = _dsa_internal_observation_fixture()
    if mutation == "lane":
        observation.query[1, 0, 0, 0] += 1.0
    else:
        observation.current_key[2, 1, 0] = 1.0

    _, contract = _canonicalize_dsa_internal_observation(
        observation,
        groups=groups,
        stage_producer_layer_ids=producers,
        hidden_size=3,
        q_lora_rank=2,
        num_heads=2,
        head_dim=2,
    )

    assert not contract["passed"]
    assert contract[
        "lane_mismatches" if mutation == "lane" else "padded_slot_mismatches"
    ]


def test_dsa_internal_observer_rejects_shape_and_dtype_drift() -> None:
    observation, groups, producers = _dsa_internal_observation_fixture()
    observation.normalized_hidden = observation.normalized_hidden.astype(
        np.float32
    )
    with pytest.raises(RuntimeError, match="normalized_hidden contract drifted"):
        _canonicalize_dsa_internal_observation(
            observation,
            groups=groups,
            stage_producer_layer_ids=producers,
            hidden_size=3,
            q_lora_rank=2,
            num_heads=2,
            head_dim=2,
        )


def test_completed_step_selected_state_uses_next_position_as_exclusive_bound() -> None:
    metadata = np.asarray(
        [
            [2, 0, 1, -1, 3, 99],
            [1, 2, 0, -1, 3, 99],
        ],
        dtype=np.int32,
    )
    record = _validate_completed_step_selected_states(
        metadata,
        selected_width=4,
        count_index=4,
        next_position=3,
        next_context_length=4,
    )
    assert record == {
        "expected_valid_count": 3,
        "next_context_length": 4,
        "next_position": 3,
        "position_context_aligned": True,
        "rows_valid": [True, True],
    }

    future_position = metadata.copy()
    future_position[0, 0] = 3
    assert not all(
        _validate_completed_step_selected_states(
            future_position,
            selected_width=4,
            count_index=4,
            next_position=3,
            next_context_length=4,
        )["rows_valid"]
    )
    assert not all(
        _validate_completed_step_selected_states(
            metadata,
            selected_width=4,
            count_index=4,
            next_position=3,
            next_context_length=3,
        )["rows_valid"]
    )


def test_raw_token_sequence_contract_is_exact_and_reports_first_drift() -> None:
    expected = np.asarray([4, 8, 15, 16, 23, 42], dtype=np.int32)
    exact = _raw_token_sequence_contract([4, 8, 15], expected)
    assert exact == {
        "compared_token_count": 3,
        "exact_prefix_match": True,
        "expected_token_ids": [4, 8, 15],
        "first_mismatch_index": None,
        "observed_token_ids": [4, 8, 15],
        "oracle_token_count": 6,
    }
    drifted = _raw_token_sequence_contract([4, 7, 15], expected)
    assert not drifted["exact_prefix_match"]
    assert drifted["first_mismatch_index"] == 1
    with pytest.raises(ValueError, match="exceeds"):
        _raw_token_sequence_contract([], expected)
    with pytest.raises(ValueError, match="exceeds"):
        _raw_token_sequence_contract(expected.tolist() + [99], expected)


def test_load_short_context_dsa_oracle_requires_exact_manifest_pin(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    from safetensors.numpy import save_file

    oracle_dir = Path(str(tmp_path))
    tensors = {
        "decode_positions": np.asarray([4], np.int32),
        "producer_layer_ids": np.asarray([0, 2], np.int32),
        "selected_positions": np.asarray(
            [[[0, 1, 2, 3], [3, 2, 1, 0]]], np.int32
        ),
        "selected_scores": np.asarray(
            [[[4, 3, 2, 1], [4, 3, 2, 1]]], np.float32
        ),
        "valid_counts": np.asarray([[4, 4]], np.int32),
    }
    save_file(tensors, oracle_dir / "dsa_events.safetensors")
    manifest = {
        "files": {"tensors": {"filename": "dsa_events.safetensors"}},
        "manifest_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        compile_module,
        "inspect_short_context_dsa_oracle",
        lambda _: manifest,
    )

    loaded_manifest, loaded_tensors = _load_short_context_dsa_oracle(
        oracle_dir,
        expected_manifest_sha256="a" * 64,
    )
    assert loaded_manifest == manifest
    for name, expected in tensors.items():
        np.testing.assert_array_equal(loaded_tensors[name], expected)
    with pytest.raises(RuntimeError, match="protected pin"):
        _load_short_context_dsa_oracle(
            oracle_dir,
            expected_manifest_sha256="b" * 64,
        )


def _dsa_observation_fixture() -> tuple[
    np.ndarray,
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    groups = ((0, 1), (2, 3))
    producers = ((0, 2), (6,))
    expected_positions = np.asarray(
        [[0, 1, 2, -1], [2, 1, 0, -1], [1, 0, 2, -1]],
        np.int32,
    )
    expected_counts = np.asarray([3, 3, 3], np.int32)
    expected_scores = np.asarray(
        [[3, 2, 1, -np.inf], [3, 2, 1, -np.inf], [3, 2, 1, -np.inf]],
        np.float32,
    )
    expected_producers = np.asarray([0, 2, 6], np.int32)
    observation = np.full((4, 2, 10), -1, np.int32)
    event = 0
    for stage, group in enumerate(groups):
        for slot, producer in enumerate(producers[stage]):
            row = np.concatenate(
                (
                    expected_positions[event],
                    expected_scores[event].view(np.int32),
                    np.asarray([expected_counts[event], producer], np.int32),
                )
            )
            observation[list(group), slot] = row
            event += 1
    return (
        observation,
        groups,
        producers,
        expected_positions,
        expected_scores,
        expected_counts,
        expected_producers,
    )


def test_dsa_observer_reconstructs_stage_slots_and_lane_replication() -> None:
    (
        observation,
        groups,
        producers,
        expected_positions,
        expected_scores,
        expected_counts,
        expected_producers,
    ) = _dsa_observation_fixture()
    record = _validate_dsa_observation_step(
        observation,
        groups=groups,
        stage_producer_layer_ids=producers,
        selected_width=4,
        expected_positions=expected_positions,
        expected_scores=expected_scores,
        expected_valid_counts=expected_counts,
        expected_producer_layer_ids=expected_producers,
        decode_position=3,
    )
    assert record["passed"]
    assert record["event_count"] == 3
    assert record["lane_mismatch_stages"] == []
    assert record["padded_slot_mismatches"] == []
    assert record["exact_selected_set_and_tail"]
    assert record["actual_device_score_order_and_ties"]
    assert record["legacy_total_order_match"]
    assert record["legacy_score_bounded_comparison"]["passed"]


def test_dsa_observer_allows_cross_backend_nontie_rank_drift() -> None:
    (
        observation,
        groups,
        producers,
        expected_positions,
        expected_scores,
        expected_counts,
        expected_producers,
    ) = _dsa_observation_fixture()
    drifted = observation.copy()
    bounded_expected_scores = expected_scores.copy()
    bounded_expected_scores[0, 1] = np.float32(2.99)
    # The executing program assigns a different descending score order to the
    # same exact set. Its scores remain canonical and inside the established
    # cross-program bound, so the raw order difference stays diagnostic.
    drifted[0:2, 0, 0:3] = np.asarray([1, 0, 2], np.int32)
    drifted_scores = np.asarray([3.01, 3.0, 1.0], np.float32).view(np.int32)
    drifted[0:2, 0, 4:7] = drifted_scores
    record = _validate_dsa_observation_step(
        drifted,
        groups=groups,
        stage_producer_layer_ids=producers,
        selected_width=4,
        expected_positions=expected_positions,
        expected_scores=bounded_expected_scores,
        expected_valid_counts=expected_counts,
        expected_producer_layer_ids=expected_producers,
        decode_position=3,
    )
    assert record["passed"]
    assert record["exact_selected_set_and_tail"]
    assert record["actual_device_score_order_and_ties"]
    assert not record["legacy_total_order_match"]
    assert record["legacy_order_mismatch_count"] == 2
    assert record["legacy_score_bounded_comparison"]["passed"]


def test_dsa_observer_records_cross_backend_score_bound_drift() -> None:
    (
        observation,
        groups,
        producers,
        expected_positions,
        expected_scores,
        expected_counts,
        expected_producers,
    ) = _dsa_observation_fixture()
    drifted = observation.copy()
    drifted[0:2, 0, 4] = np.asarray([3.2], np.float32).view(np.int32)[0]
    record = _validate_dsa_observation_step(
        drifted,
        groups=groups,
        stage_producer_layer_ids=producers,
        selected_width=4,
        expected_positions=expected_positions,
        expected_scores=expected_scores,
        expected_valid_counts=expected_counts,
        expected_producer_layer_ids=expected_producers,
        decode_position=3,
    )
    assert record["passed"]
    assert record["exact_selected_set_and_tail"]
    assert record["actual_device_score_order_and_ties"]
    assert not record["legacy_score_bounded_comparison"]["passed"]
    assert record["legacy_score_bounded_comparison"]["error"]["max_abs"] > 0.19


def test_dsa_observer_refuses_selected_set_drift() -> None:
    (
        observation,
        groups,
        producers,
        expected_positions,
        expected_scores,
        expected_counts,
        expected_producers,
    ) = _dsa_observation_fixture()
    drifted = observation.copy()
    drifted[0:2, 0, 2] = 3
    record = _validate_dsa_observation_step(
        drifted,
        groups=groups,
        stage_producer_layer_ids=producers,
        selected_width=4,
        expected_positions=expected_positions,
        expected_scores=expected_scores,
        expected_valid_counts=expected_counts,
        expected_producer_layer_ids=expected_producers,
        decode_position=3,
    )
    assert not record["passed"]
    assert not record["exact_selected_set_and_tail"]
    assert record["selected_set_mismatches"][0]["expected_only_first"] == [2]
    assert record["selected_set_mismatches"][0]["observed_only_first"] == [3]


def test_dsa_observer_refuses_lane_order_producer_and_padding_drift() -> None:
    (
        observation,
        groups,
        producers,
        expected_positions,
        expected_scores,
        expected_counts,
        expected_producers,
    ) = _dsa_observation_fixture()
    drifted = observation.copy()
    # Same selected set, but equal executing scores are in wrong position order.
    drifted[0:2, 0, 0:2] = drifted[0:2, 0, 1::-1]
    tied = np.asarray([3.0, 3.0], np.float32).view(np.int32)
    drifted[0:2, 0, 4:6] = tied
    drifted[0:2, 0, 7] = np.asarray([0.0], np.float32).view(np.int32)[0]
    # One replicated lane disagrees, one producer drifts, and dead padding is live.
    drifted[1, 1, 0] = 99
    drifted[2:4, 0, 9] = 7
    drifted[2:4, 1, 0] = 0
    record = _validate_dsa_observation_step(
        drifted,
        groups=groups,
        stage_producer_layer_ids=producers,
        selected_width=4,
        expected_positions=expected_positions,
        expected_scores=expected_scores,
        expected_valid_counts=expected_counts,
        expected_producer_layer_ids=expected_producers,
        decode_position=3,
    )
    assert not record["passed"]
    assert record["legacy_order_mismatch_count"] == 2
    assert record["first_legacy_order_mismatch"]["selected_offset"] == 0
    assert not record["actual_device_score_order_and_ties"]
    assert record["score_contract_mismatches"][0][
        "first_noncanonical_offset"
    ] == 0
    assert record["tail_mismatches"] == [
        {
            "event_index": 0,
            "position_tail_mismatch_count": 0,
            "producer_layer_id": 0,
            "score_tail_mismatch_count": 1,
        }
    ]
    assert record["lane_mismatch_stages"] == [0]
    assert record["producer_mismatches"] == [
        {"event_index": 2, "expected": 6, "observed": 7}
    ]
    assert record["padded_slot_mismatches"] == [{"slot": 1, "stage": 1}]


def _token_observation_fixture() -> tuple[
    np.ndarray, tuple[tuple[int, ...], ...]
]:
    groups = ((0, 1), (2, 3))
    candidate_ids = np.asarray([3, 1, 4, 2], np.int32)
    candidate_scores = np.asarray([10.0, 9.0, 9.0, 8.0], np.float32)
    row = np.concatenate((candidate_ids, candidate_scores.view(np.int32)))
    observation = np.full((4, 8), -1, np.int32)
    observation[list(groups[-1])] = row
    return observation, groups


def test_token_observer_validates_rank_margin_ties_and_inactive_lanes() -> None:
    observation, groups = _token_observation_fixture()
    record = _validate_token_observation_step(
        observation,
        groups=groups,
        candidate_width=4,
        expected_token_id=4,
        observed_token_id=3,
        vocab_size=8,
    )
    assert record["passed"]
    assert record["candidate_ids"] == [3, 1, 4, 2]
    assert record["expected_offset"] == 2
    assert record["expected_rank"] == 3
    assert record["expected_score"] == 9.0
    assert record["top1_top2_margin"] == 1.0
    assert record["top1_expected_margin"] == 1.0
    assert record["lane_replication"]
    assert record["inactive_rows_are_sentinel"]
    assert record["order_and_ties_valid"]
    assert record["winner_matches_output"]


def test_token_observer_refuses_noncanonical_or_corrupt_candidates() -> None:
    observation, groups = _token_observation_fixture()

    wrong_tie_order = observation.copy()
    wrong_tie_order[2:4, [1, 2]] = wrong_tie_order[2:4, [2, 1]]
    record = _validate_token_observation_step(
        wrong_tie_order,
        groups=groups,
        candidate_width=4,
        expected_token_id=4,
        observed_token_id=3,
        vocab_size=8,
    )
    assert not record["passed"]
    assert not record["order_and_ties_valid"]

    corrupt = observation.copy()
    corrupt[0, 0] = 0
    corrupt[3, 0] = 7
    record = _validate_token_observation_step(
        corrupt,
        groups=groups,
        candidate_width=4,
        expected_token_id=4,
        observed_token_id=1,
        vocab_size=8,
    )
    assert not record["passed"]
    assert not record["inactive_rows_are_sentinel"]
    assert not record["lane_replication"]
    assert not record["winner_matches_output"]


def test_observer_hlo_isolation_requires_no_alias_callback_or_collective_drift(
) -> None:
    contract = {
        "all_reduce_arity_counts": {"1": 3},
        "all_reduce_component_count": 3,
        "all_reduce_result_shape_counts": {
            "bf16[4]": 1,
            "bf16[8]": 1,
            "s32[4]": 1,
        },
        "collective_count": 5,
        "collective_counts": {
            "all-gather": 1,
            "all-reduce": 3,
            "collective-permute": 1,
        },
        "complete_token_collective_contract": {
            "score_exchange": [{"result_shapes": ["bf16[4]"]}],
            "token_id_exchange": [{"result_shapes": ["s32[4]"]}],
        },
        "passed": True,
    }
    hlo = "HloModule observer, is_scheduled=true\nENTRY main {}\n"
    exact = _observer_hlo_isolation_contract(
        hlo,
        production_contract=contract,
        observer_contract=dict(contract),
    )
    assert exact["passed"]
    assert exact["donate_argnums"] == []
    assert exact["non_token_result_shapes_match"]

    widened = {
        **contract,
        "all_reduce_result_shape_counts": {
            "bf16[8]": 1,
            "bf16[64]": 1,
            "s32[64]": 1,
        },
        "complete_token_collective_contract": {
            "score_exchange": [{"result_shapes": ["bf16[64]"]}],
            "token_id_exchange": [{"result_shapes": ["s32[64]"]}],
        },
    }
    wider_token_exchange = _observer_hlo_isolation_contract(
        hlo,
        production_contract=contract,
        observer_contract=widened,
    )
    assert wider_token_exchange["passed"]
    assert wider_token_exchange["token_exchange_shape_difference_allowed"]

    widened_non_token_drift = {
        **widened,
        "all_reduce_result_shape_counts": {
            "bf16[16]": 1,
            "bf16[64]": 1,
            "s32[64]": 1,
        },
    }
    non_token_drift = _observer_hlo_isolation_contract(
        hlo,
        production_contract=contract,
        observer_contract=widened_non_token_drift,
    )
    assert not non_token_drift["passed"]
    assert not non_token_drift["non_token_result_shapes_match"]

    aliased = _observer_hlo_isolation_contract(
        "HloModule observer, input_output_alias={ {0}: (1, {}, may-alias) }\n",
        production_contract=contract,
        observer_contract=dict(contract),
    )
    assert not aliased["passed"]
    callback = _observer_hlo_isolation_contract(
        hlo + "outside_compilation\n",
        production_contract=contract,
        observer_contract=dict(contract),
    )
    assert not callback["passed"]
    drifted = dict(contract)
    drifted["collective_count"] = 4
    collective = _observer_hlo_isolation_contract(
        hlo,
        production_contract=contract,
        observer_contract=drifted,
    )
    assert not collective["passed"]
