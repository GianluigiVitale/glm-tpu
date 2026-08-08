from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from glm_tpu.greenfield.validation import legacy_dsa_internals as subject


REPO_ROOT = Path(__file__).resolve().parents[3]


def _config(tmp_path: Path) -> subject.LegacyDsaInternalComparisonConfig:
    return subject.LegacyDsaInternalComparisonConfig(
        source_dump_dir=tmp_path / "source",
        layer0_input_dir=tmp_path / "input",
        distributed_q_a_norm_dir=tmp_path / "q_a",
        output_dir=tmp_path / "output",
        expected_run_tag="protected-internal-test",
        expected_greenfield_code_hash="1" * 40,
        expected_legacy_code_hash="2" * 40,
        expected_oracle_pin="3" * 40,
        expected_input_manifest_sha256="4" * 64,
        expected_q_a_manifest_sha256="5" * 64,
        expected_q_a_code_hash="6" * 40,
    )


def _arrays() -> dict[str, np.ndarray]:
    return {
        "normalized_hidden": np.zeros((6144,), dtype=np.uint16),
        "q_a_state": np.zeros((2048,), dtype=np.uint16),
        "query": np.arange(32 * 128, dtype=np.float32).reshape(32, 128),
        "head_weights": np.arange(32, dtype=np.float32),
        "current_key": np.arange(128, dtype=np.float32),
    }


def _write_process_files(
    config: (
        subject.LegacyDsaInternalCaptureConfig
        | subject.LegacyDsaInternalComparisonConfig
    ),
    *,
    drift_process: int | None = None,
) -> dict[str, np.ndarray]:
    values = _arrays()
    config.source_dump_dir.mkdir(parents=True)
    safe_layer = config.expected_layer_name.replace(".", "_")
    for process_index in config.expected_capture_process_indices:
        process_values = {name: value.copy() for name, value in values.items()}
        if process_index == drift_process:
            process_values["query"][0, 0] += np.float32(1)
        path = config.source_dump_dir / (
            f"internal.{safe_layer}.position{config.expected_position}."
            f"proc{process_index}.npz"
        )
        np.savez(
            path,
            artifact_kind=np.asarray(subject.ARTIFACT_KIND),
            format_version=np.asarray(1, dtype=np.int64),
            process_index=np.asarray(process_index, dtype=np.int64),
            process_count=np.asarray(config.expected_process_count, dtype=np.int64),
            layer_name=np.asarray(config.expected_layer_name),
            position=np.asarray(config.expected_position, dtype=np.int32),
            source_row=np.asarray(0, dtype=np.int32),
            run_tag=np.asarray(config.expected_run_tag),
            code_hash=np.asarray(config.expected_legacy_code_hash),
            oracle_pin=np.asarray(config.expected_oracle_pin),
            model_id=np.asarray(config.expected_model_id),
            **process_values,
            **{
                f"{name}__dtype": np.asarray(contract[1])
                for name, contract in subject.FIELD_CONTRACT.items()
            },
        )
    return values


def test_owner_capture_writes_bounded_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    values = _write_process_files(config)
    monkeypatch.setattr(
        subject,
        "_reconstruct_greenfield",
        lambda _config: (
            {name: value.copy() for name, value in values.items()},
            {
                "backend": "cpu",
                "device_count": 1,
                "device_kind": ["cpu"],
                "input_manifest_sha256": config.expected_input_manifest_sha256,
                "q_a_manifest_sha256": config.expected_q_a_manifest_sha256,
                "numerical_contract": {},
            },
        ),
    )
    result = subject.compare_legacy_dsa_internals(config)
    assert result["all_fields_elementwise_exact"] is True
    assert result["first_divergent_field"] is None
    assert len(result["process_files"]) == 1
    assert result["capture_layout"] == "topology_sharded_live_row_owner"
    assert result["capture_process_indices"] == [0]
    assert (config.output_dir / "comparison.json").is_file()
    assert (config.output_dir / "internals.npz").is_file()
    with pytest.raises(FileExistsError, match="append-only"):
        subject.compare_legacy_dsa_internals(config)


def test_generic_owner_capture_is_sealed_without_layer0_reconstruction(
    tmp_path: Path,
) -> None:
    config = subject.LegacyDsaInternalCaptureConfig(
        source_dump_dir=tmp_path / "source",
        output_dir=tmp_path / "output",
        expected_run_tag="protected-layer1-internal-test",
        expected_legacy_code_hash="2" * 40,
        expected_oracle_pin="3" * 40,
        expected_layer_name="model.layers.1.self_attn.attn",
    )
    expected = _write_process_files(config)
    result = subject.inspect_legacy_dsa_internal_capture(config)
    assert result["artifact_kind"] == subject.CAPTURE_KIND
    assert result["layer_name"] == config.expected_layer_name
    assert result["capture_process_indices"] == [0]
    assert result["fields"]["query"]["sha256"] == subject._array_sha256(
        expected["query"]
    )
    assert (config.output_dir / "capture.json").is_file()
    assert (config.output_dir / "internals.npz").is_file()
    with pytest.raises(FileExistsError, match="append-only"):
        subject.inspect_legacy_dsa_internal_capture(config)


def test_sealed_observer_comparison_aligns_the_requested_producer(
    tmp_path: Path,
) -> None:
    capture_config = subject.LegacyDsaInternalCaptureConfig(
        source_dump_dir=tmp_path / "source",
        output_dir=tmp_path / "capture",
        expected_run_tag="protected-layer1-internal-test",
        expected_legacy_code_hash="2" * 40,
        expected_oracle_pin="3" * 40,
        expected_layer_name="model.layers.1.self_attn.attn",
    )
    expected = _write_process_files(capture_config)
    subject.inspect_legacy_dsa_internal_capture(capture_config)

    event_count = len(subject.FULL_DSA_PRODUCER_LAYER_IDS)
    event_index = subject.FULL_DSA_PRODUCER_LAYER_IDS.index(1)
    observer_path = tmp_path / "observer.npz"
    observed = {
        "normalized_hidden_bfloat16_bits": np.zeros(
            (event_count, 6144), dtype=np.uint16
        ),
        "q_a_state_bfloat16_bits": np.zeros(
            (event_count, 2048), dtype=np.uint16
        ),
        "query": np.zeros((event_count, 32, 128), dtype=np.float32),
        "head_weights": np.zeros((event_count, 32), dtype=np.float32),
        "current_key": np.zeros((event_count, 128), dtype=np.float32),
    }
    observed["normalized_hidden_bfloat16_bits"][event_index] = expected[
        "normalized_hidden"
    ]
    observed["q_a_state_bfloat16_bits"][event_index] = expected["q_a_state"]
    for name in ("query", "head_weights", "current_key"):
        observed[name][event_index] = expected[name]
    np.savez(
        observer_path,
        **observed,
        producer_layer_ids=np.asarray(
            subject.FULL_DSA_PRODUCER_LAYER_IDS, dtype=np.int32
        ),
        decode_position=np.asarray([8155], dtype=np.int32),
    )
    config = subject.AcceptedGreenfieldDsaInternalComparisonConfig(
        accepted_capture_dir=capture_config.output_dir,
        greenfield_observation_path=observer_path,
        output_dir=tmp_path / "comparison",
        expected_capture_manifest_sha256=subject._file_sha256(
            capture_config.output_dir / "capture.json"
        ),
        expected_greenfield_observation_sha256=subject._file_sha256(
            observer_path
        ),
        expected_greenfield_code_hash="1" * 40,
        expected_legacy_code_hash=capture_config.expected_legacy_code_hash,
        expected_layer_id=1,
    )
    result = subject.compare_accepted_greenfield_dsa_internal_observation(
        config
    )
    assert result["all_fields_elementwise_exact"] is True
    assert result["event_index"] == event_index
    assert result["first_divergent_field"] is None
    assert (config.output_dir / "comparison.json").is_file()
    assert (config.output_dir / "internals.npz").is_file()
    with pytest.raises(FileExistsError, match="append-only"):
        subject.compare_accepted_greenfield_dsa_internal_observation(config)


def test_observer_comparison_rejects_nonproducer_layer(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a full DSA producer"):
        subject.AcceptedGreenfieldDsaInternalComparisonConfig(
            accepted_capture_dir=tmp_path / "capture",
            greenfield_observation_path=tmp_path / "observer.npz",
            output_dir=tmp_path / "comparison",
            expected_capture_manifest_sha256="1" * 64,
            expected_greenfield_observation_sha256="2" * 64,
            expected_greenfield_code_hash="3" * 40,
            expected_legacy_code_hash="4" * 40,
            expected_layer_id=3,
        )


def test_capture_refuses_cross_process_state_drift(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path), expected_capture_process_indices=(0, 1)
    )
    _write_process_files(config, drift_process=1)
    with pytest.raises(ValueError, match="replicated legacy query disagrees"):
        subject._load_legacy_capture(config)


def test_capture_refuses_wrong_owner_process(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path), expected_capture_process_indices=(1,)
    )
    _write_process_files(config)
    config = replace(config, expected_capture_process_indices=(0,))
    with pytest.raises(ValueError, match="owner-process coverage"):
        subject._load_legacy_capture(config)


def test_capture_refuses_error_sentinel(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_process_files(config)
    (config.source_dump_dir / "internal.INTERNAL.ERROR.proc0.txt").write_text(
        "failed\n"
    )
    with pytest.raises(ValueError, match="error sentinel"):
        subject._load_legacy_capture(config)


def test_protected_wrapper_reuses_short_dsa_oracle_stack() -> None:
    layer0_wrapper = REPO_ROOT / (
        "scripts/greenfield/run_capture_legacy_layer0_dsa_internals.sh"
    )
    wrapper = REPO_ROOT / "scripts/greenfield/run_capture_legacy_dsa_internals.sh"
    shared = REPO_ROOT / "scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
    layer0_source = layer0_wrapper.read_text()
    wrapper_source = wrapper.read_text()
    shared_source = shared.read_text()
    assert "GLM_GREENFIELD_DSA_INTERNALS_LAYER_ID=0" in layer0_source
    assert "run_capture_legacy_dsa_internals.sh" in layer0_source
    for required in (
        "GLM_GREENFIELD_DSA_INTERNALS_CAPTURE=1",
        "GLM_GREENFIELD_SHORT_DSA_ORACLE_PROFILE=8k",
        "run_capture_short_context_dsa_oracle.sh",
    ):
        assert required in wrapper_source
    for required in (
        "OBSERVER_COMMIT_DISTANCE=2",
        "83ff4a3576602ca844ea090550139a2ff00b0bb1",
        "GLM_DSA_DUMP_INTERNALS_LAYER",
        "GLM_GREENFIELD_DSA_INTERNALS_LAYER_ID",
        "compare_short_context_dsa_oracles",
        "compare_legacy_layer0_dsa_internals.py",
        "inspect_legacy_dsa_internals.py",
        "strict_census post",
        "dsa_event_tensors_exact",
        "topology_sharded_live_row_owner",
        "INTERNAL_OWNER",
        "TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1",
        "TPU_PROCESS_BOUNDS=1,1,1",
        "TPU_VISIBLE_DEVICES=0,1,2,3",
        "timeout --signal=TERM --kill-after=60 1800",
    ):
        assert required in shared_source


def test_recovery_reuses_source_without_reloading_model() -> None:
    recovery = REPO_ROOT / (
        "scripts/greenfield/recover_legacy_layer0_dsa_internals.sh"
    )
    source = recovery.read_text()
    for required in (
        "greenfield_legacy_layer0_dsa_internals_20260808T015254078767454Z",
        "SOURCE_GREENFIELD_PIN=46fd672220fb4d974dcf49730f362c77d6e38dfe",
        "EXPECTED_DUMP_COUNT=483",
        "SOURCE_OWNER",
        "capture_short_context_dsa_oracle.py",
        "compare_short_context_dsa_oracles",
        "compare_legacy_layer0_dsa_internals.py",
        "strict_census post",
        "remote_objects.json",
        "TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1",
        "TPU_PROCESS_BOUNDS=1,1,1",
        "TPU_VISIBLE_DEVICES=0,1,2,3",
        "timeout --signal=TERM --kill-after=60 1800",
        "GLM_GREENFIELD_DSA_INTERNALS_RECOVERED_SOURCE_DIR",
        'cp -al "$RECOVERED_SOURCE_DIR/." "$SOURCE_DIR/"',
    ):
        assert required in source
    for forbidden in ("launch_glm_32chip.sh", "glm_longctx.py"):
        assert forbidden not in source
    assert 'local label=$1 out=' not in source
    assert 'local label=$1\n  local out=' in source
