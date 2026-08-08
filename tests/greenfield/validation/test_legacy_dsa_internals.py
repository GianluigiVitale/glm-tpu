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
    config: subject.LegacyDsaInternalComparisonConfig,
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
    wrapper = REPO_ROOT / (
        "scripts/greenfield/run_capture_legacy_layer0_dsa_internals.sh"
    )
    shared = REPO_ROOT / "scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
    wrapper_source = wrapper.read_text()
    shared_source = shared.read_text()
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
        "compare_short_context_dsa_oracles",
        "compare_legacy_layer0_dsa_internals.py",
        "strict_census post",
        "dsa_event_tensors_exact",
        "topology_sharded_live_row_owner",
        "INTERNAL_OWNER",
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
    ):
        assert required in source
    for forbidden in ("launch_glm_32chip.sh", "glm_longctx.py"):
        assert forbidden not in source
    assert 'local label=$1 out=' not in source
    assert 'local label=$1\n  local out=' in source
