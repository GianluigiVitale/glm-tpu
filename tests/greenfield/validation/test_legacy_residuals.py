from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from glm_tpu.greenfield.validation.legacy_residuals import (
    LegacyResidualComparisonConfig,
    compare_legacy_residuals,
)


_CODE_HASH = "a" * 40
_ORACLE_PIN = "b" * 40
_MODEL_ID = "zai-org/GLM-5.2-FP8"
_RUN_TAG = "legacy-residual-unit"


def _bits(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=ml_dtypes.bfloat16).view(np.uint16).astype(
        np.dtype("<u2"), copy=False)


def _source_file(
    root: Path,
    *,
    process_index: int,
    boundary_start: int,
    boundary_stop: int,
    bits: np.ndarray,
) -> Path:
    path = root / f"w{process_index}" / (
        f"boundaries.position7.proc{process_index}.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        artifact_kind=np.asarray("glm52_legacy_layer_residual_shards"),
        boundary_count=np.asarray(3, dtype=np.int32),
        code_hash=np.asarray(_CODE_HASH),
        decode_rows=np.asarray(4, dtype=np.int32),
        entry_count=np.asarray(1, dtype=np.int32),
        format_version=np.asarray(1, dtype=np.int32),
        global_shape=np.asarray([3, 4, 5], dtype=np.int64),
        global_sharding=np.asarray("unit"),
        hidden_size=np.asarray(5, dtype=np.int32),
        model_id=np.asarray(_MODEL_ID),
        oracle_pin=np.asarray(_ORACLE_PIN),
        position=np.asarray(7, dtype=np.int32),
        process_count=np.asarray(2, dtype=np.int32),
        process_index=np.asarray(process_index, dtype=np.int32),
        run_tag=np.asarray(_RUN_TAG),
        storage_byte_order=np.asarray("little"),
        storage_dtype=np.asarray("<u2"),
        target_token_row=np.asarray(1, dtype=np.int32),
        value_dtype=np.asarray("bfloat16"),
        entry_0000_boundary_start=np.asarray(boundary_start, dtype=np.int32),
        entry_0000_boundary_stop=np.asarray(boundary_stop, dtype=np.int32),
        entry_0000_device=np.asarray(f"cpu:{process_index}"),
        entry_0000_hidden_start=np.asarray(0, dtype=np.int32),
        entry_0000_hidden_stop=np.asarray(5, dtype=np.int32),
        entry_0000_source_index=np.asarray("unit"),
        entry_0000_bfloat16_bits=np.asarray(bits, dtype=np.dtype("<u2")),
    )
    return path


def _fixture(tmp_path: Path) -> tuple[LegacyResidualComparisonConfig, np.ndarray]:
    legacy_values = np.arange(15, dtype=np.float32).reshape(3, 5)
    legacy_bits = _bits(legacy_values)
    source = tmp_path / "source"
    _source_file(
        source,
        process_index=0,
        boundary_start=0,
        boundary_stop=2,
        bits=legacy_bits[:2],
    )
    _source_file(
        source,
        process_index=1,
        boundary_start=2,
        boundary_stop=3,
        bits=legacy_bits[2:],
    )

    greenfield_bits = legacy_bits.copy()
    greenfield_bits[1, 3] = _bits(np.asarray([99.0], dtype=np.float32))[0]
    greenfield_npz = tmp_path / "greenfield.npz"
    # Exercise compatibility with the first protected artifact's void16 field.
    np.savez_compressed(
        greenfield_npz,
        boundary_layer_ids=np.arange(3, dtype=np.int32),
        decode_position=np.asarray([7], dtype=np.int32),
        residuals=greenfield_bits.view("V2"),
    )
    greenfield_sha = sha256(greenfield_bits.tobytes(order="C")).hexdigest()
    greenfield_contract = tmp_path / "greenfield_contract.json"
    greenfield_contract.write_text(
        json.dumps({
            "boundary_count": 3,
            "canonical_sha256": greenfield_sha,
            "decode_position": 7,
            "dtype": "bfloat16",
            "hidden_size": 5,
            "passed": True,
            "teacher_forced": True,
        }))
    config = LegacyResidualComparisonConfig(
        source_dump_dir=source,
        greenfield_npz=greenfield_npz,
        greenfield_contract=greenfield_contract,
        output_dir=tmp_path / "output",
        expected_position=7,
        expected_run_tag=_RUN_TAG,
        expected_legacy_code_hash=_CODE_HASH,
        expected_oracle_pin=_ORACLE_PIN,
        expected_model_id=_MODEL_ID,
        expected_process_count=2,
        expected_boundary_count=3,
        expected_decode_rows=4,
        expected_hidden_size=5,
    )
    return config, legacy_bits


def test_reconstructs_exact_bits_and_finds_first_boundary(tmp_path: Path) -> None:
    config, legacy_bits = _fixture(tmp_path)
    comparison = compare_legacy_residuals(config)
    assert comparison["first_divergent_boundary"] == 1
    assert comparison["divergent_boundaries"] == [1]
    assert comparison["boundary_records"][1]["differing_elements"] == 1
    assert comparison["boundary_records"][1][
        "first_differing_hidden_index"] == 3
    assert comparison["legacy"]["coverage_min"] == 1
    assert comparison["legacy"]["canonical_sha256"] == sha256(
        legacy_bits.tobytes(order="C")).hexdigest()
    with np.load(
            config.output_dir / "legacy_position_boundaries.npz",
            allow_pickle=False) as payload:
        assert payload["residual_bfloat16_bits"].dtype == np.dtype("<u2")
        np.testing.assert_array_equal(
            payload["residual_bfloat16_bits"], legacy_bits)
    with pytest.raises(FileExistsError, match="append-only"):
        compare_legacy_residuals(config)


def test_refuses_uncovered_legacy_values(tmp_path: Path) -> None:
    config, _ = _fixture(tmp_path)
    source = next((config.source_dump_dir / "w1").glob("*.npz"))
    source.unlink()
    # Replace process 1 with a duplicate of boundary 0, leaving boundary 2 open.
    _source_file(
        config.source_dump_dir,
        process_index=1,
        boundary_start=0,
        boundary_stop=1,
        bits=_bits(np.arange(5, dtype=np.float32).reshape(1, 5)),
    )
    with pytest.raises(ValueError, match="uncovered"):
        compare_legacy_residuals(config)


def test_refuses_greenfield_contract_hash_drift(tmp_path: Path) -> None:
    config, _ = _fixture(tmp_path)
    contract = json.loads(config.greenfield_contract.read_text())
    contract["canonical_sha256"] = "0" * 64
    config.greenfield_contract.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="canonical_sha256"):
        compare_legacy_residuals(config)
