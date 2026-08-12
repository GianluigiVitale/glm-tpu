from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from glm_tpu.greenfield.validation.attention_output_operand import (
    AcceptedAttentionOutputCaptureConfig,
    AttentionOutputComparisonConfig,
    MAIN_ROPE_TABLE_SHA256,
    capture_accepted_attention_output_operand,
    compare_attention_output_operands,
)


LEGACY_HASH = "1" * 40
ORACLE_HASH = "2" * 40
GREENFIELD_HASH = "3" * 40
ACCEPTED_TAG = "accepted-attention-output-test"
GREENFIELD_TAG = "greenfield-ingredients-test"
LAYER_NAME = "model.layers.0.self_attn.attn"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _bits(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.asarray(values, dtype=ml_dtypes.bfloat16)
    ).view(np.uint16)


def _accepted_values() -> np.ndarray:
    return np.linspace(-1.0, 1.0, 16_384, dtype=np.float32)


def _write_source(path: Path, values: np.ndarray, **overrides: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: dict[str, np.ndarray] = {
        "artifact_kind": np.asarray("glm52_legacy_attention_output_operand"),
        "format_version": np.asarray(1, dtype=np.int64),
        "capture_mode": np.asarray("attention_output"),
        "process_index": np.asarray(0, dtype=np.int64),
        "process_count": np.asarray(8, dtype=np.int64),
        "layer_name": np.asarray(LAYER_NAME),
        "position": np.asarray(8155, dtype=np.int32),
        "source_row": np.asarray(0, dtype=np.int32),
        "run_tag": np.asarray(ACCEPTED_TAG),
        "code_hash": np.asarray(LEGACY_HASH),
        "oracle_pin": np.asarray(ORACLE_HASH),
        "model_id": np.asarray("zai-org/GLM-5.2-FP8"),
        "attention_output": _bits(values),
        "attention_output__dtype": np.asarray("bfloat16"),
    }
    fields.update({name: np.asarray(value) for name, value in overrides.items()})
    np.savez(path, **fields)


def _capture(tmp_path: Path, values: np.ndarray | None = None) -> Path:
    source = tmp_path / "source"
    safe_layer = LAYER_NAME.replace(".", "_")
    _write_source(
        source / f"internals.{safe_layer}.position8155.proc0.npz",
        _accepted_values() if values is None else values,
    )
    output = tmp_path / "accepted"
    capture_accepted_attention_output_operand(
        AcceptedAttentionOutputCaptureConfig(
            source_dump_dir=source,
            output_dir=output,
            expected_run_tag=ACCEPTED_TAG,
            expected_legacy_code_hash=LEGACY_HASH,
            expected_oracle_pin=ORACLE_HASH,
        )
    )
    return output


def _write_ingredients(root: Path, owner_bits: np.ndarray) -> tuple[str, str]:
    root.mkdir(parents=True)
    tensor_path = root / "position_8155_ingredients.npz"
    np.savez(
        tensor_path,
        attention_output_input_bfloat16_bits=owner_bits,
    )
    array_sha = sha256(
        np.ascontiguousarray(owner_bits).tobytes(order="C")
    ).hexdigest()
    contract = {
        "active_rows": [0, 1, 2, 3],
        "arrays": {
            "attention_output_input_bfloat16_bits": {
                "dtype": "uint16",
                "sha256": array_sha,
                "shape": [4, 4096],
            }
        },
        "code_hash": GREENFIELD_HASH,
        "decode_position": 8155,
        "hlo_contract": {
            "main_rope_table_contract": {
                "named_table_parameter_count": 1,
                "passed": True,
                "table_parameter_count": 1,
            }
        },
        "main_rope_table_enabled": True,
        "main_rope_table_sha256": MAIN_ROPE_TABLE_SHA256,
        "passed": True,
        "run_tag": GREENFIELD_TAG,
        "source_state": "post_teacher_forced_prefill",
    }
    contract_path = root / "contract.json"
    contract_path.write_text(json.dumps(contract, sort_keys=True) + "\n")
    return _file_sha256(contract_path), _file_sha256(tensor_path)


def _comparison_config(
    tmp_path: Path,
    *,
    accepted: Path,
    ingredients: Path,
    contract_sha: str,
    tensor_sha: str,
    output_name: str = "comparison",
) -> AttentionOutputComparisonConfig:
    return AttentionOutputComparisonConfig(
        accepted_capture_dir=accepted,
        greenfield_ingredients_dir=ingredients,
        output_dir=tmp_path / output_name,
        expected_accepted_capture_file_sha256=_file_sha256(
            accepted / "capture.json"
        ),
        expected_accepted_run_tag=ACCEPTED_TAG,
        expected_ingredients_contract_sha256=contract_sha,
        expected_ingredients_tensor_sha256=tensor_sha,
        expected_greenfield_code_hash=GREENFIELD_HASH,
        expected_greenfield_run_tag=GREENFIELD_TAG,
        expected_legacy_code_hash=LEGACY_HASH,
        expected_oracle_pin=ORACLE_HASH,
    )


def test_capture_seals_exact_bfloat16_row_and_provenance(tmp_path: Path) -> None:
    output = _capture(tmp_path)
    manifest = json.loads((output / "capture.json").read_text())
    assert manifest["capture_layout"] == "logical_head_order_live_row"
    assert manifest["capture_process_indices"] == [0]
    assert manifest["diagnostic_only"] is True
    assert manifest["performance_claim"] is False
    with np.load(output / "attention_output.npz", allow_pickle=False) as data:
        observed = data["attention_output_bfloat16_bits"]
    assert observed.shape == (16_384,)
    assert observed.dtype == np.dtype(np.uint16)


def test_capture_refuses_wrong_provenance_and_nonfinite_data(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bad-source"
    safe_layer = LAYER_NAME.replace(".", "_")
    path = source / f"internals.{safe_layer}.position8155.proc0.npz"
    _write_source(path, _accepted_values(), run_tag="wrong")
    config = AcceptedAttentionOutputCaptureConfig(
        source_dump_dir=source,
        output_dir=tmp_path / "bad-provenance",
        expected_run_tag=ACCEPTED_TAG,
        expected_legacy_code_hash=LEGACY_HASH,
        expected_oracle_pin=ORACLE_HASH,
    )
    with pytest.raises(ValueError, match="run_tag"):
        capture_accepted_attention_output_operand(config)

    _write_source(path, np.full(16_384, np.inf, dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        capture_accepted_attention_output_operand(
            AcceptedAttentionOutputCaptureConfig(
                source_dump_dir=source,
                output_dir=tmp_path / "bad-finite",
                expected_run_tag=ACCEPTED_TAG,
                expected_legacy_code_hash=LEGACY_HASH,
                expected_oracle_pin=ORACLE_HASH,
            )
        )


def test_comparison_classifies_exact_operand_as_projection_partial(
    tmp_path: Path,
) -> None:
    accepted = _capture(tmp_path)
    accepted_bits = _bits(_accepted_values())
    ingredients = tmp_path / "ingredients"
    contract_sha, tensor_sha = _write_ingredients(
        ingredients, accepted_bits.reshape(4, 4096)
    )
    result = compare_attention_output_operands(
        _comparison_config(
            tmp_path,
            accepted=accepted,
            ingredients=ingredients,
            contract_sha=contract_sha,
            tensor_sha=tensor_sha,
        )
    )
    assert result["elementwise_exact"] is True
    assert result["mismatch_count"] == 0
    assert result["classification"] == "projection_partial_arithmetic"


def test_comparison_classifies_mismatch_and_refuses_hash_drift(
    tmp_path: Path,
) -> None:
    accepted = _capture(tmp_path)
    owner_bits = _bits(_accepted_values()).reshape(4, 4096).copy()
    owner_bits[2, 123] = _bits(np.asarray([0.5], dtype=np.float32))[0]
    ingredients = tmp_path / "ingredients"
    contract_sha, tensor_sha = _write_ingredients(ingredients, owner_bits)
    result = compare_attention_output_operands(
        _comparison_config(
            tmp_path,
            accepted=accepted,
            ingredients=ingredients,
            contract_sha=contract_sha,
            tensor_sha=tensor_sha,
        )
    )
    assert result["elementwise_exact"] is False
    assert result["mismatch_count"] == 1
    assert result["first_mismatch_index"] == 2 * 4096 + 123
    assert result["classification"] == (
        "attention_arithmetic_before_output_projection"
    )

    bad = _comparison_config(
        tmp_path,
        accepted=accepted,
        ingredients=ingredients,
        contract_sha="0" * 64,
        tensor_sha=tensor_sha,
        output_name="bad-hash",
    )
    with pytest.raises(ValueError, match="contract hash drifted"):
        compare_attention_output_operands(bad)


def test_protected_wrappers_pin_isolated_capture_and_table_on_replay() -> None:
    shared = (
        REPO_ROOT
        / "scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
    ).read_text()
    accepted = (
        REPO_ROOT
        / "scripts/greenfield/run_capture_legacy_layer0_attention_output.sh"
    ).read_text()
    ingredients = (
        REPO_ROOT
        / "scripts/greenfield/run_capture_table_on_layer0_ingredients.sh"
    ).read_text()
    for required in (
        "bf8a03e264971c8efba99a346d1e8189ef0ff518",
        "OBSERVER_COMMIT_DISTANCE=7",
        "ATTENTION_OUTPUT_CAPTURE",
        "capture_accepted_attention_output_operand.py",
        "logical_head_order_live_row",
        "accepted_attention_output_manifest_file_sha256",
    ):
        assert required in shared
    for required in (
        "GLM_GREENFIELD_DSA_INTERNALS_MODE=attention_output",
        "GLM_GREENFIELD_DSA_INTERNALS_POSITION=8155",
        "run_capture_short_context_dsa_oracle.sh",
    ):
        assert required in accepted
    for required in (
        "GLM_GREENFIELD_MAIN_ROPE_TABLE=1",
        "GLM_GREENFIELD_LAYER0_INGREDIENTS=1",
        "GLM_GREENFIELD_DSA_SCORE_DEFAULT_PRECISION=1",
        "run_short_decoder_compile_pp8.sh",
    ):
        assert required in ingredients
