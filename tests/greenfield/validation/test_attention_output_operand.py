from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from glm_tpu.greenfield.validation.attention_output_operand import (
    AcceptedAttentionProjectionCaptureConfig,
    AcceptedAttentionOutputCaptureConfig,
    AttentionProjectionComparisonConfig,
    AttentionOutputComparisonConfig,
    MAIN_ROPE_TABLE_SHA256,
    capture_accepted_attention_projection_operands,
    capture_accepted_attention_output_operand,
    compare_attention_projection_operands,
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


def _write_projection_source(
    path: Path,
    latent_values: np.ndarray,
    output_values: np.ndarray,
    **overrides: object,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: dict[str, np.ndarray] = {
        "artifact_kind": np.asarray(
            "glm52_legacy_attention_projection_operands"
        ),
        "format_version": np.asarray(1, dtype=np.int64),
        "capture_mode": np.asarray("attention_projection"),
        "process_index": np.asarray(0, dtype=np.int64),
        "process_count": np.asarray(8, dtype=np.int64),
        "layer_name": np.asarray(LAYER_NAME),
        "position": np.asarray(8155, dtype=np.int32),
        "source_row": np.asarray(0, dtype=np.int32),
        "run_tag": np.asarray(ACCEPTED_TAG),
        "code_hash": np.asarray(LEGACY_HASH),
        "oracle_pin": np.asarray(ORACLE_HASH),
        "model_id": np.asarray("zai-org/GLM-5.2-FP8"),
        "attended_latent": _bits(latent_values),
        "attended_latent__dtype": np.asarray("bfloat16"),
        "attention_output": _bits(output_values),
        "attention_output__dtype": np.asarray("bfloat16"),
    }
    fields.update({name: np.asarray(value) for name, value in overrides.items()})
    np.savez(path, **fields)


def _projection_capture(
    tmp_path: Path,
    *,
    latent_values: np.ndarray | None = None,
    output_values: np.ndarray | None = None,
) -> Path:
    source = tmp_path / "projection-source"
    safe_layer = LAYER_NAME.replace(".", "_")
    _write_projection_source(
        source / f"internals.{safe_layer}.position8155.proc0.npz",
        (
            np.linspace(-0.1, 0.1, 64 * 512, dtype=np.float32).reshape(64, 512)
            if latent_values is None
            else latent_values
        ),
        _accepted_values() if output_values is None else output_values,
    )
    output = tmp_path / "accepted-projection"
    capture_accepted_attention_projection_operands(
        AcceptedAttentionProjectionCaptureConfig(
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


def _write_projection_ingredients(
    root: Path,
    latent_bits: np.ndarray,
    output_bits: np.ndarray,
) -> tuple[str, str]:
    root.mkdir(parents=True)
    combined = np.repeat(latent_bits[None, ...], 4, axis=0)
    owner_output = output_bits.reshape(4, 4096)
    value_states = owner_output.reshape(4, 16, 256)
    tensor_path = root / "position_8155_ingredients.npz"
    np.savez(
        tensor_path,
        combined_attention_output_bfloat16_bits=combined,
        value_states_bfloat16_bits=value_states,
        attention_output_input_bfloat16_bits=owner_output,
    )
    arrays = {}
    for name, value in (
        ("combined_attention_output_bfloat16_bits", combined),
        ("value_states_bfloat16_bits", value_states),
        ("attention_output_input_bfloat16_bits", owner_output),
    ):
        arrays[name] = {
            "dtype": "uint16",
            "sha256": sha256(
                np.ascontiguousarray(value).tobytes(order="C")
            ).hexdigest(),
            "shape": list(value.shape),
        }
    contract = {
        "active_rows": [0, 1, 2, 3],
        "arrays": arrays,
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


def _projection_comparison_config(
    tmp_path: Path,
    *,
    accepted: Path,
    ingredients: Path,
    contract_sha: str,
    tensor_sha: str,
    output_name: str = "projection-comparison",
) -> AttentionProjectionComparisonConfig:
    return AttentionProjectionComparisonConfig(
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


def test_projection_comparison_localizes_w_uv_when_latent_is_exact(
    tmp_path: Path,
) -> None:
    latent_values = np.linspace(
        -0.1, 0.1, 64 * 512, dtype=np.float32
    ).reshape(64, 512)
    output_values = _accepted_values()
    accepted = _projection_capture(
        tmp_path,
        latent_values=latent_values,
        output_values=output_values,
    )
    output_bits = _bits(output_values)
    output_bits[123] ^= np.uint16(1)
    ingredients = tmp_path / "projection-ingredients"
    contract_sha, tensor_sha = _write_projection_ingredients(
        ingredients,
        _bits(latent_values),
        output_bits,
    )
    result = compare_attention_projection_operands(
        _projection_comparison_config(
            tmp_path,
            accepted=accepted,
            ingredients=ingredients,
            contract_sha=contract_sha,
            tensor_sha=tensor_sha,
        )
    )
    assert result["attended_latent"]["elementwise_exact"] is True
    assert result["attention_output"]["mismatch_count"] == 1
    assert result["classification"] == "w_uv_projection_arithmetic"


def test_projection_comparison_localizes_upstream_latent_and_refuses_replica_drift(
    tmp_path: Path,
) -> None:
    latent_values = np.linspace(
        -0.1, 0.1, 64 * 512, dtype=np.float32
    ).reshape(64, 512)
    output_values = _accepted_values()
    accepted = _projection_capture(
        tmp_path,
        latent_values=latent_values,
        output_values=output_values,
    )
    latent_bits = _bits(latent_values)
    latent_bits[7, 11] ^= np.uint16(1)
    ingredients = tmp_path / "projection-ingredients"
    contract_sha, tensor_sha = _write_projection_ingredients(
        ingredients,
        latent_bits,
        _bits(output_values),
    )
    config = _projection_comparison_config(
        tmp_path,
        accepted=accepted,
        ingredients=ingredients,
        contract_sha=contract_sha,
        tensor_sha=tensor_sha,
    )
    result = compare_attention_projection_operands(config)
    assert result["attended_latent"]["mismatch_count"] == 1
    assert result["classification"] == "attention_arithmetic_before_w_uv"

    with np.load(
        ingredients / "position_8155_ingredients.npz", allow_pickle=False
    ) as payload:
        arrays = {name: payload[name] for name in payload.files}
    arrays["combined_attention_output_bfloat16_bits"] = arrays[
        "combined_attention_output_bfloat16_bits"
    ].copy()
    arrays["combined_attention_output_bfloat16_bits"][3, 1, 2] ^= np.uint16(1)
    drifted = tmp_path / "projection-ingredients-drifted"
    drifted.mkdir()
    np.savez(drifted / "position_8155_ingredients.npz", **arrays)
    contract = json.loads((ingredients / "contract.json").read_text())
    value = arrays["combined_attention_output_bfloat16_bits"]
    contract["arrays"]["combined_attention_output_bfloat16_bits"][
        "sha256"
    ] = sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()
    (drifted / "contract.json").write_text(
        json.dumps(contract, sort_keys=True) + "\n"
    )
    drifted_config = _projection_comparison_config(
        tmp_path,
        accepted=accepted,
        ingredients=drifted,
        contract_sha=_file_sha256(drifted / "contract.json"),
        tensor_sha=_file_sha256(drifted / "position_8155_ingredients.npz"),
        output_name="projection-comparison-drifted",
    )
    with pytest.raises(ValueError, match="owners disagree"):
        compare_attention_projection_operands(drifted_config)


def test_protected_wrappers_pin_isolated_capture_and_table_on_replay() -> None:
    proven_main_rope_sha = (
        "6a22140fc2aec475399738c6fc0f29be2a6c419feb0249aee35681c607c80701"
    )
    assert MAIN_ROPE_TABLE_SHA256 == proven_main_rope_sha
    shared = (
        REPO_ROOT
        / "scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
    ).read_text()
    accepted = (
        REPO_ROOT
        / "scripts/greenfield/run_capture_legacy_layer0_attention_output.sh"
    ).read_text()
    projection = (
        REPO_ROOT
        / "scripts/greenfield/run_capture_legacy_layer0_attention_projection.sh"
    ).read_text()
    ingredients = (
        REPO_ROOT
        / "scripts/greenfield/run_capture_table_on_layer0_ingredients.sh"
    ).read_text()
    decoder_runner = (
        REPO_ROOT / "scripts/greenfield/run_short_decoder_compile_pp8.sh"
    ).read_text()
    assert proven_main_rope_sha in decoder_runner
    assert "6a22140f31c94bbb99092301902a4ed18" not in decoder_runner
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
        "11c2506480e98902d66a88309533f624c994d202",
        "OBSERVER_COMMIT_DISTANCE=8",
        "ATTENTION_PROJECTION_CAPTURE",
        "capture_accepted_attention_projection_operands.py",
        "compare_attention_projection_operands.py",
        "glm52_accepted_greenfield_attention_projection_comparison",
        "attention_projection_latent_mismatch_count",
        "79a6e290274ef470b518a6de894b44d2929ad6861d1751f0915aa4eb20cf2e9d",
        "0103e22c558d1390820bc9d39cc05b90f4555fa7c0be5de7cd33c34748a582ab",
    ):
        assert required in shared
    for required in (
        "GLM_GREENFIELD_DSA_INTERNALS_MODE=attention_output",
        "GLM_GREENFIELD_DSA_INTERNALS_POSITION=8155",
        "run_capture_short_context_dsa_oracle.sh",
    ):
        assert required in accepted
    for required in (
        "GLM_GREENFIELD_DSA_INTERNALS_MODE=attention_projection",
        "GLM_GREENFIELD_DSA_INTERNALS_POSITION=8155",
        "oracles/greenfield/glm52/attention_projection/8k",
        "run_capture_short_context_dsa_oracle.sh",
    ):
        assert required in projection
    for required in (
        "GLM_GREENFIELD_MAIN_ROPE_TABLE=1",
        "GLM_GREENFIELD_LAYER0_INGREDIENTS=1",
        "GLM_GREENFIELD_DSA_SCORE_DEFAULT_PRECISION=1",
        "run_short_decoder_compile_pp8.sh",
    ):
        assert required in ingredients
