"""Seal the accepted layer-1 fused-add/RMSNorm input operands.

The legacy observer is an oracle-only, non-returning tap.  This module does
not import legacy execution.  It authenticates the observer artifact, checks
the host FP32 reconstruction bitwise, and binds both BF16 operands to the
already sealed DB550 leaves used by the PP16 Gate-D diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import ml_dtypes
import numpy as np

from ..benchmarking.pp16_dense_boundary import derive_expected_dense_boundary_bits
from ..kernels.stage_local import STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE


SOURCE_KIND = "glm52_legacy_layer1_rms_input_operands"
CAPTURE_KIND = "glm52_accepted_layer1_rms_input_capture"
COMPARISON_KIND = "glm52_accepted_layer1_rms_input_db550_comparison"
MODEL_ID = "zai-org/GLM-5.2-FP8"
LAYER_NAME = "model.layers.1.input_layernorm"
POSITION = 8155
WIDTH = 6144
RECONSTRUCTION_SEMANTICS = (
    "float32(hidden_update_bfloat16) + float32(carried_residual_bfloat16)"
)
VLLM_PIN = "a30addc7548a9a8b9b3323a7bc3eb7d7c4895d1c"
VLLM_IR_LAYERNORM_SHA256 = (
    "d8e4380ca97d2c719836a7e15fb410a73d354b15d79fc54275b573bbb1c06910"
)
VLLM_EXECUTOR_LAYERNORM_SHA256 = (
    "53c6abdab25dc1675f26f4c8fc5ba2094f1fb4a106334e581f630436210d0c9b"
)
DB550_BOUNDARY_SHA256 = (
    "f194d757d2f9ebe27430dfec8f828ca7588e433bddb7e8d99f9b917c5aac4298"
)
STRADDLER_CLASSIFICATION_SHA256 = (
    "eebe1c5d5ba475a5faf000243d881657754fc47ed2345fe2691d923c1d457b36"
)
STRADDLER_CLASSIFICATION = "BF16_BOUNDARY_INSUFFICIENT_FOR_FP32_CAUSAL_ADJUDICATION"


@dataclass(frozen=True, slots=True)
class Layer1RmsInputCaptureConfig:
    """Immutable inputs for one append-only accepted-boundary seal."""

    source_dump_dir: Path
    output_dir: Path
    db550_boundary_path: Path
    straddler_classification_path: Path
    vllm_repository: Path
    expected_run_tag: str
    expected_legacy_code_hash: str
    expected_oracle_pin: str
    expected_db550_sha256: str = DB550_BOUNDARY_SHA256
    expected_straddler_sha256: str = STRADDLER_CLASSIFICATION_SHA256
    expected_vllm_pin: str = VLLM_PIN
    expected_vllm_ir_layernorm_sha256: str = VLLM_IR_LAYERNORM_SHA256
    expected_vllm_executor_layernorm_sha256: str = VLLM_EXECUTOR_LAYERNORM_SHA256
    expected_model_id: str = MODEL_ID
    expected_layer_name: str = LAYER_NAME
    expected_position: int = POSITION
    expected_process_count: int = 8
    expected_capture_process_index: int = 0
    expected_source_row: int = 0

    def __post_init__(self) -> None:
        if not self.expected_run_tag.strip():
            raise ValueError("expected run tag must be non-empty")
        if self.expected_process_count <= 0:
            raise ValueError("expected process count must be positive")
        if not 0 <= self.expected_capture_process_index < self.expected_process_count:
            raise ValueError("capture process index is outside the fleet")
        if self.expected_source_row < 0:
            raise ValueError("source row must be nonnegative")


def _array_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _manifest_sha256(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _require_digest(value: str, *, length: int, name: str) -> None:
    if len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} is not a lowercase digest")


def _scalar(payload: Any, name: str) -> Any:
    value = payload[name]
    if value.shape != ():
        raise ValueError(f"layer-1 RMS-input field {name} is not scalar")
    return value.item()


def _decode_bf16(bits: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(bits)
    if value.dtype != np.uint16:
        raise ValueError("layer-1 RMS-input BF16 storage is not uint16")
    return value.view(ml_dtypes.bfloat16).astype(np.float32)


def _encode_bf16(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value, dtype=ml_dtypes.bfloat16)).view(
        np.uint16
    )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def _git_output(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _validate_source_contract(config: Layer1RmsInputCaptureConfig) -> dict[str, Any]:
    if (
        _git_output(config.vllm_repository, "rev-parse", "HEAD")
        != config.expected_vllm_pin
    ):
        raise ValueError("vLLM reference pin drifted")
    if _git_output(
        config.vllm_repository, "status", "--porcelain", "--untracked-files=no"
    ):
        raise ValueError("vLLM reference tracked files are dirty")
    paths = {
        "ir_ops_layernorm": config.vllm_repository / "vllm/ir/ops/layernorm.py",
        "executor_layernorm": config.vllm_repository
        / "vllm/model_executor/layers/layernorm.py",
    }
    expected = {
        "ir_ops_layernorm": config.expected_vllm_ir_layernorm_sha256,
        "executor_layernorm": config.expected_vllm_executor_layernorm_sha256,
    }
    records: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        source_bytes = path.read_bytes()
        observed = sha256(source_bytes).hexdigest()
        if observed != expected[name]:
            raise ValueError(f"vLLM {name} source hash drifted")
        records[name] = {
            "path": path.relative_to(config.vllm_repository).as_posix(),
            "sha256": observed,
        }
    return {
        "fused_add_semantics": RECONSTRUCTION_SEMANTICS,
        "repository_pin": config.expected_vllm_pin,
        "source_files": records,
    }


def _validate_straddler(config: Layer1RmsInputCaptureConfig) -> dict[str, Any]:
    source_bytes = config.straddler_classification_path.read_bytes()
    if sha256(source_bytes).hexdigest() != config.expected_straddler_sha256:
        raise ValueError("PP16 straddler classification file hash drifted")
    report = json.loads(source_bytes.decode("utf-8"))
    if (
        report.get("artifact_kind")
        != "greenfield_pp16_feature2_layer1_straddler_classification"
        or report.get("classification") != STRADDLER_CLASSIFICATION
        or report.get("status") != "CLASSIFIED"
        or report.get("localized_boundary", {}).get("hidden_index") != 2795
        or report.get("localized_boundary", {}).get("observed_normalized_bits") != 48422
        or report.get("localized_boundary", {}).get("accepted_normalized_bits") != 48423
    ):
        raise ValueError("PP16 straddler classification contract drifted")
    return {
        "classification": STRADDLER_CLASSIFICATION,
        "file_sha256": config.expected_straddler_sha256,
        "hidden_index": 2795,
    }


def _load_observer(
    config: Layer1RmsInputCaptureConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    errors = sorted(config.source_dump_dir.rglob("*.INTERNAL.ERROR.*"))
    if errors:
        raise ValueError(f"layer-1 RMS-input error sentinel exists: {errors[0]}")
    safe_layer = config.expected_layer_name.replace("/", "_").replace(".", "_")
    paths = sorted(
        config.source_dump_dir.rglob(
            f"*.{safe_layer}.position{config.expected_position}.proc*.npz"
        )
    )
    if len(paths) != 1:
        raise ValueError("layer-1 RMS-input capture file count drifted")
    path = paths[0]
    source_bytes = path.read_bytes()
    expected_keys = {
        "artifact_kind",
        "format_version",
        "capture_mode",
        "process_index",
        "process_count",
        "layer_name",
        "position",
        "source_row",
        "reconstruction_semantics",
        "run_tag",
        "code_hash",
        "oracle_pin",
        "model_id",
        "hidden_update",
        "hidden_update__dtype",
        "carried_residual",
        "carried_residual__dtype",
        "fused_add_float32",
        "fused_add_float32__dtype",
    }
    with np.load(BytesIO(source_bytes), allow_pickle=False) as payload:
        if set(payload.files) != expected_keys:
            raise ValueError("layer-1 RMS-input observer key set drifted")
        expected_scalars = {
            "artifact_kind": SOURCE_KIND,
            "format_version": 1,
            "capture_mode": "layer1_rms_input",
            "process_index": config.expected_capture_process_index,
            "process_count": config.expected_process_count,
            "layer_name": config.expected_layer_name,
            "position": config.expected_position,
            "source_row": config.expected_source_row,
            "reconstruction_semantics": RECONSTRUCTION_SEMANTICS,
            "run_tag": config.expected_run_tag,
            "code_hash": config.expected_legacy_code_hash,
            "oracle_pin": config.expected_oracle_pin,
            "model_id": config.expected_model_id,
            "hidden_update__dtype": "bfloat16",
            "carried_residual__dtype": "bfloat16",
            "fused_add_float32__dtype": "float32",
        }
        for name, expected in expected_scalars.items():
            if _scalar(payload, name) != expected:
                raise ValueError(f"layer-1 RMS-input observer {name} drifted")
        arrays = {
            "hidden_update_bfloat16_bits": np.ascontiguousarray(
                payload["hidden_update"]
            ),
            "carried_residual_bfloat16_bits": np.ascontiguousarray(
                payload["carried_residual"]
            ),
            "fused_add_float32": np.ascontiguousarray(payload["fused_add_float32"]),
        }
    for name in ("hidden_update_bfloat16_bits", "carried_residual_bfloat16_bits"):
        if arrays[name].shape != (WIDTH,) or arrays[name].dtype != np.uint16:
            raise ValueError(f"layer-1 RMS-input {name} geometry drifted")
        if not np.isfinite(_decode_bf16(arrays[name])).all():
            raise ValueError(f"layer-1 RMS-input {name} is non-finite")
    fused = arrays["fused_add_float32"]
    if (
        fused.shape != (WIDTH,)
        or fused.dtype != np.float32
        or not np.isfinite(fused).all()
    ):
        raise ValueError("layer-1 RMS-input fused FP32 geometry drifted")
    reconstructed = np.ascontiguousarray(
        _decode_bf16(arrays["hidden_update_bfloat16_bits"])
        + _decode_bf16(arrays["carried_residual_bfloat16_bits"]),
        dtype=np.float32,
    )
    if not np.array_equal(fused.view(np.uint32), reconstructed.view(np.uint32)):
        raise ValueError("layer-1 RMS-input host FP32 reconstruction drifted")
    record = {
        "byte_count": len(source_bytes),
        "path": path.relative_to(config.source_dump_dir).as_posix(),
        "process_index": config.expected_capture_process_index,
        "sha256": sha256(source_bytes).hexdigest(),
        "source_row": config.expected_source_row,
    }
    return arrays, record


def _load_db550(
    config: Layer1RmsInputCaptureConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source_bytes = config.db550_boundary_path.read_bytes()
    if sha256(source_bytes).hexdigest() != config.expected_db550_sha256:
        raise ValueError("DB550 boundary file hash drifted")
    required = {
        "dense_virtual_partials_bfloat16_bits",
        "post_attention_residual_bfloat16_bits",
    }
    with np.load(BytesIO(source_bytes), allow_pickle=False) as payload:
        if not required.issubset(payload.files):
            raise ValueError("DB550 boundary keys drifted")
        dense_partials = np.ascontiguousarray(
            payload["dense_virtual_partials_bfloat16_bits"]
        )
        residual = np.ascontiguousarray(
            payload["post_attention_residual_bfloat16_bits"]
        )
    model_axis_device_ids = tuple(
        int(item)
        for item in np.argsort(
            np.asarray(STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE)
        )
    )
    dense, carried = derive_expected_dense_boundary_bits(
        dense_partials, residual, model_axis_device_ids
    )
    dense = np.ascontiguousarray(dense.reshape(WIDTH))
    residual = np.ascontiguousarray(residual.reshape(WIDTH))
    carried = np.ascontiguousarray(carried.reshape(WIDTH))
    fused = np.ascontiguousarray(
        _decode_bf16(dense) + _decode_bf16(residual), dtype=np.float32
    )
    return {
        "hidden_update_bfloat16_bits": dense,
        "carried_residual_bfloat16_bits": residual,
        "fused_add_float32": fused,
        "carried_bfloat16_bits": carried,
    }, {
        "file_sha256": config.expected_db550_sha256,
        "model_axis_device_ids": list(model_axis_device_ids),
    }


def _exact_comparison(expected: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    if expected.shape != observed.shape or expected.dtype != observed.dtype:
        raise ValueError("layer-1 RMS-input comparison schema drifted")
    expected_bytes = np.ascontiguousarray(expected).view(np.uint8)
    observed_bytes = np.ascontiguousarray(observed).view(np.uint8)
    mismatch = expected_bytes != observed_bytes
    count = int(np.count_nonzero(mismatch))
    return {
        "bytewise_exact": count == 0,
        "dtype": str(expected.dtype),
        "expected_sha256": _array_sha256(expected),
        "first_mismatch_byte": int(np.flatnonzero(mismatch)[0]) if count else None,
        "mismatch_byte_count": count,
        "observed_sha256": _array_sha256(observed),
        "shape": list(expected.shape),
    }


def _build_comparison(
    capture_manifest_sha256: str,
    observer: dict[str, np.ndarray],
    db550: dict[str, np.ndarray],
    db550_receipt: dict[str, Any],
    straddler_receipt: dict[str, Any],
) -> dict[str, Any]:
    rounded_fused = _encode_bf16(observer["fused_add_float32"])
    comparisons = {
        "hidden_update": _exact_comparison(
            db550["hidden_update_bfloat16_bits"],
            observer["hidden_update_bfloat16_bits"],
        ),
        "carried_residual": _exact_comparison(
            db550["carried_residual_bfloat16_bits"],
            observer["carried_residual_bfloat16_bits"],
        ),
        "fused_add_float32": _exact_comparison(
            db550["fused_add_float32"], observer["fused_add_float32"]
        ),
        "rounded_fused_add": _exact_comparison(
            db550["carried_bfloat16_bits"], rounded_fused
        ),
    }
    all_exact = all(item["bytewise_exact"] for item in comparisons.values())
    classification = (
        "ACCEPTED_FP32_SOURCE_BOUND_TO_DB550_OPERANDS"
        if all_exact
        else "ACCEPTED_FP32_SOURCE_DIVERGES_FROM_DB550_DERIVATION"
    )
    report: dict[str, Any] = {
        "accepted_capture_manifest_sha256": capture_manifest_sha256,
        "artifact_kind": COMPARISON_KIND,
        "classification": classification,
        "claim_scope": (
            "accepted layer-1 oracle boundary only; no greenfield numerical "
            "acceptance, Gate-D, DB, token-rate, latency or performance claim"
        ),
        "comparisons": comparisons,
        "db550": db550_receipt,
        "diagnostic_only": True,
        "format_version": 1,
        "operands_match_db550": all_exact,
        "performance_claim": False,
        "status": "CLASSIFIED",
        "straddler": straddler_receipt,
    }
    report["manifest_sha256"] = _manifest_sha256(report)
    return report


def capture_accepted_layer1_rms_input(
    config: Layer1RmsInputCaptureConfig,
) -> dict[str, Any]:
    """Seal and classify one accepted layer-1 FP32 RMS-input boundary."""

    for value, length, name in (
        (config.expected_legacy_code_hash, 40, "legacy code hash"),
        (config.expected_oracle_pin, 40, "oracle pin"),
        (config.expected_vllm_pin, 40, "vLLM pin"),
        (config.expected_db550_sha256, 64, "DB550"),
        (config.expected_straddler_sha256, 64, "straddler"),
        (config.expected_vllm_ir_layernorm_sha256, 64, "vLLM IR layernorm"),
        (
            config.expected_vllm_executor_layernorm_sha256,
            64,
            "vLLM executor layernorm",
        ),
    ):
        _require_digest(value, length=length, name=name)
    if config.output_dir.exists():
        raise FileExistsError(f"append-only capture exists: {config.output_dir}")
    source_contract = _validate_source_contract(config)
    straddler_receipt = _validate_straddler(config)
    observer, source_file = _load_observer(config)
    db550, db550_receipt = _load_db550(config)

    config.output_dir.mkdir(parents=True)
    tensor_path = config.output_dir / "layer1_rms_input.npz"
    _atomic_npz(tensor_path, **observer)
    tensor_bytes = tensor_path.read_bytes()
    tensors = {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "tensor_sha256": _array_sha256(value),
        }
        for name, value in observer.items()
    }
    capture: dict[str, Any] = {
        "artifact_kind": CAPTURE_KIND,
        "capture_layout": "replicated_logical_live_row",
        "capture_mode": "layer1_rms_input",
        "capture_process_indices": [config.expected_capture_process_index],
        "claim_scope": (
            "accepted non-returning oracle operands and host FP32 reconstruction "
            "only; no greenfield numerical, Gate-D or performance claim"
        ),
        "diagnostic_only": True,
        "format_version": 1,
        "layer_name": config.expected_layer_name,
        "legacy_code_hash": config.expected_legacy_code_hash,
        "model_id": config.expected_model_id,
        "oracle_pin": config.expected_oracle_pin,
        "performance_claim": False,
        "position": config.expected_position,
        "process_count": config.expected_process_count,
        "reconstruction_semantics": RECONSTRUCTION_SEMANTICS,
        "run_tag": config.expected_run_tag,
        "source_file": source_file,
        "source_row": config.expected_source_row,
        "tensor_file": {
            "byte_count": len(tensor_bytes),
            "filename": tensor_path.name,
            "sha256": sha256(tensor_bytes).hexdigest(),
        },
        "tensors": tensors,
        "vllm_source_contract": source_contract,
    }
    capture["manifest_sha256"] = _manifest_sha256(capture)
    _atomic_json(config.output_dir / "capture.json", capture)
    comparison = _build_comparison(
        capture["manifest_sha256"],
        observer,
        db550,
        db550_receipt,
        straddler_receipt,
    )
    _atomic_json(config.output_dir / "comparison.json", comparison)
    return {"capture": capture, "comparison": comparison}


def validate_layer1_rms_input_artifacts(
    config: Layer1RmsInputCaptureConfig,
) -> dict[str, Any]:
    """Re-authenticate source evidence and one existing sealed output."""

    capture_path = config.output_dir / "capture.json"
    comparison_path = config.output_dir / "comparison.json"
    tensor_path = config.output_dir / "layer1_rms_input.npz"
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if capture.get("manifest_sha256") != _manifest_sha256(capture):
        raise ValueError("layer-1 RMS-input capture manifest hash drifted")
    if comparison.get("manifest_sha256") != _manifest_sha256(comparison):
        raise ValueError("layer-1 RMS-input comparison manifest hash drifted")
    source_contract = _validate_source_contract(config)
    straddler_receipt = _validate_straddler(config)
    observer, source_file = _load_observer(config)
    db550, db550_receipt = _load_db550(config)
    tensor_bytes = tensor_path.read_bytes()
    with np.load(BytesIO(tensor_bytes), allow_pickle=False) as payload:
        if set(payload.files) != set(observer):
            raise ValueError("layer-1 RMS-input sealed tensor keys drifted")
        sealed = {name: np.ascontiguousarray(payload[name]) for name in payload.files}
    if any(
        not np.array_equal(sealed[name].view(np.uint8), value.view(np.uint8))
        for name, value in observer.items()
    ):
        raise ValueError("layer-1 RMS-input sealed tensor bytes drifted")
    if (
        capture.get("artifact_kind") != CAPTURE_KIND
        or capture.get("capture_layout") != "replicated_logical_live_row"
        or capture.get("capture_mode") != "layer1_rms_input"
        or capture.get("capture_process_indices")
        != [config.expected_capture_process_index]
        or capture.get("claim_scope")
        != (
            "accepted non-returning oracle operands and host FP32 reconstruction "
            "only; no greenfield numerical, Gate-D or performance claim"
        )
        or capture.get("diagnostic_only") is not True
        or capture.get("format_version") != 1
        or capture.get("performance_claim") is not False
        or capture.get("legacy_code_hash") != config.expected_legacy_code_hash
        or capture.get("model_id") != config.expected_model_id
        or capture.get("oracle_pin") != config.expected_oracle_pin
        or capture.get("layer_name") != config.expected_layer_name
        or capture.get("position") != config.expected_position
        or capture.get("process_count") != config.expected_process_count
        or capture.get("reconstruction_semantics") != RECONSTRUCTION_SEMANTICS
        or capture.get("run_tag") != config.expected_run_tag
        or capture.get("source_file") != source_file
        or capture.get("source_row") != config.expected_source_row
        or capture.get("vllm_source_contract") != source_contract
        or capture.get("tensor_file")
        != {
            "byte_count": len(tensor_bytes),
            "filename": tensor_path.name,
            "sha256": sha256(tensor_bytes).hexdigest(),
        }
        or capture.get("tensors")
        != {
            name: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "tensor_sha256": _array_sha256(value),
            }
            for name, value in observer.items()
        }
    ):
        raise ValueError("layer-1 RMS-input capture contract drifted")
    expected_comparison = _build_comparison(
        capture["manifest_sha256"],
        observer,
        db550,
        db550_receipt,
        straddler_receipt,
    )
    if comparison != expected_comparison:
        raise ValueError("layer-1 RMS-input comparison contract drifted")
    return {"capture": capture, "comparison": comparison}
