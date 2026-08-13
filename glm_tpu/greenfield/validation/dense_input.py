"""Seal accepted layer-0 dense input and compare it with protected DB540."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np


SOURCE_KIND = "glm52_legacy_dense_input"
CAPTURE_KIND = "glm52_accepted_dense_input_capture"
COMPARISON_KIND = "glm52_accepted_greenfield_dense_input_comparison"
MODEL_ID = "zai-org/GLM-5.2-FP8"
LAYER_NAME = "model.layers.0.self_attn.attn"
POSITION = 8155
WIDTH = 6144
PROBE_KIND = "glm52_layer0_dense_convolution_probe"
PROBE_CODE_HASH = "2f63779309b25c71c1cc7d35ff97715ae4bf631e"
PROBE_TAG = "greenfield_layer0_dense_convolution_20260813T005213127235575Z"
PROBE_CLASSIFICATION = "accepted_dense_convolution_nonexact"
PROBE_RUN_ID = 540
PROBE_NORMALIZED_MLP_SHA256 = (
    "082125fead43b25f10686705c1b6473153f4092dd5bc476f8e01a86629f0758f"
)


@dataclass(frozen=True, slots=True)
class AcceptedDenseInputCaptureConfig:
    source_dump_dir: Path
    output_dir: Path
    expected_run_tag: str
    expected_legacy_code_hash: str
    expected_oracle_pin: str
    expected_model_id: str = MODEL_ID
    expected_layer_name: str = LAYER_NAME
    expected_position: int = POSITION
    expected_process_count: int = 8
    expected_capture_process_indices: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        indices = tuple(self.expected_capture_process_indices)
        object.__setattr__(self, "expected_capture_process_indices", indices)
        if not self.expected_run_tag.strip():
            raise ValueError("expected run tag must be non-empty")
        if self.expected_process_count <= 0:
            raise ValueError("expected process count must be positive")
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("capture process indices must be non-empty and unique")
        if any(not 0 <= value < self.expected_process_count for value in indices):
            raise ValueError("capture process index is outside the fleet")


@dataclass(frozen=True, slots=True)
class DenseInputComparisonConfig:
    accepted_capture_dir: Path
    probe_dir: Path
    output_dir: Path
    expected_accepted_capture_file_sha256: str
    expected_probe_runner_sha256: str
    expected_probe_tensor_sha256: str
    expected_probe_summary_sha256: str
    expected_probe_success_sha256: str
    expected_accepted_run_tag: str
    expected_legacy_code_hash: str
    expected_oracle_pin: str
    expected_probe_code_hash: str = PROBE_CODE_HASH
    expected_probe_tag: str = PROBE_TAG
    expected_position: int = POSITION
    expected_probe_run_id: int = PROBE_RUN_ID
    expected_probe_normalized_mlp_sha256: str = PROBE_NORMALIZED_MLP_SHA256


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _manifest_sha256(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    return sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _require_digest(value: str, *, length: int, name: str) -> None:
    if len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} is not a lowercase digest")


def _scalar(payload: Any, name: str) -> Any:
    value = payload[name]
    if value.shape != ():
        raise ValueError(f"dense-input field {name} is not scalar")
    return value.item()


def _decode(bits: np.ndarray) -> np.ndarray:
    if bits.dtype != np.dtype(np.uint16):
        raise ValueError("dense-input BF16 storage is not uint16")
    return np.ascontiguousarray(bits).view(ml_dtypes.bfloat16).astype(np.float32)


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


def _load_source_rows(
    config: AcceptedDenseInputCaptureConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    errors = sorted(config.source_dump_dir.rglob("*.INTERNAL.ERROR.*"))
    if errors:
        raise ValueError(f"dense-input error sentinel exists: {errors[0]}")
    safe_layer = config.expected_layer_name.replace("/", "_").replace(".", "_")
    paths = sorted(
        config.source_dump_dir.rglob(
            f"*.{safe_layer}.position{config.expected_position}.proc*.npz"
        )
    )
    if len(paths) != len(config.expected_capture_process_indices):
        raise ValueError("dense-input capture file count drifted")
    expected_keys = {
        "artifact_kind",
        "format_version",
        "capture_mode",
        "process_index",
        "process_count",
        "layer_name",
        "position",
        "source_row",
        "run_tag",
        "code_hash",
        "oracle_pin",
        "model_id",
        "normalized_mlp",
        "normalized_mlp__dtype",
    }
    canonical: np.ndarray | None = None
    process_indices: set[int] = set()
    records: list[dict[str, Any]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != expected_keys:
                raise ValueError(f"{path}: dense-input key set drifted")
            expected_scalars = {
                "artifact_kind": SOURCE_KIND,
                "format_version": 1,
                "capture_mode": "dense_input",
                "process_count": config.expected_process_count,
                "layer_name": config.expected_layer_name,
                "position": config.expected_position,
                "source_row": 0,
                "run_tag": config.expected_run_tag,
                "code_hash": config.expected_legacy_code_hash,
                "oracle_pin": config.expected_oracle_pin,
                "model_id": config.expected_model_id,
                "normalized_mlp__dtype": "bfloat16",
            }
            for name, expected in expected_scalars.items():
                observed = _scalar(payload, name)
                if observed != expected:
                    raise ValueError(f"{path}: {name}={observed!r} != {expected!r}")
            process_index = int(_scalar(payload, "process_index"))
            if process_index in process_indices or not (
                0 <= process_index < config.expected_process_count
            ):
                raise ValueError(f"{path}: invalid/duplicate process index")
            process_indices.add(process_index)
            bits = np.ascontiguousarray(payload["normalized_mlp"])
            if bits.shape != (WIDTH,) or bits.dtype != np.dtype(np.uint16):
                raise ValueError(f"{path}: normalized-MLP tensor drifted")
            if not np.isfinite(_decode(bits)).all():
                raise ValueError(f"{path}: normalized-MLP tensor is non-finite")
            if canonical is None:
                canonical = bits.copy()
            elif not np.array_equal(canonical, bits):
                raise ValueError(f"{path}: replicated dense input disagrees")
            records.append({
                "byte_count": path.stat().st_size,
                "path": path.relative_to(config.source_dump_dir).as_posix(),
                "process_index": process_index,
                "sha256": _file_sha256(path),
            })
    if process_indices != set(config.expected_capture_process_indices) or canonical is None:
        raise ValueError("dense-input process coverage is incomplete")
    return canonical, sorted(records, key=lambda value: value["process_index"])


def capture_accepted_dense_input(
    config: AcceptedDenseInputCaptureConfig,
) -> dict[str, Any]:
    """Validate raw observer output and seal one accepted BF16 row."""

    _require_digest(config.expected_legacy_code_hash, length=40, name="legacy code hash")
    _require_digest(config.expected_oracle_pin, length=40, name="oracle pin")
    if config.output_dir.exists():
        raise FileExistsError(f"append-only capture exists: {config.output_dir}")
    bits, process_files = _load_source_rows(config)
    config.output_dir.mkdir(parents=True)
    tensor_path = config.output_dir / "dense_input.npz"
    _atomic_npz(tensor_path, normalized_mlp_bfloat16_bits=bits)
    manifest: dict[str, Any] = {
        "artifact_kind": CAPTURE_KIND,
        "capture_layout": "replicated_logical_live_row",
        "capture_mode": "dense_input",
        "capture_process_indices": list(config.expected_capture_process_indices),
        "diagnostic_only": True,
        "format_version": 1,
        "layer_name": config.expected_layer_name,
        "legacy_code_hash": config.expected_legacy_code_hash,
        "model_id": config.expected_model_id,
        "oracle_pin": config.expected_oracle_pin,
        "performance_claim": False,
        "position": config.expected_position,
        "process_count": config.expected_process_count,
        "process_files": process_files,
        "run_tag": config.expected_run_tag,
        "tensor": {
            "byte_count": tensor_path.stat().st_size,
            "filename": tensor_path.name,
            "file_sha256": _file_sha256(tensor_path),
            "shape": [WIDTH],
            "tensor_sha256": _array_sha256(bits),
        },
    }
    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    _atomic_json(config.output_dir / "capture.json", manifest)
    return manifest


def _load_capture(
    config: DenseInputComparisonConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest_path = config.accepted_capture_dir / "capture.json"
    if _file_sha256(manifest_path) != config.expected_accepted_capture_file_sha256:
        raise ValueError("accepted dense-input manifest file hash drifted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != _manifest_sha256(manifest):
        raise ValueError("accepted dense-input manifest hash drifted")
    expected = {
        "artifact_kind": CAPTURE_KIND,
        "capture_layout": "replicated_logical_live_row",
        "capture_mode": "dense_input",
        "capture_process_indices": [0],
        "diagnostic_only": True,
        "format_version": 1,
        "layer_name": LAYER_NAME,
        "legacy_code_hash": config.expected_legacy_code_hash,
        "model_id": MODEL_ID,
        "oracle_pin": config.expected_oracle_pin,
        "performance_claim": False,
        "position": config.expected_position,
        "process_count": 8,
        "run_tag": config.expected_accepted_run_tag,
    }
    if any(manifest.get(name) != value for name, value in expected.items()):
        raise ValueError("accepted dense-input manifest identity drifted")
    process_files = manifest.get("process_files")
    if (
        not isinstance(process_files, list)
        or len(process_files) != 1
        or set(process_files[0]) != {"byte_count", "path", "process_index", "sha256"}
        or process_files[0].get("process_index") != 0
        or not isinstance(process_files[0].get("byte_count"), int)
        or process_files[0]["byte_count"] <= 0
        or not str(process_files[0].get("path", "")).endswith(".proc0.npz")
    ):
        raise ValueError("accepted dense-input process ledger drifted")
    _require_digest(
        str(process_files[0].get("sha256", "")), length=64, name="source file"
    )
    tensor_path = config.accepted_capture_dir / "dense_input.npz"
    record = manifest.get("tensor", {})
    if (
        record.get("filename") != tensor_path.name
        or record.get("shape") != [WIDTH]
        or tensor_path.stat().st_size != record.get("byte_count")
        or _file_sha256(tensor_path) != record.get("file_sha256")
    ):
        raise ValueError("accepted dense-input tensor file drifted")
    with np.load(tensor_path, allow_pickle=False) as payload:
        if set(payload.files) != {"normalized_mlp_bfloat16_bits"}:
            raise ValueError("accepted dense-input tensor keys drifted")
        bits = np.ascontiguousarray(payload["normalized_mlp_bfloat16_bits"])
    if (
        bits.shape != (WIDTH,)
        or bits.dtype != np.dtype(np.uint16)
        or _array_sha256(bits) != record.get("tensor_sha256")
        or not np.isfinite(_decode(bits)).all()
    ):
        raise ValueError("accepted dense-input tensor contract drifted")
    return bits, manifest


def _comparison(expected: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    mismatch = expected != observed
    count = int(np.count_nonzero(mismatch))
    error = np.abs(_decode(expected) - _decode(observed))
    return {
        "elementwise_exact": count == 0,
        "expected_sha256": _array_sha256(expected),
        "first_mismatch_index": int(np.flatnonzero(mismatch)[0]) if count else None,
        "max_abs_error": float(np.max(error)),
        "mean_abs_error": float(np.mean(error, dtype=np.float64)),
        "mismatch_count": count,
        "observed_sha256": _array_sha256(observed),
        "shape": [WIDTH],
    }


def compare_dense_input_candidate(
    config: DenseInputComparisonConfig,
) -> dict[str, Any]:
    """Classify whether divergence precedes or follows the dense MLP input."""

    for value, name in (
        (config.expected_accepted_capture_file_sha256, "capture manifest file"),
        (config.expected_probe_runner_sha256, "probe runner"),
        (config.expected_probe_tensor_sha256, "probe tensor"),
        (config.expected_probe_summary_sha256, "probe summary"),
        (config.expected_probe_success_sha256, "probe SUCCESS"),
        (config.expected_probe_normalized_mlp_sha256, "probe normalized MLP"),
    ):
        _require_digest(value, length=64, name=name)
    _require_digest(config.expected_legacy_code_hash, length=40, name="legacy code hash")
    _require_digest(config.expected_oracle_pin, length=40, name="oracle pin")
    _require_digest(config.expected_probe_code_hash, length=40, name="probe code hash")
    if config.output_dir.exists():
        raise FileExistsError(f"append-only comparison exists: {config.output_dir}")
    accepted, capture = _load_capture(config)
    if config.probe_dir.name != config.expected_probe_tag:
        raise ValueError("dense-convolution probe tag drifted")
    paths = {
        "runner": config.probe_dir / "runner.json",
        "tensor": config.probe_dir / "dense_convolution.npz",
        "summary": config.probe_dir / "summary.json",
        "success": config.probe_dir / "SUCCESS",
    }
    expected_hashes = {
        "runner": config.expected_probe_runner_sha256,
        "tensor": config.expected_probe_tensor_sha256,
        "summary": config.expected_probe_summary_sha256,
        "success": config.expected_probe_success_sha256,
    }
    for name, expected_hash in expected_hashes.items():
        if _file_sha256(paths[name]) != expected_hash:
            raise ValueError(f"dense-convolution {name} file hash drifted")
    runner = json.loads(paths["runner"].read_text(encoding="utf-8"))
    expected_runner = {
        "artifact_kind": PROBE_KIND,
        "classification": PROBE_CLASSIFICATION,
        "code_hash": config.expected_probe_code_hash,
        "exact": False,
        "exact_arms": [],
        "performance_claim": False,
        "position": config.expected_position,
        "status": "SUCCESS",
    }
    if any(runner.get(name) != value for name, value in expected_runner.items()):
        raise ValueError("dense-convolution runner identity drifted")
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if summary != {
        "artifact_kind": PROBE_KIND,
        "classification": PROBE_CLASSIFICATION,
        "code_hash": config.expected_probe_code_hash,
        "elapsed_seconds": 8,
        "exact_arms": [],
        "performance_claim": False,
        "results_db_run_id": config.expected_probe_run_id,
        "status": "SUCCESS",
    }:
        raise ValueError("dense-convolution summary identity drifted")
    success_fields: dict[str, str] = {}
    for line in paths["success"].read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition("=")
        if not separator or name in success_fields:
            raise ValueError("dense-convolution SUCCESS syntax drifted")
        success_fields[name] = value
    if success_fields != {
        "artifact_kind": PROBE_KIND,
        "code_hash": config.expected_probe_code_hash,
        "results_db_run_id": str(config.expected_probe_run_id),
        "classification": PROBE_CLASSIFICATION,
        "exact_arms": "none",
        "performance_claim": "false",
        "evidence_sha256": success_fields.get("evidence_sha256"),
        "remote_objects_sha256": success_fields.get("remote_objects_sha256"),
        "remote_prefix": f"gs://driftbench-dsv4-uc/results/{config.expected_probe_tag}",
    }:
        raise ValueError("dense-convolution SUCCESS identity drifted")
    for name in ("evidence_sha256", "remote_objects_sha256"):
        _require_digest(success_fields.get(name, ""), length=64, name=name)
    with np.load(paths["tensor"], allow_pickle=False) as payload:
        expected_keys = {
            "accepted_layer1_normalized_bfloat16_bits",
            "dense_update_bfloat16_bits",
            "layer1_normalized_bfloat16_bits",
            "normalized_mlp_bfloat16_bits",
            "post_attention_residual_bfloat16_bits",
        }
        if set(payload.files) != expected_keys:
            raise ValueError("dense-convolution tensor keys drifted")
        candidate = np.ascontiguousarray(payload["normalized_mlp_bfloat16_bits"])
    if candidate.shape != (1, WIDTH) or candidate.dtype != np.dtype(np.uint16):
        raise ValueError("dense-convolution normalized-MLP shape drifted")
    candidate = candidate[0]
    if _array_sha256(candidate) != config.expected_probe_normalized_mlp_sha256:
        raise ValueError("dense-convolution normalized-MLP identity drifted")
    if not np.isfinite(_decode(candidate)).all():
        raise ValueError("dense-convolution normalized-MLP tensor is non-finite")
    comparison = _comparison(accepted, candidate)
    if comparison["elementwise_exact"]:
        classification = "normalized_mlp_exact_dense_arithmetic_open"
        first_open = "dense_mlp_or_cross_layer_fusion"
    else:
        classification = "normalized_mlp_nonexact"
        first_open = "post_attention_add_rmsnorm"
    result: dict[str, Any] = {
        "artifact_kind": COMPARISON_KIND,
        "accepted_capture_manifest_sha256": capture["manifest_sha256"],
        "classification": classification,
        "diagnostic_only": True,
        "first_open_boundary": first_open,
        "format_version": 1,
        "legacy_code_hash": config.expected_legacy_code_hash,
        "normalized_mlp": comparison,
        "oracle_pin": config.expected_oracle_pin,
        "performance_claim": False,
        "position": config.expected_position,
        "probe": {
            "code_hash": config.expected_probe_code_hash,
            "run_id": config.expected_probe_run_id,
            "runner_sha256": config.expected_probe_runner_sha256,
            "success_sha256": config.expected_probe_success_sha256,
            "summary_sha256": config.expected_probe_summary_sha256,
            "tag": config.expected_probe_tag,
            "tensor_sha256": config.expected_probe_tensor_sha256,
        },
        "status": "SUCCESS",
    }
    result["manifest_sha256"] = _manifest_sha256(result)
    config.output_dir.mkdir(parents=True)
    _atomic_json(config.output_dir / "comparison.json", result)
    return result
