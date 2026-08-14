"""Seal the accepted layer-0 dense boundary and classify DB540."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np


SOURCE_KIND = "glm52_legacy_dense_boundary"
CAPTURE_KIND = "glm52_accepted_dense_boundary_capture"
COMPARISON_KIND = "glm52_accepted_greenfield_dense_boundary_comparison"
MODEL_ID = "zai-org/GLM-5.2-FP8"
LAYER_NAME = "model.layers.0.self_attn.attn"
POSITION = 8155
WIDTH = 6144
PROBE_KIND = "glm52_layer0_dense_convolution_probe"
PROBE_CODE_HASH = "2f63779309b25c71c1cc7d35ff97715ae4bf631e"
PROBE_TAG = "greenfield_layer0_dense_convolution_20260813T005213127235575Z"
PROBE_CLASSIFICATION = "accepted_dense_convolution_nonexact"
PROBE_RUN_ID = 540
PROBE_DENSE_UPDATE_SHA256 = (
    "efde853254c03dd18a5f5f22733630ce0e785dfbb4eba09c41eea9085e47b4fc"
)
PROBE_POST_ATTENTION_RESIDUAL_SHA256 = (
    "a105fdbd429adb1d06a70bf71598a72a91d7b6faa83360005487ce11ce099f8e"
)
ACCEPTED_LAYER1_SHA256 = (
    "9936ee1e19049b297fd205292ebc378aee41d59401bbf56497004356998d3039"
)
PROBE_LAYER1_SHA256 = (
    "229dc8ace9bfa31fce6d6ccabc9fca49ccc55f30b9d1dd6f97a032f5117b812f"
)


@dataclass(frozen=True, slots=True)
class AcceptedDenseBoundaryCaptureConfig:
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
class DenseBoundaryComparisonConfig:
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
    expected_probe_dense_update_sha256: str = PROBE_DENSE_UPDATE_SHA256
    expected_probe_post_attention_residual_sha256: str = (
        PROBE_POST_ATTENTION_RESIDUAL_SHA256
    )
    expected_accepted_layer1_sha256: str = ACCEPTED_LAYER1_SHA256
    expected_probe_layer1_sha256: str = PROBE_LAYER1_SHA256


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
        raise ValueError(f"dense-boundary field {name} is not scalar")
    return value.item()


def _decode(bits: np.ndarray) -> np.ndarray:
    if bits.dtype != np.dtype(np.uint16):
        raise ValueError("dense-boundary BF16 storage is not uint16")
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
    config: AcceptedDenseBoundaryCaptureConfig,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    errors = sorted(config.source_dump_dir.rglob("*.INTERNAL.ERROR.*"))
    if errors:
        raise ValueError(f"dense-boundary error sentinel exists: {errors[0]}")
    safe_layer = config.expected_layer_name.replace("/", "_").replace(".", "_")
    paths = sorted(
        config.source_dump_dir.rglob(
            f"*.{safe_layer}.position{config.expected_position}.proc*.npz"
        )
    )
    if len(paths) != len(config.expected_capture_process_indices):
        raise ValueError("dense-boundary capture file count drifted")
    tensor_names = ("dense_update", "post_attention_residual")
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
        *(tensor_names),
        *(f"{name}__dtype" for name in tensor_names),
    }
    canonical: dict[str, np.ndarray] | None = None
    process_indices: set[int] = set()
    records: list[dict[str, Any]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != expected_keys:
                raise ValueError(f"{path}: dense-boundary key set drifted")
            expected_scalars = {
                "artifact_kind": SOURCE_KIND,
                "format_version": 1,
                "capture_mode": "dense_boundary",
                "process_count": config.expected_process_count,
                "layer_name": config.expected_layer_name,
                "position": config.expected_position,
                "source_row": 0,
                "run_tag": config.expected_run_tag,
                "code_hash": config.expected_legacy_code_hash,
                "oracle_pin": config.expected_oracle_pin,
                "model_id": config.expected_model_id,
                "dense_update__dtype": "bfloat16",
                "post_attention_residual__dtype": "bfloat16",
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
            tensors = {
                name: np.ascontiguousarray(payload[name]) for name in tensor_names
            }
            for name, bits in tensors.items():
                if bits.shape != (WIDTH,) or bits.dtype != np.dtype(np.uint16):
                    raise ValueError(f"{path}: {name} tensor drifted")
                if not np.isfinite(_decode(bits)).all():
                    raise ValueError(f"{path}: {name} contains non-finite data")
            if canonical is None:
                canonical = {name: bits.copy() for name, bits in tensors.items()}
            elif any(
                not np.array_equal(canonical[name], tensors[name])
                for name in tensor_names
            ):
                raise ValueError(f"{path}: replicated dense boundary disagrees")
            records.append({
                "byte_count": path.stat().st_size,
                "path": path.relative_to(config.source_dump_dir).as_posix(),
                "process_index": process_index,
                "sha256": _file_sha256(path),
            })
    if process_indices != set(config.expected_capture_process_indices) or canonical is None:
        raise ValueError("dense-boundary process coverage is incomplete")
    return canonical, sorted(records, key=lambda value: value["process_index"])


def capture_accepted_dense_boundary(
    config: AcceptedDenseBoundaryCaptureConfig,
) -> dict[str, Any]:
    """Validate raw observer output and seal both accepted BF16 rows."""

    _require_digest(config.expected_legacy_code_hash, length=40, name="legacy code hash")
    _require_digest(config.expected_oracle_pin, length=40, name="oracle pin")
    if config.output_dir.exists():
        raise FileExistsError(f"append-only capture exists: {config.output_dir}")
    tensors, process_files = _load_source_rows(config)
    config.output_dir.mkdir(parents=True)
    tensor_path = config.output_dir / "dense_boundary.npz"
    arrays = {
        f"{name}_bfloat16_bits": bits for name, bits in tensors.items()
    }
    _atomic_npz(tensor_path, **arrays)
    records = {
        name: {
            "shape": [WIDTH],
            "sha256": _array_sha256(bits),
        }
        for name, bits in tensors.items()
    }
    manifest: dict[str, Any] = {
        "artifact_kind": CAPTURE_KIND,
        "capture_layout": "replicated_logical_live_rows",
        "capture_mode": "dense_boundary",
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
        "tensor_file": {
            "byte_count": tensor_path.stat().st_size,
            "filename": tensor_path.name,
            "sha256": _file_sha256(tensor_path),
        },
        "tensors": records,
    }
    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    _atomic_json(config.output_dir / "capture.json", manifest)
    return manifest


def _load_capture(
    config: DenseBoundaryComparisonConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest_path = config.accepted_capture_dir / "capture.json"
    if _file_sha256(manifest_path) != config.expected_accepted_capture_file_sha256:
        raise ValueError("accepted dense-boundary manifest file hash drifted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != _manifest_sha256(manifest):
        raise ValueError("accepted dense-boundary manifest hash drifted")
    expected = {
        "artifact_kind": CAPTURE_KIND,
        "capture_layout": "replicated_logical_live_rows",
        "capture_mode": "dense_boundary",
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
        raise ValueError("accepted dense-boundary manifest identity drifted")
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
        raise ValueError("accepted dense-boundary process ledger drifted")
    _require_digest(str(process_files[0].get("sha256", "")), length=64, name="source file")
    tensor_path = config.accepted_capture_dir / "dense_boundary.npz"
    file_record = manifest.get("tensor_file", {})
    if (
        file_record.get("filename") != tensor_path.name
        or tensor_path.stat().st_size != file_record.get("byte_count")
        or _file_sha256(tensor_path) != file_record.get("sha256")
    ):
        raise ValueError("accepted dense-boundary tensor file drifted")
    names = ("dense_update", "post_attention_residual")
    with np.load(tensor_path, allow_pickle=False) as payload:
        if set(payload.files) != {f"{name}_bfloat16_bits" for name in names}:
            raise ValueError("accepted dense-boundary tensor keys drifted")
        tensors = {
            name: np.ascontiguousarray(payload[f"{name}_bfloat16_bits"])
            for name in names
        }
    for name, bits in tensors.items():
        record = manifest.get("tensors", {}).get(name, {})
        if (
            bits.shape != (WIDTH,)
            or bits.dtype != np.dtype(np.uint16)
            or record.get("shape") != [WIDTH]
            or _array_sha256(bits) != record.get("sha256")
            or not np.isfinite(_decode(bits)).all()
        ):
            raise ValueError(f"accepted dense-boundary {name} contract drifted")
    if set(manifest.get("tensors", {})) != set(names):
        raise ValueError("accepted dense-boundary tensor ledger drifted")
    return tensors, manifest


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


def compare_dense_boundary_candidate(
    config: DenseBoundaryComparisonConfig,
) -> dict[str, Any]:
    """Locate the first open boundary relative to protected DB540."""

    for value, name in (
        (config.expected_accepted_capture_file_sha256, "capture manifest file"),
        (config.expected_probe_runner_sha256, "probe runner"),
        (config.expected_probe_tensor_sha256, "probe tensor"),
        (config.expected_probe_summary_sha256, "probe summary"),
        (config.expected_probe_success_sha256, "probe SUCCESS"),
        (config.expected_probe_dense_update_sha256, "probe dense update"),
        (
            config.expected_probe_post_attention_residual_sha256,
            "probe post-attention residual",
        ),
        (config.expected_accepted_layer1_sha256, "accepted layer1"),
        (config.expected_probe_layer1_sha256, "probe layer1"),
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
    for name, expected in (
        ("runner", config.expected_probe_runner_sha256),
        ("tensor", config.expected_probe_tensor_sha256),
        ("summary", config.expected_probe_summary_sha256),
        ("success", config.expected_probe_success_sha256),
    ):
        if _file_sha256(paths[name]) != expected:
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
    layer1 = runner.get("layer1_comparison")
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
    success_lines = paths["success"].read_text(encoding="utf-8").splitlines()
    success_fields = dict(line.split("=", 1) for line in success_lines)
    if (
        len(success_fields) != len(success_lines)
        or success_fields.get("artifact_kind") != PROBE_KIND
        or success_fields.get("code_hash") != config.expected_probe_code_hash
        or success_fields.get("results_db_run_id")
        != str(config.expected_probe_run_id)
        or success_fields.get("classification") != PROBE_CLASSIFICATION
        or success_fields.get("exact_arms") != "none"
        or success_fields.get("performance_claim") != "false"
        or success_fields.get("remote_prefix")
        != f"gs://driftbench-dsv4-uc/results/{config.expected_probe_tag}"
    ):
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
        candidate_dense = np.ascontiguousarray(
            payload["dense_update_bfloat16_bits"]
        )
        candidate_residual = np.ascontiguousarray(
            payload["post_attention_residual_bfloat16_bits"]
        )
        accepted_layer1 = np.ascontiguousarray(
            payload["accepted_layer1_normalized_bfloat16_bits"]
        )
        candidate_layer1 = np.ascontiguousarray(
            payload["layer1_normalized_bfloat16_bits"]
        )
    if (
        candidate_dense.shape != (1, WIDTH)
        or candidate_residual.shape != (1, WIDTH)
        or accepted_layer1.shape != (WIDTH,)
        or candidate_layer1.shape != (WIDTH,)
        or any(
            value.dtype != np.dtype(np.uint16)
            for value in (
                candidate_dense,
                candidate_residual,
                accepted_layer1,
                candidate_layer1,
            )
        )
    ):
        raise ValueError("dense-convolution candidate shapes drifted")
    candidate_dense = candidate_dense[0]
    candidate_residual = candidate_residual[0]
    if _array_sha256(candidate_dense) != config.expected_probe_dense_update_sha256:
        raise ValueError("dense-convolution dense-update identity drifted")
    if (
        _array_sha256(candidate_residual)
        != config.expected_probe_post_attention_residual_sha256
    ):
        raise ValueError("dense-convolution residual identity drifted")
    if _array_sha256(accepted_layer1) != config.expected_accepted_layer1_sha256:
        raise ValueError("dense-convolution accepted layer1 identity drifted")
    if _array_sha256(candidate_layer1) != config.expected_probe_layer1_sha256:
        raise ValueError("dense-convolution candidate layer1 identity drifted")
    if not all(
        np.isfinite(_decode(value)).all()
        for value in (
            candidate_dense[0],
            candidate_residual[0],
            accepted_layer1,
            candidate_layer1,
        )
    ):
        raise ValueError("dense-convolution candidate contains non-finite data")
    observed_layer1 = _comparison(accepted_layer1, candidate_layer1)
    if observed_layer1["elementwise_exact"] or layer1 != observed_layer1:
        raise ValueError("dense-convolution layer1 verdict drifted")
    residual = _comparison(accepted["post_attention_residual"], candidate_residual)
    dense = _comparison(accepted["dense_update"], candidate_dense)
    if not residual["elementwise_exact"]:
        classification = "post_attention_residual_nonexact"
        first_open = "layer0_post_attention_residual"
    elif dense["elementwise_exact"]:
        classification = "dense_update_exact_layer1_fused_norm_open"
        first_open = "layer1_fused_add_rmsnorm"
    else:
        classification = "dense_mlp_output_nonexact"
        first_open = "dense_mlp_input_or_arithmetic"
    result: dict[str, Any] = {
        "artifact_kind": COMPARISON_KIND,
        "accepted_capture_manifest_sha256": capture["manifest_sha256"],
        "classification": classification,
        "dense_update": dense,
        "diagnostic_only": True,
        "first_open_boundary": first_open,
        "format_version": 1,
        "legacy_code_hash": config.expected_legacy_code_hash,
        "oracle_pin": config.expected_oracle_pin,
        "performance_claim": False,
        "position": config.expected_position,
        "post_attention_residual": residual,
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
