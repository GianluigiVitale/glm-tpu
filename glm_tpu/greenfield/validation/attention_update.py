"""Seal the accepted layer-0 post-o_proj row and compare PP8 candidates."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np


SOURCE_KIND = "glm52_legacy_attention_update"
CAPTURE_KIND = "glm52_accepted_attention_update_capture"
COMPARISON_KIND = "glm52_accepted_greenfield_attention_update_comparison"
MODEL_ID = "zai-org/GLM-5.2-FP8"
LAYER_NAME = "model.layers.0.self_attn.attn"
POSITION = 8155
WIDTH = 6144
PROBE_KIND = "glm52_layer0_projection_reduction_probe"
PROBE_CODE_HASH = "e2a3a74a3b2ef1fa8f3b9cb1c5d7ec65f833eafc"
PROBE_TAG = "greenfield_layer0_projection_reduction_20260812T164052560787241Z"
PROBE_CLASSIFICATION = "projection_reduction_unresolved"
PROBE_RUN_ID = 538
ACCEPTED_ATTENTION_VALUE_SHA256 = (
    "79a6e290274ef470b518a6de894b44d2929ad6861d1751f0915aa4eb20cf2e9d"
)
ATTENTION_ARMS = {
    "local": (
        "local_attention_local_dense",
        "local_attention_strategy_dense",
    ),
    "strategy_nd": (
        "strategy_attention_local_dense",
        "strategy_attention_strategy_dense",
    ),
}
ARM_STRATEGIES = {
    "local_attention_local_dense": (False, False),
    "local_attention_strategy_dense": (False, True),
    "strategy_attention_local_dense": (True, False),
    "strategy_attention_strategy_dense": (True, True),
}


@dataclass(frozen=True, slots=True)
class AcceptedAttentionUpdateCaptureConfig:
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
class AttentionUpdateComparisonConfig:
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
        raise ValueError(f"attention-update field {name} is not scalar")
    return value.item()


def _decode(bits: np.ndarray) -> np.ndarray:
    if bits.dtype != np.dtype(np.uint16):
        raise ValueError("attention-update BF16 storage is not uint16")
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


def validate_exact_remote_object_set(
    root: Path,
    remote_prefix: str,
    remote_uris: list[str],
) -> None:
    """Refuse a protected archive with missing, duplicate, or stale objects."""

    expected = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    prefix = remote_prefix.rstrip("/") + "/"
    observed: list[str] = []
    for uri in remote_uris:
        value = uri.strip()
        if not value:
            continue
        if not value.startswith(prefix):
            raise ValueError(f"remote object escaped prefix: {value}")
        observed.append(value[len(prefix):])
    if len(observed) != len(set(observed)):
        raise ValueError("remote object listing contains duplicates")
    observed_set = set(observed)
    if observed_set != expected:
        missing = sorted(expected - observed_set)
        extra = sorted(observed_set - expected)
        raise ValueError(
            f"remote object set drifted: missing={missing} extra={extra}"
        )


def _load_source_rows(
    config: AcceptedAttentionUpdateCaptureConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    errors = sorted(config.source_dump_dir.rglob("*.INTERNAL.ERROR.*"))
    if errors:
        raise ValueError(f"attention-update error sentinel exists: {errors[0]}")
    safe_layer = config.expected_layer_name.replace("/", "_").replace(".", "_")
    paths = sorted(
        config.source_dump_dir.rglob(
            f"*.{safe_layer}.position{config.expected_position}.proc*.npz"
        )
    )
    if len(paths) != len(config.expected_capture_process_indices):
        raise ValueError("attention-update capture file count drifted")
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
        "attention_update",
        "attention_update__dtype",
    }
    canonical: np.ndarray | None = None
    process_indices: set[int] = set()
    records: list[dict[str, Any]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != expected_keys:
                raise ValueError(f"{path}: attention-update key set drifted")
            expected_scalars = {
                "artifact_kind": SOURCE_KIND,
                "format_version": 1,
                "capture_mode": "attention_update",
                "process_count": config.expected_process_count,
                "layer_name": config.expected_layer_name,
                "position": config.expected_position,
                "source_row": 0,
                "run_tag": config.expected_run_tag,
                "code_hash": config.expected_legacy_code_hash,
                "oracle_pin": config.expected_oracle_pin,
                "model_id": config.expected_model_id,
                "attention_update__dtype": "bfloat16",
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
            bits = np.ascontiguousarray(payload["attention_update"])
            if bits.shape != (WIDTH,) or bits.dtype != np.dtype(np.uint16):
                raise ValueError(f"{path}: attention-update tensor drifted")
            if not np.isfinite(_decode(bits)).all():
                raise ValueError(f"{path}: attention-update contains non-finite data")
            if canonical is None:
                canonical = bits.copy()
            elif not np.array_equal(canonical, bits):
                raise ValueError(f"{path}: replicated attention update disagrees")
            records.append({
                "byte_count": path.stat().st_size,
                "path": path.relative_to(config.source_dump_dir).as_posix(),
                "process_index": process_index,
                "sha256": _file_sha256(path),
            })
    if process_indices != set(config.expected_capture_process_indices) or canonical is None:
        raise ValueError("attention-update process coverage is incomplete")
    return canonical, sorted(records, key=lambda value: value["process_index"])


def capture_accepted_attention_update(
    config: AcceptedAttentionUpdateCaptureConfig,
) -> dict[str, Any]:
    """Validate raw observer output and seal one accepted BF16 row."""

    _require_digest(config.expected_legacy_code_hash, length=40, name="legacy code hash")
    _require_digest(config.expected_oracle_pin, length=40, name="oracle pin")
    if config.output_dir.exists():
        raise FileExistsError(f"append-only capture exists: {config.output_dir}")
    bits, process_files = _load_source_rows(config)
    config.output_dir.mkdir(parents=True)
    tensor_path = config.output_dir / "attention_update.npz"
    _atomic_npz(tensor_path, attention_update_bfloat16_bits=bits)
    manifest: dict[str, Any] = {
        "artifact_kind": CAPTURE_KIND,
        "capture_layout": "replicated_logical_live_row",
        "capture_mode": "attention_update",
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
    config: AttentionUpdateComparisonConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest_path = config.accepted_capture_dir / "capture.json"
    if _file_sha256(manifest_path) != config.expected_accepted_capture_file_sha256:
        raise ValueError("accepted attention-update manifest file hash drifted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != _manifest_sha256(manifest):
        raise ValueError("accepted attention-update manifest hash drifted")
    expected = {
        "artifact_kind": CAPTURE_KIND,
        "capture_layout": "replicated_logical_live_row",
        "capture_mode": "attention_update",
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
        raise ValueError("accepted attention-update manifest identity drifted")
    record = manifest.get("tensor", {})
    process_files = manifest.get("process_files")
    if (
        not isinstance(process_files, list)
        or len(process_files) != 1
        or set(process_files[0])
        != {"byte_count", "path", "process_index", "sha256"}
        or process_files[0].get("process_index") != 0
        or not isinstance(process_files[0].get("byte_count"), int)
        or process_files[0]["byte_count"] <= 0
        or not str(process_files[0].get("path", "")).endswith(".proc0.npz")
    ):
        raise ValueError("accepted attention-update process ledger drifted")
    _require_digest(
        str(process_files[0].get("sha256", "")),
        length=64,
        name="accepted attention-update source file",
    )
    tensor_path = config.accepted_capture_dir / "attention_update.npz"
    if (
        record.get("filename") != tensor_path.name
        or record.get("shape") != [WIDTH]
        or tensor_path.stat().st_size != record.get("byte_count")
        or _file_sha256(tensor_path) != record.get("file_sha256")
    ):
        raise ValueError("accepted attention-update tensor file drifted")
    with np.load(tensor_path, allow_pickle=False) as payload:
        if set(payload.files) != {"attention_update_bfloat16_bits"}:
            raise ValueError("accepted attention-update tensor keys drifted")
        bits = np.ascontiguousarray(payload["attention_update_bfloat16_bits"])
    if (
        bits.shape != (WIDTH,)
        or bits.dtype != np.dtype(np.uint16)
        or _array_sha256(bits) != record.get("tensor_sha256")
        or not np.isfinite(_decode(bits)).all()
    ):
        raise ValueError("accepted attention-update tensor contract drifted")
    return bits, manifest


def _comparison(expected: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    mismatch = expected != observed
    mismatch_count = int(np.count_nonzero(mismatch))
    error = np.abs(_decode(expected) - _decode(observed))
    return {
        "elementwise_exact": mismatch_count == 0,
        "expected_sha256": _array_sha256(expected),
        "first_mismatch_index": (
            int(np.flatnonzero(mismatch)[0]) if mismatch_count else None
        ),
        "max_abs_error": float(np.max(error)),
        "mean_abs_error": float(np.mean(error, dtype=np.float64)),
        "mismatch_count": mismatch_count,
        "observed_sha256": _array_sha256(observed),
        "shape": [WIDTH],
    }


def compare_attention_update_candidates(
    config: AttentionUpdateComparisonConfig,
) -> dict[str, Any]:
    """Compare the accepted post-o_proj row with both sealed DB538 arms."""

    for value, name in (
        (config.expected_accepted_capture_file_sha256, "capture manifest file"),
        (config.expected_probe_runner_sha256, "probe runner"),
        (config.expected_probe_tensor_sha256, "probe tensor"),
        (config.expected_probe_summary_sha256, "probe summary"),
        (config.expected_probe_success_sha256, "probe SUCCESS"),
    ):
        _require_digest(value, length=64, name=name)
    _require_digest(config.expected_legacy_code_hash, length=40, name="legacy code hash")
    _require_digest(config.expected_oracle_pin, length=40, name="oracle pin")
    _require_digest(config.expected_probe_code_hash, length=40, name="probe code hash")
    if config.output_dir.exists():
        raise FileExistsError(f"append-only comparison exists: {config.output_dir}")
    accepted, capture = _load_capture(config)
    if config.probe_dir.name != config.expected_probe_tag:
        raise ValueError("projection/reduction probe tag drifted")
    runner_path = config.probe_dir / "runner.json"
    tensor_path = config.probe_dir / "projection_reduction.npz"
    summary_path = config.probe_dir / "summary.json"
    success_path = config.probe_dir / "SUCCESS"
    if _file_sha256(runner_path) != config.expected_probe_runner_sha256:
        raise ValueError("projection/reduction runner file hash drifted")
    if _file_sha256(tensor_path) != config.expected_probe_tensor_sha256:
        raise ValueError("projection/reduction tensor file hash drifted")
    if _file_sha256(summary_path) != config.expected_probe_summary_sha256:
        raise ValueError("projection/reduction summary file hash drifted")
    if _file_sha256(success_path) != config.expected_probe_success_sha256:
        raise ValueError("projection/reduction SUCCESS file hash drifted")
    runner = json.loads(runner_path.read_text(encoding="utf-8"))
    expected_runner = {
        "artifact_kind": PROBE_KIND,
        "classification": PROBE_CLASSIFICATION,
        "code_hash": config.expected_probe_code_hash,
        "exact_arms": [],
        "performance_claim": False,
        "position": config.expected_position,
        "status": "SUCCESS",
    }
    if any(runner.get(name) != value for name, value in expected_runner.items()):
        raise ValueError("projection/reduction runner identity drifted")
    expected_arm_names = {
        arm for aliases in ATTENTION_ARMS.values() for arm in aliases
    }
    if set(runner.get("arms", {})) != expected_arm_names:
        raise ValueError("projection/reduction arm set drifted")
    for arm in expected_arm_names:
        record = runner["arms"][arm]
        expected_attention, expected_dense = ARM_STRATEGIES[arm]
        value_comparison = record.get("value_comparison", {})
        if (
            record.get("attention_strategy_nd") is not expected_attention
            or record.get("dense_strategy_nd") is not expected_dense
            or not record.get("hlo", {}).get("contract", {}).get("passed")
            or value_comparison
            != {
                "elementwise_exact": True,
                "expected_sha256": ACCEPTED_ATTENTION_VALUE_SHA256,
                "first_mismatch_index": None,
                "max_abs_error": 0.0,
                "mean_abs_error": 0.0,
                "mismatch_count": 0,
                "observed_sha256": ACCEPTED_ATTENTION_VALUE_SHA256,
                "shape": [16384],
            }
        ):
            raise ValueError(f"projection/reduction prerequisite failed: {arm}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_summary = {
        "artifact_kind": PROBE_KIND,
        "classification": PROBE_CLASSIFICATION,
        "code_hash": config.expected_probe_code_hash,
        "elapsed_seconds": 20,
        "exact_arms": [],
        "performance_claim": False,
        "results_db_run_id": config.expected_probe_run_id,
        "status": "SUCCESS",
    }
    if summary != expected_summary:
        raise ValueError("projection/reduction summary identity drifted")
    success_fields: dict[str, str] = {}
    for line in success_path.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition("=")
        if not separator or name in success_fields:
            raise ValueError("projection/reduction SUCCESS syntax drifted")
        success_fields[name] = value
    if (
        success_fields.get("artifact_kind") != PROBE_KIND
        or success_fields.get("code_hash") != config.expected_probe_code_hash
        or success_fields.get("results_db_run_id")
        != str(config.expected_probe_run_id)
        or success_fields.get("classification") != PROBE_CLASSIFICATION
        or success_fields.get("exact_arms") != "none"
        or success_fields.get("performance_claim") != "false"
    ):
        raise ValueError("projection/reduction terminal identity drifted")
    candidates: dict[str, np.ndarray] = {}
    with np.load(tensor_path, allow_pickle=False) as payload:
        for mechanism, arms in ATTENTION_ARMS.items():
            values = []
            for arm in arms:
                name = f"attention_update_bfloat16_bits__{arm}"
                if name not in payload.files:
                    raise ValueError(f"projection/reduction tensor is missing {name}")
                bits = np.ascontiguousarray(payload[name])
                if bits.shape != (1, WIDTH) or bits.dtype != np.dtype(np.uint16):
                    raise ValueError(f"projection/reduction tensor drifted: {name}")
                values.append(bits.reshape(WIDTH))
            if not np.array_equal(values[0], values[1]):
                raise ValueError(
                    f"{mechanism} attention update changed with dense-only choice"
                )
            candidates[mechanism] = values[0]
    comparisons = {
        name: _comparison(accepted, value) for name, value in candidates.items()
    }
    exact = [name for name, value in comparisons.items() if value["elementwise_exact"]]
    if exact == ["local"]:
        classification = "local_attention_projection_exact"
    elif exact == ["strategy_nd"]:
        classification = "strategy_nd_attention_projection_exact"
    elif set(exact) == {"local", "strategy_nd"}:
        classification = "both_attention_projections_exact"
    else:
        classification = "attention_projection_arithmetic_unresolved"
    result: dict[str, Any] = {
        "artifact_kind": COMPARISON_KIND,
        "accepted_capture": {
            "file_sha256": config.expected_accepted_capture_file_sha256,
            "manifest_sha256": capture["manifest_sha256"],
            "run_tag": config.expected_accepted_run_tag,
            "tensor_sha256": _array_sha256(accepted),
        },
        "candidate_comparisons": comparisons,
        "classification": classification,
        "diagnostic_only": True,
        "exact_candidates": exact,
        "first_open_boundary": (
            "after_attention_projection" if exact else "attention_projection"
        ),
        "format_version": 1,
        "performance_claim": False,
        "position": config.expected_position,
        "probe": {
            "code_hash": config.expected_probe_code_hash,
            "runner_sha256": config.expected_probe_runner_sha256,
            "results_db_run_id": config.expected_probe_run_id,
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
