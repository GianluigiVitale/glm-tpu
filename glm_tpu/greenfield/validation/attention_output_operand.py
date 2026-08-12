"""Seal and compare the layer-0 post-W_UV/pre-o_proj BF16 operand."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np


SOURCE_KIND = "glm52_legacy_attention_output_operand"
CAPTURE_KIND = "glm52_accepted_attention_output_operand_capture"
COMPARISON_KIND = "glm52_accepted_greenfield_attention_output_comparison"
PROJECTION_SOURCE_KIND = "glm52_legacy_attention_projection_operands"
PROJECTION_CAPTURE_KIND = "glm52_accepted_attention_projection_capture"
PROJECTION_COMPARISON_KIND = (
    "glm52_accepted_greenfield_attention_projection_comparison"
)
MODEL_ID = "zai-org/GLM-5.2-FP8"
LAYER_NAME = "model.layers.0.self_attn.attn"
POSITION = 8155
WIDTH = 16_384
ATTENDED_LATENT_SHAPE = (64, 512)
MAIN_ROPE_TABLE_SHA256 = (
    "6a22140fc2aec475399738c6fc0f29be2a6c419feb0249aee35681c607c80701"
)


@dataclass(frozen=True, slots=True)
class AcceptedAttentionOutputCaptureConfig:
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
class AttentionOutputComparisonConfig:
    accepted_capture_dir: Path
    greenfield_ingredients_dir: Path
    output_dir: Path
    expected_accepted_capture_file_sha256: str
    expected_accepted_run_tag: str
    expected_ingredients_contract_sha256: str
    expected_ingredients_tensor_sha256: str
    expected_greenfield_code_hash: str
    expected_greenfield_run_tag: str
    expected_legacy_code_hash: str
    expected_oracle_pin: str
    expected_main_rope_table_sha256: str = MAIN_ROPE_TABLE_SHA256
    expected_position: int = POSITION

    def __post_init__(self) -> None:
        if not self.expected_accepted_run_tag.strip():
            raise ValueError("expected accepted run tag must be non-empty")
        if not self.expected_greenfield_run_tag.strip():
            raise ValueError("expected greenfield run tag must be non-empty")


@dataclass(frozen=True, slots=True)
class AcceptedAttentionProjectionCaptureConfig:
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
class AttentionProjectionComparisonConfig:
    accepted_capture_dir: Path
    greenfield_ingredients_dir: Path
    output_dir: Path
    expected_accepted_capture_file_sha256: str
    expected_accepted_run_tag: str
    expected_ingredients_contract_sha256: str
    expected_ingredients_tensor_sha256: str
    expected_greenfield_code_hash: str
    expected_greenfield_run_tag: str
    expected_legacy_code_hash: str
    expected_oracle_pin: str
    expected_main_rope_table_sha256: str = MAIN_ROPE_TABLE_SHA256
    expected_position: int = POSITION

    def __post_init__(self) -> None:
        if not self.expected_accepted_run_tag.strip():
            raise ValueError("expected accepted run tag must be non-empty")
        if not self.expected_greenfield_run_tag.strip():
            raise ValueError("expected greenfield run tag must be non-empty")


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


def _require_digest(value: str, *, lengths: tuple[int, ...], name: str) -> None:
    if len(value) not in lengths or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} is not a lowercase digest")


def _scalar(payload: Any, name: str) -> Any:
    value = payload[name]
    if value.shape != ():
        raise ValueError(f"attention-output field {name} is not scalar")
    return value.item()


def _decode_bfloat16(bits: np.ndarray) -> np.ndarray:
    if bits.dtype != np.dtype(np.uint16):
        raise ValueError("attention-output BF16 storage is not uint16")
    return np.ascontiguousarray(bits).view(ml_dtypes.bfloat16)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, path)


def _load_source_rows(
    config: AcceptedAttentionOutputCaptureConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    errors = sorted(config.source_dump_dir.rglob("*.INTERNAL.ERROR.*"))
    if errors:
        raise ValueError(f"attention-output error sentinel exists: {errors[0]}")
    safe_layer = config.expected_layer_name.replace("/", "_").replace(".", "_")
    paths = sorted(
        config.source_dump_dir.rglob(
            f"*.{safe_layer}.position{config.expected_position}.proc*.npz"
        )
    )
    if len(paths) != len(config.expected_capture_process_indices):
        raise ValueError("attention-output capture file count drifted")
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
        "attention_output",
        "attention_output__dtype",
    }
    canonical: np.ndarray | None = None
    process_indices: set[int] = set()
    records: list[dict[str, Any]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != expected_keys:
                raise ValueError(f"{path}: attention-output key set drifted")
            expected_scalars = {
                "artifact_kind": SOURCE_KIND,
                "format_version": 1,
                "capture_mode": "attention_output",
                "process_count": config.expected_process_count,
                "layer_name": config.expected_layer_name,
                "position": config.expected_position,
                "source_row": 0,
                "run_tag": config.expected_run_tag,
                "code_hash": config.expected_legacy_code_hash,
                "oracle_pin": config.expected_oracle_pin,
                "model_id": config.expected_model_id,
                "attention_output__dtype": "bfloat16",
            }
            for name, expected in expected_scalars.items():
                observed = _scalar(payload, name)
                if observed != expected:
                    raise ValueError(
                        f"{path}: {name}={observed!r} != {expected!r}"
                    )
            process_index = int(_scalar(payload, "process_index"))
            if process_index in process_indices or not (
                0 <= process_index < config.expected_process_count
            ):
                raise ValueError(f"{path}: invalid/duplicate process index")
            process_indices.add(process_index)
            bits = np.ascontiguousarray(payload["attention_output"])
            if bits.shape != (WIDTH,) or bits.dtype != np.dtype(np.uint16):
                raise ValueError(f"{path}: attention-output tensor drifted")
            if not np.isfinite(_decode_bfloat16(bits).astype(np.float32)).all():
                raise ValueError(f"{path}: attention-output contains non-finite data")
            if canonical is None:
                canonical = bits.copy()
            elif not np.array_equal(canonical, bits):
                raise ValueError(f"{path}: replicated attention output disagrees")
            records.append(
                {
                    "byte_count": path.stat().st_size,
                    "path": path.relative_to(config.source_dump_dir).as_posix(),
                    "process_index": process_index,
                    "sha256": _file_sha256(path),
                }
            )
    if process_indices != set(config.expected_capture_process_indices) or (
        canonical is None
    ):
        raise ValueError("attention-output process coverage is incomplete")
    return canonical, sorted(records, key=lambda value: value["process_index"])


def capture_accepted_attention_output_operand(
    config: AcceptedAttentionOutputCaptureConfig,
) -> dict[str, Any]:
    """Validate raw observer files and seal the canonical accepted row."""

    _require_digest(
        config.expected_legacy_code_hash, lengths=(40,), name="legacy code hash"
    )
    _require_digest(config.expected_oracle_pin, lengths=(40,), name="oracle pin")
    if config.output_dir.exists():
        raise FileExistsError(f"append-only capture exists: {config.output_dir}")
    bits, process_files = _load_source_rows(config)
    config.output_dir.mkdir(parents=True)
    tensor_path = config.output_dir / "attention_output.npz"
    _atomic_npz(tensor_path, attention_output_bfloat16_bits=bits)
    manifest: dict[str, Any] = {
        "artifact_kind": CAPTURE_KIND,
        "capture_layout": "logical_head_order_live_row",
        "capture_mode": "attention_output",
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
            "sha256": _file_sha256(tensor_path),
            "tensor_sha256": _array_sha256(bits),
        },
    }
    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    _atomic_json(config.output_dir / "capture.json", manifest)
    return manifest


def _load_accepted_capture(
    config: AttentionOutputComparisonConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    path = config.accepted_capture_dir / "capture.json"
    if _file_sha256(path) != config.expected_accepted_capture_file_sha256:
        raise ValueError("accepted attention-output manifest file hash drifted")
    manifest = json.loads(path.read_text())
    if manifest.get("manifest_sha256") != _manifest_sha256(manifest):
        raise ValueError("accepted attention-output manifest hash drifted")
    expected = {
        "artifact_kind": CAPTURE_KIND,
        "capture_layout": "logical_head_order_live_row",
        "capture_mode": "attention_output",
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
        raise ValueError("accepted attention-output manifest identity drifted")
    if manifest.get("capture_process_indices") != [0]:
        raise ValueError("accepted attention-output process ownership drifted")
    process_files = manifest.get("process_files")
    if (
        not isinstance(process_files, list)
        or len(process_files) != 1
        or process_files[0].get("process_index") != 0
    ):
        raise ValueError("accepted attention-output source ledger drifted")
    record = manifest.get("tensor", {})
    if record.get("filename") != "attention_output.npz":
        raise ValueError("accepted attention-output tensor filename drifted")
    tensor_path = config.accepted_capture_dir / "attention_output.npz"
    if (
        tensor_path.stat().st_size != record.get("byte_count")
        or _file_sha256(tensor_path) != record.get("sha256")
    ):
        raise ValueError("accepted attention-output tensor file drifted")
    with np.load(tensor_path, allow_pickle=False) as payload:
        if set(payload.files) != {"attention_output_bfloat16_bits"}:
            raise ValueError("accepted attention-output tensor keys drifted")
        bits = np.ascontiguousarray(payload["attention_output_bfloat16_bits"])
    if (
        bits.shape != (WIDTH,)
        or bits.dtype != np.dtype(np.uint16)
        or _array_sha256(bits) != record.get("tensor_sha256")
    ):
        raise ValueError("accepted attention-output tensor contract drifted")
    return bits, manifest


def _load_greenfield_operand(
    config: AttentionOutputComparisonConfig,
) -> tuple[np.ndarray, dict[str, Any], dict[str, str]]:
    contract_path = config.greenfield_ingredients_dir / "contract.json"
    tensor_path = (
        config.greenfield_ingredients_dir
        / f"position_{config.expected_position}_ingredients.npz"
    )
    if _file_sha256(contract_path) != config.expected_ingredients_contract_sha256:
        raise ValueError("greenfield ingredient contract hash drifted")
    if _file_sha256(tensor_path) != config.expected_ingredients_tensor_sha256:
        raise ValueError("greenfield ingredient tensor hash drifted")
    contract = json.loads(contract_path.read_text())
    expected_contract = {
        "code_hash": config.expected_greenfield_code_hash,
        "decode_position": config.expected_position,
        "main_rope_table_enabled": True,
        "main_rope_table_sha256": config.expected_main_rope_table_sha256,
        "passed": True,
        "run_tag": config.expected_greenfield_run_tag,
        "source_state": "post_teacher_forced_prefill",
    }
    if any(contract.get(name) != value for name, value in expected_contract.items()):
        raise ValueError("greenfield ingredient identity drifted")
    if contract.get("active_rows") != [0, 1, 2, 3]:
        raise ValueError("greenfield ingredient owner order drifted")
    main_rope = contract.get("hlo_contract", {}).get(
        "main_rope_table_contract", {}
    )
    if (
        not main_rope.get("passed")
        or main_rope.get("table_parameter_count") != 1
        or main_rope.get("named_table_parameter_count") != 1
    ):
        raise ValueError("greenfield ingredient main-RoPE HLO drifted")
    with np.load(tensor_path, allow_pickle=False) as payload:
        if "attention_output_input_bfloat16_bits" not in payload.files:
            raise ValueError("greenfield attention operand is absent")
        owner_bits = np.ascontiguousarray(
            payload["attention_output_input_bfloat16_bits"]
        )
    if owner_bits.shape != (4, 4096) or owner_bits.dtype != np.dtype(np.uint16):
        raise ValueError("greenfield attention operand owner shape drifted")
    if not np.isfinite(_decode_bfloat16(owner_bits).astype(np.float32)).all():
        raise ValueError("greenfield attention operand contains non-finite data")
    record = contract.get("arrays", {}).get(
        "attention_output_input_bfloat16_bits", {}
    )
    if (
        record.get("shape") != [4, 4096]
        or record.get("dtype") != "uint16"
        or record.get("sha256") != _array_sha256(owner_bits)
    ):
        raise ValueError("greenfield attention operand array ledger drifted")
    logical = np.ascontiguousarray(owner_bits.reshape(WIDTH))
    return logical, contract, {
        "contract_sha256": config.expected_ingredients_contract_sha256,
        "tensor_sha256": config.expected_ingredients_tensor_sha256,
    }


def compare_attention_output_operands(
    config: AttentionOutputComparisonConfig,
) -> dict[str, Any]:
    """Compare the accepted logical row with PP8 owners in head order."""

    for value, lengths, name in (
        (
            config.expected_accepted_capture_file_sha256,
            (64,),
            "capture manifest file",
        ),
        (config.expected_ingredients_contract_sha256, (64,), "contract"),
        (config.expected_ingredients_tensor_sha256, (64,), "tensor"),
        (config.expected_greenfield_code_hash, (40,), "greenfield code hash"),
        (config.expected_legacy_code_hash, (40,), "legacy code hash"),
        (config.expected_oracle_pin, (40,), "oracle pin"),
        (config.expected_main_rope_table_sha256, (64,), "main-RoPE table"),
    ):
        _require_digest(value, lengths=lengths, name=name)
    if config.output_dir.exists():
        raise FileExistsError(f"append-only comparison exists: {config.output_dir}")
    accepted, accepted_manifest = _load_accepted_capture(config)
    greenfield, _, ingredient_files = _load_greenfield_operand(
        config
    )
    mismatch = accepted != greenfield
    mismatch_indices = np.flatnonzero(mismatch)
    exact = bool(not mismatch_indices.size)
    delta = np.abs(
        _decode_bfloat16(accepted).astype(np.float32)
        - _decode_bfloat16(greenfield).astype(np.float32)
    )
    result: dict[str, Any] = {
        "accepted": {
            "manifest_file_sha256": (
                config.expected_accepted_capture_file_sha256
            ),
            "manifest_sha256": accepted_manifest["manifest_sha256"],
            "run_tag": accepted_manifest["run_tag"],
            "sha256": _array_sha256(accepted),
        },
        "artifact_kind": COMPARISON_KIND,
        "classification": (
            "projection_partial_arithmetic"
            if exact
            else "attention_arithmetic_before_output_projection"
        ),
        "diagnostic_only": True,
        "elementwise_exact": exact,
        "first_mismatch_index": (
            int(mismatch_indices[0]) if mismatch_indices.size else None
        ),
        "format_version": 1,
        "greenfield": {
            "code_hash": config.expected_greenfield_code_hash,
            "ingredient_files": ingredient_files,
            "run_tag": config.expected_greenfield_run_tag,
            "sha256": _array_sha256(greenfield),
        },
        "layer_name": LAYER_NAME,
        "max_abs_error": float(np.max(delta)),
        "mean_abs_error": float(np.mean(delta)),
        "mismatch_count": int(mismatch.sum()),
        "model_id": MODEL_ID,
        "oracle_pin": config.expected_oracle_pin,
        "performance_claim": False,
        "position": config.expected_position,
        "status": "SUCCESS",
        "width": WIDTH,
    }
    result["manifest_sha256"] = _manifest_sha256(result)
    config.output_dir.mkdir(parents=True)
    _atomic_json(config.output_dir / "comparison.json", result)
    return result


def _load_projection_source_rows(
    config: AcceptedAttentionProjectionCaptureConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    errors = sorted(config.source_dump_dir.rglob("*.INTERNAL.ERROR.*"))
    if errors:
        raise ValueError(f"attention-projection error sentinel exists: {errors[0]}")
    safe_layer = config.expected_layer_name.replace("/", "_").replace(".", "_")
    paths = sorted(
        config.source_dump_dir.rglob(
            f"*.{safe_layer}.position{config.expected_position}.proc*.npz"
        )
    )
    if len(paths) != len(config.expected_capture_process_indices):
        raise ValueError("attention-projection capture file count drifted")
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
        "attended_latent",
        "attended_latent__dtype",
        "attention_output",
        "attention_output__dtype",
    }
    canonical_latent: np.ndarray | None = None
    canonical_output: np.ndarray | None = None
    process_indices: set[int] = set()
    records: list[dict[str, Any]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != expected_keys:
                raise ValueError(f"{path}: attention-projection key set drifted")
            expected_scalars = {
                "artifact_kind": PROJECTION_SOURCE_KIND,
                "format_version": 1,
                "capture_mode": "attention_projection",
                "process_count": config.expected_process_count,
                "layer_name": config.expected_layer_name,
                "position": config.expected_position,
                "source_row": 0,
                "run_tag": config.expected_run_tag,
                "code_hash": config.expected_legacy_code_hash,
                "oracle_pin": config.expected_oracle_pin,
                "model_id": config.expected_model_id,
                "attended_latent__dtype": "bfloat16",
                "attention_output__dtype": "bfloat16",
            }
            for name, expected in expected_scalars.items():
                observed = _scalar(payload, name)
                if observed != expected:
                    raise ValueError(
                        f"{path}: {name}={observed!r} != {expected!r}"
                    )
            process_index = int(_scalar(payload, "process_index"))
            if process_index in process_indices or not (
                0 <= process_index < config.expected_process_count
            ):
                raise ValueError(f"{path}: invalid/duplicate process index")
            process_indices.add(process_index)
            latent_bits = np.ascontiguousarray(payload["attended_latent"])
            output_bits = np.ascontiguousarray(payload["attention_output"])
            for bits, shape, name in (
                (latent_bits, ATTENDED_LATENT_SHAPE, "attended latent"),
                (output_bits, (WIDTH,), "attention output"),
            ):
                if bits.shape != shape or bits.dtype != np.dtype(np.uint16):
                    raise ValueError(f"{path}: {name} tensor drifted")
                if not np.isfinite(
                    _decode_bfloat16(bits).astype(np.float32)
                ).all():
                    raise ValueError(f"{path}: {name} contains non-finite data")
            if canonical_latent is None:
                canonical_latent = latent_bits.copy()
                canonical_output = output_bits.copy()
            elif not (
                np.array_equal(canonical_latent, latent_bits)
                and np.array_equal(canonical_output, output_bits)
            ):
                raise ValueError(f"{path}: replicated projection operands disagree")
            records.append(
                {
                    "byte_count": path.stat().st_size,
                    "path": path.relative_to(config.source_dump_dir).as_posix(),
                    "process_index": process_index,
                    "sha256": _file_sha256(path),
                }
            )
    if (
        process_indices != set(config.expected_capture_process_indices)
        or canonical_latent is None
        or canonical_output is None
    ):
        raise ValueError("attention-projection process coverage is incomplete")
    return (
        canonical_latent,
        canonical_output,
        sorted(records, key=lambda value: value["process_index"]),
    )


def capture_accepted_attention_projection_operands(
    config: AcceptedAttentionProjectionCaptureConfig,
) -> dict[str, Any]:
    """Seal the actual accepted values immediately before and after ``W_UV``."""

    _require_digest(
        config.expected_legacy_code_hash, lengths=(40,), name="legacy code hash"
    )
    _require_digest(config.expected_oracle_pin, lengths=(40,), name="oracle pin")
    if config.output_dir.exists():
        raise FileExistsError(f"append-only capture exists: {config.output_dir}")
    latent_bits, output_bits, process_files = _load_projection_source_rows(config)
    config.output_dir.mkdir(parents=True)
    tensor_path = config.output_dir / "attention_projection.npz"
    _atomic_npz(
        tensor_path,
        attended_latent_bfloat16_bits=latent_bits,
        attention_output_bfloat16_bits=output_bits,
    )
    manifest: dict[str, Any] = {
        "artifact_kind": PROJECTION_CAPTURE_KIND,
        "capture_layout": "logical_head_order_live_row",
        "capture_mode": "attention_projection",
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
        "tensors": {
            "attended_latent_bfloat16_bits": {
                "dtype": "uint16",
                "sha256": _array_sha256(latent_bits),
                "shape": list(ATTENDED_LATENT_SHAPE),
            },
            "attention_output_bfloat16_bits": {
                "dtype": "uint16",
                "sha256": _array_sha256(output_bits),
                "shape": [WIDTH],
            },
        },
    }
    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    _atomic_json(config.output_dir / "capture.json", manifest)
    return manifest


def _load_accepted_projection_capture(
    config: AttentionProjectionComparisonConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = config.accepted_capture_dir / "capture.json"
    if _file_sha256(path) != config.expected_accepted_capture_file_sha256:
        raise ValueError("accepted attention-projection manifest file hash drifted")
    manifest = json.loads(path.read_text())
    if manifest.get("manifest_sha256") != _manifest_sha256(manifest):
        raise ValueError("accepted attention-projection manifest hash drifted")
    expected = {
        "artifact_kind": PROJECTION_CAPTURE_KIND,
        "capture_layout": "logical_head_order_live_row",
        "capture_mode": "attention_projection",
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
        raise ValueError("accepted attention-projection manifest identity drifted")
    if manifest.get("capture_process_indices") != [0]:
        raise ValueError("accepted attention-projection ownership drifted")
    process_files = manifest.get("process_files")
    if (
        not isinstance(process_files, list)
        or len(process_files) != 1
        or process_files[0].get("process_index") != 0
    ):
        raise ValueError("accepted attention-projection source ledger drifted")
    record = manifest.get("tensor_file", {})
    tensor_path = config.accepted_capture_dir / "attention_projection.npz"
    if (
        record.get("filename") != tensor_path.name
        or tensor_path.stat().st_size != record.get("byte_count")
        or _file_sha256(tensor_path) != record.get("sha256")
    ):
        raise ValueError("accepted attention-projection tensor file drifted")
    with np.load(tensor_path, allow_pickle=False) as payload:
        expected_keys = {
            "attended_latent_bfloat16_bits",
            "attention_output_bfloat16_bits",
        }
        if set(payload.files) != expected_keys:
            raise ValueError("accepted attention-projection tensor keys drifted")
        latent_bits = np.ascontiguousarray(
            payload["attended_latent_bfloat16_bits"]
        )
        output_bits = np.ascontiguousarray(
            payload["attention_output_bfloat16_bits"]
        )
    tensors = manifest.get("tensors", {})
    for bits, shape, name in (
        (latent_bits, ATTENDED_LATENT_SHAPE, "attended_latent_bfloat16_bits"),
        (output_bits, (WIDTH,), "attention_output_bfloat16_bits"),
    ):
        expected_tensor = tensors.get(name, {})
        if (
            bits.shape != shape
            or bits.dtype != np.dtype(np.uint16)
            or expected_tensor.get("shape") != list(shape)
            or expected_tensor.get("dtype") != "uint16"
            or expected_tensor.get("sha256") != _array_sha256(bits)
        ):
            raise ValueError(f"accepted {name} contract drifted")
    return latent_bits, output_bits, manifest


def _load_greenfield_projection_operands(
    config: AttentionProjectionComparisonConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    contract_path = config.greenfield_ingredients_dir / "contract.json"
    tensor_path = (
        config.greenfield_ingredients_dir
        / f"position_{config.expected_position}_ingredients.npz"
    )
    if _file_sha256(contract_path) != config.expected_ingredients_contract_sha256:
        raise ValueError("greenfield ingredient contract hash drifted")
    if _file_sha256(tensor_path) != config.expected_ingredients_tensor_sha256:
        raise ValueError("greenfield ingredient tensor hash drifted")
    contract = json.loads(contract_path.read_text())
    expected_contract = {
        "code_hash": config.expected_greenfield_code_hash,
        "decode_position": config.expected_position,
        "main_rope_table_enabled": True,
        "main_rope_table_sha256": config.expected_main_rope_table_sha256,
        "passed": True,
        "run_tag": config.expected_greenfield_run_tag,
        "source_state": "post_teacher_forced_prefill",
    }
    if any(contract.get(name) != value for name, value in expected_contract.items()):
        raise ValueError("greenfield ingredient identity drifted")
    if contract.get("active_rows") != [0, 1, 2, 3]:
        raise ValueError("greenfield ingredient owner order drifted")
    main_rope = contract.get("hlo_contract", {}).get(
        "main_rope_table_contract", {}
    )
    if (
        not main_rope.get("passed")
        or main_rope.get("table_parameter_count") != 1
        or main_rope.get("named_table_parameter_count") != 1
    ):
        raise ValueError("greenfield ingredient main-RoPE HLO drifted")
    names = (
        "combined_attention_output_bfloat16_bits",
        "value_states_bfloat16_bits",
        "attention_output_input_bfloat16_bits",
    )
    with np.load(tensor_path, allow_pickle=False) as payload:
        if any(name not in payload.files for name in names):
            raise ValueError("greenfield attention-projection operand is absent")
        combined = np.ascontiguousarray(payload[names[0]])
        value_states = np.ascontiguousarray(payload[names[1]])
        output_input = np.ascontiguousarray(payload[names[2]])
    shapes = ((4, 64, 512), (4, 16, 256), (4, 4096))
    for bits, shape, name in zip(
        (combined, value_states, output_input), shapes, names, strict=True
    ):
        record = contract.get("arrays", {}).get(name, {})
        if (
            bits.shape != shape
            or bits.dtype != np.dtype(np.uint16)
            or record.get("shape") != list(shape)
            or record.get("dtype") != "uint16"
            or record.get("sha256") != _array_sha256(bits)
        ):
            raise ValueError(f"greenfield {name} ledger drifted")
        if not np.isfinite(_decode_bfloat16(bits).astype(np.float32)).all():
            raise ValueError(f"greenfield {name} contains non-finite data")
    if not all(np.array_equal(combined[0], combined[index]) for index in range(1, 4)):
        raise ValueError("greenfield combined attention owners disagree")
    if not np.array_equal(value_states.reshape(4, 4096), output_input):
        raise ValueError("greenfield W_UV output representations disagree")
    return (
        np.ascontiguousarray(combined[0]),
        np.ascontiguousarray(output_input.reshape(WIDTH)),
        {
            "contract_sha256": config.expected_ingredients_contract_sha256,
            "tensor_sha256": config.expected_ingredients_tensor_sha256,
        },
    )


def _comparison_metrics(expected: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    mismatch_indices = np.flatnonzero(expected != observed)
    delta = np.abs(
        _decode_bfloat16(expected).astype(np.float32)
        - _decode_bfloat16(observed).astype(np.float32)
    )
    return {
        "elementwise_exact": bool(not mismatch_indices.size),
        "expected_sha256": _array_sha256(expected),
        "first_mismatch_index": (
            int(mismatch_indices[0]) if mismatch_indices.size else None
        ),
        "max_abs_error": float(np.max(delta)),
        "mean_abs_error": float(np.mean(delta)),
        "mismatch_count": int(mismatch_indices.size),
        "observed_sha256": _array_sha256(observed),
        "shape": list(expected.shape),
    }


def compare_attention_projection_operands(
    config: AttentionProjectionComparisonConfig,
) -> dict[str, Any]:
    """Locate the first mismatch on the two sides of accepted ``W_UV``."""

    for value, lengths, name in (
        (
            config.expected_accepted_capture_file_sha256,
            (64,),
            "capture manifest file",
        ),
        (config.expected_ingredients_contract_sha256, (64,), "contract"),
        (config.expected_ingredients_tensor_sha256, (64,), "tensor"),
        (config.expected_greenfield_code_hash, (40,), "greenfield code hash"),
        (config.expected_legacy_code_hash, (40,), "legacy code hash"),
        (config.expected_oracle_pin, (40,), "oracle pin"),
        (config.expected_main_rope_table_sha256, (64,), "main-RoPE table"),
    ):
        _require_digest(value, lengths=lengths, name=name)
    if config.output_dir.exists():
        raise FileExistsError(f"append-only comparison exists: {config.output_dir}")
    accepted_latent, accepted_output, accepted_manifest = (
        _load_accepted_projection_capture(config)
    )
    greenfield_latent, greenfield_output, ingredient_files = (
        _load_greenfield_projection_operands(config)
    )
    latent = _comparison_metrics(accepted_latent, greenfield_latent)
    output = _comparison_metrics(accepted_output, greenfield_output)
    if not latent["elementwise_exact"]:
        classification = "attention_arithmetic_before_w_uv"
    elif not output["elementwise_exact"]:
        classification = "w_uv_projection_arithmetic"
    else:
        classification = "projection_operands_exact"
    result: dict[str, Any] = {
        "accepted": {
            "manifest_file_sha256": config.expected_accepted_capture_file_sha256,
            "manifest_sha256": accepted_manifest["manifest_sha256"],
            "run_tag": accepted_manifest["run_tag"],
        },
        "artifact_kind": PROJECTION_COMPARISON_KIND,
        "attention_output": output,
        "attended_latent": latent,
        "classification": classification,
        "diagnostic_only": True,
        "format_version": 1,
        "greenfield": {
            "code_hash": config.expected_greenfield_code_hash,
            "ingredient_files": ingredient_files,
            "run_tag": config.expected_greenfield_run_tag,
        },
        "layer_name": LAYER_NAME,
        "model_id": MODEL_ID,
        "oracle_pin": config.expected_oracle_pin,
        "performance_claim": False,
        "position": config.expected_position,
        "status": "SUCCESS",
    }
    result["manifest_sha256"] = _manifest_sha256(result)
    config.output_dir.mkdir(parents=True)
    _atomic_json(config.output_dir / "comparison.json", result)
    return result
