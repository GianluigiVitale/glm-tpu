"""Reconstruct and compare independent legacy GLM layer boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np


@dataclass(frozen=True)
class LegacyResidualComparisonConfig:
    source_dump_dir: Path
    greenfield_npz: Path
    greenfield_contract: Path
    output_dir: Path
    expected_position: int
    expected_run_tag: str
    expected_legacy_code_hash: str
    expected_oracle_pin: str
    expected_model_id: str
    expected_process_count: int = 8
    expected_boundary_ids: tuple[int, ...] = (1, 77, 78)
    expected_model_boundary_count: int = 79
    expected_decode_rows: int = 32
    expected_hidden_size: int = 6144


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(payload: Any, name: str) -> Any:
    value = payload[name]
    if value.shape != ():
        raise ValueError(f"legacy residual field {name} is not scalar")
    return value.item()


def _explicit_bits(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind != "u" or array.dtype.itemsize != 2:
        raise ValueError(f"{name} is not an explicit uint16 bit array")
    return np.ascontiguousarray(array, dtype=np.dtype("<u2"))


def _decode_bfloat16(bits: np.ndarray) -> np.ndarray:
    little = np.ascontiguousarray(bits, dtype=np.dtype("<u2"))
    return little.view(ml_dtypes.bfloat16).astype(np.float32)


def _load_greenfield(
    path: Path,
    contract_path: Path,
    *,
    expected_position: int,
    expected_boundary_ids: tuple[int, ...],
    expected_model_boundary_count: int,
    expected_hidden_size: int,
) -> tuple[np.ndarray, dict[str, Any], str]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as payload:
        if "residual_bfloat16_bits" in payload.files:
            bits = _explicit_bits(
                payload["residual_bfloat16_bits"],
                name="greenfield residual_bfloat16_bits",
            )
        elif "residuals" in payload.files:
            # The first protected artifact predates the portable field name.
            # NumPy wrote ml_dtypes.bfloat16 as void16, but preserved both raw
            # bytes exactly.  Reinterpret only this sealed compatibility case.
            raw = np.ascontiguousarray(payload["residuals"])
            if raw.dtype.kind != "V" or raw.dtype.itemsize != 2:
                raise ValueError("greenfield compatibility residuals are not void16")
            bits = raw.view(np.uint16).astype(np.dtype("<u2"), copy=False)
        else:
            raise ValueError("greenfield residual artifact has no BF16 payload")
        layer_ids = np.asarray(payload["boundary_layer_ids"], dtype=np.int32)
        position = np.asarray(payload["decode_position"], dtype=np.int32)

    expected_shape = (expected_model_boundary_count, expected_hidden_size)
    if bits.shape != expected_shape:
        raise ValueError(
            f"greenfield residual shape {bits.shape} != {expected_shape}")
    if not np.array_equal(layer_ids, np.arange(expected_shape[0], dtype=np.int32)):
        raise ValueError("greenfield residual boundary ids drifted")
    if position.shape != (1, ) or int(position[0]) != expected_position:
        raise ValueError("greenfield residual position drifted")
    canonical_sha = sha256(bits.tobytes(order="C")).hexdigest()
    expected_contract = {
        "boundary_count": expected_shape[0],
        "canonical_sha256": canonical_sha,
        "decode_position": expected_position,
        "dtype": "bfloat16",
        "hidden_size": expected_shape[1],
        "passed": True,
        "teacher_forced": True,
    }
    for key, value in expected_contract.items():
        if contract.get(key) != value:
            raise ValueError(
                f"greenfield residual contract {key}={contract.get(key)!r} "
                f"!= {value!r}")
    if not np.isfinite(_decode_bfloat16(bits)).all():
        raise ValueError("greenfield residual artifact contains non-finite values")
    return bits[np.asarray(expected_boundary_ids, dtype=np.int32)], contract, canonical_sha


def _reconstruct_legacy(
    config: LegacyResidualComparisonConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    pattern = f"*.position{config.expected_position}.proc*.npz"
    paths = sorted(config.source_dump_dir.rglob(pattern))
    if len(paths) != config.expected_process_count:
        raise ValueError(
            f"legacy residual source file count {len(paths)} != "
            f"{config.expected_process_count}")

    boundary_ids = config.expected_boundary_ids
    shape = (len(boundary_ids), config.expected_hidden_size)
    boundary_offsets = {
        boundary: offset for offset, boundary in enumerate(boundary_ids)
    }
    canonical = np.zeros(shape, dtype=np.dtype("<u2"))
    coverage = np.zeros(shape, dtype=np.uint16)
    processes: set[int] = set()
    records: list[dict[str, Any]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            expected_scalars = {
                "artifact_kind": "glm52_legacy_selected_layer_residual_shards",
                "boundary_count": len(boundary_ids),
                "code_hash": config.expected_legacy_code_hash,
                "decode_rows": config.expected_decode_rows,
                "format_version": 2,
                "hidden_size": config.expected_hidden_size,
                "model_id": config.expected_model_id,
                "oracle_pin": config.expected_oracle_pin,
                "position": config.expected_position,
                "process_count": config.expected_process_count,
                "run_tag": config.expected_run_tag,
                "storage_byte_order": "little",
                "storage_dtype": "<u2",
                "value_dtype": "bfloat16",
            }
            for name, expected in expected_scalars.items():
                observed = _scalar(payload, name)
                if observed != expected:
                    raise ValueError(
                        f"{path}: {name}={observed!r} != {expected!r}")
            global_shape = tuple(
                int(value) for value in np.asarray(payload["global_shape"]))
            expected_global_shape = (
                len(boundary_ids),
                config.expected_decode_rows,
                config.expected_hidden_size,
            )
            if global_shape != expected_global_shape:
                raise ValueError(
                    f"{path}: global shape {global_shape} != "
                    f"{expected_global_shape}")
            observed_boundary_ids = tuple(
                int(value) for value in np.asarray(payload["boundary_ids"]))
            if observed_boundary_ids != boundary_ids:
                raise ValueError(
                    f"{path}: boundary ids {observed_boundary_ids} != "
                    f"{boundary_ids}")
            observer_equal = np.asarray(
                payload["observer_output_equal"], dtype=np.bool_)
            observer_mismatches = np.asarray(
                payload["observer_output_mismatch_count"], dtype=np.int64)
            observer_max_error = np.asarray(
                payload["observer_output_max_abs_error"], dtype=np.float32)
            expected_observer_shape = (len(boundary_ids), )
            if (observer_equal.shape != expected_observer_shape
                    or observer_mismatches.shape != expected_observer_shape
                    or observer_max_error.shape != expected_observer_shape):
                raise ValueError(f"{path}: observer isolation shape drifted")
            if (not observer_equal.all() or np.any(observer_mismatches != 0)
                    or np.any(observer_max_error != 0.0)):
                raise ValueError(
                    f"{path}: observer output differs from production")
            process_index = int(_scalar(payload, "process_index"))
            if not 0 <= process_index < config.expected_process_count:
                raise ValueError(f"{path}: invalid process index {process_index}")
            if process_index in processes:
                raise ValueError(f"duplicate process index {process_index}")
            processes.add(process_index)
            entry_count = int(_scalar(payload, "entry_count"))
            target_row = int(_scalar(payload, "target_token_row"))
            if not 0 <= target_row < config.expected_decode_rows:
                raise ValueError(f"{path}: invalid target row {target_row}")
            for entry_index in range(entry_count):
                prefix = f"entry_{entry_index:04d}"
                boundary = int(_scalar(payload, f"{prefix}_boundary_id"))
                if boundary not in boundary_offsets:
                    raise ValueError(
                        f"{path}: unexpected boundary id {boundary}")
                boundary_offset = boundary_offsets[boundary]
                hidden_start = int(_scalar(payload, f"{prefix}_hidden_start"))
                hidden_stop = int(_scalar(payload, f"{prefix}_hidden_stop"))
                if not 0 <= hidden_start < hidden_stop <= shape[1]:
                    raise ValueError(f"{path}: invalid entry bounds {prefix}")
                bits = _explicit_bits(
                    payload[f"{prefix}_bfloat16_bits"],
                    name=f"{path}:{prefix}",
                )
                expected_entry_shape = (hidden_stop - hidden_start, )
                if bits.shape != expected_entry_shape:
                    raise ValueError(
                        f"{path}: {prefix} shape {bits.shape} != "
                        f"{expected_entry_shape}")
                region = np.s_[boundary_offset, hidden_start:hidden_stop]
                overlap = coverage[region] > 0
                if overlap.any() and not np.array_equal(
                        canonical[region][overlap], bits[overlap]):
                    raise ValueError(
                        f"{path}: replicated residual payload disagrees at "
                        f"{prefix}")
                canonical_region = canonical[region]
                canonical_region[~overlap] = bits[~overlap]
                coverage[region] += 1
            records.append({
                "byte_count": path.stat().st_size,
                "entry_count": entry_count,
                "observer_output_equal": True,
                "path": path.relative_to(config.source_dump_dir).as_posix(),
                "process_index": process_index,
                "sha256": _file_sha256(path),
                "target_token_row": target_row,
            })

    if processes != set(range(config.expected_process_count)):
        raise ValueError(f"legacy residual process set incomplete: {processes}")
    gaps = int(np.count_nonzero(coverage == 0))
    if gaps:
        raise ValueError(f"legacy residual reconstruction has {gaps} uncovered values")
    if not np.isfinite(_decode_bfloat16(canonical)).all():
        raise ValueError("legacy residual reconstruction contains non-finite values")
    return canonical, coverage, sorted(records, key=lambda value: value["process_index"])


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def compare_legacy_residuals(
    config: LegacyResidualComparisonConfig,
) -> dict[str, Any]:
    """Reconstruct the legacy target row and locate its first divergence."""

    if config.output_dir.exists():
        raise FileExistsError(
            f"append-only comparison output exists: {config.output_dir}")
    if config.expected_position < 0:
        raise ValueError("expected position must be non-negative")
    if min(config.expected_process_count, config.expected_model_boundary_count,
           config.expected_decode_rows, config.expected_hidden_size) <= 0:
        raise ValueError("legacy residual dimensions must be positive")
    if (not config.expected_boundary_ids
            or len(set(config.expected_boundary_ids)) != len(
                config.expected_boundary_ids)
            or tuple(sorted(config.expected_boundary_ids)) !=
            config.expected_boundary_ids
            or any(boundary < 0
                   or boundary >= config.expected_model_boundary_count
                   for boundary in config.expected_boundary_ids)):
        raise ValueError("legacy residual selected boundary ids are invalid")

    legacy_bits, coverage, source_records = _reconstruct_legacy(config)
    greenfield_bits, greenfield_contract, greenfield_sha = _load_greenfield(
        config.greenfield_npz,
        config.greenfield_contract,
        expected_position=config.expected_position,
        expected_boundary_ids=config.expected_boundary_ids,
        expected_model_boundary_count=config.expected_model_boundary_count,
        expected_hidden_size=config.expected_hidden_size,
    )
    legacy_sha = sha256(legacy_bits.tobytes(order="C")).hexdigest()
    legacy_values = _decode_bfloat16(legacy_bits)
    greenfield_values = _decode_bfloat16(greenfield_bits)

    boundary_records: list[dict[str, Any]] = []
    divergent_boundaries: list[int] = []
    for boundary_offset, boundary in enumerate(config.expected_boundary_ids):
        bit_difference = (
            legacy_bits[boundary_offset] != greenfield_bits[boundary_offset])
        differing_elements = int(np.count_nonzero(bit_difference))
        if differing_elements:
            divergent_boundaries.append(boundary)
            first_difference = int(np.flatnonzero(bit_difference)[0])
        else:
            first_difference = None
        absolute_error = np.abs(
            legacy_values[boundary_offset] - greenfield_values[boundary_offset])
        boundary_records.append({
            "boundary": boundary,
            "bitwise_equal": differing_elements == 0,
            "differing_elements": differing_elements,
            "first_differing_hidden_index": first_difference,
            "greenfield_sha256": sha256(
                greenfield_bits[boundary_offset].tobytes(order="C")).hexdigest(),
            "legacy_sha256": sha256(
                legacy_bits[boundary_offset].tobytes(order="C")).hexdigest(),
            "max_absolute_error": float(np.max(absolute_error)),
            "mean_absolute_error": float(np.mean(absolute_error)),
        })

    comparison = {
        "artifact_kind": "glm52_legacy_greenfield_layer_residual_comparison",
        "boundary_count": len(config.expected_boundary_ids),
        "boundary_ids": list(config.expected_boundary_ids),
        "boundary_records": boundary_records,
        "decode_position": config.expected_position,
        "divergent_boundary_count": len(divergent_boundaries),
        "divergent_boundaries": divergent_boundaries,
        "first_divergent_boundary": (
            divergent_boundaries[0] if divergent_boundaries else None),
        "localization_semantics": (
            "first_divergent_boundary is the first member of the explicitly "
            "selected boundary set, not necessarily the first model boundary"
        ),
        "model_boundary_count": config.expected_model_boundary_count,
        "greenfield": {
            "canonical_sha256": greenfield_sha,
            "contract": greenfield_contract,
            "contract_path": str(config.greenfield_contract),
            "contract_sha256": _file_sha256(config.greenfield_contract),
            "npz_path": str(config.greenfield_npz),
            "npz_sha256": _file_sha256(config.greenfield_npz),
        },
        "hidden_size": config.expected_hidden_size,
        "legacy": {
            "canonical_sha256": legacy_sha,
            "code_hash": config.expected_legacy_code_hash,
            "coverage_max": int(np.max(coverage)),
            "coverage_min": int(np.min(coverage)),
            "model_id": config.expected_model_id,
            "oracle_pin": config.expected_oracle_pin,
            "process_count": config.expected_process_count,
            "run_tag": config.expected_run_tag,
            "source_files": source_records,
        },
        "storage_byte_order": "little",
        "storage_dtype": "<u2",
        "value_dtype": "bfloat16",
    }

    config.output_dir.mkdir(parents=True)
    _atomic_npz(
        config.output_dir / "legacy_position_boundaries.npz",
        boundary_layer_ids=np.asarray(
            config.expected_boundary_ids, dtype=np.int32),
        decode_position=np.asarray([config.expected_position], dtype=np.int32),
        residual_bfloat16_bits=legacy_bits,
    )
    _atomic_json(config.output_dir / "comparison.json", comparison)
    return comparison
