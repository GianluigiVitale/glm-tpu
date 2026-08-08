"""Validate and localize a sealed legacy layer-0 DSA scorer-state capture."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np

from .layer0_dsa_association import (
    inspect_distributed_q_a_norm_artifact,
    inspect_layer0_dsa_association_input,
)


ARTIFACT_KIND = "glm52_legacy_dsa_internal_state"
COMPARISON_KIND = "glm52_legacy_greenfield_dsa_internal_comparison"
MODEL_ID = "zai-org/GLM-5.2-FP8"
FIELD_ORDER = (
    "normalized_hidden",
    "q_a_state",
    "query",
    "head_weights",
    "current_key",
)
FIELD_CONTRACT = {
    "normalized_hidden": ((6144,), "bfloat16", np.dtype(np.uint16)),
    "q_a_state": ((2048,), "bfloat16", np.dtype(np.uint16)),
    "query": ((32, 128), "float32", np.dtype(np.float32)),
    "head_weights": ((32,), "float32", np.dtype(np.float32)),
    "current_key": ((128,), "float32", np.dtype(np.float32)),
}


@dataclass(frozen=True, slots=True)
class LegacyDsaInternalComparisonConfig:
    """All immutable identities required for one bounded comparison."""

    source_dump_dir: Path
    layer0_input_dir: Path
    distributed_q_a_norm_dir: Path
    output_dir: Path
    expected_run_tag: str
    expected_greenfield_code_hash: str
    expected_legacy_code_hash: str
    expected_oracle_pin: str
    expected_input_manifest_sha256: str
    expected_q_a_manifest_sha256: str
    expected_q_a_code_hash: str
    expected_model_id: str = MODEL_ID
    expected_layer_name: str = "model.layers.0.self_attn.attn"
    expected_position: int = 8155
    expected_process_count: int = 8
    expected_capture_process_indices: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        capture_indices = tuple(self.expected_capture_process_indices)
        object.__setattr__(self, "expected_capture_process_indices", capture_indices)
        if self.expected_process_count <= 0:
            raise ValueError("expected_process_count must be positive")
        if not capture_indices or len(set(capture_indices)) != len(
            capture_indices
        ):
            raise ValueError("capture process indices must be non-empty and unique")
        if any(
            process_index < 0 or process_index >= self.expected_process_count
            for process_index in capture_indices
        ):
            raise ValueError("capture process index is outside the fleet")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _scalar(payload: Any, name: str) -> Any:
    value = payload[name]
    if value.shape != ():
        raise ValueError(f"legacy DSA internal field {name} is not scalar")
    return value.item()


def _require_digest(value: str, *, name: str) -> None:
    if len(value) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} is not a lowercase digest")


def _decode_bfloat16(bits: np.ndarray) -> np.ndarray:
    if bits.dtype != np.uint16:
        raise ValueError("BF16 payload is not uint16 storage")
    return np.ascontiguousarray(bits).view(ml_dtypes.bfloat16)


def _load_legacy_capture(
    config: LegacyDsaInternalComparisonConfig,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], str]:
    errors = sorted(config.source_dump_dir.rglob("*.INTERNAL.ERROR.*"))
    if errors:
        raise ValueError(f"legacy DSA internal error sentinel exists: {errors[0]}")
    safe_layer = config.expected_layer_name.replace("/", "_").replace(".", "_")
    pattern = f"*.{safe_layer}.position{config.expected_position}.proc*.npz"
    paths = sorted(config.source_dump_dir.rglob(pattern))
    expected_capture_count = len(config.expected_capture_process_indices)
    if len(paths) != expected_capture_count:
        raise ValueError(
            f"legacy DSA internal file count {len(paths)} != "
            f"{expected_capture_count}"
        )

    expected_keys = {
        "artifact_kind",
        "format_version",
        "process_index",
        "process_count",
        "layer_name",
        "position",
        "source_row",
        "run_tag",
        "code_hash",
        "oracle_pin",
        "model_id",
    } | set(FIELD_ORDER) | {f"{name}__dtype" for name in FIELD_ORDER}
    canonical: dict[str, np.ndarray] | None = None
    process_indices: set[int] = set()
    records: list[dict[str, Any]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != expected_keys:
                raise ValueError(f"{path}: internal key set drifted")
            expected_scalars = {
                "artifact_kind": ARTIFACT_KIND,
                "format_version": 1,
                "process_count": config.expected_process_count,
                "layer_name": config.expected_layer_name,
                "position": config.expected_position,
                "source_row": 0,
                "run_tag": config.expected_run_tag,
                "code_hash": config.expected_legacy_code_hash,
                "oracle_pin": config.expected_oracle_pin,
                "model_id": config.expected_model_id,
            }
            for name, expected in expected_scalars.items():
                observed = _scalar(payload, name)
                if observed != expected:
                    raise ValueError(
                        f"{path}: {name}={observed!r} != {expected!r}"
                    )
            process_index = int(_scalar(payload, "process_index"))
            if not 0 <= process_index < config.expected_process_count or (
                process_index in process_indices
            ):
                raise ValueError(f"{path}: invalid/duplicate process index")
            process_indices.add(process_index)
            current: dict[str, np.ndarray] = {}
            for name in FIELD_ORDER:
                expected_shape, expected_tag, storage_dtype = FIELD_CONTRACT[name]
                tag = str(_scalar(payload, f"{name}__dtype"))
                value = np.ascontiguousarray(payload[name])
                if tag != expected_tag or value.shape != expected_shape or (
                    value.dtype != storage_dtype
                ):
                    raise ValueError(
                        f"{path}: {name} contract drifted: "
                        f"shape={value.shape} tag={tag} storage={value.dtype}"
                    )
                numeric = (
                    _decode_bfloat16(value).astype(np.float32)
                    if expected_tag == "bfloat16"
                    else value
                )
                if not np.isfinite(numeric).all():
                    raise ValueError(f"{path}: {name} contains non-finite values")
                current[name] = value.copy()
            if canonical is None:
                canonical = current
            else:
                for name in FIELD_ORDER:
                    if not np.array_equal(current[name], canonical[name]):
                        raise ValueError(
                            f"{path}: replicated legacy {name} disagrees"
                        )
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
        raise ValueError("legacy DSA internal owner-process coverage is incomplete")
    digest = sha256()
    for name in FIELD_ORDER:
        digest.update(name.encode("ascii"))
        digest.update(np.ascontiguousarray(canonical[name]).tobytes(order="C"))
    return (
        canonical,
        sorted(records, key=lambda value: value["process_index"]),
        digest.hexdigest(),
    )


def _reconstruct_greenfield(
    config: LegacyDsaInternalComparisonConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import jax
    import jax.numpy as jnp

    from glm_tpu.greenfield.kernels.reference.dsa_association import (
        Layer0DsaProbeGeometry,
        layer0_decode_normalized_hidden,
        layer0_dsa_scorer_internals,
    )
    from glm_tpu.greenfield.kernels.reference.fp8 import (
        dequantize_fp8_bits_block_weight,
    )

    input_manifest, arrays = inspect_layer0_dsa_association_input(
        config.layer0_input_dir,
        expected_manifest_sha256=config.expected_input_manifest_sha256,
    )
    q_manifest, q_bits = inspect_distributed_q_a_norm_artifact(
        config.distributed_q_a_norm_dir,
        expected_manifest_sha256=config.expected_q_a_manifest_sha256,
        expected_code_hash=config.expected_q_a_code_hash,
        expected_input_manifest_sha256=input_manifest["manifest_sha256"],
    )
    geometry = Layer0DsaProbeGeometry()
    unique_ids = arrays["unique_token_ids"]
    current_row = np.searchsorted(unique_ids, arrays["current_token_id"]).astype(
        np.int32
    )
    if not np.array_equal(unique_ids[current_row], arrays["current_token_id"]):
        raise ValueError("layer-0 current embedding row is unavailable")

    def bf16(name: str) -> np.ndarray:
        return _decode_bfloat16(arrays[name])

    unique_embeddings = jnp.asarray(
        _decode_bfloat16(arrays["unique_embedding_bfloat16_bits"])
    )
    input_norm = jnp.asarray(bf16("input_layernorm__weight"))
    q_residual = jnp.asarray(_decode_bfloat16(q_bits))
    wq_b = jax.jit(
        lambda bits, scale: dequantize_fp8_bits_block_weight(
            bits, scale, output_dtype=jnp.float32
        )
    )(
        jnp.asarray(arrays["self_attn__indexer__wq_b__weight"]),
        jnp.asarray(arrays["self_attn__indexer__wq_b__weight_scale_inv"]),
    )
    # Accepted PWAL repairs wk into the fused BF16 leaf; the adapter then
    # casts that already-rounded state to FP32. DB491 proves direct-FP32 wk
    # changes upstream bytes but not stored prompt keys.
    wk = jax.jit(
        lambda bits, scale: dequantize_fp8_bits_block_weight(
            bits, scale, output_dtype=jnp.bfloat16
        ).astype(jnp.float32)
    )(
        jnp.asarray(arrays["self_attn__indexer__wk__weight"]),
        jnp.asarray(arrays["self_attn__indexer__wk__weight_scale_inv"]),
    )
    normalized = jax.jit(
        lambda embeddings, row, weight: layer0_decode_normalized_hidden(
            embeddings, row, weight, geometry=geometry
        )
    )(
        unique_embeddings,
        jnp.asarray(current_row),
        input_norm,
    )
    positions = jnp.zeros((geometry.decode_rows,), dtype=jnp.int32).at[0].set(
        jnp.int32(config.expected_position)
    )
    internals = jax.jit(
        lambda hidden, q_state, pos, wq, wk_value, norm_w, norm_b, head_w: (
            layer0_dsa_scorer_internals(
                hidden,
                q_state,
                pos,
                wq,
                wk_value,
                norm_w,
                norm_b,
                head_w,
                geometry=geometry,
            )
        )
    )(
        normalized,
        q_residual,
        positions,
        wq_b,
        wk,
        jnp.asarray(bf16("self_attn__indexer__k_norm__weight")).astype(
            jnp.float32
        ),
        jnp.asarray(bf16("self_attn__indexer__k_norm__bias")).astype(
            jnp.float32
        ),
        jnp.asarray(bf16("self_attn__indexer__weights_proj__weight")).astype(
            jnp.float32
        ),
    )
    jax.block_until_ready(internals)
    normalized_host = np.asarray(normalized[0])
    q_host = np.asarray(q_residual[0])
    reconstructed = {
        "normalized_hidden": np.ascontiguousarray(normalized_host).view(np.uint16),
        "q_a_state": np.ascontiguousarray(q_host).view(np.uint16),
        "query": np.asarray(internals.query[0], dtype=np.float32),
        "head_weights": np.asarray(internals.head_weights[0], dtype=np.float32),
        "current_key": np.asarray(internals.current_keys[0], dtype=np.float32),
    }
    return reconstructed, {
        "backend": jax.default_backend(),
        "device_count": jax.device_count(),
        "device_kind": sorted({device.device_kind for device in jax.devices()}),
        "input_manifest_sha256": input_manifest["manifest_sha256"],
        "q_a_manifest_sha256": q_manifest["manifest_sha256"],
        "numerical_contract": {
            "input_rms_norm_epsilon": geometry.rms_norm_epsilon,
            "q_a_rms_norm_epsilon": geometry.q_norm_epsilon,
            "key_layer_norm_epsilon": geometry.key_norm_epsilon,
            "wk_adaptation": "raw_fp8_to_bfloat16_then_float32",
            "wq_b_adaptation": "raw_fp8_to_float32",
            "current_key_boundary": "post_rope_float32_before_bfloat16_cache_write",
        },
    }


def _comparison_record(
    actual: np.ndarray,
    reconstructed: np.ndarray,
    *,
    value_dtype: str,
) -> dict[str, Any]:
    if actual.shape != reconstructed.shape or actual.dtype != reconstructed.dtype:
        raise ValueError("legacy/greenfield internal storage contract disagrees")
    actual_numeric = (
        _decode_bfloat16(actual).astype(np.float32)
        if value_dtype == "bfloat16"
        else actual.astype(np.float32)
    )
    reconstructed_numeric = (
        _decode_bfloat16(reconstructed).astype(np.float32)
        if value_dtype == "bfloat16"
        else reconstructed.astype(np.float32)
    )
    delta = reconstructed_numeric - actual_numeric
    absolute = np.abs(delta)
    return {
        "actual_sha256": _array_sha256(actual),
        "elementwise_exact": bool(np.array_equal(actual, reconstructed)),
        "max_abs": float(absolute.max(initial=0.0)),
        "mean_abs": float(absolute.mean()),
        "mean_signed": float(delta.mean()),
        "mismatch_count": int(np.count_nonzero(actual != reconstructed)),
        "p99_abs": float(np.percentile(absolute, 99)),
        "reconstructed_sha256": _array_sha256(reconstructed),
        "shape": list(actual.shape),
        "storage_dtype": str(actual.dtype),
        "value_dtype": value_dtype,
    }


def compare_legacy_dsa_internals(
    config: LegacyDsaInternalComparisonConfig,
) -> dict[str, Any]:
    """Compare one replicated accepted event with independent reconstruction."""

    for name, value in (
        ("greenfield code hash", config.expected_greenfield_code_hash),
        ("legacy code hash", config.expected_legacy_code_hash),
        ("oracle pin", config.expected_oracle_pin),
        ("input manifest", config.expected_input_manifest_sha256),
        ("q-a manifest", config.expected_q_a_manifest_sha256),
        ("q-a code hash", config.expected_q_a_code_hash),
    ):
        _require_digest(value, name=name)
    if config.output_dir.exists():
        raise FileExistsError(f"append-only comparison exists: {config.output_dir}")

    actual, process_records, actual_sha = _load_legacy_capture(config)
    reconstructed, runtime = _reconstruct_greenfield(config)
    comparisons = {
        name: _comparison_record(
            actual[name],
            np.ascontiguousarray(reconstructed[name]),
            value_dtype=FIELD_CONTRACT[name][1],
        )
        for name in FIELD_ORDER
    }
    divergent = [
        name for name in FIELD_ORDER if not comparisons[name]["elementwise_exact"]
    ]
    comparison = {
        "artifact_kind": COMPARISON_KIND,
        "diagnostic_only": True,
        "format_version": 1,
        "greenfield_code_hash": config.expected_greenfield_code_hash,
        "legacy_code_hash": config.expected_legacy_code_hash,
        "oracle_pin": config.expected_oracle_pin,
        "model_id": config.expected_model_id,
        "run_tag": config.expected_run_tag,
        "layer_name": config.expected_layer_name,
        "position": config.expected_position,
        "process_count": config.expected_process_count,
        "owner_actual_sha256": actual_sha,
        "capture_layout": "topology_sharded_live_row_owner",
        "capture_process_indices": list(
            config.expected_capture_process_indices
        ),
        "process_files": process_records,
        "runtime": runtime,
        "fields": comparisons,
        "divergent_fields": divergent,
        "first_divergent_field": divergent[0] if divergent else None,
        "all_fields_elementwise_exact": not divergent,
        "performance_claim": False,
    }
    config.output_dir.mkdir(parents=True)
    np.savez(
        config.output_dir / "internals.npz",
        **{f"actual__{name}": actual[name] for name in FIELD_ORDER},
        **{
            f"reconstructed__{name}": reconstructed[name]
            for name in FIELD_ORDER
        },
    )
    comparison["tensor_file"] = {
        "byte_count": (config.output_dir / "internals.npz").stat().st_size,
        "filename": "internals.npz",
        "sha256": _file_sha256(config.output_dir / "internals.npz"),
    }
    (config.output_dir / "comparison.json").write_text(
        json.dumps(comparison, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    return comparison
