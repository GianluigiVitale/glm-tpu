"""Portable layer-0 prompt index-cache evidence for the 8K Gate-D probe."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


ARTIFACT_KIND = "glm52_legacy_layer0_prompt_index_cache"
FORMAT_VERSION = 1
MODEL_ID = "zai-org/GLM-5.2-FP8"
_SLICE_RE = re.compile(
    r"slice\((None|-?[0-9]+), (None|-?[0-9]+), (None|-?[0-9]+)\)"
)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _manifest_hash(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return sha256(encoded).hexdigest()


def _parse_slice_tuple(value: str, *, rank: int) -> tuple[slice, ...]:
    pieces = _SLICE_RE.findall(value)
    result = tuple(
        slice(*(None if field == "None" else int(field) for field in piece))
        for piece in pieces
    )
    if len(result) != rank or str(result) != value:
        raise ValueError(f"unsupported cache shard index: {value!r}")
    return result


def _slice_extent(index: slice, size: int) -> int:
    start, stop, step = index.indices(size)
    return len(range(start, stop, step))


@dataclass(frozen=True, slots=True)
class LegacyPromptIndexCacheConfig:
    source_dump_dir: Path
    output_dir: Path
    capture_code_hash: str
    legacy_repository_pin: str
    run_tag: str
    source_run_id: int
    source_item_row_id: int
    layer0_input_manifest_sha256: str
    prompt_token_ids_sha256: str
    expected_process_count: int = 8
    expected_local_replication: int = 4
    expected_physical_replication: int = 32
    expected_mesh_model_size: int = 32
    expected_mesh_dcp_size: int = 1
    expected_step_index: int = 4
    expected_last_chunk_tokens: int = 2011
    expected_prompt_tokens: int = 8155
    expected_physical_pages: int = 24
    expected_logical_page_size: int = 512
    expected_head_dim: int = 128

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_dump_dir", Path(self.source_dump_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        for name in (
            "capture_code_hash",
            "legacy_repository_pin",
            "layer0_input_manifest_sha256",
            "prompt_token_ids_sha256",
        ):
            value = getattr(self, name)
            if len(value) not in (40, 64) or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"invalid prompt-cache digest: {name}")
        for name in (
            "source_run_id",
            "source_item_row_id",
            "expected_process_count",
            "expected_local_replication",
            "expected_physical_replication",
            "expected_mesh_model_size",
            "expected_mesh_dcp_size",
            "expected_step_index",
            "expected_last_chunk_tokens",
            "expected_prompt_tokens",
            "expected_physical_pages",
            "expected_logical_page_size",
            "expected_head_dim",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"prompt-cache {name} must be positive")
        if self.expected_physical_replication != (
            self.expected_process_count * self.expected_local_replication
        ):
            raise ValueError("prompt-cache process/replica geometry is inconsistent")
        if self.expected_physical_replication != (
            self.expected_mesh_model_size * self.expected_mesh_dcp_size
        ):
            raise ValueError("prompt-cache physical/mesh geometry is inconsistent")
        if self.expected_logical_page_size % 32:
            raise ValueError("prompt-cache logical page must preserve packing 32")
        if not self.run_tag:
            raise ValueError("prompt-cache run tag must be non-empty")


def _load_source_cache(
    config: LegacyPromptIndexCacheConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    pattern = (
        f"index_cache.postfwd.step{config.expected_step_index:04d}.proc*.npz"
    )
    paths = sorted(config.source_dump_dir.rglob(pattern))
    if len(paths) != config.expected_process_count:
        raise ValueError(
            "prompt-cache process coverage drifted: "
            f"expected={config.expected_process_count} found={len(paths)}"
        )

    cache_bits: np.ndarray | None = None
    replica_counts: np.ndarray | None = None
    block_tables: np.ndarray | None = None
    sequence_lengths: np.ndarray | None = None
    source_records: list[dict[str, Any]] = []
    process_indices: set[int] = set()
    physical_devices: set[str] = set()
    shard_records = 0

    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            process_index = int(payload["process_index"])
            process_count = int(payload["process_count"])
            if process_count != config.expected_process_count or (
                process_index in process_indices
            ):
                raise ValueError("prompt-cache process identity drifted")
            process_indices.add(process_index)
            if int(payload["step_index"]) != config.expected_step_index or (
                str(payload["phase"]) != "postfwd"
            ):
                raise ValueError("prompt-cache snapshot phase/step drifted")
            if int(payload["num_scheduled_tokens"]) != (
                config.expected_last_chunk_tokens
            ):
                raise ValueError("prompt-cache final prefill chunk drifted")
            if payload["layer_indices"].tolist() != [0]:
                raise ValueError("prompt-cache source must contain slot zero only")
            try:
                mesh = ast.literal_eval(str(payload["mesh_shape"]))
            except (SyntaxError, ValueError) as error:
                raise ValueError("prompt-cache source mesh is not portable") from error
            expected_mesh = {
                "data": 1,
                "attn_dp": 1,
                "attn_dp_expert": 1,
                "expert": 1,
                "model": config.expected_mesh_model_size,
                "dcp": config.expected_mesh_dcp_size,
            }
            if mesh != expected_mesh:
                raise ValueError("prompt-cache source mesh drifted")
            if str(payload["layer0__dtype"]) != "bfloat16":
                raise ValueError("prompt-cache source dtype drifted")

            shape = tuple(int(value) for value in payload["layer0__shape"])
            expected_shape = (
                config.expected_physical_pages,
                config.expected_logical_page_size // 32,
                32,
                config.expected_head_dim,
            )
            if shape != expected_shape:
                raise ValueError(
                    f"prompt-cache global shape drifted: {shape}"
                )
            if cache_bits is None:
                cache_bits = np.zeros(shape, dtype=np.uint16)
                replica_counts = np.zeros(shape, dtype=np.uint8)
            elif cache_bits.shape != shape:
                raise ValueError("prompt-cache shape differs across processes")

            assert replica_counts is not None
            shard_count = int(payload["layer0__nshards"])
            if shard_count != config.expected_local_replication:
                raise ValueError("prompt-cache local replica coverage drifted")
            for shard_index in range(shard_count):
                key = f"layer0__shard{shard_index}"
                shard = np.asarray(payload[f"{key}__data"])
                if shard.dtype != np.uint16:
                    raise ValueError("prompt-cache shard is not BF16 bits")
                index = _parse_slice_tuple(
                    str(payload[f"{key}__index"]), rank=len(shape)
                )
                expected_shard_shape = tuple(
                    _slice_extent(part, size)
                    for part, size in zip(index, shape, strict=True)
                )
                if shard.shape != expected_shard_shape:
                    raise ValueError("prompt-cache shard extent drifted")
                if shard.shape != shape:
                    raise ValueError("prompt-cache source is not fully replicated")
                device = str(payload[f"{key}__device"])
                if f"process={process_index}," not in device or (
                    device in physical_devices
                ):
                    raise ValueError("prompt-cache physical device identity drifted")
                physical_devices.add(device)
                target = cache_bits[index]
                counts = replica_counts[index]
                overlap = counts > 0
                if np.any(overlap) and not np.array_equal(
                    target[overlap], shard[overlap]
                ):
                    raise ValueError("prompt-cache model replicas disagree")
                target[~overlap] = shard[~overlap]
                if np.any(counts == np.iinfo(np.uint8).max):
                    raise ValueError("prompt-cache replica counter overflow")
                counts += np.uint8(1)
                shard_records += 1

            current_tables = np.asarray(payload["meta__block_tables"])
            current_lengths = np.asarray(payload["meta__seq_lens"])
            if current_tables.dtype != np.int32 or (
                current_lengths.dtype != np.int32
            ):
                raise ValueError("prompt-cache metadata dtype drifted")
            if block_tables is None:
                block_tables = current_tables.copy()
                sequence_lengths = current_lengths.copy()
            elif not np.array_equal(block_tables, current_tables) or (
                not np.array_equal(sequence_lengths, current_lengths)
            ):
                raise ValueError("prompt-cache metadata differs across processes")

        source_records.append(
            {
                "byte_count": path.stat().st_size,
                "path": path.relative_to(config.source_dump_dir).as_posix(),
                "process_index": process_index,
                "sha256": _sha256_file(path),
            }
        )

    if process_indices != set(range(config.expected_process_count)):
        raise ValueError("prompt-cache process indices are incomplete")
    assert cache_bits is not None and replica_counts is not None
    assert block_tables is not None and sequence_lengths is not None
    if int(replica_counts.min()) != config.expected_physical_replication or (
        int(replica_counts.max()) != config.expected_physical_replication
    ):
        raise ValueError("prompt-cache physical replica coverage drifted")
    if len(physical_devices) != config.expected_physical_replication or (
        shard_records != config.expected_physical_replication
    ):
        raise ValueError("prompt-cache physical device coverage drifted")
    if sequence_lengths.ndim != 1 or sequence_lengths.size <= 0 or (
        int(sequence_lengths[0]) != config.expected_prompt_tokens
    ) or np.any(sequence_lengths[1:] != 0):
        raise ValueError("prompt-cache final sequence length drifted")
    if block_tables.size % sequence_lengths.size:
        raise ValueError("prompt-cache block table cannot be reshaped")
    request_tables = block_tables.reshape(sequence_lengths.size, -1)
    logical_blocks = (
        config.expected_prompt_tokens + config.expected_logical_page_size - 1
    ) // config.expected_logical_page_size
    if request_tables.shape[1] < logical_blocks:
        raise ValueError("prompt-cache block table is too narrow")
    live_table = request_tables[0, :logical_blocks].copy()
    if np.unique(live_table).size != logical_blocks or np.any(live_table < 0) or (
        np.any(live_table >= config.expected_physical_pages)
    ):
        raise ValueError("prompt-cache live block table is invalid")

    flat_cache = cache_bits.reshape(
        config.expected_physical_pages,
        config.expected_logical_page_size,
        config.expected_head_dim,
    )
    positions = np.arange(config.expected_prompt_tokens, dtype=np.int32)
    page_ids = live_table[positions // config.expected_logical_page_size]
    prompt_bits = flat_cache[
        page_ids, positions % config.expected_logical_page_size
    ].copy()
    import ml_dtypes

    if not np.isfinite(prompt_bits.view(ml_dtypes.bfloat16)).all():
        raise ValueError("prompt-cache live keys contain non-finite values")
    details = {
        "block_tables_sha256": _array_sha256(block_tables),
        "global_cache_bfloat16_sha256": _array_sha256(cache_bits),
        "global_cache_shape": list(cache_bits.shape),
        "live_block_table": live_table.tolist(),
        "live_block_table_sha256": _array_sha256(live_table),
        "logical_block_count": logical_blocks,
        "local_replication_per_process": config.expected_local_replication,
        "mesh_shape": expected_mesh,
        "physical_device_count": len(physical_devices),
        "physical_replication": config.expected_physical_replication,
        "physical_shard_record_count": shard_records,
        "sequence_lengths_sha256": _array_sha256(sequence_lengths),
    }
    return prompt_bits, live_table, source_records, details


def capture_legacy_prompt_index_cache(
    config: LegacyPromptIndexCacheConfig,
) -> dict[str, Any]:
    """Reconstruct and seal logical layer-0 BF16 prompt keys from DCP dumps."""

    from safetensors.numpy import save_file

    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_bits, live_table, source_records, details = _load_source_cache(config)
    tensor_path = config.output_dir / "prompt_index_cache.safetensors"
    save_file(
        {
            "live_block_table": live_table.astype(np.int32, copy=False),
            "prompt_index_key_bfloat16_bits": prompt_bits,
        },
        str(tensor_path),
        metadata={
            "artifact_kind": ARTIFACT_KIND,
            "format_version": str(FORMAT_VERSION),
        },
    )
    manifest: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "capture_code_hash": config.capture_code_hash,
        "diagnostic_only": True,
        "format_version": FORMAT_VERSION,
        "layer0_input_manifest_sha256": (
            config.layer0_input_manifest_sha256
        ),
        "model_id": MODEL_ID,
        "numerical_contract": {
            "cache_dtype": "bfloat16",
            "head_dim": config.expected_head_dim,
            "logical_page_size": config.expected_logical_page_size,
            "prompt_token_count": config.expected_prompt_tokens,
        },
        "prompt_index_key_bfloat16_sha256": _array_sha256(prompt_bits),
        "prompt_token_ids_sha256": config.prompt_token_ids_sha256,
        "run_tag": config.run_tag,
        "source": {
            "item_row_id": config.source_item_row_id,
            "legacy_repository_pin": config.legacy_repository_pin,
            "run_id": config.source_run_id,
        },
        "source_dump_files": source_records,
        "source_layout": details,
        "tensor_file": {
            "byte_count": tensor_path.stat().st_size,
            "filename": tensor_path.name,
            "sha256": _sha256_file(tensor_path),
        },
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    (config.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def inspect_legacy_prompt_index_cache(
    artifact_dir: Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Verify a compact prompt-key oracle and return copied BF16 bits."""

    from safetensors import safe_open

    artifact_dir = Path(artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    if manifest.get("artifact_kind") != ARTIFACT_KIND or (
        manifest.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError("unsupported prompt index-cache artifact")
    if manifest.get("model_id") != MODEL_ID or (
        manifest.get("diagnostic_only") is not True
    ):
        raise ValueError("prompt index-cache scope/model drifted")
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise ValueError("prompt index-cache manifest checksum mismatch")
    if expected_manifest_sha256 is not None and (
        manifest["manifest_sha256"] != expected_manifest_sha256
    ):
        raise ValueError("prompt index-cache manifest identity drifted")
    record = manifest.get("tensor_file", {})
    tensor_path = artifact_dir / str(record.get("filename", ""))
    if tensor_path.stat().st_size != record.get("byte_count") or (
        _sha256_file(tensor_path) != record.get("sha256")
    ):
        raise ValueError("prompt index-cache tensor integrity failed")
    contract = manifest["numerical_contract"]
    shape = (contract["prompt_token_count"], contract["head_dim"])
    with safe_open(tensor_path, framework="np") as handle:
        if handle.metadata() != {
            "artifact_kind": ARTIFACT_KIND,
            "format_version": str(FORMAT_VERSION),
        }:
            raise ValueError("prompt index-cache tensor metadata drifted")
        if set(handle.keys()) != {
            "live_block_table",
            "prompt_index_key_bfloat16_bits",
        }:
            raise ValueError("prompt index-cache tensor keys drifted")
        bits = handle.get_tensor("prompt_index_key_bfloat16_bits").copy()
        table = handle.get_tensor("live_block_table").copy()
    if bits.shape != shape or bits.dtype != np.uint16:
        raise ValueError("prompt index-cache tensor contract drifted")
    if table.dtype != np.int32 or table.ndim != 1 or table.size <= 0:
        raise ValueError("prompt index-cache block table contract drifted")
    if _array_sha256(bits) != manifest["prompt_index_key_bfloat16_sha256"]:
        raise ValueError("prompt index-cache logical key checksum mismatch")
    if table.tolist() != manifest["source_layout"]["live_block_table"]:
        raise ValueError("prompt index-cache block table identity drifted")
    return manifest, bits


def compare_prompt_index_key_bits(
    expected_bits: np.ndarray,
    observed_bits: np.ndarray,
) -> dict[str, Any]:
    """Return an exact BF16 cache delta without relaxing the gate."""

    import ml_dtypes

    expected = np.asarray(expected_bits)
    observed = np.asarray(observed_bits)
    if expected.shape != observed.shape or expected.ndim != 2:
        raise ValueError("prompt index-key comparison shape drifted")
    if expected.dtype != np.uint16 or observed.dtype != np.uint16:
        raise ValueError("prompt index-key comparison requires BF16 bits")
    element_mismatch = expected != observed
    row_mismatch = np.any(element_mismatch, axis=1)
    expected_f32 = expected.view(ml_dtypes.bfloat16).astype(np.float32)
    observed_f32 = observed.view(ml_dtypes.bfloat16).astype(np.float32)
    difference = np.abs(expected_f32 - observed_f32)
    mismatched_rows = np.flatnonzero(row_mismatch)
    return {
        "elementwise_exact": bool(not np.any(element_mismatch)),
        "expected_bfloat16_sha256": _array_sha256(expected),
        "first_mismatch_position": (
            None if mismatched_rows.size == 0 else int(mismatched_rows[0])
        ),
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()),
        "mismatch_count": int(np.count_nonzero(element_mismatch)),
        "mismatched_position_count": int(mismatched_rows.size),
        "observed_bfloat16_sha256": _array_sha256(observed),
        "p99_abs": float(np.percentile(difference, 99)),
        "shape": list(expected.shape),
    }


def validate_prompt_index_key_probe_hlo(
    optimized_hlo: str,
    *,
    prompt_token_count: int = 8155,
    unique_token_count: int = 37,
) -> dict[str, Any]:
    """Require the exact one-row production key scan and no hidden overlay."""

    if not isinstance(optimized_hlo, str) or not optimized_hlo.strip():
        raise ValueError("prompt index-key HLO must be non-empty text")
    for name, value in (
        ("prompt_token_count", prompt_token_count),
        ("unique_token_count", unique_token_count),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"prompt index-key {name} must be positive")

    lowered = optimized_hlo.lower()
    custom_calls = [
        line
        for line in lowered.splitlines()
        if 'custom_call_target="tpu_custom_call"' in line
    ]
    kernel_name = "greenfield_fp8_block_matmul_f32_m8_k6144_n128"
    key_kernel_count = sum(kernel_name in line for line in custom_calls)
    while_lines = [
        line for line in lowered.splitlines() if re.search(r"\bwhile\(", line)
    ]
    forbidden_operations = {
        name: lowered.count(name)
        for name in (
            "all-reduce",
            "all-gather",
            "all-to-all",
            "collective-permute",
            "reduce-scatter",
            "host_callback",
            "xla_python_cpu_callback",
        )
        if name in lowered
    }
    for instruction in ("send(", "recv("):
        if instruction in lowered:
            forbidden_operations[instruction[:-1]] = lowered.count(instruction)
    forbidden_shapes = [
        shape
        for shape in (
            "bf16[32,6144]",
            "f32[32,6144]",
            f"bf16[{prompt_token_count},6144]",
            f"f32[{prompt_token_count},6144]",
            "bf16[128,6144]",
            "f32[128,6144]",
        )
        if shape in lowered
    ]
    required_shapes = {
        "one_live_hidden_row": "bf16[1,6144]" in lowered,
        "prompt_key_output": (
            f"bf16[{prompt_token_count},128]" in lowered
        ),
        "prompt_row_indices": f"s32[{prompt_token_count}]" in lowered,
        "raw_wk_final_owner": "u8[128,6144]" in lowered,
        "wk_block_scales": "f32[1,48]" in lowered,
        "unique_embedding_rows": (
            f"bf16[{unique_token_count},6144]" in lowered
        ),
    }
    violations: list[str] = []
    if key_kernel_count != 1:
        violations.append(
            "expected one production raw-FP8 wk kernel, "
            f"found {key_kernel_count}"
        )
    if len(while_lines) != 1:
        violations.append(
            f"expected one outer prompt scan, found {len(while_lines)} loops"
        )
    if forbidden_operations:
        violations.append(
            f"forbidden communication/callback operations: {forbidden_operations}"
        )
    if forbidden_shapes:
        violations.append(
            f"forbidden dead-row/decoded-overlay shapes: {forbidden_shapes}"
        )
    missing = sorted(name for name, present in required_shapes.items() if not present)
    if missing:
        violations.append(f"missing prompt-key HLO shapes: {missing}")
    return {
        "forbidden_operations": forbidden_operations,
        "forbidden_shapes": forbidden_shapes,
        "key_kernel_count": key_kernel_count,
        "key_kernel_name": kernel_name,
        "outer_scan_while_count": len(while_lines),
        "passed": not violations,
        "required_shapes": required_shapes,
        "violations": violations,
    }


def validate_prompt_index_key_association_hlo(
    optimized_hlo: str,
    *,
    candidate: str,
    prompt_token_count: int = 8155,
    prompt_chunk: int = 2048,
    unique_token_count: int = 37,
) -> dict[str, Any]:
    """Pin the bounded key-LayerNorm/projection association candidates."""

    candidates = {
        "production_pallas_m1_divide_sqrt": ("pallas", "divide_sqrt"),
        "accepted_xla_m2048_divide_sqrt": ("xla", "divide_sqrt"),
        "accepted_xla_m2048_multiply_rsqrt": (
            "xla",
            "multiply_rsqrt",
        ),
        "accepted_xla_m2048_chunk_parameter_divide_sqrt": (
            "xla_chunk",
            "divide_sqrt",
        ),
        "accepted_xla_m2048_chunk_bf16_weight_divide_sqrt": (
            "xla_chunk_bf16_weight",
            "divide_sqrt",
        ),
        "accepted_xla_m2048_gather_chunk_bf16_weight_divide_sqrt": (
            "xla_chunk_gather_bf16_weight",
            "divide_sqrt",
        ),
    }
    if candidate not in candidates:
        raise ValueError(f"unsupported prompt-key association {candidate!r}")
    backend, norm_mode = candidates[candidate]
    if backend == "pallas":
        result = validate_prompt_index_key_probe_hlo(
            optimized_hlo,
            prompt_token_count=prompt_token_count,
            unique_token_count=unique_token_count,
        )
    elif backend == "xla":
        lowered = optimized_hlo.lower()
        convolution_lines = [
            line
            for line in lowered.splitlines()
            if re.search(
                rf"= f32\[{prompt_chunk},128\].* convolution\(", line
            )
            and "dim_labels=bf_oi->bf" in line
        ]
        while_lines = [
            line
            for line in lowered.splitlines()
            if re.search(r"\bwhile\(", line)
        ]
        forbidden_operations = {
            name: lowered.count(name)
            for name in (
                "all-reduce",
                "all-gather",
                "all-to-all",
                "collective-permute",
                "reduce-scatter",
                "host_callback",
                "xla_python_cpu_callback",
                "tpu_custom_call",
            )
            if name in lowered
        }
        forbidden_shapes = [
            shape
            for shape in (
                "bf16[32,6144]",
                "f32[32,6144]",
                f"bf16[{prompt_token_count},6144]",
                f"f32[{prompt_token_count},6144]",
                "bf16[8192,6144]",
                "f32[8192,6144]",
                "bf16[4,2048,6144]",
                "f32[4,2048,6144]",
            )
            if shape in lowered
        ]
        required_shapes = {
            "accepted_adapted_wk": "f32[128,6144]" in lowered,
            "chunk_hidden": f"bf16[{prompt_chunk},6144]" in lowered,
            "chunk_projection": f"f32[{prompt_chunk},128]" in lowered,
            "prompt_key_output": (
                f"bf16[{prompt_token_count},128]" in lowered
            ),
            "prompt_row_indices": f"s32[{prompt_token_count}]" in lowered,
            "unique_embedding_rows": (
                f"bf16[{unique_token_count},6144]" in lowered
            ),
        }
        violations: list[str] = []
        if len(convolution_lines) != 1:
            violations.append(
                "expected one accepted M2048 wk convolution, "
                f"found {len(convolution_lines)}"
            )
        if len(while_lines) != 1:
            violations.append(
                f"expected one chunk map loop, found {len(while_lines)}"
            )
        if forbidden_operations:
            violations.append(
                f"forbidden operations: {forbidden_operations}"
            )
        if forbidden_shapes:
            violations.append(
                f"forbidden hidden/dead-row shapes: {forbidden_shapes}"
            )
        missing = sorted(
            name for name, present in required_shapes.items() if not present
        )
        if missing:
            violations.append(f"missing association HLO shapes: {missing}")
        result = {
            "accepted_convolution_count": len(convolution_lines),
            "forbidden_operations": forbidden_operations,
            "forbidden_shapes": forbidden_shapes,
            "outer_map_while_count": len(while_lines),
            "passed": not violations,
            "required_shapes": required_shapes,
            "violations": violations,
        }
    else:
        lowered = optimized_hlo.lower()
        convolution_lines = [
            line
            for line in lowered.splitlines()
            if re.search(
                rf"= f32\[{prompt_chunk},128\].* convolution\(", line
            )
            and "dim_labels=bf_oi->bf" in line
        ]
        while_lines = [
            line
            for line in lowered.splitlines()
            if re.search(r"\bwhile\(", line)
        ]
        forbidden_operations = {
            name: lowered.count(name)
            for name in (
                "all-reduce",
                "all-gather",
                "all-to-all",
                "collective-permute",
                "reduce-scatter",
                "host_callback",
                "xla_python_cpu_callback",
                "tpu_custom_call",
            )
            if name in lowered
        }
        forbidden_shapes = [
            shape
            for shape in (
                "bf16[32,6144]",
                "f32[32,6144]",
                f"bf16[{prompt_token_count},6144]",
                f"f32[{prompt_token_count},6144]",
                "bf16[8192,6144]",
                "f32[8192,6144]",
                "bf16[4,2048,6144]",
                "f32[4,2048,6144]",
            )
            if shape in lowered
        ]
        required_shapes = {
            "accepted_adapted_wk": "f32[128,6144]" in lowered,
            "chunk_projection": f"f32[{prompt_chunk},128]" in lowered,
            "chunk_key_output": f"bf16[{prompt_chunk},128]" in lowered,
        }
        if backend == "xla_chunk_gather_bf16_weight":
            required_shapes.update(
                {
                    "unique_embedding_parameter": bool(
                        re.search(
                            rf"bf16\[{unique_token_count},6144\].*parameter\(",
                            lowered,
                        )
                    ),
                    "chunk_row_and_position_parameters": len(
                        re.findall(
                            rf"s32\[{prompt_chunk}\].*parameter\(", lowered
                        )
                    )
                    >= 2,
                }
            )
        else:
            required_shapes.update(
                {
                    "chunk_hidden": f"bf16[{prompt_chunk},6144]" in lowered,
                    "chunk_positions": f"s32[{prompt_chunk}]" in lowered,
                }
            )
        bf16_weight_conversion_lines = [
            line
            for line in lowered.splitlines()
            if re.search(
                r"= bf16\[128,6144\].*convert\(%[^)]+\)",
                line,
            )
        ]
        convolution_weight_bf16 = False
        if len(convolution_lines) == 1:
            operands = re.search(
                r"convolution\([^,]+,\s*(%[^,)]+)",
                convolution_lines[0],
            )
            if operands is not None:
                weight_name = operands.group(1)
                convolution_weight_bf16 = any(
                    re.search(
                        rf"^\s*{re.escape(weight_name)}\s*=\s*"
                        r"bf16\[128,6144\]",
                        line,
                    )
                    for line in lowered.splitlines()
                )
        gather_fusion_lines = [
            line
            for line in lowered.splitlines()
            if re.search(
                rf"= bf16\[{prompt_chunk},6144\].*fusion\(", line
            )
            and "kind=kcustom" in line
            and "jit(_take)/gather" in line
        ]
        raw_gather_lines = [
            line
            for line in lowered.splitlines()
            if re.search(
                rf"= bf16\[{prompt_chunk},6144\].*gather\(", line
            )
        ]
        physical_gather_lines = (
            gather_fusion_lines if gather_fusion_lines else raw_gather_lines
        )
        gather_producer_names = []
        for line in physical_gather_lines:
            producer = re.match(r"\s*(%[^ ]+)\s*=", line)
            if producer is not None:
                gather_producer_names.append(producer.group(1))
        gather_coupled_input_rms = any(
            re.search(rf"= f32\[{prompt_chunk}\].*fusion\(", line)
            and "reduce_sum" in line
            and any(name in line for name in gather_producer_names)
            for line in lowered.splitlines()
        )
        violations = []
        if len(convolution_lines) != 1:
            violations.append(
                "expected one chunk-parameter M2048 wk convolution, "
                f"found {len(convolution_lines)}"
            )
        if while_lines:
            violations.append(
                f"expected no chunk-program loop, found {len(while_lines)}"
            )
        if backend in (
            "xla_chunk_bf16_weight",
            "xla_chunk_gather_bf16_weight",
        ):
            if len(bf16_weight_conversion_lines) != 1:
                violations.append(
                    "expected one explicit FP32-to-BF16 wk conversion, "
                    f"found {len(bf16_weight_conversion_lines)}"
                )
            if not convolution_weight_bf16:
                violations.append(
                    "M2048 convolution does not consume a BF16 wk operand"
                )
        elif bf16_weight_conversion_lines:
            violations.append(
                "unexpected FP32-to-BF16 wk conversion in FP32 chunk candidate"
            )
        if backend == "xla_chunk_gather_bf16_weight":
            if len(physical_gather_lines) != 1:
                violations.append(
                    "expected one physical M2048 embedding gather, "
                    f"found {len(physical_gather_lines)}"
                )
            if not gather_coupled_input_rms:
                violations.append(
                    "input RMS reduction does not consume the gather producer"
                )
        elif physical_gather_lines:
            violations.append(
                "unexpected embedding gather in external-chunk candidate"
            )
        if forbidden_operations:
            violations.append(
                f"forbidden operations: {forbidden_operations}"
            )
        if forbidden_shapes:
            violations.append(
                f"forbidden full-prompt/dead-row shapes: {forbidden_shapes}"
            )
        missing = sorted(
            name for name, present in required_shapes.items() if not present
        )
        if missing:
            violations.append(f"missing chunk-parameter HLO shapes: {missing}")
        result = {
            "accepted_convolution_count": len(convolution_lines),
            "bf16_wk_conversion_count": len(
                bf16_weight_conversion_lines
            ),
            "convolution_weight_bf16": convolution_weight_bf16,
            "gather_coupled_input_rms": gather_coupled_input_rms,
            "physical_embedding_gather_count": len(physical_gather_lines),
            "forbidden_operations": forbidden_operations,
            "forbidden_shapes": forbidden_shapes,
            "loop_count": len(while_lines),
            "passed": not violations,
            "required_shapes": required_shapes,
            "violations": violations,
        }

    lowered = optimized_hlo.lower()
    sqrt_count = len(re.findall(r"\bsqrt\(", lowered))
    divide_count = len(re.findall(r"\bdivide\(", lowered))
    rsqrt_count = len(re.findall(r"\brsqrt\(", lowered))
    association_violations: list[str] = []
    if norm_mode == "divide_sqrt":
        if sqrt_count < 1 or divide_count < 1:
            association_violations.append(
                "divide/sqrt key LayerNorm is absent"
            )
    elif sqrt_count != 0 or divide_count != 0 or rsqrt_count < 2:
        association_violations.append(
            "multiply/rsqrt key LayerNorm identity drifted"
        )
    result["association"] = {
        "backend": backend,
        "divide_count": divide_count,
        "mode": norm_mode,
        "rsqrt_count": rsqrt_count,
        "sqrt_count": sqrt_count,
    }
    if association_violations:
        result["violations"] = [
            *result.get("violations", []),
            *association_violations,
        ]
        result["passed"] = False
    return result
