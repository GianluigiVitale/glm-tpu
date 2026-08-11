"""Bitwise legacy/PP8 layer-0 main-cache boundary comparison."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from .prompt_index_cache import _parse_slice_tuple, _slice_extent

ARTIFACT_KIND = "glm52_legacy_pp8_layer0_main_cache_comparison"
FORMAT_VERSION = 1
MODEL_ID = "zai-org/GLM-5.2-FP8"


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


def _valid_digest(value: str, lengths: tuple[int, ...]) -> bool:
    return len(value) in lengths and all(character in "0123456789abcdef"
                                         for character in value)


@dataclass(frozen=True, slots=True)
class LegacyMainCacheComparisonConfig:
    """Pinned inputs for one bounded layer-0 cache comparison."""

    source_dump_dir: Path
    ingredients_dir: Path
    output_dir: Path
    capture_code_hash: str
    legacy_repository_pin: str
    accepted_oracle_pin: str
    ingredients_code_hash: str
    ingredients_contract_sha256: str
    ingredients_tensor_sha256: str
    run_tag: str
    source_run_id: int
    source_item_row_id: int
    expected_process_count: int = 8
    expected_local_replication: int = 4
    expected_physical_replication: int = 32
    expected_mesh_model_size: int = 32
    expected_mesh_dcp_size: int = 1
    expected_prefill_step: int = 4
    expected_decode_step: int = 5
    expected_last_chunk_tokens: int = 2011
    expected_decode_tokens: int = 1
    expected_prompt_tokens: int = 8155
    expected_current_position: int = 8155
    expected_physical_pages: int = 24
    expected_logical_page_size: int = 512
    expected_cache_width: int = 640
    expected_payload_width: int = 576
    expected_cache_slot: int = 1
    expected_selected_width: int = 2048
    expected_owner_count: int = 4

    def __post_init__(self) -> None:
        for name in ("source_dump_dir", "ingredients_dir", "output_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        for name, lengths in (
            ("capture_code_hash", (40, )),
            ("legacy_repository_pin", (40, )),
            ("accepted_oracle_pin", (40, )),
            ("ingredients_code_hash", (40, )),
            ("ingredients_contract_sha256", (64, )),
            ("ingredients_tensor_sha256", (64, )),
        ):
            if not _valid_digest(getattr(self, name), lengths):
                raise ValueError(f"invalid layer-0 main-cache digest: {name}")
        for name in (
                "source_run_id",
                "source_item_row_id",
                "expected_process_count",
                "expected_local_replication",
                "expected_physical_replication",
                "expected_mesh_model_size",
                "expected_mesh_dcp_size",
                "expected_prefill_step",
                "expected_decode_step",
                "expected_last_chunk_tokens",
                "expected_decode_tokens",
                "expected_prompt_tokens",
                "expected_current_position",
                "expected_physical_pages",
                "expected_logical_page_size",
                "expected_cache_width",
                "expected_payload_width",
                "expected_cache_slot",
                "expected_selected_width",
                "expected_owner_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value,
                                                        bool) or value <= 0:
                raise ValueError(f"layer-0 main-cache {name} must be positive")
        if self.expected_decode_step != self.expected_prefill_step + 1:
            raise ValueError("main-cache capture steps must be consecutive")
        if self.expected_current_position != self.expected_prompt_tokens:
            raise ValueError(
                "main-cache current position must follow the prompt")
        if self.expected_physical_replication != (
                self.expected_process_count * self.expected_local_replication):
            raise ValueError(
                "main-cache process/replica geometry is inconsistent")
        if self.expected_physical_replication != (
                self.expected_mesh_model_size * self.expected_mesh_dcp_size):
            raise ValueError(
                "main-cache physical/mesh geometry is inconsistent")
        if self.expected_logical_page_size % 32:
            raise ValueError("main-cache page must preserve packing 32")
        if self.expected_logical_page_size % self.expected_owner_count:
            raise ValueError(
                "main-cache page must divide over the local owners")
        if self.expected_payload_width >= self.expected_cache_width:
            raise ValueError(
                "main-cache payload width must leave explicit padding")
        if not self.run_tag:
            raise ValueError("main-cache run tag must be non-empty")


def _load_ingredients(
    config: LegacyMainCacheComparisonConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    contract_path = config.ingredients_dir / "contract.json"
    tensor_path = (
        config.ingredients_dir /
        f"position_{config.expected_current_position}_ingredients.npz")
    if _sha256_file(contract_path) != config.ingredients_contract_sha256:
        raise ValueError("greenfield ingredient contract identity drifted")
    if _sha256_file(tensor_path) != config.ingredients_tensor_sha256:
        raise ValueError("greenfield ingredient tensor identity drifted")
    contract = json.loads(contract_path.read_text())
    if (contract.get("passed") is not True
            or contract.get("code_hash") != config.ingredients_code_hash or
            contract.get("decode_position") != config.expected_current_position
            or contract.get("source_state") != "post_teacher_forced_prefill"
            or contract.get("selection_exact") is not True
            or contract.get("owner_partition_passed") is not True
            or contract.get("owner_union_exact") is not True
            or contract.get("lane_replication_passed") is not True
            or contract.get("health_passed") is not True
            or contract.get("finite_passed") is not True):
        raise ValueError("greenfield ingredient contract did not pass")
    hlo_sha256 = contract.get("hlo_sha256")
    if not isinstance(hlo_sha256, str) or not _valid_digest(hlo_sha256,
                                                            (64, )):
        raise ValueError("greenfield ingredient HLO identity drifted")
    with np.load(tensor_path, allow_pickle=False) as payload:
        arrays = {
            name: np.ascontiguousarray(payload[name])
            for name in payload.files
        }
    records = contract.get("arrays")
    if not isinstance(records, dict) or set(
            arrays) != set(records) | {"decode_position"}:
        raise ValueError("greenfield ingredient array set drifted")
    for name, record in records.items():
        value = arrays[name]
        if (list(value.shape) != record.get("shape")
                or str(value.dtype) != record.get("dtype")
                or _array_sha256(value) != record.get("sha256")):
            raise ValueError(f"greenfield ingredient array drifted: {name}")
    decode_position = arrays["decode_position"]
    if (decode_position.shape != (1, ) or decode_position.dtype != np.int32
            or decode_position.tolist() != [config.expected_current_position]):
        raise ValueError("greenfield ingredient decode position drifted")
    source = {
        "contract": {
            "byte_count": contract_path.stat().st_size,
            "filename": contract_path.name,
            "sha256": config.ingredients_contract_sha256,
        },
        "tensor": {
            "byte_count": tensor_path.stat().st_size,
            "filename": tensor_path.name,
            "sha256": config.ingredients_tensor_sha256,
        },
    }
    return contract, arrays, source


def _load_legacy_step(
    config: LegacyMainCacheComparisonConfig,
    *,
    step_index: int,
    scheduled_tokens: int,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    pattern = f"main_cache.postfwd.step{step_index:04d}.proc*.npz"
    paths = sorted(config.source_dump_dir.rglob(pattern))
    if len(paths) != config.expected_process_count:
        raise ValueError(
            "main-cache process coverage drifted: "
            f"step={step_index} expected={config.expected_process_count} "
            f"found={len(paths)}")

    shape = (
        config.expected_physical_pages,
        config.expected_logical_page_size // 32,
        32,
        config.expected_cache_width,
    )
    expected_mesh = {
        "data": 1,
        "attn_dp": 1,
        "attn_dp_expert": 1,
        "expert": 1,
        "model": config.expected_mesh_model_size,
        "dcp": config.expected_mesh_dcp_size,
    }
    cache_bits = np.zeros(shape, dtype=np.uint16)
    replica_counts = np.zeros(shape, dtype=np.uint8)
    process_indices: set[int] = set()
    physical_devices: set[str] = set()
    block_tables: np.ndarray | None = None
    sequence_lengths: np.ndarray | None = None
    source_records: list[dict[str, Any]] = []
    shard_records = 0

    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            process_index = int(payload["process_index"])
            if (int(payload["process_count"]) != config.expected_process_count
                    or process_index in process_indices):
                raise ValueError("main-cache process identity drifted")
            process_indices.add(process_index)
            if (int(payload["step_index"]) != step_index
                    or str(payload["phase"]) != "postfwd" or int(
                        payload["num_scheduled_tokens"]) != scheduled_tokens):
                raise ValueError("main-cache scheduler-step identity drifted")
            if payload["layer_indices"].tolist() != [
                    config.expected_cache_slot
            ]:
                raise ValueError(
                    "main-cache source contains the wrong cache slot")
            try:
                mesh = ast.literal_eval(str(payload["mesh_shape"]))
            except (SyntaxError, ValueError) as error:
                raise ValueError(
                    "main-cache source mesh is not portable") from error
            if mesh != expected_mesh:
                raise ValueError("main-cache source mesh drifted")
            prefix = f"layer{config.expected_cache_slot}"
            if str(payload[f"{prefix}__dtype"]) != "bfloat16" or tuple(
                    int(value)
                    for value in payload[f"{prefix}__shape"]) != shape:
                raise ValueError("main-cache source shape/dtype drifted")
            shard_count = int(payload[f"{prefix}__nshards"])
            if shard_count != config.expected_local_replication:
                raise ValueError("main-cache local replica coverage drifted")
            for shard_index in range(shard_count):
                key = f"{prefix}__shard{shard_index}"
                shard = np.asarray(payload[f"{key}__data"])
                if shard.dtype != np.uint16:
                    raise ValueError("main-cache shard is not BF16 bits")
                index = _parse_slice_tuple(str(payload[f"{key}__index"]),
                                           rank=len(shape))
                expected_shard_shape = tuple(
                    _slice_extent(part, size)
                    for part, size in zip(index, shape, strict=True))
                if shard.shape != expected_shard_shape or shard.shape != shape:
                    raise ValueError(
                        "main-cache source is not fully replicated")
                device = str(payload[f"{key}__device"])
                if f"process={process_index}," not in device or (
                        device in physical_devices):
                    raise ValueError(
                        "main-cache physical device identity drifted")
                physical_devices.add(device)
                target = cache_bits[index]
                counts = replica_counts[index]
                overlap = counts > 0
                if np.any(overlap) and not np.array_equal(
                        target[overlap], shard[overlap]):
                    raise ValueError("main-cache model replicas disagree")
                target[~overlap] = shard[~overlap]
                counts += np.uint8(1)
                shard_records += 1

            current_tables = np.asarray(payload["meta__block_tables"])
            current_lengths = np.asarray(payload["meta__seq_lens"])
            if current_tables.dtype != np.int32 or current_lengths.dtype != np.int32:
                raise ValueError("main-cache metadata dtype drifted")
            if block_tables is None:
                block_tables = current_tables.copy()
                sequence_lengths = current_lengths.copy()
            elif not np.array_equal(block_tables,
                                    current_tables) or not np.array_equal(
                                        sequence_lengths, current_lengths):
                raise ValueError(
                    "main-cache metadata differs across processes")
        source_records.append({
            "byte_count":
            path.stat().st_size,
            "path":
            path.relative_to(config.source_dump_dir).as_posix(),
            "process_index":
            process_index,
            "sha256":
            _sha256_file(path),
            "step_index":
            step_index,
        })

    if process_indices != set(range(config.expected_process_count)):
        raise ValueError("main-cache process indices are incomplete")
    if (int(replica_counts.min()) != config.expected_physical_replication or
            int(replica_counts.max()) != config.expected_physical_replication
            or len(physical_devices) != config.expected_physical_replication
            or shard_records != config.expected_physical_replication):
        raise ValueError("main-cache physical replica coverage drifted")
    assert block_tables is not None and sequence_lengths is not None
    if (sequence_lengths.ndim != 1 or sequence_lengths.size <= 0
            or int(sequence_lengths[0]) != sequence_length
            or np.any(sequence_lengths[1:] != 0)):
        raise ValueError("main-cache sequence length drifted")
    if block_tables.size % sequence_lengths.size:
        raise ValueError("main-cache block table cannot be reshaped")
    request_tables = block_tables.reshape(sequence_lengths.size, -1)
    logical_blocks = (sequence_length + config.expected_logical_page_size -
                      1) // config.expected_logical_page_size
    if request_tables.shape[1] < logical_blocks:
        raise ValueError("main-cache block table is too narrow")
    live_table = request_tables[0, :logical_blocks].copy()
    if (np.unique(live_table).size != logical_blocks or np.any(live_table < 0)
            or np.any(live_table >= config.expected_physical_pages)):
        raise ValueError("main-cache live block table is invalid")
    flat_cache = cache_bits.reshape(
        config.expected_physical_pages,
        config.expected_logical_page_size,
        config.expected_cache_width,
    )
    positions = np.arange(sequence_length, dtype=np.int32)
    page_ids = live_table[positions // config.expected_logical_page_size]
    logical_rows = flat_cache[page_ids, positions %
                              config.expected_logical_page_size].copy()
    import ml_dtypes

    if not np.isfinite(logical_rows.view(ml_dtypes.bfloat16)).all():
        raise ValueError("main-cache live rows contain non-finite values")
    details = {
        "block_tables_sha256": _array_sha256(block_tables),
        "global_cache_bfloat16_sha256": _array_sha256(cache_bits),
        "global_cache_shape": list(shape),
        "live_block_table": live_table.tolist(),
        "live_block_table_sha256": _array_sha256(live_table),
        "logical_block_count": logical_blocks,
        "logical_rows_bfloat16_sha256": _array_sha256(logical_rows),
        "mesh_shape": expected_mesh,
        "physical_device_count": len(physical_devices),
        "physical_replication": config.expected_physical_replication,
        "sequence_length": sequence_length,
        "sequence_lengths_sha256": _array_sha256(sequence_lengths),
        "step_index": step_index,
    }
    return logical_rows, live_table, source_records, details


def _compare_bits(
    expected: np.ndarray,
    observed: np.ndarray,
    *,
    positions: np.ndarray,
) -> dict[str, Any]:
    import ml_dtypes

    expected = np.ascontiguousarray(expected)
    observed = np.ascontiguousarray(observed)
    positions = np.asarray(positions, dtype=np.int32)
    if (expected.shape != observed.shape or expected.dtype != np.uint16
            or observed.dtype != np.uint16 or expected.ndim != 2
            or positions.shape != (expected.shape[0], )):
        raise ValueError("main-cache comparison tensor contract drifted")
    mismatch = expected != observed
    row_mismatch = np.any(mismatch, axis=1)
    mismatch_rows = np.flatnonzero(row_mismatch)
    first_position: int | None = None
    first_dimension: int | None = None
    if mismatch_rows.size:
        first_row = min(mismatch_rows.tolist(),
                        key=lambda row: int(positions[row]))
        first_position = int(positions[first_row])
        first_dimension = int(np.flatnonzero(mismatch[first_row])[0])
    expected_f32 = expected.view(ml_dtypes.bfloat16).astype(np.float32)
    observed_f32 = observed.view(ml_dtypes.bfloat16).astype(np.float32)
    difference = np.abs(expected_f32 - observed_f32)
    return {
        "elementwise_exact": bool(not np.any(mismatch)),
        "expected_bfloat16_sha256": _array_sha256(expected),
        "first_mismatch_dimension": first_dimension,
        "first_mismatch_position": first_position,
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()),
        "mismatch_count": int(np.count_nonzero(mismatch)),
        "mismatched_position_count": int(mismatch_rows.size),
        "observed_bfloat16_sha256": _array_sha256(observed),
        "p99_abs": float(np.percentile(difference, 99)),
        "shape": list(expected.shape),
    }


def compare_legacy_layer0_main_cache(
    config: LegacyMainCacheComparisonConfig, ) -> dict[str, Any]:
    """Seal the first dataflow-ordered legacy/PP8 cache comparison."""

    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(config.output_dir)
    errors = sorted(config.source_dump_dir.rglob("*.ERROR*"))
    if errors:
        raise ValueError(
            f"legacy main-cache observer error exists: {errors[0]}")
    ingredient_contract, ingredients, ingredient_source = _load_ingredients(
        config)
    prefill_rows, prefill_table, prefill_sources, prefill_layout = (
        _load_legacy_step(
            config,
            step_index=config.expected_prefill_step,
            scheduled_tokens=config.expected_last_chunk_tokens,
            sequence_length=config.expected_prompt_tokens,
        ))
    decode_rows, decode_table, decode_sources, decode_layout = _load_legacy_step(
        config,
        step_index=config.expected_decode_step,
        scheduled_tokens=config.expected_decode_tokens,
        sequence_length=config.expected_prompt_tokens + 1,
    )
    if not np.array_equal(prefill_rows,
                          decode_rows[:config.expected_prompt_tokens]):
        raise ValueError(
            "legacy decode step mutated historical main-cache rows")

    selected = ingredients["selected_positions"]
    if selected.shape != (
            config.expected_owner_count, config.expected_selected_width) or (
                selected.dtype != np.int32) or any(
                    not np.array_equal(selected[0], selected[lane])
                    for lane in range(1, config.expected_owner_count)):
        raise ValueError("greenfield selected-position lanes drifted")
    selected_positions = selected[0].copy()
    if (np.unique(selected_positions).size != config.expected_selected_width
            or np.any(selected_positions < 0)
            or np.any(selected_positions >= config.expected_prompt_tokens)):
        raise ValueError("greenfield selected-position set drifted")

    owner_positions = ingredients["owner_selected_positions"]
    owner_counts_raw = ingredients["owner_selected_valid_counts"]
    owner_values = ingredients["owner_selected_cache_values_bfloat16_bits"]
    owner_valid = ingredients["owner_selected_cache_valid"]
    expected_owner_shapes = (owner_positions.shape == (
        config.expected_owner_count,
        config.expected_selected_width,
    ) and owner_counts_raw.shape == (config.expected_owner_count, 1)
                             and owner_values.shape == (
                                 config.expected_owner_count,
                                 config.expected_selected_width,
                                 config.expected_cache_width,
                             ) and owner_valid.shape
                             == (config.expected_owner_count, 1))
    if not expected_owner_shapes or not np.all(owner_valid):
        raise ValueError("greenfield owner cache contract drifted")
    owner_counts = owner_counts_raw[:, 0].astype(np.int32, copy=True)
    local_rows_per_page = (config.expected_logical_page_size //
                           config.expected_owner_count)
    by_position: dict[int, np.ndarray] = {}
    for owner, count_value in enumerate(owner_counts.tolist()):
        if not 0 < count_value <= config.expected_selected_width:
            raise ValueError("greenfield owner count drifted")
        positions = owner_positions[owner, :count_value]
        if (np.any(np.diff(positions) <= 0)
                or np.any((positions % config.expected_logical_page_size) //
                          local_rows_per_page != owner)
                or np.any(owner_positions[owner, count_value:] != -1)
                or np.any(owner_values[owner, count_value:] != 0)):
            raise ValueError("greenfield owner partition drifted")
        for offset, position in enumerate(positions.tolist()):
            if position in by_position:
                raise ValueError("greenfield owner positions overlap")
            by_position[position] = owner_values[owner, offset].copy()
    if set(by_position) != set(selected_positions.tolist()):
        raise ValueError("greenfield owner union differs from selected set")
    greenfield_selected = np.stack([
        by_position[int(position)] for position in selected_positions
    ]).astype(np.uint16, copy=False)
    legacy_selected = prefill_rows[selected_positions].copy()

    greenfield_current_lanes = ingredients["current_cache_row_bfloat16_bits"]
    if greenfield_current_lanes.shape != (
            config.expected_owner_count,
            config.expected_cache_width,
    ) or greenfield_current_lanes.dtype != np.uint16 or any(
            not np.array_equal(greenfield_current_lanes[0],
                               greenfield_current_lanes[lane])
            for lane in range(1, config.expected_owner_count)):
        raise ValueError("greenfield current-cache lanes drifted")
    greenfield_current = greenfield_current_lanes[0:1].copy()
    legacy_current = decode_rows[config.expected_current_position:config.
                                 expected_current_position + 1].copy()
    padding_start = config.expected_payload_width
    if (np.any(greenfield_selected[:, padding_start:] != 0)
            or np.any(legacy_selected[:, padding_start:] != 0)
            or np.any(greenfield_current[:, padding_start:] != 0)
            or np.any(legacy_current[:, padding_start:] != 0)):
        raise ValueError("layer-0 main-cache padding is nonzero")

    selected_comparison = _compare_bits(
        legacy_selected,
        greenfield_selected,
        positions=selected_positions,
    )
    current_comparison = _compare_bits(
        legacy_current,
        greenfield_current,
        positions=np.asarray([config.expected_current_position],
                             dtype=np.int32),
    )
    if not selected_comparison["elementwise_exact"]:
        classification = "prefill_main_cache"
        first_divergent_primitive: str | None = "selected_prefill_cache_rows"
    elif not current_comparison["elementwise_exact"]:
        classification = "recurrent_main_cache_producer"
        first_divergent_primitive = "current_decode_cache_row"
    else:
        classification = "cache_exact_attention_schedule_next"
        first_divergent_primitive = None

    config.output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = config.output_dir / "comparison.npz"
    np.savez(
        tensor_path,
        selected_positions=selected_positions,
        owner_selected_counts=owner_counts,
        legacy_selected_cache_bfloat16_bits=legacy_selected,
        greenfield_selected_cache_bfloat16_bits=greenfield_selected,
        legacy_current_cache_bfloat16_bits=legacy_current,
        greenfield_current_cache_bfloat16_bits=greenfield_current,
        prefill_live_block_table=prefill_table,
        decode_live_block_table=decode_table,
    )
    manifest: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "capture_code_hash": config.capture_code_hash,
        "classification": classification,
        "comparison": {
            "current_decode_cache_row": current_comparison,
            "selected_prefill_cache_rows": selected_comparison,
        },
        "diagnostic_only": True,
        "first_divergent_primitive": first_divergent_primitive,
        "format_version": FORMAT_VERSION,
        "greenfield_ingredients": {
            "code_hash": config.ingredients_code_hash,
            "contract_hlo_sha256": ingredient_contract["hlo_sha256"],
            "source_files": ingredient_source,
        },
        "legacy": {
            "accepted_oracle_pin": config.accepted_oracle_pin,
            "observer_pin": config.legacy_repository_pin,
            "source_item_row_id": config.source_item_row_id,
            "source_run_id": config.source_run_id,
        },
        "model_id": MODEL_ID,
        "numerical_contract": {
            "cache_dtype": "bfloat16",
            "cache_slot": config.expected_cache_slot,
            "cache_width": config.expected_cache_width,
            "current_position": config.expected_current_position,
            "owner_count": config.expected_owner_count,
            "payload_width": config.expected_payload_width,
            "selected_width": config.expected_selected_width,
        },
        "performance_claim": False,
        "run_tag": config.run_tag,
        "source_dump_files": prefill_sources + decode_sources,
        "source_layout": {
            "decode": decode_layout,
            "prefill": prefill_layout,
        },
        "tensor_file": {
            "byte_count": tensor_path.stat().st_size,
            "filename": tensor_path.name,
            "sha256": _sha256_file(tensor_path),
        },
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    (config.output_dir / "comparison.json").write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n")
    return manifest
