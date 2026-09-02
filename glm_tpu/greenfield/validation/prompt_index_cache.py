"""Portable layer-0 prompt index-cache evidence for the 8K Gate-D probe."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Literal

import numpy as np


ARTIFACT_KIND = "glm52_legacy_layer0_prompt_index_cache"
# GLM-5.2 layers whose DSA indexer owns a prompt index cache: the three dense
# layers and every fourth layer from six.
FULL_INDEXER_LAYERS = frozenset({0, 1, 2} | set(range(6, 78, 4)))
LEGACY_ATTENTION_LAYER_NAME = "model.layers.{layer}.self_attn.attn"
PROMPT_KEY_INTERNAL_ARTIFACT_KIND = (
    "glm52_legacy_dsa_prompt_key_internal_state"
)
PROMPT_KEY_INPUT_INTERNAL_ARTIFACT_KIND = (
    "glm52_legacy_dsa_prompt_key_input_internal_state"
)
PROMPT_KEY_INTERNAL_CAPTURE_KIND = (
    "glm52_legacy_prompt_key_internal_capture"
)
PROMPT_KEY_INPUT_INTERNAL_CAPTURE_KIND = (
    "glm52_legacy_prompt_key_input_internal_capture"
)
FORMAT_VERSION = 1
MODEL_ID = "zai-org/GLM-5.2-FP8"
PromptKeyCaptureMode = Literal["prompt_key", "prompt_key_input"]
_PROMPT_KEY_FIELDS_BY_MODE: dict[
    PromptKeyCaptureMode, dict[str, tuple[int, ...]]
] = {
    "prompt_key": {
        "pre_layer_norm_key": (128,),
        "pre_rope_key": (128,),
        "post_rope_key": (128,),
    },
    "prompt_key_input": {
        "projection_input": (6144,),
        "pre_layer_norm_key": (128,),
        "pre_rope_key": (128,),
        "post_rope_key": (128,),
    },
}
_PROMPT_KEY_RAW_KIND_BY_MODE = {
    "prompt_key": PROMPT_KEY_INTERNAL_ARTIFACT_KIND,
    "prompt_key_input": PROMPT_KEY_INPUT_INTERNAL_ARTIFACT_KIND,
}
_PROMPT_KEY_CAPTURE_KIND_BY_MODE = {
    "prompt_key": PROMPT_KEY_INTERNAL_CAPTURE_KIND,
    "prompt_key_input": PROMPT_KEY_INPUT_INTERNAL_CAPTURE_KIND,
}
_SLICE_RE = re.compile(
    r"slice\((None|-?[0-9]+), (None|-?[0-9]+), (None|-?[0-9]+)\)"
)


def prompt_index_cache_artifact_kind(layer_id: int) -> str:
    """Artifact kind of the sealed legacy prompt index cache of one layer."""

    if layer_id not in FULL_INDEXER_LAYERS:
        raise ValueError(f"layer {layer_id} owns no DSA prompt index cache")
    return f"glm52_legacy_layer{int(layer_id)}_prompt_index_cache"


MLA_LAYER_COUNT = 78


def expected_prompt_cache_slot(layer_id: int) -> int:
    """Legacy runner ``kv_caches`` slot holding ``layer_id``'s indexer cache.

    The legacy runner orders cache slots by module registration.  In the pinned
    vLLM ``DeepseekV2MLAAttention`` every MLA layer constructs an ``Indexer``
    (which registers its ``DeepseekV32IndexerCache``) before its
    ``MultiHeadLatentAttentionWrapper`` registers the main cache; skip-top-k
    layers register the indexer cache too.  So layer ``L`` owns slots ``2L``
    (index cache, 128 lanes) and ``2L + 1`` (main cache, 640 lanes); the sealed
    layer-0 captures dumped slot 0 as the index cache and slot 1 as the main
    cache.  A wrong slot cannot seal: the loader refuses any other geometry.
    """

    if (
        not isinstance(layer_id, int)
        or isinstance(layer_id, bool)
        or not 0 <= layer_id < MLA_LAYER_COUNT
    ):
        raise ValueError(f"layer {layer_id} is not an MLA layer")
    return 2 * int(layer_id)


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
    layer_id: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_dump_dir", Path(self.source_dump_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if (
            not isinstance(self.layer_id, int)
            or isinstance(self.layer_id, bool)
            or self.layer_id not in FULL_INDEXER_LAYERS
        ):
            raise ValueError("prompt-cache layer must own a DSA prompt index cache")
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

    @property
    def cache_slot(self) -> int:
        return expected_prompt_cache_slot(self.layer_id)

    @property
    def layer_name(self) -> str:
        return LEGACY_ATTENTION_LAYER_NAME.format(layer=self.layer_id)

    @property
    def artifact_kind(self) -> str:
        return prompt_index_cache_artifact_kind(self.layer_id)


@dataclass(frozen=True, slots=True)
class LegacyPromptKeyInternalConfig:
    """Immutable identities for one accepted prompt-key producer row."""

    source_dump_dir: Path
    output_dir: Path
    expected_run_tag: str
    expected_legacy_code_hash: str
    expected_oracle_pin: str
    expected_layer_name: str = "model.layers.0.self_attn.attn"
    expected_position: int = 113
    expected_process_count: int = 8
    expected_model_id: str = MODEL_ID
    expected_capture_mode: PromptKeyCaptureMode = "prompt_key"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_dump_dir", Path(self.source_dump_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        for name in ("expected_legacy_code_hash", "expected_oracle_pin"):
            value = getattr(self, name)
            if len(value) != 40 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"invalid prompt-key internal digest: {name}")
        if self.expected_position < 0 or self.expected_process_count <= 0:
            raise ValueError("prompt-key internal position/process count drifted")
        if not self.expected_run_tag or not self.expected_layer_name:
            raise ValueError("prompt-key internal identity is incomplete")
        if self.expected_capture_mode not in _PROMPT_KEY_FIELDS_BY_MODE:
            raise ValueError("prompt-key internal capture mode drifted")


def inspect_legacy_prompt_key_internal_capture(
    config: LegacyPromptKeyInternalConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validate and seal every replica of one accepted FP32 key row."""

    if config.output_dir.exists():
        raise FileExistsError(
            f"append-only prompt-key capture exists: {config.output_dir}"
        )
    errors = sorted(config.source_dump_dir.rglob("*.INTERNAL.ERROR.*"))
    if errors:
        raise ValueError(f"prompt-key observer error exists: {errors[0]}")
    safe_layer = config.expected_layer_name.replace("/", "_").replace(
        ".", "_"
    )
    paths = sorted(
        config.source_dump_dir.rglob(
            f"*.{safe_layer}.position{config.expected_position}.proc*.npz"
        )
    )
    if not paths or len(paths) > config.expected_process_count:
        raise ValueError(
            "prompt-key internal process coverage drifted: "
            f"found={len(paths)}"
        )
    field_shapes = _PROMPT_KEY_FIELDS_BY_MODE[
        config.expected_capture_mode
    ]
    fields = tuple(field_shapes)
    scalar_keys = {
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
    }
    expected_keys = scalar_keys | set(fields) | {
        f"{name}__dtype" for name in fields
    }
    canonical: dict[str, np.ndarray] | None = None
    process_indices: set[int] = set()
    records: list[dict[str, Any]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != expected_keys:
                raise ValueError(f"{path}: prompt-key internal keys drifted")
            expected_scalars = {
                "artifact_kind": _PROMPT_KEY_RAW_KIND_BY_MODE[
                    config.expected_capture_mode
                ],
                "format_version": FORMAT_VERSION,
                "capture_mode": config.expected_capture_mode,
                "process_count": config.expected_process_count,
                "layer_name": config.expected_layer_name,
                "position": config.expected_position,
                "source_row": config.expected_position,
                "run_tag": config.expected_run_tag,
                "code_hash": config.expected_legacy_code_hash,
                "oracle_pin": config.expected_oracle_pin,
                "model_id": config.expected_model_id,
            }
            for name, expected in expected_scalars.items():
                value = payload[name]
                if value.shape != () or value.item() != expected:
                    raise ValueError(
                        f"{path}: {name} identity drifted"
                    )
            process_index = int(payload["process_index"].item())
            if not 0 <= process_index < config.expected_process_count or (
                process_index in process_indices
            ):
                raise ValueError(f"{path}: prompt-key process identity drifted")
            process_indices.add(process_index)
            current: dict[str, np.ndarray] = {}
            for name in fields:
                value = np.ascontiguousarray(payload[name])
                if (
                    str(payload[f"{name}__dtype"].item()) != "float32"
                    or value.dtype != np.float32
                    or value.shape != field_shapes[name]
                    or not np.isfinite(value).all()
                ):
                    raise ValueError(
                        f"{path}: prompt-key {name} tensor drifted"
                    )
                current[name] = value
            if canonical is None:
                canonical = current
            elif any(
                not np.array_equal(current[name], canonical[name])
                for name in fields
            ):
                raise ValueError("prompt-key replicas are not bitwise equal")
            records.append(
                {
                    "byte_count": path.stat().st_size,
                    "path": path.relative_to(config.source_dump_dir).as_posix(),
                    "process_index": process_index,
                    "sha256": _sha256_file(path),
                }
            )
    assert canonical is not None
    config.output_dir.mkdir(parents=True)
    tensor_path = config.output_dir / "accepted_prompt_key_states.npz"
    np.savez(tensor_path, **canonical)
    manifest = {
        "artifact_kind": _PROMPT_KEY_CAPTURE_KIND_BY_MODE[
            config.expected_capture_mode
        ],
        "capture_mode": config.expected_capture_mode,
        "capture_process_indices": sorted(process_indices),
        "diagnostic_only": True,
        "fields": {
            name: {
                "dtype": "float32",
                "sha256": _array_sha256(canonical[name]),
                "shape": list(field_shapes[name]),
            }
            for name in fields
        },
        "format_version": FORMAT_VERSION,
        "layer_name": config.expected_layer_name,
        "legacy_code_hash": config.expected_legacy_code_hash,
        "model_id": config.expected_model_id,
        "oracle_pin": config.expected_oracle_pin,
        "performance_claim": False,
        "position": config.expected_position,
        "process_count": config.expected_process_count,
        "process_files": sorted(
            records, key=lambda value: value["process_index"]
        ),
        "run_tag": config.expected_run_tag,
        "tensor_file": {
            "byte_count": tensor_path.stat().st_size,
            "filename": tensor_path.name,
            "sha256": _sha256_file(tensor_path),
        },
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    (config.output_dir / "capture.json").write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    return manifest, canonical


def inspect_prompt_key_internal_capture_artifact(
    artifact_dir: Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_capture_mode: PromptKeyCaptureMode | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Verify and load an already-sealed accepted prompt-key capture."""

    artifact_dir = Path(artifact_dir)
    manifest = json.loads((artifact_dir / "capture.json").read_text())
    capture_mode = manifest.get("capture_mode")
    if capture_mode not in _PROMPT_KEY_FIELDS_BY_MODE or (
        manifest.get("artifact_kind")
        != _PROMPT_KEY_CAPTURE_KIND_BY_MODE[capture_mode]
    ) or manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported prompt-key internal capture artifact")
    if expected_capture_mode is not None and capture_mode != (
        expected_capture_mode
    ):
        raise ValueError("prompt-key internal capture mode identity drifted")
    if manifest.get("model_id") != MODEL_ID or (
        manifest.get("diagnostic_only") is not True
    ) or manifest.get("performance_claim") is not False:
        raise ValueError("prompt-key internal capture scope/model drifted")
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise ValueError("prompt-key internal capture manifest checksum mismatch")
    if expected_manifest_sha256 is not None and (
        manifest["manifest_sha256"] != expected_manifest_sha256
    ):
        raise ValueError("prompt-key internal capture identity drifted")

    record = manifest.get("tensor_file", {})
    tensor_path = artifact_dir / str(record.get("filename", ""))
    if tensor_path.stat().st_size != record.get("byte_count") or (
        _sha256_file(tensor_path) != record.get("sha256")
    ):
        raise ValueError("prompt-key internal capture tensor integrity failed")
    field_shapes = _PROMPT_KEY_FIELDS_BY_MODE[capture_mode]
    fields = tuple(field_shapes)
    with np.load(tensor_path, allow_pickle=False) as payload:
        if set(payload.files) != set(fields):
            raise ValueError("prompt-key internal capture field set drifted")
        states = {
            name: np.ascontiguousarray(payload[name]) for name in fields
        }
    for name, value in states.items():
        field = manifest.get("fields", {}).get(name, {})
        if value.shape != field_shapes[name] or value.dtype != np.float32 or (
            not np.isfinite(value).all()
        ):
            raise ValueError(f"prompt-key internal capture {name} drifted")
        if field != {
            "dtype": "float32",
            "sha256": _array_sha256(value),
            "shape": list(field_shapes[name]),
        }:
            raise ValueError(
                f"prompt-key internal capture {name} manifest drifted"
            )
    return manifest, states


def compare_prompt_projection_input(
    accepted: np.ndarray,
    observed: np.ndarray,
) -> dict[str, Any]:
    """Compare the actual FP32 6,144-wide row entering key projection."""

    expected = np.ascontiguousarray(accepted)
    actual = np.ascontiguousarray(observed)
    if expected.shape != (6144,) or actual.shape != (6144,) or (
        expected.dtype != np.float32 or actual.dtype != np.float32
    ):
        raise ValueError("prompt projection-input tensor drifted")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("prompt projection-input tensor is non-finite")
    mismatch_indices = np.flatnonzero(expected != actual)
    absolute_error = np.abs(expected.astype(np.float64) - actual)
    exact = mismatch_indices.size == 0
    return {
        "accepted_sha256": _array_sha256(expected),
        "elementwise_exact": exact,
        "first_mismatch_dimension": (
            None if exact else int(mismatch_indices[0])
        ),
        "max_absolute_error": float(absolute_error.max(initial=0.0)),
        "mean_absolute_error": float(absolute_error.mean()),
        "mismatch_count": int(mismatch_indices.size),
        "observed_sha256": _array_sha256(actual),
        "p99_absolute_error": float(
            np.quantile(absolute_error, 0.99, method="higher")
        ),
        "signed_mean_error": float(
            (actual.astype(np.float64) - expected).mean()
        ),
    }


def compare_prompt_key_internal_states(
    accepted: dict[str, np.ndarray],
    observed: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compare the three ordered FP32 prompt-key producer boundaries."""

    fields = (
        "pre_layer_norm_key",
        "pre_rope_key",
        "post_rope_key",
    )
    if set(accepted) != set(fields) or set(observed) != set(fields):
        raise ValueError("prompt-key comparison field set drifted")
    records: dict[str, Any] = {}
    first_divergent_field: str | None = None
    for name in fields:
        expected = np.ascontiguousarray(accepted[name])
        actual = np.ascontiguousarray(observed[name])
        if expected.shape != (128,) or actual.shape != (128,) or (
            expected.dtype != np.float32 or actual.dtype != np.float32
        ):
            raise ValueError(f"prompt-key comparison tensor drifted: {name}")
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise ValueError(f"prompt-key comparison tensor is non-finite: {name}")
        mismatch = expected != actual
        mismatch_indices = np.flatnonzero(mismatch)
        absolute_error = np.abs(expected.astype(np.float64) - actual)
        exact = mismatch_indices.size == 0
        if not exact and first_divergent_field is None:
            first_divergent_field = name
        records[name] = {
            "accepted_sha256": _array_sha256(expected),
            "elementwise_exact": exact,
            "first_mismatch_dimension": (
                None if exact else int(mismatch_indices[0])
            ),
            "max_absolute_error": float(absolute_error.max(initial=0.0)),
            "mean_absolute_error": float(absolute_error.mean()),
            "mismatch_count": int(mismatch_indices.size),
            "observed_sha256": _array_sha256(actual),
            "p99_absolute_error": float(
                np.quantile(absolute_error, 0.99, method="higher")
            ),
            "signed_mean_error": float(
                (actual.astype(np.float64) - expected).mean()
            ),
        }
    classification = {
        "pre_layer_norm_key": "projection_association",
        "pre_rope_key": "key_layer_norm_association",
        "post_rope_key": "rope_association",
        None: "producer_states_elementwise_exact",
    }[first_divergent_field]
    return {
        "all_fields_elementwise_exact": first_divergent_field is None,
        "classification": classification,
        "fields": records,
        "first_divergent_field": first_divergent_field,
    }


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
            if payload["layer_indices"].tolist() != [config.cache_slot]:
                raise ValueError(
                    "prompt-cache source must contain exactly the layer's "
                    f"indexer cache slot {config.cache_slot}"
                )
            slot_key = f"layer{config.cache_slot}"
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
            if str(payload[f"{slot_key}__dtype"]) != "bfloat16":
                raise ValueError("prompt-cache source dtype drifted")

            shape = tuple(int(value) for value in payload[f"{slot_key}__shape"])
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
            shard_count = int(payload[f"{slot_key}__nshards"])
            if shard_count != config.expected_local_replication:
                raise ValueError("prompt-cache local replica coverage drifted")
            for shard_index in range(shard_count):
                key = f"{slot_key}__shard{shard_index}"
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
    """Reconstruct and seal one layer's logical BF16 prompt keys from DCP dumps."""

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
            "artifact_kind": config.artifact_kind,
            "format_version": str(FORMAT_VERSION),
        },
    )
    manifest: dict[str, Any] = {
        "artifact_kind": config.artifact_kind,
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
    if config.layer_id != 0:
        # Layer-0 manifests keep their sealed byte format; deeper layers bind
        # the layer identity and the legacy cache slot they were dumped from.
        manifest["layer_id"] = config.layer_id
        manifest["cache_slot"] = config.cache_slot
        manifest["layer_name"] = config.layer_name
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
    layer_id = manifest.get("layer_id", 0)
    if (
        not isinstance(layer_id, int)
        or isinstance(layer_id, bool)
        or layer_id not in FULL_INDEXER_LAYERS
    ):
        raise ValueError("unsupported prompt index-cache layer")
    artifact_kind = prompt_index_cache_artifact_kind(layer_id)
    if manifest.get("artifact_kind") != artifact_kind or (
        manifest.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError("unsupported prompt index-cache artifact")
    if layer_id != 0 and (
        manifest.get("cache_slot") != expected_prompt_cache_slot(layer_id)
        or manifest.get("layer_name")
        != LEGACY_ATTENTION_LAYER_NAME.format(layer=layer_id)
    ):
        raise ValueError("prompt index-cache layer binding drifted")
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
            "artifact_kind": artifact_kind,
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


def _classify_chunk_wk_feature_slices(
    lowered_hlo: str,
) -> dict[str, Any]:
    """Recognize TPU streaming slices of the public adapted ``wk`` tensor.

    TPU HLO may stage the 128 output features of ``wk`` as four 32-row
    transfers and concatenate them before the projection.  Those rows are
    weight features, not the forbidden batch-32 hidden-state bucket.  Keep the
    exception deliberately narrow: one source ``wk`` parameter, the complete
    four-slice partition, one matching ``slice-done`` per transfer, and no
    other instruction carrying the otherwise-forbidden shape.
    """

    lines = lowered_hlo.splitlines()
    shape = "f32[32,6144]"
    shape_line_indices = {
        index for index, line in enumerate(lines) if shape in line
    }
    if not shape_line_indices:
        return {
            "done_count": 0,
            "shape_line_count": 0,
            "slice_count": 0,
            "slice_spans": [],
            "unclassified_line_count": 0,
            "valid": True,
            "wk_parameter_count": 0,
        }

    wk_parameter_pattern = re.compile(
        r'^\s*(%\S+)\s*=\s*f32\[128,6144\].*parameter\([^)]*\).*'
        r'metadata=\{op_name="wk_weight"'
    )
    wk_parameter_names = {
        match.group(1)
        for line in lines
        if (match := wk_parameter_pattern.search(line)) is not None
    }
    slice_start_pattern = re.compile(
        r"^\s*(%\S+)\s*=.*f32\[32,6144\].*"
        r"slice-start\((%\S+)\),\s*"
        r"slice=\{\[([0-9]+):([0-9]+)\],\s*\[0:6144\]\}"
    )
    slice_done_pattern = re.compile(
        r"^\s*(%\S+)\s*=\s*f32\[32,6144\].*"
        r"slice-done\((%\S+)\)"
    )

    starts: list[tuple[int, str, tuple[int, int]]] = []
    for index, line in enumerate(lines):
        match = slice_start_pattern.search(line)
        if match is None or match.group(2) not in wk_parameter_names:
            continue
        starts.append(
            (
                index,
                match.group(1),
                (int(match.group(3)), int(match.group(4))),
            )
        )
    start_names = {name for _, name, _ in starts}
    dones: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = slice_done_pattern.search(line)
        if match is None or match.group(2) not in start_names:
            continue
        dones.append((index, match.group(2)))

    expected_spans = {(0, 32), (32, 64), (64, 96), (96, 128)}
    classified_indices = {
        *(index for index, _, _ in starts),
        *(index for index, _ in dones),
    }
    spans = [span for _, _, span in starts]
    valid = (
        len(wk_parameter_names) == 1
        and len(starts) == 4
        and len(start_names) == 4
        and set(spans) == expected_spans
        and len(dones) == 4
        and {name for _, name in dones} == start_names
        and classified_indices == shape_line_indices
    )
    return {
        "done_count": len(dones),
        "shape_line_count": len(shape_line_indices),
        "slice_count": len(starts),
        "slice_spans": [list(span) for span in sorted(spans)],
        "unclassified_line_count": len(
            shape_line_indices - classified_indices
        ),
        "valid": valid,
        "wk_parameter_count": len(wk_parameter_names),
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


def validate_prompt_projection_input_hlo(
    optimized_hlo: str,
    *,
    prompt_chunk: int = 2048,
    unique_token_count: int = 37,
) -> dict[str, Any]:
    """Require one gathered M2048 input-RMS body and its FP32 output."""

    if not isinstance(optimized_hlo, str) or not optimized_hlo.strip():
        raise ValueError("prompt projection-input HLO must be non-empty text")
    for name, value in (
        ("prompt_chunk", prompt_chunk),
        ("unique_token_count", unique_token_count),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"prompt projection-input {name} must be positive")

    lowered = optimized_hlo.lower()
    lines = lowered.splitlines()
    gather_fusion_lines = [
        line
        for line in lines
        if re.search(rf"= bf16\[{prompt_chunk},6144\].*fusion\(", line)
        and "kind=kcustom" in line
        and "jit(_take)/gather" in line
    ]
    raw_gather_lines = [
        line
        for line in lines
        if re.search(rf"= bf16\[{prompt_chunk},6144\].*gather\(", line)
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
        for line in lines
    )
    while_lines = [line for line in lines if re.search(r"\bwhile\(", line)]
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
    for instruction in ("send(", "recv(", "scatter(", "convolution("):
        if instruction in lowered:
            forbidden_operations[instruction[:-1]] = lowered.count(
                instruction
            )
    forbidden_shapes = [
        shape
        for shape in (
            "bf16[8155,6144]",
            "f32[8155,6144]",
            "bf16[8192,6144]",
            "f32[8192,6144]",
            "bf16[4,2048,6144]",
            "f32[4,2048,6144]",
        )
        if shape in lowered
    ]
    dead_row_parameters = [
        line
        for line in lines
        if re.search(r"(?:bf16|f32)\[32,6144\].*parameter\(", line)
    ]
    if dead_row_parameters:
        forbidden_shapes.append("dead_parameter[32,6144]")
    required_shapes = {
        "fp32_normalized_chunk_output": bool(
            re.search(
                rf"entry_computation_layout=.*->f32\[{prompt_chunk},6144\]",
                lowered,
            )
        ),
        "input_norm_weight_parameter": bool(
            re.search(r"bf16\[6144\].*parameter\(", lowered)
        ),
        "prompt_row_parameter": bool(
            re.search(rf"s32\[{prompt_chunk}\].*parameter\(", lowered)
        ),
        "unique_embedding_parameter": bool(
            re.search(
                rf"bf16\[{unique_token_count},6144\].*parameter\(",
                lowered,
            )
        ),
    }
    violations: list[str] = []
    if len(physical_gather_lines) != 1:
        violations.append(
            "expected one physical M2048 embedding gather, "
            f"found {len(physical_gather_lines)}"
        )
    if not gather_coupled_input_rms:
        violations.append("input RMS reduction does not consume the gather")
    if while_lines:
        violations.append(
            f"expected no projection-input loop, found {len(while_lines)}"
        )
    if forbidden_operations:
        violations.append(f"forbidden operations: {forbidden_operations}")
    if forbidden_shapes:
        violations.append(
            f"forbidden full-prompt/dead-row shapes: {forbidden_shapes}"
        )
    missing = sorted(
        name for name, present in required_shapes.items() if not present
    )
    if missing:
        violations.append(f"missing projection-input HLO shapes: {missing}")
    return {
        "dead_row_parameter_count": len(dead_row_parameters),
        "forbidden_operations": forbidden_operations,
        "forbidden_shapes": forbidden_shapes,
        "gather_coupled_input_rms": gather_coupled_input_rms,
        "loop_count": len(while_lines),
        "passed": not violations,
        "physical_embedding_gather_count": len(physical_gather_lines),
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
        "accepted_xla_m2048_gather_cache_write_bf16_weight_divide_sqrt": (
            "xla_chunk_gather_cache_write_bf16_weight",
            "divide_sqrt",
        ),
        (
            "accepted_xla_m2048_gather_cache_write_bf16_weight_"
            "divide_sqrt_source_rope"
        ): (
            "xla_chunk_gather_cache_write_bf16_weight_source_rope",
            "divide_sqrt",
        ),
        (
            "accepted_xla_m2048_gather_cache_write_bf16_weight_"
            "divide_sqrt_source_rope_states"
        ): (
            "xla_chunk_gather_cache_write_bf16_weight_source_rope_states",
            "divide_sqrt",
        ),
        (
            "accepted_xla_m2048_gather_cache_write_fp32_weight_"
            "divide_sqrt_source_rope"
        ): (
            "xla_chunk_gather_cache_write_fp32_weight_source_rope",
            "divide_sqrt",
        ),
        (
            "accepted_xla_m2048_gather_cache_write_fp32_weight_"
            "divide_sqrt_source_rope_states"
        ): (
            "xla_chunk_gather_cache_write_fp32_weight_source_rope_states",
            "divide_sqrt",
        ),
        (
            "accepted_xla_m64_lax_map_gather_cache_write_fp32_weight_"
            "divide_sqrt_source_rope"
        ): (
            "xla_m64_lax_map_gather_cache_write_fp32_weight_source_rope",
            "divide_sqrt",
        ),
        (
            "accepted_xla_m64_lax_map_gather_cache_write_fp32_weight_"
            "divide_sqrt_source_rope_states"
        ): (
            "xla_m64_lax_map_gather_cache_write_fp32_weight_source_rope_states",
            "divide_sqrt",
        ),
        (
            "accepted_xla_m64_projection_keynorm_lax_map_gather_cache_write_"
            "fp32_weight_divide_sqrt_source_rope"
        ): (
            "xla_m64_projection_keynorm_lax_map_gather_cache_write_"
            "fp32_weight_source_rope",
            "divide_sqrt",
        ),
        (
            "accepted_xla_m64_projection_keynorm_lax_map_gather_cache_write_"
            "fp32_weight_divide_sqrt_source_rope_states"
        ): (
            "xla_m64_projection_keynorm_lax_map_gather_cache_write_"
            "fp32_weight_source_rope_states",
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
        wk_feature_slices = _classify_chunk_wk_feature_slices(lowered)
        m64_lax_map_backend = backend.startswith(
            (
                "xla_m64_lax_map_",
                "xla_m64_projection_keynorm_lax_map_",
            )
        )
        physical_key_norm_backend = backend.startswith(
            "xla_m64_projection_keynorm_lax_map_"
        )
        physical_projection_rows = 64 if m64_lax_map_backend else prompt_chunk
        physical_key_norm_sqrt_lines = [
            line
            for line in lowered.splitlines()
            if re.search(r"= f32\[64\].*(?:fusion|sqrt)\(", line)
            and "/sqrt" in line
        ]
        physical_key_norm_affine_lines = [
            line
            for line in lowered.splitlines()
            if re.search(r"= f32\[64,128\].*(?:fusion|add)\(", line)
            and "/add" in line
        ]
        grouped_key_norm_sqrt_lines = [
            line
            for line in lowered.splitlines()
            if re.search(r"= f32\[32,64\].*(?:fusion|sqrt)\(", line)
            and "/sqrt" in line
        ]
        convolution_lines = [
            line
            for line in lowered.splitlines()
            if re.search(
                rf"= f32\[{physical_projection_rows},128\].* convolution\(",
                line,
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
                f"bf16[{prompt_token_count},6144]",
                f"f32[{prompt_token_count},6144]",
                "bf16[8192,6144]",
                "f32[8192,6144]",
                "bf16[4,2048,6144]",
                "f32[4,2048,6144]",
            )
            if shape in lowered
        ]
        if not wk_feature_slices["valid"]:
            forbidden_shapes.append("f32[32,6144]")
        cache_write_backend = backend in (
            "xla_chunk_gather_cache_write_bf16_weight",
            "xla_chunk_gather_cache_write_bf16_weight_source_rope",
            "xla_chunk_gather_cache_write_bf16_weight_source_rope_states",
            "xla_chunk_gather_cache_write_fp32_weight_source_rope",
            "xla_chunk_gather_cache_write_fp32_weight_source_rope_states",
            "xla_m64_lax_map_gather_cache_write_fp32_weight_source_rope",
            "xla_m64_lax_map_gather_cache_write_fp32_weight_source_rope_states",
            "xla_m64_projection_keynorm_lax_map_gather_cache_write_"
            "fp32_weight_source_rope",
            "xla_m64_projection_keynorm_lax_map_gather_cache_write_"
            "fp32_weight_source_rope_states",
        )
        source_rope_backend = backend in (
            "xla_chunk_gather_cache_write_bf16_weight_source_rope",
            "xla_chunk_gather_cache_write_bf16_weight_source_rope_states",
            "xla_chunk_gather_cache_write_fp32_weight_source_rope",
            "xla_chunk_gather_cache_write_fp32_weight_source_rope_states",
            "xla_m64_lax_map_gather_cache_write_fp32_weight_source_rope",
            "xla_m64_lax_map_gather_cache_write_fp32_weight_source_rope_states",
            "xla_m64_projection_keynorm_lax_map_gather_cache_write_"
            "fp32_weight_source_rope",
            "xla_m64_projection_keynorm_lax_map_gather_cache_write_"
            "fp32_weight_source_rope_states",
        )
        state_backend = backend.endswith("_source_rope_states")
        fp32_weight_backend = backend.startswith(
            "xla_chunk_gather_cache_write_fp32_weight"
        ) or backend.startswith("xla_m64_")
        required_shapes = {
            "accepted_adapted_wk": "f32[128,6144]" in lowered,
            "chunk_projection": f"f32[{prompt_chunk},128]" in lowered,
        }
        if m64_lax_map_backend:
            required_shapes.update(
                {
                    "physical_m64_projection": "f32[64,128]" in lowered,
                    "physical_m64_projection_input": (
                        "bf16[64,6144]" in lowered
                    ),
                }
            )
        if physical_key_norm_backend:
            required_shapes.update(
                {
                    "physical_m64_key_norm_sqrt": bool(
                        physical_key_norm_sqrt_lines
                    ),
                    "physical_m64_key_norm_affine": bool(
                        physical_key_norm_affine_lines
                    ),
                }
            )
        if cache_write_backend:
            required_shapes.update(
                {
                    "accepted_cache_parameter": bool(
                        re.search(
                            r"bf16\[24,16,32,128\].*parameter\(",
                            lowered,
                        )
                    ),
                    "accepted_cache_result": bool(
                        re.search(
                            (
                                r"entry_computation_layout=.*->\("
                                r"bf16\[24,16,32,128\].*"
                                r"f32\[2048,128\].*"
                                r"f32\[2048,128\].*"
                                r"f32\[2048,128\]"
                                if state_backend
                                else r"entry_computation_layout=.*->"
                                r"bf16\[24,16,32,128\]"
                            ),
                            lowered,
                        )
                    ),
                    "live_block_table_parameter": bool(
                        re.search(r"s32\[16\].*parameter\(", lowered)
                    ),
                    "post_rope_fp32_chunk": (
                        f"f32[{prompt_chunk},128]" in lowered
                    ),
                }
            )
        else:
            required_shapes["chunk_key_output"] = (
                f"bf16[{prompt_chunk},128]" in lowered
            )
        if backend in (
            "xla_chunk_gather_bf16_weight",
            "xla_chunk_gather_cache_write_bf16_weight",
            "xla_chunk_gather_cache_write_bf16_weight_source_rope",
            "xla_chunk_gather_cache_write_bf16_weight_source_rope_states",
            "xla_chunk_gather_cache_write_fp32_weight_source_rope",
            "xla_chunk_gather_cache_write_fp32_weight_source_rope_states",
            "xla_m64_lax_map_gather_cache_write_fp32_weight_source_rope",
            "xla_m64_lax_map_gather_cache_write_fp32_weight_source_rope_states",
            "xla_m64_projection_keynorm_lax_map_gather_cache_write_"
            "fp32_weight_source_rope",
            "xla_m64_projection_keynorm_lax_map_gather_cache_write_"
            "fp32_weight_source_rope_states",
        ):
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
        convolution_weight_f32 = False
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
                convolution_weight_f32 = any(
                    re.search(
                        rf"^\s*{re.escape(weight_name)}\s*=\s*"
                        r"f32\[128,6144\]",
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
        physical_scatter_lines = [
            line
            for line in lowered.splitlines()
            if re.search(r"\bscatter\(", line)
        ]
        cache_scatter_update_bf16 = False
        if len(physical_scatter_lines) == 1:
            operands = re.search(
                r"scatter\(([^)]+)\)", physical_scatter_lines[0]
            )
            if operands is not None:
                operand_names = re.findall(
                    r"%[a-z0-9_.-]+", operands.group(1)
                )
                if operand_names:
                    update_name = operand_names[-1]
                    cache_scatter_update_bf16 = any(
                        re.search(
                            rf"^\s*{re.escape(update_name)}\s*=\s*"
                            rf"bf16\[{prompt_chunk},128\]",
                            line,
                        )
                        for line in lowered.splitlines()
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
                f"expected one physical M{physical_projection_rows} wk "
                "convolution, "
                f"found {len(convolution_lines)}"
            )
        expected_loop_count = 1 if m64_lax_map_backend else 0
        if len(while_lines) != expected_loop_count:
            violations.append(
                f"expected {expected_loop_count} projection-map loops, "
                f"found {len(while_lines)}"
            )
        if physical_key_norm_backend and grouped_key_norm_sqrt_lines:
            violations.append(
                "physical M64 key LayerNorm retained grouped [32,64] sqrt"
            )
        if backend in (
            "xla_chunk_bf16_weight",
            "xla_chunk_gather_bf16_weight",
            "xla_chunk_gather_cache_write_bf16_weight",
            "xla_chunk_gather_cache_write_bf16_weight_source_rope",
            "xla_chunk_gather_cache_write_bf16_weight_source_rope_states",
        ):
            if len(bf16_weight_conversion_lines) != 1:
                violations.append(
                    "expected one explicit FP32-to-BF16 wk conversion, "
                    f"found {len(bf16_weight_conversion_lines)}"
                )
            if not convolution_weight_bf16:
                violations.append(
                    f"M{physical_projection_rows} convolution does not "
                    "consume a BF16 wk operand"
                )
        elif bf16_weight_conversion_lines:
            violations.append(
                "unexpected FP32-to-BF16 wk conversion in FP32 chunk candidate"
            )
        if fp32_weight_backend and not convolution_weight_f32:
            violations.append(
                f"M{physical_projection_rows} convolution does not "
                "consume an FP32 wk operand"
            )
        if backend in (
            "xla_chunk_gather_bf16_weight",
            "xla_chunk_gather_cache_write_bf16_weight",
            "xla_chunk_gather_cache_write_bf16_weight_source_rope",
            "xla_chunk_gather_cache_write_bf16_weight_source_rope_states",
            "xla_chunk_gather_cache_write_fp32_weight_source_rope",
            "xla_chunk_gather_cache_write_fp32_weight_source_rope_states",
            "xla_m64_lax_map_gather_cache_write_fp32_weight_source_rope",
            "xla_m64_lax_map_gather_cache_write_fp32_weight_source_rope_states",
            "xla_m64_projection_keynorm_lax_map_gather_cache_write_"
            "fp32_weight_source_rope",
            "xla_m64_projection_keynorm_lax_map_gather_cache_write_"
            "fp32_weight_source_rope_states",
        ):
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
        if cache_write_backend:
            if len(physical_scatter_lines) != 1:
                violations.append(
                    "expected one physical flat BF16 cache scatter, "
                    f"found {len(physical_scatter_lines)}"
                )
            elif not cache_scatter_update_bf16:
                violations.append(
                    "flat cache scatter does not consume BF16 chunk updates"
                )
        elif physical_scatter_lines:
            violations.append(
                "unexpected cache scatter in compact-key candidate"
            )
        rotary_power_lines = [
            line
            for line in lowered.splitlines()
            if re.search(r"= f32\[32\].*\bpower\(", line)
        ]
        rotary_cosine_lines = [
            line
            for line in lowered.splitlines()
            if re.search(
                rf"= f32\[{prompt_chunk},32\].*\bcosine\(", line
            )
        ]
        rotary_sine_lines = [
            line
            for line in lowered.splitlines()
            if re.search(
                rf"= f32\[{prompt_chunk},32\].*\bsine\(", line
            )
        ]
        rotary_theta_constant = "constant(8e+06)" in lowered
        rotary_exponent_constant = "constant(0.015625)" in lowered
        if source_rope_backend and (
            len(rotary_power_lines) != 1
            or len(rotary_cosine_lines) != 1
            or len(rotary_sine_lines) != 1
            or not rotary_theta_constant
            or not rotary_exponent_constant
        ):
            violations.append(
                "literal accepted RoPE physical identity drifted"
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
            "convolution_weight_f32": convolution_weight_f32,
            "gather_coupled_input_rms": gather_coupled_input_rms,
            "physical_embedding_gather_count": len(physical_gather_lines),
            "physical_cache_scatter_count": len(physical_scatter_lines),
            "cache_scatter_update_bf16": cache_scatter_update_bf16,
            "rotary": {
                "cosine_count": len(rotary_cosine_lines),
                "exponent_constant": rotary_exponent_constant,
                "power_count": len(rotary_power_lines),
                "sine_count": len(rotary_sine_lines),
                "source_literal": source_rope_backend,
                "theta_constant": rotary_theta_constant,
            },
            "forbidden_operations": forbidden_operations,
            "forbidden_shapes": forbidden_shapes,
            "loop_count": len(while_lines),
            "passed": not violations,
            "physical_key_norm": {
                "affine_count": len(physical_key_norm_affine_lines),
                "enabled": physical_key_norm_backend,
                "grouped_sqrt_count": len(
                    grouped_key_norm_sqrt_lines
                ),
                "sqrt_count": len(physical_key_norm_sqrt_lines),
            },
            "required_shapes": required_shapes,
            "violations": violations,
            "wk_feature_slices": wk_feature_slices,
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
