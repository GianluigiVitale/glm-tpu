"""Fail-closed scaffold for PP16's persistent two-way feature state.

This module proves only the CPU/StableHLO and checkpoint-admission boundary.
It does not authorize a TPU acquisition or claim Gate-D correctness.  The
candidate keeps the dense update and its input residual in one packed local
``[2, 1, 3072]`` BF16 state, uses one local scalar RMS reduction, and gathers
only the completed normalized row needed by the already-proven full-N82 query
projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import struct
from typing import Any, Mapping, Sequence

import ml_dtypes
import numpy as np

from ..errors import BenchmarkValidationError, HloContractViolationError
from ..types import PlanName
from .transport_chain import (
    TransportChainConfig,
    TransportKind,
    validate_transport_pairs,
)


PP16_FEATURE2_HIDDEN_WIDTH = 6144
PP16_FEATURE2_SHARD_WIDTH = 3072
PP16_FEATURE2_CONTEXT_LENGTH = 8156
PP16_FEATURE2_RUNTIME_MANIFEST_SHA256 = (
    "b385458f233f21342855ac4c3373429c034a9e40bd85d638b16466199ff66bab"
)
PP16_FEATURE2_N82_WEIGHT_SHA256 = (
    "6e8b4efd28df17b493b1348a6ac82a05a468379e930a575a917636e1d506855d"
)
PP16_FEATURE2_N82_SCALE_SHA256 = (
    "3ca2712f5387e086f4251b5c1eea46b4bc2744911ca214c0ab9c7a3d70711cc5"
)
PP16_FEATURE2_ACCEPTED_DENSE_SHA256 = (
    "efde853254c03dd18a5f5f22733630ce0e785dfbb4eba09c41eea9085e47b4fc"
)
PP16_FEATURE2_ACCEPTED_CARRIED_SHA256 = (
    "35a601b7f174eb9204848757f709549a31e82774309929f4071c61849626044c"
)
PP16_FEATURE2_ACCEPTED_DENSE_HALF_SHA256 = (
    "021c7aba6a14b144ef26ac6e2bec0e39fe170e361bcc611ac0224c2083b8d473",
    "257d6b914d3b676db7155ce6e4f543d805bbca50d494bbf880c6a00ab631b9ad",
)
PP16_FEATURE2_ACCEPTED_INPUT_RESIDUAL_SHA256 = (
    "a105fdbd429adb1d06a70bf71598a72a91d7b6faa83360005487ce11ce099f8e"
)
PP16_FEATURE2_ACCEPTED_INPUT_RESIDUAL_HALF_SHA256 = (
    "703e8491819ef99fb5f32c62516cca48ea3690db8199466d0bfd2636c4092e8c",
    "303c4c70a61bbada41a99b8e2c8bfac291d5154ca8231bab57b8c2f9b9080ad4",
)
PP16_FEATURE2_ACCEPTED_CARRIED_HALF_SHA256 = (
    "d6d8ded037474d90b3b8f8a649dc52fc17a290a40cbdf95f64b76f6741f3486e",
    "86f5e4a963e5c2a91195e3b9d6215d9d52fac579cdac2339e96c0f8788d1d1dd",
)
PP16_FEATURE2_MINIMUM_STATE_BYTES_PER_DEVICE = 1_260_970_880
PP16_TWO_LAYER_STATE_BYTES_PER_DEVICE = 1_374_244_736
PP16_FEATURE2_SELECTED_WEIGHT_BYTES_PER_DEVICE = 1_199_760_512

_HOST_MARKERS = (
    "host_callback",
    "outside_compilation",
    "xla_ffi_python_cpu_callback",
    "xla_python_cpu_callback",
)
_COLLECTIVE_NAMES = (
    "all_gather",
    "all_reduce",
    "all_to_all",
    "collective_broadcast",
    "collective_permute",
    "reduce_scatter",
)
_MINIMUM_ACQUISITION_ROLES = frozenset(
    {
        "prompt.embedding",
        "layer0.attention_and_indexer",
        "layer0.dense",
        "layer1.feature2_input_boundary",
        "layer1.normalized_input",
        "layer1.n82_qkv_a",
        "layer1.tuple4_query_head",
        "layer1.wk_key_norm_rope",
        "layer1.index_cache_writes",
        "layer1.event1_scorer",
    }
)
_FORBIDDEN_ACQUISITION_PREFIXES = (
    "layer1.attention_output",
    "layer1.mlp",
    "layer1.dense",
    "layer2",
    "layers2_77",
    "final_norm",
    "logits",
    "sampling",
    "full_checkpoint",
)
_LAYER1_QUERY_TENSORS = frozenset(
    {
        "attention.slot_01.input_norm",
        "attention.slot_01.qkv_a.weight_bits",
        "attention.slot_01.qkv_a.scale_inv",
        "attention.slot_01.q_a_norm",
        "attention.slot_01.q_b.weight_bits",
        "attention.slot_01.q_b.scale_inv",
    }
)


@dataclass(frozen=True, slots=True)
class Feature2SplitRecord:
    source_sha256: str
    shard_sha256: tuple[str, str]
    shard_shape: tuple[int, int, int]
    concatenated_sha256: str
    exact: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "concatenated_sha256": self.concatenated_sha256,
            "exact": self.exact,
            "shard_sha256": list(self.shard_sha256),
            "shard_shape": list(self.shard_shape),
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class CompiledFeature2Boundary:
    compiled: Any
    dense_update: Any
    carried_residual: Any
    norm_weight: Any
    stablehlo: str
    optimized_hlo: str
    stablehlo_contract: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LoweredFeature2Transport:
    stablehlo: str
    stablehlo_contract: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Feature2TensorRead:
    device_slot: int
    filename: str
    name: str
    offset: int
    byte_count: int
    dtype: str
    shape: tuple[int, ...]
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "device_slot": self.device_slot,
            "dtype": self.dtype,
            "filename": self.filename,
            "name": self.name,
            "offset": self.offset,
            "sha256": self.sha256,
            "shape": list(self.shape),
        }


def split_feature2_bf16_bits(
    row_bits: np.ndarray,
) -> tuple[np.ndarray, Feature2SplitRecord]:
    """Split one semantic BF16 row and prove ordered concatenation is exact."""

    value = np.asarray(row_bits)
    if value.dtype != np.uint16 or value.shape != (1, PP16_FEATURE2_HIDDEN_WIDTH):
        raise BenchmarkValidationError(
            "feature2 split requires uint16 BF16 bits with shape [1,6144]"
        )
    value = np.ascontiguousarray(value)
    shards = np.ascontiguousarray(
        np.stack(
            (
                value[:, :PP16_FEATURE2_SHARD_WIDTH],
                value[:, PP16_FEATURE2_SHARD_WIDTH:],
            ),
            axis=0,
        )
    )
    concatenated = np.ascontiguousarray(np.concatenate(tuple(shards), axis=1))
    source_sha = sha256(value.tobytes(order="C")).hexdigest()
    concatenated_sha = sha256(concatenated.tobytes(order="C")).hexdigest()
    record = Feature2SplitRecord(
        source_sha256=source_sha,
        shard_sha256=tuple(
            sha256(shard.tobytes(order="C")).hexdigest() for shard in shards
        ),
        shard_shape=tuple(int(item) for item in shards.shape),
        concatenated_sha256=concatenated_sha,
        exact=bool(
            np.array_equal(concatenated, value)
            and source_sha == concatenated_sha
        ),
    )
    if not record.exact:
        raise BenchmarkValidationError("feature2 ordered shard concatenation drifted")
    return shards, record


def validate_feature2_accepted_state(
    dense_update_bits: np.ndarray, input_residual_bits: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pin DB550 inputs and prove their separately rounded carried output."""

    dense_update_bits = np.ascontiguousarray(dense_update_bits)
    input_residual_bits = np.ascontiguousarray(input_residual_bits)
    dense_shards, dense = split_feature2_bf16_bits(dense_update_bits)
    input_shards, input_residual = split_feature2_bf16_bits(
        input_residual_bits
    )
    carried_output_bits = np.ascontiguousarray(
        np.asarray(
            dense_update_bits.view(ml_dtypes.bfloat16).astype(np.float32)
            + input_residual_bits.view(ml_dtypes.bfloat16).astype(np.float32),
            dtype=ml_dtypes.bfloat16,
        )
    ).view(np.uint16)
    _, carried_output = split_feature2_bf16_bits(carried_output_bits)
    violations = []
    expected = (
        (
            "dense_update",
            dense,
            PP16_FEATURE2_ACCEPTED_DENSE_SHA256,
            PP16_FEATURE2_ACCEPTED_DENSE_HALF_SHA256,
        ),
        (
            "input_residual",
            input_residual,
            PP16_FEATURE2_ACCEPTED_INPUT_RESIDUAL_SHA256,
            PP16_FEATURE2_ACCEPTED_INPUT_RESIDUAL_HALF_SHA256,
        ),
    )
    for name, record, source_sha, half_shas in expected:
        if record.source_sha256 != source_sha:
            violations.append(f"accepted {name} source SHA drifted")
        if record.shard_sha256 != half_shas:
            violations.append(f"accepted {name} half SHA/order drifted")
    if carried_output.source_sha256 != PP16_FEATURE2_ACCEPTED_CARRIED_SHA256:
        violations.append("accepted carried output SHA drifted")
    if (
        carried_output.shard_sha256
        != PP16_FEATURE2_ACCEPTED_CARRIED_HALF_SHA256
    ):
        violations.append("accepted carried output half SHA/order drifted")
    if violations:
        raise BenchmarkValidationError(
            f"feature2 accepted state rejected: {violations}"
        )
    packed = np.ascontiguousarray(
        np.stack((dense_shards, input_shards), axis=1)
    )
    report = {
        "carried_output": carried_output.to_dict(),
        "dense_update": dense.to_dict(),
        "input_residual": input_residual.to_dict(),
        "packed_stage_payload_shape": [2, 2, 1, 3072],
        "passed": True,
        "violations": [],
    }
    return packed, report


def validate_feature2_n82_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_manifest_sha256: str = PP16_FEATURE2_RUNTIME_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Require identical owner-local layer-1 N82 tensors in PP16 stage zero."""

    violations: list[str] = []
    unhashed = dict(manifest)
    observed_manifest_sha = unhashed.pop("manifest_sha256", None)
    computed_manifest_sha = sha256(
        json.dumps(
            unhashed,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if observed_manifest_sha != computed_manifest_sha:
        violations.append("manifest self-hash is invalid")
    expected_header = {
        "attention_projection_layout": "fused_qkv_a_virtual_tp32_n82_v1",
        "manifest_sha256": expected_manifest_sha256,
        "model_id": "zai-org/GLM-5.2-FP8",
        "plan_id": "PP16_LP2",
    }
    for name, expected in expected_header.items():
        if manifest.get(name) != expected:
            violations.append(f"manifest {name} drifted")
    selected_files = [
        item
        for item in manifest.get("files", ())
        if item.get("stage_id") == 0 and item.get("device_slot") in (0, 1)
    ]
    if sorted(item.get("device_slot") for item in selected_files) != [0, 1]:
        violations.append("stage zero must contain exactly PP16 owner slots 0 and 1")
    expected_tensors = {
        "attention.slot_01.qkv_a.weight_bits": {
            "byte_count": 32 * 6144 * 82,
            "sha256": PP16_FEATURE2_N82_WEIGHT_SHA256,
            "transform": "fuse_qkv_a_output_shards",
            "sources": (
                {
                    "selected_shape": [2048, 6144],
                    "source_shape": [2048, 6144],
                    "source_tensor_name": "attention.slot_01.q_a.weight_bits",
                    "source_tensor_sha256": (
                        "3487ad2d9b2ff9d2bbf2c0405392d2466a5c3ccfe8017f617efbca0d9fbe25d7"
                    ),
                },
                {
                    "selected_shape": [576, 6144],
                    "source_shape": [576, 6144],
                    "source_tensor_name": "attention.slot_01.kv_a.weight_bits",
                    "source_tensor_sha256": (
                        "6b68a46ff3615547d8ba5879df2d0e5ecb8562b830f93bff9e446d5488c1e4ff"
                    ),
                },
            ),
        },
        "attention.slot_01.qkv_a.scale_inv": {
            "byte_count": 32 * 48 * 82 * 4,
            "sha256": PP16_FEATURE2_N82_SCALE_SHA256,
            "transform": "fuse_qkv_a_expanded_scales",
            "sources": (
                {
                    "selected_shape": [16, 48],
                    "source_shape": [16, 48],
                    "source_tensor_name": "attention.slot_01.q_a.scale_inv",
                    "source_tensor_sha256": (
                        "4432012ede431b2741f739ac29cef64cfbee445544afe38586c1c03a5805721b"
                    ),
                },
                {
                    "selected_shape": [5, 48],
                    "source_shape": [5, 48],
                    "source_tensor_name": "attention.slot_01.kv_a.scale_inv",
                    "source_tensor_sha256": (
                        "135aef9afce7e975c564cf48d2322ee1769f9dc0b089f27c19c406acac538059"
                    ),
                },
            ),
        },
    }
    owners: list[dict[str, Any]] = []
    for file_record in selected_files:
        slot = file_record.get("device_slot")
        filename = file_record.get("destination_filename")
        expected_filename = (
            f"base_decoder_runtime_feature/stage_00/device_slot_0{slot}.safetensors"
        )
        if (
            filename != expected_filename
            or file_record.get("device_id") != slot
            or file_record.get("stage_id") != 0
        ):
            violations.append(f"owner {slot} file identity drifted")
        tensors = {
            item.get("name"): item for item in file_record.get("tensors", ())
        }
        owner: dict[str, Any] = {"device_slot": slot, "filename": filename}
        for name, expected in expected_tensors.items():
            tensor = tensors.get(name)
            if tensor is None:
                violations.append(f"owner {slot} lacks {name}")
                continue
            for field in ("byte_count", "sha256", "transform"):
                expected_value = expected[field]
                if tensor.get(field) != expected_value:
                    violations.append(f"owner {slot} {name} {field} drifted")
            sources = tensor.get("sources", ())
            normalized_sources = tuple(
                {
                    key: source.get(key)
                    for key in (
                        "selected_shape",
                        "source_shape",
                        "source_tensor_name",
                        "source_tensor_sha256",
                    )
                }
                for source in sources
            )
            if (
                len(sources) != 2
                or normalized_sources != expected["sources"]
                or any(
                    source.get("source_device_slot") != slot
                    or source.get("source_filename") != filename
                    for source in sources
                )
            ):
                violations.append(f"owner {slot} {name} is not derived owner-locally")
            owner[name] = {
                "byte_count": tensor.get("byte_count"),
                "sha256": tensor.get("sha256"),
            }
        owners.append(owner)
    report = {
        "inferred_scale_shape": [32, 48, 82],
        "inferred_weight_shape": [32, 6144, 82],
        "manifest_sha256": computed_manifest_sha,
        "owners": owners,
        "passed": not violations,
        "violations": violations,
    }
    if violations:
        raise BenchmarkValidationError(f"feature2 N82 manifest rejected: {violations}")
    return report


def _feature2_selected_tensor_name(name: str) -> bool:
    return bool(
        name == "global.embedding"
        or name.startswith("attention.slot_00.")
        or name.startswith("indexer.slot_00.")
        or name.startswith("dense.slot_00.")
        or name in _LAYER1_QUERY_TENSORS
        or name.startswith("indexer.slot_01.")
    )


def read_feature2_owner_headers(
    runtime_root: Path, manifest: Mapping[str, Any]
) -> dict[str, bytes]:
    """Read only the two authenticated safetensors headers, never payloads."""

    root = Path(runtime_root)
    headers: dict[str, bytes] = {}
    records = [
        item
        for item in manifest.get("files", ())
        if item.get("stage_id") == 0 and item.get("device_slot") in (0, 1)
    ]
    for record in records:
        filename = record.get("destination_filename")
        header_bytes = record.get("header_bytes")
        if not isinstance(filename, str) or not isinstance(header_bytes, int):
            raise BenchmarkValidationError("feature2 owner header record is invalid")
        path = root / filename
        with path.open("rb") as stream:
            raw = stream.read(header_bytes)
        if len(raw) != header_bytes:
            raise BenchmarkValidationError(f"feature2 owner header {filename!r} is truncated")
        headers[filename] = raw
    return headers


def _decode_feature2_header(
    raw: bytes, *, filename: str, record: Mapping[str, Any]
) -> Mapping[str, Any]:
    if sha256(raw).hexdigest() != record.get("header_sha256"):
        raise BenchmarkValidationError(f"feature2 owner header {filename!r} SHA drifted")
    if len(raw) < 8:
        raise BenchmarkValidationError(f"feature2 owner header {filename!r} is truncated")
    encoded_length = struct.unpack("<Q", raw[:8])[0]
    if encoded_length + 8 != len(raw):
        raise BenchmarkValidationError(f"feature2 owner header {filename!r} length drifted")
    try:
        header = json.loads(raw[8:].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkValidationError(
            f"feature2 owner header {filename!r} is not valid JSON"
        ) from error
    metadata = header.get("__metadata__")
    expected_metadata = {
        "format": "pt",
        "greenfield_artifact_kind": "greenfield_feature_runtime_weight_file",
        "greenfield_destination_filename": filename,
        "greenfield_runtime_layout_hash": record.get("runtime_layout_hash"),
        "greenfield_source_runtime_manifest_sha256": record.get(
            "source_runtime_manifest_sha256"
        ),
    }
    if metadata != expected_metadata:
        raise BenchmarkValidationError(f"feature2 owner header {filename!r} metadata drifted")
    return header


def derive_feature2_tensor_allowlist(
    manifest: Mapping[str, Any],
    headers: Mapping[str, bytes],
    *,
    expected_manifest_sha256: str = PP16_FEATURE2_RUNTIME_MANIFEST_SHA256,
) -> tuple[Feature2TensorRead, ...]:
    """Derive exact selected tensor/range reads from the pinned final manifest."""

    validate_feature2_n82_manifest(
        manifest, expected_manifest_sha256=expected_manifest_sha256
    )
    records = [
        item
        for item in manifest.get("files", ())
        if item.get("stage_id") == 0 and item.get("device_slot") in (0, 1)
    ]
    expected_filenames = {
        item.get("destination_filename") for item in records
    }
    if set(headers) != expected_filenames:
        raise BenchmarkValidationError("feature2 owner header set drifted")
    allowlist: list[Feature2TensorRead] = []
    names_by_slot: dict[int, set[str]] = {}
    bytes_by_slot: dict[int, int] = {}
    for record in records:
        slot = int(record["device_slot"])
        filename = str(record["destination_filename"])
        header = _decode_feature2_header(headers[filename], filename=filename, record=record)
        tensor_records = {
            item.get("name"): item for item in record.get("tensors", ())
        }
        selected_names = {
            str(name)
            for name in tensor_records
            if isinstance(name, str) and _feature2_selected_tensor_name(name)
        }
        if len(selected_names) != 39:
            raise BenchmarkValidationError(
                f"feature2 owner {slot} selected tensor count drifted: {len(selected_names)}"
            )
        names_by_slot[slot] = selected_names
        ranges: list[tuple[int, int, str]] = []
        for name in sorted(selected_names):
            tensor = tensor_records[name]
            header_tensor = header.get(name)
            if not isinstance(header_tensor, Mapping):
                raise BenchmarkValidationError(
                    f"feature2 owner {slot} header lacks tensor {name!r}"
                )
            offsets = header_tensor.get("data_offsets")
            shape = header_tensor.get("shape")
            dtype = header_tensor.get("dtype")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(item, int) for item in offsets)
                or offsets[0] < 0
                or offsets[1] <= offsets[0]
                or offsets[1] > record.get("payload_bytes", -1)
                or not isinstance(shape, list)
                or not shape
                or not all(isinstance(item, int) and item > 0 for item in shape)
                or dtype not in ("BF16", "F32", "U8")
            ):
                raise BenchmarkValidationError(
                    f"feature2 owner {slot} tensor {name!r} header geometry drifted"
                )
            byte_count = offsets[1] - offsets[0]
            if byte_count != tensor.get("byte_count"):
                raise BenchmarkValidationError(
                    f"feature2 owner {slot} tensor {name!r} byte range drifted"
                )
            ranges.append((offsets[0], offsets[1], name))
            allowlist.append(
                Feature2TensorRead(
                    device_slot=slot,
                    filename=filename,
                    name=name,
                    offset=int(record["header_bytes"]) + offsets[0],
                    byte_count=byte_count,
                    dtype=dtype,
                    shape=tuple(shape),
                    sha256=str(tensor.get("sha256")),
                )
            )
        ordered_ranges = sorted(ranges)
        if any(
            left[1] > right[0]
            for left, right in zip(ordered_ranges, ordered_ranges[1:], strict=False)
        ):
            raise BenchmarkValidationError(
                f"feature2 owner {slot} selected tensor ranges overlap"
            )
        bytes_by_slot[slot] = sum(stop - start for start, stop, _ in ranges)
    if set(names_by_slot) != {0, 1} or names_by_slot[0] != names_by_slot[1]:
        raise BenchmarkValidationError("feature2 owner tensor name sets differ")
    if set(bytes_by_slot.values()) != {PP16_FEATURE2_SELECTED_WEIGHT_BYTES_PER_DEVICE}:
        raise BenchmarkValidationError(
            f"feature2 selected bytes drifted by owner: {bytes_by_slot}"
        )
    return tuple(
        sorted(allowlist, key=lambda item: (item.device_slot, item.offset, item.name))
    )


def validate_feature2_acquisition_reads(
    allowed: Sequence[Feature2TensorRead], observed: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Require loader receipts to equal every allowed tensor/range exactly once."""

    expected = [item.to_dict() for item in allowed]
    actual = [dict(item) for item in observed]
    key = lambda item: (item["device_slot"], item["offset"], item["name"])
    if sorted(actual, key=key) != sorted(expected, key=key):
        raise BenchmarkValidationError(
            "feature2 loader reads differ from the exact manifest-derived allowlist"
        )
    bytes_by_slot = {
        slot: sum(
            int(item["byte_count"])
            for item in actual
            if item["device_slot"] == slot
        )
        for slot in (0, 1)
    }
    if set(bytes_by_slot.values()) != {PP16_FEATURE2_SELECTED_WEIGHT_BYTES_PER_DEVICE}:
        raise BenchmarkValidationError("feature2 loader byte totals drifted")
    return {
        "bytes_by_slot": bytes_by_slot,
        "passed": True,
        "read_count": len(actual),
        "violations": [],
    }


def validate_feature2_acquisition_scope(
    roles: Sequence[str], *, context_length: int
) -> dict[str, Any]:
    """Refuse model work beyond the minimum layer-1 history discriminator."""

    role_set = frozenset(str(role) for role in roles)
    violations = []
    missing = sorted(_MINIMUM_ACQUISITION_ROLES - role_set)
    if missing:
        violations.append(f"missing minimum roles: {missing}")
    extras = sorted(role_set - _MINIMUM_ACQUISITION_ROLES)
    if extras:
        violations.append(f"unapproved acquisition roles: {extras}")
    forbidden = sorted(
        role
        for role in role_set
        if role.startswith(_FORBIDDEN_ACQUISITION_PREFIXES)
    )
    if forbidden:
        violations.append(f"post-discriminator model work is forbidden: {forbidden}")
    if context_length != PP16_FEATURE2_CONTEXT_LENGTH:
        violations.append(
            f"feature2 discriminator requires exactly {PP16_FEATURE2_CONTEXT_LENGTH} positions"
        )
    report = {
        "context_length": context_length,
        "dense1_excluded_bytes_per_device": (
            PP16_TWO_LAYER_STATE_BYTES_PER_DEVICE
            - PP16_FEATURE2_MINIMUM_STATE_BYTES_PER_DEVICE
        ),
        "maximum_state_bytes_per_device": PP16_FEATURE2_MINIMUM_STATE_BYTES_PER_DEVICE,
        "passed": not violations,
        "roles": sorted(role_set),
        "violations": violations,
    }
    if violations:
        raise BenchmarkValidationError(
            f"feature2 acquisition scope rejected: {violations}"
        )
    return report


def pp16_feature2_transport_pairs() -> tuple[tuple[int, int], ...]:
    """Return two slot-preserving 16-stage rings over logical devices 0..31."""

    return tuple(
        (stage * 2 + lane, ((stage + 1) % 16) * 2 + lane)
        for stage in range(16)
        for lane in range(2)
    )


def validate_feature2_transport_contract(
    config: TransportChainConfig, pairs: Sequence[Sequence[int]]
) -> dict[str, Any]:
    """Bind the existing transport primitive to two live feature lanes."""

    expected_pairs = pp16_feature2_transport_pairs()
    canonical = validate_transport_pairs(pairs, total_devices=32, stage_count=16)
    violations = []
    if config.plan is not PlanName.PP16_LP2:
        violations.append("feature2 transport requires PP16_LP2")
    if config.kind is not TransportKind.DEVICE_RESIDENT:
        violations.append("feature2 transport must be device resident")
    if (config.rows, config.width, config.dtype) != (2, 3072, "bfloat16"):
        violations.append("feature2 lane payload must pack two bf16[1,3072] rows")
    if canonical != expected_pairs:
        violations.append("feature2 transport pairs changed slot or stage ordering")
    report = {
        "lane_payload_shape": [2, 1, 3072],
        "passed": not violations,
        "stage_payload_shape": [2, 1, 3072],
        "transport_pair_count": len(canonical),
        "violations": violations,
    }
    if violations:
        raise BenchmarkValidationError(
            f"feature2 transport contract rejected: {violations}"
        )
    return report


def _stablehlo_ssa(
    stablehlo: str,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    instructions: dict[str, dict[str, Any]] = {}
    lines = stablehlo.splitlines()
    for index, raw in enumerate(lines):
        line = raw.strip()
        match = re.match(r"^(%[A-Za-z0-9_.]+)(?::[0-9]+)?\s*=\s*(.*)$", line)
        if match is None:
            continue
        name, rhs = match.groups()
        quoted = re.search(r'"(stablehlo\.[a-z_]+)"', rhs)
        plain = re.match(r"([a-z]+\.[a-z_]+)", rhs)
        opcode = quoted.group(1) if quoted else (plain.group(1) if plain else "")
        operand_text = rhs.split(" : ", 1)[0].split(" <{", 1)[0]
        operands = tuple(re.findall(r"%[A-Za-z0-9_.]+(?:#[0-9]+)?", operand_text))
        instructions[name] = {
            "index": index,
            "line": line,
            "opcode": opcode,
            "operands": operands,
        }
    return instructions, lines


def _ssa_values_for_opcode(
    instructions: Mapping[str, Mapping[str, Any]], opcode: str
) -> list[str]:
    return [
        name for name, item in instructions.items() if item.get("opcode") == opcode
    ]


def _single_ssa_consumer(
    instructions: Mapping[str, Mapping[str, Any]], value: str, opcode: str
) -> str | None:
    matches = [
        name
        for name, item in instructions.items()
        if item.get("opcode") == opcode and value in item.get("operands", ())
    ]
    return matches[0] if len(matches) == 1 else None


def _single_ssa_instruction(
    instructions: Mapping[str, Mapping[str, Any]],
    opcode: str,
    operands: tuple[str, ...],
) -> str | None:
    matches = [
        name
        for name, item in instructions.items()
        if item.get("opcode") == opcode and item.get("operands") == operands
    ]
    return matches[0] if len(matches) == 1 else None


def validate_feature2_transport_stablehlo(stablehlo: str) -> dict[str, Any]:
    """Require one live dependent 16-hop chain carrying both split rows."""

    lines = [
        line.strip()
        for line in stablehlo.splitlines()
        if '"stablehlo.collective_permute"' in line
    ]
    violations = []
    expected_pairs = pp16_feature2_transport_pairs()
    for line in lines:
        match = re.search(r"source_target_pairs\s*=\s*dense<(\[\[.*?\]\])>", line)
        pairs = ()
        if match is not None:
            pairs = tuple(
                (int(source), int(target))
                for source, target in re.findall(
                    r"\[\s*([0-9]+)\s*,\s*([0-9]+)\s*\]", match.group(1)
                )
            )
        if pairs != expected_pairs:
            violations.append("transport hop pairs changed slot/stage order")
        if "(tensor<2x1x3072xbf16>) -> tensor<2x1x3072xbf16>" not in line:
            violations.append("transport hop payload is not BF16 [2,1,3072]")
    if len(lines) != 16:
        violations.append(f"expected 16 transport hops, found {len(lines)}")
    for name in _COLLECTIVE_NAMES:
        if name == "collective_permute":
            continue
        if f'"stablehlo.{name}"' in stablehlo:
            violations.append(f"feature2 transport contains forbidden {name}")
    if any(marker in stablehlo for marker in _HOST_MARKERS):
        violations.append("feature2 transport contains a host callback marker")
    instructions, hlo_lines = _stablehlo_ssa(stablehlo)
    public_line = next(
        (
            line.strip()
            for line in stablehlo.splitlines()
            if "func.func public @main" in line
        ),
        "",
    )
    if public_line.count("tensor<32x2x1x3072xbf16>") < 2:
        violations.append(
            "global device-axis signature drifted from 32 packed BF16 feature states"
        )
    if "6144" in stablehlo:
        violations.append("transport contains a forbidden full-hidden intermediate")
    manual_values = _ssa_values_for_opcode(instructions, "sdy.manual_computation")
    if (
        len(manual_values) != 1
        or 'manual_axes={"device"}' not in instructions[manual_values[0]]["line"]
        or 'in_shardings=[<@mesh, [{"device"}, {}, {}, {}]>]'
        not in instructions[manual_values[0]]["line"]
        or 'out_shardings=[<@mesh, [{"device"}, {}, {}, {}]>]'
        not in instructions[manual_values[0]]["line"]
        or '(%arg1: tensor<1x2x1x3072xbf16>)' not in instructions[manual_values[0]]["line"]
        or 'sdy.mesh @mesh = <["device"=32]>' not in stablehlo
    ):
        violations.append("transport lacks the exact manual device-axis mapping")
    permute_values = _ssa_values_for_opcode(
        instructions, "stablehlo.collective_permute"
    )
    barrier_values = _ssa_values_for_opcode(
        instructions, "stablehlo.optimization_barrier"
    )
    if len(barrier_values) != 16:
        violations.append(
            f"expected 16 transport barriers, found {len(barrier_values)}"
        )
    if len(permute_values) == 16 and len(barrier_values) == 16:
        ordered_permutes = sorted(
            permute_values, key=lambda name: instructions[name]["index"]
        )
        ordered_barriers = sorted(
            barrier_values, key=lambda name: instructions[name]["index"]
        )
        input_reshape = _single_ssa_instruction(
            instructions, "stablehlo.reshape", ("%arg1",)
        )
        first_operands = instructions[ordered_permutes[0]]["operands"]
        first_source = first_operands[0] if len(first_operands) == 1 else None
        if (
            input_reshape is None
            or first_source is None
            or instructions.get(first_source, {}).get("opcode")
            != "stablehlo.add"
            or input_reshape
            not in instructions.get(first_source, {}).get("operands", ())
        ):
            violations.append("transport chain does not originate at the live input")
        for index, (permute, barrier) in enumerate(
            zip(ordered_permutes, ordered_barriers, strict=True)
        ):
            if instructions[barrier]["operands"] != (permute,):
                violations.append(f"transport barrier {index} bypasses its permute")
            if index and instructions[permute]["operands"] != (
                ordered_barriers[index - 1],
            ):
                violations.append(f"transport permute {index} is not chain-dependent")
        final_broadcast = _single_ssa_consumer(
            instructions, ordered_barriers[-1], "stablehlo.broadcast_in_dim"
        )
        return_line = next(
            (line.strip() for line in hlo_lines if line.strip().startswith("sdy.return ")),
            "",
        )
        if (
            final_broadcast is None
            or return_line
            != f"sdy.return {final_broadcast} : tensor<1x2x1x3072xbf16>"
        ):
            violations.append("transport chain does not reach the manual live root")
    if manual_values:
        manual_base = manual_values[0]
        if not any(
            line.strip()
            == f"return {manual_base} : tensor<32x2x1x3072xbf16>"
            for line in hlo_lines
        ):
            violations.append("transport manual result does not reach the public root")
    report = {
        "collective_permute_count": len(lines),
        "lane_count": 2,
        "passed": not violations,
        "stage_count": 16,
        "stage_payload_shape": [2, 1, 3072],
        "violations": violations,
    }
    if violations:
        raise HloContractViolationError(
            f"feature2 transport StableHLO rejected: {violations}"
        )
    return report


def lower_feature2_transport(*, devices: Sequence[Any] | None = None) -> LoweredFeature2Transport:
    """Lower, but do not execute, the forced-32 feature2 transport graph."""

    import jax
    from jax import lax
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    runtime_devices = tuple(jax.devices() if devices is None else devices)
    if len(runtime_devices) != 32:
        raise BenchmarkValidationError(
            f"feature2 transport requires exactly 32 devices, found {len(runtime_devices)}"
        )
    config = TransportChainConfig(
        plan=PlanName.PP16_LP2,
        kind=TransportKind.DEVICE_RESIDENT,
        rows=2,
        width=PP16_FEATURE2_SHARD_WIDTH,
        dtype="bfloat16",
        warmup_iterations=1,
        measured_iterations=1,
    )
    pairs = pp16_feature2_transport_pairs()
    validate_feature2_transport_contract(config, pairs)
    mesh = Mesh(np.asarray(runtime_devices, dtype=object), ("device",))
    sharding = NamedSharding(mesh, P("device", None, None, None))

    def mapped(initial: Any) -> Any:
        rank = lax.axis_index("device")
        state = initial[0] + (rank.astype(jnp.float32) + 1).astype(jnp.bfloat16)
        with jax.named_scope("greenfield_pp16_feature2_transport"):
            for _ in range(16):
                state = lax.ppermute(state, "device", pairs)
                state = lax.optimization_barrier(state)
        return state[None, ...]

    transport = jax.shard_map(
        mapped,
        mesh=mesh,
        in_specs=P("device", None, None, None),
        out_specs=P("device", None, None, None),
        check_vma=False,
    )
    host = np.zeros(
        (32, 2, 1, PP16_FEATURE2_SHARD_WIDTH), dtype=np.float32
    )
    value = jax.device_put(host.astype(jnp.bfloat16), sharding)
    lowered = jax.jit(transport).lower(value)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    return LoweredFeature2Transport(
        stablehlo=stablehlo,
        stablehlo_contract=validate_feature2_transport_stablehlo(stablehlo),
    )


def _stablehlo_operations(text: str) -> dict[str, list[str]]:
    operations = {name: [] for name in _COLLECTIVE_NAMES}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        for name in _COLLECTIVE_NAMES:
            marker = f'"stablehlo.{name}"'
            if marker not in line:
                continue
            block = [line]
            cursor = index + 1
            while cursor < len(lines) and " -> tensor<" not in " ".join(block):
                block.append(lines[cursor])
                cursor += 1
            operations[name].append("\n".join(block))
    return operations


def validate_feature2_boundary_stablehlo(stablehlo: str) -> dict[str, Any]:
    """Require one scalar FP32 reduction and one completed-BF16 LP2 gather."""

    operations = _stablehlo_operations(stablehlo)
    reductions = operations["all_reduce"]
    gathers = operations["all_gather"]
    violations = []
    if len(reductions) != 1:
        violations.append(f"expected one scalar reduction, found {len(reductions)}")
    if len(gathers) != 1:
        violations.append(f"expected one normalized gather, found {len(gathers)}")
    group_pattern = r"replica_groups\s*=\s*dense<\[\[0, 1\]\]>"
    if reductions and (
        re.search(group_pattern, reductions[0]) is None
        or re.search(r"\(tensor<f32>\)\s*->\s*tensor<f32>", reductions[0]) is None
    ):
        violations.append("RMS reduction is not one {0,1} FP32 scalar")
    if gathers and (
        re.search(group_pattern, gathers[0]) is None
        or "tensor<1x3072xbf16>" not in gathers[0]
        or "tensor<1x6144xbf16>" not in gathers[0]
    ):
        violations.append("normalized gather is not {0,1} BF16 [1,3072]->[1,6144]")
    unexpected = {
        name: len(blocks)
        for name, blocks in operations.items()
        if name not in ("all_reduce", "all_gather") and blocks
    }
    if unexpected:
        violations.append(f"unexpected collectives: {unexpected}")
    if any(marker in stablehlo for marker in _HOST_MARKERS):
        violations.append("feature2 boundary contains a host callback marker")
    instructions, hlo_lines = _stablehlo_ssa(stablehlo)
    public_line = next(
        (
            line.strip()
            for line in stablehlo.splitlines()
            if "func.func public @main" in line
        ),
        "",
    )
    if public_line.count("tensor<2x1x3072xbf16>") < 4:
        violations.append(
            "public boundary does not keep three inputs and carried state feature-sharded"
        )
    if "tensor<1x6144xbf16>" not in public_line:
        violations.append("public boundary lacks the single completed normalized row")
    forbidden_shapes = ("tensor<32x6144", "tensor<32x1x6144")
    if any(shape in stablehlo for shape in forbidden_shapes):
        violations.append("feature2 boundary contains a dead/full-pod hidden shape")
    manual_values = _ssa_values_for_opcode(instructions, "sdy.manual_computation")
    if (
        len(manual_values) != 1
        or 'manual_axes={"feature"}' not in instructions[manual_values[0]]["line"]
        or instructions[manual_values[0]]["line"].count(
            '<@mesh, [{"feature"}, {}, {}]>'
        )
        != 4
        or 'out_shardings=[<@mesh, [{"feature"}, {}, {}]>, <@mesh, [{}, {}]>]'
        not in instructions[manual_values[0]]["line"]
        or (
            "%arg3: tensor<1x1x3072xbf16>, "
            "%arg4: tensor<1x1x3072xbf16>, "
            "%arg5: tensor<1x1x3072xbf16>"
        )
        not in instructions[manual_values[0]]["line"]
        or 'sdy.mesh @mesh = <["feature"=2]>' not in stablehlo
    ):
        violations.append("boundary lacks the exact manual feature-axis mapping")

    dense_reshape = _single_ssa_instruction(
        instructions, "stablehlo.reshape", ("%arg3",)
    )
    residual_reshape = _single_ssa_instruction(
        instructions, "stablehlo.reshape", ("%arg4",)
    )
    weight_reshape = _single_ssa_instruction(
        instructions, "stablehlo.reshape", ("%arg5",)
    )
    dense_f32 = (
        None
        if dense_reshape is None
        else _single_ssa_instruction(
            instructions, "stablehlo.convert", (dense_reshape,)
        )
    )
    residual_f32 = (
        None
        if residual_reshape is None
        else _single_ssa_instruction(
            instructions, "stablehlo.convert", (residual_reshape,)
        )
    )
    total = (
        None
        if dense_f32 is None or residual_f32 is None
        else _single_ssa_instruction(
            instructions, "stablehlo.add", (dense_f32, residual_f32)
        )
    )
    square = (
        None
        if total is None
        else _single_ssa_instruction(instructions, "chlo.square", (total,))
    )
    local_reduce = None
    if square is not None:
        candidates = [
            name
            for name, item in instructions.items()
            if item.get("opcode") == "stablehlo.reduce"
            and item.get("operands", ())[:1] == (square,)
        ]
        local_reduce = candidates[0] if len(candidates) == 1 else None
    local_reduce_ok = False
    if local_reduce is not None:
        local_item = instructions[local_reduce]
        local_operands = local_item.get("operands", ())
        initializer = local_operands[1] if len(local_operands) == 2 else None
        initializer_line = instructions.get(initializer or "", {}).get(
            "line", ""
        )
        local_reduce_ok = bool(
            local_operands[:1] == (square,)
            and "stablehlo.constant dense<0.000000e+00> : tensor<f32>"
            in initializer_line
            and "applies stablehlo.add across dimensions = [0, 1]"
            in local_item["line"]
            and (
                "(tensor<1x3072xf32>, tensor<f32>) -> tensor<f32>"
                in local_item["line"]
            )
        )
    all_reduce_values = _ssa_values_for_opcode(instructions, "stablehlo.all_reduce")
    all_reduce = all_reduce_values[0] if len(all_reduce_values) == 1 else None
    if (
        None in (
            dense_reshape,
            residual_reshape,
            weight_reshape,
            dense_f32,
            residual_f32,
            total,
            square,
            local_reduce,
            all_reduce,
        )
        or instructions.get(all_reduce or "", {}).get("operands")
        != (local_reduce,)
        or not local_reduce_ok
    ):
        violations.append("boundary RMS source-to-scalar lineage drifted")

    reducer_ok = bool(
        reductions
        and re.search(
            r"\^bb0\((%arg[0-9]+): tensor<f32>, (%arg[0-9]+): tensor<f32>\):"
            r".*?(%[A-Za-z0-9_.]+) = stablehlo\.add \1, \2 : tensor<f32>"
            r".*?stablehlo\.return \3 : tensor<f32>",
            reductions[0],
            flags=re.DOTALL,
        )
        is not None
    )
    if not reducer_ok:
        violations.append("RMS all-reduce does not use the exact additive FP32 reducer")

    divide = (
        None
        if all_reduce is None
        else _single_ssa_consumer(instructions, all_reduce, "stablehlo.divide")
    )
    divisor = None
    if divide is not None:
        divisor_operands = instructions[divide]["operands"]
        if len(divisor_operands) == 2 and divisor_operands[0] == all_reduce:
            divisor = divisor_operands[1]
    epsilon_add = (
        None
        if divide is None
        else _single_ssa_consumer(instructions, divide, "stablehlo.add")
    )
    epsilon = None
    if epsilon_add is not None:
        epsilon_operands = instructions[epsilon_add]["operands"]
        if len(epsilon_operands) == 2 and epsilon_operands[0] == divide:
            epsilon = epsilon_operands[1]
    rsqrt = (
        None
        if epsilon_add is None
        else _single_ssa_instruction(
            instructions, "stablehlo.rsqrt", (epsilon_add,)
        )
    )
    inverse_broadcast = (
        None
        if rsqrt is None
        else _single_ssa_instruction(
            instructions, "stablehlo.broadcast_in_dim", (rsqrt,)
        )
    )
    normalized_f32 = (
        None
        if total is None or inverse_broadcast is None
        else _single_ssa_instruction(
            instructions, "stablehlo.multiply", (total, inverse_broadcast)
        )
    )
    normalized_bf16 = (
        None
        if normalized_f32 is None
        else _single_ssa_instruction(
            instructions, "stablehlo.convert", (normalized_f32,)
        )
    )
    weighted = (
        None
        if normalized_bf16 is None or weight_reshape is None
        else _single_ssa_instruction(
            instructions,
            "stablehlo.multiply",
            (normalized_bf16, weight_reshape),
        )
    )
    gather_values = _ssa_values_for_opcode(instructions, "stablehlo.all_gather")
    gather = gather_values[0] if len(gather_values) == 1 else None
    carried = (
        None
        if total is None
        else _single_ssa_instruction(instructions, "stablehlo.convert", (total,))
    )
    carried_broadcast = (
        None
        if carried is None
        else _single_ssa_instruction(
            instructions, "stablehlo.broadcast_in_dim", (carried,)
        )
    )
    scalar_constants_ok = bool(
        divisor is not None
        and "dense<6.144000e+03>" in instructions.get(divisor, {}).get("line", "")
        and epsilon is not None
        and "dense<9.99999974E-6>" in instructions.get(epsilon, {}).get("line", "")
    )
    if (
        None
        in (
            divide,
            epsilon_add,
            rsqrt,
            inverse_broadcast,
            normalized_f32,
            normalized_bf16,
            weighted,
            gather,
            carried,
            carried_broadcast,
        )
        or not scalar_constants_ok
        or instructions.get(gather or "", {}).get("operands") != (weighted,)
    ):
        violations.append("boundary scalar-to-weighted-gather lineage drifted")
    if total is not None:
        total_consumers = {
            name
            for name, item in instructions.items()
            if total in item.get("operands", ())
        }
        if total_consumers != {square, normalized_f32, carried}:
            violations.append("boundary total has missing or extra consumers")
    if all_reduce is not None:
        all_reduce_consumers = {
            name
            for name, item in instructions.items()
            if all_reduce in item.get("operands", ())
        }
        if all_reduce_consumers != {divide}:
            violations.append("boundary scalar reduction has missing or extra consumers")
    manual_return = next(
        (line.strip() for line in hlo_lines if line.strip().startswith("sdy.return ")),
        "",
    )
    if (
        carried_broadcast is None
        or gather is None
        or manual_return
        != (
            f"sdy.return {carried_broadcast}, {gather} : "
            "tensor<1x1x3072xbf16>, tensor<1x6144xbf16>"
        )
    ):
        violations.append("boundary carried/gather values do not reach the manual root")
    if manual_values:
        manual_base = manual_values[0]
        if not any(
            line.strip()
            == (
                f"return {manual_base}#0, {manual_base}#1 : "
                "tensor<2x1x3072xbf16>, tensor<1x6144xbf16>"
            )
            for line in hlo_lines
        ):
            violations.append("boundary manual results do not reach the public root")
    report = {
        "all_gather_count": len(gathers),
        "all_reduce_count": len(reductions),
        "passed": not violations,
        "public_signature": public_line,
        "violations": violations,
    }
    if violations:
        raise HloContractViolationError(
            f"feature2 boundary StableHLO rejected: {violations}"
        )
    return report


def feature2_embedding_mapped(
    token_id: Any,
    embedding: Any,
    *,
    axis_name: str,
    pairs: Sequence[tuple[int, int]],
) -> Any:
    """Select one token directly into the owner's persistent feature half."""

    import jax
    import jax.numpy as jnp
    from jax import lax

    if token_id.shape != () or not jnp.issubdtype(token_id.dtype, jnp.integer):
        raise ValueError("feature2 token id must be one integer scalar")
    if embedding.ndim != 2 or embedding.shape[1] != PP16_FEATURE2_HIDDEN_WIDTH:
        raise ValueError("feature2 embedding owner geometry drifted")
    if embedding.dtype != jnp.bfloat16 or embedding.shape[0] <= 0:
        raise ValueError("feature2 embedding owner must be nonempty BF16")
    canonical_pairs = tuple((int(source), int(target)) for source, target in pairs)
    if set(canonical_pairs) != {(0, 1), (1, 0)} or len(canonical_pairs) != 2:
        raise ValueError("feature2 embedding requires exact 0<->1 pairs")

    local_slot = lax.axis_index(axis_name)
    local_vocab = jnp.int32(embedding.shape[0])
    global_vocab = local_vocab * jnp.int32(2)
    local_start = local_slot.astype(jnp.int32) * local_vocab
    valid = (token_id >= jnp.int32(0)) & (token_id < global_vocab)
    owns = valid & (token_id >= local_start) & (
        token_id < local_start + local_vocab
    )
    local_id = jnp.clip(
        token_id - local_start,
        jnp.int32(0),
        local_vocab - jnp.int32(1),
    )
    half0 = lax.dynamic_slice(
        embedding,
        (local_id, jnp.int32(0)),
        (1, PP16_FEATURE2_SHARD_WIDTH),
    )
    half1 = lax.dynamic_slice(
        embedding,
        (local_id, jnp.int32(PP16_FEATURE2_SHARD_WIDTH)),
        (1, PP16_FEATURE2_SHARD_WIDTH),
    )
    half0 = jnp.where(owns, half0, jnp.zeros_like(half0))
    half1 = jnp.where(owns, half1, jnp.zeros_like(half1))
    owns_half0 = local_slot == jnp.int32(0)
    owned = jnp.where(owns_half0, half0, half1)
    peer = jnp.where(owns_half0, half1, half0)
    with jax.named_scope("greenfield_pp16_feature2_embedding_exchange"):
        received = lax.ppermute(
            peer,
            axis_name=axis_name,
            perm=canonical_pairs,
        )
    return (owned + received).astype(jnp.bfloat16)


def feature2_embedding_batch_mapped(
    token_ids: Any,
    embedding: Any,
    *,
    axis_name: str,
    pairs: Sequence[tuple[int, int]],
) -> Any:
    """Select a token chunk into persistent halves with one LP2 exchange."""

    import jax
    import jax.numpy as jnp
    from jax import lax

    if (
        token_ids.ndim != 1
        or token_ids.shape[0] <= 0
        or not jnp.issubdtype(token_ids.dtype, jnp.integer)
    ):
        raise ValueError("feature2 token batch must be one nonempty integer vector")
    if embedding.ndim != 2 or embedding.shape[1] != PP16_FEATURE2_HIDDEN_WIDTH:
        raise ValueError("feature2 embedding owner geometry drifted")
    if embedding.dtype != jnp.bfloat16 or embedding.shape[0] <= 0:
        raise ValueError("feature2 embedding owner must be nonempty BF16")
    canonical_pairs = tuple((int(source), int(target)) for source, target in pairs)
    if set(canonical_pairs) != {(0, 1), (1, 0)} or len(canonical_pairs) != 2:
        raise ValueError("feature2 embedding requires exact 0<->1 pairs")

    local_slot = lax.axis_index(axis_name)
    local_vocab = jnp.int32(embedding.shape[0])
    global_vocab = local_vocab * jnp.int32(2)
    local_start = local_slot.astype(jnp.int32) * local_vocab
    valid = (token_ids >= jnp.int32(0)) & (token_ids < global_vocab)
    owns = valid & (token_ids >= local_start) & (token_ids < local_start + local_vocab)
    local_ids = jnp.clip(
        token_ids - local_start,
        jnp.int32(0),
        local_vocab - jnp.int32(1),
    )
    rows = jnp.take(embedding, local_ids, axis=0)
    half0 = jnp.where(
        owns[:, None],
        rows[:, :PP16_FEATURE2_SHARD_WIDTH],
        jnp.zeros((token_ids.shape[0], PP16_FEATURE2_SHARD_WIDTH), jnp.bfloat16),
    )
    half1 = jnp.where(
        owns[:, None],
        rows[:, PP16_FEATURE2_SHARD_WIDTH:],
        jnp.zeros((token_ids.shape[0], PP16_FEATURE2_SHARD_WIDTH), jnp.bfloat16),
    )
    owns_half0 = local_slot == jnp.int32(0)
    owned = jnp.where(owns_half0, half0, half1)
    peer = jnp.where(owns_half0, half1, half0)
    with jax.named_scope("greenfield_pp16_feature2_embedding_batch_exchange"):
        received = lax.ppermute(peer, axis_name=axis_name, perm=canonical_pairs)
    return (owned + received).astype(jnp.bfloat16)


def feature2_add_rms_gather_mapped(
    update: Any,
    residual: Any,
    norm_weight: Any,
    *,
    axis_name: str,
    groups: Sequence[Sequence[int]],
) -> tuple[Any, Any]:
    """Add two feature halves, normalize locally, and gather only the result."""

    import jax
    import jax.numpy as jnp
    from jax import lax

    expected = (1, PP16_FEATURE2_SHARD_WIDTH)
    if update.shape != expected or residual.shape != expected:
        raise ValueError("feature2 RMS inputs must be one 3072-feature row")
    if norm_weight.shape != expected:
        raise ValueError("feature2 RMS weight must be its local feature half")
    if any(
        value.dtype != jnp.bfloat16
        for value in (update, residual, norm_weight)
    ):
        raise ValueError("feature2 RMS inputs and weight must remain BF16")
    canonical_groups = tuple(tuple(int(rank) for rank in group) for group in groups)
    if canonical_groups != ((0, 1),):
        raise ValueError("bounded feature2 RMS requires exact {0,1} group")
    with jax.named_scope("greenfield_pp16_feature2_rms"):
        total = update.astype(jnp.float32) + residual.astype(jnp.float32)
        local_square_sum = jnp.sum(lax.square(total))
        square_sum = lax.psum(
            local_square_sum,
            axis_name,
            axis_index_groups=canonical_groups,
        )
        inverse = lax.rsqrt(
            square_sum / jnp.float32(PP16_FEATURE2_HIDDEN_WIDTH)
            + jnp.float32(1e-5)
        )
        carried = total.astype(jnp.bfloat16)
        normalized_local = (
            (total * inverse).astype(jnp.bfloat16) * norm_weight
        ).astype(jnp.bfloat16)
    with jax.named_scope("greenfield_pp16_feature2_normalized_gather"):
        normalized = lax.all_gather(
            normalized_local,
            axis_name,
            axis_index_groups=canonical_groups,
            axis=1,
            tiled=True,
        )
    return carried, normalized


def feature2_add_rms_gather_batch_mapped(
    update: Any,
    residual: Any,
    norm_weight: Any,
    *,
    axis_name: str,
    groups: Sequence[Sequence[int]],
) -> tuple[Any, Any]:
    """Normalize a token chunk row-wise with one LP2 scalar-vector reduction."""

    import jax
    import jax.numpy as jnp
    from jax import lax

    if (
        update.ndim != 2
        or update.shape[0] <= 0
        or update.shape[1] != PP16_FEATURE2_SHARD_WIDTH
    ):
        raise ValueError("feature2 RMS batch must contain nonempty 3072-feature rows")
    if residual.shape != update.shape:
        raise ValueError("feature2 RMS batch residual geometry drifted")
    if norm_weight.shape != (1, PP16_FEATURE2_SHARD_WIDTH):
        raise ValueError("feature2 RMS weight must be its local feature half")
    if any(value.dtype != jnp.bfloat16 for value in (update, residual, norm_weight)):
        raise ValueError("feature2 RMS inputs and weight must remain BF16")
    canonical_groups = tuple(tuple(int(rank) for rank in group) for group in groups)
    if canonical_groups != ((0, 1),):
        raise ValueError("bounded feature2 RMS requires exact {0,1} group")
    with jax.named_scope("greenfield_pp16_feature2_rms_batch"):
        total = update.astype(jnp.float32) + residual.astype(jnp.float32)
        local_square_sum = jnp.sum(lax.square(total), axis=1)
        square_sum = lax.psum(
            local_square_sum,
            axis_name,
            axis_index_groups=canonical_groups,
        )
        inverse = lax.rsqrt(
            square_sum / jnp.float32(PP16_FEATURE2_HIDDEN_WIDTH) + jnp.float32(1e-5)
        )
        carried = total.astype(jnp.bfloat16)
        normalized_local = (
            (total * inverse[:, None]).astype(jnp.bfloat16) * norm_weight
        ).astype(jnp.bfloat16)
    with jax.named_scope("greenfield_pp16_feature2_normalized_batch_gather"):
        normalized = lax.all_gather(
            normalized_local,
            axis_name,
            axis_index_groups=canonical_groups,
            axis=1,
            tiled=True,
        )
    return carried, normalized


def build_feature2_boundary(*, devices: Sequence[Any] | None = None) -> CompiledFeature2Boundary:
    """Compile the feature2 RMS/gather scaffold on exactly two devices."""

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    runtime_devices = tuple(jax.devices() if devices is None else devices)
    if len(runtime_devices) != 2:
        raise BenchmarkValidationError(
            f"feature2 boundary requires exactly two devices, found {len(runtime_devices)}"
        )
    mesh = Mesh(np.asarray(runtime_devices, dtype=object), ("feature",))
    sharding = NamedSharding(mesh, P("feature", None, None))

    def mapped(
        dense_update: Any, carried_residual: Any, norm_weight: Any
    ) -> tuple[Any, Any]:
        carried, normalized = feature2_add_rms_gather_mapped(
            dense_update[0],
            carried_residual[0],
            norm_weight[0],
            axis_name="feature",
            groups=((0, 1),),
        )
        return carried[None, ...], normalized

    mapped_boundary = jax.shard_map(
        mapped,
        mesh=mesh,
        in_specs=(
            P("feature", None, None),
            P("feature", None, None),
            P("feature", None, None),
        ),
        out_specs=(P("feature", None, None), P(None, None)),
        check_vma=False,
    )
    host = np.linspace(
        -0.5,
        0.5,
        num=2 * PP16_FEATURE2_SHARD_WIDTH,
        dtype=np.float32,
    ).reshape(2, 1, PP16_FEATURE2_SHARD_WIDTH)
    dense_update = jax.device_put(host.astype(jnp.bfloat16), sharding)
    carried_residual = jax.device_put(host[::-1].copy().astype(jnp.bfloat16), sharding)
    norm_weight = jax.device_put(np.ones_like(host).astype(jnp.bfloat16), sharding)
    lowered = jax.jit(mapped_boundary).lower(
        dense_update, carried_residual, norm_weight
    )
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    contract = validate_feature2_boundary_stablehlo(stablehlo)
    compiled = lowered.compile()
    return CompiledFeature2Boundary(
        compiled=compiled,
        dense_update=dense_update,
        carried_residual=carried_residual,
        norm_weight=norm_weight,
        stablehlo=stablehlo,
        optimized_hlo=compiled.as_text(),
        stablehlo_contract=contract,
    )


def feature2_minimum_acquisition_roles() -> tuple[str, ...]:
    return tuple(sorted(_MINIMUM_ACQUISITION_ROLES))
