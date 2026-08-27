"""Fail-closed direct loader for a complete final-layout checkpoint.

The loader never reconstructs the source checkpoint and never dequantizes
FP8 weights on the host.  It verifies the self-contained packed artifact
before importing JAX, then reads one final-owner safetensors file at a time
and places each leaf directly on its physical owner.  FP8 is kept resident as
its exact E4M3FN byte encoding; colocated FP32 inverse scales remain separate.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
from hashlib import sha256
import json
from math import prod
from pathlib import Path
import resource
from typing import Any, Mapping, Sequence

from ..errors import CheckpointValidationError
from ..partitioning import BASE_LOAD_SET, inspect_layout_manifest
from .one_layer_loader import StageDeviceResolution
from .stream_pack import (
    DestinationFilePlan,
    DestinationTensorPlan,
    build_destination_file_plans,
)


PACKED_ARTIFACT_KIND = "greenfield_full_packed_checkpoint"
PACK_CONTROL_KIND = "greenfield_streaming_full_checkpoint_pack"
SUPPORTED_DTYPES = frozenset(("BF16", "F32", "F8_E4M3"))


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _mapping_hash(value: Mapping[str, Any], *, hash_field: str) -> str:
    unhashed = dict(value)
    observed = unhashed.pop(hash_field, None)
    computed = sha256(_canonical_json(unhashed).encode("utf-8")).hexdigest()
    if observed != computed:
        raise CheckpointValidationError(
            f"{hash_field} mismatch: observed={observed!r} computed={computed}"
        )
    return computed


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointValidationError(f"cannot parse JSON evidence {path}") from error
    if not isinstance(value, dict):
        raise CheckpointValidationError(f"JSON evidence is not an object: {path}")
    return value


def _digest(value: str, *, field: str, lengths: tuple[int, ...] = (64,)) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase digest")
    return value


@dataclass(frozen=True, slots=True)
class FullCheckpointLoadExpectation:
    """Every immutable identity required before a payload may be read."""

    packed_manifest_sha256: str
    layout_manifest_sha256: str
    source_inventory_sha256: str
    source_revision: str
    topology_hash: str
    plan_group_hash: str
    plan_manifest_sha256: str
    execution_plan_sha256: str
    layout_code_hash: str
    pack_code_hash: str
    destination: str
    plan_id: str = "PP8_LP4"
    model_id: str = "zai-org/GLM-5.2-FP8"

    def __post_init__(self) -> None:
        for field in (
            "packed_manifest_sha256",
            "layout_manifest_sha256",
            "source_inventory_sha256",
            "topology_hash",
            "plan_group_hash",
            "plan_manifest_sha256",
            "execution_plan_sha256",
        ):
            _digest(getattr(self, field), field=field)
        for field in ("layout_code_hash", "pack_code_hash"):
            _digest(getattr(self, field), field=field, lengths=(40, 64))
        if not self.source_revision.strip():
            raise ValueError("source_revision must be non-empty")
        if self.plan_id not in ("PP8_LP4", "PP16_LP2"):
            raise ValueError("complete loader supports PP8_LP4 and PP16_LP2")
        if self.model_id != "zai-org/GLM-5.2-FP8":
            raise ValueError("complete loader supports only GLM-5.2-FP8")
        if not self.destination.startswith(
            "gs://driftbench-dsv4-uc/checkpoints/greenfield/"
        ):
            raise ValueError("destination must be a greenfield approved-bucket prefix")

    @property
    def stage_size(self) -> int:
        return {"PP8_LP4": 4, "PP16_LP2": 2}[self.plan_id]


@dataclass(frozen=True, slots=True)
class VerifiedPackedCheckpoint:
    """Fully reconciled metadata and exact expected final files."""

    root: Path
    layout: Mapping[str, Any]
    packed_manifest: Mapping[str, Any]
    control: Mapping[str, Any]
    plans: tuple[DestinationFilePlan, ...]
    evidence_by_filename: Mapping[str, Mapping[str, Any]]


def verify_full_packed_checkpoint(
    root: Path,
    expectation: FullCheckpointLoadExpectation,
    *,
    require_payloads: bool = True,
) -> VerifiedPackedCheckpoint:
    """Verify metadata and, by default, every sidecar and payload size.

    ``require_payloads=False`` is only a lineage mode for a final derivative
    whose own complete payload is verified separately.  It never authorizes a
    load from this parent checkpoint.
    """

    root = Path(root)
    required = {
        "control": root / "control.json",
        "layout": root / "layout_manifest.json",
        "packed": root / "packed_manifest.json",
        "success": root / "SUCCESS",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise CheckpointValidationError(
            f"packed checkpoint is incomplete; missing {sorted(missing)}"
        )
    layout = inspect_layout_manifest(required["layout"])
    control = _read_json(required["control"])
    packed = _read_json(required["packed"])
    control_hash = _mapping_hash(control, hash_field="control_sha256")
    packed_hash = _mapping_hash(packed, hash_field="manifest_sha256")
    plan_manifest = layout.get("plan_manifest")
    if not isinstance(plan_manifest, Mapping):
        raise CheckpointValidationError("layout plan manifest is missing")
    execution_plan = plan_manifest.get("execution_plan")
    if not isinstance(execution_plan, Mapping):
        raise CheckpointValidationError("layout execution plan is missing")
    observed = {
        "destination": packed.get("destination"),
        "execution_plan_sha256": plan_manifest.get("execution_plan_sha256"),
        "layout_code_hash": layout.get("code_hash"),
        "layout_manifest_sha256": layout.get("manifest_sha256"),
        "model_id": layout.get("source", {}).get("model_id"),
        "pack_code_hash": packed.get("code_hash"),
        "packed_manifest_sha256": packed_hash,
        "plan_group_hash": layout.get("plan_group_hash"),
        "plan_id": packed.get("plan_id"),
        "plan_manifest_sha256": layout.get("plan_manifest_sha256"),
        "source_inventory_sha256": packed.get("source_inventory_sha256"),
        "source_revision": packed.get("source_revision"),
        "topology_hash": layout.get("topology_hash"),
    }
    expected = {
        field: getattr(expectation, field)
        for field in observed
    }
    mismatches = {
        field: {"expected": expected[field], "observed": observed[field]}
        for field in observed
        if observed[field] != expected[field]
    }
    if mismatches:
        raise CheckpointValidationError(
            f"complete checkpoint identity mismatch: {mismatches}"
        )
    if control.get("artifact_kind") != PACK_CONTROL_KIND:
        raise CheckpointValidationError("wrong checkpoint pack control kind")
    if packed.get("artifact_kind") != PACKED_ARTIFACT_KIND:
        raise CheckpointValidationError("wrong packed checkpoint artifact kind")
    if packed.get("control_sha256") != control_hash:
        raise CheckpointValidationError("packed manifest does not bind control")
    if control.get("layout_file_sha256") != _sha256_file(required["layout"]):
        raise CheckpointValidationError("packed layout file SHA-256 mismatch")
    common = (
        "destination",
        "layout_manifest_sha256",
        "plan_id",
        "source_inventory_sha256",
        "source_revision",
    )
    for field in common:
        if control.get(field) != packed.get(field):
            raise CheckpointValidationError(
                f"control and packed manifest disagree on {field}"
            )
    if control.get("code_hash") != expectation.pack_code_hash:
        raise CheckpointValidationError("control pack code hash drifted")
    if packed.get("layout_manifest_sha256") != layout.get("manifest_sha256"):
        raise CheckpointValidationError("packed and layout manifests disagree")
    if packed.get("source_payload_bytes") != layout.get("source", {}).get(
        "payload_bytes"
    ):
        raise CheckpointValidationError("source payload total drifted")
    if packed.get("packed_payload_bytes") != layout.get("packed_payload_bytes"):
        raise CheckpointValidationError("packed payload total drifted")
    plans = build_destination_file_plans(layout)
    file_records = packed.get("files")
    if not isinstance(file_records, list):
        raise CheckpointValidationError("packed manifest lacks its file ledger")
    by_filename: dict[str, Mapping[str, Any]] = {}
    for record in file_records:
        if not isinstance(record, Mapping):
            raise CheckpointValidationError("packed file record is not an object")
        filename = record.get("destination_filename")
        if not isinstance(filename, str) or filename in by_filename:
            raise CheckpointValidationError("packed filenames are invalid or duplicate")
        by_filename[filename] = record
    if set(by_filename) != {plan.filename for plan in plans}:
        raise CheckpointValidationError("packed file ledger is incomplete")
    total_file_bytes = 0
    total_payload_bytes = 0
    for plan in plans:
        record = by_filename[plan.filename]
        expected_record = {
            "destination_filename": plan.filename,
            "device_id": plan.device_id,
            "device_slot": plan.device_slot,
            "file_bytes": plan.file_bytes,
            "header_bytes": len(plan.header),
            "header_sha256": sha256(plan.header).hexdigest(),
            "layout_manifest_sha256": expectation.layout_manifest_sha256,
            "load_set": plan.load_set,
            "pack_code_hash": expectation.pack_code_hash,
            "payload_bytes": plan.payload_bytes,
            "stage_id": plan.stage_id,
        }
        for field, value in expected_record.items():
            if record.get(field) != value:
                raise CheckpointValidationError(
                    f"packed file {plan.filename!r} drifted field {field}"
                )
        for field in ("crc32c", "generation", "sha256"):
            if field not in record:
                raise CheckpointValidationError(
                    f"packed file {plan.filename!r} lacks {field}"
                )
        _digest(record["sha256"], field=f"{plan.filename}.sha256")
        if not isinstance(record["generation"], int) or record["generation"] <= 0:
            raise CheckpointValidationError(
                f"packed file {plan.filename!r} has invalid generation"
            )
        if not isinstance(record["crc32c"], str) or not record["crc32c"]:
            raise CheckpointValidationError(
                f"packed file {plan.filename!r} has invalid CRC32C"
            )
        if require_payloads:
            sidecar = _read_json(root / "evidence" / f"{plan.filename}.json")
            if sidecar != dict(record):
                raise CheckpointValidationError(
                    f"packed file {plan.filename!r} sidecar disagrees"
                )
            payload_path = root / plan.filename
            try:
                observed_size = payload_path.stat().st_size
            except OSError as error:
                raise CheckpointValidationError(
                    f"packed payload is missing: {plan.filename}"
                ) from error
            if observed_size != plan.file_bytes:
                raise CheckpointValidationError(
                    f"packed payload size drift for {plan.filename!r}: "
                    f"expected={plan.file_bytes} observed={observed_size}"
                )
        total_file_bytes += plan.file_bytes
        total_payload_bytes += plan.payload_bytes
    if (
        packed.get("file_count") != len(plans)
        or control.get("destination_file_count") != len(plans)
        or packed.get("packed_file_bytes") != total_file_bytes
        or control.get("expected_file_bytes") != total_file_bytes
        or packed.get("packed_payload_bytes") != total_payload_bytes
        or control.get("packed_payload_bytes") != total_payload_bytes
    ):
        raise CheckpointValidationError("packed file/byte totals do not reconcile")
    expected_success = f"{packed_hash}  packed_manifest.json\n"
    if required["success"].read_text() != expected_success:
        raise CheckpointValidationError("packed checkpoint SUCCESS marker drifted")
    return VerifiedPackedCheckpoint(
        root=root,
        layout=layout,
        packed_manifest=packed,
        control=control,
        plans=plans,
        evidence_by_filename=by_filename,
    )


@dataclass(frozen=True, slots=True)
class LoadedFinalLeaf:
    """One exact local owner leaf and its logical checkpoint contract."""

    array: Any
    logical_dtype: str
    storage_dtype: str
    shape: tuple[int, ...]
    byte_count: int
    sha256: str


@dataclass(slots=True)
class LoadedFinalLayoutStage:
    """A complete local stage; readiness exists only after all checks pass."""

    stage_id: int
    device_ids: tuple[int, ...]
    leaves_by_slot: tuple[dict[str, LoadedFinalLeaf], ...]
    state_manifest: Mapping[str, Any]
    load_record: Mapping[str, Any]
    closed: bool = False

    def close(self) -> None:
        """Release every owned device buffer without touching artifact evidence."""

        if self.closed:
            return
        for leaves in self.leaves_by_slot:
            for leaf in leaves.values():
                try:
                    leaf.array.delete()
                except (AttributeError, RuntimeError):
                    pass
            leaves.clear()
        self.closed = True
        gc.collect()


def _rss_peak_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _memory_stats(device: object) -> dict[str, int] | None:
    value = device.memory_stats()
    if value is None:
        return None
    return {
        str(name): int(number)
        for name, number in value.items()
        if isinstance(number, int) and not isinstance(number, bool)
    }


def _read_exact_into(
    stream: Any,
    byte_count: int,
    *,
    file_digest: Any,
    chunk_bytes: int,
) -> bytearray:
    value = bytearray(byte_count)
    view = memoryview(value)
    offset = 0
    while offset < byte_count:
        end = min(byte_count, offset + chunk_bytes)
        observed = stream.readinto(view[offset:end])
        if not observed:
            raise CheckpointValidationError(
                f"packed tensor truncated with {byte_count - offset} bytes remaining"
            )
        file_digest.update(view[offset : offset + observed])
        offset += observed
    return value


def _validate_finite(raw: bytearray, dtype: str, *, chunk_bytes: int) -> None:
    import numpy as np

    view = memoryview(raw)
    if dtype == "F8_E4M3":
        item_bytes = 1
        mask = 0x7F
        forbidden = 0x7F
        integer_dtype = np.uint8
    elif dtype == "BF16":
        item_bytes = 2
        mask = 0x7F80
        forbidden = 0x7F80
        integer_dtype = np.dtype("<u2")
    elif dtype == "F32":
        item_bytes = 4
        mask = 0x7F800000
        forbidden = 0x7F800000
        integer_dtype = np.dtype("<u4")
    else:
        raise CheckpointValidationError(f"unsupported loaded dtype {dtype!r}")
    aligned_chunk = max(item_bytes, chunk_bytes - chunk_bytes % item_bytes)
    for start in range(0, len(raw), aligned_chunk):
        part = view[start : min(len(raw), start + aligned_chunk)]
        values = np.frombuffer(part, dtype=integer_dtype)
        if bool(np.any((values & mask) == forbidden)):
            raise CheckpointValidationError(
                f"packed {dtype} tensor contains a non-finite encoding"
            )


def _host_array(raw: bytearray, tensor: DestinationTensorPlan) -> tuple[Any, str]:
    import ml_dtypes
    import numpy as np

    if tensor.dtype == "F8_E4M3":
        dtype = np.dtype("u1")
        storage_dtype = "U8_E4M3FN_BITS"
    elif tensor.dtype == "BF16":
        dtype = np.dtype(ml_dtypes.bfloat16)
        storage_dtype = "BF16"
    elif tensor.dtype == "F32":
        dtype = np.dtype("<f4")
        storage_dtype = "F32"
    else:
        raise CheckpointValidationError(
            f"unsupported final-layout dtype {tensor.dtype!r}"
        )
    expected = prod(tensor.shape) * dtype.itemsize
    if expected != tensor.byte_count or len(raw) != expected:
        raise CheckpointValidationError(
            f"tensor {tensor.name!r} shape/dtype bytes do not reconcile"
        )
    return np.frombuffer(raw, dtype=dtype).reshape(tensor.shape), storage_dtype


def _array_bytes_sha256(value: Any) -> str:
    import numpy as np

    contiguous = np.ascontiguousarray(value)
    byte_view = contiguous.view(np.uint8).reshape(-1)
    return sha256(memoryview(byte_view)).hexdigest()


def _delete_loaded(leaves_by_slot: Sequence[Mapping[str, LoadedFinalLeaf]]) -> None:
    for leaves in leaves_by_slot:
        for leaf in leaves.values():
            try:
                leaf.array.delete()
            except (AttributeError, RuntimeError):
                pass
    gc.collect()


def load_final_layout_stage(
    checkpoint: VerifiedPackedCheckpoint,
    expectation: FullCheckpointLoadExpectation,
    resolution: StageDeviceResolution,
    *,
    load_set: str = BASE_LOAD_SET,
    verify_device_roundtrip: bool = True,
    chunk_bytes: int = 64 * 1024 * 1024,
) -> LoadedFinalLayoutStage:
    """Load one complete physical stage directly and keep raw FP8 resident."""

    if chunk_bytes <= 0:
        raise ValueError("loader chunk_bytes must be positive")
    if load_set != BASE_LOAD_SET:
        raise ValueError("base loader refuses optional MTP unless explicitly implemented")
    if len(resolution.devices) != expectation.stage_size:
        raise CheckpointValidationError("resolved stage has the wrong local size")
    if len(set(resolution.captured_device_ids)) != expectation.stage_size:
        raise CheckpointValidationError("resolved stage device ids are duplicate")
    plan_manifest = checkpoint.layout["plan_manifest"]
    assignments = plan_manifest["execution_plan"]["stage_assignments"]
    try:
        assignment = next(
            item for item in assignments if item["stage_id"] == resolution.stage_id
        )
    except StopIteration as error:
        raise CheckpointValidationError("resolved stage is absent from the plan") from error
    if (
        tuple(assignment["device_ids"]) != resolution.captured_device_ids
        or assignment["process_index"] != resolution.captured_process_index
    ):
        raise CheckpointValidationError(
            "resolved physical devices disagree with the final-layout plan"
        )
    selected = sorted(
        (
            plan
            for plan in checkpoint.plans
            if plan.load_set == load_set and plan.stage_id == resolution.stage_id
        ),
        key=lambda item: item.device_slot,
    )
    if (
        len(selected) != expectation.stage_size
        or tuple(plan.device_slot for plan in selected)
        != tuple(range(expectation.stage_size))
        or tuple(plan.device_id for plan in selected)
        != resolution.captured_device_ids
    ):
        raise CheckpointValidationError(
            "final-layout files do not exactly cover the resolved stage"
        )

    import jax

    leaves_by_slot: list[dict[str, LoadedFinalLeaf]] = [
        {} for _ in resolution.devices
    ]
    device_before = [_memory_stats(device) for device in resolution.devices]
    host_rss_before = _rss_peak_bytes()
    state_files: list[dict[str, Any]] = []
    total_bytes = 0
    total_tensors = 0
    roundtrip_bytes = 0
    try:
        for plan, device in zip(selected, resolution.devices, strict=True):
            evidence = checkpoint.evidence_by_filename[plan.filename]
            file_digest = sha256()
            leaf_records = []
            path = checkpoint.root / plan.filename
            with path.open("rb", buffering=0) as stream:
                header = stream.read(len(plan.header))
                if header != plan.header:
                    raise CheckpointValidationError(
                        f"packed safetensors header drift for {plan.filename!r}"
                    )
                file_digest.update(header)
                for tensor in plan.tensors:
                    if tensor.dtype not in SUPPORTED_DTYPES:
                        raise CheckpointValidationError(
                            f"unsupported tensor dtype {tensor.dtype!r}"
                        )
                    raw = _read_exact_into(
                        stream,
                        tensor.byte_count,
                        file_digest=file_digest,
                        chunk_bytes=chunk_bytes,
                    )
                    _validate_finite(raw, tensor.dtype, chunk_bytes=chunk_bytes)
                    source_sha256 = sha256(memoryview(raw)).hexdigest()
                    host, storage_dtype = _host_array(raw, tensor)
                    array = jax.device_put(host, device)
                    array.block_until_ready()
                    array_devices = tuple(array.devices())
                    if len(array_devices) != 1 or array_devices[0] != device:
                        array.delete()
                        raise CheckpointValidationError(
                            f"tensor {tensor.name!r} did not land on its final owner"
                        )
                    if tuple(array.shape) != tensor.shape or array.nbytes != tensor.byte_count:
                        array.delete()
                        raise CheckpointValidationError(
                            f"device tensor {tensor.name!r} shape/bytes drifted"
                        )
                    if verify_device_roundtrip:
                        observed_sha256 = _array_bytes_sha256(jax.device_get(array))
                        if observed_sha256 != source_sha256:
                            array.delete()
                            raise CheckpointValidationError(
                                f"device round-trip mismatch for {tensor.name!r}"
                            )
                        roundtrip_bytes += tensor.byte_count
                    leaf = LoadedFinalLeaf(
                        array=array,
                        logical_dtype=tensor.dtype,
                        storage_dtype=storage_dtype,
                        shape=tensor.shape,
                        byte_count=tensor.byte_count,
                        sha256=source_sha256,
                    )
                    leaves_by_slot[plan.device_slot][tensor.name] = leaf
                    leaf_records.append(
                        {
                            "byte_count": tensor.byte_count,
                            "logical_dtype": tensor.dtype,
                            "name": tensor.name,
                            "sha256": source_sha256,
                            "shape": list(tensor.shape),
                            "storage_dtype": storage_dtype,
                        }
                    )
                    total_bytes += tensor.byte_count
                    total_tensors += 1
                    del host, raw
                if stream.read(1):
                    raise CheckpointValidationError(
                        f"packed file {plan.filename!r} has trailing bytes"
                    )
            observed_file_sha256 = file_digest.hexdigest()
            if observed_file_sha256 != evidence["sha256"]:
                raise CheckpointValidationError(
                    f"packed file SHA-256 mismatch for {plan.filename!r}"
                )
            if len(leaf_records) != len(plan.tensors):
                raise CheckpointValidationError(
                    f"packed tensor count drift for {plan.filename!r}"
                )
            names = set(leaves_by_slot[plan.device_slot])
            for tensor in plan.tensors:
                if tensor.dtype == "F8_E4M3" and f"{tensor.name}_scale_inv" not in names:
                    raise CheckpointValidationError(
                        f"FP8 tensor {tensor.name!r} lacks a colocated scale"
                    )
            state_files.append(
                {
                    "device_id": plan.device_id,
                    "device_slot": plan.device_slot,
                    "file_sha256": observed_file_sha256,
                    "filename": plan.filename,
                    "payload_bytes": plan.payload_bytes,
                    "tensor_count": len(plan.tensors),
                    "tensors": leaf_records,
                }
            )
        expected_bytes = sum(plan.payload_bytes for plan in selected)
        expected_tensors = sum(len(plan.tensors) for plan in selected)
        if total_bytes != expected_bytes or total_tensors != expected_tensors:
            raise CheckpointValidationError("loaded stage totals do not reconcile")
        state_manifest: dict[str, Any] = {
            "artifact_kind": "greenfield_loaded_final_layout_state",
            "device_ids": list(resolution.captured_device_ids),
            "device_roundtrip_verified": verify_device_roundtrip,
            "files": state_files,
            "layout_manifest_sha256": expectation.layout_manifest_sha256,
            "load_set": load_set,
            "packed_manifest_sha256": expectation.packed_manifest_sha256,
            "payload_bytes": total_bytes,
            "plan_id": expectation.plan_id,
            "stage_id": resolution.stage_id,
            "tensor_count": total_tensors,
        }
        state_manifest["manifest_sha256"] = sha256(
            _canonical_json(state_manifest).encode("utf-8")
        ).hexdigest()
        device_after = [_memory_stats(device) for device in resolution.devices]
        load_record = {
            "device_memory_after": device_after,
            "device_memory_before": device_before,
            "device_roundtrip_bytes": roundtrip_bytes,
            "device_roundtrip_verified": verify_device_roundtrip,
            "device_slot_count": len(resolution.devices),
            "fp8_device_dequantizations": 0,
            "fp8_host_dequantizations": 0,
            "host_global_concatenations": 0,
            "host_peak_rss_after_bytes": _rss_peak_bytes(),
            "host_peak_rss_before_bytes": host_rss_before,
            "loaded_payload_bytes": total_bytes,
            "loaded_tensor_count": total_tensors,
            "runtime_checkpoint_reshards": 0,
            "state_manifest_sha256": state_manifest["manifest_sha256"],
        }
        return LoadedFinalLayoutStage(
            stage_id=resolution.stage_id,
            device_ids=resolution.captured_device_ids,
            leaves_by_slot=tuple(leaves_by_slot),
            state_manifest=state_manifest,
            load_record=load_record,
        )
    except BaseException:
        _delete_loaded(leaves_by_slot)
        raise
