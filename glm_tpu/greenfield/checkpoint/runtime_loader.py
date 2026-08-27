"""Fail-closed loader for executable-ready decoder runtime weights.

The runtime artifact has one uniform file per physical device and one tensor
per compiled input slot.  Each process reads only its addressable files,
places each tensor directly on its recorded owner, then wraps those existing
single-device buffers in global ``jax.Array`` objects.  No source-leaf tree,
host-side global concatenation, or device-side checkpoint reshard is built.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
from hashlib import sha256
from math import prod
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..errors import CheckpointValidationError
from ..model.weights import DecoderRuntimeWeightLayout, RuntimeTensorSpec
from ..partitioning import BASE_LOAD_SET
from .full_loader import (
    VerifiedPackedCheckpoint,
    _array_bytes_sha256,
    _canonical_json,
    _mapping_hash,
    _memory_stats,
    _read_exact_into,
    _read_json,
    _rss_peak_bytes,
    _sha256_file,
    _validate_finite,
)
from .runtime_pack import (
    RUNTIME_FORMAT_VERSION,
    RUNTIME_LAYOUT_ARTIFACT_KIND,
    RUNTIME_PACK_CONTROL_KIND,
    RUNTIME_PACKED_ARTIFACT_KIND,
    RuntimeDestinationFilePlan,
    build_runtime_destination_file_plans,
)


def _digest(value: str, *, field: str, lengths: tuple[int, ...] = (64,)) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase digest")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeCheckpointLoadExpectation:
    """Externally pinned identities required before runtime bytes are read."""

    runtime_manifest_sha256: str
    runtime_layout_manifest_sha256: str
    runtime_layout_hash: str
    source_packed_manifest_sha256: str
    source_layout_manifest_sha256: str
    plan_hash: str
    schedule_hash: str
    pack_code_hash: str
    destination: str
    source_destination: str
    plan_id: str = "PP8_LP4"
    model_id: str = "zai-org/GLM-5.2-FP8"

    def __post_init__(self) -> None:
        for field in (
            "runtime_manifest_sha256",
            "runtime_layout_manifest_sha256",
            "runtime_layout_hash",
            "source_packed_manifest_sha256",
            "source_layout_manifest_sha256",
            "plan_hash",
            "schedule_hash",
        ):
            _digest(getattr(self, field), field=field)
        _digest(self.pack_code_hash, field="pack_code_hash", lengths=(40, 64))
        for field in ("destination", "source_destination"):
            value = getattr(self, field)
            if not value.startswith(
                "gs://driftbench-dsv4-uc/checkpoints/greenfield/"
            ):
                raise ValueError(
                    f"{field} must be a greenfield approved-bucket prefix"
                )
        if self.plan_id not in ("PP8_LP4", "PP16_LP2"):
            raise ValueError(
                "runtime loader supports only PP8_LP4 and PP16_LP2"
            )
        if self.model_id != "zai-org/GLM-5.2-FP8":
            raise ValueError("runtime loader supports only GLM-5.2-FP8")


@dataclass(frozen=True, slots=True)
class VerifiedRuntimeCheckpoint:
    """Fully reconciled metadata and exact expected runtime files."""

    root: Path
    control: Mapping[str, Any]
    runtime_layout_document: Mapping[str, Any]
    runtime_manifest: Mapping[str, Any]
    plans: tuple[RuntimeDestinationFilePlan, ...]
    evidence_by_filename: Mapping[str, Mapping[str, Any]]


def _runtime_totals(
    layout: DecoderRuntimeWeightLayout,
    plans: Sequence[RuntimeDestinationFilePlan],
) -> dict[str, int]:
    return {
        "file_count": len(plans),
        "padding_bytes": sum(device.padding_bytes for device in layout.devices),
        "runtime_file_bytes": sum(plan.file_bytes for plan in plans),
        "runtime_payload_bytes": sum(plan.payload_bytes for plan in plans),
        "source_leaf_count": layout.source_leaf_count,
        "source_payload_bytes": sum(
            device.source_bytes for device in layout.devices
        ),
        "tensor_count": len(layout.specs) * len(layout.devices),
    }


def verify_runtime_packed_checkpoint(
    root: Path,
    expectation: RuntimeCheckpointLoadExpectation,
    layout: DecoderRuntimeWeightLayout,
    source_checkpoint: VerifiedPackedCheckpoint,
    *,
    require_payloads: bool = True,
) -> VerifiedRuntimeCheckpoint:
    """Verify semantic lineage and, by default, all payloads and sidecars.

    Metadata-only mode exists solely to authenticate a final derivative's
    parent ledger.  It does not make the parent loadable.
    """

    root = Path(root)
    required = {
        "control": root / "control.json",
        "layout": root / "runtime_layout.json",
        "manifest": root / "runtime_manifest.json",
        "success": root / "SUCCESS",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise CheckpointValidationError(
            f"runtime checkpoint is incomplete; missing {sorted(missing)}"
        )
    control = _read_json(required["control"])
    layout_document = _read_json(required["layout"])
    manifest = _read_json(required["manifest"])
    control_hash = _mapping_hash(control, hash_field="control_sha256")
    layout_manifest_hash = _mapping_hash(
        layout_document,
        hash_field="manifest_sha256",
    )
    manifest_hash = _mapping_hash(manifest, hash_field="manifest_sha256")
    if manifest_hash != expectation.runtime_manifest_sha256:
        raise CheckpointValidationError("runtime manifest identity drifted")
    if layout_manifest_hash != expectation.runtime_layout_manifest_sha256:
        raise CheckpointValidationError("runtime layout manifest identity drifted")
    if control.get("artifact_kind") != RUNTIME_PACK_CONTROL_KIND:
        raise CheckpointValidationError("wrong runtime checkpoint control kind")
    if layout_document.get("artifact_kind") != RUNTIME_LAYOUT_ARTIFACT_KIND:
        raise CheckpointValidationError("wrong runtime layout artifact kind")
    if manifest.get("artifact_kind") != RUNTIME_PACKED_ARTIFACT_KIND:
        raise CheckpointValidationError("wrong runtime checkpoint artifact kind")
    if any(
        value.get("format_version") != RUNTIME_FORMAT_VERSION
        for value in (control, layout_document, manifest)
    ):
        raise CheckpointValidationError("runtime checkpoint format version drifted")
    if (
        layout_document.get("runtime_layout_hash") != layout.layout_hash
        or layout_document.get("layout") != layout.to_dict()
        or layout.layout_hash != expectation.runtime_layout_hash
    ):
        raise CheckpointValidationError("runtime semantic weight layout drifted")
    source_manifest_hash = source_checkpoint.packed_manifest.get(
        "manifest_sha256"
    )
    source_layout_hash = source_checkpoint.layout.get("manifest_sha256")
    if (
        source_manifest_hash != expectation.source_packed_manifest_sha256
        or source_layout_hash != expectation.source_layout_manifest_sha256
        or source_checkpoint.packed_manifest.get("plan_id")
        != expectation.plan_id
        or source_checkpoint.packed_manifest.get("destination")
        != expectation.source_destination
    ):
        raise CheckpointValidationError("runtime source checkpoint identity drifted")
    source_plans = tuple(
        plan for plan in source_checkpoint.plans if plan.load_set == BASE_LOAD_SET
    )
    plans = build_runtime_destination_file_plans(
        layout,
        source_plans,
        source_packed_manifest_sha256=expectation.source_packed_manifest_sha256,
    )
    totals = _runtime_totals(layout, plans)
    common = {
        "destination": expectation.destination,
        "format_version": RUNTIME_FORMAT_VERSION,
        "model_id": expectation.model_id,
        "pack_code_hash": expectation.pack_code_hash,
        "plan_hash": expectation.plan_hash,
        "plan_id": expectation.plan_id,
        "runtime_layout_hash": expectation.runtime_layout_hash,
        "runtime_layout_manifest_sha256": (
            expectation.runtime_layout_manifest_sha256
        ),
        "schedule_hash": expectation.schedule_hash,
        "source_checkpoint_destination": expectation.source_destination,
        "source_layout_manifest_sha256": (
            expectation.source_layout_manifest_sha256
        ),
        "source_packed_manifest_sha256": (
            expectation.source_packed_manifest_sha256
        ),
        **totals,
    }
    for name, value in (("control", control), ("manifest", manifest)):
        for field, expected in common.items():
            if value.get(field) != expected:
                raise CheckpointValidationError(
                    f"runtime {name} drifted field {field!r}"
                )
    if control.get("runtime_layout_file_sha256") != _sha256_file(
        required["layout"]
    ):
        raise CheckpointValidationError("runtime layout file SHA-256 drifted")
    if manifest.get("control_sha256") != control_hash:
        raise CheckpointValidationError("runtime manifest does not bind control")

    records = manifest.get("files")
    if not isinstance(records, list):
        raise CheckpointValidationError("runtime manifest lacks its file ledger")
    evidence_by_filename: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise CheckpointValidationError("runtime file record is not an object")
        filename = record.get("destination_filename")
        if not isinstance(filename, str) or filename in evidence_by_filename:
            raise CheckpointValidationError(
                "runtime filenames are invalid or duplicate"
            )
        evidence_by_filename[filename] = record
    if set(evidence_by_filename) != {plan.filename for plan in plans}:
        raise CheckpointValidationError("runtime file ledger is incomplete")

    source_evidence = source_checkpoint.evidence_by_filename
    for plan in plans:
        record = evidence_by_filename[plan.filename]
        try:
            source_record = source_evidence[plan.source_filename]
        except KeyError as error:
            raise CheckpointValidationError(
                f"runtime source ledger lacks {plan.source_filename!r}"
            ) from error
        expected_record = {
            "destination_filename": plan.filename,
            "device_id": plan.device_id,
            "device_slot": plan.device_slot,
            "file_bytes": plan.file_bytes,
            "header_bytes": len(plan.header),
            "header_sha256": sha256(plan.header).hexdigest(),
            "padding_bytes": plan.device_layout.padding_bytes,
            "payload_bytes": plan.payload_bytes,
            "runtime_layout_hash": plan.runtime_layout_hash,
            "source_file_sha256": source_record.get("sha256"),
            "source_filename": plan.source_filename,
            "source_leaf_count": plan.device_layout.source_leaf_count,
            "source_packed_manifest_sha256": (
                expectation.source_packed_manifest_sha256
            ),
            "source_payload_bytes": plan.device_layout.source_bytes,
            "stage_id": plan.stage_id,
            "tensor_count": len(plan.tensors),
        }
        for field, expected in expected_record.items():
            if record.get(field) != expected:
                raise CheckpointValidationError(
                    f"runtime file {plan.filename!r} drifted field {field!r}"
                )
        _digest(record.get("sha256"), field=f"{plan.filename}.sha256")
        if (
            not isinstance(record.get("generation"), int)
            or record["generation"] <= 0
            or not isinstance(record.get("crc32c"), str)
            or not record["crc32c"]
        ):
            raise CheckpointValidationError(
                f"runtime file {plan.filename!r} lacks GCS identity"
            )
        tensor_records = record.get("tensors")
        if not isinstance(tensor_records, list):
            raise CheckpointValidationError(
                f"runtime file {plan.filename!r} lacks tensor evidence"
            )
        by_name = {}
        for tensor_record in tensor_records:
            if not isinstance(tensor_record, Mapping):
                raise CheckpointValidationError("runtime tensor record is invalid")
            name = tensor_record.get("name")
            if not isinstance(name, str) or name in by_name:
                raise CheckpointValidationError(
                    "runtime tensor names are invalid or duplicate"
                )
            by_name[name] = tensor_record
        bindings = {
            tensor.spec.name: tensor
            for tensor in plan.device_layout.tensors
        }
        if set(by_name) != {tensor.spec.name for tensor in plan.tensors}:
            raise CheckpointValidationError(
                f"runtime tensor ledger is incomplete for {plan.filename!r}"
            )
        for tensor in plan.tensors:
            tensor_record = by_name[tensor.spec.name]
            expected_tensor = {
                "byte_count": tensor.byte_count,
                "name": tensor.spec.name,
                "padding": bindings[tensor.spec.name].is_padding,
            }
            if any(
                tensor_record.get(field) != expected
                for field, expected in expected_tensor.items()
            ):
                raise CheckpointValidationError(
                    f"runtime tensor evidence drifted for {tensor.spec.name!r}"
                )
            _digest(
                tensor_record.get("sha256"),
                field=f"{plan.filename}:{tensor.spec.name}.sha256",
            )
        if require_payloads:
            sidecar = _read_json(root / "evidence" / f"{plan.filename}.json")
            if sidecar != dict(record):
                raise CheckpointValidationError(
                    f"runtime file {plan.filename!r} sidecar disagrees"
                )
            try:
                observed_size = (root / plan.filename).stat().st_size
            except OSError as error:
                raise CheckpointValidationError(
                    f"runtime payload is missing: {plan.filename}"
                ) from error
            if observed_size != plan.file_bytes:
                raise CheckpointValidationError(
                    f"runtime payload size drift for {plan.filename!r}"
                )
    expected_success = f"{manifest_hash}  runtime_manifest.json\n"
    if required["success"].read_text() != expected_success:
        raise CheckpointValidationError("runtime checkpoint SUCCESS marker drifted")
    return VerifiedRuntimeCheckpoint(
        root=root,
        control=control,
        runtime_layout_document=layout_document,
        runtime_manifest=manifest,
        plans=plans,
        evidence_by_filename=evidence_by_filename,
    )


def _runtime_host_array(
    raw: bytearray,
    spec: RuntimeTensorSpec,
) -> tuple[Any, str]:
    import ml_dtypes
    import numpy as np

    if spec.dtype == "F8_E4M3":
        dtype = np.dtype("u1")
        storage_dtype = "U8_E4M3FN_BITS"
    elif spec.dtype == "BF16":
        dtype = np.dtype(ml_dtypes.bfloat16)
        storage_dtype = "BF16"
    elif spec.dtype == "F32":
        dtype = np.dtype("<f4")
        storage_dtype = "F32"
    else:
        raise CheckpointValidationError(
            f"unsupported runtime dtype {spec.dtype!r}"
        )
    expected = prod(spec.shape) * dtype.itemsize
    if expected != spec.byte_count or len(raw) != expected:
        raise CheckpointValidationError(
            f"runtime tensor {spec.name!r} shape/dtype bytes do not reconcile"
        )
    return np.frombuffer(raw, dtype=dtype).reshape((1, *spec.shape)), storage_dtype


def _validate_zero_padding(raw: bytearray, *, name: str) -> None:
    import numpy as np

    if bool(np.any(np.frombuffer(raw, dtype=np.uint8))):
        raise CheckpointValidationError(
            f"runtime padding tensor {name!r} contains non-zero bytes"
        )


def _delete_arrays(arrays: Sequence[Any]) -> None:
    for array in arrays:
        try:
            array.delete()
        except (AttributeError, RuntimeError):
            pass
    gc.collect()


@dataclass(slots=True)
class LoadedRuntimeCheckpoint:
    """Global runtime weight inputs backed by final-owner local buffers."""

    weights: dict[str, Any]
    addressable_device_ids: tuple[int, ...]
    state_manifest: Mapping[str, Any]
    load_record: Mapping[str, Any]
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        _delete_arrays(tuple(self.weights.values()))
        self.weights.clear()
        self.closed = True


def load_runtime_checkpoint(
    checkpoint: VerifiedRuntimeCheckpoint,
    expectation: RuntimeCheckpointLoadExpectation,
    layout: DecoderRuntimeWeightLayout,
    mesh: Any,
    *,
    axis_name: str = "device",
    verify_device_roundtrip: bool = False,
    chunk_bytes: int = 64 * 1024 * 1024,
) -> LoadedRuntimeCheckpoint:
    """Load addressable runtime files and assemble 364 sharded global arrays."""

    if chunk_bytes <= 0:
        raise ValueError("runtime loader chunk_bytes must be positive")
    if checkpoint.runtime_manifest.get("manifest_sha256") != (
        expectation.runtime_manifest_sha256
    ):
        raise CheckpointValidationError("verified runtime manifest is stale")
    if layout.layout_hash != expectation.runtime_layout_hash:
        raise CheckpointValidationError("runtime loader received the wrong layout")

    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P

    if axis_name not in tuple(mesh.axis_names):
        raise CheckpointValidationError("runtime mesh lacks the declared device axis")
    flat_devices = tuple(np.asarray(mesh.devices, dtype=object).reshape(-1))
    expected_ids = tuple(device.device_id for device in layout.devices)
    observed_ids = tuple(int(device.id) for device in flat_devices)
    if observed_ids != expected_ids:
        raise CheckpointValidationError(
            "runtime mesh order disagrees with the executable weight layout"
        )
    sharding_by_spec = {
        spec.name: NamedSharding(
            mesh,
            P(axis_name, *(None for _ in spec.shape)),
        )
        for spec in layout.specs
    }
    first_global_shape = (len(layout.devices), *layout.specs[0].shape)
    addressable_devices = tuple(
        sharding_by_spec[layout.specs[0].name]
        .addressable_devices_indices_map(first_global_shape)
        .keys()
    )
    addressable_ids = tuple(int(device.id) for device in addressable_devices)
    addressable_set = set(addressable_ids)
    if not addressable_set:
        raise CheckpointValidationError("runtime process has no addressable devices")
    selected = tuple(
        plan for plan in checkpoint.plans if plan.device_id in addressable_set
    )
    if set(plan.device_id for plan in selected) != addressable_set:
        raise CheckpointValidationError(
            "runtime files do not cover every addressable device"
        )
    by_stage: dict[int, set[int]] = {}
    for plan in selected:
        by_stage.setdefault(plan.stage_id, set()).add(plan.device_slot)
    layout_slots_by_stage: dict[int, set[int]] = {}
    for device in layout.devices:
        layout_slots_by_stage.setdefault(device.stage_id, set()).add(
            device.device_slot
        )
    stage_sizes = {len(slots) for slots in layout_slots_by_stage.values()}
    if len(stage_sizes) != 1:
        raise CheckpointValidationError(
            "runtime layout does not contain uniform LP2/LP4 stages"
        )
    stage_size = next(iter(stage_sizes), 0)
    expected_slots = set(range(stage_size))
    if stage_size not in (2, 4) or any(
        slots != expected_slots for slots in layout_slots_by_stage.values()
    ):
        raise CheckpointValidationError(
            "runtime layout does not contain complete LP2/LP4 stages"
        )
    if any(slots != expected_slots for slots in by_stage.values()):
        raise CheckpointValidationError(
            "runtime addressable files do not contain complete local stages"
        )
    plan_by_device = {plan.device_id: plan for plan in selected}
    device_by_id = {int(device.id): device for device in addressable_devices}
    local_by_name: dict[str, dict[int, Any]] = {
        spec.name: {} for spec in layout.specs
    }
    all_local_arrays: list[Any] = []
    global_weights: dict[str, Any] = {}
    device_before = [_memory_stats(device) for device in addressable_devices]
    host_rss_before = _rss_peak_bytes()
    file_records = []
    loaded_payload_bytes = 0
    loaded_tensor_count = 0
    roundtrip_bytes = 0
    try:
        for device_id in addressable_ids:
            plan = plan_by_device[device_id]
            device = device_by_id[device_id]
            evidence = checkpoint.evidence_by_filename[plan.filename]
            tensor_evidence = {
                item["name"]: item for item in evidence["tensors"]
            }
            bindings = {
                tensor.spec.name: tensor
                for tensor in plan.device_layout.tensors
            }
            file_digest = sha256()
            observed_tensors = []
            path = checkpoint.root / plan.filename
            with path.open("rb", buffering=0) as stream:
                header = stream.read(len(plan.header))
                if header != plan.header:
                    raise CheckpointValidationError(
                        f"runtime safetensors header drift for {plan.filename!r}"
                    )
                file_digest.update(header)
                for tensor in plan.tensors:
                    raw = _read_exact_into(
                        stream,
                        tensor.byte_count,
                        file_digest=file_digest,
                        chunk_bytes=chunk_bytes,
                    )
                    observed_sha256 = sha256(memoryview(raw)).hexdigest()
                    record = tensor_evidence[tensor.spec.name]
                    if observed_sha256 != record["sha256"]:
                        raise CheckpointValidationError(
                            f"runtime tensor SHA-256 mismatch for "
                            f"{tensor.spec.name!r}"
                        )
                    binding = bindings[tensor.spec.name]
                    if binding.is_padding:
                        _validate_zero_padding(raw, name=tensor.spec.name)
                    else:
                        _validate_finite(
                            raw,
                            tensor.spec.dtype,
                            chunk_bytes=chunk_bytes,
                        )
                    host, storage_dtype = _runtime_host_array(raw, tensor.spec)
                    array = jax.device_put(host, device)
                    array.block_until_ready()
                    array_devices = tuple(array.devices())
                    if len(array_devices) != 1 or array_devices[0] != device:
                        array.delete()
                        raise CheckpointValidationError(
                            f"runtime tensor {tensor.spec.name!r} did not land "
                            "on its final owner"
                        )
                    if tuple(array.shape) != (1, *tensor.spec.shape) or (
                        array.nbytes != tensor.byte_count
                    ):
                        array.delete()
                        raise CheckpointValidationError(
                            f"runtime device tensor {tensor.spec.name!r} drifted"
                        )
                    if verify_device_roundtrip:
                        roundtrip_sha256 = _array_bytes_sha256(
                            jax.device_get(array)
                        )
                        if roundtrip_sha256 != observed_sha256:
                            array.delete()
                            raise CheckpointValidationError(
                                f"runtime device round-trip mismatch for "
                                f"{tensor.spec.name!r}"
                            )
                        roundtrip_bytes += tensor.byte_count
                    local_by_name[tensor.spec.name][device_id] = array
                    all_local_arrays.append(array)
                    observed_tensors.append(
                        {
                            "byte_count": tensor.byte_count,
                            "logical_dtype": tensor.spec.dtype,
                            "name": tensor.spec.name,
                            "padding": binding.is_padding,
                            "sha256": observed_sha256,
                            "shape": list(tensor.spec.shape),
                            "storage_dtype": storage_dtype,
                        }
                    )
                    loaded_payload_bytes += tensor.byte_count
                    loaded_tensor_count += 1
                    del host, raw
                if stream.read(1):
                    raise CheckpointValidationError(
                        f"runtime file {plan.filename!r} has trailing bytes"
                    )
            observed_file_sha256 = file_digest.hexdigest()
            if observed_file_sha256 != evidence["sha256"]:
                raise CheckpointValidationError(
                    f"runtime file SHA-256 mismatch for {plan.filename!r}"
                )
            file_records.append(
                {
                    "device_id": plan.device_id,
                    "device_slot": plan.device_slot,
                    "file_sha256": observed_file_sha256,
                    "filename": plan.filename,
                    "payload_bytes": plan.payload_bytes,
                    "stage_id": plan.stage_id,
                    "tensor_count": len(observed_tensors),
                    "tensors": observed_tensors,
                }
            )

        for spec in layout.specs:
            global_shape = (len(layout.devices), *spec.shape)
            sharding = sharding_by_spec[spec.name]
            ordered_devices = tuple(
                sharding.addressable_devices_indices_map(global_shape).keys()
            )
            arrays = tuple(
                local_by_name[spec.name][int(device.id)]
                for device in ordered_devices
            )
            global_weights[spec.name] = jax.make_array_from_single_device_arrays(
                global_shape,
                sharding,
                arrays,
            )
        expected_payload = sum(plan.payload_bytes for plan in selected)
        expected_tensors = len(selected) * len(layout.specs)
        if (
            loaded_payload_bytes != expected_payload
            or loaded_tensor_count != expected_tensors
        ):
            raise CheckpointValidationError(
                "runtime loaded addressable totals do not reconcile"
            )
        state_manifest: dict[str, Any] = {
            "addressable_device_ids": list(addressable_ids),
            "artifact_kind": "greenfield_loaded_runtime_checkpoint_state",
            "device_roundtrip_verified": verify_device_roundtrip,
            "files": file_records,
            "global_tensor_count": len(global_weights),
            "loaded_payload_bytes": loaded_payload_bytes,
            "loaded_tensor_count": loaded_tensor_count,
            "plan_id": expectation.plan_id,
            "runtime_layout_hash": expectation.runtime_layout_hash,
            "runtime_manifest_sha256": expectation.runtime_manifest_sha256,
            "stage_ids": sorted(by_stage),
        }
        state_manifest["manifest_sha256"] = sha256(
            _canonical_json(state_manifest).encode("utf-8")
        ).hexdigest()
        load_record = {
            "addressable_device_count": len(addressable_devices),
            "device_memory_after": [
                _memory_stats(device) for device in addressable_devices
            ],
            "device_memory_before": device_before,
            "device_roundtrip_bytes": roundtrip_bytes,
            "device_roundtrip_verified": verify_device_roundtrip,
            "fp8_device_dequantizations": 0,
            "fp8_host_dequantizations": 0,
            "global_array_count": len(global_weights),
            "host_global_concatenations": 0,
            "host_peak_rss_after_bytes": _rss_peak_bytes(),
            "host_peak_rss_before_bytes": host_rss_before,
            "loaded_payload_bytes": loaded_payload_bytes,
            "loaded_tensor_count": loaded_tensor_count,
            "runtime_checkpoint_reshards": 0,
            "state_manifest_sha256": state_manifest["manifest_sha256"],
        }
        all_local_arrays.clear()
        local_by_name.clear()
        return LoadedRuntimeCheckpoint(
            weights=global_weights,
            addressable_device_ids=addressable_ids,
            state_manifest=state_manifest,
            load_record=load_record,
        )
    except BaseException:
        _delete_arrays(tuple(global_weights.values()))
        _delete_arrays(all_local_arrays)
        raise
