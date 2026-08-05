"""Complete plan-aware checkpoint layout manifest for Gate B.

This is the pre-payload ownership contract.  It maps every source leaf to an
explicit stage, physical device, local slot, destination file, and exact
slice, while separating the optional MTP load set from the base decoder.  A
later packer must fill file byte/checksum evidence without changing any
ownership record.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from ..errors import CheckpointValidationError
from ..types import ExecutionPlan
from .layer_assigner import PartitionedPlan
from .ownership import (
    BASE_LOAD_SET,
    MTP_LOAD_SET,
    PlacementLedger,
    PlacementRecipe,
)
from .source_inventory import SourceInventory, SourceTensor


FORMAT_VERSION = 1
ARTIFACT_KIND = "greenfield_full_checkpoint_layout"


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _manifest_hash(value: Mapping[str, Any]) -> str:
    unhashed = dict(value)
    unhashed.pop("manifest_sha256", None)
    return sha256(_canonical_json(unhashed).encode("utf-8")).hexdigest()


def _digest(value: object, *, field: str, lengths: tuple[int, ...] = (64,)) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CheckpointValidationError(f"{field} is not a lowercase digest")
    return value


def _recipe_stage(
    recipe: PlacementRecipe,
    plan: PartitionedPlan,
) -> int:
    if recipe.load_set == MTP_LOAD_SET:
        return plan.mtp_stage_id
    if recipe.layer_id is None:
        if recipe.source_name == "model.embed_tokens.weight":
            return 0
        if recipe.source_name in ("lm_head.weight", "model.norm.weight"):
            return len(plan.stages) - 1
        raise CheckpointValidationError(
            f"no destination stage for global tensor {recipe.source_name!r}"
        )
    for stage in plan.stages:
        if stage.layer_start <= recipe.layer_id < stage.layer_end_exclusive:
            return stage.stage_id
    raise CheckpointValidationError(
        f"base tensor {recipe.source_name!r} has no owning pipeline stage"
    )


def _destination_filename(load_set: str, stage: int, slot: int) -> str:
    return (
        f"{load_set}/stage_{stage:02d}/device_slot_{slot:02d}.safetensors"
    )


def build_layout_manifest(
    *,
    inventory: SourceInventory,
    ledger: PlacementLedger,
    partition: PartitionedPlan,
    source_uri: str,
    code_hash: str,
    topology_hash: str,
    plan_group_hash: str,
) -> dict[str, Any]:
    """Map all source metadata to final files and validate every byte."""

    if not source_uri.startswith("gs://driftbench-dsv4-uc/"):
        raise CheckpointValidationError("source URI must use the approved bucket")
    _digest(code_hash, field="code_hash", lengths=(40, 64))
    _digest(topology_hash, field="topology_hash")
    _digest(plan_group_hash, field="plan_group_hash")
    if ledger.source_inventory_sha256 != inventory.inventory_sha256:
        raise CheckpointValidationError(
            "placement ledger does not bind the supplied source inventory"
        )
    if partition.source_inventory_sha256 != inventory.inventory_sha256:
        raise CheckpointValidationError(
            "partition plan does not bind the supplied source inventory"
        )
    if partition.execution_plan.topology.topology_hash != topology_hash:
        raise CheckpointValidationError(
            "partition topology does not match the protected topology hash"
        )
    if ledger.local_parallel_size != partition.execution_plan.local_parallel_size:
        raise CheckpointValidationError(
            "placement and execution plan local sizes disagree"
        )
    tensors = {tensor.name: tensor for tensor in inventory.tensors}
    if set(tensors) != {recipe.source_name for recipe in ledger.recipes}:
        raise CheckpointValidationError(
            "placement recipes do not cover the exact inventory leaf set"
        )
    stages = {stage.stage_id: stage for stage in partition.stages}
    placements = []
    file_totals: dict[str, dict[str, Any]] = {}
    base_parameter: dict[tuple[int, int], int] = {}
    base_scale: dict[tuple[int, int], int] = {}
    optional_mtp: dict[tuple[int, int], int] = {}
    scale_slots: dict[str, set[tuple[int, int, str]]] = {}
    weight_slots: dict[str, set[tuple[int, int, str]]] = {}
    for recipe in ledger.recipes:
        tensor = tensors[recipe.source_name]
        stage_id = _recipe_stage(recipe, partition)
        stage = stages[stage_id]
        destinations = []
        locality = set()
        for shard in recipe.shards:
            device_id = stage.device_ids[shard.device_slot]
            filename = _destination_filename(
                recipe.load_set, stage_id, shard.device_slot
            )
            destination = shard.to_dict()
            destination.update(
                {
                    "device_id": device_id,
                    "filename": filename,
                    "stage_id": stage_id,
                }
            )
            destinations.append(destination)
            locality.add((stage_id, shard.device_slot, recipe.load_set))
            record = file_totals.setdefault(
                filename,
                {
                    "device_id": device_id,
                    "device_slot": shard.device_slot,
                    "filename": filename,
                    "load_set": recipe.load_set,
                    "planned_payload_bytes": 0,
                    "stage_id": stage_id,
                    "tensor_count": 0,
                },
            )
            if (
                record["device_id"] != device_id
                or record["stage_id"] != stage_id
                or record["device_slot"] != shard.device_slot
                or record["load_set"] != recipe.load_set
            ):
                raise CheckpointValidationError(
                    f"destination file identity collision for {filename!r}"
                )
            record["planned_payload_bytes"] += shard.byte_count
            record["tensor_count"] += 1
            key = (stage_id, shard.device_slot)
            if recipe.load_set == MTP_LOAD_SET:
                optional_mtp[key] = optional_mtp.get(key, 0) + shard.byte_count
            elif recipe.value_class == "fp8_scale":
                base_scale[key] = base_scale.get(key, 0) + shard.byte_count
            else:
                base_parameter[key] = (
                    base_parameter.get(key, 0) + shard.byte_count
                )
        if recipe.value_class == "fp8_scale":
            scale_slots[recipe.source_name] = locality
        else:
            weight_slots[recipe.source_name] = locality
        placements.append(
            {
                "destinations": destinations,
                "layer_id": recipe.layer_id,
                "layout": recipe.layout,
                "load_set": recipe.load_set,
                "packed_byte_count": recipe.packed_byte_count,
                "source": tensor.to_dict(),
                "value_class": recipe.value_class,
            }
        )
    for scale_name, slots in scale_slots.items():
        weight_name = scale_name.removesuffix("_scale_inv")
        if weight_slots.get(weight_name) != slots:
            raise CheckpointValidationError(
                f"FP8 scale {scale_name!r} is not local to its weight destinations"
            )
    for stage in partition.stages:
        for slot in range(len(stage.device_ids)):
            key = (stage.stage_id, slot)
            if base_parameter.get(key, 0) != stage.parameter_slot_bytes[slot]:
                raise CheckpointValidationError(
                    f"stage {stage.stage_id} slot {slot} parameter bytes disagree"
                )
            if base_scale.get(key, 0) != stage.scale_slot_bytes[slot]:
                raise CheckpointValidationError(
                    f"stage {stage.stage_id} slot {slot} scale bytes disagree"
                )
            if optional_mtp.get(key, 0) != stage.optional_mtp_slot_bytes[slot]:
                raise CheckpointValidationError(
                    f"stage {stage.stage_id} slot {slot} optional MTP bytes disagree"
                )
    destination_files = [file_totals[name] for name in sorted(file_totals)]
    packed_payload_bytes = sum(
        record["planned_payload_bytes"] for record in destination_files
    )
    if packed_payload_bytes != ledger.packed_payload_bytes:
        raise CheckpointValidationError(
            "destination file payload does not reconcile placement ledger"
        )
    value: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "code_hash": code_hash,
        "destination_files": destination_files,
        "format_version": FORMAT_VERSION,
        "packed_payload_bytes": packed_payload_bytes,
        "placements": placements,
        "plan_group_hash": plan_group_hash,
        "plan_id": partition.execution_plan.name.value,
        "plan_manifest": partition.to_dict(),
        "plan_manifest_sha256": partition.plan_manifest_sha256,
        "source": {
            "config_filename": inventory.config_filename,
            "config_sha256": inventory.config_sha256,
            "file_bytes": inventory.file_bytes,
            "files": [source_file.to_dict() for source_file in inventory.files],
            "index_filename": inventory.index_filename,
            "index_sha256": inventory.index_sha256,
            "inventory_sha256": inventory.inventory_sha256,
            "leaf_count": len(inventory.tensors),
            "model_id": inventory.model_id,
            "payload_bytes": inventory.payload_bytes,
            "revision": inventory.source_revision,
            "uri": source_uri,
        },
        "topology_hash": topology_hash,
    }
    value["manifest_sha256"] = _manifest_hash(value)
    validate_layout_manifest(value)
    return value


def validate_layout_manifest(value: Mapping[str, Any]) -> None:
    """Fail closed on content hash, plan hash, leaf/file, or byte drift."""

    if value.get("artifact_kind") != ARTIFACT_KIND:
        raise CheckpointValidationError("wrong checkpoint layout artifact kind")
    if value.get("format_version") != FORMAT_VERSION:
        raise CheckpointValidationError("unsupported checkpoint layout format")
    if value.get("manifest_sha256") != _manifest_hash(value):
        raise CheckpointValidationError("checkpoint layout manifest SHA-256 mismatch")
    plan_manifest = value.get("plan_manifest")
    if not isinstance(plan_manifest, Mapping):
        raise CheckpointValidationError("layout manifest lacks a plan manifest")
    execution_value = plan_manifest.get("execution_plan")
    if not isinstance(execution_value, Mapping):
        raise CheckpointValidationError("layout manifest lacks an execution plan")
    try:
        execution = ExecutionPlan.from_dict(execution_value)
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointValidationError("layout execution plan is invalid") from exc
    if plan_manifest.get("execution_plan_sha256") != execution.plan_hash:
        raise CheckpointValidationError("layout execution plan SHA-256 mismatch")
    expected_plan_manifest_hash = sha256(
        _canonical_json(plan_manifest).encode("utf-8")
    ).hexdigest()
    if value.get("plan_manifest_sha256") != expected_plan_manifest_hash:
        raise CheckpointValidationError("layout plan manifest SHA-256 mismatch")
    if value.get("plan_id") != execution.name.value:
        raise CheckpointValidationError("layout plan id disagrees with execution plan")
    if value.get("topology_hash") != execution.topology.topology_hash:
        raise CheckpointValidationError("layout topology SHA-256 mismatch")
    _digest(value.get("code_hash"), field="code_hash", lengths=(40, 64))
    _digest(value.get("plan_group_hash"), field="plan_group_hash")
    source = value.get("source")
    placements = value.get("placements")
    files = value.get("destination_files")
    if (
        not isinstance(source, Mapping)
        or not isinstance(placements, list)
        or not isinstance(files, list)
    ):
        raise CheckpointValidationError(
            "layout source, placements, and destination files are required"
        )
    names = []
    source_bytes = 0
    packed_bytes = 0
    aggregates: dict[str, tuple[int, int]] = {}
    scale_slots: dict[str, set[tuple[int, int, str]]] = {}
    weight_slots: dict[str, set[tuple[int, int, str]]] = {}
    assignments = {
        assignment.stage_id: assignment
        for assignment in execution.stage_assignments
    }
    for placement in placements:
        if not isinstance(placement, Mapping):
            raise CheckpointValidationError("layout placement must be an object")
        tensor = placement.get("source")
        destinations = placement.get("destinations")
        if not isinstance(tensor, Mapping) or not isinstance(destinations, list):
            raise CheckpointValidationError(
                "layout placement requires source and destinations"
            )
        try:
            source_tensor = SourceTensor.from_dict(tensor)
        except (TypeError, ValueError) as exc:
            raise CheckpointValidationError(
                "layout source tensor metadata is invalid"
            ) from exc
        name = source_tensor.name
        byte_count = source_tensor.byte_count
        names.append(name)
        source_bytes += byte_count
        locality = set()
        placement_bytes = 0
        for destination in destinations:
            if not isinstance(destination, Mapping):
                raise CheckpointValidationError("layout destination must be an object")
            filename = destination.get("filename")
            shard_bytes = destination.get("byte_count")
            stage_id = destination.get("stage_id")
            slot = destination.get("device_slot")
            device_id = destination.get("device_id")
            load_set = placement.get("load_set")
            if (
                not isinstance(filename, str)
                or not isinstance(shard_bytes, int)
                or shard_bytes < 0
                or not isinstance(stage_id, int)
                or not isinstance(slot, int)
                or load_set not in (BASE_LOAD_SET, MTP_LOAD_SET)
            ):
                raise CheckpointValidationError("layout destination identity is invalid")
            assignment = assignments.get(stage_id)
            if (
                assignment is None
                or not 0 <= slot < len(assignment.device_ids)
                or device_id != assignment.device_ids[slot]
                or filename != _destination_filename(load_set, stage_id, slot)
            ):
                raise CheckpointValidationError(
                    f"layout destination for {name!r} disagrees with physical plan"
                )
            previous_bytes, previous_count = aggregates.get(filename, (0, 0))
            aggregates[filename] = (
                previous_bytes + shard_bytes,
                previous_count + 1,
            )
            placement_bytes += shard_bytes
            locality.add((stage_id, slot, load_set))
        if placement.get("packed_byte_count") != placement_bytes:
            raise CheckpointValidationError(
                f"placement {name!r} packed bytes do not reconcile"
            )
        packed_bytes += placement_bytes
        target = (
            scale_slots
            if placement.get("value_class") == "fp8_scale"
            else weight_slots
        )
        target[name] = locality
    if len(names) != len(set(names)) or len(names) != source.get("leaf_count"):
        raise CheckpointValidationError(
            "layout source placement names are duplicate or incomplete"
        )
    if source_bytes != source.get("payload_bytes"):
        raise CheckpointValidationError("layout source payload bytes do not reconcile")
    if packed_bytes != value.get("packed_payload_bytes"):
        raise CheckpointValidationError("layout packed payload bytes do not reconcile")
    file_names = []
    for record in files:
        if not isinstance(record, Mapping):
            raise CheckpointValidationError("destination file record must be an object")
        filename = record.get("filename")
        if not isinstance(filename, str):
            raise CheckpointValidationError("destination filename is invalid")
        file_names.append(filename)
        if aggregates.get(filename) != (
            record.get("planned_payload_bytes"),
            record.get("tensor_count"),
        ):
            raise CheckpointValidationError(
                f"destination file {filename!r} does not reconcile placements"
            )
    if len(file_names) != len(set(file_names)) or set(file_names) != set(aggregates):
        raise CheckpointValidationError(
            "destination file ledger is duplicate or incomplete"
        )
    for scale_name, locality in scale_slots.items():
        if weight_slots.get(scale_name.removesuffix("_scale_inv")) != locality:
            raise CheckpointValidationError(
                f"scale {scale_name!r} is not colocated with its weight"
            )


def write_layout_manifest(value: Mapping[str, Any], output: Path) -> None:
    """Append-only atomic manifest writer."""

    validate_layout_manifest(value)
    path = Path(output)
    if path.exists():
        raise CheckpointValidationError(
            f"refusing to overwrite checkpoint layout manifest {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise CheckpointValidationError(
            f"stale checkpoint layout temporary exists: {temporary}"
        )
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def inspect_layout_manifest(path: Path) -> dict[str, Any]:
    """Read, fully reconcile, and return a layout manifest."""

    try:
        decoded = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointValidationError(
            f"cannot parse checkpoint layout manifest {path}"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise CheckpointValidationError("checkpoint layout manifest must be an object")
    validate_layout_manifest(decoded)
    return dict(decoded)
