"""Append-only expert-feature derivative of a protected pipeline-local layer.

The source Pallas artifact stores complete routed experts per stage chip.
This derivative preserves identical payload bytes while redistributing each
expert's intermediate dimension over the exact LP2/LP4 stage. Gate/up use the
output slice and down uses the reciprocal contraction slice. Shared/router
tensors keep their existing ownership.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import gc
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .one_layer_pallas import (
    _canonical_json,
    _manifest_hash,
    _sha256_file,
    _tensor_nbytes,
    _tensor_sha256,
    inspect_pallas_one_layer_artifact,
)


PALLAS_FEATURE_FORMAT_VERSION = 1
PALLAS_FEATURE_ARTIFACT_KIND = "greenfield_one_layer_moe_pallas_feature"
PALLAS_FEATURE_LAYOUT_ID = "selected_expert_feature_kn_v1"
_ROUTED_WEIGHTS = frozenset(("expert_gate", "expert_up", "expert_down"))
_ROUTED_SCALES = frozenset(
    ("expert_gate_scale", "expert_up_scale", "expert_down_scale")
)
_ROUTED = _ROUTED_WEIGHTS | _ROUTED_SCALES


@dataclass(frozen=True, slots=True)
class PallasFeaturePackConfig:
    """Immutable source and destination identities for the derivative."""

    source_artifact_dir: Path
    source_artifact_uri: str
    source_manifest_sha256: str
    output_dir: Path
    code_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_artifact_dir", Path(self.source_artifact_dir)
        )
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.source_artifact_uri.startswith(
            "gs://driftbench-dsv4-uc/"
        ):
            raise ValueError("feature source must use the approved bucket")
        for name in ("source_manifest_sha256",):
            value = getattr(self, name)
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if len(self.code_hash) not in (40, 64) or any(
            character not in "0123456789abcdef"
            for character in self.code_hash
        ):
            raise ValueError("code_hash must be a lowercase Git object id")


def build_pallas_feature_layout(
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact all-expert/local-feature semantic layout."""

    geometry = source_manifest["geometry"]
    experts = int(geometry["num_experts"])
    hidden = int(geometry["hidden_size"])
    intermediate = int(geometry["intermediate_size"])
    stage_size = int(geometry["stage_size"])
    if intermediate % stage_size:
        raise ValueError("routed intermediate must divide the local stage")
    local_intermediate = intermediate // stage_size
    layout: dict[str, Any] = {
        "layout_id": PALLAS_FEATURE_LAYOUT_ID,
        "routed": {
            "expert_down": {
                "order": ["expert", "local_contraction", "output"],
                "shape": [experts, local_intermediate, hidden],
                "source_transform": "concat_experts_slice_contraction",
            },
            "expert_gate": {
                "order": ["expert", "contraction", "local_output"],
                "shape": [experts, hidden, local_intermediate],
                "source_transform": "concat_experts_slice_output",
            },
            "expert_up": {
                "order": ["expert", "contraction", "local_output"],
                "shape": [experts, hidden, local_intermediate],
                "source_transform": "concat_experts_slice_output",
            },
            "scale_order": ["expert", "local_output_block", "input_block"],
        },
        "shared": {
            "order": "checkpoint_out_in",
            "source_transform": "identity",
        },
    }
    layout["layout_sha256"] = sha256(
        _canonical_json(layout).encode("utf-8")
    ).hexdigest()
    return layout


def _source_piece(
    handle: Any,
    name: str,
    *,
    destination_slot: int,
    local_intermediate: int,
    block_shape: Sequence[int],
) -> Any:
    start = destination_slot * local_intermediate
    end = start + local_intermediate
    if name in ("expert_gate", "expert_up"):
        return handle.get_slice(name)[:, :, start:end].contiguous()
    if name == "expert_down":
        return handle.get_slice(name)[:, start:end, :].contiguous()
    block_out, block_in = (int(value) for value in block_shape)
    if name in ("expert_gate_scale", "expert_up_scale"):
        return handle.get_slice(name)[
            :, start // block_out : end // block_out, :
        ].contiguous()
    if name == "expert_down_scale":
        return handle.get_slice(name)[
            :, :, start // block_in : end // block_in
        ].contiguous()
    raise ValueError(f"not a routed feature tensor: {name!r}")


def _build_tensor(
    handles: Sequence[Any],
    name: str,
    *,
    destination_slot: int,
    local_intermediate: int,
    block_shape: Sequence[int],
) -> tuple[Any, str]:
    import torch

    if name not in _ROUTED:
        return handles[destination_slot].get_tensor(name).contiguous(), "identity"
    pieces = [
        _source_piece(
            handle,
            name,
            destination_slot=destination_slot,
            local_intermediate=local_intermediate,
            block_shape=block_shape,
        )
        for handle in handles
    ]
    try:
        value = torch.cat(pieces, dim=0).contiguous()
    finally:
        del pieces
    transform = (
        "concat_experts_slice_contraction"
        if name in ("expert_down", "expert_down_scale")
        else "concat_experts_slice_output"
    )
    return value, transform


def _feature_ownership(
    name: str,
    source_ownership: Mapping[str, Any],
    *,
    destination_slot: int,
    experts: int,
    local_intermediate: int,
) -> Mapping[str, Any]:
    if name not in _ROUTED:
        return source_ownership
    start = destination_slot * local_intermediate
    return {
        "device_slot": destination_slot,
        "expert_end_exclusive": experts,
        "expert_start": 0,
        "intermediate_end_exclusive": start + local_intermediate,
        "intermediate_start": start,
        "kind": "expert_intermediate_shard",
    }


def _pack_slot(
    source_dir: Path,
    output_dir: Path,
    source_files: Sequence[Mapping[str, Any]],
    *,
    destination_slot: int,
    geometry: Mapping[str, Any],
    source_artifact_uri: str,
    source_manifest_sha256: str,
    layout_sha256: str,
) -> dict[str, Any]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    experts = int(geometry["num_experts"])
    stage_size = int(geometry["stage_size"])
    local_intermediate = int(geometry["intermediate_size"]) // stage_size
    block_shape = geometry["fp8_block_shape"]
    source_records = [
        {str(record["name"]): record for record in file["tensors"]}
        for file in source_files
    ]
    tensors: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    with ExitStack() as stack:
        handles = [
            stack.enter_context(
                safe_open(
                    source_dir / str(file["filename"]),
                    framework="pt",
                    device="cpu",
                )
            )
            for file in source_files
        ]
        names = sorted(source_records[destination_slot])
        if any(set(records) != set(names) for records in source_records):
            raise ValueError("feature source tensor keys differ across slots")
        for name in names:
            source_record = source_records[destination_slot][name]
            value, transform = _build_tensor(
                handles,
                name,
                destination_slot=destination_slot,
                local_intermediate=local_intermediate,
                block_shape=block_shape,
            )
            dtype = str(source_record["dtype"])
            shape = list(value.shape)
            byte_count = _tensor_nbytes(shape, dtype)
            source_names = [
                {
                    "device_slot": slot,
                    "sha256": records[name]["sha256"],
                    "shape": records[name]["shape"],
                }
                for slot, records in enumerate(source_records)
            ] if name in _ROUTED else [
                {
                    "device_slot": destination_slot,
                    "sha256": source_record["sha256"],
                    "shape": source_record["shape"],
                }
            ]
            tensors[name] = value
            records.append(
                {
                    "byte_count": byte_count,
                    "dtype": dtype,
                    "name": name,
                    "ownership": _feature_ownership(
                        name,
                        source_record["ownership"],
                        destination_slot=destination_slot,
                        experts=experts,
                        local_intermediate=local_intermediate,
                    ),
                    "sha256": _tensor_sha256(value),
                    "shape": shape,
                    "sources": source_names,
                    "transform": transform,
                }
            )

    filename = f"device_slot_{destination_slot:02d}.safetensors"
    path = output_dir / filename
    partial = path.with_suffix(path.suffix + ".partial")
    save_file(
        tensors,
        partial,
        metadata={
            "artifact_kind": PALLAS_FEATURE_ARTIFACT_KIND,
            "device_slot": str(destination_slot),
            "format_version": str(PALLAS_FEATURE_FORMAT_VERSION),
            "layout_sha256": layout_sha256,
            "source_artifact_uri": source_artifact_uri,
            "source_manifest_sha256": source_manifest_sha256,
        },
    )
    partial.replace(path)
    result = {
        "device_slot": destination_slot,
        "file_byte_count": path.stat().st_size,
        "filename": filename,
        "payload_byte_count": sum(item["byte_count"] for item in records),
        "sha256": _sha256_file(path),
        "source_file_sha256s": [str(file["sha256"]) for file in source_files],
        "tensors": records,
    }
    del tensors
    gc.collect()
    return result


def pack_pallas_feature_one_layer(
    config: PallasFeaturePackConfig,
) -> dict[str, Any]:
    """Redistribute expert identities into reciprocal local feature shards."""

    source = inspect_pallas_one_layer_artifact(config.source_artifact_dir)
    if source["manifest_sha256"] != config.source_manifest_sha256:
        raise ValueError("feature source Pallas manifest identity drifted")
    expected_stage_size = {"PP8_LP4": 4, "PP16_LP2": 2}.get(
        source["plan_id"]
    )
    stage_size = int(source["geometry"]["stage_size"])
    if expected_stage_size is None or stage_size != expected_stage_size:
        raise ValueError(
            "feature derivative requires exact PP8_LP4 or PP16_LP2 geometry"
        )
    if config.output_dir.exists():
        raise FileExistsError(
            f"append-only feature destination exists: {config.output_dir}"
        )
    config.output_dir.mkdir(parents=True)
    layout = build_pallas_feature_layout(source)
    source_files = sorted(
        source["files"], key=lambda record: record["device_slot"]
    )
    files = [
        _pack_slot(
            config.source_artifact_dir,
            config.output_dir,
            source_files,
            destination_slot=slot,
            geometry=source["geometry"],
            source_artifact_uri=config.source_artifact_uri.rstrip("/"),
            source_manifest_sha256=config.source_manifest_sha256,
            layout_sha256=layout["layout_sha256"],
        )
        for slot in range(stage_size)
    ]
    manifest: dict[str, Any] = {
        "artifact_kind": PALLAS_FEATURE_ARTIFACT_KIND,
        "code_hash": config.code_hash,
        "files": files,
        "format_version": PALLAS_FEATURE_FORMAT_VERSION,
        "geometry": source["geometry"],
        "layer": source["layer"],
        "layout": layout,
        "model_id": source["model_id"],
        "packed_payload_byte_count": sum(
            int(file["payload_byte_count"]) for file in files
        ),
        "plan_group_hash": source["plan_group_hash"],
        "plan_id": source["plan_id"],
        "source_artifact_kind": source["artifact_kind"],
        "source_artifact_uri": config.source_artifact_uri.rstrip("/"),
        "source_code_hash": source["code_hash"],
        "source_layout_sha256": source["layout"]["layout_sha256"],
        "source_manifest_sha256": source["manifest_sha256"],
        "source_packed_payload_byte_count": source[
            "packed_payload_byte_count"
        ],
        "source_revision": source["source_revision"],
        "topology_hash": source["topology_hash"],
    }
    if manifest["packed_payload_byte_count"] != manifest[
        "source_packed_payload_byte_count"
    ]:
        raise ValueError("feature derivative payload bytes do not reconcile")
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    (config.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    inspect_pallas_feature_one_layer_artifact(
        config.output_dir,
        source_artifact_dir=config.source_artifact_dir,
    )
    return manifest


def inspect_pallas_feature_one_layer_artifact(
    output_dir: Path,
    *,
    source_artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify all hashes and optionally reconstruct every source transform."""

    from safetensors import safe_open
    import torch

    output_dir = Path(output_dir)
    path = output_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing feature manifest {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("artifact_kind") != PALLAS_FEATURE_ARTIFACT_KIND:
        raise ValueError("not a Pallas expert-feature artifact")
    if manifest.get("format_version") != PALLAS_FEATURE_FORMAT_VERSION:
        raise ValueError("unsupported Pallas feature format")
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise ValueError("Pallas feature manifest checksum mismatch")
    if manifest.get("layout") != build_pallas_feature_layout(manifest):
        raise ValueError("Pallas feature semantic layout drifted")
    files = sorted(
        manifest.get("files", []), key=lambda record: record["device_slot"]
    )
    stage_size = int(manifest["geometry"]["stage_size"])
    expected_stage_size = {"PP8_LP4": 4, "PP16_LP2": 2}.get(
        manifest["plan_id"]
    )
    if expected_stage_size is None or stage_size != expected_stage_size:
        raise ValueError("Pallas feature artifact plan geometry is invalid")
    if len(files) != stage_size:
        raise ValueError(
            "Pallas feature artifact device-file count disagrees with its stage"
        )

    source = None
    source_files: list[Mapping[str, Any]] = []
    if source_artifact_dir is not None:
        source = inspect_pallas_one_layer_artifact(
            Path(source_artifact_dir)
        )
        if source["manifest_sha256"] != manifest["source_manifest_sha256"]:
            raise ValueError("Pallas feature source identity drifted")
        source_files = sorted(
            source["files"], key=lambda record: record["device_slot"]
        )

    payload_total = 0
    for expected_slot, file in enumerate(files):
        if int(file["device_slot"]) != expected_slot:
            raise ValueError("Pallas feature slots are not contiguous")
        packed_path = output_dir / str(file["filename"])
        if packed_path.stat().st_size != int(file["file_byte_count"]):
            raise ValueError("Pallas feature file size mismatch")
        if _sha256_file(packed_path) != file["sha256"]:
            raise ValueError("Pallas feature file checksum mismatch")
        expected = {item["name"]: item for item in file["tensors"]}
        with ExitStack() as stack:
            packed = stack.enter_context(
                safe_open(packed_path, framework="pt", device="cpu")
            )
            expected_metadata = {
                "artifact_kind": PALLAS_FEATURE_ARTIFACT_KIND,
                "device_slot": str(expected_slot),
                "format_version": str(PALLAS_FEATURE_FORMAT_VERSION),
                "layout_sha256": manifest["layout"]["layout_sha256"],
                "source_artifact_uri": manifest["source_artifact_uri"],
                "source_manifest_sha256": manifest[
                    "source_manifest_sha256"
                ],
            }
            if packed.metadata() != expected_metadata:
                raise ValueError("Pallas feature safetensor metadata drifted")
            if set(packed.keys()) != set(expected):
                raise ValueError("Pallas feature tensor keys drifted")
            source_handles = [
                stack.enter_context(
                    safe_open(
                        Path(source_artifact_dir) / str(item["filename"]),
                        framework="pt",
                        device="cpu",
                    )
                )
                for item in source_files
            ] if source is not None else []
            for name, record in expected.items():
                tensor_slice = packed.get_slice(name)
                if list(tensor_slice.get_shape()) != record["shape"] or (
                    tensor_slice.get_dtype() != record["dtype"]
                ):
                    raise ValueError("Pallas feature tensor metadata drifted")
                if _tensor_nbytes(record["shape"], record["dtype"]) != int(
                    record["byte_count"]
                ):
                    raise ValueError("Pallas feature tensor bytes drifted")
                tensor = packed.get_tensor(name)
                if _tensor_sha256(tensor) != record["sha256"]:
                    raise ValueError("Pallas feature tensor checksum mismatch")
                if source_handles:
                    reconstructed, transform = _build_tensor(
                        source_handles,
                        name,
                        destination_slot=expected_slot,
                        local_intermediate=int(
                            manifest["geometry"]["intermediate_size"]
                        ) // stage_size,
                        block_shape=manifest["geometry"]["fp8_block_shape"],
                    )
                    if transform != record["transform"] or not torch.equal(
                        tensor, reconstructed
                    ):
                        raise ValueError(
                            "Pallas feature source transform mismatch"
                        )
        payload = sum(int(item["byte_count"]) for item in expected.values())
        if payload != int(file["payload_byte_count"]):
            raise ValueError("Pallas feature file payload mismatch")
        payload_total += payload
        gc.collect()
    if payload_total != int(manifest["packed_payload_byte_count"]):
        raise ValueError("Pallas feature manifest payload mismatch")
    return manifest
