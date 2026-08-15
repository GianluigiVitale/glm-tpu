"""Bounded WS32 final-owner derivative of a sealed PP8 MoE layer.

The source is the already protected one-layer PP8 artifact, not the original
753B checkpoint.  Routed expert identities are split from four PP8 owners to
eight WS32 expert rows and every hidden dimension is split over four feature
columns.  The shared expert is reconstructed once from the four PP8 pieces,
then explicitly replicated over the eight expert rows.  Router state is
sharded over both WS32 axes; only the compact correction bias is replicated
over the four feature columns.

This module writes checkpoint FP8 bytes as U8 so the independent JAX loader
can bitcast/dequantize directly without a runtime table transpose or reshard.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping


FORMAT_VERSION = 1
ARTIFACT_KIND = "greenfield_ws32_one_layer_moe"
SOURCE_ARTIFACT_KIND = "greenfield_one_layer_moe"
PLAN_ID = "WS32_2D"
EXPERT_AXIS_SIZE = 8
FEATURE_AXIS_SIZE = 4

_OUTPUT_NAMES = frozenset(
    {
        "correction_bias",
        "expert_down_bits",
        "expert_down_scale",
        "expert_gate_bits",
        "expert_gate_scale",
        "expert_up_bits",
        "expert_up_scale",
        "router_weight",
        "shared_down_bits",
        "shared_down_scale",
        "shared_gate_bits",
        "shared_gate_scale",
        "shared_up_bits",
        "shared_up_scale",
    }
)

_FILE_RECORD_KEYS = frozenset(
    {
        "device_slot",
        "expert_coordinate",
        "feature_coordinate",
        "file_byte_count",
        "filename",
        "payload_byte_count",
        "sha256",
        "tensors",
    }
)
_TENSOR_RECORD_KEYS = frozenset(
    {
        "byte_count",
        "dtype",
        "name",
        "ownership",
        "sha256",
        "shape",
        "source_name",
        "source_slots",
        "source_slice",
    }
)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _manifest_hash(value: Mapping[str, Any]) -> str:
    without_hash = dict(value)
    without_hash.pop("manifest_sha256", None)
    return sha256(_canonical_json(without_hash).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: Any) -> str:
    import torch

    contiguous = tensor.contiguous()
    return sha256(
        memoryview(contiguous.view(torch.uint8).numpy()).cast("B")
    ).hexdigest()


def _positive(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _lowercase_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True, slots=True)
class Ws32OneLayerPackConfig:
    """Immutable identities for one append-only WS32 derivative."""

    source_manifest_path: Path
    source_payload_dir: Path
    source_artifact_uri: str
    source_manifest_sha256: str
    output_dir: Path
    code_hash: str
    mesh_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_manifest_path", Path(self.source_manifest_path)
        )
        object.__setattr__(self, "source_payload_dir", Path(self.source_payload_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.source_artifact_uri.startswith("gs://driftbench-dsv4-uc/"):
            raise ValueError("WS32 source must use the approved bucket")
        for name in ("source_manifest_sha256", "mesh_hash"):
            value = getattr(self, name)
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if len(self.code_hash) not in (40, 64) or any(
            character not in "0123456789abcdef" for character in self.code_hash
        ):
            raise ValueError("code_hash must be a lowercase Git object id")


@dataclass(frozen=True, slots=True)
class LoadedWs32OneLayerSlot:
    """One exact final-owner file loaded without a global reconstruction."""

    device_slot: int
    expert_coordinate: int
    feature_coordinate: int
    manifest_sha256: str
    file_sha256: str
    arrays: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LoadedWs32OneLayerGlobal:
    """All WS32 one-layer leaves assembled from addressable final owners."""

    manifest: Mapping[str, Any]
    arrays: Mapping[str, Any]
    local_device_slots: tuple[Mapping[str, Any], ...]
    device_memory_before: tuple[Mapping[str, int] | None, ...]
    device_memory_after: tuple[Mapping[str, int] | None, ...]


def _source_manifest(config: Ws32OneLayerPackConfig) -> dict[str, Any]:
    if not config.source_manifest_path.is_file():
        raise FileNotFoundError(
            f"missing source manifest {config.source_manifest_path}"
        )
    value = json.loads(config.source_manifest_path.read_text())
    if value.get("artifact_kind") != SOURCE_ARTIFACT_KIND:
        raise ValueError("WS32 derivative source is not the sealed one-layer kind")
    if value.get("format_version") != 1 or value.get("plan_id") != "PP8_LP4":
        raise ValueError("WS32 derivative requires the PP8 one-layer v1 source")
    if value.get("manifest_sha256") != _manifest_hash(value):
        raise ValueError("source one-layer manifest checksum mismatch")
    if value["manifest_sha256"] != config.source_manifest_sha256:
        raise ValueError("source one-layer manifest identity drifted")
    geometry = value.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("stage_size") != 4:
        raise ValueError("source one-layer geometry is not PP8 stage size four")
    hidden = _positive(geometry.get("hidden_size"), "hidden_size")
    intermediate = _positive(
        geometry.get("intermediate_size"), "intermediate_size"
    )
    experts = _positive(geometry.get("num_experts"), "num_experts")
    block = geometry.get("fp8_block_shape")
    if not isinstance(block, list) or len(block) != 2:
        raise ValueError("source FP8 block shape is invalid")
    block_out = _positive(block[0], "fp8 block out")
    block_in = _positive(block[1], "fp8 block in")
    for value_to_split, divisor, name in (
        (hidden, FEATURE_AXIS_SIZE, "hidden/feature"),
        (experts, EXPERT_AXIS_SIZE, "experts/expert-axis"),
        (hidden, block_out * FEATURE_AXIS_SIZE, "hidden/output blocks"),
        (hidden, block_in * FEATURE_AXIS_SIZE, "hidden/input blocks"),
        (intermediate, block_out, "intermediate/output blocks"),
        (intermediate, block_in, "intermediate/input blocks"),
    ):
        if value_to_split % divisor:
            raise ValueError(f"WS32 source does not divide exactly over {name}")
    files = value.get("files")
    if not isinstance(files, list) or len(files) != 4:
        raise ValueError("source one-layer manifest must contain four PP8 files")
    if {record.get("device_slot") for record in files} != set(range(4)):
        raise ValueError("source one-layer PP8 file slots are incomplete")
    return value


def _source_files(
    config: Ws32OneLayerPackConfig,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    ordered = tuple(sorted(manifest["files"], key=lambda item: item["device_slot"]))
    expected_names = {
        name.removesuffix("_bits")
        for name in _OUTPUT_NAMES
        if name not in {"correction_bias", "router_weight"}
    } | {"correction_bias", "router_weight"}
    for slot, record in enumerate(ordered):
        if record.get("device_slot") != slot:
            raise ValueError("source PP8 file order drifted")
        path = config.source_payload_dir / str(record.get("filename"))
        if not path.is_file() or path.stat().st_size != record.get("file_byte_count"):
            raise ValueError(f"source PP8 file size drifted for slot {slot}")
        if _sha256_file(path) != record.get("sha256"):
            raise ValueError(f"source PP8 file checksum drifted for slot {slot}")
        tensor_names = {item.get("name") for item in record.get("tensors", ())}
        if tensor_names != expected_names:
            raise ValueError(f"source PP8 tensor set drifted for slot {slot}")
    return ordered


def _torch_dtype_name(tensor: Any) -> str:
    names = {
        "torch.bfloat16": "BF16",
        "torch.float32": "F32",
        "torch.uint8": "U8",
    }
    try:
        return names[str(tensor.dtype)]
    except KeyError as exc:
        raise ValueError(f"unsupported WS32 tensor dtype {tensor.dtype}") from exc


def _record(
    name: str,
    tensor: Any,
    *,
    device_slot: int,
    expert_coordinate: int,
    feature_coordinate: int,
    source_slots: tuple[int, ...],
    source_name: str,
    source_slice: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "byte_count": tensor.numel() * tensor.element_size(),
        "dtype": _torch_dtype_name(tensor),
        "name": name,
        "ownership": {
            "device_slot": device_slot,
            "expert_coordinate": expert_coordinate,
            "feature_coordinate": feature_coordinate,
            "kind": "ws32_final_owner",
        },
        "sha256": _tensor_sha256(tensor),
        "shape": list(tensor.shape),
        "source_name": source_name,
        "source_slots": list(source_slots),
        "source_slice": dict(source_slice),
    }


def _fp8_bits(tensor: Any) -> Any:
    import torch

    if tensor.dtype != torch.float8_e4m3fn:
        raise ValueError("sealed one-layer weight is not E4M3FN")
    return tensor.contiguous().view(torch.uint8)


def _load_shared_and_router(handles: tuple[Any, ...]) -> dict[str, Any]:
    import torch

    combined: dict[str, Any] = {}
    for name in ("shared_gate", "shared_up", "shared_gate_scale", "shared_up_scale"):
        combined[name] = torch.cat(
            tuple(handle.get_tensor(name) for handle in handles), dim=0
        ).contiguous()
    for name in ("shared_down", "shared_down_scale"):
        combined[name] = torch.cat(
            tuple(handle.get_tensor(name) for handle in handles), dim=1
        ).contiguous()
    for name in ("router_weight", "correction_bias"):
        reference = handles[0].get_tensor(name).contiguous()
        if any(
            not torch.equal(reference, handle.get_tensor(name))
            for handle in handles[1:]
        ):
            raise ValueError(f"source PP8 replicated tensor {name!r} drifted")
        combined[name] = reference
    return combined


def _slot_metadata(
    *,
    device_slot: int,
    expert_coordinate: int,
    feature_coordinate: int,
    mesh_hash: str,
    source_manifest_sha256: str,
) -> dict[str, str]:
    return {
        "artifact_kind": ARTIFACT_KIND,
        "device_slot": str(device_slot),
        "expert_coordinate": str(expert_coordinate),
        "feature_coordinate": str(feature_coordinate),
        "format_version": str(FORMAT_VERSION),
        "mesh_hash": mesh_hash,
        "plan_id": PLAN_ID,
        "source_manifest_sha256": source_manifest_sha256,
    }


def _expected_slot_tensor_records(
    manifest: Mapping[str, Any],
    *,
    device_slot: int,
) -> dict[str, dict[str, Any]]:
    """Derive every final-owner record from the immutable WS32 geometry."""

    geometry = manifest.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("WS32 one-layer geometry is invalid")
    if geometry.get("device_count") != 32 or (
        geometry.get("expert_axis_size") != EXPERT_AXIS_SIZE
    ) or geometry.get("feature_axis_size") != FEATURE_AXIS_SIZE:
        raise ValueError("WS32 one-layer mesh geometry drifted")
    hidden = _positive(geometry.get("hidden_size"), "hidden_size")
    intermediate = _positive(
        geometry.get("intermediate_size"), "intermediate_size"
    )
    experts = _positive(geometry.get("num_experts"), "num_experts")
    block = geometry.get("fp8_block_shape")
    if not isinstance(block, list) or len(block) != 2:
        raise ValueError("WS32 one-layer FP8 block geometry is invalid")
    block_out = _positive(block[0], "fp8 block out")
    block_in = _positive(block[1], "fp8 block in")
    for value_to_split, divisor, name in (
        (hidden, FEATURE_AXIS_SIZE, "hidden/feature"),
        (experts, EXPERT_AXIS_SIZE, "experts/expert-axis"),
        (hidden, block_out * FEATURE_AXIS_SIZE, "hidden/output blocks"),
        (hidden, block_in * FEATURE_AXIS_SIZE, "hidden/input blocks"),
        (intermediate, block_out, "intermediate/output blocks"),
        (intermediate, block_in, "intermediate/input blocks"),
    ):
        if value_to_split % divisor:
            raise ValueError(f"WS32 geometry does not divide exactly over {name}")

    expert_coordinate, feature_coordinate = divmod(
        device_slot, FEATURE_AXIS_SIZE
    )
    local_experts = experts // EXPERT_AXIS_SIZE
    hidden_local = hidden // FEATURE_AXIS_SIZE
    expert_start = expert_coordinate * local_experts
    expert_end = expert_start + local_experts
    hidden_start = feature_coordinate * hidden_local
    hidden_end = hidden_start + hidden_local
    out_block_start = hidden_start // block_out
    out_block_end = hidden_end // block_out
    in_block_start = hidden_start // block_in
    in_block_end = hidden_end // block_in
    source_slot = expert_coordinate // 2
    shared_slots = [0, 1, 2, 3]
    ownership = {
        "device_slot": device_slot,
        "expert_coordinate": expert_coordinate,
        "feature_coordinate": feature_coordinate,
        "kind": "ws32_final_owner",
    }

    specifications: dict[str, tuple[str, list[int], list[int], dict[str, Any]]] = {}
    for base in ("expert_gate", "expert_up"):
        specifications[f"{base}_bits"] = (
            "U8",
            [local_experts, intermediate, hidden_local],
            [source_slot],
            {
                "expert": [expert_start, expert_end],
                "hidden_input": [hidden_start, hidden_end],
            },
        )
        specifications[f"{base}_scale"] = (
            "F32",
            [local_experts, intermediate // block_out, hidden_local // block_in],
            [source_slot],
            {
                "expert": [expert_start, expert_end],
                "hidden_input_blocks": [in_block_start, in_block_end],
            },
        )
    specifications["expert_down_bits"] = (
        "U8",
        [local_experts, hidden_local, intermediate],
        [source_slot],
        {
            "expert": [expert_start, expert_end],
            "hidden_output": [hidden_start, hidden_end],
        },
    )
    specifications["expert_down_scale"] = (
        "F32",
        [local_experts, hidden_local // block_out, intermediate // block_in],
        [source_slot],
        {
            "expert": [expert_start, expert_end],
            "hidden_output_blocks": [out_block_start, out_block_end],
        },
    )
    for base in ("shared_gate", "shared_up"):
        specifications[f"{base}_bits"] = (
            "U8",
            [intermediate, hidden_local],
            shared_slots,
            {"hidden_input": [hidden_start, hidden_end]},
        )
        specifications[f"{base}_scale"] = (
            "F32",
            [intermediate // block_out, hidden_local // block_in],
            shared_slots,
            {"hidden_input_blocks": [in_block_start, in_block_end]},
        )
    specifications["shared_down_bits"] = (
        "U8",
        [hidden_local, intermediate],
        shared_slots,
        {"hidden_output": [hidden_start, hidden_end]},
    )
    specifications["shared_down_scale"] = (
        "F32",
        [hidden_local // block_out, intermediate // block_in],
        shared_slots,
        {"hidden_output_blocks": [out_block_start, out_block_end]},
    )
    specifications["router_weight"] = (
        "BF16",
        [local_experts, hidden_local],
        shared_slots,
        {
            "expert": [expert_start, expert_end],
            "hidden_input": [hidden_start, hidden_end],
        },
    )
    specifications["correction_bias"] = (
        "F32",
        [local_experts],
        shared_slots,
        {"expert": [expert_start, expert_end]},
    )
    dtype_bytes = {"BF16": 2, "F32": 4, "U8": 1}
    records: dict[str, dict[str, Any]] = {}
    for name, (dtype, shape, source_slots, source_slice) in specifications.items():
        element_count = 1
        for extent in shape:
            element_count *= extent
        records[name] = {
            "byte_count": element_count * dtype_bytes[dtype],
            "dtype": dtype,
            "name": name,
            "ownership": dict(ownership),
            "shape": shape,
            "source_name": name.removesuffix("_bits"),
            "source_slots": list(source_slots),
            "source_slice": source_slice,
        }
    if set(records) != _OUTPUT_NAMES:
        raise AssertionError("internal WS32 final-owner schema is incomplete")
    return records


def _write_slot(
    config: Ws32OneLayerPackConfig,
    source_manifest: Mapping[str, Any],
    source_slot: int,
    source_handle: Any,
    shared: Mapping[str, Any],
    expert_coordinate: int,
    feature_coordinate: int,
) -> dict[str, Any]:
    from safetensors.torch import save_file

    geometry = source_manifest["geometry"]
    hidden = int(geometry["hidden_size"])
    experts = int(geometry["num_experts"])
    block_out, block_in = (int(item) for item in geometry["fp8_block_shape"])
    local_experts = experts // EXPERT_AXIS_SIZE
    source_local_experts = experts // 4
    source_expert_start = source_slot * source_local_experts
    expert_start = expert_coordinate * local_experts
    local_start = expert_start - source_expert_start
    local_end = local_start + local_experts
    hidden_start = feature_coordinate * (hidden // FEATURE_AXIS_SIZE)
    hidden_end = hidden_start + hidden // FEATURE_AXIS_SIZE
    out_block_start = hidden_start // block_out
    out_block_end = hidden_end // block_out
    in_block_start = hidden_start // block_in
    in_block_end = hidden_end // block_in
    device_slot = expert_coordinate * FEATURE_AXIS_SIZE + feature_coordinate

    tensors: dict[str, Any] = {}
    source_slices: dict[str, dict[str, Any]] = {}
    for base in ("expert_gate", "expert_up"):
        tensors[f"{base}_bits"] = _fp8_bits(
            source_handle.get_tensor(base)[
                local_start:local_end, :, hidden_start:hidden_end
            ].contiguous()
        )
        tensors[f"{base}_scale"] = source_handle.get_tensor(f"{base}_scale")[
            local_start:local_end, :, in_block_start:in_block_end
        ].contiguous()
        source_slices[f"{base}_bits"] = {
            "expert": [expert_start, expert_start + local_experts],
            "hidden_input": [hidden_start, hidden_end],
        }
        source_slices[f"{base}_scale"] = {
            "expert": [expert_start, expert_start + local_experts],
            "hidden_input_blocks": [in_block_start, in_block_end],
        }
    tensors["expert_down_bits"] = _fp8_bits(
        source_handle.get_tensor("expert_down")[
            local_start:local_end, hidden_start:hidden_end, :
        ].contiguous()
    )
    tensors["expert_down_scale"] = source_handle.get_tensor(
        "expert_down_scale"
    )[local_start:local_end, out_block_start:out_block_end, :].contiguous()
    source_slices["expert_down_bits"] = {
        "expert": [expert_start, expert_start + local_experts],
        "hidden_output": [hidden_start, hidden_end],
    }
    source_slices["expert_down_scale"] = {
        "expert": [expert_start, expert_start + local_experts],
        "hidden_output_blocks": [out_block_start, out_block_end],
    }

    for base in ("shared_gate", "shared_up"):
        tensors[f"{base}_bits"] = _fp8_bits(
            shared[base][:, hidden_start:hidden_end].contiguous()
        )
        tensors[f"{base}_scale"] = shared[f"{base}_scale"][
            :, in_block_start:in_block_end
        ].contiguous()
        source_slices[f"{base}_bits"] = {
            "hidden_input": [hidden_start, hidden_end]
        }
        source_slices[f"{base}_scale"] = {
            "hidden_input_blocks": [in_block_start, in_block_end]
        }
    tensors["shared_down_bits"] = _fp8_bits(
        shared["shared_down"][hidden_start:hidden_end, :].contiguous()
    )
    tensors["shared_down_scale"] = shared["shared_down_scale"][
        out_block_start:out_block_end, :
    ].contiguous()
    source_slices["shared_down_bits"] = {
        "hidden_output": [hidden_start, hidden_end]
    }
    source_slices["shared_down_scale"] = {
        "hidden_output_blocks": [out_block_start, out_block_end]
    }

    tensors["router_weight"] = shared["router_weight"][
        expert_start : expert_start + local_experts,
        hidden_start:hidden_end,
    ].contiguous()
    tensors["correction_bias"] = shared["correction_bias"][
        expert_start : expert_start + local_experts
    ].contiguous()
    source_slices["router_weight"] = {
        "expert": [expert_start, expert_start + local_experts],
        "hidden_input": [hidden_start, hidden_end],
    }
    source_slices["correction_bias"] = {
        "expert": [expert_start, expert_start + local_experts]
    }

    records = []
    for name in sorted(tensors):
        shared_source = name.startswith("shared_")
        compact_source = name in {"router_weight", "correction_bias"}
        records.append(
            _record(
                name,
                tensors[name],
                device_slot=device_slot,
                expert_coordinate=expert_coordinate,
                feature_coordinate=feature_coordinate,
                source_slots=(0, 1, 2, 3)
                if shared_source or compact_source
                else (source_slot,),
                source_name=name.removesuffix("_bits"),
                source_slice=source_slices[name],
            )
        )
    filename = f"device_slot_{device_slot:02d}.safetensors"
    path = config.output_dir / filename
    partial = path.with_suffix(path.suffix + ".partial")
    save_file(
        tensors,
        partial,
        metadata=_slot_metadata(
            device_slot=device_slot,
            expert_coordinate=expert_coordinate,
            feature_coordinate=feature_coordinate,
            mesh_hash=config.mesh_hash,
            source_manifest_sha256=config.source_manifest_sha256,
        ),
    )
    partial.replace(path)
    result = {
        "device_slot": device_slot,
        "expert_coordinate": expert_coordinate,
        "feature_coordinate": feature_coordinate,
        "file_byte_count": path.stat().st_size,
        "filename": filename,
        "payload_byte_count": sum(record["byte_count"] for record in records),
        "sha256": _sha256_file(path),
        "tensors": records,
    }
    del tensors
    gc.collect()
    return result


def pack_ws32_one_layer(config: Ws32OneLayerPackConfig) -> dict[str, Any]:
    """Write 32 exact final-owner files and commit the manifest last."""

    from safetensors import safe_open

    if config.output_dir.exists():
        raise FileExistsError(
            f"append-only WS32 one-layer destination exists: {config.output_dir}"
        )
    source_manifest = _source_manifest(config)
    source_files = _source_files(config, source_manifest)
    config.output_dir.mkdir(parents=True)
    with ExitStack() as stack:
        handles = tuple(
            stack.enter_context(
                safe_open(
                    config.source_payload_dir / record["filename"],
                    framework="pt",
                    device="cpu",
                )
            )
            for record in source_files
        )
        shared = _load_shared_and_router(handles)
        files = []
        for source_slot, handle in enumerate(handles):
            for expert_offset in range(2):
                expert_coordinate = source_slot * 2 + expert_offset
                for feature_coordinate in range(FEATURE_AXIS_SIZE):
                    files.append(
                        _write_slot(
                            config,
                            source_manifest,
                            source_slot,
                            handle,
                            shared,
                            expert_coordinate,
                            feature_coordinate,
                        )
                    )

    shared_source_bytes = sum(
        tensor.numel() * tensor.element_size()
        for name, tensor in shared.items()
        if name.startswith("shared_")
    )
    correction_bias_bytes = (
        shared["correction_bias"].numel()
        * shared["correction_bias"].element_size()
    )
    packed_payload_bytes = sum(record["payload_byte_count"] for record in files)
    expected_packed = (
        int(source_manifest["source_payload_byte_count"])
        + 7 * shared_source_bytes
        + 3 * correction_bias_bytes
    )
    if packed_payload_bytes != expected_packed:
        raise ValueError("WS32 one-layer replication bytes do not reconcile")
    manifest: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "code_hash": config.code_hash,
        "files": sorted(files, key=lambda item: item["device_slot"]),
        "format_version": FORMAT_VERSION,
        "geometry": {
            **{
                name: value
                for name, value in source_manifest["geometry"].items()
                if name != "stage_size"
            },
            "device_count": 32,
            "expert_axis_size": EXPERT_AXIS_SIZE,
            "feature_axis_size": FEATURE_AXIS_SIZE,
        },
        "layer": source_manifest["layer"],
        "mesh_hash": config.mesh_hash,
        "model_id": source_manifest["model_id"],
        "packed_payload_byte_count": packed_payload_bytes,
        "plan_id": PLAN_ID,
        "source": {
            "artifact_kind": SOURCE_ARTIFACT_KIND,
            "artifact_uri": config.source_artifact_uri.rstrip("/"),
            "manifest_sha256": config.source_manifest_sha256,
            "packed_payload_byte_count": source_manifest[
                "packed_payload_byte_count"
            ],
            "source_payload_byte_count": source_manifest[
                "source_payload_byte_count"
            ],
        },
        "replication": {
            "correction_bias_extra_bytes": 3 * correction_bias_bytes,
            "shared_expert_extra_bytes": 7 * shared_source_bytes,
        },
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    for record in manifest["files"]:
        _verify_ws32_file(
            config.output_dir,
            manifest,
            record,
            verify_tensor_hashes=True,
        )
    manifest_path = config.output_dir / "manifest.json"
    manifest_partial = config.output_dir / "manifest.json.partial"
    with manifest_partial.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    manifest_partial.replace(manifest_path)
    directory_fd = os.open(config.output_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    inspect_ws32_one_layer(config.output_dir, verify_tensor_hashes=True)
    return manifest


def _verify_ws32_file(
    root: Path,
    manifest: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    verify_tensor_hashes: bool,
) -> None:
    from safetensors import safe_open

    if not isinstance(record, Mapping) or set(record) != _FILE_RECORD_KEYS:
        raise ValueError("WS32 file record schema drifted")
    slot = record["device_slot"]
    if not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot < 32:
        raise ValueError("WS32 file device slot is invalid")
    expert_coordinate, feature_coordinate = divmod(slot, FEATURE_AXIS_SIZE)
    if record["expert_coordinate"] != expert_coordinate or (
        record["feature_coordinate"] != feature_coordinate
    ):
        raise ValueError("WS32 file coordinates do not match device slot")
    expected_filename = f"device_slot_{slot:02d}.safetensors"
    if record["filename"] != expected_filename:
        raise ValueError(f"WS32 filename mismatch for slot {slot}")
    if not _lowercase_sha256(record["sha256"]):
        raise ValueError(f"WS32 file checksum schema mismatch for slot {slot}")
    path = root / expected_filename
    if not path.is_file():
        raise ValueError(f"missing WS32 file for slot {slot}")
    if path.stat().st_size != record["file_byte_count"]:
        raise ValueError(f"WS32 file size mismatch for slot {slot}")
    if _sha256_file(path) != record["sha256"]:
        raise ValueError(f"WS32 file checksum mismatch for slot {slot}")
    tensor_records = record["tensors"]
    if not isinstance(tensor_records, list) or any(
        not isinstance(item, Mapping) for item in tensor_records
    ):
        raise ValueError(f"WS32 tensor record schema mismatch for slot {slot}")
    expected = {item.get("name"): item for item in tensor_records}
    if len(expected) != len(tensor_records) or set(expected) != _OUTPUT_NAMES:
        raise ValueError(f"WS32 tensor set mismatch for slot {slot}")
    if [item["name"] for item in tensor_records] != sorted(_OUTPUT_NAMES):
        raise ValueError(f"WS32 tensor record order mismatch for slot {slot}")
    derived = _expected_slot_tensor_records(manifest, device_slot=slot)
    for name, tensor_record in expected.items():
        if set(tensor_record) != _TENSOR_RECORD_KEYS:
            raise ValueError(f"WS32 tensor record schema mismatch for slot {slot}:{name}")
        if not _lowercase_sha256(tensor_record["sha256"]):
            raise ValueError(f"WS32 tensor checksum schema mismatch for slot {slot}:{name}")
        without_hash = dict(tensor_record)
        without_hash.pop("sha256")
        if without_hash != derived[name]:
            raise ValueError(f"WS32 tensor ownership mismatch for slot {slot}:{name}")
    expected_payload = sum(item["byte_count"] for item in derived.values())
    if record["payload_byte_count"] != expected_payload:
        raise ValueError(f"WS32 payload metadata mismatch for slot {slot}")
    if not isinstance(record["file_byte_count"], int) or isinstance(
        record["file_byte_count"], bool
    ):
        raise ValueError(f"WS32 file byte count schema mismatch for slot {slot}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(expected):
            raise ValueError(f"WS32 file tensor keys mismatch for slot {slot}")
        metadata = handle.metadata()
        source = manifest.get("source")
        if not isinstance(source, Mapping):
            raise ValueError("WS32 source manifest record is invalid")
        expected_metadata = _slot_metadata(
            device_slot=slot,
            expert_coordinate=expert_coordinate,
            feature_coordinate=feature_coordinate,
            mesh_hash=manifest.get("mesh_hash"),
            source_manifest_sha256=source.get("manifest_sha256"),
        )
        if metadata != expected_metadata:
            raise ValueError(f"WS32 file metadata mismatch for slot {slot}")
        payload = 0
        for name, tensor_record in expected.items():
            tensor_slice = handle.get_slice(name)
            if list(tensor_slice.get_shape()) != tensor_record["shape"] or (
                tensor_slice.get_dtype() != tensor_record["dtype"]
            ):
                raise ValueError(f"WS32 tensor metadata mismatch for slot {slot}:{name}")
            payload += int(tensor_record["byte_count"])
            if verify_tensor_hashes and _tensor_sha256(
                handle.get_tensor(name)
            ) != tensor_record["sha256"]:
                raise ValueError(f"WS32 tensor checksum mismatch for slot {slot}:{name}")
        if payload != record["payload_byte_count"]:
            raise ValueError(f"WS32 payload mismatch for slot {slot}")


def inspect_ws32_one_layer(
    output_dir: Path,
    *,
    verify_tensor_hashes: bool = False,
) -> dict[str, Any]:
    """Validate the complete manifest and all final-owner files."""

    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing WS32 one-layer manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("artifact_kind") != ARTIFACT_KIND or (
        manifest.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError("unsupported WS32 one-layer artifact")
    if manifest.get("plan_id") != PLAN_ID:
        raise ValueError("WS32 one-layer plan id drifted")
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise ValueError("WS32 one-layer manifest checksum mismatch")
    if not _lowercase_sha256(manifest.get("mesh_hash")):
        raise ValueError("WS32 one-layer mesh identity drifted")
    source = manifest.get("source")
    if not isinstance(source, Mapping) or not _lowercase_sha256(
        source.get("manifest_sha256")
    ):
        raise ValueError("WS32 one-layer source identity drifted")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 32:
        raise ValueError("WS32 one-layer manifest requires 32 files")
    if {record.get("device_slot") for record in files} != set(range(32)):
        raise ValueError("WS32 one-layer final owners are incomplete")
    filenames = [record.get("filename") for record in files]
    if len(set(filenames)) != 32:
        raise ValueError("WS32 one-layer filenames are not unique")
    for record in files:
        _verify_ws32_file(
            root,
            manifest,
            record,
            verify_tensor_hashes=verify_tensor_hashes,
        )
    if sum(record["payload_byte_count"] for record in files) != manifest[
        "packed_payload_byte_count"
    ]:
        raise ValueError("WS32 one-layer packed bytes do not reconcile")
    return manifest


def load_ws32_one_layer_slot(
    output_dir: Path,
    *,
    expected_manifest_sha256: str,
    device_slot: int,
    device: object | None = None,
    payload_subdirectory: str | None = None,
) -> LoadedWs32OneLayerSlot:
    """Load one verified final owner, optionally directly onto one device."""

    import ml_dtypes
    import numpy as np
    from safetensors import safe_open
    import torch

    root = Path(output_dir)
    if payload_subdirectory is None:
        payload_root = root
    else:
        payload_path = Path(payload_subdirectory)
        if (
            not payload_subdirectory
            or payload_path.is_absolute()
            or len(payload_path.parts) != 1
            or payload_path.name in (".", "..")
        ):
            raise ValueError("WS32 loader payload subdirectory is invalid")
        payload_root = root / payload_path
        if not payload_root.is_dir():
            raise ValueError("WS32 loader payload subdirectory is missing")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("manifest_sha256") != _manifest_hash(manifest) or (
        manifest.get("manifest_sha256") != expected_manifest_sha256
    ):
        raise ValueError("WS32 loader manifest identity drifted")
    if not isinstance(device_slot, int) or isinstance(device_slot, bool) or not (
        0 <= device_slot < 32
    ):
        raise ValueError("WS32 loader device slot is invalid")
    matches = [
        record for record in manifest["files"] if record["device_slot"] == device_slot
    ]
    if len(matches) != 1:
        raise ValueError("WS32 loader final owner is ambiguous")
    record = matches[0]
    _verify_ws32_file(
        payload_root, manifest, record, verify_tensor_hashes=True
    )
    arrays: dict[str, Any] = {}
    with safe_open(
        payload_root / record["filename"], framework="pt", device="cpu"
    ) as handle:
        for name in sorted(handle.keys()):
            tensor = handle.get_tensor(name).contiguous()
            if tensor.dtype == torch.bfloat16:
                array = tensor.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
            elif tensor.dtype == torch.float32:
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f"WS32 loader found non-finite {name}")
                array = tensor.numpy()
            elif tensor.dtype == torch.uint8:
                if bool(torch.any((tensor == 0x7F) | (tensor == 0xFF))):
                    raise ValueError(f"WS32 loader found non-finite FP8 bits in {name}")
                array = tensor.numpy()
            else:
                raise ValueError(f"WS32 loader found unsupported dtype for {name}")
            if device is None:
                arrays[name] = np.asarray(array)
            else:
                import jax

                arrays[name] = jax.device_put(array, device)
    return LoadedWs32OneLayerSlot(
        device_slot=device_slot,
        expert_coordinate=int(record["expert_coordinate"]),
        feature_coordinate=int(record["feature_coordinate"]),
        manifest_sha256=manifest["manifest_sha256"],
        file_sha256=record["sha256"],
        arrays=arrays,
    )


def _memory_stats(device: object) -> Mapping[str, int] | None:
    value = device.memory_stats()
    if value is None:
        return None
    return {
        str(name): int(number)
        for name, number in value.items()
        if isinstance(number, int) and not isinstance(number, bool)
    }


def _global_tensor_layouts(
    manifest: Mapping[str, Any],
) -> Mapping[str, tuple[tuple[int, ...], tuple[str | None, ...]]]:
    geometry = manifest["geometry"]
    hidden = _positive(geometry.get("hidden_size"), "hidden_size")
    intermediate = _positive(
        geometry.get("intermediate_size"), "intermediate_size"
    )
    experts = _positive(geometry.get("num_experts"), "num_experts")
    block_out, block_in = (
        _positive(value, "fp8 block")
        for value in geometry.get("fp8_block_shape", ())
    )
    return {
        "correction_bias": ((experts,), ("expert",)),
        "expert_down_bits": (
            (experts, hidden, intermediate),
            ("expert", "feature", None),
        ),
        "expert_down_scale": (
            (experts, hidden // block_out, intermediate // block_in),
            ("expert", "feature", None),
        ),
        "expert_gate_bits": (
            (experts, intermediate, hidden),
            ("expert", None, "feature"),
        ),
        "expert_gate_scale": (
            (experts, intermediate // block_out, hidden // block_in),
            ("expert", None, "feature"),
        ),
        "expert_up_bits": (
            (experts, intermediate, hidden),
            ("expert", None, "feature"),
        ),
        "expert_up_scale": (
            (experts, intermediate // block_out, hidden // block_in),
            ("expert", None, "feature"),
        ),
        "router_weight": (
            (experts, hidden),
            ("expert", "feature"),
        ),
        "shared_down_bits": (
            (hidden, intermediate),
            ("feature", None),
        ),
        "shared_down_scale": (
            (hidden // block_out, intermediate // block_in),
            ("feature", None),
        ),
        "shared_gate_bits": (
            (intermediate, hidden),
            (None, "feature"),
        ),
        "shared_gate_scale": (
            (intermediate // block_out, hidden // block_in),
            (None, "feature"),
        ),
        "shared_up_bits": (
            (intermediate, hidden),
            (None, "feature"),
        ),
        "shared_up_scale": (
            (intermediate // block_out, hidden // block_in),
            (None, "feature"),
        ),
    }


def load_ws32_one_layer_global(
    output_dir: Path,
    *,
    expected_manifest_sha256: str,
    mesh: object,
    physical_mesh: object,
    payload_subdirectory: str | None = None,
) -> LoadedWs32OneLayerGlobal:
    """Direct-load this process's four files into the exact global mesh.

    Every process reads only the files for its addressable TPU chips.  JAX
    global arrays are then assembled from those already-device-resident local
    shards; no complete expert table or hidden row is reconstructed on a host.
    """

    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    root = Path(output_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("manifest_sha256") != _manifest_hash(manifest) or (
        manifest.get("manifest_sha256") != expected_manifest_sha256
    ):
        raise ValueError("WS32 global loader manifest identity drifted")
    if manifest.get("mesh_hash") != physical_mesh.mesh_hash:
        raise ValueError("WS32 global loader physical mesh drifted")
    observed_mesh_ids = tuple(
        tuple(int(device.id) for device in row)
        for row in mesh.devices.tolist()
    )
    if observed_mesh_ids != physical_mesh.device_ids:
        raise ValueError("WS32 JAX mesh order differs from physical slots")

    addressable = tuple(
        device
        for row in mesh.devices.tolist()
        for device in row
        if int(device.process_index) == jax.process_index()
    )
    expected_addressable = (
        32
        if jax.default_backend() == "cpu" and jax.process_count() == 1
        else 4
    )
    if len(addressable) != expected_addressable or set(addressable) != set(
        jax.local_devices()
    ):
        raise ValueError(
            "WS32 global loader addressable-device geometry drifted"
        )
    slot_by_device_id = {
        device_id: slot
        for slot, device_id in enumerate(physical_mesh.flattened_device_ids)
    }
    before = tuple(_memory_stats(device) for device in addressable)
    local: dict[int, LoadedWs32OneLayerSlot] = {}
    for device in addressable:
        device_id = int(device.id)
        loaded = load_ws32_one_layer_slot(
            root,
            expected_manifest_sha256=expected_manifest_sha256,
            device_slot=slot_by_device_id[device_id],
            device=device,
            payload_subdirectory=payload_subdirectory,
        )
        if loaded.device_slot != slot_by_device_id[device_id]:
            raise ValueError("WS32 local file/device binding drifted")
        local[device_id] = loaded

    arrays: dict[str, Any] = {}
    layouts = _global_tensor_layouts(manifest)
    if set(layouts) != _OUTPUT_NAMES:
        raise AssertionError("WS32 global tensor layout coverage drifted")
    for name in sorted(layouts):
        global_shape, spec = layouts[name]
        sharding = NamedSharding(mesh, P(*spec))
        # JAX exposes ``addressable_devices`` as an unordered set.  The array
        # constructor instead consumes shards in the insertion order of its
        # device-to-index map, which follows the sharding's device assignment.
        # Using the set's iteration order can silently attach valid final-owner
        # bytes to the wrong TPU.
        shard_devices = tuple(
            sharding.addressable_devices_indices_map(global_shape)
        )
        if set(shard_devices) != set(addressable):
            raise ValueError(f"WS32 {name} addressable device set drifted")
        shards = tuple(local[int(device.id)].arrays[name] for device in shard_devices)
        expected_local_shape = sharding.shard_shape(global_shape)
        if any(tuple(shard.shape) != expected_local_shape for shard in shards):
            raise ValueError(f"WS32 {name} local shard shape drifted")
        arrays[name] = jax.make_array_from_single_device_arrays(
            global_shape, sharding, shards
        )
    jax.block_until_ready(tuple(arrays.values()))
    after = tuple(_memory_stats(device) for device in addressable)
    local_records = tuple(
        {
            "device_id": device_id,
            "device_slot": loaded.device_slot,
            "expert_coordinate": loaded.expert_coordinate,
            "feature_coordinate": loaded.feature_coordinate,
            "file_sha256": loaded.file_sha256,
        }
        for device_id, loaded in sorted(local.items())
    )
    return LoadedWs32OneLayerGlobal(
        manifest=manifest,
        arrays=arrays,
        local_device_slots=local_records,
        device_memory_before=before,
        device_memory_after=after,
    )
