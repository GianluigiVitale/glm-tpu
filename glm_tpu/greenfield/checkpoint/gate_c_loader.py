"""Direct final-owner loader for the bounded Gate C PP8 checkpoint."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import gc
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from ..errors import CheckpointValidationError
from .gate_c import inspect_gate_c_checkpoint, read_gate_c_layout
from .one_layer_loader import (
    StageDeviceResolution,
    _assemble_global,
    _device_put_dequantized,
    _memory_stats,
    _rss_peak_bytes,
    _torch_bfloat16_numpy,
)
from .stream_pack import build_destination_file_plans


FP8_BLOCK_SHAPE = (128, 128)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(
    value: object,
    *,
    field: str,
    lengths: tuple[int, ...] = (64,),
) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase hexadecimal digest")
    return value


@dataclass(frozen=True, slots=True)
class GateCLoadExpectation:
    """Immutable identities required before placing any Gate C weight."""

    packed_manifest_sha256: str
    layout_manifest_sha256: str
    parent_layout_manifest_sha256: str
    oracle_manifest_sha256: str
    source_revision: str
    topology_hash: str
    plan_group_hash: str
    packed_code_hash: str
    plan_id: str = "PP8_LP4"
    stage_id: int = 0

    def __post_init__(self) -> None:
        for field, lengths in (
            ("packed_manifest_sha256", (64,)),
            ("layout_manifest_sha256", (64,)),
            ("parent_layout_manifest_sha256", (64,)),
            ("oracle_manifest_sha256", (64,)),
            ("topology_hash", (64,)),
            ("plan_group_hash", (64,)),
            ("packed_code_hash", (40, 64)),
        ):
            _digest(getattr(self, field), field=field, lengths=lengths)
        if not self.source_revision:
            raise ValueError("source_revision must be non-empty")
        if self.plan_id != "PP8_LP4" or self.stage_id != 0:
            raise ValueError("bounded Gate C loading requires PP8 stage 0")

    @property
    def stage_size(self) -> int:
        return 4


@dataclass(slots=True)
class LoadedGateCCheckpoint:
    """Dequantized logical weights backed only by their final local owners."""

    mesh: Any
    weights: Mapping[str, Any]
    manifest: Mapping[str, Any]
    layout: Mapping[str, Any]
    state_manifest: Mapping[str, Any]
    load_record: Mapping[str, Any]

    def delete(self) -> None:
        """Release every owned device array once after validation."""

        for value in self.weights.values():
            value.delete()


def verify_gate_c_load_contract(
    artifact_dir: Path,
    expectation: GateCLoadExpectation,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash the complete bounded artifact and require all protected identities."""

    artifact_dir = Path(artifact_dir)
    manifest = inspect_gate_c_checkpoint(artifact_dir)
    layout = read_gate_c_layout(artifact_dir / "layout_manifest.json")
    expected = {
        "packed_manifest_sha256": (
            expectation.packed_manifest_sha256,
            manifest.get("manifest_sha256"),
        ),
        "layout_manifest_sha256": (
            expectation.layout_manifest_sha256,
            layout.get("manifest_sha256"),
        ),
        "manifest.layout_manifest_sha256": (
            expectation.layout_manifest_sha256,
            manifest.get("layout_manifest_sha256"),
        ),
        "parent_layout_manifest_sha256": (
            expectation.parent_layout_manifest_sha256,
            layout.get("parent_layout_manifest_sha256"),
        ),
        "oracle_manifest_sha256": (
            expectation.oracle_manifest_sha256,
            layout.get("oracle_manifest_sha256"),
        ),
        "source_revision": (
            expectation.source_revision,
            layout.get("source", {}).get("revision"),
        ),
        "topology_hash": (
            expectation.topology_hash,
            layout.get("topology_hash"),
        ),
        "plan_group_hash": (
            expectation.plan_group_hash,
            layout.get("plan_group_hash"),
        ),
        "packed_code_hash": (
            expectation.packed_code_hash,
            manifest.get("code_hash"),
        ),
        "plan_id": (expectation.plan_id, layout.get("plan_id")),
    }
    mismatches = {
        name: {"expected": pair[0], "observed": pair[1]}
        for name, pair in expected.items()
        if pair[0] != pair[1]
    }
    if mismatches:
        raise CheckpointValidationError(
            f"Gate C checkpoint load contract mismatch: {mismatches}"
        )
    return manifest, layout


def _sharding_for_placement(mesh: Any, placement: Mapping[str, Any]) -> Any:
    from jax.sharding import NamedSharding, PartitionSpec as P

    layout = placement["layout"]
    rank = len(placement["source"]["shape"])
    if layout == "replicated":
        return NamedSharding(mesh, P())
    if layout != "axis_sharded":
        raise CheckpointValidationError(
            f"unsupported Gate C device layout {layout!r}"
        )
    axes = {destination.get("axis") for destination in placement["destinations"]}
    if len(axes) != 1:
        raise CheckpointValidationError("Gate C sharded placement axes disagree")
    axis = next(iter(axes))
    if not isinstance(axis, int) or isinstance(axis, bool) or not 0 <= axis < rank:
        raise CheckpointValidationError("Gate C sharded placement axis is invalid")
    spec: list[str | None] = [None] * rank
    spec[axis] = "stage"
    return NamedSharding(mesh, P(*spec))


def _validate_fp8_pair(weight: Any, scale: Any) -> None:
    import torch

    if weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.float32:
        raise CheckpointValidationError("Gate C FP8 weight/scale dtype drifted")
    if weight.ndim != 2 or scale.ndim != 2:
        raise CheckpointValidationError("Gate C FP8 weights must be rank two")
    expected = tuple(
        (dimension + block - 1) // block
        for dimension, block in zip(
            weight.shape, FP8_BLOCK_SHAPE, strict=True
        )
    )
    if tuple(scale.shape) != expected:
        raise CheckpointValidationError(
            f"Gate C FP8 scale shape must be {expected}, got {tuple(scale.shape)}"
        )


def _device_put_plain(jax: Any, tensor: Any, device: object) -> Any:
    import torch

    if tensor.dtype == torch.bfloat16:
        if not bool(torch.isfinite(tensor.float()).all()):
            raise CheckpointValidationError("Gate C BF16 tensor is non-finite")
        host = _torch_bfloat16_numpy(tensor)
    elif tensor.dtype == torch.float32:
        if not bool(torch.isfinite(tensor).all()):
            raise CheckpointValidationError("Gate C F32 tensor is non-finite")
        host = tensor.contiguous().numpy()
    else:
        raise CheckpointValidationError(
            f"unsupported Gate C plain tensor dtype {tensor.dtype}"
        )
    value = jax.device_put(host, device)
    value.block_until_ready()
    return value


def _device_put_raw_fp8_bits(jax: Any, tensor: Any, device: object) -> Any:
    """Transfer exact E4M3FN encodings without host-side dequantization."""

    import torch

    if tensor.dtype != torch.float8_e4m3fn:
        raise CheckpointValidationError("raw Gate C FP8 tensor dtype drifted")
    if not bool(torch.isfinite(tensor.float()).all()):
        raise CheckpointValidationError("raw Gate C FP8 tensor is non-finite")
    host = tensor.contiguous().view(torch.uint8).numpy()
    value = jax.device_put(host, device)
    value.block_until_ready()
    return value


def _placement_destinations(
    placement: Mapping[str, Any],
) -> dict[int, Mapping[str, Any]]:
    by_slot = {
        destination["device_slot"]: destination
        for destination in placement["destinations"]
    }
    if set(by_slot) != {0, 1, 2, 3}:
        raise CheckpointValidationError(
            f"Gate C placement lacks four owners: {placement['source']['name']!r}"
        )
    return by_slot


def load_gate_c_checkpoint(
    artifact_dir: Path,
    expectation: GateCLoadExpectation,
    resolution: StageDeviceResolution,
    *,
    raw_fp8_names: frozenset[str] = frozenset(),
) -> LoadedGateCCheckpoint:
    """Direct-load and dequantize all 31 leaves without a host global concat."""

    manifest, layout = verify_gate_c_load_contract(artifact_dir, expectation)
    if resolution.stage_id != expectation.stage_id:
        raise CheckpointValidationError("resolved Gate C stage is not stage 0")
    if len(resolution.devices) != expectation.stage_size:
        raise CheckpointValidationError("resolved Gate C stage has the wrong size")
    plans = sorted(
        build_destination_file_plans(
            layout, validate_layout_contract=False
        ),
        key=lambda item: item.device_slot,
    )
    if (
        tuple(plan.device_slot for plan in plans) != (0, 1, 2, 3)
        or tuple(plan.device_id for plan in plans)
        != resolution.captured_device_ids
    ):
        raise CheckpointValidationError(
            "Gate C pack owners disagree with resolved physical devices"
        )

    import jax
    import numpy as np
    import torch
    from jax.sharding import Mesh
    from safetensors import safe_open

    mesh = Mesh(np.asarray(resolution.devices), ("stage",))
    placements = sorted(
        layout["placements"], key=lambda item: item["source"]["name"]
    )
    by_name = {
        placement["source"]["name"]: placement
        for placement in placements
    }
    unknown_raw_names = raw_fp8_names - {
        placement["source"]["name"]
        for placement in placements
        if placement["source"]["dtype"] == "F8_E4M3"
    }
    if unknown_raw_names:
        raise CheckpointValidationError(
            f"requested raw Gate C FP8 tensors are unavailable: {sorted(unknown_raw_names)}"
        )
    weights: dict[str, Any] = {}
    device_before = [_memory_stats(device) for device in resolution.devices]
    host_rss_before = _rss_peak_bytes()
    transfers = 0
    dequantizations = 0
    raw_leaf_count = 0
    try:
        with ExitStack() as stack:
            handles = [
                stack.enter_context(
                    safe_open(
                        Path(artifact_dir) / plan.filename,
                        framework="pt",
                        device="cpu",
                    )
                )
                for plan in plans
            ]
            consumed_scales: set[str] = set()
            for placement in placements:
                source = placement["source"]
                name = source["name"]
                if name in consumed_scales:
                    continue
                _placement_destinations(placement)
                local_arrays = []
                references = []
                if source["dtype"] == "F8_E4M3":
                    scale_name = f"{name}_scale_inv"
                    scale_placement = by_name.get(scale_name)
                    if scale_placement is None:
                        raise CheckpointValidationError(
                            f"Gate C FP8 weight lacks scale placement: {name!r}"
                        )
                    _placement_destinations(scale_placement)
                    if {
                        destination["device_slot"]
                        for destination in scale_placement["destinations"]
                    } != {
                        destination["device_slot"]
                        for destination in placement["destinations"]
                    }:
                        raise CheckpointValidationError(
                            f"Gate C FP8 scale ownership drifted: {name!r}"
                        )
                    if name in raw_fp8_names:
                        scale_arrays = []
                        for handle, device in zip(
                            handles, resolution.devices, strict=True
                        ):
                            weight = handle.get_tensor(name)
                            scale = handle.get_tensor(scale_name)
                            _validate_fp8_pair(weight, scale)
                            local_arrays.append(
                                _device_put_raw_fp8_bits(jax, weight, device)
                            )
                            scale_arrays.append(
                                _device_put_plain(jax, scale, device)
                            )
                            transfers += 2
                            del weight, scale
                        weights[name] = _assemble_global(
                            jax,
                            tuple(source["shape"]),
                            _sharding_for_placement(mesh, placement),
                            local_arrays,
                        )
                        scale_source = scale_placement["source"]
                        weights[scale_name] = _assemble_global(
                            jax,
                            tuple(scale_source["shape"]),
                            _sharding_for_placement(mesh, scale_placement),
                            scale_arrays,
                        )
                        consumed_scales.add(scale_name)
                        raw_leaf_count += 2
                        del local_arrays, scale_arrays, references
                        gc.collect()
                        continue
                    for handle, device in zip(
                        handles, resolution.devices, strict=True
                    ):
                        weight = handle.get_tensor(name)
                        scale = handle.get_tensor(scale_name)
                        _validate_fp8_pair(weight, scale)
                        local_arrays.append(
                            _device_put_dequantized(
                                jax,
                                weight,
                                scale,
                                device,
                                block_shape=FP8_BLOCK_SHAPE,
                                expert_chunk_size=1,
                            )
                        )
                        transfers += 2
                        dequantizations += 1
                        del weight, scale
                    consumed_scales.add(scale_name)
                    raw_leaf_count += 2
                elif source["dtype"] in ("BF16", "F32"):
                    for handle, device in zip(
                        handles, resolution.devices, strict=True
                    ):
                        tensor = handle.get_tensor(name).contiguous()
                        if placement["layout"] == "replicated":
                            references.append(tensor)
                        local_arrays.append(_device_put_plain(jax, tensor, device))
                        transfers += 1
                        del tensor
                    if references and any(
                        not torch.equal(references[0], other)
                        for other in references[1:]
                    ):
                        raise CheckpointValidationError(
                            f"Gate C replicated tensor differs by owner: {name!r}"
                        )
                    raw_leaf_count += 1
                else:
                    raise CheckpointValidationError(
                        f"unsupported Gate C logical dtype {source['dtype']!r}"
                    )
                sharding = _sharding_for_placement(mesh, placement)
                weights[name] = _assemble_global(
                    jax,
                    tuple(source["shape"]),
                    sharding,
                    local_arrays,
                )
                del local_arrays, references
                gc.collect()
        if raw_leaf_count != len(layout["source"]["tensors"]):
            raise CheckpointValidationError(
                "Gate C loaded raw leaf count does not reconcile"
            )
        expected_weight_names = {
            placement["source"]["name"]
            for placement in placements
            if placement["source"]["dtype"] != "F32"
            or not placement["source"]["name"].endswith("_scale_inv")
        } - consumed_scales
        expected_weight_names.update(
            f"{name}_scale_inv" for name in raw_fp8_names
        )
        if set(weights) != expected_weight_names:
            raise CheckpointValidationError(
                "Gate C semantic weight set does not reconcile"
            )
        state_manifest: dict[str, Any] = {
            "artifact_kind": "greenfield_loaded_gate_c_checkpoint",
            "captured_device_ids": list(resolution.captured_device_ids),
            "layout_manifest_sha256": layout["manifest_sha256"],
            "oracle_manifest_sha256": layout["oracle_manifest_sha256"],
            "packed_manifest_sha256": manifest["manifest_sha256"],
            "packed_payload_bytes": layout["packed_payload_bytes"],
            "plan_id": layout["plan_id"],
            "loaded_final_shard_count": sum(
                len(placement["destinations"]) for placement in placements
            ),
            "semantic_weight_count": len(weights),
            "source_tensor_count": len(layout["source"]["tensors"]),
            "stage_id": resolution.stage_id,
        }
        state_manifest["manifest_sha256"] = sha256(
            _canonical_json(state_manifest).encode("utf-8")
        ).hexdigest()
        load_record = {
            "device_dequantizations": dequantizations,
            "device_memory_after": [
                _memory_stats(device) for device in resolution.devices
            ],
            "device_memory_before": device_before,
            "device_slot_count": expectation.stage_size,
            "host_fp8_dequantizations": 0,
            "host_global_concatenations": 0,
            "host_peak_rss_after_bytes": _rss_peak_bytes(),
            "host_peak_rss_before_bytes": host_rss_before,
            "loaded_payload_bytes": layout["packed_payload_bytes"],
            "packed_single_device_transfers": transfers,
            "runtime_checkpoint_reshards": 0,
            "state_manifest_sha256": state_manifest["manifest_sha256"],
        }
        return LoadedGateCCheckpoint(
            mesh=mesh,
            weights=weights,
            manifest=manifest,
            layout=layout,
            state_manifest=state_manifest,
            load_record=load_record,
        )
    except BaseException:
        for value in weights.values():
            value.delete()
        raise
