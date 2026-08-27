"""Verify and directly load the default-off WS32 StrategyND dense overlay."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..errors import CheckpointValidationError


_SUFFIXES = (
    "merged_gate_up.weight_bits_in_out",
    "merged_gate_up.scale_inv_in_out",
    "down.weight_bits_in_out",
    "down.scale_inv_in_out",
)
_LOCAL_CONTRACT = (
    ((4, 6144, 768), np.dtype(np.uint8), (32, 6144, 768), ("expert", None, None)),
    ((4, 48, 768), np.dtype(np.float32), (32, 48, 768), ("expert", None, None)),
    ((4, 384, 1536), np.dtype(np.uint8), (32, 384, 6144), ("expert", None, "feature")),
    ((4, 3, 1536), np.dtype(np.float32), (32, 3, 6144), ("expert", None, "feature")),
)


def strategy_nd_dense_tensor_names(layer_id: int) -> tuple[str, ...]:
    if layer_id not in (0, 1, 2):
        raise CheckpointValidationError(
            "WS32 StrategyND overlay layer must be one of 0, 1, 2"
        )
    prefix = f"model.layers.{layer_id}.mlp.strategy_nd"
    return tuple(f"{prefix}.{suffix}" for suffix in _SUFFIXES)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _require_digest(value: str, *, field: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise CheckpointValidationError(f"{field} must be one lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class Ws32StrategyNdDenseOverlay:
    root: Path
    manifest: Mapping[str, Any]
    records: Mapping[tuple[int, int, int], Mapping[str, Any]]
    manifest_file_sha256: str
    success_file_sha256: str


@dataclass(frozen=True, slots=True)
class LoadedWs32StrategyNdDenseOverlay:
    arrays: Mapping[str, Any]
    local_records: tuple[Mapping[str, Any], ...]


def verify_ws32_strategy_nd_dense_overlay(
    root: Path,
    *,
    expected_manifest_sha256: str,
    expected_manifest_file_sha256: str,
    expected_success_file_sha256: str,
) -> Ws32StrategyNdDenseOverlay:
    """Bind terminal identities and all 96 exact final-owner records."""

    expected_manifest_sha256 = _require_digest(
        expected_manifest_sha256, field="expected_manifest_sha256"
    )
    expected_manifest_file_sha256 = _require_digest(
        expected_manifest_file_sha256,
        field="expected_manifest_file_sha256",
    )
    expected_success_file_sha256 = _require_digest(
        expected_success_file_sha256,
        field="expected_success_file_sha256",
    )
    manifest_path = root / "manifest.json"
    success_path = root / "SUCCESS"
    if not manifest_path.is_file() or not success_path.is_file():
        raise CheckpointValidationError(
            "WS32 StrategyND dense overlay is not terminal"
        )
    manifest_file_sha = _sha256_file(manifest_path)
    success_file_sha = _sha256_file(success_path)
    if manifest_file_sha != expected_manifest_file_sha256 or (
        success_file_sha != expected_success_file_sha256
    ):
        raise CheckpointValidationError(
            "WS32 StrategyND dense overlay terminal identity drifted"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    without_hash = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if (
        manifest.get("manifest_sha256")
        != sha256(_canonical(without_hash)).hexdigest()
        or manifest.get("manifest_sha256") != expected_manifest_sha256
    ):
        raise CheckpointValidationError(
            "WS32 StrategyND dense overlay manifest identity drifted"
        )
    success = json.loads(success_path.read_text(encoding="utf-8"))
    without_success = {
        key: value for key, value in success.items() if key != "success_sha256"
    }
    if (
        success.get("success_sha256")
        != sha256(_canonical(without_success)).hexdigest()
        or success.get("manifest_sha256") != expected_manifest_sha256
    ):
        raise CheckpointValidationError(
            "WS32 StrategyND dense overlay SUCCESS drifted"
        )
    if (
        manifest.get("artifact_kind")
        != "greenfield_ws32_strategy_nd_dense_overlay"
        or manifest.get("plan_id") != "WS32_2D"
        or manifest.get("format_version") != 1
        or manifest.get("dense_layer_ids") != [0, 1, 2]
        or manifest.get("file_count") != 96
        or len(manifest.get("files", ())) != 96
    ):
        raise CheckpointValidationError(
            "WS32 StrategyND dense overlay contract drifted"
        )
    records = {
        (
            int(item["layer_id"]),
            int(item["expert_coordinate"]),
            int(item["feature_coordinate"]),
        ): item
        for item in manifest["files"]
    }
    expected_owners = {
        (layer, expert, feature)
        for layer in range(3)
        for expert in range(8)
        for feature in range(4)
    }
    if set(records) != expected_owners:
        raise CheckpointValidationError(
            "WS32 StrategyND dense overlay owner coverage drifted"
        )
    for (layer, expert, _), record in records.items():
        if record.get("model_ranks") != list(range(expert * 4, expert * 4 + 4)):
            raise CheckpointValidationError(
                "WS32 StrategyND dense overlay rank ownership drifted"
            )
        names = strategy_nd_dense_tensor_names(layer)
        tensors = record.get("tensors", {})
        if set(tensors) != set(names):
            raise CheckpointValidationError(
                "WS32 StrategyND dense overlay tensor names drifted"
            )
        for name, (shape, dtype, _, _) in zip(names, _LOCAL_CONTRACT, strict=True):
            tensor = tensors[name]
            expected_dtype = "U8" if dtype == np.dtype(np.uint8) else "F32"
            if tensor.get("shape") != list(shape) or (
                tensor.get("dtype") != expected_dtype
            ):
                raise CheckpointValidationError(
                    "WS32 StrategyND dense overlay tensor geometry drifted"
                )
    for layer in range(3):
        for expert in range(8):
            group = [records[(layer, expert, feature)] for feature in range(4)]
            names = strategy_nd_dense_tensor_names(layer)
            for name in names[:2]:
                if len({item["tensors"][name]["sha256"] for item in group}) != 1:
                    raise CheckpointValidationError(
                        "WS32 StrategyND dense feature replicas drifted"
                    )
    return Ws32StrategyNdDenseOverlay(
        root=root,
        manifest=manifest,
        records=records,
        manifest_file_sha256=manifest_file_sha,
        success_file_sha256=success_file_sha,
    )


def load_ws32_strategy_nd_dense_overlay(
    overlay: Ws32StrategyNdDenseOverlay,
    *,
    mesh: Any,
    physical_mesh: Any,
) -> LoadedWs32StrategyNdDenseOverlay:
    """Read only addressable final-owner files and form 12 global arrays."""

    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P, SingleDeviceSharding
    from safetensors import safe_open

    if tuple(mesh.axis_names) != ("expert", "feature"):
        raise CheckpointValidationError(
            "WS32 StrategyND dense overlay mesh axes drifted"
        )
    coordinates = {
        int(device_id): (expert, feature)
        for expert, row in enumerate(physical_mesh.device_ids)
        for feature, device_id in enumerate(row)
    }
    local_values: dict[tuple[int, int], dict[str, Any]] = {}
    local_records: list[Mapping[str, Any]] = []
    for device in jax.local_devices():
        try:
            expert, feature = coordinates[int(device.id)]
        except KeyError as error:
            raise CheckpointValidationError(
                "WS32 StrategyND local device is absent from physical mesh"
            ) from error
        for layer in range(3):
            record = overlay.records[(layer, expert, feature)]
            path = overlay.root / str(record["filename"])
            if not path.is_file() or _sha256_file(path) != record["sha256"]:
                raise CheckpointValidationError(
                    "WS32 StrategyND dense local file hash drifted"
                )
            names = strategy_nd_dense_tensor_names(layer)
            with safe_open(path, framework="np") as handle:
                if set(handle.keys()) != set(names):
                    raise CheckpointValidationError(
                        "WS32 StrategyND dense local file tensor set drifted"
                    )
                arrays = {
                    name: np.ascontiguousarray(handle.get_tensor(name))
                    for name in names
                }
            for name, (shape, dtype, _, _) in zip(
                names, _LOCAL_CONTRACT, strict=True
            ):
                value = arrays[name]
                tensor = record["tensors"][name]
                if value.shape != shape or value.dtype != dtype or (
                    _sha256_array(value) != tensor["sha256"]
                ):
                    raise CheckpointValidationError(
                        "WS32 StrategyND dense local tensor drifted"
                    )
            local_values[(layer, int(device.id))] = {
                name: jax.device_put(value, SingleDeviceSharding(device))
                for name, value in arrays.items()
            }
            local_records.append(
                {
                    "device_id": int(device.id),
                    "expert_coordinate": expert,
                    "feature_coordinate": feature,
                    "file_sha256": record["sha256"],
                    "layer_id": layer,
                }
            )
    global_arrays: dict[str, Any] = {}
    for layer in range(3):
        names = strategy_nd_dense_tensor_names(layer)
        for name, (_, _, global_shape, spec) in zip(
            names, _LOCAL_CONTRACT, strict=True
        ):
            sharding = NamedSharding(mesh, P(*spec))
            devices = tuple(sharding.addressable_devices_indices_map(global_shape))
            shards = tuple(
                local_values[(layer, int(device.id))][name] for device in devices
            )
            expected_local_shape = sharding.shard_shape(global_shape)
            if any(tuple(shard.shape) != expected_local_shape for shard in shards):
                raise CheckpointValidationError(
                    "WS32 StrategyND dense JAX shard geometry drifted"
                )
            global_arrays[name] = jax.make_array_from_single_device_arrays(
                global_shape, sharding, shards
            )
    jax.block_until_ready(tuple(global_arrays.values()))
    return LoadedWs32StrategyNdDenseOverlay(
        arrays=global_arrays,
        local_records=tuple(
            sorted(
                local_records,
                key=lambda item: (int(item["device_id"]), int(item["layer_id"])),
            )
        ),
    )
