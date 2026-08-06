"""Fail-closed raw loader for the final-layout Pallas one-layer artifact."""

from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
import resource
from typing import Any, Mapping

from .one_layer_loader import (
    StageDeviceResolution,
    _assemble_global,
    _memory_stats,
    _torch_bfloat16_numpy,
    _torch_float8_bits_numpy,
    _validate_finite_float8_bits,
)
from .one_layer_pallas import inspect_pallas_one_layer_artifact


@dataclass(frozen=True, slots=True)
class PallasOneLayerLoadExpectation:
    """Exact derivative, source, plan, and code identities required to load."""

    manifest_sha256: str
    source_manifest_sha256: str
    code_hash: str
    source_revision: str
    topology_hash: str
    plan_group_hash: str
    plan_id: str = "PP8_LP4"
    model_id: str = "zai-org/GLM-5.2-FP8"
    layer: int = 3

    def __post_init__(self) -> None:
        for name in (
            "manifest_sha256",
            "source_manifest_sha256",
            "topology_hash",
            "plan_group_hash",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if len(self.code_hash) not in (40, 64) or any(
            character not in "0123456789abcdef"
            for character in self.code_hash
        ):
            raise ValueError("code_hash must be a lowercase Git object id")
        if not self.source_revision.strip():
            raise ValueError("source_revision must be non-empty")
        if self.plan_id not in ("PP8_LP4", "PP16_LP2"):
            raise ValueError("Pallas one-layer loader supports PP8 and PP16")
        if self.model_id != "zai-org/GLM-5.2-FP8" or self.layer != 3:
            raise ValueError("Pallas one-layer loader supports GLM layer 3")

    @property
    def stage_size(self) -> int:
        return {"PP8_LP4": 4, "PP16_LP2": 2}[self.plan_id]


@dataclass(slots=True)
class LoadedPallasOneLayer:
    """Raw final-owner arrays consumed directly by the composed MoE body."""

    mesh: Any
    expert_gate_bits: Any
    expert_gate_scale: Any
    expert_up_bits: Any
    expert_up_scale: Any
    expert_down_bits: Any
    expert_down_scale: Any
    shared_gate_bits: Any
    shared_gate_scale: Any
    shared_up_bits: Any
    shared_up_scale: Any
    shared_down_bits: Any
    shared_down_scale: Any
    router_weight: Any
    correction_bias: Any
    manifest: Mapping[str, Any]
    load_record: Mapping[str, Any]
    closed: bool = False

    @property
    def kernel_weights(self) -> tuple[Any, ...]:
        return (
            self.router_weight,
            self.correction_bias,
            self.expert_gate_bits,
            self.expert_gate_scale,
            self.expert_up_bits,
            self.expert_up_scale,
            self.expert_down_bits,
            self.expert_down_scale,
            self.shared_gate_bits,
            self.shared_gate_scale,
            self.shared_up_bits,
            self.shared_up_scale,
            self.shared_down_bits,
            self.shared_down_scale,
        )

    def close(self) -> None:
        if self.closed:
            return
        for value in self.kernel_weights:
            try:
                value.delete()
            except (AttributeError, RuntimeError):
                pass
        self.closed = True
        gc.collect()


def _rss_peak_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def verify_pallas_one_layer_load_contract(
    artifact_dir: Path,
    expectation: PallasOneLayerLoadExpectation,
) -> dict[str, Any]:
    """Hash all payloads before JAX import and bind every external identity."""

    manifest = inspect_pallas_one_layer_artifact(Path(artifact_dir))
    expected = {
        "code_hash": expectation.code_hash,
        "layer": expectation.layer,
        "manifest_sha256": expectation.manifest_sha256,
        "model_id": expectation.model_id,
        "plan_group_hash": expectation.plan_group_hash,
        "plan_id": expectation.plan_id,
        "source_manifest_sha256": expectation.source_manifest_sha256,
        "source_revision": expectation.source_revision,
        "topology_hash": expectation.topology_hash,
    }
    mismatches = {
        name: {"expected": value, "observed": manifest.get(name)}
        for name, value in expected.items()
        if manifest.get(name) != value
    }
    if manifest["geometry"].get("stage_size") != expectation.stage_size:
        mismatches["geometry.stage_size"] = {
            "expected": expectation.stage_size,
            "observed": manifest["geometry"].get("stage_size"),
        }
    if mismatches:
        raise ValueError(
            f"Pallas one-layer load contract mismatch: {mismatches}"
        )
    return manifest


def _put_raw_tensor(
    jax: Any,
    tensor: Any,
    name: str,
    device: Any,
    *,
    expert_chunk_size: int,
) -> tuple[Any, str]:
    import torch

    if name in {
        "expert_gate",
        "expert_up",
        "expert_down",
        "shared_gate",
        "shared_up",
        "shared_down",
    }:
        _validate_finite_float8_bits(tensor, expert_chunk_size)
        host = _torch_float8_bits_numpy(tensor)
        storage_dtype = "U8_E4M3FN_BITS"
    elif name.endswith("_scale") or name == "correction_bias":
        if tensor.dtype != torch.float32 or not bool(
            torch.isfinite(tensor).all()
        ):
            raise ValueError(f"raw Pallas tensor {name!r} must be finite FP32")
        host = tensor.contiguous().numpy()
        storage_dtype = "F32"
    elif name == "router_weight":
        if tensor.dtype != torch.bfloat16 or not bool(
            torch.isfinite(tensor.float()).all()
        ):
            raise ValueError("raw Pallas router must be finite BF16")
        host = _torch_bfloat16_numpy(tensor)
        storage_dtype = "BF16"
    else:
        raise ValueError(f"unexpected raw Pallas tensor {name!r}")
    array = jax.device_put(host, device)
    array.block_until_ready()
    if tuple(array.devices()) != (device,):
        array.delete()
        raise ValueError(f"raw Pallas tensor {name!r} missed its final owner")
    return array, storage_dtype


def load_pallas_one_layer(
    artifact_dir: Path,
    expectation: PallasOneLayerLoadExpectation,
    resolution: StageDeviceResolution,
    *,
    expert_chunk_size: int = 2,
) -> LoadedPallasOneLayer:
    """Load raw U8/FP32 final shards with no dequant, concat, or transpose."""

    if expert_chunk_size <= 0:
        raise ValueError("expert_chunk_size must be positive")
    manifest = verify_pallas_one_layer_load_contract(
        artifact_dir, expectation
    )
    if len(resolution.devices) != expectation.stage_size:
        raise ValueError("resolved stage size disagrees with Pallas artifact")

    import jax
    import numpy as np
    import torch
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from safetensors import safe_open

    geometry = manifest["geometry"]
    hidden = int(geometry["hidden_size"])
    intermediate = int(geometry["intermediate_size"])
    experts = int(geometry["num_experts"])
    stage_size = int(geometry["stage_size"])
    block_out, block_in = (
        int(value) for value in geometry["fp8_block_shape"]
    )
    mesh = Mesh(np.asarray(resolution.devices), ("expert",))
    expert_sharding = NamedSharding(mesh, P("expert"))
    shared_down_sharding = NamedSharding(mesh, P(None, "expert"))
    replicated_sharding = NamedSharding(mesh, P())
    local: dict[str, list[Any]] = {
        name: []
        for name in (
            "expert_gate",
            "expert_gate_scale",
            "expert_up",
            "expert_up_scale",
            "expert_down",
            "expert_down_scale",
            "shared_gate",
            "shared_gate_scale",
            "shared_up",
            "shared_up_scale",
            "shared_down",
            "shared_down_scale",
            "router_weight",
            "correction_bias",
        )
    }
    storage_dtypes: dict[str, str] = {}
    replicated_reference: dict[str, Any] = {}
    device_before = [_memory_stats(device) for device in resolution.devices]
    host_rss_before = _rss_peak_bytes()
    files = sorted(
        manifest["files"], key=lambda record: record["device_slot"]
    )
    global_arrays: dict[str, Any] = {}
    try:
        for slot, (file_record, device) in enumerate(
            zip(files, resolution.devices, strict=True)
        ):
            if int(file_record["device_slot"]) != slot:
                raise ValueError("Pallas packed slots are not contiguous")
            path = Path(artifact_dir) / str(file_record["filename"])
            with safe_open(path, framework="pt", device="cpu") as handle:
                for name in local:
                    tensor = handle.get_tensor(name).contiguous()
                    if name in ("router_weight", "correction_bias"):
                        reference = replicated_reference.setdefault(
                            name, tensor.clone()
                        )
                        if not torch.equal(reference, tensor):
                            raise ValueError(
                                f"replicated Pallas tensor {name!r} differs"
                            )
                    placed, storage_dtype = _put_raw_tensor(
                        jax,
                        tensor,
                        name,
                        device,
                        expert_chunk_size=expert_chunk_size,
                    )
                    local[name].append(placed)
                    storage_dtypes[name] = storage_dtype
                    del tensor, placed
            gc.collect()

        shapes = {
            "expert_gate": (experts, hidden, intermediate),
            "expert_gate_scale": (
                experts,
                intermediate // block_out,
                hidden // block_in,
            ),
            "expert_up": (experts, hidden, intermediate),
            "expert_up_scale": (
                experts,
                intermediate // block_out,
                hidden // block_in,
            ),
            "expert_down": (experts, intermediate, hidden),
            "expert_down_scale": (
                experts,
                hidden // block_out,
                intermediate // block_in,
            ),
            "shared_gate": (intermediate, hidden),
            "shared_gate_scale": (
                intermediate // block_out,
                hidden // block_in,
            ),
            "shared_up": (intermediate, hidden),
            "shared_up_scale": (
                intermediate // block_out,
                hidden // block_in,
            ),
            "shared_down": (hidden, intermediate),
            "shared_down_scale": (
                hidden // block_out,
                intermediate // block_in,
            ),
            "router_weight": (experts, hidden),
            "correction_bias": (experts,),
        }
        sharding = {
            name: (
                replicated_sharding
                if name in ("router_weight", "correction_bias")
                else shared_down_sharding
                if name in ("shared_down", "shared_down_scale")
                else expert_sharding
            )
            for name in local
        }
        global_arrays = {
            name: _assemble_global(
                jax, shapes[name], sharding[name], local[name]
            )
            for name in local
        }
    except BaseException:
        for value in global_arrays.values():
            try:
                value.delete()
            except (AttributeError, RuntimeError):
                pass
        for values in local.values():
            for value in values:
                try:
                    value.delete()
                except (AttributeError, RuntimeError):
                    pass
        raise
    finally:
        replicated_reference.clear()
        gc.collect()

    device_after = [_memory_stats(device) for device in resolution.devices]
    load_record = {
        "captured_device_ids_in_slot_order": list(
            resolution.captured_device_ids
        ),
        "device_after": device_after,
        "device_before": device_before,
        "device_fp8_dequantizations": 0,
        "host_fp8_dequantizations": 0,
        "host_global_concatenations": 0,
        "host_rss_peak_after": _rss_peak_bytes(),
        "host_rss_peak_before": host_rss_before,
        "layout_sha256": manifest["layout"]["layout_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "packed_payload_byte_count": manifest[
            "packed_payload_byte_count"
        ],
        "packed_single_device_transfers": stage_size * len(local),
        "plan_id": expectation.plan_id,
        "runtime_routed_weight_transposes": 0,
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "storage_dtypes": storage_dtypes,
    }
    return LoadedPallasOneLayer(
        mesh=mesh,
        expert_gate_bits=global_arrays["expert_gate"],
        expert_gate_scale=global_arrays["expert_gate_scale"],
        expert_up_bits=global_arrays["expert_up"],
        expert_up_scale=global_arrays["expert_up_scale"],
        expert_down_bits=global_arrays["expert_down"],
        expert_down_scale=global_arrays["expert_down_scale"],
        shared_gate_bits=global_arrays["shared_gate"],
        shared_gate_scale=global_arrays["shared_gate_scale"],
        shared_up_bits=global_arrays["shared_up"],
        shared_up_scale=global_arrays["shared_up_scale"],
        shared_down_bits=global_arrays["shared_down"],
        shared_down_scale=global_arrays["shared_down_scale"],
        router_weight=global_arrays["router_weight"],
        correction_bias=global_arrays["correction_bias"],
        manifest=manifest,
        load_record=load_record,
    )
