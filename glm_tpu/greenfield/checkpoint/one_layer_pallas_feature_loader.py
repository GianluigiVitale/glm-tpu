"""Fail-closed direct loader for expert-feature Pallas layer shards."""

from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
import resource
from typing import Any, Mapping

from .one_layer_loader import StageDeviceResolution, _assemble_global, _memory_stats
from .one_layer_pallas_feature import (
    inspect_pallas_feature_one_layer_artifact,
)
from .one_layer_pallas_loader import LoadedPallasOneLayer, _put_raw_tensor


@dataclass(frozen=True, slots=True)
class PallasFeatureLoadExpectation:
    """Exact feature artifact and external identities required to load."""

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
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if len(self.code_hash) not in (40, 64) or any(
            character not in "0123456789abcdef"
            for character in self.code_hash
        ):
            raise ValueError("code_hash must be a lowercase Git object id")
        if not self.source_revision.strip():
            raise ValueError("source_revision must be non-empty")
        if self.plan_id not in ("PP8_LP4", "PP16_LP2"):
            raise ValueError("expert-feature loader requires PP8_LP4 or PP16_LP2")
        if self.model_id != "zai-org/GLM-5.2-FP8" or self.layer != 3:
            raise ValueError("expert-feature loader supports GLM layer 3")

    @property
    def stage_size(self) -> int:
        """Return the physical width required by the feature layout."""

        return {"PP8_LP4": 4, "PP16_LP2": 2}[self.plan_id]


def _rss_peak_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def verify_pallas_feature_load_contract(
    artifact_dir: Path,
    expectation: PallasFeatureLoadExpectation,
) -> dict[str, Any]:
    """Hash the derivative and bind all source/topology/code identities."""

    manifest = inspect_pallas_feature_one_layer_artifact(Path(artifact_dir))
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
        raise ValueError(f"Pallas feature load contract mismatch: {mismatches}")
    return manifest


def load_pallas_feature_one_layer(
    artifact_dir: Path,
    expectation: PallasFeatureLoadExpectation,
    resolution: StageDeviceResolution,
    *,
    expert_chunk_size: int = 2,
) -> LoadedPallasOneLayer:
    """Direct-load final LP2/LP4 feature owners without concat or transpose."""

    if expert_chunk_size <= 0:
        raise ValueError("expert_chunk_size must be positive")
    manifest = verify_pallas_feature_load_contract(
        artifact_dir, expectation
    )
    if len(resolution.devices) != expectation.stage_size:
        raise ValueError(
            "expert-feature resolution width disagrees with the plan"
        )

    import jax
    import numpy as np
    import torch
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from safetensors import safe_open

    geometry = manifest["geometry"]
    hidden = int(geometry["hidden_size"])
    intermediate = int(geometry["intermediate_size"])
    experts = int(geometry["num_experts"])
    block_out, block_in = (
        int(value) for value in geometry["fp8_block_shape"]
    )
    mesh = Mesh(np.asarray(resolution.devices), ("feature",))
    replicated = NamedSharding(mesh, P())
    feature_up = NamedSharding(mesh, P(None, None, "feature"))
    feature_up_scale = NamedSharding(mesh, P(None, "feature", None))
    feature_down = NamedSharding(mesh, P(None, "feature", None))
    feature_down_scale = NamedSharding(mesh, P(None, None, "feature"))
    shared_up = NamedSharding(mesh, P("feature", None))
    shared_down = NamedSharding(mesh, P(None, "feature"))
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
        for slot, (file, device) in enumerate(
            zip(files, resolution.devices, strict=True)
        ):
            if int(file["device_slot"]) != slot:
                raise ValueError("expert-feature slots are not contiguous")
            path = Path(artifact_dir) / str(file["filename"])
            with safe_open(path, framework="pt", device="cpu") as handle:
                for name in local:
                    tensor = handle.get_tensor(name).contiguous()
                    if name in ("router_weight", "correction_bias"):
                        reference = replicated_reference.setdefault(
                            name, tensor.clone()
                        )
                        if not torch.equal(reference, tensor):
                            raise ValueError(
                                f"replicated feature tensor {name!r} differs"
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
            "expert_gate": feature_up,
            "expert_gate_scale": feature_up_scale,
            "expert_up": feature_up,
            "expert_up_scale": feature_up_scale,
            "expert_down": feature_down,
            "expert_down_scale": feature_down_scale,
            "shared_gate": shared_up,
            "shared_gate_scale": shared_up,
            "shared_up": shared_up,
            "shared_up_scale": shared_up,
            "shared_down": shared_down,
            "shared_down_scale": shared_down,
            "router_weight": replicated,
            "correction_bias": replicated,
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
    load_record: Mapping[str, Any] = {
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
        "packed_single_device_transfers": expectation.stage_size * len(local),
        "plan_id": expectation.plan_id,
        "routed_layout": "expert_intermediate_shard",
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
