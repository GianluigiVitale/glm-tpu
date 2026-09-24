"""Immutable contracts shared by every greenfield execution plan.

These types are deliberately independent of JAX.  Discovery code will turn
runtime JAX devices into :class:`PhysicalTopology`; compilation code will
consume an :class:`ExecutionPlan`.  Keeping the boundary pure makes the
configuration testable without initializing a TPU and gives every run a
stable, content-addressed plan fingerprint.

Python's process-randomized ``hash()`` is never provenance.  ``plan_hash`` and
``topology_hash`` are SHA-256 digests of canonical JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from hashlib import sha256
import json
from operator import mul
from typing import Any, Mapping, Sequence

from glm_tpu.exceptions import GeometryValidationError, TopologyValidationError


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fingerprint(value: Mapping[str, Any]) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: object, field: str, error: type[ValueError]) -> int:
    if not _is_int(value) or value <= 0:
        raise error(f"{field} must be a positive integer, got {value!r}")
    return value


def _nonnegative_int(
    value: object, field: str, error: type[ValueError]
) -> int:
    if not _is_int(value) or value < 0:
        raise error(f"{field} must be a non-negative integer, got {value!r}")
    return value


def _nonempty(value: object, field: str, error: type[ValueError]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error(f"{field} must be a non-empty string")
    return value


def _product(values: Sequence[int]) -> int:
    return reduce(mul, values, 1)


@dataclass(frozen=True, slots=True)
class ModelGeometry:
    """Exact compile-relevant GLM-5.2 model geometry."""

    model_id: str
    num_layers: int
    first_dense_layers: int
    hidden_size: int
    dense_intermediate_size: int
    num_routed_experts: int
    num_shared_experts: int
    routed_top_k: int
    moe_intermediate_size: int
    dsa_top_k: int
    dsa_indexer_heads: int
    dsa_indexer_head_dim: int
    index_share_group_size: int
    attention_heads: int
    kv_heads: int
    kv_lora_rank: int
    q_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    num_nextn_predict_layers: int
    max_position_embeddings: int
    vocab_size: int
    activation_dtype: str
    weight_storage_dtype: str
    fp8_block_shape: tuple[int, int]
    mlp_layer_types: tuple[str, ...]
    indexer_types: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fp8_block_shape", tuple(self.fp8_block_shape))
        object.__setattr__(self, "mlp_layer_types", tuple(self.mlp_layer_types))
        object.__setattr__(self, "indexer_types", tuple(self.indexer_types))

        for field in (
            "model_id",
            "activation_dtype",
            "weight_storage_dtype",
        ):
            _nonempty(getattr(self, field), field, GeometryValidationError)
        for field in (
            "num_layers",
            "hidden_size",
            "dense_intermediate_size",
            "num_routed_experts",
            "num_shared_experts",
            "routed_top_k",
            "moe_intermediate_size",
            "dsa_top_k",
            "dsa_indexer_heads",
            "dsa_indexer_head_dim",
            "index_share_group_size",
            "attention_heads",
            "kv_heads",
            "kv_lora_rank",
            "q_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "num_nextn_predict_layers",
            "max_position_embeddings",
            "vocab_size",
        ):
            _positive_int(getattr(self, field), field, GeometryValidationError)
        _nonnegative_int(
            self.first_dense_layers,
            "first_dense_layers",
            GeometryValidationError,
        )

        if self.first_dense_layers > self.num_layers:
            raise GeometryValidationError(
                "first_dense_layers cannot exceed num_layers"
            )
        if self.routed_top_k > self.num_routed_experts:
            raise GeometryValidationError(
                "routed_top_k cannot exceed num_routed_experts"
            )
        if self.dsa_top_k > self.max_position_embeddings:
            raise GeometryValidationError(
                "dsa_top_k cannot exceed max_position_embeddings"
            )
        if self.num_nextn_predict_layers != 1:
            raise GeometryValidationError(
                "the exact GLM-5.2 target requires one MTP layer"
            )
        if self.hidden_size % self.attention_heads:
            raise GeometryValidationError(
                "hidden_size must be divisible by attention_heads"
            )
        if self.attention_heads % self.kv_heads:
            raise GeometryValidationError(
                "attention_heads must be divisible by kv_heads"
            )
        if len(self.fp8_block_shape) != 2 or any(
            not _is_int(v) or v <= 0 for v in self.fp8_block_shape
        ):
            raise GeometryValidationError(
                "fp8_block_shape must contain exactly two positive integers"
            )
        if len(self.mlp_layer_types) != self.num_layers:
            raise GeometryValidationError(
                "mlp_layer_types must contain one entry per transformer layer"
            )
        if len(self.indexer_types) != self.num_layers:
            raise GeometryValidationError(
                "indexer_types must contain one entry per transformer layer"
            )
        if set(self.mlp_layer_types) - {"dense", "sparse"}:
            raise GeometryValidationError(
                "mlp_layer_types entries must be 'dense' or 'sparse'"
            )
        if set(self.indexer_types) - {"full", "shared"}:
            raise GeometryValidationError(
                "indexer_types entries must be 'full' or 'shared'"
            )
        expected_mlp = (
            ("dense",) * self.first_dense_layers
            + ("sparse",) * (self.num_layers - self.first_dense_layers)
        )
        if self.mlp_layer_types != expected_mlp:
            raise GeometryValidationError(
                "mlp_layer_types does not match first_dense_layers"
            )

    @classmethod
    def from_hf_config(cls, config: Mapping[str, Any]) -> "ModelGeometry":
        """Build the exact geometry from the checked-in HF configuration."""

        if config.get("model_type") != "glm_moe_dsa":
            raise GeometryValidationError(
                "expected model_type='glm_moe_dsa', got "
                f"{config.get('model_type')!r}"
            )
        quant = config.get("quantization_config")
        if not isinstance(quant, Mapping) or quant.get("quant_method") != "fp8":
            raise GeometryValidationError("the greenfield target requires FP8 weights")
        fmt = _nonempty(quant.get("fmt"), "quantization_config.fmt", GeometryValidationError)
        block_shape = quant.get("weight_block_size")
        if not isinstance(block_shape, Sequence) or isinstance(block_shape, str):
            raise GeometryValidationError(
                "quantization_config.weight_block_size must be a sequence"
            )
        try:
            return cls(
                model_id="zai-org/GLM-5.2-FP8",
                num_layers=config["num_hidden_layers"],
                first_dense_layers=config["first_k_dense_replace"],
                hidden_size=config["hidden_size"],
                dense_intermediate_size=config["intermediate_size"],
                num_routed_experts=config["n_routed_experts"],
                num_shared_experts=config["n_shared_experts"],
                routed_top_k=config["num_experts_per_tok"],
                moe_intermediate_size=config["moe_intermediate_size"],
                dsa_top_k=config["index_topk"],
                dsa_indexer_heads=config["index_n_heads"],
                dsa_indexer_head_dim=config["index_head_dim"],
                index_share_group_size=config["index_topk_freq"],
                attention_heads=config["num_attention_heads"],
                kv_heads=config["num_key_value_heads"],
                kv_lora_rank=config["kv_lora_rank"],
                q_lora_rank=config["q_lora_rank"],
                qk_nope_head_dim=config["qk_nope_head_dim"],
                qk_rope_head_dim=config["qk_rope_head_dim"],
                v_head_dim=config["v_head_dim"],
                num_nextn_predict_layers=config["num_nextn_predict_layers"],
                max_position_embeddings=config["max_position_embeddings"],
                vocab_size=config["vocab_size"],
                activation_dtype=config["dtype"],
                weight_storage_dtype=f"fp8:{fmt}",
                fp8_block_shape=tuple(block_shape),
                mlp_layer_types=tuple(config["mlp_layer_types"]),
                indexer_types=tuple(config["indexer_types"]),
            )
        except KeyError as exc:
            raise GeometryValidationError(
                f"missing required HF config field {exc.args[0]!r}"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "activation_dtype": self.activation_dtype,
            "attention_heads": self.attention_heads,
            "dense_intermediate_size": self.dense_intermediate_size,
            "dsa_indexer_head_dim": self.dsa_indexer_head_dim,
            "dsa_indexer_heads": self.dsa_indexer_heads,
            "dsa_top_k": self.dsa_top_k,
            "first_dense_layers": self.first_dense_layers,
            "fp8_block_shape": list(self.fp8_block_shape),
            "hidden_size": self.hidden_size,
            "index_share_group_size": self.index_share_group_size,
            "indexer_types": list(self.indexer_types),
            "kv_heads": self.kv_heads,
            "kv_lora_rank": self.kv_lora_rank,
            "max_position_embeddings": self.max_position_embeddings,
            "mlp_layer_types": list(self.mlp_layer_types),
            "model_id": self.model_id,
            "moe_intermediate_size": self.moe_intermediate_size,
            "num_layers": self.num_layers,
            "num_routed_experts": self.num_routed_experts,
            "num_shared_experts": self.num_shared_experts,
            "num_nextn_predict_layers": self.num_nextn_predict_layers,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "q_lora_rank": self.q_lora_rank,
            "routed_top_k": self.routed_top_k,
            "vocab_size": self.vocab_size,
            "v_head_dim": self.v_head_dim,
            "weight_storage_dtype": self.weight_storage_dtype,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelGeometry":
        return cls(**dict(value))

    @property
    def geometry_hash(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True, slots=True)
class PhysicalDevice:
    """Runtime-observed identity of one JAX-visible TPU chip."""

    device_id: int
    process_index: int
    local_device_id: int
    coordinates: tuple[int, ...]
    core_on_chip: int
    platform: str
    device_kind: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinates", tuple(self.coordinates))
        for field in (
            "device_id",
            "process_index",
            "local_device_id",
            "core_on_chip",
        ):
            _nonnegative_int(getattr(self, field), field, TopologyValidationError)
        if not self.coordinates or any(
            not _is_int(v) or v < 0 for v in self.coordinates
        ):
            raise TopologyValidationError(
                "coordinates must be a non-empty tuple of non-negative integers"
            )
        _nonempty(self.platform, "platform", TopologyValidationError)
        _nonempty(self.device_kind, "device_kind", TopologyValidationError)

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinates": list(self.coordinates),
            "core_on_chip": self.core_on_chip,
            "device_id": self.device_id,
            "device_kind": self.device_kind,
            "local_device_id": self.local_device_id,
            "platform": self.platform,
            "process_index": self.process_index,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PhysicalDevice":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class PhysicalTopology:
    """Canonical runtime device inventory; never inferred from JAX ordering."""

    slice_name: str
    topology_shape: tuple[int, ...]
    devices: tuple[PhysicalDevice, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "topology_shape", tuple(self.topology_shape))
        object.__setattr__(
            self,
            "devices",
            tuple(sorted(self.devices, key=lambda device: device.device_id)),
        )
        _nonempty(self.slice_name, "slice_name", TopologyValidationError)
        if not self.topology_shape or any(
            not _is_int(v) or v <= 0 for v in self.topology_shape
        ):
            raise TopologyValidationError(
                "topology_shape must contain positive integer dimensions"
            )
        if not self.devices:
            raise TopologyValidationError("devices must not be empty")
        if _product(self.topology_shape) != len(self.devices):
            raise TopologyValidationError(
                "topology_shape product must equal the number of devices"
            )
        dimensions = len(self.topology_shape)
        if any(len(device.coordinates) != dimensions for device in self.devices):
            raise TopologyValidationError(
                "every device coordinate must match topology dimensionality"
            )
        if any(
            coordinate >= self.topology_shape[axis]
            for device in self.devices
            for axis, coordinate in enumerate(device.coordinates)
        ):
            raise TopologyValidationError(
                "device coordinate lies outside topology_shape"
            )
        self._require_unique("device_id", [d.device_id for d in self.devices])
        self._require_unique("coordinates", [d.coordinates for d in self.devices])
        self._require_unique(
            "(process_index, local_device_id)",
            [(d.process_index, d.local_device_id) for d in self.devices],
        )
        platforms = {device.platform for device in self.devices}
        kinds = {device.device_kind for device in self.devices}
        if len(platforms) != 1 or len(kinds) != 1:
            raise TopologyValidationError(
                "all devices must report one platform and one device_kind"
            )
        local_ids: dict[int, list[int]] = {}
        for device in self.devices:
            local_ids.setdefault(device.process_index, []).append(
                device.local_device_id
            )
        for process, ids in local_ids.items():
            if sorted(ids) != list(range(len(ids))):
                raise TopologyValidationError(
                    f"process {process} local_device_id values must be contiguous from zero"
                )

    @staticmethod
    def _require_unique(field: str, values: Sequence[object]) -> None:
        if len(set(values)) != len(values):
            raise TopologyValidationError(f"duplicate physical device {field}")

    @property
    def process_indices(self) -> tuple[int, ...]:
        return tuple(sorted({device.process_index for device in self.devices}))

    @property
    def topology_hash(self) -> str:
        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "devices": [device.to_dict() for device in self.devices],
            "slice_name": self.slice_name,
            "topology_shape": list(self.topology_shape),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PhysicalTopology":
        return cls(
            slice_name=value["slice_name"],
            topology_shape=tuple(value["topology_shape"]),
            devices=tuple(
                PhysicalDevice.from_dict(device) for device in value["devices"]
            ),
        )
