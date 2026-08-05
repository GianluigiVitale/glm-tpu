"""Plan-local final ownership recipes for every GLM checkpoint leaf.

Recipes describe the packed device-slot layout without reading payloads.  A
source tensor is either replicated inside one topology-local stage, sharded
along one explicit dimension, or assigned whole by routed-expert identity.
There is no implicit/global fallback.  Layer 78 is the optional MTP block and
is explicitly separated from the base decoder load set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import re

from ..errors import PartitioningValidationError
from ..types import ModelGeometry
from .source_inventory import SourceInventory, SourceTensor


BASE_LOAD_SET = "base_decoder"
MTP_LOAD_SET = "mtp_optional"
_EXPERT = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$"
)
_SHARED = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.shared_experts\."
    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$"
)
_DENSE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\."
    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$"
)


@dataclass(frozen=True, slots=True)
class DestinationShard:
    """One exact destination slice in a stage-local physical device slot."""

    device_slot: int
    shape: tuple[int, ...]
    byte_count: int
    axis: int | None = None
    axis_start: int | None = None
    axis_end_exclusive: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        if self.device_slot < 0:
            raise PartitioningValidationError("device_slot must be non-negative")
        if self.byte_count < 0:
            raise PartitioningValidationError("shard byte_count must be non-negative")
        if self.axis is None:
            if self.axis_start is not None or self.axis_end_exclusive is not None:
                raise PartitioningValidationError(
                    "unsharded destination cannot declare an axis interval"
                )
        else:
            if not 0 <= self.axis < len(self.shape):
                raise PartitioningValidationError("destination shard axis is invalid")
            if (
                self.axis_start is None
                or self.axis_end_exclusive is None
                or self.axis_start < 0
                or self.axis_end_exclusive <= self.axis_start
            ):
                raise PartitioningValidationError(
                    "sharded destination requires a positive axis interval"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "axis_end_exclusive": self.axis_end_exclusive,
            "axis_start": self.axis_start,
            "byte_count": self.byte_count,
            "device_slot": self.device_slot,
            "shape": list(self.shape),
        }


@dataclass(frozen=True, slots=True)
class PlacementRecipe:
    """Complete final-layout recipe for one source tensor."""

    source_name: str
    source_byte_count: int
    source_dtype: str
    layer_id: int | None
    load_set: str
    layout: str
    value_class: str
    shards: tuple[DestinationShard, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "shards", tuple(sorted(self.shards, key=lambda item: item.device_slot))
        )
        if not self.source_name or self.source_byte_count < 0:
            raise PartitioningValidationError("placement source identity is invalid")
        if self.load_set not in (BASE_LOAD_SET, MTP_LOAD_SET):
            raise PartitioningValidationError(
                f"unknown placement load set {self.load_set!r}"
            )
        if self.layout not in (
            "replicated",
            "axis_sharded",
            "expert_identity",
        ):
            raise PartitioningValidationError(
                f"unknown placement layout {self.layout!r}"
            )
        if self.value_class not in ("fp8_scale", "parameter"):
            raise PartitioningValidationError(
                f"unknown placement value class {self.value_class!r}"
            )
        if not self.shards:
            raise PartitioningValidationError("placement must have destinations")
        slots = [shard.device_slot for shard in self.shards]
        if len(slots) != len(set(slots)):
            raise PartitioningValidationError(
                "placement repeats a destination device slot"
            )
        if self.layout in ("axis_sharded", "expert_identity"):
            if sum(shard.byte_count for shard in self.shards) != self.source_byte_count:
                raise PartitioningValidationError(
                    f"non-replicated placement {self.source_name!r} does not "
                    "reconcile source bytes"
                )
        elif any(shard.byte_count != self.source_byte_count for shard in self.shards):
            raise PartitioningValidationError(
                f"replicated placement {self.source_name!r} has a partial replica"
            )

    @property
    def packed_byte_count(self) -> int:
        return sum(shard.byte_count for shard in self.shards)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "layout": self.layout,
            "load_set": self.load_set,
            "packed_byte_count": self.packed_byte_count,
            "shards": [shard.to_dict() for shard in self.shards],
            "source_byte_count": self.source_byte_count,
            "source_dtype": self.source_dtype,
            "source_name": self.source_name,
            "value_class": self.value_class,
        }


@dataclass(frozen=True, slots=True)
class PlacementLedger:
    """Every inventory leaf mapped exactly once to a final local layout."""

    model_id: str
    source_inventory_sha256: str
    local_parallel_size: int
    recipes: tuple[PlacementRecipe, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "recipes", tuple(sorted(self.recipes, key=lambda item: item.source_name))
        )
        if self.local_parallel_size <= 0:
            raise PartitioningValidationError(
                "local_parallel_size must be positive"
            )
        names = [recipe.source_name for recipe in self.recipes]
        if not names or len(names) != len(set(names)):
            raise PartitioningValidationError(
                "placement ledger source names must be non-empty and unique"
            )
        for recipe in self.recipes:
            if any(
                shard.device_slot >= self.local_parallel_size
                for shard in recipe.shards
            ):
                raise PartitioningValidationError(
                    f"placement {recipe.source_name!r} references an invalid slot"
                )
            if recipe.layout in ("replicated", "axis_sharded") and len(
                recipe.shards
            ) != self.local_parallel_size:
                raise PartitioningValidationError(
                    f"placement {recipe.source_name!r} must cover every local slot"
                )

    @property
    def source_payload_bytes(self) -> int:
        return sum(recipe.source_byte_count for recipe in self.recipes)

    @property
    def packed_payload_bytes(self) -> int:
        return sum(recipe.packed_byte_count for recipe in self.recipes)

    def packed_bytes(self, *, load_set: str | None = None) -> int:
        return sum(
            recipe.packed_byte_count
            for recipe in self.recipes
            if load_set is None or recipe.load_set == load_set
        )

    def slot_bytes(
        self,
        *,
        layer_ids: Iterable[int] | None = None,
        include_non_layer: bool = False,
        load_set: str = BASE_LOAD_SET,
        value_class: str | None = None,
    ) -> tuple[int, ...]:
        selected_layers = None if layer_ids is None else set(layer_ids)
        totals = [0] * self.local_parallel_size
        for recipe in self.recipes:
            if recipe.load_set != load_set:
                continue
            if recipe.layer_id is None:
                if not include_non_layer:
                    continue
            elif selected_layers is not None and recipe.layer_id not in selected_layers:
                continue
            if value_class is not None and recipe.value_class != value_class:
                continue
            for shard in recipe.shards:
                totals[shard.device_slot] += shard.byte_count
        return tuple(totals)


def _replicated(tensor: SourceTensor, local_size: int) -> tuple[DestinationShard, ...]:
    return tuple(
        DestinationShard(
            device_slot=slot,
            shape=tensor.shape,
            byte_count=tensor.byte_count,
        )
        for slot in range(local_size)
    )


def _axis_sharded(
    tensor: SourceTensor, local_size: int, axis: int
) -> tuple[DestinationShard, ...]:
    if not 0 <= axis < len(tensor.shape):
        raise PartitioningValidationError(
            f"tensor {tensor.name!r} cannot shard missing axis {axis}"
        )
    dimension = tensor.shape[axis]
    if dimension % local_size:
        raise PartitioningValidationError(
            f"tensor {tensor.name!r} dimension {dimension} on axis {axis} "
            f"does not divide over {local_size} slots"
        )
    width = dimension // local_size
    if tensor.byte_count % local_size:
        raise PartitioningValidationError(
            f"tensor {tensor.name!r} bytes do not divide over local slots"
        )
    shard_shape = list(tensor.shape)
    shard_shape[axis] = width
    return tuple(
        DestinationShard(
            device_slot=slot,
            shape=tuple(shard_shape),
            byte_count=tensor.byte_count // local_size,
            axis=axis,
            axis_start=slot * width,
            axis_end_exclusive=(slot + 1) * width,
        )
        for slot in range(local_size)
    )


def _expert_identity(
    tensor: SourceTensor,
    *,
    local_size: int,
    expert_id: int,
    num_experts: int,
) -> tuple[DestinationShard, ...]:
    if not 0 <= expert_id < num_experts or num_experts % local_size:
        raise PartitioningValidationError(
            f"invalid expert identity layout for tensor {tensor.name!r}"
        )
    slot = expert_id // (num_experts // local_size)
    return (
        DestinationShard(
            device_slot=slot,
            shape=tensor.shape,
            byte_count=tensor.byte_count,
        ),
    )


def placement_recipe(
    tensor: SourceTensor,
    geometry: ModelGeometry,
    local_size: int,
) -> PlacementRecipe:
    layer_id = tensor.layer_id
    if layer_id is not None and layer_id > geometry.num_layers:
        raise PartitioningValidationError(
            f"checkpoint tensor {tensor.name!r} is beyond base/MTP layer range"
        )
    load_set = (
        MTP_LOAD_SET if layer_id == geometry.num_layers else BASE_LOAD_SET
    )
    value_class = "fp8_scale" if tensor.is_fp8_scale else "parameter"
    layout: str
    shards: tuple[DestinationShard, ...]

    expert = _EXPERT.fullmatch(tensor.name)
    if expert is not None:
        parsed_layer, expert_id = int(expert.group(1)), int(expert.group(2))
        if parsed_layer != layer_id:
            raise PartitioningValidationError("expert layer parser disagreement")
        layout = "expert_identity"
        shards = _expert_identity(
            tensor,
            local_size=local_size,
            expert_id=expert_id,
            num_experts=geometry.num_routed_experts,
        )
    else:
        shared = _SHARED.fullmatch(tensor.name)
        dense = _DENSE.fullmatch(tensor.name)
        if shared is not None or dense is not None:
            match = shared if shared is not None else dense
            assert match is not None
            projection = match.group(2)
            axis = 1 if projection == "down_proj" else 0
            layout = "axis_sharded"
            shards = _axis_sharded(tensor, local_size, axis)
        elif tensor.name in ("model.embed_tokens.weight", "lm_head.weight"):
            layout = "axis_sharded"
            shards = _axis_sharded(tensor, local_size, 0)
        elif tensor.name == "model.norm.weight":
            layout = "replicated"
            shards = _replicated(tensor, local_size)
        elif layer_id is None:
            raise PartitioningValidationError(
                f"no non-layer ownership rule for {tensor.name!r}"
            )
        else:
            suffix = tensor.name.split(f"model.layers.{layer_id}.", 1)[1]
            axis_zero_prefixes = (
                "self_attn.q_b_proj.",
                "self_attn.kv_b_proj.",
                "self_attn.indexer.wq_b.",
                "self_attn.indexer.weights_proj.",
            )
            if suffix.startswith(axis_zero_prefixes):
                layout = "axis_sharded"
                shards = _axis_sharded(tensor, local_size, 0)
            elif suffix.startswith("self_attn.o_proj.") or suffix == "eh_proj.weight":
                layout = "axis_sharded"
                shards = _axis_sharded(tensor, local_size, 1)
            elif suffix in (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_a_layernorm.weight",
                "self_attn.kv_a_layernorm.weight",
                "self_attn.indexer.k_norm.bias",
                "self_attn.indexer.k_norm.weight",
                "mlp.gate.weight",
                "mlp.gate.e_score_correction_bias",
                "enorm.weight",
                "hnorm.weight",
                "shared_head.norm.weight",
            ) or suffix.startswith(
                (
                    "self_attn.q_a_proj.",
                    "self_attn.kv_a_proj_with_mqa.",
                    "self_attn.indexer.wk.",
                )
            ):
                layout = "replicated"
                shards = _replicated(tensor, local_size)
            else:
                raise PartitioningValidationError(
                    f"no layer ownership rule for {tensor.name!r}"
                )

    return PlacementRecipe(
        source_name=tensor.name,
        source_byte_count=tensor.byte_count,
        source_dtype=tensor.dtype,
        layer_id=layer_id,
        load_set=load_set,
        layout=layout,
        value_class=value_class,
        shards=shards,
    )


def expected_glm_source_names(geometry: ModelGeometry) -> frozenset[str]:
    """Return the exact base-plus-one-MTP leaf set implied by the geometry."""

    names = {"lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"}
    for layer in range(geometry.num_layers + 1):
        prefix = f"model.layers.{layer}"
        names.update(
            {
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.self_attn.kv_a_layernorm.weight",
                f"{prefix}.self_attn.q_a_layernorm.weight",
            }
        )
        for projection in (
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "o_proj",
            "q_a_proj",
            "q_b_proj",
        ):
            base = f"{prefix}.self_attn.{projection}.weight"
            names.update((base, f"{base}_scale_inv"))
        indexer_full = (
            layer == geometry.num_layers
            or geometry.indexer_types[layer] == "full"
        )
        if indexer_full:
            names.update(
                {
                    f"{prefix}.self_attn.indexer.k_norm.bias",
                    f"{prefix}.self_attn.indexer.k_norm.weight",
                    f"{prefix}.self_attn.indexer.weights_proj.weight",
                    f"{prefix}.self_attn.indexer.wk.weight",
                    f"{prefix}.self_attn.indexer.wk.weight_scale_inv",
                    f"{prefix}.self_attn.indexer.wq_b.weight",
                    f"{prefix}.self_attn.indexer.wq_b.weight_scale_inv",
                }
            )
        sparse = (
            layer == geometry.num_layers
            or geometry.mlp_layer_types[layer] == "sparse"
        )
        if sparse:
            for expert in range(geometry.num_routed_experts):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    base = f"{prefix}.mlp.experts.{expert}.{projection}.weight"
                    names.update((base, f"{base}_scale_inv"))
            for projection in ("gate_proj", "up_proj", "down_proj"):
                base = f"{prefix}.mlp.shared_experts.{projection}.weight"
                names.update((base, f"{base}_scale_inv"))
            names.update(
                {
                    f"{prefix}.mlp.gate.e_score_correction_bias",
                    f"{prefix}.mlp.gate.weight",
                }
            )
        else:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                base = f"{prefix}.mlp.{projection}.weight"
                names.update((base, f"{base}_scale_inv"))
        if layer == geometry.num_layers:
            names.update(
                {
                    f"{prefix}.eh_proj.weight",
                    f"{prefix}.enorm.weight",
                    f"{prefix}.hnorm.weight",
                    f"{prefix}.shared_head.norm.weight",
                }
            )
    return frozenset(names)


def build_placement_ledger(
    inventory: SourceInventory,
    geometry: ModelGeometry,
    *,
    local_parallel_size: int,
    require_complete_model: bool = True,
) -> PlacementLedger:
    """Map every source leaf or fail; validate GLM layer/indexer schedules."""

    if inventory.model_id != geometry.model_id:
        raise PartitioningValidationError(
            "source inventory and model geometry identify different models"
        )
    if local_parallel_size not in (2, 4):
        raise PartitioningValidationError(
            "initial checkpoint ownership supports only PP8 LP4 or PP16 LP2"
        )
    source_names = {tensor.name for tensor in inventory.tensors}
    if require_complete_model:
        expected_names = expected_glm_source_names(geometry)
        if source_names != expected_names:
            raise PartitioningValidationError(
                "checkpoint leaf set disagrees with exact GLM geometry: "
                f"missing={sorted(expected_names - source_names)[:5]}, "
                f"unexpected={sorted(source_names - expected_names)[:5]}"
            )
    recipes = tuple(
        placement_recipe(tensor, geometry, local_parallel_size)
        for tensor in inventory.tensors
    )
    if sum(recipe.source_byte_count for recipe in recipes) != inventory.payload_bytes:
        raise PartitioningValidationError(
            "placement source bytes do not reconcile the source inventory"
        )
    by_layer: dict[int, list[str]] = {}
    for tensor in inventory.tensors:
        if tensor.layer_id is not None:
            by_layer.setdefault(tensor.layer_id, []).append(tensor.name)
    expected_layers = set(range(geometry.num_layers + 1))
    if set(by_layer) != expected_layers:
        raise PartitioningValidationError(
            "checkpoint must contain every base layer plus the one MTP layer"
        )
    for layer in range(geometry.num_layers):
        names = by_layer[layer]
        has_experts = any(".mlp.experts." in name for name in names)
        if has_experts != (geometry.mlp_layer_types[layer] == "sparse"):
            raise PartitioningValidationError(
                f"layer {layer} checkpoint MLP kind disagrees with geometry"
            )
        has_indexer = any(".self_attn.indexer." in name for name in names)
        if has_indexer != (geometry.indexer_types[layer] == "full"):
            raise PartitioningValidationError(
                f"layer {layer} checkpoint indexer schedule disagrees with geometry"
            )
    for tensor in inventory.tensors:
        if not tensor.is_fp8_scale:
            continue
        weight_name = tensor.name.removesuffix("_scale_inv")
        if weight_name not in source_names:
            raise PartitioningValidationError(
                f"scale tensor {tensor.name!r} has no colocated source weight"
            )
    return PlacementLedger(
        model_id=geometry.model_id,
        source_inventory_sha256=inventory.inventory_sha256,
        local_parallel_size=local_parallel_size,
        recipes=recipes,
    )
