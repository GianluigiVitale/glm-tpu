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
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence
from importlib import resources
from pathlib import Path

from glm_tpu.exceptions import GeometryValidationError
from glm_tpu.utils.io_utils import read_bounded
from glm_tpu.utils.json_utils import _fingerprint


MODEL_ID = "zai-org/GLM-5.3"
REVISION = "aca966e4e02791568aa6a4ced368624b3d897f42"
# The source bucket URI and the tokenizer/model directory are site values
# (storage.source_uri, paths.model_path in the site file; glm_tpu.config.site).
# The pinned GLM-5.3 assets (config, generation config, tokenizer config, chat template) are
# package data: hf_config/ of this package, read with importlib.resources.
HF_CONFIG_PACKAGE = "glm_tpu.models.glm_moe_dsa"
TEMPLATE_PATH = Path(*HF_CONFIG_PACKAGE.split("."), "hf_config", "chat_template.jinja")  # in a source root
TEMPLATE_SHA = "3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c"
CONFIG_SHA = "3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69"
INDEX_SHA = "e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf"
GENERATION_SHA = "ac76b43d8683d3b930126870fc8be73d8679308fe752fa1f381096d8354f6a55"
TOKENIZER_FILES = {
    "tokenizer.json": "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
    "tokenizer_config.json": "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc",
}


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: object, field: str, error: type[ValueError]) -> int:
    if not _is_int(value) or value <= 0:
        raise error(f"{field} must be a positive integer, got {value!r}")
    return value


def _nonnegative_int(value: object, field: str, error: type[ValueError]) -> int:
    if not _is_int(value) or value < 0:
        raise error(f"{field} must be a non-negative integer, got {value!r}")
    return value


def _nonempty(value: object, field: str, error: type[ValueError]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error(f"{field} must be a non-empty string")
    return value


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
            raise GeometryValidationError("first_dense_layers cannot exceed num_layers")
        if self.routed_top_k > self.num_routed_experts:
            raise GeometryValidationError("routed_top_k cannot exceed num_routed_experts")
        if self.dsa_top_k > self.max_position_embeddings:
            raise GeometryValidationError("dsa_top_k cannot exceed max_position_embeddings")
        if self.num_nextn_predict_layers != 1:
            raise GeometryValidationError("the exact GLM-5.2 target requires one MTP layer")
        if self.hidden_size % self.attention_heads:
            raise GeometryValidationError("hidden_size must be divisible by attention_heads")
        if self.attention_heads % self.kv_heads:
            raise GeometryValidationError("attention_heads must be divisible by kv_heads")
        if len(self.fp8_block_shape) != 2 or any(not _is_int(v) or v <= 0 for v in self.fp8_block_shape):
            raise GeometryValidationError("fp8_block_shape must contain exactly two positive integers")
        if len(self.mlp_layer_types) != self.num_layers:
            raise GeometryValidationError("mlp_layer_types must contain one entry per transformer layer")
        if len(self.indexer_types) != self.num_layers:
            raise GeometryValidationError("indexer_types must contain one entry per transformer layer")
        if set(self.mlp_layer_types) - {"dense", "sparse"}:
            raise GeometryValidationError("mlp_layer_types entries must be 'dense' or 'sparse'")
        if set(self.indexer_types) - {"full", "shared"}:
            raise GeometryValidationError("indexer_types entries must be 'full' or 'shared'")
        expected_mlp = ("dense",) * self.first_dense_layers + ("sparse",) * (self.num_layers - self.first_dense_layers)
        if self.mlp_layer_types != expected_mlp:
            raise GeometryValidationError("mlp_layer_types does not match first_dense_layers")

    @classmethod
    def from_hf_config(cls, config: Mapping[str, Any]) -> "ModelGeometry":
        """Build the exact geometry from the checked-in HF configuration."""

        if config.get("model_type") != "glm_moe_dsa":
            raise GeometryValidationError(f"expected model_type='glm_moe_dsa', got {config.get('model_type')!r}")
        quant = config.get("quantization_config")
        if not isinstance(quant, Mapping) or quant.get("quant_method") != "fp8":
            raise GeometryValidationError("the greenfield target requires FP8 weights")
        fmt = _nonempty(quant.get("fmt"), "quantization_config.fmt", GeometryValidationError)
        block_shape = quant.get("weight_block_size")
        if not isinstance(block_shape, Sequence) or isinstance(block_shape, str):
            raise GeometryValidationError("quantization_config.weight_block_size must be a sequence")
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
            raise GeometryValidationError(f"missing required HF config field {exc.args[0]!r}") from exc

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


def hf_config(repo=None):
    """The directory of the pinned GLM-5.3 assets: the package data of the loaded ``glm_tpu``
    (importlib.resources), or the copy in the source root ``repo`` (a checkout or a staged source)."""
    if repo is not None:
        return Path(repo) / TEMPLATE_PATH.parent
    directory = resources.files(HF_CONFIG_PACKAGE).joinpath("hf_config")
    if not isinstance(directory, Path):
        raise ValueError("glm_tpu must be installed as files on disk to read its model assets")
    return directory


def verified_template(repo, tokenizer_root):
    """Verify local assets without downloads or importing any model/device code (``repo``: a source
    root holding the package, or None for the loaded package's data)."""
    assets = hf_config(repo)
    template = read_bounded(assets / TEMPLATE_PATH.name, 64 << 10)
    if sha256(template).hexdigest() != TEMPLATE_SHA:
        raise ValueError("chat template differs from pinned GLM-5.3")
    for name, digest in (("config.json", CONFIG_SHA), ("generation_config.json", GENERATION_SHA)):
        if sha256(read_bounded(assets / name, 64 << 10)).hexdigest() != digest:
            raise ValueError("configuration differs from pinned GLM-5.3")
    for name, digest in TOKENIZER_FILES.items():
        if sha256(read_bounded(tokenizer_root / name, 32 << 20)).hexdigest() != digest:
            raise ValueError("tokenizer differs from pinned GLM-5.3")
    return template.decode()


def require_inventory(inventory):
    """A valid historical inventory is insufficient for the new weights."""
    if (
        inventory.model_id != MODEL_ID
        or inventory.source_revision != REVISION
        or inventory.config_sha256 != CONFIG_SHA
        or inventory.index_sha256 != INDEX_SHA
    ):
        raise ValueError("runtime inventory differs from pinned GLM-5.3 source")


def geometry(repo=None):
    """Use the pinned GLM-5.3 dimensions with its own checkpoint identity.

    The retained parser names its historical GLM-5.2 target unconditionally.
    Keep that frozen parser intact and bind the new identity at this boundary.
    """
    from dataclasses import replace
    from glm_tpu.config.model import ModelGeometry

    raw = read_bounded(hf_config(repo) / "config.json", 64 << 10)
    if sha256(raw).hexdigest() != CONFIG_SHA:
        raise ValueError("geometry configuration differs from pinned GLM-5.3")
    return replace(ModelGeometry.from_hf_config(json.loads(raw)), model_id=MODEL_ID)
