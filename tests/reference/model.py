"""Unsharded single-device reference forward of GLM-5.3 (``GlmMoeDsaForCausalLM``).

The reference runs op by op (no ``jit`` outside the elementwise FP8
dequantization, one device, no ``shard_map``, no Pallas) over BF16 weights
dequantized from the FP8 checkpoint tensors. One call
of :func:`forward` appends a block of consecutive tokens: prefill passes the
prompt (in any block partition), decode passes one token. Every layer, for
every row at absolute position ``p``:

1. ``normalized, residual = add_rms_norm(update, residual, input_norm)``
   (the first layer's update is the embedding and its residual zero);
2. q-a / kv-a preparation (:func:`tests.reference.attention.prepare`);
3. a *full* layer writes the rows' DSA index keys and selects each row's exact
   causal top-k (:mod:`tests.reference.dsa`); a *shared* layer (IndexShare)
   reuses the selection of the preceding full layer;
4. absorbed MLA over the selected positions (:mod:`tests.reference.attention`);
5. ``normalized, residual = add_rms_norm(attention, residual, post_norm)``;
6. dense SwiGLU (``mlp_layer_types == "dense"``) or the sparse MoE
   (:mod:`tests.reference.moe`); its output is the next layer's update.

The head applies ``add_rms_norm`` with the final norm, BF16 logits
``normalized @ lm_head.T`` (FP32 accumulation) and the greedy lowest-id
argmax. Cache layout: ``kv_cache[layer, position] = [latent | rope(k)]`` and
``index_cache[full_slot, position] = key``, both BF16, positions in order.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from glm_tpu.layers.attention._s3_attention import MlaNumericalContract
from glm_tpu.layers.attention._s3_dsa import DsaNumericalContract, SelectedPositions

from . import attention, dsa, moe
from .linear import FP8_BLOCK, dequantize, embed, greedy_token, project
from .norm import add_rms_norm


@dataclass(frozen=True)
class ReferenceConfig:
    """Everything the reference forward reads; no sharding, no engine options."""

    num_layers: int
    hidden_size: int
    vocab_size: int
    attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    index_heads: int
    index_head_dim: int
    index_top_k: int
    num_routed_experts: int
    routed_top_k: int
    moe_intermediate_size: int
    dense_intermediate_size: int
    mlp_layer_types: tuple[str, ...]
    indexer_types: tuple[str, ...]
    context_capacity: int
    rms_norm_epsilon: float = 1e-5
    rope_theta: float = 8_000_000.0
    routed_scaling_factor: float = 2.5
    fp8_block_shape: tuple[int, int] = FP8_BLOCK

    def __post_init__(self) -> None:
        if (
            len(self.mlp_layer_types) != self.num_layers
            or len(self.indexer_types) != self.num_layers
        ):
            raise ValueError("one MLP and one indexer type per layer")
        if self.indexer_types[0] != "full" or set(self.indexer_types) - {
            "full",
            "shared",
        }:
            raise ValueError("layer 0 must run the DSA indexer; types are full/shared")
        if set(self.mlp_layer_types) - {"dense", "sparse"}:
            raise ValueError("MLP types are dense/sparse")

    @classmethod
    def from_geometry(
        cls, geometry: Any, hf_config: Mapping[str, Any], *, context_capacity: int
    ) -> ReferenceConfig:
        """Dimensions from a model geometry, scalar constants from the HF ``config.json``.

        Refuses a config whose routing/RoPE semantics this reference does not implement.
        """
        expected = dict(
            n_group=1,
            topk_group=1,
            scoring_func="sigmoid",
            topk_method="noaux_tc",
            norm_topk_prob=True,
            rope_interleave=True,
            indexer_rope_interleave=True,
            hidden_act="silu",
            n_shared_experts=1,
            tie_word_embeddings=False,
        )
        drift = {
            k: hf_config.get(k) for k, v in expected.items() if hf_config.get(k) != v
        }
        rope = hf_config.get("rope_parameters", {})
        if drift or rope.get("rope_type", "default") != "default":
            raise ValueError(
                f"reference does not implement this config: {drift or rope}"
            )
        return cls(
            num_layers=geometry.num_layers,
            hidden_size=geometry.hidden_size,
            vocab_size=geometry.vocab_size,
            attention_heads=geometry.attention_heads,
            q_lora_rank=geometry.q_lora_rank,
            kv_lora_rank=geometry.kv_lora_rank,
            qk_nope_head_dim=geometry.qk_nope_head_dim,
            qk_rope_head_dim=geometry.qk_rope_head_dim,
            v_head_dim=geometry.v_head_dim,
            index_heads=geometry.dsa_indexer_heads,
            index_head_dim=geometry.dsa_indexer_head_dim,
            index_top_k=geometry.dsa_top_k,
            num_routed_experts=geometry.num_routed_experts,
            routed_top_k=geometry.routed_top_k,
            moe_intermediate_size=geometry.moe_intermediate_size,
            dense_intermediate_size=geometry.dense_intermediate_size,
            mlp_layer_types=tuple(geometry.mlp_layer_types),
            indexer_types=tuple(geometry.indexer_types),
            context_capacity=context_capacity,
            rms_norm_epsilon=float(hf_config["rms_norm_eps"]),
            rope_theta=float(rope["rope_theta"]),
            routed_scaling_factor=float(hf_config["routed_scaling_factor"]),
            fp8_block_shape=tuple(
                hf_config["quantization_config"]["weight_block_size"]
            ),
        )

    @property
    def full_layers(self) -> tuple[int, ...]:
        return tuple(i for i, kind in enumerate(self.indexer_types) if kind == "full")

    @property
    def sparse_layers(self) -> tuple[int, ...]:
        return tuple(
            i for i, kind in enumerate(self.mlp_layer_types) if kind == "sparse"
        )

    @property
    def attention_contract(self) -> MlaNumericalContract:
        return MlaNumericalContract(
            num_heads=self.attention_heads,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            packed_cache_width=self.kv_lora_rank + self.qk_rope_head_dim,
            top_k=self.index_top_k,
        )

    @property
    def indexer_contract(self) -> DsaNumericalContract:
        return DsaNumericalContract(
            hidden_size=self.hidden_size,
            q_lora_rank=self.q_lora_rank,
            num_heads=self.index_heads,
            head_dim=self.index_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            top_k=self.index_top_k,
            theta=self.rope_theta,
            key_layer_norm_epsilon=dsa.INDEX_KEY_NORM_EPSILON,
        )


class LayerWeights(NamedTuple):
    input_norm: jax.Array
    qkv_a: attention.QkvAWeights
    attention: attention.AttentionWeights
    indexer: dsa.IndexerWeights | None
    post_attention_norm: jax.Array
    mlp: moe.DenseWeights | moe.MoeWeights


class ReferenceWeights(NamedTuple):
    embedding: jax.Array  # [vocab, hidden] BF16
    layers: tuple[LayerWeights, ...]
    final_norm: jax.Array  # [hidden] BF16
    lm_head: jax.Array  # [vocab, hidden] BF16


class ReferenceState(NamedTuple):
    kv_cache: jax.Array  # [layers, capacity, kv_lora_rank + rope_dim] BF16
    index_cache: jax.Array  # [full layers, capacity, index_head_dim] BF16
    length: int  # cached positions; the next token sits at this position


class ForwardResult(NamedTuple):
    state: ReferenceState
    next_token: jax.Array  # [] int32, greedy at the block's last row
    logits: jax.Array  # [vocab] BF16 at the block's last row
    final_residual: jax.Array  # [rows, hidden] BF16 (the head's fused-norm residual)
    selections: tuple[dsa.Selection, ...]  # one per full layer, in layer order
    routes: tuple[moe.Routes, ...]  # one per sparse layer, in layer order


def load_weights(
    arrays: Mapping[str, Any], config: ReferenceConfig
) -> ReferenceWeights:
    """Bind checkpoint tensors by name and dequantize every FP8 pair to BF16.

    Refuses a missing or an unused tensor, so the reference reads exactly the
    checkpoint's name set.
    """
    used: set[str] = set()

    def raw(name: str) -> jax.Array:
        used.add(name)
        return jnp.asarray(np.asarray(arrays[name]))

    def fp8(prefix: str) -> jax.Array:
        return dequantize(
            raw(prefix + ".weight_bits"),
            raw(prefix + ".scale_inv"),
            block_shape=config.fp8_block_shape,
        )

    def dense(prefix: str) -> moe.DenseWeights:
        return moe.DenseWeights(
            fp8(prefix + ".gate_proj"),
            fp8(prefix + ".up_proj"),
            fp8(prefix + ".down_proj"),
        )

    layers = []
    for i in range(config.num_layers):
        p, a = f"model.layers.{i}", f"model.layers.{i}.self_attn"
        indexer = None
        if config.indexer_types[i] == "full":
            indexer = dsa.IndexerWeights(
                fp8(a + ".indexer.wq_b"),
                fp8(a + ".indexer.wk"),
                raw(a + ".indexer.k_norm.weight"),
                raw(a + ".indexer.k_norm.bias"),
                raw(a + ".indexer.weights_proj.weight"),
            )
        if config.mlp_layer_types[i] == "dense":
            mlp: moe.DenseWeights | moe.MoeWeights = dense(p + ".mlp")
        else:
            mlp = moe.MoeWeights(
                raw(p + ".mlp.gate.weight"),
                raw(p + ".mlp.gate.e_score_correction_bias"),
                fp8(p + ".mlp.experts.gate_proj"),
                fp8(p + ".mlp.experts.up_proj"),
                fp8(p + ".mlp.experts.down_proj"),
                dense(p + ".mlp.shared_experts"),
            )
        layers.append(
            LayerWeights(
                raw(p + ".input_layernorm.weight"),
                attention.QkvAWeights(
                    fp8(a + ".q_a_proj"),
                    raw(a + ".q_a_layernorm.weight"),
                    fp8(a + ".kv_a_proj_with_mqa"),
                    raw(a + ".kv_a_layernorm.weight"),
                ),
                attention.AttentionWeights(
                    fp8(a + ".q_b_proj"), fp8(a + ".kv_b_proj"), fp8(a + ".o_proj")
                ),
                indexer,
                raw(p + ".post_attention_layernorm.weight"),
                mlp,
            )
        )
    weights = ReferenceWeights(
        raw("model.embed_tokens.weight"),
        tuple(layers),
        raw("model.norm.weight"),
        raw("lm_head.weight"),
    )
    unused = sorted(set(arrays) - used)
    if unused:
        raise ValueError(
            f"checkpoint tensors the reference does not read: {unused[:4]}"
        )
    return weights


def initial_state(config: ReferenceConfig) -> ReferenceState:
    kv_width = config.kv_lora_rank + config.qk_rope_head_dim
    return ReferenceState(
        jnp.zeros((config.num_layers, config.context_capacity, kv_width), jnp.bfloat16),
        jnp.zeros(
            (len(config.full_layers), config.context_capacity, config.index_head_dim),
            jnp.bfloat16,
        ),
        0,
    )


def rope_table(config: ReferenceConfig) -> jax.Array:
    return attention.main_rope_table(
        config.context_capacity,
        rotary_dim=config.qk_rope_head_dim,
        theta=config.rope_theta,
    )


def forward(
    config: ReferenceConfig,
    weights: ReferenceWeights,
    state: ReferenceState,
    token_ids: Any,
    *,
    rope: jax.Array,
) -> ForwardResult:
    """Append ``token_ids`` (``[rows]``) at positions ``state.length ..`` and run the head."""
    tokens = jnp.asarray(token_ids, jnp.int32).reshape(-1)
    rows = int(tokens.shape[0])
    if rows < 1 or state.length + rows > config.context_capacity:
        raise ValueError("block is empty or exceeds the context capacity")
    if bool(jnp.any((tokens < 0) | (tokens >= config.vocab_size))):
        raise ValueError("token id outside the vocabulary")
    positions = state.length + jnp.arange(rows, dtype=jnp.int32)
    eps = config.rms_norm_epsilon
    update = embed(tokens, weights.embedding)
    residual = jnp.zeros_like(update)
    kv_cache, index_cache = state.kv_cache, state.index_cache
    selection: dsa.Selection | None = None
    selections, routes = [], []
    for layer_id, layer in enumerate(weights.layers):
        normalized, residual = add_rms_norm(
            update, residual, layer.input_norm, epsilon=eps
        )
        prepared = attention.prepare(normalized, layer.qkv_a, epsilon=eps)
        if layer.indexer is not None:
            slot = config.full_layers.index(layer_id)
            keys = dsa.index_keys(
                normalized, layer.indexer, positions, contract=config.indexer_contract
            )
            index_cache = index_cache.at[slot, positions].set(keys)
            selection = dsa.select(
                normalized,
                prepared.q_residual,
                index_cache[slot],
                positions,
                layer.indexer,
                contract=config.indexer_contract,
            )
            selections.append(selection)
        assert selection is not None  # layer 0 is a full layer (checked by the config)
        output, layer_cache = attention.mla_attention(
            prepared,
            positions,
            kv_cache[layer_id],
            SelectedPositions(selection.positions, selection.valid_counts),
            layer.attention,
            contract=config.attention_contract,
            rope_table=rope,
        )
        kv_cache = kv_cache.at[layer_id].set(layer_cache)
        normalized, residual = add_rms_norm(
            output, residual, layer.post_attention_norm, epsilon=eps
        )
        if isinstance(layer.mlp, moe.MoeWeights):
            update, layer_routes = moe.moe(
                normalized,
                layer.mlp,
                top_k=config.routed_top_k,
                routed_scaling_factor=config.routed_scaling_factor,
            )
            routes.append(layer_routes)
        else:
            update = moe.dense_mlp(normalized, layer.mlp)
    normalized, final_residual = add_rms_norm(
        update, residual, weights.final_norm, epsilon=eps
    )
    logits = project(normalized[-1:], weights.lm_head)[0]
    return ForwardResult(
        ReferenceState(kv_cache, index_cache, state.length + rows),
        greedy_token(logits),
        logits,
        final_residual,
        tuple(selections),
        tuple(routes),
    )


def generate(
    config: ReferenceConfig,
    weights: ReferenceWeights,
    prompt: Any,
    new_tokens: int,
    *,
    block_rows: int = 128,
) -> tuple[list[int], list[ForwardResult]]:
    """Greedy continuation: prefill in ``block_rows`` blocks, then one token per step."""
    rope = rope_table(config)
    state = initial_state(config)
    prompt = list(prompt)
    results = []
    for start in range(0, len(prompt), block_rows):
        results.append(
            forward(
                config, weights, state, prompt[start : start + block_rows], rope=rope
            )
        )
        state = results[-1].state
    tokens = [int(results[-1].next_token)]
    while len(tokens) < new_tokens:
        results.append(forward(config, weights, state, [tokens[-1]], rope=rope))
        state = results[-1].state
        tokens.append(int(results[-1].next_token))
    return tokens, results
