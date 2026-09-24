"""Independent FP64 NumPy restatement of ``GlmMoeDsaForCausalLM`` (shares no code).

The reference (:mod:`tests.reference.model`) reuses oracle functions under
``glm_tpu/optimized/reference`` that production also executes (router,
RoPE table and rotation, DSA keys and scores, RMSNorm, FP8 dequantization;
``VALIDATION.md`` lists them). A bug in one of them is common to the reference,
the frozen FP8 oracle and production, so their agreement cannot reveal it. This
module restates the model from the Hugging Face eager modeling
(``reference/modeling_glm_moe_dsa.py``) in plain FP64 NumPy and imports only
NumPy, ``ml_dtypes`` and the standard library
(``test_independent_restatement_imports_no_repository_code`` enforces that).

It follows the HF code with the engine's two documented conventions, which are
properties of the pinned config rather than of any implementation:

* the DSA indexer RoPE pairs are interleaved (``indexer_rope_interleave: true``;
  the HF indexer rotates half-split);
* the q-a and kv-a LoRA RMSNorms use ``rms_norm_eps`` (HF constructs them with
  its default 1e-6).

Weights are ``bf16(f32(e4m3(bits)) * scale_inv[block])`` (the dequantized BF16
tables every system reads), widened to FP64. Everything after that is FP64 with
no intermediate rounding: RMSNorm ``w * x / sqrt(mean(x^2) + eps)``, the
non-absorbed MLA (per-head keys and values expanded through ``kv_b``, exact
``cos``/``sin``, interleaved main RoPE, scale ``qk_head_dim**-0.5``, softmax over
the DSA selection), the DSA indexer (affine key LayerNorm with ``eps = 1e-6``,
RoPE on the first ``rope_dim`` features of keys and per-head queries, ReLU of
the scaled dots weighted by ``weights_proj * heads**-0.5``, top-k with ties to
the lowest position), IndexShare (a shared layer reuses the preceding full
layer's selection), the sigmoid noaux_tc router (correction bias in the choice
only, ties to the lowest expert id, normalized weights times
``routed_scaling_factor``), SwiGLU experts plus the shared expert, the final
RMSNorm and the LM head.

:func:`forward` runs a whole sequence at once. Every row's selection, routes and
attention depend only on rows at or before it, so row ``p`` is what an
incremental implementation computes for position ``p`` in any prefill partition
or decode step.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import ml_dtypes
import numpy as np

INDEX_KEY_NORM_EPSILON = (
    1e-6  # nn.LayerNorm(index_head_dim, eps=1e-6) in GlmMoeDsaIndexer
)


class Trace(NamedTuple):
    """Everything :func:`forward` computed, per row (FP64).

    Per layer: ``latent`` ``[rows, kv_lora_rank]`` (normalized), ``key_rope``
    ``[rows, rope_dim]`` (rotated, pairs interleaved), ``selected`` ``[rows, rows]``
    (the attended positions). Per full layer: ``index_keys`` ``[rows, index_head_dim]``
    (rotated), ``index_scores`` ``[rows, rows]`` (``-inf`` above the diagonal),
    ``dsa_margin`` ``[rows]`` (own ``top_k``-th minus the next score, ``+inf`` if
    none). Per sparse layer: ``router_choice`` ``[rows, experts]`` (the biased sigmoid
    scores the router ranks), ``router_margin`` ``[rows]`` (k-th minus the next
    choice) and ``expert_ids`` ``[rows, top_k]`` (the routed experts).
    """

    logits: np.ndarray  # [rows, vocab]
    residual: np.ndarray  # [rows, hidden], before the final norm
    latent: tuple[np.ndarray, ...]
    key_rope: tuple[np.ndarray, ...]
    index_keys: dict[int, np.ndarray]
    index_scores: dict[int, np.ndarray]
    selected: tuple[np.ndarray, ...]
    dsa_margin: dict[int, np.ndarray]
    router_margin: dict[int, np.ndarray]
    router_choice: dict[int, np.ndarray]
    expert_ids: dict[int, np.ndarray]


def dequantize(bits: Any, scale_inv: Any, block_shape: tuple[int, int]) -> np.ndarray:
    """E4M3FN ``bits`` (``uint8``) times the FP32 block scale, rounded to BF16, as FP64."""
    values = np.asarray(bits, np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float64)
    rows, cols = block_shape
    scale = np.asarray(scale_inv, np.float64)
    scale = np.repeat(np.repeat(scale, rows, axis=-2), cols, axis=-1)
    scale = scale[..., : values.shape[-2], : values.shape[-1]]
    # The FP64 product is exact (4 + 24 significant bits); round it to FP32, then BF16.
    product = (values * scale).astype(np.float32)
    return product.astype(ml_dtypes.bfloat16).astype(np.float64)


def load(
    arrays: Mapping[str, Any], block_shape: tuple[int, int]
) -> dict[str, np.ndarray]:
    """FP64 weights keyed by checkpoint name; an FP8 pair ``X.weight_bits``/``X.scale_inv`` becomes ``X``."""
    weights: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if name.endswith(".scale_inv"):
            continue
        if name.endswith(".weight_bits"):
            prefix = name[: -len(".weight_bits")]
            weights[prefix] = dequantize(
                value, arrays[prefix + ".scale_inv"], tuple(block_shape)
            )
        else:
            weights[name] = np.asarray(value).astype(np.float64)
    return weights


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    return weight * (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps))


def layer_norm(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float
) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps) * weight + bias


def rope(x: np.ndarray, positions: np.ndarray, theta: float) -> np.ndarray:
    """Rotate pairs ``(2j, 2j+1)`` of ``x[rows, ..., dim]`` by ``position * theta**(-2j/dim)``.

    The result keeps the pairs interleaved (HF concatenates the two halves instead;
    every dot product between two rotated vectors is the same).
    """
    dim = x.shape[-1]
    frequencies = theta ** (-np.arange(0, dim, 2, dtype=np.float64) / dim)
    angles = positions[:, None] * frequencies[None, :]
    shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (dim // 2,)
    cos, sin = np.cos(angles).reshape(shape), np.sin(angles).reshape(shape)
    even, odd = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = odd * cos + even * sin
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def swiglu(x: np.ndarray, weights: Mapping[str, np.ndarray], prefix: str) -> np.ndarray:
    gate = x @ weights[prefix + ".gate_proj"].T
    up = x @ weights[prefix + ".up_proj"].T
    return (gate * sigmoid(gate) * up) @ weights[prefix + ".down_proj"].T


def top_k_margin(
    values: np.ndarray, order: np.ndarray, k: int, live: np.ndarray
) -> np.ndarray:
    """Per row, the k-th minus the (k+1)-th of ``values`` in ``order``; +inf where ``live <= k``."""
    if values.shape[1] <= k:
        return np.full(values.shape[0], np.inf)
    rows = np.arange(values.shape[0])
    with np.errstate(invalid="ignore"):  # -inf - -inf past a short row's context
        gap = values[rows, order[:, k - 1]] - values[rows, order[:, k]]
    return np.where(live > k, gap, np.inf)


def indexer(
    x: np.ndarray,
    q_resid: np.ndarray,
    weights: Mapping[str, np.ndarray],
    prefix: str,
    config: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """``GlmMoeDsaIndexer.forward`` up to the top-k: ``(keys [rows, dim], scores [rows, rows])``.

    Row ``r`` sits at position ``r``; scores above the diagonal are ``-inf``.
    """
    rows, rope_dim = x.shape[0], config.qk_rope_head_dim
    heads, dim = config.index_heads, config.index_head_dim
    positions = np.arange(rows, dtype=np.float64)
    key = layer_norm(
        x @ weights[prefix + ".wk"].T,
        weights[prefix + ".k_norm.weight"],
        weights[prefix + ".k_norm.bias"],
        INDEX_KEY_NORM_EPSILON,
    )
    key = np.concatenate(
        (rope(key[:, :rope_dim], positions, config.rope_theta), key[:, rope_dim:]),
        axis=-1,
    )
    query = (q_resid @ weights[prefix + ".wq_b"].T).reshape(rows, heads, dim)
    query = np.concatenate(
        (
            rope(query[..., :rope_dim], positions, config.rope_theta),
            query[..., rope_dim:],
        ),
        axis=-1,
    )
    per_head = np.maximum(np.einsum("shd,td->sht", query, key) * dim**-0.5, 0.0)
    head_weights = (x @ weights[prefix + ".weights_proj.weight"].T) * heads**-0.5
    scores = np.einsum("sh,sht->st", head_weights, per_head)
    causal = np.arange(rows)[None, :] <= np.arange(rows)[:, None]
    return key, np.where(causal, scores, -np.inf)


def top_k_selection(scores: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Causal top-k of ``scores [rows, rows]`` (ties: lowest position): ``(mask, margin)``."""
    rows = scores.shape[0]
    order = np.argsort(-scores, axis=-1, kind="stable")
    mask = np.zeros((rows, rows), bool)
    for r in range(rows):
        mask[r, order[r, : min(top_k, r + 1)]] = True
    return mask, top_k_margin(scores, order, top_k, np.arange(1, rows + 1))


class AttentionInputs(NamedTuple):
    """The q-a / kv-a boundary of one layer (rows at positions ``0..rows-1``).

    ``q_resid`` ``[rows, q_lora_rank]`` is ``q_a_layernorm(q_a_proj(x))`` (also the
    indexer's input), ``q`` ``[rows, heads, nope + rope_dim]`` has its RoPE part
    rotated, ``latent`` ``[rows, kv_lora_rank]`` is ``kv_a_layernorm(compressed)``
    and ``key_rope`` ``[rows, rope_dim]`` is rotated with the pairs interleaved.
    """

    q_resid: np.ndarray
    q: np.ndarray
    latent: np.ndarray
    key_rope: np.ndarray


def attention_inputs(
    x: np.ndarray, weights: Mapping[str, np.ndarray], prefix: str, config: Any
) -> AttentionInputs:
    """``GlmMoeDsaAttention.forward`` up to the key/value expansion (``x`` is normalized)."""
    c, rows = config, x.shape[0]
    eps, nope, lora = float(c.rms_norm_epsilon), c.qk_nope_head_dim, c.kv_lora_rank
    positions = np.arange(rows, dtype=np.float64)
    q_resid = rms_norm(
        x @ weights[prefix + ".q_a_proj"].T,
        weights[prefix + ".q_a_layernorm.weight"],
        eps,
    )
    q = (q_resid @ weights[prefix + ".q_b_proj"].T).reshape(
        rows, c.attention_heads, nope + c.qk_rope_head_dim
    )
    q = np.concatenate(
        (q[..., :nope], rope(q[..., nope:], positions, c.rope_theta)), -1
    )
    compressed = x @ weights[prefix + ".kv_a_proj_with_mqa"].T
    latent = rms_norm(
        compressed[:, :lora], weights[prefix + ".kv_a_layernorm.weight"], eps
    )
    key_rope = rope(compressed[:, lora:], positions, c.rope_theta)
    return AttentionInputs(q_resid, q, latent, key_rope)


def attention(
    inputs: AttentionInputs,
    selected: np.ndarray,
    weights: Mapping[str, np.ndarray],
    prefix: str,
    config: Any,
) -> np.ndarray:
    """Softmax attention of every row over its ``selected`` positions, then ``o_proj``."""
    c, rows = config, inputs.q.shape[0]
    nope, v_dim = c.qk_nope_head_dim, c.v_head_dim
    kv = (inputs.latent @ weights[prefix + ".kv_b_proj"].T).reshape(
        rows, c.attention_heads, nope + v_dim
    )
    key_nope, values = kv[..., :nope], kv[..., nope:]
    logits = (
        np.einsum("shd,thd->hst", inputs.q[..., :nope], key_nope)
        + np.einsum("shd,td->hst", inputs.q[..., nope:], inputs.key_rope)
    ) * (nope + c.qk_rope_head_dim) ** -0.5
    logits = np.where(np.asarray(selected, bool)[None], logits, -np.inf)
    probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    attended = np.einsum("hst,thv->shv", probabilities, values)
    return (
        attended.reshape(rows, c.attention_heads * v_dim)
        @ weights[prefix + ".o_proj"].T
    )


def sparse_moe(
    x: np.ndarray,
    weights: Mapping[str, np.ndarray],
    prefix: str,
    config: Any,
    expert_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``GlmMoeDsaMoE.forward`` (``x`` normalized): ``(update, choice, routed expert ids)``.

    ``choice`` is the biased sigmoid score the router ranks; ``expert_ids`` imposes
    the routed experts (their weights still come from this router's scores).
    """
    k = config.routed_top_k
    scores = sigmoid(x @ weights[prefix + ".gate.weight"].T)
    choice = scores + weights[prefix + ".gate.e_score_correction_bias"][None, :]
    if expert_ids is None:  # ties: the lowest expert id
        expert_ids = np.argsort(-choice, axis=-1, kind="stable")[:, :k]
    ids = np.asarray(expert_ids, np.int64)
    route_weights = np.take_along_axis(scores, ids, axis=-1)
    route_weights = route_weights / (route_weights.sum(-1, keepdims=True) + 1e-20)
    route_weights = route_weights * config.routed_scaling_factor
    gate = np.einsum("rh,eih->rei", x, weights[prefix + ".experts.gate_proj"])
    up = np.einsum("rh,eih->rei", x, weights[prefix + ".experts.up_proj"])
    experts = np.einsum(
        "rei,ehi->reh",
        gate * sigmoid(gate) * up,
        weights[prefix + ".experts.down_proj"],
    )
    chosen = np.take_along_axis(experts, ids[:, :, None], axis=1)
    update = np.einsum("rk,rkh->rh", route_weights, chosen)
    return update + swiglu(x, weights, prefix + ".shared_experts"), choice, ids


def forward(
    config: Any,
    weights: Mapping[str, np.ndarray],
    tokens: Any,
    *,
    selected: Mapping[int, np.ndarray] | None = None,
    expert_ids: Mapping[int, np.ndarray] | None = None,
) -> Trace:
    """Run ``tokens`` (positions ``0..rows-1``) through the model; see the module docstring.

    ``config`` is read for plain dimensions and constants only (any object with the
    fields of :class:`tests.reference.model.ReferenceConfig`). ``selected`` (full
    layer -> ``[rows, rows]`` bool) and ``expert_ids`` (sparse layer -> ``[rows,
    top_k]``) impose another system's discrete decisions: the attended positions of
    that full layer (shared layers still reuse the preceding full layer's) and the
    routed experts (their weights still come from this model's scores). The
    returned margins and scores are always this model's own.
    """
    c = config
    selected, expert_ids = dict(selected or {}), dict(expert_ids or {})
    tokens = np.asarray(tokens, np.int64).reshape(-1)
    rows, eps = tokens.shape[0], float(c.rms_norm_epsilon)

    hidden = weights["model.embed_tokens.weight"][tokens]
    latents, key_ropes, selections = [], [], []
    index_keys, index_scores, dsa_margin = {}, {}, {}
    router_margin, router_choice, routed = {}, {}, {}
    attended: np.ndarray | None = None
    for i in range(c.num_layers):
        p, a = f"model.layers.{i}", f"model.layers.{i}.self_attn"
        x = rms_norm(hidden, weights[p + ".input_layernorm.weight"], eps)
        inputs = attention_inputs(x, weights, a, c)
        latents.append(inputs.latent)
        key_ropes.append(inputs.key_rope)
        if c.indexer_types[i] == "full":
            index_keys[i], index_scores[i] = indexer(
                x, inputs.q_resid, weights, a + ".indexer", c
            )
            own, dsa_margin[i] = top_k_selection(index_scores[i], c.index_top_k)
            attended = np.asarray(selected.get(i, own), bool)
        elif attended is None:
            raise ValueError("layer 0 must be a full DSA layer")
        selections.append(attended)  # a shared layer reuses the preceding full layer's
        hidden = hidden + attention(inputs, attended, weights, a, c)

        x = rms_norm(hidden, weights[p + ".post_attention_layernorm.weight"], eps)
        if c.mlp_layer_types[i] == "dense":
            update = swiglu(x, weights, p + ".mlp")
        else:
            update, choice, routed[i] = sparse_moe(
                x, weights, p + ".mlp", c, expert_ids.get(i)
            )
            order = np.argsort(-choice, axis=-1, kind="stable")
            router_margin[i] = top_k_margin(
                choice, order, c.routed_top_k, np.full(rows, choice.shape[1])
            )
            router_choice[i] = choice
        hidden = hidden + update

    final = rms_norm(hidden, weights["model.norm.weight"], eps)
    return Trace(
        final @ weights["lm_head.weight"].T,
        hidden,
        tuple(latents),
        tuple(key_ropes),
        index_keys,
        index_scores,
        tuple(selections),
        dsa_margin,
        router_margin,
        router_choice,
        routed,
    )
