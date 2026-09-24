"""The unsharded reference model: unit semantics and cross-validation (DESIGN 7.6, S2e).

Fast tests run the reference alone on one CPU device over a tiny synthetic
checkpoint (the fixture's tensor names, small dimensions), and compare it with
:mod:`tests.reference.independent`, an FP64 NumPy restatement that shares no code
with the reference, the oracles or production (the only check of the oracle
functions the reference and production both execute). The ``slow`` tests read
the frozen fixture v1 or run several block partitions. The ``cpu32`` test runs
:mod:`tests.reference.oracle_run` in a child with 32 forced CPU devices and
asserts the acceptance criteria of ``VALIDATION.md`` against the production
composition: every DESIGN 7.6 criterion the two accepted engines (frozen FP8
oracle, production) met against each other holds exactly for the reference; for
the others the reference is no further from production than production was from
the FP8 oracle (the measured floor, recorded from the oracle's final run in
``floors.json``). ``test_reference_matches_frozen_fp8_oracle`` was archived
together with the frozen FP8 oracle at S2f (``archive/research-20260922``);
``VALIDATION.md`` is its receipt.
"""

from __future__ import annotations

import ast
import dataclasses
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from glm_tpu.optimized.reference.dsa import SelectedPositions
from tests.reference import attention, dsa, independent, model, moe
from tests.reference.linear import dequantize, greedy_token
from tests.reference.norm import add_rms_norm

TINY = model.ReferenceConfig(
    num_layers=4,
    hidden_size=64,
    vocab_size=32,
    attention_heads=2,
    q_lora_rank=32,
    kv_lora_rank=32,
    qk_nope_head_dim=16,
    qk_rope_head_dim=8,
    v_head_dim=16,
    index_heads=2,
    index_head_dim=16,
    index_top_k=4,
    num_routed_experts=8,
    routed_top_k=2,
    moe_intermediate_size=16,
    dense_intermediate_size=32,
    mlp_layer_types=("dense", "sparse", "sparse", "sparse"),
    indexer_types=("full", "shared", "full", "shared"),
    context_capacity=32,
)


def tiny_checkpoint(
    config: model.ReferenceConfig, seed: int = 7
) -> dict[str, np.ndarray]:
    """A random checkpoint with the production tensor names (FP8 bits + FP32 block scales)."""
    rng = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {}

    def fp8(name: str, shape: tuple[int, ...]) -> None:
        bits = np.asarray(rng.normal(0, 0.15, shape), ml_dtypes.float8_e4m3fn).view(
            np.uint8
        )
        blocks = shape[:-2] + tuple(-(-d // 128) for d in shape[-2:])
        arrays[name + ".weight_bits"] = bits
        arrays[name + ".scale_inv"] = rng.uniform(0.5, 1.5, blocks).astype(np.float32)

    def bf16(
        name: str, shape: tuple[int, ...], scale: float = 0.1, offset: float = 0.0
    ) -> None:
        arrays[name] = np.asarray(
            offset + rng.normal(0, scale, shape), ml_dtypes.bfloat16
        )

    c = config
    heads, qk = c.attention_heads, c.qk_nope_head_dim + c.qk_rope_head_dim
    for i in range(c.num_layers):
        p, a = f"model.layers.{i}", f"model.layers.{i}.self_attn"
        bf16(p + ".input_layernorm.weight", (c.hidden_size,), 0.05, 1.0)
        fp8(a + ".q_a_proj", (c.q_lora_rank, c.hidden_size))
        bf16(a + ".q_a_layernorm.weight", (c.q_lora_rank,), 0.05, 1.0)
        fp8(
            a + ".kv_a_proj_with_mqa",
            (c.kv_lora_rank + c.qk_rope_head_dim, c.hidden_size),
        )
        bf16(a + ".kv_a_layernorm.weight", (c.kv_lora_rank,), 0.05, 1.0)
        fp8(a + ".q_b_proj", (heads * qk, c.q_lora_rank))
        fp8(
            a + ".kv_b_proj",
            (heads * (c.qk_nope_head_dim + c.v_head_dim), c.kv_lora_rank),
        )
        fp8(a + ".o_proj", (c.hidden_size, heads * c.v_head_dim))
        if c.indexer_types[i] == "full":
            fp8(a + ".indexer.wq_b", (c.index_heads * c.index_head_dim, c.q_lora_rank))
            fp8(a + ".indexer.wk", (c.index_head_dim, c.hidden_size))
            bf16(a + ".indexer.k_norm.weight", (c.index_head_dim,), 0.05, 1.0)
            bf16(a + ".indexer.k_norm.bias", (c.index_head_dim,))
            bf16(a + ".indexer.weights_proj.weight", (c.index_heads, c.hidden_size))
        if c.mlp_layer_types[i] == "dense":
            fp8(p + ".mlp.gate_proj", (c.dense_intermediate_size, c.hidden_size))
            fp8(p + ".mlp.up_proj", (c.dense_intermediate_size, c.hidden_size))
            fp8(p + ".mlp.down_proj", (c.hidden_size, c.dense_intermediate_size))
        else:
            e, m = c.num_routed_experts, c.moe_intermediate_size
            bf16(p + ".mlp.gate.weight", (e, c.hidden_size))
            # On the scale of the sigmoid scores' spread across experts (router logits
            # ~N(0, 0.8), scores spread by ~0.17), so the bias decides top-k choices and
            # a bias applied the wrong way changes routes (the frozen fixture's bias is
            # all zeros, VALIDATION.md).
            arrays[p + ".mlp.gate.e_score_correction_bias"] = rng.normal(
                0, 0.1, e
            ).astype(np.float32)
            fp8(p + ".mlp.experts.gate_proj", (e, m, c.hidden_size))
            fp8(p + ".mlp.experts.up_proj", (e, m, c.hidden_size))
            fp8(p + ".mlp.experts.down_proj", (e, c.hidden_size, m))
            fp8(p + ".mlp.shared_experts.gate_proj", (m, c.hidden_size))
            fp8(p + ".mlp.shared_experts.up_proj", (m, c.hidden_size))
            fp8(p + ".mlp.shared_experts.down_proj", (c.hidden_size, m))
        bf16(p + ".post_attention_layernorm.weight", (c.hidden_size,), 0.05, 1.0)
    bf16("model.embed_tokens.weight", (c.vocab_size, c.hidden_size), 1.0)
    bf16("lm_head.weight", (c.vocab_size, c.hidden_size))
    bf16("model.norm.weight", (c.hidden_size,), 0.05, 1.0)
    return arrays


@pytest.fixture(scope="module")
def tiny() -> tuple[model.ReferenceConfig, model.ReferenceWeights]:
    return TINY, model.load_weights(tiny_checkpoint(TINY), TINY)


def run_blocks(
    config: Any, weights: Any, prompt: list[int], sizes: list[int]
) -> list[model.ForwardResult]:
    rope, state, results, start = (
        model.rope_table(config),
        model.initial_state(config),
        [],
        0,
    )
    for size in sizes:
        results.append(
            model.forward(
                config, weights, state, prompt[start : start + size], rope=rope
            )
        )
        state, start = results[-1].state, start + size
    assert start == len(prompt)
    return results


def as_f32(value: Any) -> np.ndarray:
    return np.asarray(jnp.asarray(value).astype(jnp.float32))


# ----------------------------------------------------------------------------- units
def test_dequantize_equals_independent_e4m3_block_decode():
    rng = np.random.default_rng(11)
    bits = rng.integers(0, 256, (256, 384), dtype=np.uint8)
    bits[(bits & 0x7F) == 0x7F] = (
        0  # the two NaN encodings are a loader refusal, not arithmetic
    )
    scale = rng.uniform(0.25, 2.0, (2, 3)).astype(np.float32)
    expected = bits.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * np.repeat(
        np.repeat(scale, 128, 0), 128, 1
    )
    actual = np.asarray(dequantize(bits, scale))
    assert actual.dtype == ml_dtypes.bfloat16
    np.testing.assert_array_equal(
        actual.view(np.uint16), expected.astype(ml_dtypes.bfloat16).view(np.uint16)
    )


def test_tie_rules_lowest_token_position_and_expert(tiny):
    config, weights = tiny
    # Greedy head: the lowest vocabulary id among equal BF16 maxima.
    assert int(greedy_token(jnp.asarray([1.0, 3.0, 3.0, 2.0], jnp.bfloat16))) == 1
    # DSA: identical keys give equal scores; ties go to the lowest positions, -1 past the context.
    layer = weights.layers[0]
    rng = np.random.default_rng(3)
    normalized = jnp.asarray(rng.normal(0, 1, (2, config.hidden_size)), jnp.bfloat16)
    q_residual = jnp.asarray(rng.normal(0, 1, (2, config.q_lora_rank)), jnp.bfloat16)
    key = jnp.asarray(rng.normal(0, 1, (1, config.index_head_dim)), jnp.bfloat16)
    cache = jnp.repeat(key, config.context_capacity, axis=0)
    selected = dsa.select(
        normalized,
        q_residual,
        cache,
        jnp.asarray([9, 1], jnp.int32),
        layer.indexer,
        contract=config.indexer_contract,
    )
    np.testing.assert_array_equal(
        np.asarray(selected.positions), [[0, 1, 2, 3], [0, 1, -1, -1]]
    )
    np.testing.assert_array_equal(np.asarray(selected.valid_counts), [4, 2])
    assert np.isneginf(np.asarray(selected.scores)[1, 2:]).all()
    # noaux_tc: equal logits pick the lowest expert ids; the bias moves ids, never weights.
    mlp = weights.layers[1].mlp
    flat = mlp._replace(
        router=jnp.ones_like(mlp.router),
        correction_bias=jnp.zeros_like(mlp.correction_bias),
    )
    plain = moe.route(normalized[:1], flat, top_k=config.routed_top_k)
    np.testing.assert_array_equal(np.asarray(plain.expert_ids), [[0, 1]])
    assert float(plain.margin[0]) == 0.0  # a pure tie: the decision has no slack
    biased = moe.route(
        normalized[:1],
        flat._replace(correction_bias=flat.correction_bias.at[5].set(1.0)),
        top_k=config.routed_top_k,
    )
    np.testing.assert_array_equal(np.asarray(biased.expert_ids), [[5, 0]])
    np.testing.assert_array_equal(np.asarray(biased.weights), np.asarray(plain.weights))
    np.testing.assert_allclose(np.asarray(plain.weights), [[0.5, 0.5]])


def test_router_adds_the_correction_bias_to_the_sigmoid_scores(tiny):
    """noaux_tc ranks ``sigmoid(logit) + bias`` and weights by the unbiased scores.

    Hugging Face ``GlmMoeDsaMoE.route_tokens_to_experts``: ``router_logits.sigmoid()``
    plus ``e_score_correction_bias`` for the choice, the unbiased sigmoid of the chosen
    experts normalized for the weights. The logits are not flat and every misreading of
    the bias below ranks at least one row differently (asserted), so this is the check
    of how the bias is applied: the frozen fixture's bias is all zeros, so no ``cpu32``
    comparison can see it, and the shared oracle router is common to all systems.
    """
    config, weights = tiny
    mlp = weights.layers[1].mlp
    experts, hidden = mlp.router.shape
    top_k = config.routed_top_k
    assert (experts, top_k) == (8, 2)
    # One row per case; BF16-exact logits (sigmoid 0.119 unless set), one shared bias.
    logits = np.full((3, experts), -2.0)
    logits[:, 0], logits[:, 2] = 4.0, 1.65625  # sigmoid 0.982 and 0.840 in every row
    logits[0, 1] = 0.0  # 0.5 + 0.45 = 0.95 beats 0.840; sigmoid(0.45) = 0.61 does not
    logits[1, 1] = -0.84765625  # 0.3 + 0.45 = 0.75 loses to 0.840; 0.3 + 0.9 would win
    logits[2, 3] = 2.9375  # 0.950 - 0.3 = 0.65 loses; sigmoid(2.64) = 0.933 would win
    bias = np.zeros(experts, np.float32)
    bias[1], bias[3] = 0.45, -0.3
    router = np.zeros((experts, hidden), np.float32)
    router[:, :3] = logits.T  # one-hot rows: the FP32 logits are these exactly
    routes = moe.route(
        jnp.eye(3, hidden, dtype=jnp.bfloat16),
        mlp._replace(
            router=jnp.asarray(router, jnp.bfloat16),
            correction_bias=jnp.asarray(bias),
        ),
        top_k=top_k,
    )

    def ranked(choice: np.ndarray) -> np.ndarray:
        return np.argsort(-choice, axis=1, kind="stable")[:, :top_k]

    scores = independent.sigmoid(logits)
    choice = scores + bias.astype(np.float64)
    expected = ranked(choice)
    np.testing.assert_array_equal(expected, [[0, 1], [0, 2], [0, 2]])
    np.testing.assert_array_equal(np.asarray(routes.expert_ids), expected)
    chosen = np.take_along_axis(scores, expected, axis=1)
    np.testing.assert_allclose(
        np.asarray(routes.weights), chosen / chosen.sum(1, keepdims=True), rtol=1e-6
    )
    ordered = np.sort(choice, axis=1)
    np.testing.assert_allclose(
        np.asarray(routes.margin),
        ordered[:, -top_k] - ordered[:, -top_k - 1],
        rtol=0,
        atol=1e-6,
    )
    misreadings = {
        "bias before the sigmoid": independent.sigmoid(logits + bias),
        "bias halved": scores + 0.5 * bias,
        "bias doubled": scores + 2.0 * bias,
        "bias subtracted": scores - bias,
        "bias ignored": scores,
    }
    for name, wrong in misreadings.items():
        assert (np.sort(ranked(wrong), 1) != np.sort(expected, 1)).any(), name


def test_margins_when_every_candidate_is_chosen(tiny):
    """top_k equal to the context capacity or to the expert count: margin +inf, no error."""
    config, weights = tiny
    scores = jnp.asarray([[3.0, 1.0, 2.0, 0.5]] * 2)
    np.testing.assert_array_equal(
        dsa.decision_margin(scores, jnp.asarray([4, 2]), 2), [1.0, np.inf]
    )
    assert np.isposinf(dsa.decision_margin(scores, jnp.asarray([4, 4]), 4)).all()
    wide = dataclasses.replace(config, index_top_k=config.context_capacity)
    rng = np.random.default_rng(17)
    rows = 6
    normalized = jnp.asarray(rng.normal(0, 1, (rows, config.hidden_size)), jnp.bfloat16)
    selected = dsa.select(
        normalized,
        jnp.asarray(rng.normal(0, 1, (rows, config.q_lora_rank)), jnp.bfloat16),
        jnp.asarray(
            rng.normal(0, 1, (config.context_capacity, config.index_head_dim)),
            jnp.bfloat16,
        ),
        jnp.arange(rows, dtype=jnp.int32),
        weights.layers[0].indexer,
        contract=wide.indexer_contract,
    )
    assert np.isposinf(np.asarray(selected.margin)).all()
    np.testing.assert_array_equal(np.asarray(selected.valid_counts), np.arange(1, 7))
    routes = moe.route(
        normalized, weights.layers[1].mlp, top_k=config.num_routed_experts
    )
    assert np.isposinf(np.asarray(routes.margin)).all()
    np.testing.assert_allclose(np.asarray(routes.weights).sum(axis=1), 1.0, rtol=1e-6)


def test_absorbed_attention_equals_explicit_mla(tiny):
    """Absorbed q/v over the selected rows equals textbook MLA (keys/values from kv_b, FP32)."""
    config, weights = tiny
    layer = weights.layers[0]
    contract = dataclasses.replace(config.attention_contract, top_k=8)
    rows, heads = 6, config.attention_heads
    nope, rope_dim, v_dim, lora = (
        config.qk_nope_head_dim,
        config.qk_rope_head_dim,
        config.v_head_dim,
        config.kv_lora_rank,
    )
    rng = np.random.default_rng(5)
    normalized = jnp.asarray(rng.normal(0, 1, (rows, config.hidden_size)), jnp.bfloat16)
    prepared = attention.prepare(
        normalized, layer.qkv_a, epsilon=config.rms_norm_epsilon
    )
    positions = jnp.arange(rows, dtype=jnp.int32)
    # Row r attends to every position <= r, listed in a scrambled order.
    selected = np.full((rows, 8), -1, np.int32)
    for r in range(rows):
        selected[r, : r + 1] = rng.permutation(r + 1)
    table = model.rope_table(config)
    output, cache = attention.mla_attention(
        prepared,
        positions,
        jnp.zeros((config.context_capacity, lora + rope_dim), jnp.bfloat16),
        SelectedPositions(
            jnp.asarray(selected), jnp.arange(1, rows + 1, dtype=jnp.int32)
        ),
        layer.attention,
        contract=contract,
        rope_table=table,
    )

    def rotate_hf(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        # Hugging Face apply_rotary_pos_emb_interleave: pairs (0,1),(2,3).., halves concatenated.
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return np.concatenate((x1 * cos - x2 * sin, x2 * cos + x1 * sin), axis=-1)

    rope = as_f32(table)[:rows]
    cos, sin = rope[:, : rope_dim // 2], rope[:, rope_dim // 2 :]
    q = as_f32(prepared.q_residual) @ as_f32(layer.attention.q_b).T
    q = q.reshape(rows, heads, nope + rope_dim)
    kv_b = as_f32(layer.attention.kv_b).reshape(heads, nope + v_dim, lora)
    latent = as_f32(prepared.latent)
    keys_nope = np.einsum("hdc,tc->thd", kv_b[:, :nope], latent)
    values = np.einsum("hvc,tc->thv", kv_b[:, nope:], latent)
    q_rope = rotate_hf(q[..., nope:], cos[:, None], sin[:, None])
    k_rope = rotate_hf(as_f32(prepared.key_rope_input), cos, sin)
    scores = np.einsum("rhd,thd->rht", q[..., :nope], keys_nope) + np.einsum(
        "rhd,td->rht", q_rope, k_rope
    )
    scores = scores * (nope + rope_dim) ** -0.5
    scores = np.where(
        np.arange(rows)[None, None, :] <= np.arange(rows)[:, None, None],
        scores,
        -np.inf,
    )
    probabilities = np.exp(scores - scores.max(-1, keepdims=True))
    probabilities /= probabilities.sum(-1, keepdims=True)
    attended = np.einsum("rht,thv->rhv", probabilities, values).reshape(
        rows, heads * v_dim
    )
    expected = attended @ as_f32(layer.attention.o).T
    np.testing.assert_allclose(
        as_f32(output), expected, rtol=0.05, atol=0.05 * np.abs(expected).max()
    )
    # The cache holds [latent | rope(k)] at each written position and zeros elsewhere.
    np.testing.assert_array_equal(as_f32(cache[:rows, :lora]), latent)
    assert not np.any(as_f32(cache[rows:]))
    # A malformed selection (row 0 listing the future position 3) is refused, not zeroed.
    malformed = selected.copy()
    malformed[0, 1] = 3
    with pytest.raises(ValueError, match="attention contract"):
        attention.mla_attention(
            prepared,
            positions,
            cache,
            SelectedPositions(
                jnp.asarray(malformed),
                jnp.asarray([2, 2, 3, 4, 5, 6], jnp.int32),
            ),
            layer.attention,
            contract=contract,
            rope_table=table,
        )


def test_forward_is_causal(tiny):
    """A later token never changes an earlier row's cache entries or selections (same block shape)."""
    config, weights = tiny
    prompt = [(7 * i + 3) % config.vocab_size for i in range(12)]
    changed = prompt[:-1] + [(prompt[-1] + 5) % config.vocab_size]
    rope, state = model.rope_table(config), model.initial_state(config)
    a = model.forward(config, weights, state, prompt, rope=rope)
    b = model.forward(config, weights, state, changed, rope=rope)
    for name in ("kv_cache", "index_cache"):
        x, y = as_f32(getattr(a.state, name)), as_f32(getattr(b.state, name))
        np.testing.assert_array_equal(x[:, :11], y[:, :11])
        assert not np.array_equal(x[:, 11], y[:, 11])
    for sa, sb in zip(a.selections, b.selections, strict=True):
        np.testing.assert_array_equal(
            np.asarray(sa.positions)[:11], np.asarray(sb.positions)[:11]
        )
        assert int(np.asarray(sa.positions)[-1].max()) <= 11


def test_reference_executes_only_oracle_functions(tiny):
    """A forward runs no production code: every repository function it enters is an oracle
    (``glm_tpu/optimized/reference``, moved there from ``glm_tpu/greenfield/kernels/reference`` at
    S2f)."""
    config, weights = tiny
    monitoring = sys.monitoring
    tool = next(i for i in range(6) if monitoring.get_tool(i) is None)
    root = str(Path(model.__file__).resolve().parents[2]) + "/"
    entered: set[str] = set()

    def on_start(code: Any, _offset: int) -> None:
        if code.co_filename.startswith(root):
            entered.add(code.co_filename[len(root) :])

    monitoring.use_tool_id(tool, "reference-trace")
    try:
        monitoring.register_callback(tool, monitoring.events.PY_START, on_start)
        monitoring.set_events(tool, monitoring.events.PY_START)
        model.generate(config, weights, list(range(12)), 2, block_rows=12)
    finally:
        monitoring.set_events(tool, 0)
        monitoring.register_callback(tool, monitoring.events.PY_START, None)
        monitoring.free_tool_id(tool)
    outside = sorted(
        path
        for path in entered
        if not path.startswith(
            ("tests/reference/", "glm_tpu/optimized/reference/")
        )
    )
    assert entered and not outside, outside


@pytest.mark.slow
def test_prefill_block_partition_and_decode_agree(tiny):
    """One prompt block, two blocks and token-by-token decode are the same computation."""
    config, weights = tiny
    prompt = [(5 * i + 1) % config.vocab_size for i in range(12)]
    runs = {
        "one": run_blocks(config, weights, prompt, [12]),
        "two": run_blocks(config, weights, prompt, [5, 7]),
        "tokens": run_blocks(config, weights, prompt, [1] * 12),
    }
    last = {name: results[-1] for name, results in runs.items()}
    tokens = {name: int(result.next_token) for name, result in last.items()}
    assert len(set(tokens.values())) == 1, tokens
    for name in ("two", "tokens"):
        for leaf in ("kv_cache", "index_cache"):
            np.testing.assert_allclose(
                as_f32(getattr(last[name].state, leaf)),
                as_f32(getattr(last["one"].state, leaf)),
                rtol=0.02,
                atol=0.0625,
            )
    # Every row's selection: the one-block run against the per-token runs.
    for layer in range(len(config.full_layers)):
        one = np.asarray(last["one"].selections[layer].positions)
        per_token = np.concatenate(
            [np.asarray(r.selections[layer].positions) for r in runs["tokens"]]
        )
        np.testing.assert_array_equal(np.sort(one, axis=1), np.sort(per_token, axis=1))


@pytest.mark.slow
def test_reference_reads_exactly_the_fixture_checkpoint():
    from tools.equivalence import fixture

    frozen = fixture.fixture_v1(panel_geometry=True)
    config = model.ReferenceConfig.from_geometry(
        frozen.config.geometry, fixture.config_json(), context_capacity=fixture.CAPACITY
    )
    weights = model.load_weights(frozen.arrays, config)
    assert len(weights.layers) == 8 and config.full_layers == (0, 1, 2, 6)
    assert [layer.indexer is not None for layer in weights.layers] == [
        k == "full" for k in config.indexer_types
    ]
    leaves = jax.tree.leaves(weights)
    assert all(leaf.dtype in (jnp.bfloat16, jnp.float32) for leaf in leaves)
    assert weights.layers[3].mlp.expert_gate.shape == (64, 256, 1024)
    extra = dict(
        frozen.arrays, **{"model.layers.0.unexpected": np.zeros(1, np.float32)}
    )
    with pytest.raises(ValueError, match="does not read"):
        model.load_weights(extra, config)
    with pytest.raises(ValueError, match="does not implement"):
        model.ReferenceConfig.from_geometry(
            frozen.config.geometry,
            dict(fixture.config_json(), n_group=8),
            context_capacity=fixture.CAPACITY,
        )


# ----------------------------------------------------------------------------- independent restatement
# The oracle functions the reference reuses are also executed by production and the FP8
# oracle (VALIDATION.md), so the cpu32 comparisons cannot see a bug in them; these tests
# compare the reference with tests/reference/independent.py, which shares no code with any
# of them. Tolerances are fractions of the restatement's own scale (the largest |value| of
# the compared tensor, a row's logit spread). The reference's BF16 boundaries, BF16 host
# RoPE table and FP32 accumulation give 1.5-2.1 % end to end on 12 TINY variants (top_k
# 4/6/8/32 x 3 prompts) and at most 1.2 % per component; the semantic changes of the
# VALIDATION.md mutant table move a component by 4.5 % (routed scale 2.4) to 59 %.
CONTINUOUS_TOL = 0.05
# With identical inputs into one component (no upstream noise): measured at most 1.2 %
# (the MoE on this test's input draw; 0.6-0.9 % on five other draws).
COMPONENT_TOL = 0.015
# A reference decision's regret: how far below the restatement's own top-k boundary the
# worst chosen candidate scores, given the same upstream decisions. Measured at most
# 0.0079 (DSA, as a fraction of the layer's largest |score|) and 0.0030 (router, in
# sigmoid-score units) over the 12 variants.
DECISION_TOL = 0.02
# The router's regret and margin with identical inputs into the MoE (sigmoid-score units):
# the biased scores then differ by FP32 rounding only (measured at most 1.0e-7 on six input
# draws), so the routed set must be the restatement's own top-k up to a tie of that size.
ROUTER_COMPONENT_TOL = 1e-5
INDEPENDENT_IMPORTS = frozenset(
    {"__future__", "collections", "ml_dtypes", "numpy", "typing"}
)


def test_independent_restatement_imports_no_repository_code():
    tree = ast.parse(Path(independent.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module, "no relative imports"
            imported.add(node.module.split(".")[0])
    assert imported <= INDEPENDENT_IMPORTS, imported - INDEPENDENT_IMPORTS


def reference_decisions(
    config: model.ReferenceConfig, results: list[model.ForwardResult]
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """The reference's discrete decisions per row of consecutive calls.

    Returns the attended positions of every full layer (``[rows, rows]`` bool), the
    reference's scores there (NaN elsewhere) and the routed experts of every sparse
    layer (``[rows, top_k]``); refuses a malformed selection.
    """
    rows = sum(int(result.final_residual.shape[0]) for result in results)
    selected = {layer: np.zeros((rows, rows), bool) for layer in config.full_layers}
    scores = {layer: np.full((rows, rows), np.nan) for layer in config.full_layers}
    experts = {
        layer: np.zeros((rows, config.routed_top_k), np.int64)
        for layer in config.sparse_layers
    }
    start = 0
    for result in results:
        count = int(result.final_residual.shape[0])
        for layer, chosen in zip(config.full_layers, result.selections, strict=True):
            positions, values = np.asarray(chosen.positions), np.asarray(chosen.scores)
            for j in range(count):
                row, live = start + j, positions[j] >= 0
                selected[layer][row, positions[j, live]] = True
                scores[layer][row, positions[j, live]] = values[j, live]
                assert (
                    live.sum()
                    == selected[layer][row].sum()
                    == min(config.index_top_k, row + 1)
                )
                assert not selected[layer][row, row + 1 :].any(), (
                    "selection is not causal"
                )
        for layer, routes in zip(config.sparse_layers, result.routes, strict=True):
            experts[layer][start : start + count] = np.asarray(routes.expert_ids)
        start += count
    return selected, scores, experts


def relative_error(actual: Any, expected: np.ndarray) -> float:
    return float(np.abs(as_f32(actual) - expected).max() / np.abs(expected).max())


def test_reference_matches_independent_fp64_restatement(tiny):
    """A 12-token prefill plus six decode steps against the FP64 restatement (TINY, top_k 4).

    The restatement takes the reference's discrete decisions (DSA selections, routed
    experts) after checking that each is an exact top-k of its own scores up to rounding;
    with them imposed every continuous value must agree to rounding. So a near-tie that
    rounding resolves differently cannot fail the test, and no decision can hide an error.
    """
    config, weights = tiny
    arrays = tiny_checkpoint(config)
    prompt = [(5 * i + 1) % config.vocab_size for i in range(12)]
    results = run_blocks(config, weights, prompt, [12])
    rope, tokens = model.rope_table(config), [int(results[-1].next_token)]
    for _ in range(6):
        results.append(
            model.forward(config, weights, results[-1].state, [tokens[-1]], rope=rope)
        )
        tokens.append(int(results[-1].next_token))
    sequence = prompt + tokens[:-1]
    rows = len(sequence)
    selected, reference_scores, experts = reference_decisions(config, results)
    exact = independent.forward(
        config,
        independent.load(arrays, config.fp8_block_shape),
        sequence,
        selected=selected,
        expert_ids=experts,
    )

    # 1. Each decision is the restatement's own top-k up to rounding (ties: any order).
    for layer in config.full_layers:
        scores = exact.index_scores[layer]
        scale = np.abs(scores[np.isfinite(scores)]).max()
        for row in range(rows):
            chosen = scores[row, selected[layer][row]]
            boundary = np.sort(scores[row, : row + 1])[::-1][chosen.size - 1]
            assert boundary - chosen.min() <= DECISION_TOL * scale, (layer, row)
            np.testing.assert_allclose(
                reference_scores[layer][row, selected[layer][row]],
                chosen,
                rtol=0,
                atol=CONTINUOUS_TOL * scale,
                err_msg=f"DSA scores, layer {layer}, row {row}",
            )
    for layer in config.sparse_layers:
        choice = exact.router_choice[layer]
        boundary = np.sort(choice, axis=1)[:, -config.routed_top_k]
        worst = np.take_along_axis(choice, experts[layer], axis=1).min(axis=1)
        assert (boundary - worst).max() <= DECISION_TOL, (layer, boundary - worst)

    # 2. With the same decisions every continuous value agrees to rounding.
    state = results[-1].state
    kv, index = state.kv_cache[:, :rows], state.index_cache[:, :rows]
    lora = config.kv_lora_rank
    errors = {}
    for layer in range(config.num_layers):
        errors[f"latent {layer}"] = relative_error(
            kv[layer, :, :lora], exact.latent[layer]
        )
        errors[f"key rope {layer}"] = relative_error(
            kv[layer, :, lora:], exact.key_rope[layer]
        )
    for slot, layer in enumerate(config.full_layers):
        errors[f"index keys {layer}"] = relative_error(
            index[slot], exact.index_keys[layer]
        )
    errors["final residual"] = relative_error(
        jnp.concatenate([result.final_residual for result in results]), exact.residual
    )
    for step, result in enumerate(results):
        expected = exact.logits[len(prompt) - 1 + step]
        error = np.abs(as_f32(result.logits) - expected)
        errors[f"logits {step}"] = float(error.max() / np.ptp(expected))
        # Greedy: the reference's token is the restatement's unless the two are within error.
        best = int(np.argmax(expected))
        assert expected[best] - expected[tokens[step]] <= 2 * error.max(), (step, best)
    assert max(errors.values()) <= CONTINUOUS_TOL, {
        k: round(v, 4) for k, v in errors.items() if v > CONTINUOUS_TOL
    }


def test_layer_components_match_independent_fp64_restatement(tiny):
    """Identical BF16 inputs into each reference component and the restatement's.

    Without upstream noise the reference is within a few BF16 roundings (at most 1.2 %
    measured); a 4 % change of the routed scale moves the MoE by 3.2-4.8 % (six input
    draws, 4.5 % on this one), ignoring the norm weights moves the norm by 6-10 %. The
    routed set must be the restatement's own top-k up to FP32 rounding.
    """
    config, weights = tiny
    exact_weights = independent.load(tiny_checkpoint(config), config.fp8_block_shape)
    rows, rng, eps = 10, np.random.default_rng(100), config.rms_norm_epsilon
    update = jnp.asarray(rng.normal(0, 1, (rows, config.hidden_size)), jnp.bfloat16)
    residual = jnp.asarray(rng.normal(0, 1, (rows, config.hidden_size)), jnp.bfloat16)
    # A random causal DSA selection per row (score order is irrelevant to attention).
    positions = np.full((rows, config.index_top_k), -1, np.int32)
    attended = np.zeros((rows, rows), bool)
    for row in range(rows):
        count = min(config.index_top_k, row + 1)
        positions[row, :count] = rng.choice(row + 1, count, replace=False)
        attended[row, positions[row, :count]] = True
    selection = SelectedPositions(
        jnp.asarray(positions),
        jnp.minimum(jnp.arange(1, rows + 1), config.index_top_k).astype(jnp.int32),
    )
    errors = {}
    for layer in (0, 1):  # dense MLP, then sparse MoE
        p, a = f"model.layers.{layer}", f"model.layers.{layer}.self_attn"
        layer_weights = weights.layers[layer]
        normalized, _ = add_rms_norm(
            update, residual, layer_weights.input_norm, epsilon=eps
        )
        x = as_f32(normalized).astype(np.float64)
        errors[f"norm {layer}"] = relative_error(
            normalized,
            independent.rms_norm(
                as_f32(update).astype(np.float64) + as_f32(residual),
                exact_weights[p + ".input_layernorm.weight"],
                eps,
            ),
        )
        prepared = attention.prepare(normalized, layer_weights.qkv_a, epsilon=eps)
        inputs = independent.attention_inputs(x, exact_weights, a, config)
        output, cache = attention.mla_attention(
            prepared,
            jnp.arange(rows, dtype=jnp.int32),
            model.initial_state(config).kv_cache[layer],
            selection,
            layer_weights.attention,
            contract=config.attention_contract,
            rope_table=model.rope_table(config),
        )
        errors[f"q_resid {layer}"] = relative_error(prepared.q_residual, inputs.q_resid)
        errors[f"latent {layer}"] = relative_error(
            cache[:rows, : config.kv_lora_rank], inputs.latent
        )
        errors[f"key rope {layer}"] = relative_error(
            cache[:rows, config.kv_lora_rank :], inputs.key_rope
        )
        errors[f"attention {layer}"] = relative_error(
            output, independent.attention(inputs, attended, exact_weights, a, config)
        )
        if isinstance(layer_weights.mlp, moe.MoeWeights):
            actual, routes = moe.moe(
                normalized,
                layer_weights.mlp,
                top_k=config.routed_top_k,
                routed_scaling_factor=config.routed_scaling_factor,
            )
            expected, choice, ids = independent.sparse_moe(
                x, exact_weights, p + ".mlp", config, np.asarray(routes.expert_ids)
            )
            ordered = np.sort(choice, axis=1)
            boundary = ordered[:, -config.routed_top_k]
            worst = np.take_along_axis(choice, ids, axis=1).min(axis=1)
            regret = boundary - worst
            assert regret.max() <= ROUTER_COMPONENT_TOL, regret
            np.testing.assert_allclose(
                np.asarray(routes.margin),
                boundary - ordered[:, -config.routed_top_k - 1],
                rtol=0,
                atol=ROUTER_COMPONENT_TOL,
            )
        else:
            actual = moe.dense_mlp(normalized, layer_weights.mlp)
            expected = independent.swiglu(x, exact_weights, p + ".mlp")
        errors[f"mlp {layer}"] = relative_error(actual, expected)
    assert max(errors.values()) <= COMPONENT_TOL, {
        k: round(v, 4) for k, v in errors.items() if v > COMPONENT_TOL
    }


def test_indexer_matches_independent_fp64_restatement(tiny):
    """Identical BF16 inputs into the reference indexer and the restatement's (24 rows, top_k 4)."""
    config, weights = tiny
    exact_weights = independent.load(tiny_checkpoint(config), config.fp8_block_shape)
    rows, rng = 24, np.random.default_rng(13)
    positions = jnp.arange(rows, dtype=jnp.int32)
    contract = config.indexer_contract
    for layer in config.full_layers:
        normalized = jnp.asarray(
            rng.normal(0, 1, (rows, config.hidden_size)), jnp.bfloat16
        )
        q_residual = jnp.asarray(
            rng.normal(0, 1, (rows, config.q_lora_rank)), jnp.bfloat16
        )
        indexer = weights.layers[layer].indexer
        keys = dsa.index_keys(normalized, indexer, positions, contract=contract)
        cache = jnp.zeros(
            (config.context_capacity, config.index_head_dim), jnp.bfloat16
        )
        chosen = dsa.select(
            normalized,
            q_residual,
            cache.at[positions].set(keys),
            positions,
            indexer,
            contract=contract,
        )
        exact_keys, scores = independent.indexer(
            as_f32(normalized).astype(np.float64),
            as_f32(q_residual).astype(np.float64),
            exact_weights,
            f"model.layers.{layer}.self_attn.indexer",
            config,
        )
        # The cache stores the key rounded to BF16: one rounding.
        np.testing.assert_allclose(
            as_f32(keys), exact_keys, rtol=2**-8, atol=1e-3 * np.abs(exact_keys).max()
        )
        scale = np.abs(scores[np.isfinite(scores)]).max()
        selection = [p[p >= 0] for p in np.asarray(chosen.positions)]
        np.testing.assert_array_equal(
            np.asarray(chosen.valid_counts), np.minimum(np.arange(1, rows + 1), 4)
        )
        for row, picked in enumerate(selection):
            assert len(set(picked)) == picked.size and picked.max() <= row
            boundary = np.sort(scores[row, : row + 1])[::-1][picked.size - 1]
            assert boundary - scores[row, picked].min() <= 0.01 * scale, (layer, row)
            np.testing.assert_allclose(
                np.asarray(chosen.scores)[row, : picked.size],
                scores[row, picked],
                rtol=0,
                atol=0.01 * scale,
            )


# ----------------------------------------------------------------------------- cross-validation (cpu32)
# Floors: how far production was from the frozen FP8 oracle on the same prompt and schedule
# (VALIDATION.md), recorded from the oracle's final run at S2f in ``floors.json``. Where even the two accepted engines break a DESIGN 7.6 criterion, the
# reference must be no further from either engine than they are from each other. These
# are a recorded deviation from the DESIGN 7.6 gate (VALIDATION.md); they are dominated by
# the rows a flipped decision moves, so they bound the reference loosely and the semantic
# checks are the independent-restatement tests above. Measured worst on the four pairs
# (VALIDATION.md): bound ratio 1.087 x the floor, outside fraction 1.71 x the floor,
# carried-layer set-unequal steps equal to the floor's.
FLOOR_SLACK = 1.15  # worst |difference| / bound, a maximum over ~1e6-1e7 elements
FRACTION_SLACK = 2.0  # elements outside the bound (a moved row moves as a whole)


FLOORS = Path(__file__).with_name("floors.json")


def recorded_floor(prompt: str) -> dict[str, Any]:
    """The production:fp8-oracle report of ``prompt`` as far as the criteria read it."""
    import json

    return json.loads(FLOORS.read_text())["floors"][prompt]


@pytest.fixture(scope="module")
def reports() -> Any:
    from tools.equivalence.common import run_child

    cache: dict[tuple[str, str], dict[str, Any]] = {}

    def get(pair: str, prompt: str) -> dict[str, Any]:
        if (pair, prompt) not in cache:
            cache[pair, prompt] = run_child(
                "tests.reference.oracle_run",
                "--pair",
                pair,
                "--prompt",
                prompt,
                timeout=1800,
            )
        return cache[pair, prompt]

    return get


def assert_criteria(
    report: dict[str, Any], floor: dict[str, Any], *, strict: tuple[str, ...]
) -> None:
    s, f = report["summary"], floor["summary"]
    assert report["engine_inputs_equal_reference"] == dict(wk=True, rope=True)
    failed = [key for key in strict if not s[key]]
    assert not failed, (failed, s)
    for phase in ("prefill", "decode"):
        for leaf, ratio in s[phase + "_bound_ratio"].items():
            # Within the bound wherever the engines are; otherwise within the engines' own distance.
            engines = f[phase + "_bound_ratio"][leaf]
            assert ratio <= (1.0 if engines <= 1.0 else FLOOR_SLACK * engines), (
                phase,
                leaf,
                ratio,
                f,
            )
        for leaf, fraction in s[phase + "_outside_fraction"].items():
            # A zero floor fraction means the engines are within the bound: zero allowed.
            assert fraction <= FRACTION_SLACK * f[phase + "_outside_fraction"][leaf], (
                phase,
                leaf,
                fraction,
                f,
            )
    carried = [
        r["selections"][str(max(map(int, r["selections"])))] for r in report["decode"]
    ]
    floor_carried = [next(iter(r["selections"].values())) for r in floor["decode"]]
    assert sum(not c["set_equal"] for c in carried) <= sum(
        not c["set_equal"] for c in floor_carried
    )


SHORT = (
    "prefill_integer_leaves_equal",
    "prefill_float_leaves_within",
    "prefill_next_token_equal",
    "decode_tokens_equal",
    "decode_integer_state_equal",
    "decode_selections_set_equal",
    "decision_free_within",
)
PROMPT_A = (
    "prefill_integer_state_equal",
    "prefill_next_token_equal",
    "decode_integer_state_equal",
    "decision_free_within",
    "tokens_equal_or_tied",
    "selections_explained",
)


@pytest.mark.slow
@pytest.mark.cpu32
def test_reference_matches_production_composition(reports):
    assert_criteria(
        reports("reference:production", "short"),
        recorded_floor("short"),
        strict=SHORT,
    )
    assert_criteria(
        reports("reference:production", "a"),
        recorded_floor("a"),
        strict=PROMPT_A,
    )


def test_recorded_floors_are_the_final_oracle_runs():
    import json

    value = json.loads(FLOORS.read_text())
    assert value["pair"] == "production:fp8-oracle" and set(value["floors"]) == {"short", "a"}
    for floor in value["floors"].values():
        assert set(floor["summary"]) == {
            "prefill_bound_ratio", "prefill_outside_fraction", "decode_bound_ratio", "decode_outside_fraction"}
        assert floor["decode"] and all(len(r["selections"]) == 1 for r in floor["decode"])
