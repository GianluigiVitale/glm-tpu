"""The unsharded reference model: unit semantics and cross-validation (DESIGN 7.6, S2e).

Fast tests run the reference alone on one CPU device over a tiny synthetic
checkpoint (the fixture's tensor names, small dimensions). The ``slow`` test
reads the frozen fixture v1. The ``cpu32`` tests run
:mod:`tests.reference.oracle_run` in a child with 32 forced CPU devices and
assert the acceptance criteria of ``VALIDATION.md``: every DESIGN 7.6 criterion
the two accepted engines (frozen FP8 oracle, production) meet against each other
holds exactly for the reference; for the others the reference is no further from
either engine than production is from the FP8 oracle (the measured floor).
``test_reference_matches_frozen_fp8_oracle`` is archived together with the
frozen FP8 oracle (S2f); ``VALIDATION.md`` is its receipt.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from glm_tpu.greenfield.kernels.reference.dsa import SelectedPositions
from tests.reference import attention, dsa, model, moe
from tests.reference.linear import dequantize, greedy_token

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
            arrays[p + ".mlp.gate.e_score_correction_bias"] = rng.normal(
                0, 0.01, e
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
    """A forward runs no production code: every repository function it enters is an oracle.

    (Importing the reference still loads production modules through the
    ``glm_tpu.greenfield.kernels`` package initializer; VALIDATION.md lists them.)
    """
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
            ("tests/reference/", "glm_tpu/greenfield/kernels/reference/")
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


# ----------------------------------------------------------------------------- cross-validation (cpu32)
# Floors: how far production is from the frozen FP8 oracle on the same prompt and schedule
# (VALIDATION.md). Where even the two accepted engines break a DESIGN 7.6 criterion, the
# reference must be no further from either engine than they are from each other.
FLOOR_SLACK = 1.25  # max-type metrics (extreme values of ~1e6 elements): the observed worst is 1.09
FRACTION_SLACK = 2.0  # fraction of elements outside the float bound; plus 1e-5 absolute


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
            assert (
                fraction <= FRACTION_SLACK * f[phase + "_outside_fraction"][leaf] + 1e-5
            ), (phase, leaf, fraction, f)
    carried = [
        r["selections"][str(max(map(int, r["selections"])))] for r in report["decode"]
    ]
    floor_carried = [next(iter(r["selections"].values())) for r in floor["decode"]]
    assert (
        sum(not c["set_equal"] for c in carried)
        <= sum(not c["set_equal"] for c in floor_carried) + 1
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
def test_reference_matches_frozen_fp8_oracle(reports):
    assert_criteria(
        reports("reference:fp8-oracle", "short"),
        reports("production:fp8-oracle", "short"),
        strict=SHORT,
    )
    long = reports("reference:fp8-oracle", "a")
    assert_criteria(
        long,
        reports("production:fp8-oracle", "a"),
        strict=PROMPT_A + ("decode_tokens_equal",),
    )


@pytest.mark.slow
@pytest.mark.cpu32
def test_reference_matches_production_composition(reports):
    assert_criteria(
        reports("reference:production", "short"),
        reports("production:fp8-oracle", "short"),
        strict=SHORT,
    )
    assert_criteria(
        reports("reference:production", "a"),
        reports("production:fp8-oracle", "a"),
        strict=PROMPT_A,
    )
