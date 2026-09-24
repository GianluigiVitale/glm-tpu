"""Bitwise shortlist proof including ties, skew and forced fallback."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.layers.attention.dsa_indexer import (
    dsa_index_keys_from_projection,
    dsa_scores,
    local_topk_candidates,
    merge_topk_candidates_with_scores,
)
from glm_tpu.layers.contracts import DsaNumericalContract
from tests.reference.dsa import (
    distributed_exact_topk_reference,
    dsa_index_keys,
    dsa_query_and_head_weights,
    exact_topk,
    merge_topk_candidates,
)
from tests.reference.linear import linear


def test_two_stage_matches_frozen_cpu8():
    code = r"""
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, PartitionSpec as P
from glm_tpu.layers.attention.dsa_indexer import local_topk_candidates, merge_topk_candidates_with_scores, ScoredSelectedPositions
from glm_tpu.layers.attention.dsa_indexer import two_stage_topk
mesh=Mesh(np.asarray(jax.devices(), object), ('expert',))
def body(scores, positions, lengths):
    scores, positions = scores[0], positions[0]
    a, fallback = two_stage_topk(scores, positions, lengths, top_k=16, global_context_size=256, candidates_per_owner=4)
    v, p = local_topk_candidates(scores, positions, lengths, top_k=16)
    b = merge_topk_candidates_with_scores(jax.lax.all_gather(v,'expert'), jax.lax.all_gather(p,'expert'), lengths, top_k=16, global_context_size=256)
    return a,b,fallback
fn=jax.jit(jax.shard_map(body,mesh=mesh,in_specs=(P('expert'),P('expert'),P()),out_specs=(ScoredSelectedPositions(P(),P(),P()),ScoredSelectedPositions(P(),P(),P()),P()),check_vma=False))
rng=np.random.default_rng(334)
positions=np.arange(256,dtype=np.int32).reshape(8,32)
paths=set()
for trial in range(60):
    perm=np.stack([rng.permutation(32) for _ in range(8)])
    p=np.take_along_axis(positions,perm,axis=1)
    # Integers exercise exact score ties across and inside owners.
    scores=rng.integers(-8,9,size=(8,3,32)).astype(np.float32)
    if trial%3==0: scores[0]+=100
    if trial%3==1: scores.fill(0)
    lengths=np.array([0,7 if trial%2 else 64,256],np.int32)
    if trial%3==2: lengths=np.array([0,256,256],np.int32)
    if trial == 59:
        # Stripes keep every other owner's omitted -0 below the cutoff.
        # Only the omitted +0 on owner 7 forces fallback; float == misses it.
        p=np.arange(256,dtype=np.int32).reshape(32,8).T
        scores.fill(-0.0)
        scores[7,:,:20] = 0.0
        lengths[:] = 256
    a,b,f=fn(jnp.asarray(scores),jnp.asarray(p),jnp.asarray(lengths))
    for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
    np.testing.assert_array_equal(np.asarray(a.scores).view(np.uint32), np.asarray(b.scores).view(np.uint32))
    paths.add(bool(f))
    if trial == 59: assert bool(f), "signed-zero cut must fall back"
assert paths=={False,True}, paths
print('60 randomized tied/skewed trials, both cut-check branches, bitwise equal')
"""
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=8")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_physical_page_scores_match_logical_key_gather_bitwise():
    import jax.numpy as jnp
    import numpy as np
    from glm_tpu.layers.contracts import StageLocalKvLayout
    from glm_tpu.layers.attention.dsa_indexer import dsa_scores
    from glm_tpu.layers.attention.dsa_indexer import score_cache_pages

    rng = np.random.default_rng(91)
    layout = StageLocalKvLayout(logical_page_size=128, local_parallel_size=8, packed_cache_width=128)
    query = jnp.asarray(rng.normal(size=(3, 4, 128)), jnp.float32)
    cache = jnp.asarray(rng.normal(size=(4, 16, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.normal(size=(3, 4)), jnp.float32)
    tables = jnp.array([[3, 1, -1, 0]], jnp.int32)
    actual, positions = score_cache_pages(query, cache, weights, tables, layout=layout, owner=jnp.int32(5))
    logical = jnp.take(cache, jnp.clip(tables[0], 0, 3), axis=0).reshape(-1, 128)
    expected = dsa_scores(query, logical, weights, precision="highest")
    np.testing.assert_array_equal(actual, expected)
    expected_positions = np.arange(4)[:, None] * 128 + 5 * 16 + np.arange(16)[None]
    expected_positions[2] = -1
    np.testing.assert_array_equal(positions, expected_positions.reshape(-1))


# Leaf digests (sha256 of dtype | shape | bytes; positions, scores, counts, per-owner health) of the
# frozen tiled prefill DSA selector on the inputs below, recorded from its final run at S2f (it is
# archived at archive/research-20260922); the production one-pass selector produced the same leaves
# bitwise in that run. Recorded with jax/jaxlib 0.10.1 on CPU (the G3 environment).
FROZEN_TILED_SELECTOR = {
    "0,127,2048": (
        "335e89b56642ea31f7442ebceefede8b3eff87667bcb68e460a0850a79c32e07",
        "e30ebfb3327e69ff2cc0cda1417f20113af4ebc48307befb9e311584b7b457d2",
        "8875abbe17e828ccca774f63180a2af2216d2bd09f84a78d3418bc431f816253",
        "6a2cf1002bbb66778d21b8a7b414afd4a09a25dfb2e48269413e8ac2ce195a6b",
    ),
    "1,511,1984": (
        "f8debe139d164bc50eeefbb39284944f42f33d216d894a16e584a498551a2c99",
        "1cc9001ad1d0a726edf93136322468d56bfa021701dd2425e1bf8605010bf391",
        "9c44e32049f0fe6f3a7cb37ff5323a0319232b3549a0a42055119ed20df3f890",
        "6a2cf1002bbb66778d21b8a7b414afd4a09a25dfb2e48269413e8ac2ce195a6b",
    ),
    "0,0,0": (
        "987aaa266b7b9d7ed646e3482f653c99441a6b4ed0647d34b61cf9f616237043",
        "46c0e0c6c0f5616da54ece94b4569c47292a9229f24d183e523df40b2f7e4c62",
        "7f2e106fcfd5fabb886a2f7ceb6b016ae230ae25aec3dd2fa31de827ffe55c39",
        "6a2cf1002bbb66778d21b8a7b414afd4a09a25dfb2e48269413e8ac2ce195a6b",
    ),
}


def test_one_pass_prefill_matches_the_recorded_tiled_selector_cpu32():
    code = r"""
import hashlib, json
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, PartitionSpec as P
from glm_tpu.layers.attention.dsa_indexer import ScoredSelectedPositions
from glm_tpu.layers.attention.dsa_indexer import prefill_dsa_one_pass
mesh=Mesh(np.asarray(jax.devices(), object).reshape(8,4), ('expert','feature'))
def body(q,k,w,p,lengths):
    a=prefill_dsa_one_pass(q,k[0],w,p[0],lengths,candidates_per_owner=16,global_context_size=2048,top_k=64)
    # Health is owner-local, expose one element per owner.
    return a[0],a[1][None]
fn=jax.jit(jax.shard_map(body,mesh=mesh,in_specs=(P(),P('expert'),P(),P('expert'),P()),
                         out_specs=(ScoredSelectedPositions(P(),P(),P()),P('expert')),check_vma=False))
rng=np.random.default_rng(36)
q=jnp.asarray(rng.normal(size=(3,32,128)),jnp.float32)
k=jnp.asarray(rng.normal(size=(8,256,128)),jnp.bfloat16)
w=jnp.asarray(rng.normal(size=(3,32)),jnp.float32)
p=np.stack([np.arange(256,dtype=np.int32)*8+i for i in range(8)])
def digest(x):
    x=np.asarray(x); return hashlib.sha256(f"{x.dtype}|{x.shape}|".encode()+x.tobytes()).hexdigest()
out={}
for lengths in ([0,127,2048],[1,511,1984],[0,0,0]):
    out[",".join(map(str,lengths))]=[digest(x) for x in jax.tree.leaves(fn(q,k,w,jnp.asarray(p),jnp.array(lengths,jnp.int32)))]
print(json.dumps(out))
"""
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=32")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = json.loads(result.stdout.strip().splitlines()[-1])
    assert actual == {key: list(value) for key, value in FROZEN_TILED_SELECTOR.items()}


def small_contract() -> DsaNumericalContract:
    return DsaNumericalContract(
        hidden_size=6,
        q_lora_rank=4,
        num_heads=2,
        head_dim=4,
        rotary_dim=2,
        top_k=4,
        theta=100.0,
    )


def test_glm_dsa_contract_is_exact() -> None:
    contract = DsaNumericalContract()
    assert contract.hidden_size == 6144
    assert contract.q_lora_rank == 2048
    assert contract.num_heads == 32
    assert contract.head_dim == 128
    assert contract.rotary_dim == 64
    assert contract.top_k == 2048
    assert contract.theta == 8_000_000.0
    assert contract.padding_sentinel == -1


def test_dsa_contract_refuses_semantic_drift() -> None:
    with pytest.raises(ValueError, match="rotary_dim"):
        DsaNumericalContract(head_dim=4, rotary_dim=6)
    with pytest.raises(ValueError, match="FP32"):
        DsaNumericalContract(score_dtype="bfloat16")
    with pytest.raises(ValueError, match="interleaved"):
        DsaNumericalContract(interleaved_rotary=False)
    with pytest.raises(ValueError, match="tie policy"):
        DsaNumericalContract(tie_policy="score_only")
    with pytest.raises(ValueError, match="sentinel"):
        DsaNumericalContract(padding_sentinel=0)


def test_query_key_and_score_math_matches_direct_fp32_formula() -> None:
    contract = small_contract()
    hidden_query = jnp.asarray([[0.5, -1.0, 0.25, 0.75, -0.5, 1.25]], dtype=jnp.bfloat16)
    q_residual = jnp.asarray([[0.5, -0.25, 1.0, 0.75]], dtype=jnp.bfloat16)
    hidden_keys = jnp.asarray(
        [
            [0.5, 0.25, -0.5, 1.0, -1.0, 0.75],
            [-0.25, 1.0, 0.5, -0.75, 0.25, 1.25],
            [1.0, -0.5, 0.75, 0.25, 0.5, -1.0],
        ],
        dtype=jnp.bfloat16,
    )
    query_weight = (jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 31.0 - 0.5).astype(jnp.bfloat16)
    head_weight = jnp.asarray(
        [[0.5, 0.25, -0.5, 1.0, 0.0, -0.25], [-0.5, 0.75, 0.25, 0.0, 1.0, 0.5]],
        dtype=jnp.bfloat16,
    )
    key_weight = (jnp.arange(24, dtype=jnp.float32).reshape(4, 6) / 23.0 - 0.5).astype(jnp.bfloat16)
    norm_weight = jnp.asarray([1.0, 0.5, 1.5, -0.5], dtype=jnp.float32)
    norm_bias = jnp.asarray([0.0, 0.1, -0.2, 0.3], dtype=jnp.float32)
    query, weights = dsa_query_and_head_weights(
        hidden_query,
        q_residual,
        query_weight,
        head_weight,
        jnp.asarray([2], dtype=jnp.int32),
        contract=contract,
    )
    keys = dsa_index_keys(
        hidden_keys,
        key_weight,
        norm_weight,
        norm_bias,
        jnp.asarray([0, 1, 2], dtype=jnp.int32),
        contract=contract,
    )
    projected_keys = linear(hidden_keys, key_weight, output_dtype=jnp.float32)
    keys_from_projection = dsa_index_keys_from_projection(
        projected_keys,
        norm_weight,
        norm_bias,
        jnp.asarray([0, 1, 2], dtype=jnp.int32),
        contract=contract,
    )
    got = dsa_scores(query, keys, weights)
    expected_per_head = np.maximum(
        np.einsum("rhd,sd->rhs", np.asarray(query), np.asarray(keys)) * contract.head_dim**-0.5,
        0.0,
    )
    expected = np.einsum("rh,rhs->rs", np.asarray(weights), expected_per_head)
    assert query.shape == (1, 2, 4)
    assert keys.shape == (3, 4)
    assert got.shape == (1, 3)
    assert got.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(keys_from_projection), np.asarray(keys))
    np.testing.assert_allclose(np.asarray(got), expected, rtol=2e-7, atol=2e-7)


def test_index_key_norm_exposes_exact_divide_sqrt_association() -> None:
    contract = small_contract()
    projected = jnp.asarray([[0.125, -0.75, 1.25, 0.5]], dtype=jnp.float32)
    weight = jnp.asarray([1.0, 0.5, -0.25, 1.5], dtype=jnp.float32)
    bias = jnp.asarray([0.0, 0.1, -0.2, 0.3], dtype=jnp.float32)
    positions = jnp.asarray([0], dtype=jnp.int32)

    default = dsa_index_keys_from_projection(projected, weight, bias, positions, contract=contract)
    reciprocal = dsa_index_keys_from_projection(
        projected,
        weight,
        bias,
        positions,
        contract=contract,
        key_norm_mode="multiply_rsqrt",
    )
    divided = dsa_index_keys_from_projection(
        projected,
        weight,
        bias,
        positions,
        contract=contract,
        key_norm_mode="divide_sqrt",
    )
    np.testing.assert_array_equal(np.asarray(default), np.asarray(reciprocal))
    centered = np.asarray(projected) - np.asarray(projected).mean(axis=-1, keepdims=True)
    expected = centered / np.sqrt(
        np.mean(centered * centered, axis=-1, keepdims=True) + contract.key_layer_norm_epsilon
    )
    expected = expected * np.asarray(weight) + np.asarray(bias)
    np.testing.assert_allclose(np.asarray(divided), expected, rtol=2e-7, atol=2e-7)

    with pytest.raises(ValueError, match="unknown key LayerNorm association"):
        dsa_index_keys_from_projection(
            projected,
            weight,
            bias,
            positions,
            contract=contract,
            key_norm_mode="unknown",  # type: ignore[arg-type]
        )


def test_dsa_scorer_pins_highest_dot_precision() -> None:
    query = jnp.ones((1, 2, 4), dtype=jnp.float32)
    keys = jnp.ones((3, 4), dtype=jnp.float32)
    weights = jnp.ones((1, 2), dtype=jnp.float32)
    hlo = jax.jit(dsa_scores).lower(query, keys, weights).compile().as_text()
    assert "operand_precision={highest,highest}" in hlo

    default_hlo = (
        jax.jit(lambda q, k, w: dsa_scores(q, k, w, precision="default"))
        .lower(query, keys, weights)
        .compile()
        .as_text()
    )
    assert "operand_precision={highest,highest}" not in default_hlo


def test_exact_topk_has_lowest_position_ties_and_minus_one_tail() -> None:
    scores = jnp.asarray(
        [[5.0, 5.0, 4.0, 3.0, 2.0], [9.0, 8.0, 7.0, 6.0, 5.0]],
        dtype=jnp.float32,
    )
    selected = exact_topk(scores, jnp.asarray([5, 2], dtype=jnp.int32), top_k=4)
    np.testing.assert_array_equal(np.asarray(selected.positions), [[0, 1, 2, 3], [0, 1, -1, -1]])
    np.testing.assert_array_equal(np.asarray(selected.valid_counts), [4, 2])


def test_exact_topk_pads_context_shorter_than_selection_width() -> None:
    selected = exact_topk(
        jnp.asarray([[3.0, 1.0]], dtype=jnp.float32),
        jnp.asarray([2], dtype=jnp.int32),
        top_k=4,
    )
    np.testing.assert_array_equal(np.asarray(selected.positions), [[0, 1, -1, -1]])
    np.testing.assert_array_equal(np.asarray(selected.valid_counts), [2])
    empty = exact_topk(
        jnp.asarray([[3.0, 1.0]], dtype=jnp.float32),
        jnp.asarray([-1], dtype=jnp.int32),
        top_k=4,
    )
    np.testing.assert_array_equal(np.asarray(empty.positions), [[-1, -1, -1, -1]])
    np.testing.assert_array_equal(np.asarray(empty.valid_counts), [0])


def test_distributed_selection_matches_flat_including_ties_and_group_order() -> None:
    flat_scores = jnp.asarray(
        [[10.0, 10.0, 8.0, 8.0, 7.0, 7.0, 6.0, 6.0]],
        dtype=jnp.float32,
    )
    expected = exact_topk(flat_scores, jnp.asarray([8], dtype=jnp.int32), top_k=5)
    # Two striped context owners, deliberately presented in reverse owner order.
    shard_positions = jnp.asarray([[1, 3, 5, 7], [0, 2, 4, 6]], dtype=jnp.int32)
    shard_scores = jnp.stack((flat_scores[:, [1, 3, 5, 7]], flat_scores[:, [0, 2, 4, 6]]), axis=0)
    got = distributed_exact_topk_reference(
        shard_scores,
        shard_positions,
        jnp.asarray([8], dtype=jnp.int32),
        top_k=5,
        global_context_size=8,
    )
    np.testing.assert_array_equal(np.asarray(got.positions), np.asarray(expected.positions))
    np.testing.assert_array_equal(np.asarray(got.valid_counts), np.asarray(expected.valid_counts))


def test_local_candidate_width_keeps_hot_shard_winners() -> None:
    scores = jnp.asarray([[100.0, 99.0, 98.0, 97.0, 1.0]], dtype=jnp.float32)
    values, positions = local_topk_candidates(
        scores,
        jnp.asarray([0, 2, 4, 6, 8], dtype=jnp.int32),
        jnp.asarray([9], dtype=jnp.int32),
        top_k=4,
    )
    np.testing.assert_array_equal(np.asarray(positions), [[0, 2, 4, 6]])
    np.testing.assert_array_equal(np.asarray(values), [[100.0, 99.0, 98.0, 97.0]])


def test_local_ties_use_global_position_not_input_order() -> None:
    values, positions = local_topk_candidates(
        jnp.asarray([[5.0, 5.0, 5.0, 4.0]], dtype=jnp.float32),
        jnp.asarray([7, 1, 3, 5], dtype=jnp.int32),
        jnp.asarray([8], dtype=jnp.int32),
        top_k=2,
    )
    np.testing.assert_array_equal(np.asarray(positions), [[1, 3]])
    np.testing.assert_array_equal(np.asarray(values), [[5.0, 5.0]])


def test_merge_is_invariant_to_candidate_concatenation_order() -> None:
    scores = jnp.asarray([[[5.0, 4.0, 3.0]], [[5.0, 4.0, 3.0]]], dtype=jnp.float32)
    positions = jnp.asarray([[[1, 3, 5]], [[0, 2, 4]]], dtype=jnp.int32)
    valid = jnp.asarray([6], dtype=jnp.int32)
    first = merge_topk_candidates(scores, positions, valid, top_k=4, global_context_size=6)
    second = merge_topk_candidates(scores[::-1], positions[::-1], valid, top_k=4, global_context_size=6)
    np.testing.assert_array_equal(np.asarray(first.positions), [[0, 1, 2, 3]])
    np.testing.assert_array_equal(np.asarray(second.positions), np.asarray(first.positions))
    scored = merge_topk_candidates_with_scores(scores, positions, valid, top_k=4, global_context_size=6)
    np.testing.assert_array_equal(np.asarray(scored.positions), [[0, 1, 2, 3]])
    np.testing.assert_array_equal(np.asarray(scored.scores), [[5.0, 5.0, 4.0, 4.0]])


def test_decode_reference_has_one_live_row_not_batch_32() -> None:
    selected = jax.jit(lambda value: exact_topk(value, jnp.asarray([7]), top_k=4))(
        jnp.arange(7, dtype=jnp.float32)[None, :]
    )
    assert selected.positions.shape == (1, 4)
    assert selected.valid_counts.shape == (1,)


def test_dsa_shape_contracts_fail_loudly() -> None:
    contract = small_contract()
    with pytest.raises(ValueError, match="query_weight"):
        dsa_query_and_head_weights(
            jnp.ones((1, 6)),
            jnp.ones((1, 4)),
            jnp.ones((7, 4)),
            jnp.ones((2, 6)),
            jnp.asarray([0]),
            contract=contract,
        )
    with pytest.raises(ValueError, match="ranks"):
        dsa_scores(jnp.ones((1, 4)), jnp.ones((3, 4)), jnp.ones((1, 2)))
    with pytest.raises(ValueError, match="positive"):
        exact_topk(jnp.ones((1, 4)), jnp.asarray([4]), top_k=0)


def test_contract_can_change_static_fixture_sizes_without_changing_semantics() -> None:
    contract = replace(small_contract(), top_k=2)
    assert contract.tie_policy == "descending_score_then_lowest_global_position"
    assert contract.interleaved_rotary is True
