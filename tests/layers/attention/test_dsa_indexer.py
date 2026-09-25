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
    prompt_index_key_chunk,
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
"""  # noqa: E501 (child program text)
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


@pytest.mark.cpu32
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
"""  # noqa: E501 (child program text)
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


# ---------------------------------------------------------------------------------------------- prefill (cpu32)
# Ported from the research package's kernels/test_ws32_prefill_dsa.py, test_ws32_prefill_dsa_producer.py and the DSA
# part of test_prefill_index.py (archive/research-20260922; tools/migration/archive_list.txt) onto the production
# prefill indexer.
SELECTOR = r"""
import json
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from glm_tpu.layers.attention.dsa_indexer import dsa_scores, prefill_dsa_one_pass
from glm_tpu.runner.hlo_utils import parse_hlo_module
from tests.fixtures.tiny_model import cpu_mesh
from tests.reference.dsa import exact_topk
mesh = cpu_mesh()
rng = np.random.default_rng(356)
q = jnp.asarray(rng.normal(0, 0.2, (5, 32, 128)), jnp.float32)
keys = jnp.asarray(rng.normal(0, 0.2, (8, 256, 128)), jnp.bfloat16)
weights = jnp.asarray(rng.normal(0, 0.2, (5, 32)), jnp.float32).at[4].set(0)
positions = jnp.arange(2048, dtype=jnp.int32).reshape(256, 8).T  # owner e holds positions e, e + 8, ...
lengths = jnp.asarray([0, 1, 129, 1300, 2048], jnp.int32)
specs = (P(), P('expert', None, None), P(), P('expert', None), P())
values = tuple(jax.device_put(v, NamedSharding(mesh, s)) for v, s in zip((q, keys, weights, positions, lengths), specs))


def body(q, k, w, p, n):
    # production's call (prefill_dsa): default-precision scores, 512 candidates per owner
    selected, health = prefill_dsa_one_pass(q, k[0], w, p[0], n, global_context_size=2048, top_k=64,
                                            precision='default', candidates_per_owner=512)
    return selected.positions, selected.valid_counts, selected.scores, health[None, None]


mapped = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=specs, out_specs=(P(), P(), P(), P('expert', 'feature')),
                               check_vma=False))
compiled = mapped.lower(*values).compile()
selected, counts, scores, health = (np.asarray(v) for v in compiled(*values))
assert health.all()
expected_scores = dsa_scores(q, keys.transpose(1, 0, 2).reshape(2048, 128), weights, precision='default')
expected = exact_topk(expected_scores, lengths, top_k=64)
np.testing.assert_array_equal(selected, np.asarray(expected.positions))
np.testing.assert_array_equal(counts, np.asarray(expected.valid_counts))
np.testing.assert_array_equal(selected[4], np.arange(64))  # zero head weights: all ties, lowest positions win
expert = tuple(tuple(e * 4 + f for e in range(8)) for f in range(4))
collectives = parse_hlo_module(compiled.as_text()).collectives
assert collectives and all(c.replica_groups == expert for c in collectives), {c.replica_groups for c in collectives}
report = dict(collectives=sorted({c.opcode for c in collectives}))
# a repeated live position is unhealthy on its owner
bad = np.asarray(mapped(*values[:3], values[3].at[2, 3].set(values[3][2, 2]), values[4])[3])
assert not bad[2].any()
report['repeated_position_unhealthy_owners'] = sorted(int(e) for e in range(8) if not bad[e].all())
# requested counts are not proof of coverage: a table of holes, or two owners claiming the same positions
holes = jax.device_put(jnp.full((8, 256), -1, jnp.int32), NamedSharding(mesh, specs[3]))
assert not np.asarray(mapped(*values[:3], holes, values[4])[3]).any()
duplicate = values[3].at[1].set(values[3][0])
assert not np.asarray(mapped(*values[:3], duplicate, values[4])[3]).any()
print(json.dumps(report))
"""


@pytest.mark.cpu32
def test_prefill_selector_matches_the_exact_reference_cpu32():
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=32")
    result = subprocess.run([sys.executable, "-c", SELECTOR], env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert 2 in report["repeated_position_unhealthy_owners"], report


PRODUCER = r"""
import json
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from glm_tpu.layers.attention.dsa_indexer import dsa_scores, prefill_dsa, prefill_dsa_inputs, prompt_index_key_chunk
from glm_tpu.layers.attention.mla import PreparedAttention
from glm_tpu.models.glm_moe_dsa.weights import bf16_weight_specs
from glm_tpu.runner.hlo_utils import parse_hlo_module
from tests.fixtures.tiny_model import cpu_mesh, engine_inputs
from tests.reference.dsa import exact_topk
mesh = cpu_mesh()
inputs = engine_inputs(mesh, panel_geometry=True)
config = inputs.config
contract = config.dsa_contract
layer, spec = inputs.weights.layers[0], bf16_weight_specs(config).layers[0]
dsa = layer.dsa
rng = np.random.default_rng(456)
R, TOP_K = 17, contract.top_k
INT_MIN, INT_MAX = -(2**31), 2**31 - 1


def bf(shape, scale=0.1):
    return np.asarray(jnp.asarray(rng.normal(0, scale, shape), jnp.bfloat16))


def put(value, sharding=P()):
    return jax.device_put(value, NamedSharding(mesh, sharding))


# The repair weight is materialized by the loader; here deliberately unlike the checkpoint's (whose repair keys
# equal the unrepaired ones on CPU), so a key written into the wrong buffer shows. Not a provenance claim.
wk = put(bf((contract.head_dim, config.geometry.hidden_size)).astype(np.float32))


def bits(value):
    return np.asarray(value).tobytes()


CACHE = P(None, 'expert', None)
normalized = put(bf((R, config.geometry.hidden_size)), P(None, 'feature'))
prepared = PreparedAttention(normalized, normalized, put(bf((R, config.geometry.q_lora_rank))),
                             put(bf((R, config.geometry.kv_lora_rank + config.geometry.qk_rope_head_dim))))
prep_spec = PreparedAttention(P(None, 'feature'), P(None, 'feature'), P(), P())
cache = put(bf((config.page_count, config.logical_page_size, contract.head_dim)), CACHE)
# deliberately different from the live keys, so a write into the wrong buffer shows
repaired = put(np.full((config.page_count, config.logical_page_size, contract.head_dim), 7, jnp.bfloat16), CACHE)
table = put(np.asarray([[2, 0, 1]], np.int32))
PAGES = np.asarray(table)[0]


def body(p, c, r, o, n, t, w, k):
    v = prefill_dsa(p, c, r, o, n, t, w, k, contract=contract, linear_interpret=True)
    return (v.unrepaired_index_cache, v.repaired_index_cache, v.selected_positions, v.selected_valid_counts,
            v.selected_scores, v.contract_valid[None, None])


fn = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(prep_spec, CACHE, CACHE, P(), P(), P(), spec.dsa, P()),
                           out_specs=(CACHE, CACHE, P(), P(), P(), P('expert', 'feature')), check_vma=False))


def inputs_body(p, positions, live, w):
    v = prefill_dsa_inputs(p, positions, live, w, contract=contract, linear_interpret=True)
    return v.query, v.head_weights, v.keys, v.normalized_full


produce = jax.jit(jax.shard_map(inputs_body, mesh=mesh, in_specs=(prep_spec, P(), P(), spec.dsa),
                                out_specs=(P(), P(), P(), P()), check_vma=False))


def run(offset, count, p=prepared, r=repaired, c=cache, t=table, w=dsa):
    return tuple(np.asarray(v) for v in jax.block_until_ready(
        fn(p, c, r, put(np.int32(offset)), put(np.int32(count)), t, w, wk)))


def placed(base, offset, rows):
    # ``base`` with row i written at position offset + i (page table, then the position's row in its page)
    out = np.asarray(base).copy()
    for i, row in enumerate(rows):
        out[PAGES[(offset + i) // 512], (offset + i) % 512] = row
    return out


report = {}
for offset, count in ((55, 17), (505, 17), (0, 11)):
    out = run(offset, count)
    assert out[5].all(), (offset, count)
    positions = np.arange(R, dtype=np.int32) + offset
    live = np.arange(R) < count
    query, head, keys, normalized_full = produce(prepared, put(positions), put(live), dsa)
    # the unrepaired cache: exactly the live keys, each at its own position
    assert bits(out[0]) == bits(placed(cache, offset, np.asarray(keys)[:count])), (offset, count)
    # the selection: the exact reference over the logical unrepaired cache with per-row causal lengths
    logical = jnp.asarray(np.concatenate([out[0][page] for page in PAGES]))
    scores = dsa_scores(query, logical, head, precision='default')
    reference = exact_topk(scores, jnp.asarray(np.where(live, positions + 1, 0)), top_k=TOP_K)
    np.testing.assert_array_equal(out[2], np.asarray(reference.positions))
    np.testing.assert_array_equal(out[3], np.asarray(reference.valid_counts))
    chosen = np.take_along_axis(np.asarray(scores), np.maximum(out[2], 0), axis=1)
    np.testing.assert_array_equal(out[4], np.where(np.arange(TOP_K)[None] < out[3][:, None], chosen, -np.inf))
    # the repaired cache: each live row's M64 repair key (its full normalized input, the 64-row physical chunk
    # repeating the last live row) at the same position; the unrepaired key never lands there
    repeat = np.minimum(np.arange(64), max(count - 1, 0))
    repair = prompt_index_key_chunk(jnp.asarray(normalized_full)[repeat], jnp.asarray(positions[repeat]), wk,
                                    dsa.key_norm_weight, dsa.key_norm_bias, contract=contract)[:R].astype(jnp.bfloat16)
    assert bits(out[1]) == bits(placed(repaired, offset, np.asarray(repair)[:count])), (offset, count)
    assert bits(np.asarray(repair)[:count]) != bits(np.asarray(keys)[:count])
    # a different repaired history changes nothing the prompt computes
    other = run(offset, count, r=jnp.full_like(repaired, -13))
    assert all(bits(out[i]) == bits(other[i]) for i in (0, 2, 3, 4)), (offset, count)
    # head weights within the FP64 forward-error bound of the exact product (dimension-derived, not fitted)
    x = np.where(live[:, None], np.asarray(normalized, np.float64), 0)
    hw = np.asarray(dsa.head_weight_local, np.float64)
    n = config.geometry.hidden_size + 8
    gamma = n * 2.0**-24 / (1 - n * 2.0**-24)
    exact, bound = x @ hw.T * 32**-0.5, gamma * (np.abs(x) @ np.abs(hw).T) * 32**-0.5
    assert np.all(np.abs(np.asarray(head, np.float64) - exact) <= bound), (offset, count)
    report[f'{offset}+{count}'] = 'exact'
base = run(55, 17)
# a later live row's inputs reach no earlier key or selection
altered = run(55, 17, p=jax.tree.map(lambda v: v.at[-1].set(3), prepared))
assert bits(altered[0]) == bits(placed(base[0], 71, [altered[0][PAGES[0], 71]]))
assert all(bits(altered[i][:-1]) == bits(base[i][:-1]) for i in (2, 3, 4))
# NaN padding is ignored; zero live rows leave both caches unchanged with empty selections
poisoned = run(0, 11, p=jax.tree.map(lambda v: v.at[11:].set(jnp.nan), prepared))
assert all(bits(a) == bits(b) for a, b in zip(poisoned, run(0, 11)))
empty = run(0, 0, p=jax.tree.map(lambda v: jnp.full_like(v, jnp.nan), prepared))
assert empty[5].all() and bits(empty[0]) == bits(cache) and bits(empty[1]) == bits(repaired)
assert (empty[2] == -1).all() and (empty[3] == 0).all()
# invalid offsets, counts or page tables leave both caches unchanged and fail health
for offset, count, t in ((INT_MAX, 17, table), (INT_MIN, 17, table), (0, INT_MAX, table),
                         (505, 17, put(np.asarray([[2, 2, 1]], np.int32)))):
    bad = run(offset, count, t=t)
    assert not bad[5].any() and bits(bad[0]) == bits(cache) and bits(bad[1]) == bits(repaired), (offset, count)
# a NaN history key row 0 can select (position 40), a NaN live normalized or q row: unhealthy
for label, result in (('history', run(55, 17, c=cache.at[PAGES[0], 40, 0].set(jnp.nan))),
                      ('normalized', run(55, 17, p=prepared._replace(normalized_local=normalized.at[0, 0].set(jnp.nan)))),
                      ('q', run(55, 17, p=prepared._replace(q_residual=prepared.q_residual.at[0, 0].set(jnp.nan))))):
    assert not result[5].all(), label
# zero head weights: every score ties and the lowest positions win in every row
tie = run(55, 17, w=dsa._replace(head_weight_local=jnp.zeros_like(dsa.head_weight_local)))
for i in range(17):
    n = min(56 + i, TOP_K)
    np.testing.assert_array_equal(tie[2][i, :n], np.arange(n))
# every collective is in the feature-4 or the expert-8 groups
compiled = fn.lower(prepared, cache, repaired, put(np.int32(55)), put(np.int32(17)), table, dsa, wk).compile()
feature = tuple(tuple(range(e * 4, e * 4 + 4)) for e in range(8))
expert = tuple(tuple(e * 4 + f for e in range(8)) for f in range(4))
collectives = parse_hlo_module(compiled.as_text()).collectives
assert collectives and all(c.replica_groups in (feature, expert) for c in collectives)
print(json.dumps(report))
"""  # noqa: E501 (child program text)


@pytest.mark.cpu32
def test_prefill_dsa_keeps_causal_and_repair_state_separate_cpu32():
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=32")
    result = subprocess.run([sys.executable, "-c", PRODUCER], env=env, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report == {"55+17": "exact", "505+17": "exact", "0+11": "exact"}


def key_contract() -> DsaNumericalContract:
    return DsaNumericalContract(hidden_size=8, q_lora_rank=4, num_heads=1, head_dim=4, rotary_dim=2, top_k=4)


def key_chunk_arguments(rows: int) -> tuple[jax.Array, ...]:
    rng = np.random.default_rng(67)
    normalized = jnp.asarray(rng.normal(0, 1, (rows, 8)), jnp.bfloat16)
    wk = jnp.asarray(rng.normal(0, 0.5, (4, 8)), jnp.float32)
    return (
        normalized,
        jnp.arange(rows, dtype=jnp.int32) + 4096,
        wk,
        jnp.asarray([1.0, 0.75, -0.5, 1.25], jnp.bfloat16),
        jnp.asarray([0.0, 0.25, -0.125, 0.5], jnp.bfloat16),
    )


def test_prompt_key_chunk_projects_each_physical_partition_alone() -> None:
    """The repair keys' M64 association is partition-local: a chunk equals its physical partitions computed
    one by one, and the lowering is one loop over partitions with the recorded mixed-precision dot."""
    contract = key_contract()
    for rows, physical in ((128, 64), (8, 4)):
        arguments = key_chunk_arguments(rows)
        whole = prompt_index_key_chunk(*arguments, contract=contract, physical_rows=physical)
        parts = [
            prompt_index_key_chunk(
                arguments[0][start : start + physical],
                arguments[1][start : start + physical],
                *arguments[2:],
                contract=contract,
                physical_rows=physical,
            )
            for start in range(0, rows, physical)
        ]
        assert whole.dtype == jnp.float32 and whole.shape == (rows, 4)
        assert np.asarray(whole).tobytes() == np.asarray(jnp.concatenate(parts)).tobytes()
    arguments = key_chunk_arguments(8)
    stablehlo = str(
        jax.jit(lambda *a: prompt_index_key_chunk(*a, contract=contract, physical_rows=4))
        .lower(*arguments)
        .compiler_ir(dialect="stablehlo")
    )
    assert stablehlo.count("stablehlo.while") == 1
    assert stablehlo.count("precision = [DEFAULT, HIGHEST]") == 1
    assert "tensor<4x4xf32>" in stablehlo


def test_prompt_key_chunk_refuses_malformed_inputs() -> None:
    contract = key_contract()
    normalized, positions, wk, weight, bias = key_chunk_arguments(8)
    for physical in (0, True, -64):
        with pytest.raises(ValueError, match="physical prompt-key rows"):
            prompt_index_key_chunk(normalized, positions, wk, weight, bias, contract=contract, physical_rows=physical)
    with pytest.raises(ValueError, match="divide"):
        prompt_index_key_chunk(normalized[:6], positions[:6], wk, weight, bias, contract=contract, physical_rows=4)
    with pytest.raises(ValueError, match="BF16"):
        prompt_index_key_chunk(
            normalized.astype(jnp.float32), positions, wk, weight, bias, contract=contract, physical_rows=4
        )
    with pytest.raises(ValueError, match="adapted FP32"):
        prompt_index_key_chunk(
            normalized, positions, wk.astype(jnp.bfloat16), weight, bias, contract=contract, physical_rows=4
        )
    with pytest.raises(ValueError, match="one integer row per token"):
        prompt_index_key_chunk(normalized, positions[:4], wk, weight, bias, contract=contract, physical_rows=4)
