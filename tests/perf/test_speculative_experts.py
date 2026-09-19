"""M8 weight reuse preserves independent routed FP8 projections on CPU."""
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from glm_tpu.perf.fp8_routed_experts import RoutedProjectionConfig, fp8_routed_projection
from glm_tpu.perf.speculative_experts import small_expert_plan, small_expert_projection


def _cpu_pallas():
    from jax._src.pallas.mosaic import tpu_info
    tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
    tpu_info.get_tpu_info.cache_clear()


@pytest.mark.parametrize('rows', [2, 3, 5, 8])
@pytest.mark.parametrize('result_dtype', [jnp.float32, jnp.bfloat16])
def test_m8_projection_matches_independent_routes_bitwise(rows, result_dtype):
    _cpu_pallas()
    rng = np.random.default_rng(27+rows)
    local_experts, top_k, n, k = 4, 3, 256, 256
    # Each token has distinct experts, but order/reuse varies across tokens.
    routes = np.array([[1, 4, 6] if i % 2 else [6, 4, 2] for i in range(rows)], np.int32)
    lhs = jnp.asarray(rng.normal(0, .25, (rows*top_k, k)), jnp.bfloat16)
    tables = tuple((jnp.asarray(np.asarray(rng.normal(0, .125, (local_experts,n,k)),
                                           ml_dtypes.float8_e4m3fn).view(np.uint8)),
                    jnp.asarray(rng.uniform(.25, 2., (local_experts,2,2)), jnp.float32))
                   for _ in range(2))
    config = RoutedProjectionConfig(output_tile=256, contraction_tile=256)
    def body(x, weights, ids, offset):
        plan = small_expert_plan(ids, offset, num_experts=12, local_experts=local_experts)
        out, valid = small_expert_projection(x, weights, plan, config=config,
            result_dtype=result_dtype, interpret=True)
        flat = ids.reshape(-1)-offset
        expected = fp8_routed_projection(x, weights, jnp.clip(flat,0,local_experts-1),
            (flat >= 0) & (flat < local_experts), config=config,
            result_dtype=result_dtype, interpret=True)
        return out, expected, valid, plan.active_groups
    fn = jax.jit(body)
    for offset, groups in ((0,2), (4,2), (8,0)):
        out, expected, valid, active = fn(lhs,tables,jnp.asarray(routes),jnp.int32(offset))
        assert bool(valid) and int(active) == groups
        view = np.uint32 if result_dtype == jnp.float32 else np.uint16
        np.testing.assert_array_equal(np.asarray(out).view(view), np.asarray(expected).view(view))
        if offset == 8:
            assert not np.asarray(out).any()


def test_plan_packing_covers_every_owned_slot_once():
    # >8 distinct experts, every row changes route order, and capacity <local
    # experts at the small end. Also exercise the short (<8-slot) packing case.
    for rows, top_k in ((1,1), (2,2), (3,8), (8,8)):
        routes = np.stack([(np.arange(top_k)*7+i*11)%64 for i in range(rows)]).astype(np.int32)
        for offset in (0,32):
            plan = small_expert_plan(jnp.asarray(routes),jnp.int32(offset),
                                    num_experts=64,local_experts=32)
            assert bool(plan.valid)
            slots = np.asarray(plan.route_slots)
            active = int(plan.active_groups)
            expected = np.flatnonzero((routes.reshape(-1)>=offset)&(routes.reshape(-1)<offset+32))
            np.testing.assert_array_equal(np.sort(slots[slots<routes.size]),expected)
            assert np.all(slots[active:]==routes.size)
            for group in range(active):
                live=slots[group][slots[group]<routes.size]
                assert len(live)<=rows
                np.testing.assert_array_equal(routes.reshape(-1)[live],
                    np.full(live.size,int(plan.expert_ids[group])+offset))


@pytest.mark.parametrize('kind',['negative','too_large','duplicate','bad_offset'])
def test_invalid_plan_refuses_and_zeroes_output(kind):
    _cpu_pallas()
    routes=jnp.array([[0,1],[1,2]],jnp.int32)
    offset=jnp.int32(0)
    if kind=='negative': routes=routes.at[0,0].set(-1)
    if kind=='too_large': routes=routes.at[0,0].set(8)
    if kind=='duplicate': routes=routes.at[0,0].set(1)
    if kind=='bad_offset': offset=jnp.int32(1)
    plan=small_expert_plan(routes,offset,num_experts=8,local_experts=4)
    tables=((jnp.zeros((4,128,128),jnp.uint8),jnp.ones((4,1,1),jnp.float32)),)
    result,valid=small_expert_projection(jnp.ones((4,128),jnp.bfloat16),tables,plan,
        config=RoutedProjectionConfig(output_tile=128,contraction_tile=128),interpret=True)
    assert not bool(valid)
    assert not np.asarray(result).any()
