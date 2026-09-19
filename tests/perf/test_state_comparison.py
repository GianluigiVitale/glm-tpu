"""Boundary evidence must detect signed zeros, BF16 differences and sentinels."""
import jax.numpy as jnp
import pytest

from glm_tpu.perf.state_comparison import compare_addressable_state


def test_comparison_detects_numeric_and_bitwise_boundaries():
    a={'cache':jnp.asarray([0.0,1.0,float('-inf'),float('inf')],jnp.bfloat16),
       'position':jnp.asarray(12,jnp.int32)}
    b={'cache':jnp.asarray([-0.0,1.0078125,float('-inf'),0.0],jnp.bfloat16),
       'position':jnp.asarray(13,jnp.int32)}
    result=compare_addressable_state(a,b)
    assert not result['bitwise_equal']
    row=result['leaves']["['cache']"]
    assert row['elements']==4 and row['bitwise_different']==3
    assert row['nonfinite_different']==1
    assert row['max_finite_abs_difference']==0.0078125
    assert result['leaves']["['position']"]['bitwise_different']==1
    assert compare_addressable_state(a,a)['bitwise_equal']


def test_comparison_refuses_partial_state():
    a={'cache':jnp.zeros(4,jnp.bfloat16)}
    with pytest.raises(ValueError,match='structures'):
        compare_addressable_state(a,{})
    with pytest.raises(ValueError,match='geometry'):
        compare_addressable_state(a,{'cache':jnp.zeros(3,jnp.bfloat16)})
