"""Flattened cache commit: exact addressing and RAW geometry, not TPU fit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.kernels.prefill_flat_rows import apply_prefill_flat_rows
from glm_tpu.greenfield.kernels.prefill_pending_rows import apply_prefill_pending_rows
from tests.greenfield.kernels.test_prefill_pending_rows import addresses


@pytest.mark.parametrize("offset,count", [(0,128),(57,128),(505,128),(633,3),(1020,114),(1533,3)])
def test_exact_layer_rows_and_unowned_padding(offset, count):
    initial = (jnp.arange(3*3*64*4).reshape(3,3,64,4) % 251).astype(jnp.bfloat16)
    # Each layer's rows differ, so a layer/row transposition cannot hide.
    values = (jnp.arange(3*128*4).reshape(3,128,4) % 73).astype(jnp.bfloat16)

    @jax.jit
    def run(owner):
        target = addresses([[2,0,1]], offset, count, owner).targets
        rows = jnp.where((target < 192)[None,:,None], values, jnp.nan)
        return (apply_prefill_flat_rows(initial,target,rows),
                apply_prefill_pending_rows(initial,target,rows), target)

    actual, old, targets = jax.vmap(run)(jnp.arange(8,dtype=jnp.int32))
    assert np.asarray(actual).tobytes() == np.asarray(old).tobytes()
    for owner in range(8):
        expected = np.asarray(initial).copy().reshape(3,192,4)
        for row, target in enumerate(np.asarray(targets[owner])):
            if 0 <= target < 192:
                expected[:,target,:] = np.asarray(values)[:,row,:]
        assert np.asarray(actual[owner]).tobytes() == expected.tobytes()


def test_global_drop_cannot_wrap_or_write_next_layer():
    initial = jnp.full((3,2,4,4),7,jnp.bfloat16)
    target = jnp.array([-2147483648,-1,8,2147483647,0,7],jnp.int32)
    rows = jnp.full((3,6,4),jnp.nan,jnp.bfloat16)
    rows = rows.at[:,4,:].set(jnp.arange(3)[:,None]+10)
    rows = rows.at[:,5,:].set(jnp.arange(3)[:,None]+20)
    result = jax.jit(apply_prefill_flat_rows)(initial,target,rows)
    expected = initial.at[:,0,0,:].set(jnp.arange(3)[:,None]+10)
    expected = expected.at[:,1,3,:].set(jnp.arange(3)[:,None]+20)
    assert np.asarray(result).tobytes() == np.asarray(expected).tobytes()
    assert np.isfinite(np.asarray(result).astype(np.float32)).all()


def test_scatter_has_only_contiguous_width_window_and_rollback():
    def commit(old, target, rows, healthy):
        return jax.lax.cond(healthy,lambda _:apply_prefill_flat_rows(old,target,rows),
                            lambda _:old,None)
    initial=jnp.full((3,3,64,4),7,jnp.bfloat16)
    rows=jnp.ones((3,128,4),jnp.bfloat16)
    target=addresses([[2,0,1]],505,3,0).targets
    fn=jax.jit(commit,donate_argnums=(0,))
    saved=np.asarray(initial).tobytes()
    result=fn(initial,target,rows,jnp.bool_(False))
    assert np.asarray(result).tobytes()==saved and initial.is_deleted()
    module=fn.lower(jax.ShapeDtypeStruct((3,3,64,4),jnp.bfloat16),
                    target,rows,jnp.bool_(True)).compiler_ir()
    scatters=[]
    def visit(op):
        if op.operation.name=='stablehlo.scatter':scatters.append(op)
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:visit(child)
    visit(module.operation)
    assert len(scatters)==1
    scatter=scatters[0]
    dims=str(scatter.attributes['scatter_dimension_numbers'])
    assert 'update_window_dims = [1]' in dims
    assert 'inserted_window_dims = [0]' in dims
    assert 'scatter_dims_to_operand_dims = [0]' in dims
    assert str(scatter.operands[0].type)=='tensor<576x4xbf16>'
    assert str(scatter.operands[2].type)=='tensor<384x4xbf16>'


@pytest.mark.parametrize("shape,rows_shape,dtype,target_dtype", [
    ((3,4,4),(3,2,4),jnp.bfloat16,jnp.int32),
    ((3,0,4,4),(3,2,4),jnp.bfloat16,jnp.int32),
    ((3,2,4,4),(3,2,4),jnp.float32,jnp.int32),
    ((3,2,4,4),(3,2,4),jnp.bfloat16,jnp.float32),
    ((3,2,4,4),(2,2,4),jnp.bfloat16,jnp.int32),
])
def test_static_shapes_and_dtypes_refuse(shape,rows_shape,dtype,target_dtype):
    with pytest.raises(ValueError):
        apply_prefill_flat_rows(jnp.zeros(shape,dtype),jnp.zeros((2,),target_dtype),
                                jnp.zeros(rows_shape,jnp.bfloat16))


def test_flat_index_overflow_refused_without_allocating():
    with pytest.raises(ValueError,match='positive int32'):
        jax.eval_shape(apply_prefill_flat_rows,
                       jax.ShapeDtypeStruct((78,1000000,64,640),jnp.bfloat16),
                       jax.ShapeDtypeStruct((128,),jnp.int32),
                       jax.ShapeDtypeStruct((78,128,640),jnp.bfloat16))
