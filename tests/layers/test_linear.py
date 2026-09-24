"""Resident projection refusals and the canonical dense row placement of the production prefill.

Until S2f ``test_bf16_canonical_dense_cpu32`` also compared the BF16-resident canonical dense MLP with
the frozen FP8 one (output within rtol 0.02 / atol 0.01, routing and live masks equal); that oracle
is archived at ``archive/research-20260922`` and its final green run is recorded in the S2f commit
message. The production behaviour it checked stays: B114 and B128 both run, dead rows stay zero and
every owner reports healthy.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import jax.numpy as jnp
import jax
import numpy as np

from glm_tpu.layers import linear as resident
from glm_tpu.layers.linear import residual_add
from tests.reference.linear import dense_swiglu, embedding_lookup, linear, vocabulary_logits


def test_resident_projection_rejects_raw_bits_or_scales():
    lhs = jnp.ones((2,128),jnp.bfloat16)
    for weight, scale in ((jnp.ones((128,128),jnp.uint8),None),
                          (jnp.ones((128,128),jnp.bfloat16),jnp.ones((1,1),jnp.float32))):
        with pytest.raises(ValueError): resident.resident_matmul(lhs,weight,scale)


def test_bf16_canonical_dense_cpu32():
    code=r'''
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
from glm_tpu.models.glm_moe_dsa.weights import bf16_resident_weights, bf16_weight_specs
from glm_tpu.models.glm_moe_dsa.decoder_layer import prefill_dense_canonical as canonical
from tests.fixtures.tiny_model import cpu_mesh, fixture
mesh=cpu_mesh()
config,weights,_=fixture(mesh)
bf16=bf16_resident_weights(mesh,config,weights)
rng=np.random.default_rng(1926)
for rows in (114,128):
    # Late live rows expose accidentally using ordinary B128 placement.
    x=jnp.asarray(rng.normal(size=(rows,config.geometry.hidden_size)),jnp.bfloat16)
    live=jnp.arange(rows)%3!=0
    def body(x,live,resident):
        return canonical(x,live,resident.layers[0].dense,moe_contract=config.moe_contract,linear_interpret=True)
    fn=jax.jit(jax.shard_map(body,mesh=mesh,in_specs=(P(None,'feature'),P(),bf16_weight_specs(config)),
                             out_specs=(P(None,'feature'),P(),P(),P()),check_vma=False))
    b=fn(x,live,bf16)
    assert np.asarray(b[0]).shape==(rows,config.geometry.hidden_size)
    assert np.isfinite(np.asarray(b[0]).astype(np.float32)).all()
    assert np.asarray(b[-1]).all()
    assert not np.asarray(b[0])[~np.asarray(live)].any()
print('BF16 canonical B114/B128 output, live masks and health checked')
'''
    env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=32')
    p=subprocess.run([sys.executable,'-c',code],env=env,text=True,capture_output=True,timeout=300)
    assert p.returncode==0,p.stdout+p.stderr


def test_linear_preserves_checkpoint_out_in_orientation_and_leading_shape() -> None:
    hidden = jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3)
    weight = jnp.asarray([[1, 2, 3], [-1, 0, 1]], dtype=jnp.float32)
    bias = jnp.asarray([0.5, -0.5], dtype=jnp.float32)
    got = linear(hidden, weight, bias)
    expected = np.asarray(hidden) @ np.asarray(weight).T + np.asarray(bias)
    np.testing.assert_array_equal(np.asarray(got), expected)
    assert got.shape == (2, 2, 2)


def test_linear_bf16_has_explicit_bf16_output_boundary() -> None:
    hidden = jnp.asarray([[1.0, -0.5, 0.25]], dtype=jnp.bfloat16)
    weight = jnp.asarray([[2.0, 1.0, -1.0]], dtype=jnp.bfloat16)
    got = jax.jit(linear)(hidden, weight)
    assert got.dtype == jnp.bfloat16
    assert got.shape == (1, 1)
    assert float(got[0, 0]) == 1.25


def test_linear_refuses_silent_shape_broadcasts() -> None:
    with pytest.raises(ValueError, match="input width"):
        linear(jnp.ones((1, 3)), jnp.ones((2, 4)))
    with pytest.raises(ValueError, match="bias"):
        linear(jnp.ones((1, 3)), jnp.ones((2, 3)), jnp.ones((1,)))


def test_dense_swiglu_residual_embedding_and_logits_are_explicit() -> None:
    hidden = jnp.asarray([[0.5, -1.0]], dtype=jnp.float32)
    gate = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=jnp.float32)
    up = jnp.asarray([[0.5, 0.0], [0.0, 0.5], [1.0, -1.0]], dtype=jnp.float32)
    down = jnp.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]], dtype=jnp.float32)
    got = dense_swiglu(hidden, gate, up, down)
    gate_value = np.asarray(hidden) @ np.asarray(gate).T
    up_value = np.asarray(hidden) @ np.asarray(up).T
    activated = gate_value / (1.0 + np.exp(-gate_value))
    expected = (activated * up_value) @ np.asarray(down).T
    np.testing.assert_allclose(np.asarray(got), expected, rtol=2e-7, atol=2e-7)
    np.testing.assert_array_equal(
        np.asarray(residual_add(hidden, got)), np.asarray(hidden + got)
    )

    table = jnp.arange(12, dtype=jnp.bfloat16).reshape(6, 2)
    ids = jnp.asarray([[5, 0]], dtype=jnp.int32)
    embeddings = embedding_lookup(ids, table)
    np.testing.assert_array_equal(np.asarray(embeddings), np.asarray(table)[[5, 0]][None])
    logits = vocabulary_logits(embeddings, table)
    assert logits.shape == (1, 2, 6)
    assert logits.dtype == jnp.bfloat16


def test_residual_add_refuses_broadcast_or_dtype_promotion() -> None:
    with pytest.raises(ValueError, match="shapes"):
        residual_add(jnp.ones((1, 4)), jnp.ones((4,)))
    with pytest.raises(ValueError, match="dtypes"):
        residual_add(jnp.ones((1, 4), jnp.bfloat16), jnp.ones((1, 4), jnp.float32))
