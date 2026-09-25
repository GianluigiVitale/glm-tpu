"""Resident projection refusals, the multirow prefill projection and the canonical dense row placement of the
production prefill.

``prefill_linear`` projects a block of prompt rows over the resident BF16 tables and reduces the FP32 partials
over the feature-4 or expert-8 axis: the block, its one-row calls and the decode projection
(``feature_linear``/``expert_linear``) all stay within the FP64 forward-error bound of the exact product, no row
reaches another, and the only collectives are those reductions (ported from the research package's
``kernels/test_ws32_prefill_linear.py``, ``archive/research-20260922``, whose research kernels were
row-independent bit for bit; XLA's CPU dot is not, for a block's last rows).

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
    lhs = jnp.ones((2, 128), jnp.bfloat16)
    for weight, scale in (
        (jnp.ones((128, 128), jnp.uint8), None),
        (jnp.ones((128, 128), jnp.bfloat16), jnp.ones((1, 1), jnp.float32)),
    ):
        with pytest.raises(ValueError):
            resident.resident_matmul(lhs, weight, scale)


@pytest.mark.cpu32
def test_bf16_canonical_dense_cpu32():
    code = r"""
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
"""
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=32")
    p = subprocess.run([sys.executable, "-c", code], env=env, text=True, capture_output=True, timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr


PREFILL_LINEAR = r"""
import json
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from glm_tpu.layers.linear import expert_linear, feature_linear, prefill_linear
from glm_tpu.models.glm_moe_dsa.weights import bf16_weight_specs
from glm_tpu.runner.hlo_utils import parse_hlo_module
from tests.fixtures.tiny_model import cpu_mesh, engine_inputs
mesh = cpu_mesh()
inputs = engine_inputs(mesh, panel_geometry=True)
config = inputs.config
layer, spec = inputs.weights.layers[0], bf16_weight_specs(config).layers[0]
rng = np.random.default_rng(172)
feature_groups = tuple(tuple(range(e * 4, e * 4 + 4)) for e in range(8))
expert_groups = tuple(tuple(e * 4 + f for e in range(8)) for f in range(4))
report = {}
for axis, table, table_spec, width, in_spec, out_spec, decode, groups in (
    # q-a: hidden shards reduced over feature-4 (the attention preparation)
    ('feature', layer.qkv_a.q_a_local, spec.qkv_a.q_a_local, config.geometry.hidden_size, P(None, 'feature'), P(),
     feature_linear, feature_groups),
    # o-proj: head shards reduced over expert-8 (the attention output)
    ('expert', layer.attention.o_local, spec.attention.o_local, layer.attention.o_local.shape[1], P(None, 'expert'),
     P(None, 'feature'), expert_linear, expert_groups),
):
    x = jnp.asarray(rng.normal(0, 0.15, (17, width)), jnp.bfloat16).at[7].set(0)
    x = jax.device_put(x, NamedSharding(mesh, in_spec))
    batch = jax.jit(jax.shard_map(lambda v, w: prefill_linear(v, w, reduction_axis=axis, interpret=True), mesh=mesh,
                                  in_specs=(in_spec, table_spec), out_specs=out_spec, check_vma=False))
    one = jax.jit(jax.shard_map(lambda v, w: decode(v, w, axis), mesh=mesh, in_specs=(in_spec, table_spec),
                                out_specs=out_spec, check_vma=False))
    compiled = batch.lower(x, table).compile()
    actual = compiled(x, table)
    assert actual.shape[0] == 17 and actual.dtype == jnp.bfloat16
    # XLA's CPU dot may associate a row's FP32 sum differently with the row count (a block's last rows),
    # so the block, its one-row calls and the decode projection are each judged against FP64 arithmetic
    # with a dimension-derived bound: FP32 accumulation over the contracted width plus the reduction,
    # then one BF16 rounding (unit roundoff 2**-8)
    host, weight = np.asarray(x, np.float64), np.asarray(table, np.float64)
    exact, magnitude = host @ weight.T, np.abs(host) @ np.abs(weight).T
    n = width + 8
    gamma = n * 2.0**-24 / (1 - n * 2.0**-24)
    bound = gamma * magnitude + 2.0**-8 * (np.abs(exact) + gamma * magnitude)
    rows = jnp.concatenate([batch(x[i : i + 1], table) for i in range(17)])
    decoded = jnp.concatenate([one(x[i : i + 1], table) for i in range(17)])
    for name, value in (('block', actual), ('one-row', rows), ('decode', decoded)):
        assert np.all(np.abs(np.asarray(value, np.float64) - exact) <= bound), (axis, name)
    # no row reaches another at the block's shape
    other = batch(x.at[3].set(1), table)
    assert np.delete(np.asarray(other), 3, 0).tobytes() == np.delete(np.asarray(actual), 3, 0).tobytes()
    collectives = parse_hlo_module(compiled.as_text()).collectives
    assert collectives and all(c.opcode == 'all-reduce' and c.replica_groups == groups for c in collectives), axis
    report[axis] = [list(actual.shape), len(collectives)]
print(json.dumps(report))
"""


@pytest.mark.cpu32
def test_prefill_linear_matches_the_one_row_projections_cpu32():
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=32")
    p = subprocess.run([sys.executable, "-c", PREFILL_LINEAR], env=env, text=True, capture_output=True, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr


def test_prefill_linear_rejects_invalid_rows_dtype_and_axis() -> None:
    table = jnp.zeros((128, 128), jnp.bfloat16)
    for value in (jnp.ones((128,), jnp.bfloat16), jnp.ones((0, 128), jnp.bfloat16)):
        with pytest.raises(ValueError, match="nonempty"):
            resident.prefill_linear(value, table, reduction_axis="feature")
    with pytest.raises(ValueError, match="bfloat16"):
        resident.prefill_linear(jnp.ones((8, 128), jnp.float32), table, reduction_axis="feature")
    with pytest.raises(ValueError, match="feature or expert"):
        resident.prefill_linear(jnp.ones((8, 128), jnp.bfloat16), table, reduction_axis="model")
    with pytest.raises(ValueError, match="resident BF16 tables"):
        resident.prefill_linear(jnp.ones((8, 128), jnp.bfloat16), table.astype(jnp.uint8), reduction_axis="feature")


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
    np.testing.assert_array_equal(np.asarray(residual_add(hidden, got)), np.asarray(hidden + got))

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
