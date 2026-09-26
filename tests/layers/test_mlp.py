"""The dense MLP of the production prefill: the canonical row placement of its suffix and the geometry
refusals of :func:`glm_tpu.layers.mlp.prefill_dense`.

TPU results of the dense MLP depend on the physical row placement, so the prefill window runs the dense suffix
as four placements of 32 live rows in physical rows 0:32 of a 128-row call
(:func:`glm_tpu.models.glm_moe_dsa.decoder_layer.prefill_dense_canonical`, the placement ``layers/mlp.py``
names). On the CPU mesh (32 forced devices, the frozen fixture's first dense layer, distinct inputs on every
owner) the canonical placement must equal those four placements written out one by one and, on CPU, every row in
one call, for every live count of B128 and B114, with dead rows zero, padding ignored and a live non-finite row
unhealthy at its own row.
Ported from the research package's ``runtime/test_ws32_prefill_dense_canonical.py`` and, for the geometry refusals
only, from its ``kernels/test_ws32_prefill_linear.py`` (``archive/research-20260922``).
"""

from __future__ import annotations

import os
import subprocess
import sys

import jax.numpy as jnp
import pytest

from glm_tpu.layers.mlp import prefill_dense

CANONICAL = r"""
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
from glm_tpu.models.glm_moe_dsa.decoder_layer import prefill_dense_canonical, prefill_mlp
from glm_tpu.models.glm_moe_dsa.weights import bf16_weight_specs
from tests.fixtures.tiny_model import cpu_mesh, engine_inputs
mesh = cpu_mesh()
inputs = engine_inputs(mesh, panel_geometry=True)
config = inputs.config
dense, spec = inputs.weights.layers[0].dense, bf16_weight_specs(config).layers[0].dense
contract = config.moe_contract
width = config.geometry.hidden_size // 4


def put(value, sharding):
    return jax.device_put(value, NamedSharding(mesh, sharding))


def canonical_body(x, live, weights):
    out = prefill_dense_canonical(x[0, 0], live, weights, moe_contract=contract, linear_interpret=True)
    return tuple(v[None, None] for v in out)


def placements_body(x, live, weights):
    # each 32-row slice alone in physical rows 0:32 of a 128-row MLP call, then cropped and assembled
    rows = x.shape[2]
    values = jnp.pad(jnp.where(live[:, None], x[0, 0], 0), ((0, 128 - rows), (0, 0)))
    mask = jnp.pad(live, ((0, 128 - rows),))
    out = []
    for start in (0, 32, 64, 96):
        part = jnp.pad(values[start : start + 32], ((0, 96), (0, 0)))
        part_live = jnp.pad(mask[start : start + 32], ((0, 96),))
        y, ids, w, h = prefill_mlp(part, part_live, weights, None, moe_contract=contract, linear_interpret=True)
        h = h & (~part_live | jnp.all(jnp.isfinite(y), axis=1))
        y = jnp.where(part_live[:, None], y, 0)
        out.append((y[:32], ids[:32], w[:32], h[:32]))
    return tuple(jnp.concatenate([o[j] for o in out])[:rows][None, None] for j in range(4))


def full_body(x, live, weights):
    # every row in one MLP call: an independent original, not another copy of the placement schedule
    values = jnp.where(live[:, None], x[0, 0], 0)
    y, ids, w, h = prefill_mlp(values, live, weights, None, moe_contract=contract, linear_interpret=True)
    h = h & (~live | jnp.all(jnp.isfinite(y), axis=1))
    y = jnp.where(live[:, None], y, 0)
    return tuple(v[None, None] for v in (y, ids, w, h))


def wrap(fn):
    return jax.jit(jax.shard_map(fn, mesh=mesh, in_specs=(P('expert', 'feature'), P(), spec),
                                 out_specs=(P('expert', 'feature'),) * 4, check_vma=False))


canonical, placements, full = wrap(canonical_body), wrap(placements_body), wrap(full_body)


def same(actual, expected, what):
    for a, e in zip(actual, expected, strict=True):
        a, e = np.asarray(a), np.asarray(e)
        assert a.shape == e.shape and a.dtype == e.dtype and a.tobytes() == e.tobytes(), what


rng = np.random.RandomState(606)
for rows, counts in ((128, (0, 1, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128)), (114, (0, 91, 96, 97, 113, 114))):
    # distinct inputs on every owner, all in one call (the expert axis is not independent)
    data = (rng.randn(8, 4, rows, width) * 0.01).astype(jnp.bfloat16)
    x = put(data, P('expert', 'feature'))
    for count in counts:
        live = put(np.arange(rows) < count, P())
        got = canonical(x, live, dense)
        same(got, placements(x, live, dense), (rows, count))
        same(got, full(x, live, dense), ('full-row', rows, count))
        output, ids, weights, health = (np.asarray(v) for v in got)
        assert health.all() and np.count_nonzero(output[:, :, count:]) == 0, (rows, count)
        assert (ids == -1).all() and not weights.any(), (rows, count)
        poison = data.copy()
        poison[:, :, count:] = np.nan
        same(canonical(put(poison, P('expert', 'feature')), live, dense), got, ('dead rows', rows, count))
    everything = put(np.ones(rows, bool), P())
    # a live NaN makes its row unhealthy instead of disappearing in the placement
    bad = data.copy()
    bad[:, :, 0] = np.nan
    assert not np.asarray(canonical(put(bad, P('expert', 'feature')), everything, dense)[3])[:, :, 0].any()
    # one owner's NaN in a later tile stays at its own output row
    bad = data.copy()
    bad[2, 1, 33, 0] = np.nan
    health = np.asarray(canonical(put(bad, P('expert', 'feature')), everything, dense)[3])
    np.testing.assert_array_equal(health, np.asarray(full(put(bad, P('expert', 'feature')), everything, dense)[3]))
    assert not health[:, :, 33].all() and health[:, :, :33].all() and health[:, :, 34:].all(), rows
for rows, dtype in ((32, jnp.bfloat16), (128, jnp.float32)):
    try:
        prefill_dense_canonical(jnp.zeros((rows, width), dtype), jnp.ones(rows, bool), dense, moe_contract=contract,
                                linear_interpret=True)
        raise AssertionError('accepted invalid shape/dtype')
    except ValueError:
        pass
print('canonical dense placement checked')
"""


@pytest.mark.cpu32
def test_canonical_dense_equals_its_four_32_row_placements_cpu32():
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=32")
    result = subprocess.run([sys.executable, "-c", CANONICAL], env=env, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr


def test_prefill_dense_rejects_mismatched_weight_geometry() -> None:
    hidden = jnp.ones((8, 128), jnp.bfloat16)
    table = jnp.zeros((128, 128), jnp.bfloat16)
    with pytest.raises(ValueError, match="gate/up"):
        prefill_dense(hidden, table, table[:64], table)
    with pytest.raises(ValueError, match="gate/up"):
        prefill_dense(hidden, table[:, :64], table[:, :64], table)
    with pytest.raises(ValueError, match="down geometry"):
        prefill_dense(hidden, table, table, table[:64])
    with pytest.raises(ValueError, match="nonempty"):
        prefill_dense(jnp.ones((0, 128), jnp.bfloat16), table, table, table)
    with pytest.raises(ValueError, match="bfloat16"):
        prefill_dense(jnp.ones((8, 128), jnp.float32), table, table, table)
