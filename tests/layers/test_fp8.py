"""BF16-resident non-routed weights: exact table decode, frozen partition specs, one decode step.

CPU semantics on the forced 32-device mesh. Until S2f the decode test also stepped the frozen FP8
decoder beside the release decoder (same tokens and DSA selections, KV within one BF16 ulp on under
1 % of elements); that oracle is archived at ``archive/research-20260922``, its final green run is
recorded in the S2f commit message, and ``tests/reference`` (``VALIDATION.md``) carries the evidence.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.layers.fp8 import decode_fp8_table


def test_decode_table_matches_reference_dequantizer_bitwise():
    from glm_tpu.layers.fp8 import dequantize_fp8_bits_block_weight

    rng = np.random.default_rng(11)
    bits = jnp.asarray(rng.integers(0, 256, (256, 384), dtype=np.uint8))
    scale = jnp.asarray(rng.uniform(0.25, 2.0, (2, 3)), jnp.float32)
    expected = dequantize_fp8_bits_block_weight(bits, scale, block_shape=(128, 128))
    actual = decode_fp8_table(bits, scale)
    assert actual.dtype == jnp.bfloat16
    e, a = np.asarray(expected).view(np.uint16), np.asarray(actual).view(np.uint16)
    nan = np.isnan(np.asarray(expected).astype(np.float32))
    assert not np.any((e != a) & ~nan)
    with pytest.raises(ValueError):
        decode_fp8_table(bits, scale[:1])
    with pytest.raises(ValueError):
        decode_fp8_table(bits.astype(jnp.int8), scale)


def test_bf16_resident_decode_paths_agree_cpu32():
    code = r"""
import json
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
from glm_tpu.models.glm_moe_dsa.weights import bf16_weight_specs
from glm_tpu.models.glm_moe_dsa.model import build_decoder_program
from glm_tpu.models.glm_moe_dsa.model import build_packed_decoder_program
from glm_tpu.runner.programs import build_program_set
from tests.fixtures.tiny_model import cpu_mesh, engine_inputs, prefill
mesh = cpu_mesh()
inputs = engine_inputs(mesh, panel_geometry=True)
config, bf16, rope = inputs.config, inputs.weights, inputs.rope
interpret = dict(sparse_attention_interpret=True, linear_interpret=True)
# Every leaf keeps the frozen partition spec and shape; only non-routed FP8 tables changed dtype.
for leaf, spec in zip(jax.tree.leaves(bf16), jax.tree.leaves(bf16_weight_specs(config), is_leaf=lambda x: isinstance(x, P))):
    assert leaf.sharding.is_equivalent_to(NamedSharding(mesh, spec), leaf.ndim), (leaf.shape, leaf.sharding.spec, spec)
state, token = prefill(mesh, inputs, build_program_set(mesh, config, interpret=True), [30, 31, 32])
# the release decoder: 256x256 routed tiles, two-stage DSA, greedy head (the only profile)
challenger = build_decoder_program(mesh, config, **interpret)
packed = build_packed_decoder_program(mesh, config, **interpret)
owned_packed = jax.jit(packed.execute,donate_argnums=(1,))
report = dict(tokens=[], valid=True)
for step in range(3):
    out = challenger.execute(token, state, bf16, rope)
    compact = packed.execute(token, state, bf16, rope)
    for a, z in zip(jax.tree.leaves(out), jax.tree.leaves(compact.decoded)):
        np.testing.assert_array_equal(np.asarray(a).view(np.uint8), np.asarray(z).view(np.uint8))
    np.testing.assert_array_equal(np.asarray(compact.metadata),
        [int(out.next_token[0]),1,int(out.state.position[0]),int(out.state.context_lengths[0])])
    owned_state=jax.tree.map(lambda value:jnp.array(value,copy=True),state)
    donated=owned_packed(token,owned_state,bf16,rope)
    jax.block_until_ready(donated)
    for expected,actual in zip(jax.tree.leaves(compact),jax.tree.leaves(donated)):
        np.testing.assert_array_equal(np.asarray(expected).view(np.uint8),np.asarray(actual).view(np.uint8))
    report['tokens'].append(int(out.next_token[0]))
    report['valid'] = report['valid'] and bool(np.asarray(out.state.contract_valid).all())
    assert int(out.state.position[0]) == 4 + step  # prompt (3) + the tokens generated so far
    state, token = out.state, out.next_token
print(json.dumps(report))
"""
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=(os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=32").strip(),
    )
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=1500)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["valid"]
    assert all(0 <= t < 256 for t in report["tokens"]), report["tokens"]
