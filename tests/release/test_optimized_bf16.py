"""BF16-resident non-routed weights: exact decode, same tokens, state within one BF16 ulp.

CPU semantics on the forced 32-device mesh; the pod timing is in docs/perf.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.optimized.bf16_resident import decode_fp8_table
from glm_tpu.optimized.ws32_decoder_challenger import Ws32PerfOptions


def test_decode_table_matches_reference_dequantizer_bitwise():
    from glm_tpu.greenfield.kernels.reference.fp8 import dequantize_fp8_bits_block_weight

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


def test_bf16_resident_step_matches_frozen_tokens_cpu32():
    lse_attention, dsa_two_stage, fused = False, True, False
    code = r'''
import json
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
from glm_tpu.greenfield.runtime import ws32_batched_prefill as b, ws32_decoder as d
from glm_tpu.greenfield.runtime.ws32_sampled_request import build_ws32_sampled_prefill_program
from glm_tpu.greenfield.kernels.ws32_sampling import NucleusConfig
from glm_tpu.optimized.bf16_resident import bf16_resident_weights, bf16_weight_specs
from glm_tpu.optimized.fp8_routed_experts import RoutedProjectionConfig
from glm_tpu.optimized.ws32_decoder_challenger import Ws32PerfOptions, build_ws32_challenger_decoder_program
from glm_tpu.optimized.request_loop import build_packed_decoder_program
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
mesh = Mesh(np.asarray(jax.devices(), object).reshape(8, 4), ('expert', 'feature'))
def put(v, spec=P()): return jax.device_put(v, NamedSharding(mesh, spec))
config, weights, wk = fixture(mesh, panel_geometry=True)
wk = tuple(put(v) for v in wk)
rope = put(jnp.asarray(d.build_ws32_main_rope_table(config), jnp.bfloat16))
interpret = dict(sparse_attention_interpret=True, linear_interpret=True)
bf16 = bf16_resident_weights(mesh, config, weights)
# Every leaf keeps the frozen partition spec and shape; only non-routed FP8 tables changed dtype.
for leaf, spec in zip(jax.tree.leaves(bf16), jax.tree.leaves(bf16_weight_specs(config), is_leaf=lambda x: isinstance(x, P))):
    assert leaf.sharding.is_equivalent_to(NamedSharding(mesh, spec), leaf.ndim), (leaf.shape, leaf.sharding.spec, spec)
prefill = build_ws32_sampled_prefill_program(mesh, config, sampling=NucleusConfig(), block_rows=2, key_tile=128, **interpret)
state = b.make_ws32_batched_prefill_state(mesh, config, prompt_length=3)
first = prefill.execute(put(jnp.array([30, 31], jnp.int32)), put(jnp.int32(2)), state, weights, wk, rope, put(jnp.float32(.5)))
last = prefill.execute(put(jnp.array([32, -1], jnp.int32)), put(jnp.int32(1)), first.state, weights, wk, rope, put(jnp.float32(.5)))
ds, token = b.finish_ws32_batched_prefill(last)
frozen = jax.jit(d.build_ws32_decoder_program(mesh, config, **interpret).execute)
tiles = RoutedProjectionConfig(block_shape=(128, 128), output_tile=256, contraction_tile=256)
challenger = build_ws32_challenger_decoder_program(
    mesh, config, options=Ws32PerfOptions(sampler='greedy', bf16_resident=True, lse_attention=LSE_ATTENTION, dsa_two_stage=DSA_TWO_STAGE, fused_feature_reductions=FUSED, routed_projection=tiles), **interpret)
unfused = build_ws32_challenger_decoder_program(
    mesh, config, options=Ws32PerfOptions(sampler='greedy', bf16_resident=True, lse_attention=LSE_ATTENTION, dsa_two_stage=DSA_TWO_STAGE, routed_projection=tiles), **interpret) if FUSED else None
packed = build_packed_decoder_program(mesh, config, **interpret)
report = dict(tokens=[], mismatch_fraction={}, max_abs=[], valid=True)
ref_state, ch_state, ref_token = ds, ds, token
for step in range(3):
    ref = frozen(ref_token, ref_state, weights, rope)
    out = challenger.execute(ref_token, ch_state, bf16, rope)
    compact = packed.execute(ref_token, ch_state, bf16, rope)
    for a, z in zip(jax.tree.leaves(out), jax.tree.leaves(compact.decoded)):
        np.testing.assert_array_equal(np.asarray(a).view(np.uint8), np.asarray(z).view(np.uint8))
    np.testing.assert_array_equal(np.asarray(compact.metadata),
        [int(out.next_token[0]),1,int(out.state.position[0]),int(out.state.context_lengths[0])])
    if unfused is not None:
        separate = unfused.execute(ref_token, ch_state, bf16, rope)
        for a, z in zip(jax.tree.leaves(separate),jax.tree.leaves(out)):
            np.testing.assert_array_equal(np.asarray(a).view(np.uint8), np.asarray(z).view(np.uint8))
    report['tokens'].append([int(ref.next_token[0]), int(out.next_token[0])])
    report['valid'] = report['valid'] and bool(np.asarray(out.state.contract_valid).all())
    np.testing.assert_array_equal(np.asarray(ref.state.selected_positions), np.asarray(out.state.selected_positions))
    kv_ref, kv_out = np.asarray(ref.state.kv_cache_local).astype(np.float32), np.asarray(out.state.kv_cache_local).astype(np.float32)
    report['mismatch_fraction'][f'kv_step{step}'] = float(np.mean(kv_ref != kv_out))
    report['max_abs'].append(float(np.max(np.abs(kv_ref - kv_out))))
    ref_state, ch_state, ref_token = ref.state, out.state, ref.next_token
print(json.dumps(report))
'''
    code = code.replace("LSE_ATTENTION", repr(lse_attention)).replace("DSA_TWO_STAGE", repr(dsa_two_stage)).replace("FUSED", repr(fused))
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               XLA_FLAGS=(os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=32").strip())
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=1500)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["valid"]
    # Same tokens and the same DSA selections; the KV/latent values differ only by
    # FP32 accumulation order inside one contraction (at most one BF16 ulp on a
    # small fraction of elements) because the MXU operands are bit-identical.
    assert all(a == b for a, b in report["tokens"]), report["tokens"]
    assert all(fraction < 0.01 for fraction in report["mismatch_fraction"].values()), report
    assert max(report["max_abs"]) <= 0.0625, report
