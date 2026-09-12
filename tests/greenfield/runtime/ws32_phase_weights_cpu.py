"""One actual small-geometry producer: original full versus WK-only programs."""

from dataclasses import replace
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from glm_tpu.greenfield.runtime.ws32_decoder import (
    Ws32DecoderConfig, Ws32ExactDsaRawLayerWeights,
    build_ws32_exact_dsa_materializer_program, ws32_decoder_weight_names,
)
from glm_tpu.greenfield.types import ModelGeometry
from scripts.greenfield.ws32_phase_weights import PhaseWeights


mesh = Mesh(np.asarray(jax.devices()).reshape(8, 4), ("expert", "feature"))
geometry = ModelGeometry.from_hf_config(json.loads(Path("configs/glm-5.2-fp8-config.json").read_text()))
production_geometry = geometry
geometry = replace(geometry, num_layers=1, first_dense_layers=1, hidden_size=512, q_lora_rank=128,
                   indexer_types=("full",), mlp_layer_types=("dense",))
config = Ws32DecoderConfig(geometry, 8192, exact_dsa=True, host_main_rope_table=True)
raw_config = replace(config, exact_dsa=False)
rng = np.random.default_rng(812)


def put(value, spec):
    return jax.device_put(value, NamedSharding(mesh, spec))


def bits(shape, spec):
    return put(rng.normal(0, .2, shape).astype(ml_dtypes.float8_e4m3fn).view(np.uint8), spec)


def scale(shape, spec):
    return put(rng.uniform(.1, 2., shape).astype(np.float32), spec)


raw = Ws32ExactDsaRawLayerWeights(
    bits((128, 512), P(None, "feature")), scale((1, 4), P(None, "feature")),
    bits((576, 512), P(None, "feature")), scale((5, 4), P(None, "feature")),
    bits((4096, 128), P("expert", None)), scale((32, 1), P("expert", None)),
    bits((128, 512), P(None, "feature")), scale((1, 4), P(None, "feature")),
    put(rng.normal(size=(32, 512)).astype(ml_dtypes.bfloat16), P("expert", "feature")),
)
materializer = build_ws32_exact_dsa_materializer_program(mesh, config)
decoded = jax.jit(materializer.decode)((raw,))
jax.block_until_ready(decoded)
reference = jax.jit(materializer.promote)(decoded)[0].wk_weight
jax.block_until_ready(reference)
names = ws32_decoder_weight_names(raw_config)
arrays = {name: object() for name in jax.tree.leaves(names)}
arrays[names.layers[0].dsa.wk_bits_local] = raw.wk_bits_local
arrays[names.layers[0].dsa.wk_scale_local] = raw.wk_scale_local
owner = PhaseWeights(arrays, config)
jobs = owner.wk_jobs(mesh)
compiled = {name: program.lower(*args).compile() for name, program, args in jobs}
def call(stage, graph, args):
    return compiled[graph](*args)
owner.materialize_wk(call, phase=lambda name, action: action())
np.testing.assert_array_equal(np.asarray(owner.wk[0]).view(np.uint32),
                              np.asarray(reference).view(np.uint32))
assert len(owner.wk) == 1 and owner.phase == "prefill"
# Source/producer topology remains explicit, not a full-pod collective.
text = str(jobs[0][1].lower(*jobs[0][2]).compiler_ir("stablehlo"))
assert text.count('"stablehlo.all_gather"') == 2
assert "stablehlo.all_reduce" not in text
# Full production names/21 producers and original sharding, with no model data
# allocation or exact decode-tree creation. Numerical test above is reduced.
production = Ws32DecoderConfig(production_geometry, 262656, exact_dsa=True,
    strategy_nd_dense=True, host_main_rope_table=True)
raw_names = ws32_decoder_weight_names(replace(production, exact_dsa=False, strategy_nd_dense=False))
abstract_arrays = {name: object() for name in jax.tree.leaves(raw_names)}
for layer in production.full_index_slots:
    abstract_arrays[raw_names.layers[layer].dsa.wk_bits_local] = jax.ShapeDtypeStruct(
        (128, 6144), jnp.uint8, sharding=NamedSharding(mesh, P(None, "feature")))
    abstract_arrays[raw_names.layers[layer].dsa.wk_scale_local] = jax.ShapeDtypeStruct(
        (1, 48), jnp.float32, sharding=NamedSharding(mesh, P(None, "feature")))
production_owner = PhaseWeights(abstract_arrays, production)
production_jobs = production_owner.wk_jobs(mesh)
assert len(production_owner._wk_sources()) == 21
assert [name for name, _, _ in production_jobs] == ["wk_decode", "wk_promote"]
for name, fn, args in production_jobs:
    result = jax.eval_shape(fn, *args)
    assert result.shape == (128, 6144)
    assert str(result.dtype) == ("bfloat16" if name == "wk_decode" else "float32")
print("WK_PHASE_CPU32_PASS")
