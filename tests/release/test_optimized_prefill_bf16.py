"""Production prefill fold, resident projection refusals and canonical dense row placement."""
import ast
import inspect
import os
import subprocess
import sys
import types

import pytest
import jax.numpy as jnp

from glm_tpu.greenfield.kernels import ws32_prefill_attention as frozen_attention
from glm_tpu.greenfield.kernels import ws32_prefill_dense_canonical as frozen_canonical
from glm_tpu.greenfield.kernels import ws32_prefill_dsa as frozen_dsa
from glm_tpu.greenfield.kernels import ws32_prefill_layer as frozen_layer
from glm_tpu.greenfield.kernels import ws32_prefill_linear as frozen_linear
from glm_tpu.greenfield.kernels import ws32_prefill_moe as frozen_moe
from glm_tpu.greenfield.kernels import ws32_prefill_window as frozen_window
from glm_tpu.greenfield.runtime import ws32_batched_prefill as frozen_prefill
from glm_tpu.optimized import bf16_resident
from glm_tpu.optimized import prefill
from glm_tpu.optimized import prefill_attention
from glm_tpu.optimized import prefill_bf16 as resident
from glm_tpu.optimized import prefill_dense_canonical
from glm_tpu.optimized import prefill_dsa
from glm_tpu.optimized import prefill_layer
from glm_tpu.optimized import prefill_linear
from glm_tpu.optimized import prefill_moe
from glm_tpu.optimized import prefill_window

# S2d c1: the former bind_dependencies recipe (prefill_challenger + prefill_bf16.bind_bf16_prefill)
# written out. Each production function is its frozen body with exactly these globals replaced:
# (production module, function) -> (frozen module, {frozen global: production callee}).
FOLD = {
    (prefill_linear, "ws32_prefill_linear_mapped"):
        (frozen_linear, {"fp8_block_matmul_f32": resident.resident_matmul_f32}),
    (prefill_linear, "ws32_prefill_dense_mapped"):
        (frozen_linear, {"fp8_block_matmul_f32": resident.resident_matmul_f32}),
    (prefill_moe, "ws32_prefill_moe_from_routes_mapped"):
        (frozen_moe, {"fp8_block_matmul_f32": resident.resident_matmul_f32,
                      "fp8_block_matmul": resident.resident_matmul}),
    (prefill_layer, "ws32_prefill_mlp_mapped"):
        (frozen_layer, {"ws32_prefill_dense_mapped": prefill_linear.ws32_prefill_dense_mapped,
                        "ws32_prefill_moe_from_routes_mapped": prefill_moe.ws32_prefill_moe_from_routes_mapped}),
    (prefill_attention, "ws32_prefill_prepare_attention_mapped"):
        (frozen_attention, {"ws32_prefill_linear_mapped": prefill_linear.ws32_prefill_linear_mapped}),
    (prefill_dsa, "ws32_prefill_dsa_inputs_mapped"):
        (frozen_dsa, {"fp8_block_matmul_f32": resident.resident_matmul_f32}),
    (prefill_dsa, "ws32_prefill_dsa_mapped"):
        (frozen_dsa, {"ws32_prefill_dsa_inputs_mapped": prefill_dsa.ws32_prefill_dsa_inputs_mapped,
                      "ws32_prefill_dsa_from_query_mapped": prefill_dsa._one_pass_selector}),
    (prefill_layer, "ws32_prefill_transformer_layer_mapped"):
        (frozen_layer, {"ws32_prefill_prepare_attention_mapped": prefill_attention.ws32_prefill_prepare_attention_mapped,
                        "ws32_prefill_index_share_attention_mapped": prefill_attention.prefill_index_share_lse_mapped,
                        "ws32_prefill_dsa_mapped": prefill_dsa.ws32_prefill_dsa_mapped,
                        "ws32_prefill_mlp_mapped": prefill_layer.ws32_prefill_mlp_mapped}),
    (prefill_dense_canonical, "ws32_prefill_dense_canonical_mapped"):
        (frozen_canonical, {"ws32_prefill_mlp_mapped": prefill_layer.ws32_prefill_mlp_mapped}),
    # the frozen window imports the canonical dense body locally; the executed copy at module scope
    (prefill_window, "ws32_prefill_layer_window_mapped"):
        (frozen_window, {"ws32_prefill_transformer_layer_mapped": prefill_layer.ws32_prefill_transformer_layer_mapped,
                         "ws32_prefill_mlp_mapped": prefill_layer.ws32_prefill_mlp_mapped,
                         "ws32_prefill_dense_canonical_mapped":
                             prefill_dense_canonical.ws32_prefill_dense_canonical_mapped}),
    (prefill, "ws32_batched_prefill_mapped"):
        (frozen_prefill, {"ws32_prefill_transformer_layer_mapped": prefill_layer.ws32_prefill_transformer_layer_mapped,
                          "ws32_prefill_layer_window_mapped": prefill_window.ws32_prefill_layer_window_mapped}),
    (prefill, "build_ws32_batched_prefill_program"):
        (frozen_prefill, {"ws32_batched_prefill_mapped": prefill._resident_prefill_mapped,
                          "ws32_decoder_weight_specs": bf16_resident.bf16_weight_specs}),
}


def _global_names(code):
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _global_names(const)
    return names


class _Canonical(ast.NodeTransformer):
    """Renames the replaced globals; function-local imports become absolute module names."""

    def __init__(self, package, renames, drop_imports=()):
        self.package, self.renames, self.drop_imports = package, renames, set(drop_imports)

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(self.renames.get(node.id, node.id), node.ctx), node)

    def visit_ImportFrom(self, node):
        if {alias.name for alias in node.names} <= self.drop_imports:
            return None
        base = self.package.rsplit(".", node.level - 1)[0] if node.level else ""
        module = ".".join(part for part in (base, node.module) if part)
        return ast.copy_location(ast.ImportFrom(module, node.names, 0), node)


@pytest.mark.parametrize("production,name", list(FOLD), ids=[f"{m.__name__}.{n}" for m, n in FOLD])
def test_prefill_fold_is_the_binding_table_written_out(production, name):
    frozen_module, replaced = FOLD[production, name]
    frozen_fn, production_fn = getattr(frozen_module, name), getattr(production, name)
    renames = {old: new.__name__ for old, new in replaced.items()}
    local = {"ws32_prefill_dense_canonical_mapped"} if frozen_module is frozen_window else set()
    expected = _Canonical(frozen_module.__package__, renames, local).visit(ast.parse(inspect.getsource(frozen_fn)))
    actual = _Canonical(production.__package__, {}).visit(ast.parse(inspect.getsource(production_fn)))
    assert ast.dump(expected) == ast.dump(actual)
    # every global the body reads is the object the binding supplied: a replacement, else the
    # frozen module's own global (the same helper, contract type and kernel object)
    used = (_global_names(frozen_fn.__code__) & set(frozen_fn.__globals__)) | local
    for old in sorted(used):
        want = replaced[old] if old in replaced else frozen_fn.__globals__[old]
        assert production_fn.__globals__[renames.get(old, old)] is want, old
    assert set(replaced) <= used
    # the frozen FP8 oracle keeps its own callees
    for old in replaced.keys() - local:
        assert frozen_fn.__globals__[old] is not replaced[old]


def test_resident_projection_rejects_raw_bits_or_scales():
    lhs = jnp.ones((2,128),jnp.bfloat16)
    for weight, scale in ((jnp.ones((128,128),jnp.uint8),None),
                          (jnp.ones((128,128),jnp.bfloat16),jnp.ones((1,1),jnp.float32))):
        with pytest.raises(ValueError): resident.resident_matmul(lhs,weight,scale)


def test_bf16_canonical_dense_cpu32():
    code=r'''
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
from glm_tpu.optimized.bf16_resident import bf16_resident_weights, bf16_weight_specs
from glm_tpu.optimized.prefill_bf16 import _adapt_weights
from glm_tpu.optimized.prefill_dense_canonical import ws32_prefill_dense_canonical_mapped as canonical
from glm_tpu.greenfield.kernels.ws32_prefill_dense_canonical import ws32_prefill_dense_canonical_mapped
from glm_tpu.greenfield.runtime.ws32_decoder import ws32_decoder_weight_specs
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
mesh=Mesh(np.asarray(jax.devices()).reshape(8,4),('expert','feature'))
config,weights,_=fixture(mesh)
bf16=bf16_resident_weights(mesh,config,weights)
assert canonical is not ws32_prefill_dense_canonical_mapped
rng=np.random.default_rng(1926)
for rows in (114,128):
    # Late live rows expose accidentally using ordinary B128 placement.
    x=jnp.asarray(rng.normal(size=(rows,config.geometry.hidden_size)),jnp.bfloat16)
    live=jnp.arange(rows)%3!=0
    def body(x,live,raw,resident):
        expected=ws32_prefill_dense_canonical_mapped(x,live,raw.layers[0].dense,
                   moe_contract=config.moe_contract,linear_interpret=True)
        actual=canonical(x,live,_adapt_weights(resident).layers[0].dense,
                   moe_contract=config.moe_contract,linear_interpret=True)
        return expected,actual
    fn=jax.jit(jax.shard_map(body,mesh=mesh,
        in_specs=(P(None,'feature'),P(),ws32_decoder_weight_specs(config),bf16_weight_specs(config)),
        out_specs=((P(None,'feature'),P(),P(),P()),)*2,check_vma=False))
    a,b=fn(x,live,weights,bf16)
    np.testing.assert_allclose(np.asarray(a[0]).astype(np.float32),np.asarray(b[0]).astype(np.float32),rtol=.02,atol=.01)
    for aa,bb in zip(a[1:],b[1:]): np.testing.assert_array_equal(np.asarray(aa),np.asarray(bb))
    assert np.asarray(b[-1]).all()
    assert not np.asarray(b[0])[~np.asarray(live)].any()
print('BF16 canonical B114/B128 output, routing, live masks and health checked')
'''
    env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=32')
    p=subprocess.run([sys.executable,'-c',code],env=env,text=True,capture_output=True,timeout=300)
    assert p.returncode==0,p.stdout+p.stderr
