"""Bounded adapter for a pinned public fused-EP experiment, never a default.

The external Apache-2.0 sources retain Meta's copyright and license headers.
They are authenticated before loading; importing this module does not load them.
The candidate uses EP32, BF16 activations and M64/single-buffer v4 tiles. Its
FP32 accumulation/reduction order differs from TP4/EP8 M8; trained parity and
TPU latency must be established separately before any model integration.
"""
from __future__ import annotations

import ast
import dataclasses
from hashlib import sha256
import importlib.util
from pathlib import Path
import sys
import types
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

SOURCE_PIN = '9cab26a702c448c40710f504360d0d9f78e227a7'
SOURCE_HASHES = {
    'fused_moe_rs.py': '4b7adbded8ffc5cf82a2a812f36627df2c2ce8a4518675d5da8adfe71647b55b',
    'gmm_v2_gather_scatter.py': '58ead45ffd5e644bb352ab220f8b7587371552fbf61db8bac9eac9475bfd9865',
    'gmm_fused_rs_nodedup.py': '9948e24a584cb5c123bbcd052a48e16afb4ec43791cbd0d515f4d73c752796bb',
}


class FusedWeights(NamedTuple):
    w1: object
    w1_scale: object
    w2: object
    w2_scale: object


def pack_weights(gate_bits, gate_scale, up_bits, up_scale, down_bits, down_scale):
    """Preserve raw FP8 bits and FP32 block scales in upstream E/K/N order.

Output-column scales are expanded from blocks of 128 to individual columns;
the kernel expects [E, K/128, 1, N]. This increases scale storage by 128x.
Ownership resharding is separate and must be measured/admitted by the caller.
"""
    if gate_bits.ndim != 3:
        raise ValueError('gate weights must be E/I/H')
    e, i, h = gate_bits.shape
    if min(e, i, h) <= 0 or i % 128 or h % 128:
        raise ValueError('weights require positive 128-aligned dimensions')
    for bits, scale, shape in ((gate_bits, gate_scale, (e, i, h)),
                              (up_bits, up_scale, (e, i, h)),
                              (down_bits, down_scale, (e, h, i))):
        if (bits.shape != shape or bits.dtype != jnp.uint8
                or scale.shape != (e, shape[1]//128, shape[2]//128)
                or scale.dtype != jnp.float32):
            raise ValueError('FP8 bit or FP32 block-scale layout differs')

    def weight(bits):
        return lax.bitcast_convert_type(bits.transpose(0, 2, 1), jnp.float8_e4m3fn)

    def scales(value):
        return jnp.repeat(value.transpose(0, 2, 1), 128, axis=2)[:, :, None, :]

    return FusedWeights(jnp.concatenate((weight(gate_bits), weight(up_bits)), axis=2),
        jnp.concatenate((scales(gate_scale), scales(up_scale)), axis=3),
        weight(down_bits), scales(down_scale))


def pad_rows(hidden, routes, weights, *, ep_size=32):
    """Explicit padding with zero hidden/route weights and distinct dummy routes."""
    if (hidden.ndim != 2 or hidden.dtype != jnp.bfloat16 or hidden.shape[0] < 1
            or routes.shape != (hidden.shape[0], 8) or routes.dtype != jnp.int32
            or weights.shape != routes.shape or weights.dtype != jnp.float32
            or ep_size != 32):
        raise ValueError('EP32 candidate requires BF16 hidden, eight routes and FP32 weights')
    padding = (-hidden.shape[0]) % ep_size
    return (jnp.pad(hidden, ((0, padding), (0, 0))),
        jnp.concatenate((routes, jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32), (padding, 8)))),
        jnp.pad(weights, ((0, padding), (0, 0))))


def _bf16_activation(acc, fuse_act):
    if fuse_act is None:
        return acc
    if fuse_act != 'silu':
        raise ValueError('GLM candidate supports silu only')
    gate, up = jnp.split(acc.astype(jnp.bfloat16), 2, axis=-1)
    return (gate * jax.nn.sigmoid(gate) * up).astype(jnp.bfloat16)


def _single_buffer_kernel(source):
    """Consume each weight tile before reusing its sole DMA destination.

    The public loops refill a rotating buffer before compute and prefetch the
    next expert after step zero. For one buffer, defer both until consumption.
    Exact AST guards bind this correction to the authenticated source version.
    """
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'kernel_main_fused_rs')
    count = 0
    for loop in ast.walk(function):
        if not isinstance(loop, ast.For) or not isinstance(loop.target, ast.Name) or loop.target.id != 'step':
            continue
        which = next((i for i in (1, 2) if ast.dump(loop.iter) == ast.dump(
            ast.parse(f'range(total_w{i}_steps)', mode='eval').body)), None)
        if which is None:
            continue
        target = ast.parse(f'step + num_w{which}_bufs < total_w{which}_steps', mode='eval').body
        at = [i for i, n in enumerate(loop.body) if isinstance(n, ast.If)
              and ast.dump(n.test) == ast.dump(target)]
        if len(at) != 1:
            raise ValueError('public weight refill loop changed')
        i = at[0]
        expected = ast.parse(f'compute_gmm{which}_tile(buf_id, _n{which}, _k{which}, gm_id)').body[0]
        if ast.dump(loop.body[i+1]) != ast.dump(expected):
            raise ValueError('public weight consumption order changed')
        loop.body[i], loop.body[i+1] = loop.body[i+1], loop.body[i]
        if which == 1:
            cross = loop.body[i+2]
            if not isinstance(cross, ast.If) or ast.dump(cross.test) != ast.dump(ast.parse('step == 0', mode='eval').body):
                raise ValueError('public cross-expert prefetch changed')
            cross.test = ast.parse('step == total_w1_steps - 1', mode='eval').body
        count += 1
    if count != 2:
        raise ValueError('expected both weight-consumption loops')
    return ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))


def build_candidate(mesh, source_root, *, routed_scaling_factor):
    """Compile padding and activation redistribution with the routed kernel.

    Weights must already be converted to FusedWeights and sharded along E over
    all 32 chips. Weight conversion is a separate startup/HBM requirement.
    """
    if tuple(mesh.axis_names) != ('expert', 'feature') or mesh.devices.shape != (8, 4):
        raise ValueError('candidate input/output requires the existing expert8/feature4 mesh')
    epmesh = Mesh(mesh.devices.reshape(32), ('ep',))
    local = load_public_candidate(source_root)

    def body(x, ids, rw, w):
        return local(x, w.w1, w.w1_scale, None, w.w2, w.w2_scale, None, None, None,
            jnp.array([lax.axis_index('ep')*w.w1.shape[0]], jnp.int32),
            rw.astype(jnp.bfloat16), ids, activation='silu', topk=8, ep_size=32,
            ep_axis_name='ep', sp_enabled=False, fp8_post_gather=False)

    mapped = jax.shard_map(body, mesh=epmesh,
        in_specs=(P(), P(), P(), FusedWeights(*(P('ep'),)*4)), out_specs=P(), check_vma=False)

    def execute(hidden, routes, route_weights, weights):
        padded, ids, rw = pad_rows(hidden, routes, route_weights)
        out = mapped(padded, ids, rw, weights)[:hidden.shape[0]]
        return (out*jnp.asarray(routed_scaling_factor, jnp.bfloat16)).astype(jnp.bfloat16)

    return jax.jit(execute,
        in_shardings=(NamedSharding(mesh, P(None, 'feature')), NamedSharding(mesh, P()),
            NamedSharding(mesh, P()), FusedWeights(*(NamedSharding(epmesh, P('ep')),)*4)),
        out_shardings=NamedSharding(mesh, P(None, 'feature')))


def load_public_candidate(root):
    """Load authenticated source into private modules and return its local body.

    v4 edits are confined to these modules: smaller buffers, no FP8 activation
    quantization, FP32-scale/BF16-weight decoding, and BF16 SiLU boundaries.
    The TP4 feature-reduction tree and K128 accumulation order are not preserved.
    """
    root = Path(root).resolve()
    sources = {}
    for name, digest in SOURCE_HASHES.items():
        raw = (root/name).read_bytes()
        if sha256(raw).hexdigest() != digest:
            raise ValueError('public source digest differs: '+name)
        sources[name] = raw.decode()
    prefix = '_glm_public_ep_' + sha256(str(root).encode()).hexdigest()[:12]
    if prefix in sys.modules:
        if not hasattr(sys.modules[prefix], 'candidate'):
            raise RuntimeError('previous public adapter load did not complete')
        return sys.modules[prefix].candidate
    package = types.ModuleType(prefix)
    package.__path__ = [str(root)]
    sys.modules[prefix] = package
    loaded = {}
    for name in ('gmm_v2_gather_scatter', 'gmm_fused_rs_nodedup'):
        spec = importlib.util.spec_from_file_location(prefix+'.'+name, root/(name+'.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        exec(compile(sources[name+'.py'], str(root/(name+'.py')), 'exec'), module.__dict__)
        loaded[name] = module
    common = loaded['gmm_v2_gather_scatter']
    kernel = loaded['gmm_fused_rs_nodedup']
    original_tiles = kernel._select_fused_rs_block_sizes

    def v4_tiles(**kwargs):
        info = kernel.pltpu.get_tpu_info()
        if info.generation != 4 or info.vmem_capacity_bytes != 16*1024**2:
            raise ValueError('candidate is restricted to v4 VMEM geometry')
        return dataclasses.replace(original_tiles(**kwargs), tile_m=64,
                                   num_w1_bufs=1, num_w2_bufs=1)

    # Upstream's FP8 path otherwise quantizes BF16 activations back to FP8.
    kernel.get_maybe_quantize_lhs = lambda *args, **kwargs: False
    kernel._select_fused_rs_block_sizes = v4_tiles
    tree = ast.parse(sources['gmm_v2_gather_scatter.py'])
    inner = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'inner_kernel')
    old = ast.parse('block_rhs_slice.astype(jnp.bfloat16) * rhs_scale_slice.astype(jnp.bfloat16)', mode='eval').body
    new = ast.parse('(block_rhs_slice.astype(jnp.float32) * rhs_scale_slice.astype(jnp.float32)).astype(jnp.bfloat16)', mode='eval').body

    class DecodeScale(ast.NodeTransformer):
        count = 0

        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == 'scaled_rhs_slice' for t in node.targets):
                if ast.dump(node.value) != ast.dump(old):
                    raise ValueError('upstream scale expression changed')
                node.value = new
                self.count += 1
            return self.generic_visit(node)

    patch = DecodeScale()
    inner = patch.visit(inner)
    if patch.count != 1:
        raise ValueError('expected one scale-decoding expression')
    scope = dict(common.__dict__, apply_act_fn=_bf16_activation)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[inner], type_ignores=[])),
                 str(root/'gmm_v2_gather_scatter.py'), 'exec'), scope)
    kernel.inner_kernel = scope['inner_kernel']
    # Recompile only the authenticated kernel body with safe one-buffer reuse.
    exec(compile(_single_buffer_kernel(sources['gmm_fused_rs_nodedup.py']),
                 str(root/'gmm_fused_rs_nodedup.py'), 'exec'), kernel.__dict__)
    # Extract only the public per-chip call path; avoid loading its serving stack.
    names = {'_assert_fused_rs_smem_safe', '_compute_rs_routing',
             '_all_gather_token_hidden', 'moe_gmm_local_rs_nodedup'}
    body = [n for n in ast.parse(sources['fused_moe_rs.py']).body
            if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in body} != names:
        raise ValueError('public local call path changed')
    scope = dict(jax=jax, jnp=jnp, EXPERT='ep', _FUSED_RS_MAX_SAFE_SIZE_M=262144,
        _select_fused_rs_block_sizes=v4_tiles,
        gmm_v2_fused_rs_nodedup=kernel.gmm_v2_fused_rs,
        _recover_quant_block_size=common._recover_quant_block_size)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(root/'fused_moe_rs.py'), 'exec'), scope)
    package.candidate = scope['moe_gmm_local_rs_nodedup']
    return package.candidate
