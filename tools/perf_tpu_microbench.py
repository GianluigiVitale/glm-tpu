"""TPU v4 microbenchmarks: frozen WS32 bodies versus the glm_tpu.perf challengers.

Runs as eight processes (one per pod host, ``jax.distributed``) on the real
``expert=8 x feature=4`` physical mesh with SYNTHETIC FP8 weights at GLM-5.2's
exact shapes. Routed expert balance and data-dependent selections do depend
on weight values; these measurements do not establish trained-model throughput
or quality and produce no sealed evidence; every rank writes one JSON
receipt with per-call wall-time percentiles.

    # on every host (rank from the hostname suffix), coordinator = worker 0
    JAX_PLATFORMS=tpu PYTHONPATH=. python tools/perf_tpu_microbench.py \
        --coordinator 192.168.0.37:8476 --output ~/glm-run/<tag> --which moe,sampler,tiles

``--which step`` additionally builds the complete 78-layer decode step (frozen
greedy versus challenger greedy, capacity 8192) on synthetic weights; that
needs ~23 GB of HBM per chip for raw weights and a long compile.
``--which prefill_model --prompt-length 2048`` measures a complete synthetic
prompt with canonical dense B128 placement and an explicit memory check;
``--prefill-variants frozen,p1p2,p1p2_bf16`` selects its comparisons.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _rank_from_hostname() -> int:
    return int(socket.gethostname().rsplit("-w-", 1)[1])


def _timeit(fn: Any, args: tuple, *, warmup: int, iters: int) -> dict[str, float]:
    import jax
    import numpy as np

    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(iters):
        started = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append((time.perf_counter() - started) * 1e3)
    arr = np.asarray(samples)
    return dict(
        p50_ms=float(np.percentile(arr, 50)),
        p90_ms=float(np.percentile(arr, 90)),
        p99_ms=float(np.percentile(arr, 99)),
        min_ms=float(arr.min()),
        mean_ms=float(arr.mean()),
        samples=int(arr.size),
    )


def _chain_timeit(build, args, *, counts=(1, 17), warmup: int = 10, iters: int = 60) -> dict[str, Any]:
    """Time whole programs that apply a body ``n`` times inside one jit.

    ``build(n)`` returns a jitted program whose device work is ``n`` dependent
    applications of the body.  The slope between the two counts is the per-body
    device time without host dispatch; the ``n=1`` program is the request-loop
    reality (one dispatch per token) and is reported too.
    """

    out: dict[str, Any] = {}
    times = {}
    for n in counts:
        program = build(n)
        started = time.perf_counter()
        compiled = program.lower(*args).compile()
        compile_s = time.perf_counter() - started
        timing = _timeit(compiled, args, warmup=warmup, iters=iters)
        timing["compile_s"] = compile_s
        out[f"n{n}"] = timing
        times[n] = timing["p50_ms"]
    lo, hi = counts[0], counts[-1]
    out["per_body_ms_from_slope"] = (times[hi] - times[lo]) / (hi - lo)
    return out


def _ws32_mesh():
    """expert=(x,y), feature=z from the observed physical coordinates."""

    import jax
    import numpy as np
    from jax.sharding import Mesh

    by_coordinates = {tuple(int(c) for c in d.coords): d for d in jax.devices()}
    rows = [[by_coordinates[(x, y, z)] for z in range(4)] for x in range(2) for y in range(4)]
    return Mesh(np.asarray(rows, dtype=object), ("expert", "feature"))


def _sharded_random(mesh, shape, spec, kind, seed):
    """Device-side random shard generation (no host copy of large tables)."""

    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import NamedSharding, PartitionSpec as P

    sharding = NamedSharding(mesh, spec)
    local_shape = sharding.shard_shape(shape)

    axes = {axis for entry in tuple(spec) if entry is not None
            for axis in (entry if isinstance(entry, tuple) else (entry,))}
    replicated = not axes

    def body(seed_value):
        # seed is a traced operand so one compiled program serves every array
        # of the same (shape, spec, kind); 78 layers place in about a minute.
        key = jax.random.PRNGKey(seed_value[0])
        # Fold only partitioned axes. Other axes are replicas and must receive
        # identical bytes, including partially replicated matrices such as P("expert").
        for axis in ("expert", "feature"):
            if axis in axes:
                key = jax.random.fold_in(key, lax.axis_index(axis))
        if kind == "fp8":
            values = jax.random.normal(key, local_shape, jnp.float32) * 0.02
            return lax.bitcast_convert_type(values.astype(jnp.float8_e4m3fn), jnp.uint8)
        if kind == "scale":
            return jax.random.uniform(key, local_shape, jnp.float32, 0.5, 1.5)
        if kind == "bf16":
            return (jax.random.normal(key, local_shape, jnp.float32) * 0.1).astype(jnp.bfloat16)
        if kind == "ones_bf16":
            return jnp.ones(local_shape, jnp.bfloat16)
        if kind == "zeros_f32":
            return jnp.zeros(local_shape, jnp.float32)
        raise ValueError(kind)

    fn = _generator_program(mesh, tuple(shape), spec, kind, local_shape, replicated, body)
    seed_array = jax.device_put(jnp.asarray([seed], jnp.int32), NamedSharding(mesh, P()))
    return fn(seed_array)


_GENERATORS: dict = {}


def _generator_program(mesh, shape, spec, kind, local_shape, replicated, body):
    import jax
    from jax.sharding import PartitionSpec as P

    cache_key = (shape, tuple(spec), kind)
    if cache_key not in _GENERATORS:
        _GENERATORS[cache_key] = jax.jit(jax.shard_map(
            body, mesh=mesh, in_specs=(P(),), out_specs=spec, check_vma=False))
    return _GENERATORS[cache_key]


def bench_moe(mesh, report: dict, *, iters: int) -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax
    from jax.sharding import NamedSharding, PartitionSpec as P

    from glm_tpu.greenfield.kernels.reference.moe import GlmMoeNumericalContract
    from glm_tpu.greenfield.kernels.ws32 import ws32_moe_pallas_from_routes_mapped
    from glm_tpu.perf.fp8_routed_experts import (
        RoutedProjectionConfig,
        ws32_moe_grouped_routes_mapped,
    )

    contract = GlmMoeNumericalContract(stage_size=8)  # 6144 / 2048 / 256 experts / top-8
    H, I, E = contract.hidden_size, contract.intermediate_size, contract.num_experts
    specs = (
        P(None, "feature"), P(), P(),
        P("expert", None, "feature"), P("expert", None, "feature"),
        P("expert", None, "feature"), P("expert", None, "feature"),
        P("expert", "feature", None), P("expert", "feature", None),
        P(None, "feature"), P(None, "feature"), P(None, "feature"), P(None, "feature"),
        P("feature", None), P("feature", None),
    )
    shapes = (
        (1, H), (1, 8), (1, 8),
        (E, I, H), (E, I // 128, H // 128), (E, I, H), (E, I // 128, H // 128),
        (E, H, I), (E, H // 128, I // 128),
        (I, H), (I // 128, H // 128), (I, H), (I // 128, H // 128),
        (H, I), (H // 128, I // 128),
    )
    kinds = ("bf16", None, None, "fp8", "scale", "fp8", "scale", "fp8", "scale",
             "fp8", "scale", "fp8", "scale", "fp8", "scale")
    values = []
    for i, (shape, spec, kind) in enumerate(zip(shapes, specs, kinds)):
        if kind is None:
            values.append(None)
        else:
            values.append(_sharded_random(mesh, shape, spec, kind, seed=100 + i))
    replicated = NamedSharding(mesh, P())
    weights = jax.device_put(np.full((1, 8), 0.125, np.float32), replicated)
    routes = {
        "normal_one_per_owner": np.asarray([[0, 33, 66, 99, 132, 165, 198, 231]], np.int32),
        "concentrated_owner0": np.asarray([[0, 1, 2, 3, 4, 5, 6, 7]], np.int32),
        "two_per_owner": np.asarray([[0, 1, 64, 65, 128, 129, 192, 193]], np.int32),
    }

    def program(fn, n, **kw):
        def body(*v):
            def step(_, hidden):
                return fn(hidden, *v[1:], contract=contract, **kw)
            return lax.fori_loop(0, n, step, v[0])
        return jax.jit(jax.shard_map(body, mesh=mesh, in_specs=specs,
                                     out_specs=P(None, "feature"), check_vma=False))

    variants = {"frozen_pallas": (ws32_moe_pallas_from_routes_mapped, {})}
    for tiles in ((128, 128), (256, 256), (512, 512)):
        cfg = RoutedProjectionConfig(block_shape=(128, 128), output_tile=tiles[0],
                                     contraction_tile=tiles[1])
        variants[f"grouped_t{tiles[0]}x{tiles[1]}"] = (ws32_moe_grouped_routes_mapped, dict(config=cfg))
    out: dict[str, Any] = {}
    for name, (fn, kw) in variants.items():
        for route_name, route in routes.items():
            args = list(values)
            args[1] = jax.device_put(route, replicated)
            args[2] = weights
            args = tuple(args)
            timing = _chain_timeit(lambda n, fn=fn, kw=kw: program(fn, n, **kw), args, iters=iters)
            out[f"{name}/{route_name}"] = timing
            print(f"moe {name} {route_name}: per-layer(slope) {timing['per_body_ms_from_slope']:.3f} ms; "
                  f"n1 p50 {timing['n1']['p50_ms']:.3f} ms", flush=True)
    programs = {name: program(fn, 1, **kw) for name, (fn, kw) in variants.items()}
    # Cross-check: grouped vs frozen output on synthetic weights (same contract).
    args = list(values); args[1] = jax.device_put(routes["normal_one_per_owner"], replicated); args[2] = weights
    a = programs["frozen_pallas"](*args)
    b = programs["grouped_t512x512"](*args)
    # Compare this process's addressable shards only (global fetch is not allowed).
    mism = 0
    diff = 0.0
    for sa, sb in zip(a.addressable_shards, b.addressable_shards):
        x, y = np.asarray(sa.data), np.asarray(sb.data)
        mism += int(np.count_nonzero(x.view(np.uint16) != y.view(np.uint16)))
        diff = max(diff, float(np.max(np.abs(x.astype(np.float32) - y.astype(np.float32)))))
    out["local_shards_max_abs_diff_grouped512_vs_frozen"] = diff
    out["local_shards_bitwise_mismatches_grouped512_vs_frozen"] = mism
    report["moe_layer_body"] = out


def bench_sampler(mesh, report: dict, *, iters: int) -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax
    from jax.sharding import NamedSharding, PartitionSpec as P

    from glm_tpu.greenfield.kernels.ws32_io import Ws32SplitGreedySampleResult, ws32_split_final_sample_mapped
    from glm_tpu.greenfield.kernels.ws32_sampling import NucleusConfig, ws32_split_nucleus_sample_mapped
    from glm_tpu.perf.ws32_sampling_candidates import ws32_split_nucleus_sample_candidates_mapped

    H, V = 6144, 154880
    update = _sharded_random(mesh, (1, H), P(None, "feature"), "bf16", 1)
    residual = _sharded_random(mesh, (1, H), P(None, "feature"), "bf16", 2)
    norm = _sharded_random(mesh, (H,), P("feature"), "ones_bf16", 3)
    head_w = _sharded_random(mesh, (V, H), P("expert", "feature"), "bf16", 4)
    uniform = jax.device_put(np.float32(0.37), NamedSharding(mesh, P()))
    config = NucleusConfig()
    common = dict(hidden_size=H, vocab_size=V)
    specs = (P(None, "feature"), P(None, "feature"), P("feature"), P("expert", "feature"))
    out_specs = Ws32SplitGreedySampleResult(P(), P(), P(None, "feature"))
    def program(head, n, takes_uniform):
        def body(update, residual, norm_w, lm_head, *u):
            def step(_, current):
                result = head(current, residual, norm_w, lm_head, *u)
                # Make the chain depend on the sampled token so XLA cannot
                # drop the sort/sampling work as dead code.
                return jnp.where(result.token_id[0] >= 0, result.final_residual_local,
                                 jnp.zeros_like(result.final_residual_local))
            final = lax.fori_loop(0, n, step, update)
            result = head(final, residual, norm_w, lm_head, *u)
            return Ws32SplitGreedySampleResult(result.token_id, result.contract_valid, final)
        in_specs = specs + ((P(),) if takes_uniform else ())
        return jax.jit(jax.shard_map(body, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False))

    heads = {
        "frozen_greedy": (lambda a, b, c, d: ws32_split_final_sample_mapped(a, b, c, d, **common), False),
        "frozen_nucleus_full_sort": (lambda a, b, c, d, u: ws32_split_nucleus_sample_mapped(a, b, c, d, u, config=config, **common), True),
    }
    for k in (64, 256, 1024):
        heads[f"candidates_k{k}"] = (lambda a, b, c, d, u, k=k: ws32_split_nucleus_sample_candidates_mapped(
            a, b, c, d, u, config=config, candidates_per_shard=k, **common), True)
    out: dict[str, Any] = {}
    tokens = {}
    for name, (head, takes_uniform) in heads.items():
        args = (update, residual, norm, head_w) + ((uniform,) if takes_uniform else ())
        timing = _chain_timeit(lambda n, head=head, tu=takes_uniform: program(head, n, tu), args, counts=(0, 16), iters=iters)
        result = program(head, 0, takes_uniform)(*args)
        tokens[name] = int(np.asarray(result.token_id)[0])
        out[name] = timing
        print(f"sampler {name}: per-head(slope) {timing['per_body_ms_from_slope']:.3f} ms; n0 p50 {timing['n0']['p50_ms']:.3f} ms token {tokens[name]}", flush=True)
    out["tokens"] = tokens
    report["sampler_head"] = out


def _e4m3_to_bf16_bits(bits: Any) -> Any:
    """Exact float8_e4m3fn -> bfloat16 conversion with integer ops only (no f32 round trip).

    Normal values: sign<<15 | (e+120)<<7 | m<<4.  Subnormals (e == 0, m > 0)
    are m * 2^-9, renormalised per mantissa value.  NaN (e == 15, m == 7) maps
    to a BF16 NaN.  Verified against ``astype`` for all 256 encodings on CPU.
    """
    import jax.numpy as jnp
    from jax import lax

    u = bits.astype(jnp.uint16)
    sign = (u & 0x80) << 8
    exponent = (u >> 3) & 0xF
    mantissa = u & 0x7
    normal = ((exponent + 120) << 7) | (mantissa << 4)
    # subnormal table for m = 1..7 (bf16 exponent/mantissa bits), m = 0 -> zero
    sub = jnp.where(mantissa == 1, (118 << 7),
          jnp.where(mantissa == 2, (119 << 7),
          jnp.where(mantissa == 3, (119 << 7) | (1 << 6),
          jnp.where(mantissa == 4, (120 << 7),
          jnp.where(mantissa == 5, (120 << 7) | (1 << 5),
          jnp.where(mantissa == 6, (120 << 7) | (2 << 5),
          jnp.where(mantissa == 7, (120 << 7) | (3 << 5), 0)))))))
    magnitude = jnp.where(exponent == 0, sub, normal)
    nan = (exponent == 15) & (mantissa == 7)
    magnitude = jnp.where(nan, jnp.uint16(0x7FC0), magnitude)
    return lax.bitcast_convert_type((sign | magnitude).astype(jnp.uint16), jnp.bfloat16)


def _e4m3_bits_to_f32_bits(byte: Any) -> Any:
    """e4m3fn byte (as uint32 lanes) -> IEEE f32 bit pattern of the same value (exact)."""
    import jax.numpy as jnp

    sign = (byte & 0x80) << 24
    exponent = (byte >> 3) & 0xF
    mantissa = byte & 0x7
    normal = ((exponent + 120) << 23) | (mantissa << 20)
    # subnormals m * 2^-9: m=1 -> 2^-9; m=2,3 -> 2^-8 * (1 + (m-2)/2); m=4..7 -> 2^-7 * (1 + (m-4)/4)
    sub = jnp.where(mantissa == 1, (118 << 23),
          jnp.where(mantissa == 2, (119 << 23),
          jnp.where(mantissa == 3, (119 << 23) | (1 << 22),
          jnp.where(mantissa == 4, (120 << 23),
          jnp.where(mantissa == 5, (120 << 23) | (1 << 21),
          jnp.where(mantissa == 6, (120 << 23) | (2 << 21),
          jnp.where(mantissa == 7, (120 << 23) | (3 << 21), 0)))))))
    magnitude = jnp.where(exponent == 0, sub, normal)
    magnitude = jnp.where((exponent == 15) & (mantissa == 7), jnp.uint32(0x7FC00000), magnitude)
    return (sign | magnitude).astype(jnp.uint32)


def decode_matmul_variant(mode: str, *, tn: int = 512, tk: int = 512, interpret: bool = False):
    """One-row [1,K] x [N,K] projection kernels isolating the FP8 decode cost.

    mode: "f32convert" (frozen decode: bitcast->f32 * scale -> bf16, then dot),
          "decode_only" (same decode, no MXU; sums the decoded tile),
          "packed" (K viewed as uint32 words: 4 bytes per lane, integer
          e4m3->f32 bit formula, scale, bf16, dot against the K-permuted row),
          "bf16" (resident BF16 table, dot only).
    Returns fn(row_bf16 [1,K], table, scale) -> f32 [1,N] (or [1,N] partial sums).
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    bpn, bpk = tn // 128, tk // 128

    def scale_entry(slab, ki, j, ni, i):
        rows = lax.broadcasted_iota(jnp.int32, slab.shape, 0)
        cols = lax.broadcasted_iota(jnp.int32, slab.shape, 1)
        mask = (rows == (ki * bpk + j) % 8) & (cols == ni * bpn + i)
        return jnp.sum(jnp.where(mask, slab, 0.0), dtype=jnp.float32)

    def kernel(x_ref, w_ref, s_ref, out_ref, acc_ref):
        ni, ki = pl.program_id(0), pl.program_id(1)

        @pl.when(ki == 0)
        def _init():
            acc_ref[...] = jnp.zeros_like(acc_ref)

        slab = s_ref[...]
        for j in range(bpk):
            for i in range(bpn):
                sv = scale_entry(slab, ki, j, ni, i)
                if mode == "bf16":
                    decoded = w_ref[pl.ds(i * 128, 128), pl.ds(j * 128, 128)]
                    xb = x_ref[:, pl.ds(j * 128, 128)]
                elif mode == "packed":
                    words = w_ref[pl.ds(i * 128, 128), pl.ds(j * 32, 32)]          # (128, 32) uint32 = 128 bytes of K
                    parts = []
                    for b in range(4):
                        byte = (words >> (8 * b)) & 0xFF
                        parts.append(lax.bitcast_convert_type(_e4m3_bits_to_f32_bits(byte), jnp.float32))
                    decoded = (jnp.concatenate(parts, axis=1) * sv).astype(jnp.bfloat16)   # (128, 128), K permuted
                    xb = x_ref[:, pl.ds(j * 128, 128)]                                    # row pre-permuted outside
                else:
                    bits = w_ref[pl.ds(i * 128, 128), pl.ds(j * 128, 128)]
                    decoded = (lax.bitcast_convert_type(bits, jnp.float8_e4m3fn).astype(jnp.float32) * sv).astype(jnp.bfloat16)
                    xb = x_ref[:, pl.ds(j * 128, 128)]
                if mode == "decode_only":
                    acc_ref[:, pl.ds(i * 128, 128)] += jnp.sum(decoded.astype(jnp.float32), axis=0, keepdims=True)
                else:
                    acc_ref[:, pl.ds(i * 128, 128)] += lax.dot_general(
                        xb, decoded, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)

        @pl.when(ki == pl.num_programs(1) - 1)
        def _store():
            out_ref[...] = acc_ref[0:1, :]

    def fn(row, table, scale):
        K = row.shape[1]
        N = table.shape[0]
        x8 = jnp.zeros((8, K), jnp.bfloat16).at[0].set(row[0])
        if mode == "packed":
            # words: (N, K/4) uint32; byte b of word j is K index 4j+b; the kernel
            # emits per 128-block the order [b0: 32 words | b1 | b2 | b3], so permute x.
            words = lax.bitcast_convert_type(table.reshape(N, K // 4, 4), jnp.uint32)
            idx = jnp.arange(K).reshape(K // 128, 32, 4).transpose(0, 2, 1).reshape(K)
            x8 = x8[:, idx]
            w_spec = pl.BlockSpec((tn, tk // 4), lambda ni, ki: (ni, ki))
            operand = words
        else:
            w_spec = pl.BlockSpec((tn, tk), lambda ni, ki: (ni, ki))
            operand = table
        slab = jnp.pad(jnp.swapaxes(scale, 0, 1), ((0, ((K // 128 + 7) // 8) * 8 - K // 128), (0, 128 - N // 128)))
        call = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((1, N), jnp.float32),
            grid=(N // tn, K // tk),
            in_specs=(pl.BlockSpec((8, tk), lambda ni, ki: (0, ki)), w_spec,
                      pl.BlockSpec((8, 128), lambda ni, ki: ((ki * bpk) // 8, 0))),
            out_specs=pl.BlockSpec((1, tn), lambda ni, ki: (0, ni)),
            scratch_shapes=(pltpu.VMEM((8, tn), jnp.float32),),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
            interpret=interpret,
            name=f"glm_perf_decode_variant_{mode}",
        )
        return call(x8, operand, slab)

    return fn


def bench_tiles(mesh, report: dict, *, iters: int) -> None:
    """One-row FP8 projection [1,1536] x [2048,1536]: frozen 128x128 grid vs larger tiles."""

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax

    from glm_tpu.greenfield.kernels.pallas.fp8_matmul import Fp8BlockMatmulConfig, fp8_block_matmul_f32
    from glm_tpu.perf.fp8_routed_experts import RoutedProjectionConfig, fp8_routed_projection

    device = jax.local_devices()[0]
    key = jax.random.PRNGKey(7)
    N, K = 2048, 1536
    with jax.default_device(device):
        bits = lax.bitcast_convert_type(
            (jax.random.normal(key, (N, K), jnp.float32) * 0.02).astype(jnp.float8_e4m3fn), jnp.uint8)
        scale = jax.random.uniform(jax.random.fold_in(key, 1), (N // 128, K // 128), jnp.float32, 0.5, 1.5)
        row = (jax.random.normal(jax.random.fold_in(key, 2), (1, K), jnp.float32) * 0.1).astype(jnp.bfloat16)
    out: dict[str, Any] = {}

    def chain(body, n):
        def run(x, w, s, *rest):
            def step(_, row):
                y = body(row, w, s, *rest)
                return (row.astype(jnp.float32) + 1e-3 * y[:, :K]).astype(jnp.bfloat16)
            return lax.fori_loop(0, n, step, x)
        return jax.jit(run)

    frozen_body = lambda x, w, s: fp8_block_matmul_f32(
        x, w, s, config=Fp8BlockMatmulConfig(output_tile=128, contraction_tile=128))
    out["frozen_fp8_block_matmul_f32_128x128"] = _chain_timeit(
        lambda n: chain(frozen_body, n), (row, bits, scale), counts=(1, 33), iters=iters)
    ids = jnp.zeros((1,), jnp.int32)
    owned = jnp.ones((1,), bool)
    for tn, tk in ((128, 128), (256, 256), (512, 512), (1024, 512), (2048, 512)):
        cfg = RoutedProjectionConfig(output_tile=tn, contraction_tile=tk)
        body = lambda x, w, s, i, o, cfg=cfg: fp8_routed_projection(
            x, ((w[None], s[None]),), i, o, config=cfg, result_dtype=jnp.float32)[:, 0]
        key_name = f"routed_projection_1slot_t{tn}x{tk}"
        out[key_name] = _chain_timeit(lambda n, body=body: chain(body, n), (row, bits, scale, ids, owned),
                                      counts=(1, 33), iters=iters)
        ref = jax.jit(frozen_body)(row, bits, scale)
        got = jax.jit(body)(row, bits, scale, ids, owned)
        out[key_name]["bitwise_mismatches_vs_frozen"] = int(
            np.count_nonzero(np.asarray(ref).view(np.uint32) != np.asarray(got).view(np.uint32)))
    # Plain XLA alternatives for M=1: dequantize the selected [N,K] table in a
    # fusion and let XLA's dot consume it (no Pallas launch floor).
    def xla_dequant_dot(x, w, s):
        decoded = lax.bitcast_convert_type(w, jnp.float8_e4m3fn).astype(jnp.float32)
        expanded = (decoded.reshape(N // 128, 128, K // 128, 128)
                    * s[:, None, :, None]).reshape(N, K).astype(jnp.bfloat16)
        return lax.dot_general(x, expanded, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)

    out["xla_dequant_dot_f32"] = _chain_timeit(lambda n: chain(xla_dequant_dot, n), (row, bits, scale),
                                               counts=(1, 33), iters=iters)
    ref = jax.jit(frozen_body)(row, bits, scale)
    got = jax.jit(xla_dequant_dot)(row, bits, scale)
    out["xla_dequant_dot_f32"]["max_abs_diff_vs_frozen"] = float(jnp.max(jnp.abs(ref - got)))
    with jax.default_device(device):
        table = lax.bitcast_convert_type(
            (jax.random.normal(jax.random.fold_in(key, 3), (32, N, K), jnp.float32) * 0.02).astype(jnp.float8_e4m3fn), jnp.uint8)
        table_scale = jax.random.uniform(jax.random.fold_in(key, 4), (32, N // 128, K // 128), jnp.float32, 0.5, 1.5)
        expert = jnp.int32(13)

    def xla_selected_dequant_dot(x, w3, s3, e):
        w = lax.dynamic_index_in_dim(w3, e, 0, keepdims=False)
        s = lax.dynamic_index_in_dim(s3, e, 0, keepdims=False)
        return xla_dequant_dot(x, w, s)

    out["xla_selected_expert_dequant_dot_f32"] = _chain_timeit(
        lambda n: chain(xla_selected_dequant_dot, n), (row, table, table_scale, expert), counts=(1, 33), iters=iters)

    # No-decode reference: the same projection with a resident BF16 table (twice
    # the HBM bytes, no FP8 decode) -> isolates the v4 FP8-decode cost.
    with jax.default_device(device):
        bf16_table = (jax.random.normal(jax.random.fold_in(key, 5), (N, K), jnp.float32) * 0.02).astype(jnp.bfloat16)

    def bf16_dot(x, w, s):
        del s
        return lax.dot_general(x, w, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)

    out["bf16_weights_dot_no_decode"] = _chain_timeit(lambda n: chain(bf16_dot, n), (row, bf16_table, scale),
                                                     counts=(1, 33), iters=iters)

    # Integer bit-trick decode e4m3fn -> bf16 (exact), then the frozen f32 scale multiply.
    def xla_bittrick_dequant_dot(x, w, s):
        decoded = _e4m3_to_bf16_bits(w)
        expanded = (decoded.astype(jnp.float32).reshape(N // 128, 128, K // 128, 128)
                    * s[:, None, :, None]).reshape(N, K).astype(jnp.bfloat16)
        return lax.dot_general(x, expanded, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)

    out["xla_bittrick_dequant_dot_f32"] = _chain_timeit(lambda n: chain(xla_bittrick_dequant_dot, n),
                                                         (row, bits, scale), counts=(1, 33), iters=iters)
    got = jax.jit(xla_bittrick_dequant_dot)(row, bits, scale)
    out["xla_bittrick_dequant_dot_f32"]["max_abs_diff_vs_frozen"] = float(jnp.max(jnp.abs(ref - got)))

    # Pallas decode-cost decomposition (512x512 tiles): frozen-style decode + dot,
    # decode only, packed 4-bytes-per-lane integer decode + dot, resident BF16 dot.
    for mode in ("f32convert", "decode_only", "packed", "bf16"):
        variant = decode_matmul_variant(mode)
        table = bf16_table if mode == "bf16" else bits
        key_name = f"pallas_decode_variant_{mode}_t512x512"
        out[key_name] = _chain_timeit(lambda n, v=variant: chain(v, n), (row, table, scale), counts=(1, 33), iters=iters)
        if mode in ("f32convert", "packed"):
            got = jax.jit(variant)(row, table, scale)
            out[key_name]["max_abs_diff_vs_frozen"] = float(jnp.max(jnp.abs(ref - got)))
            out[key_name]["bitwise_mismatches_vs_frozen"] = int(
                np.count_nonzero(np.asarray(ref).view(np.uint32) != np.asarray(got).view(np.uint32)))
    for name, timing in out.items():
        print(f"tiles {name}: per-call(slope) {timing['per_body_ms_from_slope']*1e3:.1f} us; n1 p50 {timing['n1']['p50_ms']*1e3:.1f} us", flush=True)
    report["one_row_projection_tiles"] = out


def synthetic_decoder_weights(config, make, *, progress=None):
    """Frozen-spec weight pytree at real GLM geometry from ``make(shape, spec, kind)``.

    ``kind`` is one of fp8 / scale / bf16 / ones_bf16 / zeros_f32; ``make`` may
    return device arrays (benchmark) or ShapeDtypeStructs (shape checks).
    """

    from glm_tpu.greenfield.kernels.ws32_layer import (
        Ws32AttentionWeights, Ws32DenseWeights, Ws32DsaWeights, Ws32MoeWeights, Ws32QkvAWeights,
    )
    from glm_tpu.greenfield.runtime import ws32_decoder as decoder

    g = config.geometry
    H, hb = g.hidden_size, g.hidden_size // 128
    specs = decoder.ws32_decoder_weight_specs(config)
    rnd = make
    heads = g.attention_heads
    qk, rope, dv = g.qk_nope_head_dim, g.qk_rope_head_dim, g.v_head_dim
    qk_head = qk + rope
    layers = []
    for layer_id in range(g.num_layers):
        sp = specs.layers[layer_id]
        q = Ws32QkvAWeights(
            rnd((H,), sp.qkv_a[0], "ones_bf16"),
            rnd((g.q_lora_rank, H), sp.qkv_a[1], "fp8"), rnd((g.q_lora_rank // 128, hb), sp.qkv_a[2], "scale"),
            rnd((g.q_lora_rank,), sp.qkv_a[3], "ones_bf16"),
            rnd((g.kv_lora_rank + rope, H), sp.qkv_a[4], "fp8"),
            rnd(((g.kv_lora_rank + rope + 127) // 128, hb), sp.qkv_a[5], "scale"),
            rnd((g.kv_lora_rank,), sp.qkv_a[6], "ones_bf16"),
        )
        a = Ws32AttentionWeights(
            rnd((heads * qk_head, g.q_lora_rank), sp.attention[0], "fp8"),
            rnd((heads * qk_head // 128, g.q_lora_rank // 128), sp.attention[1], "scale"),
            rnd((heads * (qk + dv), g.kv_lora_rank), sp.attention[2], "fp8"),
            rnd((heads * (qk + dv) // 128, g.kv_lora_rank // 128), sp.attention[3], "scale"),
            rnd((H, heads * dv), sp.attention[4], "fp8"), rnd((hb, heads * dv // 128), sp.attention[5], "scale"),
        )
        d = None
        if sp.dsa is not None:
            ih, ihd = g.dsa_indexer_heads, g.dsa_indexer_head_dim
            d = Ws32DsaWeights(
                rnd((ih * ihd, g.q_lora_rank), sp.dsa[0], "fp8"),
                rnd((ih * ihd // 128, g.q_lora_rank // 128), sp.dsa[1], "scale"),
                rnd((ihd, H), sp.dsa[2], "fp8"), rnd((ihd // 128, hb), sp.dsa[3], "scale"),
                rnd((ihd,), sp.dsa[4], "ones_bf16"), rnd((ihd,), sp.dsa[5], "bf16"),
                rnd((ih, H), sp.dsa[6], "bf16"),
            )
        dense = moe = None
        if sp.dense is not None:
            di = g.dense_intermediate_size
            dense = Ws32DenseWeights(
                rnd((di, H), sp.dense[0], "fp8"), rnd((di // 128, hb), sp.dense[1], "scale"),
                rnd((di, H), sp.dense[2], "fp8"), rnd((di // 128, hb), sp.dense[3], "scale"),
                rnd((H, di), sp.dense[4], "fp8"), rnd((hb, di // 128), sp.dense[5], "scale"),
            )
        if sp.moe is not None:
            E, I = g.num_routed_experts, g.moe_intermediate_size
            moe = Ws32MoeWeights(
                rnd((E, H), sp.moe[0], "bf16"), rnd((E,), sp.moe[1], "zeros_f32"),
                rnd((E, I, H), sp.moe[2], "fp8"), rnd((E, I // 128, hb), sp.moe[3], "scale"),
                rnd((E, I, H), sp.moe[4], "fp8"), rnd((E, I // 128, hb), sp.moe[5], "scale"),
                rnd((E, H, I), sp.moe[6], "fp8"), rnd((E, hb, I // 128), sp.moe[7], "scale"),
                rnd((I, H), sp.moe[8], "fp8"), rnd((I // 128, hb), sp.moe[9], "scale"),
                rnd((I, H), sp.moe[10], "fp8"), rnd((I // 128, hb), sp.moe[11], "scale"),
                rnd((H, I), sp.moe[12], "fp8"), rnd((hb, I // 128), sp.moe[13], "scale"),
            )
        layers.append(decoder.Ws32LayerWeights(
            q, a, d, rnd((H,), sp.post_attention_norm_weight_local, "ones_bf16"), dense, moe))
        if progress is not None and layer_id % 10 == 0:
            progress(f"step: synthetic weights for layer {layer_id} placed", flush=True)
    return decoder.Ws32DecoderWeights(
        rnd((g.vocab_size, H), specs.embedding_local, "bf16"), tuple(layers),
        rnd((H,), specs.final_norm_weight_local, "ones_bf16"), rnd((g.vocab_size, H), specs.lm_head_local, "bf16"),
    )


def bench_attention(mesh, report: dict, *, iters: int, save=None, global_tiles=False) -> None:
    """D5 at real head/cache geometry; compare identical inputs before timing."""
    from dataclasses import replace
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from glm_tpu.greenfield.kernels.reference.attention import MlaNumericalContract, StageLocalKvLayout, gather_stage_local_selected_kv_aligned
    from glm_tpu.greenfield.kernels.reference.dsa import SelectedPositions
    from glm_tpu.greenfield.kernels.pallas.sparse_attention import SparseMlaConfig, pregathered_sparse_mla_pallas
    from glm_tpu.perf.lse_attention import lse_attention_mapped
    from glm_tpu.perf.global_tile_attention import global_tile_attention_mapped

    contract = MlaNumericalContract()
    layout = StageLocalKvLayout(local_parallel_size=8)
    config = SparseMlaConfig(segment_block=512 if global_tiles else 128)
    q = _sharded_random(mesh, (1, 64, 512), P(None, "expert"), "bf16", 711)
    r = _sharded_random(mesh, (1, 64, 64), P(None, "expert"), "bf16", 712)
    cache = _sharded_random(mesh, (8, 256, 64, 640), P("expert"), "bf16", 713)
    tables = jax.device_put(np.arange(256, dtype=np.int32)[None], NamedSharding(mesh, P()))
    length = jax.device_put(np.array([131072], np.int32), NamedSharding(mesh, P()))
    specs = (P(None, "expert"), P(None, "expert"), P("expert"), P(), P(), P(), P())

    def program(mode, n):
        def body(q, r, cache, tables, length, positions, count):
            def step(_, query):
                selected = SelectedPositions(positions, count)
                if mode == 'global_tiles':
                    return global_tile_attention_mapped(query,r,cache[0],tables,selected,length,
                        contract=contract,layout=layout,config=config).output
                if mode in ("lse", "lse_gathered"):
                    return lse_attention_mapped(query, r, cache[0], tables, selected, length,
                        contract=contract, layout=layout, config=config, gathered=(mode == "lse_gathered")).output
                aligned = gather_stage_local_selected_kv_aligned(cache[0], tables, selected, length,
                    layout=layout, owner_index=lax.axis_index("expert"))
                selected_cache = lax.psum(aligned.values, "expert")
                return pregathered_sparse_mla_pallas(query, r, selected_cache, aligned.valid_counts,
                    contract=replace(contract, num_heads=8), config=config)
            return lax.fori_loop(0, n, step, q)
        return jax.jit(jax.shard_map(body, mesh=mesh, in_specs=specs, out_specs=P(None, "expert"), check_vma=False))

    out = {}
    for pattern, positions in {
        "prefix64": np.arange(64),
        "balanced2048": np.arange(2048),
        "concentrated2048": (np.arange(2048)//64)*512 + np.arange(2048)%64,
    }.items():
        count = len(positions)
        positions = np.pad(positions.astype(np.int32), (0, 2048-count), constant_values=-1)[None]
        args = (q, r, cache, tables, length,
                jax.device_put(positions, NamedSharding(mesh, P())),
                jax.device_put(np.array([count], np.int32), NamedSharding(mesh, P())))
        ref = program("frozen", 1)(*args)
        actual = program("global_tiles" if global_tiles else "lse", 1)(*args)
        # Local addressable shards suffice; every rank records its own comparison.
        errors = [float(np.max(np.abs(np.asarray(a.data).astype(np.float32)-np.asarray(b.data).astype(np.float32))))
                  for a, b in zip(ref.addressable_shards, actual.addressable_shards)]
        out[pattern] = dict(max_abs=max(errors),segment_block=config.segment_block)
        for mode in (("frozen", "lse_gathered", "global_tiles") if global_tiles else ("frozen", "lse", "lse_gathered")):
            candidate = program(mode, 1)(*args)
            mode_errors = [float(np.max(np.abs(np.asarray(a.data).astype(np.float32)-np.asarray(b.data).astype(np.float32))))
                           for a, b in zip(ref.addressable_shards, candidate.addressable_shards)]
            timing = _chain_timeit(lambda n: program(mode, n), args, iters=iters)
            timing["max_abs_vs_frozen"] = max(mode_errors)
            timing['finite']=all(np.isfinite(np.asarray(s.data)).all() for s in candidate.addressable_shards)
            timing['bitwise_equal_to_frozen']=all(np.array_equal(
                np.ascontiguousarray(a.data).view(np.uint8),np.ascontiguousarray(b.data).view(np.uint8))
                for a,b in zip(ref.addressable_shards,candidate.addressable_shards))
            out[pattern][mode] = timing
            print(f"attention {pattern} {mode}: {timing['per_body_ms_from_slope']:.4f} ms; max_abs={max(errors)}", flush=True)
            report["attention_global_tiles" if global_tiles else "attention_exchange"] = out
            if save is not None:
                save()


def bench_dsa(mesh, report: dict, *, iters: int, save=None) -> None:
    """Physical-page scoring and exact shortlist merge at decode geometries."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from glm_tpu.greenfield.kernels.reference.attention import StageLocalKvLayout
    from glm_tpu.greenfield.kernels.reference.dsa import dsa_scores, local_topk_candidates, merge_topk_candidates_with_scores, ScoredSelectedPositions
    from glm_tpu.perf.dsa_candidates import score_cache_pages, two_stage_topk_mapped
    layout = StageLocalKvLayout(local_parallel_size=8)
    q = _sharded_random(mesh, (1, 32, 128), P(), "bf16", 881).astype(jnp.float32)
    hw = _sharded_random(mesh, (1, 32), P(), "bf16", 882).astype(jnp.float32)
    out = {}
    for capacity in (8192, 131072):
        pages = capacity//512
        cache = _sharded_random(mesh, (8, pages, 64, 128), P("expert"), "bf16", 883)
        tables = jax.device_put(np.arange(pages-1, -1, -1, dtype=np.int32)[None], NamedSharding(mesh, P()))
        length = jax.device_put(np.array([capacity], np.int32), NamedSharding(mesh, P()))
        def program(mode):
            def body(query, cache, head_weights, tables, length):
                owner = lax.axis_index("expert")
                if mode == "challenger":
                    scores, positions = score_cache_pages(query, cache[0], head_weights, tables, layout=layout, owner=owner)
                    return two_stage_topk_mapped(scores, positions, length, top_k=2048, global_context_size=capacity, positions_in_order=True)
                keys = jnp.take(cache[0], tables[0], axis=0).reshape(-1, 128)
                positions = (jnp.arange(pages)[:,None]*512 + owner*64 + jnp.arange(64)[None]).reshape(-1)
                scores = dsa_scores(query, keys, head_weights, precision="highest")
                v, p = local_topk_candidates(scores, positions, length, top_k=2048)
                result = merge_topk_candidates_with_scores(lax.all_gather(v,"expert"), lax.all_gather(p,"expert"), length, top_k=2048, global_context_size=capacity)
                return result, jnp.bool_(False)
            return jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(), P("expert"), P(), P(), P()), out_specs=(ScoredSelectedPositions(P(),P(),P()), P()), check_vma=False))
        args = (q, cache, hw, tables, length)
        reference = program("frozen")(*args)[0]
        out[str(capacity)] = {}
        for mode in ("frozen", "challenger"):
            fn = program(mode)
            compiled = fn.lower(*args).compile()
            result, fallback = compiled(*args)
            equal = all(np.array_equal(np.asarray(a.addressable_shards[0].data), np.asarray(b.addressable_shards[0].data)) for a,b in zip(reference,result))
            if not equal:
                raise AssertionError("DSA selection or score differs from frozen")
            timing = _timeit(compiled, args, warmup=10, iters=iters)
            timing.update(bitwise_equal=equal, fallback=bool(np.asarray(fallback)))
            out[str(capacity)][mode] = timing
            print(f"DSA {capacity} {mode}: {timing['p50_ms']:.4f} ms, fallback={timing['fallback']}", flush=True)
            report["dsa_selection"] = out
            if save is not None:
                save()


def bench_compact_feature_reduce(mesh,report:dict,*,iters:int,save=None)->None:
    """Real B512 routed gate/up buffer exchange, including compact restoration."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding,PartitionSpec as P
    from glm_tpu.perf.compact_feature_reduce import owner_compact_feature_sum
    from glm_tpu.perf.state_comparison import compare_addressable_state
    rows,width=4096,2048
    raw=_sharded_random(mesh,(8,4,2,rows,width),P('expert','feature'),'bf16',614).astype(jnp.float32)
    results={}
    report['compact_feature_reduce']=dict(scope='B512 gate/up feature sum only, including packing/restoration; no FP8 projection, activation or expert reduction; skewed fleet maximum matters',cases=results)
    for pattern,counts in (('balanced',[512]*8),('concentrated',[4096,0,0,0,0,0,0,0]),
                           ('mixed',[128,256,512,1024,1152,0,0,1024])):
        counts=np.asarray(counts,np.int32);assert int(counts.sum())==rows
        starts=np.cumsum(counts,dtype=np.int32)-counts
        put=lambda x:jax.device_put(x,NamedSharding(mesh,P('expert')))
        starts,counts=put(starts),put(counts)
        live=(jnp.arange(rows)[None]>=starts[:,None])&(jnp.arange(rows)[None]<starts[:,None]+counts[:,None])
        values=jnp.where(live[:,None,None,:,None],raw,jnp.float32(0))
        args=(values,starts,counts);jax.block_until_ready(args)
        baseline=None
        for label,compact in (('full4096',False),('bounded1024',True)):
            def body(v,s,n):
                result=(owner_compact_feature_sum(v[0,0],s[0],n[0],capacity=1024) if compact
                        else lax.psum(v[0,0],'feature').astype(jnp.bfloat16))
                return result[None]
            program=jax.jit(jax.shard_map(body,mesh=mesh,in_specs=(P('expert','feature'),P('expert'),P('expert')),
                                         out_specs=P('expert'),check_vma=False))
            executable=program.lower(*args).compile()
            output=executable(*args);jax.block_until_ready(output)
            if baseline is None:baseline=output
            agreement=compare_addressable_state(output,baseline)
            equal=bool(np.asarray(multihost_utils.process_allgather(np.bool_(agreement['bitwise_equal']))).all())
            if not equal:raise AssertionError('compact feature reduction changed result bits')
            row=_timeit(executable,args,warmup=5,iters=iters)
            row.update(bitwise_equal=equal,temp_bytes=int(executable.memory_analysis().temp_size_in_bytes))
            results.setdefault(pattern,{})[label]=row
            if save is not None:save()
            print(f'compact feature sum {pattern}/{label}: {row["p50_ms"]:.3f}ms',flush=True)


def bench_dsa_payload_merge(mesh, report: dict, *, iters: int, save=None) -> None:
    """Already-exchanged M32 DSA unions: top-k/gather versus payload sorting."""
    import jax
    import numpy as np
    from jax.sharding import NamedSharding,PartitionSpec as P
    from glm_tpu.greenfield.kernels.reference.dsa import merge_topk_candidates_with_scores,ScoredSelectedPositions
    from glm_tpu.perf.dsa_payload_merge import sort_payload_candidate_merge
    from glm_tpu.perf.state_comparison import compare_addressable_state
    replicated=NamedSharding(mesh,P())
    def put(value):return jax.device_put(value,replicated)
    results={}
    report['dsa_payload_merge']=dict(scope='isolated M32 candidate merge after exchange; no score production or all-gather timing',cases=results)
    rng=np.random.default_rng(615)
    for width in (512,2048):
        for pattern in ('random','tied','signed_zero'):
            scores=rng.normal(size=(8,32,width)).astype(np.float32)
            if pattern=='tied':scores=np.rint(scores)
            if pattern=='signed_zero':
                scores.fill(-0.0);scores.ravel()[::3]=0.0
            positions=np.stack([rng.permutation(8*width) for _ in range(32)])
            positions=positions.reshape(32,8,width).transpose(1,0,2).astype(np.int32)
            lengths=np.full(32,8*width,np.int32);lengths[0]=0;lengths[1]=17
            live=positions<lengths[None,:,None]
            scores=np.where(live,scores,-np.inf);positions=np.where(live,positions,-1)
            args=tuple(map(put,(scores,positions,lengths)))
            baseline=None
            for label,merge in (('topk_gather',merge_topk_candidates_with_scores),('payload_sort',sort_payload_candidate_merge)):
                def body(s,p,n):return merge(s,p,n,top_k=2048,global_context_size=8*width,paired_position_sort=True)
                program=jax.jit(jax.shard_map(body,mesh=mesh,in_specs=(P(),P(),P()),
                    out_specs=ScoredSelectedPositions(P(),P(),P()),check_vma=False))
                executable=program.lower(*args).compile()
                output=executable(*args);jax.block_until_ready(output)
                if baseline is None:baseline=output
                agreement=compare_addressable_state(output,baseline)
                row=_timeit(executable,args,warmup=5,iters=iters)
                row.update(bitwise_equal=agreement['bitwise_equal'],temp_bytes=int(executable.memory_analysis().temp_size_in_bytes))
                results.setdefault(str(width)+'_'+pattern,{})[label]=row
                if save is not None:save()
                if not agreement['bitwise_equal']:raise AssertionError('DSA payload merge changed score/position bits')
                print(f'DSA payload merge {width}/{pattern}/{label}: {row["p50_ms"]:.3f}ms',flush=True)


def bench_prefill_primitives(mesh, report: dict, *, iters: int, save=None) -> None:
    """P1/P2 32-row tiles at 128K, including result comparisons and health."""
    from dataclasses import replace
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from glm_tpu.greenfield.kernels.reference.attention import MlaNumericalContract, StageLocalKvLayout, gather_stage_local_selected_kv_aligned
    from glm_tpu.greenfield.kernels.reference.dsa import SelectedPositions, ScoredSelectedPositions
    from glm_tpu.greenfield.kernels.pallas.sparse_attention import SparseMlaConfig, pregathered_sparse_mla_pallas
    from glm_tpu.greenfield.kernels.prefill_dsa import ws32_prefill_dsa_from_query_mapped
    from glm_tpu.perf.lse_attention import lse_attention_mapped
    from glm_tpu.perf.dsa_candidates import prefill_dsa_one_pass_mapped
    rows, capacity = 32, 131072
    replicated = NamedSharding(mesh, P())
    length = jax.device_put(np.arange(capacity-rows+1, capacity+1, dtype=np.int32), replicated)
    query = _sharded_random(mesh, (rows,32,128), P(), "bf16", 901).astype(jnp.float32)
    heads = _sharded_random(mesh, (rows,32), P(), "bf16", 902).astype(jnp.float32)
    keys = _sharded_random(mesh, (8,capacity//8,128), P("expert"), "bf16", 903)
    def dsa_program(mode):
        def body(q,k,w,length):
            positions=(jnp.arange(capacity//512)[:,None]*512+lax.axis_index("expert")*64+jnp.arange(64)[None]).reshape(-1)
            fn = prefill_dsa_one_pass_mapped if mode == "one_pass" else ws32_prefill_dsa_from_query_mapped
            kw = {} if mode == "one_pass" else dict(key_tile=512,
                paired_position_sort=(mode == "admitted"), sorted_local_merge=(mode == "admitted"))
            selected, health = fn(q,k[0],w,positions,length,global_context_size=capacity,precision="default",**kw)
            return selected, lax.pmin(health.astype(jnp.int32), ("expert", "feature")).astype(jnp.bool_)
        return jax.jit(jax.shard_map(body,mesh=mesh,in_specs=(P(),P("expert"),P(),P()),out_specs=(ScoredSelectedPositions(P(),P(),P()),P()),check_vma=False))
    args=(query,keys,heads,length)
    ref = dsa_program("admitted")(*args)
    out = {}
    for mode in ("default", "admitted", "one_pass"):
        fn = dsa_program(mode).lower(*args).compile()
        result = fn(*args)
        equal = all(np.array_equal(np.asarray(a.addressable_shards[0].data),np.asarray(b.addressable_shards[0].data)) for a,b in zip(jax.tree.leaves(ref),jax.tree.leaves(result)))
        if not equal or not bool(np.asarray(result[1])):
            raise AssertionError("prefill DSA comparison/health failed")
        name = {"default": "dsa_default_tiles512", "admitted": "dsa_admitted_paired_sorted_tiles512", "one_pass": "dsa_one_pass"}[mode]
        out[name] = _timeit(fn,args,warmup=5,iters=iters)
        out[name].update(bitwise_equal=equal,contract_valid=True)
        print(f"prefill {name}: {out[name]['p50_ms']:.3f} ms",flush=True)
        report["prefill_primitives_m32_128k"] = out
        if save is not None: save()
    selected = ref[0]
    contract, layout = MlaNumericalContract(), StageLocalKvLayout(local_parallel_size=8)
    config = SparseMlaConfig(segment_block=512)
    q = _sharded_random(mesh,(rows,64,512),P(None,"expert"),"bf16",904)
    r = _sharded_random(mesh,(rows,64,64),P(None,"expert"),"bf16",905)
    cache = _sharded_random(mesh,(8,256,64,640),P("expert"),"bf16",906)
    tables = jax.device_put(np.broadcast_to(np.arange(256,dtype=np.int32),(rows,256)),replicated)
    def attention_program(challenger):
        def body(q,r,c,t,p,n,length):
            selected=SelectedPositions(p,n)
            if challenger:
                result=lse_attention_mapped(q,r,c[0],t,selected,length,contract=contract,layout=layout,config=config)
                return result.output,result.contract_valid
            aligned=gather_stage_local_selected_kv_aligned(c[0],t,selected,length,layout=layout,owner_index=lax.axis_index("expert"))
            full=lax.psum(aligned.values,"expert")
            result=pregathered_sparse_mla_pallas(q,r,full,aligned.valid_counts,contract=replace(contract,num_heads=8),config=config,prefill=True)
            return result,lax.pmin(aligned.contract_valid.astype(jnp.int32),"expert").astype(jnp.bool_)
        return jax.jit(jax.shard_map(body,mesh=mesh,in_specs=(P(None,"expert"),P(None,"expert"),P("expert"),P(),P(),P(),P()),out_specs=(P(None,"expert"),P()),check_vma=False))
    args=(q,r,cache,tables,selected.positions,selected.valid_counts,length)
    ref=attention_program(False)(*args)
    for challenger in (False,True):
        fn=attention_program(challenger).lower(*args).compile()
        result=fn(*args)
        error=max(float(np.max(np.abs(np.asarray(a.data).astype(np.float32)-np.asarray(b.data).astype(np.float32)))) for a,b in zip(ref[0].addressable_shards,result[0].addressable_shards))
        if not bool(np.asarray(result[1]).all()) or error > .015625:
            raise AssertionError("prefill attention comparison/health failed")
        name="attention_lse" if challenger else "attention_frozen"
        out[name]=_timeit(fn,args,warmup=5,iters=iters)
        out[name].update(max_abs_vs_frozen=error,contract_valid=True)
        print(f"prefill {name}: {out[name]['p50_ms']:.3f} ms",flush=True)
        report["prefill_primitives_m32_128k"] = out
        if save is not None: save()


def bench_fused_projections(mesh, report: dict, *, iters: int, save=None, artifact_dir: Path | None = None) -> None:
    """Diagnose fused feature sums at exact layer geometry, including NaN checks."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax
    from jax.sharding import PartitionSpec as P
    from glm_tpu.perf.bf16_resident import _dot_f32, _head_weight_partial, Bf16DsaWeights
    specs = (P(None,'feature'),P(None,'feature'),P(None,'feature'),P(None,'feature'),P('expert','feature'))
    shapes = ((1,6144),(2048,6144),(576,6144),(128,6144),(32,6144))
    args = tuple(_sharded_random(mesh,shape,spec,'bf16',1500+i) for i,(shape,spec) in enumerate(zip(shapes,specs)))
    def program(mode):
        def body(x,q,kv,wk,head):
            values = [_dot_f32(x,q),_dot_f32(x,kv),_dot_f32(x,wk),
                      _head_weight_partial(x,Bf16DsaWeights(None,wk,None,None,head))]
            if mode == 'separate': return tuple(lax.psum(v,'feature') for v in values)
            count = 2 if mode == 'qkv_only' else 4
            chunks = values[:count]
            if mode == 'padded128':
                chunks = [jnp.pad(v,((0,0),(0,(-v.shape[-1])%128))) for v in chunks]
            combined = lax.psum(jnp.concatenate(chunks,axis=-1),'feature')
            start = 0
            result = []
            for chunk,value in zip(chunks,values):
                result.append(combined[:,start:start+value.shape[-1]])
                start += chunk.shape[-1]
            result.extend(lax.psum(v,'feature') for v in values[count:])
            return tuple(result)
        return jax.jit(jax.shard_map(body,mesh=mesh,in_specs=specs,
            out_specs=(P(),P(),P(),P(None,'expert')),check_vma=False))
    ref = program('separate')(*args)
    result = {}
    report['fused_projection_diagnostic'] = result
    for mode in ('separate','fused','padded128','qkv_only'):
        executable = program(mode).lower(*args).compile()
        out = executable(*args)
        checks = []
        for name,reference,actual in zip(('q_a','kv_a','wk','head'),ref,out):
            arrays = [(np.asarray(a.data),np.asarray(b.data)) for a,b in zip(reference.addressable_shards,actual.addressable_shards)]
            finite = all(np.isfinite(b).all() for a,b in arrays)
            checks.append(dict(name=name,finite=finite,bitwise_equal=all(np.array_equal(a.view(np.uint32),b.view(np.uint32)) for a,b in arrays),
                               max_abs=max(float(np.max(np.abs(a-b))) for a,b in arrays) if finite else None,
                               nonfinite=sum(int((~np.isfinite(b)).sum()) for a,b in arrays)))
        row = _timeit(executable,args,warmup=5,iters=iters)
        row['outputs'] = checks
        if artifact_dir is not None:
            import hashlib
            hlo = executable.as_text().encode()
            artifact_dir.mkdir(parents=True,exist_ok=True)
            (artifact_dir / f'{mode}.rank{jax.process_index()}.hlo.txt').write_bytes(hlo)
            row['optimized_hlo_sha256'] = hashlib.sha256(hlo).hexdigest()
        result[mode] = row
        print(f'fused projection {mode}: {checks}',flush=True)
        if save is not None: save()


def bench_prefill_model(mesh, report: dict, *, prompt_length: int, capacity: int,
                        variants: set[str], save=None, pending_cache_rows: bool = False,
                        trace_dir: Path | None = None, owned_key_capacity: int | None = None,
                        block_rows: int = 128, wide_indexshare: bool = False,
                        sparse_slice: bool = False) -> None:
    """Complete synthetic prompt, including all 78 layers and repaired-key commits.

    Canonical B128 dense placement and admitted paired/sorted options are used
    for every variant. One warm first block is excluded; the complete timed
    prompt starts afresh. This is no trained-weight or token-quality admission.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding, PartitionSpec as P
    from glm_tpu.greenfield.runtime import ws32_decoder as decoder, ws32_batched_prefill as prefill
    from glm_tpu.greenfield.types import ModelGeometry
    from glm_tpu.greenfield.kernels.reference.prefill_index import (
        decode_stage_local_prefill_index_wk_bf16, promote_stage_local_prefill_index_wk,
    )
    from glm_tpu.perf.bf16_resident import bf16_resident_weights
    from glm_tpu.perf.prefill_challenger import build_ws32_prefill_challenger_program
    if not variants or not variants <= {'frozen', 'p1p2', 'p1p2_bf16'}:
        raise ValueError('unknown complete prefill variants')
    if owned_key_capacity is not None and variants != {'p1p2_bf16'}:
        raise ValueError('bounded full-model benchmark requires the sole p1p2_bf16 variant')
    if wide_indexshare and (variants != {'p1p2_bf16'} or owned_key_capacity is None or pending_cache_rows):
        raise ValueError('wide IndexShare benchmark requires bounded resident full-cache prefill')
    if block_rows not in (128,512,1024) or (block_rows > 128 and
            (variants != {'p1p2_bf16'} or pending_cache_rows)):
        raise ValueError('pooled full-model benchmark requires resident-only full-cache B512/B1024')
    if not 0 < prompt_length < capacity or prompt_length % block_rows:
        raise ValueError('complete prefill benchmark requires block-aligned prompt below capacity')
    geometry = ModelGeometry.from_hf_config(json.loads((REPO / 'configs/glm-5.2-fp8-config.json').read_text()))
    layer_indices = tuple(range(geometry.num_layers))
    if sparse_slice:
        from dataclasses import replace
        layer_indices = layer_indices[-8:]
        if (len(layer_indices)!=8 or geometry.indexer_types[layer_indices[0]]!='full'
                or any(geometry.mlp_layer_types[i]!='sparse' for i in layer_indices)):
            raise ValueError('registered final eight-layer sparse pattern drifted')
        geometry = replace(geometry,num_layers=8,first_dense_layers=0,
            mlp_layer_types=tuple(geometry.mlp_layer_types[i] for i in layer_indices),
            indexer_types=tuple(geometry.indexer_types[i] for i in layer_indices))
    config = decoder.Ws32DecoderConfig(geometry, capacity, host_main_rope_table=True)
    seed = [0]
    def rnd(shape, spec, kind):
        seed[0] += 1
        return _sharded_random(mesh, shape, spec, kind, seed[0])
    raw = synthetic_decoder_weights(config, rnd, progress=print)
    jax.block_until_ready(raw)
    resident = bf16_resident_weights(mesh, config, raw) if 'p1p2_bf16' in variants else None
    jax.block_until_ready(resident)
    # Same two completed executables as the admitted repair-WK preparation.
    def decode_wk(bits, scales):
        return decode_stage_local_prefill_index_wk_bf16(
            lax.all_gather(bits,'feature',axis=1,tiled=True),
            lax.all_gather(scales,'feature',axis=1,tiled=True),contract=config.dsa_contract)
    wk_decode = jax.jit(jax.shard_map(decode_wk,mesh=mesh,
        in_specs=(P(None,'feature'),P(None,'feature')),out_specs=P(),check_vma=False))
    wk_promote = jax.jit(jax.shard_map(
        lambda x: promote_stage_local_prefill_index_wk(x,contract=config.dsa_contract),
        mesh=mesh,in_specs=(P(),),out_specs=P(),check_vma=False))
    wk = []
    for layer_id in config.full_index_slots:
        d = raw.layers[layer_id].dsa
        completed = wk_decode(d.wk_bits_local,d.wk_scale_local)
        jax.block_until_ready(completed)
        wk.append(wk_promote(completed))
        jax.block_until_ready(wk[-1])
    wk = tuple(wk)
    if variants == {'p1p2_bf16'}:
        # Routed arrays are shared aliases in resident; release only the unused
        # raw non-routed roots after the original repair-WK programs complete.
        raw = None
        del d, completed
    replicated = NamedSharding(mesh,P())
    def put(value): return jax.device_put(value,replicated)
    rope = put(np.asarray(decoder.build_ws32_main_rope_table(config)))
    prompt = np.random.default_rng(319).integers(0,geometry.vocab_size,prompt_length,dtype=np.int32)
    blocks = [put(prompt[i:i+block_rows]) for i in range(0,prompt_length,block_rows)]
    count = put(np.int32(block_rows))
    initial = prefill.make_ws32_batched_prefill_state(mesh,config,prompt_length=prompt_length)
    jax.block_until_ready((initial,rope,blocks))
    options = dict(block_rows=block_rows,key_tile=512,mlp_window=True,rolled_prefix=True,
                   expert_panels=True,paired_position_sort=True,sorted_local_merge=True,canonical_dense=True,
                   pending_cache_rows=pending_cache_rows,flat_pending_rows=pending_cache_rows,
                   capture_barrier=pending_cache_rows)
    results = {}
    report['complete_prefill_synthetic'] = results
    cases = [(name,False) for name in ('frozen','p1p2','p1p2_bf16') if name in variants]
    if wide_indexshare:
        cases.append(('p1p2_bf16_wide',True))
    narrow_reference = None
    for name, use_wide in cases:
        use_resident = name in ('p1p2_bf16','p1p2_bf16_wide')
        weights = resident if use_resident else raw
        program = (prefill.build_ws32_batched_prefill_program(mesh,config,**options) if name == 'frozen'
                   else build_ws32_prefill_challenger_program(mesh,config,lse_attention=True,
                        bf16_resident=use_resident,owned_key_capacity=owned_key_capacity,
                        pooled_moe=(block_rows>128),wide_indexshare=use_wide,**options))
        started = time.perf_counter()
        executable = program.execute.lower(blocks[0],count,initial,weights,wk,rope).compile()
        mem = executable.memory_analysis()
        row = dict(prompt_length=prompt_length,capacity=capacity,layers=geometry.num_layers,
                   options=dict(options,owned_key_capacity=owned_key_capacity,pooled_moe=(block_rows>128),wide_indexshare=use_wide),compile_seconds=time.perf_counter()-started,
                   memory={k:int(getattr(mem,k,0)) for k in
                     ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes',
                      'alias_size_in_bytes','generated_code_size_in_bytes')})
        row['model_scope'] = 'synthetic final-eight-layer pattern; not full-model throughput' if sparse_slice else 'complete synthetic model'
        row['layer_indices'] = layer_indices
        import hashlib
        hlo = executable.as_text().encode()
        row['optimized_hlo'] = dict(sha256=hashlib.sha256(hlo).hexdigest(),bytes=len(hlo))
        if block_rows > 128 or wide_indexshare:
            from glm_tpu.perf.real_validation import inspect_research_hlo
            hashes=np.asarray(multihost_utils.process_allgather(
                np.frombuffer(hashlib.sha256(hlo).digest(),dtype=np.uint8)))
            if not bool((hashes==hashes[0]).all()):
                raise RuntimeError('pooled prefill compiled graphs differ across hosts')
            row['hlo_admission']=inspect_research_hlo(hlo.decode())
        del hlo
        results[name] = row
        stats = [dict(device_id=d.id,**d.memory_stats()) for d in jax.local_devices()]
        row['memory_before'] = stats
        extra = (mem.output_size_in_bytes + mem.temp_size_in_bytes - mem.alias_size_in_bytes
                 + mem.generated_code_size_in_bytes)
        fit = all(s['bytes_in_use'] + extra + 512*1024**2 < s['bytes_limit'] for s in stats)
        row['memory_admitted'] = bool(np.asarray(multihost_utils.process_allgather(np.bool_(fit))).all())
        if save is not None: save()
        if not row['memory_admitted']:
            raise RuntimeError('complete prefill memory projection refused; receipt saved')
        print(f'prefill model {name}: compiled in {row["compile_seconds"]:.1f}s; memory admitted',flush=True)
        warm = executable(blocks[0],count,initial,weights,wk,rope)
        jax.block_until_ready(warm)
        if not bool(np.asarray(warm.state.decoder.contract_valid).all()):
            raise RuntimeError('complete prefill warm block unhealthy')
        del warm
        if trace_dir is not None:
            # All hosts trace the same two warm first-block executions. These
            # have an empty prefix at the registered capacity; they are not
            # full-context attention timings or part of prompt wall time.
            host_dir = trace_dir / name / socket.gethostname()
            host_dir.mkdir(parents=True, exist_ok=True)
            profile_options = jax.profiler.ProfileOptions()
            profile_options.enable_hlo_proto = False
            with jax.profiler.trace(str(host_dir), create_perfetto_link=False,
                                    profiler_options=profile_options):
                multihost_utils.sync_global_devices('prefill_trace_ready_'+name)
                for _ in range(2):
                    traced = executable(blocks[0], count, initial, weights, wk, rope)
                    jax.block_until_ready(traced)
                    del traced
            row['trace_scope'] = 'two warm first blocks from empty prefix, excluded from timing'
            row['trace_options'] = dict(enable_hlo_proto=False)
        # Profiler teardown can differ by tens of seconds across hosts. A
        # finished host must not time that wait inside its first model block.
        multihost_utils.sync_global_devices('prefill_timing_ready_'+name)
        row['fleet_synchronized_before_timing'] = True
        current = initial
        if len(cases) == 1:
            # Do not retain a third full-context cache while alternating input
            # and output state buffers during a long prompt.
            initial = None
        times = []
        wall = time.perf_counter()
        for index, block in enumerate(blocks):
            start = time.perf_counter()
            output = executable(block,count,current,weights,wk,rope)
            jax.block_until_ready(output)
            times.append(time.perf_counter()-start)
            current = output.state
            if not bool(np.asarray(current.decoder.contract_valid).all()):
                raise RuntimeError(f'complete prefill unhealthy at block {index}')
            if (index + 1) * block_rows % 16384 == 0:
                row['progress'] = dict(completed_tokens=(index+1)*block_rows,elapsed_seconds=time.perf_counter()-wall)
                if save is not None: save()
                print(f'prefill model {name}: {(index+1)*block_rows}/{prompt_length} tokens completed',flush=True)
        elapsed = time.perf_counter()-wall
        if not bool(np.asarray(current.finished)) or int(np.asarray(current.decoder.position)[0]) != prompt_length:
            raise RuntimeError('complete prefill did not reach registered frontier')
        row.update(wall_seconds=elapsed,prompt_tokens_per_second=prompt_length/elapsed,
                   block_seconds=times,first_token=int(np.asarray(output.next_token)[0]),
                   contract_valid=True,finished=True,
                   memory_after=[dict(device_id=d.id,**d.memory_stats()) for d in jax.local_devices()])
        print(f'prefill model {name}: {elapsed:.3f}s, {prompt_length/elapsed:.2f} prompt tok/s',flush=True)
        if wide_indexshare:
            if not use_wide:
                narrow_reference = output
            else:
                from glm_tpu.perf.state_comparison import compare_addressable_state
                report['prefill_wide_state_comparison'] = compare_addressable_state(output,narrow_reference)
                narrow_reference = None
        if save is not None: save()
        del executable, output, current, program
        jax.clear_caches()


def bench_request_loop(mesh, report: dict, *, iters: int, capacity: int, save=None) -> None:
    """Paired host-loop wall timing, same sampled model and deterministic draws."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding, PartitionSpec as P
    from glm_tpu.greenfield.runtime import ws32_decoder as decoder, ws32_batched_prefill as prefill
    from glm_tpu.greenfield.runtime.ws32_request_session import RequestPolicy, Ws32RequestSession
    from glm_tpu.greenfield.types import ModelGeometry
    from glm_tpu.greenfield.kernels.ws32_sampling import NucleusConfig
    from glm_tpu.perf.bf16_resident import bf16_resident_weights
    from glm_tpu.perf.fp8_routed_experts import RoutedProjectionConfig
    from glm_tpu.perf.ws32_decoder_challenger import Ws32PerfOptions, build_ws32_challenger_decoder_program
    from glm_tpu.perf.request_loop import build_packed_decoder_program, PackedRequestSession, make_request_uniform_bank
    if iters < 1 or 64+iters+6 > capacity:
        raise ValueError('request loop benchmark must fit registered token cap')
    config = decoder.Ws32DecoderConfig(ModelGeometry.from_hf_config(
        json.loads((REPO/'configs/glm-5.2-fp8-config.json').read_text())),capacity,host_main_rope_table=True)
    seed = [0]
    def rnd(shape,spec,kind):
        seed[0] += 1
        return _sharded_random(mesh,shape,spec,kind,seed[0])
    raw = synthetic_decoder_weights(config,rnd,progress=print)
    weights = bf16_resident_weights(mesh,config,raw)
    jax.block_until_ready(weights)
    del raw
    replicated = NamedSharding(mesh,P())
    def put(value): return jax.device_put(value,replicated)
    state = decoder.make_ws32_initial_state(mesh,config)._replace(
        position=put(np.array([64],np.int32)),context_lengths=put(np.array([65],np.int32)))
    token = put(np.array([5],np.int32))
    rope = put(np.asarray(decoder.build_ws32_main_rope_table(config)))
    policy = RequestPolicy('synthetic-d4-loop',42,64,iters+6,capacity,config.geometry.vocab_size,
                           (config.geometry.vocab_size-1,))
    bank = make_request_uniform_bank(mesh,policy)
    prompt = put(np.int32(64))
    sample = NucleusConfig()
    options = Ws32PerfOptions(sampler='nucleus_candidates',bf16_resident=True,lse_attention=True,
        dsa_two_stage=True,routed_projection=RoutedProjectionConfig(output_tile=256,contraction_tile=256))
    programs = {
        'legacy_loop': build_ws32_challenger_decoder_program(mesh,config,options=options,sampling=sample).execute,
        'packed_loop': build_packed_decoder_program(mesh,config,options=options,sampling=sample).execute,
    }
    # Same artificial 64-token prefix as the standalone decode microbenchmark.
    initial = prefill.Ws32BatchedPrefillResult(
        prefill.Ws32BatchedPrefillState(state,state.index_cache_local,prompt,put(np.bool_(True))),token)
    results = {}
    report['sampled_request_loop_synthetic'] = results
    final_states = {}
    for name, fn in programs.items():
        extra = (bank,prompt) if name == 'packed_loop' else (put(np.float32(.5)),)
        started = time.perf_counter()
        compiled = fn.lower(token,state,weights,rope,*extra).compile()
        compile_seconds = time.perf_counter()-started
        votes, transfers, events = [], [], []
        last = [None]
        def vote(valid):
            before = time.perf_counter()
            answer = bool(np.asarray(multihost_utils.process_allgather(np.bool_(valid))).all())
            votes.append(time.perf_counter()-before)
            return answer
        def replicate(value):
            transfers.append(True)
            return put(value)
        def step(t,s,*uniform):
            value = compiled(t,s,weights,rope,*(extra if name == 'packed_loop' else uniform))
            last[0] = value.decoded if name == 'packed_loop' else value
            return value
        session = (PackedRequestSession if name == 'packed_loop' else Ws32RequestSession)(
            policy,decode_step=step,replicate_uniform=replicate,fleet_all=vote,deliver=events.append,
            delivery_boundary='synthetic in-memory event append',request_started=time.perf_counter())
        session.accept_prefill(initial)
        for _ in range(5):
            if session.finished: raise RuntimeError('synthetic request reached EOS during warmup')
            session.step()
        votes.clear(); transfers.clear()
        samples = []
        wall = time.perf_counter()
        while not session.finished:
            before = time.perf_counter()
            session.step()
            samples.append((time.perf_counter()-before)*1e3)
        elapsed = time.perf_counter()-wall
        final_states[name] = (session._state,last[0].final_residual_local)
        values = np.asarray(samples)
        results[name] = dict(compile_seconds=compile_seconds,samples=len(samples),
            p50_ms=float(np.percentile(values,50)),p99_ms=float(np.percentile(values,99)),
            wall_seconds=elapsed,tokens_per_second=len(samples)/elapsed,
            model_step_p50_ms=float(np.median(session.decode_seconds[5:])*1e3),
            timed_votes=len(votes),timed_uniform_transfers=len(transfers),
            vote_wall_seconds=sum(votes),tokens=[e.token_id for e in events],
            finish_reason=events[-1].finish_reason,healthy=not session.failed,
            temp_bytes=int(compiled.memory_analysis().temp_size_in_bytes))
        print(f'request loop {name}: {results[name]["tokens_per_second"]:.2f} tok/s; {len(votes)} votes',flush=True)
        if save is not None: save()
        del compiled,session
    def same_state(a,b):
        equal = jnp.bool_(True)
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
            if jnp.issubdtype(x.dtype,jnp.floating):
                dtype = jnp.uint16 if x.dtype == jnp.bfloat16 else jnp.uint32
                x,y = lax.bitcast_convert_type(x,dtype),lax.bitcast_convert_type(y,dtype)
            equal &= jnp.all(x==y)
        return lax.pmin(equal.astype(jnp.int32),('expert','feature')).astype(jnp.bool_)
    compare = jax.jit(jax.shard_map(same_state,mesh=mesh,
        in_specs=((decoder.ws32_decoder_state_specs(),P(None,'feature')),)*2,out_specs=P(),check_vma=False))
    report['request_loop_agreement'] = dict(
        all_tokens_equal=results['legacy_loop']['tokens']==results['packed_loop']['tokens'],
        final_state_and_residual_bitwise_equal=bool(np.asarray(compare(final_states['legacy_loop'],final_states['packed_loop']))))
    if save is not None: save()


def bench_prefill_panels(mesh,report:dict,*,iters:int,save=None)->None:
    """P4 routed M32 panel projections at real gate/down geometry, packing included."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax
    from jax.sharding import NamedSharding,PartitionSpec as P
    from glm_tpu.greenfield.kernels.prefill_expert_panels import build_expert_panels
    from glm_tpu.greenfield.kernels.pallas.prefill_panel_fp8 import prefill_panel_fp8_matmul
    from glm_tpu.perf.prefill_panels import wide_prefill_panel_fp8_matmul
    result={}
    report['prefill_panel_projection']=result
    for geometry in ('gate','down'):
        if geometry=='gate':
            shape,spec,xshape,xspec=(256,2048,6144),P('expert',None,'feature'),(1024,6144),P(None,'feature')
            dtype=jnp.float32
        else:
            shape,spec,xshape,xspec=(256,6144,2048),P('expert','feature',None),(1024,2048),P()
            dtype=jnp.bfloat16
        x=_sharded_random(mesh,xshape,xspec,'bf16',2701)
        bits=_sharded_random(mesh,shape,spec,'fp8',2702)
        scale=_sharded_random(mesh,(shape[0],shape[1]//128,shape[2]//128),spec,'scale',2703)
        for pattern in ('balanced','concentrated'):
            counts=np.full(256,4,np.int32) if pattern=='balanced' else np.zeros(256,np.int32)
            if pattern=='concentrated':counts[::32]=128
            counts=jax.device_put(counts,NamedSharding(mesh,P()))
            def program(width):
                def body(x,w,s,counts):
                    panels=build_expert_panels(counts,lax.axis_index('expert')*32,rows=1024,local_groups=32)
                    if width==0:
                        value,valid=prefill_panel_fp8_matmul(x,w,s,panels,result_dtype=dtype)
                    else:
                        value,valid=wide_prefill_panel_fp8_matmul(x,w,s,panels,result_dtype=dtype,output_tile=width)
                    # Partial gate outputs differ over feature owners; expose
                    # both owner axes rather than declaring false replicas.
                    return value[None,None],valid
                return jax.jit(jax.shard_map(body,mesh=mesh,in_specs=(xspec,spec,spec,P()),
                                            out_specs=(P("expert","feature"),P()),check_vma=False))
            args=(x,bits,scale,counts)
            ref=program(0)(*args)
            rows={}
            for width in (0,256,512):
                fn=program(width).lower(*args).compile()
                actual,valid=fn(*args)
                pairs=[(np.asarray(a.data),np.asarray(b.data)) for a,b in zip(ref[0].addressable_shards,actual.addressable_shards)]
                finite=all(np.isfinite(b).all() for a,b in pairs)
                equal=all(np.array_equal(a.view(np.uint8),b.view(np.uint8)) for a,b in pairs)
                row=_timeit(fn,args,warmup=5,iters=iters)
                row.update(finite=finite,bitwise_equal=equal,contract_valid=bool(np.asarray(valid)),
                    max_abs=max(float(np.max(np.abs(a.astype(np.float32)-b.astype(np.float32)))) for a,b in pairs) if finite else None)
                name='frozen_n256' if width==0 else 'challenger_n'+str(width)
                rows[name]=row
                result[geometry+'_'+pattern]=rows
                print(f'prefill panels {geometry}/{pattern}/{name}: {row["p50_ms"]:.3f}ms, bitwise={equal}',flush=True)
                if save is not None:save()


def bench_prefill_moe_pooling(mesh,report:dict,*,iters:int,save=None,capture_boundaries=False)->None:
    """From-routes MoE suffix, including collectives; fixed 1024-row workload."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding,PartitionSpec as P
    from glm_tpu.greenfield.kernels.reference.moe import GlmMoeNumericalContract
    from glm_tpu.greenfield.kernels.ws32_prefill_moe import ws32_prefill_moe_from_routes_mapped
    from glm_tpu.perf.function_bindings import bind_dependencies
    from glm_tpu.perf.prefill_bf16 import resident_matmul,resident_matmul_f32
    contract=GlmMoeNumericalContract(stage_size=8)
    gs,ds=P('expert',None,'feature'),P('expert','feature',None)
    tables=tuple(v for i in range(2) for v in (
        _sharded_random(mesh,(256,2048,6144),gs,'fp8',5911+i),
        _sharded_random(mesh,(256,16,48),gs,'scale',5913+i)))
    tables+=(_sharded_random(mesh,(256,6144,2048),ds,'fp8',5915),
             _sharded_random(mesh,(256,48,16),ds,'scale',5916),
             _sharded_random(mesh,(2048,6144),P(None,'feature'),'bf16',5917),None,
             _sharded_random(mesh,(2048,6144),P(None,'feature'),'bf16',5918),None,
             _sharded_random(mesh,(6144,2048),P('feature'),'bf16',5919),None)
    specs=(gs,gs,gs,gs,ds,ds,P(None,'feature'),None,P(None,'feature'),None,P('feature'),None)
    moe=bind_dependencies(ws32_prefill_moe_from_routes_mapped,
                         fp8_block_matmul_f32=resident_matmul_f32,fp8_block_matmul=resident_matmul)
    def body(x,ids,rw,tables):
        if capture_boundaries:
            output,healthy,boundaries=moe(x,ids,rw,*tables,contract=contract,
                expert_panels=True,capture_boundaries=True)
            # Preserve every owner's local value, including feature partials.
            # Returning boundaries changes compilation: this is diagnostic only.
            selected={name:boundaries[name][None,None] for name in (
                'routed','shared','shared_partial','shared_reduced')}
            return output,healthy,selected
        return moe(x,ids,rw,*tables,contract=contract,expert_panels=True)
    out_specs=(P(None,'feature'),P())
    if capture_boundaries:
        out_specs+=({name:P('expert','feature') for name in (
            'routed','shared','shared_partial','shared_reduced')},)
    fn=jax.jit(jax.shard_map(body,mesh=mesh,in_specs=(P(None,'feature'),P(),P(),specs),
                           out_specs=out_specs,check_vma=False))
    x=_sharded_random(mesh,(1024,6144),P(None,'feature'),'bf16',5920)
    def put(value):return jax.device_put(value,NamedSharding(mesh,P()))
    rw=put(np.full((1024,8),1/8,np.float32))
    result={};report['prefill_moe_pooled_boundaries' if capture_boundaries else 'prefill_moe_pooled_rows']=result
    for pattern in ('balanced','concentrated'):
        indices=np.tile(np.arange(8,dtype=np.int32),(1024,1))
        if pattern=='balanced':indices=(np.arange(1024,dtype=np.int32)[:,None]+32*np.arange(8,dtype=np.int32)[None,:])%256
        ids=put(indices)
        reference=None
        boundary_reference=None
        for rows in ((128,1024) if capture_boundaries else (128,256,512,1024)):
            chunks=[(x[i:i+rows],ids[i:i+rows],rw[i:i+rows],tables) for i in range(0,1024,rows)]
            executable=fn.lower(*chunks[0]).compile()
            def run():return tuple(executable(*a) for a in chunks)
            out=run();jax.block_until_ready(out)
            arrays=[np.concatenate([np.asarray(v[0].addressable_shards[i].data) for v in out],axis=0) for i in range(4)]
            if reference is None:reference=arrays
            finite=all(np.isfinite(a).all() for a in arrays)
            equal=all(np.array_equal(np.ascontiguousarray(a).view(np.uint8),np.ascontiguousarray(b).view(np.uint8)) for a,b in zip(reference,arrays))
            timing={} if capture_boundaries else _timeit(run,(),warmup=3,iters=iters)
            timing.update(total_rows=1024,rows_per_call=rows,calls=1024//rows,
                from_routes_suffix_only=True,fp32_route_sum=False,finite=finite,bitwise_equal_to_b128=equal,
                healthy=all(bool(np.asarray(v[1])) for v in out),
                max_abs=max(float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) for a,b in zip(reference,arrays)) if finite else None,
                temp_bytes=int(executable.memory_analysis().temp_size_in_bytes))
            if capture_boundaries:
                actual={name:[np.concatenate([np.asarray(v[2][name].addressable_shards[i].data)[0,0]
                    for v in out],axis=0) for i in range(4)] for name in out[0][2]}
                if boundary_reference is None:boundary_reference=actual
                timing['boundaries']={}
                for name,values in actual.items():
                    pairs=list(zip(boundary_reference[name],values))
                    boundary_finite=all(np.isfinite(v).all() for v in values)
                    timing['boundaries'][name]=dict(
                        finite=boundary_finite,
                        bitwise_equal_to_b128=all(np.array_equal(np.ascontiguousarray(a).view(np.uint8),
                            np.ascontiguousarray(b).view(np.uint8)) for a,b in pairs),
                        differing_elements=sum(int(np.count_nonzero(a!=b)) for a,b in pairs),
                        elements=sum(int(v.size) for v in values),
                        max_abs=max(float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
                            for a,b in pairs) if boundary_finite else None)
                timing.update(timing_claim=False,returned_boundaries_change_compilation=True)
            result[pattern+'_b'+str(rows)]=timing
            if capture_boundaries:
                print(f'pooled MoE boundaries {pattern}/B{rows}: bitwise={equal}',flush=True)
            else:
                print(f'pooled MoE {pattern}/B{rows}: {timing["p50_ms"]:.3f}ms per 1024 rows; bitwise={equal}',flush=True)
            if save is not None:save()


def bench_empty_routes(mesh,report:dict,*,iters:int,save=None)->None:
    """Probe forced empty-owner grid rows after large finite live outputs."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding,PartitionSpec as P
    from glm_tpu.perf.fp8_routed_experts import RoutedProjectionConfig,fp8_routed_projection
    x=_sharded_random(mesh,(8,6144),P(None,'feature'),'bf16',4911)
    tables=tuple((_sharded_random(mesh,(256,2048,6144),P('expert',None,'feature'),'fp8',4912+i),
                  _sharded_random(mesh,(256,16,48),P('expert',None,'feature'),'scale',4914+i)) for i in range(2))
    def put(v):return jax.device_put(v,NamedSharding(mesh,P()))
    ids=put(np.arange(8,dtype=np.int32))
    live=put(np.ones(8,bool));empty=put(np.zeros(8,bool))
    poison=(x*jnp.bfloat16(1e20)).astype(jnp.bfloat16)
    result={};report['empty_route_output_windows']=result
    for count,dtype in ((2,jnp.float32),(1,jnp.bfloat16)):
        programs={}
        for fixed in (False,True):
            config=RoutedProjectionConfig(output_tile=256,contraction_tile=256,write_empty_slot=fixed)
            def body(x,tables,ids,owned,config=config):
                out=fp8_routed_projection(x,tables,ids,owned,config=config,result_dtype=dtype)
                return out[None,None]
            fn=jax.jit(jax.shard_map(body,mesh=mesh,
                in_specs=(P(None,'feature'),tuple((P('expert',None,'feature'),)*2 for _ in range(count)),P(),P()),
                out_specs=P('expert','feature'),check_vma=False))
            programs[fixed]=fn.lower(x,tables[:count],ids,live).compile()
        def arrays(value):return [np.ascontiguousarray(s.data) for s in value.addressable_shards]
        reference=arrays(programs[False](x,tables[:count],ids,live))
        for fixed,fn in programs.items():
            actual=arrays(fn(x,tables[:count],ids,live))
            row=dict(live_bitwise_equal=all(np.array_equal(a.view(np.uint8),b.view(np.uint8)) for a,b in zip(reference,actual)),
                     trials=iters,empty_nonzero_counts=[],empty_nonfinite_counts=[],
                     poison_all_finite=True,poison_all_nonzero=True,
                     temp_bytes=int(fn.memory_analysis().temp_size_in_bytes))
            for _ in range(iters):
                filled=arrays(fn(poison,tables[:count],ids,live))
                row['poison_all_finite'] &= all(np.isfinite(a).all() for a in filled)
                row['poison_all_nonzero'] &= all(np.count_nonzero(a)>0 for a in filled)
                blank=arrays(fn(x,tables[:count],ids,empty))
                row['empty_nonzero_counts'].append(sum(int(np.count_nonzero(a)) for a in blank))
                row['empty_nonfinite_counts'].append(sum(int(np.count_nonzero(~np.isfinite(a))) for a in blank))
            row['poison_all_finite']=bool(row['poison_all_finite'])
            row['poison_all_nonzero']=bool(row['poison_all_nonzero'])
            label=f'tables{count}_'+('explicit_empty_store' if fixed else 'original')
            result[label]=row
            print(f'empty routed output {label}: max nonzero={max(row["empty_nonzero_counts"])} live equal={row["live_bitwise_equal"]}',flush=True)
            if save is not None:save()


def bench_owned_attention(mesh,report:dict,*,iters:int,save=None,feature_rows=False)->None:
    """Bounded owner buffers versus full-K LSE, including mixed local fallback."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding,PartitionSpec as P
    from glm_tpu.greenfield.kernels.reference.attention import MlaNumericalContract,StageLocalKvLayout,SparseAttentionResult
    from glm_tpu.greenfield.kernels.reference.dsa import SelectedPositions
    from glm_tpu.greenfield.kernels.pallas.sparse_attention import SparseMlaConfig
    from glm_tpu.perf.lse_attention import lse_attention_mapped
    from glm_tpu.perf.feature_row_attention import feature_row_lse_attention
    rows,capacity=32,131072
    contract,layout=MlaNumericalContract(),StageLocalKvLayout(local_parallel_size=8)
    q=_sharded_random(mesh,(rows,64,512),P(None,'expert'),'bf16',3901)
    r=_sharded_random(mesh,(rows,64,64),P(None,'expert'),'bf16',3902)
    cache=_sharded_random(mesh,(8,256,64,640),P('expert'),'bf16',3903)
    def put(x):return jax.device_put(x,NamedSharding(mesh,P()))
    tables=put(np.broadcast_to(np.arange(255,-1,-1,dtype=np.int32),(rows,256)))
    lengths=put(np.full(rows,capacity,np.int32))
    result={}
    report['owned_attention_buffers']=result
    for pattern in ('balanced','concentrated','mixed'):
        balanced=np.arange(2048,dtype=np.int32)
        concentrated=(np.arange(32,dtype=np.int32)[:,None]*512+np.arange(64,dtype=np.int32)[None]).reshape(-1)
        positions=np.tile(concentrated if pattern=='concentrated' else balanced,(rows,1))
        counts=np.full(rows,2048,np.int32)
        if pattern=='mixed':
            positions[0]=concentrated
            positions[-1]=-1;counts[-1]=0
        positions,counts=put(positions),put(counts)
        def program(bound,split_rows=False):
            def body(q,r,c,t,p,n,length):
                attention=feature_row_lse_attention if split_rows else lse_attention_mapped
                return attention(q,r,c[0],t,SelectedPositions(p,n),length,
                    contract=contract,layout=layout,config=SparseMlaConfig(segment_block=512),
                    validate_finite=True,owned_key_capacity=bound)
            return jax.jit(jax.shard_map(body,mesh=mesh,
                in_specs=(P(None,'expert'),P(None,'expert'),P('expert'),P(),P(),P(),P()),
                out_specs=SparseAttentionResult(P(None,'expert'),P(None,'expert'),P()),check_vma=False))
        args=(q,r,cache,tables,positions,counts,lengths)
        reference=program(None)(*args)
        rows_out={}
        cases=[(None,False),(512,False)]+([(512,True)] if feature_rows else [])
        for bound,split_rows in cases:
            executable=program(bound,split_rows).lower(*args).compile()
            output=executable(*args)
            equal=all(np.array_equal(np.ascontiguousarray(a.data).view(np.uint8),np.ascontiguousarray(b.data).view(np.uint8))
                      for x,y in zip(reference,output) for a,b in zip(x.addressable_shards,y.addressable_shards))
            row=_timeit(executable,args,warmup=5,iters=iters)
            row.update(bitwise_equal=equal,healthy=bool(np.asarray(output.contract_valid).all()),
                temp_bytes=int(executable.memory_analysis().temp_size_in_bytes))
            label='full2048' if bound is None else 'bounded512'
            if split_rows:label+='_feature_rows'
            rows_out[label]=row
            result[pattern]=rows_out
            print(f'owned attention {pattern}/{label}: {row["p50_ms"]:.3f}ms, bitwise={equal}',flush=True)
            if save is not None:save()


def bench_step(mesh, report: dict, *, iters: int, capacity: int, trace_dir: Path | None, save=None, variants: set[str] | None = None) -> None:
    """Complete 78-layer greedy decode step, frozen vs challenger, synthetic weights."""

    import json as _json

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P

    from glm_tpu.greenfield.runtime import ws32_decoder as decoder
    from glm_tpu.greenfield.types import ModelGeometry
    from glm_tpu.perf.fp8_routed_experts import RoutedProjectionConfig
    from glm_tpu.perf.ws32_decoder_challenger import Ws32PerfOptions, build_ws32_challenger_decoder_program

    raw = _json.loads((REPO / "configs/glm-5.2-fp8-config.json").read_text())
    geometry = ModelGeometry.from_hf_config(raw)
    config = decoder.Ws32DecoderConfig(geometry, capacity, host_main_rope_table=True)
    seed = [0]

    def rnd(shape, spec, kind):
        seed[0] += 1
        return _sharded_random(mesh, shape, spec, kind, seed[0])

    weights = synthetic_decoder_weights(config, rnd, progress=print)
    jax.block_until_ready(weights)
    replicated = NamedSharding(mesh, P())
    state = decoder.make_ws32_initial_state(mesh, config)
    # Pretend a 64-token prompt is resident so DSA/attention see a live prefix.
    state = state._replace(position=jax.device_put(np.asarray([64], np.int32), replicated),
                           context_lengths=jax.device_put(np.asarray([65], np.int32), replicated))
    rope_table = jax.device_put(np.asarray(decoder.build_ws32_main_rope_table(config)), replicated)
    token = jax.device_put(np.asarray([5], np.int32), replicated)
    out: dict[str, Any] = {}
    from glm_tpu.perf.bf16_resident import bf16_resident_weights

    tiles = RoutedProjectionConfig(output_tile=256, contraction_tile=256)
    frozen_execute = decoder.build_ws32_decoder_program(mesh, config).execute
    challenger_execute = build_ws32_challenger_decoder_program(
        mesh, config, options=Ws32PerfOptions(sampler="greedy", routed_projection=tiles)).execute
    bf16_execute = build_ws32_challenger_decoder_program(
        mesh, config, options=Ws32PerfOptions(sampler="greedy", routed_projection=tiles, bf16_resident=True)).execute
    started = time.perf_counter()
    bf16_weights = bf16_resident_weights(mesh, config, weights)
    jax.block_until_ready(bf16_weights)
    print(f"step: bf16-resident weights decoded in {time.perf_counter() - started:.1f} s", flush=True)

    def frozen_step(t, s, w, r):
        return frozen_execute(t, s, w, r)

    def challenger_step(t, s, w, r):
        return challenger_execute(t, s, w, r)

    def bf16_step(t, s, w, r):
        return bf16_execute(t, s, w, r)

    programs = {
        "frozen_greedy": (jax.jit(frozen_step), weights),
        "challenger_greedy_grouped_t256": (jax.jit(challenger_step), weights),
        "challenger_greedy_grouped_t256_bf16_resident": (jax.jit(bf16_step), bf16_weights),
    }
    for name, lse, dsa, fused in (
        ("bf16_lse", True, False, False), ("bf16_dsa", False, True, False),
        ("bf16_lse_dsa", True, True, False), ("bf16_lse_dsa_fused", True, True, True),
    ):
        execute = build_ws32_challenger_decoder_program(
            mesh, config, options=Ws32PerfOptions(sampler="greedy", routed_projection=tiles,
                bf16_resident=True, lse_attention=lse, dsa_two_stage=dsa,
                fused_feature_reductions=fused)).execute
        programs[name] = (execute, bf16_weights)
    if variants is not None:
        if not variants or not variants <= programs.keys():
            raise ValueError(f"unknown step variants: {variants - programs.keys()}")
        programs = {name: value for name, value in programs.items() if name in variants}
    tokens_seen: dict[str, list[int]] = {}
    for name, (fn, program_weights) in programs.items():
        started = time.perf_counter()
        compiled = fn.lower(token, state, program_weights, rope_table).compile()
        compile_s = time.perf_counter() - started
        mem = compiled.memory_analysis()
        print(f"step {name}: compiled in {compile_s:.0f} s", flush=True)
        # Steady state: feed the produced token/state back like the request loop does.
        current_state, current_token = state, token
        samples = []
        tokens_seen[name] = []
        for i in range(iters + 5):
            started = time.perf_counter()
            result = compiled(current_token, current_state, program_weights, rope_table)
            jax.block_until_ready(result)
            elapsed = (time.perf_counter() - started) * 1e3
            if i >= 5:
                samples.append(elapsed)
            current_state, current_token = result.state, result.next_token
            if i < 16:
                tokens_seen[name].append(int(np.asarray(result.next_token)[0]))
        arr = np.asarray(samples)
        from glm_tpu.perf.decode_diagnostics import activation_stats
        out[name] = dict(
            p50_ms=float(np.percentile(arr, 50)), p99_ms=float(np.percentile(arr, 99)),
            min_ms=float(arr.min()), samples=int(arr.size), compile_s=compile_s,
            tokens_per_s_at_p50=float(1e3 / np.percentile(arr, 50)),
            argument_bytes=int(getattr(mem, "argument_size_in_bytes", 0)),
            temp_bytes=int(getattr(mem, "temp_size_in_bytes", 0)),
            contract_valid=bool(np.asarray(result.state.contract_valid).all()),
            first_tokens=tokens_seen[name],
            final_residual=activation_stats(result.final_residual_local),
        )
        print(f"step {name}: p50 {out[name]['p50_ms']:.2f} ms ({out[name]['tokens_per_s_at_p50']:.2f} tok/s) p99 {out[name]['p99_ms']:.2f} ms", flush=True)
        report["decode_step_78_layers_synthetic"] = out
        if save is not None:
            save()
        if trace_dir is not None:
            # Every process traces its own four chips (the protected runs also
            # collect one XPlane per host); a single-host trace killed the run.
            host_dir = trace_dir / name / socket.gethostname()
            host_dir.mkdir(parents=True, exist_ok=True)
            with jax.profiler.trace(str(host_dir), create_perfetto_link=False):
                for _ in range(2):
                    result = compiled(current_token, current_state, program_weights, rope_table)
                    jax.block_until_ready(result)
                    current_state, current_token = result.state, result.next_token
            print(f"step {name}: trace written under {host_dir}", flush=True)
        del compiled
    report["decode_step_78_layers_synthetic"] = out


def summarize(run_dir: Path) -> dict[str, Any]:
    """Merge the eight per-rank receipts of one run into fleet min/max per metric."""

    ranks = [json.loads(p.read_text()) for p in sorted(run_dir.glob("microbench.rank*.json"))]
    if not ranks:
        raise FileNotFoundError(f"no microbench.rank*.json under {run_dir}")
    if len(ranks) != 8 or {r.get("rank") for r in ranks} != set(range(8)):
        raise ValueError("fleet summary requires exactly eight distinct rank receipts")
    if any(not r.get("finished_utc") or r.get("devices") != 32 for r in ranks):
        raise ValueError("fleet summary requires completed 32-device runs on every rank")
    identities = {(r.get("which"), r.get("jax"), json.dumps(r.get("source_sha256"), sort_keys=True)) for r in ranks}
    if len(identities) != 1:
        raise ValueError("fleet benchmark selection, runtime or source fingerprints disagree")

    def merge(values: list[Any]) -> Any:
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            return {"min": min(values), "max": max(values)} if len(set(values)) > 1 else values[0]
        if all(isinstance(v, dict) for v in values):
            if any(set(v)!=set(values[0]) for v in values):
                raise ValueError('fleet metric fields differ; cannot merge partial evidence')
            return {k: merge([v[k] for v in values]) for k in sorted(values[0])}
        return values[0] if len({json.dumps(v, sort_keys=True) for v in values}) == 1 else values

    skip = {"rank", "hostname", "jax_process_index", "started_utc", "finished_utc"}
    merged = merge([{k: v for k, v in r.items() if k not in skip} for r in ranks])
    merged["ranks"] = sorted(r["rank"] for r in ranks)
    merged["run_dir"] = run_dir.name
    merged["hosts"] = sorted(r["hostname"] for r in ranks)
    merged["schema"] = "glm_perf_tpu_microbench_fleet_summary_v1"
    from hashlib import sha256
    merged['originals_sha256']={p.name:sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(run_dir.glob('microbench.rank*.json'))}
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize", type=Path, default=None,
                        help="merge microbench.rank*.json of this run directory and print/write the fleet summary")
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--coordinator")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--which", default="moe,sampler,tiles")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--step-iters", type=int, default=100)
    parser.add_argument("--capacity", type=int, default=8192)
    parser.add_argument("--prompt-length", type=int, default=2048)
    parser.add_argument("--prefill-pending-cache-rows", action="store_true")
    parser.add_argument("--prefill-variants", default="frozen,p1p2,p1p2_bf16")
    parser.add_argument("--prefill-block-rows", type=int, choices=(128,512,1024), default=128,
                        help="Pooled MoE windows require the sole resident variant; numerical boundary")
    parser.add_argument("--step-variants", help="comma-separated program names; default compares all variants")
    parser.add_argument("--prefill-owned-key-capacity", type=int, default=None,
                        help="opt-in bounded owner buffers for the sole p1p2_bf16 full-prompt variant")
    parser.add_argument("--prefill-wide-indexshare",action="store_true",
                        help="compare narrow and <=128-row sparse shared-indexer prefixes using the same weights")
    parser.add_argument("--prefill-sparse-slice",action="store_true",
                        help="profile only the final eight-layer sparse pattern; rates are not full-model throughput")
    parser.add_argument("--trace", action="store_true",
                        help="trace two decode steps after timing, or two prefill warm first blocks before timing, on every host")
    args = parser.parse_args()
    if args.summarize is not None:
        text = json.dumps(summarize(args.summarize), indent=2, sort_keys=True)
        if args.summary_output is not None:
            args.summary_output.write_text(text + "\n")
        print(text)
        return 0
    if args.coordinator is None or args.output is None:
        parser.error("--coordinator and --output are required to run the benchmark")
    rank = _rank_from_hostname()
    args.output.mkdir(parents=True, exist_ok=True)

    import jax

    jax.distributed.initialize(coordinator_address=args.coordinator, num_processes=8, process_id=rank)
    if jax.default_backend() != "tpu" or jax.device_count() != 32 or len(jax.local_devices()) != 4:
        raise RuntimeError("expected the 8x4 TPU v4 runtime")
    mesh = _ws32_mesh()
    report: dict[str, Any] = dict(
        schema="glm_perf_tpu_microbench_v1", rank=rank, hostname=socket.gethostname(),
        jax=jax.__version__, devices=jax.device_count(),
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        which=args.which, iters=args.iters,
        synthetic_generator="partition_axes_only_v2",
    )
    import hashlib

    sources = sorted((REPO / "glm_tpu/perf").glob("*.py")) + [Path(__file__).resolve(), REPO / "configs/glm-5.2-fp8-config.json"]
    report["source_sha256"] = {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    which = set(args.which.split(","))
    report["jax_process_index"] = int(jax.process_index())
    receipt = args.output / f"microbench.rank{rank}.json"

    def save() -> None:
        receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"microbench rank={rank} process_index={jax.process_index()} devices={jax.device_count()}", flush=True)
    if "moe" in which:
        bench_moe(mesh, report, iters=args.iters)
        save()
    if "sampler" in which:
        bench_sampler(mesh, report, iters=args.iters)
        save()
    if "tiles" in which:
        bench_tiles(mesh, report, iters=args.iters)
        save()
    if "attention" in which:
        bench_attention(mesh, report, iters=args.iters, save=save)
        save()
    if "attention_global_tiles" in which:
        bench_attention(mesh,report,iters=args.iters,save=save,global_tiles=True)
        save()
    if "dsa" in which:
        bench_dsa(mesh, report, iters=args.iters, save=save)
        save()
    if "dsa_payload_merge" in which:
        bench_dsa_payload_merge(mesh,report,iters=args.iters,save=save)
        save()
    if "compact_feature_reduce" in which:
        bench_compact_feature_reduce(mesh,report,iters=args.iters,save=save)
        save()
    if "prefill" in which:
        bench_prefill_primitives(mesh, report, iters=args.iters, save=save)
        save()
    if "fused_projections" in which:
        bench_fused_projections(mesh, report, iters=args.iters,save=save,artifact_dir=args.output / "fused_projection_hlo")
        save()
    if "prefill_model" in which:
        bench_prefill_model(mesh, report, prompt_length=args.prompt_length, capacity=args.capacity,
                            variants=set(args.prefill_variants.split(",")),save=save,
                            pending_cache_rows=args.prefill_pending_cache_rows,
                            trace_dir=(args.output / "trace") if args.trace else None,
                            owned_key_capacity=args.prefill_owned_key_capacity,
                            block_rows=args.prefill_block_rows,wide_indexshare=args.prefill_wide_indexshare,
                            sparse_slice=args.prefill_sparse_slice)
        save()
    if "owned_attention" in which:
        bench_owned_attention(mesh,report,iters=args.iters,save=save)
        save()
    if "feature_row_attention" in which:
        bench_owned_attention(mesh,report,iters=args.iters,save=save,feature_rows=True)
        save()
    if "empty_routes" in which:
        bench_empty_routes(mesh,report,iters=args.iters,save=save)
        save()
    if "prefill_moe_pooling" in which:
        bench_prefill_moe_pooling(mesh,report,iters=args.iters,save=save)
        save()
    if "prefill_moe_pooling_boundaries" in which:
        bench_prefill_moe_pooling(mesh,report,iters=args.iters,save=save,capture_boundaries=True)
        save()
    if "prefill_panels" in which:
        bench_prefill_panels(mesh,report,iters=args.iters,save=save)
        save()
    if "request_loop" in which:
        bench_request_loop(mesh,report,iters=args.step_iters,capacity=args.capacity,save=save)
        save()
    if "step" in which:
        bench_step(mesh, report, iters=args.step_iters, capacity=args.capacity,
                   trace_dir=(args.output / "trace") if args.trace else None, save=save,
                   variants=set(args.step_variants.split(",")) if args.step_variants else None)
        save()
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()
    print(f"MICROBENCH_DONE rank={rank}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
