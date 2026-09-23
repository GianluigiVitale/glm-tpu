"""Fixed sampled B128/B114 production graph preparation; no launch authority.

Reuse the authenticated complete checkpoint metadata and abstract operand
builder. Capacity166912 covers every pinned prompt plus the FULL generation cap.
Both shapes use the already-tested compact pending-row/capture-barrier path
with explicit state donation. This does NOT inherit DB620's memory fit: all
resident sampled programs and repeated-request weights need actual admission.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import subprocess
from typing import Any

from glm_tpu.greenfield.validation.ws32_prefill import BatchedPrefillPlan, OWNED_STATE_CONTRACT
from glm_tpu.greenfield.validation.ws32_prefill_admission import MODEL_SOURCE
# Moved verbatim to glm_tpu.runner.kv_cache_manager (S2a); re-exported for research importers.
from glm_tpu.runner.kv_cache_manager import build_cache_initializer  # noqa: F401

BASE_PIN = "edecdd94052655031bd38b6ad3029525f1adc572"
SAMPLED_SOURCE = "glm_tpu/greenfield/runtime/ws32_sampled_request.py"
SAMPLED_SHA = "d085ea5f04aff1a0a6d15147995fc3d2c95e43e0cb07cc498983ce0d4866a0b0"
# Session release changes only host ownership, never the compiled model graph.
SESSION_SOURCE = "glm_tpu/greenfield/runtime/ws32_request_session.py"
SESSION_SHA = "71697ddc4b2867a16c4a3645abc040f9da7f9304e8ff878d1cc3f307b32cff37"
PLAN = BatchedPrefillPlan(2034, 128, 166912, mlp_window=True, tail_graph_rows=114)


def require_source(repo: Path) -> None:
    if sha256((repo/SAMPLED_SOURCE).read_bytes()).hexdigest() != SAMPLED_SHA:
        raise ValueError("sampled model source differs from reviewed scope addition")
    if sha256((repo/SESSION_SOURCE).read_bytes()).hexdigest() != SESSION_SHA:
        raise ValueError("native request ownership source differs")
    changed = subprocess.run([
        "git", "-C", str(repo), "diff", "--quiet", BASE_PIN, "--", *MODEL_SOURCE,
        ":(exclude)"+SAMPLED_SOURCE, ":(exclude)"+SESSION_SOURCE,
    ], check=False)
    if changed.returncode:
        raise ValueError("sampled benchmark changed unrelated frozen model source")
    # Whole host control/worker source must additionally be clean and pinned by
    # the protected outer entry. This graph check is not that launch authority.


@dataclass(frozen=True)
class OwnedSampledProgram:
    template: Any
    execute: Any
    ownership_contract: str = OWNED_STATE_CONTRACT


def build_prefill_pair(mesh: Any, config: Any, **options: Any) -> dict[str, OwnedSampledProgram]:
    import jax
    from glm_tpu.greenfield.kernels.ws32_sampling import NucleusConfig
    from glm_tpu.greenfield.runtime.ws32_sampled_request import build_ws32_sampled_prefill_program
    if config.context_capacity != PLAN.context_capacity:
        raise ValueError("native benchmark capacity differs")
    result = {}
    for role, rows in PLAN.graph_rows:
        program = build_ws32_sampled_prefill_program(
            mesh, config, sampling=NucleusConfig(), block_rows=rows,
            mlp_window=True, **options,
        )
        result[role] = OwnedSampledProgram(
            program, jax.jit(program.execute, donate_argnums=(2,)))
    return result


def read_metadata(repo: Path) -> Any:
    from scripts.greenfield.ws32_rolled_prefill_compile import read_metadata as original
    return original(repo, full_canonical=True, pending_cache_rows=True,
                    flat_pending_rows=True, capture_barrier=True, native_benchmark=True)


def prepare(mesh: Any, metadata: Any, *, repo: Path) -> Any:
    from scripts.greenfield.ws32_rolled_prefill_compile import prepare as original
    return original(mesh, metadata, repo=repo, full_canonical=True,
                    long_context_label="256k_e0", pending_cache_rows=True,
                    flat_pending_rows=True, capture_barrier=True, native_benchmark=True)


@dataclass(frozen=True)
class AbstractNativeCompanions:
    config: Any
    programs: dict[str, Any]
    inputs: dict[str, tuple[Any, ...]]


def prepare_companions(mesh: Any, prefill_pair: Any, *, repo: Path) -> AbstractNativeCompanions:
    """Prepare sampled decode and original materializers without weight allocation.

    Reuses the production operand construction exercised by the long companion
    test. The overlay schema is the ORIGINAL loader's contract, not invented
    shapes. Both materialization boundaries remain separately compiled calls.
    Decode/observer consume state argument1, never weights, rope or RNG input.
    These objects are for compilation/admission, NOT substitute loaded arrays.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from glm_tpu.greenfield.runtime import ws32_decoder as dec
    from glm_tpu.greenfield.runtime.ws32_sampled_request import build_ws32_sampled_decoder_program
    from glm_tpu.greenfield.kernels.ws32_sampling import NucleusConfig
    from glm_tpu.greenfield.checkpoint.ws32_strategy_nd_dense import (
        _LOCAL_CONTRACT, strategy_nd_dense_tensor_names,
    )
    from scripts.greenfield.run_short_decoder_ws32 import _geometry

    require_source(repo)
    _, _, state, raw_weights, _, rope, uniform = prefill_pair.inputs['prefill_chunk']
    config = dec.Ws32DecoderConfig(
        _geometry(), PLAN.context_capacity, exact_dsa=True,
        strategy_nd_dense=True, host_main_rope_table=True)
    raw_config = dec.Ws32DecoderConfig(
        config.geometry, config.context_capacity, host_main_rope_table=True)

    def abstract(shape: Any, dtype: Any, spec: Any = P()) -> Any:
        return jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(mesh, spec))

    def placed(values: Any, specs: Any) -> Any:
        return jax.tree.map(lambda value, spec: abstract(value.shape, value.dtype, spec),
                            values, specs)

    arrays = dict(zip(jax.tree.leaves(dec.ws32_decoder_weight_names(raw_config)),
                      jax.tree.leaves(raw_weights), strict=True))
    arrays.update({name: abstract(shape, dtype, P(*spec)) for layer in range(3)
                   for name, (_, dtype, shape, spec) in zip(
                       strategy_nd_dense_tensor_names(layer), _LOCAL_CONTRACT, strict=True)})
    weights = dec.bind_ws32_decoder_weights(
        {name: arrays[name] for name in jax.tree.leaves(dec.ws32_decoder_weight_names(config))}, config)
    materializer = dec.build_ws32_exact_dsa_materializer_program(mesh, config)
    raw = dec.select_ws32_exact_dsa_raw_weights(weights, config)
    decoded = placed(jax.eval_shape(materializer.decode, raw),
                     dec.ws32_decoded_exact_dsa_specs(config))
    exact = placed(jax.eval_shape(materializer.promote, decoded),
                   dec.ws32_exact_dsa_specs(config))
    decoder = build_ws32_sampled_decoder_program(mesh, config, sampling=NucleusConfig())
    inputs = (abstract((1,), jnp.int32), state.decoder, weights, exact, rope, uniform)
    jobs = (
        ('exact_materialize', jax.jit(materializer.decode), (raw,)),
        ('exact_promote', jax.jit(materializer.promote), (decoded,)),
        ('observer', jax.jit(decoder.observe, donate_argnums=(1,)), inputs),
        ('decode', jax.jit(decoder.execute, donate_argnums=(1,)), inputs),
        ('cache_probe', jax.jit(decoder.probe_cache_write), (state.decoder,)),
    )
    return AbstractNativeCompanions(config,
        {name: fn for name, fn, _ in jobs}, {name: args for name, _, args in jobs})


# Literal CPU TPU-target RAW registrations. Actual TPU optimized HLO, compiler
# allocations and physical peak HBM are still required before model dispatch.
RAW = {
    'prefill_chunk': (21108622, '6ec2c32213ff91f26847adb9436ee99b2ebd9090e0610bc15e111e54663fbde6'),
    'prefill_tail': (21210523, '97b5540a14d1f7427af82211d7b77db25c4292164dad2fb466d6fc640005eb8b'),
    'exact_materialize': (481942, '1d925d96f4770e6edd5cef41e1c6f8f4071e039ad93b1cf7d96380fbfdf6f36e'),
    'exact_promote': (50861, 'e38eb7a45472107a383114c25039b9cfa58c2120fe98592f62d8f47996d3ffff'),
    'observer': (28619306, 'c5e3847c0e2413432c3f076d995a9dce55a7b3d5b90dac06c78648b4d91e1932'),
    'decode': (28604427, '635c36f51af31526020afc5aea1481b1c4d91a6fe1678b43c2064c257d49c244'),
    'cache_probe': (16789, 'd54f113e7e69294bf8761bb18242a7658189741aef4f9e014989837a47fcacfa'),
}


def inspect_hlo(stable: str, optimized: str, *, repo: Path, graph: str,
                expected_optimized: str) -> dict[str, Any]:
    """Source/RAW-bound sampled profile; worker and sealer share this inspector.

    The original structural checks remain, except the explicit output-only
    vocabulary gather replaces greedy head reductions in sampled prefill.
    Fresh optimized bytes are recorded, never normalized or inherited. This
    inspector alone cannot authorize a dispatch or claim runtime memory fit.
    """
    import json
    from glm_tpu.greenfield.benchmarking.ws32_decoder import (
        validate_ws32_decoder_hlo, validate_ws32_exact_dsa_materializer_hlo,
    )
    from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import PrefillHloIndex
    from glm_tpu.greenfield.benchmarking.ws32_pallas_one_layer import (
        _computation_base, _live_instruction_closure,
    )
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
    from scripts.greenfield import ws32_delivery_hlo as original

    require_source(repo)
    if graph not in RAW or (len(stable.encode()), sha256(stable.encode()).hexdigest()) != RAW[graph]:
        raise ValueError('native sampled RAW differs from registered production graph')
    digest, identity_policy = original.optimized_identity(optimized, expected_optimized)
    pins = dict(expected_stablehlo_sha256=RAW[graph][1], expected_optimized_hlo_sha256=digest)
    if graph == 'cache_init':
        module = parse_hlo_module(optimized)
        allowed = {'parameter','constant','broadcast','iota','compare','and',
                   'reshape','tuple','copy','bitcast','slice','fusion','convert'}
        roots = [op for op in module.instructions if _computation_base(op.computation) == 'ENTRY'
                 and op.raw_line.lstrip().startswith('ROOT ')]
        pages = PLAN.context_capacity // 512
        expected = [('bf16',(78,pages,64,640)), ('bf16',(21,pages,64,128)),
                    ('s32',(1,2048)), ('s32',(1,)), ('f32',(1,2048)),
                    ('s32',(1,)), ('s32',(1,pages)), ('s32',(1,)), ('pred',(1,)),
                    ('bf16',(21,pages,64,128)), ('s32',()), ('pred',())]
        if (module.num_partitions != 32 or module.collectives or any(
                op.raw_opcode not in allowed for op in module.instructions)
                or len(roots) != 1 or [(s.dtype,s.dimensions) for s in roots[0].result_shapes] != expected):
            raise ValueError('native cache initializer contains unexpected computation/transport')
        report = dict(passed=True, instruction_count=len(module.instructions),
                      scope='PURE_FRESH_CACHE_INITIALIZER_NO_MODEL_OR_COLLECTIVES')
    elif graph.startswith('prefill_'):
        module = parse_hlo_module(optimized)
        report = original.check_index(PrefillHloIndex(module),
            context_capacity=PLAN.context_capacity, block_rows=dict(PLAN.graph_rows)[graph],
            live_instructions=_live_instruction_closure(module.instructions), nucleus_head=True)
    elif graph.startswith('exact_'):
        report = validate_ws32_exact_dsa_materializer_hlo(stable, optimized, kind=graph, **pins).to_dict()
    else:
        report = validate_ws32_decoder_hlo(stable, optimized, kind=graph, hidden_size=6144,
            exact_dsa=True, strategy_nd_dense=True, host_main_rope_table=True, **pins).to_dict()
    if report.get('passed') is not True:
        raise ValueError(f'native sampled structural refusal: {graph}: {report}')
    return json.loads(json.dumps(dict(report, graph=graph,
        profile='ws32_native_sampled_request_v1', raw_stablehlo_sha256=RAW[graph][1],
        raw_optimized_hlo_sha256=digest, optimized_identity_policy=identity_policy,
        dispatch_authorized=False, numerical_inheritance=False, runtime_hbm_proven=False)))


RAW['cache_init'] = (3901, 'e3eecd52cea4aa59734ff1d760a06dea646ae356a1dd9693a8157208d7358926')
