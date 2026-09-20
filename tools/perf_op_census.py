"""CPU-only structural census of the WS32 decode step: frozen versus challenger.

Counts, per layer kind and for whole compiled programs, the operations whose
fixed per-call latency dominates one-row decode on TPU v4: collectives,
Pallas kernel launches, conditionals and sorts.  It never touches a TPU,
never loads weights and makes no timing claim; the numbers feed
``docs/research/glm52-tpu-20260920/history/REFERENCE_LOWHANGING_FRUIT_20260919.md``.

Two views are reported:

* ``jaxpr``: primitive counts of one traced layer of each kind (dense/full,
  sparse/full, sparse/shared) and of the whole 8-layer fixture step, before
  XLA optimization.  ``inside_cond`` is the share that sits under a
  ``lax.cond`` branch and therefore executes only when taken.  Extrapolation
  to GLM-5.2's 78 layers (3 dense/full, 18 sparse/full, 57 sparse/shared)
  is arithmetic on these per-layer counts.
* ``hlo``: opcode counts of the compiled 32-device CPU executable.  CPU and
  TPU share XLA's HLO-level simplification/CSE passes, so this answers
  "does XLA de-duplicate the repeated per-layer metadata sorts"; it does not
  count Pallas kernels (interpret mode expands them) and its fusion structure
  is not the TPU one.

Run from the repository root; it re-executes itself with a forced 32-device
CPU platform when needed:

    JAX_PLATFORMS=cpu python tools/perf_op_census.py --output census.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
GLM_LAYER_KINDS = {"dense_full": 3, "sparse_full": 18, "sparse_shared": 57}
JAXPR_KEYS = (
    "psum",
    "all_gather",
    "all_to_all",
    "ppermute",
    "pmin",
    "pmax",
    "pallas_call",
    "cond",
    "sort",
    "while",
    "scan",
)
HLO_KEYS = (
    "all-reduce",
    "all-gather",
    "all-to-all",
    "collective-permute",
    "reduce-scatter",
    "sort",
    "conditional",
    "while",
    "custom-call",
)


def _ensure_cpu32() -> None:
    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count=32" in flags and os.environ.get(
        "JAX_PLATFORMS"
    ) == "cpu":
        return
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=(flags + " --xla_force_host_platform_device_count=32").strip(),
        PYTHONPATH=os.pathsep.join(
            p for p in (str(REPO), os.environ.get("PYTHONPATH", "")) if p
        ),
    )
    raise SystemExit(subprocess.call([sys.executable, *sys.argv], env=env, cwd=REPO))


def _count_jaxpr(jaxpr: Any, counts: Counter, *, inside_cond: bool = False) -> None:
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        for key in JAXPR_KEYS:
            if key in name:
                counts[key] += 1
                if inside_cond:
                    counts[key + "_inside_cond"] += 1
                break
        nested_inside = inside_cond or ("cond" in name)
        for value in eqn.params.values():
            for sub in _sub_jaxprs(value):
                _count_jaxpr(sub, counts, inside_cond=nested_inside)


def _sub_jaxprs(value: Any) -> list[Any]:
    from jax._src import core

    if isinstance(value, core.ClosedJaxpr):
        return [value.jaxpr]
    if isinstance(value, core.Jaxpr):
        return [value]
    if isinstance(value, (tuple, list)):
        out: list[Any] = []
        for item in value:
            out.extend(_sub_jaxprs(item))
        return out
    return []


def jaxpr_census(fn: Any, *args: Any) -> dict[str, int]:
    import jax

    counts: Counter = Counter()
    _count_jaxpr(jax.make_jaxpr(fn)(*args).jaxpr, counts)
    return {key: int(counts.get(key, 0)) for key in sorted(counts)}


def hlo_census(compiled_text: str) -> dict[str, int]:
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

    counts = Counter(x.opcode for x in parse_hlo_module(compiled_text).instructions)
    return {key: int(counts.get(key, 0)) for key in HLO_KEYS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="jaxpr census only (no 32-device CPU compilation of whole programs)",
    )
    args = parser.parse_args()
    _ensure_cpu32()

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from jax._src.pallas.mosaic import tpu_info

    tpu_info.registry["cpu"] = lambda: tpu_info.get_tpu_info_for_chip(
        tpu_info.ChipVersion.TPU_V4, 1
    )
    tpu_info.get_tpu_info.cache_clear()
    sys.path.insert(0, str(REPO))
    from glm_tpu.greenfield.kernels.pallas import SparseMlaConfig
    from glm_tpu.greenfield.kernels.ws32_layer import ws32_transformer_layer_mapped
    from glm_tpu.greenfield.kernels.ws32_sampling import NucleusConfig
    from glm_tpu.greenfield.runtime import ws32_batched_prefill as prefill
    from glm_tpu.greenfield.runtime import ws32_decoder as decoder
    from glm_tpu.greenfield.runtime.ws32_sampled_request import (
        build_ws32_sampled_decoder_program,
    )
    from glm_tpu.perf.fp8_routed_experts import RoutedProjectionConfig
    from glm_tpu.perf.ws32_decoder_challenger import (
        Ws32PerfOptions,
        build_ws32_challenger_decoder_program,
        ws32_transformer_layer_challenger_mapped,
    )
    from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture

    mesh = Mesh(np.asarray(jax.devices(), object).reshape(8, 4), ("expert", "feature"))
    config, weights, _ = fixture(mesh)
    state = prefill.make_ws32_batched_prefill_state(mesh, config, prompt_length=3).decoder
    rope = jax.device_put(
        jnp.asarray(decoder.build_ws32_main_rope_table(config), jnp.bfloat16),
        NamedSharding(mesh, P()),
    )
    token = jax.device_put(jnp.array([5], jnp.int32), NamedSharding(mesh, P()))
    uniform = jax.device_put(jnp.float32(0.5), NamedSharding(mesh, P()))
    sampling = NucleusConfig()
    tiles = RoutedProjectionConfig(
        block_shape=(128, 128), output_tile=128, contraction_tile=128
    )
    interpret = dict(sparse_attention_interpret=True, linear_interpret=True)

    # ---- per-layer census: one traced layer of each kind, frozen vs challenger.
    layer_ids = {"dense_full": 0, "sparse_full": 6, "sparse_shared": 3}
    local_hidden = config.geometry.hidden_size // 4
    state_specs = decoder.ws32_decoder_state_specs()
    weight_specs = decoder.ws32_decoder_weight_specs(config)

    def layer_fn(layer_id: int, challenger: bool):
        slot = config.full_index_slot_by_layer[layer_id]
        layer_specs = weight_specs.layers[layer_id]
        common = dict(
            indexer_kind=config.geometry.indexer_types[layer_id],
            mlp_kind=config.geometry.mlp_layer_types[layer_id],
            dsa_contract=config.dsa_contract,
            attention_contract=config.attention_contract,
            moe_contract=config.moe_contract,
            cache_layout=config.cache_layout,
            block_shape=config.geometry.fp8_block_shape,
            rms_norm_epsilon=config.rms_norm_epsilon,
            sparse_attention_config=SparseMlaConfig(segment_block=config.sparse_segment_block),
            **interpret,
        )

        def body(update, residual, st, layer, rope_row):
            kwargs = dict(common)
            if challenger:
                kwargs["options"] = Ws32PerfOptions(sampler="greedy", routed_projection=tiles)
                fn = ws32_transformer_layer_challenger_mapped
            else:
                fn = ws32_transformer_layer_mapped
            return fn(
                update, residual, st.kv_cache_local[layer_id],
                st.index_cache_local[0 if slot is None else slot],
                st.selected_positions, st.selected_valid_counts, st.selected_scores,
                st.position, st.block_tables, st.context_lengths,
                layer.qkv_a, layer.attention, layer.dsa,
                layer.post_attention_norm_weight_local, layer.dense, layer.moe,
                st.contract_valid, main_rope_table_row=rope_row, **kwargs,
            ).output_local

        return jax.shard_map(
            body, mesh=mesh,
            in_specs=(P(None, "feature"), P(None, "feature"), state_specs, layer_specs, P()),
            out_specs=P(None, "feature"), check_vma=False,
        )

    update = jax.device_put(
        jnp.zeros((1, config.geometry.hidden_size), jnp.bfloat16),
        NamedSharding(mesh, P(None, "feature")),
    )
    rope_row = jnp.zeros((config.geometry.qk_rope_head_dim,), jnp.bfloat16)
    per_layer: dict[str, dict[str, dict[str, int]]] = {}
    for kind, layer_id in layer_ids.items():
        per_layer[kind] = {}
        for label, challenger in (("frozen", False), ("challenger", True)):
            per_layer[kind][label] = jaxpr_census(
                layer_fn(layer_id, challenger), update, update, state,
                weights.layers[layer_id], rope_row,
            )

    def extrapolate(label: str) -> dict[str, int]:
        totals: Counter = Counter()
        for kind, count in GLM_LAYER_KINDS.items():
            for key, value in per_layer[kind][label].items():
                totals[key] += count * value
        return {key: int(totals[key]) for key in sorted(totals)}

    report: dict[str, Any] = dict(
        schema="glm_perf_op_census_v1",
        fixture="tests/greenfield/runtime/ws32_prefill_cpu_fixture.py (8 layers, 32 CPU devices)",
        glm_layer_kinds=GLM_LAYER_KINDS,
        per_layer_jaxpr=per_layer,
        extrapolated_78_layer_jaxpr={
            label: extrapolate(label) for label in ("frozen", "challenger")
        },
        notes=[
            "jaxpr counts are static maxima: cond branches are counted; *_inside_cond is the subset under lax.cond",
            "extrapolation multiplies per-layer-kind counts by GLM-5.2's layer census and excludes embedding/head",
            "hlo counts come from the compiled 32-device CPU executable; Pallas kernels are expanded by interpret mode",
        ],
    )

    # ---- whole-step census: frozen sampled/greedy heads vs challenger variants.
    programs = {
        "frozen_greedy": (
            jax.jit(decoder.build_ws32_decoder_program(mesh, config, **interpret).execute),
            (token, state, weights, rope),
        ),
        "frozen_nucleus": (
            build_ws32_sampled_decoder_program(mesh, config, sampling=sampling, **interpret).execute,
            (token, state, weights, rope, uniform),
        ),
        "challenger_grouped_nucleus": (
            build_ws32_challenger_decoder_program(
                mesh, config, options=Ws32PerfOptions(sampler="nucleus", routed_projection=tiles),
                sampling=sampling, **interpret,
            ).execute,
            (token, state, weights, rope, uniform),
        ),
        "challenger_grouped_candidates": (
            build_ws32_challenger_decoder_program(
                mesh, config,
                options=Ws32PerfOptions(sampler="nucleus_candidates", candidates_per_shard=8,
                                        routed_projection=tiles),
                sampling=sampling, **interpret,
            ).execute,
            (token, state, weights, rope, uniform),
        ),
    }
    whole: dict[str, Any] = {}
    for name, (fn, fn_args) in programs.items():
        entry = {"jaxpr": jaxpr_census(fn, *fn_args)}
        if not args.skip_compile:
            entry["hlo"] = hlo_census(fn.lower(*fn_args).compile().as_text())
        whole[name] = entry
    report["whole_step_8_layer_fixture"] = whole
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
