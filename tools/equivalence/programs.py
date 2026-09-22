"""v0 adapter: the exact device programs production compiles, as lowerable (fn, args) pairs.

At 181c013e production has no program builder: ``OrdinaryRuntime._load``
(``glm_tpu/optimized/runtime.py:129-176``) builds and compiles its programs inline and
``glm_tpu/optimized/batched_runtime.compile_batch`` adds the concurrent ones. ``load_programs``
below replicates ``_load`` call for call (same builder functions, options, sample shapes,
shardings and donation rule); the batch programs come from calling the real ``compile_batch``
with a recording runtime; the resident FP8 table decoders are captured by wrapping
``bf16_resident._decode_program`` while the real ``bf16_resident_weights`` runs. S2c replaces the
internals of this module with ``glm_tpu.runner.programs.build_program_set`` and proves G1 and G2
unchanged.

Two modes share one code path:

* ``concrete`` (fixture tier only): real fixture arrays on the 32-device CPU mesh; WK, FP8-table
  and cache-initializer programs execute on CPU exactly as ``_load`` executes them on TPU.
* ``abstract`` (both tiers): ``ShapeDtypeStruct`` values carrying ``NamedSharding``; a producer
  program's outputs take the shardings its CPU-compiled executable reports (never executed).

``adapter_consistency`` asserts on the fixture that both modes hand every program identical
arguments (shape, dtype, sharding), which is what licenses the abstract production tier.

Run as ``python -m tools.equivalence.programs --tier fixture|production`` (JAX_PLATFORMS=cpu,
32 forced CPU devices); prints one JSON line of fingerprints.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Any, Callable, Iterator

import numpy as np

from .common import canonical_json, emit, environment, require_cpu, sha256_hex, source_record

# Production constants at 181c013e (glm_tpu/optimized/request.py, runtime.py).
DONATION_THRESHOLD = 8192  # request.CAPACITY: donate when capacity > this
CACHE_INIT_SAMPLE = 2034   # runtime.py:154 sample prompt length
PREFILL_OPTIONS = dict(key_tile=512, mlp_window=True, rolled_prefix=True, expert_panels=True,
                       paired_position_sort=True, sorted_local_merge=True, canonical_dense=True)
PRODUCTION_CAPACITIES = (8192, 32768, 166912)
BATCH_CAPACITY, BATCH_SIZE = 32768, 4


@dataclass(frozen=True)
class ProgramSpec:
    name: str                 # production program name (HLO original file name)
    fn: Callable[..., Any]    # the jax.jit object production lowers (donation already applied)
    args: tuple[Any, ...]     # arguments production passes (arrays or ShapeDtypeStructs)


def _leaf_signature(leaf: Any) -> tuple[Any, ...]:
    return (tuple(int(d) for d in leaf.shape), str(leaf.dtype), getattr(leaf, "sharding", None))


def abstract_like(tree: Any) -> Any:
    import jax

    return jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), tree)


class Recorder:
    """Stand-in for ``OrdinaryRuntime``: exactly the attributes ``_load`` and ``compile_batch``
    read, plus a ``compile`` that records ``(name, fn, values)`` instead of compiling for TPU."""

    def __init__(self, mesh: Any, config: Any, *, donating: bool, concurrent_size: int, concrete: bool):
        self.mesh, self.config = mesh, config
        self.capacity = config.context_capacity
        self.donating = donating
        self.concurrent_size = concurrent_size
        self.concrete = concrete
        self.specs: list[ProgramSpec] = []
        self._outputs: dict[Any, Any] = {}

    def put(self, value: Any) -> Any:
        import jax
        from jax.sharding import NamedSharding, PartitionSpec as P

        sharding = NamedSharding(self.mesh, P())
        if self.concrete:
            return jax.device_put(value, sharding)
        array = np.asarray(value)
        return jax.ShapeDtypeStruct(array.shape, array.dtype, sharding=sharding)

    def compile(self, name: str, fn: Any, values: tuple[Any, ...], *, model: bool = True) -> Any:
        self.specs.append(ProgramSpec(name, fn, tuple(values)))
        return fn if self.concrete else self.abstract(fn)

    def abstract(self, fn: Any) -> Callable[..., Any]:
        """Outputs as ShapeDtypeStructs with the shardings the (CPU-)compiled executable reports."""
        import jax

        def call(*args: Any) -> Any:
            leaves, treedef = jax.tree.flatten(args)
            key = (id(fn), treedef, tuple(_leaf_signature(x) for x in leaves))
            if key not in self._outputs:
                compiled = fn.trace(*args).lower().compile()
                infos = jax.tree.leaves(compiled.out_info)
                shardings = jax.tree.leaves(compiled.output_shardings)
                out = [jax.ShapeDtypeStruct(i.shape, i.dtype, sharding=s)
                       for i, s in zip(infos, shardings, strict=True)]
                self._outputs[key] = (fn, jax.tree.unflatten(jax.tree.structure(compiled.out_info), out))
            return self._outputs[key][1]

        return call


def fp8_table_name(key: tuple[Any, ...]) -> str:
    bits, scale, spec, block = key
    render = ",".join("None" if axis is None else str(axis) for axis in spec)
    return (f"fp8_table[{'x'.join(map(str, bits))}/{'x'.join(map(str, scale))}/({render})"
            f"/{'x'.join(map(str, block))}]")


@contextmanager
def _recording_tables(r: Recorder) -> Iterator[None]:
    """Capture every per-table decoder ``bf16_resident_weights`` calls, in call order."""
    from glm_tpu.optimized import bf16_resident

    original = bf16_resident._decode_program
    saved = dict(bf16_resident._DECODERS)
    bf16_resident._DECODERS.clear()  # a fresh production process starts with an empty cache
    seen: set[Any] = set()

    def recording(mesh: Any, bits: Any, scale: Any, spec: Any, block: tuple[int, int]) -> Any:
        program = original(mesh, bits, scale, spec, block)
        key = (tuple(bits.shape), tuple(scale.shape), tuple(spec), tuple(block))
        if key not in seen:
            seen.add(key)
            r.specs.append(ProgramSpec(fp8_table_name(key), program, (bits, scale)))
        return program if r.concrete else r.abstract(program)

    bf16_resident._decode_program = recording
    try:
        yield
    finally:
        bf16_resident._decode_program = original
        bf16_resident._DECODERS.clear()
        bf16_resident._DECODERS.update(saved)


def load_programs(r: Recorder, raw: Any) -> None:
    """``OrdinaryRuntime._load`` from the bound checkpoint on, call for call (runtime.py:129-176
    at 181c013e). Only checkpoint I/O, admission, votes and records are left out."""
    import jax
    from jax.sharding import PartitionSpec as P

    from glm_tpu.greenfield.runtime import ws32_decoder as dec
    from glm_tpu.optimized.bf16_resident import bf16_resident_weights
    from glm_tpu.optimized.prefill_challenger import build_ws32_prefill_challenger_program
    from glm_tpu.optimized.request_loop import build_packed_decoder_program
    from scripts.greenfield.ws32_compile_originals import build_wk_programs
    from scripts.greenfield.ws32_native_benchmark_programs import build_cache_initializer

    config = r.config
    decode, promote = build_wk_programs(r.mesh, P(None, "feature"), P(None, "feature"), contract=config.dsa_contract)
    first = raw.layers[config.full_index_slots[0]].dsa
    decode = r.compile("wk_decode", decode, (first.wk_bits_local, first.wk_scale_local), model=False)
    completed = decode(first.wk_bits_local, first.wk_scale_local)
    promote = r.compile("wk_promote", promote, (completed,), model=False)
    wk = []
    for layer_id in config.full_index_slots:
        d = raw.layers[layer_id].dsa
        completed = decode(d.wk_bits_local, d.wk_scale_local)
        wk.append(promote(completed))
    r.wk = tuple(wk)
    with _recording_tables(r):
        r.weights = bf16_resident_weights(r.mesh, config, raw)
    r.rope = r.put(np.asarray(dec.build_ws32_main_rope_table(config)))
    r.initialize = r.compile("cache_init", build_cache_initializer(r.mesh, config),
                             (r.put(np.int32(CACHE_INIT_SAMPLE)),), model=False)
    initial = r.initialize(r.put(np.int32(CACHE_INIT_SAMPLE)))
    r.prefill = {}
    for rows in (128, 114):
        fn = build_ws32_prefill_challenger_program(r.mesh, config, block_rows=rows, **PREFILL_OPTIONS).execute
        if r.donating:
            fn = jax.jit(fn, donate_argnums=(2,))
        r.prefill[rows] = r.compile("prefill_" + str(rows), fn,
                                    (r.put(np.zeros(rows, np.int32)), r.put(np.int32(rows)), initial,
                                     r.weights, r.wk, r.rope))
    if r.concurrent_size:
        from glm_tpu.optimized.batched_runtime import compile_batch

        compile_batch(r, initial)
    else:
        decode_fn = build_packed_decoder_program(r.mesh, config).execute
        if r.donating:
            decode_fn = jax.jit(decode_fn, donate_argnums=(1,))
        r.decode = r.compile("decode", decode_fn, (r.put(np.array([0], np.int32)), initial.decoder, r.weights, r.rope))


# ----------------------------------------------------------------------------- tiers
GEOMETRY_ONLY = ("wk_decode", "wk_promote")
DONATION_DEPENDENT = ("prefill_128", "prefill_114", "decode")


def program_key(name: str, *, capacity: int, donating: bool, concurrent_size: int) -> str:
    """Stable record key: geometry-only programs by name; the rest qualified by capacity,
    donated variant and batch size."""
    if name in GEOMETRY_ONLY or name.startswith("fp8_table["):
        return name
    key = f"{name}@{capacity}"
    if donating and name in DONATION_DEPENDENT:
        key += "+donated"
    if name.startswith("batch_"):
        key += f"#n{concurrent_size}"
    return key


def fixture_raw(mesh: Any, frozen: Any, *, concrete: bool) -> Any:
    import jax
    from jax.sharding import NamedSharding

    from glm_tpu.greenfield.runtime.ws32_decoder import bind_ws32_decoder_weights

    from . import fixture

    if concrete:
        return fixture.bind(mesh, frozen)
    arrays = {name: jax.ShapeDtypeStruct(frozen.arrays[name].shape, frozen.arrays[name].dtype,
                                         sharding=NamedSharding(mesh, spec))
              for name, spec in fixture.name_spec_pairs(frozen.config)}
    return bind_ws32_decoder_weights(arrays, frozen.config)


def production_raw(mesh: Any, config: Any, plans: Any) -> Any:
    """Abstract checkpoint arrays exactly as ``load_ws32_runtime_checkpoint`` builds them:
    global shape, dtype and ``NamedSharding(mesh, P(*partition_spec))`` of every tensor plan."""
    import jax
    import ml_dtypes
    from jax.sharding import NamedSharding, PartitionSpec as P

    from glm_tpu.greenfield.runtime.ws32_decoder import bind_ws32_decoder_weights

    dtypes = {"U8": np.dtype(np.uint8), "F32": np.dtype(np.float32), "BF16": np.dtype(ml_dtypes.bfloat16)}
    arrays = {t.name: jax.ShapeDtypeStruct(tuple(t.global_shape), dtypes[t.dtype],
                                           sharding=NamedSharding(mesh, P(*t.partition_spec)))
              for t in plans[0].tensors}
    return bind_ws32_decoder_weights(arrays, config)


def variants(tier: str) -> list[dict[str, Any]]:
    """The production runs a tier covers (capacity, donation, concurrent size)."""
    if tier == "fixture":
        from .fixture import CAPACITY

        return [dict(capacity=CAPACITY, donating=False, concurrent_size=0),
                dict(capacity=CAPACITY, donating=True, concurrent_size=0),
                dict(capacity=CAPACITY, donating=False, concurrent_size=BATCH_SIZE)]
    if tier == "production":
        runs = [dict(capacity=c, donating=c > DONATION_THRESHOLD, concurrent_size=0) for c in PRODUCTION_CAPACITIES]
        runs.append(dict(capacity=BATCH_CAPACITY, donating=True, concurrent_size=BATCH_SIZE))
        return runs
    raise ValueError("tier must be fixture or production")


def program_specs(tier: str, mesh: Any, *, concrete: bool = False,
                  only: set[str] | None = None) -> dict[str, ProgramSpec]:
    """Ordered ``{record key: ProgramSpec}`` for a tier (duplicates across runs are skipped)."""
    from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderConfig

    out: dict[str, ProgramSpec] = {}
    if tier == "fixture":
        from . import fixture

        frozen = fixture.fixture_v1(panel_geometry=True)
        base_config = frozen.config
        raw = fixture_raw(mesh, frozen, concrete=concrete)
    else:
        if concrete:
            raise ValueError("the production tier is abstract only")
        from glm_tpu.optimized import model

        from .identities import synthetic_file_plans

        geometry = model.geometry()
        base_config = Ws32DecoderConfig(geometry, PRODUCTION_CAPACITIES[0], host_main_rope_table=True)
        raw = production_raw(mesh, base_config, synthetic_file_plans()[1])
    for run in variants(tier):
        if tier == "fixture":
            config = base_config
        else:  # OrdinaryRuntime.__init__: Ws32DecoderConfig(model.geometry(repo), capacity, host_main_rope_table=True)
            config = Ws32DecoderConfig(base_config.geometry, run["capacity"], host_main_rope_table=True)
        recorder = Recorder(mesh, config, donating=run["donating"], concurrent_size=run["concurrent_size"],
                            concrete=concrete)
        load_programs(recorder, raw)
        for spec in recorder.specs:
            key = program_key(spec.name, capacity=run["capacity"], donating=run["donating"],
                              concurrent_size=run["concurrent_size"])
            if key not in out and (only is None or key in only):
                out[key] = spec
    return out


def adapter_consistency(mesh: Any) -> dict[str, Any]:
    """Fixture tier: the abstract adapter hands every program the same (shape, dtype, sharding)
    arguments as the concrete one, which executes the producer programs exactly like ``_load``."""
    import jax

    concrete = program_specs("fixture", mesh, concrete=True)
    abstract = program_specs("fixture", mesh, concrete=False)
    mismatches = []
    if list(concrete) != list(abstract):
        mismatches.append("program list")
    for key in concrete:
        if key not in abstract:
            continue
        left, ltree = jax.tree.flatten(abstract_like(concrete[key].args))
        right, rtree = jax.tree.flatten(abstract[key].args)
        if ltree != rtree or [_leaf_signature(x) for x in left] != [_leaf_signature(x) for x in right]:
            mismatches.append(key)
    return dict(programs=len(concrete), identical=not mismatches, mismatches=mismatches)


def fingerprint_specs(specs: dict[str, ProgramSpec], *, summary: bool = True) -> dict[str, Any]:
    from . import lowering, normalize

    records: dict[str, Any] = {}
    for key, spec in specs.items():
        started = time.perf_counter()
        with lowering.location_free():
            lowered = lowering.lower_for_tpu(spec.fn, spec.args)
            record = normalize.fingerprint(lowered, spec.args, summary=summary)
        record["seconds"] = round(time.perf_counter() - started, 1)
        records[key] = record
        del lowered
    return records


def tier_digest(records: dict[str, Any]) -> str:
    return sha256_hex(canonical_json({k: [v["digest"], v["signature_digest"]] for k, v in sorted(records.items())}))


def run_tier(tier: str, *, consistency: bool = False, only: set[str] | None = None) -> dict[str, Any]:
    from . import fixture, lowering

    require_cpu()
    mesh = fixture.cpu_mesh()
    started = time.perf_counter()
    with lowering.tpu_v4_info():
        specs = program_specs(tier, mesh, only=only)
    built = time.perf_counter() - started
    records = fingerprint_specs(specs)
    result = dict(tier=tier, environment=environment(), source=source_record(), programs=records,
                  tier_digest=tier_digest(records), build_seconds=round(built, 1),
                  seconds=round(time.perf_counter() - started, 1))
    if consistency:
        with lowering.tpu_v4_info():
            result["adapter_consistency"] = adapter_consistency(mesh)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tier", choices=("fixture", "production"), required=True)
    parser.add_argument("--consistency", action="store_true", help="also run the fixture adapter-consistency check")
    parser.add_argument("--only", action="append", help="restrict to these record keys")
    parser.add_argument("--text", action="store_true", help="emit the normalized text of the single --only program")
    args = parser.parse_args(argv)
    if args.text:
        emit(normalized_text(args.tier, args.only))
        return 0
    emit(run_tier(args.tier, consistency=args.consistency, only=set(args.only) if args.only else None))
    return 0


def normalized_text(tier: str, only: list[str] | None) -> dict[str, Any]:
    """Diagnosis (``python -m tools.equivalence diff``): N3-N6 text of exactly one program."""
    from . import fixture, lowering, normalize

    if not only or len(only) != 1:
        raise SystemExit("--text needs exactly one --only program key")
    require_cpu()
    mesh = fixture.cpu_mesh()
    with lowering.tpu_v4_info():
        spec = program_specs(tier, mesh, only=set(only))[only[0]]
    with lowering.location_free():
        text = normalize.stablehlo_text(lowering.lower_for_tpu(spec.fn, spec.args))
    return dict(program=only[0], text=normalize.normalize(text)[0])


if __name__ == "__main__":
    raise SystemExit(main())
