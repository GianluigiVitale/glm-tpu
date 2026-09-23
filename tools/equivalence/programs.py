"""Device programs of the tree under test, as lowerable ``(fn, args)`` pairs built by production.

``program_specs`` constructs the real ``OrdinaryRuntime`` with its real ``__init__`` (which runs
the real ``_load`` and, for a concurrent runtime, the real ``batched_runtime.compile_batch``)
through ``driver.build_runtime``. ``_load``'s ``compile`` calls run the real ``compile`` and
``compile_program``; the fingerprint of a compiled program is taken from the ``Lowered`` that
``compile_program`` itself built (``fn.lower(*inputs)``, for the TPU platform, location-free) and
records the arguments production passed to ``Lowered.compile``. The per-table FP8 decoders
``bf16_resident_weights`` builds (compiled implicitly by jit dispatch) are lowered by the harness
from the recorded ``(fn, args)``. Only checkpoint I/O, fleet votes, device memory statistics, the
TPU compiler itself and the HLO directory are faked (``driver.py``).

The **v0 adapter** (``load_programs_v0``, below) is the S0 replica of ``_load`` with frozen
copies of its constants. It is kept only as a cross-check (``--adapter v0``): ``gates`` requires
its fingerprints to equal the real runtime's while it can still be built (it imports 181c013e
helper homes; S2a/S2c retire it).

Two modes share one code path:

* ``concrete`` (fixture tier only): real fixture arrays on the 32-device CPU mesh; WK, FP8-table
  and cache-initializer programs execute on CPU exactly as ``_load`` executes them on TPU.
* ``abstract`` (both tiers): ``ShapeDtypeStruct`` values carrying ``NamedSharding``; a producer
  program's outputs take the shardings its CPU-compiled executable reports (never executed).

``adapter_consistency`` asserts on the fixture that both modes hand every program identical
arguments (shape, dtype, sharding), compile identical programs (production's own lowering: digest,
signature and ``Lowered.compile`` arguments) and follow an identical load protocol, which is what
licenses the abstract production tier.

Run as ``python -m tools.equivalence.programs --tier fixture|production [--adapter v0]``
(JAX_PLATFORMS=cpu, 32 forced CPU devices); prints one JSON line of fingerprints.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from .common import canonical_json, emit, environment, require_cpu, sha256_hex, source_record
from .driver import ProgramRecorder, ProgramSpec, abstract_like, leaf_signature, recording_tables

# v0 adapter only: frozen copies of the 181c013e constants (glm_tpu/optimized/request.py, runtime.py).
DONATION_THRESHOLD = 8192  # request.CAPACITY: donate when capacity > this
CACHE_INIT_SAMPLE = 2034   # runtime.py:154 sample prompt length
PREFILL_OPTIONS = dict(key_tile=512, mlp_window=True, rolled_prefix=True, expert_panels=True,
                       paired_position_sort=True, sorted_local_merge=True, canonical_dense=True)
PRODUCTION_CAPACITIES = (8192, 32768, 166912)
BATCH_CAPACITY, BATCH_SIZE = 32768, 4
FIXTURE_BATCH_SIZES = (1, 2, 3, 4)  # worker: concurrent_size=len(pending), request.batch allows 1..4


class Recorder(ProgramRecorder):
    """v0 stand-in for ``OrdinaryRuntime``: the attributes ``load_programs_v0`` and
    ``compile_batch`` read, plus the recording ``compile``."""

    def __init__(self, mesh: Any, config: Any, *, donating: bool, concurrent_size: int, concrete: bool,
                 outputs: dict[Any, Any] | None = None):
        super().__init__(concrete=concrete, outputs=outputs)
        self.mesh, self.config = mesh, config
        self.context = config
        self.capacity = config.context_capacity
        self.donating = donating
        self.concurrent_size = concurrent_size

    def put(self, value: Any) -> Any:
        import jax
        from jax.sharding import NamedSharding, PartitionSpec as P

        sharding = NamedSharding(self.mesh, P())
        if self.concrete:
            return jax.device_put(value, sharding)
        array = np.asarray(value)
        return jax.ShapeDtypeStruct(array.shape, array.dtype, sharding=sharding)


def load_programs_v0(r: Recorder, raw: Any) -> None:
    """v0 adapter (cross-check only): ``OrdinaryRuntime._load`` from the bound checkpoint on, call
    for call as of 181c013e (runtime.py:129-176). Checkpoint I/O, admission, votes and records are
    left out."""
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
    with recording_tables(r):
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
def program_key(name: str, run: dict[str, Any]) -> str:
    """Record key: the program name qualified by the run that built it (``variant_key``: capacity,
    ``+donated``, ``#n<lanes>``). Every run's programs are fingerprinted under their own key, so a
    change confined to the donated or to a concurrent runtime (an option, a donation, a sample
    shape) cannot hide behind an equal program recorded from another run."""
    return f"{name}@{variant_key(run)}"


def fixture_arrays(mesh: Any, frozen: Any, *, concrete: bool) -> dict[str, Any]:
    """``{checkpoint tensor name: array}`` as the loader would hand them to ``_load``: placed with
    the production partition specs (concrete) or as ``ShapeDtypeStruct`` with those shardings."""
    import jax
    from jax.sharding import NamedSharding

    from . import fixture

    pairs = fixture.name_spec_pairs(frozen.config)
    if {name for name, _ in pairs} != set(frozen.arrays):
        raise ValueError("frozen fixture tensor set differs from the production name tree")
    if concrete:
        return {name: jax.device_put(frozen.arrays[name], NamedSharding(mesh, spec)) for name, spec in pairs}
    return {name: jax.ShapeDtypeStruct(frozen.arrays[name].shape, frozen.arrays[name].dtype,
                                       sharding=NamedSharding(mesh, spec)) for name, spec in pairs}


def production_arrays(mesh: Any, plans: Any) -> dict[str, Any]:
    """Abstract checkpoint arrays exactly as ``load_ws32_runtime_checkpoint`` builds them:
    global shape, dtype and ``NamedSharding(mesh, P(*partition_spec))`` of every tensor plan."""
    import jax
    import ml_dtypes
    from jax.sharding import NamedSharding, PartitionSpec as P

    dtypes = {"U8": np.dtype(np.uint8), "F32": np.dtype(np.float32), "BF16": np.dtype(ml_dtypes.bfloat16)}
    return {t.name: jax.ShapeDtypeStruct(tuple(t.global_shape), dtypes[t.dtype],
                                         sharding=NamedSharding(mesh, P(*t.partition_spec)))
            for t in plans[0].tensors}


def variants(tier: str) -> list[dict[str, Any]]:
    """The production runs a tier covers (capacity, donation expected by the 181c013e rule,
    concurrent size). The fixture's donated run lowers the donation threshold (driver.py)."""
    if tier == "fixture":
        from .fixture import CAPACITY

        return ([dict(capacity=CAPACITY, donating=False, concurrent_size=0),
                 dict(capacity=CAPACITY, donating=True, concurrent_size=0)]
                + [dict(capacity=CAPACITY, donating=False, concurrent_size=n) for n in FIXTURE_BATCH_SIZES])
    if tier == "production":
        runs = [dict(capacity=c, donating=c > DONATION_THRESHOLD, concurrent_size=0) for c in PRODUCTION_CAPACITIES]
        runs.append(dict(capacity=BATCH_CAPACITY, donating=True, concurrent_size=BATCH_SIZE))
        return runs
    raise ValueError("tier must be fixture or production")


def variant_key(run: dict[str, Any]) -> str:
    return f"{run['capacity']}" + ("+donated" if run["donating"] else "") + (
        f"#n{run['concurrent_size']}" if run["concurrent_size"] else "")


class TierPrograms(dict):
    """Ordered ``{record key: ProgramSpec}`` plus the recorded load protocol of every run."""

    protocol: dict[str, Any]
    defaults: dict[str, Any] | None


def program_specs(tier: str, mesh: Any, *, concrete: bool = False, only: set[str] | None = None,
                  adapter: str = "runtime", fingerprint: bool = False, keep_lowered: bool = False) -> TierPrograms:
    """Programs of every run of a tier, keyed per run (``program_key``); ``only`` filters keys.
    ``fingerprint``: fingerprint the ``Lowered`` production compiled for each program;
    ``keep_lowered``: keep that ``Lowered`` for the keys in ``only`` (``diff``, ``authenticity``)."""
    if tier == "production" and concrete:
        raise ValueError("the production tier is abstract only")
    if adapter == "v0":
        return _program_specs_v0(tier, mesh, concrete=concrete, only=only)
    if adapter != "runtime":
        raise ValueError("adapter must be runtime or v0")
    from . import driver

    out = TierPrograms()
    out.protocol = {}
    outputs: dict[Any, Any] = {}
    if tier == "fixture":
        from . import fixture

        frozen = fixture.fixture_v1(panel_geometry=True)
        arrays = fixture_arrays(mesh, frozen, concrete=concrete)
        plans, geometry = driver.fixture_plans(arrays), frozen.config.geometry
    else:
        from .identities import synthetic_file_plans

        plans = synthetic_file_plans()[1]
        arrays, geometry = production_arrays(mesh, plans), None
    for run in variants(tier):
        keep = None
        if keep_lowered and only:
            def keep(name: str, run: dict[str, Any] = run) -> bool:
                return program_key(name, run) in only
        built = driver.build_runtime(
            mesh, tier=tier, capacity=run["capacity"], concurrent_size=run["concurrent_size"], arrays=arrays,
            plans=plans, concrete=concrete, donated_fixture=tier == "fixture" and run["donating"],
            fixture_geometry=geometry, outputs=outputs, fingerprint=fingerprint, keep=keep)
        out.protocol[variant_key(run)] = built.protocol
        _add_run(out, built.recorder.specs, run, only)
        del built
    out.defaults = driver.production_defaults()
    return out


def _program_specs_v0(tier: str, mesh: Any, *, concrete: bool, only: set[str] | None) -> TierPrograms:
    from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderConfig, bind_ws32_decoder_weights

    out = TierPrograms()
    out.protocol, out.defaults = {}, None
    outputs: dict[Any, Any] = {}
    if tier == "fixture":
        from . import fixture

        frozen = fixture.fixture_v1(panel_geometry=True)
        base_config = frozen.config
        raw = bind_ws32_decoder_weights(fixture_arrays(mesh, frozen, concrete=concrete), frozen.config)
    else:
        from glm_tpu.optimized import model

        from .identities import synthetic_file_plans

        base_config = Ws32DecoderConfig(model.geometry(), PRODUCTION_CAPACITIES[0], host_main_rope_table=True)
        raw = bind_ws32_decoder_weights(production_arrays(mesh, synthetic_file_plans()[1]), base_config)
    for run in variants(tier):
        if tier == "fixture":
            config = base_config
        else:  # OrdinaryRuntime.__init__: Ws32DecoderConfig(model.geometry(repo), capacity, host_main_rope_table=True)
            config = Ws32DecoderConfig(base_config.geometry, run["capacity"], host_main_rope_table=True)
        recorder = Recorder(mesh, config, donating=run["donating"], concurrent_size=run["concurrent_size"],
                            concrete=concrete, outputs=outputs)
        load_programs_v0(recorder, raw)
        _add_run(out, recorder.specs, run, only)
    return out


def _add_run(out: dict[str, ProgramSpec], specs: list[ProgramSpec], run: dict[str, Any],
             only: set[str] | None) -> None:
    seen: set[str] = set()
    for spec in specs:
        key = program_key(spec.name, run)
        if key in seen:
            raise RuntimeError(f"run {variant_key(run)} built {spec.name} more than once")
        seen.add(key)
        if only is None or key in only:
            out[key] = spec


def adapter_consistency(mesh: Any, abstract: TierPrograms | None = None) -> dict[str, Any]:
    """Fixture tier, real runtime: the abstract run hands every program the same (shape, dtype,
    sharding) arguments, compiles the same programs (production's lowering digest and signature,
    ``Lowered.compile`` arguments) and makes the same load/admission calls as the concrete run,
    which executes the producer programs exactly like ``_load``. ``abstract``: an abstract build
    with fingerprints, reused when the caller already has one."""
    import jax

    concrete = program_specs("fixture", mesh, concrete=True, fingerprint=True)
    abstract = abstract if abstract is not None else program_specs("fixture", mesh, fingerprint=True)
    mismatches = []
    if list(concrete) != list(abstract):
        mismatches.append("program list")

    def compiled(spec: ProgramSpec) -> Any:
        record = spec.record or {}
        return record.get("digest"), record.get("signature_digest"), spec.compile_calls

    for key in concrete:
        if key not in abstract:
            continue
        left, ltree = jax.tree.flatten(abstract_like(concrete[key].args))
        right, rtree = jax.tree.flatten(abstract_like(abstract[key].args))
        if (ltree != rtree or [leaf_signature(x) for x in left] != [leaf_signature(x) for x in right]
                or compiled(concrete[key]) != compiled(abstract[key])):
            mismatches.append(key)
    if concrete.protocol != abstract.protocol:
        mismatches.append("load protocol")
    return dict(programs=len(concrete), identical=not mismatches, mismatches=mismatches)


def fingerprint_specs(specs: dict[str, ProgramSpec], *, summary: bool = True) -> dict[str, Any]:
    """Fingerprint every spec: a program production compiled keeps the fingerprint of the
    ``Lowered`` production built (``spec.record``); any other spec (FP8 tables, v0 adapter,
    self-test variants) is lowered here from its ``(fn, args)``. ``compile`` records the
    ``Lowered.compile`` arguments (None: compiled by jit dispatch). A program equal (digest and
    signature) to one recorded earlier in the tier keeps its own digests but refers to that record
    for the diagnostic summary (``same_as``), which keeps the per-run records compact."""
    from . import lowering, normalize

    records: dict[str, Any] = {}
    first: dict[tuple[str, str], str] = {}
    for key, spec in specs.items():
        if spec.record is not None:
            record = dict(spec.record)
            if not summary:
                record.pop("summary", None)
        else:
            started = time.perf_counter()
            with lowering.location_free():
                lowered = lowering.lower_for_tpu(spec.fn, spec.args)
                record = normalize.fingerprint(lowered, spec.args, summary=summary)
            record["seconds"] = round(time.perf_counter() - started, 1)
            del lowered
        record["compile"] = spec.compile_calls
        pair = (record["digest"], record["signature_digest"])
        if pair in first:
            record.pop("summary", None)
            record["same_as"] = first[pair]
        else:
            first[pair] = key
        records[key] = record
    return records


def tier_digest(records: dict[str, Any]) -> str:
    return sha256_hex(canonical_json({k: [v["digest"], v["signature_digest"]] for k, v in sorted(records.items())}))


def run_tier(tier: str, *, consistency: bool = False, only: set[str] | None = None,
             adapter: str = "runtime") -> dict[str, Any]:
    from . import fixture, lowering

    require_cpu()
    mesh = fixture.cpu_mesh()
    started = time.perf_counter()
    with lowering.tpu_v4_info():
        specs = program_specs(tier, mesh, only=only, adapter=adapter, fingerprint=adapter == "runtime")
    built = time.perf_counter() - started
    records = fingerprint_specs(specs)
    result = dict(tier=tier, adapter=adapter, environment=environment(), source=source_record(), programs=records,
                  tier_digest=tier_digest(records), runtime=specs.protocol, build_seconds=round(built, 1),
                  seconds=round(time.perf_counter() - started, 1))
    if specs.defaults is not None:
        result["defaults"] = specs.defaults
    if consistency:
        with lowering.tpu_v4_info():
            result["adapter_consistency"] = adapter_consistency(mesh, abstract=specs if only is None else None)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tier", choices=("fixture", "production"), required=True)
    parser.add_argument("--adapter", choices=("runtime", "v0"), default="runtime",
                        help="runtime: the real OrdinaryRuntime (the gate); v0: the S0 replica (cross-check)")
    parser.add_argument("--consistency", action="store_true", help="also run the fixture adapter-consistency check")
    parser.add_argument("--only", action="append", help="restrict to these record keys")
    parser.add_argument("--text", action="store_true", help="emit the normalized text of the single --only program")
    args = parser.parse_args(argv)
    if args.text:
        emit(normalized_text(args.tier, args.only, adapter=args.adapter))
        return 0
    emit(run_tier(args.tier, consistency=args.consistency, only=set(args.only) if args.only else None,
                  adapter=args.adapter))
    return 0


def normalized_text(tier: str, only: list[str] | None, *, adapter: str = "runtime") -> dict[str, Any]:
    """Diagnosis (``python -m tools.equivalence diff``): N3-N6 text of exactly one program."""
    from . import fixture, lowering, normalize

    if not only or len(only) != 1:
        raise SystemExit("--text needs exactly one --only program key")
    require_cpu()
    mesh = fixture.cpu_mesh()
    with lowering.tpu_v4_info():
        spec = program_specs(tier, mesh, only=set(only), adapter=adapter, keep_lowered=True)[only[0]]
    with lowering.location_free():
        lowered = spec.lowered if spec.lowered is not None else lowering.lower_for_tpu(spec.fn, spec.args)
        text = normalize.stablehlo_text(lowered)
    return dict(program=only[0], text=normalize.normalize(text)[0])


if __name__ == "__main__":
    raise SystemExit(main())
