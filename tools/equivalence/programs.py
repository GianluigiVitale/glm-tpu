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

Since S2c every program the runtime compiles comes from ``glm_tpu.runner.programs.build_program_set``.
The real build records which ``ProgramSet`` the runtime built and checks that it compiled exactly
those function objects (``driver.programset_identity``); the **ProgramSet cross-check**
(``--adapter programset``, run by ``gates`` in a parallel child) builds the set standalone for
every run's config and lowers each spec with the arguments the runtime passed: every program must
fingerprint exactly like the one the runtime compiled, and each spec's declared donation must be
what its lowering donates. (It replaced the S0 ``v0`` replica of ``_load``.)

Two modes share one code path:

* ``concrete`` (fixture tier only): real fixture arrays on the 32-device CPU mesh; WK, FP8-table
  and cache-initializer programs execute on CPU exactly as ``_load`` executes them on TPU.
* ``abstract`` (both tiers): ``ShapeDtypeStruct`` values carrying ``NamedSharding``; a producer
  program's outputs take the shardings its CPU-compiled executable reports (never executed).

``adapter_consistency`` asserts on the fixture that both modes hand every program identical
arguments (shape, dtype, sharding), compile identical programs (production's own lowering: digest,
signature, the jit's bound compiler options and the ``Lowered.compile`` arguments) and follow an
identical load protocol, which is what licenses the abstract production tier.

Run as ``python -m tools.equivalence.programs --tier fixture|production [--adapter programset]``
(JAX_PLATFORMS=cpu, 32 forced CPU devices); prints one JSON line of fingerprints.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from .common import canonical_json, emit, environment, require_cpu, sha256_hex, source_record
from .driver import ProgramSpec, abstract_like, leaf_signature

# The 181c013e donation rule and the runs of each tier (the build refuses a run whose ownership
# mode differs from this rule, wherever production keeps it).
DONATION_THRESHOLD = 8192  # request.CAPACITY: donate when capacity > this
PRODUCTION_CAPACITIES = (8192, 32768, 166912)
BATCH_CAPACITY, BATCH_SIZE = 32768, 4
FIXTURE_BATCH_SIZES = (1, 2, 3, 4)  # worker: concurrent_size=len(pending), request.batch allows 1..4
# The fixture's donated run: the smallest multiple of 512 above 8,192, so production's own donation
# rule (capacity > request.CAPACITY) applies -- no constant is patched, wherever the rule lives.
FIXTURE_DONATED_CAPACITY = 8704


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
    return {
        name: jax.ShapeDtypeStruct(
            frozen.arrays[name].shape, frozen.arrays[name].dtype, sharding=NamedSharding(mesh, spec)
        )
        for name, spec in pairs
    }


def production_arrays(mesh: Any, plans: Any) -> dict[str, Any]:
    """Abstract checkpoint arrays exactly as ``load_runtime_checkpoint`` builds them:
    global shape, dtype and ``NamedSharding(mesh, P(*partition_spec))`` of every tensor plan."""
    import jax
    import ml_dtypes
    from jax.sharding import NamedSharding, PartitionSpec as P

    dtypes = {"U8": np.dtype(np.uint8), "F32": np.dtype(np.float32), "BF16": np.dtype(ml_dtypes.bfloat16)}
    return {
        t.name: jax.ShapeDtypeStruct(
            tuple(t.global_shape), dtypes[t.dtype], sharding=NamedSharding(mesh, P(*t.partition_spec))
        )
        for t in plans[0].tensors
    }


def variants(tier: str) -> list[dict[str, Any]]:
    """The production runs a tier covers (capacity, donation expected by the 181c013e rule,
    concurrent size)."""
    if tier == "fixture":
        from .fixture import CAPACITY

        return [
            dict(capacity=CAPACITY, donating=CAPACITY > DONATION_THRESHOLD, concurrent_size=0),
            dict(
                capacity=FIXTURE_DONATED_CAPACITY,
                donating=FIXTURE_DONATED_CAPACITY > DONATION_THRESHOLD,
                concurrent_size=0,
            ),
        ] + [dict(capacity=CAPACITY, donating=False, concurrent_size=n) for n in FIXTURE_BATCH_SIZES]
    if tier == "production":
        runs = [dict(capacity=c, donating=c > DONATION_THRESHOLD, concurrent_size=0) for c in PRODUCTION_CAPACITIES]
        runs.append(dict(capacity=BATCH_CAPACITY, donating=True, concurrent_size=BATCH_SIZE))
        return runs
    raise ValueError("tier must be fixture or production")


def variant_key(run: dict[str, Any]) -> str:
    return (
        f"{run['capacity']}"
        + ("+donated" if run["donating"] else "")
        + (f"#n{run['concurrent_size']}" if run["concurrent_size"] else "")
    )


class TierPrograms(dict):
    """Ordered ``{record key: ProgramSpec}`` plus, per run, the recorded load protocol, whether
    the runtime compiled exactly its ``ProgramSet`` (``program_sets``) and the runtime's config;
    the cross-check adds each spec's declared donation (``donations``)."""

    protocol: dict[str, Any]
    program_sets: dict[str, Any]
    configs: dict[str, Any]
    donations: dict[str, tuple[int, ...]]


def program_specs(
    tier: str,
    mesh: Any,
    *,
    concrete: bool = False,
    only: set[str] | None = None,
    adapter: str = "runtime",
    fingerprint: bool = False,
    keep_lowered: bool = False,
) -> TierPrograms:
    """Programs of every run of a tier, keyed per run (``program_key``); ``only`` filters keys.
    ``fingerprint``: fingerprint the ``Lowered`` production compiled for each program;
    ``keep_lowered``: keep that ``Lowered`` for the keys in ``only`` (``diff``, ``authenticity``)."""
    if tier == "production" and concrete:
        raise ValueError("the production tier is abstract only")
    if adapter == "programset":
        return _program_specs_programset(tier, mesh, only=only)
    if adapter != "runtime":
        raise ValueError("adapter must be runtime or programset")
    from . import driver

    out = TierPrograms()
    out.protocol, out.program_sets, out.configs, out.donations = {}, {}, {}, {}
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
            mesh,
            tier=tier,
            capacity=run["capacity"],
            concurrent_size=run["concurrent_size"],
            arrays=arrays,
            plans=plans,
            concrete=concrete,
            fixture_geometry=geometry,
            outputs=outputs,
            fingerprint=fingerprint,
            keep=keep,
        )
        if built.protocol["record"]["state_ownership"] != ("exclusive_donated" if run["donating"] else "non_donating"):
            raise RuntimeError(
                f"run {variant_key(run)}: the runtime's donation differs from the 181c013e rule; "
                "update programs.variants"
            )
        out.protocol[variant_key(run)] = built.protocol
        out.program_sets[variant_key(run)] = built.program_set
        out.configs[variant_key(run)] = built.runtime.config
        _add_run(out, built.recorder.specs, run, only)
        del built
    return out


def _program_specs_programset(tier: str, mesh: Any, *, only: set[str] | None) -> TierPrograms:
    """The ProgramSet cross-check: for every run, ``build_program_set`` called standalone with the
    runtime's config and concurrent size, each spec paired with the arguments the real runtime
    passed to that program (abstract build), and the spec's declared ``donate_argnums``."""
    from glm_tpu.runner.programs import build_program_set

    from . import driver

    real = program_specs(tier, mesh, only=None)
    out = TierPrograms()
    out.protocol, out.program_sets, out.configs, out.donations = {}, {}, {}, {}
    for run in variants(tier):
        program_set = build_program_set(mesh, real.configs[variant_key(run)], concurrent_size=run["concurrent_size"])
        for spec in program_set.specs():
            key = program_key(spec.name, run)
            if key not in real:
                raise RuntimeError(f"run {variant_key(run)}: the runtime compiled no {spec.name}")
            if only is None or key in only:
                out[key] = driver.ProgramSpec(spec.name, spec.fn, real[key].args)
                out.donations[key] = tuple(spec.donate_argnums)
    return out


def donated_argnums(lowered: Any) -> list[int]:
    """Positional arguments every leaf of which the lowering donates."""
    import jax

    args, _ = lowered.args_info
    return [
        index
        for index, arg in enumerate(args)
        if (leaves := jax.tree.leaves(arg)) and all(leaf.donated for leaf in leaves)
    ]


def fingerprint_programset(specs: TierPrograms) -> dict[str, Any]:
    """Fingerprints of the cross-check specs, with the declared and the lowered donation."""
    from . import lowering, normalize

    records: dict[str, Any] = {}
    for key, spec in specs.items():
        started = time.perf_counter()
        with lowering.location_free():
            lowered = lowering.lower_for_tpu(spec.fn, spec.args)
            record = normalize.fingerprint(lowered, spec.args, summary=False)
            record["donated_argnums"] = donated_argnums(lowered)
        del lowered
        record["declared_donate_argnums"] = list(specs.donations[key])
        record["seconds"] = round(time.perf_counter() - started, 1)
        records[key] = record
    return records


def _add_run(out: dict[str, ProgramSpec], specs: list[ProgramSpec], run: dict[str, Any], only: set[str] | None) -> None:
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
        return (
            record.get("digest"),
            record.get("signature_digest"),
            record.get("jit_compiler_options"),
            spec.compile_calls,
        )

    for key in concrete:
        if key not in abstract:
            continue
        left, ltree = jax.tree.flatten(abstract_like(concrete[key].args))
        right, rtree = jax.tree.flatten(abstract_like(abstract[key].args))
        if (
            ltree != rtree
            or [leaf_signature(x) for x in left] != [leaf_signature(x) for x in right]
            or compiled(concrete[key]) != compiled(abstract[key])
        ):
            mismatches.append(key)
    if concrete.protocol != abstract.protocol:
        mismatches.append("load protocol")
    return dict(programs=len(concrete), identical=not mismatches, mismatches=mismatches)


def fingerprint_specs(specs: dict[str, ProgramSpec], *, summary: bool = True) -> dict[str, Any]:
    """Fingerprint every spec: a program production compiled keeps the fingerprint of the
    ``Lowered`` production built (``spec.record``); any other spec (FP8 tables, self-test
    variants) is lowered here from its ``(fn, args)``. ``compile`` records the
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


def run_tier(
    tier: str, *, consistency: bool = False, only: set[str] | None = None, adapter: str = "runtime"
) -> dict[str, Any]:
    from . import fixture, lowering

    require_cpu()
    mesh = fixture.cpu_mesh()
    started = time.perf_counter()
    with lowering.tpu_v4_info():
        specs = program_specs(tier, mesh, only=only, adapter=adapter, fingerprint=adapter == "runtime")
    built = time.perf_counter() - started
    records = fingerprint_programset(specs) if adapter == "programset" else fingerprint_specs(specs)
    result = dict(
        tier=tier,
        adapter=adapter,
        environment=environment(),
        source=source_record(),
        programs=records,
        tier_digest=tier_digest(records),
        runtime=specs.protocol,
        build_seconds=round(built, 1),
        seconds=round(time.perf_counter() - started, 1),
    )
    if adapter == "runtime":
        from . import driver, verdicts

        result["defaults"] = driver.production_defaults()
        cases = verdicts.admission_cases()
        result["verdicts"] = cases
        result["safety"] = dict(
            runs={key: verdicts.run_safety(run) for key, run in specs.protocol.items()},
            admission=verdicts.safety_verdicts(cases),
        )
        result["program_sets"] = specs.program_sets
    if consistency:
        with lowering.tpu_v4_info():
            result["adapter_consistency"] = adapter_consistency(mesh, abstract=specs if only is None else None)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tier", choices=("fixture", "production"), required=True)
    parser.add_argument(
        "--adapter",
        choices=("runtime", "programset"),
        default="runtime",
        help="runtime: the real OrdinaryRuntime (the gate); programset: build_program_set "
        "standalone with the runtime's arguments (cross-check)",
    )
    parser.add_argument("--consistency", action="store_true", help="also run the fixture adapter-consistency check")
    parser.add_argument("--only", action="append", help="restrict to these record keys")
    parser.add_argument("--text", action="store_true", help="emit the normalized text of the single --only program")
    args = parser.parse_args(argv)
    if args.text:
        emit(normalized_text(args.tier, args.only, adapter=args.adapter))
        return 0
    emit(
        run_tier(
            args.tier, consistency=args.consistency, only=set(args.only) if args.only else None, adapter=args.adapter
        )
    )
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
