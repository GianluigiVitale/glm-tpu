"""Explicit small device-only assembly executables for the completed window.

No weights or caches enter these executables. Their actual compiler allocations
can therefore join the five model/WK analyses without hidden eager JAX programs.
No launcher or hardware admission is provided here.
"""

from __future__ import annotations

from typing import Any

from scripts.greenfield import prefill_completed_window as completed

PROGRAMS = ("prepare_prefix", "prepare_wide", "prepare_narrow", "assemble")
ROW_INPUT_INDICES = (0, 1, 5, 6, 7, 18, 19)
PREFIX_ROW_INDICES = (1, 5, 6, 7, 9)
RESULT_ROW_INDICES = (0, 1, 5, 6, 7, 8, 9, 10, 11)

# Conservative per-helper guardrails for this fixed small row-only interface;
# actual compiler allocations, never these caps, enter the live execution budget.
# These are predeclared refusal ceilings, not measured allocation predictions.
ALLOCATION_CAPS = {
    "argument_size_in_bytes": 32 << 20,
    "output_size_in_bytes": 32 << 20,
    "temp_size_in_bytes": 32 << 20,
    "generated_code_size_in_bytes": 8 << 20,
    "alias_size_in_bytes": 0,
}


def build_programs(mesh: Any, *, capacity: int = 4096) -> dict[str, Any]:
    """Compile each shape once, reusing dynamic tile/count across all three cases."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    if type(capacity) is not int or capacity != 4096:
        raise ValueError("completed assembly requires registered4096 capacity")
    replicated = NamedSharding(mesh, P())
    feature = NamedSharding(mesh, P(None, "feature"))
    health = NamedSharding(mesh, P("expert", "feature", None))
    row_specs = (
        feature,
        feature,
        replicated,
        replicated,
        replicated,
        health,
        replicated,
    )
    prefix_specs = (feature, replicated, replicated, replicated, feature)
    suffix_specs = (feature, replicated, replicated, health)
    result_specs = (
        feature,
        feature,
        replicated,
        replicated,
        replicated,
        replicated,
        replicated,
        health,
        feature,
    )

    def scalars(*values):
        if any(v.shape != () or v.dtype != jnp.int32 for v in values):
            raise ValueError("assembly metadata must be scalar int32")

    def prepare_prefix(rows, offset, count, tile):
        scalars(offset, count, tile)
        if len(rows) != 7 or any(
            r.shape[2 if i == 5 else 0] != 128 for i, r in enumerate(rows)
        ):
            raise ValueError("prefix assembly requires128 input rows")
        start = jnp.clip(offset, 0, capacity - 1)
        live = jnp.clip(count, 0, 128)
        safe_tile = jnp.clip(tile, 0, 3)
        valid = (
            (offset == start)
            & (count == live)
            & (live <= capacity - start)
            & (tile == safe_tile)
        )
        first = safe_tile * 32
        sliced = tuple(
            jax.lax.dynamic_slice_in_dim(r, first, 32, axis=2 if i == 5 else 0)
            for i, r in enumerate(rows)
        )
        sliced = (*sliced[:5], sliced[5] & valid, sliced[6])
        return (
            sliced,
            start + jnp.minimum(first, capacity - 1 - start),
            jnp.clip(live - first, 0, 32),
        )

    def prepare_wide(normalized, health_rows, count):
        scalars(count)
        # Reuse the actual concatenate/live-mask definition, not a second policy.
        prefixes = tuple(
            (n, None, None, None, None, None, None, None, h, None)
            for n, h in zip(normalized, health_rows, strict=True)
        )
        values = completed.suffix_inputs(prefixes, count, None, None)
        return values[0], values[1], values[4]

    def prepare_narrow(normalized, health_rows, count, tile):
        scalars(count, tile)
        valid = (count >= 0) & (count <= 128) & (tile >= 0) & (tile < 4)
        local_count = jnp.clip(
            jnp.clip(count, 0, 128) - jnp.clip(tile, 0, 3) * 32, 0, 32
        )
        return prepare_wide((normalized,), (health_rows & valid,), local_count)

    def assemble(prefix_rows, wide, narrow):
        prefixes = tuple(
            (None, p[0], None, None, None, p[1], p[2], p[3], None, p[4])
            for p in prefix_rows
        )
        control = tuple(
            jnp.concatenate([s[i] for s in narrow], axis=2 if i == 3 else 0)
            for i in range(4)
        )

        def rows(suffix):
            result = completed.assemble_result(prefixes, suffix)
            return tuple(result[i] for i in RESULT_ROW_INDICES)

        return rows(wide), rows(control)

    return {
        "prepare_prefix": jax.jit(
            prepare_prefix,
            in_shardings=(row_specs, replicated, replicated, replicated),
            out_shardings=(row_specs, replicated, replicated),
        ),
        "prepare_wide": jax.jit(
            prepare_wide,
            in_shardings=((feature,) * 4, (health,) * 4, replicated),
            out_shardings=(feature, replicated, health),
        ),
        "prepare_narrow": jax.jit(
            prepare_narrow,
            in_shardings=(feature, health, replicated, replicated),
            out_shardings=(feature, replicated, health),
        ),
        "assemble": jax.jit(
            assemble,
            in_shardings=((prefix_specs,) * 4, suffix_specs, (suffix_specs,) * 4),
            out_shardings=(result_specs, result_specs),
        ),
    }


def prefix_arguments(values: tuple, tile: Any) -> tuple:
    """Host tuple manipulation only; no device operation or weight/cache transfer."""
    return tuple(values[i] for i in ROW_INPUT_INDICES), values[8], values[9], tile


def prepare_programs(mesh: Any) -> tuple:
    """Fixed production shapes only; no weights, caches or allocated activations."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    def abstract(shape, dtype, spec=P()):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(mesh, spec))

    def feature(rows):
        return abstract((rows, 6144), jnp.bfloat16, P(None, "feature"))

    def health(rows):
        return abstract((8, 4, rows), jnp.bool_, P("expert", "feature", None))

    def metadata(rows):
        return (
            abstract((rows, 2048), jnp.int32),
            abstract((rows,), jnp.int32),
            abstract((rows, 2048), jnp.float32),
        )

    def suffix(rows):
        return (
            feature(rows),
            abstract((rows, 8), jnp.int32),
            abstract((rows, 8), jnp.float32),
            health(rows),
        )

    scalar = abstract((), jnp.int32)
    rows = (
        feature(128),
        feature(128),
        *metadata(128),
        health(128),
        abstract((128, 64), jnp.bfloat16),
    )
    prefix = (feature(32), *metadata(32), feature(32))
    arguments = {
        "prepare_prefix": (rows, scalar, scalar, scalar),
        "prepare_wide": ((feature(32),) * 4, (health(32),) * 4, scalar),
        "prepare_narrow": (feature(32), health(32), scalar, scalar),
        "assemble": ((prefix,) * 4, suffix(128), (suffix(32),) * 4),
    }
    functions = build_programs(mesh)
    return tuple((name, functions[name], arguments[name]) for name in PROGRAMS)


def compile_programs(
    *, mesh: Any, root: Any, record: dict, journal: Any, consensus: Any, compiler: Any
) -> tuple:
    """Reuse the existing raw-graph writer, journal and matched fleet phases.

    Called only as the later numerical continuation; this does not modify the
    original DB593 five-program compilation stack or execute any helper/model.
    """
    from scripts.greenfield import prefill_completed_window_admission as admission
    from scripts.greenfield.prefill_completed_window_worker import CompletedJournal
    from scripts.greenfield.prefill_window_acquisition import fleet_step

    def prepare():
        if not isinstance(journal, CompletedJournal) or set(record["programs"]) != set(
            admission.PROGRAMS
        ):
            raise ValueError(
                "assembly compile requires five acquired models and numerical journal"
            )
        return prepare_programs(mesh)

    prepared = fleet_step(
        "prepare_assembly", prepare, root=root, record=record, consensus=consensus
    )
    compiled = []
    for name, fn, args in prepared:
        graph = fleet_step(
            "compile_" + name,
            lambda: compiler(fn, args, name, root, record, journal=journal),
            root=root,
            record=record,
            consensus=consensus,
        )
        compiled.append(graph)

        def inspect():
            stable = (root / f"{name}.stablehlo.mlir").read_text()
            optimized = (root / f"{name}.optimized_hlo.txt").read_text()
            report = journal.inspect(
                name,
                stable,
                optimized,
                lambda: inspect_program(
                    name, stable, optimized, record["programs"][name]["compiled_memory"]
                ),
            )
            record["programs"][name]["admission"] = report
            journal.phase("compile/" + name, passed=True)

        fleet_step(
            "inspect_" + name, inspect, root=root, record=record, consensus=consensus
        )
    return tuple(compiled)


def place_tiles(mesh: Any) -> tuple:
    """Place four tiny scalar constants once, with no eager arithmetic/JIT."""
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P

    sharding = NamedSharding(mesh, P())
    return tuple(
        jax.device_put(np.asarray(tile, np.int32), sharding) for tile in range(4)
    )


def attach_prefix(values: tuple, prepared: tuple, previous: tuple | None) -> tuple:
    result = list(values)
    rows, result[8], result[9] = prepared
    for i, row in zip(ROW_INPUT_INDICES, rows, strict=True):
        result[i] = row
    if previous is not None:
        result[2:5] = previous[2:5]
    return tuple(result)


def suffix_arguments(prefixes: tuple, count: Any, tile: Any | None = None) -> tuple:
    if tile is None:
        return tuple(p[0] for p in prefixes), tuple(p[8] for p in prefixes), count
    if len(prefixes) != 1:
        raise ValueError("narrow assembly requires one completed prefix")
    return prefixes[0][0], prefixes[0][8], count, tile


def attach_suffix(values: tuple, prepared: tuple) -> tuple:
    return prepared[0], prepared[1], values[16], values[17], prepared[2]


def assembly_arguments(prefixes: tuple, wide: tuple, narrow: tuple) -> tuple:
    return (
        tuple(tuple(p[i] for i in PREFIX_ROW_INDICES) for p in prefixes),
        wide,
        narrow,
    )


def attach_result(rows: tuple, last_prefix: tuple) -> tuple:
    result = [None] * 12
    for i, value in zip(RESULT_ROW_INDICES, rows, strict=True):
        result[i] = value
    result[2:5] = last_prefix[2:5]
    return tuple(result)


def _inspect_device_copies(module: Any) -> list[dict[str, str]]:
    """Validate closed, same-layout HBM/VMEM copy pairs, never host transport."""
    import re

    index = {(op.computation, op.name): op for op in module.instructions}
    users: dict[tuple[str, str], list[Any]] = {}
    for op in module.instructions:
        for operand in op.operand_names:
            users.setdefault((op.computation, operand), []).append(op)

    def shapes(op):
        result = op.raw_line.split("=", 1)[1].split(op.raw_opcode + "(", 1)[0]
        return re.findall(r"[a-z][a-z0-9]*\[[0-9,]*\](?:\{[^{}]*\})?", result)

    def space(shape):
        found = re.findall(r"S\((\d+)\)", shape)
        return int(found[0]) if len(found) == 1 else (0 if not found else -1)

    pairs = []
    completed_names = set()
    for op in module.instructions:
        if op.opcode != "copy-start":
            continue
        outputs = shapes(op)
        source = (
            index.get((op.computation, op.operand_names[0]))
            if len(op.operand_names) == 1
            else None
        )
        consumers = users.get((op.computation, op.name), [])
        done = consumers[0] if len(consumers) == 1 else None
        if (
            len(outputs) != 3
            or op.raw_line.lstrip().startswith("ROOT ")
            or source is None
            or shapes(source) != outputs[1:2]
            or outputs[2] != "u32[]{:S(2)}"
            or {space(outputs[0]), space(outputs[1])} != {0, 3}
            or re.sub(r"S\([03]\)", "", outputs[0])
            != re.sub(r"S\([03]\)", "", outputs[1])
            or done is None
            or done.opcode != "copy-done"
            or done.operand_names != (op.name,)
            or shapes(done) != outputs[:1]
        ):
            raise ValueError(
                "assembly device copy is not a closed same-layout HBM/VMEM pair"
            )
        completed_names.add((done.computation, done.name))
        pairs.append(
            dict(
                start=op.name,
                done=done.name,
                source=source.name,
                destination_shape=outputs[0],
                source_shape=outputs[1],
            )
        )
    if completed_names != {
        (op.computation, op.name)
        for op in module.instructions
        if op.opcode == "copy-done"
    }:
        raise ValueError("assembly has unmatched copy-done")
    return pairs


def inspect_program(name: str, stable: str, optimized: str, memory: dict) -> dict:
    """Narrow row-assembly operation inventory; no model/collective allowlist."""
    from hashlib import sha256
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

    if name not in PROGRAMS:
        raise ValueError("unknown assembly program")
    validate_memory(memory)
    module = parse_hlo_module(optimized)
    allowed = {
        "parameter",
        "constant",
        "broadcast",
        "reshape",
        "bitcast",
        "copy",
        "copy-start",
        "copy-done",
        "tuple",
        "get-tuple-element",
        "slice",
        "dynamic-slice",
        "fusion",
        "add",
        "subtract",
        "multiply",
        "minimum",
        "maximum",
        "clamp",
        "compare",
        "and",
        "or",
        "not",
        "select",
        "convert",
        "concatenate",
        "iota",
        "optimization-barrier",
        "pad",
    }
    unexpected = sorted({op.opcode for op in module.instructions} - allowed)
    if not module.instructions or unexpected:
        raise ValueError(f"assembly contains non-row operation: {unexpected}")
    device_copies = _inspect_device_copies(module)
    return dict(
        graph=name,
        passed=True,
        scope="ROW_ONLY_ASSEMBLY_NOT_MODEL_OR_NUMERICAL_PROOF",
        stablehlo_sha256=sha256(stable.encode()).hexdigest(),
        optimized_hlo_sha256=sha256(optimized.encode()).hexdigest(),
        compiled_memory=dict(memory),
        opcodes=sorted({op.opcode for op in module.instructions}),
        physical_collective_count=0,
        device_copy_pairs=device_copies,
        performance_claim=False,
    )


def validate_memory(memory: dict) -> None:
    if set(memory) != set(ALLOCATION_CAPS) or any(
        type(memory[k]) is not int or not 0 <= memory[k] <= cap
        for k, cap in ALLOCATION_CAPS.items()
    ):
        raise ValueError(
            "assembly compiler allocations exceed registered small-helper caps"
        )


def memory_budget(census: Any, analyses: dict, *, active_graph: str) -> dict:
    """All nine executable analyses, including final assembly, with live buffers."""
    from scripts.greenfield import prefill_completed_window_admission as admission
    from glm_tpu.greenfield.validation.ws32_prefill_memory import (
        budget_resident_execution,
    )

    names = (*admission.PROGRAMS, *PROGRAMS)
    if set(analyses) != set(names) or active_graph not in names:
        raise ValueError(
            "completed runtime requires five model/WK and four assembly analyses"
        )
    pins = admission.registered_programs()
    for name in admission.PROGRAMS:
        admission._validate_memory(name, analyses[name], pins[name])
    for name in PROGRAMS:
        validate_memory(analyses[name])
    return budget_resident_execution(
        census,
        analyses,
        active_graph=active_graph,
        resident_graphs=names,
        required_reserve_bytes=admission.REQUIRED_RESERVE_BYTES,
    )
