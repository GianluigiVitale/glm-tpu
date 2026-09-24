"""Long full-KV storage lineage, not cache address/value or peak-HBM proof.

Allow only the registered BF16 cache stack (and E0 flat view) above the short
floating-weight threshold. Every full-size storage value must derive from
ENTRY tuple argument0's field2 through storage operations and all actual callers.
Bounded update values/indices retain their source/numerical obligations.
"""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from ...optimized.hlo_contract import HloInstruction, HloShape
from .ws32_batched_moe_hlo import PrefillHloIndex
from .ws32_pallas_one_layer import _callee_attribute_text, _computation_base


def prove_large_cache_storage(
    index: PrefillHloIndex, *, context_capacity: int, native_benchmark: bool = False,
) -> tuple[dict[str, Any], frozenset[int]]:
    """Return only proven instruction IDs; no caller-supplied shape exemption."""
    if type(native_benchmark) is not bool:
        raise ValueError("native benchmark storage profile must be explicit bool")
    allowed_capacities = (131072, 262656, 166912) if native_benchmark else (131072, 262656)
    if type(context_capacity) is not int or context_capacity not in allowed_capacities:
        raise ValueError("long cache requires registered capacity")
    stack = HloShape("bf16", (78, context_capacity // 512, 64, 640))
    allowed = {stack}
    if context_capacity == 262656 or (native_benchmark and context_capacity == 166912):
        allowed.add(HloShape("bf16", (78 * (context_capacity // 512) * 64, 640)))
    large: dict[int, HloInstruction] = {}
    positions: dict[int, int] = {}
    for op in index.module.instructions:
        found = [s for s in op.result_shapes if s.dtype in ("bf16", "f32")
                 and s.element_count >= 32 * 2048 * 1536]
        if not found:
            continue
        if len(found) != 1 or found[0] not in allowed:
            raise ValueError("unregistered full-size floating shape/tuple leaf")
        if len(op.result_shapes) > 1:
            # Explicitly reject nested tuples: the parser flattens shapes, so
            # their offsets would not establish get-tuple-element ownership.
            rhs = _callee_attribute_text(op.raw_line).split(" = ", 1)[1]
            prefix = rhs.split(op.raw_opcode + "(", 1)[0]
            prefix = re.sub(r"\{[^}]*\}", "", prefix)
            if prefix.count("(") != 1 or prefix.count(")") != 1:
                raise ValueError("full cache requires a flat tuple interface")
        large[op.index] = op
        positions[op.index] = op.result_shapes.index(found[0])

    callers: dict[str, list[tuple[HloInstruction, int | None]]] = defaultdict(list)
    for op in index.module.instructions:
        if op.opcode == "fusion":
            callers[index.callee(op, "calls")].append((op, None))
        elif op.opcode == "conditional":
            branches = re.findall(r"\bbranch_computations=\{([^}]*)\}", _callee_attribute_text(op.raw_line))
            if len(branches) != 1:
                raise ValueError("ambiguous long cache branches")
            names = tuple(x.strip() for x in branches[0].split(","))
            if len(names) != 2 or len(op.operand_names) != 3:
                raise ValueError("long cache requires binary conditional")
            for branch, name in enumerate(names):
                callers[name].append((op, branch + 1))

    roots: set[int] = set()
    dependencies: dict[int, set[int]] = {}

    def storage(op: HloInstruction) -> int:
        if op.index not in large:
            raise ValueError("full cache comes from an unregistered storage operand")
        return op.index

    for ident, op in large.items():
        comp = _computation_base(op.computation)
        deps: set[int] = set()
        if op.opcode == "parameter":
            numbers = re.findall(r"\bparameter\((\d+)\)", _callee_attribute_text(op.raw_line))
            if len(numbers) != 1:
                raise ValueError("ambiguous cache parameter")
            position = int(numbers[0])
            if comp == "ENTRY":
                if position != 0 or len(op.result_shapes) < 14 or op.result_shapes[2] != stack:
                    raise ValueError("full cache must originate from ENTRY tuple field2")
                roots.add(ident)
            else:
                owners = callers.get(comp, ())
                if not owners:
                    raise ValueError("full cache parameter lacks a bound caller")
                for caller, branch in owners:
                    if branch is not None and position != 0:
                        raise ValueError("branch cache parameter must be zero")
                    arg = index.operand(caller, position if branch is None else branch)
                    if arg.result_shapes != op.result_shapes:
                        raise ValueError("full cache parameter/caller shape mismatch")
                    deps.add(storage(arg))
        elif op.opcode in ("copy", "bitcast", "dynamic-update-slice", "scatter"):
            if len(op.result_shapes) != 1 or not op.operand_names:
                raise ValueError("full cache storage interface differs")
            base = index.operand(op, 0)
            if len(base.result_shapes) != 1 or base.result_shapes[0] not in allowed:
                raise ValueError("cache storage base is not a registered cache")
            if op.opcode != "bitcast" and base.result_shapes != op.result_shapes:
                raise ValueError("cache storage update/copy shape differs")
            if op.opcode in ("copy", "bitcast") and len(op.operand_names) != 1:
                raise ValueError("cache storage copy/view arity differs")
            if any(index.operand(op, n).index in large for n in range(1, len(op.operand_names))):
                raise ValueError("cache update must be bounded, not another full array")
            deps.add(storage(base))
        elif op.opcode == "tuple":
            args = [index.operand(op, n) for n in range(len(op.operand_names))]
            if (any(len(arg.result_shapes) != 1 for arg in args)
                    or tuple(arg.result_shapes[0] for arg in args) != op.result_shapes):
                raise ValueError("cache tuple must preserve exact flat operand shapes")
            sources = [index.operand(op, n) for n in range(len(op.operand_names))
                       if index.operand(op, n).index in large]
            if len(sources) != 1 or len(sources[0].result_shapes) != 1:
                raise ValueError("full cache tuple must have one direct cache operand")
            deps.add(storage(sources[0]))
        elif op.opcode == "get-tuple-element":
            source = index.operand(op, 0)
            source_id = storage(source)
            if (len(op.result_shapes) != 1 or re.findall(r"\bindex=(\d+)", _callee_attribute_text(op.raw_line))
                    != [str(positions[source_id])]
                    or op.result_shapes[0] != source.result_shapes[positions[source_id]]):
                raise ValueError("full cache extraction must select its exact tuple field")
            deps.add(source_id)
        elif op.opcode == "fusion":
            root = index.roots[index.callee(op, "calls")]
            if root.result_shapes != op.result_shapes:
                raise ValueError("full cache fusion root shape differs")
            deps.add(storage(root))
        elif op.opcode == "conditional":
            names = re.findall(r"\bbranch_computations=\{([^}]*)\}", _callee_attribute_text(op.raw_line))[0]
            for name in names.split(","):
                root = index.roots[name.strip()]
                if root.result_shapes != op.result_shapes:
                    raise ValueError("full cache conditional root shape differs")
                deps.add(storage(root))
        else:
            raise ValueError(f"unregistered full floating computation: {op.opcode}")
        dependencies[ident] = deps
    if len(roots) != 1:
        raise ValueError("expected one full cache argument")
    proven = set(roots)
    while True:
        added = {n for n, deps in dependencies.items() if deps and deps <= proven} - proven
        if not added:
            break
        proven.update(added)
    if proven != set(large):
        raise ValueError("full cache storage has an unbound/cyclic origin")
    return dict(
        passed=True, scope="FULL_SIZE_CACHE_BASE_STORAGE_LINEAGE_ONLY",
        instruction_count=len(proven), entry_parameter=0, source_tuple_field=2,
        allowed_shapes=[s.to_dict() for s in sorted(allowed, key=lambda s: s.dimensions)],
        not_proven=["UPDATE_INDICES_AND_VALUES", "CACHE_HEALTH_AND_COMMIT_SEMANTICS", "HBM_PEAK"],
    ), frozenset(proven)
