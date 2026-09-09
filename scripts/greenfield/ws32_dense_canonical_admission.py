"""DB607-derived dense-loop placement checks; not numerical promotion.

Reuse all existing physical, kernel and helper checks. Only the two dense
suffixes move from ENTRY to four-iteration loops. Do not interpret their math.
Retained DB605 reproduction and measured live memory remain separate gates.
"""

from __future__ import annotations

from hashlib import sha256
import json
import re
from typing import Any, Mapping

from glm_tpu.greenfield.benchmarking.ws32_hlo_boolean_factors import BooleanFactors
from scripts.greenfield import ws32_dense_frontier_admission as a
from scripts.greenfield import ws32_dense_canonical as protocol

PROFILE = "ws32-dense01-canonical-placement-hlo-v1"
LOOP = "greenfield_ws32_prefill_dense_canonical/while"
STACKS = (
    ("bf16", (4, 32, 1536)),
    ("s32", (4, 32, 8)),
    ("f32", (4, 32, 8)),
    ("pred", (4, 32)),
)


def flatten(t: Any, value: Any, source: Any, dtype: str, shape: tuple) -> None:
    """One acquired row flatten, not arbitrary same-shape bitcast identity."""
    v = t.resolve(value)
    a._require(
        not v.path
        and v.op.opcode in ("reshape", "bitcast")
        and len(v.op.operand_names) == 1
        and a._shape(v, dtype, shape)
        and t.ssa.same(t.arg(v, 0), source),
        "canonical completed stack flatten differs",
    )


def canonical_suffix(index: Any, live: tuple) -> dict:
    bodies = a.prefix_bodies(index, live, loop_suffix=LOOP)
    t = a.RolledTransitions(index, 128)
    boolean = BooleanFactors(
        index, live_rows=128, live_mask=lambda _: None, nonempty_live=False
    )
    entry = index.roots["ENTRY"]
    a._require(len(entry.operand_names) == 24, "canonical output arity differs")
    records = []
    for body, layer in sorted(bodies.items(), key=lambda p: p[1]):
        loop = next(
            o
            for o in index.module.instructions
            if o.opcode == "while" and index.callee(o, "body") == body
        )
        initial = t.resolve(a.Value(index.operand(loop, 0)))
        root = a.Value(index.roots[body])
        parameter = t.parameter(body)
        a._require(len(loop.result_shapes) == 18, "canonical loop arity differs")
        for slot in range(5, 18):
            a._require(
                t.ssa.same(t.ssa.leaf(root, slot), t.ssa.leaf(parameter, slot)),
                "canonical immutable carry changed",
            )
        for slot, (dtype, dims) in enumerate(STACKS, 1):
            s = loop.result_shapes[slot]
            a._require(
                (s.dtype, s.dimensions) == (dtype, dims),
                "canonical stack interface differs",
            )
            zero = t.node(t.ssa.leaf(initial, slot), "broadcast", 1)
            a._require(
                a._shape(zero, dtype, dims)
                and t.ssa.constant(t.arg(zero, 0), dtype, 0),
                "canonical stack is not zero-initialized",
            )
            write = t.node(
                t.ssa.leaf(root, slot), "dynamic-update-slice", len(dims) + 2
            )
            a._require(
                a._shape(write, dtype, dims)
                and a._shape(t.arg(write, 1), dtype, (1, *dims[1:]))
                and t.ssa.same(t.arg(write, 0), t.ssa.leaf(parameter, slot))
                and t.ssa.same(t.arg(write, 2), t.ssa.leaf(parameter, 0))
                and all(
                    t.ssa.constant(t.arg(write, j), "s32", 0)
                    for j in range(3, len(dims) + 2)
                ),
                "canonical stack write is partial, displaced or non-own",
            )
            if slot == 1:
                tile = t.node(t.arg(write, 1), "bitcast", 1)
                selected = t.node(t.arg(tile, 0), "select", 3)
                sliced = t.node(t.arg(selected, 1), "slice", 1)
                summed = t.arg(sliced, 0)
                a._require(
                    a._shape(selected, "bf16", (32, 1536))
                    and re.search(
                        r"\bslice=\{\[0:32\],\[0:1536\]\}",
                        re.sub(r"\s", "", a._callee_attribute_text(sliced.op.raw_line)),
                    )
                    and a._shape(summed, "bf16", (128, 1536))
                    and summed.op.opcode == "all-reduce"
                    and a._computation_base(summed.op.computation) == body
                    and (summed.op.op_name or "").endswith("expert_down_reduce/psum"),
                    "canonical output must take first32 of its own dense reduction",
                )
        # ENTRY fuses both layer outputs in reverse tuple order. Resolve tuple
        # bindings rather than attributing every fusion operand to both outputs.
        output = t.node(t.ssa.leaf(a.Value(entry), 12 * layer), "select", 3)
        a._require(
            a._shape(output, "bf16", (128, 1536)), "canonical output shape differs"
        )
        flatten(t, t.arg(output, 1), t.ssa.leaf(a.Value(loop), 1), "bf16", (128, 1536))
        zero = t.node(t.arg(output, 2), "broadcast", 1)
        mask = t.node(t.arg(output, 0), "broadcast", 1)
        a._require(
            a._shape(zero, "bf16", (128, 1536))
            and t.ssa.constant(t.arg(zero, 0), "bf16", 0)
            and a._shape(mask, "pred", (128, 1536))
            and a._shape(t.arg(mask, 0), "pred", (128,))
            and "dimensions={0}" in a._callee_attribute_text(mask.op.raw_line),
            "canonical output masking interface differs",
        )
        for slot, field, dtype in ((2, 8, "s32"), (3, 9, "f32")):
            flatten(
                t,
                t.ssa.leaf(a.Value(entry), 12 * layer + field),
                t.ssa.leaf(a.Value(loop), slot),
                dtype,
                (128, 8),
            )
        health = boolean.factors(boolean.ref(entry, (12 * layer + 10,)))
        a._require(
            boolean.implies(health, boolean.ref(loop, (4,))),
            "canonical returned health does not require completed suffix health",
        )
        records.append(
            dict(
                layer=layer,
                body=body,
                complete_stack_writes=4,
                stack_slots=[1, 2, 3, 4],
                immutable_carries=list(range(5, 18)),
                output_slot=12 * layer,
                health_slot=12 * layer + 10,
            )
        )
    return dict(bodies=bodies, loops=records, arithmetic_proof=False)


def inspect_program(
    name: str, stable: str, optimized: str, memory: Mapping[str, Any]
) -> dict:
    a._require(
        name in protocol.RAW
        and (len(stable.encode()), sha256(stable.encode()).hexdigest())
        == protocol.RAW[name],
        "canonical preregistered raw graph differs",
    )
    a.validate_memory("dense01" if name == protocol.GRAPH else name, memory)
    structure = a.inspect_structure(
        "dense01" if name == protocol.GRAPH else name,
        optimized,
        suffix_check=canonical_suffix if name == protocol.GRAPH else None,
    )
    return json.loads(
        json.dumps(
            dict(
                profile=PROFILE,
                graph=name,
                passed=True,
                stablehlo_sha256=protocol.RAW[name][1],
                optimized_hlo_sha256=sha256(optimized.encode()).hexdigest(),
                compiled_memory=dict(memory),
                structure=structure,
                scope="CANONICAL_DENSE_REQUIRES_LIVE_MEMORY_AND_DB605_NARROW_REPRODUCTION",
                numerical_promotion=False,
                performance_claim=False,
            ),
            allow_nan=False,
        )
    )
