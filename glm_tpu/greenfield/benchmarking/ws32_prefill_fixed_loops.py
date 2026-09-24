"""Shared fixed-four loop interface/counter proof, extracted from dense admission.

This proves only ENTRY ownership, tuple interfaces and zero/+1/<4 induction.
Mutable stacks, immutable carries and consumers require separate checks.
"""

from __future__ import annotations

import re
from typing import Sequence

from ...optimized.hlo_contract import HloInstruction
from .ws32_batched_commit_hlo import _require, _shape
from .ws32_batched_moe_hlo import PrefillHloIndex
from .ws32_pallas_one_layer import _computation_base
from .ws32_prefill_hlo_identity import Value, attribute
from .ws32_rolled_prefill_hlo import RolledTransitions

LAYER = re.compile(r"(?:^|/)greenfield_ws32_batched_prefill/layer_(\d+)(?:/|$)")


def fixed_loop_bodies(
    index: PrefillHloIndex,
    live: Sequence[HloInstruction],
    *,
    loop_suffix: str,
    expected_layers: tuple[int, ...],
) -> dict[str, int]:
    t = RolledTransitions(index, 128)
    loops = [
        op
        for op in index.module.instructions
        if op.opcode == "while" and (op.op_name or "").endswith(loop_suffix)
    ]
    _require(len(loops) == len(expected_layers), "dense fixed loop count differs")
    live_ids = {op.index for op in live}
    bodies = {}
    for loop in loops:
        layers = LAYER.findall(loop.op_name or "")
        _require(
            len(layers) == 1
            and int(layers[0]) in expected_layers
            and loop.index in live_ids
            and _computation_base(loop.computation) == "ENTRY"
            and len(loop.operand_names) == 1,
            "dense prefix loop owner/liveness differs",
        )
        body, condition = (index.callee(loop, name) for name in ("body", "condition"))
        parameter, cond_param = t.parameter(body), t.parameter(condition)
        initial = t.resolve(Value(index.operand(loop, 0)))
        root = Value(index.roots[body])
        _require(
            initial.op.opcode == root.op.opcode == "tuple"
            and len(initial.op.operand_names)
            == len(root.op.operand_names)
            == len(loop.result_shapes)
            and initial.op.result_shapes
            == root.op.result_shapes
            == loop.result_shapes
            == parameter.op.result_shapes
            == cond_param.op.result_shapes,
            "dense loop tuple interface differs",
        )
        _require(
            t.ssa.constant(t.ssa.leaf(initial, 0), "s32", 0), "dense loop must start0"
        )
        counter = t.ssa.leaf(parameter, 0)
        update = t.node(t.ssa.leaf(root, 0), "add", 2)
        _require(
            _shape(update, "s32", ())
            and any(
                t.ssa.same(t.arg(update, j), counter)
                and t.ssa.constant(t.arg(update, 1 - j), "s32", 1)
                for j in (0, 1)
            ),
            "dense loop must advance own counter by1",
        )
        cond = t.node(Value(index.roots[condition]), "compare", 2)
        _require(
            _shape(cond, "pred", ())
            and attribute(cond.op, "direction") == "LT"
            and t.ssa.same(t.arg(cond, 0), t.ssa.leaf(cond_param, 0))
            and t.ssa.constant(t.arg(cond, 1), "s32", 4),
            "dense loop must test own counter<4",
        )
        _require(body not in bodies, "dense loops share a body")
        bodies[body] = int(layers[0])
    _require(
        sorted(bodies.values()) == list(expected_layers),
        "dense prefix layer inventory differs",
    )
    return bodies
