"""Narrow optimized-HLO proof of the route-sum → expert-combine boundary."""

from __future__ import annotations

import re
from typing import Any


def check_fp32_route_sum(
    module: Any, *, expert_scope: str | None = None, rows: int = 17
) -> dict[str, Any]:
    """Follow actual SSA/fusion roots; metadata only identifies the intended sum.

    Unknown forwarding fails closed. This is not a general numerical HLO proof:
    BF16 weighted-route production remains a separately tested input boundary.
    """
    if type(rows) is not int or rows not in (16, 17, 32, 128):
        raise ValueError("unregistered one-layer route-sum row geometry")
    computations = {}
    for op in module.instructions:
        name = re.sub(r"^ENTRY\s+", "", op.computation).split(" ", 1)[0]
        computations.setdefault(name, {})[op.name] = op

    def fp32(op):
        return (
            len(op.result_shapes) == 1
            and op.result_shapes[0].dtype == "f32"
            and not re.search(r"original_type\s*[:=]\s*\"?BF16", op.raw_line)
        )

    def callee(op, attribute):
        match = re.search(r"\b" + attribute + r"=(%?[\w.-]+)", op.raw_line)
        if not match or match[1] not in computations:
            raise ValueError("missing HLO computation binding")
        return match[1]

    def root(computation):
        roots = [
            op
            for op in computations[computation].values()
            if op.raw_line.startswith("ROOT ")
        ]
        if len(roots) != 1:
            raise ValueError("ambiguous HLO computation root")
        return roots[0]

    def reducer(op):
        computation = callee(op, "to_apply")
        nodes = list(computations[computation].values())
        params = [n for n in nodes if n.opcode == "parameter"]
        out = root(computation)
        if not (
            len(nodes) == 3
            and len(params) == 2
            and out.opcode == "add"
            and set(out.operand_names) == {p.name for p in params}
            and all(fp32(n) and n.result_shapes[0].dimensions == () for n in nodes)
        ):
            raise ValueError("route/expert reducer is not an FP32 scalar add")

    path = []

    def walk(op, computation, frames=()):
        if len(path) > 64 or not fp32(op):
            raise ValueError("non-FP32 or unbounded route-sum forwarding")
        path.append([computation, op.name, op.opcode])
        if op.opcode == "reduce":
            if (
                "greenfield_ws32_prefill_moe/fp32_route_sum" not in (op.op_name or "")
                or op.result_shapes[0].dimensions != (rows, 1536)
                or len(op.operand_shapes) != 2
                or op.operand_shapes[0].dtype != "f32"
                or op.operand_shapes[0].element_count != rows * 8 * 1536
            ):
                raise ValueError(
                    "expert input does not resolve to the live eight-route FP32 sum"
                )
            reducer(op)
            return
        if op.opcode == "fusion":
            child = callee(op, "calls")
            return walk(root(child), child, frames + ((op, computation),))
        if op.opcode == "parameter" and frames:
            number = re.search(r"\bparameter\((\d+)\)", op.raw_line)
            caller, parent = frames[-1]
            if not number or int(number[1]) >= len(caller.operand_names):
                raise ValueError("invalid fusion parameter forwarding")
            return walk(
                computations[parent][caller.operand_names[int(number[1])]],
                parent,
                frames[:-1],
            )
        if (
            op.opcode in ("copy", "bitcast", "reshape", "convert")
            and len(op.operand_names) == 1
        ):
            return walk(
                computations[computation][op.operand_names[0]], computation, frames
            )
        raise ValueError("unsupported route-sum forwarding")

    try:
        experts = [
            op
            for op in module.instructions
            if op.opcode == "all-reduce"
            and op.maximum_group_size == 8
            and (expert_scope is None or expert_scope in (op.op_name or ""))
        ]
        if len(experts) != 1 or len(experts[0].operand_names) != 1:
            raise ValueError("expected one expert combine")
        expert = experts[0]
        reducer(expert)
        computation = re.sub(r"^ENTRY\s+", "", expert.computation).split(" ", 1)[0]
        walk(computations[computation][expert.operand_names[0]], computation)
        return dict(passed=True, path=path, expert_collective=expert.name)
    except (ValueError, KeyError) as exc:
        return dict(passed=False, path=path, error=str(exc))
