"""Indexed, per-layer route-sum proof for acquired layer-major prefill HLO.

Adapts the bounded one-layer SSA proof without changing that historical gate.
This proves only the FP32 sum/combine boundary, not complete HLO admission.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from ..sharding.hlo_contract import HloInstruction, HloModule
from .ws32_decoder import _group_family
from .ws32_pallas_one_layer import _callee_attribute_text, _computation_base


_LAYER = re.compile(r"(?:^|/)greenfield_ws32_batched_prefill/layer_(\d+)(?:/|$)")
_EXPERT = "/greenfield_ws32_prefill_moe/expert_reduce/"
_ROUTE = "/greenfield_ws32_prefill_moe/fp32_route_sum/"


class PrefillHloIndex:
    """Index a module once, rather than scanning 400K instructions per layer."""

    def __init__(self, module: HloModule) -> None:
        self.module = module
        self.computations: dict[str, dict[str, HloInstruction]] = {}
        self.roots: dict[str, HloInstruction] = {}
        for op in module.instructions:
            name = _computation_base(op.computation)
            nodes = self.computations.setdefault(name, {})
            if op.name in nodes:
                raise ValueError("duplicate HLO instruction identity")
            nodes[op.name] = op
            if op.raw_line.startswith("ROOT "):
                if name in self.roots:
                    raise ValueError("ambiguous HLO computation root")
                self.roots[name] = op

    def operand(self, op: HloInstruction, position: int) -> HloInstruction:
        return self.computations[_computation_base(op.computation)][
            op.operand_names[position]
        ]

    def callee(self, op: HloInstruction, attribute: str) -> str:
        matches = re.findall(
            r"\b" + attribute + r"=(%?[\w.-]+)", _callee_attribute_text(op.raw_line)
        )
        if len(matches) != 1 or matches[0] not in self.computations:
            raise ValueError("missing/ambiguous HLO computation binding")
        return matches[0]

    def scalar_add(self, op: HloInstruction) -> None:
        name = self.callee(op, "to_apply")
        nodes = list(self.computations[name].values())
        root = self.roots[name]
        params = [n for n in nodes if n.opcode == "parameter"]
        numbers = [re.search(r"\bparameter\((\d+)\)", p.raw_line) for p in params]
        if not (
            len(nodes) == 3
            and len(params) == 2
            and root.opcode == "add"
            and len(root.operand_names) == 2
            and set(root.operand_names) == {p.name for p in params}
            and all(_f32(n, ()) for n in nodes)
            and all(numbers)
            and {number[1] for number in numbers if number is not None} == {"0", "1"}
        ):
            raise ValueError("route/expert reducer is not scalar FP32 ADD")


def _f32(op: HloInstruction, dimensions: tuple[int, ...]) -> bool:
    return (
        len(op.result_shapes) == 1
        and op.result_shapes[0].dtype == "f32"
        and op.result_shapes[0].dimensions == dimensions
        and not re.search(r'original_type\s*[:=]\s*"?BF16', op.raw_line)
    )


def _layer(op: HloInstruction) -> int:
    found = _LAYER.findall(op.op_name or "")
    if len(found) != 1:
        raise ValueError("combine/route sum lacks one exact layer scope")
    return int(found[0])


def check_batched_moe_route_sums(
    index: PrefillHloIndex,
    *,
    block_rows: int,
    live_instructions: Sequence[HloInstruction],
) -> dict[str, Any]:
    """Prove all75 live combines consume their own layer's eight-route F32 sum.

    The caller supplies the actual entry live closure, not a metadata subset.
    BF16 collective OUTPUT is allowed: observed XLA folds the post-sum cast there.
    Tuple forwarding is deliberately unsupported until actual evidence needs it.
    """
    if type(block_rows) is not int or not 1 <= block_rows <= 32:
        raise ValueError("route proof requires1..32 rows")
    proven: list[dict[str, Any]] = []
    try:
        live = {(op.computation, op.name) for op in live_instructions}
        experts = [
            op
            for op in index.module.instructions
            if _EXPERT in (op.op_name or "") and op.opcode == "all-reduce"
        ]
        if len(experts) != 75 or sorted(_layer(op) for op in experts) != list(
            range(3, 78)
        ):
            raise ValueError(
                "expected exactly one scoped combine for every MoE layer3..77"
            )
        for expert in sorted(experts, key=_layer):
            layer = _layer(expert)
            shape = (block_rows, 1536)
            if not (
                (expert.computation, expert.name) in live
                and expert.raw_opcode == "all-reduce"
                and _group_family(expert) == "expert"
                and len(expert.replica_groups) == 4
                and expert.use_global_device_ids
                and len(expert.operand_names)
                == len(expert.operand_shapes)
                == len(expert.result_shapes)
                == 1
                and expert.operand_shapes[0].dtype == "f32"
                and expert.operand_shapes[0].dimensions == shape
                and expert.result_shapes[0].dtype == "bf16"
                and expert.result_shapes[0].dimensions == shape
            ):
                raise ValueError(f"layer{layer}: dead/wrong combine groups or payload")
            index.scalar_add(expert)
            op = index.operand(expert, 0)
            frames: list[HloInstruction] = []
            path: list[list[str]] = []
            while True:
                if len(path) >= 64 or not _f32(op, shape):
                    raise ValueError(f"layer{layer}: non-FP32 or unbounded forwarding")
                path.append([_computation_base(op.computation), op.name, op.opcode])
                if op.opcode == "reduce":
                    if not (
                        _ROUTE in (op.op_name or "")
                        and _layer(op) == layer
                        and len(op.operand_shapes) == len(op.operand_names) == 2
                        and op.operand_shapes[0].dtype == "f32"
                        and op.operand_shapes[0].dimensions == (block_rows, 8, 1536)
                        and re.findall(
                            r"\bdimensions=\{([^}]*)\}",
                            _callee_attribute_text(op.raw_line),
                        )
                        == ["1"]
                    ):
                        raise ValueError(
                            f"layer{layer}: wrong layer/axis/eight-route sum"
                        )
                    routes, zero = index.operand(op, 0), index.operand(op, 1)
                    if (
                        not _f32(routes, (block_rows, 8, 1536))
                        or not _f32(zero, ())
                        or zero.opcode != "constant"
                        or not re.search(
                            r"\bconstant\(0(?:\.0*)?\)",
                            _callee_attribute_text(zero.raw_line),
                        )
                    ):
                        raise ValueError(
                            f"layer{layer}: route input or positive-zero initializer drift"
                        )
                    index.scalar_add(op)
                    break
                if op.opcode == "fusion":
                    frames.append(op)
                    op = index.roots[index.callee(op, "calls")]
                elif op.opcode == "parameter" and frames:
                    parameter = re.search(r"\bparameter\((\d+)\)", op.raw_line)
                    if parameter is None:
                        raise ValueError("invalid fusion parameter")
                    op = index.operand(frames.pop(), int(parameter[1]))
                elif (
                    op.opcode in {"copy", "bitcast", "reshape", "convert"}
                    and len(op.operand_names) == 1
                ):
                    op = index.operand(op, 0)
                else:
                    raise ValueError(f"layer{layer}: unsupported route-sum forwarding")
            proven.append(dict(layer=layer, expert_collective=expert.name, path=path))
        return dict(
            passed=True, layers=proven, scope="FP32_ROUTE_SUM_TO_EXPERT_COMBINE_ONLY"
        )
    except (ValueError, KeyError, IndexError) as error:
        return dict(
            passed=False,
            layers=proven,
            error=str(error),
            scope="FP32_ROUTE_SUM_TO_EXPERT_COMBINE_ONLY",
        )
