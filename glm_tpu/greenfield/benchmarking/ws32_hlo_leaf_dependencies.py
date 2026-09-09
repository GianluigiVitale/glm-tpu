"""Memoized selected-leaf dependency, NOT identity or boolean implication.

Fusion summaries keep parameter paths symbolic until bound at each callsite.
Explicit work stacks handle deep compiler graphs without Python recursion.
"""

from __future__ import annotations

import re

from ..sharding.hlo_contract import HloInstruction
from .ws32_batched_moe_hlo import PrefillHloIndex
from .ws32_batched_helper_hlo import _target
from .ws32_prefill_hlo_identity import attribute


PURE = frozenset(
    """add subtract multiply divide remainder minimum maximum
abs negate sqrt rsqrt sine cosine exponential log floor ceil power compare
and or xor not shift-left shift-right-arithmetic shift-right-logical
convert bitcast reshape transpose broadcast pad slice dynamic-slice concatenate select
reduce gather clamp copy copy-start copy-done is-finite iota constant
population-count count-leading-zeros round-nearest-even""".split()
)


class LeafDependencies:
    """Tokens are (parameter|terminal, numeric identity, selected tuple path).

    Arithmetic operands contribute possible dependencies even when an operation
    annihilates a value. This must NEVER be used to assert numerical correctness
    or that a health predicate necessarily gates execution.
    """

    def __init__(self, index: PrefillHloIndex, terminals: set[int]) -> None:
        self.index, self.terminals = index, frozenset(terminals)
        self.memo: dict[tuple, frozenset[tuple]] = {}

    def dependencies(
        self, op: HloInstruction, path: tuple[int, ...] = ()
    ) -> frozenset[tuple]:
        target = (op.index, path)
        stack = [(op, path)]
        active: set[tuple] = set()
        while stack:
            node, leaf = stack[-1]
            key = (node.index, leaf)
            if key in self.memo:
                stack.pop()
                active.discard(key)
                continue
            if len(self.memo) + len(stack) > 100000:
                raise ValueError("dependency slice exceeds bounded profile")
            if node.index in self.terminals:
                self.memo[key] = frozenset({("terminal", node.index, leaf)})
                continue
            if node.opcode == "parameter":
                number = re.findall(r"\bparameter\((\d+)\)", node.raw_line)
                if len(number) != 1:
                    raise ValueError("ambiguous dependency parameter")
                self.memo[key] = frozenset({("parameter", int(number[0]), leaf)})
                continue
            if node.opcode == "constant" and not leaf:
                # Literal spellings such as -inf may look like operand names
                # to the general HLO tokenizer; they are never graph edges.
                self.memo[key] = frozenset()
                continue
            extra: set[tuple] = set()
            if node.opcode == "get-tuple-element":
                children = [
                    (
                        self.index.operand(node, 0),
                        (int(attribute(node, "index")),) + leaf,
                    )
                ]
            elif node.opcode == "tuple":
                if not leaf or leaf[0] >= len(node.operand_names):
                    raise ValueError("dependency requires a selected tuple leaf")
                children = [(self.index.operand(node, leaf[0]), leaf[1:])]
            elif node.opcode == "fusion":
                root = self.index.roots[self.index.callee(node, "calls")]
                summary = self.memo.get((root.index, leaf))
                if summary is None:
                    child_key = (root.index, leaf)
                    if child_key in active:
                        raise ValueError("cyclic dependency computation")
                    active.add(key)
                    stack.append((root, leaf))
                    continue
                children = []
                for kind, number, subpath in summary:
                    if kind == "parameter":
                        children.append((self.index.operand(node, number), subpath))
                    else:
                        extra.add((kind, number, subpath))
            elif node.opcode == "custom-call" and not leaf:
                if (
                    _target(node) != "AssumeGatherIndicesInBound"
                    or len(node.operand_names) != 1
                    or node.result_shapes != self.index.operand(node, 0).result_shapes
                ):
                    raise ValueError("unsupported custom dependency")
                children = [(self.index.operand(node, 0), ())]
            elif node.opcode in PURE and not leaf:
                children = [
                    (self.index.operand(node, i), ())
                    for i in range(len(node.operand_names))
                ]
            else:
                raise ValueError(
                    f"unsupported dependency operation {node.opcode}/{leaf}"
                )
            missing = [
                (child, subpath)
                for child, subpath in children
                if (child.index, subpath) not in self.memo
            ]
            if missing:
                active.add(key)
                child, subpath = missing[0]
                if (child.index, subpath) in active:
                    raise ValueError("cyclic dependency slice")
                stack.append((child, subpath))
                continue
            for child, subpath in children:
                extra.update(self.memo[(child.index, subpath)])
            self.memo[key] = frozenset(extra)
        return self.memo[target]
