"""Operand finite checks under actual rolled-tile and suffix health guards.

Reuse the historical boolean/finite/query-layout rules, not kernel mathematics.
Global-to-tile implication is independently required by the integrated profile.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any, Sequence

from ..sharding.hlo_contract import HloInstruction, HloShape
from .ws32_batched_commit_hlo import _require
from .ws32_batched_moe_hlo import PrefillHloIndex
from .ws32_batched_operand_health_hlo import OperandHealthProof
from .ws32_batched_kernel_hlo import _LAYER
from .ws32_hlo_boolean_factors import BooleanFactors, dimensions
from .ws32_pallas_one_layer import _computation_base, _callee_attribute_text
from .ws32_prefill_hlo_identity import attribute
from .ws32_rolled_prefill_hlo import RolledTransitions, RolledTileHealth, _rows


class TileBooleanFactors(BooleanFactors):
    def _negative_live(self, value, depth=0):
        if depth >= 8:
            return None
        value = self.resolve(value)
        op = self.ops[value[0]]
        if value[1] or len(op.operand_names) != 1:
            return None
        source = self.resolve(self.operand(value, 0))
        if op.opcode == "not" and self.shape(source) == self.shape(value):
            return self.live_mask(source)
        if op.opcode == "broadcast":
            axis = self._negative_live(source, depth + 1)
            mapping = dimensions(op)
            a, b = self.shape(source), self.shape(value)
            if (
                axis is not None
                and a.dtype == b.dtype == "pred"
                and len(mapping) == len(a.dimensions)
                and len(set(mapping)) == len(mapping)
                and all(0 <= j < len(b.dimensions) for j in mapping)
                and tuple(b.dimensions[j] for j in mapping) == a.dimensions
            ):
                return mapping[axis]
        return None

    def _rule(self, key):
        value, axis = key
        op = self.ops[value[0]]
        if not value[1] and op.opcode == "or" and len(op.operand_names) == 2:
            args = [self.resolve(self.operand(value, j)) for j in (0, 1)]
            _require(
                all(self.shape(v) == self.shape(value) for v in args),
                "rolled OR shape drift",
            )
            for j in (0, 1):
                live_axis = self._negative_live(args[j])
                if live_axis is not None and axis in (None, live_axis):
                    return "union", [self._key(args[1 - j], live_axis)]
        if (
            not value[1]
            and op.opcode == "bitcast"
            and len(op.operand_names) == 1
            and self.shape(value) == HloShape("pred", (1, 32))
            and axis in (None, 1)
        ):
            source = self.resolve(self.operand(value, 0))
            # Exact acquired row-vector -> one-row matrix storage bridge.
            # Both pack consecutive first32 predicate bits in padded512 lanes;
            # no feature dimension, transpose, or padded lane becomes a row.
            if (
                self.shape(source) == HloShape("pred", (32,))
                and re.search(
                    r"=\s*pred\[1,32\]\{1,0:T\(4,128\)\(4,1\)(?:S\(\d+\))?\}",
                    op.raw_line,
                )
                and re.search(
                    r"=\s*pred\[32\]\{0:T\(512\)\(128\)\(4,1\)(?:S\(\d+\))?\}",
                    self.ops[source[0]].raw_line,
                )
            ):
                return "union", [self._key(source, None if axis is None else 0)]
        return super()._rule(key)


class OperandTileHealth(RolledTileHealth):
    def count(self, value):
        actual = self.transitions.integer_expression(self.value_ref(value), self.loop)
        allowed = [self.expected_count]
        for _ in range(2):
            allowed.append(
                self.transitions.integer_node(
                    "minimum",
                    ("constant", 32),
                    self.transitions.integer_node(
                        "maximum", ("constant", 0), allowed[-1]
                    ),
                )
            )
        return actual in allowed


class TileOperands(OperandHealthProof):
    def __init__(self, transitions: RolledTransitions, loop, health: RolledTileHealth):
        self.index, self.rows = transitions.index, 32
        self.health, self.boolean = health, health.boolean
        _require(health.inspect(loop)["passed"], "tile lacks writer health")
        stack = transitions.resolve(transitions.ssa.leaf(loop.root, loop.health_slot))
        self.known = self.boolean.factors(
            health.boolean_ref(transitions.arg(stack, 1)), axis=1
        )
        self._index_finite()

    def bridge(self, value, kind):
        if kind != "absorbed":
            return super().bridge(value, kind)
        value = self.node(value, "bitcast", 1)
        self.layout(value, (32, 8, 512), "2,1,0")

        def copied_source(view):
            # Keep the immediate copy's destination layout, BEFORE resolve
            # strips a logical copy whose source uses a different layout.
            copied = self.boolean.operand(view, 0)
            op = self.boolean.ops[copied[0]]
            _require(
                op.opcode == "copy" and len(op.operand_names) == 1,
                "absorbed view lacks layout copy",
            )
            self.layout(copied, (8, 4, 32, 128), "3,0,1,2")
            source = self.arg(copied, 0)
            self.node(source, "bitcast", 1)
            self.layout(source, (8, 4, 32, 128), "3,2,1,0")
            return self.arg(source, 0)

        source = copied_source(value)
        # Two compiler views of the same structured output, via identical
        # layout copies. The finite view is the historical [row,head,4,128]
        # bridge to [row,head,512]; no row or feature order is inferred by size.
        for finite in self.finite:
            if self.boolean.shape(finite) != HloShape("bf16", (32, 8, 4, 128)):
                continue
            if self.boolean.ops[finite[0]].opcode != "bitcast":
                continue
            self.layout(finite, (32, 8, 4, 128), "3,1,2,0")
            _require(
                copied_source(finite) == source,
                "absorbed health checks another structured output",
            )
            return finite
        raise ValueError("absorbed view lacks own finite layout witness")


def check_rolled_operand_health(
    index: PrefillHloIndex,
    *,
    block_rows: int,
    live_instructions: Sequence[HloInstruction],
) -> dict[str, Any]:
    _rows(block_rows)
    report = dict(
        passed=False,
        scope="ROLLED_OPERAND_FINITE_AND_PANEL_VALIDITY_GUARDS",
        prefix_assumption="NONEMPTY_TILE_LIVE_HEALTH; global bridge and empty no-write are separate required proofs",
        not_proven=[
            "KERNEL_OR_ROUTING_ARITHMETIC",
            "ALL_MODEL_HEALTH_CHECKS_EXIST",
            "NUMERICAL_OR_MEMORY_ADMISSION",
        ],
    )
    try:
        transitions = RolledTransitions(index, block_rows)
        loops = transitions.all_loops(live_instructions)
        health = OperandTileHealth(transitions)
        health.boolean = TileBooleanFactors(
            index, live_rows=32, live_mask=health.live_mask, nonempty_live=False
        )
        suffixes = {
            "normalized": "/greenfield_fp8_block_matmul_f32_m32_k1536_n640/pallas_call",
            "prepared_query": "/greenfield_fp8_block_matmul_m32_k2048_n2048/pallas_call",
            "sparse": "/greenfield_pregathered_sparse_mla_h8_k2048_b512_w640_prefill_m32/pallas_call",
        }
        families = defaultdict(dict)
        for op in index.module.instructions:
            if op.opcode != "custom-call":
                continue
            for family, suffix in suffixes.items():
                if not (op.op_name or "").endswith(suffix):
                    continue
                layers = _LAYER.findall(op.op_name)
                _require(len(layers) == 1, "operand lacks unique layer")
                layer = int(layers[0])
                _require(layer not in families[family], "duplicate operand anchor")
                _require(
                    0 <= layer < 78
                    and _computation_base(op.computation)
                    == index.callee(loops[layer].loop, "body"),
                    "operand outside its own prefix",
                )
                families[family][layer] = op
        _require(
            set(families) == set(suffixes)
            and all(set(v) == set(range(78)) for v in families.values()),
            "missing rolled operand coverage",
        )
        records = []
        for loop in loops:
            p = TileOperands(transitions, loop, health)
            record = dict(layer=loop.layer)
            for family in ("normalized", "prepared_query"):
                op = families[family][loop.layer]
                call = p.node(p.boolean.ref(op), "custom-call", 3)
                source = p.arg(call, 0)
                p.shape(source, "bf16", (32, 1536 if family == "normalized" else 2048))
                _require(
                    p.finite_live(source),
                    f"layer{loop.layer} {family} lacks live finite guard",
                )
                record[family] = op.name
            op = families["sparse"][loop.layer]
            call = p.node(p.boolean.ref(op), "custom-call", 3)
            p.query(p.arg(call, 1))
            cache = p.bridge(p.arg(call, 2), "cache")
            _require(
                p.finite_live(cache),
                f"layer{loop.layer} selected cache lacks finite guard",
            )
            record["sparse"] = op.name
            records.append(record)
        report["layers"] = records

        # Wide suffix guards are tied directly to the actual outer commit.
        outer = OperandHealthProof(index, block_rows)
        routers = []
        logits = {}
        biases = {}
        for op in index.module.collectives:
            layers = _LAYER.findall(op.op_name or "")
            if len(layers) != 1:
                continue
            layer = int(layers[0])
            if op.opcode == "all-gather" and op.result_shapes == (
                HloShape("f32", (block_rows, 256)),
            ):
                _require(layer not in logits, "duplicate suffix router logits")
                logits[layer] = op
            if op.opcode == "all-reduce" and op.result_shapes == (
                HloShape("bf16", (block_rows, 1536)),
                HloShape("f32", (256,)),
            ):
                # Compiler combines next-layer bias with preceding MLP output.
                _require(layer + 1 not in biases, "duplicate suffix router bias")
                biases[layer + 1] = op
        _require(
            set(logits) == set(biases) == set(range(3, 78)),
            "suffix router coverage drift",
        )
        for layer in range(3, 78):
            for op, path, axis in ((logits[layer], (), 0), (biases[layer], (1,), None)):
                value = outer.boolean.resolve(outer.boolean.ref(op, path))
                domains = outer.finite.get(value, set())
                _require(
                    None in domains or axis in domains,
                    f"layer{layer} router operand lacks finite guard",
                )
            routers.append(
                dict(layer=layer, logits=logits[layer].name, bias=biases[layer].name)
            )
        report["routers"] = routers

        parents = defaultdict(list)
        calls = []
        for op in index.module.instructions:
            if op.opcode == "conditional":
                groups = re.findall(
                    r"\bbranch_computations=\{([^}]*)\}",
                    _callee_attribute_text(op.raw_line),
                )
                _require(len(groups) == 1, "ambiguous panel conditional")
                for branch, name in enumerate(groups[0].split(",")):
                    parents[name.strip()].append((op, branch))
            if op.opcode == "custom-call" and (op.op_name or "").endswith(
                "/greenfield_prefill_expert_panel_raw_fp8/pallas_call"
            ):
                calls.append(op)
        panels = []
        for call in calls:
            owners = parents[_computation_base(call.computation)]
            _require(
                len(owners) == 1 and owners[0][1] == 1,
                "panel lacks unique valid branch",
            )
            conditional = owners[0][0]
            _require(
                _computation_base(conditional.computation) == "ENTRY",
                "panel validity outside suffix",
            )
            predicate = outer.health.predicate(outer.boolean.ref(conditional))
            predicate = outer.node(predicate, "and", 2)
            outer.shape(predicate, "pred", ())
            b = outer.boolean
            bindings = (b.ref(index.operand(conditional, 2)),)
            if bindings not in b.frame_ids:
                b.frame_ids[bindings] = len(b.frames)
                b.frames.append(bindings)
            grid = b.resolve((index.operand(call, 0).index, (), b.frame_ids[bindings]))
            outer.shape(grid, "s32", ())
            matched = []
            for j in (0, 1):
                valid, active = outer.arg(predicate, j), outer.arg(predicate, 1 - j)
                op = b.ops[active[0]]
                if (
                    not active[1]
                    and op.opcode == "compare"
                    and len(op.operand_names) == 2
                    and attribute(op, "direction") == "GT"
                    and outer.arg(active, 0) == grid
                    and outer.health.constant(outer.arg(active, 1), 0)
                    and b.implies(outer.known, valid)
                ):
                    matched.append(valid)
            _require(
                len(matched) == 1, "panel validity or actual active-grid guard drift"
            )
            layers = _LAYER.findall(call.op_name or "")
            _require(len(layers) == 1, "panel lacks own layer")
            panels.append(
                dict(
                    layer=int(layers[0]),
                    kernel=call.name,
                    validity_guard=conditional.name,
                    activity_is_not_required_for_commit=True,
                )
            )
        _require(
            len(panels) == 225
            and Counter(r["layer"] for r in panels)
            == Counter({layer: 3 for layer in range(3, 78)}),
            "panel validity coverage drift",
        )
        report.update(passed=True, panels=panels)
    except (ValueError, KeyError, IndexError) as error:
        report["error"] = str(error)
    return report
