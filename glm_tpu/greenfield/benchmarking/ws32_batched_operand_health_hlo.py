"""Operand-bound finite and grouped validity guards for acquired short prefill.

No kernel mathematics is interpreted. Only exact SSA, one duplicate-slice
equivalence and two acquired within-row storage bridges connect finite guards
to real inputs. Metadata/routing arithmetic and numerical admission are separate.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any

from ..sharding.hlo_contract import HloInstruction, HloShape
from .ws32_batched_commit_hlo import _require
from .ws32_batched_health_hlo import WriterHealthProof
from .ws32_batched_kernel_hlo import _LAYER
from .ws32_batched_moe_hlo import PrefillHloIndex
from .ws32_hlo_boolean_factors import Ref, dimensions
from .ws32_pallas_one_layer import _callee_attribute_text, _computation_base


class OperandHealthProof:
    def __init__(self, index: PrefillHloIndex, rows: int) -> None:
        self.index, self.rows = index, rows
        self.health = WriterHealthProof(index, rows)
        self.boolean = self.health.boolean
        self.known = self.boolean.factors(self.health.frontier())
        self.finite: dict[Ref, set[int | None]] = defaultdict(set)
        self.slices: dict[tuple, set[int | None]] = defaultdict(set)
        for ref, domain in self.known:
            op = self.boolean.ops[ref[0]]
            if not ref[1] and op.opcode == "is-finite" and len(op.operand_names) == 1:
                source = self.arg(ref, 0)
                _require(
                    self.boolean.shape(ref)
                    == HloShape("pred", self.boolean.shape(source).dimensions),
                    "finite guard shape differs from actual operand",
                )
                self.finite[source].add(domain)
                signature = self.slice_signature(source)
                if signature is not None:
                    self.slices[signature].add(domain)

    def arg(self, value: Ref, number: int) -> Ref:
        return self.health.arg(value, number)

    def node(self, value: Ref, opcode: str, arity: int) -> Ref:
        return self.health.node(value, opcode, arity)

    def shape(self, value: Ref, dtype: str, dims: tuple[int, ...]) -> None:
        _require(
            self.boolean.shape(value) == HloShape(dtype, dims),
            "operand health shape/dtype drift",
        )

    def constant(self, value: Ref, dtype: str, spelling: str) -> None:
        value = self.boolean.resolve(value)
        op = self.boolean.ops[value[0]]
        self.shape(value, dtype, ())
        _require(
            not value[1]
            and op.opcode == "constant"
            and re.findall(
                r"\bconstant\(([^)]*)\)", _callee_attribute_text(op.raw_line)
            )
            == [spelling],
            "operand health constant drift",
        )

    def slice_signature(self, value: Ref) -> tuple | None:
        op = self.boolean.ops[value[0]]
        if value[1] or op.opcode != "slice" or len(op.operand_names) != 1:
            return None
        # This one same-source slice equivalence is sufficient for q_absorbed.
        if self.boolean.shape(value) != HloShape("bf16", (self.rows, 8, 4, 128)):
            return None
        ranges = re.findall(r"\bslice=\{([^}]*)\}", _callee_attribute_text(op.raw_line))
        expected = f"[0:{self.rows}],[0:8],[0:4],[0:128]"
        if len(ranges) != 1 or re.sub(r"\s", "", ranges[0]) != expected:
            return None
        source = self.arg(value, 0)
        return (source, self.boolean.shape(value), expected)

    def finite_live(self, value: Ref) -> bool:
        value = self.boolean.resolve(value)
        domains = self.finite.get(value, set())
        if None in domains or 0 in domains:
            return True
        signature = self.slice_signature(value)
        domains = self.slices.get(signature, set()) if signature is not None else set()
        return None in domains or 0 in domains

    def layout(self, value: Ref, dims: tuple[int, ...], minor: str) -> None:
        self.shape(value, "bf16", dims)
        op = self.boolean.ops[value[0]]
        _require(
            not value[1]
            and re.search(
                r"=\s*bf16\["
                + ",".join(map(str, dims))
                + r"\]\{"
                + re.escape(minor)
                + r":T\(8,128\)\(2,1\)(?:S\(\d+\))?\}",
                op.raw_line,
            )
            is not None,
            "unacquired value-bitcast layout",
        )

    def bridge(self, value: Ref, kind: str) -> Ref:
        value = self.node(value, "bitcast", 1)
        source = self.arg(value, 0)
        if kind == "cache":
            self.layout(value, (self.rows, 4, 512, 640), "3,2,1,0")
            self.layout(source, (self.rows, 2048, 640), "2,1,0")
        elif kind == "absorbed":
            self.layout(value, (self.rows, 8, 512), "2,1,0")
            self.layout(source, (self.rows, 8, 4, 128), "3,1,2,0")
        else:
            raise ValueError("unknown operand health bridge")
        return source

    def padded(self, value: Ref, width: int, padding: str) -> Ref:
        value = self.node(value, "convert", 1)
        self.shape(value, "f32", (self.rows, 8, 640))
        pad = self.node(self.arg(value, 0), "pad", 2)
        self.shape(pad, "bf16", (self.rows, 8, 640))
        source = self.arg(pad, 0)
        self.shape(source, "bf16", (self.rows, 8, width))
        self.constant(self.arg(pad, 1), "bf16", "-inf")
        _require(
            re.findall(
                r"\bpadding=([^,\s]+)",
                _callee_attribute_text(self.boolean.ops[pad[0]].raw_line),
            )
            == [padding],
            "query component pad span drift",
        )
        return source

    def zero_tail(self, value: Ref) -> None:
        value = self.node(value, "convert", 1)
        self.shape(value, "f32", (self.rows, 8, 640))
        broadcast = self.node(self.arg(value, 0), "broadcast", 1)
        self.shape(broadcast, "bf16", (self.rows, 8, 640))
        _require(
            dimensions(self.boolean.ops[broadcast[0]]) == (2,),
            "query tail broadcast axis drift",
        )
        pad = self.node(self.arg(broadcast, 0), "pad", 2)
        self.shape(pad, "bf16", (640,))
        _require(
            re.findall(
                r"\bpadding=([^,\s]+)",
                _callee_attribute_text(self.boolean.ops[pad[0]].raw_line),
            )
            == ["576_0"],
            "query tail does not cover576:640",
        )
        self.constant(self.arg(pad, 1), "bf16", "-inf")
        zeros = self.node(self.arg(pad, 0), "broadcast", 1)
        self.shape(zeros, "bf16", (64,))
        _require(
            dimensions(self.boolean.ops[zeros[0]]) == (),
            "query tail not scalar broadcast",
        )
        self.constant(self.arg(zeros, 0), "bf16", "0")

    def query(self, value: Ref) -> tuple[Ref, Ref]:
        value = self.node(value, "convert", 1)
        self.shape(value, "bf16", (self.rows, 8, 640))
        outer = self.node(self.arg(value, 0), "maximum", 2)
        self.shape(outer, "f32", (self.rows, 8, 640))
        candidates = []
        for j in range(2):
            branch = self.arg(outer, j)
            if self.boolean.ops[branch[0]].opcode == "maximum":
                candidates.append(j)
        _require(len(candidates) == 1, "query packing maximum tree drift")
        inner = self.node(self.arg(outer, candidates[0]), "maximum", 2)
        self.shape(inner, "f32", (self.rows, 8, 640))
        self.zero_tail(self.arg(outer, 1 - candidates[0]))
        parts = {}
        for j in range(2):
            converted = self.node(self.arg(inner, j), "convert", 1)
            pad = self.node(self.arg(converted, 0), "pad", 2)
            source = self.arg(pad, 0)
            width = self.boolean.shape(source).dimensions[-1]
            _require(
                width in (64, 512) and width not in parts,
                "query component width/coverage drift",
            )
            parts[width] = self.padded(
                converted, width, "0_0x0_0x0_128" if width == 512 else "0_0x0_0x512_64"
            )
        absorbed, rotary = self.bridge(parts[512], "absorbed"), parts[64]
        _require(
            self.finite_live(absorbed) and self.finite_live(rotary),
            "actual sparse query component lacks LIVE finite guard",
        )
        return absorbed, rotary

    def padded_linear_input(self, op: HloInstruction) -> Ref:
        value = self.node(self.boolean.ref(op), "custom-call", 3)
        pad = self.node(self.arg(value, 0), "pad", 2)
        source = self.arg(pad, 0)
        padded = ((self.rows + 7) // 8) * 8
        dims = self.boolean.shape(source).dimensions
        _require(
            len(dims) == 2 and dims[0] == self.rows, "linear input row shape drift"
        )
        self.shape(source, "bf16", dims)
        self.shape(pad, "bf16", (padded, dims[1]))
        self.constant(self.arg(pad, 1), "bf16", "0")
        _require(
            re.findall(
                r"\bpadding=([^,\s]+)",
                _callee_attribute_text(self.boolean.ops[pad[0]].raw_line),
            )
            == [f"0_{padded-self.rows}x0_0"],
            "linear input zero padding drift",
        )
        return source

    def check(self) -> list[dict[str, Any]]:
        families = defaultdict(dict)
        padded = ((self.rows + 7) // 8) * 8
        suffixes = {
            "normalized": f"/greenfield_fp8_block_matmul_f32_m{padded}_k1536_n640/pallas_call",
            "prepared_query": f"/greenfield_fp8_block_matmul_m{padded}_k2048_n2048/pallas_call",
            "sparse": f"/greenfield_pregathered_sparse_mla_h8_k2048_b512_w640_prefill_m{self.rows}/pallas_call",
        }
        for op in self.index.module.instructions:
            if op.opcode != "custom-call":
                continue
            scope = op.op_name or ""
            for family, suffix in suffixes.items():
                if scope.endswith(suffix):
                    layers = _LAYER.findall(scope)
                    _require(len(layers) == 1, "health kernel lacks unique layer scope")
                    layer = int(layers[0])
                    _require(
                        layer not in families[family], "duplicate health kernel anchor"
                    )
                    families[family][layer] = op
        _require(
            set(families) == set(suffixes)
            and all(set(v) == set(range(78)) for v in families.values()),
            "missing78-layer actual operand anchors",
        )
        records = []
        for layer in range(78):
            record: dict[str, Any] = dict(layer=layer)
            for family in ("normalized", "prepared_query"):
                op = families[family][layer]
                source = self.padded_linear_input(op)
                _require(
                    self.finite_live(source),
                    f"layer{layer} actual {family} lacks LIVE finite guard",
                )
                record[family] = op.name
            op = families["sparse"][layer]
            call = self.node(self.boolean.ref(op), "custom-call", 3)
            absorbed, rotary = self.query(self.arg(call, 1))
            cache = self.bridge(self.arg(call, 2), "cache")
            _require(
                self.finite_live(cache),
                f"layer{layer} actual selected cache lacks LIVE finite guard",
            )
            record.update(
                sparse=op.name,
                absorbed=self.boolean.ops[absorbed[0]].name,
                rotary=self.boolean.ops[rotary[0]].name,
                cache=self.boolean.ops[cache[0]].name,
            )
            records.append(record)
        return records

    def check_grouped(self) -> list[dict[str, Any]]:
        """Bind outer source-valid guards, not inner active-tiles>0 guards.

        Zero active tiles on an expert owner is valid. The two actual nested
        conditionals distinguish this from metadata failure. The arithmetic
        producing valid counts/routes remains opaque here.
        """
        parents = defaultdict(list)
        calls = []
        for op in self.index.module.instructions:
            if op.opcode == "conditional":
                values = re.findall(
                    r"\bbranch_computations=\{([^}]*)\}",
                    _callee_attribute_text(op.raw_line),
                )
                if len(values) == 1:
                    names = [v.strip() for v in values[0].split(",")]
                    if len(names) == 2:
                        for branch, name in enumerate(names):
                            parents[name].append((op, branch))
            if op.opcode == "custom-call" and (op.op_name or "").endswith(
                "/greenfield_prefill_grouped_raw_fp8/pallas_call"
            ):
                calls.append(op)
        records = []
        for call in calls:
            layers = _LAYER.findall(call.op_name or "")
            _require(len(layers) == 1, "grouped call layer scope drift")
            immediate = parents[_computation_base(call.computation)]
            _require(
                len(immediate) == 1 and immediate[0][1] == 1,
                "grouped kernel not in sole active branch",
            )
            inner = immediate[0][0]
            enclosing = parents[_computation_base(inner.computation)]
            _require(
                len(enclosing) == 1 and enclosing[0][1] == 1,
                "grouped active guard not in sole valid branch",
            )
            outer = enclosing[0][0]
            _require(
                _computation_base(outer.computation) == "ENTRY",
                "grouped validity not bound at ENTRY",
            )
            predicate = self.health.predicate(self.boolean.ref(outer))
            _require(
                self.boolean.implies(self.known, predicate),
                "actual grouped source-valid selector does not gate commit",
            )
            records.append(
                dict(
                    layer=int(layers[0]),
                    kernel=call.name,
                    validity_guard=outer.name,
                    activity_guard=inner.name,
                )
            )
        _require(
            len(records) == 225
            and Counter(r["layer"] for r in records)
            == Counter({layer: 3 for layer in range(3, 78)})
            and len({r["validity_guard"] for r in records}) == 225,
            "grouped validity coverage drift",
        )
        return records

    def check_router(self) -> list[dict[str, Any]]:
        """Bind actual gathered logits and bias leaves, not routing arithmetic."""
        families = defaultdict(dict)
        for op in self.index.module.collectives:
            kind = None
            if op.opcode == "all-gather" and op.result_shapes == (
                HloShape("f32", (8, 32, self.rows)),
            ):
                kind = "logits"
            elif op.opcode == "all-reduce" and op.result_shapes == (
                HloShape("bf16", (self.rows, 1536)),
                HloShape("f32", (256,)),
            ):
                kind = "bias"
            if kind is None:
                continue
            layers = _LAYER.findall(op.op_name or "")
            _require(len(layers) == 1, "router collective layer scope drift")
            layer = int(layers[0])
            _require(layer not in families[kind], "duplicate router health anchor")
            families[kind][layer] = op
        _require(
            set(families) == {"logits", "bias"}
            and all(set(v) == set(range(3, 78)) for v in families.values()),
            "router health coverage drift",
        )
        records = []
        for layer in range(3, 78):
            logits, bias = families["logits"][layer], families["bias"][layer]
            for op, path, domain in ((logits, (), 2), (bias, (1,), None)):
                value = self.boolean.resolve(self.boolean.ref(op, path))
                domains = self.finite.get(value, set())
                _require(
                    None in domains or domain in domains,
                    f"layer{layer} actual router operand lacks finite guard",
                )
            records.append(dict(layer=layer, logits=logits.name, bias=bias.name))
        return records


def check_batched_operand_health(
    index: PrefillHloIndex, *, block_rows: int
) -> dict[str, Any]:
    if type(block_rows) is not int or block_rows not in (11, 17):
        raise ValueError("operand health profile requires B17/B11")
    report: dict[str, Any] = dict(
        passed=False,
        scope="SHORT_PREFILL_OPERAND_FINITE_AND_GROUPED_VALIDITY_GUARDS",
        not_proven=[
            "ROUTER_GROUPED_DSA_METADATA_ARITHMETIC",
            "ALL_MODEL_HEALTH_CHECKS_EXIST",
            "NUMERICAL_OR_MEMORY_ADMISSION",
        ],
    )
    try:
        proof = OperandHealthProof(index, block_rows)
        report.update(
            layers=proof.check(),
            grouped_validity=proof.check_grouped(),
            routers=proof.check_router(),
            finite_values=len(proof.finite),
            passed=True,
        )
    except (ValueError, KeyError, IndexError) as error:
        report["error"] = str(error)
    return report
