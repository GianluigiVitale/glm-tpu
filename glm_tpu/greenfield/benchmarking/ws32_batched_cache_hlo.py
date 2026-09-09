"""Actual accepted index-cache stacks and own-slot conditional row writers.

This bounds storage ownership only. Key arithmetic, address/mask values and
all-layer health need independent checks before the full profile can pass.
"""

from __future__ import annotations

import re
from typing import Any

from .ws32_batched_commit_hlo import _require, _shape
from .ws32_batched_moe_hlo import PrefillHloIndex
from .ws32_prefill_hlo_identity import PrefillIdentity, Value
from .ws32_pallas_one_layer import _callee_attribute_text


LAYERS = (0, 1, 2, *range(6, 78, 4))
STACK = (21, 16, 64, 128)
SLOT = (1, 16, 64, 128)
FLAT = (1024, 128)


def _slot_shape(value: Value) -> bool:
    return _shape(value, "bf16", SLOT) or (
        value.op.opcode == "conditional"
        and value.path == (1,)
        and [(s.dtype, s.dimensions) for s in value.op.result_shapes]
        == [("bf16", (16, 64, 128)), ("bf16", SLOT)]
    )


class IndexCachePaths:
    """Bind cache roots through existing selected-leaf forwarding.

    Array-layout changes are boundaries, not generic identities. Only the two
    explicit row-major cache flatten/unflatten bitcasts are traversed here.
    """

    def __init__(self, index: PrefillHloIndex, rows: int) -> None:
        self.index, self.rows = index, rows
        self.ssa = PrefillIdentity(index)

    def resolve(self, value: Value) -> Value:
        return self.ssa.resolve(value, stop_at_shape_change=True)

    def arg(self, value: Value, position: int) -> Value:
        return self.resolve(self.ssa.operand(value, position))

    def node(self, value: Value, opcode: str, arity: int) -> Value:
        value = self.resolve(value)
        _require(
            not value.path
            and value.op.opcode == opcode
            and len(value.op.operand_names) == arity,
            f"expected {opcode}/{arity}, got {value.op.name}:{value.op.opcode}{value.path}",
        )
        return value

    def layout(self, value: Value, dims: tuple[int, ...]) -> None:
        _require(_shape(value, "bf16", dims), "cache bitcast shape/dtype drift")
        # Tiled minor dimensions are unchanged; only contiguous leading row
        # dimensions collapse/expand. Memory-space annotations do not reorder.
        minor = ",".join(map(str, reversed(range(len(dims)))))
        pattern = (
            r"=\s*bf16\["
            + ",".join(map(str, dims))
            + r"\]\{"
            + re.escape(minor)
            + r":T\(8,128\)\(2,1\)(?:S\(\d+\))?\}"
        )
        _require(
            re.search(pattern, value.op.raw_line) is not None,
            "cache bitcast is not acquired contiguous tiled layout",
        )

    def bridge(
        self, value: Value, output: tuple[int, ...], source: tuple[int, ...]
    ) -> Value:
        value = self.node(value, "bitcast", 1)
        arg = self.ssa.operand(value, 0)
        self.layout(value, output)
        self.layout(arg, source)
        return self.resolve(arg)

    def own_slice(self, value: Value, slot: int, original: int) -> None:
        value = self.node(value, "slice", 1)
        _require(_shape(value, "bf16", SLOT), "old cache slice shape drift")
        ranges = re.findall(
            r"\bslice=\{([^}]*)\}", _callee_attribute_text(value.op.raw_line)
        )
        expected = f"[{slot}:{slot+1}],[0:16],[0:64],[0:128]"
        _require(
            len(ranges) == 1 and re.sub(r"\s", "", ranges[0]) == expected,
            "row writer reads another cache slot",
        )
        base = self.arg(value, 0)
        # Earlier producers may already have committed OTHER outer slots.
        # Such writes cannot change this slot; prove that rather than requiring
        # the compiler to hoist every slice directly from the ENTRY input.
        for _ in range(22):
            if self.ssa.input(base, original):
                return
            base = self.node(base, "dynamic-update-slice", 6)
            _require(
                _shape(base, "bf16", STACK) and _slot_shape(self.arg(base, 1)),
                "old cache has non-slot write",
            )
            _require(
                any(
                    self.ssa.constant(self.ssa.operand(base, 2), "s32", earlier)
                    for earlier in range(slot)
                )
                and all(
                    self.ssa.constant(self.ssa.operand(base, axis), "s32", 0)
                    for axis in (3, 4, 5)
                ),
                "old cache slot was overwritten or displaced",
            )
            base = self.arg(base, 0)
        raise ValueError("old cache slice has unbounded preceding writes")

    def row_writer(self, value: Value, slot: int, original: int) -> dict[str, Any]:
        conditional = self.resolve(value)
        _require(
            conditional.op.opcode == "conditional"
            and len(conditional.op.operand_names) == 3
            and conditional.path in ((), (1,)),
            "row conditional/leaf drift",
        )
        shapes = conditional.op.result_shapes
        _require(
            (not conditional.path and _shape(conditional, "bf16", SLOT))
            or (
                conditional.path == (1,)
                and len(shapes) == 2
                and [(s.dtype, s.dimensions) for s in shapes]
                == [("bf16", (16, 64, 128)), ("bf16", SLOT)]
            ),
            "row conditional shape drift",
        )
        old = self.resolve(self.ssa.branch(conditional, 0))
        self.own_slice(old, slot, original)
        written = self.ssa.branch(conditional, 1)
        flat = self.bridge(written, SLOT, FLAT)
        scatter = self.node(flat, "scatter", 3)
        _require(_shape(scatter, "bf16", FLAT), "row scatter shape drift")
        back = self.bridge(self.ssa.operand(scatter, 0), FLAT, SLOT)
        _require(self.ssa.same(back, old), "row scatter modifies a different old slice")
        self.replacement_scatter(scatter, width=128)
        # Return real bound values for the subsequent key/address proof, not
        # metadata labels or whole-tuple dependencies.
        return dict(
            slot=slot,
            layer=LAYERS[slot],
            conditional=conditional,
            scatter=scatter,
            indices=self.ssa.operand(scatter, 1),
            updates=self.ssa.operand(scatter, 2),
        )

    def replacement_scatter(self, scatter: Value, *, width: int) -> None:
        """Shared row replacement contract, without inferring address correctness."""
        _require(_shape(scatter, "bf16", (1024, width)), "row scatter shape drift")
        attrs = _callee_attribute_text(scatter.op.raw_line)
        for attr, expected in (
            ("update_window_dims", "1"),
            ("inserted_window_dims", "0"),
            ("scatter_dims_to_operand_dims", "0"),
        ):
            _require(
                re.findall(r"\b" + attr + r"=\{([^}]*)\}", attrs) == [expected],
                "row scatter dimensions drift",
            )
        _require(
            re.findall(r"\bindex_vector_dim=(\d+)", attrs) == ["1"],
            "row scatter index-vector dimension drift",
        )
        reducer = self.index.callee(scatter.op, "to_apply")
        nodes = list(self.index.computations[reducer].values())
        params = [p for p in nodes if p.opcode == "parameter"]
        _require(
            len(nodes) in (2, 3)
            and all(
                p.opcode in ("parameter", "bitcast") and _shape(Value(p), "bf16", ())
                for p in nodes
            )
            and len(params) == 2
            and sorted(re.findall(r"\bparameter\((\d+)\)", p.raw_line) for p in params)
            == [["0"], ["1"]],
            "scatter reducer has unexpected arithmetic",
        )
        root = self.resolve(Value(self.index.roots[reducer]))
        _require(
            root.op.opcode == "parameter"
            and re.findall(r"\bparameter\((\d+)\)", root.op.raw_line) == ["1"],
            "row scatter does not return the update",
        )
        _require(
            _shape(self.ssa.operand(scatter, 1), "s32", (self.rows,))
            and _shape(self.ssa.operand(scatter, 2), "bf16", (self.rows, width)),
            "row scatter indices/update dimensions drift",
        )

    def stack(self, value: Value, original: int) -> list[dict[str, Any]]:
        records = []
        for slot in reversed(range(21)):
            value = self.node(value, "dynamic-update-slice", 6)
            _require(_shape(value, "bf16", STACK), "outer cache stack shape drift")
            for arg, constant in ((2, slot), (3, 0), (4, 0), (5, 0)):
                _require(
                    self.ssa.constant(self.ssa.operand(value, arg), "s32", constant),
                    "outer cache slot/axis drift",
                )
            records.append(self.row_writer(self.ssa.operand(value, 1), slot, original))
            value = self.arg(value, 0)
        _require(self.ssa.input(value, original), "cache stack has wrong original base")
        return list(reversed(records))

    def accepted_stacks(self) -> dict[str, list[dict[str, Any]]]:
        root = Value(self.index.roots["ENTRY"])
        first = self.resolve(self.ssa.leaf(root, 0))
        _require(first.path == (0,), "missing atomic commit KV slot")
        commit = Value(first.op, bindings=first.bindings)
        accepted = self.ssa.branch(commit, 1)
        promotion = self.node(self.ssa.leaf(accepted, 1), "conditional", 3)
        repaired = self.stack(self.ssa.leaf(accepted, 8), 11)
        unrepaired = self.stack(self.ssa.branch(promotion, 0), 3)
        return dict(repaired=repaired, unrepaired=unrepaired)


def check_batched_index_cache_stacks(
    index: PrefillHloIndex, *, block_rows: int
) -> dict[str, Any]:
    """Own original slot, row writer and outer commit stacks for both caches."""
    if type(block_rows) is not int or block_rows not in (11, 17):
        raise ValueError("cache profile requires B17/B11")
    report: dict[str, Any] = dict(
        passed=False,
        scope="SHORT_PREFILL_INDEX_CACHE_STORAGE_OWNERSHIP",
        not_proven=[
            "COMMIT_PREDICATE_AND_ROLLBACK",
            "KEY_PROJECTION_AND_ROTARY_ARITHMETIC",
            "ROW_INDICES_AND_MASK_VALUES",
            "ALL_LAYER_HEALTH",
            "NUMERICAL_OR_MEMORY_ADMISSION",
        ],
    )
    try:
        _require(
            index.module.num_partitions == 32, "cache profile requires32partitions"
        )
        stacks = IndexCachePaths(index, block_rows).accepted_stacks()
        report.update(
            passed=True,
            stacks={
                name: [
                    dict(
                        slot=r["slot"],
                        layer=r["layer"],
                        writer=r["conditional"].op.name,
                        scatter=r["scatter"].op.name,
                    )
                    for r in records
                ]
                for name, records in stacks.items()
            },
        )
    except (ValueError, KeyError, IndexError) as error:
        report["error"] = str(error)
    return report
