"""Own completed-WK, normalized gather and repaired writer-key provenance.

Dependency is not arithmetic equivalence; real-layer and own §21 evidence remain
required. This checks which producer supplies the key, not every operation on it.
"""

from __future__ import annotations

import re
from typing import Any

from .ws32_batched_cache_hlo import IndexCachePaths, LAYERS
from .ws32_batched_commit_hlo import _require, _shape
from .ws32_batched_helper_hlo import _concat_structure, _slice_users, _target
from .ws32_batched_moe_hlo import PrefillHloIndex
from .ws32_decoder import _group_family
from .ws32_hlo_leaf_dependencies import LeafDependencies
from .ws32_pallas_one_layer import _callee_attribute_text, _computation_base
from .ws32_prefill_hlo_identity import Value


_LAYER = re.compile(r"(?:^|/)greenfield_ws32_batched_prefill/layer_(\d+)(?:/|$)")


def _layer(op) -> int:
    matches = _LAYER.findall(op.op_name or "")
    _require(len(matches) == 1, "repair lacks unambiguous producer scope")
    return int(matches[0])


def check_batched_repair_lineage(
    index: PrefillHloIndex, *, block_rows: int
) -> dict[str, Any]:
    if type(block_rows) is not int or block_rows not in (11, 17):
        raise ValueError("repair lineage requires B17/B11")
    report: dict[str, Any] = dict(
        passed=False,
        scope="SHORT_PREFILL_REPAIR_INPUT_AND_WRITER_PROVENANCE",
        not_proven=[
            "ARITHMETIC_EQUIVALENCE",
            "REPEAT_INDEX_AND_ROTARY_VALUES",
            "HEALTH_IMPLICATION",
            "NUMERICAL_OR_MEMORY_ADMISSION",
        ],
    )
    paths = IndexCachePaths(index, block_rows)
    ssa = paths.ssa

    def forward(value: Value) -> Value:
        # Exact same-shape identity transposes only. No arbitrary layout bitcast.
        for _ in range(16):
            value = ssa.resolve(value, reconstruct=False, stop_at_shape_change=True)
            if value.op.opcode != "transpose" or value.path:
                return value
            source = ssa.operand(value, 0)
            rank = len(value.op.result_shapes[0].dimensions)
            _require(
                len(value.op.operand_names) == 1
                and value.op.result_shapes == source.op.result_shapes
                and re.findall(
                    r"\bdimensions=\{([^}]*)\}",
                    _callee_attribute_text(value.op.raw_line),
                )
                == [",".join(map(str, range(rank)))],
                "nonidentity repair transpose",
            )
            value = source
        raise ValueError("unbounded repair forwarding")

    try:
        _require(
            index.module.num_partitions == 32, "repair profile requires32 partitions"
        )
        projections = [
            op
            for op in index.module.instructions
            if _computation_base(op.computation) == "ENTRY"
            and op.opcode == "fusion"
            and "/m64_repair/while/body/closed_call/dot_general" in (op.op_name or "")
        ]
        _require(
            len(projections) == 21
            and sorted(_layer(p) for p in projections) == list(LAYERS),
            "expected21 actual repair projections",
        )
        projections = {_layer(p): p for p in projections}
        padded = ((block_rows + 7) // 8) * 8
        key_name = f"/greenfield_fp8_block_matmul_f32_m{padded}_k1536_n128/pallas_call"
        key_calls = [
            p
            for p in index.module.instructions
            if p.opcode == "custom-call"
            and _target(p) == "tpu_custom_call"
            and (p.op_name or "").endswith(key_name)
        ]
        _require(
            len(key_calls) == 21
            and sorted(_layer(p) for p in key_calls) == list(LAYERS),
            "expected21 actual raw index-key calls",
        )
        key_calls = {_layer(p): p for p in key_calls}
        concats = [
            p
            for p in index.module.instructions
            if p.opcode == "custom-call"
            and _target(p) == "ConcatBitcast"
            and _shape(Value(p), "f32", (128, 6144))
        ]
        users = _slice_users(index, concats)
        dependencies = LeafDependencies(index, {p.index for p in projections.values()})
        stores = paths.accepted_stacks()["repaired"]
        records = []
        for slot, layer in enumerate(LAYERS):
            projection = projections[layer]
            _require(
                len(projection.operand_names) == 2
                and [(s.dtype, s.dimensions) for s in projection.result_shapes]
                == [("f32", (64,)), ("f32", (64, 128))],
                "repair projection interface drift",
            )
            wk = ssa.resolve(Value(index.operand(projection, 1)), reconstruct=False)
            _require(
                _shape(Value(index.operand(projection, 1)), "f32", (128, 6144)),
                "repair WK shape drift",
            )
            if wk.op.opcode == "custom-call":
                _require(_target(wk.op) == "ConcatBitcast", "unknown WK transport")
                _concat_structure(index, wk.op, users)
                done = index.operand(wk.op, 0)
                start = index.operand(done, 0)
                wk = Value(index.operand(start, 0))
            _require(ssa.input(wk, 2324 + slot), "repair uses another completed WK")

            # Inspect actual multiplication, not merely dot_general metadata.
            root = Value(index.roots[index.callee(projection, "calls")])
            dot = forward(ssa.leaf(root, 1))
            _require(
                _shape(dot, "f32", (64, 128))
                and dot.op.opcode == "convolution"
                and len(dot.op.operand_names) == 2,
                "repair projection is not acquired matrix multiply",
            )
            attrs = _callee_attribute_text(dot.op.raw_line)
            _require(
                re.findall(r"\bdim_labels=([^,\s]+)", attrs) == ["bf_oi->bf"]
                and re.findall(r"\boperand_precision=\{([^}]*)\}", attrs)
                == ["default,highest"],
                "repair matrix dimensions/precision drift",
            )
            # Compiler memory-space-only bitcast wrappers do not change tensor
            # indices: require exact dtype/dims and row-major tile annotations.
            for operand, (dtype, dims, tile) in enumerate(
                (("bf16", (64, 6144), "(8,128)(2,1)"), ("f32", (128, 6144), "(8,128)"))
            ):
                value = forward(ssa.operand(dot, operand))
                _require(
                    value.op.opcode == "bitcast" and _shape(value, dtype, dims),
                    "repair dot wrapper drift",
                )
                source = ssa.operand(value, 0)
                for v in (value, source):
                    pattern = (
                        r"=\s*"
                        + dtype
                        + r"\["
                        + ",".join(map(str, dims))
                        + r"\]\{1,0:T"
                        + re.escape(tile)
                        + r"(?:S\(\d+\))?\}"
                    )
                    _require(
                        re.search(pattern, v.op.raw_line) is not None,
                        "repair dot layout drift",
                    )
                origin = ssa.resolve(source)
                _require(
                    origin.op.opcode == "parameter"
                    and not origin.path
                    and re.findall(r"\bparameter\((\d+)\)", origin.op.raw_line)
                    == [str(operand)],
                    "repair dot uses wrong projection parameter",
                )

            repeated = forward(Value(index.operand(projection, 0)))
            _require(
                _shape(repeated, "bf16", (64, 6144))
                and repeated.op.opcode == "gather"
                and len(repeated.op.operand_names) == 2,
                "repair normalized repetition is not gather",
            )
            gather = forward(ssa.operand(repeated, 0))
            _require(
                _shape(gather, "bf16", (block_rows, 6144))
                and gather.op.raw_opcode == "all-gather"
                and _layer(gather.op) == layer
                and _group_family(gather.op) == "feature"
                and len(gather.op.replica_groups) == 8
                and gather.op.use_global_device_ids
                and re.findall(
                    r"\bdimensions=\{([^}]*)\}",
                    _callee_attribute_text(gather.op.raw_line),
                )
                == ["1"]
                and "/repair_normalized_feature_gather/" in (gather.op.op_name or ""),
                "repair reads another producer normalized gather",
            )
            _require(
                len(gather.op.operand_names) == 1
                and _shape(
                    Value(index.operand(gather.op, 0)), "bf16", (block_rows, 1536)
                ),
                "repair normalized source shape drift",
            )
            raw_key = key_calls[layer]
            pad = index.operand(raw_key, 0)
            _require(
                pad.opcode == "pad"
                and len(pad.operand_names) == 2
                and _shape(Value(pad), "bf16", (padded, 1536))
                and _shape(Value(index.operand(pad, 0)), "bf16", (block_rows, 1536))
                and re.findall(
                    r"\bpadding=([^,\s]+)", _callee_attribute_text(pad.raw_line)
                )
                == [f"0_{padded-block_rows}x0_0"]
                and ssa.constant(Value(index.operand(pad, 1)), "bf16", 0),
                "raw index-key input lacks exact zero row padding",
            )
            _require(
                index.operand(pad, 0).index == index.operand(gather.op, 0).index,
                "repair gather and own index-key projection use different normalized inputs",
            )

            store = stores[slot]
            conditional = store["conditional"]
            _require(
                _computation_base(conditional.op.computation) == "ENTRY"
                and not conditional.bindings,
                "writer is not actual ENTRY conditional",
            )
            arguments = index.operand(conditional.op, 2)
            _require(
                arguments.opcode == "tuple" and len(arguments.operand_names) == 5,
                "repair writer tuple drift",
            )
            key = Value(index.operand(arguments, 3))
            update = forward(store["updates"])
            _require(
                update.op.opcode == "select"
                and len(update.op.operand_names) == 3
                and ssa.same(ssa.operand(update, 1), key),
                "repair key does not supply actual scatter update",
            )
            terminals = dependencies.dependencies(key.op)
            actual = {item for item in terminals if item[0] == "terminal"}
            _require(
                actual
                == {
                    ("terminal", projection.index, (0,)),
                    ("terminal", projection.index, (1,)),
                },
                f"slot{slot}: missing own or mixed repair projection leaves",
            )
            records.append(
                dict(
                    slot=slot,
                    layer=layer,
                    projection=projection.name,
                    normalized_gather=gather.op.name,
                    wk_entry_leaf=2324 + slot,
                    writer=conditional.op.name,
                )
            )
        report.update(
            passed=True, producers=records, dependency_nodes=len(dependencies.memo)
        )
    except (ValueError, KeyError, IndexError, AttributeError) as error:
        report["error"] = str(error)
    return report
