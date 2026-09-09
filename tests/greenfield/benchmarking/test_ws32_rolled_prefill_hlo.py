"""Actual DB602 transitions and adversarial mutations; no TPU initialization."""

from dataclasses import replace
from hashlib import sha256
import json
import re
from pathlib import Path

import pytest

from glm_tpu.greenfield.benchmarking.ws32_batched_commit_hlo import check_batched_commit
from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import (
    PrefillHloIndex,
    check_batched_moe_route_sums,
)
from glm_tpu.greenfield.benchmarking.ws32_pallas_one_layer import (
    _live_instruction_closure,
)
from glm_tpu.greenfield.benchmarking.ws32_prefill_hlo_identity import Value
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_hlo import (
    RolledTransitions,
    RolledTileHealth,
    check_rolled_commit,
    check_rolled_loops,
    check_rolled_route_sums,
    check_rolled_tile_health,
)
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from glm_tpu.greenfield.benchmarking.ws32_batched_helper_hlo import _target
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_cache_hlo import (
    RolledCachePaths,
    check_rolled_cache_paths,
)
from glm_tpu.greenfield.benchmarking.ws32_prefill_hlo_identity import attribute
from glm_tpu.greenfield.benchmarking.ws32_batched_health_hlo import WriterHealthProof
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_last_row_hlo import (
    check_rolled_last_row_indices,
)


@pytest.mark.parametrize("rows", [False, True, 0, 1, 17, 32, 113, 115, 129, 128.0])
def test_distinct_rolled_geometry(rows):
    for check in (
        check_rolled_loops,
        check_rolled_commit,
        check_rolled_route_sums,
        check_rolled_tile_health,
    ):
        with pytest.raises(ValueError, match="exactly B128 or B114"):
            check(None, block_rows=rows, live_instructions=())


@pytest.mark.parametrize("rows", [114, 128])
def test_historical_public_guards_stay_closed(rows):
    for check in (check_batched_commit, check_batched_moe_route_sums):
        with pytest.raises(ValueError, match="requires1..32"):
            check(None, block_rows=rows, live_instructions=())


@pytest.fixture(scope="module", params=[("prefill_chunk", 128), ("prefill_tail", 114)])
def original(request):
    graph, rows = request.param
    repo = Path(__file__).resolve().parents[3]
    receipt = json.loads(
        (
            repo
            / "docs/artifacts/prefill-rolled-model-compile-db602-sealed-20260909.json"
        ).read_text()
    )
    path = (
        Path("/home/gianl/glm-run")
        / receipt["tag"]
        / "fleet/rank0"
        / (graph + ".optimized_hlo.txt")
    )
    if not path.exists():
        pytest.skip("DB602 original compiler artifact unavailable")
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == receipt["programs"][graph]["optimized_hlo_sha256"]
    module = parse_hlo_module(raw.decode())
    del raw
    index = PrefillHloIndex(module)
    live = _live_instruction_closure(module.instructions)
    adapter = RolledTransitions(index, rows)
    loops = adapter.all_loops(live)
    return index, rows, live, adapter, loops


def rewrite(index, op, **changes):
    changed = replace(op, **changes)
    return PrefillHloIndex(
        replace(
            index.module,
            instructions=tuple(
                changed if p.index == op.index else p for p in index.module.instructions
            ),
        )
    )


def test_actual_three_families_complete_stacks_and_route_commit(original):
    index, rows, live, adapter, loops = original
    assert [v.layer for v in loops] == list(range(78))
    assert [v.health_slot for v in loops].count(6) == 2
    assert [v.health_slot for v in loops].count(9) == 19
    assert [v.health_slot for v in loops].count(4) == 57
    assert sum(len(v.cache_slots) for v in loops) == 120
    assert sum(len(v.stack_slots) for v in loops) == 291
    route = check_rolled_route_sums(index, block_rows=rows, live_instructions=live)
    assert route["passed"], route
    assert len(route["layers"]) == 75
    anchors = {}
    commit = check_rolled_commit(
        index, block_rows=rows, live_instructions=live, anchors=anchors
    )
    assert commit["passed"], commit
    assert set(anchors) == {"commit", "accepted", "local_health", "consensus", "final"}
    # Mutable cache/stack outputs must NEVER resolve to the initial tuple.
    for loop in loops:
        for slot in (*loop.cache_slots, *loop.stack_slots, 0):
            result = adapter.ssa.resolve(Value(loop.loop, (slot,)))
            assert result.op.index == loop.loop.index and result.path == (slot,)


@pytest.mark.parametrize(
    "case",
    [
        "initial_counter",
        "counter_step",
        "trip_limit",
        "stack_index",
        "stack_base",
        "stack_width",
        "invariant_source",
        "cache_rollback",
        "dead_loop",
        "duplicate_layer",
    ],
)
def test_actual_recurrence_mutations_refuse(original, case):
    index, rows, live, adapter, loops = original
    loop = loops[0]
    root = loop.root
    if case == "initial_counter":
        op = adapter.resolve(adapter.ssa.leaf(loop.initial, 0)).op
        altered = rewrite(
            index, op, raw_line=op.raw_line.replace("constant(0)", "constant(1)")
        )
    elif case == "counter_step":
        op = adapter.resolve(adapter.ssa.leaf(root, 0)).op
        altered = rewrite(
            index,
            op,
            opcode="subtract",
            raw_opcode="subtract",
            raw_line=op.raw_line.replace(" add(", " subtract("),
        )
    elif case == "trip_limit":
        op = index.roots[index.callee(loop.loop, "condition")]
        altered = rewrite(
            index, op, raw_line=op.raw_line.replace("direction=LT", "direction=LE")
        )
    elif case in ("stack_index", "stack_base", "stack_width"):
        value = adapter.resolve(adapter.ssa.leaf(root, loop.health_slot))
        op = value.op
        if case == "stack_width":
            # Mutate the actual update producer, not just the annotation on use.
            producer = index.operand(op, 1)
            altered = rewrite(
                index,
                producer,
                result_shapes=(replace(producer.result_shapes[0], dimensions=(1, 16)),),
            )
        else:
            operands = list(op.operand_names)
            if case == "stack_index":
                operands[2] = operands[-1]
            else:
                operands[0] = operands[1]
            altered = rewrite(index, op, operand_names=tuple(operands))
    elif case == "invariant_source":
        op = root.op
        operands = list(op.operand_names)
        # Float cache slot has a different shape from scalar invariant; either
        # shape/own-input induction must reject, not bless generic loop copying.
        operands[loop.health_slot + 1] = operands[1]
        altered = rewrite(index, op, operand_names=tuple(operands))
    elif case == "cache_rollback":
        value = adapter.resolve(adapter.ssa.leaf(root, 1))
        op = value.op
        operands = list(op.operand_names)
        operands[1] = operands[2]
        altered = rewrite(index, op, operand_names=tuple(operands))
    elif case == "dead_loop":
        altered = index
        live = tuple(p for p in live if p.index != loop.loop.index)
    else:
        op = loops[-1].loop
        altered = rewrite(
            index, op, op_name=op.op_name.replace("layer_77/", "layer_0/")
        )
    result = check_rolled_loops(altered, block_rows=rows, live_instructions=live)
    assert not result["passed"], (case, result)


@pytest.mark.parametrize("case", ["mutable_same_shape", "different_initial_same_shape"])
def test_actual_invariant_partition_and_initial_identity(original, case):
    index, rows, live, adapter, loops = original
    loop = loops[0]
    sizes = loop.loop.result_shapes
    invariants = range(loop.health_slot + 1, len(sizes))
    if case == "mutable_same_shape":
        dst, src = next(
            (d, s) for d in invariants for s in loop.stack_slots if sizes[d] == sizes[s]
        )
    else:
        dst, src = next(
            (d, s)
            for d in invariants
            for s in invariants
            if sizes[d] == sizes[s]
            and not adapter.ssa.same(
                adapter.ssa.leaf(loop.initial, d), adapter.ssa.leaf(loop.initial, s)
            )
        )
    operands = list(loop.root.op.operand_names)
    operands[dst] = operands[src]
    changed = rewrite(index, loop.root.op, operand_names=tuple(operands))
    result = check_rolled_loops(changed, block_rows=rows, live_instructions=live)
    assert not result["passed"], (case, dst, src, result)


@pytest.mark.parametrize(
    "case", ["count_source", "tile_multiplier", "iota", "count_cap"]
)
def test_actual_tile_count_mutations(original, case):
    index, rows, live, adapter, loops = original
    loop = loops[0]
    count = adapter.tile_counts(loop)[0]
    maximum = next(
        adapter.arg(count, j)
        for j in (0, 1)
        if adapter.arg(count, j).op.opcode == "maximum"
    )
    subtract = next(
        adapter.arg(maximum, j)
        for j in (0, 1)
        if adapter.arg(maximum, j).op.opcode == "subtract"
    )
    if case in ("count_source", "count_cap"):
        carried = adapter.arg(subtract, 0)
        op = adapter.resolve(adapter.ssa.leaf(loop.initial, carried.path[0])).op
        operands = list(op.operand_names)
        # clip cap -> 0 is not another accepted external-count expression.
        if case == "count_cap":
            constant = next(
                index.operand(op, j)
                for j in (0, 1)
                if index.operand(op, j).opcode == "constant"
            )
            altered = rewrite(
                index,
                constant,
                raw_line=constant.raw_line.replace(f"constant({rows})", "constant(0)"),
            )
        else:
            # Replacing the count side by the cap must not be accepted.
            constant = next(
                j for j in (0, 1) if index.operand(op, j).opcode == "constant"
            )
            operands[1 - constant] = operands[constant]
            altered = rewrite(index, op, operand_names=tuple(operands))
    else:
        multiply = adapter.arg(subtract, 1)
        args = [adapter.arg(multiply, j) for j in (0, 1)]
        scalar = next(v for v in args if v.op.opcode != "dynamic-slice")
        dynamic = next(v for v in args if v.op.opcode == "dynamic-slice")
        if case == "tile_multiplier":
            op = adapter.resolve(adapter.ssa.leaf(loop.initial, scalar.path[0])).op
            altered = rewrite(
                index,
                op,
                raw_line=op.raw_line.replace("{32}", "{16}").replace("(32)", "(16)"),
            )
        else:
            carried = adapter.arg(dynamic, 0)
            op = adapter.resolve(adapter.ssa.leaf(loop.initial, carried.path[0])).op
            altered = rewrite(
                index,
                op,
                raw_line=op.raw_line.replace("iota_dimension=0", "iota_dimension=1"),
            )
    result = check_rolled_loops(altered, block_rows=rows, live_instructions=live)
    assert not result["passed"], (case, result)


def test_known_copy_identity_requires_ascending_slices(original):
    index, rows, _, adapter, _ = original
    op = next(
        p
        for p in index.module.instructions
        if p.opcode == "custom-call"
        and _target(p) == "ConcatBitcast"
        and [(s.dtype, s.dimensions) for s in p.result_shapes] == [("u8", (2048, 2048))]
    )
    source = index.operand(index.operand(index.operand(op, 0), 0), 0)
    assert adapter.ssa.same(Value(op), Value(source))
    operands = list(op.operand_names)
    operands[0], operands[1] = operands[1], operands[0]
    changed = rewrite(index, op, operand_names=tuple(operands))
    with pytest.raises(ValueError, match="slices reordered"):
        mutated = next(p for p in changed.module.instructions if p.index == op.index)
        RolledTransitions(changed, rows).ssa.resolve(Value(mutated))


def test_actual_nonempty_tile_health_binds_all120_writer_predicates(original):
    index, rows, live, _, _ = original
    report = check_rolled_tile_health(index, block_rows=rows, live_instructions=live)
    assert report["passed"], report
    assert len(report["layers"]) == 78
    assert sum(len(layer["writers"]) for layer in report["layers"]) == 120
    assert report["assumptions"] == ["TILE_COUNT_GT_ZERO", "ALL_LIVE_TILE_HEALTH_TRUE"]
    assert "GLOBAL_COMMIT_TO_TILE_HEALTH" in report["not_proven"]
    assert "EMPTY_TILE_NO_WRITES" in report["not_proven"]
    assert json.loads(json.dumps(report)) == report


@pytest.mark.parametrize("layer", [0, 3, 6])
def test_tile_health_cannot_replace_conjunction_with_disjunction(original, layer):
    index, rows, live, adapter, loops = original
    loop = loops[layer]
    stack = adapter.resolve(adapter.ssa.leaf(loop.root, loop.health_slot))
    op = adapter.arg(stack, 1).op
    assert op.opcode == "and"
    changed = rewrite(
        index,
        op,
        opcode="or",
        raw_opcode="or",
        raw_line=op.raw_line.replace(" and(", " or("),
    )
    transitions = RolledTransitions(changed, rows)
    actual = transitions.all_loops(live)[layer]
    report = RolledTileHealth(transitions).inspect(actual)
    assert not report["passed"], report


def test_proof_limitations_are_explicit(original):
    index, rows, live, _, _ = original
    report = check_rolled_loops(index, block_rows=rows, live_instructions=live)
    assert report["passed"], report
    assert "TILE_COUNT_TO_WRITER_AND_HEALTH_CONSUMERS" in report["not_proven"]
    assert "STACK_TO_COMMIT_HEALTH" in report["not_proven"]
    assert "NUMERICAL_CORRECTNESS_OR_MEMORY" in report["not_proven"]
    assert json.loads(json.dumps(report)) == report


def test_tile_health_callback_cache_is_independent_of_loop_order(original):
    _, _, _, adapter, loops = original
    shared = RolledTileHealth(adapter)
    for layer in (77, 6, 3, 0, 6, 77):
        assert shared.inspect(loops[layer]) == RolledTileHealth(adapter).inspect(
            loops[layer]
        )


def test_actual_all_cache_paths_and_inactive_rows(original):
    index, rows, live, _, _ = original
    report = check_rolled_cache_paths(index, block_rows=rows, live_instructions=live)
    assert report["passed"], report
    assert len(report["writers"]) == 120
    assert {k: len(v) for k, v in report["stacks"].items()} == {
        "kv": 78,
        "unrepaired": 21,
        "repaired": 21,
    }
    assert "ACTIVE_ROW_ADDRESS_ARITHMETIC" in report["not_proven"]
    assert "GLOBAL_COMMIT_HEALTH" in report["not_proven"]
    assert json.loads(json.dumps(report)) == report


@pytest.mark.parametrize(
    "case",
    [
        "initial_slice",
        "accepted_base",
        "outer_layer",
        "replacement",
        "sentinel",
        "normalize",
        "live_mask",
    ],
)
def test_actual_cache_path_mutations_refuse(original, case):
    index, rows, live, adapter, loops = original
    paths = RolledCachePaths(adapter)
    loop = loops[6]
    record = paths.writer(loop, 2)
    if case == "initial_slice":
        initial = paths.resolve(paths.ssa.leaf(loop.initial, 2))
        op = paths.arg(initial, 0).op
        altered = rewrite(index, op, raw_line=op.raw_line.replace("[3:4]", "[2:3]"))
        assert altered.module.instructions != index.module.instructions
    elif case == "accepted_base":
        # Unrepaired and repaired carried caches have identical physical shapes.
        # Redirect ONLY the accepted branch base to the other cache.
        arg = paths.ssa.operand(record["conditional"], 2)
        assert arg.op.opcode == "tuple"
        other = next(
            p
            for p in index.computations[index.callee(loop.loop, "body")].values()
            if p.opcode == "get-tuple-element"
            and p.operand_names == (loop.parameter.op.name,)
            and attribute(p, "index") == "3"
        )
        operands = list(arg.op.operand_names)
        operands[0] = other.name
        altered = rewrite(index, arg.op, operand_names=tuple(operands))
    elif case == "outer_layer":
        # Same-shaped final KV from layer76 cannot replace layer77's value.
        op = next(
            p
            for p in index.module.instructions
            if p.opcode == "get-tuple-element"
            and p.operand_names == (loops[77].loop.name,)
            and attribute(p, "index") == "1"
        )
        altered = rewrite(index, op, operand_names=(loops[76].loop.name,))
    elif case == "replacement":
        callee = index.callee(record["scatter"].op, "to_apply")
        op = index.roots[callee]
        # Return the old element rather than the update, preserving BF16 shape.
        altered = rewrite(
            index, op, raw_line=op.raw_line.replace("parameter(1)", "parameter(0)")
        )
    else:
        indices = paths.resolve(record["indices"])
        normalized = paths.arg(indices, 0)
        masked = paths.arg(normalized, 2)
        if case == "sentinel":
            op = paths.arg(paths.arg(masked, 2), 0).op
            altered = rewrite(
                index, op, raw_line=op.raw_line.replace("constant(1024)", "constant(0)")
            )
        elif case == "normalize":
            op = normalized.op
            operands = list(op.operand_names)
            operands[1], operands[2] = operands[2], operands[1]
            altered = rewrite(index, op, operand_names=tuple(operands))
        else:
            mask = paths.arg(masked, 0)
            op = next(
                paths.arg(mask, j).op
                for j in (0, 1)
                if paths.arg(mask, j).op.opcode == "compare"
                and attribute(paths.arg(mask, j).op, "direction") == "LT"
            )
            altered = rewrite(
                index, op, raw_line=op.raw_line.replace("direction=LT", "direction=GE")
            )
    result = check_rolled_cache_paths(altered, block_rows=rows, live_instructions=live)
    assert not result["passed"], (case, result)


@pytest.fixture(scope="module")
def last_row(original):
    index, rows, live, adapter, loops = original
    report = check_rolled_last_row_indices(
        index, block_rows=rows, live_instructions=live
    )
    assert report["passed"], report
    assert len(report["reads"]) == 3
    assert "SELECTED_ARRAY_VALUES_AND_PADDING_PREDICATES" in report["not_proven"]
    assert "FINAL_HEAD_INPUT_ROW" in report["not_proven"]
    assert json.loads(json.dumps(report)) == report
    anchors = {}
    assert check_rolled_commit(
        index, block_rows=rows, live_instructions=live, anchors=anchors
    )["passed"]
    return (*original, report, anchors)


@pytest.mark.parametrize(
    "case", ["offset", "count", "normalization", "column", "size", "count_bound"]
)
def test_actual_final_row_address_mutations_refuse(last_row, case):
    index, rows, live, adapter, loops, report, anchors = last_row
    value = adapter.resolve(adapter.ssa.leaf(anchors["accepted"], 2))
    if rows == 114:
        for _ in range(3):
            value = adapter.arg(value, 1)
    assert value.op.name == report["reads"][0]["read"]
    normalized = adapter.arg(value, 1)
    last = adapter.arg(normalized, 2)
    difference = adapter.arg(last, 0)
    if case == "offset":
        op = adapter.arg(difference, 1).op
        assert "constant(-1)" in op.raw_line
        altered = rewrite(
            index, op, raw_line=op.raw_line.replace("constant(-1)", "constant(-2)")
        )
    elif case in ("count", "count_bound"):
        count = adapter.arg(difference, 0)
        if case == "count_bound":
            op = adapter.arg(count, 0).op
            assert f"constant({rows})" in op.raw_line
            altered = rewrite(
                index,
                op,
                raw_line=op.raw_line.replace(
                    f"constant({rows})", f"constant({rows-1})"
                ),
            )
        else:
            clamped = adapter.arg(count, 1)
            entry = adapter.arg(clamped, 1)
            assert adapter.ssa.input(entry, 1)
            # Change only the count's GTE to another real S32 scalar input.
            op = index.operand(clamped.op, 1)
            while op.opcode == "copy":
                op = index.operand(op, 0)
            assert attribute(op, "index") == "1"
            altered = rewrite(
                index, op, raw_line=re.sub(r"\bindex=1\b", "index=12", op.raw_line)
            )
    elif case == "normalization":
        op = normalized.op
        args = list(op.operand_names)
        args[1], args[2] = args[2], args[1]
        altered = rewrite(index, op, operand_names=tuple(args))
    elif case == "column":
        op = value.op
        args = list(op.operand_names)
        args[2] = args[1]
        altered = rewrite(index, op, operand_names=tuple(args))
    else:
        op = value.op
        altered = rewrite(
            index,
            op,
            raw_line=op.raw_line.replace(
                "dynamic_slice_sizes={1,2048}", "dynamic_slice_sizes={1,2047}"
            ),
        )
    assert altered.module.instructions != index.module.instructions
    result = check_rolled_last_row_indices(
        altered, block_rows=rows, live_instructions=live
    )
    assert not result["passed"], (case, result)


def test_last_row_requires_nonempty_count_in_actual_commit(last_row):
    index, rows, live, _, _, _, _ = last_row
    proof = WriterHealthProof(index, rows)
    vote = proof.frontier()
    b = proof.boolean
    op = next(
        b.ops[ref[0]]
        for ref, _ in b.factors(vote)
        if not ref[1]
        and b.ops[ref[0]].opcode == "compare"
        and attribute(b.ops[ref[0]], "direction") == "GT"
        and proof.input(proof.arg(ref, 0), 1)
        and proof.constant(proof.arg(ref, 1), 0)
    )
    altered = rewrite(
        index, op, raw_line=op.raw_line.replace("direction=GT", "direction=GE")
    )
    result = check_rolled_last_row_indices(
        altered, block_rows=rows, live_instructions=live
    )
    assert not result["passed"], result
    assert "0<count<=B" in result["error"]
