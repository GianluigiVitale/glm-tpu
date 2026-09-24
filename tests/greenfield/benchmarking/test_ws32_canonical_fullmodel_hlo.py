"""Saved DB609 main/tail, never a fresh TPU acquisition."""

from hashlib import sha256
from pathlib import Path
import json
import re

import pytest

from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill import _inspect_index
from glm_tpu.greenfield.benchmarking.ws32_batched_prefill import UNREGISTERED
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_hlo import RolledTransitions
from glm_tpu.greenfield.benchmarking.ws32_prefill_hlo_identity import Value, attribute
from glm_tpu.greenfield.benchmarking.ws32_pallas_one_layer import _computation_base
from tests.greenfield.benchmarking.test_ws32_rolled_prefill_hlo import rewrite

from glm_tpu.optimized.hlo_contract import parse_hlo_module
from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import PrefillHloIndex
from glm_tpu.greenfield.benchmarking.ws32_pallas_one_layer import (
    _live_instruction_closure,
)
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_collective_hlo import (
    check_rolled_collectives,
)
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_kernel_hlo import (
    check_rolled_kernels,
)
from glm_tpu.greenfield.benchmarking.ws32_canonical_prefill_hlo import (
    check_canonical_dense_loops,
)
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_operand_health_hlo import (
    canonical_router_bias,
)

ROOT = Path(__file__).resolve().parents[3]
RECEIPT = (
    ROOT / "docs/artifacts/prefill-canonical-model-compile-db609-sealed-20260909.json"
)


@pytest.fixture(scope="module", params=[128, 114])
def original(request):
    receipt = json.loads(RECEIPT.read_text())
    rows = request.param
    graph = "prefill_chunk" if rows == 128 else "prefill_tail"
    path = (
        Path("/home/gianl/glm-run")
        / receipt["tag"]
        / "rank0"
        / (graph + ".optimized_hlo.txt")
    )
    if not path.exists():
        # The local copy was evicted from this host; the sealed original stays
        # in the receipt's regional archive prefix and is not re-downloaded here.
        pytest.skip("DB609 original compiler evidence unavailable locally")
    text = path.read_text()
    assert (
        sha256(text.encode()).hexdigest()
        == receipt["graphs"][graph]["optimized_hlo_sha256"]
    )
    module = parse_hlo_module(text)
    return PrefillHloIndex(module), rows, _live_instruction_closure(module.instructions)


def test_actual_canonical_schedules(original):
    index, rows, live = original
    collectives = check_rolled_collectives(
        index, block_rows=rows, live_instructions=live, canonical_dense=True
    )
    assert collectives["passed"], collectives
    assert collectives["static_collective_count"] == 788
    assert collectives["static_placement_counts"] == {
        "prefix": 474,
        "canonical_dense": 6,
        "suffix": 300,
        "outer": 8,
    }
    assert collectives["four_iteration_dense_schedule_count"] == 24
    kernels = check_rolled_kernels(
        index, block_rows=rows, live_instructions=live, canonical_dense=True
    )
    assert kernels["passed"], kernels
    assert kernels["kernel_count"] == 1047
    assert kernels["static_placement_counts"] == {
        "prefix": 588,
        "canonical_dense": 9,
        "suffix": 450,
    }
    assert kernels["four_iteration_dense_schedule_count"] == 36


def test_actual_canonical_loop_stacks_health(original):
    index, rows, live = original
    report = check_canonical_dense_loops(index, block_rows=rows, live_instructions=live)
    assert report["passed"], report
    assert report["complete_stack_writes"] == 6
    assert len(report["health"]) == 3
    assert report["numerical_admission"] is False


def test_historical_default_refuses_canonical_graph(original):
    index, rows, live = original
    for check in (check_rolled_collectives, check_rolled_kernels):
        report = check(index, block_rows=rows, live_instructions=live)
        assert report["passed"] is False
        assert (
            "not ENTRY" in report["error"]
            or "unregistered rolled kernel" in report["error"]
        )


def test_actual_separated_bias(original):
    index, rows, live = original
    assert canonical_router_bias(
        index, block_rows=rows, live_instructions=live
    ).name == ("%all-reduce.1574" if rows == 128 else "%all-reduce.1811")


def test_actual_all_twelve_proofs_json_without_numerical_promotion(original):
    index, rows, live = original
    report = _inspect_index(
        index, block_rows=rows, live_instructions=live, canonical_dense=True
    )
    assert report["violations"] == [UNREGISTERED], {
        k: v for k, v in report.items() if k.endswith("proof") and not v.get("passed")
    }
    assert report["structural_profile"] == "canonical_dense_b128_b114_v1"
    assert (
        report["passed"]
        is report["profile_registered"]
        is report["numerical_claim"]
        is False
    )
    assert json.loads(json.dumps(report, allow_nan=False)) == report


@pytest.mark.parametrize(
    "case",
    [
        "trip_count",
        "immutable",
        "displaced_write",
        "wrong_next_input",
        "mask_stride",
        "live_carry",
    ],
)
def test_actual_dense_mutations_refuse(original, case):
    index, rows, live = original
    t = RolledTransitions(index, rows)
    loops = [
        op
        for op in index.module.instructions
        if op.opcode == "while"
        and (op.op_name or "").endswith("greenfield_ws32_prefill_dense_canonical/while")
    ]
    loop = loops[0]
    body = index.callee(loop, "body")
    root = index.roots[body]
    if case == "trip_count":
        condition = index.roots[index.callee(loop, "condition")]
        op = t.arg(Value(condition), 1).op
        changed = rewrite(
            index, op, raw_line=op.raw_line.replace("constant(4)", "constant(3)")
        )
    elif case == "immutable":
        names = list(root.operand_names)
        names[3] = names[4]
        changed = rewrite(index, root, operand_names=tuple(names))
    elif case == "live_carry":
        op = t.resolve(Value(index.operand(loop, 0))).op
        names = list(op.operand_names)
        names[3] = names[2]
        changed = rewrite(index, op, operand_names=tuple(names))
    elif case == "displaced_write":
        op = t.resolve(t.ssa.leaf(Value(root), 1)).op
        assert op.opcode == "dynamic-update-slice"
        names = list(op.operand_names)
        names[2] = names[3]
        changed = rewrite(index, op, operand_names=tuple(names))
    elif case == "wrong_next_input":
        op = t.all_loops(live)[1].initial.op
        assert op.opcode == "tuple"
        names = list(op.operand_names)
        names[12] = names[13]
        changed = rewrite(index, op, operand_names=tuple(names))
    elif rows == 114:
        op = next(op for op in index.module.instructions if op.name == "%lt.67617")
        changed = rewrite(
            index, op, raw_line=op.raw_line.replace("direction=LT", "direction=GT")
        )
    else:
        op = next(
            op
            for op in index.computations["%fused_computation.30225"].values()
            if op.opcode == "constant" and "constant(32)" in op.raw_line
        )
        changed = rewrite(
            index, op, raw_line=op.raw_line.replace("constant(32)", "constant(31)")
        )
    report = check_canonical_dense_loops(
        changed, block_rows=rows, live_instructions=live
    )
    assert not report["passed"], (case, report)


@pytest.mark.parametrize("case", ["groups", "layer", "bias_source", "bias_offset"])
def test_actual_new_schedule_mutations_refuse(original, case):
    index, rows, live = original
    if case == "bias_offset":
        bias = canonical_router_bias(index, block_rows=rows, live_instructions=live)
        t = RolledTransitions(index, rows)
        op = t.resolve(Value(index.operand(bias, 0))).op
        assert op.opcode == "dynamic-update-slice"
        # Only this DUS moves; untouched74 later-bias offsets are the anchor.
        zero = next(
            p
            for p in index.computations[_computation_base(op.computation)].values()
            if p.opcode == "constant"
            and p.result_shapes[0].dtype in ("s32", "u32")
            and p.result_shapes[0].dimensions == ()
            and "constant(0)" in p.raw_line
        )
        names = list(op.operand_names)
        names[2] = zero.name
        changed = rewrite(index, op, operand_names=tuple(names))
        with pytest.raises(ValueError, match="insertion offset"):
            canonical_router_bias(changed, block_rows=rows, live_instructions=live)
        return
    if case == "bias_source":
        op = next(
            op
            for op in index.module.instructions
            if op.opcode == "get-tuple-element"
            and _computation_base(op.computation) == "ENTRY"
            and attribute(op, "index") == "111"
        )
        changed = rewrite(
            index, op, raw_line=re.sub(r"\bindex=111\b", "index=112", op.raw_line)
        )
        with pytest.raises(ValueError, match="own layer3 checkpoint"):
            canonical_router_bias(changed, block_rows=rows, live_instructions=live)
        return
    loop = next(
        op
        for op in index.module.instructions
        if op.opcode == "while"
        and (op.op_name or "").endswith("greenfield_ws32_prefill_dense_canonical/while")
    )
    body = index.callee(loop, "body")
    if case == "groups":
        op = next(
            op
            for op in index.module.collectives
            if _computation_base(op.computation) == body
        )
        groups = [list(g) for g in op.replica_groups]
        groups[0][0], groups[1][0] = groups[1][0], groups[0][0]
        changed = rewrite(index, op, replica_groups=tuple(tuple(g) for g in groups))
        check = check_rolled_collectives
    else:
        op = next(
            op
            for op in index.module.instructions
            if op.opcode == "custom-call"
            and _computation_base(op.computation) == body
            and "pallas_call" in (op.op_name or "")
        )
        changed = rewrite(
            index, op, op_name=op.op_name.replace("/layer_0/", "/layer_1/")
        )
        check = check_rolled_kernels
    report = check(
        changed, block_rows=rows, live_instructions=live, canonical_dense=True
    )
    assert not report["passed"], (case, report)


@pytest.mark.parametrize(
    "case", ["health_source", "output_crop", "padding_or_health_and"]
)
def test_actual_output_health_tail_boundaries_refuse(original, case):
    index, rows, live = original
    t = RolledTransitions(index, rows)
    loop = next(
        op
        for op in index.module.instructions
        if op.opcode == "while"
        and (op.op_name or "").endswith("greenfield_ws32_prefill_dense_canonical/while")
    )
    if case == "health_source":
        op = next(
            op
            for op in index.module.instructions
            if op.opcode == "get-tuple-element"
            and _computation_base(op.computation) == "ENTRY"
            and op.operand_names == (loop.name,)
            and attribute(op, "index") == "2"
        )
        changed = rewrite(
            index, op, raw_line=re.sub(r"\bindex=2\b", "index=3", op.raw_line)
        )
    elif rows == 114:
        initial = t.resolve(Value(index.operand(loop, 0)))
        stack = t.resolve(t.ssa.leaf(initial, 4))
        pad = t.arg(stack, 0)
        assert pad.op.opcode == "pad"
        if case == "padding_or_health_and":
            op = pad.op
            changed = rewrite(
                index,
                op,
                raw_line=op.raw_line.replace("padding=0_14x0_0", "padding=1_13x0_0"),
            )
        else:
            selected = t.arg(pad, 0)
            op = t.arg(selected, 1).op
            assert op.opcode == "slice"
            changed = rewrite(
                index, op, raw_line=op.raw_line.replace("[0:114]", "[1:115]")
            )
    elif case == "padding_or_health_and":
        op = next(op for op in index.module.instructions if op.name == "%and.68214")
        changed = rewrite(
            index,
            op,
            opcode="or",
            raw_opcode="or",
            raw_line=op.raw_line.replace(" and(", " or("),
        )
    else:
        body = index.callee(loop, "body")
        write = t.resolve(t.ssa.leaf(Value(index.roots[body]), 1))
        tile = t.arg(write, 1)
        select = t.arg(tile, 0)
        op = t.arg(select, 1).op
        assert op.opcode == "slice"
        changed = rewrite(index, op, raw_line=op.raw_line.replace("[0:32]", "[32:64]"))
    report = check_canonical_dense_loops(
        changed, block_rows=rows, live_instructions=live
    )
    assert not report["passed"], (case, report)


def test_checkpoint_bias_leaf_formula_matches_actual_source_tree():
    import jax
    from glm_tpu.optimized.geometry import ModelGeometry
    from glm_tpu.greenfield.runtime.ws32_decoder import (
        Ws32DecoderConfig,
        ws32_decoder_weight_names,
    )

    config = Ws32DecoderConfig(
        ModelGeometry.from_hf_config(
            json.loads((ROOT / "configs/glm-5.2-fp8-config.json").read_text())
        ),
        8192,
        host_main_rope_table=True,
    )
    names = jax.tree.leaves(ws32_decoder_weight_names(config))
    for layer in range(3, 78):
        leaf = 111 + 28 * (layer - 3) + 7 * ((layer - 2) // 4)
        assert (
            names[leaf - 14] == f"model.layers.{layer}.mlp.gate.e_score_correction_bias"
        )


def test_actual_cache_copy_identity_is_canonical_only_and_rejects_mutations(original):
    from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_hlo import RolledIdentity
    from glm_tpu.greenfield.benchmarking.ws32_batched_helper_hlo import _target

    index, rows, live = original
    rotated = []
    for op in index.module.instructions:
        if (
            op.opcode != "custom-call"
            or _target(op) != "ConcatBitcast"
            or op.result_shapes[0].dimensions != (21, 16, 64, 128)
        ):
            continue
        start = index.operand(index.operand(op, 0), 0)
        if "[12:18]" in start.raw_line:
            rotated.append(op)
    assert len(rotated) == (1 if rows == 128 else 0)
    if not rotated:
        return  # Tail has no rotated helper; its full12-proof test covers it.
    op = rotated[0]
    default = RolledIdentity(index)
    with pytest.raises(ValueError, match="slices reordered/incomplete"):
        default.resolve(Value(op))
    ssa = RolledIdentity(index, canonical_dense=True)
    start = index.operand(index.operand(op, 0), 0)
    assert ssa.same(ssa.resolve(Value(op)), Value(index.operand(start, 0)))
    # A third permutation is not the one acquired compiler mechanism.
    order = (1, 0, 2, 3)
    reordered = rewrite(
        index,
        op,
        operand_names=tuple(op.operand_names[j] for j in order),
        operand_shapes=tuple(op.operand_shapes[j] for j in order),
    )
    with pytest.raises(ValueError):
        RolledIdentity(reordered, canonical_dense=True).resolve(
            Value(reordered.module.instructions[op.index])
        )
    names = list(op.operand_names)
    names[1] = names[0]
    duplicate = rewrite(index, op, operand_names=tuple(names))
    with pytest.raises(ValueError):
        RolledIdentity(duplicate, canonical_dense=True).resolve(
            Value(duplicate.module.instructions[op.index])
        )
    # Preserve spans, but source one slice from another cache object.
    other = next(
        p
        for p in index.module.instructions
        if p.opcode == "get-tuple-element"
        and _computation_base(p.computation) == "ENTRY"
        and p.result_shapes == op.result_shapes
        and attribute(p, "index") == "11"
    )
    mixed = rewrite(index, start, operand_names=(other.name,))
    with pytest.raises(ValueError, match="mixes sources"):
        RolledIdentity(mixed, canonical_dense=True).resolve(
            Value(mixed.module.instructions[op.index])
        )
