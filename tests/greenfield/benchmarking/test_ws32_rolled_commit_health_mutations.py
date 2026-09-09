"""Saved DB602 global-health edge mutations; CPU only, no new graph loader."""

from dataclasses import replace

import pytest

from tests.greenfield.benchmarking.test_ws32_rolled_prefill_hlo import original, rewrite
from glm_tpu.greenfield.benchmarking.ws32_batched_health_hlo import WriterHealthProof
from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import PrefillHloIndex
from glm_tpu.greenfield.benchmarking.ws32_prefill_hlo_identity import attribute
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_health_hlo import (
    check_rolled_commit_health,
)


@pytest.fixture(scope="module")
def bridge(original):
    index, rows, live, _, loops = original
    report = check_rolled_commit_health(index, block_rows=rows, live_instructions=live)
    assert report["passed"], report
    assert len(report["stacks"]) == 78
    return index, rows, live, loops, report


def check_refusal(index, rows, live):
    report = check_rolled_commit_health(index, block_rows=rows, live_instructions=live)
    assert not report["passed"], report


def test_wrong_same_shaped_layer_health_stack(bridge):
    index, rows, live, loops, _ = bridge
    # Redirect the LAST layer to its predecessor: same family and shape, and
    # no artificial backward dependency/cycle in the earlier layer chain.
    target, other = loops[77], loops[76]
    changed = []
    for op in index.module.instructions:
        if (
            op.opcode == "get-tuple-element"
            and op.computation == target.loop.computation
            and op.operand_names == (target.loop.name,)
            and attribute(op, "index") == str(target.health_slot)
        ):
            changed.append(replace(op, operand_names=(other.loop.name,)))
    assert changed
    replacements = {op.index: op for op in changed}
    mutated = PrefillHloIndex(
        replace(
            index.module,
            instructions=tuple(
                replacements.get(op.index, op) for op in index.module.instructions
            ),
        )
    )
    check_refusal(mutated, rows, live)


def test_tail_trim_must_start_at_zero(bridge):
    index, rows, live, _, report = bridge
    if rows == 128:
        assert all(stack["trim"] is None for stack in report["stacks"])
        return
    name = report["stacks"][77]["trim"]
    op = next(op for op in index.module.instructions if op.name == name)
    assert "[0:114]" in op.raw_line
    mutated = rewrite(index, op, raw_line=op.raw_line.replace("[0:114]", "[1:115]"))
    check_refusal(mutated, rows, live)


def test_mandatory_global_conjunction_cannot_be_disjunction(bridge):
    index, rows, live, _, report = bridge
    proof = WriterHealthProof(index, rows)
    vote = proof.frontier()
    b = proof.boolean
    known = b.factors(vote)
    target = report["stacks"][77]["global_factor"]
    matches = []
    for ref, axis in known:
        op = b.ops[ref[0]]
        if op.opcode != "and" or ref[1] or len(op.operand_names) != 2:
            continue
        if any(b.ops[b.resolve(b.operand(ref, j))[0]].name == target for j in (0, 1)):
            matches.append(op)
    assert matches, "must mutate an actual immediate mandatory health conjunction"
    op = matches[0]
    mutated = rewrite(
        index,
        op,
        opcode="or",
        raw_opcode="or",
        raw_line=op.raw_line.replace(" and(", " or("),
    )
    check_refusal(mutated, rows, live)


def test_row_permutation_before_flatten_is_not_identity(bridge):
    index, rows, live, _, report = bridge
    name = report["stacks"][77]["flatten"]
    flatten = next(op for op in index.module.instructions if op.name == name)
    source = index.operand(flatten, 0)
    assert source.result_shapes[0].dimensions == (4, 32)
    # Insert a shape-valid reverse within each tile, preserving every other
    # field and the complete producer/consumer chain. This is not malformed
    # reshape geometry masquerading as a permutation mutation.
    reverse = replace(
        source,
        index=max(op.index for op in index.module.instructions) + 1,
        name="%adversarial_health_reverse",
        opcode="reverse",
        raw_opcode="reverse",
        operand_names=(source.name,),
        operand_shapes=source.result_shapes,
        raw_line=f"  %adversarial_health_reverse = pred[4,32] reverse({source.name}), dimensions={{1}}",
    )
    changed = replace(flatten, operand_names=(reverse.name,))
    instructions = []
    for op in index.module.instructions:
        if op.index == flatten.index:
            instructions.extend((reverse, changed))
        else:
            instructions.append(op)
    mutated = PrefillHloIndex(replace(index.module, instructions=tuple(instructions)))
    check_refusal(mutated, rows, live)
