"""Actual input guards: saved whole-model graphs and narrow hostile mutations."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import PrefillHloIndex
from glm_tpu.greenfield.benchmarking.ws32_batched_operand_health_hlo import (
    OperandHealthProof,
    check_batched_operand_health,
)
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from glm_tpu.greenfield.sharding.hlo_contract import HloShape


@pytest.fixture(scope="module", params=[("prefill_tail", 11), ("prefill_chunk", 17)])
def original(request):
    graph, rows = request.param
    root = Path(__file__).resolve().parents[3]
    receipt = json.loads(
        (
            root
            / "docs/artifacts/prefill-batched-seven-graph-acquisition-20260908.json"
        ).read_text()
    )
    path = (
        Path("/home/gianl/glm-run")
        / receipt["tag"]
        / "hlo"
        / (graph + ".optimized_hlo.txt")
    )
    if not path.exists():
        pytest.skip("original HLO unavailable")
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == receipt["graphs"][graph]["optimized_hlo_sha256"]
    index = PrefillHloIndex(parse_hlo_module(raw.decode()))
    proof = OperandHealthProof(index, rows)
    records = proof.check()
    return index, rows, proof, records


def rewrite(index, changes):
    return PrefillHloIndex(
        replace(
            index.module,
            instructions=tuple(
                changes.get(op.index, op) for op in index.module.instructions
            ),
        )
    )


def test_original_normalization_and_attention_inputs(original):
    _, _, proof, records = original
    assert [r["layer"] for r in records] == list(range(78))
    assert len(proof.finite) == 1259
    for key in ("normalized", "prepared_query", "sparse"):
        assert len({r[key] for r in records}) == 78


def test_original_grouped_and_router_guards(original):
    _, _, proof, _ = original
    assert len(proof.check_grouped()) == 225
    assert len(proof.check_router()) == 75


@pytest.mark.parametrize(
    "case",
    [
        "normalization_guard",
        "query_guard",
        "cache_guard",
        "query_span",
        "tail_zero",
        "cache_layout",
    ],
)
def test_actual_finite_guard_mutations(original, case):
    index, rows, proof, records = original
    b = proof.boolean
    record = records[3]
    call = b.ref(index.computations["ENTRY"][record["sparse"]])
    changes = {}
    if case in ("normalization_guard", "query_guard", "cache_guard"):
        if case == "normalization_guard":
            data = proof.padded_linear_input(
                index.computations["ENTRY"][record["normalized"]]
            )
        elif case == "query_guard":
            _, data = proof.query(proof.arg(call, 1))
        else:
            data = proof.bridge(proof.arg(call, 2), "cache")
        guards = [
            b.ops[ref[0]]
            for ref, domain in proof.known
            if not ref[1]
            and b.ops[ref[0]].opcode == "is-finite"
            and proof.arg(ref, 0) == data
        ]
        assert guards
        for op in guards:
            prefix = op.raw_line.split("is-finite(", 1)[0]
            changes[op.index] = replace(
                op,
                opcode="constant",
                raw_opcode="constant",
                operand_names=(),
                operand_shapes=(),
                raw_line=prefix + "constant(true)",
            )
    elif case == "cache_layout":
        ref = proof.arg(call, 2)
        op = b.ops[ref[0]]
        assert "{3,2,1,0:" in op.raw_line
        changes[op.index] = replace(
            op, raw_line=op.raw_line.replace("{3,2,1,0:", "{2,3,1,0:")
        )
    else:
        query = proof.arg(call, 1)
        outer = proof.arg(query, 0)
        inner_index = next(
            j for j in range(2) if b.ops[proof.arg(outer, j)[0]].opcode == "maximum"
        )
        if case == "query_span":
            inner = proof.arg(outer, inner_index)
            pads = [proof.arg(proof.arg(inner, j), 0) for j in range(2)]
            ref = next(
                v for v in pads if "padding=0_0x0_0x0_128" in b.ops[v[0]].raw_line
            )
            op = b.ops[ref[0]]
            changes[op.index] = replace(
                op,
                raw_line=op.raw_line.replace(
                    "padding=0_0x0_0x0_128", "padding=0_0x0_0x64_64"
                ),
            )
        else:
            tail = proof.arg(outer, 1 - inner_index)
            broadcast = proof.arg(tail, 0)
            pad = proof.arg(broadcast, 0)
            zeros = proof.arg(pad, 0)
            zero = proof.arg(zeros, 0)
            op = b.ops[zero[0]]
            assert "constant(0)" in op.raw_line
            changes[op.index] = replace(
                op, raw_line=op.raw_line.replace("constant(0)", "constant(nan)")
            )
    result = check_batched_operand_health(rewrite(index, changes), block_rows=rows)
    assert not result["passed"], (case, result)


@pytest.mark.parametrize("rows", [1, 12, 32, True, 17.0])
def test_unknown_profile(rows):
    with pytest.raises(ValueError, match="B17/B11"):
        check_batched_operand_health(None, block_rows=rows)


@pytest.mark.parametrize("case", ["grouped_selector", "router_logits", "router_bias"])
def test_actual_grouped_and_router_guard_mutations(original, case):
    index, rows, proof, _ = original
    b = proof.boolean
    changes = {}
    if case == "grouped_selector":
        record = proof.check_grouped()[0]
        outer = index.computations["ENTRY"][record["validity_guard"]]
        selector = index.operand(outer, 0)
        finished = next(
            op
            for op in index.computations["ENTRY"].values()
            if op.result_shapes == (HloShape("pred", ()),)
            and proof.health.input(b.ref(op), 13)
        )
        assert selector.opcode == "convert"
        changes[selector.index] = replace(
            selector,
            operand_names=(finished.name,),
            operand_shapes=finished.result_shapes,
            raw_line=selector.raw_line.replace(
                selector.operand_names[0], finished.name
            ),
        )
    else:
        record = proof.check_router()[0]
        key = "logits" if case == "router_logits" else "bias"
        op = index.computations["ENTRY"][record[key]]
        data = b.resolve(b.ref(op, () if key == "logits" else (1,)))
        guards = [
            b.ops[ref[0]]
            for ref, domain in proof.known
            if not ref[1]
            and b.ops[ref[0]].opcode == "is-finite"
            and proof.arg(ref, 0) == data
        ]
        assert guards
        for op in guards:
            changes[op.index] = replace(
                op,
                opcode="constant",
                raw_opcode="constant",
                operand_names=(),
                operand_shapes=(),
                raw_line=op.raw_line.split("is-finite(", 1)[0] + "constant(true)",
            )
    result = check_batched_operand_health(rewrite(index, changes), block_rows=rows)
    assert not result["passed"], (case, result)
