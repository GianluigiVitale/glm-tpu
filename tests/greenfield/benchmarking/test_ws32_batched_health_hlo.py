"""Hash-bound acquired graphs and actual commit/mask/writer mutations."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import PrefillHloIndex
from glm_tpu.greenfield.benchmarking.ws32_batched_health_hlo import (
    WriterHealthProof,
    check_batched_writer_health,
)
from glm_tpu.greenfield.sharding.hlo_contract import HloShape, parse_hlo_module


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
        pytest.skip("original optimized HLO unavailable")
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == receipt["graphs"][graph]["optimized_hlo_sha256"]
    index = PrefillHloIndex(parse_hlo_module(raw.decode()))
    baseline = check_batched_writer_health(index, block_rows=rows)
    assert baseline["passed"], baseline
    return index, rows, baseline


def rewrite(index, op, **changes):
    new = replace(op, **changes)
    return PrefillHloIndex(
        replace(
            index.module,
            instructions=tuple(
                new if p.index == op.index else p for p in index.module.instructions
            ),
        )
    )


def test_all_actual_cache_writers_gate(original):
    _, _, result = original
    assert len(result["writers"]) == 120
    assert {r["layer"] for r in result["writers"] if r["family"] == "kv"} == set(
        range(78)
    )
    assert all(r["passed"] for r in result["writers"])
    assert result["missing"] == []
    assert "ALL_MODEL_HEALTH_CHECKS_EXIST" in result["not_proven"]
    assert result["evaluated_values"] < 20000


@pytest.mark.parametrize(
    "case", ["vote_or", "mask_direction", "nonempty", "writer_selector"]
)
def test_actual_health_mutations_refuse(original, case):
    index, rows, baseline = original
    proof = WriterHealthProof(index, rows)
    b = proof.boolean
    vote = proof.frontier()
    if case == "vote_or":
        op = b.ops[vote[0]]
        assert op.opcode == "and"
        mutated = rewrite(
            index,
            op,
            opcode="or",
            raw_opcode="or",
            raw_line=op.raw_line.replace(" and(", " or("),
        )
    elif case in ("mask_direction", "nonempty"):
        if case == "mask_direction":
            # Use the exact final live mask from the connected row reduction.
            final = proof.node(proof.arg(proof.arg(vote, 0), 1), "reduce", 2)
            pair = proof.arg(final, 0)
            invert = next(
                proof.arg(pair, j)
                for j in range(2)
                if b.ops[proof.arg(pair, j)[0]].opcode == "not"
            )
            mask = proof.arg(invert, 0)
            assert proof.live_mask(mask) == 0
            op = b.ops[mask[0]]
            mutated = rewrite(
                index, op, raw_line=op.raw_line.replace("direction=LT", "direction=GE")
            )
        else:
            op = next(
                p
                for p in index.computations["ENTRY"].values()
                if p.opcode == "compare"
                and "direction=GT" in p.raw_line
                and proof.input(proof.arg(b.ref(p), 0), 1)
                and proof.constant(proof.arg(b.ref(p), 1), 0)
            )
            mutated = rewrite(
                index, op, raw_line=op.raw_line.replace("direction=GT", "direction=GE")
            )
    else:
        record = next(r for r in baseline["writers"] if r["family"] == "kv")
        writer = index.computations["ENTRY"][record["writer"]]
        converted = index.operand(writer, 0)
        predicate = index.operand(converted, 0)
        # Require finished for this writer; unchanged commit requires NOT
        # finished. Change the selector only, not a predicate used by both.
        finished = next(
            p
            for p in index.computations["ENTRY"].values()
            if p.result_shapes == (HloShape("pred", ()),) and proof.input(b.ref(p), 13)
        )
        mutated = rewrite(
            index,
            converted,
            operand_names=(finished.name,),
            raw_line=converted.raw_line.replace(predicate.name, finished.name),
        )
    result = check_batched_writer_health(mutated, block_rows=rows)
    assert not result["passed"], (case, result)


@pytest.mark.parametrize("rows", [1, 12, 32, True, 17.0])
def test_unknown_profile_refuses(rows):
    with pytest.raises(ValueError, match="B17/B11"):
        check_batched_writer_health(None, block_rows=rows)
