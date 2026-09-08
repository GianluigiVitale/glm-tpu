"""CPU-only exact inventory refusal tests and original acquired graph replay."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from glm_tpu.greenfield.benchmarking.ws32_batched_collective_hlo import (
    _expected,
    check_batched_collectives,
)
from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import PrefillHloIndex
from glm_tpu.greenfield.benchmarking.ws32_pallas_one_layer import (
    _live_instruction_closure,
)
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module


def fixture(rows=17):
    # Contract-generated grammar fixture for mutations. Positive production
    # evidence comes independently from the SHA-bound captured graphs below.
    def shape(s):
        return s[0] + "[" + ",".join(map(str, s[1])) + "]"

    def shapes(ss):
        return shape(ss[0]) if len(ss) == 1 else "(" + ", ".join(map(shape, ss)) + ")"

    text = ["HloModule profile, num_partitions=32"]
    for dtype, kind in (
        ("f32", "add"),
        ("bf16", "add"),
        ("s32", "add"),
        ("s32", "minimum"),
    ):
        text.append(
            f"%{dtype}_{kind} {{\n%a = {dtype}[] parameter(0)\n%b = {dtype}[] parameter(1)\nROOT %c = {dtype}[] {kind}(%a, %b)\n}}"
        )
    text.append("ENTRY %main {")
    names, outputs = [], []
    for (layer, opcode, family, kind, axis, ins, outs), count in _expected(
        rows
    ).items():
        for _ in range(count):
            n = len(names)
            args = []
            for i, s in enumerate(ins):
                args.append(f"%p{n}_{i}")
                text.append(f"{args[-1]} = {shape(s)} constant(0)")
            groups = (
                [range(i, i + 4) for i in range(0, 32, 4)]
                if family == "feature"
                else [range(i, 32, 4) for i in range(4)]
            )
            groups = (
                "{" + ",".join("{" + ",".join(map(str, g)) + "}" for g in groups) + "}"
            )
            attr = (
                f"dimensions={{{axis}}}"
                if kind == "gather"
                else f"to_apply=%{ins[0][0]}_{kind}"
            )
            scope = (
                f"greenfield_ws32_batched_prefill/layer_{layer}/operation"
                if layer >= 0
                else "outside_layers"
            )
            names.append(f"%c{n}")
            outputs.append(shapes(outs))
            text.append(
                f'{names[-1]} = {shapes(outs)} {opcode}({", ".join(args)}), replica_groups={groups}, use_global_device_ids=true, {attr}, metadata={{op_name="{scope}"}}'
            )
    text.append(f'ROOT %out = ({", ".join(outputs)}) tuple({", ".join(names)})\n}}')
    return "\n".join(text)


@pytest.fixture(scope="module")
def synthetic():
    return parse_hlo_module(fixture())


def check(module, rows=17, live=None):
    return check_batched_collectives(
        PrefillHloIndex(module),
        block_rows=rows,
        live_instructions=module.instructions if live is None else live,
    )


def mutate(module, old, **changes):
    return replace(
        module,
        instructions=tuple(
            replace(op, **changes) if op is old else op for op in module.instructions
        ),
    )


@pytest.mark.parametrize("rows", [11, 17])
def test_expected_physical_and_leaf_counts(rows):
    result = check(parse_hlo_module(fixture(rows)), rows)
    assert result["passed"], result
    assert result["collective_count"] == 787
    assert result["reduction_operand_leaves"] == 898
    assert result["gather_output_leaves"] == 159
    assert len(result["health_reductions"]) == 2
    assert result["not_proven"]


@pytest.mark.parametrize(
    "case",
    [
        "duplicate_group",
        "full_pod",
        "nonglobal",
        "async",
        "missing",
        "extra",
        "dead",
        "scope",
        "axis",
        "output_dtype",
        "input_shape",
        "arity",
    ],
)
def test_collective_drift_refuses(synthetic, case):
    op = next(op for op in synthetic.collectives if op.raw_opcode == "all-gather")
    if case == "duplicate_group":
        module = mutate(
            synthetic, op, replica_groups=op.replica_groups + (op.replica_groups[0],)
        )
    elif case == "full_pod":
        module = mutate(synthetic, op, replica_groups=(tuple(range(32)),))
    elif case == "nonglobal":
        module = mutate(synthetic, op, use_global_device_ids=False)
    elif case == "async":
        module = mutate(synthetic, op, raw_opcode="all-gather-start")
    elif case == "missing":
        module = replace(
            synthetic,
            instructions=tuple(p for p in synthetic.instructions if p is not op),
        )
    elif case == "extra":
        module = replace(
            synthetic,
            instructions=synthetic.instructions + (replace(op, name="%extra"),),
        )
    elif case == "dead":
        assert not check(
            synthetic, live=tuple(p for p in synthetic.instructions if p is not op)
        )["passed"]
        return
    elif case == "scope":
        module = mutate(
            synthetic, op, op_name="greenfield_ws32_batched_prefill/layer_3/operation"
        )
    elif case == "axis":
        module = mutate(
            synthetic,
            op,
            raw_line=op.raw_line.replace("dimensions={1}", "dimensions={0}"),
        )
    elif case == "output_dtype":
        module = mutate(
            synthetic, op, result_shapes=(replace(op.result_shapes[0], dtype="u32"),)
        )
    elif case == "input_shape":
        module = mutate(
            synthetic,
            op,
            operand_shapes=(replace(op.operand_shapes[0], dimensions=(8,)),),
        )
    else:
        module = mutate(synthetic, op, operand_names=())
    assert not check(module)["passed"], case


@pytest.mark.parametrize("replacement", ["multiply", "maximum"])
def test_non_add_reducer_refused(synthetic, replacement):
    op = next(
        p
        for p in synthetic.instructions
        if p.computation.startswith("%f32_add") and p.opcode == "add"
    )
    module = mutate(
        synthetic,
        op,
        raw_opcode=replacement,
        opcode=replacement,
        raw_line=op.raw_line.replace("add(", replacement + "("),
    )
    assert not check(module)["passed"]


def test_minimum_is_not_generic_reducer_allowance(synthetic):
    op = next(
        p
        for p in synthetic.collectives
        if p.operand_shapes[0].dtype == "s32" and p.operand_shapes[0].dimensions == (8,)
    )
    module = mutate(
        synthetic,
        op,
        raw_line=op.raw_line.replace("to_apply=%s32_add", "to_apply=%s32_minimum"),
    )
    assert not check(module)["passed"]


def test_reducer_attribute_decoy_and_duplicate_refused(synthetic):
    op = next(p for p in synthetic.collectives if p.raw_opcode == "all-reduce")
    for line in (
        op.raw_line.replace("to_apply=%f32_add, ", "").replace(
            'op_name="', 'op_name="to_apply=%f32_add '
        ),
        op.raw_line + ", to_apply=%f32_add",
    ):
        assert not check(mutate(synthetic, op, raw_line=line))["passed"]


def test_equal_global_payload_count_cannot_hide_wrong_layer(synthetic):
    op = next(
        p for p in synthetic.collectives if p.op_name and "/layer_0/" in p.op_name
    )
    assert not check(
        mutate(synthetic, op, op_name=op.op_name.replace("/layer_0/", "/layer_1/"))
    )["passed"]


def test_last_producer_has_its_own_acquired_tuple_order(synthetic):
    op = next(
        p
        for p in synthetic.collectives
        if p.op_name and "/layer_74/" in p.op_name and len(p.operand_shapes) == 4
    )
    assert [s.dimensions[-1] for s in op.operand_shapes] == [128, 4, 2048, 576]
    assert [s.dtype for s in op.result_shapes] == ["f32", "f32", "bf16", "bf16"]
    order = (3, 0, 2, 1)  # correct pairs, but layer0 order is not acquired here
    module = mutate(
        synthetic,
        op,
        operand_names=tuple(op.operand_names[i] for i in order),
        operand_shapes=tuple(op.operand_shapes[i] for i in order),
        result_shapes=tuple(op.result_shapes[i] for i in order),
    )
    assert not check(module)["passed"]


@pytest.mark.parametrize("rows", [0, 1, 12, 32, True, 17.0])
def test_unacquired_row_profile_refused(synthetic, rows):
    with pytest.raises(ValueError, match="B17/B11"):
        check(synthetic, rows)


@pytest.mark.parametrize("graph,rows", [("prefill_chunk", 17), ("prefill_tail", 11)])
def test_original_acquired_graph(graph, rows):
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
        pytest.skip("original HLO unavailable locally")
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == receipt["graphs"][graph]["optimized_hlo_sha256"]
    module = parse_hlo_module(raw.decode())
    result = check(module, rows, live=_live_instruction_closure(module.instructions))
    assert result["passed"], result
    assert result["collective_count"] == 787
    assert result["reduction_operand_leaves"] == 898
    assert result["gather_output_leaves"] == 159
