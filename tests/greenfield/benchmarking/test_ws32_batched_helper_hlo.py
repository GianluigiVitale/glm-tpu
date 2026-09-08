"""Compiler helper mechanism mutations; CPU-only original graph replay."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from glm_tpu.greenfield.benchmarking.ws32_batched_helper_hlo import (
    _expected,
    check_batched_helpers,
)
from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import PrefillHloIndex
from glm_tpu.greenfield.benchmarking.ws32_pallas_one_layer import (
    _live_instruction_closure,
)
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module


def shape(dtype, dims):
    return dtype + "[" + ",".join(map(str, dims)) + "]"


def fixture(rows=17, *, paired=False, copy_counts=None):
    # Generated profile isolates refusal mechanics. Independent positives below
    # are immutable captured compiler products, not this generated fixture.
    text = ["HloModule helpers, num_partitions=32"]
    state = (
        f"(s32[], u32[256], u32[256], f32[256], pred[256], f32[{rows+1}], s32[], s32[])"
    )
    text += [
        f"%condition {{\n%p = {state} parameter(0)\nROOT %false = pred[] constant(false)\n}}",
        f"%body {{\nROOT %p = {state} parameter(0)\n}}",
    ]
    for layer in range(3, 78):
        for pair in range(3):
            text.append(f"%scratch_{layer}_{pair} {{")
            text += [
                f'%a{i} = u32[256] custom-call(), custom_call_target="AllocateBuffer"'
                for i in range(2)
            ]
            text += [
                "%zero = s32[] constant(0)",
                "%f = f32[256] constant(0)",
                "%p = pred[256] constant(false)",
                f"%keys = f32[{rows+1}] constant(0)",
            ]
            text.append(
                f"%init = {state} tuple(%zero, %a0, %a1, %f, %p, %keys, %zero, %zero)"
            )
            text.append(
                f'ROOT %loop = {state} while(%init), condition=%condition, body=%body, metadata={{op_name="greenfield_ws32_batched_prefill/layer_{layer}/jit(searchsorted)/jit(_searchsorted_scan_impl)/while"}}\n}}'
            )
    text.append("ENTRY %main {")
    counts = _expected(rows, paired_position_sort=paired)
    for key, count in (copy_counts or {}).items():
        counts[key] = count
    for n, ((target, dtype, dims), count) in enumerate(counts.items()):
        if target == "AllocateBuffer":
            continue
        for j in range(count):
            prefix = f"%h{n}_{j}"
            full = shape(dtype, dims)
            text.append(f"{prefix}_src = {full} constant(0)")
            if target != "ConcatBitcast":
                text.append(
                    f'{prefix} = {full} custom-call({prefix}_src), custom_call_target="{target}", metadata={{op_name="jit(gather)/gather"}}'
                )
                continue
            axis = 1 if dims == (8192, 64) else 0
            if dims == (21, 16, 64, 128):
                spans = ((0, 6), (6, 12), (12, 18), (18, 21))
            elif dims == (78, 16, 64, 640):
                spans = ((0, 20), (20, 40), (60, 78), (40, 60))
            else:
                width = dims[axis] // 4
                spans = tuple((i * width, (i + 1) * width) for i in range(4))
            for part, (begin, end) in enumerate(spans):
                sizes = list(dims)
                sizes[axis] = end - begin
                sub = shape(dtype, sizes)
                ranges = [(0, d) for d in dims]
                ranges[axis] = (begin, end)
                slices = ",".join(f"[{a}:{b}]" for a, b in ranges)
                text.append(
                    f"{prefix}_s{part} = (({full}), {sub}, s32[]) slice-start({prefix}_src), slice={{{slices}}}"
                )
                text.append(f"{prefix}_d{part} = {sub} slice-done({prefix}_s{part})")
            text.append(
                f'{prefix} = {full} custom-call({", ".join(prefix+"_d"+str(i) for i in range(4))}), custom_call_target="ConcatBitcast"'
            )
    text.append("ROOT %out = s32[] constant(0)\n}")
    return "\n".join(text)


@pytest.fixture(scope="module")
def synthetic():
    return parse_hlo_module(fixture())


def check(module, rows=17, live=None, *, paired=False):
    return check_batched_helpers(
        PrefillHloIndex(module),
        block_rows=rows,
        live_instructions=module.instructions if live is None else live,
        paired_position_sort=paired,
    )


def mutate(module, old, **changes):
    return replace(
        module,
        instructions=tuple(
            replace(op, **changes) if op is old else op for op in module.instructions
        ),
    )


@pytest.mark.parametrize("rows,count", [(17, 2228), (11, 2212)])
def test_mechanism_fixture(rows, count):
    result = check(parse_hlo_module(fixture(rows)), rows)
    assert result["passed"], result
    assert result["helper_count"] == count
    assert len(result["scratch_pairs"]) == 225
    assert "CACHE_RECONSTRUCTION_IDENTITY_OR_OWNERSHIP" in result["not_proven"]


@pytest.mark.parametrize(
    "case",
    [
        "unknown",
        "side_effect",
        "wrong_dtype",
        "missing",
        "duplicate",
        "dead",
        "annotation_arity",
        "annotation_scope",
    ],
)
def test_helper_refusal(synthetic, case):
    op = next(
        p
        for p in synthetic.instructions
        if 'custom_call_target="AssumeGatherIndicesInBound"' in p.raw_line
    )
    if case == "unknown":
        m = mutate(
            synthetic,
            op,
            raw_line=op.raw_line.replace('"AssumeGatherIndicesInBound"', '"Unknown"'),
        )
    elif case == "side_effect":
        m = mutate(
            synthetic,
            op,
            raw_line=op.raw_line.replace(
                ", metadata=", ", custom_call_has_side_effect=true, metadata="
            ),
        )
    elif case == "wrong_dtype":
        m = mutate(
            synthetic, op, result_shapes=(replace(op.result_shapes[0], dtype="u32"),)
        )
    elif case == "missing":
        m = replace(
            synthetic,
            instructions=tuple(p for p in synthetic.instructions if p is not op),
        )
    elif case == "duplicate":
        m = replace(
            synthetic,
            instructions=synthetic.instructions + (replace(op, name="%extra"),),
        )
    elif case == "dead":
        assert not check(
            synthetic, live=tuple(p for p in synthetic.instructions if p is not op)
        )["passed"]
        return
    elif case == "annotation_arity":
        m = mutate(synthetic, op, operand_names=())
    else:
        m = mutate(synthetic, op, op_name="unrelated")
    assert not check(m)["passed"]


@pytest.mark.parametrize(
    "case",
    [
        "duplicate_span",
        "stride",
        "other_axis",
        "other_source",
        "wrong_done",
        "wrong_handle",
        "escaping_start",
        "escaping_done",
    ],
)
def test_concat_refusal(synthetic, case):
    op = next(
        p
        for p in synthetic.instructions
        if p.raw_opcode == "slice-start"
        and p.result_shapes[0].dimensions == (2048, 1536)
    )
    index = PrefillHloIndex(synthetic)
    done = next(
        p
        for p in synthetic.instructions
        if p.raw_opcode == "slice-done" and p.operand_names == (op.name,)
    )
    if case == "duplicate_span":
        m = mutate(synthetic, op, raw_line=op.raw_line.replace("[0:512]", "[512:1024]"))
    elif case == "stride":
        m = mutate(synthetic, op, raw_line=op.raw_line.replace("[0:512]", "[0:512:2]"))
    elif case == "other_axis":
        m = mutate(synthetic, op, raw_line=op.raw_line.replace("[0:1536]", "[0:1535]"))
    elif case == "other_source":
        source = index.operand(op, 0)
        m = mutate(synthetic, op, operand_names=("%other_source",))
        m = replace(
            m, instructions=m.instructions + (replace(source, name="%other_source"),)
        )
    elif case == "wrong_done":
        m = mutate(synthetic, done, raw_opcode="copy", opcode="copy")
    elif case == "wrong_handle":
        m = mutate(synthetic, op, result_shapes=op.result_shapes[:-1])
    else:
        target = op if case == "escaping_start" else done
        m = replace(
            synthetic,
            instructions=synthetic.instructions
            + (replace(done, name="%escape", operand_names=(target.name,)),),
        )
    result = check(m)
    assert not result["passed"], case


@pytest.mark.parametrize(
    "case",
    [
        "extra_user",
        "duplicate_pair",
        "wrong_slot",
        "wrong_layer",
        "wrong_loop",
        "wrong_shape",
        "missing_body",
    ],
)
def test_scratch_refusal(synthetic, case):
    op = next(p for p in synthetic.instructions if p.name == "%a0")
    nodes = PrefillHloIndex(synthetic).computations[op.computation.split(" ", 1)[0]]
    init, loop = nodes["%init"], nodes["%loop"]
    if case == "extra_user":
        m = replace(
            synthetic,
            instructions=synthetic.instructions
            + (replace(init, name="%escaped_scratch", operand_names=(op.name,)),),
        )
    elif case == "duplicate_pair":
        m = mutate(
            synthetic,
            init,
            operand_names=("%zero", "%a0", "%a0", *init.operand_names[3:]),
        )
    elif case == "wrong_slot":
        m = mutate(
            synthetic, init, operand_names=("%a0", "%zero", *init.operand_names[2:])
        )
    elif case == "wrong_layer":
        m = mutate(
            synthetic, loop, op_name=loop.op_name.replace("/layer_3/", "/layer_4/")
        )
    elif case == "wrong_loop":
        m = mutate(synthetic, loop, raw_opcode="fusion", opcode="fusion")
    elif case == "wrong_shape":
        m = mutate(synthetic, init, result_shapes=init.result_shapes[:-1])
    else:
        m = mutate(
            synthetic,
            loop,
            raw_line=loop.raw_line.replace("body=%body", "body=%absent"),
        )
    assert not check(m)["passed"], case


@pytest.mark.parametrize("rows", [1, 12, 32, True, 17.0])
def test_unregistered_rows_refused(synthetic, rows):
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
    assert len(result["scratch_pairs"]) == 225


@pytest.mark.parametrize("rows", [11, 17])
@pytest.mark.parametrize(
    "wk,stack,passed",
    [(0, 0, True), (21, 4, True), (21, 5, True), (22, 4, False), (21, 6, False)],
)
def test_paired_optional_copy_bounds(rows, wk, stack, passed):
    copies = {
        ("ConcatBitcast", "f32", (128, 6144)): wk,
        ("ConcatBitcast", "bf16", (21, 16, 64, 128)): stack,
    }
    module = parse_hlo_module(fixture(rows, paired=True, copy_counts=copies))
    result = check(module, rows, paired=True)
    assert result["passed"] is passed, result.get("error")
    if passed:
        assert json.loads(json.dumps(result)) == result
    assert not check(module, rows)["passed"]  # historical mode stays exact


@pytest.mark.parametrize("case", ["source", "span", "escape"])
def test_paired_copy_structure_still_required(case):
    module = parse_hlo_module(fixture(paired=True))
    op = next(
        p
        for p in module.instructions
        if p.raw_opcode == "slice-start"
        and p.result_shapes[0].dimensions == (128, 6144)
    )
    if case == "source":
        source = PrefillHloIndex(module).operand(op, 0)
        module = mutate(module, op, operand_names=("%other_copy_source",))
        module = replace(
            module,
            instructions=module.instructions
            + (replace(source, name="%other_copy_source"),),
        )
    elif case == "span":
        module = mutate(module, op, raw_line=op.raw_line.replace("[0:32]", "[32:64]"))
    else:
        done = next(
            p
            for p in module.instructions
            if p.raw_opcode == "slice-done" and p.operand_names == (op.name,)
        )
        module = replace(
            module,
            instructions=module.instructions + (replace(done, name="%copy_escape"),),
        )
    assert not check(module, paired=True)["passed"]
