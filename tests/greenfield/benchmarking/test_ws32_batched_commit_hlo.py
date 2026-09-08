"""Fail-closed commit boundary checks; CPU-only, no model performance claim."""

from hashlib import sha256
import json
from pathlib import Path

import pytest

from glm_tpu.greenfield.benchmarking.ws32_batched_commit_hlo import check_batched_commit
from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import PrefillHloIndex
from glm_tpu.greenfield.benchmarking.ws32_pallas_one_layer import (
    _live_instruction_closure,
)
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module


KV = "bf16[78,16,64,640]"
IX = "bf16[21,16,64,128]"
STATE = [
    KV,
    IX,
    "s32[1,2048]",
    "s32[1]",
    "f32[1,2048]",
    "s32[1]",
    "s32[1]",
    "pred[1]",
    IX,
    "pred[]",
]


def tup(shapes):
    return "(" + ", ".join(shapes) + ")"


def fixture(rows=17):
    inputs = [
        f"s32[{rows}]",
        "s32[]",
        *STATE[:6],
        "s32[1,16]",
        "s32[1]",
        "pred[1]",
        IX,
        "s32[]",
        "pred[]",
        KV,
        IX,
        IX,
        "pred[]",
    ]
    committed_args = [
        KV,
        IX,
        IX,
        "s32[]",
        "pred[]",
        "pred[]",
        "s32[1,2048]",
        "s32[1]",
        "f32[1,2048]",
    ]
    rollback_inputs = [2, 3, 4, 5, 6, 7, 9, 11, 13]
    false_shapes = [inputs[i] for i in rollback_inputs]
    feature = (
        "{"
        + ",".join(
            "{" + ",".join(map(str, range(i, i + 4))) + "}" for i in range(0, 32, 4)
        )
        + "}"
    )
    expert = (
        "{"
        + ",".join("{" + ",".join(map(str, range(i, 32, 4))) + "}" for i in range(4))
        + "}"
    )
    text = [
        "HloModule commit, num_partitions=32",
        """%min {
%a = s32[] parameter(0)
%b = s32[] parameter(1)
ROOT %minimum = s32[] minimum(%a, %b)
}
%identity_index {
ROOT %p = bf16[21,16,64,128] parameter(0)
}
%head_false {
%p = () parameter(0)
%no_token = s32[1] constant({-1})
%good = pred[] constant(true)
ROOT %out = (s32[1], pred[]) tuple(%no_token, %good)
}
%head_true {
%health = pred[] parameter(0)
%token = s32[1] constant({42})
ROOT %out = (s32[1], pred[]) tuple(%token, %health)
}
""",
        f"%rollback {{\n%p = {tup(false_shapes)} parameter(0)",
    ]
    text += [
        f"%f{i} = {shape} get-tuple-element(%p), index={i}"
        for i, shape in enumerate(false_shapes)
    ]
    text += [
        "%bad = pred[1] constant({0})",
        f"ROOT %out = {tup(STATE)} tuple(%f0, %f1, %f2, %f3, %f4, %f5, %f6, %bad, %f7, %f8)\n}}",
    ]
    text += [f"%commit {{\n%p = {tup(committed_args)} parameter(0)"]
    text += [
        f"%c{i} = {shape} get-tuple-element(%p), index={i}"
        for i, shape in enumerate(committed_args)
    ]
    text += [
        f"""%final_int = s32[] convert(%c4)
%active = {IX} conditional(%final_int, %c1, %c2), branch_computations={{%identity_index, %identity_index}}
%position = s32[1] bitcast(%c3)
%one = s32[] constant(1)
%length = s32[] add(%c3, %one)
%context = s32[1] bitcast(%length)
%health = pred[1] bitcast(%c5)
ROOT %out = {tup(STATE)} tuple(%c0, %active, %c6, %c7, %c8, %position, %context, %health, %c2, %c4)
}}
ENTRY %main {{
%inputs = {tup(inputs)} parameter(0)"""
    ]
    text += [
        f"%i{i} = {shape} get-tuple-element(%inputs), index={i}"
        for i, shape in enumerate(inputs)
    ]
    # Actual acquired rollback reconstruction: ordered asynchronous cache slices.
    for i, (a, b) in enumerate(((0, 6), (6, 12), (12, 18), (18, 21))):
        shape = f"bf16[{b-a},16,64,128]"
        text += [
            f"%slice_start{i} = (({IX}), {shape}, s32[]) slice-start(%i3), slice={{[{a}:{b}], [0:16], [0:64], [0:128]}}",
            f"%slice_done{i} = {shape} slice-done(%slice_start{i})",
        ]
    text += [
        f"""%reconstructed = {IX} custom-call(%slice_done0, %slice_done1, %slice_done2, %slice_done3), custom_call_target="ConcatBitcast"
%copy_start = ({IX}, {IX}, u32[]) copy-start(%reconstructed)
%original_index = {IX} copy-done(%copy_start)
%zero = s32[] constant(0)
%b = s32[] constant({rows})
%limit = s32[] constant(8192)
%last = s32[] constant(8191)
%pos = s32[] bitcast(%i7)
%positive_pos = s32[] maximum(%pos, %zero)
%offset = s32[] minimum(%last, %positive_pos)
%positive_count = s32[] maximum(%i1, %zero)
%count = s32[] minimum(%b, %positive_count)
%remaining = s32[] subtract(%limit, %offset)
%safe_count = s32[] minimum(%count, %remaining)
%end = s32[] add(%offset, %safe_count)
%eq = pred[] compare(%end, %i12), direction=EQ
%not_finished = pred[] not(%i13)
%final = pred[] and(%eq, %not_finished)
%final_int = s32[] convert(%final)
%empty = () tuple()
%head = (s32[1], pred[]) conditional(%final_int, %empty, %i17), branch_computations={{%head_false, %head_true}}
%head_token = s32[1] get-tuple-element(%head), index=0
%head_health = pred[] get-tuple-element(%head), index=1
%incoming = pred[] bitcast(%i10)
%local = pred[] and(%incoming, %head_health)
%vote = s32[] convert(%local)
%feature = s32[] all-reduce(%vote), replica_groups={feature}, use_global_device_ids=true, to_apply=%min
%expert = s32[] all-reduce(%feature), replica_groups={expert}, use_global_device_ids=true, to_apply=%min
%consensus = pred[] compare(%expert, %zero), direction=NE
%healthy_int = s32[] convert(%consensus)
%rollback_args = {tup(false_shapes)} tuple(%i2, %original_index, %i4, %i5, %i6, %i7, %i9, %i11, %i13)
%commit_args = {tup(committed_args)} tuple(%i14, %i15, %i16, %end, %final, %consensus, %i4, %i5, %i6)
%atomic = {tup(STATE)} conditional(%healthy_int, %rollback_args, %commit_args), branch_computations={{%rollback, %commit}}
%emit = pred[] and(%consensus, %final)
%emit_row = pred[1] bitcast(%emit)
%sentinel = s32[1] constant({{-1}})
%token = s32[1] select(%emit_row, %head_token, %sentinel)"""
    ]
    text += [
        f"%o{i} = {shape} get-tuple-element(%atomic), index={i}"
        for i, shape in enumerate(STATE)
    ]
    result_shapes = (
        STATE[:6] + ["s32[1,16]"] + STATE[6:9] + ["s32[]", STATE[9], "s32[1]"]
    )
    text += [
        f"ROOT %out = {tup(result_shapes)} tuple(%o0, %o1, %o2, %o3, %o4, %o5, %i8, %o6, %o7, %o8, %i12, %o9, %token)\n}}"
    ]
    return "\n".join(text)


def check(text, rows=17):
    module = parse_hlo_module(text)
    return check_batched_commit(
        PrefillHloIndex(module),
        block_rows=rows,
        live_instructions=_live_instruction_closure(module.instructions),
    )


@pytest.mark.parametrize("rows", [1, 11, 17, 32])
def test_complete_commit_boundary(rows):
    result = check(fixture(rows), rows)
    assert result["passed"], result


@pytest.mark.parametrize(
    "old,new",
    [
        ("minimum(%a, %b)", "maximum(%a, %b)"),
        ("minimum(%a, %b)", "minimum(%a, %a)"),
        ("parameter(1)\nROOT %minimum", "parameter(0)\nROOT %minimum"),
        ("all-reduce(%feature)", "all-reduce(%vote)"),
        ("use_global_device_ids=true", "use_global_device_ids=false"),
        (
            "compare(%expert, %zero), direction=NE",
            "compare(%expert, %zero), direction=EQ",
        ),
        ("convert(%consensus)", "convert(%final)"),
        ("%o0, %o1, %o2", "%o0, %i3, %o2"),
        ("%o0, %o1, %o2", "%o0, %o8, %o2"),
        ("%i7, %i9, %i11, %i13)", "%i9, %i7, %i11, %i13)"),
        ("%i2, %original_index, %i4", "%i2, %i11, %i4"),
        ("%i9, %i11, %i13)", "%i9, %i3, %i13)"),
        ("%bad = pred[1] constant({0})", "%bad = pred[1] constant({1})"),
        ("%end, %final, %consensus, %i4", "%end, %final, %i17, %i4"),
        ("compare(%end, %i12)", "compare(%end, %i1)"),
        ("not(%i13)", "not(%i17)"),
        ("constant(8191)", "constant(8190)"),
        ("subtract(%limit, %offset)", "subtract(%offset, %limit)"),
        ("%position, %context, %health", "%context, %position, %health"),
        ("conditional(%final_int, %c1, %c2)", "conditional(%final_int, %c2, %c1)"),
        ("conditional(%final_int, %c1, %c2)", "conditional(%final_int, %c2, %c2)"),
        ("%health, %c2, %c4)", "%health, %c1, %c4)"),
        ("and(%consensus, %final)", "and(%final, %final)"),
        (
            "select(%emit_row, %head_token, %sentinel)",
            "select(%emit_row, %i5, %sentinel)",
        ),
        ("%sentinel = s32[1] constant({-1})", "%sentinel = s32[1] constant({0})"),
        ("%final_int = s32[] convert(%final)", "%final_int = s32[] convert(%i17)"),
        ("%no_token = s32[1] constant({-1})", "%no_token = s32[1] constant({0})"),
        ("%good = pred[] constant(true)", "%good = pred[] constant(false)"),
        ("and(%incoming, %head_health)", "and(%incoming, %incoming)"),
        ("%slice_done0, %slice_done1", "%slice_done1, %slice_done0"),
        ("%slice_done0, %slice_done1", "%slice_done0, %slice_done0"),
        ("slice-start(%i3), slice={[6:12]", "slice-start(%i11), slice={[6:12]"),
        ("[18:21], [0:16]", "[18:21], [1:16]"),
        ('custom_call_target="ConcatBitcast"', 'custom_call_target="Other"'),
        ("copy-start(%reconstructed)", "copy-start(%i11)"),
        ("%i8, %o6, %o7", "%i4, %o6, %o7"),
    ],
)
def test_one_mutated_boundary_refuses(old, new):
    original = fixture()
    assert old in original
    result = check(original.replace(old, new, 1))
    assert not result["passed"], (old, new, result)


def test_same_shape_layout_changing_bitcast_is_not_rollback_identity():
    text = (
        fixture()
        .replace(
            "%rollback_args =",
            f"%layout_change = {IX}{{0,1,2,3}} bitcast(%original_index)\n%rollback_args =",
        )
        .replace("tuple(%i2, %original_index, %i4", "tuple(%i2, %layout_change, %i4")
    )
    result = check(text)
    assert not result["passed"]
    assert "bitcast" in result["error"]


def test_duplicate_physical_group_cannot_hide_in_frozenset():
    text = fixture().replace(
        "replica_groups={{0,1,2,3},", "replica_groups={{0,1,2,3},{0,1,2,3},", 1
    )
    assert not check(text)["passed"]


def test_metadata_cannot_spoof_sentinel_or_final_direction():
    text = fixture().replace(
        "%sentinel = s32[1] constant({-1})",
        '%sentinel = s32[1] constant({0}), metadata={op_name="constant({-1})"}',
    )
    assert not check(text)["passed"]
    text = fixture().replace(
        "compare(%end, %i12), direction=EQ",
        'compare(%end, %i12), direction=NE, metadata={op_name="direction=EQ"}',
    )
    assert not check(text)["passed"]


def test_narrow_proof_discloses_remaining_cache_health_and_memory_obligations():
    result = check(fixture())
    assert result["passed"]
    assert result == json.loads(json.dumps(result))
    assert result["rollback_inputs"]["0"] == 2
    assert "PROPOSED_CACHE_WRITES_AND_UNREPAIRED_PROVENANCE" in result["not_proven"]
    assert "ALL_LAYER_HEALTH_CONTRIBUTIONS" in result["not_proven"]
    assert "PHYSICAL_ALIASING_OR_MEMORY_FEASIBILITY" in result["not_proven"]


@pytest.mark.parametrize("graph,rows", [("prefill_chunk", 17), ("prefill_tail", 11)])
def test_original_acquired_commit(graph, rows):
    repo = Path(__file__).resolve().parents[3]
    receipt = json.loads(
        (
            repo
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
        pytest.skip("original acquired HLO unavailable locally")
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == receipt["graphs"][graph]["optimized_hlo_sha256"]
    result = check(raw.decode(), rows)
    assert result["passed"], result
