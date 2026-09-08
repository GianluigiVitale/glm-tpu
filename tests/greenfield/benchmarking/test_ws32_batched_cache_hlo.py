"""Storage-owner mutations plus the original complete compiler graphs."""

from hashlib import sha256
import json
from pathlib import Path

import pytest

from glm_tpu.greenfield.benchmarking.ws32_batched_cache_hlo import (
    IndexCachePaths,
    check_batched_index_cache_stacks,
)
from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import PrefillHloIndex
from glm_tpu.greenfield.benchmarking.ws32_batched_prefill import (
    UNREGISTERED,
    inspect_ws32_batched_prefill_hlo,
)
from glm_tpu.greenfield.benchmarking.ws32_prefill_hlo_identity import (
    PrefillIdentity,
    Value,
)
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module


IX = "bf16[21,16,64,128]{3,2,1,0:T(8,128)(2,1)}"
SLOT = "bf16[1,16,64,128]{3,2,1,0:T(8,128)(2,1)}"
FLAT = "bf16[1024,128]{1,0:T(8,128)(2,1)}"


def fixture(rows=17):
    fields = f"({SLOT}, s32[{rows}], bf16[{rows},128])"
    text = [
        "HloModule cache, num_partitions=32",
        """%replace {
%old = bf16[] parameter(0)
ROOT %update = bf16[] parameter(1)
}""",
        f"""%unchanged {{
ROOT %p = {SLOT} parameter(0)
}}
%write {{
%p = {fields} parameter(0)
%old = {SLOT} get-tuple-element(%p), index=0
%indices = s32[{rows}] get-tuple-element(%p), index=1
%updates = bf16[{rows},128] get-tuple-element(%p), index=2
%flat = {FLAT} bitcast(%old)
%scatter = {FLAT} scatter(%flat, %indices, %updates), update_window_dims={{1}}, inserted_window_dims={{0}}, scatter_dims_to_operand_dims={{0}}, index_vector_dim=1, to_apply=%replace
ROOT %out = {SLOT} bitcast(%scatter)
}}
ENTRY %main {{
%p = ({', '.join([IX]*12)}) parameter(0)
%cache = {IX} get-tuple-element(%p), index=11
%zero = s32[] constant(0)
%valid = s32[] constant(1)
%indices = s32[{rows}] constant(0)
%updates = bf16[{rows},128] constant(0)
""",
    ]
    previous = "%cache"
    for slot in range(21):
        text += [
            f"%i{slot} = s32[] constant({slot})",
            f"%old{slot} = {SLOT} slice({previous}), slice={{[{slot}:{slot+1}], [0:16], [0:64], [0:128]}}",
            f"%args{slot} = {fields} tuple(%old{slot}, %indices, %updates)",
            f"%write{slot} = {SLOT} conditional(%valid, %old{slot}, %args{slot}), branch_computations={{%unchanged, %write}}",
            f"{'ROOT ' if slot==20 else ''}%stack{slot} = {IX} dynamic-update-slice({previous}, %write{slot}, %i{slot}, %zero, %zero, %zero)",
        ]
        previous = f"%stack{slot}"
    return "\n".join(text) + "\n}"


def inspect(text, rows=17):
    index = PrefillHloIndex(parse_hlo_module(text))
    return IndexCachePaths(index, rows).stack(Value(index.roots["ENTRY"]), 11)


@pytest.mark.parametrize("rows", [11, 17])
def test_own_original_slots_through_previous_disjoint_writes(rows):
    records = inspect(fixture(rows), rows)
    assert [r["slot"] for r in records] == list(range(21))
    assert [r["layer"] for r in records] == [0, 1, 2, *range(6, 78, 4)]


@pytest.mark.parametrize(
    "old,new",
    [
        ("index=11", "index=3"),
        ("%i19 = s32[] constant(19)", "%i19 = s32[] constant(20)"),
        ("%i20, %zero, %zero, %zero", "%i20, %valid, %zero, %zero"),
        ("[20:21], [0:16]", "[19:20], [0:16]"),
        ("tuple(%old20, %indices, %updates)", "tuple(%old19, %indices, %updates)"),
        ("%stack19, %write20", "%stack18, %write20"),
        ("update_window_dims={1}", "update_window_dims={0}"),
        ("inserted_window_dims={0}", "inserted_window_dims={1}"),
        ("scatter_dims_to_operand_dims={0}", "scatter_dims_to_operand_dims={1}"),
        ("index_vector_dim=1", "index_vector_dim=0"),
        ("ROOT %update = bf16[] parameter(1)", "ROOT %update = bf16[] add(%old, %old)"),
        ("{1,0:T(8,128)(2,1)}", "{0,1:T(8,128)(2,1)}"),
        ("{3,2,1,0:T(8,128)(2,1)}", "{3,1,2,0:T(8,128)(2,1)}"),
    ],
)
def test_wrong_storage_refuses(old, new):
    text = fixture()
    assert old in text
    with pytest.raises(ValueError):
        inspect(text.replace(old, new))


def test_shape_boundary_does_not_broaden_identity():
    index = PrefillHloIndex(parse_hlo_module(fixture()))
    ssa = PrefillIdentity(index)
    value = Value(index.computations["%write"]["%out"])
    with pytest.raises(ValueError, match="nonidentity"):
        ssa.resolve(value)
    assert ssa.resolve(value, stop_at_shape_change=True) == value


def tuple_fixture():
    text = fixture()
    view = "bf16[16,64,128]{2,1,0:T(8,128)(2,1)}"
    pair = f"({view}, {SLOT})"
    text = text.replace(
        f"ROOT %p = {SLOT} parameter(0)",
        f"%p = {SLOT} parameter(0)\n%view = {view} bitcast(%p)\nROOT %pair = {pair} tuple(%view, %p)",
    ).replace(
        f"ROOT %out = {SLOT} bitcast(%scatter)",
        f"%out = {SLOT} bitcast(%scatter)\n%view = {view} bitcast(%scatter)\nROOT %pair = {pair} tuple(%view, %out)",
    )
    for slot in range(21):
        text = text.replace(
            f"%write{slot} = {SLOT} conditional", f"%write{slot} = {pair} conditional"
        )
        start = f"{'ROOT ' if slot==20 else ''}%stack{slot} ="
        text = text.replace(
            start,
            f"%chosen{slot} = {SLOT} get-tuple-element(%write{slot}), index=1\n"
            + start,
        )
        text = text.replace(f", %write{slot}, %i{slot}", f", %chosen{slot}, %i{slot}")
    return text


def test_selected_unrepaired_conditional_leaf():
    text = tuple_fixture()
    assert len(inspect(text)) == 21
    wrong = text.replace(
        "get-tuple-element(%write20), index=1", "get-tuple-element(%write20), index=0"
    )
    with pytest.raises(ValueError, match="conditional/leaf"):
        inspect(wrong)


@pytest.mark.parametrize("rows", [1, 12, 32, True, 17.0])
def test_unknown_profile_refuses(rows):
    with pytest.raises(ValueError, match="B17/B11"):
        check_batched_index_cache_stacks(
            PrefillHloIndex(parse_hlo_module(fixture())), block_rows=rows
        )


@pytest.mark.parametrize("graph,rows", [("prefill_chunk", 17), ("prefill_tail", 11)])
def test_original_storage_ownership(graph, rows):
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
    stable = path.with_name(graph + ".stablehlo.mlir")
    if not path.exists() or not stable.exists():
        pytest.skip("original optimized HLO unavailable")
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == receipt["graphs"][graph]["optimized_hlo_sha256"]
    integrated = inspect_ws32_batched_prefill_hlo(
        stable.read_text(),
        raw.decode(),
        block_rows=rows,
        expected_stablehlo_sha256=receipt["graphs"][graph]["stablehlo_sha256"],
        expected_optimized_hlo_sha256=receipt["graphs"][graph]["optimized_hlo_sha256"],
    )
    result = integrated["index_cache_storage_proof"]
    assert result["passed"], result
    assert integrated["violations"] == [UNREGISTERED], integrated["violations"]
    assert not integrated["passed"] and not integrated["profile_registered"]
    for kind in ("repaired", "unrepaired"):
        assert [r["slot"] for r in result["stacks"][kind]] == list(range(21))
    assert "ROW_INDICES_AND_MASK_VALUES" in result["not_proven"]
