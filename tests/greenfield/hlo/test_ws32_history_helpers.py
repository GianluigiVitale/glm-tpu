"""Seven saved DB611 graphs and local structural mutations; no compilation."""

from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import re

import pytest

from glm_tpu.greenfield.benchmarking.ws32_batched_helper_hlo import _scratch_pairs, _target
from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import PrefillHloIndex
from glm_tpu.greenfield.benchmarking.ws32_pallas_one_layer import _computation_base, _live_instruction_closure
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_helper_hlo import _merge_scratch
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from scripts.greenfield import ws32_history_helpers as helpers

REPO = Path(__file__).resolve().parents[3]
ROOT = Path("/home/gianl/glm-run/greenfield_fp8_ws32_history_frontier_compile_20260911T183647058550669Z/rank0")


@pytest.fixture(scope="module", autouse=True)
def no_compilation():
    import jax
    def forbidden(*a, **k):
        pytest.fail("saved-HLO helper test initialized devices or compiled")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(jax, "devices", forbidden)
        patch.setattr(jax, "device_put", forbidden)
        patch.setattr(jax.stages.Lowered, "compile", forbidden)
        patch.setattr(jax.stages.Compiled, "__call__", forbidden)
        yield


@pytest.fixture(scope="module")
def originals():
    receipt = json.loads((REPO / "docs/artifacts/prefill-history-seven-graph-db611-sealed-20260911.json").read_text())
    result = {}
    for name in helpers.PROGRAMS:
        raw = (ROOT / f"{name}.optimized_hlo.txt").read_bytes()
        pin = receipt["graphs"][name]
        assert (len(raw), sha256(raw).hexdigest()) == (pin["optimized_hlo_bytes"], pin["optimized_hlo_sha256"])
        index = PrefillHloIndex(parse_hlo_module(raw.decode()))
        live = _live_instruction_closure(index.module.instructions)
        report = helpers.check_history_helpers(index, name=name, live_instructions=live)
        assert report["passed"], (name, report)
        result[name] = index, live, report
    return result


def rewrite(index, op, **changes):
    changed = replace(op, **changes)
    return PrefillHloIndex(replace(index.module, instructions=tuple(
        changed if node.index == op.index else node for node in index.module.instructions)))


def escape(index, op):
    extra = replace(op, index=max(o.index for o in index.module.instructions) + 1,
                    name="%history_test_escape", opcode="copy", raw_opcode="copy",
                    operand_names=(op.name,), operand_shapes=op.result_shapes,
                    raw_line=f"%history_test_escape = copy({op.name})")
    return PrefillHloIndex(replace(index.module, instructions=(*index.module.instructions, extra)))


def find(index, name):
    found = [o for o in index.module.instructions if o.name == name]
    assert len(found) == 1
    return found[0]


def refuses(index, live, name):
    report = helpers.check_history_helpers(index, name=name, live_instructions=live)
    assert not report["passed"], report
    assert report["error"]


@pytest.mark.parametrize("name, count", zip(helpers.PROGRAMS, (213, 209, 213, 215, 26, 7, 117)))
def test_all_seven_actual_helper_inventories(originals, name, count):
    _, _, report = originals[name]
    assert report["helper_count"] == sum(helpers.expected_helpers(name).values()) == count
    assert report == json.loads(json.dumps(report))
    assert report["numerical_promotion"] is report["performance_claim"] is False
    if name in helpers.PROGRAMS[:4]:
        pairs = report["scratch_pairs"]
        assert Counter(p["layer"] for p in pairs if p["kind"] == "panel_search_pair") == Counter({3: 2, 4: 2, 5: 2, 6: 2})
        assert Counter(p["layer"] for p in pairs if p["kind"] == "merge_complete_halves") == Counter({0: 2, 1: 2, 2: 2, 6: 2})
        assert sum(len(p.get("scratch_copies", ())) for p in pairs) == (8 if name.endswith("128") else 0)
    elif name == "observer":
        assert len(report["scratch_pairs"]) == 3
        assert all(p["rows"] == len(p["chain"]) == 32 for p in report["scratch_pairs"])


@pytest.mark.parametrize("name", helpers.PROGRAMS)
@pytest.mark.parametrize("case", ("missing", "side_effect", "dead"))
def test_inventory_effect_liveness_refusals(originals, name, case):
    index, live, _ = originals[name]
    op = next(o for o in index.module.instructions if o.raw_opcode == "custom-call" and _target(o) != "tpu_custom_call")
    if case == "missing":
        changed = PrefillHloIndex(replace(index.module, instructions=tuple(o for o in index.module.instructions if o.index != op.index)))
    elif case == "side_effect":
        changed = rewrite(index, op, raw_line=op.raw_line + ", custom_call_has_side_effect=true")
    else:
        changed = index
        live = [o for o in live if o.index != op.index]
    refuses(changed, live, name)


@pytest.mark.parametrize("name", ("exact_promote", "observer"))
@pytest.mark.parametrize("case", ("order", "span", "mixed_source", "escape", "incomplete"))
def test_special_axis2_copies_are_closed_and_ordered(originals, name, case):
    index, live, _ = originals[name]
    op = find(index, "%custom-call.4" if name == "exact_promote" else "%custom-call.112")
    done = index.operand(op, 0)
    start = index.operand(done, 0)
    if case == "order":
        args = list(op.operand_names)
        args[0], args[1] = args[1], args[0]
        changed = rewrite(index, op, operand_names=tuple(args))
    elif case == "span":
        changed = rewrite(index, start, raw_line=start.raw_line.replace("[0:21]", "[1:22]"))
        assert find(changed, start.name).raw_line != start.raw_line
    elif case == "mixed_source":
        other = next(o for o in index.computations["ENTRY"].values()
                     if o.opcode == "parameter" and o.result_shapes == op.result_shapes
                     and o.name != start.operand_names[0])
        changed = rewrite(index, start, operand_names=(other.name,))
    elif case == "escape":
        changed = escape(index, done)
    else:
        changed = rewrite(index, done, raw_opcode="slice-start")
    refuses(changed, live, name)


@pytest.mark.parametrize("case", ("scope", "shape", "arity", "name"))
def test_decode_concatenate_annotation_exception_is_exact(originals, case):
    index, live, _ = originals["exact_decode"]
    op = find(index, "%custom-call.1")
    if case == "scope":
        changed = rewrite(index, op, op_name=op.op_name.replace("/concatenate", "/gather"))
    elif case == "shape":
        changed = rewrite(index, op, operand_shapes=(replace(op.result_shapes[0], dimensions=(1024, 2048, 1)),))
    elif case == "arity":
        changed = rewrite(index, op, operand_names=(*op.operand_names, *op.operand_names))
    else:
        changed = rewrite(index, op, name="%unregistered_annotation")
    refuses(changed, live, "exact_decode")


@pytest.mark.parametrize("case", ("search_layer", "search_escape", "search_geometry", "merge_layer", "merge_half", "merge_escape"))
def test_frontier_search_and_merge_scratch(originals, case):
    index, live, report = originals["candidate_b128"]
    if case.startswith("search"):
        row = next(p for p in report["scratch_pairs"] if p["kind"] == "panel_search_pair")
        op = find(index, row["allocations"][0])
        if case == "search_layer":
            loop = find(index, row["loop"])
            changed = rewrite(index, loop, op_name=loop.op_name.replace(f"layer_{row['layer']}/", "layer_7/"))
        elif case == "search_escape":
            changed = escape(index, op)
        else:
            changed = rewrite(index, op, result_shapes=(replace(op.result_shapes[0], dimensions=(999,)),))
    else:
        row = next(p for p in report["scratch_pairs"] if p["kind"] == "merge_complete_halves")
        completed = find(index, row["completion"])
        root = index.roots[index.callee(completed, "calls")]
        if case == "merge_layer":
            changed = rewrite(index, root, op_name=root.op_name.replace(f"layer_{row['layer']}/", "layer_7/"))
        elif case == "merge_half":
            offset = index.operand(root, 3)
            changed = rewrite(index, offset, raw_line=offset.raw_line.replace("constant(1)", "constant(0)"))
        else:
            changed = escape(index, index.operand(root, 0))
    refuses(changed, live, "candidate_b128")


@pytest.mark.parametrize("case", ("start_escape", "done_escape", "handle_shape", "unfinished", "allocation_escape"))
def test_b128_panel_async_copy_is_closed_and_complete(originals, case):
    index, live, report = originals["candidate_b128"]
    copied = next(c for p in report["scratch_pairs"] for c in p.get("scratch_copies", ()))
    start, done = find(index, copied["start"]), find(index, copied["done"])
    if case == "start_escape":
        changed = escape(index, start)
    elif case == "done_escape":
        changed = escape(index, done)
    elif case == "allocation_escape":
        changed = escape(index, find(index, copied["allocation"]))
    elif case == "handle_shape":
        shapes = list(start.result_shapes)
        shapes[0] = replace(shapes[0], dimensions=(62,))
        changed = rewrite(index, start, result_shapes=tuple(shapes))
    else:
        changed = rewrite(index, done, raw_opcode="copy-start")
    refuses(changed, live, "candidate_b128")


@pytest.mark.parametrize("case", ("initial_escape", "partial_escape", "parameter_escape", "row_hole", "duplicate_binding", "wrong_base", "early_reduce"))
def test_observer_complete32rows_before_any_escape(originals, case):
    index, live, report = originals["observer"]
    row = report["scratch_pairs"][0]
    first, second = (find(index, n) for n in row["chain"][:2])
    root = index.roots[index.callee(second, "calls")]
    base = index.operand(root, 0)
    if case == "initial_escape":
        changed = escape(index, find(index, row["allocation"]))
    elif case == "partial_escape":
        changed = escape(index, first)
    elif case == "parameter_escape":
        changed = escape(index, base)
    elif case == "row_hole":
        offset = index.operand(root, 2)
        assert "constant(1)" in offset.raw_line
        changed = rewrite(index, offset, raw_line=offset.raw_line.replace("constant(1)", "constant(0)"))
    elif case == "duplicate_binding":
        changed = rewrite(index, second, operand_names=(*second.operand_names, first.name),
                          operand_shapes=(*second.operand_shapes, first.result_shapes[0]))
    elif case == "wrong_base":
        args = list(second.operand_names)
        number = int(re.search(r"parameter\((\d+)\)", base.raw_line)[1])
        args[number] = args[1 - number]
        changed = rewrite(index, second, operand_names=tuple(args))
    else:
        changed = rewrite(index, second, opcode="reduce", raw_opcode="reduce")
    refuses(changed, live, "observer")


@pytest.mark.parametrize("layers", ((False, 1), (0, True, 2, 6), [0, 1], (0, 1, 2, 3), (3, 4, 5, 7)))
def test_shared_optional_scopes_do_not_widen_historical_defaults(layers):
    with pytest.raises(ValueError, match="layer scope"):
        _merge_scratch(None, (), set(), layer_ids=layers)
    with pytest.raises(ValueError, match="layer scope"):
        _scratch_pairs(None, (), 128, set(), panel_search=True, layer_ids=layers)


def test_default_scope_still_requires_full_original_counts():
    with pytest.raises(ValueError, match="expected300"):
        _scratch_pairs(None, (), 128, set(), panel_search=True)
    with pytest.raises(ValueError, match="expected450"):
        _scratch_pairs(None, (), 17, set())
    with pytest.raises(ValueError, match="wrong merge scratch allocation count"):
        _merge_scratch(None, (), set())
