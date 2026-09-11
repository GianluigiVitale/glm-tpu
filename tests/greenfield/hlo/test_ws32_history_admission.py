"""Saved actual DB611 graphs and narrow refusal tests; no compilation/payloads."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from scripts.greenfield import ws32_history_admission as admission
from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import (
    PrefillHloIndex, _check_moe_route_sums,
)
from glm_tpu.greenfield.benchmarking.ws32_pallas_one_layer import _live_instruction_closure
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def receipt():
    return admission.registration(REPO)


@pytest.fixture(scope="module")
def originals(receipt):
    root = Path(receipt["local_run_root"]) / "fleet/rank0"
    return {name: ((root / f"{name}.stablehlo.mlir").read_text(),
                   (root / f"{name}.optimized_hlo.txt").read_text(),
                   saved["compiled_memory"])
            for name, saved in receipt["graphs"].items()}


@pytest.mark.parametrize("name", admission.compiler.PROGRAMS)
def test_all_seven_actual_db611_reports(name, originals, receipt):
    report = admission.inspect_program(name, *originals[name], repo=REPO)
    assert report == json.loads(json.dumps(report, allow_nan=False))
    assert report["passed"] and report["profile"] == admission.PROFILE
    assert report["acquisition_sha256"] == admission.RECEIPT_SHA256
    assert report["structure"]["helpers"]["passed"]
    assert not report["numerical_promotion"] and not report["performance_claim"]
    assert report["structure"]["static_collective_count"] == receipt["graphs"][name]["independent_inventory"]["static_collectives"]
    if name.startswith(("candidate_", "control_")):
        frontier = report["structure"]["frontier"]
        assert len(frontier["prefix_bodies"]) == 7
        assert len(frontier["canonical_dense_bodies"]) == (3 if name.startswith("candidate") else 0)
        assert [r["layer"] for r in frontier["fp32_moe"]["layers"]] == [3, 4, 5, 6]
    if name.startswith("exact_"):
        assert report["structure"]["materializer"]["passed"]


@pytest.mark.parametrize("defect", ["raw", "optimized", "memory", "alias", "bool", "missing", "unknown"])
def test_identity_refusals_precede_structural_inspection(defect, originals, monkeypatch):
    name = "candidate_b128"
    raw, optimized, memory = originals[name]
    memory = dict(memory)
    if defect == "raw":
        raw += "\n"
    elif defect == "optimized":
        optimized += "\n"
    elif defect == "memory":
        memory["temp_size_in_bytes"] -= 128
    elif defect == "alias":
        memory["alias_size_in_bytes"] = 1
    elif defect == "bool":
        memory["alias_size_in_bytes"] = False
    elif defect == "missing":
        del memory["temp_size_in_bytes"]
    else:
        name = "unknown"
    monkeypatch.setattr(admission, "inspect_structure", lambda *a: pytest.fail("reached structure on identity refusal"))
    with pytest.raises(ValueError):
        admission.inspect_program(name, raw, optimized, memory, repo=REPO)


def test_changed_receipt_or_source_refused(tmp_path, monkeypatch):
    # Receipt-only fixtures do not impersonate the frozen production source.
    calls = []
    monkeypatch.setattr(admission.compiler, "require_source", lambda p: calls.append(p))
    path = tmp_path / admission.RECEIPT
    path.parent.mkdir(parents=True)
    with pytest.raises(ValueError):
        admission.registration(tmp_path)
    path.write_bytes((REPO / admission.RECEIPT).read_bytes() + b"\n")
    with pytest.raises(ValueError, match="receipt changed"):
        admission.registration(tmp_path)
    assert calls == [tmp_path, tmp_path]
    path.unlink()
    path.symlink_to(REPO / admission.RECEIPT)
    with pytest.raises(ValueError, match="linked"):
        admission.registration(tmp_path)


@pytest.fixture(scope="module")
def candidate(originals):
    module = parse_hlo_module(originals["candidate_b128"][1])
    return PrefillHloIndex(module), _live_instruction_closure(module.instructions)


def test_reduced_scope_never_weakens_historical_moe_default(candidate):
    index, live = candidate
    args = dict(block_rows=128, live_instructions=live, router_bias_tuple=True)
    assert not _check_moe_route_sums(index, **args)["passed"]
    assert _check_moe_route_sums(index, layer_ids=(3, 4, 5, 6), **args)["passed"]
    for scope in ((3, 4, 5), (3, 4, 5, 6.0), [3, 4, 5, 6], (6, 5, 4, 3)):
        assert not _check_moe_route_sums(index, layer_ids=scope, **args)["passed"]


@pytest.mark.parametrize("defect", ["operand", "axis", "zero", "dtype", "dead"])
def test_scoped_actual_moe_refusals(candidate, defect):
    index, live = candidate
    combine = next(op for op in index.module.instructions if op.opcode == "all-reduce"
        and "/layer_3/greenfield_ws32_prefill_moe/expert_reduce/" in (op.op_name or ""))
    if defect == "dead":
        live = tuple(op for op in live if op.index != combine.index)
    else:
        report = _check_moe_route_sums(index, block_rows=128, live_instructions=live,
                                     router_bias_tuple=True, layer_ids=(3, 4, 5, 6))
        scope, name, _ = report["layers"][0]["path"][-1]
        reduce = index.computations[scope][name]
        if defect == "operand":
            target = combine
            other = next(op for op in index.module.instructions if op.opcode == "all-reduce"
                and "/layer_4/greenfield_ws32_prefill_moe/expert_reduce/" in (op.op_name or ""))
            modified = replace(combine, operand_names=other.operand_names)
        elif defect == "axis":
            target, modified = reduce, replace(reduce, raw_line=reduce.raw_line.replace("dimensions={1}", "dimensions={0}"))
        elif defect == "zero":
            target = index.operand(reduce, 1)
            modified = replace(target, raw_line=target.raw_line.replace("constant(0)", "constant(-0)"))
        else:
            target = index.operand(reduce, 0)
            modified = replace(target, result_shapes=tuple(replace(s, dtype="bf16") for s in target.result_shapes))
        assert modified != target
        index = PrefillHloIndex(replace(index.module, instructions=tuple(
            modified if op.index == target.index else op for op in index.module.instructions)))
    result = _check_moe_route_sums(index, block_rows=128, live_instructions=live,
        router_bias_tuple=True, layer_ids=(3, 4, 5, 6))
    assert not result["passed"], result


@pytest.mark.parametrize("rows,canonical", [(128, True), (114, True), (128, False), (114, False)])
def test_history_expected_schedule_is_only_seven_layers(rows, canonical):
    result = admission.expected_collectives(rows, canonical)
    assert sum(result.values()) == 99
    assert {key[1] for key in result} == {-1, *range(7)}
    assert not any(key[1] == 6 and key[-2] == ("f32", (256,)) for key in result)
    assert sum(v for k, v in result.items() if k[4] == "minimum") == 2
