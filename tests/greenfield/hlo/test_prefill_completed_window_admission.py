"""Replay DB593 originals; reject graph drift and missing live allocations."""

from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from scripts.greenfield import prefill_completed_window_admission as admission
from scripts.greenfield.prefill_window_locations import location_identity
from tests.greenfield.hlo.test_prefill_window_admission import memory_inputs

ORIGINAL = Path(
    "/home/gianl/glm-run/greenfield_fp8_ws32_prefill_completed_window_acquisition_l6_"
    "20260908T164531308007030Z/fleet/rank0"
)


@pytest.fixture(scope="module")
def originals():
    # This gate must actually exercise the captured compiler output, not skip.
    record = json.loads((ORIGINAL / "runner.json").read_text())
    return {
        name: (
            (ORIGINAL / f"{name}.stablehlo.mlir").read_text(),
            (ORIGINAL / f"{name}.optimized_hlo.txt").read_text(),
            record["programs"][name]["compiled_memory"],
        )
        for name in admission.PROGRAMS
    }


@pytest.mark.parametrize("name", admission.PROGRAMS)
def test_actual_graph_and_json_report(name, originals):
    report = admission.inspect_program(name, *originals[name])
    assert report["passed"] and report["raw_graph_pair_exact"]
    assert not report["numerical_execution_authorized"]
    assert json.loads(json.dumps(report)) == report
    admission.validate_program_report(report, name, *originals[name])
    if name in ("candidate", "control"):
        assert report["fp32_route_sum"]["passed"]


@pytest.mark.parametrize("name", admission.PROGRAMS)
def test_only_seven_existing_host_coordinates_may_move(name, originals):
    stable, hlo, memory = originals[name]
    for loc in location_identity(hlo)["host_locations"]:
        pattern = rf"(?m)^({loc['location_id']} \{{file_name_id=\d+ function_name_id=\d+ )line=\d+ end_line=\d+ column=\d+ end_column=\d+(\}})$"
        hlo, count = re.subn(
            pattern,
            lambda m: m[1] + "line=111 end_line=112 column=11 end_column=12" + m[2],
            hlo,
        )
        assert count == 1
    report = admission.inspect_program(name, stable, hlo, memory)
    assert report["passed"] and not report["raw_graph_pair_exact"]


@pytest.mark.parametrize("name", admission.PROGRAMS)
@pytest.mark.parametrize("change", ["stable", "body", "stack", "memory", "bool"])
def test_graph_and_compiler_memory_drift_refuse(name, change, originals):
    stable, hlo, memory = originals[name]
    memory = dict(memory)
    if change == "stable":
        stable += "\n"
    elif change == "body":
        hlo += "\n"
    elif change == "stack":
        hlo, count = re.subn(
            r"parent_frame_id=\d+", "parent_frame_id=99999", hlo, count=1
        )
        assert count == 1
    elif change == "memory":
        memory["output_size_in_bytes"] += 1
    else:
        memory["alias_size_in_bytes"] = False
    with pytest.raises(ValueError):
        admission.inspect_program(name, stable, hlo, memory)


@pytest.mark.parametrize("change", ["type", "value", "tuple"])
def test_json_report_types_and_values_are_bound(change, originals):
    args = originals["wk_decode"]
    report = admission.inspect_program("wk_decode", *args)
    if change == "type":
        report["compiled_memory"]["alias_size_in_bytes"] = False
    elif change == "value":
        report["checks"]["paired_collective_payloads"] = False
    else:
        report["host_coordinate_identity"]["host_locations"] = tuple(
            report["host_coordinate_identity"]["host_locations"]
        )
    with pytest.raises(ValueError, match="report differs"):
        admission.validate_program_report(report, "wk_decode", *args)


@pytest.mark.parametrize("name", admission.PROGRAMS)
def test_five_resident_codes_and_retained_prefixes_are_budgeted(name):
    census, _ = memory_inputs()
    analyses = {
        n: p["compiled_memory"] for n, p in admission.registered_programs().items()
    }
    budget = admission.memory_budget(census, analyses, active_graph=name)
    code = sum(p["generated_code_size_in_bytes"] for p in analyses.values())
    assert code == 25_306_624
    assert budget["required_reserve_bytes"] == 1 << 30
    assert budget["estimate_fits"] and not budget["numerical_admission"]
    expected = (
        340_000_000
        + code
        + analyses[name]["output_size_in_bytes"]
        + analyses[name]["temp_size_in_bytes"]
    )
    assert budget["devices"][0]["estimated_peak_bytes"] == expected
    retained, _ = memory_inputs(retained_bytes=50_000_000)
    result = admission.memory_budget(retained, analyses, active_graph=name)
    assert result["devices"][0]["estimated_peak_bytes"] == expected + 50_000_000


@pytest.mark.parametrize(
    "change", ["prefix", "extra", "bool", "old", "retained", "peak"]
)
def test_budget_cannot_omit_fifth_graph_or_live_buffers(change):
    census, old = memory_inputs()
    analyses = {
        n: deepcopy(p["compiled_memory"])
        for n, p in admission.registered_programs().items()
    }
    if change == "prefix":
        analyses.pop("prefix")
    elif change == "extra":
        analyses["extra"] = analyses["prefix"]
    elif change == "bool":
        analyses["prefix"]["alias_size_in_bytes"] = False
    elif change == "old":
        analyses = old
    elif change == "retained":
        census, _ = memory_inputs(retained_bytes=32_000_000_000)
    else:
        census["devices"][0]["memory_stats"]["peak_bytes_in_use"] = 33_000_000_000
    if change in ("retained", "peak"):
        assert not admission.memory_budget(census, analyses, active_graph="candidate")[
            "estimate_fits"
        ]
    else:
        with pytest.raises(ValueError):
            admission.memory_budget(census, analyses, active_graph="candidate")


def test_receipt_bytes_and_roles_are_not_interchangeable(
    tmp_path, monkeypatch, originals
):
    changed = tmp_path / "receipt.json"
    changed.write_bytes(admission.RECEIPT.read_bytes() + b"\n")
    monkeypatch.setattr(admission, "RECEIPT", changed)
    with pytest.raises(ValueError, match="receipt changed"):
        admission.registered_programs()


def test_wrong_program_role_refuses(originals):
    with pytest.raises(ValueError, match="identity changed"):
        admission.inspect_program("candidate", *originals["control"])
    with pytest.raises(ValueError, match="unknown"):
        admission.inspect_program("unknown", *originals["control"])
