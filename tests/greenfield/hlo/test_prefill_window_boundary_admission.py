"""Original DB591 graph/schema replay and simultaneous diagnostic memory."""

from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from scripts.greenfield import prefill_window_boundary_admission as admission
from scripts.greenfield.prefill_window_locations import location_identity
from tests.greenfield.hlo.test_prefill_window_admission import memory_inputs

ORIGINAL = Path(
    "/home/gianl/glm-run/greenfield_fp8_ws32_prefill_window_boundary_acquisition_l6_"
    "20260908T143917951158196Z/fleet/rank0"
)


@pytest.fixture(scope="module")
def originals():
    if not ORIGINAL.is_dir():
        pytest.skip("requires original DB591 materialized acquisition")
    record = json.loads((ORIGINAL / "runner.json").read_text())
    return {
        name: (
            (ORIGINAL / f"{name}.stablehlo.mlir").read_text(),
            (ORIGINAL / f"{name}.optimized_hlo.txt").read_text(),
            pin["compiled_memory"],
            record["programs"][name].get("compiler_output_schema"),
        )
        for name, pin in admission.registered_programs().items()
    }


@pytest.mark.parametrize("name", admission.PROGRAMS)
def test_original_graph_and_serialized_report(name, originals):
    stable, hlo, memory, schema = originals[name]
    report = admission.inspect_program(name, stable, hlo, memory, output_schema=schema)
    assert report["passed"] and report["raw_graph_pair_exact"]
    assert not report["numerical_execution_authorized"]
    assert json.loads(json.dumps(report)) == report
    admission.validate_program_report(
        report, name, stable, hlo, memory, output_schema=schema
    )
    if name in ("candidate", "control"):
        assert report["fp32_route_sum"]["passed"]


@pytest.mark.parametrize("name", admission.PROGRAMS)
def test_only_existing_seven_host_coordinates_may_move(name, originals):
    stable, hlo, memory, schema = originals[name]
    for location in location_identity(hlo)["host_locations"]:
        pattern = rf"(?m)^({location['location_id']} \{{file_name_id=\d+ function_name_id=\d+ )line=\d+ end_line=\d+ column=\d+ end_column=\d+(\}})$"
        hlo, count = re.subn(
            pattern,
            lambda m: m[1] + "line=111 end_line=112 column=11 end_column=12" + m[2],
            hlo,
        )
        assert count == 1
    report = admission.inspect_program(name, stable, hlo, memory, output_schema=schema)
    assert report["passed"] and not report["raw_graph_pair_exact"]


@pytest.mark.parametrize("name", admission.PROGRAMS)
@pytest.mark.parametrize("change", ["stable", "body", "stack", "memory", "bool"])
def test_graph_or_allocation_drift_refuses(name, change, originals):
    stable, hlo, memory, schema = originals[name]
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
        admission.inspect_program(name, stable, hlo, memory, output_schema=schema)


@pytest.mark.parametrize("change", ["absent", "field", "shape", "dtype", "sharding"])
def test_capture_envelope_is_exact_not_only_shape_valid(change, originals):
    stable, hlo, memory, original_schema = originals["candidate"]
    schema = deepcopy(original_schema)
    if change == "absent":
        schema = None
    elif change == "field":
        key = next(iter(schema["captures"]))
        schema["captures"][key + "_wrong"] = schema["captures"].pop(key)
        schema["capture_partition_specs"][key + "_wrong"] = schema[
            "capture_partition_specs"
        ].pop(key)
    elif change == "shape":
        # A different original leaf can pass the generic envelope but not its hash.
        schema["original"][1]["shape"][-1] += 1
    elif change == "dtype":
        schema["original"][1]["dtype"] = "float32"
    else:
        schema["capture_partition_specs"]["router/input"] = ["feature", "expert"]
    with pytest.raises(ValueError):
        admission.inspect_program(
            "candidate", stable, hlo, memory, output_schema=schema
        )


def test_no_db590_model_identity_or_memory_reuse(originals):
    from scripts.greenfield import prefill_window_admission as old

    for name in ("candidate", "control"):
        stable, hlo, _, schema = originals[name]
        with pytest.raises(ValueError, match="allocation"):
            admission.inspect_program(
                name,
                stable,
                hlo,
                old.registered_programs()[name]["compiled_memory"],
                output_schema=schema,
            )
        assert admission.HOST_EQUIVALENCE[name] != old.HOST_EQUIVALENCE[name]


@pytest.mark.parametrize("change", ["type", "value", "tuple"])
def test_report_must_replay_with_json_types(change, originals):
    stable, hlo, memory, schema = originals["wk_decode"]
    report = admission.inspect_program("wk_decode", stable, hlo, memory)
    if change == "type":
        report["compiled_memory"]["alias_size_in_bytes"] = False
    elif change == "value":
        report["physical_collective_count"] += 1
    else:
        report["host_coordinate_identity"]["host_locations"] = tuple(
            report["host_coordinate_identity"]["host_locations"]
        )
    with pytest.raises(ValueError, match="report differs"):
        admission.validate_program_report(report, "wk_decode", stable, hlo, memory)


@pytest.mark.parametrize("name", admission.PROGRAMS)
def test_retained_captures_and_all_codes_count_toward_peak(name):
    census, _ = memory_inputs()
    analyses = {
        n: p["compiled_memory"] for n, p in admission.registered_programs().items()
    }
    budget = admission.memory_budget(census, analyses, active_graph=name)
    code = sum(p["generated_code_size_in_bytes"] for p in analyses.values())
    assert code == 67_827_200
    assert budget["required_reserve_bytes"] == 1 << 30
    assert budget["estimate_fits"] and not budget["numerical_admission"]
    expected = (
        340_000_000
        + code
        + analyses[name]["output_size_in_bytes"]
        + analyses[name]["temp_size_in_bytes"]
    )
    assert budget["devices"][0]["estimated_peak_bytes"] == expected
    retained, _ = memory_inputs(retained_bytes=40_000_000)
    result = admission.memory_budget(retained, analyses, active_graph=name)
    assert result["devices"][0]["estimated_peak_bytes"] == expected + 40_000_000


@pytest.mark.parametrize("change", ["missing", "bool", "old", "retained", "peak"])
def test_budget_never_ignores_capture_allocations_or_lifetime_peaks(change):
    census, old = memory_inputs()
    analyses = {
        n: dict(p["compiled_memory"])
        for n, p in admission.registered_programs().items()
    }
    if change == "retained":
        census, _ = memory_inputs(retained_bytes=32_000_000_000)
    elif change == "peak":
        census["devices"][0]["memory_stats"]["peak_bytes_in_use"] = 33_000_000_000
    elif change == "missing":
        analyses.pop("control")
    elif change == "bool":
        analyses["candidate"]["alias_size_in_bytes"] = False
    else:
        analyses = old
    if change in ("peak", "retained"):
        assert not admission.memory_budget(census, analyses, active_graph="candidate")[
            "estimate_fits"
        ]
    else:
        with pytest.raises(ValueError):
            admission.memory_budget(census, analyses, active_graph="candidate")
