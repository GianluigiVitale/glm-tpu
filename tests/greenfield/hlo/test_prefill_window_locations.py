"""Original four-graph identity and narrow coordinate-only mutations."""

import re

import pytest

from scripts.greenfield import prefill_window_admission as admission
from scripts.greenfield.prefill_window_locations import location_identity
from tests.greenfield.hlo.test_prefill_window_admission import originals


@pytest.mark.parametrize("name", admission.PROGRAMS)
def test_original_equivalence_pins_and_allowed_host_shifts(name, originals):
    stable, hlo, memory = originals[name]
    identity = location_identity(hlo)
    assert (
        identity["host_location_equivalence_sha256"] == admission.HOST_EQUIVALENCE[name]
    )
    assert (
        identity["raw_optimized_hlo_sha256"]
        == admission.registered_programs()[name]["optimized_hlo_sha256"]
    )
    locations = identity["host_locations"]
    assert len(locations) == 7
    shifted = hlo
    for location in locations:
        pattern = rf"(?m)^({location['location_id']} \{{file_name_id=\d+ function_name_id=\d+ )line=\d+ end_line=\d+ column=\d+ end_column=\d+(\}})$"
        shifted, count = re.subn(
            pattern,
            lambda m: m[1] + "line=1234 end_line=1235 column=123 end_column=124" + m[2],
            shifted,
        )
        assert count == 1
    report = admission.inspect_program(name, stable, shifted, memory)
    assert report["passed"] and not report["raw_graph_pair_exact"]
    assert report["optimized_hlo_sha256"] != identity["raw_optimized_hlo_sha256"]
    assert (
        report["host_coordinate_identity"]["host_location_equivalence_sha256"]
        == admission.HOST_EQUIVALENCE[name]
    )
    admission.validate_program_report(report, name, stable, shifted, memory)


@pytest.mark.parametrize("name", admission.PROGRAMS)
@pytest.mark.parametrize(
    "change",
    [
        "model_location",
        "function",
        "stack",
        "instruction_metadata",
        "backend",
        "extra_host",
    ],
)
def test_nothing_else_is_normalized(name, change, originals):
    stable, hlo, memory = originals[name]
    host_ids = {r["location_id"] for r in location_identity(hlo)["host_locations"]}
    if change == "model_location":
        pattern = r"(?m)^(\d+) \{file_name_id=\d+ function_name_id=\d+ line=\d+ end_line=\d+ column=\d+ end_column=\d+\}$"
        target = next(
            m[0] for m in re.finditer(pattern, hlo) if int(m[1]) not in host_ids
        )
        hlo = hlo.replace(target, re.sub(r"line=\d+", "line=98765", target, count=1), 1)
    elif change == "function":
        hlo = hlo.replace('"compile_program"', '"execute_numerical"', 1)
    elif change == "stack":
        hlo, count = re.subn(
            r"parent_frame_id=\d+", "parent_frame_id=98765", hlo, count=1
        )
        assert count == 1
    elif change == "instruction_metadata":
        hlo, count = re.subn(
            r"stack_frame_id=\d+", "stack_frame_id=98765", hlo, count=1
        )
        assert count == 1
    elif change == "backend":
        hlo, count = re.subn(
            r'"estimated_cycles":"\d+"', '"estimated_cycles":"98765"', hlo, count=1
        )
        assert count == 1
    elif change == "extra_host":
        loc = next(iter(host_ids))
        line = re.search(rf"(?m)^{loc} \{{file_name_id=.+$", hlo)[0]
        hlo = hlo.replace(
            "\n\nStackFrames",
            "\n" + re.sub(r"^\d+", "98765", line) + "\n\nStackFrames",
            1,
        )
    with pytest.raises(ValueError):
        admission.inspect_program(name, stable, hlo, memory)
