"""Paired-only DSA coordinates; raw original and executable bytes stay bound."""

from hashlib import sha256
from pathlib import Path
import re

import pytest

from glm_tpu.greenfield.validation.ws32_hlo_worker_locations import (
    DSA,
    MOE,
    WS32,
    WORKER,
    paired_worker_dsa_location_identity,
    worker_location_identity,
)
from glm_tpu.greenfield.validation.ws32_prefill_admission import (
    PAIRED_SHORT_PROFILE,
    SHORT_PROFILE,
    PAIRED_UNCHANGED_LOCATION_FINGERPRINTS,
    short_acquisition,
    short_graph_identity,
)

ROOT = Path(__file__).resolve().parents[3]


def fixture():
    return f"""HloModule example, num_partitions=32

FileNames
1 "{DSA}"
2 "{WORKER}"
3 "other.py"

FunctionNames
1 "merge_topk_candidates_with_scores"
2 "main"
3 "<module>"

FileLocations
1 {{file_name_id=1 function_name_id=1 line=42 end_line=42 column=3 end_column=3}}
2 {{file_name_id=2 function_name_id=2 line=100 end_line=100 column=8 end_column=8}}
3 {{file_name_id=2 function_name_id=3 line=200 end_line=200 column=4 end_column=4}}
4 {{file_name_id=3 function_name_id=1 line=50 end_line=50 column=1 end_column=1}}

StackFrames
1 {{file_location_id=1 parent_frame_id=2}}
2 {{file_location_id=2 parent_frame_id=3}}
3 {{file_location_id=3}}

ENTRY main {{
  %a = f32[1] parameter(0)
  ROOT %b = f32[1] add(%a,%a), metadata={{op_name="dsa/add" stack_frame_id=1}}, backend_config={{"padding":0}}
}}
"""


def test_dsa_coordinate_equivalence_is_paired_only():
    raw = fixture()
    changed = raw.replace("line=42", "line=99").replace("column=3", "column=12")
    left = paired_worker_dsa_location_identity(raw)
    right = paired_worker_dsa_location_identity(changed)
    assert (
        left["paired_location_equivalence_sha256"]
        == right["paired_location_equivalence_sha256"]
    )
    assert left["raw_optimized_hlo_sha256"] != right["raw_optimized_hlo_sha256"]
    assert left["dsa_locations"] != right["dsa_locations"]
    assert (
        worker_location_identity(raw)["worker_location_equivalence_sha256"]
        != worker_location_identity(changed)["worker_location_equivalence_sha256"]
    )


@pytest.mark.parametrize(
    "before,after",
    [
        ("line=50", "line=51"),
        (DSA, DSA + ".other"),
        ('"merge_topk_candidates_with_scores"', '"different_function"'),
        ("file_name_id=1 function_name_id=1", "file_name_id=3 function_name_id=1"),
        ("function_name_id=1 line=42", "function_name_id=2 line=42"),
        ("1 {file_name_id=1", "5 {file_name_id=1"),
        ("parent_frame_id=2", "parent_frame_id=3"),
        ("stack_frame_id=1", "stack_frame_id=2"),
        ('op_name="dsa/add"', 'op_name="dsa/subtract"'),
        ("f32[1] add", "f32[1] multiply"),
        ("add(%a,%a)", "add(%a,%b)"),
        ('"padding":0', '"padding":1'),
    ],
)
def test_all_noncoordinate_mutations_stay_bound(before, after):
    raw = fixture()
    assert before in raw
    expected = paired_worker_dsa_location_identity(raw)[
        "paired_location_equivalence_sha256"
    ]
    try:
        actual = paired_worker_dsa_location_identity(raw.replace(before, after))
    except ValueError:
        return
    assert actual["paired_location_equivalence_sha256"] != expected


@pytest.mark.parametrize("graph", sorted(PAIRED_UNCHANGED_LOCATION_FINGERPRINTS))
def test_original_five_graphs_and_only_coordinate_shift(graph):
    receipt = short_acquisition(ROOT)
    folder = Path("/home/gianl/glm-run") / receipt["tag"] / "hlo"
    raw = (folder / (graph + ".optimized_hlo.txt")).read_bytes()
    pins = receipt["graphs"][graph]
    assert sha256(raw).hexdigest() == pins["optimized_hlo_sha256"]
    stable = (folder / (graph + ".stablehlo.mlir")).read_text()
    text = raw.decode()
    kwargs = dict(
        graph=graph,
        profile=PAIRED_SHORT_PROFILE,
        repo=ROOT,
        expected_stable=pins["stablehlo_sha256"],
        expected_optimized=pins["optimized_hlo_sha256"],
    )
    result = short_graph_identity(stable, text, **kwargs)
    expected = PAIRED_UNCHANGED_LOCATION_FINGERPRINTS[graph]
    assert result["paired_location_equivalence_sha256"] == expected
    assert len(result["dsa_locations"]) == (
        41 if graph in ("decode", "observer") else 0
    )
    entries = result["dsa_locations"] or result["worker_locations"]
    location_id = entries[0]["location_id"]
    changed, count = re.subn(
        rf"(?m)^({location_id} \{{file_name_id=\d+ function_name_id=\d+ line=)\d+",
        r"\g<1>9999",
        text,
    )
    assert count == 1
    shifted = short_graph_identity(stable, changed, **kwargs)
    assert shifted["paired_location_equivalence_sha256"] == expected
    assert shifted["raw_optimized_hlo_sha256"] != result["raw_optimized_hlo_sha256"]
    if result["dsa_locations"]:
        with pytest.raises(ValueError, match="debug coordinates"):
            short_graph_identity(
                stable, changed, **{**kwargs, "profile": SHORT_PROFILE}
            )
    with pytest.raises(ValueError, match="StableHLO"):
        short_graph_identity(stable + " ", text, **kwargs)


@pytest.mark.parametrize("source", [WS32, MOE])
def test_optional_hook_coordinates_are_paired_only(source):
    raw = fixture().replace(DSA, source)
    changed = raw.replace("line=42", "line=99")
    left = paired_worker_dsa_location_identity(raw)
    right = paired_worker_dsa_location_identity(changed)
    assert (
        left["paired_location_equivalence_sha256"]
        == right["paired_location_equivalence_sha256"]
    )
    assert left["raw_optimized_hlo_sha256"] != right["raw_optimized_hlo_sha256"]
    assert (
        worker_location_identity(raw)["worker_location_equivalence_sha256"]
        != worker_location_identity(changed)["worker_location_equivalence_sha256"]
    )
    assert left["schema_version"] == "ws32_paired_reviewed_source_debug_coordinates_v2"


def test_actual_refused_observer_has_only_reviewed_coordinate_changes():
    folder = Path(
        "/home/gianl/glm-run/greenfield_ws32_short_decoder_2k_numerical_c17_hrope_bp1_ps1_20260908T225358777672108Z/hlo"
    )
    stable = (folder / "observer.stablehlo.mlir").read_text()
    text = (folder / "observer.optimized_hlo.txt").read_text()
    assert (
        sha256(text.encode()).hexdigest()
        == "8aa5ad2bea90aa3eba9b127ae5a081bcc4aaadfa094461b892e61357873f7406"
    )
    pins = short_acquisition(ROOT)["graphs"]["observer"]
    kwargs = dict(
        graph="observer",
        profile=PAIRED_SHORT_PROFILE,
        repo=ROOT,
        expected_stable=pins["stablehlo_sha256"],
        expected_optimized=pins["optimized_hlo_sha256"],
    )
    result = short_graph_identity(stable, text, **kwargs)
    assert result["ws32_locations"] and result["moe_locations"]
    assert (
        result["paired_location_equivalence_sha256"]
        == PAIRED_UNCHANGED_LOCATION_FINGERPRINTS["observer"]
    )
    # Changing even an unused location in an unrelated model file still refuses.
    changed, count = re.subn(
        r"(?m)^(\d+ \{file_name_id=1 function_name_id=\d+ line=)\d+",
        r"\g<1>9999",
        text,
        count=1,
    )
    assert count == 1
    with pytest.raises(ValueError, match="reviewed source coordinates"):
        short_graph_identity(stable, changed, **kwargs)
    # A changed source filename cannot inherit another file's coordinate scope.
    with pytest.raises(ValueError, match="reviewed source coordinates"):
        short_graph_identity(stable, text.replace(MOE, MOE + ".other"), **kwargs)
