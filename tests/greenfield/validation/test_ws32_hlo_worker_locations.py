"""Raw evidence is retained; only the worker's debug coordinates may differ."""

from pathlib import Path
from hashlib import sha256
import re

import pytest

from glm_tpu.greenfield.validation.ws32_hlo_worker_locations import (
    WORKER,
    worker_location_identity,
)
from glm_tpu.greenfield.validation.ws32_prefill_admission import (
    SHORT_PROFILE,
    WORKER_LOCATION_FINGERPRINTS,
    short_acquisition,
    short_graph_identity,
)

ROOT = Path(__file__).resolve().parents[3]


def fixture():
    return f"""HloModule module, num_partitions=32

FileNames
1 "model.py"
2 "{WORKER}"

FunctionNames
1 "model"
2 "main"
3 "<module>"

FileLocations
1 {{file_name_id=1 function_name_id=1 line=42 end_line=42 column=3 end_column=3}}
2 {{file_name_id=2 function_name_id=2 line=100 end_line=100 column=8 end_column=8}}
3 {{file_name_id=2 function_name_id=3 line=200 end_line=200 column=4 end_column=4}}

StackFrames
1 {{file_location_id=1 parent_frame_id=2}}
2 {{file_location_id=2 parent_frame_id=3}}
3 {{file_location_id=3}}

ENTRY main {{
  %a = f32[1] parameter(0)
  ROOT %b = f32[1] add(%a,%a), metadata={{op_name="model/add" stack_frame_id=1}}, backend_config={{"padding":0}}
}}
"""


def test_only_worker_coordinates_are_equivalent_and_raw_digest_changes():
    text = fixture()
    changed = text.replace("line=100", "line=800").replace("column=8", "column=12")
    left, right = worker_location_identity(text), worker_location_identity(changed)
    assert (
        left["worker_location_equivalence_sha256"]
        == right["worker_location_equivalence_sha256"]
    )
    assert left["raw_optimized_hlo_sha256"] != right["raw_optimized_hlo_sha256"]
    assert left["worker_locations"] != right["worker_locations"]


@pytest.mark.parametrize(
    "before,after",
    [
        ("line=42", "line=43"),
        ("f32[1] add", "f32[1] multiply"),
        ("add(%a,%a)", "add(%a,%b)"),
        ("f32[1] add", "f32[1]{0} add"),
        ('"padding":0', '"padding":1'),
        ("stack_frame_id=1", "stack_frame_id=2"),
        ("parent_frame_id=2", "parent_frame_id=3"),
        ('op_name="model/add"', 'op_name="other/add"'),
    ],
)
def test_executable_and_other_debug_changes_remain_bound(before, after):
    text = fixture()
    assert before in text
    assert (
        worker_location_identity(text)["worker_location_equivalence_sha256"]
        != worker_location_identity(text.replace(before, after))[
            "worker_location_equivalence_sha256"
        ]
    )


@pytest.mark.parametrize(
    "before,after",
    [
        (WORKER, "other_worker.py"),
        ('2 "main"', '2 "wrapped_main"'),
        ("file_name_id=2 function_name_id=3", "file_name_id=1 function_name_id=3"),
    ],
)
def test_unacquired_debug_structure_refuses(before, after):
    with pytest.raises(ValueError):
        worker_location_identity(fixture().replace(before, after))


@pytest.mark.parametrize("graph", sorted(WORKER_LOCATION_FINGERPRINTS))
def test_original_and_worker_shifted_graph_match_registered_identity(graph):
    receipt = short_acquisition(ROOT)
    folder = Path("/home/gianl/glm-run") / receipt["tag"] / "hlo"
    path = folder / (graph + ".optimized_hlo.txt")
    if not path.exists():
        pytest.skip("original HLO unavailable")
    raw = path.read_bytes()
    pins = receipt["graphs"][graph]
    assert sha256(raw).hexdigest() == pins["optimized_hlo_sha256"]
    stable = (folder / (graph + ".stablehlo.mlir")).read_text()
    optimized = raw.decode()
    original = short_graph_identity(
        stable,
        optimized,
        graph=graph,
        profile=SHORT_PROFILE,
        repo=ROOT,
        expected_stable=pins["stablehlo_sha256"],
        expected_optimized=pins["optimized_hlo_sha256"],
    )
    location_id = original["worker_locations"][0]["location_id"]
    pattern = rf"(?m)^({location_id} \{{file_name_id=\d+ function_name_id=\d+ line=)\d+"
    changed, count = re.subn(pattern, lambda m: m[1] + "9999", optimized)
    assert count == 1
    shifted = short_graph_identity(
        stable,
        changed,
        graph=graph,
        profile=SHORT_PROFILE,
        repo=ROOT,
        expected_stable=pins["stablehlo_sha256"],
        expected_optimized=pins["optimized_hlo_sha256"],
    )
    assert shifted["raw_optimized_hlo_sha256"] != original["raw_optimized_hlo_sha256"]
    assert (
        shifted["worker_location_equivalence_sha256"]
        == original["worker_location_equivalence_sha256"]
    )
    with pytest.raises(ValueError, match="StableHLO bytes"):
        short_graph_identity(
            stable + " ",
            optimized,
            graph=graph,
            profile=SHORT_PROFILE,
            repo=ROOT,
            expected_stable=pins["stablehlo_sha256"],
            expected_optimized=pins["optimized_hlo_sha256"],
        )
