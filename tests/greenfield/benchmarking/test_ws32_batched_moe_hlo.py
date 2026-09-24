"""Exact75-layer coverage and adversarial SSA mutations, without TPU/JAX arrays."""

from pathlib import Path
from hashlib import sha256
import json

import pytest

from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import (
    PrefillHloIndex,
    check_batched_moe_route_sums,
)
from glm_tpu.greenfield.benchmarking.ws32_pallas_one_layer import (
    _live_instruction_closure,
)
from glm_tpu.optimized.hlo_contract import parse_hlo_module


GROUPS = (
    "{"
    + ",".join("{" + ",".join(str(i) for i in range(f, 32, 4)) + "}" for f in range(4))
    + "}"
)


def fixture(rows=17):
    text = [
        "HloModule batched, num_partitions=32",
        """%add {
%p0 = f32[] parameter(0)
%p1 = f32[] parameter(1)
ROOT %out = f32[] add(%p0, %p1)
}""",
    ]
    for layer in range(3, 78):
        scope = (
            f"greenfield_ws32_batched_prefill/layer_{layer}/greenfield_ws32_prefill_moe"
        )
        text.append(
            f"""%fusion{layer} {{
%routes = f32[{rows},8,1536] parameter(0)
%zero = f32[] constant(0)
ROOT %sum = f32[{rows},1536] reduce(%routes, %zero), dimensions={{1}}, to_apply=%add, metadata={{op_name="{scope}/fp32_route_sum/reduce_sum"}}
}}"""
        )
    text.append("ENTRY %main {")
    for layer in range(3, 78):
        scope = (
            f"greenfield_ws32_batched_prefill/layer_{layer}/greenfield_ws32_prefill_moe"
        )
        text.append(
            f"""%input{layer} = f32[{rows},8,1536] parameter({layer-3})
%sum{layer} = f32[{rows},1536] fusion(%input{layer}), calls=%fusion{layer}
%expert{layer} = bf16[{rows},1536] all-reduce(%sum{layer}), replica_groups={GROUPS}, use_global_device_ids=true, to_apply=%add, metadata={{op_name="{scope}/expert_reduce/psum"}}"""
        )
    text.append(
        "ROOT %out = ("
        + ", ".join([f"bf16[{rows},1536]"] * 75)
        + ") tuple("
        + ", ".join(f"%expert{l}" for l in range(3, 78))
        + ")\n}"
    )
    return "\n".join(text)


def check(text, rows=17):
    module = parse_hlo_module(text)
    return check_batched_moe_route_sums(
        PrefillHloIndex(module),
        block_rows=rows,
        live_instructions=_live_instruction_closure(module.instructions),
    )


@pytest.mark.parametrize("rows", [1, 11, 17, 32])
def test_exact_scope_coverage_and_per_layer_path(rows):
    result = check(fixture(rows), rows)
    assert result["passed"], result
    assert [v["layer"] for v in result["layers"]] == list(range(3, 78))
    assert all(
        [p[2] for p in v["path"]] == ["fusion", "reduce"] for v in result["layers"]
    )


@pytest.mark.parametrize(
    "old,new",
    [
        ("layer_77/", "layer_3/"),
        (
            "layer_3/greenfield_ws32_prefill_moe/fp32_route_sum",
            "layer_4/greenfield_ws32_prefill_moe/fp32_route_sum",
        ),
        ("dimensions={1}", "dimensions={0}"),
        ("constant(0)", "constant(1)"),
        ("constant(0)", "constant(-0)"),
        ("f32[17,8,1536]", "f32[8,17,1536]"),
        ("ROOT %out = f32[] add", "ROOT %out = f32[] multiply"),
        ("use_global_device_ids=true", "use_global_device_ids=false"),
        ("all-reduce(%sum77)", "all-reduce(%sum3)"),
        ("%expert77 = bf16[17,1536]", "%expert77 = f32[17,1536]"),
        ("%expert77 = bf16[17,1536]", "%expert77 = bf16[11,1536]"),
        ("f32[] add(%p0, %p1)", "f32[] add(%p0, %p0)"),
        ("parameter(1)\nROOT %out", "parameter(0)\nROOT %out"),
        (
            "dimensions={1}",
            "float_type_correction_info={original_type=BF16}, dimensions={1}",
        ),
    ],
)
def test_refuses_one_bad_boundary_or_shared_reducer(old, new):
    original = fixture()
    assert old in original
    result = check(original.replace(old, new, 1))
    assert not result["passed"], (old, new, result)


def test_dead_combine_and_missing_layer_are_not_coverage():
    text = fixture().replace(", %expert77)\n}", ", %expert76)\n}")
    assert not check(text)["passed"]
    assert not check(fixture().replace("layer_77/", "unscoped/"))["passed"]


def test_wrong_exact_physical_groups_refused():
    altered = GROUPS.replace("0,4,8,12,16,20,24,28", "0,4,8,12,16,20,24,29")
    assert not check(fixture().replace(GROUPS, altered, 1))["passed"]


def test_quoted_binding_is_not_a_callee():
    text = fixture().replace("calls=%fusion77", 'metadata={op_name="calls=%fusion77"}')
    with pytest.raises(ValueError, match="omits its callee"):
        check(text)
    module = parse_hlo_module(text)
    assert not check_batched_moe_route_sums(
        PrefillHloIndex(module), block_rows=17, live_instructions=module.instructions
    )["passed"]


def test_zero_constant_cannot_be_spoofed_in_metadata():
    text = fixture().replace(
        "constant(0)", 'constant(1), metadata={op_name="constant(0)"}', 1
    )
    assert not check(text)["passed"]


def test_forwarding_parameter_and_round_trip_refusal():
    extra = """%forward {
%p = f32[17,1536] parameter(0)
ROOT %c = f32[17,1536] copy(%p)
}
"""
    text = fixture().replace("ENTRY %main {", extra + "ENTRY %main {")
    text = text.replace(
        "%expert77 =",
        "%forwarded = f32[17,1536] fusion(%sum77), calls=%forward\n%expert77 =",
    )
    text = text.replace("all-reduce(%sum77)", "all-reduce(%forwarded)")
    assert check(text)["passed"]
    text = text.replace(
        "ROOT %c = f32[17,1536] copy(%p)",
        "%rounded = bf16[17,1536] convert(%p)\nROOT %c = f32[17,1536] convert(%rounded)",
    )
    assert not check(text)["passed"]


@pytest.mark.parametrize("rows", [0, 33, True, 1.0])
def test_invalid_row_count(rows):
    with pytest.raises(ValueError, match="rows"):
        check(fixture(), rows)


@pytest.mark.parametrize("graph,rows", [("prefill_chunk", 17), ("prefill_tail", 11)])
def test_original_acquired_graph(graph, rows):
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
        pytest.skip("original acquired HLO is not available locally")
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == receipt["graphs"][graph]["optimized_hlo_sha256"]
    result = check(raw.decode(), rows)
    assert result["passed"], result
