"""Fixed graph role/byte admission and simultaneous memory, no TPU execution."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.greenfield import prefill_window_admission as admission

ORIGINAL = Path(
    "/home/gianl/glm-run/greenfield_fp8_ws32_prefill_layer_window_acquisition_l6_"
    "20260908T115105929011946Z/fleet/rank0"
)


@pytest.fixture(scope="module")
def originals():
    if not ORIGINAL.is_dir():
        pytest.skip("requires original DB590 locally materialized evidence")
    return {
        name: (
            (ORIGINAL / f"{name}.stablehlo.mlir").read_text(),
            (ORIGINAL / f"{name}.optimized_hlo.txt").read_text(),
            pin["compiled_memory"],
        )
        for name, pin in admission.registered_programs().items()
    }


@pytest.mark.parametrize("name", admission.PROGRAMS)
def test_original_graphs_exact_replay(name, originals):
    report = admission.inspect_program(name, *originals[name])
    assert report["passed"]
    assert not report["numerical_execution_authorized"]
    if name in ("candidate", "control"):
        assert report["fp32_route_sum"]["passed"]
    # Publication is JSON, not Python tuple identity.
    assert json.loads(json.dumps(report)) == report
    admission.validate_program_report(
        json.loads(json.dumps(report)), name, *originals[name]
    )


@pytest.mark.parametrize("mutation", ["value", "type", "group", "tuple"])
def test_nested_published_payload_mutation_refuses(mutation, originals):
    # WK decode is small but traverses the same full-report consumer.
    name = "wk_decode"
    report = admission.inspect_program(name, *originals[name])
    payload = report["collective_payloads"][0]
    if mutation == "value":
        payload["inputs"][0][1][0] += 1
    elif mutation == "type":
        payload["inputs"][0][1][0] = float(payload["inputs"][0][1][0])
    elif mutation == "group":
        payload["groups"][0][0] = False
    else:
        payload["groups"][0] = tuple(payload["groups"][0])
    with pytest.raises(ValueError, match="report differs"):
        admission.validate_program_report(report, name, *originals[name])


@pytest.mark.parametrize("name", admission.PROGRAMS)
@pytest.mark.parametrize("mutation", ["stable", "hlo", "role", "memory", "bool"])
def test_raw_bytes_roles_and_compiler_allocations_refuse(name, mutation, originals):
    stable, hlo, memory = originals[name]
    memory = dict(memory)
    if mutation == "stable":
        stable += "\n"
    elif mutation == "hlo":
        # Even a metadata-only difference requires diagnosis, not normalization.
        hlo += "\n"
    elif mutation == "role":
        name = admission.PROGRAMS[(admission.PROGRAMS.index(name) + 1) % 4]
    elif mutation == "memory":
        memory["temp_size_in_bytes"] += 1
    else:
        memory["alias_size_in_bytes"] = False  # False == 0 must not pass.
    with pytest.raises(ValueError):
        admission.inspect_program(name, stable, hlo, memory)


def test_changed_receipt_refuses(tmp_path, monkeypatch):
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(admission.RECEIPT.read_bytes() + b"\n")
    monkeypatch.setattr(admission, "RECEIPT", receipt)
    with pytest.raises(ValueError, match="receipt content"):
        admission.registered_programs()


def memory_inputs(retained_bytes=0):
    buffers = [dict(bytes=340_000_000)]
    if retained_bytes:
        buffers.append(dict(bytes=retained_bytes))
    census = dict(
        schema_version="ws32_prefill_resident_buffers_v1",
        includes_all_live_arrays=True,
        devices=[
            dict(
                device_id=9,
                buffers=buffers,
                accounted_resident_bytes=340_000_000 + retained_bytes,
                memory_stats=dict(
                    bytes_in_use=340_000_000 + retained_bytes,
                    peak_bytes_in_use=350_000_000 + retained_bytes,
                    bytes_limit=33_014_398_976,
                ),
            )
        ],
    )
    analyses = {
        n: dict(p["compiled_memory"])
        for n, p in admission.registered_programs().items()
    }
    return census, analyses


@pytest.mark.parametrize("name", admission.PROGRAMS)
def test_budget_all_four_code_but_only_active_outputs_scratch(name):
    census, analyses = memory_inputs()
    budget = admission.memory_budget(census, analyses, active_graph=name)
    assert budget["required_reserve_bytes"] == 1 << 30
    assert budget["estimate_fits"]
    device = budget["devices"][0]
    assert device["resident_code_bytes"] == 67_150_848
    assert device["estimated_peak_bytes"] == (
        340_000_000
        + 67_150_848
        + analyses[name]["output_size_in_bytes"]
        + analyses[name]["temp_size_in_bytes"]
    )
    census, analyses = memory_inputs(50_000_000)
    retained = admission.memory_budget(census, analyses, active_graph=name)
    assert (
        retained["devices"][0]["estimated_peak_bytes"]
        == device["estimated_peak_bytes"] + 50_000_000
    )
    assert retained["numerical_admission"] is False


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "code", "alias", "bool", "peak", "retained", "unknown"],
)
def test_budget_refuses_drift_or_reports_insufficient_headroom(mutation):
    census, analyses = memory_inputs()
    name = "candidate"
    if mutation == "missing":
        analyses.pop("wk_promote")
    elif mutation == "extra":
        analyses["unregistered"] = deepcopy(analyses[name])
    elif mutation == "code":
        analyses["wk_decode"]["generated_code_size_in_bytes"] = 0
    elif mutation == "alias":
        analyses[name]["alias_size_in_bytes"] = 1
    elif mutation == "bool":
        analyses[name]["alias_size_in_bytes"] = False
    elif mutation == "unknown":
        name = "prefill_chunk"
    else:
        device = census["devices"][0]
        if mutation == "peak":
            device["memory_stats"]["peak_bytes_in_use"] = 33_000_000_000
        else:
            device["buffers"].append(dict(bytes=32_000_000_000))
            device["accounted_resident_bytes"] += 32_000_000_000
        assert not admission.memory_budget(census, analyses, active_graph=name)[
            "estimate_fits"
        ]
        return
    with pytest.raises(ValueError):
        admission.memory_budget(census, analyses, active_graph=name)


def test_tuple_order_and_group_multiplicity_are_part_of_contract():
    payloads = admission.expected_collectives("candidate")
    tuple_key = next(k for k in payloads if len(k[2]) == 16)
    changed = list(tuple_key)
    changed[2] = tuple(reversed(changed[2]))
    assert tuple(changed) not in payloads
    changed = list(tuple_key)
    changed[1] = changed[1][:-1]
    assert tuple(changed) not in payloads
