"""Receipt-bound window compiler-memory pins, budgets and collective contract."""

from copy import deepcopy

import pytest

from scripts.greenfield import prefill_window_admission as admission


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
