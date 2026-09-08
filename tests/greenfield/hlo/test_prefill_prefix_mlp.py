"""Boundary-only discriminator: pinned full inputs, distinct routes, no promotion."""

from copy import deepcopy
import json

import numpy as np
import pytest

from scripts.greenfield import prefill_prefix_mlp_protocol as p
from scripts.greenfield import ws32_prefill_layer_campaign as campaign
from tests.greenfield.hlo.test_prefill_router_protocol import fixture_arrays, router_hlo
from tests.greenfield.hlo.test_prefill_materialized_reference import reference_hlo
from tests.greenfield.hlo.test_prefill_layer_campaign import valid_workers


def arrays_and_fixture(tmp_path, monkeypatch):
    original, ledger = fixture_arrays()
    arrays = {k: v.copy() for k, v in original.items() if k.startswith("input__")}
    fixture = dict(
        protocol=p.PROTOCOL, source_analysis_sha256=p.ANALYSIS_SHA, owners={}
    )
    for d in range(4):
        for n in p.router.FIELDS:
            arrays[f"prefix_{d}__{n}"] = original[f"reference_0__{n}"].copy()
        for n in ("router_weight", "correction_bias"):
            arrays[f"weights_{d}__{n}"] = original[f"weights_0__{n}"].copy()
        for n in ("router_input", "routes", "route_weights"):
            arrays[f"suffix_{d}__{n}"] = original[f"reference_scalar_0__{n}"].copy()
        # The suffix expectation must NOT be the fused-prefix route signature.
        arrays[f"suffix_{d}__routes"][4, 5:7] = arrays[f"suffix_{d}__routes"][4, 6:4:-1]
        arrays[f"suffix_{d}__output"] = np.zeros((17, 1536), np.uint16)
        prefix = p.router.read_capture(arrays, f"prefix_{d}", p.router.FIELDS)
        weights = p.router.read_capture(
            arrays, f"weights_{d}", ("router_weight", "correction_bias")
        )
        fixture["owners"][str(d)] = dict(
            prefix=p.fingerprints(prefix),
            weights=p.fingerprints(weights),
            prefix_routes=prefix["routes"].tolist(),
            suffix_routes=arrays[f"suffix_{d}__routes"].tolist(),
        )
    for d in range(4, 32):
        fixture["owners"][str(d)] = deepcopy(fixture["owners"][str(d % 4)])
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture))
    monkeypatch.setattr(p, "FIXTURE", path)
    return arrays, {d: d for d in range(4)}, {d: deepcopy(ledger[0]) for d in range(4)}


def test_full_reproduction_and_distinct_suffix_signature(tmp_path, monkeypatch):
    arrays, slots, ledger = arrays_and_fixture(tmp_path, monkeypatch)
    proof = p.verify_prefix(arrays, slots, ledger)
    assert len(proof["owners"]) == 4
    path = tmp_path / "boundary.npz"
    np.savez_compressed(path, **arrays)
    result = p.replay_file(path, slots, ledger)
    assert result["evidence_complete"] and not result["numerical_admission"]
    assert not result["performance_claim"]


@pytest.mark.parametrize("field", p.router.FIELDS)
def test_every_observed_prefix_field_is_pinned(tmp_path, monkeypatch, field):
    arrays, slots, ledger = arrays_and_fixture(tmp_path, monkeypatch)
    a = arrays[f"prefix_3__{field}"]
    a.flat[-1] = (not a.flat[-1]) if a.dtype == np.bool_ else a.flat[-1] + 1
    with pytest.raises(ValueError):
        p.verify_prefix(arrays, slots, ledger)


@pytest.mark.parametrize(
    "mutation", ("input", "output_nan", "routes", "extra", "weights", "owner")
)
def test_refuse_broken_suffix_or_input_binding(tmp_path, monkeypatch, mutation):
    arrays, slots, ledger = arrays_and_fixture(tmp_path, monkeypatch)
    if mutation == "input":
        arrays["suffix_0__router_input"][16, 100] += 1
    elif mutation == "output_nan":
        arrays["suffix_0__output"].view(p.router.BF16)[16, 100] = np.nan
    elif mutation == "routes":
        arrays["suffix_0__routes"] = arrays["prefix_0__routes"].copy()
    elif mutation == "extra":
        arrays["unregistered"] = np.zeros(1)
    elif mutation == "weights":
        arrays["weights_0__router_weight"][0, 0] += 1
    else:
        slots[1] = 0
    path = tmp_path / "boundary.npz"
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError):
        p.replay_file(path, slots, ledger)


def test_two_graph_hlo_contract_not_batch_or_full_layer():
    assert p.check_hlo(router_hlo("reference"), "candidate")["passed"]
    assert p.check_hlo(reference_hlo("reference"), "reference")["passed"]
    assert not p.check_hlo(router_hlo("candidate"), "candidate")["passed"]
    with pytest.raises(ValueError):
        p.check_hlo(router_hlo("reference"), "other")


def diagnostic_workers():
    records, pin, pins, ledger = valid_workers(3)
    for r in records:
        r.update(
            protocol=p.PROTOCOL,
            reference_scope=p.REFERENCE_SCOPE,
            diagnostic_only=True,
            admission_only=False,
        )
        r["programs"] = {
            n: deepcopy(next(iter(r["programs"].values()))) for n in p.PROGRAMS
        }
        owners = {str(s["device_id"]): {} for s in r["local_device_slots"]}
        r["cases"] = dict(
            boundary=dict(
                evidence_complete=True,
                input_sha256="fixed",
                replay=dict(
                    evidence_complete=True, numerical_admission=False, owners=owners
                ),
            )
        )
    return records, pin, pins, ledger


def test_distinct_fleet_classification():
    tag = f"greenfield_fp8_{p.KERNEL}_l3_test"
    assert campaign.layer_from_tag(tag) == 3 and p.is_prefix_mlp_tag(tag)
    assert not p.is_prefix_mlp_tag(tag.replace("l3", "l0"))
    files = campaign.evidence_files(3, diagnostic=True, prefix_mlp=True)
    assert "boundary.npz" in files and "empty.npz" not in files
    assert "router_batch.optimized_hlo.txt" not in files
    assert len([f for f in files if f.endswith("optimized_hlo.txt")]) == 2
    records, pin, pins, ledger = diagnostic_workers()
    campaign.validate_workers(
        records,
        pin,
        layer=3,
        pins=pins,
        ledger=ledger,
        diagnostic=True,
        prefix_mlp=True,
    )
    with pytest.raises((ValueError, KeyError)):
        campaign.validate_workers(
            records, pin, layer=3, pins=pins, ledger=ledger, diagnostic=True
        )
    with pytest.raises(ValueError):
        campaign.evidence_files(3, prefix_mlp=True)
    with pytest.raises(ValueError):
        campaign.evidence_files(3, diagnostic=True, prefix_mlp=True, materialized=True)


@pytest.mark.parametrize(
    "mutation", ("scope", "protocol", "admission", "program", "case", "memory")
)
def test_refuse_misclassified_fleet(mutation):
    records, pin, pins, ledger = diagnostic_workers()
    r = records[0]
    if mutation == "scope":
        r["reference_scope"] = "RAW_SCALAR_NOT_PROMOTED_DECODER_OR_LEGACY"
    elif mutation == "protocol":
        r["protocol"] = p.router.PROTOCOL
    elif mutation == "admission":
        r["admission_only"] = True
    elif mutation == "program":
        r["programs"]["extra"] = deepcopy(r["programs"]["reference"])
    elif mutation == "case":
        r["cases"]["boundary"]["evidence_complete"] = False
    else:
        r["programs"]["reference"]["compiled_memory"]["temp_size_in_bytes"] = (
            3 * 1024**3
        )
    with pytest.raises(ValueError):
        campaign.validate_workers(
            records,
            pin,
            layer=3,
            pins=pins,
            ledger=ledger,
            diagnostic=True,
            prefix_mlp=True,
        )
