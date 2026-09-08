"""Honest actual PREnorm observation and DB585-before-MLP integration, CPU only."""

from copy import deepcopy
from hashlib import sha256

import numpy as np
import pytest

from scripts.greenfield import prefill_observed_reference as ref
from scripts.greenfield import ws32_prefill_layer_campaign as campaign
from tests.greenfield.hlo.test_prefill_materialized_reference import (
    materialized_workers,
)
from tests.greenfield.hlo.test_prefill_prefix_mlp import arrays_and_fixture


def observed_workers():
    records, pin, pins, ledger = materialized_workers()
    for r in records:
        r.update(
            protocol=ref.PROTOCOL,
            reference_scope=ref.REFERENCE_SCOPE,
            boundary_prefix=dict(
                npz_sha256="f" * 64,
                replay=dict(
                    owners={str(s["device_id"]): {} for s in r["local_device_slots"]}
                ),
            ),
        )
    return records, pin, pins, ledger


def test_distinct_observed_mode():
    tag = f"greenfield_fp8_{ref.KERNEL}_l3_test"
    assert ref.is_observed_tag(tag) and campaign.layer_from_tag(tag) == 3
    assert not ref.is_observed_tag(tag.replace("l3", "l0"))
    files = campaign.evidence_files(3, materialized=True, observed=True)
    assert "boundary_prefix.npz" in files
    assert {f"{c}.npz" for c in campaign.CASES} <= set(files)
    records, pin, pins, ledger = observed_workers()
    campaign.validate_workers(
        records,
        pin,
        layer=3,
        pins=pins,
        ledger=ledger,
        materialized=True,
        observed=True,
    )
    with pytest.raises(ValueError):
        campaign.validate_workers(
            records, pin, layer=3, pins=pins, ledger=ledger, materialized=True
        )
    for kwargs in (
        {"observed": True},
        {"materialized": True, "observed": True, "diagnostic": True},
    ):
        with pytest.raises(ValueError):
            campaign.evidence_files(3, **kwargs)


@pytest.mark.parametrize(
    "mutation", ("prefix_missing", "prefix_owner", "prefix_hash", "scope")
)
def test_fleet_refuses_unbound_observation(mutation):
    records, pin, pins, ledger = observed_workers()
    r = records[0]
    if mutation == "prefix_missing":
        del r["boundary_prefix"]
    elif mutation == "prefix_owner":
        r["boundary_prefix"]["replay"]["owners"] = {}
    elif mutation == "prefix_hash":
        r["boundary_prefix"]["npz_sha256"] = "bad"
    else:
        r["reference_scope"] = ref.v2.REFERENCE_SCOPE
    with pytest.raises(ValueError):
        campaign.validate_workers(
            records,
            pin,
            layer=3,
            pins=pins,
            ledger=ledger,
            materialized=True,
            observed=True,
        )


@pytest.mark.parametrize(
    "mutation", (None, "fake_prenorm", "changed_input", "prefix_drift", "nan", "extra")
)
def test_original_boundary_binds_actual_norm_and_completed_input(
    tmp_path, monkeypatch, mutation
):
    arrays, slots, ledger = arrays_and_fixture(tmp_path, monkeypatch)
    prefix = {k: v for k, v in arrays.items() if not k.startswith("suffix_")}
    layer, capture = {}, {}
    for d in slots:
        norm = np.full((17, 1536), 2, dtype=ref.boundary.router.BF16).view(np.uint16)
        prefix[f"prefix_{d}__pre_attention_norm"] = norm
        layer[f"reference_{d}__normalized"] = norm.copy()
        capture[f"device_{d}"] = prefix[f"prefix_{d}__router_input"].copy()
    if mutation == "fake_prenorm":
        layer["reference_0__normalized"] = prefix["prefix_0__router_input"].copy()
    elif mutation == "changed_input":
        capture["device_0"][16, 0] += 1
    elif mutation == "prefix_drift":
        prefix["prefix_0__router_input"][16, 0] += 1
    elif mutation == "nan":
        prefix["prefix_0__pre_attention_norm"].view(ref.boundary.router.BF16)[
            0, 0
        ] = np.nan
    elif mutation == "extra":
        prefix["fabricated"] = np.zeros(1)
    path = tmp_path / "boundary_prefix.npz"
    np.savez_compressed(path, **prefix)
    np.savez_compressed(tmp_path / "boundary.npz", **layer)
    np.savez_compressed(tmp_path / "boundary.reference_input.npz", **capture)
    if mutation in ("prefix_drift", "nan", "extra"):
        with pytest.raises(ValueError):
            ref.verify_boundary_file(path, slots, ledger)
        return
    record = dict(
        boundary_prefix=dict(
            npz_sha256=sha256(path.read_bytes()).hexdigest(),
            replay=ref.verify_boundary_file(path, slots, ledger),
        )
    )
    if mutation is None:
        ref.validate_boundary_binding(tmp_path, record, slots, ledger)
    else:
        with pytest.raises(ValueError):
            ref.validate_boundary_binding(tmp_path, record, slots, ledger)


def test_assembly_keeps_actual_thirteenth_value_and_own_kv():
    values = tuple(np.full((1, 2), i) for i in range(20))
    prefix = tuple(np.full((1, 2), 100 + i) for i in range(13))
    mlp = tuple(np.full((1, 2), 200 + i) for i in range(3))
    result = ref.assemble_reference_result(values, prefix, mlp)
    assert result[11] is prefix[12] and result[2] is prefix[10]
    assert result[1] is prefix[9] and result[0] is mlp[0]
    with pytest.raises(ValueError):
        ref.assemble_reference_result(values, prefix[:12], mlp)
    with pytest.raises(ValueError):
        ref.scalar_observed_inputs(values, 0, prefix[:12])
