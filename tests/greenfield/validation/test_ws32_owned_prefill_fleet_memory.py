"""Compose owned budget replay with the existing all32-owner sealer helper.

DB615 supplies real compiler allocations, NOT runtime counters. Every census
and runtime peak below is a synthetic fixture, never TPU capacity evidence.
"""

from copy import deepcopy
import json

import pytest

from glm_tpu.greenfield.validation.ws32_prefill_fleet_memory import validate_batched_fleet_memory
from scripts.greenfield import ws32_owned_prefill_memory as owned
from tests.greenfield.validation.test_ws32_prefill_fleet_memory import ROOT, fixture as historical_fixture


def fixture(*, shared=True):
    args = historical_fixture()
    analysis = json.loads((ROOT / "docs/artifacts/prefill-capture-barrier-db615-sealed-20260912.json").read_text())["memory"]
    args["state_ownership_contract"] = owned.CONTRACT
    args["expected_prefill_analyses"] = {role: deepcopy(analysis) for role in owned.ROLES}
    for runner in args["records"]:
        census = runner["prefill_execution"]["memory_admission"]["census"]
        for device in census["devices"]:
            alias = analysis["alias_size_in_bytes"]
            device["buffers"] = [
                dict(bytes=alias, identity_mode="physical_pointer",
                     groups=["active_state", "active_inputs", "__all_live_arrays__"]),
                dict(bytes=device["accounted_resident_bytes"] - alias,
                     identity_mode="physical_pointer",
                     groups=["active_inputs", "nonstate_inputs", "__all_live_arrays__"]),
            ]
        admission = dict(schema_version=owned.SCHEMA, ownership_contract=owned.CONTRACT,
            donated_argument=2, donated_state_leaves=12,
            executable_roles={r: owned.ROLES[0] if shared else r for r in owned.ROLES},
            compiled_memory={r: deepcopy(analysis) for r in owned.ROLES[:1 if shared else 2]},
            census=census, required_reserve_bytes=args["required_reserve_bytes"])
        admission["budgets"] = owned.budgets(admission)
        runner["prefill_execution"]["memory_admission"] = admission
        runner["compiled_memory_analysis"] = deepcopy(args["expected_prefill_analyses"])
    return args


@pytest.mark.parametrize("shared", [True, False])
def test_compiler_allocations_join_all32_owners_without_inventing_runtime_fit(shared):
    args = fixture(shared=shared)
    report = validate_batched_fleet_memory(**args)
    assert report["schema_version"] == "ws32_owned_batched_fleet_memory_v1"
    assert report["state_ownership_contract"] == owned.CONTRACT
    assert report["owner_count"] == 32
    # Synthetic runtime counters remain separate from the real compiler estimate.
    assert report["final_lifetime_peak_bytes"] == 26_000_002_000
    for runner in args["records"]:
        for key in ("local_device_slots", "batched_prefill_memory", "batched_device_memory_after_execute"):
            runner[key].reverse()
    assert report == validate_batched_fleet_memory(**args)
    del args["state_ownership_contract"]
    with pytest.raises(ValueError):
        validate_batched_fleet_memory(**args)


@pytest.mark.parametrize("boundary", ["census", "prefill", "final"])
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "foreign", "process", "cpu", "reserve", "limit"])
def test_owned_mode_keeps_every_physical_owner_and_peak_check(boundary, mutation):
    args = fixture()
    runner = args["records"][0]
    rows = (runner["prefill_execution"]["memory_admission"]["census"]["devices"]
            if boundary == "census" else runner["batched_prefill_memory"
            if boundary == "prefill" else "batched_device_memory_after_execute"])
    item = rows[0]
    stats = item["memory_stats"] if boundary == "census" else item
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = deepcopy(item)
    elif mutation == "foreign":
        item["device_id"] = 31
    elif mutation == "process":
        item["process_index"] = 7
    elif mutation == "cpu":
        item["platform"] = "cpu"
    elif mutation == "reserve":
        stats["peak_bytes_in_use"] = args["expected_device_limit_bytes"] - args["required_reserve_bytes"] + 1
    else:
        stats["bytes_limit"] += 1
    with pytest.raises(ValueError):
        validate_batched_fleet_memory(**args)


@pytest.mark.parametrize("mutation", ["wrong_contract", "undeclared", "different_alias",
    "different_runner", "different_acquired", "extra_code", "role_swap", "shared_but_different",
    "slot", "host", "late_peak_drop", "record_reserve", "missing_all_live"])
def test_owned_mode_cannot_self_authorize_or_bypass_graph_and_lifetime_bindings(mutation):
    args = fixture()
    runner = args["records"][0]
    admission = runner["prefill_execution"]["memory_admission"]
    if mutation == "wrong_contract":
        args["state_ownership_contract"] = "unreviewed"
    elif mutation == "undeclared":
        args["state_ownership_contract"] = None
    elif mutation == "different_alias":
        admission["compiled_memory"]["prefill_chunk"]["alias_size_in_bytes"] -= 1
        admission["budgets"] = owned.budgets(admission)
    elif mutation == "different_runner":
        runner["compiled_memory_analysis"]["prefill_tail"]["temp_size_in_bytes"] += 1
    elif mutation == "different_acquired":
        args["expected_prefill_analyses"]["prefill_chunk"]["generated_code_size_in_bytes"] += 1
    elif mutation == "extra_code":
        admission["compiled_memory"]["decode"] = deepcopy(admission["compiled_memory"]["prefill_chunk"])
    elif mutation == "role_swap":
        admission["executable_roles"]["prefill_chunk"] = "prefill_tail"
    elif mutation == "shared_but_different":
        for r in args["records"]:
            r["compiled_memory_analysis"]["prefill_tail"]["temp_size_in_bytes"] += 1
        args["expected_prefill_analyses"]["prefill_tail"]["temp_size_in_bytes"] += 1
    elif mutation == "slot":
        runner["local_device_slots"][0]["device_slot"] = 31
    elif mutation == "host":
        runner["hostname"] = "different"
    elif mutation == "late_peak_drop":
        runner["batched_device_memory_after_execute"][0]["peak_bytes_in_use"] = 26_000_000_000
    elif mutation == "record_reserve":
        admission["required_reserve_bytes"] -= 1
        admission["budgets"] = owned.budgets(admission)
    else:
        admission["census"]["includes_all_live_arrays"] = False
    with pytest.raises(ValueError):
        validate_batched_fleet_memory(**args)
