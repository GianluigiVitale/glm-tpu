from __future__ import annotations

from copy import deepcopy

import ml_dtypes
import numpy as np
import pytest

from scripts.greenfield.prefill_layer_numerical import (
    CASES,
    ROWS,
    compare_layer_case,
    written_addresses,
)


def _case(*, layer=0, case="boundary", slot=0):
    offset, count = CASES[case]
    bf = ml_dtypes.bfloat16
    values = {
        "output": np.zeros((ROWS, 1536), bf),
        "residual": np.zeros((ROWS, 1536), bf),
        "kv": np.zeros((2, 64, 640), bf),
        "index": np.zeros((2, 64, 128), bf),
        "repair": np.zeros((2, 64, 128), bf),
        "positions": np.full((ROWS, 2048), -1, np.int32),
        "counts": np.zeros(ROWS, np.int32),
        "scores": np.full((ROWS, 2048), -np.inf, np.float32),
        "routes": np.full((ROWS, 8), -1, np.int32),
        "route_weights": np.zeros((ROWS, 8), np.float32),
        "health": np.ones(ROWS, np.bool_),
        "normalized": np.zeros((ROWS, 1536), bf),
    }
    values["output"][:count] = 0.01
    values["residual"][:count] = 0.02
    for row in range(count):
        length = offset + row + 1
        values["positions"][row, :length] = np.arange(length)
        values["counts"][row] = length
        values["scores"][row, :length] = 0
        if layer == 3:
            values["routes"][row] = np.arange(8)
            values["route_weights"][row] = 0.125
    initial = {
        name: values[name].copy()
        for name in ("kv", "index", "repair", "positions", "counts", "scores")
    }
    keys = np.full((ROWS, 128), 0.03, bf)
    for query, page, row in written_addresses(slot=slot, offset=offset, count=count):
        values["kv"][page, row, :576] = 0.01
        if layer == 0:
            values["index"][page, row] = 0.02
            values["repair"][page, row] = keys[query]
    args = dict(
        slot=slot,
        layer=layer,
        case=case,
        actual_m64_keys=keys if layer == 0 else None,
        reference_m64_keys=keys if layer == 0 else None,
    )
    return values, deepcopy(values), initial, args


@pytest.mark.parametrize("layer", [0, 3])
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("slot", [0, 7, 28, 31])
def test_complete_layer_contract_accepts_registered_identity(layer, case, slot):
    actual, reference, initial, args = _case(layer=layer, case=case, slot=slot)
    assert compare_layer_case(actual, reference, initial, **args)["passed"]


def test_host_address_mapping_crosses_stripe_and_reordered_page():
    assert written_addresses(slot=28, offset=505, count=17) == tuple(
        (i, 1, 57 + i) for i in range(7)
    )
    assert written_addresses(slot=0, offset=505, count=17) == tuple(
        (i, 0, i - 7) for i in range(7, 17)
    )
    assert written_addresses(slot=4, offset=505, count=17) == ()


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("health", "health failed"),
        ("causality", "causal coverage"),
        ("tie", "order/ties"),
        ("untouched", "untouched kv"),
        ("kv_padding", "structural padding"),
        ("repair", "own observed-input"),
        ("tail", "padded output"),
        ("nan", "nonfinite output"),
        ("wrong_dtype", "shape/dtype"),
        ("extra_field", "fields differ"),
    ],
)
def test_complete_layer_contract_refuses_corrupt_evidence(mutation, message):
    actual, reference, initial, args = _case(case="tail")
    if mutation == "health":
        actual["health"][0] = False
    elif mutation == "causality":
        actual["positions"][0, 0] = 600
    elif mutation == "tie":
        actual["positions"][0, :2] = [1, 0]
    elif mutation == "untouched":
        actual["kv"][1, 0, 0] = 0.01
    elif mutation == "kv_padding":
        actual["kv"][0, 0, 600] = 0.01
    elif mutation == "repair":
        actual["repair"][0, 0, 0] = 0.04
    elif mutation == "tail":
        actual["output"][-1, 0] = 0.01
    elif mutation == "nan":
        actual["output"][0, 0] = np.nan
    elif mutation == "wrong_dtype":
        actual["output"] = actual["output"].astype(np.float32)
    elif mutation == "extra_field":
        actual["claimed_success"] = np.array(True)
    with pytest.raises(ValueError, match=message):
        compare_layer_case(actual, reference, initial, **args)


def test_exact_route_ids_precede_moe_output_acceptance():
    actual, reference, initial, args = _case(layer=3)
    actual["routes"][0, 0] = 12
    with pytest.raises(ValueError, match="route IDs differ"):
        compare_layer_case(actual, reference, initial, **args)


def test_written_row_errors_not_diluted_by_whole_cache():
    actual, reference, initial, args = _case()
    # Only one scalar in one written row fails; huge unchanged tails cannot hide it.
    actual["kv"][0, 0, 0] = 0.5
    result = compare_layer_case(actual, reference, initial, **args)
    assert not result["passed"]
    assert not result["comparisons"]["kv_written_rows"]["rows"][0]["passed"]


def test_index_share_cannot_change_supplied_metadata():
    actual, reference, initial, args = _case(layer=3)
    actual["scores"][0, 0] = 1
    with pytest.raises(ValueError, match="IndexShare selection changed"):
        compare_layer_case(actual, reference, initial, **args)


def test_reference_repair_must_use_reference_normalization():
    actual, reference, initial, args = _case()
    args["reference_m64_keys"] = args["reference_m64_keys"].copy()
    args["reference_m64_keys"][7, 0] = 0.04
    with pytest.raises(ValueError, match="own observed-input"):
        compare_layer_case(actual, reference, initial, **args)
