"""CPU-only tests for diagnostic classification; never admission promotion."""

from copy import deepcopy

import numpy as np
import pytest

from scripts.greenfield.analyze_prefill_window_failure import (
    bounded_rows,
    route_diagnosis,
    validate_mapping,
)


def record():
    order = np.arange(32).reshape(4, 8).T.tolist()
    return dict(
        physical_device_ids=order,
        jax_process_index=3,
        local_device_slots=[
            dict(device_slot=i, device_id=order[0][i]) for i in range(4)
        ],
        programs={n: {} for n in ("candidate", "control", "wk_decode", "wk_promote")},
    )


def test_physical_mapping():
    r = record()
    processes = set()
    assert validate_mapping(r, None, processes) == r["physical_device_ids"]
    assert processes == {3}


@pytest.mark.parametrize("mutation", ["slot", "process", "graph", "local", "mesh"])
def test_mapping_mutations(mutation):
    r = record()
    mesh = deepcopy(r["physical_device_ids"])
    processes = set()
    if mutation == "slot":
        r["local_device_slots"][0]["device_id"] = 31
    elif mutation == "process":
        processes.add(3)
    elif mutation == "graph":
        del r["programs"]["control"]
    elif mutation == "local":
        r["local_device_slots"].pop()
    else:
        mesh[0], mesh[1] = mesh[1], mesh[0]
    with pytest.raises(ValueError):
        validate_mapping(r, mesh, processes)


def test_route_alignment_does_not_hide_set_mismatch():
    routes = np.tile(np.arange(8), (3, 1))
    weights = np.tile(np.arange(1, 9, dtype=np.float32) / 36, (3, 1))
    a = dict(routes=routes.copy(), route_weights=weights.copy())
    b = deepcopy(a)
    b["routes"][0] = routes[0, ::-1]
    b["route_weights"][0] = weights[0, ::-1]
    b["routes"][1, 7] = 99
    result = route_diagnosis(a, b)
    assert result["ordered_mismatch_rows"] == [0, 1]
    assert result["set_mismatch_rows"] == [1]
    assert not result["slotwise_weights"]["aggregate"]["passed"]
    aligned = result["same_set_expert_aligned_weights"]
    assert aligned["aggregate"]["passed"]
    assert aligned["aggregate"]["shape"] == [2, 8]


def test_per_row_bound_cannot_hide_in_aggregate():
    a = np.zeros((128, 8), np.float32)
    b = a.copy()
    b[27] = 3e-4
    result = bounded_rows(a, b, routes=True)
    assert result["aggregate"]["passed"]
    assert result["failed_rows"] == [27]


def test_duplicate_routes_refused():
    a = dict(
        routes=np.zeros((1, 8), np.int32), route_weights=np.ones((1, 8), np.float32) / 8
    )
    with pytest.raises(ValueError):
        route_diagnosis(a, a)
