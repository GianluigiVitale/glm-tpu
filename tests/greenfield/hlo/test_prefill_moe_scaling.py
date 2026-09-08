"""CPU phase-baseline controls; no hardware performance claim or launch."""

import copy
import json
import re

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from scripts.greenfield import prefill_moe_scaling as scaling
from scripts.greenfield import probe_ws32_prefill_moe as probe
from scripts.greenfield.prefill_moe_precision_hlo import check_fp32_route_sum
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from tests.greenfield.hlo.test_prefill_moe_bounded import fp32_hlo


def inputs(case="normal"):
    h = np.linspace(-1, 1, 6144, dtype=np.float32)[None].astype(ml_dtypes.bfloat16)
    ids = np.arange(128, 136, dtype=np.int32)[None]
    weights = np.arange(1, 9, dtype=np.float32)[None] / 36
    return probe.case_rows(h, ids, weights, case, rows=128)


def rows_hlo(rows):
    # Change tensor dimensions only, never physical replica IDs or SSA names.
    return re.sub(
        r"\[([0-9,]+)\]",
        lambda m: "["
        + ",".join(
            str(rows) if n == "17" else str(rows * 8) if n == "136" else n
            for n in m[1].split(",")
        )
        + "]",
        fp32_hlo(),
    )


@pytest.mark.parametrize("case", probe.CASES)
def test_equal_rows_and_actual_metadata(case):
    values = inputs(case)
    weight = object()
    slices = scaling.split_equal_work((*values, weight))
    assert len(slices) == 8 and all(x[3] is weight for x in slices)
    for i in range(3):
        np.testing.assert_array_equal(np.concatenate([x[i] for x in slices]), values[i])
    assert np.unique(values[0], axis=0).shape[0] == 128
    counts = []
    for ids in [values[1], *[x[1] for x in slices]]:
        actual = np.asarray(jax.jit(scaling.device_active_tiles)(jnp.asarray(ids)))
        record = scaling.occupancy(ids, actual)
        assert record["route_rows"] == ids.size
        assert sum(record["group_sizes"]) == ids.size
        assert sum(record["owner_route_rows"]) == ids.size
        assert 0 < record["useful_lane_fraction"] <= 1
        assert json.loads(json.dumps(record)) == record
        counts.append(actual)
    if case == "normal":
        assert np.count_nonzero(counts[0]) == 8
        assert counts[0].sum() < np.sum(counts[1:])
    else:
        assert np.count_nonzero(counts[0]) == 1
        # Concentrated perfect row-tile occupancy need not improve with B128.
        np.testing.assert_array_equal(counts[0], np.sum(counts[1:], axis=0))


def test_straddling_not_ceil_count_and_invalid_metadata():
    ids = inputs()[1][:16]
    actual = np.asarray(scaling.device_active_tiles(jnp.asarray(ids)))
    record = scaling.occupancy(ids, actual)
    naive = sum((n + 7) // 8 for n in record["group_sizes"])
    assert actual.sum() >= naive
    for bad in (actual + 1, actual.astype(np.float32)):
        with pytest.raises(ValueError):
            scaling.occupancy(ids, bad)
    repeated = ids.copy()
    repeated[:, 1] = repeated[:, 0]
    assert np.all(np.asarray(scaling.device_active_tiles(jnp.asarray(repeated))) == -1)
    with pytest.raises(ValueError):
        scaling.occupancy(repeated, actual)


@pytest.mark.parametrize("rows", [16, 128])
def test_row_parameter_binds_actual_route_sum_geometry(rows):
    hlo = rows_hlo(rows)
    assert check_fp32_route_sum(parse_hlo_module(hlo), rows=rows)["passed"]
    assert not check_fp32_route_sum(parse_hlo_module(hlo))["passed"]
    assert not check_fp32_route_sum(
        parse_hlo_module(hlo.replace("ROOT %sum = f32", "ROOT %sum = bf16")), rows=rows
    )["passed"]
    with pytest.raises(ValueError):
        check_fp32_route_sum(parse_hlo_module(hlo), rows=256)


def test_per_row_and_cross_geometry_bounded_comparisons():
    a = np.ones((128, 1536), dtype=ml_dtypes.bfloat16)
    assert scaling.compare_equal_work(a, a.copy(), a.copy(), a[:1])["passed"]
    bad = a.copy()
    bad[93] = 2
    assert not scaling.compare_equal_work(a, bad, a.copy(), a[:1])["passed"]
    assert not scaling.compare_equal_work(a, a.copy(), a.copy(), a[:1] * 2)["passed"]
    bad[93] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        scaling.compare_equal_work(a, bad, a, a[:1])


def measured(small=False, peer_refuse=False):
    values = inputs()
    calls = scaling.split_equal_work(values) if small else (values,)
    events = []
    tick = [0.0]

    def clock():
        tick[0] += 0.001
        return tick[0]

    def program(*args):
        events.append(("dispatch", args[0].shape[0]))
        return "result"

    def complete(result):
        assert result == "result"
        events.append(("complete",))

    result = scaling.measure_completed_calls(
        program,
        calls,
        complete=complete,
        consensus=lambda ok: ok and not peer_refuse,
        deadline=120,
        clock=clock,
    )
    return result, events


@pytest.mark.parametrize("small", [False, True])
def test_completed_measurement_and_fleet_straggler(small):
    record, events = measured(small)
    n = 8 if small else 1
    assert len(events) == 60 * 2 * n
    assert all(
        events[i][0] == "dispatch" and events[i + 1][0] == "complete"
        for i in range(0, len(events), 2)
    )
    assert len(record["samples_ms"]) == 50
    records = [copy.deepcopy(record) for _ in range(8)]
    records[-1]["samples_ms"][0] = 10.0
    result = scaling.fleet_timing(records)
    assert result["fleet_straggler_samples_ms"][0] == 10.0
    assert result["live_rows"] == 128 and result["calls_per_sample"] == n
    assert result["p50_ms_per_live_row"] == result["p50_ms_per_128_rows"] / 128
    for key, value in (
        ("warmup", 9),
        ("calls_per_sample", 2),
        ("samples_ms", [1.0]),
        ("rows_per_call", 17),
    ):
        bad = copy.deepcopy(records)
        bad[1][key] = value
        with pytest.raises(ValueError):
            scaling.fleet_timing(bad)


def test_peer_budget_refusal_before_dispatch():
    with pytest.raises(TimeoutError):
        measured(peer_refuse=True)


def test_deadline_after_completed_call_refuses():
    times = iter([0.0, 0.0, 0.1, 121.0])
    calls = []
    with pytest.raises(TimeoutError):
        scaling.measure_completed_calls(
            lambda *v: calls.append("dispatch"),
            (inputs(),),
            complete=lambda _: None,
            consensus=bool,
            deadline=120.0,
            clock=lambda: next(times),
        )
    assert calls == ["dispatch"]


@pytest.mark.parametrize("peer_only", [False, True])
def test_middle_call_failure_stops_before_next_dispatch(peer_only):
    calls = []
    votes = []
    tick = [0.0]

    def clock():
        tick[0] += 0.001
        return tick[0]

    def program(*values):
        calls.append(len(calls))
        if len(calls) == 3 and not peer_only:
            raise ValueError("injected local failure")
        return None

    def consensus(ok):
        votes.append(ok)
        # entry, then success/budget for call1 and call2, then call3 success
        return ok and not (peer_only and len(votes) == 6)

    with pytest.raises(RuntimeError, match="completed-call fleet failure"):
        scaling.measure_completed_calls(
            program,
            scaling.split_equal_work(inputs()),
            complete=lambda _: None,
            consensus=consensus,
            deadline=120.0,
            clock=clock,
        )
    assert len(calls) == 3 and len(votes) == 6
    assert votes[-1] is peer_only
