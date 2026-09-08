"""Host adapter/record tests: no backend execution or performance claim."""

from dataclasses import replace
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.runtime.ws32_batched_prefill import (
    Ws32BatchedPrefillResult,
    Ws32BatchedPrefillState,
)
from glm_tpu.greenfield.runtime.ws32_decoder import (
    Ws32DecoderConfig,
    Ws32DecoderState,
    ws32_decoder_weight_names,
)
from glm_tpu.greenfield.types import ModelGeometry
from scripts.greenfield import ws32_batched_prefill_runner as adapter


def config():
    geometry = ModelGeometry.from_hf_config(
        json.loads(Path("configs/glm-5.2-fp8-config.json").read_text())
    )
    return Ws32DecoderConfig(
        geometry,
        8192,
        exact_dsa=True,
        strategy_nd_dense=True,
        host_main_rope_table=True,
    )


def test_raw_view_is_same_loaded_objects_not_decode_overlay():
    cfg = config()
    raw = replace(cfg, exact_dsa=False, strategy_nd_dense=False)
    raw_names = jax.tree.leaves(ws32_decoder_weight_names(raw))
    overlay_names = jax.tree.leaves(ws32_decoder_weight_names(cfg))
    arrays = {name: object() for name in set(raw_names) | set(overlay_names)}
    actual_config, weights = adapter.bind_raw_prefill_weights(arrays, cfg)
    assert actual_config == raw
    for name, leaf in zip(raw_names, jax.tree.leaves(weights), strict=True):
        assert leaf is arrays[name]
    assert set(overlay_names) - set(raw_names)
    missing = dict(arrays)
    del missing[raw_names[0]]
    with pytest.raises(ValueError, match="missing retained tensors"):
        adapter.bind_raw_prefill_weights(missing, cfg)
    for bad in (
        replace(cfg, exact_dsa=False),
        replace(cfg, host_main_rope_table=False),
    ):
        with pytest.raises(ValueError, match="exact decode repair weights"):
            adapter.bind_raw_prefill_weights(arrays, bad)


def test_completed_repair_arrays_reused_without_materialization():
    cfg = config()
    exact = tuple(
        SimpleNamespace(wk_weight=jax.ShapeDtypeStruct((128, 6144), jnp.float32))
        for _ in range(21)
    )
    actual = adapter.completed_repair_weights(exact, cfg)
    assert all(a is e.wk_weight for a, e in zip(actual, exact, strict=True))
    with pytest.raises(ValueError, match="cardinality"):
        adapter.completed_repair_weights(exact[:-1], cfg)
    bad = (
        *exact[:-1],
        SimpleNamespace(wk_weight=jax.ShapeDtypeStruct((128, 6144), jnp.bfloat16)),
    )
    with pytest.raises(ValueError, match="completed full FP32"):
        adapter.completed_repair_weights(bad, cfg)


@pytest.mark.parametrize(
    "prompt,rows,split",
    [
        (2034, 17, (119, 11)),
        (8155, 17, (479, 12)),
        (34, 17, (1, 17)),
        (11, 17, (0, 11)),
    ],
)
def test_exact_static_tail_plan(prompt, rows, split):
    plan = adapter.BatchedPrefillPlan(prompt, rows, 8192)
    assert plan.split == split
    assert plan.graph_rows == (("prefill_chunk", rows), ("prefill_tail", split[1]))
    assert plan.identity()["mode"] == adapter.PREFILL_MODE
    assert plan.identity()["donate_argnums"] == []


@pytest.mark.parametrize(
    "values",
    [
        (0, 17, 8192),
        (8192, 17, 8192),
        (33, 33, 8192),
        (33, True, 8192),
        (33, 17, True),
        (33, 17, 33),
    ],
)
def test_plan_rejects_invalid_geometry(values):
    with pytest.raises(ValueError):
        adapter.BatchedPrefillPlan(*values)


def test_pair_builds_new_programs_and_input_signature(monkeypatch):
    cfg = replace(config(), exact_dsa=False, strategy_nd_dense=False)
    plan = adapter.BatchedPrefillPlan(2034, 17, 8192)
    calls = []
    monkeypatch.setattr(
        adapter,
        "build_ws32_batched_prefill_program",
        lambda mesh, c, *, block_rows: calls.append((mesh, c, block_rows)) or object(),
    )
    programs = adapter.build_graph_pair("mesh", cfg, plan)
    assert set(programs) == set(adapter.GRAPHS)
    assert calls == [("mesh", cfg, 17), ("mesh", cfg, 11)]
    with pytest.raises(ValueError, match="capacity differs"):
        adapter.build_graph_pair(
            "mesh", cfg, adapter.BatchedPrefillPlan(2034, 17, 16384)
        )
    monkeypatch.setattr(adapter, "replicated", lambda mesh, v: np.array(v, copy=True))
    tokens = np.arange(11, dtype=np.int32)
    args = adapter.graph_inputs("mesh", tokens, "state", "weights", ("wk",), "rope")
    assert (
        len(args) == 6
        and args[1].shape == ()
        and args[1].dtype == np.int32
        and int(args[1]) == 11
    )
    assert args[2:] == ("state", "weights", ("wk",), "rope")
    np.testing.assert_array_equal(args[0], tokens)


def fake_workload(monkeypatch, *, failure=None, prompt=28):
    cfg = replace(config(), exact_dsa=False, strategy_nd_dense=False)
    plan = adapter.BatchedPrefillPlan(prompt, 17, 8192)
    calls = []
    progress = []

    def fresh(mesh, config, *, prompt_length):
        decoder = Ws32DecoderState(
            np.zeros((1,), np.float32),
            np.zeros((1,), np.float32),
            np.zeros((1, 1), np.int32),
            np.zeros((1,), np.int32),
            np.zeros((1, 1), np.float32),
            np.array([0], np.int32),
            np.array([[0]], np.int32),
            np.array([1], np.int32),
            np.array([True]),
        )
        return Ws32BatchedPrefillState(
            decoder, np.ones((1,), np.float32), np.int32(prompt_length), np.bool_(False)
        )

    monkeypatch.setattr(adapter, "make_ws32_batched_prefill_state", fresh)
    monkeypatch.setattr(adapter, "replicated", lambda mesh, v: np.array(v, copy=True))

    def step(name):
        def run(tokens, count, state, weights, wk, rope):
            assert (weights, wk, rope) == ("weights", ("wk",), "rope")
            assert tokens.size == int(count) and count.shape == ()
            prior = int(state.decoder.position[0])
            end = prior + tokens.size
            np.testing.assert_array_equal(tokens, np.arange(prior, end, dtype=np.int32))
            final = end == prompt
            calls.append((name, prior, end))
            decoder = state.decoder._replace(
                position=np.array([end], np.int32),
                context_lengths=np.array([end + 1], np.int32),
                index_cache_local=(
                    state.repaired_index_local
                    if final
                    else state.decoder.index_cache_local
                ),
            )
            new = state._replace(decoder=decoder, finished=np.bool_(final))
            token = np.array([123 if final else -1], np.int32)
            if failure == "health":
                new = new._replace(
                    decoder=decoder._replace(contract_valid=np.array([False]))
                )
            if failure == "frontier":
                new = new._replace(
                    decoder=decoder._replace(position=np.array([end + 1], np.int32))
                )
            if failure == "length":
                new = new._replace(
                    decoder=decoder._replace(context_lengths=np.array([end], np.int32))
                )
            if failure == "phase":
                new = new._replace(finished=np.bool_(not final))
            if failure == "prompt":
                new = new._replace(prompt_length=np.int32(prompt + 1))
            if failure == "early_token":
                token = np.array([123], np.int32)
            return Ws32BatchedPrefillResult(new, token)

        return run

    clock = [0.0]

    def now():
        clock[0] += 0.01
        return clock[0]

    monkeypatch.setattr(adapter, "perf_counter", now)
    args = (
        "mesh",
        cfg,
        plan,
        np.arange(prompt, dtype=np.int32),
        {name: step(name) for name in adapter.GRAPHS},
        "weights",
        ("wk",),
        "rope",
    )
    return args, calls, progress


@pytest.mark.parametrize("prompt", [11, 28, 34])
def test_host_executes_exact_tail_and_reports_non_ttft_scope(monkeypatch, prompt):
    args, calls, progress = fake_workload(monkeypatch, prompt=prompt)
    decoder, token, record = adapter.execute_graph_pair(
        *args, budget_seconds=10, progress=progress.append, fleet_all=bool
    )
    assert decoder.position.tolist() == [prompt] and token.tolist() == [123]
    full, tail = args[2].split
    assert calls == [("prefill_chunk", i * 17, (i + 1) * 17) for i in range(full)] + [
        ("prefill_tail", full * 17, prompt)
    ]
    assert len(progress) == full + 1 and progress[-1]["finished"] is True
    assert (
        record["ttft_measured"] is False and record["repaired_index_installed"] is True
    )
    adapter.validate_execution_record(record, args[2])


@pytest.mark.parametrize(
    "failure", ["health", "frontier", "length", "phase", "prompt", "early_token"]
)
def test_host_refuses_bad_chunk_before_dispatching_next(monkeypatch, failure):
    args, calls, progress = fake_workload(monkeypatch, failure=failure)
    with pytest.raises(RuntimeError, match="unhealthy/wrong frontier"):
        adapter.execute_graph_pair(
            *args, budget_seconds=10, progress=progress.append, fleet_all=bool
        )
    assert len(calls) == 1 and progress == []


@pytest.mark.parametrize("budget", [False, -1, float("inf"), float("nan")])
def test_invalid_budget_refused_before_execution(monkeypatch, budget):
    args, calls, progress = fake_workload(monkeypatch)
    with pytest.raises(ValueError, match="finite positive"):
        adapter.execute_graph_pair(
            *args, budget_seconds=budget, progress=progress.append, fleet_all=bool
        )
    assert calls == []


def test_projection_stops_before_second_block(monkeypatch):
    args, calls, progress = fake_workload(monkeypatch)
    with pytest.raises(RuntimeError, match="projected/request wall"):
        adapter.execute_graph_pair(
            *args, budget_seconds=0.001, progress=progress.append, fleet_all=bool
        )
    assert len(calls) == len(progress) == 1


@pytest.mark.parametrize("peer_refusal", ["health", "budget"])
def test_peer_refusal_stops_even_when_local_host_passes(monkeypatch, peer_refusal):
    args, calls, progress = fake_workload(monkeypatch)
    votes = []

    def fleet_all(local):
        votes.append(local)
        assert local is True
        return len(votes) < (1 if peer_refusal == "health" else 2)

    with pytest.raises(RuntimeError):
        adapter.execute_graph_pair(
            *args, budget_seconds=10, progress=progress.append, fleet_all=fleet_all
        )
    assert len(calls) == 1
    assert len(votes) == (1 if peer_refusal == "health" else 2)


def test_logging_failure_votes_refuse_before_next_dispatch(monkeypatch):
    args, calls, _ = fake_workload(monkeypatch)
    votes = []

    def progress(_):
        raise OSError("log unavailable")

    def fleet_all(local):
        votes.append(local)
        return local

    with pytest.raises(RuntimeError, match="progress logging") as error:
        adapter.execute_graph_pair(
            *args, budget_seconds=10, progress=progress, fleet_all=fleet_all
        )
    assert isinstance(error.value.__cause__, OSError)
    assert votes == [True, False] and len(calls) == 1


def test_consensus_runs_for_every_block_including_final(monkeypatch):
    args, calls, progress = fake_workload(monkeypatch, prompt=34)
    votes = []

    def fleet_all(local):
        votes.append(local)
        return local

    adapter.execute_graph_pair(
        *args, budget_seconds=10, progress=progress.append, fleet_all=fleet_all
    )
    assert votes == [True] * (2 * len(calls))


@pytest.mark.parametrize(
    "mutation",
    [
        "mode",
        "identity_bool",
        "promoted",
        "health",
        "frontier",
        "token",
        "ttft",
        "missing_wall",
        "nan",
        "negative",
        "over_budget",
        "under_total",
        "extra",
    ],
)
def test_record_refuses_forged_or_incomplete_claim(monkeypatch, mutation):
    args, _, _ = fake_workload(monkeypatch)
    _, _, record = adapter.execute_graph_pair(
        *args, budget_seconds=10, progress=lambda _: None, fleet_all=bool
    )
    record = deepcopy(record)
    if mutation == "mode":
        record["identity"]["mode"] = "serial"
    if mutation == "identity_bool":
        record["identity"]["raw_prefill_exact_aliases"] = 0
    if mutation == "promoted":
        record["repaired_index_installed"] = False
    if mutation == "health":
        record["finished_healthy"] = False
    if mutation == "frontier":
        record["final_frontier"] = 27
    if mutation == "token":
        record["first_token_ready"] = -1
    if mutation == "ttft":
        record["ttft_measured"] = True
    if mutation == "missing_wall":
        record["block_wall_seconds"].pop()
    if mutation == "nan":
        record["block_wall_seconds"][0] = float("nan")
    if mutation == "negative":
        record["cache_initialization_seconds"] = -1
    if mutation == "over_budget":
        record["projected_total_seconds_max"] = 11
    if mutation == "under_total":
        record["request_prefill_seconds"] = 0
    if mutation == "extra":
        record["claimed_win"] = True
    with pytest.raises(ValueError):
        adapter.validate_execution_record(record, args[2])
