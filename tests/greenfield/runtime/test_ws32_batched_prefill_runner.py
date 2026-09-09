"""Host adapter/record tests: no backend execution or performance claim."""

from dataclasses import replace
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from functools import partial
import weakref

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
from glm_tpu.greenfield.validation.ws32_prefill_memory import (
    SCHEMA,
    budget_prefill_execution,
)


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
        lambda mesh, c, *, block_rows, **options: calls.append(
            (mesh, c, block_rows, options)
        )
        or object(),
    )
    programs = adapter.build_graph_pair("mesh", cfg, plan)
    assert set(programs) == set(adapter.GRAPHS)
    defaults = dict(
        paired_position_sort=False,
        mlp_window=False,
        rolled_prefix=False,
        expert_panels=False,
        sorted_local_merge=False,
        key_tile=4096,
    )
    assert calls == [("mesh", cfg, 17, defaults), ("mesh", cfg, 11, defaults)]
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


def test_explicit_window_plan_and_all_builder_options(monkeypatch):
    plan = adapter.BatchedPrefillPlan(2034, 128, 8192, mlp_window=True)
    assert plan.split == (15, 114)
    assert plan.identity()["mlp_window"] is True
    assert "mlp_window" not in adapter.BatchedPrefillPlan(2034, 17, 8192).identity()
    for rows, flag in ((128, False), (129, True), (128, 1)):
        with pytest.raises(ValueError):
            adapter.BatchedPrefillPlan(2034, rows, 8192, mlp_window=flag)
    calls = []
    monkeypatch.setattr(
        adapter,
        "build_ws32_batched_prefill_program",
        lambda mesh, c, **kw: calls.append(kw) or object(),
    )
    cfg = replace(config(), exact_dsa=False, strategy_nd_dense=False)
    adapter.build_graph_pair(
        "mesh",
        cfg,
        plan,
        paired_position_sort=True,
        rolled_prefix=True,
        expert_panels=True,
        sorted_local_merge=True,
        key_tile=512,
    )
    assert calls == [
        dict(
            block_rows=n,
            paired_position_sort=True,
            mlp_window=True,
            rolled_prefix=True,
            expert_panels=True,
            sorted_local_merge=True,
            key_tile=512,
        )
        for n in (128, 114)
    ]
    monkeypatch.setattr(adapter, "replicated", lambda mesh, v: np.asarray(v).copy())
    tokens = np.arange(128, dtype=np.int32)
    with pytest.raises(ValueError):
        adapter.graph_inputs("mesh", tokens, None, None, (), None)
    actual = adapter.graph_inputs("mesh", tokens, None, None, (), None, mlp_window=True)
    assert actual[0].shape == (128,) and int(actual[1]) == 128
    # Builder support is not admission: existing short worker request stays narrow.
    from glm_tpu.greenfield.validation.ws32_prefill import require_batched_profile

    with pytest.raises(ValueError, match="explicit window scope"):
        require_batched_profile(
            adapter.PREFILL_MODE,
            exact_dsa=True,
            host_main_rope_table=True,
            block_rows=128,
            long_context=None,
            adjudication_record=None,
            adjudication_sha256="0" * 64,
        )


def fake_workload(
    monkeypatch, *, failure=None, prompt=28, mlp_window=False, tail_graph_rows=None,
    live_block_rows=None
):
    cfg = replace(config(), exact_dsa=False, strategy_nd_dense=False)
    plan = adapter.BatchedPrefillPlan(
        prompt,
        128 if mlp_window else 17,
        8192,
        mlp_window=mlp_window,
        tail_graph_rows=tail_graph_rows,
        live_block_rows=live_block_rows,
    )
    calls = []
    progress = []
    # Synthetic memory inputs for these host-control tests. Real CPU buffers
    # and compiled-analysis wiring are covered in test_ws32_prefill_memory.
    census = dict(
        schema_version=SCHEMA,
        includes_all_live_arrays=True,
        devices=[
            dict(
                device_id=0,
                buffers=[dict(bytes=100)],
                accounted_resident_bytes=100,
                memory_stats=dict(
                    bytes_in_use=100, peak_bytes_in_use=100, bytes_limit=1000
                ),
            )
        ],
    )
    analyses = {
        name: dict(
            argument_size_in_bytes=100,
            output_size_in_bytes=10,
            temp_size_in_bytes=10,
            generated_code_size_in_bytes=10,
            alias_size_in_bytes=0,
        )
        for name in adapter.GRAPHS
    }

    def memory_record(
        compiled,
        trees,
        *,
        devices,
        required_reserve_bytes,
        additional_resident_executables=None
    ):
        assert not calls  # Exactly before the first model dispatch.
        assert set(trees) == {"active_inputs"} and len(trees["active_inputs"]) == 6
        return dict(
            schema_version="ws32_prefill_memory_record_v1",
            census=census,
            compiled_memory=analyses,
            required_reserve_bytes=required_reserve_bytes,
            budgets={
                name: budget_prefill_execution(
                    census,
                    analyses,
                    active_graph=name,
                    resident_graphs=adapter.GRAPHS,
                    required_reserve_bytes=required_reserve_bytes,
                )
                for name in analyses
            },
        )

    monkeypatch.setattr(adapter, "make_prefill_memory_record", memory_record)
    monkeypatch.setattr(
        adapter,
        "execute_graph_pair",
        partial(adapter.execute_graph_pair, required_memory_reserve_bytes=100),
    )

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
            physical_rows = dict(plan.graph_rows)[name]
            assert tokens.size == physical_rows and count.shape == ()
            prior = int(state.decoder.position[0])
            end = prior + int(count)
            np.testing.assert_array_equal(
                tokens[: int(count)], np.arange(prior, end, dtype=np.int32)
            )
            np.testing.assert_array_equal(
                tokens[int(count) :], np.zeros(physical_rows - int(count), np.int32)
            )
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


def test_window_host_consumes_all2034_ids_in16_calls(monkeypatch):
    args, calls, progress = fake_workload(monkeypatch, prompt=2034, mlp_window=True)
    decoder, token, record = adapter.execute_graph_pair(
        *args, budget_seconds=10, progress=progress.append, fleet_all=bool
    )
    assert calls == [("prefill_chunk", i * 128, (i + 1) * 128) for i in range(15)] + [
        ("prefill_tail", 1920, 2034)
    ]
    assert decoder.position.tolist() == [2034] and token.tolist() == [123]
    assert len(record["block_wall_seconds"]) == 16 and not record["ttft_measured"]
    adapter.validate_execution_record(record, args[2])


def test_frozen_graphs_consume8155_live_ids_in64_calls(monkeypatch):
    args, calls, progress = fake_workload(
        monkeypatch, prompt=8155, mlp_window=True, tail_graph_rows=114
    )
    plan = args[2]
    assert plan.split == (63, 91)
    assert plan.graph_rows == (("prefill_chunk", 128), ("prefill_tail", 114))
    decoder, token, record = adapter.execute_graph_pair(
        *args, budget_seconds=100, progress=progress.append, fleet_all=bool
    )
    assert calls == [("prefill_chunk", i * 128, (i + 1) * 128) for i in range(63)] + [
        ("prefill_tail", 8064, 8155)
    ]
    assert decoder.position.tolist() == [8155] and token.tolist() == [123]
    assert progress[-1]["rows"] == 91 and len(progress) == 64
    assert record["identity"]["tail_length"] == 91
    assert record["identity"]["tail_graph_rows"] == 114
    adapter.validate_execution_record(record, plan)
    forged = deepcopy(record)
    forged["identity"]["tail_length"] = 114
    with pytest.raises(ValueError, match="identity"):
        adapter.validate_execution_record(forged, plan)


@pytest.mark.parametrize("rows", [True, 114.0, 90, 129, -1])
def test_physical_tail_rejects_invalid_geometry(rows):
    with pytest.raises(ValueError, match="physical tail"):
        adapter.BatchedPrefillPlan(
            8155, 128, 8192, mlp_window=True, tail_graph_rows=rows
        )


def test_physical_input_preserves_live_count_and_ids(monkeypatch):
    monkeypatch.setattr(adapter, "replicated", lambda mesh, v: np.asarray(v).copy())
    tokens = np.arange(91, dtype=np.int32) + 30
    actual = adapter.graph_inputs(
        None, tokens, None, None, (), None, mlp_window=True, physical_rows=114
    )
    assert actual[0].shape == (114,) and actual[0].dtype == np.int32
    assert actual[1].shape == () and int(actual[1]) == 91
    np.testing.assert_array_equal(actual[0][:91], tokens)
    np.testing.assert_array_equal(actual[0][91:], np.zeros(23, np.int32))
    for rows in (True, 114.0, 90, 129):
        with pytest.raises(ValueError, match="physical input"):
            adapter.graph_inputs(
                None, tokens, None, None, (), None, mlp_window=True, physical_rows=rows
            )


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
        return len(votes) < (2 if peer_refusal == "health" else 3)

    with pytest.raises(RuntimeError):
        adapter.execute_graph_pair(
            *args, budget_seconds=10, progress=progress.append, fleet_all=fleet_all
        )
    assert len(calls) == 1
    assert len(votes) == (2 if peer_refusal == "health" else 3)


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
    assert votes == [True, True, False] and len(calls) == 1


def test_consensus_runs_for_every_block_including_final(monkeypatch):
    args, calls, progress = fake_workload(monkeypatch, prompt=34)
    votes = []

    def fleet_all(local):
        votes.append(local)
        return local

    adapter.execute_graph_pair(
        *args, budget_seconds=10, progress=progress.append, fleet_all=fleet_all
    )
    assert votes == [True] * (1 + 2 * len(calls))


def test_peer_memory_refusal_prevents_any_dispatch(monkeypatch):
    args, calls, progress = fake_workload(monkeypatch)
    votes = []

    def fleet_all(local):
        votes.append(local)
        return False

    with pytest.raises(RuntimeError, match="initial memory budget"):
        adapter.execute_graph_pair(
            *args, budget_seconds=10, progress=progress.append, fleet_all=fleet_all
        )
    assert votes == [True] and calls == []


def test_memory_capture_failure_votes_before_any_dispatch(monkeypatch):
    args, calls, progress = fake_workload(monkeypatch)

    def fail(*args, **kwargs):
        raise ValueError("device memory counters unavailable")

    monkeypatch.setattr(adapter, "make_prefill_memory_record", fail)
    votes = []
    with pytest.raises(RuntimeError, match="initial memory budget") as error:
        adapter.execute_graph_pair(
            *args,
            budget_seconds=10,
            progress=progress.append,
            fleet_all=lambda v: votes.append(v) or v,
        )
    assert isinstance(error.value.__cause__, ValueError)
    assert votes == [False] and calls == []


def test_memory_estimate_failure_is_not_overridden_by_a_forged_pass(monkeypatch):
    args, calls, progress = fake_workload(monkeypatch)
    with pytest.raises(RuntimeError, match="initial memory budget"):
        adapter.execute_graph_pair(
            *args,
            budget_seconds=10,
            progress=progress.append,
            fleet_all=bool,
            required_memory_reserve_bytes=999,
        )
    assert calls == []


def test_adapter_releases_old_cache_generation_before_next_dispatch(monkeypatch):
    args, calls, progress = fake_workload(monkeypatch, prompt=51)
    fresh = adapter.make_ws32_batched_prefill_state
    initial_cache = []

    def capture_initial(*a, **kw):
        state = fresh(*a, **kw)
        initial_cache.append(weakref.ref(state.decoder.kv_cache_local))
        return state

    monkeypatch.setattr(adapter, "make_ws32_batched_prefill_state", capture_initial)
    compiled = args[4]
    for name, run in list(compiled.items()):

        def new_generation(*inputs, original=run):
            if calls:
                assert initial_cache[0]() is None
            result = original(*inputs)
            decoder = result.state.decoder._replace(
                kv_cache_local=np.array(result.state.decoder.kv_cache_local, copy=True)
            )
            return result._replace(state=result.state._replace(decoder=decoder))

        compiled[name] = new_generation
    adapter.execute_graph_pair(
        *args, budget_seconds=10, progress=progress.append, fleet_all=bool
    )
    assert len(calls) == 3 and initial_cache[0]() is None


def test_one_time_memory_census_cost_is_not_extrapolated_per_block(monkeypatch):
    args, calls, progress = fake_workload(monkeypatch)
    original_clock = adapter.perf_counter
    original_memory = adapter.make_prefill_memory_record
    jump = [0.0]
    monkeypatch.setattr(adapter, "perf_counter", lambda: original_clock() + jump[0])

    def slow_memory(*a, **kw):
        result = original_memory(*a, **kw)
        jump[0] += 1.0
        return result

    monkeypatch.setattr(adapter, "make_prefill_memory_record", slow_memory)
    _, _, record = adapter.execute_graph_pair(
        *args, budget_seconds=1.4, progress=progress.append, fleet_all=bool
    )
    assert len(calls) == 2
    assert record["memory_admission_seconds"] >= 1
    assert record["projected_total_seconds_max"] < 1.4


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


def test_memory_publication_precedes_first_dispatch(monkeypatch):
    args, calls, progress = fake_workload(monkeypatch)
    published = []

    def memory_progress(record):
        assert not calls
        assert record["error"] is None
        assert record["memory_admission"]["required_reserve_bytes"] == 100
        published.append(record)

    adapter.execute_graph_pair(
        *args,
        budget_seconds=10,
        progress=progress.append,
        fleet_all=bool,
        memory_progress=memory_progress,
    )
    assert len(published) == 1


def test_memory_publication_failure_votes_before_any_dispatch(monkeypatch):
    args, calls, progress = fake_workload(monkeypatch)
    votes = []

    def memory_progress(record):
        raise OSError("publication failed")

    def vote(value):
        votes.append(value)
        return value

    with pytest.raises(RuntimeError, match="initial memory budget"):
        adapter.execute_graph_pair(
            *args,
            budget_seconds=10,
            progress=progress.append,
            fleet_all=vote,
            memory_progress=memory_progress,
        )
    assert votes == [False] and not calls
