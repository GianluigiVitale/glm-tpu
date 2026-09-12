"""Owner joins and admission replay, not real TPU peak evidence."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from glm_tpu.greenfield.validation.ws32_prefill_admission import (
    SHORT_PROFILE,
    SHORT_PLAN,
    SHORT_RESERVE_BYTES,
    short_acquisition,
)
from glm_tpu.greenfield.validation.ws32_prefill import PREFILL_MODE
from glm_tpu.greenfield.validation.ws32_prefill_fleet_memory import (
    validate_batched_fleet_memory,
)
from glm_tpu.greenfield.validation.ws32_prefill_memory import (
    SCHEMA,
    budget_prefill_execution,
)


PHYSICAL = tuple(
    i
    for row in (
        [0, 8, 16, 24],
        [2, 10, 18, 26],
        [4, 12, 20, 28],
        [6, 14, 22, 30],
        [1, 9, 17, 25],
        [3, 11, 19, 27],
        [5, 13, 21, 29],
        [7, 15, 23, 31],
    )
    for i in row
)
LIMIT = 33_014_398_976
ROOT = Path(__file__).resolve().parents[3]


def fixture():
    acquired = short_acquisition(ROOT)["fleet"][0]["compiled"]
    analyses = {g: acquired[g]["memory"] for g in ("prefill_chunk", "prefill_tail")}
    records, captures = [], []
    for rank, process in enumerate([3, 5, 1, 2, 0, 6, 7, 4]):
        ids = list(range(process * 4, process * 4 + 4))
        capture = dict(
            jax_process_index=process, local_device_ids=ids, hostname=f"host{rank}"
        )
        captures.append(capture)
        before, middle, last = [], [], []
        for device in ids:
            identity = dict(device_id=device, process_index=process, platform="tpu")
            stats = dict(
                bytes_in_use=25_000_000_000,
                peak_bytes_in_use=26_000_000_000,
                bytes_limit=LIMIT,
            )
            before.append(
                dict(
                    **identity,
                    memory_stats=stats,
                    buffers=[dict(bytes=25_000_000_000)],
                    accounted_resident_bytes=25_000_000_000,
                )
            )
            middle.append(
                dict(**identity, **{**stats, "peak_bytes_in_use": 26_000_001_000})
            )
            last.append(
                dict(
                    **identity,
                    **{
                        **stats,
                        "bytes_in_use": 24_000_000_000,
                        "peak_bytes_in_use": 26_000_002_000,
                    },
                )
            )
        census = dict(
            schema_version=SCHEMA,
            devices=before,
            includes_all_live_arrays=True,
            execution_peak_measured=False,
        )
        admission = dict(
            schema_version="ws32_prefill_memory_record_v1",
            census=census,
            compiled_memory=deepcopy(analyses),
            required_reserve_bytes=SHORT_RESERVE_BYTES,
            budgets={
                g: budget_prefill_execution(
                    census,
                    analyses,
                    active_graph=g,
                    resident_graphs=tuple(analyses),
                    required_reserve_bytes=SHORT_RESERVE_BYTES,
                )
                for g in analyses
            },
        )
        records.append(
            dict(
                launch_process_id=rank,
                jax_process_index=process,
                hostname=f"host{rank}",
                local_device_slots=[
                    dict(
                        device_id=i,
                        device_slot=PHYSICAL.index(i),
                        expert_coordinate=PHYSICAL.index(i) // 4,
                        feature_coordinate=PHYSICAL.index(i) % 4,
                    )
                    for i in ids
                ],
                prefill_execution={"memory_admission": admission},
                compiled_memory_analysis=deepcopy(analyses),
                batched_prefill_memory=middle,
                batched_device_memory_after_execute=last,
            )
        )
    return dict(
        records=records,
        ordered_captures=captures,
        flattened_device_ids=PHYSICAL,
        required_reserve_bytes=SHORT_RESERVE_BYTES,
        expected_device_limit_bytes=LIMIT,
        expected_prefill_analyses=analyses,
    )


def test_reorders_join_by_owner_not_position_and_lower_current_is_valid():
    args = fixture()
    expected = validate_batched_fleet_memory(**args)
    assert expected["owner_count"] == 32
    assert expected["final_lifetime_peak_bytes"] == 26_000_002_000
    for record in args["records"]:
        for key in (
            "local_device_slots",
            "batched_prefill_memory",
            "batched_device_memory_after_execute",
        ):
            record[key].reverse()
    assert validate_batched_fleet_memory(**args) == expected


@pytest.mark.parametrize("boundary", ["census", "prefill", "final"])
@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "foreign",
        "bool_id",
        "bool_process",
        "process",
        "platform",
        "limit",
        "inconsistent",
        "reserve",
    ],
)
def test_every_boundary_refuses_bad_owners_or_counters(boundary, mutation):
    args = fixture()
    r = args["records"][0]
    values = (
        r["prefill_execution"]["memory_admission"]["census"]["devices"]
        if boundary == "census"
        else (
            r["batched_prefill_memory"]
            if boundary == "prefill"
            else r["batched_device_memory_after_execute"]
        )
    )
    item = values[0]
    stats = item["memory_stats"] if boundary == "census" else item
    if mutation == "missing":
        values.pop()
    elif mutation == "duplicate":
        values[-1] = deepcopy(item)
    elif mutation == "foreign":
        item["device_id"] = 31
    elif mutation == "bool_id":
        item["device_id"] = True
    elif mutation == "bool_process":
        item["process_index"] = True
    elif mutation == "process":
        item["process_index"] = 7
    elif mutation == "platform":
        item["platform"] = "cpu"
    elif mutation == "limit":
        stats["bytes_limit"] += 1
    elif mutation == "inconsistent":
        stats["bytes_in_use"] = LIMIT
    elif mutation == "reserve":
        stats["peak_bytes_in_use"] = LIMIT - SHORT_RESERVE_BYTES + 1
    with pytest.raises(ValueError):
        validate_batched_fleet_memory(**args)


@pytest.mark.parametrize(
    "mutation",
    [
        "slot_swap",
        "slot_bool",
        "capture_duplicate",
        "launch_rank",
        "hostname",
        "reserve",
        "analysis_runner",
        "analysis_record",
        "extra_program",
        "budget_device",
        "census_is_measurement",
        "prefill_peak_decrease",
        "final_peak_decrease",
        "missing_host",
        "physical_duplicate",
        "coordinate",
    ],
)
def test_cross_boundary_and_acquisition_binding(mutation):
    args = fixture()
    r = args["records"][0]
    m = r["prefill_execution"]["memory_admission"]
    if mutation == "slot_swap":
        slots = r["local_device_slots"]
        slots[0]["device_id"], slots[1]["device_id"] = (
            slots[1]["device_id"],
            slots[0]["device_id"],
        )
    elif mutation == "slot_bool":
        r["local_device_slots"][0]["device_slot"] = True
    elif mutation == "capture_duplicate":
        args["ordered_captures"][1] = deepcopy(args["ordered_captures"][0])
    elif mutation == "launch_rank":
        r["launch_process_id"] = 4
    elif mutation == "hostname":
        r["hostname"] = "wrong"
    elif mutation == "reserve":
        args["required_reserve_bytes"] -= 1
    elif mutation == "analysis_runner":
        r["compiled_memory_analysis"]["prefill_chunk"]["temp_size_in_bytes"] += 1
    elif mutation == "analysis_record":
        m["compiled_memory"]["prefill_chunk"]["temp_size_in_bytes"] += 1
    elif mutation == "extra_program":
        m["compiled_memory"]["decode"] = deepcopy(m["compiled_memory"]["prefill_chunk"])
    elif mutation == "budget_device":
        m["budgets"]["prefill_chunk"]["devices"].pop()
    elif mutation == "census_is_measurement":
        m["census"]["execution_peak_measured"] = True
    elif mutation == "prefill_peak_decrease":
        r["batched_prefill_memory"][0]["peak_bytes_in_use"] = 25_999_999_999
    elif mutation == "final_peak_decrease":
        r["batched_device_memory_after_execute"][0][
            "peak_bytes_in_use"
        ] = 26_000_000_999
    elif mutation == "missing_host":
        args["records"].pop()
    elif mutation == "physical_duplicate":
        args["flattened_device_ids"] = (0,) * 32
    elif mutation == "coordinate":
        r["local_device_slots"][0]["feature_coordinate"] = 3
    with pytest.raises(ValueError):
        validate_batched_fleet_memory(**args)


def test_actual_sealer_memory_entry_uses_pinned_profile_and_owner_join():
    from scripts.greenfield import seal_short_decoder_ws32 as sealer

    args = fixture()
    for r in args["records"]:
        r["batched_prefill_profile"] = SHORT_PROFILE
    mesh = SimpleNamespace(flattened_device_ids=PHYSICAL)
    report = sealer._require_batched_fleet_memory(
        args["records"], tuple(args["ordered_captures"]), mesh
    )
    assert report == validate_batched_fleet_memory(**args)
    args["records"][0]["batched_prefill_profile"] = "serial"
    with pytest.raises(ValueError, match="short numerical profile is not registered"):
        sealer._require_batched_fleet_memory(
            args["records"], tuple(args["ordered_captures"]), mesh
        )


def execution_fixture():
    r = fixture()["records"][0]
    memory = r["prefill_execution"]["memory_admission"]
    blocks = SHORT_PLAN.split[0] + 1
    r.update(
        prefill_mode=PREFILL_MODE,
        batched_prefill_profile=SHORT_PROFILE,
        batched_prefill_plan=SHORT_PLAN.identity(),
        prefill_chunk_length=17,
        context_capacity=8192,
        observed_generated_token_ids=[100, 101],
        prefill_execution=dict(
            identity=SHORT_PLAN.identity(),
            budget_seconds=300.0,
            cache_initialization_seconds=1.0,
            memory_admission_seconds=1.0,
            memory_admission=memory,
            input_transfer_seconds=[0.001] * blocks,
            block_wall_seconds=[0.1] * blocks,
            projected_total_seconds_max=20.0,
            request_prefill_seconds=15.0,
            final_frontier=2034,
            finished_healthy=True,
            repaired_index_installed=True,
            first_token_ready=100,
            ttft_measured=False,
            timing_scope="input_ids_ready_to_prefill_token_ready_not_delivery",
        ),
    )
    return r


def test_actual_sealer_accounting_preserves_batched_wall_and_token_semantics():
    from scripts.greenfield import seal_short_decoder_ws32 as sealer

    r = execution_fixture()
    sealer._require_prefill_execution(
        r,
        mode="numerical",
        prompt_length=2034,
        rank=0,
        expected_chunk=17,
        prefill_mode=PREFILL_MODE,
    )
    assert "total_seconds" not in r["prefill_execution"]
    with pytest.raises(SystemExit, match="cannot authorize batched"):
        sealer._require_prefill_execution(
            r, mode="numerical", prompt_length=2034, rank=0, expected_chunk=17
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "token",
        "budget",
        "frontier",
        "profile",
        "capacity",
        "plan",
        "ttft",
        "missing_memory",
    ],
)
def test_batched_execution_refuses_changed_evidence(mutation):
    from scripts.greenfield import seal_short_decoder_ws32 as sealer

    r = execution_fixture()
    if mutation == "token":
        r["observed_generated_token_ids"][0] = 102
    elif mutation == "budget":
        r["prefill_execution"]["budget_seconds"] = 3600.0
    elif mutation == "frontier":
        r["prefill_execution"]["final_frontier"] -= 1
    elif mutation == "profile":
        r["batched_prefill_profile"] = ""
    elif mutation == "capacity":
        r["context_capacity"] = 131072
    elif mutation == "plan":
        r["batched_prefill_plan"]["block_rows"] = 32
    elif mutation == "ttft":
        r["prefill_execution"]["ttft_measured"] = True
    elif mutation == "missing_memory":
        r["prefill_execution"]["memory_admission"]["budgets"] = {}
    with pytest.raises(ValueError):
        sealer._require_prefill_execution(
            r,
            mode="numerical",
            prompt_length=2034,
            rank=0,
            expected_chunk=17,
            prefill_mode=PREFILL_MODE,
        )
