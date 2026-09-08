"""Resident alias accounting and prefill budget arithmetic, never TPU proof."""

from copy import deepcopy
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from glm_tpu.greenfield.validation import ws32_prefill_memory as memory


def census():
    return dict(
        schema_version=memory.SCHEMA,
        includes_all_live_arrays=True,
        devices=[
            dict(
                device_id=0,
                buffers=[dict(bytes=100)],
                accounted_resident_bytes=100,
                memory_stats=dict(
                    bytes_in_use=120, peak_bytes_in_use=140, bytes_limit=1000
                ),
            )
        ],
    )


def analyses():
    return {
        "prefill_chunk": dict(
            argument_size_in_bytes=90,
            output_size_in_bytes=20,
            temp_size_in_bytes=30,
            generated_code_size_in_bytes=10,
            alias_size_in_bytes=0,
        ),
        "prefill_tail": dict(
            argument_size_in_bytes=90,
            output_size_in_bytes=20,
            temp_size_in_bytes=25,
            generated_code_size_in_bytes=12,
            alias_size_in_bytes=0,
        ),
    }


def record():
    c, a = census(), analyses()
    return dict(
        schema_version="ws32_prefill_memory_record_v1",
        census=c,
        compiled_memory=a,
        required_reserve_bytes=100,
        budgets={
            name: memory.budget_prefill_execution(
                c,
                a,
                active_graph=name,
                resident_graphs=tuple(a),
                required_reserve_bytes=100,
            )
            for name in a
        },
    )


def test_only_active_scratch_and_outputs_but_both_resident_code():
    r = record()
    memory.validate_prefill_memory_record(r)
    main = r["budgets"]["prefill_chunk"]["devices"][0]
    assert main["resident_baseline_bytes"] == 120
    assert main["estimated_peak_bytes"] == 120 + 10 + 12 + 20 + 30
    assert r["budgets"]["prefill_tail"]["devices"][0]["estimated_peak_bytes"] == 187
    assert r["budgets"]["prefill_tail"]["numerical_admission"] is False


@pytest.mark.parametrize(
    "resident,used,arguments,baseline",
    [(150, 120, 90, 150), (100, 120, 160, 160), (100, 120, 90, 120)],
)
def test_baseline_cannot_omit_resident_nonarguments_or_compiler_padding(
    resident, used, arguments, baseline
):
    c, a = census(), analyses()
    c["devices"][0]["buffers"][0]["bytes"] = resident
    c["devices"][0]["accounted_resident_bytes"] = resident
    c["devices"][0]["memory_stats"]["bytes_in_use"] = used
    a["prefill_chunk"]["argument_size_in_bytes"] = arguments
    result = memory.budget_prefill_execution(
        c,
        a,
        active_graph="prefill_chunk",
        resident_graphs=tuple(a),
        required_reserve_bytes=100,
    )
    assert result["devices"][0]["resident_baseline_bytes"] == baseline


def test_prior_peak_and_insufficient_reserve_refuse():
    c, a = census(), analyses()
    c["devices"][0]["memory_stats"]["peak_bytes_in_use"] = 950
    result = memory.budget_prefill_execution(
        c,
        a,
        active_graph="prefill_chunk",
        resident_graphs=tuple(a),
        required_reserve_bytes=100,
    )
    assert result["estimate_fits"] is False
    assert result["devices"][0]["estimated_headroom_bytes"] == 50


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_tail",
        "duplicate_graph",
        "missing_code",
        "unknown_alias",
        "bool_reserve",
        "no_reserve",
        "no_live",
        "empty",
        "duplicate_device",
        "total",
        "counters",
    ],
)
def test_bad_or_incomplete_budget_inputs_refuse(mutation):
    c, a = census(), analyses()
    graphs = list(a)
    reserve = 100
    if mutation == "missing_tail":
        graphs.pop()
    if mutation == "duplicate_graph":
        graphs.append(graphs[0])
    if mutation == "missing_code":
        a["prefill_tail"]["generated_code_size_in_bytes"] = None
    if mutation == "unknown_alias":
        a["prefill_chunk"]["alias_size_in_bytes"] = 1
    if mutation == "bool_reserve":
        reserve = True
    if mutation == "no_reserve":
        reserve = 0
    if mutation == "no_live":
        c["includes_all_live_arrays"] = False
    if mutation == "empty":
        c["devices"] = []
    if mutation == "duplicate_device":
        c["devices"] *= 2
    if mutation == "total":
        c["devices"][0]["accounted_resident_bytes"] = 0
    if mutation == "counters":
        c["devices"][0]["memory_stats"]["bytes_in_use"] = 200
    with pytest.raises(ValueError):
        memory.budget_prefill_execution(
            c,
            a,
            active_graph="prefill_chunk",
            resident_graphs=graphs,
            required_reserve_bytes=reserve,
        )


def test_record_pass_is_recomputed_not_trusted():
    r = record()
    r["budgets"]["prefill_chunk"]["devices"][0]["estimated_peak_bytes"] = 0
    with pytest.raises(ValueError, match="drifted"):
        memory.validate_prefill_memory_record(r)
    r = record()
    r["required_reserve_bytes"] = 999
    with pytest.raises(ValueError, match="fails"):
        memory.validate_prefill_memory_record(r)


class StatsDevice:
    """CPU-test counter proxy, not a measured device memory assertion."""

    def __init__(self, device, *, missing=False):
        self.id = device.id
        self.platform = device.platform
        self.process_index = device.process_index
        self.missing = missing

    def memory_stats(self):
        return (
            None
            if self.missing
            else dict(
                bytes_in_use=1000000, peak_bytes_in_use=1000000, bytes_limit=10000000
            )
        )


def test_real_cpu_arrays_deduplicate_shared_roots_and_count_nonargument_residents():
    a = jnp.ones((128,), jnp.float32)
    b = jnp.zeros((256,), jnp.float32)
    a.block_until_ready()
    b.block_until_ready()
    # b is deliberately NOT supplied as a named root; live_arrays must find it.
    result = memory.capture_resident_buffers(
        {"raw": (a,), "decode": (a,)},
        devices=[StatsDevice(d) for d in jax.local_devices()],
    )
    dev = result["devices"][0]
    assert dev["group_resident_bytes"]["raw"] == 512
    assert dev["group_resident_bytes"]["decode"] == 512
    assert dev["accounted_resident_bytes"] >= 512 + 1024
    shared = [
        entry for entry in dev["buffers"] if {"raw", "decode"}.issubset(entry["groups"])
    ]
    assert len(shared) == 1 and shared[0]["bytes"] == 512
    assert shared[0]["identity_mode"] == "physical_pointer"
    assert result["execution_peak_measured"] is False
    assert all("pointer" not in entry for entry in dev["buffers"])


def test_missing_runtime_counters_refuse_before_budget():
    a = jnp.zeros((1,), jnp.float32)
    with pytest.raises(ValueError, match="counters unavailable"):
        memory.capture_resident_buffers(
            {"weights": a},
            devices=[StatsDevice(d, missing=True) for d in jax.local_devices()],
        )


def test_actual_cpu_compiled_analysis_and_census_round_trip():
    a = jnp.ones((128,), jnp.float32)
    compiled = {
        "prefill_chunk": jax.jit(lambda x: x + 1).lower(a).compile(),
        "prefill_tail": jax.jit(lambda x: x + 2).lower(a).compile(),
    }
    r = memory.make_prefill_memory_record(
        compiled,
        {"inputs": (a,)},
        devices=[StatsDevice(d) for d in jax.local_devices()],
        required_reserve_bytes=100,
    )
    memory.validate_prefill_memory_record(r)
    assert r["compiled_memory"]["prefill_chunk"]["argument_size_in_bytes"] >= a.nbytes
    assert r["census"]["execution_peak_measured"] is False


def test_pointerless_backend_uses_conservative_distinct_objects(monkeypatch):
    device = SimpleNamespace(
        id=0,
        platform="test",
        process_index=0,
        memory_stats=lambda: dict(
            bytes_in_use=128, peak_bytes_in_use=128, bytes_limit=1000
        ),
    )

    class Array:
        nbytes = 64

        def is_deleted(self):
            return False

        def block_until_ready(self):
            return self

        def on_device_size_in_bytes(self):
            return 64

        def unsafe_buffer_pointer(self):
            raise NotImplementedError()

        @property
        def addressable_shards(self):
            return [SimpleNamespace(device=device, data=self)]

    a, b = Array(), Array()
    monkeypatch.setattr(jax, "Array", Array)
    monkeypatch.setattr(jax, "live_arrays", lambda: [a, b])
    result = memory.capture_resident_buffers({"raw": a, "decode": a}, devices=[device])
    dev = result["devices"][0]
    assert dev["accounted_resident_bytes"] == 128
    assert dev["conservative_identity_fallback"] is True
    assert len(dev["buffers"]) == 2


def test_builder_binds_actual_compiled_analyses_and_one_census(monkeypatch):
    calls = []
    monkeypatch.setattr(
        memory,
        "capture_resident_buffers",
        lambda trees, *, devices: calls.append((trees, devices)) or census(),
    )
    compiled = {
        name: SimpleNamespace(
            memory_analysis=lambda fields=fields: SimpleNamespace(**fields)
        )
        for name, fields in analyses().items()
    }
    result = memory.make_prefill_memory_record(
        compiled,
        {"weights": "opaque-test"},
        devices=("test",),
        required_reserve_bytes=100,
    )
    assert len(calls) == 1
    memory.validate_prefill_memory_record(result)


def test_additional_resident_model_code_cannot_be_omitted(monkeypatch):
    monkeypatch.setattr(memory, "capture_resident_buffers", lambda *a, **kw: census())
    compiled = {
        name: SimpleNamespace(
            memory_analysis=lambda fields=fields: SimpleNamespace(**fields)
        )
        for name, fields in analyses().items()
    }
    extra_fields = {**analyses()["prefill_chunk"], "generated_code_size_in_bytes": 101}
    result = memory.make_prefill_memory_record(
        compiled,
        {"weights": "test"},
        devices=("test",),
        required_reserve_bytes=100,
        additional_resident_executables={
            "observer": SimpleNamespace(
                memory_analysis=lambda: SimpleNamespace(**extra_fields)
            )
        },
    )
    memory.validate_prefill_memory_record(result)
    assert (
        result["budgets"]["prefill_chunk"]["devices"][0]["resident_code_bytes"] == 123
    )
    forged = deepcopy(result)
    forged["budgets"]["prefill_chunk"]["resident_graphs"].remove("observer")
    with pytest.raises(ValueError, match="omits declared"):
        memory.validate_prefill_memory_record(forged)
    with pytest.raises(ValueError, match="overlap"):
        memory.make_prefill_memory_record(
            compiled,
            {"weights": "test"},
            devices=("test",),
            required_reserve_bytes=100,
            additional_resident_executables=compiled,
        )
