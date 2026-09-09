"""Production geometry + local overhead lifecycle, fake device allocations."""

from copy import deepcopy
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

from scripts.greenfield import prefill_budget_overhead as overhead
from scripts.greenfield import prefill_budget_probe as probe


class Array:
    def __init__(self, value=None, *, shape=None):
        self.deleted = False
        self.value = value
        self.shape = value.shape if shape is None else shape
        self.dtype = value.dtype if shape is None else np.dtype(ml_dtypes.bfloat16)
        local_shape = (
            self.shape
            if shape is None
            else (shape[0], shape[1], shape[2] // 8, shape[3])
        )
        self.addressable_shards = [
            SimpleNamespace(
                device=SimpleNamespace(id=i),
                data=(
                    value.copy()
                    if shape is None
                    else SimpleNamespace(
                        shape=local_shape,
                        size=int(np.prod(local_shape)),
                        dtype=self.dtype,
                    )
                ),
            )
            for i in range(4)
        ]

    def __array__(self, dtype=None, copy=None):
        assert self.value is not None and not self.deleted
        return self.value

    def delete(self):
        assert not self.deleted
        self.deleted = True


def state(config, prompt):
    from glm_tpu.greenfield.runtime.ws32_batched_prefill import Ws32BatchedPrefillState
    from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderState

    a = lambda x, dtype: Array(np.asarray(x, dtype))
    decoder = Ws32DecoderState(
        Array(shape=config.kv_cache_shape),
        Array(shape=config.index_cache_shape),
        a(np.full((1, 2048), -1), np.int32),
        a([0], np.int32),
        a(np.full((1, 2048), -np.inf), np.float32),
        a([0], np.int32),
        a(np.arange(config.page_count)[None, :], np.int32),
        a([1], np.int32),
        a([True], np.bool_),
    )
    return Ws32BatchedPrefillState(
        decoder,
        Array(shape=config.index_cache_shape),
        a(prompt, np.int32),
        a(False, np.bool_),
    )


def test_actual_full_geometry_fresh_structure_without_cache_download():
    for config, (capacity, prompt) in zip(overhead.configs(), probe.CAPACITIES):
        current = state(config, prompt)
        report = overhead.state_structure(current, config, {i: i for i in range(4)})
        independent = (
            78 * (capacity // 8) * 640 * 2
            + 2 * 21 * (capacity // 8) * 128 * 2
            + 2 * 2048 * 4
            + config.page_count * 4
            + 18
        )
        assert report["local_payload_bytes"] == {str(i): independent for i in range(4)}
        assert report["cache_content_checked"] is False
        bad = current._replace(prompt_length=Array(np.asarray(prompt - 1, np.int32)))
        with pytest.raises(ValueError, match="metadata"):
            overhead.state_structure(bad, config, {i: i for i in range(4)})


@pytest.mark.parametrize("process", [0, 1])
def test_overhead_full_lifecycle_releases_states_and_replays_report(
    monkeypatch, process
):
    import jax
    from scripts.greenfield import ws32_batched_prefill_runner as runner

    states, inputs, phases = [], [], []
    ticks = 0.0

    def clock():
        nonlocal ticks
        ticks += 0.01
        return ticks

    def phase(name, fn):
        phases.append(name)
        return fn()

    calls = SimpleNamespace(
        phase=phase,
        local_slots={i: i for i in range(4)},
        record={"jax_process_index": process},
    )
    monkeypatch.setattr(jax, "block_until_ready", lambda value: value)

    def initialize(mesh, config, *, prompt_length, clock):
        assert all(leaf.deleted for old in states for leaf in jax.tree.leaves(old))
        current = state(config, prompt_length)
        states.append(current)
        return current, dict(
            capacity=config.context_capacity,
            prompt_length=prompt_length,
            cache_initialization_seconds=0.01,
            weights_loaded=False,
            initialized_prefix_length=0,
            model_ttft_measured=False,
        )

    monkeypatch.setattr(probe, "measure_initial_state", initialize)
    monkeypatch.setattr(
        overhead,
        "capture_identified_device_memory",
        lambda devices: [
            dict(
                device_id=i,
                process_index=process,
                platform="tpu",
                bytes_in_use=(
                    4_000_000_000 if states and not states[-1].finished.deleted else 0
                ),
                peak_bytes_in_use=4_000_000_000,
                bytes_limit=33_014_398_976,
            )
            for i in range(4)
        ],
    )

    def place(mesh, value):
        array = Array(np.asarray(value))
        inputs.append(array)
        return array

    monkeypatch.setattr(runner, "replicated", place)
    report = overhead.run(calls, "mesh", clock=clock)
    assert phases == overhead.expected_stages()
    assert len(states) == 4 and all(
        leaf.deleted for old in states for leaf in jax.tree.leaves(old)
    )
    assert len(inputs) == 77 and all(a.deleted for a in inputs)
    assert len(report["delivery"]) == (5 if process == 0 else 0)
    overhead.validate_record(report, slots=calls.local_slots, process_index=process)
    for kind in (
        "model",
        "init_count",
        "shape",
        "memory",
        "resident",
        "input_count",
        "scope",
        "elapsed",
        "delivery",
    ):
        bad = deepcopy(report)
        first = bad["initialization"]["131072"]
        if kind == "model":
            bad["model_ttft_measured"] = True
        elif kind == "init_count":
            first.pop()
        elif kind == "shape":
            first[0]["structure"]["local_payload_bytes"]["0"] += 1
        elif kind == "memory":
            first[0]["after_memory"][0]["peak_bytes_in_use"] = 33_000_000_000
        elif kind == "resident":
            first[0]["after_memory"][0]["bytes_in_use"] = 0
        elif kind == "input_count":
            bad["input_placement"]["17"]["samples_seconds"].pop()
        elif kind == "scope":
            bad["input_placement"]["262144"]["scope"] = "CURRENT_BLOCK_GRAPH_INPUTS"
        elif kind == "elapsed":
            bad["elapsed_seconds"] = 241
        else:
            if process == 0:
                bad["delivery"].pop()
            else:
                bad["delivery"].append({})
        with pytest.raises(ValueError):
            overhead.validate_record(
                bad, slots=calls.local_slots, process_index=process
            )


def test_initializer_failure_releases_its_fresh_state(monkeypatch):
    import jax

    states = []
    calls = SimpleNamespace(
        phase=lambda n, f: f(),
        local_slots={i: i for i in range(4)},
        record={"jax_process_index": 0},
    )

    def init(mesh, config, *, prompt_length, clock):
        value = state(config, prompt_length)
        states.append(value)
        return value, {}

    monkeypatch.setattr(probe, "measure_initial_state", init)
    monkeypatch.setattr(
        overhead,
        "capture_identified_device_memory",
        lambda devices: [
            dict(
                device_id=i,
                process_index=0,
                platform="tpu",
                bytes_in_use=4_000_000_000,
                peak_bytes_in_use=4_000_000_000,
                bytes_limit=33_014_398_976,
            )
            for i in range(4)
        ],
    )

    def fail(*args):
        raise ValueError("wrong state")

    monkeypatch.setattr(overhead, "state_structure", fail)
    with pytest.raises(ValueError, match="wrong state"):
        overhead.run(calls, "mesh")
    assert len(states) == 1 and all(leaf.deleted for leaf in jax.tree.leaves(states[0]))
