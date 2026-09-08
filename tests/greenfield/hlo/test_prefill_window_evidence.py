"""Actual numerical worker -> JSON/journal/NPZ -> original replay, CPU fixtures.

Only executable math/device counters are fixtures. Uses all four archived DB590
graph bodies, actual compile writer, worker phases and the independent consumer.
"""

from copy import deepcopy
from contextlib import contextmanager
import json
import inspect
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from scripts.greenfield import prefill_window_acquisition as acquisition
from scripts.greenfield import prefill_window_admission as admission
from scripts.greenfield import prefill_window_evidence as evidence
from scripts.greenfield import prefill_window_protocol as protocol
from scripts.greenfield import prefill_window_worker as worker
from scripts.greenfield import probe_ws32_prefill_layer as layer
from tests.greenfield.hlo.test_prefill_window_admission import ORIGINAL
from tests.greenfield.hlo.test_prefill_window_worker import fake_memory, fixture_output


@contextmanager
def completed_worker(
    root,
    *,
    rank=0,
    slots=None,
    process=3,
    configure=None,
    consensus=lambda ok: ok,
    boundary_diagnostic=False,
):
    source = ORIGINAL
    active_admission, active_evidence = admission, evidence
    if boundary_diagnostic:
        from scripts.greenfield import (
            prefill_window_boundary_admission as active_admission,
        )
        from scripts.greenfield import (
            prefill_window_boundary_evidence as active_evidence,
        )
        from scripts.greenfield import prefill_window_boundary_worker as boundary
        from tests.greenfield.hlo.test_prefill_window_boundary_admission import (
            ORIGINAL as source,
        )
    if not source.is_dir():
        pytest.skip("requires locally materialized original DB590 graphs")
    root.mkdir(parents=True, exist_ok=True)
    patches = pytest.MonkeyPatch()
    slots = (
        (
            {0: 0, 8: 1, 16: 2, 24: 3}
            if boundary_diagnostic
            else {9: 0, 13: 1, 25: 2, 29: 3}
        )
        if slots is None
        else slots
    )
    host_now, observations, sequence = {}, {}, []
    captures = {}
    acquired_record = (
        json.loads((source / "runner.json").read_text())
        if boundary_diagnostic
        else None
    )
    pins = active_admission.registered_programs()

    def memory():
        result = fake_memory()
        for row, device in zip(result["devices"], slots):
            row.update(device_id=device, process_index=process)
        return result

    wk_values = np.zeros((128, 6144), protocol.BF16)
    wk_values[3, 7] = 1.5

    class Distributed:
        def __init__(self, values):
            self.addressable_shards = [
                SimpleNamespace(device=SimpleNamespace(id=d), data=values.copy())
                for d in slots
            ]

    class Compiled:
        def __init__(self, name):
            self.name = name

        def memory_analysis(self):
            return SimpleNamespace(**pins[self.name]["compiled_memory"])

        def as_text(self):
            return (source / f"{self.name}.optimized_hlo.txt").read_text()

        @property
        def out_info(self):
            schema = acquired_record["programs"][self.name]["compiler_output_schema"]

            def leaf(value, spec=None):
                return SimpleNamespace(
                    shape=tuple(value["shape"]),
                    dtype=np.dtype(value["dtype"]),
                    sharding=SimpleNamespace(spec=spec),
                )

            return tuple(leaf(v) for v in schema["original"]), {
                k: leaf(v, schema["capture_partition_specs"][k])
                for k, v in schema["captures"].items()
            }

        def __call__(self, *values):
            sequence.append(self.name)
            if self.name == "wk_decode":
                return Distributed(wk_values)
            if self.name == "wk_promote":
                assert sequence == ["wk_decode", "wk_promote"]
                np.testing.assert_array_equal(
                    values[0].addressable_shards[0].data, wk_values
                )
                return Distributed(wk_values.astype(np.float32))
            tile = (sequence.count("control") % 4 - 1) % 4
            result = tuple(np.full(1, len(sequence)) for _ in range(12))
            out = {}
            for d, slot in slots.items():
                fields = fixture_output(host_now, slot)
                if self.name == "control":
                    fields = {
                        k: (
                            v
                            if k in ("kv", "index", "repair")
                            else v[tile * 32 : (tile + 1) * 32]
                        )
                        for k, v in fields.items()
                    }
                out[d] = fields
            observations[id(result)] = out
            if boundary_diagnostic:
                schema = acquired_record["programs"][self.name][
                    "compiler_output_schema"
                ]["captures"]
                captured = object()
                captures[id(captured)] = {
                    d: {
                        k: np.zeros(tuple(v["shape"][2:]), np.dtype(v["dtype"]))
                        for k, v in schema.items()
                    }
                    for d in slots
                }
                return result, captured
            return result

    class Function:
        def __init__(self, name):
            self.name = name

        def lower(self, *values):
            # The real compile writer is invoked through the original outer
            # chain. New numerical worker frames must not enter compilation.
            functions = [frame.function for frame in inspect.stack()]
            assert functions[1:6] == [
                "compile_program",
                "<lambda>",
                "fleet_step",
                "acquire_programs",
                "execute_acquisition",
            ]
            assert "execute_numerical" not in functions
            return SimpleNamespace(
                compiler_ir=lambda **kw: (
                    source / f"{self.name}.stablehlo.mlir"
                ).read_text(),
                compile=lambda: Compiled(self.name),
            )

    def device_inputs(host, *args):
        host_now.clear()
        host_now.update(host)
        return (
            tuple(
                host[k]
                for k in (
                    "update",
                    "residual",
                    "kv",
                    "index",
                    "repair",
                    "positions",
                    "counts",
                    "scores",
                    "offset",
                    "count",
                    "table",
                )
            )
            + (None,) * 7
            + (host["health"], host["rope"])
        )

    original_ready = jax.block_until_ready
    patches.setattr(
        jax,
        "block_until_ready",
        lambda x: x if isinstance(x, Distributed) else original_ready(x),
    )
    patches.setattr(
        jax, "local_devices", lambda: [SimpleNamespace(id=d) for d in slots]
    )
    patches.setattr(
        layer, "_memory_stats", lambda d: fake_memory()["devices"][0]["memory_stats"]
    )
    patches.setattr(layer, "input_specs", lambda *args: ())
    patches.setattr(layer, "device_inputs", device_inputs)
    patches.setattr(worker, "local_observations", lambda r: observations[id(r)])
    if boundary_diagnostic:
        patches.setattr(boundary, "local_observations", lambda r: observations[id(r)])
        patches.setattr(
            boundary, "capture_owner_arrays", lambda r, **kw: captures[id(r)]
        )
    patches.setattr(
        acquisition,
        "prepare_programs",
        lambda **kw: tuple((n, Function(n), ()) for n in admission.PROGRAMS),
    )
    patches.setattr(worker, "capture_resident_buffers", lambda *a, **k: memory())
    patches.setattr(
        worker,
        "capture_identified_device_memory",
        lambda *a: [
            dict(
                device_id=d,
                process_index=process,
                platform="tpu",
                **fake_memory()["devices"][0]["memory_stats"],
            )
            for d in slots
        ],
    )
    record = dict(
        code_hash="a" * 40,
        launch_rank=rank,
        jax_process_index=process,
        local_device_slots=[dict(device_id=d, device_slot=s) for d, s in slots.items()],
    )
    if boundary_diagnostic:
        record.update(
            protocol=boundary.PROTOCOL,
            diagnostic_only=True,
            admission_only=False,
            reference_scope=boundary.REFERENCE_SCOPE,
            physical_device_ids=boundary.original_receipt()["physical_device_ids"],
        )
    weights = SimpleNamespace(
        dsa=SimpleNamespace(wk_bits_local=np.zeros(1), wk_scale_local=np.zeros(1))
    )
    try:
        if configure is not None:
            configure(patches, record, sequence)
        acquisition.execute_acquisition(
            args=SimpleNamespace(output_dir=root),
            record=record,
            mesh=None,
            config=None,
            weights=weights,
            local_slots=slots,
            consensus=consensus,
            capture_boundaries=boundary_diagnostic,
            boundary_diagnostic=boundary_diagnostic,
        )
        assert sequence == [
            name
            for _, name in evidence.expected_calls(
                boundary_diagnostic=boundary_diagnostic
            )
        ]
        persisted = json.loads((root / "runner.json").read_text())
        assert persisted == record
        active_evidence.validate_files(root, persisted)
        yield root, persisted
    finally:
        patches.undo()


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    with completed_worker(tmp_path_factory.mktemp("window-composed")) as result:
        yield result


def test_actual_worker_producer_and_original_file_consumer(completed):
    root, record = completed
    evidence.validate_files(root, record)


@pytest.mark.parametrize(
    "change",
    [
        "owner",
        "order",
        "count",
        "peak",
        "between",
        "budget",
        "code",
        "duration",
        "post_type",
    ],
)
def test_call_replay_refuses_mutation(completed, change):
    _, original = completed
    record = deepcopy(original)
    calls = record["call_evidence"]
    if change == "owner":
        calls[4]["post_memory"][0]["process_index"] = 4
    elif change == "order":
        calls[4], calls[5] = calls[5], calls[4]
    elif change == "count":
        calls.pop()
    elif change == "peak":
        calls[4]["post_memory"][0]["peak_bytes_in_use"] = 33_000_000_000
    elif change == "between":
        calls[3]["post_memory"][0]["peak_bytes_in_use"] += 1
    elif change == "budget":
        calls[4]["budget"]["devices"][0]["active_temp_bytes"] += 1
    elif change == "code":
        calls[4]["compiled_memory"]["wk_promote"]["generated_code_size_in_bytes"] = 0
    elif change == "duration":
        calls[4]["completed_call_seconds"] = float("nan")
    elif change == "post_type":
        calls[4]["post_memory"][0]["bytes_in_use"] = 340_000_000.0
    slots = {r["device_id"]: r["device_slot"] for r in record["local_device_slots"]}
    with pytest.raises(ValueError):
        evidence.validate_calls(record, local_slots=slots)


@pytest.mark.parametrize(
    "change", ["wk_digest", "case_digest", "comparison", "journal", "phase", "totals"]
)
def test_original_consumer_refuses_mutation(completed, change):
    root, original = completed
    record = deepcopy(original)
    if change == "wk_digest":
        record["wk_originals"]["wk_promote"] = "0" * 64
    elif change == "case_digest":
        record["cases"]["tail"]["npz_sha256"] = "0" * 64
    elif change == "comparison":
        record["cases"]["boundary"]["replay"]["passed"] = 1
    elif change == "journal":
        record["compile_journal_sha256"] = "0" * 64
    elif change == "phase":
        record["current_phase"] = "tail/comparison"
    elif change == "totals":
        record["model_executable_calls"] = 15.0
    with pytest.raises(ValueError):
        evidence.validate_files(root, record)
