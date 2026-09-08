"""Composed worker/JSON/original-array/controller tests; no hardware claims."""

import copy
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import jax
import ml_dtypes
import numpy as np
import pytest

from scripts.greenfield import prefill_moe_scaling as spec
from scripts.greenfield import prefill_moe_scaling_evidence as evidence
from scripts.greenfield import prefill_moe_scaling_worker as worker
from scripts.greenfield import probe_ws32_prefill_moe as probe
from scripts.greenfield import ws32_prefill_moe_campaign as campaign
from tests.greenfield.hlo.test_prefill_moe_scaling import rows_hlo
from tests.greenfield.hlo.test_prefill_real_moe_admission import valid_workers


class Tensor:
    def __init__(self, data):
        self.data = data

    def numpy(self):
        return self.data


def test_worker_to_original_files_and_fleet_controller(tmp_path, monkeypatch):
    # Fake model outputs/compiled analyses isolate protocol composition. Actual
    # grouping metadata, numerical comparisons and raw-HLO replay run on CPU.
    from jax.experimental import multihost_utils
    from jax.sharding import Mesh
    from glm_tpu.greenfield.benchmarking import ws32_one_layer
    from scripts.greenfield import run_real_one_layer_ws32 as real

    monkeypatch.setattr(real, "_bfloat16_numpy", lambda t: t.numpy())
    monkeypatch.setattr(ws32_one_layer, "ws32_one_layer_inputs", lambda *v: v[:3])
    monkeypatch.setattr(multihost_utils, "process_allgather", lambda x: np.repeat(x, 8))
    monkeypatch.setattr(
        worker,
        "_compiled_memory",
        lambda _: dict(
            argument_size_in_bytes=1000,
            output_size_in_bytes=1000,
            temp_size_in_bytes=1000,
        ),
    )
    monkeypatch.setattr(
        worker,
        "_memory_stats",
        lambda _: dict(peak_bytes_in_use=4000, bytes_limit=33014413312),
    )
    metadata = jax.jit(spec.device_active_tiles)
    mesh = Mesh(
        np.asarray(jax.devices()[:1], object).reshape(1, 1), ("expert", "feature")
    )
    clock_tick = [0.0]
    monkeypatch.setattr(
        worker.time,
        "monotonic",
        lambda: clock_tick.__setitem__(0, clock_tick[0] + 0.001) or clock_tick[0],
    )
    # measure_completed_calls binds its default clock at definition, use an
    # explicit deterministic timer without changing the real helper's schedule.
    measure = spec.measure_completed_calls
    monkeypatch.setattr(
        spec,
        "measure_completed_calls",
        lambda *a, **kw: measure(*a, **kw, clock=worker.time.monotonic),
    )
    oracle = {"hidden_states": Tensor(np.ones((1, 6144), ml_dtypes.bfloat16))}
    for case in probe.CASES:
        oracle[f"{case}_route_indices"] = Tensor(
            np.arange(128, 136, dtype=np.int32)[None]
        )
        oracle[f"{case}_route_weights"] = Tensor(np.ones((1, 8), np.float32) / 8)
        oracle[f"{case}_output"] = Tensor(np.ones((1, 6144), ml_dtypes.bfloat16))
    fixtures = {
        case: probe.case_rows(
            oracle["hidden_states"].numpy(),
            oracle[f"{case}_route_indices"].numpy(),
            oracle[f"{case}_route_weights"].numpy(),
            case,
            rows=128,
        )
        for case in probe.CASES
    }
    legacy = {case: oracle[f"{case}_output"].numpy() for case in probe.CASES}
    records = []
    for rank, initial in enumerate(valid_workers()):
        devices = [SimpleNamespace(id=i) for i in range(rank * 4, rank * 4 + 4)]

        def array(rows, healthy=False):
            data = np.ones(
                (1, 1) if healthy else (rows, 1536),
                bool if healthy else ml_dtypes.bfloat16,
            )
            return SimpleNamespace(
                addressable_shards=[
                    SimpleNamespace(device=d, data=data.copy()) for d in devices
                ]
            )

        class Program:
            def __init__(self, scalar=False):
                self.scalar = scalar

            def lower(self, *values):
                rows = values[0].shape[0]
                hlo = rows_hlo(rows)
                model = self

                class Compiled:
                    def as_text(self):
                        return hlo

                    def __call__(self, *v):
                        result = array(v[0].shape[0])
                        return (
                            result if model.scalar else (result, array(1, healthy=True))
                        )

                return SimpleNamespace(compiler_ir=lambda **_: hlo, compile=Compiled)

        monkeypatch.setattr(
            worker, "build_mapped", lambda *a, **kw: (Program(), Program(scalar=True))
        )
        fake_jax = SimpleNamespace(
            device_put=lambda v, _: v,
            block_until_ready=lambda x: x,
            local_devices=lambda: devices,
            jit=lambda *a, **kw: metadata,
        )
        root = tmp_path / f"rank{rank}"
        root.mkdir()
        record = {
            k: v
            for k, v in initial.items()
            if k not in ("hlo", "compiled_memory_estimate", "cases")
        }
        record.update(
            status="RUNNING",
            protocol=spec.PROTOCOL,
            admission_only=False,
            boundary_diagnostic=False,
            bounded_admission=False,
            scaling_baseline=True,
            fp32_route_sum=True,
            rows=128,
            iterations=50,
            phases={},
            cases={},
        )
        worker.run_scaling(
            SimpleNamespace(output_dir=root, process_id=rank),
            record,
            jax=fake_jax,
            mesh=mesh,
            physical_mesh=SimpleNamespace(flattened_device_ids=list(range(32))),
            loaded=SimpleNamespace(arrays={}),
            oracle=oracle,
        )
        parsed = json.loads((root / "runner.json").read_text())
        evidence.validate_files(root, parsed, fixtures=fixtures, legacy=legacy)
        records.append(parsed)
    result = evidence.aggregate(records, "b" * 40)
    evidence.validate_record(json.loads(json.dumps(result)), "b" * 40)
    assert result["latency"] is None and result["baseline_only"]
    # Execute the ACTUAL shell-embedded accounting consumer with this producer
    # record and an isolated new SQLite DB; never write the project results DB.
    wrapper = Path("scripts/greenfield/run_fp8_matmul_microbench.sh").read_text()
    accounting = next(
        block.split("\nPY\n", 1)[0]
        for block in wrapper.split("<<'PY'\n")[1:]
        if "run_dir, pin, db_path, repo, elapsed, expected_kernel" in block
    )
    (tmp_path / "runner.json").write_text(json.dumps(result))
    db = tmp_path / "test-accounting.db"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "accounting",
            str(tmp_path),
            "b" * 40,
            str(db),
            str(Path.cwd()),
            "1",
            spec.KERNEL,
        ],
    )
    exec(
        compile(accounting, "<actual-wrapper-accounting>", "exec"),
        {"__name__": "__main__"},
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert (
        "equal128" in summary["claim_scope"] and summary["performance_claim"] is False
    )
    with sqlite3.connect(db) as connection:
        assert connection.execute("select latency_ms from items").fetchall() == [
            (None,)
        ]
    for case in probe.CASES:
        a, b = (result["phase_baseline"][case][n] for n in ("b16", "b128"))
        assert a["calls_per_sample"] == 8 and b["calls_per_sample"] == 1
        assert a["p50_ms_per_128_rows"] == pytest.approx(8 * b["p50_ms_per_128_rows"])
    for mutate in (
        lambda r: r[0].update(iterations=0),
        lambda r: r[0].update(performance_claim=True),
        lambda r: r[0]["local_device_slots"][0].update(device_id=3),
        lambda r: r[0]["programs"]["b16"].update(rows=17),
        lambda r: r[0]["cases"]["normal"]["timing"]["b16"].update(
            postcheck_passed=False
        ),
        lambda r: r[0]["cases"]["normal"]["timing"].update(
            b16=r[0]["cases"]["normal"]["timing"]["b128"]
        ),
    ):
        bad = copy.deepcopy(records)
        mutate(bad)
        with pytest.raises(ValueError):
            evidence.aggregate(bad, "b" * 40)
    # Original-array replay, not a passing JSON label.
    target = tmp_path / "rank0/normal.npz"
    with np.load(target) as values:
        changed = {k: values[k].copy() for k in values.files}
    changed["control_health_0"][3, 0, 0] = False
    np.savez_compressed(target, **changed)
    with pytest.raises(ValueError, match="health evidence"):
        evidence.validate_files(
            tmp_path / "rank0", records[0], fixtures=fixtures, legacy=legacy
        )
    changed["control_health_0"][3, 0, 0] = True
    changed["control_0"][93] = np.float32(2).astype(ml_dtypes.bfloat16).view(np.uint16)
    np.savez_compressed(target, **changed)
    records[0]["cases"]["normal"]["shards"][0]["sha256"]["control"] = sha256(
        changed["control_0"].tobytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="numerical comparison"):
        evidence.validate_files(
            tmp_path / "rank0", records[0], fixtures=fixtures, legacy=legacy
        )


def test_scaling_controller_files_and_worker_cli(monkeypatch, tmp_path):
    tag = "greenfield_fp8_ws32_prefill_moe_scaling_baseline_test"
    assert campaign.is_scaling(tag)
    assert campaign.evidence_files(tag) == evidence.FILES
    assert not campaign.is_bounded(tag) and not campaign.is_boundary(tag)
    # Exercise actual argument parser and tag selection BEFORE runtime dispatch.
    monkeypatch.setattr(probe, "_git_head", lambda: "b" * 40)
    monkeypatch.setenv("GLM_GREENFIELD_RUN_TAG", tag)
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe",
            "--expected-code-hash",
            "b" * 40,
            "--coordinator-address",
            "127.0.0.1:8476",
            "--process-id",
            "0",
            "--output-dir",
            str(tmp_path),
            "--scaling-baseline",
        ],
    )
    with pytest.raises(ValueError, match="output path"):
        probe.main()


def test_wrapper_scoped_baseline_does_not_publish_ambiguous_latency():
    source = Path("scripts/greenfield/run_fp8_matmul_microbench.sh").read_text()
    assert "latency_ms=None if untimed or scaling" in source
    assert (
        "real_layer3_equal128_b16_b128_phase_baseline_v1_normal_concentrated" in source
    )
    assert "DEFAULT_WARMUP=10\n  DEFAULT_ITERATIONS=50" in source
    assert (
        "from scripts.greenfield.prefill_moe_scaling_evidence import validate_record"
        in source
    )
