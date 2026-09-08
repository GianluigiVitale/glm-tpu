"""Original-array/fleet replay for the scoped equal-work MoE phase baseline."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import ml_dtypes
import numpy as np

from scripts.greenfield import prefill_moe_scaling as spec
from scripts.greenfield import probe_ws32_prefill_moe as probe

PROGRAM_ROWS = {"b16": 16, "b128": 128, "scalar": 1}
FILES = ("runner.json", "worker.log", "normal.npz", "concentrated.npz") + tuple(
    f"{name}.{suffix}"
    for name in PROGRAM_ROWS
    for suffix in ("stablehlo.mlir", "optimized_hlo.txt")
)


def load_fixtures() -> tuple[dict[str, tuple[np.ndarray, ...]], dict[str, np.ndarray]]:
    """Small authenticated oracle only, no model payload or TPU initialization."""
    from scripts.greenfield.run_real_one_layer_ws32 import _load_oracle, _bfloat16_numpy

    _, oracle = _load_oracle(
        probe.ORACLE,
        expected_manifest_sha256=probe.ORACLE_SHA,
        pack_manifest=json.loads((probe.PACK / "manifest.json").read_text()),
    )
    return (
        {
            case: probe.case_rows(
                _bfloat16_numpy(oracle["hidden_states"]),
                oracle[f"{case}_route_indices"].numpy(),
                oracle[f"{case}_route_weights"].numpy(),
                case,
                rows=128,
            )
            for case in probe.CASES
        },
        {case: _bfloat16_numpy(oracle[f"{case}_output"]) for case in probe.CASES},
    )


def validate_files(
    root: Path,
    record: dict[str, Any],
    *,
    fixtures: dict[str, tuple[np.ndarray, ...]],
    legacy: dict[str, np.ndarray],
) -> None:
    if set(record["programs"]) != set(PROGRAM_ROWS):
        raise ValueError("baseline program inventory differs")
    for name, rows in PROGRAM_ROWS.items():
        facts = record["programs"][name]
        for suffix, key in (
            ("stablehlo.mlir", "stablehlo_sha256"),
            ("optimized_hlo.txt", "sha256"),
        ):
            data = (root / f"{name}.{suffix}").read_bytes()
            if sha256(data).hexdigest() != facts[key]:
                raise ValueError("baseline graph bytes differ")
        if rows != 1:
            actual = probe.check_hlo(
                (root / f"{name}.optimized_hlo.txt").read_text(),
                fp32_route_sum=True,
                rows=rows,
            )
            if (
                not actual["passed"]
                or json.loads(json.dumps(actual)) != facts["contract"]
            ):
                raise ValueError("baseline actual HLO contract differs")
    for case in probe.CASES:
        c = record["cases"][case]
        with np.load(root / f"{case}.npz", allow_pickle=False) as arrays:
            expected_keys = {"hidden", "routes", "weights", "active_tiles"} | {
                f"{kind}_{s['device_id']}"
                for s in c["shards"]
                for kind in (
                    "wide",
                    "control",
                    "scalar",
                    "wide_health",
                    "control_health",
                )
            }
            if set(arrays.files) != expected_keys:
                raise ValueError("baseline original array inventory differs")
            for i, name in enumerate(("hidden", "routes", "weights")):
                expected = fixtures[case][i]
                if i == 0:
                    expected = expected.view(np.uint16)
                actual = arrays[name]
                if actual.dtype != expected.dtype or not np.array_equal(
                    actual, expected
                ):
                    raise ValueError(
                        "baseline supplied input differs from authenticated fixture"
                    )
                if sha256(actual.tobytes()).hexdigest() != c["input_sha256"][i]:
                    raise ValueError("baseline input SHA differs")
            tiles = arrays["active_tiles"]
            if tiles.shape != (9, 8):
                raise ValueError("baseline active metadata inventory differs")
            ids = arrays["routes"]
            occ = [
                spec.occupancy(value, tiles[i])
                for i, value in enumerate(
                    (ids, *[ids[start : start + 16] for start in range(0, 128, 16)])
                )
            ]
            if c["occupancy"] != dict(b128=occ[0], b16=occ[1:]):
                raise ValueError("baseline active-tile accounting differs")
            for shard in c["shards"]:
                for name, shape in (
                    ("wide_health", (1, 1)),
                    ("control_health", (8, 1, 1)),
                ):
                    health = arrays[f"{name}_{shard['device_id']}"]
                    if (
                        health.dtype != np.bool_
                        or health.shape != shape
                        or sha256(health.tobytes()).hexdigest() != shard["sha256"][name]
                        or not health.all()
                        or shard["healthy"] is not True
                    ):
                        raise ValueError(
                            "baseline original health evidence differs or fails"
                        )
                outputs = []
                for name in ("wide", "control", "scalar"):
                    value = arrays[f"{name}_{shard['device_id']}"]
                    if (
                        value.dtype != np.uint16
                        or value.shape != (128, 1536)
                        or sha256(value.tobytes()).hexdigest() != shard["sha256"][name]
                    ):
                        raise ValueError("baseline original BF16 output differs")
                    outputs.append(value.view(ml_dtypes.bfloat16))
                feature = shard["device_slot"] % 4
                comparison = spec.compare_equal_work(
                    *outputs, legacy[case][:, feature * 1536 : (feature + 1) * 1536]
                )
                if (
                    comparison != shard["bounded_comparison"]
                    or not comparison["passed"]
                ):
                    raise ValueError(
                        "baseline original numerical comparison differs or fails"
                    )


def validate_workers(records: list[dict[str, Any]], pin: str) -> dict[str, Any]:
    if (
        len(records) != 8
        or any(
            {r[key] for r in records} != set(range(8))
            for key in ("launch_rank", "jax_process_index")
        )
        or len({r["hostname"] for r in records}) != 8
    ):
        raise ValueError("baseline needs eight unique physical hosts/processes")
    expected = dict(
        status="SUCCESS",
        protocol=spec.PROTOCOL,
        code_hash=pin,
        admission_only=False,
        boundary_diagnostic=False,
        bounded_admission=False,
        scaling_baseline=True,
        fp32_route_sum=True,
        performance_claim=False,
        rows=128,
        iterations=spec.ITERATIONS,
        latency=None,
        measured_phase_budget_seconds=spec.PHASE_BUDGET_SECONDS,
        packed_manifest_sha256=probe.PACK_SHA,
        oracle_manifest_sha256=probe.ORACLE_SHA,
        mesh_sha256=probe.MESH_SHA,
        topology_sha256=probe.TOPOLOGY_SHA,
        topology_fleet_sha256=probe.FLEET_SHA,
    )
    owners = []
    common_mesh = records[0]["physical_device_ids"]
    for r in records:
        if any(
            type(r.get(k)) is not type(v) or r.get(k) != v for k, v in expected.items()
        ):
            raise ValueError("baseline worker protocol/provenance differs")
        if (
            r["physical_device_ids"] != common_mesh
            or r["hlo"] != r["programs"]["b128"]
            or set(r["cases"]) != set(probe.CASES)
        ):
            raise ValueError("baseline mesh/primary graph/case inventory differs")
        if (
            not all(type(r[k]) is int and r[k] > 0 for k in ("pid", "start_ticks"))
            or not r["boot_id"]
        ):
            raise ValueError("baseline process identity missing")
        physical = np.asarray(common_mesh).reshape(-1).tolist()
        slots = r["local_device_slots"]
        if len(physical) != 32 or len(set(physical)) != 32 or len(slots) != 4:
            raise ValueError("baseline physical owner inventory differs")
        own = {s["device_slot"]: s["device_id"] for s in slots}
        if (
            len(own) != 4
            or any(
                type(s) is not int or not 0 <= s < 32 or physical[s] != d
                for s, d in own.items()
            )
            or any(not re.fullmatch("[0-9a-f]{64}", s["file_sha256"]) for s in slots)
        ):
            raise ValueError("baseline loaded shard ownership differs")
        owners.extend(own)
        stats = r["device_memory_stats_including_reference"]
        if (
            len(stats) != 4
            or {s["device_id"] for s in stats} != set(own.values())
            or any(
                not 0 < s["stats"]["peak_bytes_in_use"] < s["stats"]["bytes_limit"]
                for s in stats
            )
        ):
            raise ValueError("baseline measured HBM coverage/margin differs")
        if set(r["programs"]) != set(PROGRAM_ROWS):
            raise ValueError("baseline executable inventory differs")
        for name, rows in PROGRAM_ROWS.items():
            f = r["programs"][name]
            memory = f["compiled_memory_estimate"]
            sizes = [
                memory[k]
                for k in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "temp_size_in_bytes",
                )
            ]
            if (
                f["rows"] != rows
                or any(type(n) is not int or n < 0 for n in sizes)
                or sum(sizes) > spec.COMPILED_MEMORY_LIMIT_BYTES
            ):
                raise ValueError("baseline compiled memory/rows differ")
            if rows != 1 and f["contract"]["passed"] is not True:
                raise ValueError("baseline HLO contract failed")
        for c in r["cases"].values():
            if (
                c["passed"] is not True
                or c["fleet_passed"] is not True
                or len(c["shards"]) != 4
                or {s["device_slot"]: s["device_id"] for s in c["shards"]} != own
                or any(
                    s["healthy"] is not True
                    or s["bounded_comparison"]["passed"] is not True
                    for s in c["shards"]
                )
            ):
                raise ValueError("baseline owner numerical/health evidence failed")
            if set(c["timing"]) != {"b16", "b128"} or any(
                t.get("postcheck_passed") is not True for t in c["timing"].values()
            ):
                raise ValueError("baseline timed-output postcheck missing")
    if len(owners) != 32 or set(owners) != set(range(32)):
        raise ValueError("baseline does not cover32 unique owners")
    for name in PROGRAM_ROWS:
        if (
            len({r["programs"][name]["sha256"] for r in records}) != 1
            or len({r["programs"][name]["stablehlo_sha256"] for r in records}) != 1
        ):
            raise ValueError("baseline fleet HLO differs")
    result = {}
    for case in probe.CASES:
        if (
            len({tuple(r["cases"][case]["input_sha256"]) for r in records}) != 1
            or len(
                {
                    json.dumps(r["cases"][case]["occupancy"], sort_keys=True)
                    for r in records
                }
            )
            != 1
        ):
            raise ValueError("baseline fleet inputs/occupancy differ")
        for feature in range(4):
            for kind in ("wide", "control", "scalar"):
                if (
                    len(
                        {
                            s["sha256"][kind]
                            for r in records
                            for s in r["cases"][case]["shards"]
                            if s["device_slot"] % 4 == feature
                        }
                    )
                    != 1
                ):
                    raise ValueError("baseline feature replicas differ")
        result[case] = {
            name: spec.fleet_timing([r["cases"][case]["timing"][name] for r in records])
            for name in ("b16", "b128")
        }
        if (
            result[case]["b16"]["calls_per_sample"] != 8
            or result[case]["b128"]["calls_per_sample"] != 1
        ):
            raise ValueError("baseline control/window timing geometries swapped")
    return result


def aggregate(records: list[dict[str, Any]], pin: str) -> dict[str, Any]:
    timings = validate_workers(records, pin)
    return dict(
        status="SUCCESS",
        code_hash=pin,
        kernel=spec.KERNEL,
        protocol=spec.PROTOCOL,
        admission_only=False,
        boundary_diagnostic=False,
        bounded_admission=False,
        scaling_baseline=True,
        fp32_route_sum=True,
        baseline_only=True,
        diagnostic_only=False,
        performance_claim=False,
        latency=None,
        profiler_free_timing=True,
        warmup=spec.WARMUP,
        iterations=spec.ITERATIONS,
        selected_route_case=None,
        device_kind="TPU v4",
        workers=records,
        phase_baseline=timings,
        hlo={"sha256": records[0]["hlo"]["sha256"], "contract": {"passed": True}},
        comparison={"passed": True},
        checksum=sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
        claim_scope="MOE_PHASE_BASELINE_NOT_MODEL_PREFILL_OR_TTFT",
    )


def validate_record(record: dict[str, Any], pin: str) -> None:
    if record != aggregate(record["workers"], pin):
        raise ValueError("baseline aggregate differs from original fleet records")
