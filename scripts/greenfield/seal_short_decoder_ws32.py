#!/usr/bin/env python3
"""Independent fleet validator and DB publisher for WS32 Gate-D evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import sys
from typing import Any, Mapping

import ml_dtypes
import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_DSA_ASSOCIATION_SUMMARY_SHA256 = (
    "661142816aa64ec8d085553b427e99f62ab3f1f16b3fc87fc4fc24880d467203"
)
_DSA_ASSOCIATION_SUCCESS_SHA256 = (
    "79aba79e24026bc4c1d17aed2ca92055530b8a551ed6e1279a25300d2cb0f52b"
)

from glm_tpu.greenfield.benchmarking import (  # noqa: E402
    validate_ws32_decoder_hlo,
    validate_ws32_exact_dsa_materializer_hlo,
    validate_ws32_topology_fleet,
)
from glm_tpu.greenfield.sharding.ws32 import (  # noqa: E402
    build_ws32_physical_mesh,
)
from glm_tpu.greenfield.validation import (  # noqa: E402
    compare_ws32_dsa_step,
    compare_ws32_raw_tokens,
    load_ws32_short_context_oracle,
    validate_ws32_cache_probe,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--run-dir", required=True, type=Path)
    validate.add_argument("--topology-capture-root", required=True, type=Path)
    validate.add_argument("--token-oracle-dir", required=True, type=Path)
    validate.add_argument("--dsa-oracle-dir", required=True, type=Path)
    validate.add_argument("--mode", choices=("acquire", "numerical"), required=True)
    validate.add_argument("--context-label", choices=("2k", "8k"), required=True)
    validate.add_argument("--tag", required=True)
    validate.add_argument("--code-hash", required=True)
    validate.add_argument("--checkpoint-manifest-sha256", required=True)
    validate.add_argument("--checkpoint-success-sha256", required=True)
    validate.add_argument("--token-oracle-manifest-sha256", required=True)
    validate.add_argument("--dsa-oracle-manifest-sha256", required=True)
    validate.add_argument("--token-oracle-success-sha256", required=True)
    validate.add_argument("--dsa-oracle-success-sha256", required=True)
    validate.add_argument("--dsa-association-summary-sha256", required=True)
    validate.add_argument("--dsa-association-success-sha256", required=True)
    validate.add_argument("--topology-sha256", required=True)
    validate.add_argument("--topology-fleet-sha256", required=True)
    validate.add_argument("--mesh-sha256", required=True)
    validate.add_argument("--source-inventory-sha256", required=True)
    validate.add_argument("--context-capacity", required=True, type=int)
    validate.add_argument("--observer-steps", required=True, type=int)
    validate.add_argument("--warmup", required=True, type=int)
    validate.add_argument("--iterations", required=True, type=int)
    validate.add_argument("--trace-steps", required=True, type=int)
    validate.add_argument("--exact-dsa", choices=(0, 1), required=True, type=int)
    for graph in (
        "exact-materialize",
        "exact-promote",
        "prefill",
        "observer",
        "decode",
        "cache-probe",
    ):
        validate.add_argument(
            f"--expected-{graph}-stablehlo-sha256", required=True
        )
        validate.add_argument(
            f"--expected-{graph}-optimized-hlo-sha256", required=True
        )
    validate.add_argument("--output", required=True, type=Path)

    publish = sub.add_parser("publish-db")
    publish.add_argument("--summary", required=True, type=Path)
    publish.add_argument("--results-db", required=True, type=Path)
    publish.add_argument("--snapshot", required=True, type=Path)
    publish.add_argument("--output", required=True, type=Path)
    rollback = sub.add_parser("rollback-db")
    rollback.add_argument("--summary", required=True, type=Path)
    rollback.add_argument("--db-link", required=True, type=Path)
    rollback.add_argument("--results-db", required=True, type=Path)
    return parser.parse_args()


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_bytes(
        json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    temporary.replace(path)


def _same(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _same(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _digest_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _memory_valid(value: Any) -> bool:
    keys = {
        "bytes_in_use",
        "bytes_limit",
        "bytes_reservable_limit",
        "bytes_reserved",
        "largest_alloc_size",
        "largest_free_block_bytes",
        "num_allocs",
        "peak_bytes_in_use",
        "peak_bytes_reserved",
    }
    return bool(
        value is None
        or (
            type(value) is dict
            and set(value) == keys
            and all(type(number) is int and number >= 0 for number in value.values())
            and value["bytes_limit"] == 33_014_398_976
            and value["bytes_in_use"] <= value["peak_bytes_in_use"]
            <= value["bytes_limit"]
            and value["bytes_reserved"] <= value["peak_bytes_reserved"]
            <= value["bytes_reservable_limit"]
            <= value["bytes_limit"]
        )
    )


def _validate_run_tag(tag: str, *, context_label: str, mode: str) -> None:
    pattern = (
        rf"greenfield_ws32_short_decoder_{re.escape(context_label)}_"
        rf"{re.escape(mode)}_[0-9]{{8}}T[0-9]{{15}}Z"
    )
    if re.fullmatch(pattern, tag) is None:
        raise SystemExit("WS32 run tag contradicts active context/mode")


def _graph_valid(value: Any, *, mode: str) -> bool:
    if type(value) is dict and value.get("kind") in {
        "exact_materialize",
        "exact_promote",
    }:
        keys = {
            "collective_count",
            "instruction_count",
            "kind",
            "live_instruction_count",
            "maximum_group_size",
            "optimized_hlo_sha256",
            "passed",
            "stablehlo_sha256",
            "violations",
        }
        expected_violations = (
            []
            if mode == "numerical"
            else ["StableHLO identity drifted", "optimized HLO identity drifted"]
        )
        return bool(
            set(value) == keys
            and value["passed"] is (mode == "numerical")
            and value["violations"] == expected_violations
            and type(value["collective_count"]) is int
            and value["maximum_group_size"] <= 8
            and (
                (
                    value["kind"] == "exact_materialize"
                    and value["collective_count"] > 0
                )
                or (
                    value["kind"] == "exact_promote"
                    and value["collective_count"] == 0
                )
            )
        )
    keys = {
        "all_gather_count",
        "all_reduce_count",
        "async_collective_count",
        "collective_count",
        "expert_collective_count",
        "feature_collective_count",
        "fused_rmsnorm_collective_count",
        "forbidden_full_hidden_values",
        "instruction_count",
        "kind",
        "live_collective_count",
        "live_instruction_count",
        "maximum_group_size",
        "optimized_hlo_sha256",
        "passed",
        "rounded_first_rmsnorm_collective_count",
        "stablehlo_sha256",
        "violations",
    }
    expected_violations = (
        []
        if mode == "numerical"
        else ["StableHLO identity drifted", "optimized HLO identity drifted"]
    )
    return bool(
        type(value) is dict
        and set(value) == keys
        and value["passed"] is (mode == "numerical")
        and value["violations"] == expected_violations
        and value["collective_count"] == value["live_collective_count"]
        and type(value["collective_count"]) is int
        and value["collective_count"] > 0
        and value["async_collective_count"] == 0
        and value["maximum_group_size"] <= 8
        and value["forbidden_full_hidden_values"] == []
        and (
            (
                value["kind"] == "cache_probe"
                and value["feature_collective_count"] == 0
                and value["expert_collective_count"] > 0
                and value["fused_rmsnorm_collective_count"] == 0
                and value["rounded_first_rmsnorm_collective_count"] == 0
            )
            or (
                value["kind"] in {"decode", "observer", "prefill"}
                and value["feature_collective_count"] > 0
                and value["expert_collective_count"] > 0
                and value["fused_rmsnorm_collective_count"] == 157
                and value["rounded_first_rmsnorm_collective_count"] == 0
            )
        )
        and value["all_reduce_count"] + value["all_gather_count"]
        == value["collective_count"]
    )


def _peak(records: list[Mapping[str, Any]]) -> tuple[int, int]:
    peaks = []
    headrooms = []
    for record in records:
        for stats in record["device_memory_after_execute"]:
            if stats is None:
                continue
            peaks.append(stats["peak_bytes_in_use"])
            headrooms.append(stats["bytes_limit"] - stats["peak_bytes_in_use"])
    if len(peaks) != 32 or len(headrooms) != 32:
        raise ValueError("WS32 fleet lacks exact 32-chip HBM telemetry")
    return max(peaks), min(headrooms)


def _distribution(samples: list[float]) -> dict[str, float | int]:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not values.size:
        raise ValueError("WS32 timing samples are empty or non-vector")
    return {
        "count": int(values.size),
        "maximum_ms": float(values.max()),
        "mean_ms": float(values.mean()),
        "minimum_ms": float(values.min()),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def _validate(args: argparse.Namespace) -> int:
    _validate_run_tag(
        args.tag, context_label=args.context_label, mode=args.mode
    )
    materializer_pin_names = {
        "expected_exact_materialize_stablehlo_sha256",
        "expected_exact_materialize_optimized_hlo_sha256",
        "expected_exact_promote_stablehlo_sha256",
        "expected_exact_promote_optimized_hlo_sha256",
    }
    hlo_pins = [
        value
        for name, value in vars(args).items()
        if name.startswith("expected_")
        and "hlo_sha256" in name
        and (args.exact_dsa or name not in materializer_pin_names)
    ]
    inactive_pins = [
        value
        for name, value in vars(args).items()
        if name in materializer_pin_names and not args.exact_dsa
    ]
    if any(value != "0" * 64 for value in inactive_pins):
        raise SystemExit("default WS32 sealer requires vacant materializer pins")
    if args.mode == "acquire" and any(value != "0" * 64 for value in hlo_pins):
        raise SystemExit("WS32 acquisition sealer requires vacant active HLO pins")
    if args.mode == "numerical" and any(value == "0" * 64 for value in hlo_pins):
        raise SystemExit("WS32 numerical sealer requires acquired active HLO pins")
    association_pins = (
        args.dsa_association_summary_sha256,
        args.dsa_association_success_sha256,
    )
    if args.exact_dsa:
        if association_pins != (
            _DSA_ASSOCIATION_SUMMARY_SHA256,
            _DSA_ASSOCIATION_SUCCESS_SHA256,
        ):
            raise SystemExit("WS32 exact DSA association evidence pin drifted")
        source_summary = args.run_dir / "exact_dsa_source_summary.json"
        source_success = args.run_dir / "exact_dsa_source_SUCCESS"
        if (
            _digest_file(source_summary) != _DSA_ASSOCIATION_SUMMARY_SHA256
            or _digest_file(source_success) != _DSA_ASSOCIATION_SUCCESS_SHA256
        ):
            raise SystemExit("WS32 exact DSA source evidence bytes drifted")
        association = json.loads(source_summary.read_text(encoding="utf-8"))
        if (
            association.get("status") != "SUCCESS"
            or association.get("candidate_mechanisms_proven") is not True
            or association.get("association_restored") is not False
            or association.get("exact_arms") != ["tuple4"]
            or association.get("performance_claim") is not False
        ):
            raise SystemExit("WS32 exact DSA source classification drifted")
    elif association_pins != ("0" * 64, "0" * 64):
        raise SystemExit("default WS32 path must not claim DSA association evidence")
    runner_paths = sorted(args.run_dir.glob("fleet/runner.rank*.json"))
    if len(runner_paths) != 8:
        raise SystemExit(f"expected eight WS32 runner records, got {len(runner_paths)}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in runner_paths]
    captures = tuple(
        json.loads(
            (
                args.topology_capture_root / f"topology.rank{rank}.json"
            ).read_text(encoding="utf-8")
        )
        for rank in range(8)
    )
    topology, ordered_captures, fleet_hash = validate_ws32_topology_fleet(
        captures,
        expected_topology_sha256=args.topology_sha256,
        expected_fleet_sha256=args.topology_fleet_sha256,
        slice_name="db-v4-64-od",
    )
    physical_mesh = build_ws32_physical_mesh(topology)
    if physical_mesh.mesh_hash != args.mesh_sha256 or fleet_hash != args.topology_fleet_sha256:
        raise SystemExit("WS32 topology/mesh identity drifted")
    oracle = load_ws32_short_context_oracle(
        args.token_oracle_dir,
        args.dsa_oracle_dir,
        expected_token_manifest_sha256=args.token_oracle_manifest_sha256,
        expected_dsa_manifest_sha256=args.dsa_oracle_manifest_sha256,
        expected_token_success_sha256=args.token_oracle_success_sha256,
        expected_dsa_success_sha256=args.dsa_oracle_success_sha256,
    )

    common = {
        "checkpoint_manifest_sha256": args.checkpoint_manifest_sha256,
        "checkpoint_success_sha256": args.checkpoint_success_sha256,
        "code_hash": args.code_hash,
        "compile_only": args.mode == "acquire",
        "context_capacity": args.context_capacity,
        "dsa_oracle_manifest_sha256": args.dsa_oracle_manifest_sha256,
        "dsa_oracle_success_sha256": args.dsa_oracle_success_sha256,
        "dsa_association_summary_sha256": (
            args.dsa_association_summary_sha256
        ),
        "dsa_association_success_sha256": (
            args.dsa_association_success_sha256
        ),
        "exact_dsa": bool(args.exact_dsa),
        "mesh_sha256": args.mesh_sha256,
        "source_inventory_sha256": args.source_inventory_sha256,
        "token_oracle_manifest_sha256": args.token_oracle_manifest_sha256,
        "token_oracle_success_sha256": args.token_oracle_success_sha256,
        "topology_fleet_sha256": args.topology_fleet_sha256,
        "topology_sha256": args.topology_sha256,
        "xla_python_client_mem_fraction": ".95",
    }
    first_graphs = records[0].get("graphs")
    expected_graphs = {
        "cache_probe",
        "decode",
        "observer",
        "prefill",
    }
    if args.exact_dsa:
        expected_graphs.update({"exact_materialize", "exact_promote"})
    if type(first_graphs) is not dict or set(first_graphs) != expected_graphs:
        raise SystemExit("WS32 graph set drifted")
    slots: set[int] = set()
    all_samples: list[list[float]] = []
    expected_prompt_length = {"2k": 2034, "8k": 8155}[args.context_label]
    pre_keys = {
        "artifact_kind",
        "checkpoint_manifest_sha256",
        "checkpoint_success_sha256",
        "checkpoint_verified_device_slots",
        "code_hash",
        "compile_only",
        "compile_seconds",
        "compiled_memory_analysis",
        "context_capacity",
        "device_memory_after_compile",
        "device_memory_after_load",
        "device_memory_before_load",
        "dsa_oracle_manifest_sha256",
        "dsa_oracle_success_sha256",
        "dsa_association_summary_sha256",
        "dsa_association_success_sha256",
        "exact_dsa",
        "graphs",
        "hostname",
        "jax_process_index",
        "launch_process_id",
        "load_seconds",
        "local_device_slots",
        "mesh_sha256",
        "prompt_length",
        "source_inventory_sha256",
        "token_oracle_manifest_sha256",
        "token_oracle_success_sha256",
        "topology_fleet_sha256",
        "topology_sha256",
        "xla_python_client_mem_fraction",
    }
    numerical_keys = pre_keys | {
        "cache_write_probe",
        "correctness_passed",
        "device_memory_after_execute",
        "dsa_steps",
        "observed_generated_token_ids",
        "numerical_tensors",
        "performance_claim",
        "profiler_free_timing",
        "schema_version",
        "state",
        "status",
        "token_comparison",
        "trace",
    }
    acquisition_keys = pre_keys | {
        "performance_claim",
        "schema_version",
        "status",
    }
    exact_numerical = (
        "cache_write_probe",
        "dsa_steps",
        "observed_generated_token_ids",
        "state",
        "token_comparison",
    )
    for rank, record in enumerate(records):
        capture = ordered_captures[rank]
        if set(record) != (
            numerical_keys if args.mode == "numerical" else acquisition_keys
        ):
            raise SystemExit(f"WS32 runner schema drifted at rank {rank}")
        if any(not _same(record.get(key), value) for key, value in common.items()):
            raise SystemExit(f"WS32 immutable identity drifted at rank {rank}")
        expected_artifact_kind = (
            "greenfield_ws32_short_decoder"
            if args.mode == "numerical"
            else "greenfield_ws32_short_decoder_prevalidation"
        )
        if (
            record.get("artifact_kind") != expected_artifact_kind
            or record.get("schema_version") != 1
            or record.get("launch_process_id") != rank
            or record.get("jax_process_index") != capture["jax_process_index"]
            or record.get("hostname") != capture["hostname"]
            or not _same(record.get("graphs"), first_graphs)
        ):
            raise SystemExit(f"WS32 fleet/process/HLO identity drifted at rank {rank}")
        if record.get("prompt_length") != expected_prompt_length:
            raise SystemExit(f"WS32 prompt length drifted at rank {rank}")
        if (
            type(record.get("load_seconds")) is not float
            or not math.isfinite(record["load_seconds"])
            or record["load_seconds"] <= 0
        ):
            raise SystemExit(f"WS32 load timing drifted at rank {rank}")
        if set(record.get("compile_seconds", {})) != expected_graphs or any(
            type(value) is not float or not math.isfinite(value) or value <= 0
            for value in record["compile_seconds"].values()
        ):
            raise SystemExit(f"WS32 compile timing drifted at rank {rank}")
        memory_fields = {
            "alias_size_in_bytes",
            "argument_size_in_bytes",
            "generated_code_size_in_bytes",
            "output_size_in_bytes",
            "temp_size_in_bytes",
        }
        if set(record.get("compiled_memory_analysis", {})) != expected_graphs or any(
            set(value) != memory_fields
            or any(number is not None and (type(number) is not int or number < 0) for number in value.values())
            for value in record["compiled_memory_analysis"].values()
        ):
            raise SystemExit(f"WS32 compiled-memory schema drifted at rank {rank}")
        if any(not _graph_valid(value, mode=args.mode) for value in record["graphs"].values()):
            raise SystemExit(f"WS32 graph contract failed at rank {rank}")
        for graph, report in record["graphs"].items():
            stable = args.run_dir / "fleet_hlo" / f"{graph}.rank{rank}.stablehlo.mlir"
            optimized = args.run_dir / "fleet_hlo" / f"{graph}.rank{rank}.optimized_hlo.txt"
            stable_text = stable.read_text(encoding="utf-8")
            optimized_text = optimized.read_text(encoding="utf-8")
            if _digest_file(stable) != report["stablehlo_sha256"] or _digest_file(optimized) != report["optimized_hlo_sha256"]:
                raise SystemExit(f"WS32 HLO artifact drifted at rank {rank}/{graph}")
            if graph.startswith("exact_"):
                replay = validate_ws32_exact_dsa_materializer_hlo(
                    stable_text,
                    optimized_text,
                    expected_stablehlo_sha256=report["stablehlo_sha256"],
                    expected_optimized_hlo_sha256=(
                        report["optimized_hlo_sha256"]
                    ),
                    kind=graph,
                ).to_dict()
            else:
                replay = validate_ws32_decoder_hlo(
                    stable_text,
                    optimized_text,
                    expected_stablehlo_sha256=report["stablehlo_sha256"],
                    expected_optimized_hlo_sha256=(
                        report["optimized_hlo_sha256"]
                    ),
                    hidden_size=6144,
                    kind=graph,
                    exact_dsa=bool(args.exact_dsa),
                ).to_dict()
            normalized_record = dict(report)
            if args.mode == "acquire":
                normalized_record["passed"] = True
                normalized_record["violations"] = []
            if not _same(replay, normalized_record):
                raise SystemExit(f"WS32 HLO replay drifted at rank {rank}/{graph}")
            if args.mode == "numerical" and (
                report["stablehlo_sha256"]
                != getattr(args, f"expected_{graph}_stablehlo_sha256")
                or report["optimized_hlo_sha256"]
                != getattr(args, f"expected_{graph}_optimized_hlo_sha256")
            ):
                raise SystemExit(f"WS32 acquired HLO pin drifted at rank {rank}/{graph}")
        local_slots = record.get("local_device_slots")
        if type(local_slots) is not list or len(local_slots) != 4:
            raise SystemExit(f"WS32 local slot cardinality drifted at rank {rank}")
        expected_ids = set(capture["local_device_ids"])
        if {item.get("device_id") for item in local_slots} != expected_ids:
            raise SystemExit(f"WS32 local device ownership drifted at rank {rank}")
        verified_slots = record.get("checkpoint_verified_device_slots")
        if (
            type(verified_slots) is not list
            or len(verified_slots) != 4
            or any(type(slot) is not int for slot in verified_slots)
            or set(verified_slots)
            != {item.get("device_slot") for item in local_slots}
        ):
            raise SystemExit(f"WS32 checkpoint hash coverage drifted at rank {rank}")
        for item in local_slots:
            slot = item.get("device_slot")
            if (
                type(slot) is not int
                or item.get("expert_coordinate") != slot // 4
                or item.get("feature_coordinate") != slot % 4
                or type(item.get("file_sha256")) is not str
                or len(item["file_sha256"]) != 64
            ):
                raise SystemExit(f"WS32 slot record drifted at rank {rank}")
            slots.add(slot)
        for field in ("device_memory_before_load", "device_memory_after_load", "device_memory_after_compile"):
            if type(record.get(field)) is not list or len(record[field]) != 4 or any(not _memory_valid(item) for item in record[field]):
                raise SystemExit(f"WS32 memory record drifted: rank={rank} field={field}")
        if args.mode == "numerical":
            if record.get("status") != "SUCCESS" or record.get("correctness_passed") is not True or record.get("performance_claim") is not False:
                raise SystemExit(f"WS32 numerical terminal status drifted rank {rank}")
            if record.get("token_comparison", {}).get("exact_prefix_match") is not True:
                raise SystemExit(f"WS32 raw tokens drifted rank {rank}")
            observed_token_ids = record.get("observed_generated_token_ids")
            expected_observed_count = (
                1
                + args.observer_steps
                + args.warmup
                + args.iterations
                + args.trace_steps
            )
            if (
                type(observed_token_ids) is not list
                or len(observed_token_ids) != expected_observed_count
                or any(type(token) is not int for token in observed_token_ids)
                or len(observed_token_ids) < oracle.generated_token_ids.size
            ):
                raise SystemExit(f"WS32 raw-token cardinality drifted rank {rank}")
            expected_token_comparison = compare_ws32_raw_tokens(
                observed_token_ids[: oracle.generated_token_ids.size],
                oracle,
            )
            if not _same(record.get("token_comparison"), expected_token_comparison):
                raise SystemExit(f"WS32 token comparison recomputation drifted rank {rank}")
            if rank and any(
                not _same(record[field], records[0][field])
                for field in exact_numerical
            ):
                raise SystemExit(f"WS32 replicated numerical evidence drifted rank {rank}")
            dsa = record.get("dsa_steps")
            if type(dsa) is not list or len(dsa) != args.observer_steps or any(item.get("passed") is not True for item in dsa):
                raise SystemExit(f"WS32 DSA exactness drifted rank {rank}")
            tensor_path = args.run_dir / "fleet" / f"runner.rank{rank}.npz"
            tensor_record = record.get("numerical_tensors")
            if (
                type(tensor_record) is not dict
                or set(tensor_record) != {"arrays", "byte_count", "filename", "sha256"}
                or tensor_record.get("filename") != tensor_path.name
                or tensor_record.get("byte_count") != tensor_path.stat().st_size
                or tensor_record.get("sha256") != _digest_file(tensor_path)
            ):
                raise SystemExit(f"WS32 numerical tensor file drifted rank {rank}")
            expected_array_names = {
                "cache_contract_valid",
                "cache_index_bfloat16_bits",
                "cache_kv_bfloat16_bits",
                "cache_position",
                "dsa_producer_layer_ids",
                "dsa_selected_positions",
                "dsa_selected_scores",
                "dsa_selected_valid_counts",
            }
            with np.load(tensor_path, allow_pickle=False) as archive:
                if set(archive.files) != expected_array_names:
                    raise SystemExit(f"WS32 numerical tensor keys drifted rank {rank}")
                arrays = {
                    name: np.ascontiguousarray(archive[name])
                    for name in sorted(archive.files)
                }
            array_records = {}
            for name, value in arrays.items():
                array_records[name] = {
                    "dtype": value.dtype.name,
                    "sha256": sha256(value.view(np.uint8).tobytes()).hexdigest(),
                    "shape": list(value.shape),
                }
            if not _same(tensor_record.get("arrays"), array_records):
                raise SystemExit(f"WS32 numerical tensor manifest drifted rank {rank}")
            if rank and not _same(
                array_records, records[0]["numerical_tensors"]["arrays"]
            ):
                raise SystemExit(f"WS32 numerical tensor values disagree rank {rank}")
            expected_dsa = []
            for step in range(args.observer_steps):
                expected_dsa.append(
                    compare_ws32_dsa_step(
                        producer_layer_ids=arrays["dsa_producer_layer_ids"],
                        selected_positions=arrays["dsa_selected_positions"][step],
                        selected_valid_counts=arrays[
                            "dsa_selected_valid_counts"
                        ][step],
                        selected_scores=arrays["dsa_selected_scores"][step],
                        oracle=oracle,
                        step=step,
                    )
                )
            if not _same(dsa, expected_dsa):
                raise SystemExit(f"WS32 DSA recomputation drifted rank {rank}")
            expected_cache = validate_ws32_cache_probe(
                position=arrays["cache_position"],
                kv_rows=arrays["cache_kv_bfloat16_bits"].view(
                    ml_dtypes.bfloat16
                ),
                index_rows=arrays["cache_index_bfloat16_bits"].view(
                    ml_dtypes.bfloat16
                ),
                contract_valid=arrays["cache_contract_valid"],
                expected_position=(
                    expected_prompt_length
                    + args.observer_steps
                    + args.warmup
                    + args.iterations
                    + args.trace_steps
                    - 1
                ),
                num_layers=78,
                full_indexer_count=21,
                packed_cache_width=640,
                index_width=128,
            )
            if not _same(record.get("cache_write_probe"), expected_cache):
                raise SystemExit(f"WS32 cache recomputation drifted rank {rank}")
            if record.get("cache_write_probe", {}).get("passed") is not True or record.get("state", {}).get("contract_valid") != [True]:
                raise SystemExit(f"WS32 state/cache protection failed rank {rank}")
            final_position = (
                expected_prompt_length
                + args.observer_steps
                + args.warmup
                + args.iterations
                + args.trace_steps
            )
            if record.get("state") != {
                "context_lengths": [final_position + 1],
                "contract_valid": [True],
                "position": [final_position],
            }:
                raise SystemExit(f"WS32 recurrent state drifted rank {rank}")
            timing = record.get("profiler_free_timing")
            samples = timing.get("samples_ms") if type(timing) is dict else None
            if (
                timing.get("profiler_active") is not False
                or timing.get("iterations") != args.iterations
                or timing.get("warmup") != args.warmup
                or type(samples) is not list
                or len(samples) != args.iterations
                or any(type(value) is not float or not math.isfinite(value) or value <= 0 for value in samples)
                or not _same(timing.get("distribution"), _distribution(samples))
            ):
                raise SystemExit(f"WS32 profiler-free timing drifted rank {rank}")
            all_samples.append(samples)
            memories = record.get("device_memory_after_execute")
            if type(memories) is not list or len(memories) != 4 or any(not _memory_valid(item) or item is None for item in memories):
                raise SystemExit(f"WS32 post-execution HBM drifted rank {rank}")
            trace = record.get("trace")
            trace_path = args.run_dir / "traces" / f"trace.rank{rank}.xplane.pb"
            if (
                type(trace) is not dict
                or trace.get("steps") != args.trace_steps
                or type(trace.get("files")) is not list
                or len(trace["files"]) != 1
                or _digest_file(trace_path) != trace["files"][0].get("sha256")
                or trace_path.stat().st_size != trace["files"][0].get("byte_count")
            ):
                raise SystemExit(f"WS32 trace identity drifted rank {rank}")
        elif record.get("status") != "HLO_ACQUIRED" or record.get("performance_claim") is not False:
            raise SystemExit(f"WS32 acquisition status drifted rank {rank}")
    if slots != set(range(32)):
        raise SystemExit("WS32 fleet did not cover all 32 final owners")

    summary: dict[str, Any] = {
        "artifact_kind": "greenfield_ws32_short_decoder_fleet",
        **common,
        "context_label": args.context_label,
        "graph_sha256": {
            graph: {
                "optimized_hlo_sha256": report["optimized_hlo_sha256"],
                "stablehlo_sha256": report["stablehlo_sha256"],
            }
            for graph, report in sorted(first_graphs.items())
        },
        "mode": args.mode,
        "performance_claim": args.mode == "numerical",
        "run_tag": args.tag,
        "schema_version": 1,
        "status": "HLO_ACQUIRED" if args.mode == "acquire" else "SUCCESS",
    }
    if args.mode == "numerical":
        sys.path.insert(0, str(REPO / "scripts" / "analysis"))
        import parse_xplane

        xplane = parse_xplane.aggregate_fleet(
            args.run_dir / "traces", step_module_re=r"jit_execute_body"
        )
        if xplane["n_files"] != 8 or xplane["n_cores"] != 64 or xplane["steps_per_core"] != args.trace_steps:
            raise SystemExit("WS32 fleet XPlane coverage drifted")
        critical = np.max(np.asarray(all_samples, dtype=np.float64), axis=0)
        peak_hbm, minimum_headroom = _peak(records)
        summary.update(
            {
                "cache_write_probe": records[0]["cache_write_probe"],
                "dsa_steps": records[0]["dsa_steps"],
                "fleet_profiler_free_samples_ms": critical.tolist(),
                "maximum_peak_hbm_bytes": peak_hbm,
                "minimum_hbm_headroom_bytes": minimum_headroom,
                "numerical_array_manifest_sha256": sha256(
                    _canonical(records[0]["numerical_tensors"]["arrays"])
                ).hexdigest(),
                "observed_generated_token_ids": records[0]["observed_generated_token_ids"],
                "observed_generated_token_count": len(
                    records[0]["observed_generated_token_ids"]
                ),
                "p50_ms_per_token": float(np.percentile(critical, 50)),
                "p99_ms_per_token": float(np.percentile(critical, 99)),
                "steady_wall_tokens_per_second": float(1000.0 / np.percentile(critical, 50)),
                "token_comparison": records[0]["token_comparison"],
                "verified_generated_token_count": int(
                    records[0]["token_comparison"]["compared_count"]
                ),
                "xplane": xplane,
            }
        )
    summary["summary_sha256"] = sha256(_canonical(summary)).hexdigest()
    _write_once(args.output, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


def _publish_db(args: argparse.Namespace) -> int:
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("summary_sha256") != sha256(
        _canonical({key: value for key, value in summary.items() if key != "summary_sha256"})
    ).hexdigest() or summary.get("status") != "SUCCESS":
        raise SystemExit("WS32 summary identity/status drifted before DB publication")
    connection = sqlite3.connect(args.results_db)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("BEGIN IMMEDIATE")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        env = {
            "GLM_ENGINE": "greenfield_ws32_2d",
            "checkpoint_manifest_sha256": summary["checkpoint_manifest_sha256"],
            "checkpoint_success_sha256": summary["checkpoint_success_sha256"],
            "code_hash": summary["code_hash"],
            "context_label": summary["context_label"],
            "dsa_oracle_manifest_sha256": summary["dsa_oracle_manifest_sha256"],
            "dsa_oracle_success_sha256": summary["dsa_oracle_success_sha256"],
            "mesh_sha256": summary["mesh_sha256"],
            "plan": "WS32_2D",
            "run_tag": summary["run_tag"],
            "token_oracle_manifest_sha256": summary["token_oracle_manifest_sha256"],
            "token_oracle_success_sha256": summary["token_oracle_success_sha256"],
            "xla_python_client_mem_fraction": summary[
                "xla_python_client_mem_fraction"
            ],
        }
        cursor = connection.execute(
            "INSERT INTO runs(created_utc,model,model_revision,harness_git,fork_git,env_json,pod,note) VALUES (?,?,?,?,?,?,?,?)",
            (
                now,
                "zai-org/GLM-5.2-FP8:greenfield-WS32_2D",
                summary["checkpoint_manifest_sha256"],
                summary["code_hash"][:7],
                "oracle-only",
                json.dumps(env, sort_keys=True),
                "db-v4-64-od",
                "Protected complete WS32 short-context decoder: exact tokens/DSA/state/cache/HLO/HBM/XPlane and profiler-free wall.",
            ),
        )
        run_id = int(cursor.lastrowid)
        benchmark = f"greenfield_78layer_{summary['context_label']}_ws32"
        connection.execute(
            "INSERT INTO items(run_id,benchmark,item_id,asked_utc,prompt,gold,raw_output,extracted,correct,score,n_prompt_tokens,n_gen_tokens,latency_ms,seed,finish_reason,truncated) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                benchmark,
                "gate_d_exact_token_dsa_state_cache",
                now,
                f"Execute sealed {summary['context_label']} prompt through the complete WS32 decoder.",
                "Exact sealed raw-token prefix, executing-program DSA set/ties, state/cache/HLO/HBM/XPlane and protected wall.",
                json.dumps(summary, sort_keys=True),
                json.dumps(summary["observed_generated_token_ids"]),
                1,
                1.0,
                None,
                summary["verified_generated_token_count"],
                summary["p50_ms_per_token"],
                None,
                "protected_complete",
                0,
            ),
        )
        connection.execute(
            "INSERT INTO summary(run_id,benchmark,created_utc,n,metric,value,card_value,delta,note) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                benchmark,
                now,
                1,
                "steady_wall_tok_s",
                summary["steady_wall_tokens_per_second"],
                None,
                None,
                f"summary_sha256={summary['summary_sha256']}",
            ),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    shutil.copy2(args.results_db, args.snapshot)
    record = {
        "artifact_kind": "greenfield_ws32_short_decoder_db_link",
        "benchmark": benchmark,
        "results_db_run_id": run_id,
        "results_db_sha256": _digest_file(args.snapshot),
        "run_tag": summary["run_tag"],
        "summary_sha256": summary["summary_sha256"],
    }
    record["record_sha256"] = sha256(_canonical(record)).hexdigest()
    _write_once(args.output, record)
    print(json.dumps(record, sort_keys=True))
    return 0


def _rollback_db(args: argparse.Namespace) -> int:
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("summary_sha256") != sha256(
        _canonical(
            {key: value for key, value in summary.items() if key != "summary_sha256"}
        )
    ).hexdigest() or summary.get("status") != "SUCCESS":
        raise SystemExit("WS32 DB rollback summary drifted")
    benchmark = f"greenfield_78layer_{summary['context_label']}_ws32"
    expected_env = {
        "GLM_ENGINE": "greenfield_ws32_2d",
        "checkpoint_manifest_sha256": summary["checkpoint_manifest_sha256"],
        "checkpoint_success_sha256": summary["checkpoint_success_sha256"],
        "code_hash": summary["code_hash"],
        "context_label": summary["context_label"],
        "dsa_oracle_manifest_sha256": summary["dsa_oracle_manifest_sha256"],
        "dsa_oracle_success_sha256": summary["dsa_oracle_success_sha256"],
        "mesh_sha256": summary["mesh_sha256"],
        "plan": "WS32_2D",
        "run_tag": summary["run_tag"],
        "token_oracle_manifest_sha256": summary["token_oracle_manifest_sha256"],
        "token_oracle_success_sha256": summary["token_oracle_success_sha256"],
        "xla_python_client_mem_fraction": summary[
            "xla_python_client_mem_fraction"
        ],
    }
    linked_run_id = None
    if args.db_link.exists():
        link = json.loads(args.db_link.read_text(encoding="utf-8"))
        if link.get("record_sha256") != sha256(
            _canonical(
                {key: value for key, value in link.items() if key != "record_sha256"}
            )
        ).hexdigest() or link.get("summary_sha256") != summary["summary_sha256"]:
            raise SystemExit("WS32 DB rollback link identity drifted")
        linked_run_id = link.get("results_db_run_id")
        if (
            type(linked_run_id) is not int
            or linked_run_id <= 0
            or link.get("benchmark") != benchmark
            or link.get("run_tag") != summary["run_tag"]
        ):
            raise SystemExit("WS32 DB rollback link key drifted")
    connection = sqlite3.connect(args.results_db)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("BEGIN IMMEDIATE")
        candidates = []
        for row in connection.execute(
            "SELECT run_id,created_utc,model,model_revision,harness_git,fork_git,env_json,pod,note FROM runs"
        ).fetchall():
            try:
                environment = json.loads(row[6])
            except (TypeError, json.JSONDecodeError):
                continue
            if environment == expected_env:
                candidates.append(row)
        if not candidates and linked_run_id is None:
            connection.rollback()
            return 0
        if len(candidates) != 1:
            raise SystemExit("WS32 DB rollback did not resolve one exact run")
        run = candidates[0]
        run_id = int(run[0])
        if linked_run_id is not None and run_id != linked_run_id:
            raise SystemExit("WS32 DB rollback link/run disagreement")
        item = connection.execute(
            "SELECT benchmark,item_id,asked_utc,prompt,gold,raw_output,extracted,correct,score,n_prompt_tokens,n_gen_tokens,latency_ms,seed,finish_reason,truncated FROM items WHERE run_id=?",
            (run_id,),
        ).fetchall()
        final = connection.execute(
            "SELECT benchmark,created_utc,n,metric,value,card_value,delta,note FROM summary WHERE run_id=?",
            (run_id,),
        ).fetchall()
        created = run[1]
        expected_item = (
            benchmark,
            "gate_d_exact_token_dsa_state_cache",
            created,
            f"Execute sealed {summary['context_label']} prompt through the complete WS32 decoder.",
            "Exact sealed raw-token prefix, executing-program DSA set/ties, state/cache/HLO/HBM/XPlane and protected wall.",
            json.dumps(summary, sort_keys=True),
            json.dumps(summary["observed_generated_token_ids"]),
            1,
            1.0,
            None,
            summary["verified_generated_token_count"],
            summary["p50_ms_per_token"],
            None,
            "protected_complete",
            0,
        )
        expected_final = (
            benchmark,
            created,
            1,
            "steady_wall_tok_s",
            summary["steady_wall_tokens_per_second"],
            None,
            None,
            f"summary_sha256={summary['summary_sha256']}",
        )
        if (
            run[2:] != (
                "zai-org/GLM-5.2-FP8:greenfield-WS32_2D",
                summary["checkpoint_manifest_sha256"],
                summary["code_hash"][:7],
                "oracle-only",
                json.dumps(expected_env, sort_keys=True),
                "db-v4-64-od",
                "Protected complete WS32 short-context decoder: exact tokens/DSA/state/cache/HLO/HBM/XPlane and profiler-free wall.",
            )
            or item != [expected_item]
            or final != [expected_final]
        ):
            raise SystemExit("WS32 DB rollback refused nonexact rows")
        connection.execute("DELETE FROM summary WHERE run_id=?", (run_id,))
        connection.execute("DELETE FROM items WHERE run_id=?", (run_id,))
        connection.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return 0


def main() -> int:
    args = _args()
    if args.command == "validate":
        return _validate(args)
    if args.command == "publish-db":
        return _publish_db(args)
    return _rollback_db(args)


if __name__ == "__main__":
    raise SystemExit(main())
