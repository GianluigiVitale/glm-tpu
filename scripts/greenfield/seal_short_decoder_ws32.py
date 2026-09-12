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
import subprocess
import sys
from typing import Any, Mapping

import ml_dtypes
import numpy as np


REPO = Path(__file__).resolve().parents[2]
# Unconditionally FIRST: a PYTHONPATH entry ahead of this repository would
# otherwise shadow glm_tpu.greenfield.validation, so the sealer would check one
# tree for modifications and import the §21.2 arithmetic and the reviewed
# reference-row registry from another.
while str(REPO) in sys.path:
    sys.path.remove(str(REPO))
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
from glm_tpu.greenfield.validation.ws32_evidence import (  # noqa: E402
    EVIDENCE_LAYOUT_V1,
    EVIDENCE_LAYOUT_V2,
)
from glm_tpu.greenfield.validation.ws32_prefill import (  # noqa: E402
    PREFILL_MODE, PREFILL_MODES, SERIAL_PREFILL_MODE,
    require_fleet_prefill_mode, require_prefill_mode,
)
from glm_tpu.greenfield.validation.ws32_short_context import (  # noqa: E402
    committed_artifact_path as _committed_artifact_path,
)
from glm_tpu.greenfield.validation import ws32_delivery_quality as delivery  # noqa: E402
from glm_tpu.greenfield.validation import (  # noqa: E402
    bind_ws32_adjudication,
    compare_ws32_dsa_step,
    load_ws32_adjudicated_divergence,
    compare_ws32_dsa_within_engine,
    compare_ws32_raw_tokens,
    load_ws32_long_context_oracle,
    load_ws32_short_context_oracle,
    validate_ws32_cache_probe,
)
from glm_tpu.greenfield.validation.long_context_oracle import (
    WS32_CONTEXT_LABELS,
    WS32_PROMPT_LENGTHS,
    require_ws32_long_context_profile,
)

_ZERO_SHA = "0" * 64


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--run-dir", required=True, type=Path)
    validate.add_argument("--topology-capture-root", required=True, type=Path)
    validate.add_argument("--token-oracle-dir", type=Path, default=None)
    # Spec §23.5: L7/L8 seal against a token-only long-context oracle. There is
    # no legacy DSA capture at these lengths, so the sealer mirrors the runner's
    # within-engine contract and makes no cross-oracle claim.
    validate.add_argument("--long-context", choices=("passkey", "e0"), default=None)
    validate.add_argument("--long-context-oracle-dir", type=Path, default=None)
    validate.add_argument("--long-context-manifest-sha256", default="0" * 64)
    validate.add_argument("--long-context-success-sha256", default="0" * 64)
    validate.add_argument("--tokenizer-root", type=Path, default=None)
    validate.add_argument("--dsa-oracle-dir", type=Path, default=None)
    validate.add_argument("--mode", choices=("acquire", "numerical"), required=True)
    validate.add_argument(
        "--context-label", choices=WS32_CONTEXT_LABELS, required=True
    )
    validate.add_argument("--tag", required=True)
    validate.add_argument("--prefill-chunk", type=int, default=DEFAULT_PREFILL_CHUNK)
    validate.add_argument("--prefill-mode", choices=PREFILL_MODES, default=SERIAL_PREFILL_MODE)
    validate.add_argument("--batched-prefill-profile", default="")
    validate.add_argument("--prefill-memory-reserve-bytes", type=int, default=0)
    validate.add_argument("--prefill-budget-seconds", type=float, default=0)
    validate.add_argument("--rotary-diagnostic", choices=(0, 1), default=0, type=int)
    validate.add_argument("--host-main-rope-table", choices=(0, 1), default=0, type=int)
    # Sealed v1 prefixes stay re-verifiable: the required layout is a pin, not a constant.
    validate.add_argument(
        "--evidence-layout",
        choices=(EVIDENCE_LAYOUT_V1, EVIDENCE_LAYOUT_V2),
        default=EVIDENCE_LAYOUT_V2,
    )
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
    validate.add_argument("--checkpoint-transport", choices=("gcsfuse", "shm"), default="gcsfuse")
    validate.add_argument("--dsa-adjudication-record", type=Path)
    validate.add_argument("--dsa-adjudication-sha256", default="0" * 64)
    validate.add_argument(
        "--later-event-alarm-acknowledged", choices=(0, 1), default=0, type=int
    )
    validate.add_argument("--later-event-alarm-profile", type=Path)
    validate.add_argument("--later-event-alarm-profile-sha256", default="0" * 64)
    validate.add_argument("--later-event-alarm-lessons-pin", default="")
    validate.add_argument("--recovery-code-hash", default="")
    validate.add_argument(
        "--reviewed-ref",
        default="refs/remotes/origin/rewrite/topology-first-decode",
        help="the published branch a protected seal's run pin must be contained in",
    )
    validate.add_argument(
        "--strategy-nd-dense", choices=(0, 1), required=True, type=int
    )
    validate.add_argument(
        "--strategy-nd-dense-overlay-manifest-sha256", required=True
    )
    validate.add_argument(
        "--strategy-nd-dense-overlay-manifest-file-sha256", required=True
    )
    validate.add_argument(
        "--strategy-nd-dense-overlay-success-file-sha256", required=True
    )
    for graph in (
        "exact-materialize",
        "exact-promote",
        "prefill-chunk",
        "prefill-tail",
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


# The alarm-acknowledgement check already used an absolute git; the
# pre-registration check is the more security-critical of the two.
_GIT = "/usr/bin/git"
# The files that decide what a protected seal accepts: the sealer itself, the
# validation packages enforcing §21.2/§23.5, and the reviewed artifact tree.
# Edits elsewhere (tests, docs,
# unrelated scripts) do not change the verdict and do not block a seal.
_ENFORCEMENT_SURFACE = (
    "scripts/greenfield/seal_short_decoder_ws32.py",
    "glm_tpu/greenfield/validation",
    "glm_tpu/greenfield/benchmarking",
    "glm_tpu/greenfield/sharding",
    # §23.8: `_require_main_rope_table` recomputes the accepted table digest from
    # the reference rotary construction, the runtime's theta and the model
    # geometry, so those decide acceptance too.
    "glm_tpu/greenfield/kernels/reference",
    "glm_tpu/greenfield/runtime",
    "glm_tpu/greenfield/types.py",
    "configs/glm-5.2-fp8-config.json",
    "docs/artifacts",
    # The sealer recomputes the §23.5 token verdict by executing the runner's
    # own rule, and that rule runs the legacy extractor; both decide what a
    # seal accepts, so both are part of what a reviewer must have reviewed.
    "scripts/greenfield/run_short_decoder_ws32.py",
    # Long graph/phase admission and its source/receipt authorities also decide
    # what this sealer accepts. Keep them in the committed enforcement surface.
    "scripts/greenfield/ws32_delivery_runtime.py",
    "scripts/greenfield/ws32_delivery_wk.py",
    "scripts/greenfield/ws32_history_call_evidence.py",
    "scripts/greenfield/ws32_dense_frontier_admission.py",
    "scripts/greenfield/prefill_window_worker.py",
    "scripts/greenfield/probe_ws32_prefill_layer.py",
    "scripts/greenfield/ws32_delivery_hlo.py",
    "scripts/greenfield/ws32_delivery_programs.py",
    "scripts/greenfield/ws32_phase_weights.py",
    "scripts/greenfield/ws32_owned_prefill_memory.py",
    "scripts/greenfield/ws32_batched_prefill_runner.py",
    "scripts/greenfield/ws32_prefill_owned_state.py",
    "scripts/greenfield/ws32_capture_barrier_compile.py",
    "scripts/greenfield/ws32_flat_rows_compile.py",
    "scripts/greenfield/ws32_pending_rows_compile.py",
    "scripts/greenfield/ws32_owned_state_compile.py",
    "scripts/greenfield/ws32_delivery_compile.py",
    "scripts/greenfield/ws32_rolled_prefill_compile.py",
    "scripts/greenfield/ws32_canonical_prefill_compile.py",
    "bench/glm_longctx.py",
    "bench/extract.py",
    # Imported by glm_longctx when executing the SHA-pinned extraction utility.
    "bench/engine.py",
    "bench/provenance.py",
)
# `docs/artifacts` legitimately holds untracked outputs of the offline
# adjudicator (§21.2 calls prior attempts untracked and requires them to be left
# in place), so only TRACKED modifications are refused there; a new file is not
# a widening of what the seal accepts.
_TRACKED_ONLY_SURFACE = ("docs/artifacts",)
DEFAULT_PREFILL_CHUNK = 2048
DEFAULT_CONTEXT_CAPACITY = 8192


def _validate_run_tag(
    tag: str,
    *,
    context_label: str,
    mode: str,
    prefill_chunk: int = DEFAULT_PREFILL_CHUNK,
    context_capacity: int = DEFAULT_CONTEXT_CAPACITY,
    host_main_rope_table: bool = False,
    prefill_mode: str = SERIAL_PREFILL_MODE,
    batched_prefill_profile: str = "",
) -> None:
    """Spec §23.3/§23.8: a non-default prefill chunk, context capacity or the
    legacy-faithful main rotary table is part of the run identity."""
    if type(prefill_chunk) is not int or prefill_chunk <= 0:
        raise SystemExit("WS32 prefill chunk must be a positive integer")
    if type(context_capacity) is not int or context_capacity <= 0:
        raise SystemExit("WS32 context capacity must be a positive integer")
    suffix = "" if prefill_chunk == DEFAULT_PREFILL_CHUNK else f"_c{prefill_chunk}"
    if context_capacity != DEFAULT_CONTEXT_CAPACITY:
        suffix += f"_cap{context_capacity}"
    if host_main_rope_table:
        suffix += "_hrope"
    require_prefill_mode(prefill_mode)
    if prefill_mode == PREFILL_MODE:
        suffix += "_bp1"
        if batched_prefill_profile:
            from glm_tpu.greenfield.validation.ws32_prefill_admission import profile_is_paired, profile_is_rolled
            if profile_is_paired(batched_prefill_profile):
                suffix += "_ps1"
            if profile_is_rolled(batched_prefill_profile):
                suffix += "_rp1_ep1_lm1"
            from glm_tpu.greenfield.validation.ws32_prefill_admission import CANONICAL_PROFILES
            if batched_prefill_profile in CANONICAL_PROFILES:
                suffix += "_cd1"
            if batched_prefill_profile == delivery.PROFILE:
                suffix += "_s26"
            from glm_tpu.greenfield.validation.ws32_prefill_admission import FROZEN_LIVE32_PROFILE
            if batched_prefill_profile == FROZEN_LIVE32_PROFILE:
                suffix += "_live32"
    elif batched_prefill_profile:
        raise SystemExit("serial tag cannot bind a batched profile")
    pattern = (
        rf"greenfield_ws32_short_decoder_{re.escape(context_label)}_"
        rf"{re.escape(mode)}{re.escape(suffix)}_[0-9]{{8}}T[0-9]{{15}}Z"
    )
    if re.fullmatch(pattern, tag) is None:
        raise SystemExit("WS32 run tag contradicts active context/mode/prefill chunk/capacity")


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
        "strategy_nd_dense_expert_gather_count",
        "strategy_nd_dense_hidden_gather_count",
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


def _require_batched_fleet_memory(
    records: list[Mapping[str, Any]], captures: tuple[Mapping[str, Any], ...],
    physical_mesh: Any, *, long_context_label: str | None = None,
) -> dict[str, Any]:
    """Batched-only physical-owner binding, additional to ordinary provenance."""
    from glm_tpu.greenfield.validation.ws32_prefill_admission import (
        SHORT_RESERVE_BYTES, SHORT_DEVICE_LIMIT_BYTES, profile_is_paired,
        validate_short_compiled_memory,
    )
    from glm_tpu.greenfield.validation.ws32_prefill_fleet_memory import validate_batched_fleet_memory
    from scripts.greenfield import ws32_delivery_runtime as long_runtime

    profile = records[0].get("batched_prefill_profile")
    if profile == long_runtime.PROFILE:
        # Workload choice is supplied by the trusted sealer request, never
        # inferred from the record's ownership/schema or a claimed capacity.
        expected = long_runtime.registration(REPO, long_context_label)
        for record in records:
            if (record.get("batched_prefill_profile") != profile
                    or record.get("context_capacity") != long_runtime.programs.long_plan(long_context_label).context_capacity):
                raise ValueError("long fleet memory mixes workloads/profiles")
            long_runtime.require_role_record(record["prefill_execution"]["memory_admission"], long_context_label)
            for graph in long_runtime.ROLES:
                long_runtime.require_memory(record["compiled_memory_analysis"][graph], expected[graph]["memory"])
        return validate_batched_fleet_memory(
            records, captures, physical_mesh.flattened_device_ids,
            required_reserve_bytes=long_runtime.RESERVE,
            expected_device_limit_bytes=long_runtime.DEVICE_LIMIT,
            expected_prefill_analyses={graph: entry["memory"] for graph, entry in expected.items()},
            state_ownership_contract=long_runtime.programs.state_ownership(long_context_label),
        )
    if long_context_label is not None:
        raise ValueError("long fleet context requires explicit long profile")
    profile_is_paired(profile)
    if any(record.get("batched_prefill_profile") != profile for record in records):
        raise ValueError("batched memory requires the fixed numerical profile")
    acquired = records[0]["compiled_memory_analysis"]
    for record in records:
        for graph in ("prefill_chunk", "prefill_tail"):
            validate_short_compiled_memory(
                graph, record["compiled_memory_analysis"][graph], profile=profile, repo=REPO
            )
    return validate_batched_fleet_memory(
        records, captures, physical_mesh.flattened_device_ids,
        required_reserve_bytes=SHORT_RESERVE_BYTES,
        expected_device_limit_bytes=SHORT_DEVICE_LIMIT_BYTES,
        expected_prefill_analyses={
            graph: acquired[graph]
            for graph in ("prefill_chunk", "prefill_tail")
        },
    )


def _replay_batched_graph(
    stable: str, optimized: str, *, graph: str, args: argparse.Namespace,
) -> dict[str, Any]:
    """Re-derive bounded mode from actual text, not the worker's stored pass."""
    from scripts.greenfield import ws32_delivery_hlo as long_hlo
    if args.batched_prefill_profile == long_hlo.PROFILE:
        if graph not in {"prefill_chunk", "prefill_tail"}:
            raise ValueError("long companion graph integration remains unregistered")
        return long_hlo.inspect_hlo(
            stable, optimized, repo=REPO, context_label=args.context_label,
            role=graph,
            expected_stablehlo_sha256=getattr(args, f"expected_{graph}_stablehlo_sha256"),
            expected_optimized_sha256=getattr(args, f"expected_{graph}_optimized_hlo_sha256"),
        )
    if graph in {"prefill_chunk", "prefill_tail"}:
        from glm_tpu.greenfield.validation.ws32_prefill_admission import inspect_short_prefill_graph
        return inspect_short_prefill_graph(
            stable, optimized, graph=graph, profile=args.batched_prefill_profile, repo=REPO,
            expected_stable=getattr(args, f"expected_{graph}_stablehlo_sha256"),
            expected_optimized=getattr(args, f"expected_{graph}_optimized_hlo_sha256"),
        )
    from glm_tpu.greenfield.validation.ws32_prefill_admission import (
        profile_is_paired, short_graph_identity,
    )

    profile_is_paired(getattr(args, "batched_prefill_profile", ""))
    identity = short_graph_identity(
        stable, optimized, graph=graph, profile=args.batched_prefill_profile, repo=REPO,
        expected_stable=getattr(args, f"expected_{graph}_stablehlo_sha256"),
        expected_optimized=getattr(args, f"expected_{graph}_optimized_hlo_sha256"),
    )
    pins = dict(
        expected_stablehlo_sha256=getattr(args, f"expected_{graph}_stablehlo_sha256"),
        expected_optimized_hlo_sha256=identity["raw_optimized_hlo_sha256"],
    )
    if graph.startswith("exact_"):
        report = validate_ws32_exact_dsa_materializer_hlo(
            stable, optimized, kind=graph, **pins,
        ).to_dict()
    else:
        report = validate_ws32_decoder_hlo(
            stable, optimized, hidden_size=6144, kind=graph,
            exact_dsa=bool(args.exact_dsa), strategy_nd_dense=bool(args.strategy_nd_dense),
            host_main_rope_table=bool(args.host_main_rope_table), **pins,
        ).to_dict()
    if not _graph_valid(report, mode="numerical"):
        raise ValueError(f"batched companion graph contract failed: {graph}")
    return {**report, "source_location_identity": identity}


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
    prefill_mode = getattr(args, "prefill_mode", SERIAL_PREFILL_MODE)
    delivery_task = getattr(args, "batched_prefill_profile", "") == delivery.PROFILE
    require_prefill_mode(prefill_mode)
    if prefill_mode == PREFILL_MODE:
        from glm_tpu.greenfield.validation.ws32_prefill_admission import require_short_numerical_request, short_context
        from glm_tpu.greenfield.validation.ws32_prefill_admission import FROZEN_LIVE32_PROFILE, FROZEN_FIRST_WINDOW_PROFILE

        if args.batched_prefill_profile in (FROZEN_LIVE32_PROFILE, FROZEN_FIRST_WINDOW_PROFILE):
            raise SystemExit("live-window diagnostic cannot seal a numerical/performance promotion")

        if getattr(args, "mode", None) != "numerical" or getattr(args, "context_label", None) != short_context(args.batched_prefill_profile):
            raise SystemExit("batched seals require registered short numerical mode")
        require_short_numerical_request(args, prompt_length=WS32_PROMPT_LENGTHS[args.context_label], repo=REPO)
    # §23.5 fixes the E0 measurement window independently of caller-provided
    # timing metadata. Reject an undersized plan even for acquisition, before
    # reading artifacts or allowing hours of prefill to hide a ten-step window.
    if args.context_label == "256k_e0" and args.iterations != 256:
        raise SystemExit("WS32 E0 requires exactly 256 timed decode iterations (§23.5)")
    if args.mode == "acquire" and args.dsa_adjudication_record is not None:
        # An acquisition seals graphs, not numbers. Carrying an adjudication
        # binding into an HLO_ACQUIRED SUCCESS would read as a correctness claim
        # that nothing in an acquisition establishes. Checked before the tag so
        # the refusal names the real problem.
        raise SystemExit("WS32 acquisition seals carry no adjudication record")
    _validate_run_tag(
        args.tag,
        context_label=args.context_label,
        mode=args.mode,
        prefill_chunk=args.prefill_chunk,
        context_capacity=args.context_capacity,
        host_main_rope_table=bool(args.host_main_rope_table),
        prefill_mode=prefill_mode,
        batched_prefill_profile=getattr(args, "batched_prefill_profile", ""),
    )
    from glm_tpu.greenfield.validation.ws32_prefill_admission import require_hlo_pin_request
    require_hlo_pin_request(args, compile_only=args.mode == "acquire", repo=REPO)
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
    overlay_pins = (
        args.strategy_nd_dense_overlay_manifest_sha256,
        args.strategy_nd_dense_overlay_manifest_file_sha256,
        args.strategy_nd_dense_overlay_success_file_sha256,
    )
    if args.strategy_nd_dense:
        if any(value == "0" * 64 for value in overlay_pins):
            raise SystemExit("WS32 StrategyND dense overlay pins are vacant")
    elif any(value != "0" * 64 for value in overlay_pins):
        raise SystemExit("default WS32 path must keep dense overlay pins vacant")
    runner_paths = sorted(args.run_dir.glob("fleet/runner.rank*.json"))
    if len(runner_paths) != 8:
        raise SystemExit(f"expected eight WS32 runner records, got {len(runner_paths)}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in runner_paths]
    if require_fleet_prefill_mode(records) != prefill_mode:
        raise SystemExit("WS32 runner prefill mode contradicts sealer pin")
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
    long_context = None
    if args.long_context is not None:
        if args.long_context_oracle_dir is None:
            raise SystemExit("WS32 long-context seal needs its oracle directory")
        if args.dsa_adjudication_record is not None:
            raise SystemExit("WS32 long-context seals bind no §21.2 adjudication record")
        try:
            long_context = load_ws32_long_context_oracle(
                args.long_context_oracle_dir,
                expected_manifest_sha256=args.long_context_manifest_sha256,
                expected_success_sha256=args.long_context_success_sha256,
                expected_kind=args.long_context,
            )
        except (OSError, ValueError, KeyError) as error:
            raise SystemExit(f"WS32 long-context oracle is not loadable: {error}")
        try:
            require_ws32_long_context_profile(
                args.context_label,
                long_context,
                success_sha256=args.long_context_success_sha256,
            )
        except ValueError as error:
            raise SystemExit(str(error))
    elif args.context_label not in ("2k", "8k"):
        raise SystemExit(
            "WS32 long-context labels require the long-context oracle flags"
        )
    if long_context is None:
        if args.token_oracle_dir is None or args.dsa_oracle_dir is None:
            raise SystemExit("WS32 short-context seals need both sealed oracles")
        oracle = load_ws32_short_context_oracle(
            args.token_oracle_dir,
            args.dsa_oracle_dir,
            expected_token_manifest_sha256=args.token_oracle_manifest_sha256,
            expected_dsa_manifest_sha256=args.dsa_oracle_manifest_sha256,
            expected_token_success_sha256=args.token_oracle_success_sha256,
            expected_dsa_success_sha256=args.dsa_oracle_success_sha256,
        )
    else:
        # §23.5: there is no legacy capture at these lengths. Binding a
        # short-context oracle here would let a 2k/8k artifact stand behind a
        # long-context seal, so the run's pinned inputs record it as absent.
        if (
            args.token_oracle_dir is not None
            or args.dsa_oracle_dir is not None
            or args.token_oracle_manifest_sha256 != _ZERO_SHA
            or args.token_oracle_success_sha256 != _ZERO_SHA
            or args.dsa_oracle_manifest_sha256 != _ZERO_SHA
            or args.dsa_oracle_success_sha256 != _ZERO_SHA
        ):
            raise SystemExit("WS32 long-context seals bind no short-context oracle")
        oracle = None
    repository_root = Path(__file__).resolve().parents[2]
    if args.recovery_code_hash and not re.fullmatch(r"[0-9a-f]{40}", args.recovery_code_hash):
        raise SystemExit("WS32 recovery code hash must be a full commit id")
    if args.dsa_adjudication_record is not None or long_context is not None or prefill_mode == PREFILL_MODE:
        # §21.2: the reviewed reference-row registry lives in this repository's
        # own source, so a dirty working tree at seal time can widen what is
        # accepted and then be reverted without leaving a trace in the record.
        # An adjudicated seal therefore requires a clean tree.
        #
        # §23.5 seals require it for the same reason: the label-to-identity
        # table and the profile refusal that keep a long-context label bound to
        # its own sealed capture are source in this tree, and a §23.5 run binds
        # no adjudication record, so without this clause the runs those rules
        # govern would be the only ones sealing with them unchecked.
        _require_imports_come_from(repository_root)
        _require_clean_worktree(repository_root)
        enforcement_surface = _enforcement_surface_identity(repository_root)
        _require_reviewed_enforcement(
            repository_root,
            enforcement_surface,
            code_hash=args.code_hash,
            recovery_code_hash=args.recovery_code_hash,
            reviewed_ref=args.reviewed_ref,
        )
    else:
        enforcement_surface = None
    if args.recovery_code_hash and not re.fullmatch(r"[0-9a-f]{40}", args.recovery_code_hash):
        raise SystemExit("WS32 recovery code hash must be a full commit id")
    _committed_in_run_pin = _make_pre_registration_check(args, repository_root)

    def _committed_record_path(path: Path) -> str:
        resolved = Path(path).resolve()
        try:
            relative = str(resolved.relative_to(repository_root))
        except ValueError:
            raise SystemExit(
                f"WS32 adjudication record is outside the repository: {path}"
            )
        if not _committed_artifact_path(relative, ".json"):
            # Otherwise a record committed elsewhere sidesteps the disclosure
            # scan, which globs the reviewed directory.
            raise SystemExit(
                f"WS32 adjudication record must be a docs/artifacts/gate-*.json artifact: "
                f"{relative}"
            )
        _committed_in_run_pin(relative, "adjudication record")
        return relative

    if args.dsa_adjudication_record is None:
        if args.dsa_adjudication_sha256 != "0" * 64:
            raise SystemExit("WS32 adjudication SHA given without a record")
        dsa_adjudication = None
        expected_dsa_adjudication = None
    else:
        record_relative = _committed_record_path(args.dsa_adjudication_record)
        try:
            dsa_adjudication = load_ws32_adjudicated_divergence(
                args.dsa_adjudication_record,
                expected_sha256=args.dsa_adjudication_sha256,
                repository_root=repository_root,
            )
        except (OSError, ValueError, KeyError) as error:
            # A loader refusal is a refusal to seal, not a traceback.
            raise SystemExit(f"WS32 adjudication record is not loadable: {error}")
        try:
            bind_ws32_adjudication(
                dsa_adjudication,
                oracle,
                observer_steps=args.observer_steps,
                context_label=args.context_label,
            )
        except ValueError as error:
            raise SystemExit(f"WS32 adjudication record does not bind this run: {error}")
        for relative, label in (
            # The analysis is the record's ground and the row decides its
            # verdict, so both must have existed before the run too; otherwise
            # the numbers could be written to fit what the run produced.
            (dsa_adjudication.analysis_path, "adjudication analysis"),
            (dsa_adjudication.reference_row_path, "adjudication reference row"),
            (
                dsa_adjudication.reference_validation_path,
                "adjudication reference validation record",
            ),
        ):
            if relative is not None:
                _committed_in_run_pin(relative, label)
        if dsa_adjudication.engine_source_run == args.tag:
            # §21.2 pre-registration: a record derived from this very run would be
            # fitted to the data it judges and so could never fail. The source is
            # taken from the record the loader already SHA-bound, so there is no
            # window between the identity check and this one.
            raise SystemExit(
                "WS32 adjudication record was derived from the run being sealed"
            )
        if args.mode == "numerical":
            # §21.2 items 3-4 are RE-DERIVED from the run being sealed, the
            # sealed oracle and the pre-registered reference row. A verdict the
            # sealer merely reads is a claim by whoever wrote the file; the
            # record's job is to fix the row and the expected divergence in
            # advance, never to supply the answer. Rank 0's archive is bound to
            # rank 0's own record here; the rank loop below re-verifies every
            # rank and proves they agree.
            _rederive_ws32_adjudication(
                arrays=_rank0_dsa_arrays(args.run_dir, records[0]),
                oracle=oracle,
                adjudication=dsa_adjudication,
                repository_root=repository_root,
                rank=0,
            )
        expected_dsa_adjudication = {
            "event_index": dsa_adjudication.event_index,
            "mode": "first_divergent_event",
            "record_path": record_relative,
            "record_sha256": dsa_adjudication.record_sha256,
            "step": dsa_adjudication.step,
        }

    common = {
        "checkpoint_manifest_sha256": args.checkpoint_manifest_sha256,
        "checkpoint_success_sha256": args.checkpoint_success_sha256,
        "code_hash": args.code_hash,
        "compile_only": args.mode == "acquire",
        "context_capacity": args.context_capacity,
        "dsa_oracle_manifest_sha256": (
            None if long_context is not None else args.dsa_oracle_manifest_sha256
        ),
        "dsa_oracle_success_sha256": (
            None if long_context is not None else args.dsa_oracle_success_sha256
        ),
        "dsa_association_summary_sha256": (
            args.dsa_association_summary_sha256
        ),
        "dsa_association_success_sha256": (
            args.dsa_association_success_sha256
        ),
        "exact_dsa": bool(args.exact_dsa),
        "mesh_sha256": args.mesh_sha256,
        "source_inventory_sha256": args.source_inventory_sha256,
        "strategy_nd_dense": bool(args.strategy_nd_dense),
        "long_context": (
            None
            if long_context is None
            else {
                "depth": long_context.depth,
                "item_row_id": long_context.item_row_id,
                "kind": long_context.kind,
                "manifest_sha256": long_context.manifest_sha256,
                "source_run_id": long_context.source_run_id,
                "success_sha256": args.long_context_success_sha256,
            }
        ),
        "token_oracle_manifest_sha256": (
            None if long_context is not None else args.token_oracle_manifest_sha256
        ),
        "token_oracle_success_sha256": (
            None if long_context is not None else args.token_oracle_success_sha256
        ),
        "topology_fleet_sha256": args.topology_fleet_sha256,
        "topology_sha256": args.topology_sha256,
        "xla_python_client_mem_fraction": ".95",
    }
    if prefill_mode == PREFILL_MODE:
        from glm_tpu.greenfield.validation.ws32_prefill_admission import short_numerical_identity

        common.update(short_numerical_identity(profile=args.batched_prefill_profile))
    first_graphs = records[0].get("graphs")
    expected_graphs = {
        "cache_probe",
        "decode",
        "observer",
        "prefill_chunk",
        "prefill_tail",
    }
    if args.exact_dsa:
        expected_graphs.update({"exact_materialize", "exact_promote"})
    if type(first_graphs) is not dict or set(first_graphs) != expected_graphs:
        raise SystemExit("WS32 graph set drifted")
    slots: set[int] = set()
    # Same raw SHA pairs are required across ranks; parse each batched graph
    # once, still hash/compare EVERY rank's bytes and complete stored report.
    batched_replays: dict[tuple[str, str, str], dict[str, Any]] = {}
    all_samples: list[list[float]] = []
    expected_prompt_length = WS32_PROMPT_LENGTHS[args.context_label]
    layout_keys = (
        set() if args.evidence_layout == EVIDENCE_LAYOUT_V1 else {"evidence_layout"}
    )
    pre_keys = {
        "artifact_kind",
        "base_device_memory_after_load",
        "checkpoint_manifest_sha256",
        "checkpoint_success_sha256",
        "checkpoint_transport",
        "checkpoint_verified_device_slots",
        "code_hash",
        "compile_only",
        "compile_seconds",
        "compiled_memory_analysis",
        "context_capacity",
        "device_memory_after_compile",
        "device_memory_after_load",
        "device_memory_before_load",
        "dsa_adjudication",
        "dsa_oracle_manifest_sha256",
        "dsa_oracle_success_sha256",
        "dsa_association_summary_sha256",
        "dsa_association_success_sha256",
        "exact_dsa",
        "graphs",
        "main_rope_table",
        "hostname",
        "jax_process_index",
        "launch_process_id",
        "load_seconds",
        "local_device_slots",
        "long_context",
        "mesh_sha256",
        "prefill_chunk_length",
        "prefill_execution",
        "prompt_length",
        "rotary_diagnostic",
        "source_inventory_sha256",
        "strategy_nd_dense",
        "strategy_nd_dense_overlay",
        "token_oracle_manifest_sha256",
        "token_oracle_success_sha256",
        "topology_fleet_sha256",
        "topology_sha256",
        "xla_python_client_mem_fraction",
    }
    pre_keys |= layout_keys
    if prefill_mode == PREFILL_MODE:
        pre_keys |= set(short_numerical_identity(profile=args.batched_prefill_profile))
        if args.mode == "numerical":
            pre_keys |= {"batched_prefill_memory", "batched_prefill_profile"}
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
    if prefill_mode == PREFILL_MODE:
        numerical_keys.add("batched_device_memory_after_execute")
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
        if args.evidence_layout == EVIDENCE_LAYOUT_V1:
            if "evidence_layout" in record:
                raise SystemExit(f"WS32 evidence layout drifted at rank {rank}")
        elif record.get("evidence_layout") != args.evidence_layout:
            raise SystemExit(f"WS32 evidence layout drifted at rank {rank}")
        _require_prefill_execution(
            record,
            mode=args.mode,
            prompt_length=expected_prompt_length,
            rank=rank,
            expected_chunk=args.prefill_chunk,
            prefill_mode=prefill_mode,
        )
        if rank and not _same(
            record.get("prefill_chunk_length"), records[0].get("prefill_chunk_length")
        ):
            raise SystemExit(f"WS32 prefill chunk length disagrees at rank {rank}")
        _require_main_rope_table(
            record,
            enabled=bool(args.host_main_rope_table),
            context_capacity=args.context_capacity,
            rank=rank,
            first=records[0],
        )
        _require_rotary_diagnostic(
            record, enabled=bool(args.rotary_diagnostic), rank=rank, first=records[0]
        )
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
        if prefill_mode != PREFILL_MODE and any(not _graph_valid(value, mode=args.mode) for value in record["graphs"].values()):
            raise SystemExit(f"WS32 graph contract failed at rank {rank}")
        for graph, report in record["graphs"].items():
            stable = args.run_dir / "fleet_hlo" / f"{graph}.rank{rank}.stablehlo.mlir"
            optimized = args.run_dir / "fleet_hlo" / f"{graph}.rank{rank}.optimized_hlo.txt"
            stable_text = stable.read_text(encoding="utf-8")
            optimized_text = optimized.read_text(encoding="utf-8")
            if _digest_file(stable) != report["stablehlo_sha256"] or _digest_file(optimized) != report["optimized_hlo_sha256"]:
                raise SystemExit(f"WS32 HLO artifact drifted at rank {rank}/{graph}")
            if prefill_mode == PREFILL_MODE:
                key = (graph, report["stablehlo_sha256"], report["optimized_hlo_sha256"])
                if key not in batched_replays:
                    batched_replays[key] = _replay_batched_graph(
                        stable_text, optimized_text, graph=graph, args=args,
                    )
                replay = batched_replays[key]
            elif graph.startswith("exact_"):
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
                    kind=_linter_kind(graph),
                    exact_dsa=bool(args.exact_dsa),
                    strategy_nd_dense=bool(args.strategy_nd_dense),
                    host_main_rope_table=bool(args.host_main_rope_table),
                ).to_dict()
            normalized_record = dict(report)
            if args.mode == "acquire":
                normalized_record["passed"] = True
                normalized_record["violations"] = []
            if not _same(replay, normalized_record):
                raise SystemExit(f"WS32 HLO replay drifted at rank {rank}/{graph}")
            if args.mode == "numerical" and prefill_mode != PREFILL_MODE and (
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
        overlay = record.get("strategy_nd_dense_overlay")
        if args.strategy_nd_dense:
            if type(overlay) is not dict or set(overlay) != {
                "local_records",
                "manifest_file_sha256",
                "manifest_sha256",
                "success_file_sha256",
            }:
                raise SystemExit(
                    f"WS32 dense overlay schema drifted at rank {rank}"
                )
            if (
                overlay["manifest_sha256"] != overlay_pins[0]
                or overlay["manifest_file_sha256"] != overlay_pins[1]
                or overlay["success_file_sha256"] != overlay_pins[2]
            ):
                raise SystemExit(
                    f"WS32 dense overlay identity drifted at rank {rank}"
                )
            local_overlay = overlay["local_records"]
            if type(local_overlay) is not list or len(local_overlay) != 12:
                raise SystemExit(
                    f"WS32 dense overlay cardinality drifted at rank {rank}"
                )
            coverage = {
                (item.get("device_id"), item.get("layer_id"))
                for item in local_overlay
            }
            if coverage != {
                (device_id, layer)
                for device_id in expected_ids
                for layer in range(3)
            }:
                raise SystemExit(
                    f"WS32 dense overlay coverage drifted at rank {rank}"
                )
        elif overlay is not None:
            raise SystemExit(
                f"default WS32 path retained dense overlay at rank {rank}"
            )
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
        for field in (
            "device_memory_before_load",
            "base_device_memory_after_load",
            "device_memory_after_load",
            "device_memory_after_compile",
        ):
            if type(record.get(field)) is not list or len(record[field]) != 4 or any(not _memory_valid(item) for item in record[field]):
                raise SystemExit(f"WS32 memory record drifted: rank={rank} field={field}")
        if args.mode == "numerical":
            if record.get("status") != "SUCCESS" or record.get("correctness_passed") is not True or record.get("performance_claim") is not False:
                raise SystemExit(f"WS32 numerical terminal status drifted rank {rank}")
            comparison = record.get("token_comparison") or {}
            if delivery_task:
                if comparison.get("passkey_matches_gold") is not True or "exact_prefix_match" in comparison:
                    raise SystemExit(f"WS32 §26 task answer/claim drifted rank {rank}")
            elif long_context is None:
                if comparison.get("exact_prefix_match") is not True:
                    raise SystemExit(f"WS32 raw tokens drifted rank {rank}")
            else:
                # §23.5: nothing is "raw tokens exact" here. L7 passes on the
                # extracted passkey; L8 has no correctness oracle, and either way
                # a key claiming token exactness must not be present.
                if "exact_prefix_match" in comparison:
                    raise SystemExit(f"WS32 long-context token claim drifted rank {rank}")
                if long_context.kind == "passkey":
                    if comparison.get("passkey_matches_gold") is not True:
                        raise SystemExit(f"WS32 passkey drifted rank {rank}")
                elif comparison.get("passkey_matches_gold") is not None:
                    raise SystemExit(f"WS32 L8 declares a correctness verdict rank {rank}")
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
                or len(observed_token_ids) < (
                    # §23.5: the legacy ids are a diagnostic reference, not an
                    # answer key. Only L7 reads a fixed number of them (the
                    # twenty its criterion detokenises); requiring an L8 run to
                    # emit the legacy 256 would refuse every sealable L8 run.
                    0
                    if long_context is not None and long_context.kind != "passkey"
                    else (
                        long_context.generated_token_ids.size
                        if long_context is not None
                        else oracle.generated_token_ids.size
                    )
                )
            ):
                raise SystemExit(f"WS32 raw-token cardinality drifted rank {rank}")
            if delivery_task:
                expected_token_comparison = delivery.token_result(
                    observed_token_ids, oracle, tokenizer_root=args.tokenizer_root
                )
            elif long_context is not None:
                expected_token_comparison = _long_context_token_result(
                    observed_token_ids, long_context, tokenizer_root=args.tokenizer_root
                )
            else:
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
            _require_ranks_agree(array_records, records, rank=rank)
            if dsa_adjudication is not None:
                # Every rank is adjudicated on its OWN arrays. The rank-0 pass
                # in the adjudication block runs earlier so it is reachable
                # before the schema checks; this one restores the eight-rank
                # property rather than resting it on the agreement check above.
                _rederive_ws32_adjudication(
                    arrays=arrays,
                    oracle=oracle,
                    adjudication=dsa_adjudication,
                    repository_root=repository_root,
                    rank=rank,
                )
            expected_dsa = []
            for step in range(args.observer_steps):
                if delivery_task:
                    expected_dsa.append(delivery.dsa_result(
                        producer_layer_ids=arrays["dsa_producer_layer_ids"],
                        selected_positions=arrays["dsa_selected_positions"][step],
                        selected_valid_counts=arrays["dsa_selected_valid_counts"][step],
                        selected_scores=arrays["dsa_selected_scores"][step],
                        decode_position=int(record["prompt_length"]) + step,
                        step=step,
                    ))
                elif long_context is not None:
                    expected_dsa.append(
                        compare_ws32_dsa_within_engine(
                            producer_layer_ids=arrays["dsa_producer_layer_ids"],
                            selected_positions=arrays["dsa_selected_positions"][step],
                            selected_valid_counts=arrays[
                                "dsa_selected_valid_counts"
                            ][step],
                            selected_scores=arrays["dsa_selected_scores"][step],
                            decode_position=int(record["prompt_length"]) + step,
                            step=step,
                            expected_producer_layer_ids=arrays["dsa_producer_layer_ids"],
                        )
                    )
                else:
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
                            adjudication=dsa_adjudication,
                        )
                    )
            if record.get("checkpoint_transport") != args.checkpoint_transport:
                raise SystemExit(f"WS32 checkpoint transport drifted rank {rank}")
            if not _same(record.get("dsa_adjudication"), expected_dsa_adjudication):
                raise SystemExit(f"WS32 DSA adjudication binding drifted rank {rank}")
            if not _same(dsa, expected_dsa):
                raise SystemExit(f"WS32 DSA recomputation drifted rank {rank}")
            alarm_steps = [
                step
                for step, item in enumerate(dsa)
                if item.get("adjudication", {}).get("alarm_events")
            ]
            if alarm_steps and not args.later_event_alarm_acknowledged:
                # Spec §21.2: a later-event alarm requires a GATE_D_LESSONS entry before promotion.
                raise SystemExit(
                    "WS32 later-event divergence alarm at steps "
                    f"{alarm_steps} requires an acknowledged lessons entry before sealing"
                )
            if alarm_steps and rank == 0:
                # The acknowledgement must be bound to evidence: the committed divergence profile
                # and the pin whose GATE_D_LESSONS entry names this run tag.
                profile = args.later_event_alarm_profile
                if (
                    profile is None
                    or not profile.is_file()
                    or _digest_file(profile) != args.later_event_alarm_profile_sha256
                    or not re.fullmatch(r"[0-9a-f]{40}", args.later_event_alarm_lessons_pin or "")
                    # Bound to the pin whose enforcement this seal declares:
                    # the recovery pin when there is one, otherwise the run's
                    # own. Both are required to be published, so the lessons
                    # entry the acknowledgement rests on is published too.
                    or args.later_event_alarm_lessons_pin
                    != (args.recovery_code_hash or args.code_hash)
                ):
                    raise SystemExit("WS32 alarm acknowledgement is not bound to a profile record and lessons pin")
                lessons = subprocess.run(
                    ["/usr/bin/git", "-C", str(REPO), "show", f"{args.later_event_alarm_lessons_pin}:docs/greenfield/GATE_D_LESSONS.md"],
                    check=False, capture_output=True, text=True,
                )
                if lessons.returncode != 0 or args.tag not in lessons.stdout:
                    raise SystemExit("WS32 alarm acknowledgement pin lacks a GATE_D_LESSONS entry naming this run")
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
    batched_memory = None
    if prefill_mode == PREFILL_MODE and args.mode == "numerical":
        batched_memory = _require_batched_fleet_memory(records, ordered_captures, physical_mesh)

    summary: dict[str, Any] = {
        "artifact_kind": "greenfield_ws32_short_decoder_fleet",
        # What the §21.2 enforcement was actually run with, so a reviewer can
        # compare it against the reviewed branch rather than trust the tree it
        # happened to be sealed from.
        **({} if enforcement_surface is None else {"enforcement_surface": enforcement_surface}),
        **common,
        "context_label": args.context_label,
        "graph_sha256": {
            graph: {
                "optimized_hlo_sha256": report["optimized_hlo_sha256"],
                "stablehlo_sha256": report["stablehlo_sha256"],
            }
            for graph, report in sorted(first_graphs.items())
        },
        **({"graph_source_location_identity": {
            graph: report["source_location_identity"]
            for graph, report in sorted(first_graphs.items())
        }} if prefill_mode == PREFILL_MODE else {}),
        "checkpoint_transport": args.checkpoint_transport,
        "dsa_adjudication": expected_dsa_adjudication,
        "evidence_layout": args.evidence_layout,
        "mode": args.mode,
        "performance_claim": args.mode == "numerical",
        "recovery_code_hash": args.recovery_code_hash or None,
        "run_tag": args.tag,
        "schema_version": 1,
        "status": "HLO_ACQUIRED" if args.mode == "acquire" else "SUCCESS",
        "strategy_nd_dense_overlay_manifest_file_sha256": overlay_pins[1],
        "strategy_nd_dense_overlay_manifest_sha256": overlay_pins[0],
        "strategy_nd_dense_overlay_success_file_sha256": overlay_pins[2],
        **({"batched_fleet_memory": batched_memory} if batched_memory is not None else {}),
    }
    if args.mode == "numerical":
        sys.path.insert(0, str(REPO / "scripts" / "analysis"))
        import parse_xplane

        xplane = parse_xplane.aggregate_fleet(
            args.run_dir / "traces",
            step_module_re=_decode_step_module_re(exact_dsa=bool(args.exact_dsa)),
        )
        if xplane["n_files"] != 8 or xplane["n_cores"] != 64 or xplane["steps_per_core"] != args.trace_steps:
            raise SystemExit("WS32 fleet XPlane coverage drifted")
        critical = np.max(np.asarray(all_samples, dtype=np.float64), axis=0)
        peak_hbm, minimum_headroom = (
            _peak(records) if batched_memory is None else (
                batched_memory["final_lifetime_peak_bytes"],
                batched_memory["minimum_final_headroom_bytes"],
            )
        )
        summary.update(
            {
                "cache_write_probe": records[0]["cache_write_probe"],
                "dsa_steps": records[0]["dsa_steps"],
                "capacity_measurement": (
                    # §23.3 Step C is the SEALED SHORT workload re-run at a long
                    # capacity. A §23.5 run's capacity is intrinsic to its prompt,
                    # so labelling it a capacity measurement would misdescribe it.
                    None
                    if long_context is not None
                    or args.context_capacity == DEFAULT_CONTEXT_CAPACITY
                    else {"context_capacity": int(args.context_capacity), "default": DEFAULT_CONTEXT_CAPACITY}
                ),
                "main_rope_table": records[0].get("main_rope_table"),
                "prefill_chunk_length": records[0]["prefill_chunk_length"],
                "rotary_diagnostic": (
                    None
                    if records[0].get("rotary_diagnostic") is None
                    else {
                        key: records[0]["rotary_diagnostic"][key]
                        for key in ("failing_cells", "positions", "record_sha256", "script_sha256", "verdict")
                    }
                ),
                "prefill_execution": {
                    **records[0]["prefill_execution"],
                    "fleet_total_seconds_max": max(
                        float(record["prefill_execution"][
                            "request_prefill_seconds" if prefill_mode == PREFILL_MODE else "total_seconds"
                        ])
                        for record in records
                    ),
                },
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
                "verified_generated_token_count": (
                    # §23.5 verifies NO token at these lengths. L7 verifies an
                    # extracted passkey; the count of ids its criterion happened
                    # to detokenise is a cardinality, independent of whether any
                    # id matched, and reporting it here would read as a
                    # raw-token claim the run is forbidden to make.
                    None
                    if long_context is not None or delivery_task
                    else int(records[0]["token_comparison"]["compared_count"])
                ),
                "xplane": xplane,
            }
        )
        alarm_summary = (
            # §21.2's later-event alarm compares the engine against the legacy
            # capture; §23.5 runs have none, and their within-engine step
            # records carry no adjudication for it to read.
            None
            if long_context is not None or delivery_task
            else _later_event_alarm_summary(
                records[0]["dsa_steps"], oracle.producer_layer_ids.tolist()
            )
        )
        if alarm_summary is not None:
            alarm_summary.update(
                {
                    "acknowledged": bool(args.later_event_alarm_acknowledged),
                    "lessons_pin": args.later_event_alarm_lessons_pin or None,
                    "profile_path": (
                        str(args.later_event_alarm_profile.resolve().relative_to(repository_root))
                        if args.later_event_alarm_profile is not None
                        else None
                    ),
                    "profile_sha256": (
                        args.later_event_alarm_profile_sha256
                        if args.later_event_alarm_profile is not None
                        else None
                    ),
                }
            )
        summary["later_event_alarm"] = alarm_summary
        if delivery_task:
            basis = ["S26_TASK_PASSKEY_EXACT", "DSA_SELECTED_ROW_STRUCTURE",
                     "DSA_KERNEL_SEMANTICS_BY_SOURCE_AND_TESTS",
                     "UNSELECTED_SCORE_ROWS_NOT_RECOMPUTED",
                     "STATE_CACHE_EXACT_STRUCTURE", "NO_CROSS_ORACLE",
                     "MODEL_CARD_QUALITY_NOT_ESTABLISHED"]
        elif long_context is not None:
            # §23.5: no legacy DSA capture exists at these lengths, so no
            # cross-oracle claim may appear, and "raw tokens exact" is forbidden
            # because the legacy stored text rather than ids.
            basis = ["DSA_WITHIN_ENGINE_EXACT", "STATE_CACHE_EXACT_STRUCTURE", "NO_CROSS_ORACLE"]
            if long_context.kind == "passkey":
                basis.insert(0, "PASSKEY_EXACT")
            else:
                basis.append("NO_CORRECTNESS_ORACLE")
        else:
            basis = ["RAW_TOKENS_EXACT", "DSA_WITHIN_ENGINE_EXACT", "STATE_CACHE_EXACT_STRUCTURE"]
        if delivery_task:
            pass  # §26 does not claim cross-engine continuation or DSA bits.
        elif long_context is not None:
            # Nothing cross-oracle applies: no capture exists to be exact
            # against, and no §21.2 record may be bound (refused at load).
            if alarm_summary is not None:
                basis.append("LATER_EVENT_ALARM_ACKNOWLEDGED_WITH_LESSONS_ENTRY")
        elif expected_dsa_adjudication is None:
            basis.append("DSA_CROSS_ORACLE_EXACT_ALL_EVENTS")
        else:
            basis.append("DSA_EVENT0_EXACT")
            basis.append(
                f"DSA_EVENT{expected_dsa_adjudication['event_index']}_ADJUDICATED_S21_2"
            )
            basis.append("LATER_EVENTS_RECORDED_NOT_ADJUDICATED")
            basis.append("DEEP_LAYER_TENSORS_NOT_BOUNDED_IN_THIS_RUN")
            if alarm_summary is not None:
                basis.append("LATER_EVENT_ALARM_ACKNOWLEDGED_WITH_LESSONS_ENTRY")
        basis.append("PROTECTED_WALL_TRACE_HBM")
        if prefill_mode == PREFILL_MODE:
            basis.extend(["BATCHED_PREFILL_OWN_SHORT_NUMERICAL", "PREFILL_SPEEDUP_NOT_ESTABLISHED", "DELIVERED_TTFT_NOT_MEASURED"])
        if summary.get("capacity_measurement") is not None:
            basis.append(f"CAPACITY_MEASUREMENT_{summary['capacity_measurement']['context_capacity']}")
        if long_context is not None:
            basis.append(f"LONG_CONTEXT_{args.context_label.upper()}")
            basis.append(f"CONTEXT_CAPACITY_{int(args.context_capacity)}")
        if summary.get("main_rope_table") is not None:
            basis.append("MAIN_ROTARY_HOST_TABLE_LEGACY_FAITHFUL")
        if summary.get("rotary_diagnostic") is not None:
            basis.append(f"ROTARY_LONG_POSITION_DIAGNOSTIC_{summary['rotary_diagnostic']['verdict']}")
        summary["classification"] = ";".join(basis)
    else:
        summary["later_event_alarm"] = None
        summary["classification"] = "HLO_ACQUIRED_ONLY"
    summary["summary_sha256"] = sha256(_canonical(summary)).hexdigest()
    _write_once(args.output, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


def _enforcement_surface_identity(repository_root: Path) -> dict[str, str]:
    """The committed object ids of the code that decides what a seal accepts.

    The clean-surface check stops an uncommitted edit, but a seal driven from a
    different checkout — or from a scratch branch carrying a widened registry —
    would still have a clean tree. Recording the object id of each surface path
    puts what the seal was produced with into the sealed record, where it can be
    compared against the reviewed branch.
    """

    identity: dict[str, str] = {}
    for relative in _ENFORCEMENT_SURFACE:
        try:
            result = subprocess.run(
                [_GIT, "-C", str(repository_root), "rev-parse", f"HEAD:{relative}"],
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise SystemExit(f"WS32 seal cannot identify its enforcement surface: {error}")
        if result.returncode != 0:
            raise SystemExit(
                f"WS32 enforcement surface path is not committed at HEAD: {relative}"
            )
        identity[relative] = result.stdout.strip()
    head = subprocess.run(
        [_GIT, "-C", str(repository_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if head.returncode != 0:
        raise SystemExit("WS32 seal cannot identify its own checkout")
    identity["HEAD"] = head.stdout.strip()
    return identity


def _require_reviewed_enforcement(
    repository_root: Path,
    surface: Mapping[str, str],
    *,
    code_hash: str,
    recovery_code_hash: str,
    reviewed_ref: str,
) -> None:
    """The enforcement must be the reviewed enforcement, at a published pin.

    A clean tree proves only that the checkout matches its own HEAD. Every pin
    this seal declares — the run's, and the recovery pin when there is one —
    must be contained in the reviewed remote branch, and the enforcement surface
    must be the surface committed at one of them.

    This establishes that the enforcement code is in PUBLISHED history. It does
    not establish that it was reviewed: the operator pushes to that branch. See
    §21.2's ceiling paragraph; the residual assurance is a human reading the
    registry entry.
    """

    pins = [code_hash] + ([recovery_code_hash] if recovery_code_hash else [])
    for pin in pins:
        reachable = subprocess.run(
            [_GIT, "-C", str(repository_root), "merge-base", "--is-ancestor", pin, reviewed_ref],
            capture_output=True,
            text=True,
        )
        if reachable.returncode != 0:
            raise SystemExit(
                f"WS32 protected seal requires the pin {pin} to be published on "
                f"{reviewed_ref}; local history alone does not prove publication"
            )
    for pin in pins:
        expected = {}
        for relative in _ENFORCEMENT_SURFACE:
            result = subprocess.run(
                [_GIT, "-C", str(repository_root), "rev-parse", f"{pin}:{relative}"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                break
            expected[relative] = result.stdout.strip()
        else:
            if all(surface.get(key) == value for key, value in expected.items()):
                return
    raise SystemExit(
        "WS32 protected seal runs enforcement code that is not the code committed at "
        f"the run's pin {code_hash}"
        + (f" or the declared recovery pin {recovery_code_hash}" if recovery_code_hash else "")
    )


def _require_clean_worktree(repository_root: Path) -> None:
    """Refuse to seal a protected run from a modified enforcement surface.

    The enforcement of §21.2 items 3-4 reads `REFERENCE_ROWS` and the
    adjudication arithmetic out of this repository's source. Both are ordinary
    files: an operator could widen the registry in the working tree, seal, and
    revert. Requiring the tree to be clean puts any such edit into history,
    which is what "reviewed" means here.
    """

    code = [item for item in _ENFORCEMENT_SURFACE if item not in _TRACKED_ONLY_SURFACE]
    modified = _git_lines(
        repository_root, ["status", "--porcelain", "--"] + code, "verify the working tree"
    )
    modified += _git_lines(
        repository_root,
        ["diff", "--name-only", "HEAD", "--"] + list(_TRACKED_ONLY_SURFACE),
        "verify the artifact tree",
    )
    # `--assume-unchanged` and `--skip-worktree` make git report a modified file
    # as clean, so the flags themselves are a refusal: they hide exactly the edit
    # this check exists to catch, and they leave nothing in history. `ls-files -v`
    # tags assume-unchanged by LOWERCASING the tag letter and skip-worktree as a
    # capital `S`; only `H` (plain cached) is acceptable here.
    hidden = [
        line[2:]
        for line in _git_lines(
            repository_root,
            ["ls-files", "-v", "--"] + list(_ENFORCEMENT_SURFACE),
            "read the index",
        )
        if line[:1] != "H"
    ]
    if hidden:
        raise SystemExit(
            "WS32 protected seal refuses an enforcement surface with hidden index flags "
            "(--assume-unchanged / --skip-worktree): " + ", ".join(sorted(hidden)[:8])
        )
    if modified:
        raise SystemExit(
            "WS32 protected seal requires an unmodified enforcement surface; "
            "correctness and evidence validation depend on it. Modified: "
            + ", ".join(sorted(_status_path(item) for item in modified)[:8])
        )


def _status_path(line: str) -> str:
    """The path out of a `status --porcelain` or `diff --name-only` line."""

    text = line[3:] if len(line) > 3 and line[2] == " " else line
    return text.split(" -> ")[-1].strip().strip('"')


def _git_lines(repository_root: Path, arguments: list[str], purpose: str) -> list[str]:
    try:
        result = subprocess.run(
            [_GIT, "-C", str(repository_root), *arguments], capture_output=True, text=True
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise SystemExit(f"WS32 seal cannot {purpose}: {error}")
    if result.returncode != 0:
        raise SystemExit(f"WS32 seal cannot {purpose}: {result.stderr.strip()}")
    return [line for line in result.stdout.splitlines() if line.strip()]


def _require_imports_come_from(repository_root: Path) -> None:
    """The modules that enforce §21.2 must be this repository's own.

    Checking one tree for modifications while importing the arithmetic from
    another is the same defect as not checking at all.
    """

    from glm_tpu.greenfield import types as greenfield_types
    from glm_tpu.greenfield.kernels.reference import rotary as reference_rotary
    from glm_tpu.greenfield.validation import (
        long_context_oracle,
        ws32_first_divergent_event,
        ws32_short_context,
    )

    root = Path(repository_root).resolve()
    for module in (
        ws32_short_context,
        ws32_first_divergent_event,
        # §23.5: the label-to-identity table and the profile refusal.
        long_context_oracle,
        reference_rotary,
        greenfield_types,
    ):
        origin = Path(getattr(module, "__file__", "")).resolve()
        if not origin.is_relative_to(root):
            raise SystemExit(
                f"WS32 §21.2 enforcement is imported from outside the sealed repository: "
                f"{origin}"
            )


def _make_pre_registration_check(args: Any, repository_root: Path):
    """Return the check that proves §21.2 pre-registration from the run's pin.

    A module-level factory rather than a closure inside ``_validate`` so the
    property can be exercised by a test instead of asserted about the source.
    """

    def committed_in_run_pin(relative: str, label: str) -> None:
        """Require an artifact to be committed in the pin the run executed at.

        The run ran at ``--code-hash``, so an artifact present in that commit
        with these exact bytes existed before the run produced the data it
        judges. Checking ``HEAD`` would not do it: HEAD moves after the run.
        The sealed Gate D record satisfies this — its blob at pin 4286509 is
        the blob on disk.
        """

        try:
            committed = subprocess.run(
                [_GIT, "-C", str(repository_root), "rev-parse", f"{args.code_hash}:{relative}"],
                capture_output=True,
                text=True,
            )
            working = subprocess.run(
                [_GIT, "-C", str(repository_root), "hash-object", "--", relative],
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.SubprocessError) as error:
            # No git means the property cannot be established, which is a
            # refusal, not a traceback and not a pass.
            raise SystemExit(f"WS32 {label} commitment cannot be checked: {error}")
        if committed.returncode != 0:
            raise SystemExit(
                f"WS32 {label} is not committed in the run's own pin {args.code_hash}: "
                f"{relative}; a record written after the run is not a pre-registration"
            )
        if working.returncode != 0 or committed.stdout.strip() != working.stdout.strip():
            raise SystemExit(
                f"WS32 {label} differs from the blob committed at {args.code_hash}: {relative}"
            )

    return committed_in_run_pin


def _require_ranks_agree(
    array_records: Mapping[str, Any], records: list[Mapping[str, Any]], *, rank: int
) -> None:
    """Every rank's observed arrays must be rank 0's, value for value."""

    if rank and not _same(array_records, records[0]["numerical_tensors"]["arrays"]):
        raise SystemExit(f"WS32 numerical tensor values disagree rank {rank}")


def _long_context_token_result(
    observed_tokens: list[int], long_context: Any, *, tokenizer_root: Path | None
) -> dict[str, Any]:
    """Recompute §23.5's token result. Shared verbatim with the runner.

    Imported from the runner rather than reimplemented: two copies of a
    correctness criterion drift, and the sealer's job is to recompute the
    runner's claim with the SAME rule, not a similar one.
    """

    import importlib.util

    path = REPO / "scripts" / "greenfield" / "run_short_decoder_ws32.py"
    specification = importlib.util.spec_from_file_location("ws32_runner_rules", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module._long_context_token_result(
        observed_tokens, long_context, tokenizer_root=tokenizer_root
    )


def _rank0_dsa_arrays(run_dir: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """Rank 0's DSA observations, bound to rank 0's own runner record."""

    path = run_dir / "fleet" / "runner.rank0.npz"
    tensor_record = record.get("numerical_tensors")
    if not path.is_file():
        raise SystemExit(f"WS32 rank-0 numerical tensor file is missing: {path}")
    if (
        not isinstance(tensor_record, Mapping)
        or tensor_record.get("filename") != path.name
        or tensor_record.get("byte_count") != path.stat().st_size
        or tensor_record.get("sha256") != _digest_file(path)
    ):
        raise SystemExit("WS32 rank-0 numerical tensor file is not bound to its record")
    wanted = (
        "dsa_producer_layer_ids",
        "dsa_selected_positions",
        "dsa_selected_scores",
        "dsa_selected_valid_counts",
    )
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in wanted if name not in archive.files]
        if missing:
            raise SystemExit(f"WS32 rank-0 numerical tensor keys missing: {missing}")
        return {name: np.ascontiguousarray(archive[name]) for name in wanted}


def _rederive_ws32_adjudication(
    *,
    arrays: Mapping[str, Any],
    oracle: Any,
    adjudication: Any,
    repository_root: Path,
    rank: int,
) -> None:
    """Recompute §21.2 items 3-4 for the adjudicated event and require a pass.

    The pre-registered record supplies the reviewed FP64 reference row and the
    divergence it predicted; the numbers come from this run's own observations.
    A hand-written analysis asserting six passing checks changes nothing here.
    """

    from glm_tpu.greenfield.validation.ws32_first_divergent_event import (
        AdjudicationError,
        adjudicate_first_divergent_event,
        event_arrays,
        oracle_event_arrays,
    )

    step = adjudication.step
    event = adjudication.event_index
    relative = adjudication.reference_row_path
    if relative is None:
        raise SystemExit(
            "WS32 adjudication record names no FP64 reference row, so §21.2 items 3-4 "
            "cannot be re-derived"
        )
    reference_path = repository_root / relative
    try:
        reference = np.load(reference_path, allow_pickle=False).astype(np.float64)
    except (OSError, ValueError) as error:
        raise SystemExit(f"WS32 adjudication reference row is unreadable: {error}")
    engine_positions, engine_scores = event_arrays(
        selected_positions=arrays["dsa_selected_positions"],
        selected_scores=arrays["dsa_selected_scores"],
        selected_valid_counts=arrays["dsa_selected_valid_counts"],
        step=step,
        event=event,
    )
    oracle_positions, oracle_scores = oracle_event_arrays(
        {
            "selected_positions": oracle.selected_positions,
            "selected_scores": oracle.selected_scores,
            "valid_counts": oracle.valid_counts,
        },
        step=step,
        event=event,
    )
    try:
        recomputed = adjudicate_first_divergent_event(
            oracle_positions=oracle_positions,
            oracle_scores=oracle_scores,
            engine_positions=engine_positions,
            engine_scores=engine_scores,
            reference=reference,
            producer_layer_id=int(
                np.asarray(arrays["dsa_producer_layer_ids"])[event]
            ),
            step=step,
            event=event,
            decode_position=adjudication.decode_position,
            expected_producer_layer_id=adjudication.producer_layer_id,
        )
    except AdjudicationError as error:
        raise SystemExit(
            f"WS32 §21.2 re-derivation refuses the adjudicated event rank {rank}: {error}"
        )
    if recomputed["verdict"] != "PASS":
        failed = sorted(
            name for name, item in recomputed["checks"].items() if not item["pass"]
        )
        raise SystemExit(
            f"WS32 §21.2 items 3-4 fail on this run rank {rank}: {failed}"
        )
    if (
        tuple(recomputed["expected_only"]) != adjudication.expected_only
        or tuple(recomputed["observed_only"]) != adjudication.observed_only
    ):
        raise SystemExit(
            f"WS32 §21.2 re-derivation disagrees with the pre-registered divergence rank {rank}"
        )


def _later_event_alarm_summary(
    dsa_steps: list[dict[str, Any]], producer_layer_ids: list[int]
) -> dict[str, Any] | None:
    """Summarize spec §21.2 later-event alarms across observed steps (None when no alarm)."""
    events_by_step: dict[str, list[int]] = {}
    maximum: dict[str, Any] | None = None
    for step, item in enumerate(dsa_steps):
        adjudication = item.get("adjudication") or {}
        alarms = list(adjudication.get("alarm_events") or [])
        if alarms:
            events_by_step[str(step)] = alarms
        for entry in adjudication.get("recorded_divergence_sizes") or []:
            size = int(entry["symmetric_difference"])
            if maximum is None or size > maximum["symmetric_difference"]:
                maximum = {
                    "event_index": int(entry["event_index"]),
                    "producer_layer_id": int(producer_layer_ids[int(entry["event_index"])]),
                    "step": step,
                    "symmetric_difference": size,
                }
    if not events_by_step:
        return None
    return {
        "alarmed_step_count": len(events_by_step),
        "events_by_step": events_by_step,
        "maximum_symmetric_difference": maximum,
        "observed_step_count": len(dsa_steps),
        "rule": "spec §21.2: |E Δ O| > 1024 at a recorded later event is a diagnostic alarm requiring a GATE_D_LESSONS entry before promotion",
    }


def _decode_step_module_re(*, exact_dsa: bool) -> str:
    """Anchored XLA module name of one traced decode step.

    The WS32 decoder shard_maps ``execute_exact_body`` on the exact-DSA path and
    ``execute_body`` otherwise (``glm_tpu/greenfield/runtime/ws32_decoder.py``);
    JAX names the module ``jit_<body>(<fingerprint>)``.  Selecting the wrong
    body yields zero decode steps and the fleet XPlane aggregation refuses.
    """
    body = "execute_exact_body" if exact_dsa else "execute_body"
    return rf"^jit_{body}\("


def _linter_kind(graph: str) -> str:
    """Both prefill programs (chunk and tail) carry the ``prefill`` HLO contract."""
    return "prefill" if graph.startswith("prefill") else graph


def _require_prefill_execution(
    record: Mapping[str, Any],
    *,
    mode: str,
    prompt_length: int,
    rank: int,
    expected_chunk: int | None = None,
    prefill_mode: str = SERIAL_PREFILL_MODE,
) -> None:
    """Spec §23.2: chunked exact prefill accounting must be complete and consistent."""
    if prefill_mode == PREFILL_MODE:
        _require_batched_execution(record, mode=mode, prompt_length=prompt_length, expected_chunk=expected_chunk)
        return
    if record.get("prefill_mode", SERIAL_PREFILL_MODE) != SERIAL_PREFILL_MODE:
        raise SystemExit("serial prefill accounting cannot authorize batched evidence")
    chunk = record.get("prefill_chunk_length")
    if type(chunk) is not int or chunk <= 0 or (
        expected_chunk is not None and chunk != expected_chunk
    ):
        raise SystemExit(f"WS32 prefill chunk length drifted at rank {rank}")
    full_chunks = (prompt_length - 1) // chunk
    tail_length = prompt_length - full_chunks * chunk
    execution = record.get("prefill_execution")
    if mode != "numerical":
        if execution is not None:
            raise SystemExit(f"WS32 acquisition recorded a prefill execution at rank {rank}")
        return
    if type(execution) is not dict or set(execution) != {
        "budget_seconds",
        "chunk_length",
        "chunk_wall_seconds",
        "full_chunks",
        "projected_total_seconds_max",
        "repaired_index_installed",
        "tail_length",
        "tail_wall_seconds",
        "total_seconds",
    }:
        raise SystemExit(f"WS32 prefill execution schema drifted at rank {rank}")
    walls = execution["chunk_wall_seconds"]
    positive = lambda value: type(value) is float and math.isfinite(value) and value > 0
    if (
        execution["chunk_length"] != chunk
        or execution["full_chunks"] != full_chunks
        or execution["tail_length"] != tail_length
        or type(walls) is not list
        or len(walls) != full_chunks
        or any(not positive(value) for value in walls)
        or not positive(execution["tail_wall_seconds"])
        or not positive(execution["total_seconds"])
        or not positive(execution["budget_seconds"])
        or type(execution["projected_total_seconds_max"]) is not float
        or execution["projected_total_seconds_max"] > execution["budget_seconds"]
        or execution["total_seconds"] > execution["budget_seconds"]
        or execution["total_seconds"] < sum(walls) + execution["tail_wall_seconds"]
        or type(execution["repaired_index_installed"]) is not bool
    ):
        raise SystemExit(f"WS32 prefill execution accounting drifted at rank {rank}")


def _require_batched_execution(
    record: Mapping[str, Any], *, mode: str, prompt_length: int,
    expected_chunk: int | None, long_context_label: str | None = None,
) -> None:
    """Own bounded prompt accounting and actual first generated token binding."""
    from glm_tpu.greenfield.validation.ws32_prefill import validate_execution_record
    from glm_tpu.greenfield.validation.ws32_prefill_admission import (
        short_plan, short_budget,
    )
    from scripts.greenfield import ws32_delivery_runtime as long_runtime

    is_long = record.get("batched_prefill_profile") == long_runtime.PROFILE
    if long_context_label is not None and not is_long:
        raise ValueError("long execution context requires explicit long profile")
    plan = long_runtime.programs.long_plan(long_context_label) if is_long else short_plan(record.get("batched_prefill_profile"))
    options = {"state_ownership_contract": long_runtime.programs.state_ownership(long_context_label)} if is_long else {}
    if (
        mode != "numerical" or record.get("prefill_mode") != PREFILL_MODE
        or not _same(record.get("batched_prefill_plan"), plan.identity(**options))
        or type(prompt_length) is not int or prompt_length != plan.prompt_length
        or type(expected_chunk) is not int or expected_chunk != plan.block_rows
        or type(record.get("prefill_chunk_length")) is not int
        or record["prefill_chunk_length"] != plan.block_rows
        or type(record.get("context_capacity")) is not int
        or record["context_capacity"] != plan.context_capacity
    ):
        raise ValueError("batched prefill accounting requires fixed long numerical mode" if is_long
                         else "batched prefill accounting requires fixed short numerical mode")
    execution = record["prefill_execution"]
    validate_execution_record(execution, plan, **options)
    if is_long:
        long_runtime.require_role_record(execution["memory_admission"], long_context_label)
    budget = long_runtime.budget_seconds(long_context_label) if is_long else short_budget(record.get("batched_prefill_profile"))
    if execution["budget_seconds"] != budget:
        raise ValueError("batched prefill numerical cost ceiling drifted")
    observed = record.get("observed_generated_token_ids")
    if (
        type(observed) is not list or not observed or type(observed[0]) is not int
        or observed[0] != execution["first_token_ready"]
    ):
        raise ValueError("batched prefill first token differs from observed continuation")


def _require_main_rope_table(
    record: Mapping[str, Any],
    *,
    enabled: bool,
    context_capacity: int,
    rank: int,
    first: Mapping[str, Any],
) -> None:
    """Spec §23.8: the legacy-faithful main-attention rotary table must be present
    iff declared, built by this pin's own construction for this capacity, and
    identical across ranks."""
    value = record.get("main_rope_table")
    if not enabled:
        if value is not None:
            raise SystemExit(f"WS32 main rotary table present but not declared at rank {rank}")
        return
    if type(value) is not dict or set(value) != {
        "bytes_per_device",
        "rotary_dim",
        "rows",
        "sha256",
        "theta",
    }:
        raise SystemExit(f"WS32 main rotary table schema drifted at rank {rank}")
    from glm_tpu.greenfield.kernels.reference.rotary import (
        build_rotary_table_host,
        rotary_table_sha256,
    )
    from glm_tpu.greenfield.runtime import WS32_MAIN_ROPE_THETA

    from glm_tpu.greenfield.types import ModelGeometry

    rotary_dim = ModelGeometry.from_hf_config(
        json.loads((REPO / "configs/glm-5.2-fp8-config.json").read_text(encoding="utf-8"))
    ).qk_rope_head_dim
    if (
        value["rows"] != context_capacity
        or value["rotary_dim"] != rotary_dim
        or value["theta"] != WS32_MAIN_ROPE_THETA
        or value["bytes_per_device"] != context_capacity * rotary_dim * 2
    ):
        raise SystemExit(f"WS32 main rotary table identity drifted at rank {rank}")
    expected = rotary_table_sha256(
        build_rotary_table_host(
            context_capacity, rotary_dim=rotary_dim, theta=WS32_MAIN_ROPE_THETA
        )
    )
    if value["sha256"] != expected:
        raise SystemExit(f"WS32 main rotary table bytes drifted at rank {rank}")
    if rank and not _same(value, first.get("main_rope_table")):
        raise SystemExit(f"WS32 main rotary table disagrees at rank {rank}")


def _require_rotary_diagnostic(
    record: Mapping[str, Any], *, enabled: bool, rank: int, first: Mapping[str, Any]
) -> None:
    """Spec §23.8: the declared side program's record must be present iff enabled,
    self-consistent, produced by the committed script, and identical across ranks."""
    from glm_tpu.greenfield.validation.rotary_diagnostic import (
        script_sha256,
        verify_rotary_diagnostic_record,
    )

    value = record.get("rotary_diagnostic")
    if not enabled:
        if value is not None:
            raise SystemExit(f"WS32 rotary diagnostic present but not declared at rank {rank}")
        return
    try:
        # Protocol-pinned and TPU-only: a CPU record or a shortened window can never seal.
        verify_rotary_diagnostic_record(
            value, expected_script_sha256=script_sha256(), expected_backend="tpu", pinned=True
        )
    except ValueError as error:
        raise SystemExit(f"WS32 rotary diagnostic invalid at rank {rank}: {error}") from None
    if rank and not _same(value, first.get("rotary_diagnostic")):
        raise SystemExit(f"WS32 rotary diagnostic disagrees at rank {rank}")


def _run_environment(summary: dict[str, Any]) -> dict[str, Any]:
    """The exact env_json identity shared by DB publication and rollback.

    Rollback resolves the run by equality against this dictionary, so both
    sides must derive it from one function or a new publication key silently
    makes the run unrecoverable.
    """
    return {
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
        "strategy_nd_dense": summary["strategy_nd_dense"],
        "strategy_nd_dense_overlay_manifest_sha256": summary[
            "strategy_nd_dense_overlay_manifest_sha256"
        ],
        "token_oracle_manifest_sha256": summary["token_oracle_manifest_sha256"],
        "token_oracle_success_sha256": summary["token_oracle_success_sha256"],
        "xla_python_client_mem_fraction": summary[
            "xla_python_client_mem_fraction"
        ],
        "checkpoint_transport": summary.get("checkpoint_transport"),
        "classification": summary.get("classification"),
        "dsa_adjudication": summary.get("dsa_adjudication"),
        "later_event_alarm": summary.get("later_event_alarm"),
        "recovery_code_hash": summary.get("recovery_code_hash"),
        "prefill_chunk_length": summary.get("prefill_chunk_length"),
        "main_rope_table": summary.get("main_rope_table"),
        "capacity_measurement": summary.get("capacity_measurement"),
        "rotary_diagnostic": summary.get("rotary_diagnostic"),
        **({"validation_contract": summary["validation_contract"]}
           if "validation_contract" in summary else {}),
        # Preserve historical DB rollback identity exactly when the field is absent.
        **({"prefill_mode": require_prefill_mode(summary["prefill_mode"])}
           if "prefill_mode" in summary else {}),
        **({key: summary[key] for key in (
            "batched_prefill_profile", "batched_prefill_plan", "batched_prefill_acquisition",
            "prefill_memory_reserve_bytes", "prefill_budget_seconds",
        )} if summary.get("prefill_mode") == PREFILL_MODE else {}),
        # §23.5: which sealed long-context capture this run answered, so the DB
        # row is bound to the profile and not merely to the context label. Rows
        # are resolved for rollback by exact equality on this dictionary, so the
        # key is present only when there IS a long context: adding it
        # unconditionally would make every previously published row
        # unresolvable, and rollback reports "nothing to do" rather than failing.
        **(
            {}
            if summary.get("long_context") is None
            else {"long_context": summary["long_context"]}
        ),
    }


def _run_rows(
    summary: dict[str, Any]
) -> tuple[str, str, str, int | None, float | None]:
    """The exact run note, item id and gold shared by DB publication and rollback.

    Rollback refuses any row that does not equal what publication wrote, so
    both sides must derive the wording from this one function.
    """
    adjudicated = summary.get("dsa_adjudication") is not None
    capacity = summary.get("capacity_measurement")
    long_context = summary.get("long_context")
    if summary.get("validation_contract") == delivery.CONTRACT:
        return (
            "Protected §26 batched8K task smoke: exact extracted passkey; selected-row DSA structure, "
            "state/cache/HLO/HBM/XPlane and profiler-free wall. No cross-engine token/DSA equality "
            "or model-card quality claim; full score rows are not recomputed.",
            "s26_batched_8k_task_passkey_state_cache",
            "Extracted passkey equals sealed gold; structural protections pass.",
            1, 1.0,
        )
    if long_context:
        # Spec §23.5: an L7/L8 record. There is no legacy capture to be exact
        # against, so the wording must claim neither raw-token nor cross-oracle
        # exactness, and must say what the run's own criterion actually was.
        label = f"{long_context['kind']}"
        if long_context["kind"] == "passkey":
            note = (
                "Protected complete WS32 long-context decoder (spec §23.5 L7): extracted passkey equals gold at "
                f"depth {long_context['depth']}; within-engine DSA order/ties; state/cache/HLO/HBM/XPlane and "
                "profiler-free wall. No legacy token or DSA capture exists at this length: nothing here is a "
                "raw-tokens-exact or cross-oracle claim."
            )
            gold = (
                "Extracted passkey equal to the sealed legacy gold; within-engine DSA order/ties; "
                "state/cache/HLO/HBM/XPlane and protected wall. No cross-oracle claim."
            )
        else:
            note = (
                "Protected complete WS32 long-context decoder (spec §23.5 L8): NO correctness oracle exists at this "
                "length; within-engine DSA order/ties; state/cache/HLO/HBM/XPlane and profiler-free wall. Nothing "
                "here is a raw-tokens-exact or cross-oracle claim."
            )
            gold = (
                "No correctness oracle (spec §23.5 L8); within-engine DSA order/ties; "
                "state/cache/HLO/HBM/XPlane and protected wall."
            )
        item_id = (
            f"s23_5_long_context_{label}_row{long_context['item_row_id']}_"
            f"cap{summary['context_capacity']}"
        )
        # L8 has no correctness oracle, so it must not be recorded as correct.
        verdict = (1, 1.0) if long_context["kind"] == "passkey" else (None, None)
        return note, item_id, gold, verdict[0], verdict[1]
    note = (
        "Protected complete WS32 short-context decoder: exact tokens/DSA/state/cache/HLO/HBM/XPlane and profiler-free wall."
        if not adjudicated
        else "Protected complete WS32 short-context decoder (spec §21): exact raw tokens; DSA event 0 exact, "
        f"event {summary['dsa_adjudication']['event_index']} adjudicated against the pre-registered record, later events "
        "recorded"
        + (" (alarm acknowledged with a lessons entry)" if summary.get("later_event_alarm") else "")
        + "; within-engine DSA order/tails exact; state/cache/HLO/HBM/XPlane and profiler-free wall."
    )
    item_id = "gate_d_exact_token_dsa_state_cache" if not adjudicated else "gate_d_s21_exact_tokens_adjudicated_dsa_state_cache"
    if summary.get("prefill_mode") == PREFILL_MODE:
        item_id = "s24_batched_prefill_own_short_" + item_id
        from glm_tpu.greenfield.validation.ws32_prefill_admission import profile_is_paired
        if profile_is_paired(summary.get("batched_prefill_profile")):
            item_id = "paired_sort_" + item_id
        note = "Batched layer-major prefill, own short numerical evidence; no prefill speedup or delivered TTFT claim. " + note
    if capacity:
        # Spec §23.3 Step C: the same sealed workload at a long context capacity;
        # a measurement record, never a Gate D record.
        note = (
            f"Capacity measurement at context_capacity={capacity['context_capacity']} (spec §23.3 Step C, "
            "not a Gate D record): " + note
        )
        item_id = f"s23_capacity_measurement_cap{capacity['context_capacity']}_" + item_id
    gold = (
        "Exact sealed raw-token prefix, executing-program DSA set/ties, state/cache/HLO/HBM/XPlane and protected wall."
        if not adjudicated
        else "Exact sealed raw-token prefix; within-engine DSA order/ties; cross-oracle DSA exact at event 0 and equal to the "
        "pre-registered adjudicated divergence at the first divergent event, later events recorded; state/cache/HLO/HBM/XPlane and protected wall."
    )
    return note, item_id, gold, 1, 1.0


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
        env = _run_environment(summary)
        note, item_id, gold, correct, score = _run_rows(summary)
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
                note,
            ),
        )
        run_id = int(cursor.lastrowid)
        benchmark = f"greenfield_78layer_{summary['context_label']}_ws32"
        connection.execute(
            "INSERT INTO items(run_id,benchmark,item_id,asked_utc,prompt,gold,raw_output,extracted,correct,score,n_prompt_tokens,n_gen_tokens,latency_ms,seed,finish_reason,truncated) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                benchmark,
                item_id,
                now,
                f"Execute sealed {summary['context_label']} prompt through the complete WS32 decoder.",
                gold,
                json.dumps(summary, sort_keys=True),
                json.dumps(summary["observed_generated_token_ids"]),
                correct,
                score,
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
    expected_env = _run_environment(summary)
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
        (
            expected_note,
            expected_item_id,
            expected_gold,
            expected_correct,
            expected_score,
        ) = _run_rows(summary)
        expected_item = (
            benchmark,
            expected_item_id,
            created,
            f"Execute sealed {summary['context_label']} prompt through the complete WS32 decoder.",
            expected_gold,
            json.dumps(summary, sort_keys=True),
            json.dumps(summary["observed_generated_token_ids"]),
            expected_correct,
            expected_score,
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
                expected_note,
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
