#!/usr/bin/env python3
"""Protected-worker runner for the complete WS32 2K/8K decoder.

The shell orchestrator owns fleet locking, append-only remote publication,
results.db, and cleanup.  This process owns only one launch rank's exact JAX
runtime, checkpoint load, HLO proof, oracle comparison, timing, trace, and
compact cache witness.
"""

from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[2]
EXPECTED_WORKTREE = Path("/home/gianl/glm-tpu-topology-rewrite")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.benchmarking import (  # noqa: E402
    validate_ws32_decoder_hlo,
    validate_ws32_exact_dsa_materializer_hlo,
    validate_ws32_topology_fleet,
)
from glm_tpu.greenfield.checkpoint import (  # noqa: E402
    load_ws32_runtime_checkpoint,
    load_ws32_strategy_nd_dense_overlay,
    verify_ws32_runtime_checkpoint,
    verify_ws32_strategy_nd_dense_overlay,
)
from glm_tpu.greenfield.partitioning import inspect_source_inventory  # noqa: E402
from glm_tpu.greenfield.kernels.reference.rotary import (  # noqa: E402
    rotary_table_sha256,
)
from glm_tpu.greenfield.runtime import (  # noqa: E402
    Ws32DecoderConfig,
    bind_ws32_decoder_weights,
    build_ws32_decoder_program,
    build_ws32_exact_dsa_materializer_program,
    WS32_MAIN_ROPE_THETA,
    build_ws32_chunked_prefill_program,
    build_ws32_main_rope_table,
    make_ws32_repaired_index_buffer,
    ws32_prefill_chunk_plan,
    make_ws32_initial_state,
    select_ws32_exact_dsa_raw_weights,
    ws32_decoder_weight_names,
)
from glm_tpu.greenfield.sharding.ws32 import (  # noqa: E402
    build_ws32_physical_mesh,
)
from glm_tpu.greenfield.types import ModelGeometry  # noqa: E402
from glm_tpu.greenfield.validation.ws32_evidence import EVIDENCE_LAYOUT_V2  # noqa: E402
from glm_tpu.greenfield.validation.ws32_prefill import (  # noqa: E402
    PREFILL_MODE, PREFILL_MODES, SERIAL_PREFILL_MODE, require_batched_profile,
)
from scripts.greenfield.ws32_acquisition_journal import Ws32AcquisitionJournal  # noqa: E402
from glm_tpu.greenfield.validation import (  # noqa: E402
    compare_ws32_dsa_step,
    compare_ws32_dsa_within_engine,
    load_legacy_bench_module,
    load_ws32_long_context_oracle,
    bind_ws32_adjudication,
    compare_ws32_raw_tokens,
    load_ws32_adjudicated_divergence,
    load_ws32_short_context_oracle,
    validate_ws32_cache_probe,
)


_ZERO_SHA = "0" * 64
_XLA_MEMORY_FRACTION = ".95"
_VACANT_HLO_VIOLATIONS = {
    "StableHLO identity drifted",
    "optimized HLO identity drifted",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator-address", required=True)
    parser.add_argument("--num-processes", required=True, type=int)
    parser.add_argument("--process-id", required=True, type=int)
    parser.add_argument("--slice-name", required=True)
    parser.add_argument("--topology-capture-root", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--source-inventory", required=True, type=Path)
    # Spec §23.5: L7/L8 run against a TOKEN-ONLY long-context oracle. The legacy
    # harness captured no DSA events at 128K/256K and §23.5 forbids inventing
    # one, so these modes carry the within-engine DSA contract and make no
    # cross-oracle claim. §21's cross-oracle path stays the only path for 2k/8k.
    parser.add_argument("--long-context", choices=("passkey", "e0"), default=None)
    parser.add_argument("--tokenizer-root", type=Path, default=None)
    parser.add_argument("--long-context-oracle-dir", type=Path, default=None)
    parser.add_argument("--long-context-manifest-sha256", default=_ZERO_SHA)
    parser.add_argument("--long-context-success-sha256", default=_ZERO_SHA)
    # Not required: a §23.5 long-context run binds no short-context oracle and
    # must not be able to name one (see the refusal in main()).
    parser.add_argument("--token-oracle-dir", type=Path, default=None)
    parser.add_argument("--dsa-oracle-dir", type=Path, default=None)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument("--checkpoint-success-sha256", required=True)
    parser.add_argument("--token-oracle-manifest-sha256", required=True)
    parser.add_argument("--dsa-oracle-manifest-sha256", required=True)
    parser.add_argument("--token-oracle-success-sha256", required=True)
    parser.add_argument("--dsa-oracle-success-sha256", required=True)
    parser.add_argument("--dsa-association-summary-sha256", required=True)
    parser.add_argument("--dsa-association-success-sha256", required=True)
    parser.add_argument("--topology-sha256", required=True)
    parser.add_argument("--topology-fleet-sha256", required=True)
    parser.add_argument("--mesh-sha256", required=True)
    for graph in (
        "exact-materialize",
        "exact-promote",
        "prefill-chunk",
        "prefill-tail",
        "observer",
        "decode",
        "cache-probe",
    ):
        parser.add_argument(
            f"--expected-{graph}-stablehlo-sha256", required=True
        )
        parser.add_argument(
            f"--expected-{graph}-optimized-hlo-sha256", required=True
        )
    # Spec §23.2: the exact prefill runs as fixed-length chunks plus one tail
    # program so the graph set is the same for every prompt length and the
    # host can project the prefill wall and abort fail-closed.
    parser.add_argument("--prefill-chunk", type=int, required=True)
    parser.add_argument("--prefill-mode", choices=PREFILL_MODES, default=SERIAL_PREFILL_MODE)
    parser.add_argument("--prefill-budget-seconds", type=float, required=True)
    parser.add_argument("--batched-prefill-profile", default="")
    parser.add_argument("--prefill-memory-reserve-bytes", type=int, default=0)
    # Spec §23.8: declared side program measuring on-device rotary at long
    # positions (default off); its record is embedded in the runner record.
    # Spec §23.8: default-off legacy-faithful main-attention rotary table.
    parser.add_argument("--host-main-rope-table", choices=(0, 1), default=0, type=int)
    parser.add_argument("--rotary-diagnostic", choices=(0, 1), default=0, type=int)
    parser.add_argument("--rotary-diagnostic-positions", type=int, default=262_657)
    parser.add_argument("--context-capacity", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tensor-output", required=True, type=Path)
    parser.add_argument("--hlo-dir", required=True, type=Path)
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--compile-only", choices=(0, 1), default=0, type=int)
    parser.add_argument("--exact-dsa", choices=(0, 1), default=0, type=int)
    parser.add_argument(
        "--strategy-nd-dense", choices=(0, 1), default=0, type=int
    )
    parser.add_argument("--strategy-nd-dense-overlay-root", type=Path)
    parser.add_argument(
        "--strategy-nd-dense-overlay-manifest-sha256", default=_ZERO_SHA
    )
    parser.add_argument(
        "--strategy-nd-dense-overlay-manifest-file-sha256",
        default=_ZERO_SHA,
    )
    parser.add_argument(
        "--strategy-nd-dense-overlay-success-file-sha256",
        default=_ZERO_SHA,
    )
    parser.add_argument(
        "--checkpoint-transport", choices=("gcsfuse", "shm"), default="gcsfuse"
    )
    parser.add_argument("--dsa-adjudication-record", type=Path)
    parser.add_argument("--dsa-adjudication-sha256", default=_ZERO_SHA)
    parser.add_argument("--observer-steps", default=14, type=int)
    parser.add_argument("--warmup", default=2, type=int)
    parser.add_argument("--iterations", default=10, type=int)
    parser.add_argument("--trace-steps", default=2, type=int)
    return parser.parse_args()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _require_clean_code(expected: str) -> None:
    if REPO != EXPECTED_WORKTREE or _git_head() != expected:
        raise RuntimeError("WS32 short-decoder worktree/code identity drifted")
    dirty = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise RuntimeError("WS32 protected short decoder requires clean code")


def _atomic_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"append-only WS32 evidence exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with partial.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _atomic_npz(path: Path, **arrays: np.ndarray) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"append-only WS32 evidence exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with partial.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    records = {}
    for name, value in sorted(arrays.items()):
        contiguous = np.ascontiguousarray(value)
        records[name] = {
            "dtype": contiguous.dtype.name,
            "sha256": sha256(contiguous.view(np.uint8).tobytes()).hexdigest(),
            "shape": list(contiguous.shape),
        }
    return {
        "arrays": records,
        "byte_count": path.stat().st_size,
        "filename": path.name,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device_record(device: object, *, local_device_id: int) -> dict[str, Any]:
    runtime_local_id = device.local_hardware_id
    if runtime_local_id is not None and int(runtime_local_id) != local_device_id:
        raise ValueError("runtime and captured local device ids disagree")
    return {
        "coordinates": [int(value) for value in device.coords],
        "core_on_chip": int(device.core_on_chip),
        "device_id": int(device.id),
        "device_kind": str(device.device_kind),
        "local_device_id": int(local_device_id),
        "platform": str(device.platform),
        "process_index": int(device.process_index),
    }


def _memory_stats(device: object) -> dict[str, int] | None:
    value = device.memory_stats()
    if value is None:
        return None
    return {
        str(name): int(number)
        for name, number in value.items()
        if isinstance(number, int) and not isinstance(number, bool)
    }


def _compiled_memory(compiled: Any) -> dict[str, int | None]:
    analysis = compiled.memory_analysis()
    fields = (
        "alias_size_in_bytes",
        "argument_size_in_bytes",
        "generated_code_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
    )
    return {
        field: None
        if getattr(analysis, field, None) is None
        else int(getattr(analysis, field))
        for field in fields
    }


def _distribution(samples: list[float]) -> dict[str, float | int]:
    if not samples:
        raise ValueError("latency distribution requires samples")
    values = np.asarray(samples, dtype=np.float64)
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


def _geometry() -> ModelGeometry:
    value = json.loads(
        (REPO / "configs/glm-5.2-fp8-config.json").read_text(
            encoding="utf-8"
        )
    )
    return ModelGeometry.from_hf_config(value)


def _replicated(jax: Any, mesh: Any, value: np.ndarray) -> Any:
    from jax.sharding import NamedSharding, PartitionSpec as P

    host = np.ascontiguousarray(value)
    return jax.make_array_from_callback(
        host.shape,
        NamedSharding(mesh, P()),
        lambda _: host.copy(),
    )


def _scalar_token(jax: Any, value: Any) -> int:
    host = np.asarray(jax.device_get(value), dtype=np.int32)
    if host.shape != (1,):
        raise RuntimeError("WS32 token result geometry drifted")
    return int(host[0])


def _weight_name_leaves(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, tuple):
        return tuple(
            name for item in value for name in _weight_name_leaves(item)
        )
    raise TypeError("WS32 runner weight-name tree contains a non-string leaf")


def _linter_kind(graph: str) -> str:
    """Both prefill programs are lint-checked under the ``prefill`` contract."""
    return "prefill" if graph.startswith("prefill") else graph


def _write_graph(
    *,
    graph: str,
    lowered: Any,
    compiled: Any,
    hlo_dir: Path,
    expected_stable: str,
    expected_optimized: str,
    hidden_size: int,
    exact_dsa: bool,
    strategy_nd_dense: bool,
    host_main_rope_table: bool = False,
    prefill_mode: str = SERIAL_PREFILL_MODE,
    block_rows: int | None = None,
    acquisition_journal: Ws32AcquisitionJournal | None = None,
    batched_profile: str = "",
) -> tuple[dict[str, Any], str, str]:
    stable = str(lowered.compiler_ir(dialect="stablehlo"))
    optimized = compiled.as_text()
    stable_path = hlo_dir / f"{graph}.stablehlo.mlir"
    optimized_path = hlo_dir / f"{graph}.optimized_hlo.txt"
    _atomic_text(stable_path, stable)
    _atomic_text(optimized_path, optimized)
    def inspect() -> dict[str, Any]:
        if batched_profile and prefill_mode == PREFILL_MODE and graph in ("prefill_chunk", "prefill_tail"):
            from glm_tpu.greenfield.validation.ws32_prefill_admission import inspect_short_prefill_graph, short_plan
            if dict(short_plan(batched_profile).graph_rows)[graph] != block_rows:
                raise ValueError("worker prefill rows contradict registered profile")
            return inspect_short_prefill_graph(
                stable, optimized, graph=graph, profile=batched_profile, repo=REPO,
                expected_stable=expected_stable, expected_optimized=expected_optimized,
            )
        identity = None
        optimized_pin = expected_optimized
        if batched_profile:
            from glm_tpu.greenfield.validation.ws32_prefill_admission import short_graph_identity
            identity = short_graph_identity(
                stable, optimized, graph=graph, profile=batched_profile, repo=REPO,
                expected_stable=expected_stable, expected_optimized=expected_optimized,
            )
            optimized_pin = identity["raw_optimized_hlo_sha256"]
        if prefill_mode == PREFILL_MODE and graph in ("prefill_chunk", "prefill_tail"):
            from glm_tpu.greenfield.benchmarking.ws32_batched_prefill import inspect_ws32_batched_prefill_hlo
            from glm_tpu.greenfield.validation.ws32_prefill_admission import profile_is_paired
            report = inspect_ws32_batched_prefill_hlo(
                stable, optimized, block_rows=block_rows,
                expected_stablehlo_sha256=expected_stable,
                expected_optimized_hlo_sha256=optimized_pin,
                paired_position_sort=profile_is_paired(batched_profile) if batched_profile else False,
            )
            if batched_profile:
                from glm_tpu.greenfield.validation.ws32_prefill_admission import authorize_short_graph
                return authorize_short_graph(
                    {**report, "source_location_identity": identity}, profile=batched_profile, repo=REPO,
                )
            return report
        report = validate_ws32_decoder_hlo(
            stable, optimized,
            expected_stablehlo_sha256=expected_stable,
            expected_optimized_hlo_sha256=optimized_pin,
            hidden_size=hidden_size,
            kind=_linter_kind(graph),
            exact_dsa=exact_dsa,
            strategy_nd_dense=strategy_nd_dense,
            host_main_rope_table=host_main_rope_table,
        ).to_dict()
        return report if identity is None else {**report, "source_location_identity": identity}

    report = inspect() if acquisition_journal is None else acquisition_journal.inspect(
        graph, stable, optimized, inspect
    )
    return report, stable, optimized


def _write_exact_materializer_graph(
    *,
    graph: str,
    lowered: Any,
    compiled: Any,
    hlo_dir: Path,
    expected_stable: str,
    expected_optimized: str,
    acquisition_journal: Ws32AcquisitionJournal | None = None,
    batched_profile: str = "",
) -> dict[str, Any]:
    stable = str(lowered.compiler_ir(dialect="stablehlo"))
    optimized = compiled.as_text()
    _atomic_text(hlo_dir / f"{graph}.stablehlo.mlir", stable)
    _atomic_text(hlo_dir / f"{graph}.optimized_hlo.txt", optimized)
    def inspect() -> dict[str, Any]:
        identity = None
        optimized_pin = expected_optimized
        if batched_profile:
            from glm_tpu.greenfield.validation.ws32_prefill_admission import short_graph_identity
            identity = short_graph_identity(
                stable, optimized, graph=graph, profile=batched_profile, repo=REPO,
                expected_stable=expected_stable, expected_optimized=expected_optimized,
            )
            optimized_pin = identity["raw_optimized_hlo_sha256"]
        report = validate_ws32_exact_dsa_materializer_hlo(
            stable,
            optimized,
            expected_stablehlo_sha256=expected_stable,
            expected_optimized_hlo_sha256=optimized_pin,
            kind=graph,
        ).to_dict()
        return report if identity is None else {**report, "source_location_identity": identity}

    return inspect() if acquisition_journal is None else acquisition_journal.inspect(
        graph, stable, optimized, inspect
    )


def _require_graph_authorized(
    report: Mapping[str, Any], *, compile_only: bool
) -> None:
    if compile_only:
        # Acquisition never executes a model graph.  Preserve and validate all
        # four compiler products before reporting any structural refusal so a
        # single protected load cannot fail one graph at a time.
        return
    elif not report["passed"]:
        raise RuntimeError(
            "WS32 numerical graph failed before execution: "
            + "; ".join(report["violations"])
        )


def _require_acquisition_authorized(
    graphs: Mapping[str, Mapping[str, Any]],
    *,
    exact_dsa: bool,
) -> None:
    expected = {"cache_probe", "decode", "observer", "prefill_chunk", "prefill_tail"}
    if exact_dsa:
        expected.update({"exact_materialize", "exact_promote"})
    if set(graphs) != expected:
        raise RuntimeError("WS32 acquisition did not preserve all four graphs")
    if all(value["passed"] for value in graphs.values()) or any(
        set(value["violations"]) != _VACANT_HLO_VIOLATIONS
        for value in graphs.values()
    ):
        raise RuntimeError("WS32 HLO acquisition found structural violations")


def _publish_acquisition_result(
    prevalidation: Mapping[str, Any], *, output: Path, exact_dsa: bool,
) -> dict[str, Any]:
    """Preserve all seven graphs' compact evidence even on planned refusal.

    The wrapper uploads runner.rankN.json, not hlo/prevalidation.json. Losing
    that envelope would force another model load just to recover allocations.
    HLO_REFUSED is never a SUCCESS or an execution authorization.
    """
    try:
        _require_acquisition_authorized(prevalidation["graphs"], exact_dsa=exact_dsa)
    except RuntimeError as exc:
        _atomic_json(output, {
            **prevalidation, "performance_claim": False, "schema_version": 1,
            "status": "HLO_REFUSED", "failure": str(exc),
        })
        raise
    result = {
        **prevalidation, "performance_claim": False,
        "schema_version": 1, "status": "HLO_ACQUIRED",
    }
    _atomic_json(output, result)
    return result


def _trace_files(trace_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(trace_dir.rglob("*.xplane.pb")):
        records.append(
            {
                "byte_count": path.stat().st_size,
                "relative_path": path.relative_to(trace_dir).as_posix(),
                "sha256": _sha256_file(path),
            }
        )
    if len(records) != 1:
        raise RuntimeError(
            f"WS32 profiler produced {len(records)} XPlane files, expected one"
        )
    return records


def _initialize_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any]:
    import jax
    from jax.sharding import Mesh

    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=args.num_processes,
        process_id=args.process_id,
    )
    if (
        jax.default_backend() != "tpu"
        or jax.device_count() != 32
        or len(jax.local_devices()) != 4
        or jax.process_count() != 8
    ):
        raise RuntimeError("WS32 runner did not initialize the exact 8x4 TPU runtime")
    captures = tuple(
        json.loads(
            (
                args.topology_capture_root
                / f"topology.rank{launch_process_id}.json"
            ).read_text(encoding="utf-8")
        )
        for launch_process_id in range(8)
    )
    topology, ordered_captures, fleet_sha = validate_ws32_topology_fleet(
        captures,
        expected_topology_sha256=args.topology_sha256,
        expected_fleet_sha256=args.topology_fleet_sha256,
        slice_name=args.slice_name,
    )
    launch_capture = ordered_captures[args.process_id]
    if (
        launch_capture["hostname"] != socket.gethostname()
        or launch_capture["jax_process_index"] != jax.process_index()
        or launch_capture["local_device_ids"]
        != [int(device.id) for device in jax.local_devices()]
    ):
        raise RuntimeError("WS32 launch/JAX/topology fleet mapping drifted")
    physical_mesh = build_ws32_physical_mesh(topology)
    if physical_mesh.mesh_hash != args.mesh_sha256:
        raise ValueError("WS32 physical mesh hash drifted")
    runtime_by_id = {int(device.id): device for device in jax.devices()}
    if set(runtime_by_id) != set(physical_mesh.flattened_device_ids):
        raise ValueError("WS32 runtime device ids differ from physical mesh")
    for captured in topology.devices:
        if _device_record(
            runtime_by_id[captured.device_id],
            local_device_id=captured.local_device_id,
        ) != captured.to_dict():
            raise ValueError(
                f"WS32 runtime topology drifted at {captured.device_id}"
            )
    mesh = Mesh(
        np.asarray(
            [runtime_by_id[item] for item in physical_mesh.flattened_device_ids],
            dtype=object,
        ).reshape(8, 4),
        ("expert", "feature"),
    )
    return jax, mesh, physical_mesh, topology, fleet_sha


def _long_context_token_result(
    observed_tokens: list[int],
    long_context: Any,
    *,
    tokenizer_root: Path | None,
) -> dict[str, Any]:
    """Spec §23.5: the token result at a length with no legacy token oracle.

    For L7 the pass criterion is the extracted passkey against gold. The legacy
    ids are compared as a DIAGNOSTIC and the result deliberately carries no
    ``exact_prefix_match`` key, because §23.5 forbids labelling anything "raw
    tokens exact" at these lengths. For L8 there is no criterion at all and
    ``passkey_matches_gold`` is None.
    """

    legacy = np.asarray(long_context.generated_token_ids, dtype=np.int32)
    compared = min(len(observed_tokens), int(legacy.size))
    observed = np.asarray(observed_tokens[:compared], dtype=np.int32)
    diagnostic = {
        "compared": int(compared),
        "legacy_ids_match": bool(np.array_equal(observed, legacy[:compared])),
        "note": (
            "legacy ids are a diagnostic reference (§23.1/§23.5): the legacy stored "
            "text, not ids, and nothing here is a raw-tokens-exact claim"
        ),
    }
    result: dict[str, Any] = {
        "kind": long_context.kind,
        "legacy_diagnostic": diagnostic,
        "observed_token_count": len(observed_tokens),
        "passkey_matches_gold": None,
    }
    if long_context.kind != "passkey":
        result["criterion"] = "none (§23.5: L8 has no correctness oracle)"
        return result

    result["criterion"] = "extract_passkey(detok(first 20 greedy tokens)) == gold"
    result["gold"] = long_context.gold
    if tokenizer_root is None:
        raise ValueError("WS32 passkey mode needs --tokenizer-root to detokenise")
    from transformers import AutoTokenizer

    longctx = load_legacy_bench_module(
        "glm_longctx", pinned_files=long_context.manifest["legacy_bench_files"]
    )

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_root), trust_remote_code=True)
    text = tokenizer.decode(observed_tokens[: int(legacy.size)], skip_special_tokens=True)
    extracted = longctx.extract_passkey(text)
    result["detokenised"] = text
    result["extracted_passkey"] = extracted
    result["passkey_matches_gold"] = bool(extracted == long_context.gold)
    return result


def _require_passkey_tooling(long_context: Any, tokenizer_root: Path | None) -> None:
    """Prove the L7 criterion is computable BEFORE a multi-hour prefill.

    The pass criterion detokenises the engine's greedy tokens and extracts a
    passkey, so it needs the tokenizer and the legacy extractor on every host.
    Discovering a missing mount or import after the prefill would throw away
    hours of pod time, so the tooling is exercised here on the ORACLE's own
    stored ids: it must reproduce the legacy gold exactly.
    """

    if tokenizer_root is None:
        raise ValueError("WS32 passkey mode needs --tokenizer-root to detokenise")
    from transformers import AutoTokenizer

    longctx = load_legacy_bench_module(
        "glm_longctx", pinned_files=long_context.manifest["legacy_bench_files"]
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_root), trust_remote_code=True
    )
    text = tokenizer.decode(
        np.asarray(long_context.generated_token_ids, dtype=np.int32).tolist(),
        skip_special_tokens=True,
    )
    if longctx.extract_passkey(text) != long_context.gold:
        raise ValueError(
            "WS32 passkey tooling does not reproduce the sealed gold from the "
            "oracle's own tokens; the criterion is not computable on this host"
        )


def _batched_fleet_all(value: bool) -> bool:
    """Declared host-boundary consensus, never per-layer or per-token dispatch."""
    from jax.experimental import multihost_utils

    values = np.asarray(multihost_utils.process_allgather(np.asarray(int(value), np.int32)))
    if values.shape != (8,) or not np.isin(values, (0, 1)).all():
        raise ValueError("batched host consensus requires eight boolean votes")
    return bool(values.all())


def _execute_batched_prefill(
    *, args: Any, mesh: Any, config: Any, plan: Any, prompt_tokens: np.ndarray,
    compiled: Mapping[str, Any], weights: Any, wk: Any, rope: Any,
) -> tuple[Any, Any, dict[str, Any], list[dict[str, Any]]]:
    """Run the admitted adapter and retain partial evidence before observer work.

    main must first drop every compile-state alias. This function never receives
    or creates a second compile placeholder. The registered profile, actual HLO,
    source and all-live memory checks remain prerequisites to model dispatch.
    """
    import jax
    from scripts.greenfield import ws32_batched_prefill_runner as batched
    from glm_tpu.greenfield.validation.ws32_prefill_admission import (
        short_plan, SHORT_RESERVE_BYTES, short_budget,
        require_short_numerical_inputs, short_acquisition, require_acquired_model_source,
        validate_short_compiled_memory,
    )
    from glm_tpu.greenfield.validation.ws32_prefill_memory import capture_identified_device_memory

    identity = dict(
        artifact_kind="greenfield_ws32_batched_prefill_phase",
        code_hash=args.expected_code_hash, launch_process_id=args.process_id,
        jax_process_index=int(jax.process_index()), hostname=socket.gethostname(),
        prefill_mode=PREFILL_MODE, profile=args.batched_prefill_profile, plan=short_plan(args.batched_prefill_profile).identity(),
        prompt_ids_sha256=sha256(prompt_tokens.tobytes()).hexdigest(),
        performance_claim=False, numerical_claim=False,
    )

    def publish(stage: str, payload: Mapping[str, Any]) -> None:
        _atomic_json(
            args.output.parent / f"batched_prefill_{stage}.rank{args.process_id}.json",
            {**identity, "stage": stage, **payload},
        )

    def progress(record: Mapping[str, Any]) -> None:
        print("GREENFIELD_WS32_BATCHED_PREFILL " + json.dumps(record, sort_keys=True), flush=True)

    try:
        preflight_error = None
        try:
            receipt = short_acquisition(REPO, profile=args.batched_prefill_profile)
            graph_pins = {
                graph: {
                    form: getattr(args, f"expected_{graph}_{form}")
                    for form in ("stablehlo_sha256", "optimized_hlo_sha256")
                }
                for graph in receipt["graphs"]
            }
            require_short_numerical_inputs(
                profile=args.batched_prefill_profile, plan=plan,
                reserve_bytes=args.prefill_memory_reserve_bytes,
                budget_seconds=args.prefill_budget_seconds, graph_pins=graph_pins, repo=REPO,
            )
            require_acquired_model_source(REPO, profile=args.batched_prefill_profile)
            if set(compiled) != set(batched.GRAPHS):
                raise ValueError("batched compiled memory differs from acquired graph pair")
            for graph in batched.GRAPHS:
                validate_short_compiled_memory(
                    graph, _compiled_memory(compiled[graph]),
                    profile=args.batched_prefill_profile, repo=REPO,
                )
        except Exception as exc:
            preflight_error = exc
        try:
            publish("preflight", {"error": None if preflight_error is None else str(preflight_error)})
        except Exception as exc:
            if preflight_error is None:
                preflight_error = exc
        if not _batched_fleet_all(preflight_error is None) or preflight_error is not None:
            raise RuntimeError("batched fleet refused numerical preflight") from preflight_error
        decoder, token, execution = batched.execute_graph_pair(
            mesh, config, plan, prompt_tokens, compiled, weights, wk, rope,
            budget_seconds=short_budget(args.batched_prefill_profile),
            required_memory_reserve_bytes=SHORT_RESERVE_BYTES,
            progress=progress, fleet_all=_batched_fleet_all,
            additional_resident_executables={},
            memory_progress=lambda record: publish("memory", record),
        )
        error = None
        after = None
        try:
            batched.validate_execution_record(execution, plan)
            if execution["first_token_ready"] != int(np.asarray(token)[0]):
                raise ValueError("batched execution first token differs from actual output")
            after = capture_identified_device_memory(tuple(jax.local_devices()))
            if len(after) != 4 or any(
                r["process_index"] != jax.process_index()
                or r["bytes_limit"] - r["peak_bytes_in_use"] < SHORT_RESERVE_BYTES
                for r in after
            ):
                raise ValueError("batched post-execution memory reserve/owners failed")
            publish("complete", {"execution": execution, "device_memory_after_prefill": after})
        except Exception as exc:
            error = exc
        if not _batched_fleet_all(error is None) or error is not None:
            raise RuntimeError("batched fleet refused completed prefill evidence/memory") from error
        return decoder, token, execution, after
    except Exception as error:
        # Original errors remain the cause even if failure publication itself
        # fails; the wrapper preserves the log. No incomplete phase can seal.
        try:
            publish("failure", {"exception_type": type(error).__name__, "exception": str(error)})
        except Exception as publication_error:
            print(f"GREENFIELD_WS32_BATCHED_FAILURE_PUBLICATION {publication_error}", flush=True)
        raise


def main() -> int:
    args = parse_args()
    from glm_tpu.greenfield.validation.ws32_prefill_admission import short_plan, short_program_options
    numerical_plan = (
        short_plan(args.batched_prefill_profile)
        if args.prefill_mode == PREFILL_MODE and not args.compile_only else None
    )
    require_batched_profile(
        args.prefill_mode, exact_dsa=bool(args.exact_dsa),
        host_main_rope_table=bool(args.host_main_rope_table),
        block_rows=args.prefill_chunk, long_context=args.long_context,
        adjudication_record=args.dsa_adjudication_record,
        adjudication_sha256=args.dsa_adjudication_sha256,
        mlp_window=numerical_plan.mlp_window if numerical_plan else False,
    )
    batched_prefill = args.prefill_mode == PREFILL_MODE
    if batched_prefill and not args.compile_only:
        from glm_tpu.greenfield.validation.ws32_prefill_admission import require_short_numerical_request

        require_short_numerical_request(args, prompt_length=numerical_plan.prompt_length, repo=REPO)
    if args.compile_only and getattr(args, "batched_prefill_profile", ""):
        raise ValueError("acquisition cannot claim a registered numerical profile")
    if args.num_processes != 8 or not 0 <= args.process_id < 8:
        raise ValueError("WS32 short decoder requires eight launch processes")
    if min(args.observer_steps, args.warmup, args.iterations, args.trace_steps) < 1:
        raise ValueError("WS32 execution counts must all be positive")
    if (
        args.output.exists()
        or args.tensor_output.exists()
        or args.hlo_dir.exists()
        or args.trace_dir.exists()
    ):
        raise FileExistsError("WS32 short-decoder evidence is append-only")
    from glm_tpu.greenfield.validation.ws32_prefill_admission import require_hlo_pin_request
    require_hlo_pin_request(args, compile_only=bool(args.compile_only), repo=REPO)
    association_pins = (
        args.dsa_association_summary_sha256,
        args.dsa_association_success_sha256,
    )
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in association_pins
    ):
        raise ValueError("WS32 DSA association pins must be lowercase SHA-256")
    if (
        args.exact_dsa
        and any(value == _ZERO_SHA for value in association_pins)
    ) or (
        not args.exact_dsa
        and any(value != _ZERO_SHA for value in association_pins)
    ):
        raise ValueError("WS32 exact DSA flag/association pins drifted")
    overlay_pins = (
        args.strategy_nd_dense_overlay_manifest_sha256,
        args.strategy_nd_dense_overlay_manifest_file_sha256,
        args.strategy_nd_dense_overlay_success_file_sha256,
    )
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in overlay_pins
    ):
        raise ValueError(
            "WS32 StrategyND dense overlay pins must be lowercase SHA-256"
        )
    if args.strategy_nd_dense:
        if args.strategy_nd_dense_overlay_root is None or any(
            value == _ZERO_SHA for value in overlay_pins
        ):
            raise ValueError(
                "WS32 StrategyND dense path requires its pinned overlay"
            )
    elif args.strategy_nd_dense_overlay_root is not None or any(
        value != _ZERO_SHA for value in overlay_pins
    ):
        raise ValueError(
            "default WS32 dense path must keep overlay inputs vacant"
        )
    _require_clean_code(args.expected_code_hash)
    if os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION") != _XLA_MEMORY_FRACTION:
        raise RuntimeError("WS32 XLA allocator fraction is not pinned to .95")

    geometry = _geometry()
    config = Ws32DecoderConfig(
        geometry=geometry,
        context_capacity=args.context_capacity,
        exact_dsa=bool(args.exact_dsa),
        strategy_nd_dense=bool(args.strategy_nd_dense),
        host_main_rope_table=bool(args.host_main_rope_table),
    )
    long_context = None
    if args.long_context is not None:
        if args.long_context_oracle_dir is None:
            raise ValueError("WS32 long-context mode needs its oracle directory")
        if args.dsa_adjudication_record is not None:
            # §23.5: there is nothing at these lengths for a §21.2 record to
            # adjudicate against, and binding one would imply a cross-oracle
            # comparison that does not exist.
            raise ValueError("WS32 long-context runs bind no §21.2 adjudication record")
        if (
            args.token_oracle_dir is not None
            or args.dsa_oracle_dir is not None
            or args.token_oracle_manifest_sha256 != _ZERO_SHA
            or args.token_oracle_success_sha256 != _ZERO_SHA
            or args.dsa_oracle_manifest_sha256 != _ZERO_SHA
            or args.dsa_oracle_success_sha256 != _ZERO_SHA
        ):
            raise ValueError("WS32 long-context runs bind no short-context oracle")
        long_context = load_ws32_long_context_oracle(
            args.long_context_oracle_dir,
            expected_manifest_sha256=args.long_context_manifest_sha256,
            expected_success_sha256=args.long_context_success_sha256,
            expected_kind=args.long_context,
        )
        if long_context.kind == "passkey":
            _require_passkey_tooling(long_context, args.tokenizer_root)
        oracle = None
        prompt_token_ids = long_context.prompt_token_ids
        long_context_record = {
            "depth": long_context.depth,
            "item_row_id": long_context.item_row_id,
            "kind": long_context.kind,
            "manifest_sha256": long_context.manifest_sha256,
            "source_run_id": long_context.source_run_id,
            "success_sha256": args.long_context_success_sha256,
        }
    else:
        if args.token_oracle_dir is None or args.dsa_oracle_dir is None:
            raise ValueError("WS32 short-context runs need both sealed oracles")
        oracle = load_ws32_short_context_oracle(
            args.token_oracle_dir,
            args.dsa_oracle_dir,
            expected_token_manifest_sha256=args.token_oracle_manifest_sha256,
            expected_dsa_manifest_sha256=args.dsa_oracle_manifest_sha256,
            expected_token_success_sha256=args.token_oracle_success_sha256,
            expected_dsa_success_sha256=args.dsa_oracle_success_sha256,
        )
        prompt_token_ids = oracle.prompt_token_ids
        long_context_record = None
    # Spec §21.2 first-divergent-event adjudication: default off (exact mode).
    if batched_prefill and not args.compile_only:
        from glm_tpu.greenfield.validation.ws32_prefill_admission import require_short_numerical_request

        require_short_numerical_request(args, prompt_length=int(prompt_token_ids.size), repo=REPO)
    if args.dsa_adjudication_record is None:
        if args.dsa_adjudication_sha256 != _ZERO_SHA:
            raise ValueError("WS32 adjudication SHA given without a record")
        dsa_adjudication = None
        dsa_adjudication_record = None
    else:
        repository_root = Path(__file__).resolve().parents[2]
        dsa_adjudication = load_ws32_adjudicated_divergence(
            args.dsa_adjudication_record,
            expected_sha256=args.dsa_adjudication_sha256,
            repository_root=repository_root,
        )
        bind_ws32_adjudication(
            dsa_adjudication, oracle, observer_steps=args.observer_steps
        )
        dsa_adjudication_record = {
            "event_index": dsa_adjudication.event_index,
            "mode": "first_divergent_event",
            "record_path": str(
                args.dsa_adjudication_record.resolve().relative_to(repository_root)
            ),
            "record_sha256": dsa_adjudication.record_sha256,
            "step": dsa_adjudication.step,
        }
    required_capacity = (
        prompt_token_ids.size
        + args.observer_steps
        + args.warmup
        + args.iterations
        + args.trace_steps
    )
    if args.context_capacity < required_capacity + 1:
        raise ValueError(
            "WS32 context capacity does not cover prompt plus proof/timing steps"
        )
    if oracle is not None and args.observer_steps > oracle.decode_positions.size:
        raise ValueError("WS32 observer steps exceed the sealed DSA oracle")

    acquisition_journal = None
    if batched_prefill and not args.compile_only:
        from scripts.greenfield.ws32_acquisition_journal import Ws32NumericalJournal
        from glm_tpu.greenfield.validation.ws32_prefill_admission import short_numerical_identity

        acquisition_journal = Ws32NumericalJournal(
            args.output.with_name(f"numerical_journal.rank{args.process_id}.jsonl"),
            dict(
                **short_numerical_identity(profile=args.batched_prefill_profile), compile_only=False,
                code_hash=args.expected_code_hash, launch_process_id=args.process_id,
                hostname=socket.gethostname(),
                prompt_ids_sha256=sha256(prompt_token_ids.tobytes()).hexdigest(),
                checkpoint_manifest_sha256=args.checkpoint_manifest_sha256,
                checkpoint_success_sha256=args.checkpoint_success_sha256,
            ),
        )
        acquisition_journal.phase("runtime_initialize_started")
    jax, mesh, physical_mesh, topology, fleet_sha = _initialize_runtime(args)
    if batched_prefill and not args.compile_only:
        acquisition_journal.phase(
            "checkpoint_verify_started", jax_process_index=int(jax.process_index()),
            local_device_ids=[int(device.id) for device in jax.local_devices()],
            mesh_sha256=physical_mesh.mesh_hash, topology_fleet_sha256=fleet_sha,
        )
    rotary_diagnostic: dict[str, Any] | None = None
    if args.rotary_diagnostic:
        # Runs on this host's default local device before any model program is
        # compiled or loaded; outside the timed window and the traced steps.
        from glm_tpu.greenfield.validation.rotary_diagnostic import run_rotary_diagnostic

        rotary_diagnostic = run_rotary_diagnostic(
            positions=int(args.rotary_diagnostic_positions)
        ).record
        print(
            "GREENFIELD_WS32_ROTARY_DIAGNOSTIC "
            f"verdict={rotary_diagnostic['verdict']} failing={len(rotary_diagnostic['failing_cells'])}",
            flush=True,
        )
    inventory = inspect_source_inventory(args.source_inventory)
    local_device_ids = {int(device.id) for device in jax.local_devices()}
    local_hash_slots = tuple(
        slot
        for slot, device_id in enumerate(physical_mesh.flattened_device_ids)
        if device_id in local_device_ids
    )
    if len(local_hash_slots) != 4:
        raise RuntimeError("WS32 host does not own exactly four checkpoint slots")
    checkpoint = verify_ws32_runtime_checkpoint(
        args.checkpoint_root,
        expected_manifest_sha256=args.checkpoint_manifest_sha256,
        expected_success_sha256=args.checkpoint_success_sha256,
        expected_mesh_hash=args.mesh_sha256,
        expected_topology_hash=args.topology_sha256,
        inventory=inventory,
        geometry=geometry,
        verify_file_hashes=True,
        verify_file_hash_slots=local_hash_slots,
        local_slot_layout=args.checkpoint_transport == "shm",
    )
    load_started = time.perf_counter()
    if batched_prefill and not args.compile_only:
        acquisition_journal.phase("load_started", checkpoint_verified_device_slots=list(local_hash_slots))
    loaded = load_ws32_runtime_checkpoint(
        checkpoint,
        mesh=mesh,
        physical_mesh=physical_mesh,
    )
    device_memory_before_load = loaded.device_memory_before
    base_device_memory_after_load = loaded.device_memory_after
    local_device_slots = loaded.local_device_slots
    all_arrays = dict(loaded.arrays)
    dense_overlay = None
    loaded_dense_overlay = None
    if config.strategy_nd_dense:
        assert args.strategy_nd_dense_overlay_root is not None
        dense_overlay = verify_ws32_strategy_nd_dense_overlay(
            args.strategy_nd_dense_overlay_root,
            expected_manifest_sha256=(
                args.strategy_nd_dense_overlay_manifest_sha256
            ),
            expected_manifest_file_sha256=(
                args.strategy_nd_dense_overlay_manifest_file_sha256
            ),
            expected_success_file_sha256=(
                args.strategy_nd_dense_overlay_success_file_sha256
            ),
        )
        loaded_dense_overlay = load_ws32_strategy_nd_dense_overlay(
            dense_overlay,
            mesh=mesh,
            physical_mesh=physical_mesh,
        )
        overlap = set(all_arrays) & set(loaded_dense_overlay.arrays)
        if overlap:
            raise RuntimeError(
                "WS32 StrategyND dense overlay aliases base tensor names"
            )
        all_arrays.update(loaded_dense_overlay.arrays)
    load_seconds = time.perf_counter() - load_started
    expected_names = _weight_name_leaves(ws32_decoder_weight_names(config))
    if any(name not in all_arrays for name in expected_names):
        raise RuntimeError("WS32 combined checkpoint is missing decoder tensors")
    weights = bind_ws32_decoder_weights(
        {name: all_arrays[name] for name in expected_names}, config
    )
    raw_prefill_config = raw_prefill_weights = None
    if batched_prefill:
        from scripts.greenfield import ws32_batched_prefill_runner as batched
        raw_prefill_config, raw_prefill_weights = batched.bind_raw_prefill_weights(all_arrays, config)
    device_memory_after_load = tuple(
        _memory_stats(device) for device in jax.local_devices()
    )
    del all_arrays
    del loaded
    gc.collect()
    args.hlo_dir.mkdir(parents=True, exist_ok=False)
    graphs: dict[str, Any] = {}
    compile_seconds: dict[str, float] = {}
    compiled_memory: dict[str, Any] = {}
    if batched_prefill and args.compile_only:
        acquisition_journal = Ws32AcquisitionJournal(
            args.output.with_name(f"acquisition_journal.rank{args.process_id}.jsonl"),
            {
                "code_hash": args.expected_code_hash,
                "prefill_mode": args.prefill_mode,
                "compile_only": True,
                "hostname": socket.gethostname(),
                "launch_process_id": args.process_id,
                "jax_process_index": int(jax.process_index()),
                "checkpoint_manifest_sha256": checkpoint.manifest["manifest_sha256"],
                "checkpoint_success_sha256": checkpoint.success["success_sha256"],
                "checkpoint_transport": args.checkpoint_transport,
                "checkpoint_verified_device_slots": list(local_hash_slots),
                "local_device_slots": list(local_device_slots),
                "local_device_ids": [int(device.id) for device in jax.local_devices()],
                "source_inventory_sha256": inventory.inventory_sha256,
                "mesh_sha256": physical_mesh.mesh_hash,
                "topology_sha256": topology.topology_hash,
                "topology_fleet_sha256": fleet_sha,
                "context_capacity": args.context_capacity,
                "prompt_length": int(prompt_token_ids.size),
                "prefill_chunk_length": int(args.prefill_chunk),
                "exact_dsa": config.exact_dsa,
                "host_main_rope_table": config.host_main_rope_table,
                "strategy_nd_dense": config.strategy_nd_dense,
                "overlay_manifest_sha256": args.strategy_nd_dense_overlay_manifest_sha256,
                "overlay_manifest_file_sha256": args.strategy_nd_dense_overlay_manifest_file_sha256,
                "overlay_success_file_sha256": args.strategy_nd_dense_overlay_success_file_sha256,
                "load_seconds": load_seconds,
                "base_device_memory_after_load": list(base_device_memory_after_load),
                "device_memory_after_load": list(device_memory_after_load),
                "device_memory_before_load": list(device_memory_before_load),
                "xla_python_client_mem_fraction": _XLA_MEMORY_FRACTION,
            },
        )

    if batched_prefill and not args.compile_only:
        acquisition_journal.phase(
            "load_completed", seconds=load_seconds, local_device_slots=list(local_device_slots),
            device_memory_after_load=list(device_memory_after_load),
        )

    def begin_compile(graph: str) -> None:
        if acquisition_journal is not None:
            acquisition_journal.begin(graph)

    def record_compile(graph: str) -> None:
        if acquisition_journal is not None:
            acquisition_journal.compiled(
                graph, seconds=compile_seconds[graph], memory=compiled_memory[graph],
                device_memory=[_memory_stats(device) for device in jax.local_devices()],
            )

    exact_dsa_weights = None
    if config.exact_dsa:
        materializer = build_ws32_exact_dsa_materializer_program(mesh, config)
        raw_exact_weights = select_ws32_exact_dsa_raw_weights(weights, config)
        decode_exact_jit = jax.jit(materializer.decode)
        begin_compile("exact_materialize")
        decode_exact_lowered = decode_exact_jit.lower(raw_exact_weights)
        started = time.perf_counter()
        decode_exact_compiled = decode_exact_lowered.compile()
        compile_seconds["exact_materialize"] = time.perf_counter() - started
        compiled_memory["exact_materialize"] = _compiled_memory(
            decode_exact_compiled
        )
        record_compile("exact_materialize")
        graphs["exact_materialize"] = _write_exact_materializer_graph(
            graph="exact_materialize",
            lowered=decode_exact_lowered,
            compiled=decode_exact_compiled,
            hlo_dir=args.hlo_dir,
            expected_stable=args.expected_exact_materialize_stablehlo_sha256,
            expected_optimized=(
                args.expected_exact_materialize_optimized_hlo_sha256
            ),
            acquisition_journal=acquisition_journal,
            batched_profile=getattr(args, "batched_prefill_profile", "") if batched_prefill else "",
        )
        _require_graph_authorized(
            graphs["exact_materialize"],
            compile_only=bool(args.compile_only),
        )
        decoded_exact_weights = decode_exact_compiled(raw_exact_weights)
        jax.block_until_ready(decoded_exact_weights)
        promote_exact_jit = jax.jit(materializer.promote)
        begin_compile("exact_promote")
        promote_exact_lowered = promote_exact_jit.lower(decoded_exact_weights)
        started = time.perf_counter()
        promote_exact_compiled = promote_exact_lowered.compile()
        compile_seconds["exact_promote"] = time.perf_counter() - started
        compiled_memory["exact_promote"] = _compiled_memory(
            promote_exact_compiled
        )
        record_compile("exact_promote")
        graphs["exact_promote"] = _write_exact_materializer_graph(
            graph="exact_promote",
            lowered=promote_exact_lowered,
            compiled=promote_exact_compiled,
            hlo_dir=args.hlo_dir,
            expected_stable=args.expected_exact_promote_stablehlo_sha256,
            expected_optimized=args.expected_exact_promote_optimized_hlo_sha256,
            acquisition_journal=acquisition_journal,
            batched_profile=getattr(args, "batched_prefill_profile", "") if batched_prefill else "",
        )
        _require_graph_authorized(
            graphs["exact_promote"],
            compile_only=bool(args.compile_only),
        )
        exact_dsa_weights = promote_exact_compiled(decoded_exact_weights)
        jax.block_until_ready(exact_dsa_weights)
        decode_exact_jit.clear_cache()
        promote_exact_jit.clear_cache()
        del decode_exact_compiled
        del decode_exact_lowered
        del decoded_exact_weights
        del promote_exact_compiled
        del promote_exact_lowered
        del raw_exact_weights
        gc.collect()
    program = build_ws32_decoder_program(mesh, config)
    prompt_length = int(prompt_token_ids.size)
    full_chunks, tail_length = ws32_prefill_chunk_plan(
        prompt_length, args.prefill_chunk
    )
    state = make_ws32_initial_state(mesh, config)
    initial_token = _replicated(jax, mesh, np.asarray([-1], dtype=np.int32))
    repaired_buffer = make_ws32_repaired_index_buffer(mesh, config)
    main_rope_table_record: dict[str, Any] | None = None
    table_inputs: tuple[Any, ...] = ()
    if config.host_main_rope_table:
        host_table = build_ws32_main_rope_table(config)
        main_rope_table_record = {
            "bytes_per_device": int(host_table.nbytes),
            "rotary_dim": int(config.geometry.qk_rope_head_dim),
            "rows": int(config.context_capacity),
            "sha256": rotary_table_sha256(host_table),
            "theta": WS32_MAIN_ROPE_THETA,
        }
        table_inputs = (_replicated(jax, mesh, host_table),)
        del host_table
    # Both prefill programs are compiled and pinned before any execution so an
    # acquisition preserves the complete graph set and a numerical run never
    # discovers a tail-program refusal after hours of chunk scanning.
    prefill_programs: dict[str, Any] = {}
    prefill_compiled: dict[str, Any] = {}
    prefill_lowered: dict[str, Any] = {}
    prefill_jits: dict[str, Any] = {}
    donate = (1, 4) if exact_dsa_weights is not None else (1, 3)
    if batched_prefill:
        from glm_tpu.greenfield.runtime.ws32_batched_prefill import Ws32BatchedPrefillState
        batched_plan = batched.BatchedPrefillPlan(
            prompt_length, args.prefill_chunk, config.context_capacity,
            mlp_window=numerical_plan.mlp_window if numerical_plan else False,
            tail_graph_rows=numerical_plan.tail_graph_rows if numerical_plan else None,
        )
        prefill_programs = batched.build_graph_pair(
            mesh, raw_prefill_config, batched_plan,
            **(short_program_options(args.batched_prefill_profile) if numerical_plan else {}),
        )
        batched_wk = batched.completed_repair_weights(exact_dsa_weights, raw_prefill_config)
        # Shape placeholders share the existing fresh buffers; no second cache allocation.
        batched_state = Ws32BatchedPrefillState(
            state, repaired_buffer,
            batched.replicated(mesh, np.asarray(prompt_length, np.int32)),
            batched.replicated(mesh, np.asarray(False, np.bool_)),
        )
    for graph, length in (batched_plan.graph_rows if batched_prefill else (
        ("prefill_chunk", args.prefill_chunk),
        ("prefill_tail", tail_length),
    )):
        if batched_prefill:
            prefill_jits[graph] = prefill_programs[graph].execute
            inputs = batched.graph_inputs(
                mesh, np.zeros(length, np.int32), batched_state,
                raw_prefill_weights, batched_wk, table_inputs[0],
                mlp_window=batched_plan.mlp_window,
            )
        else:
            prefill_programs[graph] = build_ws32_chunked_prefill_program(
                mesh, config, chunk_length=length
            )
            prefill_jits[graph] = jax.jit(
                prefill_programs[graph].execute, donate_argnums=donate
            )
            # Shape-only placeholder; no prompt value is executed during acquisition.
            chunk_ids = _replicated(jax, mesh, np.zeros(length, dtype=np.int32))
            inputs = (chunk_ids, state, weights)
            if exact_dsa_weights is not None:
                inputs = (*inputs, exact_dsa_weights)
            inputs = (*inputs, repaired_buffer, *table_inputs)
        begin_compile(graph)
        prefill_lowered[graph] = prefill_jits[graph].lower(*inputs)
        started = time.perf_counter()
        prefill_compiled[graph] = prefill_lowered[graph].compile()
        compile_seconds[graph] = time.perf_counter() - started
        compiled_memory[graph] = _compiled_memory(prefill_compiled[graph])
        record_compile(graph)
        graphs[graph], _, _ = _write_graph(
            graph=graph,
            lowered=prefill_lowered[graph],
            compiled=prefill_compiled[graph],
            hlo_dir=args.hlo_dir,
            expected_stable=getattr(args, f"expected_{graph}_stablehlo_sha256"),
            expected_optimized=getattr(
                args, f"expected_{graph}_optimized_hlo_sha256"
            ),
            hidden_size=geometry.hidden_size,
            exact_dsa=config.exact_dsa,
            strategy_nd_dense=config.strategy_nd_dense,
            host_main_rope_table=config.host_main_rope_table,
            prefill_mode=args.prefill_mode,
            block_rows=length,
            acquisition_journal=acquisition_journal,
            batched_profile=getattr(args, "batched_prefill_profile", "") if batched_prefill else "",
        )
        del inputs
        if not batched_prefill:
            del chunk_ids
    for graph in ("prefill_chunk", "prefill_tail"):
        _require_graph_authorized(
            graphs[graph], compile_only=bool(args.compile_only)
        )
    prefill_execution: dict[str, Any] | None = None

    if args.compile_only:
        observer_state = state
        observer_token = initial_token
    elif batched_prefill:
        # Do not retain compile placeholders beside the adapter's fresh state.
        # Exact/promote executables have already been deleted; observer/decode
        # are compiled only after prefill is released. Both prefill programs are
        # therefore the complete resident model-executable set at this boundary.
        state = repaired_buffer = batched_state = None
        gc.collect()
        observer_state, observer_token, prefill_execution, batched_prefill_memory = _execute_batched_prefill(
            args=args, mesh=mesh, config=raw_prefill_config, plan=batched_plan,
            prompt_tokens=prompt_token_ids, compiled=prefill_compiled,
            weights=raw_prefill_weights, wk=batched_wk, rope=table_inputs[0],
        )
        raw_prefill_weights = batched_wk = None
    else:
        if not (args.prefill_budget_seconds > 0):
            raise ValueError("WS32 numerical prefill requires a positive wall budget")
        chunk_walls: list[float] = []
        prefill_started = time.perf_counter()
        current_state = state
        current_buffer = repaired_buffer
        next_token = initial_token
        total_units = full_chunks + tail_length / args.prefill_chunk
        projected_max = 0.0
        for index in range(full_chunks):
            chunk_ids = _replicated(
                jax,
                mesh,
                prompt_token_ids[
                    index * args.prefill_chunk : (index + 1) * args.prefill_chunk
                ],
            )
            inputs = (chunk_ids, current_state, weights)
            if exact_dsa_weights is not None:
                inputs = (*inputs, exact_dsa_weights)
            inputs = (*inputs, current_buffer, *table_inputs)
            chunk_started = time.perf_counter()
            result = prefill_compiled["prefill_chunk"](*inputs)
            jax.block_until_ready(result)
            chunk_walls.append(time.perf_counter() - chunk_started)
            current_state = result.state
            current_buffer = result.repaired_index_local
            next_token = result.next_token
            del result, chunk_ids, inputs
            # Fail closed on the projection, not on the worker timeout: the
            # mean chunk wall so far, scaled to the whole prompt, must fit.
            elapsed = time.perf_counter() - prefill_started
            projected = elapsed / (index + 1) * total_units
            projected_max = max(projected_max, projected)
            print(
                f"GREENFIELD_WS32_PREFILL_CHUNK {index + 1}/{full_chunks} "
                f"wall_s={chunk_walls[-1]:.3f} projected_total_s={projected:.1f}",
                flush=True,
            )
            if projected > args.prefill_budget_seconds:
                raise RuntimeError(
                    "WS32 prefill projection exceeds the wall budget: "
                    f"{projected:.1f}s > {args.prefill_budget_seconds:.1f}s"
                )
        tail_ids = _replicated(
            jax, mesh, prompt_token_ids[full_chunks * args.prefill_chunk :]
        )
        inputs = (tail_ids, current_state, weights)
        if exact_dsa_weights is not None:
            inputs = (*inputs, exact_dsa_weights)
        inputs = (*inputs, current_buffer, *table_inputs)
        tail_started = time.perf_counter()
        result = prefill_compiled["prefill_tail"](*inputs)
        jax.block_until_ready(result)
        tail_wall = time.perf_counter() - tail_started
        current_state = result.state
        current_buffer = result.repaired_index_local
        next_token = result.next_token
        del result, tail_ids, inputs
        if config.exact_dsa:
            # Install the exact M64-repaired prompt rows exactly once, after the
            # last prompt step has scored the unrepaired rows (sealed semantics).
            observer_state = current_state._replace(
                index_cache_local=current_buffer
            )
        else:
            observer_state = current_state
        observer_token = next_token
        prefill_execution = {
            "budget_seconds": float(args.prefill_budget_seconds),
            "chunk_length": int(args.prefill_chunk),
            "chunk_wall_seconds": chunk_walls,
            "full_chunks": int(full_chunks),
            "projected_total_seconds_max": float(projected_max),
            "repaired_index_installed": bool(config.exact_dsa),
            "tail_length": int(tail_length),
            "tail_wall_seconds": float(tail_wall),
            "total_seconds": float(time.perf_counter() - prefill_started),
        }
        state = None
        current_state = None
        current_buffer = None
        repaired_buffer = None

    # The prefill executables are large and never used again.  Do not retain
    # three complete-model executables concurrently in 32 GiB HBM.
    for graph in ("prefill_chunk", "prefill_tail"):
        prefill_jits[graph].clear_cache()
    del prefill_compiled
    del prefill_lowered
    del prefill_jits
    del prefill_programs
    gc.collect()

    observer_jit = jax.jit(program.observe, donate_argnums=(1,))
    observer_inputs = (observer_token, observer_state, weights)
    if exact_dsa_weights is not None:
        observer_inputs = (*observer_inputs, exact_dsa_weights)
    observer_inputs = (*observer_inputs, *table_inputs)
    begin_compile("observer")
    observer_lowered = observer_jit.lower(*observer_inputs)
    started = time.perf_counter()
    observer_compiled = observer_lowered.compile()
    compile_seconds["observer"] = time.perf_counter() - started
    compiled_memory["observer"] = _compiled_memory(observer_compiled)
    record_compile("observer")
    graphs["observer"], _, _ = _write_graph(
        graph="observer",
        lowered=observer_lowered,
        compiled=observer_compiled,
        hlo_dir=args.hlo_dir,
        expected_stable=args.expected_observer_stablehlo_sha256,
        expected_optimized=args.expected_observer_optimized_hlo_sha256,
        hidden_size=geometry.hidden_size,
        exact_dsa=config.exact_dsa,
        strategy_nd_dense=config.strategy_nd_dense,
        host_main_rope_table=config.host_main_rope_table,
        acquisition_journal=acquisition_journal,
        batched_profile=getattr(args, "batched_prefill_profile", "") if batched_prefill else "",
    )
    _require_graph_authorized(
        graphs["observer"], compile_only=bool(args.compile_only)
    )

    observed_tokens: list[int] = []
    dsa_steps: list[dict[str, Any]] = []
    observed_dsa_positions: list[np.ndarray] = []
    observed_dsa_counts: list[np.ndarray] = []
    observed_dsa_scores: list[np.ndarray] = []
    observed_dsa_producers: np.ndarray | None = None
    current_state = observer_state
    current_token = observer_token
    if not args.compile_only:
        observed_tokens.append(_scalar_token(jax, observer_token))
        for step in range(args.observer_steps):
            observer_inputs = (current_token, current_state, weights)
            if exact_dsa_weights is not None:
                observer_inputs = (*observer_inputs, exact_dsa_weights)
            observer_inputs = (*observer_inputs, *table_inputs)
            observed = observer_compiled(*observer_inputs)
            jax.block_until_ready(observed)
            host_dsa = jax.device_get(observed.dsa)
            producers = np.asarray(host_dsa.producer_layer_ids, dtype=np.int32)
            positions = np.asarray(host_dsa.selected_positions, dtype=np.int32)
            counts = np.asarray(host_dsa.selected_valid_counts, dtype=np.int32)
            scores = np.asarray(host_dsa.selected_scores, dtype=np.float32)
            if observed_dsa_producers is None:
                observed_dsa_producers = producers
            elif not np.array_equal(observed_dsa_producers, producers):
                raise RuntimeError("WS32 DSA producer identities changed by step")
            if oracle is None:
                # §23.5: no legacy DSA capture exists at 128K/256K, so the
                # engine is held to its own contract and nothing is claimed
                # against a capture that was never taken.
                comparison = compare_ws32_dsa_within_engine(
                    producer_layer_ids=producers,
                    selected_positions=positions,
                    selected_valid_counts=counts,
                    selected_scores=scores,
                    decode_position=prompt_length + step,
                    step=step,
                    expected_producer_layer_ids=observed_dsa_producers,
                )
            else:
                comparison = compare_ws32_dsa_step(
                    producer_layer_ids=producers,
                    selected_positions=positions,
                    selected_valid_counts=counts,
                    selected_scores=scores,
                    oracle=oracle,
                    step=step,
                    adjudication=dsa_adjudication,
                )
            dsa_steps.append(comparison)
            observed_dsa_positions.append(positions)
            observed_dsa_counts.append(counts)
            observed_dsa_scores.append(scores)
            current_state = observed.result.state
            current_token = observed.result.next_token
            observed_tokens.append(_scalar_token(jax, current_token))

    # The proof-only observer is also single-use.  Release it before compiling
    # the production decoder so only one complete-model executable is live
    # during timing and tracing.
    observer_jit.clear_cache()
    del observer_compiled
    del observer_lowered
    gc.collect()

    decode_jit = jax.jit(program.execute, donate_argnums=(1,))
    decode_inputs = (current_token, current_state, weights)
    if exact_dsa_weights is not None:
        decode_inputs = (*decode_inputs, exact_dsa_weights)
    decode_inputs = (*decode_inputs, *table_inputs)
    begin_compile("decode")
    decode_lowered = decode_jit.lower(*decode_inputs)
    started = time.perf_counter()
    decode_compiled = decode_lowered.compile()
    compile_seconds["decode"] = time.perf_counter() - started
    compiled_memory["decode"] = _compiled_memory(decode_compiled)
    record_compile("decode")
    graphs["decode"], _, _ = _write_graph(
        graph="decode",
        lowered=decode_lowered,
        compiled=decode_compiled,
        hlo_dir=args.hlo_dir,
        expected_stable=args.expected_decode_stablehlo_sha256,
        expected_optimized=args.expected_decode_optimized_hlo_sha256,
        hidden_size=geometry.hidden_size,
        exact_dsa=config.exact_dsa,
        strategy_nd_dense=config.strategy_nd_dense,
        host_main_rope_table=config.host_main_rope_table,
        acquisition_journal=acquisition_journal,
        batched_profile=getattr(args, "batched_prefill_profile", "") if batched_prefill else "",
    )
    _require_graph_authorized(
        graphs["decode"], compile_only=bool(args.compile_only)
    )
    probe_jit = jax.jit(program.probe_cache_write)
    begin_compile("cache_probe")
    probe_lowered = probe_jit.lower(current_state)
    started = time.perf_counter()
    probe_compiled = probe_lowered.compile()
    compile_seconds["cache_probe"] = time.perf_counter() - started
    compiled_memory["cache_probe"] = _compiled_memory(probe_compiled)
    record_compile("cache_probe")
    graphs["cache_probe"], _, _ = _write_graph(
        graph="cache_probe",
        lowered=probe_lowered,
        compiled=probe_compiled,
        hlo_dir=args.hlo_dir,
        expected_stable=args.expected_cache_probe_stablehlo_sha256,
        expected_optimized=args.expected_cache_probe_optimized_hlo_sha256,
        hidden_size=geometry.hidden_size,
        exact_dsa=config.exact_dsa,
        strategy_nd_dense=config.strategy_nd_dense,
        host_main_rope_table=config.host_main_rope_table,
        acquisition_journal=acquisition_journal,
        batched_profile=getattr(args, "batched_prefill_profile", "") if batched_prefill else "",
    )
    _require_graph_authorized(
        graphs["cache_probe"], compile_only=bool(args.compile_only)
    )
    if acquisition_journal is not None:
        acquisition_journal.close()

    batched_identity = {}
    if batched_prefill and not args.compile_only:
        from glm_tpu.greenfield.validation.ws32_prefill_admission import short_numerical_identity

        batched_identity = short_numerical_identity(profile=args.batched_prefill_profile)
    prevalidation: dict[str, Any] = {
        "artifact_kind": "greenfield_ws32_short_decoder_prevalidation",
        "checkpoint_manifest_sha256": checkpoint.manifest["manifest_sha256"],
        "checkpoint_success_sha256": checkpoint.success["success_sha256"],
        "checkpoint_transport": args.checkpoint_transport,
        "checkpoint_verified_device_slots": list(local_hash_slots),
        "code_hash": args.expected_code_hash,
        "compile_only": bool(args.compile_only),
        "compile_seconds": compile_seconds,
        "compiled_memory_analysis": compiled_memory,
        "context_capacity": args.context_capacity,
        "device_memory_after_compile": [
            _memory_stats(device) for device in jax.local_devices()
        ],
        "base_device_memory_after_load": list(
            base_device_memory_after_load
        ),
        "device_memory_after_load": list(device_memory_after_load),
        "device_memory_before_load": list(device_memory_before_load),
        "dsa_adjudication": dsa_adjudication_record,
        "evidence_layout": EVIDENCE_LAYOUT_V2,
        "dsa_oracle_manifest_sha256": (
            None if oracle is None else oracle.dsa_manifest["manifest_sha256"]
        ),
        "dsa_oracle_success_sha256": (
            None if oracle is None else oracle.dsa_success_sha256
        ),
        "dsa_association_summary_sha256": (
            args.dsa_association_summary_sha256
        ),
        "dsa_association_success_sha256": (
            args.dsa_association_success_sha256
        ),
        "exact_dsa": config.exact_dsa,
        "graphs": graphs,
        "hostname": socket.gethostname(),
        "jax_process_index": int(jax.process_index()),
        "launch_process_id": args.process_id,
        "load_seconds": load_seconds,
        "local_device_slots": list(local_device_slots),
        "mesh_sha256": physical_mesh.mesh_hash,
        "main_rope_table": main_rope_table_record,
        "prefill_chunk_length": int(args.prefill_chunk),
        **({"prefill_mode": args.prefill_mode, "batched_prefill_plan": batched_plan.identity()} if batched_prefill else {}),
        "prefill_execution": prefill_execution,
        **({
            "batched_prefill_memory": batched_prefill_memory,
            "batched_prefill_profile": args.batched_prefill_profile,
        } if batched_prefill and not args.compile_only else {}),
        **batched_identity,
        "prompt_length": int(prompt_token_ids.size),
        "rotary_diagnostic": rotary_diagnostic,
        "source_inventory_sha256": inventory.inventory_sha256,
        "strategy_nd_dense": config.strategy_nd_dense,
        "strategy_nd_dense_overlay": (
            None
            if dense_overlay is None or loaded_dense_overlay is None
            else {
                "local_records": list(loaded_dense_overlay.local_records),
                "manifest_file_sha256": (
                    dense_overlay.manifest_file_sha256
                ),
                "manifest_sha256": dense_overlay.manifest[
                    "manifest_sha256"
                ],
                "success_file_sha256": dense_overlay.success_file_sha256,
            }
        ),
        "long_context": long_context_record,
        "token_oracle_manifest_sha256": (
            None if oracle is None else oracle.token_manifest["manifest_sha256"]
        ),
        "token_oracle_success_sha256": (
            None if oracle is None else oracle.token_success_sha256
        ),
        "topology_fleet_sha256": fleet_sha,
        "topology_sha256": topology.topology_hash,
        "xla_python_client_mem_fraction": _XLA_MEMORY_FRACTION,
    }
    _atomic_json(args.hlo_dir / "prevalidation.json", prevalidation)
    graph_passed = all(value["passed"] for value in graphs.values())
    if args.compile_only:
        _publish_acquisition_result(
            prevalidation, output=args.output, exact_dsa=config.exact_dsa
        )
        print(
            f"GREENFIELD_WS32_SHORT_HLO_ACQUIRED rank={args.process_id}",
            flush=True,
        )
        return 0
    if not graph_passed:
        raise RuntimeError("WS32 complete pre-execution HLO contract failed")

    for _ in range(args.warmup):
        decode_inputs = (current_token, current_state, weights)
        if exact_dsa_weights is not None:
            decode_inputs = (*decode_inputs, exact_dsa_weights)
        decode_inputs = (*decode_inputs, *table_inputs)
        result = decode_compiled(*decode_inputs)
        jax.block_until_ready(result)
        current_state = result.state
        current_token = result.next_token
        observed_tokens.append(_scalar_token(jax, current_token))

    samples_ms = []
    for _ in range(args.iterations):
        started_ns = time.perf_counter_ns()
        decode_inputs = (current_token, current_state, weights)
        if exact_dsa_weights is not None:
            decode_inputs = (*decode_inputs, exact_dsa_weights)
        decode_inputs = (*decode_inputs, *table_inputs)
        result = decode_compiled(*decode_inputs)
        jax.block_until_ready(result)
        samples_ms.append((time.perf_counter_ns() - started_ns) / 1_000_000.0)
        current_state = result.state
        current_token = result.next_token
        observed_tokens.append(_scalar_token(jax, current_token))

    args.trace_dir.mkdir(parents=True, exist_ok=False)
    with jax.profiler.trace(str(args.trace_dir), create_perfetto_link=False):
        for _ in range(args.trace_steps):
            decode_inputs = (current_token, current_state, weights)
            if exact_dsa_weights is not None:
                decode_inputs = (*decode_inputs, exact_dsa_weights)
            decode_inputs = (*decode_inputs, *table_inputs)
            result = decode_compiled(*decode_inputs)
            jax.block_until_ready(result)
            current_state = result.state
            current_token = result.next_token
            observed_tokens.append(_scalar_token(jax, current_token))

    probe = probe_compiled(current_state)
    jax.block_until_ready(probe)
    host_probe = jax.device_get(probe)
    expected_probe_position = required_capacity - 1
    cache = validate_ws32_cache_probe(
        position=np.asarray(host_probe.position),
        kv_rows=np.asarray(host_probe.kv_rows),
        index_rows=np.asarray(host_probe.index_rows),
        contract_valid=np.asarray(host_probe.contract_valid),
        expected_position=expected_probe_position,
        num_layers=geometry.num_layers,
        full_indexer_count=len(config.full_index_slots),
        packed_cache_width=config.packed_cache_width,
        index_width=geometry.dsa_indexer_head_dim,
    )
    if observed_dsa_producers is None:
        raise RuntimeError("WS32 numerical run produced no DSA observations")
    tensor_record = _atomic_npz(
        args.tensor_output,
        cache_contract_valid=np.asarray(
            host_probe.contract_valid, dtype=np.bool_
        ),
        cache_index_bfloat16_bits=np.ascontiguousarray(
            np.asarray(host_probe.index_rows)
        ).view(np.uint16),
        cache_kv_bfloat16_bits=np.ascontiguousarray(
            np.asarray(host_probe.kv_rows)
        ).view(np.uint16),
        cache_position=np.asarray(host_probe.position, dtype=np.int32),
        dsa_producer_layer_ids=observed_dsa_producers,
        dsa_selected_positions=np.stack(observed_dsa_positions, axis=0),
        dsa_selected_scores=np.stack(observed_dsa_scores, axis=0),
        dsa_selected_valid_counts=np.stack(observed_dsa_counts, axis=0),
    )
    if oracle is None:
        tokens = _long_context_token_result(
            observed_tokens, long_context, tokenizer_root=args.tokenizer_root
        )
    else:
        oracle_token_count = min(
            len(observed_tokens), int(oracle.generated_token_ids.size)
        )
        tokens = compare_ws32_raw_tokens(
            observed_tokens[:oracle_token_count], oracle
        )
    state_position = np.asarray(
        jax.device_get(current_state.position), dtype=np.int32
    ).tolist()
    state_context = np.asarray(
        jax.device_get(current_state.context_lengths), dtype=np.int32
    ).tolist()
    state_health = np.asarray(
        jax.device_get(current_state.contract_valid), dtype=np.bool_
    ).tolist()
    if oracle is None:
        # §23.5: L7 passes iff the extracted passkey equals gold; L8 has no
        # correctness oracle, so only the engine's own contracts gate it.
        token_criterion = tokens["passkey_matches_gold"] is not False
    else:
        token_criterion = tokens["exact_prefix_match"]
    correctness_passed = bool(
        token_criterion
        and all(item["passed"] for item in dsa_steps)
        and cache["passed"]
        and state_position == [required_capacity]
        and state_context == [required_capacity + 1]
        and state_health == [True]
    )
    batched_final_memory = {}
    if batched_prefill:
        from glm_tpu.greenfield.validation.ws32_prefill_memory import capture_identified_device_memory

        batched_final_memory["batched_device_memory_after_execute"] = (
            capture_identified_device_memory(tuple(jax.local_devices()))
        )
    record = {
        **prevalidation,
        **batched_final_memory,
        "artifact_kind": "greenfield_ws32_short_decoder",
        "cache_write_probe": cache,
        "correctness_passed": correctness_passed,
        "device_memory_after_execute": [
            _memory_stats(device) for device in jax.local_devices()
        ],
        "dsa_steps": dsa_steps,
        "observed_generated_token_ids": observed_tokens,
        "numerical_tensors": tensor_record,
        "performance_claim": False,
        "profiler_free_timing": {
            "distribution": _distribution(samples_ms),
            "iterations": args.iterations,
            "profiler_active": False,
            "samples_ms": samples_ms,
            "warmup": args.warmup,
        },
        "schema_version": 1,
        "state": {
            "context_lengths": state_context,
            "contract_valid": state_health,
            "position": state_position,
        },
        "status": "SUCCESS" if correctness_passed else "ORACLE_MISMATCH",
        "token_comparison": tokens,
        "trace": {
            "files": _trace_files(args.trace_dir),
            "steps": args.trace_steps,
        },
    }
    _atomic_json(args.output, record)
    if not correctness_passed:
        raise RuntimeError("WS32 complete decoder correctness contract failed")
    print(
        "GREENFIELD_WS32_SHORT_DECODER_OK "
        f"rank={args.process_id} p50_ms="
        f"{record['profiler_free_timing']['distribution']['p50_ms']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
