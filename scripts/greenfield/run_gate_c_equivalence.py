#!/usr/bin/env python3
"""Run the real PP8 dense/DSA/IndexShare Gate C equivalence proof.

The protected path consumes only the bounded final-owner checkpoint and the
independent raw-source oracle.  The producer's score-ordered DSA result is fed
directly to the IndexShare consumer; it is never replaced by oracle state.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.benchmarking import (  # noqa: E402
    TensorTolerance,
    compare_bounded_tensor,
    stage_local_dense_gate_c,
    stage_local_dsa_fp8_gate_c,
    stage_local_dsa_gate_c,
    stage_local_index_share_gate_c,
    validate_gate_c_hlo,
    validate_sparse_attention_integration_hlo,
)
from glm_tpu.greenfield.checkpoint import (  # noqa: E402
    GateCLoadExpectation,
    StageDeviceResolution,
    load_gate_c_checkpoint,
    read_gate_c_layout,
    resolve_stage_devices,
)
from glm_tpu.greenfield.kernels.reference import (  # noqa: E402
    DsaNumericalContract,
    MlaNumericalContract,
    StageLocalKvLayout,
)
from glm_tpu.greenfield.validation import inspect_gate_c_oracle  # noqa: E402


EXPECTED_WORKTREE = Path("/home/gianl/glm-tpu-topology-rewrite")
BF16_INTERNAL_TOLERANCE = TensorTolerance(
    max_abs=0.5,
    p99_abs=0.125,
    mean_abs=0.03,
)
BF16_OUTPUT_TOLERANCE = TensorTolerance(
    max_abs=0.125,
    p99_abs=0.0625,
    mean_abs=0.02,
)
FP32_INTERNAL_TOLERANCE = TensorTolerance(
    max_abs=0.125,
    p99_abs=0.03125,
    mean_abs=0.01,
)
FP32_LSE_TOLERANCE = TensorTolerance(
    max_abs=0.01,
    p99_abs=0.005,
    mean_abs=0.002,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--topology-capture", type=Path, required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--packed-code-hash", required=True)
    parser.add_argument("--packed-manifest-sha256", required=True)
    parser.add_argument("--layout-manifest-sha256", required=True)
    parser.add_argument("--parent-layout-manifest-sha256", required=True)
    parser.add_argument("--oracle-manifest-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--topology-sha256", required=True)
    parser.add_argument("--plan-group-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hlo-dir", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--trace-steps", type=int, default=20)
    parser.add_argument(
        "--dsa-linear-backend",
        choices=("reference", "pallas"),
        default="reference",
    )
    parser.add_argument(
        "--sparse-attention-backend",
        choices=("reference", "pallas"),
        default="reference",
    )
    parser.add_argument(
        "--development-forced-cpu",
        action="store_true",
        help="Allow exactly four forced CPU devices; never protected evidence.",
    )
    return parser.parse_args()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(
            value,
            allow_nan=False,
            default=_json_default,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _memory_stats(device: object) -> dict[str, int] | None:
    value = device.memory_stats()
    if value is None:
        return None
    return {
        str(name): int(number)
        for name, number in value.items()
        if isinstance(number, int) and not isinstance(number, bool)
    }


def _device_record(device: object) -> dict[str, Any]:
    record = {
        "device_kind": str(device.device_kind),
        "id": int(device.id),
        "platform": str(device.platform),
        "process_index": int(device.process_index),
    }
    for name in ("coords", "core_on_chip", "local_hardware_id"):
        if hasattr(device, name):
            value = getattr(device, name)
            record[name] = list(value) if name == "coords" else value
    return record


def _compiled_memory_analysis(compiled: Any) -> dict[str, int | None]:
    analysis = compiled.memory_analysis()
    names = (
        "alias_size_in_bytes",
        "argument_size_in_bytes",
        "generated_code_size_in_bytes",
        "host_alias_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_generated_code_size_in_bytes",
        "host_output_size_in_bytes",
        "host_temp_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
    )
    return {
        name: (
            None
            if getattr(analysis, name, None) is None
            else int(getattr(analysis, name))
        )
        for name in names
    }


def _torch_numpy(tensor: Any) -> np.ndarray:
    import ml_dtypes
    import torch

    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    if tensor.dtype in (torch.float32, torch.int32):
        return tensor.numpy()
    raise ValueError(f"unsupported oracle dtype {tensor.dtype}")


def _put(jax: Any, mesh: Any, value: Any, spec: Any) -> Any:
    from jax.sharding import NamedSharding

    placed = jax.device_put(value, NamedSharding(mesh, spec))
    placed.block_until_ready()
    return placed


def _load_oracle(
    oracle_dir: Path,
    *,
    expectation: GateCLoadExpectation,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from safetensors import safe_open

    manifest = inspect_gate_c_oracle(oracle_dir)
    expected = {
        "manifest_sha256": expectation.oracle_manifest_sha256,
        "model_id": "zai-org/GLM-5.2-FP8",
        "source_revision": expectation.source_revision,
    }
    mismatches = {
        name: {"expected": value, "observed": manifest.get(name)}
        for name, value in expected.items()
        if manifest.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Gate C oracle identity mismatch: {mismatches}")
    path = oracle_dir / manifest["file"]["filename"]
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    return manifest, tensors


def _page_geometry(
    context_length: int, layout: StageLocalKvLayout
) -> tuple[int, int]:
    pages = math.ceil(context_length / layout.logical_page_size)
    return pages, pages * layout.local_rows_per_page


def pack_history_by_owner(
    history: np.ndarray,
    positions: np.ndarray,
    *,
    layout: StageLocalKvLayout,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack token-major history into page-striped final local ownership."""

    history = np.asarray(history)
    positions = np.asarray(positions, dtype=np.int32)
    if history.ndim != 2 or positions.shape != (history.shape[0],):
        raise ValueError("history/position shapes do not describe token rows")
    if not np.array_equal(positions, np.arange(history.shape[0], dtype=np.int32)):
        raise ValueError("Gate C history positions must be canonical and contiguous")
    pages, local_capacity = _page_geometry(history.shape[0], layout)
    del pages
    packed = np.zeros(
        (layout.local_parallel_size, local_capacity, history.shape[1]),
        dtype=history.dtype,
    )
    packed_positions = np.full(
        (layout.local_parallel_size, local_capacity), -1, dtype=np.int32
    )
    logical_page = positions // layout.logical_page_size
    within_page = positions % layout.logical_page_size
    owners = within_page // layout.local_rows_per_page
    local_rows = within_page % layout.local_rows_per_page
    local_indices = logical_page * layout.local_rows_per_page + local_rows
    packed[owners, local_indices] = history
    packed_positions[owners, local_indices] = positions
    return packed, packed_positions


def pack_cache_by_owner(
    cache: np.ndarray,
    *,
    layout: StageLocalKvLayout,
) -> np.ndarray:
    """Pack a token-major cache into ``[owner,page,local_row,width]``."""

    cache = np.asarray(cache)
    if cache.ndim != 2 or cache.shape[1] != layout.packed_cache_width:
        raise ValueError("Gate C cache shape disagrees with its layout")
    pages, _ = _page_geometry(cache.shape[0], layout)
    packed = np.zeros(
        (
            layout.local_parallel_size,
            pages,
            layout.local_rows_per_page,
            layout.packed_cache_width,
        ),
        dtype=cache.dtype,
    )
    positions = np.arange(cache.shape[0], dtype=np.int32)
    logical_page = positions // layout.logical_page_size
    within_page = positions % layout.logical_page_size
    owners = within_page // layout.local_rows_per_page
    local_rows = within_page % layout.local_rows_per_page
    packed[owners, logical_page, local_rows] = cache
    return packed


def unpack_history_by_position(
    packed: np.ndarray,
    packed_positions: np.ndarray,
    *,
    context_length: int,
) -> np.ndarray:
    """Invert an owner-packed tensor using its authenticated global positions."""

    packed = np.asarray(packed)
    packed_positions = np.asarray(packed_positions, dtype=np.int32)
    if packed.shape[:2] != packed_positions.shape:
        raise ValueError("packed tensor and position map disagree")
    trailing = packed.shape[2:]
    output = np.empty((context_length, *trailing), dtype=packed.dtype)
    seen = np.zeros((context_length,), dtype=np.bool_)
    for owner in range(packed.shape[0]):
        positions = packed_positions[owner]
        valid = (positions >= 0) & (positions < context_length)
        chosen = positions[valid]
        if seen[chosen].any():
            raise ValueError("owner-packed tensor contains duplicate positions")
        output[chosen] = packed[owner, valid]
        seen[chosen] = True
    if not seen.all():
        raise ValueError("owner-packed tensor does not cover the full context")
    return output


def unpack_cache_by_position(
    packed: np.ndarray,
    *,
    context_length: int,
    layout: StageLocalKvLayout,
) -> np.ndarray:
    """Invert page-striped cache ownership for exact whole-cache comparison."""

    packed = np.asarray(packed)
    expected_prefix = (
        layout.local_parallel_size,
        math.ceil(context_length / layout.logical_page_size),
        layout.local_rows_per_page,
        layout.packed_cache_width,
    )
    if packed.shape != expected_prefix:
        raise ValueError(
            f"owner cache expected {expected_prefix}, got {packed.shape}"
        )
    positions = np.arange(context_length, dtype=np.int32)
    logical_page = positions // layout.logical_page_size
    within_page = positions % layout.logical_page_size
    owners = within_page // layout.local_rows_per_page
    local_rows = within_page % layout.local_rows_per_page
    return packed[owners, logical_page, local_rows]


def _exact_record(observed: Any, reference: Any) -> dict[str, Any]:
    observed_array = np.asarray(observed)
    reference_array = np.asarray(reference)
    if observed_array.shape != reference_array.shape:
        raise ValueError(
            "exact comparison shape mismatch: "
            f"{observed_array.shape} != {reference_array.shape}"
        )
    equal = np.equal(observed_array, reference_array)
    mismatch_count = int(equal.size - np.count_nonzero(equal))
    return {
        "mismatch_count": mismatch_count,
        "passed": mismatch_count == 0,
        "shape": list(observed_array.shape),
    }


def _canonical_topk_from_scores(scores: Any, *, top_k: int) -> np.ndarray:
    """Select descending FP32 scores with lowest-position tie order."""

    score_array = np.asarray(scores, dtype=np.float32)
    if score_array.ndim != 2 or not 0 < top_k <= score_array.shape[1]:
        raise ValueError("canonical top-k score shape/width is invalid")
    if not np.isfinite(score_array).all():
        raise ValueError("canonical top-k scores contain non-finite values")
    positions = np.arange(score_array.shape[1], dtype=np.int32)
    selected = [
        np.lexsort((positions, -row))[:top_k]
        for row in score_array
    ]
    return np.asarray(selected, dtype=np.int32)


def _selection_drift_record(
    observed: Any,
    reference: Any,
) -> dict[str, Any]:
    observed_array = np.asarray(observed, dtype=np.int32)
    reference_array = np.asarray(reference, dtype=np.int32)
    if observed_array.shape != reference_array.shape:
        raise ValueError("selection drift shapes disagree")
    observed_set = set(int(value) for value in observed_array.reshape(-1))
    reference_set = set(int(value) for value in reference_array.reshape(-1))
    return {
        "elementwise_mismatch_count": int(
            np.count_nonzero(observed_array != reference_array)
        ),
        "elementwise_match": bool(np.array_equal(observed_array, reference_array)),
        "observed_only_count": len(observed_set - reference_set),
        "overlap_count": len(observed_set & reference_set),
        "reference_only_count": len(reference_set - observed_set),
    }


def _reassemble_selected_cache(
    values_by_owner: Any,
    positions_by_owner: Any,
    counts_by_owner: Any,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values_by_owner)
    positions = np.asarray(positions_by_owner, dtype=np.int32)
    counts = np.asarray(counts_by_owner, dtype=np.int32)
    if (
        values.ndim != 4
        or positions.shape != values.shape[:3]
        or counts.shape != values.shape[:2]
        or values.shape[1] != 1
    ):
        raise ValueError("owner-selected cache observability shapes are invalid")
    position_parts = []
    value_parts = []
    for owner in range(values.shape[0]):
        count = int(counts[owner, 0])
        if not 0 <= count <= values.shape[2]:
            raise ValueError("owner-selected cache count is invalid")
        position_parts.append(positions[owner, 0, :count])
        value_parts.append(values[owner, 0, :count])
    joined_positions = np.concatenate(position_parts)
    joined_values = np.concatenate(value_parts)
    if len(set(int(value) for value in joined_positions)) != joined_positions.size:
        raise ValueError("owner-selected cache contains duplicate live positions")
    order = np.argsort(joined_positions, stable=True)
    return joined_values[order][None, ...], joined_positions[order][None, ...]


def _bounded_record(
    observed: Any,
    reference: Any,
    tolerance: TensorTolerance,
) -> dict[str, Any]:
    return compare_bounded_tensor(observed, reference, tolerance)


def _case_record(comparisons: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "comparisons": dict(comparisons),
        "passed": all(bool(value["passed"]) for value in comparisons.values()),
    }


def _save_hlo(
    compiled: Any,
    *,
    case: str,
    hlo_dir: Path,
    sparse_attention_backend: str = "reference",
    dsa_linear_backend: str = "reference",
) -> tuple[str, dict[str, Any], Path]:
    optimized_hlo = compiled.as_text()
    path = hlo_dir / f"{case}.optimized_hlo.txt"
    path.write_text(optimized_hlo)
    contract = validate_gate_c_hlo(
        optimized_hlo,
        case=case,
        dsa_linear_backend=dsa_linear_backend,
    )
    if case == "index_share" and sparse_attention_backend == "pallas":
        integration = validate_sparse_attention_integration_hlo(optimized_hlo)
        contract["sparse_attention_integration"] = integration
        if not integration["passed"]:
            contract["passed"] = False
            contract["violations"] = list(contract["violations"]) + list(
                integration["violations"]
            )
    _atomic_write(path.with_suffix(".contract.json"), contract)
    if not contract["passed"]:
        raise RuntimeError(
            f"Gate C {case} HLO rejected: {contract['violations']}"
        )
    return sha256(optimized_hlo.encode()).hexdigest(), contract, path


def _trace_cases(
    jax: Any,
    cases: Sequence[tuple[str, Any, tuple[Any, ...]]],
    *,
    trace_root: Path,
    trace_steps: int,
) -> dict[str, Any]:
    trace_root.mkdir(parents=True, exist_ok=False)
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    tracing = False
    try:
        jax.profiler.start_trace(str(trace_root), profiler_options=options)
        tracing = True
        for case, compiled, inputs in cases:
            for step in range(trace_steps):
                with jax.profiler.TraceAnnotation(
                    "greenfield_gate_c_step",
                    gate_c_case=case,
                    step_num=step,
                ):
                    jax.block_until_ready(compiled(*inputs))
        jax.profiler.stop_trace()
        tracing = False
    finally:
        if tracing:
            jax.profiler.stop_trace()
    xplanes = tuple(sorted(trace_root.rglob("*.xplane.pb")))
    if len(xplanes) != 1:
        raise RuntimeError(
            f"expected one local Gate C XPlane, found {len(xplanes)}"
        )
    return {
        "cases": [case for case, _, _ in cases],
        "files": [
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in xplanes
        ],
        "profiler_started_after_correctness": True,
        "steps_per_case": trace_steps,
    }


def main() -> int:
    args = parse_args()
    if REPO != EXPECTED_WORKTREE:
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    code_hash = _git_head()
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale code hash: expected {args.expected_code_hash}, found {code_hash}"
        )
    if args.output.exists() or args.hlo_dir.exists():
        raise FileExistsError("Gate C evidence outputs are append-only")
    if args.trace_steps != 20:
        raise ValueError("protected Gate C trace requires exactly 20 steps per case")
    if args.development_forced_cpu and args.trace_root is not None:
        raise ValueError("development CPU mode cannot create a protected trace")
    if not args.development_forced_cpu and args.trace_root is None:
        raise ValueError("protected TPU mode requires --trace-root")

    import jax
    from jax.sharding import PartitionSpec as P

    local_devices = tuple(jax.local_devices())
    if len(local_devices) != 4 or jax.device_count() != 4 or jax.process_count() != 1:
        raise RuntimeError("Gate C requires one isolated four-device stage")
    expectation = GateCLoadExpectation(
        packed_manifest_sha256=args.packed_manifest_sha256,
        layout_manifest_sha256=args.layout_manifest_sha256,
        parent_layout_manifest_sha256=args.parent_layout_manifest_sha256,
        oracle_manifest_sha256=args.oracle_manifest_sha256,
        source_revision=args.source_revision,
        topology_hash=args.topology_sha256,
        plan_group_hash=args.plan_group_sha256,
        packed_code_hash=args.packed_code_hash,
    )
    if args.development_forced_cpu:
        if jax.default_backend() != "cpu":
            raise RuntimeError("development mode requires forced CPU devices")
        layout_manifest = read_gate_c_layout(args.artifact_dir / "layout_manifest.json")
        destination_ids = tuple(
            int(destination["device_id"])
            for destination in sorted(
                layout_manifest["placements"][0]["destinations"],
                key=lambda value: value["device_slot"],
            )
        )
        resolution = StageDeviceResolution(
            devices=local_devices,
            coordinates=((0,), (1,), (2,), (3,)),
            captured_device_ids=destination_ids,
            captured_process_index=0,
            stage_id=0,
        )
        evidence_class = "development_forced_cpu_mechanism_only"
    else:
        if jax.default_backend() != "tpu":
            raise RuntimeError("protected Gate C requires TPU")
        visible_raw = os.environ.get("TPU_VISIBLE_DEVICES", "")
        visible = tuple(int(value) for value in visible_raw.split(",") if value)
        if visible != (0, 1, 2, 3):
            raise ValueError(
                "protected Gate C requires TPU_VISIBLE_DEVICES=0,1,2,3"
            )
        resolution = resolve_stage_devices(
            local_devices,
            args.topology_capture,
            expectation,
            visible_device_indices=visible,
        )
        evidence_class = "protected_single_host_tpu_v4_gate_c"

    oracle_manifest, oracle = _load_oracle(
        args.oracle_dir, expectation=expectation
    )
    producer = f"model.layers.{oracle_manifest['producer_layer']}"
    raw_fp8_names = (
        frozenset(
            {
                f"{producer}.self_attn.indexer.wq_b.weight",
                f"{producer}.self_attn.indexer.wk.weight",
            }
        )
        if args.dsa_linear_backend == "pallas"
        else frozenset()
    )
    load_started = time.perf_counter()
    loaded = load_gate_c_checkpoint(
        args.artifact_dir,
        expectation,
        resolution,
        raw_fp8_names=raw_fp8_names,
    )
    load_seconds = time.perf_counter() - load_started
    geometry = oracle_manifest["geometry"]
    context_length = int(geometry["context_length"])
    dsa_contract = DsaNumericalContract(
        hidden_size=int(geometry["hidden_size"]),
        q_lora_rank=int(geometry["q_lora_rank"]),
        num_heads=int(geometry["indexer_heads"]),
        head_dim=int(geometry["indexer_head_dim"]),
        rotary_dim=int(geometry["indexer_rotary_dim"]),
        top_k=int(geometry["top_k"]),
        theta=float(oracle_manifest["numerical_contract"]["rope_theta"]),
    )
    mla_contract = MlaNumericalContract(
        num_heads=int(geometry["num_attention_heads"]),
        kv_lora_rank=int(geometry["kv_lora_rank"]),
        qk_nope_head_dim=int(geometry["qk_nope_head_dim"]),
        qk_rope_head_dim=int(geometry["qk_rope_head_dim"]),
        qk_head_dim=int(geometry["qk_head_dim"]),
        v_head_dim=int(geometry["v_head_dim"]),
        packed_cache_width=int(geometry["packed_cache_width"]),
        top_k=int(geometry["top_k"]),
    )
    cache_layout = StageLocalKvLayout(
        logical_page_size=512,
        local_parallel_size=4,
        packed_cache_width=mla_contract.packed_cache_width,
    )
    weights = loaded.weights
    consumer = f"model.layers.{oracle_manifest['consumer_layer']}"
    position = np.asarray([context_length - 1], dtype=np.int32)
    lengths = np.asarray([context_length], dtype=np.int32)

    def oracle_array(name: str) -> np.ndarray:
        return _torch_numpy(oracle[name])

    args.hlo_dir.mkdir(parents=True, exist_ok=False)

    dense_inputs = (
        _put(jax, loaded.mesh, oracle_array("producer_dense_residual"), P()),
        weights[f"{producer}.post_attention_layernorm.weight"],
        weights[f"{producer}.mlp.gate_proj.weight"],
        weights[f"{producer}.mlp.up_proj.weight"],
        weights[f"{producer}.mlp.down_proj.weight"],
    )

    def dense_step(*values: Any) -> Any:
        return stage_local_dense_gate_c(*values, mesh=loaded.mesh)

    compile_started = time.perf_counter()
    dense_compiled = jax.jit(dense_step).lower(*dense_inputs).compile()
    dense_compile_seconds = time.perf_counter() - compile_started
    dense_hlo_sha, dense_hlo, dense_hlo_path = _save_hlo(
        dense_compiled, case="dense", hlo_dir=args.hlo_dir
    )
    dense = jax.device_get(dense_compiled(*dense_inputs))
    dense_record = _case_record(
        {
            "normalized": _bounded_record(
                dense.normalized,
                oracle_array("producer_dense_normalized"),
                BF16_INTERNAL_TOLERANCE,
            ),
            "gate": _bounded_record(
                dense.gate,
                oracle_array("producer_dense_gate"),
                BF16_INTERNAL_TOLERANCE,
            ),
            "up": _bounded_record(
                dense.up,
                oracle_array("producer_dense_up"),
                BF16_INTERNAL_TOLERANCE,
            ),
            "activated": _bounded_record(
                dense.activated,
                oracle_array("producer_dense_activated"),
                BF16_INTERNAL_TOLERANCE,
            ),
            "update": _bounded_record(
                dense.update,
                oracle_array("producer_dense_update"),
                BF16_OUTPUT_TOLERANCE,
            ),
            "output": _bounded_record(
                dense.output,
                oracle_array("producer_dense_output"),
                BF16_OUTPUT_TOLERANCE,
            ),
        }
    )

    history, history_positions = pack_history_by_owner(
        oracle_array("producer_history_hidden"),
        oracle_array("positions"),
        layout=cache_layout,
    )
    dsa_inputs_prefix = (
        _put(jax, loaded.mesh, oracle_array("producer_decode_residual"), P()),
        _put(jax, loaded.mesh, history, P("stage", None, None)),
        _put(jax, loaded.mesh, history_positions, P("stage", None)),
        _put(jax, loaded.mesh, position, P()),
        _put(jax, loaded.mesh, lengths, P()),
        weights[f"{producer}.input_layernorm.weight"],
        weights[f"{producer}.self_attn.q_a_proj.weight"],
        weights[f"{producer}.self_attn.q_a_layernorm.weight"],
    )
    dsa_inputs_suffix = (
        weights[f"{producer}.self_attn.indexer.k_norm.weight"],
        weights[f"{producer}.self_attn.indexer.k_norm.bias"],
        weights[f"{producer}.self_attn.indexer.weights_proj.weight"],
    )
    if args.dsa_linear_backend == "pallas":
        dsa_inputs = (
            dsa_inputs_prefix
            + (
                weights[f"{producer}.self_attn.indexer.wq_b.weight"],
                weights[f"{producer}.self_attn.indexer.wq_b.weight_scale_inv"],
                weights[f"{producer}.self_attn.indexer.wk.weight"],
                weights[f"{producer}.self_attn.indexer.wk.weight_scale_inv"],
            )
            + dsa_inputs_suffix
        )

        def dsa_step(*values: Any) -> Any:
            return stage_local_dsa_fp8_gate_c(
                *values,
                mesh=loaded.mesh,
                contract=dsa_contract,
                interpret=args.development_forced_cpu,
            )
    else:
        dsa_inputs = (
            dsa_inputs_prefix
            + (
                weights[f"{producer}.self_attn.indexer.wq_b.weight"],
                weights[f"{producer}.self_attn.indexer.wk.weight"],
            )
            + dsa_inputs_suffix
        )

        def dsa_step(*values: Any) -> Any:
            return stage_local_dsa_gate_c(
                *values, mesh=loaded.mesh, contract=dsa_contract
            )

    compile_started = time.perf_counter()
    dsa_compiled = jax.jit(dsa_step).lower(*dsa_inputs).compile()
    dsa_compile_seconds = time.perf_counter() - compile_started
    dsa_hlo_sha, dsa_hlo, dsa_hlo_path = _save_hlo(
        dsa_compiled,
        case="dsa",
        hlo_dir=args.hlo_dir,
        dsa_linear_backend=args.dsa_linear_backend,
    )
    dsa_device = dsa_compiled(*dsa_inputs)
    dsa = jax.device_get(dsa_device)
    global_keys = unpack_history_by_position(
        np.asarray(dsa.index_keys_by_owner),
        history_positions,
        context_length=context_length,
    )
    owner_scores = np.asarray(dsa.scores_by_owner)[:, 0, :]
    global_scores = unpack_history_by_position(
        owner_scores[..., None],
        history_positions,
        context_length=context_length,
    ).T
    canonical_selected = _canonical_topk_from_scores(
        global_scores, top_k=dsa_contract.top_k
    )
    selected_exact = _exact_record(
        dsa.selected_positions,
        canonical_selected,
    )
    dsa_record = _case_record(
        {
            "normalized": _bounded_record(
                dsa.normalized,
                oracle_array("producer_input_normalized"),
                BF16_INTERNAL_TOLERANCE,
            ),
            "q_residual": _bounded_record(
                dsa.q_residual,
                oracle_array("producer_q_residual"),
                BF16_INTERNAL_TOLERANCE,
            ),
            "query": _bounded_record(
                dsa.query,
                oracle_array("producer_index_query"),
                FP32_INTERNAL_TOLERANCE,
            ),
            "head_weights": _bounded_record(
                dsa.head_weights,
                oracle_array("producer_index_head_weights"),
                FP32_INTERNAL_TOLERANCE,
            ),
            "index_keys": _bounded_record(
                global_keys,
                oracle_array("producer_index_keys"),
                FP32_INTERNAL_TOLERANCE,
            ),
            "scores": _bounded_record(
                global_scores,
                oracle_array("producer_index_scores"),
                FP32_INTERNAL_TOLERANCE,
            ),
            "selected_positions_exact": selected_exact,
            "selected_scores_exact_for_device_scores": _exact_record(
                dsa.selected_scores,
                np.take_along_axis(
                    global_scores, canonical_selected, axis=1
                ),
            ),
            "selected_valid_count_exact": _exact_record(
                dsa.valid_counts,
                np.asarray([dsa_contract.top_k], dtype=np.int32),
            ),
        }
    )
    dsa_record["cross_framework_diagnostics"] = {
        "raw_oracle_selection": _selection_drift_record(
            dsa.selected_positions,
            oracle_array("producer_selected_positions"),
        ),
        "raw_oracle_selected_scores": _bounded_record(
            dsa.selected_scores,
            oracle_array("producer_selected_scores"),
            FP32_INTERNAL_TOLERANCE,
        ),
        "selection_is_not_relaxed_by_tolerance": True,
    }

    final_cache = oracle_array("consumer_cache")
    cache_by_owner = pack_cache_by_owner(final_cache, layout=cache_layout)
    prewrite_cache = np.array(cache_by_owner, copy=True)
    current = context_length - 1
    current_page = current // cache_layout.logical_page_size
    current_within = current % cache_layout.logical_page_size
    current_owner = current_within // cache_layout.local_rows_per_page
    current_local_row = current_within % cache_layout.local_rows_per_page
    prewrite_cache[current_owner, current_page, current_local_row] = 0
    block_tables = np.arange(
        prewrite_cache.shape[1], dtype=np.int32
    )[None, :]
    index_inputs = (
        _put(jax, loaded.mesh, oracle_array("consumer_decode_residual"), P()),
        _put(
            jax,
            loaded.mesh,
            prewrite_cache,
            P("stage", None, None, None),
        ),
        dsa_device.selected_positions,
        dsa_device.valid_counts,
        _put(jax, loaded.mesh, position, P()),
        _put(jax, loaded.mesh, block_tables, P()),
        _put(jax, loaded.mesh, lengths, P()),
        weights[f"{consumer}.input_layernorm.weight"],
        weights[f"{consumer}.self_attn.q_a_proj.weight"],
        weights[f"{consumer}.self_attn.q_a_layernorm.weight"],
        weights[f"{consumer}.self_attn.q_b_proj.weight"],
        weights[f"{consumer}.self_attn.kv_a_proj_with_mqa.weight"],
        weights[f"{consumer}.self_attn.kv_a_layernorm.weight"],
        weights[f"{consumer}.self_attn.kv_b_proj.weight"],
        weights[f"{consumer}.self_attn.o_proj.weight"],
    )

    def index_share_step(*values: Any) -> Any:
        return stage_local_index_share_gate_c(
            *values,
            mesh=loaded.mesh,
            contract=mla_contract,
            cache_layout=cache_layout,
            rope_theta=float(
                oracle_manifest["numerical_contract"]["rope_theta"]
            ),
            sparse_attention_backend=args.sparse_attention_backend,
            sparse_attention_interpret=args.development_forced_cpu,
        )

    compile_started = time.perf_counter()
    index_compiled = jax.jit(index_share_step).lower(*index_inputs).compile()
    index_compile_seconds = time.perf_counter() - compile_started
    index_hlo_sha, index_hlo, index_hlo_path = _save_hlo(
        index_compiled,
        case="index_share",
        hlo_dir=args.hlo_dir,
        sparse_attention_backend=args.sparse_attention_backend,
    )
    index_device = index_compiled(*index_inputs)
    index = jax.device_get(index_device)
    observed_cache = unpack_cache_by_position(
        index.cache_by_owner,
        context_length=context_length,
        layout=cache_layout,
    )
    attention_positions = np.asarray(index.attention_positions)
    selected_cache, selected_cache_positions = _reassemble_selected_cache(
        index.selected_cache_by_owner,
        index.selected_cache_positions_by_owner,
        index.selected_cache_counts_by_owner,
    )
    expected_attention_positions = np.sort(canonical_selected, axis=1)
    expected_selected_cache = observed_cache[
        expected_attention_positions[0]
    ][None, ...]
    index_record = _case_record(
        {
            "selected_state_unchanged_exact": _exact_record(
                index.selected_positions,
                dsa.selected_positions,
            ),
            "attention_positions_exact": _exact_record(
                attention_positions,
                expected_attention_positions,
            ),
            "contract_valid_exact": _exact_record(
                index.contract_valid, np.asarray([True], dtype=np.bool_)
            ),
            "normalized": _bounded_record(
                index.normalized,
                oracle_array("consumer_input_normalized"),
                BF16_INTERNAL_TOLERANCE,
            ),
            "q_residual": _bounded_record(
                index.q_residual,
                oracle_array("consumer_q_residual"),
                BF16_INTERNAL_TOLERANCE,
            ),
            "q_nope": _bounded_record(
                index.q_nope,
                oracle_array("consumer_q_nope"),
                BF16_INTERNAL_TOLERANCE,
            ),
            "q_rope": _bounded_record(
                index.q_rope,
                oracle_array("consumer_q_rope"),
                BF16_INTERNAL_TOLERANCE,
            ),
            "q_absorbed": _bounded_record(
                index.q_absorbed,
                oracle_array("consumer_q_absorbed"),
                BF16_INTERNAL_TOLERANCE,
            ),
            "current_cache_row": _bounded_record(
                index.current_cache_row,
                oracle_array("consumer_current_cache_row"),
                BF16_INTERNAL_TOLERANCE,
            ),
            "whole_live_cache": _bounded_record(
                observed_cache,
                final_cache,
                BF16_INTERNAL_TOLERANCE,
            ),
            "selected_cache_positions_exact": _exact_record(
                selected_cache_positions,
                expected_attention_positions,
            ),
            "selected_cache_values_exact": _exact_record(
                selected_cache,
                expected_selected_cache,
            ),
            "attended_latent": _bounded_record(
                index.attended_latent,
                oracle_array("consumer_attended_latent"),
                BF16_OUTPUT_TOLERANCE,
            ),
            "attention_lse": _bounded_record(
                index.attention_lse,
                oracle_array("consumer_attention_lse"),
                FP32_LSE_TOLERANCE,
            ),
            "value_states": _bounded_record(
                index.value_states,
                oracle_array("consumer_value_states"),
                BF16_OUTPUT_TOLERANCE,
            ),
            "attention_update": _bounded_record(
                index.attention_update,
                oracle_array("consumer_attention_update"),
                BF16_OUTPUT_TOLERANCE,
            ),
            "output": _bounded_record(
                index.output,
                oracle_array("consumer_attention_output"),
                BF16_OUTPUT_TOLERANCE,
            ),
        }
    )
    index_record["cross_framework_diagnostics"] = {
        "raw_oracle_attention_positions": _selection_drift_record(
            attention_positions,
            oracle_array("consumer_attention_positions"),
        ),
        "raw_oracle_selected_cache": _bounded_record(
            selected_cache,
            oracle_array("consumer_selected_cache"),
            BF16_INTERNAL_TOLERANCE,
        ),
    }

    cases = {
        "dense": dense_record,
        "dsa": dsa_record,
        "index_share": index_record,
    }
    if not all(value["passed"] for value in cases.values()):
        failure = args.output.with_suffix(".failed.json")
        _atomic_write(failure, {"cases": cases, "status": "FAILED_CORRECTNESS"})
        raise RuntimeError(f"Gate C correctness rejected; details: {failure}")

    trace = None
    if args.trace_root is not None:
        trace = _trace_cases(
            jax,
            (
                ("dense", dense_compiled, dense_inputs),
                ("dsa", dsa_compiled, dsa_inputs),
                ("index_share", index_compiled, index_inputs),
            ),
            trace_root=args.trace_root,
            trace_steps=args.trace_steps,
        )
    device_memory_after = [
        _memory_stats(device) for device in resolution.devices
    ]
    hlos = {
        "dense": {
            "compile_seconds": dense_compile_seconds,
            "contract": dense_hlo,
            "memory_analysis": _compiled_memory_analysis(dense_compiled),
            "path": str(dense_hlo_path),
            "sha256": dense_hlo_sha,
        },
        "dsa": {
            "compile_seconds": dsa_compile_seconds,
            "contract": dsa_hlo,
            "memory_analysis": _compiled_memory_analysis(dsa_compiled),
            "path": str(dsa_hlo_path),
            "sha256": dsa_hlo_sha,
        },
        "index_share": {
            "compile_seconds": index_compile_seconds,
            "contract": index_hlo,
            "memory_analysis": _compiled_memory_analysis(index_compiled),
            "path": str(index_hlo_path),
            "sha256": index_hlo_sha,
        },
    }
    record = {
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases": cases,
        "code_hash": code_hash,
        "device_memory_after": device_memory_after,
        "evidence_class": evidence_class,
        "hlo": hlos,
        "hostname": socket.gethostname(),
        "index_share_state": {
            "byte_count": int(np.asarray(dsa.selected_positions).nbytes),
            "dtype": str(np.asarray(dsa.selected_positions).dtype),
            "fed_producer_device_result_directly": True,
            "score_order_preserved": True,
            "shape": list(np.asarray(dsa.selected_positions).shape),
            "sparse_attention_backend": args.sparse_attention_backend,
        },
        "dsa_linear_backend": args.dsa_linear_backend,
        "numerical_evidence_contract": {
            "cross_framework_internal_tensors": "bounded",
            "device_score_selection_and_tie_order": "elementwise_exact",
            "raw_cpu_oracle_selection_elementwise_exact": bool(
                dsa_record["cross_framework_diagnostics"][
                    "raw_oracle_selection"
                ]["elementwise_match"]
            ),
            "raw_cpu_oracle_selection_is_not_used_as_runtime_state": True,
        },
        "jax": {
            "default_backend": jax.default_backend(),
            "device_count": jax.device_count(),
            "devices_in_runtime_order": [
                _device_record(device) for device in local_devices
            ],
            "devices_in_stage_slot_order": [
                _device_record(device) for device in resolution.devices
            ],
            "process_count": jax.process_count(),
            "version": jax.__version__,
        },
        "load": {**loaded.load_record, "seconds": load_seconds},
        "oracle": {
            "file_sha256": oracle_manifest["file"]["sha256"],
            "manifest_sha256": oracle_manifest["manifest_sha256"],
        },
        "packed_checkpoint": {
            "layout_manifest_sha256": loaded.layout["manifest_sha256"],
            "manifest_sha256": loaded.manifest["manifest_sha256"],
            "payload_bytes": loaded.layout["packed_payload_bytes"],
            "state_manifest": loaded.state_manifest,
        },
        "performance_claim": False,
        "plan_group_sha256": args.plan_group_sha256,
        "plan_id": "PP8_LP4",
        "schema_version": 1,
        "source_revision": args.source_revision,
        "status": "SUCCESS",
        "topology_sha256": args.topology_sha256,
        "trace": trace,
    }
    _atomic_write(args.output, record)
    print(
        "GREENFIELD_GATE_C_OK "
        f"backend={jax.default_backend()} "
        f"selected_exact={selected_exact['passed']} "
        f"state_bytes={record['index_share_state']['byte_count']} "
        f"output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
