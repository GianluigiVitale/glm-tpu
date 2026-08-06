#!/usr/bin/env python3
"""Protected one-host PP8/PP16 execution of one real GLM sparse layer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.benchmarking import (  # noqa: E402
    REAL_LAYER_OUTPUT_TOLERANCE,
    ROUTE_WEIGHT_TOLERANCE,
    compare_bounded_tensor,
    latency_distribution,
    validate_pallas_real_layer_hlo,
    validate_real_layer_hlo,
)
from glm_tpu.greenfield.checkpoint import (  # noqa: E402
    OneLayerLoadExpectation,
    PallasOneLayerLoadExpectation,
    load_one_layer,
    load_pallas_one_layer,
    resolve_stage_devices,
)
from glm_tpu.greenfield.kernels.stage_local import (  # noqa: E402
    stage_local_moe_pallas_mapped,
)
from glm_tpu.greenfield.kernels.reference import (  # noqa: E402
    GlmMoeNumericalContract,
    route_glm_noaux_tc,
    stage_local_moe_from_routes,
)
from glm_tpu.greenfield.validation import (  # noqa: E402
    inspect_one_layer_oracle,
)


EXPECTED_WORKTREE = Path("/home/gianl/glm-tpu-topology-rewrite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--topology-capture", type=Path, required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--packed-manifest-sha256", required=True)
    parser.add_argument("--source-packed-manifest-sha256")
    parser.add_argument("--packed-code-hash")
    parser.add_argument("--oracle-manifest-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--topology-sha256", required=True)
    parser.add_argument("--plan-group-sha256", required=True)
    parser.add_argument(
        "--plan-id", choices=("PP8_LP4", "PP16_LP2"), default="PP8_LP4"
    )
    parser.add_argument(
        "--kernel", choices=("reference", "pallas"), default="reference"
    )
    parser.add_argument("--stage-id", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hlo-output", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--trace-steps", type=int, default=20)
    parser.add_argument("--expert-chunk-size", type=int, default=2)
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
    return {
        "coords": list(device.coords),
        "core_on_chip": int(device.core_on_chip),
        "device_kind": str(device.device_kind),
        "id": int(device.id),
        "local_hardware_id": device.local_hardware_id,
        "platform": str(device.platform),
        "process_index": int(device.process_index),
    }


def _torch_bfloat16_numpy(tensor: Any) -> Any:
    import ml_dtypes
    import torch

    if tensor.dtype != torch.bfloat16:
        raise ValueError("oracle BF16 conversion received another dtype")
    return (
        tensor.contiguous()
        .view(torch.uint16)
        .numpy()
        .view(ml_dtypes.bfloat16)
    )


def _replicate_oracle_tensor(jax: Any, mesh: Any, tensor: Any) -> Any:
    import torch
    from jax.sharding import NamedSharding, PartitionSpec as P

    sharding = NamedSharding(mesh, P())
    if tensor.dtype == torch.bfloat16:
        host = _torch_bfloat16_numpy(tensor)
    elif tensor.dtype in (torch.float32, torch.int32):
        host = tensor.contiguous().numpy()
    else:
        raise ValueError(f"unsupported oracle input dtype {tensor.dtype}")
    value = jax.device_put(host, sharding)
    value.block_until_ready()
    return value


def _load_oracle(
    oracle_dir: Path,
    *,
    expected_manifest_sha256: str,
    expected_source_revision: str,
    pack_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from safetensors import safe_open

    manifest = inspect_one_layer_oracle(oracle_dir)
    expected = {
        "layer": pack_manifest["layer"],
        "manifest_sha256": expected_manifest_sha256,
        "model_id": pack_manifest["model_id"],
        "source_revision": expected_source_revision,
    }
    mismatches = {
        name: {"expected": value, "observed": manifest.get(name)}
        for name, value in expected.items()
        if manifest.get(name) != value
    }
    pack_geometry = pack_manifest["geometry"]
    for name in (
        "fp8_block_shape",
        "hidden_size",
        "intermediate_size",
        "num_experts",
        "top_k",
    ):
        if manifest["geometry"].get(name) != pack_geometry.get(name):
            mismatches[f"geometry.{name}"] = {
                "expected": pack_geometry.get(name),
                "observed": manifest["geometry"].get(name),
            }
    if mismatches:
        raise ValueError(f"one-layer oracle/load mismatch: {mismatches}")
    path = oracle_dir / manifest["file"]["filename"]
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    return manifest, tensors


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


def _output_checksum(values: tuple[np.ndarray, ...]) -> str:
    digest = sha256()
    for index, value in enumerate(values):
        array = np.asarray(value)
        digest.update(
            f"{index}:{array.dtype}:{array.shape}:".encode("utf-8")
        )
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _device_get_result(jax: Any, result: Any) -> tuple[np.ndarray, ...]:
    output, route_indices, route_weights = jax.device_get(result)
    return (
        np.asarray(output),
        np.asarray(route_indices),
        np.asarray(route_weights),
    )


def _pallas_stage_step(
    hidden_states: Any,
    correction_bias: Any,
    router_weight: Any,
    expert_gate_bits: Any,
    expert_gate_scale: Any,
    expert_up_bits: Any,
    expert_up_scale: Any,
    expert_down_bits: Any,
    expert_down_scale: Any,
    shared_gate_bits: Any,
    shared_gate_scale: Any,
    shared_up_bits: Any,
    shared_up_scale: Any,
    shared_down_bits: Any,
    shared_down_scale: Any,
    local_expert_shard: Any,
    *,
    axis_name: str,
    contract: GlmMoeNumericalContract,
) -> tuple[Any, Any, Any]:
    """Adapt runner input order to the Pallas kernel's routing contract."""
    return stage_local_moe_pallas_mapped(
        hidden_states,
        router_weight,
        correction_bias,
        expert_gate_bits,
        expert_gate_scale,
        expert_up_bits,
        expert_up_scale,
        expert_down_bits,
        expert_down_scale,
        shared_gate_bits,
        shared_gate_scale,
        shared_up_bits,
        shared_up_scale,
        shared_down_bits,
        shared_down_scale,
        local_expert_shard,
        axis_name=axis_name,
        contract=contract,
    )


def _correctness_record(
    jax: Any,
    compiled: Any,
    inputs: tuple[Any, ...],
    oracle: Mapping[str, Any],
    case: str,
) -> dict[str, Any]:
    observed = _device_get_result(jax, compiled(*inputs))
    output, route_indices, route_weights = observed
    expected_indices = oracle[f"{case}_route_indices"].numpy()
    expected_weights = oracle[f"{case}_route_weights"].numpy()
    expected_output = oracle[f"{case}_output"].float().numpy()
    indices_exact = np.array_equal(route_indices, expected_indices)
    weight_comparison = compare_bounded_tensor(
        route_weights,
        expected_weights,
        ROUTE_WEIGHT_TOLERANCE,
    )
    output_comparison = compare_bounded_tensor(
        output,
        expected_output,
        REAL_LAYER_OUTPUT_TOLERANCE,
    )
    passed = (
        indices_exact
        and weight_comparison["passed"]
        and output_comparison["passed"]
    )
    return {
        "case": case,
        "observed_route_indices": route_indices.tolist(),
        "oracle_route_indices": expected_indices.tolist(),
        "output": output_comparison,
        "output_checksum": _output_checksum(observed),
        "passed": passed,
        "route_indices_exact": indices_exact,
        "route_weights": weight_comparison,
    }


def _measure_case(
    jax: Any,
    compiled: Any,
    inputs: tuple[Any, ...],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    first = compiled(*inputs)
    jax.block_until_ready(first)
    first_checksum = _output_checksum(_device_get_result(jax, first))
    for _ in range(warmup - 1):
        value = compiled(*inputs)
        jax.block_until_ready(value)
    samples_ms = []
    last = first
    for _ in range(iterations):
        started = time.perf_counter_ns()
        last = compiled(*inputs)
        jax.block_until_ready(last)
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    last_checksum = _output_checksum(_device_get_result(jax, last))
    if first_checksum != last_checksum:
        raise RuntimeError("identical one-layer invocations changed output bytes")
    distribution = latency_distribution(samples_ms)
    return {
        "first_checksum": first_checksum,
        "iterations": iterations,
        "last_checksum": last_checksum,
        "latency": distribution.to_dict(),
        "profiler_active": False,
        "samples_ms": samples_ms,
        "synchronized_each_iteration": True,
        "warmup": warmup,
    }


def main() -> int:
    args = parse_args()
    if (
        args.warmup < 200
        or args.iterations < 1000
        or args.trace_steps != 20
        or args.expert_chunk_size <= 0
    ):
        raise ValueError(
            "protected layer requires warmup>=200, iterations>=1000, "
            "trace_steps=20, and a positive loader chunk"
        )
    if args.plan_id == "PP16_LP2" and args.stage_id is None:
        raise ValueError("protected PP16 layer requires an explicit stage id")
    if args.plan_id == "PP8_LP4" and args.stage_id is not None:
        raise ValueError("protected PP8 layer derives its sole local stage")
    if args.kernel == "pallas" and (
        args.source_packed_manifest_sha256 is None
        or args.packed_code_hash is None
    ):
        raise ValueError(
            "Pallas layer requires source manifest and pack code identities"
        )
    code_hash = _git_head()
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale code hash: expected {args.expected_code_hash}, found {code_hash}"
        )
    if REPO != EXPECTED_WORKTREE:
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    if args.output.exists() or args.hlo_output.exists() or args.trace_root.exists():
        raise FileExistsError("real-layer outputs are append-only")

    import jax
    import jax.numpy as jnp

    local_devices = tuple(jax.local_devices())
    # TPU v4 libtpu rejects a standalone local 2x1x1 slice for devices 0,1.
    # PP16 therefore initializes the proven four-chip host subcube but places
    # all arrays and the executable on the selected adjacent two-chip mesh.
    expected_runtime_size = 4
    if (
        jax.default_backend() != "tpu"
        or len(local_devices) != expected_runtime_size
        or jax.device_count() != expected_runtime_size
        or jax.process_count() != 1
    ):
        raise RuntimeError(
            f"protected {args.plan_id} runner requires the standalone "
            "four-chip TPU v4 host subcube"
        )
    visible_raw = os.environ.get("TPU_VISIBLE_DEVICES", "")
    try:
        visible_device_indices = tuple(
            int(value) for value in visible_raw.split(",") if value != ""
        )
    except ValueError as error:
        raise ValueError(
            f"invalid TPU_VISIBLE_DEVICES={visible_raw!r}"
        ) from error
    if len(visible_device_indices) != expected_runtime_size:
        raise ValueError(
            f"TPU_VISIBLE_DEVICES must identify {expected_runtime_size} captured "
            f"host-local chips, got {visible_device_indices}"
        )
    if args.kernel == "pallas":
        expectation = PallasOneLayerLoadExpectation(
            manifest_sha256=args.packed_manifest_sha256,
            source_manifest_sha256=args.source_packed_manifest_sha256,
            code_hash=args.packed_code_hash,
            source_revision=args.source_revision,
            topology_hash=args.topology_sha256,
            plan_group_hash=args.plan_group_sha256,
            plan_id=args.plan_id,
        )
    else:
        expectation = OneLayerLoadExpectation(
            manifest_sha256=args.packed_manifest_sha256,
            source_revision=args.source_revision,
            topology_hash=args.topology_sha256,
            plan_group_hash=args.plan_group_sha256,
            plan_id=args.plan_id,
        )
    resolution = resolve_stage_devices(
        local_devices,
        args.topology_capture,
        expectation,
        stage_id=args.stage_id,
        visible_device_indices=visible_device_indices,
    )
    load_started = time.perf_counter()
    if args.kernel == "pallas":
        loaded = load_pallas_one_layer(
            args.artifact_dir,
            expectation,
            resolution,
            expert_chunk_size=args.expert_chunk_size,
        )
    else:
        loaded = load_one_layer(
            args.artifact_dir,
            expectation,
            resolution,
            expert_chunk_size=args.expert_chunk_size,
        )
    load_seconds = time.perf_counter() - load_started
    oracle_manifest, oracle = _load_oracle(
        args.oracle_dir,
        expected_manifest_sha256=args.oracle_manifest_sha256,
        expected_source_revision=args.source_revision,
        pack_manifest=loaded.manifest,
    )
    packed_bias = np.asarray(jax.device_get(loaded.correction_bias))
    oracle_bias = oracle["correction_bias"].numpy()
    if not np.array_equal(packed_bias, oracle_bias):
        raise ValueError("packed correction bias differs from independent oracle")

    hidden = _replicate_oracle_tensor(
        jax, loaded.mesh, oracle["hidden_states"]
    )
    concentrated_bias = _replicate_oracle_tensor(
        jax, loaded.mesh, oracle["concentrated_correction_bias"]
    )
    geometry = loaded.manifest["geometry"]
    contract = GlmMoeNumericalContract(
        hidden_size=int(geometry["hidden_size"]),
        intermediate_size=int(geometry["intermediate_size"]),
        num_experts=int(geometry["num_experts"]),
        top_k=int(geometry["top_k"]),
        stage_size=int(geometry["stage_size"]),
        routed_scaling_factor=float(
            oracle_manifest["geometry"]["routed_scaling_factor"]
        ),
        fp8_block_shape=tuple(geometry["fp8_block_shape"]),
    )

    if args.kernel == "pallas":
        from jax import lax
        from jax.sharding import PartitionSpec as P

        mapped_step = jax.shard_map(
            lambda *values: _pallas_stage_step(
                *values,
                local_expert_shard=lax.axis_index("expert").astype(jnp.int32),
                axis_name="expert",
                contract=contract,
            ),
            mesh=loaded.mesh,
            in_specs=(
                P(),
                P(),
                P(),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None),
                P("expert", None),
                P("expert", None),
                P("expert", None),
                P(None, "expert"),
                P(None, "expert"),
            ),
            out_specs=(P(), P(), P()),
            check_vma=False,
        )

        def step_fun_impl(*values: Any) -> tuple[Any, Any, Any]:
            return mapped_step(*values)

        normal_inputs = (
            hidden,
            loaded.correction_bias,
            loaded.router_weight,
            loaded.expert_gate_bits,
            loaded.expert_gate_scale,
            loaded.expert_up_bits,
            loaded.expert_up_scale,
            loaded.expert_down_bits,
            loaded.expert_down_scale,
            loaded.shared_gate_bits,
            loaded.shared_gate_scale,
            loaded.shared_up_bits,
            loaded.shared_up_scale,
            loaded.shared_down_bits,
            loaded.shared_down_scale,
        )
    else:
        def step_fun_impl(
            hidden_states: Any,
            correction_bias: Any,
            router_weight: Any,
            expert_gate: Any,
            expert_up: Any,
            expert_down: Any,
            shared_gate: Any,
            shared_up: Any,
            shared_down: Any,
        ) -> tuple[Any, Any, Any]:
            route_indices, route_weights = route_glm_noaux_tc(
                hidden_states,
                router_weight,
                correction_bias,
                top_k=contract.top_k,
            )
            output = stage_local_moe_from_routes(
                hidden_states,
                route_indices,
                route_weights,
                expert_gate,
                expert_up,
                expert_down,
                shared_gate,
                shared_up,
                shared_down,
                mesh=loaded.mesh,
                contract=contract,
            )
            return output, route_indices, route_weights

        normal_inputs = (
            hidden,
            loaded.correction_bias,
            loaded.router_weight,
            loaded.expert_gate,
            loaded.expert_up,
            loaded.expert_down,
            loaded.shared_gate,
            loaded.shared_up,
            loaded.shared_down,
        )
    concentrated_inputs = (
        hidden,
        concentrated_bias,
        *normal_inputs[2:],
    )
    compile_started = time.perf_counter()
    compiled = jax.jit(step_fun_impl).lower(*normal_inputs).compile()
    compile_seconds = time.perf_counter() - compile_started
    optimized_hlo = compiled.as_text()
    args.hlo_output.parent.mkdir(parents=True, exist_ok=True)
    args.hlo_output.write_text(optimized_hlo)
    hlo_sha256 = sha256(optimized_hlo.encode()).hexdigest()
    if args.kernel == "pallas":
        hlo_contract = validate_pallas_real_layer_hlo(
            optimized_hlo,
            hidden_size=contract.hidden_size,
            intermediate_size=contract.intermediate_size,
            local_experts=contract.local_experts,
            stage_size=contract.stage_size,
        )
    else:
        hlo_contract = validate_real_layer_hlo(
            optimized_hlo,
            hidden_size=contract.hidden_size,
            stage_size=contract.stage_size,
        )
    _atomic_write(
        args.hlo_output.with_suffix(".contract.json"), hlo_contract
    )
    if not hlo_contract["passed"]:
        raise RuntimeError(
            f"real-layer optimized HLO rejected: {hlo_contract['violations']}"
        )
    device_memory_after_compile = [
        _memory_stats(device) for device in resolution.devices
    ]
    correctness = {
        case: _correctness_record(
            jax,
            compiled,
            normal_inputs if case == "normal" else concentrated_inputs,
            oracle,
            case,
        )
        for case in ("normal", "concentrated")
    }
    if not all(record["passed"] for record in correctness.values()):
        raise RuntimeError(f"real-layer correctness rejected: {correctness}")

    timing = {
        "normal": _measure_case(
            jax,
            compiled,
            normal_inputs,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "concentrated": _measure_case(
            jax,
            compiled,
            concentrated_inputs,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
    }
    device_memory_after_timing = [
        _memory_stats(device) for device in resolution.devices
    ]

    args.trace_root.mkdir(parents=True, exist_ok=False)
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    tracing = False
    try:
        jax.profiler.start_trace(
            str(args.trace_root), profiler_options=options
        )
        tracing = True
        for step in range(args.trace_steps):
            inputs = normal_inputs if step % 2 == 0 else concentrated_inputs
            with jax.profiler.TraceAnnotation(
                "greenfield_real_one_layer_step",
                step_num=step,
                route_case="normal" if step % 2 == 0 else "concentrated",
            ):
                value = compiled(*inputs)
                jax.block_until_ready(value)
        jax.profiler.stop_trace()
        tracing = False
    finally:
        if tracing:
            jax.profiler.stop_trace()
    xplanes = tuple(sorted(args.trace_root.rglob("*.xplane.pb")))
    if len(xplanes) != 1:
        raise RuntimeError(
            f"expected exactly one fresh local XPlane, found {len(xplanes)}"
        )
    trace_record = {
        "alternating_cases": True,
        "files": [
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in xplanes
        ],
        "profiler_started_after_all_timing": True,
        "steps": args.trace_steps,
    }
    persistent_bytes_per_device = max(
        int(record["payload_byte_count"])
        for record in loaded.manifest["files"]
    )
    record = {
        "captured_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "code_hash": code_hash,
        "compile_seconds": compile_seconds,
        "compiled_memory_analysis": _compiled_memory_analysis(compiled),
        "correctness": correctness,
        "device_memory_after_compile": device_memory_after_compile,
        "device_memory_after_timing": device_memory_after_timing,
        "expected_persistent_weight_bytes_per_device": (
            persistent_bytes_per_device
        ),
        "hlo": {
            "contract": hlo_contract,
            "path": str(args.hlo_output),
            "sha256": hlo_sha256,
        },
        "hostname": socket.gethostname(),
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
            "visible_device_indices": list(visible_device_indices),
            "version": jax.__version__,
        },
        "load": {**loaded.load_record, "seconds": load_seconds},
        "model_id": loaded.manifest["model_id"],
        "kernel": args.kernel,
        "oracle": {
            "file_sha256": oracle_manifest["file"]["sha256"],
            "manifest_sha256": oracle_manifest["manifest_sha256"],
        },
        "packed_checkpoint": {
            "code_hash": loaded.manifest["code_hash"],
            "layout_sha256": loaded.manifest.get("layout", {}).get(
                "layout_sha256"
            ),
            "manifest_sha256": loaded.manifest["manifest_sha256"],
            "packed_payload_byte_count": loaded.manifest[
            "packed_payload_byte_count"
            ],
            "source_manifest_sha256": loaded.manifest.get(
                "source_manifest_sha256"
            ),
        },
        "plan_group_sha256": args.plan_group_sha256,
        "plan_id": args.plan_id,
        "profiler_free_timing": True,
        "schema_version": 1,
        "source_revision": args.source_revision,
        "status": "SUCCESS",
        "timing": timing,
        "topology_sha256": args.topology_sha256,
        "trace": trace_record,
    }
    _atomic_write(args.output, record)
    print(
        "GREENFIELD_REAL_ONE_LAYER_OK "
        f"plan={args.plan_id} "
        f"kernel={args.kernel} "
        f"host={record['hostname']} stage={resolution.stage_id} "
        f"normal_p50_ms={timing['normal']['latency']['p50_ms']:.6f} "
        "concentrated_p50_ms="
        f"{timing['concentrated']['latency']['p50_ms']:.6f} "
        f"hlo={hlo_sha256} output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
