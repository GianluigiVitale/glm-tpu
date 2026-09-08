#!/usr/bin/env python3
"""Worker for real-weight multirow MoE admission or opt-in phase baseline.

Requires an external reviewed eight-host controller holding both workload
leases and providing authenticated pre/post census and generation publication.
Do not launch this worker directly. No complete model/checkpoint is loaded.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.greenfield.microbench_fp8_matmul import (
    _atomic_json,
    _compiled_memory,
    _git_head,
    _memory_stats,
)

PROTOCOL = "ws32-prefill-real-moe-arithmetic-v1"
BOUNDARY_PROTOCOL = "ws32-prefill-real-moe-boundary-diagnostic-v1"
BOUNDED_PROTOCOL = "ws32-prefill-real-moe-fp32-route-sum-bounded-v1"
SCALING_PROTOCOL = "ws32-prefill-moe-equal128-b16-b128-baseline-v1"
ROWS = 17
CASES = ("normal", "concentrated")
PACK = Path(
    "/home/gianl/gcs-models/checkpoints/greenfield/glm52/layer3/WS32_2D/greenfield_ws32_one_layer_pack_20260815T070628458699950Z"
)
PACK_SHA = "4bf8679de10ebbba055e9d0be991495080388c6355fa449f393c28f4751e1f40"
ORACLE = Path(
    "/home/gianl/gcs-models/oracles/greenfield/glm52/layer3/greenfield_one_layer_oracle_20260805T162210370718434Z"
)
ORACLE_SHA = "c63ffa19820d5c2c39865ac8611fb313ffc6ebcd2f893c3507745a356bfdebff"
TOPOLOGY = Path(
    "/home/gianl/gcs-models/results/greenfield_topology_20260826T194116460015528Z/host_records"
)
TOPOLOGY_SHA = "294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559"
FLEET_SHA = "4a0c9a338d55b8be37dab79396569aa10fc9e85b3c7210d72a70abfafe72c301"
MESH_SHA = "de5f59cbadf2116745ee1dde921656424c9555c3ddc584dcdd66cb7845050a88"


def case_rows(
    hidden: np.ndarray,
    routes: np.ndarray,
    weights: np.ndarray,
    case: str,
    *,
    rows: int = ROWS,
) -> tuple[np.ndarray, ...]:
    """Real captured row0; distinct perturbed rows with supplied route semantics.

    Not observed routing occupancy or a model-router test. Normal moves expert
    IDs across owners; concentrated preserves the oracle's worst-owner routes.
    Route-slot permutations are applied to indices and weights together.
    """
    if (
        type(rows) is not int
        or rows not in (16, ROWS, 128)
        or case not in CASES
        or hidden.shape != (1, 6144)
        or routes.shape != (1, 8)
        or weights.shape != (1, 8)
    ):
        raise ValueError("real MoE admission case geometry drifted")
    factors = 1 + np.arange(rows, dtype=np.float32)[:, None] / 128
    values = (hidden.astype(np.float32) * factors).astype(hidden.dtype)
    values[4] = 0
    indices = np.repeat(routes.astype(np.int32), rows, axis=0)
    probabilities = np.repeat(weights.astype(np.float32), rows, axis=0)
    for row in range(1, rows):
        if case == "normal":
            indices[row] = (indices[row] + row * 17) % 256
        indices[row] = np.roll(indices[row], row % 8)
        probabilities[row] = np.roll(probabilities[row], row % 8)
    if case == "concentrated" and len(np.unique(indices // 32)) != 1:
        raise ValueError("oracle concentrated case is not one WS32 expert owner")
    return values, indices, probabilities


def check_hlo(
    hlo: str, *, fp32_route_sum: bool = False, rows: int = ROWS
) -> dict[str, Any]:
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

    module = parse_hlo_module(hlo)
    collectives = [op for op in module.instructions if op.is_collective]
    feature = tuple(tuple(range(e * 4, e * 4 + 4)) for e in range(8))
    expert = tuple(tuple(e * 4 + f for e in range(8)) for f in range(4))
    calls = [
        line.strip()
        for line in hlo.splitlines()
        if 'custom_call_target="tpu_custom_call"' in line
    ]
    grouped = [line for line in calls if "greenfield_prefill_grouped_raw_fp8" in line]
    shared = [line for line in calls if "greenfield_fp8_block_matmul" in line]
    overlays = [
        m[0]
        for m in re.finditer(r"\b(?:bf16|f32)\[([0-9,]+)\]", hlo)
        if np.prod([int(v) for v in m[1].split(",")]) >= 32 * 2048 * 1536
    ]
    groups = sorted(op.maximum_group_size for op in collectives)
    from scripts.greenfield.prefill_moe_precision_hlo import check_fp32_route_sum

    precision = check_fp32_route_sum(module, rows=rows) if fp32_route_sum else None
    return dict(
        passed=(
            (not fp32_route_sum or precision["passed"])
            and len(calls) == 6
            and len(grouped) == 3
            and len(shared) == 3
            and groups in ([4, 8], [4, 4, 8])
            and all(
                op.opcode == "all-reduce" and op.replica_groups in (feature, expert)
                for op in collectives
            )
            and sum("u8[32,2048,1536]" in line for line in grouped) == 2
            and sum("u8[32,1536,2048]" in line for line in grouped) == 1
            and not overlays
        ),
        group_sizes=groups,
        groups=[op.replica_groups for op in collectives],
        grouped_calls=len(grouped),
        shared_calls=len(shared),
        full_weight_overlays=overlays,
        fp32_route_sum_required=fp32_route_sum,
        fp32_route_sum_proof=precision,
        scope="arithmetic admission structure, not performance or acquired full-model HLO",
    )


def build_mapped(
    mesh: Any,
    *,
    contract: Any,
    interpret: bool = False,
    capture_boundaries: bool = False,
    fp32_route_sum: bool = False,
) -> tuple[Any, Any]:
    import jax
    from jax.sharding import PartitionSpec as P
    from glm_tpu.greenfield.benchmarking.ws32_one_layer import (
        WS32_ONE_LAYER_INPUT_SPECS,
    )
    from glm_tpu.greenfield.kernels.ws32 import ws32_moe_pallas_from_routes_mapped
    from glm_tpu.greenfield.kernels.ws32_prefill_moe import (
        ws32_prefill_moe_from_routes_mapped,
    )

    def candidate(*values):
        returned = ws32_prefill_moe_from_routes_mapped(
            *values,
            contract=contract,
            interpret=interpret,
            capture_boundaries=capture_boundaries,
            fp32_route_sum=fp32_route_sum,
        )
        result, health = returned[:2]
        if capture_boundaries:
            return (
                result,
                health[None, None],
                jax.tree.map(lambda x: x[None, None], returned[2]),
            )
        return result, health[None, None]

    def reference(*values):
        returned = ws32_moe_pallas_from_routes_mapped(
            *values,
            contract=contract,
            interpret=interpret,
            capture_boundaries=capture_boundaries,
        )
        if capture_boundaries:
            return returned[0], jax.tree.map(lambda x: x[None, None], returned[1])
        return returned

    common = dict(mesh=mesh, in_specs=WS32_ONE_LAYER_INPUT_SPECS, check_vma=False)
    return (
        jax.jit(
            jax.shard_map(
                candidate,
                out_specs=(
                    (P(None, "feature"), P("expert", "feature"), P("expert", "feature"))
                    if capture_boundaries
                    else (P(None, "feature"), P("expert", "feature"))
                ),
                **common,
            )
        ),
        jax.jit(
            jax.shard_map(
                reference,
                out_specs=(
                    (P(None, "feature"), P("expert", "feature"))
                    if capture_boundaries
                    else P(None, "feature")
                ),
                **common,
            )
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--coordinator-address", required=True)
    parser.add_argument("--process-id", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--boundary-diagnostic", action="store_true")
    mode.add_argument("--bounded-admission", action="store_true")
    mode.add_argument("--scaling-baseline", action="store_true")
    args = parser.parse_args()
    if (
        not 0 <= args.process_id < 8
        or REPO != Path("/home/gianl/glm-tpu-topology-rewrite")
        or _git_head() != args.expected_code_hash
    ):
        raise RuntimeError("worker code/rank/worktree identity drifted")
    tag = os.environ.get("GLM_GREENFIELD_RUN_TAG", "")
    if not re.fullmatch(
        r"greenfield_fp8_ws32_prefill_moe_"
        + (
            "scaling_baseline"
            if args.scaling_baseline
            else (
                "boundary_diagnostic"
                if args.boundary_diagnostic
                else "bounded_admission" if args.bounded_admission else "admission"
            )
        )
        + r"_[a-zA-Z0-9_]+",
        tag,
    ):
        raise ValueError("a scoped controller run tag is required")
    if args.output_dir != Path("/home/gianl/glm-run") / tag / f"rank{args.process_id}":
        raise ValueError("worker output path differs from exact tag/rank")
    output = args.output_dir / "runner.json"
    if output.exists():
        raise FileExistsError(output)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.num_processes, args.slice_name = 8, "db-v4-64-od"
    args.topology_capture_root = TOPOLOGY
    args.topology_sha256, args.topology_fleet_sha256, args.mesh_sha256 = (
        TOPOLOGY_SHA,
        FLEET_SHA,
        MESH_SHA,
    )
    record = dict(
        status="RUNNING",
        protocol=(
            SCALING_PROTOCOL
            if args.scaling_baseline
            else (
                BOUNDARY_PROTOCOL
                if args.boundary_diagnostic
                else BOUNDED_PROTOCOL if args.bounded_admission else PROTOCOL
            )
        ),
        code_hash=args.expected_code_hash,
        launch_rank=args.process_id,
        hostname=socket.gethostname(),
        pid=os.getpid(),
        start_ticks=int(
            Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
        ),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        admission_only=not (args.boundary_diagnostic or args.scaling_baseline),
        boundary_diagnostic=args.boundary_diagnostic,
        bounded_admission=args.bounded_admission,
        scaling_baseline=args.scaling_baseline,
        fp32_route_sum=args.bounded_admission or args.scaling_baseline,
        performance_claim=False,
        iterations=50 if args.scaling_baseline else 0,
        latency=None,
        rows=128 if args.scaling_baseline else ROWS,
        phases={},
        cases={},
    )
    _atomic_json(output, record)
    try:
        from scripts.greenfield.run_short_decoder_ws32 import _initialize_runtime
        from scripts.greenfield.run_real_one_layer_ws32 import (
            _load_oracle,
            _bfloat16_numpy,
            _case_result,
        )
        from glm_tpu.greenfield.checkpoint.ws32_one_layer import (
            load_ws32_one_layer_global,
        )
        from glm_tpu.greenfield.benchmarking.ws32_one_layer import ws32_one_layer_inputs
        from glm_tpu.greenfield.kernels.reference.moe import GlmMoeNumericalContract

        started = time.monotonic()
        jax, mesh, physical_mesh, topology, fleet_sha = _initialize_runtime(args)
        from jax.sharding import NamedSharding, PartitionSpec as P
        from jax.experimental import multihost_utils

        record.update(
            jax_process_index=jax.process_index(),
            mesh_sha256=physical_mesh.mesh_hash,
            physical_device_ids=physical_mesh.device_ids,
            topology_sha256=topology.topology_hash,
            topology_fleet_sha256=fleet_sha,
            versions={"jax": version("jax"), "libtpu": version("libtpu")},
        )
        record["phases"]["runtime_seconds"] = time.monotonic() - started
        _atomic_json(output, record)
        print(f"PREFILL_MOE rank={args.process_id} runtime_ready", flush=True)
        started = time.monotonic()
        loaded = load_ws32_one_layer_global(
            PACK,
            expected_manifest_sha256=PACK_SHA,
            mesh=mesh,
            physical_mesh=physical_mesh,
            payload_subdirectory="packed",
        )
        oracle_manifest, oracle = _load_oracle(
            ORACLE, expected_manifest_sha256=ORACLE_SHA, pack_manifest=loaded.manifest
        )
        record.update(
            packed_manifest_sha256=loaded.manifest["manifest_sha256"],
            oracle_manifest_sha256=oracle_manifest["manifest_sha256"],
            local_device_slots=loaded.local_device_slots,
        )
        record["phases"]["load_seconds"] = time.monotonic() - started
        _atomic_json(output, record)
        print(f"PREFILL_MOE rank={args.process_id} weights_verified", flush=True)
        if args.scaling_baseline:
            from scripts.greenfield.prefill_moe_scaling_worker import run_scaling

            run_scaling(
                args,
                record,
                jax=jax,
                mesh=mesh,
                physical_mesh=physical_mesh,
                loaded=loaded,
                oracle=oracle,
            )
            return 0
        batch, one = build_mapped(
            mesh,
            contract=GlmMoeNumericalContract(stage_size=8),
            capture_boundaries=args.boundary_diagnostic,
            fp32_route_sum=args.bounded_admission,
        )
        slot_by_device = {
            d: s for s, d in enumerate(physical_mesh.flattened_device_ids)
        }
        compiled = reference = None
        for case in ("normal",) if args.boundary_diagnostic else CASES:
            host_inputs = case_rows(
                _bfloat16_numpy(oracle["hidden_states"]),
                oracle[f"{case}_route_indices"].numpy(),
                oracle[f"{case}_route_weights"].numpy(),
                case,
            )
            values = tuple(
                jax.device_put(v, NamedSharding(mesh, s))
                for v, s in zip(host_inputs, (P(None, "feature"), P(), P()))
            )
            inputs = ws32_one_layer_inputs(*values, loaded.arrays)
            one_inputs = (*[v[:1] for v in inputs[:3]], *inputs[3:])
            if compiled is None:
                started = time.monotonic()
                lower = batch.lower(*inputs)
                (args.output_dir / "candidate.stablehlo.mlir").write_text(
                    str(lower.compiler_ir(dialect="stablehlo"))
                )
                compiled = lower.compile()
                hlo = compiled.as_text()
                (args.output_dir / "candidate.optimized_hlo.txt").write_text(hlo)
                record["phases"]["compile_seconds"] = time.monotonic() - started
                record["hlo"] = dict(
                    sha256=sha256(hlo.encode()).hexdigest(),
                    contract=check_hlo(hlo, fp32_route_sum=args.bounded_admission),
                )
                record["compiled_memory_estimate"] = memory = _compiled_memory(compiled)
                _atomic_json(output, record)
                if not record["hlo"]["contract"]["passed"]:
                    raise RuntimeError("real MoE pre-execution HLO contract failed")
                if (
                    sum(
                        memory[n]
                        for n in (
                            "argument_size_in_bytes",
                            "output_size_in_bytes",
                            "temp_size_in_bytes",
                        )
                    )
                    > 1024**3
                ):
                    raise RuntimeError(
                        "real MoE one-layer compiled allocation exceeds1GiB/chip"
                    )
                reference = one.lower(*one_inputs).compile()
                reference_hlo = reference.as_text()
                (args.output_dir / "reference.optimized_hlo.txt").write_text(
                    reference_hlo
                )
                record["reference_hlo_sha256"] = sha256(
                    reference_hlo.encode()
                ).hexdigest()
            returned = compiled(*inputs)
            jax.block_until_ready(returned)
            actual, health = returned[:2]
            references = []
            boundary_references = []
            for row in range(ROWS):
                result = reference(*[v[row : row + 1] for v in inputs[:3]], *inputs[3:])
                jax.block_until_ready(result)
                if args.boundary_diagnostic:
                    boundary_references.append(result[1])
                    result = result[0]
                references.append(result)
            legacy = _case_result(jax, references[0], oracle, case, slot_by_device)
            ref_by_device = [
                {
                    int(s.device.id): np.asarray(s.data)
                    for s in result.addressable_shards
                }
                for result in references
            ]
            health_by_device = {
                int(s.device.id): bool(np.asarray(s.data).all())
                for s in health.addressable_shards
            }
            shards = []
            tensors = {}
            for shard in actual.addressable_shards:
                device_id = int(shard.device.id)
                observed = np.asarray(shard.data)
                expected = np.concatenate(
                    [by_device[device_id] for by_device in ref_by_device]
                )
                mismatch = int(
                    np.count_nonzero(
                        observed.view(np.uint16) != expected.view(np.uint16)
                    )
                )
                bounded = None
                if args.bounded_admission:
                    from scripts.greenfield.prefill_moe_numerical import compare_outputs

                    feature = slot_by_device[device_id] % 4
                    legacy_row = _bfloat16_numpy(oracle[f"{case}_output"])
                    bounded = compare_outputs(
                        observed,
                        expected,
                        legacy_row[:, feature * 1536 : (feature + 1) * 1536],
                    )
                passed = (
                    health_by_device[device_id]
                    and (bounded["passed"] if args.bounded_admission else mismatch == 0)
                    and bool(np.isfinite(observed).all())
                )
                shards.append(
                    dict(
                        device_id=device_id,
                        device_slot=slot_by_device[device_id],
                        passed=passed,
                        finite_and_healthy=health_by_device[device_id]
                        and bool(np.isfinite(observed).all()),
                        bit_mismatches=mismatch,
                        output_sha256=sha256(observed.tobytes()).hexdigest(),
                        reference_sha256=sha256(expected.tobytes()).hexdigest(),
                        bounded_comparison=bounded,
                    )
                )
                tensors[f"actual_{device_id}"] = observed.view(np.uint16)
                tensors[f"reference_{device_id}"] = expected.view(np.uint16)
            np.savez_compressed(args.output_dir / f"{case}.npz", **tensors)
            if args.boundary_diagnostic:
                from scripts.greenfield.prefill_moe_boundaries import save_boundaries

                record["boundaries"] = save_boundaries(
                    returned[2], boundary_references, args.output_dir / "boundaries.npz"
                )
            local_passed = (
                len(shards) == 4
                and all(
                    s["finite_and_healthy"] if args.boundary_diagnostic else s["passed"]
                    for s in shards
                )
                and legacy["passed"]
            )
            # Scalar harness consensus OUTSIDE the candidate graph, not a layer collective.
            fleet_passed = bool(
                np.asarray(
                    multihost_utils.process_allgather(
                        np.asarray(local_passed, np.int32)
                    )
                ).all()
            )
            record["cases"][case] = dict(
                passed=local_passed,
                arithmetic_passed=all(s["passed"] for s in shards),
                fleet_passed=fleet_passed,
                shards=shards,
                reference_vs_legacy=legacy,
                input_sha256=[sha256(v.tobytes()).hexdigest() for v in host_inputs],
            )
            _atomic_json(output, record)
            print(
                f"PREFILL_MOE rank={args.process_id} case={case} passed={fleet_passed}",
                flush=True,
            )
            if not fleet_passed:
                raise RuntimeError(f"first real MoE arithmetic failure: {case}")
        record["device_memory_stats_including_reference"] = [
            dict(device_id=int(d.id), stats=_memory_stats(d))
            for d in jax.local_devices()
        ]
        record["status"] = "SUCCESS"
        _atomic_json(output, record)
        return 0
    except Exception as exc:
        record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        _atomic_json(output, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
