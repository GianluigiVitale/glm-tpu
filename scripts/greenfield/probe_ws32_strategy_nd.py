#!/usr/bin/env python3
"""Model-free all-host proof of the local WS32 StrategyND dense reducer."""

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
from typing import Any

import ml_dtypes
import numpy as np


REPO = Path(__file__).resolve().parents[2]
EXPECTED_REPO = Path("/home/gianl/glm-tpu-topology-rewrite")
EXPECTED_INPUT_ARRAY_SHA256 = (
    "9d9f65dddc7b622875872a33a6522c330c8fb5490c8cba14526553c211516e35"
)
EXPECTED_OUTPUT_SHA256 = (
    "efde853254c03dd18a5f5f22733630ce0e785dfbb4eba09c41eea9085e47b4fc"
)
ZERO_SHA256 = "0" * 64

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.benchmarking.ws32_one_layer import (  # noqa: E402
    validate_ws32_topology_fleet,
)
from glm_tpu.greenfield.kernels.stage_local import (  # noqa: E402
    STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE,
)
from glm_tpu.greenfield.sharding.hlo_contract import (  # noqa: E402
    parse_hlo_module,
)
from glm_tpu.greenfield.sharding.ws32 import (  # noqa: E402
    build_ws32_physical_mesh,
)


def _sha256_bytes(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _host_add(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.asarray(left + right, dtype=ml_dtypes.bfloat16)


def _host_reduce_four(values: np.ndarray, *, cross: bool) -> np.ndarray:
    pairs = ((0, 3), (1, 2)) if cross else ((0, 1), (2, 3))
    return _host_add(
        _host_add(values[pairs[0][0]], values[pairs[0][1]]),
        _host_add(values[pairs[1][0]], values[pairs[1][1]]),
    )


def _host_strategy_nd_reduce(model_partials: np.ndarray) -> np.ndarray:
    if model_partials.shape != (32, 1, 6144) or (
        model_partials.dtype != ml_dtypes.bfloat16
    ):
        raise ValueError("sealed dense partial geometry drifted")
    physical = np.stack(
        tuple(
            model_partials[position]
            for position in STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE
        ),
        axis=0,
    )
    physical_y_x_z = np.transpose(
        physical.reshape(4, 4, 2, 1, 6144),
        (1, 2, 0, 3, 4),
    )
    y_reduced = np.concatenate(
        (
            _host_reduce_four(physical_y_x_z[..., :2048], cross=False),
            _host_reduce_four(
                physical_y_x_z[..., 2048:4096], cross=True
            ),
            _host_reduce_four(physical_y_x_z[..., 4096:], cross=False),
        ),
        axis=-1,
    )
    x_reduced = _host_add(y_reduced[0], y_reduced[1])
    return np.concatenate(
        tuple(
            _host_reduce_four(
                x_reduced[..., start : start + 256],
                cross=bool((start // 256) % 2),
            )
            for start in range(0, 6144, 256)
        ),
        axis=-1,
    )


def _load_sealed_input(
    path: Path, expected_file_sha256: str
) -> tuple[np.ndarray, np.ndarray]:
    if _sha256_file(path) != expected_file_sha256:
        raise RuntimeError("DB550 NPZ file identity drifted")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {
            "accepted_dense_partials_bfloat16_bits",
            "db548_dense_partials_bfloat16_bits",
        }:
            raise RuntimeError("DB550 NPZ fields drifted")
        accepted_bits = np.ascontiguousarray(
            payload["accepted_dense_partials_bfloat16_bits"]
        )
    if accepted_bits.shape != (4, 8, 1, 6144) or (
        accepted_bits.dtype != np.uint16
    ):
        raise RuntimeError("DB550 accepted partial geometry drifted")
    if _sha256_bytes(accepted_bits) != EXPECTED_INPUT_ARRAY_SHA256:
        raise RuntimeError("DB550 accepted partial identity drifted")
    model_partials = accepted_bits.reshape(32, 1, 6144).view(
        ml_dtypes.bfloat16
    )
    expected = np.ascontiguousarray(_host_strategy_nd_reduce(model_partials))
    if _sha256_bytes(expected.view(np.uint16)) != EXPECTED_OUTPUT_SHA256:
        raise RuntimeError("host StrategyND oracle identity drifted")
    return model_partials, expected


def _lowering_contract(optimized_hlo: str) -> dict[str, Any]:
    module = parse_hlo_module(optimized_hlo)
    collectives = tuple(module.collectives)
    records = [item.to_dict() for item in collectives]
    passed = (
        len(collectives) == 1
        and collectives[0].opcode == "all-gather"
        and collectives[0].maximum_group_size == 8
        and tuple(
            shape.dimensions for shape in collectives[0].operand_shapes
        )
        == ((1, 4, 1, 1536),)
        and all(
            len(group) == 8 for group in collectives[0].replica_groups
        )
    )
    return {
        "collective_count": len(collectives),
        "collectives": records,
        "maximum_group_size": max(
            (item.maximum_group_size for item in collectives), default=0
        ),
        "passed": passed,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator-address", required=True)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--slice-name", default="db-v4-64-od")
    parser.add_argument("--topology-capture-root", required=True, type=Path)
    parser.add_argument("--topology-sha256", required=True)
    parser.add_argument("--topology-fleet-sha256", required=True)
    parser.add_argument("--mesh-sha256", required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-file-sha256", required=True)
    parser.add_argument("--mode", choices=("acquire", "numerical"), required=True)
    parser.add_argument(
        "--expected-stablehlo-sha256", default=ZERO_SHA256
    )
    parser.add_argument(
        "--expected-optimized-hlo-sha256", default=ZERO_SHA256
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hlo-dir", required=True, type=Path)
    return parser.parse_args()


def _fleet_digest(multihost_utils: Any, digest_hex: str) -> list[str]:
    digest = np.frombuffer(bytes.fromhex(digest_hex), dtype=np.uint8)
    gathered = np.asarray(multihost_utils.process_allgather(digest)).reshape(8, 32)
    result = [row.tobytes().hex() for row in gathered]
    if len(set(result)) != 1:
        raise RuntimeError(f"hosts disagree on graph identity: {result}")
    return result


def _materialize_global_bits(
    jax: Any,
    multihost_utils: Any,
    result: Any,
) -> np.ndarray:
    local_shards = tuple(result.addressable_shards)
    if len(local_shards) != 4:
        raise RuntimeError("WS32 result must have four local replica shards")
    local_values = tuple(
        np.ascontiguousarray(np.asarray(item.data)).view(np.uint16)
        for item in local_shards
    )
    if any(
        not np.array_equal(local_values[0], value)
        for value in local_values[1:]
    ):
        raise RuntimeError("local expert replicas disagree")
    indices = tuple(item.index for item in local_shards)
    starts = tuple(index[1].start for index in indices)
    if len(set(starts)) != 1 or any(
        index[0] != slice(None) or index[1].stop - index[1].start != 1536
        for index in indices
    ):
        raise RuntimeError("local feature shard placement drifted")
    fleet_values = np.asarray(
        multihost_utils.process_allgather(local_values[0])
    ).reshape(8, 1, 1536)
    fleet_starts = np.asarray(
        multihost_utils.process_allgather(np.asarray(starts[:1], dtype=np.int32))
    ).reshape(8)
    complete = np.empty((1, 6144), dtype=np.uint16)
    for start in range(0, 6144, 1536):
        matches = np.flatnonzero(fleet_starts == start)
        if matches.size != 2 or not np.array_equal(
            fleet_values[matches[0]], fleet_values[matches[1]]
        ):
            raise RuntimeError("fleet feature replicas disagree")
        complete[:, start : start + 1536] = fleet_values[matches[0]]
    jax.block_until_ready(result)
    return complete


def main() -> int:
    args = _parse_args()
    if REPO != EXPECTED_REPO:
        raise RuntimeError(f"wrong worktree: {REPO}")
    if args.num_processes != 8 or not 0 <= args.process_id < 8:
        raise ValueError("protected WS32 proof requires eight launch processes")
    code_hash = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if code_hash != args.expected_code_hash:
        raise RuntimeError("worker code hash drifted")
    if args.output.exists() or args.hlo_dir.exists():
        raise FileExistsError("WS32 StrategyND worker evidence is append-only")
    model_partials, expected = _load_sealed_input(
        args.input, args.input_file_sha256
    )
    owner_partials = model_partials.reshape(8, 4, 1, 6144)

    import jax
    from jax.experimental import multihost_utils
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from glm_tpu.greenfield.kernels.ws32 import (
        ws32_strategy_nd_dense_down_reduce_mapped,
    )

    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=args.num_processes,
        process_id=args.process_id,
    )
    try:
        if (
            jax.default_backend() != "tpu"
            or jax.device_count() != 32
            or jax.local_device_count() != 4
            or jax.process_count() != 8
        ):
            raise RuntimeError("runtime is not the exact 8-host/32-chip TPU pod")
        captures = tuple(
            json.loads(
                (
                    args.topology_capture_root / f"topology.rank{rank}.json"
                ).read_text(encoding="utf-8")
            )
            for rank in range(8)
        )
        topology, ordered, fleet_sha = validate_ws32_topology_fleet(
            captures,
            expected_topology_sha256=args.topology_sha256,
            expected_fleet_sha256=args.topology_fleet_sha256,
            slice_name=args.slice_name,
        )
        launch_capture = ordered[args.process_id]
        if (
            launch_capture["hostname"] != socket.gethostname()
            or launch_capture["jax_process_index"] != jax.process_index()
            or launch_capture["local_device_ids"]
            != [int(device.id) for device in jax.local_devices()]
        ):
            raise RuntimeError("launch/JAX/topology mapping drifted")
        physical_mesh = build_ws32_physical_mesh(topology)
        if physical_mesh.mesh_hash != args.mesh_sha256:
            raise RuntimeError("physical WS32 mesh identity drifted")
        by_id = {int(device.id): device for device in jax.devices()}
        mesh = Mesh(
            np.asarray(
                [by_id[item] for item in physical_mesh.flattened_device_ids],
                dtype=object,
            ).reshape(8, 4),
            ("expert", "feature"),
        )
        sharding = NamedSharding(
            mesh, P("expert", None, None, "feature")
        )
        sharded = jax.device_put(owner_partials, sharding)
        mapped = jax.shard_map(
            lambda values: ws32_strategy_nd_dense_down_reduce_mapped(values[0]),
            mesh=mesh,
            in_specs=P("expert", None, None, "feature"),
            out_specs=P(None, "feature"),
            check_vma=False,
        )
        lowered = jax.jit(mapped).lower(sharded)
        compiled = lowered.compile()
        stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
        optimized_hlo = compiled.as_text()
        stable_sha = sha256(stablehlo.encode()).hexdigest()
        optimized_sha = sha256(optimized_hlo.encode()).hexdigest()
        fleet_stable = _fleet_digest(multihost_utils, stable_sha)
        fleet_optimized = _fleet_digest(multihost_utils, optimized_sha)
        contract = _lowering_contract(optimized_hlo)
        args.hlo_dir.mkdir(parents=True)
        (args.hlo_dir / "strategy_nd.stablehlo.mlir").write_text(
            stablehlo, encoding="utf-8"
        )
        (args.hlo_dir / "strategy_nd.optimized_hlo.txt").write_text(
            optimized_hlo, encoding="utf-8"
        )
        if not contract["passed"]:
            raise RuntimeError(f"WS32 StrategyND HLO contract failed: {contract}")

        actual_bits = None
        mismatch_count = None
        if args.mode == "acquire":
            if (
                args.expected_stablehlo_sha256 != ZERO_SHA256
                or args.expected_optimized_hlo_sha256 != ZERO_SHA256
            ):
                raise RuntimeError("acquisition must use vacant graph pins")
            status = "HLO_ACQUIRED"
            passed = False
        else:
            if (
                stable_sha != args.expected_stablehlo_sha256
                or optimized_sha != args.expected_optimized_hlo_sha256
            ):
                raise RuntimeError("numerical graph identity differs from acquisition")
            actual_bits = _materialize_global_bits(
                jax, multihost_utils, compiled(sharded)
            )
            expected_bits = expected.view(np.uint16)
            mismatch_count = int(np.count_nonzero(actual_bits != expected_bits))
            passed = mismatch_count == 0 and (
                _sha256_bytes(actual_bits) == EXPECTED_OUTPUT_SHA256
            )
            status = "PASS" if passed else "NUMERICAL_MISMATCH"

        record = {
            "artifact_kind": "greenfield_ws32_strategy_nd_model_free_probe",
            "captured_utc": datetime.now(timezone.utc).isoformat(),
            "code_hash": code_hash,
            "execution_performed": args.mode == "numerical",
            "expected_output_bfloat16_sha256": EXPECTED_OUTPUT_SHA256,
            "fleet_optimized_hlo_sha256": fleet_optimized,
            "fleet_stablehlo_sha256": fleet_stable,
            "hostname": socket.gethostname(),
            "hlo_contract": contract,
            "input_array_sha256": EXPECTED_INPUT_ARRAY_SHA256,
            "input_file_sha256": args.input_file_sha256,
            "jax_process_index": int(jax.process_index()),
            "launch_process_id": args.process_id,
            "local_device_ids": [int(item.id) for item in jax.local_devices()],
            "mesh_sha256": physical_mesh.mesh_hash,
            "mismatch_count": mismatch_count,
            "mode": args.mode,
            "observed_output_bfloat16_sha256": (
                None if actual_bits is None else _sha256_bytes(actual_bits)
            ),
            "optimized_hlo_sha256": optimized_sha,
            "passed": passed,
            "schema_version": 1,
            "stablehlo_sha256": stable_sha,
            "status": status,
            "topology_fleet_sha256": fleet_sha,
            "topology_sha256": topology.topology_hash,
        }
        _atomic_json(args.output, record)
        multihost_utils.sync_global_devices(
            f"ws32-strategy-nd-{args.mode}-complete"
        )
        print(
            "WS32_STRATEGY_ND_OK "
            f"mode={args.mode} launch={args.process_id} "
            f"jax={jax.process_index()} status={status}",
            flush=True,
        )
        return 0 if args.mode == "acquire" or passed else 1
    finally:
        jax.distributed.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
