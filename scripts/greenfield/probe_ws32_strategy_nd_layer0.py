#!/usr/bin/env python3
"""Bounded all-host proof of WS32 layer-0 dense partial generation."""

from __future__ import annotations

import argparse
import base64
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any, Mapping

import ml_dtypes
import numpy as np


REPO = Path(__file__).resolve().parents[2]
EXPECTED_REPO = Path("/home/gianl/glm-tpu-topology-rewrite")
ZERO_SHA256 = "0" * 64
CHECKPOINT_PACK_CODE_HASH = "fbaaae3fd328ad94d2a04a24f37f49c91f34189c"
CHECKPOINT_MANIFEST_SHA256 = (
    "ec6ef9cf6b82254fc0412a1285f2efc31c47847253928b2ff0c7199ded96b3d3"
)
CHECKPOINT_MANIFEST_FILE_SHA256 = (
    "0f475ded23edfcc8933a11543bef85c94e0a6ab25dad761905bad47ad4424512"
)
CHECKPOINT_SUCCESS_FILE_SHA256 = (
    "c13045921444739fb4c3218c2f5381280d5f45757b8b3624688c2c210e36840d"
)
SOURCE_FILE_SHA256 = (
    "f194d757d2f9ebe27430dfec8f828ca7588e433bddb7e8d99f9b917c5aac4298"
)
NORMALIZED_BITS_SHA256 = (
    "082125fead43b25f10686705c1b6473153f4092dd5bc476f8e01a86629f0758f"
)
PARTIAL_BITS_SHA256 = (
    "9d9f65dddc7b622875872a33a6522c330c8fb5490c8cba14526553c211516e35"
)
FINAL_BITS_SHA256 = (
    "efde853254c03dd18a5f5f22733630ce0e785dfbb4eba09c41eea9085e47b4fc"
)
TENSOR_NAMES = (
    "merged_gate_up.weight_bits_in_out",
    "merged_gate_up.scale_inv_in_out",
    "down.weight_bits_in_out",
    "down.scale_inv_in_out",
)
GLOBAL_LAYOUTS = {
    TENSOR_NAMES[0]: ((32, 6144, 768), ("expert", None, None)),
    TENSOR_NAMES[1]: ((32, 48, 768), ("expert", None, None)),
    TENSOR_NAMES[2]: ((32, 384, 6144), ("expert", None, "feature")),
    TENSOR_NAMES[3]: ((32, 3, 6144), ("expert", None, "feature")),
}

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.benchmarking.ws32_one_layer import (  # noqa: E402
    validate_ws32_topology_fleet,
)
from glm_tpu.greenfield.sharding.hlo_contract import (  # noqa: E402
    parse_hlo_module,
)
from glm_tpu.greenfield.sharding.ws32 import (  # noqa: E402
    build_ws32_physical_mesh,
)
from scripts.greenfield.probe_ws32_strategy_nd import (  # noqa: E402
    _host_strategy_nd_reduce,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **values)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _memory_stats(device: Any) -> dict[str, int] | None:
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
    names = (
        "alias_size_in_bytes",
        "argument_size_in_bytes",
        "generated_code_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
    )
    return {
        name: None
        if getattr(analysis, name, None) is None
        else int(getattr(analysis, name))
        for name in names
    }


def _manifest_records(manifest: Mapping[str, Any]) -> dict[tuple[int, int], Any]:
    records = {
        (int(item["expert_coordinate"]), int(item["feature_coordinate"])): item
        for item in manifest.get("files", ())
    }
    if set(records) != {(expert, feature) for expert in range(8) for feature in range(4)}:
        raise RuntimeError("bounded checkpoint owner coverage drifted")
    for expert in range(8):
        expected_ranks = list(range(expert * 4, expert * 4 + 4))
        group = [records[(expert, feature)] for feature in range(4)]
        if any(item["model_ranks"] != expected_ranks for item in group):
            raise RuntimeError("bounded checkpoint model-rank ownership drifted")
        for name in TENSOR_NAMES[:2]:
            if len({item["tensors"][name]["sha256"] for item in group}) != 1:
                raise RuntimeError("bounded checkpoint feature replicas drifted")
    return records


def _inspect_checkpoint(root: Path) -> tuple[dict[str, Any], dict[tuple[int, int], Any]]:
    manifest_path = root / "manifest.json"
    success_path = root / "SUCCESS"
    if _sha256_file(manifest_path) != CHECKPOINT_MANIFEST_FILE_SHA256 or (
        _sha256_file(success_path) != CHECKPOINT_SUCCESS_FILE_SHA256
    ):
        raise RuntimeError("bounded checkpoint terminal file identity drifted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    without_hash = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != sha256(_canonical(without_hash)).hexdigest() or (
        manifest.get("manifest_sha256") != CHECKPOINT_MANIFEST_SHA256
    ):
        raise RuntimeError("bounded checkpoint manifest identity drifted")
    success = json.loads(success_path.read_text(encoding="utf-8"))
    without_success = {key: value for key, value in success.items() if key != "success_sha256"}
    if success.get("success_sha256") != sha256(_canonical(without_success)).hexdigest() or (
        success.get("manifest_sha256") != CHECKPOINT_MANIFEST_SHA256
    ):
        raise RuntimeError("bounded checkpoint SUCCESS identity drifted")
    if (
        manifest.get("artifact_kind") != "greenfield_ws32_strategy_nd_layer0_checkpoint"
        or manifest.get("code_hash") != CHECKPOINT_PACK_CODE_HASH
        or manifest.get("file_count") != 32
        or manifest.get("total_bytes") != 700_728_992
    ):
        raise RuntimeError("bounded checkpoint contract drifted")
    return manifest, _manifest_records(manifest)


def _load_source(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _sha256_file(path) != SOURCE_FILE_SHA256:
        raise RuntimeError("DB548 source file identity drifted")
    with np.load(path, allow_pickle=False) as payload:
        normalized = np.ascontiguousarray(payload["normalized_mlp_bfloat16_bits"])
        partials = np.ascontiguousarray(payload["dense_virtual_partials_bfloat16_bits"])
    if normalized.shape != (1, 6144) or normalized.dtype != np.uint16 or (
        _sha256_array(normalized) != NORMALIZED_BITS_SHA256
    ):
        raise RuntimeError("DB548 normalized input identity drifted")
    if partials.shape != (4, 8, 1, 6144) or partials.dtype != np.uint16 or (
        _sha256_array(partials) != PARTIAL_BITS_SHA256
    ):
        raise RuntimeError("DB548 partial identity drifted")
    model_partials = partials.reshape(32, 1, 6144)
    expected_final = np.ascontiguousarray(
        _host_strategy_nd_reduce(model_partials.view(ml_dtypes.bfloat16))
    ).view(np.uint16)
    if _sha256_array(expected_final) != FINAL_BITS_SHA256:
        raise RuntimeError("DB548 derived final identity drifted")
    return normalized, model_partials, expected_final


def _load_local_arrays(
    jax: Any,
    mesh: Any,
    physical_mesh: Any,
    checkpoint_root: Path,
    records: Mapping[tuple[int, int], Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from jax.sharding import NamedSharding, PartitionSpec as P, SingleDeviceSharding
    from safetensors import safe_open

    coordinates = {
        int(device_id): (expert, feature)
        for expert, row in enumerate(physical_mesh.device_ids)
        for feature, device_id in enumerate(row)
    }
    local_values: dict[int, dict[str, Any]] = {}
    local_records = []
    for device in jax.local_devices():
        expert, feature = coordinates[int(device.id)]
        record = records[(expert, feature)]
        path = checkpoint_root / record["filename"]
        if _sha256_file(path) != record["sha256"]:
            raise RuntimeError("bounded checkpoint local file hash drifted")
        with safe_open(path, framework="np") as handle:
            arrays = {name: np.ascontiguousarray(handle.get_tensor(name)) for name in TENSOR_NAMES}
        for name, value in arrays.items():
            tensor_record = record["tensors"][name]
            if list(value.shape) != tensor_record["shape"] or _sha256_array(value) != tensor_record["sha256"]:
                raise RuntimeError(f"bounded checkpoint local tensor drifted: {name}")
        local_values[int(device.id)] = {
            name: jax.device_put(value, SingleDeviceSharding(device))
            for name, value in arrays.items()
        }
        local_records.append(
            {
                "device_id": int(device.id),
                "expert_coordinate": expert,
                "feature_coordinate": feature,
                "file_sha256": record["sha256"],
            }
        )
    result = {}
    for name, (shape, spec) in GLOBAL_LAYOUTS.items():
        sharding = NamedSharding(mesh, P(*spec))
        devices = tuple(sharding.addressable_devices_indices_map(shape))
        shards = tuple(local_values[int(device.id)][name] for device in devices)
        if any(tuple(shard.shape) != sharding.shard_shape(shape) for shard in shards):
            raise RuntimeError(f"bounded checkpoint JAX shard geometry drifted: {name}")
        result[name] = jax.make_array_from_single_device_arrays(shape, sharding, shards)
    jax.block_until_ready(tuple(result.values()))
    return result, sorted(local_records, key=lambda item: item["device_id"])


def _shard_hidden(jax: Any, mesh: Any, bits: np.ndarray) -> Any:
    from jax.sharding import NamedSharding, PartitionSpec as P, SingleDeviceSharding

    sharding = NamedSharding(mesh, P(None, "feature"))
    shards = []
    for device, index in sharding.addressable_devices_indices_map(bits.shape).items():
        local = bits[index].view(ml_dtypes.bfloat16)
        shards.append(jax.device_put(local, SingleDeviceSharding(device)))
    return jax.make_array_from_single_device_arrays(bits.shape, sharding, tuple(shards))


def _lowering_contract(optimized_hlo: str) -> dict[str, Any]:
    collectives = tuple(parse_hlo_module(optimized_hlo).collectives)
    signatures = tuple(
        (
            item.opcode,
            item.maximum_group_size,
            tuple((shape.dtype, shape.dimensions) for shape in item.operand_shapes),
            tuple((shape.dtype, shape.dimensions) for shape in item.result_shapes),
        )
        for item in collectives
    )
    expected = (
        ("all-gather", 4, (("bf16", (1, 1536)),), (("bf16", (1, 6144)),)),
        ("all-gather", 8, (("bf16", (4, 1, 1536)),), (("bf16", (32, 1, 1536)),)),
    )
    expected_groups = (
        tuple(tuple(range(start, start + 4)) for start in range(0, 32, 4)),
        tuple(tuple(range(feature, 32, 4)) for feature in range(4)),
    )
    expected_scopes = (
        "greenfield_ws32_strategy_nd_dense/hidden_gather/all_gather",
        "greenfield_ws32_strategy_nd_dense/expert_gather/all_gather",
    )
    passed = (
        signatures == expected
        and tuple(item.replica_groups for item in collectives) == expected_groups
        and all(item.use_global_device_ids for item in collectives)
        and all(
            item.op_name is not None and item.op_name.endswith(scope)
            for item, scope in zip(collectives, expected_scopes, strict=True)
        )
    )
    return {
        "collective_count": len(collectives),
        "collectives": [item.to_dict() for item in collectives],
        "maximum_group_size": max((item.maximum_group_size for item in collectives), default=0),
        "passed": passed,
        "signatures": [
            [
                opcode,
                group,
                [[dtype, list(shape)] for dtype, shape in operands],
                [[dtype, list(shape)] for dtype, shape in results],
            ]
            for opcode, group, operands, results in signatures
        ],
    }


def _fleet_digest(multihost_utils: Any, digest_hex: str) -> None:
    digest = np.frombuffer(bytes.fromhex(digest_hex), dtype=np.uint8)
    gathered = np.asarray(multihost_utils.process_allgather(digest)).reshape(8, 32)
    if len({row.tobytes().hex() for row in gathered}) != 1:
        raise RuntimeError("fleet graph identity drifted")


def _slice_start(value: slice) -> int:
    return 0 if value.start is None else int(value.start)


def _materialize_partials(multihost_utils: Any, result: Any) -> np.ndarray:
    local_shards = sorted(result.addressable_shards, key=lambda item: int(item.device.id))
    values = np.stack(tuple(np.asarray(item.data).view(np.uint16) for item in local_shards))
    starts = np.asarray(
        [(_slice_start(item.index[0]), _slice_start(item.index[2])) for item in local_shards],
        dtype=np.int32,
    )
    fleet_values = np.asarray(multihost_utils.process_allgather(values)).reshape(32, 4, 1, 1536)
    fleet_starts = np.asarray(multihost_utils.process_allgather(starts)).reshape(32, 2)
    complete = np.empty((32, 1, 6144), dtype=np.uint16)
    coverage = np.zeros((8, 4), dtype=np.int32)
    for value, (expert_start, feature_start) in zip(fleet_values, fleet_starts, strict=True):
        expert = expert_start // 4
        feature = feature_start // 1536
        complete[expert_start : expert_start + 4, :, feature_start : feature_start + 1536] = value
        coverage[expert, feature] += 1
    if not np.array_equal(coverage, np.ones((8, 4), dtype=np.int32)):
        raise RuntimeError("partial evidence coverage drifted")
    return complete


def _materialize_final(multihost_utils: Any, result: Any) -> np.ndarray:
    local_shards = sorted(result.addressable_shards, key=lambda item: int(item.device.id))
    values = np.stack(tuple(np.asarray(item.data).view(np.uint16) for item in local_shards))
    starts = np.asarray([_slice_start(item.index[1]) for item in local_shards], dtype=np.int32)
    fleet_values = np.asarray(multihost_utils.process_allgather(values)).reshape(32, 1, 1536)
    fleet_starts = np.asarray(multihost_utils.process_allgather(starts)).reshape(32)
    complete = np.empty((1, 6144), dtype=np.uint16)
    for start in range(0, 6144, 1536):
        matches = np.flatnonzero(fleet_starts == start)
        if matches.size != 8 or any(
            not np.array_equal(fleet_values[matches[0]], fleet_values[index]) for index in matches[1:]
        ):
            raise RuntimeError("final feature replicas drifted")
        complete[:, start : start + 1536] = fleet_values[matches[0]]
    return complete


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
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--mode", choices=("acquire", "numerical"), required=True)
    parser.add_argument("--expected-stablehlo-sha256", default=ZERO_SHA256)
    parser.add_argument("--expected-optimized-hlo-sha256", default=ZERO_SHA256)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tensor-output", required=True, type=Path)
    parser.add_argument("--hlo-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if REPO != EXPECTED_REPO:
        raise RuntimeError(f"wrong worktree: {REPO}")
    if args.num_processes != 8 or not 0 <= args.process_id < 8:
        raise ValueError("protected layer proof requires eight launch processes")
    code_hash = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if code_hash != args.expected_code_hash:
        raise RuntimeError("worker code hash drifted")
    if args.output.exists() or args.tensor_output.exists() or args.hlo_dir.exists():
        raise FileExistsError("bounded layer worker evidence is append-only")
    _, checkpoint_records = _inspect_checkpoint(args.checkpoint_root)
    normalized_bits, expected_partial_bits, expected_final_bits = _load_source(args.source)

    import jax
    from jax.experimental import multihost_utils
    from jax.sharding import Mesh, PartitionSpec as P

    from glm_tpu.greenfield.kernels.ws32 import ws32_strategy_nd_dense_final_layout_observed_mapped

    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=args.num_processes,
        process_id=args.process_id,
    )
    try:
        if jax.default_backend() != "tpu" or jax.device_count() != 32 or jax.local_device_count() != 4:
            raise RuntimeError("runtime is not the exact 8-host/32-chip TPU pod")
        captures = tuple(
            json.loads((args.topology_capture_root / f"topology.rank{rank}.json").read_text())
            for rank in range(8)
        )
        topology, ordered, fleet_sha = validate_ws32_topology_fleet(
            captures,
            expected_topology_sha256=args.topology_sha256,
            expected_fleet_sha256=args.topology_fleet_sha256,
            slice_name=args.slice_name,
        )
        launch = ordered[args.process_id]
        if launch["hostname"] != socket.gethostname() or launch["jax_process_index"] != jax.process_index():
            raise RuntimeError("launch/JAX/topology mapping drifted")
        physical_mesh = build_ws32_physical_mesh(topology)
        if physical_mesh.mesh_hash != args.mesh_sha256 or fleet_sha != args.topology_fleet_sha256:
            raise RuntimeError("physical WS32 identity drifted")
        by_id = {int(device.id): device for device in jax.devices()}
        mesh = Mesh(
            np.asarray([by_id[item] for item in physical_mesh.flattened_device_ids], dtype=object).reshape(8, 4),
            ("expert", "feature"),
        )
        before_memory = [_memory_stats(device) for device in jax.local_devices()]
        arrays, local_records = _load_local_arrays(
            jax, mesh, physical_mesh, args.checkpoint_root, checkpoint_records
        )
        hidden = _shard_hidden(jax, mesh, normalized_bits)
        mapped = jax.shard_map(
            ws32_strategy_nd_dense_final_layout_observed_mapped,
            mesh=mesh,
            in_specs=(
                P(None, "feature"),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None, "feature"),
                P("expert", None, "feature"),
            ),
            out_specs=(P(None, "feature"), P("expert", None, "feature")),
            check_vma=False,
        )
        inputs = (hidden, *(arrays[name] for name in TENSOR_NAMES))
        lowered = jax.jit(mapped).lower(*inputs)
        compiled = lowered.compile()
        stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
        optimized_hlo = compiled.as_text()
        stable_sha = sha256(stablehlo.encode()).hexdigest()
        optimized_sha = sha256(optimized_hlo.encode()).hexdigest()
        _fleet_digest(multihost_utils, stable_sha)
        _fleet_digest(multihost_utils, optimized_sha)
        contract = _lowering_contract(optimized_hlo)
        args.hlo_dir.mkdir(parents=True)
        (args.hlo_dir / "layer0.stablehlo.mlir").write_text(stablehlo)
        (args.hlo_dir / "layer0.optimized_hlo.txt").write_text(optimized_hlo)
        if not contract["passed"]:
            _atomic_json(
                args.output,
                {
                    "artifact_kind": "greenfield_ws32_strategy_nd_layer0_hlo_refusal",
                    "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
                    "code_hash": code_hash,
                    "compiled_memory": _compiled_memory(compiled),
                    "device_memory_after": [
                        _memory_stats(device) for device in jax.local_devices()
                    ],
                    "device_memory_before": before_memory,
                    "execution_performed": False,
                    "hlo_contract": contract,
                    "host": socket.gethostname(),
                    "jax_process_index": int(jax.process_index()),
                    "launch_process_id": args.process_id,
                    "local_checkpoint_records": local_records,
                    "mesh_sha256": physical_mesh.mesh_hash,
                    "mode": args.mode,
                    "optimized_hlo_sha256": optimized_sha,
                    "performance_claim": False,
                    "source_file_sha256": SOURCE_FILE_SHA256,
                    "stablehlo_sha256": stable_sha,
                    "status": "HLO_REFUSED",
                    "tensor_file_sha256": None,
                    "topology_fleet_sha256": fleet_sha,
                    "topology_sha256": topology.topology_hash,
                },
            )
            raise RuntimeError("bounded layer HLO contract refused")
        acquisition = args.mode == "acquire"
        if acquisition:
            if args.expected_stablehlo_sha256 != ZERO_SHA256 or args.expected_optimized_hlo_sha256 != ZERO_SHA256:
                raise RuntimeError("acquisition graph pins must be vacant")
            partial_bits = np.empty((0,), dtype=np.uint16)
            final_bits = np.empty((0,), dtype=np.uint16)
            mismatch_count = None
            final_mismatch_count = None
            status = "HLO_ACQUIRED"
        else:
            if stable_sha != args.expected_stablehlo_sha256 or optimized_sha != args.expected_optimized_hlo_sha256:
                raise RuntimeError("numerical graph pin drifted")
            final, partials = compiled(*inputs)
            jax.block_until_ready((final, partials))
            partial_bits = _materialize_partials(multihost_utils, partials)
            final_bits = _materialize_final(multihost_utils, final)
            mismatch_count = int(np.count_nonzero(partial_bits != expected_partial_bits))
            final_mismatch_count = int(np.count_nonzero(final_bits != expected_final_bits))
            status = "PASS" if mismatch_count == 0 and final_mismatch_count == 0 else "ORACLE_MISMATCH"
            _atomic_npz(
                args.tensor_output,
                observed_final_bfloat16_bits=final_bits,
                observed_partial_bfloat16_bits=partial_bits,
            )
        record = {
            "artifact_kind": "greenfield_ws32_strategy_nd_layer0_proof",
            "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
            "code_hash": code_hash,
            "compiled_memory": _compiled_memory(compiled),
            "device_memory_after": [_memory_stats(device) for device in jax.local_devices()],
            "device_memory_before": before_memory,
            "execution_performed": not acquisition,
            "expected_final_bits_sha256": FINAL_BITS_SHA256,
            "expected_partial_bits_sha256": PARTIAL_BITS_SHA256,
            "final_bits_sha256": None if acquisition else _sha256_array(final_bits),
            "final_mismatch_count": final_mismatch_count,
            "hlo_contract": contract,
            "host": socket.gethostname(),
            "jax_process_index": int(jax.process_index()),
            "launch_process_id": args.process_id,
            "local_checkpoint_records": local_records,
            "mesh_sha256": physical_mesh.mesh_hash,
            "mismatch_count": mismatch_count,
            "mode": args.mode,
            "normalized_bits_sha256": NORMALIZED_BITS_SHA256,
            "optimized_hlo_sha256": optimized_sha,
            "partial_bits_sha256": None if acquisition else _sha256_array(partial_bits),
            "performance_claim": False,
            "source_file_sha256": SOURCE_FILE_SHA256,
            "stablehlo_sha256": stable_sha,
            "status": status,
            "tensor_file_sha256": None if acquisition else _sha256_file(args.tensor_output),
            "topology_fleet_sha256": fleet_sha,
            "topology_sha256": topology.topology_hash,
        }
        _atomic_json(args.output, record)
        if status == "ORACLE_MISMATCH":
            raise RuntimeError("bounded real dense layer is not bitwise exact")
        print(
            f"GREENFIELD_WS32_LAYER0_{status} rank={args.process_id} "
            f"stable={stable_sha} optimized={optimized_sha}",
            flush=True,
        )
        return 0
    finally:
        jax.distributed.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
