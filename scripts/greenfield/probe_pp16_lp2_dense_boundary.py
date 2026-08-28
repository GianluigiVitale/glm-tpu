#!/usr/bin/env python3
"""Bounded real-weight PP16 layer-0 dense/cross-layer discriminator."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_DENSE_NAMES = (
    "dense.slot_00.gate.weight_bits",
    "dense.slot_00.gate.scale_inv",
    "dense.slot_00.up.weight_bits",
    "dense.slot_00.up.scale_inv",
    "dense.slot_00.down.weight_bits",
    "dense.slot_00.down.scale_inv",
)
_NORM_NAME = "attention.slot_01.input_norm"
_EXPECTED_SHAPES = {
    "dense.slot_00.gate.weight_bits": (6144, 6144),
    "dense.slot_00.gate.scale_inv": (48, 48),
    "dense.slot_00.up.weight_bits": (6144, 6144),
    "dense.slot_00.up.scale_inv": (48, 48),
    "dense.slot_00.down.weight_bits": (6144, 6144),
    "dense.slot_00.down.scale_inv": (48, 48),
    _NORM_NAME: (6144,),
}
_EXPECTED_DTYPES = {
    "dense.slot_00.gate.weight_bits": "torch.uint8",
    "dense.slot_00.gate.scale_inv": "torch.float32",
    "dense.slot_00.up.weight_bits": "torch.uint8",
    "dense.slot_00.up.scale_inv": "torch.float32",
    "dense.slot_00.down.weight_bits": "torch.uint8",
    "dense.slot_00.down.scale_inv": "torch.float32",
    _NORM_NAME: "torch.bfloat16",
}


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(value)
    temporary.replace(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def _distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "samples_ms": [float(value) for value in array],
    }


def _memory_stats(device: Any) -> dict[str, int] | None:
    value = device.memory_stats()
    if value is None:
        return None
    return {
        str(name): int(number)
        for name, number in value.items()
        if isinstance(number, int) and not isinstance(number, bool)
    }


def _load_manifest(
    root: Path, expected_manifest_sha256: str
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    from glm_tpu.greenfield.checkpoint.full_loader import _mapping_hash

    manifest_path = root / "runtime_manifest.json"
    success_path = root / "SUCCESS"
    if not manifest_path.is_file() or not success_path.is_file():
        raise RuntimeError("PP16 feature runtime is incomplete")
    manifest = json.loads(manifest_path.read_text())
    observed = _mapping_hash(manifest, hash_field="manifest_sha256")
    if (
        observed != expected_manifest_sha256
        or manifest.get("manifest_sha256") != expected_manifest_sha256
    ):
        raise RuntimeError("PP16 feature runtime manifest identity drifted")
    if success_path.read_text() != f"{expected_manifest_sha256}  runtime_manifest.json\n":
        raise RuntimeError("PP16 feature runtime SUCCESS marker drifted")
    if (
        manifest.get("artifact_kind")
        != "greenfield_feature_runtime_packed_checkpoint"
        or manifest.get("model_id") != "zai-org/GLM-5.2-FP8"
        or manifest.get("plan_id") != "PP16_LP2"
        or not str(manifest.get("destination", "")).startswith(
            "gs://driftbench-dsv4-uc/checkpoints/greenfield/"
        )
    ):
        raise RuntimeError("PP16 feature runtime semantic identity drifted")
    by_slot: dict[int, dict[str, Any]] = {}
    for record in manifest.get("files", []):
        if record.get("stage_id") == 0 and record.get("device_slot") in (0, 1):
            by_slot[int(record["device_slot"])] = record
    if set(by_slot) != {0, 1}:
        raise RuntimeError("PP16 stage-0 owner file ledger is incomplete")
    return manifest, by_slot


def _load_owner(
    root: Path, record: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch
    from safetensors import safe_open

    slot = int(record["device_slot"])
    filename = record.get("destination_filename")
    if (
        record.get("stage_id") != 0
        or record.get("device_id") != slot
        or not isinstance(filename, str)
        or not isinstance(record.get("generation"), int)
        or record["generation"] <= 0
        or not isinstance(record.get("crc32c"), str)
        or not record["crc32c"]
    ):
        raise RuntimeError(f"PP16 owner {slot} file identity is invalid")
    path = root / filename
    if not path.is_file() or path.stat().st_size != record.get("file_bytes"):
        raise RuntimeError(f"PP16 owner {slot} payload size drifted")
    header_bytes = int(record.get("header_bytes", 0))
    with path.open("rb") as stream:
        header = stream.read(header_bytes)
    if len(header) != header_bytes or sha256(header).hexdigest() != record.get(
        "header_sha256"
    ):
        raise RuntimeError(f"PP16 owner {slot} header identity drifted")

    tensor_records = {
        item.get("name"): item
        for item in record.get("tensors", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    names = (*_DENSE_NAMES, _NORM_NAME)
    if any(name not in tensor_records for name in names):
        raise RuntimeError(f"PP16 owner {slot} tensor ledger is incomplete")
    arrays: dict[str, np.ndarray] = {}
    evidence: dict[str, Any] = {}
    with safe_open(path, framework="pt", device="cpu") as stream:
        keys = frozenset(stream.keys())
        for name in names:
            if name not in keys:
                raise RuntimeError(f"PP16 owner {slot} tensor {name!r} is absent")
            tensor = stream.get_tensor(name)
            if (
                tuple(tensor.shape) != _EXPECTED_SHAPES[name]
                or str(tensor.dtype) != _EXPECTED_DTYPES[name]
            ):
                raise RuntimeError(
                    f"PP16 owner {slot} tensor {name!r} geometry drifted"
                )
            array = (
                tensor.view(torch.uint16).numpy()
                if tensor.dtype == torch.bfloat16
                else tensor.numpy()
            )
            array = np.ascontiguousarray(array)
            tensor_record = tensor_records[name]
            digest = _sha256_array(array)
            if (
                digest != tensor_record.get("sha256")
                or array.nbytes != tensor_record.get("byte_count")
                or tensor_record.get("padding") is not False
            ):
                raise RuntimeError(
                    f"PP16 owner {slot} tensor {name!r} identity drifted"
                )
            arrays[name] = array
            evidence[name] = {
                "byte_count": int(array.nbytes),
                "dtype": str(array.dtype),
                "sha256": digest,
                "shape": list(array.shape),
            }
    return arrays, {
        "crc32c": record["crc32c"],
        "destination_filename": filename,
        "device_id": record["device_id"],
        "device_slot": slot,
        "file_bytes": record["file_bytes"],
        "file_sha256_manifest": record["sha256"],
        "generation": record["generation"],
        "header_sha256": record["header_sha256"],
        "selective_tensor_bytes": sum(
            int(item["byte_count"]) for item in evidence.values()
        ),
        "tensors": evidence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--oracle-npz", type=Path, required=True)
    parser.add_argument("--oracle-npz-sha256", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-output", type=Path, required=True)
    parser.add_argument("--hlo-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if REPO != Path("/home/gianl/glm-tpu-topology-rewrite"):
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    code_hash = _git_head()
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale code hash: expected={args.expected_code_hash} found={code_hash}"
        )
    if (args.warmup, args.iterations) != (1, 3):
        raise ValueError("bounded LP2 dense probe requires exactly 1/3 samples")
    if _sha256_file(args.oracle_npz) != args.oracle_npz_sha256:
        raise RuntimeError("sealed dense boundary oracle identity drifted")

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import ml_dtypes

    from glm_tpu.greenfield.benchmarking.pp16_dense_boundary import (
        PP16_DENSE_BOUNDARY_ACCEPTED_DENSE_SHA256,
        PP16_DENSE_BOUNDARY_ACCEPTED_RESIDUAL_SHA256,
        derive_expected_dense_boundary_bits,
        exact_bfloat16_bits,
        validate_pp16_dense_boundary_hlo,
    )
    from glm_tpu.greenfield.kernels.reference.rmsnorm import fused_add_rms_norm
    from glm_tpu.greenfield.kernels.stage_local import (
        STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE,
        stage_local_dense_fp8_mapped,
    )

    if jax.default_backend() != "tpu" or jax.local_device_count() != 4:
        raise RuntimeError(
            "LP2 dense probe requires one four-chip TPU host, got "
            f"backend={jax.default_backend()} local={jax.local_device_count()}"
        )
    devices = tuple(jax.local_devices()[:2])
    coordinates = [list(device.coords) for device in devices]
    if [device.id for device in devices] != [0, 1] or coordinates != [
        [0, 0, 0],
        [1, 0, 0],
    ]:
        raise RuntimeError(
            "LP2 dense probe lost the protected adjacent stage-0 pair: "
            f"ids={[device.id for device in devices]} coords={coordinates}"
        )

    manifest, records = _load_manifest(
        args.runtime_root, args.runtime_manifest_sha256
    )
    owners: list[dict[str, np.ndarray]] = []
    owner_evidence = []
    for slot in (0, 1):
        arrays, evidence = _load_owner(args.runtime_root, records[slot])
        owners.append(arrays)
        owner_evidence.append(evidence)

    with np.load(args.oracle_npz, allow_pickle=False) as payload:
        required = {
            "accepted_layer1_normalized_bfloat16_bits",
            "dense_virtual_partials_bfloat16_bits",
            "layer1_input_norm_bfloat16_bits",
            "normalized_mlp_bfloat16_bits",
            "post_attention_residual_bfloat16_bits",
        }
        if not required.issubset(payload.files):
            raise RuntimeError("sealed dense boundary oracle is incomplete")
        accepted_bits = np.ascontiguousarray(
            payload["accepted_layer1_normalized_bfloat16_bits"]
        )
        dense_partial_bits = np.ascontiguousarray(
            payload["dense_virtual_partials_bfloat16_bits"]
        )
        layer1_norm_bits = np.ascontiguousarray(
            payload["layer1_input_norm_bfloat16_bits"]
        )
        normalized_bits = np.ascontiguousarray(
            payload["normalized_mlp_bfloat16_bits"]
        )
        residual_bits = np.ascontiguousarray(
            payload["post_attention_residual_bfloat16_bits"]
        )
    expected_oracle = {
        "accepted": ((6144,), np.uint16),
        "dense_partials": ((4, 8, 1, 6144), np.uint16),
        "layer1_norm": ((6144,), np.uint16),
        "normalized": ((1, 6144), np.uint16),
        "residual": ((1, 6144), np.uint16),
    }
    for name, value in (
        ("accepted", accepted_bits),
        ("dense_partials", dense_partial_bits),
        ("layer1_norm", layer1_norm_bits),
        ("normalized", normalized_bits),
        ("residual", residual_bits),
    ):
        shape, dtype = expected_oracle[name]
        if value.shape != shape or value.dtype != dtype:
            raise RuntimeError(f"sealed dense boundary {name} geometry drifted")
    if not all(
        np.array_equal(owner[_NORM_NAME], layer1_norm_bits) for owner in owners
    ):
        raise RuntimeError("PP16 final runtime layer-1 norm differs from oracle")
    model_axis_device_ids = tuple(
        int(value)
        for value in np.argsort(
            np.asarray(STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE)
        )
    )
    expected_dense_bits, expected_residual_bits = (
        derive_expected_dense_boundary_bits(
            dense_partial_bits, residual_bits, model_axis_device_ids
        )
    )
    if (
        _sha256_array(expected_dense_bits)
        != PP16_DENSE_BOUNDARY_ACCEPTED_DENSE_SHA256
        or _sha256_array(expected_residual_bits)
        != PP16_DENSE_BOUNDARY_ACCEPTED_RESIDUAL_SHA256
    ):
        raise RuntimeError("derived accepted PP16 boundary state drifted")

    mesh = Mesh(np.asarray(devices, dtype=object), ("stage",))
    replicated = NamedSharding(mesh, P())

    def put_replicated_bits(bits: np.ndarray) -> Any:
        return jax.device_put(bits.view(ml_dtypes.bfloat16), replicated)

    normalized = put_replicated_bits(normalized_bits)
    residual = put_replicated_bits(residual_bits)
    layer1_norm = put_replicated_bits(layer1_norm_bits)

    device_weights = {}
    for name in _DENSE_NAMES:
        host = np.stack([owner[name] for owner in owners], axis=0)
        spec = P("stage", *(None for _ in _EXPECTED_SHAPES[name]))
        device_weights[name] = jax.device_put(host, NamedSharding(mesh, spec))

    def local_dense_boundary(
        normalized_value: Any,
        residual_value: Any,
        gate_bits_slot: Any,
        gate_scale_slot: Any,
        up_bits_slot: Any,
        up_scale_slot: Any,
        down_bits_slot: Any,
        down_scale_slot: Any,
        next_norm: Any,
    ) -> tuple[Any, Any, Any]:
        dense_update = stage_local_dense_fp8_mapped(
            residual_value,
            next_norm,
            gate_bits_slot[0],
            gate_scale_slot[0],
            up_bits_slot[0],
            up_scale_slot[0],
            down_bits_slot[0],
            down_scale_slot[0],
            axis_name="stage",
            axis_index_groups=((0, 1),),
            block_shape=(128, 128),
            epsilon=1e-5,
            linear_backend="pallas",
            precomputed_normalized=normalized_value,
            add_residual=False,
        )
        layer1_normalized, next_residual = fused_add_rms_norm(
            dense_update,
            residual_value,
            next_norm,
            epsilon=1e-5,
        )
        return dense_update, layer1_normalized, next_residual

    weight_spec = P("stage", None, None)
    mapped = jax.shard_map(
        local_dense_boundary,
        mesh=mesh,
        in_specs=(
            P(),
            P(),
            weight_spec,
            weight_spec,
            weight_spec,
            weight_spec,
            weight_spec,
            weight_spec,
            P(),
        ),
        out_specs=(P(), P(), P()),
        check_vma=False,
    )
    program_args = (
        normalized,
        residual,
        device_weights["dense.slot_00.gate.weight_bits"],
        device_weights["dense.slot_00.gate.scale_inv"],
        device_weights["dense.slot_00.up.weight_bits"],
        device_weights["dense.slot_00.up.scale_inv"],
        device_weights["dense.slot_00.down.weight_bits"],
        device_weights["dense.slot_00.down.scale_inv"],
        layer1_norm,
    )
    args.hlo_dir.mkdir(parents=True, exist_ok=True)
    compile_started = time.monotonic()
    lowered = jax.jit(mapped).lower(*program_args)
    stablehlo = lowered.as_text()
    compiled = lowered.compile()
    compile_seconds = time.monotonic() - compile_started
    optimized_hlo = compiled.as_text()
    _atomic_text(args.hlo_dir / "dense_boundary.stablehlo.mlir", stablehlo)
    _atomic_text(args.hlo_dir / "dense_boundary.optimized_hlo.txt", optimized_hlo)
    hlo_contract = validate_pp16_dense_boundary_hlo(stablehlo, optimized_hlo)
    if not hlo_contract["passed"]:
        raise RuntimeError(f"PP16 dense boundary HLO failed: {hlo_contract}")

    for _ in range(args.warmup):
        jax.block_until_ready(compiled(*program_args))
    samples_ms = []
    final_output = None
    sample_output_bits: list[tuple[np.ndarray, ...]] = []
    sample_replica_records: list[dict[str, Any]] = []

    def replicated_bits(value: Any) -> tuple[np.ndarray, dict[str, Any]]:
        arrays = []
        records = []
        for shard in value.addressable_shards:
            bits = np.ascontiguousarray(np.asarray(shard.data)).view(np.uint16)
            arrays.append(bits)
            records.append(
                {
                    "coordinates": list(shard.device.coords),
                    "device_id": int(shard.device.id),
                    "sha256": _sha256_array(bits),
                    "shape": list(bits.shape),
                }
            )
        if len(arrays) != 2:
            raise RuntimeError(
                f"PP16 replicated output has {len(arrays)} local shards"
            )
        agreed = bool(np.array_equal(arrays[0], arrays[1]))
        return arrays[0], {"agreed": agreed, "replicas": records}

    for _ in range(args.iterations):
        started = time.perf_counter()
        final_output = compiled(*program_args)
        jax.block_until_ready(final_output)
        samples_ms.append((time.perf_counter() - started) * 1000.0)
        logical = []
        replica_record = {}
        for name, value in zip(
            ("dense_update", "layer1_normalized", "next_residual"),
            final_output,
            strict=True,
        ):
            bits, record = replicated_bits(value)
            logical.append(bits)
            replica_record[name] = record
        sample_output_bits.append(tuple(logical))
        sample_replica_records.append(replica_record)
    assert final_output is not None
    dense_update_bits, observed_bits, next_residual_bits = sample_output_bits[-1]
    output_names = ("dense_update", "layer1_normalized", "next_residual")
    determinism = {
        name: {
            "passed": all(
                np.array_equal(sample_output_bits[0][index], sample[index])
                for sample in sample_output_bits[1:]
            ),
            "sample_sha256": [
                _sha256_array(sample[index]) for sample in sample_output_bits
            ],
        }
        for index, name in enumerate(output_names)
    }
    replica_agreement = {
        "passed": all(
            record[name]["agreed"]
            for record in sample_replica_records
            for name in output_names
        ),
        "samples": sample_replica_records,
    }
    dense_comparison = exact_bfloat16_bits(
        expected_dense_bits, dense_update_bits
    )
    normalized_comparison = exact_bfloat16_bits(
        accepted_bits, observed_bits.reshape(-1)
    )
    residual_comparison = exact_bfloat16_bits(
        expected_residual_bits, next_residual_bits
    )
    state_exact = bool(
        normalized_comparison["elementwise_exact"]
        and residual_comparison["elementwise_exact"]
        and all(record["passed"] for record in determinism.values())
        and replica_agreement["passed"]
    )
    status = "SUCCESS" if state_exact else "NONEXACT"

    tensor_records = {
        "accepted_layer1_normalized_bfloat16_bits": {
            "sha256": _sha256_array(accepted_bits),
            "shape": list(accepted_bits.shape),
        },
        "accepted_dense_update_bfloat16_bits": {
            "sha256": _sha256_array(expected_dense_bits),
            "shape": list(expected_dense_bits.shape),
        },
        "accepted_next_residual_bfloat16_bits": {
            "sha256": _sha256_array(expected_residual_bits),
            "shape": list(expected_residual_bits.shape),
        },
        "dense_update_bfloat16_bits": {
            "sha256": _sha256_array(dense_update_bits),
            "shape": list(dense_update_bits.shape),
        },
        "layer1_normalized_bfloat16_bits": {
            "sha256": _sha256_array(observed_bits),
            "shape": list(observed_bits.shape),
        },
        "next_residual_bfloat16_bits": {
            "sha256": _sha256_array(next_residual_bits),
            "shape": list(next_residual_bits.shape),
        },
    }
    _atomic_npz(
        args.tensor_output,
        accepted_layer1_normalized_bfloat16_bits=accepted_bits,
        accepted_dense_update_bfloat16_bits=expected_dense_bits,
        accepted_next_residual_bfloat16_bits=expected_residual_bits,
        dense_update_bfloat16_bits=dense_update_bits,
        layer1_normalized_bfloat16_bits=observed_bits,
        next_residual_bfloat16_bits=next_residual_bits,
    )
    output = {
        "artifact_kind": "greenfield_pp16_lp2_dense_boundary_discriminator",
        "claim_scope": (
            "bounded real layer-0 dense/cross-layer arithmetic and exact HLO; "
            "no full decoder, DSA event-1, Gate-D, or performance claim"
        ),
        "code_hash": code_hash,
        "compile_seconds": compile_seconds,
        "device_kind": devices[0].device_kind,
        "device_memory": [_memory_stats(device) for device in devices],
        "dense_update_comparison": dense_comparison,
        "determinism": determinism,
        "hlo_contract": hlo_contract,
        "iterations": args.iterations,
        "layer1_comparison": normalized_comparison,
        "next_residual_comparison": residual_comparison,
        "physical_group": {
            "coordinates": coordinates,
            "device_ids": [device.id for device in devices],
            "local_device_count_visible": jax.local_device_count(),
            "mesh_device_count": 2,
            "replica_groups": [[0, 1]],
        },
        "position": 8155,
        "profiler_free_timing": True,
        "replica_agreement": replica_agreement,
        "runtime": {
            "destination": manifest["destination"],
            "manifest_sha256": args.runtime_manifest_sha256,
            "owners": owner_evidence,
            "plan_hash": manifest["plan_hash"],
            "plan_id": manifest["plan_id"],
            "runtime_layout_hash": manifest["runtime_layout_hash"],
            "schedule_hash": manifest["schedule_hash"],
        },
        "source": {
            "oracle_npz_sha256": args.oracle_npz_sha256,
            "oracle_tensor_sha256": {
                "accepted_layer1": _sha256_array(accepted_bits),
                "dense_virtual_partials": _sha256_array(dense_partial_bits),
                "derived_accepted_dense_update": _sha256_array(
                    expected_dense_bits
                ),
                "derived_accepted_next_residual": _sha256_array(
                    expected_residual_bits
                ),
                "layer1_norm": _sha256_array(layer1_norm_bits),
                "normalized_mlp": _sha256_array(normalized_bits),
                "post_attention_residual": _sha256_array(residual_bits),
            },
        },
        "status": status,
        "tensor_output": {
            "filename": args.tensor_output.name,
            "records": tensor_records,
        },
        "timing": _distribution(samples_ms),
        "warmup": args.warmup,
    }
    _atomic_json(args.output, output)
    print(json.dumps(output, allow_nan=False, sort_keys=True))
    return 0 if status == "SUCCESS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
