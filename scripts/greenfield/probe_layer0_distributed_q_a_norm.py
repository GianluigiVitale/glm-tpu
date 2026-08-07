#!/usr/bin/env python3
"""Run the bounded real TP32 q-a projection/RMSNorm association on TPU."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
EXPECTED_WORKTREE = Path("/home/gianl/glm-tpu-topology-rewrite")
SEALED_LEGACY_FUSED_QKV_WEIGHT_BYTE_SUM = 2_448_103_424
SEALED_LEGACY_FUSED_QKV_SCALE_BYTE_SUM = 53_100_864


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


def _array_digest(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _manifest_hash(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return sha256(encoded).hexdigest()


def _fleet_digests(
    multihost_utils: Any,
    digest_hex: str,
    *,
    process_count: int,
) -> list[str]:
    digest = np.frombuffer(bytes.fromhex(digest_hex), dtype=np.uint8)
    gathered = np.asarray(multihost_utils.process_allgather(digest)).reshape(
        process_count, digest.size
    )
    return [row.tobytes().hex() for row in gathered]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator-address", required=True)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--hlo-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if REPO != EXPECTED_WORKTREE:
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    if args.num_processes != 8 or not 0 <= args.process_id < 8:
        raise RuntimeError("protected distributed q-a norm requires ranks 0..7")
    code_hash = _git_head()
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale code hash: expected={args.expected_code_hash} found={code_hash}"
        )

    import jax
    import jax.numpy as jnp
    from jax.experimental import multihost_utils
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import ml_dtypes

    from glm_tpu.greenfield.benchmarking.dsa_association import (
        validate_dsa_association_hlo,
    )
    from glm_tpu.greenfield.kernels.reference.dsa_association import (
        Layer0DsaProbeGeometry,
        LegacyTp32QaNormOutput,
        layer0_decode_normalized_hidden,
        legacy_tp32_fused_qkv_a_rms_norm,
        pack_legacy_fused_qkv_runtime_weights,
    )
    from glm_tpu.greenfield.validation.layer0_dsa_association import (
        inspect_layer0_dsa_association_input,
    )

    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=args.num_processes,
        process_id=args.process_id,
    )
    try:
        if jax.default_backend() != "tpu" or (
            jax.process_count() != 8
            or jax.local_device_count() != 4
            or jax.device_count() != 32
        ):
            raise RuntimeError(
                "distributed q-a norm requires TPU and exact 8x4 process geometry"
            )
        # TPU JAX topology-orders processes independently of TPU-VM worker
        # suffixes. Preserve both identities and validate each fleet bijection;
        # they are deliberately not required to be equal.
        manifest, arrays = inspect_layer0_dsa_association_input(
            args.input_dir,
            expected_manifest_sha256=args.input_manifest_sha256,
        )
        geometry = Layer0DsaProbeGeometry()
        unique_ids = arrays["unique_token_ids"]
        current_row_host = np.searchsorted(
            unique_ids, arrays["current_token_id"]
        ).astype(np.int32)
        if not np.array_equal(
            unique_ids[current_row_host], arrays["current_token_id"]
        ):
            raise RuntimeError("distributed q-a norm embedding map drifted")

        def bf16_host(name: str) -> np.ndarray:
            return arrays[name].view(ml_dtypes.bfloat16)

        local_device = jax.local_devices()[0]
        source_values = tuple(
            jax.device_put(value, local_device)
            for value in (
                arrays["self_attn__q_a_proj__weight"],
                arrays["self_attn__q_a_proj__weight_scale_inv"],
                arrays["self_attn__kv_a_proj_with_mqa__weight"],
                arrays[
                    "self_attn__kv_a_proj_with_mqa__weight_scale_inv"
                ],
            )
        )
        pack_function = jax.jit(pack_legacy_fused_qkv_runtime_weights)
        pack_started = time.monotonic()
        pack_compiled = pack_function.lower(*source_values).compile()
        pack_compile_seconds = time.monotonic() - pack_started
        runtime_weights = pack_compiled(*source_values)
        jax.block_until_ready(runtime_weights)
        global_weight_host = np.asarray(runtime_weights.global_weight)
        global_scale_host = np.ascontiguousarray(
            np.asarray(runtime_weights.global_scale, dtype=np.float32)
        )
        weight_byte_sum = int(
            global_weight_host.view(np.uint8).sum(dtype=np.uint64)
        )
        scale_byte_sum = int(
            global_scale_host.view(np.uint8).sum(dtype=np.uint64)
        )
        if (
            weight_byte_sum != SEALED_LEGACY_FUSED_QKV_WEIGHT_BYTE_SUM
            or scale_byte_sum != SEALED_LEGACY_FUSED_QKV_SCALE_BYTE_SUM
        ):
            raise RuntimeError(
                "distributed q-a norm runtime pack disagrees with sealed state"
            )

        normalize_arguments = (
            jax.device_put(
                bf16_host("unique_embedding_bfloat16_bits"), local_device
            ),
            jax.device_put(current_row_host, local_device),
            jax.device_put(bf16_host("input_layernorm__weight"), local_device),
        )
        normalize_function = jax.jit(layer0_decode_normalized_hidden)
        normalize_started = time.monotonic()
        normalize_compiled = normalize_function.lower(
            *normalize_arguments
        ).compile()
        normalize_compile_seconds = time.monotonic() - normalize_started
        normalized_host = np.asarray(
            normalize_compiled(*normalize_arguments),
            dtype=ml_dtypes.bfloat16,
        )
        q_a_norm_host = bf16_host("self_attn__q_a_layernorm__weight")

        mesh = Mesh(
            np.asarray(jax.devices(), dtype=object),
            ("legacy_model",),
        )

        def global_array(value: np.ndarray, sharding: NamedSharding) -> Any:
            contiguous = np.ascontiguousarray(value)
            return jax.make_array_from_callback(
                contiguous.shape,
                sharding,
                lambda index: contiguous[index],
            )

        distributed_arguments = (
            global_array(normalized_host, NamedSharding(mesh, P())),
            global_array(
                global_weight_host,
                NamedSharding(mesh, P(None, "legacy_model")),
            ),
            global_array(
                global_scale_host,
                NamedSharding(mesh, P(None, "legacy_model")),
            ),
            global_array(q_a_norm_host, NamedSharding(mesh, P())),
        )
        mapped = jax.shard_map(
            lambda hidden, weight, scale, norm_weight: (
                legacy_tp32_fused_qkv_a_rms_norm(
                    hidden,
                    weight,
                    scale,
                    norm_weight,
                    axis_name="legacy_model",
                    geometry=geometry,
                )
            ),
            mesh=mesh,
            in_specs=(
                P(),
                P(None, "legacy_model"),
                P(None, "legacy_model"),
                P(),
            ),
            out_specs=LegacyTp32QaNormOutput(
                P(),
                P(None, "legacy_model"),
            ),
            check_vma=False,
        )
        multihost_utils.sync_global_devices(
            "greenfield-layer0-distributed-q-a-norm-compile"
        )
        compile_started = time.monotonic()
        compiled = jax.jit(mapped).lower(*distributed_arguments).compile()
        compile_seconds = time.monotonic() - compile_started
        optimized_hlo = compiled.as_text()
        hlo_sha256 = sha256(optimized_hlo.encode()).hexdigest()
        hlo_contract = validate_dsa_association_hlo(
            optimized_hlo,
            phase="legacy_tp32_distributed_q_a_norm",
        )
        if not hlo_contract["passed"] or hlo_contract[
            "distributed_collective_violations"
        ]:
            raise RuntimeError(
                f"distributed q-a norm HLO contract failed: {hlo_contract}"
            )
        hlo_fleet = _fleet_digests(
            multihost_utils,
            hlo_sha256,
            process_count=args.num_processes,
        )
        if len(set(hlo_fleet)) != 1:
            raise RuntimeError(f"distributed q-a norm HLO differs: {hlo_fleet}")

        multihost_utils.sync_global_devices(
            "greenfield-layer0-distributed-q-a-norm-execute"
        )
        output = compiled(*distributed_arguments)
        jax.block_until_ready(output)
        q_addressable = [
            np.asarray(shard.data, dtype=ml_dtypes.bfloat16)
            for shard in output.q_residual.addressable_shards
        ]
        if len(q_addressable) != 4 or any(
            not np.array_equal(value.view(np.uint16), q_addressable[0].view(np.uint16))
            for value in q_addressable[1:]
        ):
            raise RuntimeError("replicated q-a norm result differs locally")
        q_host = np.ascontiguousarray(q_addressable[0])
        if q_host.shape != (geometry.decode_rows, geometry.q_lora_rank) or (
            not np.isfinite(q_host.astype(np.float32)).all()
        ):
            raise RuntimeError("distributed q-a norm output contract drifted")
        q_sha256 = _array_digest(q_host)
        q_fleet = _fleet_digests(
            multihost_utils,
            q_sha256,
            process_count=args.num_processes,
        )
        if len(set(q_fleet)) != 1:
            raise RuntimeError(f"distributed q-a norm result differs: {q_fleet}")

        companion_shards = [
            np.asarray(shard.data, dtype=ml_dtypes.bfloat16)
            for shard in output.qkv_a_companion_shard.addressable_shards
        ]
        if len(companion_shards) != 4 or any(
            value.shape != (geometry.decode_rows, 18)
            or not np.isfinite(value.astype(np.float32)).all()
            for value in companion_shards
        ):
            raise RuntimeError("distributed qkv companion shard drifted")

        record = {
            "status": "SUCCESS",
            "claim_scope": (
                "bounded legacy-TP32 arithmetic diagnostic only; the full-pod "
                "collectives are forbidden in greenfield production layers"
            ),
            "backend": jax.default_backend(),
            "code_hash": code_hash,
            "hostname": socket.gethostname(),
            "input_manifest_sha256": manifest["manifest_sha256"],
            "launch_process_id": args.process_id,
            "jax_process_index": jax.process_index(),
            "local_device_ids": [device.id for device in jax.local_devices()],
            "global_device_count": jax.device_count(),
            "compile_seconds": {
                "distributed_q_a_norm": compile_seconds,
                "normalize_input": normalize_compile_seconds,
                "runtime_pack": pack_compile_seconds,
            },
            "hlo_sha256": hlo_sha256,
            "hlo_contract": hlo_contract,
            "hlo_fleet_sha256": hlo_fleet,
            "q_residual": {
                "shape": list(q_host.shape),
                "dtype": "bfloat16",
                "byte_count": int(q_host.nbytes),
                "sha256": q_sha256,
                "fleet_sha256": q_fleet,
                "locally_replicated_shards": len(q_addressable),
            },
            "local_companion_shards": [
                {
                    "shape": list(value.shape),
                    "dtype": "bfloat16",
                    "sha256": _array_digest(value),
                }
                for value in companion_shards
            ],
            "runtime_layout_identity": {
                "global_weight_shape": list(global_weight_host.shape),
                "global_weight_dtype": str(global_weight_host.dtype),
                "global_weight_byte_sum": weight_byte_sum,
                "global_weight_sha256": _array_digest(global_weight_host),
                "global_scale_shape": list(global_scale_host.shape),
                "global_scale_dtype": str(global_scale_host.dtype),
                "global_scale_byte_sum": scale_byte_sum,
                "global_scale_sha256": _array_digest(global_scale_host),
            },
            "diagnostic_only_full_pod_collectives": True,
            "profiler_free_timing": False,
        }
        _atomic_json(args.output, record)

        if jax.process_index() == 0:
            from safetensors.numpy import save_file

            args.artifact_dir.mkdir(parents=True, exist_ok=False)
            args.hlo_dir.mkdir(parents=True, exist_ok=True)
            hlo_path = args.hlo_dir / "distributed_q_a_norm.optimized_hlo.txt"
            hlo_path.write_text(optimized_hlo)
            tensor_path = args.artifact_dir / "distributed_q_a_norm.safetensors"
            q_bits = q_host.view(np.uint16)
            save_file(
                {"q_residual_bfloat16_bits": q_bits},
                tensor_path,
                metadata={
                    "artifact_kind": "greenfield_distributed_q_a_norm",
                    "format_version": "1",
                },
            )
            artifact_manifest = {
                "artifact_kind": "greenfield_distributed_q_a_norm",
                "format_version": 1,
                "diagnostic_only": True,
                "code_hash": code_hash,
                "input_manifest_sha256": manifest["manifest_sha256"],
                "hlo_sha256": hlo_sha256,
                "q_residual": record["q_residual"],
                "file": {
                    "filename": tensor_path.name,
                    "byte_count": tensor_path.stat().st_size,
                    "sha256": _sha256_file(tensor_path),
                },
            }
            artifact_manifest["manifest_sha256"] = _manifest_hash(
                artifact_manifest
            )
            _atomic_json(args.artifact_dir / "manifest.json", artifact_manifest)
        multihost_utils.sync_global_devices(
            "greenfield-layer0-distributed-q-a-norm-finished"
        )
        print(json.dumps(record, sort_keys=True))
        return 0
    finally:
        jax.distributed.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
