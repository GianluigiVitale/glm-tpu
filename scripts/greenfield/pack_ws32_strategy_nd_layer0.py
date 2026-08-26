#!/usr/bin/env python3
"""Derive the bounded WS32 layer-0 StrategyND checkpoint from sealed PP8 ranks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
EXPECTED_REPO = Path("/home/gianl/glm-tpu-topology-rewrite")
SOURCE_TENSORS = (
    "dense.slot_00.merged_gate_up.weight_bits_in_out",
    "dense.slot_00.merged_gate_up.scale_inv_in_out",
    "dense.slot_00.down.weight_bits_in_out",
    "dense.slot_00.down.scale_inv_in_out",
)
SOURCE_SHAPES = {
    SOURCE_TENSORS[0]: (8, 6144, 768),
    SOURCE_TENSORS[1]: (8, 48, 768),
    SOURCE_TENSORS[2]: (8, 384, 6144),
    SOURCE_TENSORS[3]: (8, 3, 6144),
}
SOURCE_DTYPES = {
    SOURCE_TENSORS[0]: "U8",
    SOURCE_TENSORS[1]: "F32",
    SOURCE_TENSORS[2]: "U8",
    SOURCE_TENSORS[3]: "F32",
}
DESTINATION_NAMES = (
    "merged_gate_up.weight_bits_in_out",
    "merged_gate_up.scale_inv_in_out",
    "down.weight_bits_in_out",
    "down.scale_inv_in_out",
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


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-manifest-file-sha256", required=True)
    parser.add_argument("--source-success-file-sha256", required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def _source_file_records(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result = {}
    for record in manifest.get("files", []):
        if record.get("stage_id") == 0 and record.get("device_slot") in range(4):
            result[int(record["device_slot"])] = record
    if set(result) != set(range(4)):
        raise RuntimeError("sealed source manifest has no exact stage-0 owner set")
    return result


def _source_tensor_records(
    file_records: dict[int, dict[str, Any]],
) -> dict[tuple[int, str], dict[str, Any]]:
    result = {}
    for owner, file_record in file_records.items():
        by_name = {item["name"]: item for item in file_record["tensors"]}
        for name in SOURCE_TENSORS:
            try:
                tensor = by_name[name]
            except KeyError as error:
                raise RuntimeError(f"sealed source is missing {name!r}") from error
            if tensor.get("padding") or tuple(tensor.get("selected_shape", ())) not in (
                (),
                SOURCE_SHAPES[name],
            ):
                raise RuntimeError(f"sealed source tensor contract drifted: {name}")
            result[(owner, name)] = tensor
    return result


def _load_sources(
    source_root: Path,
    file_records: dict[int, dict[str, Any]],
    tensor_records: dict[tuple[int, str], dict[str, Any]],
) -> dict[tuple[int, str], np.ndarray]:
    from safetensors import safe_open

    result = {}
    for owner in range(4):
        file_record = file_records[owner]
        path = source_root / file_record["destination_filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in SOURCE_TENSORS:
                tensor = handle.get_tensor(name)
                value = np.ascontiguousarray(tensor.numpy())
                expected_dtype = np.uint8 if SOURCE_DTYPES[name] == "U8" else np.float32
                record = tensor_records[(owner, name)]
                if value.shape != SOURCE_SHAPES[name] or value.dtype != expected_dtype:
                    raise RuntimeError(
                        f"source payload geometry drifted: {owner}:{name}"
                    )
                if _sha256_array(value) != record["sha256"]:
                    raise RuntimeError(f"source tensor hash drifted: {owner}:{name}")
                result[(owner, name)] = value
    return result


def _model_rank_values(
    sources: dict[tuple[int, str], np.ndarray],
    name: str,
    ranks: tuple[int, ...],
) -> np.ndarray:
    return np.ascontiguousarray(
        np.stack(
            tuple(sources[(rank // 8, name)][rank % 8] for rank in ranks),
            axis=0,
        )
    )


def _model_ranks_for_expert(expert: int) -> tuple[int, ...]:
    if not 0 <= expert < 8:
        raise ValueError("WS32 expert coordinate is invalid")
    return tuple(range(expert * 4, expert * 4 + 4))


def _target_arrays(
    sources: dict[tuple[int, str], np.ndarray],
    *,
    expert: int,
    feature: int,
) -> tuple[tuple[int, ...], dict[str, np.ndarray]]:
    if not 0 <= expert < 8 or not 0 <= feature < 4:
        raise ValueError("WS32 target coordinate is invalid")
    ranks = _model_ranks_for_expert(expert)
    start = feature * 1536
    stop = start + 1536
    merged_bits = _model_rank_values(sources, SOURCE_TENSORS[0], ranks)
    merged_scale = _model_rank_values(sources, SOURCE_TENSORS[1], ranks)
    down_bits = _model_rank_values(sources, SOURCE_TENSORS[2], ranks)[
        ..., start:stop
    ]
    down_scale = _model_rank_values(sources, SOURCE_TENSORS[3], ranks)[
        ..., start:stop
    ]
    arrays = {
        DESTINATION_NAMES[0]: merged_bits,
        DESTINATION_NAMES[1]: merged_scale,
        DESTINATION_NAMES[2]: np.ascontiguousarray(down_bits),
        DESTINATION_NAMES[3]: np.ascontiguousarray(down_scale),
    }
    expected = {
        DESTINATION_NAMES[0]: ((4, 6144, 768), np.uint8),
        DESTINATION_NAMES[1]: ((4, 48, 768), np.float32),
        DESTINATION_NAMES[2]: ((4, 384, 1536), np.uint8),
        DESTINATION_NAMES[3]: ((4, 3, 1536), np.float32),
    }
    if any(
        arrays[name].shape != shape or arrays[name].dtype != dtype
        for name, (shape, dtype) in expected.items()
    ):
        raise RuntimeError("WS32 target tensor geometry drifted")
    return ranks, arrays


def main() -> int:
    args = _parse_args()
    if REPO != EXPECTED_REPO:
        raise RuntimeError(f"wrong worktree: {REPO}")
    code_hash = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if code_hash != args.expected_code_hash:
        raise RuntimeError("packer code hash drifted")
    if args.output_root.exists():
        raise FileExistsError("bounded checkpoint output is append-only")
    manifest_path = args.source_root / "runtime_manifest.json"
    success_path = args.source_root / "SUCCESS"
    if _sha256_file(manifest_path) != args.source_manifest_file_sha256 or (
        _sha256_file(success_path) != args.source_success_file_sha256
    ):
        raise RuntimeError("sealed source file identity drifted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("artifact_kind")
        != "greenfield_feature_runtime_packed_checkpoint"
        or manifest.get("plan_id") != "PP8_LP4"
        or manifest.get("dense_projection_layout")
        != "virtual_tp32_dense_convolution_in_out_v1"
    ):
        raise RuntimeError("sealed source checkpoint contract drifted")
    file_records = _source_file_records(manifest)
    tensor_records = _source_tensor_records(file_records)
    sources = _load_sources(args.source_root, file_records, tensor_records)

    from safetensors.torch import save_file
    import torch

    args.output_root.mkdir(parents=True)
    files = []
    for expert in range(8):
        for feature in range(4):
            ranks, arrays = _target_arrays(
                sources, expert=expert, feature=feature
            )
            relative = (
                Path(f"expert_{expert:02d}")
                / f"feature_{feature:02d}.safetensors"
            )
            destination = args.output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
            save_file(
                {name: torch.from_numpy(value) for name, value in arrays.items()},
                temporary,
                metadata={
                    "code_hash": code_hash,
                    "expert_coordinate": str(expert),
                    "feature_coordinate": str(feature),
                    "model_ranks": ",".join(str(item) for item in ranks),
                    "plan_id": "WS32_2D",
                },
            )
            temporary.replace(destination)
            files.append(
                {
                    "byte_count": destination.stat().st_size,
                    "expert_coordinate": expert,
                    "feature_coordinate": feature,
                    "filename": relative.as_posix(),
                    "model_ranks": list(ranks),
                    "sha256": _sha256_file(destination),
                    "tensors": {
                        name: {
                            "byte_count": value.nbytes,
                            "dtype": "U8" if value.dtype == np.uint8 else "F32",
                            "sha256": _sha256_array(value),
                            "shape": list(value.shape),
                        }
                        for name, value in arrays.items()
                    },
                }
            )
    source_evidence = []
    for owner in range(4):
        file_record = file_records[owner]
        source_evidence.append(
            {
                "device_slot": owner,
                "filename": file_record["destination_filename"],
                "file_sha256": file_record["sha256"],
                "tensors": {
                    name: tensor_records[(owner, name)]["sha256"]
                    for name in SOURCE_TENSORS
                },
            }
        )
    output_manifest = {
        "artifact_kind": "greenfield_ws32_strategy_nd_layer0_checkpoint",
        "code_hash": code_hash,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "files": files,
        "format_version": 1,
        "layer_id": 0,
        "model_id": "zai-org/GLM-5.2-FP8",
        "plan_id": "WS32_2D",
        "source_dense_projection_layout": manifest["dense_projection_layout"],
        "source_evidence": source_evidence,
        "source_manifest_file_sha256": args.source_manifest_file_sha256,
        "source_manifest_sha256": manifest["manifest_sha256"],
        "source_success_file_sha256": args.source_success_file_sha256,
        "total_bytes": sum(item["byte_count"] for item in files),
    }
    output_manifest["manifest_sha256"] = sha256(
        _canonical(output_manifest)
    ).hexdigest()
    _atomic_json(args.output_root / "manifest.json", output_manifest)
    success = {
        "artifact_kind": "greenfield_ws32_strategy_nd_layer0_checkpoint_success",
        "code_hash": code_hash,
        "manifest_file_sha256": _sha256_file(args.output_root / "manifest.json"),
        "manifest_sha256": output_manifest["manifest_sha256"],
    }
    success["success_sha256"] = sha256(_canonical(success)).hexdigest()
    _atomic_json(args.output_root / "SUCCESS", success)
    print(
        json.dumps(
            {
                "file_count": len(files),
                "manifest_sha256": output_manifest["manifest_sha256"],
                "success_sha256": success["success_sha256"],
                "total_bytes": output_manifest["total_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
