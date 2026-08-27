#!/usr/bin/env python3
"""Pack all three exact dense layers into final WS32 StrategyND owners."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from glm_tpu.greenfield.checkpoint.ws32_strategy_nd_dense import (
    strategy_nd_dense_tensor_names,
)


REPO = Path(__file__).resolve().parents[2]
EXPECTED_REPO = Path("/home/gianl/glm-tpu-topology-rewrite")
LAYERS = (0, 1, 2)
SOURCE_SUFFIXES = (
    "merged_gate_up.weight_bits_in_out",
    "merged_gate_up.scale_inv_in_out",
    "down.weight_bits_in_out",
    "down.scale_inv_in_out",
)
SOURCE_SHAPES = (
    (8, 6144, 768),
    (8, 48, 768),
    (8, 384, 6144),
    (8, 3, 6144),
)
SOURCE_DTYPES = (
    np.dtype(np.uint8),
    np.dtype(np.float32),
    np.dtype(np.uint8),
    np.dtype(np.float32),
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
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-manifest-file-sha256", required=True)
    parser.add_argument("--source-success-file-sha256", required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def _source_name(layer: int, suffix: str) -> str:
    return f"dense.slot_{layer:02d}.{suffix}"


def _source_file_records(manifest: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    records = {
        int(item["device_slot"]): item
        for item in manifest.get("files", ())
        if item.get("stage_id") == 0 and item.get("device_slot") in range(4)
    }
    if set(records) != set(range(4)):
        raise RuntimeError("sealed source manifest has no exact stage-0 owner set")
    return records


def _load_sources(
    source_root: Path,
    file_records: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[tuple[int, int, str], np.ndarray], list[Mapping[str, Any]]]:
    from safetensors import safe_open

    arrays: dict[tuple[int, int, str], np.ndarray] = {}
    evidence: list[Mapping[str, Any]] = []
    for owner in range(4):
        record = file_records[owner]
        path = source_root / str(record["destination_filename"])
        if not path.is_file() or _sha256_file(path) != record["sha256"]:
            raise RuntimeError("sealed source file identity drifted")
        tensor_records = {item["name"]: item for item in record["tensors"]}
        owner_evidence: dict[str, Any] = {
            "device_slot": owner,
            "filename": record["destination_filename"],
            "file_sha256": record["sha256"],
            "tensors": {},
        }
        with safe_open(path, framework="np") as handle:
            for layer in LAYERS:
                for suffix, shape, dtype in zip(
                    SOURCE_SUFFIXES,
                    SOURCE_SHAPES,
                    SOURCE_DTYPES,
                    strict=True,
                ):
                    name = _source_name(layer, suffix)
                    if name not in tensor_records:
                        raise RuntimeError(f"sealed source is missing {name!r}")
                    value = np.ascontiguousarray(handle.get_tensor(name))
                    tensor = tensor_records[name]
                    if (
                        value.shape != shape
                        or value.dtype != dtype
                        or _sha256_array(value) != tensor["sha256"]
                    ):
                        raise RuntimeError(
                            f"sealed source tensor drifted: {owner}:{name}"
                        )
                    arrays[(owner, layer, suffix)] = value
                    owner_evidence["tensors"][name] = tensor["sha256"]
        evidence.append(owner_evidence)
    return arrays, evidence


def _model_rank_values(
    sources: Mapping[tuple[int, int, str], np.ndarray],
    *,
    layer: int,
    suffix: str,
    ranks: tuple[int, ...],
) -> np.ndarray:
    return np.ascontiguousarray(
        np.stack(
            tuple(
                sources[(rank // 8, layer, suffix)][rank % 8]
                for rank in ranks
            ),
            axis=0,
        )
    )


def _target_arrays(
    sources: Mapping[tuple[int, int, str], np.ndarray],
    *,
    layer: int,
    expert: int,
    feature: int,
) -> tuple[tuple[int, ...], dict[str, np.ndarray]]:
    if layer not in LAYERS or not 0 <= expert < 8 or not 0 <= feature < 4:
        raise ValueError("WS32 StrategyND target coordinate is invalid")
    ranks = tuple(range(expert * 4, expert * 4 + 4))
    values = tuple(
        _model_rank_values(
            sources,
            layer=layer,
            suffix=suffix,
            ranks=ranks,
        )
        for suffix in SOURCE_SUFFIXES
    )
    start = feature * 1536
    stop = start + 1536
    names = strategy_nd_dense_tensor_names(layer)
    arrays = {
        names[0]: values[0],
        names[1]: values[1],
        names[2]: np.ascontiguousarray(values[2][..., start:stop]),
        names[3]: np.ascontiguousarray(values[3][..., start:stop]),
    }
    expected = (
        ((4, 6144, 768), np.dtype(np.uint8)),
        ((4, 48, 768), np.dtype(np.float32)),
        ((4, 384, 1536), np.dtype(np.uint8)),
        ((4, 3, 1536), np.dtype(np.float32)),
    )
    if any(
        arrays[name].shape != shape or arrays[name].dtype != dtype
        for name, (shape, dtype) in zip(names, expected, strict=True)
    ):
        raise RuntimeError("WS32 StrategyND target tensor geometry drifted")
    return ranks, arrays


def main() -> int:
    args = _parse_args()
    if REPO != EXPECTED_REPO:
        raise RuntimeError(f"wrong worktree: {REPO}")
    code_hash = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if code_hash != args.expected_code_hash:
        raise RuntimeError("overlay packer code hash drifted")
    if args.output_root.exists():
        raise FileExistsError("dense overlay output is append-only")
    manifest_path = args.source_root / "runtime_manifest.json"
    success_path = args.source_root / "SUCCESS"
    if _sha256_file(manifest_path) != args.source_manifest_file_sha256 or (
        _sha256_file(success_path) != args.source_success_file_sha256
    ):
        raise RuntimeError("sealed source terminal identity drifted")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        source_manifest.get("artifact_kind")
        != "greenfield_feature_runtime_packed_checkpoint"
        or source_manifest.get("plan_id") != "PP8_LP4"
        or source_manifest.get("dense_projection_layout")
        != "virtual_tp32_dense_convolution_in_out_v1"
    ):
        raise RuntimeError("sealed source checkpoint contract drifted")
    sources, source_evidence = _load_sources(
        args.source_root, _source_file_records(source_manifest)
    )

    from safetensors.torch import save_file
    import torch

    args.output_root.mkdir(parents=True)
    files = []
    for layer in LAYERS:
        for expert in range(8):
            for feature in range(4):
                ranks, arrays = _target_arrays(
                    sources,
                    layer=layer,
                    expert=expert,
                    feature=feature,
                )
                relative = Path(
                    f"layer_{layer:02d}/expert_{expert:02d}/"
                    f"feature_{feature:02d}.safetensors"
                )
                destination = args.output_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(
                    f".{destination.name}.tmp.{os.getpid()}"
                )
                save_file(
                    {
                        name: torch.from_numpy(value)
                        for name, value in arrays.items()
                    },
                    temporary,
                    metadata={
                        "code_hash": code_hash,
                        "expert_coordinate": str(expert),
                        "feature_coordinate": str(feature),
                        "layer_id": str(layer),
                        "model_ranks": ",".join(map(str, ranks)),
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
                        "layer_id": layer,
                        "model_ranks": list(ranks),
                        "sha256": _sha256_file(destination),
                        "tensors": {
                            name: {
                                "byte_count": value.nbytes,
                                "dtype": (
                                    "U8" if value.dtype == np.uint8 else "F32"
                                ),
                                "sha256": _sha256_array(value),
                                "shape": list(value.shape),
                            }
                            for name, value in arrays.items()
                        },
                    }
                )
    manifest: dict[str, Any] = {
        "artifact_kind": "greenfield_ws32_strategy_nd_dense_overlay",
        "code_hash": code_hash,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dense_layer_ids": list(LAYERS),
        "file_count": len(files),
        "files": files,
        "format_version": 1,
        "model_id": "zai-org/GLM-5.2-FP8",
        "plan_id": "WS32_2D",
        "source_dense_projection_layout": source_manifest[
            "dense_projection_layout"
        ],
        "source_evidence": source_evidence,
        "source_manifest_file_sha256": args.source_manifest_file_sha256,
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "source_success_file_sha256": args.source_success_file_sha256,
        "total_bytes": sum(item["byte_count"] for item in files),
    }
    manifest["manifest_sha256"] = sha256(_canonical(manifest)).hexdigest()
    _atomic_json(args.output_root / "manifest.json", manifest)
    success: dict[str, Any] = {
        "artifact_kind": "greenfield_ws32_strategy_nd_dense_overlay_success",
        "code_hash": code_hash,
        "manifest_file_sha256": _sha256_file(
            args.output_root / "manifest.json"
        ),
        "manifest_sha256": manifest["manifest_sha256"],
    }
    success["success_sha256"] = sha256(_canonical(success)).hexdigest()
    _atomic_json(args.output_root / "SUCCESS", success)
    print(
        json.dumps(
            {
                "file_count": len(files),
                "manifest_file_sha256": _sha256_file(
                    args.output_root / "manifest.json"
                ),
                "manifest_sha256": manifest["manifest_sha256"],
                "success_file_sha256": _sha256_file(
                    args.output_root / "SUCCESS"
                ),
                "total_bytes": manifest["total_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
