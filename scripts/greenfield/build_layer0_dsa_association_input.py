#!/usr/bin/env python3
"""Build a bounded real-input artifact for the 8K layer-0 DSA probe.

The artifact contains only the unique prompt/current-token embeddings, the
layer-0 normalization/indexer leaves, and the sealed event-0 selection.  It is
diagnostic input, never a production checkpoint or a source of model outputs.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


REPO = Path(__file__).resolve().parents[2]
EXPECTED_WORKTREE = Path("/home/gianl/glm-tpu-topology-rewrite")
EXPECTED_BRANCH = "rewrite/topology-first-decode"
SOURCE_SHARD = "model-00001-of-00141.safetensors"
FORMAT_VERSION = 2

SOURCE_NAMES = (
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.self_attn.q_a_proj.weight",
    "model.layers.0.self_attn.q_a_proj.weight_scale_inv",
    "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
    "model.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale_inv",
    "model.layers.0.self_attn.q_a_layernorm.weight",
    "model.layers.0.self_attn.indexer.wq_b.weight",
    "model.layers.0.self_attn.indexer.wq_b.weight_scale_inv",
    "model.layers.0.self_attn.indexer.wk.weight",
    "model.layers.0.self_attn.indexer.wk.weight_scale_inv",
    "model.layers.0.self_attn.indexer.k_norm.weight",
    "model.layers.0.self_attn.indexer.k_norm.bias",
    "model.layers.0.self_attn.indexer.weights_proj.weight",
)

SOURCE_CONTRACT = {
    "model.embed_tokens.weight": ((154880, 6144), "uint16"),
    "model.layers.0.input_layernorm.weight": ((6144,), "uint16"),
    "model.layers.0.self_attn.q_a_proj.weight": ((2048, 6144), "uint8"),
    "model.layers.0.self_attn.q_a_proj.weight_scale_inv": ((16, 48), "float32"),
    "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": (
        (576, 6144),
        "uint8",
    ),
    "model.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale_inv": (
        (5, 48),
        "float32",
    ),
    "model.layers.0.self_attn.q_a_layernorm.weight": ((2048,), "uint16"),
    "model.layers.0.self_attn.indexer.wq_b.weight": ((4096, 2048), "uint8"),
    "model.layers.0.self_attn.indexer.wq_b.weight_scale_inv": (
        (32, 16),
        "float32",
    ),
    "model.layers.0.self_attn.indexer.wk.weight": ((128, 6144), "uint8"),
    "model.layers.0.self_attn.indexer.wk.weight_scale_inv": (
        (1, 48),
        "float32",
    ),
    "model.layers.0.self_attn.indexer.k_norm.weight": ((128,), "uint16"),
    "model.layers.0.self_attn.indexer.k_norm.bias": ((128,), "uint16"),
    "model.layers.0.self_attn.indexer.weights_proj.weight": (
        (32, 6144),
        "uint16",
    ),
}


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args], text=True
    ).strip()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _tensor_bytes(tensor: Any) -> np.ndarray:
    import torch

    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy().copy()
    if value.dtype == torch.float8_e4m3fn:
        return value.view(torch.uint8).numpy().copy()
    return value.numpy().copy()


def _tensor_record(name: str, value: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(value)
    return {
        "byte_count": int(contiguous.nbytes),
        "dtype": str(contiguous.dtype),
        "name": name,
        "sha256": sha256(contiguous.tobytes(order="C")).hexdigest(),
        "shape": list(contiguous.shape),
    }


def _require_source_contract(source_name: str, value: np.ndarray) -> None:
    expected_shape, expected_dtype = SOURCE_CONTRACT[source_name]
    if value.shape != expected_shape or str(value.dtype) != expected_dtype:
        raise RuntimeError(
            f"source tensor contract drifted for {source_name}: "
            f"expected={expected_shape}/{expected_dtype} "
            f"found={value.shape}/{value.dtype}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--token-oracle-dir", type=Path, required=True)
    parser.add_argument("--token-oracle-sha256", required=True)
    parser.add_argument("--dsa-oracle-dir", type=Path, required=True)
    parser.add_argument("--dsa-oracle-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-code-hash", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if REPO != EXPECTED_WORKTREE:
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    if _git("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError("layer-0 DSA input must use the isolated rewrite branch")
    code_hash = _git("rev-parse", "HEAD")
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale code hash: expected={args.expected_code_hash} found={code_hash}"
        )
    if _git("status", "--porcelain"):
        raise RuntimeError("layer-0 DSA input requires a clean worktree")
    if args.output.exists():
        raise FileExistsError(f"append-only output exists: {args.output}")

    sys.path.insert(0, str(REPO))
    from glm_tpu.greenfield.validation import (
        inspect_short_context_dsa_oracle,
        inspect_short_context_oracle,
    )

    token_manifest = inspect_short_context_oracle(args.token_oracle_dir)
    dsa_manifest = inspect_short_context_dsa_oracle(args.dsa_oracle_dir)
    if token_manifest["manifest_sha256"] != args.token_oracle_sha256:
        raise RuntimeError("sealed 8K token-oracle hash drifted")
    if dsa_manifest["manifest_sha256"] != args.dsa_oracle_sha256:
        raise RuntimeError("sealed 8K DSA-oracle hash drifted")
    if dsa_manifest["token_oracle"]["manifest_sha256"] != (
        token_manifest["manifest_sha256"]
    ):
        raise RuntimeError("paired token/DSA oracle identity drifted")

    with safe_open(
        args.token_oracle_dir / token_manifest["files"]["tokens"]["filename"],
        framework="np",
    ) as handle:
        prompt_ids = handle.get_tensor("prompt_token_ids").astype(
            np.int32, copy=True
        )
        generated_ids = handle.get_tensor("generated_token_ids").astype(
            np.int32, copy=True
        )
    if prompt_ids.shape != (8155,) or generated_ids.shape != (20,):
        raise RuntimeError("the paired 8K token geometry drifted")

    with safe_open(
        args.dsa_oracle_dir / dsa_manifest["files"]["tensors"]["filename"],
        framework="np",
    ) as handle:
        expected_positions = handle.get_tensor("selected_positions")[0, 0].astype(
            np.int32, copy=True
        )
        expected_scores = handle.get_tensor("selected_scores")[0, 0].astype(
            np.float32, copy=True
        )
        expected_count = handle.get_tensor("valid_counts")[0, 0:1].astype(
            np.int32, copy=True
        )
        decode_position = handle.get_tensor("decode_positions")[0:1].astype(
            np.int32, copy=True
        )
        producer = handle.get_tensor("producer_layer_ids")[0:1].astype(
            np.int32, copy=True
        )
    if (
        expected_positions.shape != (2048,)
        or expected_scores.shape != (2048,)
        or expected_count.tolist() != [2048]
        or decode_position.tolist() != [8155]
        or producer.tolist() != [0]
    ):
        raise RuntimeError("sealed event-0 DSA geometry drifted")

    source_path = args.source_root / SOURCE_SHARD
    source_index_path = args.source_root / "model.safetensors.index.json"
    index = json.loads(source_index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("source checkpoint index has no weight map")
    required = ("model.embed_tokens.weight", *SOURCE_NAMES)
    escaped = {
        name: weight_map.get(name)
        for name in required
        if weight_map.get(name) != SOURCE_SHARD
    }
    if escaped:
        raise RuntimeError(f"layer-0 DSA inputs escaped source shard 1: {escaped}")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    unique_ids = np.unique(
        np.concatenate((prompt_ids, generated_ids[0:1])).astype(np.int32)
    )
    arrays: dict[str, np.ndarray] = {
        "current_token_id": generated_ids[0:1],
        "decode_position": decode_position,
        "expected_selected_count": expected_count,
        "expected_selected_positions": expected_positions,
        "expected_selected_scores": expected_scores,
        "prompt_token_ids": prompt_ids,
        "unique_token_ids": unique_ids,
    }
    source_records = []
    with safe_open(source_path, framework="pt", device="cpu") as handle:
        embedding_slice = handle.get_slice("model.embed_tokens.weight")
        embedding_rows = np.concatenate(
            [
                _tensor_bytes(embedding_slice[int(token) : int(token) + 1])
                for token in unique_ids
            ],
            axis=0,
        )
        if embedding_rows.shape != (unique_ids.size, 6144) or (
            embedding_rows.dtype != np.uint16
        ):
            raise RuntimeError("selected embedding-row contract drifted")
        arrays["unique_embedding_bfloat16_bits"] = embedding_rows
        source_records.append(
            {
                **_tensor_record("selected:model.embed_tokens.weight", embedding_rows),
                "source_name": "model.embed_tokens.weight",
                "selected_token_ids": unique_ids.tolist(),
            }
        )
        for source_name in SOURCE_NAMES:
            value = _tensor_bytes(handle.get_tensor(source_name))
            _require_source_contract(source_name, value)
            artifact_name = source_name.removeprefix("model.layers.0.").replace(
                ".", "__"
            )
            arrays[artifact_name] = value
            source_records.append(
                {
                    **_tensor_record(artifact_name, value),
                    "source_name": source_name,
                }
            )

    args.output.mkdir(parents=True)
    artifact_path = args.output / "layer0_dsa_input.safetensors"
    save_file(
        {name: np.ascontiguousarray(value) for name, value in arrays.items()},
        artifact_path,
        metadata={
            "artifact_kind": "greenfield_layer0_dsa_association_input",
            "format_version": str(FORMAT_VERSION),
        },
    )
    artifact_record = {
        "byte_count": artifact_path.stat().st_size,
        "filename": artifact_path.name,
        "sha256": _sha256_file(artifact_path),
    }
    manifest: dict[str, Any] = {
        "artifact_kind": "greenfield_layer0_dsa_association_input",
        "arrays": {
            name: _tensor_record(name, value)
            for name, value in sorted(arrays.items())
        },
        "code_hash": code_hash,
        "diagnostic_only": True,
        "dsa_oracle_manifest_sha256": dsa_manifest["manifest_sha256"],
        "event_index": 0,
        "file": artifact_record,
        "format_version": FORMAT_VERSION,
        "model_id": "zai-org/GLM-5.2-FP8",
        "producer_layer_id": 0,
        "source_records": source_records,
        "source_shard": {
            "byte_count": source_path.stat().st_size,
            "filename": SOURCE_SHARD,
        },
        "source_index": {
            "byte_count": source_index_path.stat().st_size,
            "filename": source_index_path.name,
            "sha256": _sha256_file(source_index_path),
        },
        "token_oracle_manifest_sha256": token_manifest["manifest_sha256"],
        "unique_token_count": int(unique_ids.size),
    }
    manifest["manifest_sha256"] = sha256(_canonical_json(manifest)).hexdigest()
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
