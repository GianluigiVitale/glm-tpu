#!/usr/bin/env python3
"""CPU-only comparison of a sealed legacy layer prompt index cache with greenfield rows.

Inputs are sealed bytes only: a legacy prompt index cache artifact produced by
``capture_legacy_prompt_index_cache.py`` (any full-indexer layer, verified by
``inspect_legacy_prompt_index_cache``) and the greenfield prompt index cache of
the same layer captured by the executed DB518 numerical run.  It reports the
exact BF16 delta (``compare_prompt_index_key_bits``) plus a row-level map:
mismatched positions, their distribution over the legacy 2,048-token prefill
chunks and over 512-row logical pages, and the first/last mismatched position.

When the legacy layer-1 DSA internals are available it also recomputes event 1
from the legacy query, head weights and current key over the *legacy* cache
under the TPU default-precision score emulation; that selection must equal the
sealed oracle's event-1 set, which authenticates the capture independently of
its manifest.  CPU evidence only: no Gate-D, decoder, DB or performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DIAGNOSE = Path(__file__).resolve().parent / "diagnose_event1_prompt_index_cache_offline.py"
LEGACY_CHUNK_TOKENS = 2048
ARTIFACT_KIND = "gate_d_layer_prompt_index_cache_offline_comparison"


def _load_diagnose_module():
    spec = importlib.util.spec_from_file_location("diagnose_event1_prompt_index_cache_offline", DIAGNOSE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def row_mismatch_map(expected_bits: np.ndarray, observed_bits: np.ndarray) -> dict[str, Any]:
    """Describe where two ``[rows, lanes]`` BF16 caches differ, row by row."""

    expected = np.asarray(expected_bits)
    observed = np.asarray(observed_bits)
    if expected.shape != observed.shape or expected.ndim != 2:
        raise ValueError("prompt cache row map requires equal [rows, lanes] shapes")
    if expected.dtype != np.uint16 or observed.dtype != np.uint16:
        raise ValueError("prompt cache row map requires BF16 bits")
    rows = expected.shape[0]
    row_mismatch = np.any(expected != observed, axis=1)
    mismatched = np.flatnonzero(row_mismatch)
    lanes_per_row = np.count_nonzero(expected != observed, axis=1)
    chunk_ids = np.arange(rows) // LEGACY_CHUNK_TOKENS
    chunk_count = int(chunk_ids.max()) + 1 if rows else 0
    per_chunk = [
        {
            "chunk": chunk,
            "rows": int(np.count_nonzero(chunk_ids == chunk)),
            "mismatched_rows": int(np.count_nonzero(row_mismatch[chunk_ids == chunk])),
        }
        for chunk in range(chunk_count)
    ]
    page_ids = np.arange(rows) // 512
    per_page = [
        int(np.count_nonzero(row_mismatch[page_ids == page]))
        for page in range(int(page_ids.max()) + 1 if rows else 0)
    ]
    return {
        "rows": int(rows),
        "mismatched_row_count": int(mismatched.size),
        "mismatched_row_fraction": float(mismatched.size / rows) if rows else 0.0,
        "first_mismatched_position": int(mismatched[0]) if mismatched.size else None,
        "last_mismatched_position": int(mismatched[-1]) if mismatched.size else None,
        "mismatched_positions_first_64": mismatched[:64].tolist(),
        "mismatched_lanes_per_row_max": int(lanes_per_row.max(initial=0)),
        "mismatched_lanes_per_row_mean_over_mismatched": (
            float(lanes_per_row[mismatched].mean()) if mismatched.size else 0.0
        ),
        "per_legacy_prefill_chunk": per_chunk,
        "mismatched_rows_per_512_page": per_page,
        "chunk_boundary_rows_only": bool(
            mismatched.size
            and np.all((mismatched % LEGACY_CHUNK_TOKENS == 0) | ((mismatched + 1) % LEGACY_CHUNK_TOKENS == 0))
        ),
    }


def compare(
    *,
    legacy_artifact_dir: Path,
    layer_id: int,
    greenfield_capture: Path,
    expected_manifest_sha256: str | None,
    with_event1_check: bool,
) -> dict[str, Any]:
    from glm_tpu.greenfield.validation.prompt_index_cache import (
        compare_prompt_index_key_bits,
        inspect_legacy_prompt_index_cache,
    )

    diagnose = _load_diagnose_module()
    manifest, legacy_bits = inspect_legacy_prompt_index_cache(
        legacy_artifact_dir, expected_manifest_sha256=expected_manifest_sha256
    )
    if int(manifest.get("layer_id", 0)) != layer_id:
        raise ValueError("legacy prompt cache artifact layer differs from the requested layer")
    capture_sha = diagnose.sha256_file(greenfield_capture)
    if greenfield_capture.resolve() == diagnose.DB518_RESULT.resolve() and (
        capture_sha != diagnose.DB518_RESULT_SHA256
    ):
        raise RuntimeError("DB518 result.npz differs from the sealed capture")
    capture = np.load(greenfield_capture)
    greenfield_bits = diagnose.prompt_cache_rows(
        capture[f"layer{layer_id}_index_cache_owners_bfloat16_bits"], rows=legacy_bits.shape[0]
    )
    delta = compare_prompt_index_key_bits(legacy_bits, greenfield_bits)
    row_map = row_mismatch_map(legacy_bits, greenfield_bits)
    result: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "layer_id": layer_id,
        "legacy_artifact": {
            "dir": str(legacy_artifact_dir),
            "manifest_sha256": manifest["manifest_sha256"],
            "prompt_index_key_bfloat16_sha256": manifest["prompt_index_key_bfloat16_sha256"],
        },
        "greenfield_capture": {"path": str(greenfield_capture), "sha256": capture_sha},
        "bitwise": delta,
        "row_map": row_map,
        "performance_claim": False,
    }
    if with_event1_check and layer_id == 1:
        internals_path = diagnose.legacy_internals_path(1)
        internals = np.load(internals_path)
        oracle_positions, _ = diagnose.load_oracle_events()
        keys = np.concatenate(
            [
                diagnose.bf16_bits_to_f32(legacy_bits),
                diagnose.bf16_round(internals["current_key"].astype(np.float32))[None, :],
            ],
            axis=0,
        )
        scores = diagnose.default_precision_dsa_scores(
            internals["query"].astype(np.float32), keys, internals["head_weights"].astype(np.float32)
        )
        oracle_set = set(oracle_positions[1].tolist())
        recomputed = diagnose.top_set(scores, len(oracle_set))
        result["event1_over_legacy_cache_vs_oracle"] = {
            "legacy_internals_sha256": diagnose.sha256_file(internals_path),
            "expected_only": sorted(oracle_set - recomputed),
            "observed_only": sorted(recomputed - oracle_set),
        }
    exact = delta["elementwise_exact"]
    event1 = result.get("event1_over_legacy_cache_vs_oracle")
    parts = [
        f"LAYER{layer_id}_PROMPT_INDEX_CACHE_"
        + ("BITWISE_EXACT" if exact else f"{row_map['mismatched_row_count']}_OF_{row_map['rows']}_ROWS_MISMATCH"),
    ]
    if event1 is not None:
        parts.append(
            "LEGACY_CACHE_REPRODUCES_ORACLE_EVENT1"
            if not event1["expected_only"] and not event1["observed_only"]
            else "LEGACY_CACHE_DOES_NOT_REPRODUCE_ORACLE_EVENT1"
        )
    parts.append("CPU_EVIDENCE_ONLY;NO_GATE_D_NO_DECODER_NO_DB_NO_PERFORMANCE_CLAIM;GATE_D_OPEN")
    result["classification"] = ";".join(parts)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--legacy-artifact-dir", type=Path, required=True)
    parser.add_argument("--layer-id", type=int, default=1)
    parser.add_argument("--expected-manifest-sha256", default=None)
    parser.add_argument("--greenfield-capture", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-event1-check", action="store_true")
    args = parser.parse_args()
    diagnose = _load_diagnose_module()
    capture = args.greenfield_capture or diagnose.DB518_RESULT
    result = compare(
        legacy_artifact_dir=args.legacy_artifact_dir,
        layer_id=args.layer_id,
        greenfield_capture=capture,
        expected_manifest_sha256=args.expected_manifest_sha256,
        with_event1_check=not args.skip_event1_check,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["classification"])
    print(json.dumps({k: result["row_map"][k] for k in ("mismatched_row_count", "first_mismatched_position", "per_legacy_prefill_chunk")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
