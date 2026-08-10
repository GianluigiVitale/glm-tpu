#!/usr/bin/env python3
"""Discriminate layer-0 DSA score precision on exact TPU inputs."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _array_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _score_delta(
    scores: np.ndarray,
    positions: np.ndarray,
    expected_scores: np.ndarray,
) -> dict[str, Any]:
    aligned = np.asarray(scores, dtype=np.float32)[
        np.asarray(positions, dtype=np.int32)
    ]
    expected = np.asarray(expected_scores, dtype=np.float32)
    difference = np.abs(aligned - expected)
    return {
        "actual_sha256": _array_sha256(aligned),
        "elementwise_exact": bool(np.array_equal(aligned, expected)),
        "expected_sha256": _array_sha256(expected),
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()),
        "mismatch_count": int(np.count_nonzero(aligned != expected)),
        "p99_abs": float(np.percentile(difference, 99)),
        "shape": list(aligned.shape),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--association-input-dir", type=Path, required=True)
    parser.add_argument("--association-input-manifest-sha256", required=True)
    parser.add_argument("--prompt-cache-dir", type=Path, required=True)
    parser.add_argument("--prompt-cache-manifest-sha256", required=True)
    parser.add_argument("--internal-observer-dir", type=Path, required=True)
    parser.add_argument("--internal-contract-sha256", required=True)
    parser.add_argument("--internal-tensor-sha256", required=True)
    parser.add_argument("--selected-observation", type=Path, required=True)
    parser.add_argument("--selected-observation-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
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

    import jax
    import ml_dtypes

    from glm_tpu.greenfield.benchmarking.dsa_association import (
        validate_dsa_association_hlo,
    )
    from glm_tpu.greenfield.kernels.reference.dsa import dsa_scores
    from glm_tpu.greenfield.validation.layer0_dsa_association import (
        compare_dsa_association_scores,
        inspect_greenfield_layer0_dsa_internal_observation,
        inspect_greenfield_layer0_dsa_selected_observation,
        inspect_layer0_dsa_association_input,
        pack_stage_local_index_keys,
        stitch_stage_local_scores,
    )
    from glm_tpu.greenfield.validation.prompt_index_cache import (
        inspect_legacy_prompt_index_cache,
    )

    if jax.default_backend() != "tpu":
        raise RuntimeError(
            f"DSA scorer discriminator requires TPU, got {jax.default_backend()}"
        )
    if jax.local_device_count() != 4:
        raise RuntimeError(
            "DSA scorer discriminator requires one four-chip TPU host, got "
            f"{jax.local_device_count()}"
        )
    device = jax.local_devices()[0]
    association_manifest, association = inspect_layer0_dsa_association_input(
        args.association_input_dir,
        expected_manifest_sha256=args.association_input_manifest_sha256,
    )
    cache_manifest, prompt_bits = inspect_legacy_prompt_index_cache(
        args.prompt_cache_dir,
        expected_manifest_sha256=args.prompt_cache_manifest_sha256,
    )
    internal_contract, internals = (
        inspect_greenfield_layer0_dsa_internal_observation(
            args.internal_observer_dir,
            expected_contract_sha256=args.internal_contract_sha256,
            expected_tensor_sha256=args.internal_tensor_sha256,
        )
    )
    current_positions, current_scores = (
        inspect_greenfield_layer0_dsa_selected_observation(
            args.selected_observation,
            expected_sha256=args.selected_observation_sha256,
        )
    )
    query_host = np.ascontiguousarray(internals["query"][0:1])
    head_weights_host = np.ascontiguousarray(internals["head_weights"][0:1])
    current_key_host = np.ascontiguousarray(internals["current_key"][0])
    global_bits, lane_bits, lane_positions = pack_stage_local_index_keys(
        prompt_bits,
        current_key_host,
        logical_page_size=512,
        local_parallel_size=4,
    )
    if global_bits.shape != (8156, 128) or lane_bits.shape != (4, 2048, 128):
        raise RuntimeError("sealed LP4 score discriminator geometry drifted")

    query = jax.device_put(query_host, device)
    head_weights = jax.device_put(head_weights_host, device)
    first_lane = jax.device_put(
        lane_bits[0].view(ml_dtypes.bfloat16), device
    )
    args.hlo_dir.mkdir(parents=True, exist_ok=True)

    def compile_scorer(
        name: str,
        scorer: Callable[[Any, Any, Any], Any],
        *,
        phase: str,
    ) -> tuple[Any, dict[str, Any], float]:
        started = time.monotonic()
        compiled = jax.jit(scorer).lower(
            query, first_lane, head_weights
        ).compile()
        compile_seconds = time.monotonic() - started
        hlo = compiled.as_text()
        hlo_path = args.hlo_dir / f"{name}.optimized_hlo.txt"
        hlo_path.write_text(hlo)
        contract = validate_dsa_association_hlo(
            hlo,
            phase=phase,  # type: ignore[arg-type]
            context=2048,
        )
        if not contract["passed"]:
            raise RuntimeError(f"{name} HLO contract failed: {contract}")
        return compiled, {
            "filename": hlo_path.name,
            "sha256": sha256(hlo.encode()).hexdigest(),
            "contract": contract,
        }, compile_seconds

    wide_compiled, wide_hlo, wide_compile_seconds = compile_scorer(
        "current_wide_score",
        dsa_scores,
        phase="local_wide_score",
    )
    default_compiled, default_hlo, default_compile_seconds = compile_scorer(
        "default_wide_score",
        lambda q, k, h: dsa_scores(q, k, h, precision="default"),
        phase="local_wide_default_score",
    )

    def execute_lanes(compiled: Any, *, wide: bool) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for bits in lane_bits:
            keys = jax.device_put(bits.view(ml_dtypes.bfloat16), device)
            value = compiled(query, keys, head_weights)
            jax.block_until_ready(value)
            host = np.asarray(value, dtype=np.float32)
            outputs.append(host[0] if wide else host)
        return np.stack(outputs)

    wide_lanes = execute_lanes(wide_compiled, wide=True)
    default_lanes = execute_lanes(default_compiled, wide=True)
    wide_scores = stitch_stage_local_scores(
        wide_lanes, lane_positions, context=8156
    )
    default_scores = stitch_stage_local_scores(
        default_lanes, lane_positions, context=8156
    )
    expected_positions = association["expected_selected_positions"]
    expected_scores = association["expected_selected_scores"]
    wide_control = compare_dsa_association_scores(
        wide_scores, current_positions, current_scores
    )
    wide_control_delta = _score_delta(
        wide_scores, current_positions, current_scores
    )
    wide_control_bitwise = (
        wide_control_delta["actual_sha256"]
        == wide_control_delta["expected_sha256"]
    )
    if not wide_control["passed"] or not wide_control_bitwise:
        raise RuntimeError(
            "current-wide scorer failed to reproduce the protected observation: "
            f"comparison={wide_control} delta={wide_control_delta}"
        )
    accepted_comparison = compare_dsa_association_scores(
        default_scores, expected_positions, expected_scores
    )
    accepted_score_delta = _score_delta(
        default_scores, expected_positions, expected_scores
    )
    candidate_restored = bool(
        accepted_comparison["passed"]
        and accepted_score_delta["actual_sha256"]
        == accepted_score_delta["expected_sha256"]
    )
    record = {
        "status": "SUCCESS",
        "code_hash": code_hash,
        "backend": jax.default_backend(),
        "device": str(device),
        "device_kind": device.device_kind,
        "claim_scope": (
            "bounded layer-0 same-shape highest-versus-default dot-precision "
            "diagnostic; no decoder, Gate-D, latency, or token-rate claim"
        ),
        "profiler_free_timing": False,
        "inputs": {
            "association_manifest_sha256": association_manifest[
                "manifest_sha256"
            ],
            "prompt_cache_manifest_sha256": cache_manifest["manifest_sha256"],
            "prompt_cache_bfloat16_sha256": cache_manifest[
                "prompt_index_key_bfloat16_sha256"
            ],
            "internal_contract_sha256": args.internal_contract_sha256,
            "internal_tensor_sha256": args.internal_tensor_sha256,
            "selected_observation_sha256": args.selected_observation_sha256,
            "layer0_reference": internal_contract["layer0_reference"],
        },
        "packing": {
            "logical_page_size": 512,
            "local_parallel_size": 4,
            "local_rows_per_page": 128,
            "logical_context": 8156,
            "lane_width": 2048,
            "global_bfloat16_sha256": _array_sha256(global_bits),
            "lane_bfloat16_sha256": _array_sha256(lane_bits),
            "lane_positions_sha256": _array_sha256(lane_positions),
        },
        "hlo": {
            "current_wide_score": wide_hlo,
            "default_wide_score": default_hlo,
        },
        "compile_seconds": {
            "current_wide_score": wide_compile_seconds,
            "default_wide_score": default_compile_seconds,
        },
        "current_wide_control": {
            "comparison": wide_control,
            "score_delta": wide_control_delta,
            "bitwise_exact": wide_control_bitwise,
            "logical_score_sha256": _array_sha256(wide_scores),
        },
        "default_precision_candidate": {
            "comparison": accepted_comparison,
            "score_delta": accepted_score_delta,
            "logical_score_sha256": _array_sha256(default_scores),
        },
        "candidate_restored": candidate_restored,
        "memory_stats": _memory_stats(device),
    }
    _atomic_json(args.output, record)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
