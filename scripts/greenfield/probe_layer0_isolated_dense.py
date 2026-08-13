#!/usr/bin/env python3
"""Replay layer-0 dense contractions one virtual rank per physical chip."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import ml_dtypes
import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.greenfield.probe_layer0_captured_rms import (  # noqa: E402
    _ACCEPTED_LAYER1_SHA,
    _CAPTURE_RESIDUAL_SHA,
    _DB548_OBSERVED_SHA,
    _array_sha256,
    _build_arm,
    _load_sources,
    _validate_captured_rms_optimized_hlo,
)
from scripts.greenfield.probe_layer0_dense_convolution import (  # noqa: E402
    _FINAL_DENSE_LAYOUT_RECORDS,
    _pack_dense_final_layout,
    _validate_optimized_hlo,
)
from scripts.greenfield.probe_layer0_projection_reduction import (  # noqa: E402
    _compare_bits,
    _load_weights,
)


_POSITION = 8155


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _validate_isolated_optimized_hlo(optimized_hlo: str) -> dict[str, Any]:
    """Reuse the final-layout matcher for one live virtual contraction."""

    return _validate_optimized_hlo(
        optimized_hlo,
        compile_rows=32,
        final_dense_layout=True,
        dense_envelope=True,
        isolated_dense=True,
    )


def _build_isolated_partial(mesh: Any) -> Any:
    import jax
    from jax import lax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    from glm_tpu.greenfield.kernels.reference.rmsnorm import fused_add_rms_norm
    from glm_tpu.greenfield.kernels.stage_local import (
        _virtual_dense_final_layout_convolution_down_partials,
    )

    def local(
        attention: Any,
        residual: Any,
        post_norm: Any,
        merged_bits: Any,
        merged_scale: Any,
        down_bits: Any,
        down_scale: Any,
    ) -> tuple[Any, Any]:
        with jax.named_scope("greenfield_isolated_dense_predense_m32"):
            attention_m32 = jnp.pad(
                attention,
                ((0, 31), (0, 0)),
                constant_values=jnp.bfloat16(0),
            )
            residual_m32 = jnp.pad(
                residual,
                ((0, 31), (0, 0)),
                constant_values=jnp.bfloat16(0),
            )
        with jax.named_scope("greenfield_isolated_dense_predense_rmsnorm"):
            normalized, carried = fused_add_rms_norm(
                attention_m32,
                residual_m32,
                post_norm,
                epsilon=1e-5,
            )
        with jax.named_scope("greenfield_isolated_dense_contraction"):
            partials = _virtual_dense_final_layout_convolution_down_partials(
                normalized,
                merged_bits[0],
                merged_scale[0],
                down_bits[0],
                down_scale[0],
                block_shape=(128, 128),
                compile_rows=32,
                virtual_shards=1,
            )
        with jax.named_scope("greenfield_isolated_dense_live_row"):
            partials = lax.optimization_barrier(partials)
            live = partials[:, :1, :]
        return live, carried

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(
            P(),
            P(),
            P(),
            P("lp4", None, None, None),
            P("lp4", None, None, None),
            P("lp4", None, None, None),
            P("lp4", None, None, None),
        ),
        out_specs=(P("lp4", None, None), P()),
        check_vma=False,
    )


def _partial_comparison(
    reference_bits: np.ndarray, observed_bits: np.ndarray
) -> dict[str, Any]:
    if reference_bits.shape != observed_bits.shape or reference_bits.dtype != np.uint16:
        raise RuntimeError("isolated partial comparison geometry drifted")
    mismatch = reference_bits != observed_bits
    per_rank = []
    for rank in range(8):
        rank_mismatch = mismatch[:, rank]
        per_rank.append(
            {
                "mismatch_count": int(np.count_nonzero(rank_mismatch)),
                "virtual_rank": rank,
            }
        )
    first = np.argwhere(mismatch)
    return {
        "elementwise_exact": not bool(first.size),
        "first_mismatch_index": (
            None if not first.size else [int(value) for value in first[0]]
        ),
        "mismatch_count": int(np.count_nonzero(mismatch)),
        "per_virtual_rank": per_rank,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-code-hash", required=True)
    for prefix in ("capture", "db548"):
        parser.add_argument(f"--{prefix}-tensor", type=Path, required=True)
        parser.add_argument(f"--{prefix}-tensor-sha256", required=True)
        parser.add_argument(f"--{prefix}-runner", type=Path, required=True)
        parser.add_argument(f"--{prefix}-runner-sha256", required=True)
        parser.add_argument(f"--{prefix}-summary", type=Path, required=True)
        parser.add_argument(f"--{prefix}-summary-sha256", required=True)
        parser.add_argument(f"--{prefix}-success", type=Path, required=True)
        parser.add_argument(f"--{prefix}-success-sha256", required=True)
    parser.add_argument("--db548-hlo", type=Path, required=True)
    parser.add_argument("--db548-hlo-sha256", required=True)
    for suffix in ("tensor", "runner", "summary", "success", "hlo"):
        parser.add_argument(f"--db549-{suffix}", type=Path, required=True)
        parser.add_argument(f"--db549-{suffix}-sha256", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
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
    (
        captured_partial_bits,
        attention_bits,
        residual_bits,
        norm_bits,
        accepted_bits,
        db548_bits,
    ) = _load_sources(args)
    weights, _weight_records = _load_weights(
        args.checkpoint_root,
        manifest_sha256=args.checkpoint_manifest_sha256,
    )
    packed, packed_records = _pack_dense_final_layout(weights)
    if packed_records != _FINAL_DENSE_LAYOUT_RECORDS:
        raise RuntimeError("isolated dense final-layout records drifted")

    import jax
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from glm_tpu.greenfield.sharding.stablehlo_dense_convolution import (
        validate_isolated_dense_partial_stablehlo,
    )
    from glm_tpu.greenfield.sharding.stablehlo_dense_convolution import (
        validate_captured_dense_rms_stablehlo,
    )

    if jax.default_backend() != "tpu" or jax.local_device_count() != 4:
        raise RuntimeError("isolated dense replay requires one four-chip TPU host")
    mesh = Mesh(np.asarray(jax.local_devices()), ("lp4",))
    replicated = NamedSharding(mesh, P())
    slot = NamedSharding(mesh, P("lp4", None, None, None))
    attention = jax.device_put(attention_bits.view(ml_dtypes.bfloat16), replicated)
    residual = jax.device_put(residual_bits.view(ml_dtypes.bfloat16), replicated)
    post_norm = jax.device_put(weights["attention.slot_00.post_norm"], replicated)
    rank_arguments = [
        (
            attention,
            residual,
            post_norm,
            *(jax.device_put(value[:, rank : rank + 1], slot) for value in packed),
        )
        for rank in range(8)
    ]
    args.hlo_dir.mkdir(parents=True, exist_ok=True)
    mapped = _build_isolated_partial(mesh)
    lowered = jax.jit(mapped).lower(*rank_arguments[0])
    stablehlo = lowered.as_text()
    stable_contract = validate_isolated_dense_partial_stablehlo(stablehlo)
    stable_path = args.hlo_dir / "isolated_dense.stablehlo.mlir"
    stable_path.write_text(stablehlo)
    if not stable_contract["passed"]:
        raise RuntimeError(f"isolated dense StableHLO failed: {stable_contract}")
    compiled = lowered.compile()
    optimized_hlo = compiled.as_text()
    optimized_contract = _validate_isolated_optimized_hlo(optimized_hlo)
    optimized_path = args.hlo_dir / "isolated_dense.optimized_hlo.txt"
    optimized_path.write_text(optimized_hlo)
    if not optimized_contract["passed"]:
        raise RuntimeError(f"isolated dense optimized HLO failed: {optimized_contract}")

    isolated_by_rank = []
    carried_by_rank = []
    for values in rank_arguments:
        partial, carried = compiled(*values)
        jax.block_until_ready((partial, carried))
        isolated_by_rank.append(np.ascontiguousarray(np.asarray(partial)))
        carried_by_rank.append(np.ascontiguousarray(np.asarray(carried)))
    isolated_partial_bits = np.stack(isolated_by_rank, axis=1).view(np.uint16)
    carried_bits = np.stack(carried_by_rank).view(np.uint16)
    if isolated_partial_bits.shape != (4, 8, 1, 6144):
        raise RuntimeError("isolated dense partial output geometry drifted")
    if carried_bits.shape != (8, 32, 6144) or any(
        _array_sha256(value) != _CAPTURE_RESIDUAL_SHA for value in carried_bits
    ):
        raise RuntimeError("isolated dense carried residual drifted")

    rms_mapped = _build_arm(mesh, split_layer1_rms=False)
    partial_sharding = NamedSharding(mesh, P("lp4", None, None, None))
    rms_arguments = (
        jax.device_put(
            captured_partial_bits.view(ml_dtypes.bfloat16), partial_sharding
        ),
        attention,
        residual,
        jax.device_put(norm_bits.view(ml_dtypes.bfloat16), replicated),
    )
    rms_lowered = jax.jit(rms_mapped).lower(*rms_arguments)
    rms_stablehlo = rms_lowered.as_text()
    rms_stable_contract = validate_captured_dense_rms_stablehlo(
        rms_stablehlo, split_layer1_rms=False
    )
    rms_stable_path = args.hlo_dir / "layer1_replay.stablehlo.mlir"
    rms_stable_path.write_text(rms_stablehlo)
    if not rms_stable_contract["passed"]:
        raise RuntimeError(f"isolated layer1 StableHLO failed: {rms_stable_contract}")
    rms_compiled = rms_lowered.compile()
    rms_optimized_hlo = rms_compiled.as_text()
    rms_optimized_contract = _validate_captured_rms_optimized_hlo(
        rms_optimized_hlo, split_layer1_rms=False
    )
    rms_optimized_path = args.hlo_dir / "layer1_replay.optimized_hlo.txt"
    rms_optimized_path.write_text(rms_optimized_hlo)
    if not rms_optimized_contract["passed"]:
        raise RuntimeError(
            f"isolated layer1 optimized HLO failed: {rms_optimized_contract}"
        )
    control = rms_compiled(*rms_arguments)
    isolated = rms_compiled(
        jax.device_put(
            isolated_partial_bits.view(ml_dtypes.bfloat16), partial_sharding
        ),
        *rms_arguments[1:],
    )
    jax.block_until_ready((control, isolated))
    control_bits = np.ascontiguousarray(np.asarray(control).reshape(6144)).view(
        np.uint16
    )
    isolated_bits = np.ascontiguousarray(
        np.asarray(isolated).reshape(6144)
    ).view(np.uint16)
    control_admissible = bool(
        _array_sha256(control_bits) == _DB548_OBSERVED_SHA
        and _compare_bits(db548_bits, control_bits)["elementwise_exact"]
    )
    if not control_admissible:
        raise RuntimeError("isolated dense replay control drifted from DB548")
    exact = bool(
        _array_sha256(isolated_bits) == _ACCEPTED_LAYER1_SHA
        and _compare_bits(accepted_bits, isolated_bits)["elementwise_exact"]
    )
    partial_comparison = _partial_comparison(
        captured_partial_bits, isolated_partial_bits
    )
    classification = (
        "isolated_virtual_contractions_exact"
        if exact
        else "isolated_virtual_contractions_nonexact"
    )
    result = {
        "artifact_kind": "glm52_layer0_isolated_dense_replay",
        "classification": classification,
        "code_hash": code_hash,
        "control_admissible": control_admissible,
        "exact": exact,
        "exact_arms": ["isolated_virtual_contractions"] if exact else [],
        "final_layout_records": packed_records,
        "hlo": {
            "isolated_dense": {
                "optimized_contract": optimized_contract,
                "optimized_sha256": sha256(optimized_hlo.encode()).hexdigest(),
                "stablehlo_contract": stable_contract,
                "stablehlo_sha256": sha256(stablehlo.encode()).hexdigest(),
            },
            "layer1_replay": {
                "optimized_contract": rms_optimized_contract,
                "optimized_sha256": sha256(rms_optimized_hlo.encode()).hexdigest(),
                "stablehlo_contract": rms_stable_contract,
                "stablehlo_sha256": sha256(rms_stablehlo.encode()).hexdigest(),
            },
        },
        "isolated_layer1_comparison": _compare_bits(accepted_bits, isolated_bits),
        "isolated_layer1_sha256": _array_sha256(isolated_bits),
        "partial_comparison": partial_comparison,
        "performance_claim": False,
        "position": _POSITION,
        "source": {
            "capture_runner_sha256": args.capture_runner_sha256,
            "capture_summary_sha256": args.capture_summary_sha256,
            "capture_success_sha256": args.capture_success_sha256,
            "capture_tensor_sha256": args.capture_tensor_sha256,
            "checkpoint_manifest_sha256": args.checkpoint_manifest_sha256,
            "db548_hlo_sha256": args.db548_hlo_sha256,
            "db548_runner_sha256": args.db548_runner_sha256,
            "db548_summary_sha256": args.db548_summary_sha256,
            "db548_success_sha256": args.db548_success_sha256,
            "db548_tensor_sha256": args.db548_tensor_sha256,
            "db549_hlo_sha256": args.db549_hlo_sha256,
            "db549_runner_sha256": args.db549_runner_sha256,
            "db549_summary_sha256": args.db549_summary_sha256,
            "db549_success_sha256": args.db549_success_sha256,
            "db549_tensor_sha256": args.db549_tensor_sha256,
        },
        "status": "SUCCESS",
        "virtual_contractions_per_chip": 1,
        "virtual_rank_batches": 8,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.tensor_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.tensor_output,
        accepted_layer1_normalized_bfloat16_bits=accepted_bits,
        captured_dense_virtual_partials_bfloat16_bits=captured_partial_bits,
        control_layer1_normalized_bfloat16_bits=control_bits,
        isolated_dense_virtual_partials_bfloat16_bits=isolated_partial_bits,
        isolated_layer1_normalized_bfloat16_bits=isolated_bits,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
