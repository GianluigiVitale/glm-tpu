#!/usr/bin/env python3
"""Bounded CPU witness: does the TPU current key change the DSA selected set?

Replays the accepted layer-1 position-8155 capsule on two forced CPU devices
through the real stage-local DSA path (accepted prompt cache, query and head
weights), then injects the archived TPU current keys from the V2 replay
(on-device rotary, rejected) and the V3 replay (host FP32 rotary row,
accepted) as ``precomputed_current_key`` and compares the exact top-k selected
positions, valid counts and scores against the capsule's recorded event.  CPU
only; no accelerator is touched.
"""

from __future__ import annotations

import json
import os
import sys
from hashlib import sha256
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.pop("TPU_VISIBLE_DEVICES", None)
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=2"
).strip()

WORKTREE = Path("/home/gianl/glm-tpu-gate-d-pp16-numerical")
CAPSULE = Path(
    "/home/gianl/gate-d-runs/greenfield_gate_d_compensated_capsule_20260831T124838Z"
)
V2_RUN = Path(
    "/home/gianl/gate-d-runs/gate_d_projection_contraction_pp16_numerical_20260901T235818944668679Z"
)
V3_RUN = Path(
    "/home/gianl/gate-d-runs/gate_d_projection_contraction_pp16_numerical_20260902T004306002075694Z"
)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    sys.path.insert(0, str(WORKTREE))
    import jax
    import jax.numpy as jnp
    import ml_dtypes
    import numpy as np
    from jax import lax
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from glm_tpu.greenfield.benchmarking.gate_d_compensated_capsule import (
        GATE_D_CAPSULE_CONTEXT_LENGTH,
        GATE_D_CAPSULE_LOGICAL_PAGE_SIZE,
        GATE_D_CAPSULE_PAGE_COUNT,
        GATE_D_CAPSULE_POSITION,
    )
    from glm_tpu.greenfield.kernels.reference.attention import StageLocalKvLayout
    from glm_tpu.greenfield.kernels.reference.dsa import DsaNumericalContract
    from glm_tpu.greenfield.kernels.reference.fp8 import (
        dequantize_fp8_bits_block_weight,
    )
    from glm_tpu.greenfield.kernels.reference.prefill_index import (
        decode_stage_local_prefill_index_wk_bf16,
        promote_stage_local_prefill_index_wk,
    )
    from glm_tpu.greenfield.kernels.reference.qkv_a import (
        FusedQkvAContract,
        one_row_fused_qkv_a_convolution,
    )
    from glm_tpu.greenfield.kernels.reference.rmsnorm import (
        fused_add_rms_norm_with_compensated_auxiliary,
    )
    from glm_tpu.greenfield.kernels.reference.rotary_table import dsa_rotary_row_host
    from glm_tpu.greenfield.kernels.stage_local import stage_local_dsa_fp8_mapped

    if (
        jax.default_backend() != "cpu"
        or len(jax.devices()) != 2
        or any(device.platform != "cpu" for device in jax.devices())
    ):
        raise RuntimeError("selection witness requires exactly two forced CPU devices")

    inputs = {k: v for k, v in np.load(CAPSULE / "candidate-inputs.npz").items()}
    state = {k: v for k, v in np.load(CAPSULE / "candidate-state.npz").items()}
    v2_key = np.load(V2_RUN / "outputs.npz")["current_key_owners"].astype(np.float32)
    v3_key = np.load(V3_RUN / "outputs.npz")["current_key_owners"].astype(np.float32)
    materialized_query, materialized_wk = [], []
    for slot in range(2):
        materialized_query.append(
            np.asarray(
                dequantize_fp8_bits_block_weight(
                    jnp.asarray(inputs["wq_b_weight_bits"][slot]),
                    jnp.asarray(inputs["wq_b_scale_inv"][slot]),
                    output_dtype=jnp.float32,
                )
            )
        )
        wk_bf16 = decode_stage_local_prefill_index_wk_bf16(
            jnp.asarray(inputs["wk_weight_bits"][slot]),
            jnp.asarray(inputs["wk_scale_inv"][slot]),
        )
        materialized_wk.append(
            np.asarray(promote_stage_local_prefill_index_wk(wk_bf16))
        )

    axis_name = "feature"
    mesh = Mesh(np.asarray(tuple(jax.devices()), dtype=object), (axis_name,))
    dsa_contract = DsaNumericalContract()
    qkv_contract = FusedQkvAContract()
    cache_layout = StageLocalKvLayout(
        logical_page_size=GATE_D_CAPSULE_LOGICAL_PAGE_SIZE,
        local_parallel_size=2,
        packed_cache_width=640,
    )

    def make_replay(*, use_host_row: bool, inject_key: bool):
        def mapped(
            hidden_update,
            residual,
            rms_weight,
            prompt_cache,
            qkv_a_weight_bits,
            qkv_a_scale_inv,
            q_a_norm_weight,
            wq_b_bits,
            wq_b_scale,
            wq_b_weight,
            wk_bits,
            wk_scale,
            wk_weight,
            key_norm_weight,
            key_norm_bias,
            head_weight,
            rope_row_owners,
            injected_key_owners,
        ):
            local_slot = lax.axis_index(axis_name)
            rms = fused_add_rms_norm_with_compensated_auxiliary(
                hidden_update, residual, rms_weight, epsilon=1e-5
            )
            qkv_a = one_row_fused_qkv_a_convolution(
                rms.output,
                qkv_a_weight_bits,
                qkv_a_scale_inv,
                q_a_norm_weight,
                contract=qkv_contract,
            )
            dsa = stage_local_dsa_fp8_mapped(
                None,
                prompt_cache[0],
                jnp.asarray([GATE_D_CAPSULE_POSITION], dtype=jnp.int32),
                jnp.arange(GATE_D_CAPSULE_PAGE_COUNT, dtype=jnp.int32)[None, :],
                jnp.asarray([GATE_D_CAPSULE_CONTEXT_LENGTH], dtype=jnp.int32),
                rms_weight,
                None,
                None,
                q_a_norm_weight,
                wq_b_bits[0],
                wq_b_scale[0],
                wk_bits[0],
                wk_scale[0],
                key_norm_weight[0],
                key_norm_bias[0],
                head_weight[0],
                local_slot,
                axis_name=axis_name,
                contract=dsa_contract,
                cache_layout=cache_layout,
                axis_index_groups=((0, 1),),
                precomputed_normalized=rms.output,
                precomputed_q_residual=qkv_a.q_residual,
                linear_backend="pallas",
                dsa_query_backend="reference",
                dsa_query_weight_aliases=(wq_b_weight[0],) * 4,
                precomputed_wk_weight=wk_weight[0],
                precomputed_current_key=injected_key_owners[0] if inject_key else None,
                dsa_head_key_exact_association=True,
                dsa_score_precision="default",
                dsa_rope_table_row=rope_row_owners[0] if use_host_row else None,
            )
            return (
                dsa.internals.current_key[None, ...],
                dsa.selected_positions[None, ...],
                dsa.valid_counts[None, ...],
                dsa.selected_scores[None, ...],
                jnp.asarray(dsa.contract_valid, dtype=jnp.uint8).reshape(1),
            )

        owner_matrix = P(axis_name, None, None)
        owner_vector = P(axis_name, None)
        return jax.jit(
            jax.shard_map(
                mapped,
                mesh=mesh,
                in_specs=(
                    P(),
                    P(),
                    P(),
                    P(axis_name, None, None, None),
                    P(),
                    P(),
                    P(),
                    owner_matrix,
                    owner_matrix,
                    owner_matrix,
                    owner_matrix,
                    owner_matrix,
                    owner_matrix,
                    owner_vector,
                    owner_vector,
                    owner_matrix,
                    owner_vector,
                    owner_matrix,
                ),
                out_specs=(
                    P(axis_name, None, None),
                    P(axis_name, None, None),
                    P(axis_name, None),
                    P(axis_name, None, None),
                    P(axis_name),
                ),
                check_vma=False,
            )
        )

    host_row = dsa_rotary_row_host(
        GATE_D_CAPSULE_POSITION,
        rotary_dim=dsa_contract.rotary_dim,
        theta=dsa_contract.theta,
    )
    rows = jnp.asarray(np.stack((host_row, host_row)))
    common = (
        jnp.asarray(
            inputs["rms_hidden_update_bf16_bits"].view(ml_dtypes.bfloat16)[None, :]
        ),
        jnp.asarray(inputs["rms_residual_bf16_bits"].view(ml_dtypes.bfloat16)[None, :]),
        jnp.asarray(inputs["rms_weight_bf16_bits"].view(ml_dtypes.bfloat16)),
        jnp.asarray(inputs["prompt_cache_bf16_bits"].view(ml_dtypes.bfloat16)),
        jnp.asarray(inputs["qkv_a_weight_bits"]),
        jnp.asarray(inputs["qkv_a_scale_inv"]),
        jnp.asarray(inputs["q_a_norm_bf16_bits"].view(ml_dtypes.bfloat16)),
        jnp.asarray(inputs["wq_b_weight_bits"]),
        jnp.asarray(inputs["wq_b_scale_inv"]),
        jnp.asarray(np.stack(materialized_query)),
        jnp.asarray(inputs["wk_weight_bits"]),
        jnp.asarray(inputs["wk_scale_inv"]),
        jnp.asarray(np.stack(materialized_wk)),
        jnp.asarray(inputs["key_norm_weight_bf16_bits"].view(ml_dtypes.bfloat16)),
        jnp.asarray(inputs["key_norm_bias_bf16_bits"].view(ml_dtypes.bfloat16)),
        jnp.asarray(inputs["head_weight_bf16_bits"].view(ml_dtypes.bfloat16)),
    )
    accepted_positions = state["event1_positions"][0].astype(np.int32)
    accepted_scores = state["event1_scores"][0].astype(np.float32)
    accepted_count = int(state["event1_valid_count"][0])
    accepted_key = state["current_key"].astype(np.float32)

    def run(name, *, use_host_row, inject_key, key=None):
        replay = make_replay(use_host_row=use_host_row, inject_key=inject_key)
        injected = jnp.asarray(
            key if key is not None else np.zeros((2, 1, 128), np.float32)
        )
        out = replay(*common, rows, injected)
        jax.block_until_ready(out)
        current_key, positions, counts, scores, valid = (np.asarray(v) for v in out)
        assert valid.tolist() == [1, 1], name
        assert positions[0].tobytes() == positions[1].tobytes(), name
        pos = positions[0, 0].astype(np.int32)
        count = int(counts[0, 0])
        sc = scores[0, 0].astype(np.float32)
        live = pos[:count]
        acc_live = accepted_positions[:accepted_count]
        return {
            "current_key_max_abs_vs_accepted": float(
                np.abs(current_key[0, 0] - accepted_key).max()
            ),
            "valid_count": count,
            "selected_positions_equal_ordered": bool(
                np.array_equal(pos, accepted_positions)
            ),
            "selected_set_equal": set(live.tolist()) == set(acc_live.tolist()),
            "selected_set_symmetric_difference": int(
                len(set(live.tolist()) ^ set(acc_live.tolist()))
            ),
            "first_order_mismatch_index": next(
                (int(i) for i in range(len(pos)) if pos[i] != accepted_positions[i]),
                None,
            ),
            "selected_scores_max_abs_vs_accepted": float(
                np.abs(sc - accepted_scores).max()
            ),
            "selected_scores_equal": bool(np.array_equal(sc, accepted_scores)),
        }

    results = {
        "cpu_default_on_device_rotary": run(
            "default", use_host_row=False, inject_key=False
        ),
        "cpu_host_rotary_row": run("host_row", use_host_row=True, inject_key=False),
        "tpu_v2_key_injected": run(
            "v2", use_host_row=False, inject_key=True, key=v2_key
        ),
        "tpu_v3_key_injected": run(
            "v3", use_host_row=False, inject_key=True, key=v3_key
        ),
    }
    report = {
        "artifact_kind": "gate_d_dsa_selection_witness_layer1_position8155",
        "analysis_scope": "CPU-only two-device replay of the accepted capsule through stage_local_dsa_fp8_mapped; archived TPU keys injected as precomputed_current_key; no accelerator access.",
        "capsule_sha256": _sha(CAPSULE / "capsule.json"),
        "capsule_inputs_sha256": _sha(CAPSULE / "candidate-inputs.npz"),
        "capsule_state_sha256": _sha(CAPSULE / "candidate-state.npz"),
        "v2_outputs_sha256": _sha(V2_RUN / "outputs.npz"),
        "v3_outputs_sha256": _sha(V3_RUN / "outputs.npz"),
        "v2_run_tag": V2_RUN.name,
        "v3_run_tag": V3_RUN.name,
        "accepted_event": {
            "valid_count": accepted_count,
            "top_k": int(accepted_positions.shape[0]),
        },
        "results": results,
        "classification": (
            "DSA_SELECTION_WITNESS_"
            + (
                "V3_EXACT_V2_DIVERGENT"
                if results["tpu_v3_key_injected"]["selected_positions_equal_ordered"]
                and not results["tpu_v2_key_injected"][
                    "selected_positions_equal_ordered"
                ]
                else "V3_EXACT_V2_EXACT"
                if results["tpu_v3_key_injected"]["selected_positions_equal_ordered"]
                else "V3_INEXACT"
            )
            + ";CPU_ONLY;DECODER_UNPROVEN;GATE_D_OPEN"
        ),
        "gate_d_closed": False,
        "performance_claim": False,
        "schema_version": 1,
    }
    sys.stdout.write(
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
