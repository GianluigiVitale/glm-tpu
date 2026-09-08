"""Default-off scalar layer3 reference with a completed BF16 MLP input.

This is a NEW reference realization, not the v1 fused scalar comparator or a
production execution path. Both programs must complete as separate executables.
"""

from __future__ import annotations

from typing import Any
import re
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np

KERNEL = "ws32_prefill_layer_materialized_admission"
PROTOCOL = "ws32-prefill-layer-materialized-bf16-reference-v2"
REFERENCE_SCOPE = (
    "RAW_SCALAR_COMPLETED_BF16_MLP_INPUT_NOT_V1_OR_PROMOTED_DECODER_OR_LEGACY"
)
PROGRAMS = ("candidate", "reference_prefix", "reference")


def is_materialized_tag(tag: str) -> bool:
    return bool(
        re.fullmatch(
            r"greenfield_fp8_ws32_prefill_layer_materialized_admission_l3_[a-zA-Z0-9_]+",
            tag,
        )
    )


def check_reference_hlo(hlo: str, name: str) -> dict[str, Any]:
    """Bound the two actual reference programs, never a candidate HLO exemption."""
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
    from scripts.greenfield.prefill_layer_hlo import FEATURE, EXPERT

    if name not in ("reference_prefix", "reference"):
        raise ValueError("unknown materialized reference program")
    prefix = name == "reference_prefix"
    module = parse_hlo_module(hlo)
    collectives = [o for o in module.instructions if o.is_collective]
    payload = Counter(
        (o.opcode, o.maximum_group_size, s.dtype, s.element_count)
        for o in collectives
        for s in (o.operand_shapes if o.opcode == "all-reduce" else o.result_shapes)
    )
    if prefix:
        expected = Counter(
            {
                ("all-reduce", 4, "f32", 1): 2,
                ("all-reduce", 4, "f32", 2048): 1,
                ("all-reduce", 4, "f32", 576): 1,
                ("all-reduce", 8, "bf16", 2048 * 640): 1,
                ("all-reduce", 8, "f32", 1536): 1,
            }
        )
        variants = [expected]
    else:
        expected = Counter(
            {
                ("all-reduce", 4, "f32", 4096): 9,
                ("all-reduce", 4, "f32", 32): 1,
                ("all-reduce", 8, "f32", 1536): 1,
            }
        )
        variants = [
            expected + Counter([(a, 8, "f32", 256), (b, 8, "f32", 256)])
            for a in ("all-reduce", "all-gather")
            for b in ("all-reduce", "all-gather")
        ]
    calls = [o for o in module.instructions if o.opcode == "custom-call"]
    pallas = [o for o in calls if 'custom_call_target="tpu_custom_call"' in o.raw_line]
    helpers = [o for o in calls if o not in pallas]
    helpers_valid = True
    for o in helpers:
        if (
            len(o.result_shapes) != 1
            or "custom_call_has_side_effect=true" in o.raw_line
        ):
            helpers_valid = False
            continue
        s = o.result_shapes[0]
        if 'custom_call_target="AssumeGatherIndicesInBound"' in o.raw_line:
            helpers_valid &= (
                s.dtype == "s32"
                and s.element_count <= 2048
                and len(o.operand_names) == 1
                and o.operand_shapes == o.result_shapes
            )
        elif 'custom_call_target="ConcatBitcast"' in o.raw_line:
            helpers_valid &= (
                s.dtype == "u8"
                and len(s.dimensions) == 2
                and s.element_count <= 2048**2
                and len(o.operand_names) == len(o.operand_shapes) == 4
                and all(
                    x.dtype == "u8"
                    and x.dimensions == (s.dimensions[0] // 4, s.dimensions[1])
                    for x in o.operand_shapes
                )
            )
        else:
            helpers_valid = False
    checks = dict(
        groups=all(
            o.replica_groups in (FEATURE, EXPERT)
            and o.opcode in ("all-reduce", "all-gather")
            for o in collectives
        ),
        payload=payload in variants,
        pallas=len(pallas) == (7 if prefix else 27)
        and sum("greenfield_fp8_block_matmul" in o.raw_line for o in pallas)
        == (4 if prefix else 27),
        helpers=helpers_valid and len(helpers) <= 8,
        no_host=not any(
            o.opcode in ("infeed", "outfeed", "send", "recv")
            for o in module.instructions
        ),
        bounded_arrays=all(
            s.element_count < 32 * 2048 * 1536
            or (
                s.dtype == "u8" and s.dimensions in ((32, 2048, 1536), (32, 1536, 2048))
            )
            for o in module.instructions
            for s in o.result_shapes
        ),
        bf16_input=prefix
        or any(
            o.opcode == "parameter"
            and any(
                s.dtype == "bf16" and s.dimensions == (1, 1536) for s in o.result_shapes
            )
            for o in module.instructions
        ),
    )
    return dict(
        passed=all(checks.values()),
        checks=checks,
        collectives=[o.to_dict() for o in collectives],
        reference_scope=REFERENCE_SCOPE,
    )


def verify_input_capture(
    path: Path, expected_sha256: str, devices: set[int], count: int
) -> None:
    from scripts.greenfield.prefill_layer_evidence import BF16

    if sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("materialized reference input capture digest differs")
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {f"device_{d}" for d in devices}:
            raise ValueError("materialized reference input owners differ")
        for value in arrays.values():
            if (
                value.shape != (count, 1536)
                or value.dtype != np.uint16
                or not np.isfinite(value.view(BF16).astype(np.float32)).all()
            ):
                raise ValueError(
                    "materialized reference input shape/dtype/finiteness differs"
                )


def build_materialized_reference(
    mesh: Any,
    input_specs: tuple[Any, ...],
    *,
    dsa_contract: Any,
    attention_contract: Any,
    moe_contract: Any,
    rms_norm_epsilon: float = 1e-5,
    linear_interpret: bool = False,
    sparse_attention_interpret: bool = False,
) -> tuple[Any, Any]:
    """Build separate JIT prefix and raw scalar MLP; never JIT their composition."""
    import jax
    from jax.sharding import PartitionSpec as P
    from glm_tpu.greenfield.kernels.ws32 import ws32_fused_add_rms_norm_mapped
    from glm_tpu.greenfield.kernels.ws32_layer import (
        ws32_attention_layer_mapped,
        ws32_mlp_mapped,
    )

    if len(input_specs) != 20:
        raise ValueError("materialized reference needs20-field layer schema")

    def prefix(*values: Any) -> tuple[Any, ...]:
        (
            u,
            r,
            kv,
            ic,
            rc,
            s,
            n,
            sc,
            offset,
            count,
            table,
            q,
            a,
            d,
            wk,
            post,
            dense,
            moe,
            health,
            rope,
        ) = values
        if (
            u.shape[0] != 1
            or d is not None
            or wk is not None
            or dense is not None
            or moe is None
        ):
            raise ValueError("materialized reference is scalar IndexShare+MoE only")
        norm, combined = ws32_fused_add_rms_norm_mapped(
            u,
            r,
            q.input_norm_weight_local,
            global_hidden_size=dsa_contract.hidden_size,
            epsilon=rms_norm_epsilon,
        )
        attention = ws32_attention_layer_mapped(
            combined,
            kv[0],
            ic[0],
            s,
            n,
            sc,
            offset[None],
            table,
            (offset + 1)[None],
            q,
            a,
            None,
            dsa_contract=dsa_contract,
            attention_contract=attention_contract,
            precomputed_normalized_local=norm,
            main_rope_table_row=rope[0],
            linear_interpret=linear_interpret,
            sparse_attention_interpret=sparse_attention_interpret,
            add_residual=False,
        )
        mlp_input, carried = ws32_fused_add_rms_norm_mapped(
            attention.output_local,
            combined,
            post,
            global_hidden_size=moe_contract.hidden_size,
            epsilon=rms_norm_epsilon,
        )
        return (
            mlp_input,
            carried,
            attention.cache_local[None],
            norm,
            (health[0, 0] & attention.contract_valid)[None, None],
        )

    def suffix(mlp_input: Any, carried: Any, post: Any, moe: Any) -> tuple[Any, ...]:
        result = ws32_mlp_mapped(
            carried,
            post,
            None,
            moe,
            mlp_kind="sparse",
            contract=moe_contract,
            rms_norm_epsilon=rms_norm_epsilon,
            linear_interpret=linear_interpret,
            precomputed_normalized_local=mlp_input,
            add_residual=False,
        )
        return result.output_local, result.route_indices, result.route_weights

    return (
        jax.jit(
            jax.shard_map(
                prefix,
                mesh=mesh,
                in_specs=input_specs,
                out_specs=(
                    P(None, "feature"),
                    P(None, "feature"),
                    P("expert", None, None, None),
                    P(None, "feature"),
                    P("expert", "feature", None),
                ),
                check_vma=False,
            )
        ),
        jax.jit(
            jax.shard_map(
                suffix,
                mesh=mesh,
                in_specs=(
                    P(None, "feature"),
                    P(None, "feature"),
                    input_specs[15],
                    input_specs[17],
                ),
                out_specs=(P(None, "feature"), P(), P()),
                check_vma=False,
            )
        ),
    )


def assemble_reference_result(
    values: tuple[Any, ...], prefix: tuple[Any, ...], mlp: tuple[Any, ...]
) -> tuple[Any, ...]:
    """Return old12-field schema while carrying only this reference's own state."""
    if len(values) != 20 or len(prefix) != 5 or len(mlp) != 3:
        raise ValueError("materialized reference result schema differs")
    if prefix[2].shape != values[2].shape or prefix[2].dtype != values[2].dtype:
        raise ValueError("materialized reference KV shape/dtype differs")
    return (
        mlp[0],
        prefix[1],
        prefix[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
        mlp[1],
        mlp[2],
        prefix[4],
        prefix[3],
    )
