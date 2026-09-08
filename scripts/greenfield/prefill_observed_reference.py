"""Full layer3 reference with all DB585 outputs plus actual PREattention norm.

Distinct from failed v2 and the12-output boundary-only diagnostic. The first12
outputs must reproduce DB585 before any MLP execution in this campaign.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
from typing import Any

import numpy as np

from scripts.greenfield import prefill_materialized_reference as v2
from scripts.greenfield import prefill_prefix_mlp_protocol as boundary
from scripts.greenfield.prefill_router_boundary import FIELDS, scalar_prefix_inputs

KERNEL = "ws32_prefill_layer_observed_admission"
PROTOCOL = "ws32-prefill-layer-db585-observed-reference-v3"
REFERENCE_SCOPE = "RAW_SCALAR_DB585_PREFIX_ACTUAL_PRENORM_COMPLETED_MLP_NOT_LEGACY"
PROGRAMS = v2.PROGRAMS
verify_input_capture = v2.verify_input_capture


def is_observed_tag(tag: str) -> bool:
    return bool(
        re.fullmatch(
            r"greenfield_fp8_ws32_prefill_layer_observed_admission_l3_[a-zA-Z0-9_]+",
            tag,
        )
    )


def build_materialized_reference(
    mesh: Any, input_specs: tuple[Any, ...], **kwargs: Any
) -> tuple[Any, Any]:
    import jax
    from scripts.greenfield.prefill_router_boundary import build_router_prefix_program

    _, suffix = v2.build_materialized_reference(mesh, input_specs, **kwargs)
    prefix_kwargs = {k: v for k, v in kwargs.items() if k != "moe_contract"}
    prefix = build_router_prefix_program(
        mesh, input_specs, batched=False, capture_pre_norm=True, **prefix_kwargs
    )
    return jax.jit(prefix), suffix


def scalar_observed_inputs(
    values: tuple[Any, ...], row: int, previous: tuple[Any, ...] | None = None
) -> tuple[Any, ...]:
    if previous is not None and len(previous) != 13:
        raise ValueError("observed prefix requires13 results")
    # Host tuple indexing only; the compiled prefix retains all13 output leaves.
    return scalar_prefix_inputs(
        values, row, None if previous is None else previous[:12]
    )


def assemble_reference_result(
    values: tuple[Any, ...], prefix: tuple[Any, ...], mlp: tuple[Any, ...]
) -> tuple[Any, ...]:
    if len(prefix) != 13:
        raise ValueError("actual PREattention normalization is missing")
    return v2.assemble_reference_result(
        values, (prefix[0], prefix[9], prefix[10], prefix[12], prefix[11]), mlp
    )


def check_reference_hlo(hlo: str, name: str) -> dict[str, Any]:
    if name == "reference_prefix":
        return boundary.check_hlo(hlo, "candidate")
    return v2.check_reference_hlo(hlo, name)


def verify_boundary_file(
    path: Path, slots: dict[int, int], ledger: dict[int, Any]
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as a:
        proof = boundary.verify_prefix(a, slots, ledger)
        expected = {f"input__{n}" for n in boundary.INPUT_FIELDS}
        for d in slots:
            expected.update(
                f"weights_{d}__{n}" for n in ("router_weight", "correction_bias")
            )
            expected.update(f"prefix_{d}__{n}" for n in (*FIELDS, "pre_attention_norm"))
            norm = a[f"prefix_{d}__pre_attention_norm"]
            if (
                norm.shape != (17, 1536)
                or norm.dtype != np.uint16
                or not np.isfinite(
                    norm.view(boundary.router.BF16).astype(np.float32)
                ).all()
            ):
                raise ValueError(
                    "actual PREattention normalization shape/dtype/finiteness differs"
                )
        if set(a.files) != expected:
            raise ValueError("observed prefix evidence inventory differs")
    return proof


def prepare_boundary(
    *,
    prefix: Any,
    values: tuple[Any, ...],
    host: dict[str, Any],
    root: Path,
    record: dict[str, Any],
    slots: dict[int, int],
    consensus: Any,
) -> list[tuple[Any, ...]]:
    """Capture once, verify globally before any MLP, retain for boundary admission."""
    import jax
    from scripts.greenfield.prefill_layer_evidence import encode_arrays
    from scripts.greenfield.prefill_router_worker import observe, stack
    from scripts.greenfield.microbench_fp8_matmul import _atomic_json
    from scripts.greenfield.ws32_prefill_layer_campaign import checkpoint_ledger

    # values come from the exact fixed boundary fixture, independently checked below.
    arrays = encode_arrays("input", host)
    for name, value in (
        ("router_weight", values[17].router_weight_local),
        ("correction_bias", values[17].correction_bias_local),
    ):
        for s in value.addressable_shards:
            arrays.update(
                encode_arrays(
                    f"weights_{int(s.device.id)}", {name: np.asarray(s.data).copy()}
                )
            )
    completed, rows, previous = [], {d: [] for d in slots}, None
    for row in range(17):
        previous = prefix(*scalar_observed_inputs(values, row, previous))
        jax.block_until_ready(previous)
        completed.append(previous)
        for d, v in observe(previous, (*FIELDS, "pre_attention_norm")).items():
            rows[d].append(v)
    for d, values_by_row in rows.items():
        arrays.update(encode_arrays(f"prefix_{d}", stack(values_by_row)))
    path = root / "boundary_prefix.npz"
    np.savez_compressed(path, **arrays)
    _, ledger = checkpoint_ledger(3)
    try:
        proof = verify_boundary_file(path, slots, ledger)
        passed = True
    except ValueError as exc:
        proof, passed = {"error": str(exc)}, False
    record["boundary_prefix"] = dict(
        npz_sha256=sha256(path.read_bytes()).hexdigest(), replay=proof
    )
    _atomic_json(root / "runner.json", record)
    if not consensus(passed):
        raise ValueError(
            "13-output prefix differs from DB585; MLP and full-layer execution forbidden"
        )
    return completed


def validate_boundary_binding(
    root: Path, record: dict[str, Any], slots: dict[int, int], ledger: dict[int, Any]
) -> None:
    path = root / "boundary_prefix.npz"
    if sha256(path.read_bytes()).hexdigest() != record["boundary_prefix"]["npz_sha256"]:
        raise ValueError("observed prefix original digest differs")
    proof = verify_boundary_file(path, slots, ledger)
    if proof != record["boundary_prefix"]["replay"]:
        raise ValueError("observed prefix original replay differs")
    with np.load(path, allow_pickle=False) as p, np.load(
        root / "boundary.npz", allow_pickle=False
    ) as c, np.load(root / "boundary.reference_input.npz", allow_pickle=False) as i:
        for d in slots:
            if not np.array_equal(p[f"prefix_{d}__router_input"], i[f"device_{d}"]):
                raise ValueError("boundary MLP input not bound to original prefix")
            if not np.array_equal(
                p[f"prefix_{d}__pre_attention_norm"], c[f"reference_{d}__normalized"]
            ):
                raise ValueError(
                    "full-layer PREattention norm is not actual prefix observation"
                )
