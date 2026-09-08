"""Two-program boundary discriminator, called only by the guarded layer worker."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

import numpy as np


def execute_diagnostic(
    *,
    args: Any,
    record: dict[str, Any],
    mesh: Any,
    config: Any,
    weights: Any,
    local_slots: dict[int, int],
    consensus: Any,
) -> None:
    import jax
    from glm_tpu.greenfield.kernels.reference.rotary import build_rotary_table_host
    from scripts.greenfield import prefill_prefix_mlp_protocol as protocol
    from scripts.greenfield.prefill_materialized_reference import (
        build_materialized_reference,
    )
    from scripts.greenfield.prefill_router_boundary import (
        build_router_prefix_program,
        scalar_prefix_inputs,
    )
    from scripts.greenfield.prefill_router_worker import observe, stack
    from scripts.greenfield.prefill_layer_evidence import encode_arrays, host_case
    from scripts.greenfield.probe_ws32_prefill_layer import (
        input_specs,
        device_inputs,
        compile_program,
    )
    from scripts.greenfield.microbench_fp8_matmul import _atomic_json
    from scripts.greenfield.ws32_prefill_layer_campaign import checkpoint_ledger

    record.update(
        protocol=protocol.PROTOCOL,
        kernel=protocol.KERNEL,
        reference_scope=protocol.REFERENCE_SCOPE,
        diagnostic_only=True,
        admission_only=False,
    )
    specs = input_specs(weights, None)
    host = host_case(
        "boundary", build_rotary_table_host(1024, rotary_dim=64, theta=8e6)
    )
    values = device_inputs(host, specs, weights, None, mesh)
    arrays = encode_arrays("input", host)
    path = args.output_dir / "boundary.npz"
    _, ledger = checkpoint_ledger(3)
    for name, value in (
        ("router_weight", weights.moe.router_weight_local),
        ("correction_bias", weights.moe.correction_bias_local),
    ):
        for shard in value.addressable_shards:
            arrays.update(
                encode_arrays(
                    f"weights_{int(shard.device.id)}",
                    {name: np.asarray(shard.data).copy()},
                )
            )

    def compile_checked(fn: Any, inputs: tuple[Any, ...], kind: str) -> Any:
        compiled = compile_program(fn, inputs, kind, args.output_dir, record)
        proof = protocol.check_hlo(compiled.as_text(), kind)
        record["programs"][kind]["hlo_contract"] = proof
        record["hlo"] = dict(
            sha256=record["programs"]["candidate"]["optimized_hlo_sha256"],
            contract=dict(
                passed=all(
                    p["hlo_contract"]["passed"] for p in record["programs"].values()
                )
            ),
        )
        _atomic_json(args.output_dir / "runner.json", record)
        m = record["programs"][kind]["compiled_memory"]
        if (
            not proof["passed"]
            or sum(
                m[n]
                for n in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "temp_size_in_bytes",
                )
            )
            > 2 * 1024**3
        ):
            raise ValueError(f"prefix/MLP pre-execution HLO/memory refuses {kind}")
        return compiled

    # Do not wrap/slice the body or trim its12 outputs: observation affects numerics.
    prefix_fn = jax.jit(
        build_router_prefix_program(
            mesh,
            specs,
            batched=False,
            dsa_contract=config.dsa_contract,
            attention_contract=config.attention_contract,
            rms_norm_epsilon=config.rms_norm_epsilon,
        )
    )
    scalar_inputs = scalar_prefix_inputs(values, 0)
    prefix = compile_checked(prefix_fn, scalar_inputs, "candidate")
    _, suffix_fn = build_materialized_reference(
        mesh,
        specs,
        dsa_contract=config.dsa_contract,
        attention_contract=config.attention_contract,
        moe_contract=config.moe_contract,
        rms_norm_epsilon=config.rms_norm_epsilon,
    )
    shapes = jax.eval_shape(prefix_fn, *scalar_inputs)
    suffix = compile_checked(
        suffix_fn, (shapes[0], shapes[9], values[15], values[17]), "reference"
    )

    completed, rows, previous = [], {d: [] for d in local_slots}, None
    for row in range(17):
        previous = prefix(*scalar_prefix_inputs(values, row, previous))
        jax.block_until_ready(previous)
        completed.append(
            previous
        )  # retain actual device buffers, no host reconstruction
        for d, v in observe(previous).items():
            rows[d].append(v)
    for d, v in rows.items():
        arrays.update(encode_arrays(f"prefix_{d}", stack(v)))
    np.savez_compressed(path, **arrays)
    try:
        proof = protocol.verify_prefix(arrays, local_slots, ledger)
        reproduced = True
    except ValueError as exc:
        proof, reproduced = {"error": str(exc)}, False
    record["cases"]["boundary"] = dict(prefix_reproduction=proof)
    _atomic_json(args.output_dir / "runner.json", record)
    if not consensus(reproduced):
        raise ValueError(
            "DB585 complete prefix fingerprint not reproduced; MLP NOT executed"
        )

    rows = {d: [] for d in local_slots}
    for p in completed:
        result = suffix(p[0], p[9], values[15], values[17])
        jax.block_until_ready(result)
        for d, v in observe(
            (p[0], *result), ("router_input", "output", "routes", "route_weights")
        ).items():
            rows[d].append(v)
    for d, v in rows.items():
        arrays.update(encode_arrays(f"suffix_{d}", stack(v)))
    np.savez_compressed(path, **arrays)
    replay = protocol.replay_file(path, local_slots, ledger)
    record["cases"]["boundary"].update(
        replay=replay,
        input_sha256=replay["input_sha256"],
        npz_sha256=sha256(path.read_bytes()).hexdigest(),
        evidence_complete=True,
    )
    if not consensus(replay["evidence_complete"]):
        raise ValueError("prefix/MLP original-array replay failed")
