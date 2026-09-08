"""Diagnostic branch of the existing selected-layer worker; no independent launch."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from scripts.greenfield.prefill_layer_evidence import encode_arrays, host_case
from scripts.greenfield.prefill_router_boundary import (
    FIELDS,
    build_router_prefix_program,
    build_router_replay_program,
    scalar_prefix_inputs,
)
from scripts.greenfield.prefill_router_protocol import (
    KERNEL,
    PROTOCOL,
    check_hlo,
    verify_prefix,
    replay_file,
)


def compile_observer(
    fn: Any, inputs: tuple[Any, ...], kind: str, root: Path, record: dict[str, Any]
) -> Any:
    """Lower the actual shard_map through JIT and preserve both graph forms."""
    import jax
    from scripts.greenfield.probe_ws32_prefill_layer import compile_program

    return compile_program(jax.jit(fn), inputs, kind, root, record)


def observe(
    values: tuple[Any, ...], fields: tuple[str, ...] = FIELDS
) -> dict[int, dict[str, np.ndarray]]:
    if len(values) != len(fields):
        raise ValueError("router observer fields differ")
    result = {}
    devices = None
    for name, value in zip(fields, values, strict=True):
        current = set()
        for shard in value.addressable_shards:
            d = int(shard.device.id)
            if d in current:
                raise ValueError("duplicate router physical owner")
            current.add(d)
            v = np.asarray(shard.data).copy()
            if name in ("partial_logits", "prefix_health"):
                v = v[0, 0]
            elif name == "kv":
                v = v[0]
            result.setdefault(d, {})[name] = v
        if devices is not None and devices != current:
            raise ValueError("router observer owners differ between leaves")
        devices = current
    return result


def stack(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if len(rows) != 17 or any(set(r) != set(rows[0]) for r in rows):
        raise ValueError("router scalar reference needs17 rows")
    result = {}
    for name in rows[0]:
        if name == "kv":
            result[name] = rows[-1][name]
        elif name == "bias":
            if any(not np.array_equal(rows[0][name], r[name]) for r in rows):
                raise ValueError("scalar bias changed between rows")
            result[name] = rows[0][name]
        else:
            result[name] = np.concatenate([r[name] for r in rows], axis=0)
    return result


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
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from glm_tpu.greenfield.kernels.reference.rotary import build_rotary_table_host
    from scripts.greenfield.probe_ws32_prefill_layer import (
        input_specs,
        device_inputs,
    )
    from scripts.greenfield.microbench_fp8_matmul import _atomic_json
    from scripts.greenfield.ws32_prefill_layer_campaign import checkpoint_ledger

    record.update(
        protocol=PROTOCOL, kernel=KERNEL, diagnostic_only=True, admission_only=False
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

    def compile_checked(fn, inputs, kind):
        compiled = compile_observer(fn, inputs, kind, args.output_dir, record)
        proof = check_hlo(compiled.as_text(), kind)
        record["programs"][kind]["hlo_contract"] = proof
        record["hlo"] = dict(
            sha256=record["programs"]["candidate"]["optimized_hlo_sha256"],
            contract={
                "passed": all(
                    p["hlo_contract"]["passed"] for p in record["programs"].values()
                )
            },
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
            raise ValueError(
                f"router diagnostic pre-execution HLO/memory refuses {kind}"
            )
        return compiled

    observed = {}
    for kind, batched in (("candidate", True), ("reference", False)):
        fn = build_router_prefix_program(
            mesh,
            specs,
            batched=batched,
            dsa_contract=config.dsa_contract,
            attention_contract=config.attention_contract,
            rms_norm_epsilon=config.rms_norm_epsilon,
        )
        inputs = values if batched else scalar_prefix_inputs(values, 0)
        compiled = compile_checked(fn, inputs, kind)
        if batched:
            out = compiled(*inputs)
            jax.block_until_ready(out)
            observed["actual"] = observe(out)
        else:
            rows = {d: [] for d in local_slots}
            previous = None
            for row in range(17):
                previous = compiled(*scalar_prefix_inputs(values, row, previous))
                jax.block_until_ready(previous)
                for d, v in observe(previous).items():
                    rows[d].append(v)
            observed["reference"] = {d: stack(v) for d, v in rows.items()}
        capture = "actual" if batched else "reference"
        for d, v in observed[capture].items():
            arrays.update(encode_arrays(f"{capture}_{d}", v))
        np.savez_compressed(path, **arrays)

    # Capture originals before deciding: a changed diagnostic graph may erase
    # the original signature. That is a refused reproduction, never a fix.
    try:
        reproduction = verify_prefix(arrays, local_slots, ledger)
        reproduced = True
    except ValueError as exc:
        reproduction, reproduced = {"error": str(exc)}, False
    record["cases"]["boundary"] = dict(prefix_reproduction=reproduction)
    _atomic_json(args.output_dir / "runner.json", record)
    if not consensus(reproduced):
        raise ValueError(
            "original17-row router signature not reproduced; saved diagnostic prefix only"
        )

    replay_fn = build_router_replay_program(mesh)
    replay_programs = {}
    sharding = NamedSharding(mesh, P(None, "feature"))
    for source in ("actual", "reference"):
        # Declared observation roundtrip for the untimed router diagnostic only.
        # Every local device receives its own captured feature slice; no global
        # reconstruction of a model activation or prefix-state carry in replay.
        h = jax.make_array_from_single_device_arrays(
            (17, 6144),
            sharding,
            [
                jax.device_put(observed[source][int(d.id)]["router_input"], d)
                for d in sharding.addressable_devices_indices_map((17, 6144))
            ],
        )
        for batch in (True, False):
            name = "router_batch" if batch else "router_scalar"
            inputs = (
                h if batch else h[:1],
                weights.moe.router_weight_local,
                weights.moe.correction_bias_local,
                jnp.ones((17 if batch else 1,), jnp.bool_),
            )
            if name not in replay_programs:
                replay_programs[name] = compile_checked(replay_fn, inputs, name)
            compiled = replay_programs[name]
            if batch:
                out = compiled(*inputs)
                jax.block_until_ready(out)
                result = observe(out, FIELDS[:7])
            else:
                rows = {d: [] for d in local_slots}
                for row in range(17):
                    out = compiled(h[row : row + 1], *inputs[1:])
                    jax.block_until_ready(out)
                    for d, v in observe(out, FIELDS[:7]).items():
                        rows[d].append(v)
                result = {d: stack(v) for d, v in rows.items()}
            for d, v in result.items():
                arrays.update(
                    encode_arrays(f"{source}_{'batch' if batch else 'scalar'}_{d}", v)
                )
            np.savez_compressed(path, **arrays)
    replay = replay_file(path, local_slots, ledger)
    record["cases"]["boundary"].update(
        replay=replay,
        npz_sha256=sha256(path.read_bytes()).hexdigest(),
        input_sha256=replay["input_sha256"],
        evidence_complete=True,
    )
    if not consensus(replay["evidence_complete"]):
        raise ValueError("router original-array replay failed")
