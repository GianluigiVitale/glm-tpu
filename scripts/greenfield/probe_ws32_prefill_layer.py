#!/usr/bin/env python3
"""Bounded real-weight full-layer worker, launched ONLY by the guarded wrapper.

One selected layer per invocation, no full-model load and no timing claim.
Original arrays are retained on failure; the controller must replay them and
validate the full fleet, HLO, provenance, archive and cleanup before admission.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.greenfield.microbench_fp8_matmul import (
    _atomic_json,
    _compiled_memory,
    _git_head,
    _memory_stats,
)
from scripts.greenfield.prefill_layer_numerical import (
    CASES,
    FIELDS,
    PROTOCOL,
    ROWS,
    compare_layer_case,
    written_addresses,
)
from scripts.greenfield.prefill_layer_evidence import (
    BF16,
    INPUT_FIELDS,
    METAMORPHIC,
    encode_arrays,
    host_case,
    input_hashes,
    local_observations,
    mutate_case,
    owner_inputs,
    replay_case,
    stack_reference,
)
from scripts.greenfield.prefill_layer_hlo import check_layer_hlo
from scripts.greenfield import prefill_materialized_reference as materialized_ref
from glm_tpu.greenfield.partitioning.source_inventory import (
    SourceInventory,
    inspect_source_inventory,
)
from scripts.greenfield.probe_ws32_prefill_moe import (
    TOPOLOGY,
    TOPOLOGY_SHA,
    FLEET_SHA,
    MESH_SHA,
)

KERNEL = "ws32_prefill_layer_admission"
PINS = REPO / "docs/artifacts/prefill-selected-layer-host-admission-20260907.json"
PAYLOAD_BYTES = {0: 21_557_920, 3: 324_821_552}


def authenticated_inventory(path: Path, expected_sha256: str) -> SourceInventory:
    """Validate the inventory and its pinned canonical digest, not JSON file bytes."""
    inventory = inspect_source_inventory(path)
    if inventory.inventory_sha256 != expected_sha256:
        raise ValueError("layer source inventory canonical hash drifted")
    return inventory


def layer_from_tag(tag: str) -> int:
    from scripts.greenfield.prefill_router_protocol import is_router_tag

    if is_router_tag(tag) or materialized_ref.is_materialized_tag(tag):
        return 3
    match = re.fullmatch(
        r"greenfield_fp8_ws32_prefill_layer_admission_l([03])_[a-zA-Z0-9_]+", tag
    )
    if match is None:
        raise ValueError("invalid complete-layer admission tag")
    return int(match[1])


def compile_program(
    fn: Any, inputs: tuple[Any, ...], name: str, root: Path, record: dict[str, Any]
) -> Any:
    """Preserve both actual graph forms and compiler memory for every program."""
    start = time.monotonic()
    lowered = fn.lower(*inputs)
    stable = str(lowered.compiler_ir(dialect="stablehlo"))
    (root / f"{name}.stablehlo.mlir").write_text(stable)
    compiled = lowered.compile()
    hlo = compiled.as_text()
    (root / f"{name}.optimized_hlo.txt").write_text(hlo)
    record.setdefault("programs", {})[name] = {
        "stablehlo_sha256": sha256(stable.encode()).hexdigest(),
        "optimized_hlo_sha256": sha256(hlo.encode()).hexdigest(),
        "compiled_memory": _compiled_memory(compiled),
        "compile_seconds": time.monotonic() - start,
    }
    _atomic_json(root / "runner.json", record)
    return compiled


def build_wk_programs(
    mesh: Any, bits_spec: Any, scale_spec: Any, *, contract: Any
) -> tuple[Any, Any]:
    """Reuse the mandatory COMPLETED BF16 decode -> separate FP32 promotion."""
    import jax
    from jax import lax
    from jax.sharding import PartitionSpec as P
    from glm_tpu.greenfield.kernels.reference.prefill_index import (
        decode_stage_local_prefill_index_wk_bf16,
        promote_stage_local_prefill_index_wk,
    )

    def decode(bits, scales):
        bits = lax.all_gather(bits, "feature", axis=1, tiled=True)
        scales = lax.all_gather(scales, "feature", axis=1, tiled=True)
        return decode_stage_local_prefill_index_wk_bf16(bits, scales, contract=contract)

    return (
        jax.jit(
            jax.shard_map(
                decode,
                mesh=mesh,
                in_specs=(bits_spec, scale_spec),
                out_specs=P(),
                check_vma=False,
            )
        ),
        jax.jit(
            jax.shard_map(
                lambda x: promote_stage_local_prefill_index_wk(x, contract=contract),
                mesh=mesh,
                in_specs=(P(),),
                out_specs=P(),
                check_vma=False,
            )
        ),
    )


def input_specs(weights: Any, wk: Any) -> tuple[Any, ...]:
    import jax
    from jax.sharding import PartitionSpec as P

    return (
        P(None, "feature"),
        P(None, "feature"),
        P("expert", None, None, None),
        P("expert", None, None, None),
        P("expert", None, None, None),
        P(),
        P(),
        P(),
        P(),
        P(),
        P(),
        *jax.tree.map(
            lambda v: v.sharding.spec,
            (
                weights.qkv_a,
                weights.attention,
                weights.dsa,
                wk,
                weights.post_attention_norm_weight_local,
                weights.dense,
                weights.moe,
            ),
        ),
        P("expert", "feature", None),
        P(),
    )


def put_host_inputs(
    host: dict[str, np.ndarray], shardings: dict[str, Any]
) -> dict[str, Any]:
    """Bit-identity-checked fixture transfer, including intentional padded NaNs.

    JAX 0.10.1 global device_put uses np.equal across hosts (NaN != NaN).
    Authenticate shape/dtype/bytes collectively before local-shard callbacks.
    This is harness input initialization, outside all candidate programs.
    """
    import jax
    from jax.experimental import multihost_utils

    arrays = {name: np.array(host[name], copy=True) for name in shardings}
    identity = {
        name: dict(
            shape=value.shape,
            dtype=value.dtype.str,
            sha256=sha256(value.tobytes()).hexdigest(),
        )
        for name, value in arrays.items()
    }
    digest = sha256(json.dumps(identity, sort_keys=True).encode()).digest()
    multihost_utils.assert_equal(
        np.frombuffer(digest, dtype=np.uint8),
        fail_message="prefill fixture shape/dtype/byte identity differs across hosts",
    )
    return {
        name: jax.make_array_from_callback(
            value.shape, shardings[name], lambda index, value=value: value[index]
        )
        for name, value in arrays.items()
    }


def device_inputs(
    host: dict[str, np.ndarray],
    specs: tuple[Any, ...],
    weights: Any,
    wk: Any,
    mesh: Any,
) -> tuple[Any, ...]:
    import jax
    from jax.sharding import NamedSharding

    indices = (*range(11), 18, 19)
    device = put_host_inputs(
        host,
        {
            name: NamedSharding(mesh, specs[i])
            for name, i in zip(INPUT_FIELDS, indices, strict=True)
        },
    )
    return (
        *(device[name] for name in INPUT_FIELDS[:11]),
        weights.qkv_a,
        weights.attention,
        weights.dsa,
        wk,
        weights.post_attention_norm_weight_local,
        weights.dense,
        weights.moe,
        device["health"],
        device["rope"],
    )


def scalar_inputs(
    values: tuple[Any, ...], row: int, previous: tuple[Any, ...] | None = None
) -> tuple[Any, ...]:
    """Reference loop alone carries its own KV and unrepaired index writes."""
    import jax.numpy as jnp

    result = list(values)
    for i in (0, 1, 5, 6, 7, 19):
        result[i] = values[i][row : row + 1]
    result[8] = values[8] + jnp.int32(row)
    result[9] = jnp.asarray(1, jnp.int32)
    result[18] = values[18][:, :, row : row + 1]
    if previous is not None:
        result[2], result[3] = previous[2], previous[3]
    return tuple(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--coordinator-address", required=True)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    tag = os.environ.get("GLM_GREENFIELD_RUN_TAG", "")
    layer = layer_from_tag(tag)
    if (
        not 0 <= args.process_id < 8
        or REPO != Path("/home/gianl/glm-tpu-topology-rewrite")
        or not re.fullmatch(r"[0-9a-f]{40}", args.expected_code_hash)
        or _git_head() != args.expected_code_hash
        or args.output_dir
        != Path("/home/gianl/glm-run") / tag / f"rank{args.process_id}"
    ):
        raise ValueError("layer worker code/rank/path drifted")
    output = args.output_dir / "runner.json"
    if output.exists():
        raise FileExistsError(output)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.num_processes, args.slice_name = 8, "db-v4-64-od"
    args.topology_capture_root = TOPOLOGY
    args.topology_sha256, args.topology_fleet_sha256, args.mesh_sha256 = (
        TOPOLOGY_SHA,
        FLEET_SHA,
        MESH_SHA,
    )
    from scripts.greenfield import prefill_router_protocol as router_protocol

    diagnostic = router_protocol.is_router_tag(tag)
    materialized = materialized_ref.is_materialized_tag(tag)
    record = dict(
        status="RUNNING",
        protocol=(
            router_protocol.PROTOCOL
            if diagnostic
            else materialized_ref.PROTOCOL if materialized else PROTOCOL
        ),
        layer=layer,
        code_hash=args.expected_code_hash,
        launch_rank=args.process_id,
        hostname=socket.gethostname(),
        pid=os.getpid(),
        start_ticks=int(
            Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
        ),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        admission_only=not diagnostic,
        diagnostic_only=diagnostic,
        performance_claim=False,
        iterations=0,
        latency=None,
        rows=ROWS,
        cases={},
        phases={},
        programs={},
        reference_scope=(
            materialized_ref.REFERENCE_SCOPE
            if materialized
            else "RAW_SCALAR_NOT_PROMOTED_DECODER_OR_LEGACY"
        ),
        state_scope="REAL_WEIGHTS_SYNTHETIC_PREFIX_AND_ACTIVATIONS",
    )
    _atomic_json(output, record)

    def phase(name: str, start: float) -> None:
        record["phases"][name] = time.monotonic() - start
        _atomic_json(output, record)
        print(
            f"PREFILL_LAYER rank={args.process_id} layer={layer} phase={name}",
            flush=True,
        )

    try:
        from scripts.greenfield.run_short_decoder_ws32 import (
            _initialize_runtime,
            _geometry,
        )
        from glm_tpu.greenfield.checkpoint.ws32_layer_subset import (
            read_ws32_layer_subset_metadata,
            load_ws32_layer_subset,
        )
        from glm_tpu.greenfield.runtime.ws32_decoder import (
            Ws32DecoderConfig,
            ws32_decoder_weight_names,
            _bind_weight_name_tree,
        )
        from glm_tpu.greenfield.kernels.reference.rotary import build_rotary_table_host
        from scripts.greenfield.prefill_layer_programs import (
            build_layer_programs,
            build_repair_program,
        )

        started = time.monotonic()
        jax, mesh, physical, topology, fleet = _initialize_runtime(args)
        from jax.sharding import NamedSharding, PartitionSpec as P
        from jax.experimental import multihost_utils

        def consensus(passed: bool) -> bool:
            # Harness-only scalar consensus, OUTSIDE every model program.
            return bool(
                np.asarray(
                    multihost_utils.process_allgather(np.asarray(passed, np.int32))
                ).all()
            )

        slot_by_id = {
            int(device): slot
            for slot, device in enumerate(physical.flattened_device_ids)
        }
        local_slots = {int(d.id): slot_by_id[int(d.id)] for d in jax.local_devices()}
        preflight_bytes = (args.output_dir / "retained_preflight.json").read_bytes()
        preflight = json.loads(preflight_bytes)
        if (
            preflight["code_hash"] != args.expected_code_hash
            or preflight["layer"] != layer
            or preflight["launch_rank"] != args.process_id
            or preflight["hostname"] != socket.gethostname()
            or {h["device_slot"] for h in preflight["headers"]}
            != set(local_slots.values())
        ):
            raise ValueError(
                "preflight retained slots differ from actual runtime owners"
            )
        record["retained_preflight_sha256"] = sha256(preflight_bytes).hexdigest()
        if len(local_slots) != 4 or jax.default_backend() != "tpu":
            raise ValueError("layer admission requires four local TPU owners")
        record.update(
            jax_process_index=jax.process_index(),
            physical_device_ids=physical.device_ids,
            mesh_sha256=physical.mesh_hash,
            topology_sha256=topology.topology_hash,
            topology_fleet_sha256=fleet,
            versions={"jax": version("jax"), "libtpu": version("libtpu")},
        )
        phase("runtime_seconds", started)
        started = time.monotonic()
        pins = json.loads(PINS.read_text())
        inventory = authenticated_inventory(
            Path(pins["source_inventory"]), pins["source_inventory_sha256"]
        )
        for name, key in (
            ("manifest.json", "manifest_file_sha256"),
            ("SUCCESS", "success_file_sha256"),
        ):
            if (
                sha256((Path(pins["checkpoint_root"]) / name).read_bytes()).hexdigest()
                != pins[key]
            ):
                raise ValueError("retained runtime metadata file hash drifted")
        geometry = _geometry()
        config = Ws32DecoderConfig(geometry=geometry, context_capacity=1024)
        subset = read_ws32_layer_subset_metadata(
            Path(pins["checkpoint_root"]),
            layer_ids=(layer,),
            local_slots=tuple(local_slots.values()),
            max_payload_bytes_per_chip=PAYLOAD_BYTES[layer],
            expected_manifest_sha256=pins["expected_manifest_sha256"],
            expected_success_sha256=pins["expected_success_sha256"],
            expected_mesh_hash=MESH_SHA,
            expected_topology_hash=TOPOLOGY_SHA,
            inventory=inventory,
            geometry=geometry,
        )
        loaded = load_ws32_layer_subset(subset, mesh=mesh, physical_mesh=physical)
        weights = _bind_weight_name_tree(
            ws32_decoder_weight_names(config).layers[layer], loaded.arrays
        )
        record.update(
            local_device_slots=loaded.local_device_slots,
            integrity_scope=loaded.integrity_scope,
            payload_bytes_per_chip=loaded.payload_bytes_per_chip,
            checkpoint_pins=pins,
            selected_layer_ids=list(loaded.layer_ids),
        )
        phase("selected_load_seconds", started)
        from scripts.greenfield.prefill_router_protocol import is_router_tag

        if is_router_tag(tag):
            from scripts.greenfield.prefill_router_worker import execute_diagnostic

            execute_diagnostic(
                args=args,
                record=record,
                mesh=mesh,
                config=config,
                weights=weights,
                local_slots=local_slots,
                consensus=consensus,
            )
            record["device_memory_stats_including_reference"] = [
                dict(device_id=int(d.id), stats=_memory_stats(d))
                for d in jax.local_devices()
            ]
            record["status"] = "SUCCESS"
            _atomic_json(output, record)
            return 0
        wk = None
        if layer == 0:
            decode, promote = build_wk_programs(
                mesh,
                weights.dsa.wk_bits_local.sharding.spec,
                weights.dsa.wk_scale_local.sharding.spec,
                contract=config.dsa_contract,
            )
            decode_values = (weights.dsa.wk_bits_local, weights.dsa.wk_scale_local)
            decoded = compile_program(
                decode, decode_values, "wk_decode", args.output_dir, record
            )(*decode_values)
            jax.block_until_ready(decoded)
            wk = compile_program(
                promote, (decoded,), "wk_promote", args.output_dir, record
            )(decoded)
            jax.block_until_ready(wk)
            decoded.delete()
        specs = input_specs(weights, wk)
        batch, scalar = build_layer_programs(
            mesh,
            specs,
            full_indexer=layer == 0,
            sparse_mlp=layer == 3,
            dsa_contract=config.dsa_contract,
            attention_contract=config.attention_contract,
            moe_contract=config.moe_contract,
            rms_norm_epsilon=config.rms_norm_epsilon,
        )
        if materialized:
            reference_prefix_fn, scalar = materialized_ref.build_materialized_reference(
                mesh,
                specs,
                dsa_contract=config.dsa_contract,
                attention_contract=config.attention_contract,
                moe_contract=config.moe_contract,
                rms_norm_epsilon=config.rms_norm_epsilon,
            )
        table = build_rotary_table_host(1024, rotary_dim=64, theta=8e6)
        compiled = reference = reference_prefix = repair = None
        for case, (offset, count) in CASES.items():
            started = time.monotonic()
            host = host_case(case, table)
            arrays = encode_arrays("input", host)
            path = args.output_dir / f"{case}.npz"
            values = device_inputs(host, specs, weights, wk, mesh)
            if compiled is None:
                compiled = compile_program(
                    batch, values, "candidate", args.output_dir, record
                )
                record["hlo"] = dict(
                    sha256=record["programs"]["candidate"]["optimized_hlo_sha256"],
                    contract=check_layer_hlo(compiled.as_text(), layer=layer),
                )
                memory = record["programs"]["candidate"]["compiled_memory"]
                _atomic_json(output, record)
                if not record["hlo"]["contract"]["passed"]:
                    raise RuntimeError(
                        "complete-layer pre-execution HLO contract failed"
                    )
                if (
                    sum(
                        memory[n]
                        for n in (
                            "argument_size_in_bytes",
                            "output_size_in_bytes",
                            "temp_size_in_bytes",
                        )
                    )
                    > 2 * 1024**3
                ):
                    raise RuntimeError("candidate compiled allocation exceeds2GiB/chip")
                if materialized:
                    reference_prefix = compile_program(
                        reference_prefix_fn,
                        scalar_inputs(values, 0),
                        "reference_prefix",
                        args.output_dir,
                        record,
                    )
                    # Compile from the actual prefix output schema; no prefix execution
                    # is needed before admitting both distinct reference graphs.
                    shapes = jax.eval_shape(
                        reference_prefix_fn, *scalar_inputs(values, 0)
                    )
                    reference = compile_program(
                        scalar,
                        (shapes[0], shapes[1], values[15], values[17]),
                        "reference",
                        args.output_dir,
                        record,
                    )
                    for name in ("reference_prefix", "reference"):
                        proof = materialized_ref.check_reference_hlo(
                            (args.output_dir / f"{name}.optimized_hlo.txt").read_text(),
                            name,
                        )
                        record["programs"][name]["hlo_contract"] = proof
                        _atomic_json(output, record)
                        memory = record["programs"][name]["compiled_memory"]
                        if (
                            not proof["passed"]
                            or sum(
                                memory[n]
                                for n in (
                                    "argument_size_in_bytes",
                                    "output_size_in_bytes",
                                    "temp_size_in_bytes",
                                )
                            )
                            > 2 * 1024**3
                        ):
                            raise ValueError(
                                f"materialized reference HLO refuses {name}"
                            )
                else:
                    reference = compile_program(
                        scalar,
                        scalar_inputs(values, 0),
                        "reference",
                        args.output_dir,
                        record,
                    )
            returned = compiled(*values)
            jax.block_until_ready(returned)
            observed = local_observations(returned)
            by_device = {device: [] for device in local_slots}
            previous = None
            materialized_inputs = {d: [] for d in local_slots}
            for row in range(count):
                row_values = scalar_inputs(values, row, previous)
                if materialized:
                    prefix = reference_prefix(*row_values)
                    jax.block_until_ready(prefix)
                    mlp = reference(
                        prefix[0], prefix[1], row_values[15], row_values[17]
                    )
                    jax.block_until_ready(mlp)
                    previous = materialized_ref.assemble_reference_result(
                        row_values, prefix, mlp
                    )
                    for shard in prefix[0].addressable_shards:
                        materialized_inputs[int(shard.device.id)].append(
                            np.asarray(shard.data).copy()
                        )
                else:
                    previous = reference(*row_values)
                jax.block_until_ready(previous)
                for device, observation in local_observations(previous).items():
                    by_device[device].append(observation)
            expected = {
                device: stack_reference(rows) for device, rows in by_device.items()
            }
            if materialized:
                capture_path = args.output_dir / f"{case}.reference_input.npz"
                capture = {
                    f"device_{d}": np.concatenate(rows).view(np.uint16)
                    for d, rows in materialized_inputs.items()
                }
                np.savez_compressed(capture_path, **capture)
                record.setdefault("reference_input_sha256", {})[case] = sha256(
                    capture_path.read_bytes()
                ).hexdigest()
            keys = {"actual": {}, "reference": {}}
            if layer == 0:
                if repair is None:
                    repair_fn = build_repair_program(mesh, contract=config.dsa_contract)
                    repair = compile_program(
                        repair_fn,
                        (
                            returned[-1],
                            values[8],
                            values[9],
                            wk,
                            weights.dsa.key_norm_weight,
                            weights.dsa.key_norm_bias,
                        ),
                        "repair",
                        args.output_dir,
                        record,
                    )
                for kind, normalized in (("actual", returned[-1]), ("reference", None)):
                    if normalized is None:
                        # Declared observation roundtrip for this untimed diagnostic
                        # only; no host stage dispatch in the candidate executable.
                        sharding = NamedSharding(mesh, P(None, "feature"))
                        devices = sharding.addressable_devices_indices_map((ROWS, 6144))
                        normalized = jax.make_array_from_single_device_arrays(
                            (ROWS, 6144),
                            sharding,
                            [
                                jax.device_put(expected[int(d.id)]["normalized"], d)
                                for d in devices
                            ],
                        )
                    repaired = repair(
                        normalized,
                        values[8],
                        values[9],
                        wk,
                        weights.dsa.key_norm_weight,
                        weights.dsa.key_norm_bias,
                    )
                    jax.block_until_ready(repaired)
                    for shard in repaired.addressable_shards:
                        device = int(shard.device.id)
                        keys[kind][device] = np.asarray(shard.data).copy()
                        if kind == "reference":
                            for q, p, r in written_addresses(
                                slot=local_slots[device], offset=offset, count=count
                            ):
                                expected[device]["repair"][p, r] = keys[kind][device][q]
            for device in local_slots:
                for kind, observations in (
                    ("actual", observed),
                    ("reference", expected),
                ):
                    arrays.update(
                        encode_arrays(f"{kind}_{device}", observations[device])
                    )
                    if layer == 0:
                        arrays.update(
                            encode_arrays(
                                f"{kind}_{device}", {"m64": keys[kind][device]}
                            )
                        )
            np.savez_compressed(path, **arrays)
            base = {}
            try:
                for device, slot in local_slots.items():
                    base[str(device)] = compare_layer_case(
                        observed[device],
                        expected[device],
                        owner_inputs(host, slot),
                        slot=slot,
                        layer=layer,
                        case=case,
                        actual_m64_keys=keys["actual"].get(device),
                        reference_m64_keys=keys["reference"].get(device),
                    )
                passed = all(value["passed"] for value in base.values())
            except ValueError as exc:
                base["error"] = str(exc)
                passed = False
            record["cases"][case] = {"base": base, "input_sha256": input_hashes(host)}
            _atomic_json(output, record)
            if not consensus(passed):
                raise RuntimeError(f"first complete-layer arithmetic failure: {case}")
            for kind in METAMORPHIC[case]:
                changed = device_inputs(
                    mutate_case(host, kind), specs, weights, wk, mesh
                )
                changed_result = compiled(*changed)
                jax.block_until_ready(changed_result)
                changed_observed = local_observations(changed_result)
                for device in local_slots:
                    arrays.update(
                        encode_arrays(f"{kind}_{device}", changed_observed[device])
                    )
                np.savez_compressed(path, **arrays)
                from scripts.greenfield.prefill_layer_evidence import check_intervention

                results = {
                    str(d): check_intervention(
                        observed[d],
                        changed_observed[d],
                        mutate_case(host, kind),
                        slot=s,
                        kind=kind,
                    )
                    for d, s in local_slots.items()
                }
                record["cases"][case][kind] = results
                _atomic_json(output, record)
                if not consensus(all(r["passed"] for r in results.values())):
                    raise RuntimeError(
                        f"first complete-layer intervention failure: {case}/{kind}"
                    )
            replay = replay_case(
                path, layer=layer, case=case, slots_by_device=local_slots
            )
            record["cases"][case].update(
                replay=replay,
                passed=replay["passed"],
                npz_sha256=sha256(path.read_bytes()).hexdigest(),
            )
            if not consensus(replay["passed"]):
                raise RuntimeError(f"original-array replay failed: {case}")
            phase(f"{case}_seconds", started)
        record["device_memory_stats_including_reference"] = [
            dict(device_id=int(d.id), stats=_memory_stats(d))
            for d in jax.local_devices()
        ]
        record["status"] = "SUCCESS"
        _atomic_json(output, record)
        return 0
    except Exception as exc:
        record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        _atomic_json(output, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
