"""Equal-work MoE baseline implementation inside the existing guarded worker.

No entry point or standalone launch. The existing worker authenticates runtime,
checkpoint and oracle; the existing bounded controller owns leases and cleanup.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from scripts.greenfield import prefill_moe_scaling as spec
from scripts.greenfield.microbench_fp8_matmul import (
    _atomic_json,
    _compiled_memory,
    _memory_stats,
)
from scripts.greenfield.probe_ws32_prefill_moe import (
    CASES,
    build_mapped,
    case_rows,
    check_hlo,
)


def run_scaling(
    args: Any,
    record: dict[str, Any],
    *,
    jax: Any,
    mesh: Any,
    physical_mesh: Any,
    loaded: Any,
    oracle: Any,
) -> None:
    """Compile twice, verify equal work, then measure fixed supplied scenarios."""
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding, PartitionSpec as P
    from glm_tpu.greenfield.benchmarking.ws32_one_layer import ws32_one_layer_inputs
    from glm_tpu.greenfield.kernels.reference.moe import GlmMoeNumericalContract
    from scripts.greenfield.run_real_one_layer_ws32 import _bfloat16_numpy

    root: Path = args.output_dir
    output = root / "runner.json"
    slot_by_device = {d: s for s, d in enumerate(physical_mesh.flattened_device_ids)}

    def consensus(ok: bool) -> bool:
        return bool(
            np.asarray(
                multihost_utils.process_allgather(np.asarray(ok, np.int32))
            ).all()
        )

    def guarded(phase: str, fn: Callable[[], Any], *, journal: bool = True) -> Any:
        # Every peer executes exactly one vote after this same local phase.
        error = None
        result = None
        try:
            result = fn()
        except Exception as exc:
            error = exc
            record["phase_error"] = f"{phase}: {type(exc).__name__}: {exc}"
        record["current_phase"] = phase
        try:
            if journal or error is not None:
                _atomic_json(output, record)
        except Exception as exc:
            error = error or exc
        if not consensus(error is None):
            raise RuntimeError(f"MoE scaling fleet refused at {phase}") from error
        return result

    candidate, scalar = build_mapped(
        mesh,
        contract=GlmMoeNumericalContract(stage_size=8),
        fp32_route_sum=True,
    )
    metadata = jax.jit(
        spec.device_active_tiles,
        in_shardings=NamedSharding(mesh, P()),
        out_shardings=NamedSharding(mesh, P()),
    )
    programs: dict[str, Any] = {}
    record["programs"] = {}
    remaining = spec.PHASE_BUDGET_SECONDS

    for case in CASES:
        started = time.monotonic()

        def prepare():
            host = case_rows(
                _bfloat16_numpy(oracle["hidden_states"]),
                oracle[f"{case}_route_indices"].numpy(),
                oracle[f"{case}_route_weights"].numpy(),
                case,
                rows=spec.WINDOW_ROWS,
            )
            values = tuple(
                jax.device_put(v, NamedSharding(mesh, s))
                for v, s in zip(
                    host,
                    (P(None, "feature"), P(), P()),
                    strict=True,
                )
            )
            inputs = ws32_one_layer_inputs(*values, loaded.arrays)
            small = spec.split_equal_work(inputs)
            # Preplace/complete ALL small inputs before correctness or timing.
            singles = tuple(
                tuple(v[r : r + 1] for v in inputs[:3]) + tuple(inputs[3:])
                for r in range(spec.WINDOW_ROWS)
            )
            jax.block_until_ready((inputs, small, singles))
            return host, inputs, small, singles

        host, wide_inputs, small_inputs, scalar_inputs = guarded(
            f"{case}/inputs", prepare
        )
        record["cases"][case] = dict(
            input_sha256=[sha256(v.tobytes()).hexdigest() for v in host]
        )
        case_record = record["cases"][case]

        if not programs:
            for name, program, values, rows in (
                ("b16", candidate, small_inputs[0], spec.CONTROL_ROWS),
                ("b128", candidate, wide_inputs, spec.WINDOW_ROWS),
                ("scalar", scalar, scalar_inputs[0], 1),
            ):

                def compile_one():
                    lower = program.lower(*values)
                    stable = str(lower.compiler_ir(dialect="stablehlo"))
                    (root / f"{name}.stablehlo.mlir").write_text(stable)
                    compiled = lower.compile()
                    hlo = compiled.as_text()
                    (root / f"{name}.optimized_hlo.txt").write_text(hlo)
                    memory = _compiled_memory(compiled)
                    facts = dict(
                        rows=rows,
                        sha256=sha256(hlo.encode()).hexdigest(),
                        stablehlo_sha256=sha256(stable.encode()).hexdigest(),
                        compiled_memory_estimate=memory,
                    )
                    record["programs"][name] = facts
                    _atomic_json(output, record)  # preserve analysis even on refusal
                    if (
                        any(
                            type(memory[n]) is not int or memory[n] < 0
                            for n in (
                                "argument_size_in_bytes",
                                "output_size_in_bytes",
                                "temp_size_in_bytes",
                            )
                        )
                        or sum(
                            memory[n]
                            for n in (
                                "argument_size_in_bytes",
                                "output_size_in_bytes",
                                "temp_size_in_bytes",
                            )
                        )
                        > spec.COMPILED_MEMORY_LIMIT_BYTES
                    ):
                        raise ValueError(
                            "MoE baseline compiled allocation exceeds1GiB/chip"
                        )
                    if rows != 1:
                        facts["contract"] = check_hlo(
                            hlo, fp32_route_sum=True, rows=rows
                        )
                        if not facts["contract"]["passed"]:
                            raise ValueError(f"{name} HLO contract refused")
                    return compiled

                programs[name] = guarded(f"compile/{name}", compile_one)
            record["hlo"] = record["programs"]["b128"]
            record["phases"]["prepare_compile_seconds"] = time.monotonic() - started

        def completed(program, values):
            result = program(*values)
            jax.block_until_ready(result)
            return result

        wide = guarded(
            f"{case}/b128_correctness", lambda: completed(programs["b128"], wide_inputs)
        )
        small = [
            guarded(
                f"{case}/b16_correctness/{i}", lambda v=v: completed(programs["b16"], v)
            )
            for i, v in enumerate(small_inputs)
        ]
        references = [
            guarded(
                f"{case}/scalar/{i}",
                lambda v=v: completed(programs["scalar"], v),
                journal=(i % 16 == 0 or i == spec.WINDOW_ROWS - 1),
            )
            for i, v in enumerate(scalar_inputs)
        ]

        def capture():
            tensors = {
                "hidden": host[0].view(np.uint16),
                "routes": host[1],
                "weights": host[2],
            }
            counts = []
            for values in (wide_inputs, *small_inputs):
                tiles = metadata(values[1])
                jax.block_until_ready(tiles)
                counts.append(np.asarray(tiles))
            tiles = np.stack(counts)
            tensors["active_tiles"] = tiles
            occupancy = [
                spec.occupancy(ids, counts[i])
                for i, ids in enumerate(
                    (host[1], *[v[1] for v in spec.split_equal_work(host)])
                )
            ]
            case_record["occupancy"] = dict(b128=occupancy[0], b16=occupancy[1:])
            outputs = {
                int(s.device.id): np.asarray(s.data) for s in wide[0].addressable_shards
            }
            health = {
                int(s.device.id): np.asarray(s.data) for s in wide[1].addressable_shards
            }
            controls = [
                {int(s.device.id): np.asarray(s.data) for s in v[0].addressable_shards}
                for v in small
            ]
            controls_health = [
                {int(s.device.id): np.asarray(s.data) for s in v[1].addressable_shards}
                for v in small
            ]
            refs = [
                {int(s.device.id): np.asarray(s.data) for s in v.addressable_shards}
                for v in references
            ]
            legacy = _bfloat16_numpy(oracle[f"{case}_output"])
            shards = []
            for device, actual in outputs.items():
                feature = slot_by_device[device] % 4
                control = np.concatenate([v[device] for v in controls])
                ref = np.concatenate([v[device] for v in refs])
                bounded = spec.compare_equal_work(
                    actual,
                    control,
                    ref,
                    legacy[:, feature * 1536 : (feature + 1) * 1536],
                )
                facts = dict(
                    device_id=device,
                    device_slot=slot_by_device[device],
                    healthy=bool(health[device].all())
                    and all(bool(v[device].all()) for v in controls_health),
                    bounded_comparison=bounded,
                    sha256={},
                )
                for name, value in (
                    ("wide", actual),
                    ("control", control),
                    ("scalar", ref),
                ):
                    tensors[f"{name}_{device}"] = value.view(np.uint16)
                    facts["sha256"][name] = sha256(value.tobytes()).hexdigest()
                for name, value in (
                    ("wide_health", health[device]),
                    ("control_health", np.stack([v[device] for v in controls_health])),
                ):
                    tensors[f"{name}_{device}"] = value
                    facts["sha256"][name] = sha256(value.tobytes()).hexdigest()
                shards.append(facts)
            np.savez_compressed(root / f"{case}.npz", **tensors)
            case_record["shards"] = shards
            case_record["passed"] = len(shards) == 4 and all(
                s["healthy"] and s["bounded_comparison"]["passed"] for s in shards
            )
            if not case_record["passed"]:
                raise ValueError(f"{case} numerical/health comparison failed")

        guarded(f"{case}/numerical_and_occupancy", capture)
        case_record["fleet_passed"] = True
        case_record["timing"] = {}
        for name, values in (("b16", small_inputs), ("b128", (wide_inputs,))):
            last = []

            def complete(result):
                jax.block_until_ready(result)
                last[:] = [result]

            started = time.monotonic()
            # Do NOT wrap this in guarded: it owns phase-matched per-call votes.
            case_record["timing"][name] = spec.measure_completed_calls(
                programs[name],
                values,
                complete=complete,
                consensus=consensus,
                deadline=started + remaining,
            )
            remaining -= time.monotonic() - started

            def postcheck():
                expected = wide if name == "b128" else small[-1]
                for actual, original in zip(last[0], expected, strict=True):
                    originals = {
                        int(s.device.id): np.asarray(s.data)
                        for s in original.addressable_shards
                    }
                    for shard in actual.addressable_shards:
                        if not np.array_equal(
                            np.asarray(shard.data), originals[int(shard.device.id)]
                        ):
                            raise ValueError(
                                "timed output or health changed from verified result"
                            )
                case_record["timing"][name]["postcheck_passed"] = True
                record["device_memory_stats_including_reference"] = [
                    dict(device_id=int(d.id), stats=_memory_stats(d))
                    for d in jax.local_devices()
                ]

            guarded(f"{case}/{name}/postcheck", postcheck)
        print(
            f"PREFILL_MOE_SCALING rank={args.process_id} case={case} complete",
            flush=True,
        )
    record.update(
        status="SUCCESS", measured_phase_budget_seconds=spec.PHASE_BUDGET_SECONDS
    )
    _atomic_json(output, record)
