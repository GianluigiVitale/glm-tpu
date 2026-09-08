"""Fixed layer6 numerical execution inside the existing protected worker.

No entry point, TPU initialization, deployment or performance promotion. The
campaign must independently replay originals and authenticate all eight hosts.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np

from glm_tpu.greenfield.validation.ws32_prefill_memory import (
    MEMORY_FIELDS,
    capture_identified_device_memory,
    capture_resident_buffers,
)
from scripts.greenfield import prefill_window_admission as admission
from scripts.greenfield import prefill_window_protocol as protocol
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from scripts.greenfield.prefill_layer_evidence import encode_arrays, local_observations
from scripts.greenfield.ws32_acquisition_journal import Ws32NumericalJournal


class WindowNumericalJournal(Ws32NumericalJournal):
    artifact_kind = "greenfield_ws32_layer_window_numerical_journal"

    def _check_identity(self, identity: Mapping[str, Any]) -> None:
        if (
            identity.get("protocol") != protocol.PROTOCOL
            or identity.get("profile") != admission.PROFILE
            or identity.get("compile_only") is not False
        ):
            raise ValueError("window journal requires fixed numerical identity")


def save_arrays(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    """Atomically replace only this run's evolving capture; preserve on failure."""
    temporary = path.with_suffix(".npz.pending")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return sha256(path.read_bytes()).hexdigest()


def validate_memory_owners(
    rows: list[dict[str, Any]], *, local_slots: Mapping[int, int], process_index: int
) -> None:
    """Join keyed counters to this already-authenticated process/physical owners."""
    if (
        len(local_slots) != 4
        or len(set(local_slots.values())) != 4
        or any(type(s) is not int or not 0 <= s < 32 for s in local_slots.values())
        or len(rows) != 4
        or {r["device_id"] for r in rows} != set(local_slots)
        or any(
            r["platform"] != "tpu" or r["process_index"] != process_index for r in rows
        )
    ):
        raise ValueError("window memory does not cover its four authenticated owners")


class BudgetedCalls:
    """Local phases vote before dispatch; completed calls vote before successors.

    This cannot recover a distributed executable whose peers hang internally;
    the existing owned worker timeout and controller recovery remain mandatory.
    """

    def __init__(
        self,
        *,
        root: Path,
        record: dict[str, Any],
        consensus: Callable[[bool], bool],
        journal: WindowNumericalJournal,
        local_slots: Mapping[int, int],
    ) -> None:
        self.root, self.record = root, record
        self.consensus, self.journal = consensus, journal
        self.local_slots = local_slots
        self.programs: dict[str, Any] = {}
        self.record["call_evidence"] = []

    def phase(self, name: str, action: Callable[[], Any]) -> Any:
        value, error = None, None
        try:
            value = action()
        except Exception as exc:
            error = exc
        try:
            self.record["current_phase"] = name
            if error is not None:
                self.record["phase_error"] = f"{name}: {type(error).__name__}: {error}"
            _atomic_json(self.root / "runner.json", self.record)
            self.journal.phase(name, passed=error is None)
        except Exception as exc:
            error = error or exc
        agreed = self.consensus(error is None)
        if error is not None:
            raise error
        if not agreed:
            raise RuntimeError(f"window numerical peer refused at {name}")
        return value

    def call(
        self,
        phase: str,
        name: str,
        values: tuple[Any, ...],
        *,
        preserve: Callable[[Any], None],
    ) -> Any:
        import jax

        entry: dict[str, Any] = dict(phase=phase, graph=name, completed=False)
        self.record["call_evidence"].append(entry)

        def preflight():
            # Actual analyses, not just the acquired receipt. Capture all live
            # arrays, including candidate results and old control cache aliases.
            analyses = {}
            for n, program in self.programs.items():
                analysis = program.memory_analysis()
                analyses[n] = {key: getattr(analysis, key) for key in MEMORY_FIELDS}
            census = capture_resident_buffers(
                {"active_inputs": values}, devices=tuple(jax.local_devices())
            )
            validate_memory_owners(
                census["devices"],
                local_slots=self.local_slots,
                process_index=int(self.record["jax_process_index"]),
            )
            budget = admission.memory_budget(census, analyses, active_graph=name)
            entry.update(census=census, compiled_memory=analyses, budget=budget)
            if not budget["estimate_fits"]:
                raise ValueError("window predispatch memory reserve failed")

        self.phase(phase + "/memory", preflight)

        def execute():
            # No host comparison, report, census or fleet vote inside interval.
            started = time.monotonic()
            result = self.programs[name](*values)
            jax.block_until_ready(result)
            entry.update(
                completed=True, completed_call_seconds=time.monotonic() - started
            )
            # Save completed originals before fallible phase publication or
            # post-memory validation can prevent the caller from receiving them.
            # Persistence/comparison is OUTSIDE the diagnostic call interval.
            preserve(result)
            return result

        result = self.phase(phase + "/execute", execute)

        def postflight():
            stats = capture_identified_device_memory(tuple(jax.local_devices()))
            entry["post_memory"] = stats
            validate_memory_owners(
                stats,
                local_slots=self.local_slots,
                process_index=int(self.record["jax_process_index"]),
            )
            before = {
                r["device_id"]: r["memory_stats"] for r in entry["census"]["devices"]
            }
            if any(
                r["bytes_limit"] != before[r["device_id"]]["bytes_limit"]
                or r["peak_bytes_in_use"] < before[r["device_id"]]["peak_bytes_in_use"]
                or r["bytes_limit"] - r["peak_bytes_in_use"]
                < admission.REQUIRED_RESERVE_BYTES
                for r in stats
            ):
                raise ValueError("window post-call memory reserve/counters failed")

        self.phase(phase + "/memory_after", postflight)
        return result


def execute_cases(
    calls: BudgetedCalls, *, weights: Any, wk: Any, mesh: Any, specs: tuple[Any, ...]
) -> None:
    """Three fixed original-array cases; no timing loop or mutable-state reuse."""
    from glm_tpu.greenfield.kernels.reference.rotary import build_rotary_table_host
    from scripts.greenfield.probe_ws32_prefill_layer import device_inputs
    import jax

    rope = calls.phase(
        "rotary_fixture",
        lambda: build_rotary_table_host(protocol.CAPACITY, rotary_dim=64, theta=8e6),
    )
    for case in protocol.CASES:
        host = calls.phase(case + "/host", lambda: protocol.host_case(case, rope))
        values = calls.phase(
            case + "/inputs", lambda: device_inputs(host, specs, weights, wk, mesh)
        )
        calls.phase(case + "/inputs_ready", lambda: jax.block_until_ready(values))
        path = calls.root / f"{case}.npz"
        arrays = encode_arrays("input", host)
        case_record: dict[str, Any] = dict(complete=False)
        calls.record["cases"][case] = case_record
        calls.phase(case + "/input_capture", lambda: save_arrays(path, arrays))

        def capture(kind, result):
            observed = local_observations(result)
            if set(observed) != set(calls.local_slots):
                raise ValueError("window output owners differ")
            for device, fields in observed.items():
                arrays.update(encode_arrays(f"{kind}_{device}", fields))
            case_record["npz_sha256"] = save_arrays(path, arrays)
            if not all(v["health"].all() for v in observed.values()):
                raise ValueError("window output health failed; originals preserved")
            return observed

        actual = calls.call(
            case + "/candidate",
            "candidate",
            values,
            preserve=lambda result: capture("actual", result),
        )
        previous = None
        controls = []
        for tile in range(4):
            inputs = calls.phase(
                case + f"/control{tile}_inputs",
                lambda: protocol.control_inputs(values, tile, previous),
            )
            calls.phase(
                case + f"/control{tile}_ready", lambda: jax.block_until_ready(inputs)
            )
            previous = calls.call(
                case + f"/control{tile}",
                "control",
                inputs,
                preserve=lambda result: controls.append(capture(f"tile{tile}", result)),
            )

        def finish():
            final = {
                key: value
                for key, value in arrays.items()
                if not key.startswith("tile")
            }
            for device in calls.local_slots:
                final.update(
                    encode_arrays(
                        f"control_{device}",
                        protocol.stack_control([c[device] for c in controls]),
                    )
                )
            case_record["npz_sha256"] = save_arrays(path, final)
            # Persist all outputs BEFORE the first fallible comparison.
            result = protocol.replay_case(
                path, case=case, slots_by_device=calls.local_slots
            )
            case_record.update(replay=result, passed=result["passed"], complete=True)
            if not result["passed"]:
                raise ValueError("window/control numerical comparison failed")

        calls.phase(case + "/comparison", finish)
        # These are read-only outputs; do not donate/delete aliases held by values.
        # Drop Python references before constructing the next independent fixture.
        del values, actual, previous, inputs, controls, arrays, host


def execute_numerical(
    *,
    args: Any,
    record: dict[str, Any],
    mesh: Any,
    config: Any,
    weights: Any,
    local_slots: Mapping[int, int],
    consensus: Callable[[bool], bool],
) -> None:
    """Compile/admit four programs, then completed WK and fixed numerical cases."""
    from scripts.greenfield.prefill_window_acquisition import (
        prepare_programs,
        fleet_step,
    )
    from scripts.greenfield.probe_ws32_prefill_layer import compile_program, input_specs

    root = args.output_dir
    record.update(
        protocol=protocol.PROTOCOL,
        profile=admission.PROFILE,
        compile_only=False,
        iterations=0,
        performance_claim=False,
        cases={},
        programs={},
    )
    journal = fleet_step(
        "numerical_journal",
        lambda: WindowNumericalJournal(
            root / "compile_journal.jsonl",
            dict(
                protocol=protocol.PROTOCOL,
                profile=admission.PROFILE,
                compile_only=False,
                code_hash=record["code_hash"],
                launch_rank=record["launch_rank"],
            ),
        ),
        record=record,
        root=root,
        consensus=consensus,
    )
    calls = BudgetedCalls(
        root=root,
        record=record,
        consensus=consensus,
        journal=journal,
        local_slots=local_slots,
    )
    try:
        prepared = calls.phase(
            "prepare",
            lambda: prepare_programs(mesh=mesh, config=config, weights=weights),
        )
        for name, fn, values in prepared:

            def compile_one():
                compiled = compile_program(
                    fn, values, name, root, record, journal=journal
                )
                stable = (root / f"{name}.stablehlo.mlir").read_text()
                hlo = (root / f"{name}.optimized_hlo.txt").read_text()
                report = journal.inspect(
                    name,
                    stable,
                    hlo,
                    lambda: admission.inspect_program(
                        name, stable, hlo, record["programs"][name]["compiled_memory"]
                    ),
                )
                record["programs"][name]["admission"] = report
                return compiled

            calls.programs[name] = calls.phase("compile/" + name, compile_one)
        del prepared, fn, values

        def preserve_wk(name, result):
            shards = {
                int(s.device.id): np.asarray(s.data) for s in result.addressable_shards
            }
            arrays = {
                str(device): (
                    value.view(np.uint16) if value.dtype == protocol.BF16 else value
                )
                for device, value in shards.items()
            }
            record.setdefault("wk_originals", {})[name] = save_arrays(
                root / f"{name}.npz", arrays
            )
            if set(shards) != set(local_slots):
                raise ValueError("WK original output owners differ")

        decoded = calls.call(
            "wk_decode",
            "wk_decode",
            (weights.dsa.wk_bits_local, weights.dsa.wk_scale_local),
            preserve=lambda result: preserve_wk("wk_decode", result),
        )
        wk = calls.call(
            "wk_promote",
            "wk_promote",
            (decoded,),
            preserve=lambda result: preserve_wk("wk_promote", result),
        )

        def capture_wk():
            source = {
                int(s.device.id): np.asarray(s.data) for s in decoded.addressable_shards
            }
            target = {
                int(s.device.id): np.asarray(s.data) for s in wk.addressable_shards
            }
            if set(source) != set(local_slots) or set(target) != set(source):
                raise ValueError("WK boundary owners differ")
            arrays = {}
            for device in source:
                arrays[f"bf16_{device}"] = source[device].view(np.uint16)
                arrays[f"fp32_{device}"] = target[device]
            record["wk_boundary_sha256"] = save_arrays(root / "wk_boundary.npz", arrays)
            for device in source:
                if (
                    source[device].shape != (128, 6144)
                    or source[device].dtype != protocol.BF16
                    or target[device].dtype != np.float32
                    or not np.isfinite(source[device]).all()
                    or not np.array_equal(
                        source[device].astype(np.float32), target[device]
                    )
                ):
                    raise ValueError("WK completed BF16-to-FP32 boundary failed")

        calls.phase("wk_boundary", capture_wk)
        del decoded
        specs = calls.phase("input_specs", lambda: input_specs(weights, wk))
        execute_cases(calls, weights=weights, wk=wk, mesh=mesh, specs=specs)
        record["model_executable_calls"] = sum(
            e["completed"]
            for e in record["call_evidence"]
            if e["graph"] in ("candidate", "control")
        )
        record["wk_executable_calls"] = sum(
            e["completed"]
            for e in record["call_evidence"]
            if e["graph"].startswith("wk_")
        )
        calls.phase("numerical_complete", lambda: None)
    finally:
        journal.close()
        record["compile_journal_sha256"] = sha256(
            (root / "compile_journal.jsonl").read_bytes()
        ).hexdigest()
        _atomic_json(root / "runner.json", record)
