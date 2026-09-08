"""Original-array and runtime-memory replay for the fixed layer6 discriminator.

The existing campaign must separately bind run/source/checkpoint/fleet identities
and generation-qualified files. This module grants no launch or promotion authority.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from scripts.greenfield import prefill_window_admission as admission
from scripts.greenfield import prefill_window_protocol as protocol
from scripts.greenfield.prefill_window_worker import (
    WindowNumericalJournal,
    validate_memory_owners,
)


def same_json(actual: Any, expected: Any, description: str) -> None:
    """Compare full serialized evidence, including nested bool/int/float types."""
    encoded = json.dumps(actual, sort_keys=True, allow_nan=False)
    if actual != json.loads(encoded) or encoded != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise ValueError(f"window {description} differs from original replay")


def validate_workers(
    records: list[dict[str, Any]],
    *,
    pin: str,
    pins: dict[str, Any],
    ledger: dict[int, Any],
    order: tuple[int, ...],
    boundary_diagnostic: bool = False,
    completed_numerical: bool = False,
    phase_baseline: bool = False,
) -> None:
    """Join numerical memory/arrays to the existing selected32-owner ledger.

    The campaign verifies unique hosts/processes, physical mesh order and all
    four cross-host graph hashes before calling this mode-specific consumer.
    """
    from scripts.greenfield.probe_ws32_prefill_moe import (
        FLEET_SHA,
        MESH_SHA,
        TOPOLOGY_SHA,
    )

    if sum((boundary_diagnostic, completed_numerical, phase_baseline)) > 1:
        raise ValueError("window numerical modes are exclusive")
    selected_admission = admission
    names = admission.PROGRAMS
    if boundary_diagnostic:
        from scripts.greenfield import (
            prefill_window_boundary_admission as selected_admission,
        )
        from scripts.greenfield import prefill_window_boundary_worker as boundary
    if completed_numerical or phase_baseline:
        from scripts.greenfield import (
            prefill_completed_window_admission as selected_admission,
        )
        from scripts.greenfield import prefill_completed_window_protocol as completed
        from scripts.greenfield.prefill_completed_window_assembly import (
            PROGRAMS as helpers,
        )

        names = (*selected_admission.PROGRAMS, *helpers)
    if phase_baseline:
        from scripts.greenfield import prefill_phase_baseline as phase
        from scripts.greenfield import prefill_phase_originals as originals
    slots = []
    for record in records:
        exact = dict(
            status="SUCCESS",
            protocol=protocol.PROTOCOL,
            profile=admission.PROFILE,
            layer=6,
            code_hash=pin,
            selected_layer_ids=[6],
            rows=128,
            control_rows=32,
            context_capacity=4096,
            key_tile=512,
            iterations=0,
            latency=None,
            reference_scope=protocol.REFERENCE_SCOPE,
            state_scope="REAL_WEIGHTS_SYNTHETIC_PREFIX_AND_ACTIVATIONS",
            integrity_scope="selected_layer_tensors_only_not_complete_checkpoint",
            checkpoint_pins=pins,
            payload_bytes_per_chip=protocol.PAYLOAD_BYTES_PER_CHIP,
            mesh_sha256=MESH_SHA,
            topology_sha256=TOPOLOGY_SHA,
            topology_fleet_sha256=FLEET_SHA,
            versions={"jax": "0.10.1", "libtpu": "0.0.41"},
            model_executable_calls=15,
            wk_executable_calls=2,
            compile_only=False,
            numerical_execution_authorized=True,
            admission_only=True,
            diagnostic_only=False,
            performance_claim=False,
            current_phase="numerical_complete",
        )
        if boundary_diagnostic:
            exact.update(
                protocol=boundary.PROTOCOL,
                profile=selected_admission.PROFILE,
                reference_scope=boundary.REFERENCE_SCOPE,
                model_executable_calls=5,
                admission_only=False,
                diagnostic_only=True,
                current_phase="boundary_complete",
                boundary_capture_complete=True,
                numerical_admission=False,
            )
        if completed_numerical:
            exact.update(
                protocol=completed.PROTOCOL,
                profile=selected_admission.PROFILE,
                reference_scope=completed.REFERENCE_SCOPE,
                independent_full_layer_admission=False,
                model_executable_calls=27,
                assembly_executable_calls=30,
            )
        if phase_baseline:
            exact.update(
                protocol=phase.PROTOCOL,
                profile=selected_admission.PROFILE,
                reference_scope=phase.SCOPE,
                independent_full_layer_admission=False,
                model_executable_calls=135,
                assembly_executable_calls=150,
                admission_only=False,
                diagnostic_only=True,
                current_phase="phase_numerical_complete",
            )
        same_json({k: record.get(k) for k in exact}, exact, "worker scope/provenance")
        if not all(
            type(record.get(k)) is int and record[k] > 0 for k in ("pid", "start_ticks")
        ) or not record.get("boot_id"):
            raise ValueError("window numerical process identity missing")
        local = record["local_device_slots"]
        if len(local) != 4 or len({s["device_slot"] for s in local}) != 4:
            raise ValueError("window numerical needs four distinct local owners")
        for s in local:
            slot = s["device_slot"]
            if type(slot) is not int or not 0 <= slot < 32:
                raise ValueError("window numerical selected slot invalid")
            expected = dict(
                device_id=int(order[slot]),
                observed_selected_tensor_sha256=ledger[slot]["selected"],
                expected_full_file_sha256_not_verified=ledger[slot]["full_sha256"],
                selected_payload_bytes=protocol.PAYLOAD_BYTES_PER_CHIP,
            )
            same_json({k: s.get(k) for k in expected}, expected, "selected bytes/owner")
            slots.append(slot)
        if set(record["programs"]) != set(names) or set(record["cases"]) != (
            set() if boundary_diagnostic or phase_baseline else set(protocol.CASES)
        ):
            raise ValueError("window numerical program/case inventory differs")
        if phase_baseline:
            same_json(
                record["original_binding"],
                originals.bind_originals(
                    record,
                    {s["device_id"]: s["device_slot"] for s in local},
                    originals.load_capsule(),
                ),
                "phase original owners",
            )
            # Full streamed budget replay happens in validate_files after download.
            if (
                len(record["phase_call_index"]) != phase.MAX_WITNESS_CALLS
                or record["call_evidence"] != []
            ):
                raise ValueError("phase completed call inventory differs")
        else:
            validate_calls(
                record,
                local_slots={s["device_id"]: s["device_slot"] for s in local},
                boundary_diagnostic=boundary_diagnostic,
                completed_numerical=completed_numerical,
            )
        same_json(
            record["hlo"],
            dict(
                sha256=record["programs"]["candidate"]["optimized_hlo_sha256"],
                contract=dict(passed=True, profile=selected_admission.PROFILE),
            ),
            "numerical HLO identity",
        )
        if boundary_diagnostic:
            same_json(
                record["original_binding"],
                boundary.bind_originals(
                    record, {s["device_id"]: s["device_slot"] for s in local}
                ),
                "original fingerprint binding",
            )
        for case in () if boundary_diagnostic or phase_baseline else protocol.CASES:
            value = record["cases"][case]
            if (
                value.get("passed") is not True
                or value.get("complete") is not True
                or value["replay"].get("passed") is not True
            ):
                raise ValueError("window numerical comparison incomplete/failed")
            if set(value["replay"]["owners"]) != {str(s["device_id"]) for s in local}:
                raise ValueError("window numerical array owners differ")
    if len(slots) != 32 or set(slots) != set(range(32)):
        raise ValueError("window numerical requires32 distinct selected owners")


def validate_record(
    record: dict[str, Any], pin: str, *, completed_numerical: bool = False
) -> None:
    """Existing wrapper's untimed numerical-only DB classification check."""
    from scripts.greenfield.ws32_prefill_layer_campaign import (
        checkpoint_ledger,
        validate_workers as validate_fleet,
    )

    exact = dict(
        status="SUCCESS",
        kernel=protocol.KERNEL,
        protocol=protocol.PROTOCOL,
        profile=admission.PROFILE,
        code_hash=pin,
        layer=6,
        latency=None,
        warmup=0,
        iterations=0,
        compile_only=False,
        numerical_execution_authorized=True,
        admission_only=True,
        baseline_only=False,
        diagnostic_only=False,
        performance_claim=False,
        profiler_free_timing=False,
        comparison=dict(passed=True, diagnostic_evidence_complete=False),
    )
    if completed_numerical:
        from scripts.greenfield import prefill_completed_window_protocol as completed
        from scripts.greenfield.prefill_completed_window_admission import PROFILE

        exact.update(
            kernel=completed.KERNEL,
            protocol=completed.PROTOCOL,
            profile=PROFILE,
            reference_scope=completed.REFERENCE_SCOPE,
            independent_full_layer_admission=False,
        )
    same_json({k: record.get(k) for k in exact}, exact, "aggregate numerical scope")
    pins, ledger = checkpoint_ledger(6)
    validate_fleet(
        record["workers"],
        pin,
        layer=6,
        pins=pins,
        ledger=ledger,
        window_numerical=not completed_numerical,
        completed_numerical=completed_numerical,
    )
    same_json(record["hlo"], record["workers"][0]["hlo"], "aggregate HLO")
    if (
        record["checksum"]
        != sha256(json.dumps(record["workers"], sort_keys=True).encode()).hexdigest()
    ):
        raise ValueError("window aggregate worker digest differs")


def expected_calls(
    *, boundary_diagnostic: bool = False, completed_numerical: bool = False
) -> tuple[tuple[str, str], ...]:
    if boundary_diagnostic and completed_numerical:
        raise ValueError("window numerical modes are exclusive")
    if completed_numerical:
        return (
            ("wk_decode", "wk_decode"),
            ("wk_promote", "wk_promote"),
            *(
                (case + "/" + phase, name)
                for case in protocol.CASES
                for phase, name in (
                    *(
                        pair
                        for tile in range(4)
                        for pair in (
                            (f"prepare_prefix{tile}", "prepare_prefix"),
                            (f"prefix{tile}", "prefix"),
                        )
                    ),
                    ("prepare_wide", "prepare_wide"),
                    ("wide", "candidate"),
                    *(
                        pair
                        for tile in range(4)
                        for pair in (
                            (f"prepare_narrow{tile}", "prepare_narrow"),
                            (f"narrow{tile}", "control"),
                        )
                    ),
                    ("assemble", "assemble"),
                )
            ),
        )
    return (
        ("wk_decode", "wk_decode"),
        ("wk_promote", "wk_promote"),
        *(
            (phase, name)
            for case in (("boundary",) if boundary_diagnostic else protocol.CASES)
            for phase, name in (
                (case + "/candidate", "candidate"),
                *((case + f"/control{tile}", "control") for tile in range(4)),
            )
        ),
    )


def validate_calls(
    record: Mapping[str, Any],
    *,
    local_slots: Mapping[int, int],
    boundary_diagnostic: bool = False,
    completed_numerical: bool = False,
) -> None:
    """Recompute every predispatch budget and bind all post-call lifetime peaks."""
    selected_admission = admission
    names = admission.PROGRAMS
    if boundary_diagnostic:
        from scripts.greenfield import prefill_window_boundary_admission

        selected_admission = prefill_window_boundary_admission
    if completed_numerical:
        from scripts.greenfield import prefill_completed_window_admission as completed
        from scripts.greenfield import (
            prefill_completed_window_assembly as selected_admission,
        )

        names = (*completed.PROGRAMS, *selected_admission.PROGRAMS)
    calls = record["call_evidence"]
    expected = expected_calls(
        boundary_diagnostic=boundary_diagnostic, completed_numerical=completed_numerical
    )
    if len(calls) != len(expected):
        raise ValueError("window requires its exact WK and model call inventory")
    validate_call_sequence(
        record,
        calls,
        expected=expected,
        local_slots=local_slots,
        names=names,
        budgeter=selected_admission.memory_budget,
    )


def validate_call_sequence(
    record: Mapping[str, Any],
    calls: Iterable[Mapping[str, Any]],
    *,
    expected: Iterable[tuple[str, str]],
    local_slots: Mapping[int, int],
    names: tuple[str, ...],
    budgeter: Callable[..., dict],
) -> None:
    """Replay ordinary or streamed witnesses with the same memory/owner rules.

    Strict exhaustion matters: the stream checks its trailing bytes only after
    its final yielded record. Never stop after just the expected call count.
    """
    last: dict[int, dict[str, int]] = {}
    for call, (phase, name) in zip(calls, expected, strict=True):
        seconds = call.get("completed_call_seconds")
        if (
            call.get("phase") != phase
            or call.get("graph") != name
            or call.get("completed") is not True
            or type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ValueError(
                "window completed-call order or diagnostic duration differs"
            )
        census = call["census"]
        validate_memory_owners(
            census["devices"],
            local_slots=local_slots,
            process_index=record["jax_process_index"],
        )
        before = {r["device_id"]: r["memory_stats"] for r in census["devices"]}
        if last and any(
            before[d]["bytes_limit"] != last[d]["bytes_limit"]
            or before[d]["peak_bytes_in_use"] < last[d]["peak_bytes_in_use"]
            for d in last
        ):
            raise ValueError("window between-call lifetime memory counters regressed")
        budget = budgeter(census, call["compiled_memory"], active_graph=name)
        same_json(
            call["compiled_memory"],
            {n: record["programs"][n]["compiled_memory"] for n in names},
            "compiled call memory",
        )
        same_json(call["budget"], budget, "memory budget")
        if budget["estimate_fits"] is not True:
            raise ValueError("window predispatch memory reserve failed")
        after = call["post_memory"]
        validate_memory_owners(
            after, local_slots=local_slots, process_index=record["jax_process_index"]
        )
        for row in after:
            prior = before[row["device_id"]]
            if (
                any(
                    type(row[k]) is not int or row[k] < 0
                    for k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
                )
                or not row["bytes_in_use"]
                <= row["peak_bytes_in_use"]
                <= row["bytes_limit"]
                or row["bytes_limit"] != prior["bytes_limit"]
                or row["peak_bytes_in_use"] < prior["peak_bytes_in_use"]
                or row["bytes_limit"] - row["peak_bytes_in_use"]
                < admission.REQUIRED_RESERVE_BYTES
            ):
                raise ValueError("window post-call memory counters/reserve failed")
        last = {r["device_id"]: r for r in after}


def validate_files(
    root: Path, record: Mapping[str, Any], *, completed_numerical: bool = False
) -> None:
    """Replay actual graph/journal/NPZ bytes, never trust the worker's pass label."""
    selected_admission, selected_protocol = admission, protocol
    journal_type = WindowNumericalJournal
    names = admission.PROGRAMS
    if completed_numerical:
        from scripts.greenfield import (
            prefill_completed_window_admission as selected_admission,
        )
        from scripts.greenfield import (
            prefill_completed_window_protocol as selected_protocol,
        )
        from scripts.greenfield import prefill_completed_window_assembly as assembly
        from scripts.greenfield.prefill_completed_window_worker import CompletedJournal

        journal_type = CompletedJournal
        names = (*selected_admission.PROGRAMS, *assembly.PROGRAMS)
        same_json(
            [
                record.get("reference_scope"),
                record.get("independent_full_layer_admission"),
                record.get("assembly_executable_calls"),
            ],
            [selected_protocol.REFERENCE_SCOPE, False, 30],
            "completed suffix scope",
        )
    if (
        record.get("protocol") != selected_protocol.PROTOCOL
        or record.get("profile") != selected_admission.PROFILE
        or record.get("compile_only") is not False
        or record.get("performance_claim") is not False
        or record.get("current_phase") != "numerical_complete"
        or set(record["programs"]) != set(names)
        or set(record["cases"]) != set(selected_protocol.CASES)
    ):
        raise ValueError("window numerical scope or completeness differs")
    same_json(
        [record["model_executable_calls"], record["wk_executable_calls"]],
        [27 if completed_numerical else 15, 2],
        "completed totals",
    )
    slots = {s["device_id"]: s["device_slot"] for s in record["local_device_slots"]}
    validate_calls(record, local_slots=slots, completed_numerical=completed_numerical)

    def inspect(name, p, stable, optimized):
        if completed_numerical and name in assembly.PROGRAMS:
            same_json(
                p["admission"],
                assembly.inspect_program(name, stable, optimized, p["compiled_memory"]),
                "assembly graph report",
            )
        else:
            selected_admission.validate_program_report(
                p["admission"], name, stable, optimized, p["compiled_memory"]
            )

    journal = validate_graph_journal(
        root,
        record,
        protocol_id=selected_protocol.PROTOCOL,
        profile=selected_admission.PROFILE,
        names=names,
        journal_type=journal_type,
        inspect=inspect,
    )
    compile_stages = ("lower_compile_started", "compiled", "raw_written", "inspected")
    stages = [r["stage"] for r in journal]
    required = ["identity", "prepare"]
    for name in names:
        required.extend((*compile_stages, "compile/" + name))
    required.append("bind_compiled")

    def call_phases(phase):
        return (phase + "/memory", phase + "/execute", phase + "/memory_after")

    required.extend(
        (
            *call_phases("wk_decode"),
            *call_phases("wk_promote"),
            "wk_boundary",
            "input_specs",
        )
    )
    if completed_numerical:
        required.extend(
            ("completed_case_bind", "assembly_tiles", "assembly_tiles_ready")
        )
    required.append("rotary_fixture")
    for case in selected_protocol.CASES:
        required.extend(
            case + "/" + s for s in ("host", "inputs", "inputs_ready", "input_capture")
        )
        if completed_numerical:
            for phase, graph in expected_calls(completed_numerical=True):
                if not phase.startswith(case + "/"):
                    continue
                if graph in ("prefix", "candidate", "control"):
                    required.extend((phase + "_inputs", phase + "_ready"))
                required.extend(call_phases(phase))
        else:
            required.extend(call_phases(case + "/candidate"))
            for tile in range(4):
                phase = case + f"/control{tile}"
                required.extend(
                    (phase + "_inputs", phase + "_ready", *call_phases(phase))
                )
        required.append(case + "/comparison")
    required.append("numerical_complete")
    same_json(stages, required, "complete numerical phase order")
    for case in selected_protocol.CASES:
        path = root / f"{case}.npz"
        declared = record["cases"][case]
        if sha256(path.read_bytes()).hexdigest() != declared["npz_sha256"]:
            raise ValueError("window original case bytes differ")
        replay = selected_protocol.replay_case(path, case=case, slots_by_device=slots)
        same_json(declared["replay"], replay, "original array comparison")
        if not (
            replay["passed"] is True
            and declared["passed"] is True
            and declared["complete"] is True
        ):
            raise ValueError("window original case comparison refused")
    validate_wk(root, record, slots)


def validate_graph_journal(
    root: Path,
    record: Mapping[str, Any],
    *,
    protocol_id: str,
    profile: str,
    names: tuple[str, ...],
    journal_type: type,
    inspect: Callable[..., None],
) -> list[dict]:
    """Shared original graph/compile replay; caller checks full phase order."""
    raw = (root / "compile_journal.jsonl").read_bytes()
    if sha256(raw).hexdigest() != record["compile_journal_sha256"]:
        raise ValueError("window numerical journal bytes differ")
    journal = [json.loads(line) for line in raw.splitlines()]
    same_json(
        journal[0]["identity"],
        dict(
            protocol=protocol_id,
            profile=profile,
            compile_only=False,
            code_hash=record["code_hash"],
            launch_rank=record["launch_rank"],
        ),
        "journal identity",
    )
    if any(
        r.get("artifact_kind") != journal_type.artifact_kind
        or r.get("status") != journal_type.status
        or r.get("performance_claim") is not False
        or r.get("numerical_claim") is not False
        or r.get("passed", True) is not True
        for r in journal
    ):
        raise ValueError("window journal scope or failed phase")
    # Compile stages must precede any numerical dispatch. Phase events retain
    # the last graph ID; distinguish them by their exact stage, not graph alone.
    compile_stages = ("lower_compile_started", "compiled", "raw_written", "inspected")
    observed = [
        (r["graph"], r["stage"]) for r in journal if r["stage"] in compile_stages
    ]
    same_json(
        [list(v) for v in observed],
        [[n, s] for n in names for s in compile_stages],
        "compile journal order",
    )
    for name in names:
        p = record["programs"][name]
        stable = (root / f"{name}.stablehlo.mlir").read_text()
        optimized = (root / f"{name}.optimized_hlo.txt").read_text()
        for text, key in (
            (stable, "stablehlo_sha256"),
            (optimized, "optimized_hlo_sha256"),
        ):
            if sha256(text.encode()).hexdigest() != p[key]:
                raise ValueError("window original graph bytes differ")
        inspect(name, p, stable, optimized)
        stages = [
            r for r in journal if r["graph"] == name and r["stage"] in compile_stages
        ]
        same_json(
            stages[1]["compiled_memory"], p["compiled_memory"], "compile allocation"
        )
        same_json(stages[1]["seconds"], p["compile_seconds"], "compile duration")
        for key in ("stablehlo_sha256", "optimized_hlo_sha256"):
            same_json(stages[2][key], p[key], "journal graph SHA")
        same_json(stages[3]["report"], p["admission"], "journal graph report")
    return journal


def validate_wk(
    root: Path, record: Mapping[str, Any], slots: Mapping[int, int]
) -> None:
    """Bind completed individual WK captures to the actual BF16/F32 boundary."""
    if set(record["wk_originals"]) != {"wk_decode", "wk_promote"}:
        raise ValueError("window WK original inventory differs")
    arrays = {}
    for name, digest in {
        **record["wk_originals"],
        "wk_boundary": record["wk_boundary_sha256"],
    }.items():
        path = root / f"{name}.npz"
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("window WK original bytes differ")
        with np.load(path, allow_pickle=False) as saved:
            arrays[name] = dict(saved)
    expected = {str(d) for d in slots}
    if any(set(arrays[n]) != expected for n in ("wk_decode", "wk_promote")) or set(
        arrays["wk_boundary"]
    ) != {f"{kind}_{d}" for d in slots for kind in ("bf16", "fp32")}:
        raise ValueError("window WK original owner inventory differs")
    for d in slots:
        raw, promoted = arrays["wk_decode"][str(d)], arrays["wk_promote"][str(d)]
        if (
            raw.dtype != np.uint16
            or raw.shape != (128, 6144)
            or promoted.dtype != np.float32
            or promoted.shape != raw.shape
            or not np.isfinite(promoted).all()
            or not np.array_equal(raw.view(protocol.BF16).astype(np.float32), promoted)
        ):
            raise ValueError("window completed WK numerical boundary differs")
        for name, value in ((f"bf16_{d}", raw), (f"fp32_{d}", promoted)):
            other = arrays["wk_boundary"][name]
            if (
                other.dtype != value.dtype
                or other.shape != value.shape
                or other.tobytes() != value.tobytes()
            ):
                raise ValueError(
                    "window completed WK boundary not its original capture"
                )
