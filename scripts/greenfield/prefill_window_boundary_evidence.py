"""Replay original diagnostic arrays, DB591 graphs, journal and seven calls.

The existing fleet collector must also authenticate selected checkpoint bytes,
unique hosts/physical owners and generation-qualified publication. No TPU or
performance authority; an instrumented signature change is retained diagnostic
evidence, never a numeric PASS for the model.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.greenfield import prefill_window_boundary_admission as admission
from scripts.greenfield import prefill_window_boundary_worker as boundary
from scripts.greenfield import prefill_window_protocol as window
from scripts.greenfield.prefill_layer_evidence import (
    INPUT_FIELDS,
    decode_arrays,
    equal_bytes,
)
from scripts.greenfield.prefill_layer_numerical import FIELDS
from scripts.greenfield.prefill_window_evidence import (
    same_json,
    validate_calls,
    validate_wk,
)


def validate_workers(records: list[dict[str, Any]], **kwargs: Any) -> None:
    from scripts.greenfield.prefill_window_evidence import validate_workers as shared

    shared(records, boundary_diagnostic=True, **kwargs)


def validate_record(record: dict[str, Any], pin: str) -> None:
    from scripts.greenfield.ws32_prefill_layer_campaign import (
        checkpoint_ledger,
        validate_workers as validate_fleet,
    )

    exact = dict(
        status="SUCCESS",
        kernel=boundary.KERNEL,
        protocol=boundary.PROTOCOL,
        profile=admission.PROFILE,
        code_hash=pin,
        layer=6,
        latency=None,
        warmup=0,
        iterations=0,
        compile_only=False,
        numerical_execution_authorized=True,
        admission_only=False,
        baseline_only=False,
        diagnostic_only=True,
        performance_claim=False,
        profiler_free_timing=False,
        comparison=dict(passed=None, diagnostic_evidence_complete=True),
    )
    same_json({k: record.get(k) for k in exact}, exact, "boundary aggregate scope")
    pins, ledger = checkpoint_ledger(6)
    validate_fleet(
        record["workers"],
        pin,
        layer=6,
        pins=pins,
        ledger=ledger,
        boundary_diagnostic=True,
    )
    same_json(record["hlo"], record["workers"][0]["hlo"], "boundary aggregate HLO")
    if (
        record["checksum"]
        != sha256(json.dumps(record["workers"], sort_keys=True).encode()).hexdigest()
    ):
        raise ValueError("boundary aggregate worker digest differs")


def replay_arrays(
    root: Path, record: Mapping[str, Any], slots: Mapping[int, int]
) -> None:
    """Recompute fingerprints, schemas and stored capture manifests from NPZ."""
    from glm_tpu.greenfield.kernels.reference.rotary import build_rotary_table_host

    path = root / "boundary.npz"
    if sha256(path.read_bytes()).hexdigest() != record["boundary_npz_sha256"]:
        raise ValueError("boundary original NPZ bytes differ")
    same_json(
        record["original_binding"],
        boundary.bind_originals(record, slots),
        "original binding",
    )
    canonical = window.host_case(
        "boundary", build_rotary_table_host(window.CAPACITY, rotary_dim=64, theta=8e6)
    )
    kinds = ("actual", "tile0", "tile1", "tile2", "tile3", "control")
    manifests, comparisons = {}, {}
    expected_keys = {f"input__{name}" for name in INPUT_FIELDS}
    with np.load(path, allow_pickle=False) as saved:
        host = decode_arrays(saved, "input", INPUT_FIELDS)
        if any(not equal_bytes(host[n], canonical[n]) for n in INPUT_FIELDS):
            raise ValueError("diagnostic fixture differs from original")
        for device, slot in slots.items():
            originals = {}
            for kind in kinds:
                prefix = f"{kind}_{device}"
                expected_keys.update(f"{prefix}__{n}" for n in FIELDS)
                originals[kind] = decode_arrays(saved, prefix, FIELDS)
                if not originals[kind]["health"].all():
                    raise ValueError("original diagnostic output health failed")
                if kind == "control":
                    continue
                schema = record["programs"][
                    "candidate" if kind == "actual" else "control"
                ]["compiler_output_schema"]["captures"]
                fields = {}
                for name, leaf in schema.items():
                    key = f"capture_{kind}_{device}__{name}"
                    expected_keys.add(key)
                    raw = saved[key]
                    dtype = np.dtype(leaf["dtype"])
                    if raw.dtype != (np.uint16 if dtype == window.BF16 else dtype):
                        raise ValueError("capture storage dtype differs")
                    value = raw.view(window.BF16) if dtype == window.BF16 else raw
                    if list(value.shape) != leaf["shape"][2:]:
                        raise ValueError("capture physical owner shape differs")
                    if (
                        dtype in (window.BF16, np.float32)
                        and not np.isfinite(value).all()
                    ):
                        raise ValueError("captured operand is nonfinite")
                    fields[name] = dict(
                        shape=list(value.shape),
                        dtype=str(value.dtype),
                        sha256=sha256(value.tobytes()).hexdigest(),
                    )
                manifests.setdefault(kind, {})[str(device)] = fields
            combined = window.stack_control([originals[f"tile{i}"] for i in range(4)])
            if any(
                not equal_bytes(combined[n], originals["control"][n]) for n in FIELDS
            ):
                raise ValueError("control concatenation is not its four originals")
            comparisons[str(device)] = {
                kind: boundary.compare_fingerprints(
                    originals[kind], slot=slot, kind=kind
                )
                for kind in ("actual", "control")
            }
        if set(saved.files) != expected_keys:
            raise ValueError("boundary NPZ original/capture inventory differs")
    same_json(record["captures"], manifests, "actual capture manifests")
    same_json(record["original_reproduction"], comparisons, "original reproduction")
    reproduced = all(
        p["signature_reproduced"]
        for pair in comparisons.values()
        for p in pair.values()
    )
    same_json(
        {
            k: record.get(k)
            for k in (
                "boundary_capture_complete",
                "original_signature_reproduced",
                "classification",
                "numerical_admission",
                "performance_claim",
            )
        },
        dict(
            boundary_capture_complete=True,
            original_signature_reproduced=reproduced,
            classification=(
                "ORIGINAL_SIGNATURE_REPRODUCED"
                if reproduced
                else "INSTRUMENTATION_PERTURBED_ORIGINAL_SIGNATURE"
            ),
            numerical_admission=False,
            performance_claim=False,
        ),
        "diagnostic outcome",
    )


def validate_files(root: Path, record: Mapping[str, Any]) -> None:
    """Full local publication replay, before the fleet collector can seal it."""
    exact = dict(
        protocol=boundary.PROTOCOL,
        profile=admission.PROFILE,
        compile_only=False,
        diagnostic_only=True,
        performance_claim=False,
        reference_scope=boundary.REFERENCE_SCOPE,
        current_phase="boundary_complete",
        model_executable_calls=5,
        wk_executable_calls=2,
    )
    same_json({k: record.get(k) for k in exact}, exact, "boundary diagnostic scope")
    if set(record["programs"]) != set(admission.PROGRAMS):
        raise ValueError("boundary diagnostic graph inventory differs")
    slots = {s["device_id"]: s["device_slot"] for s in record["local_device_slots"]}
    boundary.bind_originals(record, slots)
    validate_calls(record, local_slots=slots, boundary_diagnostic=True)
    raw = (root / "compile_journal.jsonl").read_bytes()
    if sha256(raw).hexdigest() != record["compile_journal_sha256"]:
        raise ValueError("boundary journal bytes differ")
    journal = [json.loads(line) for line in raw.splitlines()]
    same_json(
        journal[0]["identity"],
        dict(
            protocol=boundary.PROTOCOL,
            profile=admission.PROFILE,
            compile_only=False,
            code_hash=record["code_hash"],
            launch_rank=record["launch_rank"],
        ),
        "boundary journal identity",
    )
    if any(
        r.get("artifact_kind") != boundary.BoundaryJournal.artifact_kind
        or r.get("status") != boundary.BoundaryJournal.status
        or r.get("performance_claim") is not False
        or r.get("numerical_claim") is not False
        or r.get("passed", True) is not True
        for r in journal
    ):
        raise ValueError("boundary journal scope or phase failure")
    compile_stages = ("lower_compile_started", "compiled", "raw_written", "inspected")
    same_json(
        [[r["graph"], r["stage"]] for r in journal if r["stage"] in compile_stages],
        [[n, s] for n in admission.PROGRAMS for s in compile_stages],
        "boundary compile order",
    )
    for name in admission.PROGRAMS:
        program = record["programs"][name]
        stable = (root / f"{name}.stablehlo.mlir").read_text()
        hlo = (root / f"{name}.optimized_hlo.txt").read_text()
        for text, key in ((stable, "stablehlo_sha256"), (hlo, "optimized_hlo_sha256")):
            if sha256(text.encode()).hexdigest() != program[key]:
                raise ValueError("boundary raw graph bytes differ")
        admission.validate_program_report(
            program["admission"],
            name,
            stable,
            hlo,
            program["compiled_memory"],
            output_schema=program.get("compiler_output_schema"),
        )
        stages = [
            r for r in journal if r["graph"] == name and r["stage"] in compile_stages
        ]
        same_json(
            stages[1]["compiled_memory"],
            program["compiled_memory"],
            "boundary compiler allocation",
        )
        same_json(
            stages[1]["seconds"], program["compile_seconds"], "boundary compile seconds"
        )
        for key in ("stablehlo_sha256", "optimized_hlo_sha256"):
            same_json(stages[2][key], program[key], "boundary journal graph SHA")
        same_json(stages[3]["report"], program["admission"], "boundary graph report")
    required = ["identity", "prepare"]
    for name in admission.PROGRAMS:
        required.extend((*compile_stages, "compile/" + name))
    required.append("bind_compiled")

    def phases(phase):
        return (phase + "/memory", phase + "/execute", phase + "/memory_after")

    required.extend(
        (*phases("wk_decode"), *phases("wk_promote"), "wk_boundary", "input_specs")
    )
    required.extend(
        "boundary/" + s for s in ("host", "inputs", "inputs_ready", "input_capture")
    )
    required.extend(phases("boundary/candidate"))
    for tile in range(4):
        phase = f"boundary/control{tile}"
        required.extend((phase + "_inputs", phase + "_ready", *phases(phase)))
    required.extend(("boundary/reproduction", "boundary_complete"))
    same_json([r["stage"] for r in journal], required, "boundary phase order")
    validate_wk(root, record, slots)
    replay_arrays(root, record, slots)
