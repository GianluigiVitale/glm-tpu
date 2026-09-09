"""Three untimed, protected device calls; no entry point or fleet launcher.

The selected-layer parent owns initialization/weights/runtime provenance.
Every local failure votes before the next distributed call. Candidate outputs
are retained before comparison, and the collector must independently replay.
"""

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.greenfield import prefill_rolled_admission as admission
from scripts.greenfield import prefill_rolled_window as protocol
from scripts.greenfield import prefill_phase_originals as originals
from scripts.greenfield.prefill_layer_evidence import encode_arrays, local_observations
from scripts.greenfield.prefill_window_acquisition import fleet_step
from scripts.greenfield.prefill_window_worker import BudgetedCalls, save_arrays
from scripts.greenfield.ws32_acquisition_journal import Ws32NumericalJournal


class RolledJournal(Ws32NumericalJournal):
    artifact_kind = "greenfield_ws32_rolled_layer_numerical_journal_v1"

    def _check_identity(self, identity: Mapping[str, Any]) -> None:
        if (
            identity.get("protocol") != protocol.PROTOCOL
            or identity.get("profile") != admission.PROFILE
            or identity.get("compile_only") is not False
        ):
            raise ValueError("rolled journal requires distinct numerical identity")


def compact_comparison(result: dict) -> dict:
    raw = json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return dict(
        passed=result["passed"],
        report_sha256=sha256(raw).hexdigest(),
        report_bytes=len(raw),
    )


def execute(
    *,
    root: Path,
    record: dict,
    mesh: Any,
    config: Any,
    weights: Any,
    local_slots: Mapping[int, int],
    consensus: Any,
    reference: protocol.RetainedReference,
) -> None:
    import jax
    from scripts.greenfield.probe_ws32_prefill_layer import (
        compile_program,
        device_inputs,
        input_specs,
    )

    def guarded(name, action):
        return fleet_step(name, action, record=record, root=root, consensus=consensus)

    def bind():
        if (
            record.get("protocol") != protocol.PROTOCOL
            or record.get("profile") != admission.PROFILE
            or record.get("compile_only") is not False
            or dict(local_slots) != dict(reference.slots)
        ):
            raise ValueError(
                "rolled current mode/owners differ from retained reference"
            )
        record["original_binding"] = originals.bind_originals(
            record, local_slots, originals.load_capsule()
        )
        record["retained_sources"] = list(reference.sources)
        record["retained_source_seal_sha256"] = protocol.SEAL_SHA
        record["retained_original_comparison"] = dict(reference.bounded_receipt)
        admission.registered_programs()

    guarded("rolled_bind", bind)
    journal = guarded(
        "rolled_journal",
        lambda: RolledJournal(
            root / "compile_journal.jsonl",
            dict(
                protocol=protocol.PROTOCOL,
                profile=admission.PROFILE,
                compile_only=False,
                code_hash=record["code_hash"],
                launch_rank=record["launch_rank"],
            ),
        ),
    )
    try:
        programs = guarded(
            "rolled_prepare",
            lambda: protocol.prepare_programs(
                mesh=mesh, config=config, weights=weights
            ),
        )
        calls = BudgetedCalls(
            root=root,
            record=record,
            consensus=consensus,
            journal=journal,
            local_slots=local_slots,
            budgeter=admission.memory_budget,
        )
        for name, fn, args in programs:
            graph = guarded(
                "rolled_compile_" + name,
                lambda: compile_program(fn, args, name, root, record, journal=journal),
            )
            calls.programs[name] = graph

            def inspect():
                stable = (root / f"{name}.stablehlo.mlir").read_text()
                optimized = (root / f"{name}.optimized_hlo.txt").read_text()
                record["programs"][name]["admission"] = journal.inspect(
                    name,
                    stable,
                    optimized,
                    lambda: admission.inspect_program(
                        name,
                        stable,
                        optimized,
                        record["programs"][name]["compiled_memory"],
                    ),
                )

            guarded("rolled_inspect_" + name, inspect)
        capsule = originals.load_capsule()

        def preserve_wk(name, value):
            observed = {
                int(s.device.id): np.asarray(s.data).copy()
                for s in value.addressable_shards
            }
            stored = {
                str(d): v.view(np.uint16) if v.dtype == protocol.window.BF16 else v
                for d, v in observed.items()
            }
            digest = save_arrays(root / f"{name}.npz", stored)
            record.setdefault("wk_sha256", {})[name] = digest
            if set(observed) != set(local_slots):
                raise ValueError("rolled WK output owners differ; originals preserved")
            for device, value in observed.items():
                originals.check_observation(
                    capsule, slot=local_slots[device], kind=name, values={"wk": value}
                )

        decoded = calls.call(
            "wk_decode",
            "wk_decode",
            (weights.dsa.wk_bits_local, weights.dsa.wk_scale_local),
            preserve=lambda v: preserve_wk("wk_decode", v),
        )
        wk = calls.call(
            "wk_promote",
            "wk_promote",
            (decoded,),
            preserve=lambda v: preserve_wk("wk_promote", v),
        )
        values = calls.phase(
            "candidate_inputs",
            lambda: device_inputs(
                dict(reference.inputs), input_specs(weights, wk), weights, wk, mesh
            ),
        )
        calls.phase("candidate_inputs_ready", lambda: jax.block_until_ready(values))
        arrays = encode_arrays("input", reference.inputs)
        path = root / "candidate.npz"
        calls.phase("candidate_input_capture", lambda: save_arrays(path, arrays))

        def preserve_candidate(value):
            observed = local_observations(value)
            for device, fields in observed.items():
                arrays.update(encode_arrays(f"actual_{device}", fields))
            record["candidate_sha256"] = save_arrays(path, arrays)
            comparison = protocol.compare_observations(observed, reference)
            record["candidate_comparison"] = compact_comparison(comparison)
            if not comparison["passed"]:
                raise ValueError(
                    "rolled candidate fails unchanged retained-reference bounds"
                )

        calls.call("candidate", "candidate", values, preserve=preserve_candidate)
        calls.phase(
            "rolled_complete",
            lambda: record.update(
                integration_complete=True,
                model_executable_calls=1,
                wk_executable_calls=2,
                performance_claim=False,
                independent_canonical_dsa_claim=False,
            ),
        )
    finally:
        guarded("rolled_finalize", journal.close)

    def finalize():
        record["compile_journal_sha256"] = sha256(
            (root / "compile_journal.jsonl").read_bytes()
        ).hexdigest()
        record["hlo"] = dict(
            sha256=record["programs"]["candidate"]["optimized_hlo_sha256"],
            contract=dict(passed=True, profile=admission.PROFILE),
        )

    guarded("rolled_snapshot", finalize)
