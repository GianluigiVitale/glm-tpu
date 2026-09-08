"""Bounded DB591 capture execution, not yet wired to a launcher or sealer.

Uses existing budgeted calls and actual B128/B32 programs. No new model,
reference, precision setting or performance loop. Signature reproduction is
diagnostic: failure is retained, never promoted to numerical correctness.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.greenfield import prefill_window_boundary_admission as admission
from scripts.greenfield import prefill_window_protocol as window
from scripts.greenfield.prefill_layer_evidence import (
    encode_arrays,
    input_hashes,
    local_observations,
)
from scripts.greenfield.prefill_layer_numerical import FIELDS
from scripts.greenfield.prefill_window_boundary import capture_owner_arrays
from scripts.greenfield.prefill_window_worker import BudgetedCalls, save_arrays
from scripts.greenfield.ws32_acquisition_journal import Ws32NumericalJournal

PROTOCOL = window.PROTOCOL + "-actual-boundary-diagnostic-v1"
KERNEL = "ws32_prefill_window_boundary_diagnostic"
REFERENCE_SCOPE = "ORIGINAL_FAILURE_REPRODUCTION_AND_ACTUAL_OPERANDS_NOT_NUMERICAL_PASS"
ORIGINAL_RECEIPT = Path(__file__).resolve().parents[2] / (
    "docs/artifacts/prefill-window-boundary-refusal-v2-20260908.json"
)
ORIGINAL_SHA = "2b5e556dfeea92071859650b70c7ca51d882ab1d5a26d54845beee6d0e273fff"
SIGNATURE_FIELDS = ("positions", "counts", "scores", "routes", "route_weights")


class BoundaryJournal(Ws32NumericalJournal):
    artifact_kind = "greenfield_ws32_window_boundary_diagnostic_journal_v1"

    def _check_identity(self, identity: Mapping[str, Any]) -> None:
        if (
            identity.get("protocol") != PROTOCOL
            or identity.get("profile") != admission.PROFILE
            or identity.get("compile_only") is not False
        ):
            raise ValueError("boundary journal requires distinct diagnostic identity")


def original_receipt() -> dict[str, Any]:
    """Previously generation-verified evidence, pinned before capture execution.

    This uses compact original fingerprints, not a new cloud transfer or new
    interpretation of the failed verdict. Source generations remain in receipt.
    """
    raw = ORIGINAL_RECEIPT.read_bytes()
    if sha256(raw).hexdigest() != ORIGINAL_SHA:
        raise ValueError("original boundary receipt content changed")
    return json.loads(raw)


def bind_originals(
    record: Mapping[str, Any], local_slots: Mapping[int, int]
) -> dict[str, Any]:
    """Bind slots, never assume physical device id equals feature/expert slot.

    The existing selected-layer loader/collector independently authenticate
    current selected checkpoint bytes, fleet/process identity and ownership.
    """
    original = original_receipt()
    if record.get("physical_device_ids") != original["physical_device_ids"]:
        raise ValueError("boundary diagnostic physical mesh differs from originals")
    order = np.asarray(original["physical_device_ids"]).ravel().tolist()
    if (
        len(local_slots) != 4
        or len(set(local_slots.values())) != 4
        or any(
            type(device) is not int
            or type(slot) is not int
            or not 0 <= slot < 32
            or order[slot] != device
            for device, slot in local_slots.items()
        )
    ):
        raise ValueError("boundary diagnostic original owner mapping differs")
    return dict(
        receipt_sha256=ORIGINAL_SHA,
        original_tag=original["tag"],
        original_pin=original["pin"],
        original_verdict="FAILED",
        sources=original["original_sources"],
        slots_by_device={str(d): s for d, s in local_slots.items()},
        fingerprint_scope="PREVIOUSLY_GENERATION_VERIFIED_ALL12_OUTPUTS_BY_PHYSICAL_SLOT",
    )


def compare_fingerprints(
    observed: Mapping[str, np.ndarray], *, slot: int, kind: str
) -> dict[str, Any]:
    if type(slot) is not int or not 0 <= slot < 32 or kind not in ("actual", "control"):
        raise ValueError("unregistered original boundary owner/path")
    window._shape_contract(observed)
    expected = original_receipt()["owners"][str(slot)][kind + "_sha256"]
    actual = input_hashes(observed)
    if set(actual) != set(FIELDS) or set(expected) != set(FIELDS):
        raise ValueError("boundary reproduction requires all12 original fields")
    matches = {name: actual[name] == expected[name] for name in FIELDS}
    return dict(
        original_sha256=expected,
        observed_sha256=actual,
        fields_byte_identical=matches,
        signature_reproduced=all(matches[n] for n in SIGNATURE_FIELDS),
        all_outputs_reproduced=all(matches.values()),
        numerical_admission=False,
        performance_claim=False,
    )


def _capture(
    calls: BudgetedCalls,
    arrays: dict[str, np.ndarray],
    kind: str,
    result: tuple[Any, Any],
) -> dict[int, dict[str, np.ndarray]]:
    """Save originals, then all operands, before health or schema comparisons."""
    path = calls.root / "boundary.npz"
    originals, captures = result
    observed = local_observations(originals)
    for device, values in observed.items():
        arrays.update(encode_arrays(f"{kind}_{device}", values))
    calls.record["boundary_npz_sha256"] = save_arrays(path, arrays)
    operands = capture_owner_arrays(captures, slots_by_device=calls.local_slots)
    manifest = {}
    for device, values in operands.items():
        fields = {}
        for name, value in values.items():
            key = f"capture_{kind}_{device}__{name}"
            arrays[key] = value.view(np.uint16) if value.dtype == window.BF16 else value
            fields[name] = dict(
                shape=list(value.shape),
                dtype=str(value.dtype),
                sha256=sha256(value.tobytes()).hexdigest(),
            )
        manifest[str(device)] = fields
    calls.record.setdefault("captures", {})[kind] = manifest
    calls.record["boundary_npz_sha256"] = save_arrays(path, arrays)
    if set(observed) != set(calls.local_slots):
        raise ValueError("boundary original output owners differ; arrays preserved")
    schema = calls.record["programs"]["candidate" if kind == "actual" else "control"][
        "compiler_output_schema"
    ]["captures"]
    for fields in manifest.values():
        if set(fields) != set(schema) or any(
            v["shape"] != schema[n]["shape"][2:] or v["dtype"] != schema[n]["dtype"]
            for n, v in fields.items()
        ):
            raise ValueError(
                "boundary runtime capture schema differs; arrays preserved"
            )
    if not all(v["health"].all() for v in observed.values()):
        raise ValueError("boundary output health failed; arrays preserved")
    # This fixed boundary has128/128 live rows (32/32 in every control).
    # None of the captured floating fields uses an intentional infinity sentinel;
    # the original selected-score padding remains in original12, not these taps.
    if any(
        not np.isfinite(value).all()
        for values in operands.values()
        for value in values.values()
        if value.dtype in (window.BF16, np.float32)
    ):
        raise ValueError("boundary captured operand is nonfinite; arrays preserved")
    return observed


def execute_boundary_case(
    calls: BudgetedCalls, *, weights: Any, wk: Any, mesh: Any, specs: tuple[Any, ...]
) -> None:
    """One B128 and four causally carried B32 calls, no failed comparator reuse."""
    import jax
    from glm_tpu.greenfield.kernels.reference.rotary import build_rotary_table_host
    from scripts.greenfield.probe_ws32_prefill_layer import device_inputs

    def fixture():
        rope = build_rotary_table_host(window.CAPACITY, rotary_dim=64, theta=8e6)
        host = window.host_case("boundary", rope)
        if input_hashes(host) != original_receipt()["input_sha256"]:
            raise ValueError("boundary fixture differs from original failure")
        return host

    host = calls.phase("boundary/host", fixture)
    values = calls.phase(
        "boundary/inputs", lambda: device_inputs(host, specs, weights, wk, mesh)
    )
    calls.phase("boundary/inputs_ready", lambda: jax.block_until_ready(values))
    arrays = encode_arrays("input", host)
    calls.phase(
        "boundary/input_capture",
        lambda: save_arrays(calls.root / "boundary.npz", arrays),
    )
    actual_host = {}

    def capture_actual(result):
        actual_host.update(_capture(calls, arrays, "actual", result))

    actual = calls.call(
        "boundary/candidate", "candidate", values, preserve=capture_actual
    )
    previous = None
    controls = []
    for tile in range(4):
        inputs = calls.phase(
            f"boundary/control{tile}_inputs",
            lambda: window.control_inputs(values, tile, previous),
        )
        calls.phase(
            f"boundary/control{tile}_ready", lambda: jax.block_until_ready(inputs)
        )
        result = calls.call(
            f"boundary/control{tile}",
            "control",
            inputs,
            preserve=lambda result: controls.append(
                _capture(calls, arrays, f"tile{tile}", result)
            ),
        )
        previous = result[0]  # Only original12, including all3 actual cache outputs.

    def finish():
        comparisons = {}
        for device, slot in calls.local_slots.items():
            control = window.stack_control([c[device] for c in controls])
            arrays.update(encode_arrays(f"control_{device}", control))
        # Retain individual control originals AND captures, not only concatenation.
        calls.record["boundary_npz_sha256"] = save_arrays(
            calls.root / "boundary.npz", arrays
        )
        for device, slot in calls.local_slots.items():
            control = window.stack_control([c[device] for c in controls])
            comparisons[str(device)] = dict(
                actual=compare_fingerprints(
                    actual_host[device], slot=slot, kind="actual"
                ),
                control=compare_fingerprints(control, slot=slot, kind="control"),
            )
        calls.record["original_reproduction"] = comparisons
        reproduced = all(
            p["signature_reproduced"]
            for pair in comparisons.values()
            for p in pair.values()
        )
        calls.record.update(
            boundary_capture_complete=True,
            original_signature_reproduced=reproduced,
            classification=(
                "ORIGINAL_SIGNATURE_REPRODUCED"
                if reproduced
                else "INSTRUMENTATION_PERTURBED_ORIGINAL_SIGNATURE"
            ),
            numerical_admission=False,
            performance_claim=False,
        )

    calls.phase("boundary/reproduction", finish)
    # Strong references preserve all live outputs until last budgeted dispatch.
    del actual, previous, result, inputs, values
