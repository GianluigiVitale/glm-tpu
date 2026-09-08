"""Compact DB594 byte witnesses for the phase baseline, not a new oracle.

Offline generation reads only already archived originals. Runtime comparison
binds every observed field by physical slot, shape, storage dtype and bytes.
Existing loader/HLO validation still authenticates weights and executables.
"""

from __future__ import annotations

from hashlib import sha256
import io
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.greenfield import prefill_completed_window_protocol as completed
from scripts.greenfield.prefill_layer_evidence import INPUT_FIELDS, decode_arrays
from scripts.greenfield.prefill_layer_numerical import FIELDS

ARTIFACTS = Path(__file__).resolve().parents[2] / "docs/artifacts"
SEALED = ARTIFACTS / "prefill-completed-window-numerical-sealed-20260908.json"
SEALED_SHA = "1b07409746d5d3f1c755eadc86b619a72c3838d67de748fd72f275029f3448f9"
CAPSULE = ARTIFACTS / "prefill-phase-db594-originals-20260908.json"
CAPSULE_SHA = "706813a99acb3095b4782e68ebfbd94b8a342d4f9d2bddf6b65982a0fe258ffa"
COMPONENTS = {
    **{f"prefix{i}": completed.PREFIX_FIELDS for i in range(4)},
    "wide": completed.SUFFIX_FIELDS,
    **{f"narrow{i}": completed.SUFFIX_FIELDS for i in range(4)},
    "actual": FIELDS,
    "control": FIELDS,
}


def manifest(values: Mapping[str, np.ndarray]) -> dict[str, dict]:
    """Bind BF16 as its exact U16 storage, matching existing original NPZs."""
    result = {}
    for name, value in values.items():
        value = np.asarray(value)
        if value.dtype == completed.window.BF16:
            value = value.view(np.uint16)
        result[name] = dict(
            shape=list(value.shape),
            dtype=str(value.dtype),
            sha256=sha256(value.tobytes()).hexdigest(),
        )
    return result


def _json_bound(path: Path, digest: str) -> dict:
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != digest:
        raise ValueError(f"phase original digest differs: {path.name}")
    return json.loads(raw)


def build_capsule(root: Path) -> dict:
    """Derive compact witnesses from all eight SHA-bound original publications."""
    seal = _json_bound(SEALED, SEALED_SHA)
    _json_bound(root / "SUCCESS", seal["readbacks"]["SUCCESS"]["sha256"])
    ledger = _json_bound(
        root / "archive_receipts.json",
        seal["readbacks"]["archive_receipts.json"]["sha256"],
    )
    indexed = {row["name"]: row for row in ledger}
    if len(indexed) != len(ledger):
        raise ValueError("duplicate original archive names")
    sources = []

    def read(relative):
        item = indexed[f"results/{seal['tag']}/{relative}"]
        raw = (root / relative).read_bytes()
        if (
            len(raw) != item["size"]
            or sha256(raw).hexdigest() != item["original_sha256"]
        ):
            raise ValueError(f"phase original archive bytes differ: {relative}")
        sources.append(item)
        return raw

    result: dict[str, Any] = dict(
        schema_version=1,
        source_seal_sha256=SEALED_SHA,
        source_tag=seal["tag"],
        source_pin=seal["code_hash"],
        source_db=594,
        case="competitive",
        owners={},
        sources=sources,
        performance_claim=False,
        independent_full_layer_admission=False,
    )
    for rank in range(8):
        prefix = f"fleet/rank{rank}"
        record = json.loads(read(prefix + "/runner.json"))
        if (
            record["status"] != "SUCCESS"
            or record["code_hash"] != seal["code_hash"]
            or record["protocol"] != completed.PROTOCOL
            or record["launch_rank"] != rank
        ):
            raise ValueError("phase source is not the sealed numerical run")
        mesh = record["physical_device_ids"]
        if rank == 0:
            result["physical_device_ids"] = mesh
            result["mesh_sha256"] = record["mesh_sha256"]
        if (
            mesh != result["physical_device_ids"]
            or record["mesh_sha256"] != result["mesh_sha256"]
        ):
            raise ValueError("phase original physical topology differs across ranks")
        order = [d for group in mesh for d in group]
        slots = {r["device_id"]: r["device_slot"] for r in record["local_device_slots"]}
        if len(slots) != 4 or any(
            order[s] != d or str(s) in result["owners"] for d, s in slots.items()
        ):
            raise ValueError("phase original duplicate or incorrect owners")
        with np.load(
            io.BytesIO(read(prefix + "/competitive.npz")), allow_pickle=False
        ) as arrays:
            inputs = manifest(decode_arrays(arrays, "input", INPUT_FIELDS))
            if rank == 0:
                result["inputs"] = inputs
            if inputs != result["inputs"]:
                raise ValueError("phase original fixture differs across hosts")
            expected = {f"input__{n}" for n in INPUT_FIELDS}
            for device, slot in slots.items():
                owner = dict(
                    device_id=device,
                    launch_rank=rank,
                    jax_process_index=record["jax_process_index"],
                    components={},
                )
                owner["selected_tensor_sha256"] = next(
                    r["observed_selected_tensor_sha256"]
                    for r in record["local_device_slots"]
                    if r["device_id"] == device
                )
                for kind, fields in COMPONENTS.items():
                    names = {n: f"{kind}_{device}__{n}" for n in fields}
                    expected.update(names.values())
                    owner["components"][kind] = manifest(
                        {n: arrays[key] for n, key in names.items()}
                    )
                result["owners"][str(slot)] = owner
            if set(arrays.files) != expected:
                raise ValueError("phase original component inventory differs")
        for kind in ("wk_decode", "wk_promote"):
            with np.load(
                io.BytesIO(read(prefix + f"/{kind}.npz")), allow_pickle=False
            ) as arrays:
                if set(arrays.files) != {str(d) for d in slots}:
                    raise ValueError("phase original WK owners differ")
                for device, slot in slots.items():
                    result["owners"][str(slot)][kind] = manifest(
                        {"wk": arrays[str(device)]}
                    )
    if set(result["owners"]) != {str(s) for s in range(32)}:
        raise ValueError("phase originals do not cover32 physical slots")
    return result


def load_capsule() -> dict:
    return _json_bound(CAPSULE, CAPSULE_SHA)


def bind_originals(
    record: Mapping[str, Any],
    local_slots: Mapping[int, int],
    original: Mapping[str, Any],
) -> dict:
    """Require the actual physical mapping and selected checkpoint bytes."""
    mesh = record.get("physical_device_ids")
    if (
        type(mesh) not in (list, tuple)
        or len(mesh) != 8
        or any(type(row) not in (list, tuple) or len(row) != 4 for row in mesh)
        or any(type(d) is not int for row in mesh for d in row)
        or [list(row) for row in mesh] != original["physical_device_ids"]
        or record.get("mesh_sha256") != original["mesh_sha256"]
    ):
        raise ValueError("phase original physical mesh differs")
    if len(local_slots) != 4 or len(set(local_slots.values())) != 4:
        raise ValueError("phase requires four local owners")
    loaded = record["local_device_slots"]
    if len(loaded) != 4 or {r["device_id"]: r["device_slot"] for r in loaded} != dict(
        local_slots
    ):
        raise ValueError("phase loaded owner mapping differs")
    for device, slot in local_slots.items():
        if type(device) is not int or type(slot) is not int or not 0 <= slot < 32:
            raise ValueError("phase owner IDs must be exact integers")
        source = original["owners"][str(slot)]
        row = next(r for r in loaded if r["device_id"] == device)
        if (
            source["device_id"] != device
            or row["observed_selected_tensor_sha256"]
            != source["selected_tensor_sha256"]
        ):
            raise ValueError("phase checkpoint owner/selected bytes differ")
    return dict(
        source_db=594,
        capsule_sha256=CAPSULE_SHA,
        source_tag=original["source_tag"],
        slots_by_device={str(d): s for d, s in local_slots.items()},
    )


def check_observation(
    original: Mapping[str, Any],
    *,
    slot: int,
    kind: str,
    values: Mapping[str, np.ndarray],
) -> dict:
    """Exact reproduction of a sealed executing boundary, not new math equality."""
    if (
        type(slot) is not int
        or not 0 <= slot < 32
        or kind not in (*COMPONENTS, "wk_decode", "wk_promote")
    ):
        raise ValueError("unregistered phase original slot/component")
    observed = manifest(values)
    owner = original["owners"][str(slot)]
    expected = owner[kind] if kind.startswith("wk_") else owner["components"][kind]
    if observed != expected:
        raise ValueError(f"phase original output mismatch: slot{slot}/{kind}")
    return observed


class OriginalVerifier:
    """One retained capture; later samples must match the same sealed bytes.

    Call within BudgetedCalls' voted preservation hook, after timing. No JAX
    execution or cloud transfer is introduced by these host comparisons.
    """

    def __init__(
        self, calls: Any, original: Mapping[str, Any], inputs: Mapping[str, np.ndarray]
    ) -> None:
        from scripts.greenfield.prefill_layer_evidence import encode_arrays

        self.calls, self.original = calls, original
        self.binding = bind_originals(calls.record, calls.local_slots, original)
        if manifest(inputs) != original["inputs"]:
            raise ValueError("phase fixture does not reproduce original input bytes")
        self.arrays = encode_arrays("input", inputs)
        self.visits = {kind: 0 for kind in COMPONENTS}
        self.report = dict(binding=self.binding, visits=self.visits, complete=False)
        calls.record["original_authentication"] = self.report

    def capture(
        self, kind: str, observed: Mapping[int, Mapping[str, np.ndarray]]
    ) -> None:
        from scripts.greenfield.prefill_window_worker import save_arrays

        encoded = {}
        for device, values in observed.items():
            encoded.update(completed.encode(f"{kind}_{device}", values))
        first = self.visits[kind] == 0
        if first:
            self.arrays.update(encoded)
            # Preserve originals before owner/schema/hash rejection.
            self.report["first_npz_sha256"] = save_arrays(
                self.calls.root / "phase_first.npz", self.arrays
            )
        try:
            if set(observed) != set(self.calls.local_slots):
                raise ValueError("phase observation owners differ")
            for device, values in observed.items():
                check_observation(
                    self.original,
                    slot=self.calls.local_slots[device],
                    kind=kind,
                    values=values,
                )
        except Exception:
            if not first:
                self.report["failure_npz_sha256"] = save_arrays(
                    self.calls.root / "phase_failure.npz", encoded
                )
            self.report["failure_component"] = kind
            raise
        self.visits[kind] += 1
        if all(n >= 1 for n in self.visits.values()):
            self.arrays.clear()  # keep no first-traversal host tensor generation

    def component(self, kind: str, values: tuple, fields: tuple) -> None:
        self.capture(kind, completed.observe(values, fields))

    def assembly(self, rows: tuple, last_prefix: tuple) -> None:
        from scripts.greenfield.prefill_completed_window_assembly import attach_result
        from scripts.greenfield.prefill_layer_evidence import local_observations

        for kind, value in zip(("actual", "control"), rows, strict=True):
            self.capture(kind, local_observations(attach_result(value, last_prefix)))

    def finish(self, traversals: int) -> None:
        if any(v != traversals for v in self.visits.values()) or self.arrays:
            raise ValueError("phase original verification traversal coverage differs")
        self.report["complete"] = True


if __name__ == "__main__":
    import argparse
    from scripts.greenfield.microbench_fp8_matmul import _atomic_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refuse to replace an existing original capsule")
    capsule = build_capsule(args.root)
    _atomic_json(args.output, capsule)
    print(
        json.dumps(
            dict(
                path=str(args.output),
                bytes=args.output.stat().st_size,
                sha256=sha256(args.output.read_bytes()).hexdigest(),
                owners=len(capsule["owners"]),
            )
        )
    )
