"""Read-only cloud/original-array diagnosis of the first B128 refusal.

This never promotes an admission, edits a comparator, or launches TPU work.
The sole generated output is a compact append-only local diagnostic receipt.
"""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from glm_tpu.greenfield.benchmarking.one_layer import (
    REAL_LAYER_OUTPUT_TOLERANCE,
    ROUTE_WEIGHT_TOLERANCE,
    compare_bounded_tensor,
)
from scripts.greenfield import prefill_window_protocol as protocol
from scripts.greenfield.prefill_layer_evidence import (
    INPUT_FIELDS,
    decode_arrays,
    equal_bytes,
    input_hashes,
    owner_inputs,
)
from scripts.greenfield.prefill_layer_numerical import FIELDS
from scripts.greenfield.ws32_prefill_layer_campaign import (
    checkpoint_ledger,
    evidence_files,
)

TAG = "greenfield_fp8_ws32_prefill_layer_window_numerical_l6_20260908T132240925858058Z"
PIN = "0f994e373dade76ede59da179da6856d17234e5d"
PRIOR = Path("docs/artifacts/prefill-window-boundary-refusal-20260908.json")
OUTPUT = Path("docs/artifacts/prefill-window-boundary-refusal-v2-20260908.json")


def validate_mapping(record: dict, mesh: list | None, processes: set) -> list:
    """Bind physical devices to slots before cache/feature attribution."""
    order = np.asarray(record["physical_device_ids"])
    if order.shape != (8, 4) or set(order.ravel().tolist()) != set(range(32)):
        raise ValueError("expected full physical8x4 mesh")
    if mesh is not None and record["physical_device_ids"] != mesh:
        raise ValueError("fleet physical mesh differs")
    process = record["jax_process_index"]
    if type(process) is not int or not 0 <= process < 8 or process in processes:
        raise ValueError("duplicate/invalid JAX process")
    local = record["local_device_slots"]
    if len(local) != 4 or len({r["device_slot"] for r in local}) != 4:
        raise ValueError("expected four distinct local slots")
    for owner in local:
        slot, device = owner["device_slot"], owner["device_id"]
        if type(slot) is not int or not 0 <= slot < 32 or order.ravel()[slot] != device:
            raise ValueError("physical device/slot mapping differs")
    if set(record["programs"]) != {"candidate", "control", "wk_decode", "wk_promote"}:
        raise ValueError("expected exactly four registered graphs")
    processes.add(process)
    return record["physical_device_ids"]


def bounded_rows(a: np.ndarray, b: np.ndarray, *, routes: bool = False) -> dict:
    """Apply the unchanged bound, reporting each failed row rather than hiding it."""
    tolerance = ROUTE_WEIGHT_TOLERANCE if routes else REAL_LAYER_OUTPUT_TOLERANCE
    aggregate = compare_bounded_tensor(a, b, tolerance)
    rows = [compare_bounded_tensor(x, y, tolerance) for x, y in zip(a, b)]
    return dict(
        aggregate=aggregate,
        failed_rows=[i for i, r in enumerate(rows) if not r["passed"]],
        worst_row_max_abs=max((r["error"]["max_abs"] for r in rows), default=0),
        worst_row_mean_abs=max((r["error"]["mean_abs"] for r in rows), default=0),
    )


def route_diagnosis(a: dict, b: dict) -> dict:
    """Keep order/set failures explicit; expert-aligned weights are diagnostic only."""
    ar, br = a["routes"], b["routes"]
    if ar.shape != br.shape or ar.ndim != 2 or ar.shape[1] != 8:
        raise ValueError("expected aligned top-eight rows")
    if any(len(set(row)) != 8 for row in np.concatenate((ar, br))):
        raise ValueError("duplicate route cannot be aligned")
    same_set = np.all(np.sort(ar, axis=1) == np.sort(br, axis=1), axis=1)
    same_order = np.all(ar == br, axis=1)
    aligned_a = np.take_along_axis(a["route_weights"], np.argsort(ar, axis=1), axis=1)
    aligned_b = np.take_along_axis(b["route_weights"], np.argsort(br, axis=1), axis=1)
    ids = np.flatnonzero(same_set)
    aligned = bounded_rows(aligned_a[ids], aligned_b[ids], routes=True)
    aligned["failed_original_rows"] = [int(ids[i]) for i in aligned["failed_rows"]]
    return dict(
        ordered_mismatch_rows=np.flatnonzero(~same_order).tolist(),
        set_mismatch_rows=np.flatnonzero(~same_set).tolist(),
        changed_routes={
            str(i): dict(actual=ar[i].tolist(), control=br[i].tolist())
            for i in np.flatnonzero(~same_order)
        },
        slotwise_weights=bounded_rows(
            a["route_weights"], b["route_weights"], routes=True
        ),
        same_set_expert_aligned_weights=aligned,
        alignment_scope="diagnostic_only_excludes_set_mismatch_rows",
    )


def owner_diagnosis(a: dict, b: dict, initial: dict, slot: int) -> dict:
    for values in (a, b):
        protocol._shape_contract(values)
        protocol._selection_contract(values, 505, 128)
        if not values["health"].all():
            raise ValueError("original boundary health failed")
    mask = np.zeros((8, 64), np.bool_)
    for _, page, row in protocol.written_addresses(slot, 505, 128):
        mask[page, row] = True
    cache = {}
    for name in ("kv", "index", "repair"):
        cache[name] = dict(
            candidate_control_exact=equal_bytes(a[name], b[name]),
            actual_untouched_exact=equal_bytes(a[name][~mask], initial[name][~mask]),
            control_untouched_exact=equal_bytes(b[name][~mask], initial[name][~mask]),
            written_bounds=bounded_rows(a[name][mask], b[name][mask]),
        )
    return dict(
        dsa_order_mismatch_rows=np.flatnonzero(
            np.any(a["positions"] != b["positions"], axis=1)
        ).tolist(),
        dsa_set_mismatch_rows=np.flatnonzero(
            np.any(
                np.sort(a["positions"], axis=1) != np.sort(b["positions"], axis=1),
                axis=1,
            )
        ).tolist(),
        counts_exact=equal_bytes(a["counts"], b["counts"]),
        own_selected_order_valid=True,
        selection_scope="boundary_below2048_full_coverage_not_competitive_cutoff",
        normalized_pre_attention_exact=equal_bytes(a["normalized"], b["normalized"]),
        cache=cache,
        tensors={
            name: bounded_rows(a[name], b[name]) for name in ("output", "residual")
        },
        router=route_diagnosis(a, b),
        actual_sha256=input_hashes(a),
        control_sha256=input_hashes(b),
    )


def main() -> None:
    from google.cloud import storage
    from glm_tpu.greenfield.kernels.reference.rotary import build_rotary_table_host

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    bucket = storage.Client().bucket("driftbench-dsv4-uc")
    _, tensors = checkpoint_ledger(6)
    canonical = protocol.host_case(
        "boundary", build_rotary_table_host(4096, rotary_dim=64, theta=8e6)
    )
    names = set(evidence_files(6, window_numerical=True)) - {
        "competitive.npz",
        "tail.npz",
    }
    owners, hosts, devices, graph_hashes, sources = {}, set(), set(), None, []
    mesh, processes = None, set()
    for rank in range(8):
        prefix = f"results/{TAG}/workers/rank{rank}/"
        ledger_blob = bucket.get_blob(prefix + "worker_receipts.json")
        if ledger_blob is None:
            raise ValueError("missing original ledger")
        ledger_raw = ledger_blob.download_as_bytes(
            if_generation_match=ledger_blob.generation, checksum="crc32c"
        )
        receipts = json.loads(ledger_raw)
        if len(receipts) != len(names) or {r["name"] for r in receipts} != {
            prefix + n for n in names
        }:
            raise ValueError("partial inventory differs from first boundary refusal")
        files = {}
        for r in receipts:
            generation = int(r["generation"])
            blob = bucket.blob(r["name"], generation=generation)
            blob.reload(if_generation_match=generation)
            if int(blob.size) != r["size"] or blob.crc32c != r["crc32c"]:
                raise ValueError("generation size/CRC differs")
            raw = blob.download_as_bytes(
                if_generation_match=generation, checksum="crc32c"
            )
            if len(raw) != r["size"] or sha256(raw).hexdigest() != r["original_sha256"]:
                raise ValueError("original SHA/length differs")
            files[r["name"].removeprefix(prefix)] = raw
        record = json.loads(files["runner.json"])
        mesh = validate_mapping(record, mesh, processes)
        if (
            record["status"] != "FAILED"
            or record["code_hash"] != PIN
            or record["launch_rank"] != rank
            or record["protocol"] != protocol.PROTOCOL
            or record["current_phase"] != "boundary/comparison"
            or len(record["call_evidence"]) != 7
        ):
            raise ValueError("original failed worker identity/phase differs")
        if record["hostname"] in hosts:
            raise ValueError("duplicate worker host")
        hosts.add(record["hostname"])
        hashes = {}
        for name, program in record["programs"].items():
            for suffix, key in (
                ("optimized_hlo.txt", "optimized_hlo_sha256"),
                ("stablehlo.mlir", "stablehlo_sha256"),
            ):
                digest = sha256(files[f"{name}.{suffix}"]).hexdigest()
                if digest != program[key]:
                    raise ValueError("graph differs from original worker binding")
                hashes[f"{name}.{suffix}"] = digest
        if graph_hashes is not None and graph_hashes != hashes:
            raise ValueError("fleet graph identities differ")
        graph_hashes = hashes
        with np.load(BytesIO(files["boundary.npz"]), allow_pickle=False) as arrays:
            inputs = decode_arrays(arrays, "input", INPUT_FIELDS)
            if any(not equal_bytes(inputs[n], canonical[n]) for n in INPUT_FIELDS):
                raise ValueError("original fixture differs from fixed protocol")
            for owner in record["local_device_slots"]:
                device, slot = owner["device_id"], owner["device_slot"]
                if (
                    slot in owners
                    or device in devices
                    or owner["observed_selected_tensor_sha256"]
                    != tensors[slot]["selected"]
                ):
                    raise ValueError("duplicate/mismatched selected owner")
                devices.add(device)
                a = decode_arrays(arrays, f"actual_{device}", FIELDS)
                b = decode_arrays(arrays, f"control_{device}", FIELDS)
                owners[slot] = owner_diagnosis(a, b, owner_inputs(inputs, slot), slot)
        sources.append(
            dict(
                rank=rank,
                hostname=record["hostname"],
                process_index=record["jax_process_index"],
                ledger=dict(
                    name=ledger_blob.name,
                    generation=str(ledger_blob.generation),
                    size=int(ledger_blob.size),
                    crc32c=ledger_blob.crc32c,
                    sha256=sha256(ledger_raw).hexdigest(),
                ),
                originals=receipts,
                phase_error=record["phase_error"],
            )
        )
        print(
            f"rank{rank}: original16 files verified, four owner diagnostics complete",
            flush=True,
        )
    if set(owners) != set(range(32)) or len(hosts) != 8 or len(devices) != 32:
        raise ValueError("incomplete fleet")
    for slot, owner in owners.items():
        for kind in ("actual_sha256", "control_sha256"):
            for name in (
                "output",
                "residual",
                "normalized",
                "positions",
                "counts",
                "scores",
                "routes",
                "route_weights",
                "health",
            ):
                if owner[kind][name] != owners[slot % 4][kind][name]:
                    raise ValueError("same-feature expert replicas disagree")
    result = dict(
        tag=TAG,
        pin=PIN,
        diagnostic_only=True,
        numerical_admission=False,
        performance_claim=False,
        original_verdict="FAILED",
        unexecuted_cases=["competitive", "tail"],
        analysis_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        original_sources=sources,
        graph_hashes=graph_hashes,
        input_sha256=input_hashes(canonical),
        same_feature_replicas_exact=True,
        physical_device_ids=mesh,
        jax_process_indices=sorted(processes),
        prior_diagnostic=dict(
            path=str(PRIOR),
            sha256=sha256(PRIOR.read_bytes()).hexdigest(),
            limitation="original analysis did not revalidate device-to-slot mapping; v2 adds it",
        ),
        owners={str(k): v for k, v in owners.items()},
    )
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            dict(
                output=str(OUTPUT),
                bytes=OUTPUT.stat().st_size,
                example=owners[0]["router"],
            )
        )
    )


if __name__ == "__main__":
    main()
