#!/usr/bin/env python3
"""Distinct layer collector inside the existing guarded FP8 campaign wrapper.

The shell wrapper owns both leases, pre/post normal+root censuses, DB and final
generation publication. This module reuses its established SSH and exact-file
publisher. Do not invoke campaign directly or interpret a worker SUCCESS as a
protected admission. There is no infrastructure management here.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import ipaddress
import json
from pathlib import Path
import re
import shlex
import socket
import stat
import sys
from typing import Any

import numpy as np

from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from scripts.greenfield.prefill_layer_evidence import replay_case
from scripts.greenfield.prefill_layer_hlo import check_layer_hlo
from scripts.greenfield.prefill_layer_numerical import CASES, PROTOCOL
from scripts.greenfield.probe_ws32_prefill_layer import (
    KERNEL,
    PINS,
    PAYLOAD_BYTES,
    REPO,
    layer_from_tag,
    pins_for_layer,
)
from scripts.greenfield.probe_ws32_prefill_moe import FLEET_SHA, MESH_SHA, TOPOLOGY_SHA
from scripts.greenfield.ws32_prefill_moe_campaign import ssh
from scripts.greenfield import prefill_router_protocol as router_protocol
from scripts.greenfield import prefill_materialized_reference as materialized_ref
from scripts.greenfield import prefill_prefix_mlp_protocol as prefix_mlp_protocol
from scripts.greenfield import prefill_observed_reference as observed_ref
from scripts.greenfield import prefill_window_acquisition as window_acquisition
from scripts.greenfield import prefill_window_protocol as window_protocol
from scripts.greenfield import prefill_window_evidence as window_evidence
from scripts.greenfield import prefill_window_boundary_evidence as boundary_evidence
from scripts.greenfield import prefill_window_boundary_worker as boundary_protocol


def materialized_protocol(materialized: bool, observed: bool) -> Any:
    if observed and not materialized:
        raise ValueError("observed reference requires materialized classification")
    return observed_ref if observed else materialized_ref


def diagnostic_protocol(diagnostic: bool, prefix_mlp: bool) -> Any:
    if prefix_mlp and not diagnostic:
        raise ValueError("prefix MLP requires diagnostic classification")
    return prefix_mlp_protocol if prefix_mlp else router_protocol


def run_root(tag: str) -> Path:
    layer_from_tag(tag)
    return Path("/home/gianl/glm-run") / tag


def program_names(
    layer: int,
    *,
    diagnostic: bool = False,
    materialized: bool = False,
    prefix_mlp: bool = False,
    observed: bool = False,
    completed_window: bool = False,
) -> tuple[str, ...]:
    rp = diagnostic_protocol(diagnostic, prefix_mlp)
    mr = materialized_protocol(materialized, observed)
    if layer == 6 and not any((diagnostic, materialized, prefix_mlp, observed)):
        return window_acquisition.acquisition_mode(completed=completed_window)[3]
    if completed_window:
        raise ValueError("completed window requires distinct layer6 mode")
    if layer not in (0, 3):
        raise ValueError("unregistered layer")
    if materialized:
        if diagnostic or layer != 3:
            raise ValueError("materialized reference requires layer3 admission")
        return mr.PROGRAMS
    if diagnostic:
        if layer != 3:
            raise ValueError("router diagnostic requires layer3")
        return rp.PROGRAMS
    return (
        ("candidate", "reference", "wk_decode", "wk_promote", "repair")
        if layer == 0
        else ("candidate", "reference")
    )


def evidence_files(
    layer: int,
    *,
    diagnostic: bool = False,
    materialized: bool = False,
    prefix_mlp: bool = False,
    observed: bool = False,
    window_numerical: bool = False,
    boundary_diagnostic: bool = False,
    completed_window: bool = False,
) -> tuple[str, ...]:
    if completed_window and (
        layer != 6
        or any(
            (
                diagnostic,
                materialized,
                prefix_mlp,
                observed,
                window_numerical,
                boundary_diagnostic,
            )
        )
    ):
        raise ValueError("completed window evidence is compile-only and distinct")
    if boundary_diagnostic and (
        layer != 6
        or any((diagnostic, materialized, prefix_mlp, observed, window_numerical))
    ):
        raise ValueError("boundary diagnostic evidence requires distinct layer6 mode")
    if window_numerical and (
        layer != 6 or any((diagnostic, materialized, prefix_mlp, observed))
    ):
        raise ValueError("window numerical evidence requires its distinct layer6 mode")
    return (
        "runner.json",
        "retained_preflight.json",
        "worker.log",
        *(("compile_journal.jsonl",) if layer == 6 else ()),
        *(
            ("wk_decode.npz", "wk_promote.npz", "wk_boundary.npz")
            if window_numerical or boundary_diagnostic
            else ()
        ),
        *(("boundary_prefix.npz",) if observed else ()),
        *(
            f"{name}.{form}"
            for name in program_names(
                layer,
                diagnostic=diagnostic,
                materialized=materialized,
                prefix_mlp=prefix_mlp,
                observed=observed,
                completed_window=completed_window,
            )
            for form in ("stablehlo.mlir", "optimized_hlo.txt")
        ),
        *(
            f"{case}.npz"
            for case in (
                ("boundary",)
                if boundary_diagnostic
                else (
                    window_protocol.CASES
                    if window_numerical
                    else () if layer == 6 else ("boundary",) if diagnostic else CASES
                )
            )
        ),
        *(f"{case}.reference_input.npz" for case in (CASES if materialized else ())),
    )


def checkpoint_ledger(layer: int) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Read only fixed hash-bound metadata, no checkpoint payload or full copy."""
    pins = json.loads(pins_for_layer(layer).read_text())
    root = Path(pins["checkpoint_root"])
    data = (root / "manifest.json").read_bytes()
    if sha256(data).hexdigest() != pins["manifest_file_sha256"]:
        raise ValueError("controller retained manifest differs")
    if (
        sha256((root / "SUCCESS").read_bytes()).hexdigest()
        != pins["success_file_sha256"]
    ):
        raise ValueError("controller retained SUCCESS differs")
    manifest = json.loads(data)
    selected = {
        i: t["name"]
        for i, t in enumerate(manifest["tensor_schema"])
        if t["name"].startswith(f"model.layers.{layer}.")
    }
    if len(selected) != {0: 27, 3: 28, 6: 35}[layer]:
        raise ValueError("retained selected tensor inventory differs")
    return pins, {
        r["device_slot"]: {
            "full_sha256": r["sha256"],
            "selected": {name: r["tensor_sha256"][i] for i, name in selected.items()},
        }
        for r in manifest["files"]
    }


def retained_preflight(tag: str, rank: int, pin: str) -> None:
    """Authenticate four retained headers before ANY worker initializes JAX."""
    import os
    from scripts.greenfield.microbench_fp8_matmul import _git_head

    layer = layer_from_tag(tag)
    if type(rank) is not int or not 0 <= rank < 8 or _git_head() != pin:
        raise ValueError("retained preflight rank/code differs")
    if window_acquisition.is_numerical_tag(tag):
        from scripts.greenfield.prefill_window_admission import registered_programs

        registered_programs()  # All eight checks finish before any TPU initialization.
    if window_acquisition.is_boundary_diagnostic_tag(tag):
        from scripts.greenfield.prefill_window_boundary_admission import (
            registered_programs,
        )

        registered_programs()
        boundary_protocol.original_receipt()
    pins, _ = checkpoint_ledger(layer)
    checkpoint = Path(pins["checkpoint_root"])
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    expected = {r["filename"]: r for r in manifest["files"]}
    paths = [p for p in checkpoint.iterdir() if p.suffix == ".safetensors"]
    if len(paths) != 4:
        raise ValueError("retained host must have exactly four final-owner files")
    headers = []
    for path in sorted(paths):
        if path.is_symlink() or path.name not in expected:
            raise ValueError("retained slot filename/type differs")
        row = expected[path.name]
        with path.open("rb") as stream:
            observed = os.fstat(stream.fileno())
            data = stream.read(row["header_bytes"])
        digest = sha256(data).hexdigest()
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_size != row["file_bytes"]
            or digest != row["header_sha256"]
        ):
            raise ValueError("retained file header/size differs")
        headers.append(
            dict(
                device_slot=row["device_slot"],
                filename=path.name,
                file_bytes=observed.st_size,
                header_sha256=digest,
            )
        )
    root = run_root(tag) / f"rank{rank}"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "retained_preflight.json"
    if path.exists():
        raise FileExistsError(path)
    _atomic_json(
        path,
        dict(
            code_hash=pin,
            layer=layer,
            launch_rank=rank,
            hostname=socket.gethostname(),
            checkpoint_root=str(checkpoint),
            headers=headers,
            scope="HEADERS_AND_FILE_SIZES_ONLY_NOT_PAYLOAD_OR_LIVE_TOPOLOGY",
        ),
    )
    print(f"PREFILL_RETAINED_OK {socket.gethostname()}", flush=True)


def validate_workers(
    records: list[dict[str, Any]],
    pin: str,
    *,
    layer: int,
    pins: dict[str, Any],
    ledger: dict[int, dict[str, Any]],
    diagnostic: bool = False,
    materialized: bool = False,
    prefix_mlp: bool = False,
    observed: bool = False,
    window_numerical: bool = False,
    window_boundary: bool = False,
    boundary_diagnostic: bool = False,
    completed_window: bool = False,
) -> None:
    if completed_window and (
        layer != 6
        or any(
            (
                diagnostic,
                materialized,
                prefix_mlp,
                observed,
                window_numerical,
                window_boundary,
                boundary_diagnostic,
            )
        )
    ):
        raise ValueError("completed window fleet is compile-only and distinct")
    if boundary_diagnostic and (
        layer != 6
        or any(
            (
                diagnostic,
                materialized,
                prefix_mlp,
                observed,
                window_numerical,
                window_boundary,
            )
        )
    ):
        raise ValueError("boundary diagnostic requires its distinct numerical mode")
    if window_boundary and (
        layer != 6
        or any((diagnostic, materialized, prefix_mlp, observed, window_numerical))
    ):
        raise ValueError("boundary acquisition requires its distinct compile-only mode")
    if window_numerical and (
        layer != 6 or any((diagnostic, materialized, prefix_mlp, observed))
    ):
        raise ValueError("window numerical fleet requires its distinct layer6 mode")
    rp = diagnostic_protocol(diagnostic, prefix_mlp)
    mr = materialized_protocol(materialized, observed)
    if len(records) != 8 or {r["launch_rank"] for r in records} != set(range(8)):
        raise ValueError("need eight unique layer worker ranks")
    if len({r["hostname"] for r in records}) != 8 or {
        r["jax_process_index"] for r in records
    } != set(range(8)):
        raise ValueError("fleet physical host/process identity differs")
    for name in program_names(
        layer,
        diagnostic=diagnostic,
        materialized=materialized,
        prefix_mlp=prefix_mlp,
        observed=observed,
        completed_window=completed_window,
    ):
        for form in ("stablehlo_sha256", "optimized_hlo_sha256"):
            hashes = {r["programs"][name][form] for r in records}
            if len(hashes) != 1 or not re.fullmatch(
                r"[0-9a-f]{64}", next(iter(hashes))
            ):
                raise ValueError("fleet graph identities differ")
    device_orders = {
        tuple(np.asarray(r["physical_device_ids"]).reshape(-1)) for r in records
    }
    if len(device_orders) != 1:
        raise ValueError("fleet physical mesh orders differ")
    order = next(iter(device_orders))
    if len(order) != 32 or len(set(order)) != 32:
        raise ValueError("physical mesh does not name32 distinct devices")
    if layer == 6:
        module = (
            boundary_evidence
            if boundary_diagnostic
            else window_evidence if window_numerical else window_acquisition
        )
        options = (
            {}
            if window_numerical or boundary_diagnostic
            else dict(
                capture_boundaries=window_boundary, completed_window=completed_window
            )
        )
        module.validate_workers(
            records, pin=pin, pins=pins, ledger=ledger, order=order, **options
        )
        return
    slots = []
    for r in records:
        if not (
            r["status"] == "SUCCESS"
            and r["protocol"]
            == (
                rp.PROTOCOL if diagnostic else mr.PROTOCOL if materialized else PROTOCOL
            )
            and r["code_hash"] == pin
            and r["layer"] == layer
            and r["selected_layer_ids"] == [layer]
            and r["admission_only"] is (not diagnostic)
            and r.get("diagnostic_only", False) is diagnostic
            and r["performance_claim"] is False
            and r["iterations"] == 0
            and r["latency"] is None
            and r["rows"] == 17
            and r["reference_scope"]
            == (
                mr.REFERENCE_SCOPE
                if materialized
                else (
                    prefix_mlp_protocol.REFERENCE_SCOPE
                    if prefix_mlp
                    else "RAW_SCALAR_NOT_PROMOTED_DECODER_OR_LEGACY"
                )
            )
            and r["state_scope"] == "REAL_WEIGHTS_SYNTHETIC_PREFIX_AND_ACTIVATIONS"
            and r["integrity_scope"]
            == "selected_layer_tensors_only_not_complete_checkpoint"
            and r["checkpoint_pins"] == pins
            and r["payload_bytes_per_chip"] == PAYLOAD_BYTES[layer]
            and r["mesh_sha256"] == MESH_SHA
            and r["topology_sha256"] == TOPOLOGY_SHA
            and r["topology_fleet_sha256"] == FLEET_SHA
            and r["pid"] > 0
            and r["start_ticks"] > 0
            and bool(r["boot_id"])
            and set(r["programs"])
            == set(
                program_names(
                    layer,
                    diagnostic=diagnostic,
                    materialized=materialized,
                    prefix_mlp=prefix_mlp,
                    observed=observed,
                )
            )
            and set(r["cases"]) == ({"boundary"} if diagnostic else set(CASES))
            and r["hlo"]["contract"]["passed"] is True
        ):
            raise ValueError("layer worker scope/provenance differs")
        if len(r["local_device_slots"]) != 4:
            raise ValueError("layer loader does not cover four local owners")
        local = []
        for s in r["local_device_slots"]:
            slot = s["device_slot"]
            if not (
                type(slot) is int
                and 0 <= slot < 32
                and order[slot] == s["device_id"]
                and s["observed_selected_tensor_sha256"] == ledger[slot]["selected"]
                and s["expected_full_file_sha256_not_verified"]
                == ledger[slot]["full_sha256"]
                and s["selected_payload_bytes"] == PAYLOAD_BYTES[layer]
            ):
                raise ValueError("selected tensor/owner/checkpoint binding differs")
            local.append(slot)
        if len(set(local)) != 4:
            raise ValueError("duplicate selected owner")
        slots.extend(local)
        for name, p in r["programs"].items():
            m = p["compiled_memory"]
            sizes = [
                m[k]
                for k in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "temp_size_in_bytes",
                )
            ]
            if any(type(n) is not int or n < 0 for n in sizes) or (
                (diagnostic or materialized or name == "candidate")
                and sum(sizes) > 2 * 1024**3
            ):
                raise ValueError("complete-layer compiler allocation budget differs")
        stats = r["device_memory_stats_including_reference"]
        if (
            len(stats) != 4
            or {s["device_id"] for s in stats} != {order[slot] for slot in local}
            or any(
                not 0 < s["stats"]["peak_bytes_in_use"] < s["stats"]["bytes_limit"]
                for s in stats
            )
        ):
            raise ValueError("measured32-chip HBM/headroom missing")
        if diagnostic:
            c = r["cases"]["boundary"]
            if (
                c["evidence_complete"] is not True
                or c["replay"]["evidence_complete"] is not True
                or c["replay"]["numerical_admission"] is not False
                or set(c["replay"]["owners"]) != {str(order[s]) for s in local}
            ):
                raise ValueError("router diagnostic incomplete or misclassified")
            continue
        if observed:
            b = r.get("boundary_prefix", {})
            if not re.fullmatch(r"[0-9a-f]{64}", b.get("npz_sha256", "")) or set(
                b.get("replay", {}).get("owners", {})
            ) != {str(order[s]) for s in local}:
                raise ValueError("observed reference prefix proof missing")
        for case in CASES:
            c = r["cases"][case]
            if materialized and (
                not re.fullmatch(
                    r"[0-9a-f]{64}", r.get("reference_input_sha256", {}).get(case, "")
                )
                or any(
                    r["programs"][n].get("hlo_contract", {}).get("passed") is not True
                    for n in ("reference_prefix", "reference")
                )
            ):
                raise ValueError("materialized reference boundary evidence missing")
            if (
                c["passed"] is not True
                or c["replay"]["passed"] is not True
                or set(c["replay"]["owners"]) != {str(order[s]) for s in local}
            ):
                raise ValueError("complete-layer case evidence failed/incomplete")
    if len(slots) != 32 or set(slots) != set(range(32)):
        raise ValueError("layer fleet does not cover all32 owners")
    for case in ("boundary",) if diagnostic else CASES:
        if (
            len(
                {
                    json.dumps(r["cases"][case]["input_sha256"], sort_keys=True)
                    for r in records
                }
            )
            != 1
        ):
            raise ValueError("replicated layer inputs differ across hosts")


def validate_files(
    root: Path,
    record: dict[str, Any],
    *,
    diagnostic: bool = False,
    materialized: bool = False,
    prefix_mlp: bool = False,
    observed: bool = False,
) -> None:
    rp = diagnostic_protocol(diagnostic, prefix_mlp)
    mr = materialized_protocol(materialized, observed)
    raw = (root / "retained_preflight.json").read_bytes()
    preflight = json.loads(raw)
    if (
        sha256(raw).hexdigest() != record["retained_preflight_sha256"]
        or preflight["code_hash"] != record["code_hash"]
        or preflight["layer"] != record["layer"]
        or preflight["launch_rank"] != record["launch_rank"]
        or preflight["hostname"] != record["hostname"]
        or len(preflight["headers"]) != 4
        or {h["device_slot"] for h in preflight["headers"]}
        != {s["device_slot"] for s in record["local_device_slots"]}
    ):
        raise ValueError("retained preflight is not bound to the executing owners")
    if record["layer"] == 6:
        module = (
            boundary_evidence
            if record.get("protocol") == boundary_protocol.PROTOCOL
            else (
                window_evidence
                if record.get("protocol") == window_protocol.PROTOCOL
                else window_acquisition
            )
        )
        module.validate_files(root, record)
        return
    for name in program_names(
        record["layer"],
        diagnostic=diagnostic,
        materialized=materialized,
        prefix_mlp=prefix_mlp,
        observed=observed,
    ):
        for form, key in (
            ("stablehlo.mlir", "stablehlo_sha256"),
            ("optimized_hlo.txt", "optimized_hlo_sha256"),
        ):
            if (
                sha256((root / f"{name}.{form}").read_bytes()).hexdigest()
                != record["programs"][name][key]
            ):
                raise ValueError("original program bytes differ")
        if diagnostic:
            proof = rp.check_hlo((root / f"{name}.optimized_hlo.txt").read_text(), name)
            if (
                not proof["passed"]
                or json.loads(json.dumps(proof))
                != record["programs"][name]["hlo_contract"]
            ):
                raise ValueError("router diagnostic original HLO differs")
        if materialized and name in ("reference_prefix", "reference"):
            proof = mr.check_reference_hlo(
                (root / f"{name}.optimized_hlo.txt").read_text(), name
            )
            if (
                not proof["passed"]
                or json.loads(json.dumps(proof))
                != record["programs"][name]["hlo_contract"]
            ):
                raise ValueError("materialized reference original HLO differs")
    if diagnostic:
        if (
            record["hlo"]["sha256"]
            != record["programs"]["candidate"]["optimized_hlo_sha256"]
        ):
            raise ValueError("router candidate HLO binding differs")
        path = root / "boundary.npz"
        if (
            sha256(path.read_bytes()).hexdigest()
            != record["cases"]["boundary"]["npz_sha256"]
        ):
            raise ValueError("router original NPZ differs")
        _, ledger = checkpoint_ledger(3)
        slots = {s["device_id"]: s["device_slot"] for s in record["local_device_slots"]}
        replay = rp.replay_file(path, slots, ledger)
        if replay != record["cases"]["boundary"]["replay"]:
            raise ValueError("router original-array replay differs")
        return
    hlo = (root / "candidate.optimized_hlo.txt").read_text()
    proof = check_layer_hlo(hlo, layer=record["layer"])
    # JSON-normalize tuples in the parser's dictionaries before exact comparison.
    if (
        not proof["passed"]
        or json.loads(json.dumps(proof)) != record["hlo"]["contract"]
        or sha256(hlo.encode()).hexdigest() != record["hlo"]["sha256"]
    ):
        raise ValueError("original HLO proof differs/fails")
    slots = {s["device_id"]: s["device_slot"] for s in record["local_device_slots"]}
    if observed:
        _, ledger = checkpoint_ledger(3)
        observed_ref.validate_boundary_binding(root, record, slots, ledger)
    for case in CASES:
        if materialized:
            mr.verify_input_capture(
                root / f"{case}.reference_input.npz",
                record["reference_input_sha256"][case],
                set(slots),
                CASES[case][1],
            )
        path = root / f"{case}.npz"
        if sha256(path.read_bytes()).hexdigest() != record["cases"][case]["npz_sha256"]:
            raise ValueError("original layer NPZ differs")
        replay = replay_case(
            path, layer=record["layer"], case=case, slots_by_device=slots
        )
        if (
            not replay["passed"]
            or replay != record["cases"][case]["replay"]
            or replay["input_sha256"] != record["cases"][case]["input_sha256"]
        ):
            raise ValueError("controller original-array replay differs/fails")


def validate_record(
    record: dict[str, Any],
    pin: str,
    *,
    diagnostic: bool = False,
    materialized: bool = False,
    prefix_mlp: bool = False,
    observed: bool = False,
) -> None:
    rp = diagnostic_protocol(diagnostic, prefix_mlp)
    mr = materialized_protocol(materialized, observed)
    if record.get("kernel") == boundary_protocol.KERNEL:
        if any((diagnostic, materialized, prefix_mlp, observed)):
            raise ValueError("boundary diagnostic cannot use historical layer mode")
        boundary_evidence.validate_record(record, pin)
        return
    if record.get("kernel") == window_protocol.KERNEL:
        if any((diagnostic, materialized, prefix_mlp, observed)):
            raise ValueError("window numerical cannot use an historical layer mode")
        window_evidence.validate_record(record, pin)
        return
    if record.get("kernel") in (
        window_acquisition.KERNEL,
        window_acquisition.BOUNDARY_KERNEL,
        window_acquisition.COMPLETED_KERNEL,
    ):
        if any((diagnostic, materialized, prefix_mlp, observed)):
            raise ValueError("window acquisition cannot use an historical layer mode")
        window_acquisition.validate_record(record, pin)
        return
    if not (
        record["status"] == "SUCCESS"
        and record["kernel"]
        == (rp.KERNEL if diagnostic else mr.KERNEL if materialized else KERNEL)
        and record["protocol"]
        == (rp.PROTOCOL if diagnostic else mr.PROTOCOL if materialized else PROTOCOL)
        and record["code_hash"] == pin
        and record["admission_only"] is (not diagnostic)
        and record["baseline_only"] is False
        and record["diagnostic_only"] is diagnostic
        and record["performance_claim"] is False
        and record["latency"] is None
        and record["warmup"] == record["iterations"] == 0
        and record["profiler_free_timing"] is False
    ):
        raise ValueError("complete-layer aggregate classification differs")
    pins, ledger = checkpoint_ledger(record["layer"])
    validate_workers(
        record["workers"],
        pin,
        layer=record["layer"],
        pins=pins,
        ledger=ledger,
        diagnostic=diagnostic,
        prefix_mlp=prefix_mlp,
        observed=observed,
        materialized=materialized,
    )


def publish_rank(tag: str, rank: int) -> None:
    from google.cloud import storage
    from scripts.greenfield.collect_ws32_worker_evidence import (
        digest_file,
        publish_exact,
    )

    if type(rank) is not int or not 0 <= rank < 8:
        raise ValueError("invalid layer publication rank")
    root = run_root(tag) / f"rank{rank}"
    bucket = storage.Client().bucket("driftbench-dsv4-uc")
    receipts = []
    for name in evidence_files(
        layer_from_tag(tag),
        diagnostic=router_protocol.is_router_tag(tag)
        or prefix_mlp_protocol.is_prefix_mlp_tag(tag),
        prefix_mlp=prefix_mlp_protocol.is_prefix_mlp_tag(tag),
        materialized=materialized_ref.is_materialized_tag(tag)
        or observed_ref.is_observed_tag(tag),
        observed=observed_ref.is_observed_tag(tag),
        window_numerical=window_acquisition.is_numerical_tag(tag),
        boundary_diagnostic=window_acquisition.is_boundary_diagnostic_tag(tag),
        completed_window=window_acquisition.is_completed_tag(tag),
    ):
        path = root / name
        if path.is_file():
            receipts.append(
                publish_exact(
                    bucket,
                    f"results/{tag}/workers/rank{rank}/{name}",
                    path,
                    digest_file(path),
                    compressed=False,
                )
            )
    ledger = root / "worker_receipts.json"
    ledger.write_text(json.dumps(receipts, sort_keys=True) + "\n")
    publish_exact(
        bucket,
        f"results/{tag}/workers/rank{rank}/worker_receipts.json",
        ledger,
        digest_file(ledger),
        compressed=False,
    )


def collect(tag: str, pin: str) -> dict[str, Any]:
    from google.cloud import storage

    root = run_root(tag)
    layer = layer_from_tag(tag)
    window = window_acquisition.is_acquisition_tag(tag)
    window_numerical = window_acquisition.is_numerical_tag(tag)
    window_boundary = window_acquisition.is_boundary_tag(tag)
    completed_window = window_acquisition.is_completed_tag(tag)
    boundary_diagnostic = window_acquisition.is_boundary_diagnostic_tag(tag)
    prefix_mlp = prefix_mlp_protocol.is_prefix_mlp_tag(tag)
    diagnostic = router_protocol.is_router_tag(tag) or prefix_mlp
    rp = diagnostic_protocol(diagnostic, prefix_mlp)
    observed = observed_ref.is_observed_tag(tag)
    materialized = materialized_ref.is_materialized_tag(tag) or observed
    mr = materialized_protocol(materialized, observed)
    bucket = storage.Client().bucket("driftbench-dsv4-uc")
    records = []
    for rank in range(8):
        prefix = f"results/{tag}/workers/rank{rank}/"
        blob = bucket.get_blob(prefix + "worker_receipts.json")
        if blob is None:
            raise ValueError(f"missing rank{rank} receipt ledger")
        ledger_bytes = blob.download_as_bytes(if_generation_match=blob.generation)
        receipts = json.loads(ledger_bytes)
        files = evidence_files(
            layer,
            diagnostic=diagnostic,
            materialized=materialized,
            prefix_mlp=prefix_mlp,
            observed=observed,
            window_numerical=window_numerical,
            boundary_diagnostic=boundary_diagnostic,
            completed_window=completed_window,
        )
        if len(receipts) != len(files) or {r["name"] for r in receipts} != {
            prefix + n for n in files
        }:
            raise ValueError(f"incomplete rank{rank} original evidence")
        destination = root / "fleet" / f"rank{rank}"
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "worker_receipts.json").write_bytes(ledger_bytes)
        _atomic_json(
            destination / "ledger_source.json",
            dict(
                name=blob.name,
                generation=str(blob.generation),
                size=int(blob.size),
                crc32c=blob.crc32c,
                sha256=sha256(ledger_bytes).hexdigest(),
            ),
        )
        for receipt in receipts:
            generation = int(receipt["generation"])
            obj = bucket.blob(receipt["name"], generation=generation)
            obj.reload(if_generation_match=generation)
            if int(obj.size) != receipt["size"] or obj.crc32c != receipt["crc32c"]:
                raise ValueError("worker generation size/CRC differs")
            data = obj.download_as_bytes(if_generation_match=generation)
            if sha256(data).hexdigest() != receipt["original_sha256"]:
                raise ValueError("worker generation SHA differs")
            (destination / Path(receipt["name"]).name).write_bytes(data)
        record = json.loads((destination / "runner.json").read_text())
        if record["launch_rank"] != rank or record["layer"] != layer:
            raise ValueError("worker identity differs from publication path")
        validate_files(
            destination,
            record,
            diagnostic=diagnostic,
            materialized=materialized,
            prefix_mlp=prefix_mlp,
            observed=observed,
        )
        records.append(record)
    pins, ledger = checkpoint_ledger(layer)
    validate_workers(
        records,
        pin,
        layer=layer,
        pins=pins,
        ledger=ledger,
        diagnostic=diagnostic,
        prefix_mlp=prefix_mlp,
        observed=observed,
        materialized=materialized,
        window_numerical=window_numerical,
        window_boundary=window_boundary,
        boundary_diagnostic=boundary_diagnostic,
        completed_window=completed_window,
    )
    if diagnostic:
        rp.verify_fleet_replicas(root / "fleet", records)
    result = dict(
        status="SUCCESS",
        code_hash=pin,
        kernel=(rp.KERNEL if diagnostic else mr.KERNEL if materialized else KERNEL),
        protocol=(
            rp.PROTOCOL if diagnostic else mr.PROTOCOL if materialized else PROTOCOL
        ),
        layer=layer,
        admission_only=not diagnostic,
        baseline_only=False,
        diagnostic_only=diagnostic,
        performance_claim=False,
        latency=None,
        profiler_free_timing=False,
        warmup=0,
        iterations=0,
        selected_route_case=None,
        device_kind="TPU v4",
        workers=records,
        hlo={"sha256": records[0]["hlo"]["sha256"], "contract": {"passed": True}},
        comparison={
            "passed": None if diagnostic else True,
            "diagnostic_evidence_complete": diagnostic,
        },
        checksum=sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
    )
    if window:
        acquisition_kernel, acquisition_protocol, _, _ = (
            window_acquisition.acquisition_mode(
                boundary=window_boundary, completed=completed_window
            )
        )
        result.update(
            kernel=acquisition_kernel,
            protocol=acquisition_protocol,
            admission_only=False,
            diagnostic_only=True,
            compile_only=True,
            numerical_execution_authorized=False,
            hlo=dict(
                sha256=records[0]["hlo"]["sha256"],
                contract=dict(
                    passed=True,
                    scope="COMPILER_EVIDENCE_PRESENT_NOT_EXECUTION_ADMISSION",
                ),
            ),
            comparison=dict(passed=None, diagnostic_evidence_complete=True),
        )
    if window_numerical:
        from scripts.greenfield.prefill_window_admission import PROFILE

        result.update(
            kernel=window_protocol.KERNEL,
            protocol=window_protocol.PROTOCOL,
            profile=PROFILE,
            compile_only=False,
            numerical_execution_authorized=True,
            hlo=records[0]["hlo"],
        )
    if boundary_diagnostic:
        from scripts.greenfield.prefill_window_boundary_admission import PROFILE

        result.update(
            kernel=boundary_protocol.KERNEL,
            protocol=boundary_protocol.PROTOCOL,
            profile=PROFILE,
            compile_only=False,
            numerical_execution_authorized=True,
            admission_only=False,
            diagnostic_only=True,
            hlo=records[0]["hlo"],
            comparison=dict(passed=None, diagnostic_evidence_complete=True),
        )
    return result


def campaign(tag: str, pin: str) -> None:
    root = run_root(tag)
    if REPO != Path("/home/gianl/glm-tpu-topology-rewrite") or not re.fullmatch(
        r"[0-9a-f]{40}", pin
    ):
        raise ValueError("invalid complete-layer worktree/pin")
    # Same reviewed existing-repository deployment as the MoE adapter. No clone,
    # packer, provisioning or infrastructure operation. Wrapper already censused.
    command = (
        "set -euo pipefail; idx=${HOSTNAME##*-w-}; wt="
        + shlex.quote(str(REPO))
        + "; pin="
        + pin
        + "; "
        '[[ $idx =~ ^[0-7]$ && -e $wt/.git ]]; [[ -z $(git -C "$wt" status --porcelain) ]]; '
        'if [[ $idx != 0 ]]; then git -C "$wt" fetch -q origin rewrite/topology-first-decode; '
        'git -C "$wt" checkout -q --detach "$pin"; fi; '
        '[[ $(git -C "$wt" rev-parse HEAD) == "$pin" && -z $(git -C "$wt" status --porcelain) ]]; '
        'echo "PREFILL_SYNC_OK $(hostname)"'
    )
    ssh(command, output=root / "fleet_sync.log")
    markers = [
        line.split()[1]
        for line in (root / "fleet_sync.log").read_text().splitlines()
        if line.startswith("PREFILL_SYNC_OK ")
    ]
    if len(markers) != 8 or len(set(markers)) != 8:
        raise ValueError("fleet sync not eight unique hosts")
    command = (
        "set -euo pipefail; idx=${HOSTNAME##*-w-}; cd " + shlex.quote(str(REPO)) + "; "
        "JAX_PLATFORMS=cpu PYTHONPATH=. /home/gianl/vllm-env/bin/python "
        "-m scripts.greenfield.ws32_prefill_layer_campaign retained-preflight --tag "
        + shlex.quote(tag)
        + ' --rank "$idx" --pin '
        + pin
    )
    ssh(command, output=root / "retained_preflight.log")
    markers = [
        line.split()[1]
        for line in (root / "retained_preflight.log").read_text().splitlines()
        if line.startswith("PREFILL_RETAINED_OK ")
    ]
    if len(markers) != 8 or len(set(markers)) != 8:
        raise ValueError("retained preflight not eight unique hosts")
    ssh("hostname -I | awk '{print $1}'", output=root / "coordinator.log", worker="0")
    address = (
        str(
            ipaddress.ip_address(
                (root / "coordinator.log").read_text().strip().splitlines()[-1]
            )
        )
        + ":8476"
    )
    command = (
        "set -euo pipefail; idx=${HOSTNAME##*-w-}; tag="
        + shlex.quote(tag)
        + "; wt="
        + shlex.quote(str(REPO))
        + "; "
        'out=/home/gianl/glm-run/$tag/rank$idx; mkdir -p "$out"; cd "$wt"; '
        'upload(){ JAX_PLATFORMS=cpu PYTHONPATH="$wt" /home/gianl/vllm-env/bin/python '
        '-m scripts.greenfield.ws32_prefill_layer_campaign publish-rank --tag "$tag" --rank "$idx"; }; trap upload EXIT; '
        'GLM_GREENFIELD_RUN_TAG="$tag" JAX_PLATFORMS=tpu PYTHONPATH="$wt" '
        "timeout --kill-after=30s 600s /home/gianl/vllm-env/bin/python -u scripts/greenfield/probe_ws32_prefill_layer.py "
        "--expected-code-hash "
        + pin
        + " --coordinator-address "
        + shlex.quote(address)
        + ' --process-id "$idx" --output-dir "$out" >"$out/worker.log" 2>&1'
    )
    ssh(command, output=root / "fleet_launch.log", timeout=780)
    record = collect(tag, pin)
    _atomic_json(root / "runner.json", record)
    (root / "hlo").mkdir(exist_ok=True)
    (root / "hlo/candidate.optimized_hlo.txt").write_bytes(
        (root / "fleet/rank0/candidate.optimized_hlo.txt").read_bytes()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("campaign", "publish-rank", "retained-preflight")
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--pin")
    parser.add_argument("--rank", type=int)
    args = parser.parse_args()
    if args.mode == "campaign":
        campaign(args.tag, args.pin)
    elif args.mode == "retained-preflight":
        retained_preflight(args.tag, args.rank, args.pin)
    else:
        publish_rank(args.tag, args.rank)
