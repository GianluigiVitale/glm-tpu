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
from scripts.greenfield import prefill_completed_window_protocol as completed_protocol
from scripts.greenfield import prefill_phase_baseline as phase_protocol
from scripts.greenfield import prefill_phase_variant
from scripts.greenfield import prefill_phase_evidence as phase_evidence
from scripts.greenfield import prefill_rolled_window as rolled_protocol
from scripts.greenfield import prefill_rolled_evidence as rolled_evidence
from scripts.greenfield import ws32_dense_frontier_protocol as dense_protocol
from scripts.greenfield import ws32_dense_norm_protocol as norm_protocol
from scripts.greenfield import ws32_dense_canonical as canonical
from scripts.greenfield import ws32_history_protocol as history_protocol


def is_dense_tag(tag: str) -> bool:
    return dense_protocol.is_tag(tag) or norm_protocol.is_tag(tag) or canonical.is_tag(tag)


def dense_reference_kwargs(tag: str) -> dict:
    result = dict(original_root=Path("/home/gianl/glm-run")
                  / dense_protocol.ORIGINAL_TAG / "first_window_collected")
    if norm_protocol.is_tag(tag) or canonical.is_tag(tag):
        from scripts.greenfield.ws32_dense_norm_originals import TAG as original_tag

        result["norm_original_root"] = Path("/home/gianl/glm-run") / original_tag / "fleet"
    return result


def materialized_protocol(materialized: bool, observed: bool) -> Any:
    if observed and not materialized:
        raise ValueError("observed reference requires materialized classification")
    return observed_ref if observed else materialized_ref


def diagnostic_protocol(diagnostic: bool, prefix_mlp: bool) -> Any:
    if prefix_mlp and not diagnostic:
        raise ValueError("prefix MLP requires diagnostic classification")
    return prefix_mlp_protocol if prefix_mlp else router_protocol


def run_root(tag: str) -> Path:
    if not is_dense_tag(tag) and not history_protocol.is_tag(tag):
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
    completed_numerical: bool = False,
    phase_baseline: bool = False,
    rolled_window: bool = False,
) -> tuple[str, ...]:
    if rolled_window:
        if layer != 6 or any(
            (
                diagnostic,
                materialized,
                prefix_mlp,
                observed,
                completed_window,
                completed_numerical,
                phase_baseline,
            )
        ):
            raise ValueError("rolled layer requires distinct three-program mode")
        return rolled_protocol.PROGRAMS
    if phase_baseline:
        if (
            completed_numerical
            or completed_window
            or diagnostic
            or materialized
            or prefix_mlp
            or observed
            or layer != 6
        ):
            raise ValueError("phase baseline requires distinct layer6 mode")
        return phase_evidence.PROGRAMS
    rp = diagnostic_protocol(diagnostic, prefix_mlp)
    mr = materialized_protocol(materialized, observed)
    if completed_numerical:
        if layer != 6 or any(
            (diagnostic, materialized, prefix_mlp, observed, completed_window)
        ):
            raise ValueError("completed numerical requires distinct layer6 mode")
        from scripts.greenfield.prefill_completed_window_assembly import (
            PROGRAMS as helpers,
        )

        return (*window_acquisition.COMPLETED_PROGRAMS, *helpers)
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
    completed_numerical: bool = False,
    phase_baseline: bool = False,
    rolled_window: bool = False,
) -> tuple[str, ...]:
    if rolled_window:
        if layer != 6 or any(
            (
                diagnostic,
                materialized,
                prefix_mlp,
                observed,
                window_numerical,
                boundary_diagnostic,
                completed_window,
                completed_numerical,
                phase_baseline,
            )
        ):
            raise ValueError("rolled layer evidence requires distinct mode")
        return (
            "runner.json",
            "retained_preflight.json",
            "worker.log",
            "compile_journal.jsonl",
            *(
                f"{n}.{form}"
                for n in rolled_protocol.PROGRAMS
                for form in ("stablehlo.mlir", "optimized_hlo.txt")
            ),
            "wk_decode.npz",
            "wk_promote.npz",
            "candidate.npz",
        )
    if phase_baseline:
        if (
            any(
                (
                    diagnostic,
                    materialized,
                    prefix_mlp,
                    observed,
                    window_numerical,
                    boundary_diagnostic,
                    completed_window,
                    completed_numerical,
                )
            )
            or layer != 6
        ):
            raise ValueError("phase evidence requires distinct layer6 mode")
        return tuple(
            n
            for n in evidence_files(6, completed_numerical=True)
            if n not in tuple(c + ".npz" for c in window_protocol.CASES)
        ) + ("phase_first.npz", "phase_calls.jsonl.gz", "phase.xplane.pb")
    if completed_numerical and (
        layer != 6
        or any(
            (
                diagnostic,
                materialized,
                prefix_mlp,
                observed,
                window_numerical,
                boundary_diagnostic,
                completed_window,
            )
        )
    ):
        raise ValueError("completed numerical evidence requires distinct layer6 mode")
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
            if window_numerical or boundary_diagnostic or completed_numerical
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
                completed_numerical=completed_numerical,
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
                    if window_numerical or completed_numerical
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

    if type(rank) is not int or not 0 <= rank < 8 or _git_head() != pin:
        raise ValueError("retained preflight rank/code differs")
    if history_protocol.is_tag(tag):
        from google.cloud import storage
        from scripts.greenfield.ws32_history_preflight import retained_preflight as history_preflight

        history_preflight(tag=tag, rank=rank, pin=pin, root=run_root(tag) / f"rank{rank}",
                          repo=REPO, client=storage.Client())
        return
    if is_dense_tag(tag):
        from google.cloud import storage
        from scripts.greenfield.ws32_dense_frontier_preflight import retained_preflight as dense_preflight

        dense_preflight(tag=tag, rank=rank, pin=pin, root=run_root(tag) / f"rank{rank}",
                        repo=REPO, client=storage.Client())
        return
    layer = layer_from_tag(tag)
    if window_acquisition.is_numerical_tag(tag):
        from scripts.greenfield.prefill_window_admission import registered_programs

        registered_programs()  # All eight checks finish before any TPU initialization.
    if window_acquisition.is_completed_numerical_tag(tag):
        from scripts.greenfield.prefill_completed_window_admission import (
            registered_programs,
        )

        registered_programs()
    if window_acquisition.is_phase_baseline_tag(tag):
        from scripts.greenfield.prefill_phase_originals import load_capsule

        prefill_phase_variant.for_tag(tag).admission.registered_programs()
        load_capsule()
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
    if rolled_protocol.is_tag(tag):
        from scripts.greenfield.prefill_rolled_admission import registered_programs

        registered_programs()
        rolled_protocol.materialize_reference(root / "retained_reference", rank=rank)
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
    completed_numerical: bool = False,
    phase_baseline: bool = False,
    rolled_window: bool = False,
) -> None:
    if rolled_window and (
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
                completed_window,
                completed_numerical,
                phase_baseline,
            )
        )
    ):
        raise ValueError("rolled fleet requires distinct layer6 mode")
    if phase_baseline and (
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
                completed_window,
                completed_numerical,
            )
        )
    ):
        raise ValueError("phase fleet requires distinct layer6 mode")
    if completed_numerical and (
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
                completed_window,
            )
        )
    ):
        raise ValueError("completed numerical fleet requires distinct layer6 mode")
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
        completed_numerical=completed_numerical,
        phase_baseline=phase_baseline,
        rolled_window=rolled_window,
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
        if rolled_window:
            window_evidence.validate_workers(
                records,
                pin=pin,
                pins=pins,
                ledger=ledger,
                order=order,
                rolled_window=True,
            )
            return
        if phase_baseline:
            phase_evidence.validate_workers(
                records, pin=pin, pins=pins, ledger=ledger, order=order
            )
            return
        module = (
            boundary_evidence
            if boundary_diagnostic
            else (
                window_evidence
                if window_numerical or completed_numerical
                else window_acquisition
            )
        )
        options = (
            {"completed_numerical": True}
            if completed_numerical
            else (
                {}
                if window_numerical or boundary_diagnostic
                else dict(
                    capture_boundaries=window_boundary,
                    completed_window=completed_window,
                )
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
        if record.get("protocol") == rolled_protocol.PROTOCOL:
            reference = rolled_protocol.load_reference(
                Path("/home/gianl/glm-run")
                / json.loads(rolled_protocol.SEAL.read_text())["tag"],
                rank=record["launch_rank"],
            )
            rolled_evidence.validate_files(root, record, reference)
            return
        if record.get("protocol") in tuple(
            v.protocol for v in prefill_phase_variant.variants()
        ):
            phase_evidence.validate_files(root, record)
            return
        if record.get("protocol") == completed_protocol.PROTOCOL:
            window_evidence.validate_files(root, record, completed_numerical=True)
            return
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
    if record.get("kernel") == history_protocol.KERNEL:
        from scripts.greenfield import ws32_history_campaign as history_campaign

        if any((diagnostic, materialized, prefix_mlp, observed)):
            raise ValueError("history diagnostic cannot use another layer mode")
        history_campaign.validate_record(record, pin, root=run_root(record["tag"]), repo=REPO)
        return
    if record.get("kernel") in (dense_protocol.KERNEL, norm_protocol.KERNEL, canonical.KERNEL):
        from scripts.greenfield import ws32_dense_frontier_transport as dense_transport

        if any((diagnostic, materialized, prefix_mlp, observed)):
            raise ValueError("dense diagnostic cannot use another layer mode")
        dense_transport.validate_record(record, pin, root=run_root(record["tag"]) / "fleet",
                                        repo=REPO, **dense_reference_kwargs(record["tag"]))
        return
    rp = diagnostic_protocol(diagnostic, prefix_mlp)
    mr = materialized_protocol(materialized, observed)
    if record.get("kernel") == rolled_protocol.KERNEL:
        if any((diagnostic, materialized, prefix_mlp, observed)):
            raise ValueError("rolled record cannot use another layer mode")
        rolled_evidence.validate_record(record, pin)
        return
    if record.get("kernel") in tuple(
        v.kernel for v in prefill_phase_variant.variants()
    ):
        if any((diagnostic, materialized, prefix_mlp, observed)):
            raise ValueError("phase baseline cannot use historical layer mode")
        phase_evidence.validate_record(record, pin)
        return
    if record.get("kernel") == completed_protocol.KERNEL:
        if any((diagnostic, materialized, prefix_mlp, observed)):
            raise ValueError("completed numerical cannot use historical layer mode")
        window_evidence.validate_record(record, pin, completed_numerical=True)
        return
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
    if history_protocol.is_tag(tag):
        from scripts.greenfield import ws32_history_transport as history_transport

        history_transport.publish_rank(tag=tag, rank=rank, root=run_root(tag) / f"rank{rank}",
                                       client=storage.Client())
        return
    if is_dense_tag(tag):
        from scripts.greenfield import ws32_dense_frontier_transport as dense_transport

        dense_transport.publish_rank(tag=tag, rank=rank, root=run_root(tag) / f"rank{rank}",
                                     client=storage.Client())
        return
    from scripts.greenfield.collect_ws32_worker_evidence import (
        digest_file,
        publish_exact,
    )

    if type(rank) is not int or not 0 <= rank < 8:
        raise ValueError("invalid layer publication rank")
    root = run_root(tag) / f"rank{rank}"
    bucket = storage.Client().bucket("driftbench-dsv4-uc")
    receipts = []
    phase_mode = window_acquisition.is_phase_baseline_tag(tag)
    rolled_mode = rolled_protocol.is_tag(tag)
    bounded_mode = phase_mode or rolled_mode
    rank_cap = (
        rolled_protocol.MAX_RANK_BYTES if rolled_mode else phase_protocol.MAX_RANK_BYTES
    )
    uploaded_bytes = 0
    omitted = []

    def publish_file(name: str, path: Path) -> None:
        nonlocal uploaded_bytes
        size = path.stat().st_size
        if bounded_mode and (
            path.is_symlink() or uploaded_bytes + size > rank_cap - (1 << 20)
        ):
            omitted.append(dict(name=name, bytes=size, reason="rank_budget_or_symlink"))
            return
        receipts.append(
            publish_exact(
                bucket,
                f"results/{tag}/workers/rank{rank}/{name}",
                path,
                digest_file(path),
                compressed=False,
            )
        )
        uploaded_bytes += size

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
        completed_numerical=window_acquisition.is_completed_numerical_tag(tag),
        phase_baseline=window_acquisition.is_phase_baseline_tag(tag),
        rolled_window=rolled_protocol.is_tag(tag),
    ):
        path = root / name
        if path.is_file():
            publish_file(name, path)
    if window_acquisition.is_phase_baseline_tag(tag):
        record = json.loads((root / "runner.json").read_text())
        if record.get("status") != "SUCCESS":
            partial = [("phase_failure.npz", root / "phase_failure.npz")]
            partial.extend(
                (f"phase_failed_trace{i}.xplane.pb", path)
                for i, path in enumerate(
                    sorted((root / "phase_trace").rglob("*.xplane.pb"))
                )
            )
            for name, path in partial:
                if (
                    path.is_file()
                    and not path.is_symlink()
                    and path.stat().st_size <= phase_protocol.MAX_TRACE_BYTES
                ):
                    publish_file(name, path)
                elif path.exists():
                    omitted.append(dict(name=name, reason="per_file_limit_or_symlink"))
        elif (root / "phase_failure.npz").exists():
            raise ValueError("successful phase worker retained failure originals")
    if omitted:
        notice = root / "phase_publication_omissions.json"
        _atomic_json(notice, dict(omitted=omitted, originals_retained_locally=True))
        if notice.stat().st_size > (512 << 10):
            raise ValueError(
                "phase omission metadata oversized; originals remain local"
            )
        receipts.append(
            publish_exact(
                bucket,
                f"results/{tag}/workers/rank{rank}/{notice.name}",
                notice,
                digest_file(notice),
                compressed=False,
            )
        )
    ledger = root / "worker_receipts.json"
    ledger.write_text(json.dumps(receipts, sort_keys=True) + "\n")
    if bounded_mode and ledger.stat().st_size > (512 << 10):
        raise ValueError("phase publication ledger oversized; originals remain local")
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
    if history_protocol.is_tag(tag):
        from scripts.greenfield import ws32_history_campaign as history_campaign

        return history_campaign.collect(tag=tag, pin=pin, root=root, repo=REPO,
                                        client=storage.Client(), fetch=ssh)
    if is_dense_tag(tag):
        from scripts.greenfield import ws32_dense_frontier_transport as dense_transport

        return dense_transport.collect(tag=tag, pin=pin, root=root / "fleet", repo=REPO,
                                       **dense_reference_kwargs(tag),
                                       client=storage.Client())
    layer = layer_from_tag(tag)
    window = window_acquisition.is_acquisition_tag(tag)
    window_numerical = window_acquisition.is_numerical_tag(tag)
    window_boundary = window_acquisition.is_boundary_tag(tag)
    completed_window = window_acquisition.is_completed_tag(tag)
    completed_numerical = window_acquisition.is_completed_numerical_tag(tag)
    phase_baseline = window_acquisition.is_phase_baseline_tag(tag)
    rolled_window = rolled_protocol.is_tag(tag)
    boundary_diagnostic = window_acquisition.is_boundary_diagnostic_tag(tag)
    prefix_mlp = prefix_mlp_protocol.is_prefix_mlp_tag(tag)
    diagnostic = router_protocol.is_router_tag(tag) or prefix_mlp
    rp = diagnostic_protocol(diagnostic, prefix_mlp)
    observed = observed_ref.is_observed_tag(tag)
    materialized = materialized_ref.is_materialized_tag(tag) or observed
    mr = materialized_protocol(materialized, observed)
    bucket = storage.Client().bucket("driftbench-dsv4-uc")
    bounded_mode = phase_baseline or rolled_window
    phase_ledgers = phase_receipt_preflight(bucket, tag, root) if bounded_mode else {}
    records = []
    for rank in range(8):
        prefix = f"results/{tag}/workers/rank{rank}/"
        blob = (
            phase_ledgers[rank][0]
            if bounded_mode
            else bucket.get_blob(prefix + "worker_receipts.json")
        )
        if blob is None:
            raise ValueError(f"missing rank{rank} receipt ledger")
        ledger_bytes = (
            phase_ledgers[rank][1]
            if bounded_mode
            else blob.download_as_bytes(if_generation_match=blob.generation)
        )
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
            completed_numerical=completed_numerical,
            phase_baseline=phase_baseline,
            rolled_window=rolled_window,
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
        completed_numerical=completed_numerical,
        phase_baseline=phase_baseline,
        rolled_window=rolled_window,
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
    if completed_numerical:
        from scripts.greenfield.prefill_completed_window_admission import PROFILE

        result.update(
            kernel=completed_protocol.KERNEL,
            protocol=completed_protocol.PROTOCOL,
            profile=PROFILE,
            compile_only=False,
            numerical_execution_authorized=True,
            reference_scope=completed_protocol.REFERENCE_SCOPE,
            independent_full_layer_admission=False,
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
    if phase_baseline:
        from scripts.analysis.parse_xplane import load_xspace

        variant = prefill_phase_variant.for_tag(tag)
        if any(prefill_phase_variant.for_record(r) != variant for r in records):
            raise ValueError("phase tag and workers use different variants")
        for r in records:
            trace = root / "fleet" / f"rank{r['launch_rank']}" / "phase.xplane.pb"
            if list(load_xspace(str(trace)).hostnames) != [r["hostname"]]:
                raise ValueError("phase XSpace hostname differs from executing worker")
        traces = phase_protocol.aggregate_phase_trace(
            root / "fleet",
            {
                n: (root / "fleet/rank0" / f"{n}.optimized_hlo.txt").read_text()
                for n in phase_protocol.COUNTS
            },
        )
        result.update(
            kernel=variant.kernel,
            protocol=variant.protocol,
            profile=variant.admission.PROFILE,
            compile_only=False,
            numerical_execution_authorized=True,
            reference_scope=phase_protocol.SCOPE,
            independent_full_layer_admission=False,
            admission_only=False,
            diagnostic_only=True,
            hlo=records[0]["hlo"],
            comparison=dict(passed=None, diagnostic_evidence_complete=True),
            phase_trace=traces,
            phase_wall=phase_evidence.fleet_wall(records),
        )
    if rolled_window:
        from scripts.greenfield.prefill_rolled_admission import PROFILE

        result.update(
            kernel=rolled_protocol.KERNEL,
            protocol=rolled_protocol.PROTOCOL,
            profile=PROFILE,
            compile_only=False,
            numerical_execution_authorized=True,
            reference_scope=rolled_protocol.REFERENCE_SCOPE,
            independent_canonical_dsa_claim=False,
            hlo=records[0]["hlo"],
        )
    return result


def phase_receipt_preflight(bucket: Any, tag: str, root: Path) -> dict:
    """Bound all eight inventories and controller space BEFORE payload download."""
    import shutil

    rolled = rolled_protocol.is_tag(tag)
    rank_cap = (
        rolled_protocol.MAX_RANK_BYTES if rolled else phase_protocol.MAX_RANK_BYTES
    )
    fleet_cap = (
        rolled_protocol.MAX_FLEET_BYTES if rolled else phase_protocol.MAX_FLEET_BYTES
    )
    files = evidence_files(6, phase_baseline=not rolled, rolled_window=rolled)
    result, total = {}, 0
    for rank in range(8):
        prefix = f"results/{tag}/workers/rank{rank}/"
        blob = bucket.get_blob(prefix + "worker_receipts.json")
        if blob is None or not 0 < int(blob.size) <= (1 << 20):
            raise ValueError("phase receipt ledger missing or oversized")
        raw = blob.download_as_bytes(if_generation_match=blob.generation)
        receipts = json.loads(raw)
        expected = {prefix + n for n in files}
        if len(receipts) != len(expected) or {r["name"] for r in receipts} != expected:
            raise ValueError("phase receipt inventory incomplete or unexpected")
        rank_bytes = 0
        for row in receipts:
            size = row["size"]
            cap = (
                phase_protocol.MAX_TRACE_BYTES
                if row["name"].endswith("phase.xplane.pb")
                else rank_cap
            )
            if type(size) is not int or not 0 < size <= cap:
                raise ValueError("phase receipt payload size refused")
            rank_bytes += size
        if rank_bytes > rank_cap:
            raise ValueError("phase rank receipt storage budget refused")
        total += rank_bytes
        result[rank] = blob, raw
    if total > fleet_cap or total + (256 << 20) > shutil.disk_usage(root).free:
        raise ValueError("phase fleet/controller storage budget refused")
    return result


def campaign(tag: str, pin: str) -> None:
    root = run_root(tag)
    if REPO != Path("/home/gianl/glm-tpu-topology-rewrite") or not re.fullmatch(
        r"[0-9a-f]{40}", pin
    ):
        raise ValueError("invalid complete-layer worktree/pin")
    history = history_protocol.is_tag(tag)

    def send(command: str, *, output: Path, **kwargs: Any) -> None:
        if history:
            from scripts.greenfield.ws32_history_storage import run_logged

            run_logged(lambda: ssh(command, output=output, **kwargs), output)
        else:
            ssh(command, output=output, **kwargs)
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
    send(command, output=root / "fleet_sync.log")
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
    send(command, output=root / "retained_preflight.log")
    markers = [
        line.split()[1]
        for line in (root / "retained_preflight.log").read_text().splitlines()
        if line.startswith("PREFILL_RETAINED_OK ")
    ]
    if len(markers) != 8 or len(set(markers)) != 8:
        raise ValueError("retained preflight not eight unique hosts")
    send("hostname -I | awk '{print $1}'", output=root / "coordinator.log", worker="0")
    address = (
        str(
            ipaddress.ip_address(
                (root / "coordinator.log").read_text().strip().splitlines()[-1]
            )
        )
        + ":8476"
    )
    upload_timeout = "timeout --kill-after=10s 180s " if history else "timeout --kill-after=10s 120s " if is_dense_tag(tag) else ""
    worker_seconds = 900 if history else 600
    command = (
        "set -euo pipefail; idx=${HOSTNAME##*-w-}; tag="
        + shlex.quote(tag)
        + "; wt="
        + shlex.quote(str(REPO))
        + "; "
        'out=/home/gianl/glm-run/$tag/rank$idx; mkdir -p "$out"; cd "$wt"; '
        'upload(){ JAX_PLATFORMS=cpu PYTHONPATH="$wt" ' + upload_timeout + '/home/gianl/vllm-env/bin/python '
        '-m scripts.greenfield.ws32_prefill_layer_campaign publish-rank --tag "$tag" --rank "$idx"; }; trap upload EXIT; '
        'GLM_GREENFIELD_RUN_TAG="$tag" JAX_PLATFORMS=tpu PYTHONPATH="$wt" '
        f"timeout --kill-after=30s {worker_seconds}s /home/gianl/vllm-env/bin/python -u scripts/greenfield/probe_ws32_prefill_layer.py "
        "--expected-code-hash "
        + pin
        + " --coordinator-address "
        + shlex.quote(address)
        + ' --process-id "$idx" --output-dir "$out"'
        + (' 2>&1 | JAX_PLATFORMS=cpu PYTHONPATH="$wt" /home/gianl/vllm-env/bin/python '
           '-m scripts.greenfield.ws32_history_worker_storage "$out"' if history
           else ' >"$out/worker.log" 2>&1')
    )
    send(command, output=root / "fleet_launch.log", timeout=1140 if history else 780)
    record = collect(tag, pin)
    if history:
        from scripts.greenfield.ws32_history_storage import write_bytes, write_json

        write_json(root / "runner.json", record)
    else:
        _atomic_json(root / "runner.json", record)
    (root / "hlo").mkdir(exist_ok=True)
    primary = "candidate_b128" if history else canonical.GRAPH if canonical.is_tag(tag) else "dense01_norm" if norm_protocol.is_tag(tag) else "dense01" if dense_protocol.is_tag(tag) else "candidate"
    target = root / "hlo/candidate.optimized_hlo.txt"
    hlo = (root / f"fleet/rank0/{primary}.optimized_hlo.txt").read_bytes()
    if history:
        write_bytes(target, hlo)
    else:
        target.write_bytes(hlo)


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
