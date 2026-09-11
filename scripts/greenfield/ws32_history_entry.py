"""Default-off history entry inside the existing protected selected-layer probe.

The parent must finish all eight retained preflights before calling this entry;
it owns both leases, source pin, deadline, publication and authenticated cleanup.
There is no CLI, collector, numerical promotion or performance claim here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import socket
from typing import Any, Mapping

import numpy as np

from scripts.greenfield import ws32_history_admission as admission
from scripts.greenfield import ws32_history_call_evidence as evidence
from scripts.greenfield import ws32_history_execution as execution
from scripts.greenfield import ws32_history_preflight as preflight_module
from scripts.greenfield import ws32_history_protocol as protocol
from scripts.greenfield import ws32_history_runtime as runtime_module
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from scripts.greenfield.prefill_window_acquisition import fleet_step
from scripts.greenfield.prefill_window_evidence import same_json

STATUS = "DIAGNOSTIC_COMPLETED_NOT_NUMERICAL_PROMOTION"


def _retained_identity(root: Path, record: Mapping[str, Any], repo: Path) -> None:
    """Refuse stale metadata/original identities before distributed initialization."""
    admission.registration(repo)
    path = root / "retained_preflight.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("history entry retained preflight must be a regular original")
    retained = json.loads(path.read_bytes())
    if not isinstance(retained, dict):
        raise ValueError("history entry retained preflight must be an object")
    pins = protocol.original_pins(repo, record["launch_rank"])
    expected = dict(
        **{key: record[key] for key in ("tag", "code_hash", "launch_rank", "hostname", "protocol")},
        selected_layer_ids=list(protocol.LAYERS), include_embedding=True,
        context_capacity=protocol.CAPACITY, host_main_rope_table=True,
        selected_leaf_count=protocol.SELECTED_LEAVES, payload_bytes_per_chip=protocol.PAYLOAD_BYTES,
        overlay_tensor_count=protocol.OVERLAY_TENSORS, overlay_bytes_per_chip=protocol.OVERLAY_BYTES,
        prompt_ids_sha256=protocol.PROMPT_SHA,
        prompt_hash_source="AUTHENTICATED_ORIGINAL_TOKEN_ORACLE",
        original_tags=protocol.ORIGINAL_TAGS, original_pins=pins,
        receipt_sha256={branch: protocol.RECEIPTS[branch][1] for branch in protocol.BRANCHES},
        bytes=sum(v["size"] for branch in pins.values() for v in branch.values()),
        scope="HEADERS_ORIGINALS_AND_HOST_INPUTS_ONLY_NOT_SELECTED_PAYLOAD_OR_LIVE_TOPOLOGY",
        numerical_execution_available=False, numerical_admission=False,
        numerical_promotion=False, performance_claim=False,
    )
    # The retained schema has no numerical profile: it deliberately remains
    # metadata-only. Actual original bytes/owners are rebound by prepare_bound.
    same_json({key: retained.get(key) for key in expected}, expected,
              "history entry retained identity")


def _complete(record: Mapping[str, Any]) -> None:
    """Recognize fixed completed worker claims, leaving original replay to collection."""
    report, entries = record.get("history", {}), record.get("call_evidence")
    reproduction = report.get("reproduction", {})
    finalization = record.get("acquisition_phases", {}).get("history/finalize", {})
    if (record.get("kernel") != protocol.KERNEL
            or record.get("protocol") != protocol.PROTOCOL
            or record.get("profile") != admission.PROFILE
            or record.get("diagnostic_only") is not True
            or any(record.get(key) is not False for key in
                   ("admission_only", "compile_only", "numerical_promotion", "performance_claim"))
            or record.get("history_execution_complete") is not True
            or record.get("current_phase") != "history/execution_complete"
            or record.get("planned_call_count") != evidence.MAX_CALLS
            or type(record.get("iterations")) is not int or record["iterations"] != 0
            or record.get("latency") is not None
            or record.get("status") != "RUNNING"
            or any(key in record for key in ("error", "finalization_error", "call_original_error"))
            or set(record.get("programs", {})) != set(protocol.PROGRAMS)
            or any(v.get("admission", {}).get("passed") is not True
                   for v in record["programs"].values())
            or finalization.get("status") != "COMPLETE" or finalization.get("error") is not None
            or not isinstance(record.get("compile_journal_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", record["compile_journal_sha256"]) is None
            or report.get("protocol") != protocol.PROTOCOL
            or report.get("complete") is not True or report.get("attribution_eligible") is not True
            or report.get("prompt_length") != protocol.PROMPT_LENGTH
            or report.get("steps") != len(protocol.plan())
            or any(report.get(key) is not False for key in
                   ("numerical_promotion", "performance_claim", "cause_claim"))
            or set(reproduction) != set(protocol.BRANCHES)
            or any(v.get("reproduced") is not True for v in reproduction.values())
            or record.get("call_evidence_layout") != evidence.SCHEMA
            or not isinstance(entries, list) or len(entries) != evidence.MAX_CALLS):
        raise ValueError("history terminal reproduction/call inventory incomplete")
    total = 0
    for index, (reference, expected) in enumerate(zip(entries, execution.call_schedule(), strict=True)):
        if (not isinstance(reference, Mapping)
                or set(reference) != {"phase", "graph", "completed", "original"}
                or (reference["phase"], reference["graph"]) != expected
                or reference["completed"] is not True):
            raise ValueError("history terminal call reference differs")
        original = reference["original"]
        if (not isinstance(original, Mapping)
                or set(original) != {"schema", "path", "bytes", "sha256", "index"}
                or original["schema"] != evidence.SCHEMA
                or original["path"] != f"call_records/call{index:03d}.json"
                or type(original["index"]) is not int or original["index"] != index
                or type(original["bytes"]) is not int or not 0 < original["bytes"] <= evidence.MAX_CALL_BYTES
                or not isinstance(original["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", original["sha256"]) is None):
            raise ValueError("history terminal original descriptor differs")
        total += original["bytes"]
    if (total > evidence.MAX_TOTAL_BYTES or type(record.get("call_original_bytes")) is not int
            or record["call_original_bytes"] != total):
        raise ValueError("history terminal original byte total differs")


def execute(args: argparse.Namespace, *, tag: str, repo: Path) -> int:
    """Initialize once, bind selected owners, then run and vote the fixed diagnostic."""
    from scripts.greenfield.run_short_decoder_ws32 import _initialize_runtime

    root = Path(args.output_dir)
    preflight_module._plain_path(root)
    if (not protocol.is_tag(tag) or type(args.process_id) is not int or not 0 <= args.process_id < 8
            or not isinstance(args.expected_code_hash, str)
            or re.fullmatch(r"[0-9a-f]{40}", args.expected_code_hash) is None
            or root != preflight_module.RUN_ROOT / tag / f"rank{args.process_id}"
            or not root.is_dir()):
        raise ValueError("history entry tag/rank/code/path differs")
    output = root / "runner.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    record: dict[str, Any] = dict(
        status="RUNNING", tag=tag, code_hash=args.expected_code_hash,
        launch_rank=args.process_id, hostname=socket.gethostname(), pid=os.getpid(),
        start_ticks=int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        kernel=protocol.KERNEL, protocol=protocol.PROTOCOL, profile=admission.PROFILE,
        diagnostic_only=True, admission_only=False, compile_only=False,
        performance_claim=False, numerical_promotion=False, iterations=0, latency=None, programs={},
    )
    try:
        _atomic_json(output, record)
        _retained_identity(root, record, repo)
        runtime = _initialize_runtime(args)
        from jax.experimental import multihost_utils

        def consensus(passed: bool) -> bool:
            # Existing harness-only scalar vote; never model/stage dispatch.
            return bool(np.asarray(multihost_utils.process_allgather(np.asarray(passed, np.int32))).all())

        bound = runtime_module.prepare_bound(root=root, record=record, repo=repo,
                                             runtime=runtime, consensus=consensus)
        execution.execute(root=root, record=record, repo=repo, mesh=runtime[1],
                          bound=bound, consensus=consensus)

        def finalize() -> None:
            _complete(record)
            record["status"] = STATUS

        # Retain before the matched terminal vote, including local write failures.
        fleet_step("history/terminal", finalize, root=root, record=record, consensus=consensus)
        return 0
    except Exception as exc:
        record.update(status="DIAGNOSTIC_FAILED", error=f"{type(exc).__name__}: {exc}")
        try:
            _atomic_json(output, record)
        except Exception as publication_error:
            print(f"history final record publication failed: {publication_error}", flush=True)
        raise
