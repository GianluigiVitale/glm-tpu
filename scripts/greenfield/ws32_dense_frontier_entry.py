"""Dense diagnostic entry INSIDE the existing protected selected-layer probe.

No CLI or deployment. The campaign must finish all eight retained preflights
before this entry; the outer wrapper owns both leases, deadline and cleanup.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
from typing import Any

import numpy as np

from glm_tpu.greenfield.validation import ws32_prefill_admission as model_admission
from scripts.greenfield import ws32_dense_frontier_admission as admission
from scripts.greenfield import ws32_dense_frontier_protocol as protocol
from scripts.greenfield import ws32_dense_frontier_runtime as runtime_module
from scripts.greenfield import ws32_dense_norm_protocol as norm_protocol
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from scripts.greenfield.prefill_window_acquisition import fleet_step
from scripts.greenfield.prefill_window_evidence import same_json

STATUS = "DIAGNOSTIC_COMPLETED_NOT_NUMERICAL_PROMOTION"


def execute(args: argparse.Namespace, *, tag: str, repo: Path) -> int:
    """Initialize the original runtime once; finalize through a matched fleet vote."""
    from scripts.greenfield.run_short_decoder_ws32 import _initialize_runtime

    norm_mode = norm_protocol.is_tag(tag)
    if not (norm_mode or protocol.is_tag(tag)):
        raise ValueError("invalid dense diagnostic entry tag")
    inspector = admission.inspect_program
    if norm_mode:
        from scripts.greenfield.ws32_dense_norm_admission import inspect_program

        inspector = inspect_program
    root = args.output_dir
    record: dict[str, Any] = dict(
        status="RUNNING",
        tag=tag,
        code_hash=args.expected_code_hash,
        launch_rank=args.process_id,
        hostname=socket.gethostname(),
        pid=os.getpid(),
        start_ticks=int(
            Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
        ),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        kernel=norm_protocol.KERNEL if norm_mode else protocol.KERNEL,
        protocol=norm_protocol.PROTOCOL if norm_mode else protocol.PROTOCOL,
        profile=norm_protocol.PROFILE if norm_mode else admission.PROFILE,
        diagnostic_only=True,
        admission_only=False,
        compile_only=False,
        performance_claim=False,
        numerical_promotion=False,
        iterations=0,
        latency=None,
        programs={},
    )
    output = root / "runner.json"
    try:
        _atomic_json(output, record)
        # Parent already checked code/rank/path. Refuse stale host preflight and
        # changed model source BEFORE any distributed initialization or payload.
        model_admission.require_acquired_model_source(
            repo, profile=model_admission.ROLLED_SHORT_PROFILE
        )
        preflight = json.loads((root / "retained_preflight.json").read_bytes())
        expected = {
            k: record[k]
            for k in ("tag", "code_hash", "launch_rank", "hostname", "protocol")
        }
        same_json(
            {k: preflight.get(k) for k in expected},
            expected,
            "dense entry retained identity",
        )
        runtime = _initialize_runtime(args)
        from jax.experimental import multihost_utils

        def consensus(passed: bool) -> bool:
            # Harness-only scalar vote, never stage/model dispatch.
            return bool(
                np.asarray(
                    multihost_utils.process_allgather(np.asarray(passed, np.int32))
                ).all()
            )

        runtime_module.execute_bound(
            root=root,
            record=record,
            repo=repo,
            runtime=runtime,
            consensus=consensus,
            inspect_program=inspector,
        )

        def finalize() -> None:
            if norm_mode:
                report = record["dense_norm"]
                if (
                    report.get("complete") is not True
                    or report.get("reproduction", {}).get("reproduced") is not True
                    or report.get("numerical_promotion") is not False
                    or report.get("performance_claim") is not False
                    or report.get("cause_claim") is not False
                    or set(report.get("own_suffix_reproduction", {}))
                    != {n for n, _, _ in norm_protocol.CAPTURES}
                    or record.get("current_phase") != "norm/comparison"
                    or len(record.get("call_evidence", [])) != len(norm_protocol.CALLS)
                    or not all(
                        c.get("completed") is True for c in record["call_evidence"]
                    )
                ):
                    raise ValueError(
                        "norm terminal reproduction/call inventory incomplete"
                    )
                # Full original-array and journal replay remains the collector's
                # obligation; this only recognizes completed worker execution.
                record["status"] = STATUS
                return
            report = record["dense_frontier"]
            comparison = report["comparison"]
            if (
                report.get("complete") is not True
                or comparison.get("reproduced") is not True
                or comparison.get("model_calls") != 5
                or comparison.get("wk_calls") != 4
                or len(record.get("call_evidence", [])) != 9
                or not all(
                    call.get("completed") is True for call in record["call_evidence"]
                )
                or record.get("current_phase") != "dense/comparison"
            ):
                raise ValueError("dense terminal comparison/call inventory incomplete")
            record["status"] = STATUS

        # fleet_step publishes before voting, including local JSON failures.
        fleet_step(
            "norm/terminal" if norm_mode else "dense/terminal",
            finalize,
            root=root,
            record=record,
            consensus=consensus,
        )
        return 0
    except Exception as exc:
        record.update(status="DIAGNOSTIC_FAILED", error=f"{type(exc).__name__}: {exc}")
        try:
            _atomic_json(output, record)
        except Exception as publication_error:
            print(
                f"dense final record publication failed: {publication_error}",
                flush=True,
            )
        raise
