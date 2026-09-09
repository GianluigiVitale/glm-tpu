"""Weight-free two-graph continuation for the existing protected fleet runner.

This module does not launch workers or authorize numerical execution. It reuses
the actual production compiler writer, matched fleet phases and fsynced journal.
The protected outer campaign must still bind runtime/owners and publish originals.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
from typing import Any, Callable

from glm_tpu.greenfield.validation.ws32_prefill import PREFILL_MODE
from glm_tpu.greenfield.validation.ws32_prefill_admission import ROLLED_SHORT_PROFILE
from scripts.greenfield import ws32_rolled_prefill_compile as preparation
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from scripts.greenfield.prefill_window_acquisition import fleet_step
from scripts.greenfield.probe_ws32_prefill_layer import compile_program
from scripts.greenfield.ws32_acquisition_journal import Ws32AcquisitionJournal

KERNEL = "ws32_prefill_rolled_model_compile"
PROTOCOL = "ws32-rolled-b128-b114-metadata-compile-only-v1"
PROGRAMS = ("prefill_chunk", "prefill_tail")
CAPTURE = dict(
    scope="ORIGINAL_COMPILER_FILES_ONLY_NOT_HLO_OR_NUMERICAL_ADMISSION",
    numerical_execution_authorized=False,
)


def journal_identity(record: dict[str, Any]) -> dict[str, Any]:
    fixed = dict(
        kernel=KERNEL,
        protocol=PROTOCOL,
        profile=ROLLED_SHORT_PROFILE,
        prefill_mode=PREFILL_MODE,
        compile_only=True,
        weights_loaded=False,
        model_executable_calls=0,
        numerical_claim=False,
        performance_claim=False,
    )
    if any(
        type(record.get(key)) is not type(value) or record.get(key) != value
        for key, value in fixed.items()
    ):
        raise ValueError("rolled compiler-only identity differs")
    if (
        not re.fullmatch(r"[0-9a-f]{40}", record.get("code_hash", ""))
        or type(record.get("launch_rank")) is not int
        or not 0 <= record["launch_rank"] < 8
    ):
        raise ValueError("rolled compiler source/rank identity differs")
    return dict(fixed, code_hash=record["code_hash"], launch_rank=record["launch_rank"])


class RolledCompileJournal(Ws32AcquisitionJournal):
    artifact_kind = "greenfield_ws32_rolled_model_compile_journal_v1"

    def _check_identity(self, identity: dict[str, Any]) -> None:
        journal_identity(identity)

    def pair_verified(self, report: dict[str, Any]) -> None:
        self._write("preserved_pair_verified", report=report)


def execute_pair(
    root: Path,
    record: dict[str, Any],
    *,
    mesh: Any,
    repo: Path,
    consensus: Callable[[bool], bool],
) -> None:
    """Compile and preserve both programs, without ever calling an executable.

    Every local fallible phase has a fleet vote. Actual compile/write failures
    stop peers; identity/memory admission is deliberately deferred until BOTH
    originals exist. Failures leave the original journal/graphs in place for
    the outer campaign's failure publisher, not a numerical SUCCESS marker.
    """
    journal: RolledCompileJournal | None = None

    def step(name: str, action: Callable[[], Any]) -> Any:
        return fleet_step(name, action, record=record, root=root, consensus=consensus)

    def setup() -> Any:
        nonlocal journal
        identity = journal_identity(record)
        if record.get("programs") != {}:
            raise ValueError("rolled compiler continuation cannot resume/recompile")
        journal = RolledCompileJournal(root / "compile_journal.jsonl", identity)
        metadata = preparation.read_metadata(repo)
        prepared = preparation.prepare(mesh, metadata, repo=repo)
        if set(prepared.programs) != set(PROGRAMS) or set(prepared.inputs) != set(
            PROGRAMS
        ):
            raise ValueError("rolled compiler requires only main and tail")
        record["abstract_metadata"] = dict(
            manifest_sha256=prepared.manifest_sha256,
            source_inventory_sha256=prepared.source_inventory_sha256,
            payload_reads=False,
            input_kind="SHAPE_DTYPE_STRUCT_ONLY",
        )
        return prepared

    failure: Exception | None = None
    try:
        prepared = step("rolled_compile_setup", setup)
        for name in PROGRAMS:

            def compile_one() -> None:
                # The result is not invoked or transferred. No WK program exists
                # in this inventory; all inputs remain ShapeDtypeStruct leaves.
                compiled = compile_program(
                    prepared.programs[name].execute,
                    prepared.inputs[name],
                    name,
                    root,
                    record,
                    journal=journal,
                )
                del compiled
                journal.inspect(
                    name,
                    (root / f"{name}.stablehlo.mlir").read_text(),
                    (root / f"{name}.optimized_hlo.txt").read_text(),
                    lambda: dict(CAPTURE),
                )

            step("rolled_compile_" + name, compile_one)

        def validate() -> None:
            report = preparation.validate_preserved_pair(root, record, repo=repo)
            record["preserved_pair"] = report
            journal.pair_verified(report)

        step("rolled_compile_pair_validation", validate)
    except Exception as exc:
        failure = exc
        record.update(
            status="FAILED",
            compiler_acquisition_complete=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:

        def finalize() -> None:
            if journal is not None:
                journal.close()
                record["compile_journal_sha256"] = sha256(
                    (root / "compile_journal.jsonl").read_bytes()
                ).hexdigest()

        try:
            step("rolled_compile_finalize", finalize)
        except Exception as exc:
            # Keep the original compile/refusal cause if cleanup also fails.
            record["finalization_error"] = f"{type(exc).__name__}: {exc}"
            failure = failure or exc
    if failure is not None:
        record.update(status="FAILED", compiler_acquisition_complete=False)
        _atomic_json(root / "runner.json", record)
        raise failure

    def terminal() -> None:
        record.update(status="SUCCESS", compiler_acquisition_complete=True)
        _atomic_json(root / "runner.json", record)

    try:
        step("rolled_compile_terminal", terminal)
    except Exception as exc:
        record.update(
            status="FAILED",
            compiler_acquisition_complete=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        _atomic_json(root / "runner.json", record)
        raise
