"""Weight-free fixed-graph continuation for the existing protected fleet runner.

This module does not launch workers or authorize numerical execution. It reuses
the actual production compiler writer, matched fleet phases and fsynced journal.
The protected outer campaign must still bind runtime/owners and publish originals.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class CompileMode:
    kernel: str
    protocol: str
    profile: str
    programs: tuple[str, ...]
    prefix: str
    preparation: Any
    journal_type: Any


def compile_mode(
    canonical_dense: bool = False, *, full_canonical: bool = False,
    history: bool = False, delivery: bool = False, owned_state: bool = False, pending_rows: bool = False, flat_rows: bool = False, capture_barrier: bool = False,
) -> CompileMode:
    """Fixed reviewed inventories only; never a caller-supplied compiler job."""
    if any(type(flag) is not bool for flag in (canonical_dense, full_canonical, history, delivery, owned_state, pending_rows, flat_rows, capture_barrier)):
        raise ValueError("compiler mode must be a static bool")
    if sum((canonical_dense, full_canonical, history, delivery, owned_state, pending_rows, flat_rows, capture_barrier)) > 1:
        raise ValueError("compiler modes are exclusive")
    if capture_barrier:
        from scripts.greenfield import ws32_capture_barrier_compile as capture

        return CompileMode(
            capture.KERNEL, capture.PROTOCOL, capture.PROFILE,
            capture.PROGRAMS, "capture_barrier_compile", capture, CaptureBarrierCompileJournal,
        )
    if flat_rows:
        from scripts.greenfield import ws32_flat_rows_compile as flat

        return CompileMode(
            flat.KERNEL, flat.PROTOCOL, flat.PROFILE,
            flat.PROGRAMS, "flat_rows_compile", flat, FlatRowsCompileJournal,
        )
    if pending_rows:
        from scripts.greenfield import ws32_pending_rows_compile as pending

        return CompileMode(
            pending.KERNEL, pending.PROTOCOL, pending.PROFILE,
            pending.PROGRAMS, "pending_rows_compile", pending, PendingRowsCompileJournal,
        )
    if owned_state:
        from scripts.greenfield import ws32_owned_state_compile as state_compile

        return CompileMode(
            state_compile.KERNEL, state_compile.PROTOCOL, state_compile.PROFILE,
            state_compile.PROGRAMS, "owned_state_compile", state_compile, OwnedStateCompileJournal,
        )
    if delivery:
        from scripts.greenfield import ws32_delivery_compile as long_compile

        return CompileMode(
            long_compile.KERNEL, long_compile.PROTOCOL, long_compile.PROFILE,
            long_compile.PROGRAMS, "delivery_compile", long_compile, DeliveryCompileJournal,
        )
    if history:
        from scripts.greenfield import ws32_history_compile as bounded

        return CompileMode(
            bounded.KERNEL, bounded.PROTOCOL, bounded.PROFILE,
            bounded.PROGRAMS, "history_compile", bounded, HistoryCompileJournal,
        )
    if full_canonical:
        from scripts.greenfield import ws32_canonical_prefill_compile as full

        return CompileMode(
            full.KERNEL,
            full.PROTOCOL,
            full.PROFILE,
            full.PROGRAMS,
            "canonical_model_compile",
            full,
            CanonicalModelCompileJournal,
        )
    if canonical_dense:
        from scripts.greenfield import ws32_dense_canonical_compile as dense

        return CompileMode(
            dense.KERNEL,
            dense.PROTOCOL,
            dense.PROFILE,
            dense.PROGRAMS,
            "dense_canonical_compile",
            dense,
            DenseCanonicalCompileJournal,
        )
    return CompileMode(
        KERNEL,
        PROTOCOL,
        ROLLED_SHORT_PROFILE,
        PROGRAMS,
        "rolled_compile",
        preparation,
        RolledCompileJournal,
    )


def journal_identity(
    record: dict[str, Any],
    *,
    canonical_dense: bool = False,
    full_canonical: bool = False,
    history: bool = False, delivery: bool = False, owned_state: bool = False, pending_rows: bool = False, flat_rows: bool = False, capture_barrier: bool = False,
) -> dict[str, Any]:
    mode = compile_mode(canonical_dense, full_canonical=full_canonical, history=history, delivery=delivery, owned_state=owned_state, pending_rows=pending_rows, flat_rows=flat_rows, capture_barrier=capture_barrier)
    fixed = dict(
        kernel=mode.kernel,
        protocol=mode.protocol,
        profile=mode.profile,
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


class DenseCanonicalCompileJournal(RolledCompileJournal):
    artifact_kind = "greenfield_ws32_dense_canonical_compile_journal_v1"

    def _check_identity(self, identity: dict[str, Any]) -> None:
        journal_identity(identity, canonical_dense=True)


class CanonicalModelCompileJournal(RolledCompileJournal):
    artifact_kind = "greenfield_ws32_canonical_model_compile_journal_v1"

    def _check_identity(self, identity: dict[str, Any]) -> None:
        journal_identity(identity, full_canonical=True)


class HistoryCompileJournal(RolledCompileJournal):
    artifact_kind = "greenfield_ws32_history_compile_journal_v1"

    def _check_identity(self, identity: dict[str, Any]) -> None:
        journal_identity(identity, history=True)


class DeliveryCompileJournal(RolledCompileJournal):
    artifact_kind = "greenfield_ws32_delivery_compile_journal_v1"

    def _check_identity(self, identity: dict[str, Any]) -> None:
        journal_identity(identity, delivery=True)


class OwnedStateCompileJournal(RolledCompileJournal):
    artifact_kind = "greenfield_ws32_owned_state_compile_journal_v1"

    def _check_identity(self, identity: dict[str, Any]) -> None:
        journal_identity(identity, owned_state=True)


class PendingRowsCompileJournal(RolledCompileJournal):
    artifact_kind = "greenfield_ws32_pending_rows_compile_journal_v1"

    def _check_identity(self, identity: dict[str, Any]) -> None:
        journal_identity(identity, pending_rows=True)


class FlatRowsCompileJournal(RolledCompileJournal):
    artifact_kind = "greenfield_ws32_flat_rows_compile_journal_v1"

    def _check_identity(self, identity: dict[str, Any]) -> None:
        journal_identity(identity, flat_rows=True)


class CaptureBarrierCompileJournal(RolledCompileJournal):
    artifact_kind = "greenfield_ws32_capture_barrier_compile_journal_v1"

    def _check_identity(self, identity: dict[str, Any]) -> None:
        journal_identity(identity, capture_barrier=True)


def execute_pair(
    root: Path,
    record: dict[str, Any],
    *,
    mesh: Any,
    repo: Path,
    consensus: Callable[[bool], bool],
    canonical_dense: bool = False,
    full_canonical: bool = False,
    history: bool = False, delivery: bool = False, owned_state: bool = False, pending_rows: bool = False, flat_rows: bool = False, capture_barrier: bool = False,
) -> None:
    """Preserve the fixed graph set without ever calling an executable.

    Every local fallible phase has a fleet vote. Actual compile/write failures
    stop peers; identity/memory admission waits for ALL registered originals
    (main/tail, one changed dense graph, or the seven history graphs). Failures leave them in place for
    the outer campaign's failure publisher, not a numerical SUCCESS marker.
    """
    mode = compile_mode(canonical_dense, full_canonical=full_canonical, history=history, delivery=delivery, owned_state=owned_state, pending_rows=pending_rows, flat_rows=flat_rows, capture_barrier=capture_barrier)
    journal: RolledCompileJournal | None = None

    def step(name: str, action: Callable[[], Any]) -> Any:
        return fleet_step(name, action, record=record, root=root, consensus=consensus)

    def setup() -> Any:
        nonlocal journal
        identity = journal_identity(
            record, canonical_dense=canonical_dense, full_canonical=full_canonical,
            history=history, delivery=delivery, owned_state=owned_state, pending_rows=pending_rows, flat_rows=flat_rows, capture_barrier=capture_barrier,
        )
        if record.get("programs") != {}:
            raise ValueError("rolled compiler continuation cannot resume/recompile")
        journal = mode.journal_type(root / "compile_journal.jsonl", identity)
        metadata = mode.preparation.read_metadata(repo)
        prepared = mode.preparation.prepare(mesh, metadata, repo=repo)
        if set(prepared.programs) != set(mode.programs) or set(prepared.inputs) != set(
            mode.programs
        ):
            raise ValueError("compiler requires its exact registered graph inventory")
        record["abstract_metadata"] = dict(
            manifest_sha256=prepared.manifest_sha256,
            source_inventory_sha256=prepared.source_inventory_sha256,
            payload_reads=False,
            input_kind="SHAPE_DTYPE_STRUCT_ONLY",
        )
        return prepared

    failure: Exception | None = None
    try:
        prepared = step(mode.prefix + "_setup", setup)
        for name in mode.programs:

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

            step(mode.prefix + "_" + name, compile_one)

        def validate() -> None:
            report = mode.preparation.validate_preserved_pair(root, record, repo=repo)
            record["preserved_pair"] = report
            journal.pair_verified(report)

        step(mode.prefix + "_pair_validation", validate)
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
            step(mode.prefix + "_finalize", finalize)
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
        step(mode.prefix + "_terminal", terminal)
    except Exception as exc:
        record.update(
            status="FAILED",
            compiler_acquisition_complete=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        _atomic_json(root / "runner.json", record)
        raise
