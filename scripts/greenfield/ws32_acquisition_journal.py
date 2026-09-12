"""Append-only host compile diagnostics; never an acquisition authorization."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping

from glm_tpu.greenfield.validation.ws32_prefill import PREFILL_MODE


class Ws32AcquisitionJournal:
    """Fsync each stage before the next fallible phase, retaining partial work."""

    artifact_kind = "greenfield_ws32_acquisition_journal"
    status = "HLO_ACQUISITION_PARTIAL"

    def _check_identity(self, identity: Mapping[str, Any]) -> None:
        if (
            identity.get("prefill_mode") != PREFILL_MODE
            or identity.get("compile_only") is not True
        ):
            raise ValueError("partial journal is batched acquisition only")

    def __init__(self, path: Path, identity: Mapping[str, Any]) -> None:
        self._check_identity(identity)
        self.path = path
        self._graph: str | None = None
        self._stage = "initialized"
        self._stream = path.open("x", encoding="utf-8")
        self._write("identity", identity=dict(identity))
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _write(self, stage: str, **fields: Any) -> None:
        record = {
            "schema_version": 1,
            "artifact_kind": self.artifact_kind,
            "status": self.status,
            "performance_claim": False,
            "numerical_claim": False,
            "monotonic_seconds": time.monotonic(),
            "graph": self._graph,
            "stage": stage,
            **fields,
        }
        text = json.dumps(record, allow_nan=False, sort_keys=True) + "\n"
        if self.artifact_kind == "greenfield_ws32_history_numerical_journal_v1":
            from scripts.greenfield.ws32_history_worker_storage import write_journal

            write_journal(self._stream, text)
        else:
            self._stream.write(text)
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stage = stage
        print(
            f"GREENFIELD_WS32_ACQUISITION graph={self._graph} stage={stage}", flush=True
        )

    def begin(self, graph: str) -> None:
        self._graph = graph
        self._write("lower_compile_started")

    def compiled(
        self,
        graph: str,
        *,
        seconds: float,
        memory: Mapping[str, Any],
        device_memory: list[Any],
    ) -> None:
        if graph != self._graph:
            raise ValueError("compile journal graph identity drifted")
        self._write(
            "compiled",
            seconds=seconds,
            compiled_memory=dict(memory),
            device_memory=device_memory,
        )

    def inspect(
        self,
        graph: str,
        stable: str,
        optimized: str,
        validate: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        """Called only after both raw graph files have been durably written."""
        if graph != self._graph or self._stage != "compiled":
            raise ValueError("graph inspection lacks its compile memory record")
        self._write(
            "raw_written",
            stablehlo_sha256=sha256(stable.encode()).hexdigest(),
            optimized_hlo_sha256=sha256(optimized.encode()).hexdigest(),
        )
        try:
            report = validate()
        except Exception as error:
            self._write(
                "inspection_failed",
                exception_type=type(error).__name__,
                exception=str(error),
            )
            raise
        self._write("inspected", report=report)
        return report

    def close(self) -> None:
        self._stream.close()


class Ws32NumericalJournal(Ws32AcquisitionJournal):
    """Same fsynced compiler evidence plus early phases, never numerical SUCCESS."""

    artifact_kind = "greenfield_ws32_numerical_journal"
    status = "NUMERICAL_EXECUTION_PARTIAL"

    def _check_identity(self, identity: Mapping[str, Any]) -> None:
        from glm_tpu.greenfield.validation.ws32_prefill_admission import (
            profile_is_paired,
        )

        profile_is_paired(identity.get("batched_prefill_profile"))

        if (
            identity.get("prefill_mode") != PREFILL_MODE
            or identity.get("compile_only") is not False
        ):
            raise ValueError("numerical journal requires fixed short batched profile")

    def phase(self, name: str, **fields: Any) -> None:
        self._write(name, **fields)


class Ws32DeliveryJournal(Ws32NumericalJournal):
    """Same append/fsync protocol, with explicit long workload/owned identity.

    Journal creation is not launch admission. The old short journal continues
    to reject long profiles, and no partial entry can claim numerical SUCCESS.
    """

    artifact_kind = "greenfield_ws32_delivery_numerical_journal_v1"

    def _check_identity(self, identity: Mapping[str, Any]) -> None:
        from scripts.greenfield.ws32_delivery_runtime import numerical_identity

        expected = numerical_identity(identity.get("delivery_context_label"))
        if (identity.get("compile_only") is not False
                or any(identity.get(key) != value for key, value in expected.items())):
            raise ValueError("delivery journal requires fixed long workload/owned plan")
