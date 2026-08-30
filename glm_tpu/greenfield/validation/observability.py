"""Compatibility re-exports for the stdlib-only observability core."""

from ..observability import (
    ArrayObservation,
    OBSERVABILITY_SCHEMA_VERSION,
    audit_observability_contract,
    compare_array_observations,
    inspect_npz_artifact,
    write_observability_report,
)


__all__ = (
    "ArrayObservation",
    "OBSERVABILITY_SCHEMA_VERSION",
    "audit_observability_contract",
    "compare_array_observations",
    "inspect_npz_artifact",
    "write_observability_report",
)
