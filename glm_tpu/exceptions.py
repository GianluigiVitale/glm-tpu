"""Fail-closed configuration errors of the engine."""


class ConfigValidationError(ValueError):
    """Base class for an invalid immutable greenfield configuration."""


class GeometryValidationError(ConfigValidationError):
    """The model geometry is incomplete or internally inconsistent."""


class TopologyValidationError(ConfigValidationError):
    """The physical device inventory is incomplete or inconsistent."""


class PlanValidationError(ConfigValidationError):
    """An execution plan violates an explicit layout invariant."""


class CheckpointValidationError(ConfigValidationError):
    """Checkpoint metadata, ownership, or byte reconciliation is invalid."""
