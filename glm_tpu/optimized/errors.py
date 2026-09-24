"""Fail-closed configuration errors for the greenfield engine."""


class GreenfieldConfigError(ValueError):
    """Base class for an invalid immutable greenfield configuration."""


class GeometryValidationError(GreenfieldConfigError):
    """The model geometry is incomplete or internally inconsistent."""


class TopologyValidationError(GreenfieldConfigError):
    """The physical device inventory is incomplete or inconsistent."""


class PlanValidationError(GreenfieldConfigError):
    """An execution plan violates an explicit layout invariant."""


class CheckpointValidationError(GreenfieldConfigError):
    """Checkpoint metadata, ownership, or byte reconciliation is invalid."""
