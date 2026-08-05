"""Fail-closed configuration errors for the greenfield engine."""


class GreenfieldConfigError(ValueError):
    """Base class for an invalid immutable greenfield configuration."""


class GeometryValidationError(GreenfieldConfigError):
    """The model geometry is incomplete or internally inconsistent."""


class TopologyValidationError(GreenfieldConfigError):
    """The physical device inventory is incomplete or inconsistent."""


class PlanValidationError(GreenfieldConfigError):
    """An execution plan violates an explicit layout invariant."""


class HloContractViolationError(RuntimeError):
    """Lowered HLO violates an explicit physical execution contract."""


class BenchmarkValidationError(GreenfieldConfigError):
    """A synthetic benchmark does not satisfy its declared proof contract."""


class CheckpointValidationError(GreenfieldConfigError):
    """Checkpoint metadata, ownership, or byte reconciliation is invalid."""


class PartitioningValidationError(GreenfieldConfigError):
    """A layer or memory partition cannot satisfy the declared plan."""
