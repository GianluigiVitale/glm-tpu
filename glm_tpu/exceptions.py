"""Fail-closed configuration errors of the engine, and ``ApiError``, the refusal of an OpenAI-compatible request
(its HTTP status, error type and body): the engine's resident client raises it, and
:mod:`glm_tpu.entrypoints.openai.protocol` re-exports it for the serving layer."""


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


class ApiError(ValueError):
    def __init__(self, message, *, status=400, kind="invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.kind = kind

    def body(self):
        return dict(error=dict(message=str(self), type=self.kind, param=None, code=None))
