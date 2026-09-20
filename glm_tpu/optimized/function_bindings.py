"""Private dependency binding without changing frozen module globals."""
from types import FunctionType


def bind_dependencies(function, **replacements):
    if function.__closure__ is not None:
        raise ValueError("composition bindings must be module-level functions")
    if not replacements.keys() <= function.__globals__.keys():
        raise ValueError("frozen dependency name drifted")
    namespace = {**function.__globals__, **replacements}
    bound = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__)
    bound.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    bound.__annotations__ = dict(function.__annotations__)
    bound.__module__ = __name__
    bound.__doc__ = function.__doc__
    return bound
