"""On-demand partitioning exports.

The WS32 path needs only the payload-free source inventory. The PP8/PP16
layer-assignment, ownership, memory-model and layout-manifest contracts were
retired with their consumers (recoverable at b667f00f); retained names keep
their original module targets."""

from importlib import import_module as _import_module

_EXPORTS = {
    "SourceFile": "source_inventory",
    "SourceInventory": "source_inventory",
    "SourceTensor": "source_inventory",
    "inspect_source_inventory": "source_inventory",
    "read_source_inventory": "source_inventory",
    "write_source_inventory": "source_inventory",
}

__all__ = [
    "SourceFile",
    "SourceInventory",
    "SourceTensor",
    "inspect_source_inventory",
    "read_source_inventory",
    "write_source_inventory",
]


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, _, attribute = target.partition(".")
    value = getattr(_import_module("." + module, __name__), attribute or name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
