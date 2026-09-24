"""N1: location-free TPU-platform lowering on a CPU host.

Harness-only patches, active only inside ``location_free()``; production lowering is never touched:

* ``jax._src.interpreters.mlir.source_info_to_location`` returns ``ir.Location.unknown()``. Every
  JAX equation location -- and every Pallas/Mosaic kernel-body location (``mosaic/lowering.py``
  looks the function up through the module attribute at call time; ``mlir.py`` itself through its
  module globals) -- is built there, so file paths, line numbers and ``jax.named_scope`` names
  never reach the IR or the serialized Mosaic kernel bodies.
* ``jax_include_full_tracebacks_in_locations=False`` (defensive).
* The TPU v4 ``tpu_info`` registry entry for the CPU device kind, so kernels that size tiles from
  the chip description see the production chip.
* Kernel-name patch-back from ``kernel_renames.toml`` (D5, since S4.2b): the public values of
  ``glm_tpu.kernels.names.KERNEL_NAMES`` are set to their 181c013e spellings for the duration.
* ``jax.clear_caches()`` on entry and exit, so no lowering cached outside the context is reused.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import importlib
from pathlib import Path
import tomllib
from typing import Any

KERNEL_RENAMES = Path(__file__).with_name("kernel_renames.toml")
_MISSING = object()


@contextmanager
def tpu_v4_info() -> Iterator[None]:
    """Make ``get_tpu_info()`` on the CPU backend describe one TPU v4 chip (as 31 tests do)."""
    from jax._src.pallas.mosaic import tpu_info

    previous = tpu_info.registry.get("cpu", _MISSING)
    tpu_info.registry["cpu"] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
    tpu_info.get_tpu_info.cache_clear()
    try:
        yield
    finally:
        if previous is _MISSING:
            tpu_info.registry.pop("cpu", None)
        else:
            tpu_info.registry["cpu"] = previous
        tpu_info.get_tpu_info.cache_clear()


def kernel_renames() -> dict[str, Any]:
    """Parsed ``kernel_renames.toml``: ``module``, ``attribute`` and a ``names`` table new -> old."""
    if not KERNEL_RENAMES.is_file():
        return dict(names={})
    value = tomllib.loads(KERNEL_RENAMES.read_text())
    names = value.get("names", {})
    if not isinstance(names, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in names.items()):
        raise ValueError("kernel_renames.toml [names] must map strings to strings")
    return dict(value, names=names)


@contextmanager
def _kernel_name_patch_back(table: dict[str, Any] | None = None) -> Iterator[None]:
    table = kernel_renames() if table is None else table
    if not table["names"]:
        yield
        return
    module = importlib.import_module(table["module"])
    mapping = getattr(module, table["attribute"])
    saved = dict(mapping)
    reverse = table["names"]
    try:
        for key, value in list(mapping.items()):
            if value in reverse:
                mapping[key] = reverse[value]
        yield
    finally:
        mapping.clear()
        mapping.update(saved)


@contextmanager
def location_free(renames: dict[str, Any] | None = None) -> Iterator[None]:
    """N1. ``renames`` overrides ``kernel_renames.toml`` (self-test only)."""
    import jax
    from jax._src import config as jax_config
    from jax._src.interpreters import mlir
    from jaxlib.mlir import ir

    original = mlir.source_info_to_location

    def unknown_location(ctx: Any, primitive: Any, name_stack: Any, traceback: Any) -> Any:
        return ir.Location.unknown()

    jax.clear_caches()
    mlir.source_info_to_location = unknown_location
    try:
        with jax_config.include_full_tracebacks_in_locations(False), tpu_v4_info(), _kernel_name_patch_back(renames):
            yield
    finally:
        mlir.source_info_to_location = original
        jax.clear_caches()


def lower_for_tpu(jitted: Any, abstract_args: tuple[Any, ...]) -> Any:
    """Trace and lower exactly the callable production lowers (the ``jax.jit`` object, donation
    already applied the way production applies it) for the TPU platform. Call inside
    ``location_free()``."""
    return jitted.trace(*abstract_args).lower(lowering_platforms=("tpu",))
