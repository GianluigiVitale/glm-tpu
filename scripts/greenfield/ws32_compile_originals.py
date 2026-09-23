"""Compile with preserved graph originals, authenticated inventory, WK programs.

Verbatim (b667f00f) from the retired layer-admission probe
probe_ws32_prefill_layer.py: ``_write_compiler_original``, ``compile_program``,
``authenticated_inventory`` and ``build_wk_programs``. The native worker,
delivery WK preparation and phase weights import them from here so the
supported path no longer loads the campaign probe. Numerical programs,
StableHLO/optimized-HLO recording and the inventory digest check are unchanged.

S2a: the production worker no longer imports this module. ``authenticated_inventory``
moved verbatim to ``glm_tpu.greenfield.partitioning.source_inventory`` and
``build_wk_programs`` to ``glm_tpu.optimized.bf16_resident`` (both re-exported here);
production compiles through ``glm_tpu.runner.compilation_manager``, whose copies of
``compile_program`` and ``_write_compiler_original`` drop the research namespace branches
below. These research copies keep them for the research writers.
"""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import time
from typing import Any

from glm_tpu.greenfield.partitioning.source_inventory import (  # noqa: F401  (re-exported)
    authenticated_inventory,
)
from glm_tpu.optimized.bf16_resident import build_wk_programs  # noqa: F401  (re-exported)
from scripts.greenfield.microbench_fp8_matmul import (
    _atomic_json,
    _compiled_memory,
    _memory_stats,
)


def _write_compiler_original(root: Path, record: dict, name: str, form: str, text: str) -> None:
    """Preserve the writer; history and long preparation have pre-write caps."""
    if any(parent.name.startswith("native.rank") for parent in (root, *root.parents)):
        from scripts.greenfield.ws32_native_benchmark_transport import native_root, require_write_size
        path = root / f"{name}.{form}"
        if native_root(path) is not None:
            require_write_size(path, len(text.encode()))
    if root.name.startswith(("delivery_wk.rank", "delivery_decode.rank")):
        from scripts.greenfield.ws32_delivery_phase_transport import require_write_size

        require_write_size(root / f"{name}.{form}", len(text.encode()))
    if record.get("protocol") == "ws32-history-frontier-l06-two-branch-first-decode-v1":
        from scripts.greenfield.ws32_history_worker_storage import write_graph

        write_graph(root, name, form, text)
    else:
        (root / f"{name}.{form}").write_text(text)


def compile_program(
    fn: Any,
    inputs: tuple[Any, ...],
    name: str,
    root: Path,
    record: dict[str, Any],
    *,
    journal: Any = None,
) -> Any:
    """Preserve both actual graph forms and compiler memory for every program."""
    start = time.monotonic()
    if journal is not None:
        journal.begin(name)
    lowered = fn.lower(*inputs)
    stable = str(lowered.compiler_ir(dialect="stablehlo"))
    _write_compiler_original(root, record, name, "stablehlo.mlir", stable)
    compiled = lowered.compile()
    memory = _compiled_memory(compiled)
    seconds = time.monotonic() - start
    if journal is not None:
        import jax

        journal.compiled(
            name,
            seconds=seconds,
            memory=memory,
            device_memory=[
                dict(device_id=int(d.id), stats=_memory_stats(d))
                for d in jax.local_devices()
            ],
        )
    hlo = compiled.as_text()
    _write_compiler_original(root, record, name, "optimized_hlo.txt", hlo)
    if journal is not None:
        for form in ("stablehlo.mlir", "optimized_hlo.txt"):
            with (root / f"{name}.{form}").open("rb") as stream:
                os.fsync(stream.fileno())
    record.setdefault("programs", {})[name] = {
        "stablehlo_sha256": sha256(stable.encode()).hexdigest(),
        "optimized_hlo_sha256": sha256(hlo.encode()).hexdigest(),
        "compiled_memory": memory,
        "compile_seconds": seconds,
    }
    _atomic_json(root / "runner.json", record)
    return compiled
