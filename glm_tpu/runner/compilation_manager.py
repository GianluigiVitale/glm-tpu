"""Compile one device program and preserve its graph originals and compiler memory.

``compile_program`` lowers a jitted program, writes the StableHLO and the optimized-HLO originals
into the runtime's HLO directory, compiles it and records both digests, the compiler memory
analysis and the compile time in the runtime record (``runner.json`` next to the originals).
Moved verbatim in S2a out of the research compile and FP8 microbenchmark scripts
(``archive/research-20260922``): ``compile_program`` and ``_write_compiler_original``, and
``_compiled_memory``, ``_memory_stats``, ``_atomic_json``. The research writers' namespace
branches (native-benchmark, delivery and history-frontier byte caps) were dropped: they never
match the worker's HLO directory, so the production path writes exactly as before.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Any


def _write_compiler_original(root: Path, record: dict, name: str, form: str, text: str) -> None:
    """Write one compiler original, ``<name>.<form>``, into the HLO directory."""
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


def _compiled_memory(compiled: Any) -> dict[str, int]:
    """Compiler allocation estimate, explicitly not measured peak device HBM."""
    stats = compiled.memory_analysis()
    if stats is None:
        raise RuntimeError("compiled memory analysis is unavailable")
    names = (
        "argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes",
        "temp_size_in_bytes", "generated_code_size_in_bytes",
    )
    return {name: int(getattr(stats, name)) for name in names}


def _memory_stats(device: Any) -> dict[str, int] | None:
    value = device.memory_stats()
    if value is None:
        return None
    return {
        str(name): int(number)
        for name, number in value.items()
        if isinstance(number, int) and not isinstance(number, bool)
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
