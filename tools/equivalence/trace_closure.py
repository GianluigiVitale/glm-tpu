"""G7: the set of repository functions the production composition actually executes.

One CPU32 process runs, under ``sys.monitoring`` (PY_START, disabled per code object after its
first event, so the overhead is one callback per function): the real ``OrdinaryRuntime``
``__init__``/``_load`` of every fixture-tier run in concrete mode (``driver.py``; it executes the
WK, FP8-table and cache-initializer programs, and ``compile_batch`` for the concurrent runs),
tracing of every fixture-tier program, and the CPU golden composition (the real ``generate``
over prompts A and B: prefill blocks, packed decode, the request-session host loop). Recorded as
sorted ``module:qualname`` strings for code under ``glm_tpu/`` and ``scripts/``.

S1-S3 require equality with the S0 set (catches dynamic-dispatch drift the import closure
misses); from S4 the gate is informational because names change.
"""

from __future__ import annotations

import sys
from typing import Any

from .common import REPO, digest_json, emit, require_cpu

TOOL_ID = 4


def _module_name(filename: str) -> str | None:
    root = str(REPO) + "/"
    if not filename.startswith(root):
        return None
    relative = filename[len(root):]
    if not (relative.startswith("glm_tpu/") or relative.startswith("scripts/")) or not relative.endswith(".py"):
        return None
    name = relative[:-3].replace("/", ".")
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


def run() -> dict[str, Any]:
    from . import fixture, golden_run, lowering, programs

    require_cpu()
    mesh = fixture.cpu_mesh()
    executed: set[str] = set()
    monitoring = sys.monitoring

    def on_start(code: Any, offset: int) -> Any:
        module = _module_name(code.co_filename)
        if module is not None and code.co_name != "<module>":
            executed.add(f"{module}:{code.co_qualname}")
        return monitoring.DISABLE

    monitoring.use_tool_id(TOOL_ID, "glm-equivalence-trace")
    monitoring.register_callback(TOOL_ID, monitoring.events.PY_START, on_start)
    monitoring.set_events(TOOL_ID, monitoring.events.PY_START)
    try:
        with lowering.tpu_v4_info():
            for spec in programs.program_specs("fixture", mesh, concrete=True).values():
                spec.fn.trace(*spec.args)
            golden_run.composition(mesh)
    finally:
        monitoring.set_events(TOOL_ID, 0)
        monitoring.register_callback(TOOL_ID, monitoring.events.PY_START, None)
        monitoring.free_tool_id(TOOL_ID)
    functions = sorted(executed)
    modules = sorted({f.split(":", 1)[0] for f in functions})
    return dict(functions=functions, count=len(functions), modules=modules, digest=digest_json(functions))


def main() -> int:
    emit(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
