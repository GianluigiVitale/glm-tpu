"""G6: dynamic import closures of the serving stages and the static layering scan.

Each stage runs in a fresh interpreter that imports the stage's entry modules (including the
lazy imports the stage performs at run time) and records which repository modules were loaded,
which third-party top-level packages, and whether JAX was imported. The ``graph`` stage builds
and traces every fixture-tier program, which catches lazy imports inside graph construction
(e.g. the expert-panel kernels).

The static scan lists every import of ``scripts|tools|bench|benchmarks|tests|examples`` from
``glm_tpu``; at 181c013e it is non-empty (recorded), from S2 on it must be empty.

Stage entry lists name the 181c013e modules; the integrator updates them together with a
reviewed, rename-only re-record at S2f, S3 and S4.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import sys
from typing import Any

from .common import REPO, emit, run_child

STAGES: dict[str, tuple[str, ...]] = {
    # python -m glm_tpu ask ...: CLI, question preparation, controller and its run-time lazy imports
    "controller": (
        "glm_tpu.cli",
        "glm_tpu.optimized.ask",
        "scripts.release.launch_ws32_optimized_request",
        "scripts.greenfield.ws32_native_benchmark_programs",  # source_identity -> require_source
        "scripts.greenfield.watch_ws32_run",                  # idle probe text
    ),
    # worker --preflight-only: module import plus the topology binding's lazy imports
    "worker_preflight": (
        "scripts.release.ws32_optimized_worker",
        "glm_tpu.optimized.topology_binding",
        "glm_tpu.greenfield.benchmarking.ws32_one_layer",
        "glm_tpu.greenfield.sharding.ws32",
    ),
    # worker main: runtime initialization, OrdinaryRuntime and _load's lazy imports
    "worker_main": (
        "scripts.release.ws32_optimized_worker",
        "scripts.greenfield.run_short_decoder_ws32",
        "glm_tpu.optimized.runtime",
        "scripts.greenfield.ws32_compile_originals",
        "scripts.greenfield.ws32_native_benchmark_programs",
        "glm_tpu.greenfield.checkpoint.ws32_runtime_checkpoint",
        "glm_tpu.optimized.batched_runtime",
        "scripts.greenfield.ws32_native_benchmark_transport",  # imported because HLO dirs are native.rank*
        "jax.experimental.multihost_utils",
    ),
}
LAYERING_FORBIDDEN = ("scripts", "tools", "bench", "benchmarks", "tests", "examples")


def _repo_modules() -> list[str]:
    root = str(REPO) + "/"
    names = []
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path and str(path).startswith(root) and (name.startswith("glm_tpu") or name.startswith("scripts")):
            names.append(name)
    return sorted(names)


def _third_party() -> list[str]:
    std = set(sys.stdlib_module_names)
    tops = {name.split(".", 1)[0] for name in sys.modules}
    return sorted(t for t in tops if t not in std and not t.startswith("_") and not t.endswith("__mypyc")
                  and t not in ("glm_tpu", "scripts", "tools"))


def run_stage(stage: str) -> dict[str, Any]:
    if stage == "graph":
        from . import fixture, lowering, programs

        mesh = fixture.cpu_mesh()
        with lowering.tpu_v4_info():
            for spec in programs.program_specs("fixture", mesh).values():
                spec.fn.trace(*spec.args)
    else:
        for name in STAGES[stage]:
            importlib.import_module(name)
    modules = _repo_modules()
    return dict(modules=modules, count=len(modules), jax_imported="jax" in sys.modules,
                third_party=_third_party())


def static_layering() -> list[str]:
    """``path: imported module`` for every forbidden import inside glm_tpu (lazy imports included)."""
    hits = []
    for path in sorted((REPO / "glm_tpu").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                if name.split(".", 1)[0] in LAYERING_FORBIDDEN:
                    hits.append(f"{path.relative_to(REPO)}: {name}")
    return sorted(set(hits))


def record() -> dict[str, Any]:
    stages = {}
    for stage in (*STAGES, "graph"):
        stages[stage] = run_child("tools.equivalence.import_closure", "--stage", stage, timeout=1200)
    return dict(stages=stages, static_layering=static_layering(), entries={k: list(v) for k, v in STAGES.items()})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage", choices=(*STAGES, "graph"))
    args = parser.parse_args(argv)
    emit(run_stage(args.stage) if args.stage else record())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
