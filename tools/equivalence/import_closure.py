"""G6: dynamic import closures of the serving stages and the static layering scan.

Each stage runs in a fresh interpreter that performs the stage's work -- importing its entry
modules (including the lazy imports the stage performs at run time), or actually exercising the
code -- and records every repository module that got loaded (every module whose file lies in
the source tree, whatever its top-level package, except this harness), the third-party
top-level packages, and whether JAX was imported:

* ``controller``, ``worker_preflight``, ``worker_main``: the entry modules of each process; the
  controller stage also drives the real launcher ``main`` (``controller.launcher_record``: staging,
  preflight, dispatch, supervision, collection, the failure path's cleanup), so a lazy import on the
  launch path is recorded and the controller must stay JAX-free while it runs;
* ``graph``: drives the real ``OrdinaryRuntime.__init__``/``_load`` for every fixture-tier run
  (``driver.py``) and traces every program -- catches lazy imports inside ``_load`` and inside
  graph construction (e.g. the expert-panel kernels);
* ``serving``: the G9 exercise (``wire.record``): the real ``run_queued`` over the real
  ``generate``, ``run_concurrent`` over the real ``generate_batch``/``BatchedSession``, worker
  ``main``, ``resident_loop``, ``resident_controller``, ``summarize`` and the UI/API handler.

The comparison (``gates.check``) maps current names through ``closure_map.toml`` and lets a
closure only shrink: a module or third-party package that is new in a stage fails unless it is a
reviewed ``[added]`` entry; JAX may not appear in a stage that did not import it.

The static scan lists every import of ``scripts|tools|bench|benchmarks|tests|examples`` from
``glm_tpu``; at 181c013e it is non-empty (recorded); it may only shrink, and from S2 on it must be
empty.

``--light`` runs only the three entry-module stages and the static scan (``G6-static``: no 32-device
child, no runtime build; allowed while a TPU run is live); the full record adds the exercised
``graph`` and ``serving`` stages (``G6``, heavy).

Stage entry lists name the 181c013e modules; the integrator updates them together with a
reviewed, rename-only re-record at S2f, S3 and S4.
"""

from __future__ import annotations

import argparse
import ast
import importlib
from pathlib import Path
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
EXERCISED = ("graph", "serving")
LAYERING_FORBIDDEN = ("scripts", "tools", "bench", "benchmarks", "tests", "examples")
HARNESS = "tools.equivalence"


def _repo_modules() -> list[str]:
    """Every loaded module whose file lies in the source tree (any top-level name), minus the harness
    (its package and its entry script, which runs as ``__main__``)."""
    root = str(REPO) + "/"
    harness = str(Path(__file__).resolve().parent) + "/"
    names = []
    for name, module in list(sys.modules.items()):
        path = str(getattr(module, "__file__", None) or "")
        if path.startswith(root) and not path.startswith(harness) and not name.startswith(HARNESS + "."):
            names.append(name)
    return sorted(names)


def _third_party() -> list[str]:
    std = set(sys.stdlib_module_names)
    root = str(REPO) + "/"
    tops = set()
    for name, module in list(sys.modules.items()):
        top = name.split(".", 1)[0]
        path = getattr(sys.modules.get(top), "__file__", None) or ""
        if top in std or top.startswith("_") or top.endswith("__mypyc") or str(path).startswith(root):
            continue
        tops.add(top)
    return sorted(t for t in tops if t not in ("glm_tpu", "scripts", "tools", "bench", "benchmarks", "tests"))


def run_stage(stage: str) -> dict[str, Any]:
    if stage == "graph":
        from . import fixture, lowering, programs

        mesh = fixture.cpu_mesh()
        with lowering.tpu_v4_info():
            for spec in programs.program_specs("fixture", mesh).values():
                spec.fn.trace(*spec.args)
    elif stage == "serving":
        from . import wire

        wire.record()
    else:
        for name in STAGES[stage]:
            importlib.import_module(name)
        if stage == "controller":
            from .controller import launcher_record

            launcher_record()
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


def record(*, light: bool = False) -> dict[str, Any]:
    """Every stage in a fresh child (``light``: the entry-module stages only, on one CPU device)."""
    from concurrent.futures import ThreadPoolExecutor

    names = tuple(STAGES) if light else (*STAGES, *EXERCISED)
    devices = {"graph": 32}
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        futures = {stage: pool.submit(run_child, "tools.equivalence.import_closure", "--stage", stage, timeout=1200,
                                      devices=devices.get(stage, 1)) for stage in names}
        stages = {stage: future.result() for stage, future in futures.items()}
    return dict(stages=stages, static_layering=static_layering(), entries={k: list(v) for k, v in STAGES.items()},
                exercised=[] if light else list(EXERCISED))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage", choices=(*STAGES, *EXERCISED))
    parser.add_argument("--light", action="store_true", help="entry-module stages and the static scan only")
    args = parser.parse_args(argv)
    emit(run_stage(args.stage) if args.stage else record(light=args.light))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
