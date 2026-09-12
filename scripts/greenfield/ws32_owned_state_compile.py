"""One changed256K ownership graph, using the existing protected compiler route.

Original compiler evidence only: allocation ceilings are diagnostic storage/
sanity bounds, never simultaneous runtime HBM admission or numerical permission.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from scripts.greenfield import ws32_canonical_prefill_compile as canonical
from scripts.greenfield import ws32_prefill_owned_state as owned

KERNEL = "ws32_owned_state_prefill_compile"
PROTOCOL = "ws32-owned-state-e0-single-graph-compile-only-v1"
PROFILE = "ws32_owned_state_e0_abstract_compile_v1"
PROGRAMS = (owned.PROGRAM,)
RAW = dict(owned.RAW)
SOURCE = {
    "scripts/greenfield/ws32_prefill_owned_state.py": "a02054a9e4d287ed0d3e75a72ad4b6e8783fcf499f7965a03c0bc7416b7dab2d",
    "docs/artifacts/prefill-owned-state-cpu-20260912.json": "7f2125218be0255fec37d271f1c77ab0d5578a1a7b2536998ba071406fdb244c",
}
NOTE = (
    "One frozen78-layer E0 B128 graph with explicit consumed-state argument2. "
    "Authenticated metadata and abstract inputs only; no weights/WK/model "
    "dispatch or128K reacquisition. Original alias/allocation reports are NOT "
    "simultaneous runtime HBM, numerical, quality, speed or TTFT admission."
)


def require_source(repo: Path) -> None:
    canonical.require_source(repo)
    for name, digest in SOURCE.items():
        path = repo / name
        if path.is_symlink() or not path.is_file() or sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("owned-state source/prerequisite differs")
    if RAW != {"prefill_256k_state_donated": (
        20859926, "55c3d5775eb5630a61fd8e5cd54caa30e447cf92c3ed09dcafaae0217a63cba1"
    )} or PROGRAMS != tuple(RAW):
        raise ValueError("owned-state RAW registration differs")


def read_metadata(repo: Path) -> Any:
    require_source(repo)
    return canonical.read_metadata(repo)


def prepare(mesh: Any, metadata: Any, *, repo: Path) -> Any:
    require_source(repo)
    return owned.prepare_worst_capacity(mesh, metadata, repo=repo)


def memory_caps(name: str) -> dict[str, int]:
    if name not in PROGRAMS:
        raise ValueError("unregistered owned-state compiler graph")
    return dict(argument_size_in_bytes=32 << 30, output_size_in_bytes=8 << 30,
                temp_size_in_bytes=8 << 30, generated_code_size_in_bytes=1 << 30,
                alias_size_in_bytes=8 << 30)


def validate_preserved_pair(root: Path, record: Mapping[str, Any], *, repo: Path) -> dict[str, Any]:
    """Preserve actual graph and allocation first, then validate this identity.

    A zero-alias compiler result is recorded, not mistaken for sufficient reuse.
    Even a nonzero result says nothing about all-live runtime fit plus reserve.
    """
    require_source(repo)
    fixed = dict(kernel=KERNEL, protocol=PROTOCOL, profile=PROFILE,
        compile_only=True, weights_loaded=False, model_executable_calls=0,
        numerical_claim=False, performance_claim=False)
    if any(type(record.get(k)) is not type(v) or record.get(k) != v for k, v in fixed.items()):
        raise ValueError("owned-state compiler-only identity differs")
    if set(record.get("programs", {})) != set(PROGRAMS):
        raise ValueError("owned-state original graph inventory differs")
    name = PROGRAMS[0]
    saved = record["programs"][name]
    paths = (root / f"{name}.stablehlo.mlir", root / f"{name}.optimized_hlo.txt")
    if any(p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError("owned-state original file missing or symlinked")
    raw, optimized = (p.read_bytes() for p in paths)
    size, digest = RAW[name]
    if (len(raw) != size or sha256(raw).hexdigest() != digest
            or saved.get("stablehlo_sha256") != digest or not optimized
            or sha256(optimized).hexdigest() != saved.get("optimized_hlo_sha256")):
        raise ValueError("owned-state original identity differs")
    memory, caps = saved.get("compiled_memory"), memory_caps(name)
    if (not isinstance(memory, Mapping) or set(memory) != set(caps)
            or any(type(v) is not int or not 0 <= v <= caps[k] for k, v in memory.items())
            or memory["alias_size_in_bytes"] > min(memory["argument_size_in_bytes"], memory["output_size_in_bytes"])):
        raise ValueError("owned-state compiler allocation inventory/caps differ")
    return dict(graphs={name: dict(stablehlo_sha256=digest,
        optimized_hlo_sha256=sha256(optimized).hexdigest(), context_capacity=262656,
        physical_rows=128, memory=dict(memory), ownership_contract=owned.CONTRACT)},
        numerical_claim=False, performance_claim=False,
        dispatch_evidence="REQUIRES_REVIEWED_COMPILE_ONLY_WORKER_AND_JOURNAL",
        scope="ACTUAL_COMPILER_EVIDENCE_NOT_NUMERICAL_HBM_OR_ADMISSION")
