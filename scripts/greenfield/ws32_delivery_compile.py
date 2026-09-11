"""Three distinct frozen long-prefill compiler jobs; never model execution.

All four L7 depths share capacity/geometry. E0 main and tail share the same
B128 program. Acquire each distinct graph once, using the existing metadata,
compiler, journal and fleet path. Allocation ceilings below bound diagnostic
records; they do NOT establish simultaneous runtime residency or HBM fit.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
from typing import Any, Mapping

from scripts.greenfield import ws32_canonical_prefill_compile as canonical
from scripts.greenfield import ws32_rolled_prefill_compile as original

KERNEL = "ws32_delivery_long_prefill_compile"
PROTOCOL = "ws32-delivery-long-prefill-three-graph-compile-only-v1"
PROFILE = "ws32_delivery_long_prefill_abstract_compile_v1"
PROGRAMS = ("prefill_128k_chunk", "prefill_128k_tail", "prefill_256k")
# Full byte strings acquired by CPU TPU-target lowering in the preparation
# receipt; actual optimized TPU graphs remain a separate acquisition.
RAW = {
    "prefill_128k_chunk": (20691979, "ed6ab5731ef17819e210f688f634a8539d09583c11ed3f89fd04e6c60df06cf1"),
    "prefill_128k_tail": (20788062, "59843169d5ca21fe350e1e5b7384a0c53512afa7ac7e159b059ac05c69f91834"),
    "prefill_256k": (20713573, "19f1e88e871c9c47b33392fca15680c4b1c058e9d9a43c695130e799d767d497"),
}
GEOMETRY = {
    "prefill_128k_chunk": (131072, 128),
    "prefill_128k_tail": (131072, 114),
    "prefill_256k": (262656, 128),
}
NOTE = (
    "Three distinct frozen78-layer long-prefill graphs: 128K B128/B114 and "
    "256K B128 shared by main/tail. Authenticated metadata and abstract inputs "
    "only; no checkpoint payload, WK/model dispatch, numerical/quality claim, "
    "runtime peak HBM, throughput or TTFT. Actual whole-model HLO and "
    "simultaneous numerical memory admission remain separate."
)


def require_source(repo: Path) -> None:
    canonical.require_source(repo)
    if set(RAW) != set(PROGRAMS) or set(GEOMETRY) != set(PROGRAMS):
        raise ValueError("delivery compiler requires three registered graphs")
    for size, digest in RAW.values():
        if (type(size) is not int or size <= 0 or type(digest) is not str
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise ValueError("delivery compiler raw size/digest invalid")


def read_metadata(repo: Path) -> Any:
    """Local source/metadata check, also called before any TPU runtime starts."""
    require_source(repo)
    return canonical.read_metadata(repo)


def prepare(mesh: Any, metadata: Any, *, repo: Path) -> original.AbstractPrefillPair:
    """Reuse original builders; no dispatch, data placement or payload read."""
    require_source(repo)
    l7 = original.prepare(mesh, metadata, repo=repo, full_canonical=True,
                          long_context_label="128k_d1_0")
    e0 = original.prepare(mesh, metadata, repo=repo, full_canonical=True,
                          long_context_label="256k_e0")
    if (l7.manifest_sha256 != e0.manifest_sha256
            or l7.source_inventory_sha256 != e0.source_inventory_sha256):
        raise ValueError("delivery compiler metadata changed between capacities")
    jobs = {"prefill_128k_chunk": (l7, "prefill_chunk"),
            "prefill_128k_tail": (l7, "prefill_tail"),
            "prefill_256k": (e0, "prefill_chunk")}
    return original.AbstractPrefillPair(
        {name: pair.programs[role] for name, (pair, role) in jobs.items()},
        {name: pair.inputs[role] for name, (pair, role) in jobs.items()},
        l7.manifest_sha256, l7.source_inventory_sha256)


def memory_caps(name: str) -> dict[str, int]:
    """Diagnostic per-field ceilings, NOT an executable runtime memory budget."""
    if name not in PROGRAMS:
        raise ValueError("delivery compiler unregistered memory program")
    return dict(argument_size_in_bytes=32 << 30, output_size_in_bytes=8 << 30,
                temp_size_in_bytes=4 << 30, generated_code_size_in_bytes=1 << 30,
                alias_size_in_bytes=0)


def validate_preserved_pair(root: Path, record: Mapping[str, Any], *, repo: Path) -> dict[str, Any]:
    """Bind all originals before evaluating any diagnostic allocation ceiling."""
    require_source(repo)
    fixed = dict(kernel=KERNEL, protocol=PROTOCOL, profile=PROFILE, compile_only=True,
                 weights_loaded=False, model_executable_calls=0,
                 numerical_claim=False, performance_claim=False)
    if any(type(record.get(k)) is not type(v) or record.get(k) != v for k, v in fixed.items()):
        raise ValueError("delivery compiler-only identity differs")
    if set(record.get("programs", {})) != set(PROGRAMS):
        raise ValueError("delivery compiler original graph inventory differs")
    graphs = {}
    for name in PROGRAMS:
        saved = record["programs"][name]
        paths = (root / f"{name}.stablehlo.mlir", root / f"{name}.optimized_hlo.txt")
        if any(p.is_symlink() or not p.is_file() for p in paths):
            raise ValueError("delivery compiler original file missing or symlinked")
        raw, optimized = (p.read_bytes() for p in paths)
        size, digest = RAW[name]
        if (len(raw) != size or sha256(raw).hexdigest() != digest
                or saved.get("stablehlo_sha256") != digest or not optimized
                or sha256(optimized).hexdigest() != saved.get("optimized_hlo_sha256")):
            raise ValueError("delivery compiler original identity differs")
        capacity, rows = GEOMETRY[name]
        graphs[name] = dict(stablehlo_sha256=digest,
            optimized_hlo_sha256=sha256(optimized).hexdigest(),
            context_capacity=capacity, physical_rows=rows, memory=saved.get("compiled_memory"))
    for name, graph in graphs.items():
        caps, memory = memory_caps(name), graph["memory"]
        if (not isinstance(memory, Mapping) or set(memory) != set(caps)
                or any(type(v) is not int or not 0 <= v <= caps[k] for k, v in memory.items())):
            raise ValueError(f"delivery compiler allocation inventory/caps differ: {name}")
        graph["memory"] = dict(memory)
    return dict(graphs=graphs, numerical_claim=False, performance_claim=False,
        dispatch_evidence="REQUIRES_REVIEWED_COMPILE_ONLY_WORKER_AND_JOURNAL",
        scope="ACTUAL_COMPILER_EVIDENCE_NOT_NUMERICAL_HBM_OR_ADMISSION")
