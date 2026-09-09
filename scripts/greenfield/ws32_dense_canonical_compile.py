"""One changed dense0/1 graph, abstract inputs only, no WK or model dispatch.

Adapter for the existing protected compiler worker and original-file collector.
Acquisition is not numerical/HLO admission: actual suffix loops, helpers and
physical collectives are inspected from these saved originals before execution.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from scripts.greenfield import ws32_dense_canonical as candidate
from scripts.greenfield import ws32_rolled_prefill_compile as original
from scripts.greenfield.ws32_dense_frontier_admission import MEMORY_CAPS

KERNEL = "ws32_dense_canonical_compile"
PROTOCOL = "ws32-dense01-canonical-metadata-compile-only-v1"
PROFILE = "dense01_canonical_abstract_compile_v1"
PROGRAMS = (candidate.GRAPH,)
NOTE = (
    "Changed dense0/1 canonical-placement graph compiled from authenticated metadata "
    "and abstract inputs only. No checkpoint payload loading, WK/model execution, "
    "numerical correctness, numerical peak HBM, throughput or TTFT claim. "
    "Actual optimized suffix-loop/collective/helper admission remains separate."
)


def read_metadata(repo: Path) -> Any:
    return original.read_metadata(repo, canonical_dense=True)


def prepare(mesh: Any, metadata: Any, *, repo: Path) -> original.AbstractPrefillPair:
    import jax

    prepared = candidate.prepare(mesh, repo=repo)
    if prepared.manifest_sha256 != metadata.manifest["manifest_sha256"]:
        raise ValueError("canonical compiler metadata changed during preparation")
    if any(
        not isinstance(x, jax.ShapeDtypeStruct)
        for x in jax.tree.leaves(prepared.inputs)
    ):
        raise ValueError("canonical compiler requires only abstract inputs")
    # Do not construct even the unchanged WK compiler jobs for this question.
    return original.AbstractPrefillPair(
        {candidate.GRAPH: SimpleNamespace(execute=prepared.program)},
        {candidate.GRAPH: prepared.inputs},
        prepared.manifest_sha256,
        metadata.manifest["source"]["inventory_sha256"],
    )


def validate_preserved_pair(
    root: Path, record: Mapping[str, Any], *, repo: Path
) -> dict[str, Any]:
    """Reuse graph-set receipt shape; this set has exactly ONE graph, not a pair."""
    candidate.require_source(repo)
    if set(record.get("programs", {})) != set(PROGRAMS):
        raise ValueError("canonical acquisition requires only its changed graph")
    name = candidate.GRAPH
    saved = record["programs"][name]
    stable = (root / f"{name}.stablehlo.mlir").read_bytes()
    optimized = (root / f"{name}.optimized_hlo.txt").read_bytes()
    size, digest = candidate.RAW[name]
    if (
        len(stable) != size
        or sha256(stable).hexdigest() != digest
        or saved["stablehlo_sha256"] != digest
        or not optimized
        or sha256(optimized).hexdigest() != saved["optimized_hlo_sha256"]
    ):
        raise ValueError("canonical compiler original graph identity differs")
    memory = saved["compiled_memory"]
    if set(memory) != set(MEMORY_CAPS) or any(
        type(v) is not int or not 0 <= v <= MEMORY_CAPS[k] for k, v in memory.items()
    ):
        raise ValueError("canonical compiler allocation inventory/caps differ")
    return dict(
        graphs={
            name: dict(
                stablehlo_sha256=digest,
                optimized_hlo_sha256=sha256(optimized).hexdigest(),
                memory=dict(memory),
            )
        },
        dispatch_evidence="REQUIRES_REVIEWED_COMPILE_ONLY_WORKER_AND_JOURNAL",
        numerical_claim=False,
        performance_claim=False,
        scope="ACTUAL_COMPILER_EVIDENCE_NOT_NUMERICAL_HBM_OR_ADMISSION",
    )
