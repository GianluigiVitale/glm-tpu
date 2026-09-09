"""Full-model dense correction: metadata-only preparation, NOT numerical admission.

Reuse the existing main/tail compiler and full checkpoint schema. This is a
distinct source recipe: DB608's reduced two-file guard remains unchanged.
Both changed actual graphs must be preserved and independently inspected before
any complete-model run; historical numerical profiles do not authorize them.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess
from typing import Any, Mapping

from glm_tpu.greenfield.validation import ws32_prefill_admission as admission
from scripts.greenfield import ws32_rolled_prefill_compile as original


KERNEL = "ws32_prefill_canonical_model_compile"
PROTOCOL = "ws32-canonical-dense-b128-b114-metadata-compile-only-v1"
PROFILE = "ws32_canonical_dense_fullmodel_abstract_compile_v1"
PROGRAMS = ("prefill_chunk", "prefill_tail")
MODEL_SOURCE_OVERRIDES = {
    "glm_tpu/greenfield/runtime/ws32_batched_prefill.py": "e491724290133c49d29775356da84a36221cc949d0d46f0592262b2933418b66",
    "glm_tpu/greenfield/kernels/ws32_prefill_window.py": "87f6fcad74aa2bbaa0e1ef7309c768021eb6d9686a5ff767a1776fc81a6c8c5d",
    "glm_tpu/greenfield/kernels/ws32_prefill_dense_canonical.py": "513635cc9c792f94792841041e6b3b89df3dbb0725ae5506c34ba2d1df198436",
}
PREREQUISITES = {
    admission.FROZEN_RECEIPT: admission.FROZEN_RECEIPT_SHA256,
    "docs/artifacts/prefill-dense-canonical-db608-sealed-20260909.json": "3062d54e4eb1a8fd3bea86cecc4f5b2c1cabb0c52d425ff2070c69c21985bf2c",
}
# Actual production78-layer CPU TPU-target lowering. Optimized TPU originals
# are still missing: these pins cannot authorize model execution.
RAW: dict[str, tuple[int, str]] = {
    "prefill_chunk": (
        20586966,
        "d88ddff75beaed5301dda776f7fcd439b31023c19eb097b5a4351bb90cef8ec0",
    ),
    "prefill_tail": (
        20683085,
        "863ffa1248f2f23603a11e5a95b6a13bfc0edbe7f09d6b3da91c1f1095bbe697",
    ),
}
NOTE = (
    "Full78-layer canonical dense main/tail graphs compiled from authenticated "
    "metadata and abstract inputs only. No checkpoint payload loading or model/WK "
    "execution, numerical correctness, measured full-model HBM, throughput or TTFT "
    "claim. Actual whole-model HLO and numerical admission remain separate."
)


def require_source(repo: Path) -> None:
    """Three exact model files, every other model path frozen at DB603.

    This read-only check supplements, never replaces, the protected parent's
    clean published code pin, environment and checkpoint-identity checks.
    """
    for name, expected in {**MODEL_SOURCE_OVERRIDES, **PREREQUISITES}.items():
        path = repo / name
        if (
            path.is_symlink()
            or not path.is_file()
            or sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("full canonical source/prerequisite differs")
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--quiet",
            admission.FROZEN_SOURCE_PIN,
            "--",
            *admission.MODEL_SOURCE,
            *(":(exclude)" + name for name in MODEL_SOURCE_OVERRIDES),
        ],
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("full canonical changed unrelated frozen model source")


def program_options() -> dict[str, Any]:
    return {
        **admission.short_program_options(admission.ROLLED_SHORT_PROFILE),
        "canonical_dense": True,
    }


def read_metadata(repo: Path) -> Any:
    return original.read_metadata(repo, full_canonical=True)


def prepare(mesh: Any, metadata: Any, *, repo: Path) -> original.AbstractPrefillPair:
    return original.prepare(mesh, metadata, repo=repo, full_canonical=True)


def validate_preserved_pair(
    root: Path, record: Mapping[str, Any], *, repo: Path
) -> dict[str, Any]:
    """Bind both captured actual graphs/allocations, without approving dispatch."""
    require_source(repo)
    if set(RAW) != set(PROGRAMS) or set(record.get("programs", {})) != set(PROGRAMS):
        raise ValueError("full canonical requires both registered/preserved graphs")
    graphs = {}
    for name in PROGRAMS:
        saved = record["programs"][name]
        stable = (root / f"{name}.stablehlo.mlir").read_bytes()
        optimized = (root / f"{name}.optimized_hlo.txt").read_bytes()
        size, digest = RAW[name]
        if (
            len(stable) != size
            or sha256(stable).hexdigest() != digest
            or saved["stablehlo_sha256"] != digest
            or not optimized
            or sha256(optimized).hexdigest() != saved["optimized_hlo_sha256"]
        ):
            raise ValueError("full canonical compiler original identity differs")
        # Same short geometry and conservative compiler ceilings, not an
        # inheritance of the old HLO or numerical memory measurements.
        admission.validate_short_compiled_memory(
            name,
            saved["compiled_memory"],
            profile=admission.ROLLED_SHORT_PROFILE,
            repo=repo,
        )
        graphs[name] = dict(
            stablehlo_sha256=digest,
            optimized_hlo_sha256=sha256(optimized).hexdigest(),
            memory=dict(saved["compiled_memory"]),
        )
    return dict(
        graphs=graphs,
        dispatch_evidence="REQUIRES_REVIEWED_COMPILE_ONLY_WORKER_AND_JOURNAL",
        numerical_claim=False,
        performance_claim=False,
        scope="ACTUAL_COMPILER_EVIDENCE_NOT_NUMERICAL_HBM_OR_ADMISSION",
    )
