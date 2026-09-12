"""One E0 flattened-row commit graph; no numerical or runtime-fit admission.

Reuse the pending transaction and protected compiler route with a distinct
source/RAW identity. Historical failed pending-v1 remains immutable evidence.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess
from typing import Any, Mapping

from glm_tpu.greenfield.validation import ws32_prefill_admission as admission
from scripts.greenfield import ws32_pending_rows_compile as pending
from scripts.greenfield import ws32_owned_state_compile as owned
from scripts.greenfield import ws32_rolled_prefill_compile as original
from scripts.greenfield.ws32_prefill_owned_state import consume_state

PROGRAM = "prefill_256k_flat_rows"
KERNEL = "ws32_flat_rows_prefill_compile"
PROTOCOL = "ws32-flat-rows-e0-single-graph-compile-only-v1"
PROFILE = "ws32_flat_rows_e0_abstract_compile_v1"
PROGRAMS = (PROGRAM,)
NOTE = (
    "One78-layer E0 B128 graph with flattened layer/physical-row commit and "
    "state-only donation. Metadata/abstract inputs only; zero weights/WK/model calls. "
    "Actual compiler allocation is not runtime HBM, quality, speed or TTFT. "
    "No128K baseline reacquisition or precision change."
)
MODEL_SOURCE_OVERRIDES = {
    **pending.MODEL_SOURCE_OVERRIDES,
    "glm_tpu/greenfield/runtime/ws32_batched_prefill.py": "f85042d39e44c529df3ee2e792d503e1223709876419cefd8b55b194ebf7c2d0",
    "glm_tpu/greenfield/kernels/prefill_flat_rows.py": "4b5f2f3ec6fcd667dae72d924e2bf7406b83873ff43dd9ecee7824e2d02b61f2",
}
PREREQUISITES = {
    **pending.PREREQUISITES,
    "docs/artifacts/prefill-pending-rows-compile-refusal-20260912.json": "e6ac3beebc6f61d5766653d97b26491ad7d171337848ba47ba9720fbeac00f9a",
}
# Actual production metadata-only TPU-target lowering; not a TPU compilation.
RAW = {PROGRAM: (21100974, "891f2b8ae51e58ce924b2a7cd7c91c3dba610d5092c05d92ffd15a7e2133eae9")}


def require_source(repo: Path) -> None:
    if RAW != {PROGRAM: (21100974, "891f2b8ae51e58ce924b2a7cd7c91c3dba610d5092c05d92ffd15a7e2133eae9")} or PROGRAMS != (PROGRAM,):
        raise ValueError("flat-row RAW registration differs")
    for name, expected in {**MODEL_SOURCE_OVERRIDES, **PREREQUISITES}.items():
        path = repo / name
        if path.is_symlink() or not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("flat rows source/prerequisite differs")
    result = subprocess.run([
        "git", "-C", str(repo), "diff", "--quiet", admission.FROZEN_SOURCE_PIN,
        "--", *admission.MODEL_SOURCE,
        *(":(exclude)" + name for name in MODEL_SOURCE_OVERRIDES),
    ], check=False)
    if result.returncode != 0:
        raise ValueError("flat rows changed unrelated frozen model source")


def read_metadata(repo: Path) -> Any:
    return original.read_metadata(repo, full_canonical=True,
                                  pending_cache_rows=True, flat_pending_rows=True)


def prepare(mesh: Any, metadata: Any, *, repo: Path) -> original.AbstractPrefillPair:
    pair = original.prepare(mesh, metadata, repo=repo, full_canonical=True,
                            long_context_label="256k_e0", pending_cache_rows=True,
                            flat_pending_rows=True)
    return original.AbstractPrefillPair(
        {PROGRAM: consume_state(pair.programs["prefill_chunk"])},
        {PROGRAM: pair.inputs["prefill_chunk"]},
        pair.manifest_sha256, pair.source_inventory_sha256,
    )


def memory_caps(name: str) -> dict[str, int]:
    if name != PROGRAM:
        raise ValueError("unregistered flat-row compiler graph")
    return owned.memory_caps(owned.PROGRAMS[0])


def validate_preserved_pair(root: Path, record: Mapping[str, Any], *, repo: Path) -> dict[str, Any]:
    require_source(repo)
    return owned.validate_single_original(
        root, record, kernel=KERNEL, protocol=PROTOCOL, profile=PROFILE,
        programs=PROGRAMS, raw_registration=RAW, caps=memory_caps(PROGRAM),
    )
