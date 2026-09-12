"""E0 compact-capture lifetime candidate; no numerical/runtime admission.

DB614 supplies the actual late-capture diagnosis. Reuse its metadata, flat row
commit and consumed-state wrapper; add one layer-local ordering boundary only.
The protected outer route is not yet wired. RAW is the actual production
metadata-only lowering; it is not an acquired TPU executable or runtime fit.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess
from typing import Any, Mapping

from glm_tpu.greenfield.validation import ws32_prefill_admission as admission
from scripts.greenfield import ws32_flat_rows_compile as flat
from scripts.greenfield import ws32_owned_state_compile as owned
from scripts.greenfield import ws32_rolled_prefill_compile as original
from scripts.greenfield.ws32_prefill_owned_state import consume_state

PROGRAM = "prefill_256k_capture_barrier"
KERNEL = "ws32_capture_barrier_prefill_compile"
PROTOCOL = "ws32-capture-barrier-e0-single-graph-compile-only-v1"
PROFILE = "ws32_capture_barrier_e0_abstract_compile_v1"
PROGRAMS = (PROGRAM,)
NOTE = (
    "One78-layer E0 B128 graph with compact capture and all continuation fields "
    "through one layer-local optimization barrier. State-only donation, unchanged "
    "flat commit/rollback/repair. Metadata only; zero weights/WK/model calls. "
    "Actual optimized lifetime/allocation is required; no runtime-fit or speed claim."
)
MODEL_SOURCE_OVERRIDES = {
    **flat.MODEL_SOURCE_OVERRIDES,
    "glm_tpu/greenfield/runtime/ws32_batched_prefill.py": "fac791cc83a2814d35b69f55db5d50cf671addf7a6c756690878d5e11703c996",
}
PREREQUISITES = {
    **flat.PREREQUISITES,
    "docs/artifacts/prefill-flat-rows-db614-sealed-20260912.json": "0aded3ed5e84efad70f3a078e2f302209b250802ebd0a175b1f787383ac063a9",
    "docs/artifacts/prefill-flat-rows-db614-lifetime-diagnosis-20260912.json": "df826a12291b3bf854d2a007f266c34c5882721b03d70ab17568979de0cc244f",
}
RAW = {PROGRAM: (21130653, "4dd736e7e668a0194515860759ab038dd209c8744a48bcd3b4faf5b0d5f892d5")}


def require_source(repo: Path) -> None:
    if RAW != {PROGRAM: (21130653, "4dd736e7e668a0194515860759ab038dd209c8744a48bcd3b4faf5b0d5f892d5")} or PROGRAMS != (PROGRAM,):
        raise ValueError("capture barrier RAW registration differs")
    for name, expected in {**MODEL_SOURCE_OVERRIDES, **PREREQUISITES}.items():
        path = repo / name
        if path.is_symlink() or not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("capture barrier source/prerequisite differs")
    result = subprocess.run([
        "git", "-C", str(repo), "diff", "--quiet", admission.FROZEN_SOURCE_PIN,
        "--", *admission.MODEL_SOURCE,
        *(":(exclude)" + name for name in MODEL_SOURCE_OVERRIDES),
    ], check=False)
    if result.returncode != 0:
        raise ValueError("capture barrier changed unrelated frozen model source")


def read_metadata(repo: Path) -> Any:
    return original.read_metadata(repo, full_canonical=True, pending_cache_rows=True,
                                  flat_pending_rows=True, capture_barrier=True)


def prepare(mesh: Any, metadata: Any, *, repo: Path) -> original.AbstractPrefillPair:
    pair = original.prepare(mesh, metadata, repo=repo, full_canonical=True,
                            long_context_label="256k_e0", pending_cache_rows=True,
                            flat_pending_rows=True, capture_barrier=True)
    return original.AbstractPrefillPair(
        {PROGRAM: consume_state(pair.programs["prefill_chunk"])},
        {PROGRAM: pair.inputs["prefill_chunk"]},
        pair.manifest_sha256, pair.source_inventory_sha256,
    )


def memory_caps(name: str) -> dict[str, int]:
    if name != PROGRAM:
        raise ValueError("unregistered capture barrier compiler graph")
    return owned.memory_caps(owned.PROGRAMS[0])


def validate_preserved_pair(root: Path, record: Mapping[str, Any], *, repo: Path) -> dict[str, Any]:
    require_source(repo)
    return owned.validate_single_original(
        root, record, kernel=KERNEL, protocol=PROTOCOL, profile=PROFILE,
        programs=PROGRAMS, raw_registration=RAW, caps=memory_caps(PROGRAM),
    )
