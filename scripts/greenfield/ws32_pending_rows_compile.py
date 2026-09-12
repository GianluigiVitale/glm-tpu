"""One pending-row E0 graph; protected compiler evidence, no numerical opt-in.

Reuse authenticated production inputs, unchanged model kernels and explicit
state-only ownership. This new source recipe must never widen old source guards.
Uses the existing protected worker/collector with a distinct one-graph identity.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess
from typing import Any, Mapping

from glm_tpu.greenfield.validation import ws32_prefill_admission as admission
from scripts.greenfield import ws32_canonical_prefill_compile as canonical
from scripts.greenfield import ws32_rolled_prefill_compile as original
from scripts.greenfield.ws32_prefill_owned_state import consume_state
from scripts.greenfield import ws32_owned_state_compile as owned


PROGRAM = "prefill_256k_pending_rows"
KERNEL = "ws32_pending_rows_prefill_compile"
PROTOCOL = "ws32-pending-rows-e0-single-graph-compile-only-v1"
PROFILE = "ws32_pending_rows_e0_abstract_compile_v1"
PROGRAMS = (PROGRAM,)
NOTE = (
    "One78-layer E0 B128 graph with pending cache rows and consumed-state argument2. "
    "Authenticated metadata and abstract inputs only; zero weights/WK/model calls. "
    "Compiler allocation/alias originals are not runtime HBM, numerical quality, "
    "throughput or TTFT admission. No128K baseline reacquisition."
)
MODEL_SOURCE_OVERRIDES = {
    **canonical.MODEL_SOURCE_OVERRIDES,
    "glm_tpu/greenfield/runtime/ws32_batched_prefill.py": "b67a86d4d5336f0926a7b176fae2675a2a007c5ef6e28668493fa219025b48b9",
    "glm_tpu/greenfield/kernels/prefill_pending_rows.py": "bc9a7f837109df6e9ae47981d0399873e186723c1f3f0eb52ea2d70a02e352ab",
}
PREREQUISITES = {
    **canonical.PREREQUISITES,
    **owned.SOURCE,
    "docs/artifacts/prefill-pending-rows-cpu-20260912.json": "45eb9840480c3a6561868bbff1aabe9bc01042c70419a39d4eeefd31a5be9371",
    "docs/artifacts/prefill-owned-state-db613-sealed-20260912.json": "5893c4eb0d8f064df09ca3d7d8b9626eafc049fe5b7ba993d9796fb2fed5c37b",
}


def require_source(repo: Path) -> None:
    """Bind the two changed model files and original canonical/DB613 basis."""
    if RAW != {PROGRAM: (21096032, "1ede24a42a146898af7ea22ffb6d0a0e87178ada33695cbbb540477a9bdea873")} or PROGRAMS != (PROGRAM,):
        raise ValueError("pending-row RAW registration differs")
    for name, expected in {**MODEL_SOURCE_OVERRIDES, **PREREQUISITES}.items():
        path = repo / name
        if (
            path.is_symlink()
            or not path.is_file()
            or sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("pending rows source/prerequisite differs")
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
        raise ValueError("pending rows changed unrelated frozen model source")


def read_metadata(repo: Path) -> Any:
    return original.read_metadata(repo, full_canonical=True, pending_cache_rows=True)


def prepare(mesh: Any, metadata: Any, *, repo: Path) -> original.AbstractPrefillPair:
    pair = original.prepare(
        mesh,
        metadata,
        repo=repo,
        full_canonical=True,
        long_context_label="256k_e0",
        pending_cache_rows=True,
    )
    return original.AbstractPrefillPair(
        {PROGRAM: consume_state(pair.programs["prefill_chunk"])},
        {PROGRAM: pair.inputs["prefill_chunk"]},
        pair.manifest_sha256,
        pair.source_inventory_sha256,
    )


# Actual production metadata-only CPU TPU-target lowering; not TPU compilation.
RAW = {
    PROGRAM: (21096032, "1ede24a42a146898af7ea22ffb6d0a0e87178ada33695cbbb540477a9bdea873")
}


def memory_caps(name: str) -> dict[str, int]:
    if name != PROGRAM:
        raise ValueError("unregistered pending-row compiler graph")
    return owned.memory_caps(owned.PROGRAMS[0])


def validate_preserved_pair(
    root: Path, record: Mapping[str, Any], *, repo: Path
) -> dict[str, Any]:
    require_source(repo)
    return owned.validate_single_original(
        root, record, kernel=KERNEL, protocol=PROTOCOL, profile=PROFILE,
        programs=PROGRAMS, raw_registration=RAW, caps=memory_caps(PROGRAM),
    )
