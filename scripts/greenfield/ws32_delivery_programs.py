"""Fixed long-delivery program selection from retained compiler evidence.

Metadata/abstract preparation only: this is not HLO, memory or dispatch admission.
L7 retains its original non-donating pair; E0 uses DB615's consumed-state graph
for BOTH roles, with one program object. Historical source recipes stay strict.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

from glm_tpu.greenfield.validation.ws32_delivery_prefill import long_plan
from scripts.greenfield import ws32_capture_barrier_compile as capture
from scripts.greenfield import ws32_delivery_compile as historical
from scripts.greenfield import ws32_rolled_prefill_compile as original
from scripts.greenfield.ws32_prefill_owned_state import CONTRACT

PREREQUISITES = {
    "docs/artifacts/prefill-capture-barrier-db615-sealed-20260912.json": "a6bfafd61990c1ae67e1f6dd3818e1cee24ab7b63718fc59a8cf5ec54e9728d2",
    "docs/artifacts/prefill-delivery-long-compile-oom-20260911.json": "08205198675bce0dd828648946f128ce15b70290ef6574787ae66b6f30c9676f",
}


def require_source(repo: Path) -> None:
    """Bind the current DB615 model tree, including default-off cache changes."""
    capture.require_source(repo)
    for name, expected in PREREQUISITES.items():
        path = repo / name
        if path.is_symlink() or not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("delivery source prerequisite differs")
    if raw_registration("128k_d1_0") != {
        "prefill_chunk": (20691979, "ed6ab5731ef17819e210f688f634a8539d09583c11ed3f89fd04e6c60df06cf1"),
        "prefill_tail": (20788062, "59843169d5ca21fe350e1e5b7384a0c53512afa7ac7e159b059ac05c69f91834"),
    }:
        raise ValueError("delivery original L7 RAW registration differs")


def state_ownership(context_label: str) -> str | None:
    """Trusted workload choice, never an ownership value from worker evidence."""
    long_plan(context_label)
    return CONTRACT if context_label == "256k_e0" else None


def raw_registration(context_label: str) -> dict[str, tuple[int, str]]:
    """Exact original graph bytes; not a short-profile or execution exemption."""
    long_plan(context_label)
    if context_label == "256k_e0":
        return dict.fromkeys(("prefill_chunk", "prefill_tail"), capture.RAW[capture.PROGRAM])
    return {role: historical.RAW["prefill_128k_" + role.removeprefix("prefill_")]
            for role in ("prefill_chunk", "prefill_tail")}


def read_metadata(repo: Path) -> Any:
    """Authenticate the existing complete metadata without loading payloads."""
    return original.read_metadata(repo, full_canonical=True, delivery_source=True)


def prepare(mesh: Any, metadata: Any, *, repo: Path,
            context_label: str) -> original.AbstractPrefillPair:
    """Reuse acquired programs at fixed L7/E0 geometry, with abstract inputs."""
    long_plan(context_label)  # Reject unknown workloads before source/metadata work.
    require_source(repo)
    if context_label != "256k_e0":
        return original.prepare(mesh, metadata, repo=repo, full_canonical=True,
                                long_context_label=context_label, delivery_source=True)
    pair = capture.prepare(mesh, metadata, repo=repo)
    program, inputs = pair.programs[capture.PROGRAM], pair.inputs[capture.PROGRAM]
    return original.AbstractPrefillPair(
        dict.fromkeys(("prefill_chunk", "prefill_tail"), program),
        dict.fromkeys(("prefill_chunk", "prefill_tail"), inputs),
        pair.manifest_sha256, pair.source_inventory_sha256,
    )
