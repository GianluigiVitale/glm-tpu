"""Source-bound corrective2K recipe; no inherited numerical or speed result.

Reuse DB609's actual graph pair and the existing protected short-run machinery.
Only dense0..2 changes; old profiles retain their original model-source guards.
8K is deliberately not registered here: it follows this correction's own2K.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any


PROFILE = "ws32_b128_b114_2k_cap8192_canonical_dense_v1"
SOURCE_PIN = "94b8081c5eea57f618e0a2dcfc32beb327e6c2bc"
COMPILER_RECEIPT = (
    "docs/artifacts/prefill-canonical-model-compile-db609-sealed-20260909.json"
)
COMPILER_SHA256 = "f6079d834255f9ca9a2c7199a14ceec729c0da04a438b0008021b81035558747"
STRUCTURAL_RECEIPT = (
    "docs/artifacts/prefill-canonical-fullmodel-hlo-local-20260909.json"
)
STRUCTURAL_SHA256 = "d6ce933c8b3a2a4f076fdbac6cb16d80e15a3bc847d516db30352c7cf7686f4a"


def registration(repo: Path) -> dict[str, Any]:
    """Compose previously reviewed source/raw/memory recipes by exact content.

    The compiler module is imported only when called to avoid an import cycle
    with its existing shared short-memory validator. It is a metadata utility,
    never a model execution import. Its source guard is also reused separately
    at pre-load admission; these evidence checks alone do not authorize dispatch.
    """
    from scripts.greenfield import ws32_canonical_prefill_compile as compiler
    from . import ws32_prefill_admission as short

    evidence = {
        **compiler.PREREQUISITES,
        COMPILER_RECEIPT: COMPILER_SHA256,
        STRUCTURAL_RECEIPT: STRUCTURAL_SHA256,
    }
    for name, expected in evidence.items():
        path = repo / name
        if path.is_symlink() or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("canonical short prerequisite evidence drifted")
    acquired = json.loads((repo / COMPILER_RECEIPT).read_text())
    if acquired["code_hash"] != SOURCE_PIN or set(acquired["graphs"]) != set(
        compiler.RAW
    ):
        raise ValueError("canonical short acquired graph/source identity drifted")
    graphs = {}
    for graph, (size, digest) in compiler.RAW.items():
        actual = acquired["graphs"][graph]
        if (actual["stablehlo_bytes"], actual["stablehlo_sha256"]) != (size, digest):
            raise ValueError("canonical short raw graph registration drifted")
        graphs[graph] = dict(stablehlo_bytes=size, stablehlo_sha256=digest)
    return dict(
        profile=PROFILE,
        model_source_pin=SOURCE_PIN,
        model_source_overrides=dict(compiler.MODEL_SOURCE_OVERRIDES),
        evidence=evidence,
        graphs=graphs,
        acquired_graphs=acquired["graphs"],
        memory=short.rolled_registration(repo)["memory"],
        numerical_inheritance=False,
        performance_claim=False,
    )


def require_source(repo: Path) -> None:
    from scripts.greenfield.ws32_canonical_prefill_compile import (
        require_source as original,
    )

    registration(repo)
    original(repo)
