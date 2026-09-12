"""Explicit state-ownership candidate for the demonstrated long-prefill OOM.

Not a numerical profile or launcher. Historical programs/guards remain intact.
The caller consumes argument 2 and must subsequently use ONLY result.state,
including after a refused transaction. No input state may be retried or observed
after dispatch. A thrown dispatch is terminal, not permission to reuse buffers.
Weight, WK, token/count and RoPE arguments are never donated.

Donation permits allocation reuse; it does not promise aliases, sufficient HBM,
TPU numerical equivalence, or removal of the internal rollback cache generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax

from glm_tpu.greenfield.runtime.ws32_batched_prefill import Ws32BatchedPrefillProgram
from scripts.greenfield import ws32_rolled_prefill_compile as original


CONTRACT = "ws32-prefill-consumed-state-argument2-v1"
DONATE_ARGNUMS = (2,)
PROGRAM = "prefill_256k_state_donated"


@dataclass(frozen=True)
class OwnedStatePrefillProgram:
    """Distinct type: never present a donating executable as the old profile."""

    original_program: Ws32BatchedPrefillProgram
    execute: Any
    ownership_contract: str = CONTRACT


def consume_state(program: Ws32BatchedPrefillProgram) -> OwnedStatePrefillProgram:
    """Wrap the unchanged computation with an explicit, opt-in ownership ABI.

    This does not donate weights or change the model's commit/refuse branches.
    In the failure branch original cache VALUES survive in the returned state;
    Python references to consumed inputs do not retain ownership of those values.
    """
    if type(program) is not Ws32BatchedPrefillProgram:
        raise TypeError("state ownership requires an original batched program")
    return OwnedStatePrefillProgram(
        program, jax.jit(program.execute, donate_argnums=DONATE_ARGNUMS)
    )


def prepare_worst_capacity(
    mesh: Any, metadata: Any, *, repo: Path
) -> original.AbstractPrefillPair:
    """One changed E0 graph, abstract inputs only; no repeated 128K acquisition.

    The existing preparation builds both roles without lowering or executing
    them. At E0 they have identical B128 geometry; only one is retained here.
    No checkpoint payload read, placement, model/WK call or HBM admission.
    """
    pair = original.prepare(
        mesh, metadata, repo=repo, full_canonical=True,
        long_context_label="256k_e0",
    )
    return original.AbstractPrefillPair(
        {PROGRAM: consume_state(pair.programs["prefill_chunk"])},
        {PROGRAM: pair.inputs["prefill_chunk"]},
        pair.manifest_sha256, pair.source_inventory_sha256,
    )


# CPU TPU-target lowering, not an actual TPU optimized-graph acquisition.
RAW = {PROGRAM: (20859926, "55c3d5775eb5630a61fd8e5cd54caa30e447cf92c3ed09dcafaae0217a63cba1")}
