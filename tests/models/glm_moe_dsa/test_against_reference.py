"""The unsharded reference model against the production composition (``cpu32``; DESIGN 7.6).

``test_reference_matches_production_composition`` runs :mod:`tests.reference.oracle_run` in a child with
32 forced CPU devices on the frozen fixture v1 (the three-token prompt ``short`` and the 157-token prompt
``a``, eight decode steps, the reference teacher-forced with production's tokens) and asserts the acceptance
criteria of ``tests/reference/VALIDATION.md`` against production's program set:

* every DESIGN 7.6 criterion that the two accepted engines, the frozen FP8 oracle and production, met
  against each other holds exactly for the reference (``SHORT``, ``PROMPT_A``);
* for the others (decode floats on both prompts, prompt-A prefill floats, DSA set equality and score order)
  the reference is no further from production than production was from the FP8 oracle: the floors of
  ``floors.json``, recorded from the oracle's final run before it was archived at
  ``archive/research-20260922``, with the slack ``FLOOR_SLACK`` and ``FRACTION_SLACK``. These floor-relative
  criteria are the deviation from the DESIGN 7.6 gate that the integrator sanctioned on 2026-09-23.

Known limits (``VALIDATION.md``): the criteria are dominated by the rows a flipped discrete decision moves, so
a small systematic error can hide under them (a routed scale of 2.4 instead of 2.5 passes); the fixture's
RMSNorm and DSA key-norm weights are all ones and its router correction bias is all zeros, so how production
applies them is invisible here. The semantic check is the FP64 restatement of
``tests/reference/test_reference_consistency.py`` (fast tier), whose TINY checkpoint has non-unit norm
weights and a nonzero bias.

Semantic choices of the reference, shared with production, where the engine contract differs from the
vendored Hugging Face eager modeling: the indexer RoPE pairs are interleaved (``indexer_rope_interleave:
true``); the q-a/kv-a LoRA RMSNorm uses ``rms_norm_eps`` (1e-5; HF constructs them with 1e-6);
``routed_scaling_factor`` scales the routed sum (HF: the routing weights; equal in exact arithmetic); the main
RoPE output keeps its pairs interleaved (HF concatenates the halves; the q.k product is identical); the DSA
head weights are FP32 in both. Within the engine contract the reference uses one association where the
engine has two (the indexer key LayerNorm divides by ``sqrt`` in prefill and multiplies by ``rsqrt`` in
decode; the routed sum is FP32 in prefill and one owner's routes are summed in BF16 first in decode); they
differ by FP32 rounding only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


# ----------------------------------------------------------------------------- cross-validation (cpu32)
# Floors: how far production was from the frozen FP8 oracle on the same prompt and schedule
# (VALIDATION.md), recorded from the oracle's final run at S2f in ``floors.json``. Where even the
# two accepted engines break a DESIGN 7.6 criterion, the reference must be no further from either
# engine than they are from each other. These are a recorded deviation from the DESIGN 7.6 gate
# (VALIDATION.md); they are dominated by the rows a flipped decision moves, so they bound the
# reference loosely and the semantic checks are the independent-restatement tests of
# tests/reference/test_reference_consistency.py. Measured worst on the four pairs (VALIDATION.md):
# bound ratio 1.087 x the floor, outside fraction 1.71 x the floor, carried-layer set-unequal steps
# equal to the floor's.
FLOOR_SLACK = 1.15  # worst |difference| / bound, a maximum over ~1e6-1e7 elements
FRACTION_SLACK = 2.0  # elements outside the bound (a moved row moves as a whole)


FLOORS = Path(__file__).with_name("floors.json")


def recorded_floor(prompt: str) -> dict[str, Any]:
    """The production:fp8-oracle report of ``prompt`` as far as the criteria read it."""
    import json

    return json.loads(FLOORS.read_text())["floors"][prompt]


@pytest.fixture(scope="module")
def reports() -> Any:
    from tools.equivalence.common import run_child

    cache: dict[tuple[str, str], dict[str, Any]] = {}

    def get(pair: str, prompt: str) -> dict[str, Any]:
        if (pair, prompt) not in cache:
            cache[pair, prompt] = run_child(
                "tests.reference.oracle_run",
                "--pair",
                pair,
                "--prompt",
                prompt,
                timeout=1800,
            )
        return cache[pair, prompt]

    return get


def assert_criteria(report: dict[str, Any], floor: dict[str, Any], *, strict: tuple[str, ...]) -> None:
    s, f = report["summary"], floor["summary"]
    assert report["engine_inputs_equal_reference"] == dict(wk=True, rope=True)
    failed = [key for key in strict if not s[key]]
    assert not failed, (failed, s)
    for phase in ("prefill", "decode"):
        for leaf, ratio in s[phase + "_bound_ratio"].items():
            # Within the bound wherever the engines are; otherwise within the engines' own distance.
            engines = f[phase + "_bound_ratio"][leaf]
            assert ratio <= (1.0 if engines <= 1.0 else FLOOR_SLACK * engines), (
                phase,
                leaf,
                ratio,
                f,
            )
        for leaf, fraction in s[phase + "_outside_fraction"].items():
            # A zero floor fraction means the engines are within the bound: zero allowed.
            assert fraction <= FRACTION_SLACK * f[phase + "_outside_fraction"][leaf], (
                phase,
                leaf,
                fraction,
                f,
            )
    carried = [r["selections"][str(max(map(int, r["selections"])))] for r in report["decode"]]
    floor_carried = [next(iter(r["selections"].values())) for r in floor["decode"]]
    assert sum(not c["set_equal"] for c in carried) <= sum(not c["set_equal"] for c in floor_carried)


SHORT = (
    "prefill_integer_leaves_equal",
    "prefill_float_leaves_within",
    "prefill_next_token_equal",
    "decode_tokens_equal",
    "decode_integer_state_equal",
    "decode_selections_set_equal",
    "decision_free_within",
)
PROMPT_A = (
    "prefill_integer_state_equal",
    "prefill_next_token_equal",
    "decode_integer_state_equal",
    "decision_free_within",
    "tokens_equal_or_tied",
    "selections_explained",
)


@pytest.mark.slow
@pytest.mark.cpu32
def test_reference_matches_production_composition(reports):
    assert_criteria(
        reports("reference:production", "short"),
        recorded_floor("short"),
        strict=SHORT,
    )
    assert_criteria(
        reports("reference:production", "a"),
        recorded_floor("a"),
        strict=PROMPT_A,
    )


def test_recorded_floors_are_the_final_oracle_runs():
    import json

    value = json.loads(FLOORS.read_text())
    assert value["pair"] == "production:fp8-oracle" and set(value["floors"]) == {"short", "a"}
    for floor in value["floors"].values():
        assert set(floor["summary"]) == {
            "prefill_bound_ratio",
            "prefill_outside_fraction",
            "decode_bound_ratio",
            "decode_outside_fraction",
        }
        assert floor["decode"] and all(len(r["selections"]) == 1 for r in floor["decode"])
