"""Spec §21.2 items 3-4: the first-divergent-event adjudication itself.

The computation lives here, in the package the SEALER imports, rather than in
the offline tool, for the same reason the reviewed reference-row registry does:
a verdict the sealer merely reads is a claim by whoever wrote the file.  The
sealer re-derives items 3-4 from the run's own observations, the sealed oracle
and the pre-registered reference row, and refuses if the re-derivation does not
pass.  The pre-registered record fixes the row and the divergence sets; the
numbers are computed fresh from the data being sealed.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# §21.2 items 3-4: one pre-registered kappa scales the caps and the bias rule.
KAPPA = 2.0
# §21.2 amendment 2026-09-05 22:20Z: the population standard deviation, which is
# what the sealed Gate D adjudication computed and a strictly tighter bound than
# the sample form.
STD_DDOF = 0


class AdjudicationError(ValueError):
    """The event cannot be adjudicated under §21.2 items 3-4 at all."""


def adjudicate_first_divergent_event(
    *,
    oracle_positions: np.ndarray,
    oracle_scores: np.ndarray,
    engine_positions: np.ndarray,
    engine_scores: np.ndarray,
    reference: np.ndarray,
    producer_layer_id: int,
    step: int,
    event: int,
    decode_position: int | None = None,
    expected_producer_layer_id: int | None = None,
    kappa: float = KAPPA,
) -> dict[str, Any]:
    """Adjudicate one event. Pure: arrays in, verdict out."""

    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim != 1:
        raise AdjudicationError("reference row must be a one-dimensional score vector")
    if decode_position is not None and reference.shape != (decode_position + 1,):
        # The cutoff is the k-th largest over the whole row, so a row built for
        # a different decode position silently shifts it.
        raise AdjudicationError(
            f"reference row length {reference.shape[0]} does not match decode position "
            f"{decode_position} (expected {decode_position + 1})"
        )
    if expected_producer_layer_id is not None and producer_layer_id != expected_producer_layer_id:
        raise AdjudicationError(
            f"adjudicated event produces layer {producer_layer_id}, but the reference row "
            f"was built for layer {expected_producer_layer_id}"
        )
    oracle_positions = np.asarray(oracle_positions, dtype=np.int64)
    engine_positions = np.asarray(engine_positions, dtype=np.int64)
    oracle_scores = np.asarray(oracle_scores, dtype=np.float64)
    engine_scores = np.asarray(engine_scores, dtype=np.float64)
    if engine_positions.size != oracle_positions.size:
        raise AdjudicationError(
            f"engine selected {engine_positions.size} positions, oracle "
            f"{oracle_positions.size}: a count mismatch is not an adjudicable boundary swap"
        )
    expected = set(oracle_positions.tolist())
    observed = set(engine_positions.tolist())
    if len(expected) != oracle_positions.size or len(observed) != engine_positions.size:
        # dict(zip(...)) would silently keep the last score of a repeated
        # position and shrink the comparison set.
        raise AdjudicationError("selected positions repeat within an event")
    outside = sorted(
        position
        for position in expected | observed
        if position < 0 or position >= reference.size
    )
    if outside:
        # Indexing the FP64 row with these would wrap or raise; either way the
        # verdict would be meaningless.
        raise AdjudicationError(
            f"selected positions fall outside the reference row of {reference.size}: "
            f"{outside[:8]}"
        )
    shared = sorted(expected & observed)
    if not shared:
        raise AdjudicationError("engine and oracle share no selected position")
    oracle_by_position = dict(zip(oracle_positions.tolist(), oracle_scores))
    engine_by_position = dict(zip(engine_positions.tolist(), engine_scores))
    oracle_delta = np.array([oracle_by_position[p] for p in shared]) - reference[shared]
    engine_delta = np.array([engine_by_position[p] for p in shared]) - reference[shared]
    order = np.argsort(-reference, kind="stable")
    top_k = int(oracle_positions.size)
    cutoff = float(reference[order[top_k - 1]])
    epsilon = float(np.abs(oracle_delta).max())
    swapped = sorted(expected ^ observed)
    band = {int(p): float(reference[p] - cutoff) for p in swapped}
    count = len(shared)
    band_size = int(np.count_nonzero(np.abs(reference - cutoff) <= epsilon))
    bias_bound = kappa * abs(float(oracle_delta.mean())) + 3.0 * float(
        engine_delta.std(ddof=STD_DDOF)
    ) / np.sqrt(count)
    checks = {
        "cap_max_abs": {
            "engine": float(np.abs(engine_delta).max()),
            "bound": kappa * epsilon,
            "pass": bool(np.abs(engine_delta).max() <= kappa * epsilon),
        },
        "cap_std": {
            "engine": float(engine_delta.std(ddof=STD_DDOF)),
            "bound": kappa * float(oracle_delta.std(ddof=STD_DDOF)),
            "pass": bool(
                engine_delta.std(ddof=STD_DDOF) <= kappa * oracle_delta.std(ddof=STD_DDOF)
            ),
        },
        "bias": {
            "engine_mean": float(engine_delta.mean()),
            "bound": bias_bound,
            "pass": bool(abs(float(engine_delta.mean())) <= bias_bound),
        },
        "reference_band": {
            "maximum_abs_offset": max((abs(v) for v in band.values()), default=0.0),
            "bound": epsilon,
            "pass": all(abs(v) <= epsilon for v in band.values()),
        },
        "reference_band_capacity": {
            # §21.2 item 3: the reference ambiguity band cannot explain more
            # swaps than it holds.
            "band_size": band_size,
            "swapped": len(swapped),
            "pass": len(swapped) <= band_size,
        },
        "equal_sized_disjoint_swap": {
            "expected_only": len(expected - observed),
            "observed_only": len(observed - expected),
            "pass": len(expected - observed) == len(observed - expected) > 0,
        },
    }
    return {
        "checks": checks,
        "cutoff_reference_score": cutoff,
        "epsilon_oracle_vs_reference": epsilon,
        "event_index": event,
        "expected_only": sorted(expected - observed),
        "kappa": kappa,
        "observed_only": sorted(observed - expected),
        "oracle_delta": {
            "max_abs": float(np.abs(oracle_delta).max()),
            "mean": float(oracle_delta.mean()),
            "std": float(oracle_delta.std(ddof=STD_DDOF)),
        },
        "engine_delta": {
            "max_abs": float(np.abs(engine_delta).max()),
            "mean": float(engine_delta.mean()),
            "std": float(engine_delta.std(ddof=STD_DDOF)),
        },
        "reference_band_size": band_size,
        "std_ddof": STD_DDOF,
        "producer_layer_id": int(producer_layer_id),
        "shared_positions": count,
        "step": step,
        "swap_reference_offsets": band,
        "top_k": top_k,
        "verdict": "PASS" if all(item["pass"] for item in checks.values()) else "FAIL",
    }


ADJUDICATION_CHECKS = frozenset(
    {
        "bias",
        "cap_max_abs",
        "cap_std",
        "equal_sized_disjoint_swap",
        "reference_band",
        "reference_band_capacity",
    }
)


def event_arrays(
    *,
    selected_positions: np.ndarray,
    selected_scores: np.ndarray,
    selected_valid_counts: np.ndarray,
    step: int,
    event: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The valid prefix of one observed event, from a run's own arrays."""

    count = int(np.asarray(selected_valid_counts)[step, event, 0])
    positions = np.asarray(selected_positions)[step, event, 0, :count]
    scores = np.asarray(selected_scores)[step, event, 0, :count]
    return np.asarray(positions, dtype=np.int64), np.asarray(scores, dtype=np.float64)


def oracle_event_arrays(
    oracle: Mapping[str, Any], *, step: int, event: int
) -> tuple[np.ndarray, np.ndarray]:
    """The valid prefix of one oracle event."""

    count = int(np.asarray(oracle["valid_counts"])[step, event])
    positions = np.asarray(oracle["selected_positions"])[step, event, :count]
    scores = np.asarray(oracle["selected_scores"])[step, event, :count]
    return np.asarray(positions, dtype=np.int64), np.asarray(scores, dtype=np.float64)
