#!/usr/bin/env python3
"""CPU-only localization of the persistent 8K event-1 DSA mismatch.

Three executed protected 8K runs with the hidden-width accepted RMS schedule
left event 1 (layer-1 indexer, decode position 8155) at seven selected-set
mismatches while event 0 stayed exact.  This diagnostic rederives, from sealed
bytes alone, where the event-1 deviation lives:

* the legacy layer-1 DSA query, head weights and current key at position 8155
  (sealed ``dsa_internals`` recovery oracle);
* the greenfield layer-1 prompt index cache captured by the executed DB518
  numerical run (``layer1_index_cache_owners_bfloat16_bits``, two owners,
  16 pages, 256 rows, 128 lanes, position ``p`` at owner ``(p % 512) // 256``,
  page ``p // 512``, row ``p % 256``);
* the sealed short-context DSA oracle (legacy selected positions/scores);
* the protected runs' DSA observer captures.

It recomputes the signed DSA scores exactly as ``dsa_scores`` does under TPU
default precision (one BF16 pass on both operands, FP32 accumulation, FP32
``128**-0.5`` scale, ReLU, signed head weighting) and compares the resulting
top-2048 set with the oracle and with the device.  Event 0 over the already
proven-exact layer-0 prompt cache calibrates the emulation: its set must be
exact and its score residual at FP32-ULP level.

If event 1 recomputed from the *legacy* decode-side tensors over the
*greenfield* prompt cache reproduces the device selection exactly while
differing from the oracle by the same swapped positions, then the greenfield
layer-1 prompt cache by itself is sufficient to produce the observed event-1
set mismatch: substituting the legacy decode-side tensors does not change the
selection.  This does not prove the protected run's own query, head weights or
current key are legacy-exact (they are not captured), and it does not exclude
smaller decode-side deviations below the selection threshold; it identifies
the prompt cache as the first boundary that must be made exact.  This is CPU
evidence: it proves no fix and makes no Gate-D, decoder, DB or performance
claim.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ORACLE_ROOT = Path("/home/gianl/gcs-models/oracles/greenfield/glm52")
DSA_EVENTS_ORACLE = (
    ORACLE_ROOT
    / "short_context_dsa/8k"
    / "greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z"
    / "oracle/dsa_events.safetensors"
)
DB518_RESULT = Path(
    "/home/gianl/glm-run/greenfield_pp16_feature2_layer0_db518_numerical_20260829T115022665987633Z/result.npz"
)
DB518_RESULT_SHA256 = (
    "534bacc54d74992f5a8ab4d422f9fa0947523d59325b4bfa272d4fbeb56262f0"
)
DSA_EVENTS_ORACLE_SHA256 = (
    "b591a4622a8c646799989f235bc98bcf8ae99e9e04d2ad55a982d70ba00dde82"
)
LEGACY_INTERNALS_SHA256 = {
    0: "212eb4cc9e76f865bac71f1c75635a19620b70d3ccb9e95d30e2c45f6f4aa835",
    1: "eb7a250072d23531707130deaabb6906590c6661e70637fe2cedf8069d35a9da",
}
RUN_ROOT = Path("/home/gianl/glm-run")
ACCEPTED_RUN_TAG = (
    "greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_"
    "token_splitres_prefill_keyfix_queryexact_headkeyexact_scoredefault_mainrope_"
    "ras_pregatheredb512_strategynd_o_densefinalconv_oracle_dsa_metaparent_trace2_"
    "20260902T213510823966642Z"
)
BASELINE_RUN_TAG = (
    "greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_"
    "token_splitres_prefill_keyfix_queryexact_headkeyexact_scoredefault_mainrope_"
    "pregatheredb512_strategynd_o_densefinalconv_oracle_dsa_metaparent_trace2_"
    "20260828T111624599159951Z"
)
OBSERVER_RELATIVE = "dsa_observer/step_00_position_8155.npz"
DECODE_POSITION = 8155
PROMPT_ROWS = 8155
SELECTED_WIDTH = 2048
HEAD_DIM = 128
PAGE_ROWS = 512
ROWS_PER_OWNER = 256
EVENT_LAYERS = {0: 0, 1: 1}
DEFAULT_OUTPUT = Path(
    "docs/artifacts/gate-d-event1-layer1-prompt-cache-offline-diagnosis.json"
)
ARTIFACT_KIND = "gate_d_event1_layer1_prompt_cache_offline_diagnosis"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bf16_round(values: np.ndarray) -> np.ndarray:
    """Round FP32 values to BF16 (round-to-nearest-even) and widen back."""

    import ml_dtypes

    return (
        np.asarray(values, dtype=np.float32)
        .astype(ml_dtypes.bfloat16)
        .astype(np.float32)
    )


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    import ml_dtypes

    array = np.ascontiguousarray(np.asarray(bits, dtype=np.uint16))
    return array.view(ml_dtypes.bfloat16).astype(np.float32)


def prompt_cache_rows(owner_bits: np.ndarray, *, rows: int) -> np.ndarray:
    """Map the DB518 owner/page/row cache capture to ``[rows, lanes]`` BF16 bits.

    Position ``p`` lives at owner ``(p % 512) // 256``, page ``p // 512`` and
    owner-local row ``p % 256`` (the DB518 position-113 comparison layout).
    """

    bits = np.asarray(owner_bits)
    if bits.ndim != 4 or bits.dtype != np.uint16:
        raise ValueError("prompt cache capture must be uint16 [owner, page, row, lane]")
    owners, pages, page_rows, _ = bits.shape
    if owners * page_rows != PAGE_ROWS or page_rows != ROWS_PER_OWNER:
        raise ValueError("prompt cache capture owner/page geometry drifted")
    if rows > pages * PAGE_ROWS:
        raise ValueError("prompt cache capture holds fewer rows than requested")
    positions = np.arange(rows, dtype=np.int64)
    local = positions % PAGE_ROWS
    return np.ascontiguousarray(
        bits[local // ROWS_PER_OWNER, positions // PAGE_ROWS, local % ROWS_PER_OWNER]
    )


def default_precision_dsa_scores(
    query: np.ndarray, keys: np.ndarray, head_weights: np.ndarray
) -> np.ndarray:
    """Emulate ``dsa_scores(..., precision="default")`` on TPU.

    Default matmul precision rounds both operands to BF16 and accumulates the
    products in FP32; the scale ``HEAD_DIM**-0.5`` is an FP32 constant; ReLU
    precedes the signed head weighting.  Products are accumulated in FP64 and
    rounded to FP32 once, which stays within FP32 ULPs of any TPU summation
    order for 128-term dots.
    """

    query = np.asarray(query, dtype=np.float32)
    keys = np.asarray(keys, dtype=np.float32)
    head_weights = np.asarray(head_weights, dtype=np.float32)
    if query.ndim != 2 or query.shape[1] != HEAD_DIM:
        raise ValueError("query must be [heads, 128]")
    if keys.ndim != 2 or keys.shape[1] != HEAD_DIM:
        raise ValueError("keys must be [rows, 128]")
    if head_weights.shape != (query.shape[0],):
        raise ValueError("head weights must be [heads]")
    per_head = bf16_round(query).astype(np.float64) @ bf16_round(keys).astype(
        np.float64
    ).T
    per_head = per_head.astype(np.float32).astype(np.float64) * np.float64(
        np.float32(HEAD_DIM**-0.5)
    )
    per_head = np.maximum(per_head, 0.0)
    return (head_weights.astype(np.float64)[:, None] * per_head).sum(axis=0)


def top_set(scores: np.ndarray, width: int) -> set[int]:
    return set(np.argsort(-scores, kind="stable")[:width].tolist())


def observer_event(observation: np.ndarray, event: int) -> dict[int, float]:
    """Decode one stage-0 event row of the protected DSA observer capture."""

    if observation.ndim != 3 or observation.shape[2] != 2 * SELECTED_WIDTH + 2:
        raise ValueError("observer layout drifted from (ranks, slots, 2W+2)")
    row = observation[0][event]
    positions = row[:SELECTED_WIDTH]
    scores = np.ascontiguousarray(row[SELECTED_WIDTH : 2 * SELECTED_WIDTH]).view(
        np.float32
    )
    count = int(row[2 * SELECTED_WIDTH])
    if int(row[2 * SELECTED_WIDTH + 1]) != EVENT_LAYERS[event]:
        raise ValueError("observer producer layer drifted")
    return dict(zip(positions[:count].tolist(), scores[:count].tolist()))


def compare_sets(
    reference: dict[int, float], candidate_scores: np.ndarray
) -> dict[str, Any]:
    candidate = top_set(candidate_scores, len(reference))
    reference_set = set(reference)
    deltas = np.array(
        [candidate_scores[p] - reference[p] for p in reference], dtype=np.float64
    )
    return {
        "expected_only": sorted(reference_set - candidate),
        "observed_only": sorted(candidate - reference_set),
        "score_delta_mean_abs": float(np.abs(deltas).mean()),
        "score_delta_max_abs": float(np.abs(deltas).max()),
        "score_delta_nonzero": int(np.count_nonzero(deltas)),
    }


def legacy_internals_path(layer: int) -> Path:
    pattern = str(
        ORACLE_ROOT
        / "dsa_internals/8k/recovery"
        / f"greenfield_legacy_layer{layer}_dsa_internals_recovery_*"
        / "source_dumps/w2"
        / f"internals.model_layers_{layer}_self_attn_attn.position{DECODE_POSITION}.proc0.npz"
    )
    matches = sorted(glob.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one legacy layer-{layer} internals npz, found {matches}")
    return Path(matches[0])


def load_oracle_events() -> tuple[np.ndarray, np.ndarray]:
    from safetensors import safe_open

    if sha256_file(DSA_EVENTS_ORACLE) != DSA_EVENTS_ORACLE_SHA256:
        raise RuntimeError("sealed 8K DSA events oracle differs from the pinned bytes")
    with safe_open(str(DSA_EVENTS_ORACLE), framework="np") as handle:
        positions = np.asarray(handle.get_tensor("selected_positions"))[0]
        scores = np.asarray(handle.get_tensor("selected_scores"))[0].astype(np.float32)
        producers = np.asarray(handle.get_tensor("producer_layer_ids"))
        decode_positions = np.asarray(handle.get_tensor("decode_positions"))
    producers = producers[0] if producers.ndim == 2 else producers
    if int(decode_positions.reshape(-1)[0]) != DECODE_POSITION:
        raise RuntimeError("oracle decode position drifted")
    for event, layer in EVENT_LAYERS.items():
        if int(producers[event]) != layer:
            raise RuntimeError("oracle producer layer drifted")
    return positions, scores


def diagnose(*, accepted_run: Path, baseline_run: Path) -> dict[str, Any]:
    if sha256_file(DB518_RESULT) != DB518_RESULT_SHA256:
        raise RuntimeError("DB518 result.npz differs from the sealed capture")
    db518 = np.load(DB518_RESULT)
    oracle_positions, oracle_scores = load_oracle_events()
    observers = {
        "accepted_hidden_width_schedule": np.load(accepted_run / OBSERVER_RELATIVE)[
            "observation"
        ],
        "baseline_default_schedule": np.load(baseline_run / OBSERVER_RELATIVE)[
            "observation"
        ],
    }
    inputs = {
        "db518_result_npz": {"path": str(DB518_RESULT), "sha256": DB518_RESULT_SHA256},
        "dsa_events_oracle": {
            "path": str(DSA_EVENTS_ORACLE),
            "sha256": sha256_file(DSA_EVENTS_ORACLE),
        },
        "observer_accepted": {
            "run_tag": accepted_run.name,
            "sha256": sha256_file(accepted_run / OBSERVER_RELATIVE),
        },
        "observer_baseline": {
            "run_tag": baseline_run.name,
            "sha256": sha256_file(baseline_run / OBSERVER_RELATIVE),
        },
    }
    events: dict[str, Any] = {}
    for event, layer in EVENT_LAYERS.items():
        internals_path = legacy_internals_path(layer)
        internals_sha = sha256_file(internals_path)
        if internals_sha != LEGACY_INTERNALS_SHA256[layer]:
            raise RuntimeError(f"legacy layer-{layer} internals differ from the sealed capture")
        inputs[f"legacy_layer{layer}_internals"] = {
            "path": str(internals_path.relative_to(ORACLE_ROOT)),
            "sha256": internals_sha,
        }
        internals = np.load(internals_path)
        if str(internals["layer_name"]) != f"model.layers.{layer}.self_attn.attn" or (
            int(internals["position"]) != DECODE_POSITION
        ):
            raise RuntimeError("legacy internals identity drifted")
        prompt_keys = bf16_bits_to_f32(
            prompt_cache_rows(
                db518[f"layer{layer}_index_cache_owners_bfloat16_bits"], rows=PROMPT_ROWS
            )
        )
        current_key = bf16_round(internals["current_key"].astype(np.float32))
        keys = np.concatenate([prompt_keys, current_key[None, :]], axis=0)
        scores = default_precision_dsa_scores(
            internals["query"].astype(np.float32),
            keys,
            internals["head_weights"].astype(np.float32),
        )
        oracle = dict(
            zip(oracle_positions[event].tolist(), oracle_scores[event].tolist())
        )
        record: dict[str, Any] = {
            "producer_layer_id": layer,
            "keys": "legacy_decode_side_over_greenfield_db518_prompt_cache",
            "vs_oracle": compare_sets(oracle, scores),
        }
        for name, observation in observers.items():
            device = observer_event(observation, event)
            record[f"vs_device_{name}"] = compare_sets(device, scores)
            device_vs_oracle = {
                "expected_only": sorted(set(oracle) - set(device)),
                "observed_only": sorted(set(device) - set(oracle)),
            }
            record[f"device_{name}_vs_oracle"] = device_vs_oracle
        events[f"event_{event}"] = record

    event0 = events["event_0"]
    event1 = events["event_1"]
    calibration_exact = (
        not event0["vs_oracle"]["expected_only"]
        and not event0["vs_oracle"]["observed_only"]
        and event0["vs_oracle"]["score_delta_max_abs"] < 1e-5
    )
    accepted = event1["vs_device_accepted_hidden_width_schedule"]
    reproduces_device = not accepted["expected_only"] and not accepted["observed_only"]
    same_swaps_as_oracle = (
        event1["vs_oracle"]["expected_only"]
        == event1["device_accepted_hidden_width_schedule_vs_oracle"]["expected_only"]
        and event1["vs_oracle"]["observed_only"]
        == event1["device_accepted_hidden_width_schedule_vs_oracle"]["observed_only"]
    )
    sufficient = calibration_exact and reproduces_device and same_swaps_as_oracle
    classification = ";".join(
        [
            "EVENT0_CALIBRATION_EXACT" if calibration_exact else "EVENT0_CALIBRATION_FAILED",
            (
                "EVENT1_LEGACY_DECODE_SIDE_OVER_GREENFIELD_PROMPT_CACHE_REPRODUCES_DEVICE_SELECTION"
                if reproduces_device
                else "EVENT1_DEVICE_SELECTION_NOT_REPRODUCED"
            ),
            (
                "EVENT1_LAYER1_PROMPT_INDEX_CACHE_SUFFICIENT_FOR_OBSERVED_SET_MISMATCH"
                if sufficient
                else "EVENT1_PROMPT_INDEX_CACHE_SUFFICIENCY_NOT_SHOWN"
            ),
            "DECODE_SIDE_EXACTNESS_NOT_PROVEN;CPU_EVIDENCE_ONLY;NO_FIX_PROVEN;NO_GATE_D_NO_DECODER_NO_DB_NO_PERFORMANCE_CLAIM;GATE_D_OPEN",
        ]
    )
    return {
        "artifact_kind": ARTIFACT_KIND,
        "classification": classification,
        "decode_position": DECODE_POSITION,
        "events": events,
        "inputs": inputs,
        "interpretation": (
            "Event 1 recomputed from the legacy layer-1 query, head weights and "
            "current key over the greenfield DB518 layer-1 prompt index cache "
            "reproduces the protected accepted-schedule run's selection exactly "
            "and the oracle's swapped positions exactly. The greenfield layer-1 "
            "prompt cache is therefore sufficient, on its own, to produce the "
            "observed event-1 set mismatch; substituting legacy decode-side "
            "tensors does not change the selection. This does not prove the "
            "protected run's own query/head weights/current key are legacy-exact "
            "and does not exclude smaller decode-side deviations below the "
            "selection threshold; it identifies the layer-1 prompt cache built "
            "by the teacher-forced prefill (single-token decode arithmetic per "
            "prompt row, versus the legacy's batched prefill) as the first "
            "boundary that must be made exact. The layer-0 prompt index cache is "
            "already proven exact; layer-0 prompt main-cache rows differed only "
            "in the main-RoPE suffix (DB530/DB531)."
            if sufficient
            else "Sufficiency criteria not met; see events."
        ),
        "method": {
            "score_emulation": "bf16 operands, fp64 products/accumulate rounded once to fp32, fp32 128**-0.5 scale, relu, signed head weights",
            "prompt_cache_layout": "owner=(p%512)//256, page=p//512, row=p%256; current position key from legacy internals rounded to bf16",
            "selected_width": SELECTED_WIDTH,
        },
        "performance_claim": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--accepted-run", type=Path, default=RUN_ROOT / ACCEPTED_RUN_TAG)
    parser.add_argument("--baseline-run", type=Path, default=RUN_ROOT / BASELINE_RUN_TAG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = diagnose(accepted_run=args.accepted_run, baseline_run=args.baseline_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["classification"])
    for name in ("event_0", "event_1"):
        record = result["events"][name]
        print(
            name,
            "vs oracle:",
            len(record["vs_oracle"]["expected_only"]),
            "/",
            len(record["vs_oracle"]["observed_only"]),
            f"mean|d|={record['vs_oracle']['score_delta_mean_abs']:.3e}",
            "| vs accepted device:",
            len(record["vs_device_accepted_hidden_width_schedule"]["expected_only"]),
            "/",
            len(record["vs_device_accepted_hidden_width_schedule"]["observed_only"]),
            f"mean|d|={record['vs_device_accepted_hidden_width_schedule']['score_delta_mean_abs']:.3e}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
