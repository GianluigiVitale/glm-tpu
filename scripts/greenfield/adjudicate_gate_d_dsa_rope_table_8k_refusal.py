#!/usr/bin/env python3
"""Adjudicate the 2026-09-02 DSA-host-rotary-table protected 8K refusal on CPU.

Compares the archived DSA observer tensors of two protected fail-closed 8K runs
(2026-08-28 without the table, 2026-09-02 with the table) against the accepted
legacy DSA oracle, position-aligned per event, from archived bytes only. No JAX,
no TPU, no model. Pure numpy + safetensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

SELECTED_WIDTH = 2048
ORACLE_DIR = Path(
    "/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/8k/"
    "greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z/oracle"
)
RUN_ROOT = Path("/home/gianl/glm-run")
RUNS = {
    "without_table_20260828": (
        "greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_"
        "splitres_prefill_keyfix_queryexact_headkeyexact_scoredefault_mainrope_pregatheredb512_"
        "strategynd_o_densefinalconv_oracle_dsa_metaparent_trace2_20260828T111624599159951Z"
    ),
    "with_table_20260902": (
        "greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_"
        "splitres_prefill_keyfix_queryexact_headkeyexact_scoredefault_mainrope_dr_pregatheredb512_"
        "strategynd_o_densefinalconv_oracle_dsa_metaparent_trace2_20260902T021346091708582Z"
    ),
}
EXPECTED_SHA256 = {
    "oracle_events": "b591a4622a8c646799989f235bc98bcf8ae99e9e04d2ad55a982d70ba00dde82",
    "without_table_20260828": "17f0916de291be472437319c848f3f4331d5bdd6dac325755ef1f4ecbaf8bd7e",
    "with_table_20260902": "dfc21a7b7bc22db0fb66c1e84673cef4d3c887d5432e7b9e162b87eee5bad6de",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_rows(observation: np.ndarray, event: int, slots_per_stage: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    # Stage-0 lane 0 holds slots 0..slots_per_stage-1 (events 0..); later stages
    # are read from their first lane. The observer stores every stage's rows on
    # every lane of that stage; producer id is bound in the row itself.
    stage, slot = divmod(event, slots_per_stage)
    lane = stage * 8  # eight ranks per PP8 stage group
    row = observation[lane, slot]
    positions = row[:SELECTED_WIDTH]
    scores = np.ascontiguousarray(row[SELECTED_WIDTH : 2 * SELECTED_WIDTH]).view(np.float32)
    return positions, scores, int(row[2 * SELECTED_WIDTH]), int(row[2 * SELECTED_WIDTH + 1])


def compare(observation: np.ndarray, oracle: dict[str, np.ndarray]) -> dict:
    producers = oracle["producer_layer_ids"].tolist()
    exp_pos = oracle["selected_positions"][0]
    exp_sc = oracle["selected_scores"][0]
    counts = oracle["valid_counts"][0]
    events = []
    # Recover the schedule from the rows themselves: walk lanes/slots and match producer ids.
    found: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
    for lane in range(observation.shape[0]):
        for slot in range(observation.shape[1]):
            row = observation[lane, slot]
            producer = int(row[2 * SELECTED_WIDTH + 1])
            if producer < 0 or producer in found:
                continue
            positions = row[:SELECTED_WIDTH]
            scores = np.ascontiguousarray(row[SELECTED_WIDTH : 2 * SELECTED_WIDTH]).view(np.float32)
            found[producer] = (positions, scores, int(row[2 * SELECTED_WIDTH]))
    for event, producer in enumerate(producers):
        positions, scores, count = found[producer]
        expected_count = int(counts[event])
        live = positions[:count]
        expected_live = exp_pos[event][:expected_count]
        expected_map = {int(p): float(s) for p, s in zip(expected_live, exp_sc[event][:expected_count])}
        aligned = np.array(
            [scores[i] - expected_map[int(live[i])] for i in range(count) if int(live[i]) in expected_map],
            dtype=np.float64,
        )
        order_mismatch = int(np.count_nonzero(positions != exp_pos[event]))
        events.append(
            {
                "event_index": event,
                "producer_layer_id": producer,
                "valid_count": count,
                "expected_valid_count": expected_count,
                "expected_only_count": int(np.setdiff1d(expected_live, live).size),
                "observed_only_count": int(np.setdiff1d(live, expected_live).size),
                "order_mismatch_count": order_mismatch,
                "aligned_count": int(aligned.size),
                "aligned_nonzero_count": int(np.count_nonzero(aligned)),
                "aligned_max_abs": float(np.abs(aligned).max()) if aligned.size else None,
                "aligned_mean_abs": float(np.abs(aligned).mean()) if aligned.size else None,
                "aligned_mean_signed": float(aligned.mean()) if aligned.size else None,
            }
        )
    e0 = events[0]
    pos0, sc0, c0 = found[producers[0]]
    exp0 = {int(p): float(s) for p, s in zip(exp_pos[0][: counts[0]], exp_sc[0][: counts[0]])}
    position_zero_delta = None
    for i in range(c0):
        if int(pos0[i]) == 0:
            position_zero_delta = float(sc0[i] - exp0[0])
    return {
        "events": events,
        "event0_bit_exact": e0["aligned_nonzero_count"] == 0 and e0["order_mismatch_count"] == 0,
        "event0_position_zero_score_delta": position_zero_delta,
        "first_event_with_set_mismatch": next(
            (e["event_index"] for e in events if e["expected_only_count"] or e["observed_only_count"]), None
        ),
        "total_order_mismatch_count": int(sum(e["order_mismatch_count"] for e in events)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    oracle_path = ORACLE_DIR / "dsa_events.safetensors"
    hashes = {"oracle_events": sha256_file(oracle_path)}
    oracle = load_file(str(oracle_path))
    assert int(oracle["decode_positions"][0]) == 8155
    result: dict = {"artifact_kind": "gate_d_dsa_rope_table_8k_refusal_adjudication", "schema_version": 1, "decode_position": 8155, "runs": {}}
    for name, tag in RUNS.items():
        path = RUN_ROOT / tag / "dsa_observer" / "step_00_position_8155.npz"
        hashes[name] = sha256_file(path)
        observation = np.load(path)["observation"]
        result["runs"][name] = {"tag": tag, "observation_sha256": hashes[name], **compare(observation, oracle)}
    for key, expected in EXPECTED_SHA256.items():
        if hashes[key] != expected:
            raise SystemExit(f"SHA drift for {key}: {hashes[key]} != {expected}")
    without = result["runs"]["without_table_20260828"]
    with_table = result["runs"]["with_table_20260902"]
    e1_without = without["events"][1]
    e1_with = with_table["events"][1]
    result["inputs_sha256"] = hashes
    result["findings"] = {
        "event0_bit_exact_without_table": without["event0_bit_exact"],
        "event0_bit_exact_with_table": with_table["event0_bit_exact"],
        "event0_with_table_aligned_max_abs": with_table["events"][0]["aligned_max_abs"],
        "event0_with_table_position_zero_delta": with_table["event0_position_zero_score_delta"],
        "event1_set_mismatch_without_table": e1_without["expected_only_count"],
        "event1_set_mismatch_with_table": e1_with["expected_only_count"],
        "event1_aligned_mean_abs_without_table": e1_without["aligned_mean_abs"],
        "event1_aligned_mean_abs_with_table": e1_with["aligned_mean_abs"],
    }
    f = result["findings"]
    verdict = (
        "HOST_ROTARY_TABLE_HYPOTHESIS_REFUTED_AT_ENGINE_LEVEL"
        if f["event0_bit_exact_without_table"] and not f["event0_bit_exact_with_table"]
        and f["event1_set_mismatch_with_table"] >= f["event1_set_mismatch_without_table"] - 1
        else "INCONCLUSIVE"
    )
    result["classification"] = f"{verdict};PROTECTED_8K_REFUSED_TWICE_AT_EVENT_1;GATE_D_OPEN"
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["classification"])
    print(json.dumps(f, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
