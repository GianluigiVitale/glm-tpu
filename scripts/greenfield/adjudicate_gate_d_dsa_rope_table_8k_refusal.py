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
    "legacy_glm_dsa_indexer_py": "d44225e3a1952ef81d57eb30f6bd466d46e3e2cd0288d4cc798d182cf7a2e626",
}
# Run provenance: the two protected runs are 183 commits apart, so the event-0 delta is
# consistent with the table but is not attributed to the table alone by this artifact.
RUN_PROVENANCE = {
    "without_table_20260828": {
        "pin": "570cc453f58a27ac0d06b129076c78e2f187ddd6",
        "dsa_rope_table": 0,
        "files": {
            "orchestrator.log": "8a2f66043dab98a4cd0953a74eb4431c0fd71eaa6395b0fbadb563ebb3cc0ccb",
            "sync.txt": "c5ee268ab6e498d4ce1caa1d029be4683aed554b484e0ad845a1980cfc374c8c",
            "hlo/decoder_78layer_8k_token.hlo_contract.json": "f0fcb1f5065e187283e0120862327380bb93f9498ad6d5a2a3cb83962c7db561",
            "hlo/decoder_78layer_8k_token_dsa_observer.hlo_contract.json": "0e5b733fd2381965733e06cbb888604f0b4054df04ec9cc135dae9a314eddda7",
            "decoder.rank0.log": "b337184c8a63939c0c6da3930dccbb1e6709bbb081934048158fd6b04e67058e",
        },
    },
    "with_table_20260902": {
        "pin": "ba7d1e7240125d75119769784d12aada5a2cfa84",
        "dsa_rope_table": 1,
        "files": {
            "orchestrator.log": "b48324ef2f74ff7500ee0c73d58909cefea311f402145247d381fcae0fcb6c7c",
            "sync.txt": "224705cb3f20f6308ada0f2f4af79b4c84d53aa3eb1e1143fabca2d0c06e8dae",
            "hlo/decoder_78layer_8k_token.hlo_contract.json": "895296468d6eb5bdf32db0460d8f48050916ce1de0a2ef93e198afb37e96c399",
            "hlo/decoder_78layer_8k_token_dsa_observer.hlo_contract.json": "2f2c92088a017ed310405fba15e75f15c9fb84e7e027a441cb96e0c578334b04",
            "decoder.rank0.log": "e451581eb3370858cb3437581250064bb8c043d707c445b0a9b986588eee65b7",
        },
    },
}
LEGACY_INDEXER = Path("/home/gianl/tpu-inference/tpu_inference/layers/vllm/custom_ops/glm_dsa_indexer.py")
LEGACY_PIN = "b3c25df47ac98783912dc658878181ec0a8ae16d"
REPLICAS_PER_PRODUCER = 4


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
    # Recover the schedule from the rows themselves and enforce the observer schema:
    # int32 rows of width 2*W+2, every oracle producer present exactly REPLICAS_PER_PRODUCER
    # times (one row per rank of its stage group), replicas bit-identical, no foreign producer.
    if observation.dtype != np.dtype(np.int32) or observation.ndim != 3:
        raise SystemExit(f"observer schema drift: {observation.dtype} {observation.shape}")
    if observation.shape[2] != 2 * SELECTED_WIDTH + 2:
        raise SystemExit(f"observer row width drift: {observation.shape[2]}")
    rows_by_producer: dict[int, list[np.ndarray]] = {}
    for lane in range(observation.shape[0]):
        for slot in range(observation.shape[1]):
            row = observation[lane, slot]
            producer = int(row[2 * SELECTED_WIDTH + 1])
            if producer < 0:
                continue
            rows_by_producer.setdefault(producer, []).append(row)
    if sorted(rows_by_producer) != sorted(producers):
        raise SystemExit(f"producer set drift: {sorted(rows_by_producer)} != {sorted(producers)}")
    found: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
    for producer, rows in rows_by_producer.items():
        if len(rows) != REPLICAS_PER_PRODUCER:
            raise SystemExit(f"producer {producer} has {len(rows)} replicas, expected {REPLICAS_PER_PRODUCER}")
        for other in rows[1:]:
            if not np.array_equal(rows[0], other):
                raise SystemExit(f"producer {producer} replicas differ bitwise")
        row = rows[0]
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
        raw_bits_equal = bool(
            order_mismatch == 0
            and count == expected_count
            and np.array_equal(
                scores[:count].view(np.uint32), exp_sc[event][:expected_count].view(np.uint32)
            )
        )
        events.append(
            {
                "event_index": event,
                "producer_layer_id": producer,
                "valid_count": count,
                "expected_valid_count": expected_count,
                "expected_only_count": int(np.setdiff1d(expected_live, live).size),
                "observed_only_count": int(np.setdiff1d(live, expected_live).size),
                "order_mismatch_count": order_mismatch,
                "raw_bits_equal_to_oracle": raw_bits_equal,
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
        "event0_bit_exact": e0["raw_bits_equal_to_oracle"],
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
        run_dir = RUN_ROOT / tag
        path = run_dir / "dsa_observer" / "step_00_position_8155.npz"
        hashes[name] = sha256_file(path)
        provenance = RUN_PROVENANCE[name]
        bound_files = {}
        for rel, expected in provenance["files"].items():
            actual = sha256_file(run_dir / rel)
            if actual != expected:
                raise SystemExit(f"{name}: {rel} SHA drift {actual} != {expected}")
            bound_files[rel] = actual
        orchestrator = (run_dir / "orchestrator.log").read_text()
        if f"PIN={provenance['pin']}" not in orchestrator:
            raise SystemExit(f"{name}: pin {provenance['pin']} absent from orchestrator.log")
        contract = json.loads((run_dir / "hlo/decoder_78layer_8k_token.hlo_contract.json").read_text())
        if bool(contract.get("dsa_rope_table_enabled", False)) != bool(provenance["dsa_rope_table"]):
            raise SystemExit(f"{name}: HLO contract dsa_rope_table_enabled disagrees with provenance")
        observation = np.load(path)["observation"]
        result["runs"][name] = {
            "tag": tag,
            "pin": provenance["pin"],
            "dsa_rope_table": provenance["dsa_rope_table"],
            "main_rope_table_enabled": bool(contract.get("main_rope_table_enabled", False)),
            "bound_files_sha256": bound_files,
            "observation_sha256": hashes[name],
            **compare(observation, oracle),
        }
    hashes["legacy_glm_dsa_indexer_py"] = sha256_file(LEGACY_INDEXER)
    legacy_src = LEGACY_INDEXER.read_text()
    rope_fn = legacy_src[legacy_src.index("def rope_cos_sin(") : legacy_src.index("def apply_rope(")]
    legacy_on_device_cos_sin = "jnp.cos(freqs), jnp.sin(freqs)" in rope_fn and "positions[:, None] * inv_freq[None, :]" in rope_fn
    if not legacy_on_device_cos_sin:
        raise SystemExit("legacy rope_cos_sin body drifted from the expected on-device jnp.cos/jnp.sin form")
    result["legacy_source"] = {
        "path": str(LEGACY_INDEXER),
        "git_pin": LEGACY_PIN,
        "sha256": hashes["legacy_glm_dsa_indexer_py"],
        "rope_cos_sin_uses_on_device_jnp_cos_sin": legacy_on_device_cos_sin,
    }
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
        "HOST_ROTARY_TABLE_REFUTED_AS_LEGACY_FAITHFULNESS_FIX"
        if f["event0_bit_exact_without_table"] and not f["event0_bit_exact_with_table"]
        and f["event1_set_mismatch_with_table"] >= f["event1_set_mismatch_without_table"] - 1
        and result["legacy_source"]["rope_cos_sin_uses_on_device_jnp_cos_sin"]
        else "INCONCLUSIVE"
    )
    result["scope_note"] = (
        "The two runs are at different pins (570cc453 vs ba7d1e72, 183 commits apart); the event-0 "
        "delta is consistent with the table but is not attributed to the table alone. The decisive "
        "facts are (a) event 0 bit-exact against the legacy oracle without the table and (b) the legacy "
        "indexer evaluating cos/sin on device, which together refute the table as a faithfulness fix. "
        "The location and mechanism of the event-1 divergence are not determined by this artifact."
    )
    result["classification"] = (
        f"{verdict};EVENT_0_DELTA_ATTRIBUTION_TO_TABLE_ALONE_NOT_PROVEN;"
        "EVENT_1_MECHANISM_UNDETERMINED;PROTECTED_8K_REFUSED_TWICE_AT_EVENT_1;GATE_D_OPEN"
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["classification"])
    print(json.dumps(f, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
