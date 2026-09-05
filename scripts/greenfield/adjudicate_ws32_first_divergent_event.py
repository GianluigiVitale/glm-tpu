#!/usr/bin/env python3
"""Adjudicate a protected WS32 run's first divergent DSA event (spec §21.2 items 3-4).

CPU only: it reads one run's observer arrays, the sealed legacy DSA oracle and the
archived FP64 reference row, applies the §21.2 boundary test, and emits both the
analysis and the pre-registration record that a *later* protected run is sealed
against.  It never re-uses the analysed run as its own adjudication basis: the
record names the run it was derived from, and the sealer binds it by SHA-256 to a
different run's pins.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

KAPPA = 2.0


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _oracle_event(oracle_dir: Path, step: int, event: int) -> tuple[np.ndarray, np.ndarray]:
    from safetensors.numpy import load_file

    data = load_file(str(oracle_dir / "dsa_events.safetensors"))
    count = int(data["valid_counts"][step, event])
    return (
        np.asarray(data["selected_positions"][step, event, :count], dtype=np.int64),
        np.asarray(data["selected_scores"][step, event, :count], dtype=np.float64),
    )


def _engine_event(archive: Path, step: int, event: int) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(archive, allow_pickle=False) as handle:
        count = int(handle["dsa_selected_valid_counts"][step, event, 0])
        positions = np.asarray(handle["dsa_selected_positions"][step, event, 0, :count], dtype=np.int64)
        scores = np.asarray(handle["dsa_selected_scores"][step, event, 0, :count], dtype=np.float64)
        producer = int(np.asarray(handle["dsa_producer_layer_ids"])[event])
    return positions, scores, producer


def _first_divergent_event(archive: Path, oracle_dir: Path, steps: int, events: int) -> tuple[int, int]:
    for step in range(steps):
        for event in range(events):
            oracle_positions, _ = _oracle_event(oracle_dir, step, event)
            engine_positions, _, _ = _engine_event(archive, step, event)
            if set(engine_positions.tolist()) != set(oracle_positions.tolist()):
                return step, event
    raise SystemExit("no divergent DSA event: the run is exact against the oracle")


def adjudicate(
    *,
    archive: Path,
    oracle_dir: Path,
    reference_row: Path,
    step: int,
    event: int,
    kappa: float = KAPPA,
) -> dict:
    reference = np.load(reference_row).astype(np.float64)
    oracle_positions, oracle_scores = _oracle_event(oracle_dir, step, event)
    engine_positions, engine_scores, producer = _engine_event(archive, step, event)
    expected = set(oracle_positions.tolist())
    observed = set(engine_positions.tolist())
    shared = sorted(expected & observed)
    if not shared:
        raise SystemExit("engine and oracle share no selected position")
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
    checks = {
        "cap_max_abs": {
            "engine": float(np.abs(engine_delta).max()),
            "bound": kappa * epsilon,
            "pass": bool(np.abs(engine_delta).max() <= kappa * epsilon),
        },
        "cap_std": {
            "engine": float(engine_delta.std()),
            "bound": kappa * float(oracle_delta.std()),
            "pass": bool(engine_delta.std() <= kappa * oracle_delta.std()),
        },
        "bias": {
            "engine_mean": float(engine_delta.mean()),
            "bound": 2.0 * abs(float(oracle_delta.mean()))
            + 3.0 * float(engine_delta.std()) / np.sqrt(count),
            "pass": bool(
                abs(float(engine_delta.mean()))
                <= 2.0 * abs(float(oracle_delta.mean()))
                + 3.0 * float(engine_delta.std()) / np.sqrt(count)
            ),
        },
        "reference_band": {
            "maximum_abs_offset": max((abs(v) for v in band.values()), default=0.0),
            "bound": epsilon,
            "pass": all(abs(v) <= epsilon for v in band.values()),
        },
        "equal_sized_disjoint_swap": {
            "expected_only": len(expected - observed),
            "observed_only": len(observed - expected),
            "pass": len(expected - observed) == len(observed - expected) > 0,
        },
    }
    return {
        "artifact_kind": "gate_d_ws32_first_divergent_event_adjudication",
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
            "std": float(oracle_delta.std()),
        },
        "engine_delta": {
            "max_abs": float(np.abs(engine_delta).max()),
            "mean": float(engine_delta.mean()),
            "std": float(engine_delta.std()),
        },
        "producer_layer_id": producer,
        "shared_positions": count,
        "step": step,
        "swap_reference_offsets": band,
        "top_k": top_k,
        "verdict": "PASS" if all(item["pass"] for item in checks.values()) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observer-npz", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--reference-row", type=Path, required=True)
    parser.add_argument("--engine-source-run", required=True)
    parser.add_argument("--context", choices=("2k", "8k"), required=True)
    parser.add_argument("--decode-position", type=int, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--event-index", type=int, default=None)
    parser.add_argument("--observer-steps", type=int, default=14)
    parser.add_argument("--events", type=int, default=21)
    parser.add_argument("--later-event-alarm", type=int, default=1024)
    parser.add_argument("--date-utc", required=True)
    parser.add_argument("--analysis-output", type=Path, required=True)
    parser.add_argument("--record-output", type=Path, required=True)
    parser.add_argument("--basis", action="append", default=[])
    args = parser.parse_args()

    from glm_tpu.greenfield.validation.short_context_dsa_oracle import (  # noqa: E402
        inspect_short_context_dsa_oracle,
    )
    from glm_tpu.greenfield.validation.short_context_oracle import (  # noqa: E402
        inspect_short_context_oracle,
    )

    dsa_manifest = inspect_short_context_dsa_oracle(args.oracle_dir)
    token_manifest = dsa_manifest["token_oracle"]["manifest_sha256"]
    if args.step is None or args.event_index is None:
        step, event = _first_divergent_event(
            args.observer_npz, args.oracle_dir, args.observer_steps, args.events
        )
    else:
        step, event = args.step, args.event_index
    analysis = adjudicate(
        archive=args.observer_npz,
        oracle_dir=args.oracle_dir,
        reference_row=args.reference_row,
        step=step,
        event=event,
    )
    analysis["engine_source_run"] = args.engine_source_run
    analysis["reference_row"] = {
        "path": str(args.reference_row),
        "sha256": _digest(args.reference_row),
    }
    _write_once(args.analysis_output, analysis)
    basis = [
        {"path": item, "sha256": _digest(REPO_ROOT / item)} for item in args.basis
    ]
    basis.append(
        {
            "path": str(args.analysis_output.relative_to(REPO_ROOT)),
            "sha256": _digest(args.analysis_output),
        }
    )
    record = {
        "artifact_kind": "gate_d_ws32_8k_adjudicated_divergence",
        "schema_version": 1,
        "spec_section": "21.2 items 3-4 (scope: first divergent event)",
        "date_utc": args.date_utc,
        "context": args.context,
        "decode_position": args.decode_position,
        "step": step,
        "event_index": event,
        "producer_layer_id": analysis["producer_layer_id"],
        "expected_only": analysis["expected_only"],
        "observed_only": analysis["observed_only"],
        "later_event_alarm": args.later_event_alarm,
        "basis": basis,
        "oracle": {
            "dsa_manifest_sha256": dsa_manifest["manifest_sha256"],
            "token_manifest_sha256": token_manifest,
        },
        "engine_source_run": args.engine_source_run,
        "semantics": (
            "A protected WS32 run passes DSA correctness only if every event before this one is exact "
            "against the oracle, this event differs from the oracle by exactly these two position sets, "
            "and every event thereafter keeps exact counts, tails and within-engine score order; "
            "later-event set differences are recorded with their sizes and raise a diagnostic alarm "
            "above later_event_alarm without refusing. Raw tokens must remain exact for all protected steps."
        ),
        "gate_d_closed": False,
        "performance_claim": False,
    }
    _write_once(args.record_output, record)
    print(
        json.dumps(
            {
                "analysis_sha256": _digest(args.analysis_output),
                "event_index": event,
                "record_sha256": _digest(args.record_output),
                "step": step,
                "swapped": len(analysis["expected_only"]),
                "verdict": analysis["verdict"],
            },
            sort_keys=True,
        )
    )
    return 0 if analysis["verdict"] == "PASS" else 2


def _write_once(path: Path, value: dict) -> None:
    if path.exists():
        raise SystemExit(f"append-only adjudication output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
