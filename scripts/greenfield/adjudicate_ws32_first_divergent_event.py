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
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

KAPPA = 2.0
# §21.2 item 4 requires "the same pre-registered kappa = 2" as items 3's caps, so
# the bias factor IS kappa and moves with it; a rerun at kappa = 1 tightens all
# three tests together.
# §21.2 amendment 2026-09-05 22:20Z pins the deviation statistic to the
# population standard deviation (numpy ddof=0), which is what the sealed Gate D
# adjudication computed. ddof=0 yields a smaller s and therefore a strictly
# tighter bound than the sample form; the tool reproduces the sealed number
# bit-for-bit rather than silently changing the statistic.
STD_DDOF = 0
ANALYSIS_ARTIFACT_KIND = "gate_d_ws32_first_divergent_event_adjudication"
# §21.5 records two reference conventions whose rows have identical length and
# identical producer layer. The convention is therefore not inferable from the
# row and must be declared and carried into the record.
REFERENCE_CONVENTIONS = ("rms_norm_eps_1e-5", "rms_norm_eps_1e-6")
COMMITTED_ARTIFACT_DIR = REPO_ROOT / "docs" / "artifacts"
COMMITTED_ARTIFACT_PREFIX = "gate-d-"


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


def _scan_window(archive: Path, oracle_dir: Path) -> tuple[int, int]:
    """Return the (steps, events) window both archives actually contain.

    The window is a property of the evidence, never an operator choice: a CLI
    value that truncates the scan could pre-register the wrong event, and one
    that over-runs it would index out of bounds.
    """

    from safetensors.numpy import load_file

    oracle = load_file(str(oracle_dir / "dsa_events.safetensors"))
    with np.load(archive, allow_pickle=False) as handle:
        engine_shape = np.asarray(handle["dsa_selected_valid_counts"]).shape
    oracle_shape = np.asarray(oracle["valid_counts"]).shape
    if len(engine_shape) != 3 or len(oracle_shape) != 2:
        raise SystemExit(
            f"unexpected observation geometry: engine {engine_shape}, oracle {oracle_shape}"
        )
    if engine_shape[:2] != oracle_shape[:2]:
        raise SystemExit(
            f"engine observation window {engine_shape[:2]} does not match the oracle's "
            f"{oracle_shape[:2]}"
        )
    return int(engine_shape[0]), int(engine_shape[1])


def _first_divergent_event(archive: Path, oracle_dir: Path) -> tuple[int, int]:
    from safetensors.numpy import load_file

    steps, events = _scan_window(archive, oracle_dir)
    oracle = load_file(str(oracle_dir / "dsa_events.safetensors"))
    with np.load(archive, allow_pickle=False) as handle:
        engine_positions = np.asarray(handle["dsa_selected_positions"])
        engine_counts = np.asarray(handle["dsa_selected_valid_counts"])
    for step in range(steps):
        for event in range(events):
            oracle_count = int(oracle["valid_counts"][step, event])
            expected = set(oracle["selected_positions"][step, event, :oracle_count].tolist())
            count = int(engine_counts[step, event, 0])
            observed = set(engine_positions[step, event, 0, :count].tolist())
            if observed != expected:
                return step, event
    raise SystemExit("no divergent DSA event: the run is exact against the oracle")


def adjudicate(
    *,
    archive: Path,
    oracle_dir: Path,
    reference_row: Path,
    step: int,
    event: int,
    decode_position: int | None = None,
    expected_producer_layer_id: int | None = None,
    kappa: float = KAPPA,
) -> dict:
    reference = np.load(reference_row).astype(np.float64)
    if reference.ndim != 1:
        raise SystemExit("reference row must be a one-dimensional score vector")
    if decode_position is not None and reference.shape != (decode_position + 1,):
        # The reference cutoff is the k-th largest over the whole row, so a row
        # built for a different decode position silently shifts it.
        raise SystemExit(
            f"reference row length {reference.shape[0]} does not match decode position "
            f"{decode_position} (expected {decode_position + 1})"
        )
    oracle_positions, oracle_scores = _oracle_event(oracle_dir, step, event)
    engine_positions, engine_scores, producer = _engine_event(archive, step, event)
    if expected_producer_layer_id is not None and producer != expected_producer_layer_id:
        raise SystemExit(
            f"adjudicated event produces layer {producer}, but the reference row was "
            f"built for layer {expected_producer_layer_id}"
        )
    if engine_positions.size != oracle_positions.size:
        raise SystemExit(
            f"engine selected {engine_positions.size} positions, oracle {oracle_positions.size}: "
            "a count mismatch is not an adjudicable boundary swap"
        )
    expected = set(oracle_positions.tolist())
    observed = set(engine_positions.tolist())
    shared = sorted(expected & observed)
    if not shared:
        raise SystemExit("engine and oracle share no selected position")
    if len(expected) != oracle_positions.size or len(observed) != engine_positions.size:
        # dict(zip(...)) would silently keep the last score of a repeated
        # position and shrink the comparison set.
        raise SystemExit("selected positions repeat within an event")
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
        "artifact_kind": ANALYSIS_ARTIFACT_KIND,
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
        "bias_factor": kappa,
        "reference_band_size": band_size,
        "std_ddof": STD_DDOF,
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
    parser.add_argument("--reference-row-sha256", required=True)
    parser.add_argument("--reference-convention", choices=REFERENCE_CONVENTIONS, required=True)
    parser.add_argument("--engine-source-run", required=True)
    parser.add_argument("--context", choices=("2k", "8k"), required=True)
    parser.add_argument("--decode-position", type=int, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--event-index", type=int, default=None)
    # The reference row is event-dependent: it must be declared for the event
    # being adjudicated, and the row length must match the decode position.
    parser.add_argument("--reference-producer-layer-id", type=int, required=True)
    # The scan window is read from the archives; these only assert what the
    # operator believed it to be, and a disagreement is a refusal.
    parser.add_argument("--observer-steps", type=int, default=None)
    parser.add_argument("--events", type=int, default=None)
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

    reference_row, reference_relative, reference_digest = _committed_reference_row(
        args.reference_row, args.reference_row_sha256
    )

    archive = args.observer_npz.resolve()
    if not _archive_belongs_to_run(archive, args.engine_source_run):
        # Without this the declared source run is free text and the sealer's
        # anti-circularity guard can be walked past by mislabelling the archive.
        raise SystemExit(
            f"observer archive {archive} does not belong to run {args.engine_source_run}"
        )
    archive_digest = _digest(archive)

    steps, events = _scan_window(archive, args.oracle_dir)
    for name, declared in (("--observer-steps", args.observer_steps), ("--events", args.events)):
        actual = steps if name == "--observer-steps" else events
        if declared is not None and declared != actual:
            raise SystemExit(f"{name}={declared} disagrees with the archived window {actual}")
    step, event = _resolve_event(archive, args.oracle_dir, args.step, args.event_index)
    analysis = adjudicate(
        archive=archive,
        oracle_dir=args.oracle_dir,
        reference_row=reference_row,
        step=step,
        event=event,
        decode_position=args.decode_position,
        expected_producer_layer_id=args.reference_producer_layer_id,
    )
    analysis["engine_source_run"] = args.engine_source_run
    analysis["observation_window"] = {"events": events, "steps": steps}
    analysis["observer_archive"] = {"path": str(archive), "sha256": archive_digest}
    analysis["reference_row"] = {
        "convention": args.reference_convention,
        "path": str(reference_relative),
        "sha256": reference_digest,
    }
    basis = [
        {"path": item, "sha256": _digest(REPO_ROOT / item)} for item in args.basis
    ]
    _require_prior_attempts_disclosed(args.analysis_output, analysis, basis)
    _write_once(args.analysis_output, analysis)
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
        "reference_row": {
            "convention": args.reference_convention,
            "path": str(reference_relative),
            "sha256": reference_digest,
        },
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
    if analysis["verdict"] != "PASS":
        # A failed adjudication must not leave a schema-valid pre-registration
        # artifact behind: the loader validates the schema, not the verdict.
        print(
            json.dumps(
                {
                    "analysis_sha256": _digest(args.analysis_output),
                    "event_index": event,
                    "failed_checks": sorted(
                        name for name, item in analysis["checks"].items() if not item["pass"]
                    ),
                    "record_written": False,
                    "step": step,
                    "verdict": analysis["verdict"],
                },
                sort_keys=True,
            )
        )
        return 2
    _write_once(args.record_output, record)
    print(
        json.dumps(
            {
                "analysis_sha256": _digest(args.analysis_output),
                "event_index": event,
                "record_sha256": _digest(args.record_output),
                "record_written": True,
                "step": step,
                "swapped": len(analysis["expected_only"]),
                "verdict": analysis["verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


def _committed_reference_row(
    reference_row: Path, expected_sha256: str
) -> tuple[Path, str, str]:
    """Bind the FP64 reference row to a committed, pre-registered artifact.

    The row fixes ``eps``, the cutoff, the band and every bound, so it decides
    the verdict outright; §21.5 also records two conventions whose rows have the
    same length and the same producer layer, so nothing about the file itself
    distinguishes the right one. It must therefore be a committed artifact of
    this repository AND match a SHA-256 declared on the command line.
    """

    resolved = Path(reference_row).resolve()
    try:
        relative = str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        relative = None
    if (
        relative is None
        or resolved.parent != COMMITTED_ARTIFACT_DIR
        or not resolved.name.startswith(COMMITTED_ARTIFACT_PREFIX)
        or resolved.suffix != ".npy"
    ):
        raise SystemExit(
            "reference row must be a committed docs/artifacts/gate-d-*.npy artifact of this "
            f"repository: {reference_row}"
        )
    if not _is_tracked(relative):
        # Placing a file in docs/artifacts is not the same as committing it.
        # A tracked row is reviewable; an untracked one is a local artefact
        # that could have been fitted to the run under adjudication.
        raise SystemExit(f"reference row is not tracked by git: {relative}")
    digest = _digest(resolved)
    if digest != expected_sha256:
        raise SystemExit(
            f"reference row SHA-256 {digest} does not match the pre-registered {expected_sha256}"
        )
    return resolved, relative, digest


def _is_tracked(relative: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch", "--", relative],
        capture_output=True,
    )
    return result.returncode == 0


def _resolve_event(
    archive: Path, oracle_dir: Path, step: int | None, event_index: int | None
) -> tuple[int, int]:
    """Locate the first divergent event, whatever the operator asked for.

    The record asserts that every event before the adjudicated one is exact, so
    the scan runs even when a step and event are named explicitly; naming a
    later, benign event is a refusal rather than a silent unwarranted claim.
    """

    first_step, first_event = _first_divergent_event(archive, oracle_dir)
    if step is None and event_index is None:
        return first_step, first_event
    if step is None or event_index is None:
        raise SystemExit("--step and --event-index must be given together")
    if (step, event_index) != (first_step, first_event):
        raise SystemExit(
            f"requested event (step {step}, event {event_index}) is not the first divergent "
            f"event (step {first_step}, event {first_event})"
        )
    return first_step, first_event


def _archive_belongs_to_run(archive: Path, run_tag: str) -> bool:
    """True when the observer archive is demonstrably this run's own output.

    A protected run writes ``/home/gianl/glm-run/<tag>/runner.rank0.npz`` and a
    ``summary.json`` naming the tag beside it; either witness is accepted, an
    unlabelled copy is not.
    """

    if archive.parent.name == run_tag:
        return True
    summary = archive.parent / "summary.json"
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("run_tag") == run_tag


def _require_prior_attempts_disclosed(
    analysis_output: Path, analysis: dict, basis: list[dict]
) -> None:
    """Refuse to write unless every earlier attempt on this event is in the basis.

    ``_write_once`` only blocks re-use of one path, so an operator could
    otherwise re-run with a different output name until a PASS appeared and
    register a record whose basis names only the attempt that passed.
    """

    declared = {item["path"] for item in basis}
    undisclosed = []
    for candidate in sorted(analysis_output.parent.glob("*.json")):
        if candidate.resolve() == analysis_output.resolve():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) or payload.get("artifact_kind") != ANALYSIS_ARTIFACT_KIND:
            continue
        same_event = (
            payload.get("engine_source_run") == analysis["engine_source_run"]
            and payload.get("step") == analysis["step"]
            and payload.get("event_index") == analysis["event_index"]
        )
        if not same_event:
            continue
        try:
            relative = str(candidate.resolve().relative_to(REPO_ROOT))
        except ValueError:
            relative = str(candidate)
        if relative not in declared:
            undisclosed.append(relative)
    if undisclosed:
        raise SystemExit(
            "earlier adjudication attempts on this event are not declared in --basis: "
            + ", ".join(undisclosed)
        )


def _write_once(path: Path, value: dict) -> None:
    if path.exists():
        raise SystemExit(f"append-only adjudication output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
