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
REFERENCE_IMPLEMENTATION = "scripts/greenfield/reference_cpu"
# §21.2 item 3: the FP64 reference row must be "validated against the legacy
# intermediate captures before use", with its implementation and source hash
# recorded. Being committed under a plausible name is not that validation, so
# the rows usable for adjudication are enumerated here, keyed by the event they
# were built for, and each names the reviewed record that validated it. A new
# event needs a new reviewed entry, which is exactly the pre-registration step
# this table exists to force.
REFERENCE_ROWS = {
    ("8k", 8155, 1, "rms_norm_eps_1e-5"): {
        "path": "docs/artifacts/gate-d-event1-fp64-reference-row-20260905.npy",
        "sha256": "bfde8bd92d9f88452d68b3e9d3112a4848b0c0ef086c7207b2431021b233d9a0",
        "validation_path": (
            "docs/artifacts/gate-d-event1-math-reference-adjudication-20260905.json"
        ),
        "validation_sha256": (
            "aeb6ddfa415f60894456cd79cda08d30b30d7994e48b85fc7c0a89e32e6c7e09"
        ),
    },
}
REMOTE_RESULTS_PREFIX = "gs://driftbench-dsv4-uc/results"


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
    steps = min(int(engine_shape[0]), int(oracle_shape[0]))
    events = min(int(engine_shape[1]), int(oracle_shape[1]))
    if steps <= 0 or events <= 0:
        raise SystemExit(
            f"empty observation window: engine {engine_shape[:2]}, oracle {oracle_shape[:2]}"
        )
    # The scan covers the prefix both archives hold. A shorter observation
    # window is adjudicable over what it contains; it is recorded in the
    # analysis so the scope of "first divergent" is never implicit.
    return steps, events


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

    # Both outputs are validated before anything is computed or written: a
    # relative or off-tree path used to raise from `relative_to` only after the
    # analysis had already been written, and the retry then tripped the
    # append-only guard.
    args.analysis_output = _committed_output_path(args.analysis_output, ".json")
    args.record_output = _committed_output_path(args.record_output, ".json")

    from glm_tpu.greenfield.validation.short_context_dsa_oracle import (  # noqa: E402
        inspect_short_context_dsa_oracle,
    )
    from glm_tpu.greenfield.validation.short_context_oracle import (  # noqa: E402
        inspect_short_context_oracle,
    )

    dsa_manifest = inspect_short_context_dsa_oracle(args.oracle_dir)
    token_manifest = dsa_manifest["token_oracle"]["manifest_sha256"]

    reference_row, reference_relative, reference_digest, reference_entry = (
        _committed_reference_row(
            args.reference_row,
            args.reference_row_sha256,
            context=args.context,
            decode_position=args.decode_position,
            producer_layer_id=args.reference_producer_layer_id,
            convention=args.reference_convention,
        )
    )
    reference_implementation = _reference_implementation_tree()

    archive = args.observer_npz.resolve()
    archive_md5 = _require_archive_belongs_to_run(archive, args.engine_source_run)
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
    analysis["observer_archive"] = {
        "md5": archive_md5,
        "path": str(archive),
        "sha256": archive_digest,
    }
    analysis["reference_row"] = {
        "convention": args.reference_convention,
        "implementation_tree_sha1": reference_implementation,
        "path": str(reference_relative),
        "sha256": reference_digest,
    }
    basis = [
        {"path": item, "sha256": _digest(REPO_ROOT / item)} for item in args.basis
    ]
    # The row and the record that validated it are part of the ground the
    # adjudication stands on, so they are named whether or not the operator
    # remembered to pass them.
    for path in (reference_relative, reference_entry["validation_path"]):
        entry = {"path": path, "sha256": _digest(REPO_ROOT / path)}
        if entry not in basis:
            basis.append(entry)
    # Prior attempts are DISCLOSED, not relied upon: they are a separate list so
    # that declaring a failed attempt cannot make the record unsealable.
    prior_attempts = _require_prior_attempts_disclosed(args.analysis_output, analysis)
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
        "prior_attempts": prior_attempts,
        "reference_row": {
            "convention": args.reference_convention,
            "implementation_tree_sha1": reference_implementation,
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
    reference_row: Path,
    expected_sha256: str,
    *,
    context: str,
    decode_position: int,
    producer_layer_id: int,
    convention: str,
) -> tuple[Path, str, str, dict]:
    """Bind the FP64 reference row to a reviewed, pre-registered artifact.

    The row fixes ``eps``, the cutoff, the band and every bound, so whoever
    chooses it chooses the verdict.  Four things must hold, and each closes a
    demonstrated attack: the path is a committed ``docs/artifacts/gate-d-*.npy``
    artifact; its CONTENT is identical to the committed blob (a tracked path can
    be overwritten in the working tree); the (event, convention) it is used for
    appears in ``REFERENCE_ROWS`` with this exact path and digest (a fitted row
    committed under a new plausible name is still not a reviewed row); and the
    caller declares the digest it expected.
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
    if not resolved.exists():
        raise SystemExit(f"reference row does not exist: {relative}")
    _require_committed_content(relative)
    key = (context, int(decode_position), int(producer_layer_id), convention)
    entry = REFERENCE_ROWS.get(key)
    if entry is None:
        raise SystemExit(
            "no reviewed FP64 reference row is registered for "
            f"(context {context}, position {decode_position}, layer {producer_layer_id}, "
            f"{convention}); §21.2 item 3 requires a reviewed row before adjudication"
        )
    if entry["path"] != relative:
        raise SystemExit(
            f"the reviewed reference row for this event is {entry['path']}, not {relative}"
        )
    digest = _digest(resolved)
    if digest != entry["sha256"]:
        raise SystemExit(
            f"reference row content {digest} is not the reviewed row {entry['sha256']}"
        )
    if digest != expected_sha256:
        raise SystemExit(
            f"reference row SHA-256 {digest} does not match the pre-registered {expected_sha256}"
        )
    _require_committed_content(entry["validation_path"])
    validation_digest = _digest(REPO_ROOT / entry["validation_path"])
    if validation_digest != entry["validation_sha256"]:
        raise SystemExit(
            f"the reference row's validation record {entry['validation_path']} drifted"
        )
    return resolved, relative, digest, entry


def _require_committed_content(relative: str) -> None:
    """Refuse a path whose working-tree bytes differ from the committed blob.

    ``git ls-files`` only says the path is in the index; the demonstrated attack
    overwrote a tracked artifact in place and kept a tracked path.
    """

    committed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", f"HEAD:{relative}"],
        capture_output=True,
        text=True,
    )
    if committed.returncode != 0:
        raise SystemExit(f"artifact is not committed at HEAD: {relative}")
    working = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "hash-object", "--", relative],
        capture_output=True,
        text=True,
    )
    if working.returncode != 0:
        raise SystemExit(f"artifact cannot be hashed: {relative}")
    if committed.stdout.strip() != working.stdout.strip():
        raise SystemExit(
            f"artifact differs from the committed blob: {relative} "
            f"(HEAD {committed.stdout.strip()[:12]}, working tree {working.stdout.strip()[:12]})"
        )


def _reference_implementation_tree() -> str:
    """§21.2 item 3: record the reference implementation's source hash."""

    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", f"HEAD:{REFERENCE_IMPLEMENTATION}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"the reference implementation {REFERENCE_IMPLEMENTATION} is not committed at HEAD"
        )
    return result.stdout.strip()


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


def _remote_archive_md5(run_tag: str) -> str | None:
    """MD5 of the observer archive this run itself uploaded, or None."""

    uri = f"{REMOTE_RESULTS_PREFIX}/{run_tag}/host_records/runner.rank0.npz"
    result = subprocess.run(
        ["gcloud", "storage", "hash", "--hex", uri],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("md5_hash:"):
            return stripped.split(":", 1)[1].strip()
    return None


def _local_md5(path: Path) -> str:
    from hashlib import md5

    digest = md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_archive_belongs_to_run(archive: Path, run_tag: str) -> str:
    """Bind the archive to evidence the run itself published, not to a name.

    A directory name and a hand-written ``summary.json`` are both operator
    writable: copying the run being sealed into a directory named after some
    other tag would defeat the sealer's anti-circularity guard.  The binding is
    therefore the object the protected run uploaded under its own tag with
    ``--no-clobber``; a mismatch, or an archive with no published counterpart,
    is a refusal.
    """

    remote = _remote_archive_md5(run_tag)
    if remote is None:
        raise SystemExit(
            f"run {run_tag} has published no observer archive at "
            f"{REMOTE_RESULTS_PREFIX}/{run_tag}/host_records/runner.rank0.npz; the declared "
            "source run cannot be bound to this file"
        )
    local = _local_md5(archive)
    if local != remote:
        raise SystemExit(
            f"observer archive {archive} does not match the archive run {run_tag} published "
            f"(local md5 {local}, published {remote})"
        )
    return remote


def _require_prior_attempts_disclosed(analysis_output: Path, analysis: dict) -> list[dict]:
    """Return every earlier attempt on this event, refusing if any is hidden.

    ``_write_once`` only blocks re-use of one path, so an operator could
    otherwise re-run until a PASS appeared.  Attempts are recognised by their
    SHAPE — a JSON naming this run, step and event and carrying ``checks`` —
    rather than by ``artifact_kind``, which an operator can edit.  They are
    returned as ``prior_attempts`` and are disclosure, never ground: nothing
    downstream requires them to have passed.
    """

    attempts: list[dict] = []
    for candidate in sorted(COMMITTED_ARTIFACT_DIR.glob("*.json")):
        if candidate.resolve() == analysis_output.resolve():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) or "checks" not in payload:
            continue
        if (
            payload.get("engine_source_run") != analysis["engine_source_run"]
            or payload.get("step") != analysis["step"]
            or payload.get("event_index") != analysis["event_index"]
        ):
            continue
        relative = str(candidate.resolve().relative_to(REPO_ROOT))
        attempts.append(
            {
                "path": relative,
                "sha256": _digest(candidate),
                "verdict": str(payload.get("verdict")),
            }
        )
    return attempts


def _committed_output_path(path: Path, suffix: str) -> Path:
    """Every adjudication output lives in the reviewed artifact directory.

    Writing an attempt elsewhere is how the disclosure scan was bypassed: it
    globs one directory, so that directory is where outputs must go.
    """

    resolved = Path(path).resolve()
    if (
        resolved.parent != COMMITTED_ARTIFACT_DIR
        or not resolved.name.startswith(COMMITTED_ARTIFACT_PREFIX)
        or resolved.suffix != suffix
    ):
        raise SystemExit(
            f"adjudication outputs must be docs/artifacts/gate-d-*{suffix} paths: {path}"
        )
    return resolved


def _write_once(path: Path, value: dict) -> None:
    if path.exists():
        raise SystemExit(f"append-only adjudication output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
