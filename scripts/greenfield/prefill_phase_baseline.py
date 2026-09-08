"""Staged DB594 phase sampling; no launcher, TPU initialization or promotion.

Reuse the actual completed-window traversal and BudgetedCalls. This is not an
end-to-end prefill benchmark: final assembly consumes BOTH suffix realizations.
The caller must bind retained weights, DB594 graphs/originals, WK and owners.
"""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import gzip
import json
import os
from pathlib import Path
import re
import time
import zlib
from typing import Any, Callable, Iterator

from scripts.greenfield.prefill_completed_window_worker import execute_window
from scripts.greenfield.prefill_window_worker import BudgetedCalls
from scripts.greenfield.ws32_acquisition_journal import Ws32NumericalJournal

PROTOCOL = "ws32-layer6-db594-competitive-phase-baseline-v1"
KERNEL = "ws32_prefill_completed_phase_baseline"

WARMUP = 3
SAMPLES = 10
TRACED = 2
MAX_SECONDS = 180.0
SCOPE = "PHASE_SUM_ESTIMATE_NOT_END_TO_END_MISSING_INDEPENDENT_FINAL_ASSEMBLY"
ORDER = "FOUR_PREFIX_THEN_WIDE_THEN_FOUR_NARROW_THEN_COMPARISON_ASSEMBLY"
COUNTS = dict(
    prepare_prefix=4,
    prefix=4,
    prepare_wide=1,
    candidate=1,
    prepare_narrow=4,
    control=4,
    assemble=1,
)
SEQUENCE = (
    ("prepare_prefix", "prefix") * 4
    + ("prepare_wide", "candidate")
    + ("prepare_narrow", "control") * 4
    + ("assemble",)
)
MAX_WITNESS_CALLS = (WARMUP + SAMPLES + TRACED) * len(SEQUENCE) + 2  # optional WK
MAX_WITNESS_BYTES = 32 << 20  # per host; no repeated full inventories in summaries
MAX_WITNESS_RECORD_BYTES = 8 << 20


class PhaseJournal(Ws32NumericalJournal):
    artifact_kind = "greenfield_ws32_prefill_phase_baseline_journal_v1"

    def _check_identity(self, identity: dict) -> None:
        from scripts.greenfield.prefill_completed_window_admission import PROFILE

        if (
            identity.get("protocol") != PROTOCOL
            or identity.get("profile") != PROFILE
            or identity.get("compile_only") is not False
        ):
            raise ValueError("phase baseline requires distinct journal identity")


def phase_summary(entries: list[dict]) -> dict:
    """Only completed dispatch intervals; no mixed-phase end-to-end assertion."""
    if tuple(e["graph"] for e in entries) != SEQUENCE:
        raise ValueError("phase traversal graph order/counts differ")
    times = dict.fromkeys(COUNTS, 0.0)
    for entry in entries:
        dt = entry.get("completed_call_seconds")
        if (
            entry.get("completed") is not True
            or type(dt) not in (int, float)
            or not 0 <= dt < float("inf")
        ):
            raise ValueError("phase sample is incomplete or has invalid duration")
        times[entry["graph"]] += dt
    prefix = times["prepare_prefix"] + times["prefix"]
    wide = times["prepare_wide"] + times["candidate"]
    narrow = times["prepare_narrow"] + times["control"]
    return dict(
        scope=SCOPE,
        order=ORDER,
        graph_seconds=times,
        shared_prefix_seconds=prefix,
        wide_suffix_seconds=wide,
        narrow_suffix_seconds=narrow,
        wide_path_partial_phase_sum_seconds=prefix + wide,
        narrow_path_partial_phase_sum_seconds=prefix + narrow,
        comparison_only_assembly_seconds=times["assemble"],
        all_completed_call_seconds=sum(times.values()),
        independent_final_assembly_included=False,
        end_to_end_performance_claim=False,
    )


class CompactPhaseCalls(BudgetedCalls):
    """Stream full budget witnesses once, keep only compact per-call pointers.

    The existing budget/owner/pre/post/vote logic is unchanged. Do not append
    hundreds of full live-buffer inventories to the continually rewritten JSON.
    A failed stream write is itself voted before another device call.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.evidence_path = self.root / "phase_calls.jsonl.gz"
        if self.evidence_path.exists():
            raise ValueError("phase witness stream must be new")
        self.samples: list[dict] = []
        self.record["phase_call_index"] = self.samples

    def call(self, phase: str, name: str, values: tuple, *, preserve: Callable) -> Any:
        try:
            return super().call(phase, name, values, preserve=preserve)
        finally:
            # Original partial/completed witness survives execution or postpeak
            # refusal as well. Only one full entry lives in runner.json at once.
            def archive():
                entries = self.record["call_evidence"]
                if len(entries) != 1:
                    raise ValueError("phase stream requires one current call")
                entry = entries[0]
                raw = (
                    json.dumps(
                        entry, sort_keys=True, separators=(",", ":"), allow_nan=False
                    )
                    + "\n"
                ).encode()
                if len(raw) > MAX_WITNESS_RECORD_BYTES:
                    raise ValueError("phase witness exceeds bounded8MiB record")
                member = gzip.compress(raw, mtime=0)
                with self.evidence_path.open("ab") as stream:
                    offset = stream.tell()
                    if (
                        len(self.samples) >= MAX_WITNESS_CALLS
                        or offset + len(member) > MAX_WITNESS_BYTES
                    ):
                        raise ValueError("phase witness stream exceeds storage budget")
                    stream.write(member)
                    stream.flush()
                    os.fsync(stream.fileno())
                compact = {k: entry[k] for k in ("phase", "graph", "completed")}
                if "completed_call_seconds" in entry:
                    compact["completed_call_seconds"] = entry["completed_call_seconds"]
                compact.update(
                    offset=offset,
                    compressed_bytes=len(member),
                    raw_sha256=sha256(raw).hexdigest(),
                )
                self.samples.append(compact)
                entries.clear()

            self.phase(phase + "/archive_call", archive)


def read_call_witnesses(path: Path, index: list[dict]) -> Iterator[dict]:
    """Stream exact gzip members; no overlap, gaps, hidden suffix or zip bombs.

    Callers must exhaust this iterator and separately validate memory/phase
    semantics. This authenticates the compact pointers, not a worker verdict.
    """
    if (
        not index
        or len(index) > MAX_WITNESS_CALLS
        or path.stat().st_size > MAX_WITNESS_BYTES
    ):
        raise ValueError("phase witness inventory/storage budget differs")
    offset = 0
    with path.open("rb") as stream:
        for compact in index:
            length = compact.get("compressed_bytes")
            if (
                type(compact.get("offset")) is not int
                or compact["offset"] != offset
                or type(length) is not int
                or not 0 < length <= MAX_WITNESS_BYTES
            ):
                raise ValueError("phase witness offsets/lengths differ")
            packed = stream.read(length)
            inflater = zlib.decompressobj(wbits=31)
            raw = inflater.decompress(packed, MAX_WITNESS_RECORD_BYTES + 1)
            if (
                len(packed) != length
                or len(raw) > MAX_WITNESS_RECORD_BYTES
                or not inflater.eof
                or inflater.unused_data
                or inflater.unconsumed_tail
                or sha256(raw).hexdigest() != compact.get("raw_sha256")
            ):
                raise ValueError("phase witness member/digest differs")
            entry = json.loads(raw)
            expected = {k: entry[k] for k in ("phase", "graph", "completed")}
            if "completed_call_seconds" in entry:
                expected["completed_call_seconds"] = entry["completed_call_seconds"]
            expected.update(
                offset=offset,
                compressed_bytes=length,
                raw_sha256=sha256(raw).hexdigest(),
            )
            # JSON types matter (True must not stand in for an integer/time).
            if json.dumps(compact, sort_keys=True, allow_nan=False) != json.dumps(
                expected, sort_keys=True, allow_nan=False
            ):
                raise ValueError("phase witness compact fields differ")
            yield entry
            offset += length
        if stream.read(1):
            raise ValueError("phase witness has unindexed trailing bytes")


@contextmanager
def voted_trace(
    calls: Any, path: Path, *, start: Callable, stop: Callable
) -> Iterator[None]:
    """Every host votes entry and exit, including hosts whose peer failed entry."""
    entered = False

    def enter():
        nonlocal entered
        if path.exists():
            raise ValueError("phase trace directory must be new")
        start(str(path))
        entered = True

    try:
        calls.phase("phase_trace/start", enter)
        yield
    finally:

        def leave():
            if entered:
                stop()

        calls.phase("phase_trace/stop", leave)


def run_samples(
    calls: CompactPhaseCalls,
    *,
    values: tuple,
    tiles: tuple,
    verify_component: Callable,
    verify_assembly: Callable,
    trace_start: Callable,
    trace_stop: Callable,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Repeat identical initial state; validation never enters a timed interval.

    Verifiers must bind the first traversal to original DB594 competitive-case
    bytes and later traversals to the same outputs. This staged caller interface
    is not itself that authentication or hardware authorization.
    """
    result = dict(
        scope=SCOPE,
        order=ORDER,
        warmup=WARMUP,
        iterations=SAMPLES,
        trace_iterations=TRACED,
        case="competitive",
        rows=128,
        initial_offset=2553,
        warmup_samples=[],
        wall_samples=[],
        traced_samples=[],
        performance_claim=False,
        complete=False,
    )
    calls.record["phase_baseline"] = result
    started = clock()

    def one(kind, index):
        first = len(calls.samples)
        sample_started = clock()
        # execute_window holds its own temporary generations only for this call;
        # never assign its cache outputs to immutable values across samples.
        execute_window(
            calls,
            case=f"{kind}{index}",
            values=values,
            tiles=tiles,
            capture=verify_component,
            preserve_assembly=verify_assembly,
        )

        def finish():
            # Summary failures must vote here, not skip into trace-stop while
            # healthy peers are still voting sample completion.
            sample = phase_summary(calls.samples[first:])
            sample["whole_traversal_seconds_including_checks"] = (
                clock() - sample_started
            )
            sample["excluded_checks_control_seconds"] = (
                sample["whole_traversal_seconds_including_checks"]
                - sample["all_completed_call_seconds"]
            )
            elapsed = clock() - started
            if (
                not 0 <= sample["excluded_checks_control_seconds"] < float("inf")
                or not 0 <= elapsed <= MAX_SECONDS
            ):
                raise ValueError("phase baseline clock/budget refused")
            result[kind + "_samples"].append(sample)

        calls.phase(f"{kind}{index}/sample_complete", finish)

    for i in range(WARMUP):
        one("warmup", i)
    for i in range(SAMPLES):
        one("wall", i)
    with voted_trace(
        calls, calls.root / "phase_trace", start=trace_start, stop=trace_stop
    ):
        for i in range(TRACED):
            one("traced", i)
    result["complete"] = True
    calls.phase("phase_baseline/complete", lambda: None)
    return result


def run_competitive(
    calls: CompactPhaseCalls, *, weights: Any, wk: Any, mesh: Any, specs: tuple
) -> None:
    """Existing worker continuation: fixed DB594 fixture, not three-case rerun."""
    import jax
    from glm_tpu.greenfield.kernels.reference.rotary import build_rotary_table_host
    from scripts.greenfield import prefill_completed_window_admission as admission
    from scripts.greenfield import prefill_completed_window_assembly as assembly
    from scripts.greenfield import prefill_phase_originals as originals
    from scripts.greenfield.prefill_window_protocol import CAPACITY, host_case
    from scripts.greenfield.probe_ws32_prefill_layer import device_inputs

    def bind():
        if (
            not isinstance(calls, CompactPhaseCalls)
            or not isinstance(calls.journal, PhaseJournal)
            or calls.budgeter is not assembly.memory_budget
            or set(calls.programs) != set((*admission.PROGRAMS, *assembly.PROGRAMS))
            or calls.record.get("protocol") != PROTOCOL
            or calls.record.get("reference_scope") != SCOPE
            or calls.record.get("independent_full_layer_admission") is not False
            or calls.record.get("profile") != admission.PROFILE
            or calls.record.get("compile_only") is not False
            or calls.record.get("performance_claim") is not False
        ):
            raise ValueError("phase baseline lacks distinct protected continuation")
        original = originals.load_capsule()
        originals.bind_originals(calls.record, calls.local_slots, original)
        return original

    original = calls.phase("phase_original_bind", bind)
    host = calls.phase(
        "phase_fixture",
        lambda: host_case(
            "competitive", build_rotary_table_host(CAPACITY, rotary_dim=64, theta=8e6)
        ),
    )
    verifier = calls.phase(
        "phase_verifier", lambda: originals.OriginalVerifier(calls, original, host)
    )
    values = calls.phase(
        "phase_inputs", lambda: device_inputs(host, specs, weights, wk, mesh)
    )
    tiles = calls.phase("phase_tiles", lambda: assembly.place_tiles(mesh))
    calls.phase("phase_inputs_ready", lambda: jax.block_until_ready((values, tiles)))
    run_samples(
        calls,
        values=values,
        tiles=tiles,
        verify_component=verifier.component,
        verify_assembly=verifier.assembly,
        trace_start=jax.profiler.start_trace,
        trace_stop=jax.profiler.stop_trace,
    )
    calls.phase(
        "phase_original_complete", lambda: verifier.finish(WARMUP + SAMPLES + TRACED)
    )


def trace_groups(programs: dict[str, str]) -> dict[str, dict]:
    """Use acquired module names; explicitly disclose shared suffix names."""
    if set(programs) != set(COUNTS):
        raise ValueError("trace requires seven executed phase graphs")
    groups: dict[str, dict] = {}
    for name, text in programs.items():
        match = re.match(r"HloModule ([A-Za-z0-9_.-]+),", text)
        if match is None:
            raise ValueError("missing original HLO module name")
        module = match[1]
        group = groups.setdefault(module, dict(graphs=[], expected_calls_per_core=0))
        group["graphs"].append(name)
        group["expected_calls_per_core"] += COUNTS[name] * TRACED
    for module, group in groups.items():
        group["module_regex"] = (
            r"(?<![A-Za-z0-9_])" + re.escape(module) + r"(?![A-Za-z0-9_])"
        )
        group["mixed_graphs"] = len(group["graphs"]) > 1
    return groups


def aggregate_phase_trace(root: Path, programs: dict[str, str]) -> dict:
    """Reuse fleet parser, refuse absent hosts/cores/calls, retain raw categories."""
    from scripts.analysis.parse_xplane import aggregate_fleet

    reports = {}
    for module, group in trace_groups(programs).items():
        report = aggregate_fleet(root, step_module_re=group["module_regex"])
        if (
            report["n_files"] != 8
            or report["n_cores"] != 64
            or report["steps_per_core"] != group["expected_calls_per_core"]
        ):
            raise ValueError(
                "prefill trace lacks exact eight-host64-core phase coverage"
            )
        reports[module] = dict(selection=group, trace=report)
    return dict(
        scope="TRACED_PHASE_ATTRIBUTION_NOT_PROFILER_FREE_THROUGHPUT",
        programs=reports,
        performance_claim=False,
        suffix_trace_may_mix_b128_and_b32=True,
        category_times_are_fleet_means_not_critical_path=True,
        utilization_claim=False,
        cycle_metric_caveat=(
            "Raw idle_pct/step_cycle_ms/pct_step_cycle include intervening other "
            "graphs and host checks; they are not phase hardware-idle or MXU "
            "utilization. Trace bytes/flops are profiler estimates, not assumed "
            "hardware bandwidth counters. Unclassified time remains in raw reports."
        ),
    )
