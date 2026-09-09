"""Protected full-model worker continuation; no alternate launcher or model path."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

from glm_tpu.greenfield.validation.ws32_prefill_admission import (
    FROZEN_FIRST_WINDOW_PROFILE, require_acquired_model_source,
    short_budget, short_numerical_identity, validate_short_compiled_memory,
)
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from scripts.greenfield.prefill_window_worker import BudgetedCalls
from scripts.greenfield.ws32_prefill_frontier_worker import execute_first_window, memory_budget


def execute(
    *, args: Any, repo: Any, jax: Any, mesh: Any, physical_mesh: Any,
    config: Any, prompt_tokens: Any, compiled: Mapping[str, Any], graphs: Mapping[str, Any],
    compiled_memory: Mapping[str, Any], identity: Mapping[str, Any], journal: Any,
    weights: Any, wk: Any, rope: Any, consensus: Callable[[bool], bool],
) -> None:
    """Run only admitted original B128 and publish diagnostic, never short SUCCESS.

    Source/graphs/checkpoint are authenticated by the existing main worker. This
    entry repeats source/profile/inventory/memory checks at its dispatch boundary.
    Global original replay and authenticated cleanup remain outer obligations.
    """
    started = time.monotonic()
    root = args.output.parent / f"first_window.rank{args.process_id}"
    # Constructor/write failures join a vote before any successor device call.
    error = None
    calls = None
    try:
        if args.batched_prefill_profile != FROZEN_FIRST_WINDOW_PROFILE:
            raise ValueError("first-window entry requires its diagnostic profile")
        require_acquired_model_source(repo, profile=args.batched_prefill_profile)
        if (set(compiled) != {"prefill_chunk"}
                or set(graphs) != {"exact_materialize", "exact_promote", "prefill_chunk"}
                or set(compiled_memory) != set(graphs)
                or any(not r["passed"] for r in graphs.values())):
            raise ValueError("first-window actual compiled inventory/admission differs")
        for graph, memory in compiled_memory.items():
            validate_short_compiled_memory(graph, memory, profile=args.batched_prefill_profile, repo=repo)
        runtime = tuple(jax.local_devices())
        runtime_ids = {int(d.id) for d in runtime}
        local_slots = {int(d): slot for slot, d in enumerate(physical_mesh.flattened_device_ids)
                       if int(d) in runtime_ids}
        if len(local_slots) != 4 or set(local_slots) != runtime_ids:
            raise ValueError("first-window physical owner binding differs")
        root.mkdir(exist_ok=False)
        record = dict(
            **identity, **short_numerical_identity(profile=args.batched_prefill_profile),
            artifact_kind="greenfield_ws32_first_window_diagnostic",
            code_hash=args.expected_code_hash, launch_process_id=args.process_id,
            jax_process_index=int(jax.process_index()), local_slots={str(k): v for k, v in local_slots.items()},
            graphs=dict(graphs), compiled_memory_analysis=dict(compiled_memory),
            diagnostic_started_monotonic_seconds=started,
            status="DIAGNOSTIC_PARTIAL", numerical_promotion=False, performance_claim=False,
        )
        _atomic_json(args.output, record)

        def bounded_consensus(ok: bool) -> bool:
            within = time.monotonic() - started <= short_budget(args.batched_prefill_profile)
            return consensus(ok and within)

        calls = BudgetedCalls(root=root, record=record, consensus=bounded_consensus,
                              journal=journal, local_slots=local_slots, budgeter=memory_budget)
        calls.programs = dict(compiled)
    except Exception as exc:
        error = exc
    agreed = consensus(error is None)
    if error is not None:
        raise error
    if not agreed:
        raise RuntimeError("first-window peer refused entry")
    try:
        execute_first_window(calls, mesh=mesh, config=config, prompt_tokens=prompt_tokens,
                             weights=weights, wk=wk, rope=rope)
        calls.record["status"] = "DIAGNOSTIC_COMPLETED_NOT_NUMERICAL_PROMOTION"
    except Exception:
        calls.record["status"] = "DIAGNOSTIC_FAILED"
        raise
    finally:
        # A final publication failure must also join the same peer vote.
        try:
            calls.phase("first_window/final_publication", lambda: _atomic_json(args.output, calls.record))
        finally:
            close_error = None
            try:
                journal.close()
            except Exception as exc:
                close_error = exc
            closed = consensus(close_error is None)
            if close_error is not None:
                raise close_error
            if not closed:
                raise RuntimeError("first-window peer refused journal close")
        if calls.record["status"] == "DIAGNOSTIC_COMPLETED_NOT_NUMERICAL_PROMOTION":
            # Terminal host-only publication follows the last matched vote and
            # journal close. Both copies must agree at collection; a partial
            # write or expired deadline cannot look like a completed diagnostic.
            # The300s boundary includes voted journal close, not these terminal
            # host-only writes or the separately bounded EXIT upload.
            ended = time.monotonic()
            if ended - started > short_budget(args.batched_prefill_profile):
                raise RuntimeError("first-window finalization exceeded continuation budget")
            calls.record["diagnostic_closed_monotonic_seconds"] = ended
            _atomic_json(args.output, calls.record)
            _atomic_json(root / "runner.json", calls.record)
