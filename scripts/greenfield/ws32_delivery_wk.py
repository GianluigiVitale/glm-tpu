"""Original completed WK preparation, using the existing voted call machinery.

No launcher or model math. The full worker/collector must bind this phase to
the verified checkpoint and long request. One compact original per call avoids
rewriting the full model's live-buffer census for every later call.
"""

from __future__ import annotations

import gc
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import time

import jax
import numpy as np

from glm_tpu.greenfield.validation.ws32_prefill_memory import budget_resident_execution
from scripts.greenfield import ws32_delivery_runtime as runtime
from scripts.greenfield import ws32_dense_frontier_admission as wk_admission
from scripts.greenfield.ws32_history_call_evidence import HistoryCalls
from scripts.greenfield.ws32_phase_weights import PhaseWeights

ROLES = ("wk_decode", "wk_promote")


def memory_budget(census: Mapping, analyses: Mapping, *, active_graph: str) -> dict:
    if set(analyses) != set(ROLES) or active_graph not in ROLES:
        raise ValueError("delivery WK requires exactly two resident programs")
    for name, analysis in analyses.items():
        wk_admission.validate_memory(name, analysis)
    if any(row["memory_stats"]["bytes_limit"] != runtime.DEVICE_LIMIT
           for row in census["devices"]):
        raise ValueError("delivery WK device limit differs")
    return budget_resident_execution(census, analyses, active_graph=active_graph,
        resident_graphs=ROLES, required_reserve_bytes=runtime.RESERVE)


def preserve_output(value: Any, entry: dict, *, graph: str, slots: Mapping[int, int]) -> None:
    """Keep per-owner hashes before a post-call failure, not another weight copy.

    Original admitted WK math and checkpoint checks establish semantics; these
    hashes/finite checks describe the actual completed boundary, not an offline
    reconstruction of every output. No full WK or checkpoint payload is saved.
    """
    dtype = "bfloat16" if graph == "wk_decode" else "float32"
    if (graph not in ROLES or not isinstance(value, jax.Array)
            or value.is_deleted() or value.shape != (128, 6144)
            or str(value.dtype) != dtype or not value.sharding.is_fully_replicated):
        raise ValueError("delivery WK completed boundary differs")
    rows = []
    entry["output"] = rows
    for shard in value.addressable_shards:
        data = np.asarray(shard.data)
        raw = np.ascontiguousarray(data).view(np.uint8)
        rows.append(dict(device_id=int(shard.device.id),
            slot=slots.get(int(shard.device.id)), shape=list(data.shape),
            dtype=str(data.dtype), bytes=int(data.nbytes),
            sha256=sha256(raw).hexdigest(), finite=bool(np.isfinite(data).all())))
    if (len(rows) != 4 or {r["device_id"] for r in rows} != set(slots)
            or any(r["slot"] is None or not r["finite"] for r in rows)
            or len({r["sha256"] for r in rows}) != 1):
        raise ValueError("delivery WK completed owners/replicas/health differ")


def prepare(
    *, repo: Path, root: Path, owner: PhaseWeights, mesh: Any, journal: Any,
    consensus: Any, local_slots: Mapping[int, int], identity: Mapping[str, Any],
    native_benchmark: bool = False,
) -> dict[str, Any]:
    """Compile two original WK jobs; execute42 voted calls; drop all code roots.

    Parent must not compile prefill or decode companions before this phase.
    Actual all-live census includes the complete raw model and earlier WK outputs.
    Returned record contains no array or executable; owner retains only raw+WK.
    """
    from scripts.greenfield.ws32_compile_originals import compile_program
    from scripts.greenfield.microbench_fp8_matmul import _atomic_json
    started = time.perf_counter()
    # The path is inside the protected run, whose existing uploader/collector
    # must include it before the outer long launch is enabled.
    if type(native_benchmark) is not bool:
        raise ValueError("WK source profile must be an explicit boolean")
    phase_contract = "ws32_native_sampled_request_v1" if native_benchmark else runtime.PROFILE
    record = dict(identity, artifact_kind="ws32_delivery_wk_phase_v1",
                  phase_contract=phase_contract, complete=False,
                  programs={}, model_math_changed=False, performance_claim=False)
    calls = HistoryCalls(root=root, record=record, consensus=consensus,
        journal=journal, local_slots=local_slots, budgeter=memory_budget)
    jobs = ()
    released = False
    created = False
    def release():
        nonlocal released
        calls.programs.clear()
        for _, fn, _ in jobs:
            fn.clear_cache()
        gc.collect()
        released = True
    try:
        # Refuse an existing directory/link BEFORE BudgetedCalls can write its
        # evolving runner.json. Vote the mkdir result without touching a prefix
        # we did not create; a failed phase must never overwrite an older run.
        creation_error = None
        try:
            root.mkdir(exist_ok=False)
            created = True
        except Exception as error:
            creation_error = error
        agreed = consensus(creation_error is None)
        if creation_error is not None:
            raise creation_error
        if not agreed:
            raise RuntimeError("delivery WK peer refused fresh phase directory")
        def preflight():
            if native_benchmark:
                from scripts.greenfield.ws32_native_benchmark_programs import require_source
                require_source(repo)
            else:
                runtime.programs.require_source(repo)
            if owner.phase != "raw" or owner.decode_weights is not None:
                raise ValueError("delivery WK requires raw-only phase")
            if (len(owner.raw_config.full_index_slots) != 21
                    or owner.raw_config.geometry.hidden_size != 6144):
                raise ValueError("delivery WK requires original full model geometry")
            return owner.wk_jobs(mesh)
        jobs = calls.phase("delivery_wk/preflight", preflight)
        for name, fn, values in jobs:
            def compile_one():
                program = compile_program(fn, values, name, root, record, journal=journal)
                calls.programs[name] = program
                stable = (root / f"{name}.stablehlo.mlir").read_text()
                optimized = (root / f"{name}.optimized_hlo.txt").read_text()
                report = journal.inspect(name, stable, optimized,
                    lambda: wk_admission.inspect_program(name, stable, optimized,
                        record["programs"][name]["compiled_memory"]))
                record["programs"][name]["admission"] = report
            calls.phase(f"delivery_wk/compile/{name}", compile_one)
        # Neither original program can execute before both are inspected.
        def call(stage, name, values):
            return calls.call(stage, name, values, preserve=lambda result:
                preserve_output(result, record["call_evidence"][-1], graph=name, slots=local_slots))
        owner.materialize_wk(call, phase=calls.phase)
        calls.phase("delivery_wk/code_released", release)
        def finish():
            expected = [(f"layer{layer}/{name}", name)
                        for layer in owner.raw_config.full_index_slots for name in ROLES]
            actual = [(r["phase"], r["graph"]) for r in record["call_evidence"]]
            if actual != expected or not all(r["completed"] for r in record["call_evidence"]):
                raise ValueError("delivery WK completed call schedule differs")
            record["complete"] = True
            record["phase_wall_seconds"] = time.perf_counter() - started
        calls.phase("delivery_wk/complete", finish)
        return record
    except Exception as error:
        owner.phase = "failed"
        record["complete"] = False
        record["failure"] = f"{type(error).__name__}: {error}"
        try:
            if created:
                _atomic_json(root / "runner.json", record)
        except Exception as publication_error:
            # Preserve the primary cause; compiler/call originals and journal
            # remain independently available to the protected failure collector.
            record["failure_publication_error"] = str(publication_error)
        raise
    finally:
        if not released:
            release()
        # All temporary compiler/argument references die on return. No call to
        # global clear_caches, no model buffer deletion, no missing-code fiction.
