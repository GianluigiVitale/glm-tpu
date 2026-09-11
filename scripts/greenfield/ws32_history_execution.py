"""Bounded history numerical continuation, not a launcher or collector.

The protected parent supplies an authenticated BoundHistoryRuntime and the
existing fleet consensus. Nine actual compiler originals precede admission;
eight WK plus two exact materializer calls precede the fixed two histories.
Every dispatch retains original memory/ownership/vote checks through HistoryCalls.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping

from scripts.greenfield import ws32_history_admission as admission
from scripts.greenfield import ws32_history_protocol as protocol
from scripts.greenfield import ws32_history_runtime as runtime_module
from scripts.greenfield import ws32_history_worker as worker
from scripts.greenfield.prefill_window_acquisition import fleet_step
from scripts.greenfield.ws32_acquisition_journal import Ws32NumericalJournal
from scripts.greenfield.ws32_history_call_evidence import HistoryCalls, MAX_CALLS


def call_schedule() -> tuple[tuple[str, str], ...]:
    """Exact331-call identity for execution and independent original collection."""
    materializers = tuple((f"layer{layer}/{name}", name)
                         for layer in protocol.PRODUCERS
                         for name in ("wk_decode", "wk_promote"))
    steps = tuple((f"history/{s.index}/{s.branch}", s.program) for s in protocol.plan())
    result = (*materializers, ("history/exact_decode", "exact_decode"),
              ("history/exact_promote", "exact_promote"), *steps,
              *((f"history/observer/{b}", "observer") for b in protocol.BRANCHES))
    if len(result) != MAX_CALLS:
        raise ValueError("history fixed call schedule changed")
    return result


class HistoryJournal(Ws32NumericalJournal):
    artifact_kind = "greenfield_ws32_history_numerical_journal_v1"

    def _check_identity(self, identity: Mapping[str, Any]) -> None:
        if (identity.get("protocol") != protocol.PROTOCOL
                or identity.get("profile") != admission.PROFILE
                or identity.get("compile_only") is not False
                or identity.get("diagnostic_only") is not True):
            raise ValueError("history journal requires fixed diagnostic identity")


def execute(*, root: Path, record: dict, repo: Path, mesh: Any,
            bound: runtime_module.BoundHistoryRuntime,
            consensus: Callable[[bool], bool]) -> None:
    """Run only inside the separately guarded/default-off protected parent.

    No cloud publication, runtime initialization, sampling, model promotion or
    cleanup is authorized here. The collector must rederive all original evidence.
    """
    from scripts.greenfield.probe_ws32_prefill_layer import compile_program
    from scripts.greenfield.ws32_history_materializers import capture_wk, capture_exact

    def guarded(name: str, action: Callable[[], Any]) -> Any:
        return fleet_step(name, action, record=record, root=root, consensus=consensus)

    journal, failure = None, None
    try:
        def setup() -> HistoryCalls:
            nonlocal journal
            if (record.get("protocol") != protocol.PROTOCOL
                    or record.get("profile") != admission.PROFILE
                    or record.get("compile_only") is not False
                    or record.get("diagnostic_only") is not True
                    or record.get("programs") != {}
                    or record.get("call_evidence", []) != []
                    or record.get("materializer_originals", {}) != {}
                    or tuple(name for name, _, _ in bound.jobs) != protocol.PROGRAMS
                    or len(bound.layers) != 7 or len(bound.observer_layers) != 7
                    or len(bound.raw_exact) != 4):
                raise ValueError("history continuation identity/program inventory differs")
            if (root.is_symlink() or (root / "call_records").exists()
                    or (root / "call_records").is_symlink()
                    or (root / "materializers").exists()
                    or (root / "materializers").is_symlink()
                    or any(root.glob("*.npz*"))
                    or any((root / f"{name}.{form}").exists()
                           or (root / f"{name}.{form}").is_symlink()
                           for name in protocol.PROGRAMS
                           for form in ("stablehlo.mlir", "optimized_hlo.txt"))):
                raise ValueError("history continuation refuses prior originals")
            admission.registration(repo)
            record.update(planned_call_count=len(call_schedule()),
                          numerical_promotion=False, performance_claim=False,
                          history_execution_complete=False)
            journal = HistoryJournal(root / "compile_journal.jsonl", dict(
                protocol=protocol.PROTOCOL, profile=admission.PROFILE,
                compile_only=False, diagnostic_only=True,
                code_hash=record["code_hash"], launch_rank=record["launch_rank"]))
            return HistoryCalls(root=root, record=record, consensus=consensus,
                journal=journal, local_slots=bound.local_slots,
                budgeter=runtime_module.memory_budget)

        calls = guarded("history/setup", setup)
        for name, fn, values in bound.jobs:
            def compile_one() -> None:
                compiled = compile_program(fn, values, name, root, record, journal=journal)
                journal.inspect(name, (root / f"{name}.stablehlo.mlir").read_text(),
                                (root / f"{name}.optimized_hlo.txt").read_text(),
                                lambda: dict(scope="ORIGINAL_CAPTURE_ONLY_NOT_ADMISSION"))
                calls.programs[name] = compiled
            guarded(f"history/compile/{name}", compile_one)

        def admit() -> None:
            if tuple(calls.programs) != protocol.PROGRAMS:
                raise ValueError("history actual compiler inventory differs")
            for name in protocol.PROGRAMS:
                report = admission.inspect_program(name,
                    (root / f"{name}.stablehlo.mlir").read_text(),
                    (root / f"{name}.optimized_hlo.txt").read_text(),
                    record["programs"][name]["compiled_memory"], repo=repo)
                record["programs"][name]["admission"] = report
                if report.get("passed") is not True:
                    raise ValueError("history actual compiler admission refused")
        calls.phase("history/admission", admit)

        wk = []
        for layer_id in protocol.PRODUCERS:
            layer = bound.layers[layer_id]
            operands = (layer.dsa.wk_bits_local, layer.dsa.wk_scale_local)
            decoded = calls.call(f"layer{layer_id}/wk_decode", "wk_decode", operands,
                preserve=lambda value: capture_wk(root, record, bound.local_slots,
                    layer=layer_id, name="wk_decode", value=value, inputs=operands))
            promoted = calls.call(f"layer{layer_id}/wk_promote", "wk_promote", (decoded,),
                preserve=lambda value: capture_wk(root, record, bound.local_slots,
                    layer=layer_id, name="wk_promote", value=value, inputs=(decoded,)))
            wk.append(promoted)
            del decoded, promoted, operands
        decoded = calls.call("history/exact_decode", "exact_decode", (bound.raw_exact,),
            preserve=lambda value: capture_exact(root, record, bound,
                name="exact_decode", value=value, inputs=(bound.raw_exact,)))
        exact = calls.call("history/exact_promote", "exact_promote", (decoded,),
            preserve=lambda value: capture_exact(root, record, bound,
                name="exact_promote", value=value, inputs=(decoded,)))
        del decoded
        worker.execute_history(calls, mesh=mesh, config=bound.config,
            plan=protocol.plan(), prompt_tokens=bound.prompt_tokens,
            embedding=bound.embedding, layers=bound.layers,
            observer_layers=bound.observer_layers, wk=tuple(wk), exact=exact,
            rope=bound.rope, originals=bound.originals)

        def finish() -> None:
            entries = record["call_evidence"]
            if (tuple((e.get("phase"), e.get("graph")) for e in entries) != call_schedule()
                    or any(e.get("completed") is not True or "original" not in e for e in entries)
                    or record.get("history", {}).get("complete") is not True
                    or record["history"].get("attribution_eligible") is not True):
                raise ValueError("history complete call/reproduction inventory differs")
            record["history_execution_complete"] = True
        calls.phase("history/execution_complete", finish)
    except Exception as exc:
        failure = exc
        record.update(status="DIAGNOSTIC_FAILED", error=f"{type(exc).__name__}: {exc}")
    finally:
        def finalize() -> None:
            if journal is not None:
                journal.close()
                record["compile_journal_sha256"] = sha256(
                    (root / "compile_journal.jsonl").read_bytes()).hexdigest()
        try:
            guarded("history/finalize", finalize)
        except Exception as exc:
            record["finalization_error"] = f"{type(exc).__name__}: {exc}"
            failure = failure or exc
    if failure is not None:
        record["status"] = "DIAGNOSTIC_FAILED"
        record.setdefault("error", f"{type(failure).__name__}: {failure}")
        raise failure
