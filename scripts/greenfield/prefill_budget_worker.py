"""Missing-budget DSA continuation using existing protected execution machinery.

No CLI, topology initialization, weights, deployment or cleanup implementation.
The outer microbenchmark campaign must authenticate topology/code and collect
originals before this can become protected baseline evidence. Cache/input/token
overhead measurements are separate from this DSA sampling interval.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from math import prod
import re
from time import perf_counter
from typing import Any

import numpy as np

from scripts.greenfield import prefill_budget_probe as probe
from scripts.greenfield.prefill_window_worker import BudgetedCalls, save_arrays
from scripts.greenfield.ws32_acquisition_journal import Ws32NumericalJournal
from glm_tpu.greenfield.validation.ws32_prefill_memory import (
    MEMORY_FIELDS,
    budget_resident_execution,
)

PROFILE = "ws32-dsa-budget-production512-default-paired-v1"
PROGRAMS = tuple(f"dsa_c{capacity}" for capacity, _ in probe.CAPACITIES)
RESERVE_BYTES = 1 << 30
EXPERT_GROUPS = tuple(tuple(range(feature, 32, 4)) for feature in range(4))


class BudgetJournal(Ws32NumericalJournal):
    artifact_kind = "greenfield_ws32_prefill_budget_journal"

    def _check_identity(self, identity):
        if (
            identity.get("protocol") != probe.PROTOCOL
            or identity.get("profile") != PROFILE
            or identity.get("compile_only") is not False
        ):
            raise ValueError("unregistered missing-budget journal identity")


def validate_memory(memory: dict) -> None:
    if (
        set(memory) != set(MEMORY_FIELDS)
        or any(type(v) is not int or v < 0 for v in memory.values())
        or memory["argument_size_in_bytes"] > 64 << 20
        or memory["output_size_in_bytes"] > 2 << 20
        or memory["temp_size_in_bytes"] > 1 << 30
        or memory["generated_code_size_in_bytes"] > 64 << 20
        or memory["alias_size_in_bytes"] != 0
    ):
        raise ValueError("DSA budget compiler allocation outside fixed ceiling")


def memory_budget(census: dict, analyses: dict, *, active_graph: str) -> dict:
    if set(analyses) != set(PROGRAMS) or active_graph not in PROGRAMS:
        raise ValueError("DSA budget requires exactly two resident programs")
    for memory in analyses.values():
        validate_memory(memory)
    return budget_resident_execution(
        census,
        analyses,
        active_graph=active_graph,
        resident_graphs=PROGRAMS,
        required_reserve_bytes=RESERVE_BYTES,
    )


def inspect_program(name: str, stable: str, optimized: str, memory: dict) -> dict:
    """Baseline actual HLO: two candidate leaves, only four expert8 groups.

    Record exact raw hashes; this is not a candidate equivalence certificate.
    Tuple-combined gathers may contain both leaves, with no extra payload.
    Physical slot binding is supplied by the existing topology fleet validator.
    """
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

    if name not in PROGRAMS:
        raise ValueError("unknown DSA budget program")
    validate_memory(memory)
    module = parse_hlo_module(optimized)
    payloads, groups = Counter(), []
    for op in module.instructions:
        if op.opcode in ("infeed", "outfeed", "send", "recv", "send-done", "recv-done"):
            raise ValueError("DSA budget contains host/staged transport")
        if op.opcode == "custom-call" and re.search(
            r"callback|host_transfer|host_compute", op.raw_line, re.IGNORECASE
        ):
            raise ValueError("DSA budget contains host custom call")
        if not op.is_collective:
            continue
        if op.opcode != "all-gather" or op.replica_groups != EXPERT_GROUPS:
            raise ValueError("DSA budget collective leaves expert8 groups")
        if len(op.operand_shapes) != len(op.result_shapes):
            raise ValueError("DSA candidate gather tuple arity differs")
        for source, target in zip(op.operand_shapes, op.result_shapes, strict=True):
            if (
                source.dtype not in ("s32", "f32")
                or source.dtype != target.dtype
                or prod(source.dimensions) != probe.ROWS * probe.TOP_K
                or prod(target.dimensions) != 8 * probe.ROWS * probe.TOP_K
            ):
                raise ValueError("DSA budget contains noncandidate gather payload")
            payloads[source.dtype] += 1
        groups.append([list(group) for group in op.replica_groups])
    if payloads != Counter(s32=1, f32=1):
        raise ValueError("DSA budget must exchange exactly positions and scores")
    return dict(
        passed=True,
        stablehlo_sha256=sha256(stable.encode()).hexdigest(),
        optimized_hlo_sha256=sha256(optimized.encode()).hexdigest(),
        compiled_memory=memory,
        collective_groups=groups,
        exchanged_candidate_leaves=dict(payloads),
        numerical_claim=False,
        model_performance_claim=False,
    )


def prepare(calls: BudgetedCalls, mesh: Any, *, compiler=None) -> dict:
    """All allocation, reference construction and compilation precede deadline.

    Compiler defaults to the existing fsynced raw graph/journal writer. Failure
    votes use BudgetedCalls.phase, and no distributed executable is dispatched.
    """
    import jax
    from scripts.greenfield.probe_ws32_prefill_layer import compile_program

    compiler = compile_program if compiler is None else compiler

    def bind():
        if (
            not isinstance(calls.journal, BudgetJournal)
            or calls.record.get("protocol") != probe.PROTOCOL
            or calls.record.get("profile") != PROFILE
            or calls.record.get("compile_only") is not False
            or calls.programs
        ):
            raise ValueError("DSA budget continuation identity differs")
        calls.budgeter = memory_budget
        calls.record["budget_cases"] = {}

    calls.phase("budget/bind", bind)
    prepared = {}
    for case in probe.cases():

        def fixture():
            inputs = probe.make_dsa_inputs(mesh, case)
            jax.block_until_ready(inputs)
            expected = probe.expected_selection(case.valid_lengths)
            return dict(case=case, inputs=inputs, expected=expected)

        prepared[case.name] = calls.phase(case.name + "/prepare", fixture)

    for name, (capacity, _) in zip(PROGRAMS, probe.CAPACITIES, strict=True):
        first = next(v for v in prepared.values() if v["case"].capacity == capacity)

        def compile_one():
            fn = probe.build_dsa_program(mesh, capacity=capacity)
            compiled = compiler(
                fn,
                first["inputs"],
                name,
                calls.root,
                calls.record,
                journal=calls.journal,
            )
            # Keep journal.compiled -> raw-written -> inspected contiguous;
            # a phase publication between them would overwrite the stage.
            stable = (calls.root / f"{name}.stablehlo.mlir").read_text()
            optimized = (calls.root / f"{name}.optimized_hlo.txt").read_text()
            report = calls.journal.inspect(
                name,
                stable,
                optimized,
                lambda: inspect_program(
                    name,
                    stable,
                    optimized,
                    calls.record["programs"][name]["compiled_memory"],
                ),
            )
            calls.record["programs"][name]["admission"] = report
            return compiled

        calls.programs[name] = calls.phase(name + "/compile_inspect", compile_one)

    # The all-equal-score run exercises ties at each capacity's full valid prefix.
    # Reuse compiled programs; head values are dynamic, not static constants.
    for entry in prepared.values():
        case = entry["case"]
        if case.last_valid_length != case.prompt_length:
            continue

        def tie_fixture():
            inputs = probe.make_dsa_inputs(mesh, case, tied=True)
            jax.block_until_ready(inputs)
            return inputs, probe.expected_selection(case.valid_lengths, tied=True)

        entry["tied"] = calls.phase(case.name + "/tie_prepare", tie_fixture)
    return prepared


def preserve_output(calls: BudgetedCalls, label: str, result: tuple) -> str:
    """Small original arrays per actual local device, not rank-relabelled copies."""
    selected, health = result
    arrays = {}
    for field, value in zip(
        ("positions", "valid_counts", "scores", "health"),
        (*selected, health),
        strict=True,
    ):
        seen = set()
        for shard in value.addressable_shards:
            device = int(shard.device.id)
            if device in seen or device not in calls.local_slots:
                raise ValueError("DSA budget capture has an unauthenticated owner")
            seen.add(device)
            arrays[f"device{device}_{field}"] = np.asarray(shard.data)
        if seen != set(calls.local_slots):
            raise ValueError("DSA budget capture owner coverage differs")
    path = calls.root / f"{label}.npz"
    digest = save_arrays(path, arrays)
    calls.record.setdefault("budget_originals", {})[path.name] = digest
    return digest


def run_samples(calls: BudgetedCalls, prepared: dict, *, clock=perf_counter) -> dict:
    """Two untimed tied checks, then six cases sharing one120s sampling deadline."""

    def bind():
        if (
            set(prepared) != {case.name for case in probe.cases()}
            or set(calls.programs) != set(PROGRAMS)
            or any(prepared[c.name]["case"] != c for c in probe.cases())
        ):
            raise ValueError("DSA budget prepared workload differs")

    calls.phase("budget/sampling_bind", bind)
    for case in probe.cases():
        if case.last_valid_length != case.prompt_length:
            continue
        inputs, expected = prepared[case.name]["tied"]
        output = calls.call(
            case.name + "/tie",
            f"dsa_c{case.capacity}",
            inputs,
            preserve=lambda result: preserve_output(calls, case.name + "_tie", result),
        )
        calls.record.setdefault("budget_ties", {})[case.name] = calls.phase(
            case.name + "/tie_check", lambda: probe.check_output(output, expected)
        )
        del output

    started = calls.phase("budget/sampling_started", clock)
    for case in probe.cases():
        entry = prepared[case.name]
        report = probe.sample_dsa(
            calls,
            f"dsa_c{case.capacity}",
            entry["inputs"],
            entry["expected"],
            case_name=case.name,
            budget_started=started,
            clock=clock,
            preserve=lambda index, output: preserve_output(
                calls, case.name + f"_sample{index}", output
            ),
        )
        calls.record["budget_cases"][case.name] = report
        calls.phase(case.name + "/complete", lambda: None)
    return calls.record["budget_cases"]
