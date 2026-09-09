"""Layer6 compiler acquisition with an explicit fixed numerical continuation.

The existing protected layer campaign owns deployment, leases and publication.
Default acquisition never executes model/WK programs. The numerical continuation
requires exact registered graphs and per-call simultaneous memory admission.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping

from scripts.greenfield import prefill_window_protocol as window
from scripts.greenfield.microbench_fp8_matmul import _atomic_json, _memory_stats
from scripts.greenfield.prefill_layer_hlo import FEATURE, EXPERT
from scripts.greenfield.ws32_acquisition_journal import Ws32AcquisitionJournal

KERNEL = "ws32_prefill_layer_window_acquisition"
PROTOCOL = window.PROTOCOL + "-compile-only"
PROGRAMS = ("wk_decode", "wk_promote", "candidate", "control")
PINS = Path(__file__).resolve().parents[2] / (
    "docs/artifacts/prefill-window-layer6-host-admission-20260908.json"
)
REFERENCE_SCOPE = "B128_WINDOW_VS_FOUR_B32_LAYERS_COMPILED_NOT_EXECUTED"
BOUNDARY_KERNEL = "ws32_prefill_window_boundary_acquisition"
BOUNDARY_PROTOCOL = window.PROTOCOL + "-boundary-compile-only-v1"
BOUNDARY_REFERENCE_SCOPE = "B128_B32_ACTUAL_BOUNDARY_OUTPUTS_COMPILED_NOT_EXECUTED"
COMPLETED_KERNEL = "ws32_prefill_completed_window_acquisition"
COMPLETED_PROTOCOL = "ws32-prefill-layer6-completed-prefix-window128-compile-only-v1"
COMPLETED_PROGRAMS = ("wk_decode", "wk_promote", "prefix", "candidate", "control")
COMPLETED_REFERENCE_SCOPE = (
    "SHARED_COMPLETED_B32_PREFIX_B128_VS_B32_SUFFIX_COMPILED_NOT_EXECUTED"
)


def acquisition_mode(*, boundary: bool = False, completed: bool = False) -> tuple:
    """Explicit disjoint compile-only variants; no numerical admission implied."""
    if boundary and completed:
        raise ValueError("completed prefix and boundary capture modes are exclusive")
    if completed:
        return (
            COMPLETED_KERNEL,
            COMPLETED_PROTOCOL,
            COMPLETED_REFERENCE_SCOPE,
            COMPLETED_PROGRAMS,
        )
    if boundary:
        return BOUNDARY_KERNEL, BOUNDARY_PROTOCOL, BOUNDARY_REFERENCE_SCOPE, PROGRAMS
    return KERNEL, PROTOCOL, REFERENCE_SCOPE, PROGRAMS


def memory_scope(*, completed: bool = False) -> str:
    count = "FIVE" if completed else "FOUR"
    return f"SELECTED_WEIGHTS_AND_{count}_COMPILED_PROGRAMS_NO_NUMERICAL_SCRATCH_OR_OUTPUTS"


MEMORY_KEYS = (
    "argument_size_in_bytes",
    "output_size_in_bytes",
    "alias_size_in_bytes",
    "temp_size_in_bytes",
    "generated_code_size_in_bytes",
)


def is_acquisition_tag(tag: str) -> bool:
    return (
        (
            re.fullmatch(r"greenfield_fp8_" + KERNEL + r"_l6_[a-zA-Z0-9_]+", tag)
            is not None
        )
        or is_boundary_tag(tag)
        or is_completed_tag(tag)
    )


def is_completed_tag(tag: str) -> bool:
    return (
        re.fullmatch(r"greenfield_fp8_" + COMPLETED_KERNEL + r"_l6_[a-zA-Z0-9_]+", tag)
        is not None
    )


def is_boundary_tag(tag: str) -> bool:
    return (
        re.fullmatch(r"greenfield_fp8_" + BOUNDARY_KERNEL + r"_l6_[a-zA-Z0-9_]+", tag)
        is not None
    )


def is_numerical_tag(tag: str) -> bool:
    return (
        re.fullmatch(r"greenfield_fp8_" + window.KERNEL + r"_l6_[a-zA-Z0-9_]+", tag)
        is not None
    )


def is_boundary_diagnostic_tag(tag: str) -> bool:
    from scripts.greenfield.prefill_window_boundary_worker import KERNEL

    return (
        re.fullmatch(r"greenfield_fp8_" + KERNEL + r"_l6_[a-zA-Z0-9_]+", tag)
        is not None
    )


def is_window_tag(tag: str) -> bool:
    return (
        is_acquisition_tag(tag)
        or is_numerical_tag(tag)
        or is_boundary_diagnostic_tag(tag)
        or is_completed_numerical_tag(tag)
        or is_phase_baseline_tag(tag)
    )


def is_completed_numerical_tag(tag: str) -> bool:
    from scripts.greenfield.prefill_completed_window_protocol import KERNEL

    return (
        re.fullmatch(r"greenfield_fp8_" + KERNEL + r"_l6_[a-zA-Z0-9_]+", tag)
        is not None
    )


def is_phase_baseline_tag(tag: str) -> bool:
    from scripts.greenfield.prefill_phase_baseline import KERNEL
    from scripts.greenfield.prefill_phase_variant import PAIRED_KERNEL

    return (
        re.fullmatch(
            r"greenfield_fp8_(?:" + KERNEL + "|" + PAIRED_KERNEL + r")_l6_[a-zA-Z0-9_]+",
            tag,
        )
        is not None
    )


class WindowJournal(Ws32AcquisitionJournal):
    artifact_kind = "greenfield_ws32_layer_window_acquisition_journal"

    def _check_identity(self, identity: dict[str, Any]) -> None:
        if (
            identity.get("protocol") != PROTOCOL
            or identity.get("compile_only") is not True
        ):
            raise ValueError("window journal requires its compile-only protocol")


class WindowBoundaryJournal(Ws32AcquisitionJournal):
    artifact_kind = "greenfield_ws32_window_boundary_acquisition_journal_v1"

    def _check_identity(self, identity: dict[str, Any]) -> None:
        if (
            identity.get("protocol") != BOUNDARY_PROTOCOL
            or identity.get("compile_only") is not True
        ):
            raise ValueError("boundary journal requires its compile-only protocol")

    def output_schema(self, schema: dict[str, Any]) -> None:
        self._write("compiler_output_schema", schema=schema)


class CompletedWindowJournal(Ws32AcquisitionJournal):
    artifact_kind = "greenfield_ws32_completed_window_acquisition_journal_v1"

    def _check_identity(self, identity: dict[str, Any]) -> None:
        if (
            identity.get("protocol") != COMPLETED_PROTOCOL
            or identity.get("compile_only") is not True
        ):
            raise ValueError("completed window requires its compile-only protocol")


def fleet_step(
    name: str,
    action: Callable[[], Any],
    *,
    record: dict[str, Any],
    root: Path,
    consensus: Callable[[bool], bool],
) -> Any:
    """Convert local validation/load/compile/log errors into a matched fleet vote.

    An action must not contain a distributed executable dispatch. Compilation
    and local transfer can fail; peers must vote before the next phase begins.
    """
    start = time.monotonic()
    value, failure = None, None
    try:
        record.setdefault("acquisition_phases", {})[name] = {"status": "RUNNING"}
        _atomic_json(root / "runner.json", record)
        value = action()
    except Exception as exc:
        failure = exc
    record["acquisition_phases"][name] = dict(
        status="FAILED" if failure else "COMPLETE",
        seconds=time.monotonic() - start,
        error=None if failure is None else f"{type(failure).__name__}: {failure}",
    )
    try:
        _atomic_json(root / "runner.json", record)
    except Exception as exc:
        failure = failure or exc
    if not consensus(failure is None):
        if failure is not None:
            raise failure
        raise RuntimeError(f"window acquisition peer failed: {name}")
    if failure is not None:  # A malformed consensus must never suppress a local error.
        raise failure
    return value


def inspect_graph(hlo: str) -> dict[str, Any]:
    """Record actual operations, with no exact-inventory or numerical approval."""
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

    module = parse_hlo_module(hlo)
    collectives = [op for op in module.instructions if op.is_collective]
    calls = [op for op in module.instructions if op.opcode == "custom-call"]
    targets = Counter()
    for op in calls:
        match = re.search(r'custom_call_target="([^"]+)"', op.raw_line)
        targets[match[1] if match else "<missing>"] += 1
    return dict(
        profile_registered=False,
        numerical_execution_authorized=False,
        structural_observations=dict(
            physical_groups_local=all(
                op.replica_groups in (FEATURE, EXPERT) for op in collectives
            ),
            no_host_transport=not any(
                op.opcode in ("infeed", "outfeed", "send", "recv")
                for op in module.instructions
            ),
            no_full_weight_expansion=not any(
                shape.dtype in ("bf16", "f32")
                and shape.element_count >= 32 * 2048 * 1536
                for op in module.instructions
                for shape in op.result_shapes
            ),
        ),
        collectives=[op.to_dict() for op in collectives],
        custom_calls=[op.to_dict() for op in calls],
        custom_call_targets=dict(sorted(targets.items())),
    )


def acquire_programs(
    programs: tuple[tuple[str, Any, tuple[Any, ...]], ...],
    *,
    root: Path,
    record: dict[str, Any],
    consensus: Callable[[bool], bool],
    compiler: Callable[..., Any],
    inspector: Callable[[str], dict[str, Any]] = inspect_graph,
    numerical_context: Mapping[str, Any] | None = None,
    capture_boundaries: bool = False,
    boundary_diagnostic: bool = False,
    completed_window: bool = False,
    completed_numerical: bool = False,
    phase_baseline: bool = False,
) -> tuple[Any, ...]:
    """Preserve graphs/memory before inspection; default mode NEVER calls them.

    Acquisition defers parser refusal until all graphs have been collected.
    Numerical continuation refuses immediately before any executable dispatch.
    A real lowering/compilation failure votes and stops before further work.
    Returning executables keeps the entire mode's set resident for its snapshot.
    """
    numerical = numerical_context is not None
    _, protocol, _, names = acquisition_mode(
        boundary=capture_boundaries, completed=completed_window
    )
    if phase_baseline and (
        not completed_window
        or not numerical
        or completed_numerical
        or boundary_diagnostic
    ):
        raise ValueError("phase baseline requires distinct completed numerical context")
    if completed_numerical and not (completed_window and numerical):
        raise ValueError("completed numerical requires explicit completed context")
    if completed_window and (
        (numerical and not (completed_numerical or phase_baseline))
        or boundary_diagnostic
    ):
        raise ValueError(
            "completed window is compile-only; numerical continuation unavailable"
        )
    if boundary_diagnostic and not (capture_boundaries and numerical):
        raise ValueError(
            "boundary diagnostic requires explicit captured numerical context"
        )
    if capture_boundaries and numerical and not boundary_diagnostic:
        raise ValueError("boundary acquisition cannot execute numerical continuation")
    numerical_protocol = window.PROTOCOL
    if phase_baseline:
        from scripts.greenfield import prefill_phase_baseline as phase

        from scripts.greenfield.prefill_phase_variant import for_record

        variant = fleet_step(
            "phase_variant", lambda: for_record(record),
            record=record, root=root, consensus=consensus,
        )
        numerical_protocol = variant.protocol
    if completed_numerical:
        from scripts.greenfield import prefill_completed_window_worker as completed

        numerical_protocol = completed.protocol.PROTOCOL
    if boundary_diagnostic:
        from scripts.greenfield import prefill_window_boundary_worker as boundary

        numerical_protocol = boundary.PROTOCOL
    if tuple(p[0] for p in programs) != names or record.get("protocol") != (
        numerical_protocol if numerical else protocol
    ):
        raise ValueError("window acquisition program/protocol inventory differs")
    if numerical:
        from scripts.greenfield.prefill_window_worker import WindowNumericalJournal
        from scripts.greenfield import prefill_window_admission as admission

        if completed_numerical or phase_baseline:
            from scripts.greenfield import (
                prefill_completed_window_admission as admission,
            )
            if phase_baseline:
                admission = variant.admission

        if boundary_diagnostic:
            from scripts.greenfield import (
                prefill_window_boundary_admission as admission,
            )
    journal_type = (
        phase.PhaseJournal
        if phase_baseline
        else (
            (
                completed.CompletedJournal
                if completed_numerical
                else (
                    boundary.BoundaryJournal
                    if boundary_diagnostic
                    else WindowNumericalJournal
                )
            )
            if numerical
            else (
                CompletedWindowJournal
                if completed_window
                else WindowBoundaryJournal if capture_boundaries else WindowJournal
            )
        )
    )
    identity = dict(
        protocol=numerical_protocol if numerical else protocol,
        compile_only=not numerical,
        code_hash=record["code_hash"],
        launch_rank=record["launch_rank"],
    )
    if numerical:
        identity["profile"] = admission.PROFILE
    journal = fleet_step(
        "journal",
        lambda: journal_type(
            root / "compile_journal.jsonl",
            identity,
        ),
        record=record,
        root=root,
        consensus=consensus,
    )
    compiled = []
    try:
        if numerical:
            fleet_step(
                "prepared_journal",
                lambda: journal.phase("prepare", passed=True),
                record=record,
                root=root,
                consensus=consensus,
            )
        for name, fn, values in programs:
            graph = fleet_step(
                f"compile_{name}",
                lambda: compiler(fn, values, name, root, record, journal=journal),
                record=record,
                root=root,
                consensus=consensus,
            )
            compiled.append(graph)

            # compile_program already fsynced memory before fallible text inspection.
            def inspect() -> None:
                stable = (root / f"{name}.stablehlo.mlir").read_text()
                hlo = (root / f"{name}.optimized_hlo.txt").read_text()
                schema_options = {}
                if boundary_diagnostic and name in ("candidate", "control"):
                    from scripts.greenfield.prefill_window_boundary import (
                        compiler_output_schema,
                    )

                    schema = compiler_output_schema(graph.out_info, name=name)
                    record["programs"][name]["compiler_output_schema"] = schema
                    schema_options["output_schema"] = schema
                try:
                    report = journal.inspect(
                        name,
                        stable,
                        hlo,
                        lambda: (
                            admission.inspect_program(
                                name,
                                stable,
                                hlo,
                                record["programs"][name]["compiled_memory"],
                                **schema_options,
                            )
                            if numerical
                            else inspector(hlo)
                        ),
                    )
                    record["programs"][name][
                        "admission" if numerical else "inventory"
                    ] = report
                    if numerical:
                        journal.phase("compile/" + name, passed=True)
                except Exception as exc:
                    record["programs"][name][
                        "inspection_error"
                    ] = f"{type(exc).__name__}: {exc}"
                    if numerical:
                        raise

            fleet_step(
                f"inspect_{name}",
                inspect,
                record=record,
                root=root,
                consensus=consensus,
            )
            if (
                capture_boundaries
                and not boundary_diagnostic
                and name in ("candidate", "control")
            ):

                def preserve_schema() -> None:
                    from scripts.greenfield.prefill_window_boundary import (
                        compiler_output_schema,
                        output_schema_error,
                    )

                    schema = compiler_output_schema(graph.out_info, name=name)
                    journal.output_schema(schema)
                    record["programs"][name]["compiler_output_schema"] = schema
                    # Persist actual metadata BEFORE validation, then collect
                    # the remaining graph even if this envelope is malformed.
                    error = output_schema_error(schema, name=name)
                    if error is not None:
                        record["programs"][name]["output_schema_error"] = error

                fleet_step(
                    f"output_schema_{name}",
                    preserve_schema,
                    record=record,
                    root=root,
                    consensus=consensus,
                )
        if numerical:
            # Crucially AFTER all compilation: preserve DB590's outer stack.
            # This continuation executes, whereas the default acquisition never does.
            from scripts.greenfield.prefill_window_worker import execute_numerical

            if completed_numerical or phase_baseline:
                from scripts.greenfield import prefill_completed_window_assembly

                compiled.extend(
                    prefill_completed_window_assembly.compile_programs(
                        mesh=numerical_context["mesh"],
                        root=root,
                        record=record,
                        journal=journal,
                        consensus=consensus,
                        compiler=compiler,
                    )
                )
            execute_numerical(
                **numerical_context,
                record=record,
                consensus=consensus,
                compiled=tuple(compiled),
                journal=journal,
                **({"boundary_diagnostic": True} if boundary_diagnostic else {}),
                **({"completed_numerical": True} if completed_numerical else {}),
                **({"phase_baseline": True} if phase_baseline else {}),
            )
        return tuple(compiled)
    finally:
        if numerical or capture_boundaries or completed_window:

            def close_journal() -> None:
                try:
                    journal.close()
                finally:
                    if numerical:
                        record["compile_journal_sha256"] = sha256(
                            (root / "compile_journal.jsonl").read_bytes()
                        ).hexdigest()
                        _atomic_json(root / "runner.json", record)

            fleet_step(
                "close_compile_journal",
                close_journal,
                record=record,
                root=root,
                consensus=consensus,
            )
        else:
            journal.close()


def prepare_programs(
    *,
    mesh: Any,
    config: Any,
    weights: Any,
    capture_boundaries: bool = False,
    completed_window: bool = False,
    paired_position_sort: bool = False,
    expert_panels: bool = False,
) -> tuple[tuple[str, Any, tuple[Any, ...]], ...]:
    """Shared actual programs/abstract inputs; no model or WK execution."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from glm_tpu.greenfield.kernels.reference.rotary import build_rotary_table_host
    from scripts.greenfield.prefill_layer_evidence import INPUT_FIELDS
    from scripts.greenfield.probe_ws32_prefill_layer import (
        input_specs,
        build_wk_programs,
    )

    acquisition_mode(boundary=capture_boundaries, completed=completed_window)
    if type(expert_panels) is not bool or (expert_panels and not completed_window):
        raise ValueError("expert panels require explicit completed suffix")
    if type(paired_position_sort) is not bool or (
        paired_position_sort and not completed_window
    ):
        raise ValueError("paired position sort requires explicit completed prefix")

    def prepare() -> tuple[tuple[str, Any, tuple[Any, ...]], ...]:
        if (
            config.context_capacity != window.CAPACITY
            or weights.dsa is None
            or weights.moe is None
            or weights.dense is not None
        ):
            raise ValueError("window acquisition is fixed full-indexer/MoE layer6")
        decode, promote = build_wk_programs(
            mesh,
            weights.dsa.wk_bits_local.sharding.spec,
            weights.dsa.wk_scale_local.sharding.spec,
            contract=config.dsa_contract,
        )
        # Separate BF16 decode and FP32 promote graphs still define the later
        # numerical WK boundary. Here their outputs are shapes, not computed data.
        decoded = jax.ShapeDtypeStruct(
            (128, 6144), jnp.bfloat16, sharding=NamedSharding(mesh, P())
        )
        wk = jax.ShapeDtypeStruct(
            (128, 6144), jnp.float32, sharding=NamedSharding(mesh, P())
        )
        specs = input_specs(weights, wk)
        host = window.host_case(
            "competitive",
            build_rotary_table_host(window.CAPACITY, rotary_dim=64, theta=8e6),
        )

        def values_for(rows: int) -> tuple[Any, ...]:
            fields = []
            for name, index in zip(INPUT_FIELDS, (*range(11), 18, 19), strict=True):
                value = host[name]
                shape = list(value.shape)
                if index in (0, 1, 5, 6, 7, 19):
                    shape[0] = rows
                elif index == 18:
                    shape[2] = rows
                fields.append(
                    jax.ShapeDtypeStruct(
                        tuple(shape),
                        value.dtype,
                        sharding=NamedSharding(mesh, specs[index]),
                    )
                )
            return (
                *fields[:11],
                weights.qkv_a,
                weights.attention,
                weights.dsa,
                wk,
                weights.post_attention_norm_weight_local,
                weights.dense,
                weights.moe,
                *fields[11:],
            )

        wk_programs = (
            (
                "wk_decode",
                decode,
                (weights.dsa.wk_bits_local, weights.dsa.wk_scale_local),
            ),
            ("wk_promote", promote, (decoded,)),
        )
        if completed_window:
            from scripts.greenfield.prefill_completed_window import (
                build_completed_window_programs,
            )

            prefix, suffix = build_completed_window_programs(
                mesh,
                specs,
                full_indexer=True,
                sparse_mlp=True,
                key_tile=window.KEY_TILE,
                dsa_contract=config.dsa_contract,
                attention_contract=config.attention_contract,
                moe_contract=config.moe_contract,
                rms_norm_epsilon=config.rms_norm_epsilon,
                paired_position_sort=paired_position_sort,
                expert_panels=expert_panels,
            )

            def suffix_values(rows: int) -> tuple:
                def abstract(shape: tuple, dtype: Any, spec: Any) -> Any:
                    return jax.ShapeDtypeStruct(
                        shape, dtype, sharding=NamedSharding(mesh, spec)
                    )

                return (
                    abstract((rows, 6144), jnp.bfloat16, P(None, "feature")),
                    abstract((rows,), jnp.bool_, P()),
                    weights.dense,
                    weights.moe,
                    abstract((8, 4, rows), jnp.bool_, P("expert", "feature", None)),
                )

            return (
                *wk_programs,
                ("prefix", prefix, values_for(window.CONTROL_ROWS)),
                ("candidate", suffix, suffix_values(window.ROWS)),
                ("control", suffix, suffix_values(window.CONTROL_ROWS)),
            )

        builder = window.build_programs
        if capture_boundaries:
            from scripts.greenfield.prefill_window_boundary import (
                build_boundary_programs,
            )

            builder = build_boundary_programs
        wide, small = builder(
            mesh,
            specs,
            dsa_contract=config.dsa_contract,
            attention_contract=config.attention_contract,
            moe_contract=config.moe_contract,
            rms_norm_epsilon=config.rms_norm_epsilon,
        )
        return (
            *wk_programs,
            ("candidate", wide, values_for(window.ROWS)),
            ("control", small, values_for(window.CONTROL_ROWS)),
        )

    return prepare()


def execute_acquisition(
    *,
    args: Any,
    record: dict[str, Any],
    mesh: Any,
    config: Any,
    weights: Any,
    consensus: Callable[[bool], bool],
    local_slots: Mapping[int, int] | None = None,
    capture_boundaries: bool = False,
    boundary_diagnostic: bool = False,
    completed_window: bool = False,
    completed_numerical: bool = False,
    phase_baseline: bool = False,
) -> None:
    """Original compile stack; default compile-only, explicit numerical continuation."""
    import jax
    from scripts.greenfield.probe_ws32_prefill_layer import compile_program

    context = None
    _, _, _, names = acquisition_mode(
        boundary=capture_boundaries, completed=completed_window
    )
    if phase_baseline and (
        not completed_window
        or local_slots is None
        or completed_numerical
        or boundary_diagnostic
    ):
        raise ValueError("phase baseline requires distinct completed owners")
    if completed_numerical and not (completed_window and local_slots is not None):
        raise ValueError("completed numerical requires explicit completed owners")
    if completed_window and (
        (local_slots is not None and not (completed_numerical or phase_baseline))
        or boundary_diagnostic
    ):
        raise ValueError("completed window cannot enter numerical continuation")
    if boundary_diagnostic and not (capture_boundaries and local_slots is not None):
        raise ValueError("boundary diagnostic requires captured numerical owners")
    if capture_boundaries and local_slots is not None and not boundary_diagnostic:
        raise ValueError("boundary acquisition cannot execute numerical continuation")
    if local_slots is not None:
        from scripts.greenfield import prefill_window_admission as admission

        numerical_protocol = window.PROTOCOL
        if phase_baseline:
            from scripts.greenfield import (
                prefill_completed_window_admission as admission,
            )
            from scripts.greenfield import prefill_phase_baseline as phase

            from scripts.greenfield.prefill_phase_variant import for_record

            variant = fleet_step(
                "phase_variant", lambda: for_record(record),
                record=record, root=args.output_dir, consensus=consensus,
            )
            admission = variant.admission
            numerical_protocol = variant.protocol
            record.update(
                reference_scope=phase.SCOPE, independent_full_layer_admission=False
            )
        if completed_numerical:
            from scripts.greenfield import (
                prefill_completed_window_admission as admission,
            )
            from scripts.greenfield import prefill_completed_window_protocol

            numerical_protocol = prefill_completed_window_protocol.PROTOCOL
            record.update(
                reference_scope=prefill_completed_window_protocol.REFERENCE_SCOPE,
                independent_full_layer_admission=False,
            )
        if boundary_diagnostic:
            from scripts.greenfield import (
                prefill_window_boundary_admission as admission,
            )
            from scripts.greenfield.prefill_window_boundary_worker import (
                PROTOCOL as numerical_protocol,
            )

        record.update(
            protocol=numerical_protocol,
            profile=admission.PROFILE,
            compile_only=False,
            iterations=0,
            performance_claim=False,
            cases={},
            programs={},
        )
        context = dict(
            args=args,
            mesh=mesh,
            config=config,
            weights=weights,
            local_slots=local_slots,
        )
    programs = fleet_step(
        "prepare",
        lambda: prepare_programs(
            mesh=mesh,
            config=config,
            weights=weights,
            capture_boundaries=capture_boundaries,
            completed_window=completed_window,
            paired_position_sort=variant.paired_position_sort if phase_baseline else False,
        ),
        record=record,
        root=args.output_dir,
        consensus=consensus,
    )
    compiled = acquire_programs(
        programs,
        root=args.output_dir,
        record=record,
        consensus=consensus,
        compiler=compile_program,
        numerical_context=context,
        capture_boundaries=capture_boundaries,
        boundary_diagnostic=boundary_diagnostic,
        completed_window=completed_window,
        completed_numerical=completed_numerical,
        phase_baseline=phase_baseline,
    )
    if context is not None:

        def finish_numerical() -> None:
            record["hlo"] = dict(
                sha256=record["programs"]["candidate"]["optimized_hlo_sha256"],
                contract=dict(passed=True, profile=admission.PROFILE),
            )
            _atomic_json(args.output_dir / "runner.json", record)

        fleet_step(
            "numerical_snapshot",
            finish_numerical,
            record=record,
            root=args.output_dir,
            consensus=consensus,
        )
        return

    def finish() -> None:
        record["device_memory_stats_including_reference"] = [
            dict(device_id=int(d.id), stats=_memory_stats(d))
            for d in jax.local_devices()
        ]
        record["resident_programs_at_snapshot"] = list(names)
        record["model_executable_calls"] = 0
        record["wk_executable_calls"] = 0
        record["memory_scope"] = memory_scope(completed=completed_window)
        record["numerical_execution_authorized"] = False
        record["compile_journal_sha256"] = sha256(
            (args.output_dir / "compile_journal.jsonl").read_bytes()
        ).hexdigest()
        record["hlo"] = dict(
            sha256=record["programs"]["candidate"]["optimized_hlo_sha256"],
            contract=dict(
                passed=True, scope="COMPILER_EVIDENCE_PRESENT_NOT_EXECUTION_ADMISSION"
            ),
        )

    fleet_step(
        "snapshot", finish, record=record, root=args.output_dir, consensus=consensus
    )
    assert len(compiled) == len(names)  # Keep every executable live through snapshot.


def validate_workers(
    records: list[dict[str, Any]],
    *,
    pin: str,
    pins: dict[str, Any],
    ledger: dict[int, Any],
    order: tuple[int, ...],
    capture_boundaries: bool = False,
    completed_window: bool = False,
) -> None:
    """Window-specific classification plus original32-owner selected-byte binding.

    Caller already verifies eight unique hosts/processes, mesh order and the
    four graph hashes on every host. These assertions add no execution authority.
    """
    from scripts.greenfield.probe_ws32_prefill_moe import (
        FLEET_SHA,
        MESH_SHA,
        TOPOLOGY_SHA,
    )

    _, protocol, reference_scope, names = acquisition_mode(
        boundary=capture_boundaries, completed=completed_window
    )
    slots = []
    for record in records:
        exact = dict(
            status="SUCCESS",
            protocol=protocol,
            layer=6,
            code_hash=pin,
            selected_layer_ids=[6],
            rows=128,
            control_rows=32,
            context_capacity=4096,
            key_tile=512,
            iterations=0,
            latency=None,
            reference_scope=reference_scope,
            state_scope="REAL_WEIGHTS_SYNTHETIC_PREFIX_AND_ACTIVATIONS",
            integrity_scope="selected_layer_tensors_only_not_complete_checkpoint",
            checkpoint_pins=pins,
            payload_bytes_per_chip=window.PAYLOAD_BYTES_PER_CHIP,
            mesh_sha256=MESH_SHA,
            topology_sha256=TOPOLOGY_SHA,
            topology_fleet_sha256=FLEET_SHA,
            versions={"jax": "0.10.1", "libtpu": "0.0.41"},
            model_executable_calls=0,
            wk_executable_calls=0,
            resident_programs_at_snapshot=list(names),
            memory_scope=memory_scope(completed=completed_window),
            cases={},
        )
        if (
            any(record.get(k) != v for k, v in exact.items())
            or any(
                record.get(k) is not v
                for k, v in dict(
                    compile_only=True,
                    numerical_execution_authorized=False,
                    admission_only=False,
                    diagnostic_only=True,
                    performance_claim=False,
                ).items()
            )
            or set(record["programs"]) != set(names)
        ):
            raise ValueError("window acquisition scope/provenance differs")
        if not all(
            type(record.get(k)) is int and record[k] > 0 for k in ("pid", "start_ticks")
        ) or not record.get("boot_id"):
            raise ValueError("window acquisition process identity missing")
        local = record["local_device_slots"]
        if len(local) != 4 or len({s["device_slot"] for s in local}) != 4:
            raise ValueError("window acquisition needs four distinct local owners")
        for s in local:
            slot = s["device_slot"]
            if (
                type(slot) is not int
                or not 0 <= slot < 32
                or not (
                    s["device_id"] == order[slot]
                    and s["observed_selected_tensor_sha256"] == ledger[slot]["selected"]
                    and s["expected_full_file_sha256_not_verified"]
                    == ledger[slot]["full_sha256"]
                    and s["selected_payload_bytes"] == window.PAYLOAD_BYTES_PER_CHIP
                )
            ):
                raise ValueError("window selected bytes/physical owner differ")
            slots.append(slot)
        for program in record["programs"].values():
            memory = program["compiled_memory"]
            if set(memory) != set(MEMORY_KEYS) or any(
                type(v) is not int or v < 0 for v in memory.values()
            ):
                raise ValueError("window compiler allocation missing/invalid")
            if memory["alias_size_in_bytes"] > min(
                memory["argument_size_in_bytes"], memory["output_size_in_bytes"]
            ):
                raise ValueError("window compiler alias accounting invalid")
        if capture_boundaries:
            from scripts.greenfield.prefill_window_boundary import (
                validate_schema_record,
            )

            for name in ("candidate", "control"):
                validate_schema_record(record["programs"][name], name=name)
        stats = record["device_memory_stats_including_reference"]
        if (
            len(stats) != 4
            or {s["device_id"] for s in stats} != {s["device_id"] for s in local}
            or any(
                not 0
                < s["stats"]["bytes_in_use"]
                <= s["stats"]["peak_bytes_in_use"]
                < s["stats"]["bytes_limit"]
                for s in stats
            )
        ):
            raise ValueError("window compile residency HBM/headroom missing")
        if record.get("hlo") != dict(
            sha256=record["programs"]["candidate"]["optimized_hlo_sha256"],
            contract=dict(
                passed=True, scope="COMPILER_EVIDENCE_PRESENT_NOT_EXECUTION_ADMISSION"
            ),
        ):
            raise ValueError("window acquisition falsely classified as HLO admission")
    if len(slots) != 32 or set(slots) != set(range(32)):
        raise ValueError("window acquisition does not cover32 distinct owners")


def validate_files(root: Path, record: dict[str, Any]) -> None:
    """Replay saved graphs and fsynced compile records, not a claimed HLO verdict."""
    protocol = record.get("protocol")
    if protocol not in (PROTOCOL, BOUNDARY_PROTOCOL, COMPLETED_PROTOCOL):
        raise ValueError("unknown window acquisition protocol")
    boundary = protocol == BOUNDARY_PROTOCOL
    completed = protocol == COMPLETED_PROTOCOL
    _, _, _, names = acquisition_mode(boundary=boundary, completed=completed)
    raw = (root / "compile_journal.jsonl").read_bytes()
    if sha256(raw).hexdigest() != record["compile_journal_sha256"]:
        raise ValueError("window acquisition journal bytes differ")
    journal = [json.loads(line) for line in raw.splitlines()]
    if not journal or journal[0].get("identity") != dict(
        protocol=protocol,
        compile_only=True,
        code_hash=record["code_hash"],
        launch_rank=record["launch_rank"],
    ):
        raise ValueError("window acquisition journal identity differs")
    expected_stages = [(None, "identity")]
    for name in names:
        p = record["programs"][name]
        for form, key in (
            ("stablehlo.mlir", "stablehlo_sha256"),
            ("optimized_hlo.txt", "optimized_hlo_sha256"),
        ):
            if sha256((root / f"{name}.{form}").read_bytes()).hexdigest() != p[key]:
                raise ValueError("window original graph bytes differ")
        try:
            replay = inspect_graph((root / f"{name}.optimized_hlo.txt").read_text())
        except Exception as exc:
            if (
                p.get("inspection_error") != f"{type(exc).__name__}: {exc}"
                or "inventory" in p
            ):
                raise ValueError("window original parser failure differs") from exc
        else:
            if (
                json.loads(json.dumps(replay)) != p.get("inventory")
                or "inspection_error" in p
            ):
                raise ValueError("window original graph inventory differs")
        stages = [row for row in journal if row["graph"] == name]
        expected_stages.extend(
            (name, stage)
            for stage in (
                "lower_compile_started",
                "compiled",
                "raw_written",
                "inspection_failed" if "inspection_error" in p else "inspected",
            )
        )
        schema_stage = boundary and name in ("candidate", "control")
        if schema_stage:
            from scripts.greenfield.prefill_window_boundary import (
                validate_schema_record,
            )

            schema = p["compiler_output_schema"]
            validate_schema_record(p, name=name)
            expected_stages.append((name, "compiler_output_schema"))
            if len(stages) != 5 or stages[4].get("schema") != schema:
                raise ValueError("boundary compiler output schema journal differs")
        if (
            len(stages) != (5 if schema_stage else 4)
            or stages[1].get("compiled_memory") != p["compiled_memory"]
            or stages[1].get("seconds") != p["compile_seconds"]
        ):
            raise ValueError("window compile journal memory/time binding differs")
        if any(
            stages[2].get(k) != p[k]
            for k in ("stablehlo_sha256", "optimized_hlo_sha256")
        ):
            raise ValueError("window compile journal raw graph binding differs")
        if "inventory" in p and stages[3].get("report") != p["inventory"]:
            raise ValueError("window journal inspection differs")
    if [(row["graph"], row["stage"]) for row in journal] != expected_stages or any(
        row.get("artifact_kind")
        != (
            CompletedWindowJournal.artifact_kind
            if completed
            else (
                WindowBoundaryJournal.artifact_kind
                if boundary
                else WindowJournal.artifact_kind
            )
        )
        or row.get("numerical_claim") is not False
        or row.get("performance_claim") is not False
        for row in journal
    ):
        raise ValueError("window compile journal order/scope differs")


def validate_record(record: dict[str, Any], pin: str) -> None:
    from scripts.greenfield.ws32_prefill_layer_campaign import (
        checkpoint_ledger,
        validate_workers as validate_fleet,
    )

    boundary = record.get("kernel") == BOUNDARY_KERNEL
    completed = record.get("kernel") == COMPLETED_KERNEL
    kernel, protocol, _, _ = acquisition_mode(boundary=boundary, completed=completed)
    if any(
        record.get(k) != v
        for k, v in dict(
            status="SUCCESS",
            kernel=kernel,
            protocol=protocol,
            code_hash=pin,
            layer=6,
            latency=None,
            warmup=0,
            iterations=0,
            comparison=dict(passed=None, diagnostic_evidence_complete=True),
        ).items()
    ) or any(
        record.get(k) is not v
        for k, v in dict(
            compile_only=True,
            numerical_execution_authorized=False,
            admission_only=False,
            baseline_only=False,
            diagnostic_only=True,
            performance_claim=False,
            profiler_free_timing=False,
        ).items()
    ):
        raise ValueError("window aggregate acquisition classification differs")
    pins, ledger = checkpoint_ledger(6)
    validate_fleet(
        record["workers"],
        pin,
        layer=6,
        pins=pins,
        ledger=ledger,
        window_boundary=boundary,
        completed_window=completed,
    )
    if record["hlo"] != record["workers"][0]["hlo"]:
        raise ValueError("window aggregate HLO scope differs")
