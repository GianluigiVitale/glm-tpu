"""Layer6 compiler acquisition only; no model or WK executable is dispatched.

The existing protected layer campaign owns deployment, leases and publication.
Acquiring a graph is not admission to execute it, even if its basic inventory
looks plausible. Exact new profiles and simultaneous numerical memory follow.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import time
from typing import Any, Callable

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
MEMORY_KEYS = (
    "argument_size_in_bytes",
    "output_size_in_bytes",
    "alias_size_in_bytes",
    "temp_size_in_bytes",
    "generated_code_size_in_bytes",
)


def is_acquisition_tag(tag: str) -> bool:
    return (
        re.fullmatch(r"greenfield_fp8_" + KERNEL + r"_l6_[a-zA-Z0-9_]+", tag)
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
) -> tuple[Any, ...]:
    """Preserve every graph, memory record and inspection failure; NEVER call it.

    A parser refusal is deferred until both layer graphs have been collected.
    A real lowering/compilation failure votes and stops before further work.
    Returning executables keeps all four resident for a final memory snapshot.
    """
    if tuple(p[0] for p in programs) != PROGRAMS or record.get("protocol") != PROTOCOL:
        raise ValueError("window acquisition program/protocol inventory differs")
    journal = fleet_step(
        "journal",
        lambda: WindowJournal(
            root / "compile_journal.jsonl",
            dict(
                protocol=PROTOCOL,
                compile_only=True,
                code_hash=record["code_hash"],
                launch_rank=record["launch_rank"],
            ),
        ),
        record=record,
        root=root,
        consensus=consensus,
    )
    compiled = []
    try:
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
                try:
                    report = journal.inspect(name, stable, hlo, lambda: inspector(hlo))
                    record["programs"][name]["inventory"] = report
                except Exception as exc:
                    record["programs"][name][
                        "inspection_error"
                    ] = f"{type(exc).__name__}: {exc}"

            fleet_step(
                f"inspect_{name}",
                inspect,
                record=record,
                root=root,
                consensus=consensus,
            )
        return tuple(compiled)
    finally:
        journal.close()


def execute_acquisition(
    *,
    args: Any,
    record: dict[str, Any],
    mesh: Any,
    config: Any,
    weights: Any,
    consensus: Callable[[bool], bool],
) -> None:
    """Compile from abstract prompt/WK arrays: no model inputs or WK execution."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from glm_tpu.greenfield.kernels.reference.rotary import build_rotary_table_host
    from scripts.greenfield.prefill_layer_evidence import INPUT_FIELDS
    from scripts.greenfield.probe_ws32_prefill_layer import (
        input_specs,
        build_wk_programs,
        compile_program,
    )

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

        wide, small = window.build_programs(
            mesh,
            specs,
            dsa_contract=config.dsa_contract,
            attention_contract=config.attention_contract,
            moe_contract=config.moe_contract,
            rms_norm_epsilon=config.rms_norm_epsilon,
        )
        return (
            (
                "wk_decode",
                decode,
                (weights.dsa.wk_bits_local, weights.dsa.wk_scale_local),
            ),
            ("wk_promote", promote, (decoded,)),
            ("candidate", wide, values_for(window.ROWS)),
            ("control", small, values_for(window.CONTROL_ROWS)),
        )

    programs = fleet_step(
        "prepare", prepare, record=record, root=args.output_dir, consensus=consensus
    )
    compiled = acquire_programs(
        programs,
        root=args.output_dir,
        record=record,
        consensus=consensus,
        compiler=compile_program,
    )

    def finish() -> None:
        record["device_memory_stats_including_reference"] = [
            dict(device_id=int(d.id), stats=_memory_stats(d))
            for d in jax.local_devices()
        ]
        record["resident_programs_at_snapshot"] = list(PROGRAMS)
        record["model_executable_calls"] = 0
        record["wk_executable_calls"] = 0
        record["memory_scope"] = (
            "SELECTED_WEIGHTS_AND_FOUR_COMPILED_PROGRAMS_NO_NUMERICAL_SCRATCH_OR_OUTPUTS"
        )
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
    assert len(compiled) == 4  # Keep all executable references live through snapshot.


def validate_workers(
    records: list[dict[str, Any]],
    *,
    pin: str,
    pins: dict[str, Any],
    ledger: dict[int, Any],
    order: tuple[int, ...],
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

    slots = []
    for record in records:
        exact = dict(
            status="SUCCESS",
            protocol=PROTOCOL,
            layer=6,
            code_hash=pin,
            selected_layer_ids=[6],
            rows=128,
            control_rows=32,
            context_capacity=4096,
            key_tile=512,
            iterations=0,
            latency=None,
            reference_scope=REFERENCE_SCOPE,
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
            resident_programs_at_snapshot=list(PROGRAMS),
            memory_scope="SELECTED_WEIGHTS_AND_FOUR_COMPILED_PROGRAMS_NO_NUMERICAL_SCRATCH_OR_OUTPUTS",
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
            or set(record["programs"]) != set(PROGRAMS)
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
    raw = (root / "compile_journal.jsonl").read_bytes()
    if sha256(raw).hexdigest() != record["compile_journal_sha256"]:
        raise ValueError("window acquisition journal bytes differ")
    journal = [json.loads(line) for line in raw.splitlines()]
    if not journal or journal[0].get("identity") != dict(
        protocol=PROTOCOL,
        compile_only=True,
        code_hash=record["code_hash"],
        launch_rank=record["launch_rank"],
    ):
        raise ValueError("window acquisition journal identity differs")
    expected_stages = [(None, "identity")]
    for name in PROGRAMS:
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
        if (
            len(stages) != 4
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
        row.get("artifact_kind") != WindowJournal.artifact_kind
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

    if any(
        record.get(k) != v
        for k, v in dict(
            status="SUCCESS",
            kernel=KERNEL,
            protocol=PROTOCOL,
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
    validate_fleet(record["workers"], pin, layer=6, pins=pins, ledger=ledger)
    if record["hlo"] != record["workers"][0]["hlo"]:
        raise ValueError("window aggregate HLO scope differs")
