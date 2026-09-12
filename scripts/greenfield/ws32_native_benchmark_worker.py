"""Cold-load and bind the actual native benchmark runtime.

Worker component, not an unleased launcher. The outer runner initializes and
authenticates the existing fleet, pins clean published source, transports the
existing final-layout checkpoint and owns timeout/publication/cleanup. This
module uses that loader and ORIGINAL WK/materializer boundaries; it never
executes a legacy engine, repacks weights or replays long-context campaigns.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import gc
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping

import jax
from glm_tpu.greenfield.checkpoint.ws32_runtime_checkpoint import (
    load_ws32_runtime_checkpoint, verify_ws32_runtime_checkpoint,
)
from glm_tpu.greenfield.runtime import ws32_decoder as decoder
from scripts.greenfield import ws32_delivery_decode as preparation
from scripts.greenfield import ws32_native_benchmark_programs as programs
from scripts.greenfield import ws32_native_benchmark_memory as memory
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from scripts.greenfield.prefill_window_worker import BudgetedCalls
from scripts.greenfield.ws32_acquisition_journal import Ws32NumericalJournal
from scripts.greenfield.ws32_native_benchmark_runtime import bind_admitted_runtime
from scripts.greenfield.ws32_phase_weights import PhaseWeights

PROFILE = "ws32_native_sampled_request_v1"


class NativeBenchmarkJournal(Ws32NumericalJournal):
    """Original fsynced journal, with a separate sampled-request identity."""

    artifact_kind = "greenfield_ws32_native_benchmark_journal_v1"

    def __init__(self, path: Path, identity: Mapping[str, Any]):
        super().__init__(path, identity)
        from scripts.greenfield.ws32_native_benchmark_transport import BoundedJournalStream
        self._stream = BoundedJournalStream(self._stream)

    def _check_identity(self, identity: Mapping[str, Any]) -> None:
        if (identity.get("profile") != PROFILE
                or identity.get("compile_only") is not False
                or identity.get("context_capacity") != programs.PLAN.context_capacity
                or type(identity.get("jax_process_index")) is not int
                or not 0 <= identity["jax_process_index"] < 8
                or type(identity.get("launch_process_id")) is not int
                or not 0 <= identity["launch_process_id"] < 8
                or re.fullmatch(r"[0-9a-f]{40}", identity.get("code_hash", "")) is None):
            raise ValueError("native journal requires explicit sampled fleet identity")


@dataclass
class LoadedNativeWorker:
    runtime: Any
    compiled: dict[str, Any]
    reports: dict[str, Any]
    record: dict[str, Any]


class _RawCheckedFunction:
    """Guard the original writer's compile call without moving shared source.

    Original compile_program first writes RAW then invokes compile(). This
    adapter checks those same RAW bytes at that boundary; the common writer
    and WK builder locations stay byte-for-byte unchanged.
    """
    def __init__(self, fn: Any, expected: tuple[int, str]):
        self.fn, self.expected = fn, expected

    def lower(self, *values: Any) -> Any:
        lowered = self.fn.lower(*values)
        expected = self.expected

        class CheckedLowered:
            def compiler_ir(self, dialect: str) -> Any:
                return lowered.compiler_ir(dialect=dialect)

            def compile(self) -> Any:
                raw = str(lowered.compiler_ir(dialect="stablehlo")).encode()
                if (len(raw), sha256(raw).hexdigest()) != expected:
                    raise ValueError("registered RAW differs before compilation")
                return lowered.compile()
        return CheckedLowered()


def _compile_inspected(name: str, fn: Any, values: tuple[Any, ...], *,
                       calls: BudgetedCalls, repo: Path) -> Any:
    """Keep literal compiler originals BEFORE any inspection or execution."""
    from scripts.greenfield.probe_ws32_prefill_layer import compile_program
    from scripts.greenfield.ws32_delivery_hlo import FRESH_OPTIMIZED_MARKER

    compiled = compile_program(_RawCheckedFunction(fn, programs.RAW[name]),
        values, name, calls.root, calls.record, journal=calls.journal)
    stable = (calls.root / f"{name}.stablehlo.mlir").read_text()
    optimized = (calls.root / f"{name}.optimized_hlo.txt").read_text()
    report = calls.journal.inspect(name, stable, optimized, lambda: programs.inspect_hlo(
        stable, optimized, repo=repo, graph=name,
        expected_optimized=FRESH_OPTIMIZED_MARKER))
    calls.record["programs"][name]["admission"] = report
    calls.programs[name] = compiled
    return compiled


def _release_programs(calls: BudgetedCalls, jits: Mapping[str, Any]) -> None:
    """Drop this phase's compiler roots without globally clearing other caches."""
    failure = None
    for fn in jits.values():
        try:
            fn.clear_cache()
        except Exception as exc:
            failure = failure or exc
    calls.programs.clear()
    gc.collect()
    if failure is not None:
        raise failure


def materialize_exact(*, mesh: Any, config: Any, weights: Any,
                      calls: BudgetedCalls, repo: Path) -> Any:
    """Original two completed calls, retaining raw/WK roots for later requests.

    Unlike the old one-way long helper, this uses the explicit native HLO/source
    profile. The original conservative preparation budget still counts every
    live array and one then two resident executables. No model math is copied.
    """
    jits: dict[str, Any] = {}
    source = raw = decoded = compiled = None
    failure = None
    try:
        def setup():
            programs.require_source(repo)
            if (not config.exact_dsa or not config.strategy_nd_dense
                    or calls.programs or calls.record["call_evidence"]):
                raise ValueError("native materializer needs a fresh exact preparation phase")
            return decoder.build_ws32_exact_dsa_materializer_program(mesh, config)
        source = calls.phase("native_exact/setup", setup)
        raw = calls.phase("native_exact/raw", lambda:
            decoder.select_ws32_exact_dsa_raw_weights(weights, config))
        for name, method in (("exact_materialize", source.decode),
                             ("exact_promote", source.promote)):
            values = (raw,) if name == "exact_materialize" else (decoded,)
            jits[name] = jax.jit(method)
            compiled = calls.phase(f"native_exact/compile/{name}", lambda:
                _compile_inspected(name, jits[name], values, calls=calls, repo=repo))
            result = calls.call(f"delivery_decode/{name}", name, values,
                preserve=lambda value: preparation.preserve_schema(
                    value, calls.record["call_evidence"][-1]))
            if name == "exact_materialize":
                decoded = result  # completed BF16, never fused with promotion
            else:
                return result
    except Exception as exc:
        failure = exc
        raise
    finally:
        source = raw = decoded = compiled = values = method = None
        try:
            calls.phase("native_exact/code_released", lambda: _release_programs(calls, jits))
        except Exception as exc:
            if failure is None:
                raise
            calls.record["release_error"] = f"{type(exc).__name__}: {exc}"


def _check_abstract(actual: Any, abstract: Any, *, name: str) -> None:
    """Loaded views must have the production compiler's complete ABI/sharding."""
    if jax.tree.structure(actual) != jax.tree.structure(abstract):
        raise ValueError(f"native loaded {name} tree differs from registered program")
    for value, expected in zip(jax.tree.leaves(actual), jax.tree.leaves(abstract), strict=True):
        if (not isinstance(value, jax.Array) or value.is_deleted()
                or value.shape != expected.shape or value.dtype != expected.dtype
                or not value.sharding.is_equivalent_to(expected.sharding, len(value.shape))):
            raise ValueError(f"native loaded {name} shape/dtype/sharding differs")


def load_runtime(*, args: Any, repo: Path, root: Path, mesh: Any,
                 physical_mesh: Any, topology: Any, fleet_sha: str,
                 consensus: Callable[[bool], bool],
                 preserve_memory: Callable[[str, Mapping[str, Any]], None]) -> LoadedNativeWorker:
    """Verify/load once, prepare original weights, compile and bind warm requests.

    Called ONLY after the existing outer fleet initialization/ownership checks.
    Refuses CPU, wrong physical mapping, dirty code and mismatched checkpoint
    identities. Preparation code is gone before all six model roles coexist.
    This returns a real runtime, not a benchmark score or a sealed run.
    """
    from scripts.greenfield import run_short_decoder_ws32 as original
    from scripts.greenfield.probe_ws32_prefill_layer import authenticated_inventory
    from scripts.greenfield import ws32_delivery_wk
    from scripts.greenfield.ws32_batched_prefill_runner import replicated
    from glm_tpu.greenfield.runtime import build_ws32_main_rope_table, WS32_MAIN_ROPE_THETA
    from glm_tpu.greenfield.kernels.reference.rotary import rotary_table_sha256

    # Initialization itself belongs to the unchanged protected fleet helper.
    # Vote every local preflight before a later collective/device placement.
    def vote(action):
        value = error = None
        try:
            value = action()
        except Exception as exc:
            error = exc
        agreed = consensus(error is None)
        if error is not None:
            raise error
        if type(agreed) is not bool or not agreed:
            raise RuntimeError("native cold-load peer preflight refused")
        return value

    def preflight():
        original._require_clean_code(args.expected_code_hash)
        programs.require_source(repo)
        recipe = json.loads((repo / "configs/greenfield-ws32-batched-acquisition.json").read_text())["environment"]
        for field, key in (
            ("strategy_nd_dense_overlay_manifest_sha256", "MANIFEST_SHA"),
            ("strategy_nd_dense_overlay_manifest_file_sha256", "MANIFEST_FILE_SHA"),
            ("strategy_nd_dense_overlay_success_file_sha256", "SUCCESS_FILE_SHA"),
        ):
            if getattr(args, field) != recipe["GLM_GREENFIELD_WS32_STRATEGY_ND_DENSE_OVERLAY_" + key]:
                raise ValueError("native overlay differs from the tested checkpoint recipe")
        if (repo.resolve() != original.REPO or jax.default_backend() != "tpu"
                or jax.device_count() != 32 or jax.process_count() != 8
                or len(jax.local_devices()) != 4
                or physical_mesh.mesh_hash != args.mesh_sha256
                or topology.topology_hash != args.topology_sha256
                or fleet_sha != args.topology_fleet_sha256
                or args.context_capacity != programs.PLAN.context_capacity
                or args.checkpoint_transport != "shm"):
            raise ValueError("native cold-load fleet/capacity/transport differs")
        local_ids = {int(d.id) for d in jax.local_devices()}
        slots = {device_id: slot for slot, device_id in
                 enumerate(physical_mesh.flattened_device_ids) if device_id in local_ids}
        if len(slots) != 4:
            raise ValueError("native cold-load requires four physical owners")
        return slots
    slots = vote(preflight)
    # No writes to an existing run phase; failed prefixes are never reused.
    vote(lambda: root.mkdir(exist_ok=False))
    identity = dict(profile=PROFILE, compile_only=False, code_hash=args.expected_code_hash,
        launch_process_id=args.process_id, jax_process_index=int(jax.process_index()),
        context_capacity=programs.PLAN.context_capacity, mesh_sha256=args.mesh_sha256,
        topology_sha256=args.topology_sha256, topology_fleet_sha256=fleet_sha,
        local_slots=[dict(device_id=d, slot=s) for d, s in sorted(slots.items())])
    journal = vote(lambda: NativeBenchmarkJournal(root / "journal.jsonl", identity))
    record = dict(identity, artifact_kind="ws32_native_cold_load_v1", complete=False,
                  programs={}, phase_seconds={}, performance_claim=False)
    calls = BudgetedCalls(root=root, record=record, consensus=consensus,
        journal=journal, local_slots=slots, budgeter=preparation.memory_budget)
    compiled: dict[str, Any] = {}
    jits: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        config = decoder.Ws32DecoderConfig(original._geometry(), programs.PLAN.context_capacity,
            exact_dsa=True, strategy_nd_dense=True, host_main_rope_table=True)
        # Original abstract metadata binds the complete2310-leaf layout. Full
        # verifier below still hashes this host's four real checkpoint files.
        metadata = calls.phase("native/load_metadata", lambda: programs.read_metadata(repo))
        pins = json.loads((repo / "docs/artifacts/prefill-window-layer6-host-admission-20260908.json").read_text())
        inventory = calls.phase("native/source_inventory", lambda: authenticated_inventory(
            args.source_inventory, pins["source_inventory_sha256"]))
        def verify():
            if (args.checkpoint_manifest_sha256 != metadata.manifest["manifest_sha256"]
                    or args.checkpoint_success_sha256 != metadata.success["success_sha256"]):
                raise ValueError("native actual checkpoint differs from abstract compiler metadata")
            return verify_ws32_runtime_checkpoint(args.checkpoint_root,
                expected_manifest_sha256=args.checkpoint_manifest_sha256,
                expected_success_sha256=args.checkpoint_success_sha256,
                expected_mesh_hash=args.mesh_sha256, expected_topology_hash=args.topology_sha256,
                inventory=inventory, geometry=config.geometry, verify_file_hashes=True,
                verify_file_hash_slots=tuple(sorted(slots.values())), local_slot_layout=True)
        checkpoint = calls.phase("native/checkpoint_verified", verify)
        load_start = time.perf_counter()
        loaded = calls.phase("native/checkpoint_loaded", lambda:
            load_ws32_runtime_checkpoint(checkpoint, mesh=mesh, physical_mesh=physical_mesh))
        record["phase_seconds"]["checkpoint_load"] = time.perf_counter() - load_start
        record["checkpoint"] = dict(manifest_sha256=checkpoint.manifest["manifest_sha256"],
            success_sha256=checkpoint.success["success_sha256"],
            source_inventory_sha256=inventory.inventory_sha256,
            verified_slots=sorted(slots.values()), local_device_slots=list(loaded.local_device_slots),
            memory_before=list(loaded.device_memory_before), memory_after=list(loaded.device_memory_after))
        arrays = dict(loaded.arrays)
        owner = calls.phase("native/raw_view", lambda: PhaseWeights(arrays, config))
        del loaded
        record["wk_preparation"] = ws32_delivery_wk.prepare(repo=repo,
            root=root / "wk", owner=owner, mesh=mesh, journal=journal,
            consensus=consensus, local_slots=slots, identity=identity, native_benchmark=True)

        # The original verified overlay loader operates BEFORE any resident
        # model code. Keep raw views; do not call the one-way begin_decode().
        prep = preparation.open_phase(root=root / "exact", journal=journal,
            consensus=consensus, local_slots=slots, identity=identity)
        prep.record["phase_contract"] = PROFILE
        overlay_start = time.perf_counter()
        overlay, loaded_overlay = prep.phase("native/overlay", lambda:
            original._load_dense_overlay(args=args, config=config, mesh=mesh,
                physical_mesh=physical_mesh, all_arrays=arrays,
                before_load=lambda verified: preparation.overlay_preflight(
                    verified, prep, owner.resident_roots())))
        prep.phase("native/overlay_completed", lambda: preparation.overlay_completed(prep))
        record["phase_seconds"]["overlay_load"] = time.perf_counter() - overlay_start
        names = jax.tree.leaves(decoder.ws32_decoder_weight_names(config))
        weights = prep.phase("native/decode_view", lambda:
            decoder.bind_ws32_decoder_weights({name: arrays[name] for name in names}, config))
        record["overlay"] = dict(manifest_sha256=overlay.manifest["manifest_sha256"],
            manifest_file_sha256=overlay.manifest_file_sha256,
            success_file_sha256=overlay.success_file_sha256)
        del arrays, loaded_overlay
        exact = materialize_exact(mesh=mesh, config=config, weights=weights, calls=prep, repo=repo)
        record["exact_preparation"] = preparation.finish(prep)
        # materialize_exact returns only promoted arrays, never a preparation
        # executable or decoded temporary. Original WK helper also drops code.
        gc.collect()
        pair = calls.phase("native/prefill_programs", lambda: programs.prepare(mesh, metadata, repo=repo))
        companions = calls.phase("native/companion_programs", lambda:
            programs.prepare_companions(mesh, pair, repo=repo))
        def place_rope():
            host = build_ws32_main_rope_table(config)
            record["rope"] = dict(rows=config.context_capacity, bytes_per_device=host.nbytes,
                theta=WS32_MAIN_ROPE_THETA, sha256=rotary_table_sha256(host))
            result = replicated(mesh, host)
            jax.block_until_ready(result)
            return result
        rope = calls.phase("native/rope", place_rope)
        def check_views():
            _, _, _, raw_shape, wk_shape, rope_shape, _ = pair.inputs["prefill_chunk"]
            _check_abstract(owner.raw_weights, raw_shape, name="raw_weights")
            _check_abstract(owner.wk, wk_shape, name="WK")
            _check_abstract(rope, rope_shape, name="rope")
            _check_abstract(weights, companions.inputs["decode"][2], name="decode_weights")
            _check_abstract(exact, companions.inputs["decode"][3], name="exact_weights")
        calls.phase("native/loaded_ABI", check_views)
        try:
            for name in memory.ROLES:
                if name.startswith("prefill_"):
                    fn, values = pair.programs[name].execute, pair.inputs[name]
                elif name == "cache_init":
                    fn = programs.build_cache_initializer(mesh, owner.raw_config)
                    values = (pair.inputs["prefill_chunk"][2].prompt_length,)
                else:
                    fn, values = companions.programs[name], companions.inputs[name]
                jits[name] = fn
                compiled[name] = calls.phase(f"native/resident_compile/{name}", lambda:
                    _compile_inspected(name, fn, values, calls=calls, repo=repo))
        finally:
            # Keep explicit Compiled handles; drop redundant JIT/lowering caches.
            # No compiled preparation handle is in pair/companions (abstract only).
            for fn in jits.values():
                fn.clear_cache()
        reports = {name: record["programs"][name]["admission"] for name in memory.ROLES}
        native = calls.phase("native/bind", lambda: bind_admitted_runtime(
            repo=repo, mesh=mesh, raw_config=owner.raw_config, decode_config=config,
            raw_weights=owner.raw_weights, decode_weights=weights, wk=owner.wk,
            exact_weights=exact, rope=rope, compiled=compiled, reports=reports,
            local_slots=slots, process_index=int(jax.process_index()),
            preserve_memory=preserve_memory, fleet_all=consensus))
        # Admission is repeated at request start, but prove the very first cache
        # can fit now without advancing that request admission phase machine.
        def admit_first_cache():
            snapshot = memory.make_record(compiled, None, retained_roots=native.resident_roots(),
                devices=tuple(jax.local_devices()), local_slots=slots,
                process_index=int(jax.process_index()), phase="before_cache")
            _atomic_json(root / "initial_memory.json", snapshot)
            memory.validate_record(snapshot)
        calls.phase("native/initial_memory", admit_first_cache)
        record["cold_load_compile_seconds"] = time.perf_counter() - started
        record["complete"] = True
        calls.phase("native/cold_ready", lambda: None)
        return LoadedNativeWorker(native, compiled, reports, record)
    except Exception as exc:
        record["complete"] = False
        record["failure"] = f"{type(exc).__name__}: {exc}"
        try:
            _atomic_json(root / "runner.json", record)
        except Exception as publication_error:
            record["failure_publication_error"] = str(publication_error)
        raise
    finally:
        journal.close()
