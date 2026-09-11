"""Selected-owner history runtime binding, not execution or launch authority.

Reuse the selected checkpoint/overlay loaders and the frozen seven-job abstract
preparation exactly once. The protected parent owns compilation, admission,
materializer calls, history execution, publication and authenticated cleanup.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import json
import re
import socket
from typing import Any, Callable, Mapping

import numpy as np

from glm_tpu.greenfield.checkpoint.ws32_layer_subset import load_ws32_layer_subset
from glm_tpu.greenfield.checkpoint.ws32_strategy_nd_dense import load_ws32_strategy_nd_dense_overlay
from glm_tpu.greenfield.runtime.ws32_decoder import (
    Ws32DecoderConfig, _bind_weight_name_tree, ws32_decoder_weight_names,
)
from glm_tpu.greenfield.types import ModelGeometry
from glm_tpu.greenfield.validation.ws32_prefill_memory import (
    MEMORY_FIELDS, budget_resident_execution, capture_resident_buffers,
)
from scripts.greenfield import ws32_history_compile as compiler
from scripts.greenfield import ws32_history_observer_prepare as observer
from scripts.greenfield import ws32_history_preflight as preflight_module
from scripts.greenfield import ws32_history_protocol as protocol
from scripts.greenfield.prefill_window_acquisition import fleet_step
from scripts.greenfield.prefill_window_evidence import same_json
from scripts.greenfield.prefill_window_worker import validate_memory_owners


@dataclass(frozen=True)
class BoundHistoryRuntime:
    config: Any
    prepared: Any
    jobs: tuple[tuple[str, Any, tuple], ...]
    embedding: Any
    layers: tuple
    observer_layers: tuple
    raw_exact: tuple
    rope: Any
    prompt_tokens: np.ndarray
    originals: Mapping[str, Mapping[str, np.ndarray]]
    local_slots: Mapping[int, int]

    def resident_roots(self) -> dict[str, Any]:
        """Aliases remain visible; the existing census deduplicates real pointers."""
        return dict(embedding=self.embedding, raw_layers=self.layers,
                    observer_layers=self.observer_layers, raw_exact=self.raw_exact,
                    rope=self.rope)


def bind(*, root: Path, record: dict, repo: Path, runtime: tuple) -> tuple:
    """Authenticate preflight/originals against the actual four local TPU owners."""
    jax, mesh, physical, topology, fleet = runtime
    preflight_module._plain_path(root)
    rank = record.get("launch_rank")
    if (type(rank) is not int or not 0 <= rank < 8
            or record.get("protocol") != protocol.PROTOCOL
            or record.get("compile_only") is not False
            or record.get("diagnostic_only") is not True
            or not protocol.is_tag(record.get("tag"))
            or re.fullmatch(r"[0-9a-f]{40}", str(record.get("code_hash"))) is None):
        raise ValueError("history runtime record identity differs")
    runners, originals, identity = preflight_module.load_originals(
        root / "retained_reference", repo=repo, rank=rank)
    prior = runners["candidate"]
    path = root / "retained_preflight.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("history retained preflight must be a regular original")
    raw = path.read_bytes()
    preflight = json.loads(raw)
    expected = dict(protocol=protocol.PROTOCOL, code_hash=record["code_hash"],
        launch_rank=rank, tag=record["tag"], hostname=socket.gethostname(),
        selected_layer_ids=list(protocol.LAYERS), include_embedding=True,
        context_capacity=protocol.CAPACITY, host_main_rope_table=True,
        selected_leaf_count=protocol.SELECTED_LEAVES, payload_bytes_per_chip=protocol.PAYLOAD_BYTES,
        overlay_tensor_count=protocol.OVERLAY_TENSORS, overlay_bytes_per_chip=protocol.OVERLAY_BYTES,
        prompt_ids_sha256=protocol.PROMPT_SHA,
        prompt_hash_source="AUTHENTICATED_ORIGINAL_TOKEN_ORACLE",
        original_oracle_pins={key: prior[key] for key in preflight_module.ORACLE_KEYS},
        main_rope_table=prior["main_rope_table"],
        strategy_nd_dense_overlay=prior["strategy_nd_dense_overlay"],
        numerical_execution_available=False, numerical_admission=False,
        numerical_promotion=False, performance_claim=False, **identity)
    same_json({key: preflight.get(key) for key in expected}, expected, "history retained preflight")
    if (jax.default_backend() != "tpu" or jax.process_count() != 8 or jax.device_count() != 32
            or jax.process_index() != prior["jax_process_index"]
            or socket.gethostname() != prior["hostname"]):
        raise ValueError("history actual runtime/original identity differs")
    for actual, key in ((physical.mesh_hash, "mesh_sha256"),
                        (topology.topology_hash, "topology_sha256"),
                        (fleet, "topology_fleet_sha256")):
        if any(runner[key] != actual for runner in runners.values()):
            raise ValueError("history physical topology differs from originals")
    devices = np.asarray(mesh.devices, object)
    if devices.shape != (8, 4) or tuple(mesh.axis_names) != ("expert", "feature"):
        raise ValueError("history actual mesh axes/shape differ")
    same_json([[int(d.id) for d in row] for row in devices], physical.device_ids,
              "history physical mesh ordering")
    ids = tuple(int(d) for d in physical.flattened_device_ids)
    if len(ids) != 32 or set(ids) != set(range(32)):
        raise ValueError("history physical mesh owners differ")
    slot_by_id = {device: slot for slot, device in enumerate(ids)}
    local = tuple(jax.local_devices())
    slots = {int(d.id): slot_by_id.get(int(d.id)) for d in local}
    expected_slots = {v["device_id"]: v["device_slot"] for v in prior["local_device_slots"]}
    if (len(local) != 4 or len(slots) != 4 or slots != expected_slots
            or any(d.platform != "tpu" or d.process_index != jax.process_index() for d in local)
            or sorted(v["device_slot"] for v in preflight["headers"]) != sorted(slots.values())):
        raise ValueError("history actual local owners differ")
    same_json(preflight["local_device_slots"], prior["local_device_slots"], "history original owners")
    record.update(jax_process_index=jax.process_index(), physical_device_ids=physical.device_ids,
        mesh_sha256=physical.mesh_hash, topology_sha256=topology.topology_hash,
        topology_fleet_sha256=fleet, retained_preflight_sha256=sha256(raw).hexdigest(),
        versions={key: version(key) for key in ("jax", "libtpu")}, **identity)
    return runners, originals, slots, preflight


def compiler_jobs(mesh: Any, metadata: Any, config: Any, *, repo: Path,
                  prepared: Any = None) -> tuple[Any, tuple]:
    """Nine abstract jobs; the optional seven-job preparation is reused, not rebuilt."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from scripts.greenfield.probe_ws32_prefill_layer import build_wk_programs

    compiler.require_source(repo)
    if prepared is None:
        prepared = compiler.prepare(mesh, metadata, repo=repo)
    if (set(prepared.programs) != set(compiler.PROGRAMS)
            or set(prepared.inputs) != set(compiler.PROGRAMS)
            or prepared.manifest_sha256 != metadata.manifest["manifest_sha256"]
            or prepared.source_inventory_sha256 != metadata.manifest["source"]["inventory_sha256"]
            or any(not isinstance(v, jax.ShapeDtypeStruct) or v.sharding is None
                   for v in jax.tree.leaves(prepared.inputs))):
        raise ValueError("history abstract compiler preparation differs")
    layers = prepared.inputs["candidate_b128"][6]
    if len(layers) != 7:
        raise ValueError("history abstract layer count differs")
    first = layers[protocol.PRODUCERS[0]].dsa
    for value, shape, dtype in ((first.wk_bits_local, (128, 6144), jnp.uint8),
                                (first.wk_scale_local, (1, 48), jnp.float32)):
        if (value.shape != shape or value.dtype != dtype
                or value.sharding.spec != P(None, "feature")):
            raise ValueError("history producer WK source shape/dtype/sharding differs")
    for layer in protocol.PRODUCERS:
        for field in ("wk_bits_local", "wk_scale_local"):
            left, right = getattr(first, field), getattr(layers[layer].dsa, field)
            if (left.shape, left.dtype, left.sharding.spec) != (right.shape, right.dtype, right.sharding.spec):
                raise ValueError("history producer WK compiler interfaces differ")
    decode, promote = build_wk_programs(mesh, first.wk_bits_local.sharding.spec,
        first.wk_scale_local.sharding.spec, contract=config.dsa_contract)
    decoded = jax.ShapeDtypeStruct((128, 6144), jnp.bfloat16, sharding=NamedSharding(mesh, P()))
    jobs = (("wk_decode", decode, (first.wk_bits_local, first.wk_scale_local)),
            ("wk_promote", promote, (decoded,)),
            *((name, prepared.programs[name].execute, prepared.inputs[name]) for name in compiler.PROGRAMS))
    if tuple(name for name, _, _ in jobs) != protocol.PROGRAMS:
        raise ValueError("history nine compiler jobs differ")
    return prepared, jobs


def _tree_contract(actual: Any, abstract: Any) -> None:
    import jax
    if jax.tree.structure(actual) != jax.tree.structure(abstract):
        raise ValueError("history loaded weight tree differs from abstract compiler inputs")
    for value, expected in zip(jax.tree.leaves(actual), jax.tree.leaves(abstract), strict=True):
        if (not isinstance(value, jax.Array) or not isinstance(expected, jax.ShapeDtypeStruct)
                or expected.sharding is None
                or tuple(value.shape) != tuple(expected.shape) or value.dtype != expected.dtype
                or not value.sharding.is_equivalent_to(expected.sharding, len(expected.shape))):
            raise ValueError("history loaded shape/dtype/sharding differs from compiler inputs")


def _selected_owners(loaded: Any, subset: Any, slots: Mapping[int, int]) -> None:
    schema = subset.metadata.plans[0].tensors
    expected = {slot: dict(device_id=device, device_slot=slot,
        expected_full_file_sha256_not_verified=subset.metadata.records_by_slot[slot]["sha256"],
        observed_selected_tensor_sha256={schema[i].name: subset.metadata.records_by_slot[slot]["tensor_sha256"][i]
                                        for i in subset.tensor_indices},
        selected_payload_bytes=protocol.PAYLOAD_BYTES) for device, slot in slots.items()}
    rows = loaded.local_device_slots
    if len(rows) != 4 or len({v["device_slot"] for v in rows}) != 4:
        raise ValueError("history selected owner inventory differs")
    same_json(sorted(rows, key=lambda v: v["device_slot"]),
              [expected[slot] for slot in sorted(expected)], "history selected owner payloads")


def prepare_bound(*, root: Path, record: dict, repo: Path, runtime: tuple,
                  consensus: Callable[[bool], bool], prepared: Any = None) -> BoundHistoryRuntime:
    """Bind/load shared operands under voted phases; zero executable dispatch."""
    jax, mesh, physical, _, _ = runtime
    def step(name: str, action: Callable) -> Any:
        return fleet_step(name, action, root=root, record=record, consensus=consensus)

    runners, originals, slots, retained = step("history/bind_runtime", lambda: bind(
        root=root, record=record, repo=repo, runtime=runtime))
    prior = runners["candidate"]

    def prepare_inputs() -> tuple:
        pins, subset = preflight_module.selected_metadata(repo, tuple(sorted(slots.values())))
        same_json(pins, retained["checkpoint_pins"], "history checkpoint pins")
        overlay = preflight_module.bind_metadata(repo, runners, pins, subset)
        same_json(retained["headers"], [{key: subset.metadata.records_by_slot[slot][key]
            for key in ("device_slot", "filename", "file_bytes", "header_sha256")}
            for slot in sorted(slots.values())], "history retained headers")
        if retained["overlay_manifest_sha256"] != overlay.manifest["manifest_sha256"]:
            raise ValueError("history retained overlay manifest differs")
        config = Ws32DecoderConfig(ModelGeometry.from_dict(subset.metadata.manifest["geometry"]),
                                  protocol.CAPACITY, host_main_rope_table=True)
        tokens, rope = preflight_module.host_inputs(runners, config)
        abstract, jobs = compiler_jobs(mesh, subset.metadata, config, repo=repo, prepared=prepared)
        names = ws32_decoder_weight_names(config)
        tree = (names.embedding_local, names.layers[:7])
        if len(jax.tree.leaves(tree)) != protocol.SELECTED_LEAVES:
            raise ValueError("history selected name tree differs")
        record.update(checkpoint_pins=pins, checkpoint_manifest_sha256=abstract.manifest_sha256,
            checkpoint_success_sha256=pins["expected_success_sha256"],
            source_inventory_sha256=abstract.source_inventory_sha256,
            prompt_ids_sha256=protocol.PROMPT_SHA, main_rope_table=prior["main_rope_table"],
            original_oracle_pins={key: prior[key] for key in preflight_module.ORACLE_KEYS})
        return subset, overlay, config, tokens, rope, abstract, jobs, tree
    subset, overlay, config, tokens, host_rope, abstract, jobs, tree = step(
        "history/prepare_original_inputs", prepare_inputs)

    def load_raw() -> tuple:
        loaded = load_ws32_layer_subset(subset, mesh=mesh, physical_mesh=physical)
        if (loaded.layer_ids != protocol.LAYERS or loaded.include_embedding is not True
                or loaded.payload_bytes_per_chip != protocol.PAYLOAD_BYTES
                or set(loaded.arrays) != set(jax.tree.leaves(tree))):
            raise ValueError("history selected loaded inventory differs")
        _selected_owners(loaded, subset, slots)
        embedding, layers = _bind_weight_name_tree(tree, loaded.arrays)
        _tree_contract((embedding, layers), abstract.inputs["candidate_b128"][5:7])
        record.update(local_device_slots=loaded.local_device_slots,
            integrity_scope=loaded.integrity_scope, payload_bytes_per_chip=loaded.payload_bytes_per_chip,
            selected_layer_ids=list(loaded.layer_ids), selected_leaf_count=len(loaded.arrays),
            include_embedding=True,
            selected_load_device_memory_before=loaded.device_memory_before,
            selected_load_device_memory_after=loaded.device_memory_after)
        return embedding, layers
    embedding, layers = step("history/load_selected", load_raw)

    def load_overlay() -> tuple:
        loaded = load_ws32_strategy_nd_dense_overlay(overlay, mesh=mesh, physical_mesh=physical)
        key = lambda row: (row["device_id"], row["layer_id"])
        same_json(sorted(loaded.local_records, key=key),
                  sorted(prior["strategy_nd_dense_overlay"]["local_records"], key=key),
                  "history loaded overlay owners")
        _, observer_layers, raw_exact = observer.bind_selected_views(config, embedding, layers, loaded.arrays)
        _tree_contract(observer_layers, abstract.inputs["observer"][7])
        _tree_contract((raw_exact,), abstract.inputs["exact_decode"])
        record["strategy_nd_dense_overlay"] = prior["strategy_nd_dense_overlay"]
        record.update(overlay_tensor_count=len(loaded.arrays), overlay_bytes_per_chip=protocol.OVERLAY_BYTES)
        return observer_layers, raw_exact
    observer_layers, raw_exact = step("history/load_overlay", load_overlay)

    def place_rope() -> Any:
        from jax.sharding import NamedSharding, PartitionSpec as P
        rope = jax.make_array_from_callback(host_rope.shape, NamedSharding(mesh, P()),
                                           lambda index: host_rope[index])
        jax.block_until_ready(rope)
        _tree_contract(rope, abstract.inputs["candidate_b128"][8])
        return rope
    rope = step("history/place_original_rope", place_rope)
    return BoundHistoryRuntime(config, abstract, jobs, embedding, layers, observer_layers,
                               raw_exact, rope, tokens, originals, slots)


def memory_budget(census: Mapping, analyses: Mapping, *, active_graph: str) -> dict:
    """All nine executable allocations plus actual all-live buffers and 1GiB reserve."""
    if set(analyses) != set(protocol.PROGRAMS) or active_graph not in protocol.PROGRAMS:
        raise ValueError("history memory requires all nine resident programs")
    for name, values in analyses.items():
        caps = protocol.memory_caps(name)
        if (set(values) != set(MEMORY_FIELDS) or any(type(values[k]) is not int
                or not 0 <= values[k] <= caps[k] for k in MEMORY_FIELDS)):
            raise ValueError("history actual compiler allocation exceeds registered cap")
    return budget_resident_execution(census, analyses, active_graph=active_graph,
        resident_graphs=protocol.PROGRAMS, required_reserve_bytes=protocol.RESERVE)


def capture_memory(bound: BoundHistoryRuntime, programs: Mapping[str, Any], *,
                   devices: tuple, process_index: int, active_graph: str,
                   additional_roots: Mapping[str, Any] | None = None) -> dict:
    """A non-dispatch snapshot; call again as materializers/results/caches become live.

    The parent must retain both histories and all materialized operands. Include
    them as named roots for attribution; the reused census also discovers every
    JAX-live array, including outputs aliased by Python or omitted from labels.
    BudgetedCalls must use memory_budget before EVERY subsequent device call.
    """
    if set(programs) != set(protocol.PROGRAMS):
        raise ValueError("history memory snapshot requires nine actual executables")
    roots = bound.resident_roots()
    if set(roots) & set(additional_roots or {}):
        raise ValueError("history additional resident roots overwrite shared state")
    roots.update(additional_roots or {})
    analyses = {}
    for name in protocol.PROGRAMS:
        analysis = programs[name].memory_analysis()
        analyses[name] = {key: getattr(analysis, key, None) for key in MEMORY_FIELDS}
    census = capture_resident_buffers(roots, devices=devices)
    validate_memory_owners(census["devices"], local_slots=bound.local_slots, process_index=process_index)
    budget = memory_budget(census, analyses, active_graph=active_graph)
    return dict(census=census, compiled_memory=analyses, budget=budget,
                numerical_admission=False, runtime_peak_hbm_measured=False)
