"""Two production graph acquisitions from authenticated metadata, never weights.

This is preparation for the existing protected fleet wrapper, NOT a launcher.
No array allocation, checkpoint payload read, WK or model executable dispatch.
The ordinary full loader remains mandatory for the subsequent numerical run.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from glm_tpu.greenfield.checkpoint.ws32_runtime_checkpoint import (
    Ws32RuntimeMetadata,
    _read_ws32_runtime_metadata,
)
from glm_tpu.greenfield.partitioning.source_inventory import inspect_source_inventory
from glm_tpu.greenfield.types import ModelGeometry
from glm_tpu.greenfield.validation import ws32_prefill_admission as admission


@dataclass(frozen=True)
class AbstractPrefillPair:
    programs: Mapping[str, Any]
    inputs: Mapping[str, tuple[Any, ...]]
    manifest_sha256: str
    source_inventory_sha256: str


def read_metadata(repo: Path) -> Ws32RuntimeMetadata:
    """Reuse the full metadata verifier: manifest/SUCCESS/inventory, zero payload."""
    admission.require_acquired_model_source(
        repo, profile=admission.ROLLED_SHORT_PROFILE
    )
    pins = json.loads(
        (
            repo / "docs/artifacts/prefill-window-layer6-host-admission-20260908.json"
        ).read_text()
    )
    inventory = inspect_source_inventory(Path(pins["source_inventory"]))
    if inventory.inventory_sha256 != pins["source_inventory_sha256"]:
        raise ValueError("abstract prefill source inventory differs")
    geometry = ModelGeometry.from_hf_config(
        json.loads((repo / "configs/glm-5.2-fp8-config.json").read_text())
    )
    return _read_ws32_runtime_metadata(
        Path(pins["checkpoint_root"]),
        inventory=inventory,
        geometry=geometry,
        expected_manifest_sha256=pins["expected_manifest_sha256"],
        expected_success_sha256=pins["expected_success_sha256"],
        expected_mesh_hash=pins["expected_mesh_sha256"],
        expected_topology_hash=pins["expected_topology_sha256"],
    )


def prepare(
    mesh: Any, metadata: Ws32RuntimeMetadata, *, repo: Path
) -> AbstractPrefillPair:
    """Bind the existing production programs to ShapeDtypeStruct leaves only."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from glm_tpu.greenfield.runtime.ws32_decoder import (
        Ws32DecoderConfig,
        Ws32DecoderState,
        ws32_decoder_weight_names,
    )
    from glm_tpu.greenfield.runtime.ws32_batched_prefill import Ws32BatchedPrefillState
    from scripts.greenfield import ws32_batched_prefill_runner as adapter

    admission.require_acquired_model_source(
        repo, profile=admission.ROLLED_SHORT_PROFILE
    )
    plan = admission.short_plan(admission.ROLLED_SHORT_PROFILE)
    geometry = ModelGeometry.from_hf_config(
        json.loads((repo / "configs/glm-5.2-fp8-config.json").read_text())
    )
    if metadata.manifest["geometry"] != geometry.to_dict():
        raise ValueError("abstract model geometry differs from checkpoint metadata")
    config = Ws32DecoderConfig(
        geometry, plan.context_capacity, host_main_rope_table=True
    )
    schema = {item["name"]: item for item in metadata.manifest["tensor_schema"]}
    if len(schema) != len(metadata.manifest["tensor_schema"]):
        raise ValueError("abstract model tensor schema duplicates names")
    dtypes = {"BF16": jnp.bfloat16, "F32": jnp.float32, "U8": jnp.uint8}

    def abstract(shape: tuple[int, ...], dtype: Any, spec: Any = P()) -> Any:
        return jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(mesh, spec))

    arrays = {}
    for name in jax.tree.leaves(ws32_decoder_weight_names(config)):
        item = schema[name]
        arrays[name] = abstract(
            tuple(item["global_shape"]),
            dtypes[item["dtype"]],
            P(*item["partition_spec"]),
        )
    raw_config, weights = adapter.bind_raw_prefill_weights(
        arrays, replace(config, exact_dsa=True, strategy_nd_dense=True)
    )
    if raw_config != config:
        raise ValueError("abstract raw prefill configuration differs")
    decoder = Ws32DecoderState(
        abstract(config.kv_cache_shape, jnp.bfloat16, P(None, None, "expert", None)),
        abstract(config.index_cache_shape, jnp.bfloat16, P(None, None, "expert", None)),
        abstract((1, 2048), jnp.int32),
        abstract((1,), jnp.int32),
        abstract((1, 2048), jnp.float32),
        abstract((1,), jnp.int32),
        abstract((1, 16), jnp.int32),
        abstract((1,), jnp.int32),
        abstract((1,), jnp.bool_),
    )
    state = Ws32BatchedPrefillState(
        decoder,
        decoder.index_cache_local,
        abstract((), jnp.int32),
        abstract((), jnp.bool_),
    )
    wk = tuple(
        abstract((geometry.dsa_indexer_head_dim, geometry.hidden_size), jnp.float32)
        for _ in config.full_index_slots
    )
    rope = abstract(config.main_rope_table_shape, jnp.bfloat16)
    programs = adapter.build_graph_pair(
        mesh,
        config,
        plan,
        **admission.short_program_options(admission.ROLLED_SHORT_PROFILE),
    )
    inputs = {
        name: (
            abstract((rows,), jnp.int32),
            abstract((), jnp.int32),
            state,
            weights,
            wk,
            rope,
        )
        for name, rows in plan.graph_rows
    }
    if any(
        not isinstance(leaf, jax.ShapeDtypeStruct)
        for values in inputs.values()
        for leaf in jax.tree.leaves(values)
    ):
        raise ValueError("abstract prefill preparation allocated concrete inputs")
    return AbstractPrefillPair(
        programs,
        inputs,
        metadata.manifest["manifest_sha256"],
        metadata.manifest["source"]["inventory_sha256"],
    )


def validate_preserved_pair(
    root: Path, record: Mapping[str, Any], *, repo: Path
) -> dict[str, Any]:
    """After BOTH compiles, bind preserved originals; never approve numerics.

    The future wrapper must perform bounded generation-qualified publication
    and all-host replay. This local check does not replace either protection.
    """
    registration = admission.rolled_registration(repo)
    if set(record.get("programs", {})) != {"prefill_chunk", "prefill_tail"}:
        raise ValueError("rolled compile evidence must preserve exactly both graphs")
    graphs = {}
    for name, pins in registration["graphs"].items():
        saved = record["programs"][name]
        stable = (root / f"{name}.stablehlo.mlir").read_bytes()
        optimized = (root / f"{name}.optimized_hlo.txt").read_bytes()
        if (
            len(stable) != pins["stablehlo_bytes"]
            or sha256(stable).hexdigest() != pins["stablehlo_sha256"]
            or saved["stablehlo_sha256"] != pins["stablehlo_sha256"]
            or not optimized
            or sha256(optimized).hexdigest() != saved["optimized_hlo_sha256"]
        ):
            raise ValueError("rolled preserved graph identity differs")
        admission.validate_short_compiled_memory(
            name,
            saved["compiled_memory"],
            profile=admission.ROLLED_SHORT_PROFILE,
            repo=repo,
        )
        graphs[name] = dict(
            stablehlo_sha256=sha256(stable).hexdigest(),
            optimized_hlo_sha256=sha256(optimized).hexdigest(),
            memory=saved["compiled_memory"],
        )
    return dict(
        graphs=graphs,
        dispatch_evidence="REQUIRES_REVIEWED_COMPILE_ONLY_WORKER_AND_JOURNAL",
        numerical_claim=False,
        performance_claim=False,
        scope="ACTUAL_COMPILER_EVIDENCE_NOT_NUMERICAL_HBM_OR_ADMISSION",
    )
