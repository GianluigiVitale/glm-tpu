"""Production metadata/abstract inputs for the diagnostic, without weight reads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.greenfield.ws32_dense_frontier_program import build_program
from scripts.greenfield.ws32_rolled_prefill_compile import read_metadata


@dataclass(frozen=True)
class PreparedDenseFrontier:
    program: Any
    inputs: tuple[Any, ...]
    config: Any
    tensor_names: tuple[str, ...]
    payload_bytes_per_chip: int
    manifest_sha256: str


def prepare(mesh: Any, *, repo: Path) -> PreparedDenseFrontier:
    """Use original production geometry and final-layout leaves, all abstract.

    No head weights, full checkpoint load, WK execution or TPU launch. Estimated
    payload bytes are not HBM admission: actual allocations remain mandatory.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from glm_tpu.greenfield.types import ModelGeometry
    from glm_tpu.greenfield.runtime.ws32_decoder import (
        Ws32DecoderConfig, _bind_weight_name_tree, ws32_decoder_weight_names,
    )

    metadata = read_metadata(repo)
    geometry = ModelGeometry.from_dict(metadata.manifest["geometry"])
    config = Ws32DecoderConfig(geometry, 8192, host_main_rope_table=True)
    names = ws32_decoder_weight_names(config)
    selected_names = (names.embedding_local, names.layers[:2])
    selected = frozenset(jax.tree.leaves(selected_names))
    schemas = {item["name"]: item for item in metadata.manifest["tensor_schema"]}
    dtypes = {"BF16": jnp.bfloat16, "F32": jnp.float32, "U8": jnp.uint8}

    def abstract(shape: tuple[int, ...], dtype: Any, spec: Any = P()) -> Any:
        return jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(mesh, spec))

    arrays = {name: abstract(tuple(schemas[name]["global_shape"]),
                            dtypes[schemas[name]["dtype"]], P(*schemas[name]["partition_spec"]))
              for name in selected}
    embedding, layers = _bind_weight_name_tree(selected_names, arrays)
    caches = tuple(tuple(abstract((8, 16, 64, width), jnp.bfloat16,
                                 P("expert", None, None, None))
                         for width in (640, 128, 128)) for _ in range(2))
    wk = tuple(abstract((geometry.dsa_indexer_head_dim, geometry.hidden_size), jnp.float32)
               for _ in range(2))
    payloads = {sum(t.byte_count for t in plan.tensors if t.name in selected)
                for plan in metadata.plans}
    if len(payloads) != 1:
        raise ValueError("dense frontier selected ownership byte counts differ")
    inputs = (abstract((128,), jnp.int32), abstract((), jnp.int32),
              abstract((), jnp.int32), abstract((1, 16), jnp.int32), caches,
              embedding, layers, wk, abstract(config.main_rope_table_shape, jnp.bfloat16))
    return PreparedDenseFrontier(build_program(mesh, config), inputs, config,
                                 tuple(sorted(selected)), payloads.pop(),
                                 metadata.manifest["manifest_sha256"])
