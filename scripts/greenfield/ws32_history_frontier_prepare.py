"""Selected production abstractions; no payload reads, placements or dispatch.

This raw-prefill preparation does NOT prepare the exact-decode observer. That
observer additionally requires the original three-layer StrategyND overlay and
four exact-DSA materializations, and must reproduce retained step0 observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.greenfield.ws32_history_frontier import (
    HistoryCaches, LAYERS, PRODUCERS, build_program,
)
from scripts.greenfield.ws32_rolled_prefill_compile import read_metadata


@dataclass(frozen=True)
class PreparedHistoryFrontier:
    programs: dict[str, Any]
    inputs: dict[str, tuple[Any, ...]]
    config: Any
    tensor_names: tuple[str, ...]
    payload_bytes_per_chip: int
    manifest_sha256: str
    cache_bytes_per_chip_per_branch: int
    wk_bytes_per_chip: int


def prepare(mesh: Any, *, repo: Path) -> PreparedHistoryFrontier:
    """Bind both original schedules to only embedding and raw layers0..6.

    Byte counts describe persistent selected operands, NOT runtime peak HBM.
    Both branches share weights/WK/RoPE; each owns all three cache families.
    No speculative fix or changed production source is selected here.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from glm_tpu.greenfield.types import ModelGeometry
    from glm_tpu.greenfield.checkpoint.ws32_layer_subset import ws32_expected_layer_names
    from glm_tpu.greenfield.runtime.ws32_decoder import (
        Ws32DecoderConfig, _bind_weight_name_tree, ws32_decoder_weight_names,
    )

    metadata = read_metadata(repo, full_canonical=True)
    geometry = ModelGeometry.from_dict(metadata.manifest["geometry"])
    config = Ws32DecoderConfig(geometry, 8192, host_main_rope_table=True)
    names = ws32_decoder_weight_names(config)
    selected_names = (names.embedding_local, names.layers[:7])
    selected = frozenset(jax.tree.leaves(selected_names))
    expected = frozenset(("model.embed_tokens.weight",)).union(
        *(ws32_expected_layer_names(geometry, i) for i in LAYERS))
    if selected != expected:
        raise ValueError("history frontier selection is not seven complete raw layers plus embedding")
    schemas = {v["name"]: v for v in metadata.manifest["tensor_schema"]}
    if len(schemas) != len(metadata.manifest["tensor_schema"]):
        raise ValueError("history frontier duplicate tensor schema")
    dtypes = {"BF16": jnp.bfloat16, "F32": jnp.float32, "U8": jnp.uint8}

    def abstract(shape: tuple[int, ...], dtype: Any, spec: Any = P()) -> Any:
        return jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(mesh, spec))

    arrays = {name: abstract(tuple(schemas[name]["global_shape"]),
                            dtypes[schemas[name]["dtype"]], P(*schemas[name]["partition_spec"]))
              for name in selected}
    embedding, layers = _bind_weight_name_tree(selected_names, arrays)
    def cache(width: int) -> Any:
        return abstract((8, config.page_count, 64, width), jnp.bfloat16,
                        P("expert", None, None, None))
    caches = HistoryCaches(tuple(cache(640) for _ in LAYERS),
                           tuple(cache(128) for _ in PRODUCERS),
                           tuple(cache(128) for _ in PRODUCERS))
    wk = tuple(abstract((128, geometry.hidden_size), jnp.float32) for _ in PRODUCERS)
    payloads = {sum(t.byte_count for t in plan.tensors if t.name in selected)
                for plan in metadata.plans}
    if len(payloads) != 1:
        raise ValueError("history frontier selected owner byte counts differ")
    programs, inputs = {}, {}
    for branch, canonical in (("canonical", True), ("live32", False)):
        for rows in (128, 114):
            name = f"{branch}_b{rows}"
            programs[name] = build_program(mesh, config, block_rows=rows, canonical_dense=canonical)
            inputs[name] = (abstract((rows,), jnp.int32), abstract((), jnp.int32),
                abstract((), jnp.int32), abstract((1, config.page_count), jnp.int32),
                caches, embedding, layers, wk,
                abstract(config.main_rope_table_shape, jnp.bfloat16), abstract((), jnp.bool_))
    return PreparedHistoryFrontier(programs, inputs, config, tuple(sorted(selected)),
        payloads.pop(), metadata.manifest["manifest_sha256"],
        config.page_count * 64 * 2 * (7 * 640 + 8 * 128),
        len(PRODUCERS) * 128 * geometry.hidden_size * 4)
