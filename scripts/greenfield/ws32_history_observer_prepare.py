"""Original overlay/materializer abstractions for the bounded history observer.

Only metadata is read. Numerical loading still uses the existing selected raw
loader and overlay owner loader; these abstract objects do not certify payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.greenfield.ws32_history_observer import build_program, observer_config


@dataclass(frozen=True)
class PreparedHistoryObserver:
    program: Any
    inputs: tuple[Any, ...]
    config: Any
    materializer: Any
    raw_exact: tuple[Any, ...]
    decoded_exact: tuple[Any, ...]
    exact: tuple[Any, ...]
    overlay: Any
    overlay_tensor_names: tuple[str, ...]
    overlay_bytes_per_chip: int


def bind_selected_views(raw_config: Any, embedding: Any, raw_layers: tuple,
                        overlay_arrays: Mapping[str, Any]) -> tuple[Any, tuple, tuple]:
    """Share original leaves; absent head fields are never executed or loaded."""
    import jax
    from glm_tpu.greenfield.runtime.ws32_decoder import (
        Ws32DecoderWeights, _bind_weight_name_tree, select_ws32_exact_dsa_raw_weights,
        ws32_decoder_weight_names,
    )
    from glm_tpu.greenfield.checkpoint.ws32_strategy_nd_dense import strategy_nd_dense_tensor_names

    config = observer_config(raw_config)
    required_overlay = {name for i in range(3) for name in strategy_nd_dense_tensor_names(i)}
    if len(raw_layers) != 7 or set(overlay_arrays) != required_overlay:
        raise ValueError("history observer requires seven raw layers and original12 overlay tensors")
    raw_names = ws32_decoder_weight_names(raw_config)
    raw_name_leaves = jax.tree.leaves((raw_names.embedding_local, raw_names.layers[:7]))
    raw_values = jax.tree.leaves((embedding, raw_layers))
    if len(raw_name_leaves) != len(raw_values):
        raise ValueError("history observer raw weight tree differs")
    raw_tree = jax.tree.structure((raw_names.embedding_local, raw_names.layers[:7]))
    if raw_tree != jax.tree.structure((embedding, raw_layers)):
        raise ValueError("history observer raw weight branch differs")
    arrays = dict(zip(raw_name_leaves, raw_values, strict=True))
    if set(arrays) & set(overlay_arrays):
        raise ValueError("history observer overlay aliases raw names")
    arrays.update(overlay_arrays)
    names = ws32_decoder_weight_names(config)
    bound_embedding, layers = _bind_weight_name_tree((names.embedding_local, names.layers), arrays)
    # Existing selector only reads layers. None is deliberately NOT a dummy
    # head array; this incomplete container must never enter a decoder builder.
    raw_exact = select_ws32_exact_dsa_raw_weights(
        Ws32DecoderWeights(bound_embedding, layers, None, None), config)
    return config, layers, raw_exact


def prepare(mesh: Any, *, raw: Any, repo: Path) -> PreparedHistoryObserver:
    """Construct observer/materializer inputs using verified original overlay metadata."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from glm_tpu.greenfield.checkpoint.ws32_strategy_nd_dense import (
        _LOCAL_CONTRACT, strategy_nd_dense_tensor_names, verify_ws32_strategy_nd_dense_overlay,
    )
    from glm_tpu.greenfield.runtime.ws32_decoder import build_ws32_exact_dsa_materializer_program

    env = json.loads((repo / "configs/greenfield-ws32-batched-acquisition.json").read_text())["environment"]
    prefix = "GLM_GREENFIELD_WS32_STRATEGY_ND_DENSE_OVERLAY_"
    overlay = verify_ws32_strategy_nd_dense_overlay(Path(env[prefix + "ROOT"]),
        expected_manifest_sha256=env[prefix + "MANIFEST_SHA"],
        expected_manifest_file_sha256=env[prefix + "MANIFEST_FILE_SHA"],
        expected_success_file_sha256=env[prefix + "SUCCESS_FILE_SHA"])
    def abstract(shape: tuple, dtype: Any, spec: Any = P()) -> Any:
        return jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(mesh, spec))
    arrays = {name: abstract(shape, dtype, P(*spec))
              for i in range(3)
              for name, (_, dtype, shape, spec) in zip(
                  strategy_nd_dense_tensor_names(i), _LOCAL_CONTRACT, strict=True)}
    inputs = raw.inputs["canonical_b128"]
    config, layers, raw_exact = bind_selected_views(raw.config, inputs[5], inputs[6], arrays)
    materializer = build_ws32_exact_dsa_materializer_program(mesh, config)
    decoded = jax.eval_shape(materializer.decode, raw_exact)
    # Promotion names one decoded query four times, but as program outputs XLA
    # materializes four distinct buffers; budget four owners per slot.
    exact = jax.eval_shape(materializer.promote, decoded)
    program = build_program(mesh, config)
    observer_inputs = (abstract((1,), jnp.int32), abstract((1,), jnp.int32),
        abstract((1,), jnp.int32), inputs[3], inputs[4].kv, inputs[4].repaired,
        inputs[5], layers, exact, inputs[8], inputs[9])
    byte_counts = {sum(v["byte_count"] for i in range(3)
                      for v in overlay.records[(i, e, f)]["tensors"].values())
                   for e in range(8) for f in range(4)}
    if len(byte_counts) != 1:
        raise ValueError("history observer overlay owner payload sizes differ")
    return PreparedHistoryObserver(program, observer_inputs, config, materializer,
        raw_exact, decoded, exact, overlay, tuple(sorted(arrays)), byte_counts.pop())
