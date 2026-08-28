from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from glm_tpu.greenfield.benchmarking.pp16_feature2_prefill import (
    build_feature2_prefill_graph,
    load_feature2_prefill_inputs,
    validate_feature2_prefill_graph,
)
from glm_tpu.greenfield.benchmarking.pp16_feature_sharded_state import (
    Feature2TensorRead,
    derive_feature2_tensor_allowlist,
    read_feature2_owner_headers,
)
from glm_tpu.greenfield.errors import BenchmarkValidationError


REAL_PP16_FEATURE2_RUNTIME = Path(
    "/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/"
    "PP16_LP2/greenfield_runtime_feature_qkv_direct_pp16_"
    "20260827T164842844148623Z"
)
REAL_8K_ORACLE = Path(
    "/home/gianl/gcs-models/oracles/greenfield/glm52/short_context/8k/"
    "greenfield_short_context_oracle_8k_20260807T172307269147351Z/oracle"
)


def _real_allowlist():
    manifest = json.loads(
        (REAL_PP16_FEATURE2_RUNTIME / "runtime_manifest.json").read_text()
    )
    headers = read_feature2_owner_headers(REAL_PP16_FEATURE2_RUNTIME, manifest)
    return derive_feature2_tensor_allowlist(manifest, headers)


def _graph():
    return build_feature2_prefill_graph(
        _real_allowlist(), load_feature2_prefill_inputs(REAL_8K_ORACLE)
    )


@pytest.mark.skipif(
    not REAL_PP16_FEATURE2_RUNTIME.is_dir() or not REAL_8K_ORACLE.is_dir(),
    reason="protected PP16 runtime or 8K oracle is unavailable",
)
def test_real_feature2_prefill_graph_is_causal_and_evidence_bound() -> None:
    graph = _graph()
    assert validate_feature2_prefill_graph(graph) == {
        "context_length": 8156,
        "graph_sha256": graph.graph_sha256,
        "local_group": [0, 1],
        "node_count": 34,
        "selected_read_count": 78,
        "selected_weight_count": 39,
        "valid_rows_by_chunk": [2048, 2048, 2048, 2012],
    }
    assert graph.inputs.current_token_id == 220
    assert [chunk.start for chunk in graph.inputs.chunks] == [0, 2048, 4096, 6144]
    assert [chunk.stop for chunk in graph.inputs.chunks] == [2048, 4096, 6144, 8156]
    assert not any(tensor.shape == (8156, 6144) for tensor in graph.tensors)
    scorer = next(
        node for node in graph.nodes if node.name == "layer1_event1_scorer"
    )
    assert "layer1_index_cache_4" in scorer.inputs
    current = next(
        node for node in graph.nodes if node.name == "layer1_current_row"
    )
    assert "layer1_normalized_3" in current.inputs
    assert "current_position" in current.inputs


@pytest.mark.skipif(
    not REAL_PP16_FEATURE2_RUNTIME.is_dir() or not REAL_8K_ORACLE.is_dir(),
    reason="protected PP16 runtime or 8K oracle is unavailable",
)
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("accepted_input", "tensor/version"),
        ("missing_key_dependency", "causal node/cache"),
        ("broken_cache_carry", "causal node/cache"),
        ("wrong_current_chunk", "causal node/cache"),
        ("attention_output", "causal node/cache"),
        ("dead_rows", "decode_rows"),
        ("short_context", "context_length"),
        ("wrong_tail", "chunk"),
        ("full_group", "local_group"),
        ("wrong_ties", "tie_policy"),
        ("full_hidden", "tensor/version"),
        ("dense1_weight", "weight-role"),
        ("range_sha", "selected range"),
        ("position_sha", "source tensors"),
        ("partial_carried_liveness", "carried liveness"),
    ],
)
def test_feature2_prefill_graph_refuses_semantic_mutations(
    mutation: str, message: str
) -> None:
    graph = _graph()
    if mutation == "accepted_input":
        tensors = list(graph.tensors)
        tensors[0] = replace(tensors[0], name="accepted_topk")
        changed = replace(graph, tensors=tuple(tensors))
    elif mutation == "missing_key_dependency":
        nodes = list(graph.nodes)
        nodes[-1] = replace(
            nodes[-1],
            inputs=tuple(
                value
                for value in nodes[-1].inputs
                if value != "layer1_index_cache_4"
            ),
        )
        changed = replace(graph, nodes=tuple(nodes))
    elif mutation == "broken_cache_carry":
        nodes = list(graph.nodes)
        target = next(
            i
            for i, node in enumerate(nodes)
            if node.name == "layer0_attention_indexer_2"
        )
        nodes[target] = replace(
            nodes[target],
            inputs=tuple(
                "layer0_kv_cache_0" if value == "layer0_kv_cache_2" else value
                for value in nodes[target].inputs
            ),
        )
        changed = replace(graph, nodes=tuple(nodes))
    elif mutation == "wrong_current_chunk":
        nodes = list(graph.nodes)
        target = next(
            i for i, node in enumerate(nodes) if node.name == "layer1_current_row"
        )
        nodes[target] = replace(
            nodes[target],
            inputs=tuple(
                "layer1_normalized_2"
                if value == "layer1_normalized_3"
                else value
                for value in nodes[target].inputs
            ),
        )
        changed = replace(graph, nodes=tuple(nodes))
    elif mutation == "attention_output":
        nodes = list(graph.nodes)
        nodes[-1] = replace(nodes[-1], operation="layer1_attention_output")
        changed = replace(graph, nodes=tuple(nodes))
    elif mutation == "dead_rows":
        changed = replace(graph, decode_rows=32)
    elif mutation == "short_context":
        changed = replace(graph, context_length=2048)
    elif mutation == "wrong_tail":
        chunks = list(graph.inputs.chunks)
        chunks[-1] = replace(chunks[-1], valid_rows=2048, stop=8192)
        changed = replace(graph, inputs=replace(graph.inputs, chunks=tuple(chunks)))
    elif mutation == "full_group":
        changed = replace(graph, local_group=tuple(range(32)))
    elif mutation == "wrong_ties":
        changed = replace(graph, tie_policy="score_only")
    elif mutation == "full_hidden":
        tensors = list(graph.tensors)
        target = next(
            i for i, tensor in enumerate(tensors) if tensor.name == "embedding_chunk_0"
        )
        tensors[target] = replace(tensors[target], shape=(8156, 6144))
        changed = replace(graph, tensors=tuple(tensors))
    elif mutation == "dense1_weight":
        roles = list(graph.weight_roles)
        role = roles[2]
        roles[2] = replace(
            role,
            weights=role.weights[:-1] + ("dense.slot_01.down.weight_bits",),
        )
        changed = replace(graph, weight_roles=tuple(roles))
    elif mutation == "range_sha":
        reads = list(graph.selected_reads)
        reads[0] = Feature2TensorRead(
            **{**reads[0].to_dict(), "sha256": "0" * 64}
        )
        changed = replace(graph, selected_reads=tuple(reads))
    elif mutation == "partial_carried_liveness":
        nodes = list(graph.nodes)
        target = next(
            i
            for i, node in enumerate(nodes)
            if node.name == "layer1_carried_liveness_1"
        )
        nodes[target] = replace(
            nodes[target],
            inputs=tuple(
                value
                for value in nodes[target].inputs
                if value != "valid_mask_1"
            ),
        )
        changed = replace(graph, nodes=tuple(nodes))
    else:
        sources = list(graph.inputs.sources)
        sources[1] = replace(sources[1], sha256="0" * 64)
        changed = replace(
            graph, inputs=replace(graph.inputs, sources=tuple(sources))
        )
    with pytest.raises(BenchmarkValidationError, match=message):
        validate_feature2_prefill_graph(changed)
