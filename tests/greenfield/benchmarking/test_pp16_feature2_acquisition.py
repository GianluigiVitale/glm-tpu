from __future__ import annotations

from pathlib import Path

import pytest

from glm_tpu.greenfield.benchmarking.pp16_feature2_acquisition import (
    inspect_feature2_event1_lineage,
)


TOKEN_ORACLE = Path(
    "/home/gianl/gcs-models/oracles/greenfield/glm52/short_context/8k/"
    "greenfield_short_context_oracle_8k_20260807T172307269147351Z/oracle"
)
DSA_ORACLE = Path(
    "/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/8k/"
    "greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z/"
    "oracle"
)
LAYER1_INTERNAL = Path(
    "/home/gianl/gcs-models/oracles/greenfield/glm52/dsa_internals/8k/layer1/"
    "greenfield_layer1_dsa_internal_comparison_20260808T115135394251231Z/"
    "internals.npz"
)
DB529_INTERNAL = Path(
    "/home/gianl/glm-run/"
    "greenfield_layer0_dsa_scorer_association_20260810T164030202890642Z/"
    "inputs/internal"
)


@pytest.mark.skipif(
    not all(
        path.exists()
        for path in (TOKEN_ORACLE, DSA_ORACLE, LAYER1_INTERNAL, DB529_INTERNAL)
    ),
    reason="protected feature2 event-1 lineage is unavailable",
)
def test_feature2_event1_lineage_is_exact_and_cross_sealed() -> None:
    report = inspect_feature2_event1_lineage(
        token_oracle_dir=TOKEN_ORACLE,
        dsa_oracle_dir=DSA_ORACLE,
        layer1_internal_reference=LAYER1_INTERNAL,
        db529_internal_dir=DB529_INTERNAL,
    )
    assert report["event_index"] == 1
    assert report["producer_layer_id"] == 1
    assert report["decode_position"] == 8155
    assert report["expected_valid_count"] == 2048
    assert report["target_source"] == (
        "sealed_layer1_reference_and_short_context_dsa_oracle"
    )
    assert report["db529_mechanism_source_mismatch_counts"] == {
        "accepted__current_key": 128,
        "accepted__head_weights": 32,
        "accepted__normalized_hidden": 3960,
        "accepted__q_a_state": 942,
        "accepted__query": 4096,
    }
    assert set(report) == {
        "db529_internal_contract_sha256",
        "db529_internal_tensor_sha256",
        "db529_mechanism_source_mismatch_counts",
        "decode_position",
        "dsa_oracle_manifest_sha256",
        "event_index",
        "expected_positions_sha256",
        "expected_scores_sha256",
        "expected_valid_count",
        "layer1_current_key_sha256",
        "layer1_head_weights_sha256",
        "layer1_internal_reference_sha256",
        "layer1_normalized_hidden_sha256",
        "layer1_q_a_state_sha256",
        "layer1_query_sha256",
        "producer_layer_id",
        "target_source",
        "token_oracle_manifest_sha256",
    }
