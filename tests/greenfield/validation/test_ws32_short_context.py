from __future__ import annotations

from pathlib import Path

import numpy as np

from glm_tpu.greenfield.validation.ws32_short_context import (
    compare_ws32_dsa_step,
    compare_ws32_raw_tokens,
    load_ws32_short_context_oracle,
    validate_ws32_cache_probe,
)


TOKEN_DIR = Path(
    "/home/gianl/gcs-models/oracles/greenfield/glm52/short_context/2k/"
    "greenfield_short_context_oracle_20260806T202544155912103Z/oracle"
)
DSA_DIR = Path(
    "/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/2k/"
    "greenfield_short_context_dsa_oracle_recovery_20260806T231905802593249Z/oracle"
)
TOKEN_SHA = "f580c14954bcbd0d973b6fe8158520992a18a1375ed88cff9cceb8e01c7efe19"
DSA_SHA = "71224832652ce61024786d39d43dcbfdc6cde76bf2eff0350531b272f38f4f57"
TOKEN_SUCCESS_SHA = "07700db5a732f04663f0298625bbdbb68a1c73e63652a3aef27f398993e86eec"
DSA_SUCCESS_SHA = "c091d0b56f712eb2f106ee248f69f599586ade0d5c202cbb5b46b8aa115411b2"


def test_ws32_short_context_replays_sealed_oracle_and_refuses_drift() -> None:
    oracle = load_ws32_short_context_oracle(
        TOKEN_DIR,
        DSA_DIR,
        expected_token_manifest_sha256=TOKEN_SHA,
        expected_dsa_manifest_sha256=DSA_SHA,
        expected_token_success_sha256=TOKEN_SUCCESS_SHA,
        expected_dsa_success_sha256=DSA_SUCCESS_SHA,
    )
    assert oracle.prompt_token_ids.shape == (2034,)
    assert oracle.selected_positions.shape == (14, 21, 2048)
    exact = compare_ws32_dsa_step(
        producer_layer_ids=oracle.producer_layer_ids,
        selected_positions=oracle.selected_positions[0, :, None, :],
        selected_valid_counts=oracle.valid_counts[0, :, None],
        selected_scores=oracle.selected_scores[0, :, None, :],
        oracle=oracle,
        step=0,
    )
    assert exact["passed"], exact
    tokens = compare_ws32_raw_tokens(
        oracle.generated_token_ids[:15].tolist(), oracle
    )
    assert tokens["exact_prefix_match"]

    mutated_positions = oracle.selected_positions[0, :, None, :].copy()
    mutated_positions[0, 0, 0] = mutated_positions[0, 0, 1]
    refused = compare_ws32_dsa_step(
        producer_layer_ids=oracle.producer_layer_ids,
        selected_positions=mutated_positions,
        selected_valid_counts=oracle.valid_counts[0, :, None],
        selected_scores=oracle.selected_scores[0, :, None, :],
        oracle=oracle,
        step=0,
    )
    assert not refused["passed"]

    wrong = oracle.generated_token_ids[:3].copy()
    wrong[1] += 1
    assert not compare_ws32_raw_tokens(wrong, oracle)["exact_prefix_match"]

    with np.testing.assert_raises_regex(ValueError, "SUCCESS identity"):
        load_ws32_short_context_oracle(
            TOKEN_DIR,
            DSA_DIR,
            expected_token_manifest_sha256=TOKEN_SHA,
            expected_dsa_manifest_sha256=DSA_SHA,
            expected_token_success_sha256="0" * 64,
            expected_dsa_success_sha256=DSA_SUCCESS_SHA,
        )


def test_ws32_cache_probe_requires_every_layer_and_index_slot() -> None:
    kv = np.ones((3, 8), dtype=np.float32)
    index = np.ones((2, 4), dtype=np.float32)
    exact = validate_ws32_cache_probe(
        position=np.asarray([7], dtype=np.int32),
        kv_rows=kv,
        index_rows=index,
        contract_valid=np.asarray([True]),
        expected_position=7,
        num_layers=3,
        full_indexer_count=2,
        packed_cache_width=8,
        index_width=4,
    )
    assert exact["passed"]
    kv[1] = 0
    refused = validate_ws32_cache_probe(
        position=np.asarray([7], dtype=np.int32),
        kv_rows=kv,
        index_rows=index,
        contract_valid=np.asarray([True]),
        expected_position=7,
        num_layers=3,
        full_indexer_count=2,
        packed_cache_width=8,
        index_width=4,
    )
    assert not refused["passed"]
