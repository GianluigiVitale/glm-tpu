"""Spec §23.5: the long-context correctness contract (L7 passkey, L8 e0)."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts/greenfield/run_short_decoder_ws32.py"
SEALER = ROOT / "scripts/greenfield/seal_short_decoder_ws32.py"


def _events(width: int = 8, events: int = 3, live: int = 3):
    positions = np.full((events, 1, width), -1, dtype=np.int32)
    scores = np.full((events, 1, width), -np.inf, dtype=np.float32)
    counts = np.zeros((events, 1), dtype=np.int32)
    chosen = [10, 4, 7][:live]
    values = [0.9, 0.5, 0.5][:live]
    order = sorted(range(live), key=lambda i: (-values[i], chosen[i]))
    for event in range(events):
        positions[event, 0, :live] = [chosen[i] for i in order]
        scores[event, 0, :live] = [values[i] for i in order]
        counts[event, 0] = live
    return positions, counts, scores, np.arange(events, dtype=np.int32)


def test_within_engine_contract_accepts_a_conforming_observation() -> None:
    from glm_tpu.greenfield.validation import compare_ws32_dsa_within_engine

    positions, counts, scores, ids = _events()
    result = compare_ws32_dsa_within_engine(
        producer_layer_ids=ids, selected_positions=positions,
        selected_valid_counts=counts, selected_scores=scores,
        decode_position=100, step=0, expected_producer_layer_ids=ids,
    )
    assert result["passed"] is True
    assert result["cross_oracle"] is False
    assert result["contract"] == "within_engine_only"


def test_within_engine_contract_refuses_each_violation() -> None:
    """It is not a formality: each of these is a real device-contract failure."""
    from glm_tpu.greenfield.validation import compare_ws32_dsa_within_engine

    positions, counts, scores, ids = _events()

    def check(**overrides):
        arguments = dict(
            producer_layer_ids=ids, selected_positions=positions,
            selected_valid_counts=counts, selected_scores=scores,
            decode_position=100, step=0, expected_producer_layer_ids=ids,
        )
        arguments.update(overrides)
        return compare_ws32_dsa_within_engine(**arguments)["passed"]

    swapped = positions.copy()
    swapped[0, 0, 0], swapped[0, 0, 1] = swapped[0, 0, 1], swapped[0, 0, 0]
    assert check(selected_positions=swapped) is False, "tie order must be canonical"

    dirty = scores.copy()
    dirty[1, 0, 5] = 0.0
    assert check(selected_scores=dirty) is False, "the padding tail must be the sentinel"

    assert check(decode_position=5) is False, "positions must be in range"

    repeated = positions.copy()
    repeated[0, 0, 1] = repeated[0, 0, 0]
    assert check(selected_positions=repeated) is False, "positions must be distinct"

    assert check(expected_producer_layer_ids=np.asarray([0, 1, 9], np.int32)) is False

    overflow = counts.copy()
    overflow[0, 0] = 99
    assert check(selected_valid_counts=overflow) is False


def _runner():
    specification = importlib.util.spec_from_file_location("ws32_runner_lc", RUNNER)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_the_long_context_token_result_never_claims_raw_token_exactness() -> None:
    """§23.5 forbids it: the legacy stored TEXT at these lengths, not ids."""

    class _Oracle:
        kind = "e0"
        generated_token_ids = np.asarray([1, 2, 3], dtype=np.int32)
        gold = None

    result = _runner()._long_context_token_result([1, 2, 3], _Oracle(), tokenizer_root=None)
    assert "exact_prefix_match" not in result
    assert result["passkey_matches_gold"] is None
    assert result["legacy_diagnostic"]["legacy_ids_match"] is True
    assert "diagnostic" in result["legacy_diagnostic"]["note"]


def test_the_runner_refuses_an_adjudication_record_in_long_context_mode() -> None:
    """There is nothing at these lengths for a §21.2 record to adjudicate."""
    source = RUNNER.read_text(encoding="utf-8")
    assert "WS32 long-context runs bind no §21.2 adjudication record" in source
    assert "WS32 long-context seals bind no §21.2 adjudication record" in (
        SEALER.read_text(encoding="utf-8")
    )


def test_the_sealer_classification_makes_no_cross_oracle_claim() -> None:
    """PASSKEY_EXACT/NO_CROSS_ORACLE, and never RAW_TOKENS_EXACT."""
    source = SEALER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {"PASSKEY_EXACT", "NO_CROSS_ORACLE", "NO_CORRECTNESS_ORACLE"} <= literals

    start = source.index('basis = ["DSA_WITHIN_ENGINE_EXACT"')
    end = source.index('summary["classification"]', start)
    block = source[start:end]
    assert "RAW_TOKENS_EXACT" in block, "the short-context branch must still be present"
    long_branch = block[: block.index("else:")]
    assert "RAW_TOKENS_EXACT" not in long_branch
    assert "DSA_CROSS_ORACLE_EXACT_ALL_EVENTS" not in long_branch


def test_the_sealer_recomputes_the_runners_own_token_rule() -> None:
    """Two copies of a correctness criterion drift; the sealer imports one."""
    source = SEALER.read_text(encoding="utf-8")
    start = source.index("def _long_context_token_result(")
    end = source.index("def _rank0_dsa_arrays(")
    body = source[start:end]
    assert "run_short_decoder_ws32.py" in body
    assert "_long_context_token_result(" in body
