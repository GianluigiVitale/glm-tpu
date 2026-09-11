"""Explicit §26 task smoke; never cross-engine bits or model-card parity.

The worker and sealer use the same criterion on their own original arrays.
Historical §21 profiles remain unchanged. No model execution is imported.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from .long_context_oracle import load_legacy_bench_module
from .ws32_short_context import compare_ws32_dsa_within_engine, compare_ws32_raw_tokens


PROFILE = "ws32_b128_b114_8k_cap8192_delivery_s26_v1"
CONTRACT = "s26_passkey_task_not_incidental_continuation_v1"
TOKEN_MANIFEST_SHA256 = "e4fbcbdbf0fc8b1969e2f82ee457ab1563db4a8b37d2dea2bc4d1e828a13acf2"
PRODUCERS = (0, 1, 2, *range(6, 78, 4))
PREREQUISITES = {
    "docs/artifacts/prefill-canonical-short-db610-sealed-20260909.json": "952c30cde60768c572d13d8dfe192cc44c0ea20b9974202a687901be235de91c",
}
BENCH_SHA256 = {
    "glm_longctx.py": "45ddc6bc97e701f41f3870da13c5eba8039d907a494a0b6cf17270363ac7efd8",
    "dsa_throughput.py": "f7216be2864c03ffb5b079c1ca80396c59b2727a52dbf6298052bdef789696bb",
    "extract.py": "e3f03914974ff1ef0839d3690e076b3d86e9894c70fce07ea407be07581051a5",
    "provenance.py": "d5364547f8b0c01ba03aaf13bffaed45410587c7a1828ebc803356f9bd54daed",
    "engine.py": "93a1a05dac9839bf1d10d2e04442e5677767360381fa9e6cd96f683778f1b286",
}


def require_prerequisites(repo: Path) -> None:
    for name, digest in PREREQUISITES.items():
        path = repo / name
        if path.is_symlink() or sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("delivery corrected-short prerequisite differs")


def token_result(
    observed_tokens: list[int], oracle: Any, *, tokenizer_root: Path | None
) -> dict[str, Any]:
    """Exact extracted answer from first20 tokens, using the pinned benchmark.

    The token oracle still authenticates the prompt, gold and tokenizer. Its
    continuation is retained as a diagnostic, never the acceptance criterion.
    """
    manifest = oracle.token_manifest
    if (
        manifest.get("manifest_sha256") != TOKEN_MANIFEST_SHA256
        or int(oracle.prompt_token_ids.size) != 8155
        or int(oracle.generated_token_ids.size) != 20
        or type(observed_tokens) is not list
        or len(observed_tokens) < 20
        or any(type(token) is not int or not 0 <= token < 154880 for token in observed_tokens)
    ):
        raise ValueError("delivery task requires the sealed8K prompt and valid token IDs")
    if tokenizer_root is None:
        raise ValueError("delivery task requires its pinned local tokenizer")
    tokenizer_root = Path(tokenizer_root)
    files = manifest["tokenizer_files"]
    if set(files) != {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}:
        raise ValueError("delivery tokenizer inventory differs")
    for name, record in files.items():
        path = tokenizer_root / name
        if (
            path.is_symlink()
            or path.stat().st_size != record["byte_count"]
            or sha256(path.read_bytes()).hexdigest() != record["sha256"]
        ):
            raise ValueError("delivery tokenizer differs from sealed input")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_root), local_files_only=True, trust_remote_code=False
    )
    bench = load_legacy_bench_module(
        "glm_longctx",
        pinned_files={name: {"sha256": digest} for name, digest in BENCH_SHA256.items()},
    )
    gold = manifest["source"]["gold"]
    if type(gold) is not str or not gold.isdigit():
        raise ValueError("delivery task has no exact gold passkey")
    # Exercise the actual tooling before any expensive run and on every replay.
    reference_text = tokenizer.decode(oracle.generated_token_ids.tolist(), skip_special_tokens=True)
    if bench.extract_passkey(reference_text) != gold:
        raise ValueError("delivery tokenizer/extractor does not reproduce sealed gold")
    text = tokenizer.decode(observed_tokens[:20], skip_special_tokens=True)
    extracted = bench.extract_passkey(text)
    return {
        "contract": CONTRACT,
        "kind": "passkey",
        "criterion": "extract_passkey(detok(first20 tokens)) == sealed gold",
        "gold": gold,
        "detokenised": text,
        "extracted_passkey": extracted,
        "passkey_matches_gold": extracted == gold,
        "observed_token_count": len(observed_tokens),
        "legacy_diagnostic": compare_ws32_raw_tokens(observed_tokens[:20], oracle),
        "model_card_quality_claim": False,
    }


def dsa_result(**observations: Any) -> dict[str, Any]:
    """Selected-row structure plus exact production producer/count geometry.

    This does not recompute scores for unselected keys. Own-score top-k
    semantics additionally rely on the frozen kernel/source and its tests.
    """
    expected = np.asarray(PRODUCERS, np.int32)
    observations = {**observations, "expected_producer_layer_ids": expected}
    result = compare_ws32_dsa_within_engine(**observations)
    positions = np.asarray(observations["selected_positions"])
    counts = np.asarray(observations["selected_valid_counts"])
    scores = np.asarray(observations["selected_scores"])
    producers = np.asarray(observations["producer_layer_ids"])
    geometry_ok = (
        producers.shape == (21,) and producers.dtype == np.int32
        and positions.shape == (21, 1, 2048) and positions.dtype == np.int32
        and scores.shape == positions.shape and scores.dtype == np.float32
        and counts.shape == (21, 1) and counts.dtype == np.int32
        and np.all(counts == min(2048, observations["decode_position"] + 1))
    )
    return {
        **result,
        "passed": bool(result["passed"] and geometry_ok),
        "validation_contract": CONTRACT,
        "production_geometry_passed": bool(geometry_ok),
        "scope": "SELECTED_ROW_STRUCTURE;UNSELECTED_SCORE_ROWS_NOT_RECOMPUTED",
    }
