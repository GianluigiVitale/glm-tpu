"""Independent terminal validation for the bounded WS32 DSA discriminator."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..benchmarking.dsa_association import validate_dsa_association_hlo
from ..benchmarking.ws32_dsa_association import (
    classify_ws32_dsa_query_head_contract,
    validate_ws32_dsa_component_hlo,
)
from .layer0_dsa_association import (
    compare_dsa_association_scores,
    inspect_greenfield_layer0_dsa_internal_observation,
    inspect_layer0_dsa_association_input,
    pack_stage_local_index_keys,
)
from .prompt_index_cache import inspect_legacy_prompt_index_cache


ARTIFACT_KIND = "greenfield_ws32_layer0_dsa_association"
SOURCE_DB_RUN_ID = 529
SOURCE_CODE_HASH = "0cd3db063852630c6309f1966020ba3085aa9877"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _comparison(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    expected = np.ascontiguousarray(expected)
    actual = np.ascontiguousarray(actual)
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        raise ValueError("WS32 DSA terminal comparison geometry drifted")
    if np.issubdtype(expected.dtype, np.inexact) and (
        not np.isfinite(expected).all() or not np.isfinite(actual).all()
    ):
        raise ValueError("WS32 DSA comparison contains a non-finite value")
    mismatch_count = int(np.count_nonzero(actual != expected))
    if np.issubdtype(expected.dtype, np.inexact):
        difference = np.abs(
            actual.astype(np.float64) - expected.astype(np.float64)
        )
        max_abs = float(difference.max(initial=0.0))
        mean_abs = float(difference.mean())
    else:
        max_abs = float(mismatch_count != 0)
        mean_abs = float(mismatch_count / max(1, expected.size))
    return {
        "actual_sha256": _sha256_array(actual),
        "elementwise_exact": mismatch_count == 0,
        "expected_sha256": _sha256_array(expected),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "mismatch_count": mismatch_count,
        "shape": list(expected.shape),
    }


def _require_digest(
    value: Any, *, field: str, lengths: tuple[int, ...] = (64,)
) -> None:
    if not isinstance(value, str) or len(value) not in lengths or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"WS32 DSA {field} is not a SHA-256 digest")


def _require_positive_float(value: Any, *, field: str) -> None:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"WS32 DSA {field} is not a positive finite float")


def _require_exact_comparison_schema(
    value: Any,
    *,
    field: str,
) -> None:
    keys = {
        "actual_sha256",
        "elementwise_exact",
        "expected_sha256",
        "max_abs",
        "mean_abs",
        "mismatch_count",
        "shape",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"WS32 DSA {field} comparison schema drifted")
    _require_digest(value["actual_sha256"], field=f"{field} actual hash")
    _require_digest(value["expected_sha256"], field=f"{field} expected hash")
    if type(value["elementwise_exact"]) is not bool or (
        type(value["mismatch_count"]) is not int
        or value["mismatch_count"] < 0
        or not isinstance(value["shape"], list)
        or any(type(item) is not int or item < 0 for item in value["shape"])
    ):
        raise ValueError(f"WS32 DSA {field} comparison types drifted")
    for name in ("max_abs", "mean_abs"):
        number = value[name]
        if type(number) is not float or not math.isfinite(number) or number < 0:
            raise ValueError(f"WS32 DSA {field} {name} drifted")


def _require_selection_schema(value: Any, *, field: str) -> None:
    keys = {
        "actual_cutoff_score",
        "actual_top_positions",
        "context",
        "expected_cutoff_score",
        "passed",
        "position_mismatch_count",
        "score_all_finite",
        "score_correlation",
        "score_max_abs",
        "score_mean_abs",
        "score_mean_signed",
        "score_p99_abs",
        "selected_order_exact",
        "selected_set_exact",
        "swapped_position_count",
        "top_k",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"WS32 DSA {field} selection schema drifted")
    for name in (
        "passed",
        "score_all_finite",
        "selected_order_exact",
        "selected_set_exact",
    ):
        if type(value[name]) is not bool:
            raise ValueError(f"WS32 DSA {field} {name} is not boolean")
    for name in (
        "context",
        "position_mismatch_count",
        "swapped_position_count",
        "top_k",
    ):
        if type(value[name]) is not int or value[name] < 0:
            raise ValueError(f"WS32 DSA {field} {name} is not an integer")
    positions = value["actual_top_positions"]
    if not isinstance(positions, list) or any(
        type(position) is not int or position < 0 for position in positions
    ):
        raise ValueError(f"WS32 DSA {field} top positions drifted")
    for name in (
        "actual_cutoff_score",
        "expected_cutoff_score",
        "score_correlation",
        "score_max_abs",
        "score_mean_abs",
        "score_mean_signed",
        "score_p99_abs",
    ):
        if type(value[name]) is not float or not math.isfinite(value[name]):
            raise ValueError(f"WS32 DSA {field} {name} is not finite float")


def _require_simple_graph_schema(value: Any, *, field: str) -> None:
    if not isinstance(value, dict) or set(value) != {
        "contract",
        "optimized_hlo",
        "stablehlo",
    }:
        raise ValueError(f"WS32 DSA {field} graph schema drifted")


def inspect_ws32_dsa_source_summary(
    path: Path,
    *,
    expected_sha256: str,
    association_manifest_sha256: str,
    prompt_cache_manifest_sha256: str,
    internal_tensor_sha256: str,
) -> dict[str, Any]:
    """Bind the bounded probe to the protected DB529 accepted source event."""

    path = Path(path)
    _require_digest(expected_sha256, field="source summary hash")
    if _sha256_file(path) != expected_sha256:
        raise ValueError("WS32 DSA source summary identity drifted")
    summary = json.loads(path.read_text())
    if not isinstance(summary, dict) or set(summary) != {
        "candidate_restored",
        "claim_scope",
        "code_hash",
        "elapsed_seconds",
        "results_db_run_id",
        "runner",
        "status",
    }:
        raise ValueError("WS32 DSA source summary schema drifted")
    runner = summary["runner"]
    if (
        summary["status"] != "SUCCESS"
        or summary["code_hash"] != SOURCE_CODE_HASH
        or summary["results_db_run_id"] != SOURCE_DB_RUN_ID
        or summary["candidate_restored"] is not True
        or not isinstance(runner, dict)
        or runner.get("status") != "SUCCESS"
        or runner.get("code_hash") != SOURCE_CODE_HASH
        or runner.get("candidate_restored") is not True
        or runner.get("default_precision_candidate", {})
        .get("score_delta", {})
        .get("elementwise_exact")
        is not True
        or runner.get("default_precision_candidate", {})
        .get("comparison", {})
        .get("passed")
        is not True
    ):
        raise ValueError("WS32 DSA source summary verdict drifted")
    inputs = runner.get("inputs")
    if not isinstance(inputs, dict) or (
        inputs.get("association_manifest_sha256")
        != association_manifest_sha256
        or inputs.get("prompt_cache_manifest_sha256")
        != prompt_cache_manifest_sha256
        or inputs.get("internal_tensor_sha256") != internal_tensor_sha256
    ):
        raise ValueError("WS32 DSA source summary input provenance drifted")
    return summary


def _require_graph(
    run_dir: Path,
    record: dict[str, Any],
    *,
    referenced: set[str],
    expected_contract_passed: bool = True,
) -> None:
    if record.get("contract", {}).get("passed") is not expected_contract_passed:
        raise ValueError("WS32 DSA graph contract verdict drifted")
    for kind, suffix in (
        ("optimized_hlo", ".optimized_hlo.txt"),
        ("stablehlo", ".stablehlo.mlir"),
    ):
        value = record.get(kind)
        if not isinstance(value, dict) or set(value) != {"filename", "sha256"}:
            raise ValueError("WS32 DSA graph identity schema drifted")
        filename = value["filename"]
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(suffix)
            or filename in referenced
        ):
            raise ValueError("WS32 DSA graph filename drifted")
        path = run_dir / "hlo" / filename
        _require_digest(value["sha256"], field=f"{filename} graph hash")
        if _sha256_file(path) != value["sha256"]:
            raise ValueError("WS32 DSA graph hash drifted")
        referenced.add(filename)


def _optimized_hlo(run_dir: Path, record: dict[str, Any]) -> str:
    return (run_dir / "hlo" / record["optimized_hlo"]["filename"]).read_text()


def _stablehlo(run_dir: Path, record: dict[str, Any]) -> str:
    return (run_dir / "hlo" / record["stablehlo"]["filename"]).read_text()


def _require_component_contract(
    run_dir: Path,
    record: dict[str, Any],
    *,
    expected_entry_parameters: dict[tuple[str, tuple[int, ...]], int],
    expected_stablehlo_component: str | None = None,
    expected_collective_groups: tuple[int, ...] = (),
    required_live_markers: tuple[str, ...] = (),
    required_16k_fusion_shape: tuple[int, int] | None = None,
    expected_passed: bool = True,
) -> None:
    recomputed = validate_ws32_dsa_component_hlo(
        _optimized_hlo(run_dir, record),
        expected_entry_parameters=expected_entry_parameters,
        stablehlo=_stablehlo(run_dir, record),
        expected_stablehlo_component=expected_stablehlo_component,
        expected_collective_groups=expected_collective_groups,
        required_live_markers=required_live_markers,
        required_16k_fusion_shape=required_16k_fusion_shape,
    )
    if record.get("contract") != recomputed or (
        recomputed["passed"] is not expected_passed
    ):
        raise ValueError("WS32 DSA component HLO verdict drifted")


def validate_ws32_layer0_dsa_association(
    run_dir: Path,
    *,
    expected_code_hash: str,
    association_manifest_sha256: str,
    prompt_cache_manifest_sha256: str,
    internal_contract_sha256: str,
    internal_tensor_sha256: str,
    source_summary_sha256: str,
) -> dict[str, Any]:
    """Recompute the bounded verdict from pinned sources and raw arrays."""

    run_dir = Path(run_dir)
    runner_path = run_dir / "runner.json"
    runner = json.loads(runner_path.read_text())
    expected_runner_keys = {
        "artifact_kind",
        "association_restored",
        "backend",
        "candidates",
        "candidate_mechanisms_proven",
        "claim_scope",
        "code_hash",
        "device",
        "device_kind",
        "diagnostic_only",
        "exact_arms",
        "format_version",
        "inputs",
        "key",
        "one_live_row",
        "performance_claim",
        "q_a",
        "score_packing",
        "status",
        "tensor_file",
    }
    if set(runner) != expected_runner_keys:
        raise ValueError("WS32 DSA runner schema drifted")
    if (
        runner["artifact_kind"] != ARTIFACT_KIND
        or runner["format_version"] != 2
        or runner["status"] != "SUCCESS"
        or runner["code_hash"] != expected_code_hash
        or runner["backend"] != "tpu"
        or runner["diagnostic_only"] is not True
        or runner["performance_claim"] is not False
        or runner["one_live_row"] is not True
    ):
        raise ValueError("WS32 DSA runner identity/scope drifted")
    if runner["claim_scope"] != (
        "bounded one-layer position-8155 WS32 DSA association only; "
        "no decoder, Gate-D, latency, or token-rate claim"
    ) or any(
        not isinstance(runner[name], str) or not runner[name]
        for name in ("device", "device_kind")
    ):
        raise ValueError("WS32 DSA runner device/claim scope drifted")
    if set(runner["candidates"]) != {"tuple4", "tuple8"}:
        raise ValueError("WS32 DSA candidate set drifted")
    candidate_statuses = {
        arm: runner["candidates"][arm].get("status")
        if isinstance(runner["candidates"][arm], dict)
        else None
        for arm in ("tuple4", "tuple8")
    }
    if any(
        status not in {"SUCCESS", "HLO_REJECTED"}
        for status in candidate_statuses.values()
    ):
        raise ValueError("WS32 DSA candidate status drifted")
    if not isinstance(runner["exact_arms"], list) or any(
        arm not in ("tuple4", "tuple8") for arm in runner["exact_arms"]
    ) or len(set(runner["exact_arms"])) != len(runner["exact_arms"]):
        raise ValueError("WS32 DSA exact-arm schema drifted")
    if type(runner["association_restored"]) is not bool or (
        type(runner["candidate_mechanisms_proven"]) is not bool
    ):
        raise ValueError("WS32 DSA terminal classification types drifted")
    _require_digest(
        expected_code_hash, field="expected code hash", lengths=(40, 64)
    )
    for name, value in (
        ("association manifest", association_manifest_sha256),
        ("prompt cache manifest", prompt_cache_manifest_sha256),
        ("internal contract", internal_contract_sha256),
        ("internal tensor", internal_tensor_sha256),
    ):
        _require_digest(value, field=name)

    association_manifest, association = inspect_layer0_dsa_association_input(
        run_dir / "inputs" / "association",
        expected_manifest_sha256=association_manifest_sha256,
    )
    cache_manifest, prompt_bits = inspect_legacy_prompt_index_cache(
        run_dir / "inputs" / "prompt_cache",
        expected_manifest_sha256=prompt_cache_manifest_sha256,
    )
    internal_contract, internals = inspect_greenfield_layer0_dsa_internal_observation(
        run_dir / "inputs" / "internal",
        expected_contract_sha256=internal_contract_sha256,
        expected_tensor_sha256=internal_tensor_sha256,
    )
    source_summary = inspect_ws32_dsa_source_summary(
        run_dir / "inputs" / "source" / "summary.json",
        expected_sha256=source_summary_sha256,
        association_manifest_sha256=association_manifest_sha256,
        prompt_cache_manifest_sha256=prompt_cache_manifest_sha256,
        internal_tensor_sha256=internal_tensor_sha256,
    )
    expected_inputs = {
        "association_manifest_sha256": association_manifest[
            "manifest_sha256"
        ],
        "internal_contract_sha256": internal_contract_sha256,
        "internal_tensor_sha256": internal_tensor_sha256,
        "layer0_reference": internal_contract["layer0_reference"],
        "prompt_cache_manifest_sha256": cache_manifest["manifest_sha256"],
        "prompt_cache_sha256": cache_manifest[
            "prompt_index_key_bfloat16_sha256"
        ],
        "source_code_hash": SOURCE_CODE_HASH,
        "source_db_run_id": SOURCE_DB_RUN_ID,
        "source_summary_sha256": source_summary_sha256,
    }
    if runner["inputs"] != expected_inputs:
        raise ValueError("WS32 DSA source provenance drifted")

    tensor_record = runner["tensor_file"]
    if not isinstance(tensor_record, dict) or set(tensor_record) != {
        "byte_count", "filename", "sha256"
    } or (
        tensor_record["filename"] != "ws32_layer0_dsa_association.npz"
    ) or type(tensor_record["byte_count"]) is not int or (
        tensor_record["byte_count"] <= 0
    ):
        raise ValueError("WS32 DSA tensor record drifted")
    _require_digest(tensor_record["sha256"], field="tensor file hash")
    tensor_path = run_dir / tensor_record["filename"]
    if tensor_path.stat().st_size != tensor_record["byte_count"] or (
        _sha256_file(tensor_path) != tensor_record["sha256"]
    ):
        raise ValueError("WS32 DSA tensor file integrity failed")
    tensor_contract = {
        "accepted_current_key": ((1, 128), np.dtype(np.float32)),
        "accepted_head_weights": ((1, 32), np.dtype(np.float32)),
        "accepted_normalized_bfloat16_bits": (
            (1, 6144),
            np.dtype(np.uint16),
        ),
        "accepted_q_a_bfloat16_bits": ((1, 2048), np.dtype(np.uint16)),
        "accepted_query": ((1, 32, 128), np.dtype(np.float32)),
        "exact_current_key": ((1, 128), np.dtype(np.float32)),
        "ws32_current_q_a_bfloat16_bits": (
            (1, 2048),
            np.dtype(np.uint16),
        ),
    }
    for arm in ("tuple4", "tuple8"):
        if candidate_statuses[arm] != "SUCCESS":
            continue
        tensor_contract.update(
            {
                f"{arm}_expected_position_scores": (
                    (2048,),
                    np.dtype(np.float32),
                ),
                f"{arm}_head_weights": ((1, 32), np.dtype(np.float32)),
                f"{arm}_logical_scores": ((8156,), np.dtype(np.float32)),
                f"{arm}_accepted_q_query": (
                    (1, 32, 128),
                    np.dtype(np.float32),
                ),
                f"{arm}_current_q_query": (
                    (1, 32, 128),
                    np.dtype(np.float32),
                ),
            }
        )
    with np.load(tensor_path, allow_pickle=False) as bundle:
        if set(bundle.files) != set(tensor_contract):
            raise ValueError("WS32 DSA tensor key set drifted")
        tensors = {name: np.asarray(bundle[name]).copy() for name in bundle.files}
    for name, (shape, dtype) in tensor_contract.items():
        if tensors[name].shape != shape or tensors[name].dtype != dtype:
            raise ValueError(f"WS32 DSA tensor contract drifted: {name}")

    expected_normalized = np.ascontiguousarray(
        internals["normalized_hidden_bfloat16_bits"][0:1]
    )
    expected_q = np.ascontiguousarray(
        internals["q_a_state_bfloat16_bits"][0:1]
    )
    expected_query = np.ascontiguousarray(internals["query"][0:1])
    expected_head = np.ascontiguousarray(internals["head_weights"][0:1])
    expected_key = np.ascontiguousarray(internals["current_key"][0:1])
    source_copies = {
        "accepted_normalized_bfloat16_bits": expected_normalized,
        "accepted_q_a_bfloat16_bits": expected_q,
        "accepted_query": expected_query,
        "accepted_head_weights": expected_head,
        "accepted_current_key": expected_key,
    }
    for name, expected in source_copies.items():
        if not np.array_equal(tensors[name], expected):
            raise ValueError(f"WS32 DSA copied source drifted: {name}")

    q_comparison = _comparison(
        expected_q, tensors["ws32_current_q_a_bfloat16_bits"]
    )
    key_comparison = _comparison(expected_key, tensors["exact_current_key"])
    if not isinstance(runner["q_a"], dict) or set(runner["q_a"]) != {
        "comparison", "compile_seconds", "hlo"
    }:
        raise ValueError("WS32 DSA q-a schema drifted")
    if not isinstance(runner["key"], dict) or set(runner["key"]) != {
        "comparison", "hlo", "key_norm_mode"
    }:
        raise ValueError("WS32 DSA key schema drifted")
    _require_exact_comparison_schema(
        runner["q_a"]["comparison"], field="q-a"
    )
    _require_exact_comparison_schema(
        runner["key"]["comparison"], field="current key"
    )
    _require_positive_float(runner["q_a"]["compile_seconds"], field="q-a compile")
    _require_simple_graph_schema(runner["q_a"]["hlo"], field="q-a")
    key_graph = runner["key"]["hlo"]
    if not isinstance(key_graph, dict) or set(key_graph) != {
        "contract", "optimized_hlo", "stablehlo", "wk_decode", "wk_promote"
    }:
        raise ValueError("WS32 DSA key graph schema drifted")
    _require_simple_graph_schema(key_graph["wk_decode"], field="wk decode")
    _require_simple_graph_schema(key_graph["wk_promote"], field="wk promote")
    if runner["q_a"]["comparison"] != q_comparison or (
        runner["key"]["comparison"] != key_comparison
        or runner["key"]["key_norm_mode"] != "divide_sqrt"
    ):
        raise ValueError("WS32 DSA q/key verdict was not recomputed")

    expected_positions = association["expected_selected_positions"]
    expected_scores = association["expected_selected_scores"]
    exact_arms = []
    for arm in ("tuple4", "tuple8"):
        candidate = runner["candidates"][arm]
        common_candidate_keys = {
            "alias_count",
            "compile_seconds",
            "hlo",
            "local_heads",
            "status",
        }
        successful_candidate_keys = common_candidate_keys | {
            "accepted_q_query_comparison",
            "current_q_query_comparison",
            "head_comparison",
            "score",
        }
        rejected_candidate_keys = common_candidate_keys | {"rejection"}
        if not isinstance(candidate, dict):
            raise ValueError(f"WS32 DSA {arm} candidate schema drifted")
        candidate_keys = frozenset(candidate)
        if candidate_keys not in {
            frozenset(successful_candidate_keys),
            frozenset(rejected_candidate_keys),
        }:
            raise ValueError(f"WS32 DSA {arm} candidate schema drifted")
        if candidate["status"] != candidate_statuses[arm] or (
            (candidate["status"] == "SUCCESS")
            != (candidate_keys == frozenset(successful_candidate_keys))
        ):
            raise ValueError(f"WS32 DSA {arm} status/schema drifted")
        if not isinstance(candidate["compile_seconds"], dict) or set(
            candidate["compile_seconds"]
        ) != {"materializer", "query_head"}:
            raise ValueError(f"WS32 DSA {arm} compile schema drifted")
        for name, value in candidate["compile_seconds"].items():
            _require_positive_float(value, field=f"{arm} {name} compile")
        if not isinstance(candidate["hlo"], dict) or set(candidate["hlo"]) != {
            "materializer", "query_head"
        }:
            raise ValueError(f"WS32 DSA {arm} HLO schema drifted")
        for graph_name, graph in candidate["hlo"].items():
            _require_simple_graph_schema(graph, field=f"{arm} {graph_name}")
        alias_count = int(arm.removeprefix("tuple"))
        if type(candidate["alias_count"]) is not int or (
            candidate["alias_count"] != alias_count
        ) or type(candidate["local_heads"]) is not int or (
            candidate["local_heads"] != 32 // alias_count
        ):
            raise ValueError(f"WS32 DSA {arm} ownership drifted")
        if candidate["status"] == "HLO_REJECTED":
            if candidate["rejection"] != {
                "component": "query_head",
                "reason": "16k_fusion_hypothesis_rejected",
            } or candidate["hlo"]["materializer"].get("contract", {}).get(
                "passed"
            ) is not True or candidate["hlo"]["query_head"].get(
                "contract", {}
            ).get(
                "passed"
            ) is not False:
                raise ValueError(f"WS32 DSA {arm} rejection drifted")
            if classify_ws32_dsa_query_head_contract(
                candidate["hlo"]["query_head"]["contract"]
            ) != "HYPOTHESIS_REJECTED":
                raise ValueError(f"WS32 DSA {arm} rejection class drifted")
            continue
        for comparison_name in (
            "accepted_q_query_comparison",
            "current_q_query_comparison",
            "head_comparison",
        ):
            _require_exact_comparison_schema(
                candidate[comparison_name], field=f"{arm} {comparison_name}"
            )
        accepted_q_query_comparison = _comparison(
            expected_query, tensors[f"{arm}_accepted_q_query"]
        )
        current_q_query_comparison = _comparison(
            expected_query, tensors[f"{arm}_current_q_query"]
        )
        head_comparison = _comparison(
            expected_head, tensors[f"{arm}_head_weights"]
        )
        logical_scores = tensors[f"{arm}_logical_scores"]
        aligned = np.ascontiguousarray(logical_scores[expected_positions])
        if not np.array_equal(
            aligned, tensors[f"{arm}_expected_position_scores"]
        ):
            raise ValueError(f"WS32 DSA {arm} aligned score copy drifted")
        aligned_comparison = _comparison(expected_scores, aligned)
        selection = compare_dsa_association_scores(
            logical_scores, expected_positions, expected_scores
        )
        score = candidate["score"]
        if not isinstance(score, dict) or set(score) != {
            "aligned_score_comparison",
            "compile_seconds",
            "comparison",
            "hlo",
            "logical_score_sha256",
        }:
            raise ValueError(f"WS32 DSA {arm} score schema drifted")
        _require_positive_float(
            score["compile_seconds"], field=f"{arm} score compile"
        )
        _require_digest(
            score["logical_score_sha256"], field=f"{arm} logical score hash"
        )
        _require_exact_comparison_schema(
            score["aligned_score_comparison"], field=f"{arm} aligned score"
        )
        _require_selection_schema(score["comparison"], field=f"{arm} score")
        _require_simple_graph_schema(score["hlo"], field=f"{arm} score")
        if (
            candidate["accepted_q_query_comparison"]
            != accepted_q_query_comparison
            or candidate["current_q_query_comparison"]
            != current_q_query_comparison
            or candidate["head_comparison"] != head_comparison
            or score["aligned_score_comparison"] != aligned_comparison
            or score["comparison"] != selection
            or score["logical_score_sha256"] != _sha256_array(logical_scores)
        ):
            raise ValueError(f"WS32 DSA {arm} verdict was not recomputed")
        if (
            accepted_q_query_comparison["elementwise_exact"]
            and head_comparison["elementwise_exact"]
            and selection["passed"]
            and aligned_comparison["elementwise_exact"]
        ):
            exact_arms.append(arm)

    exact_arms.sort()
    association_restored = bool(
        q_comparison["elementwise_exact"]
        and key_comparison["elementwise_exact"]
        and exact_arms
    )
    candidate_mechanisms_proven = bool(
        key_comparison["elementwise_exact"] and exact_arms
    )
    if runner["exact_arms"] != exact_arms or (
        runner["association_restored"] is not association_restored
        or runner["candidate_mechanisms_proven"]
        is not candidate_mechanisms_proven
    ):
        raise ValueError("WS32 DSA terminal classification drifted")

    global_bits, lane_bits, lane_positions = pack_stage_local_index_keys(
        prompt_bits,
        tensors["exact_current_key"][0],
        logical_page_size=512,
        local_parallel_size=8,
    )
    expected_packing = {
        "global_bfloat16_sha256": _sha256_array(global_bits),
        "lane_bfloat16_sha256": _sha256_array(lane_bits),
        "lane_positions_sha256": _sha256_array(lane_positions),
        "lane_width": 1024,
        "local_parallel_size": 8,
        "logical_context": 8156,
        "logical_page_size": 512,
    }
    if runner["score_packing"] != expected_packing:
        raise ValueError("WS32 DSA score packing provenance drifted")

    referenced: set[str] = set()
    _require_graph(run_dir, runner["q_a"]["hlo"], referenced=referenced)
    _require_graph(run_dir, runner["key"]["hlo"], referenced=referenced)
    _require_graph(
        run_dir, runner["key"]["hlo"]["wk_decode"], referenced=referenced
    )
    _require_graph(
        run_dir, runner["key"]["hlo"]["wk_promote"], referenced=referenced
    )
    for arm in ("tuple4", "tuple8"):
        for name in ("materializer", "query_head"):
            _require_graph(
                run_dir,
                runner["candidates"][arm]["hlo"][name],
                referenced=referenced,
                expected_contract_passed=(
                    name == "materializer"
                    or candidate_statuses[arm] == "SUCCESS"
                ),
            )
        if candidate_statuses[arm] == "SUCCESS":
            _require_graph(
                run_dir,
                runner["candidates"][arm]["score"]["hlo"],
                referenced=referenced,
            )

    _require_component_contract(
        run_dir,
        runner["q_a"]["hlo"],
        expected_entry_parameters={
            ("bf16", (1, 1536)): 1,
            ("u8", (2048, 1536)): 1,
            ("f32", (16, 12)): 1,
            ("bf16", (2048,)): 1,
        },
        expected_collective_groups=(4,),
        required_live_markers=(
            "greenfield_fp8_block_matmul_f32_m8_k1536_n2048",
            "greenfield_ws32_linear/feature_reduce",
        ),
    )
    for arm, alias_count in (("tuple4", 4), ("tuple8", 8)):
        local_heads = 32 // alias_count
        local_width = local_heads * 128
        scale_rows = local_width // 128
        candidate = runner["candidates"][arm]
        _require_component_contract(
            run_dir,
            candidate["hlo"]["materializer"],
            expected_entry_parameters={
                ("u8", (local_width, 2048)): 1,
                ("f32", (scale_rows, 16)): 1,
            },
            expected_stablehlo_component=f"{arm}_materializer",
        )
        _require_component_contract(
            run_dir,
            candidate["hlo"]["query_head"],
            expected_entry_parameters={
                ("bf16", (1, 2048)): 1,
                ("bf16", (1, 6144)): 1,
                ("f32", (local_width, 2048)): alias_count,
                ("bf16", (local_heads, 6144)): 1,
                ("s32", (1,)): 1,
            },
            expected_stablehlo_component=f"{arm}_query_head",
            required_16k_fusion_shape=(alias_count, local_width),
            expected_passed=(candidate_statuses[arm] == "SUCCESS"),
        )
        if candidate_statuses[arm] != "SUCCESS":
            continue
        score_record = candidate["score"]["hlo"]
        score_association_contract = validate_dsa_association_hlo(
            _optimized_hlo(run_dir, score_record),
            phase="local_wide_default_score",
            context=1024,
        )
        score_component_contract = validate_ws32_dsa_component_hlo(
            _optimized_hlo(run_dir, score_record),
            stablehlo=_stablehlo(run_dir, score_record),
            expected_stablehlo_component="default_score",
            expected_entry_parameters={
                ("f32", (1, 32, 128)): 1,
                ("bf16", (1024, 128)): 1,
                ("f32", (1, 32)): 1,
            },
        )
        expected_score_contract = {
            "association": score_association_contract,
            "component": score_component_contract,
            "passed": True,
        }
        if score_record.get("contract") != expected_score_contract or (
            score_association_contract["passed"] is not True
            or score_component_contract["passed"] is not True
        ):
            raise ValueError(f"WS32 DSA {arm} scorer HLO verdict drifted")
    key_hlo = runner["key"]["hlo"]
    _require_component_contract(
        run_dir,
        key_hlo["wk_decode"],
        expected_entry_parameters={
            ("u8", (128, 6144)): 1,
            ("f32", (1, 48)): 1,
        },
        expected_stablehlo_component="wk_decode",
    )
    _require_component_contract(
        run_dir,
        key_hlo["wk_promote"],
        expected_entry_parameters={("bf16", (128, 6144)): 1},
        expected_stablehlo_component="wk_promote",
    )
    _require_component_contract(
        run_dir,
        key_hlo,
        expected_entry_parameters={
            ("bf16", (1, 6144)): 1,
            ("f32", (128, 6144)): 1,
            ("bf16", (128,)): 2,
            ("s32", (1,)): 1,
        },
        expected_stablehlo_component="exact_current_key",
    )
    actual_hlo = {
        path.name for path in (run_dir / "hlo").iterdir() if path.is_file()
    }
    if actual_hlo != referenced:
        raise ValueError("WS32 DSA HLO object set drifted")

    return {
        "artifact_kind": ARTIFACT_KIND,
        "association_restored": association_restored,
        "candidate_mechanisms_proven": candidate_mechanisms_proven,
        "code_hash": expected_code_hash,
        "exact_arms": exact_arms,
        "hlo_file_count": len(referenced),
        "runner_sha256": _sha256_file(runner_path),
        "status": "VALIDATED",
        "tensor_sha256": tensor_record["sha256"],
    }
