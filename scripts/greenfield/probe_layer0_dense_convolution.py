#!/usr/bin/env python3
"""Test the accepted layer-0 dense convolution arithmetic after exact attention."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import ml_dtypes
import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.greenfield.probe_layer0_projection_reduction import (  # noqa: E402
    _compare_bits,
    _computation_id,
    _file_sha256,
    _instruction_key,
    _load_weights,
    _shape_signatures,
    _tuple_index,
    _value_depends_on,
)


_POSITION = 8155
_DB538_CODE_HASH = "e2a3a74a3b2ef1fa8f3b9cb1c5d7ec65f833eafc"
_DB538_RUN_ID = 538
_ACCEPTED_LAYER1_SHA256 = (
    "9936ee1e19049b297fd205292ebc378aee41d59401bbf56497004356998d3039"
)
_ATTENTION_UPDATE_SHA256 = (
    "68afed86921584fb673abb11a563e359a1533210ec2483e71ee79b88c2b0bde7"
)
_NORMALIZED_MLP_SHA256 = (
    "082125fead43b25f10686705c1b6473153f4092dd5bc476f8e01a86629f0758f"
)
_COMBINED_RESIDUAL_SHA256 = (
    "02d045b9a0ec5ab22a711bd6a964564f707be0848683381104e83331020e31a3"
)
_POST_ATTENTION_RESIDUAL_SHA256 = (
    "a105fdbd429adb1d06a70bf71598a72a91d7b6faa83360005487ce11ce099f8e"
)
_ACCEPTED_M32_CODE_HASH = "ec114ae91f65c10fc3a77dcffa7ca283fa7be129"
_ACCEPTED_M32_LEGACY_HASH = "b3c25df47ac98783912dc658878181ec0a8ae16d"
_ACCEPTED_M32_RAW_HLO_SHA256 = (
    "3cd750810982608f9a3a7d557497c58f61159cc3dcdeb521f1377ba8c93fb775"
)
_ACCEPTED_M32_RUN_ID = 532
_DENSE_WEIGHT_NAMES = (
    "dense.slot_00.gate.weight_bits",
    "dense.slot_00.gate.scale_inv",
    "dense.slot_00.up.weight_bits",
    "dense.slot_00.up.scale_inv",
    "dense.slot_00.down.weight_bits",
    "dense.slot_00.down.scale_inv",
    "attention.slot_01.input_norm",
)


def _array_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _require_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file() or _file_sha256(path) != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 drifted")


def _load_db538(args: argparse.Namespace) -> tuple[np.ndarray, ...]:
    """Load only the exact post-attention boundary sealed by DB538/DB539."""

    _require_file(args.db538_runner, args.db538_runner_sha256, "DB538 runner")
    _require_file(args.db538_tensor, args.db538_tensor_sha256, "DB538 tensor")
    _require_file(args.db538_summary, args.db538_summary_sha256, "DB538 summary")
    _require_file(args.db538_success, args.db538_success_sha256, "DB538 SUCCESS")
    runner = json.loads(args.db538_runner.read_text())
    arms = runner.get("arms", {})
    expected_mismatches = {
        "local_attention_local_dense": 4022,
        "strategy_attention_local_dense": 2318,
        "local_attention_strategy_dense": 4044,
        "strategy_attention_strategy_dense": 2388,
    }
    if (
        runner.get("artifact_kind")
        != "glm52_layer0_projection_reduction_probe"
        or runner.get("code_hash") != _DB538_CODE_HASH
        or runner.get("position") != _POSITION
        or runner.get("status") != "SUCCESS"
        or runner.get("classification") != "projection_reduction_unresolved"
        or runner.get("exact_arms") != []
        or set(arms) != set(expected_mismatches)
        or any(
            arms[name]["layer1_comparison"]["mismatch_count"] != count
            for name, count in expected_mismatches.items()
        )
        or any(
            not arms[name]["value_comparison"]["elementwise_exact"]
            for name in expected_mismatches
        )
    ):
        raise RuntimeError("DB538 runner contract drifted")
    summary = json.loads(args.db538_summary.read_text())
    if (
        summary.get("artifact_kind") != runner["artifact_kind"]
        or summary.get("classification") != runner["classification"]
        or summary.get("code_hash") != _DB538_CODE_HASH
        or summary.get("exact_arms") != []
        or summary.get("results_db_run_id") != _DB538_RUN_ID
        or summary.get("status") != "SUCCESS"
        or summary.get("performance_claim") is not False
    ):
        raise RuntimeError("DB538 summary contract drifted")
    success = dict(
        line.split("=", 1)
        for line in args.db538_success.read_text().splitlines()
        if line
    )
    if (
        success.get("artifact_kind") != runner["artifact_kind"]
        or success.get("code_hash") != _DB538_CODE_HASH
        or success.get("results_db_run_id") != str(_DB538_RUN_ID)
        or success.get("classification") != runner["classification"]
        or success.get("exact_arms") != "none"
        or success.get("performance_claim") != "false"
    ):
        raise RuntimeError("DB538 terminal contract drifted")

    with np.load(args.db538_tensor, allow_pickle=False) as payload:
        accepted_layer1 = np.ascontiguousarray(
            payload["accepted_layer1_normalized_bfloat16_bits"]
        )
        residual = np.ascontiguousarray(
            payload["combined_residual_bfloat16_bits"]
        )
        attention_local = np.ascontiguousarray(
            payload[
                "attention_update_bfloat16_bits__strategy_attention_local_dense"
            ]
        )
        attention_strategy = np.ascontiguousarray(
            payload[
                "attention_update_bfloat16_bits__strategy_attention_strategy_dense"
            ]
        )
        normalized_local = np.ascontiguousarray(
            payload[
                "normalized_mlp_bfloat16_bits__strategy_attention_local_dense"
            ]
        )
        normalized_strategy = np.ascontiguousarray(
            payload[
                "normalized_mlp_bfloat16_bits__strategy_attention_strategy_dense"
            ]
        )
    expected = (
        (accepted_layer1, (6144,), _ACCEPTED_LAYER1_SHA256),
        (residual, (1, 6144), _COMBINED_RESIDUAL_SHA256),
        (attention_local, (1, 6144), _ATTENTION_UPDATE_SHA256),
        (normalized_local, (1, 6144), _NORMALIZED_MLP_SHA256),
    )
    if any(
        value.dtype != np.uint16
        or value.shape != shape
        or _array_sha256(value) != expected_sha
        for value, shape, expected_sha in expected
    ) or not np.array_equal(attention_local, attention_strategy) or (
        not np.array_equal(normalized_local, normalized_strategy)
    ):
        raise RuntimeError("DB538 exact post-attention tensors drifted")
    post_attention = np.asarray(
        residual.view(ml_dtypes.bfloat16).astype(np.float32)
        + attention_local.view(ml_dtypes.bfloat16).astype(np.float32),
        dtype=ml_dtypes.bfloat16,
    )
    if (
        post_attention.shape != (1, 6144)
        or _array_sha256(post_attention.view(np.uint16))
        != _POST_ATTENTION_RESIDUAL_SHA256
    ):
        raise RuntimeError("post-attention residual reconstruction drifted")
    return (
        normalized_local.view(ml_dtypes.bfloat16),
        post_attention,
        accepted_layer1,
    )


def _load_accepted_m32_source(args: argparse.Namespace) -> dict[str, str]:
    """Authenticate the exact accepted M32 lowering that motivates the arm."""

    required = (
        args.accepted_m32_hlo,
        args.accepted_m32_hlo_sha256,
        args.accepted_m32_summary,
        args.accepted_m32_summary_sha256,
        args.accepted_m32_success,
        args.accepted_m32_success_sha256,
    )
    if any(value is None for value in required):
        raise RuntimeError("M32 discriminator requires the complete DB532 source")
    _require_file(
        args.accepted_m32_hlo,
        args.accepted_m32_hlo_sha256,
        "accepted M32 HLO",
    )
    _require_file(
        args.accepted_m32_summary,
        args.accepted_m32_summary_sha256,
        "accepted M32 summary",
    )
    _require_file(
        args.accepted_m32_success,
        args.accepted_m32_success_sha256,
        "accepted M32 SUCCESS",
    )
    raw_digest = sha256()
    with gzip.open(args.accepted_m32_hlo, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            raw_digest.update(chunk)
    summary = json.loads(args.accepted_m32_summary.read_text())
    hlo = summary.get("hlo", {})
    if (
        summary.get("artifact_kind")
        != "accepted_decode_projection_lowering_v1"
        or summary.get("code_hash") != _ACCEPTED_M32_CODE_HASH
        or summary.get("legacy_code_hash") != _ACCEPTED_M32_LEGACY_HASH
        or summary.get("diagnostic_only") is not True
        or summary.get("performance_claim") is not False
        or summary.get("status") != "SUCCESS"
        or summary.get("hlo_raw_sha256") != _ACCEPTED_M32_RAW_HLO_SHA256
        or raw_digest.hexdigest() != _ACCEPTED_M32_RAW_HLO_SHA256
        or hlo.get("category_counts")
        != {"attention": 78, "dense_mlp": 3, "moe_tuple": 75}
        or hlo.get("compile_bucket_rows") != 32
        or hlo.get("partition_count") != 32
        or hlo.get("reduction_dtype") != "bf16"
        or hlo.get("result_width") != 6144
    ):
        raise RuntimeError("accepted M32 lowering contract drifted")
    success = dict(
        line.split("=", 1)
        for line in args.accepted_m32_success.read_text().splitlines()
        if line
    )
    if (
        success.get("code_hash") != _ACCEPTED_M32_CODE_HASH
        or success.get("legacy_repository_pin") != _ACCEPTED_M32_LEGACY_HASH
        or success.get("source_run_id") != str(_ACCEPTED_M32_RUN_ID)
        or success.get("accepted_decode_projection_capture") != "true"
        or success.get("accepted_decode_projection_diagnostic_only") != "true"
        or success.get("accepted_decode_projection_reduction_dtype") != "bf16"
        or success.get("accepted_decode_projection_strategy") != "StrategyND"
    ):
        raise RuntimeError("accepted M32 terminal contract drifted")
    return {
        "accepted_m32_hlo_raw_sha256": _ACCEPTED_M32_RAW_HLO_SHA256,
        "accepted_m32_hlo_sha256": args.accepted_m32_hlo_sha256,
        "accepted_m32_success_sha256": args.accepted_m32_success_sha256,
        "accepted_m32_summary_sha256": args.accepted_m32_summary_sha256,
    }


def _validate_stablehlo(
    stablehlo: str, *, compile_rows: int = 1, layer1_only: bool = False
) -> dict[str, Any]:
    from glm_tpu.greenfield.sharding.stablehlo_dense_convolution import (
        validate_dense_convolution_stablehlo,
    )

    return validate_dense_convolution_stablehlo(
        stablehlo, compile_rows=compile_rows, layer1_only=layer1_only
    )


def _validate_optimized_hlo(
    optimized_hlo: str, *, compile_rows: int = 1, layer1_only: bool = False
) -> dict[str, Any]:
    from glm_tpu.greenfield.sharding.hlo_contract import (
        COLLECTIVE_OPCODES,
        parse_hlo_module,
    )

    if compile_rows not in (1, 32):
        raise ValueError("dense optimized-HLO compile rows must be 1 or 32")
    module = parse_hlo_module(optimized_hlo)
    by_key = {_instruction_key(item): item for item in module.instructions}
    users: dict[tuple[str, str], list[Any]] = {}
    for item in module.instructions:
        for operand in item.operand_names:
            users.setdefault((item.computation, operand), []).append(item)

    layout_only = {
        "bitcast",
        "copy",
        "convert",
        "optimization-barrier",
        "reshape",
    }

    def operand(item: Any, index: int) -> Any | None:
        if index >= len(item.operand_names):
            return None
        return by_key.get((item.computation, item.operand_names[index]))

    def is_m32_row0_slice(item: Any) -> bool:
        if compile_rows != 32 or item.raw_opcode != "slice":
            return False
        operand_shapes = _shape_signatures(item.operand_shapes)
        result_shapes = _shape_signatures(item.result_shapes)
        if len(operand_shapes) != 1 or len(result_shapes) != 1:
            return False
        compact = re.sub(r"\s+", "", item.raw_line)
        two_dimensional = re.fullmatch(
            r"(bf16|f32)\[32,([0-9]+)\]", operand_shapes[0]
        )
        if two_dimensional is not None:
            dtype, width = two_dimensional.groups()
            return (
                result_shapes[0] == f"{dtype}[1,{width}]"
                and f"slice={{[0:1],[0:{width}]}}" in compact
            )
        three_dimensional = re.fullmatch(
            r"(bf16|f32)\[8,32,([0-9]+)\]", operand_shapes[0]
        )
        if three_dimensional is None:
            return False
        dtype, width = three_dimensional.groups()
        return (
            result_shapes[0] == f"{dtype}[8,1,{width}]"
            and f"slice={{[0:8],[0:1],[0:{width}]}}" in compact
        )

    def unwrap_layout(item: Any | None) -> Any | None:
        seen: set[tuple[str, str]] = set()
        while (
            item is not None
            and item.raw_opcode in layout_only
            and len(item.operand_names) == 1
            and _instruction_key(item) not in seen
        ):
            seen.add(_instruction_key(item))
            item = operand(item, 0)
        return item

    def unwrap(item: Any | None) -> Any | None:
        seen: set[tuple[str, str]] = set()
        while item is not None and _instruction_key(item) not in seen:
            seen.add(_instruction_key(item))
            if item.raw_opcode in layout_only or is_m32_row0_slice(item):
                if len(item.operand_names) != 1:
                    return item
                item = operand(item, 0)
                continue
            return item
        return item

    def local_depends(value: Any | None, source: Any) -> bool:
        if value is None or value.computation != source.computation:
            return False
        source_key = _instruction_key(source)
        pending = [_instruction_key(value)]
        seen: set[tuple[str, str]] = set()
        while pending:
            key = pending.pop()
            if key == source_key:
                return True
            if key in seen:
                continue
            seen.add(key)
            item = by_key.get(key)
            if item is not None:
                pending.extend(
                    (item.computation, name) for name in item.operand_names
                )
        return False

    def is_one(item: Any | None) -> bool:
        item = unwrap(item)
        if item is None:
            return False
        if item.raw_opcode == "constant":
            return re.search(
                r"\bconstant\(1(?:\.0*)?(?:[eE][+-]?0+)?\)",
                item.raw_line,
            ) is not None
        if item.raw_opcode in {"broadcast", "broadcast-in-dim"} and len(
            item.operand_names
        ) == 1:
            return is_one(operand(item, 0))
        return False

    def has_bf16_correction(item: Any) -> bool:
        compact = re.sub(r"\s+", "", item.raw_line)
        return (
            '"float_type_correction_info":' in compact
            and '"original_type":"BF16"' in compact
        )

    def exact_bf16_round(item: Any) -> bool:
        return (
            len(item.result_shapes) == 1
            and item.result_shapes[0].dtype == "bf16"
        ) or has_bf16_correction(item)

    instructions_by_computation: dict[str, list[Any]] = {}
    callers_by_computation: dict[str, list[Any]] = {}
    for item in module.instructions:
        instructions_by_computation.setdefault(
            _computation_id(item.computation), []
        ).append(item)
        if item.raw_opcode == "fusion":
            match = re.search(r"\bcalls=%?([^,\s}\]]+)", item.raw_line)
            if match is not None:
                callers_by_computation.setdefault(match.group(1), []).append(item)

    external_value_memo: dict[tuple[str, str], tuple[Any, ...]] = {}

    def exact_external_values(item: Any) -> tuple[Any, ...]:
        """Map one exact internal value to its caller result/GTE values."""

        key = _instruction_key(item)
        if key in external_value_memo:
            return external_value_memo[key]
        if item.computation.startswith("ENTRY "):
            result = (item,)
            external_value_memo[key] = result
            return result
        computation = _computation_id(item.computation)
        roots = [
            value
            for value in instructions_by_computation.get(computation, ())
            if value.raw_line.lstrip().startswith("ROOT ")
        ]
        callers = callers_by_computation.get(computation, ())
        if len(roots) != 1 or len(callers) != 1:
            external_value_memo[key] = ()
            return ()
        root = roots[0]
        caller = callers[0]
        result = []
        if root.raw_opcode == "tuple":
            for index, name in enumerate(root.operand_names):
                value = by_key.get((root.computation, name))
                if unwrap_layout(value) is not item:
                    continue
                selected = [
                    user
                    for user in users.get(_instruction_key(caller), ())
                    if user.raw_opcode == "get-tuple-element"
                    and _tuple_index(user) == index
                ]
                if len(selected) != 1:
                    external_value_memo[key] = ()
                    return ()
                result.append(selected[0])
        elif unwrap_layout(root) is item:
            result.append(caller)
        external_value_memo[key] = tuple(result)
        return tuple(result)

    def fully_externalized_value(item: Any) -> Any | None:
        seen: set[tuple[str, str]] = set()
        while not item.computation.startswith("ENTRY "):
            key = _instruction_key(item)
            if key in seen:
                return None
            seen.add(key)
            values = exact_external_values(item)
            if len(values) != 1:
                return None
            item = values[0]
        return item

    def external_parameter_value(item: Any | None) -> Any | None:
        """Resolve a nested-fusion parameter to its exact outer operand."""

        seen: set[tuple[str, str]] = set()
        while item is not None and item.raw_opcode == "parameter":
            key = _instruction_key(item)
            if key in seen or item.computation.startswith("ENTRY "):
                break
            seen.add(key)
            match = re.search(r"\bparameter\(([0-9]+)\)", item.raw_line)
            callers = callers_by_computation.get(
                _computation_id(item.computation), ()
            )
            if match is None or len(callers) != 1:
                return None
            index = int(match.group(1))
            item = operand(callers[0], index)
        return item

    def exact_layout_source(value: Any | None, source: Any) -> bool:
        item = value
        seen: set[tuple[str, str]] = set()
        while item is not None and _instruction_key(item) not in seen:
            seen.add(_instruction_key(item))
            if item.raw_opcode in layout_only and len(item.operand_names) == 1:
                item = operand(item, 0)
                continue
            if item.raw_opcode == "parameter" and not item.computation.startswith(
                "ENTRY "
            ):
                item = external_parameter_value(item)
                continue
            break
        return unwrap(item) is unwrap(source)

    convolutions = [
        item for item in module.instructions if item.raw_opcode == "convolution"
    ]
    gate_up = []
    down = []
    unexpected = []
    virtual_rank_pattern = re.compile(
        r"greenfield_dense_convolution_virtual_rank_([0-7][0-7])"
    )

    def virtual_rank(item: Any) -> int | None:
        matches = virtual_rank_pattern.findall(item.op_name or "")
        if len(matches) != 1:
            return None
        value = int(matches[0])
        return value if 0 <= value < 8 else None

    for item in convolutions:
        operands = _shape_signatures(item.operand_shapes)
        results = _shape_signatures(item.result_shapes)
        exact_dimension_labels = "dim_labels=bf_io->bf" in re.sub(
            r"\s+", "", item.raw_line
        )
        gate_geometries = {
            (
                (f"bf16[{compile_rows},6144]", "bf16[6144,768]"),
                (f"f32[{compile_rows},768]",),
            )
        }
        down_geometries = {
            (
                (f"bf16[{compile_rows},384]", "bf16[384,6144]"),
                (f"f32[{compile_rows},6144]",),
            )
        }
        if compile_rows == 1:
            gate_geometries.add(
                (("bf16[8,6144]", "bf16[6144,768]"), ("f32[8,768]",))
            )
            down_geometries.add(
                (("bf16[8,384]", "bf16[384,6144]"), ("f32[8,6144]",))
            )
        if exact_dimension_labels and (operands, results) in gate_geometries:
            gate_up.append(item)
        elif exact_dimension_labels and (operands, results) in down_geometries:
            down.append(item)
        else:
            unexpected.append(item.name)
    async_collectives = [
        item.name
        for item in module.instructions
        if any(
            item.raw_opcode == f"{opcode}-{suffix}"
            for opcode in COLLECTIVE_OPCODES
            for suffix in ("start", "done")
        )
    ]
    collectives = list(module.collectives)
    scoped = [
        item
        for item in collectives
        if "greenfield_strategy_nd_row0_dense_convolution_down"
        in (item.op_name or "").split("/")
    ]
    violations = []
    if module.num_partitions != 4 or module.num_replicas not in (None, 1):
        violations.append("optimized module cardinality drifted")
    if len(gate_up) != 8 or len(down) != 8 or unexpected:
        violations.append(
            "optimized convolution geometry/count drifted: "
            f"gate_up={len(gate_up)} down={len(down)} unexpected={unexpected}"
        )
    if async_collectives:
        violations.append(f"async collectives are forbidden: {async_collectives}")
    if len(collectives) != 1 or len(scoped) != 1:
        violations.append(
            "expected one exact dense StrategyND collective: "
            f"all={len(collectives)} scoped={len(scoped)}"
        )
    collective = scoped[0] if len(scoped) == 1 else None
    if collective is not None and (
        collective.opcode != "all-gather"
        or collective.replica_groups != ((0, 1, 2, 3),)
        or not collective.use_global_device_ids
        or _shape_signatures(collective.operand_shapes)
        != ("bf16[8,1,6144]",)
        or _shape_signatures(collective.result_shapes)
        not in {("bf16[4,8,1,6144]",), ("bf16[32,1,6144]",)}
        or "dimensions={0}" not in collective.raw_line
    ):
        violations.append("dense StrategyND collective geometry drifted")
    lineage: dict[str, Any] = {}
    gate_by_rank: dict[int | None, Any] = {}
    down_by_rank: dict[int | None, Any] = {}
    gate_value_by_rank: dict[int, Any] = {}
    down_value_by_rank: dict[int, Any] = {}
    if len(gate_up) == 8 and len(down) == 8:
        gate_by_rank = {virtual_rank(item): item for item in gate_up}
        down_by_rank = {virtual_rank(item): item for item in down}
        lineage["gate_up_virtual_ranks"] = sorted(
            rank for rank in gate_by_rank if rank is not None
        )
        lineage["down_virtual_ranks"] = sorted(
            rank for rank in down_by_rank if rank is not None
        )
        if set(gate_by_rank) != set(range(8)) or set(down_by_rank) != set(
            range(8)
        ):
            violations.append("optimized virtual-rank scopes drifted")
        else:
            for rank in range(8):
                gate_values = exact_external_values(gate_by_rank[rank])
                down_values = exact_external_values(down_by_rank[rank])
                if len(gate_values) == 1:
                    gate_value_by_rank[rank] = gate_values[0]
                if len(down_values) == 1:
                    down_value_by_rank[rank] = down_values[0]
                elif compile_rows == 32:
                    computation = _computation_id(
                        down_by_rank[rank].computation
                    )
                    roots = [
                        item
                        for item in instructions_by_computation.get(
                            computation, ()
                        )
                        if item.raw_line.lstrip().startswith("ROOT ")
                    ]
                    callers = callers_by_computation.get(
                        computation, ()
                    )
                    if (
                        len(roots) == 1
                        and roots[0].raw_opcode == "dynamic-update-slice"
                        and len(callers) == 1
                    ):
                        down_value_by_rank[rank] = callers[0]
        if compile_rows == 32:
            down_to_gate = {
                down_by_rank[rank].name: (
                    [gate_by_rank[rank].name]
                    if rank in gate_value_by_rank
                    and rank in down_value_by_rank
                    else []
                )
                for rank in range(8)
            }
        else:
            down_to_gate = {
                item.name: [
                    producer.name
                    for rank, producer in gate_by_rank.items()
                    if rank is not None
                    and rank in gate_value_by_rank
                    and virtual_rank(item) in down_value_by_rank
                    and _value_depends_on(
                        module,
                        down_value_by_rank[virtual_rank(item)],
                        gate_value_by_rank[rank],
                    )
                ]
                for item in down
            }
        gate_consumers = {
            producer.name: [
                item.name
                for item in down
                if producer.name in down_to_gate[item.name]
            ]
            for producer in gate_up
        }
        lineage["down_to_gate_up"] = down_to_gate
        lineage["gate_up_to_down"] = gate_consumers
        if any(len(values) != 1 for values in down_to_gate.values()) or any(
            len(values) != 1 for values in gate_consumers.values()
        ):
            violations.append("gate/up-to-down convolution bijection drifted")
        elif all(rank in gate_by_rank and rank in down_by_rank for rank in range(8)):
            wrong_rank_edges = [
                rank
                for rank in range(8)
                if down_to_gate[down_by_rank[rank].name]
                != [gate_by_rank[rank].name]
            ]
            if wrong_rank_edges:
                violations.append(
                    "gate/up-to-down virtual-rank association drifted: "
                    f"{wrong_rank_edges}"
                )
            activation_contract: dict[str, Any] = {}
            activation_shapes = {
                f"bf16[{compile_rows},384]",
                f"f32[{compile_rows},384]",
            }
            if compile_rows == 1:
                activation_shapes.update({"bf16[8,384]", "f32[8,384]"})
            for rank in range(8):
                scoped = [
                    item
                    for item in module.instructions
                    if virtual_rank(item) == rank
                    and set(_shape_signatures(item.result_shapes))
                    & activation_shapes
                ]
                selected = {
                    opcode: [
                        item
                        for item in scoped
                        if item.raw_opcode == opcode
                    ]
                    for opcode in (
                        "negate",
                        "exponential",
                        "add",
                        "divide",
                        "multiply",
                    )
                }
                activation_contract[str(rank)] = {
                    opcode: [item.name for item in items]
                    for opcode, items in selected.items()
                }
                expected_counts = {
                    "negate": 1,
                    "exponential": 1,
                    "add": 1,
                    "divide": 1,
                    "multiply": 2,
                }
                if any(
                    len(selected[opcode]) != count
                    for opcode, count in expected_counts.items()
                ):
                    violations.append(
                        f"virtual rank {rank} optimized SwiGLU arithmetic drifted"
                    )
                    continue
                negate = selected["negate"][0]
                exponential = selected["exponential"][0]
                denominator = selected["add"][0]
                sigmoid = selected["divide"][0]
                multiplies = selected["multiply"]
                gate = unwrap(operand(negate, 0))
                if (
                    gate is None
                    or gate.raw_opcode != "slice"
                    or f"slice={{[0:{compile_rows}],[0:384]}}"
                    not in re.sub(r"\s+", "", gate.raw_line)
                    or unwrap(operand(exponential, 0)) is not negate
                ):
                    violations.append(
                        f"virtual rank {rank} optimized gate/sigmoid source drifted"
                    )
                    continue
                denominator_operands = tuple(
                    unwrap(operand(denominator, index))
                    for index in range(len(denominator.operand_names))
                )
                if len(denominator_operands) != 2 or not (
                    exponential in denominator_operands
                    and any(is_one(item) for item in denominator_operands)
                ):
                    violations.append(
                        f"virtual rank {rank} optimized sigmoid denominator drifted"
                    )
                    continue
                sigmoid_operands = tuple(
                    unwrap(operand(sigmoid, index))
                    for index in range(len(sigmoid.operand_names))
                )
                if (
                    len(sigmoid_operands) != 2
                    or not is_one(sigmoid_operands[0])
                    or sigmoid_operands[1] is not denominator
                ):
                    violations.append(
                        f"virtual rank {rank} optimized sigmoid division drifted"
                    )
                    continue
                silu = next(
                    (
                        item
                        for item in multiplies
                        if Counter(
                            _instruction_key(value)
                            for value in (
                                unwrap(operand(item, 0)),
                                unwrap(operand(item, 1)),
                            )
                            if value is not None
                        )
                        == Counter((_instruction_key(gate), _instruction_key(sigmoid)))
                    ),
                    None,
                )
                activated = next(
                    (
                        item
                        for item in multiplies
                        if item is not silu
                        and any(
                            unwrap(operand(item, index)) is silu
                            for index in range(len(item.operand_names))
                        )
                    ),
                    None,
                )
                activated_operands = (
                    ()
                    if activated is None
                    else tuple(
                        unwrap(operand(activated, index))
                        for index in range(len(activated.operand_names))
                    )
                )
                up = next(
                    (item for item in activated_operands if item is not silu), None
                )
                if (
                    silu is None
                    or activated is None
                    or len(activated_operands) != 2
                    or up is None
                    or up.raw_opcode != "slice"
                    or f"slice={{[0:{compile_rows}],[384:768]}}"
                    not in re.sub(r"\s+", "", up.raw_line)
                    or unwrap(operand(gate, 0)) is not unwrap(operand(up, 0))
                    or not exact_bf16_round(silu)
                    or not exact_bf16_round(activated)
                ):
                    violations.append(
                        f"virtual rank {rank} optimized SwiGLU operand graph drifted"
                    )
                    continue
                source = unwrap(operand(gate, 0))
                if source is None:
                    violations.append(
                        f"virtual rank {rank} optimized SwiGLU source is absent"
                    )
                    continue
                outer_source = external_parameter_value(source)
                source_bound = (
                    outer_source is not None
                    and rank in gate_value_by_rank
                    and unwrap(outer_source) is unwrap(gate_value_by_rank[rank])
                )
                activated_values = exact_external_values(activated)
                result_bound = (
                    len(activated_values) == 1
                    and unwrap(operand(down_by_rank[rank], 0))
                    is unwrap(activated_values[0])
                    and rank in down_value_by_rank
                )
                if not source_bound or not result_bound:
                    violations.append(
                        f"virtual rank {rank} optimized fused SwiGLU binding drifted"
                    )
                    continue
                activation_contract[str(rank)]["exact_operand_graph"] = True
            lineage["activation_contract"] = activation_contract
    def exact_s32_constant(item: Any | None, expected: int) -> bool:
        return (
            item is not None
            and item.raw_opcode == "constant"
            and _shape_signatures(item.result_shapes) == ("s32[]",)
            and re.search(
                rf"\bconstant\({expected}\)(?:,|\s|$)", item.raw_line
            )
            is not None
        )

    def exact_m32_update_stack() -> tuple[list[Any], Any | None, list[str]]:
        """Recognize TPU's exact fused lowering of the ordered M32 stack."""

        roots_by_rank: dict[int, tuple[Any, Any]] = {}
        errors: list[str] = []
        for rank in range(8):
            if rank not in down_by_rank:
                continue
            computation = _computation_id(down_by_rank[rank].computation)
            roots = [
                item
                for item in instructions_by_computation.get(computation, ())
                if item.raw_line.lstrip().startswith("ROOT ")
            ]
            callers = callers_by_computation.get(computation, ())
            if (
                len(roots) == 1
                and roots[0].raw_opcode == "dynamic-update-slice"
            ):
                if len(callers) != 1:
                    errors.append(f"rank {rank} fused stack caller drifted")
                else:
                    roots_by_rank[rank] = (roots[0], callers[0])
        if not roots_by_rank:
            return [], None, []
        if set(roots_by_rank) != set(range(8)):
            return [], None, ["M32 fused stack does not contain all eight ranks"]

        callers: list[Any] = []
        for rank in range(8):
            root, caller = roots_by_rank[rank]
            root_operands = [
                operand(root, index) for index in range(len(root.operand_names))
            ]
            if (
                caller.raw_opcode != "fusion"
                or not caller.computation.startswith("ENTRY ")
                or virtual_rank(caller) != rank
                or _shape_signatures(caller.result_shapes)
                != ("bf16[8,32,6144]",)
                or len(root_operands) != 5
                or _shape_signatures(root.operand_shapes)
                != (
                    "bf16[8,32,6144]",
                    "bf16[1,32,6144]",
                    "s32[]",
                    "s32[]",
                    "s32[]",
                )
                or _shape_signatures(root.result_shapes)
                != ("bf16[8,32,6144]",)
            ):
                errors.append(f"rank {rank} fused stack geometry drifted")
                continue
            inserted = root_operands[1]
            rounded = None if inserted is None else operand(inserted, 0)
            if (
                inserted is None
                or inserted.raw_opcode != "bitcast"
                or _shape_signatures(inserted.operand_shapes)
                != ("bf16[32,6144]",)
                or _shape_signatures(inserted.result_shapes)
                != ("bf16[1,32,6144]",)
                or rounded is None
                or rounded.raw_opcode != "convert"
                or _shape_signatures(rounded.operand_shapes)
                != ("f32[32,6144]",)
                or _shape_signatures(rounded.result_shapes)
                != ("bf16[32,6144]",)
                or operand(rounded, 0) is not down_by_rank[rank]
            ):
                errors.append(f"rank {rank} fused stack BF16 insertion drifted")
                continue
            if not (
                exact_s32_constant(root_operands[2], rank)
                and exact_s32_constant(root_operands[3], 0)
                and exact_s32_constant(root_operands[4], 0)
            ):
                errors.append(f"rank {rank} fused stack index drifted")
                continue
            base = root_operands[0]
            if rank == 0:
                if (
                    base is None
                    or base.raw_opcode != "custom-call"
                    or base.operand_shapes
                    or _shape_signatures(base.result_shapes)
                    != ("bf16[8,32,6144]",)
                    or 'custom_call_target="AllocateBuffer"'
                    not in base.raw_line
                ):
                    errors.append("rank 0 fused stack allocation drifted")
                    continue
            elif (
                base is None
                or base.raw_opcode != "parameter"
                or _shape_signatures(base.result_shapes)
                != ("bf16[8,32,6144]",)
                or re.search(r"\bparameter\(0\)", base.raw_line) is None
                or len(callers) != rank
                or external_parameter_value(base) is not callers[rank - 1]
            ):
                errors.append(f"rank {rank} fused stack predecessor drifted")
                continue
            callers.append(caller)
        if errors or len(callers) != 8:
            return callers, None, errors
        slices = [
            item
            for item in module.instructions
            if is_m32_row0_slice(item)
            and _shape_signatures(item.result_shapes) == ("bf16[8,1,6144]",)
            and exact_layout_source(operand(item, 0), callers[-1])
            and collective is not None
            and unwrap_layout(operand(collective, 0)) is item
        ]
        if len(slices) != 1:
            errors.append(
                "M32 fused stack exact live-row slice is absent or ambiguous"
            )
            return callers, None, errors
        return callers, slices[0], errors

    if collective is not None:
        source_convolutions = {
            down_by_rank[rank].name
            for rank, value in down_value_by_rank.items()
            if _value_depends_on(module, collective, value)
        }
        all_stack_candidates = [
            item
            for item in module.instructions
            if item.raw_opcode == "concatenate"
            and _shape_signatures(item.result_shapes)
            in {
                (f"bf16[8,{compile_rows},6144]",),
                (f"f32[8,{compile_rows},6144]",),
            }
            and len(item.operand_names) == 8
        ]
        live_row_slices: list[Any] = []
        fused_stack_callers: list[Any] = []
        fused_stack_errors: list[str] = []
        if compile_rows == 32:
            fused_stack_callers, fused_slice, fused_stack_errors = (
                exact_m32_update_stack()
            )
            if fused_stack_callers or fused_stack_errors:
                stack_candidates = []
                if fused_stack_errors:
                    violations.extend(fused_stack_errors)
                elif fused_slice is not None:
                    live_row_slices = [fused_slice]
                    source_convolutions = {item.name for item in down}
                    lineage["m32_fused_stack_callers"] = [
                        item.name for item in fused_stack_callers
                    ]
                    lineage["ordered_stack_sources"] = [
                        [rank] for rank in range(8)
                    ]
            else:
                stack_slice_pairs = [
                    (stack, item)
                    for stack in all_stack_candidates
                    for item in module.instructions
                    if is_m32_row0_slice(item)
                    and exact_layout_source(operand(item, 0), stack)
                    and (
                        (external := fully_externalized_value(item)) is not None
                    )
                    and unwrap_layout(operand(collective, 0))
                    is unwrap_layout(external)
                ]
                stack_candidates = [stack for stack, _item in stack_slice_pairs]
                live_row_slices = [item for _stack, item in stack_slice_pairs]
            lineage["m32_live_row_slices"] = [
                item.name for item in live_row_slices
            ]
        else:
            stack_candidates = [
                item
                for item in all_stack_candidates
                if _value_depends_on(module, collective, item)
            ]
        lineage["collective_convolution_sources"] = sorted(source_convolutions)
        if source_convolutions != {item.name for item in down}:
            violations.append("collective does not consume exactly eight down results")
        if not fused_stack_callers and len(stack_candidates) != 1:
            violations.append(
                "optimized dense stack is absent or ambiguous: "
                f"{[item.name for item in stack_candidates]}"
            )
        elif not fused_stack_callers and all(
            rank in down_by_rank for rank in range(8)
        ):
            stack = stack_candidates[0]
            stack_operands = [
                by_key.get((stack.computation, name))
                for name in stack.operand_names
            ]
            ordered_sources = [
                [
                    rank
                    for rank in range(8)
                    if rank in down_value_by_rank
                    and stack_operand is not None
                    and (
                        exact_layout_source(
                            stack_operand, down_value_by_rank[rank]
                        )
                        if compile_rows == 32
                        else _value_depends_on(
                            module, stack_operand, down_value_by_rank[rank]
                        )
                    )
                ]
                for stack_operand in stack_operands
            ]
            lineage["ordered_stack_sources"] = ordered_sources
            if ordered_sources != [[rank] for rank in range(8)]:
                violations.append("optimized dense stack row order drifted")
        if compile_rows == 32 and len(live_row_slices) != 1:
            violations.append(
                "optimized M32 exact live-row slice drifted: "
                f"{lineage['m32_live_row_slices']}"
            )
    roots = [
        item
        for item in module.instructions
        if item.computation.startswith("ENTRY ")
        and item.raw_line.lstrip().startswith("ROOT ")
    ]
    by_key = {_instruction_key(item): item for item in module.instructions}
    expected_result_shapes = (
        ("bf16[1,6144]",)
        if layer1_only
        else ("bf16[1,6144]", "bf16[1,6144]")
    )
    if (
        len(roots) != 1
        or _shape_signatures(roots[0].result_shapes) != expected_result_shapes
        or collective is None
    ):
        violations.append("probe entry result geometry drifted")
    else:
        if layer1_only:
            dense_result = None
            layer1_result = roots[0]
        else:
            returned = [
                by_key.get((roots[0].computation, name))
                for name in roots[0].operand_names
            ]
            dense_result = returned[0] if len(returned) == 2 else None
            layer1_result = returned[1] if len(returned) == 2 else None
        if (
            layer1_result is None
            or (not layer1_only and dense_result is None)
            or not _value_depends_on(module, layer1_result, collective)
            or (
                dense_result is not None
                and not _value_depends_on(module, dense_result, collective)
            )
        ):
            violations.append("probe results bypass the dense StrategyND collective")
        else:
            dense_consumer = dense_result or layer1_result
            entry_parameters = [
                item
                for item in module.instructions
                if item.computation.startswith("ENTRY ")
                and item.raw_opcode == "parameter"
            ]
            results = (
                [layer1_result]
                if layer1_only
                else [dense_result, layer1_result]
            )
            parameter_sources = [
                {
                    parameter.name
                    for parameter in entry_parameters
                    if _value_depends_on(module, result, parameter)
                }
                for result in results
                if result is not None
            ]
            lineage["result_parameter_sources"] = [
                sorted(values) for values in parameter_sources
            ]
            if layer1_only:
                expected_parameters = {item.name for item in entry_parameters}
                if parameter_sources != [expected_parameters]:
                    violations.append(
                        "layer-1-only result does not consume every exact input"
                    )
            elif len(parameter_sources) != 2:
                violations.append("probe result parameter lineage is incomplete")
            else:
                layer1_only_sources = parameter_sources[1] - parameter_sources[0]
                layer1_only_shapes = {
                    shape
                    for parameter in entry_parameters
                    if parameter.name in layer1_only_sources
                    for shape in _shape_signatures(parameter.result_shapes)
                }
                lineage["layer1_only_parameter_shapes"] = sorted(
                    layer1_only_shapes
                )
                if not {
                    "bf16[1,6144]",
                    "bf16[6144]",
                }.issubset(layer1_only_shapes):
                    violations.append(
                        "layer-1 result bypasses residual or RMSNorm weight"
                    )
            association_scope = (
                "greenfield_strategy_nd_row0_dense_convolution_down/"
                "greenfield_strategy_nd_row0_association"
            )
            association_adds = [
                item
                for item in module.instructions
                if item.raw_opcode == "add"
                and association_scope in (item.op_name or "")
            ]
            association_shapes: Counter[tuple[int, ...]] = Counter()
            for item in association_adds:
                dimensions = (
                    item.result_shapes[0].dimensions
                    if len(item.result_shapes) == 1
                    else ()
                )
                while dimensions and dimensions[0] == 1:
                    dimensions = dimensions[1:]
                association_shapes[dimensions] += 1
            expected_association_shapes = {
                "y": association_shapes[(2, 4, 1, 2048)],
                "x": association_shapes[(4, 1, 6144)],
                "z": association_shapes[(256,)],
            }
            lineage["association_add_shapes"] = expected_association_shapes
            if expected_association_shapes != {"y": 9, "x": 1, "z": 72}:
                violations.append(
                    "optimized StrategyND y/x/z add geometry drifted: "
                    f"{expected_association_shapes}"
                )
            def association_add_is_live(item: Any) -> bool:
                values = exact_external_values(item)
                if len(values) == 1:
                    external = values[0]
                else:
                    computation = _computation_id(item.computation)
                    roots = [
                        value
                        for value in instructions_by_computation.get(
                            computation, ()
                        )
                        if value.raw_line.lstrip().startswith("ROOT ")
                    ]
                    callers = callers_by_computation.get(computation, ())
                    if (
                        len(roots) != 1
                        or len(callers) != 1
                        or not local_depends(roots[0], item)
                    ):
                        return False
                    external = callers[0]
                return _value_depends_on(
                    module, external, collective
                ) and _value_depends_on(module, dense_consumer, external)

            if expected_association_shapes == {"y": 9, "x": 1, "z": 72} and any(
                not association_add_is_live(item) for item in association_adds
            ):
                violations.append("optimized StrategyND add lineage drifted")
            if expected_association_shapes == {"y": 9, "x": 1, "z": 72}:
                def normalized_dimensions(item: Any) -> tuple[int, ...]:
                    dimensions = item.result_shapes[0].dimensions
                    while dimensions and dimensions[0] == 1:
                        dimensions = dimensions[1:]
                    return dimensions

                add_by_external_key = {
                    _instruction_key(values[0]): _instruction_key(item)
                    for item in association_adds
                    if len(values := exact_external_values(item)) == 1
                }

                def nearest_adds(
                    item: Any | None,
                    candidates: set[tuple[str, str]],
                ) -> set[tuple[str, str]]:
                    if item is None:
                        return set()
                    pending = [_instruction_key(item)]
                    seen: set[tuple[str, str]] = set()
                    found: set[tuple[str, str]] = set()
                    while pending:
                        key = pending.pop()
                        if key in seen:
                            continue
                        seen.add(key)
                        if key in candidates:
                            found.add(key)
                            continue
                        external_add = add_by_external_key.get(key)
                        if external_add in candidates:
                            found.add(external_add)
                            continue
                        value = by_key.get(key)
                        if value is not None:
                            pending.extend(
                                (value.computation, name)
                                for name in value.operand_names
                            )
                    return found

                all_association_add_keys = {
                    _instruction_key(item) for item in association_adds
                }

                def slice_ranges(item: Any) -> tuple[tuple[int, int], ...] | None:
                    match = re.search(r"\bslice=\{([^}]*)\}", item.raw_line)
                    if match is None:
                        return None
                    ranges = tuple(
                        (int(start), int(end))
                        for start, end in re.findall(
                            r"\[([0-9]+):([0-9]+)\]", match.group(1)
                        )
                    )
                    return ranges or None

                def branch_slices(
                    item: Any | None,
                    computation: str,
                ) -> list[Any]:
                    if item is None or item.computation != computation:
                        return []
                    pending = [_instruction_key(item)]
                    seen: set[tuple[str, str]] = set()
                    result = []
                    while pending:
                        key = pending.pop()
                        if key in seen or key in all_association_add_keys:
                            continue
                        seen.add(key)
                        value = by_key.get(key)
                        if value is None or value.computation != computation:
                            continue
                        if value.raw_opcode == "slice":
                            result.append(value)
                        if value.is_collective:
                            continue
                        pending.extend(
                            (value.computation, name)
                            for name in value.operand_names
                        )
                    return result

                def branch_identity(
                    item: Any | None,
                    computation: str,
                    label: str,
                ) -> tuple[int, int] | None:
                    slices = [
                        ranges
                        for value in branch_slices(item, computation)
                        if (ranges := slice_ranges(value)) is not None
                    ]
                    if label == "y":
                        row_slices = [
                            ranges
                            for ranges in slices
                            if len(ranges) == 5
                            and ranges[0][1] == ranges[0][0] + 1
                            and 0 <= ranges[0][0] < 4
                            and ranges[1:4]
                            == ((0, 2), (0, 4), (0, 1))
                            and ranges[4][1] - ranges[4][0] == 2048
                        ]
                        broad_slices = [
                            ranges
                            for ranges in slices
                            if len(ranges) == 5
                            and ranges[:4]
                            == ((0, 4), (0, 2), (0, 4), (0, 1))
                            and ranges[4][1] - ranges[4][0] == 2048
                            and ranges[4][0] in (0, 2048, 4096)
                        ]
                        if len(row_slices) != 1 or len(broad_slices) > 1:
                            return None
                        band_source = (
                            broad_slices[0] if broad_slices else row_slices[0]
                        )
                        if band_source[4][0] not in (0, 2048, 4096):
                            return None
                        return row_slices[0][0][0], band_source[4][0] // 2048
                    if label == "z":
                        row_slices = [
                            ranges
                            for ranges in slices
                            if len(ranges) == 3
                            and ranges[0][1] == ranges[0][0] + 1
                            and 0 <= ranges[0][0] < 4
                            and ranges[1] == (0, 1)
                            and ranges[2][1] - ranges[2][0] == 256
                        ]
                        broad_slices = [
                            ranges
                            for ranges in slices
                            if len(ranges) == 3
                            and ranges[0] == (0, 4)
                            and ranges[1] == (0, 1)
                            and ranges[2][1] - ranges[2][0] == 256
                            and ranges[2][0] in range(0, 6144, 256)
                        ]
                        if len(row_slices) != 1 or len(broad_slices) > 1:
                            return None
                        segment_source = (
                            broad_slices[0] if broad_slices else row_slices[0]
                        )
                        if segment_source[2][0] not in range(0, 6144, 256):
                            return None
                        return (
                            row_slices[0][0][0],
                            segment_source[2][0] // 256,
                        )
                    if label == "x":
                        row_slices = [
                            ranges
                            for ranges in slices
                            if len(ranges) == 4
                            and ranges[0][1] == ranges[0][0] + 1
                            and 0 <= ranges[0][0] < 2
                            and ranges[1:]
                            == ((0, 4), (0, 1), (0, 6144))
                        ]
                        if len(row_slices) != 1:
                            return None
                        return row_slices[0][0][0], 0
                    raise AssertionError(label)

                def component_value(root: Any) -> Any | None:
                    values = exact_external_values(root)
                    return values[0] if len(values) == 1 else None

                def exact_components(
                    label: str,
                    values: list[Any],
                    expected_count: int,
                ) -> tuple[bool, dict[int, Any], dict[str, Any]]:
                    candidates = {_instruction_key(item) for item in values}
                    components: dict[int, Any] = {}
                    pairings: dict[str, list[list[int]]] = {}
                    exact = True
                    parents: dict[tuple[str, str], set[tuple[str, str]]] = {}
                    for key in candidates:
                        value = by_key[key]
                        if (
                            len(value.operand_names) != 2
                            or unwrap(operand(value, 0))
                            is unwrap(operand(value, 1))
                            or not exact_bf16_round(value)
                        ):
                            exact = False
                        parents[key] = set().union(
                            *(
                                nearest_adds(operand(value, index), candidates)
                                for index in range(2)
                            )
                        )
                        parents[key].discard(key)
                    children = Counter(
                        child
                        for parent_values in parents.values()
                        for child in parent_values
                    )
                    roots = [key for key in candidates if children[key] == 0]
                    leaves = [key for key, values in parents.items() if not values]
                    if (
                        len(candidates) != expected_count * 3
                        or len(roots) != expected_count
                        or len(leaves) != expected_count * 2
                        or any(len(values) not in (0, 2) for values in parents.values())
                        or any(count != 1 for count in children.values())
                    ):
                        exact = False
                    for root_key in roots:
                        leaf_adds = parents.get(root_key, set())
                        if (
                            len(leaf_adds) != 2
                            or any(parents.get(key) for key in leaf_adds)
                        ):
                            exact = False
                            continue
                        component_pairs: list[frozenset[int]] = []
                        component_ids: set[int] = set()
                        for leaf_key in leaf_adds:
                            leaf = by_key[leaf_key]
                            identities = [
                                branch_identity(
                                    operand(leaf, index), leaf.computation, label
                                )
                                for index in range(2)
                            ]
                            if (
                                any(value is None for value in identities)
                                or identities[0] == identities[1]
                            ):
                                exact = False
                                continue
                            typed_identities = [
                                value for value in identities if value is not None
                            ]
                            if len({value[1] for value in typed_identities}) != 1:
                                exact = False
                                continue
                            component_ids.add(typed_identities[0][1])
                            component_pairs.append(
                                frozenset(value[0] for value in typed_identities)
                            )
                        if len(component_ids) != 1 or len(component_pairs) != 2:
                            exact = False
                            continue
                        component_id = next(iter(component_ids))
                        expected_pairs = (
                            {frozenset((0, 3)), frozenset((1, 2))}
                            if component_id % 2
                            else {frozenset((0, 1)), frozenset((2, 3))}
                        )
                        if (
                            set(component_pairs) != expected_pairs
                            or set().union(*component_pairs) != set(range(4))
                            or component_id in components
                        ):
                            exact = False
                            continue
                        value = component_value(by_key[root_key])
                        if value is None:
                            exact = False
                            continue
                        components[component_id] = value
                        pairings[str(component_id)] = sorted(
                            [sorted(pair) for pair in component_pairs]
                        )
                    if set(components) != set(range(expected_count)):
                        exact = False
                    return exact, components, {
                        "component_ids": sorted(components),
                        "leaf_pairings": pairings,
                    }

                def ordered_concatenate(
                    components: dict[int, Any],
                    result_dimensions: tuple[int, ...],
                    concat_dimension: int,
                    consumer: Any,
                ) -> tuple[Any | None, list[list[int]]]:
                    candidates = []
                    orders = []
                    for item in module.instructions:
                        if (
                            item.raw_opcode != "concatenate"
                            or normalized_dimensions(item) != result_dimensions
                            or len(item.operand_names) != len(components)
                            or f"dimensions={{{concat_dimension}}}"
                            not in re.sub(r"\s+", "", item.raw_line)
                            or not _value_depends_on(module, consumer, item)
                        ):
                            continue
                        item_order = [
                            [
                                component_id
                                for component_id, value in components.items()
                                if operand(item, index) is not None
                                and unwrap(operand(item, index)) is unwrap(value)
                            ]
                            for index in range(len(item.operand_names))
                        ]
                        if item_order == [
                            [component_id]
                            for component_id in range(len(components))
                        ]:
                            candidates.append(item)
                            orders.append(item_order)
                    if len(candidates) != 1:
                        return None, orders
                    return candidates[0], orders[0]

                def parameter_index(item: Any | None) -> int | None:
                    if item is None or item.raw_opcode != "parameter":
                        return None
                    match = re.search(r"\bparameter\(([0-9]+)\)", item.raw_line)
                    return None if match is None else int(match.group(1))

                def integer_constant(item: Any | None) -> int | None:
                    if item is None or item.raw_opcode != "constant":
                        return None
                    match = re.search(r"\bconstant\((-?[0-9]+)\)", item.raw_line)
                    return None if match is None else int(match.group(1))

                def called_root(caller: Any) -> Any | None:
                    match = re.search(
                        r"\bcalls=%?([^,\s}\]]+)", caller.raw_line
                    )
                    if match is None:
                        return None
                    roots = [
                        item
                        for item in instructions_by_computation.get(
                            match.group(1), ()
                        )
                        if item.raw_line.lstrip().startswith("ROOT ")
                    ]
                    return roots[0] if len(roots) == 1 else None

                def ordered_padded_concatenate(
                    components: dict[int, Any], consumer: Any
                ) -> tuple[Any | None, list[list[int]]]:
                    matches = []
                    for caller in module.instructions:
                        if (
                            caller.raw_opcode != "fusion"
                            or normalized_dimensions(caller)
                            != (2, 4, 1, 6144)
                            or len(caller.operand_names) != 3
                            or not _value_depends_on(module, consumer, caller)
                        ):
                            continue
                        caller_order = [
                            [
                                component_id
                                for component_id, value in components.items()
                                if unwrap(operand(caller, index)) is unwrap(value)
                            ]
                            for index in range(3)
                        ]
                        flattened = [values[0] for values in caller_order if len(values) == 1]
                        if sorted(flattened) != list(range(3)):
                            continue
                        root = called_root(caller)
                        if root is None:
                            continue
                        computation = _computation_id(root.computation)
                        callee_items = instructions_by_computation.get(computation, ())
                        pads = [item for item in callee_items if item.raw_opcode == "pad"]
                        maxima = [
                            item for item in callee_items if item.raw_opcode == "maximum"
                        ]
                        negative_infinity = [
                            item
                            for item in callee_items
                            if item.raw_opcode == "constant"
                            and "constant(-inf)" in item.raw_line
                        ]
                        exact = (
                            len(pads) == 3
                            and len(maxima) == 2
                            and len(negative_infinity) == 1
                            and all(has_bf16_correction(item) for item in maxima)
                            and all(local_depends(root, item) for item in pads)
                        )

                        maximum_keys: set[tuple[str, str]] = set()

                        def maximum_pad_leaves(
                            item: Any | None,
                        ) -> set[tuple[str, str]] | None:
                            item = unwrap(item)
                            if item is None:
                                return None
                            if item.raw_opcode == "pad":
                                return {_instruction_key(item)}
                            if (
                                item.raw_opcode != "maximum"
                                or not has_bf16_correction(item)
                                or len(item.operand_names) != 2
                            ):
                                return None
                            maximum_keys.add(_instruction_key(item))
                            left = maximum_pad_leaves(operand(item, 0))
                            right = maximum_pad_leaves(operand(item, 1))
                            if left is None or right is None:
                                return None
                            return left | right

                        pad_leaves = maximum_pad_leaves(root)
                        exact = exact and (
                            pad_leaves
                            == {_instruction_key(item) for item in pads}
                            and maximum_keys
                            == {_instruction_key(item) for item in maxima}
                        )
                        for caller_index, values in enumerate(caller_order):
                            if len(values) != 1:
                                exact = False
                                continue
                            component_id = values[0]
                            parameter_pads = [
                                item
                                for item in pads
                                if parameter_index(unwrap(operand(item, 0)))
                                == caller_index
                            ]
                            low = component_id * 2048
                            high = (2 - component_id) * 2048
                            expected_padding = (
                                f"padding=0_0x0_0x0_0x{low}_{high}"
                            )
                            if (
                                len(parameter_pads) != 1
                                or expected_padding
                                not in re.sub(r"\s+", "", parameter_pads[0].raw_line)
                                or unwrap(operand(parameter_pads[0], 1))
                                is not negative_infinity[0]
                                or _shape_signatures(parameter_pads[0].operand_shapes)
                                != ("bf16[2,4,1,2048]", "bf16[]")
                                or _shape_signatures(parameter_pads[0].result_shapes)
                                != ("bf16[2,4,1,6144]",)
                            ):
                                exact = False
                        if exact:
                            matches.append(caller)
                    if len(matches) != 1:
                        return None, []
                    return matches[0], [[index] for index in range(3)]

                def ordered_dynamic_update_chain(
                    components: dict[int, Any], consumer: Any
                ) -> tuple[Any | None, list[list[int]]]:
                    callers: dict[int, Any] = {}
                    for caller in module.instructions:
                        if (
                            caller.raw_opcode != "fusion"
                            or normalized_dimensions(caller) != (6144,)
                            or len(caller.operand_names) not in (1, 2)
                        ):
                            continue
                        direct_components = [
                            (index, component_id)
                            for index in range(len(caller.operand_names))
                            for component_id, value in components.items()
                            if unwrap(operand(caller, index)) is unwrap(value)
                        ]
                        if len(direct_components) != 1:
                            continue
                        update_index, component_id = direct_components[0]
                        root = unwrap(called_root(caller))
                        if (
                            root is None
                            or root.raw_opcode != "dynamic-update-slice"
                            or len(root.operand_names) != 4
                            or _shape_signatures(root.operand_shapes)
                            != (
                                "bf16[1,6144]",
                                "bf16[1,256]",
                                "s32[]",
                                "s32[]",
                            )
                            or _shape_signatures(root.result_shapes)
                            != ("bf16[1,6144]",)
                            or parameter_index(unwrap(operand(root, 1)))
                            != update_index
                            or integer_constant(unwrap(operand(root, 2))) != 0
                            or integer_constant(unwrap(operand(root, 3)))
                            != component_id * 256
                            or component_id in callers
                        ):
                            continue
                        base = unwrap(operand(root, 0))
                        if component_id == 0:
                            exact_base = (
                                len(caller.operand_names) == 1
                                and update_index == 0
                                and base is not None
                                and base.raw_opcode == "custom-call"
                                and not base.operand_names
                                and 'custom_call_target="AllocateBuffer"'
                                in base.raw_line
                            )
                        else:
                            exact_base = (
                                len(caller.operand_names) == 2
                                and update_index == 1
                                and parameter_index(base) == 0
                            )
                        if exact_base:
                            callers[component_id] = caller
                    if set(callers) != set(range(len(components))):
                        return None, []
                    for component_id in range(1, len(components)):
                        if unwrap(operand(callers[component_id], 0)) is not unwrap(
                            callers[component_id - 1]
                        ):
                            return None, []
                    result = callers[len(components) - 1]
                    if not _value_depends_on(module, consumer, result):
                        return None, []
                    return result, [
                        [component_id] for component_id in range(len(components))
                    ]

                y_values = [
                    item
                    for item in association_adds
                    if normalized_dimensions(item) == (2, 4, 1, 2048)
                ]
                z_values = [
                    item
                    for item in association_adds
                    if normalized_dimensions(item) == (256,)
                ]
                x_values = [
                    item
                    for item in association_adds
                    if normalized_dimensions(item) == (4, 1, 6144)
                ]
                y_exact, y_components, y_details = exact_components(
                    "y", y_values, 3
                )
                z_exact, z_components, z_details = exact_components(
                    "z", z_values, 24
                )
                association_graph = {
                    "y": {"component_count": len(y_components)},
                    "x": {"component_count": 1 if len(x_values) == 1 else 0},
                    "z": {"component_count": len(z_components)},
                }
                x_exact = (
                    len(x_values) == 1
                    and len(x_values[0].operand_names) == 2
                    and exact_bf16_round(x_values[0])
                )
                x_value = component_value(x_values[0]) if x_exact else None
                if x_exact:
                    x_identities = [
                        branch_identity(
                            operand(x_values[0], index),
                            x_values[0].computation,
                            "x",
                        )
                        for index in range(2)
                    ]
                    x_exact = (
                        x_identities == [(0, 0), (1, 0)]
                        or x_identities == [(1, 0), (0, 0)]
                    ) and x_value is not None
                y_concat, y_order = (
                    ordered_concatenate(
                        y_components, (2, 4, 1, 6144), 3, x_value
                    )
                    if y_exact and x_exact and x_value is not None
                    else (None, [])
                )
                if (
                    y_concat is None
                    and y_exact
                    and x_exact
                    and x_value is not None
                ):
                    y_concat, y_order = ordered_padded_concatenate(
                        y_components, x_value
                    )
                if y_concat is None:
                    y_exact = False
                    x_exact = False
                elif not _value_depends_on(module, x_value, y_concat):
                    x_exact = False
                if x_value is None or any(
                    not _value_depends_on(module, value, x_value)
                    for value in z_components.values()
                ):
                    z_exact = False
                z_concat, z_order = (
                    ordered_concatenate(
                        z_components, (6144,), 1, dense_consumer
                    )
                    if z_exact
                    else (None, [])
                )
                if z_concat is None and z_exact:
                    z_concat, z_order = ordered_dynamic_update_chain(
                        z_components, dense_consumer
                    )
                if z_concat is None:
                    z_exact = False
                association_graph["y"].update(
                    y_details | {"ordered_components": y_order, "exact": y_exact}
                )
                association_graph["x"].update(
                    {"leaf_rows": [0, 1] if x_exact else [], "exact": x_exact}
                )
                association_graph["z"].update(
                    z_details | {"ordered_components": z_order, "exact": z_exact}
                )
                for label, exact in (
                    ("y", y_exact),
                    ("x", x_exact),
                    ("z", z_exact),
                ):
                    if not exact:
                        violations.append(
                            f"optimized StrategyND {label} physical association drifted"
                        )
                lineage["association_edge_graph"] = association_graph

            rms_scope = "greenfield_dense_convolution_layer1_rmsnorm"
            rms_items = [
                item
                for item in module.instructions
                if rms_scope in (item.op_name or "").split("/")
            ]
            rms_semantics = Counter(
                (item.op_name or "").split("/")[-1] for item in rms_items
            )
            rms_adds = [item for item in rms_items if item.raw_opcode == "add"]
            rms_reductions = [
                item for item in rms_items if item.raw_opcode == "reduce"
            ]
            rms_rsqrt = [item for item in rms_items if item.raw_opcode == "rsqrt"]
            rms_divides = [item for item in rms_items if item.raw_opcode == "divide"]
            rms_multiply_or_square = [
                item
                for item in rms_items
                if item.raw_opcode in {"multiply", "square"}
            ]
            rms_contract = {
                "add_count": len(rms_adds),
                "divide_count": len(rms_divides),
                "multiply_or_square_count": len(rms_multiply_or_square),
                "reduce_count": len(rms_reductions),
                "rsqrt_count": len(rms_rsqrt),
                "semantic_counts": dict(sorted(rms_semantics.items())),
            }
            lineage["rmsnorm_contract"] = rms_contract
            rms_rows = 32 if layer1_only else 1
            rms_f32_shape = f"f32[{rms_rows},6144]"
            rms_bf16_shape = f"bf16[{rms_rows},6144]"
            rms_reduction_shape = f"f32[{rms_rows}]"
            rms_reduction_operand_shapes = (
                _shape_signatures(rms_reductions[0].operand_shapes)
                if len(rms_reductions) == 1
                else ()
            )
            exact_m32_reduction_geometry_core = (
                not layer1_only
                or (
                    len(rms_reductions) == 1
                    and rms_reduction_operand_shapes[:1] == (rms_f32_shape,)
                    and _shape_signatures(rms_reductions[0].result_shapes)
                    == (rms_reduction_shape,)
                    and "dimensions={1}" in re.sub(
                        r"\s+", "", rms_reductions[0].raw_line
                    )
                )
            )
            entry_norm_parameters = [
                item
                for item in module.instructions
                if item.computation.startswith("ENTRY ")
                and item.raw_opcode == "parameter"
                and _shape_signatures(item.result_shapes) == ("bf16[6144]",)
            ]

            semantic_layout_only = layout_only | {
                "broadcast",
                "broadcast-in-dim",
                "copy-done",
                "copy-start",
            }

            def exact_semantic_source(value: Any | None, source: Any) -> bool:
                seen: set[tuple[str, str]] = set()
                while value is not None and _instruction_key(value) not in seen:
                    if value is source:
                        return True
                    seen.add(_instruction_key(value))
                    if (
                        value.raw_opcode in semantic_layout_only
                        and len(value.operand_names) == 1
                    ):
                        value = operand(value, 0)
                        continue
                    if (
                        value.raw_opcode == "parameter"
                        and not value.computation.startswith("ENTRY ")
                    ):
                        value = external_parameter_value(value)
                        continue
                    if value.raw_opcode == "parameter" and not value.computation.startswith(
                        "ENTRY "
                    ):
                        value = external_parameter_value(value)
                        continue
                    return False
                return value is source

            def exact_chain_source(value: Any | None, source: Any) -> bool:
                """Bind an operand to one semantic value across exact fusion exits."""

                candidates = [source]
                external = fully_externalized_value(source)
                if external is not None and external is not source:
                    candidates.append(external)
                return any(
                    exact_semantic_source(value, candidate)
                    for candidate in candidates
                )

            def exact_m32_row0_chain_source(
                value: Any | None, source: Any
            ) -> bool:
                """Bind the optimized live-row slice to one exact M32 value."""

                candidates = [source]
                external = fully_externalized_value(source)
                if external is not None and external is not source:
                    candidates.append(external)
                seen: set[tuple[str, str]] = set()
                sliced = False
                while value is not None and _instruction_key(value) not in seen:
                    if any(value is candidate for candidate in candidates):
                        return sliced
                    seen.add(_instruction_key(value))
                    if (
                        value.raw_opcode in semantic_layout_only
                        and len(value.operand_names) == 1
                    ):
                        value = operand(value, 0)
                        continue
                    if (
                        value.raw_opcode == "parameter"
                        and not value.computation.startswith("ENTRY ")
                    ):
                        value = external_parameter_value(value)
                        continue
                    if (
                        not sliced
                        and is_m32_row0_slice(value)
                        and _shape_signatures(value.operand_shapes)
                        == (rms_f32_shape,)
                        and _shape_signatures(value.result_shapes)
                        == ("f32[1,6144]",)
                        and len(value.operand_names) == 1
                    ):
                        sliced = True
                        value = operand(value, 0)
                        continue
                    return False
                return (
                    any(value is candidate for candidate in candidates)
                    and sliced
                )

            def exact_constant(
                value: Any | None, expected: np.float32
            ) -> bool:
                seen: set[tuple[str, str]] = set()
                while value is not None and _instruction_key(value) not in seen:
                    seen.add(_instruction_key(value))
                    if (
                        value.raw_opcode in semantic_layout_only
                        and len(value.operand_names) == 1
                    ):
                        value = operand(value, 0)
                        continue
                    if value.raw_opcode != "constant":
                        return False
                    match = re.search(
                        r"\bconstant\(([-+]?(?:[0-9]+(?:\.[0-9]*)?|"
                        r"\.[0-9]+)(?:[eE][-+]?[0-9]+)?)\)",
                        value.raw_line,
                    )
                    return (
                        match is not None
                        and np.float32(float(match.group(1))) == expected
                    )
                return False

            def exact_add_reducer(item: Any) -> bool:
                match = re.search(r"\bto_apply=%?([^,\s}\]]+)", item.raw_line)
                if match is None:
                    return False
                values = instructions_by_computation.get(match.group(1), ())
                parameters = [
                    value for value in values if value.raw_opcode == "parameter"
                ]
                roots = [
                    value
                    for value in values
                    if value.raw_line.lstrip().startswith("ROOT ")
                ]
                if (
                    len(parameters) != 2
                    or len(roots) != 1
                    or roots[0].raw_opcode != "add"
                    or len(roots[0].operand_names) != 2
                ):
                    return False
                root_operands = {
                    _instruction_key(unwrap_layout(operand(roots[0], index)))
                    for index in range(2)
                    if unwrap_layout(operand(roots[0], index)) is not None
                }
                return root_operands == {
                    _instruction_key(value) for value in parameters
                }

            def semantic_origins(item: Any) -> frozenset[tuple[str, str]]:
                origins = []
                for index in range(len(item.operand_names)):
                    value = operand(item, index)
                    seen: set[tuple[str, str]] = set()
                    while value is not None and _instruction_key(value) not in seen:
                        seen.add(_instruction_key(value))
                        if (
                            value.raw_opcode in semantic_layout_only
                            and len(value.operand_names) == 1
                        ):
                            value = operand(value, 0)
                            continue
                        if (
                            value.raw_opcode == "parameter"
                            and not value.computation.startswith("ENTRY ")
                        ):
                            value = external_parameter_value(value)
                            continue
                        break
                    if value is not None:
                        origins.append(_instruction_key(value))
                return frozenset(origins)

            residual_adds = [
                item
                for item in rms_adds
                if _shape_signatures(item.result_shapes)
                in {(rms_f32_shape,), (rms_bf16_shape,)}
            ]
            square_candidates = [
                item
                for item in rms_items
                if (item.op_name or "").split("/")[-1] == "square"
                and item.raw_opcode in {"multiply", "square"}
                and (
                    (
                        item.raw_opcode == "square"
                        and len(item.operand_names) == 1
                    )
                    or (
                        item.raw_opcode == "multiply"
                        and len(item.operand_names) == 2
                        and unwrap_layout(operand(item, 0))
                        is unwrap_layout(operand(item, 1))
                    )
                )
            ]
            mean_candidates = [
                item
                for item in rms_items
                if (item.op_name or "").split("/")[-1] == "div"
                and item.raw_opcode in {"divide", "multiply"}
                and len(item.operand_names) == 2
            ]
            mean_shapes = {
                f"f32[{rms_rows}]",
                f"f32[{rms_rows},1]",
            }
            if rms_rows == 1:
                mean_shapes.add("f32[]")
            variance_candidates = [
                item
                for item in rms_adds
                if _shape_signatures(item.result_shapes)[0:1]
                and _shape_signatures(item.result_shapes)[0] in mean_shapes
                and len(item.operand_names) == 2
            ]

            exact_reduction_pairs: list[tuple[Any, Any, Any]] = []
            if len(square_candidates) == 1 and len(rms_reductions) == 1:
                square = square_candidates[0]
                reduction = rms_reductions[0]
                square_source = operand(square, 0)
                matching_residual_adds = [
                    item
                    for item in residual_adds
                    if exact_chain_source(square_source, item)
                ]
                if (
                    len(matching_residual_adds) == 1
                    and len(reduction.operand_names) == 2
                    and exact_chain_source(operand(reduction, 0), square)
                    and exact_constant(operand(reduction, 1), np.float32(0.0))
                    and exact_add_reducer(reduction)
                ):
                    exact_reduction_pairs.append(
                        (reduction, square, matching_residual_adds[0])
                    )

            exact_mean_pairs: list[tuple[Any, Any, Any]] = []
            for mean in mean_candidates:
                for reduction, square, residual_add in exact_reduction_pairs:
                    if mean.raw_opcode == "divide":
                        exact = exact_chain_source(
                            operand(mean, 0), reduction
                        ) and exact_constant(
                            operand(mean, 1), np.float32(6144.0)
                        )
                    else:
                        exact = any(
                            exact_chain_source(
                                operand(mean, reduction_index), reduction
                            )
                            and exact_constant(
                                operand(mean, 1 - reduction_index),
                                np.float32(1.0 / 6144.0),
                            )
                            for reduction_index in range(2)
                        )
                    if exact:
                        exact_mean_pairs.append((mean, square, residual_add))

            exact_variance_pairs: list[tuple[Any, Any, Any]] = []
            for variance in variance_candidates:
                for mean, square, residual_add in exact_mean_pairs:
                    if any(
                        exact_chain_source(
                            operand(variance, mean_index), mean
                        )
                        and exact_constant(
                            operand(variance, 1 - mean_index),
                            np.float32(1.0e-5),
                        )
                        for mean_index in range(2)
                    ):
                        exact_variance_pairs.append(
                            (variance, square, residual_add)
                        )

            exact_rsqrt_pairs: list[tuple[Any, Any, Any]] = []
            for rsqrt in rms_rsqrt:
                if len(rsqrt.operand_names) != 1:
                    continue
                for variance, square, residual_add in exact_variance_pairs:
                    if exact_chain_source(operand(rsqrt, 0), variance):
                        exact_rsqrt_pairs.append(
                            (rsqrt, square, residual_add)
                        )

            normalized_f32_shapes = {rms_f32_shape}
            normalized_bf16_shapes = {rms_bf16_shape}
            weighted_result_shapes = {rms_bf16_shape}
            if layer1_only:
                normalized_f32_shapes.add("f32[1,6144]")
                normalized_bf16_shapes.add("bf16[1,6144]")
                weighted_result_shapes.add("bf16[1,6144]")
            exact_normalized_values: dict[tuple[str, str], Any] = {}
            for item in rms_multiply_or_square:
                if (
                    (item.op_name or "").split("/")[-1] != "mul"
                    or item.raw_opcode != "multiply"
                    or len(item.operand_names) != 2
                    or not set(_shape_signatures(item.result_shapes)).issubset(
                        normalized_f32_shapes
                    )
                    or len(item.result_shapes) != 1
                ):
                    continue
                for rsqrt, _square, reduction_residual_add in exact_rsqrt_pairs:
                    for normalized_index in range(2):
                        normalized_residual_adds = [
                            candidate
                            for candidate in residual_adds
                            if (
                                exact_chain_source(
                                    operand(item, normalized_index), candidate
                                )
                                if _shape_signatures(item.result_shapes)
                                == (rms_f32_shape,)
                                else exact_m32_row0_chain_source(
                                    operand(item, normalized_index), candidate
                                )
                            )
                        ]
                        if (
                            len(normalized_residual_adds) == 1
                            and semantic_origins(normalized_residual_adds[0])
                            == semantic_origins(reduction_residual_add)
                            and exact_chain_source(
                                operand(item, 1 - normalized_index), rsqrt
                            )
                        ):
                            exact_normalized_values[_instruction_key(item)] = item

            exact_reduction_operand_graph = (
                len(exact_reduction_pairs) == 1
                and len(exact_mean_pairs) == 1
                and len(exact_variance_pairs) == 1
                and len(exact_rsqrt_pairs) == 1
                and len(exact_normalized_values) == 1
            )

            def exact_rounded_normalized_source(value: Any | None) -> bool:
                seen: set[tuple[str, str]] = set()
                normalized = None
                while value is not None and _instruction_key(value) not in seen:
                    seen.add(_instruction_key(value))
                    if (
                        value.raw_opcode == "convert"
                        and len(value.operand_names) == 1
                        and len(value.result_shapes) == 1
                        and _shape_signatures(value.result_shapes)[0]
                        in normalized_bf16_shapes
                    ):
                        normalized = operand(value, 0)
                        break
                    if has_bf16_correction(value):
                        normalized = value
                        break
                    if (
                        value.raw_opcode in layout_only
                        and len(value.operand_names) == 1
                    ):
                        value = operand(value, 0)
                        continue
                    return False
                return (
                    normalized is not None
                    and _instruction_key(normalized) in exact_normalized_values
                )

            weighted_external_values: dict[tuple[str, str], Any] = {}
            for item in rms_items:
                if (
                    (item.op_name or "").split("/")[-1] != "mul"
                    or not exact_bf16_round(item)
                    or item.raw_opcode != "multiply"
                    or len(item.operand_names) != 2
                    or len(entry_norm_parameters) != 1
                ):
                    continue
                item_operands = [operand(item, index) for index in range(2)]
                if not any(
                    exact_semantic_source(item_operands[norm_index], entry_norm_parameters[0])
                    and exact_rounded_normalized_source(
                        item_operands[1 - norm_index]
                    )
                    for norm_index in range(2)
                ):
                    continue
                external = fully_externalized_value(item)
                if (
                    external is not None
                    and len(external.result_shapes) == 1
                    and _shape_signatures(external.result_shapes)[0]
                    in weighted_result_shapes
                ):
                    weighted_external_values[_instruction_key(external)] = external
            exact_result_binding = (
                len(weighted_external_values) == 1
                and exact_layout_source(
                    layer1_result, next(iter(weighted_external_values.values()))
                )
            )
            exact_m32_reduction_geometry = (
                exact_m32_reduction_geometry_core
                and (
                    not layer1_only
                    or (
                        exact_reduction_operand_graph
                        and len(weighted_external_values) == 1
                        and exact_result_binding
                    )
                )
            )
            rms_contract["exact_m32_reduction_geometry"] = (
                exact_m32_reduction_geometry
            )
            external_reductions = {
                _instruction_key(value): value
                for item in rms_reductions
                if (value := fully_externalized_value(item)) is not None
            }
            external_rsqrt = {
                _instruction_key(value): value
                for item in rms_rsqrt
                if (value := fully_externalized_value(item)) is not None
            }
            exact_cross_fusion_reduction_lineage = (
                exact_reduction_operand_graph
                and len(external_reductions) == 1
                and len(external_rsqrt) == 1
                and len(weighted_external_values) == 1
            )
            rms_contract["exact_reduction_operand_graph"] = (
                exact_reduction_operand_graph
            )
            rms_contract["exact_weighted_operand_graph"] = (
                len(weighted_external_values) == 1
            )
            rms_contract["exact_cross_fusion_reduction_lineage"] = (
                exact_cross_fusion_reduction_lineage
            )
            rms_contract["exact_result_binding"] = exact_result_binding
            rms_contract["weighted_external_values"] = sorted(
                item.name for item in weighted_external_values.values()
            )
            direct_rms_exact = True
            if rms_items and all(
                item.computation == layer1_result.computation
                for item in rms_items
            ):
                direct_rms_exact = exact_reduction_operand_graph
            rms_contract["direct_exact_operand_graph"] = direct_rms_exact
            if (
                not {
                    "add",
                    "div",
                    "mul",
                    "reduce_sum",
                    "rsqrt",
                    "square",
                }.issubset(rms_semantics)
                or not direct_rms_exact
                or not exact_result_binding
                or not exact_m32_reduction_geometry
                or not exact_cross_fusion_reduction_lineage
                or not all(
                    _value_depends_on(module, layer1_result, parameter)
                    for parameter in entry_parameters
                    if any(
                        shape in {"bf16[1,6144]", "bf16[6144]"}
                        for shape in _shape_signatures(parameter.result_shapes)
                    )
                )
            ):
                violations.append("optimized residual/RMSNorm graph drifted")
    custom_calls = [
        item.name
        for item in module.instructions
        if item.raw_opcode == "custom-call"
        and 'custom_call_target="tpu_custom_call"' in item.raw_line
    ]
    forbidden = [
        marker
        for marker in ("host_callback", "xla_python_cpu_callback", " outfeed(")
        if marker in optimized_hlo
    ]
    if custom_calls:
        violations.append(f"unexpected Pallas custom calls: {custom_calls}")
    if forbidden:
        violations.append(f"forbidden HLO markers: {forbidden}")
    return {
        "async_collectives": async_collectives,
        "collective_count": len(collectives),
        "compile_rows": compile_rows,
        "convolution_count": len(convolutions),
        "down_convolution_count": len(down),
        "gate_up_convolution_count": len(gate_up),
        "lineage": lineage,
        "live_rows": 1,
        "result_mode": "layer1_only" if layer1_only else "dense_and_layer1",
        "num_partitions": module.num_partitions,
        "num_replicas": module.num_replicas,
        "passed": not violations,
        "unexpected_convolutions": unexpected,
        "violations": violations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--compile-rows", type=int, choices=(1, 32), default=1)
    parser.add_argument("--layer1-only", action="store_true")
    parser.add_argument("--db538-runner", type=Path, required=True)
    parser.add_argument("--db538-runner-sha256", required=True)
    parser.add_argument("--db538-tensor", type=Path, required=True)
    parser.add_argument("--db538-tensor-sha256", required=True)
    parser.add_argument("--db538-summary", type=Path, required=True)
    parser.add_argument("--db538-summary-sha256", required=True)
    parser.add_argument("--db538-success", type=Path, required=True)
    parser.add_argument("--db538-success-sha256", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument("--accepted-m32-hlo", type=Path)
    parser.add_argument("--accepted-m32-hlo-sha256")
    parser.add_argument("--accepted-m32-summary", type=Path)
    parser.add_argument("--accepted-m32-summary-sha256")
    parser.add_argument("--accepted-m32-success", type=Path)
    parser.add_argument("--accepted-m32-success-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-output", type=Path, required=True)
    parser.add_argument("--hlo-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if REPO != Path("/home/gianl/glm-tpu-topology-rewrite"):
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    code_hash = _git_head()
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale code hash: expected={args.expected_code_hash} found={code_hash}"
        )
    if args.layer1_only and args.compile_rows != 32:
        raise RuntimeError("layer-1-only discriminator requires accepted M32 geometry")
    normalized, post_attention, accepted_layer1_bits = _load_db538(args)
    m32_arguments = (
        args.accepted_m32_hlo,
        args.accepted_m32_hlo_sha256,
        args.accepted_m32_summary,
        args.accepted_m32_summary_sha256,
        args.accepted_m32_success,
        args.accepted_m32_success_sha256,
    )
    if args.compile_rows == 1 and any(
        value is not None for value in m32_arguments
    ):
        raise RuntimeError("accepted M32 source is forbidden for the M1 arm")
    accepted_m32_source = (
        _load_accepted_m32_source(args) if args.compile_rows == 32 else {}
    )
    weights, weight_records = _load_weights(
        args.checkpoint_root,
        manifest_sha256=args.checkpoint_manifest_sha256,
    )

    import jax
    from jax import lax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from glm_tpu.greenfield.kernels.reference.rmsnorm import fused_add_rms_norm
    from glm_tpu.greenfield.kernels.stage_local import (
        STRATEGY_ND_ROW0_REDUCTION_ASSOCIATION,
        _reduce_virtual_tp32_bf16_partials,
        _virtual_dense_convolution_down_partials,
    )

    if jax.default_backend() != "tpu" or jax.local_device_count() != 4:
        raise RuntimeError("dense convolution probe requires one four-chip TPU host")
    mesh = Mesh(np.asarray(jax.local_devices()), ("lp4",))
    replicated = NamedSharding(mesh, P())
    slot_three = NamedSharding(mesh, P("lp4", None, None))
    arguments = (
        jax.device_put(normalized, replicated),
        jax.device_put(post_attention, replicated),
        jax.device_put(weights[_DENSE_WEIGHT_NAMES[0]], slot_three),
        jax.device_put(weights[_DENSE_WEIGHT_NAMES[1]], slot_three),
        jax.device_put(weights[_DENSE_WEIGHT_NAMES[2]], slot_three),
        jax.device_put(weights[_DENSE_WEIGHT_NAMES[3]], slot_three),
        jax.device_put(weights[_DENSE_WEIGHT_NAMES[4]], slot_three),
        jax.device_put(weights[_DENSE_WEIGHT_NAMES[5]], slot_three),
        jax.device_put(weights[_DENSE_WEIGHT_NAMES[6]], replicated),
    )
    in_specs = (
        P(),
        P(),
        P("lp4", None, None),
        P("lp4", None, None),
        P("lp4", None, None),
        P("lp4", None, None),
        P("lp4", None, None),
        P("lp4", None, None),
        P(),
    )
    groups = ((0, 1, 2, 3),)

    def local(
        normalized_input: Any,
        post_attention_residual: Any,
        gate_bits_slot: Any,
        gate_scale_slot: Any,
        up_bits_slot: Any,
        up_scale_slot: Any,
        down_bits_slot: Any,
        down_scale_slot: Any,
        layer1_norm: Any,
    ) -> Any:
        dense_input = normalized_input
        if args.compile_rows == 32:
            with jax.named_scope(
                "greenfield_dense_convolution_m32_compile_geometry"
            ):
                dense_input = jnp.pad(
                    normalized_input,
                    ((0, 31), (0, 0)),
                    mode="constant",
                    constant_values=jnp.bfloat16(0),
                )
        partials = _virtual_dense_convolution_down_partials(
            dense_input,
            gate_bits_slot[0],
            gate_scale_slot[0],
            up_bits_slot[0],
            up_scale_slot[0],
            down_bits_slot[0],
            down_scale_slot[0],
            block_shape=(128, 128),
            compile_rows=args.compile_rows,
        )
        if args.compile_rows == 32:
            with jax.named_scope(
                "greenfield_dense_convolution_m32_geometry_anchor"
            ):
                partials = lax.optimization_barrier(partials)
            with jax.named_scope(
                "greenfield_dense_convolution_m32_live_row"
            ):
                partials = partials[:, :1, :]
        with jax.named_scope(
            "greenfield_strategy_nd_row0_dense_convolution_down"
        ):
            dense_update = _reduce_virtual_tp32_bf16_partials(
                partials,
                axis_name="lp4",
                groups=groups,
                association=STRATEGY_ND_ROW0_REDUCTION_ASSOCIATION,
            )
        rms_dense_update = dense_update
        rms_residual = post_attention_residual
        if args.layer1_only:
            with jax.named_scope(
                "greenfield_dense_convolution_layer1_m32_reduction_geometry"
            ):
                rms_dense_update = jnp.pad(
                    dense_update,
                    ((0, 31), (0, 0)),
                    mode="constant",
                    constant_values=jnp.bfloat16(0),
                )
                rms_residual = jnp.pad(
                    post_attention_residual,
                    ((0, 31), (0, 0)),
                    mode="constant",
                    constant_values=jnp.bfloat16(0),
                )
        with jax.named_scope("greenfield_dense_convolution_layer1_rmsnorm"):
            layer1 = fused_add_rms_norm(
                rms_dense_update,
                rms_residual,
                layer1_norm,
                epsilon=1e-5,
            )[0]
        if args.layer1_only:
            with jax.named_scope(
                "greenfield_dense_convolution_layer1_m32_live_row"
            ):
                layer1 = layer1[:1, :]
        return layer1 if args.layer1_only else (dense_update, layer1)

    out_specs = P() if args.layer1_only else (P(), P())
    mapped = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )
    args.hlo_dir.mkdir(parents=True, exist_ok=True)
    lowered = jax.jit(mapped).lower(*arguments)
    stablehlo = lowered.as_text()
    stablehlo_contract = _validate_stablehlo(
        stablehlo,
        compile_rows=args.compile_rows,
        layer1_only=args.layer1_only,
    )
    stablehlo_path = args.hlo_dir / "dense_convolution.stablehlo.mlir"
    stablehlo_path.write_text(stablehlo)
    if not stablehlo_contract["passed"]:
        raise RuntimeError(
            f"dense convolution StableHLO failed: {stablehlo_contract}"
        )
    compiled = lowered.compile()
    optimized_hlo = compiled.as_text()
    optimized_contract = _validate_optimized_hlo(
        optimized_hlo,
        compile_rows=args.compile_rows,
        layer1_only=args.layer1_only,
    )
    optimized_path = args.hlo_dir / "dense_convolution.optimized_hlo.txt"
    optimized_path.write_text(optimized_hlo)
    if not optimized_contract["passed"]:
        raise RuntimeError(
            f"dense convolution optimized HLO failed: {optimized_contract}"
        )
    compiled_result = compiled(*arguments)
    if args.layer1_only:
        layer1 = compiled_result
        jax.block_until_ready(layer1)
        dense_update_bits = None
    else:
        dense_update, layer1 = compiled_result
        jax.block_until_ready((dense_update, layer1))
        dense_update_bits = np.ascontiguousarray(np.asarray(dense_update)).view(
            np.uint16
        )
    layer1_bits = np.ascontiguousarray(np.asarray(layer1).reshape(6144)).view(
        np.uint16
    )
    comparison = _compare_bits(accepted_layer1_bits, layer1_bits)
    exact = comparison["elementwise_exact"]
    arm = (
        "accepted_m32_dense_cross_layer"
        if args.layer1_only
        else (
            "accepted_m32_dense_convolution"
            if args.compile_rows == 32
            else "accepted_dense_convolution"
        )
    )
    classification = f"{arm}_{'exact' if exact else 'nonexact'}"
    result = {
        "artifact_kind": "glm52_layer0_dense_convolution_probe",
        "classification": classification,
        "code_hash": code_hash,
        "compile_rows": args.compile_rows,
        "diagnostic_dead_rows": args.compile_rows - 1,
        "exact": exact,
        "exact_arms": [arm] if exact else [],
        "hlo": {
            "optimized_contract": optimized_contract,
            "optimized_sha256": sha256(optimized_hlo.encode()).hexdigest(),
            "stablehlo_contract": stablehlo_contract,
            "stablehlo_sha256": sha256(stablehlo.encode()).hexdigest(),
        },
        "layer1_comparison": comparison,
        "performance_claim": False,
        "live_rows": 1,
        "result_mode": "layer1_only" if args.layer1_only else "dense_and_layer1",
        "position": _POSITION,
        "source": {
            "checkpoint_manifest_sha256": args.checkpoint_manifest_sha256,
            "db538_runner_sha256": args.db538_runner_sha256,
            "db538_tensor_sha256": args.db538_tensor_sha256,
            "db538_summary_sha256": args.db538_summary_sha256,
            "db538_success_sha256": args.db538_success_sha256,
            "post_attention_residual_sha256": _POST_ATTENTION_RESIDUAL_SHA256,
            **accepted_m32_source,
        },
        "status": "SUCCESS",
        "weight_records": weight_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.tensor_output.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        "accepted_layer1_normalized_bfloat16_bits": accepted_layer1_bits,
        "compile_rows": np.array(args.compile_rows, dtype=np.int32),
        "layer1_normalized_bfloat16_bits": layer1_bits,
        "normalized_mlp_bfloat16_bits": np.ascontiguousarray(normalized).view(
            np.uint16
        ),
        "post_attention_residual_bfloat16_bits": np.ascontiguousarray(
            post_attention
        ).view(np.uint16),
    }
    if dense_update_bits is not None:
        tensors["dense_update_bfloat16_bits"] = dense_update_bits
    np.savez(args.tensor_output, **tensors)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
