"""Exactness and HLO gates for the bounded PP16 dense boundary probe."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import re
from typing import Any

import ml_dtypes
import numpy as np

from ..sharding.hlo_contract import parse_hlo_module
from .association_fingerprint import replay_db533_strategy_nd_row0_bits


_HOST_MARKERS = (
    "host_callback",
    "outside_compilation",
    "xla_ffi_python_cpu_callback",
    "xla_python_cpu_callback",
)
_FORBIDDEN_STABLE_COLLECTIVES = (
    "stablehlo.all_gather",
    "stablehlo.all_to_all",
    "stablehlo.collective_permute",
    "stablehlo.reduce_scatter",
)
PP16_DENSE_BOUNDARY_ACCEPTED_DENSE_SHA256 = (
    "efde853254c03dd18a5f5f22733630ce0e785dfbb4eba09c41eea9085e47b4fc"
)
PP16_DENSE_BOUNDARY_ACCEPTED_RESIDUAL_SHA256 = (
    "35a601b7f174eb9204848757f709549a31e82774309929f4071c61849626044c"
)

_PP16_DENSE_NAMES = (
    "dense.slot_00.gate.weight_bits",
    "dense.slot_00.gate.scale_inv",
    "dense.slot_00.up.weight_bits",
    "dense.slot_00.up.scale_inv",
    "dense.slot_00.down.weight_bits",
    "dense.slot_00.down.scale_inv",
)
_PP16_FINAL_LAYOUT_CONTRACT = {
    "dense.slot_00.merged_gate_up.weight_bits_in_out": {
        "byte_count": 150_994_944,
        "dtype": "uint8",
        "sha256": (
            "82c93c0fafda7afa3853e3a689aace78be61bec88c5ef779bb0e9efeb834facf"
        ),
        "shape": [2, 16, 6144, 768],
    },
    "dense.slot_00.merged_gate_up.scale_inv_in_out": {
        "byte_count": 4_718_592,
        "dtype": "float32",
        "sha256": (
            "9b4bfee8b15d04545a277ba4e1ea0d3b73427a0f2cebf0a5f5fa4d7ab32bf8b3"
        ),
        "shape": [2, 16, 48, 768],
    },
    "dense.slot_00.down.weight_bits_in_out": {
        "byte_count": 75_497_472,
        "dtype": "uint8",
        "sha256": (
            "8654c1ebb6f0ef29b1d3919699c08ca2b81e058bf3f9cd0827a9889994d72f7e"
        ),
        "shape": [2, 16, 384, 6144],
    },
    "dense.slot_00.down.scale_inv_in_out": {
        "byte_count": 2_359_296,
        "dtype": "float32",
        "sha256": (
            "f37e87987745d263d152ff17415d21d572fa3c73ad5888207f1aa44916f74bdd"
        ),
        "shape": [2, 16, 3, 6144],
    },
}


def pack_pp16_dense_final_layout(
    owners: tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
) -> tuple[tuple[np.ndarray, ...], dict[str, dict[str, Any]]]:
    """Pack two PP16 owners into 16 accepted ``[in, out]`` rank shards."""

    if len(owners) != 2:
        raise ValueError("PP16 final-layout dense pack requires two owners")
    expected = {
        _PP16_DENSE_NAMES[0]: ((6144, 6144), np.uint8),
        _PP16_DENSE_NAMES[1]: ((48, 48), np.float32),
        _PP16_DENSE_NAMES[2]: ((6144, 6144), np.uint8),
        _PP16_DENSE_NAMES[3]: ((48, 48), np.float32),
        _PP16_DENSE_NAMES[4]: ((6144, 6144), np.uint8),
        _PP16_DENSE_NAMES[5]: ((48, 48), np.float32),
    }
    for owner in owners:
        if set(owner) != set(_PP16_DENSE_NAMES):
            raise ValueError("PP16 final-layout dense source set drifted")
        for name, (shape, dtype) in expected.items():
            value = owner[name]
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(
                    f"PP16 final-layout source {name!r} drifted: "
                    f"shape={value.shape} dtype={value.dtype}"
                )

    merged_bits = np.empty((2, 16, 6144, 768), dtype=np.uint8)
    merged_scale = np.empty((2, 16, 48, 768), dtype=np.float32)
    down_bits = np.empty((2, 16, 384, 6144), dtype=np.uint8)
    down_scale = np.empty((2, 16, 3, 6144), dtype=np.float32)
    for owner_index, owner in enumerate(owners):
        gate = owner[_PP16_DENSE_NAMES[0]]
        gate_scale = owner[_PP16_DENSE_NAMES[1]]
        up = owner[_PP16_DENSE_NAMES[2]]
        up_scale = owner[_PP16_DENSE_NAMES[3]]
        down = owner[_PP16_DENSE_NAMES[4]]
        compact_down_scale = owner[_PP16_DENSE_NAMES[5]]
        for shard in range(16):
            start = shard * 384
            stop = start + 384
            scale_start = shard * 3
            scale_stop = scale_start + 3
            merged_bits[owner_index, shard] = np.concatenate(
                (gate[start:stop].T, up[start:stop].T), axis=1
            )
            merged_scale[owner_index, shard] = np.concatenate(
                (
                    np.repeat(
                        gate_scale[scale_start:scale_stop].T, 128, axis=1
                    ),
                    np.repeat(
                        up_scale[scale_start:scale_stop].T, 128, axis=1
                    ),
                ),
                axis=1,
            )
            down_bits[owner_index, shard] = down[:, start:stop].T
            down_scale[owner_index, shard] = np.repeat(
                compact_down_scale[:, scale_start:scale_stop].T,
                128,
                axis=1,
            )
    packed = tuple(
        np.ascontiguousarray(value)
        for value in (merged_bits, merged_scale, down_bits, down_scale)
    )
    names = (
        "dense.slot_00.merged_gate_up.weight_bits_in_out",
        "dense.slot_00.merged_gate_up.scale_inv_in_out",
        "dense.slot_00.down.weight_bits_in_out",
        "dense.slot_00.down.scale_inv_in_out",
    )
    records = {
        name: {
            "byte_count": int(value.nbytes),
            "dtype": str(value.dtype),
            "sha256": sha256(value.tobytes(order="C")).hexdigest(),
            "shape": list(value.shape),
            "transform": "pp16_2x16_accepted_in_out_dense",
        }
        for name, value in zip(names, packed, strict=True)
    }
    return packed, records


def validate_pp16_dense_final_layout_records(
    records: dict[str, dict[str, Any]],
) -> None:
    """Bind the 2x16 diagnostic pack to DB550's exact 4x8 payload bytes."""

    expected = {
        name: {
            **contract,
            "transform": "pp16_2x16_accepted_in_out_dense",
        }
        for name, contract in _PP16_FINAL_LAYOUT_CONTRACT.items()
    }
    if records != expected:
        raise ValueError(
            "PP16 final-layout dense bytes drifted from the DB550-proven payload"
        )


def derive_expected_dense_boundary_bits(
    dense_partial_bits: np.ndarray,
    post_attention_residual_bits: np.ndarray,
    model_axis_device_ids: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Derive accepted dense update and carried residual from sealed leaves."""

    dense_partial_bits = np.ascontiguousarray(dense_partial_bits)
    post_attention_residual_bits = np.ascontiguousarray(
        post_attention_residual_bits
    )
    if (
        dense_partial_bits.dtype != np.uint16
        or dense_partial_bits.shape != (4, 8, 1, 6144)
    ):
        raise ValueError("dense boundary partial bits geometry drifted")
    if (
        post_attention_residual_bits.dtype != np.uint16
        or post_attention_residual_bits.shape != (1, 6144)
    ):
        raise ValueError("dense boundary carried-input bits geometry drifted")
    dense_bits = replay_db533_strategy_nd_row0_bits(
        dense_partial_bits.reshape(32, 6144), model_axis_device_ids
    ).reshape(1, 6144)
    carried_bits = np.ascontiguousarray(
        np.asarray(
            dense_bits.view(ml_dtypes.bfloat16).astype(np.float32)
            + post_attention_residual_bits.view(ml_dtypes.bfloat16).astype(
                np.float32
            ),
            dtype=ml_dtypes.bfloat16,
        )
    ).view(np.uint16)
    return dense_bits, carried_bits


def replay_pp16_strategy_nd_y_x_z_bits(
    dense_partial_bits: np.ndarray,
) -> np.ndarray:
    """Replay DB533 as local-y, one LP2-x combine, then local-z."""

    bits = np.ascontiguousarray(dense_partial_bits)
    if bits.dtype != np.uint16 or bits.shape != (4, 8, 1, 6144):
        raise ValueError("PP16 StrategyND replay requires 32 sealed BF16 leaves")
    partials = bits.reshape(32, 1, 6144).view(ml_dtypes.bfloat16)

    def add(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.asarray(
            left.astype(np.float32) + right.astype(np.float32),
            dtype=ml_dtypes.bfloat16,
        )

    def reduce_four(values: np.ndarray, *, cross: bool) -> np.ndarray:
        if cross:
            return add(add(values[0], values[3]), add(values[1], values[2]))
        return add(add(values[0], values[1]), add(values[2], values[3]))

    owner_y_reduced = []
    for owner in range(2):
        physical_y_z = partials[owner * 16 : (owner + 1) * 16].reshape(
            4, 4, 1, 6144
        )
        owner_y_reduced.append(
            np.concatenate(
                (
                    reduce_four(physical_y_z[..., :2048], cross=False),
                    reduce_four(
                        physical_y_z[..., 2048:4096], cross=True
                    ),
                    reduce_four(physical_y_z[..., 4096:], cross=False),
                ),
                axis=-1,
            )
        )
    x_reduced = add(owner_y_reduced[0], owner_y_reduced[1])
    reduced = np.concatenate(
        tuple(
            reduce_four(
                x_reduced[..., start : start + 256],
                cross=bool((start // 256) % 2),
            )
            for start in range(0, 6144, 256)
        ),
        axis=-1,
    )
    return np.ascontiguousarray(reduced).view(np.uint16)


def exact_bfloat16_bits(
    expected: np.ndarray, observed: np.ndarray
) -> dict[str, Any]:
    """Compare two BF16 bit arrays without silently converting arithmetic."""

    expected = np.ascontiguousarray(expected)
    observed = np.ascontiguousarray(observed)
    if expected.dtype != np.uint16 or observed.dtype != np.uint16:
        raise ValueError("exact BF16 comparison requires uint16 bit arrays")
    if expected.shape != observed.shape:
        raise ValueError(
            "exact BF16 comparison shape drifted: "
            f"expected={expected.shape} observed={observed.shape}"
        )
    mismatches = np.flatnonzero(expected.reshape(-1) != observed.reshape(-1))
    return {
        "elementwise_exact": bool(mismatches.size == 0),
        "expected_sha256": sha256(expected.tobytes(order="C")).hexdigest(),
        "first_mismatch_flat_index": (
            None if mismatches.size == 0 else int(mismatches[0])
        ),
        "mismatch_count": int(mismatches.size),
        "observed_sha256": sha256(observed.tobytes(order="C")).hexdigest(),
        "shape": list(expected.shape),
    }


def validate_pp16_dense_boundary_hlo(
    stablehlo: str,
    optimized_hlo: str,
    *,
    hidden_size: int = 6144,
    association: str = "lp2_single",
) -> dict[str, Any]:
    """Require the selected LP2 dense association and no dead/global rows."""

    if association not in {
        "lp2_single",
        "strategy_nd_y_x_z",
        "strategy_nd_y_x_z_final_layout",
    }:
        raise ValueError("unknown PP16 dense boundary association")
    final_layout = association == "strategy_nd_y_x_z_final_layout"

    module = parse_hlo_module(optimized_hlo)
    violations: list[str] = []
    collectives = module.collectives
    expected_groups = ((0, 1),)
    if len(collectives) != 1:
        violations.append(
            f"expected exactly one optimized collective, found {len(collectives)}"
        )
    for collective in collectives:
        if collective.opcode != "all-reduce":
            violations.append(
                f"expected LP2 all-reduce, found {collective.opcode}"
            )
        if collective.replica_groups != expected_groups:
            violations.append(
                "LP2 collective group drifted: "
                f"expected={expected_groups} observed={collective.replica_groups}"
            )
        payloads = {
            (shape.dtype.lower(), shape.dimensions)
            for shape in collective.result_shapes
        }
        expected_payload = (
            (1, hidden_size)
            if association == "lp2_single"
            else (4, 1, hidden_size)
        )
        if not any(
            dtype in {"bf16", "bfloat16"}
            and dimensions == expected_payload
            for dtype, dimensions in payloads
        ):
            violations.append(
                "LP2 combine lost the exact association payload: "
                f"expected={expected_payload} observed={sorted(payloads)}"
            )
    if module.num_partitions != 2:
        violations.append(
            f"expected exactly two partitions, found {module.num_partitions}"
        )

    by_key = {
        (instruction.computation, instruction.name): instruction
        for instruction in module.instructions
    }

    kernel_name = (
        "greenfield_fp8_fused_block_swiglu_"
        f"m8_h{hidden_size}_i"
        f"{hidden_size if association == 'lp2_single' else 384}"
        f"_o{hidden_size}"
    )
    custom_calls = tuple(
        instruction
        for instruction in module.instructions
        if 'custom_call_target="tpu_custom_call"' in instruction.raw_line
    )
    kernel_count = sum(kernel_name in item.raw_line for item in custom_calls)
    expected_kernel_count = (
        0 if final_layout else 1 if association == "lp2_single" else 16
    )
    if (
        len(custom_calls) != expected_kernel_count
        or kernel_count != expected_kernel_count
    ):
        violations.append(
            "PP16 dense kernel set drifted: "
            f"custom_calls={len(custom_calls)} matching_kernel={kernel_count} "
            f"expected={expected_kernel_count}"
        )

    convolution_scope = "greenfield_dense_convolution_virtual_rank_"
    convolutions = tuple(
        instruction
        for instruction in module.instructions
        if instruction.raw_opcode == "convolution"
        and convolution_scope in (instruction.op_name or "")
    )
    gate_ranks: Counter[int] = Counter()
    down_ranks: Counter[int] = Counter()
    malformed_convolutions: list[str] = []

    def exact_rhs_layout(instruction: Any, shape: str) -> bool:
        if len(instruction.operand_names) != 2:
            return False
        producer = by_key.get(
            (instruction.computation, instruction.operand_names[1])
        )
        return bool(
            producer is not None
            and producer.opcode
            in {
                "bitcast",
                "copy",
                "convert",
                "fusion",
                "optimization-barrier",
                "parameter",
                "reshape",
            }
            and re.search(
                rf"=\s*{re.escape(shape)}\{{1,0(?::|\}})",
                producer.raw_line,
            )
        )

    for instruction in convolutions:
        match = re.search(
            r"greenfield_dense_convolution_virtual_rank_([0-9]{2})/",
            instruction.op_name or "",
        )
        operands = tuple(
            (shape.dtype.lower(), shape.dimensions)
            for shape in instruction.operand_shapes
        )
        results = tuple(
            (shape.dtype.lower(), shape.dimensions)
            for shape in instruction.result_shapes
        )
        exact_dimensions = "dim_labels=bf_io->bf" in "".join(
            instruction.raw_line.split()
        )
        if match is None:
            malformed_convolutions.append(instruction.name)
        elif operands == (
            ("bf16", (1, hidden_size)),
            ("bf16", (hidden_size, 768)),
        ) and results == (("f32", (1, 768)),) and exact_dimensions and (
            exact_rhs_layout(instruction, "bf16[6144,768]")
        ):
            gate_ranks[int(match.group(1))] += 1
        elif operands == (
            ("bf16", (1, 384)),
            ("bf16", (384, hidden_size)),
        ) and results == (("f32", (1, hidden_size)),) and exact_dimensions and (
            exact_rhs_layout(instruction, "bf16[384,6144]")
        ):
            down_ranks[int(match.group(1))] += 1
        else:
            malformed_convolutions.append(instruction.name)
    expected_ranks = Counter({rank: 1 for rank in range(16)})
    if final_layout:
        if (
            len(convolutions) != 32
            or gate_ranks != expected_ranks
            or down_ranks != expected_ranks
            or malformed_convolutions
        ):
            violations.append(
                "PP16 final-layout convolution set drifted: "
                f"count={len(convolutions)} gate={dict(gate_ranks)} "
                f"down={dict(down_ranks)} malformed={malformed_convolutions}"
            )
    elif convolutions:
        violations.append(
            "fused-Pallas PP16 association unexpectedly contains dense convolutions"
        )

    def computation_symbol(value: str) -> str:
        parts = value.split()
        if parts and parts[0] == "ENTRY":
            parts = parts[1:]
        return parts[0] if parts else value

    parameters: dict[str, dict[int, tuple[str, str]]] = {}
    roots: dict[str, tuple[str, str]] = {}
    for instruction in module.instructions:
        symbol = computation_symbol(instruction.computation)
        if instruction.opcode == "parameter":
            match = re.search(r"\bparameter\(([0-9]+)\)", instruction.raw_line)
            if match is not None:
                parameters.setdefault(symbol, {})[int(match.group(1))] = (
                    instruction.computation,
                    instruction.name,
                )
        if instruction.raw_line.lstrip().startswith("ROOT "):
            roots[symbol] = (instruction.computation, instruction.name)

    forward_edges: dict[
        tuple[str, str],
        list[tuple[tuple[str, str], str, int | None]],
    ] = {}

    def add_edge(
        source: tuple[str, str],
        target: tuple[str, str],
        kind: str,
        index: int | None = None,
    ) -> None:
        if source in by_key and target in by_key:
            forward_edges.setdefault(source, []).append((target, kind, index))

    call_opcodes = {"call", "conditional", "fusion", "while"}
    for instruction in module.instructions:
        target = (instruction.computation, instruction.name)
        structural_line = re.sub(
            r'"(?:\\.|[^"\\])*"',
            '""',
            re.sub(r"/\*.*?\*/", "", instruction.raw_line),
        )
        if instruction.opcode == "parameter":
            continue
        if instruction.opcode == "tuple":
            for index, operand_name in enumerate(instruction.operand_names):
                add_edge(
                    (instruction.computation, operand_name),
                    target,
                    "tuple",
                    index,
                )
        elif instruction.opcode == "get-tuple-element":
            index_match = re.search(r"\bindex=([0-9]+)", instruction.raw_line)
            if instruction.operand_names and index_match is not None:
                add_edge(
                    (instruction.computation, instruction.operand_names[0]),
                    target,
                    "get-tuple-element",
                    int(index_match.group(1)),
                )
        elif instruction.opcode not in call_opcodes:
            for operand_name in instruction.operand_names:
                add_edge(
                    (instruction.computation, operand_name),
                    target,
                    "ordinary",
                )
        called = re.search(r"\bcalls=(%[A-Za-z0-9_.:-]+)", structural_line)
        if called is None:
            while_body = re.search(
                r"\bbody=(%[A-Za-z0-9_.:-]+)", structural_line
            )
            if (
                instruction.opcode == "while"
                and len(instruction.operand_names) == 1
                and while_body is not None
            ):
                symbol = while_body.group(1)
                parameter = parameters.get(symbol, {}).get(0)
                if parameter is not None:
                    add_edge(
                        (
                            instruction.computation,
                            instruction.operand_names[0],
                        ),
                        parameter,
                        "identity",
                    )
                root = roots.get(symbol)
                if root is not None:
                    add_edge(root, target, "identity")
                continue
            branch_match = re.search(
                r"\bbranch_computations=\{([^}]*)\}", structural_line
            )
            if branch_match is None:
                continue
            branch_symbols = tuple(
                item.strip() for item in branch_match.group(1).split(",")
            )
            if (
                instruction.opcode != "conditional"
                or len(instruction.operand_names) != len(branch_symbols) + 1
            ):
                continue
            for branch_index, symbol in enumerate(branch_symbols):
                parameter = parameters.get(symbol, {}).get(0)
                if parameter is not None:
                    add_edge(
                        (
                            instruction.computation,
                            instruction.operand_names[branch_index + 1],
                        ),
                        parameter,
                        "identity",
                    )
                root = roots.get(symbol)
                if root is not None:
                    add_edge(root, target, "identity")
        else:
            symbol = called.group(1)
            for index, operand_name in enumerate(instruction.operand_names):
                parameter = parameters.get(symbol, {}).get(index)
                if parameter is not None:
                    add_edge(
                        (instruction.computation, operand_name),
                        parameter,
                        "identity",
                    )
            root = roots.get(symbol)
            if root is not None:
                add_edge(root, target, "identity")

    def first_reached_targets(
        seed: tuple[str, str], targets: set[tuple[str, str]]
    ) -> set[tuple[str, str]]:
        ValuePath = tuple[int, ...] | None
        pending: list[tuple[tuple[str, str], ValuePath]] = [(seed, ())]
        visited: set[tuple[tuple[str, str], ValuePath]] = {(seed, ())}
        reached: set[tuple[str, str]] = set()
        while pending:
            current, path = pending.pop()
            for target, kind, index in forward_edges.get(current, ()):
                target_instruction = by_key[target]
                if kind == "identity":
                    target_path = path
                elif kind == "tuple":
                    assert index is not None
                    target_path = None if path is None else (index, *path)
                elif kind == "get-tuple-element":
                    assert index is not None
                    if path is None or path == () or path[0] != index:
                        continue
                    target_path = path[1:]
                else:
                    exact_tuple_identity = (
                        target_instruction.opcode
                        in {"copy", "optimization-barrier"}
                        and len(target_instruction.operand_names) == 1
                        and len(target_instruction.result_shapes) > 1
                    )
                    target_path = (
                        path
                        if exact_tuple_identity
                        else None
                        if len(target_instruction.result_shapes) > 1
                        else ()
                    )
                if target in targets:
                    reached.add(target)
                    continue
                state = (target, target_path)
                if state not in visited:
                    visited.add(state)
                    pending.append(state)
        return reached

    custom_keys = {(item.computation, item.name) for item in custom_calls}
    convolution_keys = {
        (item.computation, item.name) for item in convolutions
    }
    gate_keys = {
        (item.computation, item.name)
        for item in convolutions
        if item.result_shapes
        and item.result_shapes[0].dimensions == (1, 768)
    }
    down_keys = convolution_keys - gate_keys
    collective_keys = {
        (item.computation, item.name) for item in collectives
    }
    dense_producer_keys = convolution_keys if final_layout else custom_keys
    producer_collective_links = {
        key: first_reached_targets(key, collective_keys)
        for key in dense_producer_keys
    }
    missing_collective_inputs = sorted(
        f"{computation}:{name}"
        for (computation, name), targets in producer_collective_links.items()
        if len(targets) != 1
    )
    if missing_collective_inputs:
        violations.append(
            "dense producers do not all feed the LP2 combine: "
            f"{missing_collective_inputs}"
        )
    gate_down_links = {
        key: first_reached_targets(key, down_keys) for key in gate_keys
    }
    exact_gate_down_bijection = bool(
        final_layout
        and len(gate_down_links) == 16
        and all(len(targets) == 1 for targets in gate_down_links.values())
        and Counter(
            target
            for targets in gate_down_links.values()
            for target in targets
        )
        == Counter({key: 1 for key in down_keys})
    )
    if final_layout and not exact_gate_down_bijection:
        violations.append(
            "PP16 final-layout gate/down convolution bijection drifted"
        )

    forbidden_shapes: list[dict[str, Any]] = []
    for instruction in module.instructions:
        for shape in instruction.result_shapes:
            if shape.dimensions in {
                (32, hidden_size),
                (32, 1, hidden_size),
                (1, 32, hidden_size),
            }:
                forbidden_shapes.append(
                    {
                        "dimensions": list(shape.dimensions),
                        "instruction": instruction.name,
                    }
                )
    if forbidden_shapes:
        violations.append("optimized HLO reconstructs a 32-row hidden tensor")

    host_markers = sorted(
        marker
        for marker in _HOST_MARKERS
        if marker in optimized_hlo.lower() or marker in stablehlo.lower()
    )
    if host_markers:
        violations.append(f"host execution markers present: {host_markers}")

    stable_collective_count = stablehlo.count("stablehlo.all_reduce")
    if stable_collective_count != 1:
        violations.append(
            "StableHLO expected exactly one all_reduce, found "
            f"{stable_collective_count}"
        )
    stable_forbidden = sorted(
        marker for marker in _FORBIDDEN_STABLE_COLLECTIVES if marker in stablehlo
    )
    if stable_forbidden:
        violations.append(
            f"StableHLO contains forbidden collectives: {stable_forbidden}"
        )
    compact_stable = "".join(stablehlo.split())
    stable_group_exact = (
        "replica_groups=dense<[[0,1]]>:tensor<1x2xi64>" in compact_stable
    )
    if not stable_group_exact:
        violations.append("StableHLO lost explicit replica group [[0,1]]")
    compact_stable_lines = tuple(
        "".join(line.split())
        for line in stablehlo.splitlines()
        if "stablehlo.convolution" in line
    )
    stable_attributes = (
        "dim_numbers=[b,f]x[i,o]->[b,f]",
        "window={stride=[],pad=[],lhs_dilate=[],rhs_dilate=[],reverse=[]}",
        "batch_group_count=1:i64,feature_group_count=1:i64",
        "precision_config=[#stablehlo<precisionDEFAULT>,"
        "#stablehlo<precisionDEFAULT>]",
    )
    stable_gate_convolution_count = sum(
        "(tensor<1x6144xbf16>,tensor<6144x768xbf16>)"
        "->tensor<1x768xf32>" in line
        and all(attribute in line for attribute in stable_attributes)
        for line in compact_stable_lines
    )
    stable_down_convolution_count = sum(
        "(tensor<1x384xbf16>,tensor<384x6144xbf16>)"
        "->tensor<1x6144xf32>" in line
        and all(attribute in line for attribute in stable_attributes)
        for line in compact_stable_lines
    )
    expected_stable_convolutions = 16 if final_layout else 0
    if (
        stable_gate_convolution_count != expected_stable_convolutions
        or stable_down_convolution_count != expected_stable_convolutions
        or len(compact_stable_lines) != 2 * expected_stable_convolutions
    ):
        violations.append(
            "StableHLO final-layout convolution set drifted: "
            f"gate={stable_gate_convolution_count} "
            f"down={stable_down_convolution_count} "
            f"total={len(compact_stable_lines)} "
            f"expected_each={expected_stable_convolutions}"
        )

    entry_roots = tuple(
        instruction
        for instruction in module.instructions
        if instruction.computation.startswith("ENTRY ")
        and instruction.raw_line.lstrip().startswith("ROOT ")
    )
    root_shapes = Counter(
        (shape.dtype.lower(), shape.dimensions)
        for root in entry_roots
        for shape in root.result_shapes
    )
    expected_root_shapes = Counter({("bf16", (1, hidden_size)): 3})
    if len(entry_roots) != 1 or root_shapes != expected_root_shapes:
        violations.append(
            "optimized HLO lost the three one-row boundary outputs: "
            f"{dict(root_shapes)}"
        )
    entry_root_keys = {
        (instruction.computation, instruction.name)
        for instruction in entry_roots
    }
    collective_root_links = {
        key: first_reached_targets(key, entry_root_keys)
        for key in collective_keys
    }
    collective_reaches_root = bool(
        len(collectives) == 1
        and all(len(targets) == 1 for targets in collective_root_links.values())
    )
    if not collective_reaches_root:
        violations.append("LP2 combine does not feed the returned boundary")

    return {
        "collectives": [item.to_dict() for item in collectives],
        "association": association,
        "collective_input_convolution_count": sum(
            len(targets) == 1
            for key, targets in producer_collective_links.items()
            if key in convolution_keys
        ),
        "collective_input_custom_call_count": sum(
            len(targets) == 1
            for key, targets in producer_collective_links.items()
            if key in custom_keys
        ),
        "collective_reaches_root": collective_reaches_root,
        "custom_call_count": len(custom_calls),
        "dense_convolution_count": len(convolutions),
        "dense_down_convolution_ranks": dict(sorted(down_ranks.items())),
        "dense_gate_down_bijection": exact_gate_down_bijection,
        "dense_gate_convolution_ranks": dict(sorted(gate_ranks.items())),
        "expected_kernel_name": kernel_name,
        "forbidden_shapes": forbidden_shapes,
        "host_markers": host_markers,
        "kernel_count": kernel_count,
        "num_partitions": module.num_partitions,
        "optimized_hlo_sha256": sha256(optimized_hlo.encode()).hexdigest(),
        "passed": not violations,
        "stable_group_exact": stable_group_exact,
        "stablehlo_all_reduce_count": stable_collective_count,
        "stablehlo_dense_down_convolution_count": (
            stable_down_convolution_count
        ),
        "stablehlo_dense_gate_convolution_count": (
            stable_gate_convolution_count
        ),
        "stablehlo_forbidden_collectives": stable_forbidden,
        "stablehlo_sha256": sha256(stablehlo.encode()).hexdigest(),
        "violations": violations,
    }
