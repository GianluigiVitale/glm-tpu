"""Model-free canary for one TPU v4 32-way BF16 reduction instance.

The pinned legacy artifact is a 2,048-row prefill executable whose projection
uses a three-color ``StrategyND`` all-reduce.  This module feeds a one-row
collective carrying the same backend label deterministic cancellation-heavy BF16
inputs, preserves the raw input/output bits, and replays a bounded family of
topology-shaped pincer trees offline.  Backend-label equality does not prove
payload chunking, member mapping, or addition association equality with either
the prefill collective or the separate 32-row decode executable.  It is a
non-gating diagnostic only: the full-pod collective is never an admissible
repeated-layer operation in the greenfield engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

import numpy as np

from ..errors import BenchmarkValidationError
from ..sharding.hlo_contract import (
    CollectiveExpectation,
    HloContractPolicy,
    HloInstruction,
    HloLintReport,
    lint_hlo,
    parse_hlo_module,
)


ACCEPTED_PROMPT_PROJECTION_HLO_SHA256 = (
    "51d014de3776bae76b9c4068bf9ed3b06b690b2ce65531f68d372738b63d47f0"
)
STRATEGY_ND_DEBUG = (
    "\nStrategyND{colors:3 phases:3 cores:{4 2 4},{2 4 4},{4 4 2} "
    "nophase0:0 reserved_sflags:1 cross_module_on_2d_plane:0 "
    "has_reordering_map:0 use_routing_table_indices:0 num_dims:3 "
    "active_dims:111 dim_sizes:4,2,4 dim_used:{0 1 2},{1 2 0},{2 0 1} "
    "use_ndnway:0 asymmetric_max_colors:1 devices:1,1,1 stride:1, 1, 1}"
)
STRATEGY_ND_ALGORITHM = {
    "debug": STRATEGY_ND_DEBUG,
    "emitter": "RotatedPincerEmitter",
    "strategy": "StrategyND",
}
STRATEGY_ND_PHASE_ORDERS = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
_BALANCED_FOUR_WAY_TREES = (
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
)


@dataclass(frozen=True, slots=True)
class StrategyNdFingerprintConfig:
    trials: int = 32
    width: int = 6144
    seed: int = 0x47524C4D

    def __post_init__(self) -> None:
        for field in ("trials", "width", "seed"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool):
                raise BenchmarkValidationError(f"{field} must be an integer")
        if self.trials <= 0 or self.width <= 0 or self.seed < 0:
            raise BenchmarkValidationError(
                "trials/width must be positive and seed must be non-negative"
            )

    def require_protected_contract(self) -> None:
        if self.trials != 32 or self.width != 6144:
            raise BenchmarkValidationError(
                "protected StrategyND fingerprint requires 32 trials at width 6144"
            )

    def to_dict(self) -> dict[str, int]:
        return {"seed": self.seed, "trials": self.trials, "width": self.width}


@dataclass(frozen=True, slots=True)
class CompiledStrategyNdFingerprint:
    config: StrategyNdFingerprintConfig
    compiled: Any
    input_sharding: Any
    member_device_ids: tuple[int, ...]
    optimized_hlo: str
    hlo_report: HloLintReport
    collective_algorithm: Mapping[str, Any]


def array_sha256(value: np.ndarray) -> str:
    """Hash dtype, shape, and C-order bytes so raw captures are unambiguous."""

    array = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(f"dtype={array.dtype.str};shape={array.shape};".encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def bfloat16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16
    return values.view(np.float32)


def float32_to_bfloat16_bits(values: np.ndarray) -> np.ndarray:
    """Round finite float32 values to BF16 using round-to-nearest-even."""

    floats = np.asarray(values, dtype=np.float32)
    if not np.isfinite(floats).all():
        raise BenchmarkValidationError("BF16 fingerprint values must be finite")
    raw = floats.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((raw >> 16) & np.uint32(1))
    return ((raw + rounding_bias) >> 16).astype(np.uint16)


def generate_strategy_nd_input_bits(
    config: StrategyNdFingerprintConfig,
) -> np.ndarray:
    """Return deterministic cancellation-heavy bits as ``[trial, member, width]``."""

    rng = np.random.Generator(np.random.PCG64(config.seed))
    shape = (config.trials, config.width, 16)
    raw = rng.bit_generator.random_raw(shape)
    exponent = np.uint16(121) + (raw % np.uint64(12)).astype(np.uint16)
    mantissa = ((raw >> np.uint64(8)) % np.uint64(128)).astype(np.uint16)
    magnitude = (exponent << np.uint16(7)) | mantissa

    leaves = np.empty((config.trials, config.width, 32), dtype=np.uint16)
    leaves[:, :, 0::2] = magnitude
    leaves[:, :, 1::2] = magnitude | np.uint16(0x8000)

    # Perturb alternating cancellation pairs by one or two BF16 ULPs.  Exact
    # pairs keep the total bounded; perturbed pairs make association visible.
    perturb_raw = rng.bit_generator.random_raw(shape)
    perturb_mask = (np.arange(16) % 2 == 1)[None, None, :]
    perturb = np.where(
        perturb_mask,
        (perturb_raw % np.uint64(2)).astype(np.uint16) + np.uint16(1),
        np.uint16(0),
    )
    negative_magnitude = leaves[:, :, 1::2] & np.uint16(0x7FFF)
    negative_magnitude = np.minimum(
        negative_magnitude.astype(np.uint32) + perturb.astype(np.uint32),
        np.uint32(0x7F7F),
    ).astype(np.uint16)
    leaves[:, :, 1::2] = negative_magnitude | np.uint16(0x8000)

    # A different deterministic rank permutation per trial/column prevents
    # the structured pairs from favoring one hypothesized topology tree.
    permutation_keys = rng.bit_generator.random_raw(leaves.shape)
    permutation = np.argsort(permutation_keys, axis=2, kind="stable")
    permuted = np.take_along_axis(leaves, permutation, axis=2)
    result = np.ascontiguousarray(permuted.transpose(0, 2, 1))
    decoded = bfloat16_bits_to_float32(result)
    if not np.isfinite(decoded).all():
        raise BenchmarkValidationError("generated BF16 fingerprint contains non-finite values")
    return result


def strategy_nd_fingerprint_hlo_policy(
    member_device_ids: Sequence[int],
) -> HloContractPolicy:
    members = tuple(int(device_id) for device_id in member_device_ids)
    if len(members) != 32 or sorted(members) != list(range(32)):
        raise BenchmarkValidationError(
            "StrategyND fingerprint member ids must permute physical devices 0..31"
        )
    return HloContractPolicy(
        name="strategy-nd-association-fingerprint",
        total_devices=32,
        repeated_region_patterns=(r"strategy_nd_association_fingerprint",),
        maximum_repeated_collective_group_size=32,
        expected_repeated_replica_groups=(members,),
        expected_collectives=(CollectiveExpectation("all-reduce", 1),),
        partition_id_to_device_id=members,
        forbidden_row_width_pairs=(),
        allow_full_pod_repeated_collectives=True,
    )


def _backend_config(instruction: HloInstruction) -> Mapping[str, Any]:
    marker = "backend_config="
    start = instruction.raw_line.find(marker)
    if start < 0:
        raise BenchmarkValidationError("fingerprint all-reduce has no backend_config")
    value = instruction.raw_line[start + len(marker) :].lstrip()
    try:
        parsed, end = json.JSONDecoder().raw_decode(value)
    except json.JSONDecodeError as error:
        raise BenchmarkValidationError(
            "fingerprint all-reduce backend_config is not valid JSON"
        ) from error
    if value[end:].strip().strip(","):
        raise BenchmarkValidationError(
            "fingerprint all-reduce backend_config has unparsed trailing data"
        )
    if not isinstance(parsed, dict):
        raise BenchmarkValidationError("fingerprint backend_config must be an object")
    return parsed


def validate_strategy_nd_fingerprint_hlo(
    optimized_hlo: str,
    member_device_ids: Sequence[int],
) -> tuple[HloLintReport, Mapping[str, Any]]:
    report = lint_hlo(
        parse_hlo_module(optimized_hlo),
        strategy_nd_fingerprint_hlo_policy(member_device_ids),
    )
    report.raise_for_violations()
    reductions = tuple(
        item for item in report.module.collectives if item.opcode == "all-reduce"
    )
    if len(reductions) != 1:
        raise BenchmarkValidationError(
            f"fingerprint requires one physical all-reduce, found {len(reductions)}"
        )
    reduction = reductions[0]
    shapes = tuple((shape.dtype, shape.dimensions) for shape in reduction.result_shapes)
    if shapes != (("bf16", (1, 6144)),):
        raise BenchmarkValidationError(
            f"fingerprint all-reduce result must be bf16[1,6144], got {shapes}"
        )
    backend = _backend_config(reduction)
    algorithm = backend.get("collective_algorithm_config")
    if algorithm != STRATEGY_ND_ALGORITHM:
        raise BenchmarkValidationError(
            "fingerprint collective algorithm differs from the byte-pinned accepted StrategyND config"
        )
    return report, dict(algorithm)


def _fingerprint_function() -> Any:
    import jax
    from jax import lax
    import jax.numpy as jnp

    def fingerprint(local_bits: Any) -> Any:
        with jax.named_scope("strategy_nd_association_fingerprint"):
            payload = lax.bitcast_convert_type(local_bits[0], jnp.bfloat16)
            reduced = lax.psum(payload, "member")
            reduced = lax.optimization_barrier(reduced)
            return lax.bitcast_convert_type(reduced, jnp.uint16)

    return fingerprint


def build_strategy_nd_fingerprint(
    config: StrategyNdFingerprintConfig,
    member_device_ids: Sequence[int],
    *,
    devices: Sequence[Any] | None = None,
    enforce_hlo_contract: bool = True,
) -> CompiledStrategyNdFingerprint:
    """Compile the one-collective fingerprint executable."""

    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    runtime_devices = tuple(jax.devices() if devices is None else devices)
    members = tuple(int(device_id) for device_id in member_device_ids)
    by_id = {int(device.id): device for device in runtime_devices}
    if len(runtime_devices) != 32 or set(by_id) != set(range(32)):
        raise BenchmarkValidationError(
            "StrategyND fingerprint requires contiguous global TPU ids 0..31"
        )
    strategy_nd_fingerprint_hlo_policy(members)
    mesh = Mesh(np.asarray([by_id[item] for item in members], dtype=object), ("member",))
    input_sharding = NamedSharding(mesh, P("member", None, None))
    mapped = jax.shard_map(
        _fingerprint_function(),
        mesh=mesh,
        in_specs=P("member", None, None),
        out_specs=P(None, None),
        check_vma=False,
    )
    executable = jax.jit(mapped)
    example = jax.device_put(
        np.zeros((32, 1, config.width), dtype=np.uint16), input_sharding
    )
    compiled = executable.lower(example).compile()
    optimized_hlo = compiled.as_text()
    if enforce_hlo_contract:
        report, algorithm = validate_strategy_nd_fingerprint_hlo(
            optimized_hlo, members
        )
    else:
        report = lint_hlo(
            parse_hlo_module(optimized_hlo),
            strategy_nd_fingerprint_hlo_policy(members),
        )
        algorithm = {}
    return CompiledStrategyNdFingerprint(
        config=config,
        compiled=compiled,
        input_sharding=input_sharding,
        member_device_ids=members,
        optimized_hlo=optimized_hlo,
        hlo_report=report,
        collective_algorithm=algorithm,
    )


def execute_strategy_nd_fingerprint(
    compiled: CompiledStrategyNdFingerprint,
    input_bits: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Execute all trials and return representative replicated output bits."""

    import jax

    bits = np.asarray(input_bits, dtype=np.uint16)
    expected_shape = (compiled.config.trials, 32, compiled.config.width)
    if bits.shape != expected_shape:
        raise BenchmarkValidationError(
            f"fingerprint input bits must have shape {expected_shape}, got {bits.shape}"
        )
    outputs = []
    local_replica_hashes = []
    for trial in range(compiled.config.trials):
        value = jax.device_put(bits[trial, :, None, :], compiled.input_sharding)
        result = compiled.compiled(value)
        jax.block_until_ready(result)
        local = [
            np.asarray(jax.device_get(shard.data), dtype=np.uint16).reshape(
                compiled.config.width
            )
            for shard in sorted(
                result.addressable_shards, key=lambda shard: int(shard.device.id)
            )
        ]
        hashes = tuple(array_sha256(item) for item in local)
        expected_local_replicas = len(compiled.input_sharding.addressable_devices)
        if len(local) != expected_local_replicas or len(set(hashes)) != 1:
            raise BenchmarkValidationError(
                "fingerprint output is not byte-identical on every local replica"
            )
        outputs.append(local[0])
        local_replica_hashes.append(hashes)

    # A repeated first trial catches nondeterministic network arithmetic.
    repeated_value = jax.device_put(bits[0, :, None, :], compiled.input_sharding)
    repeated = compiled.compiled(repeated_value)
    jax.block_until_ready(repeated)
    repeated_host = np.asarray(
        jax.device_get(
            min(repeated.addressable_shards, key=lambda shard: int(shard.device.id)).data
        ),
        dtype=np.uint16,
    ).reshape(compiled.config.width)
    output_bits = np.ascontiguousarray(np.stack(outputs))
    if not np.array_equal(output_bits[0], repeated_host):
        raise BenchmarkValidationError(
            "repeated StrategyND fingerprint trial produced different BF16 bits"
        )
    return output_bits, {
        "input_bits_sha256": array_sha256(bits),
        "local_replica_output_sha256_by_trial": [
            list(hashes) for hashes in local_replica_hashes
        ],
        "output_bits_sha256": array_sha256(output_bits),
        "repeated_first_trial_sha256": array_sha256(repeated_host),
    }


def _bf16_add(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return bfloat16_bits_to_float32(
        float32_to_bfloat16_bits(
            np.asarray(left, dtype=np.float32) + np.asarray(right, dtype=np.float32)
        )
    )


def _evaluate_four_way_tree(values: np.ndarray, axis: int, tree: Any) -> np.ndarray:
    if isinstance(tree, int):
        return np.take(values, tree, axis=axis)
    return _bf16_add(
        _evaluate_four_way_tree(values, axis, tree[0]),
        _evaluate_four_way_tree(values, axis, tree[1]),
    )


def _candidate_output_bits(
    values: np.ndarray,
    phase_order: tuple[int, int, int],
    trees: Mapping[int, Any],
) -> np.ndarray:
    remaining = [0, 1, 2]
    state = values
    for dimension in phase_order:
        axis = 1 + remaining.index(dimension)
        if dimension == 1:
            state = _bf16_add(
                np.take(state, 0, axis=axis), np.take(state, 1, axis=axis)
            )
        else:
            state = _evaluate_four_way_tree(state, axis, trees[dimension])
        remaining.remove(dimension)
    return float32_to_bfloat16_bits(state)


def _tree_name(tree: Any) -> str:
    if isinstance(tree, int):
        return str(tree)
    return f"({_tree_name(tree[0])}+{_tree_name(tree[1])})"


def analyze_strategy_nd_fingerprint(
    input_bits: np.ndarray,
    output_bits: np.ndarray,
    member_device_ids: Sequence[int],
    device_coordinates: Mapping[int, Sequence[int]],
    *,
    block_width: int = 128,
) -> dict[str, Any]:
    """Replay the bounded balanced-pincer StrategyND candidate family offline."""

    inputs = np.asarray(input_bits, dtype=np.uint16)
    outputs = np.asarray(output_bits, dtype=np.uint16)
    if inputs.ndim != 3 or inputs.shape[1] != 32:
        raise BenchmarkValidationError("input_bits must have shape [trial,32,width]")
    if outputs.shape != (inputs.shape[0], inputs.shape[2]):
        raise BenchmarkValidationError(
            "output_bits must have shape [trial,width] aligned with input_bits"
        )
    members = tuple(int(item) for item in member_device_ids)
    if len(members) != 32 or sorted(members) != list(range(32)):
        raise BenchmarkValidationError("member_device_ids must permute 0..31")
    coordinates = {
        int(device_id): tuple(int(item) for item in coordinate)
        for device_id, coordinate in device_coordinates.items()
    }
    if set(coordinates) != set(range(32)) or any(
        len(coordinate) != 3 for coordinate in coordinates.values()
    ):
        raise BenchmarkValidationError(
            "device_coordinates must map physical devices 0..31 to three coordinates"
        )
    if block_width <= 0 or inputs.shape[2] % block_width:
        raise BenchmarkValidationError("block_width must divide the fingerprint width")
    if inputs.shape[2] % 3:
        raise BenchmarkValidationError(
            "fingerprint width must be divisible by three for band analysis"
        )

    decoded = bfloat16_bits_to_float32(inputs)
    candidate_rows = []
    exact_masks = []
    mapping_specs = (
        ("physical_y_x_z", (1, 0, 2)),
        ("physical_z_x_y", (2, 0, 1)),
    )
    for mapping_name, coordinate_order in mapping_specs:
        mapped_targets = [
            tuple(coordinates[device_id][index] for index in coordinate_order)
            for device_id in members
        ]
        if len(set(mapped_targets)) != 32 or any(
            not (
                0 <= target[0] < 4
                and 0 <= target[1] < 2
                and 0 <= target[2] < 4
            )
            for target in mapped_targets
        ):
            raise BenchmarkValidationError(
                f"{mapping_name} must bijectively map devices onto a 4x2x4 box"
            )
        topology_values = np.empty(
            (inputs.shape[0], 4, 2, 4, inputs.shape[2]), dtype=np.float32
        )
        for member_index, device_id in enumerate(members):
            coordinate = coordinates[device_id]
            target = tuple(coordinate[index] for index in coordinate_order)
            topology_values[(slice(None), *target, slice(None))] = decoded[
                :, member_index, :
            ]
        for phase_order in STRATEGY_ND_PHASE_ORDERS:
            for dimension_zero_tree in _BALANCED_FOUR_WAY_TREES:
                for dimension_two_tree in _BALANCED_FOUR_WAY_TREES:
                    candidate = _candidate_output_bits(
                        topology_values,
                        phase_order,
                        {0: dimension_zero_tree, 2: dimension_two_tree},
                    )
                    trial_matches = candidate == outputs
                    exact_mask = np.all(trial_matches, axis=0)
                    candidate_id = (
                        f"{mapping_name}:phase={''.join(map(str, phase_order))}:"
                        f"d0={_tree_name(dimension_zero_tree)}:"
                        f"d2={_tree_name(dimension_two_tree)}"
                    )
                    candidate_rows.append(
                        {
                            "candidate_id": candidate_id,
                            "exact_columns": int(exact_mask.sum()),
                            "matching_trial_elements": int(trial_matches.sum()),
                            "third_exact_columns": [
                                int(part.sum())
                                for part in np.split(exact_mask, 3)
                            ],
                        }
                    )
                    exact_masks.append(exact_mask)

    mask_matrix = np.stack(exact_masks)
    per_column_count = mask_matrix.sum(axis=0)
    union = per_column_count > 0
    candidate_rows.sort(
        key=lambda item: (
            -item["exact_columns"],
            -item["matching_trial_elements"],
            item["candidate_id"],
        )
    )
    # Build block summaries from the original mask order; candidate ids are
    # recovered directly from a separate immutable list to avoid rank drift.
    original_ids = []
    for mapping_name, _ in mapping_specs:
        for phase_order in STRATEGY_ND_PHASE_ORDERS:
            for dimension_zero_tree in _BALANCED_FOUR_WAY_TREES:
                for dimension_two_tree in _BALANCED_FOUR_WAY_TREES:
                    original_ids.append(
                        f"{mapping_name}:phase={''.join(map(str, phase_order))}:"
                        f"d0={_tree_name(dimension_zero_tree)}:"
                        f"d2={_tree_name(dimension_two_tree)}"
                    )
    blocks = []
    for start in range(0, inputs.shape[2], block_width):
        end = start + block_width
        counts = mask_matrix[:, start:end].sum(axis=1)
        leaders = sorted(
            range(len(original_ids)),
            key=lambda index: (-int(counts[index]), original_ids[index]),
        )[:3]
        blocks.append(
            {
                "end": end,
                "start": start,
                "top_candidates": [
                    {
                        "candidate_id": original_ids[index],
                        "exact_columns": int(counts[index]),
                    }
                    for index in leaders
                    if counts[index]
                ],
                "union_exact_columns": int(union[start:end].sum()),
            }
        )

    ambiguity_values, ambiguity_counts = np.unique(
        per_column_count, return_counts=True
    )
    return {
        "analysis_scope": (
            "bounded balanced four-way axis-pincer trees; raw artifacts permit broader replay"
        ),
        "block_width": block_width,
        "blocks": blocks,
        "candidate_count": len(original_ids),
        "candidate_family_exhaustive": False,
        "column_candidate_count_histogram": {
            str(int(value)): int(count)
            for value, count in zip(ambiguity_values, ambiguity_counts, strict=True)
        },
        "input_bits_sha256": array_sha256(inputs),
        "output_bits_sha256": array_sha256(outputs),
        "top_candidates": candidate_rows[:20],
        "trials": inputs.shape[0],
        "uncovered_column_count": int((~union).sum()),
        "uncovered_columns": np.flatnonzero(~union).astype(int).tolist(),
        "union_exact_column_count": int(union.sum()),
        "union_exact_fraction": float(union.mean()),
        "width": inputs.shape[2],
    }
