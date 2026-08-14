"""Model-free canary for the accepted TPU-v4 M32 projection reduction.

DB532 seals the accepted decode executable's exact ``bf16[32,6144]`` physical
all-reduce: sorted global ranks 0--31, the TPU tiled result layout, and the full
three-color ``StrategyND`` algorithm config.  This module repeatedly executes
that shape-identical collective.  Each cancellation-heavy trial is replicated
across all 32 physical rows, so every fixed buffer position receives repeated
observations even when StrategyND assigns different colors to different rows.
It preserves the raw input/output bits and replays a bounded family of
topology-shaped pincer trees separately for every row.

The result is a non-gating numerical-association diagnostic.  It does not make
the accepted full-pod reduction admissible in the greenfield engine and it does
not claim model correctness or performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
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


ACCEPTED_DECODE_PROJECTION_HLO_RAW_SHA256 = (
    "3cd750810982608f9a3a7d557497c58f61159cc3dcdeb521f1377ba8c93fb775"
)
ACCEPTED_DECODE_PROJECTION_HLO_GZIP_SHA256 = (
    "25041bfbcf319b6c6fc4c5888cb22548b246cccba784791796fe9e8f57199e4c"
)
ACCEPTED_DECODE_PROJECTION_MANIFEST_SHA256 = (
    "9257e28b0ee8d6851d03caaf862717e34db6c8014af5e579a0c44d7d1174e487"
)
ACCEPTED_DECODE_RESULT_LAYOUT = "{1,0:T(8,128)(2,1)S(3)}"
ACCEPTED_DECODE_BUCKET_ROWS = 32
ACCEPTED_TP32_MODEL_AXIS_RECIPE = (
    "mesh_utils.create_device_mesh:shape=1,1,1,1,32,1:"
    "allow_split_physical_axes=true:model_axis=4"
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

# DB550 seals the accepted oracle's layer-0 pre-psum dense partials in model-
# axis order.  These pins let a model-free run replay the real values through
# the exact M32 collective without loading any checkpoint weights.
ACCEPTED_DENSE_PARTIALS_NPZ_SHA256 = (
    "e5977248acbe7582db351178b3fc823c6f87db46f8b6143b763319a6b299582c"
)
ACCEPTED_DENSE_PARTIALS_RAW_SHA256 = (
    "9d9f65dddc7b622875872a33a6522c330c8fb5490c8cba14526553c211516e35"
)
ACCEPTED_DENSE_PARTIALS_SHAPE = (4, 8, 1, 6144)
ACCEPTED_DENSE_PARTIALS_KEYS = (
    "accepted_dense_partials_bfloat16_bits",
    "db548_dense_partials_bfloat16_bits",
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


def accepted_tp32_model_axis_device_ids(
    devices: Sequence[Any] | None = None,
) -> tuple[int, ...]:
    """Replay the accepted oracle's model-axis device-order recipe.

    The pinned oracle's logged six-axis mesh is created with
    ``mesh_utils.create_device_mesh((1, 1, 1, 1, 32, 1), ...)`` and model is
    its only nontrivial axis.  Physical association fingerprints are keyed by
    global device id, while checkpoint projection shards are keyed by model-
    axis position, so the protected artifact must retain this permutation
    rather than assume they are equal.
    """

    import jax
    from jax.experimental import mesh_utils

    runtime_devices = tuple(jax.devices() if devices is None else devices)
    by_id = {int(device.id): device for device in runtime_devices}
    if len(runtime_devices) != 32 or set(by_id) != set(range(32)):
        raise BenchmarkValidationError(
            "accepted TP32 model-axis replay requires global device ids 0..31"
        )
    ordered = tuple(by_id[index] for index in range(32))
    mesh_devices = mesh_utils.create_device_mesh(
        (1, 1, 1, 1, 32, 1),
        ordered,
        allow_split_physical_axes=True,
    )
    result = tuple(
        int(device.id) for device in np.asarray(mesh_devices).reshape(-1)
    )
    if len(result) != 32 or sorted(result) != list(range(32)):
        raise BenchmarkValidationError(
            "accepted TP32 model-axis replay did not produce a device permutation"
        )
    return result


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


def model_axis_to_physical_input_bits(
    model_bits: np.ndarray,
    model_axis_device_ids: Sequence[int],
) -> np.ndarray:
    """Reorder model-axis BF16 bits into sorted physical-device order."""

    bits = np.asarray(model_bits)
    if bits.dtype != np.uint16 or bits.ndim != 2 or bits.shape[0] != 32:
        raise BenchmarkValidationError(
            "model-axis BF16 bits must have shape [32,width] and dtype uint16"
        )
    mapping = tuple(int(device_id) for device_id in model_axis_device_ids)
    if len(mapping) != 32 or sorted(mapping) != list(range(32)):
        raise BenchmarkValidationError(
            "model_axis_device_ids must bijectively map model positions to 0..31"
        )
    physical = np.empty_like(bits)
    physical[np.asarray(mapping, dtype=np.int32)] = bits
    return np.ascontiguousarray(physical)


def replay_db533_strategy_nd_row0_bits(
    model_bits: np.ndarray,
    model_axis_device_ids: Sequence[int],
) -> np.ndarray:
    """Replay DB533's measured physical-row-zero BF16 association in NumPy."""

    physical = model_axis_to_physical_input_bits(
        model_bits, model_axis_device_ids
    )
    if physical.shape[1] != 6144:
        raise BenchmarkValidationError(
            "DB533 row-zero replay requires width 6144"
        )
    # Physical device ids are x-fastest in [z,y,x].  DB533's row-zero tree is
    # y -> x -> z with column-dependent pincer pairing on y and z.
    values = bfloat16_bits_to_float32(physical).reshape(4, 4, 2, 6144)
    values = values.transpose(1, 2, 0, 3)  # [y,x,z,hidden]

    def reduce_four(source: np.ndarray, *, cross: bool) -> np.ndarray:
        if cross:
            return _bf16_add(
                _bf16_add(source[0], source[3]),
                _bf16_add(source[1], source[2]),
            )
        return _bf16_add(
            _bf16_add(source[0], source[1]),
            _bf16_add(source[2], source[3]),
        )

    y_reduced = np.concatenate(
        (
            reduce_four(values[..., :2048], cross=False),
            reduce_four(values[..., 2048:4096], cross=True),
            reduce_four(values[..., 4096:], cross=False),
        ),
        axis=-1,
    )
    x_reduced = _bf16_add(y_reduced[0], y_reduced[1])
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
    return np.ascontiguousarray(float32_to_bfloat16_bits(reduced))


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
    if members != tuple(range(32)):
        raise BenchmarkValidationError(
            "accepted M32 fingerprint requires sorted global member ids 0..31"
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
    positions = _unquoted_attribute_positions(instruction.raw_line, marker)
    if len(positions) != 1:
        raise BenchmarkValidationError("fingerprint all-reduce has no backend_config")
    start = positions[0]
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


def _unquoted_attribute_positions(value: str, marker: str) -> tuple[int, ...]:
    """Locate HLO attributes outside quoted strings and C comments."""

    positions: list[int] = []
    index = 0
    quoted = False
    escaped = False
    comment = False
    while index < len(value):
        if comment:
            if value.startswith("*/", index):
                comment = False
                index += 2
            else:
                index += 1
            continue
        if quoted:
            character = value[index]
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            index += 1
            continue
        if value.startswith("/*", index):
            comment = True
            index += 2
            continue
        if value[index] == '"':
            quoted = True
            index += 1
            continue
        if value.startswith(marker, index):
            positions.append(index)
            index += len(marker)
            continue
        index += 1
    return tuple(positions)


def _unquoted_text(value: str) -> str:
    result = list(value)
    index = 0
    quoted = False
    escaped = False
    comment = False
    while index < len(value):
        if comment:
            result[index] = " "
            if value.startswith("*/", index):
                result[index + 1] = " "
                comment = False
                index += 2
            else:
                index += 1
            continue
        if quoted:
            result[index] = " "
            character = value[index]
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            index += 1
            continue
        if value.startswith("/*", index):
            result[index] = result[index + 1] = " "
            comment = True
            index += 2
            continue
        if value[index] == '"':
            result[index] = " "
            quoted = True
        index += 1
    return "".join(result)


def _instruction_has_exact_prefix(
    instruction: HloInstruction,
    signature: str,
) -> bool:
    line = instruction.raw_line.lstrip()
    if line.startswith("ROOT "):
        line = line[len("ROOT ") :]
    return line.startswith(
        f"{instruction.name} = {signature} {instruction.raw_opcode}("
    )


def _computation_base(value: str) -> str:
    return value.split(" ", 1)[0]


def _validate_fingerprint_value_flow(
    report: HloLintReport,
    reduction: HloInstruction,
) -> None:
    """Bind the exact ENTRY input through the collective to its live result."""

    entry = tuple(
        item
        for item in report.module.instructions
        if item.computation == reduction.computation
    )
    roots = tuple(
        item for item in entry if item.raw_line.lstrip().startswith("ROOT ")
    )
    parameters = tuple(item for item in entry if item.raw_opcode == "parameter")
    exact_bf16 = f"bf16[32,6144]{ACCEPTED_DECODE_RESULT_LAYOUT}"
    if len(roots) != 1 or len(parameters) != 1:
        raise BenchmarkValidationError(
            "fingerprint ENTRY input/result lineage is not unique"
        )
    parameter = parameters[0]
    root = roots[0]

    # The small synthetic fixture uses a direct BF16 ENTRY parameter.  TPU
    # codegen uses the second exact form below.
    if len(entry) == 2:
        if (
            parameter.operand_names != ("0",)
            or not _instruction_has_exact_prefix(parameter, exact_bf16)
            or reduction.operand_names != (parameter.name,)
            or root is not reduction
        ):
            raise BenchmarkValidationError(
                "fingerprint direct ENTRY operand layout/value flow drifted"
            )
        return

    # Exact TPU lowering: U16 input -> one kLoop bitcast-convert/bitcast fusion
    # -> synchronous all-reduce -> live U16 bitcast-convert ROOT.
    if len(entry) != 4:
        raise BenchmarkValidationError(
            "fingerprint TPU ENTRY contains unapproved arithmetic or dead values"
        )
    fusion_candidates = tuple(item for item in entry if item.raw_opcode == "fusion")
    if len(fusion_candidates) != 1:
        raise BenchmarkValidationError("fingerprint input fusion is not unique")
    fusion = fusion_candidates[0]
    if (
        parameter.operand_names != ("0",)
        or not _instruction_has_exact_prefix(
            parameter, "u16[1,32,6144]{2,1,0:T(8,128)(2,1)}"
        )
        or fusion.operand_names != (parameter.name,)
        or not _instruction_has_exact_prefix(fusion, exact_bf16)
        or reduction.operand_names != (fusion.name,)
        or root.raw_opcode != "bitcast-convert"
        or root.operand_names != (reduction.name,)
        or not _instruction_has_exact_prefix(
            root, "u16[32,6144]{1,0:T(8,128)(2,1)}"
        )
    ):
        raise BenchmarkValidationError("fingerprint TPU ENTRY value flow drifted")
    calls = re.findall(
        r"(?:^|,\s*)calls=([A-Za-z0-9_.%:-]+)(?:,|$)",
        _unquoted_text(fusion.raw_line),
    )
    if len(calls) != 1:
        raise BenchmarkValidationError("fingerprint input fusion callee drifted")
    callee = tuple(
        item
        for item in report.module.instructions
        if _computation_base(item.computation) == calls[0]
    )
    callee_roots = tuple(
        item for item in callee if item.raw_line.lstrip().startswith("ROOT ")
    )
    callee_parameters = tuple(
        item for item in callee if item.raw_opcode == "parameter"
    )
    if len(callee) != 3 or len(callee_roots) != 1 or len(callee_parameters) != 1:
        raise BenchmarkValidationError("fingerprint input fusion body drifted")
    callee_parameter = callee_parameters[0]
    conversion = tuple(item for item in callee if item.raw_opcode == "bitcast-convert")
    callee_root = callee_roots[0]
    if (
        len(conversion) != 1
        or callee_parameter.operand_names != ("0",)
        or not _instruction_has_exact_prefix(
            callee_parameter, "u16[1,32,6144]{2,1,0:T(8,128)(2,1)}"
        )
        or conversion[0].operand_names != (callee_parameter.name,)
        or not _instruction_has_exact_prefix(
            conversion[0], "bf16[1,32,6144]{2,1,0:T(8,128)(2,1)}"
        )
        or callee_root.raw_opcode != "bitcast"
        or callee_root.operand_names != (conversion[0].name,)
        or not _instruction_has_exact_prefix(callee_root, exact_bf16)
    ):
        raise BenchmarkValidationError(
            "fingerprint input fusion is not the exact U16-to-BF16 layout path"
        )


def validate_strategy_nd_reduction(
    report: HloLintReport,
    reduction: HloInstruction,
) -> Mapping[str, Any]:
    """Validate the shared physical StrategyND operation, excluding consumers."""

    if reduction.raw_opcode != "all-reduce":
        raise BenchmarkValidationError(
            "fingerprint requires one synchronous all-reduce"
        )
    shapes = tuple((shape.dtype, shape.dimensions) for shape in reduction.result_shapes)
    if shapes != (("bf16", (32, 6144)),):
        raise BenchmarkValidationError(
            f"fingerprint all-reduce result must be bf16[32,6144], got {shapes}"
        )
    operand_shapes = tuple(
        (shape.dtype, shape.dimensions) for shape in reduction.operand_shapes
    )
    if operand_shapes != (("bf16", (32, 6144)),):
        raise BenchmarkValidationError(
            "fingerprint all-reduce operand must be bf16[32,6144], "
            f"got {operand_shapes}"
        )
    exact_result = f"bf16[32,6144]{ACCEPTED_DECODE_RESULT_LAYOUT} all-reduce("
    if exact_result not in reduction.raw_line:
        raise BenchmarkValidationError(
            "fingerprint all-reduce does not preserve the accepted M32 TPU result layout"
        )
    if len(reduction.operand_names) != 1:
        raise BenchmarkValidationError(
            "fingerprint all-reduce requires one exact M32 operand"
        )
    if not _instruction_has_exact_prefix(
        reduction, f"bf16[32,6144]{ACCEPTED_DECODE_RESULT_LAYOUT}"
    ):
        raise BenchmarkValidationError(
            "fingerprint all-reduce result does not preserve the accepted M32 TPU layout"
        )
    backend = _backend_config(reduction)
    algorithm = backend.get("collective_algorithm_config")
    if algorithm != STRATEGY_ND_ALGORITHM:
        raise BenchmarkValidationError(
            "fingerprint collective algorithm differs from the byte-pinned accepted StrategyND config"
        )
    to_apply_match = re.search(
        r"(?:^|,\s*)to_apply=([A-Za-z0-9_.%:-]+)(?:,|$)",
        _unquoted_text(reduction.raw_line),
    )
    if to_apply_match is None:
        raise BenchmarkValidationError("fingerprint all-reduce reducer is missing")
    reducer_name = to_apply_match.group(1)
    reducer = tuple(
        item
        for item in report.module.instructions
        if _computation_base(item.computation) == reducer_name
    )
    parameters = tuple(item for item in reducer if item.raw_opcode == "parameter")
    roots = tuple(
        item for item in reducer if item.raw_line.lstrip().startswith("ROOT ")
    )
    if (
        len(reducer) != 3
        or len(parameters) != 2
        or {item.operand_names for item in parameters} != {("0",), ("1",)}
        or any(
            tuple((shape.dtype, shape.dimensions) for shape in item.result_shapes)
            != (("bf16", ()),)
            for item in parameters
        )
        or len(roots) != 1
        or roots[0].raw_opcode != "add"
        or set(roots[0].operand_names) != {item.name for item in parameters}
        or tuple((shape.dtype, shape.dimensions) for shape in roots[0].result_shapes)
        != (("bf16", ()),)
    ):
        raise BenchmarkValidationError(
            "fingerprint all-reduce reducer is not the exact scalar BF16 add"
        )
    return dict(algorithm)


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
    algorithm = validate_strategy_nd_reduction(report, reduction)
    _validate_fingerprint_value_flow(report, reduction)
    return report, algorithm


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
    """Compile one exact-M32 collective reused by every trial invocation."""

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
        np.zeros((32, ACCEPTED_DECODE_BUCKET_ROWS, config.width), dtype=np.uint16),
        input_sharding,
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
    """Execute and repeat a bank of exact-M32 trial invocations."""

    import jax

    bits = np.asarray(input_bits, dtype=np.uint16)
    expected_shape = (compiled.config.trials, 32, compiled.config.width)
    if bits.shape != expected_shape:
        raise BenchmarkValidationError(
            f"fingerprint input bits must have shape {expected_shape}, got {bits.shape}"
        )
    expected_local_replicas = len(compiled.input_sharding.addressable_devices)

    def execute_bank() -> tuple[np.ndarray, list[tuple[str, ...]]]:
        outputs = []
        replica_hashes = []
        for trial in range(compiled.config.trials):
            # The canonical artifact remains [trial, member, width].  Repeating
            # one trial over all rows preserves the exact M32 buffer geometry
            # while making trial comparisons refer to fixed physical elements.
            distributed_bits = np.ascontiguousarray(
                np.broadcast_to(
                    bits[trial, :, None, :],
                    (32, ACCEPTED_DECODE_BUCKET_ROWS, compiled.config.width),
                )
            )
            value = jax.device_put(distributed_bits, compiled.input_sharding)
            result = compiled.compiled(value)
            jax.block_until_ready(result)
            local = [
                np.asarray(jax.device_get(shard.data), dtype=np.uint16).reshape(
                    ACCEPTED_DECODE_BUCKET_ROWS, compiled.config.width
                )
                for shard in sorted(
                    result.addressable_shards,
                    key=lambda shard: int(shard.device.id),
                )
            ]
            hashes = tuple(array_sha256(item) for item in local)
            if len(local) != expected_local_replicas or len(set(hashes)) != 1:
                raise BenchmarkValidationError(
                    "fingerprint output is not byte-identical on every local replica"
                )
            outputs.append(local[0])
            replica_hashes.append(hashes)
        return np.ascontiguousarray(np.stack(outputs)), replica_hashes

    output_bits, local_replica_hashes = execute_bank()
    repeated_bits, repeated_local_replica_hashes = execute_bank()
    if not np.array_equal(output_bits, repeated_bits):
        raise BenchmarkValidationError(
            "repeated StrategyND fingerprint bank produced different BF16 bits"
        )
    return output_bits, {
        "compile_bucket_rows": ACCEPTED_DECODE_BUCKET_ROWS,
        "determinism_repeat_invocations": compiled.config.trials,
        "input_bits_sha256": array_sha256(bits),
        "input_rows_replicated": True,
        "invocation_count": 2 * compiled.config.trials,
        "local_replica_output_sha256_by_trial": [
            list(hashes) for hashes in local_replica_hashes
        ],
        "measured_trial_invocations": compiled.config.trials,
        "output_bits_sha256": array_sha256(output_bits),
        "repeated_local_replica_output_sha256_by_trial": [
            list(hashes) for hashes in repeated_local_replica_hashes
        ],
        "repeated_output_bits_sha256": array_sha256(repeated_bits),
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


def analyze_m32_strategy_nd_fingerprint(
    input_bits: np.ndarray,
    output_bits: np.ndarray,
    member_device_ids: Sequence[int],
    device_coordinates: Mapping[int, Sequence[int]],
    *,
    block_width: int = 128,
) -> dict[str, Any]:
    """Replay repeated trials independently at every physical M32 row."""

    inputs = np.asarray(input_bits, dtype=np.uint16)
    outputs = np.asarray(output_bits, dtype=np.uint16)
    if inputs.ndim != 3 or inputs.shape[1] != 32:
        raise BenchmarkValidationError("input_bits must have shape [trial,32,width]")
    expected_output_shape = (
        inputs.shape[0],
        ACCEPTED_DECODE_BUCKET_ROWS,
        inputs.shape[2],
    )
    if outputs.shape != expected_output_shape:
        raise BenchmarkValidationError(
            "M32 output_bits must have shape "
            f"{expected_output_shape}, got {outputs.shape}"
        )

    rows = []
    output_groups: dict[str, list[int]] = {}
    for physical_row in range(ACCEPTED_DECODE_BUCKET_ROWS):
        row_output = np.ascontiguousarray(outputs[:, physical_row, :])
        row_analysis = analyze_strategy_nd_fingerprint(
            inputs,
            row_output,
            member_device_ids,
            device_coordinates,
            block_width=block_width,
        )
        row_analysis["physical_row"] = physical_row
        row_hash = array_sha256(row_output)
        row_analysis["row_output_bits_sha256"] = row_hash
        rows.append(row_analysis)
        output_groups.setdefault(row_hash, []).append(physical_row)

    union_counts = [row["union_exact_column_count"] for row in rows]
    return {
        "analysis_scope": (
            "per-physical-row bounded balanced four-way axis-pincer trees; "
            "raw artifacts permit broader replay"
        ),
        "block_width": block_width,
        "candidate_count_per_row": rows[0]["candidate_count"],
        "candidate_family_exhaustive": False,
        "compile_bucket_rows": ACCEPTED_DECODE_BUCKET_ROWS,
        "input_bits_sha256": array_sha256(inputs),
        "maximum_row_union_exact_column_count": max(union_counts),
        "minimum_row_union_exact_column_count": min(union_counts),
        "output_bits_sha256": array_sha256(outputs),
        "row_output_equivalence_groups": [
            {"row_output_bits_sha256": digest, "rows": group_rows}
            for digest, group_rows in sorted(output_groups.items())
        ],
        "rows": rows,
        "total_union_exact_column_count": sum(union_counts),
        "trials_per_physical_row": inputs.shape[0],
        "unique_row_output_count": len(output_groups),
        "width": inputs.shape[2],
    }
