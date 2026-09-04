"""Exact-M2048 model-free StrategyND row-zero association discriminator.

The accepted GLM prefill executable reduces a physical ``bf16[2048,6144]``
buffer across all 32 TPU-v4 chips.  The older DB533 fingerprint established
the association for a different ``bf16[32,6144]`` decode buffer.  This module
keeps the M2048 contract separate: it compiles one unbatched full-shape
collective, executes deterministic trials sequentially, and transfers only the
post-collective row-zero bits to the host.

This is diagnostic evidence only.  It neither makes a full-pod collective
admissible in the greenfield decoder nor proves that a recovered association
caused a model mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from .association_fingerprint import (
    STRATEGY_ND_ALGORITHM,
    StrategyNdFingerprintConfig,
    analyze_strategy_nd_fingerprint,
    array_sha256,
    generate_strategy_nd_input_bits,
)


M2048_ROWS = 2048
M2048_WIDTH = 6144
M2048_RESULT_LAYOUT = "{1,0:T(8,128)(2,1)S(3)}"
M2048_INPUT_LAYOUT = "{2,1,0:T(8,128)(2,1)}"
M2048_TRIALS = 32
M2048_SEED = 0x4D323038
ACCEPTED_M2048_HLO_RAW_SHA256 = (
    "e7371f4887ecf9fa38d381dcbaf3d5953294dbcda3079cf6846635722db07216"
)
ACCEPTED_M2048_HLO_GZIP_SHA256 = (
    "51d014de3776bae76b9c4068bf9ed3b06b690b2ce65531f68d372738b63d47f0"
)
ACCEPTED_M2048_SOURCE_TAG = (
    "greenfield_accepted_prompt_projection_lowering_recovery_"
    "20260809T105858202006975Z"
)


@dataclass(frozen=True, slots=True)
class M2048StrategyNdFingerprintConfig:
    """Immutable protected geometry for the exact-shape discriminator."""

    trials: int = M2048_TRIALS
    rows: int = M2048_ROWS
    width: int = M2048_WIDTH
    seed: int = M2048_SEED

    def __post_init__(self) -> None:
        for field in ("trials", "rows", "width", "seed"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool):
                raise BenchmarkValidationError(f"{field} must be an integer")
        if self.trials <= 0 or self.rows <= 0 or self.width <= 0 or self.seed < 0:
            raise BenchmarkValidationError(
                "trials/rows/width must be positive and seed non-negative"
            )

    def require_protected_contract(self) -> None:
        if (
            self.trials != M2048_TRIALS
            or self.rows != M2048_ROWS
            or self.width != M2048_WIDTH
            or self.seed != M2048_SEED
        ):
            raise BenchmarkValidationError(
                "protected M2048 fingerprint requires 32 trials, "
                "2048x6144 geometry, and the pinned seed"
            )

    def row0_config(self) -> StrategyNdFingerprintConfig:
        return StrategyNdFingerprintConfig(
            trials=self.trials, width=self.width, seed=self.seed
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "rows": self.rows,
            "seed": self.seed,
            "trials": self.trials,
            "width": self.width,
        }


@dataclass(frozen=True, slots=True)
class CompiledM2048StrategyNdFingerprint:
    config: M2048StrategyNdFingerprintConfig
    compiled: Any
    input_sharding: Any
    member_device_ids: tuple[int, ...]
    optimized_hlo: str
    stablehlo: str
    hlo_report: HloLintReport
    collective_algorithm: Mapping[str, Any]


def generate_m2048_row0_input_bits(
    config: M2048StrategyNdFingerprintConfig,
) -> np.ndarray:
    """Generate only the decisive per-member row-zero inputs.

    The returned artifact is small ``[trial,32,6144]``.  Execution expands it
    into a real per-device ``[2048,6144]`` parameter with zero-filled rows
    1..2047; no broadcast exists in the compiled program.
    """

    config.require_protected_contract()
    return generate_strategy_nd_input_bits(config.row0_config())


def m2048_strategy_nd_hlo_policy(
    member_device_ids: Sequence[int],
) -> HloContractPolicy:
    members = tuple(int(item) for item in member_device_ids)
    if members != tuple(range(32)):
        raise BenchmarkValidationError(
            "M2048 fingerprint requires sorted global member ids 0..31"
        )
    return HloContractPolicy(
        name="m2048-strategy-nd-association-fingerprint",
        total_devices=32,
        repeated_region_patterns=(r"m2048_strategy_nd_association_fingerprint",),
        maximum_repeated_collective_group_size=32,
        expected_repeated_replica_groups=(members,),
        expected_collectives=(CollectiveExpectation("all-reduce", 1),),
        partition_id_to_device_id=members,
        forbidden_row_width_pairs=(),
        allow_full_pod_repeated_collectives=True,
    )


def _unquoted(value: str) -> str:
    result = list(value)
    quoted = False
    escaped = False
    comment = False
    index = 0
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


def _backend_algorithm(instruction: HloInstruction) -> Mapping[str, Any]:
    marker = "backend_config="
    plain = _unquoted(instruction.raw_line)
    position = plain.find(marker)
    if position < 0 or plain.find(marker, position + len(marker)) >= 0:
        raise BenchmarkValidationError("M2048 all-reduce backend config drifted")
    source = instruction.raw_line[position + len(marker) :].lstrip()
    try:
        backend, end = json.JSONDecoder().raw_decode(source)
    except json.JSONDecodeError as error:
        raise BenchmarkValidationError(
            "M2048 all-reduce backend config is invalid JSON"
        ) from error
    if source[end:].strip().strip(",") or not isinstance(backend, dict):
        raise BenchmarkValidationError("M2048 backend config has trailing data")
    algorithm = backend.get("collective_algorithm_config")
    if algorithm != STRATEGY_ND_ALGORITHM:
        raise BenchmarkValidationError(
            "M2048 collective differs from the pinned StrategyND config"
        )
    return dict(algorithm)


def _computation_id(header: str) -> str:
    value = header.removeprefix("ENTRY ").split(None, 1)[0]
    return value.removeprefix("%").split("(", 1)[0]


def _called_computation(instruction: HloInstruction, attribute: str) -> str:
    match = re.search(
        rf"(?:^|,\s*){re.escape(attribute)}=%?([^,\s}}\]]+)(?:,|$)",
        _unquoted(instruction.raw_line),
    )
    if match is None:
        raise BenchmarkValidationError(
            f"M2048 {instruction.raw_opcode} lacks {attribute}"
        )
    return match.group(1)


def _shape(instruction: HloInstruction) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((item.dtype, item.dimensions) for item in instruction.result_shapes)


def _entry_depends_on(
    instructions: Sequence[HloInstruction],
    value: HloInstruction,
    source: HloInstruction,
) -> bool:
    """Conservative same-computation dependency for the deliberately tiny ENTRY."""

    by_name = {item.name: item for item in instructions}
    visiting: set[str] = set()
    memo: dict[str, bool] = {}

    def visit(name: str) -> bool:
        if name == source.name:
            return True
        if name in memo or name in visiting:
            return memo.get(name, False)
        current = by_name.get(name)
        if current is None:
            return False
        visiting.add(name)
        result = any(visit(operand) for operand in current.operand_names)
        visiting.remove(name)
        memo[name] = result
        return result

    return visit(value.name)


def _entry_ancestors(
    instructions: Sequence[HloInstruction], value: HloInstruction
) -> tuple[HloInstruction, ...]:
    by_name = {item.name: item for item in instructions}
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        item = by_name.get(name)
        if item is None:
            raise BenchmarkValidationError(
                f"M2048 ENTRY operand is undefined: {name}"
            )
        seen.add(name)
        if item.raw_opcode == "parameter":
            return
        for operand in item.operand_names:
            visit(operand)

    visit(value.name)
    return tuple(item for item in instructions if item.name in seen)


def _require_representation_only_fusion(
    report: HloLintReport,
    fusion: HloInstruction,
    *,
    require_row0_slice: bool,
) -> int:
    if len(fusion.operand_names) != 1:
        raise BenchmarkValidationError(
            "M2048 representation fusion must have exactly one caller operand"
        )
    callee_name = _called_computation(fusion, "calls")
    callee = tuple(
        item
        for item in report.module.instructions
        if _computation_id(item.computation) == callee_name
    )
    allowed = {
        "parameter",
        "bitcast-convert",
        "bitcast",
        "reshape",
        "copy",
    }
    if require_row0_slice:
        allowed.add("slice")
    parameters = tuple(item for item in callee if item.raw_opcode == "parameter")
    roots = tuple(
        item for item in callee if item.raw_line.lstrip().startswith("ROOT ")
    )
    if (
        not callee
        or len(parameters) != 1
        or parameters[0].operand_names != ("0",)
        or len(roots) != 1
        or any(item.raw_opcode not in allowed for item in callee)
    ):
        raise BenchmarkValidationError(
            "M2048 representation fusion structure drifted"
        )
    by_name = {item.name: item for item in callee}
    live: set[str] = set()
    current = roots[0]
    slices: list[HloInstruction] = []
    while True:
        if current.name in live:
            raise BenchmarkValidationError("M2048 representation fusion is cyclic")
        live.add(current.name)
        if current.raw_opcode == "parameter":
            break
        if len(current.operand_names) != 1:
            raise BenchmarkValidationError(
                "M2048 representation fusion path is not unary"
            )
        if current.raw_opcode == "slice":
            slices.append(current)
        operand = by_name.get(current.operand_names[0])
        if operand is None:
            raise BenchmarkValidationError(
                "M2048 representation fusion has an undefined operand"
            )
        current = operand
    if current is not parameters[0] or live != set(by_name):
        raise BenchmarkValidationError(
            "M2048 representation fusion contains a dead or alternate path"
        )
    if require_row0_slice:
        if len(slices) != 1:
            raise BenchmarkValidationError(
                "M2048 fused row-zero path does not contain one live slice"
            )
        row_slice = slices[0]
        if (
            "slice={[0:1], [0:6144]}" not in row_slice.raw_line
            or _shape(row_slice) != (("bf16", (1, M2048_WIDTH)),)
            or tuple(
                (item.dtype, item.dimensions) for item in row_slice.operand_shapes
            )
            != (("bf16", (M2048_ROWS, M2048_WIDTH)),)
        ):
            raise BenchmarkValidationError("M2048 fused row-zero slice drifted")
    elif slices:
        raise BenchmarkValidationError(
            "M2048 representation-only fusion unexpectedly slices its input"
        )
    return len(slices)


def _validate_entry_representation_path(
    report: HloLintReport,
    entry: Sequence[HloInstruction],
    value: HloInstruction,
    reduction: HloInstruction,
    *,
    require_row0_slice: bool,
) -> None:
    """Prove one unary representation path from the reduction to an output."""

    by_name = {item.name: item for item in entry}
    seen: set[str] = set()
    slice_count = 0
    current = value
    allowed = {
        "optimization-barrier",
        "opt-barrier",
        "bitcast-convert",
        "bitcast",
        "reshape",
        "copy",
        "fusion",
    }
    if require_row0_slice:
        allowed.add("slice")
    while current is not reduction:
        if current.name in seen or current.raw_opcode not in allowed:
            raise BenchmarkValidationError(
                "M2048 output is not a unary representation path from the collective"
            )
        seen.add(current.name)
        if len(current.operand_names) != 1:
            raise BenchmarkValidationError(
                "M2048 output representation path has an alternate operand"
            )
        if current.raw_opcode == "slice":
            slice_count += 1
            if (
                "slice={[0:1], [0:6144]}" not in current.raw_line
                or _shape(current) != (("bf16", (1, M2048_WIDTH)),)
                or tuple(
                    (item.dtype, item.dimensions) for item in current.operand_shapes
                )
                != (("bf16", (M2048_ROWS, M2048_WIDTH)),)
            ):
                raise BenchmarkValidationError("M2048 direct row-zero slice drifted")
        elif current.raw_opcode == "fusion":
            slice_count += _require_representation_only_fusion(
                report,
                current,
                require_row0_slice=require_row0_slice,
            )
        operand = by_name.get(current.operand_names[0])
        if operand is None:
            raise BenchmarkValidationError(
                "M2048 output representation path has an undefined operand"
            )
        current = operand
    if slice_count != int(require_row0_slice):
        raise BenchmarkValidationError(
            "M2048 row-zero post-collective slice is not unique and output-live"
        )


def _validate_scalar_bf16_add(
    report: HloLintReport, reduction: HloInstruction
) -> None:
    reducer_name = _called_computation(reduction, "to_apply")
    reducer = tuple(
        item
        for item in report.module.instructions
        if _computation_id(item.computation) == reducer_name
    )
    parameters = tuple(item for item in reducer if item.raw_opcode == "parameter")
    roots = tuple(item for item in reducer if item.raw_line.lstrip().startswith("ROOT "))
    if (
        len(reducer) != 3
        or len(parameters) != 2
        or {item.operand_names for item in parameters} != {("0",), ("1",)}
        or any(_shape(item) != (("bf16", ()),) for item in parameters)
        or len(roots) != 1
        or roots[0].raw_opcode != "add"
        or set(roots[0].operand_names) != {item.name for item in parameters}
        or _shape(roots[0]) != (("bf16", ()),)
    ):
        raise BenchmarkValidationError(
            "M2048 all-reduce reducer is not the exact scalar BF16 add"
        )


def validate_m2048_strategy_nd_fingerprint_hlo(
    optimized_hlo: str,
    member_device_ids: Sequence[int],
) -> tuple[HloLintReport, Mapping[str, Any]]:
    """Fail closed on the full-shape collective and both output-live roots."""

    report = lint_hlo(
        parse_hlo_module(optimized_hlo),
        m2048_strategy_nd_hlo_policy(member_device_ids),
    )
    report.raise_for_violations()
    reductions = tuple(
        item for item in report.module.collectives if item.opcode == "all-reduce"
    )
    if len(reductions) != 1:
        raise BenchmarkValidationError(
            f"M2048 fingerprint requires one all-reduce, found {len(reductions)}"
        )
    reduction = reductions[0]
    exact = (("bf16", (M2048_ROWS, M2048_WIDTH)),)
    if reduction.raw_opcode != "all-reduce":
        raise BenchmarkValidationError("M2048 fingerprint requires synchronous all-reduce")
    if _shape(reduction) != exact or tuple(
        (item.dtype, item.dimensions) for item in reduction.operand_shapes
    ) != exact:
        raise BenchmarkValidationError(
            "M2048 all-reduce operand/result must be bf16[2048,6144]"
        )
    if (
        f"bf16[{M2048_ROWS},{M2048_WIDTH}]{M2048_RESULT_LAYOUT} all-reduce("
        not in reduction.raw_line
    ):
        raise BenchmarkValidationError("M2048 TPU result layout drifted")
    if len(reduction.operand_names) != 1:
        raise BenchmarkValidationError("M2048 all-reduce operand count drifted")
    _validate_scalar_bf16_add(report, reduction)
    algorithm = _backend_algorithm(reduction)

    entry = tuple(
        item
        for item in report.module.instructions
        if item.computation == reduction.computation
    )
    parameters = tuple(item for item in entry if item.raw_opcode == "parameter")
    roots = tuple(item for item in entry if item.raw_line.lstrip().startswith("ROOT "))
    if len(parameters) != 1 or len(roots) != 1 or roots[0].raw_opcode != "tuple":
        raise BenchmarkValidationError("M2048 ENTRY parameter/root contract drifted")
    parameter = parameters[0]
    root = roots[0]
    if (
        _shape(parameter) != (("u16", (1, M2048_ROWS, M2048_WIDTH)),)
        or parameter.operand_names != ("0",)
        or (
            f"u16[1,{M2048_ROWS},{M2048_WIDTH}]{M2048_INPUT_LAYOUT} parameter(0)"
            not in parameter.raw_line
        )
        or len(root.operand_names) != 2
        or _shape(root)
        != (
            ("u16", (M2048_ROWS, M2048_WIDTH)),
            ("u16", (M2048_WIDTH,)),
        )
    ):
        raise BenchmarkValidationError("M2048 unbatched input/output geometry drifted")
    if not _entry_depends_on(entry, reduction, parameter):
        raise BenchmarkValidationError("M2048 collective does not consume the sole input")
    reduction_ancestors = _entry_ancestors(entry, reduction)
    allowed_precollective = {
        "parameter",
        "bitcast-convert",
        "bitcast",
        "reshape",
        "copy",
        "fusion",
        "all-reduce",
    }
    if any(item.raw_opcode not in allowed_precollective for item in reduction_ancestors):
        raise BenchmarkValidationError(
            "M2048 collective input contains unapproved source arithmetic"
        )
    for item in reduction_ancestors:
        if item.raw_opcode == "fusion":
            _require_representation_only_fusion(
                report, item, require_row0_slice=False
            )
    root_values = tuple(
        next((item for item in entry if item.name == name), None)
        for name in root.operand_names
    )
    if any(item is None for item in root_values):
        raise BenchmarkValidationError("M2048 tuple root has undefined operands")
    full, row0 = root_values
    assert full is not None and row0 is not None
    if _shape(full) != (("u16", (M2048_ROWS, M2048_WIDTH)),) or _shape(row0) != (
        ("u16", (M2048_WIDTH,)),
    ):
        raise BenchmarkValidationError("M2048 tuple root order/shape drifted")
    if not _entry_depends_on(entry, full, reduction) or not _entry_depends_on(
        entry, row0, reduction
    ):
        raise BenchmarkValidationError("M2048 output roots are not collective-live")
    _validate_entry_representation_path(
        report, entry, full, reduction, require_row0_slice=False
    )
    _validate_entry_representation_path(
        report, entry, row0, reduction, require_row0_slice=True
    )
    forbidden = {
        "all-gather",
        "all-to-all",
        "collective-permute",
        "reduce-scatter",
        "send",
        "recv",
    }
    if any(item.raw_opcode in forbidden for item in report.module.instructions):
        raise BenchmarkValidationError("M2048 fingerprint has extra communication")
    if any(
        token in optimized_hlo
        for token in ("host_callback", 'CustomCall("xla_python', "python_callback")
    ):
        raise BenchmarkValidationError("M2048 fingerprint has a host callback")
    return report, algorithm


def _m2048_fingerprint_function() -> Any:
    import jax
    from jax import lax
    import jax.numpy as jnp

    def fingerprint(local_bits: Any) -> tuple[Any, Any]:
        with jax.named_scope("m2048_strategy_nd_association_fingerprint"):
            payload = lax.bitcast_convert_type(local_bits[0], jnp.bfloat16)
            reduced = lax.psum(payload, "member")
            reduced = lax.optimization_barrier(reduced)
            full_bits = lax.bitcast_convert_type(reduced, jnp.uint16)
            row0_bits = lax.bitcast_convert_type(reduced[0], jnp.uint16)
            return full_bits, row0_bits

    return fingerprint


def build_m2048_strategy_nd_fingerprint(
    config: M2048StrategyNdFingerprintConfig,
    member_device_ids: Sequence[int],
    *,
    devices: Sequence[Any] | None = None,
    enforce_hlo_contract: bool = True,
) -> CompiledM2048StrategyNdFingerprint:
    """Compile one exact-M2048 collective; trials remain separate calls."""

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    config.require_protected_contract()
    runtime_devices = tuple(jax.devices() if devices is None else devices)
    members = tuple(int(item) for item in member_device_ids)
    by_id = {int(device.id): device for device in runtime_devices}
    if len(runtime_devices) != 32 or set(by_id) != set(range(32)):
        raise BenchmarkValidationError(
            "M2048 fingerprint requires global TPU ids 0..31"
        )
    m2048_strategy_nd_hlo_policy(members)
    mesh = Mesh(np.asarray([by_id[item] for item in members], dtype=object), ("member",))
    input_sharding = NamedSharding(mesh, P("member", None, None))
    mapped = jax.shard_map(
        _m2048_fingerprint_function(),
        mesh=mesh,
        in_specs=P("member", None, None),
        out_specs=(P(None, None), P(None)),
        check_vma=False,
    )
    executable = jax.jit(mapped)
    abstract = jax.ShapeDtypeStruct(
        (32, config.rows, config.width), jnp.uint16, sharding=input_sharding
    )
    lowered = executable.lower(abstract)
    compiled = lowered.compile()
    optimized_hlo = compiled.as_text()
    stablehlo = lowered.as_text()
    if enforce_hlo_contract:
        report, algorithm = validate_m2048_strategy_nd_fingerprint_hlo(
            optimized_hlo, members
        )
    else:
        report = lint_hlo(
            parse_hlo_module(optimized_hlo),
            m2048_strategy_nd_hlo_policy(members),
        )
        algorithm = {}
    return CompiledM2048StrategyNdFingerprint(
        config=config,
        compiled=compiled,
        input_sharding=input_sharding,
        member_device_ids=members,
        optimized_hlo=optimized_hlo,
        stablehlo=stablehlo,
        hlo_report=report,
        collective_algorithm=algorithm,
    )


def _materialize_trial(
    compiled: CompiledM2048StrategyNdFingerprint,
    row0_bits: np.ndarray,
) -> Any:
    import jax
    from jax.sharding import SingleDeviceSharding

    bits = np.asarray(row0_bits, dtype=np.uint16)
    if bits.shape != (32, compiled.config.width):
        raise BenchmarkValidationError("M2048 row-zero trial must be uint16[32,6144]")
    global_shape = (32, compiled.config.rows, compiled.config.width)
    indices = compiled.input_sharding.addressable_devices_indices_map(global_shape)
    member_index = {
        device_id: index
        for index, device_id in enumerate(compiled.member_device_ids)
    }
    shards = []

    def complete_slice(value: object, size: int) -> bool:
        return (
            isinstance(value, slice)
            and value.start in (None, 0)
            and value.stop in (None, size)
            and value.step in (None, 1)
        )

    for device in sorted(indices, key=lambda item: int(item.id)):
        index = indices[device]
        member_slice, row_slice, width_slice = index
        if (
            not isinstance(member_slice, slice)
            or (member_slice.start, member_slice.stop, member_slice.step) != (int(device.id), int(device.id) + 1, None)
            or not complete_slice(row_slice, compiled.config.rows)
            or not complete_slice(width_slice, compiled.config.width)
        ):
            raise BenchmarkValidationError("M2048 addressable shard mapping drifted")
        host = np.zeros((1, compiled.config.rows, compiled.config.width), dtype=np.uint16)
        host[0, 0, :] = bits[member_index[int(device.id)]]
        shards.append(jax.device_put(host, SingleDeviceSharding(device)))
    return jax.make_array_from_single_device_arrays(
        global_shape, compiled.input_sharding, shards
    )


def execute_m2048_strategy_nd_fingerprint(
    compiled: CompiledM2048StrategyNdFingerprint,
    input_row0_bits: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run two sequential banks and transfer only replicated row zero."""

    import jax

    bits = np.asarray(input_row0_bits, dtype=np.uint16)
    expected = (compiled.config.trials, 32, compiled.config.width)
    if bits.shape != expected:
        raise BenchmarkValidationError(
            f"M2048 input bits must have shape {expected}, got {bits.shape}"
        )
    expected_replicas = len(compiled.input_sharding.addressable_devices)

    def bank() -> tuple[np.ndarray, list[tuple[str, ...]]]:
        outputs = []
        replica_hashes = []
        for trial in range(compiled.config.trials):
            value = _materialize_trial(compiled, bits[trial])
            full_device, row0_device = compiled.compiled(value)
            jax.block_until_ready((full_device, row0_device))
            local = [
                np.asarray(jax.device_get(shard.data), dtype=np.uint16).reshape(
                    compiled.config.width
                )
                for shard in sorted(
                    row0_device.addressable_shards,
                    key=lambda shard: int(shard.device.id),
                )
            ]
            hashes = tuple(array_sha256(item) for item in local)
            if len(local) != expected_replicas or len(set(hashes)) != 1:
                raise BenchmarkValidationError(
                    "M2048 row zero is not identical on every local replica"
                )
            outputs.append(local[0])
            replica_hashes.append(hashes)
        return np.ascontiguousarray(np.stack(outputs)), replica_hashes

    output_bits, replica_hashes = bank()
    repeated_bits, repeated_replica_hashes = bank()
    if not np.array_equal(output_bits, repeated_bits):
        raise BenchmarkValidationError("M2048 repeated bank is nondeterministic")
    local_host_transfer_bytes = (
        2
        * compiled.config.trials
        * expected_replicas
        * compiled.config.width
        * np.dtype(np.uint16).itemsize
    )
    if 32 % expected_replicas:
        raise BenchmarkValidationError(
            "M2048 local replica count does not divide the 32-device fleet"
        )
    return output_bits, {
        "collective_input_shape": [M2048_ROWS, M2048_WIDTH],
        "determinism_repeat_invocations": compiled.config.trials,
        "full_output_device_resident": True,
        "fleet_host_transfer_bytes": local_host_transfer_bytes
        * (32 // expected_replicas),
        "input_row0_bits_sha256": array_sha256(bits),
        "input_rows_1_through_2047_zero": True,
        "invocation_count": 2 * compiled.config.trials,
        "local_replica_row0_sha256_by_trial": [list(item) for item in replica_hashes],
        "measured_trial_invocations": compiled.config.trials,
        "output_row0_bits_sha256": array_sha256(output_bits),
        "repeated_local_replica_row0_sha256_by_trial": [
            list(item) for item in repeated_replica_hashes
        ],
        "repeated_output_row0_bits_sha256": array_sha256(repeated_bits),
        "retained_output_artifact_bytes": int(output_bits.nbytes),
        "row0_slice_after_collective": True,
        "this_process_host_transfer_bytes": local_host_transfer_bytes,
    }


def analyze_m2048_row0_association(
    input_row0_bits: np.ndarray,
    output_row0_bits: np.ndarray,
    member_device_ids: Sequence[int],
    device_coordinates: Mapping[int, Sequence[int]],
) -> dict[str, Any]:
    """Require one declared StrategyND candidate for every hidden lane."""

    analysis = analyze_strategy_nd_fingerprint(
        input_row0_bits,
        output_row0_bits,
        member_device_ids,
        device_coordinates,
        block_width=128,
    )
    histogram = analysis["column_candidate_count_histogram"]
    unique = histogram == {"1": M2048_WIDTH}
    analysis.update(
        {
            "collective_geometry": [M2048_ROWS, M2048_WIDTH],
            "declared_candidate_family": (
                "two valid 4x2x4 physical-axis mappings x three StrategyND "
                "phase rotations x three balanced pairings per four-way axis"
            ),
            "declared_candidate_family_count": 54,
            "every_lane_has_exactly_one_candidate": unique,
            "row_index": 0,
        }
    )
    if not unique or analysis["uncovered_column_count"] != 0:
        raise BenchmarkValidationError(
            "M2048 row-zero association is not unique in the declared StrategyND family"
        )
    return analysis


def m2048_source_identity() -> dict[str, str]:
    return {
        "accepted_hlo_gzip_sha256": ACCEPTED_M2048_HLO_GZIP_SHA256,
        "accepted_hlo_raw_sha256": ACCEPTED_M2048_HLO_RAW_SHA256,
        "accepted_source_tag": ACCEPTED_M2048_SOURCE_TAG,
        "generator": "PCG64 cancellation-pair fingerprint v1",
    }
