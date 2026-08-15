"""Distinct raw-FP8 Pallas challenger for the bounded WS32 real layer."""

from __future__ import annotations

from typing import Any

import jax

from .ws32_one_layer import (
    WS32_ONE_LAYER_INPUT_SPECS,
    WS32_ONE_LAYER_OUTPUT_SPEC,
)
from ..kernels.reference.moe import GlmMoeNumericalContract
from ..kernels.ws32 import ws32_moe_pallas_from_routes_mapped


def build_ws32_pallas_one_layer_mapped(
    mesh: Any,
    *,
    contract: GlmMoeNumericalContract,
    interpret: bool = False,
) -> Any:
    """Return the default-off one-row Pallas WS32 shard-map body."""

    if contract.stage_size != 8 or contract.local_experts != 32:
        raise ValueError("real WS32 Pallas layer requires expert8 ownership")

    def body(*values: Any) -> Any:
        with jax.named_scope("greenfield_ws32_pallas_real_layer3"):
            return ws32_moe_pallas_from_routes_mapped(
                *values,
                contract=contract,
                expert_axis="expert",
                feature_axis="feature",
                interpret=interpret,
            )

    return jax.shard_map(
        body,
        mesh=mesh,
        in_specs=WS32_ONE_LAYER_INPUT_SPECS,
        out_specs=WS32_ONE_LAYER_OUTPUT_SPEC,
        check_vma=False,
    )


# Keep the mapped body above byte-for-byte at its acquired source coordinates:
# TPU optimized HLO records those line numbers in its FileLocations table.
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import re

from .ws32_one_layer import Ws32OneLayerHloReport, validate_ws32_one_layer_hlo
from ..sharding.hlo_contract import HloInstruction, parse_hlo_module


WS32_PALLAS_REAL_LAYER_STABLEHLO_SHA256 = (
    "fa11961d0d3393d93191f9d591eb7d9efd12a32f54fc046a2983e200579e6118"
)
WS32_PALLAS_REAL_LAYER_OPTIMIZED_HLO_SHA256 = (
    "17ee208a3f828e72910f569bad7a8923a9e24a41001234bc3688223458e7ff5d"
)
# This is the SHA-256 of the exact optimized-HLO instruction closure reachable
# from the ENTRY root.  Unlike a module-wide digest, it cannot be satisfied by
# leaving a reviewed kernel/reduction as a dead decoy and routing a different
# computation to the result.  It is deliberately compiler-version specific.
WS32_PALLAS_REAL_LAYER_LIVE_CLOSURE_SHA256 = (
    "0a03b130cfd56ff832141973bfaf23abe9e4ae530a4adb9e4fbaffaf226c6a51"
)

_F32_KERNEL = "greenfield_fp8_block_matmul_f32_m8_k1536_n2048"
_BF16_KERNEL = "greenfield_fp8_block_matmul_m8_k2048_n1536"
_PALLAS_SCOPE = "jit(body)/shard_map/greenfield_ws32_pallas_real_layer3/"
_FEATURE_GROUPS = tuple(
    tuple(range(start, start + 4)) for start in range(0, 32, 4)
)
_EXPERT_GROUPS = tuple(
    tuple(range(offset, 32, 4)) for offset in range(4)
)
_F32_RESULT_PREFIX = "f32[8,2048]{1,0:T(8,128)S(3)}"
_BF16_RESULT_PREFIX = "bf16[8,1536]{1,0:T(8,128)(2,1)S(3)}"
_F32_OPERAND_SHAPES = (
    ("bf16", (8, 1536)),
    ("u8", (2048, 1536)),
    ("f32", (16, 128)),
)
_BF16_OPERAND_SHAPES = (
    ("bf16", (8, 2048)),
    ("u8", (1536, 2048)),
    ("f32", (16, 128)),
)
_OPERAND_LAYOUT_CONSTRAINTS = (
    "operand_layout_constraints={bf16[8,1536]{1,0}, "
    "u8[2048,1536]{1,0}, f32[16,128]{1,0}}"
)
_BF16_OPERAND_LAYOUT_CONSTRAINTS = (
    "operand_layout_constraints={bf16[8,2048]{1,0}, "
    "u8[1536,2048]{1,0}, f32[16,128]{1,0}}"
)


@dataclass(frozen=True, slots=True)
class Ws32PallasOneLayerHloReport:
    """Exact source/kernel/collective proof for the Pallas WS32 layer."""

    base: Ws32OneLayerHloReport
    stablehlo_f32_pallas_count: int
    stablehlo_bf16_pallas_count: int
    stablehlo_exact_reducer_count: int
    optimized_f32_pallas_count: int
    optimized_bf16_pallas_count: int
    live_pallas_count: int
    route_region_count: int
    exact_route_region_count: int
    shared_pallas_count: int
    live_instruction_count: int
    live_closure_sha256: str
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.violations

    @property
    def stablehlo_sha256(self) -> str:
        return self.base.stablehlo_sha256

    @property
    def optimized_hlo_sha256(self) -> str:
        return self.base.optimized_hlo_sha256

    def to_dict(self) -> dict[str, Any]:
        result = self.base.to_dict()
        result.update(
            {
                "exact_route_region_count": self.exact_route_region_count,
                "live_closure_sha256": self.live_closure_sha256,
                "live_instruction_count": self.live_instruction_count,
                "live_pallas_count": self.live_pallas_count,
                "optimized_bf16_pallas_count": self.optimized_bf16_pallas_count,
                "optimized_f32_pallas_count": self.optimized_f32_pallas_count,
                "passed": self.passed,
                "route_region_count": self.route_region_count,
                "shared_pallas_count": self.shared_pallas_count,
                "stablehlo_bf16_pallas_count": self.stablehlo_bf16_pallas_count,
                "stablehlo_exact_reducer_count": self.stablehlo_exact_reducer_count,
                "stablehlo_f32_pallas_count": self.stablehlo_f32_pallas_count,
                "violations": list(self.violations),
            }
        )
        return result


def _computation_base(value: str) -> str:
    return value.split(" ", 1)[0]


def _called_computations(instruction: HloInstruction) -> tuple[str, ...]:
    result: list[str] = []
    consumed: set[str] = set()
    if instruction.raw_opcode in {"fusion", "call"}:
        match = re.search(r"\bcalls=([^,\s]+)", instruction.raw_line)
        if match is None:
            raise ValueError(f"HLO caller {instruction.name} omits its callee")
        result.append(match.group(1))
        consumed.add("calls")
    if instruction.raw_opcode == "conditional":
        match = re.search(
            r"\bbranch_computations=\{([^}]*)\}", instruction.raw_line
        )
        if match is None:
            raise ValueError(f"conditional {instruction.name} omits branches")
        result.extend(item.strip() for item in match.group(1).split(","))
        consumed.add("branch_computations")
    if instruction.raw_opcode in {
        "all-reduce",
        "all-reduce-start",
        "reduce-scatter",
        "reduce-scatter-start",
    }:
        match = re.search(r"\bto_apply=([^,\s]+)", instruction.raw_line)
        if match is None:
            raise ValueError(f"collective {instruction.name} omits its reducer")
        result.append(match.group(1))
        consumed.add("to_apply")
    callee_attributes = {
        "body",
        "branch_computations",
        "called_computations",
        "calls",
        "condition",
        "to_apply",
    }
    unconsumed = tuple(
        attribute
        for attribute in sorted(callee_attributes - consumed)
        if re.search(rf"\b{attribute}\s*=", instruction.raw_line)
    )
    if unconsumed:
        raise ValueError(
            f"HLO instruction {instruction.name} has unsupported callee attributes "
            f"{unconsumed}"
        )
    return tuple(result)


def _live_instruction_closure(
    instructions: tuple[HloInstruction, ...],
) -> tuple[HloInstruction, ...]:
    """Return the complete cross-computation instruction closure of ENTRY."""

    by_computation: dict[str, dict[str, HloInstruction]] = defaultdict(dict)
    roots: dict[str, HloInstruction] = {}
    for instruction in instructions:
        computation = _computation_base(instruction.computation)
        by_computation[computation][instruction.name] = instruction
        if instruction.raw_line.lstrip().startswith("ROOT "):
            if computation in roots:
                raise ValueError(f"multiple HLO roots in {computation}")
            roots[computation] = instruction
    entry_roots = tuple(
        root for name, root in roots.items() if name == "ENTRY" or name == "main"
    )
    if len(entry_roots) != 1:
        raise ValueError(f"expected one ENTRY root, found {len(entry_roots)}")

    live: set[tuple[str, str]] = set()

    def visit(computation: str, name: str) -> None:
        key = (computation, name)
        if key in live:
            return
        instruction = by_computation.get(computation, {}).get(name)
        if instruction is None:
            raise ValueError(f"undefined live HLO value {computation}:{name}")
        live.add(key)
        for operand in instruction.operand_names:
            if operand.startswith("%"):
                visit(computation, operand)
        for callee in _called_computations(instruction):
            root = roots.get(callee)
            if root is None:
                raise ValueError(f"live HLO callee {callee} has no root")
            visit(callee, root.name)

    entry_root = entry_roots[0]
    visit(_computation_base(entry_root.computation), entry_root.name)
    return tuple(
        instruction
        for instruction in sorted(instructions, key=lambda item: item.index)
        if (_computation_base(instruction.computation), instruction.name) in live
    )


def _live_closure_sha256(instructions: tuple[HloInstruction, ...]) -> str:
    live = _live_instruction_closure(instructions)
    payload = [
        {
            "computation": instruction.computation,
            "raw_line": instruction.raw_line,
        }
        for instruction in live
    ]
    return sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _custom_call_target(instruction: HloInstruction) -> str | None:
    matches = re.findall(
        r'(?<!\\)\bcustom_call_target="([^"\\]+)"', instruction.raw_line
    )
    return matches[0] if len(matches) == 1 else None


def _is_pallas_call(instruction: HloInstruction) -> bool:
    return (
        instruction.raw_opcode == "custom-call"
        and _custom_call_target(instruction) == "tpu_custom_call"
        and instruction.op_name is not None
        and instruction.op_name.startswith(_PALLAS_SCOPE)
        and instruction.op_name.endswith("/pallas_call")
    )


def _shape_signature(instruction: HloInstruction) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((shape.dtype, shape.dimensions) for shape in instruction.result_shapes)


def _operand_shape_signature(
    instruction: HloInstruction,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((shape.dtype, shape.dimensions) for shape in instruction.operand_shapes)


def _exact_pallas_call(instruction: HloInstruction, kernel: str) -> bool:
    if not _is_pallas_call(instruction) or instruction.op_name is None:
        return False
    if f"/{kernel}/pallas_call" not in instruction.op_name:
        return False
    if kernel == _F32_KERNEL:
        return (
            _shape_signature(instruction) == (("f32", (8, 2048)),)
            and _operand_shape_signature(instruction) == _F32_OPERAND_SHAPES
            and instruction.raw_line.split(" = ", 1)[1].startswith(
                f"{_F32_RESULT_PREFIX} custom-call("
            )
            and instruction.raw_line.count(_OPERAND_LAYOUT_CONSTRAINTS) == 1
        )
    if kernel == _BF16_KERNEL:
        return (
            _shape_signature(instruction) == (("bf16", (8, 1536)),)
            and _operand_shape_signature(instruction) == _BF16_OPERAND_SHAPES
            and instruction.raw_line.split(" = ", 1)[1].startswith(
                f"{_BF16_RESULT_PREFIX} custom-call("
            )
            and instruction.raw_line.count(_BF16_OPERAND_LAYOUT_CONSTRAINTS) == 1
        )
    return False


def _stablehlo_pallas_calls(stablehlo: str) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in stablehlo.splitlines()
        if "stablehlo.custom_call @tpu_custom_call(" in line
    )


def _stablehlo_kernel_name(line: str) -> str | None:
    without_comments = re.sub(r"/\*.*?\*/", "", line)
    matches = re.findall(r'(?<!\\)\bkernel_name\s*=\s*"([^"\\]+)"', without_comments)
    return matches[0] if len(matches) == 1 else None


def _exact_stablehlo_pallas_call(line: str, kernel: str) -> bool:
    if _stablehlo_kernel_name(line) != kernel:
        return False
    common = (
        "operand_layouts = [dense<[1, 0]> : tensor<2xindex>, "
        "dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], "
        "result_layouts = [dense<[1, 0]> : tensor<2xindex>]"
    )
    signature = (
        ": (tensor<8x1536xbf16>, tensor<2048x1536xui8>, "
        "tensor<16x128xf32>) -> tensor<8x2048xf32>"
        if kernel == _F32_KERNEL
        else ": (tensor<8x2048xbf16>, tensor<1536x2048xui8>, "
        "tensor<16x128xf32>) -> tensor<8x1536xbf16>"
    )
    return line.count(common) == 1 and line.endswith(signature)


def _stablehlo_exact_reducer_count(stablehlo: str) -> int:
    lines = stablehlo.splitlines()
    exact = 0
    for index, line in enumerate(lines):
        if '"stablehlo.all_reduce"(' not in line:
            continue
        block = "\n".join(lines[index : index + 5])
        arguments = re.search(
            r"\^bb0\((%[A-Za-z0-9_.$#-]+): tensor<f32>, "
            r"(%[A-Za-z0-9_.$#-]+): tensor<f32>\):",
            block,
        )
        added = re.search(
            r"(%[A-Za-z0-9_.$#-]+) = stablehlo\.add "
            r"(%[A-Za-z0-9_.$#-]+), (%[A-Za-z0-9_.$#-]+) : tensor<f32>",
            block,
        )
        returned = re.search(
            r"stablehlo\.return (%[A-Za-z0-9_.$#-]+) : tensor<f32>", block
        )
        if (
            arguments is not None
            and added is not None
            and returned is not None
            and added.group(2, 3) == arguments.group(1, 2)
            and returned.group(1) == added.group(1)
        ):
            exact += 1
    return exact


def validate_ws32_pallas_one_layer_hlo(
    stablehlo: str,
    optimized_hlo: str,
    *,
    expected_stablehlo_sha256: str,
    expected_optimized_hlo_sha256: str,
    expected_optimized_collective_result_dtype: str | None,
) -> Ws32PallasOneLayerHloReport:
    """Validate the exact live raw-FP8 Pallas WS32 one-layer schedule."""

    base = validate_ws32_one_layer_hlo(
        stablehlo,
        optimized_hlo,
        expected_stablehlo_sha256=expected_stablehlo_sha256,
        expected_optimized_hlo_sha256=expected_optimized_hlo_sha256,
        expected_optimized_collective_result_dtype=(
            expected_optimized_collective_result_dtype
        ),
    )
    violations = list(base.violations)

    stable_calls = _stablehlo_pallas_calls(stablehlo)
    stable_f32 = tuple(
        line for line in stable_calls if _stablehlo_kernel_name(line) == _F32_KERNEL
    )
    stable_bf16 = tuple(
        line for line in stable_calls if _stablehlo_kernel_name(line) == _BF16_KERNEL
    )
    if len(stable_calls) != 27 or len(stable_f32) != 18 or len(stable_bf16) != 9:
        violations.append(
            "StableHLO Pallas kernel cardinality drifted: "
            f"total={len(stable_calls)} f32={len(stable_f32)} bf16={len(stable_bf16)}"
        )
    if not all(_exact_stablehlo_pallas_call(line, _F32_KERNEL) for line in stable_f32):
        violations.append("StableHLO FP32 Pallas call contract drifted")
    if not all(_exact_stablehlo_pallas_call(line, _BF16_KERNEL) for line in stable_bf16):
        violations.append("StableHLO BF16 Pallas call contract drifted")
    stable_reducers = _stablehlo_exact_reducer_count(stablehlo)
    if stable_reducers != 10:
        violations.append(
            f"StableHLO exact scalar-add reducer count drifted: {stable_reducers}"
        )

    module = parse_hlo_module(optimized_hlo)
    try:
        live = _live_instruction_closure(module.instructions)
        live_digest = _live_closure_sha256(module.instructions)
    except ValueError as error:
        live = ()
        live_digest = ""
        violations.append(f"optimized HLO live closure is invalid: {error}")
    parsed_computations = {
        _computation_base(item.computation) for item in module.instructions
    }
    live_computations = {
        _computation_base(item.computation) for item in live
    }
    unreachable_computations = tuple(
        sorted(parsed_computations - live_computations)
    )
    if unreachable_computations:
        violations.append(
            "optimized HLO has unreachable computations: "
            f"{unreachable_computations}"
        )
    live_keys = {
        (_computation_base(item.computation), item.name) for item in live
    }
    pallas = tuple(item for item in module.instructions if _is_pallas_call(item))
    f32_calls = tuple(
        item for item in pallas if _exact_pallas_call(item, _F32_KERNEL)
    )
    bf16_calls = tuple(
        item for item in pallas if _exact_pallas_call(item, _BF16_KERNEL)
    )
    if len(pallas) != 27 or len(f32_calls) != 18 or len(bf16_calls) != 9:
        violations.append(
            "optimized HLO exact Pallas kernel contract drifted: "
            f"total={len(pallas)} f32={len(f32_calls)} bf16={len(bf16_calls)}"
        )
    live_pallas = tuple(
        item
        for item in pallas
        if (_computation_base(item.computation), item.name) in live_keys
    )
    if len(live_pallas) != 27:
        violations.append(
            f"optimized HLO has dead Pallas calls: live={len(live_pallas)} total={len(pallas)}"
        )
    if live_digest != WS32_PALLAS_REAL_LAYER_LIVE_CLOSURE_SHA256:
        violations.append("optimized HLO exact live-root closure drifted")
    if any(
        item.raw_opcode.endswith(("-start", "-done"))
        and any(
            token in item.raw_opcode
            for token in (
                "all-gather", "all-reduce", "all-to-all", "collective-permute", "reduce-scatter"
            )
        )
        for item in module.instructions
    ):
        violations.append("optimized HLO contains an asynchronous collective")

    route_collectives: dict[int, HloInstruction] = {}
    route_pattern = re.compile(r"/route_([0-7])_feature_reduce/psum$")
    for item in module.collectives:
        match = None if item.op_name is None else route_pattern.search(item.op_name)
        if match is not None:
            route = int(match.group(1))
            if route in route_collectives:
                violations.append(f"route {route} has duplicate feature reductions")
            route_collectives[route] = item
    exact_route_regions = 0
    for route in range(8):
        collective = route_collectives.get(route)
        if collective is None:
            continue
        computation = _computation_base(collective.computation)
        region_pallas = tuple(
            item
            for item in pallas
            if _computation_base(item.computation) == computation
        )
        roots = tuple(
            item
            for item in module.instructions
            if _computation_base(item.computation) == computation
            and item.raw_line.lstrip().startswith("ROOT ")
        )
        if (
            Counter(_shape_signature(item) for item in region_pallas)
            == Counter({(("f32", (8, 2048)),): 2, (("bf16", (8, 1536)),): 1})
            and len(roots) == 1
            and all(
                (computation, item.name) in live_keys
                for item in (*region_pallas, collective, roots[0])
            )
            and collective.replica_groups == _FEATURE_GROUPS
            and _shape_signature(collective) == (("bf16", (2, 1, 2048)),)
            and _operand_shape_signature(collective)
            == (("f32", (2, 1, 2048)),)
        ):
            exact_route_regions += 1
    if set(route_collectives) != set(range(8)) or exact_route_regions != 8:
        violations.append(
            "optimized HLO route/Pallas/feature-reduction bijection drifted: "
            f"routes={sorted(route_collectives)} exact={exact_route_regions}"
        )

    entry_pallas = tuple(
        item for item in pallas if _computation_base(item.computation) == "ENTRY"
    )
    shared_collectives = tuple(
        item
        for item in module.collectives
        if item.op_name is not None
        and item.op_name.endswith("/shared_feature_reduce/psum")
    )
    expert_collectives = tuple(
        item
        for item in module.collectives
        if item.op_name is not None
        and item.op_name.endswith("/routed_expert_reduce/psum")
    )
    if (
        Counter(_shape_signature(item) for item in entry_pallas)
        != Counter({(("f32", (8, 2048)),): 2, (("bf16", (8, 1536)),): 1})
        or len(shared_collectives) != 1
        or shared_collectives[0].replica_groups != _FEATURE_GROUPS
        or len(expert_collectives) != 1
        or expert_collectives[0].replica_groups != _EXPERT_GROUPS
    ):
        violations.append("optimized HLO shared/expert live graph drifted")

    return Ws32PallasOneLayerHloReport(
        base=base,
        stablehlo_f32_pallas_count=len(stable_f32),
        stablehlo_bf16_pallas_count=len(stable_bf16),
        stablehlo_exact_reducer_count=stable_reducers,
        optimized_f32_pallas_count=len(f32_calls),
        optimized_bf16_pallas_count=len(bf16_calls),
        live_pallas_count=len(live_pallas),
        route_region_count=len(route_collectives),
        exact_route_region_count=exact_route_regions,
        shared_pallas_count=len(entry_pallas),
        live_instruction_count=len(live),
        live_closure_sha256=live_digest,
        violations=tuple(dict.fromkeys(violations)),
    )
