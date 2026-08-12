#!/usr/bin/env python3
"""Replay layer 0 after DB537's exact latent and test reduction association."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

import ml_dtypes
import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_POSITION = 8155
_LATENT_SHA256 = "923e9bfeb65864868cef359cf98ebad67f2756794ad142ac87e66b71877a2d2a"
_VALUE_SHA256 = "79a6e290274ef470b518a6de894b44d2929ad6861d1751f0915aa4eb20cf2e9d"
_LAYER1_SHA256 = "9936ee1e19049b297fd205292ebc378aee41d59401bbf56497004356998d3039"
_COMBINED_RESIDUAL_SHA256 = (
    "7de70cc78359374e5903afc156bc579362036cf459c3d03249761c76f22efea3"
)
_ASSOCIATION_CODE_HASH = "a9e6307bdad70b883ba82456fbf8f4bdf8db5ac6"
_ACCEPTED_MODEL_AXIS_DEVICE_IDS = (
    0,
    8,
    16,
    24,
    2,
    10,
    18,
    26,
    4,
    12,
    20,
    28,
    6,
    14,
    22,
    30,
    1,
    9,
    17,
    25,
    3,
    11,
    19,
    27,
    5,
    13,
    21,
    29,
    7,
    15,
    23,
    31,
)
_WEIGHT_CONTRACT: dict[str, tuple[tuple[int, ...], str, bool]] = {
    "attention.slot_00.kv_b.weight_bits": ((7168, 512), "uint8", False),
    "attention.slot_00.kv_b.scale_inv": ((56, 4), "float32", False),
    "attention.slot_00.o.weight_bits": ((6144, 4096), "uint8", False),
    "attention.slot_00.o.scale_inv": ((48, 32), "float32", False),
    "attention.slot_00.post_norm": ((6144,), "bfloat16", True),
    "dense.slot_00.gate.weight_bits": ((3072, 6144), "uint8", False),
    "dense.slot_00.gate.scale_inv": ((24, 48), "float32", False),
    "dense.slot_00.up.weight_bits": ((3072, 6144), "uint8", False),
    "dense.slot_00.up.scale_inv": ((24, 48), "float32", False),
    "dense.slot_00.down.weight_bits": ((6144, 3072), "uint8", False),
    "dense.slot_00.down.scale_inv": ((48, 24), "float32", False),
    "attention.slot_01.input_norm": ((6144,), "bfloat16", True),
}


@dataclass(frozen=True, slots=True)
class Arm:
    name: str
    attention_strategy_nd: bool
    dense_strategy_nd: bool


_ARMS = (
    Arm("local_attention_local_dense", False, False),
    Arm("strategy_attention_local_dense", True, False),
    Arm("local_attention_strategy_dense", False, True),
    Arm("strategy_attention_strategy_dense", True, True),
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _compare_bits(expected: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    if expected.shape != observed.shape or expected.dtype != np.uint16 or (
        observed.dtype != np.uint16
    ):
        raise ValueError("projection comparison shape/dtype drifted")
    expected_f32 = expected.view(ml_dtypes.bfloat16).astype(np.float32)
    observed_f32 = observed.view(ml_dtypes.bfloat16).astype(np.float32)
    absolute = np.abs(observed_f32 - expected_f32)
    mismatch = expected != observed
    return {
        "elementwise_exact": bool(np.array_equal(expected, observed)),
        "expected_sha256": _array_sha256(expected),
        "first_mismatch_index": (
            int(np.flatnonzero(mismatch)[0]) if np.any(mismatch) else None
        ),
        "max_abs_error": float(absolute.max(initial=0.0)),
        "mean_abs_error": float(absolute.mean()),
        "mismatch_count": int(np.count_nonzero(mismatch)),
        "observed_sha256": _array_sha256(observed),
        "shape": list(expected.shape),
    }


def _require_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file() or _file_sha256(path) != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 drifted")


def _load_sources(args: argparse.Namespace) -> tuple[np.ndarray, ...]:
    _require_file(
        args.attention_arithmetic,
        args.attention_arithmetic_sha256,
        "DB537 tensor",
    )
    _require_file(args.attention_runner, args.attention_runner_sha256, "DB537 runner")
    runner = json.loads(args.attention_runner.read_text())
    if (
        runner.get("artifact_kind") != "glm52_layer0_attention_arithmetic_probe"
        or runner.get("classification") != "exact_arithmetic_arm_identified"
        or "pregathered_h16_b512" not in runner.get("exact_arms", ())
        or "pregathered_h16_b512" not in runner.get("greenfield_cache_exact_arms", ())
    ):
        raise RuntimeError("DB537 exact-latent contract drifted")
    with np.load(args.attention_arithmetic, allow_pickle=False) as payload:
        key = "attended_latent_bfloat16_bits__pregathered_h16_b512__greenfield_cache"
        if key not in payload.files:
            raise RuntimeError("DB537 exact B512 latent is missing")
        latent_bits = np.ascontiguousarray(payload[key])
    if (
        latent_bits.shape != (64, 512)
        or latent_bits.dtype != np.uint16
        or _array_sha256(latent_bits) != _LATENT_SHA256
    ):
        raise RuntimeError("DB537 exact B512 latent drifted")

    _require_file(
        args.accepted_projection,
        args.accepted_projection_sha256,
        "accepted projection",
    )
    _require_file(
        args.accepted_projection_capture,
        args.accepted_projection_capture_sha256,
        "accepted projection capture",
    )
    capture = json.loads(args.accepted_projection_capture.read_text())
    if (
        capture.get("capture_mode") != "attention_projection"
        or capture.get("position") != _POSITION
        or capture.get("tensors", {})
        .get("attended_latent_bfloat16_bits", {})
        .get("shape")
        != [64, 512]
        or capture.get("tensors", {})
        .get("attention_output_bfloat16_bits", {})
        .get("shape")
        != [16384]
        or capture.get("tensors", {})
        .get("attended_latent_bfloat16_bits", {})
        .get("dtype")
        != "uint16"
        or capture.get("tensors", {})
        .get("attended_latent_bfloat16_bits", {})
        .get("sha256")
        != _LATENT_SHA256
        or capture.get("tensors", {})
        .get("attention_output_bfloat16_bits", {})
        .get("dtype")
        != "uint16"
        or capture.get("tensors", {})
        .get("attention_output_bfloat16_bits", {})
        .get("sha256")
        != _VALUE_SHA256
    ):
        raise RuntimeError("accepted projection capture contract drifted")
    with np.load(args.accepted_projection, allow_pickle=False) as payload:
        if set(payload.files) != {
            "attended_latent_bfloat16_bits",
            "attention_output_bfloat16_bits",
        }:
            raise RuntimeError("accepted projection fields drifted")
        accepted_latent = np.ascontiguousarray(
            payload["attended_latent_bfloat16_bits"]
        )
        accepted_value = np.ascontiguousarray(
            payload["attention_output_bfloat16_bits"]
        )
    if (
        accepted_latent.shape != (64, 512)
        or accepted_value.shape != (16384,)
        or accepted_latent.dtype != np.uint16
        or accepted_value.dtype != np.uint16
        or _array_sha256(accepted_latent) != _LATENT_SHA256
        or _array_sha256(accepted_value) != _VALUE_SHA256
        or not np.array_equal(latent_bits, accepted_latent)
    ):
        raise RuntimeError("accepted projection tensor identity drifted")

    _require_file(args.ingredients, args.ingredients_sha256, "layer-0 ingredients")
    _require_file(
        args.ingredients_contract,
        args.ingredients_contract_sha256,
        "ingredient contract",
    )
    contract = json.loads(args.ingredients_contract.read_text())
    if (
        contract.get("decode_position") != _POSITION
        or contract.get("source_state") != "post_teacher_forced_prefill"
        or not contract.get("passed")
    ):
        raise RuntimeError("layer-0 ingredient source contract drifted")
    with np.load(args.ingredients, allow_pickle=False) as payload:
        if "combined_residual_bfloat16_bits" not in payload.files:
            raise RuntimeError("layer-0 combined residual is missing")
        residual_bits = np.ascontiguousarray(
            payload["combined_residual_bfloat16_bits"]
        )
    if (
        residual_bits.shape != (4, 6144)
        or residual_bits.dtype != np.uint16
        or _array_sha256(residual_bits) != _COMBINED_RESIDUAL_SHA256
        or not np.all(residual_bits == residual_bits[0])
    ):
        raise RuntimeError("layer-0 combined residual drifted")

    _require_file(
        args.layer1_reference,
        args.layer1_reference_sha256,
        "layer-1 reference",
    )
    with np.load(args.layer1_reference, allow_pickle=False) as payload:
        if "accepted__normalized_hidden" not in payload.files:
            raise RuntimeError("accepted layer-1 normalized row is missing")
        layer1_bits = np.ascontiguousarray(payload["accepted__normalized_hidden"])
        position = int(payload["position"])
        layer_id = int(payload["layer_id"])
    if (
        layer1_bits.shape != (6144,)
        or layer1_bits.dtype != np.uint16
        or _array_sha256(layer1_bits) != _LAYER1_SHA256
        or position != _POSITION
        or layer_id != 1
    ):
        raise RuntimeError("accepted layer-1 reference drifted")
    return (
        latent_bits.view(ml_dtypes.bfloat16),
        accepted_value,
        residual_bits[0].view(ml_dtypes.bfloat16)[None, :],
        layer1_bits,
    )


def _load_association_analysis(
    path: Path, *, expected_sha256: str
) -> dict[str, Any]:
    """Bind the diagnostic StrategyND helper to protected DB533 evidence."""

    _require_file(path, expected_sha256, "StrategyND association analysis")
    analysis = json.loads(path.read_text())
    rows = analysis.get("rows")
    if (
        analysis.get("code_hash") != _ASSOCIATION_CODE_HASH
        or tuple(analysis.get("accepted_model_axis_device_ids", ()))
        != _ACCEPTED_MODEL_AXIS_DEVICE_IDS
        or analysis.get("compile_bucket_rows") != 32
        or analysis.get("trials_per_physical_row") != 32
        or analysis.get("width") != 6144
        or analysis.get("block_width") != 128
        or analysis.get("minimum_row_union_exact_column_count") != 6144
        or analysis.get("maximum_row_union_exact_column_count") != 6144
        or analysis.get("total_union_exact_column_count") != 32 * 6144
        or not isinstance(rows, list)
        or len(rows) != 32
    ):
        raise RuntimeError("StrategyND association analysis contract drifted")
    for physical_row, row in enumerate(rows):
        if (
            row.get("physical_row") != physical_row
            or row.get("width") != 6144
            or row.get("trials") != 32
            or row.get("union_exact_column_count") != 6144
            or row.get("uncovered_column_count") != 0
        ):
            raise RuntimeError(
                f"StrategyND row {physical_row} association evidence drifted"
            )
    return {
        "accepted_model_axis_device_ids": list(_ACCEPTED_MODEL_AXIS_DEVICE_IDS),
        "analysis_code_hash": _ASSOCIATION_CODE_HASH,
        "analysis_sha256": expected_sha256,
        "row_count": len(rows),
        "trials_per_physical_row": 32,
        "width": 6144,
    }


def _load_weights(
    checkpoint_root: Path, *, manifest_sha256: str
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    from safetensors import safe_open

    manifest_path = checkpoint_root / "runtime_manifest.json"
    _require_file(manifest_path, manifest_sha256, "runtime manifest")
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files")
    if (
        manifest.get("artifact_kind")
        != "greenfield_feature_runtime_packed_checkpoint"
        or manifest.get("plan_id") != "PP8_LP4"
        or manifest.get("file_count") != 32
        or not isinstance(files, list)
        or len(files) != 32
    ):
        raise RuntimeError("runtime checkpoint contract drifted")
    manifest_files = {
        record.get("destination_filename"): record for record in files
    }
    if len(manifest_files) != 32 or None in manifest_files:
        raise RuntimeError("runtime checkpoint filenames are not unique")
    by_name: dict[str, list[np.ndarray]] = {
        name: [] for name in _WEIGHT_CONTRACT
    }
    records: list[dict[str, Any]] = []
    for slot in range(4):
        relative = Path(
            f"base_decoder_runtime_feature/stage_00/device_slot_{slot:02d}.safetensors"
        )
        tensor_path = checkpoint_root / relative
        evidence_path = checkpoint_root / "evidence" / relative.with_suffix(
            ".safetensors.json"
        )
        manifest_file = manifest_files.get(relative.as_posix())
        if (
            manifest_file is None
            or manifest_file.get("stage_id") != 0
            or manifest_file.get("device_slot") != slot
        ):
            raise RuntimeError(f"runtime manifest lacks stage-0 slot {slot}")
        evidence = json.loads(evidence_path.read_text())
        if evidence != manifest_file:
            raise RuntimeError(f"slot {slot} evidence differs from runtime manifest")
        header_bytes = manifest_file.get("header_bytes")
        if (
            not isinstance(header_bytes, int)
            or header_bytes <= 8
            or tensor_path.stat().st_size != manifest_file.get("file_bytes")
        ):
            raise RuntimeError(f"slot {slot} checkpoint file metadata drifted")
        with tensor_path.open("rb") as stream:
            header_sha = sha256(stream.read(header_bytes)).hexdigest()
        if header_sha != manifest_file.get("header_sha256"):
            raise RuntimeError(f"slot {slot} checkpoint header drifted")
        manifest_tensors = {item["name"]: item for item in manifest_file["tensors"]}
        if len(manifest_tensors) != len(manifest_file["tensors"]):
            raise RuntimeError(f"slot {slot} tensor names are not unique")
        slot_records = []
        with safe_open(tensor_path, framework="np") as handle:
            for name, (shape, dtype, _) in _WEIGHT_CONTRACT.items():
                record = manifest_tensors.get(name)
                if record is None or name not in handle.keys():
                    raise RuntimeError(f"slot {slot} lacks {name}")
                value = np.ascontiguousarray(handle.get_tensor(name))
                observed_sha = _array_sha256(value)
                if (
                    value.shape != shape
                    or str(value.dtype) != dtype
                    or value.nbytes != record.get("byte_count")
                    or record.get("padding") is not False
                    or observed_sha != record.get("sha256")
                ):
                    raise RuntimeError(f"slot {slot} tensor contract drifted: {name}")
                by_name[name].append(value)
                slot_records.append({"name": name, "sha256": observed_sha})
        records.append(
            {
                "device_slot": slot,
                "destination_filename": relative.as_posix(),
                "evidence_file_sha256": _file_sha256(evidence_path),
                "header_sha256": header_sha,
                "tensors": slot_records,
            }
        )
    for name, (_, _, replicated) in _WEIGHT_CONTRACT.items():
        if replicated and any(
            not np.array_equal(by_name[name][0], value)
            for value in by_name[name][1:]
        ):
            raise RuntimeError(f"replicated tensor differs across slots: {name}")
    result: dict[str, np.ndarray] = {}
    for name, (_, _, replicated) in _WEIGHT_CONTRACT.items():
        result[name] = by_name[name][0] if replicated else np.stack(by_name[name])
    return result, records


_INSTRUCTION_SUFFIX_RE = re.compile(r"^(?P<kernel>[A-Za-z0-9_]+)(?:\.[0-9]+)?$")


def _pallas_calls(module: Any) -> list[tuple[Any, str | None]]:
    calls: list[tuple[Any, str | None]] = []
    for instruction in module.instructions:
        if (
            instruction.raw_opcode != "custom-call"
            or 'custom_call_target="tpu_custom_call"'
            not in instruction.raw_line
        ):
            continue
        match = _INSTRUCTION_SUFFIX_RE.fullmatch(
            instruction.name.removeprefix("%")
        )
        kernel = match.group("kernel") if match is not None else None
        op_parts = (instruction.op_name or "").split("/")
        if (
            kernel is None
            or len(op_parts) < 2
            or op_parts[-2:] != [kernel, "pallas_call"]
        ):
            kernel = None
        calls.append((instruction, kernel))
    return calls


def _shape_signatures(shapes: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(
        f"{shape.dtype}[{','.join(map(str, shape.dimensions))}]"
        for shape in shapes
    )


def _instruction_key(instruction: Any) -> tuple[str, str]:
    return instruction.computation, instruction.name


def _computation_id(header: str) -> str:
    value = header.removeprefix("ENTRY ").split(None, 1)[0]
    return value.removeprefix("%").split("(", 1)[0]


def _tuple_index(instruction: Any) -> int | None:
    match = re.search(r"\bindex=([0-9]+)", instruction.raw_line)
    return None if match is None else int(match.group(1))


def _fusion_operand_indices(
    module: Any,
    fusion: Any,
    *,
    result_index: int | None = None,
    fusion_stack: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[set[int], list[str]]:
    """Map a fusion result to only caller operands used by its called root."""

    fusion_key = _instruction_key(fusion)
    if fusion_key in fusion_stack:
        return set(), [f"recursive fusion lineage: {fusion.name}"]
    match = re.search(r"\bcalls=%?([^,\s}\]]+)", fusion.raw_line)
    if match is None:
        return set(), [f"fusion lacks called computation: {fusion.name}"]
    callee = match.group(1)
    instructions = tuple(
        item
        for item in module.instructions
        if _computation_id(item.computation) == callee
    )
    by_key = {_instruction_key(item): item for item in module.instructions}
    roots = tuple(
        item
        for item in instructions
        if item.raw_line.lstrip().startswith("ROOT ")
    )
    if len(roots) != 1:
        return set(), [f"fusion callee root drifted: {fusion.name}"]
    errors: list[str] = []
    visiting: set[tuple[str, str]] = set()

    def dependencies(instruction: Any, selected: int | None = None) -> set[int]:
        key = _instruction_key(instruction)
        if key in visiting:
            errors.append(f"cyclic fusion value: {instruction.name}")
            return set()
        visiting.add(key)
        try:
            if instruction.opcode == "parameter":
                parameter_match = re.search(
                    r"\bparameter\(([0-9]+)\)", instruction.raw_line
                )
                if parameter_match is None or selected is not None:
                    errors.append(
                        f"malformed fusion parameter: {instruction.name}"
                    )
                    return set()
                return {int(parameter_match.group(1))}
            if instruction.opcode == "get-tuple-element":
                index = _tuple_index(instruction)
                if index is None or len(instruction.operand_names) != 1:
                    errors.append(
                        f"malformed fusion tuple selection: {instruction.name}"
                    )
                    return set()
                producer = by_key.get(
                    (instruction.computation, instruction.operand_names[0])
                )
                if producer is None:
                    errors.append(
                        f"undefined fusion tuple producer: {instruction.name}"
                    )
                    return set()
                return dependencies(producer, index)
            if instruction.opcode == "tuple":
                if selected is None:
                    indices = range(len(instruction.operand_names))
                elif selected < len(instruction.operand_names):
                    indices = (selected,)
                else:
                    errors.append(f"fusion tuple index drifted: {instruction.name}")
                    return set()
                result: set[int] = set()
                for index in indices:
                    operand = by_key.get(
                        (instruction.computation, instruction.operand_names[index])
                    )
                    if operand is None:
                        errors.append(
                            f"undefined fusion tuple operand: {instruction.name}"
                        )
                    else:
                        result.update(dependencies(operand))
                return result
            if instruction.opcode == "fusion":
                nested_indices, nested_errors = _fusion_operand_indices(
                    module,
                    instruction,
                    result_index=selected,
                    fusion_stack=fusion_stack | {fusion_key},
                )
                errors.extend(nested_errors)
                result = set()
                for index in nested_indices:
                    if index >= len(instruction.operand_names):
                        errors.append(
                            f"nested fusion operand index drifted: {instruction.name}"
                        )
                        continue
                    operand = by_key.get(
                        (instruction.computation, instruction.operand_names[index])
                    )
                    if operand is None:
                        errors.append(
                            f"undefined nested fusion operand: {instruction.name}"
                        )
                    else:
                        result.update(dependencies(operand))
                return result
            if selected is not None:
                errors.append(
                    f"tuple index applied to non-tuple fusion value: "
                    f"{instruction.name}"
                )
                return set()
            result = set()
            for operand_name in instruction.operand_names:
                operand = by_key.get((instruction.computation, operand_name))
                if operand is None:
                    errors.append(
                        f"undefined fusion operand {operand_name}: "
                        f"{instruction.name}"
                    )
                else:
                    result.update(dependencies(operand))
            return result
        finally:
            visiting.remove(key)

    indices = dependencies(roots[0], result_index)
    invalid = sorted(
        index for index in indices if index >= len(fusion.operand_names)
    )
    if invalid:
        errors.append(f"fusion caller operand indices drifted: {invalid}")
        indices.difference_update(invalid)
    return indices, errors


def _value_depends_on(module: Any, value: Any, source: Any) -> bool:
    """Test exact HLO value dependency while respecting tuple element selection."""

    by_key = {_instruction_key(item): item for item in module.instructions}
    source_key = _instruction_key(source)
    visiting: set[tuple[str, str]] = set()

    def depends(key: tuple[str, str]) -> bool:
        if key == source_key:
            return True
        if key in visiting:
            return False
        instruction = by_key.get(key)
        if instruction is None:
            return False
        visiting.add(key)
        try:
            operands = instruction.operand_names
            if instruction.opcode == "get-tuple-element":
                index = _tuple_index(instruction)
                if index is None or len(operands) != 1:
                    return False
                producer = by_key.get((instruction.computation, operands[0]))
                if producer is None:
                    return False
                if producer.opcode == "tuple":
                    if index >= len(producer.operand_names):
                        return False
                    return depends(
                        (instruction.computation, producer.operand_names[index])
                    )
                if producer.opcode == "fusion":
                    indices, errors = _fusion_operand_indices(
                        module, producer, result_index=index
                    )
                    return not errors and any(
                        depends(
                            (
                                instruction.computation,
                                producer.operand_names[operand_index],
                            )
                        )
                        for operand_index in indices
                    )
                return False
            if instruction.opcode == "fusion":
                indices, errors = _fusion_operand_indices(module, instruction)
                return not errors and any(
                    depends((instruction.computation, operands[index]))
                    for index in indices
                )
            return any(
                depends((instruction.computation, operand))
                for operand in operands
            )
        finally:
            visiting.remove(key)

    return depends(_instruction_key(value))


def _bounded_pallas_sources(
    module: Any,
    start_keys: tuple[tuple[str, str], ...],
    *,
    kernel_by_key: Mapping[tuple[str, str], str | None],
) -> tuple[set[tuple[str, str]], list[str]]:
    """Trace values back through bounded layout-only transforms to Pallas calls."""

    allowed = {
        "bitcast",
        "concatenate",
        "copy",
        "dynamic-slice",
        "get-tuple-element",
        "fusion",
        "reshape",
        "slice",
    }
    by_key = {_instruction_key(item): item for item in module.instructions}
    sources: set[tuple[str, str]] = set()
    errors: list[str] = []
    visited: set[tuple[str, str]] = set()

    def visit(key: tuple[str, str]) -> None:
        if key in visited:
            return
        visited.add(key)
        instruction = by_key.get(key)
        if instruction is None:
            errors.append(f"undefined lineage operand {key[1]}")
            return
        if key in kernel_by_key:
            sources.add(key)
            return
        if instruction.opcode not in allowed:
            errors.append(
                f"forbidden lineage opcode {instruction.opcode}: "
                f"{instruction.name}"
            )
            return
        if instruction.opcode == "get-tuple-element":
            index = _tuple_index(instruction)
            if index is None or len(instruction.operand_names) != 1:
                errors.append(f"malformed tuple selection: {instruction.name}")
                return
            producer = by_key.get(
                (instruction.computation, instruction.operand_names[0])
            )
            if producer is None or len(instruction.result_shapes) != 1:
                errors.append(f"invalid tuple selection: {instruction.name}")
                return
            if producer.opcode == "tuple":
                if (
                    index >= len(producer.operand_names)
                    or index >= len(producer.operand_shapes)
                    or instruction.result_shapes[0]
                    != producer.operand_shapes[index]
                ):
                    errors.append(f"invalid tuple selection: {instruction.name}")
                    return
                visit((instruction.computation, producer.operand_names[index]))
                return
            if producer.opcode == "fusion":
                if (
                    index >= len(producer.result_shapes)
                    or instruction.result_shapes[0]
                    != producer.result_shapes[index]
                ):
                    errors.append(f"invalid fusion tuple selection: {instruction.name}")
                    return
                indices, fusion_errors = _fusion_operand_indices(
                    module, producer, result_index=index
                )
                errors.extend(fusion_errors)
                for operand_index in indices:
                    visit(
                        (
                            instruction.computation,
                            producer.operand_names[operand_index],
                        )
                    )
                return
            errors.append(f"invalid tuple producer: {instruction.name}")
            return
        if instruction.opcode == "fusion":
            indices, fusion_errors = _fusion_operand_indices(module, instruction)
            errors.extend(fusion_errors)
            for index in indices:
                visit((instruction.computation, instruction.operand_names[index]))
            return
        shapes = instruction.operand_shapes + instruction.result_shapes
        if not shapes or any(shape.dtype != "bf16" for shape in shapes):
            errors.append(f"non-BF16 lineage transform: {instruction.name}")
            return
        if not instruction.operand_names or len(instruction.result_shapes) != 1:
            errors.append(f"ambiguous lineage transform: {instruction.name}")
            return
        operand_elements = sum(
            shape.element_count for shape in instruction.operand_shapes
        )
        result_elements = instruction.result_shapes[0].element_count
        if instruction.opcode in {"bitcast", "copy", "reshape"} and (
            len(instruction.operand_shapes) != 1
            or operand_elements != result_elements
        ):
            errors.append(f"non-shape-only transform: {instruction.name}")
            return
        if instruction.opcode == "concatenate" and (
            "dimensions={0}" not in instruction.raw_line
            or operand_elements != result_elements
        ):
            errors.append(f"invalid partial concatenate: {instruction.name}")
            return
        if instruction.opcode in {"slice", "dynamic-slice", "fusion"} and (
            result_elements > operand_elements
        ):
            errors.append(f"expanding partial transform: {instruction.name}")
            return
        for operand in instruction.operand_names:
            visit((instruction.computation, operand))

    for start_key in start_keys:
        visit(start_key)
    return sources, errors


def _collective_pallas_sources(
    module: Any,
    collective: Any,
    *,
    kernel_by_key: Mapping[tuple[str, str], str | None],
) -> tuple[set[tuple[str, str]], list[str]]:
    """Trace one collective operand back through bounded layout transforms."""
    if len(collective.operand_names) != 1:
        return set(), [f"collective {collective.name} lacks one data operand"]
    return _bounded_pallas_sources(
        module,
        ((collective.computation, collective.operand_names[0]),),
        kernel_by_key=kernel_by_key,
    )


def _is_bf16_add_reducer(module: Any, collective: Any) -> bool:
    match = re.search(r"\bto_apply=%?([^,\s}\]]+)", collective.raw_line)
    if match is None:
        return False
    reducer = match.group(1)

    def computation_id(header: str) -> str:
        value = header.removeprefix("ENTRY ").split(None, 1)[0]
        return value.removeprefix("%").split("(", 1)[0]

    roots = [
        item
        for item in module.instructions
        if computation_id(item.computation) == reducer
        and item.raw_line.lstrip().startswith("ROOT ")
    ]
    return bool(
        len(roots) == 1
        and roots[0].opcode == "add"
        and _shape_signatures(roots[0].operand_shapes) == ("bf16[]", "bf16[]")
        and _shape_signatures(roots[0].result_shapes) == ("bf16[]",)
    )


def _validate_hlo(optimized_hlo: str, arm: Arm) -> dict[str, Any]:
    from glm_tpu.greenfield.sharding.hlo_contract import (
        COLLECTIVE_OPCODES,
        parse_hlo_module,
    )

    module = parse_hlo_module(optimized_hlo)
    async_collectives = [
        instruction.name
        for instruction in module.instructions
        if any(
            instruction.raw_opcode == f"{opcode}-{suffix}"
            for opcode in COLLECTIVE_OPCODES
            for suffix in ("start", "done")
        )
    ]
    collectives = list(module.collectives)
    escaped = [
        instruction.name
        for instruction in collectives
        if instruction.replica_groups != ((0, 1, 2, 3),)
    ]
    calls = _pallas_calls(module)
    required: dict[str, int] = {
        "greenfield_fp8_structured_kv_b_value_h16_l512_v256": 1,
        (
            "greenfield_fp8_block_matmul_m8_k512_n6144"
            if arm.attention_strategy_nd
            else "greenfield_fp8_block_matmul_m8_k4096_n6144"
        ): 8 if arm.attention_strategy_nd else 1,
        (
            "greenfield_fp8_fused_block_swiglu_m8_h6144_i384_o6144"
            if arm.dense_strategy_nd
            else "greenfield_fp8_fused_block_swiglu_m8_h6144_i3072_o6144"
        ): 8 if arm.dense_strategy_nd else 1,
    }
    counts = Counter(kernel for _, kernel in calls)
    violations = [
        f"expected {count} {name} calls, found {counts.get(name, 0)}"
        for name, count in required.items()
        if counts.get(name, 0) != count
    ]
    if module.num_partitions != 4 or module.num_replicas not in (None, 1):
        violations.append(
            "probe module cardinality drifted: "
            f"partitions={module.num_partitions} replicas={module.num_replicas}"
        )
    unexpected_calls = [
        instruction.name
        for instruction, kernel in calls
        if kernel not in required
    ]
    if unexpected_calls:
        violations.append(f"unexpected TPU custom calls: {unexpected_calls}")
    if async_collectives:
        violations.append(f"async collectives are forbidden: {async_collectives}")
    if len(collectives) != 2:
        violations.append(
            f"expected 2 local collectives, found {len(collectives)}"
        )
    if escaped:
        violations.append(f"collective escaped LP4: {escaped}")
    expected_collective_scopes = {
        (
            "greenfield_strategy_nd_row0_attention_output"
            if arm.attention_strategy_nd
            else "greenfield_local_attention_output_reduction"
        ): "all-gather" if arm.attention_strategy_nd else "all-reduce",
        (
            "greenfield_strategy_nd_row0_dense_down"
            if arm.dense_strategy_nd
            else "greenfield_local_dense_down_reduction"
        ): "all-gather" if arm.dense_strategy_nd else "all-reduce",
    }
    observed_scopes: dict[str, list[str]] = {
        scope: [] for scope in expected_collective_scopes
    }
    collective_by_scope: dict[str, list[Any]] = {
        scope: [] for scope in expected_collective_scopes
    }
    for instruction in collectives:
        matches = [
            scope
            for scope in expected_collective_scopes
            if scope in (instruction.op_name or "").split("/")
        ]
        if len(matches) != 1:
            violations.append(
                f"collective {instruction.name} lacks one exact diagnostic scope"
            )
            continue
        observed_scopes[matches[0]].append(instruction.opcode)
        collective_by_scope[matches[0]].append(instruction)
    for scope, expected_opcode in expected_collective_scopes.items():
        if observed_scopes[scope] != [expected_opcode]:
            violations.append(
                f"scope {scope} expected one {expected_opcode}, "
                f"found {observed_scopes[scope]}"
            )
    for scope, expected_opcode in expected_collective_scopes.items():
        scoped = collective_by_scope[scope]
        if len(scoped) != 1:
            continue
        instruction = scoped[0]
        if not instruction.use_global_device_ids:
            violations.append(f"scope {scope} lacks global device ids")
        if expected_opcode == "all-reduce":
            if (
                _shape_signatures(instruction.operand_shapes)
                != ("bf16[1,6144]",)
                or _shape_signatures(instruction.result_shapes)
                != ("bf16[1,6144]",)
            ):
                violations.append(f"scope {scope} local reduction shape drifted")
            if not _is_bf16_add_reducer(module, instruction):
                violations.append(f"scope {scope} lacks a BF16 add reducer")
        else:
            result_shapes = _shape_signatures(instruction.result_shapes)
            if (
                _shape_signatures(instruction.operand_shapes)
                != ("bf16[8,1,6144]",)
                or result_shapes
                not in {
                    ("bf16[4,8,1,6144]",),
                    ("bf16[32,1,6144]",),
                }
                or "dimensions={0}" not in instruction.raw_line
            ):
                violations.append(f"scope {scope} StrategyND gather shape drifted")

    geometry = {
        "greenfield_fp8_structured_kv_b_value_h16_l512_v256": {
            "operands": {
                ("bf16[1,16,512]", "u8[7168,512]", "f32[56,4]"),
                ("bf16[16,8,512]", "u8[7168,512]", "f32[192,8,128]"),
            },
            "results": {
                ("bf16[1,16,256]",),
                ("bf16[48,8,128]",),
            },
        },
        "greenfield_fp8_block_matmul_m8_k4096_n6144": {
            "operands": {
                ("bf16[1,4096]", "u8[6144,4096]", "f32[48,32]"),
                ("bf16[8,4096]", "u8[6144,4096]", "f32[32,128]"),
            },
            "results": {("bf16[1,6144]",), ("bf16[8,6144]",)},
        },
        "greenfield_fp8_block_matmul_m8_k512_n6144": {
            "operands": {
                ("bf16[1,512]", "u8[6144,512]", "f32[48,4]"),
                ("bf16[8,512]", "u8[6144,512]", "f32[8,128]"),
            },
            "results": {("bf16[1,6144]",), ("bf16[8,6144]",)},
        },
        "greenfield_fp8_fused_block_swiglu_m8_h6144_i3072_o6144": {
            "operands": {
                (
                    "bf16[1,6144]", "u8[3072,6144]", "f32[24,48]",
                    "u8[3072,6144]", "f32[24,48]", "u8[6144,3072]",
                    "f32[48,24]",
                ),
                (
                    "bf16[8,6144]", "u8[3072,6144]", "f32[48,128]",
                    "u8[3072,6144]", "f32[48,128]", "u8[6144,3072]",
                    "f32[24,128]",
                ),
            },
            "results": {("bf16[1,6144]",), ("bf16[8,6144]",)},
        },
        "greenfield_fp8_fused_block_swiglu_m8_h6144_i384_o6144": {
            "operands": {
                (
                    "bf16[1,6144]", "u8[384,6144]", "f32[3,48]",
                    "u8[384,6144]", "f32[3,48]", "u8[6144,384]",
                    "f32[48,3]",
                ),
                (
                    "bf16[8,6144]", "u8[384,6144]", "f32[48,128]",
                    "u8[384,6144]", "f32[48,128]", "u8[6144,384]",
                    "f32[8,128]",
                ),
            },
            "results": {("bf16[1,6144]",), ("bf16[8,6144]",)},
        },
    }
    kernel_by_key = {
        _instruction_key(instruction): kernel for instruction, kernel in calls
    }
    calls_by_kernel: dict[str, list[Any]] = {name: [] for name in required}
    for instruction, kernel in calls:
        if kernel in calls_by_kernel:
            calls_by_kernel[kernel].append(instruction)
            expected = geometry[kernel]
            if (
                _shape_signatures(instruction.operand_shapes)
                not in expected["operands"]
                or _shape_signatures(instruction.result_shapes)
                not in expected["results"]
            ):
                violations.append(
                    f"Pallas geometry drifted: {instruction.name}: "
                    f"operands={_shape_signatures(instruction.operand_shapes)} "
                    f"results={_shape_signatures(instruction.result_shapes)}"
                )

    attention_scope, dense_scope = tuple(expected_collective_scopes)
    attention_kernel = (
        "greenfield_fp8_block_matmul_m8_k512_n6144"
        if arm.attention_strategy_nd
        else "greenfield_fp8_block_matmul_m8_k4096_n6144"
    )
    dense_kernel = (
        "greenfield_fp8_fused_block_swiglu_m8_h6144_i384_o6144"
        if arm.dense_strategy_nd
        else "greenfield_fp8_fused_block_swiglu_m8_h6144_i3072_o6144"
    )
    lineage: dict[str, list[str]] = {}
    for scope, kernel in (
        (attention_scope, attention_kernel),
        (dense_scope, dense_kernel),
    ):
        scoped = collective_by_scope[scope]
        if len(scoped) != 1:
            continue
        sources, errors = _collective_pallas_sources(
            module, scoped[0], kernel_by_key=kernel_by_key
        )
        expected_sources = {
            _instruction_key(item) for item in calls_by_kernel[kernel]
        }
        lineage[scope] = sorted(name for _, name in sources)
        if errors:
            violations.extend(f"scope {scope}: {error}" for error in errors)
        if sources != expected_sources:
            violations.append(
                f"scope {scope} Pallas lineage drifted: "
                f"expected={sorted(name for _, name in expected_sources)} "
                f"observed={lineage[scope]}"
            )

    value_calls = calls_by_kernel[
        "greenfield_fp8_structured_kv_b_value_h16_l512_v256"
    ]
    if len(value_calls) == 1:
        expected_value_source = {_instruction_key(value_calls[0])}
        for instruction in calls_by_kernel[attention_kernel]:
            if not instruction.operand_names:
                violations.append(
                    f"W_UV does not feed attention projection {instruction.name}"
                )
                continue
            sources, errors = _bounded_pallas_sources(
                module,
                ((instruction.computation, instruction.operand_names[0]),),
                kernel_by_key=kernel_by_key,
            )
            if errors:
                violations.extend(
                    f"attention activation {instruction.name}: {error}"
                    for error in errors
                )
            if sources != expected_value_source:
                violations.append(
                    f"W_UV does not exclusively feed attention projection "
                    f"{instruction.name}: observed="
                    f"{sorted(name for _, name in sources)}"
                )
    if len(collective_by_scope[attention_scope]) == 1:
        attention_collective = collective_by_scope[attention_scope][0]
        for instruction in calls_by_kernel[dense_kernel]:
            activation = (
                next(
                    (
                        item
                        for item in module.instructions
                        if _instruction_key(item)
                        == (
                            instruction.computation,
                            instruction.operand_names[0],
                        )
                    ),
                    None,
                )
                if instruction.operand_names
                else None
            )
            if (
                activation is None
                or not _value_depends_on(
                    module,
                    activation,
                    attention_collective,
                )
            ):
                violations.append(
                    f"attention reduction does not feed dense {instruction.name}"
                )

    entry_roots = [
        item
        for item in module.instructions
        if item.computation.startswith("ENTRY ")
        and item.raw_line.lstrip().startswith("ROOT ")
    ]
    expected_root_shapes = (
        "bf16[1,16,256]",
        "bf16[1,6144]",
        "bf16[1,6144]",
        "bf16[1,6144]",
        "bf16[1,6144]",
    )
    if (
        len(entry_roots) != 1
        or _shape_signatures(entry_roots[0].result_shapes)
        != expected_root_shapes
    ):
        violations.append("probe entry result geometry drifted")
    elif any(
        not _value_depends_on(module, entry_roots[0], instruction)
        for instruction in (
            *value_calls,
            *collective_by_scope[attention_scope],
            *collective_by_scope[dense_scope],
        )
    ):
        violations.append("probe entry result lacks required live dataflow")
    forbidden = [
        marker
        for marker in (
            "host_callback",
            "xla_python_cpu_callback",
            " outfeed(",
        )
        if marker in optimized_hlo
    ]
    if forbidden:
        violations.append(f"forbidden HLO markers: {forbidden}")
    return {
        "collective_count": len(collectives),
        "collective_groups": [
            list(map(list, item.replica_groups)) for item in collectives
        ],
        "escaped_collectives": escaped,
        "async_collectives": async_collectives,
        "expected_collective_scopes": expected_collective_scopes,
        "observed_collective_scopes": observed_scopes,
        "expected_pallas_call_counts": required,
        "observed_pallas_call_counts": {
            name: counts.get(name, 0) for name in required
        },
        "pallas_custom_call_count": len(calls),
        "unexpected_pallas_custom_calls": unexpected_calls,
        "pallas_collective_lineage": lineage,
        "num_partitions": module.num_partitions,
        "num_replicas": module.num_replicas,
        "passed": not violations,
        "violations": violations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--attention-arithmetic", type=Path, required=True)
    parser.add_argument("--attention-arithmetic-sha256", required=True)
    parser.add_argument("--attention-runner", type=Path, required=True)
    parser.add_argument("--attention-runner-sha256", required=True)
    parser.add_argument("--accepted-projection", type=Path, required=True)
    parser.add_argument("--accepted-projection-sha256", required=True)
    parser.add_argument("--accepted-projection-capture", type=Path, required=True)
    parser.add_argument("--accepted-projection-capture-sha256", required=True)
    parser.add_argument("--ingredients", type=Path, required=True)
    parser.add_argument("--ingredients-sha256", required=True)
    parser.add_argument("--ingredients-contract", type=Path, required=True)
    parser.add_argument("--ingredients-contract-sha256", required=True)
    parser.add_argument("--layer1-reference", type=Path, required=True)
    parser.add_argument("--layer1-reference-sha256", required=True)
    parser.add_argument("--association-analysis", type=Path, required=True)
    parser.add_argument("--association-analysis-sha256", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-output", type=Path, required=True)
    parser.add_argument("--hlo-dir", type=Path, required=True)
    return parser.parse_args()


def _load_runtime_symbols() -> tuple[Any, ...]:
    """Resolve every deferred JAX/runtime dependency before loading the TPU."""

    import jax
    from jax import lax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from glm_tpu.greenfield.kernels.pallas import (
        Fp8BlockMatmulConfig,
        fp8_structured_kv_b_value,
    )
    from glm_tpu.greenfield.kernels.reference.rmsnorm import fused_add_rms_norm
    from glm_tpu.greenfield.kernels.stage_local import (
        STRATEGY_ND_ROW0_REDUCTION_ASSOCIATION,
        _STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE,
        _reduce_virtual_tp32_bf16_partials,
        _stage_fp8_linear,
        _virtual_attention_output_partials,
        stage_local_dense_fp8_mapped,
    )

    return (
        jax,
        lax,
        jnp,
        Mesh,
        NamedSharding,
        P,
        Fp8BlockMatmulConfig,
        fp8_structured_kv_b_value,
        fused_add_rms_norm,
        STRATEGY_ND_ROW0_REDUCTION_ASSOCIATION,
        _STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE,
        _reduce_virtual_tp32_bf16_partials,
        _stage_fp8_linear,
        _virtual_attention_output_partials,
        stage_local_dense_fp8_mapped,
    )


def main() -> int:
    args = parse_args()
    if REPO != Path("/home/gianl/glm-tpu-topology-rewrite"):
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    code_hash = _git_head()
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale code hash: expected={args.expected_code_hash} found={code_hash}"
        )
    latent, accepted_value_bits, residual, accepted_layer1_bits = _load_sources(args)
    association = _load_association_analysis(
        args.association_analysis,
        expected_sha256=args.association_analysis_sha256,
    )
    weights, weight_records = _load_weights(
        args.checkpoint_root,
        manifest_sha256=args.checkpoint_manifest_sha256,
    )

    (
        jax,
        lax,
        jnp,
        Mesh,
        NamedSharding,
        P,
        Fp8BlockMatmulConfig,
        fp8_structured_kv_b_value,
        fused_add_rms_norm,
        STRATEGY_ND_ROW0_REDUCTION_ASSOCIATION,
        _STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE,
        _reduce_virtual_tp32_bf16_partials,
        _stage_fp8_linear,
        _virtual_attention_output_partials,
        stage_local_dense_fp8_mapped,
    ) = _load_runtime_symbols()

    inverse_model_axis = [0] * 32
    for model_position, physical_device in enumerate(
        association["accepted_model_axis_device_ids"]
    ):
        inverse_model_axis[physical_device] = model_position
    if tuple(inverse_model_axis) != _STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE:
        raise RuntimeError("production StrategyND association differs from DB533")

    if jax.default_backend() != "tpu" or jax.local_device_count() != 4:
        raise RuntimeError("projection probe requires one four-chip TPU host")
    mesh = Mesh(np.asarray(jax.local_devices()), ("lp4",))
    replicated = NamedSharding(mesh, P())
    slot_three = NamedSharding(mesh, P("lp4", None, None))
    arguments = (
        jax.device_put(latent, replicated),
        jax.device_put(residual, replicated),
        jax.device_put(weights["attention.slot_00.kv_b.weight_bits"], slot_three),
        jax.device_put(weights["attention.slot_00.kv_b.scale_inv"], slot_three),
        jax.device_put(weights["attention.slot_00.o.weight_bits"], slot_three),
        jax.device_put(weights["attention.slot_00.o.scale_inv"], slot_three),
        jax.device_put(weights["attention.slot_00.post_norm"], replicated),
        jax.device_put(weights["dense.slot_00.gate.weight_bits"], slot_three),
        jax.device_put(weights["dense.slot_00.gate.scale_inv"], slot_three),
        jax.device_put(weights["dense.slot_00.up.weight_bits"], slot_three),
        jax.device_put(weights["dense.slot_00.up.scale_inv"], slot_three),
        jax.device_put(weights["dense.slot_00.down.weight_bits"], slot_three),
        jax.device_put(weights["dense.slot_00.down.scale_inv"], slot_three),
        jax.device_put(weights["attention.slot_01.input_norm"], replicated),
    )
    in_specs = (
        P(), P(), P("lp4", None, None), P("lp4", None, None),
        P("lp4", None, None), P("lp4", None, None), P(),
        P("lp4", None, None), P("lp4", None, None),
        P("lp4", None, None), P("lp4", None, None),
        P("lp4", None, None), P("lp4", None, None), P(),
    )
    fp8_config = Fp8BlockMatmulConfig(
        block_shape=(128, 128), output_tile=128, contraction_tile=128
    )
    groups = ((0, 1, 2, 3),)

    def make_local(arm: Arm) -> Any:
        def local(
            attended: Any,
            combined_residual: Any,
            kv_bits_slot: Any,
            kv_scale_slot: Any,
            o_bits_slot: Any,
            o_scale_slot: Any,
            post_norm: Any,
            gate_bits_slot: Any,
            gate_scale_slot: Any,
            up_bits_slot: Any,
            up_scale_slot: Any,
            down_bits_slot: Any,
            down_scale_slot: Any,
            layer1_norm: Any,
        ) -> tuple[Any, Any, Any, Any, Any]:
            slot = lax.axis_index("lp4")
            local_attended = lax.dynamic_slice_in_dim(
                attended, slot * jnp.int32(16), 16, axis=0
            )[None, ...]
            kv_bits = kv_bits_slot[0]
            kv_scale = kv_scale_slot[0]
            o_bits = o_bits_slot[0]
            o_scale = o_scale_slot[0]
            value_states = fp8_structured_kv_b_value(
                local_attended,
                kv_bits,
                kv_scale,
                qk_nope_head_dim=192,
                config=fp8_config,
            )
            output_input = value_states.reshape(1, 4096)
            if arm.attention_strategy_nd:
                with jax.named_scope("greenfield_strategy_nd_row0_attention_output"):
                    attention_update = _reduce_virtual_tp32_bf16_partials(
                        _virtual_attention_output_partials(
                            output_input,
                            o_bits,
                            o_scale,
                            block_shape=(128, 128),
                            linear_interpret=False,
                        ),
                        axis_name="lp4",
                        groups=groups,
                        association=STRATEGY_ND_ROW0_REDUCTION_ASSOCIATION,
                    )
            else:
                with jax.named_scope(
                    "greenfield_local_attention_output_reduction"
                ):
                    attention_update = lax.psum(
                        _stage_fp8_linear(
                            output_input,
                            o_bits,
                            o_scale,
                            block_shape=(128, 128),
                            backend="pallas",
                            interpret=False,
                        ),
                        "lp4",
                        axis_index_groups=groups,
                    )
            normalized_mlp, post_attention_residual = fused_add_rms_norm(
                attention_update,
                combined_residual,
                post_norm,
                epsilon=1e-5,
            )
            dense_scope = (
                "greenfield_strategy_nd_row0_dense_down"
                if arm.dense_strategy_nd
                else "greenfield_local_dense_down_reduction"
            )
            with jax.named_scope(dense_scope):
                dense_update = stage_local_dense_fp8_mapped(
                    post_attention_residual,
                    post_norm,
                    gate_bits_slot[0],
                    gate_scale_slot[0],
                    up_bits_slot[0],
                    up_scale_slot[0],
                    down_bits_slot[0],
                    down_scale_slot[0],
                    axis_name="lp4",
                    axis_index_groups=groups,
                    precomputed_normalized=normalized_mlp,
                    add_residual=False,
                    linear_backend="pallas",
                    virtual_tp32_reduction_association=(
                        STRATEGY_ND_ROW0_REDUCTION_ASSOCIATION
                        if arm.dense_strategy_nd
                        else None
                    ),
                )
            layer1_normalized = fused_add_rms_norm(
                dense_update,
                post_attention_residual,
                layer1_norm,
                epsilon=1e-5,
            )[0]
            return (
                value_states,
                attention_update,
                normalized_mlp,
                dense_update,
                layer1_normalized,
            )

        return local

    args.hlo_dir.mkdir(parents=True, exist_ok=True)
    arm_records: dict[str, Any] = {}
    tensor_payload: dict[str, np.ndarray] = {
        "accepted_attended_latent_bfloat16_bits": latent.view(np.uint16),
        "accepted_attention_value_bfloat16_bits": accepted_value_bits,
        "accepted_layer1_normalized_bfloat16_bits": accepted_layer1_bits,
        "combined_residual_bfloat16_bits": residual.view(np.uint16),
    }
    exact_arms = []
    for arm in _ARMS:
        mapped = jax.shard_map(
            make_local(arm),
            mesh=mesh,
            in_specs=in_specs,
            out_specs=(
                P("lp4", None, None), P(), P(), P(), P(),
            ),
            check_vma=False,
        )
        lowered = jax.jit(mapped).lower(*arguments)
        stablehlo = lowered.as_text()
        compiled = lowered.compile()
        optimized_hlo = compiled.as_text()
        (args.hlo_dir / f"{arm.name}.stablehlo.mlir").write_text(stablehlo)
        (args.hlo_dir / f"{arm.name}.optimized_hlo.txt").write_text(
            optimized_hlo
        )
        hlo_contract = _validate_hlo(optimized_hlo, arm)
        if not hlo_contract["passed"]:
            raise RuntimeError(
                f"projection/reduction HLO failed: {arm.name}: {hlo_contract}"
            )
        (value, attention_update, normalized_mlp, dense_update, layer1) = compiled(
            *arguments
        )
        jax.block_until_ready(
            (value, attention_update, normalized_mlp, dense_update, layer1)
        )
        value_bits = np.ascontiguousarray(np.asarray(value).reshape(16384)).view(
            np.uint16
        )
        layer1_bits = np.ascontiguousarray(np.asarray(layer1).reshape(6144)).view(
            np.uint16
        )
        value_comparison = _compare_bits(accepted_value_bits, value_bits)
        if not value_comparison["elementwise_exact"]:
            raise RuntimeError(f"post-W_UV prerequisite is nonexact: {arm.name}")
        layer1_comparison = _compare_bits(accepted_layer1_bits, layer1_bits)
        if layer1_comparison["elementwise_exact"]:
            exact_arms.append(arm.name)
        arm_records[arm.name] = {
            "attention_strategy_nd": arm.attention_strategy_nd,
            "dense_strategy_nd": arm.dense_strategy_nd,
            "hlo": {
                "contract": hlo_contract,
                "optimized_sha256": sha256(optimized_hlo.encode()).hexdigest(),
                "stablehlo_sha256": sha256(stablehlo.encode()).hexdigest(),
            },
            "layer1_comparison": layer1_comparison,
            "value_comparison": value_comparison,
        }
        tensor_payload[f"attention_value_bfloat16_bits__{arm.name}"] = value_bits
        tensor_payload[f"attention_update_bfloat16_bits__{arm.name}"] = (
            np.ascontiguousarray(np.asarray(attention_update)).view(np.uint16)
        )
        tensor_payload[f"normalized_mlp_bfloat16_bits__{arm.name}"] = (
            np.ascontiguousarray(np.asarray(normalized_mlp)).view(np.uint16)
        )
        tensor_payload[f"dense_update_bfloat16_bits__{arm.name}"] = (
            np.ascontiguousarray(np.asarray(dense_update)).view(np.uint16)
        )
        tensor_payload[f"layer1_normalized_bfloat16_bits__{arm.name}"] = layer1_bits

    classification = (
        "exact_projection_reduction_arm_identified"
        if exact_arms
        else "projection_reduction_unresolved"
    )
    result = {
        "artifact_kind": "glm52_layer0_projection_reduction_probe",
        "arms": arm_records,
        "classification": classification,
        "code_hash": code_hash,
        "exact_arms": exact_arms,
        "performance_claim": False,
        "position": _POSITION,
        "source": {
            "attention_arithmetic_sha256": args.attention_arithmetic_sha256,
            "attention_runner_sha256": args.attention_runner_sha256,
            "accepted_projection_sha256": args.accepted_projection_sha256,
            "accepted_projection_capture_sha256": (
                args.accepted_projection_capture_sha256
            ),
            "association_analysis_sha256": args.association_analysis_sha256,
            "ingredients_sha256": args.ingredients_sha256,
            "layer1_reference_sha256": args.layer1_reference_sha256,
        },
        "status": "SUCCESS",
        "strategy_nd_evidence": association,
        "weight_records": weight_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.tensor_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.tensor_output, **tensor_payload)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
