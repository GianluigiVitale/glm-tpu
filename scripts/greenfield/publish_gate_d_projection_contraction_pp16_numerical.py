#!/usr/bin/env python3
"""Append-only publisher for the bounded PP16 host-rope projection numerical replay (V3)."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import stat
import struct
import subprocess
import sys
import types
import zipfile
from collections.abc import Mapping
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

REPO = Path("/home/gianl/glm-tpu-gate-d-pp16-numerical")
RUN_ROOT = Path("/home/gianl/gate-d-runs")
SOURCE_PATH = (
    "scripts/greenfield/publish_gate_d_projection_contraction_pp16_numerical.py"
)
BASE_PATH = "scripts/greenfield/publish_gate_d_projection_contraction_pp16_hlo.py"
BASE_PIN = "986378238ac6458307aea69ef1f5e12bf82bc020"
BASE_SHA256 = "f3f20a01fd37bb82988cd77f69fa7b0a780d120568bab0db4162f42e7855bc97"
BUCKET_NAME = "driftbench-dsv4-uc"
REMOTE_ROOT = "results/greenfield/glm52/gate_d_projection_contraction_pp16_numerical/"
TAG_PATTERN_TEXT = r"gate_d_projection_contraction_pp16_numerical_[0-9]{8}T[0-9]{15}Z"
TAG_PATTERN = re.compile(TAG_PATTERN_TEXT)
# V3 binds HLO structure (audited by the immutable driver from the sealed
# validation module) and re-checks the archived texts here without a parser.
HLO_MODULE_NAME = "jit__projection_host_rope_local"
HLO_FORBIDDEN_TEXT = (
    " cosine(",
    " sine(",
    " power(",
    "all-gather",
    "all-reduce",
    "all-to-all",
    "collective-permute",
    "reduce-scatter",
    "custom-call",
    "infeed",
    "outfeed",
    "host_callback",
    "outside_compilation",
)
STABLEHLO_FORBIDDEN_TEXT = (
    "stablehlo.cosine",
    "stablehlo.sine",
    "stablehlo.power",
    "stablehlo.all_",
    "stablehlo.collective_",
    "stablehlo.custom_call",
    "stablehlo.infeed",
    "stablehlo.outfeed",
)
STABLEHLO_REQUIRED_TEXT = (
    "mhlo.num_partitions = 2",
    'sdy.mesh @mesh = <["feature"=2]>',
    "tensor<2x1x6144xbf16>",
    "tensor<2x128x6144xf32>",
    "tensor<2x64xf32>",
    "stablehlo.dot_general",
)
CAPSULE_INPUTS_PATH = Path(
    "/home/gianl/gate-d-runs/greenfield_gate_d_compensated_capsule_20260831T124838Z/"
    "candidate-inputs.npz"
)
DSA_ROPE_POSITION = 8155
DSA_ROTARY_DIM = 64
DSA_ROTARY_THETA = 8_000_000.0
EXPECTED_DSA_ROPE_ROW_SHA256 = (
    "748aa6122b8d83cfcf65928d33d1ac10392b7cc968617f324a75a62168d3a9c0"
)
PROJECTION_TOLERANCE = 1e-6
KEY_TOLERANCE = 1e-6
IMPLIED_COS_SIN_TOLERANCE = 1e-6
EXPECTED_NORMALIZED_OWNER_SHA256 = (
    "28b7db466b74f462b5cd3109cb052bcce0c8be185d3f24199239f75136b22c20"
)
# Exact committed sources of the host-rope graph and its validation, as pinned
# by the immutable driver; the runner report must carry the same identities.
HOST_ROPE_SOURCE_SHA256S = {
    "glm_tpu/greenfield/benchmarking/gate_d_projection_contraction_pp16_host_rope.py": (
        "29f8b72340f316b04504e21e5aa4554ce8836ad5592494753a880fababb739bf"
    ),
    "glm_tpu/greenfield/kernels/reference/dsa_host_rope.py": (
        "74ead92ca1f73cfeac2b58f95ab9ba08e2fae50a0d1bbad3b4b66eda42e615d4"
    ),
    "glm_tpu/greenfield/kernels/reference/rotary_table.py": (
        "82fce9e3e1bc64aa8d1c48fcfe2041a34887922bd329e90aeb3d477f401d7e58"
    ),
    "glm_tpu/greenfield/validation/gate_d_projection_host_rope_numerical.py": (
        "e8640f930913b4bb598bbc4fbfd61eb651a691b7a8a674f03bcd82a60c24bb5b"
    ),
}
EXPECTED_WK_WEIGHT_FP32_SHA256 = (
    "b2c67e0fdf4d7292494778233e9813d8256f2fb88b6f7376870379832fbab24b"
)
EXPECTED_OUTPUT_LAYOUT = {
    "current_key_owners": ((2, 1, 128), "<f4", 4),
    "normalized_hidden_owners": ((2, 1, 6144), "<u2", 2),
    "projected_key_owners": ((2, 1, 128), "<f4", 4),
}
EXPECTED_PREDECESSORS = {
    "capsule_input_sha256": "dd5f1cbb37b722591531635a21cfa9f1d6d0157997a4f70138900d0000be8b1b",
    "capsule_sha256": "5b7ad71f37dbbcda0ee36a9fc0c42a7ca45d619a9e741307386755e68e87c1a4",
    "capsule_state_sha256": "68ee47b1fcbf317fd41da51aa26c9ba1a8dafe0e3e95ccbbfa73285f4e46f236",
    "execution_authority_sha256": "8b8c9cc79a18679628582a66c418a7f63e06553091928df58defbdde3971a660",
    "frontier_authority_sha256": "4a6be0f33f221df46f384cd2047da1aa664ae4c45763e68f0f1df05f8644e20c",
    "hlo_success_authority_sha256": "54eb6105b81d9ffdbdd3b90735059fb324701509fe8efd003033bb3cff7e0ad7",
    "host_materialization_authority_sha256": "a7e5b393f7c61181f6b2aab9531d788fb27594d9cf980055c165b6f278ba4f99",
}
EXPECTED_PHYSICAL_GROUP = {
    "coordinates": [[0, 0, 0], [1, 0, 0]],
    "device_ids": [0, 1],
    "local_device_count_visible": 4,
    "mesh_device_count": 2,
    "process_index": 0,
    "stage_id": 0,
}
EXPECTED_COMPILER_ENVIRONMENT = {
    "GLM_GATE_D_PROJECTION_CONTRACTION_NUMERICAL": "1",
    "HOME": "/home/gianl",
    "JAX_ENABLE_COMPILATION_CACHE": "0",
    "JAX_PLATFORMS": "tpu",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TPU_CHIPS_PER_PROCESS_BOUNDS": "2,2,1",
    "TPU_PROCESS_BOUNDS": "1,1,1",
    "TPU_VISIBLE_DEVICES": "0,1,2,3",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": ".50",
}
SUCCESS_PAYLOAD = (
    "census_post.txt",
    "census_pre.txt",
    "dependencies.json",
    "dsa_rope_row.f32le",
    "evidence.json",
    "hlo/projection_contraction_pp16_stage0.optimized_hlo.txt",
    "hlo/projection_contraction_pp16_stage0.stablehlo.mlir",
    "mirror.sha256",
    "orchestrator.sealed.log",
    "outputs.npz",
    "publisher_runtime.json",
    "remote_vacancy.raw.txt",
    "remote_vacancy.txt",
    "runner.json",
    "runner.log",
    "summary.json",
    "sync.txt",
)
WRAPPER_WRITE_MEMBERS = {
    "census_failure_exit.txt",
    "census_post.txt",
    "census_pre.txt",
    "mirror.sha256",
    "remote_vacancy.raw.txt",
    "remote_vacancy.txt",
    "runner.log",
    "sync.txt",
}
_GIT_ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_PROTOCOL_FROM_USER": "0",
    "GIT_SSH_COMMAND": "/bin/false",
    "GIT_TERMINAL_PROMPT": "0",
    "HOME": "/nonexistent",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
}


def _git_bytes(*arguments: str) -> bytes:
    return subprocess.check_output(
        ["/usr/bin/git", "-C", str(REPO), *arguments], env=_GIT_ENVIRONMENT
    )


def _snapshot(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(f"unsafe publisher source: {path}")
        chunks = []
        while block := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(block)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (
            len(raw) != before.st_size
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise RuntimeError(f"publisher source changed while reading: {path}")
        return raw
    finally:
        os.close(descriptor)


def _verify_running_source(code_pin: str, expected_sha256: str) -> None:
    if _git_bytes("for-each-ref", "--format=%(refname)", "refs/replace"):
        raise RuntimeError("projection numerical publisher found replacement refs")
    raw = _snapshot(Path(__file__))
    if (
        _git_bytes("rev-parse", "HEAD").decode().strip() != code_pin
        or _git_bytes("rev-parse", f"{code_pin}^{{commit}}").decode().strip()
        != code_pin
        or raw != _git_bytes("show", f"{code_pin}:{SOURCE_PATH}")
        or sha256(raw).hexdigest() != expected_sha256
    ):
        raise RuntimeError("projection numerical publisher is not the committed blob")


def _load_base(code_pin: str) -> types.ModuleType:
    path = REPO / BASE_PATH
    raw = _snapshot(path)
    if (
        raw != _git_bytes("show", f"{code_pin}:{BASE_PATH}")
        or raw != _git_bytes("show", f"{BASE_PIN}:{BASE_PATH}")
        or sha256(raw).hexdigest() != BASE_SHA256
    ):
        raise RuntimeError("projection numerical publisher base bytes drifted")
    module = types.ModuleType("_gate_d_projection_publisher_base")
    module.__file__ = str(path)
    module.__package__ = None
    exec(compile(raw, str(path), "exec"), module.__dict__)  # noqa: S102
    module.REPO = REPO
    module.RUN_ROOT = RUN_ROOT
    module.BUCKET_NAME = BUCKET_NAME
    module.REMOTE_ROOT = REMOTE_ROOT
    module.TAG_PATTERN = TAG_PATTERN
    module._EXPECTED_COMPILER_ENVIRONMENT = dict(EXPECTED_COMPILER_ENVIRONMENT)
    module._git_bytes = lambda *args: _git_bytes(*args)
    return module


def _publication_runtime(base: Any) -> dict[str, Any]:
    result = base.validate_publication_runtime()
    result["artifact_kind"] = (
        "gate_d_projection_contraction_pp16_numerical_publisher_runtime"
    )
    return result


def _require_preterminal(base: Any, run_fd: int) -> None:
    base.require_preterminal(run_fd)
    for name in ("NUMERICAL_RESULT", "terminal_upload_receipt.json"):
        try:
            os.stat(name, dir_fd=run_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        raise RuntimeError("projection numerical terminal publication already began")


def _expected_status(runner: Mapping[str, Any]) -> tuple[bool, str, str]:
    numerical = runner.get("numerical")
    accepted = (
        numerical.get("accepted_tpu_host_rope_faithful")
        if isinstance(numerical, Mapping)
        else None
    )
    if type(accepted) is not bool:
        raise RuntimeError("projection numerical classification is absent")
    status = "NUMERICAL_ACCEPTED" if accepted else "NUMERICAL_REJECTED"
    classification = (
        "BOUNDED_TPU_HOST_ROPE_KEY_FAITHFUL;ROTARY_ROOT_CAUSE_FIX_BOUNDED_PROOF;"
        "DECODER_UNPROVEN;GATE_D_OPEN"
        if accepted
        else "BOUNDED_TPU_HOST_ROPE_KEY_UNFAITHFUL;ROTARY_FIX_REJECTED;GATE_D_OPEN"
    )
    return accepted, status, classification


def _parse_npy(raw: bytes) -> tuple[tuple[int, ...], str, bytes]:
    if not raw.startswith(b"\x93NUMPY") or len(raw) < 10:
        raise RuntimeError("projection numerical NPY header is invalid")
    major, minor = raw[6], raw[7]
    if (major, minor) == (1, 0):
        header_size = int.from_bytes(raw[8:10], "little")
        header_start = 10
    elif major in (2, 3) and minor == 0:
        if len(raw) < 12:
            raise RuntimeError("projection numerical NPY v2 header is truncated")
        header_size = int.from_bytes(raw[8:12], "little")
        header_start = 12
    else:
        raise RuntimeError("projection numerical NPY version is unsupported")
    header_end = header_start + header_size
    if header_end > len(raw) or header_size > 1 << 20:
        raise RuntimeError("projection numerical NPY header size is unsafe")
    try:
        header = ast.literal_eval(raw[header_start:header_end].decode("latin1").strip())
    except (SyntaxError, ValueError, UnicodeDecodeError) as error:
        raise RuntimeError("projection numerical NPY header is malformed") from error
    if (
        not isinstance(header, dict)
        or set(header) != {"descr", "fortran_order", "shape"}
        or header.get("fortran_order") is not False
        or not isinstance(header.get("shape"), tuple)
        or not all(type(item) is int and item >= 0 for item in header["shape"])
        or not isinstance(header.get("descr"), str)
    ):
        raise RuntimeError("projection numerical NPY schema drifted")
    return header["shape"], header["descr"], raw[header_end:]


def _f32_list(payload: bytes) -> list[float]:
    return [value for (value,) in struct.iter_unpack("<f", payload)]


def _bf16_bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def _fp8_e4m3_value(bits: int) -> float:
    sign = -1.0 if bits & 0x80 else 1.0
    exponent = (bits >> 3) & 0xF
    mantissa = bits & 0x7
    if exponent == 0:
        value = mantissa * (2.0**-9)
    elif exponent == 15 and mantissa == 7:
        value = float("nan")
    else:
        value = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
    return sign * value


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _round_to_bf16(value: float) -> float:
    """Round one finite FP32 value to the nearest BF16 (ties to even), as float."""

    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    lower = bits & 0xFFFF
    upper = bits >> 16
    if lower > 0x8000 or (lower == 0x8000 and (upper & 1)):
        upper += 1
    return struct.unpack("<f", struct.pack("<I", (upper & 0xFFFF) << 16))[0]


def _capsule_inputs() -> dict[str, tuple[tuple[int, ...], str, bytes]]:
    raw = _snapshot(CAPSULE_INPUTS_PATH)
    if sha256(raw).hexdigest() != EXPECTED_PREDECESSORS["capsule_input_sha256"]:
        raise RuntimeError("projection numerical capsule inputs drifted")
    members = {}
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        for name in (
            "wk_weight_bits",
            "wk_scale_inv",
            "key_norm_weight_bf16_bits",
            "key_norm_bias_bf16_bits",
        ):
            members[name] = _parse_npy(archive.read(f"{name}.npy"))
    return members


def _materialize_wk_owner0(
    inputs: Mapping[str, tuple[tuple[int, ...], str, bytes]],
) -> list[list[float]]:
    """FP32 wk rows of owner 0 from the pinned FP8 capsule inputs (pure Python)."""

    bits_shape, bits_dtype, bits_payload = inputs["wk_weight_bits"]
    scale_shape, scale_dtype, scale_payload = inputs["wk_scale_inv"]
    if bits_shape != (2, 128, 6144) or bits_dtype != "|u1":
        raise RuntimeError("projection numerical wk bits layout drifted")
    if scale_shape != (2, 1, 48) or scale_dtype != "<f4":
        raise RuntimeError("projection numerical wk scale layout drifted")
    scales = _f32_list(scale_payload)[:48]
    lookup = [_fp8_e4m3_value(code) for code in range(256)]
    owner_bits = bits_payload[: 128 * 6144]
    rows: list[list[float]] = []
    for row in range(128):
        start = row * 6144
        rows.append(
            [
                _round_to_bf16(
                    _f32(lookup[owner_bits[start + column]] * scales[column // 128])
                )
                for column in range(6144)
            ]
        )
    return rows


def _reference_key_f64(
    projected: list[float],
    weight: list[float],
    bias: list[float],
    rope_row: list[float],
) -> list[float]:
    width = len(projected)
    mean = math.fsum(projected) / width
    variance = math.fsum((value - mean) ** 2 for value in projected) / width
    scale = 1.0 / math.sqrt(variance + 1e-6)
    normalized = [
        (value - mean) * scale * w + b
        for value, w, b in zip(projected, weight, bias, strict=True)
    ]
    half = DSA_ROTARY_DIM // 2
    rotated = list(normalized)
    for pair in range(half):
        a, b = normalized[2 * pair], normalized[2 * pair + 1]
        rotated[2 * pair] = a * rope_row[pair] - b * rope_row[half + pair]
        rotated[2 * pair + 1] = a * rope_row[half + pair] + b * rope_row[pair]
    return rotated


def _validate_output_npz(
    raw: bytes,
    runner: Mapping[str, Any],
    numerical: Mapping[str, Any],
    rope_row_raw: bytes,
) -> bool:
    """Re-derive the V3 faithfulness verdict from archived bytes alone."""

    if (
        len(rope_row_raw) != 4 * DSA_ROTARY_DIM
        or sha256(rope_row_raw).hexdigest() != EXPECTED_DSA_ROPE_ROW_SHA256
    ):
        raise RuntimeError("projection numerical host rotary row bytes drifted")

    expected_names = {f"{name}.npy" for name in EXPECTED_OUTPUT_LAYOUT}
    runner_arrays = runner.get("output_arrays")
    numerical_arrays = numerical.get("outputs")
    if (
        not isinstance(runner_arrays, Mapping)
        or set(runner_arrays) != set(EXPECTED_OUTPUT_LAYOUT)
        or not isinstance(numerical_arrays, Mapping)
        or set(numerical_arrays) != set(EXPECTED_OUTPUT_LAYOUT)
        or numerical.get("structural") is not True
    ):
        raise RuntimeError("projection numerical output array catalogue drifted")
    arrays: dict[str, bytes] = {}
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        infos = archive.infolist()
        names = archive.namelist()
        if (
            len(names) != len(set(names))
            or set(names) != expected_names
            or sum(item.file_size for item in infos) > 1 << 20
            or any(item.is_dir() or item.flag_bits & 0x1 for item in infos)
        ):
            raise RuntimeError("projection numerical output NPZ catalogue drifted")
        for name, (shape, dtype, item_size) in EXPECTED_OUTPUT_LAYOUT.items():
            member = archive.read(f"{name}.npy")
            observed_shape, observed_dtype, payload = _parse_npy(member)
            owner_size = len(payload) // 2
            if (
                observed_shape != shape
                or observed_dtype != dtype
                or len(payload) != item_size * shape[0] * shape[1] * shape[2]
                or payload[:owner_size] != payload[owner_size:]
            ):
                raise RuntimeError(f"projection numerical output bytes drifted: {name}")
            owner_hash = sha256(payload[:owner_size]).hexdigest()
            runner_record = runner_arrays[name]
            numerical_record = numerical_arrays[name]
            if (
                not isinstance(runner_record, Mapping)
                or runner_record
                != {
                    "array_sha256": sha256(payload).hexdigest(),
                    "shape": list(shape),
                    "storage_dtype": dtype,
                }
                or not isinstance(numerical_record, Mapping)
                or numerical_record.get("owner_sha256") != [owner_hash, owner_hash]
                or numerical_record.get("owners_equal") is not True
                or numerical_record.get("shape_exact") is not True
                or numerical_record.get("dtype_exact") is not True
                or numerical_record.get("finite") is not True
            ):
                raise RuntimeError(
                    f"projection numerical output record drifted: {name}"
                )
            arrays[name] = payload[:owner_size]
    normalized_exact = (
        sha256(arrays["normalized_hidden_owners"]).hexdigest()
        == EXPECTED_NORMALIZED_OWNER_SHA256
    )
    if (
        numerical_arrays["normalized_hidden_owners"].get("witness_exact")
        is not normalized_exact
    ):
        raise RuntimeError("projection numerical normalized witness flag drifted")
    hidden = [
        _bf16_bits_to_float(bits)
        for (bits,) in struct.iter_unpack("<H", arrays["normalized_hidden_owners"])
    ]
    projected = _f32_list(arrays["projected_key_owners"])
    key = _f32_list(arrays["current_key_owners"])
    if any(not math.isfinite(v) for v in (*hidden, *projected, *key)):
        raise RuntimeError("projection numerical outputs are not finite")
    inputs = _capsule_inputs()
    wk_rows = _materialize_wk_owner0(inputs)
    reference_projection = [
        math.fsum(w * h for w, h in zip(row, hidden, strict=True)) for row in wk_rows
    ]
    projection_error = max(
        abs(a - b) for a, b in zip(projected, reference_projection, strict=True)
    )
    projection_ok = projection_error <= PROJECTION_TOLERANCE
    rope_row = _f32_list(rope_row_raw)
    if any(not math.isfinite(v) for v in rope_row):
        raise RuntimeError("projection numerical host rotary row is not finite")
    if runner.get("dsa_rope_row") != {
        "array_sha256": sha256(rope_row_raw + rope_row_raw).hexdigest(),
        "position": DSA_ROPE_POSITION,
        "row_sha256": EXPECTED_DSA_ROPE_ROW_SHA256,
        "rotary_dim": DSA_ROTARY_DIM,
        "shape": [2, DSA_ROTARY_DIM],
        "storage_dtype": "<f4",
        "theta": DSA_ROTARY_THETA,
    }:
        raise RuntimeError("projection numerical host rotary row record drifted")
    _, _, weight_payload = inputs["key_norm_weight_bf16_bits"]
    _, _, bias_payload = inputs["key_norm_bias_bf16_bits"]
    weight = [
        _bf16_bits_to_float(b)
        for (b,) in struct.iter_unpack("<H", weight_payload[:256])
    ]
    bias = [
        _bf16_bits_to_float(b) for (b,) in struct.iter_unpack("<H", bias_payload[:256])
    ]
    reference_key = _reference_key_f64(projected, weight, bias, rope_row)
    key_errors = [abs(a - b) for a, b in zip(key, reference_key, strict=True)]
    key_ok = max(key_errors) <= KEY_TOLERANCE
    pre_rotation = _reference_key_f64(projected, weight, bias, [1.0] * 32 + [0.0] * 32)
    worst_cos = worst_sin = 0.0
    half = DSA_ROTARY_DIM // 2
    for pair in range(half):
        a, b = pre_rotation[2 * pair], pre_rotation[2 * pair + 1]
        ap, bp = key[2 * pair], key[2 * pair + 1]
        radius = a * a + b * b
        if radius <= 0.0:
            continue
        worst_cos = max(worst_cos, abs((a * ap + b * bp) / radius - rope_row[pair]))
        worst_sin = max(
            worst_sin, abs((a * bp - b * ap) / radius - rope_row[half + pair])
        )
    implied_ok = (
        worst_cos <= IMPLIED_COS_SIN_TOLERANCE
        and worst_sin <= IMPLIED_COS_SIN_TOLERANCE
    )
    projection_record = numerical_arrays["projected_key_owners"]
    key_record = numerical_arrays["current_key_owners"]
    implied_record = key_record.get("implied_rotary")

    def _metric_matches(record: Any, name: str, expected: float) -> bool:
        value = record.get(name) if isinstance(record, Mapping) else None
        return (
            type(value) is float
            and math.isfinite(value)
            and abs(value - expected) <= 1e-9
        )

    if (
        projection_record.get("within_tolerance") is not projection_ok
        or projection_record.get("tolerance") != PROJECTION_TOLERANCE
        or not _metric_matches(
            projection_record, "max_abs_error_vs_f64_reference", projection_error
        )
        or key_record.get("within_tolerance") is not key_ok
        or key_record.get("tolerance") != KEY_TOLERANCE
        or key_record.get("implied_rotary_within_tolerance") is not implied_ok
        or not _metric_matches(
            key_record, "max_abs_error_vs_f64_reference", max(key_errors)
        )
        or not _metric_matches(
            key_record, "rotary_max_abs_error", max(key_errors[:DSA_ROTARY_DIM])
        )
        or not _metric_matches(
            key_record, "nonrotary_max_abs_error", max(key_errors[DSA_ROTARY_DIM:])
        )
        or not isinstance(implied_record, Mapping)
        or set(implied_record) != {"max_abs_cos_error", "max_abs_sin_error"}
        or not _metric_matches(implied_record, "max_abs_cos_error", worst_cos)
        or not _metric_matches(implied_record, "max_abs_sin_error", worst_sin)
    ):
        raise RuntimeError(
            "projection numerical faithfulness flags differ from output bytes"
        )
    derived_accepted = all((normalized_exact, projection_ok, key_ok, implied_ok))
    if numerical.get("accepted_tpu_host_rope_faithful") is not derived_accepted:
        raise RuntimeError(
            "projection numerical acceptance flag differs from output bytes"
        )
    return derived_accepted


def _validate_dependencies(
    base: Any,
    runner: Mapping[str, Any],
    dependencies_raw: bytes,
    code_pin: str,
) -> None:
    dependencies = json.loads(dependencies_raw)
    identity = runner.get("compiler_dependency_manifest")
    if not isinstance(dependencies, Mapping) or not isinstance(identity, Mapping):
        raise TypeError("projection numerical dependency authority is absent")
    accelerator = dependencies.get("accelerator_device_nodes_observed_mapped")
    native = dependencies.get("native_mappings")
    modules = dependencies.get("python_modules")
    base._validate_accelerator_device_mapping_records(accelerator, verify_live=True)
    base._validate_dependency_records(native, "native_mappings", verify_live=True)
    base._validate_dependency_records(modules, "python_modules", verify_live=True)
    if (
        set(dependencies)
        != {
            "accelerator_device_observation_scope",
            "accelerator_device_nodes_observed_mapped",
            "artifact_kind",
            "code_hash",
            "dependency_sites",
            "environment",
            "native_mappings",
            "python_modules",
            "python_runtime",
            "sealed_project_source",
        }
        or set(identity)
        != {
            "accelerator_device_node_count",
            "accelerator_device_nodes_sha256",
            "byte_count",
            "filename",
            "native_mapping_count",
            "python_module_count",
            "sha256",
        }
        or dependencies.get("artifact_kind")
        != "gate_d_projection_contraction_pp16_numerical_dependencies"
        or dependencies.get("code_hash") != code_pin
        or dependencies.get("environment") != EXPECTED_COMPILER_ENVIRONMENT
        or identity.get("filename") != "dependencies.json"
        or identity.get("byte_count") != len(dependencies_raw)
        or identity.get("sha256") != sha256(dependencies_raw).hexdigest()
        or identity.get("accelerator_device_node_count") != len(accelerator)
        or identity.get("accelerator_device_nodes_sha256")
        != base._accelerator_device_nodes_sha256(accelerator)
        or identity.get("native_mapping_count") != len(native)
        or identity.get("python_module_count") != len(modules)
        or runner.get("sealed_project_source")
        != dependencies.get("sealed_project_source")
    ):
        raise RuntimeError("projection numerical dependency authority drifted")


def _prepare_success(
    base: Any,
    run_fd: int,
    *,
    code_pin: str,
    run_tag: str,
    remote: str,
    elapsed: int,
) -> dict[str, bytes]:
    runner_raw = base.snapshot_member(run_fd, "runner.json")
    dependencies_raw = base.snapshot_member(run_fd, "dependencies.json")
    publisher_runtime_raw = base.snapshot_member(
        run_fd, "publisher_runtime.json", limit=1 << 20
    )
    runner = json.loads(runner_raw)
    if not isinstance(runner, Mapping):
        raise TypeError("projection numerical runner report is not an object")
    accepted, status, classification = _expected_status(runner)
    expected_keys = {
        "artifact_kind",
        "claim_scope",
        "classification",
        "code_hash",
        "compiled_executable_invocation_count",
        "compiler_dependency_manifest",
        "dsa_rope_row",
        "gate_d_closed",
        "host_rope_source_sha256s",
        "host_transfer_count",
        "hlo",
        "memory_after_compile",
        "memory_after_execute",
        "memory_before_compile",
        "numerical",
        "output_artifact",
        "output_arrays",
        "performance_claim",
        "physical_group",
        "predecessors",
        "projection_source_authority",
        "root_cause_fix_proven",
        "runtime",
        "schema_version",
        "sealed_project_source",
        "status",
        "tpu_execution_elapsed_ns_diagnostic_only",
        "tpu_numerical_execution_performed",
        "wk_weight",
    }
    if (
        set(runner) != expected_keys
        or runner.get("artifact_kind")
        != "gate_d_projection_host_rope_pp16_numerical_replay"
        or runner.get("schema_version") != 2
        or runner.get("code_hash") != code_pin
        or runner.get("status") != status
        or runner.get("classification") != classification
        or runner.get("claim_scope")
        != "One exact-input, one-invocation, two-chip PP16 projection/current-key witness with a host FP32 rotary row only. No decoder, 8K, token, performance, or Gate-D closure claim."
        or runner.get("compiled_executable_invocation_count") != 1
        or runner.get("host_transfer_count") != 1
        or runner.get("tpu_numerical_execution_performed") is not True
        or runner.get("gate_d_closed") is not False
        or runner.get("root_cause_fix_proven") is not False
        or runner.get("performance_claim") is not False
        or runner.get("physical_group") != EXPECTED_PHYSICAL_GROUP
        or runner.get("predecessors") != EXPECTED_PREDECESSORS
        or runner.get("host_rope_source_sha256s") != HOST_ROPE_SOURCE_SHA256S
        or runner.get("projection_source_authority")
        != base._projection_contraction_source_authority(code_pin)
        or runner.get("wk_weight")
        != {
            "array_sha256": EXPECTED_WK_WEIGHT_FP32_SHA256,
            "semantic_dtype": "float32",
            "shape": [2, 128, 6144],
        }
        or type(runner.get("tpu_execution_elapsed_ns_diagnostic_only")) is not int
        or runner["tpu_execution_elapsed_ns_diagnostic_only"] <= 0
    ):
        raise RuntimeError("projection numerical runner claim boundary drifted")
    _validate_dependencies(base, runner, dependencies_raw, code_pin)
    numerical = runner["numerical"]
    numerical_outputs = numerical.get("outputs")
    base_flags = {
        "dtype_exact",
        "finite",
        "owner_sha256",
        "owners_equal",
        "shape_exact",
    }
    if (
        set(numerical) != {"accepted_tpu_host_rope_faithful", "outputs", "structural"}
        or numerical.get("structural") is not True
        or not isinstance(numerical_outputs, Mapping)
        or set(numerical_outputs)
        != {
            "current_key_owners",
            "normalized_hidden_owners",
            "projected_key_owners",
        }
        or not all(isinstance(record, Mapping) for record in numerical_outputs.values())
        or set(numerical_outputs["normalized_hidden_owners"])
        != base_flags | {"witness_exact"}
        or set(numerical_outputs["projected_key_owners"])
        != base_flags
        | {"max_abs_error_vs_f64_reference", "tolerance", "within_tolerance"}
        or set(numerical_outputs["current_key_owners"])
        != base_flags
        | {
            "implied_rotary",
            "implied_rotary_within_tolerance",
            "max_abs_error_vs_f64_reference",
            "nonrotary_max_abs_error",
            "rotary_max_abs_error",
            "tolerance",
            "within_tolerance",
        }
        or accepted
        != all(
            (
                numerical_outputs["normalized_hidden_owners"].get("witness_exact")
                is True,
                numerical_outputs["projected_key_owners"].get("within_tolerance")
                is True,
                numerical_outputs["current_key_owners"].get("within_tolerance") is True,
                numerical_outputs["current_key_owners"].get(
                    "implied_rotary_within_tolerance"
                )
                is True,
            )
        )
    ):
        raise RuntimeError("projection numerical output classification drifted")
    hlo_files = {
        "optimized_hlo": "hlo/projection_contraction_pp16_stage0.optimized_hlo.txt",
        "stablehlo": "hlo/projection_contraction_pp16_stage0.stablehlo.mlir",
    }
    hlo_payload: dict[str, bytes] = {}
    runner_hlo = runner.get("hlo")
    if not isinstance(runner_hlo, Mapping) or set(runner_hlo) != {
        "optimized_hlo",
        "stablehlo",
    }:
        raise RuntimeError("projection numerical HLO identity catalogue drifted")
    for kind, relative in hlo_files.items():
        raw = base.snapshot_member(run_fd, relative)
        text = raw.decode("utf-8", errors="strict")
        lowered = text.lower()
        identity = runner_hlo.get(kind, {})
        structure = identity.get("structure") if isinstance(identity, Mapping) else None
        if (
            not isinstance(identity, Mapping)
            or identity.get("byte_count") != len(raw)
            or identity.get("sha256") != sha256(raw).hexdigest()
            or not isinstance(structure, Mapping)
            or structure.get("collective_count") != 0
            or structure.get("transcendental_count") != 0
            or structure.get("owner_count") != 2
            or structure.get("live_rows_per_owner") != 1
        ):
            raise RuntimeError(f"projection numerical {kind} identity drifted")
        if kind == "optimized_hlo":
            if (
                not text.startswith(f"HloModule {HLO_MODULE_NAME},")
                or "num_partitions=2" not in text
                or any(token in lowered for token in HLO_FORBIDDEN_TEXT)
                or structure.get("module_name") != HLO_MODULE_NAME
                or structure.get("entry_parameter_count") != 5
                or structure.get("contraction_input_width") != 6144
            ):
                raise RuntimeError(
                    "projection numerical optimized HLO contract drifted"
                )
        elif any(item not in text for item in STABLEHLO_REQUIRED_TEXT) or any(
            token in lowered for token in STABLEHLO_FORBIDDEN_TEXT
        ):
            raise RuntimeError("projection numerical StableHLO contract drifted")
        hlo_payload[relative] = raw
    output_raw = base.snapshot_member(run_fd, "outputs.npz")
    if runner.get("output_artifact") != {
        "byte_count": len(output_raw),
        "filename": "outputs.npz",
        "sha256": sha256(output_raw).hexdigest(),
    }:
        raise RuntimeError("projection numerical output artifact drifted")
    rope_row_raw = base.snapshot_member(run_fd, "dsa_rope_row.f32le", limit=4096)
    if (
        _validate_output_npz(output_raw, runner, numerical, rope_row_raw)
        is not accepted
    ):
        raise RuntimeError("projection numerical status differs from output bytes")
    mirror_raw = base.snapshot_member(run_fd, "mirror.sha256", limit=2 << 20)
    base._validate_mirror_replay(mirror_raw, code_pin)
    sync_raw = base.snapshot_member(run_fd, "sync.txt", limit=1 << 20)
    expected_sync = (
        f"SYNC_OK {os.uname().nodename} {code_pin} origin_and_same_region_mirror\n"
    ).encode("ascii")
    if sync_raw != expected_sync:
        raise RuntimeError("projection numerical host code authority drifted")
    vacancy_raw = base.snapshot_member(run_fd, "remote_vacancy.raw.txt", limit=1 << 20)
    vacancy_summary = base.snapshot_member(run_fd, "remote_vacancy.txt", limit=1 << 20)
    base._validate_remote_vacancy_evidence(vacancy_raw, vacancy_summary, remote)
    census_pre = base.snapshot_member(run_fd, "census_pre.txt")
    census_post = base.snapshot_member(run_fd, "census_post.txt")
    base._require_census(census_pre, "CENSUS_OK")
    base._require_census(census_post, "CENSUS_OK")
    orchestrator = base.snapshot_member(run_fd, "orchestrator.log", limit=16 << 20)
    base.write_member_exclusive(run_fd, "orchestrator.sealed.log", orchestrator)
    summary = {
        "artifact_kind": "gate_d_projection_contraction_pp16_numerical_summary",
        "classification": classification,
        "code_hash": code_pin,
        "elapsed_seconds": elapsed,
        "gate_d_closed": False,
        "performance_claim": False,
        "remote_prefix": remote,
        "root_cause_fix_proven": False,
        "run_tag": run_tag,
        "status": status,
        "tpu_numerical_execution_performed": True,
    }
    summary_raw = base._canonical(summary)
    base.write_member_exclusive(run_fd, "summary.json", summary_raw)
    payload = {
        "census_post.txt": census_post,
        "census_pre.txt": census_pre,
        "dependencies.json": dependencies_raw,
        "dsa_rope_row.f32le": rope_row_raw,
        **hlo_payload,
        "mirror.sha256": mirror_raw,
        "orchestrator.sealed.log": orchestrator,
        "outputs.npz": output_raw,
        "publisher_runtime.json": publisher_runtime_raw,
        "remote_vacancy.raw.txt": vacancy_raw,
        "remote_vacancy.txt": vacancy_summary,
        "runner.json": runner_raw,
        "runner.log": base.snapshot_member(run_fd, "runner.log", limit=256 << 20),
        "summary.json": summary_raw,
        "sync.txt": sync_raw,
    }
    evidence = {
        "artifact_kind": "gate_d_projection_contraction_pp16_numerical_local_evidence",
        "classification": classification,
        "code_hash": code_pin,
        "files": [base._record(name, payload[name]) for name in sorted(payload)],
        "gate_d_closed": False,
        "performance_claim": False,
        "root_cause_fix_proven": False,
        "run_tag": run_tag,
        "status": status,
    }
    evidence_raw = base._canonical(evidence)
    base.write_member_exclusive(run_fd, "evidence.json", evidence_raw)
    payload["evidence.json"] = evidence_raw
    if tuple(sorted(payload)) != tuple(sorted(SUCCESS_PAYLOAD)):
        raise RuntimeError("projection numerical success inventory drifted")
    if base.local_members(run_fd) != set(SUCCESS_PAYLOAD) | {"orchestrator.log"}:
        raise RuntimeError("projection numerical local preterminal inventory drifted")
    return payload


def _result_authority_line(
    status: str,
    marker_sha256: str,
    terminal_record: Mapping[str, Any],
) -> str:
    generation = terminal_record.get("generation")
    terminal_sha256 = terminal_record.get("sha256")
    if (
        status not in {"NUMERICAL_ACCEPTED", "NUMERICAL_REJECTED"}
        or not re.fullmatch(r"[0-9a-f]{64}", marker_sha256)
        or type(generation) is not str
        or not re.fullmatch(r"[0-9]+", generation)
        or type(terminal_sha256) is not str
        or not re.fullmatch(r"[0-9a-f]{64}", terminal_sha256)
    ):
        raise RuntimeError("projection numerical result authority drifted")
    return (
        f"NUMERICAL_RESULT status={status} marker_sha256={marker_sha256} "
        f"terminal_generation={generation} terminal_sha256={terminal_sha256}"
    )


def _publish_success(
    base: Any,
    run_dir: Path,
    remote: str,
    *,
    code_pin: str,
    elapsed: int,
    publication_runtime_raw: bytes,
    run_dir_fd: int,
) -> str:
    run_tag = base.validate_run_dir(run_dir)
    run_fd = base._run_fd(run_dir, run_dir_fd)
    try:
        _require_preterminal(base, run_fd)
        base.require_publication_runtime(run_fd, publication_runtime_raw)
        payload = _prepare_success(
            base,
            run_fd,
            code_pin=code_pin,
            run_tag=run_tag,
            remote=remote,
            elapsed=elapsed,
        )
        runner = json.loads(payload["runner.json"])
        bucket, prefix = base._bucket_and_prefix(remote, None)
        base._require_never_used_prefix(bucket, prefix)
        records = []
        for relative in sorted(payload):
            record = base._upload_bound(bucket, prefix + relative, payload[relative])
            record["path"] = relative
            records.append(record)
        ledger_raw = base._canonical(
            {
                "artifact_kind": "gate_d_projection_contraction_pp16_numerical_remote_ledger",
                "objects": records,
                "run_tag": run_tag,
            }
        )
        base.write_member_exclusive(run_fd, "remote_objects.json", ledger_raw)
        ledger_record = base._upload_bound(
            bucket, prefix + "remote_objects.json", ledger_raw
        )
        ledger_record["path"] = "remote_objects.json"
        for record in records:
            base._replay_bound(bucket, prefix + record["path"], record)
        base._replay_bound(bucket, prefix + "remote_objects.json", ledger_record)
        expected_remote = set(payload) | {"remote_objects.json"}
        if base._observed_names(bucket, prefix) != expected_remote:
            raise RuntimeError("projection numerical preterminal remote set drifted")
        marker = {
            "artifact_kind": "gate_d_projection_contraction_pp16_numerical_result",
            "evidence_sha256": sha256(payload["evidence.json"]).hexdigest(),
            "gate_d_closed": False,
            "performance_claim": False,
            "remote_ledger": ledger_record,
            "root_cause_fix_proven": False,
            "run_tag": run_tag,
            "status": runner["status"],
            "summary_sha256": sha256(payload["summary.json"]).hexdigest(),
            "tpu_numerical_execution_performed": True,
        }
        marker["marker_payload_sha256"] = sha256(base._canonical(marker)).hexdigest()
        terminal_raw = base._canonical(marker)
        base.write_member_exclusive(run_fd, "NUMERICAL_RESULT", terminal_raw)
        if base.local_members(run_fd) != set(SUCCESS_PAYLOAD) | {
            "orchestrator.log",
            "remote_objects.json",
            "NUMERICAL_RESULT",
        }:
            raise RuntimeError("projection numerical local terminal inventory drifted")
        # This must remain the final remote mutation in the successful result path.
        terminal_record = base._upload_bound(
            bucket, prefix + "NUMERICAL_RESULT", terminal_raw
        )
        terminal_record["path"] = "NUMERICAL_RESULT"
        base._replay_bound(bucket, prefix + "NUMERICAL_RESULT", terminal_record)
        if base._observed_names(bucket, prefix) != expected_remote | {
            "NUMERICAL_RESULT"
        }:
            raise RuntimeError("projection numerical terminal remote set drifted")
        base.write_member_exclusive(
            run_fd,
            "terminal_upload_receipt.json",
            base._canonical(
                {
                    "artifact_kind": "gate_d_projection_contraction_pp16_numerical_terminal_receipt",
                    "remote": remote + "/NUMERICAL_RESULT",
                    "terminal": terminal_record,
                }
            ),
        )
        if base.local_members(run_fd) != set(SUCCESS_PAYLOAD) | {
            "orchestrator.log",
            "remote_objects.json",
            "NUMERICAL_RESULT",
            "terminal_upload_receipt.json",
        }:
            raise RuntimeError(
                "projection numerical terminal receipt inventory drifted"
            )
        return _result_authority_line(
            runner["status"],
            marker["marker_payload_sha256"],
            terminal_record,
        )
    finally:
        os.close(run_fd)


def _publish_diagnostic(
    base: Any,
    run_dir: Path,
    remote: str,
    *,
    code_pin: str,
    status: int,
    publication_runtime_raw: bytes,
    run_dir_fd: int,
) -> None:
    run_tag = base.validate_run_dir(run_dir)
    run_fd = base._run_fd(run_dir, run_dir_fd)
    try:
        _require_preterminal(base, run_fd)
        base.require_publication_runtime(run_fd, publication_runtime_raw)
        failure_raw = base._canonical(
            {
                "artifact_kind": "gate_d_projection_contraction_pp16_numerical_failure",
                "code_hash": code_pin,
                "exit_status": status,
                "gate_d_closed": False,
                "performance_claim": False,
                "root_cause_fix_proven": False,
                "run_tag": run_tag,
                "terminal_payload_present": False,
            }
        )
        base.write_member_exclusive(run_fd, "failure_status.json", failure_raw)
        orchestrator = base.snapshot_member(run_fd, "orchestrator.log", limit=16 << 20)
        base.write_member_exclusive(run_fd, "orchestrator.failure.log", orchestrator)
        excluded = {
            "orchestrator.log",
            "NUMERICAL_RESULT",
            "diagnostic_objects.json",
            "diagnostic_upload_receipt.json",
        }
        members = [
            relative
            for relative in sorted(base.local_members(run_fd))
            if relative not in excluded
        ]
        if not members or len(members) > 64:
            raise RuntimeError("projection numerical diagnostic inventory is unsafe")
        payload = {name: base.snapshot_member(run_fd, name) for name in members}
        vacancy_raw = base.snapshot_member(
            run_fd, "remote_vacancy.raw.txt", limit=1 << 20
        )
        vacancy_summary = base.snapshot_member(
            run_fd, "remote_vacancy.txt", limit=1 << 20
        )
        base._validate_remote_vacancy_evidence(vacancy_raw, vacancy_summary, remote)
        bucket, prefix = base._bucket_and_prefix(remote, None)
        base._require_never_used_prefix(bucket, prefix)
        diagnostic_prefix = prefix + "diagnostic/"
        records = []
        for relative in sorted(payload):
            record = base._upload_bound(
                bucket, diagnostic_prefix + relative, payload[relative]
            )
            record["path"] = relative
            records.append(record)
        for record in records:
            base._replay_bound(bucket, diagnostic_prefix + record["path"], record)
        if base._observed_names(bucket, diagnostic_prefix) != set(payload):
            raise RuntimeError("projection numerical diagnostic object set drifted")
        ledger_raw = base._canonical(
            {
                "artifact_kind": "gate_d_projection_contraction_pp16_numerical_diagnostic_ledger",
                "objects": records,
                "run_tag": run_tag,
            }
        )
        base.write_member_exclusive(run_fd, "diagnostic_objects.json", ledger_raw)
        # With no numerical result terminal, the diagnostic ledger is final remotely.
        ledger_record = base._upload_bound(
            bucket, diagnostic_prefix + "diagnostic_objects.json", ledger_raw
        )
        ledger_record["path"] = "diagnostic_objects.json"
        base._replay_bound(
            bucket, diagnostic_prefix + "diagnostic_objects.json", ledger_record
        )
        if base._observed_names(bucket, diagnostic_prefix) != set(payload) | {
            "diagnostic_objects.json"
        }:
            raise RuntimeError("projection numerical diagnostic terminal set drifted")
        base.write_member_exclusive(
            run_fd,
            "diagnostic_upload_receipt.json",
            base._canonical(
                {
                    "artifact_kind": "gate_d_projection_contraction_pp16_numerical_diagnostic_terminal_receipt",
                    "remote": remote + "/diagnostic/diagnostic_objects.json",
                    "terminal": ledger_record,
                }
            ),
        )
    finally:
        os.close(run_fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--run-dir-fd", type=int)
    modes = parser.add_subparsers(dest="mode", required=True)
    init = modes.add_parser("init")
    init.add_argument("--run-dir", type=Path, required=True)
    success = modes.add_parser("success")
    success.add_argument("--run-dir", type=Path, required=True)
    success.add_argument("--remote-prefix", required=True)
    success.add_argument("--elapsed", type=int, required=True)
    diagnostic = modes.add_parser("diagnostic")
    diagnostic.add_argument("--run-dir", type=Path, required=True)
    diagnostic.add_argument("--remote-prefix", required=True)
    diagnostic.add_argument("--status", type=int, required=True)
    write = modes.add_parser("write")
    write.add_argument("--run-dir", type=Path, required=True)
    write.add_argument("--member", choices=sorted(WRAPPER_WRITE_MEMBERS), required=True)
    append = modes.add_parser("append-log")
    append.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    os.umask(0o077)
    arguments = parse_args()
    base = _load_base(arguments.expected_code_hash)
    base.validate_environment()
    publication_runtime_raw = base._canonical(_publication_runtime(base))
    _verify_running_source(
        arguments.expected_code_hash, arguments.expected_source_sha256
    )
    if arguments.mode != "init" and arguments.run_dir_fd != 7:
        raise RuntimeError("projection numerical publisher requires run-directory fd 7")
    if arguments.mode == "init":
        print(
            "RUN_IDENTITY "
            + base.initialize_run_dir(
                arguments.run_dir, publication_runtime_raw=publication_runtime_raw
            ),
            flush=True,
        )
    elif arguments.mode == "success":
        print(
            _publish_success(
                base,
                arguments.run_dir,
                arguments.remote_prefix,
                code_pin=arguments.expected_code_hash,
                elapsed=arguments.elapsed,
                publication_runtime_raw=publication_runtime_raw,
                run_dir_fd=arguments.run_dir_fd,
            ),
            flush=True,
        )
    elif arguments.mode == "diagnostic":
        _publish_diagnostic(
            base,
            arguments.run_dir,
            arguments.remote_prefix,
            code_pin=arguments.expected_code_hash,
            status=arguments.status,
            publication_runtime_raw=publication_runtime_raw,
            run_dir_fd=arguments.run_dir_fd,
        )
    elif arguments.mode == "write":
        run_fd = base._run_fd(arguments.run_dir, arguments.run_dir_fd)
        try:
            _require_preterminal(base, run_fd)
            base.require_publication_runtime(run_fd, publication_runtime_raw)
            base.write_preterminal_member(run_fd, arguments.member, sys.stdin.buffer)
        finally:
            os.close(run_fd)
    else:
        run_fd = base._run_fd(arguments.run_dir, arguments.run_dir_fd)
        try:
            _require_preterminal(base, run_fd)
            base.require_publication_runtime(run_fd, publication_runtime_raw)
            payload = sys.stdin.buffer.read((1 << 20) + 1)
            if len(payload) > 1 << 20:
                raise RuntimeError("projection numerical log append exceeds limit")
            base.append_preterminal_log(run_fd, payload)
        finally:
            os.close(run_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
