#!/usr/bin/env python3
"""Run one protected two-chip Gate-D projection numerical discriminator.

The default-off driver authenticates the successful compile-only predecessor,
loads only the sealed layer-1 normalized/key operands, recompiles the identical
PP16 stage-zero callable, invokes it exactly once, and performs one bounded
host transfer.  Acceptance proves only this projection/current-key witness;
Gate D, the decoder, 8K, and performance remain open.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import stat
import subprocess
import sys
import time
import types
import zipfile
from collections.abc import Mapping
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from typing import Any

WORKTREE = Path("/home/gianl/glm-tpu-gate-d-pp16-numerical")
RUN_ROOT = Path("/home/gianl/gate-d-runs")
DRIVER_REPOSITORY_PATH = (
    "scripts/greenfield/run_gate_d_projection_contraction_pp16_numerical.py"
)
HLO_HELPER_REPOSITORY_PATH = (
    "scripts/greenfield/acquire_gate_d_projection_contraction_pp16_hlo.py"
)
HLO_HELPER_PIN = "5cc3b4181ac5ac2b3c630130521b4f3da15a1d3e"
HLO_HELPER_SHA256 = "c660d50eb60054c9b267840230fb69bbf7259104416dc96c7b2ee01a2a14934a"
TOPOLOGY_SHA256 = "49cf6bb1a553985855556d1401ad85918669df52d12f8dc5150f247d18b325eb"
PROJECTION_SOURCE_SHA256 = (
    "5744eee0ef2cf35a4566cc0de165be1338160daaae3f4dd133554b2aa8280e9f"
)
CAPSULE_SHA256 = "5b7ad71f37dbbcda0ee36a9fc0c42a7ca45d619a9e741307386755e68e87c1a4"
CAPSULE_INPUT_SHA256 = (
    "dd5f1cbb37b722591531635a21cfa9f1d6d0157997a4f70138900d0000be8b1b"
)
CAPSULE_STATE_SHA256 = (
    "68ee47b1fcbf317fd41da51aa26c9ba1a8dafe0e3e95ccbbfa73285f4e46f236"
)
CAPSULE_EXECUTION_AUTHORITY_SHA256 = (
    "8b8c9cc79a18679628582a66c418a7f63e06553091928df58defbdde3971a660"
)
HOST_MATERIALIZATION_AUTHORITY_SHA256 = (
    "a7e5b393f7c61181f6b2aab9531d788fb27594d9cf980055c165b6f278ba4f99"
)
FRONTIER_AUTHORITY_SHA256 = (
    "4a6be0f33f221df46f384cd2047da1aa664ae4c45763e68f0f1df05f8644e20c"
)
HLO_SUCCESS_AUTHORITY_SHA256 = (
    "54eb6105b81d9ffdbdd3b90735059fb324701509fe8efd003033bb3cff7e0ad7"
)
EXPECTED_WK_WEIGHT_FP32_SHA256 = (
    "b2c67e0fdf4d7292494778233e9813d8256f2fb88b6f7376870379832fbab24b"
)
EXPECTED_STABLEHLO_SHA256 = (
    "4b3fa252e837d381208453b06d7e369947fa4c6c58c795edc8ff5a8ea8c9e2e3"
)
EXPECTED_OPTIMIZED_HLO_SHA256 = (
    "817ba2ed87c33ec928f834fcf3a003ce63d3dedf4061a7c64fe354a26ac498ea"
)
EXPECTED_STABLEHLO_BYTES = 7420
EXPECTED_OPTIMIZED_HLO_BYTES = 31857
# The compile-only acquisition driver and this numerical driver are different
# installed files, so XLA's optimized-HLO debug metadata (FileNames entry and
# two call-site FileLocations) differs while the graph body and StableHLO are
# byte-identical.  The reviewed bridge artifact pins exactly those three
# substitutions; the expected numerical HLO is derived from the accepted
# preimage and nothing else is normalized.
HLO_SOURCE_LOCATION_BRIDGE_SHA256 = (
    "c2732f7184416cce87839b4cadf552b94d983aba06fa56b687eaacc03ca26bea"
)
EXPECTED_NUMERICAL_OPTIMIZED_HLO_SHA256 = (
    "70485b066b44233d82c2a728f09753f8a71306074fdd8b856112e46ab1f60564"
)
EXPECTED_NUMERICAL_OPTIMIZED_HLO_BYTES = 31861
HLO_ACQUIRER_INSTALL_PATH = (
    "/usr/local/libexec/glm-tpu/gate-d-projection-contraction-pp16-hlo-v3/"
    "acquire_gate_d_projection_contraction_pp16_hlo.py"
)
NUMERICAL_DRIVER_INSTALL_PATH = (
    "/usr/local/libexec/glm-tpu/gate-d-projection-contraction-pp16-numerical-v2/"
    "run_gate_d_projection_contraction_pp16_numerical.py"
)
HLO_ACQUIRER_MODULE_CALL_LINE = 1506
HLO_ACQUIRER_LOWER_CALL_LINE = 1365
NUMERICAL_DRIVER_MODULE_CALL_LINE = 837
NUMERICAL_DRIVER_LOWER_CALL_LINE = 676
TAG_PATTERN = r"gate_d_projection_contraction_pp16_numerical_[0-9]{8}T[0-9]{15}Z"
_MAX_NPZ_MEMBER_BYTES = 64 * 1024 * 1024
_MAX_NPZ_TOTAL_BYTES = 32 * 1024 * 1024
_F_GET_SEALS = 1034
_MEMFD_SEALS = 0x0001 | 0x0002 | 0x0004 | 0x0008
_EXPECTED_ENVIRONMENT = {
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
_STATE_NAMES = {
    "cache_history",
    "current_key",
    "event1_positions",
    "event1_scores",
    "event1_valid_count",
    "head_weights",
    "normalized",
    "query",
    "rms_hidden_update",
    "rms_input",
    "rms_residual",
}


def _canonical(value: Any) -> str:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


def _array_sha256(value: Any, np: Any) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _snapshot_regular(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(f"unsafe regular-file identity: {path}")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(block)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if (
            len(raw) != before.st_size
            or identity
            != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise RuntimeError(f"regular file changed while reading: {path}")
        return raw
    finally:
        os.close(descriptor)


def _git_bytes(*arguments: str) -> bytes:
    return subprocess.check_output(
        ["/usr/bin/git", "-C", str(WORKTREE), *arguments], env=_GIT_ENVIRONMENT
    )


def _git_text(*arguments: str) -> str:
    return _git_bytes(*arguments).decode("ascii", errors="strict").strip()


def _verify_running_source(code_pin: str, expected_sha256: str) -> None:
    if _git_bytes("for-each-ref", "--format=%(refname)", "refs/replace"):
        raise RuntimeError("Gate-D numerical source repository has replacement refs")
    raw = _snapshot_regular(Path(__file__))
    committed = _git_bytes("show", f"{code_pin}:{DRIVER_REPOSITORY_PATH}")
    if (
        raw != committed
        or sha256(raw).hexdigest() != expected_sha256
        or _git_text("rev-parse", f"{code_pin}^{{commit}}") != code_pin
        or _git_text("rev-parse", "HEAD") != code_pin
    ):
        raise RuntimeError(
            "Gate-D projection numerical driver is not the committed blob"
        )


def _load_hlo_helper(code_pin: str) -> types.ModuleType:
    path = WORKTREE / HLO_HELPER_REPOSITORY_PATH
    raw = _snapshot_regular(path)
    if (
        raw != _git_bytes("show", f"{code_pin}:{HLO_HELPER_REPOSITORY_PATH}")
        or raw != _git_bytes("show", f"{HLO_HELPER_PIN}:{HLO_HELPER_REPOSITORY_PATH}")
        or sha256(raw).hexdigest() != HLO_HELPER_SHA256
    ):
        raise RuntimeError("Gate-D projection compile helper bytes drifted")
    module = types.ModuleType("_gate_d_projection_exact_hlo_helper")
    module.__file__ = str(path)
    module.__package__ = None
    exec(compile(raw, str(path), "exec"), module.__dict__)  # noqa: S102
    module.REPO = WORKTREE
    module.RUN_ROOT = RUN_ROOT
    module.TAG_PATTERN = TAG_PATTERN
    module._EXPECTED_ENVIRONMENT = dict(_EXPECTED_ENVIRONMENT)
    module._git_bytes = lambda *args, repo=WORKTREE: subprocess.check_output(
        ["/usr/bin/git", "-C", str(repo), *args], env=_GIT_ENVIRONMENT
    )
    return module


def _load_bound_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    raw = _snapshot_regular(path)
    if sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError(f"Gate-D {label} bytes drifted")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError(f"Gate-D {label} root is not an object")
    return value


def _load_npz(raw: bytes, expected_names: set[str], np: Any) -> dict[str, Any]:
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        infos = archive.infolist()
        names = archive.namelist()
        if (
            len(names) != len(set(names))
            or set(names) != {f"{name}.npy" for name in expected_names}
            or sum(item.file_size for item in infos) > _MAX_NPZ_TOTAL_BYTES
            or any(
                item.is_dir()
                or item.flag_bits & 0x1
                or item.file_size > _MAX_NPZ_MEMBER_BYTES
                for item in infos
            )
        ):
            raise RuntimeError("Gate-D projection NPZ catalogue drifted")
    with np.load(BytesIO(raw), allow_pickle=False) as archive:
        if set(archive.files) != expected_names:
            raise RuntimeError("Gate-D projection NPZ array catalogue drifted")
        return {name: np.ascontiguousarray(archive[name]) for name in expected_names}


def _validate_input_arrays(
    values: Mapping[str, Any], records: Mapping[str, Any], np: Any
) -> None:
    if set(values) != set(records):
        raise RuntimeError("Gate-D projection input array set drifted")
    for name, value in values.items():
        record = records[name]
        if (
            not isinstance(record, Mapping)
            or list(value.shape) != record.get("shape")
            or value.dtype.str != record.get("storage_dtype")
            or _array_sha256(value, np) != record.get("array_sha256")
        ):
            raise RuntimeError(f"Gate-D projection input array drifted: {name}")


def _fp8_lookup(np: Any) -> Any:
    values = []
    for bits in range(256):
        sign = -1.0 if bits & 0x80 else 1.0
        exponent = (bits >> 3) & 0xF
        mantissa = bits & 0x7
        if exponent == 0:
            value = mantissa * (2.0**-9)
        elif exponent == 15 and mantissa == 7:
            value = float("nan")
        else:
            value = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
        values.append(sign * value)
    return np.asarray(values, dtype=np.float32)


def _materialize_wk(inputs: Mapping[str, Any], np: Any, ml_dtypes: Any) -> Any:
    bits = inputs["wk_weight_bits"]
    scales = inputs["wk_scale_inv"]
    rows = np.arange(bits.shape[-2]) // 128
    columns = np.arange(bits.shape[-1]) // 128
    expanded = scales[..., rows[:, None], columns[None, :]]
    decoded = _fp8_lookup(np)[bits.astype(np.int32)] * expanded.astype(np.float32)
    bf16 = np.ascontiguousarray(decoded, dtype=ml_dtypes.bfloat16)
    return np.ascontiguousarray(bf16.astype(np.float32), dtype=np.float32)


def _deterministic_npz(values: Mapping[str, Any], np: Any) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(values):
            payload = BytesIO()
            np.lib.format.write_array(
                payload, np.ascontiguousarray(values[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100400 << 16
            archive.writestr(info, payload.getvalue())
    return output.getvalue()


def _host_outputs(result: Any, jax: Any, np: Any) -> dict[str, Any]:
    transferred = jax.device_get(result)
    return {
        name: np.ascontiguousarray(value)
        for name, value in zip(result._fields, transferred, strict=True)
    }


def expected_hlo_source_location_replacements() -> list[dict[str, Any]]:
    return [
        {
            "new": NUMERICAL_DRIVER_INSTALL_PATH,
            "old": HLO_ACQUIRER_INSTALL_PATH,
            "occurrence_count": 1,
            "surface": "FileNames",
        },
        {
            "new": (
                f"line={NUMERICAL_DRIVER_MODULE_CALL_LINE} "
                f"end_line={NUMERICAL_DRIVER_MODULE_CALL_LINE}"
            ),
            "old": (
                f"line={HLO_ACQUIRER_MODULE_CALL_LINE} "
                f"end_line={HLO_ACQUIRER_MODULE_CALL_LINE}"
            ),
            "occurrence_count": 1,
            "surface": "FileLocations module call",
        },
        {
            "new": (
                f"line={NUMERICAL_DRIVER_LOWER_CALL_LINE} "
                f"end_line={NUMERICAL_DRIVER_LOWER_CALL_LINE}"
            ),
            "old": (
                f"line={HLO_ACQUIRER_LOWER_CALL_LINE} "
                f"end_line={HLO_ACQUIRER_LOWER_CALL_LINE}"
            ),
            "occurrence_count": 1,
            "surface": "FileLocations lower call",
        },
    ]


def derive_hlo_from_source_location_replacements(
    accepted_optimized_hlo: bytes, replacements: list[dict[str, Any]]
) -> bytes:
    """Apply the exact single-occurrence substitutions; nothing else changes."""

    derived = accepted_optimized_hlo
    for replacement in replacements:
        old = replacement["old"].encode("ascii")
        new = replacement["new"].encode("ascii")
        if (
            replacement["occurrence_count"] != 1
            or derived.count(old) != 1
            or derived.count(new) != 0
        ):
            raise RuntimeError("Gate-D HLO source-location occurrence drifted")
        derived = derived.replace(old, new, 1)
    return derived


def validate_hlo_source_location_bridge(
    report: Mapping[str, Any], accepted_optimized_hlo: bytes
) -> tuple[bytes, dict[str, Any]]:
    """Derive the numerical-driver optimized HLO from the accepted preimage."""

    replacements = expected_hlo_source_location_replacements()
    expected = {
        "artifact_kind": (
            "gate_d_projection_contraction_pp16_hlo_source_location_bridge"
        ),
        "classification": (
            "SOURCE_LOCATION_METADATA_ONLY_HASH_DRIFT;"
            "GRAPH_BODY_UNCHANGED_BY_EXACT_PREIMAGE_DERIVATION;"
            "STABLEHLO_BYTE_IDENTICAL;TPU_EXECUTABLE_NOT_INVOKED;"
            "NUMERICAL_UNPROVEN;NO_PERFORMANCE_CLAIM;GATE_D_OPEN"
        ),
        "derived_numerical_hlo": {
            "byte_count": EXPECTED_NUMERICAL_OPTIMIZED_HLO_BYTES,
            "sha256": EXPECTED_NUMERICAL_OPTIMIZED_HLO_SHA256,
        },
        "derivation": {
            "method": (
                "Apply exactly the three listed single-occurrence UTF-8 "
                "substitutions to the accepted optimized-HLO bytes, with no "
                "normalization or other mutation, then hash the complete result."
            ),
            "replacement_count": 3,
            "replacements": replacements,
        },
        "gate_d_closed": False,
        "numerical_claim": False,
        "observed_failure": {
            "compiled_executable_invocation_count": 0,
            "driver_install_path": (
                "/usr/local/libexec/glm-tpu/"
                "gate-d-projection-contraction-pp16-numerical-v1/"
                "run_gate_d_projection_contraction_pp16_numerical.py"
            ),
            "optimized_hlo_bytes_preserved": True,
            "optimized_hlo_sha256": (
                "f783a7d8096cd29855ddab01cbb154ede69aee505a5ca34d3b6c255fc8635158"
            ),
            "run_tag": (
                "gate_d_projection_contraction_pp16_numerical_20260901T233855937688834Z"
            ),
            "stablehlo_sha256": EXPECTED_STABLEHLO_SHA256,
        },
        "performance_claim": False,
        "schema_version": 1,
        "source_hlo": {
            "byte_count": EXPECTED_OPTIMIZED_HLO_BYTES,
            "run_tag": "gate_d_projection_contraction_pp16_hlo_20260901T213605719107105Z",
            "sha256": EXPECTED_OPTIMIZED_HLO_SHA256,
        },
        "verification_scope": (
            "Under SHA-256 collision resistance, the numerical driver's compiler "
            "text equals the exact accepted optimized-HLO preimage after only the "
            "authenticated installed-path and call-site line metadata "
            "substitutions. This authorizes only the exact numerical-driver raw "
            "HLO hash; it does not relax StableHLO identity, HLO locality, "
            "numerical, performance, or Gate-D gates."
        ),
        "tpu_numerical_execution_performed": False,
    }
    if report != expected:
        raise RuntimeError("Gate-D HLO source-location bridge schema drifted")
    if (
        len(accepted_optimized_hlo) != EXPECTED_OPTIMIZED_HLO_BYTES
        or sha256(accepted_optimized_hlo).hexdigest() != EXPECTED_OPTIMIZED_HLO_SHA256
    ):
        raise RuntimeError("Gate-D accepted optimized-HLO preimage drifted")
    derived = derive_hlo_from_source_location_replacements(
        accepted_optimized_hlo, replacements
    )
    if (
        len(derived) != EXPECTED_NUMERICAL_OPTIMIZED_HLO_BYTES
        or sha256(derived).hexdigest() != EXPECTED_NUMERICAL_OPTIMIZED_HLO_SHA256
    ):
        raise RuntimeError("Gate-D derived numerical optimized-HLO drifted")
    return derived, {
        "artifact_sha256": HLO_SOURCE_LOCATION_BRIDGE_SHA256,
        "derived_numerical_hlo": expected["derived_numerical_hlo"],
        "replacement_count": 3,
        "source_hlo": expected["source_hlo"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--expected-driver-sha256", required=True)
    parser.add_argument("--topology-authority", type=Path, required=True)
    parser.add_argument("--projection-source", type=Path, required=True)
    parser.add_argument("--hlo-success-authority", type=Path, required=True)
    parser.add_argument("--accepted-stablehlo", type=Path, required=True)
    parser.add_argument("--accepted-optimized-hlo", type=Path, required=True)
    parser.add_argument("--hlo-source-location-bridge", type=Path, required=True)
    parser.add_argument("--frontier-authority", type=Path, required=True)
    parser.add_argument("--host-materialization-authority", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--capsule-inputs", type=Path, required=True)
    parser.add_argument("--capsule-state", type=Path, required=True)
    parser.add_argument("--capsule-execution-authority", type=Path, required=True)
    parser.add_argument("--execute-once", type=int, choices=(1,), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-dir-fd", type=int, choices=(7,), required=True)
    return parser.parse_args()


def main() -> int:
    os.umask(0o077)
    args = parse_args()
    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not sys.flags.ignore_environment
    ):
        raise RuntimeError("Gate-D projection numerical replay requires Python -I -S")
    helper = _load_hlo_helper(args.expected_code_hash)
    python_runtime = helper._validate_python_runtime()
    dependency_sites = helper._validate_dependency_sites()
    compiler_environment = helper._validate_environment()
    _verify_running_source(args.expected_code_hash, args.expected_driver_sha256)

    topology = _load_bound_json(
        args.topology_authority, TOPOLOGY_SHA256, "topology authority"
    )
    stage_zero = helper.validate_topology(topology)
    projection_source = _load_bound_json(
        args.projection_source, PROJECTION_SOURCE_SHA256, "projection source authority"
    )
    projection_source_binding = helper.validate_projection_contraction_source(
        projection_source
    )
    helper.verify_projection_contraction_source_git_blobs(
        args.expected_code_hash, projection_source_binding
    )
    capsule = _load_bound_json(args.capsule, CAPSULE_SHA256, "capsule")
    execution_authority = _load_bound_json(
        args.capsule_execution_authority,
        CAPSULE_EXECUTION_AUTHORITY_SHA256,
        "capsule execution authority",
    )
    host_materialization = _load_bound_json(
        args.host_materialization_authority,
        HOST_MATERIALIZATION_AUTHORITY_SHA256,
        "host-materialization authority",
    )
    frontier = _load_bound_json(
        args.frontier_authority, FRONTIER_AUTHORITY_SHA256, "frontier authority"
    )
    hlo_success = _load_bound_json(
        args.hlo_success_authority,
        HLO_SUCCESS_AUTHORITY_SHA256,
        "HLO success authority",
    )
    accepted_stablehlo = _snapshot_regular(args.accepted_stablehlo)
    accepted_optimized_hlo = _snapshot_regular(args.accepted_optimized_hlo)
    if (
        len(accepted_stablehlo) != EXPECTED_STABLEHLO_BYTES
        or sha256(accepted_stablehlo).hexdigest() != EXPECTED_STABLEHLO_SHA256
        or len(accepted_optimized_hlo) != EXPECTED_OPTIMIZED_HLO_BYTES
        or sha256(accepted_optimized_hlo).hexdigest() != EXPECTED_OPTIMIZED_HLO_SHA256
    ):
        raise RuntimeError("Gate-D accepted projection HLO bytes drifted")
    if Path(__file__) != Path(NUMERICAL_DRIVER_INSTALL_PATH):
        raise RuntimeError("Gate-D numerical driver is not the bridged installed path")
    hlo_bridge_raw = _snapshot_regular(args.hlo_source_location_bridge)
    if sha256(hlo_bridge_raw).hexdigest() != HLO_SOURCE_LOCATION_BRIDGE_SHA256:
        raise RuntimeError("Gate-D HLO source-location bridge bytes drifted")
    derived_optimized_hlo, hlo_bridge_binding = validate_hlo_source_location_bridge(
        json.loads(hlo_bridge_raw.decode("ascii", errors="strict")),
        accepted_optimized_hlo,
    )
    input_raw = _snapshot_regular(args.capsule_inputs)
    state_raw = _snapshot_regular(args.capsule_state)
    if (
        sha256(input_raw).hexdigest() != CAPSULE_INPUT_SHA256
        or sha256(state_raw).hexdigest() != CAPSULE_STATE_SHA256
    ):
        raise RuntimeError("Gate-D projection capsule NPZ bytes drifted")
    run_fd = helper._open_inherited_run_dir(args.run_dir, args.run_dir_fd)
    helper._write_run_member_exclusive(
        run_fd, "hlo/acquired_preimage.stablehlo.mlir", accepted_stablehlo
    )
    helper._write_run_member_exclusive(
        run_fd, "hlo/acquired_preimage.optimized_hlo.txt", accepted_optimized_hlo
    )
    helper._write_run_member_exclusive(
        run_fd, "hlo/source_location_bridge.json", hlo_bridge_raw
    )
    source_fd, source_path, source_snapshot = helper._sealed_git_source_archive(
        args.expected_code_hash, repo=WORKTREE
    )
    helper._install_sealed_source_path(source_path)
    helper._validate_compiler_import_path(source_path)

    import jax
    import ml_dtypes
    import numpy as np

    from glm_tpu.greenfield.benchmarking.gate_d_projection_contraction_pp16 import (
        build_gate_d_projection_contraction_pp16,
    )
    from glm_tpu.greenfield.validation.gate_d_projection_contraction_numerical import (
        classify_projection_outputs,
        projection_state_records,
        validate_predecessors,
        validate_wk_weight,
    )

    project_modules = helper._verify_project_imports(source_path)
    predecessor_binding = validate_predecessors(
        capsule=capsule,
        execution_authority=execution_authority,
        host_materialization=host_materialization,
        frontier=frontier,
        hlo_success=hlo_success,
    )
    if jax.default_backend() != "tpu" or jax.local_device_count() != 4:
        raise RuntimeError(
            "Gate-D projection replay requires one four-chip TPU-v4 host"
        )
    if bool(jax.config.jax_enable_compilation_cache):
        raise RuntimeError("Gate-D projection replay requires disabled JAX cache")
    devices = tuple(jax.local_devices()[:2])
    live_stage_zero = {
        "coordinates": [list(device.coords) for device in devices],
        "device_ids": [int(device.id) for device in devices],
        "process_index": int(devices[0].process_index),
        "stage_id": 0,
    }
    if live_stage_zero != stage_zero or any(
        int(device.process_index) != 0 or str(device.device_kind) != "TPU v4"
        for device in devices
    ):
        raise RuntimeError(f"Gate-D live PP16 topology drifted: {live_stage_zero}")
    observed_group = {
        **live_stage_zero,
        "local_device_count_visible": int(jax.local_device_count()),
        "mesh_device_count": len(devices),
    }

    input_records = execution_authority.get("input_arrays")
    if not isinstance(input_records, Mapping):
        raise TypeError("Gate-D capsule input records are absent")
    inputs = _load_npz(input_raw, set(input_records), np)
    _validate_input_arrays(inputs, input_records, np)
    state = _load_npz(state_raw, _STATE_NAMES, np)
    state_records = projection_state_records(capsule)
    normalized_bits = state["normalized"]
    normalized_record = state_records["normalized"]
    if (
        list(normalized_bits.shape) != normalized_record["shape"]
        or normalized_bits.dtype.str != normalized_record["storage_dtype"]
        or _array_sha256(normalized_bits, np) != normalized_record["array_sha256"]
    ):
        raise RuntimeError("Gate-D normalized captured array drifted")
    normalized = normalized_bits.view(ml_dtypes.bfloat16)
    wk_weight = _materialize_wk(inputs, np, ml_dtypes)
    wk_identity = validate_wk_weight(wk_weight, np)
    arguments = (
        normalized,
        wk_weight,
        inputs["key_norm_weight_bf16_bits"].view(ml_dtypes.bfloat16),
        inputs["key_norm_bias_bf16_bits"].view(ml_dtypes.bfloat16),
    )
    replay = build_gate_d_projection_contraction_pp16(devices=devices)
    dependencies_before = helper._compiler_dependency_records(source_path)
    if not dependencies_before["accelerator_device_nodes_observed_mapped"]:
        raise RuntimeError("TPU accelerator device mapping is absent before compile")
    memory_before = [helper._memory_stats(device) for device in devices]
    abstract_arguments = tuple(
        jax.ShapeDtypeStruct(value.shape, value.dtype) for value in arguments
    )
    lowered = replay.lower(*abstract_arguments)
    stablehlo = lowered.as_text().encode("utf-8")
    compiled = lowered.compile()
    optimized_hlo = compiled.as_text().encode("utf-8")
    helper._write_run_member_exclusive(
        run_fd, "hlo/projection_contraction_pp16_stage0.stablehlo.mlir", stablehlo
    )
    helper._write_run_member_exclusive(
        run_fd,
        "hlo/projection_contraction_pp16_stage0.optimized_hlo.txt",
        optimized_hlo,
    )
    if stablehlo != accepted_stablehlo or optimized_hlo != derived_optimized_hlo:
        raise RuntimeError("Gate-D projection numerical executable HLO drifted")
    memory_after_compile = [helper._memory_stats(device) for device in devices]

    invocation_count = 0
    started_ns = time.monotonic_ns()
    invocation_count += 1
    result = compiled(*arguments)
    jax.block_until_ready(result)
    completed_ns = time.monotonic_ns()
    host = _host_outputs(result, jax, np)
    if invocation_count != 1:
        raise AssertionError("Gate-D projection executable invocation count drifted")
    classification = classify_projection_outputs(host, np)
    archived = {
        name: (
            np.ascontiguousarray(value).view(np.uint16)
            if str(value.dtype) == "bfloat16"
            else np.ascontiguousarray(value)
        )
        for name, value in host.items()
    }
    output_records = {
        name: {
            "array_sha256": _array_sha256(value, np),
            "shape": list(value.shape),
            "storage_dtype": value.dtype.str,
        }
        for name, value in sorted(archived.items())
    }
    output_raw = _deterministic_npz(archived, np)
    output_identity = helper._write_run_member_exclusive(
        run_fd, "outputs.npz", output_raw
    )

    dependencies_after = helper._compiler_dependency_records(source_path)
    if not dependencies_after["accelerator_device_nodes_observed_mapped"]:
        raise RuntimeError("TPU accelerator device mapping is absent after execution")
    helper._require_dependency_prefix_stable(dependencies_before, dependencies_after)
    if (
        helper._verify_project_imports(source_path) != project_modules
        or fcntl.fcntl(source_fd, _F_GET_SEALS) != _MEMFD_SEALS
    ):
        raise RuntimeError("Gate-D sealed project import closure drifted")
    helper._validate_compiler_import_path(source_path)
    if helper._validate_python_runtime_storage() != python_runtime:
        raise RuntimeError("Gate-D sealed Python runtime changed")
    if helper._validate_dependency_sites() != dependency_sites:
        raise RuntimeError("Gate-D sealed dependency sites changed")
    dependency_manifest = {
        "accelerator_device_observation_scope": (
            helper._ACCELERATOR_DEVICE_OBSERVATION_SCOPE
        ),
        "accelerator_device_nodes_observed_mapped": dependencies_after[
            "accelerator_device_nodes_observed_mapped"
        ],
        "artifact_kind": "gate_d_projection_contraction_pp16_numerical_dependencies",
        "code_hash": args.expected_code_hash,
        "dependency_sites": dependency_sites,
        "environment": compiler_environment,
        "native_mappings": dependencies_after["native_mappings"],
        "python_modules": dependencies_after["python_modules"],
        "python_runtime": python_runtime,
        "sealed_project_source": source_snapshot,
    }
    dependency_raw = (_canonical(dependency_manifest) + "\n").encode("ascii")
    dependency_identity = helper._write_run_member_exclusive(
        run_fd, "dependencies.json", dependency_raw
    )
    accepted = classification["accepted_tpu_projection_match"]
    report = {
        "artifact_kind": "gate_d_projection_contraction_pp16_numerical_replay",
        "claim_scope": (
            "One exact-input, one-invocation, two-chip PP16 projection/current-key "
            "witness only. No decoder, 8K, token, performance, or Gate-D closure claim."
        ),
        "classification": (
            "BOUNDED_TPU_PROJECTION_NUMERICAL_ACCEPTED;ROOT_CAUSE_FIX_UNPROVEN;GATE_D_OPEN"
            if accepted
            else "BOUNDED_TPU_PROJECTION_NUMERICAL_REJECTED;PROJECTION_CONTRACTION_MISMATCH;GATE_D_OPEN"
        ),
        "code_hash": args.expected_code_hash,
        "compiled_executable_invocation_count": invocation_count,
        "compiler_dependency_manifest": {
            **dependency_identity,
            "accelerator_device_node_count": len(
                dependencies_after["accelerator_device_nodes_observed_mapped"]
            ),
            "accelerator_device_nodes_sha256": (
                helper._accelerator_device_nodes_sha256(
                    dependencies_after["accelerator_device_nodes_observed_mapped"]
                )
            ),
            "filename": "dependencies.json",
            "native_mapping_count": len(dependencies_after["native_mappings"]),
            "python_module_count": len(dependencies_after["python_modules"]),
        },
        "gate_d_closed": False,
        "host_transfer_count": 1,
        "hlo": {
            "optimized_hlo": {
                "byte_count": len(optimized_hlo),
                "sha256": sha256(optimized_hlo).hexdigest(),
            },
            "source_location_bridge": hlo_bridge_binding,
            "stablehlo": {
                "byte_count": len(stablehlo),
                "sha256": sha256(stablehlo).hexdigest(),
            },
        },
        "memory_after_compile": memory_after_compile,
        "memory_after_execute": [helper._memory_stats(device) for device in devices],
        "memory_before_compile": memory_before,
        "numerical": classification,
        "output_artifact": {**output_identity, "filename": "outputs.npz"},
        "output_arrays": output_records,
        "performance_claim": False,
        "physical_group": observed_group,
        "predecessors": predecessor_binding,
        "projection_source_authority": projection_source_binding,
        "root_cause_fix_proven": False,
        "runtime": {
            "jax": version("jax"),
            "jaxlib": version("jaxlib"),
            "libtpu": version("libtpu"),
            "ml_dtypes": version("ml_dtypes"),
            "numpy": version("numpy"),
        },
        "schema_version": 1,
        "sealed_project_source": source_snapshot,
        "status": "NUMERICAL_ACCEPTED" if accepted else "NUMERICAL_REJECTED",
        "tpu_execution_elapsed_ns_diagnostic_only": completed_ns - started_ns,
        "tpu_numerical_execution_performed": True,
        "wk_weight": wk_identity,
    }
    helper._write_run_member_exclusive(
        run_fd, "runner.json", (_canonical(report) + "\n").encode("ascii")
    )
    os.close(run_fd)
    os.close(source_fd)
    print(
        "GATE_D_PROJECTION_CONTRACTION_PP16_"
        f"{report['status']} execution_count=1 gate_d_open=true",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
