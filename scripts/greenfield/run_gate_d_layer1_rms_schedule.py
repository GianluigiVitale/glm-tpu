#!/usr/bin/env python3
"""Bounded layer-1 RMS schedule discriminator: one exact-input, four-chip replay.

Sealed inputs are the DB548 dense-partial capture (32 BF16 down partials,
attention update, combined residual, layer-1 norm weight, accepted legacy row)
and the DB548 envelope row.  Two arms compile and execute once each: the control
arm that reproduced DB548 and the accepted-schedule arm (FP32 carry, one FP32
barrier on the ``[32,6144]`` sum, ``f32[32,6144] -> f32[32]`` variance reduce).
The certificate ``gate-d-layer1-scale-frontier-certificate.json`` proved the two
sealed rows are exact functions of the same FP32 input and differ only in the
FP32 scale; this replay asks whether the accepted reduce schedule lands the
scale in the accepted window.  No decoder, token, performance or Gate-D claim.
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
DRIVER_REPOSITORY_PATH = "scripts/greenfield/run_gate_d_layer1_rms_schedule.py"
HLO_HELPER_REPOSITORY_PATH = (
    "scripts/greenfield/acquire_gate_d_projection_contraction_pp16_hlo.py"
)
HLO_HELPER_PIN = "5cc3b4181ac5ac2b3c630130521b4f3da15a1d3e"
HLO_HELPER_SHA256 = "c660d50eb60054c9b267840230fb69bbf7259104416dc96c7b2ee01a2a14934a"
DB548_CAPTURE_SHA256 = (
    "f194d757d2f9ebe27430dfec8f828ca7588e433bddb7e8d99f9b917c5aac4298"
)
DB548_ENVELOPE_SHA256 = (
    "6cb76623bd79e1712b6c323fa786e516abd05f6f0ca51a367bc871c4a920480f"
)
CAPTURE_ARRAY_SHA256S = {
    "accepted_layer1_normalized_bfloat16_bits": (
        "9936ee1e19049b297fd205292ebc378aee41d59401bbf56497004356998d3039"
    ),
    "attention_update_bfloat16_bits": (
        "68afed86921584fb673abb11a563e359a1533210ec2483e71ee79b88c2b0bde7"
    ),
    "combined_residual_bfloat16_bits": (
        "02d045b9a0ec5ab22a711bd6a964564f707be0848683381104e83331020e31a3"
    ),
    "dense_virtual_partials_bfloat16_bits": (
        "9d9f65dddc7b622875872a33a6522c330c8fb5490c8cba14526553c211516e35"
    ),
    "layer1_input_norm_bfloat16_bits": (
        "10e34f4f99c638b29557526283205071c1ac8f81f168f4a6817e7e1def4b6c87"
    ),
}
CAPTURE_ARRAY_NAMES = {
    *CAPTURE_ARRAY_SHA256S,
    "compile_rows",
    "normalized_mlp_bfloat16_bits",
    "post_attention_m32_bfloat16_bits",
    "post_attention_residual_bfloat16_bits",
}
ENVELOPE_ARRAY_NAMES = {
    "accepted_layer1_normalized_bfloat16_bits",
    "attention_update_bfloat16_bits",
    "combined_residual_bfloat16_bits",
    "compile_rows",
    "layer1_normalized_bfloat16_bits",
    "normalized_mlp_bfloat16_bits",
    "post_attention_residual_bfloat16_bits",
}
DB548_ROW_SHA256 = "9b52a04e2852719237f4465b28665cbc213b635763303b554bb12345e99a4005"
NUMERICAL_DRIVER_INSTALL_PATH = (
    "/usr/local/libexec/glm-tpu/gate-d-layer1-rms-schedule-v3/"
    "run_gate_d_layer1_rms_schedule.py"
)
# Exact committed sources that define the bounded claim; the sealed archive
# already derives from the pinned commit, these pins add fail-closed identity.
SCHEDULE_SOURCE_SHA256S = {
    "glm_tpu/greenfield/benchmarking/gate_d_layer1_rms_schedule.py": (
        "590df9e14baebe0e65f0c535e6e2f31af3731dd2058a8d9ce1d51a47484a77b5"
    ),
    "glm_tpu/greenfield/validation/gate_d_layer1_rms_schedule.py": (
        "54fe6ee32635bfbb37bae7719fe1cadfec052d931810c5324ee11f4411959343"
    ),
    "glm_tpu/greenfield/validation/hlo_dependency.py": (
        "dd0243d1bc19aeb98e8af969dc51004ad5d316961821bd269a3a1d8071d45cc0"
    ),
    "glm_tpu/greenfield/sharding/stablehlo_dense_convolution.py": (
        "51321329a167f75d2cb92ff13868a9f294ba98239ce5a3e0209abbbfa0a8e0b9"
    ),
    "glm_tpu/greenfield/kernels/reference/rmsnorm.py": (
        "d6fca18425fb851c2ba3e1a12dacd45c8dadef43b33d6a1e6da72dbfb2eb6276"
    ),
    "glm_tpu/greenfield/kernels/stage_local.py": (
        "2c42794f0632f40b7af81d1868b9c43d50652f20295a49eed04a1cf49dc98e8a"
    ),
}
TAG_PATTERN = r"gate_d_layer1_rms_schedule_[0-9]{8}T[0-9]{15}Z"
_MAX_NPZ_MEMBER_BYTES = 64 * 1024 * 1024
_MAX_NPZ_TOTAL_BYTES = 32 * 1024 * 1024
_F_GET_SEALS = 1034
_MEMFD_SEALS = 0x0001 | 0x0002 | 0x0004 | 0x0008
_EXPECTED_ENVIRONMENT = {
    "GLM_GATE_D_LAYER1_RMS_SCHEDULE": "1",
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
        raise RuntimeError("Gate-D schedule source repository has replacement refs")
    raw = _snapshot_regular(Path(__file__))
    committed = _git_bytes("show", f"{code_pin}:{DRIVER_REPOSITORY_PATH}")
    if (
        raw != committed
        or sha256(raw).hexdigest() != expected_sha256
        or _git_text("rev-parse", f"{code_pin}^{{commit}}") != code_pin
        or _git_text("rev-parse", "HEAD") != code_pin
    ):
        raise RuntimeError(
            "Gate-D layer-1 RMS schedule driver is not the committed blob"
        )


def _load_hlo_helper(code_pin: str) -> types.ModuleType:
    path = WORKTREE / HLO_HELPER_REPOSITORY_PATH
    raw = _snapshot_regular(path)
    if (
        raw != _git_bytes("show", f"{code_pin}:{HLO_HELPER_REPOSITORY_PATH}")
        or raw != _git_bytes("show", f"{HLO_HELPER_PIN}:{HLO_HELPER_REPOSITORY_PATH}")
        or sha256(raw).hexdigest() != HLO_HELPER_SHA256
    ):
        raise RuntimeError("Gate-D compile helper bytes drifted")
    module = types.ModuleType("_gate_d_layer1_rms_schedule_hlo_helper")
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


def _verify_schedule_sources(code_pin: str) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in sorted(SCHEDULE_SOURCE_SHA256S.items()):
        raw = _snapshot_regular(WORKTREE / relative)
        if raw != _git_bytes("show", f"{code_pin}:{relative}"):
            raise RuntimeError(
                f"Gate-D schedule source is not the committed blob: {relative}"
            )
        digest = sha256(raw).hexdigest()
        if digest != expected:
            raise RuntimeError(f"Gate-D schedule source hash drifted: {relative}")
        observed[relative] = digest
    return observed


def _require_arrays(
    arrays: Mapping[str, Any], expected: Mapping[str, str], np: Any
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, expected_sha in sorted(expected.items()):
        value = np.ascontiguousarray(arrays[name])
        digest = _array_sha256(value, np)
        if value.dtype != np.dtype(np.uint16) or digest != expected_sha:
            raise RuntimeError(f"Gate-D sealed input array drifted: {name}")
        observed[name] = digest
    return observed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--expected-driver-sha256", required=True)
    parser.add_argument("--db548-capture", type=Path, required=True)
    parser.add_argument("--db548-envelope", type=Path, required=True)
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
        raise RuntimeError("Gate-D layer-1 RMS schedule replay requires Python -I -S")
    helper = _load_hlo_helper(args.expected_code_hash)
    python_runtime = helper._validate_python_runtime()
    dependency_sites = helper._validate_dependency_sites()
    compiler_environment = helper._validate_environment()
    _verify_running_source(args.expected_code_hash, args.expected_driver_sha256)
    if Path(__file__) != Path(NUMERICAL_DRIVER_INSTALL_PATH):
        raise RuntimeError("Gate-D schedule driver is not the V1 installed path")
    schedule_sources = _verify_schedule_sources(args.expected_code_hash)
    capture_raw = _snapshot_regular(args.db548_capture)
    envelope_raw = _snapshot_regular(args.db548_envelope)
    if (
        sha256(capture_raw).hexdigest() != DB548_CAPTURE_SHA256
        or sha256(envelope_raw).hexdigest() != DB548_ENVELOPE_SHA256
    ):
        raise RuntimeError("Gate-D sealed DB548 NPZ bytes drifted")
    run_fd = helper._open_inherited_run_dir(args.run_dir, args.run_dir_fd)
    source_fd, source_path, source_snapshot = helper._sealed_git_source_archive(
        args.expected_code_hash, repo=WORKTREE
    )
    helper._install_sealed_source_path(source_path)
    helper._validate_compiler_import_path(source_path)

    import jax
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P
    import ml_dtypes
    import numpy as np

    from glm_tpu.greenfield.benchmarking.gate_d_layer1_rms_schedule import (
        build_layer1_rms_schedule_arm,
    )
    from glm_tpu.greenfield.validation.gate_d_layer1_rms_schedule import (
        audit_layer1_rms_schedule_optimized_hlo,
        audit_layer1_rms_schedule_stablehlo,
        classify_layer1_rms_schedule_outputs,
    )

    project_modules = helper._verify_project_imports(source_path)
    if jax.default_backend() != "tpu" or jax.local_device_count() != 4:
        raise RuntimeError("Gate-D schedule replay requires one four-chip TPU-v4 host")
    if bool(jax.config.jax_enable_compilation_cache):
        raise RuntimeError("Gate-D schedule replay requires disabled JAX cache")
    devices = tuple(jax.local_devices())
    if any(
        int(device.process_index) != 0 or str(device.device_kind) != "TPU v4"
        for device in devices
    ):
        raise RuntimeError("Gate-D schedule replay devices are not local TPU v4")
    observed_group = {
        "coordinates": [list(device.coords) for device in devices],
        "device_ids": [int(device.id) for device in devices],
        "local_device_count_visible": int(jax.local_device_count()),
        "mesh_device_count": len(devices),
        "process_index": 0,
    }

    capture = _load_npz(capture_raw, CAPTURE_ARRAY_NAMES, np)
    envelope = _load_npz(envelope_raw, ENVELOPE_ARRAY_NAMES, np)
    input_identities = _require_arrays(capture, CAPTURE_ARRAY_SHA256S, np)
    db548_row = np.ascontiguousarray(envelope["layer1_normalized_bfloat16_bits"])
    if _array_sha256(db548_row, np) != DB548_ROW_SHA256 or not np.array_equal(
        envelope["accepted_layer1_normalized_bfloat16_bits"],
        capture["accepted_layer1_normalized_bfloat16_bits"],
    ):
        raise RuntimeError("Gate-D DB548 envelope rows drifted")
    accepted_row = np.ascontiguousarray(
        capture["accepted_layer1_normalized_bfloat16_bits"]
    )
    mesh = Mesh(np.asarray(devices), ("lp4",))
    replicated = NamedSharding(mesh, P())
    partial_sharding = NamedSharding(mesh, P("lp4", None, None, None))
    arguments = (
        jax.device_put(
            capture["dense_virtual_partials_bfloat16_bits"].view(ml_dtypes.bfloat16),
            partial_sharding,
        ),
        jax.device_put(
            capture["attention_update_bfloat16_bits"].view(ml_dtypes.bfloat16),
            replicated,
        ),
        jax.device_put(
            capture["combined_residual_bfloat16_bits"].view(ml_dtypes.bfloat16),
            replicated,
        ),
        jax.device_put(
            capture["layer1_input_norm_bfloat16_bits"].view(ml_dtypes.bfloat16),
            replicated,
        ),
    )
    dependencies_before = helper._compiler_dependency_records(source_path)
    if not dependencies_before["accelerator_device_nodes_observed_mapped"]:
        raise RuntimeError("TPU accelerator device mapping is absent before compile")
    memory_before = [helper._memory_stats(device) for device in devices]

    arms: dict[str, dict[str, Any]] = {}
    rows: dict[str, Any] = {}
    invocation_count = 0
    for arm_name, fp32_carry in (("control", False), ("schedule", True)):
        replay = jax.jit(
            build_layer1_rms_schedule_arm(mesh, fp32_carry_schedule=fp32_carry)
        )
        lowered = replay.lower(*arguments)
        stablehlo = lowered.as_text().encode("utf-8")
        compiled = lowered.compile()
        optimized_hlo = compiled.as_text().encode("utf-8")
        helper._write_run_member_exclusive(
            run_fd, f"hlo/layer1_rms_schedule_{arm_name}.stablehlo.mlir", stablehlo
        )
        helper._write_run_member_exclusive(
            run_fd,
            f"hlo/layer1_rms_schedule_{arm_name}.optimized_hlo.txt",
            optimized_hlo,
        )
        stable_contract = audit_layer1_rms_schedule_stablehlo(
            stablehlo.decode("utf-8", errors="strict"), fp32_carry_schedule=fp32_carry
        )
        optimized_contract = audit_layer1_rms_schedule_optimized_hlo(
            optimized_hlo.decode("utf-8", errors="strict"),
            fp32_carry_schedule=fp32_carry,
        )
        if not stable_contract["passed"] or not optimized_contract["passed"]:
            raise RuntimeError(
                f"Gate-D schedule {arm_name} arm HLO contract failed: "
                f"{stable_contract['violations']} {optimized_contract['violations']}"
            )
        started_ns = time.monotonic_ns()
        invocation_count += 1
        result = compiled(*arguments)
        jax.block_until_ready(result)
        completed_ns = time.monotonic_ns()
        bits = np.ascontiguousarray(np.asarray(result).reshape(6144)).view(np.uint16)
        rows[arm_name] = bits
        arms[arm_name] = {
            "fp32_carry_schedule": fp32_carry,
            "hlo": {
                "optimized_hlo": {
                    "byte_count": len(optimized_hlo),
                    "sha256": sha256(optimized_hlo).hexdigest(),
                    "structure": optimized_contract,
                },
                "stablehlo": {
                    "byte_count": len(stablehlo),
                    "sha256": sha256(stablehlo).hexdigest(),
                    "structure": stable_contract,
                },
            },
            "output_sha256": _array_sha256(bits, np),
            "tpu_execution_elapsed_ns_diagnostic_only": completed_ns - started_ns,
        }
    if invocation_count != 2:
        raise AssertionError("Gate-D schedule executable invocation count drifted")
    classification = classify_layer1_rms_schedule_outputs(
        control_bits=rows["control"],
        schedule_bits=rows["schedule"],
        db548_bits=db548_row,
        accepted_bits=accepted_row,
    )
    archived = {
        "accepted_layer1_normalized_bfloat16_bits": accepted_row,
        "control_layer1_normalized_bfloat16_bits": rows["control"],
        "db548_layer1_normalized_bfloat16_bits": db548_row,
        "schedule_layer1_normalized_bfloat16_bits": rows["schedule"],
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
        "artifact_kind": "gate_d_layer1_rms_schedule_dependencies",
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
    harness = classification["harness_admissible"]
    exact = classification["schedule_arm_exact"]
    if exact:
        status = "SCHEDULE_ARM_EXACT"
        classification_text = (
            "LAYER1_RMS_ACCEPTED_SCHEDULE_REPRODUCES_ACCEPTED_ROW;"
            "SCALE_FRONTIER_CLOSED_AT_BOUNDED_SCOPE;DECODER_UNPROVEN;GATE_D_OPEN"
        )
    else:
        status = "SCHEDULE_ARM_NONEXACT"
        classification_text = (
            "LAYER1_RMS_ACCEPTED_SCHEDULE_DOES_NOT_REPRODUCE_ACCEPTED_ROW;"
            "SAME_SHAPE_LAYOUT_EMITTER_ORDER_HYPOTHESIS_REJECTED;GATE_D_OPEN"
        )
    report = {
        "arms": arms,
        "artifact_kind": "gate_d_layer1_rms_schedule_replay",
        "claim_scope": (
            "One exact-input, two-executable, four-chip layer-1 RMS schedule "
            "witness on sealed DB548 bytes. No decoder, 8K, token, performance, "
            "or Gate-D closure claim."
        ),
        "classification": classification_text,
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
        "host_transfer_count": 2,
        "input_arrays": input_identities,
        "memory_after_compile_and_execute": [
            helper._memory_stats(device) for device in devices
        ],
        "memory_before_compile": memory_before,
        "numerical": classification,
        "output_artifact": {**output_identity, "filename": "outputs.npz"},
        "output_arrays": output_records,
        "performance_claim": False,
        "physical_group": observed_group,
        "root_cause_fix_proven": False,
        "runtime": {
            "jax": version("jax"),
            "jaxlib": version("jaxlib"),
            "libtpu": version("libtpu"),
            "ml_dtypes": version("ml_dtypes"),
            "numpy": version("numpy"),
        },
        "schedule_source_sha256s": schedule_sources,
        "schema_version": 1,
        "sealed_db548_capture_sha256": DB548_CAPTURE_SHA256,
        "sealed_db548_envelope_sha256": DB548_ENVELOPE_SHA256,
        "sealed_project_source": source_snapshot,
        "status": status,
        "tpu_numerical_execution_performed": True,
    }
    if not harness:
        report["classification"] = (
            "LAYER1_RMS_CONTROL_ARM_DID_NOT_REPRODUCE_DB548;REPLAY_REFUSED;GATE_D_OPEN"
        )
        report["status"] = "HARNESS_REFUSED"
    helper._write_run_member_exclusive(
        run_fd, "runner.json", (_canonical(report) + "\n").encode("ascii")
    )
    os.close(run_fd)
    os.close(source_fd)
    if not harness:
        # The control arm is the harness: without an exact DB548 reproduction
        # the run is a diagnostic, never a result.
        raise RuntimeError(
            "Gate-D schedule control arm did not reproduce the protected DB548 row"
        )
    print(
        f"GATE_D_LAYER1_RMS_SCHEDULE_{status} execution_count=2 gate_d_open=true",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
