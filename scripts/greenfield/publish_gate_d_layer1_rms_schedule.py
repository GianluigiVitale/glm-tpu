#!/usr/bin/env python3
"""Append-only publisher for the bounded layer-1 RMS schedule replay (V1)."""

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
SOURCE_PATH = "scripts/greenfield/publish_gate_d_layer1_rms_schedule.py"
BASE_PATH = "scripts/greenfield/publish_gate_d_projection_contraction_pp16_hlo.py"
BASE_PIN = "986378238ac6458307aea69ef1f5e12bf82bc020"
BASE_SHA256 = "f3f20a01fd37bb82988cd77f69fa7b0a780d120568bab0db4162f42e7855bc97"
BUCKET_NAME = "driftbench-dsv4-uc"
REMOTE_ROOT = "results/greenfield/glm52/gate_d_layer1_rms_schedule/"
TAG_PATTERN_TEXT = r"gate_d_layer1_rms_schedule_[0-9]{8}T[0-9]{15}Z"
TAG_PATTERN = re.compile(TAG_PATTERN_TEXT)
ARMS = ("control", "schedule")
HLO_FORBIDDEN_TEXT = (
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
    "stablehlo.all_reduce",
    "stablehlo.all_to_all",
    "stablehlo.collective_",
    "stablehlo.custom_call",
    "stablehlo.infeed",
    "stablehlo.outfeed",
)
STABLEHLO_REQUIRED_TEXT = (
    "mhlo.num_partitions = 4",
    "tensor<4x8x1x6144xbf16>",
    "tensor<32x6144xf32>",
    "stablehlo.rsqrt",
)
STABLEHLO_SCHEDULE_REQUIRED_TEXT = (
    "stablehlo.optimization_barrier",
    "tensor<32xf32>",
)
DB548_CAPTURE_SHA256 = "f194d757d2f9ebe27430dfec8f828ca7588e433bddb7e8d99f9b917c5aac4298"
DB548_ENVELOPE_SHA256 = "6cb76623bd79e1712b6c323fa786e516abd05f6f0ca51a367bc871c4a920480f"
DB548_ROW_SHA256 = "9b52a04e2852719237f4465b28665cbc213b635763303b554bb12345e99a4005"
ACCEPTED_ROW_SHA256 = "9936ee1e19049b297fd205292ebc378aee41d59401bbf56497004356998d3039"
INPUT_ARRAY_SHA256S = {
    "accepted_layer1_normalized_bfloat16_bits": ACCEPTED_ROW_SHA256,
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
# Exact committed sources of the schedule arms and their validation, as pinned
# by the immutable driver; the runner report must carry the same identities.
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
EXPECTED_OUTPUT_LAYOUT = {
    "accepted_layer1_normalized_bfloat16_bits": ((6144,), "<u2", 2),
    "control_layer1_normalized_bfloat16_bits": ((6144,), "<u2", 2),
    "db548_layer1_normalized_bfloat16_bits": ((6144,), "<u2", 2),
    "schedule_layer1_normalized_bfloat16_bits": ((6144,), "<u2", 2),
}
EXPECTED_PHYSICAL_GROUP = {
    "coordinates": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
    "device_ids": [0, 1, 2, 3],
    "local_device_count_visible": 4,
    "mesh_device_count": 4,
    "process_index": 0,
}
STATUS_EXACT = "SCHEDULE_ARM_EXACT"
STATUS_NONEXACT = "SCHEDULE_ARM_NONEXACT"
CLASSIFICATION_EXACT = (
    "LAYER1_RMS_ACCEPTED_SCHEDULE_REPRODUCES_ACCEPTED_ROW;"
    "SCALE_FRONTIER_CLOSED_AT_BOUNDED_SCOPE;DECODER_UNPROVEN;GATE_D_OPEN"
)
CLASSIFICATION_NONEXACT = (
    "LAYER1_RMS_ACCEPTED_SCHEDULE_DOES_NOT_REPRODUCE_ACCEPTED_ROW;"
    "SAME_SHAPE_LAYOUT_EMITTER_ORDER_HYPOTHESIS_REJECTED;GATE_D_OPEN"
)
CLAIM_SCOPE = (
    "One exact-input, two-executable, four-chip layer-1 RMS schedule witness on "
    "sealed DB548 bytes. No decoder, 8K, token, performance, or Gate-D closure claim."
)
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
    "evidence.json",
    "hlo/layer1_rms_schedule_control.optimized_hlo.txt",
    "hlo/layer1_rms_schedule_control.stablehlo.mlir",
    "hlo/layer1_rms_schedule_schedule.optimized_hlo.txt",
    "hlo/layer1_rms_schedule_schedule.stablehlo.mlir",
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
        raise RuntimeError("layer-1 RMS schedule publisher found replacement refs")
    raw = _snapshot(Path(__file__))
    if (
        _git_bytes("rev-parse", "HEAD").decode().strip() != code_pin
        or _git_bytes("rev-parse", f"{code_pin}^{{commit}}").decode().strip()
        != code_pin
        or raw != _git_bytes("show", f"{code_pin}:{SOURCE_PATH}")
        or sha256(raw).hexdigest() != expected_sha256
    ):
        raise RuntimeError("layer-1 RMS schedule publisher is not the committed blob")


def _load_base(code_pin: str) -> types.ModuleType:
    path = REPO / BASE_PATH
    raw = _snapshot(path)
    if (
        raw != _git_bytes("show", f"{code_pin}:{BASE_PATH}")
        or raw != _git_bytes("show", f"{BASE_PIN}:{BASE_PATH}")
        or sha256(raw).hexdigest() != BASE_SHA256
    ):
        raise RuntimeError("layer-1 RMS schedule publisher base bytes drifted")
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
    result["artifact_kind"] = "gate_d_layer1_rms_schedule_publisher_runtime"
    return result


def _require_preterminal(base: Any, run_fd: int) -> None:
    base.require_preterminal(run_fd)
    for name in ("NUMERICAL_RESULT", "terminal_upload_receipt.json"):
        try:
            os.stat(name, dir_fd=run_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        raise RuntimeError("layer-1 RMS schedule terminal publication already began")


def _expected_status(runner: Mapping[str, Any]) -> tuple[bool, str, str]:
    numerical = runner.get("numerical")
    exact = (
        numerical.get("schedule_arm_exact") if isinstance(numerical, Mapping) else None
    )
    harness = (
        numerical.get("harness_admissible") if isinstance(numerical, Mapping) else None
    )
    if type(exact) is not bool or harness is not True:
        raise RuntimeError("layer-1 RMS schedule classification is absent or refused")
    status = STATUS_EXACT if exact else STATUS_NONEXACT
    classification = CLASSIFICATION_EXACT if exact else CLASSIFICATION_NONEXACT
    return exact, status, classification


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


def _validate_output_npz(raw: bytes, runner: Mapping[str, Any], numerical: Mapping[str, Any]) -> bool:
    """Re-derive the schedule verdict from archived bytes alone (no numpy)."""

    expected_names = {f"{name}.npy" for name in EXPECTED_OUTPUT_LAYOUT}
    runner_arrays = runner.get("output_arrays")
    if not isinstance(runner_arrays, Mapping) or set(runner_arrays) != set(
        EXPECTED_OUTPUT_LAYOUT
    ):
        raise RuntimeError("layer-1 RMS schedule output records drifted")
    payloads: dict[str, bytes] = {}
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        names = archive.namelist()
        if set(names) != expected_names or len(names) != len(expected_names):
            raise RuntimeError("layer-1 RMS schedule output NPZ members drifted")
        for name in sorted(names):
            member = archive.getinfo(name)
            if member.file_size > 1 << 20 or member.compress_type != zipfile.ZIP_STORED:
                raise RuntimeError("layer-1 RMS schedule output member is unsafe")
            shape, dtype, payload = _parse_npy(archive.read(name))
            key = name[: -len(".npy")]
            expected_shape, expected_dtype, item = EXPECTED_OUTPUT_LAYOUT[key]
            record = runner_arrays[key]
            if (
                shape != expected_shape
                or dtype != expected_dtype
                or len(payload) != item * 6144
                or record.get("array_sha256") != sha256(payload).hexdigest()
                or record.get("shape") != list(expected_shape)
                or record.get("storage_dtype") != expected_dtype
            ):
                raise RuntimeError(f"layer-1 RMS schedule output drifted: {key}")
            payloads[key] = payload
    db548 = payloads["db548_layer1_normalized_bfloat16_bits"]
    accepted = payloads["accepted_layer1_normalized_bfloat16_bits"]
    control = payloads["control_layer1_normalized_bfloat16_bits"]
    schedule = payloads["schedule_layer1_normalized_bfloat16_bits"]
    if (
        sha256(db548).hexdigest() != DB548_ROW_SHA256
        or sha256(accepted).hexdigest() != ACCEPTED_ROW_SHA256
    ):
        raise RuntimeError("layer-1 RMS schedule reference rows drifted")
    mismatch_count = sum(
        1
        for index in range(6144)
        if db548[2 * index : 2 * index + 2] != accepted[2 * index : 2 * index + 2]
    )
    if mismatch_count != 1 or db548[2 * 2795 : 2 * 2795 + 2] == accepted[2 * 2795 : 2 * 2795 + 2]:
        raise RuntimeError("layer-1 RMS schedule references do not differ only at 2795")
    if control != db548:
        raise RuntimeError("layer-1 RMS schedule control arm did not reproduce DB548")
    exact = schedule == accepted
    control_record = numerical.get("control_vs_db548")
    schedule_record = numerical.get("schedule_vs_accepted")
    if (
        not isinstance(control_record, Mapping)
        or control_record.get("elementwise_exact") is not True
        or control_record.get("mismatch_count") != 0
        or control_record.get("observed_sha256") != sha256(control).hexdigest()
        or not isinstance(schedule_record, Mapping)
        or schedule_record.get("elementwise_exact") is not exact
        or schedule_record.get("observed_sha256") != sha256(schedule).hexdigest()
        or schedule_record.get("expected_sha256") != ACCEPTED_ROW_SHA256
        or numerical.get("schedule_arm_exact") is not exact
        or numerical.get("harness_admissible") is not True
    ):
        raise RuntimeError("layer-1 RMS schedule numerical records disagree with bytes")
    return exact


def _validate_dependencies(
    base: Any,
    runner: Mapping[str, Any],
    dependencies_raw: bytes,
    code_pin: str,
) -> None:
    dependencies = json.loads(dependencies_raw)
    identity = runner.get("compiler_dependency_manifest")
    if not isinstance(dependencies, Mapping) or not isinstance(identity, Mapping):
        raise TypeError("layer-1 RMS schedule dependency authority is absent")
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
        != "gate_d_layer1_rms_schedule_dependencies"
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
        raise RuntimeError("layer-1 RMS schedule dependency authority drifted")


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
        raise TypeError("layer-1 RMS schedule runner report is not an object")
    exact, status, classification = _expected_status(runner)
    expected_keys = {
        "arms",
        "artifact_kind",
        "claim_scope",
        "classification",
        "code_hash",
        "compiled_executable_invocation_count",
        "compiler_dependency_manifest",
        "gate_d_closed",
        "host_transfer_count",
        "input_arrays",
        "memory_after_compile_and_execute",
        "memory_before_compile",
        "numerical",
        "output_artifact",
        "output_arrays",
        "performance_claim",
        "physical_group",
        "root_cause_fix_proven",
        "runtime",
        "schedule_source_sha256s",
        "schema_version",
        "sealed_db548_capture_sha256",
        "sealed_db548_envelope_sha256",
        "sealed_project_source",
        "status",
        "tpu_numerical_execution_performed",
    }
    if (
        set(runner) != expected_keys
        or runner.get("artifact_kind") != "gate_d_layer1_rms_schedule_replay"
        or runner.get("schema_version") != 1
        or runner.get("code_hash") != code_pin
        or runner.get("status") != status
        or runner.get("classification") != classification
        or runner.get("claim_scope") != CLAIM_SCOPE
        or runner.get("compiled_executable_invocation_count") != 2
        or runner.get("host_transfer_count") != 2
        or runner.get("tpu_numerical_execution_performed") is not True
        or runner.get("gate_d_closed") is not False
        or runner.get("root_cause_fix_proven") is not False
        or runner.get("performance_claim") is not False
        or runner.get("physical_group") != EXPECTED_PHYSICAL_GROUP
        or runner.get("schedule_source_sha256s") != SCHEDULE_SOURCE_SHA256S
        or runner.get("input_arrays") != INPUT_ARRAY_SHA256S
        or runner.get("sealed_db548_capture_sha256") != DB548_CAPTURE_SHA256
        or runner.get("sealed_db548_envelope_sha256") != DB548_ENVELOPE_SHA256
    ):
        raise RuntimeError("layer-1 RMS schedule runner claim boundary drifted")
    _validate_dependencies(base, runner, dependencies_raw, code_pin)
    numerical = runner["numerical"]
    if set(numerical) != {
        "accepted_layer1_normalized_sha256",
        "control_vs_db548",
        "db548_layer1_normalized_sha256",
        "harness_admissible",
        "schedule_arm_exact",
        "schedule_vs_accepted",
        "schedule_vs_db548",
    } or (
        numerical.get("accepted_layer1_normalized_sha256") != ACCEPTED_ROW_SHA256
        or numerical.get("db548_layer1_normalized_sha256") != DB548_ROW_SHA256
    ):
        raise RuntimeError("layer-1 RMS schedule output classification drifted")
    arms = runner.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
        raise RuntimeError("layer-1 RMS schedule arm catalogue drifted")
    hlo_payload: dict[str, bytes] = {}
    for arm in ARMS:
        record = arms[arm]
        if (
            not isinstance(record, Mapping)
            or record.get("fp32_carry_schedule") is not (arm == "schedule")
            or type(record.get("tpu_execution_elapsed_ns_diagnostic_only")) is not int
            or record["tpu_execution_elapsed_ns_diagnostic_only"] <= 0
            or not isinstance(record.get("hlo"), Mapping)
            or set(record["hlo"]) != {"optimized_hlo", "stablehlo"}
        ):
            raise RuntimeError(f"layer-1 RMS schedule {arm} arm record drifted")
        for kind, suffix in (
            ("optimized_hlo", "optimized_hlo.txt"),
            ("stablehlo", "stablehlo.mlir"),
        ):
            relative = f"hlo/layer1_rms_schedule_{arm}.{suffix}"
            raw = base.snapshot_member(run_fd, relative)
            text = raw.decode("utf-8", errors="strict")
            lowered = text.lower()
            identity = record["hlo"].get(kind, {})
            structure = (
                identity.get("structure") if isinstance(identity, Mapping) else None
            )
            if (
                not isinstance(identity, Mapping)
                or identity.get("byte_count") != len(raw)
                or identity.get("sha256") != sha256(raw).hexdigest()
                or not isinstance(structure, Mapping)
                or structure.get("passed") is not True
                or structure.get("violations") != []
            ):
                raise RuntimeError(f"layer-1 RMS schedule {arm} {kind} identity drifted")
            if kind == "optimized_hlo":
                if (
                    "num_partitions=4" not in text
                    or any(token in lowered for token in HLO_FORBIDDEN_TEXT)
                    or text.count("all-gather-start(") + text.count("all-gather(") != 1
                    or structure.get("collective_count") != 1
                    or structure.get("fp32_carry_schedule") is not (arm == "schedule")
                    or (
                        arm == "schedule"
                        and len(structure.get("accepted_scheduled_reduction_values") or [])
                        != 1
                    )
                ):
                    raise RuntimeError(
                        f"layer-1 RMS schedule {arm} optimized HLO contract drifted"
                    )
            elif (
                any(item not in text for item in STABLEHLO_REQUIRED_TEXT)
                or any(token in lowered for token in STABLEHLO_FORBIDDEN_TEXT)
                or text.count("stablehlo.all_gather") != 1
                or (
                    arm == "schedule"
                    and any(item not in text for item in STABLEHLO_SCHEDULE_REQUIRED_TEXT)
                )
                or structure.get("split_layer1_rms") is not (arm == "schedule")
            ):
                raise RuntimeError(
                    f"layer-1 RMS schedule {arm} StableHLO contract drifted"
                )
            hlo_payload[relative] = raw
    output_raw = base.snapshot_member(run_fd, "outputs.npz")
    if runner.get("output_artifact") != {
        "byte_count": len(output_raw),
        "filename": "outputs.npz",
        "sha256": sha256(output_raw).hexdigest(),
    }:
        raise RuntimeError("layer-1 RMS schedule output artifact drifted")
    if _validate_output_npz(output_raw, runner, numerical) is not exact:
        raise RuntimeError("layer-1 RMS schedule status differs from output bytes")
    mirror_raw = base.snapshot_member(run_fd, "mirror.sha256", limit=2 << 20)
    base._validate_mirror_replay(mirror_raw, code_pin)
    sync_raw = base.snapshot_member(run_fd, "sync.txt", limit=1 << 20)
    expected_sync = (
        f"SYNC_OK {os.uname().nodename} {code_pin} origin_and_same_region_mirror\n"
    ).encode("ascii")
    if sync_raw != expected_sync:
        raise RuntimeError("layer-1 RMS schedule host code authority drifted")
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
        "artifact_kind": "gate_d_layer1_rms_schedule_summary",
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
        "artifact_kind": "gate_d_layer1_rms_schedule_local_evidence",
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
        raise RuntimeError("layer-1 RMS schedule success inventory drifted")
    if base.local_members(run_fd) != set(SUCCESS_PAYLOAD) | {"orchestrator.log"}:
        raise RuntimeError("layer-1 RMS schedule local preterminal inventory drifted")
    return payload


def _result_authority_line(
    status: str,
    marker_sha256: str,
    terminal_record: Mapping[str, Any],
) -> str:
    generation = terminal_record.get("generation")
    terminal_sha256 = terminal_record.get("sha256")
    if (
        status not in {STATUS_EXACT, STATUS_NONEXACT}
        or not re.fullmatch(r"[0-9a-f]{64}", marker_sha256)
        or type(generation) is not str
        or not re.fullmatch(r"[0-9]+", generation)
        or type(terminal_sha256) is not str
        or not re.fullmatch(r"[0-9a-f]{64}", terminal_sha256)
    ):
        raise RuntimeError("layer-1 RMS schedule result authority drifted")
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
                "artifact_kind": "gate_d_layer1_rms_schedule_remote_ledger",
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
            raise RuntimeError("layer-1 RMS schedule preterminal remote set drifted")
        marker = {
            "artifact_kind": "gate_d_layer1_rms_schedule_result",
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
            raise RuntimeError("layer-1 RMS schedule local terminal inventory drifted")
        # This must remain the final remote mutation in the successful result path.
        terminal_record = base._upload_bound(
            bucket, prefix + "NUMERICAL_RESULT", terminal_raw
        )
        terminal_record["path"] = "NUMERICAL_RESULT"
        base._replay_bound(bucket, prefix + "NUMERICAL_RESULT", terminal_record)
        if base._observed_names(bucket, prefix) != expected_remote | {
            "NUMERICAL_RESULT"
        }:
            raise RuntimeError("layer-1 RMS schedule terminal remote set drifted")
        base.write_member_exclusive(
            run_fd,
            "terminal_upload_receipt.json",
            base._canonical(
                {
                    "artifact_kind": "gate_d_layer1_rms_schedule_terminal_receipt",
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
                "layer-1 RMS schedule terminal receipt inventory drifted"
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
                "artifact_kind": "gate_d_layer1_rms_schedule_failure",
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
            raise RuntimeError("layer-1 RMS schedule diagnostic inventory is unsafe")
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
            raise RuntimeError("layer-1 RMS schedule diagnostic object set drifted")
        ledger_raw = base._canonical(
            {
                "artifact_kind": "gate_d_layer1_rms_schedule_diagnostic_ledger",
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
            raise RuntimeError("layer-1 RMS schedule diagnostic terminal set drifted")
        base.write_member_exclusive(
            run_fd,
            "diagnostic_upload_receipt.json",
            base._canonical(
                {
                    "artifact_kind": "gate_d_layer1_rms_schedule_diagnostic_terminal_receipt",
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
        raise RuntimeError("layer-1 RMS schedule publisher requires run-directory fd 7")
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
                raise RuntimeError("layer-1 RMS schedule log append exceeds limit")
            base.append_preterminal_log(run_fd, payload)
        finally:
            os.close(run_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
