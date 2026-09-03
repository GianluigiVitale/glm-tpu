#!/usr/bin/env python3
"""Append-only publisher for the original-DB518 chunk-0 discriminator."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import stat
import subprocess
import sys
import types
import zipfile
from collections.abc import Mapping
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

REPO = Path("/home/gianl/glm-tpu-topology-rewrite")
RUN_ROOT = Path("/home/gianl/gate-d-runs")
INSTALL_PATH = Path(
    "/usr/local/libexec/glm-tpu/gate-d-original-db518-prompt-key-v1/"
    "publish_gate_d_original_db518_prompt_key_chunk0.py")
SOURCE_PATH = (
    "scripts/greenfield/publish_gate_d_original_db518_prompt_key_chunk0.py")
CONTRACT_PATH = Path(
    "/usr/local/libexec/glm-tpu/gate-d-original-db518-prompt-key-v1/"
    "original_db518_prompt_key.py")
CONTRACT_SOURCE_PATH = (
    "glm_tpu/greenfield/validation/original_db518_prompt_key.py")
CONTRACT_SHA256 = (
    "056500262426a07b5258c41d8750c09f9eee2301073f884575db794772c4021c")
PROMPT_CONTRACT_SOURCE_PATH = (
    "glm_tpu/greenfield/validation/prompt_index_cache.py")
PARENT_PATH = "scripts/greenfield/publish_gate_d_layer1_prompt_chunk0_geometry.py"
PARENT_PIN = "d9c624a77590b8933ccdabc55ea937113138835f"
PARENT_SHA256 = "8b1e05534419661c9e69207c1cfab2794beaec03007f0fcb15a184977a33f15d"
BUCKET_NAME = "driftbench-dsv4-uc"
REMOTE_ROOT = "results/greenfield/glm52/original_db518_prompt_key_chunk0/"
TAG_PATTERN = re.compile(
    r"greenfield_original_db518_prompt_key_chunk0_[0-9]{8}T[0-9]{15}Z")
STATUS_EXACT = "ORIGINAL_DB518_CHUNK0_EXACT"
STATUS_NONEXACT = "ORIGINAL_DB518_CHUNK0_NONEXACT"
CLASSIFICATION_EXACT = ("ORIGINAL_DB518_PRODUCER_BOUNDARY_REPRODUCED_BITWISE;"
                        "CONSUMER_UNPROVEN;DECODER_UNPROVEN;GATE_D_OPEN")
CLASSIFICATION_NONEXACT = ("ORIGINAL_DB518_PRODUCER_BOUNDARY_NOT_REPRODUCED;"
                           "STOP_BEFORE_CONSUMER;DECODER_UNPROVEN;GATE_D_OPEN")
EXPECTED_CHUNK_BITS_SHA256 = (
    "96d261cbef56bccc20c3d2bee275b995f168e0ff7609713baae21d0a6ea3887c")
EXPECTED_WK_SHA256 = (
    "d680f7b1c2fed426174c41ee9153d9db47f5db8ec0e803e20d8853ec1f483469")
EXPECTED_ROW_MAPPING_SHA256 = (
    "0f87a6717bb33fcd8b7c0d0475151e60377a959fec841bcb1a33bd5cffca46fd")
EXPECTED_POSITIONS_SHA256 = (
    "cc76b029564c7257d6c27e130546ac40603f1e3ae5efc1106b2656294f599ec5")
EXPECTED_INPUT_MANIFEST_SHA256 = (
    "574f3553e6106a997e780b6b2a321bce86ad358b19c38989e84e2a4914b73141")
EXPECTED_CACHE_MANIFEST_SHA256 = (
    "acc631e71148922448eb03c839f71544c80ca00cea47b639bdd80eb34567fdab")
EXPECTED_DB518_CACHE_SHA256 = (
    "3808d502f3ea1829bf12ab7585d66f15dd83bf640657a17c35daabf5ab1859d1")
EXPECTED_DB518_CODE_HASH = "86243115452920fe4244bb77a9bbf4c44110aeab"
EXPECTED_DB518_COMPARISON_MANIFEST_SHA256 = (
    "1d80d088561181a63e734a91cd0124c4011cfc4151198488c052740050d66fe5")
EXPECTED_DB518_TAG = "greenfield_layer0_prompt_key_norm_m64_20260809T122010714691723Z"
SUCCESS_PAYLOAD = (
    "census_post.txt",
    "census_pre.txt",
    "evidence.json",
    "hlo/original_db518_chunk0.optimized_hlo.txt",
    "hlo/original_db518_chunk0.stablehlo.mlir",
    "hlo/wk_decode.optimized_hlo.txt",
    "hlo/wk_decode.stablehlo.mlir",
    "hlo/wk_promote.optimized_hlo.txt",
    "hlo/wk_promote.stablehlo.mlir",
    "mirror.sha256",
    "orchestrator.sealed.log",
    "producer_arrays.npz",
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
EXPECTED_OUTPUT_LAYOUT = {
    "accepted_chunk_bits": ((2048, 128), "<u2"),
    "candidate_chunk_bits": ((2048, 128), "<u2"),
    "embedding_rows": ((2048, ), "<i4"),
    "positions": ((2048, ), "<i4"),
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
        [
            "/usr/bin/git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-c",
            "core.attributesFile=/dev/null",
            "-C",
            str(REPO),
            *arguments,
        ],
        env=_GIT_ENVIRONMENT,
    )


def _snapshot(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(f"unsafe publisher source: {path}")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(block)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (len(raw) != before.st_size or
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
             before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size,
                                     after.st_mtime_ns, after.st_ctime_ns) or
            (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)):
            raise RuntimeError(
                f"publisher source changed while reading: {path}")
        return raw
    finally:
        os.close(descriptor)


def _load_parent(code_pin: str) -> types.ModuleType:
    raw = _snapshot(REPO / PARENT_PATH)
    if (raw != _git_bytes("show", f"{code_pin}:{PARENT_PATH}")
            or raw != _git_bytes("show", f"{PARENT_PIN}:{PARENT_PATH}")
            or sha256(raw).hexdigest() != PARENT_SHA256):
        raise RuntimeError("original-DB518 publisher parent bytes drifted")
    module = types.ModuleType("_gate_d_original_db518_publisher_parent")
    module.__file__ = str(REPO / PARENT_PATH)
    module.__package__ = None
    exec(compile(raw, module.__file__, "exec"), module.__dict__)  # noqa: S102
    module.REPO = REPO
    module.RUN_ROOT = RUN_ROOT
    module.BUCKET_NAME = BUCKET_NAME
    module.REMOTE_ROOT = REMOTE_ROOT
    module.TAG_PATTERN = TAG_PATTERN
    module.WRAPPER_WRITE_MEMBERS = set(WRAPPER_WRITE_MEMBERS)
    module._git_bytes = lambda *args: _git_bytes(*args)
    return module


def _verify_running_source(code_pin: str, expected_sha256: str) -> None:
    if Path(__file__) != INSTALL_PATH:
        raise RuntimeError(
            "publisher is not executing from the immutable capsule")
    raw = _snapshot(INSTALL_PATH)
    metadata = os.stat(INSTALL_PATH, follow_symlinks=False)
    if (metadata.st_uid != 0 or metadata.st_gid != 0
            or stat.S_IMODE(metadata.st_mode) != 0o555
            or os.listxattr(INSTALL_PATH, follow_symlinks=False) or _git_bytes(
                "for-each-ref", "--format=%(refname)", "refs/replace")
            or _git_bytes("rev-parse", "HEAD").decode().strip() != code_pin
            or raw != _git_bytes("show", f"{code_pin}:{SOURCE_PATH}")
            or sha256(raw).hexdigest() != expected_sha256):
        raise RuntimeError("publisher is not the committed immutable blob")


def _load_contract(code_pin: str) -> types.ModuleType:
    raw = _snapshot(CONTRACT_PATH)
    if (sha256(raw).hexdigest() != CONTRACT_SHA256 or raw != _git_bytes(
            "show", f"{code_pin}:{CONTRACT_SOURCE_PATH}")):
        raise RuntimeError("original-DB518 contract bytes drifted")
    prompt_raw = _git_bytes("show",
                            f"{code_pin}:{PROMPT_CONTRACT_SOURCE_PATH}")
    package_names = (
        "glm_tpu",
        "glm_tpu.greenfield",
        "glm_tpu.greenfield.validation",
    )
    for name in package_names:
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = []  # type: ignore[attr-defined]
            package.__package__ = name.rpartition(".")[0]
            sys.modules[name] = package
    prompt_name = "glm_tpu.greenfield.validation.prompt_index_cache"
    prompt = types.ModuleType(prompt_name)
    prompt.__file__ = str(REPO / PROMPT_CONTRACT_SOURCE_PATH)
    prompt.__package__ = "glm_tpu.greenfield.validation"
    sys.modules[prompt_name] = prompt
    exec(compile(prompt_raw, prompt.__file__, "exec"),
         prompt.__dict__)  # noqa: S102
    contract_name = "glm_tpu.greenfield.validation.original_db518_prompt_key"
    contract = types.ModuleType(contract_name)
    contract.__file__ = str(CONTRACT_PATH)
    contract.__package__ = "glm_tpu.greenfield.validation"
    sys.modules[contract_name] = contract
    exec(compile(raw, contract.__file__, "exec"),
         contract.__dict__)  # noqa: S102
    return contract


def _parse_npz(parent: Any, raw: bytes) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        names = archive.namelist()
        expected = {f"{name}.npy" for name in EXPECTED_OUTPUT_LAYOUT}
        if set(names) != expected or len(names) != len(expected):
            raise RuntimeError("original-DB518 NPZ inventory drifted")
        total = 0
        for name in names:
            member = archive.getinfo(name)
            total += member.file_size
            if (member.compress_type != zipfile.ZIP_STORED
                    or member.flag_bits & 0x1 or member.file_size > 2 << 20
                    or member.compress_size != member.file_size
                    or total > 4 << 20):
                raise RuntimeError("original-DB518 NPZ member is unsafe")
            shape, dtype, payload = parent._parse_npy(archive.read(name))
            key = name.removesuffix(".npy")
            expected_shape, expected_dtype = EXPECTED_OUTPUT_LAYOUT[key]
            elements = 1
            for dimension in expected_shape:
                elements *= dimension
            width = 2 if expected_dtype == "<u2" else 4
            if shape != expected_shape or dtype != expected_dtype or len(
                    payload) != elements * width:
                raise RuntimeError(f"original-DB518 NPZ member drifted: {key}")
            payloads[key] = payload
    return payloads


def _mismatch_counts(left: bytes, right: bytes) -> tuple[int, int]:
    if len(left) != len(right) or len(left) != 2048 * 128 * 2:
        raise RuntimeError("original-DB518 comparison geometry drifted")
    lanes = 0
    rows = 0
    row_bytes = 128 * 2
    for start in range(0, len(left), row_bytes):
        row_lanes = sum(left[index:index + 2] != right[index:index + 2]
                        for index in range(start, start + row_bytes, 2))
        lanes += row_lanes
        rows += row_lanes != 0
    return rows, lanes


def _validate_helper_hlo(parent: Any, optimized: str, stablehlo: str,
                         kind: str) -> None:
    nodes, root = parent._entry_graph(optimized)
    parameters = {
        name: node
        for name, node in nodes.items() if node["opcode"] == "parameter"
    }
    live = parent._ancestors(nodes, root)
    shapes = sorted(node["shapes"] for node in parameters.values())
    forbidden = (
        "all-gather(",
        "all-reduce(",
        "all-to-all(",
        "collective-permute(",
        "reduce-scatter(",
        "host_callback",
        "python_callback",
    )
    if any(token in optimized or token in stablehlo for token in forbidden):
        raise RuntimeError(
            f"{kind} helper contains forbidden communication/callback")
    if kind == "decode":
        expected = sorted((("u8", (128, 6144)), ("f32", (1, 48))))
        if shapes != expected or nodes[root]["shapes"] != (("bf16",
                                                            (128, 6144)), ):
            raise RuntimeError("wk decode helper boundary drifted")
        if not set(parameters).issubset(live):
            raise RuntimeError("wk decode helper has a dead input")
    elif kind == "promote":
        if (shapes != [("bf16", (128, 6144))]
                or nodes[root]["opcode"] != "convert"
                or nodes[root]["shapes"] != (("f32", (128, 6144)), )
                or tuple(nodes[root]["operands"]) != tuple(parameters)):
            raise RuntimeError(
                "wk promotion helper is not the exact one-op boundary")
    else:
        raise RuntimeError("unknown original-DB518 helper kind")


def _validate_runner_and_outputs(
        parent: Any, contract: Any, base: Any, run_fd: int, code_pin: str,
        run_tag: str) -> tuple[bytes, Mapping[str, Any], bool]:
    runner_raw = base.snapshot_member(run_fd, "runner.json", limit=4 << 20)
    runner = json.loads(runner_raw)
    expected_probe_sha = sha256(
        _git_bytes(
            "show",
            f"{code_pin}:scripts/greenfield/"
            "probe_original_db518_prompt_key_chunk0.py",
        )).hexdigest()
    expected_runner_keys = {
        "accepted_chunk_bits_sha256",
        "adapted_wk",
        "artifact_kind",
        "backend",
        "candidate_chunk_bits_sha256",
        "claim_scope",
        "code_hash",
        "compile_seconds",
        "device",
        "execute_seconds",
        "host_transfer_count_after_completion",
        "hlo",
        "import_closure_module_count",
        "input_authority",
        "memory_after",
        "memory_before",
        "mismatched_lanes",
        "mismatched_rows",
        "original_db518_chunk0_exact",
        "performance_claim",
        "producer_invocation_count",
        "provenance",
        "run_tag",
        "source_sha256",
        "status",
    }
    if (type(runner) is not dict
            or runner_raw != (json.dumps(runner, indent=2, sort_keys=True) +
                              "\n").encode("ascii")
            or set(runner) != expected_runner_keys
            or type(runner.get("compile_seconds")) not in {int, float}
            or runner["compile_seconds"] < 0
            or type(runner.get("execute_seconds")) not in {int, float}
            or runner["execute_seconds"] < 0
            or type(runner.get("import_closure_module_count")) is not int
            or runner["import_closure_module_count"] < 50
            or not isinstance(runner.get("provenance"), Mapping)
            or runner["provenance"].get("probe_sha256") != expected_probe_sha
            or runner.get("source_sha256") != expected_probe_sha
            or runner.get("run_tag") != run_tag or runner.get("claim_scope")
            != ("One original-DB518 layer-0 prompt-key producer chunk only; "
                "no consumer, decoder, Gate-D or performance claim.")):
        raise RuntimeError("original-DB518 runner claim boundary drifted")
    arrays_raw = base.snapshot_member(run_fd,
                                      "producer_arrays.npz",
                                      limit=4 << 20)
    payloads = _parse_npz(parent, arrays_raw)
    accepted = payloads["accepted_chunk_bits"]
    candidate = payloads["candidate_chunk_bits"]
    row_mapping_sha = sha256(payloads["embedding_rows"]).hexdigest()
    positions_sha = sha256(payloads["positions"]).hexdigest()
    rows, lanes = _mismatch_counts(candidate, accepted)
    candidate_sha = sha256(candidate).hexdigest()
    exact = rows == 0 and lanes == 0 and candidate_sha == EXPECTED_CHUNK_BITS_SHA256
    if (sha256(accepted).hexdigest() != EXPECTED_CHUNK_BITS_SHA256
            or runner.get("artifact_kind")
            != "gate_d_original_db518_prompt_key_chunk0_discriminator"
            or runner.get("accepted_chunk_bits_sha256")
            != EXPECTED_CHUNK_BITS_SHA256
            or runner.get("candidate_chunk_bits_sha256") != candidate_sha
            or runner.get("code_hash") != code_pin
            or runner.get("mismatched_rows") != rows
            or runner.get("mismatched_lanes") != lanes
            or runner.get("producer_invocation_count") != 1
            or runner.get("host_transfer_count_after_completion") != 1
            or runner.get("performance_claim") is not False
            or runner.get("status") != "COMPLETED"
            or runner.get("original_db518_chunk0_exact") is not exact
            or runner.get("adapted_wk") != {
                "byte_sum": 193298069,
                "dtype": "float32",
                "sha256": EXPECTED_WK_SHA256,
                "shape": [128, 6144],
            } or runner.get("backend") != "tpu"
            or runner.get("device", {}).get("device_count") != 4
            or runner.get("device", {}).get("local_device_count") != 4
            or runner.get("device", {}).get("device_kind") != "TPU v4"
            or runner.get("device", {}).get("process_index") != 0
            or row_mapping_sha != EXPECTED_ROW_MAPPING_SHA256
            or positions_sha != EXPECTED_POSITIONS_SHA256 or runner.get(
                "input_authority",
                {}).get("row_mapping_sha256") != EXPECTED_ROW_MAPPING_SHA256
            or runner.get("input_authority", {}).get("positions_sha256")
            != EXPECTED_POSITIONS_SHA256
            or runner.get("input_authority", {}).get("input_manifest_sha256")
            != EXPECTED_INPUT_MANIFEST_SHA256 or runner.get(
                "input_authority", {}).get("accepted_cache_manifest_sha256")
            != EXPECTED_CACHE_MANIFEST_SHA256 or runner.get(
                "input_authority", {}).get("original_db518_cache_sha256")
            != EXPECTED_DB518_CACHE_SHA256 or runner.get(
                "input_authority",
                {}).get("original_db518_code_hash") != EXPECTED_DB518_CODE_HASH
            or runner.get("input_authority",
                          {}).get("original_db518_comparison_manifest_sha256")
            != EXPECTED_DB518_COMPARISON_MANIFEST_SHA256
            or runner.get("input_authority",
                          {}).get("original_db518_tag") != EXPECTED_DB518_TAG):
        raise RuntimeError(
            "original-DB518 numerical records disagree with archived arrays")
    hlo_names = (
        "original_db518_chunk0.optimized_hlo.txt",
        "original_db518_chunk0.stablehlo.mlir",
        "wk_decode.optimized_hlo.txt",
        "wk_decode.stablehlo.mlir",
        "wk_promote.optimized_hlo.txt",
        "wk_promote.stablehlo.mlir",
    )
    hlo_raw = {
        name: base.snapshot_member(run_fd, f"hlo/{name}", limit=256 << 20)
        for name in hlo_names
    }
    for name, raw in hlo_raw.items():
        field = name.replace("original_db518_chunk0.", "").replace(".", "_")
        # The runner uses explicit keys; validate below rather than deriving names.
        if not raw:
            raise RuntimeError(f"empty original-DB518 HLO artifact: {field}")
    hlo = runner.get("hlo")
    if (not isinstance(hlo, Mapping) or set(hlo) != {
            "optimized_contract",
            "optimized_sha256",
            "stablehlo_contract",
            "stablehlo_sha256",
            "wk_decode_optimized_sha256",
            "wk_decode_stablehlo_sha256",
            "wk_promote_optimized_sha256",
            "wk_promote_stablehlo_sha256",
    }):
        raise RuntimeError("original-DB518 HLO record is absent")
    main_opt = hlo_raw["original_db518_chunk0.optimized_hlo.txt"]
    main_stable = hlo_raw["original_db518_chunk0.stablehlo.mlir"]
    decode_opt = hlo_raw["wk_decode.optimized_hlo.txt"]
    decode_stable = hlo_raw["wk_decode.stablehlo.mlir"]
    promote_opt = hlo_raw["wk_promote.optimized_hlo.txt"]
    promote_stable = hlo_raw["wk_promote.stablehlo.mlir"]
    _validate_helper_hlo(parent, decode_opt.decode(), decode_stable.decode(),
                         "decode")
    _validate_helper_hlo(parent, promote_opt.decode(), promote_stable.decode(),
                         "promote")
    optimized_contract = contract.validate_original_db518_prompt_key_hlo(
        main_opt.decode("utf-8", errors="strict"))
    stablehlo_contract = contract.validate_original_db518_prompt_key_stablehlo(
        main_stable.decode("utf-8", errors="strict"))
    if not optimized_contract["passed"] or not stablehlo_contract["passed"]:
        raise RuntimeError("publisher independently rejected producer HLO")
    expected_hashes = {
        "optimized_sha256": sha256(main_opt).hexdigest(),
        "stablehlo_sha256": sha256(main_stable).hexdigest(),
        "wk_decode_optimized_sha256": sha256(decode_opt).hexdigest(),
        "wk_decode_stablehlo_sha256": sha256(decode_stable).hexdigest(),
        "wk_promote_optimized_sha256": sha256(promote_opt).hexdigest(),
        "wk_promote_stablehlo_sha256": sha256(promote_stable).hexdigest(),
    }
    if any(hlo.get(name) != value for name, value in expected_hashes.items()):
        raise RuntimeError("original-DB518 HLO digest record drifted")
    if (hlo.get("optimized_contract") != optimized_contract
            or hlo.get("stablehlo_contract") != stablehlo_contract):
        raise RuntimeError(
            "producer HLO contract record disagrees with publisher")
    return arrays_raw, runner, exact


def _prepare_success(
    parent: Any,
    contract: Any,
    base: Any,
    run_fd: int,
    *,
    code_pin: str,
    run_tag: str,
    remote: str,
    elapsed: int,
) -> tuple[dict[str, bytes], str, str, bool]:
    arrays, runner, exact = _validate_runner_and_outputs(
        parent, contract, base, run_fd, code_pin, run_tag)
    status = STATUS_EXACT if exact else STATUS_NONEXACT
    classification = CLASSIFICATION_EXACT if exact else CLASSIFICATION_NONEXACT
    mirror = base.snapshot_member(run_fd, "mirror.sha256", limit=2 << 20)
    base._validate_mirror_replay(mirror, code_pin)
    sync = base.snapshot_member(run_fd, "sync.txt", limit=1 << 20)
    expected_sync = f"SYNC_OK {os.uname().nodename} {code_pin} origin_and_same_region_mirror\n".encode(
    )
    if sync != expected_sync:
        raise RuntimeError("original-DB518 host code authority drifted")
    vacancy_raw = base.snapshot_member(run_fd,
                                       "remote_vacancy.raw.txt",
                                       limit=1 << 20)
    vacancy_summary = base.snapshot_member(run_fd,
                                           "remote_vacancy.txt",
                                           limit=1 << 20)
    base._validate_remote_vacancy_evidence(vacancy_raw, vacancy_summary,
                                           remote)
    census_pre = base.snapshot_member(run_fd, "census_pre.txt")
    census_post = base.snapshot_member(run_fd, "census_post.txt")
    base._require_census(census_pre, "CENSUS_OK")
    base._require_census(census_post, "CENSUS_OK")
    orchestrator = base.snapshot_member(run_fd,
                                        "orchestrator.log",
                                        limit=16 << 20)
    base.write_member_exclusive(run_fd, "orchestrator.sealed.log",
                                orchestrator)
    summary = {
        "artifact_kind": "gate_d_original_db518_prompt_key_chunk0_summary",
        "classification": classification,
        "code_hash": code_pin,
        "elapsed_seconds": elapsed,
        "gate_d_closed": False,
        "original_db518_chunk0_exact": exact,
        "performance_claim": False,
        "remote_prefix": remote,
        "run_tag": run_tag,
        "status": status,
        "tpu_probe_execution_performed": True,
    }
    summary_raw = base._canonical(summary)
    base.write_member_exclusive(run_fd, "summary.json", summary_raw)
    payload = {
        "census_post.txt":
        census_post,
        "census_pre.txt":
        census_pre,
        "hlo/original_db518_chunk0.optimized_hlo.txt":
        base.snapshot_member(run_fd,
                             "hlo/original_db518_chunk0.optimized_hlo.txt",
                             limit=256 << 20),
        "hlo/original_db518_chunk0.stablehlo.mlir":
        base.snapshot_member(run_fd,
                             "hlo/original_db518_chunk0.stablehlo.mlir",
                             limit=256 << 20),
        "hlo/wk_decode.optimized_hlo.txt":
        base.snapshot_member(run_fd,
                             "hlo/wk_decode.optimized_hlo.txt",
                             limit=256 << 20),
        "hlo/wk_decode.stablehlo.mlir":
        base.snapshot_member(run_fd,
                             "hlo/wk_decode.stablehlo.mlir",
                             limit=256 << 20),
        "hlo/wk_promote.optimized_hlo.txt":
        base.snapshot_member(run_fd,
                             "hlo/wk_promote.optimized_hlo.txt",
                             limit=256 << 20),
        "hlo/wk_promote.stablehlo.mlir":
        base.snapshot_member(run_fd,
                             "hlo/wk_promote.stablehlo.mlir",
                             limit=256 << 20),
        "mirror.sha256":
        mirror,
        "orchestrator.sealed.log":
        orchestrator,
        "producer_arrays.npz":
        arrays,
        "publisher_runtime.json":
        base.snapshot_member(run_fd, "publisher_runtime.json", limit=1 << 20),
        "remote_vacancy.raw.txt":
        vacancy_raw,
        "remote_vacancy.txt":
        vacancy_summary,
        "runner.json":
        base.snapshot_member(run_fd, "runner.json", limit=4 << 20),
        "runner.log":
        base.snapshot_member(run_fd, "runner.log", limit=256 << 20),
        "summary.json":
        summary_raw,
        "sync.txt":
        sync,
    }
    evidence = {
        "artifact_kind":
        "gate_d_original_db518_prompt_key_chunk0_local_evidence",
        "classification": classification,
        "code_hash": code_pin,
        "files":
        [base._record(name, payload[name]) for name in sorted(payload)],
        "gate_d_closed": False,
        "performance_claim": False,
        "run_tag": run_tag,
        "status": status,
    }
    evidence_raw = base._canonical(evidence)
    base.write_member_exclusive(run_fd, "evidence.json", evidence_raw)
    payload["evidence.json"] = evidence_raw
    if tuple(sorted(payload)) != tuple(sorted(SUCCESS_PAYLOAD)):
        raise RuntimeError("original-DB518 success inventory drifted")
    if base.local_members(run_fd) != set(SUCCESS_PAYLOAD) | {
            "orchestrator.log"
    }:
        raise RuntimeError(
            "original-DB518 local preterminal inventory drifted")
    return payload, status, classification, exact


def _result_authority_line(status: str, marker_sha256: str,
                           terminal: Mapping[str, Any]) -> str:
    generation = terminal.get("generation")
    terminal_sha256 = terminal.get("sha256")
    if (status not in {STATUS_EXACT, STATUS_NONEXACT}
            or re.fullmatch(r"[0-9a-f]{64}", marker_sha256) is None
            or type(generation) is not str or not generation.isdigit()
            or type(terminal_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", terminal_sha256) is None):
        raise RuntimeError("original-DB518 result authority drifted")
    return (
        f"PROBE_RESULT status={status} marker_sha256={marker_sha256} "
        f"terminal_generation={generation} terminal_sha256={terminal_sha256}")


def _publish_success(
    parent: Any,
    contract: Any,
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
        parent._require_preterminal(base, run_fd)
        base.require_publication_runtime(run_fd, publication_runtime_raw)
        payload, status, classification, exact = _prepare_success(
            parent,
            contract,
            base,
            run_fd,
            code_pin=code_pin,
            run_tag=run_tag,
            remote=remote,
            elapsed=elapsed,
        )
        bucket, prefix = base._bucket_and_prefix(remote, None)
        base._require_never_used_prefix(bucket, prefix)
        records = []
        for relative in sorted(payload):
            record = base._upload_bound(bucket, prefix + relative,
                                        payload[relative])
            record["path"] = relative
            records.append(record)
        ledger_raw = base._canonical({
            "artifact_kind":
            "gate_d_original_db518_prompt_key_chunk0_remote_ledger",
            "objects": records,
            "run_tag": run_tag
        })
        base.write_member_exclusive(run_fd, "remote_objects.json", ledger_raw)
        ledger = base._upload_bound(bucket, prefix + "remote_objects.json",
                                    ledger_raw)
        ledger["path"] = "remote_objects.json"
        for record in records:
            base._replay_bound(bucket, prefix + record["path"], record)
        base._replay_bound(bucket, prefix + "remote_objects.json", ledger)
        expected_remote = set(payload) | {"remote_objects.json"}
        if base._observed_names(bucket, prefix) != expected_remote:
            raise RuntimeError("original-DB518 preterminal remote set drifted")
        marker = {
            "artifact_kind": "gate_d_original_db518_prompt_key_chunk0_result",
            "classification": classification,
            "evidence_sha256": sha256(payload["evidence.json"]).hexdigest(),
            "gate_d_closed": False,
            "original_db518_chunk0_exact": exact,
            "performance_claim": False,
            "remote_ledger": ledger,
            "run_tag": run_tag,
            "status": status,
            "summary_sha256": sha256(payload["summary.json"]).hexdigest(),
            "tpu_probe_execution_performed": True,
        }
        marker["marker_payload_sha256"] = sha256(
            base._canonical(marker)).hexdigest()
        terminal_raw = base._canonical(marker)
        base.write_member_exclusive(run_fd, "PROBE_RESULT", terminal_raw)
        terminal = base._upload_bound(bucket, prefix + "PROBE_RESULT",
                                      terminal_raw)
        terminal["path"] = "PROBE_RESULT"
        base._replay_bound(bucket, prefix + "PROBE_RESULT", terminal)
        if base._observed_names(bucket,
                                prefix) != expected_remote | {"PROBE_RESULT"}:
            raise RuntimeError("original-DB518 terminal remote set drifted")
        base.write_member_exclusive(
            run_fd,
            "terminal_upload_receipt.json",
            base._canonical({
                "artifact_kind":
                "gate_d_original_db518_prompt_key_chunk0_terminal_receipt",
                "remote": remote + "/PROBE_RESULT",
                "terminal": terminal,
            }),
        )
        return _result_authority_line(status, marker["marker_payload_sha256"],
                                      terminal)
    finally:
        os.close(run_fd)


def _publish_diagnostic(
    parent: Any,
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
        parent._require_preterminal(base, run_fd)
        base.require_publication_runtime(run_fd, publication_runtime_raw)
        failure_raw = base._canonical({
            "artifact_kind": "gate_d_original_db518_prompt_key_chunk0_failure",
            "code_hash": code_pin,
            "exit_status": status,
            "gate_d_closed": False,
            "performance_claim": False,
            "run_tag": run_tag,
        })
        base.write_member_exclusive(run_fd, "failure_status.json", failure_raw)
        orchestrator = base.snapshot_member(run_fd,
                                            "orchestrator.log",
                                            limit=16 << 20)
        base.write_member_exclusive(run_fd, "orchestrator.failure.log",
                                    orchestrator)
        vacancy_raw = base.snapshot_member(run_fd,
                                           "remote_vacancy.raw.txt",
                                           limit=1 << 20)
        vacancy_summary = base.snapshot_member(run_fd,
                                               "remote_vacancy.txt",
                                               limit=1 << 20)
        base._validate_remote_vacancy_evidence(vacancy_raw, vacancy_summary,
                                               remote)
        excluded = {
            "orchestrator.log", "diagnostic_objects.json",
            "diagnostic_upload_receipt.json"
        }
        members = [
            name for name in sorted(base.local_members(run_fd))
            if name not in excluded
        ]
        if not members or len(members) > 64:
            raise RuntimeError("original-DB518 diagnostic inventory is unsafe")
        payload = {
            name: base.snapshot_member(run_fd, name)
            for name in members
        }
        bucket, prefix = base._bucket_and_prefix(remote, None)
        base._require_never_used_prefix(bucket, prefix)
        records = []
        diagnostic_prefix = prefix + "diagnostic/"
        for relative in sorted(payload):
            record = base._upload_bound(bucket, diagnostic_prefix + relative,
                                        payload[relative])
            record["path"] = relative
            records.append(record)
        for record in records:
            base._replay_bound(bucket, diagnostic_prefix + record["path"],
                               record)
        if base._observed_names(bucket, diagnostic_prefix) != set(payload):
            raise RuntimeError("original-DB518 diagnostic object set drifted")
        ledger_raw = base._canonical({
            "artifact_kind":
            "gate_d_original_db518_prompt_key_chunk0_diagnostic_ledger",
            "objects": records,
            "run_tag": run_tag
        })
        base.write_member_exclusive(run_fd, "diagnostic_objects.json",
                                    ledger_raw)
        terminal = base._upload_bound(
            bucket, diagnostic_prefix + "diagnostic_objects.json", ledger_raw)
        terminal["path"] = "diagnostic_objects.json"
        base._replay_bound(bucket,
                           diagnostic_prefix + "diagnostic_objects.json",
                           terminal)
        if base._observed_names(bucket, diagnostic_prefix) != set(payload) | {
                "diagnostic_objects.json"
        }:
            raise RuntimeError(
                "original-DB518 diagnostic terminal set drifted")
        base.write_member_exclusive(
            run_fd,
            "diagnostic_upload_receipt.json",
            base._canonical({
                "artifact_kind":
                "gate_d_original_db518_prompt_key_chunk0_diagnostic_receipt",
                "remote": remote + "/diagnostic/diagnostic_objects.json",
                "terminal": terminal,
            }),
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
    write.add_argument("--member",
                       choices=sorted(WRAPPER_WRITE_MEMBERS),
                       required=True)
    modes.add_parser("append-log").add_argument("--run-dir",
                                                type=Path,
                                                required=True)
    return parser.parse_args()


def main() -> int:
    os.umask(0o077)
    arguments = parse_args()
    parent = _load_parent(arguments.expected_code_hash)
    contract = _load_contract(arguments.expected_code_hash)
    base = parent._load_base(arguments.expected_code_hash)
    base.validate_environment()
    publication_runtime = base.validate_publication_runtime()
    publication_runtime[
        "artifact_kind"] = "gate_d_original_db518_prompt_key_chunk0_publisher_runtime"
    publication_runtime_raw = base._canonical(publication_runtime)
    _verify_running_source(arguments.expected_code_hash,
                           arguments.expected_source_sha256)
    if arguments.run_dir_fd != 7:
        raise RuntimeError("publisher requires run-directory fd 7")
    if arguments.mode == "init":
        print(
            "RUN_IDENTITY " + parent._initialize_retained_run_dir(
                base, arguments.run_dir, arguments.run_dir_fd,
                publication_runtime_raw),
            flush=True,
        )
    elif arguments.mode == "success":
        print(
            _publish_success(
                parent,
                contract,
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
            parent,
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
            parent._require_preterminal(base, run_fd)
            base.require_publication_runtime(run_fd, publication_runtime_raw)
            base.write_preterminal_member(run_fd, arguments.member,
                                          sys.stdin.buffer)
        finally:
            os.close(run_fd)
    else:
        run_fd = base._run_fd(arguments.run_dir, arguments.run_dir_fd)
        try:
            parent._require_preterminal(base, run_fd)
            base.require_publication_runtime(run_fd, publication_runtime_raw)
            raw = sys.stdin.buffer.read((1 << 20) + 1)
            if len(raw) > 1 << 20:
                raise RuntimeError("publisher log append exceeds limit")
            base.append_preterminal_log(run_fd, raw)
        finally:
            os.close(run_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
