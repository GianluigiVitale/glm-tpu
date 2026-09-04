#!/usr/bin/env python3
"""Append-only publisher for the bounded layer-1 prompt chunk-0 probe."""

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
    "/usr/local/libexec/glm-tpu/gate-d-layer1-prompt-chunk0-geometry-v7/"
    "publish_gate_d_layer1_prompt_chunk0_geometry.py"
)
SOURCE_PATH = "scripts/greenfield/publish_gate_d_layer1_prompt_chunk0_geometry.py"
PROBE_SOURCE_PATH = "scripts/greenfield/probe_layer1_prompt_chunk0_geometry.py"
PARSER_CONTRACT_SOURCE_PATH = "glm_tpu/greenfield/validation/chunk0_embedding_hlo.py"
BOUNDARY_CONTRACT_SOURCE_PATH = (
    "glm_tpu/greenfield/validation/original_db518_normalized_boundary_hlo.py"
)
CONSUMER_CONTRACT_SOURCE_PATH = (
    "glm_tpu/greenfield/validation/chunk0_real_layer_consumer_hlo.py"
)
PARSER_CONTRACT_SHA256 = (
    "e239c20b1a206061c9116726421343d81d5ff989be5e8f8440d5c59106eb9757"
)
BOUNDARY_CONTRACT_SHA256 = (
    "35757aab4a616a2f1073f78503c29075c3e43cb684e4e1faf7d835630876809e"
)
CONSUMER_CONTRACT_SHA256 = (
    "4ef7bbb0dbd74e5317cc653e67ef72dc5fd6ea472b9e486fc99c9ed410422aec"
)
BASE_PATH = "scripts/greenfield/publish_gate_d_projection_contraction_pp16_hlo.py"
BASE_PIN = "986378238ac6458307aea69ef1f5e12bf82bc020"
BASE_SHA256 = "f3f20a01fd37bb82988cd77f69fa7b0a780d120568bab0db4162f42e7855bc97"
BUCKET_NAME = "driftbench-dsv4-uc"
REMOTE_ROOT = "results/greenfield/glm52/layer1_prompt_chunk0_geometry/"
TAG_PATTERN = re.compile(
    r"greenfield_layer1_prompt_chunk0_geometry_[0-9]{8}T[0-9]{15}Z"
)
CLAIM_SCOPE = (
    "Completed device normalization consumed by a separate full layer-0 path; "
    "row 0 is the only decisive legacy comparison, rows >= 1 use a non-legacy "
    "softmax and are diagnostic only; no decoder, Gate-D, DB or performance claim."
)
INPUTS = {
    "checkpoint_index_sha256": "e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf",
    "checkpoint_shard_sha256": "cd4b389324d8ed223c28a9e16a718228fa4d24ce09cdc0d0db92fd722df8e6d8",
    "db518_layer0_chunk0_bits_sha256": "96d261cbef56bccc20c3d2bee275b995f168e0ff7609713baae21d0a6ea3887c",
    "db518_layer1_chunk0_bits_sha256": "26bba9eaedf885caf210496b2f17c90d2565597b6347437463fd59fa31ff8f30",
    "db518_result_sha256": "534bacc54d74992f5a8ab4d422f9fa0947523d59325b4bfa272d4fbeb56262f0",
    "layer0_input_manifest_sha256": "574f3553e6106a997e780b6b2a321bce86ad358b19c38989e84e2a4914b73141",
    "legacy_layer1_bits_sha256": "8d656d7103b4e3646f25b88bc4b7998d7a68c5419a1efeeea2e242a8077b4e4d",
    "legacy_layer1_chunk0_bits_sha256": "9a577a9eb24faef5295f8ee8bb1c3252424444e3c57dc4f20985b9853664f257",
    "legacy_layer1_manifest_sha256": "d9058cc6584aca784212706789e72b4981754cb9880e553122b5e854bc3961ac",
    "softmax_scale": 0.0625,
    "weight_digests_sha256": "5a49ab9a8c6a6dc7ee41cf104856e4709c849b0c42cb68d00e52c33241552463",
}
STATUS_EXACT = "REAL_LAYER_CONSUMER_ROW0_EXACT"
STATUS_NONEXACT = "REAL_LAYER_CONSUMER_ROW0_NONEXACT"
CLASSIFICATION_EXACT = (
    "COMPLETED_NORMALIZATION_REAL_LAYER_CONSUMER_REPRODUCES_LEGACY_LAYER1_KEY;"
    "ROWS_1_PLUS_NONLEGACY_ATTENTION_DIAGNOSTIC_ONLY;DECODER_UNPROVEN;GATE_D_OPEN"
)
CLASSIFICATION_NONEXACT = (
    "COMPLETED_NORMALIZATION_REAL_LAYER_CONSUMER_DOES_NOT_REPRODUCE_LEGACY_LAYER1_KEY;"
    "CONSUMER_NUMERICS_INCOMPLETE;DECODER_UNPROVEN;GATE_D_OPEN"
)
SUCCESS_PAYLOAD = (
    "census_post.txt",
    "census_pre.txt",
    "evidence.json",
    "hlo/normalized_boundary.optimized_hlo.txt",
    "hlo/normalized_boundary.stablehlo.mlir",
    "hlo/normalized_key_control.optimized_hlo.txt",
    "hlo/normalized_key_control.stablehlo.mlir",
    "hlo/real_layer_consumer.optimized_hlo.txt",
    "hlo/real_layer_consumer.stablehlo.mlir",
    "hlo/wk_decode.optimized_hlo.txt",
    "hlo/wk_decode.stablehlo.mlir",
    "hlo/wk_promote.optimized_hlo.txt",
    "hlo/wk_promote.stablehlo.mlir",
    "mirror.sha256",
    "orchestrator.sealed.log",
    "probe_arrays.npz",
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
    "keys0_bits": ((2048, 128), "<u2"),
    "keys1_bits": ((2048, 128), "<u2"),
    "greenfield_layer0_bits": ((2048, 128), "<u2"),
    "greenfield_layer1_bits": ((2048, 128), "<u2"),
    "legacy_layer1_bits": ((2048, 128), "<u2"),
    "normalized1_chunk_bits": ((2048, 6144), "<u2"),
}
ROW_WIDTHS = {
    "normalized0_row0": 6144,
    "q_a_row0": 2048,
    "latent_row0": 512,
    "attention_row0": 6144,
    "attention_pairwise_row0": 6144,
    "carried0_row0": 6144,
    "dense_row0": 6144,
    "carried1_row0": 6144,
    "normalized1_row0": 6144,
}
for _name, _width in ROW_WIDTHS.items():
    EXPECTED_OUTPUT_LAYOUT[f"{_name}_bits"] = ((_width,), "<u2")

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
        chunks: list[bytes] = []
        while block := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(block)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (
            len(raw) != before.st_size
            or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise RuntimeError(f"publisher source changed while reading: {path}")
        return raw
    finally:
        os.close(descriptor)


def _verify_running_source(code_pin: str, expected_sha256: str) -> None:
    if Path(__file__) != INSTALL_PATH:
        raise RuntimeError(
            "chunk-0 publisher is not executing from the immutable capsule"
        )
    raw = _snapshot(INSTALL_PATH)
    metadata = os.stat(INSTALL_PATH, follow_symlinks=False)
    if (
        metadata.st_uid != 0
        or metadata.st_gid != 0
        or stat.S_IMODE(metadata.st_mode) != 0o555
        or os.listxattr(INSTALL_PATH, follow_symlinks=False)
        or _git_bytes("for-each-ref", "--format=%(refname)", "refs/replace")
        or _git_bytes("rev-parse", "HEAD").decode().strip() != code_pin
        or raw != _git_bytes("show", f"{code_pin}:{SOURCE_PATH}")
        or sha256(raw).hexdigest() != expected_sha256
    ):
        raise RuntimeError("chunk-0 publisher is not the committed immutable blob")


def _load_base(code_pin: str) -> types.ModuleType:
    path = REPO / BASE_PATH
    raw = _snapshot(path)
    if (
        raw != _git_bytes("show", f"{code_pin}:{BASE_PATH}")
        or raw != _git_bytes("show", f"{BASE_PIN}:{BASE_PATH}")
        or sha256(raw).hexdigest() != BASE_SHA256
    ):
        raise RuntimeError("chunk-0 publisher base bytes drifted")
    module = types.ModuleType("_gate_d_chunk0_publisher_base")
    module.__file__ = str(path)
    module.__package__ = None
    exec(compile(raw, str(path), "exec"), module.__dict__)  # noqa: S102
    module.REPO = REPO
    module.RUN_ROOT = RUN_ROOT
    module.BUCKET_NAME = BUCKET_NAME
    module.REMOTE_ROOT = REMOTE_ROOT
    module.TAG_PATTERN = TAG_PATTERN
    module._WRAPPER_WRITE_MEMBERS = set(WRAPPER_WRITE_MEMBERS)
    module._EXPECTED_ENVIRONMENT = {
        "HOME": "/home/gianl",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    module._git_bytes = lambda *args: _git_bytes(*args)
    return module


def _parse_npy(raw: bytes) -> tuple[tuple[int, ...], str, bytes]:
    if not raw.startswith(b"\x93NUMPY") or len(raw) < 10:
        raise RuntimeError("chunk-0 NPY header is invalid")
    major, minor = raw[6], raw[7]
    if (major, minor) == (1, 0):
        size = int.from_bytes(raw[8:10], "little")
        start = 10
    elif major in (2, 3) and minor == 0 and len(raw) >= 12:
        size = int.from_bytes(raw[8:12], "little")
        start = 12
    else:
        raise RuntimeError("chunk-0 NPY version is unsupported")
    end = start + size
    if end > len(raw) or size > 1 << 20:
        raise RuntimeError("chunk-0 NPY header size is unsafe")
    try:
        header = ast.literal_eval(raw[start:end].decode("latin1").strip())
    except (SyntaxError, ValueError, UnicodeDecodeError) as error:
        raise RuntimeError("chunk-0 NPY header is malformed") from error
    if (
        type(header) is not dict
        or set(header) != {"descr", "fortran_order", "shape"}
        or header["fortran_order"] is not False
        or type(header["shape"]) is not tuple
        or not all(type(value) is int and value >= 0 for value in header["shape"])
        or type(header["descr"]) is not str
    ):
        raise RuntimeError("chunk-0 NPY schema drifted")
    return header["shape"], header["descr"], raw[end:]


def _mismatch_rows(left: bytes, right: bytes, width: int) -> list[int]:
    if len(left) != len(right) or len(left) % (2 * width):
        raise RuntimeError("chunk-0 row comparison geometry drifted")
    rows = len(left) // (2 * width)
    return [
        sum(
            left[(row * width + lane) * 2 : (row * width + lane + 1) * 2]
            != right[(row * width + lane) * 2 : (row * width + lane + 1) * 2]
            for lane in range(width)
        )
        for row in range(rows)
    ]


def _validate_npz(
    raw: bytes,
    runner: Mapping[str, Any],
    *,
    reference_sha256s: Mapping[str, str] | None = None,
) -> bool:
    payloads: dict[str, bytes] = {}
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        names = archive.namelist()
        expected = {f"{name}.npy" for name in EXPECTED_OUTPUT_LAYOUT}
        if set(names) != expected or len(names) != len(expected):
            raise RuntimeError("chunk-0 NPZ inventory drifted")
        total = 0
        for name in names:
            member = archive.getinfo(name)
            total += member.file_size
            if (
                member.compress_type != zipfile.ZIP_STORED
                or member.flag_bits & 0x1
                or member.file_size > 32 << 20
                or member.compress_size != member.file_size
                or total > 96 << 20
            ):
                raise RuntimeError("chunk-0 NPZ member is unsafe")
            shape, dtype, payload = _parse_npy(archive.read(name))
            key = name.removesuffix(".npy")
            expected_shape, expected_dtype = EXPECTED_OUTPUT_LAYOUT[key]
            elements = 1
            for dimension in expected_shape:
                elements *= dimension
            if (
                shape != expected_shape
                or dtype != expected_dtype
                or len(payload) != elements * 2
            ):
                raise RuntimeError(f"chunk-0 NPZ member drifted: {key}")
            payloads[key] = payload
    keys0 = payloads["keys0_bits"]
    keys1 = payloads["keys1_bits"]
    greenfield0 = payloads["greenfield_layer0_bits"]
    greenfield1 = payloads["greenfield_layer1_bits"]
    legacy1 = payloads["legacy_layer1_bits"]
    control = _mismatch_rows(keys0, greenfield0, 128)
    legacy = _mismatch_rows(keys1, legacy1, 128)
    greenfield = _mismatch_rows(keys1, greenfield1, 128)
    row0 = runner.get("row0")
    chunk = runner.get("chunk0_vs_legacy")
    chunk_greenfield = runner.get("chunk0_vs_greenfield_db518")
    expected_references = (
        {
            "greenfield_layer0_bits": INPUTS["db518_layer0_chunk0_bits_sha256"],
            "greenfield_layer1_bits": INPUTS["db518_layer1_chunk0_bits_sha256"],
            "legacy_layer1_bits": INPUTS["legacy_layer1_chunk0_bits_sha256"],
        }
        if reference_sha256s is None
        else dict(reference_sha256s)
    )
    if set(expected_references) != {
        "greenfield_layer0_bits",
        "greenfield_layer1_bits",
        "legacy_layer1_bits",
    } or any(
        sha256(payloads[name]).hexdigest() != expected
        for name, expected in expected_references.items()
    ):
        raise RuntimeError("chunk-0 reference array identity drifted")
    if not all(isinstance(value, Mapping) for value in (row0, chunk, chunk_greenfield)):
        raise RuntimeError("chunk-0 numerical records are absent")
    expected_row0 = {
        "attention_row0_db533_vs_pairwise_lanes": _mismatch_rows(
            payloads["attention_row0_bits"],
            payloads["attention_pairwise_row0_bits"],
            6144,
        )[0],
        "legacy_geometry_vs_greenfield_db518_lanes": greenfield[0],
        "legacy_geometry_vs_legacy_lanes": legacy[0],
    }
    expected_chunk = {
        "lanes_mismatched": sum(legacy),
        "per_64_row_block_mean": [
            sum(legacy[index : index + 64]) / 64 for index in range(0, 2048, 64)
        ],
        "per_row_first_16": legacy[:16],
        "rows": 2048,
        "rows_exact": sum(value == 0 for value in legacy),
    }
    expected_greenfield = {
        "lanes_mismatched": sum(greenfield),
        "rows_exact": sum(value == 0 for value in greenfield),
    }
    if (
        any(control)
        or runner.get("arrays_sha256") != sha256(raw).hexdigest()
        or runner.get("control_layer0_keys_vs_db518_mismatched_rows")
        != sum(value != 0 for value in control)
        or runner.get("control_layer0_keys_vs_db518_mismatched_lanes") != sum(control)
        or dict(row0) != expected_row0
        or dict(chunk) != expected_chunk
        or dict(chunk_greenfield) != expected_greenfield
    ):
        raise RuntimeError("chunk-0 numerical records disagree with archived arrays")
    return legacy[0] == 0


def _entry_graph(
    optimized: str,
) -> tuple[dict[str, dict[str, Any]], str]:
    """Parse the optimized ENTRY computation needed by the wk boundary audit."""

    lines = optimized.splitlines()
    try:
        start = next(
            index for index, line in enumerate(lines) if line.startswith("ENTRY ")
        )
    except StopIteration as error:
        raise RuntimeError("chunk-0 wk ENTRY computation is absent") from error
    nodes: dict[str, dict[str, Any]] = {}
    root = ""
    instruction = re.compile(
        r"^\s*(?P<root>ROOT )?(?P<name>%[A-Za-z0-9_.-]+) = "
        r"(?P<result>.*?)\b(?P<opcode>[a-z][a-z0-9-]*)\("
    )
    shape = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\[([0-9,]*)\]")
    for line in lines[start + 1 :]:
        if line == "}":
            break
        match = instruction.match(line)
        if match is None:
            continue
        call_start = match.end() - 1
        depth = 0
        call_end = -1
        for index in range(call_start, len(line)):
            if line[index] == "(":
                depth += 1
            elif line[index] == ")":
                depth -= 1
                if depth == 0:
                    call_end = index
                    break
        if call_end < 0:
            raise RuntimeError("chunk-0 wk HLO call syntax is unbalanced")
        result_shapes = tuple(
            (
                item.group(1),
                tuple(int(value) for value in item.group(2).split(",") if value),
            )
            for item in shape.finditer(match.group("result"))
        )
        name = match.group("name")
        nodes[name] = {
            "opcode": match.group("opcode"),
            "operands": tuple(
                re.findall(r"%[A-Za-z0-9_.-]+", line[call_start + 1 : call_end])
            ),
            "raw": line,
            "shapes": result_shapes,
        }
        if match.group("root"):
            if root:
                raise RuntimeError("chunk-0 wk ENTRY has multiple roots")
            root = name
    if not nodes or not root:
        raise RuntimeError("chunk-0 wk ENTRY graph is incomplete")
    return nodes, root


def _ancestors(nodes: Mapping[str, Mapping[str, Any]], name: str) -> set[str]:
    pending = [name]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        if current not in nodes:
            raise RuntimeError("chunk-0 wk ENTRY operand is unresolved")
        visited.add(current)
        pending.extend(nodes[current]["operands"])
        if len(visited) > 10000:
            raise RuntimeError("chunk-0 wk ENTRY graph is unbounded")
    return visited


def _stablehlo_main_signature(stablehlo: str) -> str:
    match = re.search(
        r"^  func\.func public @main\((.*)\) -> (.*) \{$", stablehlo, re.M
    )
    if match is None:
        raise RuntimeError("chunk-0 wk StableHLO public signature is absent")
    return match.group(0)


def _require_wk_hlo_boundaries(
    decode_optimized: str,
    decode_stablehlo: str,
    promote_optimized: str,
    promote_stablehlo: str,
) -> None:
    """Prove the exact completed raw-to-BF16-to-FP32 wk helpers."""

    decode_nodes, decode_root = _entry_graph(decode_optimized)
    promote_nodes, promote_root = _entry_graph(promote_optimized)
    parameter = lambda nodes: {
        name: node for name, node in nodes.items() if node["opcode"] == "parameter"
    }
    decode_parameters = parameter(decode_nodes)
    promote_parameters = parameter(promote_nodes)
    wk_shape = ("f32", (128, 6144))
    raw_shape = ("u8", (128, 6144))
    scale_shape = ("f32", (1, 48))
    bf16_shape = ("bf16", (128, 6144))
    helper_communication = (
        "all-gather(",
        "all-reduce(",
        "all-to-all(",
        "collective-permute(",
        "reduce-scatter(",
    )
    decode_parameter_shapes = sorted(
        node["shapes"] for node in decode_parameters.values()
    )
    promote_parameter_shapes = sorted(
        node["shapes"] for node in promote_parameters.values()
    )
    decode_live = _ancestors(decode_nodes, decode_root)
    promote_live = _ancestors(promote_nodes, promote_root)
    decode_signature = _stablehlo_main_signature(decode_stablehlo)
    promote_signature = _stablehlo_main_signature(promote_stablehlo)
    if (
        decode_parameter_shapes != [(("f32", (1, 48)),), (("u8", (128, 6144)),)]
        or decode_nodes[decode_root]["shapes"] != (bf16_shape,)
        or decode_nodes[decode_root]["opcode"] != "convert"
        or len(decode_nodes[decode_root]["operands"]) != 1
        or not set(decode_parameters) <= decode_live
        or promote_parameter_shapes != [(bf16_shape,)]
        or promote_nodes[promote_root]["shapes"] != (wk_shape,)
        or promote_nodes[promote_root]["opcode"] != "convert"
        or promote_nodes[promote_root]["operands"] != tuple(promote_parameters)
        or promote_live != {promote_root, *promote_parameters}
        or any(
            token in decode_optimized
            or token in decode_stablehlo
            or token in promote_optimized
            or token in promote_stablehlo
            for token in helper_communication
        )
        or decode_signature.count("tensor<128x6144xui8>") != 1
        or decode_signature.count("tensor<1x48xf32>") != 1
        or decode_signature.count("tensor<128x6144xbf16>") != 1
        or promote_signature.count("tensor<128x6144xbf16>") != 1
        or promote_signature.count("tensor<128x6144xf32>") != 1
        or "stablehlo.convert" not in decode_stablehlo
        or "stablehlo.convert" not in promote_stablehlo
    ):
        raise RuntimeError("chunk-0 wk executable boundary drifted")


def _load_hlo_contracts(code_pin: str) -> tuple[types.ModuleType, types.ModuleType]:
    """Load exact committed boundary and real-consumer validators."""

    sources = (
        (PARSER_CONTRACT_SOURCE_PATH, PARSER_CONTRACT_SHA256),
        (BOUNDARY_CONTRACT_SOURCE_PATH, BOUNDARY_CONTRACT_SHA256),
        (CONSUMER_CONTRACT_SOURCE_PATH, CONSUMER_CONTRACT_SHA256),
    )
    raw_by_source: dict[str, bytes] = {}
    for source, expected_sha256 in sources:
        adjacent = Path(__file__).with_name(Path(source).name)
        path = adjacent if adjacent.is_file() else REPO / source
        if Path(__file__) == INSTALL_PATH and path != adjacent:
            raise RuntimeError("installed chunk-0 HLO contract is absent")
        raw = _snapshot(path)
        if sha256(raw).hexdigest() != expected_sha256 or raw != _git_bytes(
            "show", f"{code_pin}:{source}"
        ):
            raise RuntimeError("chunk-0 HLO contract bytes drifted")
        raw_by_source[source] = raw

    for name in (
        "glm_tpu",
        "glm_tpu.greenfield",
        "glm_tpu.greenfield.validation",
    ):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = []  # type: ignore[attr-defined]
            package.__package__ = name.rpartition(".")[0]
            sys.modules[name] = package

    loaded: dict[str, types.ModuleType] = {}
    for source in (
        PARSER_CONTRACT_SOURCE_PATH,
        BOUNDARY_CONTRACT_SOURCE_PATH,
        CONSUMER_CONTRACT_SOURCE_PATH,
    ):
        stem = Path(source).stem
        name = f"glm_tpu.greenfield.validation.{stem}"
        module = types.ModuleType(name)
        module.__file__ = str(Path(__file__).with_name(Path(source).name))
        module.__package__ = "glm_tpu.greenfield.validation"
        sys.modules[name] = module
        exec(
            compile(raw_by_source[source], module.__file__, "exec"), module.__dict__
        )  # noqa: S102
        loaded[stem] = module
    return (
        loaded[Path(BOUNDARY_CONTRACT_SOURCE_PATH).stem],
        loaded[Path(CONSUMER_CONTRACT_SOURCE_PATH).stem],
    )


def _same_json_value(left: Any, right: Any) -> bool:
    """Compare a runner-restored value with a freshly computed contract."""

    encode = lambda value: json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return encode(left) == encode(right)


def _prepare_success(
    base: Any,
    run_fd: int,
    *,
    code_pin: str,
    run_tag: str,
    remote: str,
    elapsed: int,
) -> tuple[dict[str, bytes], str, str, bool]:
    runner_raw = base.snapshot_member(run_fd, "runner.json", limit=2 << 20)
    runner = json.loads(runner_raw)
    if type(runner) is not dict or runner_raw != (
        json.dumps(runner, indent=2, sort_keys=True) + "\n"
    ).encode("ascii"):
        raise RuntimeError("chunk-0 runner report is not canonical")
    expected_probe_sha = sha256(
        _git_bytes("show", f"{code_pin}:{PROBE_SOURCE_PATH}")
    ).hexdigest()
    provenance = runner.get("provenance")
    hlo = runner.get("hlo")
    if (
        set(runner)
        != {
            "arrays_sha256",
            "artifact_kind",
            "chunk0_vs_greenfield_db518",
            "chunk0_vs_legacy",
            "claim_scope",
            "code_hash",
            "control_layer0_keys_vs_db518_mismatched_lanes",
            "control_layer0_keys_vs_db518_mismatched_rows",
            "elapsed_seconds",
            "execution_boundary",
            "forbidden_hlo_tokens",
            "hlo",
            "import_closure_module_count",
            "inputs",
            "provenance",
            "row0",
            "run_tag",
            "status",
        }
        or runner.get("artifact_kind")
        != "greenfield_layer1_prompt_chunk0_legacy_geometry_probe"
        or runner.get("code_hash") != code_pin
        or runner.get("run_tag") != run_tag
        or runner.get("status") not in {"SUCCESS", "DIAGNOSTIC"}
        or runner.get("claim_scope") != CLAIM_SCOPE
        or runner.get("inputs") != INPUTS
        or runner.get("forbidden_hlo_tokens") != []
        or type(runner.get("elapsed_seconds")) not in {int, float}
        or runner["elapsed_seconds"] < 0
        or type(runner.get("import_closure_module_count")) is not int
        or runner["import_closure_module_count"] < 50
        or not isinstance(provenance, Mapping)
        or provenance.get("probe_sha256") != expected_probe_sha
        or not isinstance(hlo, Mapping)
        or runner.get("execution_boundary")
        != {
            "final_host_transfers": 1,
            "key_control_invocations": 1,
            "normalization_to_consumer_host_transfers": 0,
            "normalizer_invocations": 1,
            "real_layer_consumer_invocations": 1,
        }
        or set(hlo)
        != {
            "key_contract",
            "normalization_contract",
            "normalized_boundary_optimized_byte_count",
            "normalized_boundary_optimized_sha256",
            "normalized_boundary_stablehlo_byte_count",
            "normalized_boundary_stablehlo_sha256",
            "normalized_key_control_optimized_byte_count",
            "normalized_key_control_optimized_sha256",
            "normalized_key_control_stablehlo_byte_count",
            "normalized_key_control_stablehlo_sha256",
            "real_layer_consumer_contract",
            "real_layer_consumer_optimized_byte_count",
            "real_layer_consumer_optimized_sha256",
            "real_layer_consumer_stablehlo_byte_count",
            "real_layer_consumer_stablehlo_sha256",
            "wk_decode_optimized_byte_count",
            "wk_decode_optimized_sha256",
            "wk_decode_stablehlo_byte_count",
            "wk_decode_stablehlo_sha256",
            "wk_promote_optimized_byte_count",
            "wk_promote_optimized_sha256",
            "wk_promote_stablehlo_byte_count",
            "wk_promote_stablehlo_sha256",
        }
    ):
        raise RuntimeError("chunk-0 runner claim boundary drifted")
    hlo_files = {
        "normalized_boundary_optimized": (
            "hlo/normalized_boundary.optimized_hlo.txt",
            64 << 20,
        ),
        "normalized_boundary_stablehlo": (
            "hlo/normalized_boundary.stablehlo.mlir",
            64 << 20,
        ),
        "normalized_key_control_optimized": (
            "hlo/normalized_key_control.optimized_hlo.txt",
            64 << 20,
        ),
        "normalized_key_control_stablehlo": (
            "hlo/normalized_key_control.stablehlo.mlir",
            64 << 20,
        ),
        "real_layer_consumer_optimized": (
            "hlo/real_layer_consumer.optimized_hlo.txt",
            512 << 20,
        ),
        "real_layer_consumer_stablehlo": (
            "hlo/real_layer_consumer.stablehlo.mlir",
            512 << 20,
        ),
        "wk_decode_optimized": ("hlo/wk_decode.optimized_hlo.txt", 32 << 20),
        "wk_decode_stablehlo": ("hlo/wk_decode.stablehlo.mlir", 32 << 20),
        "wk_promote_optimized": ("hlo/wk_promote.optimized_hlo.txt", 32 << 20),
        "wk_promote_stablehlo": ("hlo/wk_promote.stablehlo.mlir", 32 << 20),
    }
    hlo_raw = {
        name: base.snapshot_member(run_fd, member, limit=limit)
        for name, (member, limit) in hlo_files.items()
    }
    hlo_texts = {
        name: raw.decode("utf-8", errors="strict") for name, raw in hlo_raw.items()
    }
    forbidden = ("host_callback", 'CustomCall("xla_python', "python_callback")
    _require_wk_hlo_boundaries(
        hlo_texts["wk_decode_optimized"],
        hlo_texts["wk_decode_stablehlo"],
        hlo_texts["wk_promote_optimized"],
        hlo_texts["wk_promote_stablehlo"],
    )
    boundary_contract, consumer_contract = _load_hlo_contracts(code_pin)
    normalization_contract = (
        boundary_contract.require_completed_normalization_boundary_hlo(
            hlo_texts["normalized_boundary_optimized"],
            hlo_texts["normalized_boundary_stablehlo"],
        )
    )
    key_contract = boundary_contract.require_normalized_key_control_boundary_hlo(
        hlo_texts["normalized_key_control_optimized"],
        hlo_texts["normalized_key_control_stablehlo"],
    )
    real_consumer_contract = consumer_contract.require_real_layer_consumer_hlo(
        hlo_texts["real_layer_consumer_optimized"],
        hlo_texts["real_layer_consumer_stablehlo"],
    )
    if (
        any(token in text for token in forbidden for text in hlo_texts.values())
        or not _same_json_value(
            hlo.get("normalization_contract"), normalization_contract
        )
        or not _same_json_value(hlo.get("key_contract"), key_contract)
        or not _same_json_value(
            hlo.get("real_layer_consumer_contract"), real_consumer_contract
        )
        or any(
            hlo.get(f"{name}_byte_count") != len(raw)
            or hlo.get(f"{name}_sha256") != sha256(raw).hexdigest()
            for name, raw in hlo_raw.items()
        )
    ):
        raise RuntimeError("chunk-0 HLO evidence drifted")
    arrays = base.snapshot_member(run_fd, "probe_arrays.npz", limit=96 << 20)
    row0_exact = _validate_npz(arrays, runner)
    status = STATUS_EXACT if row0_exact else STATUS_NONEXACT
    classification = CLASSIFICATION_EXACT if row0_exact else CLASSIFICATION_NONEXACT
    expected_runner_status = "SUCCESS" if row0_exact else "DIAGNOSTIC"
    if runner.get("status") != expected_runner_status:
        raise RuntimeError("chunk-0 runner status disagrees with archived arrays")
    mirror = base.snapshot_member(run_fd, "mirror.sha256", limit=2 << 20)
    base._validate_mirror_replay(mirror, code_pin)
    sync = base.snapshot_member(run_fd, "sync.txt", limit=1 << 20)
    if (
        sync
        != f"SYNC_OK {os.uname().nodename} {code_pin} origin_and_same_region_mirror\n".encode(
            "ascii"
        )
    ):
        raise RuntimeError("chunk-0 host code authority drifted")
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
        "artifact_kind": "gate_d_layer1_prompt_chunk0_geometry_summary",
        "classification": classification,
        "code_hash": code_pin,
        "elapsed_seconds": elapsed,
        "gate_d_closed": False,
        "performance_claim": False,
        "real_layer_consumer_row0_exact": row0_exact,
        "remote_prefix": remote,
        "root_cause_fix_proven": False,
        "run_tag": run_tag,
        "status": status,
        "tpu_probe_execution_performed": True,
    }
    summary_raw = base._canonical(summary)
    base.write_member_exclusive(run_fd, "summary.json", summary_raw)
    payload = {
        "census_post.txt": census_post,
        "census_pre.txt": census_pre,
        **{member: hlo_raw[name] for name, (member, _limit) in hlo_files.items()},
        "mirror.sha256": mirror,
        "orchestrator.sealed.log": orchestrator,
        "probe_arrays.npz": arrays,
        "publisher_runtime.json": base.snapshot_member(
            run_fd, "publisher_runtime.json", limit=1 << 20
        ),
        "remote_vacancy.raw.txt": vacancy_raw,
        "remote_vacancy.txt": vacancy_summary,
        "runner.json": runner_raw,
        "runner.log": base.snapshot_member(run_fd, "runner.log", limit=256 << 20),
        "summary.json": summary_raw,
        "sync.txt": sync,
    }
    evidence = {
        "artifact_kind": "gate_d_layer1_prompt_chunk0_geometry_local_evidence",
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
        raise RuntimeError("chunk-0 success inventory drifted")
    if base.local_members(run_fd) != set(SUCCESS_PAYLOAD) | {"orchestrator.log"}:
        raise RuntimeError("chunk-0 local preterminal inventory drifted")
    return payload, status, classification, row0_exact


def _result_authority_line(
    status: str, marker_sha256: str, terminal: Mapping[str, Any]
) -> str:
    generation = terminal.get("generation")
    terminal_sha256 = terminal.get("sha256")
    if (
        status not in {STATUS_EXACT, STATUS_NONEXACT}
        or re.fullmatch(r"[0-9a-f]{64}", marker_sha256) is None
        or type(generation) is not str
        or not generation.isdigit()
        or type(terminal_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", terminal_sha256) is None
    ):
        raise RuntimeError("chunk-0 result authority drifted")
    return (
        f"PROBE_RESULT status={status} marker_sha256={marker_sha256} "
        f"terminal_generation={generation} terminal_sha256={terminal_sha256}"
    )


def _require_preterminal(base: Any, run_fd: int) -> None:
    """Reject every local success or diagnostic terminal for this publisher."""

    base.require_preterminal(run_fd)
    for name in ("PROBE_RESULT", "terminal_upload_receipt.json"):
        try:
            os.stat(name, dir_fd=run_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        raise RuntimeError("local chunk-0 terminal publication has already begun")


def _initialize_retained_run_dir(
    base: Any,
    run_dir: Path,
    run_dir_fd: int,
    publication_runtime_raw: bytes,
) -> str:
    """Initialize the already-open directory created and retained by the launcher."""

    run_fd = base._run_fd(run_dir, run_dir_fd)
    try:
        with os.scandir(run_fd) as entries:
            observed = {
                entry.name: entry.stat(follow_symlinks=False) for entry in entries
            }
        hlo = observed.get("hlo")
        if (
            set(observed) != {"hlo"}
            or hlo is None
            or not stat.S_ISDIR(hlo.st_mode)
            or stat.S_IMODE(hlo.st_mode) != 0o700
            or hlo.st_uid != os.geteuid()
            or hlo.st_gid != os.getegid()
        ):
            raise RuntimeError("retained chunk-0 run directory is not pristine")
        base.write_member_exclusive(
            run_fd, "publisher_runtime.json", publication_runtime_raw
        )
        metadata = os.fstat(run_fd)
        return f"{metadata.st_dev}:{metadata.st_ino}"
    finally:
        os.close(run_fd)


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
        payload, status, classification, row0_exact = _prepare_success(
            base,
            run_fd,
            code_pin=code_pin,
            run_tag=run_tag,
            remote=remote,
            elapsed=elapsed,
        )
        bucket, prefix = base._bucket_and_prefix(remote, None)
        # This second three-surface history check is immediately before upload.
        base._require_never_used_prefix(bucket, prefix)
        records = []
        for relative in sorted(payload):
            record = base._upload_bound(bucket, prefix + relative, payload[relative])
            record["path"] = relative
            records.append(record)
        ledger_raw = base._canonical(
            {
                "artifact_kind": "gate_d_layer1_prompt_chunk0_geometry_remote_ledger",
                "objects": records,
                "run_tag": run_tag,
            }
        )
        base.write_member_exclusive(run_fd, "remote_objects.json", ledger_raw)
        ledger = base._upload_bound(bucket, prefix + "remote_objects.json", ledger_raw)
        ledger["path"] = "remote_objects.json"
        for record in records:
            base._replay_bound(bucket, prefix + record["path"], record)
        base._replay_bound(bucket, prefix + "remote_objects.json", ledger)
        expected_remote = set(payload) | {"remote_objects.json"}
        if base._observed_names(bucket, prefix) != expected_remote:
            raise RuntimeError("chunk-0 preterminal remote set drifted")
        marker = {
            "artifact_kind": "gate_d_layer1_prompt_chunk0_geometry_result",
            "classification": classification,
            "evidence_sha256": sha256(payload["evidence.json"]).hexdigest(),
            "gate_d_closed": False,
            "performance_claim": False,
            "real_layer_consumer_row0_exact": row0_exact,
            "remote_ledger": ledger,
            "root_cause_fix_proven": False,
            "run_tag": run_tag,
            "status": status,
            "summary_sha256": sha256(payload["summary.json"]).hexdigest(),
            "tpu_probe_execution_performed": True,
        }
        marker["marker_payload_sha256"] = sha256(base._canonical(marker)).hexdigest()
        terminal_raw = base._canonical(marker)
        base.write_member_exclusive(run_fd, "PROBE_RESULT", terminal_raw)
        terminal = base._upload_bound(bucket, prefix + "PROBE_RESULT", terminal_raw)
        terminal["path"] = "PROBE_RESULT"
        base._replay_bound(bucket, prefix + "PROBE_RESULT", terminal)
        if base._observed_names(bucket, prefix) != expected_remote | {"PROBE_RESULT"}:
            raise RuntimeError("chunk-0 terminal remote set drifted")
        base.write_member_exclusive(
            run_fd,
            "terminal_upload_receipt.json",
            base._canonical(
                {
                    "artifact_kind": "gate_d_layer1_prompt_chunk0_geometry_terminal_receipt",
                    "remote": remote + "/PROBE_RESULT",
                    "terminal": terminal,
                }
            ),
        )
        return _result_authority_line(status, marker["marker_payload_sha256"], terminal)
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
                "artifact_kind": "gate_d_layer1_prompt_chunk0_geometry_failure",
                "code_hash": code_pin,
                "exit_status": status,
                "gate_d_closed": False,
                "performance_claim": False,
                "root_cause_fix_proven": False,
                "run_tag": run_tag,
            }
        )
        base.write_member_exclusive(run_fd, "failure_status.json", failure_raw)
        orchestrator = base.snapshot_member(run_fd, "orchestrator.log", limit=16 << 20)
        base.write_member_exclusive(run_fd, "orchestrator.failure.log", orchestrator)
        vacancy_raw = base.snapshot_member(
            run_fd, "remote_vacancy.raw.txt", limit=1 << 20
        )
        vacancy_summary = base.snapshot_member(
            run_fd, "remote_vacancy.txt", limit=1 << 20
        )
        base._validate_remote_vacancy_evidence(vacancy_raw, vacancy_summary, remote)
        members = [
            name
            for name in sorted(base.local_members(run_fd))
            if name
            not in {
                "orchestrator.log",
                "diagnostic_objects.json",
                "diagnostic_upload_receipt.json",
            }
        ]
        if not members or len(members) > 64:
            raise RuntimeError("chunk-0 diagnostic inventory is unsafe")
        payload = {name: base.snapshot_member(run_fd, name) for name in members}
        bucket, prefix = base._bucket_and_prefix(remote, None)
        base._require_never_used_prefix(bucket, prefix)
        records = []
        diagnostic_prefix = prefix + "diagnostic/"
        for relative in sorted(payload):
            record = base._upload_bound(
                bucket, diagnostic_prefix + relative, payload[relative]
            )
            record["path"] = relative
            records.append(record)
        for record in records:
            base._replay_bound(bucket, diagnostic_prefix + record["path"], record)
        if base._observed_names(bucket, diagnostic_prefix) != set(payload):
            raise RuntimeError("chunk-0 diagnostic object set drifted")
        ledger_raw = base._canonical(
            {
                "artifact_kind": "gate_d_layer1_prompt_chunk0_geometry_diagnostic_ledger",
                "objects": records,
                "run_tag": run_tag,
            }
        )
        base.write_member_exclusive(run_fd, "diagnostic_objects.json", ledger_raw)
        terminal = base._upload_bound(
            bucket, diagnostic_prefix + "diagnostic_objects.json", ledger_raw
        )
        terminal["path"] = "diagnostic_objects.json"
        base._replay_bound(
            bucket, diagnostic_prefix + "diagnostic_objects.json", terminal
        )
        if base._observed_names(bucket, diagnostic_prefix) != set(payload) | {
            "diagnostic_objects.json"
        }:
            raise RuntimeError("chunk-0 diagnostic terminal object set drifted")
        base.write_member_exclusive(
            run_fd,
            "diagnostic_upload_receipt.json",
            base._canonical(
                {
                    "artifact_kind": "gate_d_layer1_prompt_chunk0_geometry_diagnostic_receipt",
                    "remote": remote + "/diagnostic/diagnostic_objects.json",
                    "terminal": terminal,
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
    publication_runtime = base.validate_publication_runtime()
    publication_runtime["artifact_kind"] = (
        "gate_d_layer1_prompt_chunk0_geometry_publisher_runtime"
    )
    publication_runtime_raw = base._canonical(publication_runtime)
    _verify_running_source(
        arguments.expected_code_hash, arguments.expected_source_sha256
    )
    if arguments.run_dir_fd != 7:
        raise RuntimeError("chunk-0 publisher requires run-directory fd 7")
    if arguments.mode == "init":
        print(
            "RUN_IDENTITY "
            + _initialize_retained_run_dir(
                base,
                arguments.run_dir,
                arguments.run_dir_fd,
                publication_runtime_raw,
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
            raw = sys.stdin.buffer.read((1 << 20) + 1)
            if len(raw) > 1 << 20:
                raise RuntimeError("chunk-0 log append exceeds limit")
            base.append_preterminal_log(run_fd, raw)
        finally:
            os.close(run_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
