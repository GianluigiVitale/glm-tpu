#!/usr/bin/env python3
"""Fail-closed offline adjudication for one compensated PP16 TPU HLO run.

This tool never imports JAX and never initializes a backend.  It authenticates
the compile-only acquisition, parses the optimized fusion/use-def graph, and
independently replays every archived object by immutable GCS generation.  A
passing report proves only HLO locality and candidate-policy structure.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import stat
import struct
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

RUN_TAG = "gate_d_compensated_pp16_hlo_20260901T070612366187759Z"
CODE_HASH = "94518b7d4ce788157afa98b7bc1f8144613852d5"
OPTIMIZED_SHA256 = "b63623498d82f67824b3be8998c09448440870b753cc1aa95d7a7765422c692b"
STABLEHLO_SHA256 = "55d7940c2aa9f0f7cda463cb816f3a5475985aab89050a5aafc22c2c88792a83"
ADMISSION_SHA256 = "7cd7e569ed9ed5fd978d933efd4229d65906ae28312e264863b8336e4cc6b37d"
TOPOLOGY_SHA256 = "49cf6bb1a553985855556d1401ad85918669df52d12f8dc5150f247d18b325eb"
CAPSULE_SHA256 = "5b7ad71f37dbbcda0ee36a9fc0c42a7ca45d619a9e741307386755e68e87c1a4"
CAPSULE_INPUT_SHA256 = (
    "dd5f1cbb37b722591531635a21cfa9f1d6d0157997a4f70138900d0000be8b1b"
)
REMOTE_PREFIX = (
    "gs://driftbench-dsv4-uc/results/greenfield/glm52/gate_d_pp16_hlo/" + RUN_TAG
)
LEDGER_GENERATION = 1788247356986225
TERMINAL_GENERATION = 1788247358068984
LEDGER_SHA256 = "4be4e469bc5e68314217b96036b64b1871f453be2174181b0aff2085a41214d4"
TERMINAL_SHA256 = "3bd7439551e23ef630f27cee9ca5d8b8113a114447c0c6250ae5c17b2e36e8eb"

_EXPECTED_FILES = {
    "HLO_ACQUIRED": 0o400,
    "census_post.txt": 0o400,
    "census_pre.txt": 0o400,
    "dependencies.json": 0o400,
    "evidence.json": 0o400,
    "hlo/compensated_pp16_stage0.optimized_hlo.txt": 0o400,
    "hlo/compensated_pp16_stage0.stablehlo.mlir": 0o400,
    "mirror.sha256": 0o400,
    "orchestrator.log": 0o600,
    "orchestrator.sealed.log": 0o400,
    "publisher_runtime.json": 0o400,
    "remote_objects.json": 0o400,
    "remote_vacancy.raw.txt": 0o400,
    "remote_vacancy.txt": 0o400,
    "runner.json": 0o400,
    "runner.log": 0o400,
    "summary.json": 0o400,
    "sync.txt": 0o400,
    "terminal_upload_receipt.json": 0o400,
}
_EXPECTED_DIRS = {"", "hlo"}
_REMOTE_LEDGER_PATHS = {
    "census_post.txt",
    "census_pre.txt",
    "dependencies.json",
    "evidence.json",
    "hlo/compensated_pp16_stage0.optimized_hlo.txt",
    "hlo/compensated_pp16_stage0.stablehlo.mlir",
    "mirror.sha256",
    "orchestrator.sealed.log",
    "publisher_runtime.json",
    "remote_vacancy.raw.txt",
    "remote_vacancy.txt",
    "runner.json",
    "runner.log",
    "summary.json",
    "sync.txt",
}
_COLLECTIVES = {
    "all-gather",
    "all-reduce",
    "all-to-all",
    "collective-broadcast",
    "collective-permute",
    "reduce-scatter",
    "send",
    "send-done",
    "recv",
    "recv-done",
}
_FORBIDDEN_TOKENS = (
    "host_callback",
    "outside_compilation",
    "xla_ffi_python_cpu_callback",
    "xla_python_cpu_callback",
    "infeed",
    "outfeed",
    "send-done",
    "recv-done",
)
_ROOT_OUTPUTS = (
    ("f32", (1, 1, 6144)),
    ("bf16", (1, 1, 6144)),
    ("f32", (1, 1, 32, 128)),
    ("f32", (1, 1, 32)),
    ("f32", (1, 1, 128)),
    ("bf16", (1, 16, 256, 128)),
    ("s32", (1, 1, 2048)),
    ("s32", (1, 1)),
    ("f32", (1, 1, 2048)),
    ("u8", (1,)),
)
_RUNNER_OUTPUT_SPEC = (
    ("rms_input_fp32_owners", (2, 1, 6144), "float32"),
    ("normalized_hidden_owners", (2, 1, 6144), "bfloat16"),
    ("query_owners", (2, 1, 32, 128), "float32"),
    ("head_weights_owners", (2, 1, 32), "float32"),
    ("current_key_owners", (2, 1, 128), "float32"),
    ("index_cache_owners", (2, 16, 256, 128), "bfloat16"),
    ("selected_positions_owners", (2, 1, 2048), "int32"),
    ("valid_counts_owners", (2, 1), "int32"),
    ("selected_scores_owners", (2, 1, 2048), "float32"),
    ("contract_valid_owners", (2,), "uint8"),
)


class Refusal(RuntimeError):
    """The candidate cannot receive structural HLO acceptance."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Refusal(message)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Refusal(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                Refusal(f"non-finite JSON value in {label}: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Refusal(f"invalid JSON in {label}: {error}") from error


def _read_regular(path: Path, *, expected_mode: int | None = None) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"not a regular file: {path}")
        _require(before.st_nlink == 1, f"unsafe link count: {path}")
        if expected_mode is not None:
            _require(
                stat.S_IMODE(before.st_mode) == expected_mode,
                f"mode drifted for {path}",
            )
        chunks = []
        while block := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        _require(
            identity(before) == identity(after), f"file changed while read: {path}"
        )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sha(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _positive_int(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0, label
    )
    return value


def _decimal_generation(value: Any, label: str) -> int:
    _require(
        isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value) is not None,
        f"{label} must be a canonical positive decimal string",
    )
    return int(value)


def _false(value: Any, label: str) -> None:
    _require(value is False, f"{label} must be false")


def _inventory(run_dir: Path) -> dict[str, bytes]:
    _require(run_dir.name == RUN_TAG, "run tag/path drifted")
    entries = sorted(
        run_dir.rglob("*"), key=lambda item: item.relative_to(run_dir).as_posix()
    )
    files: dict[str, bytes] = {}
    dirs = {""}
    for path in entries:
        relative = path.relative_to(run_dir).as_posix()
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            dirs.add(relative)
            _require(
                stat.S_IMODE(metadata.st_mode) == 0o700,
                f"directory mode drifted: {relative}",
            )
        elif stat.S_ISREG(metadata.st_mode):
            _require(relative in _EXPECTED_FILES, f"unexpected run file: {relative}")
            files[relative] = _read_regular(
                path, expected_mode=_EXPECTED_FILES[relative]
            )
        else:
            raise Refusal(f"unsupported run entry: {relative}")
    _require(dirs == _EXPECTED_DIRS, f"run directories drifted: {sorted(dirs)}")
    _require(set(files) == set(_EXPECTED_FILES), "run file catalogue incomplete")
    _require(
        files["orchestrator.log"] == files["orchestrator.sealed.log"],
        "orchestrator seal mismatch",
    )
    return files


def _validate_acquisition(
    files: Mapping[str, bytes], source_root: Path
) -> dict[str, Any]:
    runner = _json(files["runner.json"], "runner.json")
    summary = _json(files["summary.json"], "summary.json")
    evidence = _json(files["evidence.json"], "evidence.json")
    marker = _json(files["HLO_ACQUIRED"], "HLO_ACQUIRED")
    receipt = _json(files["terminal_upload_receipt.json"], "terminal receipt")
    dependencies = _json(files["dependencies.json"], "dependencies.json")

    for item, label in (
        (runner, "runner"),
        (summary, "summary"),
        (evidence, "evidence"),
    ):
        _require(item["code_hash"] == CODE_HASH, f"{label} code pin drifted")
        _false(item["gate_d_closed"], f"{label}.gate_d_closed")
        _false(item["numerical_claim"], f"{label}.numerical_claim")
        _false(item["performance_claim"], f"{label}.performance_claim")
    for item, label in ((summary, "summary"), (evidence, "evidence")):
        _require(item["run_tag"] == RUN_TAG, f"{label} run tag drifted")
    _require(runner["compile_only"] is True, "runner is not compile-only")
    _require(
        isinstance(runner["compiled_executable_invocation_count"], int)
        and not isinstance(runner["compiled_executable_invocation_count"], bool)
        and runner["compiled_executable_invocation_count"] == 0,
        "compiled executable was invoked",
    )
    _false(
        runner["tpu_numerical_execution_performed"], "runner TPU numerical execution"
    )
    _false(
        summary["tpu_numerical_execution_performed"], "summary TPU numerical execution"
    )
    _require(runner["status"] == "HLO_ACQUIRED_UNADJUDICATED", "runner status drifted")
    _require(summary["status"] == runner["status"], "summary status drifted")
    _require(evidence["status"] == runner["status"], "evidence status drifted")
    _require(marker["status"] == runner["status"], "marker status drifted")
    _false(marker["adjudicated"], "acquisition marker adjudicated")
    _false(marker["gate_d_closed"], "marker Gate D")
    _false(marker["numerical_claim"], "marker numerical claim")
    _false(marker["performance_claim"], "marker performance claim")
    _false(
        marker["tpu_numerical_execution_performed"], "marker TPU numerical execution"
    )

    optimized = files["hlo/compensated_pp16_stage0.optimized_hlo.txt"]
    stable = files["hlo/compensated_pp16_stage0.stablehlo.mlir"]
    _require(
        (_sha(optimized), len(optimized)) == (OPTIMIZED_SHA256, 267335),
        "optimized HLO identity drifted",
    )
    _require(
        (_sha(stable), len(stable)) == (STABLEHLO_SHA256, 77085),
        "StableHLO identity drifted",
    )
    for record, expected in (
        (runner["hlo"]["optimized"], (OPTIMIZED_SHA256, 267335)),
        (runner["hlo"]["stablehlo"], (STABLEHLO_SHA256, 77085)),
    ):
        _require(
            (record["sha256"], record["byte_count"]) == expected,
            "runner HLO receipt drifted",
        )
    _require(
        summary["optimized_hlo_sha256"] == OPTIMIZED_SHA256,
        "summary optimized HLO drifted",
    )
    _require(
        summary["stablehlo_sha256"] == STABLEHLO_SHA256, "summary StableHLO drifted"
    )
    _require(
        runner["admission_report_sha256"] == ADMISSION_SHA256,
        "admission binding drifted",
    )
    _require(
        runner["topology_authority_sha256"] == TOPOLOGY_SHA256,
        "topology binding drifted",
    )
    _require(
        runner["capsule_input_authority"]
        == {
            "capsule_sha256": CAPSULE_SHA256,
            "input_artifact_sha256": CAPSULE_INPUT_SHA256,
            "raw_input_array_count": 14,
        },
        "capsule binding drifted",
    )
    _require(
        tuple(
            (item["name"], tuple(item["shape"]), item["dtype"])
            for item in runner["output_spec"]
        )
        == _RUNNER_OUTPUT_SPEC,
        "runner owner/output specification drifted",
    )
    physical = runner["physical_group"]
    _require(
        physical
        == {
            "coordinates": [[0, 0, 0], [1, 0, 0]],
            "device_ids": [0, 1],
            "local_device_count_visible": 4,
            "mesh_device_count": 2,
            "process_index": 0,
            "stage_id": 0,
        },
        "physical stage-zero LP2 binding drifted",
    )
    runtime = runner["runtime"]
    _require(
        runtime["backend_platform"] == "tpu" and runtime["device_kind"] == "TPU v4",
        "runtime hardware drifted",
    )
    _require(
        (runtime["jax"], runtime["jaxlib"], runtime["libtpu"])
        == ("0.10.1", "0.10.1", "0.0.41"),
        "compiler runtime versions drifted",
    )
    dep_receipt = runner["compiler_dependency_manifest"]
    _require(
        dep_receipt["sha256"] == _sha(files["dependencies.json"]),
        "dependency manifest hash drifted",
    )
    _require(
        dep_receipt["byte_count"] == len(files["dependencies.json"]),
        "dependency manifest size drifted",
    )
    _require(
        (
            dep_receipt["accelerator_device_node_count"],
            dep_receipt["python_module_count"],
            dep_receipt["native_mapping_count"],
        )
        == (4, 682, 50),
        "dependency census drifted",
    )
    _require(dependencies["code_hash"] == CODE_HASH, "dependency code pin drifted")
    _require(
        len(dependencies["accelerator_device_nodes_observed_mapped"]) == 4,
        "accelerator census drifted",
    )
    _require(
        runner["sealed_project_source"]
        == {
            "archive_sha256": "2c8eb1bbb7884130edf4edea388164442b37c5d19fa38f4e028ca1ef7eb55eef",
            "file_manifest_count": 142,
            "file_manifest_sha256": "0f07643b95b7ed4f2f6b4aa9237c64e6bb43e68514d3865dbf9470b573dd24f4",
            "loaded_module_count": 108,
        },
        "sealed project source receipt drifted",
    )

    admission = _read_regular(
        source_root
        / "docs/artifacts/gate-d-precompile-admission-v2-compensated-capsule.json"
    )
    topology = _read_regular(
        source_root / "docs/artifacts/gate-d-runtime-locality-authority.json"
    )
    _require(_sha(admission) == ADMISSION_SHA256, "source admission authority drifted")
    _require(_sha(topology) == TOPOLOGY_SHA256, "source topology authority drifted")
    git_head = (
        subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
        .stdout.decode()
        .strip()
    )
    _require(git_head == CODE_HASH, "source worktree pin drifted")

    evidence_records = {item["path"]: item for item in evidence["files"]}
    _require(
        len(evidence_records) == 14 == len(evidence["files"]),
        "local evidence catalogue drifted",
    )
    for path, record in evidence_records.items():
        _require(path in files, f"evidence path absent: {path}")
        _require(
            (record["sha256"], record["byte_count"])
            == (_sha(files[path]), len(files[path])),
            f"local evidence receipt drifted: {path}",
        )
    _require(
        marker["evidence_sha256"] == _sha(files["evidence.json"]),
        "marker evidence hash drifted",
    )
    _require(
        marker["summary_sha256"] == _sha(files["summary.json"]),
        "marker summary hash drifted",
    )
    remote_ledger = marker["remote_ledger"]
    _require(
        remote_ledger
        == {
            "crc32c": "aP8UHA==",
            "generation": str(LEDGER_GENERATION),
            "path": "remote_objects.json",
            "sha256": LEDGER_SHA256,
            "size": 2701,
        },
        "marker ledger receipt drifted",
    )
    terminal = receipt["terminal"]
    _require(
        terminal
        == {
            "crc32c": "HK0N6Q==",
            "generation": str(TERMINAL_GENERATION),
            "path": "HLO_ACQUIRED",
            "sha256": TERMINAL_SHA256,
            "size": 742,
        },
        "terminal receipt drifted",
    )
    _require(
        receipt["remote"] == REMOTE_PREFIX + "/HLO_ACQUIRED",
        "terminal remote path drifted",
    )
    return {"runner": runner, "marker": marker, "receipt": receipt}


def _balanced_end(value: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(value)):
        if value[index] == "(":
            depth += 1
        elif value[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    raise Refusal("unbalanced optimized HLO instruction")


@dataclass(frozen=True)
class Instruction:
    computation: str
    name: str
    result_type: str
    opcode: str
    operands: tuple[str, ...]
    attributes: str
    root: bool
    raw: str


@dataclass(frozen=True)
class HloGraph:
    header: str
    entry: str
    computations: Mapping[str, tuple[Instruction, ...]]

    def by_name(self, computation: str) -> dict[str, Instruction]:
        items = self.computations[computation]
        result = {item.name: item for item in items}
        _require(len(result) == len(items), f"duplicate instruction in {computation}")
        return result


def parse_optimized_hlo(text: str) -> HloGraph:
    lines = text.splitlines()
    _require(
        lines and lines[0].startswith("HloModule "), "optimized HLO header missing"
    )
    current: str | None = None
    entry: str | None = None
    computations: dict[str, list[Instruction]] = defaultdict(list)
    for raw in lines[1:]:
        line = raw.strip()
        if not line:
            continue
        if line.endswith("{") and " = " not in line:
            match = re.match(r"(?:ENTRY\s+)?(%[A-Za-z0-9_.-]+)\s*\(", line)
            if match:
                current = match.group(1)
                if line.startswith("ENTRY "):
                    _require(entry is None, "multiple ENTRY computations")
                    entry = current
            continue
        if line == "}":
            current = None
            continue
        if current is None or " = " not in line:
            continue
        root = line.startswith("ROOT ")
        body = line[5:] if root else line
        name, rhs = body.split(" = ", 1)
        opcode_match = re.search(r"\b([a-z][a-z0-9-]*)\(", rhs)
        _require(opcode_match is not None, f"cannot parse opcode: {line}")
        opening = opcode_match.end() - 1
        closing = _balanced_end(rhs, opening)
        result_type = rhs[: opcode_match.start()].strip()
        operands = tuple(re.findall(r"%[A-Za-z0-9_.-]+", rhs[opening + 1 : closing]))
        computations[current].append(
            Instruction(
                current,
                name,
                result_type,
                opcode_match.group(1),
                operands,
                rhs[closing + 1 :],
                root,
                line,
            )
        )
    _require(entry is not None, "ENTRY computation absent")
    _require(computations.get(entry), "ENTRY instructions absent")
    return HloGraph(
        lines[0], entry, {key: tuple(value) for key, value in computations.items()}
    )


def _shape(value: str) -> tuple[str, tuple[int, ...]]:
    match = re.match(r"([a-z0-9]+)\[([0-9,]*)\]", value)
    _require(match is not None, f"unsupported HLO shape: {value}")
    dims = (
        ()
        if not match.group(2)
        else tuple(int(item) for item in match.group(2).split(","))
    )
    return match.group(1), dims


def _calls(instruction: Instruction) -> str:
    match = re.search(r"\bcalls=(%[A-Za-z0-9_.-]+)", instruction.attributes)
    _require(match is not None, f"fusion call target absent: {instruction.name}")
    return match.group(1)


def _index(instruction: Instruction) -> int:
    match = re.search(r"\bindex=([0-9]+)", instruction.attributes)
    _require(match is not None, f"tuple index absent: {instruction.name}")
    return int(match.group(1))


def _parameter_number(instruction: Instruction) -> int:
    match = re.search(r"\bparameter\(([0-9]+)\)", instruction.raw)
    _require(match is not None, f"parameter number absent: {instruction.name}")
    return int(match.group(1))


def _consumers(items: Iterable[Instruction]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for item in items:
        for operand in item.operands:
            result[operand].add(item.name)
    return result


def _ancestors(root: str, by_name: Mapping[str, Instruction]) -> set[str]:
    seen: set[str] = set()
    pending = [root]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        item = by_name.get(name)
        if item:
            pending.extend(item.operands)
    return seen


def _validate_fusion_bodies(
    graph: HloGraph,
    source_fusion: Instruction,
    subtract_fusion: Instruction,
    add_fusion: Instruction,
) -> dict[str, str]:
    source_name, subtract_name, add_name = map(
        _calls, (source_fusion, subtract_fusion, add_fusion)
    )
    source = graph.by_name(source_name)
    sub = graph.by_name(subtract_name)
    add = graph.by_name(add_name)

    source_root = next((item for item in source.values() if item.root), None)
    _require(
        source_root is not None
        and source_root.opcode == "tuple"
        and len(source_root.operands) == 2,
        "RMS source fusion root drifted",
    )
    source_value = source[source_root.operands[1]]
    _require(
        source_value.opcode == "add"
        and _shape(source_value.result_type) == ("f32", (1, 6144)),
        "RMS FP32 source add drifted",
    )
    for operand in source_value.operands:
        _require(
            source[operand].opcode == "convert"
            and _shape(source[operand].result_type) == ("f32", (1, 6144)),
            "RMS source conversion drifted",
        )
        parent = source[source[operand].operands[0]]
        _require(
            parent.opcode == "parameter"
            and _shape(parent.result_type) == ("bf16", (1, 6144)),
            "RMS operand source drifted",
        )
    reduce_value = source[source_root.operands[0]]
    _require(
        reduce_value.opcode == "reduce"
        and source_value.name in _ancestors(reduce_value.name, source),
        "RMS reduction no longer shares exact FP32 input",
    )

    sub_root = next((item for item in sub.values() if item.root), None)
    _require(
        sub_root is not None
        and sub_root.opcode == "subtract"
        and _shape(sub_root.result_type) == ("f32", (1, 6144)),
        "compensation subtract drifted",
    )
    original, rounded = sub_root.operands
    _require(
        sub[original].opcode == "parameter" and _parameter_number(sub[original]) == 0,
        "compensation original source drifted",
    )
    _require(
        sub[rounded].opcode == "convert", "compensation FP32 restore convert absent"
    )
    bf16_round = sub[sub[rounded].operands[0]]
    _require(
        bf16_round.opcode == "convert"
        and bf16_round.operands == (original,)
        and _shape(bf16_round.result_type) == ("bf16", (1, 6144)),
        "compensation BF16 round drifted",
    )

    add_root = next((item for item in add.values() if item.root), None)
    _require(
        add_root is not None
        and add_root.opcode == "add"
        and _shape(add_root.result_type) == ("f32", (1, 6144)),
        "restored witness add drifted",
    )
    rounded_add, correction = add_root.operands
    _require(
        add[correction].opcode == "parameter"
        and _parameter_number(add[correction]) == 0,
        "restored correction source drifted",
    )
    _require(
        add[rounded_add].opcode == "convert", "restored rounded FP32 convert absent"
    )
    bf16_add = add[add[rounded_add].operands[0]]
    _require(
        bf16_add.opcode == "convert"
        and _shape(bf16_add.result_type) == ("bf16", (1, 6144)),
        "restored BF16 round drifted",
    )
    original_add = add[bf16_add.operands[0]]
    _require(
        original_add.opcode == "parameter" and _parameter_number(original_add) == 1,
        "restored original source drifted",
    )
    return {"source": source_name, "subtract": subtract_name, "restore": add_name}


def validate_optimized_hlo(text: str) -> dict[str, Any]:
    graph = parse_optimized_hlo(text)
    _require(
        "num_partitions=2" in graph.header, "optimized HLO is not two-partition SPMD"
    )
    entry = graph.by_name(graph.entry)
    root = next((item for item in entry.values() if item.root), None)
    _require(
        root is not None and root.opcode == "tuple" and len(root.operands) == 10,
        "ENTRY root tuple drifted",
    )
    _require(
        tuple(_shape(entry[name].result_type) for name in root.operands)
        == _ROOT_OUTPUTS,
        "partition-local root shapes drifted",
    )

    collectives = [
        item
        for items in graph.computations.values()
        for item in items
        if item.opcode in _COLLECTIVES
        or item.opcode.removesuffix("-start") in _COLLECTIVES
    ]
    _require(
        len(collectives) == 3
        and all(item.opcode == "all-gather" for item in collectives),
        "physical collective surface drifted",
    )
    expected_collectives = {
        (2, ("f32", (2, 1, 2064)), ("f32", (1, 1, 2064))),
        (3, ("s32", (2, 1, 2048)), ("s32", (1, 1, 2048))),
        (4, ("f32", (2, 1, 2048)), ("f32", (1, 1, 2048))),
    }
    observed_collectives = set()
    all_items = {
        item.name: item for items in graph.computations.values() for item in items
    }
    for item in collectives:
        channel = re.search(r"\bchannel_id=([0-9]+)", item.attributes)
        _require(channel is not None, "collective channel absent")
        _require(
            "replica_groups={{0,1}}" in item.attributes,
            "collective replica group drifted",
        )
        _require("dimensions={0}" in item.attributes, "collective dimension drifted")
        _require(
            "use_global_device_ids=true" in item.attributes,
            "collective lacks global ids",
        )
        _require(
            len(item.operands) == 1 and item.operands[0] in all_items,
            "collective operand drifted",
        )
        observed_collectives.add(
            (
                int(channel.group(1)),
                _shape(item.result_type),
                _shape(all_items[item.operands[0]].result_type),
            )
        )
    _require(
        observed_collectives == expected_collectives, "collective signatures drifted"
    )

    lowered = text.lower()
    for token in _FORBIDDEN_TOKENS:
        _require(token not in lowered, f"forbidden host/transport token: {token}")

    customs = [
        item
        for items in graph.computations.values()
        for item in items
        if item.opcode == "custom-call"
    ]
    signatures = Counter()
    for item in customs:
        target = re.search(r'custom_call_target="([^"]+)"', item.attributes)
        _require(target is not None, "custom-call target absent")
        operand_shapes = tuple(
            _shape(all_items[name].result_type) for name in item.operands
        )
        signatures[(target.group(1), _shape(item.result_type), operand_shapes)] += 1
    expected_customs = Counter(
        {
            ("AssumeGatherIndicesInBound", ("s32", (4096,)), (("s32", (4096,)),)): 4,
            ("AssumeGatherIndicesInBound", ("s32", (2048,)), (("s32", (2048,)),)): 2,
            ("AssumeGatherIndicesInBound", ("s32", (1024,)), (("s32", (1024,)),)): 1,
            ("AllocateBuffer", ("f32", (2, 1, 1024)), ()): 1,
            ("AllocateBuffer", ("bf16", (32, 1, 82)), ()): 1,
            (
                "ConcatBitcast",
                ("bf16", (16, 256, 128)),
                (("bf16", (4, 256, 128)),) * 4,
            ): 1,
        }
    )
    _require(
        signatures == expected_customs,
        "optimized custom-call allowlist/signatures drifted",
    )

    # The exact graph has one tuple-producing RMS source fusion. Identify it from
    # its GTE index-1 consumer instead of trusting compiler-generated names.
    source_candidates = []
    for item in entry.values():
        if (
            item.opcode == "get-tuple-element"
            and _index(item) == 1
            and _shape(item.result_type) == ("f32", (1, 6144))
        ):
            parent = entry.get(item.operands[0])
            if parent and parent.opcode == "fusion" and len(parent.operands) == 2:
                source_candidates.append((parent, item))
    _require(len(source_candidates) == 1, "unique RMS FP32 source fusion not found")
    source_fusion, source_value = source_candidates[0]
    consumers = _consumers(entry.values())
    source_users = consumers[source_value.name]
    sub_candidates = [
        entry[name]
        for name in source_users
        if entry[name].opcode == "fusion"
        and 'op_name="jit(mapped)/shard_map/sub"' in entry[name].attributes
    ]
    add_candidates = [
        item
        for item in entry.values()
        if item.opcode == "fusion"
        and source_value.name in item.operands
        and any(
            operand in {candidate.name for candidate in sub_candidates}
            for operand in item.operands
        )
    ]
    _require(
        len(sub_candidates) == 1 and len(add_candidates) == 1,
        "compensated witness ENTRY chain drifted",
    )
    subtract_fusion, add_fusion = sub_candidates[0], add_candidates[0]
    _require(
        subtract_fusion.operands == (source_value.name,),
        "compensation ENTRY operand binding drifted",
    )
    _require(
        add_fusion.operands == (subtract_fusion.name, source_value.name),
        "restore ENTRY operand binding drifted",
    )
    _require(
        consumers[subtract_fusion.name] == {add_fusion.name},
        "compensation correction escaped auxiliary chain",
    )
    bitcast_candidates = [
        entry[name]
        for name in consumers[add_fusion.name]
        if entry[name].opcode == "bitcast"
    ]
    _require(
        len(bitcast_candidates) == 1
        and consumers[add_fusion.name] == {bitcast_candidates[0].name},
        "restored witness root path drifted",
    )
    witness_root = bitcast_candidates[0]
    _require(
        root.operands[0] == witness_root.name
        and consumers[witness_root.name] == {root.name},
        "restored witness is not root slot zero only",
    )
    _require(
        source_users
        == {subtract_fusion.name, add_fusion.name, "%reshape_multiply_fusion"},
        "RMS FP32 source consumer set drifted",
    )
    _require(
        "%reshape_multiply_fusion" in entry
        and entry["%reshape_multiply_fusion"].opcode == "fusion",
        "primary RMS path absent",
    )
    auxiliary = {subtract_fusion.name, add_fusion.name, witness_root.name}
    for index, operand in enumerate(root.operands[1:], start=1):
        _require(
            not (_ancestors(operand, entry) & auxiliary),
            f"auxiliary contaminates primary root slot {index}",
        )
    fusion_bodies = _validate_fusion_bodies(
        graph, source_fusion, subtract_fusion, add_fusion
    )
    return {
        "collective_count": 3,
        "collectives": [
            {
                "channel_id": channel,
                "opcode": "all-gather",
                "replica_groups": [[0, 1]],
                "dimension": 0,
                "use_global_device_ids": True,
                "result": {"dtype": result[0], "shape": list(result[1])},
                "operand": {"dtype": operand[0], "shape": list(operand[1])},
            }
            for channel, result, operand in sorted(observed_collectives)
        ],
        "custom_call_counts": dict(
            sorted(Counter(key[0] for key in signatures.elements()).items())
        ),
        "fusion_bodies": fusion_bodies,
        "logical_row_axis": 1,
        "num_partitions": 2,
        "owner_semantics": "partition-derived-from-shard-map-and-stage0-topology;not-a-tensor-row-axis",
        "partition_local_root_shapes": [
            {"dtype": dtype, "shape": list(shape)} for dtype, shape in _ROOT_OUTPUTS
        ],
        "primary_noninterference": True,
        "rooted_compensated_witness": True,
    }


def validate_stablehlo(text: str) -> dict[str, Any]:
    _require(
        "mhlo.num_partitions = 2" in text
        and 'sdy.mesh @mesh = <["feature"=2]>' in text,
        "StableHLO mesh drifted",
    )
    _require(
        "tensor<1x6144xbf16>" in text and "tensor<32x6144xbf16>" not in text,
        "StableHLO logical row contract drifted",
    )
    manual = [
        line.strip() for line in text.splitlines() if "sdy.manual_computation" in line
    ]
    _require(len(manual) == 1, "StableHLO manual owner computation drifted")
    _require(
        " out_shardings=[" in manual[0] and "] manual_axes=" in manual[0],
        "StableHLO owner output sharding syntax drifted",
    )
    output_shardings = (
        manual[0].split(" out_shardings=[", 1)[1].split("] manual_axes=", 1)[0]
    )
    _require(
        'manual_axes={"feature"}' in manual[0]
        and output_shardings.count('<@mesh, [{"feature"}') == 10,
        "StableHLO owner output shardings drifted",
    )
    expected_chain = """      %5 = stablehlo.convert %arg14 : (tensor<1x6144xbf16>) -> tensor<1x6144xf32>
      %6 = stablehlo.convert %arg15 : (tensor<1x6144xbf16>) -> tensor<1x6144xf32>
      %7 = stablehlo.add %5, %6 : tensor<1x6144xf32>
      %8 = stablehlo.convert %7 : (tensor<1x6144xf32>) -> tensor<1x6144xbf16>
      %9 = stablehlo.convert %8 : (tensor<1x6144xbf16>) -> tensor<1x6144xf32>
      %10 = stablehlo.subtract %7, %9 : tensor<1x6144xf32>
      %11 = stablehlo.optimization_barrier %10 : tensor<1x6144xf32>
      %12 = stablehlo.add %9, %11 : tensor<1x6144xf32>
      %13 = stablehlo.optimization_barrier %12 : tensor<1x6144xf32>"""
    _require(
        text.count(expected_chain) == 1, "StableHLO compensated causal slice drifted"
    )
    all_gathers = [
        line.strip() for line in text.splitlines() if "stablehlo.all_gather" in line
    ]
    _require(len(all_gathers) == 3, "StableHLO all-gather count drifted")
    _require(
        all(
            "replica_groups = dense<[[0, 1]]>" in line
            and "use_global_device_ids" in line
            for line in all_gathers
        ),
        "StableHLO collective locality drifted",
    )
    for token in (
        "stablehlo.infeed",
        "stablehlo.outfeed",
        "stablehlo.send",
        "stablehlo.recv",
        "stablehlo.custom_call",
    ):
        _require(token not in text, f"forbidden StableHLO effect: {token}")
    return {
        "all_gather_count": 3,
        "compensated_slice_rooted": True,
        "logical_rows": 1,
        "manual_axes": ["feature"],
        "mesh_axis": "feature",
        "mesh_size": 2,
        "owner_output_sharding_count": 10,
    }


_CRC32C_TABLE: tuple[int, ...] | None = None


def _crc32c(payload: bytes) -> str:
    global _CRC32C_TABLE
    if _CRC32C_TABLE is None:
        table = []
        polynomial = 0x82F63B78
        for value in range(256):
            crc = value
            for _ in range(8):
                crc = (crc >> 1) ^ (polynomial if crc & 1 else 0)
            table.append(crc)
        _CRC32C_TABLE = tuple(table)
    crc = 0xFFFFFFFF
    for value in payload:
        crc = _CRC32C_TABLE[(crc ^ value) & 0xFF] ^ (crc >> 8)
    return base64.b64encode(struct.pack(">I", crc ^ 0xFFFFFFFF)).decode()


def _gcloud(
    gcloud: Path, arguments: Sequence[str], *, allow_exact_vacant: bool = False
) -> bytes:
    result = subprocess.run(
        [str(gcloud), "storage", *arguments],
        check=False,
        capture_output=True,
        env={
            "CLOUDSDK_CORE_DISABLE_PROMPTS": "1",
            "HOME": "/home/gianl",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/snap/bin:/usr/bin:/bin",
            "PYTHONWARNINGS": "ignore",
        },
    )
    vacant = b"ERROR: (gcloud.storage.ls) One or more URLs matched no objects.\n"
    if (
        allow_exact_vacant
        and result.returncode == 1
        and result.stdout == b""
        and result.stderr == vacant
    ):
        return b"[]"
    _require(
        result.returncode == 0,
        f"gcloud failed: {' '.join(arguments)}: {result.stderr.decode(errors='replace')}",
    )
    return result.stdout


def replay_remote(files: Mapping[str, bytes], gcloud: Path) -> dict[str, Any]:
    ledger = _json(files["remote_objects.json"], "local remote ledger")
    _require(
        set(ledger) == {"artifact_kind", "objects", "run_tag"}
        and ledger["artifact_kind"] == "gate_d_compensated_pp16_hlo_remote_ledger"
        and ledger["run_tag"] == RUN_TAG
        and len(ledger["objects"]) == 15,
        "remote ledger catalogue drifted",
    )
    _require(
        {item.get("path") for item in ledger["objects"]} == _REMOTE_LEDGER_PATHS,
        "remote ledger path set drifted",
    )
    records = list(ledger["objects"])
    records.append(
        {
            "path": "remote_objects.json",
            "generation": str(LEDGER_GENERATION),
            "sha256": LEDGER_SHA256,
            "size": 2701,
            "crc32c": "aP8UHA==",
        }
    )
    records.append(
        {
            "path": "HLO_ACQUIRED",
            "generation": str(TERMINAL_GENERATION),
            "sha256": TERMINAL_SHA256,
            "size": 742,
            "crc32c": "HK0N6Q==",
        }
    )
    _require(
        len({item["path"] for item in records}) == 17,
        "remote path catalogue has duplicates",
    )
    replayed = []
    for item in records:
        _require(
            set(item) == {"crc32c", "generation", "path", "sha256", "size"},
            "remote object receipt schema drifted",
        )
        path = item["path"]
        _require(isinstance(path, str) and path, "remote object path is invalid")
        generation = _decimal_generation(
            item["generation"], f"invalid generation for {path}"
        )
        size = _positive_int(item["size"], f"invalid size for {path}")
        _require(
            isinstance(item["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None,
            f"invalid SHA-256 for {path}",
        )
        _require(
            isinstance(item["crc32c"], str)
            and re.fullmatch(r"[A-Za-z0-9+/]{6}==", item["crc32c"]) is not None,
            f"invalid CRC32C for {path}",
        )
        payload = _gcloud(gcloud, ["cat", f"{REMOTE_PREFIX}/{path}#{generation}"])
        _require(
            (len(payload), _sha(payload), _crc32c(payload))
            == (size, item["sha256"], item["crc32c"]),
            f"generation replay drifted: {path}",
        )
        if path in files:
            _require(
                payload == files[path],
                f"remote/local bytes differ: {path}",
            )
        replayed.append(
            {
                "path": path,
                "generation": generation,
                "size": len(payload),
                "sha256": _sha(payload),
                "crc32c": _crc32c(payload),
            }
        )

    listing = _json(
        _gcloud(gcloud, ["ls", "--json", "--all-versions", REMOTE_PREFIX + "/**"]),
        "all-version listing",
    )
    expected_urls = {
        f"{REMOTE_PREFIX}/{item['path']}#{item['generation']}" for item in replayed
    }
    _require(isinstance(listing, list), "all-version listing is not a list")
    replay_by_path = {item["path"]: item for item in replayed}
    observed_urls = set()
    remote_name_prefix = REMOTE_PREFIX.removeprefix("gs://driftbench-dsv4-uc/") + "/"
    for item in listing:
        _require(
            isinstance(item, dict) and set(item) == {"metadata", "type", "url"},
            "all-version listing item schema drifted",
        )
        _require(
            item["type"] == "cloud_object" and isinstance(item["url"], str),
            "all-version listing item type drifted",
        )
        metadata = item["metadata"]
        _require(
            isinstance(metadata, dict)
            and all(
                isinstance(metadata.get(key), str)
                for key in ("bucket", "crc32c", "generation", "name", "size")
            ),
            "all-version listing metadata schema drifted",
        )
        _require(
            metadata["bucket"] == "driftbench-dsv4-uc"
            and metadata["name"].startswith(remote_name_prefix),
            "all-version listing object identity drifted",
        )
        relative = metadata["name"].removeprefix(remote_name_prefix)
        _require(relative in replay_by_path, "all-version listing path drifted")
        record = replay_by_path[relative]
        generation = _decimal_generation(
            metadata["generation"], f"listing generation for {relative}"
        )
        expected_url = f"gs://{metadata['bucket']}/{metadata['name']}#{generation}"
        _require(
            item["url"] == expected_url, f"listing URL/metadata drifted: {relative}"
        )
        observed_urls.add(item["url"])
        _require(
            (
                generation,
                _decimal_generation(metadata["size"], f"listing size for {relative}"),
                metadata["crc32c"],
            )
            == (record["generation"], record["size"], record["crc32c"]),
            f"listing metadata drifted: {relative}",
        )
    _require(
        observed_urls == expected_urls and len(listing) == 17,
        "live/all-version remote catalogue drifted",
    )
    soft = _json(
        _gcloud(
            gcloud,
            ["ls", "--json", "--soft-deleted", REMOTE_PREFIX + "/**"],
            allow_exact_vacant=True,
        ),
        "soft-deleted listing",
    )
    _require(soft == [], "soft-deleted remote objects exist")
    generations = [item["generation"] for item in replayed]
    _require(
        max(generations) == TERMINAL_GENERATION
        and generations[-1] == TERMINAL_GENERATION,
        "terminal is not strictly last",
    )
    _require(
        LEDGER_GENERATION < TERMINAL_GENERATION
        and max(item["generation"] for item in replayed[:-2]) < LEDGER_GENERATION,
        "preterminal generation ordering drifted",
    )
    return {
        "all_version_object_count": 17,
        "generation_replay_count": 17,
        "ledger_generation": LEDGER_GENERATION,
        "soft_deleted_object_count": 0,
        "terminal_generation": TERMINAL_GENERATION,
        "terminal_strictly_last": True,
    }


def adjudicate(run_dir: Path, source_root: Path, gcloud: Path) -> dict[str, Any]:
    tool_source = _read_regular(Path(__file__).resolve())
    files = _inventory(run_dir)
    acquisition = _validate_acquisition(files, source_root)
    optimized = validate_optimized_hlo(
        files["hlo/compensated_pp16_stage0.optimized_hlo.txt"].decode()
    )
    stable = validate_stablehlo(
        files["hlo/compensated_pp16_stage0.stablehlo.mlir"].decode()
    )
    remote = replay_remote(files, gcloud)
    return {
        "artifact_kind": "gate_d_compensated_pp16_hlo_adjudication",
        "classification": "HLO_LOCALITY_POLICY_ACCEPTED;TPU_NUMERICAL_UNPROVEN;NO_PERFORMANCE_CLAIM;GATE_D_OPEN",
        "claim_scope": "Exact compile-only compensated PP16 stage-zero optimized-HLO locality, rooted auxiliary witness, primary noninterference, compiler provenance, and generation-qualified archive integrity. The executable was never invoked; this is not TPU numerical equality, decoder correctness, performance, or Gate-D closure.",
        "code_hash": CODE_HASH,
        "gate_d_closed": False,
        "hlo": {
            "optimized": {
                "sha256": OPTIMIZED_SHA256,
                "byte_count": 267335,
                **optimized,
            },
            "stablehlo": {"sha256": STABLEHLO_SHA256, "byte_count": 77085, **stable},
        },
        "numerical_claim": False,
        "optimized_hlo_locality_policy_accepted": True,
        "performance_claim": False,
        "provenance": {
            "admission_sha256": ADMISSION_SHA256,
            "capsule_input_sha256": CAPSULE_INPUT_SHA256,
            "capsule_sha256": CAPSULE_SHA256,
            "compiler_dependency_manifest_sha256": acquisition["runner"][
                "compiler_dependency_manifest"
            ]["sha256"],
            "topology_authority_sha256": TOPOLOGY_SHA256,
        },
        "remote_archive": remote,
        "run_tag": RUN_TAG,
        "schema_version": 1,
        "tpu_numerical_execution_performed": False,
        "tpu_numerical_proven": False,
        "tpu_successor_authorized": False,
        "validator": {
            "byte_count": len(tool_source),
            "path": "scripts/greenfield/adjudicate_gate_d_compensated_pp16_hlo.py",
            "sha256": _sha(tool_source),
            "stdlib_only": True,
        },
    }


def _descriptor_bytes(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    while block := os.read(descriptor, 1024 * 1024):
        chunks.append(block)
    return b"".join(chunks)


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
    )


def _directory_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
    )


def _write_exclusive(output_root: Path, payload: bytes) -> Path:
    _require(output_root.is_absolute(), "output root must be absolute")
    root_descriptor = os.open(
        output_root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    try:
        root_before = os.fstat(root_descriptor)
        _require(stat.S_ISDIR(root_before.st_mode), "output root is not a directory")
        _require(
            stat.S_IMODE(root_before.st_mode) == 0o700
            and root_before.st_uid == os.geteuid()
            and root_before.st_gid == os.getegid(),
            "output root ownership/mode is unsafe",
        )
        _require(os.listdir(root_descriptor) == [], "output root is not vacant")
        descriptor = os.open(
            "report.json",
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o400,
            dir_fd=root_descriptor,
        )
        try:
            created = os.fstat(descriptor)
            _require(
                stat.S_ISREG(created.st_mode)
                and created.st_nlink == 1
                and created.st_uid == os.geteuid()
                and created.st_gid == os.getegid(),
                "created report identity is unsafe",
            )
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                _require(written > 0, "output write stalled")
                view = view[written:]
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
            sealed = os.fstat(descriptor)
            _require(
                stat.S_IMODE(sealed.st_mode) == 0o400
                and sealed.st_size == len(payload),
                "sealed report metadata drifted",
            )
            _require(
                _descriptor_bytes(descriptor) == payload,
                "same-descriptor report replay drifted",
            )
            named_descriptor = os.open(
                "report.json",
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=root_descriptor,
            )
            try:
                named = os.fstat(named_descriptor)
                _require(
                    _file_identity(named) == _file_identity(sealed),
                    "named report inode drifted",
                )
                _require(
                    _descriptor_bytes(named_descriptor) == payload,
                    "named report replay drifted",
                )
            finally:
                os.close(named_descriptor)
            os.fsync(root_descriptor)
            root_after = os.fstat(root_descriptor)
            _require(
                _directory_identity(root_after) == _directory_identity(root_before)
                and os.listdir(root_descriptor) == ["report.json"],
                "output root identity/catalogue drifted",
            )
        finally:
            os.close(descriptor)
    finally:
        os.close(root_descriptor)
    return output_root / "report.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--gcloud", type=Path, default=Path("/snap/bin/gcloud"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = adjudicate(args.run_dir, args.source_root, args.gcloud)
        payload = _canonical(report)
        output = _write_exclusive(args.output_root, payload)
    except (
        OSError,
        KeyError,
        Refusal,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        print(f"HLO_ADJUDICATION_REFUSED {error}", file=os.sys.stderr)
        return 1
    print(f"HLO_ADJUDICATION_ACCEPTED output={output} sha256={_sha(payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
