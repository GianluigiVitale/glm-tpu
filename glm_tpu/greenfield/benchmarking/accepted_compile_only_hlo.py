"""Admission checks for one accepted-DB485 compile-only TPU HLO capture.

The capture is useful only when the freshly compiled executables are the
same class as the accepted DB485 oracle.  This module therefore treats all
three XLA fingerprints as an ordered identity gate and refuses any evidence
that contains a generation marker.  A passing report is compiler-mechanism
evidence only: it has no numerical, performance, DB, or Gate-D claim.
"""

from __future__ import annotations

import gzip
import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Any

from ..errors import BenchmarkValidationError
from .callback_executable_class import ACCEPTED_FINGERPRINTS, TOKEN_BUCKETS

ACCEPTED_FULL_CODE_PIN = "b3c25df47ac98783912dc658878181ec0a8ae16d"
ACCEPTED_SHORT_CODE_PIN = ACCEPTED_FULL_CODE_PIN[:12]
ACCEPTED_VLLM_PIN = "a30addc7548a9a8b9b3323a7bc3eb7d7c4895d1c"
ACCEPTED_VLLM_VERSION = "0.1.dev1+ga30addc75"
ACCEPTED_VLLM_VERSION_FILE_SHA256 = (
    "91651499622913dc1cf3eb6418f60a7f99e4c3f23231fe4ce3609f44d018e50a"
)
ACCEPTED_ENGINE_KWARGS = {
    "async_scheduling": False,
    "decode_context_parallel_size": 1,
    "disable_log_stats": False,
    "distributed_executor_backend": "ray",
    "dtype": "bfloat16",
    "enable_expert_parallel": True,
    "enable_prefix_caching": False,
    "gpu_memory_utilization": 0.90,
    "kv_cache_dtype": "auto",
    "load_format": "runai_streamer",
    "max_model_len": 8704,
    "max_num_batched_tokens": 2048,
    "max_num_seqs": 1,
    "model": "gs://driftbench-dsv4-uc/models/GLM-5.2-FP8",
    "model_loader_extra_config": {
        "concurrency": 32,
        "memory_limit": 32 * 1024**3,
    },
    "num_gpu_blocks_override": 24,
    "tensor_parallel_size": 32,
    "trust_remote_code": True,
}
ACCEPTED_RUNTIME_CONFIG = {
    "engine_kwargs": ACCEPTED_ENGINE_KWARGS,
    "tpu_inference_pin": ACCEPTED_FULL_CODE_PIN,
    "vllm_pin": ACCEPTED_VLLM_PIN,
}
ACCEPTED_RUNTIME_CONFIG_JSON = json.dumps(
    ACCEPTED_RUNTIME_CONFIG, separators=(",", ":"), sort_keys=True
)
ACCEPTED_RUNTIME_CONFIG_SHA256 = sha256(
    ACCEPTED_RUNTIME_CONFIG_JSON.encode()
).hexdigest()
ARTIFACT_KIND = "greenfield_accepted_db485_compile_only_hlo"
CLAIM_SCOPE = {
    "db_run_id": None,
    "gate_d_claim": False,
    "numerical_claim": False,
    "performance_claim": False,
}

_PID_RE = re.compile(r"\(EngineCore pid=(?P<pid>\d+)\)")
_FINGERPRINT_RE = re.compile(
    r"\(HLO module jit_step_fun_impl\): "
    r"(?P<kind>Executable fingerprint \(including data segments\)|"
    r"Executable fingerprint|Host transfer fingerprint):"
    r"(?P<digest>[0-9a-f]{64})"
)
_BUCKET_RE = re.compile(
    r"Compilation of worker0 backbone --> \{'num_tokens': (?P<tokens>\d+), "
    r"'num_reqs': 1\} finished"
)
_HLO_FILENAME_RE = re.compile(
    r"^jit_step_fun_impl\.m(?P<tokens>\d+)\."
    r"(?P<raw>module_(?P<module>\d+)\.jit_step_fun_impl\."
    r"cl_\d+\.after_codegen\.txt)\.gz$"
)
_OWNER_RECEIPT_RE = re.compile(
    r"^(?P<kind>HLO_OWNER|HLO_NONOWNER) "
    r"(?P<host>[A-Za-z0-9.-]+-w-(?P<worker>[0-7]))"
    r"(?: count=(?P<count>\d+))?$"
)
_FORBIDDEN_EXECUTION_MARKERS = (
    "[longctx]",
    "correct=True",
    "running one exact",
    "raw passkey item",
)


def _append_distinct(values: list[Any], value: Any) -> None:
    if not values or values[-1] != value:
        values.append(value)


def _ordered_triples(
    records: list[tuple[str, str]],
) -> tuple[tuple[str, str, str], ...]:
    kinds = (
        "Executable fingerprint",
        "Executable fingerprint (including data segments)",
        "Host transfer fingerprint",
    )
    if len(records) % len(kinds):
        raise BenchmarkValidationError("compile-only fingerprint triple count drifted")
    triples: list[tuple[str, str, str]] = []
    for offset in range(0, len(records), len(kinds)):
        group = records[offset : offset + len(kinds)]
        if tuple(kind for kind, _ in group) != kinds:
            raise BenchmarkValidationError(
                "compile-only fingerprint triple order drifted"
            )
        triple = tuple(digest for _, digest in group)
        _append_distinct(triples, triple)
    return tuple(triples)


def validate_compile_only_log(raw: bytes, *, run_tag: str) -> dict[str, Any]:
    """Authenticate a no-generation compile and enforce DB485 executable identity."""

    text = raw.decode("utf-8", errors="strict")
    start = (
        f"ACCEPTED_DB485_COMPILE_ONLY_START tag={run_tag} "
        f"code={ACCEPTED_FULL_CODE_PIN} generate_calls=0"
    )
    complete = f"ACCEPTED_DB485_COMPILE_ONLY_COMPLETE tag={run_tag} generate_calls=0"
    if text.count(start) != 1 or text.count(complete) != 1:
        raise BenchmarkValidationError("compile-only lifecycle markers drifted")
    if any(marker in text for marker in _FORBIDDEN_EXECUTION_MARKERS):
        raise BenchmarkValidationError("compile-only log contains a generation marker")
    runtime_marker = (
        "ACCEPTED_DB485_RUNTIME_CONFIG "
        f"sha256={ACCEPTED_RUNTIME_CONFIG_SHA256} "
        f"json={ACCEPTED_RUNTIME_CONFIG_JSON}"
    )
    if text.count(runtime_marker) != 1:
        raise BenchmarkValidationError("compile-only runtime configuration drifted")

    records_by_pid: dict[int, list[tuple[str, str]]] = {}
    buckets_by_pid: dict[int, list[int]] = {}
    code_lines_by_pid: dict[int, list[str]] = {}
    config_lines_by_pid: dict[int, list[str]] = {}
    for line in text.splitlines():
        pid_match = _PID_RE.search(line)
        if pid_match is None:
            continue
        pid = int(pid_match.group("pid"))
        fingerprint = _FINGERPRINT_RE.search(line)
        if fingerprint is not None:
            records_by_pid.setdefault(pid, []).append(
                (fingerprint.group("kind"), fingerprint.group("digest"))
            )
        bucket = _BUCKET_RE.search(line)
        if bucket is not None:
            _append_distinct(
                buckets_by_pid.setdefault(pid, []), int(bucket.group("tokens"))
            )
        if "GLM_CODE_FINGERPRINT:" in line:
            code_lines_by_pid.setdefault(pid, []).append(line)
        if "Initializing a V1 LLM engine" in line and "compilation_config=" in line:
            config_lines_by_pid.setdefault(pid, []).append(line)

    fingerprint_pids = [pid for pid, records in records_by_pid.items() if records]
    if len(fingerprint_pids) != 1:
        raise BenchmarkValidationError("compile-only fingerprint owner count drifted")
    pid = fingerprint_pids[0]
    code_lines = code_lines_by_pid.get(pid, [])
    expected_code = f"GLM_CODE_FINGERPRINT: git={ACCEPTED_SHORT_CODE_PIN} dirty=0"
    if not code_lines or any(expected_code not in line for line in code_lines):
        raise BenchmarkValidationError("compile-only code fingerprint drifted")
    config_lines = config_lines_by_pid.get(pid, [])
    effective_config_markers = (
        "(v0.1.dev1+ga30addc75)",
        "speculative_config=None",
        "trust_remote_code=True",
        "dtype=torch.bfloat16",
        "max_seq_len=8704",
        "load_format=runai_streamer",
        "tensor_parallel_size=32",
        "pipeline_parallel_size=1",
        "data_parallel_size=1",
        "decode_context_parallel_size=1",
        "served_model_name=gs://driftbench-dsv4-uc/models/GLM-5.2-FP8",
        "enable_prefix_caching=False",
        "'debug_dump_path': None",
    )
    if len(config_lines) != 1 or any(
        marker not in config_lines[0] for marker in effective_config_markers
    ):
        raise BenchmarkValidationError("compile-only engine configuration drifted")

    buckets = tuple(buckets_by_pid.get(pid, ()))
    if buckets != TOKEN_BUCKETS:
        raise BenchmarkValidationError("compile-only token-bucket order drifted")
    triples = _ordered_triples(records_by_pid[pid])
    if triples != ACCEPTED_FINGERPRINTS:
        raise BenchmarkValidationError(
            "compile-only executable class differs from DB485"
        )

    return {
        "artifact_kind": ARTIFACT_KIND,
        "claim_scope": CLAIM_SCOPE,
        "engine_pid": pid,
        "fingerprints": [
            {
                "executable": executable,
                "executable_including_data_segments": including_data,
                "host_transfer": host_transfer,
                "num_tokens": num_tokens,
            }
            for num_tokens, (executable, including_data, host_transfer) in zip(
                TOKEN_BUCKETS, triples, strict=True
            )
        ],
        "log_sha256": sha256(raw).hexdigest(),
        "runtime_configuration": ACCEPTED_RUNTIME_CONFIG,
        "runtime_configuration_sha256": ACCEPTED_RUNTIME_CONFIG_SHA256,
        "run_tag": run_tag,
        "source_code_pin": ACCEPTED_FULL_CODE_PIN,
        "status": "DB485_EXECUTABLE_CLASS_MATCH",
    }


def validate_hlo_owner_receipts(raw: bytes) -> dict[str, Any]:
    """Require one compile owner and seven nonowners across exact workers 0--7."""

    all_lines = raw.decode("utf-8", errors="strict").splitlines()
    lines = [line for line in all_lines if line.startswith("HLO_")]
    matches = [_OWNER_RECEIPT_RE.fullmatch(line) for line in lines]
    if len(lines) != 8 or any(match is None for match in matches):
        raise BenchmarkValidationError("HLO owner receipts are malformed or incomplete")
    records = [match.groupdict() for match in matches if match is not None]
    workers = [int(record["worker"]) for record in records]
    hosts = [record["host"] for record in records]
    owners = [record for record in records if record["kind"] == "HLO_OWNER"]
    if (
        sorted(workers) != list(range(8))
        or len(set(hosts)) != 8
        or len(owners) != 1
        or owners[0]["count"] != "7"
        or any(
            record["count"] is not None
            for record in records
            if record["kind"] == "HLO_NONOWNER"
        )
    ):
        raise BenchmarkValidationError("HLO owner/nonowner host contract drifted")
    return {
        "owner_host": owners[0]["host"],
        "owner_worker": int(owners[0]["worker"]),
        "status": "ONE_HLO_OWNER_SEVEN_NONOWNERS",
        "worker_count": 8,
    }


def validate_scheduled_hlo_files(paths: list[Path]) -> dict[str, Any]:
    """Seal one shape-bound scheduled HLO for each accepted token bucket."""

    if len(paths) != len(TOKEN_BUCKETS):
        raise BenchmarkValidationError(
            f"expected seven scheduled HLO files, found {len(paths)}"
        )
    files: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_shas: set[str] = set()
    module_ids: list[int] = []
    ordered_paths: list[tuple[int, int, str, Path]] = []
    for path in paths:
        filename = _HLO_FILENAME_RE.fullmatch(path.name)
        if filename is None:
            raise BenchmarkValidationError("scheduled HLO filename contract drifted")
        ordered_paths.append(
            (
                int(filename.group("tokens")),
                int(filename.group("module")),
                filename.group("raw"),
                path,
            )
        )
    ordered_paths.sort(key=lambda item: item[0])
    if tuple(item[0] for item in ordered_paths) != TOKEN_BUCKETS:
        raise BenchmarkValidationError("scheduled HLO token-bucket mapping drifted")
    for tokens, module_id, raw_name, path in ordered_paths:
        if path.name in seen_names:
            raise BenchmarkValidationError("scheduled HLO filename contract drifted")
        seen_names.add(path.name)
        compressed = path.read_bytes()
        try:
            raw = gzip.decompress(compressed)
        except (EOFError, OSError) as exc:
            raise BenchmarkValidationError(
                f"invalid scheduled HLO gzip: {path}"
            ) from exc
        text = raw.decode("utf-8", errors="strict")
        header = text.splitlines()[0] if text else ""
        if (
            not header.startswith("HloModule jit_step_fun_impl, is_scheduled=true")
            or "num_partitions=32" not in header
        ):
            raise BenchmarkValidationError(f"unscheduled or wrong HLO module: {path}")
        bucket_counts = {
            bucket: len(
                re.findall(
                    rf"bf16\[{bucket},6144\].*all-reduce\(.*"
                    r"VllmRowParallelLinear/shard_map/psum",
                    text,
                )
            )
            for bucket in TOKEN_BUCKETS
        }
        if bucket_counts[tokens] != 156 or any(
            count for bucket, count in bucket_counts.items() if bucket != tokens
        ):
            raise BenchmarkValidationError(
                f"scheduled HLO bucket shape contract drifted: {path}"
            )
        raw_sha = sha256(raw).hexdigest()
        if raw_sha in seen_shas:
            raise BenchmarkValidationError("duplicate scheduled HLO module bytes")
        seen_shas.add(raw_sha)
        module_ids.append(module_id)
        files.append(
            {
                "compressed_bytes": len(compressed),
                "compressed_sha256": sha256(compressed).hexdigest(),
                "name": path.name,
                "module_id": module_id,
                "num_tokens": tokens,
                "raw_name": raw_name,
                "raw_bytes": len(raw),
                "raw_sha256": raw_sha,
            }
        )
    if module_ids != sorted(module_ids) or len(module_ids) != len(set(module_ids)):
        raise BenchmarkValidationError("scheduled HLO module order drifted")
    return {
        "file_count": len(files),
        "files": files,
        "status": "SEVEN_SCHEDULED_HLO_MODULES_SEALED",
    }
