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

from ...optimized.errors import BenchmarkValidationError
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
_REPLICA_RECEIPT_RE = re.compile(
    r"^HLO_REPLICA "
    r"(?P<host>[A-Za-z0-9.-]+-w-(?P<worker>[0-7])) "
    r"count=(?P<count>\d+) "
    r"identity_sha256=(?P<identity_sha256>[0-9a-f]{64}) "
    r"audit_sha256=(?P<audit_sha256>[0-9a-f]{64})$"
)
_WORKER_HOST_RE = re.compile(r"^(?P<host>[A-Za-z0-9.-]+-w-(?P<worker>[0-7]))$")
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


def validate_worker_host_receipts(raw: bytes, *, marker: str) -> dict[int, str]:
    """Return the authenticated worker-to-Linux-hostname mapping for a fleet receipt."""

    if not re.fullmatch(r"[A-Z][A-Z0-9_]+", marker):
        raise BenchmarkValidationError("worker-host receipt marker is invalid")
    prefix = f"{marker} "
    lines = [
        line.removeprefix(prefix)
        for line in raw.decode("utf-8", errors="strict").splitlines()
        if line.startswith(prefix)
    ]
    matches = [_WORKER_HOST_RE.fullmatch(line) for line in lines]
    if len(lines) != 8 or any(match is None for match in matches):
        raise BenchmarkValidationError(
            "worker-host receipts are malformed or incomplete"
        )
    mapping = {
        int(match.group("worker")): match.group("host")
        for match in matches
        if match is not None
    }
    if len(mapping) != 8 or sorted(mapping) != list(range(8)):
        raise BenchmarkValidationError("worker-host receipt mapping drifted")
    return mapping


def validate_hlo_replica_evidence(
    raw: bytes,
    audit_dir: Path,
    expected_hosts: dict[int, str],
    *,
    require_common_identity: bool = True,
) -> dict[str, Any]:
    """Require eight host audits with one common raw-HLO identity."""

    expected_matches = {
        worker: _WORKER_HOST_RE.fullmatch(host)
        for worker, host in expected_hosts.items()
    }
    if (
        sorted(expected_hosts) != list(range(8))
        or len(set(expected_hosts.values())) != 8
        or any(
            match is None or int(match.group("worker")) != worker
            for worker, match in expected_matches.items()
        )
    ):
        raise BenchmarkValidationError("expected worker-host mapping drifted")
    all_lines = raw.decode("utf-8", errors="strict").splitlines()
    lines = [line for line in all_lines if line.startswith("HLO_")]
    matches = [_REPLICA_RECEIPT_RE.fullmatch(line) for line in lines]
    if len(lines) != 8 or any(match is None for match in matches):
        raise BenchmarkValidationError(
            "HLO replica receipts are malformed or incomplete"
        )
    records = [match.groupdict() for match in matches if match is not None]
    workers = [int(record["worker"]) for record in records]
    hosts = [record["host"] for record in records]
    if (
        sorted(workers) != list(range(8))
        or len(set(hosts)) != 8
        or any(record["count"] != "7" for record in records)
        or any(
            record["host"] != expected_hosts.get(int(record["worker"]))
            for record in records
        )
    ):
        raise BenchmarkValidationError("HLO replica host contract drifted")
    try:
        audit_paths = sorted(audit_dir.iterdir())
    except OSError as exc:
        raise BenchmarkValidationError(
            "HLO replica audit directory is unavailable"
        ) from exc
    if any(not path.is_file() or path.is_symlink() for path in audit_paths) or [
        path.name for path in audit_paths
    ] != [f"worker_{worker}.tsv" for worker in range(8)]:
        raise BenchmarkValidationError("HLO replica audit file set drifted")

    audits = []
    common_identities: set[bytes] = set()
    expected_columns = (
        "bucket\traw_path\traw_bytes\traw_sha256\tsealed_name\t"
        "compressed_bytes\tcompressed_sha256"
    )
    for record in sorted(records, key=lambda item: int(item["worker"])):
        worker = int(record["worker"])
        path = audit_dir / f"worker_{worker}.tsv"
        audit_bytes = path.read_bytes()
        if sha256(audit_bytes).hexdigest() != record["audit_sha256"]:
            raise BenchmarkValidationError("HLO replica audit receipt hash drifted")
        audit_lines = audit_bytes.decode("utf-8", errors="strict").splitlines()
        if (
            len(audit_lines) != 14
            or audit_lines[0] != "schema\thlo_replica_audit_v2"
            or audit_lines[1] != f"host\t{record['host']}"
            or audit_lines[2] != f"worker\t{worker}"
            or audit_lines[3] != f"identity_sha256\t{record['identity_sha256']}"
            or not re.fullmatch(r"raw_inventory_sha256\t[0-9a-f]{64}", audit_lines[4])
            or not re.fullmatch(r"bucket_map_sha256\t[0-9a-f]{64}", audit_lines[5])
            or audit_lines[6] != expected_columns
        ):
            raise BenchmarkValidationError("HLO replica audit header drifted")
        rows = []
        module_ids = []
        for line in audit_lines[7:]:
            fields = line.split("\t")
            if len(fields) != 7:
                raise BenchmarkValidationError("HLO replica audit row drifted")
            (
                bucket,
                raw_path,
                raw_bytes,
                raw_sha,
                sealed_name,
                compressed_bytes,
                compressed_sha,
            ) = fields
            filename = _HLO_FILENAME_RE.fullmatch(sealed_name)
            if (
                not bucket.isdigit()
                or not raw_bytes.isdigit()
                or int(raw_bytes) <= 0
                or not compressed_bytes.isdigit()
                or int(compressed_bytes) <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", raw_sha)
                or not re.fullmatch(r"[0-9a-f]{64}", compressed_sha)
                or not raw_path
                or raw_path.startswith("/")
                or ".." in Path(raw_path).parts
                or filename is None
                or int(filename.group("tokens")) != int(bucket)
                or filename.group("raw") != Path(raw_path).name
            ):
                raise BenchmarkValidationError("HLO replica audit row drifted")
            module_ids.append(int(filename.group("module")))
            rows.append(
                {
                    "compressed_bytes": int(compressed_bytes),
                    "compressed_sha256": compressed_sha,
                    "num_tokens": int(bucket),
                    "raw_bytes": int(raw_bytes),
                    "raw_path": raw_path,
                    "raw_sha256": raw_sha,
                    "sealed_name": sealed_name,
                }
            )
        if (
            tuple(row["num_tokens"] for row in rows) != TOKEN_BUCKETS
            or module_ids != sorted(module_ids)
            or len(module_ids) != len(set(module_ids))
        ):
            raise BenchmarkValidationError("HLO replica audit bucket order drifted")
        identity_bytes = "".join(
            f"{row['num_tokens']}\t{row['raw_bytes']}\t{row['raw_sha256']}\n"
            for row in rows
        ).encode()
        if sha256(identity_bytes).hexdigest() != record["identity_sha256"]:
            raise BenchmarkValidationError("HLO replica common identity hash drifted")
        common_identities.add(identity_bytes)
        audits.append(
            {
                "audit_bytes": len(audit_bytes),
                "audit_sha256": record["audit_sha256"],
                "bucket_map_sha256": audit_lines[5].split("\t", 1)[1],
                "host": record["host"],
                "raw_inventory_sha256": audit_lines[4].split("\t", 1)[1],
                "rows": rows,
                "worker": worker,
            }
        )
    if require_common_identity and len(common_identities) != 1:
        raise BenchmarkValidationError("HLO replica raw identities diverged")
    common_identity = (
        next(iter(common_identities)) if len(common_identities) == 1 else None
    )
    canonical = next(record for record in records if record["worker"] == "0")
    return {
        "audits": audits,
        "canonical_host": canonical["host"],
        "canonical_worker": 0,
        "common_identity": (
            [
                {
                    "num_tokens": int(line.split("\t")[0]),
                    "raw_bytes": int(line.split("\t")[1]),
                    "raw_sha256": line.split("\t")[2],
                }
                for line in common_identity.decode().splitlines()
            ]
            if common_identity is not None
            else None
        ),
        "common_identity_count": len(common_identities),
        "common_identity_sha256": (
            sha256(common_identity).hexdigest() if common_identity is not None else None
        ),
        "replica_count": 8,
        "status": (
            "EIGHT_MATCHING_RAW_HLO_REPLICAS"
            if len(common_identities) == 1
            else "EIGHT_AUDITED_DIVERGENT_RAW_HLO_REPLICAS"
        ),
        "worker_count": 8,
    }


def validate_hlo_replica_payload(
    hlo_dir: Path, audit: dict[str, Any]
) -> dict[str, Any]:
    """Bind one copied compact payload to its full per-host audit."""

    paths = [
        path
        for path in hlo_dir.iterdir()
        if path.is_file() and path.name.endswith(".txt.gz")
    ]
    report = validate_scheduled_hlo_files(paths)
    expected_names = {
        *(item["name"] for item in report["files"]),
        "bucket_hlo_map.tsv",
        "raw_hlo_inventory.txt",
        "replica_audit.tsv",
        "replica_identity.tsv",
    }
    entries = list(hlo_dir.iterdir())
    if (
        any(not path.is_file() or path.is_symlink() for path in entries)
        or {path.name for path in entries} != expected_names
    ):
        raise BenchmarkValidationError(
            "canonical HLO directory contains an unexpected entry"
        )
    actual_identity = [
        {
            "num_tokens": item["num_tokens"],
            "raw_bytes": item["raw_bytes"],
            "raw_sha256": item["raw_sha256"],
        }
        for item in report["files"]
    ]
    identity_bytes = "".join(
        f"{item['num_tokens']}\t{item['raw_bytes']}\t{item['raw_sha256']}\n"
        for item in actual_identity
    ).encode()
    audit_rows = audit["rows"]
    if (
        (hlo_dir / "replica_identity.tsv").read_bytes() != identity_bytes
        or sha256((hlo_dir / "replica_audit.tsv").read_bytes()).hexdigest()
        != audit["audit_sha256"]
        or sha256((hlo_dir / "raw_hlo_inventory.txt").read_bytes()).hexdigest()
        != audit["raw_inventory_sha256"]
        or sha256((hlo_dir / "bucket_hlo_map.tsv").read_bytes()).hexdigest()
        != audit["bucket_map_sha256"]
        or len(audit_rows) != len(report["files"])
        or any(
            row["num_tokens"] != item["num_tokens"]
            or row["raw_bytes"] != item["raw_bytes"]
            or row["raw_sha256"] != item["raw_sha256"]
            or row["sealed_name"] != item["name"]
            or row["compressed_bytes"] != item["compressed_bytes"]
            or row["compressed_sha256"] != item["compressed_sha256"]
            for row, item in zip(audit_rows, report["files"], strict=True)
        )
    ):
        raise BenchmarkValidationError("HLO payload differs from replica audit")
    report["replica_identity"] = {
        "records": actual_identity,
        "sha256": sha256(identity_bytes).hexdigest(),
    }
    return report


def validate_canonical_hlo_replica(
    hlo_dir: Path, replica_report: dict[str, Any]
) -> dict[str, Any]:
    """Bind worker 0's copied payload to the common eight-host identity."""

    report = validate_hlo_replica_payload(hlo_dir, replica_report["audits"][0])
    if (
        report["replica_identity"]["records"] != replica_report["common_identity"]
        or report["replica_identity"]["sha256"]
        != replica_report["common_identity_sha256"]
    ):
        raise BenchmarkValidationError(
            "canonical HLO payload differs from replica evidence"
        )
    return report


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
