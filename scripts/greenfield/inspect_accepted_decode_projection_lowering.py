#!/usr/bin/env python3
"""Seal the accepted 32-row decode projection collective lowering.

The input is a compact, run-owned copy of the final TPU ``after_codegen``
module produced by the accepted legacy oracle.  This diagnostic never imports
or executes the legacy model path and makes no performance claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any


TARGET_OP = 'op_name="jit(step_fun_impl)/VllmRowParallelLinear/shard_map/psum"'
TARGET_RESULT = "bf16[32,6144]"
TARGET_PARTITIONS = 32
TARGET_GROUP = tuple(range(TARGET_PARTITIONS))
TARGET_COUNTS = {"attention": 78, "dense_mlp": 3, "moe_tuple": 75}


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "byte_count": path.stat().st_size,
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
    }


def _section(text: str, heading: str, next_headings: tuple[str, ...]) -> str:
    match = re.search(rf"^{re.escape(heading)}$", text, re.MULTILINE)
    if match is None:
        return ""
    start = match.end()
    ends = []
    for next_heading in next_headings:
        next_match = re.search(
            rf"^{re.escape(next_heading)}$", text[start:], re.MULTILINE
        )
        if next_match is not None:
            ends.append(start + next_match.start())
    return text[start : min(ends) if ends else len(text)]


def _stack_chains(text: str) -> dict[int, tuple[dict[str, Any], ...]]:
    files = {
        int(identifier): value
        for identifier, value in re.findall(
            r'^([0-9]+) "([^"]+)"$',
            _section(text, "FileNames", ("FunctionNames", "FileLocations")),
            re.MULTILINE,
        )
    }
    functions = {
        int(identifier): value
        for identifier, value in re.findall(
            r'^([0-9]+) "([^"]+)"$',
            _section(text, "FunctionNames", ("FileLocations",)),
            re.MULTILINE,
        )
    }
    locations = {
        int(identifier): (int(file_id), int(function_id), int(line))
        for identifier, file_id, function_id, line in re.findall(
            r"^([0-9]+) \{file_name_id=([0-9]+) function_name_id=([0-9]+) "
            r"line=([0-9]+) ",
            _section(text, "FileLocations", ("StackFrames",)),
            re.MULTILINE,
        )
    }
    frames = {
        int(identifier): (int(location_id), int(parent_id))
        for identifier, location_id, parent_id in re.findall(
            r"^([0-9]+) \{file_location_id=([0-9]+) "
            r"parent_frame_id=([0-9]+)\}$",
            _section(text, "StackFrames", ()),
            re.MULTILINE,
        )
    }

    result: dict[int, tuple[dict[str, Any], ...]] = {}
    for frame_id in frames:
        chain: list[dict[str, Any]] = []
        current = frame_id
        seen: set[int] = set()
        while current in frames and current not in seen:
            seen.add(current)
            location_id, parent_id = frames[current]
            if location_id in locations:
                file_id, function_id, line = locations[location_id]
                chain.append(
                    {
                        "file": files.get(file_id, ""),
                        "function": functions.get(function_id, ""),
                        "line": line,
                    }
                )
            if parent_id == current:
                break
            current = parent_id
        result[frame_id] = tuple(chain)
    return result


def _replica_group(line: str) -> tuple[int, ...]:
    match = re.search(r"replica_groups=\{\{([^}]+)\}\}", line)
    if match is None:
        raise ValueError("decode projection replica group is unavailable")
    return tuple(int(value) for value in match.group(1).split(","))


def _backend_config(line: str) -> dict[str, Any]:
    if "backend_config=" not in line:
        raise ValueError("decode projection backend config is unavailable")
    value = json.loads(line.split("backend_config=", 1)[1])
    if not isinstance(value, dict):
        raise ValueError("decode projection backend config is not an object")
    return value


def _bf16_add_reducers(lines: list[str]) -> set[str]:
    reducers: set[str] = set()
    current: str | None = None
    bf16_scalar_parameters: set[int] = set()
    has_bf16_root_add = False
    for line in lines:
        header = re.match(
            r"^(?!ENTRY\b)(%?[^\s(]+)(?:\s+\([^)]*\)\s+->\s+\S+)?\s+\{$",
            line,
        )
        if header is not None:
            current = header.group(1)
            bf16_scalar_parameters = set()
            has_bf16_root_add = False
            continue
        if current is None:
            continue
        parameter = re.search(
            r"^\s*%?[^ ]+ = bf16\[\][^ ]* parameter\(([01])\)", line
        )
        if parameter is not None:
            bf16_scalar_parameters.add(int(parameter.group(1)))
        if re.search(r"^\s*ROOT %?[^ ]+ = bf16\[\][^ ]* add\(", line):
            has_bf16_root_add = True
        if line == "}":
            if bf16_scalar_parameters == {0, 1} and has_bf16_root_add:
                reducers.add(current)
            current = None
    return reducers


def _classify(result_shape: str, chain: tuple[dict[str, Any], ...]) -> str:
    components = result_shape.count(TARGET_RESULT)
    if components == 2:
        return "moe_tuple"
    if components != 1:
        raise ValueError(f"decode projection result shape drifted: {result_shape}")
    if any(
        str(frame["file"]).endswith("deepseek_v2.py")
        and frame["function"] == "DeepseekV2DecoderLayer.forward"
        and frame["line"] == 1218
        for frame in chain
    ):
        return "dense_mlp"
    return "attention"


def _inspect_hlo_text(text: str) -> dict[str, Any] | None:
    if not text.startswith("HloModule jit_step_fun_impl"):
        return None
    if "is_scheduled=true" not in text or TARGET_OP not in text:
        return None
    partition_match = re.search(r"\bnum_partitions=([0-9]+)\b", text[:200_000])
    if partition_match is None:
        raise ValueError("decode HLO partition count is unavailable")
    partitions = int(partition_match.group(1))
    if partitions != TARGET_PARTITIONS:
        raise ValueError(
            f"decode HLO partition count drifted: {partitions} != {TARGET_PARTITIONS}"
        )

    chains = _stack_chains(text)
    lines = text.splitlines()
    bf16_add_reducers = _bf16_add_reducers(lines)
    records: list[dict[str, Any]] = []
    for line in lines:
        if " all-reduce(" not in line or TARGET_OP not in line:
            continue
        shape_match = re.match(r"^\s*%?[^ ]+ = (.+?) all-reduce\(", line)
        frame_match = re.search(r"stack_frame_id=([0-9]+)", line)
        reducer_match = re.search(r"to_apply=(%?[^,\s]+)", line)
        if shape_match is None or frame_match is None or reducer_match is None:
            raise ValueError("decode projection all-reduce syntax is unresolved")
        result_shape = shape_match.group(1)
        if TARGET_RESULT not in result_shape:
            raise ValueError(
                f"decode projection target-op result shape drifted: {result_shape}"
            )
        frame_id = int(frame_match.group(1))
        chain = chains.get(frame_id, ())
        if not chain or not (
            str(chain[0]["file"]).endswith("layers/common/linear.py")
            and chain[0]["function"] == "sharded_quantized_matmul.<locals>.wrapper"
            and chain[0]["line"] == 234
        ):
            raise ValueError(
                f"decode projection source chain drifted for frame {frame_id}: {chain}"
            )
        if _replica_group(line) != TARGET_GROUP:
            raise ValueError("decode projection escaped the exact 32-rank model group")
        if "use_global_device_ids=true" not in line:
            raise ValueError("decode projection does not use global device ids")
        reducer = reducer_match.group(1)
        if reducer not in bf16_add_reducers:
            raise ValueError(f"decode projection reducer is not BF16 add: {reducer}")
        backend = _backend_config(line)
        algorithm = backend.get("collective_algorithm_config")
        if not isinstance(algorithm, dict) or not all(
            algorithm.get(key) for key in ("emitter", "strategy", "debug")
        ):
            raise ValueError("decode projection collective algorithm is incomplete")
        records.append(
            {
                "backend_config": backend,
                "category": _classify(result_shape, chain),
                "result_shape": result_shape,
                "source_chain": chain,
            }
        )

    counts = Counter(record["category"] for record in records)
    if dict(counts) != TARGET_COUNTS:
        raise ValueError(
            f"decode projection collective counts drifted: {dict(counts)} "
            f"!= {TARGET_COUNTS}"
        )
    algorithms = {
        json.dumps(
            record["backend_config"]["collective_algorithm_config"],
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in records
    }
    if len(algorithms) != 1:
        raise ValueError(
            f"decode projection collective algorithm is nonuniform: {len(algorithms)}"
        )
    category_variants: dict[str, list[dict[str, Any]]] = {}
    for category in TARGET_COUNTS:
        variants = {
            json.dumps(
                {
                    "backend_config": record["backend_config"],
                    "result_shape": record["result_shape"],
                    "source_chain": record["source_chain"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            for record in records
            if record["category"] == category
        }
        category_variants[category] = [json.loads(value) for value in sorted(variants)]
    return {
        "category_counts": dict(counts),
        "category_variants": category_variants,
        "collective_algorithm_config": json.loads(next(iter(algorithms))),
        "collective_count": len(records),
        "compile_bucket_rows": 32,
        "partition_count": partitions,
        "replica_group": list(TARGET_GROUP),
        "reduction_dtype": "bf16",
        "result_width": 6144,
        "use_global_device_ids": True,
    }


def inspect_capture(
    *,
    source_dump_dir: Path,
    output: Path,
    code_hash: str,
    legacy_code_hash: str,
    run_tag: str,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"append-only output exists: {output}")
    candidates = sorted(
        source_dump_dir.rglob("accepted_decode.after_codegen.txt.gz")
    )
    if not 1 <= len(candidates) <= 8:
        raise ValueError(
            "accepted decode HLO requires 1..8 compile-owner files, "
            f"found {len(candidates)}"
        )

    inspections: list[dict[str, Any]] = []
    raw_hashes: list[str] = []
    compressed_hashes: list[str] = []
    owners: list[int] = []
    for path in candidates:
        worker_parts = [part for part in path.parts if re.fullmatch(r"w[0-7]", part)]
        if len(worker_parts) != 1:
            raise ValueError(f"decode HLO worker identity is unresolved: {path}")
        owners.append(int(worker_parts[0][1:]))
        raw = gzip.open(path, "rb").read()
        raw_hashes.append(sha256(raw).hexdigest())
        compressed_hashes.append(_sha256_file(path))
        inspected = _inspect_hlo_text(raw.decode(errors="replace"))
        if inspected is None:
            raise ValueError(f"source is not the scheduled decode HLO: {path}")
        inspections.append(inspected)
    if len(set(owners)) != len(owners):
        raise ValueError(f"duplicate decode HLO compile owners: {owners}")
    if len(set(raw_hashes)) != 1 or len(set(compressed_hashes)) != 1:
        raise ValueError("accepted decode HLO bytes differ across compile owners")
    variants = {
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        for value in inspections
    }
    if len(variants) != 1:
        raise ValueError("accepted decode HLO inspection differs across compile owners")

    output.mkdir(parents=True)
    sealed_hlo = output / "jit_step_fun_impl.m32.after_codegen_hlo.txt.gz"
    os.link(candidates[0], sealed_hlo)
    if sealed_hlo.stat().st_ino != candidates[0].stat().st_ino:
        raise ValueError("decode HLO hard-link identity mismatch")
    lowering = inspections[0]
    summary: dict[str, Any] = {
        "artifact_kind": "accepted_decode_projection_lowering_v1",
        "code_hash": code_hash,
        "compile_owner_count": len(owners),
        "compile_owner_workers": sorted(owners),
        "diagnostic_only": True,
        "hlo": lowering,
        "hlo_raw_sha256": raw_hashes[0],
        "legacy_code_hash": legacy_code_hash,
        "performance_claim": False,
        "protected_request_sequences": 1,
        "run_tag": run_tag,
        "status": "SUCCESS",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    files = sorted(
        [_record(sealed_hlo, output), _record(output / "summary.json", output)],
        key=lambda value: value["path"],
    )
    manifest_payload = {
        "artifact_kind": summary["artifact_kind"],
        "code_hash": code_hash,
        "files": files,
        "legacy_code_hash": legacy_code_hash,
        "run_tag": run_tag,
    }
    manifest_sha256 = sha256(
        json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(
            {**manifest_payload, "manifest_sha256": manifest_sha256},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {**summary, "manifest_sha256": manifest_sha256}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dump-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--legacy-code-hash", required=True)
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args()
    result = inspect_capture(
        source_dump_dir=args.source_dump_dir,
        output=args.output,
        code_hash=args.expected_code_hash,
        legacy_code_hash=args.legacy_code_hash,
        run_tag=args.run_tag,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
