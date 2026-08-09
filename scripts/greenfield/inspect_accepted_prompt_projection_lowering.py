#!/usr/bin/env python3
"""Seal the accepted M2048 prompt-key projection's TPU lowering.

This diagnostic consumes one protected eight-host phase-profile capture and
the module-filtered XLA dump produced by the accepted legacy oracle.  It does
not import or execute the legacy model path.
"""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import parse_xplane  # noqa: E402  (repository-local reused parser)


TARGET_SOURCE = "glm_dsa_indexer.py:1122"
TARGET_OP = "jit(step_fun_impl)/dot_general:"
TARGET_HLO_OP = 'op_name="jit(step_fun_impl)/dot_general"'
TARGET_LOGICAL_ROWS = 2048
TARGET_PHYSICAL_ROWS = 64
TARGET_PHYSICAL_RESULT = "f32[64,128]"
TARGET_PROFILE_RESULT = TARGET_PHYSICAL_RESULT
TARGET_PARTITION_COUNT = 32
TARGET_INVOCATIONS_PER_CORE = 21


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


def _worker_path(path: Path, worker: int) -> bool:
    return f"w{worker}" in path.parts


def _link_profiles(trace_root: Path, output: Path) -> list[dict[str, Any]]:
    xplanes = sorted(trace_root.rglob("*.xplane.pb"))
    traces = sorted(trace_root.rglob("*.trace.json.gz"))
    if len(xplanes) != 8 or len(traces) != 8:
        raise ValueError(
            "accepted projection profile must contain exactly eight XPlanes "
            f"and eight JSON traces, got xplanes={len(xplanes)} traces={len(traces)}"
        )

    linked: list[dict[str, Any]] = []
    for worker in range(8):
        worker_xplanes = [path for path in xplanes if _worker_path(path, worker)]
        worker_traces = [path for path in traces if _worker_path(path, worker)]
        if len(worker_xplanes) != 1 or len(worker_traces) != 1:
            raise ValueError(
                f"worker {worker} profile coverage drifted: "
                f"xplanes={len(worker_xplanes)} traces={len(worker_traces)}"
            )
        destinations = (
            (worker_xplanes[0], output / f"trace.worker{worker}.xplane.pb"),
            (worker_traces[0], output / f"trace.worker{worker}.trace.json.gz"),
        )
        for source, destination in destinations:
            # The gathered source tree and compact result live under the same
            # append-only run directory.  Hard links preserve the profiler
            # bytes after raw-source reclamation without temporarily doubling
            # the large fleet XPlane footprint on worker 0.
            os.link(source, destination)
            source_stat = source.stat()
            destination_stat = destination.stat()
            if (
                source_stat.st_dev != destination_stat.st_dev
                or source_stat.st_ino != destination_stat.st_ino
            ):
                raise ValueError(f"profile hard-link identity mismatch: {source}")
            linked.append(_record(destination, output))
    return linked


def _projection_signature(stats: dict[str, Any]) -> bool:
    source = str(stats.get("source", ""))
    shape = str(stats.get("shape_with_layout", ""))
    return (
        source.endswith(TARGET_SOURCE)
        and stats.get("tf_op") == TARGET_OP
        and TARGET_PROFILE_RESULT in shape
        and stats.get("hlo_category") == "convolution fusion"
    )


def _inspect_xplane(path: Path) -> dict[str, Any]:
    xspace = parse_xplane.load_xspace(path)
    hosts = [value for value in xspace.hostnames if value]
    if len(hosts) != 1:
        raise ValueError(f"expected one XPlane hostname in {path}, got {hosts}")

    cores: list[dict[str, Any]] = []
    for plane in xspace.planes:
        if not parse_xplane.is_device_plane(plane):
            continue
        event_metadata = dict(plane.event_metadata.items())
        stat_metadata = {
            key: value.name for key, value in plane.stat_metadata.items()
        }
        module_line = parse_xplane.find_line(plane, "XLA Modules")
        ops_line = parse_xplane.find_line(plane, "XLA Ops")
        if module_line is None or ops_line is None:
            raise ValueError(f"missing XLA module/op line in {path} {plane.name}")

        step_windows: list[tuple[int, int]] = []
        for event in module_line.events:
            metadata = event_metadata.get(event.metadata_id)
            name = (
                (metadata.display_name or metadata.name)
                if metadata is not None
                else str(event.metadata_id)
            )
            if re.search(r"jit_step_fun_impl", name):
                step_windows.append(
                    (event.offset_ps, event.offset_ps + event.duration_ps)
                )
        if len(step_windows) != 1:
            raise ValueError(
                f"expected one profiled prefill module in {path} {plane.name}, "
                f"got {len(step_windows)}"
            )
        start, end = step_windows[0]

        target_ids: dict[int, dict[str, Any]] = {}
        for metadata_id, metadata in event_metadata.items():
            stats = parse_xplane.metadata_stats(metadata, stat_metadata)
            if _projection_signature(stats):
                target_ids[metadata_id] = {
                    "hlo_category": stats["hlo_category"],
                    "program_id": stats.get("program_id"),
                    "shape_with_layout": stats["shape_with_layout"],
                    "source": stats["source"],
                    "tf_op": stats["tf_op"],
                }

        events = []
        for event in ops_line.events:
            midpoint = event.offset_ps + event.duration_ps // 2
            if event.metadata_id in target_ids and start <= midpoint < end:
                events.append(target_ids[event.metadata_id])
        if len(events) != TARGET_INVOCATIONS_PER_CORE:
            raise ValueError(
                f"projection invocation count drifted in {path} {plane.name}: "
                f"{len(events)} != {TARGET_INVOCATIONS_PER_CORE}"
            )
        signatures = {
            json.dumps(event, sort_keys=True, separators=(",", ":"))
            for event in events
        }
        if len(signatures) != 1:
            raise ValueError(
                f"projection physical signature is inconsistent in {path} "
                f"{plane.name}: {len(signatures)} variants"
            )
        cores.append({
            "invocation_count": len(events),
            "plane": plane.name,
            "signature": events[0],
        })

    if len(cores) != 8:
        raise ValueError(f"expected eight TPU planes in {path}, got {len(cores)}")
    return {"cores": cores, "host": hosts[0], "path": str(path)}


def _hlo_section(
    text: str,
    heading: str,
    next_headings: tuple[str, ...],
) -> str:
    match = re.search(rf"^{re.escape(heading)}$", text, re.MULTILINE)
    if match is None:
        return ""
    start = match.end()
    ends = []
    for next_heading in next_headings:
        next_match = re.search(
            rf"^{re.escape(next_heading)}$",
            text[start:],
            re.MULTILINE,
        )
        if next_match is not None:
            ends.append(start + next_match.start())
    return text[start:min(ends) if ends else len(text)]


def _parse_hlo_stack_frames(text: str) -> dict[int, tuple[str, int]]:
    # FileNames and FunctionNames use the same `<id> "value"` spelling.
    # Parse the named sections rather than allowing later function-name IDs to
    # overwrite file paths with the same integer ID.
    file_names_text = _hlo_section(
        text,
        "FileNames",
        ("FunctionNames", "FileLocations"),
    )
    locations_text = _hlo_section(text, "FileLocations", ("StackFrames",))
    frames_text = _hlo_section(text, "StackFrames", ())
    file_names = {
        int(match.group(1)): match.group(2)
        for match in re.finditer(
            r'^([0-9]+) "([^"]+)"$', file_names_text, re.MULTILINE
        )
    }
    locations = {
        int(match.group(1)): (int(match.group(2)), int(match.group(3)))
        for match in re.finditer(
            r"^([0-9]+) \{file_name_id=([0-9]+) function_name_id=[0-9]+ "
            r"line=([0-9]+) ",
            locations_text,
            re.MULTILINE,
        )
    }
    frames = {
        int(match.group(1)): (int(match.group(2)), int(match.group(3)))
        for match in re.finditer(
            r"^([0-9]+) \{file_location_id=([0-9]+) parent_frame_id=([0-9]+)\}$",
            frames_text,
            re.MULTILINE,
        )
    }

    resolved: dict[int, tuple[str, int]] = {}
    for frame_id in frames:
        seen: set[int] = set()
        current = frame_id
        while current not in seen and current in frames:
            seen.add(current)
            location_id, parent_id = frames[current]
            if location_id in locations:
                file_id, line = locations[location_id]
                filename = file_names.get(file_id, "")
                if filename.endswith("glm_dsa_indexer.py"):
                    resolved[frame_id] = (filename, line)
                    break
            if parent_id == current:
                break
            current = parent_id
    return resolved


_ASSIGNMENT = re.compile(
    r"^\s*(%[^ ]+) = ((?:bf16|f32)\[[^\]]+\]\{[^}]+\}) ", re.MULTILINE
)
_CONVOLUTION = re.compile(
    r"^\s*(%[^ ]+) = (f32\[64,128\]\{[^}]+\}) convolution\("
    r"(%[^,]+), (%[^)]+)\).*stack_frame_id=([0-9]+)",
)


def _inspect_hlo_text(text: str) -> dict[str, Any] | None:
    if not text.startswith("HloModule jit_step_fun_impl"):
        return None
    if "is_scheduled=true" not in text or TARGET_PHYSICAL_RESULT not in text:
        return None
    partition_match = re.search(r"\bnum_partitions=([0-9]+)\b", text[:200_000])
    if partition_match is None:
        raise ValueError("optimized HLO partition count is unavailable")
    partition_count = int(partition_match.group(1))
    if partition_count != TARGET_PARTITION_COUNT:
        raise ValueError(
            "optimized HLO partition count drifted: "
            f"{partition_count} != {TARGET_PARTITION_COUNT}"
        )
    frame_sources = _parse_hlo_stack_frames(text)
    assignments = {match.group(1): match.group(2) for match in _ASSIGNMENT.finditer(text)}

    convolutions: list[dict[str, Any]] = []
    for line in text.splitlines():
        if " convolution(" not in line or TARGET_HLO_OP not in line:
            continue
        match = _CONVOLUTION.match(line)
        if match is None:
            continue
        frame_id = int(match.group(5))
        source = frame_sources.get(frame_id)
        if source is None or source[1] != 1122:
            continue
        lhs, rhs = match.group(3), match.group(4)
        convolutions.append({
            "lhs_producer": lhs,
            "lhs_shape": assignments.get(lhs),
            "result": match.group(1),
            "result_shape": match.group(2),
            "rhs_producer": rhs,
            "rhs_shape": assignments.get(rhs),
            "source": f"{source[0]}:{source[1]}",
            "stack_frame_id": frame_id,
        })
    if not convolutions:
        return None
    if len(convolutions) != TARGET_INVOCATIONS_PER_CORE:
        raise ValueError(
            "optimized HLO projection convolution count drifted: "
            f"{len(convolutions)} != {TARGET_INVOCATIONS_PER_CORE}"
        )
    if any(value["lhs_shape"] is None or value["rhs_shape"] is None
           for value in convolutions):
        raise ValueError("optimized HLO projection operand layout is unresolved")

    fusions: list[dict[str, Any]] = []
    for line in text.splitlines():
        if (
            " fusion(" not in line
            or TARGET_HLO_OP not in line
            or "convolution_algorithm_config" not in line
            or TARGET_PHYSICAL_RESULT not in line
        ):
            continue
        frame_match = re.search(r"stack_frame_id=([0-9]+)", line)
        if frame_match is None:
            continue
        frame_id = int(frame_match.group(1))
        source = frame_sources.get(frame_id)
        if source is None or source[1] != 1122:
            continue
        backend_text = line.split("backend_config=", 1)[1]
        backend = json.loads(backend_text)
        shape_match = re.match(r"^\s*%[^ ]+ = (.+?) fusion\(", line)
        if shape_match is None:
            raise ValueError("optimized HLO projection fusion shape is unavailable")
        fusions.append({
            "backend_config": backend,
            "output_shape": shape_match.group(1),
            "source": f"{source[0]}:{source[1]}",
            "stack_frame_id": frame_id,
        })
    if len(fusions) != TARGET_INVOCATIONS_PER_CORE:
        raise ValueError(
            f"optimized HLO projection fusion count drifted: {len(fusions)} "
            f"!= {TARGET_INVOCATIONS_PER_CORE}"
        )

    physical_variants = {
        json.dumps({
            "lhs_shape": value["lhs_shape"],
            "result_shape": value["result_shape"],
            "rhs_shape": value["rhs_shape"],
        }, sort_keys=True, separators=(",", ":"))
        for value in convolutions
    }
    fusion_variants = {
        json.dumps({
            "backend_config": value["backend_config"],
            "output_shape": value["output_shape"],
        }, sort_keys=True, separators=(",", ":"))
        for value in fusions
    }
    if len(physical_variants) != 1 or len(fusion_variants) != 1:
        raise ValueError(
            "accepted projection lowering is not uniform across 21 full-indexer layers: "
            f"physical={len(physical_variants)} fusion={len(fusion_variants)}"
        )
    physical = json.loads(next(iter(physical_variants)))
    fusion = json.loads(next(iter(fusion_variants)))
    emitter = fusion["backend_config"].get(
        "convolution_algorithm_config", {}
    ).get("emitter")
    if not emitter:
        raise ValueError("accepted projection convolution emitter is unavailable")
    return {
        "convolution_count": len(convolutions),
        "convolution": physical,
        "emitter": emitter,
        "fusion_output_shape": fusion["output_shape"],
        "megacore_config": fusion["backend_config"].get("megacore_config", {}),
        "partition_count": partition_count,
        "source": convolutions[0]["source"],
        "window_config": fusion["backend_config"].get("window_config", {}),
    }


def _copy_and_inspect_hlo(
    trace_root: Path,
    output: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[tuple[Path, str, dict[str, Any]]] = []
    # after_codegen is the final physical TPU program.  Earlier optimized HLO
    # still exposes the global M=2048 operation and cannot prove the per-chip
    # convolution emitter/window association that caused the numerical drift.
    for path in sorted(trace_root.rglob("*after_codegen.txt")):
        text = path.read_text(errors="replace")
        result = _inspect_hlo_text(text)
        if result is not None:
            candidates.append((path, text, result))
    if not candidates:
        raise ValueError(
            "no scheduled current-pin physical-M64 jit_step_fun_impl HLO dump found"
        )

    variants = {
        json.dumps(result, sort_keys=True, separators=(",", ":"))
        for _, _, result in candidates
    }
    if len(variants) != 1:
        raise ValueError(
            "accepted physical-M64 HLO differs across dump owners: "
            f"{len(variants)}"
        )

    records: list[dict[str, Any]] = []
    for index, (source, text, _) in enumerate(candidates):
        destination = output / (
            f"jit_step_fun_impl.m64.owner{index}.after_codegen_hlo.txt.gz"
        )
        with destination.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
                stream.write(text.encode())
        records.append(_record(destination, output))
    return records, json.loads(next(iter(variants)))


def inspect_capture(
    *,
    trace_root: Path,
    output: Path,
    code_hash: str,
    source_capture_code_hash: str | None = None,
    legacy_code_hash: str,
    run_tag: str,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"append-only output exists: {output}")
    output.mkdir(parents=True)
    capture_code_hash = source_capture_code_hash or code_hash

    profile_records = _link_profiles(trace_root, output)
    xplane_results = [
        _inspect_xplane(output / f"trace.worker{worker}.xplane.pb")
        for worker in range(8)
    ]
    hosts = [result["host"] for result in xplane_results]
    if len(set(hosts)) != 8:
        raise ValueError(f"XPlane host identity is not fleet-unique: {hosts}")
    core_signatures = [
        core["signature"]
        for result in xplane_results
        for core in result["cores"]
    ]
    variants = {
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        for value in core_signatures
    }
    if len(core_signatures) != 64 or len(variants) != 1:
        raise ValueError(
            f"fleet projection signature drifted: cores={len(core_signatures)} "
            f"variants={len(variants)}"
        )

    hlo_records, hlo = _copy_and_inspect_hlo(trace_root, output)
    summary: dict[str, Any] = {
        "artifact_kind": "accepted_prompt_projection_lowering_v2",
        "code_hash": code_hash,
        "diagnostic_only": True,
        "hosts": sorted(hosts),
        "hlo": hlo,
        "legacy_code_hash": legacy_code_hash,
        "profile": {
            "core_count": 64,
            "file_count": 8,
            "invocations_per_core": TARGET_INVOCATIONS_PER_CORE,
            "signature": json.loads(next(iter(variants))),
            "steps_per_core": 1,
        },
        "shape_association": {
            "logical_row_count": TARGET_LOGICAL_ROWS,
            "physical_row_count": TARGET_PHYSICAL_ROWS,
            "profile_physical_result": TARGET_PROFILE_RESULT,
            "physical_hlo_result": TARGET_PHYSICAL_RESULT,
            "physical_row_shards": TARGET_PARTITION_COUNT,
        },
        "run_tag": run_tag,
        "source_capture_code_hash": capture_code_hash,
        "status": "SUCCESS",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    records = sorted(
        [*profile_records, *hlo_records, _record(output / "summary.json", output)],
        key=lambda value: value["path"],
    )
    manifest_payload = {
        "artifact_kind": summary["artifact_kind"],
        "code_hash": code_hash,
        "files": records,
        "legacy_code_hash": legacy_code_hash,
        "run_tag": run_tag,
        "source_capture_code_hash": capture_code_hash,
    }
    manifest_sha256 = sha256(
        json.dumps(
            manifest_payload, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    manifest = {**manifest_payload, "manifest_sha256": manifest_sha256}
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return {**summary, "manifest_sha256": manifest_sha256}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--source-capture-code-hash")
    parser.add_argument("--legacy-code-hash", required=True)
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args()
    result = inspect_capture(
        trace_root=args.trace_root,
        output=args.output,
        code_hash=args.expected_code_hash,
        source_capture_code_hash=args.source_capture_code_hash,
        legacy_code_hash=args.legacy_code_hash,
        run_tag=args.run_tag,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
