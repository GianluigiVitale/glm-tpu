#!/usr/bin/env python3
"""Offline adjudication of the layer-1 RMS schedule bounded run's diagnostic archive.

The run ``gate_d_layer1_rms_schedule_20260902T061305905714981Z`` executed both
arms on TPU and failed closed at publication (stale publisher constant), so its
evidence lives only in the generation-bound ``diagnostic/`` prefix and the local
run directory.  This CPU-only adjudicator verifies, from bytes alone: every
ledger object (local bytes, size, SHA-256; remote generation, size, CRC32C via
``gcloud storage objects describe``), the terminal-last ledger receipt, the
absence of any success/result/DB claim (false-claim boundary), the runner's
claim scope, both arms' HLO byte identities, and rederives the two output rows
against the sealed DB548 and accepted references with the fixed publisher's
pure-Python rederivation.  It makes no Gate-D, decoder or performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

RUN_TAG = "gate_d_layer1_rms_schedule_20260902T061305905714981Z"
RUN_DIR = Path("/home/gianl/gate-d-runs") / RUN_TAG
REMOTE_PREFIX = (
    "gs://driftbench-dsv4-uc/results/greenfield/glm52/gate_d_layer1_rms_schedule/"
    + RUN_TAG
)
CODE_PIN = "839c8fb7cba0168cac5f63e64eb79df3f95fc518"
PUBLISHER = Path(__file__).resolve().parent / "publish_gate_d_layer1_rms_schedule.py"
EXPECTED = {
    "runner.json": "aa1970dcf0607ee988fed26705b2ba3c20526633adb5fb19985c894764cda281",
    "outputs.npz": "a8ef84f16bf80f5bddc6ba1002bbe6b02ed4db831e90eaa2f08d663a58a0f59b",
    "dependencies.json": "c355230ba70c2f649b11fa975414b6479287f70ed0a898af218fe2f57d55a078",
    "hlo/layer1_rms_schedule_control.optimized_hlo.txt": "3ca700512e54361439bbd73da703282cbe744c3b94bc2bc1b8b90ad612cc06a4",
    "hlo/layer1_rms_schedule_control.stablehlo.mlir": "c71ad5307e24fabe1cd639ad504cb0f62ce5c5ae649ff736704d3e5c02bf9b77",
    "hlo/layer1_rms_schedule_schedule.optimized_hlo.txt": "ac4e969ee4d7ef10fe039a495b32eb3e383d5f3840c39f827001c16ed142c421",
    "hlo/layer1_rms_schedule_schedule.stablehlo.mlir": "ec9f7988843bffa4f07812e9fc3eacbb7498ab98cfcab6ac6a17d9f4047af31f",
    "census_pre.txt": "e46be63c2f9e0eb3c6e1bf2d438f8d8fc463879cf37dbe4f8468bb7070d69e0f",
    "census_post.txt": "5e993974c39b5187d3c780fca48f64becfb7a2f4841688a0a77355acfd103762",
    "mirror.sha256": "20f371f5e8d2dc8eb5e66b365cd131fb03c66efef40b402cfa5524054cb13e98",
}
LEDGER_SHA256 = "0e5dded79181dc49ef6f271dbf49238ac422408cc3de5e0d90700f838ad9669c"
FORBIDDEN_REMOTE = ("NUMERICAL_RESULT", "SUCCESS", "summary.json", "evidence.json", "remote_objects.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_remote(name: str) -> dict:
    completed = subprocess.run(
        ["/snap/bin/gcloud", "storage", "objects", "describe", name, "--format=json"],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": "/snap/bin:/usr/bin:/bin", "HOME": "/home/gianl", "PYTHONWARNINGS": "ignore"},
        timeout=120,
    )
    return json.loads(completed.stdout)


def list_remote(prefix: str, *flags: str) -> list[str]:
    completed = subprocess.run(
        ["/snap/bin/gcloud", "storage", "ls", *flags, prefix + "/**"],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": "/snap/bin:/usr/bin:/bin", "HOME": "/home/gianl", "PYTHONWARNINGS": "ignore"},
        timeout=120,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip().startswith("gs://")]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-remote", action="store_true")
    args = parser.parse_args()
    result: dict = {"artifact_kind": "gate_d_layer1_rms_schedule_diagnostic_adjudication", "schema_version": 1, "run_tag": RUN_TAG, "code_pin": CODE_PIN, "remote_prefix": REMOTE_PREFIX}
    problems: list[str] = []
    ledger_raw = (RUN_DIR / "diagnostic_objects.json").read_bytes()
    if hashlib.sha256(ledger_raw).hexdigest() != LEDGER_SHA256:
        problems.append("ledger bytes drifted")
    ledger = json.loads(ledger_raw)
    receipt = json.loads((RUN_DIR / "diagnostic_upload_receipt.json").read_text())
    if receipt["terminal"]["sha256"] != LEDGER_SHA256 or receipt["remote"] != REMOTE_PREFIX + "/diagnostic/diagnostic_objects.json":
        problems.append("terminal receipt does not bind the ledger")
    objects = {item["path"]: item for item in ledger["objects"]}
    local_records = {}
    for path, record in sorted(objects.items()):
        local = RUN_DIR / path
        digest = sha256_file(local)
        local_records[path] = {"sha256": digest, "size": local.stat().st_size, "generation": record["generation"]}
        if digest != record["sha256"] or local.stat().st_size != record["size"]:
            problems.append(f"local bytes drift from ledger: {path}")
    for path, expected in EXPECTED.items():
        if local_records.get(path, {}).get("sha256") != expected:
            problems.append(f"expected identity drift: {path}")
    remote_checked = {}
    if not args.skip_remote:
        for path, record in sorted(objects.items()):
            info = describe_remote(f"{REMOTE_PREFIX}/diagnostic/{path}")
            remote_checked[path] = {"generation": str(info.get("generation")), "size": int(info.get("size", -1)), "crc32c": info.get("crc32c_hash")}
            if str(info.get("generation")) != record["generation"] or int(info.get("size", -1)) != record["size"] or info.get("crc32c_hash") != record["crc32c"]:
                problems.append(f"remote object drift: {path}")
        ledger_info = describe_remote(f"{REMOTE_PREFIX}/diagnostic/diagnostic_objects.json")
        remote_checked["diagnostic_objects.json"] = {"generation": str(ledger_info.get("generation")), "size": int(ledger_info.get("size", -1)), "crc32c": ledger_info.get("crc32c_hash")}
        if str(ledger_info.get("generation")) != receipt["terminal"]["generation"] or ledger_info.get("crc32c_hash") != receipt["terminal"]["crc32c"]:
            problems.append("remote ledger generation/crc drift from receipt")
        live = list_remote(REMOTE_PREFIX)
        versions = list_remote(REMOTE_PREFIX, "--all-versions")
        expected_live = {f"{REMOTE_PREFIX}/diagnostic/{path}" for path in objects} | {f"{REMOTE_PREFIX}/diagnostic/diagnostic_objects.json"}
        live_set = {line.split("#", 1)[0] for line in live}
        if live_set != expected_live:
            problems.append(f"live remote set differs from ledger: extra={sorted(live_set - expected_live)[:4]} missing={sorted(expected_live - live_set)[:4]}")
        if any(any(f"/{name}" in line for name in FORBIDDEN_REMOTE) for line in versions):
            problems.append("remote history carries a success/result/ledger claim object")
        if len({line.split("#", 1)[0] for line in versions}) != len(versions):
            problems.append("remote history carries noncurrent versions of diagnostic objects")
        result["remote_objects"] = remote_checked
        result["remote_live_count"] = len(live)
        result["remote_all_versions_count"] = len(versions)
    for name in ("NUMERICAL_RESULT", "SUCCESS", "summary.json", "evidence.json", "remote_objects.json", "terminal_upload_receipt.json"):
        if (RUN_DIR / name).exists():
            problems.append(f"local run directory carries a claim object: {name}")
    runner = json.loads((RUN_DIR / "runner.json").read_text())
    failure = json.loads((RUN_DIR / "failure_status.json").read_text())
    if failure != {"artifact_kind": "gate_d_layer1_rms_schedule_failure", "code_hash": CODE_PIN, "exit_status": 1, "gate_d_closed": False, "performance_claim": False, "root_cause_fix_proven": False, "run_tag": RUN_TAG, "terminal_payload_present": False}:
        problems.append("failure status record drifted")
    if runner.get("code_hash") != CODE_PIN or runner.get("status") != "SCHEDULE_ARM_EXACT" or runner.get("gate_d_closed") is not False or runner.get("performance_claim") is not False or runner.get("root_cause_fix_proven") is not False or runner.get("compiled_executable_invocation_count") != 2:
        problems.append("runner claim boundary drifted")
    for arm, suffixes in (("control", ("optimized_hlo", "stablehlo")), ("schedule", ("optimized_hlo", "stablehlo"))):
        for kind in suffixes:
            relative = f"hlo/layer1_rms_schedule_{arm}.{'optimized_hlo.txt' if kind == 'optimized_hlo' else 'stablehlo.mlir'}"
            identity = runner["arms"][arm]["hlo"][kind]
            if identity["sha256"] != local_records[relative]["sha256"] or identity["byte_count"] != local_records[relative]["size"] or identity["structure"].get("passed") is not True:
                problems.append(f"runner HLO identity drift: {relative}")
    dependencies_raw = (RUN_DIR / "dependencies.json").read_bytes()
    manifest = runner["compiler_dependency_manifest"]
    if manifest["sha256"] != hashlib.sha256(dependencies_raw).hexdigest() or manifest["byte_count"] != len(dependencies_raw):
        problems.append("dependency manifest identity drift")
    spec = importlib.util.spec_from_file_location("schedule_publisher", PUBLISHER)
    publisher = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(publisher)
    output_raw = (RUN_DIR / "outputs.npz").read_bytes()
    if runner["output_artifact"] != {"byte_count": len(output_raw), "filename": "outputs.npz", "sha256": hashlib.sha256(output_raw).hexdigest()}:
        problems.append("output artifact identity drift")
    try:
        exact = publisher._validate_output_npz(output_raw, runner, runner["numerical"])
    except Exception as error:  # noqa: BLE001
        exact = None
        problems.append(f"output rederivation refused: {error}")
    if exact is not True:
        problems.append("output rederivation did not confirm SCHEDULE_ARM_EXACT")
    if json.loads(dependencies_raw).get("environment") != publisher.EXPECTED_COMPILER_ENVIRONMENT:
        problems.append("dependency environment differs from the fixed publisher contract")
    for census in ("census_pre.txt", "census_post.txt"):
        text = (RUN_DIR / census).read_text()
        hosts = [line.split()[1] for line in text.splitlines() if line.startswith("CENSUS_OK ")]
        if len(hosts) != 8 or len(set(hosts)) != 8:
            problems.append(f"{census} is not 8/8")
    result.update({"expected_identities": EXPECTED, "local_records": local_records, "ledger_sha256": LEDGER_SHA256, "publisher_sha256": sha256_file(PUBLISHER), "runner_status": runner.get("status"), "runner_classification": runner.get("classification"), "numerical": {k: runner["numerical"][k] for k in ("harness_admissible", "schedule_arm_exact")}, "schedule_vs_db548_mismatch_count": runner["numerical"]["schedule_vs_db548"]["mismatch_count"], "problems": problems})
    result["classification"] = ("BOUNDED_TPU_LAYER1_RMS_SCHEDULE_ARM_EXACT;DIAGNOSTIC_ARCHIVE_ADJUDICATED;GENERATION_BOUND_LEDGER_REPLAYED;" if not problems else "DIAGNOSTIC_ARCHIVE_ADJUDICATION_REFUSED;") + "NO_SUCCESS_NO_DB_NO_PERFORMANCE_NO_GATE_D_CLAIM;DECODER_UNPROVEN;GATE_D_OPEN"
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["classification"])
    if problems:
        for problem in problems:
            print("PROBLEM", problem)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
