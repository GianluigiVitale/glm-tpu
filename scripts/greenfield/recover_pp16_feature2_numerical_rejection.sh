#!/usr/bin/env bash
# CPU-only, append-only sealer for the rejected protected PP16 feature2 capture.
set -euo pipefail

readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly APPROVED_LOCATION=US-CENTRAL2
readonly SOURCE_TAG=greenfield_pp16_feature2_prefill_numerical_20260829T001056567299421Z
readonly SOURCE_RUN_DIR=/home/gianl/glm-run/$SOURCE_TAG
readonly SOURCE_REMOTE=$APPROVED_BUCKET/results/$SOURCE_TAG
readonly RUN_PIN=363a52b7c8a4201ffbbb899352b7159c26a95b80
readonly ORIGINAL_LEDGER_SHA=d857d2e98cf18eb505d447acd52acf062106af75e83ac0918fa64c13105e96c2
readonly CAPTURE_SHA=be3dda446cfbde547af49dd5ea371af690b553bcc8414907add1b0a2503b9d0d
readonly POST_CENSUS_SHA=cd1f25488e876acf475068131c5eb7f1ca25dabea158c553de86483aa8a40798
readonly FEATURE2_GRAPH_SHA=ab5be45aecf3b0b5d87ad76af8076bc9351823529a08c0eadb414b072b31cb2d
readonly EXPECTED_DB_MAX_RUN=564
readonly TOKEN_ORACLE_DIR=/home/gianl/gcs-models/oracles/greenfield/glm52/short_context/8k/greenfield_short_context_oracle_8k_20260807T172307269147351Z/oracle
readonly DSA_ORACLE_DIR=/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/8k/greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z/oracle
readonly LAYER1_INTERNAL_REFERENCE=/home/gianl/gcs-models/oracles/greenfield/glm52/dsa_internals/8k/layer1/greenfield_layer1_dsa_internal_comparison_20260808T115135394251231Z/internals.npz
readonly DB529_INTERNAL_DIR=/home/gianl/glm-run/greenfield_layer0_dsa_scorer_association_20260810T164030202890642Z/inputs/internal
readonly DB550_BOUNDARY=/home/gianl/gcs-models/results/greenfield_layer0_dense_partial_capture_20260813T200736889447458Z/dense_partial_capture.npz

[[ ${GLM_GREENFIELD_PP16_FEATURE2_RECOVER:-0} == 1 ]] || {
  echo "PP16 feature2 numerical rejection recovery is default-off" >&2
  exit 2
}
RECOVERY_MODE=${GLM_GREENFIELD_PP16_FEATURE2_RECOVERY_MODE:-off}
readonly RECOVERY_MODE
[[ $RECOVERY_MODE == validate_only || $RECOVERY_MODE == seal_rejected ]] || {
  echo "set recovery mode to validate_only or seal_rejected" >&2
  exit 2
}
for command_name in flock gcloud git sha256sum; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "required command is unavailable: $command_name" >&2
    exit 2
  }
done

RECOVERY_PIN=$(git -C "$WORKTREE" rev-parse HEAD)
RECOVERY_TAG=${GLM_GREENFIELD_PP16_FEATURE2_RECOVERY_TAG:-greenfield_pp16_feature2_numerical_recovery_$(date -u +%Y%m%dT%H%M%S%NZ)}
RECOVERY_DIR=/home/gianl/glm-run/$RECOVERY_TAG
RECOVERY_REMOTE=$APPROVED_BUCKET/results/$RECOVERY_TAG
readonly RECOVERY_PIN RECOVERY_TAG RECOVERY_DIR RECOVERY_REMOTE

[[ $RECOVERY_TAG =~ ^greenfield_pp16_feature2_numerical_recovery_[0-9]{8}T[0-9]{15}Z$ ]]
[[ $(git -C "$WORKTREE" rev-parse --show-toplevel) == "$WORKTREE" ]]
[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]]
[[ -z $(git -C "$WORKTREE" status --porcelain) ]]
[[ $(git -C "$WORKTREE" ls-remote origin "refs/heads/$BRANCH" | awk '{print $1}') == "$RECOVERY_PIN" ]]
[[ $(gcloud storage buckets describe "$APPROVED_BUCKET" --format='value(location)') == "$APPROVED_LOCATION" ]]
[[ -d $SOURCE_RUN_DIR && -r $RESULTS_DB && ! -e $RECOVERY_DIR ]]
[[ ! -e $SOURCE_RUN_DIR/NUMERICAL_EXACT && ! -e $SOURCE_RUN_DIR/NUMERICAL_REJECTED ]]

# Nonblocking acquisition proves both protected locks were free after the runner
# exited. Retain them while authenticating and publishing the recovery capsule.
exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  echo "another protected TPU workflow holds the global lease" >&2
  exit 1
}
exec 8>/home/gianl/.glm-tpu-rsync.lock
flock -n 8 || {
  echo "the same-region repository mirror is active" >&2
  exit 1
}
if pgrep -af '[e]xecute_pp16_feature2_prefill[.]py' >/dev/null; then
  echo "the original feature2 runner process is still alive" >&2
  exit 1
fi

mkdir -p "$RECOVERY_DIR/upload_receipts"
say() {
  echo "[pp16-feature2-recovery $(date -u +%H:%M:%S)] $*" |
    tee -a "$RECOVERY_DIR/orchestrator.log"
}

remote_prefix_is_vacant() {
  local output="$RECOVERY_DIR/remote_vacancy.stdout"
  local error="$RECOVERY_DIR/remote_vacancy.stderr" status
  if gcloud storage ls "$RECOVERY_REMOTE/**" >"$output" 2>"$error"; then
    [[ ! -s $output && ! -s $error ]]
    return
  else
    status=$?
  fi
  [[ $status -eq 1 && ! -s $output ]] &&
    [[ $(grep -c '^ERROR:' "$error") -eq 1 ]] &&
    [[ $(tail -n 1 "$error") == \
      "ERROR: (gcloud.storage.ls) One or more URLs matched no objects." ]]
}

remote_prefix_is_vacant || {
  say "ABORT: distinct recovery prefix is not vacant"
  exit 2
}
printf '%s\n' \
  "recovery_remote=$RECOVERY_REMOTE" \
  "vacancy_checked_before_local_evidence=true" \
  >"$RECOVERY_DIR/remote_vacancy.txt"

[[ $(sha256sum "$SOURCE_RUN_DIR/diagnostic.evidence.sha256" | awk '{print $1}') == "$ORIGINAL_LEDGER_SHA" ]]
(cd "$SOURCE_RUN_DIR" && sha256sum -c diagnostic.evidence.sha256 >/dev/null)
[[ $(cat "$SOURCE_RUN_DIR/failure_status.txt") == original_status=1 ]]
[[ $(sha256sum "$SOURCE_RUN_DIR/result.npz" | awk '{print $1}') == "$CAPTURE_SHA" ]]
[[ $(sha256sum "$SOURCE_RUN_DIR/census_post.txt" | awk '{print $1}') == "$POST_CENSUS_SHA" ]]

say "authenticating the immutable source object set, generations and CRC32C"
JAX_PLATFORMS=cpu PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$SOURCE_RUN_DIR" "$SOURCE_REMOTE" "$RECOVERY_DIR/source_remote_objects.json" <<'PY'
import base64
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

import google_crc32c

run_dir = Path(sys.argv[1])
remote = sys.argv[2]
output = Path(sys.argv[3])
ledger_path = run_dir / "diagnostic.evidence.sha256"
expected: dict[str, str] = {}
for line in ledger_path.read_text().splitlines():
    digest, relative = line.split(maxsplit=1)
    relative = relative.removeprefix("*").removeprefix("./")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or relative in expected:
        raise SystemExit("source diagnostic ledger path is unsafe or duplicated")
    expected[relative] = digest
expected["diagnostic.evidence.sha256"] = sha256(ledger_path.read_bytes()).hexdigest()

listing = subprocess.run(
    ["gcloud", "storage", "ls", "--recursive", f"{remote}/**"],
    check=True,
    stdout=subprocess.PIPE,
    text=True,
).stdout.splitlines()
expected_uris = {f"{remote}/diagnostic/{name}" for name in expected}
if set(listing) != expected_uris or len(listing) != len(expected_uris):
    raise SystemExit("source remote diagnostic object set drifted")

records = []
for relative, expected_sha in sorted(expected.items()):
    uri = f"{remote}/diagnostic/{relative}"
    metadata = json.loads(
        subprocess.run(
            ["gcloud", "storage", "objects", "describe", uri, "--format=json"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout
    )
    generation = str(metadata.get("generation", ""))
    if not generation.isdecimal():
        raise SystemExit(f"source remote generation is missing: {relative}")
    process = subprocess.Popen(
        ["gcloud", "storage", "cat", f"{uri}#{generation}"],
        stdout=subprocess.PIPE,
    )
    assert process.stdout is not None
    digest = sha256()
    checksum = google_crc32c.Checksum()
    byte_count = 0
    while chunk := process.stdout.read(8 * 1024 * 1024):
        digest.update(chunk)
        checksum.update(chunk)
        byte_count += len(chunk)
    if process.wait() != 0 or digest.hexdigest() != expected_sha:
        raise SystemExit(f"source remote object hash drifted: {relative}")
    crc32c = base64.b64encode(checksum.digest()).decode("ascii")
    remote_crc32c = metadata.get("crc32c_hash", metadata.get("crc32c"))
    if int(metadata["size"]) != byte_count or remote_crc32c != crc32c:
        raise SystemExit(f"source remote object CRC/size drifted: {relative}")
    records.append(
        {
            "byte_count": byte_count,
            "crc32c": crc32c,
            "generation": generation,
            "relative_name": relative,
            "sha256": expected_sha,
            "uri": uri,
        }
    )

manifest = {
    "artifact_kind": "greenfield_pp16_feature2_source_remote_manifest",
    "exact_object_count": len(records),
    "objects": records,
    "source_remote": remote,
    "status": "SOURCE_DIAGNOSTIC_AUTHENTICATED",
}
output.write_text(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n")
PY

say "recomputing the numerical rejection and cleanup classification without TPU"
JAX_PLATFORMS=cpu PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$SOURCE_RUN_DIR" "$RECOVERY_DIR" "$RUN_PIN" "$RECOVERY_PIN" \
  "$RECOVERY_TAG" "$RECOVERY_REMOTE" "$RESULTS_DB" "$EXPECTED_DB_MAX_RUN" \
  "$TOKEN_ORACLE_DIR" "$DSA_ORACLE_DIR" "$LAYER1_INTERNAL_REFERENCE" \
  "$DB529_INTERNAL_DIR" "$DB550_BOUNDARY" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sqlite3
import sys

from glm_tpu.greenfield.benchmarking.pp16_feature2_numerical import (
    compare_feature2_numerical_capture,
)

(
    source,
    recovery,
    run_pin,
    recovery_pin,
    recovery_tag,
    recovery_remote,
    db_path,
    expected_db_max,
    token,
    dsa,
    layer1,
    db529,
    db550,
) = sys.argv[1:]
source = Path(source)
recovery = Path(recovery)
token, dsa, layer1, db529, db550 = map(Path, (token, dsa, layer1, db529, db550))
runner = json.loads((source / "runner.json").read_text())
if (
    runner.get("status") != "NUMERICAL_CAPTURED"
    or runner.get("code_hash") != run_pin
    or runner.get("graph_sha256")
    != "ab5be45aecf3b0b5d87ad76af8076bc9351823529a08c0eadb414b072b31cb2d"
    or runner.get("main_executed") is not True
    or runner.get("main_execution_count") != 1
    or runner.get("compile_only") is not False
    or runner.get("numerical_claim") is not False
    or runner.get("performance_claim") is not False
):
    raise SystemExit("source feature2 capture claim boundary drifted")
if runner.get("physical_group") != {
    "coordinates": [[0, 0, 0], [1, 0, 0]],
    "device_ids": [0, 1],
    "local_device_count_visible": 4,
    "mesh_device_count": 2,
}:
    raise SystemExit("source feature2 physical LP2 group drifted")
runtime = runner.get("runtime_pins", {})
if (
    {name: runtime.get(name) for name in ("jax", "jaxlib", "libtpu")}
    != {"jax": "0.10.1", "jaxlib": "0.10.1", "libtpu": "0.0.41"}
    or runtime.get("backend_platform") != "tpu"
    or runtime.get("device_kind") != "TPU v4"
    or not runtime.get("platform_version")
):
    raise SystemExit("source feature2 compiler/runtime pins drifted")
main = runner.get("hlo", {}).get("feature2_main", {})
canonical = main.get("execution_canonical_hlo", {})
if (
    main.get("stablehlo", {}).get("sha256")
    != "127bf089f93bc9dd4f1b85576e8e70267a752d2be90dee74525322b5148a955e"
    or canonical.get("sha256")
    != "fb5aaf025005f3fbb5a3c66e6a719ec3a78fb86d344afcf6288e6e93720310f7"
    or canonical.get("byte_count") != 7_870_521
    or canonical.get("stripped_stack_frame_references") != 16_170
    or canonical.get("canonicalizer_version") != 1
    or canonical.get("canonicalizer_code_hash") != run_pin
):
    raise SystemExit("source feature2 HLO pins drifted")
for record in runner.get("hlo", {}).values():
    for kind in ("stablehlo", "optimized_hlo"):
        identity = record.get(kind, {})
        path = source / "hlo" / identity.get("filename", "")
        if (
            path.parent != source / "hlo"
            or not path.is_file()
            or sha256(path.read_bytes()).hexdigest() != identity.get("sha256")
        ):
            raise SystemExit(f"source feature2 {kind} identity drifted")

after_cleanup = runner.get("memory", {}).get("after_cleanup")
if (
    not isinstance(after_cleanup, list)
    or len(after_cleanup) != 2
    or any(
        record.get("bytes_in_use") != 72_812_032
        or record.get("num_allocs") != 6
        for record in after_cleanup
    )
):
    raise SystemExit("source in-process cleanup telemetry drifted")

census = source / "census_post.txt"
if sha256(census.read_bytes()).hexdigest() != (
    "cd1f25488e876acf475068131c5eb7f1ca25dabea158c553de86483aa8a40798"
):
    raise SystemExit("source post-process census hash drifted")
lines = census.read_text().splitlines()
hosts = [line.split()[1] for line in lines if line.startswith("CENSUS_OK ")]
if len(hosts) != 8 or len(set(hosts)) != 8:
    raise SystemExit("source post-process census is not exact authenticated 8/8")
if any("CENSUS_BAD" in line or "CENSUS_BUSY" in line for line in lines):
    raise SystemExit("source post-process census contains a refusal marker")

connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
db_max = connection.execute("SELECT max(run_id) FROM runs").fetchone()[0]
pattern = f"%{recovery_tag}%"
db_matches = connection.execute(
    "SELECT count(*) FROM runs WHERE model LIKE ? OR coalesce(note, '') LIKE ? "
    "OR coalesce(env_json, '') LIKE ?",
    (pattern, pattern, pattern),
).fetchone()[0]
source_matches = connection.execute(
    "SELECT count(*) FROM runs WHERE model LIKE '%feature2%' "
    "OR coalesce(note, '') LIKE '%feature2%' "
    "OR coalesce(env_json, '') LIKE '%feature2%'"
).fetchone()[0]
integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
connection.close()
if db_max != int(expected_db_max) or db_matches != 0 or source_matches != 0 or integrity != "ok":
    raise SystemExit(
        "results DB drifted or contains a feature2 numerical/performance row: "
        f"max={db_max} recovery={db_matches} source={source_matches} integrity={integrity}"
    )

comparison = compare_feature2_numerical_capture(
    source / "result.npz",
    token_oracle_dir=token,
    dsa_oracle_dir=dsa,
    layer1_internal_reference=layer1,
    db529_internal_dir=db529,
    db550_boundary=db550,
)
expected_mismatches = {
    "carried_bfloat16_bits": 968,
    "contract_valid": 0,
    "event1_positions": 1852,
    "event1_scores": 2048,
    "event1_valid_counts": 0,
    "layer1_current_key_bfloat16_bits": 0,
}
if (
    comparison.get("status") != "NUMERICAL_REJECTED"
    or comparison.get("exact") is not False
    or comparison.get("capture_sha256")
    != "be3dda446cfbde547af49dd5ea371af690b553bcc8414907add1b0a2503b9d0d"
    or comparison.get("mismatch_counts") != expected_mismatches
    or comparison.get("captured_array_sha256", {}).get(
        "current_carried_halves_bfloat16_bits"
    )
    != "3f6c86ed6e96a59adfe706a522297bf83c2ed0802a36ede9f06a88cf6f3f53d2"
    or comparison.get("expected_sha256", {}).get("carried_bfloat16_bits")
    != "35a601b7f174eb9204848757f709549a31e82774309929f4071c61849626044c"
):
    raise SystemExit("source feature2 numerical rejection drifted")

(recovery / "comparison.json").write_text(
    json.dumps(comparison, allow_nan=False, indent=2, sort_keys=True) + "\n"
)
shutil.copyfile(census, recovery / "source_census_post.txt")
cleanup = {
    "artifact_kind": "greenfield_pp16_feature2_recovery_cleanup_authentication",
    "authenticated_census_hosts": sorted(hosts),
    "authenticated_census_host_count": 8,
    "in_process_bytes_in_use_per_device": [72_812_032, 72_812_032],
    "in_process_num_allocs_per_device": [6, 6],
    "in_process_release_passed": False,
    "locks_free_before_recovery": True,
    "original_runner_process_exited": True,
    "post_process_census_sha256": sha256(census.read_bytes()).hexdigest(),
    "terminal_cleanup_authentication_passed": True,
}
(recovery / "cleanup_authentication.json").write_text(
    json.dumps(cleanup, allow_nan=False, indent=2, sort_keys=True) + "\n"
)
summary = {
    "artifact_kind": "greenfield_pp16_feature2_numerical_recovery_summary",
    "claim_scope": (
        "immutable CPU-only classification of one protected numerical capture; "
        "no SUCCESS, DB, Gate-D, token-rate, or performance claim"
    ),
    "cleanup_authentication": cleanup,
    "comparison_status": "NUMERICAL_REJECTED",
    "exact": False,
    "feature2_graph_sha256": runner["graph_sha256"],
    "gate_d_passed": False,
    "main_execution_count": 1,
    "mismatch_counts": expected_mismatches,
    "next_action": (
        "freeze this graph; offline replay of real DB550 full-width partials "
        "through the feature-half y-x-z reducer before any new TPU action"
    ),
    "numerical_claim": False,
    "performance_claim": False,
    "recovery_code_hash": recovery_pin,
    "recovery_remote": recovery_remote,
    "recovery_tag": recovery_tag,
    "results_db": {
        "integrity": integrity,
        "max_run_id": db_max,
        "matching_feature2_rows": source_matches,
        "matching_recovery_rows": db_matches,
    },
    "run_code_hash": run_pin,
    "source_remote": (
        "gs://driftbench-dsv4-uc/results/"
        "greenfield_pp16_feature2_prefill_numerical_20260829T001056567299421Z"
    ),
    "source_tag": (
        "greenfield_pp16_feature2_prefill_numerical_20260829T001056567299421Z"
    ),
    "status": "NUMERICAL_REJECTED",
}
(recovery / "summary.json").write_text(
    json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n"
)
PY

cp "$RECOVERY_DIR/orchestrator.log" "$RECOVERY_DIR/orchestrator.sealed.log"
(
  cd "$RECOVERY_DIR"
  sha256sum cleanup_authentication.json comparison.json orchestrator.sealed.log \
    remote_vacancy.stderr remote_vacancy.stdout remote_vacancy.txt \
    source_census_post.txt source_remote_objects.json summary.json
) >"$RECOVERY_DIR/recovery.evidence.sha256"
(cd "$RECOVERY_DIR" && sha256sum -c recovery.evidence.sha256 >/dev/null)

upload_preterminal() {
  local ledger=$1 relative receipt
  while read -r _ relative; do
    relative=${relative#\*}
    relative=${relative#./}
    receipt="$RECOVERY_DIR/upload_receipts/${relative//\//_}.stderr"
    gcloud storage cp --if-generation-match=0 --print-created-message \
      "$RECOVERY_DIR/$relative" "$RECOVERY_REMOTE/$relative" \
      >"${receipt%.stderr}.stdout" 2>"$receipt"
  done <"$ledger"
  gcloud storage cp --if-generation-match=0 --print-created-message \
    "$ledger" "$RECOVERY_REMOTE/$(basename "$ledger")" \
    >"$RECOVERY_DIR/upload_receipts/recovery.evidence.stdout" \
    2>"$RECOVERY_DIR/upload_receipts/recovery.evidence.stderr"
}

verify_preterminal() {
  local expected relative observed ledger_sha remote_ledger_sha
  ledger_sha=$(sha256sum "$RECOVERY_DIR/recovery.evidence.sha256" | awk '{print $1}')
  remote_ledger_sha=$(gcloud storage cat \
    "$RECOVERY_REMOTE/recovery.evidence.sha256" 2>/dev/null |
    sha256sum | awk '{print $1}')
  [[ $ledger_sha == "$remote_ledger_sha" ]] || return 1
  while read -r expected relative; do
    relative=${relative#\*}
    relative=${relative#./}
    observed=$(gcloud storage cat "$RECOVERY_REMOTE/$relative" 2>/dev/null |
      sha256sum | awk '{print $1}')
    [[ $expected == "$observed" ]] || return 1
  done <"$RECOVERY_DIR/recovery.evidence.sha256"
  diff -u \
    <(
      {
        echo "$RECOVERY_REMOTE/recovery.evidence.sha256"
        while read -r _ relative; do
          relative=${relative#\*}
          relative=${relative#./}
          echo "$RECOVERY_REMOTE/$relative"
        done <"$RECOVERY_DIR/recovery.evidence.sha256"
      } | sort
    ) \
    <(gcloud storage ls --recursive "$RECOVERY_REMOTE/**" 2>/dev/null | sort) \
    >/dev/null
}

if [[ $RECOVERY_MODE == validate_only ]]; then
  say "VALIDATED_ONLY source/authentication/comparison/cleanup pass; no remote recovery objects written"
  exit 0
fi

say "publishing the distinct recovery archive append-only"
upload_preterminal "$RECOVERY_DIR/recovery.evidence.sha256"
verify_preterminal || {
  say "ABORT: recovery preterminal archive verification failed"
  exit 70
}

JAX_PLATFORMS=cpu /home/gianl/vllm-env/bin/python - \
  "$RECOVERY_DIR/summary.json" "$RECOVERY_DIR/comparison.json" \
  "$RECOVERY_DIR/cleanup_authentication.json" \
  "$RECOVERY_DIR/source_remote_objects.json" \
  "$RECOVERY_DIR/recovery.evidence.sha256" \
  "$RECOVERY_DIR/NUMERICAL_REJECTED" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

summary, comparison, cleanup, sources, evidence, output = map(Path, sys.argv[1:])
record = {
    "artifact_kind": "greenfield_pp16_feature2_numerical_recovery_terminal",
    "cleanup_authentication_sha256": sha256(cleanup.read_bytes()).hexdigest(),
    "comparison_sha256": sha256(comparison.read_bytes()).hexdigest(),
    "evidence_sha256": sha256(evidence.read_bytes()).hexdigest(),
    "source_remote_manifest_sha256": sha256(sources.read_bytes()).hexdigest(),
    "status": "NUMERICAL_REJECTED",
    "summary_sha256": sha256(summary.read_bytes()).hexdigest(),
}
raw = json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
record["marker_self_sha256"] = sha256(raw).hexdigest()
output.write_text(json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n")
PY

terminal_publication_started=0
terminal_verified=0
rollback_unverified_terminal() {
  local describe="$RECOVERY_DIR/upload_receipts/terminal.rollback.describe.json"
  local receipt="$RECOVERY_DIR/upload_receipts/terminal_create.stderr"
  local generation
  [[ -s $receipt ]] || return 1
  gcloud storage objects describe "$RECOVERY_REMOTE/NUMERICAL_REJECTED" \
    --format=json >"$describe" 2>/dev/null || return 1
  generation=$(JAX_PLATFORMS=cpu /home/gianl/vllm-env/bin/python - \
    "$RECOVERY_DIR/NUMERICAL_REJECTED" "$receipt" "$describe" \
    "$RECOVERY_REMOTE/NUMERICAL_REJECTED" <<'PY'
import base64
import json
from pathlib import Path
import re
import sys

import google_crc32c

path, receipt, metadata = map(Path, sys.argv[1:4])
remote = sys.argv[4]
record = json.loads(metadata.read_text())
checksum = google_crc32c.Checksum(path.read_bytes())
crc32c = base64.b64encode(checksum.digest()).decode("ascii")
assert path.stat().st_size == int(record["size"])
assert crc32c == record.get("crc32c_hash", record.get("crc32c"))
generation = str(record.get("generation", ""))
assert generation.isdecimal()
matches = re.findall(r"(gs://[^\s]+)#([0-9]+)", receipt.read_text())
assert matches == [(remote, generation)]
print(generation)
PY
  ) || return 1
  gcloud storage rm --if-generation-match="$generation" \
    "$RECOVERY_REMOTE/NUMERICAL_REJECTED" \
    >"$RECOVERY_DIR/upload_receipts/terminal.rollback.stdout" \
    2>"$RECOVERY_DIR/upload_receipts/terminal.rollback.stderr"
}
on_exit() {
  local status=$?
  if [[ $terminal_publication_started -eq 1 && $terminal_verified -eq 0 ]]; then
    rollback_unverified_terminal || status=71
  fi
  trap - EXIT
  exit "$status"
}
trap on_exit EXIT

terminal_publication_started=1
gcloud storage cp --if-generation-match=0 --print-created-message \
  "$RECOVERY_DIR/NUMERICAL_REJECTED" "$RECOVERY_REMOTE/NUMERICAL_REJECTED" \
  >"$RECOVERY_DIR/upload_receipts/terminal_create.stdout" \
  2>"$RECOVERY_DIR/upload_receipts/terminal_create.stderr"
sync -f "$RECOVERY_DIR/upload_receipts/terminal_create.stderr"
gcloud storage objects describe "$RECOVERY_REMOTE/NUMERICAL_REJECTED" --format=json \
  >"$RECOVERY_DIR/upload_receipts/terminal.describe.json"

JAX_PLATFORMS=cpu /home/gianl/vllm-env/bin/python - \
  "$RECOVERY_DIR/NUMERICAL_REJECTED" \
  "$RECOVERY_DIR/upload_receipts/terminal_create.stderr" \
  "$RECOVERY_DIR/upload_receipts/terminal.describe.json" \
  "$RECOVERY_REMOTE/NUMERICAL_REJECTED" <<'PY'
import base64
import json
from pathlib import Path
import re
import sys

import google_crc32c

path, receipt, metadata = map(Path, sys.argv[1:4])
remote = sys.argv[4]
record = json.loads(metadata.read_text())
checksum = google_crc32c.Checksum(path.read_bytes())
crc32c = base64.b64encode(checksum.digest()).decode("ascii")
matches = re.findall(r"(gs://[^\s]+)#([0-9]+)", receipt.read_text())
assert matches == [(remote, str(record["generation"]))]
assert path.stat().st_size == int(record["size"])
assert crc32c == record.get("crc32c_hash", record.get("crc32c"))
PY

local_terminal_sha=$(sha256sum "$RECOVERY_DIR/NUMERICAL_REJECTED" | awk '{print $1}')
remote_terminal_sha=$(gcloud storage cat "$RECOVERY_REMOTE/NUMERICAL_REJECTED" |
  sha256sum | awk '{print $1}')
[[ $local_terminal_sha == "$remote_terminal_sha" ]]
diff -u \
  <(
    {
      echo "$RECOVERY_REMOTE/NUMERICAL_REJECTED"
      echo "$RECOVERY_REMOTE/recovery.evidence.sha256"
      while read -r _ relative; do
        relative=${relative#\*}
        relative=${relative#./}
        echo "$RECOVERY_REMOTE/$relative"
      done <"$RECOVERY_DIR/recovery.evidence.sha256"
    } | sort
  ) \
  <(gcloud storage ls --recursive "$RECOVERY_REMOTE/**" 2>/dev/null | sort) \
  >/dev/null
[[ -z $(gcloud storage ls "$SOURCE_REMOTE/NUMERICAL_REJECTED" 2>/dev/null || true) ]]
[[ -z $(gcloud storage ls "$SOURCE_REMOTE/SUCCESS" 2>/dev/null || true) ]]

terminal_verified=1
trap - EXIT
say "NUMERICAL_REJECTED sealed=$RECOVERY_REMOTE source_untouched=true no_DB_Gate-D_or_performance_claim=true"
