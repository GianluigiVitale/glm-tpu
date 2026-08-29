#!/usr/bin/env bash
# CPU-only recovery seal for the completed full-width PP16 rejection.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly APPROVED_LOCATION=US-CENTRAL2
readonly SOURCE_TAG=greenfield_pp16_feature2_prefill_numerical_20260829T051119686986506Z
readonly SOURCE_RUN_DIR=/home/gianl/glm-run/$SOURCE_TAG
readonly SOURCE_REMOTE=$APPROVED_BUCKET/results/$SOURCE_TAG
readonly TOKEN_ORACLE_DIR=/home/gianl/gcs-models/oracles/greenfield/glm52/short_context/8k/greenfield_short_context_oracle_8k_20260807T172307269147351Z/oracle
readonly DSA_ORACLE_DIR=/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/8k/greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z/oracle
readonly LAYER1_INTERNAL_REFERENCE=/home/gianl/gcs-models/oracles/greenfield/glm52/dsa_internals/8k/layer1/greenfield_layer1_dsa_internal_comparison_20260808T115135394251231Z/internals.npz
readonly DB529_INTERNAL_DIR=/home/gianl/glm-run/greenfield_layer0_dsa_scorer_association_20260810T164030202890642Z/inputs/internal
readonly DB550_BOUNDARY=/home/gianl/gcs-models/results/greenfield_layer0_dense_partial_capture_20260813T200736889447458Z/dense_partial_capture.npz
readonly HALF_WIDTH_CAPTURE=/home/gianl/glm-run/greenfield_pp16_feature2_prefill_numerical_20260829T001056567299421Z/result.npz
readonly DB518_PROMPT_CACHE_DIR=/home/gianl/glm-run/greenfield_layer0_dsa_scorer_association_20260810T164030202890642Z/inputs/prompt_cache

[[ ${GLM_GREENFIELD_PP16_FEATURE2_FULL_WIDTH_RECOVER:-0} == 1 ]] || {
  echo "PP16 full-width feature2 recovery is default-off" >&2
  exit 2
}
RECOVERY_MODE=${GLM_GREENFIELD_PP16_FEATURE2_FULL_WIDTH_RECOVERY_MODE:-off}
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
RECOVERY_TAG=${GLM_GREENFIELD_PP16_FEATURE2_FULL_WIDTH_RECOVERY_TAG:-greenfield_pp16_feature2_full_width_recovery_$(date -u +%Y%m%dT%H%M%S%NZ)}
RECOVERY_DIR=/home/gianl/glm-run/$RECOVERY_TAG
RECOVERY_REMOTE=$APPROVED_BUCKET/results/$RECOVERY_TAG
readonly RECOVERY_PIN RECOVERY_TAG RECOVERY_DIR RECOVERY_REMOTE

[[ $RECOVERY_TAG =~ ^greenfield_pp16_feature2_full_width_recovery_[0-9]{8}T[0-9]{15}Z$ ]]
[[ $(git -C "$WORKTREE" rev-parse --show-toplevel) == "$WORKTREE" ]]
[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]]
[[ -z $(git -C "$WORKTREE" status --porcelain) ]]
[[ $(git -C "$WORKTREE" ls-remote origin "refs/heads/$BRANCH" | awk '{print $1}') == "$RECOVERY_PIN" ]]
[[ $(gcloud storage buckets describe "$APPROVED_BUCKET" --format='value(location)') == "$APPROVED_LOCATION" ]]
[[ -d $SOURCE_RUN_DIR && -r $RESULTS_DB && ! -e $RECOVERY_DIR ]]
[[ -f $SOURCE_RUN_DIR/NUMERICAL_REJECTED && ! -e $SOURCE_RUN_DIR/NUMERICAL_EXACT ]]

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

mkdir -p "$RECOVERY_DIR/source_failure" "$RECOVERY_DIR/upload_receipts"
say() {
  echo "[pp16-feature2-full-width-recovery $(date -u +%H:%M:%S)] $*" |
    tee -a "$RECOVERY_DIR/orchestrator.log"
}

has_eight_unique_markers() {
  local file=$1 marker=$2
  [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | wc -l) -eq 8 ]] &&
    [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | sort -u | wc -l) -eq 8 ]]
}

strict_census() {
  local label=$1
  local out="$RECOVERY_DIR/census_${label}.txt"
  local carrier="${RECOVERY_TAG}_${label}" ray_enum command
  # shellcheck disable=SC2016
  ray_enum='GLM_CENSUS_CARRIER='"$carrier"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "[a]cquire_pp16_feature2_prefill[.]py|[e]xecute_pp16_feature2_prefill[.]py|[c]ompile_short_decoder[.]py|[r]un_short_decoder|[V]LLM::EngineCore|[R]ayWorkerWrapper|[m]icrobench" 2>/dev/null || true); holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; if [[ $tools_ok -ne 1 || $ray_rc -ne 0 || $docker_rc -ne 0 ]]; then echo "CENSUS_BAD $(hostname)"; elif [[ -n $ray_pids || -n $generic || -n $holders ]] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [[ -z $ray_pids ]] || echo "ray_stop_pids: $ray_pids"; [[ -z $generic ]] || echo "$generic"; [[ -z $holders ]] || echo "libtpu holders: $holders"; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" gcloud compute tpus tpu-vm ssh \
    "$POD" --zone "$ZONE" --worker=all --command="$command" >"$out" 2>&1 ||
    return 1
  has_eight_unique_markers "$out" CENSUS_OK
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

for name in NUMERICAL_REJECTED orchestrator.log terminal_create.receipt.stderr \
  terminal_create.stdout \
  terminal_upload.describe.json terminal_rollback.describe.json \
  terminal_rollback.stderr; do
  cp "$SOURCE_RUN_DIR/$name" "$RECOVERY_DIR/source_failure/$name"
done

say "proving fresh pre-recovery 8/8 zero work"
strict_census recovery_pre

say "recomputing the complete local numerical/HLO/cleanup/terminal rejection"
JAX_PLATFORMS=cpu PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$SOURCE_RUN_DIR" "$SOURCE_REMOTE" "$TOKEN_ORACLE_DIR" \
  "$DSA_ORACLE_DIR" "$LAYER1_INTERNAL_REFERENCE" "$DB529_INTERNAL_DIR" \
  "$DB550_BOUNDARY" "$HALF_WIDTH_CAPTURE" "$DB518_PROMPT_CACHE_DIR" \
  "$RESULTS_DB" "$RECOVERY_TAG" \
  "$RECOVERY_DIR/local_authentication.json" <<'PY'
import json
from pathlib import Path
import sys

from glm_tpu.greenfield.benchmarking.pp16_feature2_recovery import (
    authenticate_feature2_full_width_rejection,
)

(
    source,
    source_remote,
    token,
    dsa,
    layer1,
    db529,
    db550,
    half_width,
    db518_prompt_cache,
    results_db,
    recovery_tag,
    output,
) = sys.argv[1:]
record = authenticate_feature2_full_width_rejection(
    Path(source),
    source_remote=source_remote,
    token_oracle_dir=Path(token),
    dsa_oracle_dir=Path(dsa),
    layer1_internal_reference=Path(layer1),
    db529_internal_dir=Path(db529),
    db550_boundary=Path(db550),
    half_width_capture=Path(half_width),
    db518_prompt_cache_dir=Path(db518_prompt_cache),
    results_db=Path(results_db),
    recovery_tag=recovery_tag,
)
Path(output).write_text(
    json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n"
)
PY

say "authenticating all 28 immutable source objects by generation/CRC/SHA"
JAX_PLATFORMS=cpu PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$SOURCE_RUN_DIR" "$SOURCE_REMOTE" \
  "$RECOVERY_DIR/local_authentication.json" \
  "$RECOVERY_DIR/source_remote_objects.json" <<'PY'
import base64
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import google_crc32c

source = Path(sys.argv[1])
remote = sys.argv[2]
authentication = json.loads(Path(sys.argv[3]).read_text())
output = Path(sys.argv[4])
expected = dict(authentication["ledger_entries"])
expected["evidence.sha256"] = authentication["source_ledger_sha256"]
expected["NUMERICAL_REJECTED"] = authentication["terminal"]["sha256"]
listing = subprocess.run(
    ["gcloud", "storage", "ls", "--recursive", f"{remote}/**"],
    check=True,
    stdout=subprocess.PIPE,
    text=True,
).stdout.splitlines()
expected_uris = {f"{remote}/{name}" for name in expected}
if len(expected) != 28 or set(listing) != expected_uris:
    raise SystemExit("feature2 source remote object set drifted")
records = []
for relative, expected_sha in sorted(expected.items()):
    uri = f"{remote}/{relative}"
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
        raise SystemExit(f"feature2 source generation missing: {relative}")
    process = subprocess.Popen(
        ["gcloud", "storage", "cat", f"{uri}#{generation}"],
        stdout=subprocess.PIPE,
    )
    if process.stdout is None:
        raise SystemExit(f"feature2 source generation read has no stdout: {relative}")
    digest = sha256()
    checksum = google_crc32c.Checksum()
    byte_count = 0
    while chunk := process.stdout.read(8 * 1024 * 1024):
        digest.update(chunk)
        checksum.update(chunk)
        byte_count += len(chunk)
    crc32c = base64.b64encode(checksum.digest()).decode("ascii")
    crc_values = [
        str(metadata[key])
        for key in ("crc32c_hash", "crc32c")
        if metadata.get(key) not in (None, "")
    ]
    if not crc_values or len(set(crc_values)) != 1:
        raise SystemExit(f"feature2 source remote CRC absent/conflicting: {relative}")
    remote_crc32c = crc_values[0]
    if (
        process.wait() != 0
        or digest.hexdigest() != expected_sha
        or int(metadata.get("size", -1)) != byte_count
        or remote_crc32c != crc32c
    ):
        raise SystemExit(f"feature2 source remote content drifted: {relative}")
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
terminal = authentication["terminal"]
terminal_remote = next(
    record for record in records if record["relative_name"] == "NUMERICAL_REJECTED"
)
if (
    terminal_remote["generation"] != terminal["generation"]
    or terminal_remote["crc32c"] != terminal["crc32c"]
):
    raise SystemExit("feature2 source terminal generation/CRC drifted")
manifest = {
    "artifact_kind": "greenfield_pp16_feature2_full_width_source_remote_manifest",
    "exact_object_count": len(records),
    "objects": records,
    "source_remote": remote,
    "status": "SOURCE_REJECTION_AUTHENTICATED",
}
output.write_text(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n")
PY

say "proving fresh post-recovery 8/8 zero work"
strict_census recovery_post

JAX_PLATFORMS=cpu PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RECOVERY_DIR/local_authentication.json" \
  "$RECOVERY_DIR/source_remote_objects.json" \
  "$RECOVERY_DIR/census_recovery_pre.txt" \
  "$RECOVERY_DIR/census_recovery_post.txt" \
  "$RECOVERY_PIN" "$RECOVERY_TAG" "$RECOVERY_REMOTE" \
  "$RECOVERY_DIR/summary.json" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

local_path, remote_path, pre_path, post_path = map(Path, sys.argv[1:5])
recovery_pin, recovery_tag, recovery_remote = sys.argv[5:8]
output = Path(sys.argv[8])
local = json.loads(local_path.read_text())
remote = json.loads(remote_path.read_text())

def census(path):
    lines = path.read_text().splitlines()
    hosts = sorted(line.split()[1] for line in lines if line.startswith("CENSUS_OK "))
    if len(hosts) != 8 or len(set(hosts)) != 8 or any(
        "CENSUS_BAD" in line or "CENSUS_BUSY" in line for line in lines
    ):
        raise SystemExit("feature2 recovery fresh census drifted")
    return {"hosts": hosts, "sha256": sha256(path.read_bytes()).hexdigest()}

summary = {
    "artifact_kind": "greenfield_pp16_feature2_full_width_recovery_summary",
    "claim_scope": (
        "CPU-only recovery of one already-completed protected numerical rejection; "
        "no model rerun, DB, Gate-D, token-rate, or performance claim"
    ),
    "db518_cache_localization": local["db518_cache_localization"],
    "exact": False,
    "fresh_census_post": census(post_path),
    "fresh_census_pre": census(pre_path),
    "gate_d_passed": False,
    "local_authentication_sha256": sha256(local_path.read_bytes()).hexdigest(),
    "main_execution_count": 1,
    "mismatch_counts": local["comparison"]["mismatch_counts"],
    "next_action": (
        "freeze full-width-rounded-then-slice; add observation-only layer-0 roots "
        "at earliest prompt-cache mismatch position 113 / hidden index 35 before "
        "another model run"
    ),
    "numerical_claim": False,
    "original_terminal_generation": local["terminal"]["generation"],
    "original_terminal_sha256": local["terminal"]["sha256"],
    "performance_claim": False,
    "recovery_code_hash": recovery_pin,
    "recovery_remote": recovery_remote,
    "recovery_tag": recovery_tag,
    "source_code_hash": local["source_code_hash"],
    "full_width_vs_half_width": local["full_width_vs_half_width"],
    "source_remote_manifest_sha256": sha256(remote_path.read_bytes()).hexdigest(),
    "source_tag": local["source_tag"],
    "status": "NUMERICAL_REJECTED",
}
output.write_text(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n")
PY

cp "$RECOVERY_DIR/orchestrator.log" "$RECOVERY_DIR/orchestrator.sealed.log"
(
  cd "$RECOVERY_DIR"
  find source_failure -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum census_recovery_post.txt census_recovery_pre.txt \
    local_authentication.json orchestrator.sealed.log remote_vacancy.stderr \
    remote_vacancy.stdout remote_vacancy.txt source_remote_objects.json summary.json
) >"$RECOVERY_DIR/recovery.evidence.sha256"
(cd "$RECOVERY_DIR" && sha256sum -c recovery.evidence.sha256 >/dev/null)

if [[ $RECOVERY_MODE == validate_only ]]; then
  say "VALIDATED_ONLY source/local/remote/terminal/census rejection pass; no recovery objects written"
  exit 0
fi

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

say "publishing the distinct recovery archive append-only"
upload_preterminal "$RECOVERY_DIR/recovery.evidence.sha256"
verify_preterminal || {
  say "ABORT: recovery preterminal archive verification failed"
  exit 70
}

JAX_PLATFORMS=cpu /home/gianl/vllm-env/bin/python - \
  "$RECOVERY_DIR/summary.json" "$RECOVERY_DIR/local_authentication.json" \
  "$RECOVERY_DIR/source_remote_objects.json" \
  "$RECOVERY_DIR/recovery.evidence.sha256" \
  "$RECOVERY_DIR/NUMERICAL_REJECTED" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

summary, local, sources, evidence, output = map(Path, sys.argv[1:])
record = {
    "artifact_kind": "greenfield_pp16_feature2_full_width_recovery_terminal",
    "evidence_sha256": sha256(evidence.read_bytes()).hexdigest(),
    "local_authentication_sha256": sha256(local.read_bytes()).hexdigest(),
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
generation = str(record.get("generation", ""))
crc_values = [
    str(record[key])
    for key in ("crc32c_hash", "crc32c")
    if record.get(key) not in (None, "")
]
matches = re.findall(r"(gs://[^\s]+)#([0-9]+)", receipt.read_text())
if (
    not generation.isdecimal()
    or not crc_values
    or len(set(crc_values)) != 1
    or matches != [(remote, generation)]
    or path.stat().st_size != int(record.get("size", -1))
    or crc32c != crc_values[0]
):
    raise SystemExit("recovery-terminal rollback authentication failed")
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
gcloud storage objects describe "$RECOVERY_REMOTE/NUMERICAL_REJECTED" \
  --format=json >"$RECOVERY_DIR/upload_receipts/terminal.describe.json"

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
crc_values = [
    str(record[key])
    for key in ("crc32c_hash", "crc32c")
    if record.get(key) not in (None, "")
]
matches = re.findall(r"(gs://[^\s]+)#([0-9]+)", receipt.read_text())
if (
    not crc_values
    or len(set(crc_values)) != 1
    or matches != [(remote, str(record.get("generation", "")))]
    or path.stat().st_size != int(record.get("size", -1))
    or crc32c != crc_values[0]
):
    raise SystemExit("recovery-terminal authentication failed")
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
[[ $(gcloud storage objects describe "$SOURCE_REMOTE/NUMERICAL_REJECTED" \
  --format='value(generation)') == 1787980539108624 ]]

terminal_verified=1
trap - EXIT
say "NUMERICAL_REJECTED sealed=$RECOVERY_REMOTE source_untouched=true no_model_rerun_DB_Gate-D_or_performance_claim=true"
