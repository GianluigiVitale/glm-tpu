#!/usr/bin/env bash
# Recover DB516's completed accepted run after its unbounded HLO gather was interrupted.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly LEGACY_PIN=b3c25df47ac98783912dc658878181ec0a8ae16d
readonly SOURCE_TAG=greenfield_accepted_prompt_projection_lowering_20260809T064221425525336Z
readonly SOURCE_CAPTURE_PIN=643d0926030c092ee3742e9bfe3b5c8d2511bed4
readonly SOURCE_RUN_DIR=/home/gianl/glm-run/$SOURCE_TAG
readonly SOURCE_REMOTE_ROOT=/tmp/$SOURCE_TAG
readonly TARGET_HLO=module_15877.jit_step_fun_impl.cl_914450892.after_codegen.txt
readonly TARGET_HLO_SHA=e7371f4887ecf9fa38d381dcbaf3d5953294dbcda3079cf6846635722db07216
readonly TARGET_HLO_SIZE=84017584
readonly MIN_RECOVERY_FREE_BYTES=3700000000
readonly RUN_ID=516
readonly ITEM_ROW_ID=1801
readonly SOURCE_HARNESS=a4a17ac
readonly SOURCE_FORK=b3c25df47
readonly EXPECTED_DUMP_COUNT=483
readonly EXPECTED_PROMPT_TOKENS=8155
readonly EXPECTED_GENERATED_TOKENS=20
readonly TOKEN_ORACLE_TAG=greenfield_short_context_oracle_8k_20260807T172307269147351Z
readonly TOKEN_ORACLE_SHA=e4fbcbdbf0fc8b1969e2f82ee457ab1563db4a8b37d2dea2bc4d1e828a13acf2
readonly TOKEN_ORACLE_DIR=/home/gianl/gcs-models/oracles/greenfield/glm52/short_context/8k/$TOKEN_ORACLE_TAG/oracle
readonly REFERENCE_DSA_ORACLE=/home/gianl/glm-run/greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z/oracle
readonly ORIGINAL_DUMP_PREFIX=/tmp/$SOURCE_TAG/topk.npz

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_ACCEPTED_PROJECTION_RECOVERY_TAG:-greenfield_accepted_prompt_projection_lowering_recovery_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
SOURCE_DIR=$RUN_DIR/source_dumps
ORACLE_DIR=$RUN_DIR/oracle
LOWERING_DIR=$RUN_DIR/accepted_prompt_projection_lowering
REMOTE_PREFIX=${GLM_GREENFIELD_ACCEPTED_PROJECTION_RECOVERY_REMOTE_PREFIX:-$APPROVED_BUCKET/oracles/greenfield/glm52/prompt_projection_lowering/8k/$TAG}

[[ $(git -C "$WORKTREE" rev-parse --show-toplevel) == "$WORKTREE" &&
   $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing accepted projection recovery outside the greenfield worktree" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing accepted projection recovery from a dirty worktree" >&2
  exit 2
}
[[ ! -e $RUN_DIR ]] || {
  echo "append-only recovery directory exists: $RUN_DIR" >&2
  exit 2
}
[[ -r $RESULTS_DB && -r $TOKEN_ORACLE_DIR/manifest.json &&
   -r $REFERENCE_DSA_ORACLE/manifest.json ]] || {
  echo "DB or sealed oracle prerequisite is unavailable" >&2
  exit 2
}

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  echo "another protected pod workflow holds the global lease" >&2
  exit 1
}

mkdir -p "$RUN_DIR" "$SOURCE_DIR"
say() {
  echo "[accepted-projection-recovery $(date -u +%H:%M:%S)] $*" |
    tee -a "$RUN_DIR/orchestrator.log"
}

has_eight_unique_markers() {
  local file=$1 marker=$2
  [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | wc -l) -eq 8 ]] &&
    [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | sort -u | wc -l) -eq 8 ]]
}

strict_census() {
  local label=$1
  local out="$RUN_DIR/census_${label}.txt"
  local carrier="${TAG}_${label}"
  local ray_enum command
  # shellcheck disable=SC2016
  ray_enum='GLM_CENSUS_CARRIER='"$carrier"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[c]ompile_short_decoder[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname)"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" gcloud compute tpus tpu-vm ssh "$POD" \
    --zone "$ZONE" --worker=all --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

on_exit() {
  local status=$?
  if [[ $status -ne 0 ]]; then
    if [[ ${post_census_done:-0} -eq 0 ]]; then
      strict_census failure_exit || true
    fi
    say "FAILED status=$status; preserving bounded recovery diagnostics"
    gcloud storage cp --no-clobber "$RUN_DIR/orchestrator.log" \
      "$REMOTE_PREFIX/diagnostic_local/orchestrator.log" >/dev/null 2>&1 || true
  fi
}
post_census_done=0
trap on_exit EXIT

say "SOURCE_TAG=$SOURCE_TAG SOURCE_CAPTURE_PIN=$SOURCE_CAPTURE_PIN SEALER_PIN=$PIN"
say "RUN_DIR=$RUN_DIR REMOTE_PREFIX=$REMOTE_PREFIX"
strict_census pre || {
  say "ABORT: fleet is not authenticated eight-host zero work"
  exit 1
}
free_bytes=$(df --output=avail -B1 /home/gianl | tail -1 | tr -d ' ')
printf 'free_bytes=%s\nminimum_free_bytes=%s\n' \
  "$free_bytes" "$MIN_RECOVERY_FREE_BYTES" >"$RUN_DIR/disk_preflight.txt"
[[ $free_bytes -ge $MIN_RECOVERY_FREE_BYTES ]] || {
  say "ABORT: bounded recovery lacks local disk reserve"
  exit 1
}

has_eight_unique_markers "$SOURCE_RUN_DIR/fleet_integrity.txt" INTEGRITY_OK || {
  say "ABORT: DB516 source lacks exact fleet load/state integrity"
  exit 1
}
has_eight_unique_markers "$SOURCE_RUN_DIR/fleet_prefill_profile_integrity.txt" PROFILE_OK || {
  say "ABORT: DB516 source lacks eight-host profile integrity"
  exit 1
}
grep -q "GREENFIELD_PIN=$SOURCE_CAPTURE_PIN" "$SOURCE_RUN_DIR/orchestrator.log" || {
  say "ABORT: DB516 source pin drifted"
  exit 1
}
grep -q '\[longctx\].*correct=True' "$SOURCE_RUN_DIR/legacy.log" || {
  say "ABORT: DB516 protected item was not correct"
  exit 1
}
[[ ! -s $SOURCE_RUN_DIR/census_failure_exit.txt && ! -s $SOURCE_RUN_DIR/stop.txt ]] || {
  say "ABORT: interrupted-source markers changed"
  exit 1
}

/home/gianl/vllm-env/bin/python - "$RESULTS_DB" "$RUN_DIR/source_identity.json" <<'PY'
import json
import sqlite3
import sys
from pathlib import Path

connection = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
connection.row_factory = sqlite3.Row
rows = connection.execute(
    "SELECT r.run_id,i.id AS item_row_id,r.harness_git,r.fork_git,i.correct,"
    "i.n_prompt_tokens,i.n_gen_tokens,i.raw_output FROM runs r JOIN items i "
    "ON i.run_id=r.run_id WHERE r.run_id=516 ORDER BY i.id"
).fetchall()
assert len(rows) == 1
value = dict(rows[0])
assert value == {
    "run_id": 516,
    "item_row_id": 1801,
    "harness_git": "a4a17ac",
    "fork_git": "b3c25df47",
    "correct": 1,
    "n_prompt_tokens": 8155,
    "n_gen_tokens": 20,
    "raw_output": " 881446. Do not forget it. There and back again. The grass is green",
}
Path(sys.argv[2]).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY

say "archiving only the required raw source directly from all eight hosts"
for worker in 0 1 2 3 4 5 6 7; do
  destination="$REMOTE_PREFIX/source_dumps/w$worker"
  # The remote shell must expand source paths and preserve its hostname.
  # shellcheck disable=SC2016
  command='set -euo pipefail; root='"$SOURCE_REMOTE_ROOT"'; destination='"$destination"'; expected_hlo_sha='"$TARGET_HLO_SHA"'; profile="$root/prefill_projection_profile"; xplanes=$(find "$profile" -type f -name "*.xplane.pb" | wc -l); traces=$(find "$profile" -type f -name "*.trace.json.gz" | wc -l); stats=$(find "$profile" -type f -name "batch_composition_stats_*.json" | wc -l); [ "$xplanes" -eq 1 ] && [ "$traces" -eq 1 ] && [ "$stats" -eq 2 ]; gcloud storage cp --recursive --no-clobber "$profile" "$destination/" >/dev/null; topk_count=$(find "$root" -maxdepth 1 -type f -name "topk.step*.evt*.proc*.npz" | wc -l); if [ "$topk_count" -eq '"$EXPECTED_DUMP_COUNT"' ]; then gcloud storage cp --no-clobber "$root"/topk.step*.evt*.proc*.npz "$destination/" >/dev/null; echo TOPK_OWNER $(hostname) '"$worker"'; elif [ "$topk_count" -ne 0 ]; then exit 1; fi; hlo_matches=$(find "$root/prefill_projection_hlo" -type f -name "*after_codegen.txt" -exec sha256sum {} + | awk -v expected="$expected_hlo_sha" '\''$1 == expected {print $2}'\''); [ "$(printf "%s\n" "$hlo_matches" | sed "/^$/d" | wc -l)" -eq 1 ]; hlo=$hlo_matches; [ "$(stat -c %s "$hlo")" -eq '"$TARGET_HLO_SIZE"' ]; gcloud storage cp --no-clobber "$hlo" "$destination/prefill_projection_hlo/accepted_projection.after_codegen.txt" >/dev/null; echo HLO_ARCHIVE $(hostname) '"$worker"' "$expected_hlo_sha" '"$TARGET_HLO_SIZE"'; echo RAW_ARCHIVE_OK $(hostname)'
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker="$worker" \
    --command="$command" >>"$RUN_DIR/source_archive.txt" 2>&1
done
has_eight_unique_markers "$RUN_DIR/source_archive.txt" RAW_ARCHIVE_OK || {
  say "ABORT: selected raw source did not archive from every host"
  exit 1
}
[[ $(grep -c '^TOPK_OWNER ' "$RUN_DIR/source_archive.txt") -eq 1 ]] || {
  say "ABORT: DSA source must have exactly one 483-file owner"
  exit 1
}
topk_worker=$(awk '$1 == "TOPK_OWNER" {print $3}' "$RUN_DIR/source_archive.txt")
[[ $topk_worker =~ ^[0-7]$ ]] || {
  say "ABORT: DSA source owner is invalid"
  exit 1
}
has_eight_unique_markers "$RUN_DIR/source_archive.txt" HLO_ARCHIVE || {
  say "ABORT: exact physical-M64 HLO did not archive from every host"
  exit 1
}
[[ $(awk '$1 == "HLO_ARCHIVE" {print $4, $5}' "$RUN_DIR/source_archive.txt" |
  sort -u | wc -l) -eq 1 ]] || {
  say "ABORT: fleet physical-M64 HLO bytes are not uniform"
  exit 1
}

say "materializing eight profiles and one verified physical-M64 HLO"
for worker in 0 1; do
  mkdir -p "$SOURCE_DIR/w$worker"
  cp -al "$SOURCE_RUN_DIR/source_dumps/w$worker/prefill_projection_profile" \
    "$SOURCE_DIR/w$worker/"
done
mkdir -p "$SOURCE_DIR/w0/prefill_projection_hlo"
ln "$SOURCE_RUN_DIR/source_dumps/w0/prefill_projection_hlo/$TARGET_HLO" \
  "$SOURCE_DIR/w0/prefill_projection_hlo/accepted_projection.after_codegen.txt"

say "downloading the six missing profiles from their durable prefix"
for worker in 2 3 4 5 6 7; do
  mkdir -p "$SOURCE_DIR/w$worker"
  gcloud storage cp --recursive \
    "$REMOTE_PREFIX/source_dumps/w$worker/prefill_projection_profile" \
    "$SOURCE_DIR/w$worker/" >/dev/null
done

xplane_count=$(find "$SOURCE_DIR" -type f -name '*.xplane.pb' | wc -l)
trace_count=$(find "$SOURCE_DIR" -type f -name '*.trace.json.gz' | wc -l)
stats_count=$(find "$SOURCE_DIR" -type f -name 'batch_composition_stats_*.json' | wc -l)
hlo_count=$(find "$SOURCE_DIR" -type f -name '*after_codegen.txt' | wc -l)
[[ $xplane_count -eq 8 && $trace_count -eq 8 && $stats_count -eq 16 &&
   $hlo_count -eq 1 ]] || {
  say "ABORT: profile/HLO coverage drifted xplane=$xplane_count trace=$trace_count stats=$stats_count hlo=$hlo_count"
  exit 1
}

say "verifying profile and fleet-HLO bytes by local CRC32C"
/home/gianl/vllm-env/bin/python - "$SOURCE_DIR" "$REMOTE_PREFIX/source_dumps" <<'PY' \
  >"$RUN_DIR/profile_hlo_remote_objects.json"
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import google_crc32c

root = Path(sys.argv[1])
prefix = sys.argv[2]
paths = [
    (path, path.relative_to(root).as_posix())
    for path in sorted(root.rglob("*"))
    if path.is_file()
]

def crc32c(path):
    checksum = google_crc32c.Checksum()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return base64.b64encode(checksum.digest()).decode()

def describe(item):
    path, relative = item
    value = json.loads(subprocess.run(
        ["gcloud", "storage", "objects", "describe", f"{prefix}/{relative}", "--format=json"],
        check=True, capture_output=True, text=True,
    ).stdout)
    remote_crc32c = value.get("crc32c_hash") or value.get("crc32c")
    local_crc32c = crc32c(path)
    if (
        int(value["size"]) != path.stat().st_size
        or remote_crc32c != local_crc32c
    ):
        raise SystemExit(f"raw remote object verification failed: {relative}")
    return {"crc32c": remote_crc32c, "generation": value["generation"], "path": relative, "size": int(value["size"])}

with ThreadPoolExecutor(max_workers=16) as executor:
    records = list(executor.map(describe, paths))

local_hlo = root / "w0/prefill_projection_hlo/accepted_projection.after_codegen.txt"
local_hlo_crc32c = crc32c(local_hlo)
fleet_hlo = []
for worker in range(8):
    relative = f"w{worker}/prefill_projection_hlo/accepted_projection.after_codegen.txt"
    value = json.loads(subprocess.run(
        ["gcloud", "storage", "objects", "describe", f"{prefix}/{relative}", "--format=json"],
        check=True, capture_output=True, text=True,
    ).stdout)
    remote_crc32c = value.get("crc32c_hash") or value.get("crc32c")
    if int(value["size"]) != local_hlo.stat().st_size or remote_crc32c != local_hlo_crc32c:
        raise SystemExit(f"fleet HLO remote object differs: {relative}")
    fleet_hlo.append({"crc32c": remote_crc32c, "generation": value["generation"], "path": relative, "size": int(value["size"])})

print(json.dumps({"fleet_hlo_objects": fleet_hlo, "local_objects": records}, indent=2, sort_keys=True))
PY

say "sealing the current-pin physical M64 projection association"
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
  "$WORKTREE/scripts/greenfield/inspect_accepted_prompt_projection_lowering.py" \
  --trace-root "$SOURCE_DIR" \
  --output "$LOWERING_DIR" \
  --expected-code-hash "$PIN" \
  --source-capture-code-hash "$SOURCE_CAPTURE_PIN" \
  --legacy-code-hash "$LEGACY_PIN" \
  --run-tag "$TAG" >"$RUN_DIR/accepted_prompt_projection_lowering_summary.json"

say "recording and reclaiming only unselected interrupted HLO paths"
find "$SOURCE_RUN_DIR/source_dumps/w0/prefill_projection_hlo" \
  "$SOURCE_RUN_DIR/source_dumps/w1/prefill_projection_hlo" -type f \
  -printf '%p\t%s\n' | sort >"$RUN_DIR/local_reclamation_files.tsv"
unselected_count=$(wc -l <"$RUN_DIR/local_reclamation_files.tsv")
unselected_bytes=$(awk -F '\t' '{sum += $2} END {print sum + 0}' \
  "$RUN_DIR/local_reclamation_files.tsv")
printf '{"discarded_file_count":%s,"discarded_path_bytes":%s,"reason":"unselected XLA modules from an interrupted overbroad gather","retained_evidence":"eight remote target HLO objects plus sealed compressed HLO","path_inventory":"local_reclamation_files.tsv"}\n' \
  "$unselected_count" "$unselected_bytes" >"$RUN_DIR/local_reclamation.json"
find "$SOURCE_RUN_DIR/source_dumps/w0/prefill_projection_hlo" \
  "$SOURCE_RUN_DIR/source_dumps/w1/prefill_projection_hlo" -type f -delete
find "$SOURCE_DIR" -type f \( -path '*/prefill_projection_profile/*' -o \
  -name '*after_codegen.txt' \) -delete

say "downloading the exact DSA source from worker $topk_worker"
mkdir -p "$SOURCE_DIR/w$topk_worker"
gcloud storage cp \
  "$REMOTE_PREFIX/source_dumps/w$topk_worker/topk.step*.evt*.proc*.npz" \
  "$SOURCE_DIR/w$topk_worker/" >/dev/null
dump_count=$(find "$SOURCE_DIR" -type f -name 'topk.step*.evt*.proc*.npz' | wc -l)
[[ $dump_count -eq $EXPECTED_DUMP_COUNT ]] || {
  say "ABORT: compact DSA source coverage drifted topk=$dump_count"
  exit 1
}

say "verifying every DSA source object by local CRC32C"
/home/gianl/vllm-env/bin/python - "$SOURCE_DIR" "$REMOTE_PREFIX/source_dumps" <<'PY' \
  >"$RUN_DIR/source_remote_objects.json"
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import google_crc32c

root = Path(sys.argv[1])
prefix = sys.argv[2]
paths = [(path, path.relative_to(root).as_posix()) for path in sorted(root.rglob("*")) if path.is_file()]

def crc32c(path):
    checksum = google_crc32c.Checksum()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return base64.b64encode(checksum.digest()).decode()

def describe(item):
    path, relative = item
    value = json.loads(subprocess.run(
        ["gcloud", "storage", "objects", "describe", f"{prefix}/{relative}", "--format=json"],
        check=True, capture_output=True, text=True,
    ).stdout)
    remote_crc32c = value.get("crc32c_hash") or value.get("crc32c")
    if int(value["size"]) != path.stat().st_size or remote_crc32c != crc32c(path):
        raise SystemExit(f"DSA remote object verification failed: {relative}")
    return {"crc32c": remote_crc32c, "generation": value["generation"], "path": relative, "size": int(value["size"])}

with ThreadPoolExecutor(max_workers=16) as executor:
    records = list(executor.map(describe, paths))
print(json.dumps({"objects": records}, indent=2, sort_keys=True))
PY

mkdir -p "$RUN_DIR/source_capture"
for name in census_pre.txt census_failure_exit.txt disk_preflight.txt \
  dump_hosts.txt fleet_integrity.txt fleet_prefill_profile_integrity.txt launch.log legacy.log \
  legacy_summary.json legacy_summary.jsonl orchestrator.log prereq.txt \
  raylet_env.txt raylet_prefill_profile_env.txt stop.txt; do
  cp "$SOURCE_RUN_DIR/$name" "$RUN_DIR/source_capture/$name"
done
printf 'source_tag=%s\nsource_capture_pin=%s\nsealer_pin=%s\nsource_wrapper_terminal_status=interrupted_during_unbounded_gather\n' \
  "$SOURCE_TAG" "$SOURCE_CAPTURE_PIN" "$PIN" >"$RUN_DIR/source_recovery.txt"

say "sealing DB516 DSA events and comparing the immutable 8K oracle"
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
  "$WORKTREE/scripts/greenfield/capture_short_context_dsa_oracle.py" \
  --results-db "$RESULTS_DB" \
  --token-oracle-dir "$TOKEN_ORACLE_DIR" \
  --source-dump-dir "$SOURCE_DIR" \
  --output "$ORACLE_DIR" \
  --expected-code-hash "$PIN" \
  --source-capture-code-hash "$SOURCE_CAPTURE_PIN" \
  --legacy-repository-pin "$LEGACY_PIN" \
  --token-oracle-manifest-sha256 "$TOKEN_ORACLE_SHA" \
  --run-id "$RUN_ID" \
  --item-row-id "$ITEM_ROW_ID" \
  --expected-harness-git "$SOURCE_HARNESS" \
  --expected-fork-git "$SOURCE_FORK" \
  --expected-benchmark passkey_L8192_d0.5 \
  --expected-model-uri gs://driftbench-dsv4-uc/models/GLM-5.2-FP8 \
  --expected-prompt-tokens "$EXPECTED_PROMPT_TOKENS" \
  --expected-generated-tokens "$EXPECTED_GENERATED_TOKENS" \
  --expected-seed 1093997 \
  --expected-gold 881446 \
  --expected-oob-dir /home/gianl/gcs-models/models/GLM-5.2-FP8 \
  --expected-dump-prefix "$ORIGINAL_DUMP_PREFIX" \
  --expected-process-count 8 \
  --first-source-step 5 \
  --decode-step-count 14 \
  --first-decode-position 8155 \
  --selected-width 2048 >"$RUN_DIR/capture.json"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$REFERENCE_DSA_ORACLE" "$ORACLE_DIR" >"$RUN_DIR/dsa_exact_comparison.json" <<'PY'
import json
from pathlib import Path
import sys
from glm_tpu.greenfield.validation import compare_short_context_dsa_oracles

value = compare_short_context_dsa_oracles(Path(sys.argv[1]), Path(sys.argv[2]))
assert value["exact"]
print(json.dumps(value, indent=2, sort_keys=True))
PY

/home/gianl/vllm-env/bin/python - "$RESULTS_DB" "$RUN_DIR/results_ckpt.db" <<'PY'
import sqlite3
import sys
source = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
destination = sqlite3.connect(sys.argv[2])
source.backup(destination)
assert destination.execute("pragma integrity_check").fetchone()[0] == "ok"
destination.close()
source.close()
PY

say "removing the exact archived source tag from all workers"
# The hostname and bounded root check intentionally execute on each TPU host.
# shellcheck disable=SC2016
cleanup='set -e; root='"$SOURCE_REMOTE_ROOT"'; [ "$root" = /tmp/greenfield_accepted_prompt_projection_lowering_20260809T064221425525336Z ]; find "$root" -depth -delete; [ ! -e "$root" ]; echo SOURCE_CLEAN_OK $(hostname)'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$cleanup" >"$RUN_DIR/source_cleanup.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/source_cleanup.txt" SOURCE_CLEAN_OK || {
  say "ABORT: archived DB516 source tag did not clean on all hosts"
  exit 1
}

strict_census post || {
  say "ABORT: post-recovery fleet is not authenticated eight-host zero work"
  exit 1
}
post_census_done=1

say "freezing compact recovered evidence"
cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
/home/gianl/vllm-env/bin/python - "$RUN_DIR" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
records = []
for path in sorted(root.rglob("*")):
    relative = path.relative_to(root).as_posix()
    if not path.is_file() or relative == "orchestrator.log" or path.name in {"SUCCESS", "evidence_sha256.json"}:
        continue
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    records.append({"byte_count": path.stat().st_size, "path": relative, "sha256": digest.hexdigest()})
(root / "evidence_sha256.json").write_text(json.dumps({"files": records}, indent=2, sort_keys=True) + "\n")
PY

say "uploading recovered DB516 evidence append-only"
gcloud storage cp --recursive --no-clobber "$RUN_DIR"/* "$REMOTE_PREFIX/" >/dev/null

/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$REMOTE_PREFIX" <<'PY' \
  >"$RUN_DIR/remote_objects.json"
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import google_crc32c

root = Path(sys.argv[1])
prefix = sys.argv[2]
paths = []
for path in sorted(root.rglob("*")):
    relative = path.relative_to(root).as_posix()
    if path.is_file() and relative != "orchestrator.log" and path.name not in {"SUCCESS", "remote_objects.json"}:
        paths.append((path, relative))

def crc32c(path):
    checksum = google_crc32c.Checksum()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return base64.b64encode(checksum.digest()).decode()

def describe(item):
    path, relative = item
    value = json.loads(subprocess.run(
        ["gcloud", "storage", "objects", "describe", f"{prefix}/{relative}", "--format=json"],
        check=True, capture_output=True, text=True,
    ).stdout)
    remote_crc32c = value.get("crc32c_hash") or value.get("crc32c")
    if int(value["size"]) != path.stat().st_size or remote_crc32c != crc32c(path):
        raise SystemExit(f"remote object verification failed: {relative}")
    return {"crc32c": remote_crc32c, "generation": value["generation"], "path": relative, "size": int(value["size"])}

with ThreadPoolExecutor(max_workers=16) as executor:
    records = list(executor.map(describe, paths))
print(json.dumps({"objects": records}, indent=2, sort_keys=True))
PY
gcloud storage cp --no-clobber "$RUN_DIR/remote_objects.json" \
  "$REMOTE_PREFIX/remote_objects.json" >/dev/null

/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$REMOTE_PREFIX" "$PIN" "$SOURCE_CAPTURE_PIN" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
oracle = json.loads((root / "oracle" / "manifest.json").read_text())
lowering = json.loads((root / "accepted_prompt_projection_lowering" / "summary.json").read_text())
lowering_manifest = json.loads((root / "accepted_prompt_projection_lowering" / "manifest.json").read_text())
exact = json.loads((root / "dsa_exact_comparison.json").read_text())
assert exact["exact"]
assert lowering["status"] == "SUCCESS"
assert lowering["profile"]["file_count"] == 8
assert lowering["profile"]["core_count"] == 64
assert lowering["profile"]["steps_per_core"] == 1
assert lowering["profile"]["invocations_per_core"] == 21
assert lowering["hlo"]["convolution_count"] == 21
assert lowering["shape_association"]["physical_hlo_result"] == "f32[64,128]"
assert lowering["shape_association"]["profile_physical_result"] == "f32[64,128]"
profile_hlo_remote = json.loads((root / "profile_hlo_remote_objects.json").read_text())
assert len(profile_hlo_remote["fleet_hlo_objects"]) == 8
values = {
    "artifact_kind": "accepted_prompt_projection_lowering_recovery_v1",
    "sealer_code_hash": sys.argv[3],
    "source_capture_code_hash": sys.argv[4],
    "source_run_id": oracle["source"]["run_id"],
    "source_item_row_id": oracle["source"]["item_row_id"],
    "dsa_oracle_manifest_sha256": oracle["manifest_sha256"],
    "dsa_event_tensors_exact": "true",
    "accepted_prompt_projection_manifest_sha256": lowering_manifest["manifest_sha256"],
    "accepted_prompt_projection_physical_result": lowering["shape_association"]["physical_hlo_result"],
    "accepted_prompt_projection_emitter": lowering["hlo"]["emitter"],
    "accepted_prompt_projection_convolution_count": lowering["hlo"]["convolution_count"],
    "accepted_prompt_projection_xplane_file_count": lowering["profile"]["file_count"],
    "accepted_prompt_projection_core_count": lowering["profile"]["core_count"],
    "accepted_prompt_projection_hlo_host_count": len(profile_hlo_remote["fleet_hlo_objects"]),
    "source_wrapper_terminal_status": "interrupted_during_unbounded_gather",
    "evidence_sha256": sha256((root / "evidence_sha256.json").read_bytes()).hexdigest(),
    "source_remote_objects_sha256": sha256((root / "source_remote_objects.json").read_bytes()).hexdigest(),
    "profile_hlo_remote_objects_sha256": sha256((root / "profile_hlo_remote_objects.json").read_bytes()).hexdigest(),
    "remote_objects_sha256": sha256((root / "remote_objects.json").read_bytes()).hexdigest(),
    "remote_prefix": sys.argv[2],
}
(root / "SUCCESS").write_text("".join(f"{key}={value}\n" for key, value in values.items()))
PY
gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" "$REMOTE_PREFIX/SUCCESS" >/dev/null
local_success_sha=$(sha256sum "$RUN_DIR/SUCCESS" | awk '{print $1}')
remote_success_sha=$(gcloud storage cat "$REMOTE_PREFIX/SUCCESS" | sha256sum | awk '{print $1}')
[[ $local_success_sha == "$remote_success_sha" ]] || {
  say "ABORT: remote terminal SUCCESS checksum mismatch"
  exit 1
}

trap - EXIT
say "SUCCESS source_run=$RUN_ID item=$ITEM_ROW_ID physical=f32[64,128]"
