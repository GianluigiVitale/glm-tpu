#!/usr/bin/env bash
# No-TPU sealer for the protected PP16 LP2 dense-boundary NONEXACT run.
set -euo pipefail

readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly APPROVED_LOCATION=US-CENTRAL2
readonly SOURCE_TAG=greenfield_pp16_lp2_dense_boundary_20260828T133048539902952Z
readonly RUN_DIR=/home/gianl/glm-run/$SOURCE_TAG
readonly REMOTE_PREFIX=$APPROVED_BUCKET/results/$SOURCE_TAG
readonly RUN_PIN=1f5c88ebd1c1dc5206075c22da26ede8cff90f2f
readonly ORIGINAL_LEDGER_SHA=738b4f03707d7b782ad9ce470760478cf7dbac92c0d490cb6cf2dc51e016a0b1
readonly EXPECTED_DB_MAX_RUN=564

[[ ${GLM_GREENFIELD_PP16_LP2_DENSE_BOUNDARY_RECOVER:-0} == 1 ]] || {
  echo "PP16 LP2 dense-boundary failure recovery is default-off" >&2
  exit 2
}
command -v gsutil >/dev/null 2>&1 && gsutil help rsync >/dev/null 2>&1 || {
  echo "gsutil rsync is required for authenticated recovery" >&2
  exit 2
}

RECOVERY_PIN=$(git -C "$WORKTREE" rev-parse HEAD)
readonly RECOVERY_PIN
[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing recovery outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing recovery from a dirty worktree" >&2
  exit 2
}
[[ $(git -C "$WORKTREE" ls-remote origin "refs/heads/$BRANCH" | awk '{print $1}') == "$RECOVERY_PIN" ]] || {
  echo "recovery pin is not the exact pushed branch head" >&2
  exit 2
}
[[ $(gcloud storage buckets describe "$APPROVED_BUCKET" --format='value(location)') == "$APPROVED_LOCATION" ]]
[[ -d $RUN_DIR && -r $RESULTS_DB && ! -e $RUN_DIR/SUCCESS ]]

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  echo "another protected TPU workflow holds the global lease" >&2
  exit 1
}
exec 8>/home/gianl/.glm-tpu-rsync.lock
flock 8

remote_objects=$(gcloud storage ls --recursive "$REMOTE_PREFIX/**" 2>/dev/null || true)
if grep -Fqx "$REMOTE_PREFIX/SUCCESS" <<<"$remote_objects"; then
  echo "refusing recovery over a remote SUCCESS marker" >&2
  exit 2
fi
unexpected_remote=$(grep -v "^$REMOTE_PREFIX/diagnostic/" <<<"$remote_objects" |
  grep -v -F "$REMOTE_PREFIX/REJECTED" || true)
[[ -z $unexpected_remote ]] || {
  echo "remote prefix contains an unexpected object" >&2
  exit 2
}

[[ $(sha256sum "$RUN_DIR/diagnostic.evidence.sha256" | awk '{print $1}') == "$ORIGINAL_LEDGER_SHA" ]]
(cd "$RUN_DIR" && sha256sum -c diagnostic.evidence.sha256 >/dev/null)
[[ $(cat "$RUN_DIR/failure_status.txt") == original_status=3 ]]

/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$RUN_PIN" "$RESULTS_DB" "$EXPECTED_DB_MAX_RUN" <<'PY'
import json
from pathlib import Path
import sqlite3
import sys

run_dir = Path(sys.argv[1])
run_pin = sys.argv[2]
db_path = sys.argv[3]
expected_db_max = int(sys.argv[4])
runner = json.loads((run_dir / "runner.json").read_text())
if runner.get("status") != "NONEXACT" or runner.get("code_hash") != run_pin:
    raise SystemExit("source run status or code pin drifted")
if not runner.get("hlo_contract", {}).get("passed"):
    raise SystemExit("source HLO contract did not pass")
if runner["physical_group"].get("device_ids") != [0, 1]:
    raise SystemExit("source physical device group drifted")
if runner["physical_group"].get("coordinates") != [[0, 0, 0], [1, 0, 0]]:
    raise SystemExit("source physical coordinates drifted")
if runner["physical_group"].get("replica_groups") != [[0, 1]]:
    raise SystemExit("source replica group drifted")
if not all(record.get("passed") for record in runner.get("determinism", {}).values()):
    raise SystemExit("source determinism evidence drifted")
if not runner.get("replica_agreement", {}).get("passed"):
    raise SystemExit("source replica agreement drifted")
if any(
    runner[name].get("elementwise_exact")
    for name in (
        "dense_update_comparison",
        "next_residual_comparison",
        "layer1_comparison",
    )
):
    raise SystemExit("source NONEXACT comparisons drifted")

def census_ok(path: Path) -> bool:
    hosts = {
        line.split()[1]
        for line in path.read_text().splitlines()
        if line.startswith("CENSUS_OK ") and len(line.split()) >= 2
    }
    return len(hosts) == 8

for name in ("census_pre.txt", "census_failure_exit.txt"):
    if not census_ok(run_dir / name):
        raise SystemExit(f"source census is not authenticated 8/8: {name}")

connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
max_run = connection.execute("SELECT max(run_id) FROM runs").fetchone()[0]
count = connection.execute(
    "SELECT count(*) FROM runs WHERE model LIKE '%pp16-lp2-dense%'"
).fetchone()[0]
integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
connection.close()
if max_run != expected_db_max or count != 0 or integrity != "ok":
    raise SystemExit(
        f"results DB drifted: max_run={max_run} pp16_rows={count} integrity={integrity}"
    )
PY

cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
(
  cd "$RUN_DIR"
  find . -type f ! -name recovery.evidence.sha256 ! -name REJECTED -print0 |
    sort -z | xargs -0 sha256sum
) >"$RUN_DIR/recovery.evidence.sha256"
(cd "$RUN_DIR" && sha256sum -c recovery.evidence.sha256 >/dev/null)

verify_remote_ledger() {
  local expected relative observed ledger_sha remote_ledger_sha
  ledger_sha=$(sha256sum "$RUN_DIR/recovery.evidence.sha256" | awk '{print $1}')
  remote_ledger_sha=$(gcloud storage cat \
    "$REMOTE_PREFIX/diagnostic/recovery.evidence.sha256" 2>/dev/null |
    sha256sum | awk '{print $1}')
  [[ $ledger_sha == "$remote_ledger_sha" ]] || return 1
  while read -r expected relative; do
    relative=${relative#\*}
    relative=${relative#./}
    observed=$(gcloud storage cat \
      "$REMOTE_PREFIX/diagnostic/$relative" 2>/dev/null |
      sha256sum | awk '{print $1}')
    [[ $expected == "$observed" ]] || return 1
  done <"$RUN_DIR/recovery.evidence.sha256"
  diff -u \
    <(
      {
        echo "$REMOTE_PREFIX/diagnostic/recovery.evidence.sha256"
        while read -r _ relative; do
          relative=${relative#\*}
          relative=${relative#./}
          echo "$REMOTE_PREFIX/diagnostic/$relative"
        done <"$RUN_DIR/recovery.evidence.sha256"
      } | sort
    ) \
    <(gcloud storage ls --recursive "$REMOTE_PREFIX/diagnostic/**" 2>/dev/null | sort) \
    >/dev/null
}

if grep -Fqx "$REMOTE_PREFIX/REJECTED" <<<"$remote_objects"; then
  [[ -r $RUN_DIR/REJECTED ]] || {
    echo "remote REJECTED exists without its local source" >&2
    exit 2
  }
  verify_remote_ledger
  local_rejected_sha=$(sha256sum "$RUN_DIR/REJECTED" | awk '{print $1}')
  remote_rejected_sha=$(gcloud storage cat "$REMOTE_PREFIX/REJECTED" | sha256sum | awk '{print $1}')
  [[ $local_rejected_sha == "$remote_rejected_sha" ]]
  echo "PP16_LP2_DENSE_BOUNDARY_REJECTION_ALREADY_SEALED tag=$SOURCE_TAG"
  exit 0
fi

uploaded=0
for attempt in 1 2; do
  if gsutil -m rsync -r -x '(^|/)REJECTED$' "$RUN_DIR" \
    "$REMOTE_PREFIX/diagnostic" >/dev/null 2>&1 &&
    verify_remote_ledger; then
    uploaded=1
    break
  fi
done
[[ $uploaded -eq 1 ]] || {
  echo "authenticated recovery upload failed after two attempts" >&2
  exit 70
}

/home/gianl/vllm-env/bin/python - \
  "$RUN_DIR/runner.json" "$RUN_DIR/recovery.evidence.sha256" \
  "$RUN_DIR/REJECTED" "$SOURCE_TAG" "$RUN_PIN" "$RECOVERY_PIN" \
  "$REMOTE_PREFIX" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

runner_path = Path(sys.argv[1])
ledger_path = Path(sys.argv[2])
output = Path(sys.argv[3])
record = {
    "artifact_kind": "greenfield_pp16_lp2_dense_boundary_rejection",
    "run_tag": sys.argv[4],
    "run_code_hash": sys.argv[5],
    "recovery_code_hash": sys.argv[6],
    "remote_prefix": sys.argv[7],
    "original_status": 3,
    "runner_sha256": sha256(runner_path.read_bytes()).hexdigest(),
    "recovery_ledger_sha256": sha256(ledger_path.read_bytes()).hexdigest(),
    "performance_claim": False,
    "gate_d_passed": False,
    "status": "REJECTED",
}
raw = json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
record["rejected_sha256"] = sha256(raw).hexdigest()
output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY

gcloud storage cp --no-clobber "$RUN_DIR/REJECTED" "$REMOTE_PREFIX/REJECTED" >/dev/null
local_rejected_sha=$(sha256sum "$RUN_DIR/REJECTED" | awk '{print $1}')
remote_rejected_sha=$(gcloud storage cat "$REMOTE_PREFIX/REJECTED" | sha256sum | awk '{print $1}')
[[ $local_rejected_sha == "$remote_rejected_sha" ]]
[[ -z $(gcloud storage ls "$REMOTE_PREFIX/SUCCESS" 2>/dev/null || true) ]]

echo "PP16_LP2_DENSE_BOUNDARY_REJECTION_SEALED tag=$SOURCE_TAG run_pin=$RUN_PIN recovery_pin=$RECOVERY_PIN"
