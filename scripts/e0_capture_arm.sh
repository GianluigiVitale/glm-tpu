#!/bin/bash
# e0_capture_arm.sh <sparse|dense> — one provenance-complete 20-step E0
# decode capture through the in-worker GLM_JAX_TRACE hook. Success requires a
# clean 8-host/64-core fleet, exactly 20 selected steps per core, the requested
# arm's device-op identity, a successful throughput driver, and durable GCS
# archival. Every retry has a separate local trace directory.
set -uo pipefail

case "${1:-}" in
  sparse|dense) ARM="$1" ;;
  *) echo "usage: $0 <sparse|dense>" >&2; exit 2 ;;
esac

ZONE=us-central2-b
POD=db-v4-64-od
PIN=94b746433
TRACE_STEPS=20
OOB_DIR=/home/gianl/gcs-models/models/GLM-5.2-FP8
TAG=e0cap_${ARM}_$(date -u +%Y%m%dT%H%M%S%NZ)
RUN_DIR=$HOME/glm-run/$TAG
GCS_RUN=gs://driftbench-dsv4-uc/results/$TAG
HEALTH_PROOF_LOG="${HEALTH_PROOF_LOG:?set HEALTH_PROOF_LOG to a fresh protected 5K health log}"
HEALTH_RAYLET_ENVS="${HEALTH_RAYLET_ENVS:?set HEALTH_RAYLET_ENVS to its 8-host exact env census}"
HEALTH_RESULTS_DB="${HEALTH_RESULTS_DB:-$HOME/glm-tpu/bench/results.db}"
HEALTH_MAX_AGE_S="${HEALTH_MAX_AGE_S:-1800}"
DRIVER_TIMEOUT_S="${DRIVER_TIMEOUT_S:-7200}"
[ "$HEALTH_MAX_AGE_S" -le 1800 ] 2>/dev/null || {
  echo "HEALTH_MAX_AGE_S must be an integer <= 1800" >&2
  exit 2
}
ALERT_FILE=$RUN_DIR/DISK_ALERT
mkdir -p "$RUN_DIR"
say() { echo "[e0cap-$ARM $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"; }

# Common local lease for every protected resume-health/E0 pod workflow.
exec 9>"$HOME/glm-run/.glm_pod_workload.lock"
flock -n 9 || { say "ABORT: another protected pod workflow holds the lock"; exit 1; }

TRC_BASE="GLM_FLIGHT_RECORDER=1 GLM_JAX_TRACE_SKIP=4 GLM_JAX_TRACE_STEPS=$TRACE_STEPS"
LIBTPU='LIBTPU_INIT_ARGS="--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false"'
if [ "$ARM" = sparse ]; then
RENV='GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=8 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 GLM_WRITE_PROBE=1 GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_CHECKSUM=1 GLM_STATE_HASH_REF=/tmp/golden.json GLM_WK_OOB_DIR='"$OOB_DIR"' GLM_WK_OOB_GOLDEN=/tmp/golden.json GLM_EXPECT_CODE_HASH='"$PIN"
  DEXTRA="GLM_STATE_HASH_REF=/tmp/golden.json GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=8 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2"
  # Expanded only after injection into the remote command.
  # shellcheck disable=SC2016
  RAYLET_ARM_CHECK='grep -qx "GLM_DSA_MODE=pallas_decode" "$env_file"'
else
  RENV='GLM_MLA_DCP=1 GLM_DCP=8 GLM_DCP_SCATTER_IMPL=pageloop GLM_WRITE_PROBE=1 GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_NAN_CHECK_IGNORE=.self_attn.indexer. GLM_LOAD_CHECKSUM=1 GLM_STATE_HASH_REF=/tmp/golden.json GLM_STATE_HASH_REF_IGNORE=.self_attn.indexer. GLM_WK_OOB_DIR='"$OOB_DIR"' GLM_WK_OOB_GOLDEN=/tmp/golden.json GLM_EXPECT_CODE_HASH='"$PIN"
  DEXTRA="GLM_STATE_HASH_REF=/tmp/golden.json GLM_STATE_HASH_REF_IGNORE=.self_attn.indexer. GLM_LOAD_NAN_CHECK_IGNORE=.self_attn.indexer. GLM_MLA_DCP=1 GLM_DCP=8 GLM_DCP_SCATTER_IMPL=pageloop"
  # shellcheck disable=SC2016
  RAYLET_ARM_CHECK='! grep -q "^GLM_DSA_MODE=" "$env_file"'
fi
DRIVER="NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray \
OMP_NUM_THREADS=1 HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 \
DISABLE_WEIGHT_REQUANTIZATION=1 REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn \
TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 \
GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_CHECKSUM=1 \
GLM_WK_OOB_DIR=$OOB_DIR GLM_WK_OOB_GOLDEN=/tmp/golden.json \
GLM_EXPECT_CODE_HASH=$PIN $DEXTRA"
REFUSAL_RE='StateHashMismatchError|LoadNanCheckError|PwalNanCheckError|CodeFingerprintMismatchError'

has_8_unique_markers() {
  local file="$1" marker="$2" lines unique
  lines=$(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | wc -l)
  unique=$(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | sort -u | wc -l)
  [ "$lines" -eq 8 ] && [ "$unique" -eq 8 ]
}

# The first census is strict: the launcher's reset is broad, so no Ray/vLLM,
# known benchmark, ASPt process/container, or libtpu holder may exist anywhere.
initial_pod_census() {
  local label="${1:-initial}"
  local out=$RUN_DIR/census_${label}.txt
  # This is a literal remote program; expansion must happen on each worker.
  # shellcheck disable=SC2016
  local cmd='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[r]aylet|[g]lm_longctx[.]py|[d]sa_throughput[.]py|[r]un_bench[.]py|[g]ate_sparse128k[.]sh|[s]tage256k[.]sh|[b]ench_run[.]sh" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$cmd" > "$out" 2>&1 || { tee -a "$RUN_DIR/orchestrator.log" < "$out"; return 1; }
  tee -a "$RUN_DIR/orchestrator.log" < "$out"
  has_8_unique_markers "$out" CENSUS_OK
}

# After the initial zero-work census, Ray processes belong to this capture.
# Before a retry reset, still refuse if an ASPt-specific process/container has
# appeared; also preserve a full process/lock census in the run bundle.
retry_ownership_census() {
  local try="$1"
  local out=$RUN_DIR/census_retry_owner_t${try}.txt
  # shellcheck disable=SC2016
  local cmd='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; pids=$(pgrep -f "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[r]aylet" 2>/dev/null || true); holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); bad=""; for p in $pids $holders; do env_file=/proc/$p/environ; if [ ! -r "$env_file" ] || ! tr "\0" "\n" < "$env_file" | grep -qx "GLM_EXPECT_CODE_HASH='"$PIN"'" || ! tr "\0" "\n" < "$env_file" | grep -qx "GLM_JAX_TRACE_STEPS='"$TRACE_STEPS"'" || ! tr "\0" "\n" < "$env_file" | grep -qx "GLM_JAX_TRACE_DIR='"$TRACE_REMOTE"'" || ! '"$RAYLET_ARM_CHECK"'; then bad="$bad $p"; fi; done; if [ "$tools_ok" -ne 1 ] || [ "$docker_rc" -ne 0 ] || [ -n "$bad" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "RETRY_OWNER_BAD $(hostname) bad_pids=$bad"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "RETRY_OWNER_OK $(hostname)"; [ -n "$pids" ] && ps -o pid,args -p $(echo "$pids" | tr "\n" " "); [ -n "$holders" ] && echo "libtpu holders: $holders"; fi'
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$cmd" > "$out" 2>&1 || { tee -a "$RUN_DIR/orchestrator.log" < "$out"; return 1; }
  tee -a "$RUN_DIR/orchestrator.log" < "$out"
  has_8_unique_markers "$out" RETRY_OWNER_OK
}

ensure_oob_mounts() {
  local out="$1"
  # hostname expands on the remote worker.
  # shellcheck disable=SC2016
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command='mountpoint -q ~/gcs-models || gcsfuse --implicit-dirs -o ro --stat-cache-ttl 1h --type-cache-ttl 1h driftbench-dsv4-uc ~/gcs-models >/dev/null 2>&1; test -r '"$OOB_DIR"'/model.safetensors.index.json && echo "OOB_OK $(hostname)"' \
    > "$out" 2>&1 || return 1
  has_8_unique_markers "$out" OOB_OK
}

static_preflight() {
  local golden_log=$RUN_DIR/golden_census.txt hash_log=$RUN_DIR/code_census.txt
  [ -z "${GLM_STATE_HASH_WRITE:-}" ] || { say "ABORT: GLM_STATE_HASH_WRITE is armed"; return 1; }
  if pgrep -af 'gate_sparse128k[.]sh|stage256k[.]sh|bench_run[.]sh|glm_longctx[.]py|dsa_throughput[.]py|run_bench[.]py' \
      > "$RUN_DIR/census_local.txt" 2>&1; then
    tee -a "$RUN_DIR/orchestrator.log" < "$RUN_DIR/census_local.txt"
    say "ABORT: local workload collision"
    return 1
  fi
  # shellcheck disable=SC2016
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command='test -r /tmp/golden.json && echo "GOLDEN_OK $(hostname)"' \
    > "$golden_log" 2>&1 || true
  has_8_unique_markers "$golden_log" GOLDEN_OK || {
    say "ABORT: golden manifests not proven on 8 unique hosts"; return 1; }
  # shellcheck disable=SC2016
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command='cd ~/tpu-inference || exit; h=$(git rev-parse --short=9 HEAD); d=$(git status --porcelain --untracked-files=no | wc -l); if [ "$h" = '"$PIN"' ] && [ "$d" -eq 0 ]; then echo HASH_OK "$(hostname)" "$h"; else echo HASH_BAD "$(hostname)" "$h" dirty="$d"; fi' \
    > "$hash_log" 2>&1 || true
  has_8_unique_markers "$hash_log" HASH_OK || {
    say "ABORT: clean fork pin not proven on 8 unique hosts @ $PIN"; return 1; }
  ensure_oob_mounts "$RUN_DIR/oob_preflight.txt" || {
    say "ABORT: OOB mirror not proven on 8 unique hosts"; return 1; }
  bash "$HOME/glm-tpu/scripts/disk_watchdog.sh" check | tee -a "$RUN_DIR/orchestrator.log"
}

verify_raylet_envs() {
  local out=$RUN_DIR/raylet_env_t${1}.txt
  # shellcheck disable=SC2016
  local cmd='P=$(pgrep -x raylet | head -1); env_file=/tmp/e0_raylet_env_check_$$; [ -n "$P" ] && tr "\0" "\n" < "/proc/$P/environ" > "$env_file" && grep -qx "GLM_JAX_TRACE_STEPS='"$TRACE_STEPS"'" "$env_file" && grep -qx "GLM_JAX_TRACE_DIR='"$TRACE_REMOTE"'" "$env_file" && grep -qx "GLM_EXPECT_CODE_HASH='"$PIN"'" "$env_file" && grep -qx "GLM_WK_OOB_DIR='"$OOB_DIR"'" "$env_file" && grep -qx "GLM_WK_OOB_GOLDEN=/tmp/golden.json" "$env_file" && grep -qx "GLM_DCP=8" "$env_file" && '"$RAYLET_ARM_CHECK"' && echo "RAYLET_ENV_OK $(hostname)"; rc=$?; rm -f "$env_file"; exit "$rc"'
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$cmd" > "$out" 2>&1 || true
  tee -a "$RUN_DIR/orchestrator.log" < "$out"
  has_8_unique_markers "$out" RAYLET_ENV_OK
}

validate_analysis() {
  "$HOME/vllm-env/bin/python" - "$1" "$ARM" \
    "$HOME/glm-tpu/scripts/analysis" <<'PY'
import json
import sys

path, arm, parser_dir = sys.argv[1:]
sys.path.insert(0, parser_dir)
import parse_xplane

with open(path) as f:
    d = json.load(f)
parse_xplane.validate_fleet_expectations(
    d, n_files=8, n_cores=64, n_hosts=8, cores_per_host=8,
    steps_per_core=20, arm=arm, dsa_invocations_per_step=78,
    all_reduce_invocations_per_step=232)
assert d["device_step_ms"] > 0
assert d["busy_ms_per_step"] > 0
assert d["ops"].get("all-reduce", {}).get("invocations_per_step", 0) > 0
print("ANALYSIS_VALID", arm, d["n_files"], d["n_cores"], d["steps_per_core"])
PY
}

mkdir -p "$RUN_DIR"
if ! {
  printf 'tag=%s\narm=%s\npin=%s\ntrace_steps=%s\noob_dir=%s\ngcs_run=%s\n' \
    "$TAG" "$ARM" "$PIN" "$TRACE_STEPS" "$OOB_DIR" "$GCS_RUN"
  printf 'raylet_envs=%s %s GLM_JAX_TRACE_DIR=<per-try-nonce> %s\ndriver_envs=%s\n' \
    "$RENV" "$TRC_BASE" "$LIBTPU" "$DRIVER"
  printf 'glm_tpu_head=%s\nglm_tpu_dirty=%s\ntpu_inference_head=%s\ntpu_inference_dirty=%s\n' \
    "$(git -C "$HOME/glm-tpu" rev-parse HEAD)" \
    "$(git -C "$HOME/glm-tpu" status --porcelain --untracked-files=no | wc -l)" \
    "$(git -C "$HOME/tpu-inference" rev-parse HEAD)" \
    "$(git -C "$HOME/tpu-inference" status --porcelain --untracked-files=no | wc -l)"
} > "$RUN_DIR/config.txt"; then
  say "ABORT: could not stage capture configuration"
  exit 1
fi
if ! cp "$HOME/glm-tpu/scripts/e0_capture_arm.sh" "$RUN_DIR/e0_capture_arm.sh" ||
    ! cp "$HOME/glm-tpu/scripts/analysis/parse_xplane.py" "$RUN_DIR/parse_xplane.py"; then
  say "ABORT: could not snapshot capture tooling"
  exit 1
fi

say "mandatory zero-work pod census before any launch action"
initial_pod_census || { say "ABORT: shared-pod collision or incomplete census"; exit 1; }
static_preflight || exit 1
if [ ! -r "$HEALTH_PROOF_LOG" ] || [ ! -r "$HEALTH_RAYLET_ENVS" ] ||
    ! has_8_unique_markers "$HEALTH_RAYLET_ENVS" HEALTH_ENV_OK ||
    ! grep -q "GLM_CODE_FINGERPRINT: git=$PIN.*dirty=0.*GLM_WK_OOB_DIR.*GLM_WK_OOB_GOLDEN" "$HEALTH_PROOF_LOG" ||
    ! grep -q 'manifest VERIFIED.*repeated 7x across cluster' "$HEALTH_PROOF_LOG" ||
    ! grep -q '\[longctx\] L=5000 .*correct=True' "$HEALTH_PROOF_LOG"; then
  say "ABORT: health proof lacks exact 8-host repair envs + pinned VERIFIED + 5K correct=True"
  exit 1
fi
HEALTH_LINK=$RUN_DIR/health_link.json
if ! "$HOME/vllm-env/bin/python" - "$HEALTH_RESULTS_DB" "$HEALTH_PROOF_LOG" \
    "$HEALTH_RAYLET_ENVS" \
    "$HEALTH_LINK" "$PIN" "$OOB_DIR" "$HEALTH_MAX_AGE_S" <<'PY'
import datetime
import json
import pathlib
import re
import sqlite3
import sys

db, log_path, raylet_path, out, pin, oob, max_age = sys.argv[1:]
text = pathlib.Path(log_path).read_text(errors="replace")
run_ids = {int(value) for value in re.findall(r"\[longctx\] run_id=(\d+)", text)}
assert len(run_ids) == 1, run_ids
run_id = next(iter(run_ids))
conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row
row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
assert row is not None
run = dict(row)
env = json.loads(run["env_json"])
created = datetime.datetime.fromisoformat(run["created_utc"]).timestamp()
age = datetime.datetime.now(datetime.timezone.utc).timestamp() - created
assert 0 <= age <= int(max_age), (age, max_age)
assert run["fork_git"] == pin
tag_match = re.fullmatch(r"resume health proof \((resume_health_[0-9TZ]+)\)", run["note"])
assert tag_match, run["note"]
tag = tag_match.group(1)
assert env["attention_path"] == "dsa-sparse:pallas_decode"
assert env["lengths"] == [5000] and env["depths"] == [0.5] and env["trials"] == 1
os_env = env["os_env"]
expected = {
    "GLM_WK_OOB_DIR": oob,
    "GLM_WK_OOB_GOLDEN": "/tmp/golden.json",
    "GLM_STATE_HASH_REF": "/tmp/golden.json",
    "GLM_EXPECT_CODE_HASH": pin,
    "GLM_DCP": "4",
    "GLM_DSA_MODE": "pallas_decode",
}
assert {key: os_env.get(key) for key in expected} == expected
raylet_lines = [line for line in pathlib.Path(raylet_path).read_text().splitlines()
                if line.startswith("HEALTH_ENV_OK ")]
assert len(raylet_lines) == 8
hosts = set()
suffix = (f"HEALTH_TAG={tag} GLM_WK_OOB_DIR={oob} "
          "GLM_WK_OOB_GOLDEN=/tmp/golden.json "
          "GLM_STATE_HASH_REF=/tmp/golden.json "
          f"GLM_EXPECT_CODE_HASH={pin} GLM_DCP=4 GLM_DSA_MODE=pallas_decode")
for line in raylet_lines:
    _, host, values = line.split(maxsplit=2)
    assert values == suffix, (values, suffix)
    hosts.add(host)
assert len(hosts) == 8, hosts
items = [dict(item) for item in conn.execute(
    "SELECT benchmark,item_id,correct,n_prompt_tokens,n_gen_tokens "
    "FROM items WHERE run_id = ?", (run_id,)
)]
assert items == [{
    "benchmark": "passkey_L5000_d0.5", "item_id": "t0", "correct": 1,
    "n_prompt_tokens": 4977, "n_gen_tokens": 20,
}], items
summary = [dict(item) for item in conn.execute(
    "SELECT benchmark,n,metric,value FROM summary WHERE run_id = ? ORDER BY benchmark",
    (run_id,)
)]
assert summary == [
    {"benchmark": "longctx_passkey", "n": 0, "metric": "acc", "value": 100.0},
    {"benchmark": "passkey_L5000_d0.5", "n": 1, "metric": "acc", "value": 100.0},
], summary
conn.close()
with open(out, "w") as f:
    json.dump({"age_s": age, "health_tag": tag, "run": run, "env": env,
               "items": items, "summary": summary}, f, indent=2)
print("HEALTH_LINK_VALID", run_id, f"age_s={age:.1f}")
PY
then
  say "ABORT: embedded health run is stale, wrong-pin/config, or not exact 5K success"
  exit 1
fi
if ! cp "$HEALTH_PROOF_LOG" "$RUN_DIR/health_proof.log" ||
    ! cp "$HEALTH_RAYLET_ENVS" "$RUN_DIR/health_raylet_envs.txt" ||
    ! sha256sum "$RUN_DIR/health_proof.log" "$RUN_DIR/health_raylet_envs.txt" \
      > "$RUN_DIR/health_proof.sha256"; then
  say "ABORT: could not stage health proof"
  exit 1
fi
say "health proof attached: embedded run <=${HEALTH_MAX_AGE_S}s pin=$PIN exact repair envs, 8-host manifest VERIFIED, 5K correct=True"

ALERT_FILE="$ALERT_FILE" INTERVAL_S=120 setsid nohup \
  bash "$HOME/glm-tpu/scripts/disk_watchdog.sh" watch </dev/null \
  > "$RUN_DIR/disk_watch.log" 2>&1 &
WATCH_PID=$!
DRIVER_SESSION=""
stop_driver_session() {
  if [[ "$DRIVER_SESSION" =~ ^[0-9]+$ ]] &&
      kill -0 -- "-$DRIVER_SESSION" 2>/dev/null; then
    say "stopping task-owned driver session $DRIVER_SESSION"
    kill -TERM -- "-$DRIVER_SESSION" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      kill -0 -- "-$DRIVER_SESSION" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$DRIVER_SESSION" 2>/dev/null || true
  fi
  DRIVER_SESSION=""
}
cleanup() {
  stop_driver_session
  kill "$WATCH_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM

for try in 1 2 3 4 5 6; do
  if [ "$try" -gt 1 ]; then
    retry_ownership_census "$try" || { say "ABORT: retry processes are not positively capture-owned"; exit 1; }
    gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
      --command='~/vllm-env/bin/ray stop --force >/dev/null 2>&1' >/dev/null 2>&1 || {
        say "ABORT: retry Ray reset failed"; exit 1; }
    sleep 45
  fi
  bash "$HOME/glm-tpu/scripts/disk_watchdog.sh" check | tee -a "$RUN_DIR/orchestrator.log" || {
    say "ABORT: disk preflight failed before try $try"; exit 1; }
  [ ! -s "$ALERT_FILE" ] || { say "ABORT: disk watcher alert"; exit 1; }

  TRACE_REMOTE=/tmp/glm-jaxtrace_${TAG}_t${try}
  TRC="$TRC_BASE GLM_JAX_TRACE_DIR=$TRACE_REMOTE"
  TRY_START_EPOCH=$(date +%s)
  CLEAR_LOG=$RUN_DIR/trace_clear_t${try}.txt
  # hostname expands on each remote worker.
  # shellcheck disable=SC2016
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command='rm -rf '"$TRACE_REMOTE"'; if [ ! -e '"$TRACE_REMOTE"' ]; then echo "TRACE_CLEAR_OK $(hostname)"; else echo "TRACE_CLEAR_BAD $(hostname)"; fi' \
    > "$CLEAR_LOG" 2>&1 || true
  if ! has_8_unique_markers "$CLEAR_LOG" TRACE_CLEAR_OK; then
    say "try $try: exact nonce trace-path clear not proven on 8 unique hosts"
    continue
  fi
  # Close the census-to-kill window as far as this shared pod permits: this is
  # the final command before the broad-reset launcher.
  initial_pod_census "prelaunch_t${try}" || {
    say "ABORT: zero-work census failed immediately before launcher"
    exit 1
  }
  say "try $try: launching protected $ARM engine"
  if ! EXTRA_ENVS="$RENV $TRC $LIBTPU" TPU_MIN_TOKEN_BUCKET=32 \
      bash "$HOME/glm-tpu/scripts/launch_glm_32chip.sh" \
      > "$RUN_DIR/launch_t${try}.log" 2>&1; then
    say "try $try: launcher failed"; continue
  fi
  ensure_oob_mounts "$RUN_DIR/oob_postlaunch_t${try}.txt" || {
    say "try $try: post-launch OOB not proven on 8 unique hosts"; continue; }
  NODES=$("$HOME/vllm-env/bin/ray" status 2>/dev/null | grep -cE '^ 1 node_' || true)
  [ "${NODES:-0}" -eq 8 ] || { say "try $try: Ray nodes ${NODES:-0}/8"; continue; }
  verify_raylet_envs "$try" || { say "try $try: raylet env verification failed"; continue; }

  LOG=$RUN_DIR/driver_t${try}.log
  # The background PID is the new session leader. EXIT/signal/disk/timeout
  # cleanup can therefore terminate the exact task-owned process group.
  # The single-quoted program is intentionally evaluated by the child bash.
  # shellcheck disable=SC2016,SC2086
  setsid --wait bash -c '
    cd "$1" || exit 1
    set -a; . "$2"; set +a
    if [ -n "${GLM_STATE_HASH_WRITE:-}" ]; then
      echo "ABORT: GLM_STATE_HASH_WRITE came from .env" >&2
      exit 91
    fi
    unset GLM_STATE_HASH_WRITE
    shift 2
    exec "$@"
  ' e0-driver "$HOME/glm-tpu/bench" "$HOME/glm-tpu/.env" \
    env $DRIVER $TRC "$HOME/vllm-env/bin/python" -u dsa_throughput.py \
    --ctxs 262144 --num-seqs 1 --measure-tokens 256 --gmu 0.90 \
    --max-batched-tokens 2048 --num-gpu-blocks 66 --max-len 262400 \
    --out-json "$RUN_DIR/throughput_t${try}.json" \
    --note "E0 capture $ARM ($TAG) try=$try" </dev/null > "$LOG" 2>&1 &
  W=$!
  DRIVER_SESSION=$W
  sleep 1
  DRIVER_SID=$(ps -o sid= -p "$W" 2>/dev/null | tr -d ' ')
  if kill -0 "$W" 2>/dev/null && [ "$DRIVER_SID" != "$W" ]; then
    say "ABORT: could not establish task-owned driver session (pid=$W sid=${DRIVER_SID:-none})"
    stop_driver_session
    wait "$W" 2>/dev/null || true
    exit 1
  fi
  WAITED=0
  while kill -0 "$W" 2>/dev/null; do
    sleep 30
    WAITED=$((WAITED + 30))
    if [ -s "$ALERT_FILE" ]; then
      say "ABORT: disk watcher alert during driver"
      stop_driver_session
      wait "$W" 2>/dev/null || true
      exit 1
    fi
    if [ "$WAITED" -ge "$DRIVER_TIMEOUT_S" ]; then
      say "ABORT: driver timeout after ${WAITED}s"
      stop_driver_session
      wait "$W" 2>/dev/null || true
      exit 1
    fi
  done
  wait "$W" 2>/dev/null
  DRIVER_RC=$?
  DRIVER_SESSION=""
  echo "DRIVER_EXIT=$DRIVER_RC" >> "$LOG"

  TRACE_META=$RUN_DIR/remote_trace_meta_t${try}.txt
  # shellcheck disable=SC2016
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command='find '"$TRACE_REMOTE"' -type f -name "*.xplane.pb" -printf "TRACE_FILE $(hostname) %T@ %p\n" 2>/dev/null' \
    > "$TRACE_META" 2>&1 || true
  TRACE_META_OK=0
  if has_8_unique_markers "$TRACE_META" TRACE_FILE &&
      "$HOME/vllm-env/bin/python" - "$TRACE_META" "$TRACE_REMOTE" \
        "$TRY_START_EPOCH" <<'PY'
import pathlib
import sys
import time

meta, nonce_path, started = sys.argv[1], sys.argv[2], float(sys.argv[3])
rows = []
for line in pathlib.Path(meta).read_text().splitlines():
    if not line.startswith("TRACE_FILE "):
        continue
    _, host, mtime, path = line.split(maxsplit=3)
    rows.append((host, float(mtime), path))
assert len(rows) == 8 and len({row[0] for row in rows}) == 8, rows
assert all(path.startswith(nonce_path + "/") for _, _, path in rows), rows
now = time.time()
assert all(started - 5 <= mtime <= now + 5 for _, mtime, _ in rows), rows
assert max(mtime for _, mtime, _ in rows) - min(mtime for _, mtime, _ in rows) <= 120, rows
print("TRACE_META_VALID", len(rows), "skew_s=", max(r[1] for r in rows) - min(r[1] for r in rows))
PY
  then
    TRACE_META_OK=1
  fi
  if [ "$TRACE_META_OK" -ne 1 ]; then
    say "try $try: fresh nonce/timestamp trace identity failed"
    continue
  fi

  TRY_TRACE=$RUN_DIR/trace_t${try}
  mkdir -p "$TRY_TRACE"
  SCP_OK=0
  for w in 0 1 2 3 4 5 6 7; do
    if gcloud compute tpus tpu-vm scp --zone "$ZONE" --worker="$w" --recurse \
        "$POD:$TRACE_REMOTE" "$TRY_TRACE/w$w" >/dev/null 2>&1; then
      SCP_OK=$((SCP_OK + 1))
    else
      say "try $try: trace SCP failed on worker $w"
    fi
  done
  TRACE_MANIFEST=$RUN_DIR/trace_manifest_t${try}.txt
  TRACE_MANIFEST_OK=0
  if find "$TRY_TRACE" -name '*.xplane.pb' -print0 | sort -z | \
      xargs -0 -r sha256sum > "$TRACE_MANIFEST" &&
      [ "$(wc -l < "$TRACE_MANIFEST")" -eq 8 ]; then
    TRACE_MANIFEST_OK=1
  fi

  ANALYSIS_JSON=$RUN_DIR/analysis_t${try}.json
  ANALYSIS_MD=$RUN_DIR/analysis_t${try}.md
  PARSE_OK=0
  if [ "$SCP_OK" -eq 8 ] && [ "$TRACE_MANIFEST_OK" -eq 1 ] &&
      "$HOME/vllm-env/bin/python" "$HOME/glm-tpu/scripts/analysis/parse_xplane.py" \
        fleet "$TRY_TRACE" "$ANALYSIS_JSON" > "$ANALYSIS_MD" 2>&1 &&
      validate_analysis "$ANALYSIS_JSON" >> "$ANALYSIS_MD" 2>&1; then
    PARSE_OK=1
  fi

  if grep -q 'DRIVER_EXIT=0' "$LOG" && grep -q 'manifest VERIFIED' "$LOG" &&
      [ "$PARSE_OK" -eq 1 ]; then
    RUN_LINK=$RUN_DIR/run_link_t${try}.json
    HARNESS_SHORT=$(git -C "$HOME/glm-tpu" rev-parse --short HEAD)
    if ! "$HOME/vllm-env/bin/python" - \
      "$HOME/glm-tpu/bench/results.db" "$RUN_DIR/results_ckpt.db" \
      "$RUN_LINK" "$TAG" "$ARM" "$PIN" "$HARNESS_SHORT" \
      "$TRACE_REMOTE" "$TRY_START_EPOCH" "$try" <<'PY'
import datetime
import json
import sqlite3
import sys

db, snapshot, out, tag, arm, pin, harness, trace_dir, started, attempt = sys.argv[1:]
src = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
src.row_factory = sqlite3.Row
rows = src.execute(
    "SELECT * FROM runs WHERE note = ? ORDER BY run_id",
    (f"E0 capture {arm} ({tag}) try={attempt}",)
).fetchall()
assert len(rows) == 1, f"expected one tagged DB run, got {len(rows)}"
run = dict(rows[0])
env = json.loads(run["env_json"])
expected_path = "dsa-sparse:pallas_decode" if arm == "sparse" else "dense-mla"
assert run["harness_git"] == harness, (run["harness_git"], harness)
assert run["fork_git"] == pin, (run["fork_git"], pin)
assert env["attention_path"] == expected_path, env["attention_path"]
assert env["ctxs"] == [262144] and env["num_seqs"] == 1
assert env["measure_tokens"] == 256 and env["dcp"] == 8
os_env = env["os_env"]
assert os_env["GLM_JAX_TRACE_DIR"] == trace_dir
assert os_env["GLM_JAX_TRACE_STEPS"] == "20"
assert os_env["GLM_EXPECT_CODE_HASH"] == pin
created = datetime.datetime.fromisoformat(run["created_utc"]).timestamp()
assert created >= float(started) - 5, (created, started)
summary = [dict(r) for r in src.execute(
    "SELECT * FROM summary WHERE run_id = ? ORDER BY id", (run["run_id"],)
)]
items = [dict(r) for r in src.execute(
    "SELECT id,run_id,benchmark,item_id,n_prompt_tokens,n_gen_tokens,latency_ms,finish_reason "
    "FROM items WHERE run_id = ? ORDER BY id", (run["run_id"],)
)]
assert len(summary) == 2, summary
by_benchmark = {r["benchmark"]: r for r in summary}
assert set(by_benchmark) == {"dsa_throughput", "dsa_throughput_ctx262144"}
assert by_benchmark["dsa_throughput_ctx262144"]["metric"] == "decode_tok_s"
assert by_benchmark["dsa_throughput_ctx262144"]["value"] > 0
assert len(items) == 1, items
item = items[0]
assert item["benchmark"] == "dsa_throughput_ctx262144"
assert item["item_id"] == "seq0" and item["n_prompt_tokens"] == 262144
assert item["n_gen_tokens"] == 256, item

dst = sqlite3.connect(snapshot)
src.backup(dst)
assert dst.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
snap = dst.execute("SELECT note FROM runs WHERE run_id = ?", (run["run_id"],)).fetchone()
assert snap and snap[0] == run["note"]
dst.close()
src.close()
with open(out, "w") as f:
    json.dump({"run": run, "env": env, "summary": summary, "items": items}, f, indent=2)
print("RUN_LINK_VALID", run["run_id"], expected_path, trace_dir)
PY
    then
      say "try $try: exact tagged results.db linkage/snapshot failed"
      continue
    fi
    if [ ! -s "$RUN_LINK" ] || [ ! -s "$RUN_DIR/results_ckpt.db" ] ||
        [ ! -s "$RUN_DIR/throughput_t${try}.json" ]; then
      say "try $try: required DB artifacts missing"
      continue
    fi
    say "CAPTURE VALID: driver+manifest, 8 hosts, 64 cores, $TRACE_STEPS steps/core, arm=$ARM"
    ARCHIVE_LOG_TMP=/tmp/${TAG}_archive.log
    if ! gcloud storage rsync -r "$RUN_DIR" "$GCS_RUN" > "$ARCHIVE_LOG_TMP" 2>&1; then
      say "ABORT: durable archive failed (local evidence retained at $RUN_DIR)"
      exit 1
    fi
    cp "$ARCHIVE_LOG_TMP" "$RUN_DIR/archive.log" || {
      say "ABORT: could not stage archive log"; exit 1; }
    say "SUCCESS: durable archive $GCS_RUN"
    if ! gcloud storage cp "$RUN_DIR/orchestrator.log" "$GCS_RUN/orchestrator.log" \
        >> "$ARCHIVE_LOG_TMP" 2>&1 ||
        ! gcloud storage cp "$RUN_DIR/archive.log" "$GCS_RUN/archive.log" \
          >> "$ARCHIVE_LOG_TMP" 2>&1; then
      say "ABORT: final archive confirmation failed"
      exit 1
    fi
    cp "$ARCHIVE_LOG_TMP" "$RUN_DIR/archive.log" || {
      say "ABORT: could not finalize archive log"; exit 1; }
    rm -f "$ARCHIVE_LOG_TMP"
    exit 0
  fi

  if grep -aqE "$REFUSAL_RE" "$LOG"; then
    say "try $try: draw REFUSED — redraw"
  else
    say "try $try: incomplete driver=$(grep -c 'DRIVER_EXIT=0' "$LOG" || true) manifest=$(grep -c 'manifest VERIFIED' "$LOG" || true) scp=$SCP_OK/8 parse=$PARSE_OK"
  fi
done
say "FAILED in 6 tries"
exit 1
