#!/usr/bin/env bash
# Protected single-host TPU validation for the private upstream GLM DSA stack.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly REPO=/home/gianl/tpu-inference-glm-baseline
readonly VLLM_REPO=/home/gianl/vllm-build-lkg-d626108b
readonly PYTHON=/home/gianl/vllm-env/bin/python
readonly BASE=5e2c7128bc74a75493f07930f3a749bcb272a3cb
readonly VLLM_PIN=d626108b1841888ec90aced33367149a6bbc7e4b
readonly BUCKET=gs://driftbench-dsv4-uc

usage() {
  echo "usage: $0 EXPECTED_HEAD TAG PYTEST_ARG..." >&2
  exit 2
}

[[ $# -ge 3 ]] || usage
EXPECTED_HEAD=$1
TAG=$2
shift 2
PYTEST_ARGS=("$@")
[[ $EXPECTED_HEAD =~ ^[0-9a-f]{40}$ ]] || usage
[[ $TAG =~ ^upstream_[a-z0-9_]+_[0-9]{8}T[0-9]{6}Z$ ]] || usage

RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$BUCKET/results/$TAG
[[ ! -e $RUN_DIR ]] || {
  echo "refusing to reuse run directory: $RUN_DIR" >&2
  exit 2
}
mkdir -p "$RUN_DIR"

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  echo "another protected TPU workflow holds the pod lock" >&2
  exit 1
}
exec 8>/home/gianl/.glm-tpu-rsync.lock
flock -n 8 || {
  echo "the repository mirror is active; retry outside its window" >&2
  exit 1
}

[[ $(git -C "$REPO" rev-parse HEAD) == "$EXPECTED_HEAD" ]] || {
  echo "current head does not match expected head" >&2
  exit 2
}
[[ $(git -C "$VLLM_REPO" rev-parse HEAD) == "$VLLM_PIN" ]] || {
  echo "vLLM pin drifted" >&2
  exit 2
}
[[ -z $(git -C "$REPO" status --porcelain) ]] || {
  echo "refusing protected validation from a dirty worktree" >&2
  exit 2
}
[[ $(gcloud storage buckets describe "$BUCKET" --format='value(location)') == \
  US-CENTRAL2 ]] || {
  echo "approved bucket is not in exact US-CENTRAL2" >&2
  exit 2
}
[[ -z $(gcloud storage ls "$REMOTE_PREFIX/**" 2>/dev/null || true) ]] || {
  echo "refusing to reuse remote prefix: $REMOTE_PREFIX" >&2
  exit 2
}

has_eight_unique_markers() {
  local file=$1 marker=$2
  [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | wc -l) -eq 8 ]] &&
    [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" |
      sort -u | wc -l) -eq 8 ]]
}

strict_census() {
  local label=$1
  local out="$RUN_DIR/census_${label}.txt"
  local carrier="${TAG}_${label}"
  local ray_enum command
  # shellcheck disable=SC2016
  ray_enum='GLM_CENSUS_CARRIER='"$carrier"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'
  # shellcheck disable=SC2016
  command='tools=1; command -v pgrep >/dev/null || tools=0; command -v fuser >/dev/null || tools=0; sudo -n true >/dev/null 2>&1 || tools=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[d]sa_throughput[.]py|[p]ytest" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname)"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -z "$ray_pids" ] || echo "ray: $ray_pids"; [ -z "$generic" ] || echo "$generic"; [ -z "$holders" ] || echo "libtpu: $holders"; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" gcloud compute tpus tpu-vm ssh "$POD" \
    --zone "$ZONE" --worker=all --command="$command" >"$out" 2>&1
  has_eight_unique_markers "$out" CENSUS_OK
}

strict_census pre
{
  echo "TPU_INFERENCE_BASE $BASE"
  echo "TPU_INFERENCE_HEAD $EXPECTED_HEAD"
  echo "WORKTREE_SNAPSHOT_SHA $(git -C "$REPO" diff --binary HEAD | sha256sum | cut -d' ' -f1)"
  echo "VLLM_PIN $VLLM_PIN"
  printf 'PYTEST_ARGS'
  printf ' %q' "${PYTEST_ARGS[@]}"
  printf '\n'
  echo "JAX_PLATFORMS tpu"
  echo "TPU_CHIPS_PER_PROCESS_BOUNDS 2,2,1"
  echo "TPU_PROCESS_BOUNDS 1,1,1"
  echo "TPU_VISIBLE_DEVICES 0,1,2,3"
  echo "BUCKET_LOCATION US-CENTRAL2"
  echo "STARTED_AT_UTC $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$RUN_DIR/provenance.txt"

set +e
(
  cd "$REPO"
  env JAX_PLATFORMS=tpu \
    TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_VISIBLE_DEVICES=0,1,2,3 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$REPO:$VLLM_REPO" \
    "$PYTHON" -m pytest -q -s "${PYTEST_ARGS[@]}"
) >"$RUN_DIR/pytest.txt" 2>&1
pytest_rc=$?
set -e
echo "$pytest_rc" >"$RUN_DIR/pytest.exit_code"

census_rc=0
strict_census post || census_rc=$?
{
  echo "FINISHED_AT_UTC $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "PYTEST_EXIT_CODE $pytest_rc"
  echo "POST_CENSUS_EXIT_CODE $census_rc"
} >>"$RUN_DIR/provenance.txt"

(
  cd "$RUN_DIR"
  sha256sum census_pre.txt census_post.txt provenance.txt pytest.txt \
    pytest.exit_code >evidence.sha256
)
gcloud storage cp --no-clobber "$RUN_DIR"/*.txt "$RUN_DIR/evidence.sha256" \
  "$REMOTE_PREFIX/"
gcloud storage cat "$REMOTE_PREFIX/evidence.sha256" |
  cmp - "$RUN_DIR/evidence.sha256"
sha256sum "$RUN_DIR/evidence.sha256" | tee "$RUN_DIR/manifest_list.sha256"
gcloud storage cp --no-clobber "$RUN_DIR/manifest_list.sha256" "$REMOTE_PREFIX/"

[[ $pytest_rc -eq 0 && $census_rc -eq 0 ]] || exit 1
