#!/usr/bin/bash
# Protected one-start optimized-HLO acquisition for the admitted Gate-D candidate.
# Security boundary: invoke only through the separately reviewed literal
# `/usr/bin/env -i ... /usr/bin/bash --noprofile --norc <this-file>` command.
# The checks below are fail-fast validation after Bash startup; they are not a
# substitute for that pre-start environment/interpreter boundary.
set -euo pipefail

readonly WRAPPER_ABS=/home/gianl/glm-tpu-topology-rewrite/scripts/greenfield/run_gate_d_compensated_pp16_hlo.sh
[[ ${GLM_GATE_D_WRAPPER_SANITIZED:-0} == 1 ]] || {
  echo "invoke only through the reviewed absolute env-i/bash entry command" >&2
  exit 2
}
[[ ${HOME:-} == /home/gianl ]]
[[ ${LANG:-} == C && ${LC_ALL:-} == C ]]
[[ ${PATH:-} == /snap/bin:/usr/bin:/bin:/home/gianl/vllm-env/bin ]]
[[ ${PYTHONDONTWRITEBYTECODE:-} == 1 ]]
while IFS='=' read -r environment_name _; do
  case "$environment_name" in
    GLM_GATE_D_COMPENSATED_PP16_HLO_ACQUIRE | GLM_GATE_D_COMPENSATED_PP16_MODE | \
      GLM_GATE_D_COMPENSATED_PP16_TAG | GLM_GATE_D_WRAPPER_SANITIZED | HOME | LANG | \
      LC_ALL | PATH | PWD | PYTHONDONTWRITEBYTECODE | SHLVL | _) ;;
    *)
      echo "unexpected wrapper environment name: $environment_name" >&2
      exit 2
      ;;
  esac
done < <(/usr/bin/env)
cd /

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly BUCKET=gs://driftbench-dsv4-uc
readonly LOCATION=US-CENTRAL2
readonly ADMISSION=$WORKTREE/docs/artifacts/gate-d-precompile-admission-v2-compensated-capsule.json
readonly ADMISSION_SHA=7cd7e569ed9ed5fd978d933efd4229d65906ae28312e264863b8336e4cc6b37d
readonly TOPOLOGY=$WORKTREE/docs/artifacts/gate-d-runtime-locality-authority.json
readonly TOPOLOGY_SHA=49cf6bb1a553985855556d1401ad85918669df52d12f8dc5150f247d18b325eb
readonly SEALED_CPU_REPLAY=$WORKTREE/glm_tpu/greenfield/benchmarking/gate_d_compensated_capsule.py
readonly SEALED_CPU_REPLAY_SHA=0f1930c079bd7244452e84dca6d0bbf9da077ea133c685f0f908760379e5313d
readonly TPU_REPLAY=$WORKTREE/glm_tpu/greenfield/benchmarking/gate_d_compensated_pp16_hlo.py
readonly TPU_REPLAY_SHA=a75b6eeb21ca94cd70910553f5804b8f1127ac88ca3c8b5545bec993adae54fc
readonly DRIVER=$WORKTREE/scripts/greenfield/acquire_gate_d_compensated_pp16_hlo.py
readonly DRIVER_SHA=526aad55d1e14b46b8c72a3061469913edb39ff3d29dda242515aaa33b4748e4
readonly PUBLISHER=$WORKTREE/scripts/greenfield/publish_gate_d_compensated_pp16_hlo.py
readonly PUBLISHER_SHA=2d83ab19a5abcd0e8df6df46ef5002c482e68c8822299efcef8ede325f59d874
readonly STORAGE_SITE_BUILDER=$WORKTREE/scripts/greenfield/build_gate_d_storage_site_capsule.py
readonly STORAGE_SITE_BUILDER_SHA=b7f4f869ae9b98edf5195185bf49fffcf61ef6800a2b707127694b66424c5989
readonly PUBLISHER_PYTHON=/opt/glm-tpu/gate-d-python-3.12.13-021044895e95/bin/python3.12
readonly DRIVER_PYTHON=/home/gianl/vllm-env/bin/python

[[ ${GLM_GATE_D_COMPENSATED_PP16_HLO_ACQUIRE:-0} == 1 ]] || {
  echo "Gate-D compensated PP16 HLO acquisition is default-off" >&2
  exit 2
}
[[ ${GLM_GATE_D_COMPENSATED_PP16_MODE:-off} == compile_only ]] || {
  echo "set GLM_GATE_D_COMPENSATED_PP16_MODE=compile_only" >&2
  exit 2
}

readonly PIN=$(git -C "$WORKTREE" rev-parse HEAD)
readonly TAG=${GLM_GATE_D_COMPENSATED_PP16_TAG:-}
[[ $TAG =~ ^gate_d_compensated_pp16_hlo_[0-9]{8}T[0-9]{15}Z$ ]] || {
  echo "unsafe Gate-D PP16 HLO tag: $TAG" >&2
  exit 2
}
readonly RUN_DIR=/home/gianl/gate-d-runs/$TAG
readonly REMOTE_PREFIX=$BUCKET/results/greenfield/glm52/gate_d_pp16_hlo/$TAG

[[ $(git -C "$WORKTREE" rev-parse --show-toplevel) == "$WORKTREE" ]]
[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]]
[[ -z $(git -C "$WORKTREE" status --porcelain) ]]
[[ $(git -C "$WORKTREE" ls-remote origin "refs/heads/$BRANCH" | awk '{print $1}') == "$PIN" ]]
[[ $(sha256sum "$ADMISSION" | awk '{print $1}') == "$ADMISSION_SHA" ]]
[[ $(sha256sum "$TOPOLOGY" | awk '{print $1}') == "$TOPOLOGY_SHA" ]]
[[ $(sha256sum "$SEALED_CPU_REPLAY" | awk '{print $1}') == "$SEALED_CPU_REPLAY_SHA" ]]
[[ $(sha256sum "$TPU_REPLAY" | awk '{print $1}') == "$TPU_REPLAY_SHA" ]]
[[ $(sha256sum "$DRIVER" | awk '{print $1}') == "$DRIVER_SHA" ]]
[[ $(sha256sum "$PUBLISHER" | awk '{print $1}') == "$PUBLISHER_SHA" ]]
[[ $(sha256sum "$STORAGE_SITE_BUILDER" | awk '{print $1}') == "$STORAGE_SITE_BUILDER_SHA" ]]
[[ $(gcloud storage buckets describe "$BUCKET" --format='value(location)') == "$LOCATION" ]]
[[ $(gcloud compute tpus tpu-vm describe "$POD" --zone "$ZONE" --format='value(state)') == READY ]]

publisher() {
  /usr/bin/env -i \
    HOME=/home/gianl \
    LANG=C \
    LC_ALL=C \
    PATH=/usr/bin:/bin \
    PYTHONDONTWRITEBYTECODE=1 \
    "$PUBLISHER_PYTHON" -I -S -B "$PUBLISHER" \
      --expected-code-hash "$PIN" \
      --expected-source-sha256 "$PUBLISHER_SHA" \
      --run-dir-fd 7 "$@"
}

publisher_init() {
  /usr/bin/env -i \
    HOME=/home/gianl \
    LANG=C \
    LC_ALL=C \
    PATH=/usr/bin:/bin \
    PYTHONDONTWRITEBYTECODE=1 \
    "$PUBLISHER_PYTHON" -I -S -B "$PUBLISHER" \
      --expected-code-hash "$PIN" \
      --expected-source-sha256 "$PUBLISHER_SHA" "$@"
}

publish_member() {
  local member=$1
  publisher write --run-dir "$RUN_DIR" --member "$member"
}

run_identity=$(publisher_init init --run-dir "$RUN_DIR")
[[ $run_identity =~ ^RUN_IDENTITY\ ([0-9]+:[0-9]+)$ ]]
readonly EXPECTED_RUN_IDENTITY=${BASH_REMATCH[1]}
exec 7<"$RUN_DIR"
readonly OBSERVED_RUN_IDENTITY=$(/usr/bin/stat -Lc '%d:%i' /proc/self/fd/7)
[[ $OBSERVED_RUN_IDENTITY == "$EXPECTED_RUN_IDENTITY" ]]
/usr/bin/flock -n 7 || {
  echo "Gate-D run-directory supervisor lock is unavailable" >&2
  exit 1
}

say() {
  local line="[gate-d-pp16-hlo $(date -u +%H:%M:%S)] $*"
  printf '%s\n' "$line"
  printf '%s\n' "$line" | publisher append-log --run-dir "$RUN_DIR"
}

has_eight_unique_markers() {
  local file=$1 marker=$2
  [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | wc -l) -eq 8 ]] &&
    [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | sort -u | wc -l) -eq 8 ]]
}

strict_census() {
  local label=$1 member="census_${label}.txt"
  local carrier="${TAG}_${label}" ray_enum command output command_status
  # shellcheck disable=SC2016
  ray_enum='GLM_CENSUS_CARRIER='"$carrier"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "[a]cquire_gate_d_compensated_pp16_hlo[.]py|[V]LLM::EngineCore|[R]ayWorkerWrapper|[r]ay start --address|[r]ay start --head" 2>/dev/null || true); holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; if [[ $tools_ok -ne 1 || $ray_rc -ne 0 || $docker_rc -ne 0 ]]; then echo "CENSUS_BAD $(hostname)"; elif [[ -n $ray_pids || -n $generic || -n $holders ]] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; else echo "CENSUS_OK $(hostname)"; fi'
  set +e
  output=$(GLM_CENSUS_CARRIER="$carrier" timeout --signal=TERM --kill-after=10 180 \
    gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
      --command="$command" 2>&1)
  command_status=$?
  set -e
  printf '%s\n' "$output" | publish_member "$member" || return 1
  [[ $command_status -eq 0 ]] || return 1
  has_eight_unique_markers "/proc/self/fd/7/$member" CENSUS_OK
}

publish_diagnostic() {
  local status=$1
  publisher diagnostic --run-dir "$RUN_DIR" --remote-prefix "$REMOTE_PREFIX" \
      --status "$status"
}

post_census_done=0
terminal_written=0
remote_vacant=0
on_exit() {
  local status=$? census_status=0
  if [[ $post_census_done -eq 0 ]]; then
    strict_census failure_exit || census_status=$?
  fi
  if [[ $status -ne 0 && $terminal_written -eq 0 && $remote_vacant -eq 1 && \
        ! -e /proc/self/fd/7/HLO_ACQUIRED && ! -L /proc/self/fd/7/HLO_ACQUIRED ]]; then
    say "FAILED status=$status; publishing bounded diagnostic without HLO_ACQUIRED"
    publish_diagnostic "$status" || census_status=1
  fi
  if [[ $census_status -ne 0 ]]; then
    trap - EXIT
    exit 71
  fi
  return "$status"
}

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  say "ABORT: another protected TPU workflow holds the global lease"
  exit 1
}
exec 8>/home/gianl/.glm-tpu-rsync.lock
flock 8
trap on_exit EXIT

say "RUN_DIR=$RUN_DIR PIN=$PIN mode=compile_only devices=0,1"
set +e
vacancy_output=$(timeout --signal=TERM --kill-after=10 60 \
  gcloud storage ls "$REMOTE_PREFIX/**" 2>&1)
vacancy_rc=$?
set -e
printf '%s\n' "$vacancy_output" | publish_member remote_vacancy.raw.txt
if [[ $vacancy_rc -eq 0 || $vacancy_rc -ne 1 ]] ||
   ! grep -q 'matched no objects' <<<"$vacancy_output"; then
  say "ABORT: append-only remote-prefix vacancy check failed"
  exit 2
fi
printf 'VACANT %s\n' "$REMOTE_PREFIX" | publish_member remote_vacancy.txt
remote_vacant=1

say "verifying current source bytes in the locked US-CENTRAL2 repository mirror"
mirror_records=
for relative in \
  docs/artifacts/gate-d-precompile-admission-v2-compensated-capsule.json \
  docs/artifacts/gate-d-runtime-locality-authority.json \
  glm_tpu/greenfield/benchmarking/gate_d_compensated_capsule.py \
  glm_tpu/greenfield/benchmarking/gate_d_compensated_pp16_hlo.py \
  scripts/greenfield/acquire_gate_d_compensated_pp16_hlo.py \
  scripts/greenfield/build_gate_d_storage_site_capsule.py \
  scripts/greenfield/publish_gate_d_compensated_pp16_hlo.py \
  scripts/greenfield/run_gate_d_compensated_pp16_hlo.sh; do
  local_sha=$(sha256sum "$WORKTREE/$relative" | awk '{print $1}')
  remote_sha=$(gcloud storage cat "$BUCKET/repos/glm-tpu-topology-rewrite/$relative" 2>/dev/null |
    sha256sum | awk '{print $1}')
  [[ $local_sha == "$remote_sha" ]]
  mirror_records+="$local_sha  $relative"$'\n'
done
printf '%s' "$mirror_records" | publish_member mirror.sha256

strict_census pre || {
  say "ABORT: pre-run census is not authenticated 8/8 zero work"
  exit 1
}

say "synchronizing exact pushed pin across all eight hosts"
# shellcheck disable=SC2016
sync_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; pin='"$PIN"'; branch='"$BRANCH"'; origin='"$ORIGIN"'; wt='"$WORKTREE"'; if [[ $idx == 0 ]]; then [[ -e "$wt/.git" && $(git -C "$wt" rev-parse HEAD) == "$pin" && -z $(git -C "$wt" status --porcelain) ]]; else if [[ -e "$wt/.git" ]]; then [[ -z $(git -C "$wt" status --porcelain) ]]; git -C "$wt" fetch -q "$origin" "$branch"; git -C "$wt" checkout -q --detach "$pin"; else git clone -q --filter=blob:none --no-checkout --single-branch --branch "$branch" "$origin" "$wt"; git -C "$wt" checkout -q --detach "$pin"; fi; fi; [[ $(git -C "$wt" rev-parse HEAD) == "$pin" && -z $(git -C "$wt" status --porcelain) ]]; echo "SYNC_OK $(hostname) $pin"'
set +e
sync_output=$(timeout --signal=TERM --kill-after=10 300 \
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$sync_command" 2>&1)
sync_status=$?
set -e
printf '%s\n' "$sync_output" | publish_member sync.txt
[[ $sync_status -eq 0 ]] || {
  say "ABORT: exact eight-host code synchronization command failed"
  exit 1
}
has_eight_unique_markers /proc/self/fd/7/sync.txt SYNC_OK || {
  say "ABORT: exact eight-host code synchronization failed"
  exit 1
}

say "lowering and compiling one abstract-input PP16 stage-zero graph; invocation forbidden"
started=$(date +%s)
set +e
(
  cd /
  /usr/bin/env -i \
    HOME=/home/gianl \
    JAX_PLATFORMS=tpu \
    JAX_ENABLE_COMPILATION_CACHE=0 \
    LANG=C \
    LC_ALL=C \
    PATH=/home/gianl/vllm-env/bin:/usr/bin:/bin \
    PYTHONDONTWRITEBYTECODE=1 \
    TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_VISIBLE_DEVICES=0,1,2,3 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=.50 \
    /usr/bin/timeout --signal=TERM --kill-after=60 3600 \
      "$DRIVER_PYTHON" -I -S -B -u "$DRIVER" \
        --expected-code-hash "$PIN" \
        --expected-driver-sha256 "$DRIVER_SHA" \
        --admission-report "$ADMISSION" \
        --admission-report-sha256 "$ADMISSION_SHA" \
        --topology-authority "$TOPOLOGY" \
        --topology-authority-sha256 "$TOPOLOGY_SHA" \
        --compile-only 1 \
        --run-dir "$RUN_DIR" \
        --run-dir-fd 7
) 2>&1 | publish_member runner.log
pipeline_status=("${PIPESTATUS[@]}")
set -e
if [[ ${pipeline_status[0]} -ne 0 || ${pipeline_status[1]} -ne 0 ]]; then
  say "ABORT: compile-only process or append-only log publisher failed"
  exit 1
fi
elapsed=$(( $(date +%s) - started ))
say "compile-only process exited successfully in ${elapsed}s"

strict_census post || {
  say "ABORT: post-run census is not authenticated 8/8 zero work"
  exit 1
}
post_census_done=1

say "publishing generation-bound archive; terminal upload is the final mutation"
publisher success --run-dir "$RUN_DIR" --remote-prefix "$REMOTE_PREFIX" \
    --elapsed "$elapsed"
terminal_written=1
trap - EXIT
printf '%s\n' "HLO_ACQUIRED_UNADJUDICATED archive=$REMOTE_PREFIX execution_count=0"
