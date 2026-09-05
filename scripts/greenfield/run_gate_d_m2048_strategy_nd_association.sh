#!/usr/bin/bash
# Protected eight-host exact-M2048 StrategyND association discriminator.
# Invoke only through the reviewed root-owned launcher.
set -euo pipefail

read -r -d '' RUNTIME_BOUNDARY_VERIFIER <<'PY' || true
import fcntl
import hashlib
import os
import stat
from pathlib import Path

root = Path("/opt/glm-tpu/locks")
launcher = Path("/opt/glm-tpu/bin/launch_gate_d_m2048_strategy_nd_v4.py")
parent_fd = os.open(root, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW)
parent = os.fstat(parent_fd)
if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != 0 or parent.st_gid != 0 or stat.S_IMODE(parent.st_mode) & 0o022:
    raise SystemExit("unsafe immutable lock parent")
for descriptor, name in zip((11, 12), ("glm_pod_workload.lock", "glm_tpu_rsync.lock"), strict=True):
    held = os.fstat(descriptor)
    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(held.st_mode)
        or held.st_nlink != 1
        or held.st_uid != 0
        or held.st_gid != 0
        or stat.S_IMODE(held.st_mode) != 0o666
        or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)
    ):
        raise SystemExit("unsafe immutable lock identity: " + name)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
os.close(parent_fd)

run_root = Path("/home/gianl/gate-d-runs")
tag = os.environ["GLM_GATE_D_M2048_TAG"]
run_root_fd = os.open(
    run_root, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
)
try:
    held = os.fstat(7)
    named = os.stat(tag, dir_fd=run_root_fd, follow_symlinks=False)
    if (
        not stat.S_ISDIR(held.st_mode)
        or stat.S_IMODE(held.st_mode) != 0o700
        or held.st_uid != os.geteuid()
        or held.st_gid != os.getegid()
        or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)
    ):
        raise SystemExit("unsafe inherited run-directory descriptor")
    fcntl.flock(7, fcntl.LOCK_EX | fcntl.LOCK_NB)
finally:
    os.close(run_root_fd)

wrapper_fd = int(os.environ["GLM_GATE_D_WRAPPER_MEMFD"])
if wrapper_fd != 10:
    raise SystemExit("unexpected protected wrapper descriptor")
wrapper = os.fstat(wrapper_fd)
required_seals = 0x0001 | 0x0002 | 0x0004 | 0x0008
if (
    not stat.S_ISREG(wrapper.st_mode)
    or wrapper.st_nlink != 0
    or fcntl.fcntl(wrapper_fd, getattr(fcntl, "F_GET_SEALS", 1034)) != required_seals
):
    raise SystemExit("protected wrapper descriptor is not a sealed memfd")
raw = bytearray()
offset = 0
while block := os.pread(wrapper_fd, 1024 * 1024, offset):
    raw.extend(block)
    offset += len(block)
if len(raw) != wrapper.st_size or hashlib.sha256(raw).hexdigest() != os.environ["GLM_GATE_D_WRAPPER_SHA256"]:
    raise SystemExit("protected wrapper descriptor hash drifted")

launcher_parent = os.stat(launcher.parent, follow_symlinks=False)
launcher_fd = os.open(launcher, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
try:
    before = os.fstat(launcher_fd)
    named = os.stat(launcher, follow_symlinks=False)
    if (
        not stat.S_ISDIR(launcher_parent.st_mode)
        or launcher_parent.st_uid != 0
        or launcher_parent.st_gid != 0
        or stat.S_IMODE(launcher_parent.st_mode) & 0o022
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != 0
        or before.st_gid != 0
        or stat.S_IMODE(before.st_mode) != 0o555
        or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino)
    ):
        raise SystemExit("unsafe root-owned protected launcher")
    launcher_raw = bytearray()
    while block := os.read(launcher_fd, 1024 * 1024):
        launcher_raw.extend(block)
    after = os.fstat(launcher_fd)
    if (
        len(launcher_raw) != before.st_size
        or (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        or hashlib.sha256(launcher_raw).hexdigest() != os.environ["GLM_GATE_D_LAUNCHER_SHA256"]
    ):
        raise SystemExit("root-owned protected launcher changed")
finally:
    os.close(launcher_fd)
PY

[[ ${GLM_GATE_D_M2048_WRAPPER_SANITIZED:-0} == 1 ]] || {
  echo "invoke only through the reviewed root-owned launcher" >&2
  exit 2
}
[[ ${HOME:-} == /home/gianl ]]
[[ ${LANG:-} == C && ${LC_ALL:-} == C ]]
[[ ${PATH:-} == /snap/bin:/usr/bin:/bin ]]
[[ ${PYTHONDONTWRITEBYTECODE:-} == 1 ]]
while IFS='=' read -r environment_name _; do
  case "$environment_name" in
    GLM_GATE_D_IMMUTABLE_LOCKS_HELD | GLM_GATE_D_LAUNCHER_PIN | \
    GLM_GATE_D_LAUNCHER_SHA256 | GLM_GATE_D_M2048 | \
    GLM_GATE_D_M2048_MODE | GLM_GATE_D_M2048_TAG | \
    GLM_GATE_D_M2048_WRAPPER_SANITIZED | GLM_GATE_D_WRAPPER_MEMFD | \
    GLM_GATE_D_WRAPPER_SHA256 | HOME | LANG | LC_ALL | PATH | PWD | \
    PYTHONDONTWRITEBYTECODE | SHLVL | _) ;;
    *) echo "unexpected wrapper environment name: $environment_name" >&2; exit 2 ;;
  esac
done < <(/usr/bin/env)

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly BUCKET=gs://driftbench-dsv4-uc
readonly LOCATION=US-CENTRAL2
readonly CAPSULE=/usr/local/libexec/glm-tpu/gate-d-m2048-strategy-nd-v4
readonly PROBE=$CAPSULE/probe_m2048_strategy_nd_association.py
readonly PROBE_SHA=f708163ab9731cfa8195c78d0bb4b458e3a44c1858faecab331716427c70c39b
readonly PUBLISHER=$CAPSULE/publish_gate_d_m2048_strategy_nd_association.py
readonly PUBLISHER_SHA=fbf02bd4216b20e21855d877837c972db98736c636608bf74262f20126e51148
readonly MIRROR_VERIFIER=$CAPSULE/verify_gate_d_rewrite_same_region_git_mirror.py
readonly MIRROR_VERIFIER_SHA=8747647d88a4cf2d435a46b6ed94067ecc05910d1917c24122cba88f8d037c0a
readonly SEALED_PYTHON=/opt/glm-tpu/gate-d-python-3.12.13-021044895e95/bin/python3.12
readonly SEALED_PYTHON_SHA=021044895e95be79dc2f110367607e684119afbc8ce75f6f0eec94844e0acec7
readonly VACANCY_EXPECTED='ERROR: (gcloud.storage.ls) One or more URLs matched no objects.'
readonly SOFT_DELETE_DISABLED_EXPECTED='ERROR: (gcloud.storage.ls) HTTPError 400: Soft delete policy is required to list soft-deleted versions'

[[ ${GLM_GATE_D_IMMUTABLE_LOCKS_HELD:-0} == 1 ]]
/usr/bin/python3 -I -S -B -c "$RUNTIME_BOUNDARY_VERIFIER"
exec 10<&-
[[ ${GLM_GATE_D_M2048:-0} == 1 ]] || {
  echo "Gate-D M2048 fingerprint is default-off" >&2; exit 2;
}
[[ ${GLM_GATE_D_M2048_MODE:-off} == execute_once ]] || {
  echo "set GLM_GATE_D_M2048_MODE=execute_once" >&2; exit 2;
}

git_local() {
  /usr/bin/env -i GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 \
    GIT_NO_LAZY_FETCH=1 GIT_NO_REPLACE_OBJECTS=1 GIT_OPTIONAL_LOCKS=0 \
    GIT_PROTOCOL_FROM_USER=0 GIT_SSH_COMMAND=/bin/false GIT_TERMINAL_PROMPT=0 \
    HOME=/nonexistent LANG=C LC_ALL=C PATH=/usr/bin:/bin \
    /usr/bin/git -c core.fsmonitor=false -c core.untrackedCache=false \
      -c core.hooksPath=/dev/null -c diff.external= \
      -c core.attributesFile=/dev/null -C "$WORKTREE" "$@"
}
git_remote() {
  /usr/bin/env -i GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 \
    GIT_NO_LAZY_FETCH=1 GIT_NO_REPLACE_OBJECTS=1 GIT_OPTIONAL_LOCKS=0 \
    GIT_TERMINAL_PROMPT=0 HOME=/home/gianl LANG=C LC_ALL=C PATH=/usr/bin:/bin \
    /usr/bin/git -c core.fsmonitor=false \
      -c 'core.sshCommand=/usr/bin/ssh -oBatchMode=yes -oClearAllForwardings=yes -oForwardAgent=no' "$@"
}

readonly PIN=$(git_local rev-parse HEAD)
[[ $PIN == "${GLM_GATE_D_LAUNCHER_PIN:-}" && $PIN =~ ^[0-9a-f]{40}$ ]]
[[ $(git_local rev-parse --show-toplevel) == "$WORKTREE" ]]
[[ $(git_local branch --show-current) == "$BRANCH" ]]
[[ $(git_local remote get-url origin) == "$ORIGIN" ]]
[[ -z $(git_local status --porcelain=v1 --untracked-files=all) ]]
[[ -z $(git_local for-each-ref --format='%(refname)' refs/replace) ]]
readonly ORIGIN_TIP=$(git_remote ls-remote --exit-code "$ORIGIN" "refs/heads/$BRANCH" | /usr/bin/awk '{print $1}')
[[ $ORIGIN_TIP == "$PIN" ]] || { echo "origin pin drifted" >&2; exit 2; }

readonly TAG=${GLM_GATE_D_M2048_TAG:-}
[[ $TAG =~ ^greenfield_m2048_strategy_nd_[0-9]{8}T[0-9]{15}Z$ ]] || {
  echo "unsafe Gate-D M2048 tag: $TAG" >&2; exit 2;
}
cd /
readonly RUN_ROOT=/home/gianl/gate-d-runs
readonly RUN_DIR=$RUN_ROOT/$TAG
readonly REMOTE_PREFIX=$BUCKET/results/greenfield/glm52/m2048_strategy_nd/$TAG

for binding in \
  "$PROBE:$PROBE_SHA" \
  "$PUBLISHER:$PUBLISHER_SHA" \
  "$MIRROR_VERIFIER:$MIRROR_VERIFIER_SHA" \
  "$SEALED_PYTHON:$SEALED_PYTHON_SHA"; do
  path=${binding%:*}; digest=${binding##*:}
  [[ $(/usr/bin/sha256sum "$path" | /usr/bin/awk '{print $1}') == "$digest" ]] || {
    echo "bound runtime digest drifted: $path" >&2; exit 2;
  }
done
[[ $(/snap/bin/gcloud storage buckets describe "$BUCKET" --format='value(location)') == "$LOCATION" ]]
readonly SOFT_DELETE_RETENTION_SECONDS=$(/snap/bin/gcloud storage buckets describe \
  "$BUCKET" --format='value(soft_delete_policy.retentionDurationSeconds)')
[[ $SOFT_DELETE_RETENTION_SECONDS == 0 ]]
[[ $(/snap/bin/gcloud compute tpus tpu-vm describe "$POD" --zone "$ZONE" --format='value(state)') == READY ]]

publisher() {
  /usr/bin/env -i HOME=/home/gianl LANG=C LC_ALL=C PATH=/usr/bin:/bin \
    PYTHONDONTWRITEBYTECODE=1 "$SEALED_PYTHON" -I -S -B "$PUBLISHER" \
      --expected-code-hash "$PIN" --expected-source-sha256 "$PUBLISHER_SHA" \
      --run-dir-fd 7 "$@"
}
publish_member() { publisher write --run-dir "$RUN_DIR" --member "$1"; }

vacancy_zero_retention_surfaces() {
  local live versions deleted live_rc versions_rc deleted_rc
  set +e
  live=$(PYTHONWARNINGS=ignore /usr/bin/timeout --signal=TERM --kill-after=10 60 \
    /snap/bin/gcloud storage ls "$REMOTE_PREFIX/**" 2>&1); live_rc=$?
  versions=$(PYTHONWARNINGS=ignore /usr/bin/timeout --signal=TERM --kill-after=10 60 \
    /snap/bin/gcloud storage ls --all-versions "$REMOTE_PREFIX/**" 2>&1); versions_rc=$?
  deleted=$(PYTHONWARNINGS=ignore /usr/bin/timeout --signal=TERM --kill-after=10 60 \
    /snap/bin/gcloud storage ls --soft-deleted --exhaustive "$REMOTE_PREFIX/**" 2>&1); deleted_rc=$?
  set -e
  [[ $SOFT_DELETE_RETENTION_SECONDS == 0 && \
     $live_rc -eq 1 && $live == "$VACANCY_EXPECTED" && \
     $versions_rc -eq 1 && $versions == "$VACANCY_EXPECTED" && \
     $deleted_rc -eq 1 && $deleted == "$SOFT_DELETE_DISABLED_EXPECTED" ]] || return 1
  /usr/bin/printf '%s\n' \
    'soft_delete_retention_seconds=0' \
    'scope=live flags=none returncode=1' "$live" \
    'scope=all_versions flags=--all-versions returncode=1' "$versions" \
    'scope=soft_deleted flags=--soft-deleted,--exhaustive returncode=1 availability=policy_disabled' "$deleted"
}
readonly VACANCY_RAW=$(vacancy_zero_retention_surfaces) || {
  echo "ABORT: zero-retention remote-prefix preflight is not canonical" >&2; exit 2;
}
readonly VACANCY_SUMMARY=$(/usr/bin/printf '%s\n' \
  'SOFT_DELETE_RETENTION_SECONDS 0' \
  "VACANT live $REMOTE_PREFIX" \
  "VACANT all_versions $REMOTE_PREFIX" \
  "UNAVAILABLE_POLICY_DISABLED soft_deleted $REMOTE_PREFIX")

run_identity=$(publisher init --run-dir "$RUN_DIR")
[[ $run_identity =~ ^RUN_IDENTITY\ ([0-9]+:[0-9]+)$ ]]
readonly EXPECTED_RUN_IDENTITY=${BASH_REMATCH[1]}
readonly OBSERVED_RUN_IDENTITY=$(/usr/bin/stat -Lc '%d:%i' /proc/self/fd/7)
[[ $OBSERVED_RUN_IDENTITY == "$EXPECTED_RUN_IDENTITY" ]]
/usr/bin/flock -n 7 || { echo "run-directory supervisor lock unavailable" >&2; exit 1; }

say() {
  local line="[m2048 $(/usr/bin/date -u +%H:%M:%S)] $*"
  /usr/bin/printf '%s\n' "$line"
  /usr/bin/printf '%s\n' "$line" | publisher append-log --run-dir "$RUN_DIR"
}
has_eight_unique_markers() {
  local file=$1 marker=$2
  [[ $(/usr/bin/awk -v marker="$marker" '$1 == marker {print $2}' "$file" | /usr/bin/wc -l) -eq 8 ]] &&
    [[ $(/usr/bin/awk -v marker="$marker" '$1 == marker {print $2}' "$file" | /usr/bin/sort -u | /usr/bin/wc -l) -eq 8 ]]
}
strict_census() {
  local label=$1 member="census_${1}.txt" carrier="${TAG}_${1}" ray_enum command output status
  # shellcheck disable=SC2016
  ray_enum='GLM_CENSUS_CARRIER='"$carrier"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "[p]robe_m2048_strategy_nd_association[.]py|[V]LLM::EngineCore|[R]ayWorkerWrapper|[r]ay start --address|[r]ay start --head" 2>/dev/null || true); holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; if [[ $tools_ok -ne 1 || $ray_rc -ne 0 || $docker_rc -ne 0 ]]; then echo "CENSUS_BAD $(hostname)"; elif [[ -n $ray_pids || -n $generic || -n $holders ]] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; else echo "CENSUS_OK $(hostname)"; fi'
  set +e
  output=$(GLM_CENSUS_CARRIER="$carrier" /usr/bin/timeout --signal=TERM --kill-after=10 180 \
    /snap/bin/gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
      --command="$command" 2>&1); status=$?
  set -e
  /usr/bin/printf '%s\n' "$output" | publish_member "$member" || return 1
  [[ $status -eq 0 ]] || return 1
  has_eight_unique_markers "/proc/self/fd/7/$member" CENSUS_OK
}

post_census_done=0
terminal_written=0
remote_vacant=0
on_exit() {
  local status=$? census_status=0
  if [[ $post_census_done -eq 0 ]]; then strict_census failure_exit || census_status=$?; fi
  if [[ $status -ne 0 && $terminal_written -eq 0 && $remote_vacant -eq 1 ]]; then
    say "FAILED status=$status; publishing bounded diagnostic"
    publisher diagnostic --run-dir "$RUN_DIR" --remote-prefix "$REMOTE_PREFIX" --status "$status" || census_status=1
  fi
  if [[ $census_status -ne 0 ]]; then trap - EXIT; exit 71; fi
  return "$status"
}

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
/usr/bin/flock -n 9 || { echo "secondary pod lease unavailable" >&2; exit 1; }
exec 8>/home/gianl/.glm-tpu-rsync.lock
/usr/bin/flock 8
trap on_exit EXIT

/usr/bin/printf '%s\n' "$VACANCY_RAW" | publish_member remote_vacancy.raw.txt
/usr/bin/printf '%s\n' "$VACANCY_SUMMARY" | publish_member remote_vacancy.txt
remote_vacant=1
say "RUN_DIR=$RUN_DIR PIN=$PIN origin_tip=$ORIGIN_TIP devices=0..31"
say "replaying pushed pin and locked US-CENTRAL2 Git mirror closure"
set +e
/usr/bin/env -i HOME=/home/gianl LANG=C LC_ALL=C PATH=/usr/bin:/bin \
  PYTHONDONTWRITEBYTECODE=1 "$SEALED_PYTHON" -I -S -B "$MIRROR_VERIFIER" \
    --expected-code-hash "$PIN" --expected-source-sha256 "$MIRROR_VERIFIER_SHA" | \
  publish_member mirror.sha256
mirror_status=("${PIPESTATUS[@]}")
set -e
[[ ${mirror_status[0]} -eq 0 && ${mirror_status[1]} -eq 0 ]] || {
  say "ABORT: same-region mirror replay failed"; exit 1;
}

# Each host must already contain the exact pin and immutable capsule/runtime.
# shellcheck disable=SC2016
sync_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; [[ $idx =~ ^[0-7]$ ]]; pin='"$PIN"'; wt='"$WORKTREE"'; capsule='"$CAPSULE"'; git_local() { /usr/bin/env -i GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 GIT_NO_LAZY_FETCH=1 GIT_NO_REPLACE_OBJECTS=1 GIT_OPTIONAL_LOCKS=0 GIT_PROTOCOL_FROM_USER=0 GIT_SSH_COMMAND=/bin/false GIT_TERMINAL_PROMPT=0 HOME=/nonexistent LANG=C LC_ALL=C PATH=/usr/bin:/bin /usr/bin/git -c core.fsmonitor=false -c core.untrackedCache=false -c core.hooksPath=/dev/null -c core.attributesFile=/dev/null -C "$wt" "$@"; }; [[ $(git_local rev-parse HEAD) == "$pin" ]]; [[ -z $(git_local status --porcelain=v1 --untracked-files=all) ]]; [[ -z $(git_local for-each-ref --format="%(refname)" refs/replace) ]]; [[ $(/usr/bin/sha256sum "$capsule/probe_m2048_strategy_nd_association.py" | /usr/bin/awk "{print \$1}") == '"$PROBE_SHA"' ]]; [[ $(/usr/bin/sha256sum "$capsule/publish_gate_d_m2048_strategy_nd_association.py" | /usr/bin/awk "{print \$1}") == '"$PUBLISHER_SHA"' ]]; [[ -x '"$SEALED_PYTHON"' ]]; echo "SYNC_OK $(hostname) $pin"'
set +e
/snap/bin/gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$sync_command" 2>&1 | publish_member sync.txt
sync_status=("${PIPESTATUS[@]}")
set -e
[[ ${sync_status[0]} -eq 0 && ${sync_status[1]} -eq 0 ]] || {
  say "ABORT: exact eight-host prerequisite check failed"; exit 1;
}
has_eight_unique_markers /proc/self/fd/7/sync.txt SYNC_OK || {
  say "ABORT: prerequisite check did not authenticate eight hosts"; exit 1;
}
strict_census pre || { say "ABORT: pre-run census is not authenticated 8/8 zero work"; exit 1; }

coordinator=$(/snap/bin/gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=0 \
  --command="hostname -I" 2>/dev/null | /usr/bin/grep -Eo '192\.168\.[0-9]+\.[0-9]+' | /usr/bin/head -1)
[[ -n $coordinator ]] || { say "ABORT: worker-0 coordinator address missing"; exit 1; }
coordinator="$coordinator:8476"
say "executing one exact-M2048 full-pod fingerprint coordinator=$coordinator"
started=$(/usr/bin/date +%s)
# shellcheck disable=SC2016
execute_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; [[ $idx =~ ^[0-7]$ ]]; /usr/bin/env -i HOME=/home/gianl JAX_ENABLE_COMPILATION_CACHE=0 JAX_PLATFORMS=tpu LANG=C LC_ALL=C PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=.20 GLM_GATE_D_M2048_TAG='"$TAG"' /usr/bin/timeout --signal=TERM --kill-after=60 1800 '"$SEALED_PYTHON"' -I -S -B -u '"$PROBE"' --code-pin '"$PIN"' --expected-source-sha256 '"$PROBE_SHA"' --run-tag '"$TAG"' --coordinator-address '"$coordinator"' --num-processes 8 --process-id "$idx"'
set +e
/snap/bin/gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$execute_command" 2>&1 | publish_member distributed.raw.log
probe_status=("${PIPESTATUS[@]}")
set -e
elapsed=$(( $(/usr/bin/date +%s) - started ))
/usr/bin/printf 'gcloud_status=%s publisher_status=%s elapsed_seconds=%s\n' \
  "${probe_status[0]}" "${probe_status[1]}" "$elapsed" | publish_member runner.log
[[ ${probe_status[0]} -eq 0 && ${probe_status[1]} -eq 0 ]] || {
  say "ABORT: distributed probe or raw-log retention failed"; exit 1;
}
has_eight_unique_markers /proc/self/fd/7/distributed.raw.log M2048_OK || {
  say "ABORT: distributed probe did not authenticate eight completions"; exit 1;
}
say "probe exited successfully in ${elapsed}s"
strict_census post || { say "ABORT: post-run census is not authenticated 8/8 zero work"; exit 1; }
post_census_done=1

say "publishing generation-bound archive after a fresh zero-retention vacancy check"
result_authority=$(publisher success --run-dir "$RUN_DIR" \
  --remote-prefix "$REMOTE_PREFIX" --elapsed "$elapsed")
if [[ $result_authority =~ ^M2048_RESULT\ status=M2048_ASSOCIATION_UNIQUE\ marker_sha256=([0-9a-f]{64})\ terminal_generation=([0-9]+)\ terminal_sha256=([0-9a-f]{64})$ ]]; then
  terminal_written=1
else
  echo "Gate-D M2048 publisher authority drifted" >&2; exit 2
fi
trap - EXIT
/usr/bin/printf '%s\n' "$result_authority" "M2048_ASSOCIATION_UNIQUE gate_d_open=true"
