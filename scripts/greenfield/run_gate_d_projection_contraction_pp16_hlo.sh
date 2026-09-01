#!/usr/bin/bash
# Protected one-start optimized-HLO acquisition for the admitted Gate-D candidate.
# Invoke only through the root-owned descriptor-bound launcher.
set -euo pipefail

read -r -d '' RUNTIME_BOUNDARY_VERIFIER <<'RUNTIME_BOUNDARY_VERIFIER_EOF' || true
import fcntl
import hashlib
import os
import stat
from pathlib import Path

root = Path("/opt/glm-tpu/locks")
names = ("glm_pod_workload.lock", "glm_tpu_rsync.lock")
launcher = Path("/opt/glm-tpu/bin/launch_gate_d_projection_contraction_pp16_hlo_v1.py")
parent_fd = os.open(
    root, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
)
parent = os.fstat(parent_fd)
if (
    not stat.S_ISDIR(parent.st_mode)
    or parent.st_uid != 0
    or parent.st_gid != 0
    or stat.S_IMODE(parent.st_mode) & 0o022
):
    raise SystemExit("unsafe immutable lock parent")

for descriptor, name in zip((11, 12), names, strict=True):
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

wrapper_fd = int(os.environ["GLM_GATE_D_WRAPPER_MEMFD"])
if wrapper_fd != 10:
    raise SystemExit("unexpected protected wrapper descriptor")
wrapper = os.fstat(wrapper_fd)
f_get_seals = getattr(fcntl, "F_GET_SEALS", 1034)
required_seals = 0x0001 | 0x0002 | 0x0004 | 0x0008
if (
    not stat.S_ISREG(wrapper.st_mode)
    or wrapper.st_nlink != 0
    or fcntl.fcntl(wrapper_fd, f_get_seals) != required_seals
):
    raise SystemExit("protected wrapper descriptor is not a sealed memfd")
raw = bytearray()
offset = 0
while block := os.pread(wrapper_fd, 1024 * 1024, offset):
    raw.extend(block)
    offset += len(block)
if len(raw) != wrapper.st_size or hashlib.sha256(raw).hexdigest() != os.environ[
    "GLM_GATE_D_WRAPPER_SHA256"
]:
    raise SystemExit("protected wrapper descriptor hash drifted")

launcher_parent = os.stat(launcher.parent, follow_symlinks=False)
launcher_fd = os.open(launcher, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
try:
    held = os.fstat(launcher_fd)
    named = os.stat(launcher, follow_symlinks=False)
    if (
        not stat.S_ISDIR(launcher_parent.st_mode)
        or launcher_parent.st_uid != 0
        or launcher_parent.st_gid != 0
        or stat.S_IMODE(launcher_parent.st_mode) & 0o022
        or not stat.S_ISREG(held.st_mode)
        or held.st_nlink != 1
        or held.st_uid != 0
        or held.st_gid != 0
        or stat.S_IMODE(held.st_mode) != 0o555
        or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)
    ):
        raise SystemExit("unsafe root-owned protected launcher")
    launcher_raw = bytearray()
    while block := os.read(launcher_fd, 1024 * 1024):
        launcher_raw.extend(block)
    after = os.fstat(launcher_fd)
    if (
        len(launcher_raw) != held.st_size
        or (held.st_size, held.st_mtime_ns, held.st_ctime_ns)
        != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        or hashlib.sha256(launcher_raw).hexdigest()
        != os.environ["GLM_GATE_D_LAUNCHER_SHA256"]
    ):
        raise SystemExit("root-owned protected launcher changed")
finally:
    os.close(launcher_fd)
RUNTIME_BOUNDARY_VERIFIER_EOF

read -r -d '' GIT_AUTHORITY_VERIFIER <<'GIT_AUTHORITY_VERIFIER_EOF' || true
import os
import re
import subprocess
import sys
from pathlib import Path

if len(sys.argv) != 5:
    raise SystemExit("expected worktree, branch, origin and pin")
worktree = Path(sys.argv[1])
branch = sys.argv[2]
origin = sys.argv[3]
pin = sys.argv[4]
if (
    not worktree.is_absolute()
    or not re.fullmatch(r"[A-Za-z0-9._/-]+", branch)
    or not re.fullmatch(r"[0-9a-f]{40}", pin)
    or origin != "git@github.com:GianluigiVitale/glm-tpu.git"
):
    raise SystemExit("invalid Git authority arguments")

git_environment = (
    "GIT_CONFIG_GLOBAL=/dev/null",
    "GIT_CONFIG_NOSYSTEM=1",
    "GIT_NO_LAZY_FETCH=1",
    "GIT_NO_REPLACE_OBJECTS=1",
    "GIT_OPTIONAL_LOCKS=0",
    "GIT_PROTOCOL_FROM_USER=0",
    "GIT_SSH_COMMAND=/usr/bin/ssh -oBatchMode=yes -oClearAllForwardings=yes -oForwardAgent=no",
    "GIT_TERMINAL_PROMPT=0",
    "HOME=/home/gianl",
    "LANG=C",
    "LC_ALL=C",
    "PATH=/usr/bin:/bin",
)
git_command = (
    "/usr/bin/env",
    "-i",
    *git_environment,
    "/usr/bin/git",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.untrackedCache=false",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "diff.external=",
    "-c",
    "core.attributesFile=/dev/null",
    "-c",
    (
        "core.sshCommand=/usr/bin/ssh -oBatchMode=yes "
        "-oClearAllForwardings=yes -oForwardAgent=no"
    ),
)


def git(cwd: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        [*git_command, *arguments],
        cwd=cwd,
        env={},
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0 or completed.stderr:
        raise SystemExit(
            "hardened Git command failed: "
            f"returncode={completed.returncode} stderr={completed.stderr!r}"
        )
    return completed.stdout


expected_ref = f"refs/heads/{branch}"
if git(worktree, "rev-parse", "--show-toplevel") != f"{worktree}\n".encode():
    raise SystemExit("Git top-level authority drifted")
if git(worktree, "rev-parse", "HEAD") != f"{pin}\n".encode():
    raise SystemExit("Git commit authority drifted")
if git(worktree, "branch", "--show-current") != f"{branch}\n".encode():
    raise SystemExit("Git branch authority drifted")
if git(worktree, "status", "--porcelain=v1", "--untracked-files=all"):
    raise SystemExit("Git worktree is not clean")
if git(worktree, "for-each-ref", "--format=%(refname)", "refs/replace"):
    raise SystemExit("Git repository has forbidden replacement refs")
if git(
    worktree,
    "config",
    "--local",
    "--includes=false",
    "--get-regexp",
    r"^remote\.origin\.(url|pushurl)$",
) != f"remote.origin.url {origin}\n".encode():
    raise SystemExit("Git origin configuration drifted")
if git(Path("/"), "ls-remote", "--refs", origin, expected_ref) != (
    f"{pin}\t{expected_ref}\n".encode()
):
    raise SystemExit("Git origin branch record drifted")
GIT_AUTHORITY_VERIFIER_EOF

[[ ${GLM_GATE_D_PROJECTION_CONTRACTION_HLO_WRAPPER_SANITIZED:-0} == 1 ]] || {
  echo "invoke only through the reviewed root-owned launcher" >&2
  exit 2
}
[[ ${HOME:-} == /home/gianl ]]
[[ ${LANG:-} == C && ${LC_ALL:-} == C ]]
[[ ${PATH:-} == /snap/bin:/usr/bin:/bin:/home/gianl/vllm-env/bin ]]
[[ ${PYTHONDONTWRITEBYTECODE:-} == 1 ]]
while IFS='=' read -r environment_name _; do
  case "$environment_name" in
    GLM_GATE_D_PROJECTION_CONTRACTION_PP16_HLO_ACQUIRE | GLM_GATE_D_PROJECTION_CONTRACTION_PP16_MODE | \
      GLM_GATE_D_PROJECTION_CONTRACTION_PP16_TAG | GLM_GATE_D_PROJECTION_CONTRACTION_HLO_WRAPPER_SANITIZED | \
      GLM_GATE_D_IMMUTABLE_LOCKS_HELD | GLM_GATE_D_LAUNCHER_PIN | \
      GLM_GATE_D_LAUNCHER_SHA256 | GLM_GATE_D_WRAPPER_MEMFD | \
      GLM_GATE_D_WRAPPER_SHA256 | HOME | LANG | LC_ALL | PATH | PWD | \
      PYTHONDONTWRITEBYTECODE | SHLVL | _) ;;
    *)
      echo "unexpected wrapper environment name: $environment_name" >&2
      exit 2
      ;;
  esac
done < <(/usr/bin/env)

[[ ${GLM_GATE_D_IMMUTABLE_LOCKS_HELD:-0} == 1 ]]
/usr/bin/python3 -I -S -B -c "$RUNTIME_BOUNDARY_VERIFIER"
exec 10<&-

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=tooling/gate-d-compensated-pp16-numerical
readonly ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly WORKTREE=/home/gianl/glm-tpu-gate-d-pp16-numerical
readonly BUCKET=gs://driftbench-dsv4-uc
readonly LOCATION=US-CENTRAL2
readonly TOPOLOGY=$WORKTREE/docs/artifacts/gate-d-runtime-locality-authority.json
readonly TOPOLOGY_SHA=49cf6bb1a553985855556d1401ad85918669df52d12f8dc5150f247d18b325eb
readonly PROJECTION_SOURCE=$WORKTREE/docs/artifacts/gate-d-projection-contraction-pp16-source.json
readonly PROJECTION_SOURCE_SHA=5744eee0ef2cf35a4566cc0de165be1338160daaae3f4dd133554b2aa8280e9f
readonly HLO_SOURCE=$WORKTREE/docs/artifacts/gate-d-projection-contraction-hlo-acquisition-source.json
readonly HLO_SOURCE_SHA=a94395ef13f5cfa7de7bc221fdc33d4498637a02ffc55d3cba985834dd99b93a
readonly IMMUTABLE_CAPSULE_ROOT=/usr/local/libexec/glm-tpu/gate-d-projection-contraction-pp16-hlo-v1
readonly DRIVER=$IMMUTABLE_CAPSULE_ROOT/acquire_gate_d_projection_contraction_pp16_hlo.py
readonly DRIVER_SHA=c660d50eb60054c9b267840230fb69bbf7259104416dc96c7b2ee01a2a14934a
readonly PUBLISHER=$IMMUTABLE_CAPSULE_ROOT/publish_gate_d_projection_contraction_pp16_hlo.py
readonly PUBLISHER_SHA=2ad2b8bd3ef9fdb1797546c3ddf1797b56facfa488a5d999b8ab9444fff335d8
readonly MIRROR_VERIFIER=$IMMUTABLE_CAPSULE_ROOT/verify_gate_d_same_region_git_mirror.py
readonly MIRROR_VERIFIER_SHA=091208165a149989f14c5c9b9d1cbe7ff20537e2c81b16319eea9603984e859b
readonly STORAGE_SITE_BUILDER=$WORKTREE/scripts/greenfield/build_gate_d_storage_site_capsule.py
readonly STORAGE_SITE_BUILDER_SHA=b7f4f869ae9b98edf5195185bf49fffcf61ef6800a2b707127694b66424c5989
readonly PUBLISHER_PYTHON=/opt/glm-tpu/gate-d-python-3.12.13-021044895e95/bin/python3.12
readonly DRIVER_PYTHON=/opt/glm-tpu/gate-d-python-3.12.13-021044895e95/bin/python3.12

[[ ${GLM_GATE_D_PROJECTION_CONTRACTION_PP16_HLO_ACQUIRE:-0} == 1 ]] || {
  echo "Gate-D projection-contraction PP16 HLO acquisition is default-off" >&2
  exit 2
}
[[ ${GLM_GATE_D_PROJECTION_CONTRACTION_PP16_MODE:-off} == compile_only ]] || {
  echo "set GLM_GATE_D_PROJECTION_CONTRACTION_PP16_MODE=compile_only" >&2
  exit 2
}

readonly PIN=${GLM_GATE_D_LAUNCHER_PIN:-}
[[ $PIN =~ ^[0-9a-f]{40}$ ]]
readonly TAG=${GLM_GATE_D_PROJECTION_CONTRACTION_PP16_TAG:-}
[[ $TAG =~ ^gate_d_projection_contraction_pp16_hlo_[0-9]{8}T[0-9]{15}Z$ ]] || {
  echo "unsafe Gate-D PP16 HLO tag: $TAG" >&2
  exit 2
}
cd /
readonly RUN_DIR=/home/gianl/gate-d-runs/$TAG
readonly REMOTE_PREFIX=$BUCKET/results/greenfield/glm52/gate_d_projection_contraction_pp16_hlo/$TAG
readonly VACANCY_EXPECTED='ERROR: (gcloud.storage.ls) One or more URLs matched no objects.'

/usr/bin/python3 -I -S -B -c "$GIT_AUTHORITY_VERIFIER" \
  "$WORKTREE" "$BRANCH" "$ORIGIN" "$PIN"
[[ $(sha256sum "$TOPOLOGY" | awk '{print $1}') == "$TOPOLOGY_SHA" ]]
[[ $(sha256sum "$PROJECTION_SOURCE" | awk '{print $1}') == "$PROJECTION_SOURCE_SHA" ]]
[[ $(sha256sum "$HLO_SOURCE" | awk '{print $1}') == "$HLO_SOURCE_SHA" ]]
[[ $(sha256sum "$DRIVER" | awk '{print $1}') == "$DRIVER_SHA" ]]
[[ $(sha256sum "$PUBLISHER" | awk '{print $1}') == "$PUBLISHER_SHA" ]]
[[ $(sha256sum "$MIRROR_VERIFIER" | awk '{print $1}') == "$MIRROR_VERIFIER_SHA" ]]
[[ $(sha256sum "$STORAGE_SITE_BUILDER" | awk '{print $1}') == "$STORAGE_SITE_BUILDER_SHA" ]]
[[ $(gcloud storage buckets describe "$BUCKET" --format='value(location)') == "$LOCATION" ]]
[[ $(gcloud compute tpus tpu-vm describe "$POD" --zone "$ZONE" --format='value(state)') == READY ]]

# Prove that the append-only tag has never existed before creating its local
# run directory.  Live, noncurrent, and soft-deleted namespaces are separate
# Cloud Storage listing surfaces and all three must be canonically vacant.
set +e
vacancy_live_output=$(PYTHONWARNINGS=ignore timeout --signal=TERM --kill-after=10 60 \
  gcloud storage ls "$REMOTE_PREFIX/**" 2>&1)
vacancy_live_rc=$?
vacancy_versions_output=$(PYTHONWARNINGS=ignore timeout --signal=TERM --kill-after=10 60 \
  gcloud storage ls --all-versions "$REMOTE_PREFIX/**" 2>&1)
vacancy_versions_rc=$?
vacancy_soft_deleted_output=$(PYTHONWARNINGS=ignore \
  timeout --signal=TERM --kill-after=10 60 \
  gcloud storage ls --soft-deleted --exhaustive "$REMOTE_PREFIX/**" 2>&1)
vacancy_soft_deleted_rc=$?
set -e
if [[ $vacancy_live_rc -ne 1 || $vacancy_live_output != "$VACANCY_EXPECTED" || \
      $vacancy_versions_rc -ne 1 || \
      $vacancy_versions_output != "$VACANCY_EXPECTED" || \
      $vacancy_soft_deleted_rc -ne 1 || \
      $vacancy_soft_deleted_output != "$VACANCY_EXPECTED" ]]; then
  echo "ABORT: append-only remote-prefix history is not canonically vacant" >&2
  exit 2
fi
readonly VACANCY_RAW=$(printf '%s\n' \
  'scope=live flags=none returncode=1' "$vacancy_live_output" \
  'scope=all_versions flags=--all-versions returncode=1' \
  "$vacancy_versions_output" \
  'scope=soft_deleted flags=--soft-deleted,--exhaustive returncode=1' \
  "$vacancy_soft_deleted_output")
readonly VACANCY_SUMMARY=$(printf 'VACANT %s %s\n' \
  live "$REMOTE_PREFIX" all_versions "$REMOTE_PREFIX" \
  soft_deleted "$REMOTE_PREFIX")

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
  local line="[gate-d-projection-contraction-pp16-hlo $(date -u +%H:%M:%S)] $*"
  printf '%s\n' "$line"
  printf '%s\n' "$line" | publisher append-log --run-dir "$RUN_DIR"
}

has_eight_unique_markers() {
  local file=$1 marker=$2
  [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | wc -l) -eq 8 ]] &&
    [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | sort -u | wc -l) -eq 8 ]]
}

strict_census() {
  local label=$1
  local member="census_${label}.txt"
  local carrier="${TAG}_${label}" ray_enum command output command_status
  # shellcheck disable=SC2016
  ray_enum='GLM_CENSUS_CARRIER='"$carrier"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "[a]cquire_gate_d_projection_contraction_pp16_hlo[.]py|[a]cquire_gate_d_forced_round_pp16_hlo[.]py|[a]cquire_gate_d_compensated_pp16_hlo[.]py|[r]un_gate_d_compensated_pp16_numerical[.]py|[V]LLM::EngineCore|[R]ayWorkerWrapper|[r]ay start --address|[r]ay start --head" 2>/dev/null || true); holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; if [[ $tools_ok -ne 1 || $ray_rc -ne 0 || $docker_rc -ne 0 ]]; then echo "CENSUS_BAD $(hostname)"; elif [[ -n $ray_pids || -n $generic || -n $holders ]] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; else echo "CENSUS_OK $(hostname)"; fi'
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
printf '%s\n' "$VACANCY_RAW" | publish_member remote_vacancy.raw.txt
printf '%s\n' "$VACANCY_SUMMARY" | publish_member remote_vacancy.txt
remote_vacant=1

say "replaying the complete locked US-CENTRAL2 Git mirror and exact origin pin"
set +e
(
  /usr/bin/env -i \
    HOME=/home/gianl \
    LANG=C \
    LC_ALL=C \
    PATH=/usr/bin:/bin \
    PYTHONDONTWRITEBYTECODE=1 \
    /usr/bin/timeout --signal=TERM --kill-after=30 1200 \
      "$PUBLISHER_PYTHON" -I -S -B "$MIRROR_VERIFIER" \
        --expected-code-hash "$PIN" \
        --expected-source-sha256 "$MIRROR_VERIFIER_SHA"
) 2>&1 | publish_member mirror.sha256
mirror_status=("${PIPESTATUS[@]}")
set -e
if [[ ${mirror_status[0]} -ne 0 || ${mirror_status[1]} -ne 0 ]]; then
  say "ABORT: same-region Git mirror or append-only mirror publisher failed"
  exit 1
fi
printf 'SYNC_OK %s %s compile_host_only=1 sealed_source_archive=1\n' \
  "$(/usr/bin/hostname)" "$PIN" | publish_member sync.txt

strict_census pre || {
  say "ABORT: pre-run census is not authenticated 8/8 zero work"
  exit 1
}

say "lowering and compiling one abstract-input PP16 stage-zero graph; invocation forbidden"
started=$(date +%s)
set +e
(
  cd /
  /usr/bin/env -i \
    HOME=/home/gianl \
    GLM_GATE_D_PROJECTION_CONTRACTION_HLO=1 \
    JAX_PLATFORMS=tpu \
    JAX_ENABLE_COMPILATION_CACHE=0 \
    LANG=C \
    LC_ALL=C \
    PATH=/usr/bin:/bin \
    PYTHONDONTWRITEBYTECODE=1 \
    TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_VISIBLE_DEVICES=0,1,2,3 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=.50 \
    /usr/bin/timeout --signal=TERM --kill-after=60 3600 \
      "$DRIVER_PYTHON" -I -S -B -u "$DRIVER" \
        --expected-code-hash "$PIN" \
        --expected-driver-sha256 "$DRIVER_SHA" \
        --topology-authority "$TOPOLOGY" \
        --topology-authority-sha256 "$TOPOLOGY_SHA" \
        --projection-contraction-source "$PROJECTION_SOURCE" \
        --projection-contraction-source-sha256 "$PROJECTION_SOURCE_SHA" \
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
