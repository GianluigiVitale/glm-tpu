#!/usr/bin/bash
# Protected exactly-once Gate-D layer-1 RMS schedule discriminator (four-chip, sealed DB548 bytes).
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
launcher = Path("/opt/glm-tpu/bin/launch_gate_d_layer1_rms_schedule_v3.py")
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
    "--no-includes",
    "--get-regexp",
    r"^remote\.origin\.(url|pushurl)$",
) != f"remote.origin.url {origin}\n".encode():
    raise SystemExit("Git origin configuration drifted")
if git(Path("/"), "ls-remote", "--refs", origin, expected_ref) != (
    f"{pin}\t{expected_ref}\n".encode()
):
    raise SystemExit("Git origin branch record drifted")
GIT_AUTHORITY_VERIFIER_EOF

read -r -d '' NUMERICAL_RESULT_VERIFIER <<'NUMERICAL_RESULT_VERIFIER_EOF' || true
import base64
import hashlib
import json
import os
import re
import stat
import sys

if len(sys.argv) != 4:
    raise SystemExit("expected run tag, remote prefix and run-directory fd")
run_tag = sys.argv[1]
remote_prefix = sys.argv[2]
try:
    run_fd = int(sys.argv[3])
except ValueError as error:
    raise SystemExit("invalid run-directory descriptor") from error
if (
    not re.fullmatch(
        r"gate_d_layer1_rms_schedule_[0-9]{8}T[0-9]{15}Z",
        run_tag,
    )
    or remote_prefix
    != (
        "gs://driftbench-dsv4-uc/results/greenfield/glm52/"
        f"gate_d_layer1_rms_schedule/{run_tag}"
    )
    or not stat.S_ISDIR(os.fstat(run_fd).st_mode)
):
    raise SystemExit("numerical terminal invocation drifted")


def read_exact(name):
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=run_fd,
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o400
        ):
            raise SystemExit("unsafe numerical terminal member: " + name)
        raw = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            raw.extend(block)
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=run_fd, follow_symlinks=False)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if (
            len(raw) != before.st_size
            or identity(before) != identity(after)
            or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise SystemExit("numerical terminal member changed: " + name)
        return bytes(raw)
    finally:
        os.close(descriptor)


def canonical(value):
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def parse_canonical(name):
    raw = read_exact(name)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit("invalid numerical terminal JSON: " + name) from error
    if type(value) is not dict or canonical(value) != raw:
        raise SystemExit("noncanonical numerical terminal JSON: " + name)
    return raw, value


hex64 = re.compile(r"[0-9a-f]{64}")


def valid_crc32c(value):
    if type(value) is not str:
        return False
    try:
        return len(base64.b64decode(value, validate=True)) == 4
    except (ValueError, base64.binascii.Error):
        return False


terminal_raw, terminal = parse_canonical("NUMERICAL_RESULT")
if set(terminal) != {
    "artifact_kind",
    "evidence_sha256",
    "gate_d_closed",
    "marker_payload_sha256",
    "performance_claim",
    "remote_ledger",
    "root_cause_fix_proven",
    "run_tag",
    "status",
    "summary_sha256",
    "tpu_numerical_execution_performed",
}:
    raise SystemExit("numerical result schema drifted")
status = terminal["status"]
if (
    terminal["artifact_kind"]
    != "gate_d_layer1_rms_schedule_result"
    or terminal["run_tag"] != run_tag
    or status not in {"SCHEDULE_ARM_EXACT", "SCHEDULE_ARM_NONEXACT"}
    or terminal["gate_d_closed"] is not False
    or terminal["performance_claim"] is not False
    or terminal["root_cause_fix_proven"] is not False
    or terminal["tpu_numerical_execution_performed"] is not True
    or any(
        hex64.fullmatch(terminal[name]) is None
        for name in ("evidence_sha256", "summary_sha256")
    )
):
    raise SystemExit("numerical result claims drifted")
self_hash = terminal["marker_payload_sha256"]
unsigned = dict(terminal)
del unsigned["marker_payload_sha256"]
if (
    type(self_hash) is not str
    or hex64.fullmatch(self_hash) is None
    or hashlib.sha256(canonical(unsigned)).hexdigest() != self_hash
):
    raise SystemExit("numerical result self-binding drifted")

ledger = terminal["remote_ledger"]
if (
    type(ledger) is not dict
    or set(ledger) != {"crc32c", "generation", "path", "sha256", "size"}
    or ledger["path"] != "remote_objects.json"
    or type(ledger["generation"]) is not str
    or not ledger["generation"].isdigit()
    or not valid_crc32c(ledger["crc32c"])
    or type(ledger["sha256"]) is not str
    or hex64.fullmatch(ledger["sha256"]) is None
    or type(ledger["size"]) is not int
    or type(ledger["size"]) is bool
    or ledger["size"] <= 0
):
    raise SystemExit("numerical result ledger binding drifted")

_, receipt = parse_canonical("terminal_upload_receipt.json")
if (
    set(receipt) != {"artifact_kind", "remote", "terminal"}
    or receipt["artifact_kind"]
    != "gate_d_layer1_rms_schedule_terminal_receipt"
    or receipt["remote"] != remote_prefix + "/NUMERICAL_RESULT"
):
    raise SystemExit("numerical terminal receipt schema drifted")
record = receipt["terminal"]
if (
    type(record) is not dict
    or set(record) != {"crc32c", "generation", "path", "sha256", "size"}
    or record["path"] != "NUMERICAL_RESULT"
    or type(record["generation"]) is not str
    or not record["generation"].isdigit()
    or not valid_crc32c(record["crc32c"])
    or type(record["sha256"]) is not str
    or record["sha256"] != hashlib.sha256(terminal_raw).hexdigest()
    or type(record["size"]) is not int
    or type(record["size"]) is bool
    or record["size"] != len(terminal_raw)
):
    raise SystemExit("numerical terminal receipt binding drifted")
print(
    f"NUMERICAL_RESULT status={status} marker_sha256={self_hash} "
    f"terminal_generation={record['generation']} "
    f"terminal_sha256={record['sha256']}"
)
NUMERICAL_RESULT_VERIFIER_EOF

[[ ${GLM_GATE_D_NUMERICAL_WRAPPER_SANITIZED:-0} == 1 ]] || {
  echo "invoke only through the reviewed root-owned launcher" >&2
  exit 2
}
[[ ${HOME:-} == /home/gianl ]]
[[ ${LANG:-} == C && ${LC_ALL:-} == C ]]
[[ ${PATH:-} == /snap/bin:/usr/bin:/bin:/home/gianl/vllm-env/bin ]]
[[ ${PYTHONDONTWRITEBYTECODE:-} == 1 ]]
while IFS='=' read -r environment_name _; do
  case "$environment_name" in
      GLM_GATE_D_LAYER1_RMS_SCHEDULE | GLM_GATE_D_LAYER1_RMS_SCHEDULE_MODE | \
      GLM_GATE_D_LAYER1_RMS_SCHEDULE_TAG | GLM_GATE_D_IMMUTABLE_LOCKS_HELD | \
      GLM_GATE_D_LAUNCHER_PIN | GLM_GATE_D_LAUNCHER_SHA256 | \
      GLM_GATE_D_NUMERICAL_WRAPPER_SANITIZED | GLM_GATE_D_WRAPPER_MEMFD | \
      GLM_GATE_D_WRAPPER_SHA256 | HOME | LANG | LC_ALL | PATH | PWD | \
      PYTHONDONTWRITEBYTECODE | SHLVL | _) ;;
    *)
      echo "unexpected wrapper environment name: $environment_name" >&2
      exit 2
      ;;
  esac
done < <(/usr/bin/env)

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=tooling/gate-d-compensated-pp16-numerical
readonly ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly WORKTREE=/home/gianl/glm-tpu-gate-d-pp16-numerical
readonly BUCKET=gs://driftbench-dsv4-uc
readonly LOCATION=US-CENTRAL2
readonly DB548_CAPTURE=/home/gianl/glm-run/greenfield_layer0_dense_partial_capture_20260813T200736889447458Z/dense_partial_capture.npz
readonly DB548_CAPTURE_SHA=f194d757d2f9ebe27430dfec8f828ca7588e433bddb7e8d99f9b917c5aac4298
readonly DB548_ENVELOPE=/home/gianl/glm-run/greenfield_layer0_dense_envelope_cross_layer_20260813T120703034434907Z/dense_envelope_cross_layer.npz
readonly DB548_ENVELOPE_SHA=6cb76623bd79e1712b6c323fa786e516abd05f6f0ca51a367bc871c4a920480f
readonly FRONTIER_CERTIFICATE=$WORKTREE/docs/artifacts/gate-d-layer1-scale-frontier-certificate.json
readonly FRONTIER_CERTIFICATE_SHA=980bbb3933866ebc0228882d9c2f76d0b212e6347f268821a71f5bcab219db3f
readonly IMMUTABLE_CAPSULE_ROOT=/usr/local/libexec/glm-tpu/gate-d-layer1-rms-schedule-v1
readonly DRIVER=$IMMUTABLE_CAPSULE_ROOT/run_gate_d_layer1_rms_schedule.py
readonly DRIVER_SHA=0c6755f4df3cb7949348ac8de28998219a0b5a0201039bdcce164c765ad069cb
readonly PUBLISHER=$IMMUTABLE_CAPSULE_ROOT/publish_gate_d_layer1_rms_schedule.py
readonly PUBLISHER_SHA=b3ec837e8535d70d15982790cb8a94f4f36e49fe4b7a115ddbd097ff47be8b24
readonly MIRROR_VERIFIER=$IMMUTABLE_CAPSULE_ROOT/verify_gate_d_same_region_git_mirror.py
readonly MIRROR_VERIFIER_SHA=091208165a149989f14c5c9b9d1cbe7ff20537e2c81b16319eea9603984e859b
readonly DRIVER_PYTHON=/opt/glm-tpu/gate-d-python-3.12.13-021044895e95/bin/python3.12
readonly PUBLISHER_PYTHON=/opt/glm-tpu/gate-d-python-3.12.13-021044895e95/bin/python3.12

[[ ${GLM_GATE_D_IMMUTABLE_LOCKS_HELD:-0} == 1 ]]
/usr/bin/python3 -I -S -B -c "$RUNTIME_BOUNDARY_VERIFIER"
exec 10<&-

[[ ${GLM_GATE_D_LAYER1_RMS_SCHEDULE:-0} == 1 ]] || {
  echo "Gate-D layer-1 RMS schedule replay is default-off" >&2
  exit 2
}
[[ ${GLM_GATE_D_LAYER1_RMS_SCHEDULE_MODE:-off} == execute_once ]] || {
  echo "set GLM_GATE_D_LAYER1_RMS_SCHEDULE_MODE=execute_once" >&2
  exit 2
}

git_local() {
  /usr/bin/env -i \
    GIT_CONFIG_GLOBAL=/dev/null \
    GIT_CONFIG_NOSYSTEM=1 \
    GIT_NO_LAZY_FETCH=1 \
    GIT_NO_REPLACE_OBJECTS=1 \
    GIT_OPTIONAL_LOCKS=0 \
    GIT_PROTOCOL_FROM_USER=0 \
    GIT_SSH_COMMAND=/bin/false \
    GIT_TERMINAL_PROMPT=0 \
    HOME=/nonexistent \
    LANG=C \
    LC_ALL=C \
    PATH=/usr/bin:/bin \
    /usr/bin/git -c core.fsmonitor=false -c core.untrackedCache=false \
      -c core.attributesFile=/dev/null -C "$WORKTREE" "$@"
}

readonly PIN=$(git_local rev-parse HEAD)
[[ $PIN == "${GLM_GATE_D_LAUNCHER_PIN:-}" ]]
/usr/bin/python3 -I -S -B -c "$GIT_AUTHORITY_VERIFIER" \
  "$WORKTREE" "$BRANCH" "$ORIGIN" "$PIN"
readonly TAG=${GLM_GATE_D_LAYER1_RMS_SCHEDULE_TAG:-}
[[ $TAG =~ ^gate_d_layer1_rms_schedule_[0-9]{8}T[0-9]{15}Z$ ]] || {
  echo "unsafe Gate-D PP16 numerical tag: $TAG" >&2
  exit 2
}
cd /
readonly RUN_ROOT=/home/gianl/gate-d-runs
readonly RUN_DIR=$RUN_ROOT/$TAG
readonly REMOTE_PREFIX=$BUCKET/results/greenfield/glm52/gate_d_layer1_rms_schedule/$TAG
readonly VACANCY_EXPECTED='ERROR: (gcloud.storage.ls) One or more URLs matched no objects.'

[[ $(git_local rev-parse --show-toplevel) == "$WORKTREE" ]]
[[ $(git_local branch --show-current) == "$BRANCH" ]]
[[ $(git_local remote get-url origin) == "$ORIGIN" ]]
[[ -z $(git_local status --porcelain) ]]
[[ $(/usr/bin/sha256sum "$DB548_CAPTURE" | /usr/bin/awk '{print $1}') == "$DB548_CAPTURE_SHA" ]]
[[ $(/usr/bin/sha256sum "$DB548_ENVELOPE" | /usr/bin/awk '{print $1}') == "$DB548_ENVELOPE_SHA" ]]
[[ $(/usr/bin/sha256sum "$FRONTIER_CERTIFICATE" | /usr/bin/awk '{print $1}') == "$FRONTIER_CERTIFICATE_SHA" ]]
[[ $(/usr/bin/sha256sum "$DRIVER" | /usr/bin/awk '{print $1}') == "$DRIVER_SHA" ]]
[[ $(/usr/bin/sha256sum "$PUBLISHER" | /usr/bin/awk '{print $1}') == "$PUBLISHER_SHA" ]]
[[ $(/usr/bin/sha256sum "$MIRROR_VERIFIER" | /usr/bin/awk '{print $1}') == "$MIRROR_VERIFIER_SHA" ]]
[[ $(/snap/bin/gcloud storage buckets describe "$BUCKET" --format='value(location)') == "$LOCATION" ]]
[[ $(/snap/bin/gcloud compute tpus tpu-vm describe "$POD" --zone "$ZONE" --format='value(state)') == READY ]]

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

# Prove that the append-only tag has never existed before creating its local
# run directory. Live, noncurrent, and soft-deleted namespaces are separate
# Cloud Storage listing surfaces and all three must be canonically vacant.
set +e
vacancy_live_output=$(PYTHONWARNINGS=ignore \
  /usr/bin/timeout --signal=TERM --kill-after=10 60 \
  /snap/bin/gcloud storage ls "$REMOTE_PREFIX/**" 2>&1)
vacancy_live_rc=$?
vacancy_versions_output=$(PYTHONWARNINGS=ignore \
  /usr/bin/timeout --signal=TERM --kill-after=10 60 \
  /snap/bin/gcloud storage ls --all-versions "$REMOTE_PREFIX/**" 2>&1)
vacancy_versions_rc=$?
vacancy_soft_deleted_output=$(PYTHONWARNINGS=ignore \
  /usr/bin/timeout --signal=TERM --kill-after=10 60 \
  /snap/bin/gcloud storage ls --soft-deleted --exhaustive \
    "$REMOTE_PREFIX/**" 2>&1)
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
readonly VACANCY_RAW=$(/usr/bin/printf '%s\n' \
  'scope=live flags=none returncode=1' "$vacancy_live_output" \
  'scope=all_versions flags=--all-versions returncode=1' \
  "$vacancy_versions_output" \
  'scope=soft_deleted flags=--soft-deleted,--exhaustive returncode=1' \
  "$vacancy_soft_deleted_output")
readonly VACANCY_SUMMARY=$(/usr/bin/printf 'VACANT %s %s\n' \
  live "$REMOTE_PREFIX" all_versions "$REMOTE_PREFIX" \
  soft_deleted "$REMOTE_PREFIX")

run_identity=$(publisher_init init --run-dir "$RUN_DIR")
[[ $run_identity =~ ^RUN_IDENTITY\ ([0-9]+:[0-9]+)$ ]]
readonly EXPECTED_RUN_IDENTITY=${BASH_REMATCH[1]}
exec 7<"$RUN_DIR"
readonly OBSERVED_RUN_IDENTITY=$(/usr/bin/stat -Lc '%d:%i' /proc/self/fd/7)
[[ $OBSERVED_RUN_IDENTITY == "$EXPECTED_RUN_IDENTITY" ]]
/usr/bin/flock -n 7 || {
  echo "Gate-D numerical run-directory supervisor lock is unavailable" >&2
  exit 1
}

say() {
  local line="[gate-d-layer1-rms-schedule $(/usr/bin/date -u +%H:%M:%S)] $*"
  /usr/bin/printf '%s\n' "$line"
  /usr/bin/printf '%s\n' "$line" | publisher append-log --run-dir "$RUN_DIR"
}

has_eight_unique_markers() {
  local file=$1 marker=$2
  [[ $(/usr/bin/awk -v marker="$marker" '$1 == marker {print $2}' "$file" | /usr/bin/wc -l) -eq 8 ]] &&
    [[ $(/usr/bin/awk -v marker="$marker" '$1 == marker {print $2}' "$file" | /usr/bin/sort -u | /usr/bin/wc -l) -eq 8 ]]
}

strict_census() {
  local label=$1
  local member="census_${label}.txt"
  local carrier="${TAG}_${label}"
  local ray_enum command output command_status
  # shellcheck disable=SC2016
  ray_enum='GLM_CENSUS_CARRIER='"$carrier"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "[r]un_gate_d_layer1_rms_schedule[.]py|[V]LLM::EngineCore|[R]ayWorkerWrapper|[r]ay start --address|[r]ay start --head" 2>/dev/null || true); holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; if [[ $tools_ok -ne 1 || $ray_rc -ne 0 || $docker_rc -ne 0 ]]; then echo "CENSUS_BAD $(hostname)"; elif [[ -n $ray_pids || -n $generic || -n $holders ]] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; else echo "CENSUS_OK $(hostname)"; fi'
  set +e
  output=$(GLM_CENSUS_CARRIER="$carrier" /usr/bin/timeout --signal=TERM --kill-after=10 180 \
    /snap/bin/gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
      --command="$command" 2>&1)
  command_status=$?
  set -e
  /usr/bin/printf '%s\n' "$output" | publish_member "$member" || return 1
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
        ! -e /proc/self/fd/7/NUMERICAL_RESULT ]]; then
    say "FAILED status=$status; publishing bounded diagnostic"
    publish_diagnostic "$status" || census_status=1
  fi
  if [[ $census_status -ne 0 ]]; then
    trap - EXIT
    exit 71
  fi
  return "$status"
}

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
/usr/bin/flock -n 9 || {
  say "ABORT: another protected TPU workflow holds the global lease"
  exit 1
}
exec 8>/home/gianl/.glm-tpu-rsync.lock
/usr/bin/flock 8
trap on_exit EXIT

say "RUN_DIR=$RUN_DIR PIN=$PIN mode=execute_once devices=0,1,2,3"
/usr/bin/printf '%s\n' "$VACANCY_RAW" | publish_member remote_vacancy.raw.txt
/usr/bin/printf '%s\n' "$VACANCY_SUMMARY" | publish_member remote_vacancy.txt
remote_vacant=1

say "replaying pushed pin and full locked US-CENTRAL2 Git mirror closure"
set +e
/usr/bin/env -i \
  HOME=/home/gianl \
  LANG=C \
  LC_ALL=C \
  PATH=/usr/bin:/bin \
  PYTHONDONTWRITEBYTECODE=1 \
  "$PUBLISHER_PYTHON" -I -S -B "$MIRROR_VERIFIER" \
    --expected-code-hash "$PIN" \
    --expected-source-sha256 "$MIRROR_VERIFIER_SHA" | \
  publish_member mirror.sha256
mirror_pipeline_status=("${PIPESTATUS[@]}")
set -e
if [[ ${mirror_pipeline_status[0]} -ne 0 || ${mirror_pipeline_status[1]} -ne 0 ]]; then
  say "ABORT: same-region Git mirror replay or append-only publication failed"
  exit 1
fi
/usr/bin/printf 'SYNC_OK %s %s origin_and_same_region_mirror\n' \
  "$(/usr/bin/hostname)" "$PIN" | publish_member sync.txt

strict_census pre || {
  say "ABORT: pre-run census is not authenticated 8/8 zero work"
  exit 1
}

say "executing one exact-input four-chip layer-1 RMS schedule discriminator"
started=$(/usr/bin/date +%s)
set +e
(
  cd /
  /usr/bin/env -i \
    HOME=/home/gianl \
    GLM_GATE_D_LAYER1_RMS_SCHEDULE=1 \
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
        --db548-capture "$DB548_CAPTURE" \
        --db548-envelope "$DB548_ENVELOPE" \
        --execute-once 1 \
        --run-dir "$RUN_DIR" \
        --run-dir-fd 7
) 2>&1 | publish_member runner.log
pipeline_status=("${PIPESTATUS[@]}")
set -e
if [[ ${pipeline_status[0]} -ne 0 || ${pipeline_status[1]} -ne 0 ]]; then
  say "ABORT: numerical process or append-only log publisher failed"
  exit 1
fi
elapsed=$(( $(/usr/bin/date +%s) - started ))
say "numerical process exited successfully in ${elapsed}s"

strict_census post || {
  say "ABORT: post-run census is not authenticated 8/8 zero work"
  exit 1
}
post_census_done=1

say "publishing generation-bound archive; terminal upload is the final mutation"
result_authority=$(publisher success --run-dir "$RUN_DIR" \
  --remote-prefix "$REMOTE_PREFIX" --elapsed "$elapsed")
if [[ $result_authority =~ ^NUMERICAL_RESULT\ status=(SCHEDULE_ARM_EXACT|SCHEDULE_ARM_NONEXACT)\ marker_sha256=([0-9a-f]{64})\ terminal_generation=([0-9]+)\ terminal_sha256=([0-9a-f]{64})$ ]]; then
  readonly result_status=${BASH_REMATCH[1]}
else
  echo "Gate-D numerical publisher authority drifted" >&2
  exit 2
fi
local_result_authority=$(/usr/bin/python3 -I -S -B -c "$NUMERICAL_RESULT_VERIFIER" \
  "$TAG" "$REMOTE_PREFIX" 7)
[[ $local_result_authority == "$result_authority" ]] || {
  echo "Gate-D numerical local/remote authority mismatch" >&2
  exit 2
}
terminal_written=1
trap - EXIT
case "$result_status" in
  SCHEDULE_ARM_EXACT)
    /usr/bin/printf '%s\n' \
      "SCHEDULE_ARM_EXACT archive=$REMOTE_PREFIX execution_count=2 gate_d_open=true"
    exit 0
    ;;
  SCHEDULE_ARM_NONEXACT)
    /usr/bin/printf '%s\n' \
      "SCHEDULE_ARM_NONEXACT archive=$REMOTE_PREFIX execution_count=2 gate_d_open=true"
    exit 3
    ;;
  *)
    echo "Gate-D numerical terminal status drifted" >&2
    exit 2
    ;;
esac
