#!/usr/bin/env bash
set -euo pipefail

# One accepted-tree, initialization-only HLO acquisition.  This is an oracle
# mechanism probe: it creates no request, output token, DB row, timing claim,
# numerical claim, or Gate-D result.

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly BRANCH=rewrite/topology-first-decode
readonly ACCEPTED_REPO=/home/gianl/tpu-inference
readonly ACCEPTED_PIN=b3c25df47ac98783912dc658878181ec0a8ae16d
readonly ACCEPTED_SHORT=${ACCEPTED_PIN:0:12}
readonly ACCEPTED_TRACKED_FILES=947
readonly VLLM_REPO=/home/gianl/vllm-build-a30addc
readonly VLLM_PIN=a30addc7548a9a8b9b3323a7bc3eb7d7c4895d1c
readonly VLLM_TRACKED_FILES=5493
readonly VLLM_ARCHIVE_FILES=5494
readonly VLLM_VERSION=0.1.dev1+ga30addc75
readonly VLLM_VERSION_FILE=$WORKTREE/scripts/greenfield/resources/accepted_vllm_version.py
readonly VLLM_VERSION_FILE_SHA=91651499622913dc1cf3eb6418f60a7f99e4c3f23231fe4ce3609f44d018e50a
readonly HARNESS_REPO=/home/gianl/glm-tpu
readonly HARNESS_PIN=3409bf58a758e7bfa398eb92051db701d623e6b9
readonly LAUNCHER_SHA=b587a8a39262a185c3386989bc9471953b407dc9adbfcc0ae30bb50db68bea23
readonly NETWORK_VALIDATOR_SHA=00bc87d99e087434eb429a66a3e10df9f2bd9140eac92c2c60ac62ac39ae2db1
readonly MODEL_ROOT=/home/gianl/gcs-models/models/GLM-5.2-FP8
readonly GOLDEN_ROOT=/home/gianl/gcs-models/manifests/golden_v1
readonly GOLDEN_SHA=916d421a10de9495086c0ad52645c9937746c88bae87a5ca1a69483bfe60d45d
readonly GOLDEN_BYTES=321146
readonly CALLBACK_CERTIFICATE=$WORKTREE/docs/artifacts/callback-executable-class-certificate.json
readonly CALLBACK_CERTIFICATE_SHA=6e58bca961c0629799683ccfb75efda195206885e36975e497630ec379076e48
readonly BUCKET=gs://driftbench-dsv4-uc
readonly DRIVER=$WORKTREE/scripts/greenfield/compile_accepted_db485_hlo.py
readonly SEALER=$WORKTREE/scripts/greenfield/seal_accepted_db485_compile_only_hlo.py
readonly PUBLISHER=$WORKTREE/scripts/greenfield/publish_accepted_db485_compile_only_hlo.py

readonly TAG=${GLM_ACCEPTED_DB485_COMPILE_ONLY_TAG:-accepted_db485_compile_only_hlo_$(date -u +%Y%m%dT%H%M%S%NZ)}
[[ $TAG =~ ^[A-Za-z0-9_]+$ ]] || {
  echo "unsafe compile-only tag: $TAG" >&2
  exit 2
}
readonly RUN_DIR=/home/gianl/glm-run/$TAG
readonly REMOTE_PREFIX=$BUCKET/oracles/greenfield/glm52/accepted_compile_only_hlo/$TAG
readonly CODE_BUNDLE_LOCAL=$RUN_DIR/.transport_glm_accepted_${TAG}.bundle
readonly CODE_BUNDLE_REMOTE=/tmp/glm_accepted_${TAG}.bundle
readonly CODE_RUNTIME=/tmp/glm_accepted_${TAG}
readonly CODE_RUNTIME_TEMP=/tmp/glm_accepted_${TAG}.tmp
readonly VLLM_ARCHIVE_LOCAL=$RUN_DIR/.transport_glm_vllm_${TAG}.tar.gz
readonly VLLM_TAR_LOCAL=$RUN_DIR/.transport_glm_vllm_${TAG}.tar
readonly VLLM_ARCHIVE_REMOTE=/tmp/glm_vllm_${TAG}.tar.gz
readonly VLLM_RUNTIME=/tmp/glm_vllm_${TAG}
readonly DUMP_ROOT=/tmp/$TAG
readonly HLO_RAW=$DUMP_ROOT/hlo_raw
readonly HLO_COMPACT=$DUMP_ROOT/hlo_compact
readonly XLA_CACHE=$DUMP_ROOT/xla_cache
readonly TOPK_PREFIX=$DUMP_ROOT/topk.npz
readonly HLO_LOCAL=$RUN_DIR/hlo
readonly HLO_AUDIT_LOCAL=$RUN_DIR/hlo_replica_audits
readonly HLO_DIVERGENCE_LOCAL=$RUN_DIR/divergent_hlo_replicas

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing a dirty greenfield worktree" >&2
  exit 2
}
[[ $(git -C "$WORKTREE" rev-parse HEAD) == \
   $(git -C "$WORKTREE" rev-parse "origin/$BRANCH") ]] || {
  echo "greenfield HEAD is not pushed" >&2
  exit 2
}
[[ $(git -C "$ACCEPTED_REPO" rev-parse HEAD) == "$ACCEPTED_PIN" &&
   -z $(git -C "$ACCEPTED_REPO" status --porcelain --untracked-files=no) &&
   $(git -C "$ACCEPTED_REPO" ls-tree -r --name-only HEAD | wc -l) -eq \
     $ACCEPTED_TRACKED_FILES ]] || {
  echo "accepted TPU-inference source identity drifted" >&2
  exit 2
}
[[ $(git -C "$VLLM_REPO" rev-parse HEAD) == "$VLLM_PIN" &&
   -z $(git -C "$VLLM_REPO" status --porcelain --untracked-files=no) &&
   $(git -C "$VLLM_REPO" ls-tree -r --name-only HEAD | wc -l) -eq \
     $VLLM_TRACKED_FILES ]] || {
  echo "accepted vLLM source identity drifted" >&2
  exit 2
}
[[ $(git -C "$HARNESS_REPO" rev-parse HEAD) == "$HARNESS_PIN" &&
   -z $(git -C "$HARNESS_REPO" status --porcelain --untracked-files=no) &&
   $(sha256sum "$HARNESS_REPO/scripts/launch_glm_32chip.sh" | awk '{print $1}') == \
     "$LAUNCHER_SHA" &&
   $(sha256sum "$WORKTREE/scripts/validate_ray_network.sh" | awk '{print $1}') == \
     "$NETWORK_VALIDATOR_SHA" ]] || {
  echo "protected launch/network helper identity drifted" >&2
  exit 2
}
[[ $(sha256sum "$CALLBACK_CERTIFICATE" | awk '{print $1}') == \
   "$CALLBACK_CERTIFICATE_SHA" ]] || {
  echo "callback executable-class certificate drifted" >&2
  exit 2
}
[[ $(sha256sum "$VLLM_VERSION_FILE" | awk '{print $1}') == \
   "$VLLM_VERSION_FILE_SHA" ]] || {
  echo "accepted generated vLLM version file drifted" >&2
  exit 2
}
[[ $(timeout --signal=TERM --kill-after=10 60 \
  gcloud storage buckets describe "$BUCKET" --format='value(location)') == \
   US-CENTRAL2 ]] || {
  echo "approved bucket is no longer US-CENTRAL2" >&2
  exit 2
}
[[ $(timeout --signal=TERM --kill-after=10 60 \
  gcloud compute tpus tpu-vm describe "$POD" --zone "$ZONE" \
  --format='value(state)') == READY ]] || {
  echo "existing TPU pod is not READY" >&2
  exit 2
}
[[ ! -e $RUN_DIR && ! -L $RUN_DIR &&
   ! -e $CODE_BUNDLE_LOCAL && ! -L $CODE_BUNDLE_LOCAL &&
   ! -e $VLLM_ARCHIVE_LOCAL && ! -L $VLLM_ARCHIVE_LOCAL &&
   ! -e $VLLM_TAR_LOCAL && ! -L $VLLM_TAR_LOCAL ]] || {
  echo "append-only local or transport path already exists" >&2
  exit 2
}
mkdir -p "$RUN_DIR" "$HLO_LOCAL"

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  echo "another protected TPU workflow holds the global lease" >&2
  exit 1
}

say() {
  echo "[accepted-compile-only $(date -u +%H:%M:%S)] $*" |
    tee -a "$RUN_DIR/orchestrator.log"
}

has_eight_unique_markers() {
  local file=$1 marker=$2 expected_fields=${3:-2}
  bash "$WORKTREE/scripts/validate_ray_network.sh" receipts \
    "$marker" 8 '' "$expected_fields" <"$file"
}

strict_census() {
  local label=$1
  local out="$RUN_DIR/census_${label}.txt"
  local carrier="${TAG}_${label}"
  local ray_enum command
  # shellcheck disable=SC2016
  ray_enum='GLM_CENSUS_CARRIER='"$carrier"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[r]ay start --address|[r]ay start --head|[c]ompile_accepted_db485_hlo[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname)"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" \
    bash "$WORKTREE/scripts/validate_ray_network.sh" bounded 120 \
    gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

stop_owned_runtime() {
  local command
  # shellcheck disable=SC2016
  command='timeout --signal=TERM --kill-after=5 -- 20 /home/gianl/vllm-env/bin/ray stop -f >/dev/null 2>&1 || true; sudo pkill -TERM -f "[r]ay start --address|[r]ay start --head" >/dev/null 2>&1 || true; sudo pkill -TERM -f "VLLM::[E]ngineCore" >/dev/null 2>&1 || true; sudo pkill -TERM -f "[R]ayWorkerWrapper" >/dev/null 2>&1 || true; sudo pkill -TERM -x raylet >/dev/null 2>&1 || true; sleep 2; sudo pkill -KILL -f "[r]ay start --address|[r]ay start --head" >/dev/null 2>&1 || true; sudo pkill -KILL -f "VLLM::[E]ngineCore" >/dev/null 2>&1 || true; sudo pkill -KILL -f "[R]ayWorkerWrapper" >/dev/null 2>&1 || true; sudo pkill -KILL -x raylet >/dev/null 2>&1 || true; sudo rm -f /tmp/libtpu_lockfile; echo STOP_OK $(hostname)'
  bash "$WORKTREE/scripts/validate_ray_network.sh" bounded 120 \
    gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$command" >"$RUN_DIR/stop.txt" 2>&1 || return 1
  has_eight_unique_markers "$RUN_DIR/stop.txt" STOP_OK
}

DRIVER_SESSION=""
stop_driver_session() {
  if [[ $DRIVER_SESSION =~ ^[0-9]+$ ]] &&
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

cleanup_run_owned_sources() {
  local label=$1 retain_dump=${2:-0} command marker expected_fields
  # Every recursive deletion is constrained to this validated tag's exact
  # /tmp names; preflight refuses any pre-existing target.
  # shellcheck disable=SC2016
  command='code='"$CODE_RUNTIME"'; code_tmp='"$CODE_RUNTIME_TEMP"'; code_bundle='"$CODE_BUNDLE_REMOTE"'; vllm='"$VLLM_RUNTIME"'; vllm_archive='"$VLLM_ARCHIVE_REMOTE"'; dump='"$DUMP_ROOT"'; retain_dump='"$retain_dump"'; safe=1; printf "%s\n" "$code" | grep -Eq "^/tmp/glm_accepted_[A-Za-z0-9_]+$" || safe=0; [ "$code_tmp" = "${code}.tmp" ] || safe=0; [ "$code_bundle" = "${code}.bundle" ] || safe=0; printf "%s\n" "$vllm" | grep -Eq "^/tmp/glm_vllm_[A-Za-z0-9_]+$" || safe=0; [ "$vllm_archive" = "${vllm}.tar.gz" ] || safe=0; [ "$dump" = /tmp/'"$TAG"' ] || safe=0; [[ $retain_dump == 0 || $retain_dump == 1 ]] || safe=0; if [ "$safe" -ne 1 ]; then echo "SOURCE_CLEAN_BAD $(hostname) unsafe_target"; exit 0; fi; rm -rf -- "$code" "$code_tmp" "$vllm"; rm -f -- "$code_bundle" "$vllm_archive"; if [ "$retain_dump" -eq 0 ]; then rm -rf -- "$dump"; fi; sources_absent=0; [ ! -e "$code" ] && [ ! -e "$code_tmp" ] && [ ! -e "$code_bundle" ] && [ ! -e "$vllm" ] && [ ! -e "$vllm_archive" ] && sources_absent=1; if [ "$sources_absent" -eq 1 ] && [ "$retain_dump" -eq 0 ] && [ ! -e "$dump" ] && [ ! -L "$dump" ]; then echo "SOURCE_CLEAN_OK $(hostname)"; elif [ "$sources_absent" -eq 1 ] && [ "$retain_dump" -eq 1 ] && [ -d "$dump" ] && [ ! -L "$dump" ]; then echo "SOURCE_RETAINED $(hostname) dump=$dump"; else echo "SOURCE_CLEAN_BAD $(hostname) residual"; fi'
  bash "$WORKTREE/scripts/validate_ray_network.sh" bounded 180 \
    gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$command" >"$RUN_DIR/source_cleanup_${label}.txt" \
    2>"$RUN_DIR/source_cleanup_${label}_ssh.txt" || return 1
  if [[ $retain_dump -eq 1 ]]; then
    marker=SOURCE_RETAINED
    expected_fields=3
  else
    marker=SOURCE_CLEAN_OK
    expected_fields=2
  fi
  has_eight_unique_markers \
    "$RUN_DIR/source_cleanup_${label}.txt" "$marker" "$expected_fields"
}

preserve_divergent_hlo_replicas() {
  local worker target deadline remaining per_copy copy_rc=0 validation_rc
  # Fail closed from the first preservation instruction: any local/transport
  # error after this point makes the exact remote tag dumps cleanup-ineligible.
  hlo_preservation_incomplete=1
  if [[ $runtime_started -eq 1 ]]; then
    stop_owned_runtime || {
      hlo_preservation_incomplete=1
      return 1
    }
    strict_census preservation_pre || {
      hlo_preservation_incomplete=1
      return 1
    }
    post_census_done=1
    runtime_started=0
  fi
  mkdir -p "$HLO_DIVERGENCE_LOCAL/audits"
  deadline=$((SECONDS + 600))
  for worker in 0 1 2 3 4 5 6 7; do
    remaining=$((deadline - SECONDS))
    if [[ $remaining -le 0 ]]; then
      copy_rc=1
      break
    fi
    per_copy=$remaining
    [[ $per_copy -gt 120 ]] && per_copy=120
    target="$HLO_DIVERGENCE_LOCAL/worker_$worker"
    mkdir -p "$target"
    if ! bash "$WORKTREE/scripts/validate_ray_network.sh" bounded "$per_copy" \
      gcloud compute tpus tpu-vm scp --zone "$ZONE" --worker="$worker" \
      "$POD:$HLO_COMPACT/*" "$target/" \
      >"$HLO_DIVERGENCE_LOCAL/worker_${worker}_copy.txt" \
      2>"$HLO_DIVERGENCE_LOCAL/worker_${worker}_copy_ssh.txt"; then
      copy_rc=1
      continue
    fi
    if [[ -f $target/replica_audit.tsv && ! -L $target/replica_audit.tsv ]]; then
      cp -- "$target/replica_audit.tsv" \
        "$HLO_DIVERGENCE_LOCAL/audits/worker_$worker.tsv"
    else
      copy_rc=1
    fi
  done
  set +e
  PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
    "$RUN_DIR/hlo_replicas.txt" "$HLO_DIVERGENCE_LOCAL" \
    >"$HLO_DIVERGENCE_LOCAL/preservation_validation.txt" \
    2>"$HLO_DIVERGENCE_LOCAL/preservation_validation_stderr.txt" <<'PY'
from pathlib import Path
import sys
from glm_tpu.greenfield.benchmarking.accepted_compile_only_hlo import (
    validate_hlo_replica_evidence,
    validate_hlo_replica_payload,
    validate_worker_host_receipts,
)
root = Path(sys.argv[2])
expected_hosts = validate_worker_host_receipts(
    (Path(sys.argv[1]).parent / "prereq.txt").read_bytes(), marker="PREREQ_OK"
)
report = validate_hlo_replica_evidence(
    Path(sys.argv[1]).read_bytes(),
    root / "audits",
    expected_hosts,
    require_common_identity=False,
)
for audit in report["audits"]:
    validate_hlo_replica_payload(root / f"worker_{audit['worker']}", audit)
print(f"EIGHT_HLO_REPLICAS_PRESERVED identity_count={report['common_identity_count']}")
PY
  validation_rc=$?
  set -e
  if [[ $copy_rc -eq 0 && $validation_rc -eq 0 ]]; then
    hlo_preservation_verified=1
    hlo_preservation_incomplete=0
    return 0
  fi
  return 1
}

prepare_hlo_audit_collection() {
  if mkdir "$HLO_AUDIT_LOCAL"; then
    return 0
  fi
  preserve_divergent_hlo_replicas || true
  return 1
}

runtime_started=0
sources_prepared=0
post_census_done=0
terminal_success=0
remote_prefix_owned=0
terminal_publication_started=0
hlo_preservation_verified=0
hlo_preservation_incomplete=0

on_exit() {
  local status=$?
  local attempt
  stop_driver_session
  if [[ $runtime_started -eq 1 ]]; then
    for attempt in 1 2; do
      stop_owned_runtime || true
      if strict_census "failure_exit_${attempt}"; then
        post_census_done=1
        runtime_started=0
        break
      fi
    done
  elif [[ $post_census_done -eq 0 ]] && strict_census failure_exit; then
    post_census_done=1
  fi
  if [[ $sources_prepared -eq 1 && $post_census_done -eq 1 ]]; then
    if [[ $hlo_preservation_incomplete -eq 1 ]]; then
      cleanup_run_owned_sources failure_exit 1 || true
    else
      cleanup_run_owned_sources failure_exit 0 || true
    fi
  fi
  rm -f -- "$CODE_BUNDLE_LOCAL" "$VLLM_ARCHIVE_LOCAL" "$VLLM_TAR_LOCAL"
  if [[ $status -ne 0 && $terminal_success -eq 0 &&
        $remote_prefix_owned -eq 1 && $terminal_publication_started -eq 0 ]]; then
    say "FAILED status=$status; uploading failure diagnostics only"
    failure_upload_timeout=300
    [[ -d $HLO_DIVERGENCE_LOCAL ]] && failure_upload_timeout=1800
    bash "$WORKTREE/scripts/validate_ray_network.sh" bounded \
      "$failure_upload_timeout" \
      gcloud storage cp --recursive --no-clobber "$RUN_DIR" \
        "$REMOTE_PREFIX/diagnostic_local/" >/dev/null 2>&1 || true
  fi
  return "$status"
}
trap on_exit EXIT

set +e
remote_listing=$(bash "$WORKTREE/scripts/validate_ray_network.sh" bounded 60 \
  gcloud storage ls "$REMOTE_PREFIX/**" 2>&1)
remote_rc=$?
set -e
printf '%s\n' "$remote_listing" >"$RUN_DIR/remote_prefix_preflight.txt"
if [[ $remote_rc -eq 0 ]] || [[ $remote_rc -ne 1 ]] ||
   ! grep -q 'matched no objects' "$RUN_DIR/remote_prefix_preflight.txt"; then
  say "ABORT: append-only remote-prefix vacancy check failed"
  exit 1
fi
remote_prefix_owned=1

strict_census pre || {
  say "ABORT: fleet is not eight-host zero work"
  exit 1
}
MIN_FREE_GB=20 WARN_FREE_GB=25 bash "$WORKTREE/scripts/disk_watchdog.sh" check \
  >"$RUN_DIR/disk_preflight.txt" 2>&1 || {
  say "ABORT: run-owned HLO/cache space is below the 20-GiB reserve"
  exit 1
}

say "building pin-derived source transports"
git -C "$ACCEPTED_REPO" bundle create "$CODE_BUNDLE_LOCAL" HEAD
git -C "$ACCEPTED_REPO" bundle verify "$CODE_BUNDLE_LOCAL" \
  >"$RUN_DIR/code_bundle_verify.txt" 2>&1
git -C "$VLLM_REPO" archive --format=tar \
  --output "$VLLM_TAR_LOCAL" "$VLLM_PIN"
tar --append --file "$VLLM_TAR_LOCAL" \
  --transform='s#^accepted_vllm_version.py$#vllm/_version.py#' \
  -C "$(dirname "$VLLM_VERSION_FILE")" "$(basename "$VLLM_VERSION_FILE")"
gzip -n -c "$VLLM_TAR_LOCAL" >"$VLLM_ARCHIVE_LOCAL"
rm -f -- "$VLLM_TAR_LOCAL"
CODE_BUNDLE_SHA=$(sha256sum "$CODE_BUNDLE_LOCAL" | awk '{print $1}')
VLLM_ARCHIVE_SHA=$(sha256sum "$VLLM_ARCHIVE_LOCAL" | awk '{print $1}')
readonly CODE_BUNDLE_SHA VLLM_ARCHIVE_SHA
printf 'pin=%s\nbundle_sha256=%s\ntracked_entries=%s\n' \
  "$ACCEPTED_PIN" "$CODE_BUNDLE_SHA" "$ACCEPTED_TRACKED_FILES" \
  >"$RUN_DIR/code_identity.txt"
printf 'pin=%s\narchive_sha256=%s\ntracked_entries=%s\narchive_files=%s\nversion=%s\nversion_file_sha256=%s\n' \
  "$VLLM_PIN" "$VLLM_ARCHIVE_SHA" "$VLLM_TRACKED_FILES" \
  "$VLLM_ARCHIVE_FILES" "$VLLM_VERSION" "$VLLM_VERSION_FILE_SHA" \
  >"$RUN_DIR/vllm_identity.txt"

# Refuse every pre-existing run-owned remote path before copying.
# shellcheck disable=SC2016
vacancy='code='"$CODE_RUNTIME"'; code_tmp='"$CODE_RUNTIME_TEMP"'; code_bundle='"$CODE_BUNDLE_REMOTE"'; vllm='"$VLLM_RUNTIME"'; vllm_archive='"$VLLM_ARCHIVE_REMOTE"'; dump='"$DUMP_ROOT"'; if [ ! -e "$code" ] && [ ! -L "$code" ] && [ ! -e "$code_tmp" ] && [ ! -L "$code_tmp" ] && [ ! -e "$code_bundle" ] && [ ! -L "$code_bundle" ] && [ ! -e "$vllm" ] && [ ! -L "$vllm" ] && [ ! -e "$vllm_archive" ] && [ ! -L "$vllm_archive" ] && [ ! -e "$dump" ] && [ ! -L "$dump" ]; then echo "VACANT_OK $(hostname)"; else echo "VACANT_BAD $(hostname)"; fi'
bash "$WORKTREE/scripts/validate_ray_network.sh" bounded 120 \
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$vacancy" >"$RUN_DIR/transport_vacancy.txt" \
  2>"$RUN_DIR/transport_vacancy_ssh.txt"
has_eight_unique_markers "$RUN_DIR/transport_vacancy.txt" VACANT_OK || {
  say "ABORT: a run-owned fleet path already exists"
  exit 1
}
sources_prepared=1

: >"$RUN_DIR/source_copy.txt"
for worker in 0 1 2 3 4 5 6 7; do
  if bash "$WORKTREE/scripts/validate_ray_network.sh" bounded 180 \
      gcloud compute tpus tpu-vm scp --zone "$ZONE" --worker="$worker" \
      "$CODE_BUNDLE_LOCAL" "$POD:$CODE_BUNDLE_REMOTE" >/dev/null 2>&1 &&
     bash "$WORKTREE/scripts/validate_ray_network.sh" bounded 180 \
      gcloud compute tpus tpu-vm scp --zone "$ZONE" --worker="$worker" \
      "$VLLM_ARCHIVE_LOCAL" "$POD:$VLLM_ARCHIVE_REMOTE" >/dev/null 2>&1; then
    echo "SOURCE_COPY_OK $worker" >>"$RUN_DIR/source_copy.txt"
  else
    echo "SOURCE_COPY_BAD $worker" >>"$RUN_DIR/source_copy.txt"
  fi
done
has_eight_unique_markers "$RUN_DIR/source_copy.txt" SOURCE_COPY_OK || {
  say "ABORT: exact sources did not reach all hosts"
  exit 1
}

# shellcheck disable=SC2016
install_sources='set -e; code_bundle='"$CODE_BUNDLE_REMOTE"'; code_sha='"$CODE_BUNDLE_SHA"'; code='"$CODE_RUNTIME"'; code_tmp='"$CODE_RUNTIME_TEMP"'; code_pin='"$ACCEPTED_PIN"'; code_files='"$ACCEPTED_TRACKED_FILES"'; vllm_archive='"$VLLM_ARCHIVE_REMOTE"'; vllm_sha='"$VLLM_ARCHIVE_SHA"'; vllm='"$VLLM_RUNTIME"'; vllm_files='"$VLLM_ARCHIVE_FILES"'; vllm_version='"$VLLM_VERSION"'; version_sha='"$VLLM_VERSION_FILE_SHA"'; [ "$(sha256sum "$code_bundle" | cut -d " " -f 1)" = "$code_sha" ] || { echo "SOURCE_SYNC_BAD $(hostname) code_bundle_sha"; exit 0; }; [ "$(sha256sum "$vllm_archive" | cut -d " " -f 1)" = "$vllm_sha" ] || { echo "SOURCE_SYNC_BAD $(hostname) vllm_archive_sha"; exit 0; }; git clone -q --no-checkout "$code_bundle" "$code_tmp" || { echo "SOURCE_SYNC_BAD $(hostname) clone"; exit 0; }; git -C "$code_tmp" bundle verify "$code_bundle" >/dev/null 2>&1 || { echo "SOURCE_SYNC_BAD $(hostname) bundle_verify"; exit 0; }; git -C "$code_tmp" checkout -q --detach "$code_pin" || { echo "SOURCE_SYNC_BAD $(hostname) checkout"; exit 0; }; mv "$code_tmp" "$code"; mkdir "$vllm"; tar -xzf "$vllm_archive" -C "$vllm"; actual_code=$(git -C "$code" rev-parse HEAD); dirty=$(git -C "$code" status --porcelain | wc -l); actual_code_files=$(git -C "$code" ls-tree -r --name-only HEAD | wc -l); actual_vllm_files=$(find "$vllm" \( -type f -o -type l \) | wc -l); actual_version_sha=$(sha256sum "$vllm/vllm/_version.py" 2>/dev/null | cut -d " " -f 1); actual_vllm_version=$(PYTHONPATH="$vllm" /home/gianl/vllm-env/bin/python -c "import pathlib,vllm; pathlib.Path(vllm.__file__).resolve().relative_to(pathlib.Path(\"$vllm\").resolve()); print(vllm.__version__)" 2>/dev/null || true); rm -f -- "$code_bundle" "$vllm_archive"; if [ "$actual_code" = "$code_pin" ] && [ "$dirty" -eq 0 ] && [ "$actual_code_files" -eq "$code_files" ] && [ "$actual_vllm_files" -eq "$vllm_files" ] && [ "$actual_version_sha" = "$version_sha" ] && [ "$actual_vllm_version" = "$vllm_version" ] && [ ! -e "$code_bundle" ] && [ ! -e "$vllm_archive" ]; then echo "SOURCE_SYNC_OK $(hostname) code=$actual_code code_files=$actual_code_files vllm_files=$actual_vllm_files vllm_version=$actual_vllm_version version_sha=$actual_version_sha"; else echo "SOURCE_SYNC_BAD $(hostname) code=$actual_code dirty=$dirty code_files=$actual_code_files vllm_files=$actual_vllm_files vllm_version=$actual_vllm_version version_sha=$actual_version_sha"; fi'
bash "$WORKTREE/scripts/validate_ray_network.sh" bounded 300 \
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$install_sources" >"$RUN_DIR/source_sync.txt" \
  2>"$RUN_DIR/source_sync_ssh.txt"
has_eight_unique_markers "$RUN_DIR/source_sync.txt" SOURCE_SYNC_OK 7 || {
  say "ABORT: exact source reconstruction failed"
  exit 1
}

# Rehydrate only a missing golden file; refuse a non-identical existing file.
# shellcheck disable=SC2016
golden_sync='set -e; idx=${HOSTNAME##*-w-}; source='"$GOLDEN_ROOT"'/golden.rank${idx}.json; dest=/tmp/golden.json; temp=/tmp/golden_sync_$$.tmp; expected_sha='"$GOLDEN_SHA"'; expected_bytes='"$GOLDEN_BYTES"'; disposition=existing; case "$idx" in 0|1|2|3|4|5|6|7) ;; *) echo "GOLDEN_SYNC_BAD $(hostname) rank"; exit 0 ;; esac; mount_source=$(findmnt -T "$source" -n -o SOURCE 2>/dev/null); mount_type=$(findmnt -T "$source" -n -o FSTYPE 2>/dev/null); mount_options=$(findmnt -T "$source" -n -o OPTIONS 2>/dev/null); [ "$mount_source" = driftbench-dsv4-uc ] && [ "$mount_type" = fuse.gcsfuse ] && printf ",%s," "$mount_options" | grep -q ",ro," && [ -r "$source" ] || { echo "GOLDEN_SYNC_BAD $(hostname) source"; exit 0; }; [ "$(sha256sum "$source" | cut -d " " -f 1)" = "$expected_sha" ] && [ "$(stat -c %s "$source")" -eq "$expected_bytes" ] || { echo "GOLDEN_SYNC_BAD $(hostname) source_identity"; exit 0; }; if [ -e "$dest" ]; then [ -r "$dest" ] || { echo "GOLDEN_SYNC_BAD $(hostname) unreadable_destination"; exit 0; }; else disposition=installed; cp -- "$source" "$temp"; chmod 0444 "$temp"; mv "$temp" "$dest"; fi; actual_sha=$(sha256sum "$dest" | cut -d " " -f 1); actual_bytes=$(stat -c %s "$dest"); if [ "$actual_sha" = "$expected_sha" ] && [ "$actual_bytes" -eq "$expected_bytes" ]; then echo "GOLDEN_SYNC_OK $(hostname) sha256=$actual_sha bytes=$actual_bytes disposition=$disposition"; else echo "GOLDEN_SYNC_BAD $(hostname) destination_identity"; fi'
bash "$WORKTREE/scripts/validate_ray_network.sh" bounded 180 \
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$golden_sync" >"$RUN_DIR/golden_sync.txt" \
  2>"$RUN_DIR/golden_sync_ssh.txt"
has_eight_unique_markers "$RUN_DIR/golden_sync.txt" GOLDEN_SYNC_OK 5 || {
  say "ABORT: exact golden manifest is unavailable"
  exit 1
}

# Source, mount, dump-path vacancy and local-space gate before Ray/model work.
# shellcheck disable=SC2016
prereq='code='"$CODE_RUNTIME"'; vllm='"$VLLM_RUNTIME"'; pin='"$ACCEPTED_PIN"'; expected_vllm='"$VLLM_VERSION"'; expected_version_sha='"$VLLM_VERSION_FILE_SHA"'; model='"$MODEL_ROOT"'; dump='"$DUMP_ROOT"'; code_ok=$(git -C "$code" rev-parse HEAD 2>/dev/null); dirty=$(git -C "$code" status --porcelain 2>/dev/null | wc -l); actual_version_sha=$(sha256sum "$vllm/vllm/_version.py" 2>/dev/null | cut -d " " -f 1); actual_vllm=$(PYTHONPATH="$vllm" /home/gianl/vllm-env/bin/python -c "import pathlib,vllm; pathlib.Path(vllm.__file__).resolve().relative_to(pathlib.Path(\"$vllm\").resolve()); print(vllm.__version__)" 2>/dev/null || true); source=$(findmnt -T "$model" -n -o SOURCE 2>/dev/null); type=$(findmnt -T "$model" -n -o FSTYPE 2>/dev/null); options=$(findmnt -T "$model" -n -o OPTIONS 2>/dev/null); free=$(df -BG --output=avail /tmp | tail -1 | tr -d " G"); if [ "$code_ok" = "$pin" ] && [ "$dirty" -eq 0 ] && [ "$actual_version_sha" = "$expected_version_sha" ] && [ "$actual_vllm" = "$expected_vllm" ] && [ -r /tmp/golden.json ] && [ -r "$model/model.safetensors.index.json" ] && [ "$source" = driftbench-dsv4-uc ] && [ "$type" = fuse.gcsfuse ] && printf ",%s," "$options" | grep -q ",ro," && [ ! -e "$dump" ] && [ "$free" -ge 20 ]; then echo "PREREQ_OK $(hostname)"; else echo "PREREQ_BAD $(hostname) code=$code_ok dirty=$dirty vllm=$actual_vllm version_sha=$actual_version_sha mount=$source type=$type options=$options free=$free"; fi'
bash "$WORKTREE/scripts/validate_ray_network.sh" bounded 180 \
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$prereq" >"$RUN_DIR/prereq.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/prereq.txt" PREREQ_OK || {
  say "ABORT: exact accepted runtime/model/disk prerequisite failed"
  exit 1
}

COMMON_ENVS='PYTHONPATH='"$CODE_RUNTIME:$VLLM_RUNTIME"' GLM_ACCEPTED_DB485_COMPILE_ONLY_TAG='"$TAG"' GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=1 GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_DCP_SCATTER_IMPL=flat GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 GLM_WRITE_PROBE=1 GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_CHECKSUM=1 GLM_STATE_HASH_REF=/tmp/golden.json GLM_WK_OOB_DIR='"$MODEL_ROOT"' GLM_WK_OOB_GOLDEN=/tmp/golden.json GLM_DSA_DUMP_TOPK='"$TOPK_PREFIX"' GLM_DSA_DUMP_TOPK_EVENTS=all GLM_DSA_DUMP_TOPK_SKIP_WARMUP=1 GLM_EXPECT_CODE_HASH='"$ACCEPTED_SHORT"' VLLM_XLA_CACHE_PATH='"$XLA_CACHE"' XLA_FLAGS="--xla_dump_to='"$HLO_RAW"' --xla_dump_hlo_as_text --xla_dump_hlo_as_long_text=false --xla_dump_hlo_module_re=jit_step_fun_impl --xla_dump_hlo_pass_re=after_codegen"'
RAYLET_ENVS="$COMMON_ENVS LIBTPU_INIT_ARGS=\"--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false\""
DRIVER_ENVS='NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray OMP_NUM_THREADS=1 HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 DISABLE_WEIGHT_REQUANTIZATION=1 REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 RUNAI_STREAMER_CONCURRENCY=32 RUNAI_STREAMER_MEMORY_LIMIT=34359738368 JAX_SHARE_BINARY_BETWEEN_HOSTS=1 JAX_SHARE_BINARY_BETWEEN_HOSTS_TIMEOUT_MS=120000 '

say "launching the exact accepted runtime; no request will be submitted"
runtime_started=1
RAY_JOIN_TIMEOUT_SECONDS=120 EXTRA_ENVS="$RAYLET_ENVS" TPU_MIN_TOKEN_BUCKET=32 \
  bash "$WORKTREE/scripts/validate_ray_network.sh" bounded 300 \
  bash "$HARNESS_REPO/scripts/launch_glm_32chip.sh" \
  >"$RUN_DIR/launch.log" 2>&1

bash "$WORKTREE/scripts/validate_ray_network.sh" bounded 120 \
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$prereq" >"$RUN_DIR/prereq_post_launch.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/prereq_post_launch.txt" PREREQ_OK || {
  say "ABORT: Ray launch changed an accepted prerequisite"
  exit 1
}

# shellcheck disable=SC2016
env_check='p=$(pgrep -x raylet | head -1); f=/tmp/accepted_compile_env_$$; [ -n "$p" ] && tr "\0" "\n" < /proc/$p/environ > "$f"; if grep -qx "PYTHONPATH='"$CODE_RUNTIME:$VLLM_RUNTIME"'" "$f" && grep -qx "GLM_ACCEPTED_DB485_COMPILE_ONLY_TAG='"$TAG"'" "$f" && grep -qx "GLM_EXPECT_CODE_HASH='"$ACCEPTED_SHORT"'" "$f" && grep -qx "VLLM_XLA_CACHE_PATH='"$XLA_CACHE"'" "$f" && grep -qx "XLA_FLAGS=--xla_dump_to='"$HLO_RAW"' --xla_dump_hlo_as_text --xla_dump_hlo_as_long_text=false --xla_dump_hlo_module_re=jit_step_fun_impl --xla_dump_hlo_pass_re=after_codegen" "$f"; then echo "ENV_OK $(hostname)"; else echo "ENV_BAD $(hostname)"; fi; rm -f "$f"'
bash "$WORKTREE/scripts/validate_ray_network.sh" bounded 120 \
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$env_check" >"$RUN_DIR/raylet_env.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/raylet_env.txt" ENV_OK || {
  say "ABORT: exact compile-only raylet environment mismatch"
  exit 1
}

say "initializing and compiling seven accepted buckets"
# shellcheck disable=SC2086
set +e
LIBTPU_INIT_ARGS='--xla_latency_hiding_scheduler_rerun=5 --xla_tpu_rwb_fusion=false' \
  PYTHONPATH="$CODE_RUNTIME:$VLLM_RUNTIME" \
  GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode GLM_DSA_DCP=1 GLM_DCP=1 \
  GLM_DCP_SCATTER_IMPL=pageloop GLM_DSA_DCP_SCATTER_IMPL=flat \
  GLM_DSA_SCORER=xla GLM_DSA_DCP_PREFILL_ATTN=segment \
  GLM_DSA_BT_WIDTH=owned GLM_DSA_MERGE_IMPL=v2 \
  GLM_DSA_OWNED_SEG_IMPL=v2 GLM_DSA_SEG_GATHER_IMPL=v2 GLM_WRITE_PROBE=1 \
  GLM_PWAL_NAN_CHECK=1 GLM_LOAD_NAN_CHECK=1 GLM_LOAD_CHECKSUM=1 \
  GLM_STATE_HASH_REF=/tmp/golden.json GLM_WK_OOB_DIR="$MODEL_ROOT" \
  GLM_WK_OOB_GOLDEN=/tmp/golden.json GLM_DSA_DUMP_TOPK="$TOPK_PREFIX" \
  GLM_DSA_DUMP_TOPK_EVENTS=all GLM_DSA_DUMP_TOPK_SKIP_WARMUP=1 \
  GLM_EXPECT_CODE_HASH="$ACCEPTED_SHORT" VLLM_XLA_CACHE_PATH="$XLA_CACHE" \
  GLM_ACCEPTED_DB485_COMPILE_ONLY_TAG="$TAG" setsid --wait env $DRIVER_ENVS \
  /home/gianl/vllm-env/bin/python -u "$DRIVER" \
  >"$RUN_DIR/compile.log" 2>&1 &
driver_pid=$!
DRIVER_SESSION=$driver_pid
sleep 1
driver_sid=$(ps -o sid= -p "$driver_pid" 2>/dev/null | tr -d ' ')
if kill -0 "$driver_pid" 2>/dev/null && [[ $driver_sid != "$driver_pid" ]]; then
  say "ABORT: task-owned driver session was not established"
  stop_driver_session
  wait "$driver_pid" 2>/dev/null || true
  exit 1
fi
driver_waited=0
while kill -0 "$driver_pid" 2>/dev/null; do
  sleep 5
  driver_waited=$((driver_waited + 5))
  if [[ $driver_waited -ge 7200 ]]; then
    say "ABORT: compile-only driver exceeded 7200 seconds"
    stop_driver_session
    wait "$driver_pid" 2>/dev/null || true
    exit 124
  fi
done
wait "$driver_pid"
driver_rc=$?
DRIVER_SESSION=""
set -e
[[ $driver_rc -eq 0 ]] || {
  say "ABORT: compile-only driver failed rc=$driver_rc"
  exit "$driver_rc"
}

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR/compile.log" "$TAG" >"$RUN_DIR/fingerprint_precheck.txt" <<'PY'
from pathlib import Path
import sys
from glm_tpu.greenfield.benchmarking.accepted_compile_only_hlo import validate_compile_only_log
report = validate_compile_only_log(Path(sys.argv[1]).read_bytes(), run_tag=sys.argv[2])
print(report["status"])
PY

# Compact every host-local replica. Each module is selected by its exact
# 156-operation row-parallel result shape, not merely by count/order. All eight
# fixed raw-HLO identities must match before worker 0 is a copy source. Full
# per-host audits are collected before any remote evidence is removed.
# shellcheck disable=SC2016
compact='set -euo pipefail; raw='"$HLO_RAW"'; compact='"$HLO_COMPACT"'; host=$(hostname); worker=${host##*-w-}; [[ $host =~ -w-[0-7]$ && $worker =~ ^[0-7]$ ]] || { echo "HLO_BAD $host worker_suffix"; exit 0; }; if [ ! -e "$raw" ]; then echo "HLO_BAD $host absent_root"; exit 0; fi; if [ ! -d "$raw" ] || [ ! -r "$raw" ] || [ ! -x "$raw" ]; then echo "HLO_BAD $host raw_root"; exit 0; fi; total=$(find "$raw" -type f | wc -l); if [ "$total" -eq 0 ]; then echo "HLO_BAD $host empty_root"; exit 0; fi; after_count=$(find "$raw" -type f -name "*after_codegen.txt" | wc -l); [ "$after_count" -eq 7 ] || { echo "HLO_BAD $host after_codegen=$after_count total=$total"; exit 0; }; mkdir "$compact"; find "$raw" -type f -printf "%P\t%s\n" | LC_ALL=C sort >"$compact/raw_hlo_inventory.txt"; : >"$compact/bucket_hlo_map.tsv"; : >"$compact/replica_identity.tsv"; : >"$compact/replica_audit_rows.tmp"; last_module=-1; ok=0; for bucket in 32 64 128 256 512 1024 2048; do pattern="bf16\\[$bucket,6144\\].*all-reduce\\(.*VllmRowParallelLinear/shard_map/psum"; mapfile -t candidates < <(find "$raw" -type f -name "*after_codegen.txt" -exec grep -lE "$pattern" {} + 2>/dev/null || true); [ "${#candidates[@]}" -eq 1 ] || { echo "HLO_BAD $host bucket=$bucket candidates=${#candidates[@]}"; exit 0; }; file=${candidates[0]}; matches=$(grep -cE "$pattern" "$file"); [ "$matches" -eq 156 ] || { echo "HLO_BAD $host bucket=$bucket matches=$matches"; exit 0; }; header=$(head -n 1 "$file"); [[ $header == "HloModule jit_step_fun_impl, is_scheduled=true"* && $header == *"num_partitions=32"* ]] || { echo "HLO_BAD $host bucket=$bucket header"; exit 0; }; base=$(basename "$file"); [[ $base =~ ^module_([0-9]+)\.jit_step_fun_impl\.cl_[0-9]+\.after_codegen\.txt$ ]] || { echo "HLO_BAD $host bucket=$bucket filename=$base"; exit 0; }; module=$((10#${BASH_REMATCH[1]})); [ "$module" -gt "$last_module" ] || { echo "HLO_BAD $host bucket=$bucket module_order"; exit 0; }; last_module=$module; relative=${file#"$raw"/}; raw_size=$(stat -c %s "$file"); raw_sha=$(sha256sum "$file" | awk '\''{print $1}'\''); output="jit_step_fun_impl.m${bucket}.${base}.gz"; gzip -n -c "$file" >"$compact/$output"; gzip -t "$compact/$output"; roundtrip_size=$(gzip -dc "$compact/$output" | wc -c); roundtrip_sha=$(gzip -dc "$compact/$output" | sha256sum | awk '\''{print $1}'\''); [ "$roundtrip_size" -eq "$raw_size" ] && [ "$roundtrip_sha" = "$raw_sha" ] || { echo "HLO_BAD $host bucket=$bucket gzip_roundtrip"; exit 0; }; compressed_size=$(stat -c %s "$compact/$output"); compressed_sha=$(sha256sum "$compact/$output" | awk '\''{print $1}'\''); printf "%s\t%s\t%s\t%s\n" "$bucket" "$relative" "$raw_size" "$output" >>"$compact/bucket_hlo_map.tsv"; printf "%s\t%s\t%s\n" "$bucket" "$raw_size" "$raw_sha" >>"$compact/replica_identity.tsv"; printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$bucket" "$relative" "$raw_size" "$raw_sha" "$output" "$compressed_size" "$compressed_sha" >>"$compact/replica_audit_rows.tmp"; ok=$((ok+1)); done; [ "$ok" -eq 7 ] || { echo "HLO_BAD $host compact_count=$ok"; exit 0; }; identity_sha=$(sha256sum "$compact/replica_identity.tsv" | awk '\''{print $1}'\''); inventory_sha=$(sha256sum "$compact/raw_hlo_inventory.txt" | awk '\''{print $1}'\''); bucket_map_sha=$(sha256sum "$compact/bucket_hlo_map.tsv" | awk '\''{print $1}'\''); { printf "schema\thlo_replica_audit_v2\nhost\t%s\nworker\t%s\nidentity_sha256\t%s\nraw_inventory_sha256\t%s\nbucket_map_sha256\t%s\nbucket\traw_path\traw_bytes\traw_sha256\tsealed_name\tcompressed_bytes\tcompressed_sha256\n" "$host" "$worker" "$identity_sha" "$inventory_sha" "$bucket_map_sha"; cat "$compact/replica_audit_rows.tmp"; } >"$compact/replica_audit.tsv"; rm -f "$compact/replica_audit_rows.tmp"; audit_sha=$(sha256sum "$compact/replica_audit.tsv" | awk '\''{print $1}'\''); find "$raw" -depth -delete; echo "HLO_REPLICA $host count=$ok identity_sha256=$identity_sha audit_sha256=$audit_sha"'
if ! bash "$WORKTREE/scripts/validate_ray_network.sh" bounded 900 \
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$compact" >"$RUN_DIR/hlo_replicas.txt" \
  2>"$RUN_DIR/hlo_replicas_ssh.txt"; then
  preserve_divergent_hlo_replicas || true
  say "ABORT: fleet HLO compaction transport failed"
  exit 1
fi
prepare_hlo_audit_collection || {
  say "ABORT: local HLO audit directory could not be created"
  exit 1
}
audit_copy_rc=0
for worker in 0 1 2 3 4 5 6 7; do
  bash "$WORKTREE/scripts/validate_ray_network.sh" bounded 180 \
    gcloud compute tpus tpu-vm scp --zone "$ZONE" --worker="$worker" \
    "$POD:$HLO_COMPACT/replica_audit.tsv" \
    "$HLO_AUDIT_LOCAL/worker_$worker.tsv" \
    >>"$RUN_DIR/hlo_audit_copy.txt" \
    2>>"$RUN_DIR/hlo_audit_copy_ssh.txt" || audit_copy_rc=1
done
if [[ $audit_copy_rc -ne 0 ]]; then
  preserve_divergent_hlo_replicas || true
  say "ABORT: one or more host HLO audits could not be collected"
  exit 1
fi
set +e
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR/hlo_replicas.txt" "$HLO_AUDIT_LOCAL" "$RUN_DIR/prereq.txt" \
  >"$RUN_DIR/hlo_replica_validation.txt" \
  2>"$RUN_DIR/hlo_replica_validation_stderr.txt" <<'PY'
from pathlib import Path
import sys
from glm_tpu.greenfield.benchmarking.accepted_compile_only_hlo import (
    validate_hlo_replica_evidence,
    validate_worker_host_receipts,
)
expected_hosts = validate_worker_host_receipts(
    Path(sys.argv[3]).read_bytes(), marker="PREREQ_OK"
)
report = validate_hlo_replica_evidence(
    Path(sys.argv[1]).read_bytes(), Path(sys.argv[2]), expected_hosts
)
print(f"{report['status']} canonical_worker={report['canonical_worker']} canonical_host={report['canonical_host']} identity_sha256={report['common_identity_sha256']}")
PY
replica_validation_rc=$?
set -e
if [[ $replica_validation_rc -ne 0 ]]; then
  preserve_divergent_hlo_replicas || true
  say "ABORT: eight-host raw-HLO identity validation failed"
  exit 1
fi
canonical_worker=$(sed -n 's/.* canonical_worker=\([0-7]\) .*/\1/p' \
  "$RUN_DIR/hlo_replica_validation.txt")
[[ $canonical_worker == 0 ]] || {
  say "ABORT: exact eight-replica HLO validation failed"
  preserve_divergent_hlo_replicas || true
  exit 1
}
if ! bash "$WORKTREE/scripts/validate_ray_network.sh" bounded 900 \
  gcloud compute tpus tpu-vm scp --zone "$ZONE" \
  --worker="$canonical_worker" "$POD:$HLO_COMPACT/*" "$HLO_LOCAL/" \
  >"$RUN_DIR/hlo_copy.txt" 2>"$RUN_DIR/hlo_copy_ssh.txt"; then
  preserve_divergent_hlo_replicas || true
  say "ABORT: canonical worker-0 HLO copy failed"
  exit 1
fi
set +e
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR/hlo_replicas.txt" "$HLO_AUDIT_LOCAL" "$HLO_LOCAL" \
  "$RUN_DIR/prereq.txt" \
  >"$RUN_DIR/hlo_canonical_validation.txt" \
  2>"$RUN_DIR/hlo_canonical_validation_stderr.txt" <<'PY'
from pathlib import Path
import sys
from glm_tpu.greenfield.benchmarking.accepted_compile_only_hlo import (
    validate_canonical_hlo_replica,
    validate_hlo_replica_evidence,
    validate_worker_host_receipts,
)
expected_hosts = validate_worker_host_receipts(
    Path(sys.argv[4]).read_bytes(), marker="PREREQ_OK"
)
replicas = validate_hlo_replica_evidence(
    Path(sys.argv[1]).read_bytes(), Path(sys.argv[2]), expected_hosts
)
report = validate_canonical_hlo_replica(Path(sys.argv[3]), replicas)
print(f"{report['status']} identity_sha256={replicas['common_identity_sha256']}")
PY
canonical_validation_rc=$?
set -e
if [[ $canonical_validation_rc -ne 0 ]]; then
  preserve_divergent_hlo_replicas || true
  say "ABORT: canonical worker-0 HLO payload validation failed"
  exit 1
fi

# Model-load integrity is required even though there is no inference request.
# It runs after the HLO payload is safely local so a late integrity refusal
# cannot discard another expensive compiler acquisition.
# shellcheck disable=SC2016
integrity='logs=/tmp/ray/session_latest/logs; checksum=$(grep -Rhs --include="worker-*.out" -E "\[GLM_LOAD_CHECKSUM\].*SUMMARY verified=1882 mismatches=0 skipped=312$" "$logs" 2>/dev/null | tail -1); state=$(grep -Rhs --include="worker-*.out" -E "\[GLM_STATE_HASH\].*manifest VERIFIED leaves=2455 combined=371110325 ref=/tmp/golden.json$" "$logs" 2>/dev/null | tail -1); refusal=$(grep -Rhs --include="worker-*.out" -E "StateHashMismatchError|LoadNanCheckError|PwalNanCheckError|LoadChecksumError|CodeFingerprintMismatchError" "$logs" 2>/dev/null | tail -1); printf "%s\n%s\n" "$checksum" "$state"; if [ -n "$checksum" ] && [ -n "$state" ] && [ -z "$refusal" ]; then echo "INTEGRITY_OK $(hostname)"; else echo "INTEGRITY_BAD $(hostname)"; fi'
bash "$WORKTREE/scripts/validate_ray_network.sh" bounded 180 \
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$integrity" >"$RUN_DIR/fleet_integrity.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/fleet_integrity.txt" INTEGRITY_OK || {
  say "ABORT: accepted model-load integrity failed"
  exit 1
}

stop_owned_runtime
strict_census post
post_census_done=1
runtime_started=0
cleanup_run_owned_sources post
sources_prepared=0
rm -f -- "$CODE_BUNDLE_LOCAL" "$VLLM_ARCHIVE_LOCAL" "$VLLM_TAR_LOCAL"

say "zero-work and source cleanup authenticated; sealing immutable payload"
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python "$SEALER" \
  --run-tag "$TAG" --run-dir "$RUN_DIR" --log "$RUN_DIR/compile.log" \
  --hlo-dir "$HLO_LOCAL" --hlo-replica-receipts "$RUN_DIR/hlo_replicas.txt" \
  --hlo-replica-audits "$HLO_AUDIT_LOCAL" \
  --output "$RUN_DIR/manifest.json" \
  --greenfield-pin "$(git -C "$WORKTREE" rev-parse HEAD)" \
  --accepted-bundle-sha256 "$CODE_BUNDLE_SHA" \
  --vllm-archive-sha256 "$VLLM_ARCHIVE_SHA" \
  --vllm-version "$VLLM_VERSION" \
  --vllm-version-file-sha256 "$VLLM_VERSION_FILE_SHA" \
  --launcher-sha256 "$LAUNCHER_SHA" \
  --network-validator-sha256 "$NETWORK_VALIDATOR_SHA" \
  --callback-certificate-sha256 "$CALLBACK_CERTIFICATE_SHA" \
  --accepted-source-pin "$ACCEPTED_PIN" --vllm-source-pin "$VLLM_PIN" \
  --harness-pin "$HARNESS_PIN" \
  --remote-prefix "$REMOTE_PREFIX" >/dev/null
manifest_sha=$(sha256sum "$RUN_DIR/manifest.json" | awk '{print $1}')
publication_receipt=$(timeout --signal=TERM --kill-after=30 1800 \
  /home/gianl/vllm-env/bin/python "$PUBLISHER" nonterminal \
    --run-dir "$RUN_DIR" --remote-prefix "$REMOTE_PREFIX")
read -r remote_ledger_generation remote_ledger_sha < <(
  /home/gianl/vllm-env/bin/python -c \
    'import json,sys; x=json.loads(sys.argv[1]); print(x["generation"], x["sha256"])' \
    "$publication_receipt"
)
printf '%s\n' \
  'artifact_kind=greenfield_accepted_db485_compile_only_hlo' \
  "run_tag=$TAG" \
  "accepted_code_pin=$ACCEPTED_PIN" \
  "vllm_pin=$VLLM_PIN" \
  "harness_pin=$HARNESS_PIN" \
  "callback_certificate_sha256=$CALLBACK_CERTIFICATE_SHA" \
  "launcher_sha256=$LAUNCHER_SHA" \
  "network_validator_sha256=$NETWORK_VALIDATOR_SHA" \
  "greenfield_pin=$(git -C "$WORKTREE" rev-parse HEAD)" \
  "manifest_sha256=$manifest_sha" \
  "remote_objects_generation=$remote_ledger_generation" \
  "remote_objects_sha256=$remote_ledger_sha" \
  'db_run_id=None' \
  'numerical_claim=false' \
  'performance_claim=false' \
  'gate_d_claim=false' \
  "remote_prefix=$REMOTE_PREFIX" >"$RUN_DIR/SUCCESS"
terminal_publication_started=1
timeout --signal=TERM --kill-after=30 300 \
  /home/gianl/vllm-env/bin/python "$PUBLISHER" terminal \
    --run-dir "$RUN_DIR" --remote-prefix "$REMOTE_PREFIX" >/dev/null
terminal_success=1
trap - EXIT
echo "SUCCESS DB485 executable class matched; seven bucket-bound HLOs sealed; no generation or Gate-D claim"
