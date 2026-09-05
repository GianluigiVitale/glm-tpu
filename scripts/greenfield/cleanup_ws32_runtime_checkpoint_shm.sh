#!/usr/bin/env bash
# Remove the tmpfs-resident WS32 checkpoint from every host and the controller (exact path only).
set -euo pipefail
[[ ${GLM_GREENFIELD_WS32_SHM_CLEANUP:-0} == 1 ]] || {
  echo "WS32 shm cleanup is default-off; set GLM_GREENFIELD_WS32_SHM_CLEANUP=1" >&2
  exit 2
}
readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly SEALED_TAG=greenfield_ws32_runtime_pack_20260815T214050854386790Z
readonly SHM_ROOT=/dev/shm/glm-ws32-runtime/$SEALED_TAG
exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || { echo "ABORT: another protected workflow holds the fleet lease" >&2; exit 1; }
# shellcheck disable=SC2016
command='set -euo pipefail; shm='"$SHM_ROOT"'; if pgrep -af "[r]un_short_decoder_ws32[.]py|[p]ack_ws32_runtime_checkpoint[.]py" >/dev/null; then echo "SHM_CLEAN_BUSY $(hostname)"; exit 1; fi; if [[ -d $shm && $shm == /dev/shm/glm-ws32-runtime/greenfield_ws32_runtime_pack_* ]]; then rm -rf -- "$shm" "$shm.identity.json"; fi; rmdir /dev/shm/glm-ws32-runtime 2>/dev/null || true; [[ ! -e $shm ]] && echo "SHM_CLEAN_OK $(hostname)"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all --command="$command"
if [[ -d $SHM_ROOT ]]; then rm -rf -- "$SHM_ROOT" "$SHM_ROOT.identity.json"; fi
rmdir /dev/shm/glm-ws32-runtime 2>/dev/null || true
echo "SHM_CLEAN_OK controller"
