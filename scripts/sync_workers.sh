#!/bin/bash
# Sync the working branches onto ALL 8 hosts of the db-v4-64-od pod so every
# Ray worker runs the same code (each host's raylet/engine imports its own
# ~/tpu-inference checkout — cross-host drift causes silent divergence).
#
# Idempotent + NO force operations: fetch, checkout (a no-op if already on the
# branch; creates the tracking branch on first run), then `git pull --ff-only`
# (refuses to rewrite local history — a diverged or dirty checkout fails LOUDLY
# instead of being clobbered). Each host echoes
#   "<hostname>: <repo> @ <short-hash>"
# so drift is immediately visible: all 8 lines must show the SAME hash per repo.
#
# Usage: bash ~/glm-tpu/scripts/sync_workers.sh
#        TPU_INFERENCE_BRANCH=glm-5.2-v4-next bash ~/glm-tpu/scripts/sync_workers.sh
#
# TPU_INFERENCE_BRANCH selects the fork branch (default glm-5.2-v4 = mainline;
# the Stage-2 staging switch uses glm-5.2-v4-next — docs/11-pod-runbook.md §2).
# The branch MUST exist on origin (push it first): the workers pull from
# origin, never from this host.
set -u

ZONE=us-central2-b
POD=db-v4-64-od

TPU_INFERENCE_BRANCH="${TPU_INFERENCE_BRANCH:-glm-5.2-v4}"
GLM_TPU_BRANCH="${GLM_TPU_BRANCH:-main}"

FAILED=0

# Escaped $(...) run on the REMOTE host; ~ expands remotely (same user/layout
# on all 8 hosts).
SYNC_TPU_INFERENCE="cd ~/tpu-inference && git fetch origin && git checkout $TPU_INFERENCE_BRANCH && git pull --ff-only origin $TPU_INFERENCE_BRANCH && echo \"\$(hostname): tpu-inference @ \$(git rev-parse --short HEAD) dirty=\$(git status --porcelain | wc -l)\""
SYNC_GLM_TPU="if [ -d ~/glm-tpu ]; then cd ~/glm-tpu && git fetch origin && git checkout $GLM_TPU_BRANCH && git pull --ff-only origin $GLM_TPU_BRANCH && echo \"\$(hostname): glm-tpu @ \$(git rev-parse --short HEAD) dirty=\$(git status --porcelain | wc -l)\"; else echo \"\$(hostname): glm-tpu ABSENT (only worker 0 / the driver needs it)\"; fi"

echo "[1/2] sync ~/tpu-inference -> $TPU_INFERENCE_BRANCH on all workers"
if ! gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$SYNC_TPU_INFERENCE"; then
  echo "WARNING: tpu-inference sync failed on at least one worker — fix before launching." >&2
  FAILED=1
fi

echo "[2/2] sync ~/glm-tpu -> $GLM_TPU_BRANCH on all workers"
if ! gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$SYNC_GLM_TPU"; then
  echo "WARNING: glm-tpu sync failed on at least one worker — fix before launching." >&2
  FAILED=1
fi

echo "sync done — verify all 8 hashes above match per repo."
exit "$FAILED"
