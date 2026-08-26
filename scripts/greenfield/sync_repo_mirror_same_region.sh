#!/bin/bash
# Same-region repository mirror for the GLM work. Called every 5 min from cron.
# The TPU and this bucket are both in us-central2. Never route this mirror
# through the legacy Europe gcsfuse mount.

set -u

BUCKET=gs://driftbench-dsv4-uc
LOG=/home/gianl/sync-glm-tpu.log
MIN_FILES=50
EXCLUDE='(^|/)bench/results[.]db-(journal|wal|shm)$'

PAIRS=(
  "/home/gianl/glm-tpu:repos/glm-tpu"
  "/home/gianl/glm-tpu-topology-rewrite:repos/glm-tpu-topology-rewrite"
)

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$LOG"; }

# Fail closed if the configured destination ever stops being the exact
# us-central2 bucket. This check occurs before any command carrying -d.
location=$(gcloud storage buckets describe "$BUCKET" --format='value(location)' 2>/dev/null)
if [ "$location" != "US-CENTRAL2" ]; then
    log "SKIP: $BUCKET location is '$location', expected US-CENTRAL2"
    exit 0
fi

rc=0
for pair in "${PAIRS[@]}"; do
    src="${pair%%:*}"
    rel="${pair#*:}"
    dst="$BUCKET/$rel"

    if [ ! -d "$src" ]; then
        log "SKIP $rel: source $src does not exist"
        continue
    fi
    n=$(find "$src" -type f 2>/dev/null | wc -l)
    if [ "$n" -lt "$MIN_FILES" ]; then
        log "SKIP $rel: source has only $n files (< $MIN_FILES) - refusing -d"
        continue
    fi

    if out=$(gsutil -m rsync -r -d -e -x "$EXCLUDE" "$src" "$dst" 2>&1); then
        log "OK $rel ($n files)"
    else
        rc=$?
        log "RSYNC FAILED $rel rc=$rc: $(echo "$out" | tail -2 | tr '\n' ' ')"
    fi
done

exit $rc
