#!/bin/bash
# Same-region repository mirror for the GLM work. Called every 5 min from cron.
# The TPU and this bucket are both in us-central2. Never route this mirror
# through the legacy Europe gcsfuse mount.

set -u
set -o pipefail

BUCKET=gs://driftbench-dsv4-uc
LOG=/home/gianl/sync-glm-tpu.log
MIN_FILES=50
EXCLUDE='(^|/)bench/results[.]db-(journal|wal|shm)$'
EVIDENCE_SOURCE=/home/gianl/glm-tpu-gate-d-evidence
EVIDENCE_REL=repos/glm-tpu-gate-d-evidence
EVIDENCE_VALIDATOR=scripts/greenfield/validate_gate_d_evidence_ref.py
EVIDENCE_MIRROR=scripts/greenfield/mirror_gate_d_evidence.py

PAIRS=(
  "/home/gianl/glm-tpu:repos/glm-tpu"
  "/home/gianl/glm-tpu-topology-rewrite:repos/glm-tpu-topology-rewrite"
  "/home/gianl/glm-tpu-gate-d-evidence:repos/glm-tpu-gate-d-evidence"
  "/home/gianl/tpu-inference-glm-baseline:repos/tpu-inference-glm-baseline"
  "/home/gianl/tpu-inference-greenfield-layer1-rms-input-observer:repos/tpu-inference-greenfield-layer1-rms-input-observer"
)

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$LOG"; }

active_evidence_export=""
cleanup_evidence_export() {
    target="$1"
    case "$target" in
        /home/gianl/glm-run/gate-d-evidence-mirror.????????)
            if [ -d "$target" ] && [ ! -L "$target" ]; then rm -rf -- "$target"; fi
            ;;
        *)
            log "REFUSE unsafe evidence export cleanup target: $target"
            return 1
            ;;
    esac
}
cleanup_on_exit() {
    if [ -n "$active_evidence_export" ]; then
        cleanup_evidence_export "$active_evidence_export"
    fi
}
trap cleanup_on_exit EXIT
trap 'exit 130' INT TERM

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
    sync_src="$src"
    evidence_export=""

    if [ ! -d "$src" ]; then
        log "SKIP $rel: source $src does not exist"
        continue
    fi
    if [ "$rel" = "$EVIDENCE_REL" ]; then
        canonical=$(readlink -f -- "$src" 2>/dev/null)
        if [ "$src" != "$EVIDENCE_SOURCE" ] || [ "$canonical" != "$EVIDENCE_SOURCE" ]; then
            log "SKIP $rel: noncanonical evidence source"
            continue
        fi
        validator="$src/$EVIDENCE_VALIDATOR"
        if [ ! -f "$validator" ] || [ -L "$validator" ]; then
            log "SKIP $rel: evidence validator is absent or unsafe"
            continue
        fi
        if ! evidence_report=$(
            /usr/bin/env -i \
                HOME=/home/gianl LANG=C LC_ALL=C PATH=/usr/bin:/bin \
                /usr/bin/python3 -I -S -B "$validator" \
                --repository "$src" --mode committed --require-remote-equal 2>&1
        ); then
            log "SKIP $rel: evidence validator refused: $(echo "$evidence_report" | tail -1)"
            continue
        fi
        mirror_helper="$src/$EVIDENCE_MIRROR"
        if [ ! -f "$mirror_helper" ] || [ -L "$mirror_helper" ]; then
            log "SKIP $rel: evidence mirror helper is absent or unsafe"
            continue
        fi
        evidence_export=$(mktemp -d /home/gianl/glm-run/gate-d-evidence-mirror.XXXXXXXX) || {
            log "SKIP $rel: cannot create evidence export"
            continue
        }
        active_evidence_export="$evidence_export"
        if ! mirror_report=$(
            /usr/bin/printf '%s\n' "$evidence_report" | \
                /usr/bin/env -i \
                HOME=/home/gianl LANG=C LC_ALL=C PATH=/usr/bin:/bin \
                /usr/bin/python3 -I -S -B "$mirror_helper" \
                --repository "$src" --output-root "$evidence_export" 2>&1
        ); then
            cleanup_evidence_export "$evidence_export"
            active_evidence_export=""
            log "SKIP $rel: authenticated append-only evidence mirror failed: $(echo "$mirror_report" | tail -1)"
            continue
        fi
        log "OK $rel: $mirror_report"
        cleanup_evidence_export "$evidence_export"
        active_evidence_export=""
        continue
    fi
    n=$(find "$sync_src" -type f 2>/dev/null | wc -l)
    if [ "$n" -lt "$MIN_FILES" ]; then
        log "SKIP $rel: source has only $n files (< $MIN_FILES) - refusing -d"
        if [ -n "$evidence_export" ]; then
            cleanup_evidence_export "$evidence_export"
            active_evidence_export=""
        fi
        continue
    fi

    if out=$(gsutil -m rsync -r -d -e -x "$EXCLUDE" "$sync_src" "$dst" 2>&1); then
        log "OK $rel ($n files)"
    else
        rc=$?
        log "RSYNC FAILED $rel rc=$rc: $(echo "$out" | tail -2 | tr '\n' ' ')"
    fi
    if [ -n "$evidence_export" ]; then
        cleanup_evidence_export "$evidence_export"
        active_evidence_export=""
    fi
done

exit $rc
