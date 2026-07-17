#!/bin/bash
# dump_archiver.sh — quota'd dump archival: GCS or /dev/null, never "parked".
#
# Ops debt from the gate2 postmortem (RESEARCH_LOG 2026-07-13 15:40): dump
# sets parked on worker /tmp and home dirs caused FOUR disk incidents —
# 17G of pageloop-era files on w-2 (invisible to top-N size listings),
# 55G of probe archives on w-4 (raylet died on every join for 4h). The
# standing rule this script enforces: a dump either goes to
# gs://driftbench-dsv4-uc (us-central2 — NEVER the EU bucket) immediately,
# or it is deleted. Local disks are not archives.
#
# Modes (all operate on EVERY pod host via gcloud --worker=all):
#   archive <tag> [--last-step-only]
#       tar each host's dump files (default glob: /tmp/dcp_*.npz
#       /tmp/glm_flight_* /tmp/dsa_topk_*; override with SRC_GLOB) to
#       gs://driftbench-dsv4-uc/dumps/<tag>/<host>.tar.gz, then DELETE the
#       local files. --last-step-only keeps only the HIGHEST-step npz per
#       (prefix,proc) — the post-prefill cache, the only file the byte-diff
#       protocol needs (deterministic prefill ⇒ compare final caches) —
#       and deletes the rest unarchived (per-step intermediates).
#   purge
#       delete the dump globs everywhere without archiving (for dumps whose
#       verdict is already recorded — "decisive sets in GCS, protocol runs
#       are regenerable", the 2026-07-11 hygiene rule).
#   du
#       show per-host usage of the dump globs (the incident on w-2 was
#       invisible to naive listings — this counts the globs directly).
set -u

ZONE=us-central2-b
POD=db-v4-64-od
BUCKET=gs://driftbench-dsv4-uc/dumps
SRC_GLOB="${SRC_GLOB:-/tmp/dcp_*.npz /tmp/glm_flight_* /tmp/dsa_topk_*}"

MODE="${1:-}"
TAG="${2:-}"

remote() { # run $1 on all 8 hosts
  gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$1"
}

case "$MODE" in
  du)
    remote 'T=0; for f in '"$SRC_GLOB"'; do [ -e "$f" ] && T=$((T + $(du -sm "$f" 2>/dev/null | cut -f1))); done; echo "$(hostname): ${T}M in dump globs"' \
      | grep -E "^t1v-" | sort
    ;;
  purge)
    remote 'rm -f '"$SRC_GLOB"' 2>/dev/null; echo "$(hostname): purged"' \
      | grep -E "^t1v-" | sort
    ;;
  archive)
    [ -n "$TAG" ] || { echo "usage: dump_archiver.sh archive <tag> [--last-step-only]" >&2; exit 2; }
    LAST_ONLY=""
    [ "${3:-}" = "--last-step-only" ] && LAST_ONLY=1
    # Remote per-host: (optionally thin to last step per prefix+proc), tar,
    # stream to GCS (same-region), then delete the local originals. The
    # dump filename convention is <prefix>.step<NNNN>.proc<P>.npz
    # (dcp_cache_dump.py) — the last-step filter groups on everything but
    # the step number.
    CMD='set -u
H=$(hostname)
FILES=$(ls '"$SRC_GLOB"' 2>/dev/null || true)
[ -z "$FILES" ] && { echo "$H: nothing to archive"; exit 0; }
if [ -n "'"$LAST_ONLY"'" ]; then
  KEEP=$(echo "$FILES" | grep -E "\.step[0-9]+\.proc[0-9]+\.npz$" \
    | sed -E "s/\.step[0-9]+\.(proc[0-9]+\.npz)$/ \1/" | sort -u \
    | while read -r pre proc; do
        ls "${pre}".step*."${proc}" 2>/dev/null | sort -V | tail -1
      done)
  NONSTEP=$(echo "$FILES" | grep -vE "\.step[0-9]+\.proc[0-9]+\.npz$" || true)
  DROP=$(echo "$FILES" | grep -vxF -f <(printf "%s\n%s\n" "$KEEP" "$NONSTEP" | sed "/^$/d") || true)
  [ -n "$DROP" ] && echo "$DROP" | xargs -r rm -f
  FILES=$(printf "%s\n%s\n" "$KEEP" "$NONSTEP" | sed "/^$/d")
fi
N=$(echo "$FILES" | wc -l)
echo "$FILES" | tar czf - -T - 2>/dev/null \
  | gcloud storage cp - "'"$BUCKET/$TAG"'/$H.tar.gz" >/dev/null 2>&1 \
  && { echo "$FILES" | xargs -r rm -f; echo "$H: archived $N files -> '"$BUCKET/$TAG"'/$H.tar.gz + purged"; } \
  || echo "$H: ARCHIVE FAILED — local files KEPT (do not purge blind)"'
    remote "$CMD" | grep -E "^t1v-" | sort
    ;;
  *)
    echo "usage: dump_archiver.sh {archive <tag> [--last-step-only]|purge|du}" >&2
    exit 2
    ;;
esac
