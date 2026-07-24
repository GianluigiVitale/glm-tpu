#!/bin/bash
# disk_watchdog.sh — pod-wide disk-pressure guard (ops debt from the gate2
# postmortem, RESEARCH_LOG 2026-07-13 15:40: FOUR disk incidents this
# campaign — w-2 twice, w-0, and w-4 at 0 bytes free, which killed every
# raylet join for 4h AND confounded the engine-lottery attribution).
#
# Disk pressure is a FIRST-CLASS failure mode: a raylet on a full disk dies
# on join; an engine on a nearly-full disk degrades in ways that mimic
# model-level defects. No long run starts, and no verdict is trusted,
# without this check.
#
# Modes:
#   check              one-shot pre-flight over all 8 hosts. Exit 0 = all
#                      hosts >= MIN_FREE_GB; exit 1 = any host below (names
#                      it). Orchestrators call this before EVERY launch.
#   watch              poll loop (INTERVAL_S, default 120 s). On any host
#                      dropping below MIN_FREE_GB it writes the flag file
#                      ~/glm-run/DISK_ALERT (host + free GB + timestamp,
#                      appended) and keeps polling. Run watchers read the
#                      flag to classify a failure as INFRA, never a model
#                      verdict. WARN_FREE_GB breaches are logged only.
#
# Env knobs: MIN_FREE_GB (default 15), WARN_FREE_GB (default 25),
#            INTERVAL_S (default 120), ALERT_FILE (default ~/glm-run/DISK_ALERT)
set -u

ZONE=us-central2-b
POD=db-v4-64-od
MIN_FREE_GB="${MIN_FREE_GB:-15}"
WARN_FREE_GB="${WARN_FREE_GB:-25}"
INTERVAL_S="${INTERVAL_S:-120}"
ALERT_FILE="${ALERT_FILE:-$HOME/glm-run/DISK_ALERT}"

MODE="${1:-check}"

poll_once() {
  # One line per host: "<hostname> <free-GB>". gcloud runs the 8 ssh
  # sessions; a host that cannot even answer is itself an alert.
  timeout 120 gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" \
    --worker=all \
    --command='echo "$(hostname) $(df -B1G --output=avail / | tail -1 | tr -d " ")"' \
    2>/dev/null | grep -E "^t1v-" | sort
}

classify() {
  # stdin: poll_once lines. Prints per-host status; returns 1 on any BREACH.
  local rc=0 host free
  local n=0
  while read -r host free; do
    n=$((n + 1))
    if [ "${free:-0}" -lt "$MIN_FREE_GB" ]; then
      echo "BREACH $host ${free}G free (< ${MIN_FREE_GB}G)"
      rc=1
    elif [ "$free" -lt "$WARN_FREE_GB" ]; then
      echo "WARN   $host ${free}G free (< ${WARN_FREE_GB}G)"
    else
      echo "OK     $host ${free}G free"
    fi
  done
  if [ "$n" -lt 8 ]; then
    echo "BREACH only $n/8 hosts answered the disk poll (unreachable host = alert)"
    rc=1
  fi
  return "$rc"
}

case "$MODE" in
  check)
    if poll_once | classify; then
      echo "disk check: ALL 8 HOSTS OK (>= ${MIN_FREE_GB}G free)"
      exit 0
    else
      echo "disk check: FAILED — clear space (scripts/dump_archiver.sh) before launching." >&2
      exit 1
    fi
    ;;
  watch)
    mkdir -p "$(dirname "$ALERT_FILE")"
    # 2026-07-24: an ssh/control-plane transient made three consecutive
    # polls reach 0/8 hosts; the old loop wrote BREACH lines immediately
    # and a healthy gate depth was aborted as INFRA on a false alarm
    # (every host had >50G free). A failed POLL is not a disk verdict:
    # only a host that ANSWERS with low disk breaches immediately;
    # unreachability escalates only after UNREACH_MAX consecutive misses
    # (default 5 = ~10 min sustained — a genuinely dead host still trips).
    UNREACH_MAX="${UNREACH_MAX:-5}"
    unreach=0
    echo "disk watch: every ${INTERVAL_S}s; alert -> $ALERT_FILE (min ${MIN_FREE_GB}G; unreach escalates at ${UNREACH_MAX} consecutive)"
    while true; do
      raw=$(poll_once)
      n=$(echo "$raw" | grep -c "^t1v-" || true)
      if [ "${n:-0}" -lt 8 ]; then
        unreach=$((unreach + 1))
        echo "disk watch: poll reached only ${n:-0}/8 hosts (${unreach}/${UNREACH_MAX} consecutive) — ssh transient, NOT a disk verdict" >&2
        if [ "$unreach" -ge "$UNREACH_MAX" ]; then
          {
            echo "=== DISK_ALERT $(date -u +%FT%TZ) ==="
            echo "BREACH pod unreachable for ${unreach} consecutive polls (~$((unreach * INTERVAL_S / 60)) min sustained)"
          } >> "$ALERT_FILE"
          unreach=0
        fi
      else
        unreach=0
        out=$(echo "$raw" | classify) || {
          {
            echo "=== DISK_ALERT $(date -u +%FT%TZ) ==="
            echo "$out" | grep -E "^BREACH"
          } >> "$ALERT_FILE"
          echo "$out" | grep -E "^BREACH" >&2
        }
      fi
      sleep "$INTERVAL_S"
    done
    ;;
  *)
    echo "usage: disk_watchdog.sh [check|watch]" >&2
    exit 2
    ;;
esac
