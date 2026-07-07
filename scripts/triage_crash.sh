#!/usr/bin/env bash
# triage_crash.sh — automated post-mortem for GLM-5.2 pod-run crashes.
#
# Automates what used to be hand-grepped after every roaming core-halt
# ('Error Interrupt' / RuntimeUnexpectedCoreHalt on a single host):
#
#   1. LOG PARSE (works on any existing run log, e.g. ~/glm-run/probeA5.log):
#      - halted host IP(s) + first fatal timestamp,
#      - deduped Python-level error lines (JaxRuntimeError / core-halt),
#      - per-host LAST observed serving step + composition from the
#        [OBSERVE_COMPILES] lines (needs DSV4_OBSERVE_COMPILES=1 on the run;
#        Ray DEDUPLICATES repeated log lines across the cluster, so per-host
#        last-step from the log is a LOWER BOUND — the flight recorder is
#        the authoritative source),
#      - jit program names mentioned near the fatal region (hints: the log
#        is asynchronously interleaved),
#      - RESOURCE / ICI lines, deduped with counts.
#
#   2. FLIGHT-RECORDER FETCH (skipped with --no-fetch): tails the last 100
#      lines of /tmp/glm_flight_*.jsonl from EVERY pod host (the black box
#      written by the fork's runner/flight_recorder.py when the run had
#      GLM_FLIGHT_RECORDER=1) and aligns them: per-host last step, the last
#      common step, and which host stopped logging EARLIER than its peers —
#      that host is the diverged/halted one.
#
# Usage:
#   bash scripts/triage_crash.sh <run_log> [--no-fetch]
#
# Env overrides: GLM_POD (db-v4-64-od), GLM_ZONE (us-central2-b),
#                GLM_FLIGHT_GLOB (/tmp/glm_flight_*.jsonl)
set -uo pipefail

POD="${GLM_POD:-db-v4-64-od}"
ZONE="${GLM_ZONE:-us-central2-b}"
FLIGHT_GLOB="${GLM_FLIGHT_GLOB:-/tmp/glm_flight_*.jsonl}"

usage() { echo "usage: bash scripts/triage_crash.sh <run_log> [--no-fetch]" >&2; }

LOG=""
NO_FETCH=0
for arg in "$@"; do
  case "$arg" in
    --no-fetch) NO_FETCH=1 ;;
    -h|--help) usage; exit 0 ;;
    *) LOG="$arg" ;;
  esac
done
if [ -z "$LOG" ] || [ ! -f "$LOG" ]; then
  usage; echo "error: run log not found: '$LOG'" >&2; exit 2
fi

CLEAN="$(mktemp)"
FLIGHT_RAW="$(mktemp)"
trap 'rm -f "$CLEAN" "$FLIGHT_RAW" "$FLIGHT_RAW.err"' EXIT
# Strip ANSI color so every regex below sees plain text.
sed 's/\x1b\[[0-9;]*m//g' "$LOG" > "$CLEAN"

echo "== triage: $LOG =="
echo

# ── 1. halted host(s) + first fatal timestamp ────────────────────────────────
echo "-- fatal 'Error Interrupt' (halted host = the TPU that died) --"
awk '
  /Received Error Interrupt! fatal: true/ {
    ip = "worker-0(local)"
    if (match($0, /ip=[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/))
      ip = substr($0, RSTART + 3, RLENGTH - 3)
    if (!(ip in first)) {
      ts = "?"
      if (match($0, /[EWIF][0-9][0-9][0-9][0-9] [0-9:]+\.[0-9]+/))
        ts = substr($0, RSTART, RLENGTH)
      first[ip] = ts; order[++n] = ip
    }
    count[ip]++
  }
  END {
    for (i = 1; i <= n; i++) {
      ip = order[i]
      printf "  %-18s first fatal at %s  (%d interrupt lines)\n", \
             ip, first[ip], count[ip]
    }
  }' "$CLEAN"
HALTED_IPS="$(awk '
  /Received Error Interrupt! fatal: true/ {
    ip = "worker-0(local)"
    if (match($0, /ip=[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/))
      ip = substr($0, RSTART + 3, RLENGTH - 3)
    if (!(ip in seen)) { seen[ip] = 1; printf "%s ", ip }
  }' "$CLEAN")"
if [ -z "$HALTED_IPS" ]; then
  echo "  (no fatal 'Error Interrupt' lines in this log)"
fi
FATAL_LINE="$(grep -n -m1 "Received Error Interrupt! fatal: true" "$CLEAN" | cut -d: -f1 || true)"
echo

# ── 2. Python-level error lines, deduped ─────────────────────────────────────
echo "-- error summary (deduped) --"
ERRS="$(grep -E "JaxRuntimeError|RuntimeUnexpectedCoreHalt|Bad StatusOr" "$CLEAN" \
  | sed -E 's/^.*(jax\.errors\.|ray\.exceptions\.)/\1/' \
  | sed -E 's/^.*(E[0-9]{4}: RuntimeUnexpectedCoreHalt)/\1/' \
  | sed -E 's/[[:space:]]+$//' | sort -u | head -6 || true)"
if [ -n "$ERRS" ]; then echo "  ${ERRS//$'\n'/$'\n'  }"; else echo "  (none found)"; fi
echo

# ── 3. per-host last observed step + composition (from the run log) ─────────
echo "-- per-host last observed step (log [OBSERVE_COMPILES]; Ray-deduped => lower bound) --"
awk -v halted="$HALTED_IPS" '
  BEGIN { split(halted, hs, " "); for (i in hs) if (hs[i] != "") hset[hs[i]] = 1 }
  /\[OBSERVE_COMPILES\] step=/ {
    ip = "worker-0(local)"
    if (match($0, /ip=[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/))
      ip = substr($0, RSTART + 3, RLENGTH - 3)
    if (match($0, /step=[0-9]+/)) s = substr($0, RSTART + 5, RLENGTH - 5) + 0
    else next
    comp = ""
    if (match($0, /step=[0-9]+ [^:]+:/)) {
      comp = substr($0, RSTART, RLENGTH - 1); sub(/^step=[0-9]+ /, "", comp)
    }
    if (!(ip in last) || s > last[ip]) { last[ip] = s; lcomp[ip] = comp }
    if (!(ip in seen)) { seen[ip] = 1; order[++n] = ip }
  }
  END {
    if (n == 0) {
      print "  (no [OBSERVE_COMPILES] lines — run had DSV4_OBSERVE_COMPILES off;"
      print "   rely on the flight-recorder section below)"
      exit
    }
    mx = 0
    for (i = 1; i <= n; i++) if (last[order[i]] > mx) mx = last[order[i]]
    for (i = 1; i <= n; i++) {
      ip = order[i]
      mark = (ip in hset) ? "  <-- HALTED HOST" : ""
      printf "  %-18s last step=%-6d %-32s (delta vs max: -%d)%s\n", \
             ip, last[ip], lcomp[ip], mx - last[ip], mark
    }
    printf "  cluster max step in log: %d\n", mx
  }' "$CLEAN"
echo

# ── 4. jit program names near the fatal region (hints) ───────────────────────
echo "-- jit program names within the fatal window (interleaved log => hints only) --"
if [ -n "$FATAL_LINE" ]; then
  START=$(( FATAL_LINE > 200 ? FATAL_LINE - 200 : 1 ))
  END=$(( FATAL_LINE + 400 ))
  PROGS="$(sed -n "${START},${END}p" "$CLEAN" \
    | grep -oE "(HLO module |module: |DISABLED\. )jit_[A-Za-z0-9_]*" \
    | sed -E 's/^(HLO module |module: |DISABLED\. )//' \
    | sort | uniq -c | sort -rn | head -8 || true)"
  if [ -n "$PROGS" ]; then echo "  ${PROGS//$'\n'/$'\n'  }"; else echo "  (none in window)"; fi
else
  echo "  (no fatal line to anchor the window)"
fi
echo

# ── 5. RESOURCE / ICI lines, deduped ─────────────────────────────────────────
echo "-- RESOURCE / ICI lines (deduped, with counts) --"
RESICI="$(grep -E "RESOURCE_EXHAUSTED|Resource exhausted|[^A-Za-z]ICI[^A-Za-z]|ici_" "$CLEAN" \
  | sed -E 's/^\(EngineCore[^)]*\) //g; s/^\(pid=[^)]*\) //; s/^\(Ray[^)]*\) //; s/^[EWIF][0-9]{4} [0-9:.]+[[:space:]]+[0-9]+ //; s/[[:space:]]+$//' \
  | sort | uniq -c | sort -rn | head -10 || true)"
if [ -n "$RESICI" ]; then echo "  ${RESICI//$'\n'/$'\n'  }"; else echo "  (none found)"; fi
echo

# ── 6. flight recorder: fetch + cross-host step alignment ────────────────────
echo "-- flight recorder (GLM_FLIGHT_RECORDER=1 runs; ${FLIGHT_GLOB} on every host) --"
if [ "$NO_FETCH" -eq 1 ]; then
  echo "  (skipped: --no-fetch)"
  exit 0
fi
# shellcheck disable=SC2016  # $(hostname) must expand on the REMOTE host
if ! timeout 180 gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command='echo "FLIGHT_HOST $(hostname)"; tail -n 100 '"$FLIGHT_GLOB"' 2>/dev/null || echo "NO_FLIGHT_FILES"' \
    > "$FLIGHT_RAW" 2> "$FLIGHT_RAW.err"; then
  echo "  (fetch FAILED — pod unreachable or gcloud error; rerun with --no-fetch"
  echo "   for the log-only report. stderr tail:)"
  tail -3 "$FLIGHT_RAW.err" 2>/dev/null | sed 's/^/   | /'
  exit 1
fi
awk -v halted="$HALTED_IPS" '
  BEGIN { split(halted, hs, " "); for (i in hs) if (hs[i] != "") hset[hs[i]] = 1 }
  /^FLIGHT_HOST / { host = $2; key = ""; if (!(host in horder)) { horder[host] = ++hn; hosts[hn] = host }; next }
  /^NO_FLIGHT_FILES/ { if (host != "") nofiles[host] = 1; next }
  /^==> .* <==$/ { key = host "|" $2; next }
  /"ev":"step"/ {
    if (host == "") next
    k = (key != "") ? key : host "|(only-file)"
    if (match($0, /"ts":[0-9.]+/))   ts = substr($0, RSTART + 5, RLENGTH - 5) + 0
    if (match($0, /"step":[0-9]+/))  st = substr($0, RSTART + 7, RLENGTH - 7) + 0
    else next
    lastts[k] = ts; laststep[k] = st; lastline[k] = $0; hostof[k] = host
    next
  }
  /"ev":"/ {  # lifecycle events: remember the newest per host|file
    if (host == "") next
    k = (key != "") ? key : host "|(only-file)"
    if (match($0, /"ev":"[a-z_]+"/))
      lastev[k] = substr($0, RSTART + 6, RLENGTH - 7)
    hostof[k] = host
    next
  }
  function field(line, name,    r) {
    r = "\"" name "\":[-0-9]+"
    if (match(line, r)) return substr(line, RSTART + length(name) + 3,
                                      RLENGTH - length(name) - 3) + 0
    return -1
  }
  END {
    if (hn == 0) { print "  (no host sections in fetch output)"; exit }
    # newest file per host wins (stale files from earlier runs lose on ts)
    for (k in lastts) { h = hostof[k]
      if (!(h in best) || lastts[k] > lastts[best[h]]) best[h] = k }
    mx = -1; mn = -1; nsteps = 0
    for (i = 1; i <= hn; i++) { h = hosts[i]
      if (h in best) { s = laststep[best[h]]; nsteps++
        if (s > mx) mx = s
        if (mn < 0 || s < mn) mn = s } }
    for (i = 1; i <= hn; i++) {
      h = hosts[i]
      if (!(h in best)) {
        printf "  %-24s NO flight data%s\n", h,
               (h in nofiles) ? " (no files — was GLM_FLIGHT_RECORDER=1 baked into the raylet env?)" : ""
        continue
      }
      k = best[h]; l = lastline[k]
      mark = ""
      if (mx >= 0 && laststep[k] < mx) mark = "  <-- stopped " (mx - laststep[k]) " step(s) EARLY (diverged?)"
      printf "  %-24s last step=%-7d reqs=%-3d real=%-5d padded=%-5d decode_only=%d chunks=%d finished=%d%s\n", \
             h, laststep[k], field(l, "num_reqs"), field(l, "real_tokens"), \
             field(l, "padded_tokens"), field(l, "decode_only"), \
             field(l, "num_prefill_chunks"), field(l, "finished"), mark
    }
    if (nsteps > 0) {
      printf "  last common step across hosts: %d | cluster max: %d | spread: %d\n", mn, mx, mx - mn
      print  "  read: the host whose last step is LOWEST stopped logging first — that"
      print  "  worker diverged/halted; compare its composition line with its peers."
    }
  }' "$FLIGHT_RAW"
