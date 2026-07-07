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
#      common step, and which host stopped logging earlier than its peers.
#      Round-6 F1 fix (docs/reviews/round6-observability.md): the fetch uses
#      --output-directory (one file per worker — `gcloud --worker=all` on a
#      shared stdout INTERLEAVES the 8 ssh streams and misattributes hosts)
#      AND tags every line with the remote hostname (`FR|<host>|...`), so
#      attribution never depends on marker/payload adjacency.
#      Round-6 F4 caveat, printed in the output header: the recorder logs at
#      DISPATCH — under sync scheduling the last line IS the dying step
#      (lag 0); under ASYNC (the vLLM default) the halted step is the last
#      line MINUS 0 or 1 (at a page boundary that is the difference between
#      the one-page and two-page program). The rule for THIS run is derived
#      from the log's async_scheduling engine arg when present.
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
FLIGHT_DIR="$(mktemp -d)"
trap 'rm -f "$CLEAN" "$FLIGHT_RAW" "$FLIGHT_RAW.err"; rm -rf "$FLIGHT_DIR"' EXIT
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
# Round-6 F4: the recorder hook runs AFTER dispatch, so "last logged step" ==
# last DISPATCHED step. Sync scheduling blocks on the step's own tokens =>
# lag 0; ASYNC (the vLLM default) blocks on step N-1 => the halted step is
# the last line minus 0 OR 1. Recover the mode from the run log's engine args
# (also recorded in recorder_init events since the round-6 recorder fix).
# Printed BEFORE the fetch so --no-fetch triage sees the rule too (§3's
# log-derived last steps need the same reading).
SCHED_MODE="$(grep -oE "'async_scheduling': (True|False)" "$CLEAN" | head -1 | grep -oE "True|False" || true)"
echo "  step-attribution rule (the recorder logs at DISPATCH, not completion):"
echo "    sync  (async_scheduling=False): halted step = last logged step (lag 0)"
echo "    ASYNC (True/unset = the vLLM DEFAULT): halted step = last logged step - 0..1"
echo "          (at a page boundary that is the one-page vs two-page program!)"
case "$SCHED_MODE" in
  False) echo "    this log: async_scheduling=False -> lag 0 (exact attribution)" ;;
  True)  echo "    this log: async_scheduling=True -> subtract 0..1 steps" ;;
  *)     echo "    this log: async_scheduling not found -> assume ASYNC (subtract 0..1)" ;;
esac
if [ "$NO_FETCH" -eq 1 ]; then
  echo "  (fetch skipped: --no-fetch)"
  exit 0
fi
# Round-6 F1: fetch into per-worker files (--output-directory writes
# {WORKER_ID}.log per worker — no shared-stdout interleaving) AND tag every
# remote line with its hostname so the awk below never attributes a line by
# adjacency to a marker. Both defenses verified against the installed SDK
# (threads share stdout without --output-directory).
# shellcheck disable=SC2016  # $(hostname) + $H must expand on the REMOTE host
if ! timeout 180 gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --output-directory "$FLIGHT_DIR" \
    --command='H=$(hostname); { echo "FLIGHT_HOST"; tail -n 100 '"$FLIGHT_GLOB"' 2>/dev/null || echo "NO_FLIGHT_FILES"; } | sed "s/^/FR|$H|/"' \
    > "$FLIGHT_RAW.err" 2>&1; then
  echo "  (fetch FAILED — pod unreachable or gcloud error; rerun with --no-fetch"
  echo "   for the log-only report. stderr tail:)"
  tail -3 "$FLIGHT_RAW.err" 2>/dev/null | sed 's/^/   | /'
  exit 1
fi
cat "$FLIGHT_DIR"/*.log > "$FLIGHT_RAW" 2>/dev/null || true
awk -v halted="$HALTED_IPS" '
  BEGIN { split(halted, hs, " "); for (i in hs) if (hs[i] != "") hset[hs[i]] = 1 }
  {
    # every trusted line is "FR|<hostname>|<payload>" (remote-tagged, F1)
    if (substr($0, 1, 3) != "FR|") next
    rest = substr($0, 4); p = index(rest, "|"); if (p == 0) next
    host = substr(rest, 1, p - 1); payload = substr(rest, p + 1)
    if (!(host in horder)) { horder[host] = ++hn; hosts[hn] = host }
    if (payload == "FLIGHT_HOST") next
    if (payload == "NO_FLIGHT_FILES") { nofiles[host] = 1; next }
    if (payload ~ /^==> .* <==$/) {         # tail multi-file section header
      split(payload, a, " "); fkey[host] = host "|" a[2]; next
    }
    k = (host in fkey) ? fkey[host] : host "|(only-file)"
    ts = -1                                  # F7: never reuse a stale ts
    if (match(payload, /"ts":[0-9.]+/)) ts = substr(payload, RSTART + 5, RLENGTH - 5) + 0
    if (payload ~ /"ev":"step"/) {
      st = -1
      if (match(payload, /"step":[0-9]+/)) st = substr(payload, RSTART + 7, RLENGTH - 7) + 0
      if (st < 0) next
      lastts[k] = ts; laststep[k] = st; lastline[k] = payload; hostof[k] = host
      next
    }
    if (match(payload, /"ev":"[a-z_]+"/)) {  # lifecycle: newest per host|file
      lastev[k] = substr(payload, RSTART + 6, RLENGTH - 7)
      lastevts[k] = ts; hostof[k] = host
      next
    }
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
    for (k in lastev) { h = hostof[k]                 # F7: lifecycle fallback
      if (!(h in bestev) || lastevts[k] > lastevts[bestev[h]]) bestev[h] = k }
    mx = -1; mn = -1; nsteps = 0; mxts = -1
    for (i = 1; i <= hn; i++) { h = hosts[i]
      if (h in best) { s = laststep[best[h]]; nsteps++
        if (s > mx) mx = s
        if (mn < 0 || s < mn) mn = s
        if (lastts[best[h]] > mxts) mxts = lastts[best[h]] } }
    for (i = 1; i <= hn; i++) {
      h = hosts[i]
      if (!(h in best)) {
        if (h in bestev) {                            # F7: crashed in load/warmup?
          k = bestev[h]
          printf "  %-24s no step lines; newest event=%s ts=%.3f (crashed in load/warmup? a\n", \
                 h, lastev[k], lastevts[k]
          print  "                           fresh post-relaunch file has lifecycle events only)"
        } else {
          printf "  %-24s NO flight data%s\n", h,
                 (h in nofiles) ? " (no files — was GLM_FLIGHT_RECORDER=1 baked into the raylet env?)" : ""
        }
        continue
      }
      k = best[h]; l = lastline[k]
      mark = ""
      if (mx >= 0 && laststep[k] < mx) mark = "  <-- stopped " (mx - laststep[k]) " step(s) EARLY (diverged? see caveat below)"
      if (mxts >= 0 && lastts[k] >= 0 && mxts - lastts[k] > 120)   # F2/F3 discrimination
        mark = mark "  <-- recorder quiet " int(mxts - lastts[k]) "s before cluster-max ts:" \
               " recorder-death, not worker-death? (grep worker logs for BLACK BOX DEAD /" \
               " flight recorder DISABLED)"
      printf "  %-24s last step=%-7d ts=%-14.3f reqs=%-3d real=%-5d padded=%-5d decode_only=%d chunks=%d finished=%d%s\n", \
             h, laststep[k], lastts[k], field(l, "num_reqs"), field(l, "real_tokens"), \
             field(l, "padded_tokens"), field(l, "decode_only"), \
             field(l, "num_prefill_chunks"), field(l, "finished"), mark
    }
    if (nsteps > 0) {
      printf "  last common step across hosts: %d | cluster max: %d | spread: %d\n", mn, mx, mx - mn
      print  "  read: apply the step-attribution rule above to the LOWEST last step."
      print  "  caveat (round-6 F4): a mid-collective halt hangs the peers at the SAME"
      print  "  step (sync) or +1 (async) — expected spread 0-1, NOT a growing gap; a"
      print  "  big spread means either a halt that let peers progress, or a host whose"
      print  "  RECORDER died early (check the ts column / the recorder-death mark)."
    }
  }' "$FLIGHT_RAW"
