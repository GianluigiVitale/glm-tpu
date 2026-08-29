#!/bin/bash
# Bring up the 32-chip Ray cluster on the db-v4-64-od v4-64 pod (8 hosts x 4
# chips) for the GLM-5.2 run, with the required env baked into every raylet.
#
# LOAD-BEARING PATTERN (inherited from the proven DSV4 launcher,
# ~/moe-tpu/scripts/launch_32chip.sh): vLLM's Ray executor only carries VLLM_*
# plus a small fixed allow-list of env vars over to the Ray TPU workers — NOT
# our custom ones. Ray workers DO inherit the raylet's environment, so every
# custom env below MUST be exported before `ray start` on EVERY host (head +
# workers), i.e. baked into the raylet env here. Setting them only on the
# driver silently leaves the workers without them.
#
# Prereqs: fork + harness synced on all 8 workers (scripts/sync_workers.sh);
# venv at ~/vllm-env on all workers; firewall rule allow-ray-pod-internal
# (tcp:1024-65535 on 192.168.0.0/16) so Ray's ports work across the private
# TPU fabric (the fabric only opens TPU ports by default).
#
# ── SHARED-POD COLLISION POLICY (READ BEFORE RUNNING) ────────────────────────
# This pod is SHARED: the ASPt serving stack COHABITS on these same 8 hosts
# (a Gemma FP8 TP=4 vLLM server in docker, a Qwen3-Reranker vLLM proc on w-5,
# Qdrant on w-0 — see the owner's aspt-pod-restart-recovery notes).
# The stop phase below (STOP_CMD) runs AS ROOT on ALL 8 workers and its kill
# patterns are NOT GLM-specific:
#   CAN kill (and WILL, silently):
#     - sudo pkill -9 -f 'VLLM::[E]ngineCore'  -> EVERY vLLM engine on each
#       host, including ASPt's dockerized Gemma + the reranker (vLLM v1 titles
#       every engine proc "VLLM::EngineCore"; container procs are visible in
#       the host PID namespace, run as root, and sudo reaches them).
#     - sudo pkill -9 -x raylet                -> ANY user's raylet.
#     - sudo pkill -9 -f '[R]ayWorkerWrapper'  -> any Ray worker, any user.
#   CANNOT kill: qdrant, gcsfuse mounts, non-vLLM
#     python (parity harnesses, ASPt's non-vLLM JAX procs). Such a survivor
#     may still HOLD the libtpu flock while `sudo rm -f /tmp/libtpu_lockfile`
#     removes the lock inode — the next engine then fails later at device-open
#     (DEADLINE_EXCEEDED-style) instead of the unambiguous "ABORTED: lockfile".
#   All stop-phase output is DISCARDED (>/dev/null ...; true): kills and sudo
#   failures are silent by design — nothing here logs what was killed.
# POLICY — COORDINATE BEFORE RUNNING:
#   * If the ASPt vLLM server is up (check on the workers: `docker ps`,
#     `pgrep -af EngineCore`), do NOT launch until its owner agrees — this
#     script WILL SIGKILL it on every host, with zero output.
#   * The reverse collision is just as real: ASPt's documented "nuclear reset"
#     (`sudo pkill -9 python` on all workers) kills a GLM driver/EngineCore
#     mid-benchmark, leaving a half-recorded run in bench/results.db.
#   * The 32 TPU chips cannot be shared anyway — one serving stack at a time.
#
# Usage:
#   bash ~/glm-tpu/scripts/launch_glm_32chip.sh [--dry-run]
set -u

ZONE=us-central2-b
POD=db-v4-64-od
HEAD_IP=$(hostname -i)  # worker 0 = this host (was hardcoded 192.168.0.8 — the pod was re-created since DSV4 and w-0 is now .21)
RAY=~/vllm-env/bin/ray
RECEIPT_VALIDATOR=$(cd "$(dirname "$0")" && pwd)/validate_ray_network.sh
# The protected parent owns the workload lease; launcher descendants must not
# retain that descriptor if the parent dies while gcloud/ssh is still alive.
exec 9>&-
JOIN_TIMEOUT_SECONDS=${RAY_JOIN_TIMEOUT_SECONDS:-120}
[[ $JOIN_TIMEOUT_SECONDS =~ ^[1-9][0-9]*$ && $JOIN_TIMEOUT_SECONDS -le 300 ]] || {
  echo "RAY_JOIN_TIMEOUT_SECONDS must be an integer in 1..300" >&2
  exit 2
}

# ── Env baked into every raylet (GLM-5.2 Stage-1 set) ────────────────────────
# NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray
#   The vLLM/torchax model path + the Ray multi-host backend (same trio as the
#   DSV4 and GLM-5.1 launchers).
# OMP_NUM_THREADS=1
#   Required: OpenMP-after-fork segfaults in the engine-core subprocess
#   otherwise.
# HF_HUB_DISABLE_XET=1
#   Plain HTTP hub transfers (the xet backend is flaky on the pod).
# TPU_DISABLE_DSA_INDEXER=1
#   GLM-5.x config.json ships index_topk, which makes vLLM try to run the DSA
#   indexer in the forward pass; it is not ported to torchax/TPU and the
#   Pallas DSA kernel has not landed -> disable the indexer forward and fall
#   back to DENSE MLA. Drop this once the Pallas DSA kernel lands.
# DISABLE_WEIGHT_REQUANTIZATION=1
#   Keep the FP8 weights CHECKPOINT-EXACT (block scales as shipped). Do NOT
#   set REQUANTIZE_WEIGHT_DTYPE=bfloat16 here: that is the DSV4 load-time
#   dequant-to-bf16 path and it OOMs at GLM-5.2's 753B scale.
# RUNAI_STREAMER_CONCURRENCY=32 RUNAI_STREAMER_MEMORY_LIMIT=34359738368 (32 GiB)
#   Tune the runai streamer's per-host parallel byte-range streaming for
#   load_format="runai_streamer" reading gs:// DIRECTLY (no gcsfuse mount, no
#   disk cache). Read on every host at weight load -> must be in the raylet env.
# JAX_SHARE_BINARY_BETWEEN_HOSTS (passthrough, default 0 = off, byte-identical)
#   One leader host compiles each executable and broadcasts the binary via the
#   jax coordination store; followers load instead of compiling — kills the
#   per-host compile-stagger launch race. Must be set BEFORE
#   jax.distributed.initialize (the raylet env is inherited by the engine
#   procs -> satisfied). Timeout 120 s fail-fast when enabled.
# GLM_FLIGHT_RECORDER (passthrough, default 0 = off, byte-identical)
#   Per-step flight recorder (the black box): every worker appends one JSON
#   line per serving step to /tmp/glm_flight_<host>_<pid>.jsonl (fork
#   runner/flight_recorder.py). Read by the WORKER processes -> must be in
#   the raylet env. After a crash: bash scripts/triage_crash.sh <run_log>.
#
# Adding future envs: append KEY=VALUE to the single ENVS string below (it is
# used verbatim on the head and on every worker), or pass one-offs without
# editing the script:
#   EXTRA_ENVS="GLM_FOO=1 GLM_BAR=2" bash scripts/launch_glm_32chip.sh
ENVS="export NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray OMP_NUM_THREADS=1 HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 DISABLE_WEIGHT_REQUANTIZATION=1 REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn TPU_MIN_TOKEN_BUCKET=${TPU_MIN_TOKEN_BUCKET:-32} RUNAI_STREAMER_CONCURRENCY=32 RUNAI_STREAMER_MEMORY_LIMIT=34359738368 JAX_SHARE_BINARY_BETWEEN_HOSTS=${JAX_SHARE_BINARY_BETWEEN_HOSTS:-1} JAX_SHARE_BINARY_BETWEEN_HOSTS_TIMEOUT_MS=${JAX_SHARE_BINARY_BETWEEN_HOSTS_TIMEOUT_MS:-120000} GLM_FLIGHT_RECORDER=${GLM_FLIGHT_RECORDER:-0}${EXTRA_ENVS:+ $EXTRA_ENVS}"

usage() {
  cat <<'EOF'
Usage: bash ~/glm-tpu/scripts/launch_glm_32chip.sh [--dry-run]

  --dry-run   print the exact commands (stop / head start / join / status)
              instead of executing them; nothing is run locally or over ssh.

Optional env passthrough:
  JAX_SHARE_BINARY_BETWEEN_HOSTS=1    leader-compile + broadcast executables
  GLM_FLIGHT_RECORDER=1               per-step flight recorder on every worker
                                      (/tmp/glm_flight_*.jsonl; triage_crash.sh)
  EXTRA_ENVS="KEY=VALUE ..."          extra envs baked into every raylet
EOF
}

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

dry() { printf 'DRY-RUN> %s\n' "$*"; }

# Same stop hygiene as the DSV4 launcher: force-stop ray, kill stray raylets /
# engine cores / ray workers. The approved read-only gcsfuse model mirror is
# preserved because protected wrappers require it after Ray starts.
# ⚠ These patterns kill ANY vLLM/Ray proc as root on all 8 hosts — including
# the cohabiting ASPt stack. See the SHARED-POD COLLISION POLICY in the header.
# Round-6 F8 (docs/reviews/round6-observability.md): flight-recorder files
# accumulate across relaunches (per-pid, never pruned; /tmp pressure at crash
# time is what trips the recorder's fail-open) — prune all but the 8 newest
# /tmp/glm_flight_* per host. The just-crashed run's files are always the
# newest, so they survive a relaunch; still fetch (triage_crash.sh) BEFORE
# relaunching per the docs/10 run-book.
STOP_CMD="timeout --signal=TERM --kill-after=5 -- 20 $RAY stop -f >/dev/null 2>&1 || true; sudo pkill -9 -f 'VLLM::[E]ngineCore' >/dev/null 2>&1; sudo pkill -9 -f '[R]ayWorkerWrapper' >/dev/null 2>&1; sudo pkill -9 -x raylet >/dev/null 2>&1; sudo rm -f /tmp/libtpu_lockfile; ls -1t /tmp/glm_flight_* 2>/dev/null | tail -n +9 | xargs -r sudo rm -f >/dev/null 2>&1; true"
FAILURE_STOP_CMD='idx=${HOSTNAME##*-w-}; timeout --signal=TERM --kill-after=5 -- 20 /home/gianl/vllm-env/bin/ray stop -f >/dev/null 2>&1 || true; sudo pkill -TERM -f "[r]ay start --address|[r]ay start --head" >/dev/null 2>&1 || true; sudo pkill -TERM -f "VLLM::[E]ngineCore" >/dev/null 2>&1 || true; sudo pkill -TERM -f "[R]ayWorkerWrapper" >/dev/null 2>&1 || true; sudo pkill -TERM -x raylet >/dev/null 2>&1 || true; sleep 2; sudo pkill -KILL -f "[r]ay start --address|[r]ay start --head" >/dev/null 2>&1 || true; sudo pkill -KILL -f "VLLM::[E]ngineCore" >/dev/null 2>&1 || true; sudo pkill -KILL -f "[R]ayWorkerWrapper" >/dev/null 2>&1 || true; sudo pkill -KILL -x raylet >/dev/null 2>&1 || true; sudo rm -f /tmp/libtpu_lockfile; echo "LAUNCH_FAILURE_CLEAN_OK $idx"'
runtime_owned=0

cleanup_failed_launch() {
  local status=$?
  local cleanup_output=
  trap - EXIT
  if [[ $status -ne 0 && $runtime_owned -eq 1 ]]; then
    if ! cleanup_output=$(bash "$RECEIPT_VALIDATOR" bounded 120 \
      gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
      --command="$FAILURE_STOP_CMD" 2>&1); then
      printf '%s\n' "$cleanup_output" >&2
      echo "ABORT: bounded launcher failure cleanup failed" >&2
      return "$status"
    fi
    printf '%s\n' "$cleanup_output" >&2
    printf '%s\n' "$cleanup_output" | \
      bash "$RECEIPT_VALIDATOR" receipts LAUNCH_FAILURE_CLEAN_OK 8 \
        0,1,2,3,4,5,6,7 || {
      echo "ABORT: launcher failure cleanup receipts are incomplete" >&2
      return "$status"
    }
  fi
  return "$status"
}
trap cleanup_failed_launch EXIT

# Worker join: bake ENVS into the raylet, then join the head. Escaped $(...)
# and $? run on the REMOTE host, not here. Preserve the real exit status: a
# failed worker must fail the launcher instead of being hidden by `true`.
JOIN_CMD="$ENVS; $RAY start --address=$HEAD_IP:6379 --node-ip-address=\$(hostname -i) >/tmp/rayjoin.log 2>&1; rc=\$?; echo \"RAY_JOIN_RC_\$rc \${HOSTNAME##*-w-}\"; exit \$rc"

# Soft sanity check: this script must run on worker 0 (the Ray head).
if ! hostname -i 2>/dev/null | grep -qw -- "$HEAD_IP"; then
  echo "WARNING: $HEAD_IP is not among this host's IPs — run this on worker 0 (the head)." >&2
fi

echo "[1/3] stop any existing ray + stray procs on all workers"
if (( DRY_RUN )); then
  dry "gcloud compute tpus tpu-vm ssh $POD --zone $ZONE --worker=all --command=\"$STOP_CMD\""
else
  bash "$RECEIPT_VALIDATOR" bounded 120 \
    gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$STOP_CMD" >/dev/null 2>&1 || {
    echo "ABORT: bounded fleet stop failed" >&2
    exit 1
  }
  runtime_owned=1
fi

# NOTE: NO gcsfuse mount step — the runai streamer reads gs:// directly.

echo "[2/3] start ray head on w-0 (with env)"
if (( DRY_RUN )); then
  dry "$ENVS"
  dry "$RAY stop -f; sleep 2"
  dry "$RAY start --head --port=6379 --node-ip-address=$HEAD_IP --disable-usage-stats"
else
  eval "$ENVS"
  bash "$RECEIPT_VALIDATOR" bounded 30 "$RAY" stop -f >/dev/null 2>&1 || {
    echo "ABORT: bounded local Ray stop failed" >&2
    exit 1
  }
  sleep 2
  if ! HEAD_OUTPUT=$(bash "$RECEIPT_VALIDATOR" bounded 60 \
    "$RAY" start --head --port=6379 \
    --node-ip-address="$HEAD_IP" --disable-usage-stats 2>&1); then
    printf '%s\n' "$HEAD_OUTPUT"
    echo "ABORT: Ray head failed to start" >&2
    exit 1
  fi
  printf '%s\n' "$HEAD_OUTPUT"
  grep -qi "runtime started" <<<"$HEAD_OUTPUT" || {
    echo "ABORT: Ray head success receipt is absent" >&2
    exit 1
  }
fi

echo "[3/3] join workers 1-7 (with env)"
if (( DRY_RUN )); then
  dry "probe workers 1-7 -> $HEAD_IP:6379; require 7 unique TCP6379_OK receipts"
  dry "timeout $JOIN_TIMEOUT_SECONDS gcloud compute tpus tpu-vm ssh $POD --zone $ZONE --worker=1,2,3,4,5,6,7 --command=\"$JOIN_CMD\""
else
  PROBE_CMD='idx=${HOSTNAME##*-w-}; if timeout 3 bash -c "</dev/tcp/'"$HEAD_IP"'/6379" >/dev/null 2>&1; then echo "TCP6379_OK $idx"; else echo "TCP6379_BAD $idx"; exit 1; fi'
  if ! PROBE_OUTPUT=$(bash "$RECEIPT_VALIDATOR" bounded 60 \
    gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" \
    --worker=1,2,3,4,5,6,7 --command="$PROBE_CMD" 2>&1); then
    printf '%s\n' "$PROBE_OUTPUT"
    echo "ABORT: worker-to-head TCP/6379 connectivity probe failed" >&2
    exit 1
  fi
  printf '%s\n' "$PROBE_OUTPUT"
  if ! printf '%s\n' "$PROBE_OUTPUT" | \
    bash "$RECEIPT_VALIDATOR" receipts TCP6379_OK 7 1,2,3,4,5,6,7; then
    echo "ABORT: worker-to-head TCP/6379 receipts are incomplete" >&2
    exit 1
  fi

  if ! JOIN_OUTPUT=$(bash "$RECEIPT_VALIDATOR" bounded "$JOIN_TIMEOUT_SECONDS" \
    gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" \
    --worker=1,2,3,4,5,6,7 --command="$JOIN_CMD" 2>&1); then
    printf '%s\n' "$JOIN_OUTPUT"
    bash "$RECEIPT_VALIDATOR" bounded 30 \
      gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" \
      --worker=1,2,3,4,5,6,7 \
      --command='echo "RAY_JOIN_DIAGNOSTIC $(hostname)"; tail -80 /tmp/rayjoin.log 2>/dev/null || true' \
      2>&1 || true
    echo "ABORT: bounded Ray worker join failed or timed out" >&2
    exit 1
  fi
  printf '%s\n' "$JOIN_OUTPUT"
  if ! printf '%s\n' "$JOIN_OUTPUT" | \
    bash "$RECEIPT_VALIDATOR" receipts RAY_JOIN_RC_0 7 1,2,3,4,5,6,7; then
    echo "ABORT: successful Ray worker-join receipts are incomplete" >&2
    exit 1
  fi
fi

if (( DRY_RUN )); then
  dry "$RAY status   # expect 8 nodes / 32 TPU"
else
  sleep 3
  echo "=== ray status ==="
  if ! STATUS_OUTPUT=$(bash "$RECEIPT_VALIDATOR" bounded 30 \
    "$RAY" status 2>&1); then
    printf '%s\n' "$STATUS_OUTPUT"
    echo "ABORT: bounded Ray status failed" >&2
    exit 1
  fi
  printf '%s\n' "$STATUS_OUTPUT" | grep -E "^ 1 node_|/.*TPU|Total Usage" | head
  NODE_COUNT=$(grep -cE '^ 1 node_' <<<"$STATUS_OUTPUT")
  echo "nodes: $NODE_COUNT (expect 8 nodes / 32 TPU)"
  if [[ $NODE_COUNT -ne 8 ]] || ! grep -qE '^ 0[.]0/32[.]0 TPU$' <<<"$STATUS_OUTPUT"; then
    echo "ABORT: Ray resource contract is not exactly 8 nodes / 32 TPU" >&2
    exit 1
  fi
fi
runtime_owned=0
trap - EXIT
