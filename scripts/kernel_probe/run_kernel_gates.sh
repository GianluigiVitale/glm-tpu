#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# run_kernel_gates.sh — orchestrate the single-chip §2a/§2b DSA kernel gates.
# Runbook: ~/glm-tpu/docs/11-pod-runbook.md §2a/§2b.
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │ THIS CONSUMES EXCLUSIVE SINGLE-CHIP ACCESS (TPU_VISIBLE_DEVICES=0).        │
# │ GPQA and ANY vLLM/Ray engine on this host MUST be DONE first — the gates   │
# │ run on ONE idle v4 chip and will collide with a live engine otherwise.    │
# │ This script GUARDS against that (libtpu lockfile + engine pgrep) and       │
# │ ABORTS rather than fighting for the chip.                                  │
# └──────────────────────────────────────────────────────────────────────────┘
#
# What it does (cheapest-risk-first, per the adopted audit):
#   0. Ensure ~/tpu-inference is on branch glm-5.2-v4-next (the staging branch;
#      DSA kernels frozen at c1456937 behavior). Refuses to auto-switch a dirty
#      tree.
#   1. Guard exclusive access: no /tmp/libtpu_lockfile, no engine process.
#   2. §2a — probe_2a_indexer_compile.py (Mosaic compile of the (1,H) w-tile).
#            tee -> docs/artifacts/kernelprobe-2a-<date>.log
#   3. ONLY IF §2a prints "GATE 2a: ACCEPT" (exit 0): §2b — probe_2b_parity.py.
#            tee -> docs/artifacts/kernelprobe-2b-<date>.log
#      If §2a REJECTs: stop, point at the documented fallback, exit non-zero.
#
# Usage (owner / main-thread, POST-GPQA, on the pod worker-0):
#   bash ~/glm-tpu/scripts/kernel_probe/run_kernel_gates.sh
#
# The DSA kernels are FROZEN (c1456937); this is TEST-HARNESS orchestration and
# touches no kernel logic.
set -uo pipefail

# --- locations ---------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLM_TPU_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FORK_DIR="${TPU_INFERENCE_DIR:-${HOME}/tpu-inference}"
PY="${GLM_PY:-${HOME}/vllm-env/bin/python}"
ARTIFACTS="${GLM_TPU_ROOT}/docs/artifacts"
DATE="$(date -u +%Y%m%d)"
EXPECTED_BRANCH="glm-5.2-v4-next"
LOG_2A="${ARTIFACTS}/kernelprobe-2a-${DATE}.log"
LOG_2B="${ARTIFACTS}/kernelprobe-2b-${DATE}.log"

mkdir -p "${ARTIFACTS}"

banner() { printf '\n\033[1m%s\033[0m\n' "==== $* ===="; }
die()    { printf '\033[31mABORT: %s\033[0m\n' "$*" >&2; exit 2; }

banner "run_kernel_gates.sh — single-chip §2a/§2b DSA kernel gates"
echo "glm-tpu root : ${GLM_TPU_ROOT}"
echo "fork dir     : ${FORK_DIR}"
echo "python       : ${PY}"
echo "artifacts    : ${ARTIFACTS}"
[ -x "${PY}" ] || die "python interpreter not found/executable: ${PY} (set GLM_PY)"
[ -d "${FORK_DIR}/.git" ] || die "fork repo not found at ${FORK_DIR} (set TPU_INFERENCE_DIR)"

# --- 0. ensure the fork is on glm-5.2-v4-next --------------------------------
banner "step 0 — ensure ${FORK_DIR} is on ${EXPECTED_BRANCH}"
CUR_BRANCH="$(git -C "${FORK_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
if [ "${CUR_BRANCH}" != "${EXPECTED_BRANCH}" ]; then
  echo "fork is on '${CUR_BRANCH}', need '${EXPECTED_BRANCH}'."
  if [ -n "$(git -C "${FORK_DIR}" status --porcelain)" ]; then
    die "fork tree is DIRTY — refusing to auto-switch. Commit/stash in ${FORK_DIR}, then re-run."
  fi
  echo "tree clean — checking out ${EXPECTED_BRANCH}..."
  git -C "${FORK_DIR}" checkout "${EXPECTED_BRANCH}" \
    || die "git checkout ${EXPECTED_BRANCH} failed in ${FORK_DIR}"
fi
echo "fork now on  : $(git -C "${FORK_DIR}" rev-parse --abbrev-ref HEAD) @ $(git -C "${FORK_DIR}" rev-parse --short HEAD)"

# --- 1. guard exclusive single-chip access -----------------------------------
banner "step 1 — guard exclusive chip access (no engine may be running)"
if [ -e /tmp/libtpu_lockfile ]; then
  die "/tmp/libtpu_lockfile present — a TPU process holds the chip. Stop GPQA/the engine first."
fi
# pgrep guard: any live vLLM/Ray/bench engine on this host is a collision.
ENGINE_PAT='run_bench.py|vllm|ray::|raylet|EngineCore|launch_glm_32chip'
HITS="$(pgrep -af "${ENGINE_PAT}" | grep -v -e 'run_kernel_gates' -e "pgrep" || true)"
if [ -n "${HITS}" ]; then
  echo "${HITS}"
  die "an engine/Ray/bench process is running — this gate needs the chip exclusively."
fi
echo "clear — no lockfile, no engine process detected."
export TPU_VISIBLE_DEVICES=0
echo "TPU_VISIBLE_DEVICES=${TPU_VISIBLE_DEVICES} (single chip)"

# --- 2. §2a — indexer-kernel Mosaic compile probe ----------------------------
banner "step 2 — §2a indexer-kernel Mosaic compile probe (interpret=False)"
echo "log -> ${LOG_2A}"
TPU_VISIBLE_DEVICES=0 "${PY}" \
  "${SCRIPT_DIR}/probe_2a_indexer_compile.py" 2>&1 | tee "${LOG_2A}"
RC_2A=${PIPESTATUS[0]}
echo "§2a exit code: ${RC_2A}"

if [ "${RC_2A}" -ne 0 ]; then
  banner "§2a did NOT ACCEPT — stopping before §2b"
  echo "grep verdict:"; grep -E 'GATE 2a:' "${LOG_2A}" || true
  echo
  echo "REJECT path: apply the documented fallback (broadcast-multiply w[0,:,None]*s"
  echo "+ sublane reduction over H), re-run §2a, commit the fallback as the gate-fix."
  echo "(see docs/01-dsa-kernel-design.md §1.3, indexer_kernel.py _indexer_scores_kernel.)"
  exit 1
fi

# --- 3. §2b — real-MXU parity (only after §2a ACCEPT) ------------------------
banner "step 3 — §2b real-MXU parity (indexer + sparse-MLA + pack_new_kv OOB)"
echo "log -> ${LOG_2B}"
TPU_VISIBLE_DEVICES=0 "${PY}" \
  "${SCRIPT_DIR}/probe_2b_parity.py" --tag "${DATE}" 2>&1 | tee "${LOG_2B}"
RC_2B=${PIPESTATUS[0]}
echo "§2b exit code: ${RC_2B}"

banner "kernel gates complete"
echo "§2a: $( [ "${RC_2A}" -eq 0 ] && echo ACCEPT || echo REJECT )   (${LOG_2A})"
echo "§2b: $( [ "${RC_2B}" -eq 0 ] && echo 'PASS — parity green on silicon' || echo FAIL )   (${LOG_2B})"
echo "results file: ${ARTIFACTS}/kernelprobe-2b-${DATE}.results.txt"
if [ "${RC_2B}" -eq 0 ]; then
  echo "-> §2a/§2b GREEN: proceed to the staging switch + engine-level DSA (runbook §2/§3)."
else
  echo "-> §2b FAILED: inspect the verbatim deltas above; do NOT proceed to engine DSA."
fi
exit "${RC_2B}"
