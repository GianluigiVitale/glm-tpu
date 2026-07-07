#!/usr/bin/env bash
# backup_bundle.sh — VERIFIED plug-and-play state bundle -> GCS (house rule:
# durable backups the moment anything works; this VM is ephemeral).
#
# Bundles the CURRENT working state:
#   * glm-tpu.bundle                      — git bundle of ~/glm-tpu (all
#                                           branches/tags + HEAD; clone-able)
#   * tpu-inference-<branch>.bundle       — git bundle of the fork branch
#                                           (default glm-5.2-v4, full history
#                                           incl. all Stage-2 commits)
#   * results.db                          — bench/results.db snapshot taken
#                                           via the sqlite BACKUP API (safe
#                                           against a concurrently writing
#                                           engine run) + integrity_check
#   * glm-run-logs.tar.gz                 — ~/glm-run/*.log (compressed)
#   * docs.tar.gz                         — glm-tpu docs/ AS ON DISK (captures
#                                           uncommitted doc edits too)
#   * MANIFEST.txt                        — commit hashes, dirty-file honesty
#                                           list, sha256 + byte sizes
#
# Every git bundle is VERIFIED (git bundle verify + an actual test clone whose
# tip hash must match the source) BEFORE upload; the upload is verified by
# listing the destination and comparing every remote byte size to the local
# file. Prints the restore one-liner at the end.
#
# ⛔ COST RULE (CLAUDE.md): uploads go ONLY to gs://driftbench-dsv4-uc
# (US-CENTRAL2, same region as the pod). NEVER gs://driftbench-storage —
# that bucket is EUROPE-WEST4 (dearer storage + cross-region egress).
set -euo pipefail

BUCKET="gs://driftbench-dsv4-uc"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEST="${BUCKET}/backups/glm-tpu/${STAMP}"

GLM_TPU="${GLM_TPU_DIR:-$HOME/glm-tpu}"
FORK="${TPU_INFERENCE_DIR:-$HOME/tpu-inference}"
FORK_BRANCH="${FORK_BRANCH:-glm-5.2-v4}"
LOG_DIR="${GLM_RUN_DIR:-$HOME/glm-run}"
RESULTS_DB="$GLM_TPU/bench/results.db"

# Hard guard: refuse ANY destination outside the us-central2 backup prefix.
case "$DEST" in
    "gs://driftbench-dsv4-uc/backups/glm-tpu/"*) ;;
    *) echo "FATAL: destination '$DEST' is outside" \
            "gs://driftbench-dsv4-uc/backups/glm-tpu/ — refusing." >&2
       exit 1 ;;
esac

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
echo "== backup_bundle $STAMP -> $DEST"

# ---- 1. git bundles (verified plug-and-play) --------------------------------
bundle_and_verify() {  # <repo_dir> <bundle_path> <clone_branch|-> <refs...>
    local repo="$1" out="$2" branch="$3"; shift 3
    git -C "$repo" bundle create "$out" "$@"
    git -C "$repo" bundle verify "$out" >/dev/null
    # a real test clone — 'bundle verify' alone does not prove clone-ability
    local vdir="$WORK/verify-$(basename "$out" .bundle)" want got
    if [ "$branch" = "-" ]; then
        git clone -q --no-checkout "$out" "$vdir"
        want="$(git -C "$repo" rev-parse HEAD)"
    else
        git clone -q --no-checkout -b "$branch" "$out" "$vdir"
        want="$(git -C "$repo" rev-parse "$branch")"
    fi
    got="$(git -C "$vdir" rev-parse HEAD)"
    rm -rf "$vdir"
    if [ "$want" != "$got" ]; then
        echo "FATAL: $out test clone tip $got != source $want" >&2
        exit 1
    fi
    echo "   verified $(basename "$out") (tip $got)"
}

echo "-- git bundle: glm-tpu (HEAD $(git -C "$GLM_TPU" rev-parse --short HEAD))"
bundle_and_verify "$GLM_TPU" "$WORK/glm-tpu.bundle" - --branches --tags HEAD

echo "-- git bundle: tpu-inference $FORK_BRANCH" \
     "($(git -C "$FORK" rev-parse --short "$FORK_BRANCH"))"
bundle_and_verify "$FORK" "$WORK/tpu-inference-${FORK_BRANCH}.bundle" \
    "$FORK_BRANCH" "$FORK_BRANCH"

# ---- 2. results.db snapshot (concurrent-writer safe) -------------------------
echo "-- results.db snapshot (sqlite backup API)"
python3 - "$RESULTS_DB" "$WORK/results.db" <<'PY'
import sqlite3, sys
src = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
dst = sqlite3.connect(sys.argv[2])
src.backup(dst)              # atomic page-level snapshot, writer-safe
ok = dst.execute("PRAGMA integrity_check").fetchone()[0]
n_items = dst.execute("SELECT COUNT(*) FROM items").fetchone()[0]
n_runs = dst.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
dst.close(); src.close()
assert ok == "ok", f"snapshot integrity_check: {ok}"
print(f"   snapshot ok: {n_runs} runs, {n_items} items, integrity_check=ok")
PY

# ---- 3. run logs + docs ------------------------------------------------------
HAVE_LOGS=0
if compgen -G "$LOG_DIR/*.log" >/dev/null; then
    echo "-- logs: $LOG_DIR/*.log"
    (cd "$LOG_DIR" && tar czf "$WORK/glm-run-logs.tar.gz" ./*.log)
    HAVE_LOGS=1
else
    echo "-- logs: none found in $LOG_DIR (skipped)"
fi
echo "-- docs: $GLM_TPU/docs (as on disk, incl. uncommitted edits)"
tar czf "$WORK/docs.tar.gz" -C "$GLM_TPU" docs

# ---- 4. manifest (honesty: what the bundles do NOT capture) ------------------
# The redirect target exists while the block runs, so the hash loop MUST skip
# MANIFEST.txt — hashing the half-written manifest itself would record a
# self-hash that can never verify.
{
    echo "backup_bundle $STAMP  host=$(hostname)  dest=$DEST"
    echo
    echo "glm-tpu HEAD:        $(git -C "$GLM_TPU" rev-parse HEAD)"
    echo "tpu-inference $FORK_BRANCH: $(git -C "$FORK" rev-parse "$FORK_BRANCH")"
    echo
    echo "UNCOMMITTED files (NOT in the bundles — docs/ edits ARE in docs.tar.gz):"
    echo "--- glm-tpu:";        git -C "$GLM_TPU" status --short || true
    echo "--- tpu-inference:";  git -C "$FORK" status --short || true
    echo
    echo "files (sha256  bytes  name; MANIFEST.txt itself excluded):"
    (cd "$WORK" && for f in *; do
         [ "$f" = "MANIFEST.txt" ] && continue
         printf '%s  %10d  %s\n' "$(sha256sum "$f" | cut -d' ' -f1)" \
                "$(stat -c %s "$f")" "$f"
     done)
} > "$WORK/MANIFEST.txt"

# Loud honesty at run time too: committed state is what the bundles carry.
DIRTY="$(git -C "$GLM_TPU" status --porcelain; git -C "$FORK" status --porcelain)"
if [ -n "$DIRTY" ]; then
    echo "!! WARNING: uncommitted/untracked files exist (listed in MANIFEST.txt)."
    echo "!! Git bundles carry COMMITTED state only — commit first if those"
    echo "!! files must survive a VM loss (docs/ edits are in docs.tar.gz)."
fi

# ---- 5. upload + verify ------------------------------------------------------
echo "-- uploading to $DEST/"
gcloud storage cp "$WORK"/* "$DEST/" >/dev/null

echo "-- verifying upload (remote listing + byte-size check)"
LISTING="$(gcloud storage ls -l "$DEST/**")"
echo "$LISTING"
fail=0
for f in "$WORK"/*; do
    base="$(basename "$f")"
    local_size="$(stat -c %s "$f")"
    remote_size="$(echo "$LISTING" | awk -v u="$DEST/$base" '$NF==u {print $1}')"
    if [ "$remote_size" != "$local_size" ]; then
        echo "VERIFY FAIL: $base local=$local_size remote='${remote_size:-MISSING}'" >&2
        fail=1
    fi
done
[ "$fail" -eq 0 ] || exit 1
n_files="$(ls "$WORK" | wc -l)"
total="$(du -sh "$WORK" | cut -f1)"
echo "== VERIFIED: $n_files files, $total total, all byte sizes match at $DEST/"

# ---- 6. restore one-liner ----------------------------------------------------
LOGS_STEP=""
[ "$HAVE_LOGS" -eq 1 ] && \
    LOGS_STEP=" && mkdir -p ~/glm-run && tar xzf glm-run-logs.tar.gz -C ~/glm-run"
cat <<EOF

RESTORE (any machine with gcloud + git):
  gcloud storage cp '$DEST/*' . && git clone glm-tpu.bundle glm-tpu && git clone -b $FORK_BRANCH tpu-inference-${FORK_BRANCH}.bundle tpu-inference && cp results.db glm-tpu/bench/results.db && tar xzf docs.tar.gz -C glm-tpu${LOGS_STEP}
EOF
