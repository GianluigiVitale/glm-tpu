#!/usr/bin/env bash
# Bounded one-host probe: layer-1 prompt keys of chunk 0 in the legacy prefill
# geometry, compared with the sealed legacy layer-1 prompt cache.
#
# Protections: pod lease; canonically vacant append-only remote prefix across
# live, all-version and soft-deleted surfaces (exact gcloud grammar); run
# directory created atomically after the lease; sanitized Git (no replacement
# refs, no lazy fetch, no user config) authenticating the pin against origin;
# the probe is extracted from the committed blob and verifies its own bytes;
# project modules are imported from a sealed in-memory archive of committed
# blobs under the root-owned immutable interpreter (-I -S -B) with sealed
# JAX/libtpu sites; eight-host zero-work censuses (the proven oracle-wrapper
# census) before and after; generation-bound remote ledger sealed as the
# terminal object and validated as the exact remote set; no local evidence
# mutation after upload. Invoke sanitized:
#   /usr/bin/env -i HOME=/home/gianl PATH=/usr/bin:/bin:/snap/bin LANG=C LC_ALL=C \
#     GLM_GREENFIELD_CHUNK0_GEOMETRY_TAG=<approved tag> \
#     /bin/bash /home/gianl/glm-tpu-topology-rewrite/scripts/greenfield/run_probe_layer1_prompt_chunk0_geometry.sh
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly GREENFIELD_ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly PROBE_REPOSITORY_PATH=scripts/greenfield/probe_layer1_prompt_chunk0_geometry.py
readonly INPUT_DIR=/home/gianl/glm-run/greenfield_layer0_dsa_input_fused_qkv_20260807T202538052784486Z
readonly INPUT_MANIFEST_SHA=574f3553e6106a997e780b6b2a321bce86ad358b19c38989e84e2a4914b73141
readonly CHECKPOINT_ROOT=/home/gianl/gcs-models/models/GLM-5.2-FP8
readonly CHECKPOINT_INDEX_SHA=e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf
readonly WEIGHT_DIGESTS_RELATIVE=docs/artifacts/gate-d-chunk0-probe-weight-digests.json
readonly WEIGHT_DIGESTS_SHA=5a49ab9a8c6a6dc7ee41cf104856e4709c849b0c42cb68d00e52c33241552463
readonly LEGACY_LAYER1_CACHE_DIR=/home/gianl/glm-run/greenfield_legacy_layer1_prompt_index_cache_20260903T000356727206404Z/prompt_index_cache
readonly LEGACY_LAYER1_MANIFEST_SHA=d9058cc6584aca784212706789e72b4981754cb9880e553122b5e854bc3961ac
readonly DB518_RESULT=/home/gianl/glm-run/greenfield_pp16_feature2_layer0_db518_numerical_20260829T115022665987633Z/result.npz
readonly DB518_RESULT_SHA=534bacc54d74992f5a8ab4d422f9fa0947523d59325b4bfa272d4fbeb56262f0
readonly SEALED_PYTHON=/opt/glm-tpu/gate-d-python-3.12.13-021044895e95/bin/python3.12
readonly SEALED_PYTHON_SHA=021044895e95be79dc2f110367607e684119afbc8ce75f6f0eec94844e0acec7
readonly GCLOUD=/snap/bin/gcloud
readonly VACANCY_EXPECTED='ERROR: (gcloud.storage.ls) One or more URLs matched no objects.'

TAG=${GLM_GREENFIELD_CHUNK0_GEOMETRY_TAG:-}
[[ -n $TAG && $TAG =~ ^greenfield_layer1_prompt_chunk0_geometry_[0-9TZ]+$ ]] || {
  echo "GLM_GREENFIELD_CHUNK0_GEOMETRY_TAG must be an approved composed tag" >&2
  exit 2
}
readonly TAG
readonly RUN_DIR=/home/gianl/glm-run/$TAG
readonly REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG

# Sanitized Git: no user/system config, no lazy fetch, no replacement objects,
# no prompts; local operations only.
git_local() {
  /usr/bin/env -i \
    GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 GIT_NO_LAZY_FETCH=1 \
    GIT_NO_REPLACE_OBJECTS=1 GIT_OPTIONAL_LOCKS=0 GIT_PROTOCOL_FROM_USER=0 \
    GIT_SSH_COMMAND=/bin/false GIT_TERMINAL_PROMPT=0 HOME=/nonexistent \
    LANG=C LC_ALL=C PATH=/usr/bin:/bin \
    /usr/bin/git -c core.fsmonitor=false -c core.untrackedCache=false \
      -c core.attributesFile=/dev/null "$@"
}
# Same sanitization, but SSH is allowed for the single read-only origin query.
git_remote_query() {
  /usr/bin/env -i \
    GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 GIT_NO_LAZY_FETCH=1 \
    GIT_NO_REPLACE_OBJECTS=1 GIT_OPTIONAL_LOCKS=0 GIT_TERMINAL_PROMPT=0 \
    HOME=/home/gianl LANG=C LC_ALL=C PATH=/usr/bin:/bin \
    /usr/bin/git -c core.fsmonitor=false "$@"
}

[[ $(git_local -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing probe outside $BRANCH" >&2
  exit 2
}
PIN=$(git_local -C "$WORKTREE" rev-parse HEAD)
[[ $PIN =~ ^[0-9a-f]{40}$ ]] || { echo "invalid pin" >&2; exit 2; }
readonly PIN
[[ -z $(git_local -C "$WORKTREE" status --porcelain=v1 --untracked-files=all) ]] || {
  echo "refusing probe from a dirty canonical worktree" >&2
  exit 2
}
[[ -z $(git_local -C "$WORKTREE" for-each-ref --format='%(refname)' refs/replace) ]] || {
  echo "repository has forbidden replacement refs" >&2
  exit 2
}
[[ $(git_local -C "$WORKTREE" remote get-url origin) == "$GREENFIELD_ORIGIN" ]] || {
  echo "canonical worktree origin drifted" >&2
  exit 2
}
ORIGIN_TIP=$(git_remote_query -C "$WORKTREE" ls-remote --exit-code "$GREENFIELD_ORIGIN" "refs/heads/$BRANCH" | /usr/bin/awk '{print $1}')
readonly ORIGIN_TIP
[[ $ORIGIN_TIP == "$PIN" ]] || {
  echo "canonical HEAD $PIN is not the origin tip of $BRANCH ($ORIGIN_TIP)" >&2
  exit 2
}
[[ -r $INPUT_DIR/manifest.json && -r $CHECKPOINT_ROOT/model.safetensors.index.json &&
   -r $LEGACY_LAYER1_CACHE_DIR/manifest.json && -r $DB518_RESULT ]] || {
  echo "inputs missing" >&2
  exit 2
}
[[ $(/usr/bin/sha256sum "$CHECKPOINT_ROOT/model.safetensors.index.json" | /usr/bin/awk '{print $1}') == "$CHECKPOINT_INDEX_SHA" ]] || {
  echo "checkpoint index digest drifted" >&2
  exit 2
}
[[ $(/usr/bin/sha256sum "$SEALED_PYTHON" | /usr/bin/awk '{print $1}') == "$SEALED_PYTHON_SHA" ]] || {
  echo "sealed interpreter bytes drifted" >&2
  exit 2
}

# Pod lease first; everything below is serialized pod work.
exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
/usr/bin/flock -n 9 || {
  echo "ABORT: another protected pod workflow holds the global lease" >&2
  exit 1
}

# Three-surface remote vacancy: each listing must fail with exactly the
# canonical gcloud response and produce no other output.
vacancy_check() {
  local flags=("$@")
  local output rc
  set +e
  output=$(PYTHONWARNINGS=ignore /usr/bin/timeout --signal=TERM --kill-after=10 90 \
    "$GCLOUD" storage ls "${flags[@]}" "$REMOTE_PREFIX/**" 2>&1)
  rc=$?
  set -e
  [[ $rc -eq 1 && $output == "$VACANCY_EXPECTED" ]] || {
    echo "ABORT: remote prefix not canonically vacant (flags=${flags[*]:-none} rc=$rc): $output" >&2
    exit 1
  }
  printf 'VACANT flags=%s rc=%s\n' "${flags[*]:-none}" "$rc"
}
VACANCY_RECEIPTS=$(vacancy_check; vacancy_check --all-versions; vacancy_check --soft-deleted --exhaustive)
readonly VACANCY_RECEIPTS

# Atomic, symlink-safe run directory creation after the lease.
[[ ! -e $RUN_DIR && ! -L $RUN_DIR ]] || { echo "append-only run path already exists" >&2; exit 2; }
/usr/bin/mkdir "$RUN_DIR"
/usr/bin/mkdir "$RUN_DIR/hlo" "$RUN_DIR/probe" "$RUN_DIR/source"
printf '%s\n' "$VACANCY_RECEIPTS" >"$RUN_DIR/remote_vacancy.txt"
readonly POST_LOG=/home/gianl/glm-run/$TAG.post_upload.log

sealed=0
say() {
  local line="[chunk0-geometry $(/usr/bin/date -u +%H:%M:%S)] $*"
  /usr/bin/printf '%s\n' "$line"
  if [[ $sealed -eq 0 ]]; then
    /usr/bin/printf '%s\n' "$line" >>"$RUN_DIR/orchestrator.log"
  else
    /usr/bin/printf '%s\n' "$line" >>"$POST_LOG"
  fi
}

# Committed bytes only: the probe and the digest record are extracted from the
# pin's blobs (never from the worktree). The probe re-verifies its own bytes
# against the same blob and imports project code from a sealed blob archive.
readonly PROBE_LOCAL=$RUN_DIR/source/probe_layer1_prompt_chunk0_geometry.py
readonly DIGESTS_LOCAL=$RUN_DIR/source/weight_digests.json
git_local -C "$WORKTREE" show "$PIN:$PROBE_REPOSITORY_PATH" >"$PROBE_LOCAL"
git_local -C "$WORKTREE" show "$PIN:$WEIGHT_DIGESTS_RELATIVE" >"$DIGESTS_LOCAL"
[[ $(/usr/bin/sha256sum "$DIGESTS_LOCAL" | /usr/bin/awk '{print $1}') == "$WEIGHT_DIGESTS_SHA" ]] || {
  say "ABORT: committed weight digest record drifted from the pinned digest"
  exit 1
}
PROBE_SHA=$(/usr/bin/sha256sum "$PROBE_LOCAL" | /usr/bin/awk '{print $1}')
readonly PROBE_SHA

has_eight_unique_markers() {
  local file=$1 marker=$2
  [[ $(/usr/bin/awk -v marker="$marker" '$1 == marker {print $2}' "$file" | /usr/bin/wc -l) -eq 8 ]] &&
    [[ $(/usr/bin/awk -v marker="$marker" '$1 == marker {print $2}' "$file" | /usr/bin/sort -u | /usr/bin/wc -l) -eq 8 ]]
}

# The proven eight-host zero-work census of the protected oracle wrapper: the
# carrier process and its ancestors are excluded via a census-carrier marker,
# Ray processes are enumerated by psutil, generic launchers by bracketed
# patterns that cannot match their own command line.
strict_census() {
  local label=$1
  local out="$RUN_DIR/census_${label}.txt"
  local carrier="${TAG}_${label}"
  local ray_enum command
  # shellcheck disable=SC2016
  ray_enum='GLM_CENSUS_CARRIER='"$carrier"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[r]ay start --address|[r]ay start --head|[g]lm_longctx[.]py|[c]ompile_short_decoder[.]py|[p]robe_layer1_prompt_chunk0_geometry[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname)"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" /usr/bin/timeout --signal=TERM --kill-after=10 300 \
    "$GCLOUD" compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
    --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

post_census_done=0
on_exit() {
  local status=$?
  if [[ $post_census_done -eq 0 ]]; then
    if ! strict_census failure_exit; then
      say "CENSUS_UNVERIFIED: failure-exit census is not an authenticated 8/8 zero-work state"
      /usr/bin/touch "$RUN_DIR/CENSUS_UNVERIFIED"
    fi
  fi
  if [[ $status -ne 0 && $sealed -eq 0 ]]; then
    say "FAILED status=$status; partial evidence preserved at $RUN_DIR"
    "$GCLOUD" storage cp --recursive --no-clobber "$RUN_DIR" "$REMOTE_PREFIX/diagnostic/" >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT

say "RUN_DIR=$RUN_DIR PIN=$PIN origin_tip=$ORIGIN_TIP probe_sha=$PROBE_SHA input=$INPUT_MANIFEST_SHA legacy_layer1=$LEGACY_LAYER1_MANIFEST_SHA db518=$DB518_RESULT_SHA checkpoint_index=$CHECKPOINT_INDEX_SHA weight_digests=$WEIGHT_DIGESTS_SHA sealed_python=$SEALED_PYTHON_SHA"
strict_census pre || {
  say "ABORT: pre-run census is not eight-host zero work"
  exit 1
}

say "running the bounded chunk-0 legacy-geometry probe (committed blob, sealed archive imports, sealed interpreter)"
started=$(/usr/bin/date +%s)
set +e
/usr/bin/env -i HOME=/home/gianl PATH=/usr/bin:/bin LANG=C LC_ALL=C PYTHONDONTWRITEBYTECODE=1 \
  JAX_PLATFORMS=tpu \
  TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_PROCESS_BOUNDS=1,1,1 \
  TPU_VISIBLE_DEVICES=0,1,2,3 \
  /usr/bin/timeout --signal=TERM --kill-after=60 3600 \
  "$SEALED_PYTHON" -I -S -B "$PROBE_LOCAL" \
    --code-pin "$PIN" \
    --repository "$WORKTREE" \
    --run-tag "$TAG" \
    --input-dir "$INPUT_DIR" \
    --input-manifest-sha256 "$INPUT_MANIFEST_SHA" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --checkpoint-index-sha256 "$CHECKPOINT_INDEX_SHA" \
    --weight-digests "$DIGESTS_LOCAL" \
    --weight-digests-sha256 "$WEIGHT_DIGESTS_SHA" \
    --legacy-layer1-cache-dir "$LEGACY_LAYER1_CACHE_DIR" \
    --legacy-layer1-manifest-sha256 "$LEGACY_LAYER1_MANIFEST_SHA" \
    --db518-result "$DB518_RESULT" \
    --db518-result-sha256 "$DB518_RESULT_SHA" \
    --output "$RUN_DIR/probe" \
    --hlo-dir "$RUN_DIR/hlo" \
  >"$RUN_DIR/runner.log" 2>&1
probe_status=$?
set -e
elapsed=$(( $(/usr/bin/date +%s) - started ))
say "probe exited status=$probe_status in ${elapsed}s"

strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1
[[ $probe_status -eq 0 ]] || {
  say "probe FAILED (status $probe_status); diagnostics preserved, no SUCCESS"
  exit 1
}
say "sealing local evidence"
sealed=1  # orchestrator.log is frozen from here; later messages go to $POST_LOG
(
  cd "$RUN_DIR"
  /usr/bin/find hlo probe source -type f -print0 | /usr/bin/sort -z | /usr/bin/xargs -0 /usr/bin/sha256sum
  /usr/bin/sha256sum runner.log orchestrator.log census_pre.txt census_post.txt remote_vacancy.txt
) >"$RUN_DIR/evidence.sha256"
/usr/bin/touch "$RUN_DIR/SUCCESS"

say "uploading append-only results and building the generation-bound ledger"
"$GCLOUD" storage cp --recursive --no-clobber "$RUN_DIR" "$REMOTE_PREFIX/" >/dev/null
"$SEALED_PYTHON" -I -S -B - "$RUN_DIR" "$REMOTE_PREFIX" "$GCLOUD" <<'PY' >"$RUN_DIR/remote_objects.json"
import json, subprocess, sys
from pathlib import Path
root, prefix, gcloud = Path(sys.argv[1]), sys.argv[2].rstrip("/"), sys.argv[3]
paths = sorted((p, p.relative_to(root).as_posix()) for p in root.rglob("*") if p.is_file() and p.name != "remote_objects.json")
def local_crc32c(path):
    out = subprocess.run([gcloud, "storage", "hash", "--skip-md5", str(path)], check=True, capture_output=True, text=True, env={"PYTHONWARNINGS": "ignore", "PATH": "/usr/bin:/bin:/snap/bin", "HOME": "/home/gianl"}).stdout
    for line in out.splitlines():
        if line.strip().startswith("crc32c_hash:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit(f"local crc32c unavailable: {path}")
records = []
for path, relative in paths:
    remote = json.loads(subprocess.run([gcloud, "storage", "objects", "describe", f"{prefix}/{relative}", "--format=json"], check=True, capture_output=True, text=True, env={"PYTHONWARNINGS": "ignore", "PATH": "/usr/bin:/bin:/snap/bin", "HOME": "/home/gianl"}).stdout)
    crc32c = remote.get("crc32c_hash") or remote.get("crc32c")
    if int(remote["size"]) != path.stat().st_size or crc32c != local_crc32c(path):
        raise SystemExit(f"remote object verification failed: {relative}")
    records.append({"crc32c": crc32c, "generation": str(remote["generation"]), "path": relative, "size": int(remote["size"])})
print(json.dumps({"objects": records, "prefix": prefix}, indent=2, sort_keys=True))
PY
"$GCLOUD" storage cp --no-clobber "$RUN_DIR/remote_objects.json" "$REMOTE_PREFIX/remote_objects.json" >/dev/null
[[ $(/usr/bin/sha256sum "$RUN_DIR/remote_objects.json" | /usr/bin/awk '{print $1}') == \
   $("$GCLOUD" storage cat "$REMOTE_PREFIX/remote_objects.json" | /usr/bin/sha256sum | /usr/bin/awk '{print $1}') ]] || {
  say "ABORT: terminal ledger checksum mismatch"
  exit 1
}
"$SEALED_PYTHON" -I -S -B - "$RUN_DIR" "$REMOTE_PREFIX" "$GCLOUD" <<'PY'
import subprocess, sys
from pathlib import Path
root, prefix, gcloud = Path(sys.argv[1]), sys.argv[2].rstrip("/") + "/", sys.argv[3]
expected = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
listing = [line.strip() for line in subprocess.run([gcloud, "storage", "ls", f"{prefix}**"], check=True, capture_output=True, text=True, env={"PYTHONWARNINGS": "ignore", "PATH": "/usr/bin:/bin:/snap/bin", "HOME": "/home/gianl"}).stdout.splitlines() if line.strip()]
observed = []
for uri in listing:
    if not uri.startswith(prefix):
        raise SystemExit(f"remote object escaped prefix: {uri}")
    observed.append(uri[len(prefix):])
if len(observed) != len(set(observed)):
    raise SystemExit("remote object listing contains duplicates")
if set(observed) != expected:
    raise SystemExit(f"remote object set drifted: missing={sorted(expected - set(observed))} extra={sorted(set(observed) - expected)}")
print(f"REMOTE_SET_EXACT {len(observed)}")
PY
say "SUCCESS tag=$TAG remote=$REMOTE_PREFIX"
