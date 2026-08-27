#!/usr/bin/env bash
# Protected, diagnostic-only PP16 complete-decoder compile/HLO/HBM acquisition.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly BRANCH=rewrite/topology-first-decode
readonly ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly APPROVED_LOCATION=US-CENTRAL2
readonly SOURCE_TAG=greenfield_full_pack_pp16_20260827T032310295108546Z
readonly SOURCE_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/packed/PP16_LP2/$SOURCE_TAG
readonly SOURCE_MANIFEST_SHA=13ad2e926b44bee86e620d877d4c266dbacc4f9a59b352fc4f2c8ef534efedb5
readonly SOURCE_RUNTIME_TAG=greenfield_runtime_pack_pp16_20260827T091323450875229Z
readonly SOURCE_RUNTIME_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime/PP16_LP2/$SOURCE_RUNTIME_TAG
readonly SOURCE_RUNTIME_MANIFEST_SHA=b0f624666e921a93e92c625e0e1248aaa9162c45a9be974f4491755c8604d2e5
readonly SOURCE_FEATURE_TAG=greenfield_runtime_feature_pack_pp16_20260827T095428043535926Z
readonly SOURCE_FEATURE_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/PP16_LP2/$SOURCE_FEATURE_TAG
readonly SOURCE_FEATURE_MANIFEST_SHA=0f1bb2718a700fb2eee23dc9f172cd9e5cbd1639396d8c8fa8421e1e3b52b6f1
readonly RUNTIME_TAG=greenfield_runtime_feature_qkv_direct_pp16_20260827T164842844148623Z
readonly RUNTIME_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/PP16_LP2/$RUNTIME_TAG
readonly RUNTIME_MANIFEST_SHA=b385458f233f21342855ac4c3373429c034a9e40bd85d638b16466199ff66bab
RUNTIME_KIND=${GLM_GREENFIELD_PP16_RUNTIME_KIND:-pallas_feature}
[[ $RUNTIME_KIND == pallas_feature || $RUNTIME_KIND == pallas_feature_linear ]]
readonly RUNTIME_KIND

cd "$WORKTREE"
[[ ${GLM_GREENFIELD_PP16_COMPILE_ACQUISITION:-0} == 1 ]] || {
  echo "PP16 complete-decoder compile acquisition is default-off" >&2
  exit 2
}
PIN=$(git rev-parse HEAD)
TAG=${GLM_GREENFIELD_PP16_COMPILE_TAG:-greenfield_short_decoder_compile_pp16_acquisition_$(date -u +%Y%m%dT%H%M%S%NZ)}
[[ $TAG =~ ^greenfield_short_decoder_compile_pp16_acquisition_[0-9]{8}T[0-9]{15}Z$ ]]
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG
readonly PIN TAG RUN_DIR REMOTE_PREFIX

[[ $(git branch --show-current) == "$BRANCH" ]]
[[ -z $(git status --porcelain) ]]
[[ ! -e $RUN_DIR ]]
for prerequisite in \
  "$SOURCE_ROOT/SUCCESS" "$SOURCE_ROOT/packed_manifest.json" \
  "$SOURCE_RUNTIME_ROOT/SUCCESS" "$SOURCE_RUNTIME_ROOT/runtime_manifest.json" \
  "$SOURCE_FEATURE_ROOT/SUCCESS" "$SOURCE_FEATURE_ROOT/runtime_manifest.json" \
  "$RUNTIME_ROOT/SUCCESS" "$RUNTIME_ROOT/runtime_manifest.json"; do
  [[ -r $prerequisite ]] || {
    echo "missing PP16 compile prerequisite: $prerequisite" >&2
    exit 2
  }
done
mkdir -p "$RUN_DIR/host_records" "$RUN_DIR/host_logs" "$RUN_DIR/hlo"

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  echo "another protected TPU workflow holds the global lease" >&2
  exit 1
}
exec 8>/home/gianl/.glm-tpu-rsync.lock
flock 8

say() {
  echo "[compile-pp16 $(date -u +%H:%M:%S)] $*" |
    tee -a "$RUN_DIR/orchestrator.log"
}

has_eight_unique_markers() {
  local file=$1 marker=$2
  [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | wc -l) -eq 8 ]] &&
    [[ $(awk -v marker="$marker" '$1 == marker {print $2}' "$file" | sort -u | wc -l) -eq 8 ]]
}

strict_census() {
  local label=$1
  local out="$RUN_DIR/census_${label}.txt"
  local carrier="${TAG}_${label}"
  local ray_enum command
  # shellcheck disable=SC2016
  ray_enum='GLM_CENSUS_CARRIER='"$carrier"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[c]ompile_short_decoder[.]py|[r]un_short_decoder_ws32[.]py|[l]oad_full_checkpoint_stage[.]py" 2>/dev/null || true); holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray_stop_pids: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" gcloud compute tpus tpu-vm ssh "$POD" \
    --zone "$ZONE" --worker=all --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

post_census_done=0
terminal_success_done=0
on_exit() {
  local status=$?
  if [[ $post_census_done -eq 0 ]]; then
    strict_census failure_exit || true
  fi
  if [[ $status -ne 0 && $terminal_success_done -eq 0 ]]; then
    say "FAILED status=$status; preserving nonterminal diagnostics"
    gcloud storage cp --recursive --no-clobber "$RUN_DIR" \
      "$REMOTE_PREFIX/diagnostic_local/" >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT

say "PIN=$PIN TAG=$TAG profile=2K runtime=$RUNTIME_KIND warmup=1 iterations=1 trace=0"
[[ $(git ls-remote "$ORIGIN" "refs/heads/$BRANCH" | awk '{print $1}') == "$PIN" ]]
[[ $(gcloud storage buckets describe "$APPROVED_BUCKET" --format='value(location)') == "$APPROVED_LOCATION" ]]
[[ $(gcloud compute tpus tpu-vm describe "$POD" --zone "$ZONE" --format='value(state,health)') == $'READY\tHEALTHY' ]]
gcloud storage objects list "$REMOTE_PREFIX/**" --format='value(name)' \
  >"$RUN_DIR/remote_vacancy.txt"
[[ ! -s $RUN_DIR/remote_vacancy.txt ]]

/home/gianl/vllm-env/bin/python - \
  "$SOURCE_ROOT/packed_manifest.json" "$SOURCE_MANIFEST_SHA" \
  "$SOURCE_RUNTIME_ROOT/runtime_manifest.json" "$SOURCE_RUNTIME_MANIFEST_SHA" \
  "$SOURCE_FEATURE_ROOT/runtime_manifest.json" "$SOURCE_FEATURE_MANIFEST_SHA" \
  "$RUNTIME_ROOT/runtime_manifest.json" "$RUNTIME_MANIFEST_SHA" \
  "$RUN_DIR/preflight.json" "$PIN" "$TAG" "$REMOTE_PREFIX" \
  "$RUNTIME_KIND" <<'PY'
import json
from pathlib import Path
import sys

source_path, source_sha, base_path, base_sha, source_feature_path, source_feature_sha, runtime_path, runtime_sha, output, pin, tag, remote, runtime_kind = sys.argv[1:]
for path, expected in ((source_path, source_sha), (base_path, base_sha), (source_feature_path, source_feature_sha), (runtime_path, runtime_sha)):
    value = json.loads(Path(path).read_text())
    if value.get("manifest_sha256") != expected:
        raise SystemExit(f"manifest identity drifted: {path}")
record = {
    "approved_bucket": "gs://driftbench-dsv4-uc",
    "bucket_location": "US-CENTRAL2",
    "code_hash": pin,
    "complete_token_path": True,
    "context_capacity": 2048,
    "feature_fuse_route_weighting": False,
    "feature_output_tile": 128,
    "feature_source_verification_mode": "feature_runtime_with_metadata_parent",
    "feature_reconstruct_down_fp32": False,
    "gate_d_claim": False,
    "iterations": 1,
    "numerical_claim": False,
    "performance_claim": False,
    "plan_id": "PP16_LP2",
    "remote_prefix": remote,
    "run_tag": tag,
    "runtime_manifest_sha256": runtime_sha,
    "runtime_kind": runtime_kind,
    "source_packed_manifest_sha256": source_sha,
    "source_feature_runtime_manifest_sha256": source_feature_sha,
    "source_runtime_manifest_sha256": base_sha,
    "split_residual_state": True,
    "trace_steps": 0,
    "warmup": 1,
}
Path(output).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY

strict_census pre || {
  say "ABORT: pre-run census is not authenticated 8/8 zero work"
  exit 1
}

say "syncing exact pushed pin and same-region PP16 artifacts on all hosts"
# shellcheck disable=SC2016
sync_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; wt='"$WORKTREE"'; pin='"$PIN"'; branch='"$BRANCH"'; origin='"$ORIGIN"'; source='"$SOURCE_ROOT"'; source_runtime='"$SOURCE_RUNTIME_ROOT"'; source_feature='"$SOURCE_FEATURE_ROOT"'; runtime='"$RUNTIME_ROOT"'; if [[ "$idx" == 0 ]]; then [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]]; else [[ -e "$wt/.git" ]] && [[ -z $(git -C "$wt" status --porcelain) ]]; git -C "$wt" fetch -q "$origin" "$branch"; git -C "$wt" checkout -q --detach "$pin"; fi; [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]] && [[ -r "$source/SUCCESS" && -r "$source/packed_manifest.json" ]] && [[ -r "$source_runtime/SUCCESS" && -r "$source_runtime/runtime_manifest.json" ]] && [[ -r "$source_feature/SUCCESS" && -r "$source_feature/runtime_manifest.json" ]] && [[ -r "$runtime/SUCCESS" && -r "$runtime/runtime_manifest.json" ]] && findmnt -T "$source" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse" && findmnt -T "$source_runtime" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse" && findmnt -T "$source_feature" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse" && findmnt -T "$runtime" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse" && echo "SYNC_OK $(hostname) $pin"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$sync_command" >"$RUN_DIR/sync.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/sync.txt" SYNC_OK || {
  say "ABORT: exact eight-host sync/artifact check failed"
  exit 1
}

coordinator=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=0 \
  --command="hostname -I" 2>/dev/null | grep -Eo '192\.168\.[0-9]+\.[0-9]+' | head -1)
[[ -n $coordinator ]] || {
  say "ABORT: worker-0 coordinator address is unavailable"
  exit 1
}
coordinator="$coordinator:8476"
say "launching one complete PP16 2K compile and diagnostic execution coordinator=$coordinator"
# shellcheck disable=SC2016
execute_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; tag='"$TAG"'; wt='"$WORKTREE"'; remote='"$REMOTE_PREFIX"'; run=/home/gianl/glm-run/$tag; mkdir -p "$run/hlo"; output="$run/decoder.rank${idx}.json"; log="$run/decoder.rank${idx}.log"; upload() { [[ -f "$log" ]] && gcloud storage cp --no-clobber "$log" "$remote/host_logs/" >/dev/null 2>&1 || true; [[ -f "$output" ]] && gcloud storage cp --no-clobber "$output" "$remote/host_records/" >/dev/null 2>&1 || true; if compgen -G "$run/hlo/*" >/dev/null; then gcloud storage cp --no-clobber "$run"/hlo/* "$remote/hlo/" >/dev/null 2>&1 || true; fi; }; trap upload EXIT; cd "$wt"; env JAX_PLATFORMS=tpu XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH="$wt" GLM_GREENFIELD_RUN_TAG="$tag" timeout --signal=TERM --kill-after=60 10800 /home/gianl/vllm-env/bin/python -u scripts/greenfield/compile_short_decoder.py --coordinator-address '"$coordinator"' --num-processes 8 --process-id "$idx" --expected-code-hash '"$PIN"' --runtime-kind '"$RUNTIME_KIND"' --verify-device-roundtrip 0 --feature-source-metadata-only 1 --feature-output-tile 128 --feature-fuse-route-weighting 0 --feature-reconstruct-down-fp32 0 --complete-token-path 1 --split-residual-state 1 --prefill-index-repair 0 --dsa-query-exact-association 1 --dsa-head-key-exact-association 0 --dsa-score-default-precision 0 --main-rope-table 0 --pregathered-b512-attention 0 --strategy-nd-attention-projection 0 --dense-final-layout-convolution 0 --runtime-root '"$RUNTIME_ROOT"' --runtime-manifest-sha256 '"$RUNTIME_MANIFEST_SHA"' --source-runtime-root '"$SOURCE_RUNTIME_ROOT"' --source-runtime-manifest-sha256 '"$SOURCE_RUNTIME_MANIFEST_SHA"' --source-feature-runtime-root '"$SOURCE_FEATURE_ROOT"' --source-feature-runtime-manifest-sha256 '"$SOURCE_FEATURE_MANIFEST_SHA"' --source-checkpoint-root '"$SOURCE_ROOT"' --source-packed-manifest-sha256 '"$SOURCE_MANIFEST_SHA"' --context-capacity 2048 --warmup 1 --iterations 1 --trace-steps 0 --output "$output" >"$log" 2>&1; trap - EXIT; upload; echo "DECODER_HOST_OK $(hostname) rank=$idx"'
execute_status=0
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$execute_command" >"$RUN_DIR/execute.txt" 2>&1 || execute_status=$?
if [[ $execute_status -ne 0 ]] || \
  ! has_eight_unique_markers "$RUN_DIR/execute.txt" DECODER_HOST_OK; then
  say "compile failed; retrieving narrow host/HLO diagnostics"
  gcloud storage cp "$REMOTE_PREFIX/host_logs/*" "$RUN_DIR/host_logs/" \
    >/dev/null 2>&1 || true
  gcloud storage cp "$REMOTE_PREFIX/host_records/*" "$RUN_DIR/host_records/" \
    >/dev/null 2>&1 || true
  gcloud storage cp "$REMOTE_PREFIX/hlo/*" "$RUN_DIR/hlo/" \
    >/dev/null 2>&1 || true
  exit 1
fi

gcloud storage cp "$REMOTE_PREFIX/host_logs/*" "$RUN_DIR/host_logs/" >/dev/null
gcloud storage cp "$REMOTE_PREFIX/host_records/*" "$RUN_DIR/host_records/" >/dev/null
gcloud storage cp "$REMOTE_PREFIX/hlo/*" "$RUN_DIR/hlo/" >/dev/null
strict_census post || {
  say "ABORT: post-run census is not authenticated 8/8 zero work"
  exit 1
}
post_census_done=1

say "validating full-graph PP16 locality, one-row shapes, load identity and 32-chip HBM"
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
  scripts/greenfield/validate_short_decoder_compile_pp16.py \
  --run-dir "$RUN_DIR" --code-hash "$PIN" --runtime-kind "$RUNTIME_KIND"

cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
(
  cd "$RUN_DIR"
  find host_records host_logs hlo -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum summary.json preflight.json remote_vacancy.txt sync.txt execute.txt \
    census_pre.txt census_post.txt orchestrator.sealed.log
) >"$RUN_DIR/evidence.sha256"
(cd "$RUN_DIR" && sha256sum -c evidence.sha256 >/dev/null)

say "publishing nonterminal diagnostic evidence"
for folder in host_records host_logs hlo; do
  gcloud storage cp --recursive --no-clobber "$RUN_DIR/$folder" \
    "$REMOTE_PREFIX/" >/dev/null
done
for name in summary.json preflight.json remote_vacancy.txt sync.txt execute.txt \
  census_pre.txt census_post.txt orchestrator.sealed.log evidence.sha256; do
  gcloud storage cp --no-clobber "$RUN_DIR/$name" "$REMOTE_PREFIX/$name" \
    >/dev/null
done

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
  scripts/greenfield/seal_short_decoder_compile_pp16.py \
  --run-dir "$RUN_DIR" --remote-prefix "$REMOTE_PREFIX" \
  --code-hash "$PIN" --run-tag "$TAG"
terminal_success_done=1
trap - EXIT
echo "[compile-pp16 $(date -u +%H:%M:%S)] COMPLETE $TAG"
