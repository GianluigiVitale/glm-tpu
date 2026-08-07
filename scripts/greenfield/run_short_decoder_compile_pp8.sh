#!/usr/bin/env bash
# Protected real 78-layer / 2K PP8 body or complete-token execution.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly GREENFIELD_ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly ORACLE_REPO=/home/gianl/tpu-inference
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly WARMUP=${GLM_GREENFIELD_SHORT_DECODER_WARMUP:-2}
readonly ITERATIONS=${GLM_GREENFIELD_SHORT_DECODER_ITERATIONS:-10}
readonly TRACE_STEPS=${GLM_GREENFIELD_SHORT_DECODER_TRACE_STEPS:-0}
readonly RUNTIME_KIND=${GLM_GREENFIELD_DECODER_RUNTIME_KIND:-pallas_feature_linear}
readonly FEATURE_FUSE_ROUTE_WEIGHTING=${GLM_GREENFIELD_FEATURE_FUSE_ROUTE_WEIGHTING:-0}
readonly FEATURE_RECONSTRUCT_DOWN_FP32=${GLM_GREENFIELD_FEATURE_RECONSTRUCT_DOWN_FP32:-0}
readonly COMPLETE_TOKEN_PATH=${GLM_GREENFIELD_COMPLETE_TOKEN_PATH:-0}
readonly SHORT_CONTEXT_ORACLE=${GLM_GREENFIELD_SHORT_CONTEXT_ORACLE:-0}
readonly SHORT_CONTEXT_ORACLE_TAG=greenfield_short_context_oracle_20260806T202544155912103Z
readonly SHORT_CONTEXT_ORACLE_ROOT=/home/gianl/gcs-models/oracles/greenfield/glm52/short_context/2k/$SHORT_CONTEXT_ORACLE_TAG
readonly SHORT_CONTEXT_ORACLE_DIR=$SHORT_CONTEXT_ORACLE_ROOT/oracle
readonly SHORT_CONTEXT_ORACLE_MANIFEST_SHA=f580c14954bcbd0d973b6fe8158520992a18a1375ed88cff9cceb8e01c7efe19
readonly SHORT_CONTEXT_DSA_ORACLE=${GLM_GREENFIELD_SHORT_CONTEXT_DSA_ORACLE:-0}
readonly SHORT_CONTEXT_DSA_ORACLE_TAG=greenfield_short_context_dsa_oracle_recovery_20260806T231905802593249Z
readonly SHORT_CONTEXT_DSA_ORACLE_ROOT=/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/2k/$SHORT_CONTEXT_DSA_ORACLE_TAG
readonly SHORT_CONTEXT_DSA_ORACLE_DIR=$SHORT_CONTEXT_DSA_ORACLE_ROOT/oracle
readonly SHORT_CONTEXT_DSA_ORACLE_MANIFEST_SHA=71224832652ce61024786d39d43dcbfdc6cde76bf2eff0350531b272f38f4f57
if [[ -n ${GLM_GREENFIELD_FEATURE_OUTPUT_TILE+x} ]]; then
  FEATURE_OUTPUT_TILE=$GLM_GREENFIELD_FEATURE_OUTPUT_TILE
elif [[ $RUNTIME_KIND == reference ]]; then
  FEATURE_OUTPUT_TILE=128
else
  FEATURE_OUTPUT_TILE=256
fi
readonly FEATURE_OUTPUT_TILE
readonly SOURCE_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/packed/PP8_LP4/greenfield_full_pack_pp8_20260805T182222755355852Z
readonly SOURCE_MANIFEST_SHA=0869493164a3a63797ea61d88c575f35bea8aa50790c46aa21ce6f0f7c4c78f1
readonly SOURCE_RUNTIME_TAG=greenfield_runtime_pack_pp8_20260806T002756318310857Z
readonly SOURCE_RUNTIME_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime/PP8_LP4/$SOURCE_RUNTIME_TAG
readonly SOURCE_RUNTIME_MANIFEST_SHA=fdedaae31fb3c094266272ed48dfe62bb098257a78272b93c14eafbd57e31dec
[[ $FEATURE_OUTPUT_TILE == 128 || $FEATURE_OUTPUT_TILE == 256 ]] || {
  echo "feature output tile must be 128 or 256" >&2
  exit 2
}
[[ $FEATURE_FUSE_ROUTE_WEIGHTING == 0 || $FEATURE_FUSE_ROUTE_WEIGHTING == 1 ]] || {
  echo "feature route-weight fusion must be 0 or 1" >&2
  exit 2
}
[[ $FEATURE_RECONSTRUCT_DOWN_FP32 == 0 || $FEATURE_RECONSTRUCT_DOWN_FP32 == 1 ]] || {
  echo "feature FP32 reconstruction must be 0 or 1" >&2
  exit 2
}
[[ $COMPLETE_TOKEN_PATH == 0 || $COMPLETE_TOKEN_PATH == 1 ]] || {
  echo "complete token path must be 0 or 1" >&2
  exit 2
}
[[ $SHORT_CONTEXT_ORACLE == 0 || $SHORT_CONTEXT_ORACLE == 1 ]] || {
  echo "short-context oracle flag must be 0 or 1" >&2
  exit 2
}
[[ $SHORT_CONTEXT_DSA_ORACLE == 0 || $SHORT_CONTEXT_DSA_ORACLE == 1 ]] || {
  echo "short-context DSA oracle flag must be 0 or 1" >&2
  exit 2
}
if [[ $SHORT_CONTEXT_ORACLE == 1 ]]; then
  [[ $COMPLETE_TOKEN_PATH == 1 ]] || {
    echo "short-context oracle requires complete token path" >&2
    exit 2
  }
  ((1 + WARMUP + ITERATIONS <= 20)) || {
    echo "correctness window exceeds the 20-token oracle" >&2
    exit 2
  }
  ((2034 + WARMUP + ITERATIONS + TRACE_STEPS <= 2048)) || {
    echo "oracle prompt plus recurrent/trace steps exceeds 2K capacity" >&2
    exit 2
  }
fi
if [[ $SHORT_CONTEXT_DSA_ORACLE == 1 ]]; then
  [[ $SHORT_CONTEXT_ORACLE == 1 && $COMPLETE_TOKEN_PATH == 1 ]] || {
    echo "short-context DSA oracle requires token oracle and complete token path" >&2
    exit 2
  }
  [[ $WARMUP == 2 && $ITERATIONS == 10 && $TRACE_STEPS == 2 ]] || {
    echo "2K Gate D requires warmup=2 iterations=10 trace_steps=2" >&2
    exit 2
  }
fi
case "$RUNTIME_KIND" in
  reference)
    readonly RUNTIME_TAG=$SOURCE_RUNTIME_TAG
    readonly RUNTIME_ROOT=$SOURCE_RUNTIME_ROOT
    readonly RUNTIME_MANIFEST_SHA=$SOURCE_RUNTIME_MANIFEST_SHA
    readonly RUNTIME_LAYOUT_HASH=841a18f6dbbf329243482f97f01d28ca22211d63d323fca859477349cc0abcac
    readonly SPARSE_MOE_BACKEND=reference
    readonly HLO_BACKEND_CONTRACT=tpu_v4_pp8_reference
    ;;
  pallas_feature)
    readonly RUNTIME_TAG=greenfield_runtime_feature_pack_pp8_20260806T064010287072141Z
    readonly RUNTIME_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/PP8_LP4/$RUNTIME_TAG
    readonly RUNTIME_MANIFEST_SHA=54e2f89b1832b994acbf9ef36f5f6ce68c942d9146efc4d7c15360d68b6d9917
    readonly RUNTIME_LAYOUT_HASH=ba21c4ec1500837f17a53047da98a7ed7c3a06782ddffe49d8bd796d4d0d1c9e
    readonly SPARSE_MOE_BACKEND=pallas_feature
    readonly HLO_BACKEND_CONTRACT=tpu_v4_pp8_pallas_feature
    ;;
  pallas_feature_linear)
    readonly RUNTIME_TAG=greenfield_runtime_feature_pack_pp8_20260806T064010287072141Z
    readonly RUNTIME_ROOT=/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/PP8_LP4/$RUNTIME_TAG
    readonly RUNTIME_MANIFEST_SHA=54e2f89b1832b994acbf9ef36f5f6ce68c942d9146efc4d7c15360d68b6d9917
    readonly RUNTIME_LAYOUT_HASH=ba21c4ec1500837f17a53047da98a7ed7c3a06782ddffe49d8bd796d4d0d1c9e
    readonly SPARSE_MOE_BACKEND=pallas_feature
    readonly HLO_BACKEND_CONTRACT=tpu_v4_pp8_pallas_feature_linear
    ;;
  *)
    echo "unsupported decoder runtime kind: $RUNTIME_KIND" >&2
    exit 2
    ;;
esac
if [[ $RUNTIME_KIND == reference && $FEATURE_OUTPUT_TILE != 128 ]]; then
  echo "a non-default feature output tile requires a feature runtime" >&2
  exit 2
fi
if [[ $RUNTIME_KIND == reference && $FEATURE_FUSE_ROUTE_WEIGHTING != 0 ]]; then
  echo "feature route-weight fusion requires a feature runtime" >&2
  exit 2
fi
if [[ $RUNTIME_KIND == reference && $FEATURE_RECONSTRUCT_DOWN_FP32 != 0 ]]; then
  echo "feature FP32 reconstruction requires a feature runtime" >&2
  exit 2
fi
if [[ $FEATURE_RECONSTRUCT_DOWN_FP32 == 1 && $FEATURE_FUSE_ROUTE_WEIGHTING == 1 ]]; then
  echo "feature FP32 reconstruction is incompatible with route-weight fusion" >&2
  exit 2
fi

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
ORACLE_PIN=$(git -C "$ORACLE_REPO" rev-parse HEAD)
TILE_SUFFIX=
if [[ $FEATURE_OUTPUT_TILE != 128 ]]; then
  TILE_SUFFIX=_ot${FEATURE_OUTPUT_TILE}
fi
readonly TILE_SUFFIX
FUSION_SUFFIX=
if [[ $FEATURE_FUSE_ROUTE_WEIGHTING == 1 ]]; then
  FUSION_SUFFIX=_wsum
fi
readonly FUSION_SUFFIX
RECONSTRUCTION_SUFFIX=
if [[ $FEATURE_RECONSTRUCT_DOWN_FP32 == 1 ]]; then
  RECONSTRUCTION_SUFFIX=_downf32
fi
readonly RECONSTRUCTION_SUFFIX
TOKEN_SUFFIX=
if [[ $COMPLETE_TOKEN_PATH == 1 ]]; then
  TOKEN_SUFFIX=_token
fi
readonly TOKEN_SUFFIX
ORACLE_SUFFIX=
if [[ $SHORT_CONTEXT_ORACLE == 1 ]]; then
  ORACLE_SUFFIX=_oracle
fi
if [[ $SHORT_CONTEXT_DSA_ORACLE == 1 ]]; then
  ORACLE_SUFFIX=_oracle_dsa
fi
readonly ORACLE_SUFFIX
TAG=${GLM_GREENFIELD_SHORT_DECODER_TAG:-greenfield_short_decoder_compile_pp8_${RUNTIME_KIND}${TILE_SUFFIX}${RECONSTRUCTION_SUFFIX}${FUSION_SUFFIX}${TOKEN_SUFFIX}${ORACLE_SUFFIX}_trace${TRACE_STEPS}_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing decoder compile outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing decoder compile from a dirty worktree" >&2
  exit 2
}
[[ -f $SOURCE_ROOT/SUCCESS && -f $SOURCE_RUNTIME_ROOT/SUCCESS && -f $RUNTIME_ROOT/SUCCESS ]] || {
  echo "protected source/runtime checkpoint is unavailable" >&2
  exit 2
}
if [[ $SHORT_CONTEXT_ORACLE == 1 ]] && {
  [[ ! -f $SHORT_CONTEXT_ORACLE_ROOT/SUCCESS ]] ||
    [[ ! -f $SHORT_CONTEXT_ORACLE_DIR/manifest.json ]] ||
    [[ ! -f $SHORT_CONTEXT_ORACLE_DIR/tokens.safetensors ]]
}; then
  echo "protected short-context token oracle is unavailable" >&2
  exit 2
fi
if [[ $SHORT_CONTEXT_DSA_ORACLE == 1 ]] && {
  [[ ! -f $SHORT_CONTEXT_DSA_ORACLE_ROOT/SUCCESS ]] ||
    [[ ! -f $SHORT_CONTEXT_DSA_ORACLE_DIR/manifest.json ]] ||
    [[ ! -f $SHORT_CONTEXT_DSA_ORACLE_DIR/dsa_events.safetensors ]]
}; then
  echo "protected short-context DSA oracle is unavailable" >&2
  exit 2
fi
[[ -r $RESULTS_DB ]] || {
  echo "results database is unavailable" >&2
  exit 2
}
[[ ! -e $RUN_DIR ]] || {
  echo "append-only run directory exists: $RUN_DIR" >&2
  exit 2
}
mkdir -p "$RUN_DIR/host_records" "$RUN_DIR/host_logs" "$RUN_DIR/hlo" \
  "$RUN_DIR/traces" "$RUN_DIR/dsa_observer"

say() {
  echo "[short-decoder-pp8 $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
}

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  say "ABORT: another protected pod workflow holds the global lease"
  exit 1
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
  local ray_enum
  # shellcheck disable=SC2016
  ray_enum='GLM_CENSUS_CARRIER='"$carrier"' /home/gianl/vllm-env/bin/python -c "import os,psutil,subprocess; from ray.autoscaler._private.constants import RAY_PROCESSES; carrier=os.environ[\"GLM_CENSUS_CARRIER\"]; marked={p.pid for p in psutil.process_iter([\"environ\"]) if (p.info[\"environ\"] or {}).get(\"GLM_CENSUS_CARRIER\")==carrier}; me=psutil.Process(); skip={me.pid}|{p.pid for p in me.parents()}|marked; out={p.pid for p in psutil.process_iter([\"name\",\"cmdline\"]) if p.pid not in skip and any(k in ((p.info[\"name\"] or \"\") if f else subprocess.list2cmdline(p.info[\"cmdline\"] or [])) for k,f in RAY_PROCESSES)}; print(\" \".join(map(str,sorted(out))))"'
  local command
  # shellcheck disable=SC2016
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[g]lm_longctx[.]py|[d]sa_throughput[.]py|[m]icrobench_collectives[.]py|[m]icrobench_pipeline_transport[.]py|[r]un_real_one_layer[.]py|[l]oad_full_checkpoint_stage[.]py|[p]ack_runtime_checkpoint[.]py|[c]ompile_short_decoder[.]py" 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray_stop_pids: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; echo "$containers" | grep -Ei "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt" || true; else echo "CENSUS_OK $(hostname)"; fi'
  GLM_CENSUS_CARRIER="$carrier" gcloud compute tpus tpu-vm ssh "$POD" \
    --zone "$ZONE" --worker=all --command="$command" >"$out" 2>&1 || return 1
  has_eight_unique_markers "$out" CENSUS_OK
}

post_census_done=0
on_exit() {
  local status=$?
  if [[ $post_census_done -eq 0 ]]; then
    strict_census failure_exit || true
  fi
  if [[ $status -ne 0 ]]; then
    say "FAILED status=$status; preserving diagnostics at $RUN_DIR"
    gcloud storage cp --recursive --no-clobber "$RUN_DIR" \
      "$REMOTE_PREFIX/diagnostic_local/" >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT

say "RUN_DIR=$RUN_DIR PIN=$PIN RUNTIME_KIND=$RUNTIME_KIND FEATURE_OUTPUT_TILE=$FEATURE_OUTPUT_TILE FEATURE_FUSE_ROUTE_WEIGHTING=$FEATURE_FUSE_ROUTE_WEIGHTING FEATURE_RECONSTRUCT_DOWN_FP32=$FEATURE_RECONSTRUCT_DOWN_FP32 COMPLETE_TOKEN_PATH=$COMPLETE_TOKEN_PATH SHORT_CONTEXT_ORACLE=$SHORT_CONTEXT_ORACLE SHORT_CONTEXT_DSA_ORACLE=$SHORT_CONTEXT_DSA_ORACLE WARMUP=$WARMUP ITERATIONS=$ITERATIONS TRACE_STEPS=$TRACE_STEPS"
say "RUNTIME=$RUNTIME_MANIFEST_SHA SOURCE_RUNTIME=$SOURCE_RUNTIME_MANIFEST_SHA SOURCE=$SOURCE_MANIFEST_SHA"
strict_census pre || {
  say "ABORT: pre-run census is not eight-host zero work"
  exit 1
}

say "syncing exact code and runtime/source artifacts"
# shellcheck disable=SC2016
sync_command='set -euo pipefail; pin='"$PIN"'; branch='"$BRANCH"'; origin='"$GREENFIELD_ORIGIN"'; wt='"$WORKTREE"'; source_root='"$SOURCE_ROOT"'; source_runtime_root='"$SOURCE_RUNTIME_ROOT"'; runtime_root='"$RUNTIME_ROOT"'; oracle_mode='"$SHORT_CONTEXT_ORACLE"'; oracle_root='"$SHORT_CONTEXT_ORACLE_ROOT"'; oracle_dir='"$SHORT_CONTEXT_ORACLE_DIR"'; dsa_oracle_mode='"$SHORT_CONTEXT_DSA_ORACLE"'; dsa_oracle_root='"$SHORT_CONTEXT_DSA_ORACLE_ROOT"'; dsa_oracle_dir='"$SHORT_CONTEXT_DSA_ORACLE_DIR"'; if [[ ${HOSTNAME##*-w-} == 0 ]]; then [[ -e "$wt/.git" ]] && [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]]; else if [[ -e "$wt/.git" ]]; then [[ -z $(git -C "$wt" status --porcelain) ]]; git -C "$wt" fetch -q origin "$branch"; git -C "$wt" checkout -q --detach "$pin"; elif [[ -e "$wt" ]]; then echo "stale non-repository path $wt" >&2; exit 1; else git clone -q --filter=blob:none --no-checkout --single-branch --branch "$branch" "$origin" "$wt"; git -C "$wt" checkout -q --detach "$pin"; fi; fi; oracle_ok=1; if [[ $oracle_mode == 1 ]]; then [[ -r "$oracle_root/SUCCESS" && -r "$oracle_dir/manifest.json" && -r "$oracle_dir/tokens.safetensors" ]] && findmnt -T "$oracle_root" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse" || oracle_ok=0; fi; dsa_oracle_ok=1; if [[ $dsa_oracle_mode == 1 ]]; then [[ -r "$dsa_oracle_root/SUCCESS" && -r "$dsa_oracle_dir/manifest.json" && -r "$dsa_oracle_dir/dsa_events.safetensors" ]] && findmnt -T "$dsa_oracle_root" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse" || dsa_oracle_ok=0; fi; [[ $oracle_ok == 1 && $dsa_oracle_ok == 1 ]] && [[ $(git -C "$wt" rev-parse HEAD) == "$pin" ]] && [[ -z $(git -C "$wt" status --porcelain) ]] && [[ -r "$source_root/SUCCESS" ]] && [[ -r "$source_runtime_root/SUCCESS" ]] && [[ -r "$runtime_root/SUCCESS" ]] && findmnt -T "$source_runtime_root" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse" && findmnt -T "$runtime_root" -n -o SOURCE,FSTYPE | grep -q "driftbench-dsv4-uc fuse.gcsfuse" && echo "SYNC_OK $(hostname) $pin"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$sync_command" >"$RUN_DIR/sync.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/sync.txt" SYNC_OK || {
  say "ABORT: eight-host sync/artifact prerequisite failed"
  exit 1
}

coordinator=$(gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=0 \
  --command="hostname -I" 2>/dev/null | grep -Eo '192\.168\.[0-9]+\.[0-9]+' | head -1)
[[ -n $coordinator ]] || {
  say "ABORT: worker-0 coordinator address is unavailable"
  exit 1
}
coordinator="$coordinator:8476"
say "launching real 78-layer 2K load/compile coordinator=$coordinator"
# shellcheck disable=SC2016
execute_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; tag='"$TAG"'; wt='"$WORKTREE"'; remote='"$REMOTE_PREFIX"'; feature_output_tile='"$FEATURE_OUTPUT_TILE"'; feature_fuse_route_weighting='"$FEATURE_FUSE_ROUTE_WEIGHTING"'; feature_reconstruct_down_fp32='"$FEATURE_RECONSTRUCT_DOWN_FP32"'; complete_token_path='"$COMPLETE_TOKEN_PATH"'; short_context_oracle='"$SHORT_CONTEXT_ORACLE"'; oracle_dir='"$SHORT_CONTEXT_ORACLE_DIR"'; oracle_sha='"$SHORT_CONTEXT_ORACLE_MANIFEST_SHA"'; short_context_dsa_oracle='"$SHORT_CONTEXT_DSA_ORACLE"'; dsa_oracle_dir='"$SHORT_CONTEXT_DSA_ORACLE_DIR"'; dsa_oracle_sha='"$SHORT_CONTEXT_DSA_ORACLE_MANIFEST_SHA"'; run=/home/gianl/glm-run/$tag; mkdir -p "$run/hlo"; output="$run/decoder.rank${idx}.json"; log="$run/decoder.rank${idx}.log"; upload() { gcloud storage cp --no-clobber "$log" "$output" "$remote/host_records/" >/dev/null 2>&1 || true; if compgen -G "$run/hlo/*" >/dev/null; then gcloud storage cp --no-clobber "$run"/hlo/* "$remote/hlo/" >/dev/null 2>&1 || true; fi; if compgen -G "$run/dsa_observer/*" >/dev/null; then gcloud storage cp --no-clobber "$run"/dsa_observer/* "$remote/dsa_observer/" >/dev/null 2>&1 || true; fi; xplane=$(find "$run/trace" -type f -name "*.xplane.pb" 2>/dev/null | head -1 || true); if [[ -n $xplane ]]; then gcloud storage cp --no-clobber "$xplane" "$remote/traces/trace.rank${idx}.xplane.pb" >/dev/null 2>&1 || true; fi; }; trap upload EXIT; cd "$wt"; trace_args=(); if [[ '"$TRACE_STEPS"' -gt 0 ]]; then trace_args=(--trace-root "$run/trace" --trace-steps '"$TRACE_STEPS"'); fi; oracle_args=(); if [[ $short_context_oracle == 1 ]]; then oracle_args=(--short-context-oracle-dir "$oracle_dir" --short-context-oracle-manifest-sha256 "$oracle_sha"); fi; dsa_oracle_args=(); if [[ $short_context_dsa_oracle == 1 ]]; then dsa_oracle_args=(--short-context-dsa-oracle-dir "$dsa_oracle_dir" --short-context-dsa-oracle-manifest-sha256 "$dsa_oracle_sha"); fi; env JAX_PLATFORMS=tpu XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH="$wt" GLM_GREENFIELD_RUN_TAG="$tag" timeout --signal=TERM --kill-after=60 10800 /home/gianl/vllm-env/bin/python -u scripts/greenfield/compile_short_decoder.py --coordinator-address '"$coordinator"' --num-processes 8 --process-id "$idx" --expected-code-hash '"$PIN"' --runtime-kind '"$RUNTIME_KIND"' --feature-output-tile "$feature_output_tile" --feature-fuse-route-weighting "$feature_fuse_route_weighting" --feature-reconstruct-down-fp32 "$feature_reconstruct_down_fp32" --complete-token-path "$complete_token_path" --runtime-root '"$RUNTIME_ROOT"' --runtime-manifest-sha256 '"$RUNTIME_MANIFEST_SHA"' --source-runtime-root '"$SOURCE_RUNTIME_ROOT"' --source-runtime-manifest-sha256 '"$SOURCE_RUNTIME_MANIFEST_SHA"' --source-checkpoint-root '"$SOURCE_ROOT"' --source-packed-manifest-sha256 '"$SOURCE_MANIFEST_SHA"' --context-capacity 2048 --warmup '"$WARMUP"' --iterations '"$ITERATIONS"' "${trace_args[@]}" "${oracle_args[@]}" "${dsa_oracle_args[@]}" --output "$output" >"$log" 2>&1; trap - EXIT; upload; echo "DECODER_HOST_OK $(hostname) rank=$idx"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$execute_command" >"$RUN_DIR/execute.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/execute.txt" DECODER_HOST_OK || {
  say "ABORT: real decoder load/compile did not pass 8/8"
  exit 1
}

gcloud storage cp "$REMOTE_PREFIX/host_records/decoder.rank*.json" \
  "$RUN_DIR/host_records/" >/dev/null
gcloud storage cp "$REMOTE_PREFIX/host_records/decoder.rank*.log" \
  "$RUN_DIR/host_logs/" >/dev/null
gcloud storage cp "$REMOTE_PREFIX/hlo/*" "$RUN_DIR/hlo/" >/dev/null
if [[ $TRACE_STEPS -gt 0 ]]; then
  gcloud storage cp "$REMOTE_PREFIX/traces/trace.rank*.xplane.pb" \
    "$RUN_DIR/traces/" >/dev/null
fi
if [[ $SHORT_CONTEXT_DSA_ORACLE == 1 ]]; then
  gcloud storage cp "$REMOTE_PREFIX/dsa_observer/*" \
    "$RUN_DIR/dsa_observer/" >/dev/null
fi

say "validating fleet agreement and recording diagnostic DB linkage"
/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$PIN" "$ORACLE_PIN" \
  "$RESULTS_DB" "$WORKTREE" "$ORACLE_REPO" "$RUNTIME_KIND" \
  "$FEATURE_OUTPUT_TILE" "$FEATURE_FUSE_ROUTE_WEIGHTING" "$FEATURE_RECONSTRUCT_DOWN_FP32" "$COMPLETE_TOKEN_PATH" "$SHORT_CONTEXT_ORACLE" "$SHORT_CONTEXT_ORACLE_MANIFEST_SHA" "$SHORT_CONTEXT_DSA_ORACLE" "$SHORT_CONTEXT_DSA_ORACLE_MANIFEST_SHA" "$SPARSE_MOE_BACKEND" "$HLO_BACKEND_CONTRACT" "$RUNTIME_MANIFEST_SHA" \
  "$RUNTIME_LAYOUT_HASH" "$WARMUP" "$ITERATIONS" "$TRACE_STEPS" <<'PY'
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sqlite3
import sys

import numpy as np

(
    run_dir,
    pin,
    oracle_pin,
    db_path,
    repo,
    oracle_repo,
    runtime_kind,
    feature_output_tile,
    feature_fuse_route_weighting,
    feature_reconstruct_down_fp32,
    complete_token_path,
    short_context_oracle,
    short_context_oracle_manifest_sha256,
    short_context_dsa_oracle,
    short_context_dsa_oracle_manifest_sha256,
    sparse_moe_backend,
    hlo_backend_contract,
    runtime_manifest_sha256,
    runtime_layout_hash,
    warmup,
    iterations,
    trace_steps,
) = sys.argv[1:]
feature_output_tile = int(feature_output_tile)
feature_fuse_route_weighting = bool(int(feature_fuse_route_weighting))
feature_reconstruct_down_fp32 = bool(int(feature_reconstruct_down_fp32))
complete_token_path = bool(int(complete_token_path))
short_context_oracle = bool(int(short_context_oracle))
short_context_dsa_oracle = bool(int(short_context_dsa_oracle))
warmup = int(warmup)
iterations = int(iterations)
trace_steps = int(trace_steps)
run_dir = Path(run_dir)
records = [json.loads(path.read_text()) for path in sorted((run_dir / "host_records").glob("*.json"))]
if len(records) != 8:
    raise SystemExit(f"expected eight host records, got {len(records)}")
if {record["launch_process_id"] for record in records} != set(range(8)):
    raise SystemExit("launch process ids do not cover 0..7")
if {record["jax_process_index"] for record in records} != set(range(8)):
    raise SystemExit("JAX process indices do not cover 0..7")
if len({record["hostname"] for record in records}) != 8:
    raise SystemExit("host records are not fleet-distinct")
for field in (
    "code_hash",
    "dsa_observer_hlo_sha256",
    "optimized_hlo_sha256",
    "plan_hash",
    "runtime_layout_hash",
    "runtime_manifest_sha256",
    "schedule_hash",
    "state_layout_hash",
    "topology_hash",
):
    values = {record[field] for record in records}
    if len(values) != 1:
        raise SystemExit(f"fleet field {field} disagrees: {sorted(values)}")
if {record["code_hash"] for record in records} != {pin}:
    raise SystemExit("fleet used stale code")
if {record["runtime_kind"] for record in records} != {runtime_kind}:
    raise SystemExit("fleet runtime kind drifted")
if {record["feature_output_tile"] for record in records} != {
    feature_output_tile
}:
    raise SystemExit("fleet feature output tile drifted")
if {record["feature_fuse_route_weighting"] for record in records} != {
    feature_fuse_route_weighting
}:
    raise SystemExit("fleet feature route-weight fusion drifted")
if {record["feature_reconstruct_down_fp32"] for record in records} != {
    feature_reconstruct_down_fp32
}:
    raise SystemExit("fleet feature FP32 reconstruction drifted")
if {record["complete_token_path"] for record in records} != {
    complete_token_path
}:
    raise SystemExit("fleet complete token-path flag drifted")
if {record["prefill_used"] for record in records} != {
    short_context_oracle
}:
    raise SystemExit("fleet short-context prefill flag drifted")
if {record["schema_version"] for record in records} != {7}:
    raise SystemExit("fleet decoder record schema drifted")
if short_context_oracle:
    for field in ("prefill_hlo_sha256",):
        values = {record[field] for record in records}
        if len(values) != 1 or None in values:
            raise SystemExit(f"fleet field {field} disagrees: {values}")
if {record["sparse_moe_backend"] for record in records} != {sparse_moe_backend}:
    raise SystemExit("fleet sparse MoE backend drifted")
expected_linear_backend = (
    "pallas" if runtime_kind == "pallas_feature_linear" else "reference"
)
if {record["linear_backend"] for record in records} != {expected_linear_backend}:
    raise SystemExit("fleet FP8 linear backend drifted")
if {record["runtime_manifest_sha256"] for record in records} != {runtime_manifest_sha256}:
    raise SystemExit("fleet runtime manifest drifted")
if {record["runtime_layout_hash"] for record in records} != {runtime_layout_hash}:
    raise SystemExit("fleet runtime layout drifted")
for record in records:
    if (
        record["body_only"] == complete_token_path
        or record["transformer_body_timing_only"] == complete_token_path
        or record["raw_token_claim"] != short_context_oracle
        or not record["hlo_contract"]["passed"]
        or not record["metadata_passed"]
        or not record["token_passed"]
    ):
        raise SystemExit("step/HLO/metadata claim contract failed")
    if complete_token_path:
        token = record["token_contract"]
        if (
            token is None
            or token["synthetic_initial_state"] == short_context_oracle
            or token["prefill_used"] != short_context_oracle
            or not token["all_active_lanes_equal"]
            or not token["all_in_vocabulary"]
            or len(token["profiler_free_window_tokens"]) != iterations
        ):
            raise SystemExit("complete token-path output contract failed")
        if short_context_oracle:
            raw = token["raw_token_sequence"]
            oracle = token["short_context_oracle"]
            if (
                raw is None
                or not raw["exact_prefix_match"]
                or raw["compared_token_count"] != 1 + warmup + iterations
                or raw["observed_token_ids"] != raw["expected_token_ids"]
                or oracle is None
                or oracle["manifest_sha256"]
                != short_context_oracle_manifest_sha256
                or oracle["prompt_token_count"] != 2034
                or record["prefill_hlo_contract"] is None
                or not record["prefill_hlo_contract"]["passed"]
                or record["prefill_hlo_contract"]["loop_count"] != 1
                or record["prefill_compile_seconds"] is None
                or record["prefill_wall_ms"] is None
            ):
                raise SystemExit("real-prompt token/prefill contract failed")
        elif (
            token["raw_token_sequence"] is not None
            or token["short_context_oracle"] is not None
            or record["prefill_hlo_contract"] is not None
            or record["prefill_hlo_sha256"] is not None
        ):
            raise SystemExit("synthetic token path contains oracle evidence")
    elif record["token_contract"] is not None:
        raise SystemExit("body-only record unexpectedly contains a token")
if complete_token_path and len(
    {json.dumps(record["token_contract"], sort_keys=True) for record in records}
) != 1:
    raise SystemExit("fleet token contracts disagree")
if any(
    record["warmup"] != warmup
    or record["iterations"] != iterations
    or record["profiler_free_body_wall"]["count"] != iterations
    for record in records
):
    raise SystemExit("decoder timing configuration drifted")
if any(record["hlo_contract"]["violations"] for record in records):
    raise SystemExit("decoder HLO has violations")
if {record["hlo_contract"]["complete_token_path"] for record in records} != {
    complete_token_path
}:
    raise SystemExit("decoder HLO token-path contract drifted")
if any(
    record["hlo_contract"]["backend_contract"] != hlo_backend_contract
    for record in records
):
    raise SystemExit("decoder HLO backend contract drifted")
if short_context_oracle:
    for record in records:
        prefill = record["prefill_hlo_contract"]
        if (
            prefill["backend_contract"] != hlo_backend_contract
            or prefill["prompt_length"] != 2034
            or prefill["violations"]
            or len(set(record["fleet_prefill_hlo_hashes"])) != 1
            or record["fleet_prefill_hlo_hashes"][0]
            != record["prefill_hlo_sha256"]
        ):
            raise SystemExit("prefill HLO/fleet contract drifted")
if short_context_dsa_oracle:
    if len(
        {
            json.dumps(record["dsa_observer_contract"], sort_keys=True)
            for record in records
        }
    ) != 1:
        raise SystemExit("fleet DSA observer contracts disagree")
    for record in records:
        dsa = record["dsa_observer_contract"]
        observer_hlo = record["dsa_observer_hlo_contract"]
        isolation = record["dsa_observer_isolation_contract"]
        steps = dsa["step_records"] if dsa is not None else []
        if (
            dsa is None
            or not dsa["passed"]
            or not dsa["all_steps_passed"]
            or dsa["decode_step_count"] != 14
            or dsa["event_count"] != 21
            or dsa["manifest_sha256"]
            != short_context_dsa_oracle_manifest_sha256
            or not dsa["observer_executed_before_production"]
            or not dsa["prefill_state_preserved_without_donation"]
            or dsa["production_executable_observer_enabled"]
            or not dsa["score_comparison"]["compared"]
            or dsa["score_comparison"]["cross_backend_total_order_is_gate"]
            or not dsa["score_comparison"][
                "executing_score_order_and_ties_are_gate"
            ]
            or dsa["score_comparison"][
                "legacy_scores_use_position_aligned_bounded_gate"
            ]
            or not dsa["score_comparison"][
                "legacy_scores_use_position_aligned_diagnostic"
            ]
            or dsa["score_comparison"]["legacy_score_tolerance"]
            != {"max_abs": 0.125, "mean_abs": 0.01, "p99_abs": 0.03125}
            or not dsa["score_comparison"][
                "selected_set_against_legacy_is_exact_gate"
            ]
            or len(dsa["observation_artifacts"]) != 14
            or dsa["token_observation_candidates"] != 16
            or dsa["prefill_token_sequence"] is None
            or not dsa["prefill_token_sequence"]["exact_prefix_match"]
            or dsa["prefill_token_sequence"]["compared_token_count"] != 1
            or dsa["prefill_token_sequence"]["observed_token_ids"]
            != dsa["prefill_token_sequence"]["expected_token_ids"]
            or dsa["token_oracle_offset"] != 1
            or not dsa["token_sequence"]["exact_prefix_match"]
            or dsa["token_sequence"]["compared_token_count"] != 14
            or dsa["token_sequence"]["observed_token_ids"]
            != dsa["token_sequence"]["expected_token_ids"]
            or [step["decode_position"] for step in steps]
            != list(range(2034, 2048))
            or not all(
                step["passed"]
                and step["position_passed"]
                and step["exact_selected_set_and_tail"]
                and step["actual_device_score_order_and_ties"]
                and step["event_count"] == 21
                and not step["lane_mismatch_stages"]
                and not step["padded_slot_mismatches"]
                and not step["producer_mismatches"]
                and not step["count_mismatches"]
                and not step["selected_set_mismatches"]
                and not step["tail_mismatches"]
                and not step["score_contract_mismatches"]
                and step["legacy_score_bounded_comparison"][
                    "coverage_complete"
                ]
                and len(step["observation_sha256"]) == 64
                and len(step["token_observation_sha256"]) == 64
                and step["token_observation"]["passed"]
                and step["token_observation"]["candidate_width"] == 16
                and len(step["token_observation"]["candidate_ids"]) == 16
                and len(step["token_observation"]["candidate_scores"]) == 16
                and step["token_observation"]["lane_replication"]
                and step["token_observation"][
                    "inactive_rows_are_sentinel"
                ]
                and step["token_observation"]["ids_valid"]
                and step["token_observation"]["scores_finite"]
                and step["token_observation"]["order_and_ties_valid"]
                and step["token_observation"]["winner_matches_output"]
                for step in steps
            )
            or observer_hlo is None
            or not observer_hlo["passed"]
            or observer_hlo["backend_contract"] != hlo_backend_contract
            or observer_hlo["token_observation_candidates"] != 16
            or observer_hlo["collective_counts"]
            != record["hlo_contract"]["collective_counts"]
            or isolation is None
            or not isolation["passed"]
            or isolation["donate_argnums"]
            or isolation["input_output_alias_present"]
            or isolation["callback_markers"]
            or not isolation["collective_contract_matches_production"]
            or not isolation["non_token_result_shapes_match"]
            or not isolation["token_exchange_shape_difference_allowed"]
            or record["dsa_observer_compile_seconds"] is None
            or record["dsa_observer_hlo_sha256"] is None
            or len(set(record["fleet_dsa_observer_hlo_hashes"])) != 1
            or record["fleet_dsa_observer_hlo_hashes"][0]
            != record["dsa_observer_hlo_sha256"]
            or not record[
                "trace_and_timing_use_observer_free_production_executable"
            ]
        ):
            raise SystemExit("exact DSA observer/Gate D contract failed")
    artifact_records = records[0]["dsa_observer_contract"][
        "observation_artifacts"
    ]
    artifact_names = [record["filename"] for record in artifact_records]
    artifact_paths = sorted((run_dir / "dsa_observer").glob("*.npz"))
    if [path.name for path in artifact_paths] != artifact_names:
        raise SystemExit("DSA observer artifact inventory drifted")
    for record, path in zip(artifact_records, artifact_paths, strict=True):
        with np.load(path) as bundle:
            if set(bundle.files) != {
                "decode_position",
                "observation",
                "token_observation",
            }:
                raise SystemExit("DSA observer artifact fields drifted")
            decode_position = np.asarray(bundle["decode_position"])
            observation = np.asarray(bundle["observation"])
            token_observation = np.asarray(bundle["token_observation"])
        digest = hashlib.sha256(
            np.ascontiguousarray(observation).tobytes()
        ).hexdigest()
        token_digest = hashlib.sha256(
            np.ascontiguousarray(token_observation).tobytes()
        ).hexdigest()
        if (
            decode_position.shape != (1,)
            or int(decode_position[0])
            not in range(2034, 2048)
            or observation.shape != (32, 5, 4098)
            or observation.dtype != np.dtype(np.int32)
            or digest != record["observation_sha256"]
            or token_observation.shape != (32, 32)
            or token_observation.dtype != np.dtype(np.int32)
            or token_digest != record["token_observation_sha256"]
        ):
            raise SystemExit("DSA observer artifact tensor/hash drifted")
else:
    for record in records:
        if any(
            record[name] is not None
            for name in (
                "dsa_observer_compile_seconds",
                "dsa_observer_contract",
                "dsa_observer_hlo_contract",
                "dsa_observer_hlo_sha256",
                "dsa_observer_isolation_contract",
                "fleet_dsa_observer_hlo_hashes",
            )
        ):
            raise SystemExit("unrequested DSA observer evidence is present")
if runtime_kind in ("pallas_feature", "pallas_feature_linear"):
    selected_kernel = "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512"
    if feature_output_tile != 128:
        selected_kernel += f"_ot{feature_output_tile}"
    if feature_reconstruct_down_fp32:
        selected_kernel += "_downf32"
    if feature_fuse_route_weighting:
        selected_kernel += "_wsum"
    expected_kernel_counts = {
        selected_kernel: 75,
        "greenfield_fp8_block_up_gate_m8_k6144_n512": 75,
        "greenfield_fp8_block_matmul_m8_k512_n6144": 75,
    }
    for record in records:
        feature = record["hlo_contract"]["pallas_feature_contract"]
        if (
            not feature["passed"]
            or feature["feature_output_tile"] != feature_output_tile
            or feature["fuse_route_weighting"]
            != feature_fuse_route_weighting
            or feature["reconstruct_down_fp32"]
            != feature_reconstruct_down_fp32
            or feature["kernel_counts"] != expected_kernel_counts
            or feature["expected_kernel_counts"] != expected_kernel_counts
            or feature["forbidden_decoded_expert_overlays"]
        ):
            raise SystemExit("feature-Pallas HLO kernel/overlay contract drifted")
if runtime_kind == "pallas_feature_linear":
    expected_linear_kernel_counts = {
        "greenfield_fp8_block_matmul_m8_k6144_n2048": 78,
        "greenfield_fp8_block_matmul_m8_k2048_n4096": 78,
        "greenfield_fp8_block_matmul_m8_k6144_n640": 78,
        "greenfield_fp8_block_matmul_m8_k4096_n6144": 78,
        "greenfield_fp8_structured_kv_b_q_absorb_h16_p192_l512": 78,
        "greenfield_fp8_structured_kv_b_value_h16_l512_v256": 78,
        "greenfield_fp8_block_matmul_f32_m8_k2048_n1024": 21,
        "greenfield_fp8_block_matmul_f32_m8_k6144_n128": 21,
        "greenfield_fp8_fused_block_swiglu_m8_h6144_i3072_o6144": 3,
    }
    for record in records:
        linear = record["hlo_contract"]["pallas_stage_linear_contract"]
        if (
            not linear["passed"]
            or linear["kernel_counts"] != expected_linear_kernel_counts
            or linear["expected_kernel_counts"]
            != expected_linear_kernel_counts
            or linear["forbidden_decoded_weight_overlays"]
        ):
            raise SystemExit("stage-linear Pallas HLO contract drifted")
if any(record["load_record"]["runtime_checkpoint_reshards"] != 0 for record in records):
    raise SystemExit("runtime loader performed a checkpoint reshard")
if any(record["load_record"]["host_global_concatenations"] != 0 for record in records):
    raise SystemExit("runtime loader performed a host global concat")
if any(record["load_record"]["fp8_host_dequantizations"] != 0 for record in records):
    raise SystemExit("runtime loader performed a host FP8 dequantization")
if any(record["load_record"]["fp8_device_dequantizations"] != 0 for record in records):
    raise SystemExit("runtime loader performed a device FP8 dequantization")
if any(record["load_record"]["loaded_payload_bytes"] != 104_272_169_728 for record in records):
    raise SystemExit("runtime loader payload bytes per host drifted")
xplane = None
if trace_steps:
    traces = sorted((run_dir / "traces").glob("trace.rank*.xplane.pb"))
    if len(traces) != 8:
        raise SystemExit(f"expected eight decoder XPlanes, got {len(traces)}")
    for record, path in zip(records, traces, strict=True):
        trace = record["trace"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (
            trace is None
            or trace["steps"] != trace_steps
            or not trace["profiler_started_after_profiler_free_timing"]
            or len(trace["files"]) != 1
            or trace["files"][0]["sha256"] != digest
            or trace["files"][0]["size_bytes"] != path.stat().st_size
        ):
            raise SystemExit("decoder trace record drifted")
    sys.path.insert(0, str(Path(repo) / "scripts" / "analysis"))
    import parse_xplane

    xplane = parse_xplane.aggregate_fleet(
        run_dir / "traces",
        step_module_re=r"jit_mapped",
    )
    if (
        xplane["n_files"] != 8
        or xplane["n_cores"] != 64
        or xplane["steps_per_core"] != trace_steps
    ):
        raise SystemExit("decoder fleet XPlane inventory drifted")
else:
    if any(record["trace"] is not None for record in records):
        raise SystemExit("unrequested decoder trace was captured")

def peak(record):
    values = []
    for stats in record["device_memory_after_execute"]:
        if stats is not None:
            for key in ("peak_bytes_in_use", "peak_bytes", "bytes_in_use"):
                if key in stats:
                    values.append(stats[key])
                    break
    return max(values) if values else None

fleet_p50 = max(record["profiler_free_body_wall"]["p50_ms"] for record in records)
fleet_p99 = max(record["profiler_free_body_wall"]["p99_ms"] for record in records)
peaks = [value for value in map(peak, records) if value is not None]
summary = {
    "artifact_kind": (
        "greenfield_real_78layer_2k_decoder_"
        + (
            "gate_d_fleet"
            if short_context_dsa_oracle
            else (
                "token_oracle_fleet"
                if short_context_oracle
                else (
                    "token_mechanism_fleet"
                    if complete_token_path
                    else "body_fleet"
                )
            )
        )
    ),
    "answer_tokens_per_second": (
        1000.0 / fleet_p50 if short_context_dsa_oracle else None
    ),
    "body_only": not complete_token_path,
    "code_hash": pin,
    "compile_seconds_max": max(record["compile_seconds"] for record in records),
    "context_capacity": 2048,
    "fleet_p50_body_ms": fleet_p50,
    "fleet_p99_body_ms": fleet_p99,
    "fleet_p50_complete_step_ms": (
        fleet_p50 if complete_token_path else None
    ),
    "fleet_p99_complete_step_ms": (
        fleet_p99 if complete_token_path else None
    ),
    "feature_output_tile": feature_output_tile,
    "feature_fuse_route_weighting": feature_fuse_route_weighting,
    "feature_reconstruct_down_fp32": feature_reconstruct_down_fp32,
    "complete_token_path": complete_token_path,
    "dsa_observer_compile_seconds_max": (
        max(record["dsa_observer_compile_seconds"] for record in records)
        if short_context_dsa_oracle
        else None
    ),
    "dsa_observer_contract": records[0]["dsa_observer_contract"],
    "dsa_observer_hlo_contract": records[0]["dsa_observer_hlo_contract"],
    "dsa_observer_hlo_sha256": records[0]["dsa_observer_hlo_sha256"],
    "dsa_observer_isolation_contract": records[0][
        "dsa_observer_isolation_contract"
    ],
    "gate_d_passed": short_context_dsa_oracle,
    "hlo_contract": records[0]["hlo_contract"],
    "host_count": 8,
    "iterations": iterations,
    "maximum_peak_hbm_bytes": max(peaks) if peaks else None,
    "optimized_hlo_sha256": records[0]["optimized_hlo_sha256"],
    "plan_hash": records[0]["plan_hash"],
    "prefill_compile_seconds_max": (
        max(record["prefill_compile_seconds"] for record in records)
        if short_context_oracle
        else None
    ),
    "prefill_hlo_contract": records[0]["prefill_hlo_contract"],
    "prefill_hlo_sha256": records[0]["prefill_hlo_sha256"],
    "prefill_used": short_context_oracle,
    "prefill_wall_ms_max": (
        max(record["prefill_wall_ms"] for record in records)
        if short_context_oracle
        else None
    ),
    "raw_token_claim": short_context_oracle,
    "runtime_kind": runtime_kind,
    "runtime_layout_hash": records[0]["runtime_layout_hash"],
    "runtime_manifest_sha256": records[0]["runtime_manifest_sha256"],
    "schedule_hash": records[0]["schedule_hash"],
    "state_layout_hash": records[0]["state_layout_hash"],
    "sparse_moe_backend": sparse_moe_backend,
    "topology_hash": records[0]["topology_hash"],
    "trace_steps": trace_steps,
    "short_context_oracle_manifest_sha256": (
        short_context_oracle_manifest_sha256
        if short_context_oracle
        else None
    ),
    "short_context_dsa_oracle_manifest_sha256": (
        short_context_dsa_oracle_manifest_sha256
        if short_context_dsa_oracle
        else None
    ),
    "synthetic_initial_state": (
        complete_token_path and not short_context_oracle
    ),
    "token_contract": records[0]["token_contract"],
    "transformer_body_timing_only": not complete_token_path,
    "warmup": warmup,
    "xplane": xplane,
}
(run_dir / "xplane_summary.json").write_text(
    json.dumps(xplane, indent=2, sort_keys=True) + "\n"
)
sys.path.insert(0, str(Path(repo) / "bench"))
import provenance as pv

conn = pv.connect(db_path)
scope = (
    "gate-d"
    if short_context_dsa_oracle
    else (
        "token-oracle"
        if short_context_oracle
        else ("token-mechanism" if complete_token_path else "body")
    )
)
run_id = pv.start_run(
    conn,
    model=f"zai-org/GLM-5.2-FP8:greenfield-78layer-2k-{scope}",
    revision=records[0]["runtime_manifest_sha256"],
    env={
        "GLM_ENGINE": f"greenfield_pp8_decoder_{scope}",
        "greenfield_complete_token_path": complete_token_path,
        "greenfield_short_context_oracle": short_context_oracle,
        "greenfield_short_context_oracle_manifest_sha256": (
            short_context_oracle_manifest_sha256
            if short_context_oracle
            else None
        ),
        "greenfield_short_context_dsa_oracle": short_context_dsa_oracle,
        "greenfield_short_context_dsa_oracle_manifest_sha256": (
            short_context_dsa_oracle_manifest_sha256
            if short_context_dsa_oracle
            else None
        ),
        "greenfield_runtime_kind": runtime_kind,
        "greenfield_feature_output_tile": feature_output_tile,
        "greenfield_feature_fuse_route_weighting": (
            feature_fuse_route_weighting
        ),
        "greenfield_feature_reconstruct_down_fp32": (
            feature_reconstruct_down_fp32
        ),
        "greenfield_sparse_moe_backend": sparse_moe_backend,
        "greenfield_code_hash": pin,
        "legacy_oracle_code_hash": oracle_pin,
        "hlo_sha256": records[0]["optimized_hlo_sha256"],
        "dsa_observer_hlo_sha256": records[0][
            "dsa_observer_hlo_sha256"
        ],
        "plan_hash": records[0]["plan_hash"],
        "runtime_layout_hash": records[0]["runtime_layout_hash"],
        "runtime_manifest_sha256": records[0]["runtime_manifest_sha256"],
        "state_layout_hash": records[0]["state_layout_hash"],
    },
    note=(
        "Protected real 78-layer 2K transformer-body compile/run with "
        f"feature output tile {feature_output_tile} and fused route weighting "
        f"{feature_fuse_route_weighting}, FP32 routed-down reconstruction "
        f"{feature_reconstruct_down_fp32}; "
        + (
            "real 2,034-token prompt with exact raw tokens and exact all-event "
            "DSA observer evidence; protected 2K Gate D."
            if short_context_dsa_oracle
            else (
                "real 2,034-token prompt with an exact sealed raw-token prefix; "
                "not Gate D until full DSA event-order evidence passes."
                if short_context_oracle
                else (
                    "complete token mechanism from synthetic state, no correctness "
                    "or tok/s claim."
                    if complete_token_path
                    else "no token or tok/s claim."
                )
            )
        )
    ),
    harness_repo=repo,
    fork_repo=oracle_repo,
)
pv.record_item(
    conn,
    run_id,
    benchmark=(
        f"greenfield_78layer_2k_{scope}_pp8"
        + (f"_ot{feature_output_tile}" if feature_output_tile != 128 else "")
        + ("_downf32" if feature_reconstruct_down_fp32 else "")
        + ("_wsum" if feature_fuse_route_weighting else "")
    ),
    item_id=(
        "gate_d_exact_token_and_dsa"
        if short_context_dsa_oracle
        else (
            "raw_token_prefix"
            if short_context_oracle
            else ("token_step_mechanism" if complete_token_path else "body_step")
        )
    ),
    prompt=(
        "Execute the sealed 2,034-token short-context prompt through device-"
        "resident prefill and recurrent PP8 decode."
        if short_context_oracle
        else (
            "Execute the real 78-layer PP8 decoder step with "
            f"feature output tile {feature_output_tile} and fused route weighting "
            f"{feature_fuse_route_weighting}, FP32 routed-down reconstruction "
            f"{feature_reconstruct_down_fp32}."
        )
    ),
    gold=(
        "Exact tokens and executing-device DSA sets/ties; legacy scores retained diagnostically."
        if short_context_dsa_oracle
        else (
            "Exact raw token IDs against the sealed accepted legacy prefix."
            if short_context_oracle
            else (
                "Local-only HLO, exact pipeline state, direct runtime load, and "
                "bounded token mechanism without a raw-correctness claim."
            )
        )
    ),
    raw_output=json.dumps(summary, sort_keys=True),
    extracted=str(summary["hlo_contract"]["collective_counts"]),
    correct=True,
    score=1.0,
    latency_ms=fleet_p50,
)
pv.finalize(
    conn,
    run_id,
    benchmark=(
        f"greenfield_78layer_2k_{scope}_pp8"
        + (f"_ot{feature_output_tile}" if feature_output_tile != 128 else "")
        + ("_downf32" if feature_reconstruct_down_fp32 else "")
        + ("_wsum" if feature_fuse_route_weighting else "")
    ),
    metric="contract_valid",
    value=1.0,
    note=(
        "Protected real-prompt 2K Gate D passes exact raw tokens, all-event DSA, "
        "state/cache, observer isolation, wall/HBM, fresh XPlane, and cleanup."
        if short_context_dsa_oracle
        else (
            "Exact raw-token prefix and real-prompt recurrent wall pass; full Gate D "
            "still requires exact all-event DSA evidence."
            if short_context_oracle
            else (
                "Synthetic-state complete token timing is not raw-token correctness "
                "or answer tok/s."
                if complete_token_path
                else "Profiler-free body timing is not complete decode latency or tok/s."
            )
        )
    ),
)
conn.close()
summary["results_db_run_id"] = run_id
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
source = sqlite3.connect(db_path)
snapshot = sqlite3.connect(run_dir / "results_ckpt.db")
source.backup(snapshot)
snapshot.close()
source.close()
if sqlite3.connect(run_dir / "results_ckpt.db").execute("PRAGMA integrity_check").fetchone()[0] != "ok":
    raise SystemExit("results DB snapshot integrity failed")
print(f"SHORT_DECODER_STEP_VALID db_run={run_id} p50_ms={fleet_p50:.6f}")
PY

strict_census post || {
  say "ABORT: post-run census is not eight-host zero work"
  exit 1
}
post_census_done=1

say "sealing and archiving protected diagnostic evidence"
cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
(
  cd "$RUN_DIR"
  find host_records host_logs hlo traces dsa_observer -type f -print0 | \
    sort -z | xargs -0 sha256sum
  sha256sum summary.json xplane_summary.json results_ckpt.db census_pre.txt census_post.txt \
    sync.txt execute.txt orchestrator.sealed.log
) >"$RUN_DIR/evidence.sha256"
DB_RUN=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["results_db_run_id"])' \
  "$RUN_DIR/summary.json")
printf 'db_run=%s\n' "$DB_RUN" >"$RUN_DIR/SUCCESS"
gcloud storage cp --recursive --no-clobber "$RUN_DIR" "$REMOTE_PREFIX/" >/dev/null
gcloud storage cp "$RUN_DIR/SUCCESS" "$REMOTE_PREFIX/SUCCESS" --no-clobber >/dev/null
trap - EXIT
echo "SHORT_DECODER_STEP_OK $TAG DB=$DB_RUN"
