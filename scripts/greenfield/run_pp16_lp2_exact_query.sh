#!/usr/bin/env bash
# Protected seconds-scale real-weight PP16 LP2 exact-query discriminator.
set -euo pipefail

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly APPROVED_LOCATION=US-CENTRAL2
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly SOURCE_TAG=greenfield_ws32_layer0_dsa_association_20260816T101637335765581Z
readonly SOURCE_ROOT=/home/gianl/glm-run/$SOURCE_TAG/inputs
readonly SOURCE_REMOTE=$APPROVED_BUCKET/results/$SOURCE_TAG
readonly SOURCE_SUCCESS_SHA=79aba79e24026bc4c1d17aed2ca92055530b8a551ed6e1279a25300d2cb0f52b
readonly ASSOCIATION_MANIFEST_SHA=574f3553e6106a997e780b6b2a321bce86ad358b19c38989e84e2a4914b73141
readonly ASSOCIATION_FILE_SHA=be643e339cd7e1fe3a31470750cf8e1b447d95a471e0a1600b2dd2544879d7f9
readonly INTERNAL_CONTRACT_SHA=9bdab5023b5775b787e15c3c76d542eab921bd7a704602b4502b2305fad03d4c
readonly INTERNAL_TENSOR_SHA=c2fdeccfdcc81363fe01a566c34bf6c04f2f44b0a18e7b545e76fdf0f0d4560b

[[ ${GLM_GREENFIELD_PP16_LP2_EXACT_QUERY:-0} == 1 ]] || {
  echo "PP16 LP2 exact-query discriminator is default-off" >&2
  exit 2
}
[[ ${GLM_GREENFIELD_PP16_LP2_EXACT_QUERY_MODE:-off} == bounded ]] || {
  echo "set GLM_GREENFIELD_PP16_LP2_EXACT_QUERY_MODE=bounded" >&2
  exit 2
}

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_PP16_LP2_EXACT_QUERY_TAG:-greenfield_pp16_lp2_exact_query_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
REMOTE_PREFIX=$APPROVED_BUCKET/results/$TAG
readonly PIN TAG RUN_DIR REMOTE_PREFIX

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]]
[[ -z $(git -C "$WORKTREE" status --porcelain) ]]
[[ $(git -C "$WORKTREE" ls-remote origin "refs/heads/$BRANCH" | awk '{print $1}') == "$PIN" ]]
[[ $(gcloud storage buckets describe "$APPROVED_BUCKET" --format='value(location)') == "$APPROVED_LOCATION" ]]
[[ -r $RESULTS_DB && ! -e $RUN_DIR ]]
mkdir -p "$RUN_DIR/hlo"

say() {
  echo "[pp16-lp2-query $(date -u +%H:%M:%S)] $*" |
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
  command='tools_ok=1; command -v pgrep >/dev/null 2>&1 || tools_ok=0; command -v fuser >/dev/null 2>&1 || tools_ok=0; sudo -n true >/dev/null 2>&1 || tools_ok=0; ray_pids=$('"$ray_enum"' 2>/dev/null); ray_rc=$?; generic=$(pgrep -af "VLLM::[E]ngineCore|[R]ayWorkerWrapper|[p]robe_pp16_lp2_exact_query[.]py|[c]ompile_short_decoder[.]py|[r]un_real_one_layer[.]py|[m]icrobench" 2>/dev/null || true); holders=$(sudo -n fuser /tmp/libtpu_lockfile 2>/dev/null || true); containers=$(sudo -n docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Command}}" 2>/dev/null); docker_rc=$?; if [ "$tools_ok" -ne 1 ] || [ "$ray_rc" -ne 0 ] || [ "$docker_rc" -ne 0 ]; then echo "CENSUS_BAD $(hostname): census tool failed"; elif [ -n "$ray_pids" ] || [ -n "$generic" ] || [ -n "$holders" ] || echo "$containers" | grep -Eqi "[v]llm|[g]emma|[q]wen|[r]erank|[a]spt"; then echo "CENSUS_BUSY $(hostname)"; [ -n "$ray_pids" ] && echo "ray_stop_pids: $ray_pids"; [ -n "$generic" ] && echo "$generic"; [ -n "$holders" ] && echo "libtpu holders: $holders"; else echo "CENSUS_OK $(hostname)"; fi'
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
    say "FAILED status=$status; compact diagnostic evidence retained"
    gcloud storage cp --recursive --no-clobber "$RUN_DIR" \
      "$REMOTE_PREFIX/diagnostic/" >/dev/null 2>&1 || true
  fi
}

exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  say "ABORT: another protected TPU workflow holds the global lease"
  exit 1
}
exec 8>/home/gianl/.glm-tpu-rsync.lock
flock 8
trap on_exit EXIT

say "RUN_DIR=$RUN_DIR PIN=$PIN source_db=554 mesh=LP2 warmup=1 iterations=3"
if gcloud storage ls "$REMOTE_PREFIX/**" >"$RUN_DIR/remote_vacancy.txt" 2>&1; then
  say "ABORT: append-only remote prefix already exists"
  exit 2
fi
echo "VACANT $REMOTE_PREFIX" >"$RUN_DIR/remote_vacancy.txt"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$SOURCE_ROOT" "$SOURCE_REMOTE" "$SOURCE_SUCCESS_SHA" \
  "$ASSOCIATION_MANIFEST_SHA" "$ASSOCIATION_FILE_SHA" \
  "$INTERNAL_CONTRACT_SHA" "$INTERNAL_TENSOR_SHA" \
  "$RUN_DIR/source_identity.json" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

root=Path(sys.argv[1]); remote=sys.argv[2]; success_sha=sys.argv[3]
manifest_sha,file_sha,contract_sha,tensor_sha=sys.argv[4:8]; output=Path(sys.argv[8])
paths={
    'association_file': (root/'association/layer0_dsa_input.safetensors',file_sha),
    'internal_contract': (root/'internal/contract.json',contract_sha),
    'internal_tensor': (root/'internal/position_8155_internals.npz',tensor_sha),
}
for name,(path,expected) in paths.items():
    if not path.is_file() or sha256(path.read_bytes()).hexdigest()!=expected:
        raise SystemExit(f'{name} local hash drifted')
manifest=json.loads((root/'association/manifest.json').read_text())
if manifest.get('manifest_sha256')!=manifest_sha:
    raise SystemExit('association manifest identity drifted')
success=subprocess.check_output(['gcloud','storage','cat',remote+'/SUCCESS'])
if sha256(success).hexdigest()!=success_sha:
    raise SystemExit('DB554 remote SUCCESS drifted')
record={
    'source_tag': remote.rsplit('/',1)[-1],
    'source_remote': remote,
    'source_db_run_id': 554,
    'source_success_sha256': success_sha,
    'association_manifest_sha256': manifest_sha,
    'association_file_sha256': file_sha,
    'internal_contract_sha256': contract_sha,
    'internal_tensor_sha256': tensor_sha,
}
output.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
PY

strict_census pre || {
  say "ABORT: pre-run census is not authenticated 8/8 zero work"
  exit 1
}

say "synchronizing the exact pushed pin on all eight hosts"
# shellcheck disable=SC2016
sync_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; pin='"$PIN"'; branch='"$BRANCH"'; origin='"$ORIGIN"'; wt='"$WORKTREE"'; if [[ $idx == 0 ]]; then [[ -e "$wt/.git" && $(git -C "$wt" rev-parse HEAD) == "$pin" && -z $(git -C "$wt" status --porcelain) ]]; else [[ -e "$wt/.git" && -z $(git -C "$wt" status --porcelain) ]]; git -C "$wt" fetch -q origin "$branch"; git -C "$wt" checkout -q --detach "$pin"; fi; [[ $(git -C "$wt" rev-parse HEAD) == "$pin" && -z $(git -C "$wt" status --porcelain) ]]; echo "SYNC_OK $(hostname) $pin"'
gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$sync_command" >"$RUN_DIR/sync.txt" 2>&1
has_eight_unique_markers "$RUN_DIR/sync.txt" SYNC_OK || {
  say "ABORT: exact eight-host sync failed"
  exit 1
}

say "running real layer-0 exact query on adjacent TPU devices 0/1"
started=$(date +%s)
(
  cd "$WORKTREE"
  JAX_PLATFORMS=tpu \
    TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_VISIBLE_DEVICES=0,1,2,3 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=.95 \
    PYTHONPATH="$WORKTREE" \
    timeout --signal=TERM --kill-after=30 600 \
    /home/gianl/vllm-env/bin/python -u \
      scripts/greenfield/probe_pp16_lp2_exact_query.py \
      --expected-code-hash "$PIN" \
      --association-input-dir "$SOURCE_ROOT/association" \
      --association-input-manifest-sha256 "$ASSOCIATION_MANIFEST_SHA" \
      --association-input-file-sha256 "$ASSOCIATION_FILE_SHA" \
      --internal-observer-dir "$SOURCE_ROOT/internal" \
      --internal-contract-sha256 "$INTERNAL_CONTRACT_SHA" \
      --internal-tensor-sha256 "$INTERNAL_TENSOR_SHA" \
      --warmup 1 --iterations 3 \
      --output "$RUN_DIR/runner.json" --hlo-dir "$RUN_DIR/hlo"
) >"$RUN_DIR/runner.log" 2>&1
elapsed=$(( $(date +%s) - started ))
say "runner completed in ${elapsed}s"

strict_census post || {
  say "ABORT: post-run census is not authenticated 8/8 zero work"
  exit 1
}
post_census_done=1

say "validating arithmetic/HLO and publishing one diagnostic DB row"
PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR" "$PIN" "$RESULTS_DB" "$WORKTREE" "$elapsed" <<'PY'
from __future__ import annotations
import json
from pathlib import Path
import sqlite3
import sys

run_dir,pin,db_path,repo,elapsed=sys.argv[1:]
run_dir=Path(run_dir); runner=json.loads((run_dir/'runner.json').read_text())
if runner.get('status')!='SUCCESS' or runner.get('code_hash')!=pin:
    raise SystemExit('runner identity failed')
if not runner['query_comparison']['elementwise_exact'] or not runner['head_comparison']['elementwise_exact']:
    raise SystemExit('real query/head is not bitwise exact')
if not runner['hlo']['materializer']['contract']['passed'] or not runner['hlo']['query_head']['contract']['passed'] or not runner['hlo']['query_head']['stablehlo_contract']['passed']:
    raise SystemExit('materializer/query HLO contract failed')
query_contract=runner['hlo']['query_head']['contract']
if query_contract['exact_chunks_per_local_owner']!=2 or query_contract['tuple4_reduction_fusion_count']!=1 or query_contract['expected_runtime_tuple4_reduction_count']!=2:
    raise SystemExit('LP2 two-chunk tuple4 contract drifted')
if runner['physical_group']['device_ids']!=[0,1] or runner['physical_group']['coordinates']!=[[0,0,0],[1,0,0]]:
    raise SystemExit('physical LP2 group drifted')

sys.path.insert(0,str(Path(repo)/'bench'))
import provenance as pv
conn=pv.connect(db_path)
run_id=pv.start_run(
    conn,
    model='zai-org/GLM-5.2-FP8:greenfield-pp16-lp2-exact-query',
    revision='db554-layer0-tuple4-chunk2-v1',
    env={
        'GLM_ENGINE':'greenfield_pp16_lp2_exact_query',
        'greenfield_code_hash':pin,
        'source_db_run_id':554,
        'source':runner['source'],
        'physical_group':runner['physical_group'],
        'query_hlo_sha256':runner['hlo']['query_head']['optimized_hlo']['sha256'],
        'materializer_hlo_sha256':runner['hlo']['materializer']['optimized_hlo']['sha256'],
    },
    note='Protected bounded real layer-0 PP16 LP2 exact-query discriminator.',
    harness_repo=repo,
    fork_repo=None,
)
pv.record_item(
    conn,run_id,
    benchmark='greenfield_pp16_lp2_exact_query',
    item_id='layer0_position8155_lp2_tuple4_chunk2',
    prompt='Accepted DB554 layer-0 q-a/normalized state and real FP8 WQ_B/head weights.',
    gold='Bitwise accepted query/head with two local 1,024-row tuple4 groups.',
    raw_output=json.dumps(runner,sort_keys=True),
    extracted=runner['query_comparison']['actual_sha256'],
    correct=True,score=1.0,latency_ms=runner['timing']['p50_ms'],
)
pv.finalize(
    conn,run_id,
    benchmark='greenfield_pp16_lp2_exact_query',
    metric='bitwise_query_head_exact',value=1.0,
    note='Diagnostic only; no decoder, Gate-D, latency, or token-rate claim.',
)
conn.close()
summary={
    'status':'SUCCESS','code_hash':pin,'elapsed_seconds':int(elapsed),
    'results_db_run_id':run_id,'runner':runner,
    'claim_scope':'bounded real layer-0 arithmetic/HLO diagnostic only',
    'performance_claim':False,'gate_d_passed':False,
}
(run_dir/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
source=sqlite3.connect(db_path); snapshot=sqlite3.connect(run_dir/'results_ckpt.db')
source.backup(snapshot); snapshot.close(); source.close()
if sqlite3.connect(run_dir/'results_ckpt.db').execute('PRAGMA integrity_check').fetchone()[0]!='ok':
    raise SystemExit('results DB snapshot integrity failed')
print(f'PP16_LP2_QUERY_VALID db_run={run_id}')
PY

cp "$RUN_DIR/orchestrator.log" "$RUN_DIR/orchestrator.sealed.log"
(
  cd "$RUN_DIR"
  find hlo -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum runner.json runner.log source_identity.json summary.json \
    results_ckpt.db census_pre.txt census_post.txt sync.txt \
    remote_vacancy.txt orchestrator.sealed.log
) >"$RUN_DIR/evidence.sha256"
(cd "$RUN_DIR" && sha256sum -c evidence.sha256 >/dev/null)

gcloud storage cp --recursive --no-clobber "$RUN_DIR/hlo" "$REMOTE_PREFIX/" >/dev/null
gcloud storage cp --no-clobber \
  "$RUN_DIR/runner.json" "$RUN_DIR/runner.log" \
  "$RUN_DIR/source_identity.json" "$RUN_DIR/summary.json" \
  "$RUN_DIR/results_ckpt.db" "$RUN_DIR/evidence.sha256" \
  "$RUN_DIR/census_pre.txt" "$RUN_DIR/census_post.txt" \
  "$RUN_DIR/sync.txt" "$RUN_DIR/remote_vacancy.txt" \
  "$RUN_DIR/orchestrator.sealed.log" "$REMOTE_PREFIX/" >/dev/null

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$RUN_DIR/summary.json" "$RUN_DIR/SUCCESS" "$TAG" "$REMOTE_PREFIX" <<'PY'
import json
from hashlib import sha256
from pathlib import Path
import sys
summary_path=Path(sys.argv[1]); output=Path(sys.argv[2]); tag=sys.argv[3]; remote=sys.argv[4]
summary=json.loads(summary_path.read_text())
record={
    'artifact_kind':'greenfield_pp16_lp2_exact_query_success',
    'code_hash':summary['code_hash'],
    'results_db_run_id':summary['results_db_run_id'],
    'run_tag':tag,
    'remote_prefix':remote,
    'summary_sha256':sha256(summary_path.read_bytes()).hexdigest(),
    'status':'SUCCESS',
}
raw=json.dumps(record,allow_nan=False,separators=(',',':'),sort_keys=True).encode()
record['success_sha256']=sha256(raw).hexdigest()
output.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
PY
gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" \
  "$REMOTE_PREFIX/SUCCESS" >/dev/null
local_success_sha=$(sha256sum "$RUN_DIR/SUCCESS" | awk '{print $1}')
remote_success_sha=$(gcloud storage cat "$REMOTE_PREFIX/SUCCESS" | sha256sum | awk '{print $1}')
[[ $local_success_sha == "$remote_success_sha" ]]

db_run=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["results_db_run_id"])' \
  "$RUN_DIR/summary.json")
say "SUCCESS DB=$db_run archive=$REMOTE_PREFIX"
trap - EXIT
