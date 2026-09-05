#!/usr/bin/env bash
# Seal one pinned accepted legacy long-context result (spec §23.1) as a token
# oracle: the four 128K passkey depths of DB run 403 or the 256K E0 prompt of
# DB run 402, rebuilt from provenance (seed/generator) and byte-verified against
# the digests the legacy harness stored.
set -euo pipefail

readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly TOKENIZER_ROOT=/home/gianl/gcs-models/models/GLM-5.2-FP8
readonly LEGACY_REPO=/home/gianl/tpu-inference
readonly LEGACY_PIN=b3c25df47ac98783912dc658878181ec0a8ae16d
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc

readonly LEGACY_BENCH_ROOT=/home/gianl/glm-tpu/bench
PROFILE=${GLM_GREENFIELD_LONG_CONTEXT_ORACLE_PROFILE:?set 128k_d0.0|128k_d0.05|128k_d0.95|128k_d1.0|256k_e0}
KIND=passkey
SOURCE_RUN_ID=403
SOURCE_HARNESS_GIT=a4a17ac
SOURCE_FORK_GIT=b3c25df47
SOURCE_PROMPT_TOKENS=127363
SOURCE_GENERATED_TOKENS=20
SOURCE_CONTEXT_LENGTH=128000
case "$PROFILE" in
  128k_d0.0)  SOURCE_ITEM_ROW_ID=1517; SOURCE_DEPTH=0.0;  SOURCE_SEED=16780345; SOURCE_GOLD=705269 ;;
  128k_d0.05) SOURCE_ITEM_ROW_ID=1518; SOURCE_DEPTH=0.05; SOURCE_SEED=16781195; SOURCE_GOLD=824794 ;;
  128k_d0.95) SOURCE_ITEM_ROW_ID=1519; SOURCE_DEPTH=0.95; SOURCE_SEED=16796495; SOURCE_GOLD=289958 ;;
  128k_d1.0)  SOURCE_ITEM_ROW_ID=1520; SOURCE_DEPTH=1.0;  SOURCE_SEED=16797345; SOURCE_GOLD=891482 ;;
  256k_e0)
    KIND=e0
    SOURCE_RUN_ID=402
    SOURCE_ITEM_ROW_ID=1516
    SOURCE_PROMPT_TOKENS=262144
    SOURCE_GENERATED_TOKENS=256
    SOURCE_CONTEXT_LENGTH=262144
    SOURCE_SEED=34353209
    SOURCE_DEPTH=
    SOURCE_GOLD=
    ;;
  *)
    echo "unsupported long-context oracle profile: $PROFILE" >&2
    exit 2
    ;;
esac
if [[ $KIND == passkey ]]; then
  SOURCE_BENCHMARK=passkey_L${SOURCE_CONTEXT_LENGTH}_d${SOURCE_DEPTH}
  KIND_CLI=(--expected-depth "$SOURCE_DEPTH" --expected-gold "$SOURCE_GOLD")
else
  SOURCE_BENCHMARK=dsa_throughput_ctx${SOURCE_CONTEXT_LENGTH}
  KIND_CLI=()
fi
TAG_PREFIX=greenfield_long_context_oracle_${PROFILE}
readonly PROFILE KIND SOURCE_RUN_ID SOURCE_ITEM_ROW_ID SOURCE_HARNESS_GIT
readonly SOURCE_FORK_GIT SOURCE_BENCHMARK SOURCE_PROMPT_TOKENS SOURCE_CONTEXT_LENGTH
readonly SOURCE_GENERATED_TOKENS SOURCE_SEED SOURCE_GOLD SOURCE_DEPTH TAG_PREFIX

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_LONG_CONTEXT_ORACLE_TAG:-${TAG_PREFIX}_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
ORACLE_DIR=$RUN_DIR/oracle
REMOTE_PREFIX=$APPROVED_BUCKET/oracles/greenfield/glm52/long_context/$PROFILE/$TAG

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing long-context oracle outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing long-context oracle from a dirty worktree" >&2
  exit 2
}
[[ $(git -C "$LEGACY_REPO" rev-parse HEAD) == "$LEGACY_PIN" ]] || {
  echo "legacy repository pin changed" >&2
  exit 2
}
[[ -r $RESULTS_DB ]] || {
  echo "accepted provenance database is unavailable" >&2
  exit 2
}
for filename in tokenizer.json tokenizer_config.json chat_template.jinja; do
  [[ -r $TOKENIZER_ROOT/$filename ]] || {
    echo "tokenizer file is unavailable: $filename" >&2
    exit 2
  }
done
[[ ! -e $RUN_DIR ]] || {
  echo "append-only run directory exists: $RUN_DIR" >&2
  exit 2
}

mkdir -p "$RUN_DIR"
exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
flock -n 9 || {
  echo "another protected pod workflow holds the global lease" >&2
  exit 1
}

say() {
  echo "[long-context-oracle $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
}

say "RUN_DIR=$RUN_DIR PIN=$PIN LEGACY_PIN=$LEGACY_PIN"
say "PROFILE=$PROFILE SOURCE=results.db run=$SOURCE_RUN_ID item_row=$SOURCE_ITEM_ROW_ID REMOTE_PREFIX=$REMOTE_PREFIX"
git -C "$WORKTREE" status --short --branch >"$RUN_DIR/greenfield_status.txt"
git -C "$LEGACY_REPO" status --short --branch >"$RUN_DIR/legacy_status.txt"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
  "$WORKTREE/scripts/greenfield/capture_long_context_oracle.py" \
  --kind "$KIND" \
  --legacy-bench-root "$LEGACY_BENCH_ROOT" \
  --expected-context-length "$SOURCE_CONTEXT_LENGTH" \
  "${KIND_CLI[@]}" \
  --results-db "$RESULTS_DB" \
  --tokenizer-root "$TOKENIZER_ROOT" \
  --output "$ORACLE_DIR" \
  --expected-code-hash "$PIN" \
  --legacy-repository-pin "$LEGACY_PIN" \
  --run-id "$SOURCE_RUN_ID" \
  --item-row-id "$SOURCE_ITEM_ROW_ID" \
  --expected-harness-git "$SOURCE_HARNESS_GIT" \
  --expected-fork-git "$SOURCE_FORK_GIT" \
  --expected-benchmark "$SOURCE_BENCHMARK" \
  --expected-model-uri gs://driftbench-dsv4-uc/models/GLM-5.2-FP8 \
  --expected-prompt-tokens "$SOURCE_PROMPT_TOKENS" \
  --expected-generated-tokens "$SOURCE_GENERATED_TOKENS" \
  --expected-seed "$SOURCE_SEED" >"$RUN_DIR/capture.json"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$ORACLE_DIR" <<'PY' >"$RUN_DIR/inspection.json"
import json
from pathlib import Path
import sys

from glm_tpu.greenfield.validation import inspect_long_context_oracle

manifest = inspect_long_context_oracle(Path(sys.argv[1]))
print(json.dumps({
    "artifact_kind": manifest["artifact_kind"],
    "generated_token_count": manifest["generated_token_count"],
    "generated_token_ids_sha256": manifest["generated_token_ids_sha256"],
    "manifest_sha256": manifest["manifest_sha256"],
    "prompt_token_count": manifest["prompt_token_count"],
    "prompt_token_ids_sha256": manifest["prompt_token_ids_sha256"],
    "source": manifest["source"],
}, indent=2, sort_keys=True))
PY

say "uploading append-only oracle evidence"

/home/gianl/vllm-env/bin/python - "$RUN_DIR" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
records = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.name in {"SUCCESS", "evidence_sha256.json"}:
        continue
    digest = sha256(path.read_bytes()).hexdigest()
    records.append({
        "byte_count": path.stat().st_size,
        "path": str(path.relative_to(root)),
        "sha256": digest,
    })
(root / "evidence_sha256.json").write_text(
    json.dumps({"files": records}, indent=2, sort_keys=True) + "\n"
)
PY

gcloud storage cp --recursive --no-clobber "$RUN_DIR"/* \
  "$REMOTE_PREFIX/" >/dev/null

/home/gianl/vllm-env/bin/python - "$RUN_DIR" "$REMOTE_PREFIX" <<'PY' \
  >"$RUN_DIR/remote_objects.json"
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
prefix = sys.argv[2]
records = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.name in {"SUCCESS", "remote_objects.json"}:
        continue
    relative = path.relative_to(root).as_posix()
    completed = subprocess.run(
        [
            "gcloud", "storage", "objects", "describe",
            f"{prefix}/{relative}", "--format=json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    remote = json.loads(completed.stdout)
    if int(remote["size"]) != path.stat().st_size:
        raise SystemExit(f"remote size mismatch: {relative}")
    crc32c = remote.get("crc32c_hash") or remote.get("crc32c")
    if not crc32c:
        raise SystemExit(f"remote CRC32C is missing: {relative}")
    records.append({
        "crc32c": crc32c,
        "generation": remote["generation"],
        "path": relative,
        "size": int(remote["size"]),
    })
print(json.dumps({"objects": records}, indent=2, sort_keys=True))
PY
gcloud storage cp --no-clobber "$RUN_DIR/remote_objects.json" \
  "$REMOTE_PREFIX/remote_objects.json" >/dev/null

manifest_sha=$(/home/gianl/vllm-env/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' \
  "$ORACLE_DIR/manifest.json")
evidence_sha=$(sha256sum "$RUN_DIR/evidence_sha256.json" | awk '{print $1}')
remote_objects_sha=$(sha256sum "$RUN_DIR/remote_objects.json" | awk '{print $1}')
cat >"$RUN_DIR/SUCCESS" <<EOF
artifact_kind=greenfield_long_context_legacy_oracle
code_hash=$PIN
legacy_repository_pin=$LEGACY_PIN
manifest_sha256=$manifest_sha
evidence_sha256=$evidence_sha
remote_objects_sha256=$remote_objects_sha
prompt_token_count=$SOURCE_PROMPT_TOKENS
generated_token_count=$SOURCE_GENERATED_TOKENS
kind=$KIND
context_length=$SOURCE_CONTEXT_LENGTH
remote_prefix=$REMOTE_PREFIX
EOF
gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS" \
  "$REMOTE_PREFIX/SUCCESS" >/dev/null
local_success_sha=$(sha256sum "$RUN_DIR/SUCCESS" | awk '{print $1}')
remote_success_sha=$(gcloud storage cat "$REMOTE_PREFIX/SUCCESS" | sha256sum | awk '{print $1}')
[[ $local_success_sha == "$remote_success_sha" ]] || {
  echo "remote SUCCESS checksum mismatch" >&2
  exit 2
}
echo "[long-context-oracle $(date -u +%H:%M:%S)] SUCCESS manifest=$manifest_sha prompt=$SOURCE_PROMPT_TOKENS generated=$SOURCE_GENERATED_TOKENS"
