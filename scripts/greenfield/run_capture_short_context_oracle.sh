#!/usr/bin/env bash
# Seal one pinned accepted short-context passkey result as a token oracle.
set -euo pipefail

readonly BRANCH=rewrite/topology-first-decode
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db
readonly TOKENIZER_ROOT=/home/gianl/gcs-models/models/GLM-5.2-FP8
readonly LEGACY_REPO=/home/gianl/tpu-inference
readonly LEGACY_PIN=b3c25df47ac98783912dc658878181ec0a8ae16d
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc

PROFILE=${GLM_GREENFIELD_SHORT_CONTEXT_ORACLE_PROFILE:-2k}
case "$PROFILE" in
  2k)
    SOURCE_RUN_ID=139
    SOURCE_ITEM_ROW_ID=628
    SOURCE_HARNESS_GIT=b8e891e
    SOURCE_FORK_GIT=f0c63c302
    SOURCE_BENCHMARK=passkey_L2040_d0.25
    SOURCE_PROMPT_TOKENS=2034
    SOURCE_GENERATED_TOKENS=20
    SOURCE_SEED=283835
    SOURCE_GOLD=110391
    TAG_PREFIX=greenfield_short_context_oracle
    ;;
  8k)
    # Pair the protected 8K token oracle with the current immutable legacy
    # execution pin used by the all-event DSA capture.  Historical run 104 is
    # preserved in its already-sealed artifact, but its continuation differs
    # after the passkey phrase and therefore cannot define the same recurrent
    # trajectory as the fresh run-485 DSA tensors.
    SOURCE_RUN_ID=485
    SOURCE_ITEM_ROW_ID=1769
    SOURCE_HARNESS_GIT=a4a17ac
    SOURCE_FORK_GIT=b3c25df47
    SOURCE_BENCHMARK=passkey_L8192_d0.5
    SOURCE_PROMPT_TOKENS=8155
    SOURCE_GENERATED_TOKENS=20
    SOURCE_SEED=1093997
    SOURCE_GOLD=881446
    TAG_PREFIX=greenfield_short_context_oracle_8k
    ;;
  *)
    echo "unsupported short-context oracle profile: $PROFILE" >&2
    exit 2
    ;;
esac
readonly PROFILE SOURCE_RUN_ID SOURCE_ITEM_ROW_ID SOURCE_HARNESS_GIT
readonly SOURCE_FORK_GIT SOURCE_BENCHMARK SOURCE_PROMPT_TOKENS
readonly SOURCE_GENERATED_TOKENS SOURCE_SEED SOURCE_GOLD TAG_PREFIX

PIN=$(git -C "$WORKTREE" rev-parse HEAD)
TAG=${GLM_GREENFIELD_SHORT_CONTEXT_ORACLE_TAG:-${TAG_PREFIX}_$(date -u +%Y%m%dT%H%M%S%NZ)}
RUN_DIR=/home/gianl/glm-run/$TAG
ORACLE_DIR=$RUN_DIR/oracle
REMOTE_PREFIX=$APPROVED_BUCKET/oracles/greenfield/glm52/short_context/$PROFILE/$TAG

[[ $(git -C "$WORKTREE" branch --show-current) == "$BRANCH" ]] || {
  echo "refusing short-context oracle outside $BRANCH" >&2
  exit 2
}
[[ -z $(git -C "$WORKTREE" status --porcelain) ]] || {
  echo "refusing short-context oracle from a dirty worktree" >&2
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
  echo "[short-context-oracle $(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/orchestrator.log"
}

say "RUN_DIR=$RUN_DIR PIN=$PIN LEGACY_PIN=$LEGACY_PIN"
say "PROFILE=$PROFILE SOURCE=results.db run=$SOURCE_RUN_ID item_row=$SOURCE_ITEM_ROW_ID REMOTE_PREFIX=$REMOTE_PREFIX"
git -C "$WORKTREE" status --short --branch >"$RUN_DIR/greenfield_status.txt"
git -C "$LEGACY_REPO" status --short --branch >"$RUN_DIR/legacy_status.txt"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \
  "$WORKTREE/scripts/greenfield/capture_short_context_oracle.py" \
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
  --expected-seed "$SOURCE_SEED" \
  --expected-gold "$SOURCE_GOLD" >"$RUN_DIR/capture.json"

PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \
  "$ORACLE_DIR" <<'PY' >"$RUN_DIR/inspection.json"
import json
from pathlib import Path
import sys

from glm_tpu.greenfield.validation import inspect_short_context_oracle

manifest = inspect_short_context_oracle(Path(sys.argv[1]))
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
artifact_kind=greenfield_short_context_legacy_oracle
code_hash=$PIN
legacy_repository_pin=$LEGACY_PIN
manifest_sha256=$manifest_sha
evidence_sha256=$evidence_sha
remote_objects_sha256=$remote_objects_sha
prompt_token_count=$SOURCE_PROMPT_TOKENS
generated_token_count=$SOURCE_GENERATED_TOKENS
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
echo "[short-context-oracle $(date -u +%H:%M:%S)] SUCCESS manifest=$manifest_sha prompt=$SOURCE_PROMPT_TOKENS generated=$SOURCE_GENERATED_TOKENS"
