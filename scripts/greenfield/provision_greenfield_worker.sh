#!/usr/bin/env bash
# Idempotent environment-only provisioning for a replacement greenfield TPU VM.
# This script does not initialize JAX/TPU, launch a model, or install legacy execution code.
set -euo pipefail

[[ ${GLM_GREENFIELD_PROVISION:-0} == 1 ]] || {
  echo "greenfield provisioning is default-off; set GLM_GREENFIELD_PROVISION=1" >&2
  exit 2
}
[[ $# -eq 1 && $1 =~ ^[0-9a-f]{40}$ ]] || {
  echo "usage: provision_greenfield_worker.sh <expected-commit>" >&2
  exit 2
}

readonly EXPECTED_PIN=$1
readonly BRANCH=rewrite/topology-first-decode
readonly ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly APPROVED_BUCKET=gs://driftbench-dsv4-uc
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly VENV=/home/gianl/vllm-env
readonly MODEL_MOUNT=/home/gianl/gcs-models
readonly UV=/home/gianl/.local/bin/uv
readonly GCSFUSE_URI=$APPROVED_BUCKET/artifacts/greenfield/gcsfuse-3.11.2-linux-amd64
readonly GCSFUSE_SHA=298bc02d8a6fd6948bf93aa69aee0ff74cf07339e1c018eeabb5d80db93a2225
WORKER_INDEX=${HOSTNAME##*-w-}
readonly WORKER_INDEX

say() { printf 'GREENFIELD_PROVISION worker=%s %s\n' "$WORKER_INDEX" "$*"; }

say "ssh identity"
mkdir -p /home/gianl/.ssh
if [[ ! -f /home/gianl/.ssh/id_ed25519 ]]; then
  gcloud storage cp "$APPROVED_BUCKET/artifacts/secrets/id_ed25519" \
    /home/gianl/.ssh/id_ed25519 >/dev/null
  gcloud storage cp "$APPROVED_BUCKET/artifacts/secrets/id_ed25519.pub" \
    /home/gianl/.ssh/id_ed25519.pub >/dev/null
fi
chmod 600 /home/gianl/.ssh/id_ed25519
touch /home/gianl/.ssh/known_hosts
if ! ssh-keygen -F github.com -f /home/gianl/.ssh/known_hosts >/dev/null; then
  ssh-keyscan github.com >>/home/gianl/.ssh/known_hosts 2>/dev/null
fi
export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=yes"

say "python environment"
if [[ ! -x $UV ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh \
    >/tmp/greenfield_uv_install.log 2>&1
fi
"$UV" python install 3.12.13 >/tmp/greenfield_uv_python.log 2>&1
uv_base=/home/gianl/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu
if [[ ! -e $uv_base/bin/python3.12 ]]; then
  uv_real=$(find /home/gianl/.local/share/uv/python -maxdepth 1 -mindepth 1 \
    -type d -name 'cpython-3.12*-linux-x86_64-gnu' | sort -V | tail -1)
  [[ -n $uv_real ]]
  ln -sfn "$uv_real" "$uv_base"
fi
if [[ ! -x $VENV/bin/python ]]; then
  [[ ! -e $VENV ]] || {
    echo "refusing to replace an incomplete existing venv" >&2
    exit 1
  }
  provision_tmp=$(mktemp -d /tmp/greenfield-provision.XXXXXXXX)
  cleanup_tmp() {
    if [[ -d ${provision_tmp:-} && $provision_tmp == /tmp/greenfield-provision.* ]]; then
      rm -r -- "$provision_tmp"
    fi
  }
  trap cleanup_tmp EXIT
  gcloud storage cp "$APPROVED_BUCKET/artifacts/vllm-env.tar.gz" \
    "$provision_tmp/vllm-env.tar.gz" >/dev/null
  tar xzf "$provision_tmp/vllm-env.tar.gz" -C /home/gianl
  [[ -x $VENV/bin/python ]]
  cleanup_tmp
  trap - EXIT
fi
"$VENV/bin/python" -c \
  'import google.cloud.storage, jax, numpy, safetensors; print(jax.__version__)' \
  >/tmp/greenfield_python_version.txt

say "exact repository pin"
if [[ -e $WORKTREE ]]; then
  git -C "$WORKTREE" rev-parse --is-inside-work-tree >/dev/null
  [[ -z $(git -C "$WORKTREE" status --porcelain) ]]
  git -C "$WORKTREE" fetch -q origin "$BRANCH"
else
  git clone -q --filter=blob:none --no-checkout --single-branch \
    --branch "$BRANCH" "$ORIGIN" "$WORKTREE"
fi
git -C "$WORKTREE" checkout -q --detach "$EXPECTED_PIN"
[[ $(git -C "$WORKTREE" rev-parse HEAD) == "$EXPECTED_PIN" ]]
[[ -z $(git -C "$WORKTREE" status --porcelain) ]]

say "read-only approved-bucket mount"
if ! command -v gcsfuse >/dev/null; then
  gcsfuse_tmp=$(mktemp /tmp/greenfield-gcsfuse.XXXXXXXX)
  cleanup_gcsfuse() {
    if [[ -f ${gcsfuse_tmp:-} && $gcsfuse_tmp == /tmp/greenfield-gcsfuse.* ]]; then
      rm -- "$gcsfuse_tmp"
    fi
  }
  trap cleanup_gcsfuse EXIT
  gcloud storage cp "$GCSFUSE_URI" "$gcsfuse_tmp" >/dev/null
  [[ $(sha256sum "$gcsfuse_tmp" | awk '{print $1}') == "$GCSFUSE_SHA" ]]
  sudo -n install -m 0755 "$gcsfuse_tmp" /usr/local/bin/gcsfuse
  cleanup_gcsfuse
  trap - EXIT
fi
[[ $(sha256sum "$(command -v gcsfuse)" | awk '{print $1}') == "$GCSFUSE_SHA" ]]
mkdir -p "$MODEL_MOUNT"
if ! findmnt -T "$MODEL_MOUNT" -n -o SOURCE,FSTYPE \
  | grep -q 'driftbench-dsv4-uc fuse.gcsfuse'; then
  mountpoint -q "$MODEL_MOUNT" && {
    echo "refusing to replace an unexpected model mount" >&2
    exit 1
  }
  gcsfuse --implicit-dirs -o ro --stat-cache-ttl 1h --type-cache-ttl 1h \
    driftbench-dsv4-uc "$MODEL_MOUNT" >/tmp/greenfield_gcsfuse.log 2>&1
fi
findmnt -T "$MODEL_MOUNT" -n -o SOURCE,FSTYPE \
  | grep -q 'driftbench-dsv4-uc fuse.gcsfuse'

say "PROVISION_OK pin=$EXPECTED_PIN"
