#!/bin/bash
# Provision one FRESH db-v4-64-od pod worker for GLM-5.2 serving.
# Written 2026-07-16 after the pod recreation wiped all 8 host disks (the
# "lost VM" incident): every worker needs the venv + fork + vLLM@LKG back
# before any Ray launch. Mirrors the DSV4 provision_worker.sh, adapted:
#   - fork branch glm-5.2-v4-next (the integrated staging branch the pod runs)
#   - vLLM@LKG pin read from the GLM branch itself (.buildkite/vllm_lkg.version)
#   - no moe-tpu dependency (build steps inlined)
#   - all bulk artifacts pulled from the SAME-REGION us-central2 bucket
#     (gs://driftbench-dsv4-uc) per the CLAUDE.md cost rules.
# Idempotent: a fully-provisioned worker exits early.
set -u
N=$(hostname | sed 's/.*-w-//')
PY=$HOME/vllm-env/bin/python
say() { echo "W$N| $*"; }
ok_full() { [ -d "$HOME/tpu-inference/.git" ] && "$PY" -c "import tpu_inference, vllm" 2>/dev/null; }

if ok_full; then
  say "ALREADY OK @ $(git -C "$HOME/tpu-inference" rev-parse --short HEAD 2>/dev/null)"
  exit 0
fi

say "[0/4] ssh keys + uv..."
mkdir -p ~/.ssh
[ -f ~/.ssh/id_ed25519 ] || {
  gcloud storage cp gs://driftbench-dsv4-uc/artifacts/secrets/id_ed25519 ~/.ssh/id_ed25519 >/tmp/prov0.log 2>&1
  gcloud storage cp gs://driftbench-dsv4-uc/artifacts/secrets/id_ed25519.pub ~/.ssh/id_ed25519.pub >>/tmp/prov0.log 2>&1
  chmod 600 ~/.ssh/id_ed25519
}
ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null
export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new"
command -v "$HOME/.local/bin/uv" >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >>/tmp/prov0.log 2>&1
UV="$HOME/.local/bin/uv"

say "[1/4] venv tarball (same-region)..."
if [ ! -d "$HOME/vllm-env" ]; then
  gcloud storage cp gs://driftbench-dsv4-uc/artifacts/vllm-env.tar.gz "$HOME/vllm-env.tar.gz" >/tmp/prov1.log 2>&1
  tar xzf "$HOME/vllm-env.tar.gz" -C "$HOME/" >>/tmp/prov1.log 2>&1
  rm -f "$HOME/vllm-env.tar.gz"
fi
"$UV" python install 3.12.13 >>/tmp/prov1.log 2>&1 || true
# The venv hard-codes a non-patch-versioned base-interpreter dir; bridge it.
VENV_PY_DIR="$HOME/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu"
if [ ! -e "$VENV_PY_DIR/bin/python3.12" ]; then
  REAL_PY=$(ls -d "$HOME"/.local/share/uv/python/cpython-3.12*-linux-x86_64-gnu 2>/dev/null | sort -V | tail -1)
  [ -n "$REAL_PY" ] && ln -sfn "$REAL_PY" "$VENV_PY_DIR"
fi

say "[2/4] fork glm-5.2-v4-next + editable..."
[ -d "$HOME/tpu-inference/.git" ] || git clone --branch glm-5.2-v4-next git@github.com:GianluigiVitale/tpu-inference.git "$HOME/tpu-inference" >/tmp/prov2.log 2>&1
git -C "$HOME/tpu-inference" fetch origin glm-5.2-v4-next >>/tmp/prov2.log 2>&1
git -C "$HOME/tpu-inference" checkout glm-5.2-v4-next >>/tmp/prov2.log 2>&1
git -C "$HOME/tpu-inference" pull --ff-only origin glm-5.2-v4-next >>/tmp/prov2.log 2>&1
# Clear any stale site-packages shadow before the editable install.
rm -rf "$HOME"/vllm-env/lib/python3.12/site-packages/tpu_inference "$HOME"/vllm-env/lib/python3.12/site-packages/tpu_inference-*.dist-info
VIRTUAL_ENV="$HOME/vllm-env" "$UV" pip install -e "$HOME/tpu-inference" --no-deps --no-build-isolation --python "$PY" >/tmp/prov2b.log 2>&1

say "[3/4] vLLM@LKG into ~/vllm-build..."
LKG=$(git -C "$HOME/tpu-inference" show glm-5.2-v4-next:.buildkite/vllm_lkg.version | tr -d '[:space:]')
[ -d "$HOME/vllm-build/.git" ] || git clone --filter=blob:none https://github.com/vllm-project/vllm.git "$HOME/vllm-build" >/tmp/prov3.log 2>&1
git -C "$HOME/vllm-build" fetch --depth 1 origin "$LKG" >>/tmp/prov3.log 2>&1
git -C "$HOME/vllm-build" checkout "$LKG" >>/tmp/prov3.log 2>&1
"$UV" pip install --python "$PY" cmake ninja wheel "setuptools>=77,<81" "setuptools-scm>=8" "setuptools-rust>=1.9.0" "packaging>=24.2" >>/tmp/prov3.log 2>&1
"$UV" pip uninstall --python "$PY" vllm-tpu vllm >>/tmp/prov3.log 2>&1 || true
VLLM_TARGET_DEVICE=tpu "$UV" pip install --python "$PY" --no-build-isolation --no-deps -e "$HOME/vllm-build" >/tmp/prov3b.log 2>&1

say "[4/4] verify..."
if ok_full; then
  say "PROVISIONED OK @ $(git -C "$HOME/tpu-inference" rev-parse --short HEAD) (jax=$("$PY" -c 'import jax;print(jax.__version__)' 2>/dev/null) libtpu=$("$PY" -c 'import importlib.metadata as m;print(m.version("libtpu"))' 2>/dev/null))"
else
  say "VERIFY FAILED — fork=$(tail -1 /tmp/prov2b.log 2>/dev/null) vllm=$(tail -2 /tmp/prov3b.log 2>/dev/null | tr '\n' ' ')"
  exit 1
fi
