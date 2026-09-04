#!/usr/bin/bash
# Restore the sealed Gate-D runtime and install the M2048 capsule on all hosts.
# This is an install-only prerequisite: it never imports JAX or starts TPU work.
set -euo pipefail

[[ ${GLM_GATE_D_M2048_INSTALL:-0} == 1 ]] || {
  echo "Gate-D M2048 fleet installation is default-off" >&2; exit 2;
}
[[ ${GLM_GATE_D_M2048_INSTALL_MODE:-off} == install_only ]] || {
  echo "set GLM_GATE_D_M2048_INSTALL_MODE=install_only" >&2; exit 2;
}
while IFS='=' read -r name _; do
  case "$name" in
    GLM_GATE_D_M2048_INSTALL | GLM_GATE_D_M2048_INSTALL_MODE | \
    GLM_GATE_D_M2048_INSTALL_PIN | HOME | LANG | LC_ALL | PATH | PWD | \
    PYTHONDONTWRITEBYTECODE | SHLVL | _) ;;
    *) echo "unexpected install environment name: $name" >&2; exit 2 ;;
  esac
done < <(/usr/bin/env)
[[ ${HOME:-} == /home/gianl && ${LANG:-} == C && ${LC_ALL:-} == C ]]
[[ ${PATH:-} == /snap/bin:/usr/bin:/bin && ${PYTHONDONTWRITEBYTECODE:-} == 1 ]]

readonly POD=db-v4-64-od
readonly ZONE=us-central2-b
readonly BRANCH=rewrite/topology-first-decode
readonly ORIGIN=git@github.com:GianluigiVitale/glm-tpu.git
readonly WORKTREE=/home/gianl/glm-tpu-topology-rewrite
readonly EXPECTED_PIN=${GLM_GATE_D_M2048_INSTALL_PIN:-}
readonly PROVISIONER_TARGET=/opt/glm-tpu/bin/provision_gate_d_python_runtime.py
readonly PROVISIONER_SHA=2b9c8c2b981be639ad0eb16388c6fbfdd4765adfa2ef9a1986ae4b1b37ec0594
readonly BOOTSTRAP=bootstrap_gate_d_provisioner.py
readonly BOOTSTRAP_SHA=8ca7e0c1ca878e249fc9ecc3b4c2227696ae53232af4bd2a46eb6c07a44aba1f
readonly REPO_REFRESHER=refresh_gate_d_m2048_worker_repository.py
readonly REPO_REFRESHER_SHA=f248ff6717113c2376e19e4495fd2b33314280b181b97c783996ec510b7f5b0e
readonly WORKER_REPO_PRESTATE_PIN=80bcd0edab9f4a1d7b0085af89dc4159cbf2254c
readonly PYTHON_NAME=gate-d-python-3.12.13-021044895e95
readonly PYTHON_TREE=308748a9a3c3758a6b4f233aa5c034e8cb419362dbeafe0322448be40170d616
readonly JAX_NAME=gate-d-jax-site-55233c63939e
readonly JAX_TREE=55233c63939ea28485cdf2f0fc3d9c1d2ce4d9d93aad828e94498d712a26a0df
readonly LIBTPU_NAME=gate-d-libtpu-site-db7598c867f3
readonly LIBTPU_TREE=db7598c867f370756813cbf1536ad8ef7b1d9c167975e9e1724bd9b4fee78eca
readonly INSTALL_SOURCE_NAME=gate-d-m2048-install-v1
readonly INSTALL_SOURCE_TREE=5ef7c0eec7ef58e86454166647511b0d836adfc5446409274a212fd9d6bff15a
readonly INSTALLER=install_gate_d_m2048_strategy_nd_runtime.py
readonly LAUNCHER=launch_gate_d_m2048_strategy_nd_association.py
readonly PROBE=probe_m2048_strategy_nd_association.py
readonly PUBLISHER=publish_gate_d_m2048_strategy_nd_association.py
readonly MIRROR=verify_gate_d_rewrite_same_region_git_mirror.py
readonly INSTALLER_SHA=4f87e989d6a2aef4a93609498c6929ade58adffc39cbf3592feb4a5168136221
readonly LAUNCHER_SHA=31af1b772f45a33195fa6d87a8a10cbab5187a4a226b04139cf515e0e881c0eb
readonly PROBE_SHA=1debe946e35311014e667fed863871eed4bf3afeaa9aeb27445f50b2ea233774
readonly PUBLISHER_SHA=83602a623fd6392515c63a1017c89f7ed7c06f5d0a7e1ae4e6aca519f99db9be
readonly MIRROR_SHA=764013fbb101ec79f9822da10f057e33e58cccc0d3eeafa194d5b3bca2aed8c4

git_local() {
  /usr/bin/env -i GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 \
    GIT_NO_LAZY_FETCH=1 GIT_NO_REPLACE_OBJECTS=1 GIT_OPTIONAL_LOCKS=0 \
    GIT_PROTOCOL_FROM_USER=0 GIT_SSH_COMMAND=/bin/false GIT_TERMINAL_PROMPT=0 \
    HOME=/nonexistent LANG=C LC_ALL=C PATH=/usr/bin:/bin \
    /usr/bin/git -c core.fsmonitor=false -c core.untrackedCache=false \
      -c core.hooksPath=/dev/null -c core.attributesFile=/dev/null \
      -C "$WORKTREE" "$@"
}
git_remote() {
  /usr/bin/env -i GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 \
    GIT_NO_LAZY_FETCH=1 GIT_NO_REPLACE_OBJECTS=1 GIT_OPTIONAL_LOCKS=0 \
    GIT_TERMINAL_PROMPT=0 HOME=/home/gianl LANG=C LC_ALL=C PATH=/usr/bin:/bin \
    /usr/bin/git -c core.fsmonitor=false \
      -c 'core.sshCommand=/usr/bin/ssh -oBatchMode=yes -oClearAllForwardings=yes -oForwardAgent=no' "$@"
}

readonly PIN=$(git_local rev-parse HEAD)
[[ $EXPECTED_PIN =~ ^[0-9a-f]{40}$ && $PIN == "$EXPECTED_PIN" ]]
[[ $(git_local branch --show-current) == "$BRANCH" ]]
[[ $(git_local remote get-url origin) == "$ORIGIN" ]]
[[ -z $(git_local status --porcelain=v1 --untracked-files=all) ]]
[[ -z $(git_local for-each-ref --format='%(refname)' refs/replace) ]]
readonly ORIGIN_RECORD=$(git_remote ls-remote --exit-code "$ORIGIN" "refs/heads/$BRANCH")
[[ $ORIGIN_RECORD == "$PIN"$'\t'"refs/heads/$BRANCH" ]]
exec 9>/home/gianl/glm-run/.glm_pod_workload.lock
/usr/bin/flock -n 9 || { echo "secondary pod lease unavailable" >&2; exit 1; }
exec 8>/home/gianl/.glm-tpu-rsync.lock
/usr/bin/flock 8

has_unique_markers() {
  local file=$1 marker=$2 expected=$3
  [[ $(/usr/bin/awk -v marker="$marker" '$1 == marker {print $2}' "$file" | /usr/bin/wc -l) -eq $expected ]] &&
    [[ $(/usr/bin/awk -v marker="$marker" '$1 == marker {print $2}' "$file" | /usr/bin/sort -u | /usr/bin/wc -l) -eq $expected ]]
}
has_eight_unique_markers() {
  has_unique_markers "$1" "$2" 8
}

readonly REPORT=/home/gianl/gate-d-runs/m2048-install-$PIN.log
[[ ! -e $REPORT ]]

# Workers 1..7 are clean, full, detached repositories at the observed sealed
# prestate. Refresh them serially to avoid the previously observed corruption
# from simultaneous GitHub SSH fetches. The refresher itself is the exact
# future-commit blob, transported in this command and executed from a sealed
# memfd; it refuses dirty, linked, shallow, promisor, branch-attached, replaced
# or unknown-prestate repositories before mutation.
readonly REPO_REFRESHER_B64=$(git_local show "$PIN:scripts/greenfield/$REPO_REFRESHER" | /usr/bin/base64 -w0)
[[ $(/usr/bin/printf '%s' "$REPO_REFRESHER_B64" | /usr/bin/base64 -d | /usr/bin/sha256sum | /usr/bin/awk '{print $1}') == "$REPO_REFRESHER_SHA" ]]
# shellcheck disable=SC2016
repo_refresh_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; [[ $idx == "$expected_worker" && $idx =~ ^[1-7]$ ]]; encoded='"$REPO_REFRESHER_B64"'; expected='"$REPO_REFRESHER_SHA"'; target='"$PIN"'; prestate='"$WORKER_REPO_PRESTATE_PIN"'; loader="import fcntl,hashlib,os,sys; raw=sys.stdin.buffer.read(); expected=sys.argv[1]; target=sys.argv[2]; prestate=sys.argv[3]; assert hashlib.sha256(raw).hexdigest()==expected; fd=os.memfd_create(\"gate-d-m2048-repo-refresher\",os.MFD_CLOEXEC|getattr(os,\"MFD_ALLOW_SEALING\",2)); stream=os.fdopen(os.dup(fd),\"wb\",closefd=True); written=stream.write(raw); stream.flush(); stream.close(); assert written==len(raw); os.fchmod(fd,0o400); seals=getattr(fcntl,\"F_SEAL_SEAL\",1)|getattr(fcntl,\"F_SEAL_SHRINK\",2)|getattr(fcntl,\"F_SEAL_GROW\",4)|getattr(fcntl,\"F_SEAL_WRITE\",8); fcntl.fcntl(fd,getattr(fcntl,\"F_ADD_SEALS\",1033),seals); assert fcntl.fcntl(fd,getattr(fcntl,\"F_GET_SEALS\",1034))==seals; os.set_inheritable(fd,True); path=f\"/proc/self/fd/{fd}\"; os.execve(\"/usr/bin/python3\",[\"/usr/bin/python3\",\"-I\",\"-S\",\"-B\",path,\"--target-pin\",target,\"--prestate-pin\",prestate],{\"HOME\":\"/home/gianl\",\"LANG\":\"C\",\"LC_ALL\":\"C\",\"PATH\":\"/usr/bin:/bin\",\"PYTHONDONTWRITEBYTECODE\":\"1\"})"; /usr/bin/printf "%s" "$encoded" | /usr/bin/base64 -d | /usr/bin/env -i HOME=/home/gianl LANG=C LC_ALL=C PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -I -S -B -c "$loader" "$expected" "$target" "$prestate"'
for worker in 1 2 3 4 5 6 7; do
  /snap/bin/gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker="$worker" \
    --command="expected_worker=$worker; $repo_refresh_command" >>"$REPORT" 2>&1
done
has_unique_markers "$REPORT" REPO_REFRESH_OK 7

# Verify all repositories together. Worker 0 remains the clean branch authority
# used by the launcher; refreshed workers remain detached at the exact pin.
# shellcheck disable=SC2016
sync_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; [[ $idx =~ ^[0-7]$ ]]; pin='"$PIN"'; branch='"$BRANCH"'; origin='"$ORIGIN"'; wt='"$WORKTREE"'; run_root=/home/gianl/gate-d-runs; git_local() { /usr/bin/env -i GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 GIT_NO_LAZY_FETCH=1 GIT_NO_REPLACE_OBJECTS=1 GIT_OPTIONAL_LOCKS=0 GIT_PROTOCOL_FROM_USER=0 GIT_SSH_COMMAND=/bin/false GIT_TERMINAL_PROMPT=0 HOME=/nonexistent LANG=C LC_ALL=C PATH=/usr/bin:/bin /usr/bin/git -c core.fsmonitor=false -c core.untrackedCache=false -c core.hooksPath=/dev/null -c core.attributesFile=/dev/null -C "$wt" "$@"; }; if [[ $idx == 0 ]]; then [[ -e $wt/.git ]]; [[ $(git_local branch --show-current) == "$branch" ]]; elif [[ ! -e $wt ]]; then /usr/bin/env -i GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 GIT_NO_LAZY_FETCH=1 GIT_NO_REPLACE_OBJECTS=1 GIT_OPTIONAL_LOCKS=0 GIT_TERMINAL_PROMPT=0 HOME=/home/gianl LANG=C LC_ALL=C PATH=/usr/bin:/bin GIT_SSH_COMMAND="/usr/bin/ssh -oBatchMode=yes -oClearAllForwardings=yes -oForwardAgent=no" /usr/bin/git -c core.fsmonitor=false -c core.hooksPath=/dev/null clone -q --filter=blob:none --single-branch --branch "$branch" "$origin" "$wt"; elif [[ ! -e $wt/.git ]]; then echo "stale non-repository worktree" >&2; exit 1; fi; [[ $(git_local rev-parse HEAD) == "$pin" ]]; [[ -z $(git_local status --porcelain=v1 --untracked-files=all) ]]; [[ -z $(git_local for-each-ref --format="%(refname)" refs/replace) ]]; if [[ ! -e $run_root ]]; then /usr/bin/mkdir -m 0700 "$run_root"; fi; [[ -d $run_root && ! -L $run_root ]]; [[ $(/usr/bin/readlink -f -- "$run_root") == "$run_root" ]]; [[ $(/usr/bin/stat -c "%F:%U:%G:%a" -- "$run_root") == "directory:gianl:gianl:700" ]]; /usr/bin/python3 -I -S -B -c "import os,sys; assert not os.listxattr(sys.argv[1],follow_symlinks=False)" "$run_root"; echo "REPO_OK $(hostname) $pin"'
/snap/bin/gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$sync_command" >>"$REPORT" 2>&1
has_eight_unique_markers "$REPORT" REPO_OK

readonly TRANSFER_NAME=gate-d-m2048-runtime-source-$PIN
prepare_transfer='set -euo pipefail; path=/home/gianl/'"$TRANSFER_NAME"'; [[ ! -e $path ]]; /usr/bin/mkdir -m 0700 "$path"; echo "TRANSFER_READY $(hostname)"'
/snap/bin/gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=1-7 \
  --command="$prepare_transfer" >>"$REPORT" 2>&1

# All traffic remains inside the TPU pod's us-central2-b placement.
/snap/bin/gcloud compute tpus tpu-vm scp --zone "$ZONE" --worker=1-7 \
  --recurse --compress \
  "/opt/glm-tpu/$PYTHON_NAME" "/opt/glm-tpu/$JAX_NAME" "/opt/glm-tpu/$LIBTPU_NAME" \
  "$POD:/home/gianl/$TRANSFER_NAME/" >>"$REPORT" 2>&1

# Execute the bootstrap helper itself from an exact sealed Git blob. It passes
# the provisioner to root through a second sealed memfd and publishes with
# RENAME_NOREPLACE; no privileged process reads a mutable worktree pathname.
# shellcheck disable=SC2016
runtime_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; wt='"$WORKTREE"'; pin='"$PIN"'; bootstrap='"$BOOTSTRAP"'; helper_sha='"$BOOTSTRAP_SHA"'; provisioner='"$PROVISIONER_TARGET"'; expected='"$PROVISIONER_SHA"'; git_local() { /usr/bin/env -i GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 GIT_NO_LAZY_FETCH=1 GIT_NO_REPLACE_OBJECTS=1 GIT_OPTIONAL_LOCKS=0 GIT_PROTOCOL_FROM_USER=0 GIT_SSH_COMMAND=/bin/false GIT_TERMINAL_PROMPT=0 HOME=/nonexistent LANG=C LC_ALL=C PATH=/usr/bin:/bin /usr/bin/git -c core.fsmonitor=false -c core.untrackedCache=false -c core.hooksPath=/dev/null -c core.attributesFile=/dev/null -C "$wt" "$@"; }; loader="import fcntl,hashlib,os,sys; raw=sys.stdin.buffer.read(); expected=sys.argv[1]; pin=sys.argv[2]; assert hashlib.sha256(raw).hexdigest()==expected; fd=os.memfd_create(\"gate-d-provisioner-bootstrap\",os.MFD_CLOEXEC|getattr(os,\"MFD_ALLOW_SEALING\",2)); stream=os.fdopen(os.dup(fd),\"wb\",closefd=True); written=stream.write(raw); stream.flush(); stream.close(); assert written==len(raw); os.fchmod(fd,0o400); seals=getattr(fcntl,\"F_SEAL_SEAL\",1)|getattr(fcntl,\"F_SEAL_SHRINK\",2)|getattr(fcntl,\"F_SEAL_GROW\",4)|getattr(fcntl,\"F_SEAL_WRITE\",8); fcntl.fcntl(fd,getattr(fcntl,\"F_ADD_SEALS\",1033),seals); assert fcntl.fcntl(fd,getattr(fcntl,\"F_GET_SEALS\",1034))==seals; os.set_inheritable(fd,True); path=f\"/proc/self/fd/{fd}\"; os.execve(\"/usr/bin/python3\",[\"/usr/bin/python3\",\"-I\",\"-S\",\"-B\",path,\"--bootstrap\",expected,pin],{\"HOME\":\"/home/gianl\",\"LANG\":\"C\",\"LC_ALL\":\"C\",\"PATH\":\"/usr/bin:/bin\",\"PYTHONDONTWRITEBYTECODE\":\"1\"})"; git_local show "$pin:scripts/greenfield/$bootstrap" | /usr/bin/env -i HOME=/home/gianl LANG=C LC_ALL=C PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -I -S -B -c "$loader" "$helper_sha" "$pin"; [[ "$provisioner" == /opt/glm-tpu/bin/provision_gate_d_python_runtime.py ]]; [[ -f $provisioner && ! -L $provisioner ]]; [[ $(/usr/bin/readlink -f -- "$provisioner") == "$provisioner" ]]; [[ $(/usr/bin/stat -c "%F:%h:%U:%G:%a" -- "$provisioner") == "regular file:1:root:root:555" ]]; [[ $(/usr/bin/sha256sum "$provisioner" | /usr/bin/awk "{print \$1}") == "$expected" ]]; /usr/bin/python3 -I -S -B -c "import os,sys; assert not os.listxattr(sys.argv[1],follow_symlinks=False)" "$provisioner"; if [[ $idx == 0 ]]; then source_root=/opt/glm-tpu; else source_root=/home/gianl/'"$TRANSFER_NAME"'; fi; for binding in '"$PYTHON_NAME:$PYTHON_TREE"' '"$JAX_NAME:$JAX_TREE"' '"$LIBTPU_NAME:$LIBTPU_TREE"'; do name=${binding%%:*}; tree=${binding##*:}; /usr/bin/sudo -n /usr/bin/env -i HOME=/root LANG=C LC_ALL=C PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -I -S "$provisioner" --source "$source_root/$name" --target "/opt/glm-tpu/$name" --expected-tree-sha256 "$tree"; done; echo "RUNTIME_OK $(hostname)"'
/snap/bin/gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$runtime_command" >>"$REPORT" 2>&1
has_eight_unique_markers "$REPORT" RUNTIME_OK

# Build a small, digest-bound user staging tree from the exact commit. The
# generic root provisioner publishes it without replacing an existing target.
# shellcheck disable=SC2016
install_command='set -euo pipefail; wt='"$WORKTREE"'; pin='"$PIN"'; stage=/home/gianl/gate-d-runs/'"$INSTALL_SOURCE_NAME"'-staging-$pin; target=/opt/glm-tpu/'"$INSTALL_SOURCE_NAME"'; git_local() { /usr/bin/env -i GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 GIT_NO_LAZY_FETCH=1 GIT_NO_REPLACE_OBJECTS=1 GIT_OPTIONAL_LOCKS=0 GIT_PROTOCOL_FROM_USER=0 GIT_SSH_COMMAND=/bin/false GIT_TERMINAL_PROMPT=0 HOME=/nonexistent LANG=C LC_ALL=C PATH=/usr/bin:/bin /usr/bin/git -c core.fsmonitor=false -c core.untrackedCache=false -c core.hooksPath=/dev/null -c core.attributesFile=/dev/null -C "$wt" "$@"; }; [[ ! -e $stage ]]; /usr/bin/install -d -m 0755 "$stage"; for binding in '"$INSTALLER:$INSTALLER_SHA"' '"$LAUNCHER:$LAUNCHER_SHA"' '"$PROBE:$PROBE_SHA"' '"$PUBLISHER:$PUBLISHER_SHA"' '"$MIRROR:$MIRROR_SHA"'; do name=${binding%%:*}; digest=${binding##*:}; /usr/bin/install -m 0555 "$wt/scripts/greenfield/$name" "$stage/$name"; [[ $(/usr/bin/sha256sum "$stage/$name" | /usr/bin/awk "{print \$1}") == "$digest" ]]; [[ $(git_local show "$pin:scripts/greenfield/$name" | /usr/bin/sha256sum | /usr/bin/awk "{print \$1}") == "$digest" ]]; done; /usr/bin/sudo -n /usr/bin/env -i HOME=/root LANG=C LC_ALL=C PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -I -S /opt/glm-tpu/bin/provision_gate_d_python_runtime.py --source "$stage" --target "$target" --expected-tree-sha256 '"$INSTALL_SOURCE_TREE"'; /usr/bin/sudo -n /usr/bin/env -i HOME=/root LANG=C LC_ALL=C PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -I -S -B "$target/'"$INSTALLER"'"; echo "M2048_INSTALL_OK $(hostname) launcher_invoked=false"'
/snap/bin/gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$install_command" >>"$REPORT" 2>&1
has_eight_unique_markers "$REPORT" M2048_INSTALL_OK

# Remove only the per-pin unprivileged transfer trees created above. The
# immutable /opt targets and tiny source staging remain for later audit.
cleanup_command='set -euo pipefail; path=/home/gianl/'"$TRANSFER_NAME"'; [[ -d $path && $path == /home/gianl/gate-d-m2048-runtime-source-'"$PIN"' ]]; /usr/bin/rm -rf -- "$path"; [[ ! -e $path ]]; echo "TRANSFER_CLEAN $(hostname)"'
/snap/bin/gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=1-7 \
  --command="$cleanup_command" >>"$REPORT" 2>&1

/usr/bin/printf 'M2048_FLEET_INSTALL_COMPLETE pin=%s report=%s launcher_invoked=false\n' \
  "$PIN" "$REPORT"
