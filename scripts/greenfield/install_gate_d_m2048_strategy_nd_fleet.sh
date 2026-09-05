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
readonly PROVISIONER_TARGET=/opt/glm-tpu/bin/provision_gate_d_runtime_archive.py
readonly PROVISIONER_SHA=2582e9b6c91fe11a8489a2359f830ab786b5856463ea896172da1f7ab79b4319
readonly GENERIC_PROVISIONER_TARGET=/opt/glm-tpu/bin/provision_gate_d_python_runtime.py
readonly GENERIC_PROVISIONER_SHA=2b9c8c2b981be639ad0eb16388c6fbfdd4765adfa2ef9a1986ae4b1b37ec0594
readonly BOOTSTRAP=bootstrap_gate_d_provisioner.py
readonly BOOTSTRAP_SHA=d70e550adce72a7699142763099292cd60f09416b7f467e8a11f4c86ebd1b137
readonly REPO_REFRESHER=refresh_gate_d_m2048_worker_repository.py
readonly REPO_REFRESHER_SHA=f248ff6717113c2376e19e4495fd2b33314280b181b97c783996ec510b7f5b0e
readonly WORKER_REPO_PRESTATE_PIN=b7708936416452a7b453908abc729bfacbcb00c8
readonly PYTHON_NAME=gate-d-python-3.12.13-021044895e95
readonly PYTHON_TREE=308748a9a3c3758a6b4f233aa5c034e8cb419362dbeafe0322448be40170d616
readonly JAX_NAME=gate-d-jax-site-55233c63939e
readonly JAX_TREE=55233c63939ea28485cdf2f0fc3d9c1d2ce4d9d93aad828e94498d712a26a0df
readonly LIBTPU_NAME=gate-d-libtpu-site-db7598c867f3
readonly LIBTPU_TREE=db7598c867f370756813cbf1536ad8ef7b1d9c167975e9e1724bd9b4fee78eca
readonly INSTALL_SOURCE_NAME=gate-d-m2048-install-v4
readonly INSTALL_SOURCE_TREE=d764e13dd3ee79f89345aee829496602341faa232c3958c7d6605b60e70f105d
readonly INSTALLER=install_gate_d_m2048_strategy_nd_runtime.py
readonly LAUNCHER=launch_gate_d_m2048_strategy_nd_association.py
readonly PROBE=probe_m2048_strategy_nd_association.py
readonly PUBLISHER=publish_gate_d_m2048_strategy_nd_association.py
readonly MIRROR=verify_gate_d_rewrite_same_region_git_mirror.py
readonly INSTALLER_SHA=f34955aa345f0e81a45c7249e3b6aad2a9b4869406891a48ec31a8b0ecb943e2
readonly LAUNCHER_SHA=332128c5f91ed87d93021709b22e3690f7c2a06c1be62cf4acbd41b78e59382d
readonly PROBE_SHA=f708163ab9731cfa8195c78d0bb4b458e3a44c1858faecab331716427c70c39b
readonly PUBLISHER_SHA=fbf02bd4216b20e21855d877837c972db98736c636608bf74262f20126e51148
readonly MIRROR_SHA=8747647d88a4cf2d435a46b6ed94067ecc05910d1917c24122cba88f8d037c0a

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
readonly PYTHON_ARCHIVE_NAME=$PYTHON_NAME-$PIN.tar
readonly PYTHON_ARCHIVE=/opt/glm-tpu/$PYTHON_ARCHIVE_NAME

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

# Publish the versioned archive provisioner from the exact future-commit blob
# before any archive exists. The prior generic provisioner remains immutable
# and is independently checked where it handles the symlink-free trees.
# shellcheck disable=SC2016
bootstrap_command='set -euo pipefail; wt='"$WORKTREE"'; pin='"$PIN"'; bootstrap='"$BOOTSTRAP"'; helper_sha='"$BOOTSTRAP_SHA"'; provisioner='"$PROVISIONER_TARGET"'; expected='"$PROVISIONER_SHA"'; generic='"$GENERIC_PROVISIONER_TARGET"'; generic_expected='"$GENERIC_PROVISIONER_SHA"'; git_local() { /usr/bin/env -i GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 GIT_NO_LAZY_FETCH=1 GIT_NO_REPLACE_OBJECTS=1 GIT_OPTIONAL_LOCKS=0 GIT_PROTOCOL_FROM_USER=0 GIT_SSH_COMMAND=/bin/false GIT_TERMINAL_PROMPT=0 HOME=/nonexistent LANG=C LC_ALL=C PATH=/usr/bin:/bin /usr/bin/git -c core.fsmonitor=false -c core.untrackedCache=false -c core.hooksPath=/dev/null -c core.attributesFile=/dev/null -C "$wt" "$@"; }; loader="import fcntl,hashlib,os,sys; raw=sys.stdin.buffer.read(); expected=sys.argv[1]; pin=sys.argv[2]; assert hashlib.sha256(raw).hexdigest()==expected; fd=os.memfd_create(\"gate-d-provisioner-bootstrap\",os.MFD_CLOEXEC|getattr(os,\"MFD_ALLOW_SEALING\",2)); stream=os.fdopen(os.dup(fd),\"wb\",closefd=True); written=stream.write(raw); stream.flush(); stream.close(); assert written==len(raw); os.fchmod(fd,0o400); seals=getattr(fcntl,\"F_SEAL_SEAL\",1)|getattr(fcntl,\"F_SEAL_SHRINK\",2)|getattr(fcntl,\"F_SEAL_GROW\",4)|getattr(fcntl,\"F_SEAL_WRITE\",8); fcntl.fcntl(fd,getattr(fcntl,\"F_ADD_SEALS\",1033),seals); assert fcntl.fcntl(fd,getattr(fcntl,\"F_GET_SEALS\",1034))==seals; os.set_inheritable(fd,True); path=f\"/proc/self/fd/{fd}\"; os.execve(\"/usr/bin/python3\",[\"/usr/bin/python3\",\"-I\",\"-S\",\"-B\",path,\"--bootstrap\",expected,pin],{\"HOME\":\"/home/gianl\",\"LANG\":\"C\",\"LC_ALL\":\"C\",\"PATH\":\"/usr/bin:/bin\",\"PYTHONDONTWRITEBYTECODE\":\"1\"})"; git_local show "$pin:scripts/greenfield/$bootstrap" | /usr/bin/env -i HOME=/home/gianl LANG=C LC_ALL=C PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -I -S -B -c "$loader" "$helper_sha" "$pin"; for binding in "$provisioner:$expected" "$generic:$generic_expected"; do path=${binding%%:*}; digest=${binding##*:}; [[ -f $path && ! -L $path && $(/usr/bin/readlink -f -- "$path") == "$path" ]]; [[ $(/usr/bin/stat -c "%F:%h:%U:%G:%a" -- "$path") == "regular file:1:root:root:555" ]]; [[ $(/usr/bin/sha256sum "$path" | /usr/bin/awk "{print \$1}") == "$digest" ]]; /usr/bin/python3 -I -S -B -c "import os,sys; assert not os.listxattr(sys.argv[1],follow_symlinks=False)" "$path"; done; echo "BOOTSTRAP_OK $(hostname)"'
/snap/bin/gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all \
  --command="$bootstrap_command" >>"$REPORT" 2>&1
has_eight_unique_markers "$REPORT" BOOTSTRAP_OK

# The archive lives in the root-owned runtime directory. Its creator opens the
# output with O_EXCL|O_NOFOLLOW, writes through the retained descriptor, checks
# the source tree before and after, and returns the exact cleanup identity.
archive_record=$(/usr/bin/sudo -n /usr/bin/env -i HOME=/root LANG=C LC_ALL=C \
  PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -I -S -B \
  "$PROVISIONER_TARGET" create --source "/opt/glm-tpu/$PYTHON_NAME" \
  --archive-name "$PYTHON_ARCHIVE_NAME" --expected-tree-sha256 "$PYTHON_TREE")
read -r archive_marker python_archive_sha python_archive_device \
  python_archive_inode python_archive_size archive_extra <<<"$archive_record"
[[ $archive_marker == ARCHIVE_READY && -z ${archive_extra:-} ]]
[[ $python_archive_sha =~ ^[0-9a-f]{64}$ ]]
[[ $python_archive_device =~ ^[0-9]+$ && $python_archive_inode =~ ^[0-9]+$ ]]
[[ $python_archive_size =~ ^[1-9][0-9]*$ ]]
readonly PYTHON_ARCHIVE_SHA=$python_archive_sha
readonly PYTHON_ARCHIVE_DEVICE=$python_archive_device
readonly PYTHON_ARCHIVE_INODE=$python_archive_inode
readonly PYTHON_ARCHIVE_SIZE=$python_archive_size

readonly TRANSFER_NAME=gate-d-m2048-runtime-source-$PIN
prepare_transfer='set -euo pipefail; path=/home/gianl/'"$TRANSFER_NAME"'; [[ ! -e $path ]]; /usr/bin/mkdir -m 0700 "$path"; echo "TRANSFER_READY $(hostname)"'
/snap/bin/gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=1-7 \
  --command="$prepare_transfer" >>"$REPORT" 2>&1

# All traffic remains inside the TPU pod's us-central2-b placement.
/snap/bin/gcloud compute tpus tpu-vm scp --zone "$ZONE" --worker=1-7 \
  --recurse --compress \
  "$PYTHON_ARCHIVE" "/opt/glm-tpu/$JAX_NAME" "/opt/glm-tpu/$LIBTPU_NAME" \
  "$POD:/home/gianl/$TRANSFER_NAME/" >>"$REPORT" 2>&1

# The archive helper reads the transferred tar through one retained no-follow
# descriptor and publishes only an authenticated root-owned tree. No tar
# extraction or Python-runtime materialization occurs in the user-writable root.
# shellcheck disable=SC2016
runtime_command='set -euo pipefail; idx=${HOSTNAME##*-w-}; [[ $idx =~ ^[0-7]$ ]]; archive_provisioner='"$PROVISIONER_TARGET"'; archive_expected='"$PROVISIONER_SHA"'; generic_provisioner='"$GENERIC_PROVISIONER_TARGET"'; generic_expected='"$GENERIC_PROVISIONER_SHA"'; source_root=/home/gianl/'"$TRANSFER_NAME"'; archive_name='"$PYTHON_ARCHIVE_NAME"'; archive_sha='"$PYTHON_ARCHIVE_SHA"'; archive_size='"$PYTHON_ARCHIVE_SIZE"'; python_name='"$PYTHON_NAME"'; python_tree='"$PYTHON_TREE"'; jax_name='"$JAX_NAME"'; jax_tree='"$JAX_TREE"'; libtpu_name='"$LIBTPU_NAME"'; libtpu_tree='"$LIBTPU_TREE"'; for binding in "$archive_provisioner:$archive_expected" "$generic_provisioner:$generic_expected"; do path=${binding%%:*}; digest=${binding##*:}; [[ -f $path && ! -L $path && $(/usr/bin/readlink -f -- "$path") == "$path" ]]; [[ $(/usr/bin/stat -c "%F:%h:%U:%G:%a" -- "$path") == "regular file:1:root:root:555" ]]; [[ $(/usr/bin/sha256sum "$path" | /usr/bin/awk "{print \$1}") == "$digest" ]]; /usr/bin/python3 -I -S -B -c "import os,sys; assert not os.listxattr(sys.argv[1],follow_symlinks=False)" "$path"; done; if [[ $idx == 0 ]]; then for binding in "$python_name:$python_tree" "$jax_name:$jax_tree" "$libtpu_name:$libtpu_tree"; do name=${binding%%:*}; tree=${binding##*:}; /usr/bin/sudo -n /usr/bin/env -i HOME=/root LANG=C LC_ALL=C PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -I -S "$generic_provisioner" --source "/opt/glm-tpu/$name" --target "/opt/glm-tpu/$name" --expected-tree-sha256 "$tree"; done; else /usr/bin/python3 -I -S -B -c "import os,sys; assert set(os.listdir(sys.argv[1]))==set(sys.argv[2:])" "$source_root" "$archive_name" "$jax_name" "$libtpu_name"; /usr/bin/sudo -n /usr/bin/env -i HOME=/root LANG=C LC_ALL=C PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -I -S -B "$archive_provisioner" install --archive "$source_root/$archive_name" --archive-sha256 "$archive_sha" --archive-size "$archive_size" --source-name "$python_name" --target-name "$python_name" --expected-tree-sha256 "$python_tree"; for binding in "$jax_name:$jax_tree" "$libtpu_name:$libtpu_tree"; do name=${binding%%:*}; tree=${binding##*:}; /usr/bin/sudo -n /usr/bin/env -i HOME=/root LANG=C LC_ALL=C PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -I -S "$generic_provisioner" --source "$source_root/$name" --target "/opt/glm-tpu/$name" --expected-tree-sha256 "$tree"; done; fi; echo "RUNTIME_OK $(hostname)"'
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

/usr/bin/sudo -n /usr/bin/env -i HOME=/root LANG=C LC_ALL=C PATH=/usr/bin:/bin \
  PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -I -S -B "$PROVISIONER_TARGET" \
  remove --archive-name "$PYTHON_ARCHIVE_NAME" \
  --expected-sha256 "$PYTHON_ARCHIVE_SHA" \
  --expected-device "$PYTHON_ARCHIVE_DEVICE" \
  --expected-inode "$PYTHON_ARCHIVE_INODE" \
  --expected-size "$PYTHON_ARCHIVE_SIZE"
[[ ! -e $PYTHON_ARCHIVE ]]

/usr/bin/printf 'M2048_FLEET_INSTALL_COMPLETE pin=%s report=%s launcher_invoked=false\n' \
  "$PIN" "$REPORT"
