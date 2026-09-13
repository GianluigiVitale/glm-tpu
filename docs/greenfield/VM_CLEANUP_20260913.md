# VM local cleanup — 2026-09-13

## Update 12:08Z — flood relieved, backlog still needs compression

Owner-approved healthagent-only1GiB limits now verified on0..6;7 unchanged.
No restart command/TPU/VM change. Agents2/5 exited2 after initial updates and
were automatically replaced; reapplied1GiB to exact replacement containers.
Original surviving OOM counters and all kern.log sizes stable over117s;
syslog growth on0..6~100KB/host, not the previous multi-GB/hour flood.
This short observation does not prove a memory-leak fix or persistent supervisor
configuration. Runtime limits may reset on container recreation.
APT downloaded caches also cleared on1/4 to permit Docker metadata writes;
installed packages retained. No further evidence/weights deleted in this step.
Free bytes0/1/4:3,924,123,648 /144,850,944 /146,513,920. Active log backlog
remains. Lossless rotation/compression must recover6GiB before benchmark launch.
Receipts: ../artifacts/healthagent-{limit-update,replacement-followup}-20260913.json.
The earlier measurements and approval-pending statements below are historical.

Owner authorized deleting verified useless/superseded local copies, without
destroying essential work. Root disk began100% full. Last free-space reading:
10,092,761,088B (10.09 decimal GB). This is not cloud storage cleanup.

## Removed or compacted

- APT downloaded package archives: `sudo apt-get clean`; packages/environments
  remain installed. Downloads can be fetched again. No exact reclaimed-byte
  claim: concurrent logs were growing.
- Six installer archives from `.vscode-server/data/CachedExtensionVSIXs`:
  anthropic.claude-code versions2.1.267,2.1.268,2.1.269 and openai.chatgpt
  versions26.903.71938,26.908.31748,26.908.40401, all linux-x64 (~1.05GB).
  No open holders. Installed extensions are separate; installers can be downloaded.
- Three identical old build libraries, each696,294,952B:
  `/home/gianl/gate-d-runs/libtpu-site-build-final.Vgphgx/second/libtpu/libtpu.so`,
  `/home/gianl/gate-d-runs/libtpu-site-build.sIcAEX/first/libtpu/libtpu.so`,
  `/home/gianl/gate-d-runs/libtpu-site-build.sIcAEX/second/libtpu/libtpu.so`.
  Retain `/home/gianl/gate-d-runs/libtpu-site-build-final.Vgphgx/first/libtpu/libtpu.so`
  as the exact local recovery source. All four SHA256:
  `087537754507b346bade7adbf5f8f99e70d7fd5992369208564227f79e7f69a3`.
  No open holders; no installed production environment changed. To reproduce
  an old staging layout, copy that retained binary back to the exact deleted path.
- Inactive editor copies,2,106,540,032 allocated bytes:
  `/home/gianl/.vscode-server/cli/servers/Stable-08d4889f9ec4a1685d257b9b95de036c8e1ce1e5`,
  `/home/gianl/.vscode-server/cli/servers/Stable-520fb30b2d3d324b4cb2342f6e88e2cd93751de1`,
  `/home/gianl/.vscode-server/extensions/openai.chatgpt-26.908.31748-linux-x64`.
  Fresh root /proc argv/exe/cwd/maps/fd scan found zero references to all three.
  Selected extension26.908.40401 and actually executing26.903.71938 kept;
  active servers645f29cc/88e44fa0 and outer CLI110a328e kept. Redownload old
  versions if explicitly needed. No user settings, workspaces or sessions removed.
- Closed `/var/log/kern.log.1` and `/var/log/syslog.1` compressed usinggzip-1.
  Original sizes2,103,978,169+2,630,076,516B; `.1.gz` sizes192,920,179+
  249,450,854B. Savings4,291,683,652B, full log contents retained.
  Both gzip integrity tests passed. A first compression attempt ran out of
  space and preserved originals; after cache deletion the retry succeeded.
  Active kern.log/syslog, authentication logs and journal were not discarded.
- 110 same-region archived backups,2,481,510,208B:
  96 local staging overlay shards and14 nonprimary database snapshots. Exact
  local paths, generation-qualified recovery URIs, SHA256/CRC/bytes/inodes and
  deletion outcomes are in `../artifacts/vm-local-backup-review-20260913.json`
  and `../artifacts/vm-local-backup-eviction-20260913.json`. Review SHA256:
  `4b92c960f695e44aee1954d2ee4b893b461356aa4044975773407f7048423bda`.
  Existing eviction engine held both leases, verified all files first, then
  reverified each immediately before descriptor-relative unlink. Cloud originals
  unchanged. The runner's tmpfs checkpoint/GCS overlay paths were not targets.

## Preserved and remaining

Git history/worktrees, primary results DB, all cloud evidence/model objects,
active tmpfs weights, active editor versions, Codex/Claude conversations and
attachments, potentially unique diagnostic scratch, and unverified historical
originals are retained. No blanket deletion based on age or directory name.

The main remaining large local collection is `glm-run` (~42GB before the
2.48GB verified eviction), which includes compiler originals and scientific
evidence, not uniformly useless backups. Any further removal needs exact
recovery/dependency verification, not recursive deletion of the run root.

The healthagent cgroup is still512MiB and its OOM counter reached4,310,220;
active kern.log+syslog are10,553,484,448B and growing. Cleanup alone does not
make a24h campaign safe. Specific permission for a reversible agent-only
limit increase was requested; no container/TPU/VM change was performed.
No new benchmark answer or quality result is claimed. Existing long-context
DB616–620 stay complete and must not be rerun.
