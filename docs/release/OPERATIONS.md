# Operations and recovery constraints

Targets the existing 8-host/32-chip TPU v4 setup; does not provision infrastructure.
Never manage TPU/VM/node/queued resources, especially `db-v4-64-od-qr4`.

## Admission

Validate code/plan pins, topology, complete checkpoint/scale manifests, payload
integrity, per-chip memory and disk headroom before launch. Existing benchmark
requires 6 GiB free per host. Historical fit does not admit a changed capacity,
executable set or deployment. Use original protections, not a second launcher
that bypasses them. Both workload and sync leases serialize operations.

CPU tests require `JAX_PLATFORMS=cpu`. Never launch alongside an active campaign.
Its source/enforcement remain frozen through execution AND sealing. Release
preparation runs in an isolated worktree and does not touch its dependencies.

## Reviewed deployment branch

The release launcher accepts `--reviewed-branch main` (the default) or an explicit
`release/...` branch. The controller requires a clean checkout, unchanged model
source, the owner's private repository origin, and an exact code pin equal to
the selected remote branch HEAD. The canonical controller may be on that named
branch or detached at that exact pin; another named branch is refused. Detached
deployment avoids checking out the same branch in two Git worktrees and leaves
the research and release branch references intact. A detached checkout is not
permission to use an unpublished pin or bypass clean/model-source checks.
Workers independently check the origin and
fetched HEAD before checking out the pin. A moving branch is a refusal, not
permission to deploy unreviewed code.

Attach must use the original recorded branch, tag and pin. Historical launch
records without a branch field mean `rewrite/topology-first-decode`, never main;
pass that branch explicitly when using this launcher with historical records.

This removes the hard-coded research **branch**, not the existing fixed-site
worktree/path admission. The canonical execution path is still
`/home/gianl/glm-tpu-topology-rewrite`; the isolated release worktree is not
launch-admitted. Do not switch that live checkout or deploy this change during
the current campaign or sealing. Main deployment and generic user inference
remain release checklist items. No new TPU run has validated this launcher.

### Cutover order (not yet executed)

1. Establish that the original benchmark controller/workers are terminal from
   PID/start/boot ownership, not from elapsed time. Collect and seal its original
   evidence, including an explicitly recorded recovery pin if needed. A partial
   quality set remains partial; do not regenerate questions to simplify cutover.
2. Acquire both canonical leases and verify an authenticated idle eight-host
   census, source cleanliness, published owner refs and disk/RAM/retained assets.
   Fix any 6 GiB disk-floor shortfall only with verified expendable local
   copies; preserve active originals and do not make full-size weight backups.
3. Preserve the research branch at its published pin. Fetch the reviewed release
   ref and verify its exact expected commit before switching the **canonical
   checkout** to that detached commit. Do not move the research ref, force a
   branch already checked out elsewhere, reset files, or change the canonical
   path to evade source admission. The release worktree stays on its branch.
4. End the cutover lease scope, then invoke the default-off user controller with
   the same reviewed ref/pin. It independently reacquires both leases, repeats
   clean source/idle-fleet/asset checks and synchronizes the existing workers.
   A competing owner, changed ref or dirty checkout is a refusal, not a retry.
5. Run one bounded ordinary user prompt through the actual loader and response
   path. Preserve first-token output, continuation, terminal stop reason, fresh
   graph/HBM/trace evidence and authenticated cleanup. A cap-limited response
   need not contain a final answer; report that honestly. One token alone cannot
   validate decode. Diagnose new source/HLO mismatches rather than registering
   hashes blindly. Never repeat the already sealed 128K/256K campaigns here.
6. Seal the original user result, finish release review, verify regional mirror
   coverage after the sync lease is free, then merge the eligible release into
   private main. Record actual execution and release pins separately if later
   documentation changes differ. Keep research history and branches intact.

This sequence does not authorize source switching while the current campaign
or its sealing is active. CPU branch tests are preparation, not cutover evidence.

## Weights and storage

Weights are external, not in Git. Current runtime uses four final-layout
safetensors shards per host in tmpfs plus the dense overlay and canonical
source/metadata. Tmpfs is volatile; mounted weights alone do not prove cold
recovery. Release reconstruction instructions must be verified against retained
sources before claiming reproducibility.

Only `gs://driftbench-dsv4-uc`, US-CENTRAL2; live storage below 2.5 TB (decimal),
soft delete off. No full-size safety copies. Artifacts over 100 GB need a
peak/retained/replacement budget.
Private questions/answers and raw databases stay outside Git; retain compact receipts.

## Observe and recover

Existing watchdog verifies process/libtpu ownership. Manual checks >=10 minutes
apart unless diagnosing failure. Authenticate PID/start/boot/argv, not just logs.
Timeout is not restart authority. Recover the SAME original tag/pin rather than
rerunning successful model work; never fabricate terminal/publication markers.

Seal original request/trace/state/memory evidence to DB and exact regional
generations, then require authenticated 8/8 zero-work cleanup. Same live-session
resume is not process-crash KV recovery. Regional mirror verification waits
for the active sync lease; do not disable backups to make a release check pass.

The new user controller must run on worker0, where both canonical leases live.
It remains default-off and not deployment-admitted. It rejects attach-time
request/deadline overrides and observes ambiguous SSH dispatch without repeating
it. A prelaunch refusal and an unknown process wait are different: only the
former can produce an explicit worker_started=false ended record. Collection
success is transport only; user semantic replay/sealing still remains required.
See [user controller scope](INFERENCE.md) before attempting any invocation.

### User request troubleshooting

| Observation | Action |
|---|---|
| Default-off or wrong source/ref refusal | Finish deployment admission; do not bypass the guard or switch live source. |
| Workload or mirror lease busy | Observe the existing owner; never launch a duplicate or remove a lock. |
| Disk, memory or archive-cap refusal | Diagnose the exact bounded resource; do not lower the floor or create a full-size safety copy. |
| SSH observation lost during dispatch | Keep observing the same original owners; do not resend worker dispatch. |
| Worker succeeded, upload failed | Diagnose storage/authentication/headroom, then use same-pin `--attach --republish-originals` under both leases. |
| Model failed or no authenticated ended marker | Preserve the prefix and original logs; upload recovery cannot make this a completed request. |
| Collection/replay/DB/archive interrupted | Attach to the same originals; immutable rows/objects are reused, not regenerated. |
| Response stops at token cap during reasoning | Report the terminal reason; it is not necessarily a completed final answer or a quality pass. |

Upload-only recovery preserves original failed markers in a separate archived
receipt. It does not change model bytes, extend generation, waive original replay
or repair missing/corrupted evidence. See [the exact recovery scope](INFERENCE.md#recover-a-failed-upload-without-repeating-the-response).

The release worktree's explicit backup pair is staged but not installed. Follow
[mirror cutover](MIRROR_CUTOVER.md) only after the active sync lease is released;
the versioned script is not a way to bypass the installed cron's locks.
