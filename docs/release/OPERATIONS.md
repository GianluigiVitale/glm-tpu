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
the selected remote branch HEAD. Workers independently check the origin and
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
