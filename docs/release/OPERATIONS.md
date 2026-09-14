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
