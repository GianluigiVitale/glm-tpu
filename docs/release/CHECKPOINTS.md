# Checkpoint loading and recovery

Weights are external to Git. The working setup uses canonical FP8 source weights,
a small dense-layout overlay, and32 final-owner shards distributed across host
RAM (four shards per host). It does **not** require preserving PP8/PP16 experiment
payloads or another full WS32 runtime copy in the bucket.

## Required retained objects

All cloud paths below are relative to `gs://driftbench-dsv4-uc` in US-CENTRAL2.

| Object/prefix | Why retained |
|---|---|
| `models/GLM-5.2-FP8/` |141 canonical weight shards, model index/configuration and tokenizer; direct source of reconstruction |
| `checkpoints/greenfield/glm52/plans/PP8_LP4/greenfield_checkpoint_plan_pp8_20260805T180552087295643Z/source_inventory.json` |Complete source tensor/header inventory; the PP8 path name does not make a PP8 weight pack a dependency |
| `results/greenfield_ws32_runtime_pack_20260815T214050854386790Z/` |Preserved sealed WS32 manifest, SUCCESS and source-generation/hash provenance |
| `checkpoints/greenfield/glm52/overlays/WS32_2D/greenfield_ws32_strategy_nd_dense_overlay_pack_20260827T002508229552699Z/` |96 dense-overlay tensor files plus sealed manifest/SUCCESS |
| `results/greenfield_topology_20260826T194116460015528Z/host_records/` |Per-host slot/topology mapping checked against the actual runtime |

Keep existing compact scientific evidence and original recovery receipts as well.
This list is not permission to delete other objects: the release dependency audit
and preservation rules still apply.

The read-only2026-09-14 check found:

- Canonical weight files: **755,632,050,320 bytes**. All141 current GCS generations,
  sizes and CRCs match the preserved source ledger, whose SHA256 values agree with
  the sealed runtime manifest.
- Dense-overlay weight files: **2,102,200,128 bytes**, all96 present at the sealed
  sizes. Their payload hashes were not reread; this check is weaker than the
  protected loader's payload verification.
- Reconstructed32 runtime files: **786,181,673,984 bytes** in RAM across the hosts,
  about98.273GB per host. This is not a second retained GCS weight copy.

Combined canonical+overlay weight files are approximately**757.734GB decimal**,
excluding tokenizer, metadata and results. This is not the entire bucket's live
usage. Exact metadata/generation receipt:
[checkpoint-recovery-metadata-20260914.json](checkpoint-recovery-metadata-20260914.json).

Recheck retained identities without reading weights or modifying anything:

```bash
JAX_PLATFORMS=cpu python tools/check_checkpoint_recovery.py
```

The command authenticates the sealed metadata, compares141 canonical objects and
checks96 overlay file sizes. It never packs, loads, deletes or uploads weights.
It is a metadata audit, **not a fresh end-to-end recovery test**.

## Normal loading

The loader expects the local, physically assigned four shards at:

```text
/dev/shm/glm-ws32-runtime/greenfield_ws32_runtime_pack_20260815T214050854386790Z/
```

That directory also holds the sealed `manifest.json` and `SUCCESS`. Canonical
weights, inventory, overlay and topology are available via the existing approved
GCS mount at `/home/gianl/gcs-models`. Normal inference reuses this working set;
it must not recreate weights merely because the code branch changed.

The protected native worker uses `verify_ws32_runtime_checkpoint` and
`load_ws32_runtime_checkpoint`, then the original WK/exact materializers and
all-resident HLO/HBM admission. Metadata being present does not bypass payload,
scale, slot ownership or device-memory checks. Tmpfs contents disappear on host
restart; same-live-session KV resume does not recover from that loss.

## Reconstruct only when the RAM runtime is genuinely absent

The retained implementation is
`scripts/greenfield/run_ws32_runtime_checkpoint_shm_pack.sh`, calling the unchanged
`pack_ws32_runtime_checkpoint.py pack-slots` directly from canonical source.
Do **not** use `run_ws32_runtime_checkpoint_pack.sh` for ordinary recovery: that
older wrapper creates another full bucket layout.

The release RAM wrapper is default-off. It now requires the reviewed owner
branch/published pin, both workload and sync leases, authenticated idle workers,
the existing approved mount and topology, an absent output directory, and at
least100.5GB free tmpfs per host. Each host rebuilds only its four assigned shards
and checks every resulting file's full SHA256 and byte count against the sealed
manifest. Only small evidence records are uploaded; runtime weights stay in RAM.

Do not run recovery during the active benchmark or sealing. Do not remove its
existing working checkpoint to exercise this recipe. After release admission and
only if reconstruction is needed, the canonical execution checkout must be on
the explicitly selected, clean, published branch:

```bash
GLM_GREENFIELD_WS32_SHM_PACK=1 \
GLM_GREENFIELD_WS32_SHM_PACK_BRANCH=main \
bash scripts/greenfield/run_ws32_runtime_checkpoint_shm_pack.sh
```

This is a guarded, future recovery command, not an instruction to execute now.
It still uses the existing canonical execution path and existing hosts; it does
not provision, stop or recreate infrastructure. A partial target is a refusal
requiring exact ownership diagnosis, not permission to delete an arbitrary path.

Before an actual reconstruction, report the temporary RAM/disk budget and retained
storage impact: approximately786.182GB fleet RAM for runtime files, minimum
100.5GB free tmpfs/host, existing canonical+overlay objects retained, no new full
GCS runtime. The wrapper checks approved regional storage and reserves10GiB below
the2.5TB live cap before publication. Small evidence publication consumes storage.
The latest release wrapper has syntax/default-off/guard tests only; no fresh
pack was performed for cleanup. Preserve that distinction in release claims.
