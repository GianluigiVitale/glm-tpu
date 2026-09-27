# GLM-5.3 checkpoint and capacity

Weights are external to Git and to the wheel.

## The source

The official FP8 `zai-org/GLM-5.3` checkpoint is pinned at revision
`aca966e4e02791568aa6a4ced368624b3d897f42`: 141 shards, 755,632,050,320 bytes.
Every upstream SHA-256 and every cloud generation was verified when it was
acquired, and a completion marker, `SOURCE_COMPLETE.json`, binds the completed
acquisition. The site file names where the source lives (`storage.source_uri`,
under one of `storage.allowed_source_uri_prefixes`) and where the tokenizer
files and the completion marker are on each host (`paths.model_path`). The
chat template, `config.json` and `generation_config.json` ship in
`glm_tpu/models/glm_moe_dsa/hf_config/` and are checked against pinned SHA-256
digests, and `tokenizer.json` and `tokenizer_config.json` at the model path are
checked against pinned digests, before a request is prepared. The `LICENSE` in
`hf_config/` is shipped as is; it is neither pinned nor checked.

## The packed runtime checkpoint

The runtime reads a packed checkpoint: 32 owner files, one per device slot, in
their final expert-8 x feature-4 layout, each behind a fixed header. Every host
holds only its own four slots, in a tmpfs directory (`checkpoint.root`, inside
`checkpoint.namespace`). A `manifest.json` records every file and tensor and is
published only after every source, destination file and tensor checksum was
verified; a separately published, self-hashed `SUCCESS` seal completes the
checkpoint. The measured GLM-5.3 pack has 32 files, 786,181,673,984 bytes,
about 98.27 GB of host RAM per host. The ordinary path binds the dense layers
directly from these owner files and builds its resident BF16 tensors from them;
no separate dense overlay is needed.

The site file's `[checkpoint]` table pins the checkpoint the runtime may load:
the source inventory and its digest, the manifest and SUCCESS digests, and the
digest of `SOURCE_COMPLETE.json`; its `[topology]` table pins the topology
binding, the fleet mapping and the mesh digest.
[`examples/site.example.toml`](../../examples/site.example.toml) documents every
key.

Before it opens a device, every worker checks the completion marker and the
model identity, re-derives every file record from the source inventory and the
pinned geometry, verifies the manifest, SUCCESS, mesh and topology pins, and
hashes its own four files (`glm_tpu/runner/tpu_runner.py`,
`glm_tpu/model_loader/sharded_state/verify.py`). The loader checks every tensor's
SHA-256 and finiteness again as it places it on the device.

## Inventory and verify on local files

Two read-only commands expose the checkpoint library; neither contacts another
host, takes a lock or imports JAX.

```bash
python -m glm_tpu checkpoint inventory /path/to/GLM-5.3-FP8 \
  --output /path/to/new-inventory.json --model-id zai-org/GLM-5.3 \
  --revision aca966e4e02791568aa6a4ced368624b3d897f42
python -m glm_tpu checkpoint verify --site /path/to/site.toml
python -m glm_tpu checkpoint verify --site /path/to/site.toml --slots 0 1 2 3 --local-slot-layout
```

`checkpoint inventory` reads the index, every shard's safetensors header and
`config.json` of a local source directory (never a tensor payload), writes the
inventory to a new file (an existing file is refused) and reads it back.

`checkpoint verify` checks a sealed runtime checkpoint against the site file's
pins: the source inventory's digest, the manifest, the SUCCESS seal, the mesh
and topology digests, every file record re-derived from the inventory and the
GLM-5.3 geometry, the size of every file present, and the SHA-256 of every file
(or only of `--slots`). `--root` verifies another directory than
`checkpoint.root`; `--local-slot-layout` accepts a directory that holds only the
`--slots` files, as a worker's does. Both commands print a JSON report and exit
0, or print the library's refusal as JSON on standard error and exit 1.

What `checkpoint verify` does not check, and a worker does: the source
completion marker `SOURCE_COMPLETE.json`, the model identity recorded in the
inventory, which four slots the host owns (a worker derives them from the
topology binding; pass them with `--slots` and `--local-slot-layout`), and, when
`--root` is given, that the directory lies inside `checkpoint.namespace`. Every
pin comes from the site file; no flag overrides one, so verifying a copy sealed
with other pins needs a site file with those pins.

## Packing

Packing is a fleet workflow that this repository does not contain: a private
driver stages the source code and the controller-resolved site configuration to
every host and runs `python -m glm_tpu.model_loader.pack_worker` there. The pack
worker verifies the staged source and the pins, then writes its host's owner
files with `pack_runtime_slots` (`glm_tpu/model_loader/sharded_state/writer.py`)
from the source inventory, the pinned geometry and the host-to-slot binding;
`assemble_owner_manifest` (`glm_tpu/model_loader/sharded_state/manifest.py`)
combines the eight hosts' hashed owner receipts into the manifest. The driver
must follow the current contracts: the pack worker reads only the site
configuration staged owner-only as `site.json` in its run directory
(`SiteConfig.resolved_json()` of the site file) and requires `--site-sha256` with
that file's SHA-256; the remote helpers are the files of
`glm_tpu/executor/remote/`, sent with one JSON argument; cleanup
(`glm_tpu.executor.multihost_executor.cleanup_owned`) takes the authenticated
host list, the site's fleet, the run's helper snapshot and the pack worker module.

A repack needs the site operator, both workload and sync locks, authenticated
idle hosts and about 107 GB of free tmpfs per host. It is not a wheel-only
bootstrap. Never repack an intact live checkpoint or invent a completion seal;
reconstruct only when the files are genuinely absent (tmpfs is lost when a host
restarts), and keep the source, its history and the recovery receipts.

## Capacity

The fleet has 32 TPU v4 chips with 32 GiB of HBM each. The reported runtime
memory limit was 33,014,398,976 bytes per chip; the maximum observed use was
28,789,189,632 bytes per chip. Host tmpfs storage of the checkpoint is separate
from HBM. Packed bytes, resident BF16 weights, caches and compiler temporaries
together explain why file size alone is not a concurrency budget. Four 32K
caches passed with short inputs; no larger claim is made. The context profiles
and their admission results are in
[glm53-context-profiles-20260922.json](glm53-context-profiles-20260922.json).

The GLM-5.2 weights were deliberately retired; their checkpoint commands are
history and must not be run against GLM-5.3. The GLM-5.2 page is preserved:
`git show glm-5.2:docs/release/CHECKPOINTS.md`.
