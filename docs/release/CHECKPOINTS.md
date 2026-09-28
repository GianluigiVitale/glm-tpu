# GLM-5.3 checkpoint and capacity

Weights are external to Git and to the wheel.

## The source

The official FP8 `zai-org/GLM-5.3` checkpoint is pinned at revision
`aca966e4e02791568aa6a4ced368624b3d897f42`: 141 shards, 755,632,050,320 bytes.
A completion marker, `SOURCE_COMPLETE.json`, binds a verified copy: every shard
and metadata file equal to the upstream repository in size and digest
(`checkpoint mark-source`, [below](#mark-the-source)). The site file names where the source lives (`storage.source_uri`,
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

## Mark the source

```bash
python -m glm_tpu checkpoint mark-source /path/to/GLM-5.3-FP8
python -m glm_tpu checkpoint mark-source /path/to/copy --output /path/to/new-marker.json --upstream-marker /path/to/GLM-5.3-FP8/SOURCE_COMPLETE.json
```

`checkpoint mark-source` hashes every safetensors shard that the index names and
the metadata files the engine reads (`config.json`, `generation_config.json`,
`model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json`,
`chat_template.jinja`), then compares each file's size and digest with the
upstream listing: by default the Hugging Face repository `zai-org/GLM-5.3` at the
pinned revision (the LFS SHA-256 of each shard, the git blob id of each small
file; a network call, nothing is downloaded), or with `--upstream-marker` the
digests of an earlier marker, to check another copy of the same source. Only
when the index names exactly the upstream shards, there are 141 of them with
755,632,050,320 bytes and every file agrees does it write the marker, a new
owner-only file (default: `SOURCE_COMPLETE.json` in the source directory; an
existing file is refused). The report on standard output holds the marker's
SHA-256, the site's `checkpoint.source_complete_sha256`. The workers and the pack
worker read the marker from `paths.model_path` on every host, so every host needs
the same bytes there. Hashing the full source takes a while (`--workers`, default
8, files in parallel); a refusal prints the differences as JSON on standard error
and writes nothing.

## Packing

`checkpoint pack` packs, seals and installs the checkpoint on the eight hosts, on
their CPUs (no TPU is opened). It is a fleet job with the controller's rules
([OPERATIONS](OPERATIONS.md#the-topology-binding)): the site's launch policy, both
workload locks for the whole job and the sync locks until staging and the CPU preflight are done, idle
hosts before and after, cleanup of only its own authenticated processes, and a
collection that never overwrites. It stages the pinned commit, the resolved site
(`site.json`, bound by `--site-sha256`) and the site's topology binding, then runs
`python -m glm_tpu.model_loader.pack_worker` on every host; each host checks the
staged source, the pins, the completion marker, the source inventory and the
binding before it reads a byte of the source.

```bash
python -m glm_tpu checkpoint pack --site ~/.config/glm-tpu/site.toml --preflight-only
python -m glm_tpu checkpoint pack --site ~/.config/glm-tpu/site.toml
python -m glm_tpu checkpoint pack --site ~/.config/glm-tpu/site.toml --recover-seal /path/to/kept-seal
python -m glm_tpu checkpoint pack --site ~/.config/glm-tpu/site.toml --compare-seal /path/to/kept-seal --tensors 8
```

- **Pack** (no mode flag): every host writes its four slots into
  `checkpoint.root` (a new directory directly inside `checkpoint.namespace`,
  named `greenfield_ws32_runtime_pack_<UTC>`; the run directory under
  `paths.run_root` takes the same name, so a name is used once), hashing each file and
  tensor it writes (`pack_runtime_slots`). The controller combines the eight
  hashed receipts into the manifest (`assemble_owner_manifest`, with each source
  shard's upstream SHA-256 from `SOURCE_COMPLETE.json`), writes the self-hashed
  `SUCCESS` seal for it (with the SHA-256s of this run's preflight, terminal and
  post-run idle records), stages both into the run's `seal/`, and every host
  installs them into its root and verifies the root as a worker does before a
  load. The report holds the new `manifest_sha256` and `success_sha256` for the
  site's `[checkpoint]` table.
- **Recover** (`--recover-seal DIR`, a directory holding a kept `manifest.json`
  and `SUCCESS`): the same pack, then every file record of the eight receipts
  must equal the kept manifest on every key, and only then is the kept seal
  installed byte for byte. The site must already pin that seal, and
  `checkpoint.root` needs a new name (the first pack's run directory keeps the
  old one; the seal does not depend on the root's name). This rebuilds a
  checkpoint that a host restart removed from tmpfs without inventing a new
  seal.
- **Compare** (`--compare-seal DIR`, a directory holding a `manifest.json`): a dry
  run that packs nothing and writes nothing to tmpfs. Every host compares all 32
  file plans (re-derived from the inventory and the geometry) with the manifest
  and re-derives `--tensors` tensors of each of its slots (0 for all) in memory
  from the source, as the packer places them, comparing their SHA-256s with the
  manifest's. The choice covers the FP8 and BF16 weights, routed experts and each
  sharded axis. The report lists every difference and the command exits 1 when
  there is one. `--tensors 0` holds one whole slot's tensors in memory at a time
  (about 25 GB) and reads the whole source once per host.
- **Preflight** (`--preflight-only`): the pack's checks on every host (among them
  that `checkpoint.root` does not exist yet and that there is enough tmpfs for the
  host's slots plus an 8 GiB reserve), then each host's facts; nothing is packed.

The packed files live in tmpfs, and systemd-logind removes a user's `/dev/shm`
files when that user's last session on a host ends (`RemoveIPC`): before a pack,
enable lingering for the operator account (`loginctl enable-linger`) or turn
`RemoveIPC` off on every host. A pack needs the site operator, about 107 GB of
free tmpfs per host and hours of CPU time on every host (the measured GLM-5.3 pack took 2 h 40 min through a
read-only cloud-storage mount); `--wall-seconds` bounds it. Nothing is retried. A
failed pack leaves its partial files and receipts for inspection; a new attempt
needs a new `checkpoint.root` name, or the partial root removed on the affected
hosts once its owner is proven. Never repack an intact live checkpoint or invent a
completion seal; reconstruct only when the files are genuinely absent (tmpfs is
lost when a host restarts), and keep the source, its history and the kept seal.

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
