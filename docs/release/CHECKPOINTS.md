# GLM-5.3 checkpoint and capacity

Weights are external to Git. Official FP8 `zai-org/GLM-5.3` is pinned at
`aca966e4e02791568aa6a4ced368624b3d897f42`:141shards,755632050320bytes.
The canonical source is `gs://driftbench-dsv4-uc/models/GLM-5.3-FP8/`, US-CENTRAL2.
All upstream SHA256s and cloud generations were verified; SOURCE_COMPLETE.json
binds the completed acquisition. Config/tokenizer/template/license identities
are recorded separately from frozen GLM-5.2 evidence.

[configs/glm53-site.json](../../configs/glm53-site.json) gives the exact inventory,
manifest,SUCCESS and completion hashes and paths used by inference. On each host,
four physically assigned owner files live under
`/dev/shm/glm-ws32-runtime/greenfield_ws32_runtime_pack_20260921T194714002533000Z/`.
All32files total786181673984bytes, approximately98.27GB host RAM per host.
Their SHA256s were freshly checked across all8hosts. The ordinary path already
carries dense weights in these owners and creates resident BF16 tensors; it does
not consume the legacy dense overlay. No second full packed GCS copy is required.

Source inventory is under
`checkpoints/greenfield/glm53/greenfield_ws32_runtime_pack_20260921T194148809384000Z/`.
Sealed metadata and verified regional backups are under the corresponding
`greenfield_ws32_runtime_pack_20260921T194714002533000Z/seal/` prefix.
Topology captures are authenticated by the retained rebinding before live device
initialization; private site access and the existing approved GCS mount are required.

The fixed fleet has32TPUv4chips, each with32GiB physical HBM. Reported runtime
memory limit was33014398976bytes/chip; maximum observed use28789189632bytes/chip.
Host tmpfs checkpoint storage is separate from HBM. Packed bytes, resident BF16
weights, caches and compiler temporaries explain why file size alone is not a
concurrency budget. Four32K caches passed with short inputs; no larger claim.

Recovery uses the retained `pack_ws32_runtime_slots` library through
[scripts/release/ws32_pack_worker.py](../../scripts/release/ws32_pack_worker.py),
with the exact new model geometry, source inventory and host-to-slot binding.
[Distributed manifest assembly](../../glm_tpu/optimized/checkpoint.py) combines
real hashed owner receipts; the loader rechecks each local payload. The protected
packing/sealing drivers and exact receipts remain in the private migration handoff.
Recovery requires the site operator, both workload/sync locks, authenticated idle
hosts and approximately107GB free tmpfs per host. It is not a portable wheel-only
bootstrap. Never repack an intact live checkpoint or invent completion seals.

Tmpfs is lost on host restart. Reconstruct only when genuinely absent; preserve
source, history and recovery receipts. The old5.2weights were deliberately
retired. Legacy checkpoint commands are history and must not be run against5.3.
Prior details remain at `git show glm-5.2:docs/release/CHECKPOINTS.md`.
