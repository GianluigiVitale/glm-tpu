# GLM-5.3 migration

The owner requested freezing GLM-5.2, removing its stored weights, and moving
the same TPU implementation to GLM-5.3. The completed GLM-5.2 code is preserved
at private tag [glm-5.2](https://github.com/GianluigiVitale/glm-tpu/releases/tag/glm-5.2),
commit `edbced29315b6afce92cf994ae70785bf8f6d995`, with its verified source archive.
This branch is migration work, not a claim that GLM-5.3 inference has passed.

## Source choice and compatibility

Use [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3), its official FP8
checkpoint, pinned at `aca966e4e02791568aa6a4ced368624b3d897f42`.
It has 141 weight shards totaling755,632,050,320bytes. The BF16 variant has
282shards totaling1,506,667,387,408bytes; it is not the chosen representation.

Compared with the retained GLM-5.2-FP8 configuration, architecture fields and
normalized quantization configuration match. The declared Transformers version
changes from5.12.0 to5.15.0; this does not authorize an environment upgrade.
All118,629tensor-to-file mappings match. Tokenizer JSON, tokenizer configuration
and generation configuration match, but the separate chat template changes.
All141weight-file SHA256s differ from the current GLM-5.2-FP8 source revision.
These metadata checks support reusing the implementation; they do not prove
new-weight numerical behavior or answer correctness.

The new chat template adds reasoning-effort and tool-response behavior. It must
be used and separately pinned before inference. GLM-5.3 uses its own
[license](https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE), not the old
MIT label. Preserve that file with model assets and update attribution when
integrating them. Do not rewrite the frozen GLM-5.2 source/evidence identities.

## GLM-5.2 weight retirement

Owner-authorized deletion removed359inventoried cloud weight files totaling
1,706,606,624,800bytes from `models/GLM-5.2-FP8/` and
`checkpoints/greenfield/glm52/` in `gs://driftbench-dsv4-uc` (US-CENTRAL2).
Deleting68runtime/MTP/partial weight files reclaimed812,943,126,528bytes of
host RAM filesystem capacity across the eight hosts. All eight hosts passed
authenticated idle checks before and after. No TPU resources were changed.

Source inventories, config/tokenizer files, manifests, scientific evidence,
DB616–621, code, branches, licenses and Git history remain preserved. The exact
generation/inode inventories and deletion receipts are private under
`/home/gianl/glm-run/glm53_migration_20260921/` and
`gs://driftbench-dsv4-uc/results/glm53_migration_20260921/`.
Running GLM-5.2 again requires restoring its external weights; the release tag
does not contain them. Historical receipts remain historical facts, not claims
that retired payloads are still present.

## Next execution boundary

Acquire the pinned FP8 source into `models/GLM-5.3-FP8/` in the same bucket,
streaming without a full disk copy. Verify every byte count and upstream SHA256,
preserve headers/provenance, and publish a completion marker only after all
shards and metadata pass. Reconcile one completion notification, not polling.

Then adapt model/template identities, rebuild only the required owner shards
and dense overlay using the retained packing path, and check affected CPU and
bounded real-weight inference behavior. No GLM-5.2 answer/speed receipt qualifies
GLM-5.3. Keep main's last release and the tagged snapshot intact until the new
implementation is usable. No new resources, environment upgrades, automatic
workload retries or research campaign.
