# GLM-5.3 migration history

**Final hardware outcome:** four correct completed answers, normal EOS, fresh
graph/memory checks, all-host agreement and eight-host cleanup passed at
`c2f60efe`, run `optimized_request_20260921T223026777871Z`.
[Current results](STATUS.md) · [Receipt](glm53-four-answers-20260921.json).
The sections below preserve intermediate states and superseded pending gates.
Exact pre-release text remains at `git show c2f60efe:docs/release/GLM53_MIGRATION.md`.

The owner requested freezing GLM-5.2, removing its stored weights, and moving
the same TPU implementation to GLM-5.3. The completed GLM-5.2 code is preserved
at private tag [glm-5.2](https://github.com/GianluigiVitale/glm-tpu/releases/tag/glm-5.2),
commit `edbced29315b6afce92cf994ae70785bf8f6d995`, with its verified source archive.
The completed migration retains the earlier failures and recovery details below.

## Complete source and current preparation

All141FP8 shards,755,632,050,320bytes, completed verification on September21
at19:39:56UTC. The pinned upstream SHA256s, cloud generations and completion
marker readback passed, along with eight-host cleanup. Private evidence is under
`glm53_migration_20260921/acquisition-resume-ranged-20260921/`.
The initial continuation stopped on a truncated HTTP body for shard31;
authenticated byte-range recovery retained38verified shards and completed the
remaining103. Original failed receipts and source generations remain preserved.

The first packing preflight rejected the new inventory because the retained
geometry parser unconditionally names GLM-5.2. No owner files were written;
eight-host cleanup passed and the completed inventory was preserved. The ordinary
packing and inference boundaries now use the pinned GLM-5.3 configuration and
model identity while retaining every numerical dimension and the frozen parser.
CPU checks cover this distinction. Packing subsequently completed on all8hosts
at `90777a44`, producing32owner files (786,181,673,984bytes). All local file hashes
passed the retained checkpoint verifier; eight-host cleanup and regional seal
metadata backup/readback passed. Inference and answer validation remain pending.
`configs/glm53-site.json` now binds these actual checkpoint/source identities.

The full CPU release check at `e5404830` passed627tests with one skip, plus
source/content/package checks. The geometry correction has separate affected
checks; passing CPU checks do not establish real-weight answers or speed.

## Acquisition recovery

The owner has authorized autonomous diagnosis, fixes and retries to finish the
release. The current continuation preserves the17verified source generations and
acquires the124missing shards. Its transfer checks now retain exact failure
phases, byte counts, hashes and traceback locations. Eight offline transfer and
failure-reporting cases passed. The original failed script and receipts remain
unchanged in the private task folder; current dispatch/terminal evidence lives
under `acquisition-resume-20260921/`. A resumed transfer is still not inference
validation. [Current goal and authorization](../../goal.md).

## Preserved initial acquisition failure

The dispatched acquisition exited unsuccessfully at2026-09-21T14:16:26UTC.
Shard `model-00015-of-00141.safetensors` raised `AssertionError`. The original
handler retained the file and exception type but discarded the assertion text
and traceback, so the precise failing check is unknown. A hash mismatch or
network failure has not been established.

Seventeen of141shards,91,184,236,216bytes, were verified and preserved. Their
cloud generations, sizes and CRC32C values match the receipts, whose SHA256s
match the pinned upstream LFS metadata. The failed shard was not published;
there is no `SOURCE_COMPLETE.json`. All eight cleanup logs identify distinct
authenticated idle hosts, and both workload leases were released.

The failed gate is **complete verified canonical source acquisition**. Packing
and GLM-5.3 inference had not started. The initial attempt stopped under the
then-current no-retry instruction, subsequently superseded by the owner's
recovery authorization above. Original evidence and partial source remain
intact. [Failure receipt](glm53-acquisition-failure-20260921.json).
The request integration at `fc95150d` has69passing affected CPU checks and a
passing offline package check; these do not qualify the missing weights.

## Source choice and compatibility

Use [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3), its official FP8
checkpoint, pinned at `aca966e4e02791568aa6a4ced368624b3d897f42`.
It has 141 weight shards totaling755,632,050,320bytes. The BF16 variant has
282shards totaling1,506,667,387,408bytes; it is not the chosen representation.

Compared with the retained GLM-5.2-FP8 configuration, architecture fields and
normalized quantization configuration match. The declared Transformers version
changes from5.12.0 to5.15.0; the pinned local tokenizer passed preparation without
an environment upgrade.
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

The ordinary request path now pins GLM-5.3 and rejects old GLM-5.2 prepared
requests. The new template, configuration and license are retained separately
under `reference/hf-glm53/` and `licenses/GLM-5.3.txt`; the old snapshot is intact.
Omitting `--max-new-tokens` grants all remaining context slots in8K/32K mode.
For the four32K requests, thinking and the answer share that space at maximum
reasoning effort. EOS, context exhaustion and an explicitly requested shorter
output cap are reported separately. The legacy128K profile retains its separate
163,840-output-token ceiling.

The ordinary worker reads `configs/glm53-site.json`, binding the verified source
inventory and packed checkpoint digests. It also verifies the pinned source-
completion marker before opening devices. Missing or incomplete assets are
rejected. The site configuration is now installed; real-weight inference remains
the next acceptance step.

The ordinary runtime binds dense layers directly from its base owner shards and
prepares their resident BF16 representation. Its decoder configuration has
`strategy_nd_dense=False`; it does not call the legacy dense-overlay loader.
Therefore this migration needs the owner shards, not a separate legacy overlay
or a rebuilt PP8 checkpoint. This preserves the tested ordinary numerical path.

Distributed preparation uses the retained `pack_ws32_runtime_slots` implementation
on each host's four owner slots. The new receipt assembler reproduces the existing
checkpoint manifest without requiring all32files on one host. Tiny real-file CPU
tests compare it with the original complete packer, verify each host's local files
through the retained loader, and reject missing or inconsistent owner evidence.
This is preparation coverage; real GLM-5.3 packing and inference remain pending.

Local preparation of the unchanged four GSM8K questions with the actual pinned
tokenizer/template yielded prompt lengths100,62,85,70 and respective output
allowances32,668,32,706,32,683,32,698. Each prompt plus allowance equals32,768.
This is CPU input preparation, not four model answers. Private gold references
were not read by preparation or included in the model inputs.

After the verified GLM-5.3 runtime checkpoint binding is installed, the intended
acceptance command is:

```bash
JAX_PLATFORMS=cpu python -m glm_tpu ask --questions /private/questions.json \
  --context 32k --concurrent --wall-seconds 14400
```

There is no separate output or thinking cap in this command. Four hours is an
operational deadline, including cold startup; a timeout still means incomplete.
`--prepare-only` performs local preparation without a fleet launch. During
migration, worker admission requires the new verified GLM-5.3 checkpoint binding;
acquired source files alone do not make this inference command ready.

Acquire the pinned FP8 source into `models/GLM-5.3-FP8/` in the same bucket,
streaming without a full disk copy. Verify every byte count and upstream SHA256,
preserve headers/provenance, and publish a completion marker only after all
shards and metadata pass. Reconcile one completion notification, not polling.

Then rebuild only the required owner shards using the retained packing path,
install the verified site binding, and check affected CPU and
bounded real-weight inference behavior. No GLM-5.2 answer/speed receipt qualifies
GLM-5.3. Keep main's last release and the tagged snapshot intact until the new
implementation is usable. Diagnose and repair failures, preserving receipts and
verified progress, and retry under the latest owner authorization.
