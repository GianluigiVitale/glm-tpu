# GLM-5.3 migration history

**Final hardware outcome:** four correct completed answers, normal EOS, fresh
graph and memory checks, all-host agreement and eight-host cleanup passed at
`c2f60efe`, run `optimized_request_20260921T223026777871Z`.
[Current results](STATUS.md) · [Receipt](glm53-four-answers-20260921.json).
The sections below preserve intermediate states and superseded pending gates, as
they were written during the migration; module and command names are updated to
the current tree where a reader would look them up. The exact pre-release text is
`git show c2f60efe:docs/release/GLM53_MIGRATION.md`.

The migration froze GLM-5.2, removed its stored weights, and moved the same TPU
implementation to GLM-5.3. The completed GLM-5.2 code is preserved
at the tag `glm-5.2`, commit `edbced29315b6afce92cf994ae70785bf8f6d995`, with its
verified source archive. The completed migration retains the earlier failures and
recovery details below.

## Complete source and current preparation

All 141 FP8 shards, 755,632,050,320 bytes, completed verification on September 21
at 19:39:56 UTC. The pinned upstream SHA-256s, the cloud generations and the
completion-marker readback passed, along with eight-host cleanup. The private
evidence is kept with the migration's operational records. The initial
continuation stopped on a truncated HTTP body for shard 31; authenticated
byte-range recovery retained 38 verified shards and completed the remaining 103.
The original failed receipts and source generations remain preserved.

The first packing preflight rejected the new inventory because the retained
geometry parser unconditionally names GLM-5.2. No slot files were written;
eight-host cleanup passed and the completed inventory was preserved. The ordinary
packing and inference boundaries now use the pinned GLM-5.3 configuration and
model identity while keeping every numerical dimension and the unchanged parser
(`glm_tpu/config/model.py`). CPU checks cover this distinction. Packing then
completed on all 8 hosts at `90777a44`, producing 32 slot files (786,181,673,984
bytes). All local file hashes passed the checkpoint verifier; eight-host cleanup
and the regional backup and readback of the seal metadata passed. Inference and
answer validation were still pending at that point. The site configuration (then
the committed `configs/glm53-site.json`, now the untracked site file's
`[checkpoint]` table) binds these checkpoint and source identities.

The full CPU release check at `e5404830` passed 627 tests with one skip, plus
source, content and package checks. The geometry correction had separate affected
checks; passing CPU checks do not establish real-weight answers or speed.

## Acquisition recovery

Diagnosis, fixes and retries then continued autonomously, as authorized for
the release. The continuation preserved the 17 verified source generations and
acquired the 124 missing shards. Its transfer checks retain exact failure phases,
byte counts, hashes and traceback locations. Eight offline transfer and
failure-reporting cases passed. The original failed script and receipts stayed
unchanged in the private task folder. A resumed transfer is not inference
validation. The goal and authorization at the time:
`git show archive/research-20260922:goal.md`.

## Preserved initial acquisition failure

The dispatched acquisition exited unsuccessfully at 2026-09-21T14:16:26 UTC.
Shard `model-00015-of-00141.safetensors` raised `AssertionError`. The original
handler kept the file and the exception type but discarded the assertion text and
the traceback, so the precise failing check is unknown. A hash mismatch or a
network failure has not been established.

Seventeen of 141 shards, 91,184,236,216 bytes, were verified and preserved. Their
cloud generations, sizes and CRC32C values match the receipts, whose SHA-256s
match the pinned upstream LFS metadata. The failed shard was not published;
there was no `SOURCE_COMPLETE.json`. All eight cleanup logs identify distinct
authenticated idle hosts, and both workload leases were released.

The failed gate was **complete verified canonical source acquisition**. Packing
and GLM-5.3 inference had not started. The initial attempt stopped under the
then-current no-retry rule, later superseded by the recovery authorization
above. The original evidence and the partial source remain intact.
The failure receipt is preserved at the tag `archive/research-20260922`
(`git show archive/research-20260922:docs/release/glm53-acquisition-failure-20260921.json`).
The request integration at `fc95150d` had 69 passing affected CPU checks and a
passing offline package check; these do not qualify the missing weights.

## Source choice and compatibility

Use [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3), its official FP8
checkpoint, pinned at `aca966e4e02791568aa6a4ced368624b3d897f42`. It has 141
weight shards totaling 755,632,050,320 bytes. The BF16 variant has 282 shards
totaling 1,506,667,387,408 bytes; it is not the chosen representation.

Compared with the retained GLM-5.2-FP8 configuration, the architecture fields and
the normalized quantization configuration match. The declared Transformers
version changes from 5.12.0 to 5.15.0; the pinned local tokenizer passed
preparation without an environment upgrade. All 118,629 tensor-to-file mappings
match. The tokenizer JSON, the tokenizer configuration and the generation
configuration match, but the separate chat template changes. All 141 weight-file
SHA-256s differ from the GLM-5.2-FP8 source revision. These metadata checks
support reusing the implementation; they do not prove new-weight numerical
behaviour or answer correctness.

The new chat template adds reasoning-effort and tool-response behaviour. It must
be used and pinned separately before inference. GLM-5.3 uses its own
[license](https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE), not the old
MIT label; the file is kept with the model assets and the attribution is in the
[notices](../../THIRD_PARTY_NOTICES.md). Do not rewrite the frozen GLM-5.2 source
and evidence identities.

## GLM-5.2 weight retirement

An authorized deletion removed 359 inventoried cloud weight files totaling
1,706,606,624,800 bytes from the GLM-5.2 model and checkpoint prefixes of the
project bucket. Deleting 68 runtime, MTP and partial weight files reclaimed
812,943,126,528 bytes of host RAM filesystem capacity across the eight hosts.
All eight hosts passed authenticated idle checks before and after. No TPU
resource was changed.

Source inventories, configuration and tokenizer files, manifests, scientific
evidence, code, branches, licenses and Git history remain preserved. The exact
generation and inode inventories and the deletion receipts are private. Running
GLM-5.2 again requires restoring its external weights; the release tag does not
contain them. Historical receipts remain historical facts, not claims that
retired payloads are still present.

## Next execution boundary

The ordinary request path now pins GLM-5.3 and rejects old GLM-5.2 prepared
requests. The new template, configuration and license are kept separately under
`glm_tpu/models/glm_moe_dsa/hf_config/` (with its `LICENSE`); the old snapshot is
intact at the tags. Omitting `--max-new-tokens` grants all remaining context
slots in 8K and 32K mode. For the four 32K requests, thinking and the answer
share that space at maximum reasoning effort. EOS, context exhaustion and an
explicitly requested shorter output cap are reported separately. The 128K
profile keeps its separate 163,840-output-token ceiling.

The ordinary worker reads the controller-resolved site configuration staged with
each run (`site.json`, bound by `--site-sha256`; its `[checkpoint]` table is
documented in [examples/site.example.toml](../../examples/site.example.toml)),
binding the verified source inventory and the packed checkpoint digests. It also
verifies the pinned source-completion marker before opening devices. Missing or
incomplete assets are rejected. With the site configuration installed,
real-weight inference was the next acceptance step.

The ordinary runtime binds the dense layers directly from its base slot files
and prepares their resident BF16 representation; it does not call a dense-overlay
loader. The migration therefore needed the slot files, not a separate overlay
or a rebuilt checkpoint of the older pipeline layout, which keeps the tested
ordinary numerical path.

Distributed preparation uses `pack_runtime_slots`
(`glm_tpu/model_loader/sharded_state/writer.py`) on each host's four slots.
The receipt assembler (`assemble_owner_manifest`,
`glm_tpu/model_loader/sharded_state/manifest.py`) reproduces the checkpoint
manifest without needing all 32 files on one host. Tiny real-file CPU tests
compare it with the complete packer, verify each host's local files through the
verifier and loader, and reject missing or inconsistent per-slot evidence. This was
preparation coverage; real GLM-5.3 packing and inference were still pending.

Local preparation of the unchanged four GSM8K questions with the actual pinned
tokenizer and template yielded prompt lengths 100, 62, 85 and 70 and output
allowances 32,668, 32,706, 32,683 and 32,698. Each prompt plus its allowance
equals 32,768. This is CPU input preparation, not four model answers. Private
gold references were not read by the preparation or included in the model inputs.

After the verified GLM-5.3 runtime checkpoint binding was installed, the intended
acceptance command was:

```bash
JAX_PLATFORMS=cpu python -m glm_tpu ask --questions /path/to/questions.json \
  --context 32k --concurrent --wall-seconds 14400
```

There is no separate output or thinking cap in this command. Four hours is an
operational deadline, including the cold start; a timeout still means
incomplete. `--prepare-only` performs local preparation without a fleet launch.
Worker admission requires the verified GLM-5.3 checkpoint binding; acquired
source files alone do not make this inference command ready.

The acquisition streamed the pinned FP8 source into the project bucket without a
full disk copy, verified every byte count and upstream SHA-256, preserved headers
and provenance, and published a completion marker only after all shards and
metadata passed. The required slot files were then rebuilt with the packing
path, the verified site binding installed, and the affected CPU and bounded
real-weight inference behaviour checked. No GLM-5.2 answer or speed receipt
qualifies GLM-5.3.

## Resident release follow-up, 2026-09-22

The original four-chat release remains preserved at `e7898dff` and its receipts.
Executable `5c3c1d6b` added optional residency, a private sequential inbox and a
32K ordinary default. The solo run measured 13.57 tokens/s; its same loaded model
then processed GSM8K test rows 0–769, with 740 correct completed finals and three
context-exhausted attempts among the 30 unsuccessful cases. The remaining 549
test questions were not run. This partial evaluation supersedes the four-example
check as the larger quality sample; it does not replace the separate concurrency
evidence or constitute a full GSM8K score. No answers were rerun to improve it.
See the [current status](STATUS.md) and the [receipt](glm53-resident-results-20260922.json).

The scorer and the completion notifier were stopped, the remaining inbox entries
kept outside the active queue, and all eight model workers authenticated live.
The original private inputs, references, outputs, unrun inputs and per-row
receipts remain preserved with the operational records. Neither prompts nor raw
outputs are in the repository. This work was reviewed only by the AI coding
agent that built it, not by independent human review.
