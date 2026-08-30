# Gate D deep-research handoff: exact one-row DSA root cause and legal fix

**Status date:** 2026-08-30
**Repository:** `/home/gianl/glm-tpu-topology-rewrite`
**Branch:** `rewrite/topology-first-decode`
**Target:** `zai-org/GLM-5.2-FP8` on the existing 32-chip TPU-v4 pod
**Audience:** Fable 5 Max deep-research reviewer
**Gate status:** **OPEN**

## 1. TL;DR

Gate D is approximately **75% complete only as rough planning shorthand**, but it is binary and has
not passed. The percentage has no measured denominator: it is not completion evidence, confidence
or an ETA. Most architecture, checkpoint, runtime, HLO, memory, trace, state/cache, token and
observability infrastructure exists. The shorthand “remaining 25%” contains the decisive
correctness work:

1. explain the first cutoff-active DSA divergence physically, not just correlate it;
2. define a genuinely new legal one-row, topology-local mechanism upstream of that divergence;
3. falsify it with one coherent state before compilation;
4. preserve the accepted BF16 recurrent arithmetic and exact DSA set/tie order on TPU;
5. pass one complete protected cutoff-active short decoder with fresh HLO, HBM, XPlane, wall,
   database, archive and clean-fleet evidence.

The first known downstream mismatch is at layer 1, position 8,155, normalized hidden element 2,795:
accepted BF16 bits `27bd`, PP16 candidate `26bd`. Query/head/current-key/scorer drift follows. A
DEFAULT-precision scorer is exact when given accepted inputs, so the scorer is not the root cause.
The missing observation is the accepted, unperturbed FP32 row entering layer-1 RMSNorm. Existing
callbacks changed the executable and repeated the bad result, so their FP32 bytes are not an oracle.

We have eliminated many attractive but false solutions: callbacks, dead M32 rows, full-pod state,
mixed accepted/candidate caches, tolerance relaxation, direct persistent FP32 recurrence, split-K
qkv-a, gather-before-weight/output ownership, scalar correction, Pallas/direct/source-fused/layout
variants and unchanged retries. Two abstract device-only shadow forms remain, but neither currently
has source/plan/StableHLO/coherent-state authority. No compilation or TPU run is authorized.

## 2. What Gate D actually requires

Gate D is the complete short-context decoder gate. A pass requires all of the following in one
protected run:

- complete 78-layer decoder execution;
- correct raw generated tokens;
- cutoff-active exact DSA selected sets and oracle-relative tie order;
- exact state, checkpoint-load and cache-write protections;
- no repeated 32-chip transformer-layer collective;
- one real decode row, with no batch-32 diagnostic/dead rows;
- fresh multi-host optimized-HLO and XPlane evidence;
- measured per-chip HBM headroom;
- profiler-free steady wall measurement;
- exact code, plan, checkpoint and executable identities;
- `bench/results.db` linkage, same-region archive and authenticated 8/8 zero-work cleanup.

CPU equality, synthetic StableHLO, a compile-only acquisition, one exact layer, correct tokens with
vacuous DSA selection, or a callback-perturbed observer cannot close the gate.

## 3. What “75% complete” means

This percentage is rough planning shorthand, not a benchmark metric, confidence level, earned-value
measure or ETA. Gate D remains 0% passed until every closure condition succeeds together.

### Largely complete

- Gates A and B infrastructure: topology-aware PP8/PP16 execution, local groups, plan-aware final
  layouts, packing, direct loading and integrity evidence.
- Complete decoder execution exists for short contexts.
- Local HLO, HBM, XPlane, wall, DB/archive and cleanup machinery has been exercised.
- Dense, cache, query, key, scorer and many DSA boundaries have exact isolated evidence.
- Immutable typed observability, causal-frontier classification, source/HLO inspection and offline
  mechanism admission exist.
- Historical mechanisms are fingerprinted and tombstoned, preventing expensive repetitions.

### Still missing

- accepted unperturbed FP32 layer-1 RMS-input bytes at the causal frontier;
- a proven physical explanation for the one-bit BF16 normalized-state difference;
- a legal source-bound one-row mechanism that changes physical dependency without changing accepted
  BF16 primary arithmetic;
- a candidate-coherent state capsule for that mechanism;
- optimized TPU-HLO confirmation of the intended physical value flow;
- exact cutoff-active event-1 DSA behavior from the candidate's own state;
- a complete protected short-decoder pass.

Because the missing work is the semantic closure criterion, schedule risk is higher than “25% of
elapsed time” suggests.

## 4. Current strongest evidence

### 4.1 Protected 2K result is useful but does not close strict Gate D

DB563 has exact tokens, state/cache, local HLO, HBM, fresh XPlanes, wall, DB/archive and cleanup at
`245.639880 ms/token` p50 and `4.071000 tok/s`. However `context_capacity == dsa_top_k == 2048`, so
selected-set equality is cutoff-vacuous. Its historical comparator treated cross-backend total
order and bounded scores as diagnostic; all 14 oracle total-order/score-bound diagnostics fail.
Therefore it cannot prove strict cutoff-active DSA ranking.

Compact authority:
`docs/artifacts/db563-gate-d-strict-reclassification.json`.

### 4.2 Cutoff-active 8K localizes the real blocker

- PP8 and PP16 can produce correct early tokens and state boundaries but diverge at layer 1/event 1.
- DB518 repairs the entire 8,155-row layer-0 prompt index cache and makes position 113 exact.
- DB518 still has one layer-1 normalized BF16 miss at hidden 2,795 before q-a/query/head/event-1
  divergence.
- WS32 produces 20/20 correct tokens and exact event 0, then differs by seven layer-1/event-1
  selected-position swaps.
- The same downstream symptom across structurally different plans points upstream of the scorer.

### 4.3 Downstream DSA mechanisms have been isolated

- DB503/504 establish the production q-a packing/association.
- DB505--518 isolate and repair prompt index-cache construction.
- DB525/526 establish the grouped query association.
- DB527 establishes current-key owner/divide-sqrt behavior.
- DB529 shows the DEFAULT-precision scorer matches accepted logical scores, selected set, order and
  ties when supplied accepted inputs.

Conclusion: another scorer/top-k tweak is downstream and cannot move the first causal frontier.

### 4.4 Accepted HLO narrows the semantic contract

The accepted DB485 consumer receives an explicit `bf16[32,6144]` weighted row, bitcasts BF16,
decodes N82 weights to BF16 and accumulates convolution in FP32. No pre-round FP32 RMS operand
crosses that qkv-a boundary. Accepted recurrence is logically:

```text
FP32(BF16 dense update) + FP32(BF16 carried residual)
    -> BF16 rounded recurrent state
    -> FP32 RMS calculation / BF16 normalized consumer boundary
```

Directly replacing recurrent BF16 state with an unrounded FP32 shadow changes model semantics and
is source-rejected. Logical BF16 HLO edges do not prove where TPU physically materializes rounds,
copies or tuple dependencies.

### 4.5 Exact observability frontier

The immutable auditor reports:

```text
OBSERVABILITY_GAP;GATE_D_OPEN;NO_TPU_SUCCESSOR
```

Both accepted and candidate authorities lack the unperturbed FP32 row entering layer-1 RMSNorm.
The first observable downstream mismatch is normalized BF16 hidden 2,795. Candidate current-key and
accepted cache/event-1 state are also not jointly present in the same old capsule. This proves an
evidence frontier, not the root cause.

Authorities:

- `configs/greenfield-gate-d-observability.json`
- `docs/artifacts/gate-d-observability-frontier.json`
- `glm_tpu/greenfield/observability.py`
- `scripts/greenfield/audit_observability.py`

## 5. Exact remaining blockers

### 5.1 Resolve the missing physical state

We need either:

1. an unchanged-executable way to expose the accepted FP32 layer-1 RMS input; or
2. a deterministic physical/mechanical proof that makes the missing runtime bytes unnecessary.

Any observation must preserve executable identity, raw outputs and DSA events. A host callback,
print, extra rooted output or consumer that changes optimized HLO is not observationally neutral.

### 5.2 Define a legal mechanism at the correct frontier

The mechanism must:

- have one logical row;
- remain inside one physical LP2 or LP4 group;
- avoid full-pod hidden reconstruction;
- add no host/Ray/Python stage dispatch;
- preserve the accepted BF16 primary recurrence and output bits;
- introduce a real device-side causal dependency, not an unused value or renamed layout;
- act at `layer1.rms_input_fp32`, before normalized/query/head divergence;
- provide its own complete cache/query/head/current-key/scorer state.

### 5.3 Construct candidate-coherent authority

Before compilation, one candidate identity must bind:

- committed source blob and distinct accepted/candidate AST symbols;
- exact mechanism fingerprint and normal form;
- complete 32-rank plan-local group map, layouts and physical owners;
- causal parser-verified StableHLO whose primary backward slice equals the accepted BF16 recurrence;
- one immutable state capsule at layer 1, position 8,155 containing:
  - FP32 RMS input `[6144]`;
  - two BF16 normalized owner rows `[6144]`;
  - BF16 cache history `[2,16,256,128]`;
  - two FP32 query owner slices `[32,128]`;
  - two FP32 head-weight slices `[32]`;
  - FP32 current key `[128]`;
  - event-1 positions/scores `[1,2048]` and valid count `[1]`;
- exact raw selected-slice SHAs with no slice reuse or mixed authority.

### 5.4 Prove optimized TPU lowering

An offline admission does not prove TPU behavior. One separately reviewed compile-only acquisition
must show that optimized HLO preserves:

- the exact BF16 primary value flow;
- the intended auxiliary dependency;
- only declared LP2/LP4 physical groups;
- one row and no M32/dead diagnostic lanes;
- no callback/custom-call/host effect;
- no optimizer collapse back to a tombstoned boundary.

Only after that can the smallest numerical event-1 replay be reviewed.

### 5.5 Close with the complete decoder

If the bounded event-1 replay is exact, integrate it default-off and run the protected cutoff-active
short decoder. Only the full evidence bundle closes Gate D.

## 6. Two unresolved abstract mechanisms

These are declarations, not implementations or authorization.

### 6.1 `auxiliary_device_tuple_dependency`

Carry the accepted BF16 primary state unchanged while an FP32 shadow remains live through a
device-only auxiliary result/dependency. The auxiliary must cause a real physical distinction while
remaining semantically isolated from primary arithmetic.

Open questions:

- Can StableHLO/JAX express such a dependency without an extra rooted output changing codegen?
- Can an auxiliary survive optimization without a custom call, callback, dead row or full-pod
  collective?
- Can its physical liveness influence the RMS/weighted-output lowering while the primary bits remain
  exactly accepted?

### 6.2 `compensated_auxiliary_dependency`

Keep a compensated FP32 quantity live only through a device consumer while cancellation preserves
the BF16 primary recurrence. A mere add/subtract spelling is insufficient: the compensation must
be algebraically and physically causal, exactly cancelled for primary outputs, and distinct from
direct unrounded-state substitution.

Open questions:

- Is exact cancellation guaranteed under TPU FP32 association and compiler reassociation rules?
- What StableHLO structure prevents reassociation or dead-code elimination without changing primary
  semantics?
- Is it genuinely distinct after optimized TPU HLO, or does it collapse to the tuple form or an
  already-rejected layout/fusion family?

The staged, blocked pre-fix v2 baseline reports this result for both:

```text
MISSING_SOURCE_AST_AUTHORITY
MISSING_PLAN_AUTHORITY
MISSING_CAUSAL_STABLEHLO_AUTHORITY
MISSING_CANDIDATE_COHERENT_CAPSULE
```

That report and its code are staged but uncommitted, and its correction-only Sol review returned
**BLOCK**. The live unstaged/untracked candidate-fix bytes differ from that staged baseline. They
must never be mixed into one authority. The pre-fix batch had passed 35/35 focused and 104/104
adjacent CPU-forced offline tests, but those results do not validate the live fixes. The current
working tree has not completed focused, adjacent or adversarial re-verification. Neither the staged
baseline nor the live candidate fixes authorize compilation or TPU work. Only a replacement report
that is reviewed, committed and reproduced byte-for-byte can supersede the blocked baseline.

### 6.3 Admission-proof blockers found by adversarial review

These are tool/evidence blockers in addition to the model root cause. The current working tree has
candidate fixes for all six, but they remain open until hostile tests and correction-only review
pass. Fable should distinguish these proof-system defects from the unresolved numerical mechanism:

1. **Parser supply chain:** hashing the validator Python and version string did not seal imported
   jaxlib/MLIR Python modules and native libraries under mutable `site-packages`, nor eliminate a
   same-version replacement/import race. Candidate fix: exact file manifest plus a private verified
   parser tree opened through retained directory descriptors.
2. **Source-to-HLO binding:** authenticated source AST and StableHLO could still be independent; a
   rehashed source symbol could reuse synthetic HLO. Candidate fix: derive one accepted authority
   and candidate semantic digest from exact AST normal forms and bind the allowed HLO slice to it.
3. **RMS-frontier identity:** a certificate could select an arbitrary operation/result as the
   auxiliary source. Candidate fix: derive the FP32 frontier from the operand immediately preceding
   the primary BF16 conversion instead of accepting a certificate-selected index.
4. **Mechanism identity:** the compensated survivor was recognized by the presence of an `add` and
   `subtract`, not an exact cancellation graph. Candidate fix: candidate-specific canonical
   auxiliary backward-slice digests and hostile non-cancelling/tuple-substitution tests.
5. **Physical locality:** a content-derived micro-plan could call physically nonlocal ranks local.
   Candidate fix: bind exact PP8/PP16 groups to the protected runtime-topology authority and require
   full 32-rank coverage with the expected topology and plan hashes.
6. **Owner-axis identity:** `owner0`/`owner1` roles did not bind NPZ prefixes or the cache owner axis
   to physical owners. Candidate fix: exact prefixes, owner-axis slots and owner-id maps, with swap
   and reversal attacks.

The candidate fixes are not themselves a model solution. Even if all six pass, the two mechanisms
still need real source, plan, causal StableHLO and candidate-coherent state authorities before one
can be admitted. Research should focus on that remaining numerical/physical mechanism, while also
challenging whether the candidate proof fixes are sufficient.

## 7. Problems encountered and what fixed them

### 7.1 Correct tokens hid invalid DSA proof

**Problem:** At 2K, `context <= top_k`; selecting all valid positions makes set equality vacuous.
Candidate-internal tie checks do not prove oracle-relative ranking.
**Fix:** Reclassify DB563; require cutoff-active contexts and exact oracle sets/ties/scores.

### 7.2 Downstream fixes were mistaken for causal fixes

**Problem:** q-a, cache, query, key or scorer changes could improve a downstream boundary while the
first normalized-state bit was already wrong.
**Fix:** Ordered typed watchpoints and first-divergence bisection. A candidate must move the first
missing/divergent watchpoint, not merely change the final DSA set.

### 7.3 Mixed-state replay created incoherent counterfactuals

**Problem:** Combining accepted query/current rows with DB518 cache history can produce an attractive
answer that no executable ever generated.
**Fix:** One coherence id and one code/source/plan/HLO identity for RMS, normalized, cache, query,
head, key and scorer state. Missing state is a refusal, not permission to borrow it.

### 7.4 Observers changed the program

**Problem:** Callbacks, rooted outputs and host consumers changed executable class and repeated the
rejected behavior. The observed value was therefore not accepted authority.
**Fix:** Compare optimized-HLO/executable identity before trusting a capture. Prefer device buffers
plus one bounded post-run transfer. Tombstone callback-derived FP32 bytes.

### 7.5 Logical HLO was confused with physical materialization

**Problem:** A logical BF16 edge, tuple or copy does not prove where TPU rounds or retains FP32 bits.
StableHLO can optimize to the same rejected physical graph.
**Fix:** Preserve source, StableHLO and optimized TPU HLO separately; inspect live SSA producer-to-
consumer flow, layouts, groups, reducers and roots.

### 7.6 Candidate names disguised duplicate mechanisms

**Problem:** Repackaging gather-before-weight, scalar correction or another tree under a new name
caused repeated expensive work.
**Fix:** Canonical five-field fingerprints: association, consumer boundary, reduction,
representation and transport. Maintain an append-only tombstone catalogue.

### 7.7 Old artifacts were relabeled as new candidate evidence

**Problem:** DB518 arrays belong to DB518's code/plan/HLO and omit RMS input/current key. Callback
arrays belong to a perturbed executable.
**Fix:** Raw-array SHA, slice, dtype, shape, role, owner, layout and authority-tuple binding. Audit
capsule constructability before writing candidate code.

### 7.8 Text parsing could forge HLO causality

**Problem:** Regexes can match comments, string attributes, fake calls, constants or non-causal
operation names. Cardinality-only checks accept a ring across the pod.
**Partial/pending fix:** Pinned jaxlib MLIR parsing, actual SSA backward slices and an exact
accepted-primary slice digest replaced regex-only reasoning. The live working tree additionally
attempts imported-parser sealing, exact auxiliary causality and collective components constrained
to protected plan-local groups; those additions remain unverified and are not authority.

### 7.9 Ambient Git could forge source authority

**Problem:** `PATH`, `GIT_DIR`, object directories or replace objects can change what a commit/path
query returns.
**Fix:** SHA-pin `/usr/bin/git`, execute its open fd, scrub `GIT_*`, use `--no-replace-objects`, retain
one repository dirfd, require regular committed blobs and bind tree object id to exact bytes/AST.

### 7.10 Long TPU runs were used before cheap falsification

**Problem:** Compile/model-load/full-decoder attempts consumed hours for hypotheses rejectable in
seconds.
**Fix:** smallest-test ladder: metadata -> source/AST -> coherent NPZ -> parsed StableHLO ->
compile-only HLO -> bounded event replay -> full decoder. Stop at the first failed authority.

### 7.11 Evidence could be overwritten, redirected or partially archived

**Problem:** Symlinks, compression bombs, duplicate JSON/NPZ members, concurrent writers and partial
cleanup can make a plausible report unauthoritative.
**Fix:** no-follow dirfd traversal, bounded parsing/decompression, exact schemas, raw hashes,
append-only `O_EXCL` writes, fsync, inode-aware failure cleanup, DB/archive receipts and 8/8 census.

## 8. Methodology that unlocked progress

1. **Start with the oracle and inventory.** Register every accepted, candidate and negative artifact
   before designing a run.
2. **Define a causal watchpoint order.** Observe residual/RMS input, normalized bits, cache, query,
   head, key and scorer rather than only final tokens.
3. **Bisect the first divergence.** Move upstream until the first missing or differing raw bit; never
   optimize downstream first.
4. **Record exact semantics.** Include dtype, shape, bit pattern, position, layer, layout, physical
   owners and reduction association.
5. **Require coherent state.** Every replay uses state produced by one real candidate identity.
6. **Separate source, StableHLO, optimized HLO and runtime truth.** Each answers a different question.
7. **Falsify offline first.** A failed metadata/source/state/parser gate prevents JAX/model/TPU work.
8. **Attack the proof.** Mutate SHAs, ASTs, modes, groups, comments, strings, dtypes, shapes, owners,
   slices, archives, symlinks and writer timing.
9. **Protect expensive runs.** Preflight locks/tags, preserve per-host evidence, use fresh traces,
   measure profiler-free wall and authenticate cleanup.
10. **Tombstone failures by mechanism fingerprint.** Reuse negative evidence instead of retrying the
    same arithmetic under a new label.
11. **Keep claims narrow.** “Parser-valid”, “compile-only” and “one exact boundary” never become
    “Gate D passed”.
12. **Review before escalation.** Adversarial review precedes compilation and again precedes a
    numerical TPU run.

This methodology changed the project from repeated full-decoder experimentation into a fail-closed
causal search. It did not yet produce the final mechanism, but it removed most false search space.

## 9. Tombstoned proposals: do not recommend without a provably new mechanism

- unchanged PP8, PP16 or WS32 8K retry;
- any `context <= top_k` result as ranking proof;
- host callback, print, Python/Ray dispatch or observer-rooted output;
- batch-32/M32 dead rows or accepted 32-lane diagnostic geometry;
- full-pod hidden reconstruction or repeated 32-chip layer collective;
- tolerance relaxation, approximate top-k or changed tie policy;
- mixing accepted and candidate cache/query/head/key/scorer state;
- direct persistent unrounded FP32 recurrent state;
- rounded-then-widened shadow with no physical distinction;
- unused/no-consumer shadow;
- gather-before-weight, output-ownership or non-rooted direct fusion;
- scalar/global correction, coordinate patch or another uniform tree;
- BF16-origin `wk`, q-a-round, direct/source-fused/Pallas/layout-only families;
- PP16 split-K/K-half N82 qkv-a arm;
- another scorer precision/top-k family downstream of the known normalized-state miss;
- CPU HLO or synthetic-kernel success presented as TPU numerical proof.

If research revisits an item, it must identify the exact new physical association/consumer/
reduction/representation/transport fingerprint and explain why sealed evidence does not apply.

## 10. High-value deep-research questions

Please search papers, compiler documentation, JAX/XLA/StableHLO issues and TPU architecture sources
for mechanisms—not generic inference advice—addressing these questions:

1. How can an XLA/StableHLO program keep an auxiliary FP32 value physically live on TPU while
   guaranteeing a bit-identical BF16 primary recurrence and no host/custom-call effect?
2. Are there documented device-only token/control/data dependencies, aliasing, donation, tuple,
   barrier or side-effect-free constructs that survive optimization and can be verified in optimized
   HLO?
3. How can one express exact cancellation that XLA cannot reassociate away, while proving primary
   BF16 output identity?
4. Can reproducible/reassociation-controlled reductions reproduce a 32-partial accepted result from
   a two- or four-chip local group without dead rows or full-pod communication?
5. Can a local correction term be derived from exact BF16 partials so the final BF16 row matches the
   accepted reduction for every input, not merely the captured prompt?
6. What TPU-v4 layout, accumulator and conversion behavior could explain a one-bit normalized BF16
   difference despite logically identical BF16 association?
7. What techniques prove whether a BF16 conversion is physically materialized versus retained as
   an FP32 producer-consumer fusion across RMSNorm and qkv-a?
8. Are there MLIR/XLA tools or formal methods for comparing floating-point SSA graphs under exact
   rounding/association constraints?
9. Can Pallas express the accepted reduction/rounding boundary explicitly with only LP2/LP4
   communication and a single live row?
10. Is there a plan-local residual/RMS ownership scheme that avoids reconstruction but preserves the
    accepted exact bit pattern through layer boundaries?
11. How should DSA top-k equivalence be proved under cutoff-active conditions, including set,
    total-order, ties, sentinels and IndexShare reuse?
12. What is the smallest decisive TPU experiment for each serious proposal, and what exact HLO/raw
    bit observation would falsify it before a full decoder run?

## 11. Constraints for any proposed solution

Reject a proposal unless it explicitly satisfies all of these:

- native JAX/StableHLO/Pallas path independent of legacy execution;
- one live decode row;
- exact accepted BF16 recurrence and exact DSA sets/ties;
- physical LP2 or LP4 repeated communication only;
- no host callback, Python dispatch, Ray transfer or host-staged critical path;
- no dead diagnostic rows and no 32-chip hidden reconstruction;
- candidate-coherent cache/query/head/key/scorer state;
- source, plan, StableHLO and optimized-HLO identities can be sealed;
- default-off implementation and a bit-exact fallback;
- a seconds/minutes falsification test exists before any hours-long run;
- full protected run can produce HBM/HLO/XPlane/wall/DB/archive/cleanup evidence.

Performance ideas are secondary until exact cutoff-active Gate D correctness passes.

## 12. Requested Fable output

Return a research report with:

1. a ranked list of **three to five genuinely distinct mechanisms**;
2. primary-source citations and direct links for every compiler/hardware claim;
3. an exact source/StableHLO sketch for each mechanism;
4. why it preserves BF16 primary bits and remains one-row/local;
5. how it differs from every tombstoned family;
6. expected optimized TPU-HLO signature and prohibited signature;
7. the smallest offline test, compile-only test and bounded numerical test;
8. explicit falsification criteria;
9. risk of compiler collapse, reassociation, observer perturbation or mixed authority;
10. a final recommendation: implement, investigate further, or reject.

Do not recommend a full decoder run first. Do not infer local performance from papers. Clearly mark
inference versus directly documented behavior.

## 13. Local evidence and tools to read first

### Governing and status

- `goal.md`
- `docs/glm-tpu-revolution.md`
- `HANDOFF.md`
- `docs/greenfield/REUSE_INVENTORY.md`
- `docs/greenfield/EVIDENCE_MAP.md`
- `docs/greenfield/GATE_D_LESSONS.md`
- `docs/greenfield/GATE_D_OBSERVABILITY_PLAYBOOK.md`

### Core compact artifacts

- `docs/artifacts/db563-gate-d-strict-reclassification.json`
- `docs/artifacts/gate-d-observability-frontier.json`
- `docs/artifacts/gate-d-mechanism-admission-frontier.json`
- `docs/artifacts/plan-local-persistent-fp32-shadow-source-rejection.json`
- `docs/artifacts/gate-d-shadow-variant-adjudication.json`
- `docs/artifacts/gate-d-capsule-constructability.json`
- `docs/artifacts/gate-d-precompile-admission-v2.json`
- `docs/artifacts/db485-layer1-rms-hlo-causality.json`
- `docs/artifacts/pp16-feature2-qkv-khalf-event1-cpu-rejection.json`

### Relevant implementation

- `glm_tpu/greenfield/observability.py`
- `glm_tpu/greenfield/gate_d_admission.py`
- `glm_tpu/greenfield/gate_d_precompile_admission.py`
- `scripts/greenfield/validate_gate_d_stablehlo.py`
- `glm_tpu/greenfield/benchmarking/live_ssa_diff.py`
- `glm_tpu/greenfield/sharding/hlo_contract.py`
- `glm_tpu/greenfield/validation/legacy_dsa_internals.py`
- `glm_tpu/greenfield/validation/legacy_main_cache.py`
- `glm_tpu/greenfield/validation/legacy_residuals.py`
- `glm_tpu/greenfield/validation/prompt_index_cache.py`
- `bench/results.db`

## 14. Bottom line

The project is not stuck because it lacks another large test. It is stuck because exact BF16
physical association at one upstream layer boundary is unresolved and most intuitive attempts
either change semantics, perturb codegen, use illegal topology, or lack coherent state. The useful
research contribution is a **new, mechanically precise, one-row local dependency/reduction
mechanism with a cheap falsification path**. Once that exists, the existing observability and
protection stack can test it without repeating the previous two weeks of expensive failures.
