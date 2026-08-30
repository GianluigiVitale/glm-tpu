# Gate D Observability and Root-Cause Playbook

This is the end-to-end operating guide for making GLM-5.2 Gate D failures visible, proving the
first causal divergence, admitting only genuinely new mechanisms, and converting one protected run
into durable evidence. It consolidates the useful tools, mistakes and non-repeat rules accumulated
across the campaign. It does not replace `goal.md`, `docs/glm-tpu-revolution.md`, protected
artifacts or the exact mutable status in `HANDOFF.md`.

## 1. What observability means here

Observability is not more logging. It is the ability to answer, from immutable evidence:

1. Which exact code, plan, executable and checkpoint produced a value?
2. Which logical watchpoint, layer, position, dtype, shape, layout and physical owner did it belong
   to?
3. Which bytes were present before and after the first causal boundary?
4. Did all compared values come from one coherent candidate execution and cache history?
5. Did instrumentation preserve executable identity, output tokens and exact DSA behavior?
6. Did the physical HLO perform the declared local communication and no hidden global work?
7. Can the result be reproduced, queried, archived and tied to a clean eight-host lifecycle?

A print, label, tensor name, CPU calculation or plausible HLO substring answers only a fragment of
those questions. Gate D needs the complete chain.

## 2. Current proven frontier

Gate D is open. The immutable accepted/DB518 comparison currently establishes:

- neither authority preserves the unperturbed FP32 row entering layer-1 RMSNorm;
- the first observable downstream mismatch is normalized BF16 hidden index 2795: accepted raw bits
  `27bd`, DB518 `26bd`;
- q-a, query, head weights and event-1 DSA diverge later;
- candidate current-key and accepted cache/event-1 evidence are incomplete for a cross-authority
  replay;
- logical accepted/DB518 HLO association agrees at the BF16 boundaries, but that does not prove
  physical BF16 materialization across DB518's FP32 tuple/copy boundary;
- all catalogued mechanisms currently fail offline admission; this is a finite-catalogue result,
  not a proof that no legal mechanism exists;
- no compilation or TPU successor is authorized by the observability/admission evidence.

The required coherent candidate watchpoints are:

1. `layer1.rms_input_fp32`
2. `layer1.normalized`
3. `layer1.cache_history`
4. `layer1.query`
5. `layer1.head_weights`
6. `layer1.current_key`
7. `layer1.scorer_event1`

The next mechanism must expose or deterministically resolve the first item and bind all seven to
one code/plan/executable/coherence authority.

## 3. The observability stack

### 3.1 Operational black box

`docs/10-observability.md` records the legacy pod-run patterns worth preserving conceptually:

- per-worker append-only flight records survive crashes and distinguish worker death from recorder
  death;
- crash triage aligns host, timestamp, step composition, scheduling lag and compile activity;
- periodic engine statistics expose stalls and bucket churn without a profiler;
- sharding round-trip and cache-sanity guards fail loudly at the first dropped stripe;
- recorder overhead, rotation, failure behavior and scheduling attribution are tested explicitly.

The greenfield engine must not import the legacy execution path. Reuse the pattern: bounded
device/host state, explicit semantics, append-only output, failure-visible telemetry and an offline
reader.

### 3.2 Immutable numerical watchpoints

Core: `glm_tpu/greenfield/observability.py`

Contract: `configs/greenfield-gate-d-observability.json`

CLI: `scripts/greenfield/audit_observability.py`

Hostile tests: `tests/greenfield/validation/test_observability.py`

This stdlib-only path:

- opens bounded regular files, snapshots and SHA-authenticates them before parsing;
- parses NPY/NPZ without importing JAX or executing a model;
- rejects duplicate members, unsupported/Fortran dtypes, wrong shapes, wrong byte counts and raw
  array SHA drift;
- binds each source to role, trust, observation method, code pin, plan authority, executable
  identity and coherence id;
- verifies those identities against sealed JSON evidence;
- orders typed watchpoints and reports the first unobservable point or first raw-bit divergence;
- refuses ambiguous accepted/candidate authority and mixed candidate coherence;
- writes canonical append-only reports and refuses occupied or symlinked paths.

Reproduction pattern:

```bash
/home/gianl/vllm-env/bin/python -S \
  scripts/greenfield/audit_observability.py \
  --contract configs/greenfield-gate-d-observability.json \
  --contract-sha256 <exact-sha256> \
  --output <new-append-only-path>
```

`python -S` is intentional: the audit must not discover or initialize JAX/TPU backends.

### 3.3 Mechanism admission before JAX

Core: `glm_tpu/greenfield/gate_d_admission.py`

Contract: `configs/greenfield-gate-d-mechanism-admission.json`

CLI: `scripts/greenfield/admit_gate_d_mechanisms.py`

Hostile tests: `tests/greenfield/validation/test_gate_d_admission.py`

Admission rejects a candidate before compilation when it:

- is not a true one-row mechanism;
- uses a collective group larger than four;
- reconstructs hidden state across the full pod;
- adds a host effect or stage dispatch;
- acts only downstream of the first missing watchpoint;
- matches a sealed five-field mechanism fingerprint;
- lacks a complete typed candidate observability contract;
- fails to bind common code, plan, executable and coherence identity;
- lacks SHA-bound layout and owner evidence;
- traverses a symlinked input/output parent or targets an occupied report path.

The five fingerprint fields are `association`, `consumer_boundary`, `reduction`, `representation`
and `transport`. They prevent a closed mechanism from becoming “new” by changing its id or setting
`family_id` to null.

Even `OFFLINE_CANDIDATE_ADMITTED` always leaves `tpu_successor_authorized=false`. Compilation and
execution require separate review and evidence.

### 3.4 Domain-specific state readers

Use the existing readers before creating another capture format:

- `validation/legacy_dsa_internals.py`: exact DSA internal tensors and event alignment;
- `validation/legacy_main_cache.py`: cache values/ownership at the accepted boundary;
- `validation/legacy_residuals.py`: layer residual provenance and raw-bit comparisons;
- `validation/prompt_index_cache.py`: prompt-index cache construction and current-slot handling;
- `validation/short_context_dsa_oracle.py`: sealed cutoff-active DSA authority;
- `benchmarking/live_ssa_diff.py`: live optimized-HLO SSA/value-flow differences;
- `sharding/hlo_contract.py`: collective types, groups, shapes and repeated-region constraints.

Locate current implementations with:

```bash
rg --files glm_tpu/greenfield scripts/greenfield tests/greenfield | \
  rg '(observ|capture|compare|inspect|probe|trace|hlo|dsa|cache|residual|admission)'
```

The authoritative reuse/evidence indexes are:

- `docs/greenfield/REUSE_INVENTORY.md`
- `configs/greenfield-reuse-inventory.json`
- `docs/greenfield/EVIDENCE_MAP.md`
- `docs/greenfield/GATE_D_LESSONS.md`
- `docs/artifacts/`
- `HANDOFF.md`
- `bench/results.db`

## 4. End-to-end debugging workflow

### Phase 0 — establish authority and a clean boundary

1. Read `goal.md`, the full rewrite specification, this playbook and current handoff.
2. Verify the exact worktree, branch, local/origin pins and all repository pins used by the run.
3. Verify the workload lock and an idle fleet. Do not initialize JAX during an offline check.
4. Identify the accepted oracle, candidate, plan manifest, checkpoint manifest and executable/HLO
   identity that would be compared.
5. Search the reuse inventory and tombstones before writing code or spending TPU time.

Stop if any authority, checksum, owner, cache predecessor or executable identity is unknown.

### Phase 1 — ask one falsifiable question

Write the hypothesis as:

```text
At watchpoint X, mechanism Y changes physical association Z while preserving A/B/C.
If true, exact bytes or exact DSA event E will match; if false, mismatch M will remain.
```

Name:

- the first watchpoint affected;
- inputs and outputs;
- logical and physical owners;
- dtype/shape/layout;
- legal local collective groups;
- the smallest input that can falsify it;
- forbidden host effects, dead rows and global reconstruction;
- success and rejection artifacts.

If the question begins downstream of the current frontier, it cannot establish root cause.

### Phase 2 — build observability before the mechanism

Define typed immutable watchpoints first. For every array bind:

- source artifact path and SHA;
- NPZ key and index prefix;
- semantic/storage dtype;
- exact shape, raw byte length and raw SHA;
- layer and token position;
- layout and physical owner ids with sealed evidence;
- code pin, plan hash, executable identity and coherence id.

Prefer device buffers and one bounded post-run transfer. A host callback, print or extra root is a
new consumer and may change fusion/materialization/scheduling. Instrumentation is acceptable only
after executable identity, outputs and DSA are shown unchanged.

### Phase 3 — offline mechanism admission

1. Compute the structured mechanism fingerprint.
2. Compare it with every sealed family, independent of the candidate name.
3. Run the admission CLI under `python -S`.
4. Attack the contract: replaced evidence, duplicate/nonfinite JSON, wrong identity, missing typed
   watchpoint, wrong array SHA/shape/dtype, renamed null-family mechanism, final/intermediate
   symlinks, occupied output and mixed coherence.
5. Reject locally if any reason remains.

Do not write a TPU wrapper for a rejected or incomplete candidate.

### Phase 4 — smallest semantic and HLO checks

Only after offline admission:

1. run pure reference arithmetic on the smallest real captured row/state;
2. force CPU backends explicitly for JAX semantic tests;
3. lower the smallest relevant program and inspect exact StableHLO/optimized HLO;
4. prove one live row, exact local groups and absence of host callbacks/global hidden gathers;
5. prove the intended value flows from producer through consumer, not merely that an operation
   string or shape exists;
6. compare full raw bits and exact cutoff-active selected sets/tie order.

CPU tests prove semantics. HLO proves lowering structure. Neither proves TPU numerical association
or performance.

### Phase 5 — compile-only acquisition when necessary

Compile only when an optimized TPU graph is the missing evidence and no preserved HLO answers the
question. Before compiling:

- seal code/runtime/generated-version identity on all eight hosts;
- predeclare the exact expected bucket/module count and physical signatures;
- collect fixed-schema audits from every host;
- do not assume worker 0 is the only HLO owner;
- preserve divergent sets before cleanup;
- execute zero requests when the acquisition is compile-only;
- publish an explicit no-numerical/no-performance terminal contract.

A compile-only artifact never becomes a token, DSA, Gate D or performance result.

### Phase 6 — one protected numerical discriminator

Run only the bounded candidate authorized by a separate review. The wrapper must enforce:

- exact committed/pushed/mirrored pin and default-off flag;
- authenticated workload ownership and unique tag;
- preflight source/checkpoint/oracle/plan/HLO identities;
- no warmups or repetitions beyond the reviewed design;
- exact typed watchpoint capture;
- output tokens, DSA, state/load/cache and replica agreement;
- per-chip HBM and physical collective contract when in scope;
- failure-preserving diagnostics;
- authenticated eight-host zero-work cleanup.

A failed discriminator is useful if it is bounded, coherent and sealed. Never repeat it unchanged.

### Phase 7 — compare, seal and publish

1. Snapshot and hash before parsing.
2. Compare watchpoints in causal order and stop at the first gap/divergence.
3. Separate fact, inference and unproved hypothesis.
4. Create canonical reports with `O_EXCL`; refuse dangling or intermediate symlinks.
5. Upload generation-zero nonterminal objects first, replay exact names/bytes, then publish
   `SUCCESS` or `REJECTED` alone and last.
6. Link qualifying protected results to `bench/results.db`; diagnostic-only artifacts must say
   `db_run_id=None` and keep all numerical/performance/Gate claims false.
7. Verify the exact `US-CENTRAL2` bucket and locked repository mirror.
8. Record artifact SHAs, reviewer verdict, cleanup receipts and exact next action in the handoff,
   evidence map, reuse inventory and lessons.

## 5. Mistakes that must not recur

| Mistake | Why it failed | Permanent rule/tool |
|---|---|---|
| Rerunning an unchanged 8K graph | Reproduced known DSA drift at high cost | Tombstone graph/executable fingerprints; require a new admitted mechanism |
| Treating `context <= top_k` as DSA proof | Selected-set equality is cutoff-vacuous | Use cutoff-active contexts and oracle-relative set plus tie order |
| Calling exact M32 a legal solution | It depends on 31 sentinel rows and full-pod geometry | `decode_batch1` has one live row; admission rejects dead rows/global reconstruction |
| Adding `jax.debug.callback` to observe RMS input | Added a consumer, changed executable class and reproduced wrong DSA | Callback class is tombstoned; require unchanged executable identity |
| Reconstructing FP32 on the host | Host formula is not the unperturbed TPU value | Capture device-produced typed bytes or leave the watchpoint unobservable |
| Borrowing accepted cache for candidate arithmetic | Cache/query/head/key/scorer no longer share one history | Require one candidate coherence id and complete state |
| Using CPU as a TPU numerical oracle | CPU controls failed to reproduce TPU event-1 ordering | CPU only admits semantics; protected TPU evidence decides association |
| Reading shapes/names as value-flow proof | Matching shapes can carry the wrong source | Use SSA/callee/root traversal and SHA-pinned producer-to-consumer certificates |
| Inferring physical materialization from correction metadata | Logical BF16 association does not prove an HBM boundary | State the limit; acquire optimized causal evidence only if necessary |
| Renaming a layout/Pallas/output variant | Same mechanism returned under a new id | Compare canonical five-field fingerprints independent of `family_id` |
| Naming seven watchpoints over one opaque blob | Proved only arbitrary bytes existed | Reuse typed observability contract; validate raw array SHAs and identity evidence |
| Checking only a final symlink component | A parent symlink can redirect evidence | Component-wise dirfd traversal with `O_DIRECTORY|O_NOFOLLOW`; final `O_NOFOLLOW` |
| Assuming HLO dumps exist only on worker 0 | All eight hosts materialized the seven buckets | Audit all hosts, compare bucket/size/SHA, then choose a canonical copy |
| Running a full compile before validating the sealer | Post-compile ownership assumptions discarded expensive output | Exercise every sealer branch and failure-preservation path first |
| Omitting generated runtime version files | Reconstructed runtime imported a different semantic version | Pin/import `_version.py` and runtime identity on every host before model work |
| Using `find | wc` under `set -euo pipefail` as existence proof | Missing/empty roots were misclassified | Classify missing, link, type, readability and nonempty state explicitly |
| Publishing `SUCCESS` too early | Later diagnostics mutated a supposedly terminal archive | Replay full inventory; publish terminal marker alone and last |
| Letting offline tests discover TPU | An adjacent run initialized the local TPU client | Set `JAX_PLATFORMS=cpu` explicitly; use `python -S` when JAX is unnecessary |
| Claiming profiler/HLO/device-only speed | Does not establish clean user-visible latency | Require profiler-free wall, fresh XPlanes, correctness, HBM, DB/archive/cleanup |
| Syncing through EU/gcsfuse | Added transfer cost and unreliable POSIX rename behavior | Use locked `gsutil` mirror only to `gs://driftbench-dsv4-uc` after location check |

## 6. What the tools unlocked

The observability work produced concrete progress that blind retries could not:

- cache guards localized a silently dropped DCP stripe to a sharding round-trip boundary;
- typed accepted/DB518 comparison moved the Gate D frontier to the missing layer-1 FP32 RMS input;
- executable fingerprinting proved the callback observer was perturbing, not authoritative;
- accepted compile-only HLO established logical BF16 association and closed unsupported fusion
  stories without another numerical run;
- coherent event-1 replay rejected the LP2 K-half arm before metal;
- structured admission closed renamed formula/layout/Pallas/ownership families without recompiling;
- all-host HLO auditing corrected the false one-owner dump assumption;
- append-only publication, exact lifecycle ownership and cleanup turned failures into reusable
  evidence rather than lost time.

The general lesson is: create the debugger before asking the system to answer the hypothesis.

## 7. Decision tree

```text
Is the first causal watchpoint observable?
  no  -> design a non-perturbing typed capture or deterministic resolution
  yes -> are accepted and candidate authorities complete and coherent?
           no  -> acquire missing same-run state; never borrow another run
           yes -> does the candidate fingerprint match a sealed family?
                    yes -> reject offline
                    no  -> does it satisfy one-row/local/no-host/no-global rules?
                             no  -> reject offline
                             yes -> do CPU reference and HLO causal contracts pass?
                                      no  -> reject offline
                                      yes -> separate review for one bounded TPU discriminator
```

After a protected run:

```text
identity/integrity failure -> preserve and stop
instrumentation changed executable/output/DSA -> tombstone observer
first mismatch unchanged -> reject mechanism; do not rerun
bounded exact mechanism -> integrate behind default-off flag, then full Gate D review
```

## 8. Gate D closure remains end-to-end

No internal tensor or diagnostic closes Gate D. Closure requires one complete cutoff-active short
decoder with:

- exact raw tokens;
- exact oracle-relative DSA selected sets and tie order;
- state, load and cache integrity;
- no repeated 32-chip layer collective or hidden reconstruction;
- one real decode row and topology-local groups;
- measured per-chip HBM headroom;
- fresh eight-host XPlanes and profiler-free steady wall;
- exact code/plan/checkpoint/HLO provenance;
- `bench/results.db` linkage and same-region archive;
- authenticated eight-host zero-work cleanup.

Only after that may the project advance to protected 128K, 256K E0, identical-condition PP8/PP16/
WS32 adjudication and speculation.

## 9. Maintenance rules

- Add a watchpoint only when it narrows a causal question; record its semantics and cost.
- Every new parser or publisher needs hostile replacement, malformed input, duplicate key, path,
  symlink, occupation and append-only tests.
- Keep telemetry default off and prove its enabled path does not change the measured executable.
- Preserve immutable evidence and tombstones; never silently rewrite a report or DB row.
- Update this playbook when a failure changes a permanent rule, not for transient status.
- Put current status and exact next action in `HANDOFF.md`, not here.
