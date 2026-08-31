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

1. `layer1.rms_operands_bf16` (`hidden_update`, `residual`)
2. `layer1.rms_input_fp32` (must equal the independently derived BF16-operand sum)
3. `layer1.normalized`
4. `layer1.cache_history`
5. `layer1.query`
6. `layer1.head_weights`
7. `layer1.current_key`
8. `layer1.scorer_event1`

The next mechanism must expose or deterministically resolve the FP32 frontier and bind all eight to
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

### 3.4 Capsule constructability and sequencing

Core: `glm_tpu/greenfield/capsule_constructability.py`

Contract: `configs/greenfield-gate-d-capsule-constructability.json`

CLI: `scripts/greenfield/audit_capsule_constructability.py`

Hostile tests: `tests/greenfield/validation/test_capsule_constructability.py`

Run this before attempting to turn an abstract mechanism into a precompile capsule. It
SHA-authenticates the sealed DB518 NPZ, inventories the seven required watchpoints, binds the two
surviving shadow declarations and inspects the current admission schemas from SHA-bound source AST
without JAX.

The current result is fail-closed for two independent reasons:

- DB518 lacks the position-8155 `layer1.rms_input_fp32` and `layer1.current_key` arrays; its other
  state belongs to DB518's old code/HLO authority and cannot be relabeled as a new variant;
- the existing observability/admission v1 path requires a non-null executable-identity SHA, has no
  explicit source/AST authority, and rejects `stablehlo` or an uncompiled candidate identity. A
  generic `executable_fingerprint` label must not be overloaded to hide that missing authority. V1
  therefore cannot faithfully implement the reviewed order “source semantics + causal StableHLO +
  offline capsule/admission, then compile-only acquisition.”

Do not weaken v1 or rewrite its historical evidence. The append-only repair now exists:

- core: `glm_tpu/greenfield/gate_d_precompile_admission.py`;
- contract: `configs/greenfield-gate-d-precompile-admission-v2.json`;
- CLI: `scripts/greenfield/admit_gate_d_precompile.py`;
- hostile tests: `tests/greenfield/validation/test_gate_d_precompile_admission.py`;
- source-authority report: `docs/artifacts/gate-d-precompile-admission-v2.json`;
- current compensated-plan report:
  `docs/artifacts/gate-d-precompile-admission-v2-compensated-plan.json`;
- current compensated StableHLO-source report:
  `docs/artifacts/gate-d-precompile-admission-v2-compensated-stablehlo-source.json`;
- current real-HLO report: `docs/artifacts/gate-d-precompile-admission-v2-stablehlo.json`.

V2 authenticates committed blob/AST identity for concrete or declarative source authority, a
content-derived plan, structural StableHLO and typed eight-watchpoint capsule invariants. The tuple
survivor binds concrete commit `c8b2200` and exact sealed PP16 plan/watchpoint authority. The
compensated survivor binds distinct concrete default-off source at `e16d74f` plus its own
candidate-bound PP16 plan/watchpoint authority, but has no causal StableHLO, pinned producer or
coherent capsule authority. Declarative fixtures remain non-admissible.
Its parent process remains
stdlib-only; HLO fixture validation runs in a SHA-bound offline child using jaxlib MLIR without
importing `jax`, compiling or initializing a backend. The child uses an exact
content/symlink/safe-mode-tree-SHA root-owned CPython runtime, a bound non-root UID, `-I -S`,
`cwd=/` and no inherited import path. Provisioning is trusted manual administration outside parser
authority: reviewed bytes are installed as one fixed root-owned mode-0555 tool, only that installed
tool is run through exact `/usr/bin/python3 -I -S` or its equivalent absolute-path isolated
shebang. That pre-start invocation is the security boundary; the runtime check only detects
accidental misinvocation after startup. Admission requires it to byte-match bound source.
It nofollow-validates exact root:root `/opt` parents, atomically publishes with
`RENAME_NOREPLACE`, and strips privilege bits/xattrs. Parser Python sources execute from memfds independently rehashed after sealing against
writes/growth/shrinkage/further seals; every native parser
library must be mapped from its matching sealed inode, while every other pathname-backed mapping
must be root-owned, non-writable, identity-matched and content-recorded. It also inherits the exact v1
frontier and closed fingerprints rather than resetting history. The current report accepts the
tuple candidate's concrete source, PP16 plan and exact real causal StableHLO authority. Its capsule
bindings remain deliberately null because coherent event replay already rejected the tuple;
reported capsule gaps are schema-state only, not authority for a new capsule or admission rerun.
The compensated candidate now has a candidate-bound PP16 plan/watchpoint authority and retains
three gaps: causal StableHLO, pinned producer and coherent capsule. A distinct default-off
StableHLO producer/parser source contract now exists but has not been installed or run. Its plan binding maps the
source result `restored_input_rms_fp32[2,1,6144]` to
`layer1.rms_input_fp32.value[6144]`, requiring bitwise owner agreement before prefix `[0,0]`
selection; future capsule bytes must independently derive from both sealed BF16 operands. V2
always leaves candidate admission and TPU
authorization false. Exact module/function metadata and the complete 533-Python/46-native lowering
dependency manifests are bound by canonical digest/count, not trusted labels. Current
core/contract/report SHAs are `5403069b...b6e7`, `3f1c817c...55f4` and
`f495209a...02f1`; compensated certificate and plan-file/content SHAs are
`237095c7...6fb0`, `7d0a5615...dbdf` and `eb2c050b...b6dc`. Historical tuple PP16 plan
authority remains `98b4fa21...7880`. Current validator SHA is `17fadedf...0c19`;
historical tuple-report validator SHA remains `4338229a...6b6d`. Installed/source
provisioner and runtime-tree SHAs remain `2b9c8c2b...0594` and `308748a9...d616`. Full admission
coverage is 166/166; adjacent and producer-source suites pass 83/83. Historical authority tests must read exact pinned Git blobs, never current
worktree files.

### 3.5 Domain-specific state readers

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

### 3.6 End-to-end tool map by observation plane

No single tool is authoritative for the whole run. Use the smallest row below that can answer the
current question, then combine planes only at the terminal gate.

| Observation plane | Primary tools | What they establish | What they cannot establish |
|---|---|---|---|
| Operational black box | `docs/10-observability.md` patterns: append-only flight records, engine statistics, scheduling/compile attribution and sharding/cache guards | Whether a stall, crash, dropped stripe or recorder failure preceded the numerical symptom | Greenfield numerical truth; reuse the interfaces and failure semantics, never the legacy execution path |
| Typed numerical state | `observability.py`, `audit_observability.py`, `legacy_dsa_internals.py`, `legacy_main_cache.py`, `legacy_residuals.py`, `prompt_index_cache.py` | Immutable bytes, dtype/shape/slice, causal watchpoint order, source authority and same-run coherence | TPU causality when a value was not captured; physical execution identity; performance |
| Exact DSA behavior | `short_context_dsa_oracle.py`, the `capture_*dsa*`, `compare_*dsa*` and scorer/query association tools | Raw event state, selected-set membership, order/ties, scores, positions and current-key/cache inputs | Ranking correctness at `context <= top_k`; accepted conclusions from a hybrid cache/history |
| Logical and physical HLO | `hlo_contract.py`, `inspect_hlo_contract.py`, `live_ssa_diff.py`, `diff_layer0_live_ssa.py`, compile-only acquisition/sealer tools | Live SSA producer-to-consumer flow, layouts, reducer bodies, collective groups, roots, call graph and structural deltas | Hidden runtime bytes, numerical equality or wall latency; logical BF16 edges alone do not prove physical materialization |
| Topology and transport | `topology/discover.py`, `topology/groups.py`, `capture_topology.sh`, `inspect_topology.py`, `transport_chain.py`, `trace_pipeline_transport.py` | Runtime device permutation, explicit local groups, device-resident stage transfer, payload bytes and launch/collective structure | Complete decoder correctness or clean profiler-free latency from a synthetic transfer alone |
| Raw output and quality | sealed short-context oracle/capture tools, complete-decoder validators and the GLM quality/passkey harness indexed in `REUSE_INVENTORY.md` | Raw token identity, prompt/position alignment and quality outcomes bound to an exact run | Internal root cause or DSA correctness from token equality alone |
| Checkpoint, load and cache integrity | final-layout manifests, `inspect_{packed,runtime}_checkpoint.py`, `load_checkpoint_probe.py`, state/hash/write probes and terminal fleet validators | Exact source/final-owner bytes, checksums, finite state, slot ownership, direct load, cache health and replica agreement | Numerical equivalence of the decoder merely because all bytes loaded correctly |
| HBM | plan memory model, allocator preflight, per-worker runtime memory records and terminal fleet sealer | Planned capacity separately from measured per-chip peak, compiler overlays and required headroom | Performance or safety from a plan estimate alone; measured peak is mandatory for a protected result |
| Wall and device trace | `scripts/analysis/extract_steady_decode.py`, `scripts/analysis/parse_xplane.py` | Profiler-free warmed wall distribution separately from fresh XPlane device-step attribution | Correctness, provenance or user-visible speed from a profiler-contaminated trace alone |
| Provenance and lifecycle | `bench/provenance.py`, protected launchers, fleet lease/census guards, terminal sealers | Code/plan/checkpoint/run identity, DB linkage, unique ownership, pre/post 8-host zero work and terminal publication | Numerical truth from a worker `passed` field or DB row without the sealed raw arrays |
| Archive and recovery | protected `seal_*`, `publish_*` and `recover_*` wrappers plus generation/CRC/SHA ledgers | Durable negative evidence, exact remote object identity, recovery without recompute and failure-preserving cleanup | Promotion of a diagnostic or compile-only run into a numerical/performance result |
| Mechanism search control | `gate_d_admission.py`, `admit_gate_d_mechanisms.py`, `gate_d_precompile_admission.py`, `admit_gate_d_precompile.py`, reuse inventory and family-closure artifact | Duplicate-family detection, one-row/local/no-host legality, first-watchpoint relevance, source/AST-plan-StableHLO authority and completeness of candidate state | TPU authorization; an offline pass is only permission to seek separate review |

Use `rg --files` rather than relying on this table when locating a tool. The table describes stable
roles; exact wrappers evolve and their committed code pin is part of every result.

`prepare_compact_failure_archive.py` is bound to one historical run, not a generic publisher.
`scripts/dump_archiver.sh` is a legacy quota helper with destructive cleanup and only a weak remote
size check; it is never terminal/protected publication evidence. Current archival must use the
run-specific protected sealer/recovery path, prove immutable generation/CRC/SHA and exact object-set
equality, and delete local evidence only after that equality is authenticated.

### 3.7 Immutable teaching artifacts

These compact artifacts are the quickest way to recover *why* the current rules exist:

| Artifact | Durable lesson |
|---|---|
| `docs/artifacts/gate-d-observability-frontier.json` | The first missing accepted/candidate authority is layer-1 `rms_input_fp32`; hidden 2795 is the first observable downstream BF16 mismatch. |
| `docs/artifacts/layer1-rms-input-observer-perturbation-rejection.json` | A callback can expose bytes while invalidating the executable/output authority that made those bytes useful. |
| `docs/artifacts/callback-executable-class-certificate.json` | The rejected callback executions form a distinct, reproducible executable class; do not rearm that observer. |
| `docs/artifacts/accepted-db485-compile-only-hlo-success.json` | Compile-only acquisition can preserve a complete eight-host graph set without issuing a decode request. |
| `docs/artifacts/db485-layer1-rms-hlo-causality.json` | Accepted logical BF16 association is pinned causally, but physical BF16 materialization remains unproved. |
| `docs/artifacts/gate-d-mechanism-admission-frontier.json` | Every currently catalogued candidate is closed or incomplete; this is finite-catalogue evidence, not impossibility. |
| `docs/artifacts/pp16-feature2-qkv-khalf-event1-cpu-rejection.json` | A mechanically valid downstream arm can be rejected offline when coherent event replay still misses the oracle. |
| `docs/artifacts/plan-local-persistent-fp32-shadow-source-rejection.json` | Direct unrounded FP32-shadow substitution is source/HLO-incompatible; rounded or auxiliary-consumer forms remain separate and unadjudicated. |
| `docs/artifacts/gate-d-capsule-constructability.json` | DB518 cannot supply a new variant's coherent capsule, and admission v1 cannot represent the required precompile StableHLO authority; build append-only v2 rather than weakening history. |
| `docs/artifacts/gate-d-precompile-admission-v2.json` | V2 validates the precompile schema without weakening v1; the tuple survivor now has concrete source plus exact PP16 plan authority but still lacks causal StableHLO and producer-bound capsule provenance. |

Always recompute a file's SHA before citing it. The SHA is the identity; the filename is only a
human-readable locator.

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

### Phase 3 — offline mechanism and authority preflight

1. Run the constructability audit; reject reuse when any watchpoint is absent or belongs to an old
   code/HLO authority.
2. Compute the structured mechanism fingerprint.
3. Compare it with every sealed family, independent of the candidate name.
4. Define the smallest source-bound implementation and source-semantics certificate.
5. For a precompile source/StableHLO candidate, use and review the append-only v2 path. Preserve v1
   for candidates that already have its executable authority; never weaken or overload v1.

Do not write a TPU wrapper for a duplicate, illegal or source-unbound candidate.

### Phase 4 — source-bound semantics, StableHLO and precompile admission

Before TPU compilation:

1. run pure reference arithmetic on the smallest real captured row/state;
2. force CPU backends explicitly for JAX semantic tests;
3. lower the smallest relevant source-bound program and inspect exact StableHLO;
4. prove one live row, exact local groups and absence of host callbacks/global hidden gathers;
5. prove the intended value flows from producer through consumer, not merely that an operation
   string or shape exists;
6. construct one offline candidate-coherent eight-watchpoint capsule under the same source and
   StableHLO authority;
7. run `scripts/greenfield/admit_gate_d_precompile.py` under `python -S`, then attack evidence,
   identities, raw slices, shapes/dtypes, coherence, renamed families, symlinks and occupied
   outputs;
8. reject locally if any condition remains incomplete.

CPU tests prove semantics. StableHLO proves pre-TPU lowering structure. Neither proves optimized
TPU association, numerical behavior or performance.

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
| Directly substituting an unrounded FP32 residual shadow | Removes the sealed accepted BF16 recurrent round | Reject direct substitution; fingerprint rounded/auxiliary-consumer variants separately |
| Borrowing accepted cache for candidate arithmetic | Cache/query/head/key/scorer no longer share one history | Require one candidate coherence id and complete state |
| Using CPU as a TPU numerical oracle | CPU controls failed to reproduce TPU event-1 ordering | CPU only admits semantics; protected TPU evidence decides association |
| Reading shapes/names as value-flow proof | Matching shapes can carry the wrong source | Use SSA/callee/root traversal and SHA-pinned producer-to-consumer certificates |
| Inferring physical materialization from correction metadata | Logical BF16 association does not prove an HBM boundary | State the limit; acquire optimized causal evidence only if necessary |
| Renaming a layout/Pallas/output variant | Same mechanism returned under a new id | Compare canonical five-field fingerprints independent of `family_id` |
| Naming watchpoints over one opaque blob | Proved only arbitrary bytes existed | Reuse typed observability contract; validate raw array SHAs and identity evidence |
| Checking only a final symlink component | A parent symlink can redirect evidence | Component-wise dirfd traversal with `O_DIRECTORY|O_NOFOLLOW`; final `O_NOFOLLOW` |
| Assuming HLO dumps exist only on worker 0 | All eight hosts materialized the seven buckets | Audit all hosts, compare bucket/size/SHA, then choose a canonical copy |
| Running a full compile before validating the sealer | Post-compile ownership assumptions discarded expensive output | Exercise every sealer branch and failure-preservation path first |
| Omitting generated runtime version files | Reconstructed runtime imported a different semantic version | Pin/import `_version.py` and runtime identity on every host before model work |
| Using `find | wc` under `set -euo pipefail` as existence proof | Missing/empty roots were misclassified | Classify missing, link, type, readability and nonempty state explicitly |
| Publishing `SUCCESS` too early | Later diagnostics mutated a supposedly terminal archive | Replay full inventory; publish terminal marker alone and last |
| Reopening inputs by pathname after preflight | A same-name replacement can mix histories across one replay | Snapshot every upstream before execution; retain/revalidate file descriptors and inode/size/hash identities |
| Letting an execution receipt choose its own producer/import closure | A hostile committed replay can fabricate coherent-looking exact outputs | Independently pin producer/replay blobs and the complete replay-visible committed source manifest |
| Assuming a sentinel format from its filename | A pinned `SUCCESS` was sha256sum text, not JSON, and spent a reviewed start before replay | Inspect and hostile-test exact sentinel bytes during source-only preflight |
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
- retained-fd tensor receipts and independent executable-closure pins closed TOCTOU and
  self-selected-producer substitution before any replay;
- all-host HLO auditing corrected the false one-owner dump assumption;
- append-only publication, exact lifecycle ownership and cleanup turned failures into reusable
  evidence rather than lost time.

The general lesson is: create the debugger before asking the system to answer the hypothesis.

### 6.1 Unlock sequence: from blind retries to a causal frontier

The most important progress was not a faster kernel but a sequence of questions that became
answerable:

1. State/hash/cache guards showed whether the failure was load corruption, dropped ownership or
   genuine arithmetic drift.
2. Accepted internal captures and typed comparisons moved the search from final tokens to exact
   layer/event boundaries.
3. Pre-reduction partial captures closed checkpoint decode, FP8 scales, contractions and local
   leaves before testing association.
4. Exact StrategyND and attention/cache replays separated formula correctness from physical
   reduction and producer/consumer ownership.
5. Live-SSA diffing replaced source-level guesses with one observed scheduled-graph delta at a
   time.
6. Observer executable fingerprints exposed that additional consumers were changing the system
   being measured.
7. Compile-only all-host HLO acquisition recovered physical graphs without contaminating a
   numerical result or paying for another full run.
8. The immutable watchpoint auditor proved the current evidence gap instead of silently borrowing
   state from incompatible histories.
9. Canonical mechanism fingerprints turned many renamed variants into immediate offline
   rejections.
10. Exact archive, DB and cleanup contracts made a failed run reusable; it no longer disappeared
    into logs or required an identical rerun.

### 6.2 Questions now answerable quickly

| Question | Fastest reliable answer |
|---|---|
| Did bytes change, and where first? | Immutable typed watchpoint audit in causal order |
| Is a candidate using mixed histories? | Code/plan/executable/coherence binding before arithmetic |
| Is the idea actually new? | Five-field fingerprint against the sealed family catalogue |
| Did a named HLO operation affect the live output? | Exact SSA/callee/root traversal with hostile mutations |
| Did instrumentation perturb execution? | Compare executable, executable-including-data and host-transfer fingerprint triples plus outputs/DSA |
| Is the collective local and live? | Physical replica groups plus producer-to-root HLO certificate on every host |
| Is a short DSA pass meaningful? | Require `context > top_k`, or classify it only as state/tail coverage |
| Is a run fast? | Profiler-free warmed wall; use XPlane only for attribution |
| Can a failed run be reused? | Generation/CRC/SHA ledger, preserved raw HLO/tensors/logs and explicit diagnostic-only claims |
| Is the fleet safe for the next action? | Lease ownership plus authenticated 8-host zero-work census |

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
- Treat metadata authority and payload residency as separate identities. A local manifest/SUCCESS
  copy does not imply its multi-GB destination files are beneath the same directory. Bind the
  payload root independently and, for mounted object storage, require one exact filesystem/source,
  read-only mount options, no longer-covering nested mount, opened-descriptor mount-ID/device
  binding, and unchanged mount authority before and after retained-fd payload reads.
- Never copy reviewed bytes into a privileged executable location directly from a same-UID mutable
  path. Stage a one-file/tree capsule, bind its tree SHA, and use the sealed no-replace provisioner;
  only then hardlink and atomically replace the named executable before checking inode/SHA/mode.

## 10. Minimal incident packet

When a run or offline audit fails, preserve enough information for another session to diagnose it
without rerunning:

- UTC tag, hypothesis, candidate fingerprint and first watchpoint;
- Git/code/runtime/generated-version, plan, checkpoint and oracle identities;
- exact command, environment flags, host/JAX-process/device map and local replica groups;
- preflight and postflight fleet census plus workload owner;
- raw typed arrays and their schema/shape/dtype/SHA, not only summaries;
- every StableHLO/optimized-HLO file from every host, with canonical and raw digests;
- executable, executable-including-data and host-transfer fingerprints;
- tokens and complete cutoff-active DSA event arrays;
- cache/state/load/replica/HBM records;
- profiler-free per-step wall samples and, only when required, fresh XPlanes;
- stdout/stderr, failure trap output, upload receipts and exact remote generation/CRC/SHA ledger;
- explicit claims matrix: numerical, Gate D, HLO, performance, recovery and cleanup true/false;
- reviewer verdict, terminal classification and the one exact next action.

If any item was never produced, record it as absent. Never fill an evidence gap with inference.

For offline parser acquisitions, preflight the exact parent interpreter against repository import
requirements before consuming the reviewed start. Keep fixture and real-lowering slice hashes
separate: a fixture hash must be reproducible from its declared operation graph, while only the
sealed immutable parser can promote the committed graph hash. Parent comparison failures must name
each mismatched field with expected and observed compact values; a single aggregate drift string is
insufficient observability and causes unnecessary protected retries.

## 11. Definition of an observable mechanism

A mechanism is eligible to request a separate compile-only review only when all of the following
are true:

- its hypothesis names one first causal watchpoint and one falsifying outcome;
- it has a unique canonical fingerprint not already closed by evidence;
- it uses one logical decode row, groups of at most four, device-only execution and no full-pod
  hidden reconstruction;
- all eight Gate-D candidate watchpoints come from one authenticated history, including both BF16
  RMS operands and their independently derived FP32 sum;
- its smallest real captured-input reference test passes;
- its expected source/StableHLO and structural contract names the required live value flow and
  rejects every locally representable hostile decoy;
- the proposed acquisition is bounded, performs no numerical request and cannot publish a
  numerical, Gate-D or performance claim;
- failure, archive, DB rollback and eight-host cleanup paths have been tested locally;
- an immutable adversarial review has approved the exact staged diff and compile-only evidence
  contract.

Eligibility is not authorization. Only a separate compile-only review may authorize acquisition of
the TPU-specific lowering. A later numerical-run review is eligible only after that exact
StableHLO/optimized-HLO pair is preserved, independently replayed against its causal/locality
contract, mutation-tested, and bound to the intended executable identity; the complete candidate
state and protected numerical wrapper must then receive another immutable review.

Until then, the mechanism is an idea, not a TPU experiment. Neither this checklist, its persistence
review nor a compile-only result authorizes TPU numerical execution. Only the separate
execution-only review may authorize one bounded protected discriminator, and even an exact
discriminator does not close Gate D.
