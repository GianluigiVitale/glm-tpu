# Round-8 adversarial review — kernel-freeze conformance + artifact honesty

- **Target:** `~/tpu-inference-next`, branch `glm-5.2-v4-next` @ `82fe3f766c1335afe78f19abe1e1258a6779ee53` (verified `git rev-parse HEAD`).
- **Freeze references:** DSA kernels @ `c145693728d46b45e902110405f542f301f2d443`; `mla/v2/kv_utils.py` @ `02e44b36` (merge of the pg2 OOB fix `63427f86`).
- **Method:** read-only git forensics (blob hashes, `--full-history` path logs, ancestry checks, code-line-only diffs), artifact count reconciliation, diff-based env-gate enumeration vs fork base `97938b62`. **TPU never touched; no python executed against the target tree.**
- **Reviewer stance:** adversarial — every claim below was attacked before being accepted (attack log in §6).

## Verdict summary

| Audit item | Verdict |
|---|---|
| (1) Kernel freeze (dsa/* @ c1456937, kv_utils.py @ 02e44b36) | **HOLDS — byte-identical**, no touch-then-revert, working tree clean |
| (2) Round-5/6/7 fix commits: wiring only, no kernel-math change | **HOLDS** on every commit reachable from HEAD; **one round-6 fix never landed** (Finding 1) |
| (3) Artifact integrity (2 committed logs) | **INTACT and honest** — hashes resolve, every count reconciles exactly |
| (4) Env-gate default-off discipline | **HOLDS for all 20 new gates** — no gate whose default deviates from pre-change behavior |

Numbered findings: 1 MEDIUM (unlanded fix), 2 scope/clarity notes. No freeze violation. No artifact dishonesty.

---

## 1. Kernel-freeze conformance — HOLDS (byte-identical)

Blob-hash comparison (stronger than `git diff`; a hash match is byte-identity by construction):

| File | Freeze ref blob | 82fe3f76 blob | Identical |
|---|---|---|---|
| `tpu_inference/kernels/dsa/__init__.py` | c1456937: `c1df7016…` | `c1df7016…` | YES |
| `tpu_inference/kernels/dsa/indexer_kernel.py` | c1456937: `927fe5cd…` | `927fe5cd…` | YES |
| `tpu_inference/kernels/dsa/sparse_mla_kernel.py` | c1456937: `a6898108…` | `a6898108…` | YES |
| `tpu_inference/kernels/mla/v2/kv_utils.py` | 02e44b36: `e90eda06…` | `e90eda06…` | YES |

Supporting checks, all passed:

- `git diff c1456937 82fe3f76 -- tpu_inference/kernels/dsa` → **empty**. `git diff 02e44b36 82fe3f76 -- …/kv_utils.py` → **empty**.
- Both freeze refs are **ancestors of HEAD** (`git merge-base --is-ancestor`), so the diffs are along real history, not cross-branch tree comparisons.
- **Touch-then-revert attack:** `git log --full-history <ref>..82fe3f76 -- <paths>` lists ONLY merge commits (0b213d86, cda23c81, c6492178, 025bdd25, 534cd74d, f095a8ca, f5f10b63) — merges appear in full-history path logs when either parent differs, but each merge's resulting tree is blob-identical to the freeze state (verified above at HEAD; every non-merge commit touching these paths is an ancestor of the freeze ref itself). **No non-merge commit changed any frozen file after its freeze point.**
- Working tree: `git status --porcelain tpu_inference/kernels/` and `git diff HEAD -- tpu_inference/kernels/` both **empty** — no uncommitted kernel edits hiding drift.
- The two kernel-touching non-merge commits found anywhere in range are pre-freeze or the freeze itself: `c8a51543` (round-5 kernel fixes) is an **ancestor of c1456937** — i.e., *inside* the frozen baseline; `63427f86` (pack_new_kv OOB fix) is the change 02e44b36 merges — i.e., it *is* the frozen kv_utils state.

**Scope note (Finding 2, not a violation):** the freeze as specified covers `kernels/dsa/*` and `kernels/mla/v2/kv_utils.py` only. `kernels/mla/v2/kernel.py` changed substantially since c1456937 (+686/−211: the S2 GLM_MLA_DCP/LSE variant `29a51331` plus the comment-only `70aa6825`). Its gate-off path is claimed **dataflow-identical, not byte-identical** (scalar address hoists above paired `dma_start`s; DMA order/operands unchanged) — a claim that round-6 F1 itself corrected from an earlier "byte-identical" over-claim, and that is pinned by `tests/layers/common/test_mla_dcp.py::test_gate_off_traced_jaxpr_byte_identity` (byte-identity at the `mla_attention` level with the kernel mocked) and `test_dcp_gate_is_noop_without_dcp_axis`. Anyone reading "kernel freeze" as covering all of `kernels/mla/v2/` would be misled; the honest, corrected claim is in the tree (see §2, 70aa6825).

## 2. Round-5/6/7 fix commits — wiring contracts/guards/adapters only

Each fix commit's diff was reduced to executable lines (comments/docstrings stripped) and inspected for changes to score math, top-k selection, or attention semantics.

| Commit | Round | Reachable from HEAD | Freeze relation | What the executable diff does |
|---|---|---|---|---|
| `c8a51543` r5fix | 5 | yes (via c780caa6) | **ancestor of c1456937** — part of the frozen baseline | Wiring: `_linear_weight_f32` fp8-dequant **adapter** (dequantizes fp8-resident `wq_b` codes+scales before scoring instead of silently scoring raw codes — input-correctness adapter; the score formula/top-k it feeds is untouched); `_check_xla_ref_positions` host **guard** (xla_ref refuses non-arange positions loudly). Its kernel-file changes pre-date the freeze and are frozen content. |
| `53c5e5ee` | 5 | yes | ancestor of c1456937 | `_axes_tuple` axis-name **normalization** (fixes `set(str)` decomposition under the 2D scheme), DP-mesh trace-time **guard**, docstrings. No math. |
| `70aa6825` F1 | 6 | yes | after freeze; touches `kernels/mla/v2/kernel.py` | **Comments/docstrings only** — verified line-by-line: every hunk in `kernel.py` and `attention_interface.py` edits comment or docstring text (the claim-accuracy correction of §1's scope note). Commit message says "no executable line changed" and the diff confirms it. |
| `413db9e1` F2/F3/F4a | 6 | yes | after freeze | `flight_recorder.py` + one `tpu_runner.py` call-site: passes `async_scheduling` provenance into `recorder_init`. Observability only; no kernel or wiring math. |
| `755b1719` finding-1 | 6 | **NO** | after freeze | See **Finding 1** below — never merged to -next. |
| `2e41b08a` 2int follow-ups | 7 | yes (via f5f10b63) | after freeze | (a) `precompute_indexer_params`/`stored_indexer_params`: the *same* weight-adaptation math moved from per-step to load time (PWAL), byte-identity **test-pinned** (`test_precomputed_indexer_params_byte_identity`, adapter call-counted); fallback path is the verbatim old per-step adapter. (b) `GLM_DSA_SCORER` gate: default `'xla'` keeps the **verbatim** pre-existing `paged_indexer_scores(...)` call in the `else` branch; the Pallas scorer only runs when explicitly set. No score/top-k semantics changed on any default path. |
| `63427f86` pg2 OOB | 7 | yes | **is** the frozen kv_utils state (merged as 02e44b36) | The one sanctioned kernel-logic change: removes an OOB VMEM read. Reviewed separately (round7-pg2-oob.md). |

**Positive assurance:** in every fix commit reachable from HEAD, `glm_dsa_indexer.py` / `mla_attention.py` changes are adapters (dtype/dequant/param-storage), guards (position/mesh/axis-name), gates (trace-time env dispatch), or documentation. I found **no change to score computation, top-k selection, or attention semantics** in any of them. The only numerical-behavior change any of them causes is c8a51543's dequant adapter in the fp8-resident config — which replaced provably-garbage inputs (raw fp8 codes scored without scales) with correct ones, is pre-freeze, and is part of the frozen baseline.

### Finding 1 — MEDIUM (unlanded round-6 fix; honesty intact): `755b1719` is stranded on `glm-5.2-v4-2a2`, not on -next

- The round-6 paged-indexer review's finding 1 (MEDIUM, latent: `valid=None` default makes pad-clobber protection opt-in) was fixed by `755b1719` ("pad mask derived internally, no longer opt-in") — but that commit's **only** containing branch is `glm-5.2-v4-2a2`. The 2int line merged the 2a2 branch at its parent `3ca5c85c` (pre-fix, merge `f57156ad`), and nothing re-applied it.
- **On HEAD today:** `glm_dsa_indexer.py` still has `write_indexer_keys(..., valid=None)` (line 729-732), `compute_topk_indices_paged(..., valid=None)` (line 828), `topk_indices_for_layer_paged(..., valid=None)` (line 911) — the exact latent hazard the review described (an unmasked pad row resolves to an IN-RANGE live slot and silently overwrites a live cached key).
- **Refutation attempt (why this is not CRITICAL):** I hunted for an unprotected production caller and found none. The only non-test call sites are `mla_attention.py:1059` and `:1153`, and both pass `valid=tok_valid` (derived from `token_request_ids` + live-request masking). `compute_topk_indices_paged` / `topk_indices_for_layer_paged` have **no** production caller on -next (tests and docstrings only). So the hazard is exactly as the round-6 review graded it: latent until a future (Stage-3) caller forgets the keyword.
- **Honesty check — passes:** RESEARCH_LOG.md:558 records the fix as living on "`glm-5.2-v4-2a2` @ 755b1719" (a feature branch, each "committed + pushed on its feature branch") and explicitly lists the 2a2↔r5fix cross-merge as a pending integration step. No document claims the fix is on -next. This is a *tracking* gap, not a dishonesty.
- **Action:** either cherry-pick/port `755b1719` onto -next before any Stage-3 caller touches the paged entry points, or record a loud TODO at the three `valid=None` signatures.

## 3. Artifact integrity — both artifacts INTACT; counts reconcile exactly

Both artifacts are committed in `~/glm-tpu` @ `6c62eb6` ("artifacts: pin clean-clone CPU-suite reproductions"), branch `main`, in sync with `origin/main`.

### 3a. `docs/artifacts/stage2-cpu-suite-c1456937.log`

- **Header hash exists:** `c145693728d46b45e902110405f542f301f2d443` resolves in the target repo to exactly the commit the header names ("dsa: Stage-2 sparse PREFILL…"). Branch name in header (`glm-5.2-v4-sparse-prefill`) matches the commit's branch line in the graph.
- **Counts reconcile:**
  - `collected 124 items` = summary `124 passed` (0 failed, 0 skipped).
  - Result lines counted independently: `grep -c "^tests/.*PASSED"` = **124** (the two extra raw "PASSED" strings are prose: the header's XLA_FLAGS note and the "Per-file PASSED:" label).
  - Per-file breakdown in the artifact (24/42/32/10/12/4 across the six files) **matches the actual per-file result-line counts exactly** — all six files, no deviation.
- **Honesty positive:** the artifact's CLAIM CHECK section itself *refutes* the relayed "Stage-2: 140+ CPU tests pass" claim ("NOT reproduced as stated at this commit… 124 passed"), records the shortfall as a count deviation, and notes c1456937's own commit message claimed 108. An evidence artifact that documents the failure to reproduce the headline claim is the opposite of artifact laundering.
- Methodology stated in the header is sound: fresh clone (not a worktree), PYTHONPATH forced to the clone to defeat the editable install, `JAX_PLATFORMS=cpu` exported before python, backend verified `cpu` pre-run.

### 3b. `docs/artifacts/next-cross-suite-999f0307.log`

- **Header hash exists:** `999f03078fdc46cc4979444de9b1b3ed9eb72e15` resolves to exactly the named commit ("tests: append --xla_force_host_platform_device_count=8…"); the header correctly notes the branch tip had already advanced (detached checkout at the tasked commit) — precise provenance, not hand-waving.
- **Counts reconcile:**
  - `collected 373 items` = 250 passed + 123 skipped exactly.
  - Independent result-line counts: **250** `PASSED`, **123** `SKIPPED` (extra raw-string matches are prose in the reconciliation section).
  - The claim-check arithmetic inside (243 + 1P + 6P = 250; 103 + 20 = 123 skipped) is internally consistent.
- The artifact **discloses** that the merge agent's exact pytest invocation was never written down and that the file list is a reconstruction, recorded as evidence in three steps — the honest treatment of a relayed agent-to-agent figure.
- Verdict "REPRODUCED EXACTLY" for 250P/123S/0F is supported by the embedded pytest transcript.

## 4. Env-gate sweep — every new gate defaults to pre-change behavior

Enumeration method (to catch non-`GLM_*` gates): `git diff 97938b62 82fe3f76 -- 'tpu_inference/*.py'` filtered to added `os.environ`/`getenv` reads, plus a `GLM_[A-Z_]+` identifier sweep of the whole production tree, plus the bench harness in `~/glm-tpu/bench/engine.py`. 97938b62 = merge-base with `origin/main` (fork base).

| Gate | Where read | Default (unset) | Unset behavior = pre-change? |
|---|---|---|---|
| `GLM_DSA_MODE` | `glm_dsa_indexer.glm_dsa_mode()` | `'off'`; invalid value → loud `ValueError` | **YES.** Off: `mla_attention.py:722-731` falls through to the pre-existing upstream dispatch (dense path); `kv_cache_manager._dsa_indexer_spec_for_mode` returns `None` → no indexer cache spec, Stage-1 allocation byte-identical. Test-pinned: `test_mode_off_no_behavioral_change`, `test_dsa_indexer_spec_none_when_mode_off`. |
| `GLM_DSA_SCORER` | `glm_dsa_scorer()` | `'xla'`; invalid → `ValueError` | **YES.** Default keeps the verbatim `paged_indexer_scores` call; double-gated (only consulted under `GLM_DSA_MODE=pallas_decode`). |
| `GLM_MLA_HEAD_SHARDED` | `attention_interface.py:623` | `"0"` → off | **YES.** Also defers to explicit caller shardings even when set. Tests: `test_gate_is_noop_at_mesh_product_one`, `test_gate_defers_to_explicit_shardings`. |
| `GLM_MLA_DCP` | `attention_interface.py` (`_dcp`) | `"0"` → off | **YES.** Requires BOTH the env AND a dcp mesh axis > 1. Gate-off trace byte-identity test-pinned at the `mla_attention` level (`test_gate_off_traced_jaxpr_byte_identity`); kernel body dataflow-identical (the corrected round-6 F1 claim — see §1 scope note). |
| `GLM_FLIGHT_RECORDER` | `flight_recorder.py:71` | `!= "1"` → `maybe_create_flight_recorder` returns `None` | **YES.** No recorder object, no hot-path writes. |
| `GLM_FLIGHT_RECORDER_DIR` | `flight_recorder.py` | `/tmp` | **YES** — only consulted when the recorder gate is already on. |
| `GLM_ASYNC_SCHED` | fork: `flight_recorder.py:102` (provenance only — recorded, never gates); bench: `engine.py:87` | fork: recorded as-is; bench: acts **only when `== "0"`** (sets `async_scheduling=False`) | **YES.** Unset = vLLM's own default (async on for the Ray TPU executor). Note: inverted-sense gate — see Finding 3. |
| `GLM_LOG_STATS` | bench `engine.py:95` (driver-side only; not in tpu_inference) | acts only when `== "1"` (`disable_log_stats=False`) | **YES** — unset = quiet, vLLM offline default unchanged. |
| `GLM_DCP` (bench-side, additional) | bench `engine.py` | unset/0/empty → kwarg **absent** from `LLM(...)` | **YES** — byte-identical engine build, per the code comment and kwarg-name verification note. |
| `TPU_MLA_V4_KV_PAGES` / `TPU_MLA_V4_QUERIES` | `attention_interface.py:827/830` | `"1,1,1"` / `"1,8,8"` | **YES** — verified against the introducing commit `a429be54`: the defaults are exactly the previously hard-coded constants `(1,1,1)` / `(1,8,8)`. |
| `TPU_MIN_TOKEN_BUCKET` | `tpu_runner.py:747` | `0` → floor is `max(0, 16, …)` | **YES** — verified against `97938b62:tpu_runner.py:711`, whose floor was `max(16, …)`; `max(0, …)` is arithmetically identical. |
| `DSV4_PRECOMPILE_FWD_CHAIN` | `compilation_manager.py:231` | `"0"` → off | YES |
| `DSV4_OBSERVE_COMPILES` | `tpu_runner.py:501` | unset → off | YES |
| `DSV4_DECODE_COMP_BLOCK` / `DSV4_DECODE_KV_BLOCK` / `DSV4_PREFILL_KV_BLOCK` / `DSV4_DECODE_RECOMPUTE_BLOCK` | `deepseek_v4_attention.py` | `0` → "the full/validated reference, unchanged" (per in-code comments) | YES |
| `DSV4_DEBUG_CACHE` | `deepseek_v4_attention.py:1712` | unset → no debug prints | YES |
| `DSV4_DECODE_CACHE` | `deepseek_v4_attention.py:1350` | unset → "default prefill topology … unchanged" | YES |
| `DSV4_SHARD_ATTN` | `deepseek_v4_attention.py:1264` | `!= "1"` → "the proven replicated path" | YES |
| `DSV4_PALLAS_DECODE` | `deepseek_v4_attention.py:1841/2006` | unset → XLA decode reference | YES |

**No gate was found whose unset default is not the pre-change behavior.** `REQUANTIZE_WEIGHT_DTYPE`, `DISABLE_WEIGHT_REQUANTIZATION`, `RAIDEN_*`, `TPU_OFFLOAD_*` etc. also appear in the tree but pre-date the fork base (absent from the added-lines diff) — out of scope.

### Finding 3 — NOTE (operator ergonomics, not a violation): `GLM_ASYNC_SCHED` is an inverted-sense gate

Every other GLM gate is "set 1 to enable a new path"; `GLM_ASYNC_SCHED` is "set **0** to *disable* an existing vLLM default". Unset still equals pre-change behavior (discipline holds), but an operator pattern-matching "GLM_* unset = safest" would leave async scheduling ON — which docs/06 implicates in the pod core-halt finish-step correlation. The bench comment documents this correctly; flagging here so runbooks say it too. Related, already-documented caveat (attention_interface.py:614-618): all `GLM_*` gates are raw trace-time reads — Ray forwards only `VLLM_*` to workers, so multi-host launchers MUST bake them into the raylet env or hosts trace different SPMD programs.

## 5. Numbered findings

1. **MEDIUM — round-6 paged-indexer finding-1 fix (`755b1719`) never landed on -next.** HEAD keeps `valid=None` opt-in defaults on `write_indexer_keys` / `compute_topk_indices_paged` / `topk_indices_for_layer_paged` (glm_dsa_indexer.py:729-732, 828, 911). Mitigated today: both production call sites pass `valid=tok_valid`; the paged entry points have no production caller yet. RESEARCH_LOG honestly records the fix's actual location; nothing claims it is on -next. Port the commit (or equivalent) before Stage-3 wiring touches these entry points. (§2, Finding 1.)
2. **NOTE — freeze scope is narrower than the phrase "kernel freeze" suggests.** `kernels/mla/v2/kernel.py` legitimately changed (+686/−211, gated `GLM_MLA_DCP`) and its gate-off path is dataflow-identical, not byte-identical — the corrected claim from round-6 F1, verified comments-only in `70aa6825` and test-pinned. Freeze statements should name the four frozen files explicitly. (§1.)
3. **NOTE — `GLM_ASYNC_SCHED` acts on `"0"`, not `"1"`.** Unset = vLLM default (async ON). Discipline holds, but runbooks should state the inverted sense given async's role in the halt investigation. (§4.)

## 6. Attack log — refutations attempted, all failed (positive assurance)

1. **Hidden kernel drift via revert:** `--full-history` path logs across both freeze ranges → only TREESAME merges; blob hashes at HEAD equal freeze blobs; working tree clean. No way to hide a change under these three checks simultaneously.
2. **Cross-branch diff illusion:** confirmed both freeze refs are ancestors of HEAD, so the empty diffs are along real history.
3. **Kernel math smuggled through wiring commits:** stripped every r5/6/7 fix diff to executable lines; found only adapters/guards/gates. `70aa6825`'s kernel.py hunks verified comment-only, matching its "no executable line changed" claim.
4. **Artifact count forgery:** independently recounted result lines and per-file totals in both logs; every number matches the summary lines and the artifacts' own breakdowns; both header hashes resolve to commits whose subjects match the headers.
5. **Artifact over-claim:** artifact 1 *documents* the failure to reproduce "140+" (124 actual) rather than hiding it; artifact 2 discloses its file-list reconstruction. No laundering found.
6. **Un-swept gate:** enumerated env reads by diff against the fork base (catches non-GLM prefixes) — found and cleared `TPU_MLA_V4_*` (defaults = former hard-coded constants, verified in `a429be54`) and `TPU_MIN_TOKEN_BUCKET` (default floor arithmetically identical to `97938b62`'s), plus ten DSV4_* opt-ins, plus three bench-side GLM knobs.
7. **Default-path contamination by PWAL:** `precompute_indexer_params` only runs when `GLM_DSA_MODE != off` (mla_attention.py:532-533), so the default load path registers no extra Parameters.

*Round-8 reviewer, 2026-07-08. Read-only audit; no TPU access; no code executed from the target tree.*
