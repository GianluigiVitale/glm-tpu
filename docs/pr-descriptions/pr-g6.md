# PR-G6 — [Kernel][DSA] Lightning-indexer + top-k sparse-MLA decode Pallas kernels for TPU

**Branch:** `pr-g6-dsa-kernels` (worktree `~/tpu-inference-prs`, pushed to the fork)
**Commits:** `0ac6eb9e4` (the five files at the silicon-validated `glm-5.2-v4-next` state
@ `886eaceb` — identical through the **local** `-next` head `cda8a707`, unpushed at cut
time — plus a yapf-0.43.0 normalization, AST-identical) + `5bf3e5927` (docstring/comment
accuracy pass from the post-cut adversarial review; code-AST-identical — `ast.dump`
equality with docstrings stripped; see "Deltas vs the dev branch" below). Squash on
submission if preferred.
**Base:** `97938b62` (fork main, 2026-06-13).
Forward-port to `vllm-project/tpu-inference` main @ `99a662a1` (2026-07-08): **conflict-free**
(`git merge-tree` trial merge clean — the PR is five new files; no `kernels/dsa/` path exists
on the tip).
**Status:** ready for owner review. **KERNELS ONLY** — no serving wiring (see Layering).

---

## PR body draft (paste below into the GitHub PR)

# Description

Self-contained JAX/Pallas kernels for **DSA** (DeepSeek Sparse Attention: the
lightning-indexer + top-k + sparse-MLA mechanism of DeepSeek-V3.2) on TPU — the attention
path of the `GlmMoeDsa` model family (GLM-5.x, DeepSeek-V3.2-style checkpoints, Kimi-K2.6).
To our knowledge these are the first public JAX/Pallas kernels for this DSA variant on TPU:
**exact** top-k over the **uncompressed** MLA latent KV cache (the in-tree
`kernels/experimental/deepseek_v4/` stack is the adjacent-but-different DeepSeek-V4
mechanism — KV-compressor-based, with StreamIndex selection at compressed-block
granularity, wired to `vllm.models.deepseek_v4`; see the duplicate-work section). PR #2324, which enables the
GLM family on v4, runs with the indexer **disabled** (`TPU_DISABLE_DSA_INDEXER`, "until a
JAX-native DSA lands") — this PR is that missing piece, at the kernel layer.

New package `tpu_inference/kernels/dsa/` (imports only `jax`/`jax.experimental.pallas`;
no other tpu_inference dependency):

- **`indexer_kernel.py`** — lightning-indexer scoring over the paged indexer k-cache
  (`relu(q·k · D^-0.5)` → signed weighted head-sum, the HF `modeling_glm_moe_dsa` math)
  as BOTH a pure-XLA blocked scorer (memory-bounded `lax.map` oracle — also usable as a
  reference core) and a Pallas kernel (scalar-prefetched flattened block tables, one
  physical page per grid step, double-buffered HBM→VMEM streaming — the idioms of the
  validated DSV4 CSA kernel), plus **exact hierarchical blocked top-k** (per-block
  `lax.top_k` + rebase + one merge `lax.top_k`, provably equal to flat `lax.top_k`;
  `approx_max_k` deliberately not used). Validity/causality by exact `-inf` fill; output
  indices `-1`-padded with `n_valid = min(kv_len, topk)`.
- **`sparse_mla_kernel.py`** — decode-shaped sparse-MLA consuming the selected indices:
  `gather_kv_segment` (XLA) pulls the top-k latent rows from the paged MLA cache into a
  fixed `[R, K, kv_width]` segment; the Pallas kernel runs a tiled flash decode over it
  (online softmax, PV over the 512-wide nope latent only — the mla.v2 W_UV-outside
  contract; prefix-length masking; fully-masked rows exactly zero), with a same-math XLA
  oracle in-file. Sparse cost is context-independent: at GLM-5.2's `index_topk=2048`,
  ~64× less latent HBM traffic than dense at 128K ctx (break-even ctx = 2048).

Both kernels carry their contracts in docstrings (dtype equality asserted at the
boundary, page/tile sizes 128-multiples, pad-page-id DMA note, tie semantics).

**Silicon validation (single-chip TPU v4, 2026-07-08).** Two gates, run to a
pre-registered protocol with archived artifacts (paths below are in the submitter's
harness repo):

- *Mosaic compile gate (2a)*: `interpret=False` compile+run of the indexer kernel.
  Three genuine v4 Mosaic lowering rejects were hit and fixed — (1) the fp32 `(1, H)`
  w-tile violates the (8,128) block-divisibility rule at R>1 → w is passed
  host-transposed `[H, R]` as a full-array block; (2) a 2-D `(1, page_size)` output
  block hits the same sublane rule → 3-D `[R, nb, P]` full-tail output block with a
  dynamic second-minor row store; (3) a lane-dim dynamic slice fails Mosaic's provable
  128-alignment check (E2003) → static lane-mask select + broadcast-multiply head-sum
  (only summation order differs from the dot_general form). Final run: **ACCEPT**
  (`kernelprobe-2a-20260708.log` … `-try4.log`).
- *MXU parity gate (2b)*: **all green on real silicon**
  (`kernelprobe-2b-20260708-metal2.results.txt`): indexer Pallas-vs-XLA fp32
  max|Δ| = 2.4e-7 (bar 5e-4); **selected set EXACT — 0 non-tie index mismatches — vs an
  independent HF-math oracle** (fp32); bf16 selection: 0 mismatches outside the 2^-8
  k-th-score boundary band (see Risk for the S2 semantics); sparse-MLA vs oracle fp32
  9.5e-7 (bar 2e-6) / bf16 1.95e-3 (bar 2.5e-3); cache-write OOB geometry probe clean at
  kv_len 511/512/513. Full-disclosure notes: the first metal parity run "failed" A2 —
  root-caused to the *oracle* running at default (bf16) MXU precision, not to the kernel
  (oracle since pinned to `highest`; `kernelprobe-2b-20260708-metal.results.txt` archives
  the failure); gate-A parity rows ran at probe shape k=64 (topk=2048 selection is
  covered exactly by the CPU suites and by gate B's 2048-slot segment); the parity gate
  is a correctness gate — **no on-metal performance numbers are claimed yet** (the
  microbench vs the XLA core is an explicitly remaining item).

## Why this is not duplicating an existing PR

Live sweep 2026-07-08 (GitHub search API + files/diff reads, queries: `DSA`,
`sparse attention`, `indexer`, `lightning indexer`, `top-k attention`, `sparse MLA`,
`NSA`, `deepseek sparse`, `GLM`, plus a title scan of the 50 newest open PRs and file
lists of every candidate): **no open PR ships DSA kernels.**

- **#2324** (GLM-5.1-FP8 on v4, head `c2822bd7`, 2026-07-04): adds **no** kernel files and
  explicitly **disables** the indexer — `envs.py` adds
  `"DISABLE_DSA_INDEXER": env_bool("TPU_DISABLE_DSA_INDEXER", ...)` ("Skip the DSA indexer
  forward call (not yet ported to torchax/TPU); falls back to dense MLA"), its launch
  script exports `TPU_DISABLE_DSA_INDEXER=1`, and `model_loader.py` says "until a
  JAX-native DSA lands". This PR provides exactly that missing kernel layer and shares
  zero files with #2324.
- **Merged main** (checked at `99a662a1`): `kernels/experimental/deepseek_v4/`
  (#2903/#2905/#2980, 2026-06-22..24) is the closest in-tree work — a DeepSeek-**V4**
  StreamIndex top-k + topk-consuming MLA kernel built around a KV *compressor*
  (`compression_ratio`), model-wired to `vllm.models.deepseek_v4` and marked
  experimental. It does not implement the DSv3.2/GLM DSA contract this PR targets
  (exact top-k, uncompressed latent cache, `GlmMoeDsa` indexer math with signed head
  gates and interleaved-rope-compatible layout). No `kernels/dsa/` path exists on main.
- #2988 (MLA TP fix), #3073 (Qwen3-Next/GDN), #3062 (SparseCore gather), #3096 (RPA-v3
  CP): checked, unrelated.

# Tests

All CPU-only (`JAX_PLATFORMS=cpu`, Pallas interpret mode); no TPU needed to review:

```bash
JAX_PLATFORMS=cpu python -m pytest tests/kernels/test_dsa_indexer_kernel.py \
    tests/kernels/test_dsa_sparse_mla.py -q
```

Result: **66 passed** (24 indexer + 42 sparse-MLA), ~95 s.

- `test_dsa_indexer_kernel.py` (24): parity vs an HF-math oracle
  (projections+rope+scoring+selection; these 4 cases auto-skip if the external reference
  file is absent — 62 pass + 4 skip in that environment); Pallas-vs-XLA to 8K ctx at
  topk 64 and 2048 (fp32+bf16, tie-aware set equality); hierarchical-vs-flat `lax.top_k`
  exactness incl. adversarial merge widths (100, 128, 8192); engineered tie handling;
  the short-context contract (kv_len ∈ {0, 1, topk−1, topk} → all valid positions
  returned, `-1` fill, `n_valid` clamp); shuffled-page-table bit-invariance; the
  mixed-dtype q/cache contract.
- `test_dsa_sparse_mla.py` (42): kernel == in-file XLA oracle (R×valid×dtype grid);
  tile-size bit-invariance (`seg_block` 128…2048); composition gates — identity top-k
  over a *shuffled* paged cache == an independent fp64 dense-MLA reference, plus random
  subsets; gather layout round-trips against the upstream mla/v1 reference cache writer
  across `(kv_packing, page_size)` layouts; inertness proofs (poisoning the rope-pad
  lanes `[576:640]` and the masked tail rows changes nothing, bit-identical);
  fully-masked rows exactly zero; padded-slot zeroing.

Adversarial baseline: the suites test new modules — on the pristine base they fail at
collection (package absent), so pass/fail is attributable to this PR's code only. The
bf16 tolerance bars were re-measured on real MXU (above), not trusted from interpret
mode.

# Risk

Additive only: five new files, no existing file touched, nothing imports the new package
yet — zero behavior change for every current model/path. Scope limits (also in the
docstrings):

- **Decode-shaped.** The scoring kernel's contract is one query per request over the
  paged history; the sparse-MLA kernel consumes a gathered top-k segment. **Prefill is
  out of scope here** — the intended prefill treatment (dense-shaped masked-XLA flash
  with a top-k mask bias, per the design's cost analysis: a per-query gather explodes at
  prefill) ships with the integration follow-up, not in these kernels.
- **Block configs are v4-validated defaults, not a tuning claim.** Tile knobs
  (`seg_block=512`, one page per grid step, 128-multiple page sizes) were chosen for
  v4's 16 MiB VMEM and validated for correctness (tile-size invariance is tested);
  multi-page tiles / SMEM residency at extreme shapes (R=64 @ 128K ctx) and the
  performance microbench are documented follow-ups.
- **bf16 selection semantics (S2).** With bf16 scoring, "0 mismatches vs an fp32
  reference" is unachievable by construction (~2^-8 relative score perturbation swaps
  near-equal, not-exactly-tied boundary scores). The kernel's bf16 gate is therefore
  band-exactness: every selection difference must lie within `|s − s_kth| ≤ 2^-8·scale`
  of the k-th score (metal, k=64 probe rows: 0 out-of-band, 0–2 in-band swaps per 64;
  CPU round-5 measurement at ctx 4K–16K, topk 2048: 1–2 non-tie in-band swaps per 2048,
  0.05–0.10%). fp32 scoring is selected-set-exact modulo exact ties. Callers wanting
  strict reproducibility should score in fp32 (the tests pin both modes).
- The Pallas scoring kernel requires `page_size % 128 == 0` when compiled
  (`interpret=False`); the XLA twin has no such constraint.

# Disclosure

Portions of this change were developed with AI assistance (Claude); every line has been
reviewed and is defended by the human submitter.
(Commits carry `Co-authored-by: Claude` trailers.)

---

## Layering (why kernels-only, and what comes next)

This PR is deliberately the bottom layer: pure kernels + oracles + contracts, importable
and testable with zero serving-stack coupling (the whole suite runs on CPU). The serving
integration — indexer weights/rope module, paged indexer-k and latent cache writers,
decode dispatch (sparse decode / dense fallback), IndexShare, and the masked-XLA sparse
prefill — exists and is pod-validated on the dev branch (`glm-5.2-v4-next`) and will be
proposed as a follow-on PR that *consumes* this package (and composes with #2324's model
enablement). Reviewing the kernel math/contracts independently of vLLM wiring is the
point of the split.

## Validation provenance (artifact paths, submitter's harness repo `glm-tpu`)

- GATE 2a (Mosaic compile, v4 metal): `docs/artifacts/kernelprobe-2a-20260708.log`
  (REJECT 1: w-tile (8,128) rule), `-try2.log` (REJECT 2: 2-D output block), `-try3.log`
  (REJECT 3: E2003 lane-dim dynamic slice), `-try4.log` (**ACCEPT**, 05:05 UTC).
- GATE 2b (MXU parity, v4 metal): `docs/artifacts/kernelprobe-2b-20260708-metal.results.txt`
  (first run: A2 FAIL — oracle-precision bug, archived), `-metal2.results.txt`
  (**ALL PASS**, 05:09 UTC: A1 2.384e-7; A2 selected-set exact, 0 non-tie mismatches;
  A3 bf16 0 out-of-band at ε=2^-8; B fp32 9.537e-7 / bf16 1.953e-3; C clean),
  `-interpret.results.txt` (wiring check), `kernelprobe-2b-audit-interpret.results.txt`
  (clean-provenance rerun on `glm-5.2-v4-next @ 886eaceb` after the oracle-precision
  audit). Honesty notes (chain-of-custody): (1) the two metal results files record the
  *main checkout's* branch in their header (with an in-file WARNING) while the kernels
  under test were imported from the `-next` worktree (the 2a tracebacks show the
  worktree path). (2) `886eaceb` was committed at 05:19 UTC — *after* both metal runs
  (2a ACCEPT 05:05, 2b metal2 05:09): the gates ran the worktree's then-uncommitted
  tree, and the audit rerun at `886eaceb` is **CPU interpret-mode** (a
  wiring/parity check, not metal), so no artifact mechanically pins the metal numbers
  to the commit bytes. Corroboration that the committed code is what passed: the 2a
  try-1 REJECT reproduces exactly the `(1, H)` w-BlockSpec that only pre-fix trees
  carry (the main checkout's `02e44b36` kernels still have it), so the ACCEPTing run
  cannot have used them. (3) The try4 log banner ("Mosaic accepted the (1,H) w-tile
  matmul LHS") is stale probe-banner text — by try4 that layout had been replaced by
  the lane-mask head-sum; the code that ran is what this PR ships. Resolution for all
  three: the checklist below makes the owner's clean re-run from the PR branch
  **required** before submission.
- Probe configs: 2a `R=2 ctx=512 page=128 H=32 D=128 topk=64`; 2b gate A
  `R=4, L∈{512,300,129,128}, k=64`; gate B `R=4, seg_valid∈{17,128,1000,2048}` over the
  full 2048-slot segment. Single chip of `db-v4-64-od` worker 0.
- CPU regression on the dev branch at freeze: 144/144 across the six DSA test files
  (this PR carries the two kernel-suite files, 66 of those).

## Deltas vs the dev branch (for the reviewer of this re-cut)

vs `glm-5.2-v4-next` (kernel files last touched `886eaceb` = `origin/glm-5.2-v4-next`;
byte-identical through the local head `cda8a707`, whose round-9 commits `215f8ddbc` +
`cda8a707` touch the *integration* layer only — note `cda8a707` is unpushed, a backup gap
for the *dev* branch, not this PR):

- `0ac6eb9e4`: **format-only** — yapf 0.43.0 (the repo pre-commit pin) applied to all
  five files; full `ast.dump` equality verified per file. The code is semantically
  identical to what the silicon gates ran.
- `5bf3e5927`: **docstring/comment-only** (code-AST-identical, docstrings-stripped
  `ast.dump` equality verified per file) — the adversarial review of this cut found the
  indexer module docstring still describing the pre-Mosaic-fix kernel ((1,H)-matmul-LHS
  w tile, [1,P] output tile, stale VMEM table) and listing the silicon-validated items
  as "Remaining for real-TPU validation"; the test docstring said the bf16/fp32 bars
  "MUST be re-measured on TPU before upstreaming" (done 2026-07-08); the "no public
  JAX/Pallas DSA kernel" sentence needed the deepseek_v4 scoping; the `mla/dsv4`
  scaffolding references needed fork-side (not in-tree) markers; the listed seg_block
  sweep was missing 768. **These accuracy fixes exist only on the PR branch — fold them
  back to `-next` (owner; the dev branch was deliberately not touched by this cut).**

isort/ruff unavailable locally — owner runs `pre-commit run --all-files` before
submitting.

## Submitter checklist (not part of the PR body)

- [ ] `pre-commit run --all-files` (isort/ruff were unavailable locally; yapf already
      applied); add DCO `Signed-off-by` when pushing.
- [ ] **Required before submission**: re-run the 2a/2b probes from the PR branch
      checkout itself for clean single-checkout provenance (scripts in
      `glm-tpu/scripts/kernel_probe/`; single chip, minutes) — closes the
      chain-of-custody notes above (metal runs predate the `886eaceb` commit; the
      audit rerun was CPU-interpret).
- [ ] Fold the `5bf3e5927` docstring-accuracy fixes back to `glm-5.2-v4-next`; push the
      dev branch's local round-9 commits (`cda8a707`) — currently unpushed (backup
      discipline).
- [ ] Decide the coordination message with #2324 (this PR supplies the kernel layer its
      `TPU_DISABLE_DSA_INDEXER` bypass is waiting on) and with the
      `kernels/experimental/deepseek_v4/` owners (adjacent DSv4 mechanism — position,
      don't compete).
- [ ] The 66-test suite runs on CPU in CI as-is; 4 oracle-parity cases skip outside the
      harness environment (by design — external HF-math reference).
- [ ] Follow-on PR (serving wiring: indexer module, cache writers, dispatch, IndexShare,
      masked-XLA sparse prefill) staged after this lands.
