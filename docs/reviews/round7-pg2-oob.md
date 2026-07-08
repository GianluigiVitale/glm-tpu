# Round 7 adversarial review — pg2: pack_new_kv OOB VMEM read fix (MERGED)

- **Target**: fork commit `63427f86` ("[MLA v2] Fix OOB VMEM read in pack_new_kv at bkv-block-boundary decode"), branch `glm-5.2-v4-pg2`, **already merged into the live pod branch** `glm-5.2-v4` as `02e44b36`. Reviewed the merged mainline state in a fresh detached worktree off `origin/glm-5.2-v4` (== local `glm-5.2-v4`, same SHA).
- **Scope**: `tpu_inference/kernels/mla/v2/kv_utils.py` (+45/-8: `read_next` clamp in `pack_new_kv` and `pack_new_kv_reference`), `tests/kernels/mla_v2_pack_new_kv_oob_test.py` (new, 69 cases); the surrounding merge-loop dataflow in `kernels/mla/v2/kernel.py` audited read-only for sibling OOBs.
- **Method**: read-only source audit; independent re-derivation of the merge-loop index algebra; a 5,540-case CPU differential sweep (real `pack_new_kv` under `pltpu.InterpretParams`, fixed vs pre-fix `87abdf53` parent vs numpy oracle) in a scratchpad venv with the repo-pinned `jax==0.10.1`; re-ran the shipped 69-case suite post-fix AND against the pre-fix code; traced-jaxpr diff for the cost question. All under `JAX_PLATFORMS=cpu`; the TPU was never touched.
- **Verdict**: **the fix is correct and the "never-consumed" claim is proven** — I could not construct any case, across kv_packing ∈ {1, 2, 4, 32}, bkv_sz ∈ {64, 256, 512, 1024, 2048}, single- and multi-token packs, both merge-branch variants and the reference variant, where the clamp changes a consumed value; the pre-fix OOB fires *exactly* where the analytic predicate says and always exactly one row past the buffer. Findings below are claim-accuracy issues in the commit's root-cause narrative and verification story (F1, F2), one latent sibling OOB in the (gated-off) transposed path (F3), a live-branch test-coverage gap (F4), and doc nits.

---

## The re-derived proof (attack 1)

Setup (`kv_utils.py:71-136`, buffer shape `[S+2, P, dim]` with `S = bkv_sz/P`, `P = kv_packing = bkvc_vmem_ref.shape[1]`):

- `t = offset % bkv_sz`, `ρ = t % P`, `u = update_sz`, `npo = (q_end - kv_len + offset) % P`.
- Caller invariants (from `_fetch_bkv`, kernel.py:929-939, 1108-1109 and the `pl.when(update_sz > 0)` guard at kernel.py:1272): `u ≥ 1`, `t + u ≤ bkv_sz`, and `new_kv_len_start ≥ q_start ≥ 0`.
- Loop: `iters = cdiv(ρ + u, P)`; source cursor start `n0 = cdiv(t, P) + δ` where `δ = (npo − ρ) // P ∈ {−1, 0}`, i.e. **δ = 0 iff npo ≥ ρ** (see F5: the in-file comment states the opposite inequality and is wrong).
- Unrolling the `fori_loop`: iteration `i` **consumes** `buf[n0+i]` (curr) and `buf[n0+i+1]` (next) and **prefetches** `buf[n0+i+2]`. So the set of reads is `{n0, …, n0+iters+1}`; the *last* read `buf[n0+iters+1]` is written only into the loop-carried tuple, which `lax.fori_loop`'s discarded return value is the sole consumer of (kv_utils.py:252-264) — dead by construction.
- Bounds: with `t+u ≤ SP`, algebra gives `iters ≤ S − (t−ρ)/P`, hence **max consumed index `n0 + iters ≤ S + 1 + δ ≤ S+1 = rows−1`** (and ≤ S when ρ = 0, since then δ = 0 and cdiv(t,P) = t/P). Every consumed read is in bounds; the clamp `min(idx, rows−1)` is an identity on all of them.
- The dead prefetch hits `rows = S+2` **iff `n0 + iters = S+1`, i.e. iff ρ ≠ 0 ∧ npo ≥ ρ ∧ t + u > bkv_sz − P** (the update *ends inside the last packed word* — not only exactly at the boundary; multi-token prefill tails that stop 1..P−1 tokens short of the boundary trigger it too). `curr`'s initial read `n0` is never negative (δ = −1 requires ρ ≥ 1 ⇒ cdiv(t,P) ≥ 1).
- Consumption-side cross-check: in both merge branches (`u32` shift path and `num_sublanes>1` roll path) the final iteration's *consumed* next (`buf[n0+iters] = buf[S+1]`) can reach the output only for source sub-positions that the fetch DMA actually staged — I verified `_fetch_bkv` fills rows exactly up to `cdiv(t,P) + cdiv(npo+u,P) − 1 ≤ S+1` (kernel.py:1063-1086), which is precisely why the buffer has the `+2` rows. The clamp does not touch that row's value.

So: OOB predicate `ρ≠0 ∧ npo≥ρ ∧ t+u > bkv_sz−P`, overrun always exactly one row, value always dead. The fix (clamping the 4 prefetch sites in `pack_new_kv`, kv_utils.py:135-136, 241-242, and the same 4 in `pack_new_kv_reference`, 306-307, 380-381) is semantics-preserving. **Empirically confirmed exactly** (below).

## Empirical verification (attacks 2 and 4)

All in `jax==0.10.1` (repo pin; the system `jax 0.6.2` cannot even import `kv_utils` — `jax.dtypes.itemsize_bits` doesn't exist there), scratchpad venv, vllm stubbed.

1. **Shipped suite, post-fix**: 69/69 pass (30.8 s).
2. **Shipped suite, pre-fix** (`kv_utils.pack_new_kv` monkeypatched to the `87abdf53` parent): **13 failed / 56 passed — the exact 13 cases my predicate predicts**, per-parametrization: `no_oob_bf16[2,4,10,16]` (4), `decode_kv_len_sweep_bf16[10-512]` (1, and *only* that one of 42), `fp8[4-509],[4-510],[4-511],[6-509],[7-509],[7-510]` (6 — note `[6-511]` and `[5-*]` pass, the asymmetry that pins the true δ sign, F5), `multi_token[3],[5]` (2 — u=2,8 have ρ=0 and pass). Every failure is the interpreter's `IndexError: Out-of-bounds read … [(258, …)] but buffer has shape (258, 2, 128)` — the suite is genuinely regression-sensitive and the commit's "fails with IndexError on the unfixed code" claim holds.
3. **My 5,540-case differential sweep** (`scratchpad/adv_sweep.py`): exhaustive t∈[0,64) × u-lattice × all npo at bkv=64 for bf16(P=2)/fp8(P=4); production-scale spot grids at bkv ∈ {256,512,1024,2048} incl. multi-block offsets, decode T=1, u=P straddles, and full-tail prefills (t=1, u=bkv−1 — a *multi-word* OOB geometry the shipped suite has no analogue of); **kv_packing=32** (bf16 sublanes=16, fp8 sublanes=8 — the `pltpu.roll` branch, see F4) at bkv 512/1024; fp32 P=1; `pack_new_kv_reference` at P∈{2,4,32}. Results, 1,209 predicted-OOB / 4,331 clean, **zero deviations**:
   - fixed variant: never raises, **bit-exact vs the oracle over the WHOLE buffer** (incl. the scratch rows S, S+1 — pack provably never writes them) in all 5,540 cases;
   - pre-fix variant: raises **iff** the predicate, and the parsed OOB row always equals `rows` (never rows+1 or more);
   - **byte-identity**: on all 4,331 non-OOB cases pre-fix output bytes == fixed output bytes. Together with (1)+(2) this verifies the commit's "byte-identical outputs on non-OOB cases" claim far beyond its kv 505/511/513 spot checks.
4. **Fatal-geometry probe** (`scratchpad/probe.py`): offset=511/u=1/kv_len=512, q_end even → pre-fix reads row 258 of 258 (raises), fixed clean; q_end odd → pre-fix clean. Matches the commit's trigger exactly (at P=2).

## Findings (most severe first)

### F1. MEDIUM (root-cause narrative vs serving config): the commit's trigger arithmetic is kv_packing=2-specific, but the serving default allocates the MLA cache with kv_packing=32

- **Files**: `tpu_inference/envs.py:404-405` (`MLA_KV_PACKING_SIZE` default **32**), `runner/kv_cache.py:69-77` (passes it to `mla.get_kv_cache_shape` for every MLA cache), kernel derives `P` from the allocated shape (kernel.py:361-368).
- **Attack**: the commit narrates the pod halt as "bf16: even q_end → odd seq slot in a pure-decode batch" and the repro "OOB at kv_len==512 for every odd slot" — that is the P=2 world (buffer `[258, 2, ·]`). Under the *default* env the pod buffer is `[18, 32, 640]` and the trigger becomes `npo ≥ ρ` with ρ = 511 % 32 = **31**, i.e. `q_end ≡ 0 (mod 32)`. For waveB's described composition (crossing decode request at slot 9 ⇒ q_end = 10, npo = 9 < 31) **the pre-fix code does not OOB at P=32 at that request** — I verified this class of case directly in the sweep. Since the fix demonstrably changed pod behavior (pre-fix 100% fatal at the crossing config, post-fix GSM8K n=32 clean — RESEARCH_LOG 2026-07-07 15:05), an OOB *was* firing; that is arithmetically consistent only if (a) the pod ran with `MLA_KV_PACKING_SIZE=2` (no override found in this repo, `~/glm-tpu/bench/`, or configs — but pod env is not visible from this box), or (b) the actual halting geometry differed from the described decode-slot-parity story (at P=32 with TOKEN_BUCKET=32 chunked prefill, `q_end ≡ 0 (mod 32)` is common — a prefill chunk ending in a page's last 32-token word triggers it).
- **Not affected**: the fix itself — the clamp is P-generic and my sweep proves it at P=32 (both that the pre-fix roll branch OOBs on the same predicate, row 18 of 18, and that the fixed branch is oracle-bit-exact).
- **Action**: log the allocated MLA cache shape (or `MLA_KV_PACKING_SIZE`) at pod boot once and pin the narrative; if the pod really is P=32, re-derive which request actually crossed (flight-recorder kv_lens + cu_q_lens make this a 5-minute check) so the "root cause pinned" statement is precise, and add P=32 cases to the shipped suite (F4).

### F2. LOW-MEDIUM (verification claim): the commit's "full 4-instance mla_ragged_paged_attention … under interpret" run is not reproducible with any jax available under the repo pin

- **Claim attacked**: commit message — full-kernel CPU interpret repro of the waveB composition, pre-fix OOB / post-fix clean + numpy-MLA match, and full-kernel pre/post byte-identity at kv 505/511/513.
- **Evidence**: I built exactly that harness (`scratchpad/full_kernel_repro.py`: 11 decode reqs, crossing req at slot 9, bf16, page 512, blocks (1,1), decode_batch_size=4, `pl.pallas_call` forced to `InterpretParams`). Under the repo-pinned `jax==0.10.1` **and** the newest installable `0.10.2`, the interpreter dies in *both* pre- and post-fix runs before reaching `pack_new_kv`: `AttributeError: 'BitcastTransform' object has no attribute 'indices'` (`jax/_src/pallas/mosaic/interpret/utils.py:356`) — the kernel's `bkvc_x2_ref.bitcast(jnp.uint32)` accesses (kernel.py:1693-1706) are unsupported by the interpret machinery (same wall round-5 hit; `to_range` is used for the interpreter's actual data movement, so it cannot be safely stubbed). Either the fix author ran a locally patched interpreter / unreleased jax (harness not committed), or the claim describes something narrower than stated.
- **Consequence**: the *committed* artifacts substantiate only the pack-level verification (which is excellent — 69 cases + regression-sensitivity confirmed here). The full-kernel byte-identity and numpy-MLA-match claims are unverifiable from the repo; treat them as informal. The pod GSM8K result is the actual full-stack evidence.
- **Action**: commit the full-kernel repro harness (with whatever interpreter patch it needed), or reword the commit-message verification note in the merge/PR description.

### F3. LOW (latent sibling OOB, gated off): the transposed-cache fetch can DMA the new-KV staging block past the bkv VMEM buffer

- **File**: kernel.py:846-873 (`_fetch_transposed_bkv` staging copy), buffer `[…, lkv_dim, bkv_sz + 256]` (kernel.py:317); same range re-read by `copy_partial` (kernel.py:1179-1232).
- **Analysis**: dst lane range is `[align(c,128) + 128, align(c,128) + 128 + align_up(a+u,128))` with `c = bkv_sz_frm_cache`, `a = new_kv_len_start % 128`. Exhaustive numeric check (bkv=512): worst case `c=1, a=2, u=511` ⇒ dst end **896 > 768** — a 128-lane DMA write overrun of the scratch buffer (and matching staged-read overrun), same `disable_bounds_checks=True` consequences as the fixed bug. Reachable only with `MLA_TRANSPOSE_KV_CACHE=1` (default False, envs.py:411-412; off on the pod) **and** chunked prefill with an unaligned cached remainder in the block. This is upstream #2864 code, untouched by the port and by this fix — the fix's "covers all pipelined reads" claim is about the non-transposed pack path and remains true.
- **Action**: none for pg2. Track as an upstream issue if transposed mode is ever enabled (round-5 F4c already declared it incompatible with the DSA gather; consider a hard `NotImplementedError` for GLM configs).

### F4. LOW (test gap on the possibly-live branch): the shipped 69-case suite only exercises kv_packing == dtype-packing (the `num_sublanes==1` u32 branch)

- All 69 cases use bf16/P=2 or fp8/P=4. If the pod allocates at the serving default `MLA_KV_PACKING_SIZE=32` (F1), production decode misalignment goes through the `num_sublanes>1` `pltpu.roll` branch (kv_utils.py:172-205) — flagged as having *zero* test coverage since round-5. My sweep now covers it pack-level on CPU (bf16 P=32 sublanes=16, fp8 P=32 sublanes=8: pre-fix OOBs per the same predicate at row 18/18; fixed bit-exact vs oracle incl. multi-word full-tail packs), but nothing in the repo's tests does.
- **Action**: add a `kv_packing=32` parametrization to `mla_v2_pack_new_kv_oob_test.py` (the harness generalizes trivially — decouple `kv_packing` from `get_dtype_packing`, cf. `scratchpad/adv_sweep.py`).

### F5. LOW (doc/comment): the `(-offset_diff) // kv_packing` comment has the inequality backwards (pre-existing upstream, now repeated in the fix's commit message)

- kv_utils.py:107-109 and 289-291 say "0 if new_kv_packing_offset <= kv_packing_offset, −1 if >". Truth (floor division): **0 iff npo ≥ ρ, −1 iff npo < ρ**. Discriminating evidence: fp8 offset=511 (ρ=3), pre-fix — q_end=4 (npo=3) OOBs, q_end=6 (npo=1) does **not** (suite run 2 above; the stale direction predicts the opposite for npo=1). The wrong comment is harmless to the code but actively misleads exactly the kind of alignment reasoning this bug required; the OOB-guard comment added by the fix (kv_utils.py:113-127) is correct because it references the *expression* value, not the inequality.
- **Action**: one-line comment fix at both sites.

### F6. INFO (nits, no action strictly required)

- The suite compares only `[:valid_rows]` (test:148-160); pack provably never writes rows S..S+1 and my sweep's full-buffer compare passes, so the assert could be tightened for free.
- `pack_new_kv_reference` has no callers or tests anywhere in the repo — dead code; clamping it anyway was the right consistency call (it shares the OOB pattern; sweep-verified too).
- `update_sz == 0` cannot reach the loop (kernel.py:1272 `pl.when` guard), and even if it did, the pre-loop reads `n0, n0+1 ≤ S+1` are in bounds.

## Attacks attempted that FAILED (positive assurance)

- **Consumed-value path for the clamped read**: unrolled both merge branches (u32 shift: `curr>>32−s | next<<s` with the `shift_bits==0` select; roll: `iota<roll` select of `shift_roll(curr/next)` with the `roll_amount==0` select) hunting for any dest lane that takes the *prefetched* (not the consumed) next word — none exists; the prefetch value's only sink is the discarded loop carry. The 4,331-case byte-identity is the machine check of the same statement.
- **Multi-token / boundary-straddle geometries**: full-tail prefill from odd offset (t=1, u=bkv−1: `iters`=S, trailing read S+2 — OOBs pre-fix, clamp exact), u=P and u=P+1 straddles, updates ending 1..P−1 short of the boundary (OOB *without* touching the boundary — a geometry the commit text doesn't mention but the clamp covers), multi-block offsets (block-1 decode, the pod's kv 513..520 walk). All bit-exact.
- **kv_packing edges**: P=1 (fp32; ρ≡0 ⇒ provably no OOB — confirmed, and pack still oracle-exact), P=32 both dtypes (F4).
- **Sibling unclamped accesses** (attack 3): audited every dynamic index in kv_utils and kernel.py. In-bounds by proof: `curr` at `n0 ∈ [0, S]`; dest merge reads/stores at `kv_packing_idx ≤ S−1`; `_fetch_bkv` staging fills to at most row S+1 (exact fit); `page_idx` already clamped (kernel.py:1001-1002); SMEM id arrays use static offsets (`+2/+4` within allocated extents); v1 kernel has no pipelined next-row pattern. The only genuine sibling problem found is the gated-off transposed path (F3).
- **Suite-oracle fidelity**: independently re-implemented the DMA staging layout + oracle (bkv/kp-generic) and cross-validated the shipped `_build_case` at its geometries — agrees; the shipped suite's oracle is faithful to `_fetch_bkv`'s word-aligned append semantics.

## Cost (attack 5)

Traced-jaxpr diff of the pack-level program, fixed vs pre-fix (`scratchpad/jaxpr_diff.py`): **+6 scalar `min` ops, −0 ops** (163 vs 157 eqns), no new buffers, no shape changes; scratch shapes in kernel.py are untouched by the commit ⇒ **VMEM identical by construction**. The mins are int32 scalar-unit ops against a Python-constant `rows−1`, two of them inside the merge loop (loop trip ≤ S). Latency impact is structurally negligible; final confirmation of compiled Mosaic scheduling is TPU-only (below).

## What only the TPU / pod can verify

1. **The allocated cache shape / `MLA_KV_PACKING_SIZE` in the serving env** (settles F1: P=2 vs P=32 narrative; if P=32, identify the true halting request from the flight-recorder step).
2. Compiled-kernel equivalence beyond the clamp: Mosaic scheduling/VMEM report pre/post fix (expect identical VMEM, ±ε latency; nothing in the jaxpr suggests otherwise).
3. The commit's hardware-halt *mechanism* claims (garbage-read when the row aliases the next buffer, halt only for `bkv_sem_idx==1` + last batch item / last allocation) — plausible allocation-layout reasoning, not checkable off-TPU.
4. Full-kernel pre/post byte-identity on real compositions (F2) — a one-off on-TPU A/B (pre-fix binary is one commit back) if anyone wants the claim restored to "verified".
5. F3's transposed-path overrun (needs `MLA_TRANSPOSE_KV_CACHE=1` on TPU to demonstrate; CPU interpret can't run that path either — BitcastTransform).

## Repro artifacts (scratchpad, session `8712a40e…`)

- `wt-pg2-review/` — detached worktree @ `02e44b36`; `venv/` — python3.12 + `jax==0.10.1` + pytest (+ stubs/vllm shim).
- `adv_sweep.py` (5,540-case differential sweep — ALL OK, 635 s), `probe.py` (fatal-geometry probe + interpreter-OOB validation), `run_suite_prefix.py` (pre-fix suite run: 13F/56P exact-match to predicate), `jaxpr_diff.py` (+6 `min`), `full_kernel_repro.py` (F2 evidence), `kv_utils_prefix.py` (parent-commit module).
