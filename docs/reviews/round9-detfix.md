# Round 9 adversarial review — boot-determinism fix (glm-5.2-v4-det)

- **Target**: fork commit `215f8ddb` (branch `glm-5.2-v4-det`), reviewed in its MERGED state
  `cda8a707` on `glm-5.2-v4-next` (HEAD of `~/tpu-inference` at review time; worktree clean;
  `git diff 886eaceb cda8a707 --name-only` = exactly the two claimed files).
- **Change under review**: (1) canonical ascending-position stable sort of the top-k selection
  before `gather_kv_segment` in `_sparse_decode_branch`
  (`tpu_inference/layers/vllm/custom_ops/mla_attention.py:1199-1204`); (2) V-side zeroing of
  non-live rows before the PV einsum in `glm_dsa_masked_prefill_attention`
  (`mla_attention.py:343`); (3) new CPU gate file
  `tests/layers/vllm/test_glm_dsa_boot_determinism.py`.
- **Method**: read-only adversarial review; all execution CPU-only (`JAX_PLATFORMS=cpu` exported
  before every python; Pallas `interpret=True`). The live pod run was not touched.
- **Verdict**: **the fix is correct as merged.** Every attacked claim survived. Two non-blocking
  findings: a stale docstring (F1) and an unmeasured per-layer sort cost that is linear in the
  decode-token bucket and should be profiled on metal (F2). One residual-risk note that is
  inherent to DSA, not a defect of this change (R1).

---

## 1. Test evidence on the merged tree (claim e)

All run from `/home/gianl/tpu-inference` @ `cda8a707` with `/home/gianl/vllm-env/bin/python`,
`JAX_PLATFORMS=cpu`:

| suite | result |
|---|---|
| `tests/layers/vllm/test_glm_dsa_boot_determinism.py` (the 4 new gates) | **4 passed** (73.5 s) |
| `test_glm_dsa_pallas_decode.py` + `test_glm_dsa_sparse_prefill.py` | **30 passed** (234 s) |
| `test_glm_dsa_indexer.py` + `test_glm_dsa_mtp_index_share.py` + `tests/kernels/test_dsa_indexer_kernel.py` + `tests/kernels/test_dsa_sparse_mla.py` | **114 passed** (264 s) |

Matches the commit message's claimed counts (4 / 30 / 114) exactly. Additionally two review
probes of mine (scratchpad `probe_sort_edges.py`, `probe_amplifier.py`, transcripts below) both
passed.

---

## 2. Claim (a): the sort preserves the exact selected set and the §3.1 tail-suffix contract

The fix is three lines: `key = where(idx >= 0, idx, INT32_MAX)`, `order = argsort(key, axis=1,
stable=True)`, `seg_indices = take_along_axis(idx, order, axis=1)`.

**Refutation attempts, all failed:**

1. **Multiset preservation** — `take_along_axis(idx, argsort(...))` applies a permutation of each
   row; a permutation preserves the row's multiset unconditionally, for ANY input (including
   contract-violating input). No construction can make it drop or invent an index.
2. **All-`-1` rows / `n_valid == 0`** (padded batch slots): every key is `INT32_MAX`; the stable
   argsort is the identity; the row passes through unchanged, `seg_valid` stays 0, and
   `_finalize` zeroes the row as before. Verified (probe E1/E2, exact identity asserted).
3. **`topk > S` (short ctx)**: `hierarchical_topk` fills slots `>= n_valid = min(kv_len, topk, S)`
   with `-1` positionally (`indexer_kernel.py:477-479`); post-sort the valid prefix is ascending
   and the `-1` tail is intact. Verified (probe E3 plus 200 randomized rows, E6).
4. **Duplicate positions** — proven impossible from the producer: per-merge-block `lax.top_k`
   returns distinct in-block indices, block rebasing (`+ block_id * merge_block`,
   `indexer_kernel.py:464-465`) keeps candidate index sets disjoint across blocks, and the final
   `top_k` selects distinct candidate slots; `take_along_axis` over distinct slots of pairwise
   distinct values cannot repeat. Randomized search (60 trials, heavy score ties, `kv_len` from 0
   to S, `merge_block < topk` and `> topk`) found: no duplicates, `n_valid` exact, prefix causal,
   `-1` strictly a tail suffix (probe E9). Round-5 review reached the same conclusion
   independently ("Duplicates impossible (disjoint per-block candidates)"). Even hypothetically,
   a duplicate survives the sort as an adjacent pair — the multiset (and hence the pre-fix
   behavior, double-counting in the softmax) is unchanged, so the sort introduces no NEW failure
   mode (probe E4).
5. **Sort-key collision**: a valid index equal to `INT32_MAX` would misclassify as a pad, but
   positions are bounded by `max_blocks * page_size` (< 2^31 by ~5 orders of magnitude in any
   real config). Unreachable.
6. **Pad-condition mismatch**: the sort keys on `idx >= 0`, and `gather_kv_segment` counts
   `seg_valid` from the same predicate (`sparse_mla_kernel.py:393-394`). Any negative value (not
   just `-1`) is treated identically by both. Consistent by construction; `seg_valid` is a
   permutation-invariant count, asserted equal pre/post-sort on garbage-shuffled inputs
   (probe E7).
7. **Gathered content equivalence**: rows gathered from the sorted indices are exactly the
   position-sorted permutation of the rows gathered from the unsorted indices (probe E7,
   elementwise `array_equal` per row).

**Positive assurance**: the sort is strictly SAFER than the pre-fix code against one upstream
pathology — interleaved `-1` entries (only producible by NaN indexer scores breaking `top_k`'s
suffix guarantee, round-5 F4b). Pre-fix, interleaving desynchronized the count-based `seg_valid`
from the prefix mask (wrong output); post-fix the sort moves all pads to the tail and the
count/prefix agreement is restored mechanically. See F3 for the flip side.

---

## 3. Claim (b): "exact math / permutation invariance" and the kernel's order contract

**The correct reading of "exact"** (and the commit is careful about this): the new outputs are
NOT bit-equal to pre-fix builds — reordering the segment reorders the flash accumulation, which
is the entire point. What is claimed and what holds: the post-fix output is an exactly-valid
fp evaluation of attention over the same selected set, and it is a function of the SET only.

**Kernel order-dependence audit** (`sparse_mla_kernel.py`, full read):

- `_online_update` (:79-92): running max + rescale; no early exit, no monotonicity assumption, no
  branch on score order. Any segment order is a valid flash schedule.
- Masking (:155-161): the single mask is `j_abs < n_valid` — a POSITIONAL prefix property of the
  segment, exactly what the sort preserves (pads keyed `INT32_MAX` land at the tail). No other
  mask exists.
- `_finalize` (:95-111): the `_MASK_VALUE` (-0.7*f32max) sink fold contributes
  `exp(_MASK_VALUE - m) == 0.0` for any live row regardless of accumulation order, and the
  fully-masked-row select (`m > _MASK_VALUE`) is order-free. The `-1e37`-class finalize has no
  order assumption.
- **The round-5 F2 "descending-order contract note"**: I re-read both the round-5 review text and
  the current `gather_kv_segment` docstring. The note says, verbatim: "the gather does not
  require sorted (`indices_are_sorted=False`, attention permutation-invariant) — contract
  compatible". Descending-score order was a DESCRIPTION of the producer, never a requirement of
  the gather or kernel; the only actual contract is the `-1` tail suffix, which the sort
  preserves (section 2). **No documented kernel contract is violated.** However the docstring
  sentence "`topk_indices` come in DESCENDING-SCORE order (not position-sorted)"
  (`sparse_mla_kernel.py:357-359`) is now factually stale for the production caller — finding F1.
  (`indices_are_sorted=False` in the `jnp.take` remains correct: rows are now per-request
  ascending, but the flattened cross-request row ids are not globally sorted and pads clamp to
  row 0.)
- **Golden/bit-compat blast radius**: the dense-parity decode gates compare against an fp64 numpy
  reference at `atol=2e-5` (`test_glm_dsa_pallas_decode.py:375,449`), not against stored bit
  goldens of the old order; the selection-equality assertions are on the STASH (unsorted).
  Nothing pins the old summation order. All 148 tests green post-change confirms.

**Load-bearing demonstration** (probe_amplifier.py): same selected set (K=64 of ctx=100, 2 reqs),
two orders, NO sort → `max|diff| = 8.3e-7` through gather + kernel (interpret); with the
canonical sort → bit-identical. The commit's ~1.1e-8 order-noise class is real and the sort
removes it.

---

## 4. Claim (c): IndexShare stash stays score-ordered while the gather re-sorts

**Caller census**: `grep -rn gather_kv_segment` over `tpu_inference/` — exactly ONE production
callsite, `mla_attention.py:1203`, inside `_sparse_decode_branch`, AFTER the sort (all other hits
are docs/tests). Because the sort lives inside the one branch that feeds the gather, every path
is covered by construction:

- **Full layers**: fresh score-ordered `hierarchical_topk` output → sorted at the gather.
- **Shared layers** (same forward): `fetch_shared_topk_indices()` returns the SAME stash array →
  the same pure-function sort → identical `seg_indices` for every layer of the share group.
  Consistency across shared layers is exact (same input, same deterministic argsort), and
  determinism only requires per-layer set-dependence, which now holds.
- **MTP-G4 preseeded reuse (draft steps >= 1)**: traced end to end —
  `eagle3.py:703-706` (`step0_aux[last_token_indices]`, a row-select that leaves per-row order
  untouched) → `draft_step_fun_impl` seeds the context (`vllm_model_wrapper.py:821`) →
  `topk_context_preseeded()` demotes the draft's full layer to fetch-only → `indices =
  shared_indices` (`mla_attention.py:1146`) → the preseeded `lax.cond(sparse_step,
  _sparse_decode_branch, _dense_fallback_branch)` (:1280) enters the SAME branch with the SAME
  sort. **The draft sorts too; there is no order divergence between target and draft**, and more
  importantly there is no path that feeds score-ordered indices into the gather.
- **Set-based consumers of the (still score-ordered) stash**: sparse-prefill scatter mask
  (`mla_attention.py:300-303`, boolean scatter — order-free), the preseeded gate
  `jnp.any(shared_indices != INVALID_INDEX)` (:1279, order-free), `sparse_bias_from_indices`
  (xla_ref mode, order-free), and the MTP emit/reseed carriage (row-select only). Verified each.

One deliberate coupling worth knowing: the stash's score order is PINNED BY TESTS
(`test_glm_dsa_mtp_index_share.py:372` and `test_glm_dsa_pallas_decode.py:443` assert the stash
equals the reference indexer's score-ordered output). So the author's choice to sort at the
gather rather than at the producer is not just minimal-blast-radius, it is what the existing test
contracts require. A future perf hoist of the sort into `_score_and_select` (see F2) must update
those assertions deliberately.

---

## 5. Claim (d): the masked-prefill PV V-side zeroing cannot change live outputs

**`p == 0.0` exactly on every non-live column — proof** (`mla_attention.py:324-332`):
`s = where(live, s, -inf)` is applied AFTER the QK einsum, so garbage (even NaN) in a masked
column's score is replaced, not propagated. For a non-dead row, `m_new` is finite (some live
score), so `p = exp(-inf - m_new) = exp(-inf) = +0.0` exactly (IEEE). For dead rows
(`m_new == -inf`) `p` is forced to `0.0` by the `where(dead, ...)`. The only escape would be
`m_new == +inf` from an f32 QK overflow of LIVE (written, finite, bf16-bounded) data — not
reachable, and pre-existing behavior regardless.

**Therefore** the pre-fix PV contribution of a masked column was `0.0 * v`. With finite `v` that
is `±0.0`; the fix changes it to exactly `+0.0`. Adding `±0.0` terms cannot change any nonzero
partial sum, and `+0.0 + (-0.0) = +0.0` under round-to-nearest — the only conceivable
divergence is the SIGN of an exactly-zero accumulator lane, which is observationally inert
(compares equal, and any downstream product/sum with nonzero data is unaffected). Under the
production `jnp.zeros` allocation the masked `v` was already `+0.0`, so the change is
bit-identical — which is precisely what `test_masked_prefill_pv_ignores_page_tail_garbage`
asserts (`np.array_equal` of the zeros-tail baseline against nan/inf/1e30 tails), and it passes.

**Was a matching V-fix needed in the decode kernel?** No — checked: the decode segment's pad
slots gather the request's OWN row 0 (`safe_t = 0` → `block_tables[r, 0]`), a written row for
any live sequence; the kernel-side `tile_pad` is `jnp.pad` zeros; and a `seg_valid == 0` row is
zeroed in `_finalize` through a NaN-killing `where`. The garbage-cache gates (finite AND NaN/Inf)
passing on the decode path confirm all three.

---

## 6. Claim (f): cost of the per-layer `[T, topk]` int32 sort

**What actually gets sorted** (read from the code, not the commit message): `indices` is
`[num_tokens, topk] = [T, 2048]` int32 where `T` is the PADDED DECODE TOKEN BUCKET of the step
(1 token per request on a pure-decode step, bucket-padded — e.g. T=8..256 depending on serving
batch; NOT 2048 rows per token). The sort executes once per layer per decode step, inside the
decode `lax.cond` branch only (prefill/mixed steps never pay it). GLM-5.2: **78 layers** run the
sparse decode branch per step (21 full + 57 shared; design doc §5), plus the MTP layer per draft
step on the reuse path.

**Estimate** (v4, argsort = variadic bitonic sort of (key, iota), ~66 compare stages for width
2048, ~5 elementwise ops per compare-exchange, replicated on every chip — the array is not
sharded): per step ≈ `78 layers x T x 2048 x 66 x 5` ≈ `53M x T` element-ops. At T=16 that is
~0.9G ops ≈ **0.5–1.7 ms/step** (0.5–1.9e12 elementwise ops/s sustained VPU range); at T=64,
**2–7 ms/step**. Against a ~20 ms decode-step floor (32B active bf16 params / 4 chips at
~0.8 TB/s) that is ~2–8% at T=16 but potentially **10–30% at T=64** if the pessimistic end
holds. XLA:TPU sort lowerings have historically been at the slow end, so this is flagged rather
than waved off — finding F2 with concrete mitigations (one of them 2x for free). The extra cond
arena is ~3 x [T, 2048] i32 — negligible next to the pre-existing 20 GiB seg_kv NOTE at
`mla_attention.py:1171-1179`.

---

## 7. Findings

**F1 (docs, minor — stale contract note).** `sparse_mla_kernel.py:357-359`
(`gather_kv_segment` docstring): "`topk_indices` come in DESCENDING-SCORE order (not
position-sorted) — attention is permutation invariant, so no sort is required or assumed" is now
false in its descriptive half: the sole production caller passes ascending-position order
(canonical, round 9). The normative half (tail-suffix `-1` contract, no order assumed) is intact
and satisfied. Update the sentence; also worth recording that the round-5 wish ("a future kernel
exploiting index monotonicity would silently break") is now inverted — the valid prefix IS
position-monotonic, which a future gather/kernel may legitimately exploit.

**F2 (perf, medium — measure on metal before scaling batch).** 78 argsorts of `[T, 2048]` int32
per decode step, linear in T (section 6; 0.5–7 ms/step across plausible T — up to double-digit
percent of a decode step at large batch if the pessimistic VPU number holds). Not gating for the
current 4-item sequential benchmark protocol (T small). Mitigations, cheapest first:
(i) replace argsort+take_along_axis with the single-operand form
`sorted_key = jnp.sort(where(idx >= 0, idx, INT32_MAX)); seg_indices = where(sorted_key ==
INT32_MAX, -1, sorted_key)` — exactly equivalent (verified on 300 randomized rows, including
all-pad and single-key rows) and roughly halves the sort payload by dropping the iota carry;
(ii) bound the branch's rows to live requests (the existing arena NOTE already wants this);
(iii) hoist the sort into `_score_and_select` and stash position-ordered indices — cuts 78 sorts
to 21+1, but the stash's score order is pinned by `test_glm_dsa_mtp_index_share.py:372` and
`test_glm_dsa_pallas_decode.py:443` and by the GPU-parity framing of the MTP emit, so (iii) is a
deliberate contract change, not a drop-in.

**F3 (robustness note, non-blocking).** The sort silently REPAIRS interleaved `-1` rows — an
upstream contract violation only producible by NaN indexer scores (round-5 F4b: no NaN guard
exists upstream, unchanged by this commit). Strictly safer output-wise, but it converts a
formerly loud corruption into a quiet one: the `-1` slots still represent selections lost to
NaN. If a NaN guard is ever added, it belongs on the scores, not here.

**R1 (residual risk, inherent — not a defect of this change).** The fix makes decode output a
function of the selected SET. At `ctx <= topk` the set is always the full causal history, so the
observed GSM8K-class divergence (runs 53 vs 55, ctx << 2048) is fully closed against ulp-level
score perturbations. At `ctx > 2048`, an across-boot ulp perturbation can still flip the k-th
top-k boundary and change the SET itself — no gather-side ordering can fix that; it is DSA's
semantics. The commit's on-metal follow-up (same-boot replicate probe, persistent-compile-cache
pinning, executable-fingerprint diff) is the right discriminator for the remaining executable-
identity question and should still be run.

---

## 8. Positive assurance summary

- The merged tree is clean and contains exactly the reviewed change (two files) on top of
  `886eaceb`.
- The sort preserves the selected multiset, the `-1` tail-suffix contract, `seg_valid`, and the
  gathered row content (as a permutation) for every edge case constructed: all-pad rows,
  `n_valid = 0`, `topk > S`, duplicates (impossible from the producer; harmless if forced),
  interleaved pads, randomized sets (probes E1-E9, all passed).
- The 2c kernel has no dependence on segment order beyond the tail-suffix prefix mask; no
  documented contract (round-5 F2 note included) is violated; `_finalize` and `_online_update`
  audited order-free.
- Every production path into `gather_kv_segment` — full, shared, and MTP preseeded draft —
  passes through the sort; the stash and all its other consumers are order-insensitive.
- The PV V-zeroing provably cannot alter any live output (exact `p == 0` on the complement of
  `live`), is bit-identical under production zeros allocation, and closes a real 0*Inf hole for
  garbage caches; the decode kernel needed no analogous fix (pads clamp to written rows).
- Full CPU regression on the merged tree: **148/148 passed** (4 boot-determinism + 30 decode/
  prefill + 114 indexer/MTP/kernel), matching the commit's claim, plus 2 independent review
  probes; the order-amplifier is demonstrated real pre-fix (8.3e-7) and bit-collapsed post-fix.

Reviewed by: round-9 adversarial reviewer (CPU-only, read-only). Probes:
`/tmp/claude-2001/-home-gianl/8712a40e-7bbe-436d-b0ad-d676cbd502d5/scratchpad/probe_sort_edges.py`,
`.../probe_amplifier.py`.
